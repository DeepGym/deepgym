"""Daytona's desktop (xfce + Chromium), run as a live session's agent user."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable

from deepgym.exceptions import ActionError, SandboxError
from deepgym.models import Action
from deepgym.sandboxes.daytona.sandbox import DaytonaSandbox
from deepgym.sandboxes.protocol import ExecResult

# PATH of the agent's xfce session, so of every app it starts.
_SESSION_PATH = '/usr/local/bin:/usr/bin:/bin'
# The X display Daytona runs the desktop on.
_DISPLAY = ':0'

# Daytona starts the xfce session by name, and no folder ahead of /usr/local/bin on its
# PATH has one (the check after start proves the session is the agent's). The wrapper
# reruns itself as root, then starts the session as the agent in its offline network
# namespace, on a D-Bus of its own.
_SESSION_WRAPPER = """#!/bin/sh
[ "$(id -u)" = 0 ] || exec sudo -n -E "$0" "$@"
exec {as_agent} env -i DISPLAY="$DISPLAY" HOME={home} LANG="$LANG" LC_ALL="$LC_ALL" \\
  PATH={path} dbus-run-session -- /usr/bin/xfce4-session "$@"
"""

# x11vnc runs as the sudoer outside the agent's network namespace and by default acts on
# root-window properties any X client can set: VNC_CONNECT (connect out) and X11VNC_REMOTE
# (remote control). Turn both off, and with -safer its -gui and -connect file channels too.
_X11VNC_RC = 'novncconnect\nnoremote\nnocmds\nsafer\n'
# The only flags x11vnc may run with: the ones Daytona starts it with. Any other flag, such
# as one that skips that file or undoes a line of it, refuses the desktop.
_X11VNC_FLAGS = '-display -forever -shared -rfbport'

_CHROMIUM_POLICY = json.dumps(
    {
        'DeveloperToolsAvailability': 2,
        'URLBlocklist': [
            'javascript://*',
            'chrome://*',
            'devtools://*',
            'file://*',
            'view-source:*',
        ],
    }
)

# Terminals, run dialogs, Python's IDLE, and the panel and desktop menus that can launch anything.
# This names known launchers, so it can't be complete: it is defense in depth. What the agent
# can still start runs as the agent, offline, with no way to root, the verifier or Daytona's
# daemon; those boundaries don't depend on this list.
_LAUNCHERS = (
    'xfce4-terminal x-terminal-emulator xterm uxterm xfce4-appfinder exo-open idle idle3 '
    'xfce4-panel xfdesktop xfce4-settings-manager xfce4-settings-editor xfce4-keyboard-settings'
)

# Printed last by both scripts below. Without it the desktop counts as not locked down.
_LOCKED = 'deepgym-desktop-locked'

# Runs as root before the desktop starts, so nothing the session loads predates it. Every
# step must take (nothing is skipped quietly), and the end checks, as the agent, that
# nothing it locked still runs or loads.
_LOCKDOWN = """set -eu
if pgrep -x 'Xvfb|xfce4-session|x11vnc' >/dev/null; then
  echo 'the desktop is already running, ahead of its lockdown'; exit 1
fi
rm -f /usr/local/bin/xfce4-session
printf %s {wrapper} > /usr/local/bin/xfce4-session
chmod 755 /usr/local/bin/xfce4-session
# x11vnc reads ~/.x11vncrc, with the HOME Daytona's daemon (pid 1) runs with. It runs as
# that home's owner, not root, so the file must be owned by and readable to that user, or
# x11vnc silently skips it and none of the options below take effect.
home=$(tr '\\0' '\\n' < /proc/1/environ | sed -n 's/^HOME=//p')
[ -d "$home" ] || {{ echo "x11vnc has no home to read its options from: '$home'"; exit 1; }}
printf %s {rc} > "$home/.x11vncrc"
chown "$(stat -c %u:%g "$home")" "$home/.x11vncrc"
chmod 600 "$home/.x11vncrc"
mkdir -p /etc/chromium/policies/managed
printf %s {policy} > /etc/chromium/policies/managed/deepgym.json
if command -v chromium >/dev/null; then
  grep -qF /etc/chromium/policies /usr/lib/chromium/chromium || {{
    echo 'this Chromium does not read /etc/chromium/policies'; exit 1; }}
fi
# Thunar's custom actions, "Open Terminal Here" included, all come from its uca plugin, and
# every GLib "Open With" launch, a typed custom command included, runs through
# gio-launch-desktop: find both wherever the image keeps them.
found=$(find /usr \\( -name thunar-uca.so -o -name gio-launch-desktop \\))
case "$found" in *gio-launch-desktop*) ;; *) echo 'gio-launch-desktop not found'; exit 1;; esac
for path in $found; do
  case "$path" in *.so) rm "$path";; *) chmod 000 "$path";; esac
done
rm -f /etc/xdg/Thunar/uca.xml
for bin in {launchers}; do
  if path=$(command -v "$bin"); then chmod 000 "$(readlink -f "$path")"; fi
done
left=$({as_agent} /bin/sh -c {agent_check} check $found)
[ -z "$left" ] || {{ echo "$left"; exit 1; }}
echo {marker}
"""

# Runs as the agent, on the session's PATH: prints each launcher it can still run, and
# each plugin or helper path (its arguments) it can still load or run.
_AGENT_CHECK = """PATH={path}
for bin in {launchers}; do
  if p=$(command -v "$bin") && [ -x "$p" ]; then echo "the agent can still run $bin"; fi
done
for path in "$@"; do
  if [ -r "$path" ] || [ -x "$path" ]; then echo "the agent can still use $path"; fi
done
"""

# Runs as root once Daytona has started the desktop: the session must be the agent's (so the
# wrapper ran), the rc file must be the one x11vnc's own user can read, and remote control
# must actually be off. The last check queries the running x11vnc instead of trusting its
# argv, which x11vnc rewrites to hide flags like -passwd and -norc; the argv allowlist stays
# as defense in depth for a flag it does show.
_CHECK_STARTED = """set -eu
agent=$(stat -c %u {home})
owners= vnc=
for _ in $(seq 60); do
  owners=$(ps -C xfce4-session -o uid= | tr -d ' ' | sort -u)
  if [ "$owners" = "$agent" ] && vnc=$(pgrep -x x11vnc); then break; fi
  sleep 0.5
done
[ "$owners" = "$agent" ] || {{
  echo "xfce4-session runs as uid '$owners', not as the agent ($agent)"; exit 1; }}
[ -n "$vnc" ] || {{ echo 'x11vnc is not running'; exit 1; }}
for pid in $vnc; do
  for arg in $(tr '\\0' '\\n' < /proc/$pid/cmdline | tail -n +2 | sed 's/^--/-/'); do
    case "$arg" in -*) case " {flags} " in *" $arg "*) ;; *)
      echo "x11vnc runs with $arg, which is not an allowed flag"; exit 1;; esac;; esac
  done
  home=$(tr '\\0' '\\n' < /proc/$pid/environ | sed -n 's/^HOME=//p')
  rc="$home/.x11vncrc"
  [ "$(stat -c %u "$rc" 2>/dev/null)" = "$(stat -c %u /proc/$pid)" ] || {{
    echo "x11vnc cannot read $rc as the user it runs as"; exit 1; }}
  [ "$(cat "$rc")" = "$(printf %s {rc_value})" ] || {{
    echo "x11vnc reads no lockdown options from $home"; exit 1; }}
done
if timeout 8 x11vnc -display {display} -Q ping 2>/dev/null | grep -q 'ans='; then
  echo 'x11vnc still answers remote control, so -noremote did not take effect'; exit 1
fi
echo {marker}
"""


def _confirm(step: str, done: ExecResult) -> None:
    """Accept a lockdown *step* only if its script exited 0 and printed :data:`_LOCKED` last.

    Raises:
        SandboxError: Otherwise.
    """
    if done.exit_code != 0 or done.output.split()[-1:] != [_LOCKED]:
        raise SandboxError(f'Refusing the desktop, its {step} failed: {done.output[-500:]}')


class DaytonaDesktop:
    """Daytona's xfce desktop, run as a session's agent user instead of the sudoer.

    Daytona starts Xvfb, the xfce session and x11vnc as the sandbox's default user, a
    passwordless sudoer whose processes also relay every command's output, and whatever
    a GUI app can start, the agent driving it can too. So the session (window manager,
    file manager, panels, and every app they start) runs as the agent, offline, and the
    lockdown removes the usual ways out of the GUI. Only Xvfb, x11vnc and Daytona's input
    relay stay with the sudoer; none of them runs commands for X clients. The lockdown
    fails closed: a step that doesn't take, or an image laid out other than it expects,
    stops the session instead of starting a desktop that isn't locked down.
    """

    def __init__(self, box: DaytonaSandbox, action_timeout: int, start_timeout: int) -> None:
        self._box = box
        self._action_timeout = action_timeout
        self._start_timeout = start_timeout

    def start(self, as_agent: str, home: str, as_root: Callable[[str, int], ExecResult]) -> None:
        """Lock the desktop down, start it with its session run through *as_agent*, then
        check that the running desktop took the lockdown.

        Args:
            as_agent: Root command prefix that runs a command as the agent, offline.
            home: The agent's home directory.
            as_root: Runs a script as root, out of the image's environment, with a timeout.

        Raises:
            SandboxError: If a lockdown step or check fails, or the desktop doesn't start.
        """
        wrapper = _SESSION_WRAPPER.format(as_agent=as_agent, home=home, path=_SESSION_PATH)
        agent_check = _AGENT_CHECK.format(path=_SESSION_PATH, launchers=_LAUNCHERS)
        rc = shlex.quote(_X11VNC_RC)
        lockdown = _LOCKDOWN.format(
            wrapper=shlex.quote(wrapper),
            rc=rc,
            policy=shlex.quote(_CHROMIUM_POLICY),
            launchers=_LAUNCHERS,
            as_agent=as_agent,
            agent_check=shlex.quote(agent_check),
            marker=_LOCKED,
        )
        _confirm('lockdown', as_root(lockdown, 60))
        try:
            self._box.computer_use.start(request_timeout=self._start_timeout)
        except Exception as exc:
            raise SandboxError(f'Failed to start desktop: {exc}') from exc
        check = _CHECK_STARTED.format(
            home=shlex.quote(home), flags=_X11VNC_FLAGS, rc_value=rc, display=_DISPLAY,
            marker=_LOCKED,
        )  # fmt: skip
        _confirm('check', as_root(check, 60))

    def act(self, action: Action) -> None:
        """Apply a GUI action through Daytona computer use.

        Raises:
            ActionError: If the desktop rejects the input (e.g. an unknown key name).
            SandboxError: If the Daytona computer-use call fails for any other reason.
        """
        try:
            self._dispatch(action)
        except Exception as exc:
            # ponytail: Daytona reports bad input as a generic daemon 500, so match its wording.
            message = str(exc).lower()
            if 'invalid' in message or 'unsupported' in message:
                raise ActionError(f'{action.action_type} rejected: {exc}') from exc
            raise SandboxError(f'{action.action_type} failed: {exc}') from exc

    def _dispatch(self, action: Action) -> None:
        mouse = self._box.computer_use.mouse
        keyboard = self._box.computer_use.keyboard
        deadline = self._action_timeout
        match action.action_type:
            case 'click':
                mouse.click(action.x, action.y, request_timeout=deadline)
            case 'double_click':
                mouse.click(action.x, action.y, double=True, request_timeout=deadline)
            case 'right_click':
                mouse.click(action.x, action.y, button='right', request_timeout=deadline)
            case 'move':
                mouse.move(action.x, action.y, request_timeout=deadline)
            case 'drag':
                mouse.drag(action.x, action.y, action.to_x, action.to_y, request_timeout=deadline)
            case 'scroll':
                mouse.scroll(
                    action.x, action.y, action.content, action.amount, request_timeout=deadline
                )
            case 'type':
                keyboard.type(action.content, request_timeout=deadline)
            case 'key':
                keyboard.hotkey(action.content, request_timeout=deadline)

    def screenshot(self) -> str:
        """Return a full-resolution PNG so pixel coordinates match click targets.

        Raises:
            SandboxError: If the screenshot fails.
        """
        from daytona import ScreenshotOptions

        try:
            shot = self._box.computer_use.screenshot.take_compressed(
                ScreenshotOptions(fmt='png', scale=1.0), request_timeout=self._action_timeout
            )
        except Exception as exc:
            raise SandboxError(f'Failed to take screenshot: {exc}') from exc
        return shot.screenshot
