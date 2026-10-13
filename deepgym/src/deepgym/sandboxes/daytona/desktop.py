"""Daytona's desktop (xfce + Chromium), run as a live session's agent user."""

from __future__ import annotations

import json
import shlex

from deepgym.exceptions import ActionError, SandboxError
from deepgym.models import Action
from deepgym.sandboxes.daytona.sandbox import DaytonaSandbox

# Daytona starts the xfce session by name, and /usr/local/bin comes first on its PATH. The
# wrapper reruns itself as root, then starts the session as the agent in its offline
# network namespace, on a D-Bus of its own.
_SESSION_WRAPPER = """#!/bin/sh
[ "$(id -u)" = 0 ] || exec sudo -n -E "$0" "$@"
exec {as_agent} env -i DISPLAY="$DISPLAY" HOME={home} LANG="$LANG" LC_ALL="$LC_ALL" \\
  PATH=/usr/local/bin:/usr/bin:/bin dbus-run-session -- /usr/bin/xfce4-session "$@"
"""

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
_LAUNCHERS = (
    'xfce4-terminal x-terminal-emulator xterm uxterm xfce4-appfinder exo-open idle idle3 '
    'xfce4-panel xfdesktop xfce4-settings-manager xfce4-settings-editor xfce4-keyboard-settings'
)

# Runs as root before the desktop starts, so nothing the session loads predates it.
_LOCKDOWN = """set -e
printf %s {wrapper} > /usr/local/bin/xfce4-session
chmod 755 /usr/local/bin/xfce4-session
# x11vnc runs as the sudoer and by default obeys remote-control commands from any X client.
printf 'noremote\\nnocmds\\n' > "$HOME/.x11vncrc"
mkdir -p /etc/chromium/policies/managed
printf %s {policy} > /etc/chromium/policies/managed/deepgym.json
# Thunar's custom actions, "Open Terminal Here" included, all come from this plugin.
rm -f /usr/lib/*/thunarx-3/thunar-uca.so /etc/xdg/Thunar/uca.xml
# Every GLib "Open With" launch, a typed custom command included, runs through this helper.
chmod 000 /usr/lib/*/glib-2.0/gio-launch-desktop 2>/dev/null || true
for bin in {launchers}; do
  path=$(command -v "$bin") && chmod 000 "$(readlink -f "$path")" || true
done
"""


class DaytonaDesktop:
    """Daytona's xfce desktop, run as a session's agent user instead of the sudoer.

    Daytona starts Xvfb, the xfce session and x11vnc as the sandbox's default user, a
    passwordless sudoer whose processes also relay every command's output, and whatever
    a GUI app can start, the agent driving it can too. So the session (window manager,
    file manager, panels, and every app they start) runs as the agent, offline, and the
    lockdown removes the usual ways out of the GUI. Only Xvfb, x11vnc and Daytona's input
    relay stay with the sudoer; none of them runs commands for X clients.
    """

    def __init__(self, box: DaytonaSandbox, action_timeout: int, start_timeout: int) -> None:
        self._box = box
        self._action_timeout = action_timeout
        self._start_timeout = start_timeout

    def start(self, as_agent: str, home: str) -> None:
        """Lock the desktop down, then start it with its session run through *as_agent*.

        Args:
            as_agent: Root command prefix that runs a command as the agent, offline.
            home: The agent's home directory.

        Raises:
            SandboxError: If the lockdown fails or the desktop doesn't start.
        """
        wrapper = _SESSION_WRAPPER.format(as_agent=as_agent, home=home)
        script = _LOCKDOWN.format(
            wrapper=shlex.quote(wrapper),
            policy=shlex.quote(_CHROMIUM_POLICY),
            launchers=_LAUNCHERS,
        )
        locked = self._box.exec(script, timeout=60)
        if locked.exit_code != 0:
            raise SandboxError(f'Desktop lockdown failed: {locked.output[-500:]}')
        try:
            self._box.computer_use.start(request_timeout=self._start_timeout)
        except Exception as exc:
            raise SandboxError(f'Failed to start desktop: {exc}') from exc

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
