"""Live episode sessions: one sandbox, many agent actions, one final score.

A Session keeps a sandbox alive across steps so agents can act on real state
(shell, files, or a desktop), then runs the final verifier against that state.

WARNING: LocalBackend runs agent actions on the host without isolation.
Use a sandbox provider (Daytona for desktops) for untrusted agents.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import posixpath
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NamedTuple, Protocol
from uuid import uuid4

from deepgym.exceptions import (
    ActionError,
    DeepGymError,
    SandboxError,
    TimeoutError,
    VerifierError,
)
from deepgym.judge import (
    RUNTIME_FILE,
    RUNTIME_SOURCE,
    SANDBOX_ROOT,
    SOLVER_USER,
    local_isolation,
)
from deepgym.models import (
    Action,
    MultiTurnEnvironment,
    Observation,
    RunResult,
    Trajectory,
    checked_score,
    is_verdict,
)
from deepgym.sandbox import (
    _extract_json_line,
    build_run_result,
    check_exit,
    error_result,
    parse_verifier_output,
    split_wrapped_output,
    wrap_stderr,
)
from deepgym.sandboxes import TRUSTED_PATH, ExecResult, Sandbox

if TYPE_CHECKING:
    from deepgym.sandboxes.daytona import DaytonaDesktop

logger = logging.getLogger(__name__)

_SANDBOX_WORKDIR = '/tmp/deepgym'
# Unprivileged user that runs sandbox session commands, file writes and the desktop.
_AGENT_USER = 'dgagent'
_AGENT_HOME = f'/home/{_AGENT_USER}'
# Separate unprivileged user for verifiers, so the agent can't read, signal or write to them.
_VERIFIER_USER = 'dgverifier'
_ADD_USER = 'useradd --system --no-create-home --shell /usr/sbin/nologin'
# Root-only directory where file writes are staged before the session user copies them.
_STAGING_DIR = '/opt/deepgym-staging'
# Verifier scripts: readable by the verifier user only.
_VERIFIER_DIR = '/opt/deepgym-verifier'
# deepgym_solution for verifiers, readable by the solver user it starts code as.
_RUNTIME_DIR = '/opt/deepgym-runtime'
# The loopback-only network namespace, bind-mounted so it outlives any process.
_NETNS = '/opt/deepgym-netns'
# Folders root trusts and DeepGym's own. The agent owns its work directory, so one that
# holds or sits in any of these would let it plant files that root later runs or reads.
_PROTECTED_PATHS = tuple(
    PurePosixPath(path)
    for path in (
        '/bin', '/boot', '/dev', '/etc', '/lib', '/lib32', '/lib64', '/libx32', '/proc',
        '/root', '/run', '/sbin', '/sys', '/usr', '/var',
        SANDBOX_ROOT, _STAGING_DIR, _VERIFIER_DIR, _RUNTIME_DIR, _NETNS,
    )
)  # fmt: skip
# Seconds to take over, pack or unpack a work directory (it may be large).
_TRANSFER_TIMEOUT_S = 600
# Largest gzipped work directory archive_workdir hands back; it is held in memory.
_MAX_ARCHIVE_BYTES = 256 << 20
# Image variables the agent keeps, besides HOME and PATH: locale and terminal only.
# Every command here clears the whole image environment at the provider, so no loader or
# startup variable (LD_*, BASH_ENV, GCONV_PATH, PERL5OPT, ...) can point a root process at
# a file the agent wrote. An allowlist needs no list of the dangerous names.
_AGENT_KEEP = ('LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'TZ')
_DISPLAY = ':0'
_STEP_OUTPUT = '_step_output.txt'


class StepResult(NamedTuple):
    """Outcome of one agent action."""

    observation: Observation
    reward: float
    done: bool


class VerifierOutput(NamedTuple):
    """What a session verifier printed, with stdout and stderr kept apart."""

    exit_code: int
    stdout: str
    stderr: str


class SessionBackend(Protocol):
    """Where a session's actions actually run."""

    sandbox_id: str
    python: str

    def exec(self, command: str, timeout: int, env: dict[str, str] | None = None) -> ExecResult:
        """Run a bash command in the work directory."""
        ...

    def write_file(self, path: str, content: bytes) -> None:
        """Write a file relative to the work directory."""
        ...

    def verify(self, code: str, timeout: int, env: dict[str, str] | None = None) -> VerifierOutput:
        """Run verifier *code* in the work directory, out of the agent's reach where possible.

        Where the backend can isolate code from the verifier, it sets ``DEEPGYM_RUNTIME``
        and the other variables ``deepgym_solution`` needs. Its stdout and stderr come back
        apart, so only stdout can carry a verdict.
        """
        ...

    def start_desktop(self) -> None:
        """Start the desktop so GUI actions and screenshots work."""
        ...

    def act(self, action: Action) -> None:
        """Apply a GUI action."""
        ...

    def screenshot(self) -> str:
        """Return the screen as a base64 PNG."""
        ...

    def close(self) -> None:
        """Release the sandbox."""
        ...


class LocalBackend:
    """Run actions in a host temp directory. Dev/testing only: no isolation, no desktop."""

    sandbox_id = 'local-session'
    python = shlex.quote(sys.executable)

    def __init__(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix='deepgym_session_'))
        self._groups: set[int] = set()
        # deepgym_solution for verifiers, outside the work directory the agent writes to.
        self._runtime = Path(tempfile.mkdtemp(prefix='deepgym_runtime_'), RUNTIME_FILE)
        self._runtime.write_text(RUNTIME_SOURCE, encoding='utf-8')

    def exec(self, command: str, timeout: int, env: dict[str, str] | None = None) -> ExecResult:
        """Run *command* with bash in the temp directory; output is stdout, then stderr.

        Raises:
            TimeoutError: If the command exceeds *timeout* seconds.
        """
        exit_code, stdout, stderr = self._run(command, timeout, env)
        return ExecResult(exit_code, stdout + ('\n' + stderr if stderr else ''))

    def _run(self, command: str, timeout: int, env: dict[str, str] | None) -> VerifierOutput:
        """Run *command* with bash in the temp directory, keeping stdout and stderr apart.

        Raises:
            TimeoutError: If the command exceeds *timeout* seconds.
        """
        proc_env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin', **(env or {})}
        # Own process group, so background children die with the command or the session.
        proc = subprocess.Popen(
            ['bash', '-c', command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self._dir,
            env=proc_env,
            start_new_session=True,
        )
        self._groups.add(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_group(proc.pid)
            proc.communicate()
            raise TimeoutError(f'Command exceeded {timeout}s timeout') from exc
        return VerifierOutput(proc.returncode, stdout, stderr)

    def write_file(self, path: str, content: bytes) -> None:
        """Write *content* to *path* inside the temp directory."""
        target = self._dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def verify(self, code: str, timeout: int, env: dict[str, str] | None = None) -> VerifierOutput:
        """Run verifier *code* inline; it shares the agent's user here.

        Code it loads through ``deepgym_solution`` runs in the local judge's sandbox.
        """
        runtime = {
            'DEEPGYM_ISOLATION': json.dumps(local_isolation([str(self._runtime.parent)])),
            'DEEPGYM_RUNTIME': str(self._runtime),
            'DEEPGYM_PYTHON': sys.executable,
        }
        return self._run(_inline_python(self.python, code), timeout, {**runtime, **(env or {})})

    def start_desktop(self) -> None:
        """Refuse: the local backend has no display."""
        raise DeepGymError('Desktop environments need the Daytona backend')

    def act(self, action: Action) -> None:
        """Refuse: the local backend has no display."""
        raise DeepGymError('GUI actions need the Daytona backend')

    def screenshot(self) -> str:
        """Refuse: the local backend has no display."""
        raise DeepGymError('Screenshots need the Daytona backend')

    def close(self) -> None:
        """Kill leftover background processes and delete the temp directory."""
        for group in self._groups:
            _kill_group(group)
        shutil.rmtree(self._dir, ignore_errors=True)
        shutil.rmtree(self._runtime.parent, ignore_errors=True)


# Runs as root inside a fresh network namespace: bring loopback up (it starts down and
# slim images have no `ip`), then sleep forever to keep the namespace alive.
_LOOPBACK_UP_THEN_SLEEP = """
import fcntl, os, socket, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
flags = struct.unpack('16sH', fcntl.ioctl(s, 0x8913, struct.pack('16sH', b'lo', 0)))[1]
fcntl.ioctl(s, 0x8914, struct.pack('16sH', b'lo', flags | 1))
os.execvp('sleep', ['sleep', 'infinity'])
"""


# Lets the verifier user start the code it loads (deepgym_solution), and nothing else, as the
# solver with every privilege dropped: sudo runs setpriv as root, which clears the solver's
# supplementary groups (sudo would otherwise restore an image group membership) and sets
# no-new-privs and an empty bounding set before exec. The rule pins every setpriv argument
# but the command, so it can only drop to the solver. A printf format: the %s are setpriv's
# path and the solver's uid and gid, and it must match :func:`_drop_to` argument for argument.
_SUDOERS_RULE = (
    f'{_VERIFIER_USER} ALL=(root) NOPASSWD: '
    '%s --reuid=%s --regid=%s --clear-groups --inh-caps=-all '
    '--bounding-set=-all --no-new-privs -- *\\n'
)

# Each session user's uid and gid must be that account's alone and its only group, so dropping
# to them grants that fresh account's rights and nobody else's. Ruling out root isn't enough:
# a uid or primary gid shared with any other account, or a supplementary group the image put
# the user in (a stray ``shadow:x:42:dgsolver`` line), would carry that account's or group's
# rights too. The passwd gid check catches an account whose primary gid has no group entry.
_IDS_ARE_EXCLUSIVE = (
    f'for u in {_AGENT_USER} {_VERIFIER_USER} {SOLVER_USER}; do '
    'id "$u" >/dev/null 2>&1 || continue; '
    'uid=$(id -u "$u"); gid=$(id -g "$u"); '
    '[ "$(getent passwd | cut -d: -f3 | grep -Fcx "$uid")" = 1 ] && '
    '[ "$(getent passwd | cut -d: -f4 | grep -Fcx "$gid")" = 1 ] && '
    '[ "$(getent group | cut -d: -f3 | grep -Fcx "$gid")" = 1 ] && '
    '[ "$(id -Gn "$u")" = "$u" ] '
    '|| { echo "$u shares an id or a group with another account"; exit 1; }; '
    'done'
)

# Make a root-owned directory the agent can neither redirect nor rewrite. Refuses a symlink
# or a path that resolves elsewhere (a link anywhere above it), creates it, then forces root
# ownership and the mode, which ``mkdir -p -m`` leaves untouched on a directory that already
# exists. Defined once and called for every harness directory.
_HARDEN_DIR = (
    'harden() { '
    'if [ -L "$1" ]; then echo "$1 is a symlink"; exit 1; fi; '
    'mkdir -p "$1"; '
    'if [ "$(realpath "$1")" != "$1" ]; then echo "$1 resolves elsewhere"; exit 1; fi; '
    'chown 0:0 "$1"; chmod "$2" "$1"; }'
)


class SandboxBackend:
    """Run actions in a throwaway sandbox from any provider (see :mod:`deepgym.sandboxes`).

    Agent commands, file writes and the desktop (given one) run as one unprivileged
    user and verifiers as another, both with every capability dropped: root owns the
    binaries every provider exec runs with full privileges, so the agent must never be
    root, and a different user means it can't read, signal or write into a verifier.
    The agent also gets no-new-privs and an empty bounding set, so no setuid binary
    lifts it. Where sudo exists, the verifier keeps it, limited to starting the code it
    loads through ``deepgym_solution`` as a third user that can't touch the verifier,
    with no-new-privs like the agent;
    elsewhere the verifier is locked down like the agent and ``deepgym_solution`` is
    unavailable. When the provider doesn't block the network itself, everything joins
    one loopback-only network namespace, so servers started during setup stay reachable
    across steps while the internet and the host are not.

    The agent owns the work directory, *workdir* (``/tmp/deepgym`` unless given), and
    everything already in it. Every command root runs here resolves binaries through
    :data:`~deepgym.sandboxes.TRUSTED_PATH` with an empty environment (:func:`trusted`), or
    by absolute path, never through the image's PATH, which may include that directory.
    """

    python = 'python3'

    def __init__(
        self,
        sandbox: Sandbox,
        workdir: str = _SANDBOX_WORKDIR,
        *,
        desktop: DaytonaDesktop | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._desktop = desktop
        self.sandbox_id = sandbox.id
        try:
            try:
                self._workdir = check_workdir(workdir)
            except ValueError as exc:
                raise SandboxError(str(exc)) from exc
            quoted = shlex.quote(self._workdir)
            users = sandbox.exec(
                trusted(
                    'set -eu\n'
                    "command -v setpriv >/dev/null || { echo 'setpriv missing'; exit 1; }\n"
                    f'{_HARDEN_DIR}\n'
                    'harden /opt 755\n'
                    f'useradd --system --create-home --home-dir {_AGENT_HOME} '
                    f'--shell /usr/sbin/nologin {_AGENT_USER}\n'
                    f'chmod 755 {_AGENT_HOME}\n'
                    f'{_ADD_USER} {_VERIFIER_USER}\n'
                    f'harden {_STAGING_DIR} 700\n'
                    f'harden {_VERIFIER_DIR} 750\n'
                    f'chgrp {_VERIFIER_USER} {_VERIFIER_DIR}\n'
                    'solver=0; solver_uid=-; solver_gid=-\n'
                    'if command -v sudo >/dev/null && [ -d /etc/sudoers.d ]; then\n'
                    f'  {_ADD_USER} {SOLVER_USER}\n'
                    f'  solver_uid=$(id -u {SOLVER_USER}); solver_gid=$(id -g {SOLVER_USER})\n'
                    # sudo skips names with a dot, so the rule counts only once visudo has
                    # passed it and a rename puts it in place whole, never through a link.
                    '  rule=$(mktemp /etc/sudoers.d/.deepgym.XXXXXX)\n'
                    f'  printf {shlex.quote(_SUDOERS_RULE)} '
                    '"$(command -v setpriv)" "$solver_uid" "$solver_gid" > "$rule"\n'
                    '  chmod 440 "$rule"; visudo -cqf "$rule"\n'
                    '  mv -f "$rule" /etc/sudoers.d/deepgym\n'
                    f'  harden {_RUNTIME_DIR} 755\n'
                    '  solver=1\n'
                    'fi\n'
                    f'mkdir -p {quoted}\n'
                    f'{_IDS_ARE_EXCLUSIVE}\n'
                    f'echo "$(id -u {_AGENT_USER}) $(id -g {_AGENT_USER}) '
                    f'$(id -u {_VERIFIER_USER}) $(id -g {_VERIFIER_USER}) '
                    '$solver $solver_uid $solver_gid '
                    '$(command -v setpriv) $(command -v nsenter || echo -) '
                    '$(command -v sudo || echo -)"\n'
                    f'realpath {quoted}\n'
                ),
                timeout=_TRANSFER_TIMEOUT_S,
            )
            if users.exit_code != 0:
                raise SandboxError(f'Could not create the session users: {users.output[-500:]}')
            *_, ids, real = users.output.strip().splitlines()
            (agent_uid, agent_gid, verifier_uid, verifier_gid, solver, solver_uid, solver_gid,
             setpriv, nsenter, sudo) = ids.split()[-10:]  # fmt: skip
            # Setup proved them exclusive; this guards the parse before they reach setpriv.
            user_ids = [agent_uid, agent_gid, verifier_uid, verifier_gid]
            if solver == '1':
                user_ids += [solver_uid, solver_gid]
            if not all(v.isdecimal() and int(v) > 0 for v in user_ids):
                raise SandboxError(f'Refusing session users with root or unknown ids: {ids!r}')
            # The folder may be a link in the image: own and use where it really leads.
            try:
                self._workdir = check_workdir(real.strip())
            except ValueError as exc:
                raise SandboxError(f'{workdir!r} leads to {real.strip()!r}: {exc}') from exc
            owned = sandbox.exec(
                trusted(f'chown -R {_AGENT_USER}: {shlex.quote(self._workdir)}'),
                timeout=_TRANSFER_TIMEOUT_S,
            )
            if owned.exit_code != 0:
                raise SandboxError(f'Could not hand over {self._workdir}: {owned.output[-500:]}')
            self._as_agent = _drop_to(setpriv, agent_uid, agent_gid)
            self._as_verifier = _drop_to(
                setpriv, verifier_uid, verifier_gid, allow_sudo=solver == '1'
            )
            self._runtime_env: dict[str, str] = {}
            if solver == '1':
                runtime = f'{_RUNTIME_DIR}/{RUNTIME_FILE}'
                sandbox.upload(RUNTIME_SOURCE.encode('utf-8'), runtime)
                # sudo runs setpriv as root so it can clear the solver's groups; the pinned
                # rule allows only this exact drop, so the verifier can't stay root with it.
                drop = shlex.split(_drop_to(setpriv, solver_uid, solver_gid))
                prefix = [sudo, '-n', '-u', 'root', '--', *drop]
                self._runtime_env = {
                    'DEEPGYM_ISOLATION': json.dumps({'kind': 'user', 'prefix': prefix}),
                    'DEEPGYM_RUNTIME': runtime,
                    'DEEPGYM_PYTHON': self.python,
                }
            # Every image variable, cleared at the provider so the first binary of each
            # root command (sh, setpriv, ...) never loads what the image's env names.
            self._image = _image_env(sandbox)
            self._clear = dict.fromkeys(self._image, '')
            self._offline = ''
            if not sandbox.network_blocked:
                self._make_offline_namespace()
                self._offline = f'{shlex.quote(nsenter)} --net={_NETNS} -- '
            locale = {name: self._image[name] for name in _AGENT_KEEP if self._image.get(name)}
            self._agent_env = {
                'HOME': _AGENT_HOME,
                'PATH': self._image.get('PATH') or TRUSTED_PATH,
                **locale,
            }
            self._verifier_base = {'DEEPGYM_AGENT_HOME': _AGENT_HOME, **locale}
        except BaseException:
            self.close()
            raise

    @property
    def workdir(self) -> str:
        """The work directory, as it really is in the sandbox (links resolved)."""
        return self._workdir

    def _make_offline_namespace(self) -> None:
        """Make a loopback-only network namespace and bind-mount it at ``_NETNS``.

        The holder process brings loopback up, then execs ``sleep``; waiting for that
        proves the namespace exists and is ready. The bind mount keeps it alive and
        addressable without trusting a PID that could be reused.
        """
        made = self._sandbox.exec(
            trusted(
                f'setsid unshare --net -- python3 -I -c {shlex.quote(_LOOPBACK_UP_THEN_SLEEP)} '
                '>/dev/null 2>&1 & pid=$!; '
                'for _ in $(seq 100); do '
                '[ "$(cat /proc/$pid/comm 2>/dev/null)" = sleep ] && break; sleep 0.05; done; '
                '[ "$(cat /proc/$pid/comm 2>/dev/null)" = sleep ] && '
                '[ "$(readlink /proc/$pid/ns/net)" != "$(readlink /proc/1/ns/net)" ] && '
                f'touch {_NETNS} && mount --bind /proc/$pid/ns/net {_NETNS} && '
                f'nsenter --net={_NETNS} true'
            ),
            env=self._clear,
            timeout=30,
        )
        if made.exit_code != 0:
            raise SandboxError(
                'Could not isolate the session network; the image needs util-linux '
                f'(unshare, nsenter, mount): {made.output[-500:]}'
            )

    def exec(self, command: str, timeout: int, env: dict[str, str] | None = None) -> ExecResult:
        """Run *command* with bash in the work directory, as the session user, offline.

        Raises:
            TimeoutError: If the command exceeds *timeout* seconds.
        """
        return self._sandbox.exec(
            f'{self._offline}{self._as_agent} bash -c {shlex.quote(command)}',
            cwd=self._workdir,
            env={**self._clear, **self._agent_env, **(env or {})},
            timeout=timeout,
        )

    def exec_root(
        self, command: str, timeout: int, *, offline: bool, env: Mapping[str, str]
    ) -> ExecResult:
        """Run a task's own *command* (such as its tests) with bash as root in the work directory.

        It gets only *env*, never the image's environment, whose startup variables
        (BASH_ENV, PYTHONSTARTUP, ...) could name files in the work directory. Only for a
        sandbox no agent process runs in, such as a grader's fresh one. *offline* joins the
        loopback-only namespace when the provider doesn't block the network.

        Raises:
            TimeoutError: If the command exceeds *timeout* seconds.
        """
        prefix = self._offline if offline else ''
        variables = ' '.join(shlex.quote(f'{key}={value}') for key, value in env.items())
        return self._sandbox.exec(
            f'{prefix}/usr/bin/env -i -- {variables} /bin/bash -c {shlex.quote(command)}',
            cwd=self._workdir,
            env=self._clear,
            timeout=timeout,
        )

    def exec_trusted(self, command: str, timeout: int) -> ExecResult:
        """Run a harness *command* as root with an empty environment and only trusted folders
        on PATH (see :func:`trusted`), in the work directory.

        Raises:
            TimeoutError: If the command exceeds *timeout* seconds.
        """
        return self._sandbox.exec(
            trusted(command), cwd=self._workdir, env=self._clear, timeout=timeout
        )

    def archive_workdir(self) -> bytes:
        """Kill every agent process, then return the work directory as a gzipped tar.

        Root packs it once nothing of the agent's is left running, so nothing changes
        mid-archive, and only if the folder still really is the work directory: an agent
        that can write its parent could have swapped it for a link to somewhere else.
        Symlinks inside are stored as links, never followed.

        Raises:
            SandboxError: If packing fails.
        """
        staged = f'{_STAGING_DIR}/{uuid4().hex}.tgz'
        quoted = shlex.quote(self._workdir)
        # ponytail: the archive comes back through exec output, so it is held in memory
        # here; a provider download() would stream it if work directories outgrow the cap.
        packed = self._sandbox.exec(
            trusted(
                f"{self._as_agent} sh -c 'kill -9 -1' >/dev/null 2>&1; "
                f'[ "$(realpath {quoted})" = {quoted} ] || '
                f'{{ echo "the work directory was moved or replaced by a link"; exit 1; }}; '
                f'tar -C {quoted} -czf {staged} . && '
                f'size=$(wc -c < {staged}) && if [ "$size" -gt {_MAX_ARCHIVE_BYTES} ]; then '
                f'echo "the archive is $size bytes, over {_MAX_ARCHIVE_BYTES}"; '
                f'rm -f {staged}; exit 1; fi'
            ),
            env=self._clear,
            timeout=_TRANSFER_TIMEOUT_S,
        )
        if packed.exit_code != 0:
            raise SandboxError(f'Failed to archive {self._workdir}: {packed.output[-500:]}')
        encoded = self._sandbox.exec(
            trusted(f'base64 {staged}; status=$?; rm -f {staged}; exit $status'),
            env=self._clear,
            timeout=_TRANSFER_TIMEOUT_S,
        )
        try:
            if encoded.exit_code != 0:
                raise ValueError(encoded.output[-500:])
            return base64.b64decode(''.join(encoded.output.split()), validate=True)
        except ValueError as exc:
            raise SandboxError(f'Failed to read the archive: {exc}') from exc

    def restore_workdir(self, archive: bytes) -> None:
        """Replace the work directory with *archive* (from :meth:`archive_workdir`).

        The agent user unpacks it, so a crafted archive can't write where root could.
        Then root takes the files over, as in a freshly built image, so tools that check
        ownership (git) trust them, and strips every setuid, setgid and sticky bit; links
        are changed, never followed.

        Raises:
            SandboxError: If unpacking fails.
        """
        staged = f'{_STAGING_DIR}/{uuid4().hex}.tgz'
        self._sandbox.upload(archive, staged)
        quoted = shlex.quote(self._workdir)
        done = self._sandbox.exec(
            trusted(
                f'rm -rf {quoted} && mkdir -p {quoted} && chown {_AGENT_USER}: {quoted} && '
                f'{self._as_agent} tar -xzpf - -C {quoted} < {staged} && '
                f'chown -hR 0:0 {quoted} && chmod -R a-st {quoted}; '
                f'status=$?; rm -f {staged}; exit $status'
            ),
            env=self._clear,
            timeout=_TRANSFER_TIMEOUT_S,
        )
        if done.exit_code != 0:
            raise SandboxError(f'Failed to restore {self._workdir}: {done.output[-500:]}')

    def write_file(self, path: str, content: bytes) -> None:
        """Write *content* to *path* under the work directory, as the session user.

        Root stages the bytes in a private directory and feeds them on stdin to a write
        done as the session user, so the agent's symlinks can't redirect a root write.

        Raises:
            SandboxError: If the write fails.
        """
        staged = f'{_STAGING_DIR}/{uuid4().hex}'
        self._sandbox.upload(content, staged)
        write = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        target = shlex.quote(f'{self._workdir}/{path}')
        done = self._sandbox.exec(
            trusted(
                f'{self._as_agent} sh -c {shlex.quote(write)} write {target} < {staged}; '
                f'status=$?; rm -f {staged}; exit $status'
            ),
            env=self._clear,
            timeout=60,
        )
        if done.exit_code != 0:
            raise SandboxError(f'Failed to write {path}: {done.output[-500:]}')

    def verify(self, code: str, timeout: int, env: dict[str, str] | None = None) -> VerifierOutput:
        """Run verifier *code* as the verifier user, from a file only that user can read.

        Its HOME is its own folder, which nobody but root can write: the agent's home holds
        dotfiles the agent wrote, which tools a verifier runs (git, pip, ...) would obey.
        ``DEEPGYM_AGENT_HOME`` says where the agent's home is, for verifiers that inspect it.
        The provider mixes stdout and stderr, so they are split as in the judge (see
        :func:`~deepgym.sandbox.wrap_stderr`).

        Raises:
            TimeoutError: If the verifier exceeds *timeout* seconds.
        """
        script = f'{_VERIFIER_DIR}/{uuid4().hex}.py'
        self._sandbox.upload(code.encode('utf-8'), script)
        command, marker = wrap_stderr(
            f'{self._offline}{self._as_verifier} {self.python} -I {script}'
        )
        done = self._sandbox.exec(
            trusted(
                f'{command}; status=$?; rm -f {script}; exit $status',
                {'HOME': _VERIFIER_DIR, **self._verifier_base, **self._runtime_env, **(env or {})},
            ),
            cwd=self._workdir,
            env=self._clear,
            timeout=timeout,
        )
        return VerifierOutput(done.exit_code, *split_wrapped_output(done.output, marker))

    def start_desktop(self) -> None:
        """Start the desktop with its session and apps running as the agent, offline.

        Raises:
            DeepGymError: If this provider has no desktop.
            SandboxError: If the desktop doesn't start.
        """
        self._need_desktop('Desktop environments').start(
            f'{self._offline}{self._as_agent}', _AGENT_HOME, self.exec_trusted
        )
        self._agent_env['DISPLAY'] = _DISPLAY

    def act(self, action: Action) -> None:
        """Apply a GUI action.

        Raises:
            DeepGymError: If this provider has no desktop.
        """
        self._need_desktop('GUI actions').act(action)

    def screenshot(self) -> str:
        """Return the screen as a base64 PNG.

        Raises:
            DeepGymError: If this provider has no desktop.
        """
        return self._need_desktop('Screenshots').screenshot()

    def close(self) -> None:
        """Delete the sandbox.

        Raises:
            SandboxError: If the provider reports the delete failed; the sandbox may still exist.
        """
        if self._sandbox.delete() is False:
            raise SandboxError(f'Could not delete sandbox {self.sandbox_id}')

    def _need_desktop(self, what: str) -> DaytonaDesktop:
        """Return the desktop, or refuse *what* if this provider has none."""
        if self._desktop is None:
            raise DeepGymError(f'{what} need the Daytona backend')
        return self._desktop


def _image_env(sandbox: Sandbox) -> dict[str, str]:
    """Return the image's environment (every NAME=VALUE), read NUL-delimited.

    NUL separators, not newlines, so a value with a newline can't look like two variables.
    """
    listed = sandbox.exec('/usr/bin/env -0', timeout=30).output
    return dict(item.split('=', 1) for item in listed.split('\0') if '=' in item)


def resolve_outside(backend: SandboxBackend, path: str, *writable: str) -> str:
    """Return a PATH of *path*'s entries that resolve, in the sandbox, outside writable trees.

    Each entry is realpath-resolved by the sandbox with NUL-delimited output, so every
    spelling (``//app/bin``) and every link is seen as the kernel sees it, with no parser
    differential. Relative and empty entries (the shell reads them against the working
    folder) don't start with ``/`` and are dropped, as are entries resolving into the work
    directory or any of *writable*. Falls back to :data:`~deepgym.sandboxes.TRUSTED_PATH`.
    """
    entries = [entry for entry in path.split(':') if entry.startswith('/')]
    if not entries:
        return TRUSTED_PATH
    lookups = ' '.join(f'realpath -ez -- {shlex.quote(entry)} 2>/dev/null;' for entry in entries)
    resolved = backend.exec_trusted(lookups, 30).output.split('\0')
    blocked = (backend.workdir, *writable)
    kept = [
        real
        for real in resolved
        if real.startswith('/') and not any(PurePosixPath(real).is_relative_to(b) for b in blocked)
    ]
    return ':'.join(dict.fromkeys(kept)) or TRUSTED_PATH


def check_workdir(workdir: str, reserved: Sequence[str] = ()) -> str:
    """Return *workdir*, normalized, if an agent may own it.

    ``.``, ``..`` and repeated slashes are resolved first (a leading ``//`` too), so no
    spelling of a protected folder slips through. This is lexical; a sandbox also checks
    where the folder really leads before handing it over.

    Args:
        workdir: The folder the agent will own, along with everything in it.
        reserved: More folders it must neither hold nor sit in.

    Raises:
        ValueError: If it isn't absolute, holds /tmp, or holds or sits in a folder root
            trusts (/usr, /etc, /root, ...), DeepGym's own, or *reserved*.
    """
    if not workdir.startswith('/'):
        raise ValueError(f'Work directory {workdir!r} must be an absolute path')
    normal = '/' + posixpath.normpath(workdir).lstrip('/')
    path = PurePosixPath(normal)
    if PurePosixPath('/tmp').is_relative_to(path):
        raise ValueError(f'Work directory {workdir!r} would hold /tmp')
    for other in (*_PROTECTED_PATHS, *map(PurePosixPath, reserved)):
        if other.is_relative_to(path) or path.is_relative_to(other):
            raise ValueError(f'Work directory {workdir!r} overlaps {other}')
    return normal


def trusted(command: str, env: Mapping[str, str] | None = None) -> str:
    """Wrap *command* to run with only *env* and :data:`~deepgym.sandboxes.TRUSTED_PATH`.

    Nothing from the image's environment (PATH, PYTHON*, LD_*) reaches it, and
    ``/usr/bin/env`` and ``/bin/sh`` are named by absolute path. ``--`` ends env's options,
    so no variable is read as one (``-S`` would run a new command).
    """
    assignments = {'PATH': TRUSTED_PATH, **(env or {})}
    variables = ' '.join(shlex.quote(f'{key}={value}') for key, value in assignments.items())
    return f'/usr/bin/env -i -- {variables} /bin/sh -c {shlex.quote(command)}'


def _drop_to(setpriv: str, uid: str, gid: str, *, allow_sudo: bool = False) -> str:
    """Command prefix that drops to *uid*/*gid* with no capabilities, via *setpriv* by path.

    It also sets no-new-privs and empties the bounding set so no setuid binary can
    regain privileges, unless *allow_sudo*: sudo needs both, and its rules then decide.
    """
    lockdown = [] if allow_sudo else ['--bounding-set=-all', '--no-new-privs']
    return shlex.join([
        setpriv, f'--reuid={uid}', f'--regid={gid}', '--clear-groups', '--inh-caps=-all',
        *lockdown, '--',
    ])  # fmt: skip


class Session:
    """Run one episode of a multi-turn environment against a live backend.

    Usage::

        with dg.session(env) as session:
            obs = session.start()
            while not session.done:
                obs, reward, done = session.step(agent(obs))
            result = session.score()
    """

    def __init__(self, env: MultiTurnEnvironment, backend: SessionBackend) -> None:
        self._env = env
        self._backend = backend
        self._trajectory = Trajectory()
        self._obs: Observation | None = None
        self._step = 0
        self._done = False
        self._step_verdict: dict[str, object] | None = None
        self._step_error: VerifierError | None = None
        # One sandbox, one actor at a time: concurrent step/score/close calls queue up.
        self._lock = threading.Lock()

    @property
    def done(self) -> bool:
        """Whether the episode has ended."""
        return self._done

    @property
    def trajectory(self) -> Trajectory:
        """Everything that happened so far."""
        return self._trajectory

    @property
    def step_verdict(self) -> dict[str, object] | None:
        """The step verifier output of the last step (score, passed, details, ...).

        None before any step verifier ran, or if the last one failed.

        ``exit_code`` is the verifier's exit code, in place of any key it printed by that name.
        """
        return self._step_verdict

    @property
    def sandbox_id(self) -> str:
        """Identifier of the backing sandbox."""
        return self._backend.sandbox_id

    def start(self) -> Observation:
        """Write files, start the desktop, run setup, and return the first observation.

        Raises:
            DeepGymError: If the session was already started.
            SandboxError: If setup exits non-zero.
            TimeoutError: If setup exceeds ``setup_timeout``.
        """
        with self._lock:
            if self._obs is not None:
                raise DeepGymError('Session already started; open a new session per episode')
            env = self._env
            for path, content in env.files.items():
                self._backend.write_file(path, content.encode('utf-8'))
            if env.desktop:
                self._backend.start_desktop()
            output = ''
            if env.setup_script:
                output = _checked(
                    'Setup script', self._backend.exec(env.setup_script, env.setup_timeout)
                )
            if env.setup_code:
                output = _checked(
                    'Setup code', self._run_python('_setup.py', env.setup_code, env.setup_timeout)
                )
            self._obs = self._observe(output, done=False)
            return self._obs

    def step(self, action: Action) -> StepResult:
        """Apply one action and return the new observation, step reward, and done flag.

        Raises:
            DeepGymError: If the session is not running.
            ActionError: If the action is not allowed here or the desktop rejects it.
            VerifierError: If the step verifier fails or gives no verdict. The episode is
                then over: the step gets no reward and :meth:`score` reports no verdict.
        """
        with self._lock:
            if self._obs is None or self._done:
                raise DeepGymError('Session is not running; call start() or open a new session')
            if action.action_type not in self._env.allowed_actions:
                raise ActionError(
                    f'Action {action.action_type!r} not allowed here. '
                    f'Allowed: {sorted(self._env.allowed_actions)}'
                )

            self._step += 1
            output, exit_code = self._apply(action)
            try:
                reward = self._step_reward(output) if self._env.step_verifier_code else 0.0
            except VerifierError as exc:
                self._step_error = exc
                try:
                    self._record(action, output, exit_code, reward=None, done=True)
                except Exception as late:
                    # The verifier's failure decides the episode, not the screenshot after it.
                    raise exc from late
                raise
            done = (
                action.action_type == 'done'
                or _signals_done(output)
                or self._step >= self._env.max_steps
            )
            observation = self._record(action, output, exit_code, reward=reward, done=done)
            return StepResult(observation, reward, done)

    def score(self) -> RunResult:
        """Run the final verifier against the current state.

        An episode whose step verifier failed has no verdict: the final verifier doesn't
        run, and the result carries that failure as its ``error_type``.

        Raises:
            VerifierError: If the verifier output is not valid protocol JSON, or the
                verifier crashed or was killed after printing it.
            TimeoutError: If the verifier exceeds the per-step timeout.
        """
        with self._lock:
            if self._step_error is not None:
                return error_result(self._step_error, self.sandbox_id)
            if not self._env.final_verifier_code:
                return RunResult(
                    score=0.0,
                    passed=False,
                    output='No final verifier provided',
                    stderr='',
                    exit_code=0,
                    execution_time_ms=0.0,
                    sandbox_id=self.sandbox_id,
                    error_type='no_verifier',
                )
            start = time.perf_counter()
            exit_code, stdout, stderr = self._backend.verify(
                _guarded(self._env.final_verifier_code), self._env.timeout_per_step
            )
            stderr = stderr.strip()
            result = parse_verifier_output(stdout, f'exit_code={exit_code} stderr: {stderr!r}')
            elapsed_ms = (time.perf_counter() - start) * 1000
            return build_run_result(
                result, elapsed_ms, self.sandbox_id, stderr=stderr, exit_code=exit_code
            )

    def close(self) -> None:
        """Release the backend, waiting for any in-flight step to finish."""
        with self._lock:
            self._backend.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *args: object) -> None:
        """Close the session; a failed delete is logged, never traded for the body's outcome.

        An explicit close() raises instead, for callers that retry it (the API server).
        """
        try:
            self.close()
        except SandboxError:
            logger.warning(
                'Could not delete sandbox %s; it may still exist', self.sandbox_id, exc_info=True
            )

    def _apply(self, action: Action) -> tuple[str, int | None]:
        """Execute *action*; return its text output and exit code (None if not a command)."""
        timeout = self._env.timeout_per_step
        try:
            match action.action_type:
                case 'code':
                    exit_code, output = self._run_python('_step.py', action.content)
                    return output, exit_code
                case 'bash':
                    exit_code, output = self._backend.exec(action.content, timeout)
                    return output, exit_code
                case 'write_file':
                    self._backend.write_file(str(action.path), action.content.encode('utf-8'))
                    return f'wrote {action.path}', None
                case 'wait':
                    time.sleep(min(action.amount, timeout))
                    return '', None
                case 'done' | 'screenshot':
                    return '', None
        except TimeoutError:
            return f'[timed out after {timeout}s]', None
        self._backend.act(action)
        return '', None

    def _run_python(self, filename: str, code: str, timeout: int | None = None) -> ExecResult:
        self._backend.write_file(filename, code.encode('utf-8'))
        return self._backend.exec(
            f'{self._backend.python} {filename}', timeout or self._env.timeout_per_step
        )

    def _record(
        self,
        action: Action,
        output: str,
        exit_code: int | None,
        *,
        reward: float | None,
        done: bool,
    ) -> Observation:
        """Append the step to the trajectory and return the new current observation.

        A step without a verdict (*reward* None) adds no step reward, never a 0.
        """
        self._trajectory.steps.append((self._obs, action))
        if reward is not None:
            self._trajectory.step_rewards.append(reward)
            self._trajectory.total_reward += reward
        self._done = done
        self._obs = self._observe(output, done=done, exit_code=exit_code)
        if done:
            self._trajectory.final_observation = self._obs
        return self._obs

    def _observe(self, output: str, *, done: bool, exit_code: int | None = None) -> Observation:
        return Observation(
            content=output,
            step=self._step,
            done=done,
            screenshot=self._backend.screenshot() if self._env.desktop else None,
            metadata=None if exit_code is None else {'exit_code': exit_code},
        )

    def _step_reward(self, output: str) -> float:
        """Run the step verifier and return its score in [0, 1].

        Only its stdout counts, so warnings on stderr can't break the plain-float form.

        Raises:
            InvalidScoreError: If the score is NaN, infinite, or outside [0, 1].
            VerifierError: If the step verifier times out, prints neither a float nor
                JSON with a score, doesn't exit cleanly (see :func:`check_exit`), or
                reports its own failure (exit 2, or an ``error_type`` that isn't the
                model's fault; see :func:`~deepgym.models.is_verdict`).
        """
        self._step_verdict = None
        self._backend.write_file(_STEP_OUTPUT, output.encode('utf-8'))
        try:
            exit_code, stdout, _ = self._backend.verify(
                _guarded(self._env.step_verifier_code),
                self._env.timeout_per_step,
                env={'STEP_OUTPUT_PATH': _STEP_OUTPUT, 'STEP_NUMBER': str(self._step)},
            )
        except TimeoutError as exc:
            raise VerifierError(f'Step verifier timed out at step {self._step}') from exc
        stdout = stdout.strip()
        json_str = _extract_json_line(stdout)
        try:
            data = json.loads(json_str) if json_str else {'score': float(stdout)}
            value = data['score']
        except (ValueError, TypeError, KeyError) as exc:
            raise VerifierError(
                f'Step verifier output unparseable at step {self._step}: {stdout!r}'
            ) from exc
        score = checked_score(value)
        check_exit(exit_code, score, data.get('passed'))
        error_type = data.get('error_type')
        if not isinstance(error_type, str | None) or not is_verdict(exit_code, error_type):
            raise VerifierError(
                f'Step verifier gave no verdict at step {self._step}: exit code {exit_code}, '
                f'error_type {error_type!r}, details {data.get("details")!r}'
            )
        self._step_verdict = {**data, 'exit_code': exit_code}
        return score


def run_episode(
    session: Session, agent: Callable[[Observation], Action]
) -> tuple[Trajectory, RunResult]:
    """Drive *session* with *agent* until done, then score it.

    Args:
        session: A fresh, unstarted session. It is closed afterwards.
        agent: Callable mapping an observation to the next action.

    Returns:
        Tuple of (trajectory, final_result).
    """
    with session:
        obs = session.start()
        while not session.done:
            obs = session.step(agent(obs)).observation
        return session.trajectory, session.score()


def _kill_group(group: int) -> None:
    """SIGKILL a process group; it may already be gone."""
    # ponytail: a group id could be reused within one session's lifetime; accepted, low risk.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group, signal.SIGKILL)


def _checked(what: str, result: ExecResult) -> str:
    """Return *result*'s output, or raise if the command failed."""
    if result.exit_code != 0:
        raise SandboxError(f'{what} exited {result.exit_code}: {result.output[-2000:]}')
    return result.output


# Runs every session verifier. Python exits 1 on an uncaught exception, the protocol's
# "failed" code, so a crash becomes 2 (verifier error) and never counts as a scored
# failure. It also puts deepgym_solution on the path where the backend ships it.
_VERIFIER_GUARD = """\
import os
import sys
import traceback

if 'DEEPGYM_RUNTIME' in os.environ:
    sys.path.insert(0, os.path.dirname(os.environ['DEEPGYM_RUNTIME']))
try:
    exec(compile({code!r}, '<verifier>', 'exec'), {{'__name__': '__main__'}})
except SystemExit:
    raise
except BaseException:
    traceback.print_exc()
    raise SystemExit(2)
"""


def _guarded(code: str) -> str:
    """Wrap verifier *code* in :data:`_VERIFIER_GUARD`."""
    return _VERIFIER_GUARD.format(code=code)


def _inline_python(python: str, code: str) -> str:
    """Build a command that runs *code* isolated (-I) without writing it to disk.

    Verifiers never touch the sandbox filesystem, so an agent process can't swap
    the file between upload and execution.
    """
    encoded = base64.b64encode(code.encode('utf-8')).decode('ascii')
    loader = f"import base64;exec(compile(base64.b64decode('{encoded}'),'<verifier>','exec'))"
    return f'{python} -I -c "{loader}"'


def _signals_done(output: str) -> bool:
    """Return True if output has a line 'DONE' or a JSON line with "done": true."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == 'DONE':
            return True
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                if json.loads(stripped).get('done') is True:
                    return True
            except (json.JSONDecodeError, AttributeError):
                continue
    return False
