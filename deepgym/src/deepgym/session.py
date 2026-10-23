"""Live episode sessions: one sandbox, many agent actions, one final score.

A Session keeps a sandbox alive across steps so agents can act on real state
(shell, files, or a desktop), then runs the final verifier against that state.

WARNING: LocalBackend runs agent actions on the host without isolation.
Use a sandbox provider (Daytona for desktops) for untrusted agents.
"""

from __future__ import annotations

import base64
import contextlib
import errno
import json
import logging
import math
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
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol
from uuid import uuid4

from pydantic import JsonValue

from deepgym.cli_runtime import CliExecution, CliRequest, EventObserver, execute_cli
from deepgym.exceptions import (
    ActionError,
    DeepGymError,
    SandboxDiedError,
    SandboxError,
    SandboxExpiredError,
    SessionClosedError,
    TimeoutError,
    VerifierError,
    is_infrastructure_failure,
)
from deepgym.judge import (
    DEADLINE_GRACE_S,
    RUNTIME_FILE,
    RUNTIME_SOURCE,
    SANDBOX_ROOT,
    SOLVER_USER,
    ids_are_exclusive,
    local_isolation,
    sandbox_python,
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
    ExecutionResult,
    _extract_json_line,
    _judged_zero,
    build_run_result,
    check_exit,
    error_result,
    parse_verifier_output,
    solution_timeout,
    split_wrapped_output,
    wrap_stderr,
)
from deepgym.sandboxes import (
    TRUSTED_PATH,
    ExecResult,
    Sandbox,
    SandboxImage,
    SandboxTimeoutError,
    sealed_path,
)
from deepgym.tools import HttpTool, lost_response, redact, run_command
from deepgym.verifier_template import guard_crashes

if TYPE_CHECKING:
    from deepgym.sandboxes.daytona import DaytonaDesktop

logger = logging.getLogger(__name__)

LOCAL_IMAGE = SandboxImage(None, None, 'the local backend runs on the host, not in an image')

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
# errno names for a write the agent's own files block (a folder where the file goes, a file
# where a folder goes, a link or mode it set, a name too long, a FIFO with no reader), raised
# before any byte is written. Any other write failure is the sandbox's. Names, as the
# numbers vary by platform.
_BLOCKED_ERRNOS = (
    'EACCES', 'EEXIST', 'EISDIR', 'ELOOP', 'ENAMETOOLONG', 'ENOTDIR', 'ENXIO', 'EPERM',
)  # fmt: skip
# Opens a write target without blocking on a FIFO the agent left there (ENXIO instead).
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NONBLOCK
# Run as the session user to write stdin to argv[1]: exits 3 if the agent's files block it,
# including a link it planted at the path, whatever the error following that link.
_WRITE_AS_AGENT = f"""
import errno, os, sys
try:
    os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)
    fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NONBLOCK, 0o666)
except OSError as exc:
    print(exc, file=sys.stderr)
    blocked = exc.errno in {{getattr(errno, n) for n in {_BLOCKED_ERRNOS}}}
    sys.exit(3 if blocked or os.path.islink(sys.argv[1]) else 1)
with os.fdopen(fd, 'wb') as out:
    out.write(sys.stdin.buffer.read())
"""


class VerifierUploadError(SandboxError):
    """Raise when the provider fails to take a session verifier's code.

    It goes to the verifier's own folder, but on the disk the agent shares, so once the
    agent has acted it may be why (it filled the disk or its inodes): as for any write
    after an action, that is its judged 0 unless the network or provider failed (see
    :func:`failure_result`). Before any action, it is ours: no verdict.
    """


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

    def exec_input(self, command: str, data: str, timeout: float) -> ExecResult:
        """Run a bash command as exec does, with *data* on its stdin; where the backend
        isolates the agent, *data* is never in a file the session user can read."""
        ...

    def write_file(self, path: str, content: bytes) -> None:
        """Write a file relative to the work directory.

        ActionError means the agent's own files are in the way, not that the sandbox failed.
        """
        ...

    def verify(self, code: str, timeout: int, env: dict[str, str] | None = None) -> VerifierOutput:
        """Run verifier *code* in the work directory, out of the agent's reach where possible.

        Where the backend can isolate code from the verifier, it sets ``DEEPGYM_RUNTIME``
        and the other variables ``deepgym_solution`` needs, with *timeout* as the deadline
        for the code it loads; the verifier gets ``DEADLINE_GRACE_S`` more to report. Its
        stdout and stderr come back apart, so only stdout can carry a verdict.
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

    def booted_image(self) -> SandboxImage:
        """Return the image the sandbox booted and its digest (see :class:`Sandbox`)."""
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
        return _joined(self._run(command, timeout, env))

    def exec_input(self, command: str, data: str, timeout: float) -> ExecResult:
        """Run *command* as :meth:`exec` does, with *data* on its stdin.

        Raises:
            TimeoutError: If the command exceeds *timeout* seconds.
        """
        return _joined(self._run(command, timeout, None, data))

    def _run(
        self, command: str, timeout: float, env: dict[str, str] | None, data: str | None = None
    ) -> VerifierOutput:
        """Run *command* with bash in the temp directory, keeping stdout and stderr apart.

        Raises:
            SandboxTimeoutError: If the command exceeds *timeout* seconds, with its stdout.
        """
        proc_env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin', **(env or {})}
        # Own process group, so background children die with the command or the session.
        proc = subprocess.Popen(
            ['bash', '-c', command],
            stdin=None if data is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors='replace',
            cwd=self._dir,
            env=proc_env,
            start_new_session=True,
        )
        self._groups.add(proc.pid)
        try:
            stdout, stderr = proc.communicate(data, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_group(proc.pid)
            stdout, _ = proc.communicate()
            raise SandboxTimeoutError(f'Command exceeded {timeout}s timeout', stdout) from exc
        return VerifierOutput(proc.returncode, stdout, stderr)

    def write_file(self, path: str, content: bytes) -> None:
        """Write *content* to *path* inside the temp directory.

        Raises:
            ActionError: If the agent's own files are in the way (see ``_BLOCKED_ERRNOS``).
            SandboxError: If the write fails any other way.
        """
        target = self._dir / path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, _WRITE_FLAGS, 0o666)
        except OSError as exc:
            if errno.errorcode.get(exc.errno) in _BLOCKED_ERRNOS or target.is_symlink():
                raise ActionError(f'Failed to write {path}: {exc}') from exc
            raise SandboxError(f'Failed to write {path}: {exc}') from exc
        try:
            with os.fdopen(fd, 'wb') as file:
                file.write(content)
        except OSError as exc:
            raise SandboxError(f'Failed to write {path}: {exc}') from exc

    def verify(self, code: str, timeout: int, env: dict[str, str] | None = None) -> VerifierOutput:
        """Run verifier *code* inline; it shares the agent's user here.

        Code it loads through ``deepgym_solution`` runs in the local judge's sandbox, which
        reads the runtime and the agent's working directory: the files the agent wrote.
        """
        isolation = local_isolation([str(self._runtime.parent), str(self._dir)])
        runtime = {
            'DEEPGYM_ISOLATION': json.dumps(isolation),
            'DEEPGYM_RUNTIME': str(self._runtime),
            'DEEPGYM_PYTHON': sandbox_python(),
            'DEEPGYM_DEADLINE': str(time.time() + timeout),
            'DEEPGYM_POLICY_DIR': str(self._dir),
        }
        command = _inline_python(self.python, code)
        return self._run(command, timeout + DEADLINE_GRACE_S, {**runtime, **(env or {})})

    def start_desktop(self) -> None:
        """Refuse: the local backend has no display."""
        raise DeepGymError('Desktop environments need the Daytona backend')

    def act(self, action: Action) -> None:
        """Refuse: the local backend has no display."""
        raise DeepGymError('GUI actions need the Daytona backend')

    def screenshot(self) -> str:
        """Refuse: the local backend has no display."""
        raise DeepGymError('Screenshots need the Daytona backend')

    def booted_image(self) -> SandboxImage:
        """Return :data:`LOCAL_IMAGE`: there is no image."""
        return LOCAL_IMAGE

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

# Dropping to a session user must grant that fresh account's rights and nobody else's.
_IDS_ARE_EXCLUSIVE = ids_are_exclusive(_AGENT_USER, _VERIFIER_USER, SOLVER_USER)

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

# Run in the sandbox at setup to prove only root can change what TRUSTED_PATH finds.
_SEALED_PATH_CHECK = Path(sealed_path.__file__).read_text(encoding='utf-8')


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
            # Root and verifiers run what TRUSTED_PATH finds, so only root may change any of
            # it. Checked once the agent owns its work directory, which no folder on it may
            # lead into; anything but a clean pass refuses the sandbox.
            sealed = sandbox.exec(
                trusted(
                    f'{self.python} -I -c {shlex.quote(_SEALED_PATH_CHECK)} '
                    f'{shlex.join(TRUSTED_PATH.split(":"))}'
                ),
                timeout=60,
            )
            if sealed.exit_code != 0 or sealed.output.strip() != sealed_path.SEALED:
                raise SandboxError(f'Refusing the sandbox: {sealed.output[-500:]}')
            self._as_agent = _drop_to(setpriv, agent_uid, agent_gid)
            self._agent_drop = self._as_agent
            self._agent_namespace = ''
            self._nsenter = nsenter
            self._agent_uid = int(agent_uid)
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
            if self._offline:
                self._make_agent_namespace()
        except BaseException:
            self.close()
            raise

    @property
    def workdir(self) -> str:
        """The work directory, as it really is in the sandbox (links resolved)."""
        return self._workdir

    def _make_agent_namespace(self) -> None:
        directory = f'{_STAGING_DIR}/agent-processes'
        self._agent_scope = directory
        control = f'{directory}/control.py'
        _checked('Agent containment directory', self.exec_trusted(f'mkdir -m 700 {directory}', 30))
        source = Path(__file__).with_name('_pid_namespace.py').read_bytes()
        self._sandbox.upload(source, control)
        _checked(
            'Agent PID namespace',
            self.exec_trusted(f'python3 -I {control} create {directory} {self._agent_uid}', 30),
        )
        self._agent_namespace = (
            f'{shlex.quote(self._nsenter)} --pid={directory}/pid --mount={directory}/mnt -- '
        )
        self._as_agent = self._agent_namespace + self._agent_drop

    def stop_agent(self) -> None:
        """Destroy all setup and CLI descendants before grading; refuse uncertain cleanup.

        Root services and the verifier never enter this PID namespace.

        Raises:
            SandboxError: If kernel teardown or the no-survivor check fails.
        """
        directory = self._agent_scope
        _checked(
            'Agent PID namespace teardown',
            self.exec_trusted(f'python3 -I {directory}/control.py stop {directory}', 30),
        )

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

    def exec_input(self, command: str, data: str, timeout: float) -> ExecResult:
        """Run *command* as the session user, offline, with *data* on its stdin.

        Root uploads *data* to its private staging directory and opens it for the command,
        so the session user gets the bytes but never a path to them, and deletes it after (a
        command the timeout stops can leave it there, still root-only). The command gets
        only :data:`~deepgym.sandboxes.TRUSTED_PATH`, never the image's environment.

        Raises:
            TimeoutError: If the command exceeds *timeout* seconds.
        """
        staged = f'{_STAGING_DIR}/{uuid4().hex}'
        self._sandbox.upload(data.encode('utf-8'), staged)
        return self._sandbox.exec(
            trusted(
                f'{self._offline}{self._as_agent} {command} < {staged}; '
                f'status=$?; rm -f {staged}; exit $status'
            ),
            cwd=self._workdir,
            env=self._clear,
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

    def run_cli(self, request: CliRequest, on_event: EventObserver) -> CliExecution:
        """Run the CLI with fixed-provider egress and kill every agent process on exit.

        Args:
            request: CLI command and explicit model-readable credential opt-in.
            on_event: Observer returning a stop reason to terminate the CLI.

        Returns:
            Protected captured events and process termination status.

        Raises:
            SandboxError: If supervision or isolated provider transport is unavailable.
        """
        return execute_cli(self, request, on_event)

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
            ActionError: If the agent's work directory can't be packed (it was moved or
                replaced by a link, is over the size cap, or the disk is full): its doing.
            SandboxError: If reading the archive back fails.
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
            raise ActionError(f'Failed to archive {self._workdir}: {packed.output[-500:]}')
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
            ActionError: If the agent's own files are in the way (see ``_BLOCKED_ERRNOS``).
            SandboxError: If the write fails any other way.
        """
        staged = f'{_STAGING_DIR}/{uuid4().hex}'
        self._sandbox.upload(content, staged)
        target = shlex.quote(f'{self._workdir}/{path}')
        write = f'{self.python} -I -c {shlex.quote(_WRITE_AS_AGENT)} {target}'
        done = self._sandbox.exec(
            trusted(
                f'{self._as_agent} {write} < {staged}; status=$?; rm -f {staged}; exit $status'
            ),
            env=self._clear,
            timeout=60,
        )
        if done.exit_code == 3:
            raise ActionError(f'Failed to write {path}: {done.output[-500:]}')
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
            SandboxError: If the sandbox stopped (see :class:`~deepgym.sandboxes.Sandbox`).
            VerifierUploadError: If the provider fails to take the verifier's code.
        """
        script = f'{_VERIFIER_DIR}/{uuid4().hex}.py'
        command, marker = wrap_stderr(
            f'{self._offline}{self._as_verifier} {self.python} -I {script}'
        )
        # The deadline is set by the sandbox's own clock, as the judge's is.
        deadline = f'export DEEPGYM_DEADLINE=$(( $(date +%s) + {timeout} ))'
        try:
            self._sandbox.upload(code.encode('utf-8'), script)
        except (SandboxDiedError, SandboxExpiredError):
            raise
        except Exception as exc:
            raise VerifierUploadError(f'Could not upload the verifier: {exc}') from exc
        done = self._sandbox.exec(
            trusted(
                f'{deadline}; {command}; status=$?; rm -f {script}; exit $status',
                {
                    'HOME': _VERIFIER_DIR,
                    'DEEPGYM_POLICY_DIR': f'{self._workdir}{os.pathsep}{_AGENT_HOME}',
                    **self._verifier_base,
                    **self._runtime_env,
                    **(env or {}),
                },
            ),
            cwd=self._workdir,
            env=self._clear,
            timeout=timeout + DEADLINE_GRACE_S,
        )
        return VerifierOutput(done.exit_code, *split_wrapped_output(done.output, marker))

    def start_desktop(self) -> None:
        """Start the desktop with its session and apps running as the agent, offline.

        Raises:
            DeepGymError: If this provider has no desktop.
            SandboxError: If the desktop doesn't start.
        """
        self._need_desktop('Desktop environments').start(
            f'{self._offline}{self._as_agent}',
            _AGENT_HOME,
            self._agent_env['PATH'],
            self.exec_trusted,
        )
        self._agent_env['DISPLAY'] = _DISPLAY

    def act(self, action: Action, *, timeout: float | None = None) -> None:
        """Apply a GUI action.

        Raises:
            DeepGymError: If this provider has no desktop.
        """
        if timeout is None:
            self._need_desktop('GUI actions').act(action)
        else:
            self._need_desktop('GUI actions').act(action, timeout=timeout)

    def screenshot(self, *, timeout: float | None = None) -> str:
        """Return the screen as a base64 PNG.

        Raises:
            DeepGymError: If this provider has no desktop.
        """
        if timeout is None:
            return self._need_desktop('Screenshots').screenshot()
        return self._need_desktop('Screenshots').screenshot(timeout=timeout)

    def booted_image(self) -> SandboxImage:
        """Return the image the sandbox booted and its digest, as its provider reports them."""
        return self._sandbox.booted_image()

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
        self._cli_finalized = False
        self._step_verdict: dict[str, object] | None = None
        # The episode's result, once a failed step fixed it (see step()).
        self._failed_result: RunResult | None = None
        # The session's start, and the seconds spent applying the agent's actions since.
        self._opened = time.monotonic()
        self._acting_s = 0.0
        # Request records of the tool writes sent, until settled (see score()).
        self._unsettled: list[JsonValue] = []
        # One sandbox, one actor at a time: concurrent step/score/close calls queue up.
        self._lock = threading.Lock()
        # Set by close() whether or not the delete succeeds (see _check_open).
        self._closed = False

    @property
    def done(self) -> bool:
        """Whether the episode has ended."""
        return self._done

    @property
    def trajectory(self) -> Trajectory:
        """Everything that happened so far."""
        return self._trajectory

    @property
    def tools(self) -> tuple[HttpTool, ...]:
        """The API tools the environment declares, for tool-calling agents."""
        return self._env.tools

    @property
    def step_verdict(self) -> dict[str, object] | None:
        """The step verifier output of the last step (score, passed, details, ...).

        None before any step verifier ran, or if the last step got no verdict from it.

        ``exit_code`` is the verifier's exit code, in place of any key it printed by that name.
        """
        return self._step_verdict

    @property
    def failed_result(self) -> RunResult | None:
        """The result a failed step fixed (see :meth:`step`); None while no step failed."""
        return self._failed_result

    @property
    def sandbox_id(self) -> str:
        """Identifier of the backing sandbox."""
        return self._backend.sandbox_id

    def booted_image(self) -> SandboxImage:
        """Return the image the backing sandbox booted and its digest, or why there is none."""
        return self._backend.booted_image()

    def start(self) -> Observation:
        """Write files, start the desktop, run setup, and return the first observation.

        Raises:
            DeepGymError: If the session was already started or is closed.
            SandboxError: If setup exits non-zero.
            TimeoutError: If setup exceeds ``setup_timeout``.
        """
        with self._lock:
            self._check_open()
            if self._obs is not None:
                raise DeepGymError('Session already started; open a new session per episode')
            env = self._env
            if env.root_service_start:
                if not isinstance(self._backend, SandboxBackend):
                    raise SandboxError('root_service_start requires an isolated SandboxBackend')
                _checked(
                    'Root service startup',
                    self._backend.exec_root(
                        env.root_service_start,
                        env.setup_timeout,
                        offline=True,
                        env={'PATH': TRUSTED_PATH},
                    ),
                )
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

    def step(self, action: Action, *, timeout: float | None = None) -> StepResult:
        """Apply one action and return the new observation, step reward, and done flag.

        The new observation (with its screenshot) is taken before the step verifier runs,
        and the step is recorded only once both are in. A step that fails any other way
        than a rejected action ends the episode: the action may have changed the sandbox, so
        it stays on the record, with no step reward and no retry. :meth:`score` then reports
        what :func:`failure_result` makes of the failure: no verdict only where it is
        ours for certain and the agent didn't use the sandbox up, else a judged 0. A step
        verifier that times out after reading the agent's output gives the step a judged 0
        (``solution_timeout``) instead, and the episode goes on.

        A tool action's arguments are checked against its declared tool before it counts
        as a step. Only an explicit ``done`` ends an episode of tool actions: their output
        is the app's, so a DONE line in it never does.

        ``timeout`` bounds agent execution and the following screenshot by one remaining
        wall-time deadline. Desktop overrides require SandboxBackend. Verifier time is
        separate; provider transport can take its own cleanup grace after a deadline.

        Raises:
            DeepGymError: If the session is not running or is closed.
            ActionError: If the action is not allowed here, names no declared tool or has
                arguments its tool rejects, the desktop rejects it, or the agent's own files
                are in the way of a write (the episode goes on, unless they blocked the
                step's output file).
            VerifierError: If the step verifier fails, gives no verdict, or times out before
                it reads the agent's output.
            SandboxError: If the sandbox fails to apply the action, take the screenshot or
                run the step verifier.
        """
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('Step timeout must be finite and positive')
        deadline = time.monotonic() + timeout if timeout is not None else None
        with self._lock:
            self._check_open()
            if self._obs is None or self._done:
                raise DeepGymError('Session is not running; call start() or open a new session')
            if action.action_type not in self._env.allowed_actions:
                raise ActionError(
                    f'Action {action.action_type!r} not allowed here. '
                    f'Allowed: {sorted(self._env.allowed_actions)}'
                )
            if action.action_type == 'cli':
                raise ActionError('CLI actions can only be executed through Session.run_cli')
            if (
                timeout is not None
                and self._env.desktop
                and not isinstance(self._backend, SandboxBackend)
            ):
                raise ActionError('Desktop deadline overrides require SandboxBackend')
            call = self._tool_call(action) if action.action_type == 'tool' else None

            self._step += 1
            self._step_verdict = None
            try:
                output, exit_code = self._apply(action, timeout=_remaining(deadline), call=call)
            except ActionError:
                raise
            except BaseException as exc:
                self._end(action, None, exc)
                raise
            done = (
                action.action_type == 'done'
                or (action.action_type != 'tool' and _signals_done(output))
                or (self._env.max_steps is not None and self._step >= self._env.max_steps)
            )
            observation: Observation | None = None
            try:
                observation = self._observe(
                    output, done=done, exit_code=exit_code, timeout=_remaining(deadline)
                )
                reward = self._step_reward(output) if self._env.step_verifier_code else 0.0
            except BaseException as exc:
                # Here an ActionError means the agent's own files blocked the step's output.
                self._end(action, observation, exc)
                raise
            self._record(action, observation, reward=reward, done=done)
            return StepResult(observation, reward, done)

    def run_cli(self, request: CliRequest, on_event: EventObserver) -> CliExecution:
        """Record one real CLI action with nested events, without replaying its commands.

        CLI agents require shell access and SandboxBackend. Completion is explicit:
        stdout DONE lines do not stop supervision. All dgagent processes, including its
        desktop, are killed before returning, so the final observation is text-only.

        Args:
            request: CLI process and opt-in for model-readable provider credentials.
            on_event: Return a stop reason when parsed events exceed the caller's budget.

        Returns:
            Captured stdout, events, exit code, and any budget stop reason.

        Raises:
            DeepGymError: If the session is not running or is closed.
            ActionError: If the environment disallows shell execution or CLI provider egress.
            SandboxError: If the backend cannot isolate or supervise the CLI safely.
        """
        with self._lock:
            self._check_open()
            if self._obs is None or self._done:
                raise DeepGymError('Session is not running; call start() or open a new session')
            if not {'bash', 'cli'} & self._env.allowed_actions:
                raise ActionError('CLI agents require bash or cli in allowed_actions')
            if not self._env.cli_provider_egress:
                raise ActionError(
                    'CLI provider egress is disabled; set cli_provider_egress=True on the '
                    'environment before creating the session'
                )
            if not isinstance(self._backend, SandboxBackend):
                raise SandboxError('CLI agents require an isolated SandboxBackend')
            action = Action(action_type='cli', content=shlex.join(request.argv))
            self._step += 1
            self._step_verdict = None
            started = time.monotonic()
            events: list[dict[str, JsonValue]] = []

            def observe(event: dict[str, JsonValue]) -> str | None:
                events.append(event)
                return on_event(event)

            try:
                result = self._backend.run_cli(request, observe)
                self._cli_finalized = True
                action = action.model_copy(update={'events': result.events})
                observation = Observation(
                    content=result.output,
                    step=self._step,
                    done=True,
                    metadata={
                        'exit_code': result.exit_code,
                        'stop_reason': result.stop_reason,
                        'cli_version': result.cli_version,
                        'version_source': 'sandbox CLI --version',
                        'version_error': result.version_error,
                    },
                )
                reward = self._step_reward(result.output) if self._env.step_verifier_code else 0.0
                self._record(action, observation, reward=reward, done=True)
                return result
            except BaseException as exc:
                action = action.model_copy(update={'events': tuple(events)})
                self._end(action, None, exc)
                raise
            finally:
                self._acting_s += time.monotonic() - started

    def finalize_cli(self) -> None:
        """Stop the CLI episode's agent population even when no CLI was launched.

        The shared backend teardown includes setup descendants. Successful runtime
        execution already performed it; no extra action or reward is recorded here.
        GUI agents never call this method, so their desktops survive normal scoring.
        Already-ended episodes retain their terminal observation, including unknown state.

        Raises:
            DeepGymError: If the session is closed.
            SandboxError: If containment is unavailable or cannot be stopped.
                The stored failure has no verdict and prevents subsequent verification.
        """
        with self._lock:
            self._check_open()
            if self._cli_finalized:
                return
            was_done = self._done
            self._done = True
            try:
                if not isinstance(self._backend, SandboxBackend):
                    raise SandboxError('CLI finalization requires an isolated SandboxBackend')
                self._backend.stop_agent()
            except BaseException as exc:
                self._failed_result = error_result(exc, self.sandbox_id)
                raise
            self._cli_finalized = True
            if not was_done and self._obs is not None:
                self._obs = self._obs.model_copy(update={'done': True})
                self._trajectory.final_observation = self._obs

    def abandon(self, error_type: Literal['provider_error', 'setup_error'], detail: str) -> None:
        """End the episode with no verdict: it stopped through nothing the policy did.

        That is the model provider failing (``provider_error``) or the agent harness never
        starting the model (``setup_error``). :meth:`score` then returns that result
        without running the verifier: whatever state the policy reached, it didn't finish.
        A step that failed first keeps its own result.
        """
        with self._lock:
            if self._failed_result is None:
                self._failed_result = RunResult(
                    score=0.0,
                    passed=False,
                    output=f'Not graded: {detail}',
                    stderr='',
                    exit_code=1,
                    execution_time_ms=0.0,
                    sandbox_id=self.sandbox_id,
                    error_type=error_type,
                )
            self._done = True

    def score(self) -> RunResult:
        """Run the final verifier against the current state.

        An episode that a failed step ended (see :meth:`step`) keeps the result that step
        fixed, with the failure as its ``error_type``; the final verifier doesn't run. A
        sandbox that dies under the final verifier is the agent's doing once it has acted
        (a judged 0); before any action, it has no verdict. So is a verifier that times out
        after reading the agent's output (``solution_timeout``), and a provider lifetime the
        agent's own actions ran out (see :meth:`_ran_out_the_clock`).

        Every tool write sent may still change the app after its response (a queued job, a
        write whose runner was stopped), so the environment's ``root_tool_settle`` settles
        them first. If it can't (none declared, or it fails), the episode gets no verdict
        (``unsettled_write``), whatever the verifier would say, and the verifier doesn't run.

        Raises:
            DeepGymError: If the session is closed.
            VerifierError: If, before it read the agent's output, the verifier timed out or
                its output is not valid protocol JSON or isn't backed by its exit code; or its
                code couldn't be uploaded. After, any of these is the agent's judged 0
                (``solution_error``), but for the agent's code failing to launch.
            TimeoutError: If the verifier timed out and the provider returned nothing.
        """
        with self._lock:
            if self._failed_result is not None:  # fixed already: no sandbox needed
                return self._failed_result
            self._check_open()
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
            if unsettled := self._settle():
                return _unsettled_result(unsettled, self._unsettled, self.sandbox_id)
            start = time.perf_counter()
            acted = bool(self._trajectory.steps)
            try:
                code = self._env.final_verifier_code
                (exit_code, stdout, stderr), consulted = self._verify(code, acted=acted)
            except SandboxDiedError as exc:
                if not acted:
                    return error_result(exc, self.sandbox_id)
                return failure_result(exc, self.sandbox_id)
            except VerifierUploadError as exc:
                if not acted:
                    raise VerifierError(str(exc)) from exc
                return failure_result(exc, self.sandbox_id)
            except Exception as exc:
                if acted and self._ran_out_the_clock(exc):
                    return failure_result(exc, self.sandbox_id, exhausted=True)
                raise
            stderr = stderr.strip()
            elapsed_ms = (time.perf_counter() - start) * 1000
            try:
                verdict = parse_verifier_output(stdout, f'exit_code={exit_code} stderr: {stderr!r}')
                result = build_run_result(
                    verdict, elapsed_ms, self.sandbox_id, stderr=stderr, exit_code=exit_code
                )
            except VerifierError as exc:
                if not consulted or _launch_failed(stdout):
                    raise
                result = _failed_after_reading(exc, elapsed_ms, self.sandbox_id, stderr)
            if consulted and not result.has_verdict and result.error_type != 'sandbox_error':
                detail = VerifierError(f'{result.error_type}: {result.output}')
                return _failed_after_reading(detail, elapsed_ms, self.sandbox_id, stderr)
            return result

    def close(self) -> None:
        """Release the backend, waiting for any in-flight step to finish.

        The session counts as closed from here on, even if the delete fails, so a call
        queued behind this one is refused (see :meth:`_check_open`). Calling close() again
        retries a failed delete.
        """
        with self._lock:
            self._closed = True
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

    def _check_open(self) -> None:
        """Refuse to use the backend after close(): its sandbox may be deleted, and a step
        failing there would be scored as the agent's doing when it was the host's.

        Raises:
            SessionClosedError: If the session is closed.
        """
        if self._closed:
            raise SessionClosedError('Session is closed; open a new session')

    def _apply(
        self,
        action: Action,
        *,
        timeout: float | None = None,
        call: dict[str, JsonValue] | None = None,
    ) -> tuple[str, int | None]:
        """Execute *action*; return its text output and exit code (None if not a command).

        A tool action sends *call*, from :meth:`_tool_call`. Its time, however it ends,
        counts as the agent's (see :meth:`_ran_out_the_clock`).
        """
        bounded = timeout is not None
        timeout = (
            min(timeout, self._env.timeout_per_step)
            if timeout is not None
            else self._env.timeout_per_step
        )
        started = time.monotonic()
        try:
            try:
                match action.action_type:
                    case 'code':
                        exit_code, output = self._run_python('_step.py', action.content, timeout)
                        return output, exit_code
                    case 'bash':
                        exit_code, output = self._backend.exec(action.content, timeout)
                        return output, exit_code
                    case 'write_file':
                        self._backend.write_file(str(action.path), action.content.encode('utf-8'))
                        return f'wrote {action.path}', None
                    case 'tool':
                        assert call is not None
                        return self._send(call, timeout)
                    case 'wait':
                        time.sleep(min(action.amount, timeout))
                        return '', None
                    case 'done' | 'screenshot':
                        return '', None
            except TimeoutError:
                return f'[timed out after {timeout}s]', None
            if bounded and isinstance(self._backend, SandboxBackend):
                self._backend.act(action, timeout=timeout)
            else:
                self._backend.act(action)
            return '', None
        finally:
            self._acting_s += time.monotonic() - started

    def _tool_call(self, action: Action) -> dict[str, JsonValue]:
        """Return the request *action* sends with its declared tool (see HttpTool.prepare).

        Raises:
            ActionError: If no declared tool has its name, or the tool rejects its arguments.
        """
        tool = next((tool for tool in self._env.tools if tool.name == action.tool), None)
        if tool is None:
            names = sorted(tool.name for tool in self._env.tools)
            raise ActionError(f'No tool named {action.tool!r}. Declared: {names}')
        try:
            return tool.prepare(action.arguments)
        except ValueError as exc:
            raise ActionError(f'Tool {tool.name!r} rejected the call: {exc}') from exc

    def _send(self, request: Mapping[str, JsonValue], timeout: float) -> tuple[str, int | None]:
        """Send *request* from the sandbox; return the runner's record of it and its exit code.

        The request reaches the constant runner command as data on its stdin, never through
        a file or argument the agent user could read or change, and every tool's header
        values are redacted from what comes back. It is never retried: a runner the timeout
        stops reports no response, and a mutation it may have sent is recorded as uncertain.

        A mutation counts as unsettled from before it is sent until :meth:`score` settles
        it, since neither a response nor a stopped runner shows the app is done with it;
        only a runner that proves it never connected clears it.
        """
        if request['mutation']:
            self._unsettled.append(request['record'])
        command = run_command(self._backend.python)
        try:
            exit_code, output = self._backend.exec_input(command, json.dumps(request), timeout)
        except TimeoutError:
            exit_code = None
            output = lost_response(request, f'the request was stopped after {timeout}s')
        if request['mutation'] and _never_sent(output):
            self._unsettled.pop()
        return redact(output, self._env.tools), exit_code

    def _settle(self) -> str | None:
        """Settle the tool writes sent with the environment's ``root_tool_settle``; return
        why they stay unsettled, or None once none are.

        Only the environment's own reconciliation settles them: an app may still apply a
        write it answered (queued it, say) or whose runner was stopped, and a model's later
        reads prove nothing.
        """
        if not self._unsettled:
            return None
        command, timeout = self._env.root_tool_settle, self._env.setup_timeout
        if not command:
            return 'the environment declares no root_tool_settle'
        try:
            if isinstance(self._backend, SandboxBackend):
                settled = self._backend.exec_root(
                    command, timeout, offline=True, env={'PATH': TRUSTED_PATH}
                )
            else:
                settled = self._backend.exec(command, timeout)
        except TimeoutError:
            return f'root_tool_settle timed out after {timeout}s'
        except DeepGymError as exc:  # the sandbox failed under it: nothing is proven
            return f'root_tool_settle failed: {type(exc).__name__}'
        if settled.exit_code != 0:
            return f'root_tool_settle exited {settled.exit_code}'
        self._unsettled.clear()
        return None

    def _run_python(self, filename: str, code: str, timeout: float | None = None) -> ExecResult:
        self._backend.write_file(filename, code.encode('utf-8'))
        return self._backend.exec(
            f'{self._backend.python} {filename}', timeout or self._env.timeout_per_step
        )

    def _end(self, action: Action, observation: Observation | None, exc: BaseException) -> None:
        """End the episode after *action*'s step failed with *exc* (see :func:`failure_result`).

        The action may have changed the sandbox, so it is recorded, with no step reward.
        *observation* is the one taken after the action, if any.
        """
        exhausted = self._ran_out_the_clock(exc)
        self._failed_result = failure_result(exc, self.sandbox_id, exhausted=exhausted)
        if observation is not None:
            observation = observation.model_copy(update={'done': True})
        self._record(action, observation, reward=None, done=True)

    def _record(
        self,
        action: Action,
        observation: Observation | None,
        *,
        reward: float | None,
        done: bool,
    ) -> None:
        """Append the step to the trajectory; *observation* becomes the current one.

        A step without a verdict (*reward* None) adds no step reward, never a 0. A failed
        step may have no *observation*; then the episode's final observation is None, not
        the one from before the action.
        """
        self._trajectory.steps.append((self._obs, action))
        if reward is not None:
            self._trajectory.step_rewards.append(reward)
            self._trajectory.total_reward += reward
        self._done = done
        if observation is not None:
            self._obs = observation
        if done:
            self._trajectory.final_observation = observation

    def _observe(
        self, output: str, *, done: bool, exit_code: int | None = None, timeout: float | None = None
    ) -> Observation:
        screenshot = None
        if self._env.desktop:
            if timeout is not None and isinstance(self._backend, SandboxBackend):
                screenshot = self._backend.screenshot(timeout=timeout)
            else:
                screenshot = self._backend.screenshot()
        return Observation(
            content=output,
            step=self._step,
            done=done,
            screenshot=screenshot,
            metadata=None if exit_code is None else {'exit_code': exit_code},
        )

    def _step_reward(self, output: str) -> float:
        """Run the step verifier and return its score in [0, 1].

        Only its stdout counts, so warnings on stderr can't break the plain-float form.

        Raises:
            InvalidScoreError: If the score is NaN, infinite, or outside [0, 1].
            VerifierError: If, before it read the agent's output, the step verifier times
                out, prints neither a float nor JSON with a score, doesn't exit cleanly (see
                :func:`check_exit`), or reports its own failure (exit 2, or an
                ``error_type`` that isn't the model's fault; see
                :func:`~deepgym.models.is_verdict`). After, any of these is the step's judged
                0 (``solution_error``), but for the agent's code failing to launch.
        """
        self._backend.write_file(_STEP_OUTPUT, output.encode('utf-8'))
        try:
            (exit_code, stdout, _), consulted = self._verify(
                self._env.step_verifier_code,
                {'STEP_OUTPUT_PATH': _STEP_OUTPUT, 'STEP_NUMBER': str(self._step)},
            )
        except TimeoutError as exc:
            raise VerifierError(f'Step verifier timed out at step {self._step}') from exc
        try:
            return self._step_score(exit_code, stdout.strip())
        except VerifierError as exc:
            if not consulted or _launch_failed(stdout):
                raise
            details = f"The step verifier failed after reading the agent's output: {exc}"
            verdict = _judged_zero(details, 'solution_error').verifier_result
            self._step_verdict = {**verdict.model_dump(), 'exit_code': 1}
            return 0.0

    def _step_score(self, exit_code: int, stdout: str) -> float:
        """Return the score in the step verifier's *stdout* (see :meth:`_step_reward`)."""
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

    def _verify(
        self, code: str, env: dict[str, str] | None = None, *, acted: bool = True
    ) -> tuple[VerifierOutput, bool]:
        """Run session verifier *code* against the sandbox as it stands.

        It gets a fresh ``DEEPGYM_CONSULTED`` mark in its code (see :func:`_guarded`). It
        prints the mark plus ``:start`` before any of its code runs, and the mark itself once
        the agent's output is in play (it read the agent's work directory, ran a program there
        or on the agent's files, or started the agent's code, see
        :func:`~deepgym.verifier_template.guard_crashes`). Both come as ``\\n<line>\\n`` and
        are cut from its stdout exactly, so what it printed reads as if they weren't there.
        After the mark, a timeout is the agent's judged 0 (``solution_timeout``, as in a single
        run, see :func:`~deepgym.sandbox.solution_timeout`), and so is the verifier being
        killed (``solution_error``). So is a timeout whose output the backend lost (no start
        line), since nothing then shows the verifier hadn't read the agent's output. A judged
        0 comes back as if the verifier had printed it. All of that needs an agent that has
        *acted*: before any action, its work directory holds only what the environment put
        there, so none of these failures is its doing (a :class:`VerifierError`).

        Returns:
            What the verifier printed, and whether it read the agent's output.

        Raises:
            VerifierError: If the verifier timed out before the mark, or failed in any of
                these ways before the agent acted.
        """
        mark = uuid4().hex
        consulted, started = f'\n{mark}\n', f'\n{mark}:start\n'
        timeout = self._env.timeout_per_step
        try:
            exit_code, stdout, stderr = self._backend.verify(_guarded(code, mark), timeout, env)
        except SandboxTimeoutError as exc:
            if not acted:
                raise VerifierError(
                    f'The verifier timed out before the agent acted: {exc}'
                ) from exc
            if consulted in exc.output:
                return _reported(solution_timeout(timeout)), True
            if started not in exc.output:
                details = f'The verifier timed out and nothing it printed came back: {exc}'
                return _reported(_judged_zero(details, 'policy_sandbox_failure')), True
            raise VerifierError(
                f"The verifier timed out before it read the agent's output: {exc}"
            ) from exc
        read = acted and consulted in stdout
        if read and not 0 <= exit_code < 128:
            details = f"The verifier was killed (exit {exit_code}) handling the agent's output"
            return _reported(_judged_zero(details, 'solution_error')), True
        stdout = stdout.replace(consulted, '').replace(started, '')
        return VerifierOutput(exit_code, stdout, stderr), read

    def _ran_out_the_clock(self, exc: BaseException) -> bool:
        """Whether *exc* is a provider lifetime the agent's actions ran out (see
        :func:`ran_out_the_clock`)."""
        return ran_out_the_clock(exc, self._acting_s, time.monotonic() - self._opened)


def run_episode(
    session: Session, agent: Callable[[Observation], Action]
) -> tuple[Trajectory, RunResult]:
    """Drive *session* with *agent* until done, then score it.

    Args:
        session: A fresh, unstarted session. It is closed afterwards.
        agent: Callable mapping an observation to the next action.

    Returns:
        Tuple of (trajectory, final_result). A step that failed after its action may have
        run ends the episode, and the result is the one it fixed (see :meth:`Session.step`).
    """
    with session:
        obs = session.start()
        while not session.done:
            try:
                obs = session.step(agent(obs)).observation
            except Exception:
                if not session.done:  # the agent failed, or its action was rejected
                    raise
        return session.trajectory, session.score()


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Step exhausted the agent wall-time budget')
    return remaining


def ran_out_the_clock(exc: BaseException, acting_s: float, elapsed_s: float) -> bool:
    """Whether *exc* is the provider retiring the sandbox at the end of its lifetime after
    the agent's own actions took at least half of the *elapsed_s* seconds (*acting_s*): then
    the agent ran the clock out (stalling with sleeps or waits, say), not the host.

    ponytail: half is a line, not a measure of fault; an agent that stalls for less needs
    the host to have spent as long (inference, say) for the lifetime to run out.
    """
    return isinstance(exc, SandboxExpiredError) and 2 * acting_s >= elapsed_s


def failure_result(exc: BaseException, sandbox_id: str, *, exhausted: bool = False) -> RunResult:
    """Return the result of an episode that *exc* ended once the agent had acted.

    The rule for sessions, shared with :mod:`deepgym.integrations.verl_tool` and Harbor's
    agent phase. Only a failure that is ours for certain has no verdict:

    * a network or provider failure, or the provider refusing our credentials (see
      :func:`~deepgym.exceptions.is_infrastructure_failure`);
    * a verifier that couldn't run (its launch failed), failed, or timed out
      before it read the agent's output (:class:`~deepgym.exceptions.VerifierError`; after,
      the verifier reports the agent's 0 itself, see :meth:`Session._verify`);
    * the provider retiring the sandbox at the end of its lifetime
      (:class:`~deepgym.exceptions.SandboxExpiredError`), unless *exhausted*: the agent's
      own actions ran the clock out (see :meth:`Session._ran_out_the_clock`);
    * an interrupt.

    Anything else follows from the agent's actions (its sandbox died, its files blocked a
    harness write, a write failed on a disk it fills or a path it controls, a command that
    hung): a judged 0 (``policy_sandbox_failure``), or breaking the sandbox would dodge the
    score. So is a failure the rule can't read.
    """
    if not isinstance(exc, Exception):
        return error_result(exc, sandbox_id)
    try:
        ours = isinstance(exc, VerifierError | SandboxExpiredError) and not exhausted
        ours = ours or is_infrastructure_failure(exc)
        result = error_result(exc, sandbox_id)
    except Exception:  # an exception that breaks the rule is never a reason to mask it
        ours = False
        result = error_result(SandboxError(type(exc).__name__), sandbox_id)
    if ours:
        return result
    return result.model_copy(update={'error_type': 'policy_sandbox_failure'})


def _joined(output: VerifierOutput) -> ExecResult:
    """Return a command's *output* as one text: stdout, then stderr."""
    exit_code, stdout, stderr = output
    return ExecResult(exit_code, stdout + ('\n' + stderr if stderr else ''))


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


# Puts deepgym_solution on a session verifier's path where the backend ships it.
_RUNTIME_ON_PATH = """\
import os as _dg_os
import sys as _dg_sys

if 'DEEPGYM_RUNTIME' in _dg_os.environ:
    _dg_sys.path.insert(0, _dg_os.path.dirname(_dg_os.environ['DEEPGYM_RUNTIME']))
"""


def _guarded(code: str, mark: str) -> str:
    """Return session verifier *code* with deepgym_solution on its path and *mark* as its
    DEEPGYM_CONSULTED, exiting 2 on a crash (see
    :func:`~deepgym.verifier_template.guard_crashes`) so it never counts as a failed run.

    The mark goes in the code, which only the verifier user reads, not in a variable on a
    command line any process in the sandbox can list.
    """
    prelude = (
        f'{_RUNTIME_ON_PATH}_dg_os.environ["DEEPGYM_CONSULTED"] = {mark!r}\n'
        f'_dg_os.write(1, b"\\n{mark}:start\\n")\n'
    )
    return guard_crashes(code, prelude)


def _launch_failed(stdout: str) -> bool:
    """Whether a verifier's last protocol line reports that the agent's code couldn't be
    launched (the crash guard's ``sandbox_error``): the machine's failure, even after the
    verifier read the agent's output."""
    try:
        report = json.loads(_extract_json_line(stdout) or 'null')
    except ValueError:
        return False
    return isinstance(report, dict) and report.get('error_type') == 'sandbox_error'


def _failed_after_reading(
    exc: BaseException, elapsed_ms: float, sandbox_id: str, stderr: str
) -> RunResult:
    """Return the judged 0 of a verifier that failed after reading the agent's output."""
    details = f"The verifier failed after reading the agent's output: {exc}"
    verdict = _judged_zero(details, 'solution_error').verifier_result
    return build_run_result(verdict, elapsed_ms, sandbox_id, stderr=stderr, exit_code=1)


def _unsettled_result(why: str, writes: Sequence[JsonValue], sandbox_id: str) -> RunResult:
    """Return the result of an episode whose tool *writes* the app may still apply: no
    verdict, whatever the score would be, since any write could still raise or lower it. A
    policy that learns to leave writes unsettled shows in the masked rate (see
    :mod:`deepgym.integrations.reward`)."""
    listed = json.dumps(writes, ensure_ascii=False)
    return RunResult(
        score=0.0,
        passed=False,
        output=f'Not graded, tool writes unsettled ({why}): {listed}',
        stderr='',
        exit_code=1,
        execution_time_ms=0.0,
        sandbox_id=sandbox_id,
        error_type='unsettled_write',
    )


def _never_sent(output: str) -> bool:
    """Whether a tool runner's record shows its request never reached the app."""
    try:
        return json.loads(output).get('uncertain') is False
    except (ValueError, AttributeError):
        return False


def _reported(executed: ExecutionResult) -> VerifierOutput:
    """Return *executed*, a judged 0, as the verifier would have printed it."""
    return VerifierOutput(executed.exit_code, executed.verifier_result.model_dump_json(), '')


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
