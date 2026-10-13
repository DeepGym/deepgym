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
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
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
)
from deepgym.sandbox import (
    _extract_json_line,
    build_run_result,
    check_exit,
    parse_verifier_output,
)
from deepgym.sandboxes import ExecResult, Sandbox

if TYPE_CHECKING:
    from deepgym.sandboxes.daytona import DaytonaDesktop

logger = logging.getLogger(__name__)

_SANDBOX_WORKDIR = '/tmp/deepgym'
# Unprivileged user that runs sandbox session commands, file writes and the desktop.
_AGENT_USER = 'dgagent'
_AGENT_HOME = f'/home/{_AGENT_USER}'
# Separate unprivileged user for verifiers, so the agent can't read, signal or write to them.
_VERIFIER_USER = 'dgverifier'
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
        SANDBOX_ROOT, _STAGING_DIR, _VERIFIER_DIR, _NETNS,
    )
)  # fmt: skip
# Seconds to take over, pack or unpack a work directory (it may be large).
_TRANSFER_TIMEOUT_S = 600
# Largest gzipped work directory archive_workdir hands back; it is held in memory.
_MAX_ARCHIVE_BYTES = 256 << 20
_DISPLAY = ':0'
_STEP_OUTPUT = '_step_output.txt'


class StepResult(NamedTuple):
    """Outcome of one agent action."""

    observation: Observation
    reward: float
    done: bool


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

    def verify(
        self, code: str, timeout: int, env: dict[str, str] | None = None, *, quiet: bool = False
    ) -> ExecResult:
        """Run verifier *code* in the work directory, out of the agent's reach where possible.

        Where the backend can isolate code from the verifier, it sets ``DEEPGYM_RUNTIME``
        and the other variables ``deepgym_solution`` needs. *quiet* discards stderr.
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
        """Run *command* with bash in the temp directory.

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
        return ExecResult(proc.returncode, stdout + ('\n' + stderr if stderr else ''))

    def write_file(self, path: str, content: bytes) -> None:
        """Write *content* to *path* inside the temp directory."""
        target = self._dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def verify(
        self, code: str, timeout: int, env: dict[str, str] | None = None, *, quiet: bool = False
    ) -> ExecResult:
        """Run verifier *code* inline; it shares the agent's user here.

        Code it loads through ``deepgym_solution`` runs in the local judge's sandbox.
        """
        runtime = {
            'DEEPGYM_ISOLATION': json.dumps(local_isolation([str(self._runtime.parent)])),
            'DEEPGYM_RUNTIME': str(self._runtime),
            'DEEPGYM_PYTHON': sys.executable,
        }
        return self.exec(
            _inline_python(self.python, code, quiet=quiet), timeout, {**runtime, **(env or {})}
        )

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


# Lets the verifier user start code as the solver user (deepgym_solution), and nothing else.
_SUDOERS_RULE = f'{_VERIFIER_USER} ALL=({SOLVER_USER}) NOPASSWD: ALL'


class SandboxBackend:
    """Run actions in a throwaway sandbox from any provider (see :mod:`deepgym.sandboxes`).

    Agent commands, file writes and the desktop (given one) run as one unprivileged
    user and verifiers as another, both with every capability dropped: root owns the
    binaries every provider exec runs with full privileges, so the agent must never be
    root, and a different user means it can't read, signal or write into a verifier.
    The agent also gets no-new-privs and an empty bounding set, so no setuid binary
    lifts it. Where sudo exists, the verifier keeps it, limited to starting the code it
    loads through ``deepgym_solution`` as a third user that can't touch the verifier;
    elsewhere the verifier is locked down like the agent and ``deepgym_solution`` is
    unavailable. When the provider doesn't block the network itself, everything joins
    one loopback-only network namespace, so servers started during setup stay reachable
    across steps while the internet and the host are not.

    The agent owns the work directory, *workdir* (``/tmp/deepgym`` unless given), and
    everything already in it.
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
        self._workdir = workdir
        quoted = shlex.quote(workdir)
        try:
            try:
                check_workdir(workdir)
            except ValueError as exc:
                raise SandboxError(str(exc)) from exc
            users = sandbox.exec(
                "command -v setpriv >/dev/null || { echo 'setpriv missing'; exit 1; } && "
                f'useradd --system --create-home --home-dir {_AGENT_HOME} '
                f'--shell /usr/sbin/nologin {_AGENT_USER} && chmod 755 {_AGENT_HOME} && '
                f'useradd --system --no-create-home --shell /usr/sbin/nologin {_VERIFIER_USER} && '
                f'mkdir -p {quoted} && chown -R {_AGENT_USER}: {quoted} && '
                f'mkdir -p -m 700 {_STAGING_DIR} && mkdir -p -m 750 {_VERIFIER_DIR} && '
                f'chgrp {_VERIFIER_USER} {_VERIFIER_DIR} && solver=0 && '
                'if command -v sudo >/dev/null && [ -d /etc/sudoers.d ]; then '
                f'useradd --system --no-create-home --shell /usr/sbin/nologin {SOLVER_USER} && '
                f'echo {shlex.quote(_SUDOERS_RULE)} > /etc/sudoers.d/deepgym && '
                f'chmod 440 /etc/sudoers.d/deepgym && mkdir -p -m 755 {_RUNTIME_DIR} && solver=1; '
                f'fi && echo "$(id -u {_AGENT_USER}) $(id -g {_AGENT_USER}) '
                f'$(id -u {_VERIFIER_USER}) $(id -g {_VERIFIER_USER}) $solver"',
                timeout=_TRANSFER_TIMEOUT_S,
            )
            if users.exit_code != 0:
                raise SandboxError(f'Could not create the session users: {users.output[-500:]}')
            agent_uid, agent_gid, verifier_uid, verifier_gid, solver = users.output.split()[-5:]
            self._as_agent = _drop_to(agent_uid, agent_gid)
            self._as_verifier = _drop_to(verifier_uid, verifier_gid, allow_sudo=solver == '1')
            self._env = {'HOME': _AGENT_HOME}
            self._runtime_env: dict[str, str] = {}
            if solver == '1':
                runtime = f'{_RUNTIME_DIR}/{RUNTIME_FILE}'
                sandbox.upload(RUNTIME_SOURCE.encode('utf-8'), runtime)
                prefix = ['sudo', '-n', '-u', SOLVER_USER, '--']
                self._runtime_env = {
                    'DEEPGYM_ISOLATION': json.dumps({'kind': 'user', 'prefix': prefix}),
                    'DEEPGYM_RUNTIME': runtime,
                    'DEEPGYM_PYTHON': self.python,
                }
            self._offline = ''
            if not sandbox.network_blocked:
                self._make_offline_namespace()
                self._offline = f'nsenter --net={_NETNS} -- '
        except BaseException:
            self.close()
            raise

    def _make_offline_namespace(self) -> None:
        """Make a loopback-only network namespace and bind-mount it at ``_NETNS``.

        The holder process brings loopback up, then execs ``sleep``; waiting for that
        proves the namespace exists and is ready. The bind mount keeps it alive and
        addressable without trusting a PID that could be reused.
        """
        made = self._sandbox.exec(
            f'setsid unshare --net -- python3 -c {shlex.quote(_LOOPBACK_UP_THEN_SLEEP)} '
            '>/dev/null 2>&1 & pid=$!; '
            'for _ in $(seq 100); do '
            '[ "$(cat /proc/$pid/comm 2>/dev/null)" = sleep ] && break; sleep 0.05; done; '
            '[ "$(cat /proc/$pid/comm 2>/dev/null)" = sleep ] && '
            '[ "$(readlink /proc/$pid/ns/net)" != "$(readlink /proc/1/ns/net)" ] && '
            f'touch {_NETNS} && mount --bind /proc/$pid/ns/net {_NETNS} && '
            f'nsenter --net={_NETNS} true',
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
            env={**self._env, **(env or {})},
            timeout=timeout,
        )

    def exec_root(self, command: str, timeout: int, *, offline: bool) -> ExecResult:
        """Run trusted *command* with bash as root in the work directory.

        Only for a sandbox no agent has acted in, such as a grader's fresh one. *offline*
        joins the loopback-only namespace when the provider doesn't block the network.

        Raises:
            TimeoutError: If the command exceeds *timeout* seconds.
        """
        prefix = self._offline if offline else ''
        return self._sandbox.exec(
            f'{prefix}bash -c {shlex.quote(command)}', cwd=self._workdir, timeout=timeout
        )

    def archive_workdir(self) -> bytes:
        """Kill every agent process, then return the work directory as a gzipped tar.

        Root packs it once nothing of the agent's is left running, so nothing changes
        mid-archive. Symlinks are stored as links, never followed.

        Raises:
            SandboxError: If packing fails.
        """
        staged = f'{_STAGING_DIR}/{uuid4().hex}.tgz'
        # ponytail: the archive comes back through exec output, so it is held in memory
        # here; a provider download() would stream it if work directories outgrow the cap.
        packed = self._sandbox.exec(
            f"{self._as_agent} sh -c 'kill -9 -1' >/dev/null 2>&1; "
            f'tar -C {shlex.quote(self._workdir)} -czf {staged} . && '
            f'size=$(wc -c < {staged}) && if [ "$size" -gt {_MAX_ARCHIVE_BYTES} ]; then '
            f'echo "the archive is $size bytes, over {_MAX_ARCHIVE_BYTES}"; '
            f'rm -f {staged}; exit 1; fi',
            timeout=_TRANSFER_TIMEOUT_S,
        )
        if packed.exit_code != 0:
            raise SandboxError(f'Failed to archive {self._workdir}: {packed.output[-500:]}')
        encoded = self._sandbox.exec(
            f'base64 {staged}; status=$?; rm -f {staged}; exit $status',
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
        ownership (git) trust them; links are changed, never followed.

        Raises:
            SandboxError: If unpacking fails.
        """
        staged = f'{_STAGING_DIR}/{uuid4().hex}.tgz'
        self._sandbox.upload(archive, staged)
        quoted = shlex.quote(self._workdir)
        done = self._sandbox.exec(
            f'rm -rf {quoted} && mkdir -p {quoted} && chown {_AGENT_USER}: {quoted} && '
            f'{self._as_agent} tar -xzpf - -C {quoted} < {staged} && chown -hR 0:0 {quoted}; '
            f'status=$?; rm -f {staged}; exit $status',
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
            f'{self._as_agent} sh -c {shlex.quote(write)} write {target} < {staged}; '
            f'status=$?; rm -f {staged}; exit $status',
            timeout=60,
        )
        if done.exit_code != 0:
            raise SandboxError(f'Failed to write {path}: {done.output[-500:]}')

    def verify(
        self, code: str, timeout: int, env: dict[str, str] | None = None, *, quiet: bool = False
    ) -> ExecResult:
        """Run verifier *code* as the verifier user, from a file only that user can read.

        Raises:
            TimeoutError: If the verifier exceeds *timeout* seconds.
        """
        script = f'{_VERIFIER_DIR}/{uuid4().hex}.py'
        self._sandbox.upload(code.encode('utf-8'), script)
        stderr = ' 2>/dev/null' if quiet else ''
        return self._sandbox.exec(
            f'{self._offline}{self._as_verifier} {self.python} -I {script}{stderr}; '
            f'status=$?; rm -f {script}; exit $status',
            cwd=self._workdir,
            env={**self._env, **self._runtime_env, **(env or {})},
            timeout=timeout,
        )

    def start_desktop(self) -> None:
        """Start the desktop with its session and apps running as the agent, offline.

        Raises:
            DeepGymError: If this provider has no desktop.
            SandboxError: If the desktop doesn't start.
        """
        self._need_desktop('Desktop environments').start(
            f'{self._offline}{self._as_agent}', _AGENT_HOME
        )
        self._env['DISPLAY'] = _DISPLAY

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
        """Delete the sandbox."""
        self._sandbox.delete()

    def _need_desktop(self, what: str) -> DaytonaDesktop:
        """Return the desktop, or refuse *what* if this provider has none."""
        if self._desktop is None:
            raise DeepGymError(f'{what} need the Daytona backend')
        return self._desktop


def check_workdir(workdir: str, reserved: Sequence[str] = ()) -> str:
    """Return *workdir* if an agent may own it.

    Args:
        workdir: The folder the agent will own, along with everything in it.
        reserved: More folders it must neither hold nor sit in.

    Raises:
        ValueError: If it isn't absolute, has ``..``, holds /tmp, or holds or sits in a
            folder root trusts (/usr, /etc, /root, ...), DeepGym's own, or *reserved*.
    """
    path = PurePosixPath(workdir)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Work directory {workdir!r} must be an absolute path without ".."')
    if PurePosixPath('/tmp').is_relative_to(path):
        raise ValueError(f'Work directory {workdir!r} would hold /tmp')
    for other in (*_PROTECTED_PATHS, *map(PurePosixPath, reserved)):
        if other.is_relative_to(path) or path.is_relative_to(other):
            raise ValueError(f'Work directory {workdir!r} overlaps {other}')
    return workdir


def _drop_to(uid: str, gid: str, *, allow_sudo: bool = False) -> str:
    """Command prefix that drops to *uid*/*gid* with no capabilities.

    It also sets no-new-privs and empties the bounding set so no setuid binary can
    regain privileges, unless *allow_sudo*: sudo needs both, and its rules then decide.
    """
    lockdown = [] if allow_sudo else ['--bounding-set=-all', '--no-new-privs']
    return shlex.join([
        'setpriv', f'--reuid={uid}', f'--regid={gid}', '--clear-groups', '--inh-caps=-all',
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
            VerifierError: If the step verifier fails; the episode is then over.
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
            except VerifierError:
                self._record(action, output, exit_code, reward=0.0, done=True)
                raise
            done = (
                action.action_type == 'done'
                or _signals_done(output)
                or self._step >= self._env.max_steps
            )
            return self._record(action, output, exit_code, reward=reward, done=done)

    def score(self) -> RunResult:
        """Run the final verifier against the current state.

        Raises:
            VerifierError: If the verifier output is not valid protocol JSON, or the
                verifier crashed or was killed after printing it.
            TimeoutError: If the verifier exceeds the per-step timeout.
        """
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
        with self._lock:
            start = time.perf_counter()
            exit_code, stdout = self._backend.verify(
                _guarded(self._env.final_verifier_code), self._env.timeout_per_step
            )
            result = parse_verifier_output(stdout, f'exit_code={exit_code}')
            elapsed_ms = (time.perf_counter() - start) * 1000
            return build_run_result(result, elapsed_ms, self.sandbox_id, exit_code=exit_code)

    def close(self) -> None:
        """Release the backend, waiting for any in-flight step to finish."""
        with self._lock:
            self._backend.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

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
        self, action: Action, output: str, exit_code: int | None, *, reward: float, done: bool
    ) -> StepResult:
        """Append the step to the trajectory and advance the current observation."""
        self._trajectory.steps.append((self._obs, action))
        self._trajectory.step_rewards.append(reward)
        self._trajectory.total_reward += reward
        self._done = done
        self._obs = self._observe(output, done=done, exit_code=exit_code)
        if done:
            self._trajectory.final_observation = self._obs
        return StepResult(self._obs, reward, done)

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

        Stderr is discarded so warnings can't break the plain-float form.

        Raises:
            InvalidScoreError: If the score is NaN, infinite, or outside [0, 1].
            VerifierError: If the step verifier times out, prints neither a float nor
                JSON with a score, or doesn't exit cleanly (see :func:`check_exit`).
        """
        self._step_verdict = None
        self._backend.write_file(_STEP_OUTPUT, output.encode('utf-8'))
        try:
            exit_code, stdout = self._backend.verify(
                _guarded(self._env.step_verifier_code),
                self._env.timeout_per_step,
                env={'STEP_OUTPUT_PATH': _STEP_OUTPUT, 'STEP_NUMBER': str(self._step)},
                quiet=True,
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


def _inline_python(python: str, code: str, *, quiet: bool = False) -> str:
    """Build a command that runs *code* isolated (-I) without writing it to disk.

    Verifiers never touch the sandbox filesystem, so an agent process can't swap
    the file between upload and execution. *quiet* discards stderr.
    """
    encoded = base64.b64encode(code.encode('utf-8')).decode('ascii')
    loader = f"import base64;exec(compile(base64.b64decode('{encoded}'),'<verifier>','exec'))"
    return f'{python} -I -c "{loader}"' + (' 2>/dev/null' if quiet else '')


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
