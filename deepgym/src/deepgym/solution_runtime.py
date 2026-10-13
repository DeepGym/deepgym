"""Run untrusted solution code in its own isolated process, away from the judge.

Verifiers (the judge) use this instead of importing or running the solution
themselves::

    from deepgym_solution import load_solution, run_solution

    solution = load_solution(solution_path)    # module proxy; every call runs remotely
    result = solution.two_sum([2, 7, 11], 9)    # per-call timeout, plain data only

    proc = run_solution(solution_path, cwd=work_dir, env={'OUT': work_dir})  # as a script

    # As a client of a server the judge runs on 127.0.0.1:port; no other network.
    proc = run_solution(solution_path, [f'http://127.0.0.1:{port}/'], loopback_port=port)

    # A repo task: apply the model's diff to fixture files in memory, then import the result.
    files = apply_patch(solution_source(solution_path), {'calc.py': fixture_text})
    solution = load_files(files, 'calc.py')

The solution process can't read the judge's files or memory, so it can't see
expected answers, patch the verifier, or print a fake score. Only plain data
crosses the boundary (None, bool, int, float, str, bytes, list, tuple, set,
frozenset, dict). Any other object comes back as an opaque handle, so a value
that claims to equal everything can't fake a pass.

Each call is bounded by DEEPGYM_CASE_TIMEOUT and by the run's overall
DEEPGYM_DEADLINE, so a slow case fails on its own instead of sinking the run.

Every outcome of a solution call gets exactly one classification:

* Solution fault, a judged 0: the solution raises, exits (any code, or early), is
  signalled or killed, times out, runs out of memory, returns invalid or unparseable
  output, breaks the protocol or closes its pipe, or exhausts a resource (disk, inodes,
  processes, file descriptors), even when that makes the verifier's own later reads or
  writes fail. Each reaches the verifier as a :class:`SolutionError`; uncaught, the
  crash guard scores it 0 (``error_type`` solution_timeout for a timeout).
* Infra, no verdict: the preflight before any solution code runs fails (too little free
  disk or inodes, an isolation launcher that can't start code), or the runtime can't
  launch or hand-shake before then. That raises :class:`LaunchError`, which a verifier
  lets propagate (it isn't an ``Exception``); the crash guard then ends the run with no
  verdict. Provider and transport failures, and the verifier's own setup failing before
  it starts the solution, are no verdict outside this file.
* Anything else, once solution code has run (see :func:`consulted`): solution fault.

So a solution can't dodge its score by breaking the verifier. Load test data before
the solution: a verifier that fails after starting it scores 0.

Stdlib only: this file is copied next to the verifier as ``deepgym_solution.py``
and also runs as the solution-side server (``python -I deepgym_solution.py --serve``),
loopback forwarder (``--forward``), judge confiner (``--confine``) and solution watcher
(``--watch``), which kills what a judge started if the judge dies first.
"""

from __future__ import annotations

import atexit
import base64
import builtins
import contextlib
import functools
import importlib.util
import json
import math
import os
import queue
import re
import selectors
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

_PATH = '/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin'
# Huge ints (e.g. EvalPlus inputs) must survive json round trips on both sides.
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
_MAX_DEPTH = 100
_MAX_MESSAGE = 64 * 1024 * 1024
_LOAD_TIMEOUT_S = 30.0
_READY = '{"ready": true}\n'
"""The server's first line, written before it touches the solution."""
_CONFINE_GRACE_S = 3
"""Seconds past the deadline a confined judge gets; the host waits 5 (DEADLINE_GRACE_S)."""


class SolutionError(Exception):
    """Raise when the solution failed: it raised, exited, hung, died or broke the protocol.

    Every failure that comes from the solution process is one, so a verifier can tell it
    from its own failures and the sandbox's. A builtin exception the solution raised keeps
    its type too (``except ValueError`` still works), as does a timeout (``TimeoutError``,
    or ``subprocess.TimeoutExpired`` from :func:`run_solution`).
    """


class SolutionTimeoutError(SolutionError):
    """Raise when the solution runs past its time limit: a judged 0, never a verifier error.

    Also a ``TimeoutError``, or ``subprocess.TimeoutExpired`` from :func:`run_solution`.
    """


@functools.cache
def _marked(kind: type[Exception], base: type[SolutionError] = SolutionError) -> type[Exception]:
    """Return *kind* as a *base* too (see :class:`SolutionError`)."""
    return type(kind.__name__, (kind, base), {'__module__': kind.__module__})


class LaunchError(BaseException):
    """Raise when the judge can't start the solution's process (out of processes, say)
    before any solution code has run.

    The machine failed, not the solution: a verifier lets this propagate, so the run ends
    as a verifier error with no verdict instead of scoring the solution 0. It isn't an
    ``Exception``, so a verifier's ``except Exception`` can't turn it into a score. Once
    solution code has run, the same failure is a :class:`SolutionError` (see the module
    docstring).
    """


# ---------------------------------------------------------------------------
# Plain-data encoding shared by both sides
# ---------------------------------------------------------------------------


_PLAIN = (bool, int, float, str, bytes, list, tuple, set, frozenset, dict)


def _plain(obj: Any) -> Any:
    """Turn builtin subclasses (Counter, IntEnum, np.float64...) and numpy scalars into
    their plain base type, so overridden methods like __eq__ never reach the judge."""
    if type(obj) in _PLAIN or obj is None:
        return obj
    for base in _PLAIN:
        if isinstance(obj, base):
            return base(obj)
    if type(obj).__module__ == 'numpy' and getattr(obj, 'ndim', None) == 0:
        return obj.item()
    return obj


def _encode(obj: Any, handles: dict[int, Any] | None, depth: int = 0) -> Any:
    """Encode *obj* as JSON-safe tagged data; unknown objects become handles or errors."""
    if depth > _MAX_DEPTH:
        raise ValueError('value is nested too deeply to send across the solution boundary')
    obj = _plain(obj)
    kind = type(obj)
    if obj is None or kind in (bool, int, str):
        return obj
    if kind is float:
        return obj if math.isfinite(obj) else {'$f': repr(obj)}
    if kind is list:
        return [_encode(item, handles, depth + 1) for item in obj]
    if kind in (tuple, set, frozenset):
        tag = {tuple: '$t', set: '$s', frozenset: '$fs'}[kind]
        return {tag: [_encode(item, handles, depth + 1) for item in obj]}
    if kind is dict:
        return {
            '$d': [
                [_encode(k, handles, depth + 1), _encode(v, handles, depth + 1)]
                for k, v in obj.items()
            ]
        }
    if kind is bytes:
        return {'$b': base64.b64encode(obj).decode('ascii')}
    if isinstance(obj, _RemoteObject):
        return {'$h': obj._handle}
    if handles is None:
        raise TypeError(f'cannot send {kind.__name__} to the solution; use plain data')
    handle = len(handles) + 1
    handles[handle] = obj
    return {'$h': handle, 'type': kind.__name__}


def _decode(data: Any, resolve: Callable[[dict], Any]) -> Any:
    """Rebuild plain Python data from the tagged encoding; *resolve* maps handle tags."""
    if isinstance(data, list):
        return [_decode(item, resolve) for item in data]
    if not isinstance(data, dict):
        return data
    if '$f' in data:
        return float(data['$f'])
    if '$t' in data:
        return tuple(_decode(item, resolve) for item in data['$t'])
    if '$s' in data:
        return {_decode(item, resolve) for item in data['$s']}
    if '$fs' in data:
        return frozenset(_decode(item, resolve) for item in data['$fs'])
    if '$d' in data:
        return {_decode(k, resolve): _decode(v, resolve) for k, v in data['$d']}
    if '$b' in data:
        return base64.b64decode(data['$b'])
    if '$h' in data:
        return resolve(data)
    raise ValueError(f'unknown value tag: {sorted(data)}')


# ---------------------------------------------------------------------------
# Launching code in the isolated domain
# ---------------------------------------------------------------------------


def _isolation() -> dict:
    return json.loads(os.environ.get('DEEPGYM_ISOLATION', '{"kind": "none"}'))


def _python() -> str:
    return os.environ.get('DEEPGYM_PYTHON', sys.executable)


def _runtime() -> str:
    """Path of this file as the isolated solution can read it."""
    return os.environ.get('DEEPGYM_RUNTIME', os.path.abspath(__file__))


def _macos_profile(
    spec: dict, writable: Sequence[str], allow_fork: bool, loopback_port: int | None
) -> str:
    def paths(items: Sequence[str]) -> str:
        return ' '.join(f'(subpath {json.dumps(os.path.realpath(p))})' for p in items) or (
            '(literal "/nonexistent")'
        )

    hide, expose, rehide = spec['hide'], spec.get('expose', []), spec.get('rehide', [])
    loopback = (
        f'(allow network-outbound (remote ip "localhost:{int(loopback_port)}"))'
        if loopback_port is not None
        else ''
    )

    # Later rules win: hide whole trees, then re-expose this run's own directories.
    # Hiding denies reading contents and listings, not stat, so tools that resolve
    # their working path (git) still work. No Mach IPC or AppleEvents: they let a
    # process ask launchd or another app to start something outside the sandbox.
    return (
        f'(version 1)(allow default)(deny network*){loopback}(deny mach-lookup)'
        '(deny appleevent-send)'
        f'(deny file-read-data {paths(hide)})'
        f'(allow file-read-data {paths([*expose, *writable])})'
        f'(deny file-read-data {paths(rehide)})'
        f'(deny file-write*)(allow file-write* {paths(writable)} (literal "/dev/null"))'
        # Without a separate user there is no reliable way to kill a child that
        # calls setsid(), so no forking unless the caller needs subprocesses.
        + ('' if allow_fork else '(deny process-fork)')
    )


def _share_with_solution(workdir: str) -> None:
    """Let the solution user write in *workdir* and read what the judge put there.

    Works through file descriptors opened with O_NOFOLLOW, so a symlink swapped
    in mid-walk is refused instead of followed to a judge-owned file.
    """
    no_follow = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    top = os.open(workdir, no_follow | os.O_DIRECTORY)
    try:
        os.fchmod(top, 0o777)
    finally:
        os.close(top)
    for _, dirs, files, dir_fd in os.fwalk(workdir, follow_symlinks=False):
        for name, flags, mode in [
            *((d, no_follow | os.O_DIRECTORY, 0o777) for d in dirs),
            *((f, no_follow, None) for f in files),
        ]:
            try:
                fd = os.open(name, flags, dir_fd=dir_fd)
            except OSError:
                continue
            try:
                os.fchmod(fd, mode if mode is not None else os.fstat(fd).st_mode | 0o644)
            finally:
                os.close(fd)


def _launch_argv(
    command: Sequence[str],
    workdir: str,
    extra_env: dict[str, str] | None = None,
    allow_fork: bool = False,
    loopback_port: int | None = None,
) -> list[str]:
    """Wrap *command* to run as the isolated solution: empty env, writes only in *workdir*.

    *loopback_port* only matters on macOS, where the solution shares the judge's
    network stack; elsewhere :func:`_loopback` forwards the port. A ``none`` spec with
    ``no_fork`` (inside a macOS-confined judge, see :func:`_confine`) lets the command
    start no processes at all, *allow_fork* or not: nothing there could reap one that
    detaches.
    """
    spec = _isolation()
    writable = [workdir]
    clean = ['env', '-i', f'PATH={_PATH}', f'HOME={workdir}', f'TMPDIR={workdir}']
    clean += ['PYTHONDONTWRITEBYTECODE=1', *(f'{k}={v}' for k, v in (extra_env or {}).items())]
    kind = spec['kind']
    if kind == 'user':
        _share_with_solution(workdir)
        return [*spec['prefix'], *clean, *command]
    if kind == 'macos':
        profile = _macos_profile(spec, writable, allow_fork, loopback_port)
        return ['sandbox-exec', '-p', profile, *clean, *command]
    if kind == 'bwrap':
        argv = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc']
        for path in spec['hide']:
            argv += ['--tmpfs', path]
        for path in spec.get('expose', []):
            argv += ['--ro-bind', path, path]
        for path in spec.get('rehide', []):
            argv += ['--tmpfs', path]
        for path in writable:
            argv += ['--bind', path, path]
        argv += ['--unshare-all', '--die-with-parent', '--new-session', '--chdir', workdir]
        return [*argv, *clean, *command]
    if spec.get('no_fork'):
        return [*clean, _python(), '-I', '-S', '-c', _NO_FORK, *command]
    return [*clean, *command]


def _relay(listener: socket.socket, connect: Callable[[], socket.socket]) -> None:
    """In the background, pipe each connection accepted on *listener* to a fresh connect().

    Stops accepting once *listener* is closed.
    """

    def pipe(source: socket.socket, sink: socket.socket) -> None:
        with contextlib.suppress(OSError):
            while data := source.recv(65536):
                sink.sendall(data)
        with contextlib.suppress(OSError):
            sink.shutdown(socket.SHUT_WR)

    def splice(client: socket.socket) -> None:
        with contextlib.suppress(OSError), client, connect() as upstream:
            back = threading.Thread(target=pipe, args=(upstream, client), daemon=True)
            back.start()
            pipe(client, upstream)
            back.join()

    def accept() -> None:
        with contextlib.suppress(OSError):
            while True:
                client, _ = listener.accept()
                threading.Thread(target=splice, args=(client,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()


# Run as a prefix: the command (argv[1:]) and whatever it runs may start no processes.
_NO_FORK = (
    'import os, resource, sys; resource.setrlimit(resource.RLIMIT_NPROC, (0, 0)); '
    'os.execvp(sys.argv[1], sys.argv[1:])'
)

# Run as a prefix: wait for the judge's go on fd argv[1] (none if the judge died first: run
# nothing), say on fd argv[2] that the solution's Python is up, close both so the script
# never holds them, then exec the rest.
_STARTED = (
    'import os, sys; go, fd = int(sys.argv[1]), int(sys.argv[2]); '
    'os.read(go, 1) or os._exit(1); os.close(go); os.write(fd, b"1"); os.close(fd); '
    'os.execvp(sys.argv[3], sys.argv[3:])'
)

# Run in a child: bind the inherited socket (argv[1]) at argv[2] with umask 0.
_BIND_ANY_USER = (
    'import os, socket, sys; os.umask(0); socket.socket(fileno=int(sys.argv[1])).bind(sys.argv[2])'
)


@contextlib.contextmanager
def _loopback(port: int | None) -> Iterator[list[str]]:
    """Yield a command prefix that gives the solution the judge's 127.0.0.1:*port*.

    Where the solution has its own network namespace (bwrap, or a sandbox user
    with an offline namespace), the judge's loopback isn't the solution's. The
    prefix starts a forwarder there that listens on 127.0.0.1:*port* and pipes
    each connection through a Unix socket to the judge's port. Elsewhere the port
    is reachable as it is (macOS allows it in the profile), so the prefix is empty.

    Each call gets its own randomly named directory in the runtime directory, not in
    the solution-writable workdir: the judge owns it, bwrap mounts it read-only and a
    sandbox user may enter it, but the solution can't predict, create or replace
    anything there, so concurrent calls sharing a workdir can't interfere. Modes are
    set at creation or through a file descriptor, never by path, and the judge's own
    umask never changes.
    """
    spec = _isolation()
    if port is None or not (spec['kind'] == 'bwrap' or spec.get('own_network')):
        yield []
        return
    socket_dir = tempfile.mkdtemp(dir=os.path.dirname(_runtime()))
    fd = os.open(socket_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(fd, 0o711)  # the solution may enter it, not list or write it
    finally:
        os.close(fd)
    path = os.path.join(socket_dir, 'loopback.sock')
    try:
        with socket.socket(socket.AF_UNIX) as listener:
            # Connecting needs write permission on the socket and the solution may be
            # another user, so it must be created 0o777. umask is process-wide, so a child
            # binds the shared socket with umask 0 rather than the judge loosening its own.
            fileno = listener.fileno()
            subprocess.run(
                [sys.executable, '-I', '-S', '-c', _BIND_ANY_USER, str(fileno), path],
                pass_fds=(fileno,),
                capture_output=True,
                check=True,
            )
            listener.listen()
            _relay(listener, lambda: socket.create_connection(('127.0.0.1', port)))
            yield [_python(), '-I', '-B', _runtime(), '--forward', str(port), path]
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
            os.rmdir(socket_dir)


def _scratch_dir() -> str:
    """A fresh directory the solution may write in, separate from its own (read-only) file."""
    return tempfile.mkdtemp(prefix='deepgym_run_')


@contextlib.contextmanager
def _start_signal() -> Iterator[
    tuple[list[str], tuple[int, ...], Callable[[], bool], Callable[[], None]]
]:
    """Yield a command prefix that holds the solution until released and then writes to a
    pipe once its Python is up, the descriptors to pass on, a check of whether it started,
    and the release.

    The hold means a judge that dies before handing the solution to its watcher (see
    :func:`_watch_group`) starts nothing: the prefix sees the go pipe close and exits.
    The script never holds the pipes, so it can't fake a launch failure. setpriv and
    unshare pass them on; sudo (a sandbox judge that isn't root) closes them, so there
    nothing is held or checked.
    """
    spec = _isolation()
    if any(os.path.basename(arg) == 'sudo' for arg in spec.get('prefix', [])):
        yield [], (), lambda: True, lambda: None
        return
    read_end, write_end = os.pipe()
    go_read, go_write = os.pipe()
    os.set_blocking(read_end, False)

    def started() -> bool:
        try:
            return os.read(read_end, 1) == b'1'
        except BlockingIOError:
            return False

    def release() -> None:
        os.write(go_write, b'1')

    prefix = [_python(), '-I', '-S', '-c', _STARTED, str(go_read), str(write_end)]
    try:
        yield prefix, (go_read, write_end), started, release
    finally:
        for fd in (read_end, write_end, go_read, go_write):
            os.close(fd)


_consulted = False
_preflight_passed = False
# The preflight's floor: less free space than this before the solution runs is the sandbox's.
_MIN_FREE_BYTES = 16 << 20
_MIN_FREE_INODES = 256


_announced = False


def _announce() -> None:
    """Print DEEPGYM_CONSULTED once, when solution code may start: a timeout or a sandbox
    death the host sees after it is the solution's doing (see JudgeJob.consulted_mark)."""
    global _announced
    mark = os.environ.get('DEEPGYM_CONSULTED')
    if mark and not _announced:
        _announced = True
        print(f'\n{mark}', flush=True)


def consulted() -> bool:
    """Whether solution code has started in this verifier; from then on a verifier crash
    is the solution's failure (see deepgym.verifier_template.guard_crashes)."""
    return _consulted


def _launch_failure(reason: str) -> BaseException:
    """Return the error for the judge failing to start the solution, as the module says.

    Before any solution code has run, the machine failed: :class:`LaunchError`. After, the
    solution may be why (a full disk, processes it used up): :class:`SolutionError`.
    """
    if _consulted:
        return SolutionError(f'could not start the solution again: {reason}')
    return LaunchError(f'could not start the solution: {reason}')


@contextlib.contextmanager
def _launching() -> Iterator[None]:
    """Turn the judge's own errors while starting the solution into :func:`_launch_failure`.

    That is OS and subprocess errors and RuntimeError (a thread that can't start); no
    solution code runs inside.
    """
    try:
        yield
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise _launch_failure(str(exc)) from exc


def _preflight() -> None:
    """Check, once and before any solution code runs, that the sandbox can host it.

    There must be free disk and inodes where runs write, and the isolation launcher (sudo,
    setpriv, bwrap, sandbox-exec) must start an empty program from a fresh directory:
    starting the launcher doesn't prove it can start code. Run inside :func:`_launching`,
    so a failure is the sandbox's; exhaustion after this is the solution's.

    Raises:
        OSError: If space is short or the launcher can't start code.
        subprocess.TimeoutExpired: If the launcher doesn't finish in time.
    """
    global _preflight_passed
    if _preflight_passed:
        return
    workdir = _scratch_dir()
    disk = os.statvfs(workdir)
    free_bytes = disk.f_bavail * disk.f_frsize
    # A file system that reports no inode counts at all (f_files 0) has no inode limit.
    if free_bytes < _MIN_FREE_BYTES or (disk.f_files and disk.f_favail < _MIN_FREE_INODES):
        raise OSError(f'only {free_bytes} bytes and {disk.f_favail} inodes free')
    probe = subprocess.run(
        _launch_argv([_python(), '-I', '-S', '-c', 'pass'], workdir),
        cwd=workdir,
        capture_output=True,
        timeout=min(_LOAD_TIMEOUT_S, _remaining()),
        start_new_session=True,
    )
    if probe.returncode != 0:
        error = probe.stderr.decode('utf-8', 'replace')[-500:]
        raise OSError(f'the isolation launcher exited {probe.returncode}: {error}')
    _preflight_passed = True


def _kill_group(group: int) -> None:
    """Kill every process in process group *group*: an isolated solution and its children."""
    _sweep_solver_user()
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group, signal.SIGKILL)


def _sweep_solver_user() -> None:
    """With a sandbox user, kill every process it runs, not just one group: children that
    call setsid() would otherwise outlive the call and could tamper with files the judge
    reads next. A concurrent load_solution server restarts on its next call."""
    spec = _isolation()
    if spec['kind'] == 'user':
        kill_all = [*spec['prefix'], 'bash', '-c', 'kill -9 -1']
        subprocess.run(kill_all, capture_output=True, check=False)


def _kill(proc: subprocess.Popen) -> None:
    """Kill an isolated process and everything in its process group, and reap it."""
    _kill_group(proc.pid)
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=5)


_watcher: subprocess.Popen | None = None


def _start_watcher() -> None:
    """Start this judge's watcher (see :func:`_watch`) once, before its first solution."""
    global _watcher
    if _watcher is None:
        _watcher = subprocess.Popen(
            [sys.executable, '-I', '-B', _runtime(), '--watch'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        atexit.register(_stop_watcher)


def _stop_watcher() -> None:
    """At the judge's own exit, let the watcher finish first, so its sweep of a sandbox
    user never lands on the next grade's solution."""
    if _watcher is None:
        return
    with contextlib.suppress(OSError):
        _watcher.stdin.close()
    with contextlib.suppress(subprocess.TimeoutExpired):
        _watcher.wait(timeout=10)


def _watch_group(group: int) -> None:
    """Have the watcher kill process group *group* (a solution just started) if this judge
    ends without doing so itself: killed by its host, say.

    Raises:
        OSError: If the watcher can't be told (it died, say).
    """
    _watcher.stdin.write(f'{group}\n'.encode())
    _watcher.stdin.flush()


def _remaining() -> float:
    """Seconds left before the run's overall deadline."""
    deadline = os.environ.get('DEEPGYM_DEADLINE')
    return math.inf if deadline is None else max(0.0, float(deadline) - time.time())


def _call_timeout() -> float:
    """Seconds one call may take: the per-case limit, capped by the run's deadline."""
    return min(float(os.environ.get('DEEPGYM_CASE_TIMEOUT', '10')), _remaining())


def run_solution(
    solution_path: str,
    args: Sequence[str] = (),
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    input: str | None = None,
    timeout: float | None = None,
    allow_subprocess: bool = False,
    loopback_port: int | None = None,
) -> subprocess.CompletedProcess:
    """Run the solution as a script in the isolated domain.

    *cwd* (default: a fresh scratch directory) is the only place it may write;
    pass any output location inside it and read results back with
    :func:`read_output`. *env* adds variables to an otherwise empty environment.
    Anything the script leaves running is killed when it exits. On macOS the
    script can't start subprocesses unless *allow_subprocess* is set (a child
    that detaches could outlive it there). *loopback_port* lets it connect to a
    server the judge runs on 127.0.0.1 at that port; the rest of the network stays
    unreachable.

    Raises:
        subprocess.TimeoutExpired: If it runs past *timeout* (default: the case timeout);
            a :class:`SolutionError` too.
        LaunchError: If the process can't be started, or never got Python going, before
            any solution code has run; after that, a :class:`SolutionError`.
    """
    global _consulted
    path = os.path.abspath(solution_path)
    command = [_python(), '-B', path, *args]
    limit = _call_timeout() if timeout is None else min(timeout, _remaining())
    with contextlib.ExitStack() as stack:
        with _launching():
            _preflight()
            _start_watcher()
            workdir = os.path.abspath(cwd or _scratch_dir())
            # Output goes to files, not pipes: a forked child holding a pipe open would
            # keep us waiting (and alive to tamper) after the script itself exits.
            out = stack.enter_context(tempfile.TemporaryFile())
            err = stack.enter_context(tempfile.TemporaryFile())
            forwarder = stack.enter_context(_loopback(loopback_port))
            prefix, signal_fds, started, release = stack.enter_context(_start_signal())
            launch = [*prefix, *forwarder, *command]
            argv = _launch_argv(launch, workdir, env, allow_subprocess, loopback_port)
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                cwd=workdir,
                start_new_session=True,
                pass_fds=signal_fds,
            )
            try:
                _watch_group(proc.pid)
                release()
                _announce()
            except BaseException:
                _kill(proc)  # held until released, so no solution code ran
                raise
        timed_out = False
        try:
            proc.communicate(None if input is None else input.encode(), timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Read before the cleanup, which a solution that used up the processes can break.
            ran = started()
            _consulted = _consulted or ran
            with _launching():
                _kill(proc)
        # Solution code used up the time, this launch's or earlier calls' (it may never start).
        if timed_out and _consulted:
            raise _marked(subprocess.TimeoutExpired, SolutionTimeoutError)(argv, limit)
        if not ran:  # a launcher that never got Python going ran no solution code
            raise _launch_failure('the solution process did not start')
        out.seek(0)
        err.seek(0)
        stdout = out.read(_MAX_MESSAGE).decode('utf-8', 'replace')
        stderr = err.read(_MAX_MESSAGE).decode('utf-8', 'replace')
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _open_inside(path: str, root: str) -> int:
    """Open *path* for reading, walking from *root* one component at a time.

    No component may be a symlink, so a solution can't redirect the judge to its
    own files, even by swapping a directory between a check and the open.
    """
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    parts = rel.split(os.sep)
    if rel in ('.', '') or parts[0] == '..' or os.path.isabs(rel):
        raise PermissionError(f'{path} is not inside {root}')
    fd = os.open(os.path.realpath(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)


def read_output(path: str, root: str) -> str:
    """Read a text file the solution produced, refusing symlinks and paths outside *root*.

    A solution could otherwise point its "output" at a judge file and have the
    judge read, say, expected answers back as if it had produced them.

    Raises:
        OSError: If the file is missing, reached through a symlink, or outside *root*.
    """
    with os.fdopen(_open_inside(path, root), encoding='utf-8') as handle:
        return handle.read()


def read_output_bytes(path: str, root: str, limit: int = _MAX_MESSAGE) -> bytes:
    """Binary twin of :func:`read_output`, reading at most *limit* bytes."""
    with os.fdopen(_open_inside(path, root), 'rb') as handle:
        return handle.read(limit)


def solution_source(solution_path: str, name: str | None = None) -> str:
    """Return the solution's source text, or one top-level def/class by *name*.

    Reads the file without running it, for static checks such as hardcoding.
    If *name* isn't a top-level def or class (a lambda, an alias, a conditional
    def), the whole file is returned so the check still sees the code.
    """
    with open(solution_path, encoding='utf-8') as handle:
        source = handle.read()
    if name is None:
        return source
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in tree.body:
        if isinstance(node, defs) and node.name == name:
            return ast.get_source_segment(source, node) or source
    return source


_HUNK_HEADER = re.compile(r'@@ -(\d+)(?:,(\d+))? \+\d+(?:,(\d+))? @@')
# Git's extended headers for the file operations apply_patch refuses.
_FILE_OPS = re.compile(r'(?:new|deleted) file mode |(?:rename|copy) (?:from|to) ')
# str.splitlines also breaks at U+2028 and other separators a string literal may contain.
_LINE = re.compile(r'.*\n|.+\Z')


def _diff_path(header: str) -> str:
    """File name from a ``--- a/x`` or ``+++ b/x`` line, without the a/ or b/ prefix.

    Git quotes a path with special characters C-style (``"b/caf\\303\\251.py"``), or with
    the characters left as they are under ``core.quotePath=false``; that is decoded first.
    """
    path = header[4:].split('\t')[0].strip()
    if len(path) > 1 and path[0] == path[-1] == '"':
        # Literal characters go in as UTF-8 bytes, which unicode_escape maps one to one
        # onto Latin-1 characters, exactly as it maps the octal escapes.
        escaped = path[1:-1].encode('utf-8').decode('unicode_escape')
        path = escaped.encode('latin-1').decode('utf-8')
    return path[2:] if path.startswith(('a/', 'b/')) else path


def _apply_hunks(name: str, text: str, lines: list[str], index: int) -> tuple[str, int]:
    """Apply the hunks starting at ``lines[index]`` to *text*; return it and the next index.

    As ``git apply`` does, text after the last hunk is left alone unless it starts like
    a hunk header (``@@ -``), which makes the patch corrupt.

    Raises:
        ValueError: If no hunk follows the file header, or a hunk is malformed or
            doesn't match.
    """
    source = _LINE.findall(text)
    # Added lines take the file's line ending; the patch's own was stripped with its \r.
    newline = '\r\n' if source and source[0].endswith('\r\n') else '\n'
    output: list[str] = []
    pos, first = 0, index
    while index < len(lines) and (header := _HUNK_HEADER.match(lines[index])):
        old_left, new_left = int(header[2] or 1), int(header[3] or 1)
        # A pure insertion (-N,0) goes after line N; any other hunk starts at line N.
        start = int(header[1]) if old_left == 0 else max(int(header[1]) - 1, 0)
        if start < pos:
            raise ValueError(f'{name}: hunks overlap or are out of order')
        if start > len(source):
            raise ValueError(f'{name}: hunk starts at line {header[1]}, past the end of the file')
        output += source[pos:start]
        pos, index, tag = start, index + 1, ' '
        # Counts cover the hunk's lines; a "\ No newline" marker may follow the last one.
        while (
            old_left > 0 or new_left > 0 or (index < len(lines) and lines[index].startswith('\\'))
        ):
            if index >= len(lines):
                raise ValueError(f'{name}: patch ends inside a hunk')
            line = lines[index]
            index += 1
            if line.startswith('\\'):  # "\ No newline at end of file" for the line before
                if tag == '+':
                    output[-1] = output[-1].removesuffix(newline)
                continue
            # A blank line inside a hunk is blank context whose leading space was stripped.
            tag, body = (line[0], line[1:]) if line else (' ', '')
            if tag == '+':
                output.append(body + newline)
                new_left -= 1
                continue
            if tag not in ' -':
                raise ValueError(f'{name}: unexpected line in hunk: {line[:80]!r}')
            if pos >= len(source) or source[pos].rstrip('\r\n') != body:
                raise ValueError(f'{name}: hunk does not match line {pos + 1}')
            if tag == ' ':
                output.append(source[pos])
                new_left -= 1
            old_left -= 1
            pos += 1
        if old_left < 0 or new_left < 0:
            raise ValueError(f'{name}: hunk line counts do not match its header')
    if index < len(lines) and lines[index].startswith('@@ -'):
        raise ValueError(f'{name}: malformed hunk header {lines[index][:80]!r}')
    if index == first:
        raise ValueError(f'{name}: no hunk follows the file header')
    return ''.join(output + source[pos:]), index


def apply_patch(patch: str, files: dict[str, str]) -> dict[str, str]:
    """Apply a unified diff to in-memory *files* and return the patched copy.

    Text in, text out: nothing touches the filesystem, and only files already in
    *files* can change (creating, deleting or renaming files isn't supported), so a
    patch can't reach anything the judge didn't hand over. Hunks must match exactly
    at the line their header names; text around the diff (prose, code fences) is ignored.

    Raises:
        ValueError: If the patch has no file diffs, edits an unknown file, creates,
            deletes, renames or copies one, or a hunk doesn't match.
    """
    patched = dict(files)
    lines = [line.removesuffix('\n').removesuffix('\r') for line in _LINE.findall(patch)]
    if any(_FILE_OPS.match(line) for line in lines):
        raise ValueError('creating, deleting, renaming or copying files is not supported')
    index, applied = 0, False
    while index < len(lines) - 1:
        if not (lines[index].startswith('--- ') and lines[index + 1].startswith('+++ ')):
            index += 1
            continue
        old, name = _diff_path(lines[index]), _diff_path(lines[index + 1])
        if '/dev/null' in (old, name):
            raise ValueError('creating or deleting files is not supported')
        if old != name:
            raise ValueError(f'patch renames {old!r} to {name!r}; renaming is not supported')
        if name not in patched:
            raise ValueError(f'patch edits {name!r}, which is not one of {sorted(files)}')
        patched[name], index = _apply_hunks(name, patched[name], lines, index + 2)
        applied = True
    if not applied:
        raise ValueError('no unified diff found (expected --- a/<file> and +++ b/<file> lines)')
    return patched


# ---------------------------------------------------------------------------
# Judge side: module proxy over a pipe
# ---------------------------------------------------------------------------


class _Runner:
    """One solution-side server process, restarted after a timeout or crash."""

    def __init__(self, solution_path: str, files: dict[str, str] | None = None) -> None:
        self._path = solution_path if files is not None else os.path.abspath(solution_path)
        self._files = files
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue[str | None] = queue.Queue()
        self._next_id = 0
        self.exports: dict[str, str] = {}
        self._start()
        atexit.register(self.close)

    def _start(self) -> None:
        global _consulted
        with _launching():
            _preflight()
            _start_watcher()
            workdir = _scratch_dir()
            path = self._path
            if self._files is not None:
                # Fresh copies on every (re)start: earlier runs may have rewritten them.
                for rel, text in self._files.items():
                    target = os.path.join(workdir, rel)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with open(target, 'w', encoding='utf-8') as handle:
                        handle.write(text)
                path = os.path.join(workdir, self._path)
            command = [_python(), '-I', '-B', _runtime(), '--serve', path]
            if self._files is not None:
                command.append(workdir)
            self._proc = subprocess.Popen(
                _launch_argv(command, workdir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=workdir,
                start_new_session=True,
            )
            try:
                _watch_group(self._proc.pid)
                # Requests are written without blocking, so a server that stops reading
                # can't hold a call past its deadline (see _write).
                os.set_blocking(self._proc.stdin.fileno(), False)
                self._replies = queue.Queue()
                reader = threading.Thread(target=self._read, args=(self._proc, self._replies))
                reader.daemon = True
                reader.start()
            except BaseException:
                self._dead()
                raise
        # The server reports in before it touches the solution, so a failure before that
        # (a launcher that never got Python going) is the machine's until solution code
        # has run (see _launch_failure).
        try:
            ready = self._replies.get(timeout=min(_LOAD_TIMEOUT_S, _remaining()))
        except queue.Empty:
            ready = None
        if ready != _READY:
            self._dead()
            if _consulted and _remaining() <= 0:  # earlier calls used up the run
                raise _marked(TimeoutError, SolutionTimeoutError)('the run is out of time')
            raise _launch_failure('the solution process did not start')
        _consulted = True
        _announce()
        self.exports = self._send({'op': 'load'}, min(_LOAD_TIMEOUT_S, _remaining()))

    @staticmethod
    def _read(proc: subprocess.Popen, replies: queue.Queue) -> None:
        while raw := proc.stdout.readline(_MAX_MESSAGE):
            replies.put(raw.decode('utf-8', 'replace'))
        replies.put(None)

    def request(self, message: dict) -> Any:
        """Send *message*, restarting the server first if a previous call killed it."""
        if self._proc is None or self._proc.poll() is not None:
            self._start()
        return self._send(message, _call_timeout())

    def _send(self, message: dict, timeout: float) -> Any:
        if timeout <= 0:
            raise _marked(TimeoutError, SolutionTimeoutError)('the run is out of time')
        # The deadline covers sending the request too, not just waiting for the reply.
        deadline = time.monotonic() + timeout
        self._next_id += 1
        message = {**message, 'id': self._next_id}
        try:
            sent = self._write((json.dumps(message) + '\n').encode(), deadline)
        except OSError as exc:
            self._dead()
            raise SolutionError('the solution process exited') from exc
        if not sent:
            self._dead()
            raise _marked(TimeoutError, SolutionTimeoutError)(
                f'solution call exceeded {timeout:.1f}s'
            )
        while True:
            try:
                raw = self._replies.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                self._dead()
                exceeded = f'solution call exceeded {timeout:.1f}s'
                raise _marked(TimeoutError, SolutionTimeoutError)(exceeded) from None
            if raw is None:
                self._dead()
                raise SolutionError('the solution process exited during the call')
            try:
                reply = json.loads(raw)
                if reply['id'] != message['id']:
                    continue
                if not reply['ok']:
                    break
                return _decode(reply['value'], lambda tag: _RemoteObject(self, int(tag['$h'])))
            except Exception as exc:  # noqa: BLE001 - anything malformed is the solution's fault
                self._dead()
                raise SolutionError(f'malformed reply from the solution: {exc}') from exc
        raise _remote_exception(str(reply.get('type', '')), str(reply.get('message', '')))

    def _write(self, data: bytes, deadline: float) -> bool:
        """Write *data* to the server's stdin by *deadline*; return False if it doesn't fit.

        A pipe holds about 64 KiB, so a big request to a server that stopped reading
        (stuck or stopped, say) would otherwise block here for good.
        """
        fd = self._proc.stdin.fileno()
        pending = memoryview(data)
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_WRITE)
            while pending:
                if not selector.select(deadline - time.monotonic()):
                    return False
                pending = pending[os.write(fd, pending) :]
        return True

    def _dead(self) -> None:
        if self._proc is not None:
            _kill(self._proc)
        self._proc = None

    def close(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.stdin.close()
            _kill(self._proc)
            self._proc = None


def _remote_exception(name: str, message: str) -> SolutionError:
    """Rebuild a builtin exception type by name, as a SolutionError too; anything else (or
    a type that can't be rebuilt from a message) becomes a plain SolutionError."""
    kind = getattr(builtins, name, None)
    if isinstance(kind, type) and issubclass(kind, Exception):
        with contextlib.suppress(Exception):
            return _marked(kind)(message)
    return SolutionError(f'{name}: {message}')


class _RemoteCallable:
    def __init__(self, runner: _Runner, target: dict, name: str) -> None:
        self._runner = runner
        self._target = target
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._runner.request(
            {
                'op': 'call',
                'target': self._target,
                'args': _encode(list(args), None),
                'kwargs': _encode(kwargs, None),
            }
        )

    def __repr__(self) -> str:
        return f'<solution callable {self.__name__}>'


class _RemoteObject:
    """An object living in the solution process; attribute access and calls run there."""

    def __init__(self, runner: _Runner, handle: int) -> None:
        self._runner = runner
        self._handle = handle

    def __getattr__(self, name: str) -> _RemoteCallable:
        if name.startswith('_'):
            raise AttributeError(name)
        return _RemoteCallable(self._runner, {'handle': self._handle, 'attr': name}, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _RemoteCallable(self._runner, {'handle': self._handle}, 'object')(*args, **kwargs)

    def __repr__(self) -> str:
        return f'<solution object #{self._handle}>'


class _RemoteModule:
    """The solution module as the judge sees it: names known, everything runs remotely."""

    def __init__(self, runner: _Runner) -> None:
        self._runner = runner

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(f'solution has no attribute {name!r}')
        kind = self._runner.exports.get(name)
        if kind is None:
            raise _marked(AttributeError)(f'solution has no attribute {name!r}')
        target = {'name': name}
        if kind == 'callable':
            return _RemoteCallable(self._runner, target, name)
        return self._runner.request({'op': 'get', 'target': target})

    def __dir__(self) -> list[str]:
        return sorted(self._runner.exports)


def solution_namespace(module: _RemoteModule) -> dict[str, Any]:
    """Names for exec-style test code: every function as a proxy, plus module-level
    values that can be sent cheaply (others are skipped instead of failing the run)."""
    namespace: dict[str, Any] = {}
    for name, kind in module._runner.exports.items():
        if kind == 'callable':
            namespace[name] = _RemoteCallable(module._runner, {'name': name}, name)
            continue
        with contextlib.suppress(Exception):
            namespace[name] = getattr(module, name)
    return namespace


def load_solution(solution_path: str) -> _RemoteModule:
    """Import the solution in the isolated domain and return a proxy for it.

    Raises:
        SolutionError: The solution's own import error, also of the type it raised
            (SyntaxError, ImportError, ...), or a TimeoutError if importing takes too long.
    """
    return _RemoteModule(_Runner(solution_path))


def load_files(files: dict[str, str], entry: str) -> _RemoteModule:
    """Import *entry* from *files* in the isolated domain and return a proxy for it.

    For code the judge assembles itself, such as a fixture repo with the model's patch
    applied (see :func:`apply_patch`). The files are written into the isolated
    process's own fresh directory, the only place it can read and write besides its
    runtime, so the entry module can import its siblings. Paths are relative.

    Raises:
        ValueError: If a path is absolute, climbs out with '..', or *entry* isn't in *files*.
        Exception: The code's own import error (SyntaxError, ImportError, ...).
        TimeoutError: If importing takes too long.
    """
    for rel in files:
        parts = rel.replace('\\', '/').split('/')
        if os.path.isabs(rel) or '..' in parts or '' in parts:
            raise ValueError(f'file path {rel!r} must be relative, without ".." or empty parts')
    if entry not in files:
        raise ValueError(f'entry {entry!r} is not one of {sorted(files)}')
    return _RemoteModule(_Runner(entry, dict(files)))


# ---------------------------------------------------------------------------
# Solution side: the server
# ---------------------------------------------------------------------------


def _serve(solution_path: str, root: str | None = None) -> None:
    """Load the solution and answer calls over the original stdin/stdout pipes.

    With *root* (files from :func:`load_files`), the entry is imported the normal way
    with *root* on the path, so its packages run their ``__init__``, it sits in
    ``sys.modules`` under its dotted name and import cycles resolve; otherwise it is
    module ``solution`` with its own directory on the path.

    It exits when the judge closes the pipe, even if the solution left threads
    running, and within half a second of the judge dying mid-call (its parent
    changes), so a server never outlives its run where nothing else would reap it.
    """
    requests = os.fdopen(os.dup(0), 'r', encoding='utf-8')
    replies = os.fdopen(os.dup(1), 'w', encoding='utf-8')
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(2, 1)
    handles: dict[int, Any] = {}
    module: Any = None
    # Bound before the solution loads: it may rebind os or time.
    getppid, sleep, exit_now, parent = os.getppid, time.sleep, os._exit, os.getppid()

    def watch_judge() -> None:
        while getppid() == parent:
            sleep(0.5)
        exit_now(1)

    threading.Thread(target=watch_judge, daemon=True).start()
    replies.write(_READY)
    replies.flush()

    def decode(data: Any) -> Any:
        return _decode(data, lambda tag: handles[tag['$h']])

    def resolve(target: dict) -> Any:
        if 'handle' in target:
            obj = handles[target['handle']]
            return getattr(obj, target['attr']) if 'attr' in target else obj
        return getattr(module, target['name'])

    for line in requests:
        message = json.loads(line)
        try:
            op = message['op']
            if op == 'load':
                # -I leaves the module's own directory off the path.
                if root is None:
                    sys.path.insert(0, os.path.dirname(os.path.abspath(solution_path)))
                    spec = importlib.util.spec_from_file_location('solution', solution_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    # A top-level file named like a module this process already loaded
                    # (json.py, say) would import as that module, not as the file.
                    tops = {entry.removesuffix('.py') for entry in os.listdir(root)}
                    if taken := sorted(tops & sys.modules.keys()):
                        raise ImportError(
                            f'files {taken} name modules the runtime already loaded; rename them'
                        )
                    name = os.path.splitext(os.path.relpath(solution_path, root))[0]
                    name = name.replace(os.sep, '.').removesuffix('.__init__')  # pkg, once
                    sys.path.insert(0, root)
                    module = importlib.import_module(name)
                exports = {
                    name: 'callable' if callable(obj) else 'value'
                    for name, obj in vars(module).items()
                    if not name.startswith('_')
                }
                value = _encode(exports, None)
            else:
                target = resolve(message['target'])
                if op == 'call':
                    target = target(*decode(message['args']), **decode(message['kwargs']))
                value = _encode(target, handles)
            reply = {'id': message['id'], 'ok': True, 'value': value}
        except BaseException as exc:  # noqa: BLE001 - every solution failure goes back to the judge
            reply = {
                'id': message.get('id'),
                'ok': False,
                'type': type(exc).__name__,
                'message': str(exc)[:2000],
            }
        replies.write(json.dumps(reply) + '\n')
        replies.flush()
    exit_now(0)


def _forward(port: int, judge_socket: str, command: list[str]) -> int:
    """Serve 127.0.0.1:*port* by piping to *judge_socket*, run *command*, return its exit code."""

    def connect() -> socket.socket:
        upstream = socket.socket(socket.AF_UNIX)
        upstream.connect(judge_socket)
        return upstream

    _relay(socket.create_server(('127.0.0.1', port)), connect)
    return subprocess.call(command)


def _watch() -> None:
    """Kill each process group the judge names on stdin once the judge is gone.

    The judge holds the other end of stdin, so it closes however the judge ends: its own
    exit, a crash, or a host that killed it, which runs none of the judge's cleanup. The
    watcher runs in its own session, so a host that kills the judge's process group
    leaves it to finish, and in its own process, so a solution that hogs the interpreter
    (a long C call holds the lock every thread needs) can't stop it. Each second it drops
    groups that are gone; killing an unrelated group would take the system reusing one's
    id within that second.
    """
    groups: set[int] = set()
    seen = False
    pending = b''
    with selectors.DefaultSelector() as selector:
        selector.register(0, selectors.EVENT_READ)
        while True:
            if selector.select(1.0):
                data = os.read(0, 4096)
                if not data:
                    break
                *lines, pending = (pending + data).split(b'\n')
                groups.update(int(line) for line in lines if line)
                seen = seen or bool(groups)
            groups = {group for group in groups if _group_exists(group)}
    if seen:  # a child that left its group (setsid) is still the sandbox user's
        _sweep_solver_user()
    for group in groups:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(group, signal.SIGKILL)


def _group_exists(group: int) -> bool:
    """Whether process group *group* still has a process (perhaps a sandbox user's)."""
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _confine(workdir: str, command: list[str]) -> int:
    """Run *command* (a whole judge) in this process's DEEPGYM_ISOLATION sandbox.

    It may write only in *workdir*, sees nothing of the host beyond what the spec
    exposes, and has no network; it keeps this process's working directory. Whatever
    it starts shares that sandbox, so it is told to add no isolation of its own
    (sandbox-exec can't nest): for a judge that might run code in-process (as under
    mutation testing), the sandbox holds the entire run.

    Nothing outlives it on CPU: the judge runs in its own process group, killed when it
    exits or at the run's deadline (plus :data:`_CONFINE_GRACE_S`, inside the host's own
    timeout), and every process of the run inherits a hard CPU limit of about twice the
    run's time, which setsid() doesn't escape. Under bwrap the PID namespace dies with
    the run. sandbox-exec has no such namespace, so there the code the judge launches
    (load_solution and run_solution, allow_subprocess or not) may start no processes;
    code the judge ran in-process could still start one that detaches and lingers,
    idle, which is why mutation testing runs locally only under bwrap (see
    :func:`deepgym.qa.mutants.mutation_check`).

    Returns:
        The judge's exit code, or 124 if it outran the deadline.
    """
    import resource  # Unix only, like the sandboxes this runs in

    env = {key: value for key, value in os.environ.items() if key != 'DEEPGYM_ISOLATION'}
    inner = json.dumps({'kind': 'none', 'no_fork': _isolation()['kind'] == 'macos'})
    argv = _launch_argv(command, workdir, {**env, 'DEEPGYM_ISOLATION': inner}, allow_fork=True)
    if '--chdir' in argv:  # bwrap: start where an unconfined judge would, not in workdir
        argv[argv.index('--chdir') + 1] = os.getcwd()
    seconds = _remaining()
    limit = None
    if math.isfinite(seconds):
        limit = seconds + _CONFINE_GRACE_S
        cpu = math.ceil(2 * limit)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    judge = subprocess.Popen(argv, start_new_session=True)
    try:
        return judge.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        return 124
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(judge.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            judge.wait(timeout=5)


if __name__ == '__main__' and sys.argv[1:2] == ['--serve']:
    _serve(sys.argv[2], *sys.argv[3:4])
elif __name__ == '__main__' and sys.argv[1:2] == ['--forward']:
    sys.exit(_forward(int(sys.argv[2]), sys.argv[3], sys.argv[4:]))
elif __name__ == '__main__' and sys.argv[1:2] == ['--confine']:
    sys.exit(_confine(sys.argv[2], sys.argv[3:]))
elif __name__ == '__main__' and sys.argv[1:2] == ['--watch']:
    _watch()
elif __name__ == '__main__':
    usage = (
        '--serve <solution.py> | --forward <port> <socket> <command...> '
        '| --confine <dir> <command...> | --watch'
    )
    print(f'usage: {shlex.quote(sys.argv[0])} {usage}', file=sys.stderr)
    sys.exit(2)
