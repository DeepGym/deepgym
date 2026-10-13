"""Run untrusted solution code in its own isolated process, away from the judge.

Verifiers (the judge) use this instead of importing or running the solution
themselves::

    from deepgym_solution import load_solution, run_solution

    solution = load_solution(solution_path)    # module proxy; every call runs remotely
    result = solution.two_sum([2, 7, 11], 9)    # per-call timeout, plain data only

    proc = run_solution(solution_path, cwd=work_dir, env={'OUT': work_dir})  # as a script

    # As a client of a server the judge runs on 127.0.0.1:port; no other network.
    proc = run_solution(solution_path, [f'http://127.0.0.1:{port}/'], loopback_port=port)

The solution process can't read the judge's files or memory, so it can't see
expected answers, patch the verifier, or print a fake score. Only plain data
crosses the boundary (None, bool, int, float, str, bytes, list, tuple, set,
frozenset, dict). Any other object comes back as an opaque handle, so a value
that claims to equal everything can't fake a pass.

Each call is bounded by DEEPGYM_CASE_TIMEOUT and by the run's overall
DEEPGYM_DEADLINE, so a slow case fails on its own instead of sinking the run.

Stdlib only: this file is copied next to the verifier as ``deepgym_solution.py``
and also runs as the solution-side server (``python -I deepgym_solution.py --serve``)
and loopback forwarder (``--forward``).
"""

from __future__ import annotations

import atexit
import base64
import builtins
import contextlib
import importlib.util
import json
import math
import os
import queue
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


class SolutionError(Exception):
    """Raise when the solution raised a non-builtin exception or broke the protocol."""


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
    network stack; elsewhere :func:`_loopback` forwards the port.
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


def _kill(proc: subprocess.Popen) -> None:
    """Kill an isolated process and everything in its process group."""
    spec = _isolation()
    if spec['kind'] == 'user':
        # Kill every process of the solution user, not just this group: children
        # that call setsid() would otherwise outlive the call and could tamper with
        # files the judge reads next. A concurrent load_solution server restarts on
        # its next call.
        kill_all = [*spec['prefix'], 'bash', '-c', 'kill -9 -1']
        subprocess.run(kill_all, capture_output=True, check=False)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=5)


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
        subprocess.TimeoutExpired: If it runs past *timeout* (default: the case timeout).
    """
    path = os.path.abspath(solution_path)
    workdir = os.path.abspath(cwd or _scratch_dir())
    command = [_python(), '-B', path, *args]
    limit = _call_timeout() if timeout is None else min(timeout, _remaining())
    # Output goes to files, not pipes: a forked child holding a pipe open would
    # keep us waiting (and alive to tamper) after the script itself exits.
    with (
        tempfile.TemporaryFile() as out,
        tempfile.TemporaryFile() as err,
        _loopback(loopback_port) as forwarder,
    ):
        proc = subprocess.Popen(
            _launch_argv([*forwarder, *command], workdir, env, allow_subprocess, loopback_port),
            stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            cwd=workdir,
            start_new_session=True,
        )
        try:
            proc.communicate(None if input is None else input.encode(), timeout=limit)
        finally:
            _kill(proc)
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


# ---------------------------------------------------------------------------
# Judge side: module proxy over a pipe
# ---------------------------------------------------------------------------


class _Runner:
    """One solution-side server process, restarted after a timeout or crash."""

    def __init__(self, solution_path: str) -> None:
        self._path = os.path.abspath(solution_path)
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue[str | None] = queue.Queue()
        self._next_id = 0
        self.exports: dict[str, str] = {}
        self._start()
        atexit.register(self.close)

    def _start(self) -> None:
        workdir = _scratch_dir()
        command = [_python(), '-I', '-B', _runtime(), '--serve', self._path]
        self._proc = subprocess.Popen(
            _launch_argv(command, workdir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=workdir,
            start_new_session=True,
        )
        # Requests are written without blocking, so a server that stops reading can't
        # hold a call past its deadline (see _write).
        os.set_blocking(self._proc.stdin.fileno(), False)
        self._replies = queue.Queue()
        threading.Thread(target=self._read, args=(self._proc, self._replies), daemon=True).start()
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
            raise TimeoutError('the run is out of time')
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
            raise TimeoutError(f'solution call exceeded {timeout:.1f}s')
        while True:
            try:
                raw = self._replies.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                self._dead()
                raise TimeoutError(f'solution call exceeded {timeout:.1f}s') from None
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


def _remote_exception(name: str, message: str) -> BaseException:
    """Rebuild builtin exception types by name; anything else becomes SolutionError."""
    kind = getattr(builtins, name, None)
    if isinstance(kind, type) and issubclass(kind, Exception):
        with contextlib.suppress(Exception):
            return kind(message)
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
        kind = self._runner.exports.get(name)
        if kind is None or name.startswith('_'):
            raise AttributeError(f'solution has no attribute {name!r}')
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
        Exception: The solution's own import error (SyntaxError, ImportError, ...).
        TimeoutError: If importing takes too long.
    """
    return _RemoteModule(_Runner(solution_path))


# ---------------------------------------------------------------------------
# Solution side: the server
# ---------------------------------------------------------------------------


def _serve(solution_path: str) -> None:
    """Load the solution and answer calls over the original stdin/stdout pipes."""
    requests = os.fdopen(os.dup(0), 'r', encoding='utf-8')
    replies = os.fdopen(os.dup(1), 'w', encoding='utf-8')
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(2, 1)
    handles: dict[int, Any] = {}
    module: Any = None

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
                spec = importlib.util.spec_from_file_location('solution', solution_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
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


def _forward(port: int, judge_socket: str, command: list[str]) -> int:
    """Serve 127.0.0.1:*port* by piping to *judge_socket*, run *command*, return its exit code."""

    def connect() -> socket.socket:
        upstream = socket.socket(socket.AF_UNIX)
        upstream.connect(judge_socket)
        return upstream

    _relay(socket.create_server(('127.0.0.1', port)), connect)
    return subprocess.call(command)


if __name__ == '__main__' and sys.argv[1:2] == ['--serve']:
    _serve(sys.argv[2])
elif __name__ == '__main__' and sys.argv[1:2] == ['--forward']:
    sys.exit(_forward(int(sys.argv[2]), sys.argv[3], sys.argv[4:]))
elif __name__ == '__main__':
    usage = '--serve <solution.py> | --forward <port> <socket> <command...>'
    print(f'usage: {shlex.quote(sys.argv[0])} {usage}', file=sys.stderr)
    sys.exit(2)
