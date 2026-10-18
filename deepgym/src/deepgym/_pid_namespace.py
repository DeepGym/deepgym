"""Keep every session agent process in one kernel-owned lifetime (sandbox-only)."""

import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


def keep(directory: Path) -> None:
    """Act as the trusted PID 1, reaping orphaned setup and agent children."""
    if os.getpid() != 1:
        raise RuntimeError('Agent keeper is not PID 1')

    def reap(signum: int, frame: object) -> None:
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass

    signal.signal(signal.SIGCHLD, reap)
    (directory / 'ready').touch()
    while True:
        signal.pause()


def create(directory: Path, uid: int) -> None:
    """Create and pin PID/mount namespaces before any agent process can start.

    Args:
        directory: Root-only control directory created by SandboxBackend.
        uid: Exclusive, non-root agent UID.

    Raises:
        RuntimeError: If the namespace keeper cannot be established.
        subprocess.SubprocessError: If kernel containment is unavailable.
    """
    if uid <= 0:
        raise ValueError('Refusing to contain root as an agent')
    descriptor = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(descriptor, 0)
    finally:
        os.close(descriptor)
    wrapper = subprocess.Popen(
        [
            'unshare',
            '--pid',
            '--fork',
            '--mount-proc',
            '--kill-child=KILL',
            '--',
            sys.executable,
            '-I',
            __file__,
            'keep',
            str(directory),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (directory / 'ready').exists():
            if wrapper.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Could not create the agent PID namespace')
            time.sleep(0.01)
        children = Path(f'/proc/{wrapper.pid}/task/{wrapper.pid}/children').read_text().split()
        if len(children) != 1:
            raise RuntimeError('Cannot identify the agent namespace init')
        init = int(children[0])
        for namespace in ('pid', 'mnt'):
            target = directory / namespace
            target.touch(mode=0o600)
            subprocess.run(
                ['mount', '--bind', f'/proc/{init}/ns/{namespace}', str(target)],
                check=True,
                timeout=10,
            )
        (directory / 'identity.json').write_text(json.dumps({'pid': init, 'uid': uid}))
    except BaseException:
        wrapper.kill()
        wrapper.wait(timeout=10)
        raise


def verify_empty(uid: int) -> None:
    """Refuse live agent survivors after the kernel has destroyed their PID namespace.

    This scan is a postcondition, never the mechanism used to terminate processes.
    All unprivileged creation paths must enter the now-dead namespace first.
    """
    if uid <= 0:
        raise ValueError('Refusing to inspect root as an agent')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = dict(
                line.split(':', 1)
                for line in (entry / 'status').read_text().splitlines()
                if ':' in line
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields['Uid'].split()[0]) == uid and not fields['State'].strip().startswith('Z'):
            raise RuntimeError('Agent process survived PID namespace teardown; grading is unsafe')


def stop(directory: Path) -> None:
    """Kill the pinned namespace init from its ancestor, then verify kernel teardown.

    Raises:
        RuntimeError: If the init identity or namespace death cannot be established.
        OSError: If pidfds or namespace introspection are unavailable.
    """
    identity = json.loads((directory / 'identity.json').read_text())
    if identity['uid'] <= 0 or identity['pid'] <= 1:
        raise ValueError('Refusing root UID or host init as an agent namespace')
    if (directory / 'stopped').exists():
        verify_empty(identity['uid'])
        return
    descriptor = os.pidfd_open(identity['pid'])
    try:
        actual = Path(f'/proc/{identity["pid"]}/ns/pid').stat().st_ino
        if actual != (directory / 'pid').stat().st_ino:
            raise RuntimeError('Agent namespace init identity changed; refusing PID reuse')
        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        if not select.select([descriptor], [], [], 10)[0]:
            raise RuntimeError('Agent PID namespace did not terminate')
    finally:
        os.close(descriptor)
    verify_empty(identity['uid'])
    (directory / 'stopped').touch()


if __name__ == '__main__':
    operation, path, *arguments = sys.argv[1:]
    if operation == 'keep':
        keep(Path(path))
    elif operation == 'create':
        create(Path(path), int(arguments[0]))
    elif operation == 'stop':
        stop(Path(path))
    else:
        raise ValueError('Unknown namespace operation')
