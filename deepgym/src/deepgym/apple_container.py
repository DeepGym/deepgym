"""Apple `container` backend: each sandbox is a throwaway lightweight Linux VM on macOS.

Needs Apple's container CLI (https://github.com/apple/container) with its system
service running (``container system start``). :class:`AppleContainer` exposes
the same small surface the judge and sessions use on Daytona sandboxes
(``process.exec``, ``fs.upload_file``), so both backends
share one code path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from typing import NamedTuple
from uuid import uuid4

from deepgym.exceptions import SandboxError

DEFAULT_IMAGE = os.environ.get('DEEPGYM_CONTAINER_IMAGE', 'python:3.12-slim')
_CLI = 'container'
_START_TIMEOUT_S = 300
# Exit codes from coreutils `timeout`: 124 after SIGTERM, 137 after the follow-up SIGKILL.
_TIMEOUT_EXIT_CODES = frozenset({124, 137})


class ContainerTimeoutError(Exception):
    """Raise when a command inside a container outruns its timeout."""


class ExecResponse(NamedTuple):
    """Exit code and combined output, shaped like Daytona's exec response."""

    exit_code: int
    result: str


def apple_container_available() -> bool:
    """Return True if the container CLI is installed and its service is running."""
    if shutil.which(_CLI) is None:
        return False
    try:
        status = subprocess.run(
            [_CLI, 'system', 'status'], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return status.returncode == 0 and 'running' in status.stdout


def _cli(*args: str, input: bytes | None = None, timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [_CLI, *args], input=input, capture_output=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        raise SandboxError(
            'Apple container CLI not found; install it and run `container system start`'
        ) from exc


class AppleContainer:
    """A running container VM, deleted by :meth:`delete`.

    ``process`` and ``fs`` point back at this object so judge and session code
    written against Daytona's ``sandbox.process.exec`` / ``sandbox.fs.upload_file``
    works unchanged.

    The container gets CAP_SYS_ADMIN and CAP_NET_ADMIN so root setup code can
    create an offline network namespace (with loopback up). Untrusted code (the
    judged solution, session commands) always runs as a non-root user with every
    capability dropped, so it can't leave that namespace, change the network, or
    tamper with the binaries root runs.
    """

    def __init__(self, image: str = DEFAULT_IMAGE) -> None:
        self.id = f'deepgym-{uuid4().hex[:12]}'
        started = _cli(
            'run', '--detach', '--rm', '--name', self.id, '--label', 'deepgym=1',
            '--cap-add', 'CAP_SYS_ADMIN', '--cap-add', 'CAP_NET_ADMIN',
            image, 'sleep', 'infinity',
            timeout=_START_TIMEOUT_S,
        )  # fmt: skip
        if started.returncode != 0:
            raise SandboxError(f'Failed to start container: {started.stderr.decode()[-1000:]}')
        self.process = self
        self.fs = self

    def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecResponse:
        """Run *command* with sh in the container; output is stdout followed by stderr.

        Raises:
            ContainerTimeoutError: If the command outruns *timeout* seconds.
        """
        argv = ['exec']
        if cwd:
            argv += ['--workdir', cwd]
        for key, value in (env or {}).items():
            argv += ['--env', f'{key}={value}']
        argv.append(self.id)
        if timeout:
            argv += ['timeout', '--kill-after=2', str(timeout)]
        argv += ['sh', '-c', command]
        start = time.monotonic()
        try:
            done = _cli(*argv, timeout=(timeout or 3600) + 15)
        except subprocess.TimeoutExpired as exc:
            raise ContainerTimeoutError(f'command exceeded {timeout}s') from exc
        if timeout and done.returncode in _TIMEOUT_EXIT_CODES:
            if time.monotonic() - start >= timeout:
                raise ContainerTimeoutError(f'command exceeded {timeout}s')
        output = done.stdout.decode('utf-8', 'replace') + done.stderr.decode('utf-8', 'replace')
        return ExecResponse(done.returncode, output)

    def upload_file(
        self, content: bytes, path: str, timeout: int = 120, as_user: Sequence[str] = ()
    ) -> None:
        """Write *content* to *path* inside the container, creating parent directories.

        *as_user* is an argv prefix (e.g. ``setpriv --reuid=...``) that runs the write
        with less privilege; by default it runs as root.

        Raises:
            SandboxError: If the write fails.
        """
        script = 'mkdir -p "$(dirname "$1")" && cat > "$1"'
        done = _cli(
            'exec', '--interactive', self.id, *as_user, 'sh', '-c', script, 'upload', path,
            input=content, timeout=timeout,
        )  # fmt: skip
        if done.returncode != 0:
            raise SandboxError(f'Failed to upload {path}: {done.stderr.decode()[-500:]}')

    def delete(self) -> None:
        """Stop and remove the container; errors are ignored (it may already be gone)."""
        _cli('delete', '--force', self.id, timeout=60)
