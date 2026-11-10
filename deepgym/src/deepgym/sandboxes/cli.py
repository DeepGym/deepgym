"""Container CLIs (Apple ``container``, ``docker``) as sandboxes.

Each sandbox is one detached ``sleep infinity`` container. Commands go through
``<cli> exec`` and files through its stdin. The two CLIs take the same flags for
everything used here, so a provider only names its CLI and how it differs.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from deepgym.exceptions import SandboxDiedError, SandboxError, SandboxUnreachableError
from deepgym.sandboxes.protocol import (
    TIMEOUT_EXIT_CODES,
    TRUSTED_PATH,
    ExecResult,
    SandboxImage,
    SandboxTimeoutError,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = os.environ.get('DEEPGYM_CONTAINER_IMAGE', 'python:3.12-slim')
# Caps per sandbox, so hostile code can't starve the machine running the containers.
_LIMITS = ('--cpus', '2', '--memory', '2g')
_START_TIMEOUT_S = 300
_BUILD_TIMEOUT_S = 1800
_NAME_PREFIX = 'deepgym-'
# How much longer than a command's timeout the host waits for the CLI to answer.
_WAIT_GRACE_S = 15
# Root-only folder where each exec marks, by the container's clock, that its command runs.
_EXEC_DIR = '/run/deepgym-exec'
# Runs an exec's command ("$3"...): writes the container's uptime to its marker ($1) while
# the command runs, then ends stdout and stderr with its trailer ($2) and exit code. A
# marker it can't write (a full disk) never stops the command or changes its output.
_WRAPPER = (
    'm=$1 t=$2; shift 2; '
    f'{{ [ -d {_EXEC_DIR} ] || (PATH={TRUSTED_PATH}; mkdir -p -m 700 {_EXEC_DIR}); '
    'read -r up _ < /proc/uptime && echo "$up" > "$m"; } 2>/dev/null; '
    '"$@"; rc=$?; '
    f'(PATH={TRUSTED_PATH}; rm -f "$m") 2>/dev/null; '
    'printf "%s%s\\n" "$t" "$rc"; printf "%s%s\\n" "$t" "$rc" >&2; exit "$rc"'
)
# Prints the container's uptime, then the one in marker $1 if its command still runs.
_READ_MARKER = (
    'read -r now _ < /proc/uptime && echo "$now" && '
    '{ read -r up < "$1" && echo "$up"; } 2>/dev/null; exit 0'
)


class RunningSandbox(BaseModel):
    """A live DeepGym container as its CLI lists it."""

    model_config = ConfigDict(frozen=True)

    id: str
    image: str
    state: str
    started: str


class CliContainer:
    """A running container, deleted by :meth:`delete`."""

    cli: ClassVar[str]
    run_flags: ClassVar[tuple[str, ...]] = ()
    delete_args: ClassVar[tuple[str, ...]]
    list_args: ClassVar[tuple[str, ...]]
    network_blocked: ClassVar[bool]

    def __init__(self, image: str | None = None) -> None:
        self.id = f'{_NAME_PREFIX}{uuid4().hex[:12]}'
        self.image = image or DEFAULT_IMAGE
        try:
            started = self._cli(
                'run', '--detach', '--rm', '--name', self.id, '--label', 'deepgym=1',
                *_LIMITS, *self.run_flags, self.image, 'sleep', 'infinity',
                timeout=_START_TIMEOUT_S,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            # The container may still come up after we stop waiting; don't leak it.
            self.delete()
            raise SandboxError(f'Container did not start within {_START_TIMEOUT_S}s') from exc
        if started.returncode != 0:
            raise SandboxError(f'Failed to start container: {started.stderr.decode()}')

    @classmethod
    def build(cls, context: Path) -> str:
        """Build the Dockerfile in *context*, unless an identical build exists; return its tag.

        The tag hashes every path in *context* with its permission bits, and each file's
        bytes or each link's target, so changing any of them means a new build.

        Raises:
            SandboxError: If the build fails or times out.
        """
        digest = hashlib.sha256()
        for path in sorted(context.rglob('*')):
            mode = path.lstat().st_mode
            if path.is_symlink():
                data = os.readlink(path).encode()
            elif path.is_file():
                data = path.read_bytes()
            else:
                data = b''
            digest.update(
                f'{path.relative_to(context).as_posix()}\0{mode:o}\0{len(data)}\0'.encode()
            )
            digest.update(data)
        tag = f'{_NAME_PREFIX}build:{digest.hexdigest()[:16]}'
        if cls._cli('image', 'inspect', tag, timeout=30).returncode == 0:
            return tag
        try:
            built = cls._cli('build', '--tag', tag, str(context), timeout=_BUILD_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f'Building {context} took over {_BUILD_TIMEOUT_S}s') from exc
        if built.returncode != 0:
            raise SandboxError(f'Failed to build {context}: {built.stderr.decode()}')
        return tag

    @classmethod
    def available(cls) -> bool:
        """Return True if the CLI is installed and its daemon or service answers."""
        if shutil.which(cls.cli) is None:
            return False
        try:
            info = subprocess.run([cls.cli, 'info'], capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return False
        return info.returncode == 0

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        """Run *command* with sh in the container; output is stdout followed by stderr.

        A wrapper (see ``_WRAPPER``) reports how the command ended. Without that report the
        CLI or its daemon lost the outcome, which is ours, even while the container still
        lists: unless the container stopped, or its own clock shows the command still
        running past its deadline (the command's timeout).

        Raises:
            SandboxTimeoutError: If the command outruns *timeout* seconds.
            SandboxDiedError: If the container stopped during the command.
            SandboxUnreachableError: If the CLI lost the command's outcome any other way.
        """
        tag = uuid4().hex
        trailer, marker = f'deepgym-exit-{tag}:', f'{_EXEC_DIR}/{tag}'
        argv = ['exec']
        if cwd:
            argv += ['--workdir', cwd]
        for key, value in (env or {}).items():
            argv += ['--env', f'{key}={value}']
        argv += [self.id, '/bin/sh', '-c', _WRAPPER, 'sh', marker, trailer]
        if timeout:
            argv += ['/usr/bin/timeout', '--kill-after=2', str(timeout)]
        argv += ['/bin/sh', '-c', command]
        deadline = timeout or 3600
        start = time.monotonic()
        cause: subprocess.TimeoutExpired | None = None
        try:
            done = self._cli(*argv, timeout=deadline + _WAIT_GRACE_S)
            out, err, ended = done.stdout, done.stderr, f'exit {done.returncode}'
        except subprocess.TimeoutExpired as exc:
            out, err, ended, cause = exc.stdout or b'', exc.stderr or b'', 'no answer', exc
        stdout, stderr = _before(out, trailer), _before(err, trailer)
        if stdout is None or stderr is None:
            why = f'{ended}: {err.decode("utf-8", "replace")}'
            raise self._lost(marker, deadline, out + err, why) from cause
        code, output = stdout[1], stdout[0] + stderr[0]
        if timeout and code in TIMEOUT_EXIT_CODES:
            if time.monotonic() - start >= timeout:
                raise SandboxTimeoutError(f'command exceeded {timeout}s', output)
        return ExecResult(code, output)

    def _lost(
        self, marker: str, deadline: int, printed: bytes, why: str
    ) -> SandboxError | SandboxTimeoutError:
        """Return the error for an exec whose outcome the CLI didn't give, though it gave
        *printed* and ended as *why* says.

        Raises:
            SandboxUnreachableError: If the CLI can't list containers either.
        """
        output = printed.decode('utf-8', 'replace')
        if self._gone():
            return SandboxDiedError(f'Container {self.id} stopped', output)
        running_for = self._running_for(marker)
        if running_for is not None and running_for >= deadline:
            return SandboxTimeoutError(f'command exceeded {deadline}s', output)
        return SandboxUnreachableError(f'{self.cli} lost a command in {self.id}, {why}')

    def _running_for(self, marker: str) -> float | None:
        """Seconds the command *marker* belongs to has run, by the container's clock; None
        if it never started or has ended, or the container can't say."""
        try:
            done = self._cli(
                'exec', self.id, '/bin/sh', '-c', _READ_MARKER, 'sh', marker, timeout=30
            )
            now, started = map(float, done.stdout.split())
        except (subprocess.TimeoutExpired, ValueError):
            return None
        return now - started

    def _gone(self) -> bool:
        """Whether the container no longer runs.

        Raises:
            SandboxUnreachableError: If the CLI can't list containers, so what it reported
                about the command can't be trusted either.
        """
        try:
            return all(box.id != self.id for box in self.running())
        except SandboxError as exc:
            raise SandboxUnreachableError(f'{self.cli} cannot list containers: {exc}') from exc

    def booted_image(self) -> SandboxImage:
        """Return the image the container runs and its digest, as the CLI reports them.

        A failed lookup is the reason there is no digest, never an error.
        """
        try:
            return self._inspect_image()
        except (SandboxError, subprocess.TimeoutExpired, ValueError, LookupError, TypeError) as exc:
            return SandboxImage(self.image, None, f'could not inspect container {self.id}: {exc}')

    def _inspect_image(self) -> SandboxImage:
        """Ask the CLI which image the container runs.

        Raises:
            SandboxError: If the CLI fails.
            ValueError, LookupError, TypeError: If its answer isn't shaped as expected.
        """
        raise NotImplementedError

    @classmethod
    def _answer(cls, *args: str) -> str:
        """Return the stdout of ``<cli> <args>``.

        Raises:
            SandboxError: If the command fails.
        """
        done = cls._cli(*args, timeout=30)
        if done.returncode != 0:
            raise SandboxError(done.stderr.decode('utf-8', 'replace').strip())
        return done.stdout.decode('utf-8', 'replace')

    def upload(self, content: bytes, path: str) -> None:
        """Write *content* to *path* inside the container, creating parent directories.

        Raises:
            SandboxDiedError: If the container has stopped.
            SandboxError: If the write fails any other way.
        """
        script = f'PATH={TRUSTED_PATH}; mkdir -p "$(dirname "$1")" && cat > "$1"'
        try:
            done = self._cli(
                'exec', '--interactive', self.id, '/bin/sh', '-c', script, 'upload', path,
                input=content, timeout=120,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            if self._gone():
                raise SandboxDiedError(f'Container {self.id} stopped uploading {path}') from exc
            raise SandboxError(f'Upload of {path} timed out') from exc
        if done.returncode != 0:
            message = f'Failed to upload {path}: {done.stderr.decode()}'
            raise SandboxDiedError(message) if self._gone() else SandboxError(message)

    def delete(self) -> bool:
        """Stop and remove the container; never raises, and a failure is logged.

        Returns:
            False if removing it failed and it still runs, or the CLI can't list containers.
        """
        try:
            self.remove(self.id)
        except SandboxError as exc:
            with contextlib.suppress(SandboxError):
                if all(box.id != self.id for box in self.running()):
                    return True  # Gone already: --rm, or killed from the console.
            logger.warning('Failed to remove container %s: %s', self.id, exc)
            return False
        return True

    @classmethod
    def remove(cls, name: str) -> None:
        """Force-remove the container *name*, whoever started it.

        Raises:
            SandboxError: If the CLI is missing or the removal fails.
        """
        try:
            done = cls._cli(*cls.delete_args, name, timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f'Removing {name} timed out') from exc
        if done.returncode != 0:
            raise SandboxError(f'Failed to remove {name}: {done.stderr.decode().strip()}')

    @classmethod
    def running(cls) -> list[RunningSandbox]:
        """List live DeepGym containers (named ``deepgym-...``).

        Raises:
            SandboxError: If the CLI is missing or can't list containers.
        """
        try:
            done = cls._cli(*cls.list_args, timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f'{cls.cli} did not list containers within 30s') from exc
        if done.returncode != 0:
            detail = done.stderr.decode().strip()
            raise SandboxError(f'{cls.cli} could not list containers: {detail}')
        listed = cls._parse_listing(done.stdout.decode())
        return [box for box in listed if box.id.startswith(_NAME_PREFIX)]

    @staticmethod
    def _parse_listing(output: str) -> list[RunningSandbox]:
        """Parse the output of ``<cli> <list_args>``."""
        raise NotImplementedError

    @classmethod
    def _cli(
        cls, *args: str, input: bytes | None = None, timeout: float
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                [cls.cli, *args], input=input, capture_output=True, timeout=timeout, check=False
            )
        except FileNotFoundError as exc:
            raise SandboxError(f'{cls.cli} CLI not found') from exc


def _before(stream: bytes, trailer: str) -> tuple[str, int] | None:
    """Return what *stream* held before the wrapper's *trailer*, and the exit code the
    trailer gives; None if the trailer never came. What follows it came from a child the
    command left running, not from the command."""
    head, found, tail = stream.rpartition(trailer.encode())
    code = tail.partition(b'\n')[0]
    if not found or not code.isdigit():
        return None
    return head.decode('utf-8', 'replace'), int(code)
