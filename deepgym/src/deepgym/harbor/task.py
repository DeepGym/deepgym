"""One Harbor task folder, read and validated.

A task is ``instruction.md``, ``task.toml``, ``tests/test.sh`` (writes a reward under
``/logs/verifier/``), an optional ``solution/solve.sh``, and either a prebuilt
``[environment].docker_image`` or an ``environment/Dockerfile``
(https://docs.harborframework.com/tasks/overview). DeepGym reads single-step Linux
tasks and ignores task.toml fields it doesn't use.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import os
import re
import stat
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    ValidationInfo,
    field_serializer,
    field_validator,
)

from deepgym.models import fingerprinting
from deepgym.session import check_workdir

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

NetworkMode = Literal['public', 'no-network', 'allowlist']

STAGING_DIR = '/opt/deepgym-harbor'
"""Where the agent's script and the tests' archive are uploaded."""

HARNESS_PATHS = ('/logs', '/tests', '/solution', STAGING_DIR)
"""Folders the harness writes; a work directory must neither hold nor sit in one."""

# Harbor drops leading markdown comments (canary strings) from the instruction.
_LEADING_COMMENTS = re.compile(r'\A(?:\s*<!--.*?-->)+\s*', re.DOTALL)
# A [verifier].env name tests/test.sh can get: a shell identifier, matched whole.
_ENV_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def pack_files(files: Mapping[str, bytes]) -> bytes:
    """Return *files* (relative path -> content) as a gzipped tar, every file mode 755.

    The same files give the same bytes on any Python with the same zlib: the gzip header's
    time is 0, not now, and the level is pinned (Python 3.15 lowers the default).
    """
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode='wb', compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode='w') as archive,
    ):
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class _Table(BaseModel):
    """A task.toml table: DeepGym validates the fields it reads and ignores the rest."""

    model_config = ConfigDict(extra='ignore', frozen=True)


class _EnvironmentTable(_Table):
    docker_image: str | None = None
    workdir: str | None = None
    network_mode: NetworkMode | None = None
    allow_internet: bool | None = None
    os: Literal['linux'] = 'linux'

    @property
    def network(self) -> NetworkMode | None:
        """``network_mode``, or what the deprecated ``allow_internet`` maps to."""
        if self.network_mode is not None or self.allow_internet is None:
            return self.network_mode
        return 'public' if self.allow_internet else 'no-network'


class _AgentTable(_Table):
    timeout_sec: float | None = Field(default=None, gt=0)


class _VerifierTable(_Table):
    timeout_sec: float = Field(default=600, gt=0)
    network_mode: NetworkMode | None = None
    environment: _EnvironmentTable | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator('env')
    @classmethod
    def checked_env(cls, value: dict[str, str]) -> dict[str, str]:
        """Refuse variables tests/test.sh can't get as written.

        A name must be a shell identifier: ``env`` would read ``-S`` as an option, and
        ``A=B``, a space or an empty name as an odd assignment. ``${VAR}`` values read host
        variables, which never go into a sandbox, and no argument can hold a NUL.
        """
        names = sorted(name for name in value if not _ENV_NAME.fullmatch(name))
        if names:
            raise ValueError(
                '[verifier].env names must be letters, digits and _, not starting with a '
                f'digit: {names}'
            )
        templated = sorted(name for name, text in value.items() if '${' in text)
        if templated:
            raise ValueError(f'[verifier].env reads host variables {templated}; not supported')
        nul = sorted(name for name, text in value.items() if '\0' in text)
        if nul:
            raise ValueError(f'[verifier].env values hold a NUL character: {nul}')
        return value


class _TaskToml(_Table):
    environment: _EnvironmentTable = Field(default_factory=_EnvironmentTable)
    agent: _AgentTable = Field(default_factory=_AgentTable)
    verifier: _VerifierTable = Field(default_factory=_VerifierTable)
    steps: list[dict] | None = None

    @property
    def verifier_network(self) -> NetworkMode:
        """The tests' network: the [verifier] override, else the verifier's own sandbox's
        baseline, else the task's, else Harbor's default ``public``."""
        separate = self.verifier.environment
        return (
            self.verifier.network_mode
            or (separate.network if separate else None)
            or self.environment.network
            or 'public'
        )


class HarborTask(BaseModel):
    """A validated Harbor task folder, with its tests and solution read into memory.

    Example:
        >>> task = HarborTask.from_dir(Path('tasks/hello-world'))
        >>> task.docker_image, sorted(task.tests)
        ('ubuntu:24.04', ['test.sh', 'test_outputs.py'])
    """

    # Test and solution files may be binary, so JSON carries them as URL-safe base64 (the
    # alphabet with - and _), and reads them back the same way (see _files_from_json).
    model_config = ConfigDict(frozen=True, ser_json_bytes='base64')

    name: str = Field(min_length=1)
    """The folder's name, unique within its dataset."""

    path: Path
    """The task folder; its ``environment/`` is the image's build context."""

    instruction: str
    """What the agent is asked to do: instruction.md without leading comments."""

    docker_image: str | None = None
    """Prebuilt image to boot; None means build ``environment/Dockerfile``."""

    workdir: str | None = None
    """Where the agent works and tests run; None means the image's WORKDIR."""

    agent_timeout_sec: float | None = Field(default=None, gt=0)
    """Seconds the agent may run; None means task.toml sets no limit."""

    verifier_timeout_sec: float = Field(default=600, gt=0)
    """Seconds tests/test.sh may run."""

    verifier_network: NetworkMode = 'public'
    """Network tests/test.sh may use."""

    verifier_env: dict[str, str] = Field(default_factory=dict)
    """Variables task.toml's ``[verifier].env`` gives tests/test.sh, on top of the basics."""

    tests: dict[str, bytes]
    """Files under tests/ by relative path, uploaded to /tests only for grading."""

    solution: dict[str, bytes] = Field(default_factory=dict)
    """Files under solution/ by relative path (solve.sh is the reference); may be empty."""

    @field_serializer('path', when_used='json')
    def _path_json(self, path: Path, info: SerializationInfo) -> str | None:
        """The folder; in a fingerprint (:data:`~deepgym.models.FINGERPRINT`), the hash of
        the build context when the task builds its image, else None: where the folder is
        doesn't change the task, and its tests and solution are fields of their own.

        Raises:
            ValueError: If the build context links outside the task folder.
        """
        if not fingerprinting(info):
            return str(path)
        return None if self.docker_image else _tree_sha256(self.build_context)

    @field_validator('workdir')
    @classmethod
    def workdir_is_safe(cls, value: str | None) -> str | None:
        """Reject a work directory the agent mustn't own (see ``check_workdir``)."""
        return None if value is None else check_workdir(value, HARNESS_PATHS)

    @field_validator('tests', 'solution', mode='before')
    @classmethod
    def _files_from_json(cls, value: object, info: ValidationInfo) -> object:
        """Decode files from JSON, where they are URL-safe base64 with padding, as written.

        Only JSON input: from Python a str stays pydantic's default, its UTF-8 bytes, so a
        text script can be passed as a string.

        Raises:
            ValueError: If a file in JSON isn't valid URL-safe base64.
        """
        if info.mode != 'json' or not isinstance(value, dict):
            return value
        return {
            name: base64.b64decode(text, altchars=b'-_', validate=True)
            if isinstance(text, str)
            else text
            for name, text in value.items()
        }

    @field_validator('tests')
    @classmethod
    def has_test_script(cls, value: dict[str, bytes]) -> dict[str, bytes]:
        """Require tests/test.sh, the verifier's entry point."""
        if 'test.sh' not in value:
            raise ValueError('tests/test.sh is missing')
        return value

    @field_validator('solution')
    @classmethod
    def has_solve_script(cls, value: dict[str, bytes]) -> dict[str, bytes]:
        """Require solve.sh in a non-empty solution/."""
        if value and 'solve.sh' not in value:
            raise ValueError('solution/ has no solve.sh')
        return value

    @property
    def dockerfile(self) -> Path | None:
        """``environment/Dockerfile`` if the task has one."""
        path = self.path / 'environment' / 'Dockerfile'
        return path if path.is_file() else None

    @property
    def build_context(self) -> Path:
        """``environment/``, resolved, once it is known to stay inside the task folder.

        Raises:
            ValueError: If ``environment/`` is a link, or a link inside it leads outside it.
        """
        return _build_context(self.path)

    @classmethod
    def from_dir(cls, path: Path) -> HarborTask:
        """Read and validate the task folder at *path*.

        Raises:
            ValueError: If a required file is missing, task.toml doesn't parse, a field
                is invalid, a file links outside the folder, or the task is multi-step.
        """
        try:
            config = _TaskToml.model_validate(tomllib.loads(_read(path, 'task.toml').decode()))
            instruction = _read(path, 'instruction.md').decode()
            if config.steps or (path / 'steps').is_dir():
                raise ValueError('multi-step tasks are not supported')
            task = cls(
                name=path.name,
                path=path,
                instruction=_LEADING_COMMENTS.sub('', instruction),
                docker_image=config.environment.docker_image,
                workdir=config.environment.workdir,
                agent_timeout_sec=config.agent.timeout_sec,
                verifier_timeout_sec=config.verifier.timeout_sec,
                verifier_network=config.verifier_network,
                verifier_env=config.verifier.env,
                tests=_read_tree(path, 'tests'),
                solution=_read_tree(path, 'solution'),
            )
            context = path / 'environment'
            if context.is_symlink() or context.exists():
                _build_context(path)
            return task
        except (OSError, ValueError) as exc:
            raise ValueError(f'Invalid Harbor task {path}: {exc}') from exc


def _build_context(task_dir: Path) -> Path:
    """Return *task_dir*'s ``environment/`` resolved, if it stays inside the folder.

    A builder sends the whole context to the image and follows a link at its root, so a
    context that is a link, or holds one leading outside it, could copy host files into
    the image.

    Raises:
        ValueError: If ``environment/`` is a link, or a link inside it leads outside it.
    """
    context = task_dir / 'environment'
    if context.is_symlink():
        raise ValueError('environment/ is a link; it must be a folder in the task')
    real = context.resolve()
    for path in context.rglob('*'):
        if path.is_symlink() and not path.resolve().is_relative_to(real):
            raise ValueError(f'{path.relative_to(task_dir)} links outside environment/')
    return real


_HASHED_MODE = 0o7000 | 0o555 | 0o002
"""Mode bits the build-folder hash keeps: setuid, setgid and sticky, read and execute
for everyone, and write for others. Owner and group write are left out: they are what a
machine's umask (022 or 002) changes, and a task's files are owned by root in the image."""


def _tree_sha256(root: Path) -> str:
    """Hash the folder at *root* as a builder sends it: every entry under it by its path
    relative to *root*, its type and mode bits, a file by its content, a link by
    its target (never followed), and a folder, empty or not, by its path and mode alone.
    The mode keeps every bit but owner and group write (see :data:`_HASHED_MODE`), so a
    lost read bit, a world-writable file or a setuid one changes the hash and a different
    umask doesn't. A missing folder hashes as empty."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):  # doesn't descend into linked folders
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            content = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
        elif stat.S_ISREG(mode):
            content = _file_sha256(path)
        else:
            content = ''
        # A path never holds NUL, and content is empty or 64 hex digits: entries can't run together.
        kept = stat.S_IFMT(mode) | (mode & _HASHED_MODE)
        entry = f'{path.relative_to(root).as_posix()}\0{kept:o}\0{content}\n'
        digest.update(entry.encode('utf-8', 'surrogateescape'))  # any file name hashes
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash the file at *path*, a megabyte at a time."""
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _read(task_dir: Path, rel: str) -> bytes:
    """Read *rel* in *task_dir*, refusing a symlink that leads outside the folder.

    A dataset is untrusted: a link to a host file would carry it into a sandbox or a prompt.
    """
    path = task_dir / rel
    if not path.resolve().is_relative_to(task_dir.resolve()):
        raise ValueError(f'{rel} links outside the task folder')
    return path.read_bytes()


def _read_tree(task_dir: Path, rel: str) -> dict[str, bytes]:
    """Read every file under *task_dir*/*rel* by path relative to it (empty if absent)."""
    root = task_dir / rel
    return {
        path.relative_to(root).as_posix(): _read(task_dir, path.relative_to(task_dir).as_posix())
        for path in sorted(root.rglob('*'))
        if path.is_file()
    }
