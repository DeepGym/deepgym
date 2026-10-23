"""Find Harbor tasks in a local folder, a git repository or a Hugging Face dataset.

Remote datasets are shallow-cloned once into ``~/.deepgym/benchmarks/harbor/``, one folder
per URL, and reused only while git confirms the folder is a clone of that URL.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from deepgym.exceptions import DeepGymError
from deepgym.harbor.task import HarborTask

_CLONE_TIMEOUT_S = 1800
_HF_DATASET = re.compile(r'\w[\w.-]*/[\w.-]+')


def dataset_root(source: str | Path) -> Path:
    """Return a local folder holding *source*'s tasks.

    Args:
        source: A local folder, a git URL (GitHub, GitLab, or
            ``https://huggingface.co/datasets/org/name``), or ``org/name`` for a Hugging
            Face dataset.

    Raises:
        DeepGymError: If *source* is none of those, cloning it fails, or its cache folder
            isn't a clone of it.
    """
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve()
    url = str(source)
    if '://' not in url and not url.startswith('git@'):
        if not _HF_DATASET.fullmatch(url):
            raise DeepGymError(f'{url!r} is not a folder, a git URL or a Hugging Face dataset')
        url = f'https://huggingface.co/datasets/{url}'
    target = _cache_dir() / _cache_name(url)
    if not target.is_dir():
        _clone(url, target)
    if _cloned_from(target) != url:
        raise DeepGymError(f'{target} is not a clone of {url}; delete it to clone again')
    return target


def task_dirs(root: Path) -> list[Path]:
    """Return the task folders (each holds a task.toml) under *root*, sorted by path.

    Raises:
        DeepGymError: If two task folders share a name, which is what tasks go by.
    """
    tomls = root.rglob('task.toml')
    dirs = sorted({toml.parent for toml in tomls if '.git' not in toml.relative_to(root).parts})
    clashes = [name for name, count in Counter(d.name for d in dirs).items() if count > 1]
    if clashes:
        raise DeepGymError(f'Task folders in {root} share names: {sorted(clashes)}')
    return dirs


def load_tasks(source: str | Path, limit: int | None = None) -> list[HarborTask]:
    """Load and validate the tasks in *source* (see :func:`dataset_root`), at most *limit*.

    Raises:
        DeepGymError: If *source* can't be found or cloned, or holds no task.
        ValueError: If a task folder is invalid.
    """
    dirs = task_dirs(dataset_root(source))[:limit]
    if not dirs:
        raise DeepGymError(f'No Harbor tasks (folders with a task.toml) in {source}')
    return [HarborTask.from_dir(task_dir) for task_dir in dirs]


def _cache_dir() -> Path:
    """Where remote datasets are cloned."""
    return Path.home() / '.deepgym' / 'benchmarks' / 'harbor'


def _cache_name(url: str) -> str:
    """The cache folder name for *url*: readable, and keyed by a digest of the whole URL.

    The readable part alone isn't unique: ``host/team_one/task`` and ``host/team/one_task``
    both read ``host_team_one_task``.
    """
    readable = re.sub(r'[^\w.-]+', '_', url.split('://')[-1])
    return f'{readable}-{hashlib.sha256(url.encode()).hexdigest()[:16]}'


def _cloned_from(clone: Path) -> str | None:
    """The URL *clone* was cloned from, as git recorded it; None if it isn't a git clone."""
    try:
        done = subprocess.run(
            ['git', '--git-dir', clone / '.git', 'config', '--local', '--get', 'remote.origin.url'],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise DeepGymError(f'Could not read where {clone} was cloned from: {exc}') from exc
    return done.stdout.strip() if done.returncode == 0 else None


def _clone(url: str, target: Path) -> None:
    """Shallow-clone *url* to *target*, atomically: a failed clone leaves nothing behind.

    Raises:
        DeepGymError: If git is missing, fails, or takes too long.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        clone = Path(tmp) / 'repo'
        try:
            done = subprocess.run(
                ['git', 'clone', '--depth', '1', '--', url, str(clone)],
                capture_output=True,
                text=True,
                timeout=_CLONE_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DeepGymError(f'Could not clone {url}: {exc}') from exc
        if done.returncode != 0:
            raise DeepGymError(f'Could not clone {url}: {done.stderr.strip()[-500:]}')
        try:
            clone.rename(target)
        except OSError as exc:
            # Another process may have cloned it first; its copy is as good as ours.
            if not target.is_dir():
                raise DeepGymError(f'Could not move the clone of {url} into place: {exc}') from exc
