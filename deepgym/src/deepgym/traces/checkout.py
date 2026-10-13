"""Which deepgym code an episode ran: the git checkout the package runs from, if any."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1]
_IMPORTED_AT = time.time()


def git_checkout(package: Path, since: float | None = None) -> tuple[str | None, bool | None]:
    """Return the git commit *package* is checked out at, and whether its files differ from it.

    The package must be tracked by the repository around it, so an install inside some
    other project's checkout (a virtualenv in it, say) never reports that project's commit.
    ``GIT_*`` variables are dropped, so they can't point git at another repository, and
    git takes no optional locks, so it never blocks a commit running at the same time.

    Args:
        package: The package folder.
        since: A time; a tracked file modified after it also counts as a difference.

    Returns:
        ``(commit, dirty)``; ``(None, None)`` when *package* isn't tracked in a git
        checkout, or git can't tell.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ['git', '--no-optional-locks', '-C', str(package), *args],
                capture_output=True, text=True, timeout=10, env=env, check=False,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    tracked = git('ls-files', '-z', '--', '.')
    if not tracked:
        return None, None
    head = git('rev-parse', '--verify', 'HEAD')
    status = git('status', '--porcelain', '--untracked-files=normal', '--', '.')
    if head is None or status is None:
        return None, None
    names = filter(None, tracked.split('\0'))
    touched = since is not None and any(_modified_after(package / name, since) for name in names)
    return head.strip(), bool(status.strip()) or touched


def _modified_after(path: Path, since: float) -> bool:
    try:
        return path.lstat().st_mtime > since
    except OSError:
        return True


def deepgym_checkout() -> tuple[str | None, bool | None]:
    """Return the commit of the deepgym checkout this code runs from, and whether it's dirty.

    Dirty means deepgym's files differ from the commit, or changed after deepgym was
    imported: code loaded before that keeps running, so they may no longer show it.

    Returns:
        ``(commit, dirty)``, or ``(None, None)`` when not running from a git checkout.
    """
    return git_checkout(_PACKAGE, since=_IMPORTED_AT)
