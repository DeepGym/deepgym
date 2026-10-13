"""Refuse to publish a build that isn't exactly the tagged release.

The publish job runs this after ``python -m build``: ``dist/`` must hold only the tagged
version's wheel and sdist (so a tag that disagrees with the package version, or a stale
artifact, fails), and the wheel must ship the console UI.

    python scripts/check_release.py 0.4.3
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

CONSOLE_INDEX = 'deepgym/console/static/index.html'


def release_problem(version: str, dist: Path) -> str | None:
    """Return why *dist* isn't a clean build of *version*, or None if it is.

    Args:
        version: The release version, the tag without its leading ``v``.
        dist: The directory ``python -m build`` wrote to.

    Returns:
        A message naming the problem, or None.
    """
    wheel = f'deepgym-{version}-py3-none-any.whl'
    expected = {wheel, f'deepgym-{version}.tar.gz'}
    found = {path.name for path in dist.iterdir()}
    if found != expected:
        return f'{dist} holds {sorted(found)}, expected {sorted(expected)}'
    with zipfile.ZipFile(dist / wheel) as archive:
        if CONSOLE_INDEX not in archive.namelist():
            return f'{wheel} has no console UI ({CONSOLE_INDEX})'
    return None


if __name__ == '__main__':
    sys.exit(release_problem(sys.argv[1], Path('dist')))
