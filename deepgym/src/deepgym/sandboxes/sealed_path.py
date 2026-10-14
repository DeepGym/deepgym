"""Prove that only root can change what root runs through a PATH, from inside the sandbox.

Run as root with ``python3 -I -c <this file's source> folder...``; stdlib only, so any image's
Python runs it. Every folder, each folder above it, every entry in it, and every path a link
among them leads through must be owned by root and closed to group and other writes. An ACL
that lets anyone else write shows as group write, so it fails too. A link is followed, not
refused (``/bin`` links to ``usr/bin`` in most images): the folder holding it and everything
it leads through are checked. A missing folder passes, as only root can create one in a
folder that passed.

It prints :data:`SEALED` and exits 0, or prints why not and exits 1. Anything it can't check
(an entry it can't stat, a link loop, a folder that is a file) fails, never passes.
"""

from __future__ import annotations

import os
import stat
import sys
from os import listdir, lstat, readlink
from os.path import dirname, join

SEALED = 'sealed'
_MAX_LINKS = 40


class UnsealedError(Exception):
    """Someone besides root could change this path."""


def _checked(path: str) -> os.stat_result | None:
    """Return *path*'s own lstat, None if it doesn't exist; raise unless only root can
    change it. A link's own owner and mode don't matter, only its folder and target."""
    try:
        info = lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISLNK(info.st_mode) and (info.st_uid != 0 or info.st_mode & 0o022):
        mode = stat.filemode(info.st_mode)
        raise UnsealedError(f'{path} is owned by uid {info.st_uid} with mode {mode}')
    return info


def resolve(path: str, links: int = 0) -> str | None:
    """Return where absolute *path* really leads, or None if that doesn't exist, after
    checking ``/``, every component on the way, and every link's target the same way."""
    if links > _MAX_LINKS:
        raise UnsealedError(f'more than {_MAX_LINKS} links resolving {path}')
    _checked('/')
    real, is_dir = '/', True
    for name in path.split('/'):
        if not is_dir:  # as the kernel: nothing, not even '.', '..' or '/', follows a file
            raise NotADirectoryError(f'{real} is not a folder, in {path}')
        if name in ('', '.'):
            continue
        if name == '..':
            real = dirname(real)
            continue
        here = join(real, name)
        info = _checked(here)
        if info is None:
            return None
        if stat.S_ISLNK(info.st_mode):
            here = resolve(join(real, readlink(here)), links + 1)
            if here is None:
                return None
            info = lstat(here)
        real, is_dir = here, stat.S_ISDIR(info.st_mode)
    return real


def check(folders: list[str]) -> None:
    """Raise unless only root can change *folders*, what's in them and what leads to them.

    Raises:
        UnsealedError: If something besides root could change one of them.
        OSError: If a path can't be checked.
    """
    for folder in folders:
        real = resolve(folder)
        if real is not None:
            for name in sorted(listdir(real)):
                resolve(join(real, name))


if __name__ == '__main__':
    try:
        check(sys.argv[1:])
    except (OSError, UnsealedError) as exc:
        print(f'root would run what others can change: {exc}')
        sys.exit(1)
    print(SEALED)
