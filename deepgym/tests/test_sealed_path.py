"""The in-sandbox check that only root can change what TRUSTED_PATH finds."""

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from deepgym.sandboxes import sealed_path
from deepgym.sandboxes.sealed_path import UnsealedError, check
from deepgym.session import _SEALED_PATH_CHECK


@pytest.fixture()
def image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    """A root-built image under tmp_path, as lstat reports it: tmp_path, the folders above
    it and everything in it are owned by root, with their real types and modes. Map a path
    to a uid in the returned dict to change its owner, or to an OSError to fail its lstat."""
    base = tmp_path.resolve()
    owners: dict[Path, int | OSError] = {}
    real_lstat = os.lstat

    def lstat(path: str) -> os.stat_result:
        owner = owners.get(Path(path), 0)
        if isinstance(owner, OSError):
            raise owner
        if not Path(path).is_relative_to(base):
            return os.stat_result((stat.S_IFDIR | 0o755, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        info = real_lstat(path)
        return os.stat_result((*info[:4], owner, *info[5:]))

    monkeypatch.setattr(sealed_path, 'lstat', lstat)
    usr = base / 'usr'
    (usr / 'bin').mkdir(parents=True)
    (base / 'opt').mkdir()
    for folder in (usr, usr / 'bin', base / 'opt'):
        folder.chmod(0o755)
    (usr / 'bin' / 'date').write_text('#!/bin/sh\n')
    (usr / 'bin' / 'date').chmod(0o755)
    return base, owners


def _layout(case: str, base: Path, owners: dict) -> tuple[list[Path], Path | str | None]:
    """Change the image for *case*; return the folders to check and what the check must
    name as changeable (None if it must pass)."""
    usr, opt = base / 'usr', base / 'opt'
    bin_, date = usr / 'bin', usr / 'bin' / 'date'
    if case == 'missing folder':
        return [usr / 'sbin'], None
    if case == 'linked folder':  # /bin -> usr/bin, as in merged-/usr images
        (base / 'bin').symlink_to('usr/bin')
        return [base / 'bin'], None
    if case == 'dangling link':
        (bin_ / 'gone').symlink_to(opt / 'gone')
    elif case == 'world-writable folder':
        bin_.chmod(0o777)
        return [bin_], bin_
    elif case == 'group-writable folder':
        bin_.chmod(0o775)
        return [bin_], bin_
    elif case == 'folder owned by another user':
        owners[bin_] = 1000
        return [bin_], bin_
    elif case == 'writable folder above':
        usr.chmod(0o777)
        return [bin_], usr
    elif case == 'writable program':
        date.chmod(0o777)
        return [bin_], date
    elif case == 'program owned by another user':
        owners[date] = 1000
        return [bin_], date
    elif case == 'folder linked into a writable folder':
        (opt / 'bin').mkdir(mode=0o755)
        opt.chmod(0o777)
        (base / 'local').symlink_to(opt / 'bin')
        return [base / 'local'], opt
    elif case == 'program linked into a writable folder':
        (opt / 'tool').write_text('#!/bin/sh\n')
        (opt / 'tool').chmod(0o755)
        opt.chmod(0o777)
        (bin_ / 'tool').symlink_to(opt / 'tool')
        return [bin_], opt
    elif case == 'link loop':
        (bin_ / 'loop').symlink_to('loop')
        return [bin_], 'links resolving'
    elif case == 'unreadable entry':
        owners[date] = PermissionError(13, 'Permission denied', str(date))
        return [bin_], 'Permission denied'
    elif case == 'folder that is a file':
        return [date], 'Not a directory'
    elif case == 'path through a program':  # the kernel refuses /usr/bin/date/..
        (base / 'up').symlink_to(f'{date}/..')
        return [base / 'up'], 'is not a folder'
    return [bin_], None


@pytest.mark.parametrize(
    'case',
    [
        'sealed', 'missing folder', 'linked folder', 'dangling link',
        'world-writable folder', 'group-writable folder', 'folder owned by another user',
        'writable folder above', 'writable program', 'program owned by another user',
        'folder linked into a writable folder', 'program linked into a writable folder',
        'link loop', 'unreadable entry', 'folder that is a file', 'path through a program',
    ],
)  # fmt: skip
def test_check_passes_only_what_root_alone_can_change(image: tuple[Path, dict], case: str) -> None:
    """Everything that leads to a program, and the program, must be root's alone; a link is
    followed, and whatever can't be checked fails."""
    folders, changeable = _layout(case, *image)
    if changeable is None:
        check([str(folder) for folder in folders])
        return
    named = re.escape(f'{changeable} is owned by') if isinstance(changeable, Path) else changeable
    with pytest.raises((UnsealedError, OSError), match=named):
        check([str(folder) for folder in folders])


@pytest.mark.parametrize('writable', [False, True], ids=['sealed', 'unsealed'])
def test_script_prints_sealed_or_says_why(
    image: tuple[Path, dict],
    writable: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """The source setup sends runs as ``__main__``: it prints exactly ``sealed`` and exits 0,
    or says why not and exits 1. It sees the image through the same faked lstat."""
    base, _ = image
    bin_ = base / 'usr' / 'bin'
    if writable:
        bin_.chmod(0o777)
    code = 0
    with monkeypatch.context() as patched:
        patched.setattr(os, 'lstat', sealed_path.lstat)  # what the script imports from os
        patched.setattr(sys, 'argv', ['-c', str(bin_), str(base / 'usr' / 'sbin')])
        try:
            exec(compile(_SEALED_PATH_CHECK, '<setup>', 'exec'), {'__name__': '__main__'})
        except SystemExit as exc:
            code = exc.code
    if writable:
        why = f'{bin_} is owned by uid 0 with mode drwxrwxrwx'
        expected = (1, f'root would run what others can change: {why}\n')
    else:
        expected = (0, f'{sealed_path.SEALED}\n')
    assert (code, capsys.readouterr().out) == expected


def test_python_runs_the_script_as_setup_does(tmp_path: Path) -> None:
    """``python3 -I -c <source>`` runs it. A world-writable folder fails on any host,
    whatever the folders above it are, so this holds wherever the tests run."""
    folder = tmp_path / 'bin'
    folder.mkdir()
    folder.chmod(0o777)
    done = subprocess.run(
        [sys.executable, '-I', '-c', _SEALED_PATH_CHECK, str(folder)],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert (done.returncode, done.stderr) == (1, '')
    assert done.stdout.startswith('root would run what others can change: ')
