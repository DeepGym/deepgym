"""The release check refuses any build that isn't exactly the tagged version."""

import sys
import zipfile
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from check_release import CONSOLE_INDEX, release_problem  # noqa: E402


def _build(dist: Path, version: str, *, console: bool = True) -> None:
    """Write what ``python -m build`` would for *version*: a wheel and an sdist."""
    with zipfile.ZipFile(dist / f'deepgym-{version}-py3-none-any.whl', 'w') as wheel:
        wheel.writestr('deepgym/__init__.py', '')
        if console:
            wheel.writestr(CONSOLE_INDEX, '<html></html>')
    (dist / f'deepgym-{version}.tar.gz').write_bytes(b'')


def test_a_clean_build_of_the_tagged_version_passes(tmp_path: Path) -> None:
    _build(tmp_path, '0.4.3')
    assert release_problem('0.4.3', tmp_path) is None


def test_a_tag_that_is_not_the_package_version_fails(tmp_path: Path) -> None:
    _build(tmp_path, '0.4.2')
    assert 'expected' in release_problem('0.4.3', tmp_path)


def test_stale_artifacts_in_dist_fail(tmp_path: Path) -> None:
    _build(tmp_path, '0.4.2')
    _build(tmp_path, '0.4.3')
    assert release_problem('0.4.3', tmp_path) is not None


def test_a_wheel_without_the_console_fails(tmp_path: Path) -> None:
    _build(tmp_path, '0.4.3', console=False)
    assert 'console UI' in release_problem('0.4.3', tmp_path)
