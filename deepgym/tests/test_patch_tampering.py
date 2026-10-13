"""The patch runner refuses diffs that touch test infrastructure outside the hidden tests.

Runs through the sandbox code path (``sandbox_dg``); local mode refuses patch tasks.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.adversarial import PATCH_TAMPERING_ATTACKS, new_files_diff
from deepgym.benchmark_envs import RepoPatchEnvironment

_CALC = 'def add(a, b):\n    return a - b\n'
_FIX = {'calc.py': 'def add(a, b):\n    return a + b\n'}
_BASE_TEST = 'from calc import add\n\n\ndef test_zero():\n    assert add(0, 0) == 0\n'
_HIDDEN_TEST = _BASE_TEST + '\n\ndef test_add():\n    assert add(1, 2) == 3\n'
_PYPROJECT = '[project]\nname = "calc"\nversion = "1.0"\n\n[tool.pytest.ini_options]\n'
_PACKAGE = '{\n  "name": "calc",\n  "version": "1.0.0",\n  "scripts": {"test": "pytest"}\n}\n'


def _git(repo: Path, *args: str) -> str:
    config = ('user.email=t@example.com', 'user.name=t', 'commit.gpgsign=0')
    command = ['git', '-C', str(repo), *(arg for item in config for arg in ('-c', item))]
    return subprocess.run([*command, *args], check=True, capture_output=True).stdout.decode()


def _diff(repo: Path, files: dict[str, str]) -> str:
    """Return the diff that writes *files* into *repo*, leaving the repo unchanged."""
    for name, content in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(content)
    _git(repo, 'add', '-A')
    diff = _git(repo, 'diff', '--cached')
    _git(repo, 'reset', '--hard', '-q')
    _git(repo, 'clean', '-fdq')
    return diff


def _env(tmp_path: Path, extra: dict[str, str] | None = None) -> RepoPatchEnvironment:
    """A repo whose hidden test patch adds a check only a real fix of add() passes."""
    repo = tmp_path / 'repo'
    files = {'calc.py': _CALC, 'tests/test_calc.py': _BASE_TEST, **(extra or {})}
    for name, content in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(content)
    _git(repo, 'init', '-q')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-qm', 'base')
    pytest_cmd = f'PYTHONPATH=. "{sys.executable}" -m pytest -q tests/test_calc.py'
    return RepoPatchEnvironment(
        task='Fix add().',
        repo_dir=str(repo),
        test_command=f'cd "{repo}" && {pytest_cmd}',
        test_patch=_diff(repo, {'tests/test_calc.py': _HIDDEN_TEST}),
        timeout=120,
    )


@pytest.mark.parametrize('attack', sorted(PATCH_TAMPERING_ATTACKS))
def test_test_infra_diff_is_refused(tmp_path: Path, sandbox_dg: DeepGym, attack: str) -> None:
    env = _env(tmp_path)
    # A real fix doesn't excuse the tampering.
    diff = _diff(Path(env.repo_dir), _FIX) + new_files_diff(PATCH_TAMPERING_ATTACKS[attack])

    result = sandbox_dg.run(env, f'```diff\n{diff}```')

    assert result.score == 0.0
    assert result.error_type == 'test_infra_tampering'
    assert result.has_verdict
    assert result.cases and all(not case.passed for case in result.cases)


def test_source_only_fix_scores_full(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    env = _env(tmp_path)

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), _FIX))

    assert result.score == 1.0
    assert result.error_type is None


def test_repo_with_its_own_conftest_still_scores(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    conftest = 'import pytest\n\n\n@pytest.fixture\ndef two():\n    return 2\n'
    env = _env(tmp_path, {'conftest.py': conftest})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), _FIX))

    assert result.score == 1.0
    assert result.error_type is None


@pytest.mark.parametrize(
    ('extra', 'edit', 'refused'),
    [
        pytest.param(
            {'pyproject.toml': _PYPROJECT},
            {'pyproject.toml': _PYPROJECT.replace('"1.0"', '"1.1"')},
            False,
            id='pyproject-version',
        ),
        pytest.param(
            {'pyproject.toml': _PYPROJECT},
            {'pyproject.toml': _PYPROJECT + 'addopts = "-q"\n'},
            True,
            id='pyproject-pytest-section',
        ),
        pytest.param(
            {'package.json': _PACKAGE},
            {'package.json': _PACKAGE.replace('1.0.0', '1.1.0')},
            False,
            id='package-version',
        ),
        pytest.param(
            {'package.json': _PACKAGE},
            {'package.json': _PACKAGE.replace('"pytest"', '"pytest -q"')},
            True,
            id='package-scripts',
        ),
    ],
)
def test_config_edits_outside_the_test_section_are_allowed(
    tmp_path: Path,
    sandbox_dg: DeepGym,
    extra: dict[str, str],
    edit: dict[str, str],
    refused: bool,
) -> None:
    env = _env(tmp_path, extra)

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), {**_FIX, **edit}))

    assert result.error_type == ('test_infra_tampering' if refused else None)
    assert result.score == (0.0 if refused else 1.0)
