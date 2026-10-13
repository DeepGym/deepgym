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
_PYPROJECT_QUOTED_PLUGIN = _PYPROJECT.replace(
    '[tool.pytest.ini_options]', '[tool."pytest".ini_options]\naddopts = "-p dg_exit"'
)
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


_CARGO = (
    '[package]\nname = "calc"\nversion = "1.0.0"\n\n'
    '[lib]\ntest = true\n\n[[bin]]\nname = "calc"\ntest = false\n'
)
# Same lines, swapped between targets: the failing lib tests stop running.
_CARGO_SWAPPED = (
    '[package]\nname = "calc"\nversion = "1.0.0"\n\n'
    '[lib]\ntest = false\n\n[[bin]]\nname = "calc"\ntest = true\n'
)
# Disabling auto-discovery drops hidden integration tests without touching a target table.
_CARGO_NO_AUTOTESTS = _CARGO.replace(
    'version = "1.0.0"\n', 'version = "1.0.0"\nautotests = false\n'
)
# A real fix may add or bump a dependency; that is not test tampering.
_PACKAGE_DEP_BUMP = _PACKAGE.replace(
    '"scripts": {"test": "pytest"}\n',
    '"scripts": {"test": "pytest"},\n  "dependencies": {"left-pad": "^1.3.0"}\n',
)
# Repointing a runner dependency to a local stub needs the stub package to declare a bin;
# that stub package.json is what gets caught (its own bin key), even if the root edit is allowed.
_STUB_PACKAGE = '{\n  "name": "stub",\n  "version": "1.0.0",\n  "bin": {"jest": "./fake.js"}\n}\n'
_SETUP_CFG = '[metadata]\nname = calc\n\n[tool:pytest]\npython_files = test_*.py\n'
# pytest reads option names case-sensitively; ConfigParser lowercases them unless told not to.
_SETUP_CFG_CASE = _SETUP_CFG.replace('python_files', 'PYTHON_FILES')
# A %(...)s value raises during interpolation; the comparison must fail closed, not crash.
_SETUP_CFG_INTERP = _SETUP_CFG + 'addopts = -p %(missing)s\n'
# Repointing workspace selection skips the failing package without touching a target table.
_CARGO_WS = '[workspace]\nmembers = ["core", "decoy"]\ndefault-members = ["core"]\n'
_CARGO_WS_EDIT = _CARGO_WS.replace('["core"]', '["decoy"]')


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
            {'pyproject.toml': _PYPROJECT + 'addopts = "-p dg_exit"\n'},
            True,
            id='pyproject-pytest-section',
        ),
        pytest.param(
            # A quoted table spelling tomllib normalizes but line/section text matching misses.
            {'pyproject.toml': _PYPROJECT},
            {'pyproject.toml': _PYPROJECT_QUOTED_PLUGIN},
            True,
            id='pyproject-quoted-section',
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
        pytest.param(
            # Swapping which target runs tests leaves the line set unchanged; parsing catches it.
            {'Cargo.toml': _CARGO},
            {'Cargo.toml': _CARGO_SWAPPED},
            True,
            id='cargo-target-swap',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO},
            {'Cargo.toml': _CARGO_NO_AUTOTESTS},
            True,
            id='cargo-autotests-off',
        ),
        pytest.param(
            {'package.json': _PACKAGE},
            {'package.json': _PACKAGE_DEP_BUMP},
            False,
            id='package-dep-bump-allowed',
        ),
        pytest.param(
            # The stub package a runner repoint points at must declare a bin; that is caught.
            {'package.json': _PACKAGE},
            {'package.json': _PACKAGE, 'stub/package.json': _STUB_PACKAGE},
            True,
            id='package-stub-bin',
        ),
        pytest.param(
            {'setup.cfg': _SETUP_CFG},
            {'setup.cfg': _SETUP_CFG.replace('name = calc', 'name = calculator')},
            False,
            id='setup-cfg-metadata',
        ),
        pytest.param(
            {'setup.cfg': _SETUP_CFG},
            {'setup.cfg': _SETUP_CFG_CASE},
            True,
            id='setup-cfg-key-case',
        ),
        pytest.param(
            {'setup.cfg': _SETUP_CFG},
            {'setup.cfg': _SETUP_CFG_INTERP},
            True,
            id='setup-cfg-interpolation-fails-closed',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_WS},
            {'Cargo.toml': _CARGO_WS_EDIT},
            True,
            id='cargo-workspace-members',
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


@pytest.mark.parametrize(
    ('name', 'refused'),
    [
        ('.npmrc', True),
        ('pytest.toml', True),
        ('.cargo/config.toml', True),
        # Build files and lockfiles are often part of a real fix, so they are not refused
        # (a tampered `make test` or stubbed dependency is caught by the per-test report gate).
        ('Makefile', False),
        ('package-lock.json', False),
        ('go.sum', False),
        ('scripts/latest.sh', False),
    ],
)
def test_launcher_configs_refused_build_files_allowed(
    tmp_path: Path, sandbox_dg: DeepGym, name: str, refused: bool
) -> None:
    # .npmrc / pytest.toml / .cargo/config.toml select or configure the test runner, so a new
    # one is refused; a Makefile, lockfile or unrelated shell script is left to a real fix.
    env = _env(tmp_path)

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), {name: 'x = 1\n', **_FIX}))

    assert result.error_type == ('test_infra_tampering' if refused else None)
    assert result.score == (0.0 if refused else 1.0)


def test_data_fixture_in_test_dir_is_allowed(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    # A real fix may change a data fixture under a test directory; only executable code there
    # is refused. (Weakening a data oracle is a separate problem the report gate addresses.)
    env = _env(tmp_path)
    edit = {'tests/data/expected.yml': 'v: 2\n', **_FIX}

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), edit))

    assert result.error_type is None
    assert result.score == 1.0


def test_production_package_named_test_is_refused_known_false_positive(
    tmp_path: Path, sandbox_dg: DeepGym
) -> None:
    # Known stopgap false positive: a fix under a production package literally named ``test``
    # is refused without running. The per-test report gate removes this by running the tests.
    env = _env(tmp_path, {'app/test/client.py': 'VALUE = 1\n'})
    edit = {'app/test/client.py': 'VALUE = 2\n', **_FIX}

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), edit))

    assert result.error_type == 'test_infra_tampering'


def test_crafted_and_symlinked_paths_never_forge(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    env = _env(tmp_path)
    # A symlink of a plain (non-infra) name: caught only by the mode/symlink rule, not the
    # filename denylist. ./ and // paths git rejects outright; a quoted non-ASCII conftest we
    # refuse by basename. None may forge a pass.
    symlink_hunk = (
        'diff --git a/shim.py b/shim.py\nnew file mode 120000\n'
        '--- /dev/null\n+++ b/shim.py\n@@ -0,0 +1 @@\n+payload\n'
        '\\ No newline at end of file\n'
    )
    # 0120000 and a trailing space are non-canonical symlink spellings git still accepts.
    symlink_noncanonical = (
        'diff --git a/shim2.py b/shim2.py\nnew file mode 0120000 \n'
        '--- /dev/null\n+++ b/shim2.py\n@@ -0,0 +1 @@\n+payload\n'
        '\\ No newline at end of file\n'
    )
    crafted = {
        'symlink': symlink_hunk,
        'symlink_noncanonical': symlink_noncanonical,
        'dot_slash': new_files_diff({'./conftest.py': 'import os\nos._exit(0)\n'}),
        'double_slash': new_files_diff({'pkg//conftest.py': 'import os\nos._exit(0)\n'}),
        'quoted': new_files_diff({'tëst_dir/conftest.py': 'import os\nos._exit(0)\n'}),
    }
    for label, diff in crafted.items():
        result = sandbox_dg.run(env, diff)
        assert result.score == 0.0, label
        assert not result.passed, label


def test_symlink_refusal_names_the_mode_reason(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    env = _env(tmp_path)
    symlink_hunk = (
        'diff --git a/shim.py b/shim.py\nnew file mode 120000\n'
        '--- /dev/null\n+++ b/shim.py\n@@ -0,0 +1 @@\n+payload\n'
        '\\ No newline at end of file\n'
    )

    result = sandbox_dg.run(env, symlink_hunk)

    assert result.error_type == 'test_infra_tampering'
    assert result.cases and any('symlink' in (case.error or '') for case in result.cases)


def test_qa_marks_unappliable_tampering_attack_inconclusive(
    tmp_path: Path, sandbox_dg: DeepGym, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A repo that already ships conftest.py makes the conftest attack fail to apply; that
    # says nothing about the guard, so QA must report it inconclusive, never resisted.
    from deepgym import adversarial

    one_attack = {'conftest': {'conftest.py': 'x\n'}}
    monkeypatch.setattr(adversarial, 'PATCH_TAMPERING_ATTACKS', one_attack)
    env = _env(tmp_path, {'conftest.py': 'import pytest\n'})

    report = adversarial.AdversarialTester(sandbox_dg).test(env, ['patch_tampering'])

    assert report.results[0].inconclusive
    assert not report.is_robust
