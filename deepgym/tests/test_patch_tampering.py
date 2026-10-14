"""The patch runner refuses diffs that touch test infrastructure outside the hidden tests,
reports an environment, repo or task it finds broken before the patch without a verdict,
and scores every outcome after the patch.

Runs through the sandbox code path (``sandbox_dg``); local mode refuses patch tasks.
"""

import itertools
import json
import shutil
import subprocess
import sys
import sysconfig
import time
import types
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.adversarial import PATCH_TAMPERING_ATTACKS, new_files_diff
from deepgym.benchmark_envs import (
    _TESTS_STARTED,
    PatchVerifier,
    RepoPatchEnvironment,
    SWEBenchProEnvironment,
    _parse_runner_output,
)
from deepgym.exceptions import VerifierError
from deepgym.models import VerifierResult, is_verdict
from tests.conftest import RunnerSandbox

_CALC = 'def add(a, b):\n    return a - b\n'
_FIX = {'calc.py': 'def add(a, b):\n    return a + b\n'}
_BASE_TEST = 'from calc import add\n\n\ndef test_zero():\n    assert add(0, 0) == 0\n'
_HIDDEN_TEST = _BASE_TEST + '\n\ndef test_add():\n    assert add(1, 2) == 3\n'
_PYPROJECT_PLAIN = '[project]\nname = "calc"\nversion = "1.0"\n'
_PYPROJECT = _PYPROJECT_PLAIN + '\n[tool.pytest.ini_options]\n'
_PYPROJECT_QUOTED_PLUGIN = _PYPROJECT.replace(
    '[tool.pytest.ini_options]', '[tool."pytest".ini_options]\naddopts = "-p dg_exit"'
)
_PACKAGE = '{\n  "name": "calc",\n  "version": "1.0.0",\n  "scripts": {"test": "pytest"}\n}\n'


def _git(repo: Path, *args: str) -> str:
    config = ('user.email=t@example.com', 'user.name=t', 'commit.gpgsign=0')
    command = ['git', '-C', str(repo), *(arg for item in config for arg in ('-c', item))]
    return subprocess.run([*command, *args], check=True, capture_output=True).stdout.decode()


def _diff(repo: Path, files: dict[str, str | None]) -> str:
    """Return the diff that writes *files* into *repo* (None deletes one), leaving it unchanged."""
    for name, content in files.items():
        if content is None:
            (repo / name).unlink()
            continue
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


def _runner() -> dict:
    """The patch runner's functions, exec'd from the script the sandbox runs."""
    from deepgym.benchmark_envs import PatchVerifier

    namespace: dict = {'__name__': 'deepgym_patch_runner'}
    exec(compile(PatchVerifier.build_runner_script(), 'runner', 'exec'), namespace)
    return namespace


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
# Repointing a runner dependency to a local stub needs the stub package to declare a bin; a
# nested package's bin is uncertain, so a pass that stub forges is masked, never rewarded.
_STUB_PACKAGE = '{\n  "name": "stub",\n  "version": "1.0.0",\n  "bin": {"jest": "./fake.js"}\n}\n'
# The root package's own bin and its scripts: `test` runs `unit`, `build` is unrelated.
_PACKAGE_CLI = (
    '{\n  "name": "calc",\n  "bin": {"calculator": "./dist/main.js"},\n'
    '  "scripts": {"test": "npm run unit", "unit": "pytest", "build": "tsc"}\n}\n'
)
_SETUP_CFG = '[metadata]\nname = calc\n\n[tool:pytest]\npython_files = test_*.py\n'
# pytest reads option names case-sensitively; ConfigParser lowercases them unless told not to.
_SETUP_CFG_CASE = _SETUP_CFG.replace('python_files', 'PYTHON_FILES')
# A %(...)s value raises during interpolation; the comparison must fail closed, not crash.
_SETUP_CFG_INTERP = _SETUP_CFG + 'addopts = -p %(missing)s\n'
# A fix may correct a console script; only the pytest11 entry-point group loads pytest plugins.
_SETUP_CFG_ENTRY = (
    '[metadata]\nname = calc\n\n[options.entry_points]\n'
    'console_scripts =\n    calc = calc.cli:old_main\n'
)
# Repointing workspace selection skips the failing package without touching a target table.
_CARGO_WS = '[workspace]\nmembers = ["core", "decoy"]\ndefault-members = ["core"]\n'
_CARGO_WS_EDIT = _CARGO_WS.replace('["core"]', '["decoy"]')

# Codex review of the hand-written scanner (14da7b0); each must stay fixed with the real parser.
_PY_QUIET = '[project]\nname = "calc"\n\n[tool.pytest.ini_options]\naddopts = "-q"\n'
# 1: an inline table is pytest config too.
_PY_INLINE = 'tool = { pytest = { ini_options = { addopts = "-k passing" } } }\n' + _PYPROJECT_PLAIN
# 2: a unicode-escaped quoted key names the same table.
_PY_UNICODE = _PY_QUIET.replace('[tool.pytest.ini_options]', '[tool."pyte\\u0073t".ini_options]')
_PY_UNICODE = _PY_UNICODE.replace('-q', '-k passing')
# 3: a header-looking line inside a multiline string must not switch tables.
_PY_MULTILINE = '[tool.pytest.ini_options]\nmarkers = """\n[project]\n"""\naddopts = "-q"\n'
# 4: array order matters (filterwarnings is applied in order).
_PY_WARN = '[tool.pytest.ini_options]\nfilterwarnings = ["ignore::UserWarning", "error"]\n'
_PY_WARN_SWAPPED = '[tool.pytest.ini_options]\nfilterwarnings = ["error", "ignore::UserWarning"]\n'
_CARGO_TWO_TESTS = (
    '[package]\nname = "calc"\nversion = "1.0.0"\n\n'
    '[[test]]\nname = "a"\nrequired-features = ["x"]\n\n[[test]]\nname = "b"\n'
)
_CARGO_TWO_TESTS_MOVED = (
    '[package]\nname = "calc"\nversion = "1.0.0"\n\n'
    '[[test]]\nname = "a"\n\n[[test]]\nname = "b"\nrequired-features = ["x"]\n'
)
# 5: emptying a default feature disables a [[test]] that requires it.
_CARGO_FEATURE = (
    '[package]\nname = "calc"\nversion = "1.0.0"\n\n[features]\ndefault = ["validation"]\n'
    'validation = []\n\n[[test]]\nname = "t"\nrequired-features = ["validation"]\n'
)
# 6: package metadata is not test configuration.
_CARGO_METADATA = (
    '[package]\nname = "calc"\nversion = "1.0.0"\n\n[package.metadata]\nrelease-notes = "a"\n'
)
# A [dependencies] bump can be the fix; dev- and build-dependencies only build tests and build.rs.
_CARGO_DEPS = (
    '[package]\nname = "calc"\nversion = "1.0.0"\n\n[dependencies]\nserde = "1.0.100"\n\n'
    '[dev-dependencies]\nproptest = "1.0.0"\n\n[build-dependencies]\ncc = "1.0.0"\n'
)
# Members inheriting version.workspace = true are released by bumping [workspace.package];
# [workspace.dependencies], shared by every member, stays compared.
_CARGO_WS_PACKAGE = (
    '[workspace]\nmembers = ["core"]\n\n[workspace.package]\nversion = "0.9.0"\n'
    'edition = "2021"\n\n[workspace.dependencies]\nserde = "1.0.100"\n'
)
# Example and bench targets run under `cargo test` when their test flag is on.
_CARGO_TARGETS = (
    '[package]\nname = "calc"\nversion = "1.0.0"\n\n'
    '[[example]]\nname = "demo"\ntest = false\n\n[[bench]]\nname = "speed"\ntest = false\n'
)
# tox runs its [testenv] commands, wherever the tox config sits.
_TOX_TOML = '[env_run_base]\ncommands = [["pytest"]]\n'
_PY_TOX_LEGACY = _PYPROJECT_PLAIN + (
    '\n[tool.tox]\nlegacy_tox_ini = """\n[testenv]\ncommands = pytest\n"""\n'
)
_PY_TOX_NATIVE = _PYPROJECT_PLAIN + '\n[tool.tox.env_run_base]\ncommands = [["pytest"]]\n'
_PY_ENTRY = _PYPROJECT_PLAIN + '\n[project.scripts]\ncalc = "calc.cli:old_main"\n'
_PACKAGE_WORKSPACES = _PACKAGE.replace('"version"', '"workspaces": ["packages/*"],\n  "version"')


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
            # A top-level dotted key reads as pytest config to a real parser; refused.
            {'pyproject.toml': _PYPROJECT_PLAIN},
            {'pyproject.toml': 'tool.pytest.ini_options.addopts = "-p dg"\n' + _PYPROJECT_PLAIN},
            True,
            id='pyproject-dotted-key',
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
            {'setup.cfg': _SETUP_CFG_ENTRY},
            {'setup.cfg': _SETUP_CFG_ENTRY.replace('old_main', 'main')},
            False,
            id='setup-cfg-console-scripts-allowed',
        ),
        pytest.param(
            {'setup.cfg': _SETUP_CFG_ENTRY},
            {'setup.cfg': _SETUP_CFG_ENTRY + 'pytest11 =\n    dg = dg_plugin\n'},
            True,
            id='setup-cfg-pytest11',
        ),
        pytest.param(
            {'pyproject.toml': _PY_ENTRY},
            {'pyproject.toml': _PY_ENTRY.replace('old_main', 'main')},
            False,
            id='pyproject-scripts-allowed',
        ),
        pytest.param(
            {'pyproject.toml': _PY_ENTRY},
            {'pyproject.toml': _PY_ENTRY + '\n[project.entry-points.pytest11]\ndg = "dg_plugin"\n'},
            True,
            id='pyproject-pytest11',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_WS},
            {'Cargo.toml': _CARGO_WS_EDIT},
            True,
            id='cargo-workspace-members',
        ),
        pytest.param(
            {'pyproject.toml': _PYPROJECT_PLAIN},
            {'pyproject.toml': _PY_INLINE},
            True,
            id='codex1-inline-table',
        ),
        pytest.param(
            {'pyproject.toml': _PY_QUIET},
            {'pyproject.toml': _PY_UNICODE},
            True,
            id='codex2-unicode-escaped-header',
        ),
        pytest.param(
            {'pyproject.toml': _PY_MULTILINE},
            {'pyproject.toml': _PY_MULTILINE.replace('-q', '-k passing')},
            True,
            id='codex3-header-inside-multiline-string',
        ),
        pytest.param(
            {'pyproject.toml': _PY_WARN},
            {'pyproject.toml': _PY_WARN_SWAPPED},
            True,
            id='codex4-filterwarnings-order',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_TWO_TESTS},
            {'Cargo.toml': _CARGO_TWO_TESTS_MOVED},
            True,
            id='codex4-required-features-moved',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_FEATURE},
            {'Cargo.toml': _CARGO_FEATURE.replace('default = ["validation"]', 'default = []')},
            True,
            id='codex5-test-feature-disabled',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_METADATA},
            {'Cargo.toml': _CARGO_METADATA.replace('"a"', '"b"')},
            False,
            id='codex6-package-metadata-allowed',
        ),
        pytest.param(
            {'pyproject.toml': _PY_QUIET},
            {'pyproject.toml': _PY_QUIET.replace('"-q"', '"-q"  # quieter output')},
            False,
            id='codex7-comment-only-allowed',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_DEPS},
            {'Cargo.toml': _CARGO_DEPS.replace('1.0.100', '1.0.200')},
            False,
            id='cargo-dependency-bump-allowed',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_DEPS},
            {'Cargo.toml': _CARGO_DEPS.replace('proptest = "1.0.0"', 'proptest = "1.1.0"')},
            True,
            id='cargo-dev-dependencies',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_DEPS},
            {'Cargo.toml': _CARGO_DEPS.replace('cc = "1.0.0"', 'cc = "1.1.0"')},
            True,
            id='cargo-build-dependencies',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_WS_PACKAGE},
            {'Cargo.toml': _CARGO_WS_PACKAGE.replace('0.9.0', '0.9.1')},
            False,
            id='cargo-workspace-package-version-allowed',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_WS_PACKAGE},
            {'Cargo.toml': _CARGO_WS_PACKAGE.replace('1.0.100', '1.0.200')},
            True,
            id='cargo-workspace-dependencies',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_TARGETS},
            {'Cargo.toml': _CARGO_TARGETS.replace('"demo"\ntest = false', '"demo"\ntest = true')},
            True,
            id='cargo-example-test-flag',
        ),
        pytest.param(
            {'Cargo.toml': _CARGO_TARGETS},
            {'Cargo.toml': _CARGO_TARGETS.replace('"speed"\ntest = false', '"speed"\ntest = true')},
            True,
            id='cargo-bench-test-flag',
        ),
        pytest.param(
            {'tox.toml': _TOX_TOML},
            {'tox.toml': _TOX_TOML.replace('"pytest"', '"true"')},
            True,
            id='tox-toml-commands',
        ),
        pytest.param(
            {'tox.ini': '[testenv]\ncommands = pytest\n'},
            {'tox.ini': '[testenv]\ncommands = true\n'},
            True,
            id='tox-ini-commands',
        ),
        pytest.param(
            {'pyproject.toml': _PY_TOX_LEGACY},
            {'pyproject.toml': _PY_TOX_LEGACY.replace('commands = pytest', 'commands = true')},
            True,
            id='pyproject-legacy-tox-commands',
        ),
        pytest.param(
            {'pyproject.toml': _PY_TOX_NATIVE},
            {'pyproject.toml': _PY_TOX_NATIVE.replace('"pytest"', '"true"')},
            True,
            id='pyproject-tox-env-commands',
        ),
        pytest.param(
            {'package.json': _PACKAGE_WORKSPACES},
            {'package.json': _PACKAGE_WORKSPACES.replace('packages/*', 'packages/core')},
            True,
            id='package-workspaces',
        ),
        pytest.param(
            # A new go.mod below the root splits off a module `go test ./...` then skips.
            {},
            {'pkg/go.mod': 'module example.com/pkg\n'},
            True,
            id='nested-go-mod-added',
        ),
        pytest.param(
            {'pkg/go.mod': 'module example.com/pkg\n'},
            {'pkg/go.mod': None},
            True,
            id='nested-go-mod-removed',
        ),
        pytest.param(
            {'pkg/go.mod': 'module example.com/pkg\n\ngo 1.21\n'},
            {'pkg/go.mod': 'module example.com/pkg\n\ngo 1.22\n'},
            False,
            id='nested-go-mod-edit-allowed',
        ),
        pytest.param(
            {},
            {'go.mod': 'module example.com/calc\n'},
            False,
            id='root-go-mod-added-allowed',
        ),
    ],
)
def test_config_edits_outside_the_test_section_are_allowed(
    tmp_path: Path,
    sandbox_dg: DeepGym,
    extra: dict[str, str],
    edit: dict[str, str | None],
    refused: bool,
) -> None:
    env = _env(tmp_path, extra)

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), {**_FIX, **edit}))

    assert result.error_type == ('test_infra_tampering' if refused else None)
    assert result.score == (0.0 if refused else 1.0)


@pytest.mark.parametrize(
    ('name', 'refused'),
    [
        # Build scripts and runner config run code, so `make test` / a gradle task / build.rs
        # could be rewritten to exit 0 without running the tests: refused.
        ('.npmrc', True),
        ('pytest.toml', True),
        ('.cargo/config.toml', True),
        ('Makefile', True),
        ('pom.xml', True),
        ('build.rs', True),
        ('vendor/github.com/x/y.go', True),
        # A lockfile or go.sum cannot forge a pass on its own (the stub a repoint needs is
        # caught by its bin), and a real dependency fix updates them: allowed.
        ('package-lock.json', False),
        ('go.sum', False),
        ('scripts/latest.sh', False),
        # tox config and go.work pick what `tox` / `go test` run: refused.
        ('tox.toml', True),
        ('go.work', True),
        # Only what a runner itself treats as a test or its config is a test file: Go tests
        # are *_test.go, so cmd/test.go is code; so is a doc named after a tool. The tools'
        # own setup and config files are not.
        ('cmd/test.go', False),
        ('src/spec.go', False),
        ('calc_test.go', True),
        ('docs/jest.md', False),
        ('jest.setup.js', True),
        ('vitest.offline.config.ts', True),
        ('src/setupTests.ts', True),
        # A setup file runs inside the test process: refused when named for a test tool or
        # kept in a test directory, but an ordinary *.setup.* module is code.
        ('src/test.setup.ts', True),
        ('e2e/auth.setup.ts', True),
        ('tests/global.setup.ts', True),
        ('src/i18n.setup.ts', False),
    ],
)
def test_paths_refused_by_name(
    tmp_path: Path, sandbox_dg: DeepGym, name: str, refused: bool
) -> None:
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


def test_vendored_toml_parser_matches_the_tools() -> None:
    # The inlined parser (what a pre-3.11 image uses) must read dotted keys and reject
    # duplicates exactly like tomllib, or a dotted-key pytest config would slip past.
    from deepgym.benchmark_envs import _vendored_toml_source

    namespace: dict = {}
    exec(compile(_vendored_toml_source(), 'vendored', 'exec'), namespace)
    loads, decode_error = namespace['loads'], namespace['TOMLDecodeError']

    parsed = loads('tool.pytest.ini_options.addopts = "-p evil"\n')
    assert parsed['tool']['pytest']['ini_options']['addopts'] == '-p evil'
    inline = loads('[tool]\npytest = {ini_options = {addopts = "x"}}')
    assert inline['tool']['pytest']['ini_options']['addopts'] == 'x'
    with pytest.raises(decode_error):
        loads('[a]\nx = 1\nx = 2\n')
    with pytest.raises(decode_error):
        loads('[a]\n[a]\n')


def test_config_comparison_needs_no_tomllib(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    # The runner executes in the task image's Python, where tomllib (3.11+) may be absent.
    # The TOML comparison must carry a vendored parser so a 3.10 image does not wrongly
    # refuse a legitimate pyproject edit. Guard against losing the vendored fallback.
    from deepgym.benchmark_envs import PatchVerifier

    script = PatchVerifier.build_runner_script()
    assert '_VENDORED_TOML' in script
    assert 'def loads' in script  # the inlined parser's entry point

    env = _env(tmp_path, {'pyproject.toml': _PYPROJECT})
    # A version bump outside the pytest section is allowed even with no TOML library present.
    edit = {'pyproject.toml': _PYPROJECT.replace('"1.0"', '"2.0"'), **_FIX}

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), edit))

    assert result.error_type is None
    assert result.score == 1.0


def test_vendored_toml_source_needs_no_getsource(monkeypatch: pytest.MonkeyPatch) -> None:
    # A compiled or sourceless tomllib/tomli has no source for inspect.getsource, and a host
    # may have neither module: the parser ships as package data instead.
    import inspect

    from deepgym.benchmark_envs import PatchVerifier, _vendored_toml_source

    def no_source(obj: object) -> str:
        raise OSError('could not get source code')

    monkeypatch.setattr(inspect, 'getsource', no_source)
    monkeypatch.setitem(sys.modules, 'tomllib', None)
    monkeypatch.setitem(sys.modules, 'tomli', None)

    namespace: dict = {}
    exec(compile(_vendored_toml_source(), 'vendored', 'exec'), namespace)

    assert namespace['loads']('[tool.pytest]\naddopts = "-q"\n') == {
        'tool': {'pytest': {'addopts': '-q'}}
    }
    assert '_VENDORED_TOML' in PatchVerifier.build_runner_script()


@pytest.mark.parametrize(
    ('target', 'fails', 'error_type'),
    [
        pytest.param('test_patch', False, 'validation_timeout', id='test-patch-timeout'),
        pytest.param('model_patch', False, 'validation_timeout', id='model-patch-timeout'),
        pytest.param('git_root', False, 'validation_timeout', id='git-root-timeout'),
        # Guessing the top-level instead could check the wrong, missing files and grade a
        # patch that edits the real ones.
        pytest.param('git_root', True, 'validation_error', id='git-root-unknown'),
    ],
)
def test_validation_faults_have_no_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    target: str,
    fails: bool,
    error_type: str,
) -> None:
    # git stalling or failing while the runner lists paths or finds the repo top-level is the
    # runner's failure, not the model's: no verdict, so trainers mask it rather than grade it.
    env = _env(tmp_path)
    patch_file = tmp_path / 'candidate.diff'
    patch_file.write_text(_diff(Path(env.repo_dir), _FIX))
    task_file = tmp_path / 'task.json'
    task = {'repo_dir': env.repo_dir, 'test_patch': env.test_patch, 'timeout': 60}
    task_file.write_text(json.dumps({**task, 'test_commands': [env.test_command]}))
    hit = {
        'test_patch': lambda args: '--numstat' in args and args[-1].endswith('test_patch.diff'),
        'model_patch': lambda args: '--numstat' in args and args[-1] == str(patch_file),
        'git_root': lambda args: 'rev-parse' in args,
    }[target]
    real_run = subprocess.run

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if not hit(args):
            return real_run(args, **kwargs)
        if fails:
            return subprocess.CompletedProcess(args, 128, b'', b'fatal: not a git repository')
        raise subprocess.TimeoutExpired(args, 60)

    runner = _runner()
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['runner', str(task_file), str(patch_file)])

    exit_code = runner['main']()

    result = VerifierResult.model_validate_json(capsys.readouterr().out)
    assert result.error_type == error_type
    assert result.score == 0.0
    assert not is_verdict(exit_code, result.error_type)


def test_patch_paths_have_one_spelling(tmp_path: Path) -> None:
    # Every comparison uses git's -z path through normpath: ./ci.ini, sub/../ci.ini and ci.ini/
    # are all ci.ini. A path that leaves the repo has no such form; it stays raw, and the
    # runner treats it as uncertain.
    runner = _runner()
    env = _env(tmp_path, {'ci.ini': '[pytest]\n'})
    patch_file = tmp_path / 'spellings.diff'
    patch_file.write_text(
        ''.join(
            f'diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n'
            '@@ -1 +1,2 @@\n [pytest]\n+# edited\n'
            for name in ('./ci.ini', 'sub/../ci.ini', 'ci.ini/', '../ci.ini')
        )
    )

    paths = runner['_patch_paths'](Path(env.repo_dir), patch_file, 60)

    assert paths == ['ci.ini', '../ci.ini']
    assert [runner['_normalize'](path) for path in paths] == ['ci.ini', None]


def test_unlisted_workspace_package_key_has_no_verdict(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    # Only known metadata in [workspace.package] is a plain release bump; edition (or any key
    # off that list) could change how members build, so the edit is uncertain and the pass it
    # gets here is masked.
    env = _env(tmp_path, {'Cargo.toml': _CARGO_WS_PACKAGE})
    edit = {'Cargo.toml': _CARGO_WS_PACKAGE.replace('2021', '2024'), **_FIX}

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), edit))

    assert result.error_type == 'validation_uncertain'
    assert not result.has_verdict


# (error_type, score, has_verdict) of a run graded on its tests, masked, or failed with a verdict.
_OUTCOMES = {
    'graded': (None, 1.0, True),
    'masked': ('validation_uncertain', 0.0, False),
    'failed': (None, 0.0, True),
    'refused': ('test_infra_tampering', 0.0, True),
}
_EDITED_INI = '[pytest]\n# edited\n'


@pytest.mark.parametrize(
    ('command', 'edit', 'outcome'),
    [
        # A file the test command mentions may be what it loads. The tests still run: a failure
        # is the normal, trained 0, and only a pass, which the edit may have caused, is masked.
        pytest.param('{pytest} -c ci.ini', {'ci.ini': _EDITED_INI}, 'masked', id='tests-pass'),
        # So is a run the edit stopped (pytest then collects nothing, exit 5).
        pytest.param(
            '{pytest} -c ci.ini',
            {'ci.ini': '[pytest]\naddopts = -k nothing\n'},
            'failed',
            id='tests-fail',
        ),
        # A compile step (gcc -c src/calc.c; here wc -c calc.py) mentions the fixed file itself.
        pytest.param('wc -c calc.py && {pytest}', {}, 'masked', id='compile-step-fix'),
        pytest.param(
            'wc -c calc.py && {pytest}',
            {'calc.py': _CALC.replace('-', '*')},
            'failed',
            id='compile-step-wrong-fix',
        ),
        # -p names a plugin by its module; ci.ini stays mentioned however the shell splits the
        # lines or changes directory (sub/ci.ini by its basename).
        pytest.param(
            '{pytest} -p localplugin', {'localplugin.py': 'V = 2\n'}, 'masked', id='module'
        ),
        pytest.param(
            'mkdir -p build\n{pytest} -c ci.ini', {'ci.ini': _EDITED_INI}, 'masked', id='multiline'
        ),
        pytest.param(
            '(cd sub && true); {pytest} -c ci.ini',
            {'ci.ini': _EDITED_INI},
            'masked',
            id='subshell-cd',
        ),
        pytest.param(
            '(cd sub && true); {pytest} -c ci.ini',
            {'sub/ci.ini': _EDITED_INI},
            'masked',
            id='subshell-cd-sub',
        ),
        # The command may spell the path another way; ci.ini is still mentioned.
        pytest.param(
            '{pytest} -c ./sub/../ci.ini', {'ci.ini': _EDITED_INI}, 'masked', id='spelled-path'
        ),
        # A value attached to a short flag still names the file (pytest takes -cci.ini).
        pytest.param('{pytest} -cci.ini', {'ci.ini': _EDITED_INI}, 'masked', id='attached-value'),
        # calc.py inside tests/test_calc.py is no mention: a plain fix is graded.
        pytest.param('{pytest} -c ci.ini -p localplugin', {}, 'graded', id='fix-only'),
        # A basename behind a folder that may be its own still counts: a cwd-relative ./ci.ini,
        # a spelling that normalizes to sub/ci.ini, or a partial path after a cd.
        pytest.param(
            '(cd sub && cat ./ci.ini); {pytest}',
            {'sub/ci.ini': _EDITED_INI},
            'masked',
            id='cwd-relative',
        ),
        pytest.param(
            'cat sub/./ci.ini && {pytest}',
            {'sub/ci.ini': _EDITED_INI},
            'masked',
            id='spelled-folder',
        ),
        pytest.param(
            '(cd deep && cat sub/ci.ini); {pytest}',
            {'deep/sub/ci.ini': _EDITED_INI},
            'masked',
            id='partial-path',
        ),
        # A basename behind another folder names a different file (NodeBB tests its
        # src/socket.io/posts.js with test/posts.js): the fix is graded.
        pytest.param(
            'echo test/posts.js && {pytest}',
            {'src/socket.io/posts.js': 'module.exports = {};\n'},
            'graded',
            id='basename-in-another-folder',
        ),
    ],
)
def test_files_the_test_command_mentions_are_uncertain(
    tmp_path: Path, sandbox_dg: DeepGym, command: str, edit: dict[str, str], outcome: str
) -> None:
    files = {'ci.ini': '[pytest]\n', 'sub/ci.ini': '[pytest]\n', 'localplugin.py': 'V = 1\n'}
    env = _env(tmp_path, files)
    run_tests = f'PYTHONPATH=. "{sys.executable}" -m pytest -q tests/test_calc.py'
    env = env.model_copy(update={'test_command': command.format(pytest=run_tests)})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), {**_FIX, **edit}))

    assert (result.error_type, result.score, result.has_verdict) == _OUTCOMES[outcome]


@pytest.mark.parametrize(
    ('edit', 'outcome'),
    [
        # The root package's own bin (correcting calculator's entry point) and an unrelated
        # script are a fix's business; a script named for tests is refused.
        pytest.param(
            {'package.json': _PACKAGE_CLI.replace('main.js', 'cli.js')}, 'graded', id='root-bin'
        ),
        pytest.param(
            {'package.json': _PACKAGE_CLI.replace('"tsc"', '"tsc -p ."')},
            'graded',
            id='other-script',
        ),
        pytest.param(
            {'package.json': _PACKAGE_CLI.replace('"npm run unit"', '"true"')},
            'refused',
            id='test-script',
        ),
        # A script the test script runs by name, or a nested package's bin (a stub a repointed
        # dependency puts on the PATH), may be the runner: uncertain, so a pass is masked.
        pytest.param(
            {'package.json': _PACKAGE_CLI.replace('"pytest"', '"pytest -q"')},
            'masked',
            id='script-a-test-runs',
        ),
        pytest.param({'stub/package.json': _STUB_PACKAGE}, 'masked', id='nested-bin'),
    ],
)
def test_package_json_scripts_and_bin(
    tmp_path: Path, sandbox_dg: DeepGym, edit: dict[str, str], outcome: str
) -> None:
    env = _env(tmp_path, {'package.json': _PACKAGE_CLI})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), {**_FIX, **edit}))

    assert (result.error_type, result.score, result.has_verdict) == _OUTCOMES[outcome]


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


@pytest.mark.parametrize(('name', 'refused'), [('bin/calc', False), ('conftest.py', True)])
def test_exec_bit_flip_is_refused_only_on_test_infra(
    tmp_path: Path, sandbox_dg: DeepGym, name: str, refused: bool
) -> None:
    # Restoring an executable's 100755 is an ordinary fix; on a test file or test config the
    # flip is refused with the file itself (git lists a mode-only change).
    env = _env(tmp_path, {'bin/calc': '#!/bin/sh\n', 'conftest.py': ''})
    flip = f'diff --git a/{name} b/{name}\nold mode 100644\nnew mode 100755\n'

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), _FIX) + flip)

    assert result.error_type == ('test_infra_tampering' if refused else None)
    assert result.score == (0.0 if refused else 1.0)


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


# A harness that can't run, found before the patch, has no verdict; after it, all is scored.

_PYTEST = f'PYTHONPATH=. "{sys.executable}" -m pytest -q'


@pytest.mark.parametrize(
    'command',
    [
        # Each would exit 127, 126 or 1 (no module) before running a test, whatever the patch.
        pytest.param('no-such-test-runner -q', id='runner-not-on-path'),
        pytest.param('cd tests && no-such-test-runner', id='runner-after-cd'),
        pytest.param('true | no-such-test-runner', id='runner-in-pipe'),
        pytest.param('true # setup\nno-such-test-runner', id='runner-after-a-comment'),
        pytest.param(f'"{sys.executable}" -m no_such_test_module', id='module-not-importable'),
        pytest.param(f'"{sys.executable}" -u -m no_such_test_module', id='module-after-flags'),
        pytest.param('/opt/no-such-dir/runner', id='absolute-path-missing'),
        pytest.param('/dev/null', id='absolute-path-not-executable'),
    ],
)
def test_environment_that_cannot_run_the_tests_has_no_verdict(
    tmp_path: Path, sandbox_dg: DeepGym, command: str
) -> None:
    # The preflight, before the patch is applied: a missing runner, toolchain or module is the
    # environment's fault, so trainers mask the sample.
    env = _env(tmp_path).model_copy(update={'test_command': command})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), _FIX))

    assert (result.error_type, result.has_verdict) == ('preflight_failed', False)


@pytest.mark.parametrize(
    'command',
    [
        pytest.param(f'{_PYTEST} tests/test_calc.py', id='pytest'),
        pytest.param(
            f'cd tests && PYTHONPATH=.. "{sys.executable}" -m pytest -q test_calc.py', id='after-cd'
        ),
        pytest.param(f'true > /dev/null 2>&1 && {_PYTEST} tests/test_calc.py', id='redirects'),
        pytest.param(f'2>/dev/null {_PYTEST} tests', id='leading-redirect'),
        pytest.param(f'case x in x) true ;; esac; {_PYTEST} tests', id='case'),
        pytest.param(f'for f in tests; do echo "$f"; done; {_PYTEST} tests', id='loop'),
        pytest.param(f'PATH="./bin:$PATH" bin-runner 2>&-; {_PYTEST} tests', id='path-change'),
        pytest.param(f'./run.sh 2>/dev/null; {_PYTEST} tests/test_calc.py', id='repo-script'),
        # Commands that may never run, and shell context a static lookup can't see (Codex review).
        pytest.param(f'if false; then no-such-test-runner; fi; {_PYTEST} tests', id='branch'),
        pytest.param(
            f'if command -v opt-runner >/dev/null; then opt-runner; else {_PYTEST} tests; fi',
            id='optional-runner',
        ),
        pytest.param(f'no-such-test-runner || {_PYTEST} tests', id='fallback'),
        pytest.param(f'run_tests() {{ {_PYTEST} tests; }}; run_tests', id='function'),
        pytest.param(f'export PYTHONPATH=.; "{sys.executable}" -m pytest -q tests', id='export'),
        pytest.param(f'PYTHONPATH="$PWD" "{sys.executable}" -m pytest -q tests', id='computed'),
        pytest.param(f'PYTHONPATH=. "{sys.executable}" -u -m pytest -q tests', id='python-flags'),
        pytest.param(f'no-such-test-runner ||\n{_PYTEST} tests', id='fallback-on-next-line'),
        pytest.param(f'{_PYTEST} -k "" tests', id='empty-argument'),
        pytest.param(f'{_PYTEST} \\\n  tests', id='line-continuation'),
        pytest.param(
            f'PYTHONPATH=. "{sys.executable}" -m "$(printf pytest)" -q tests', id='computed-module'
        ),
    ],
)
def test_preflight_passes_runnable_commands(
    tmp_path: Path, sandbox_dg: DeepGym, command: str
) -> None:
    # Builtins, loops, redirection targets and repo paths are no programs to look up, and what
    # follows a PATH change is looked up elsewhere; the fix is graded.
    env = _env(tmp_path).model_copy(update={'test_command': command})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), _FIX))

    assert (result.score, result.error_type) == (1.0, None)


@pytest.mark.parametrize(
    'setup',
    [
        pytest.param('{venv} &&', id='and-list'),
        pytest.param('{venv} || exit 1;', id='or-list'),
        pytest.param('if true; then {venv}; fi;', id='compound'),
    ],
)
def test_preflight_passes_a_runner_the_command_creates(
    tmp_path: Path, sandbox_dg: DeepGym, setup: str
) -> None:
    # The venv's python does not exist until the command's own setup runs (Codex review), so
    # only what runs before anything else must be found up front.
    venv_dir = tmp_path / 'venv'
    venv = f'"{sys.executable}" -m venv --without-pip "{venv_dir}"'
    site = sysconfig.get_paths()['purelib']
    tests = f'PYTHONPATH=".:{site}" "{venv_dir}/bin/python" -m pytest -q tests/test_calc.py'
    env = _env(tmp_path).model_copy(update={'test_command': f'{setup.format(venv=venv)} {tests}'})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), _FIX))

    assert (result.score, result.error_type) == (1.0, None)


_BROKEN_UNTIL_FIXED = 'import sys\nimport calc\nif calc.add(1, 2) != 3:\n    sys.exit(1)\n'


@pytest.mark.parametrize(
    ('files', 'command', 'fix'),
    [
        # `python -m pkg.check` is found by its top-level package, never by importing pkg's
        # __init__, where the bug the patch fixes may live.
        pytest.param(
            {'checkpkg/__init__.py': _BROKEN_UNTIL_FIXED, 'checkpkg/check.py': ''},
            f'PYTHONPATH=. "{sys.executable}" -m checkpkg.check',
            _FIX,
            id='package-init',
        ),
        # Nor is the repo put on the probing python's path, where its sitecustomize would run.
        pytest.param(
            {'sitecustomize.py': _BROKEN_UNTIL_FIXED},
            f'{_PYTEST} tests',
            _FIX,
            id='sitecustomize',
        ),
    ],
)
def test_preflight_runs_no_repo_code(
    tmp_path: Path, sandbox_dg: DeepGym, files: dict, command: str, fix: dict
) -> None:
    # Code the patch fixes must not run before it (Codex review): it would fail the preflight
    # and mask every answer, right ones included.
    env = _env(tmp_path, files).model_copy(update={'test_command': command})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), fix))

    assert (result.score, result.error_type) == (1.0, None)


def test_timeout_kills_what_the_command_started(tmp_path: Path) -> None:
    # A child the timed-out command left behind would otherwise run on into the next test
    # command (Codex review).
    flag = tmp_path / 'late'

    proc = _runner()['_run'](f'(sleep 1.5; touch "{flag}") & sleep 30', cwd=tmp_path, timeout=1)

    time.sleep(1)
    assert proc.returncode == -9
    assert not flag.exists()


@pytest.mark.parametrize('verdict', ['', '{broken}', '{}', '{"score": 1.0, "passed": true}'])
def test_runner_stopped_after_the_tests_started_scores_zero(verdict: str) -> None:
    # After the tests-started line, anything but a valid verdict its exit code backs (here exit
    # 1, which can't back a pass) is the patch's doing (Codex review).
    output = f'{_TESTS_STARTED}\n{verdict}\nmark '

    executed = _parse_runner_output(output, 'mark', 1)

    assert executed.verifier_result.error_type == 'policy_sandbox_failure'
    assert (executed.verifier_result.score, executed.exit_code) == (0.0, 1)


@pytest.mark.parametrize(
    ('command', 'edit'),
    [
        # The environment ran the command before the patch, so whatever happens after it is the
        # patch's to own: no tests collected, a usage error, 127, a kill, any exit code.
        pytest.param('{pytest} tests/test_calc.py -k nothing', {}, id='pytest-5-none-collected'),
        pytest.param('{pytest} tests/test_gone.py', {}, id='pytest-4-usage-error'),
        pytest.param("sh -c 'exit 127'", {}, id='127'),
        pytest.param('exit 137', {}, id='137'),
        pytest.param('kill -KILL $$', {}, id='sigkill'),
        pytest.param('exit 3', {}, id='3'),
        # Test output that is not UTF-8 once crashed the runner.
        pytest.param("printf '\\377\\n'; exit 1", {}, id='not-utf8'),
        pytest.param(
            "printf 'from calc import sub\\n' > tests/test_sub.py && {pytest} tests/test_sub.py",
            {},
            id='pytest-2-collection-error',
        ),
        # Code under test that forces what looks like a harness failure (Codex review).
        pytest.param('{pytest} tests', {'calc.py': 'import os\nos._exit(127)\n'}, id='code-127'),
        pytest.param(
            '{pytest} tests',
            {'calc.py': 'import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n'},
            id='code-sigterms',
        ),
        pytest.param('{pytest} tests', {'calc.py': 'raise SystemExit(1)\n'}, id='code-pytest-3'),
    ],
)
def test_failures_after_the_patch_are_scored(
    tmp_path: Path, sandbox_dg: DeepGym, command: str, edit: dict[str, str]
) -> None:
    env = _env(tmp_path)
    env = env.model_copy(update={'test_command': command.format(pytest=_PYTEST)})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), edit or _FIX))

    assert (result.score, result.error_type, result.has_verdict) == (0.0, None, True)


def _run_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    runner: dict,
    task: dict,
) -> tuple[int, VerifierResult]:
    """Run *runner*'s main() in this process on _env's repo, the fix and *task*.

    Returns:
        Its exit code and its verdict, the last line it printed.
    """
    env = _env(tmp_path)
    patch_file = tmp_path / 'candidate.diff'
    patch_file.write_text(_diff(Path(env.repo_dir), _FIX))
    task_file = tmp_path / 'task.json'
    base = {'repo_dir': env.repo_dir, 'test_patch': env.test_patch, 'apply_credit': 0.0}
    task_file.write_text(json.dumps({**base, **task}))
    monkeypatch.setattr(sys, 'argv', ['runner', str(task_file), str(patch_file)])
    exit_code = runner['main']()
    return exit_code, VerifierResult.model_validate_json(capsys.readouterr().out.splitlines()[-1])


@pytest.mark.parametrize(
    ('task', 'outcome'),
    [
        # A test the patch made hang: the runner kills it at the time limit, before the provider
        # would, and scores 0 (a hang used to crash the runner, which masked the sample).
        pytest.param({'timeout': 3, 'test_commands': ['sleep 30']}, (None, 0.0), id='test-hangs'),
        # Each command gets its share of the time, and none starts once it is gone; at 1s a
        # command, eight ran past the provider's limit and were masked (Codex review).
        pytest.param({'timeout': 2, 'test_commands': ['sleep 30'] * 8}, (None, 0.0), id='many'),
        # Less than a second left is still time to run a quick test (Codex review).
        pytest.param({'timeout': 1, 'test_commands': ['true']}, (None, 1.0), id='subsecond'),
        # Setup that runs past the limit: the harness's doing, no verdict.
        pytest.param(
            {'timeout': 2, 'setup_commands': 'sleep 30', 'test_commands': ['true']},
            ('setup_failed', 0.0),
            id='setup-hangs',
        ),
    ],
)
def test_time_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    task: dict,
    outcome: tuple[str | None, float],
) -> None:
    started = time.monotonic()

    exit_code, result = _run_main(tmp_path, monkeypatch, capsys, _runner(), task)

    assert (result.error_type, result.score) == outcome
    assert is_verdict(exit_code, result.error_type) is (outcome[0] is None)
    assert time.monotonic() - started < task['timeout'] + 2


def test_no_time_left_for_the_tests_has_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # The steps before the tests used up the time limit (a clock that jumps 10s a reading):
    # the harness's doing, so no verdict.
    runner = _runner()
    runner['time'] = types.SimpleNamespace(monotonic=itertools.count(0, 10).__next__)

    exit_code, result = _run_main(
        tmp_path, monkeypatch, capsys, runner, {'timeout': 5, 'test_commands': ['true']}
    )

    assert (result.error_type, is_verdict(exit_code, result.error_type)) == (
        'validation_timeout',
        False,
    )


def test_runner_stopped_by_the_tests_is_scored(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    # Code the patch wrote runs as root and can kill the runner, here the test command killing
    # its parent. The runner had said the tests were running, so the patch scores 0 instead of
    # the sample being masked.
    env = _env(tmp_path).model_copy(update={'test_command': 'kill -KILL $PPID'})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), _FIX))

    outcome = (result.score, result.error_type, result.has_verdict)
    assert outcome == (0.0, 'policy_sandbox_failure', True)


def test_patch_that_cannot_be_written_is_the_patch_fault(
    tmp_path: Path, sandbox_dg: DeepGym
) -> None:
    # git apply --check passes a diff adding both foo and foo/bar, but writing it fails: the
    # patch's doing, so a scored 0, never a masked write error (Codex review).
    env = _env(tmp_path)
    diff = _diff(Path(env.repo_dir), _FIX) + new_files_diff({'foo': 'x\n', 'foo/bar': 'y\n'})

    result = sandbox_dg.run(env, diff)

    assert (result.error_type, result.has_verdict) == ('patch_apply_failed', True)


def test_missing_repo_is_not_blamed_on_the_patch(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    # An image without its repo_dir is broken: git's "cannot change to" there must not read as
    # a patch that does not apply, which is the model's fault.
    env = _env(tmp_path)
    patch = _diff(Path(env.repo_dir), _FIX)
    env = env.model_copy(update={'repo_dir': str(tmp_path / 'missing'), 'test_patch': ''})

    result = sandbox_dg.run(env, patch)

    assert (result.error_type, result.has_verdict) == ('validation_error', False)


def test_hidden_tests_are_restored_in_a_subdirectory_task(
    tmp_path: Path, sandbox_dg: DeepGym
) -> None:
    # Git names a patch's paths from the repository top, not from repo_dir (Codex review): an
    # answer that also edits the hidden test must not keep that edit and break the test patch.
    env = _env(tmp_path, {'pkg/calc.py': _CALC, 'pkg/tests/test_calc.py': _BASE_TEST})
    repo = Path(env.repo_dir)
    env = env.model_copy(
        update={
            'repo_dir': str(repo / 'pkg'),
            'test_command': f'{_PYTEST} tests/test_calc.py',
            'test_patch': _diff(repo, {'pkg/tests/test_calc.py': _HIDDEN_TEST}),
        }
    )
    edit = {'pkg/calc.py': _FIX['calc.py'], 'pkg/tests/test_calc.py': _BASE_TEST + '# edited\n'}

    result = sandbox_dg.run(env, _diff(repo, edit))

    assert (result.score, result.error_type) == (1.0, None)


@pytest.mark.parametrize('command', ['', ' \n'])
def test_blank_test_command_is_an_invalid_task(
    tmp_path: Path, sandbox_dg: DeepGym, command: str
) -> None:
    # A blank command exits 0 without running a test, so it once scored 1.0.
    env = _env(tmp_path).model_copy(update={'test_command': command})

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), _FIX))

    assert (result.score, result.error_type, result.has_verdict) == (0.0, 'invalid_task', False)


def test_blank_command_in_a_list_runs_nothing(tmp_path: Path, sandbox_dg: DeepGym) -> None:
    # One blank entry in SWE-bench Pro's explicit test_commands voids the task before any run.
    env = _env(tmp_path)
    repo = Path(env.repo_dir)
    row = {
        'instance_id': 'blank-1',
        'repo': 'fixture/calc',
        'repo_source': str(repo),
        'base_commit': _git(repo, 'rev-parse', 'HEAD').strip(),
        'test_patch': env.test_patch,
        'test_commands': [f'touch "{tmp_path / "ran"}"', ' '],
    }

    result = sandbox_dg.run(SWEBenchProEnvironment(), _diff(repo, _FIX), task=row)

    assert (result.error_type, result.has_verdict) == ('invalid_task', False)
    assert not (tmp_path / 'ran').exists()


_OLD_PYTHONS = [
    pytest.param(
        shutil.which(f'python{version}'),
        id=f'python{version}',
        marks=pytest.mark.skipif(
            shutil.which(f'python{version}') is None, reason=f'needs python{version} on PATH'
        ),
    )
    for version in ('3.8', '3.9')
]


@pytest.mark.parametrize('python', _OLD_PYTHONS)
def test_runner_runs_on_old_image_pythons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox_dg: DeepGym, python: str
) -> None:
    # The runner runs in the image's python3, 3.8 or 3.9 on some SWE-bench Pro images (2 of 10
    # sampled ansible ones). Neither has tomllib, so the inlined parser judges the pyproject edit.
    monkeypatch.setattr('deepgym.core.open_sandbox', lambda provider, image: RunnerSandbox(python))
    env = _env(tmp_path, {'pyproject.toml': _PYPROJECT})
    edit = {'pyproject.toml': _PYPROJECT.replace('"1.0"', '"2.0"'), **_FIX}

    result = sandbox_dg.run(env, _diff(Path(env.repo_dir), edit))

    assert (result.score, result.error_type) == (1.0, None)


def _stub_runner(monkeypatch: pytest.MonkeyPatch, source: str) -> RepoPatchEnvironment:
    """A patch environment whose sandbox runs *source* in place of the patch runner."""
    monkeypatch.setattr(PatchVerifier, 'build_runner_script', staticmethod(lambda: source))
    return RepoPatchEnvironment(task='Fix add().', repo_dir='/repo', test_command='true')


def test_runner_verdict_is_read_from_stdout_only(
    monkeypatch: pytest.MonkeyPatch, sandbox_dg: DeepGym
) -> None:
    # Providers merge stdout and stderr. A JSON line on the runner's stderr (a warning, a crash
    # after the verdict) must not replace the verdict on its stdout, as for the judges.
    verdict = json.dumps({'score': 1.0, 'passed': True})
    noise = json.dumps({'score': 0.0, 'passed': False})
    script = f'import sys\nprint({verdict!r})\nprint({noise!r}, file=sys.stderr)\n'
    env = _stub_runner(monkeypatch, script)

    result = sandbox_dg.run(env, 'diff --git a/calc.py b/calc.py\n')

    assert (result.score, result.passed, result.stderr) == (1.0, True, noise)


def test_runner_without_a_verdict_raises_a_typed_error(
    monkeypatch: pytest.MonkeyPatch, sandbox_dg: DeepGym
) -> None:
    # A runner that dies before printing its JSON is a verifier error, as for the judges:
    # dg.run raises VerifierError, and a batch returns the run without a verdict.
    env = _stub_runner(monkeypatch, "raise SystemExit('runner crashed')\n")

    with pytest.raises(VerifierError, match='runner crashed'):
        sandbox_dg.run(env, 'diff --git a/calc.py b/calc.py\n')
    batch = sandbox_dg.run_batch(env, ['diff --git a/calc.py b/calc.py\n'])

    assert (batch.results[0].error_type, batch.results[0].has_verdict) == ('verifier_error', False)


def _fake_cargo(path: Path) -> Path:
    """A cargo that writes the toolchain dirs it ran with to the returned file."""
    path.parent.mkdir(parents=True)
    path.write_text('#!/bin/sh\necho "CARGO_HOME=$CARGO_HOME RUSTUP_HOME=$RUSTUP_HOME" >"$0.ran"\n')
    path.chmod(0o755)
    return path.with_name('cargo.ran')


@pytest.mark.parametrize('image_cargo', [False, True], ids=['flattened', 'cargo-on-path'])
def test_flattened_image_gets_its_rust_toolchain_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    image_cargo: bool,
) -> None:
    # MiMo's Docker Hub images are flattened: rustup's /usr/local/cargo and /usr/local/rustup
    # are there, but the ENV that put them on PATH is gone, so every answer, right or wrong,
    # failed with "cargo: command not found". An image whose PATH has a cargo keeps it.
    toolchain = tmp_path / 'toolchain'
    rustup_cargo = _fake_cargo(toolchain / 'cargo' / 'bin' / 'cargo')
    image_ran = _fake_cargo(tmp_path / 'image' / 'cargo')
    (toolchain / 'rustup').mkdir()
    runner = _runner()
    runner.update(_CARGO_HOME=str(toolchain / 'cargo'), _RUSTUP_HOME=str(toolchain / 'rustup'))
    path = f'{tmp_path / "image"}:/usr/bin:/bin' if image_cargo else '/usr/bin:/bin'
    for name in ('CARGO_HOME', 'RUSTUP_HOME'):  # unset, and restored after the runner sets them
        monkeypatch.setenv(name, '')
        monkeypatch.delenv(name)
    monkeypatch.setenv('PATH', path)
    task = {'timeout': 60, 'test_commands': ['cargo test']}

    exit_code, result = _run_main(tmp_path, monkeypatch, capsys, runner, task)

    assert (result.score, exit_code) == (1.0, 0)
    if image_cargo:
        assert image_ran.read_text() == 'CARGO_HOME= RUSTUP_HOME=\n'
        assert not rustup_cargo.exists()
    else:
        dirs = f'CARGO_HOME={toolchain}/cargo RUSTUP_HOME={toolchain}/rustup\n'
        assert rustup_cargo.read_text() == dirs
