"""Tests for the MiMo-V2.6-RL code importer and RepoPatchEnvironment scoring.

Scoring runs through the sandbox code path (``sandbox_dg``); local mode refuses.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from deepgym import DeepGym, load_environment
from deepgym.benchmark_envs import RepoPatchEnvironment
from deepgym.exceptions import SandboxError
from deepgym.models import Environment
from deepgym.registry import _load_env_from_dir
from deepgym.sandboxes.docker import DockerContainer

scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from import_mimo_rl import build_code_env, load_image_map  # noqa: E402

_IMAGE = 'format-code-task-000001:latest'
_HUB_IMAGE = 'docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-000001'

# Long enough that the hidden patch's appended check doesn't touch the first line.
_BASE_TEST = (
    'from calc import add\n\n\ndef check_zero():\n    assert add(0, 0) == 0\n\n\ncheck_zero()\n'
)
_HIDDEN_CHECK = 'assert add(1, 2) == 3\n'
_FIX = 'def add(a, b):\n    return a + b\n'
# What a tampering diff puts at the top of a hidden test file.
_EXIT_EARLY = 'raise SystemExit(0)\n'

# A real task (447 MB image): hashicorp/vault-action, make the `secrets` input optional.
_LIVE_ID = 'format-code-task-001457'
_LIVE_IMAGE = f'docker.io/xiaomimimo/mimo-v2.6-rl-oss:{_LIVE_ID}'
_LIVE_FIX = """```diff
diff --git a/src/action.js b/src/action.js
--- a/src/action.js
+++ b/src/action.js
@@ -14,7 +14,7 @@ async function exportSecrets() {
     const exportEnv = core.getInput('exportEnv', { required: false }) != 'false';
     const exportToken = (core.getInput('exportToken', { required: false }) || 'false').toLowerCase() != 'false';

-    const secretsInput = core.getInput('secrets', { required: true });
+    const secretsInput = core.getInput('secrets', { required: false }) || '';
     const secretRequests = parseSecretsInput(secretsInput);

     const vaultMethod = (core.getInput('method', { required: false }) || 'token').toLowerCase();
```"""  # noqa: E501
_LIVE_NOOP = """diff --git a/NOTES.md b/NOTES.md
new file mode 100644
--- /dev/null
+++ b/NOTES.md
@@ -0,0 +1 @@
+not a fix
"""


def _live_ready() -> bool:
    """Docker is up, the task is imported, and its image is already pulled."""
    if not (Path.home() / '.deepgym/environments/mimo_code' / _LIVE_ID).is_dir():
        return False
    if not DockerContainer.available():
        return False
    inspect = subprocess.run(['docker', 'image', 'inspect', _LIVE_IMAGE], capture_output=True)
    return inspect.returncode == 0


def _git(repo: Path, *args: str) -> str:
    config = ('user.email=t@example.com', 'user.name=t', 'commit.gpgsign=0')
    config += ('core.autocrlf=false', 'core.quotePath=true')
    command = ['git', '-C', str(repo), *(arg for item in config for arg in ('-c', item))]
    proc = subprocess.run([*command, *args], check=True, capture_output=True)
    return proc.stdout.decode()


def _diff(repo: Path, files: dict[str, str | None]) -> str:
    """Return the diff that writes *files* into *repo* (None deletes), leaving it unchanged."""
    for name, content in files.items():
        if content is None:
            (repo / name).unlink()
            continue
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(content.encode())
    _git(repo, 'add', '-A')
    diff = _git(repo, 'diff', '--cached', '-M', '--binary')
    _git(repo, 'reset', '--hard', '-q')
    _git(repo, 'clean', '-fdq')
    return diff


def _fixture(
    root: Path, test_file: str = 'tests/test_calc.py', renamed_from: str | None = None
) -> tuple[dict, dict[str, str]]:
    """A code row shaped like the dataset's, over a fresh repo, plus candidate diffs.

    The base repo has a test file (at *renamed_from* if given). The hidden test patch
    moves it to *test_file* if needed and appends a check only a real fix passes.
    """
    repo = root / 'testbed'
    base_file = renamed_from or test_file
    (repo / base_file).parent.mkdir(parents=True)
    (repo / base_file).write_text(_BASE_TEST)
    (repo / 'calc.py').write_text('def add(a, b):\n    return a - b\n')
    _git(repo, 'init', '-q')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-qm', 'base')

    hidden: dict[str, str | None] = {
        test_file: _BASE_TEST + _HIDDEN_CHECK,
        'tests/fixture.txt': 'crlf\r\nline\r\n',
        'mimo_test_command.sh': f'cd "{repo}"\nPYTHONPATH=. "{sys.executable}" "{test_file}"\n',
    }
    if renamed_from:
        hidden[renamed_from] = None
    test_patch = _diff(repo, hidden)
    tamper = _diff(repo, {base_file: _EXIT_EARLY + _BASE_TEST})
    answers = {
        'good': f'```diff\n{_diff(repo, {"calc.py": _FIX})}```',
        'wrong': _diff(repo, {'calc.py': 'def add(a, b):\n    return a * b\n'}),
        'empty': 'I could not find the bug.',
        'tamper': tamper,
        'good_and_tamper': _diff(repo, {'calc.py': _FIX, base_file: _EXIT_EARLY + _BASE_TEST}),
        # The same edit as a traditional diff: no `diff --git` or `index` line.
        'tamper_traditional': ''.join(
            line
            for line in tamper.splitlines(keepends=True)
            if not line.startswith(('diff --git ', 'index '))
        ),
    }
    instance = {
        'cwd': str(repo),
        'dataset_type': 'opensource-code',
        'docker_image': _IMAGE,
        'instance_id': 'format-code-task-000001',
        'problem_statement': 'add() subtracts instead of adding.',
        'test_command': f'bash "{repo}/mimo_test_command.sh"',
        'test_patch': test_patch,
        'verifier_timeout_sec': 120,
    }
    row = {
        'data_source': 'opensource-code',
        'prompt': [{'role': 'user', 'content': instance['problem_statement']}],
        'reward_model': {'ground_truth': '', 'style': 'rule'},
        'extra_info': {
            'instance_id': instance['instance_id'],
            'instance_json': json.dumps(instance),
        },
    }
    return row, answers


def _env(row: dict, out: Path) -> Environment:
    """Import *row* into *out* and load it back through the registry loader."""
    entry = build_code_env(row, {_IMAGE: _HUB_IMAGE}, out)
    return _load_env_from_dir(out / entry['id'])


def _tests_pass_if_edits_kept(row: dict, diff: str) -> bool:
    """Apply *diff* then the hidden test patch with no restore step, and run the tests."""
    instance = json.loads(row['extra_info']['instance_json'])
    repo = Path(instance['cwd'])
    for patch in (diff, instance['test_patch']):
        subprocess.run(['git', '-C', str(repo), 'apply'], input=patch.encode(), check=True)
    return subprocess.run(['bash', str(repo / 'mimo_test_command.sh')]).returncode == 0


def test_load_image_map(tmp_path: Path) -> None:
    path = tmp_path / 'image-mapping.jsonl'
    path.write_text(json.dumps({'dataset_image': _IMAGE, 'dockerhub_image': _HUB_IMAGE}) + '\n')

    assert load_image_map(path) == {_IMAGE: _HUB_IMAGE}


def test_build_code_env_loads_as_repo_patch_env(tmp_path: Path) -> None:
    row, _ = _fixture(tmp_path)
    instance = json.loads(row['extra_info']['instance_json'])

    entry = build_code_env(row, {_IMAGE: _HUB_IMAGE}, tmp_path / 'mimo_code')
    env = _load_env_from_dir(tmp_path / 'mimo_code' / instance['instance_id'])

    assert entry['path'] == 'environments/mimo_code/format-code-task-000001'
    assert isinstance(env, RepoPatchEnvironment)
    assert env.image == _HUB_IMAGE
    assert env.repo_dir == instance['cwd']
    assert env.test_command == instance['test_command']
    assert env.timeout == 120
    assert '\r' in env.test_patch
    assert env.test_patch == instance['test_patch']
    assert instance['problem_statement'] in env.task


@pytest.mark.parametrize(
    ('answer', 'expected'),
    [
        ('good', 1.0),
        ('wrong', 0.0),
        ('empty', 0.0),
        ('tamper', 0.0),
        # The fix counts; the edit to the hidden test file is undone.
        ('good_and_tamper', 1.0),
    ],
)
def test_scoring_in_sandbox(
    tmp_path: Path, sandbox_dg: DeepGym, answer: str, expected: float
) -> None:
    row, answers = _fixture(tmp_path)

    result = sandbox_dg.run(_env(row, tmp_path / 'mimo_code'), answers[answer])

    assert result.score == expected
    assert result.passed is (expected == 1.0)
    assert result.reward_components == {'apply': 0.0, 'tests': expected}


@pytest.mark.parametrize(
    ('test_file', 'renamed_from', 'answer'),
    [
        # The hidden patch renames a test file; the diff edits the rename's source.
        pytest.param('tests/test_calc.py', 'tests/test_old.py', 'tamper', id='renamed'),
        pytest.param('tests/test_calc.py', None, 'tamper_traditional', id='traditional'),
        # Non-ASCII name: git spells it "a/tests/t\303\253st_calc.py".
        pytest.param('tests/tëst_calc.py', None, 'tamper', id='quoted'),
    ],
)
def test_diff_cannot_edit_hidden_tests(
    tmp_path: Path,
    sandbox_dg: DeepGym,
    test_file: str,
    renamed_from: str | None,
    answer: str,
) -> None:
    naive_row, naive_answers = _fixture(tmp_path / 'naive', test_file, renamed_from)
    row, answers = _fixture(tmp_path / 'attack', test_file, renamed_from)
    fixed_row, fixed_answers = _fixture(tmp_path / 'fixed', test_file, renamed_from)

    attack = sandbox_dg.run(_env(row, tmp_path / 'envs'), answers[answer])
    fixed = sandbox_dg.run(_env(fixed_row, tmp_path / 'envs'), fixed_answers['good_and_tamper'])

    assert _tests_pass_if_edits_kept(naive_row, naive_answers[answer])
    assert attack.score == 0.0
    assert fixed.score == 1.0


def test_attack_diffs_use_the_spellings_under_test(tmp_path: Path) -> None:
    row, renamed = _fixture(tmp_path / 'renamed', renamed_from='tests/test_old.py')
    _, plain = _fixture(tmp_path / 'plain')
    _, quoted = _fixture(tmp_path / 'quoted', test_file='tests/tëst_calc.py')

    assert (
        'rename from tests/test_old.py'
        in json.loads(row['extra_info']['instance_json'])['test_patch']
    )
    assert 'tests/test_old.py' in renamed['tamper']
    assert 'diff --git' not in plain['tamper_traditional']
    assert '"a/tests/t\\303\\253st_calc.py"' in quoted['tamper']


def test_local_mode_refuses(tmp_path: Path) -> None:
    row, answers = _fixture(tmp_path)
    env = _env(row, tmp_path / 'mimo_code')

    with pytest.raises(SandboxError, match="mode='docker'"):
        DeepGym(mode='local').run(env, answers['good'])

    assert (tmp_path / 'testbed' / 'calc.py').read_text().endswith('a - b\n')


# Shares the live-container worker so the leak check in test_sandboxes_live stays exact.
@pytest.mark.xdist_group('live-containers')
@pytest.mark.skipif(
    not _live_ready(),
    reason=f'needs Docker, scripts/import_mimo_rl.py run, and {_LIVE_IMAGE} pulled',
)
@pytest.mark.parametrize(('answer', 'expected'), [(_LIVE_NOOP, 0.0), (_LIVE_FIX, 1.0)])
def test_live_docker_task(answer: str, expected: float) -> None:
    env = load_environment(f'mimo_code/{_LIVE_ID}')

    result = DeepGym(mode='docker').run(env, answer)

    assert result.score == expected
    assert result.error_type is None
