"""Tests for the MiMo-V2.6-RL code importer and RepoPatchEnvironment scoring."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from deepgym import DeepGym, load_environment
from deepgym.benchmark_envs import RepoPatchEnvironment
from deepgym.registry import _load_env_from_dir
from deepgym.sandboxes.docker import DockerContainer

scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from import_mimo_rl import build_code_env, load_image_map  # noqa: E402

_IMAGE = 'format-code-task-000001:latest'
_HUB_IMAGE = 'docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-000001'

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
    config = ('user.email=t@example.com', 'user.name=t', 'core.autocrlf=false', 'commit.gpgsign=0')
    command = ['git', '-C', str(repo), *(arg for item in config for arg in ('-c', item))]
    proc = subprocess.run([*command, *args], check=True, capture_output=True)
    return proc.stdout.decode()


def _diff(repo: Path, files: dict[str, str]) -> str:
    """Return the diff that writes *files* into *repo*, leaving the repo unchanged."""
    for name, content in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(content.encode())
    _git(repo, 'add', '-A')
    diff = _git(repo, 'diff', '--cached', '--binary')
    _git(repo, 'reset', '--hard', '-q')
    _git(repo, 'clean', '-fdq')
    return diff


def _fixture(tmp_path: Path) -> tuple[dict, dict[str, str]]:
    """A code row shaped like the dataset's, over a fresh repo, plus candidate diffs."""
    repo = tmp_path / 'testbed'
    repo.mkdir()
    _git(repo, 'init', '-q')
    (repo / 'calc.py').write_text('def add(a, b):\n    return a - b\n')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-qm', 'base')

    run_tests = f'cd "{repo}"\nPYTHONPATH=. "{sys.executable}" tests/test_calc.py\n'
    test_patch = _diff(
        repo,
        {
            'tests/test_calc.py': 'from calc import add\n\nassert add(1, 2) == 3\n',
            'tests/fixture.txt': 'crlf\r\nline\r\n',
            'mimo_test_command.sh': run_tests,
        },
    )
    fix = 'def add(a, b):\n    return a + b\n'
    answers = {
        'good': f'```diff\n{_diff(repo, {"calc.py": fix})}```',
        'wrong': _diff(repo, {'calc.py': 'def add(a, b):\n    return a * b\n'}),
        'empty': 'I could not find the bug.',
        # Its own copy of the hidden test file is dropped; the fix still counts.
        'good_with_test_copy': _diff(
            repo, {'calc.py': fix, 'tests/test_calc.py': 'assert False\n'}
        ),
        'tamper_tests_only': _diff(repo, {'tests/test_calc.py': 'pass\n'}),
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
    assert env.test_patch == instance['test_patch']  # byte-exact, \r included
    assert instance['problem_statement'] in env.task


@pytest.mark.parametrize(
    ('answer', 'expected'),
    [
        ('good', 1.0),
        ('wrong', 0.0),
        ('empty', 0.0),
        ('good_with_test_copy', 1.0),
        ('tamper_tests_only', 0.0),
    ],
)
def test_local_scoring(tmp_path: Path, answer: str, expected: float) -> None:
    row, answers = _fixture(tmp_path)
    build_code_env(row, {_IMAGE: _HUB_IMAGE}, tmp_path / 'mimo_code')
    env = _load_env_from_dir(tmp_path / 'mimo_code' / 'format-code-task-000001')

    result = DeepGym(mode='local').run(env, answers[answer])

    assert result.score == expected
    assert result.passed is (expected == 1.0)
    assert result.reward_components == {'apply': 0.0, 'tests': expected}


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
