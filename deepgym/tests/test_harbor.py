"""Harbor tasks: parsing, loading, rewards, registration, the CLI and local mode.

Sandbox isolation is in test_harbor_isolation.py, export in test_harbor_export.py and
real-container runs in test_harbor_live.py.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from deepgym import registry as registry_module
from deepgym.benchmark_envs import TerminalBenchEnvironment
from deepgym.cli import main
from deepgym.core import DeepGym
from deepgym.exceptions import DeepGymError, InvalidScoreError, SandboxError, VerifierError
from deepgym.harbor import HarborEnvironment, HarborTask, dataset_root, load_tasks, task_dirs
from deepgym.harbor.grading import reward_result
from deepgym.harbor.task import HARNESS_PATHS
from deepgym.qa import QACheck, run_qa
from deepgym.registry import list_environments, load_environment, register_benchmark
from deepgym.sandboxes import ExecResult
from deepgym.session import check_workdir

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'harbor'
TASKS = FIXTURES / 'tasks'
HELLO = TASKS / 'hello'
BUILT = TASKS / 'hello-built'


@pytest.fixture()
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty registry cache, so tests never touch ~/.deepgym."""
    cache_dir = tmp_path / 'environments'
    cache_dir.mkdir()
    monkeypatch.setattr(registry_module, '_get_cache_dir', lambda: cache_dir)
    return cache_dir


def _task_copy(tmp_path: Path, source: Path = HELLO, name: str | None = None) -> Path:
    target = tmp_path / (name or source.name)
    shutil.copytree(source, target)
    return target


# ---------------------------------------------------------------------------
# Task folders
# ---------------------------------------------------------------------------


def test_parses_a_prebuilt_image_task() -> None:
    task = HarborTask.from_dir(HELLO)

    assert task.name == 'hello'
    assert task.instruction == 'Write the word `hello` to `/app/hello.txt`.\n'
    assert task.docker_image == 'python:3.12-slim'
    assert task.dockerfile is None
    assert task.workdir == '/app'
    assert (task.agent_timeout_sec, task.verifier_timeout_sec) == (60.0, 60.0)
    assert task.verifier_network == 'no-network'
    assert sorted(task.tests) == ['test.sh']
    assert task.solution['answer.txt'] == b'hello\n'


def test_a_task_cannot_choose_a_snapshot(tmp_path: Path) -> None:
    """task.toml has no snapshot field: one written there never reaches the sandbox."""
    folder = _task_copy(tmp_path)
    toml = folder / 'task.toml'
    toml.write_text(
        toml.read_text().replace('[environment]', '[environment]\nsnapshot = "attacker"', 1)
    )

    env = HarborEnvironment.from_dir(folder)

    assert env.snapshot is None
    assert env.image == 'python:3.12-slim'


def test_parses_a_dockerfile_task_with_harbor_defaults() -> None:
    task = HarborTask.from_dir(BUILT)

    assert task.docker_image is None
    assert task.dockerfile == BUILT / 'environment' / 'Dockerfile'
    assert task.workdir is None
    assert task.agent_timeout_sec is None
    assert task.verifier_network == 'no-network'
    assert task.solution == {}


@pytest.mark.parametrize(
    ('folder', 'message'),
    [('bad-toml', 'Invalid Harbor task'), ('bad-timeout', 'greater than 0')],
)
def test_rejects_an_invalid_task_toml(folder: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HarborTask.from_dir(FIXTURES / 'invalid' / folder)


@pytest.mark.parametrize(
    ('toml', 'message'),
    [
        ('[environment]\nworkdir = "/logs/verifier"\n', 'overlaps /logs'),
        ('[environment]\nworkdir = "/"\n', 'would hold /tmp'),
        ('[environment]\nworkdir = "/root"\n', 'overlaps /root'),
        ('[environment]\nworkdir = "//usr/bin"\n', 'overlaps /usr'),
        ('[environment]\nos = "windows"\n', 'linux'),
        ('[verifier]\nnetwork_mode = "sometimes"\n', 'no-network'),
        ('[[steps]]\nname = "one"\n', 'multi-step'),
        ('[verifier.env]\nKEY = "${HOST_SECRET}"\n', 'host variables'),
        # Names env could read as an option or an odd assignment; $ would pass 'A\n'.
        ('[verifier.env]\n-S = "x"\n', 'names must be'),
        ('[verifier.env]\n"A B" = "x"\n', 'names must be'),
        ('[verifier.env]\n"A=B" = "x"\n', 'names must be'),
        ('[verifier.env]\n"" = "x"\n', 'names must be'),
        ('[verifier.env]\n"A\\n" = "x"\n', 'names must be'),
        ('[verifier.env]\n1A = "x"\n', 'names must be'),
        ('[verifier.env]\nKEY = "x\\u0000y"\n', 'NUL'),
    ],
)
def test_rejects_unsupported_settings(tmp_path: Path, toml: str, message: str) -> None:
    task = _task_copy(tmp_path)
    (task / 'task.toml').write_text(toml, encoding='utf-8')

    with pytest.raises(ValueError, match=message):
        HarborTask.from_dir(task)


def test_rejects_missing_test_script_and_solve_script(tmp_path: Path) -> None:
    task = _task_copy(tmp_path)
    (task / 'solution' / 'solve.sh').unlink()
    with pytest.raises(ValueError, match='solve.sh'):
        HarborTask.from_dir(task)
    (task / 'tests' / 'test.sh').unlink()
    with pytest.raises(ValueError, match='test.sh'):
        HarborTask.from_dir(task)


def test_rejects_a_link_out_of_the_task_folder(tmp_path: Path) -> None:
    secret = tmp_path / 'secret.txt'
    secret.write_text('host file', encoding='utf-8')
    task = _task_copy(tmp_path)
    (task / 'tests' / 'leak.txt').symlink_to(secret)

    with pytest.raises(ValueError, match='links outside'):
        HarborTask.from_dir(task)


def test_rejects_a_build_context_that_links_out_of_the_task(tmp_path: Path) -> None:
    secret = tmp_path / 'secret.txt'
    secret.write_text('host file', encoding='utf-8')
    linked = _task_copy(tmp_path / 'a', BUILT)
    shutil.rmtree(linked / 'environment')
    (linked / 'environment').symlink_to('..', target_is_directory=True)
    with pytest.raises(ValueError, match='environment/ is a link'):
        HarborTask.from_dir(linked)

    inner = _task_copy(tmp_path / 'b', BUILT)
    (inner / 'environment' / 'data.txt').symlink_to(secret)
    with pytest.raises(ValueError, match='links outside environment/'):
        HarborTask.from_dir(inner)

    inside = _task_copy(tmp_path / 'c', BUILT)
    (inside / 'environment' / 'Dockerfile.link').symlink_to('Dockerfile')
    assert HarborTask.from_dir(inside).build_context == (inside / 'environment').resolve()


def test_verifier_network_falls_back_from_verifier_to_environment(tmp_path: Path) -> None:
    task = _task_copy(tmp_path)
    toml = task / 'task.toml'

    toml.write_text('[environment]\nallow_internet = false\n', encoding='utf-8')
    assert HarborTask.from_dir(task).verifier_network == 'no-network'
    toml.write_text(
        '[verifier]\nnetwork_mode = "public"\n[environment]\nnetwork_mode = "no-network"\n',
        encoding='utf-8',
    )
    assert HarborTask.from_dir(task).verifier_network == 'public'
    toml.write_text('', encoding='utf-8')
    assert HarborTask.from_dir(task).verifier_network == 'public'


@pytest.mark.parametrize(
    'path', ['/app', '/srv/task', '/home/agent/work', '/testbed', '/tmp/work', '/opt/app']
)
def test_check_workdir_accepts_ordinary_folders(path: str) -> None:
    assert check_workdir(path, HARNESS_PATHS) == path


@pytest.mark.parametrize(
    'path',
    [
        'app', '/', '/app/../logs', '/tests', '/logs', '/logs/x', '/solution',
        '/opt/deepgym-harbor', '/opt/deepgym/judge', '/opt', '/tmp', '/root', '/root/work',
        '/usr/local', '/etc/x', '/var/lib/dpkg', '//usr/bin', '///etc', '/./usr/bin',
        '/srv/../root', '//', '/app/../../usr', '/opt/deepgym-runtime', '/opt/deepgym-runtime/x',
    ],
)  # fmt: skip
def test_check_workdir_rejects_paths_root_trusts_or_the_harness_writes(path: str) -> None:
    with pytest.raises(ValueError):
        check_workdir(path, HARNESS_PATHS)


@pytest.mark.parametrize(
    ('path', 'normal'),
    [('//app', '/app'), ('/app/./src/', '/app/src'), ('/srv/x/../app', '/srv/app')],
)
def test_check_workdir_returns_the_normalized_path(path: str, normal: str) -> None:
    assert check_workdir(path, HARNESS_PATHS) == normal


# ---------------------------------------------------------------------------
# Loading datasets
# ---------------------------------------------------------------------------


def test_load_tasks_finds_every_task_folder_in_order() -> None:
    assert task_dirs(FIXTURES) == [
        FIXTURES / 'invalid' / 'bad-timeout',
        FIXTURES / 'invalid' / 'bad-toml',
        HELLO,
        BUILT,
    ]
    assert [task.name for task in load_tasks(TASKS)] == ['hello', 'hello-built']
    assert [task.name for task in load_tasks(TASKS, limit=1)] == ['hello']


def test_load_tasks_raises_on_an_invalid_task_or_an_empty_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='bad-t'):
        load_tasks(FIXTURES / 'invalid')
    with pytest.raises(DeepGymError, match='No Harbor tasks'):
        load_tasks(tmp_path)


def test_task_dirs_refuses_two_tasks_with_one_name(tmp_path: Path) -> None:
    shutil.copytree(HELLO, tmp_path / 'a' / 'hello')
    shutil.copytree(HELLO, tmp_path / 'b' / 'hello')
    with pytest.raises(DeepGymError, match='share names'):
        task_dirs(tmp_path)


def test_dataset_root_resolves_a_relative_task_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(HELLO)
    assert dataset_root('.') == HELLO
    assert [task.name for task in load_tasks('.')] == ['hello']


@pytest.mark.parametrize('source', ['./missing-folder', 'not a dataset', '/no/such/dir'])
def test_dataset_root_rejects_unknown_sources(source: str) -> None:
    with pytest.raises(DeepGymError, match='not a folder'):
        dataset_root(source)


def test_dataset_root_clones_a_git_repository_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / 'repo'
    shutil.copytree(TASKS, repo)
    git = ['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@example.com']
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run([*git, 'add', '.'], check=True)
    subprocess.run([*git, 'commit', '-q', '-m', 'tasks'], check=True)
    monkeypatch.setattr('deepgym.harbor.loader._cache_dir', lambda: tmp_path / 'cache')

    root = dataset_root(f'file://{repo}')
    shutil.rmtree(repo)

    assert root.parent == tmp_path / 'cache'
    assert [task.name for task in load_tasks(f'file://{repo}')] == ['hello', 'hello-built']


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

_PASSED_TESTS = ExecResult(0, 'ok')


def test_reward_txt_is_the_score() -> None:
    result = reward_result('reward.txt\n1\n', _PASSED_TESTS)
    assert (result.score, result.passed) == (1.0, True)
    assert reward_result('reward.txt\n0.25', _PASSED_TESTS).passed is False


def test_reward_json_uses_its_reward_key_else_the_mean() -> None:
    labeled = reward_result('reward.json\n{"reward": 0.5, "style": 1}', _PASSED_TESTS)
    assert labeled.score == 0.5
    assert labeled.reward_components == {'reward': 0.5, 'style': 1.0}
    assert reward_result('reward.json\n{"a": 1, "b": 0}', _PASSED_TESTS).score == 0.5


@pytest.mark.parametrize(
    'listing', ['reward.txt\nnan', 'reward.txt\n1.5', 'reward.json\n{"r": -1}']
)
def test_out_of_range_rewards_are_verifier_errors(listing: str) -> None:
    with pytest.raises(InvalidScoreError):
        reward_result(listing, _PASSED_TESTS)


@pytest.mark.parametrize(
    'listing',
    [
        'none',
        'reward.txt\npassed',
        'reward.json\n[1]',
        'reward.json\n{}',
        'reward.json\n{"a": "x"}',
    ],
)
def test_missing_or_unreadable_rewards_are_verifier_errors(listing: str) -> None:
    with pytest.raises(VerifierError):
        reward_result(listing, ExecResult(1, 'tests crashed'))


# ---------------------------------------------------------------------------
# Registry and qa
# ---------------------------------------------------------------------------


def test_registered_tasks_load_by_name(cache: Path) -> None:
    metadata = {'type': 'tool-use', 'tags': ['harbor']}
    registry = register_benchmark('fixtures', [HELLO, BUILT], metadata)

    env = load_environment('fixtures/hello')

    assert registry == cache / 'fixtures' / 'registry.json'
    assert isinstance(env, HarborEnvironment)
    assert env.harbor_task.docker_image == 'python:3.12-slim'
    assert {entry['path'] for entry in list_environments()} >= {'fixtures/hello'}
    register_benchmark('fixtures', [HELLO], metadata)
    with pytest.raises(ValueError, match='not found'):
        load_environment('fixtures/hello-built')


def test_a_registry_json_inside_a_linked_task_is_not_read(cache: Path, tmp_path: Path) -> None:
    task = _task_copy(tmp_path)
    (task / 'tests' / 'registry.json').write_text('not json', encoding='utf-8')

    register_benchmark('fixtures', [task], {})

    assert any(entry['path'] == 'fixtures/hello' for entry in list_environments())


def test_register_refuses_to_overwrite_another_benchmark(cache: Path) -> None:
    (cache / 'humaneval' / 'HumanEval_0').mkdir(parents=True)
    with pytest.raises(ValueError, match='another benchmark'):
        register_benchmark('humaneval', [HELLO], {})
    with pytest.raises(ValueError, match='plain folder name'):
        register_benchmark('../escape', [HELLO], {})


def test_qa_checks_a_harbor_task_against_its_solve_sh(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_benchmark('fx', [HELLO, BUILT], {})
    seen: list[str | None] = []

    def single_turn_checks(env: HarborEnvironment, reference: str | None, *args: object) -> list:
        seen.append(reference)
        return [QACheck(name='reference_passes', status='pass', details='')]

    def hack_resistant(env: HarborEnvironment, dg: DeepGym) -> QACheck:
        raise AssertionError('Python attack payloads ran against a bash runner')

    monkeypatch.setattr('deepgym.qa.gate.single_turn_checks', single_turn_checks)
    monkeypatch.setattr('deepgym.qa.gate.hack_resistant', hack_resistant)
    dg = DeepGym(mode='local')

    report = run_qa('fx/hello', dg)

    checks = {check.name: check for check in report.checks}
    assert report.status == 'pass'
    assert checks['hack_resistant'].status == 'skip' and not checks['hack_resistant'].passed
    assert seen == [HarborEnvironment.from_dir(HELLO).reference_solution]
    assert run_qa('fx/hello-built', dg).status == 'skip'


# ---------------------------------------------------------------------------
# Local mode and Terminal-Bench
# ---------------------------------------------------------------------------


def test_local_mode_refuses_to_run_harbor_tasks(local_dg: DeepGym) -> None:
    with pytest.raises(SandboxError, match="mode='docker'"):
        local_dg.run(HarborEnvironment.from_dir(HELLO), 'echo hello > /app/hello.txt')


def test_terminal_bench_loads_tasks_through_the_harbor_parser(tmp_path: Path) -> None:
    env = TerminalBenchEnvironment(task_root=TASKS)

    assert env.list_task_ids() == ['hello', 'hello-built']
    assert isinstance(env.get_task('hello-built'), HarborTask)
    assert env.default_task_id == 'hello'
    (request,) = env.prepare_batch_requests(['echo hi'], task_id='hello')
    assert isinstance(request.env, HarborEnvironment) and request.env.name == 'hello'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(*args: str) -> int:
    with patch.object(sys, 'argv', ['deepgym', *args]):
        try:
            main()
        except SystemExit as exc:
            return int(exc.code or 0)
    return 0


def test_cli_import_lists_valid_tasks(capsys: pytest.CaptureFixture) -> None:
    assert _cli('harbor', 'import', str(TASKS)) == 0
    out = capsys.readouterr().out
    assert 'ok    hello        image=python:3.12-slim  solution=yes' in out
    assert 'image=environment/Dockerfile' in out
    assert '2 of 2 tasks valid' in out


def test_cli_import_reports_invalid_tasks_and_exits_1(capsys: pytest.CaptureFixture) -> None:
    assert _cli('harbor', 'import', str(FIXTURES), '--limit', '2') == 1
    out = capsys.readouterr().out
    assert out.count('FAIL') == 2 and '0 of 2 tasks valid' in out


def test_cli_import_registers_valid_tasks(cache: Path, capsys: pytest.CaptureFixture) -> None:
    assert _cli('harbor', 'import', str(TASKS), '--register', 'fx') == 0
    assert isinstance(load_environment('fx/hello'), HarborEnvironment)
    assert 'load_environment("fx/hello")' in capsys.readouterr().out


def test_cli_export_writes_task_folders(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert _cli('harbor', 'export', 'coin_change', 'two_sum', '--out', str(tmp_path)) == 0
    assert capsys.readouterr().out.split() == [
        str(tmp_path / 'coin_change'),
        str(tmp_path / 'two_sum'),
    ]
    assert HarborTask.from_dir(tmp_path / 'two_sum').solution
