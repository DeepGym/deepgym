"""Exporting DeepGym environments as Harbor tasks, and reading them back."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from deepgym.benchmark_envs import TerminalBenchEnvironment
from deepgym.computer_use import ComputerUseEnvironment
from deepgym.exceptions import DeepGymError
from deepgym.harbor import HarborEnvironment, HarborTask, export_environment
from deepgym.harbor.export import SOLUTION_PATH
from deepgym.judge import RUNTIME_SOURCE, JudgeJob
from deepgym.models import Environment
from deepgym.registry import load_environment
from tests.test_harbor import HELLO

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_export_round_trips_a_builtin_environment(tmp_path: Path) -> None:
    env = load_environment('coin_change')
    assert isinstance(env, Environment)

    task = HarborTask.from_dir(export_environment(env, tmp_path))

    assert task.name == 'coin_change'
    assert task.instruction.startswith(env.task.rstrip())
    assert task.instruction.endswith(f'`{SOLUTION_PATH}`.\n')
    assert task.docker_image is None
    assert task.dockerfile is not None
    assert task.dockerfile.read_text().startswith('FROM python:3.12-slim\n')
    assert (task.workdir, task.verifier_network) == ('/app', 'no-network')
    assert task.verifier_timeout_sec == env.timeout + 60
    judge = JudgeJob.for_environment(env, '').judge_files()
    exported = {
        rel.removeprefix('deepgym/judge/'): data.decode() for rel, data in task.tests.items()
    }
    assert {rel: exported[rel] for rel in judge} == judge
    assert task.tests['deepgym/runtime/deepgym_solution.py'].decode() == RUNTIME_SOURCE
    assert env.reference_solution.rstrip('\n') in task.solution['solve.sh'].decode()


def test_exported_task_toml_asks_for_a_separate_offline_verifier_and_a_nonroot_agent(
    tmp_path: Path,
) -> None:
    folder = export_environment(load_environment('two_sum'), tmp_path)

    config = tomllib.loads((folder / 'task.toml').read_text())

    assert config['artifacts'] == [SOLUTION_PATH]
    assert config['task']['name'] == 'deepgym/two_sum'
    assert config['agent'] == {'user': 'agent', 'network_mode': 'no-network'}
    assert config['verifier']['environment_mode'] == 'separate'
    assert config['verifier']['environment'] == {'network_mode': 'no-network'}
    assert 'useradd --create-home agent' in (folder / 'environment' / 'Dockerfile').read_text()


def test_exported_scripts_are_valid_bash_and_quote_the_judge_environment(
    tmp_path: Path,
) -> None:
    env = load_environment('two_sum').model_copy(
        update={'env_vars': {'SEED': "1'; touch /tmp/pwned; echo '$(id)"}}
    )
    folder = export_environment(env, tmp_path)

    for script in ('tests/test.sh', 'tests/judge_setup.sh', 'solution/solve.sh'):
        assert subprocess.run(['bash', '-n', str(folder / script)]).returncode == 0
    test_sh = (folder / 'tests' / 'test.sh').read_text()
    assert "'SEED=1'\"'\"'; touch /tmp/pwned; echo '\"'\"'$(id)'" in test_sh
    assert 'setpriv' in test_sh and '--reuid=dgsolver' in test_sh
    assert test_sh.index('rm -f /logs/verifier/reward.json') < test_sh.index('verifier.py')
    assert test_sh.index("sh -c 'kill -9 -1'") < test_sh.index('python3 -I - ')
    assert '> /opt/deepgym/verdict.out' in test_sh


def _run_exported_parser(tmp_path: Path, verdict: str, status: int) -> tuple[int, Path]:
    """Run the exported reward parser as test.sh does, from a work dir with a decoy json.py.

    Returns its exit code and the folder it writes reward.json to.
    """
    test_sh = (
        export_environment(load_environment('two_sum'), tmp_path) / 'tests/test.sh'
    ).read_text()
    invocation, rest = test_sh.split(" <<'EOF'\n", 1)
    parser = rest.split('\nEOF\n', 1)[0]
    flags = shlex.split(invocation.splitlines()[-1])
    python_flags = flags[flags.index('python3') + 1 : flags.index('-')]
    assert python_flags == ['-I']
    logs, workdir = tmp_path / 'logs', tmp_path / 'app'
    logs.mkdir()
    workdir.mkdir()
    (workdir / 'json.py').write_text('raise SystemExit("decoy json imported")\n')
    (tmp_path / 'verdict.out').write_text(verdict + '\n')
    done = subprocess.run(
        [sys.executable, *python_flags, '-', str(tmp_path / 'verdict.out'), str(status)],
        input=parser.replace('/logs/verifier', str(logs)),
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    return done.returncode, logs


def test_exported_reward_parser_never_imports_from_the_work_directory(tmp_path: Path) -> None:
    code, logs = _run_exported_parser(tmp_path, '{"score": 1.0, "passed": true}', 0)

    assert code == 0
    assert json.loads((logs / 'reward.json').read_text()) == {'reward': 1.0}
    test_sh = (tmp_path / 'two_sum' / 'tests' / 'test.sh').read_text()
    assert test_sh.index('cd / ') < test_sh.index('python3')
    interpreters = [line for line in test_sh.splitlines() if 'python3' in line or 'bash /' in line]
    assert interpreters and all(line.startswith('/usr/bin/env -i ') for line in interpreters)


@pytest.mark.parametrize(
    ('verdict', 'status', 'reward'),
    [
        ('{"score": 0.4, "passed": false}', 1, 0.4),
        ('{"score": 0.0, "passed": false, "error_type": "invalid_patch"}', 2, 0.0),
        ('{"score": 1.0, "passed": true}', 3, None),
        ('{"score": 1.0, "passed": true}', 1, None),
        ('{"score": 0.5, "passed": false}', 2, None),
        ('{"score": 1.0, "passed": true, "error_type": "sandbox_error"}', 0, None),
        ('{"score": true, "passed": true}', 0, None),
        ('no verdict at all', 0, None),
    ],
)
def test_exported_reward_needs_an_exit_code_that_backs_it(
    tmp_path: Path, verdict: str, status: int, reward: float | None
) -> None:
    code, logs = _run_exported_parser(tmp_path, verdict, status)

    if reward is None:
        assert code != 0 and not (logs / 'reward.json').exists()
    else:
        assert code == 0 and json.loads((logs / 'reward.json').read_text()) == {'reward': reward}


def test_export_reimports_as_a_runnable_harbor_environment(tmp_path: Path) -> None:
    env = HarborEnvironment.from_dir(export_environment(load_environment('two_sum'), tmp_path))

    assert env.reference_solution is not None
    assert env.harbor_task.verifier_network == 'no-network'


@pytest.mark.parametrize(
    'env',
    [
        ComputerUseEnvironment(task='t', name='desk'),
        TerminalBenchEnvironment(),
        HarborEnvironment.from_dir(HELLO),
        Environment(task='t', verifier_code='return 1.0'),
        Environment(name='../up', task='t', verifier_code='return 1.0'),
    ],
)
def test_export_refuses_envs_it_cannot_express(tmp_path: Path, env: Environment) -> None:
    with pytest.raises(DeepGymError):
        export_environment(env, tmp_path)


@pytest.mark.parametrize('image', [None, 'python:3.12-slim'])
def test_export_refuses_a_snapshot_runtime(tmp_path: Path, image: str | None) -> None:
    """Its dependencies would silently vanish from a plain image."""
    env = load_environment('two_sum').model_copy(update={'snapshot': 'numpy-env', 'image': image})

    with pytest.raises(DeepGymError, match='snapshot'):
        export_environment(env, tmp_path)
    assert not (tmp_path / 'two_sum').exists()


def test_export_never_overwrites(tmp_path: Path) -> None:
    env = load_environment('two_sum')
    export_environment(env, tmp_path)
    with pytest.raises(DeepGymError, match='already exists'):
        export_environment(env, tmp_path)
