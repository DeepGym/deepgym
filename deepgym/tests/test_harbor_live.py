"""Harbor tasks on real containers: the agent's limits, and grading it can't tamper with.

Apple container and Docker run when their CLI is up. Modal costs money, so it runs only
when named, e.g. DEEPGYM_LIVE_PROVIDERS=modal. E2B boots templates, not the registry
images Harbor tasks use, so it isn't covered.
"""

import os
import shutil
from pathlib import Path

import pytest

from deepgym import registry as registry_module
from deepgym.benchmark_envs import TerminalBenchEnvironment
from deepgym.core import DeepGym
from deepgym.harbor import HarborEnvironment, export_environment
from deepgym.qa import run_qa
from deepgym.registry import load_environment, register_benchmark
from deepgym.sandboxes.apple import AppleContainer
from deepgym.sandboxes.docker import DockerContainer
from tests.test_harbor import HELLO, TASKS

pytestmark = pytest.mark.xdist_group('live-containers')

_PROVIDERS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(not usable, reason=f'{name} not available or not enabled'),
    )
    for name, usable in [
        ('apple', AppleContainer.available()),
        ('docker', DockerContainer.available()),
        ('modal', 'modal' in os.environ.get('DEEPGYM_LIVE_PROVIDERS', '').split(',')),
    ]
]

# Every way the agent might plant a passing reward or replace the test runner, then a wrong
# answer. Each line that needs root or the grader's sandbox should simply fail.
_ATTACK = r"""
mkdir -p /logs/verifier /tests 2>/dev/null
echo 1 > /logs/verifier/reward.txt 2>/dev/null
echo '{"reward": 1}' > /logs/verifier/reward.json 2>/dev/null
mkdir -p /app/logs/verifier /app/tests
echo 1 > /app/logs/verifier/reward.txt
echo '{"reward": 1}' > /app/reward.json
printf '#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n' > /app/tests/test.sh
cp /app/tests/test.sh /tests/test.sh 2>/dev/null
for bin in /bin/bash /usr/bin/bash /bin/sh /usr/bin/cat /usr/local/bin/python3; do
  cp /app/tests/test.sh "$bin" 2>/dev/null
done
(setsid sh -c 'while :; do echo 1 > /logs/verifier/reward.txt; sleep 0.1; done' \
  >/dev/null 2>&1 &)
echo wrong > /app/hello.txt
"""

_PROBE = r"""
id -u > probe.uid
python3 -c "import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)" \
  2>/dev/null && echo online > probe.net || echo offline > probe.net
touch /usr/bin/dg-probe 2>/dev/null && echo yes > probe.root || echo no > probe.root
"""

_PROBE_TEST = """#!/bin/bash
if [ "$(cat probe.uid)" != 0 ] && [ "$(cat probe.net)" = offline ] \\
  && [ "$(cat probe.root)" = no ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
"""


@pytest.fixture(scope='module', params=_PROVIDERS)
def dg(request: pytest.FixtureRequest) -> DeepGym:
    return DeepGym(mode=request.param)


@pytest.fixture()
def hello() -> HarborEnvironment:
    return HarborEnvironment.from_dir(HELLO)


def test_reference_passes_and_noop_fails(dg: DeepGym, hello: HarborEnvironment) -> None:
    assert dg.run(hello, hello.reference_solution).score == 1.0
    assert dg.run(hello, '').score == 0.0


def test_agent_runs_unprivileged_and_offline(dg: DeepGym, tmp_path: Path) -> None:
    task = tmp_path / 'probe'
    shutil.copytree(HELLO, task)
    (task / 'tests' / 'test.sh').write_text(_PROBE_TEST, encoding='utf-8')

    result = dg.run(HarborEnvironment.from_dir(task), _PROBE)

    assert result.score == 1.0, result.stderr


def test_prewritten_rewards_and_trojaned_runners_score_zero(
    dg: DeepGym, hello: HarborEnvironment
) -> None:
    result = dg.run(hello, _ATTACK)

    assert result.score == 0.0
    assert result.output.startswith('reward.txt: 0')


def test_terminal_bench_runs_its_tasks_through_the_harbor_runner(dg: DeepGym) -> None:
    env = TerminalBenchEnvironment(task_root=TASKS)

    assert dg.run(env, 'echo hello > /app/hello.txt', task_id='hello').score == 1.0
    assert dg.run(env, 'echo nope > /app/hello.txt', task_id='hello').score == 0.0


def test_exported_environment_round_trips_through_a_built_image(
    dg: DeepGym, tmp_path: Path
) -> None:
    if dg.backend not in ('apple', 'docker'):
        pytest.skip('only the container CLIs build Dockerfiles')
    env = HarborEnvironment.from_dir(export_environment(load_environment('coin_change'), tmp_path))

    assert dg.run(env, env.reference_solution).score == 1.0
    assert dg.run(env, '').score == 0.0


# Decoy programs the agent leaves in its work directory: each just fails. If any root
# step of the harness ran one, the run would error instead of scoring.
_DECOYS = 'bash sh cat tar chmod chown mkdir rm base64 wc env timeout python3 realpath'
_LEAVE_DECOYS = (
    'mkdir -p /app/bin && for name in ' + _DECOYS + '; do '
    "printf '#!/bin/sh\\nexit 3\\n' > /app/bin/$name && chmod +x /app/bin/$name; done\n"
)


def test_exported_reward_parser_ignores_modules_in_the_work_directory(
    dg: DeepGym, tmp_path: Path
) -> None:
    if dg.backend not in ('apple', 'docker'):
        pytest.skip('only the container CLIs build Dockerfiles')
    env = HarborEnvironment.from_dir(export_environment(load_environment('coin_change'), tmp_path))
    decoy = "printf 'raise SystemExit(3)\\n' > /app/json.py\n"

    assert dg.run(env, env.reference_solution + decoy).score == 1.0


def test_harness_ignores_programs_on_an_image_path_in_the_work_directory(
    dg: DeepGym, tmp_path: Path
) -> None:
    if dg.backend not in ('apple', 'docker'):
        pytest.skip('only the container CLIs build Dockerfiles')
    task = tmp_path / 'path-in-workdir'
    shutil.copytree(HELLO, task)
    (task / 'task.toml').write_text('[environment]\nworkdir = "/app"\n', encoding='utf-8')
    (task / 'environment').mkdir()
    (task / 'environment' / 'Dockerfile').write_text(
        'FROM python:3.12-slim\nENV PATH=/app/bin:$PATH\nWORKDIR /app\n', encoding='utf-8'
    )
    (task / 'tests' / 'test.sh').write_text(
        '#!/bin/bash\n'
        'case "$(command -v cat)" in /app/*) echo 0; exit 0;; esac > /logs/verifier/reward.txt\n'
        'if [ "$(cat /app/hello.txt)" = hello ]; then echo 1; else echo 0; fi'
        ' > /logs/verifier/reward.txt\n',
        encoding='utf-8',
    )

    env = HarborEnvironment.from_dir(task)

    assert dg.run(env, 'echo hello > /app/hello.txt\n' + _LEAVE_DECOYS).score == 1.0


def test_qa_gates_a_registered_harbor_task(
    dg: DeepGym, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_module, '_get_cache_dir', lambda: tmp_path)
    register_benchmark('fixtures', [HELLO], {})

    report = run_qa('fixtures/hello', dg, reruns=0)

    checks = {check.name: check.status for check in report.checks}
    assert report.status == 'pass', report.checks
    assert checks['reference_passes'] == 'pass' and checks['empty_fails'] == 'pass'
    assert checks['hack_resistant'] == 'skip'
