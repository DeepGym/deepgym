"""Live Daytona: the solution runs as a separate user and can't game the judge."""

import os

import pytest

from deepgym import DeepGym, load_environment
from deepgym.models import Environment
from tests.test_isolation import WRONG, _env
from tests.test_session import ALWAYS_EQUAL, FIXED, debug_fix_score

pytestmark = pytest.mark.skipif(
    not os.getenv('DAYTONA_API_KEY'), reason='needs DAYTONA_API_KEY for live sandboxes'
)


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    return DeepGym(mode='daytona')


def test_reference_passes_and_wrong_fails(dg: DeepGym) -> None:
    env = load_environment('coin_change')
    reference = env.verifier_path.with_name('reference_solution.py').read_text()
    assert dg.run(env, reference).score == 1.0
    assert dg.run(env, WRONG).score < 1.0


@pytest.mark.parametrize(
    'hack',
    [
        'import os\nprint(\'{"score": 1.0, "passed": true}\', flush=True)\nos._exit(0)\n',
        'class A:\n    def __eq__(self, o):\n        return True\n'
        'def coin_change(coins, amount):\n    return A()\n',
        'import __main__\n__main__.ref_coin_change = lambda c, a: 0\n' + WRONG,
    ],
    ids=['fake_score_line', 'equals_everything', 'patch_judge'],
)
def test_hacks_fail(dg: DeepGym, hack: str) -> None:
    assert dg.run(load_environment('coin_change'), hack).score < 1.0


def test_solution_user_cannot_touch_judge(dg: DeepGym) -> None:
    env = _env(
        """
import os
from pathlib import Path
secret = Path(__file__).with_name('secret.txt')
secret.write_text('s3cret')
facts = sol.probe(str(secret), os.getpid())
leaks = [name for name, value in facts.items() if value]
return {'score': 0.0 if leaks else 1.0, 'passed': not leaks, 'details': str(facts)}
"""
    )
    solution = (
        'import os, signal\n'
        'def probe(secret, judge_pid):\n'
        '    facts = {}\n'
        '    for name, fn in {\n'
        '        "read_secret": lambda: open(secret).read(),\n'
        '        "list_judge": lambda: os.listdir(os.path.dirname(secret)),\n'
        '        "judge_stdout": lambda: open(f"/proc/{judge_pid}/fd/1", "w").write("x"),\n'
        '        "signal_judge": lambda: os.kill(judge_pid, 0),\n'
        '        "is_root": lambda: os.getuid() == 0 or None,\n'
        '    }.items():\n'
        '        try:\n'
        '            facts[name] = bool(fn())\n'
        '        except Exception:\n'
        '            facts[name] = False\n'
        '    return facts\n'
    )
    result = dg.run(env, solution)
    assert result.passed, result.output


def test_slow_case_fails_alone(dg: DeepGym) -> None:
    env = _env(
        """
ok = 0
for n in (1, 2, 3):
    try:
        ok += sol.double(n) == 2 * n
    except TimeoutError:
        pass
return ok / 3
""",
        case_timeout=2,
        timeout=30,
    )
    solution = 'def double(n):\n    while n == 2:\n        pass\n    return 2 * n\n'
    assert dg.run(env, solution).score == pytest.approx(2 / 3)


def test_script_env_and_support_files(dg: DeepGym) -> None:
    env = load_environment('file_organizer')
    reference = env.verifier_path.with_name('reference_solution.py').read_text()
    assert dg.run(env, reference).passed
    helper = Environment(
        task='t',
        verifier_code='from scoring import half\n\ndef verify(solution_path):\n    return half()\n',
        support_files={'scoring.py': 'def half():\n    return 0.5\n'},
    )
    assert dg.run(helper, '').score == 0.5


def test_setsid_daemon_dies_with_its_script(dg: DeepGym) -> None:
    verifier = (
        'import tempfile, time\n'
        'from pathlib import Path\n'
        'from deepgym_solution import run_solution\n\n'
        'def verify(solution_path):\n'
        '    work = tempfile.mkdtemp()\n'
        '    run_solution(solution_path, cwd=work, env={"OUT": work})\n'
        '    time.sleep(3)\n'
        '    return 0.0 if Path(work, "late.txt").exists() else 1.0\n'
    )
    solution = (
        'import os, time\n'
        'if os.fork() == 0:\n'
        '    os.setsid()\n'
        '    time.sleep(1.5)\n'
        '    open(os.path.join(os.environ["OUT"], "late.txt"), "w").write("tampered")\n'
        '    os._exit(0)\n'
    )
    assert dg.run(Environment(task='t', verifier_code=verifier), solution).passed


def test_session_verifier_loads_agent_code_as_another_user(dg: DeepGym) -> None:
    for buggy, expected in ((FIXED, 1.0), (ALWAYS_EQUAL, 0.0)):
        assert debug_fix_score(dg, buggy) == expected


def test_judge_temp_files_are_private(dg: DeepGym) -> None:
    env = _env(
        """
import tempfile
from pathlib import Path
private = [Path(tempfile.gettempdir(), 'expected.json'), Path('/tmp/expected2.json')]
for path in private:
    path.write_text('EXPECTED')
leaked = [str(p) for p in private if sol.read(str(p))]
return {'score': 0.0 if leaked else 1.0, 'passed': not leaked, 'details': str(leaked)}
"""
    )
    solution = (
        'def read(path):\n'
        '    try:\n'
        '        return open(path).read()\n'
        '    except OSError:\n'
        '        return None\n'
    )
    result = dg.run(env, solution)
    assert result.passed, result.output
