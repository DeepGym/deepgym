"""Tests for deepgym.verifier_template — wrap_verifier and JSON protocol."""

import contextlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from deepgym import AsyncDeepGym, DeepGym
from deepgym.exceptions import InvalidScoreError
from deepgym.integrations.reward import AsyncRewardFunction, RewardFunction
from deepgym.models import Environment
from deepgym.verifier_template import wrap_verifier


def _run_wrapped(verifier_body: str, solution_code: str) -> tuple[dict, int]:
    """Write wrapped verifier + solution to temp files, run, return (json_output, exit_code)."""
    wrapped = wrap_verifier(verifier_body)
    with tempfile.TemporaryDirectory(prefix='test_vt_') as tmpdir:
        tmp = Path(tmpdir)
        verifier_path = tmp / 'verifier.py'
        solution_path = tmp / 'solution.py'
        verifier_path.write_text(wrapped, encoding='utf-8')
        solution_path.write_text(solution_code, encoding='utf-8')

        proc = subprocess.run(
            [sys.executable, str(verifier_path), str(solution_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = json.loads(proc.stdout.strip())
        return output, proc.returncode


class TestWrapVerifierReturnTypes:
    """Verify that wrap_verifier normalizes different return types to the JSON protocol."""

    def test_float_return(self) -> None:
        body = 'return 0.85\n'
        output, code = _run_wrapped(body, '# empty solution\n')
        assert output['score'] == 0.85
        assert output['passed'] is True
        assert output['schema_version'] == '1.0'
        assert code == 0

    def test_float_return_failing(self) -> None:
        body = 'return 0.3\n'
        output, code = _run_wrapped(body, '# empty\n')
        assert output['score'] == 0.3
        assert output['passed'] is False
        assert code == 1

    def test_bool_return_true(self) -> None:
        body = 'return True\n'
        output, code = _run_wrapped(body, '# empty\n')
        # bool is subclass of int, so it hits the (int, float) branch.
        # True == 1 so score is 1.0 and passed is True (1.0 >= 0.5).
        assert output['score'] == 1.0
        assert output['passed'] is True
        assert code == 0

    def test_bool_return_false(self) -> None:
        body = 'return False\n'
        output, code = _run_wrapped(body, '# empty\n')
        # False == 0 so score is 0.0 and passed is False (0.0 >= 0.5 is False).
        assert output['score'] == 0.0
        assert output['passed'] is False
        assert code == 1

    def test_dict_return(self) -> None:
        body = 'return {"score": 0.75, "passed": True, "details": "3/4 ok"}\n'
        output, code = _run_wrapped(body, '# empty\n')
        assert output['score'] == 0.75
        assert output['passed'] is True
        assert output['details'] == '3/4 ok'
        assert code == 0

    def test_dict_return_with_extras(self) -> None:
        body = (
            'return {\n'
            '    "score": 1.0,\n'
            '    "passed": True,\n'
            '    "reward_components": {"style": 0.9},\n'
            '    "metrics": {"time_ms": 42},\n'
            '    "seed": 7,\n'
            '}\n'
        )
        output, code = _run_wrapped(body, '# empty\n')
        assert output['reward_components'] == {'style': 0.9}
        assert output['metrics'] == {'time_ms': 42}
        assert output['seed'] == 7

    @pytest.mark.parametrize('body', ['return "not a valid type"\n', 'pass\n'])
    def test_unsupported_return_is_a_verifier_error(self, body: str) -> None:
        output, code = _run_wrapped(body, '# empty\n')
        assert (output['score'], output['passed'], code) == (0.0, False, 2)
        assert output['error_type'] == 'runtime_error'
        assert 'expected a float, bool or dict' in output['details']

    @pytest.mark.parametrize('flag', ['passed', 'truncated'])
    def test_non_bool_flag_is_a_verifier_error(self, flag: str) -> None:
        body = f'return {{"score": 1.0, "passed": True, "{flag}": "false"}}\n'
        output, code = _run_wrapped(body, '# empty\n')
        assert (output['passed'], output['error_type'], code) == (False, 'runtime_error', 2)
        assert f"{flag} must be True or False, not 'false'" in output['details']

    def test_truncated_is_kept(self) -> None:
        body = 'return {"score": 0.5, "passed": False, "truncated": True}\n'
        output, code = _run_wrapped(body, '# empty\n')
        assert (output['truncated'], code) == (True, 1)

    def test_exception_in_user_code(self) -> None:
        body = 'raise ValueError("intentional")\n'
        output, code = _run_wrapped(body, '# empty\n')
        assert output['score'] == 0.0
        assert output['passed'] is False
        assert output['error_type'] == 'runtime_error'
        assert 'intentional' in output['details']
        assert code == 2


class TestSchemaVersion:
    """Verify schema_version is always '1.0' in output."""

    def test_schema_version_on_pass(self) -> None:
        output, _ = _run_wrapped('return 1.0\n', '# e\n')
        assert output['schema_version'] == '1.0'

    def test_schema_version_on_fail(self) -> None:
        output, _ = _run_wrapped('return 0.0\n', '# e\n')
        assert output['schema_version'] == '1.0'

    def test_schema_version_on_error(self) -> None:
        output, _ = _run_wrapped('raise RuntimeError("boom")\n', '# e\n')
        assert output['schema_version'] == '1.0'


class TestExitCodes:
    """Verify exit codes: 0 for pass, 1 for fail, 2 for error."""

    def test_exit_0_on_pass(self) -> None:
        _, code = _run_wrapped('return 1.0\n', '# e\n')
        assert code == 0

    def test_exit_1_on_fail(self) -> None:
        _, code = _run_wrapped('return 0.0\n', '# e\n')
        assert code == 1

    def test_exit_2_on_error(self) -> None:
        _, code = _run_wrapped('raise Exception("boom")\n', '# e\n')
        assert code == 2


class TestTruncatedField:
    """Verify truncated field is present and False for normal runs."""

    def test_truncated_false_on_pass(self) -> None:
        output, _ = _run_wrapped('return 1.0\n', '# e\n')
        assert output['truncated'] is False

    def test_truncated_false_on_error(self) -> None:
        output, _ = _run_wrapped('raise Exception("x")\n', '# e\n')
        assert output['truncated'] is False


class TestScoreValidation:
    """Out-of-range or non-finite scores are verifier errors, never clamped into reward."""

    @pytest.mark.parametrize('body', ['return 5.0\n', 'return -2.0\n', 'return float("nan")\n'])
    def test_invalid_score_is_an_error(self, body: str) -> None:
        output, code = _run_wrapped(body, '# e\n')
        assert (output['score'], output['error_type'], code) == (0.0, 'invalid_score', 2)

    def test_float_noise_is_clamped(self) -> None:
        output, _ = _run_wrapped('return 1.0000000001\n', '# e\n')
        assert output['score'] == 1.0


def _crash_after(score: float, mode: str) -> str:
    """A verifier that prints *score* as a failed run, then crashes outside any handler."""
    report = f'print(json.dumps({{"score": {score}, "passed": False}}), flush=True)'
    if mode == 'script':
        return f'import json\nif __name__ == "__main__":\n    {report}\n    raise ValueError\n'
    return (
        f'import json\n{report}\nraise RuntimeError("bug")\n\n'
        'def verify(solution_path, test_cases_path=None):\n    return 1.0\n'
    )


# Use the solution's answer with no handling of their own, so whatever the solution does
# wrong reaches the crash guard.
_CALLS_SOLUTION = (
    'from deepgym_solution import load_solution\n\n'
    'def verify(solution_path, test_cases_path=None):\n'
    '    return 1.0 if load_solution(solution_path).double(2) + 0 == 4 else 0.0\n'
)
_PARSES_OUTPUT = (
    'import json\n'
    'from deepgym_solution import run_solution\n\n'
    'def verify(solution_path, test_cases_path=None):\n'
    '    return 1.0 if json.loads(run_solution(solution_path).stdout) == 4 else 0.0\n'
)


class TestCrashGuard:
    """A crash never keeps a printed score (Python exits 1 on a crash, the protocol's "failed"
    code). Before the solution runs it is the verifier's failure, with no verdict; once the
    solution has run it is the solution's, a judged 0, so breaking the verifier never pays."""

    @pytest.mark.parametrize('mode', ['script', 'verify_function'])
    @pytest.mark.parametrize('score', [0.75, 0.0])
    async def test_a_crash_before_the_solution_runs_is_no_reward(
        self, mode: str, score: float
    ) -> None:
        env = Environment(task='t', verifier_code=_crash_after(score, mode), verifier_mode=mode)

        rewards = RewardFunction(env, DeepGym(mode='local'))(['x'])
        async_rewards = await AsyncRewardFunction(env, AsyncDeepGym(mode='local'))(['x'])

        assert rewards == async_rewards == [None]

    @pytest.mark.parametrize('mode', ['body', 'verify_function'])
    @pytest.mark.parametrize('result', ['float("nan")', '"four"'], ids=['nan', 'not_a_number'])
    @pytest.mark.parametrize('consulted', [False, True], ids=['own_result', 'after_the_solution'])
    def test_an_invalid_result_is_the_solutions_once_it_ran(
        self, mode: str, result: str, consulted: bool
    ) -> None:
        load = '    load_solution(solution_path)\n' if consulted else ''
        body = f'    from deepgym_solution import load_solution\n{load}    return {result}\n'
        if mode == 'verify_function':
            body = f'def verify(solution_path, test_cases_path=None):\n{body}'
        else:
            body = body.replace('\n    ', '\n').removeprefix('    ')
        env = Environment(task='t', verifier_code=body, verifier_mode=mode)

        if not consulted:  # the verifier's own result: no reward (an invalid score is refused)
            with contextlib.suppress(InvalidScoreError):
                assert not DeepGym(mode='local').run(env, 'x = 1\n').has_verdict
            return
        run = DeepGym(mode='local').run(env, 'x = 1\n')
        assert (run.has_verdict, run.score, run.error_type) == (True, 0.0, 'solution_error')

    @pytest.mark.parametrize(
        ('shape', 'mode'),
        [
            ('script', 'script'),
            ('verify_function', 'verify_function'),
            ('body', 'body'),
            ('body', 'auto'),
        ],
    )
    def test_running_the_solution_itself_counts_as_consulting_it(
        self, shape: str, mode: str
    ) -> None:
        runs_it = {
            'script': (
                'import json, sys\n\n'
                'if __name__ == "__main__":\n'
                '    exec(open(sys.argv[1]).read())\n'
                '    print(json.dumps({"score": 1.0, "passed": True}))\n'
            ),
            'verify_function': (
                'def verify(solution_path, test_cases_path=None):\n'
                '    exec(open(solution_path).read())\n'
                '    return 1.0\n'
            ),
            'body': 'exec(open(solution_path).read())\nreturn 1.0\n',
        }
        env = Environment(task='t', verifier_code=runs_it[shape], verifier_mode=mode)

        run = DeepGym(mode='local').run(env, 'raise ValueError("the solution\'s bug")\n')

        assert (run.has_verdict, run.score, run.error_type) == (True, 0.0, 'solution_error')

    def test_a_solution_failure_is_a_model_fault(self) -> None:
        verifier = (
            'from deepgym_solution import load_solution\nreturn load_solution(solution_path).f()\n'
        )
        env = Environment(task='t', verifier_code=verifier, verifier_mode='body')

        run = DeepGym(mode='local').run(env, 'def f():\n    raise ValueError("bug")\n')

        assert (run.has_verdict, run.score, run.error_type) == (True, 0.0, 'solution_error')

    def test_a_broken_stderr_never_keeps_a_printed_score(self) -> None:
        verifier = (
            'import json, sys\n\n'
            'class Full:\n'
            '    def write(self, text):\n'
            '        raise OSError(28, "No space left on device")\n'
            '    flush = write\n\n'
            'if __name__ == "__main__":\n'
            '    print(json.dumps({"score": 0.75, "passed": False}), flush=True)\n'
            '    sys.stderr = Full()  # a full disk under a file-backed stderr\n'
            '    raise ValueError\n'
        )
        env = Environment(task='t', verifier_code=verifier)

        result = DeepGym(mode='local').run(env, 'x = 1')

        assert (result.exit_code, result.has_verdict) == (2, False)

    async def test_a_crash_after_the_solution_runs_is_a_judged_zero(self) -> None:
        verifier = (
            'import json, sys\n'
            'from deepgym_solution import load_solution\n\n'
            'if __name__ == "__main__":\n'
            '    load_solution(sys.argv[1])\n'
            '    print(json.dumps({"score": 0.75, "passed": False}), flush=True)\n'
            '    raise ValueError\n'
        )
        env = Environment(task='t', verifier_code=verifier)

        rewards = RewardFunction(env, DeepGym(mode='local'))(['x = 1'])
        async_rewards = await AsyncRewardFunction(env, AsyncDeepGym(mode='local'))(['x = 1'])

        assert rewards == async_rewards == [0.0]

    @pytest.mark.parametrize(
        ('verifier', 'solution'),
        [
            (_CALLS_SOLUTION, 'def double(n):\n    raise ValueError("no")\n'),
            (_CALLS_SOLUTION, 'def double(n):\n    raise ImportError("not the verifier\'s")\n'),
            (_CALLS_SOLUTION, 'def double(n):\n    raise TimeoutError("not the verifier\'s")\n'),
            (_CALLS_SOLUTION, 'import os\n\ndef double(n):\n    os._exit(3)\n'),
            (_CALLS_SOLUTION, 'def double(n):\n    while True:\n        pass\n'),
            (_CALLS_SOLUTION, 'def double(n):\n    return "four"\n'),
            (_CALLS_SOLUTION, 'def double(n) return 4\n'),
            (_PARSES_OUTPUT, 'print("four")\n'),
            (_PARSES_OUTPUT, 'import sys\nsys.exit(3)\n'),
        ],
        ids=[
            'raises',
            'raises_import_error',
            'raises_timeout',
            'exits',
            'hangs',
            'returns_garbage',
            'syntax_error',
            'prints_garbage',
            'exits_nonzero',
        ],
    )
    def test_whatever_the_solution_does_wrong_is_a_judged_zero(
        self, verifier: str, solution: str
    ) -> None:
        env = Environment(task='t', verifier_code=verifier, case_timeout=1)

        result = DeepGym(mode='local').run(env, solution)

        assert (result.score, result.passed, result.has_verdict) == (0.0, False, True)

    @pytest.mark.parametrize(
        ('verifier', 'solution'),
        [(_CALLS_SOLUTION, 'def double(n):\n    return 2 * n\n'), (_PARSES_OUTPUT, 'print(4)\n')],
    )
    def test_a_working_solution_still_passes(self, verifier: str, solution: str) -> None:
        env = Environment(task='t', verifier_code=verifier)
        assert DeepGym(mode='local').run(env, solution).score == 1.0

    def test_a_verifier_setup_failure_has_no_verdict(self) -> None:
        verifier = (
            'from deepgym_solution import load_solution\n\n'
            'def verify(solution_path, test_cases_path=None):\n'
            '    cases = open("missing_cases.json").read()\n'
            '    return 1.0 if load_solution(solution_path).double(2) == 4 else 0.0\n'
        )
        env = Environment(task='t', verifier_code=verifier)

        result = DeepGym(mode='local').run(env, 'def double(n):\n    return 2 * n\n')

        assert (result.has_verdict, result.error_type) == (False, 'runtime_error')

    def test_a_deliberate_failed_exit_keeps_partial_credit(self) -> None:
        verifier = (
            'from __future__ import annotations\n'
            'import json, sys\n'
            'if __name__ == "__main__":\n'
            '    print(json.dumps({"score": 0.4, "passed": False}))\n'
            '    sys.exit(1)\n'
        )

        result = DeepGym(mode='local').run(Environment(task='t', verifier_code=verifier), 'x')

        assert (result.score, result.passed, result.has_verdict) == (0.4, False, True)
