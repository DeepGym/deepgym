"""Tests for deepgym.verifier_template — wrap_verifier and JSON protocol."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

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
