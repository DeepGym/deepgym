"""Tests for deepgym.sandbox.LocalExecutor."""

from pathlib import Path

import pytest

from deepgym.exceptions import SandboxError, TimeoutError, VerifierError
from deepgym.models import VerifierResult
from deepgym.sandbox import ExecutionResult, LocalExecutor

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'


@pytest.fixture()
def executor() -> LocalExecutor:
    """Return a fresh LocalExecutor."""
    return LocalExecutor()


def _standalone_verifier(return_expr: str) -> str:
    """Build a standalone verifier script that returns the given expression.

    Unlike wrap_verifier, standalone scripts use sys.exit() so stdout is
    properly flushed before process exit.
    """
    return (
        '#!/usr/bin/env python3\n'
        'import sys, json\n'
        'def _run(solution_path, test_cases_path=None):\n'
        f'    {return_expr}\n'
        'if __name__ == "__main__":\n'
        '    sp = sys.argv[1] if len(sys.argv) > 1 else "solution.py"\n'
        '    result = _run(sp)\n'
        '    if isinstance(result, (int, float)):\n'
        '        out = {"schema_version":"1.0","score":float(result),"passed":float(result)>=0.5,"details":None,"truncated":False}\n'
        '    elif isinstance(result, dict):\n'
        '        out = {"schema_version":"1.0","score":float(result.get("score",0)),"passed":bool(result.get("passed",False)),"details":result.get("details"),"truncated":False}\n'
        '    else:\n'
        '        out = {"schema_version":"1.0","score":0.0,"passed":False,"details":"bad type","truncated":False}\n'
        '    out["score"] = max(0.0, min(1.0, out["score"]))\n'
        '    print(json.dumps(out))\n'
        '    sys.exit(0 if out["passed"] else 1)\n'
    )


def _make_ready_verifier(verifier_path: Path) -> str:
    """Read a standalone verifier file and return it as-is."""
    return verifier_path.read_text(encoding='utf-8')


class TestLocalExecutorRun:
    """Core run behaviour of LocalExecutor."""

    def test_passing_solution(self, executor: LocalExecutor) -> None:
        verifier_code = _standalone_verifier('return 1.0')
        result = executor.run(
            verifier_code=verifier_code,
            solution_code='# anything\n',
            timeout=10,
        )
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.verifier_result, VerifierResult)
        assert result.verifier_result.score == 1.0
        assert result.verifier_result.passed is True
        assert isinstance(result.stderr, str)
        assert isinstance(result.exit_code, int)

    def test_failing_solution(self, executor: LocalExecutor) -> None:
        verifier_code = _standalone_verifier('return 0.0')
        result = executor.run(
            verifier_code=verifier_code,
            solution_code='# anything\n',
            timeout=10,
        )
        assert isinstance(result, ExecutionResult)
        assert result.verifier_result.score == 0.0
        assert result.verifier_result.passed is False

    def test_timeout(self, executor: LocalExecutor) -> None:
        slow_verifier = (
            '#!/usr/bin/env python3\n'
            'import sys, time, json\n'
            'time.sleep(60)\n'
            'print(json.dumps({"schema_version":"1.0","score":1.0,"passed":True,"truncated":False}))\n'
        )
        with pytest.raises(TimeoutError):
            executor.run(
                verifier_code=slow_verifier,
                solution_code='# slow\n',
                timeout=1,
            )

    def test_invalid_verifier_no_json_output(self, executor: LocalExecutor) -> None:
        bad_verifier = (
            '#!/usr/bin/env python3\nimport sys\nprint("this is not json")\nsys.exit(0)\n'
        )
        with pytest.raises(VerifierError):
            executor.run(
                verifier_code=bad_verifier,
                solution_code='# anything\n',
                timeout=10,
            )

    def test_result_is_execution_result_instance(self, executor: LocalExecutor) -> None:
        verifier_code = _standalone_verifier(
            'return {"score": 0.7, "passed": True, "details": "ok"}'
        )
        result = executor.run(
            verifier_code=verifier_code,
            solution_code='# x\n',
            timeout=10,
        )
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.verifier_result, VerifierResult)
        assert result.verifier_result.details == 'ok'

    def test_invalid_env_var_key_rejected(self, executor: LocalExecutor) -> None:
        verifier_code = _standalone_verifier('return 1.0')
        with pytest.raises(SandboxError, match='Invalid environment variable name'):
            executor.run(
                verifier_code=verifier_code,
                solution_code='# x\n',
                timeout=10,
                env_vars={'; rm -rf /': 'bad'},
            )


class TestExampleEnvironments:
    """Ensure all 3 example environments pass with their reference solutions."""

    def test_sorting_example(self, executor: LocalExecutor) -> None:
        verifier_code = _make_ready_verifier(ENVS_DIR / 'python_sorting' / 'verifier.py')
        solution_code = (ENVS_DIR / 'python_sorting' / 'reference_solution.py').read_text(
            encoding='utf-8'
        )
        result = executor.run(
            verifier_code=verifier_code,
            solution_code=solution_code,
            timeout=30,
        )
        assert result.verifier_result.passed is True
        assert result.verifier_result.score == 1.0

    def test_two_sum_example(self, executor: LocalExecutor) -> None:
        verifier_code = _make_ready_verifier(ENVS_DIR / 'two_sum' / 'verifier.py')
        solution_code = (ENVS_DIR / 'two_sum' / 'reference_solution.py').read_text(encoding='utf-8')
        result = executor.run(
            verifier_code=verifier_code,
            solution_code=solution_code,
            timeout=30,
        )
        assert result.verifier_result.passed is True
        assert result.verifier_result.score == 1.0

    def test_string_manipulation_example(self, executor: LocalExecutor) -> None:
        verifier_code = _make_ready_verifier(ENVS_DIR / 'string_manipulation' / 'verifier.py')
        solution_code = (ENVS_DIR / 'string_manipulation' / 'reference_solution.py').read_text(
            encoding='utf-8'
        )
        result = executor.run(
            verifier_code=verifier_code,
            solution_code=solution_code,
            timeout=30,
        )
        assert result.verifier_result.passed is True
        assert result.verifier_result.score == 1.0
