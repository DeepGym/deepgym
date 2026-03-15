"""Tests for per-test-case trace support.

Verify that CaseResult validates correctly, verifiers with cases
return them through VerifierResult -> RunResult -> GymInfo, and
RewardFunction.per_test_rewards works for fine-grained reward shaping.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym.core import DeepGym
from deepgym.gym import DeepGymEnv, GymInfo
from deepgym.integrations.reward import RewardFunction
from deepgym.models import (
    Environment,
    RunResult,
    CaseResult,
    VerifierResult,
)

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'
COIN_CHANGE_DIR = ENVS_DIR / 'coin_change'


# ---------------------------------------------------------------------------
# CaseResult validation
# ---------------------------------------------------------------------------


class TestCaseResultModel:
    """Validate CaseResult model construction and constraints."""

    def test_minimal_creation(self) -> None:
        """Create with only required field (passed)."""
        tc = CaseResult(passed=True)
        assert tc.passed is True
        assert tc.id == ''
        assert tc.score == 1.0
        assert tc.input_summary == ''
        assert tc.expected_summary == ''
        assert tc.actual_summary == ''
        assert tc.error is None
        assert tc.execution_time_ms == 0.0

    def test_full_creation(self) -> None:
        """Create with all fields populated."""
        tc = CaseResult(
            id='test_0',
            passed=False,
            score=0.0,
            input_summary='coins=[1,5], amount=11',
            expected_summary='3',
            actual_summary='-1',
            error=None,
            execution_time_ms=1.5,
        )
        assert tc.id == 'test_0'
        assert tc.passed is False
        assert tc.score == 0.0
        assert tc.input_summary == 'coins=[1,5], amount=11'

    def test_score_bounds_low(self) -> None:
        """Score below 0.0 is rejected."""
        with pytest.raises(ValidationError):
            CaseResult(passed=False, score=-0.1)

    def test_score_bounds_high(self) -> None:
        """Score above 1.0 is rejected."""
        with pytest.raises(ValidationError):
            CaseResult(passed=True, score=1.1)

    def test_with_error(self) -> None:
        """Create with error message."""
        tc = CaseResult(passed=False, score=0.0, error='IndexError: out of range')
        assert tc.error == 'IndexError: out of range'


# ---------------------------------------------------------------------------
# VerifierResult with cases
# ---------------------------------------------------------------------------


class TestVerifierResultWithCases:
    """Validate VerifierResult with per-test-case breakdown."""

    def test_cases_default_none(self) -> None:
        """Cases default to None."""
        vr = VerifierResult(score=0.5, passed=True)
        assert vr.cases is None

    def test_cases_populated(self) -> None:
        """VerifierResult accepts a list of CaseResult."""
        cases = [
            CaseResult(id='test_0', passed=True, score=1.0),
            CaseResult(id='test_1', passed=False, score=0.0),
        ]
        vr = VerifierResult(score=0.5, passed=False, cases=cases)
        assert vr.cases is not None
        assert len(vr.cases) == 2
        assert vr.cases[0].id == 'test_0'
        assert vr.cases[1].passed is False

    def test_cases_from_json(self) -> None:
        """VerifierResult parses cases from JSON (as verifier output would produce)."""
        json_str = (
            '{"schema_version":"1.0","score":0.67,"passed":false,'
            '"cases":[{"id":"t0","passed":true,"score":1.0},'
            '{"id":"t1","passed":false,"score":0.0},'
            '{"id":"t2","passed":true,"score":1.0}]}'
        )
        vr = VerifierResult.model_validate_json(json_str)
        assert vr.score == 0.67
        assert vr.cases is not None
        assert len(vr.cases) == 3
        assert vr.cases[0].id == 't0'
        assert vr.cases[1].passed is False

    def test_cases_empty_list(self) -> None:
        """Empty cases list is valid."""
        vr = VerifierResult(score=0.0, passed=False, cases=[])
        assert vr.cases == []


# ---------------------------------------------------------------------------
# RunResult with cases
# ---------------------------------------------------------------------------


class TestRunResultWithCases:
    """Validate RunResult passes cases through."""

    def test_cases_default_none(self) -> None:
        """RunResult.cases defaults to None."""
        rr = RunResult(
            score=0.5,
            passed=True,
            output='ok',
            stderr='',
            exit_code=0,
            execution_time_ms=100.0,
            sandbox_id='local',
        )
        assert rr.cases is None

    def test_cases_populated(self) -> None:
        """RunResult carries cases from verifier."""
        cases = [
            CaseResult(id='test_0', passed=True),
            CaseResult(id='test_1', passed=False, score=0.0),
        ]
        rr = RunResult(
            score=0.5,
            passed=False,
            output='1/2 passed',
            stderr='',
            exit_code=1,
            execution_time_ms=50.0,
            sandbox_id='local',
            cases=cases,
        )
        assert rr.cases is not None
        assert len(rr.cases) == 2
        assert rr.cases[0].passed is True


# ---------------------------------------------------------------------------
# Coin change verifier emits per-test traces (end-to-end)
# ---------------------------------------------------------------------------


class TestCoinChangeTracesEndToEnd:
    """Verify the coin_change verifier emits per-test-case traces."""

    @pytest.fixture()
    def coin_env(self) -> Environment:
        """Return the coin_change environment."""
        return Environment(
            task=(COIN_CHANGE_DIR / 'task.md').read_text(encoding='utf-8'),
            verifier_path=COIN_CHANGE_DIR / 'verifier.py',
            difficulty='medium',
        )

    @pytest.fixture()
    def coin_solution(self) -> str:
        """Return the reference solution for coin_change."""
        return (COIN_CHANGE_DIR / 'reference_solution.py').read_text(encoding='utf-8')

    @pytest.fixture()
    def local_dg(self) -> DeepGym:
        """Return a local-mode DeepGym client."""
        return DeepGym(mode='local')

    def test_cases_flow_through_to_run_result(
        self, coin_env: Environment, coin_solution: str, local_dg: DeepGym
    ) -> None:
        """Coin change verifier returns per-test cases in RunResult."""
        result = local_dg.run(coin_env, coin_solution)
        assert result.cases is not None
        assert len(result.cases) > 0
        # All should pass for the reference solution
        for case in result.cases:
            assert case.passed is True
            assert case.score == 1.0
            assert case.id.startswith('test_')

    def test_cases_have_input_summaries(
        self, coin_env: Environment, coin_solution: str, local_dg: DeepGym
    ) -> None:
        """Each case has an input_summary describing the test."""
        result = local_dg.run(coin_env, coin_solution)
        assert result.cases is not None
        for case in result.cases:
            assert 'coins=' in case.input_summary
            assert 'amount=' in case.input_summary

    def test_failing_solution_has_failed_cases(
        self, coin_env: Environment, local_dg: DeepGym
    ) -> None:
        """A bad solution produces failed test cases."""
        bad_solution = 'def coin_change(coins, amount):\n    return 0\n'
        result = local_dg.run(coin_env, bad_solution)
        assert result.cases is not None
        failed = [c for c in result.cases if not c.passed]
        assert len(failed) > 0


# ---------------------------------------------------------------------------
# Cases flow through to GymInfo
# ---------------------------------------------------------------------------


class TestCasesInGymInfo:
    """Verify cases flow through DeepGymEnv step() to GymInfo."""

    @pytest.fixture()
    def coin_env(self) -> Environment:
        """Return the coin_change environment."""
        return Environment(
            task=(COIN_CHANGE_DIR / 'task.md').read_text(encoding='utf-8'),
            verifier_path=COIN_CHANGE_DIR / 'verifier.py',
            difficulty='medium',
        )

    @pytest.fixture()
    def coin_solution(self) -> str:
        """Return the reference solution for coin_change."""
        return (COIN_CHANGE_DIR / 'reference_solution.py').read_text(encoding='utf-8')

    @pytest.fixture()
    def local_dg(self) -> DeepGym:
        """Return a local-mode DeepGym client."""
        return DeepGym(mode='local')

    def test_gym_info_has_cases(
        self, coin_env: Environment, coin_solution: str, local_dg: DeepGym
    ) -> None:
        """GymInfo.cases is populated after stepping with a verifier that emits cases."""
        env = DeepGymEnv(coin_env, dg=local_dg)
        env.reset()
        _, _, _, info = env.step(coin_solution)
        assert isinstance(info, GymInfo)
        assert info.cases is not None
        assert len(info.cases) > 0

    def test_gym_info_cases_none_without_traces(self, local_dg: DeepGym) -> None:
        """GymInfo.cases is None when the verifier does not emit cases."""
        simple_env = Environment(
            task='Write solve(x) -> x*2',
            verifier_code=(
                'import importlib.util\n'
                'spec = importlib.util.spec_from_file_location("solution", solution_path)\n'
                'mod = importlib.util.module_from_spec(spec)\n'
                'spec.loader.exec_module(mod)\n'
                'if hasattr(mod, "solve") and mod.solve(2) == 4:\n'
                '    return 1.0\n'
                'return 0.0\n'
            ),
        )
        env = DeepGymEnv(simple_env, dg=local_dg)
        env.reset()
        _, _, _, info = env.step('def solve(x):\n    return x * 2\n')
        assert info.cases is None


# ---------------------------------------------------------------------------
# RewardFunction.per_test_rewards
# ---------------------------------------------------------------------------


class TestPerTestRewards:
    """Verify RewardFunction.per_test_rewards returns per-test breakdowns."""

    @pytest.fixture()
    def coin_env(self) -> Environment:
        """Return the coin_change environment."""
        return Environment(
            task=(COIN_CHANGE_DIR / 'task.md').read_text(encoding='utf-8'),
            verifier_path=COIN_CHANGE_DIR / 'verifier.py',
            difficulty='medium',
        )

    @pytest.fixture()
    def coin_solution(self) -> str:
        """Return the reference solution for coin_change."""
        return (COIN_CHANGE_DIR / 'reference_solution.py').read_text(encoding='utf-8')

    @pytest.fixture()
    def local_dg(self) -> DeepGym:
        """Return a local-mode DeepGym client."""
        return DeepGym(mode='local')

    def test_empty_returns_empty(self, coin_env: Environment, local_dg: DeepGym) -> None:
        """per_test_rewards([]) returns []."""
        rf = RewardFunction(env=coin_env, dg=local_dg)
        assert rf.per_test_rewards([]) == []

    def test_returns_per_test_breakdown(
        self, coin_env: Environment, coin_solution: str, local_dg: DeepGym
    ) -> None:
        """per_test_rewards returns dict with test IDs and overall score."""
        rf = RewardFunction(env=coin_env, dg=local_dg)
        rewards = rf.per_test_rewards([coin_solution])
        assert len(rewards) == 1
        reward_dict = rewards[0]
        assert 'overall' in reward_dict
        assert reward_dict['overall'] >= 0.9
        # Should have test_0, test_1, ... keys
        test_keys = [k for k in reward_dict if k != 'overall']
        assert len(test_keys) > 0
        for k in test_keys:
            assert k.startswith('test_')
            assert reward_dict[k] in (0.0, 1.0)

    def test_fallback_without_cases(self, local_dg: DeepGym) -> None:
        """per_test_rewards falls back to overall when verifier has no cases."""
        simple_env = Environment(
            task='Write solve(x) -> x*2',
            verifier_code=(
                'import importlib.util\n'
                'spec = importlib.util.spec_from_file_location("solution", solution_path)\n'
                'mod = importlib.util.module_from_spec(spec)\n'
                'spec.loader.exec_module(mod)\n'
                'if hasattr(mod, "solve") and mod.solve(2) == 4:\n'
                '    return 1.0\n'
                'return 0.0\n'
            ),
        )
        rf = RewardFunction(env=simple_env, dg=local_dg)
        rewards = rf.per_test_rewards(['def solve(x):\n    return x * 2\n'])
        assert len(rewards) == 1
        assert rewards[0] == {'overall': 1.0}

    def test_multiple_outputs(
        self, coin_env: Environment, coin_solution: str, local_dg: DeepGym
    ) -> None:
        """per_test_rewards handles multiple outputs."""
        bad_solution = 'def coin_change(coins, amount):\n    return 0\n'
        rf = RewardFunction(env=coin_env, dg=local_dg)
        rewards = rf.per_test_rewards([coin_solution, bad_solution])
        assert len(rewards) == 2
        # Good solution should score higher overall
        assert rewards[0]['overall'] > rewards[1]['overall']
