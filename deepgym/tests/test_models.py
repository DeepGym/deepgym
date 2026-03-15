"""Tests for deepgym.models — Pydantic data models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym.models import (
    BatchResult,
    Environment,
    RunResult,
    VerifierResult,
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TestEnvironment:
    """Validate Environment model construction and validators."""

    def test_verifier_code_provided(self) -> None:
        env = Environment(task='Do something', verifier_code='return 1.0')
        assert env.verifier_code == 'return 1.0'
        assert env.verifier_path is None

    def test_verifier_path_provided(self, tmp_path: Path) -> None:
        p = tmp_path / 'v.py'
        p.write_text('return True')
        env = Environment(task='Do something', verifier_path=p)
        assert env.verifier_path == p

    def test_neither_verifier_code_nor_path_raises(self) -> None:
        with pytest.raises(ValidationError, match='verifier_code or verifier_path'):
            Environment(task='Do something')

    def test_default_values(self) -> None:
        env = Environment(task='t', verifier_code='return 1.0')
        assert env.language == 'python'
        assert env.timeout == 30
        assert env.difficulty == 'medium'
        assert env.domain == 'coding'
        assert env.tags == []
        assert env.test_cases is None
        assert env.snapshot is None
        assert env.env_vars is None

    def test_difficulty_literal_valid_values(self) -> None:
        for d in ('easy', 'medium', 'hard'):
            env = Environment(task='t', verifier_code='x', difficulty=d)
            assert env.difficulty == d

    def test_difficulty_literal_invalid_value(self) -> None:
        with pytest.raises(ValidationError):
            Environment(task='t', verifier_code='x', difficulty='extreme')


# ---------------------------------------------------------------------------
# VerifierResult
# ---------------------------------------------------------------------------


class TestVerifierResult:
    """Validate VerifierResult score bounds and defaults."""

    def test_valid_score(self) -> None:
        vr = VerifierResult(score=0.5, passed=True)
        assert vr.score == 0.5

    def test_score_zero(self) -> None:
        vr = VerifierResult(score=0.0, passed=False)
        assert vr.score == 0.0

    def test_score_one(self) -> None:
        vr = VerifierResult(score=1.0, passed=True)
        assert vr.score == 1.0

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            VerifierResult(score=-0.1, passed=False)

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            VerifierResult(score=1.1, passed=True)

    def test_default_fields(self) -> None:
        vr = VerifierResult(score=0.8, passed=True)
        assert vr.schema_version == '1.0'
        assert vr.details is None
        assert vr.reward_components is None
        assert vr.metrics is None
        assert vr.seed is None
        assert vr.truncated is False
        assert vr.error_type is None


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


class TestRunResult:
    """Validate RunResult creation with all fields."""

    def test_full_creation(self) -> None:
        rr = RunResult(
            score=0.9,
            passed=True,
            output='ok',
            stderr='',
            exit_code=0,
            execution_time_ms=123.4,
            sandbox_id='local',
            reward_components={'correctness': 0.9},
            metrics={'mem': 10},
            seed=42,
            truncated=False,
            error_type=None,
        )
        assert rr.score == 0.9
        assert rr.passed is True
        assert rr.sandbox_id == 'local'
        assert rr.reward_components == {'correctness': 0.9}
        assert rr.seed == 42

    def test_minimal_creation(self) -> None:
        rr = RunResult(
            score=0.0,
            passed=False,
            output='',
            stderr='',
            exit_code=1,
            execution_time_ms=0,
            sandbox_id='test',
        )
        assert rr.truncated is False
        assert rr.error_type is None


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------


class TestBatchResult:
    """Validate BatchResult aggregation."""

    def test_aggregation(self) -> None:
        r1 = RunResult(
            score=1.0,
            passed=True,
            output='',
            stderr='',
            exit_code=0,
            execution_time_ms=10,
            sandbox_id='a',
        )
        r2 = RunResult(
            score=0.0,
            passed=False,
            output='',
            stderr='',
            exit_code=1,
            execution_time_ms=20,
            sandbox_id='b',
        )
        br = BatchResult(
            results=[r1, r2],
            total=2,
            passed=1,
            failed=1,
            avg_score=0.5,
            execution_time_ms=30,
        )
        assert br.total == 2
        assert br.passed == 1
        assert br.failed == 1
        assert br.avg_score == 0.5
        assert len(br.results) == 2
