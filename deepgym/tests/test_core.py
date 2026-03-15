"""Tests for deepgym.core — DeepGym client orchestration."""

from pathlib import Path

import pytest

from deepgym.core import DeepGym
from deepgym.models import BatchResult, Environment, RunResult

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'


class TestDeepGymInit:
    """Verify DeepGym initialization in local mode."""

    def test_local_mode(self) -> None:
        dg = DeepGym(mode='local')
        assert dg._local_executor is not None
        assert dg._daytona is None


class TestDeepGymRun:
    """Test single runs via DeepGym.run()."""

    def test_sorting_example(
        self, local_dg: DeepGym, sorting_env: Environment, sorting_solution: str
    ) -> None:
        result = local_dg.run(sorting_env, sorting_solution)
        assert isinstance(result, RunResult)
        assert result.passed is True
        assert result.score == 1.0
        assert result.sandbox_id == 'local'

    def test_two_sum_example(
        self, local_dg: DeepGym, two_sum_env: Environment, two_sum_solution: str
    ) -> None:
        result = local_dg.run(two_sum_env, two_sum_solution)
        assert isinstance(result, RunResult)
        assert result.passed is True
        assert result.score == 1.0

    def test_string_manipulation_example(
        self, local_dg: DeepGym, string_env: Environment, string_solution: str
    ) -> None:
        result = local_dg.run(string_env, string_solution)
        assert isinstance(result, RunResult)
        assert result.passed is True
        assert result.score == 1.0

    def test_bad_solution_scores_low(self, local_dg: DeepGym, sorting_env: Environment) -> None:
        bad_solution = '# empty file with no sort_list function\npass\n'
        result = local_dg.run(sorting_env, bad_solution)
        assert result.score < 1.0
        assert result.passed is False


class TestDeepGymRunBatch:
    """Test batch runs via DeepGym.run_batch()."""

    def test_ordered_results(
        self, local_dg: DeepGym, sorting_env: Environment, sorting_solution: str
    ) -> None:
        solutions = [sorting_solution, sorting_solution]
        batch = local_dg.run_batch(sorting_env, solutions)
        assert isinstance(batch, BatchResult)
        assert batch.total == 2
        assert len(batch.results) == 2
        # Both should pass with same high score.
        for r in batch.results:
            assert r.passed is True

    def test_mix_of_passing_and_failing(
        self, local_dg: DeepGym, sorting_env: Environment, sorting_solution: str
    ) -> None:
        good = sorting_solution
        bad = '# no sort_list here\npass\n'
        batch = local_dg.run_batch(sorting_env, [good, bad])
        assert batch.total == 2
        assert batch.passed >= 1
        assert batch.failed >= 1
        # The good solution should pass, the bad one should fail.
        scores = [r.score for r in batch.results]
        assert max(scores) == 1.0
        assert min(scores) < 1.0
