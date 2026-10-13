"""Tests for deepgym.core — DeepGym client orchestration."""

import asyncio
from pathlib import Path

import pytest

from deepgym.async_core import AsyncDeepGym
from deepgym.core import DeepGym, build_eval_result
from deepgym.exceptions import SandboxError, VerifierError
from deepgym.models import BatchResult, Environment, EvalResult, RunResult

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


def _eval(client: str, outputs: dict[str, str]) -> EvalResult:
    """Evaluate the patched suite with the sync or async local client."""
    if client == 'sync':
        return DeepGym(mode='local').eval('suite', outputs)
    return asyncio.run(AsyncDeepGym(mode='local').eval('suite', outputs))


class TestEval:
    """eval() keeps suite order, names each result's environment, and classifies failures."""

    @pytest.fixture()
    def outputs(self, sorting_env: Environment, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
        """Answers for a suite of a sandbox failure, an unanswered env and an inline env."""
        suite = [
            # Keyed python_sorting by its verifier file; the env var name fails the sandbox.
            sorting_env.model_copy(update={'env_vars': {'BAD NAME': 'x'}}),
            Environment(name='unanswered', task='t', verifier_code='return 1.0'),
            # Named but without a verifier file, like a repo-patch environment.
            Environment(name='inline', task='t', verifier_code='return 1.0'),
        ]
        monkeypatch.setattr('deepgym.registry.load_suite', lambda name: suite)
        return {'python_sorting': '', 'inline': ''}

    @pytest.mark.parametrize('client', ['sync', 'async'])
    def test_results_follow_the_suite(self, outputs: dict[str, str], client: str) -> None:
        result = _eval(client, outputs)
        assert result.env_names == ['python_sorting', 'unanswered', 'inline']
        assert [r.sandbox_id for r in result.results] == ['error', 'skipped', 'local']

    @pytest.mark.parametrize('client', ['sync', 'async'])
    def test_sandbox_failure_has_no_verdict(self, outputs: dict[str, str], client: str) -> None:
        failed = _eval(client, outputs).results[0]
        assert (failed.error_type, failed.has_verdict) == ('sandbox_error', False)

    def test_named_env_without_a_verifier_file_runs(self, outputs: dict[str, str]) -> None:
        assert _eval('sync', outputs).results[2].score == 1.0

    @pytest.mark.parametrize('client', ['sync', 'async'])
    def test_totals_count_no_verdict_as_zero_and_judged_apart(
        self, outputs: dict[str, str], client: str
    ) -> None:
        result = _eval(client, outputs)
        # The sandbox failure has no verdict; the unanswered env is a judged 0.
        assert (result.total, result.passed, result.no_verdict_count) == (3, 1, 1)
        assert result.pass_rate == result.avg_score == pytest.approx(1 / 3)
        assert result.judged_count == 2
        assert (result.judged_pass_rate, result.judged_mean_score) == (0.5, 0.5)


def test_runs_without_a_verdict_never_raise_the_headline() -> None:
    judged = [_judged_result(1.0), _judged_result(0.0)]
    baseline = build_eval_result('suite', 'm', ['a', 'b'], judged)

    with_unscored = build_eval_result(
        'suite', 'm', ['a', 'b', 'c', 'd'], [*judged, SandboxError('down'), VerifierError('bug')]
    )

    assert with_unscored.pass_rate < baseline.pass_rate == 0.5
    assert with_unscored.avg_score < baseline.avg_score == 0.5
    assert (with_unscored.judged_pass_rate, with_unscored.no_verdict_count) == (0.5, 2)


def _judged_result(score: float) -> RunResult:
    """A judged result with *score*, passed when it is 1.0."""
    return RunResult(
        score=score, passed=score == 1.0, output='', stderr='', exit_code=int(score != 1.0),
        execution_time_ms=1.0, sandbox_id='local',
    )  # fmt: skip


# Scores 1.0 for "ok", 0.0 otherwise, and crashes (no verdict) on "CRASH".
_FLAKY_VERIFIER = Environment(
    task='t',
    verifier_code=(
        'content = open(solution_path).read()\n'
        'if "CRASH" in content:\n'
        '    raise RuntimeError("verifier bug")\n'
        'return 1.0 if "ok" in content else 0.0\n'
    ),
)


@pytest.mark.parametrize('client', ['sync', 'async'])
def test_batch_totals_count_no_verdict_as_zero_and_judged_apart(client: str) -> None:
    outputs = ['ok = 1\n', 'nope = 1\n', 'CRASH = 1\n']
    batch = (
        DeepGym(mode='local').run_batch(_FLAKY_VERIFIER, outputs)
        if client == 'sync'
        else asyncio.run(AsyncDeepGym(mode='local').run_batch(_FLAKY_VERIFIER, outputs))
    )
    assert (batch.total, batch.passed, batch.failed) == (3, 1, 2)
    assert batch.avg_score == pytest.approx(1 / 3)
    assert (batch.no_verdict_count, batch.judged_count) == (1, 2)
    assert (batch.judged_pass_rate, batch.judged_mean_score) == (0.5, 0.5)


class TestAsyncParallelism:
    """A concurrency limit below 1 is refused instead of hanging forever."""

    @pytest.mark.parametrize('call', ['run_batch', 'eval'])
    def test_zero_is_rejected(
        self, sorting_env: Environment, monkeypatch: pytest.MonkeyPatch, call: str
    ) -> None:
        monkeypatch.setattr('deepgym.registry.load_suite', lambda name: [sorting_env])
        client = AsyncDeepGym(mode='local')
        work = (
            client.run_batch(sorting_env, ['x'], max_parallel=0)
            if call == 'run_batch'
            else client.eval('suite', {'python_sorting': 'x'}, max_parallel=0)
        )
        with pytest.raises(ValueError, match='max_parallel'):
            asyncio.run(asyncio.wait_for(work, timeout=5))
