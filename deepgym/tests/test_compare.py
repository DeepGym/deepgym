"""Tests for comparing two runs task by task: flips, rates, no verdicts, changes, the test."""

from __future__ import annotations

import itertools
import json
import math
import random
import time
from datetime import datetime, timezone
from fractions import Fraction

import pytest

from deepgym.compare import Attempt, attempts_from_traces, compare
from deepgym.models import MultiTurnEnvironment, RunResult
from deepgym.passk import pass_at_k, pass_hat_k
from deepgym.traces import AgentMeta, EnvFingerprint, Provenance, Trace


def _attempts(key: str, solved: int, failed: int = 0, **fields: object) -> list[Attempt]:
    """*solved* passing attempts and *failed* zero-score ones at *key*."""
    return [
        Attempt(key=key, score=float(won), has_verdict=True, solved=won, **fields)
        for won in [True] * solved + [False] * failed
    ]


def _no_verdict(key: str, error_type: str | None = 'timeout', **fields: object) -> Attempt:
    return Attempt(key=key, score=0.0, has_verdict=False, solved=False, error_type=error_type,
                   **fields)  # fmt: skip


def _sides(counts: list[tuple[int, int, int, int]]) -> tuple[list[Attempt], list[Attempt]]:
    """Task i has ``(n_a, c_a, n_b, c_b)``: attempts and solves on A, then on B."""
    a, b = [], []
    for i, (n_a, c_a, n_b, c_b) in enumerate(counts):
        a += _attempts(f't{i}', c_a, n_a - c_a)
        b += _attempts(f't{i}', c_b, n_b - c_b)
    return a, b


def _single(outcomes: list[tuple[bool, bool]]) -> tuple[list[Attempt], list[Attempt]]:
    """One attempt per task and side, task i solved on A and B as *outcomes[i]* says."""
    return _sides([(1, int(won_a), 1, int(won_b)) for won_a, won_b in outcomes])


def _mcnemar(improved: int, regressed: int) -> float:
    """Exact McNemar: two-sided binomial test on the discordant pairs."""
    n = improved + regressed
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(improved, regressed) + 1)) / 2**n)


def test_single_attempt_flips_of_each_clean_class() -> None:
    a, b = _single([(False, True), (True, False), (True, True), (False, False)])

    report = compare(a, b)

    assert [(row.key, row.flip, row.delta) for row in report.tasks] == [
        ('t0', 'improved', 1.0), ('t1', 'regressed', -1.0),
        ('t2', 'both_solved', 0.0), ('t3', 'both_failed', 0.0),
    ]  # fmt: skip
    assert report.flips == {'improved': 1, 'regressed': 1, 'both_solved': 1, 'both_failed': 1,
                            'mixed': 0}  # fmt: skip
    assert report.compared == 4
    assert report.significance.test == 'exact_stratified_permutation'
    assert (report.significance.b_better, report.significance.a_better) == (1, 1)
    assert report.significance.ties == 2


def test_multi_attempt_rates_and_mixed() -> None:
    a = _attempts('clean', 0, 3) + _attempts('partial', 1, 3) + _attempts('flat', 2, 2)
    b = _attempts('clean', 2) + _attempts('partial', 3, 1) + _attempts('flat', 1, 1)

    rows = {row.key: row for row in compare(a, b).tasks}

    assert rows['clean'].flip == 'improved'  # 0/3 -> 2/2 is a clean flip
    assert (rows['partial'].flip, rows['partial'].delta) == ('mixed', pytest.approx(0.5))
    assert (rows['partial'].a.solve_rate, rows['partial'].b.solve_rate) == (0.25, 0.75)
    assert (rows['partial'].a.attempts, rows['partial'].a.solved) == (4, 1)
    assert (rows['flat'].flip, rows['flat'].delta) == ('mixed', 0.0)  # rates equal, not a flip


def test_mean_score_delta_uses_judged_attempts() -> None:
    a = [Attempt(key='t', score=score, has_verdict=True, solved=False) for score in (0.2, 0.4)]
    b = [Attempt(key='t', score=0.9, has_verdict=True, solved=True), _no_verdict('t')]

    (row,) = compare(a, b).tasks

    assert (row.a.mean_score, row.b.mean_score) == (pytest.approx(0.3), 0.9)
    assert row.score_delta == pytest.approx(0.6)
    assert row.flip == 'improved'  # solved is the verifier's pass, not a 1.0 score


def test_tasks_on_one_side_are_listed_not_compared() -> None:
    a = _attempts('shared', 1) + _attempts('dropped', 0, 1)
    b = _attempts('shared', 1) + _attempts('new', 1)

    report = compare(a, b)

    assert (report.only_a, report.only_b, report.compared) == (('dropped',), ('new',), 1)
    rows = {row.key: row for row in report.tasks}
    assert rows['dropped'].status == 'only_a' and rows['dropped'].b is None
    assert rows['new'].status == 'only_b' and rows['new'].a is None
    assert rows['new'].flip is None
    assert report.a.passk.tasks == report.b.passk.tasks == 1
    assert (report.a.attempts, report.b.attempts) == (2, 2)


def test_no_verdict_is_left_out_of_every_rate_and_counted_by_error_type() -> None:
    a = _attempts('t', 1) + [_no_verdict('t'), _no_verdict('t', 'provider_error')]
    b = _attempts('t', 1) + [_no_verdict('t', None), _no_verdict('other')]

    report = compare(a, b)

    row = report.tasks[1]
    assert (row.key, row.a.attempts, row.a.no_verdict, row.a.solve_rate) == ('t', 1, 2, 1.0)
    assert row.flip == 'both_solved'  # not dragged down by the attempts without a verdict
    assert report.a.no_verdict == {'provider_error': 1, 'timeout': 1}
    assert report.b.no_verdict == {'timeout': 1, 'verifier_error': 1}  # on every task
    assert report.a.passk.pass_at_k == {1: 1.0}


def test_task_without_any_verdict_on_one_side_is_uncomparable() -> None:
    a = [_no_verdict('t'), _no_verdict('t')] + _attempts('ok', 1)
    b = _attempts('t', 1) + _attempts('ok', 1)

    report = compare(a, b)

    assert report.uncomparable == ('t',)
    rows = {row.key: row for row in report.tasks}
    assert (rows['t'].status, rows['t'].flip, rows['t'].delta) == ('uncomparable', None, None)
    assert rows['t'].a.solve_rate is None
    assert report.flips['improved'] == 0  # not read as A failing it
    assert report.a.passk.bands == {'never_solved': 0, 'learnable': 0, 'always_solved': 1}


def test_nothing_compared_has_no_rates_or_test() -> None:
    report = compare([_no_verdict('t')], _attempts('t', 1))

    assert report.compared == 0
    assert report.a.passk is None and report.b.passk is None
    assert (report.k_limit, report.k_limit_tasks) == (None, ())
    assert report.significance is None


@pytest.mark.parametrize(
    ('a_hashes', 'b_hashes', 'status', 'unverified'),
    [
        (['x'], ['y'], 'changed', False),  # the environment changed between the runs
        (['x', 'y'], ['x', 'x'], 'ambiguous', False),  # it changed during run A
        (['x', 'x'], ['y', 'z'], 'ambiguous', False),
        (['x'], ['x'], 'compared', False),
        (['x', None], ['x'], 'compared', True),  # an unknown hash is no change, but unverified
        ([None], ['x'], 'compared', True),
        ([None], [None], 'compared', True),
    ],
)
def test_environment_hashes_decide_whether_a_task_is_compared(
    a_hashes: list[str | None], b_hashes: list[str | None], status: str, unverified: bool
) -> None:
    a = [att for fp in a_hashes for att in _attempts('t', 1, fingerprint=fp)]
    b = [att for fp in b_hashes for att in _attempts('t', 0, 1, fingerprint=fp)]

    report = compare(a, b)

    assert report.tasks[0].status == status
    assert report.changed == (('t',) if status == 'changed' else ())
    assert report.ambiguous == (('t',) if status == 'ambiguous' else ())
    assert report.unverified == (('t',) if unverified else ())
    assert report.flips['regressed'] == (status == 'compared')


@pytest.mark.parametrize('ties', [1, 3])
def test_single_attempt_test_is_exact_mcnemar(ties: int) -> None:
    for improved, regressed in itertools.product(range(9), repeat=2):
        outcomes = [(False, True)] * improved + [(True, False)] * regressed
        outcomes += [(True, True), (False, False)] * ties

        significance = compare(*_single(outcomes)).significance

        expected = _mcnemar(improved, regressed)
        assert significance.p_value == pytest.approx(expected, rel=1e-12), (improved, regressed)
        assert (significance.b_better, significance.a_better) == (improved, regressed)


@pytest.mark.parametrize(
    ('improved', 'regressed', 'p_value'),
    [(0, 5, 2 / 32), (1, 9, 2 * 11 / 1024), (6, 0, 2 / 64), (3, 3, 1.0), (0, 0, 1.0)],
)
def test_known_mcnemar_values(improved: int, regressed: int, p_value: float) -> None:
    outcomes = [(False, True)] * improved + [(True, False)] * regressed + [(True, True)]

    assert compare(*_single(outcomes)).significance.p_value == pytest.approx(p_value, rel=1e-12)


def test_test_matches_mcnemar_for_many_tasks() -> None:
    outcomes = [(False, True)] * 260 + [(True, False)] * 340

    p_value = compare(*_single(outcomes)).significance.p_value

    assert p_value == pytest.approx(_mcnemar(260, 340), rel=1e-10)
    assert 0.001 < p_value < 0.01


def test_a_vanishing_p_value_reads_as_tiny_or_zero() -> None:
    p_value = compare(*_single([(False, True)] * 1200)).significance.p_value

    assert 0.0 <= p_value < 1e-300  # 2 / 2^1200 is below what a float holds


def test_one_multi_attempt_task_by_hand() -> None:
    # 8 attempts, 4 solved, 4 on B: B solved 3. P(X >= 3) = (C(4,3)C(4,1) + 1) / C(8,4).
    report = compare(*_sides([(4, 1, 4, 3)]))

    assert report.significance.p_value == pytest.approx(2 * 17 / 70, rel=1e-12)


@pytest.mark.parametrize(
    'counts',
    [
        [(2, 1, 3, 2), (1, 0, 4, 3), (3, 3, 1, 0)],
        [(1, 1, 4, 1), (1, 0, 4, 4), (2, 0, 2, 2), (4, 4, 4, 4)],
        [(3, 1, 2, 2), (2, 2, 3, 0)],
    ],
)
def test_multi_attempt_test_matches_every_permutation(
    counts: list[tuple[int, int, int, int]],
) -> None:
    # Within each task, every way to deal its pooled attempts between A and B is equally
    # likely if the models are the same; tally B's total solves over all of them.
    per_task = []
    for n_a, c_a, n_b, c_b in counts:
        pool = [1] * (c_a + c_b) + [0] * (n_a + n_b - c_a - c_b)
        per_task.append([sum(deal) for deal in itertools.combinations(pool, n_b)])
    totals = [sum(choice) for choice in itertools.product(*per_task)]
    observed = sum(c_b for *_, c_b in counts)
    lower = Fraction(sum(t <= observed for t in totals), len(totals))
    upper = Fraction(sum(t >= observed for t in totals), len(totals))

    p_value = compare(*_sides(counts)).significance.p_value

    assert p_value == pytest.approx(float(min(1, 2 * min(lower, upper))), rel=1e-12)


def test_identical_models_with_unequal_attempts_are_not_flagged() -> None:
    # A samples each task once and B four times, from the same solve probabilities. A
    # sign test on rate changes rejects most of these; a valid test rejects at most ~5%.
    rng = random.Random(7)
    runs, rejected = 300, 0
    for _ in range(runs):
        chances = [rng.uniform(0.05, 0.4) for _ in range(40)]
        counts = [
            (1, int(rng.random() < p), 4, sum(rng.random() < p for _ in range(4))) for p in chances
        ]
        rejected += compare(*_sides(counts)).significance.p_value < 0.05

    assert rejected / runs <= 0.06


def test_pass_at_k_sized_suite_runs_in_seconds() -> None:
    # HumanEval-sized: 164 tasks sampled 100 times a side, solve rates spread over [0, 1].
    rng = random.Random(3)
    counts = []
    for _ in range(164):
        rate = rng.betavariate(0.6, 0.6)
        counts.append((100, sum(rng.random() < rate for _ in range(100)),
                       100, sum(rng.random() < rate for _ in range(100))))  # fmt: skip
    a, b = _sides(counts)

    started = time.perf_counter()
    significance = compare(a, b).significance
    elapsed = time.perf_counter() - started

    assert 0.0 <= significance.p_value <= 1.0
    assert elapsed < 5.0, f'{elapsed:.1f} s'


def test_a_real_difference_is_found_with_unequal_attempts() -> None:
    rng = random.Random(11)
    counts = [
        (1, int(rng.random() < 0.2), 4, sum(rng.random() < 0.6 for _ in range(4)))
        for _ in range(40)
    ]

    assert compare(*_sides(counts)).significance.p_value < 0.001


def test_side_pass_at_k_over_compared_tasks_with_common_ks() -> None:
    a = _attempts('x', 2, 2) + _attempts('y', 0, 4) + _attempts('z', 4, 4)
    b = _attempts('x', 1, 1) + _attempts('y', 2, 1) + _attempts('z', 4)

    report = compare(a, b)

    # B has only 2 judged attempts at x, so k is 1 and 2 on both sides.
    assert (report.k_limit, report.k_limit_tasks) == (2, ('x',))
    assert report.a.passk.pass_at_k == pytest.approx(
        {1: (0.5 + 0.0 + 0.5) / 3, 2: (pass_at_k(4, 2, 2) + 0.0 + pass_at_k(8, 4, 2)) / 3}
    )
    assert report.a.passk.pass_hat_k[2] == pytest.approx(
        (pass_hat_k(4, 2, 2) + pass_hat_k(8, 4, 2)) / 3
    )
    assert report.b.passk.pass_at_k[1] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert report.a.passk.bands == {'never_solved': 1, 'learnable': 2, 'always_solved': 0}
    assert report.b.passk.bands == {'never_solved': 0, 'learnable': 2, 'always_solved': 1}


def test_attempt_solved_without_a_verdict_is_refused() -> None:
    with pytest.raises(ValueError, match='without a verdict cannot be solved'):
        Attempt(key='t', score=1.0, has_verdict=False, solved=True)


def test_report_round_trips_as_json() -> None:
    report = compare(*_single([(False, True), (True, True)]))

    assert type(report).model_validate_json(report.model_dump_json()) == report


ENV = MultiTurnEnvironment(name='todo_list', task='Add a todo.', final_verifier_code='print(1)')
STARTED = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _trace(trace_id: str, result: RunResult | None, env: MultiTurnEnvironment = ENV,
           **fingerprint: int) -> Trace:  # fmt: skip
    provenance = Provenance(
        env=EnvFingerprint.of(env, **fingerprint),
        agent=AgentMeta(name='react-shell'),
        provider='local',
        deepgym_version='0.6.6',
        started_at=STARTED,
    )
    return Trace(trace_id=trace_id, task=env.task, provenance=provenance, result=result)


def _result(score: float, **fields: object) -> RunResult:
    base = {'score': score, 'passed': score == 1.0, 'output': '', 'stderr': '', 'exit_code': 0,
            'execution_time_ms': 1.0, 'sandbox_id': 'local'}  # fmt: skip
    return RunResult.model_validate(base | fields)


def test_trace_adapter_keys_results_and_verdicts() -> None:
    unnamed = ENV.model_copy(update={'name': None})
    traces = [
        _trace('solved', _result(1.0), variant=3, seed=7),
        _trace('gated', _result(0.6, passed=True), variant=3, seed=8),  # a rubric pass
        _trace('unpassed', _result(1.0, passed=False)),
        _trace('crashed', _result(0.0, exit_code=2)),
        _trace('unscored', None),
        _trace('timed_out', _result(0.0, error_type='solution_timeout')),  # the model's fault
        _trace('anonymous', _result(1.0), unnamed),
    ]

    attempts = attempts_from_traces(traces)

    fields = [(a.key, a.score, a.has_verdict, a.solved, a.error_type) for a in attempts]
    task_sha = EnvFingerprint.of(unnamed).task_sha256
    assert fields == [
        ('["todo_list", 3]', 1.0, True, True, None),
        ('["todo_list", 3]', 0.6, True, True, None),  # seeds don't split a task
        ('["todo_list", 0]', 1.0, True, False, None),  # no variant recorded: variant 0
        ('["todo_list", 0]', 0.0, False, False, None),
        ('["todo_list", 0]', 0.0, False, False, 'incomplete'),
        ('["todo_list", 0]', 0.0, True, False, 'solution_timeout'),
        (json.dumps([None, 0, task_sha]), 1.0, True, True, None),
    ]
    assert attempts[0].fingerprint == EnvFingerprint.of(ENV).environment_sha256


def test_trace_keys_cannot_collide() -> None:
    spaced = ENV.model_copy(update={'name': 'todo variant=1'})
    traces = [_trace('plain', _result(1.0), variant=1), _trace('spaced', _result(1.0), spaced)]

    keys = {attempt.key for attempt in attempts_from_traces(traces)}

    assert keys == {'["todo_list", 1]', '["todo variant=1", 0]'}


def test_trace_adapter_flags_an_edited_task() -> None:
    edited = ENV.model_copy(update={'task': 'Add two todos.'})
    a = attempts_from_traces([_trace('a1', _result(1.0))])
    b = attempts_from_traces([_trace('b1', _result(1.0), edited)])

    assert compare(a, b).changed == ('["todo_list", 0]',)


def test_unrecorded_variant_pairs_with_variant_0() -> None:
    a = attempts_from_traces([_trace('a1', _result(0.0))])  # dg.trace without variant=
    b = attempts_from_traces([_trace('b1', _result(1.0), variant=0)])

    report = compare(a, b)

    assert (report.compared, report.flips['improved']) == (1, 1)


def test_trace_adapter_refuses_a_trace_twice() -> None:
    with pytest.raises(ValueError, match='share ids'):
        attempts_from_traces([_trace('same', _result(1.0)), _trace('SAME', _result(0.0))])
