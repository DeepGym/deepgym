"""Compare two runs on the same tasks: model A against model B, or one checkpoint against another.

Each side is a list of :class:`Attempt`, one per graded sample of a task. :func:`compare`
pairs tasks by key and reports, per task, each side's solve rate and the change; which
tasks flipped from failed to solved (improved) or back (regressed); and overall each
side's mean pass@k and pass^k with an exact paired test of whether B solves more or less
than A.

An attempt without a verdict is counted per side by ``error_type`` and left out of every
rate, never counted as a fail, as in the reward contract. A task whose attempts on one side
all lack a verdict is uncomparable; one whose environment hash differs between the sides is
changed, and one with several hashes on a side is ambiguous. None of them enters a flip
count, a pass rate or the test.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepgym.passk import PassKSummary, default_ks, summarize
from deepgym.traces.schema import EnvFingerprint, Trace, check_unique_ids

Flip = Literal['improved', 'regressed', 'both_solved', 'both_failed', 'mixed']
"""How a compared task moved from A to B: every attempt failed on A and solved on B
(improved), the reverse (regressed), solved or failed on both, or anything else (mixed,
when a side solved only some of its attempts)."""

TaskStatus = Literal['compared', 'only_a', 'only_b', 'uncomparable', 'changed', 'ambiguous']

PairedTest = Literal['exact_stratified_permutation']


class Attempt(BaseModel):
    """One graded sample of one task."""

    model_config = ConfigDict(frozen=True)

    key: str
    """Which task; attempts with the same key on both sides are paired."""

    score: float = Field(ge=0, le=1)
    has_verdict: bool
    """False when the verifier or sandbox failed instead of judging the attempt."""

    solved: bool
    """The verifier passed it (``passed``); only an attempt with a verdict can be solved."""

    error_type: str | None = None
    """Why there is no verdict; one without an error type counts as ``verifier_error``,
    as the console stores it."""

    fingerprint: str | None = None
    """Hash of the environment definition the attempt ran against; None when unknown."""

    @model_validator(mode='after')
    def check_solved(self) -> Attempt:
        """Reject an attempt solved without a verdict."""
        if self.solved and not self.has_verdict:
            raise ValueError('an attempt without a verdict cannot be solved')
        return self


class TaskSide(BaseModel):
    """One side's attempts at one task."""

    model_config = ConfigDict(frozen=True)

    attempts: int
    """Attempts with a verdict."""

    solved: int
    no_verdict: int
    solve_rate: float | None
    """``solved / attempts``, which is also the task's unbiased pass@1; None without a
    judged attempt."""

    mean_score: float | None
    """Mean score of the judged attempts; None without one."""


class TaskComparison(BaseModel):
    """One task on both sides. Flip and deltas are None unless the task was compared."""

    model_config = ConfigDict(frozen=True)

    key: str
    status: TaskStatus
    a: TaskSide | None
    """None when the task isn't on side A."""

    b: TaskSide | None
    flip: Flip | None = None
    delta: float | None = None
    """B's solve rate minus A's."""

    score_delta: float | None = None
    """B's mean score minus A's."""


class SideSummary(BaseModel):
    """One side overall."""

    model_config = ConfigDict(frozen=True)

    attempts: int
    """Every attempt on this side, with a verdict or not, on every task."""

    no_verdict: dict[str, int]
    """Attempts without a verdict per error type, on every task."""

    passk: PassKSummary | None
    """Mean pass@k and pass^k and band counts over the compared tasks, for k = 1, each
    power of 2 below the report's ``k_limit``, and ``k_limit``; None without a compared
    task."""


class Significance(BaseModel):
    """Exact two-sided test of whether B solves the compared tasks more or less than A.

    A stratified permutation test (the exact conditional test behind Cochran-Mantel-
    Haenszel): if A and B are the same model, a task's judged attempts are exchangeable, so
    given how many it solved across both sides, B's share is hypergeometric. The statistic
    is B's total solves; its null distribution is those tasks' distributions convolved, and
    the p-value is twice the smaller tail, at most 1. With one judged attempt per task and
    side it is exact McNemar. Unequal attempt counts don't bias it. The convolution runs
    in floating point and drops probabilities below 1e-300 from its tails, so the p-value
    is accurate to float precision down to about 1e-290, drifts below that, and reads as 0
    only once it is under about 1e-300.
    """

    model_config = ConfigDict(frozen=True)

    test: PairedTest
    b_better: int
    """Compared tasks where B's solve rate is higher than A's."""

    a_better: int
    ties: int
    p_value: float


class CompareReport(BaseModel):
    """Side B against side A, task by task and overall."""

    model_config = ConfigDict(frozen=True)

    tasks: tuple[TaskComparison, ...]
    """Every task on either side, by key."""

    compared: int
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]
    uncomparable: tuple[str, ...]
    """On both sides, but every attempt on one side (or both) lacks a verdict."""

    changed: tuple[str, ...]
    """On both sides, but A's environment hash differs from B's."""

    ambiguous: tuple[str, ...]
    """On both sides, but one side's attempts ran against more than one environment hash."""

    unverified: tuple[str, ...]
    """Compared tasks where some attempt had no environment hash, so a change can't be
    ruled out."""

    k_limit: int | None
    """The fewest judged attempts any compared task has on either side: the largest k in
    each side's pass@k. None without a compared task."""

    k_limit_tasks: tuple[str, ...]
    """The compared tasks with only ``k_limit`` judged attempts on a side."""

    flips: dict[Flip, int]
    a: SideSummary
    b: SideSummary
    significance: Significance | None
    """None without a compared task."""


def compare(a: Sequence[Attempt], b: Sequence[Attempt]) -> CompareReport:
    """Compare side *b* against side *a*, task by task.

    Args:
        a: The baseline's attempts (model A, or the earlier checkpoint).
        b: The attempts to compare against it.

    Returns:
        The report. Every task on either side is in it; only compared tasks enter the
        flip counts, the pass rates and the significance test.

    Example:
        >>> base = [Attempt(key='t1', score=0.0, has_verdict=True, solved=False)]
        >>> new = [Attempt(key='t1', score=1.0, has_verdict=True, solved=True)]
        >>> compare(base, new).flips['improved']
        1
    """
    by_a, by_b = _by_key(a), _by_key(b)
    rows: list[TaskComparison] = []
    pairs: dict[str, tuple[TaskSide, TaskSide]] = {}
    for key in sorted(by_a.keys() | by_b.keys()):
        row, pair = _task(key, by_a.get(key, []), by_b.get(key, []))
        rows.append(row)
        if pair is not None:
            pairs[key] = pair
    fewest = {key: min(side_a.attempts, side_b.attempts) for key, (side_a, side_b) in pairs.items()}
    k_limit = min(fewest.values(), default=None)
    ks = [] if k_limit is None else default_ks(k_limit)
    flips = Counter(row.flip for row in rows)

    def keys(status: TaskStatus) -> tuple[str, ...]:
        return tuple(row.key for row in rows if row.status == status)

    return CompareReport(
        tasks=tuple(rows),
        compared=len(pairs),
        only_a=keys('only_a'),
        only_b=keys('only_b'),
        uncomparable=keys('uncomparable'),
        changed=keys('changed'),
        ambiguous=keys('ambiguous'),
        unverified=tuple(
            key
            for key in pairs
            if any(attempt.fingerprint is None for attempt in (*by_a[key], *by_b[key]))
        ),
        k_limit=k_limit,
        k_limit_tasks=tuple(key for key, size in fewest.items() if size == k_limit),
        flips={flip: flips[flip] for flip in get_args(Flip)},
        a=_summary(a, [side_a for side_a, _ in pairs.values()], ks),
        b=_summary(b, [side_b for _, side_b in pairs.values()], ks),
        significance=_significance(list(pairs.values())) if pairs else None,
    )


def attempts_from_traces(traces: Sequence[Trace]) -> list[Attempt]:
    """Return one attempt per trace, for :func:`compare`.

    The key is ``json.dumps([name, variant])`` of the environment, e.g. ``["todo_list", 3]``,
    with variant 0 when none was recorded (``load_environment``'s default); an unnamed one
    adds its task hash, ``[null, variant, "<task sha256>"]``. A trace that ran another
    variant than the one recorded is still caught by the environment hash. The seed is not
    part of the key: deepgym only records the seed, so
    traces that differ only in seed are attempts at one task. An attempt is solved when the
    verifier passed it with a verdict. The fingerprint is the environment hash, which
    covers the task and the verifiers. A trace never scored has no verdict, under
    ``incomplete``.

    Raises:
        ValueError: If two traces share an id: one episode would count twice.
    """
    check_unique_ids(traces)
    return [_trace_attempt(trace) for trace in traces]


def _trace_attempt(trace: Trace) -> Attempt:
    env = trace.provenance.env
    result = trace.result
    return Attempt(
        key=_trace_key(env),
        score=0.0 if result is None else result.score,
        has_verdict=trace.has_verdict,
        solved=result is not None and trace.has_verdict and result.passed,
        error_type='incomplete' if result is None else result.error_type,
        fingerprint=env.environment_sha256,
    )


def _trace_key(env: EnvFingerprint) -> str:
    variant = 0 if env.variant is None else env.variant
    parts = [env.name, variant] if env.name else [None, variant, env.task_sha256]
    return json.dumps(parts)


def _by_key(attempts: Iterable[Attempt]) -> dict[str, list[Attempt]]:
    grouped: defaultdict[str, list[Attempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.key].append(attempt)
    return grouped


def _side(attempts: Sequence[Attempt]) -> TaskSide:
    judged = [attempt for attempt in attempts if attempt.has_verdict]
    solved = sum(attempt.solved for attempt in judged)
    return TaskSide(
        attempts=len(judged),
        solved=solved,
        no_verdict=len(attempts) - len(judged),
        solve_rate=solved / len(judged) if judged else None,
        mean_score=statistics.fmean(attempt.score for attempt in judged) if judged else None,
    )


_FLIPS: dict[tuple[float, float], Flip] = {
    (0.0, 1.0): 'improved',
    (1.0, 0.0): 'regressed',
    (1.0, 1.0): 'both_solved',
    (0.0, 0.0): 'both_failed',
}
"""Solve rates of A and B that make a clean flip; any other pair is mixed."""


def _task(
    key: str, a: Sequence[Attempt], b: Sequence[Attempt]
) -> tuple[TaskComparison, tuple[TaskSide, TaskSide] | None]:
    """Return the task's row, and both its sides when it is compared."""
    side_a, side_b = (_side(attempts) if attempts else None for attempts in (a, b))
    if side_a is None or side_b is None:
        status: TaskStatus = 'only_b' if side_a is None else 'only_a'
        return TaskComparison(key=key, status=status, a=side_a, b=side_b), None
    status = _shared_status(a, b, side_a, side_b)
    if status != 'compared':
        return TaskComparison(key=key, status=status, a=side_a, b=side_b), None
    rate_a, rate_b = side_a.solved / side_a.attempts, side_b.solved / side_b.attempts
    row = TaskComparison(
        key=key,
        status=status,
        a=side_a,
        b=side_b,
        flip=_FLIPS.get((rate_a, rate_b), 'mixed'),
        delta=rate_b - rate_a,
        score_delta=_minus(side_b.mean_score, side_a.mean_score),
    )
    return row, (side_a, side_b)


def _shared_status(
    a: Sequence[Attempt], b: Sequence[Attempt], side_a: TaskSide, side_b: TaskSide
) -> TaskStatus:
    """Status of a task on both sides; an unknown hash (None) is never a change."""
    hashes_a, hashes_b = ({attempt.fingerprint for attempt in each} - {None} for each in (a, b))
    if len(hashes_a) > 1 or len(hashes_b) > 1:
        return 'ambiguous'
    if hashes_a and hashes_b and hashes_a != hashes_b:
        return 'changed'
    if not (side_a.attempts and side_b.attempts):
        return 'uncomparable'
    return 'compared'


def _minus(x: float | None, y: float | None) -> float | None:
    return None if x is None or y is None else x - y


def _summary(
    attempts: Sequence[Attempt], compared: Sequence[TaskSide], ks: list[int]
) -> SideSummary:
    errors = Counter(
        attempt.error_type or 'verifier_error' for attempt in attempts if not attempt.has_verdict
    )
    passk = summarize([(side.attempts, side.solved) for side in compared], ks) if compared else None
    return SideSummary(attempts=len(attempts), no_verdict=dict(sorted(errors.items())), passk=passk)


def _significance(pairs: Sequence[tuple[TaskSide, TaskSide]]) -> Significance:
    lowest, null = 0, [1.0]  # P(B's total solves = lowest + i) is null[i]
    for a, b in pairs:
        low, task = _null_pmf(a, b)
        lowest += low
        if len(task) > 1:  # one possible count shifts the total without spreading it
            null, dropped = _trim(_convolve(null, task))
            lowest += dropped
    observed = sum(b.solved for _, b in pairs) - lowest
    tail = (
        min(math.fsum(null[: observed + 1]), math.fsum(null[observed:]))
        if 0 <= observed < len(null)
        else 0.0  # beyond the dropped tails: under about 1e-300
    )
    # Exact sign of each task's b.rate - a.rate, from integer counts.
    signs = Counter(
        (b.solved * a.attempts > a.solved * b.attempts)
        - (b.solved * a.attempts < a.solved * b.attempts)
        for a, b in pairs
    )
    return Significance(
        test='exact_stratified_permutation',
        b_better=signs[1],
        a_better=signs[-1],
        ties=signs[0],
        p_value=min(1.0, 2 * tail),
    )


def _null_pmf(a: TaskSide, b: TaskSide) -> tuple[int, list[float]]:
    """B's solves at one task if A and B are the same model: the fewest possible, and the
    hypergeometric probability ``C(K, x) * C(N - K, n_b - x) / C(N, n_b)`` of each count x
    from there, for N judged attempts on both sides with K solved and n_b on B."""
    total, solved = a.attempts + b.attempts, a.solved + b.solved
    low, high = max(0, b.attempts - (total - solved)), min(b.attempts, solved)
    ways = math.comb(total, b.attempts)
    pmf = [
        math.comb(solved, x) * math.comb(total - solved, b.attempts - x) / ways
        for x in range(low, high + 1)
    ]
    return low, pmf


_NEGLIGIBLE = 1e-300
"""Probabilities this small are dropped from the ends of the null distribution."""


def _trim(pmf: list[float]) -> tuple[list[float], int]:
    """Drop negligible probabilities from both ends of *pmf*; return it and how many led."""
    first = next(i for i, p in enumerate(pmf) if p > _NEGLIGIBLE)
    last = len(pmf) - next(i for i, p in enumerate(reversed(pmf)) if p > _NEGLIGIBLE)
    return pmf[first:last], first


def _convolve(x: list[float], y: list[float]) -> list[float]:
    """Convolve two probability lists, looping over the shorter one."""
    # ponytail: quadratic in the support, ~0.8 s for 164 tasks of 100 attempts a side all
    # near a 50% solve rate (0.25 s for a realistic spread); FFT if suites grow far past it.
    if len(y) > len(x):
        x, y = y, x
    out = [0.0] * (len(x) + len(y) - 1)
    for j, yj in enumerate(y):
        out[j : j + len(x)] = [o + xi * yj for o, xi in zip(out[j : j + len(x)], x)]
    return out
