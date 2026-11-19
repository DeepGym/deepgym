"""Score analytics for a model sampled several times per task: pass@k, pass^k, difficulty.

A task sampled ``n`` times with ``c`` solved samples gives unbiased estimates of
pass@k (at least one of k attempts solves it) and pass^k (all k attempts solve it,
a reliability measure), and a difficulty band: group-relative RL learns nothing from
a task the model always or never solves.
"""

from __future__ import annotations

import math
import numbers
import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

Band = Literal['never_solved', 'learnable', 'always_solved']
"""Difficulty of one task from its solve rate: none, some or all samples solved."""


class TaskDifficulty(BaseModel):
    """Difficulty of one task sampled ``n`` times."""

    model_config = ConfigDict(frozen=True)

    band: Band
    solve_rate: float
    outside_band: bool
    """True when the solve rate is below ``low`` or above ``high``."""


class PassKSummary(BaseModel):
    """Mean pass@k and pass^k over tasks for each k, and how many tasks fall in each band."""

    model_config = ConfigDict(frozen=True)

    tasks: int
    pass_at_k: dict[int, float]
    pass_hat_k: dict[int, float]
    bands: dict[Band, int]
    outside_band: int


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate the chance that at least one of *k* attempts solves the task.

    The unbiased estimator ``1 - C(n - c, k) / C(n, k)`` (Chen et al. 2021, HumanEval),
    computed from a sum of k logarithms rather than from binomials, so a tiny result
    keeps its precision.

    Args:
        n: Samples drawn.
        c: Samples that solved the task.
        k: Attempts, 1 to n.

    Returns:
        pass@k in [0, 1]; 1.0 whenever fewer than k samples failed.

    Raises:
        ValueError: If an input isn't an int, or k or c is outside its range.

    Example:
        >>> round(pass_at_k(10, 3, 2), 4)
        0.5333
    """
    _validate(n, c, k)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    # 1 - prod(1 - c / (n - i)), kept exact when the result is tiny
    return -math.expm1(math.fsum(math.log1p(-c / (n - i)) for i in range(k)))


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Estimate the chance that all *k* attempts solve the task (pass^k).

    The unbiased estimator ``C(c, k) / C(n, k)`` (tau-bench, Yao et al. 2024), computed
    as a product of k ratios rather than from binomials.

    Args:
        n: Samples drawn.
        c: Samples that solved the task.
        k: Attempts, 1 to n.

    Returns:
        pass^k in [0, 1]; 0.0 whenever fewer than k samples solved it.

    Raises:
        ValueError: If an input isn't an int, or k or c is outside its range.

    Example:
        >>> round(pass_hat_k(10, 3, 2), 4)
        0.0667
    """
    _validate(n, c, k)
    return _choose_ratio(c, n, k)


def task_difficulty(n: int, c: int, low: float = 0.1, high: float = 0.9) -> TaskDifficulty:
    """Place one task in a band by its solve rate ``c / n``.

    ``never_solved`` when no sample solved it, ``always_solved`` when every sample did, else
    ``learnable``. ``outside_band`` also flags a learnable task solved too rarely or too
    often to train on, by the inclusive band ``[low, high]`` (10%-90% by default).

    Args:
        n: Samples drawn.
        c: Samples that solved the task.
        low: Lowest solve rate inside the band.
        high: Highest solve rate inside the band.

    Returns:
        The task's band, solve rate and whether that rate is outside ``[low, high]``.

    Raises:
        ValueError: If n or c isn't an int or is out of range, or not
            ``0 <= low <= high <= 1``.
    """
    _validate(n, c, 1)
    _validate_band(low, high)
    rate = c / n
    band: Band = 'never_solved' if c == 0 else 'always_solved' if c == n else 'learnable'
    return TaskDifficulty(band=band, solve_rate=rate, outside_band=not low <= rate <= high)


def default_ks(n: int) -> list[int]:
    """Return the k worth reporting for *n* samples: 1, each power of 2 below n, and n.

    Example:
        >>> default_ks(12)
        [1, 2, 4, 8, 12]
    """
    return sorted({2**power for power in range(n.bit_length())} | {n})


def summarize(
    tasks: Sequence[tuple[int, int]], ks: Sequence[int], low: float = 0.1, high: float = 0.9
) -> PassKSummary:
    """Average pass@k and pass^k over *tasks* for each k in *ks*, and count tasks per band.

    Args:
        tasks: ``(n, c)`` per task: samples drawn and samples that solved it.
        ks: Attempt counts to report; each must be at most every task's n.
        low: Lowest solve rate inside the band (see :func:`task_difficulty`).
        high: Highest solve rate inside the band.

    Returns:
        Mean pass@k and pass^k keyed by k, task counts per band (zeros included), and
        how many tasks fall outside ``[low, high]``.

    Raises:
        ValueError: If *tasks* or *ks* is empty, or any input is invalid as in
            :func:`pass_at_k` and :func:`task_difficulty`.
    """
    _validate_band(low, high)
    if not tasks:
        raise ValueError('summarize needs at least one task')
    if not ks:
        raise ValueError('summarize needs at least one k')
    for index, (n, c) in enumerate(tasks):
        for k in ks:
            try:
                _validate(n, c, k)
            except ValueError as exc:
                raise ValueError(f'task {index}: {exc}') from exc
    difficulties = [task_difficulty(n, c, low, high) for n, c in tasks]
    counts = Counter(difficulty.band for difficulty in difficulties)
    return PassKSummary(
        tasks=len(tasks),
        pass_at_k={k: statistics.fmean(pass_at_k(n, c, k) for n, c in tasks) for k in ks},
        pass_hat_k={k: statistics.fmean(pass_hat_k(n, c, k) for n, c in tasks) for k in ks},
        bands={band: counts[band] for band in get_args(Band)},
        outside_band=sum(difficulty.outside_band for difficulty in difficulties),
    )


def _choose_ratio(m: int, n: int, k: int) -> float:
    """``C(m, k) / C(n, k)`` for ``0 <= m <= n``, as a product of k ratios."""
    if m < k:
        return 0.0
    return math.prod((m - i) / (n - i) for i in range(k))


def _validate_band(low: float, high: float) -> None:
    """Raise ValueError unless low and high are real numbers with ``0 <= low <= high <= 1``."""
    for name, value in (('low', low), ('high', high)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f'{name} must be a number, got {value!r}')
    if not 0 <= low <= high <= 1:
        raise ValueError(f'need 0 <= low <= high <= 1, got low={low}, high={high}')


def _validate(n: int, c: int, k: int) -> None:
    """Raise ValueError unless n, c and k are ints with ``1 <= k <= n`` and ``0 <= c <= n``."""
    for name, value in (('n', n), ('c', c), ('k', k)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f'{name} must be an int, got {value!r}')
    if n < 1:
        raise ValueError(f'n must be at least 1, got {n}')
    if not 1 <= k <= n:
        raise ValueError(f'k must be between 1 and n={n}, got {k}')
    if not 0 <= c <= n:
        raise ValueError(f'c must be between 0 and n={n}, got {c}')
