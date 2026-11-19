"""Tests for pass@k, pass^k, the difficulty band and their aggregate over tasks."""

from __future__ import annotations

import itertools
import math

import pytest

from deepgym.passk import pass_at_k, pass_hat_k, summarize, task_difficulty


def test_known_values() -> None:
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_hat_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(10, 3, 2) == pytest.approx(1 - 21 / 45)  # 1 - C(7,2)/C(10,2)
    assert pass_hat_k(10, 3, 2) == pytest.approx(3 / 45)  # C(3,2)/C(10,2)


@pytest.mark.parametrize('n', [1, 2, 5, 16])
def test_limits(n: int) -> None:
    assert all(pass_at_k(n, c, n) == 1.0 for c in range(1, n + 1))
    assert pass_at_k(n, 0, n) == 0.0
    assert all(pass_hat_k(n, n, k) == 1.0 for k in range(1, n + 1))
    assert all(pass_hat_k(n, c, k) == 0.0 for k in range(1, n + 1) for c in range(k))


def test_estimators_match_counting_every_subset_of_k_samples() -> None:
    for n in range(1, 8):
        for c in range(n + 1):
            samples = [True] * c + [False] * (n - c)
            for k in range(1, n + 1):
                subsets = list(itertools.combinations(samples, k))
                any_solved = sum(any(subset) for subset in subsets) / len(subsets)
                all_solved = sum(all(subset) for subset in subsets) / len(subsets)
                assert pass_at_k(n, c, k) == pytest.approx(any_solved), (n, c, k)
                assert pass_hat_k(n, c, k) == pytest.approx(all_solved), (n, c, k)


def test_large_counts_match_exact_binomials() -> None:
    n, k = 10_000, 1_000  # C(n, k) has over 1,400 digits; int / int rounds exactly

    exact = math.comb(n - 10, k) / math.comb(n, k)

    assert 0.3 < exact < 0.4
    assert pass_at_k(n, 10, k) == pytest.approx(1 - exact, rel=1e-12, abs=0)
    assert pass_hat_k(n, n - 10, k) == pytest.approx(exact, rel=1e-12, abs=0)


@pytest.mark.parametrize(
    ('n', 'c', 'k', 'message'),
    [
        (10, 3, 0, 'k must be between 1 and n=10, got 0'),
        (10, 3, 11, 'k must be between 1 and n=10, got 11'),
        (10, -1, 1, 'c must be between 0 and n=10, got -1'),
        (10, 11, 1, 'c must be between 0 and n=10, got 11'),
        (0, 0, 1, 'n must be at least 1, got 0'),
        (10.0, 3, 1, 'n must be an int, got 10.0'),
        (10, 3.0, 1, 'c must be an int, got 3.0'),
        (10, 3, True, 'k must be an int, got True'),
        (10, '3', 1, "c must be an int, got '3'"),
    ],
)
def test_invalid_inputs(n: int, c: int, k: int, message: str) -> None:
    for estimator in (pass_at_k, pass_hat_k):
        with pytest.raises(ValueError, match=message):
            estimator(n, c, k)


@pytest.mark.parametrize(
    ('n', 'c', 'band', 'outside'),
    [
        (10, 0, 'never_solved', True),
        (10, 1, 'learnable', False),  # exactly 10%: inside
        (30, 3, 'learnable', False),  # 0.1 * 30 rounds above 3; c / n doesn't
        (20, 1, 'learnable', True),  # 5%
        (10, 9, 'learnable', False),  # exactly 90%: inside
        (30, 27, 'learnable', False),
        (20, 19, 'learnable', True),  # 95%
        (10, 10, 'always_solved', True),
    ],
)
def test_band_edges(n: int, c: int, band: str, outside: bool) -> None:
    difficulty = task_difficulty(n, c)

    assert (difficulty.band, difficulty.outside_band) == (band, outside)
    assert difficulty.solve_rate == c / n


def test_band_is_configurable_and_checked() -> None:
    assert task_difficulty(10, 2, low=0.25, high=0.75).outside_band
    assert not task_difficulty(10, 0, low=0.0, high=1.0).outside_band
    for low, high in ((0.9, 0.1), (-0.1, 0.9), (0.1, 1.1), (math.nan, 0.9)):
        with pytest.raises(ValueError, match='need 0 <= low <= high <= 1'):
            task_difficulty(10, 5, low=low, high=high)
    with pytest.raises(ValueError, match='c must be between 0 and n=10'):
        task_difficulty(10, 11)


def test_summarize() -> None:
    tasks = [(4, 0), (4, 2), (4, 4), (20, 1)]

    summary = summarize(tasks, [1, 2, 4])

    assert summary.tasks == 4
    assert summary.pass_at_k == pytest.approx(
        {1: (0 + 0.5 + 1 + 0.05) / 4, 2: (0 + 5 / 6 + 1 + 0.1) / 4, 4: (0 + 1 + 1 + 0.2) / 4}
    )
    assert summary.pass_hat_k == pytest.approx(
        {1: (0 + 0.5 + 1 + 0.05) / 4, 2: (0 + 1 / 6 + 1 + 0) / 4, 4: (0 + 0 + 1 + 0) / 4}
    )
    assert summary.bands == {'never_solved': 1, 'learnable': 2, 'always_solved': 1}
    assert summary.outside_band == 3  # all but (4, 2)
    assert summarize([(4, 2)], [1]).bands == {'never_solved': 0, 'learnable': 1, 'always_solved': 0}


def test_summarize_rejects_a_k_above_any_task_n_and_no_tasks() -> None:
    with pytest.raises(ValueError, match='k must be between 1 and n=4, got 8'):
        summarize([(8, 3), (4, 1)], [8])
    with pytest.raises(ValueError):
        summarize([], [1])


def test_a_tiny_pass_at_k_keeps_its_precision() -> None:
    assert pass_at_k(10**9, 1, 1) == pytest.approx(1e-9, rel=1e-12)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'tasks': [], 'ks': [1]}, 'at least one task'),
        ({'tasks': [(4, 2)], 'ks': []}, 'at least one k'),
        ({'tasks': [(4, 2), (2, 1)], 'ks': [4]}, r'task 1: k must be between 1 and n=2'),
        ({'tasks': [], 'ks': [], 'low': 0.9, 'high': 0.1}, 'low <= high'),
        ({'tasks': [(10, 5)], 'ks': [1], 'low': True}, 'low must be a number'),
        ({'tasks': [(10, 5)], 'ks': [1], 'high': 'a'}, 'high must be a number'),
    ],
)
def test_summarize_refuses_bad_input_up_front(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        summarize(**kwargs)
