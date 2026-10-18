"""Offline checks for shared episode accounting."""

import pytest
from pydantic import ValidationError

from deepgym.agents.base import Budget, BudgetExceeded, BudgetTracker, Cost, TokenPrices, Usage


@pytest.mark.parametrize(
    'field,value',
    [
        ('max_turns', 0),
        ('max_seconds', 0),
        ('max_seconds', float('nan')),
        ('max_cost_usd', -1),
        ('max_cost_usd', float('inf')),
    ],
)
def test_invalid_budget(field, value):
    with pytest.raises(ValidationError):
        Budget(**{field: value})


def test_turn_budget():
    tracker = BudgetTracker(Budget(max_turns=1))
    tracker.before_turn()
    with pytest.raises(BudgetExceeded, match='turn_budget'):
        tracker.before_turn()
    assert tracker.turns == 1


def test_time_counts_all_elapsed_time(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('deepgym.agents.base.time.monotonic', lambda: clock[0])
    tracker = BudgetTracker(Budget(max_seconds=5))
    clock[0] += 5
    with pytest.raises(BudgetExceeded, match='time_budget'):
        tracker.before_turn()


def test_disjoint_usage_and_price_provenance():
    prices = TokenPrices(
        input_per_million=2,
        output_per_million=10,
        cache_read_per_million=0.5,
        cache_write_per_million=2.5,
        source='test rate card 2026-10-18',
    )
    usage = Usage(
        input_tokens=1_000, output_tokens=100, cache_read_tokens=200, cache_write_tokens=400
    )
    tracker = BudgetTracker(Budget(max_cost_usd=0.006))
    tracker.add_usage(usage, prices.estimate(usage))
    tracker.add_usage(usage, prices.estimate(usage))
    assert tracker.usage.input_tokens == 2000
    assert tracker.cost.usd == pytest.approx(0.0082)
    assert 'test rate card 2026-10-18' in tracker.cost.provenance
    assert tracker.reason == 'cost_budget'


def test_unknown_cost_stays_unknown_and_fails_closed():
    tracker = BudgetTracker(Budget(max_cost_usd=1))
    tracker.add_usage(Usage(), Cost(provenance='response missing usage'))
    tracker.add_usage(Usage(input_tokens=10), Cost(usd=0.1, provenance='estimate'))
    assert tracker.cost.usd is None
    with pytest.raises(BudgetExceeded, match='cost_budget'):
        tracker.check()


def test_zero_budget_blocks_first_call():
    tracker = BudgetTracker(Budget(max_cost_usd=0))
    with pytest.raises(BudgetExceeded, match='cost_budget'):
        tracker.before_turn()
    assert tracker.turns == 0


def test_value_models_are_frozen():
    with pytest.raises(ValidationError):
        Budget().max_turns = 2
