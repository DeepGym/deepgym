"""Agent contracts and shared, monotonic episode accounting."""

from __future__ import annotations

import math
import time
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from deepgym.exceptions import DeepGymError
from deepgym.models import Observation, Trajectory
from deepgym.session import Session

TerminationReason = Literal[
    'completed',
    'turn_budget',
    'time_budget',
    'cost_budget',
    'session_done',
    'refusal',
    'safety_check',
    'error',
    'provider_error',
]


class Budget(BaseModel):
    """Bound episode work; USD limits stop new calls after reported spend reaches the cap.

    An in-flight response can exceed the USD limit. Provider billing and model-readable
    CLI credentials cannot be hard-capped by this local accounting mechanism.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    max_turns: int | None = Field(default=30, ge=1)  # None: no turn limit
    max_seconds: float | None = Field(default=300, gt=0)  # None: no wall-clock limit
    max_cost_usd: float | None = Field(default=None, ge=0)


class Usage(BaseModel):
    """Record disjoint token categories; output includes reasoning tokens."""

    model_config = ConfigDict(frozen=True)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)


class Cost(BaseModel):
    """Report USD with its measurement or estimation provenance; None means unknown."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    usd: float | None = Field(default=None, ge=0)
    provenance: str = Field(default='unavailable', min_length=1)


class TokenPrices(BaseModel):
    """Supply model-specific USD rates per million tokens and their dated source.

    Rates are caller-supplied, never silently inferred from a mutable model alias.
    Cache prices are required so cached tokens cannot become silently free.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    cache_read_per_million: float = Field(ge=0)
    cache_write_per_million: float = Field(ge=0)
    source: str = Field(min_length=1)

    def estimate(self, usage: Usage) -> Cost:
        """Calculate cost from provider usage and caller-supplied rates."""
        total = (
            usage.input_tokens * self.input_per_million
            + usage.output_tokens * self.output_per_million
            + usage.cache_read_tokens * self.cache_read_per_million
            + usage.cache_write_tokens * self.cache_write_per_million
        ) / 1_000_000
        return Cost(usd=total, provenance=f'provider token usage * caller rates: {self.source}')


class AgentResult(BaseModel):
    """Return the observed trajectory and accounting, without grading or closing it."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    trajectory: Trajectory
    usage: Usage = Field(default_factory=Usage)
    cost: Cost = Field(default_factory=Cost)
    termination_reason: TerminationReason
    turns: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0, ge=0)
    final_text: str = ''
    error: str | None = None
    cli_version: str | None = None
    """Installed CLI version observed by the sandbox supervisor via --version."""


@runtime_checkable
class Agent(Protocol):
    """Drive an already-started session; its caller owns scoring and cleanup."""

    def run(
        self, session: Session, initial: Observation, *, task: str, budget: Budget
    ) -> AgentResult:
        """Execute the task within budget and return observed work."""
        ...


class BudgetExceeded(DeepGymError):  # noqa: N818 - control-flow signal, like StopIteration
    """Signal that no further provider calls or actions are permitted."""

    def __init__(self, reason: TerminationReason) -> None:
        super().__init__(reason)
        self.reason = reason


class BudgetTracker:
    """Share cumulative accounting across every turn, including failed calls.

    Check before calls and actions, and after responses. SDK request timeouts use
    remaining wall time; Session propagates remaining deadlines to backend operations.
    SDK transport timeouts alone do not guarantee an absolute response deadline.
    """

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.started = time.monotonic()
        self.turns = 0
        self.usage = Usage()
        self.cost = Cost(usd=0, provenance='no provider calls')

    @property
    def remaining_seconds(self) -> float:
        """Return remaining wall time, including retries and tools; inf without a limit."""
        if self.budget.max_seconds is None:
            return math.inf
        return max(0, self.budget.max_seconds - (time.monotonic() - self.started))

    @property
    def request_timeout(self) -> float | None:
        """Deadline for one provider call or action; None without a wall-clock limit."""
        return None if self.budget.max_seconds is None else self.remaining_seconds

    @property
    def reason(self) -> TerminationReason | None:
        """Return an exhausted time or spend bound, without checking turn admission."""
        if self.remaining_seconds <= 0:
            return 'time_budget'
        limit = self.budget.max_cost_usd
        if limit is not None and (self.cost.usd is None or self.cost.usd >= limit):
            return 'cost_budget'
        return None

    def check(self) -> None:
        """Raise BudgetExceeded when time or spend disallows further work."""
        if reason := self.reason:
            raise BudgetExceeded(reason)

    def before_turn(self) -> None:
        """Admit one provider call, raising BudgetExceeded at an exhausted bound."""
        self.check()
        if self.budget.max_turns is not None and self.turns >= self.budget.max_turns:
            raise BudgetExceeded('turn_budget')
        self.turns += 1

    def add_usage(self, usage: Usage, cost: Cost) -> None:
        """Accumulate disjoint response usage and propagate unknown costs."""
        self.usage = Usage(
            **{
                name: getattr(self.usage, name) + value
                for name, value in usage.model_dump().items()
            }
        )
        total = None if self.cost.usd is None or cost.usd is None else self.cost.usd + cost.usd
        sources = list(dict.fromkeys((self.cost.provenance, cost.provenance)))
        self.cost = Cost(
            usd=total,
            provenance='; '.join(source for source in sources if source != 'no provider calls'),
        )

    def result(
        self,
        session: Session,
        reason: TerminationReason,
        final_text: str = '',
        error: str | None = None,
    ) -> AgentResult:
        """Snapshot accounting and the session's real actions without grading."""
        return AgentResult(
            trajectory=session.trajectory.model_copy(deep=True),
            usage=self.usage,
            cost=self.cost,
            termination_reason=reason,
            turns=self.turns,
            elapsed_seconds=time.monotonic() - self.started,
            final_text=final_text,
            error=error,
        )
