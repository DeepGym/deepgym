"""Agent contracts and shared, monotonic episode accounting."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from deepgym.exceptions import DeepGymError, SessionClosedError, is_infrastructure_failure
from deepgym.models import Action, Observation, Trajectory
from deepgym.session import Session, StepResult
from deepgym.traces.capture import TracedSession
from deepgym.traces.schema import TraceMessage, Usage, sum_usage

logger = logging.getLogger(__name__)

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
    'setup_error',
    'output_limit',
]
"""Why an agent stopped. ``provider_error`` (the model or sandbox provider, or the network,
failed) and ``setup_error`` (the harness never started the model: its CLI missing or
refused) are nothing the policy did, so the episode gets no verdict (see
:meth:`BudgetTracker.result`). Every other reason is scored. A run the harness cut short is
graded with ``truncated`` set, so it shows as cut short rather than failed: a turn, time or
cost budget ran out, or (``output_limit``) a CLI run printed more than the host keeps (see
:data:`deepgym._cli_worker.OUTPUT_LIMIT_BYTES`)."""

CUT_SHORT: frozenset[TerminationReason] = frozenset(
    {'turn_budget', 'time_budget', 'cost_budget', 'output_limit'}
)
"""Reasons the harness cut a run short; it is graded with ``truncated`` set."""

# Error types a provider names in an error body, mid-stream too (sent under the stream's 200).
_PROVIDER_ERROR_TYPES = frozenset(
    {'api_error', 'overloaded_error', 'rate_limit_error', 'timeout_error', 'server_error'}
)


def provider_failed(exc: BaseException) -> bool:
    """Whether *exc* is the provider's or the network's failure rather than the policy's.

    That is a failure :func:`~deepgym.exceptions.is_infrastructure_failure` names (transport,
    5xx, 429, our credentials refused), or a provider error body naming one of its own
    failures (overloaded, say), which a stream reports mid-way under a 200 status.
    """
    body = getattr(exc, 'body', None)
    error = body.get('error', body) if isinstance(body, dict) else None
    kind = error.get('type') if isinstance(error, dict) else None
    return kind in _PROVIDER_ERROR_TYPES or is_infrastructure_failure(exc)


class Budget(BaseModel):
    """Bound episode work; USD limits stop new calls after reported spend reaches the cap.

    An in-flight response can exceed the USD limit. Provider billing and model-readable
    CLI credentials cannot be hard-capped by this local accounting mechanism.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    max_turns: int | None = Field(default=30, ge=1)  # None: no turn limit
    max_seconds: float | None = Field(default=300, gt=0)  # None: no wall-clock limit
    max_cost_usd: float | None = Field(default=None, ge=0)


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
    messages: tuple[TraceMessage, ...] = ()
    """Every model message as exchanged with the provider, for agents that record them (the
    API tool agents): the system prompt, the task, each full response with its usage, and
    each tool result. Complete however the run ended; empty for other agents."""


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
        self.usage = Usage(reasoning_tokens=0)
        self.cost = Cost(usd=0, provenance='no provider calls')
        self._held: list[tuple[Usage | None, float | None, bool]] = []
        self._in_flight = False  # a call was admitted and no usage has come back for it

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
        self._in_flight = True

    def add_usage(self, usage: Usage | None, cost: Cost, *, complete: bool = True) -> None:
        """Accumulate a response's usage and cost, propagating unknown costs, and hold them
        for the trace (see :meth:`take_unrecorded`).

        Args:
            usage: Disjoint token usage; None when the response reported none.
            cost: Its cost, with provenance.
            complete: False when *usage* is only part of what the response used.
        """
        self._in_flight = False
        if usage is not None:
            self.usage = self.usage + usage
        total = None if self.cost.usd is None or cost.usd is None else self.cost.usd + cost.usd
        sources = list(dict.fromkeys((self.cost.provenance, cost.provenance)))
        self.cost = Cost(
            usd=total,
            provenance='; '.join(source for source in sources if source != 'no provider calls'),
        )
        self.hold(usage, cost.usd, complete=complete)

    def hold(self, usage: Usage | None, cost_usd: float | None, *, complete: bool = True) -> None:
        """Hold usage for the next trace step or the trace's unattached usage, without
        counting it again here (a CLI run's, which its events already counted).

        Args:
            usage: Token usage; None when it is unknown.
            cost_usd: Its USD cost; None when unknown.
            complete: False when *usage* is only part of what was used.
        """
        self._held.append((usage, cost_usd, complete and usage is not None))

    def take_unrecorded(self) -> tuple[Usage | None, float | None, bool]:
        """Return and release what is held: the usage reported (None without any), its USD
        (None when any of it has no cost) and whether every call reported all its usage."""
        held, self._held = self._held, []
        usage, cost_usd = sum_usage((tokens, usd) for tokens, usd, _ in held)
        return usage, cost_usd, all(done for *_, done in held)

    def record_unattached(self, session: Session | TracedSession) -> None:
        """Record what is held as *session*'s unattached usage, if it is a TracedSession,
        and mark the trace's usage incomplete if any of it is, or if a call is still in
        flight (an exception escaped before its usage came back).

        Idempotent, since what it records is released, so :meth:`result` and an agent's
        ``finally`` can both call it. A trace already closed is left as it is (logged), so
        an error that ended the run is never masked by this.
        """
        if not isinstance(session, TracedSession):
            return
        if self._in_flight:
            self._in_flight = False
            self.hold(None, None, complete=False)
        usage, cost_usd, complete = self.take_unrecorded()
        try:
            if usage is not None:
                session.record_unattached_usage(usage, cost_usd)
            if not complete:
                session.mark_usage_incomplete()
        except SessionClosedError:
            logger.warning('Trace %s closed before its last usage was recorded', session.trace_id)

    def failure(self, exc: BaseException) -> TerminationReason:
        """Why an episode that *exc* ended stopped: ``time_budget`` once the clock ran out,
        ``provider_error`` if :func:`provider_failed`, else ``error``."""
        if self.remaining_seconds <= 0:
            return 'time_budget'
        return 'provider_error' if provider_failed(exc) else 'error'

    def result(
        self,
        session: Session,
        reason: TerminationReason,
        final_text: str = '',
        error: str | None = None,
    ) -> AgentResult:
        """Snapshot accounting and the session's real actions without grading.

        A reason that is nothing the policy did (``provider_error``, ``setup_error``)
        also abandons the session, so whoever scores it gets no verdict, not a 0. Under a
        TracedSession, usage no step took is recorded first (see :meth:`record_unattached`),
        so the trace's totals are the episode's.
        """
        self.record_unattached(session)
        if reason in ('provider_error', 'setup_error'):
            session.abandon(reason, error or reason)
        elif reason in CUT_SHORT:
            session.cut_short()
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


def apply_action(
    session: Session | TracedSession,
    action: Action,
    tracker: BudgetTracker,
    messages: Sequence[TraceMessage] = (),
) -> StepResult:
    """Apply *action* within the remaining time.

    Under a TracedSession the step also records *messages* and the usage and cost of the
    responses since the last step (see :meth:`BudgetTracker.take_unrecorded`): the first
    action of a reply carries the reply's usage, later ones none. Usage that includes a
    response without usage goes outside any step instead, and usage a refused step didn't
    record stays held.
    """
    timeout = tracker.request_timeout
    if not isinstance(session, TracedSession):
        return session.step(action, timeout=timeout)
    usage, cost_usd, complete = tracker.take_unrecorded()
    if not complete:
        # A response that reported no usage can't be priced on a step: record it outside.
        tracker.hold(usage, cost_usd, complete=False)
        tracker.record_unattached(session)
        usage = cost_usd = None
    recorded = session.step_count
    try:
        return session.step(action, messages, usage=usage, cost_usd=cost_usd, timeout=timeout)
    except BaseException:
        if usage is not None and session.step_count == recorded:
            tracker.hold(usage, cost_usd)  # refused before recording: hold it for later
        raise
