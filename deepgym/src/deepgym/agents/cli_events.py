"""Interpret CLI JSON events as observations, never executable Session actions.

Claude usage is per assistant message until its cumulative result arrives. Codex
reports usage only at ``turn.completed`` (the entire user task, not one model call).
Unknown event types remain in the Session's raw nested events for forward fidelity.
"""

from __future__ import annotations

from typing import Literal

from pydantic import JsonValue

from deepgym.agents.base import BudgetTracker, Cost, TerminationReason, TokenPrices, Usage

_PROVIDER_CODES = frozenset(
    {
        'usageLimitExceeded',
        'rateLimitExceeded',
        'serverOverloaded',
        'internalServerError',
        'unauthorized',
    }
)
_PROVIDER_TRANSPORT_CODES = frozenset(
    {
        'httpConnectionFailed',
        'responseStreamConnectionFailed',
        'responseStreamDisconnected',
        'responseTooManyFailedAttempts',
    }
)


def _codex_error_reason(detail: dict[str, JsonValue]) -> Literal['error', 'provider_error']:
    """Classify only explicit provider codes; message-only exec errors stay generic."""
    code = detail.get('codexErrorInfo')
    if isinstance(code, str) and code in _PROVIDER_CODES:
        return 'provider_error'
    if isinstance(code, dict) and len(code) == 1:
        name, metadata = next(iter(code.items()))
        if name in _PROVIDER_TRANSPORT_CODES and isinstance(metadata, dict):
            status = metadata.get('httpStatusCode')
            if status is None or (
                type(status) is int and (status in (401, 403, 408, 429) or 500 <= status <= 599)
            ):
                return 'provider_error'
    return 'error'


def _object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    return value


def _tokens(data: dict[str, JsonValue], key: str, *, optional: bool = False) -> int:
    value = data.get(key, 0 if optional else None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f'{key} must be a non-negative integer')
    return value


def _usage(data: dict[str, JsonValue], provider: str) -> Usage:
    cached = _tokens(
        data,
        'cache_read_input_tokens' if provider == 'anthropic' else 'cached_input_tokens',
        optional=True,
    )
    input_tokens = _tokens(data, 'input_tokens')
    if provider == 'openai':
        input_tokens -= cached
        if input_tokens < 0:
            raise ValueError('cached_input_tokens exceeds input_tokens')
    return Usage(
        input_tokens=input_tokens,
        output_tokens=_tokens(data, 'output_tokens'),
        cache_read_tokens=cached,
        cache_write_tokens=_tokens(data, 'cache_creation_input_tokens', optional=True),
    )


class CliEvents:
    """Track a CLI stream while leaving all side effects to the CLI process.

    Turns are unique Claude assistant messages or completed Codex activity items.
    Neither JSON stream exposes every request before it executes; caps are enforced
    at observed event boundaries with supervisor polling latency. A cancelled run's
    token usage can be incomplete. Cost provenance states that limitation.

    Args:
        provider: CLI event dialect.
        tracker: Episode accounting and monotonic deadline.
        prices: Optional caller-supplied rates for token estimates.
    """

    def __init__(
        self,
        provider: Literal['anthropic', 'openai'],
        tracker: BudgetTracker,
        prices: TokenPrices | None = None,
    ) -> None:
        self.provider = provider
        self.tracker = tracker
        self.prices = prices
        self.completed = False
        self.final_text = ''
        self.error: str | None = None
        self._error_reason: Literal['error', 'provider_error'] = 'error'
        self.stop_reason: TerminationReason | None = None
        self._seen: set[str] = set()
        self._messages: dict[str, Usage | None] = {}
        self._unreported_messages: set[str] = set()
        self._reported_usage = False
        self.tracker.cost = Cost(provenance='CLI usage/cost not yet reported')

    def consume(self, event: dict[str, JsonValue]) -> str | None:
        """Consume one event and request supervisor termination when a bound fires.

        Args:
            event: One decoded stdout JSON object.

        Returns:
            A termination reason, or None to keep reading.
        """
        first_stop = self.stop_reason
        try:
            if not isinstance(event.get('type'), str):
                raise ValueError('CLI event requires a string type')
            if self.provider == 'anthropic':
                self._claude(event)
            else:
                self._codex(event)
        except (ValueError, TypeError) as exc:
            self.error = f'Invalid {self.provider} CLI event: {exc}'
            self._error_reason = 'error'
        if first_stop is not None:
            self.stop_reason = first_stop
            return first_stop
        if self.error:
            self.stop_reason = self._error_reason
        elif self.tracker.remaining_seconds <= 0:
            self.stop_reason = 'time_budget'
        elif (
            self.tracker.budget.max_cost_usd is not None
            and self.tracker.cost.usd is not None
            and self.tracker.cost.usd >= self.tracker.budget.max_cost_usd
        ):
            self.stop_reason = 'cost_budget'
        elif (
            not self.completed
            and self.tracker.budget.max_turns is not None
            and self.tracker.turns >= self.tracker.budget.max_turns
        ):
            self.stop_reason = 'turn_budget'
        return self.stop_reason

    def finish(self, exit_code: int, stop_reason: str | None) -> TerminationReason:
        """Validate terminal evidence and label incomplete accounting on interruption.

        Args:
            exit_code: Supervised process exit code.
            stop_reason: Supervisor deadline or event-bound stop reason.

        Returns:
            The observed episode termination reason.
        """
        reason = self.stop_reason or stop_reason
        passthrough = ('turn_budget', 'time_budget', 'cost_budget', 'output_limit')
        if reason not in (*passthrough, 'provider_error', 'error'):
            if self.error is None and exit_code != 0:
                self.error = f'{self.provider} CLI exited with code {exit_code}'
            if self.error is None and not self.completed:
                self.error = f'{self.provider} CLI exited without a terminal completion event'
            if self.error is None and (not self._reported_usage or self._unreported_messages):
                self.error = f'{self.provider} CLI completed without token usage'
            reason = self._error_reason if self.error else 'completed'
        if reason != 'completed':
            self.tracker.cost = self.tracker.cost.model_copy(
                update={
                    'provenance': self.tracker.cost.provenance
                    + '; incomplete observed CLI subtotal: additional usage may be unreported',
                }
            )
        return reason

    def _record_usage(self, usage: Usage, *, cumulative: bool = False) -> None:
        if cumulative:
            if any(
                value < getattr(self.tracker.usage, key)
                for key, value in usage.model_dump().items()
            ):
                raise ValueError('cumulative result usage is below observed assistant usage')
            self.tracker.usage = usage
            self._unreported_messages.clear()
        else:
            self.tracker.usage = Usage(
                **{
                    key: getattr(self.tracker.usage, key) + value
                    for key, value in usage.model_dump().items()
                }
            )
        self._reported_usage = True
        self.tracker.cost = (
            self.prices.estimate(self.tracker.usage)
            if self.prices is not None
            else Cost(provenance='CLI reported token usage; USD unavailable without caller rates')
        )
        if self._unreported_messages:
            self.tracker.cost = Cost(
                provenance='Some Claude assistant messages omitted token usage'
            )

    def _claude(self, event: dict[str, JsonValue]) -> None:
        kind = event['type']
        if kind == 'assistant':
            message = _object(event.get('message'), 'assistant message')
            identity = message.get('id')
            if not isinstance(identity, str) or not identity:
                raise ValueError('assistant message requires an id')
            if identity not in self._messages:
                self.tracker.turns += 1
            if message.get('usage') is None:
                self._messages.setdefault(identity, None)
                self._unreported_messages.add(identity)
                self.tracker.cost = Cost(provenance='Claude assistant message omitted token usage')
                if self.tracker.budget.max_cost_usd is not None:
                    raise ValueError('assistant usage is required for a finite USD budget')
            else:
                usage = _usage(_object(message['usage'], 'assistant usage'), self.provider)
                previous = self._messages.get(identity) or Usage()
                delta = {
                    key: value - getattr(previous, key) for key, value in usage.model_dump().items()
                }
                if any(value < 0 for value in delta.values()):
                    raise ValueError('assistant message usage counters regressed')
                self._messages[identity] = usage
                self._unreported_messages.discard(identity)
                self._record_usage(Usage(**delta))
            content = message.get('content')
            if not isinstance(content, list):
                raise ValueError('assistant content must be an array')
            texts = [
                block['text']
                for block in content
                if isinstance(block, dict)
                and block.get('type') == 'text'
                and isinstance(block.get('text'), str)
            ]
            if texts:
                self.final_text = '\n'.join(texts)
        elif kind == 'result':
            subtype = event.get('subtype')
            if subtype == 'error_max_turns':
                self.stop_reason = 'turn_budget'
            elif subtype == 'error_max_budget_usd':
                self.stop_reason = 'cost_budget'
            elif event.get('is_error') is True or subtype != 'success':
                self.error = str(event.get('errors') or event.get('result') or subtype)
            else:
                self.completed = True
                if isinstance(event.get('result'), str):
                    self.final_text = event['result']
            if event.get('usage') is not None:
                self._record_usage(
                    _usage(_object(event['usage'], 'result usage'), self.provider),
                    cumulative=True,
                )
            if event.get('total_cost_usd') is not None:
                self.tracker.cost = Cost(
                    usd=event['total_cost_usd'],
                    provenance=(
                        'Claude Code result.total_cost_usd (CLI-reported, not billing verified)'
                    ),
                )

    def _codex(self, event: dict[str, JsonValue]) -> None:
        kind = event['type']
        if kind == 'item.completed':
            item = _object(event.get('item'), 'completed item')
            identity = item.get('id')
            if not isinstance(identity, str) or not identity:
                raise ValueError('completed item requires an id')
            if identity in self._seen:
                return
            self._seen.add(identity)
            self.tracker.turns += 1
            if item.get('type') == 'agent_message':
                if not isinstance(item.get('text'), str):
                    raise ValueError('agent_message requires text')
                self.final_text = item['text']
        elif kind == 'turn.completed':
            self._record_usage(_usage(_object(event.get('usage'), 'turn usage'), self.provider))
            self.completed = True
        elif kind == 'error':
            # exec can strip willRetry; only a terminal event/exit proves failure.
            _object(event.get('error', event), 'CLI error')
        elif kind == 'turn.failed':
            detail = event.get('error', event)
            detail = _object(detail, 'CLI error')
            self.error = str(detail.get('message') or 'CLI turn failed')
            self._error_reason = _codex_error_reason(detail)
