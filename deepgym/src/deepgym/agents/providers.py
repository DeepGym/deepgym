"""Provider request plumbing shared by the host-side agents: accounting and response guards.

USD enforcement uses returned usage and caller-supplied prices, so one response can
overshoot the cap. Only a complete response is ever acted on: an Anthropic stream must
reach message_stop, and its tool calls run only if it stopped for them; an OpenAI response
must have status completed.
"""

from __future__ import annotations

from typing import Any, Literal

from deepgym.agents.base import Budget, BudgetTracker, Cost, TokenPrices, Usage

# The SDK retries connection drops, 429s and 5xx with backoff; one blip shouldn't end an
# hours-long episode. A retried request may be billed twice; usage counts only answers.
SDK_RETRIES = 4
ANTHROPIC_MAX_TOKENS = (
    128_000  # Opus 5.5's maximum output (the API states it); streamed, as the SDK requires
)


def _snapshot(stream: Any) -> Any:
    """The accumulated message, or None when no message_start arrived (the SDK asserts then)."""
    try:
        return stream.current_message_snapshot
    except AssertionError:
        return None


def _incomplete(
    tracker: BudgetTracker, message: dict[str, Any], prices: TokenPrices | None
) -> None:
    """Keep the usage an unfinished stream reported, with the total bill marked unresolved."""
    data = message.get('usage')
    usage = _anthropic_usage(data) if isinstance(data, dict) else Usage()
    unresolved = 'incomplete streamed response: partial usage, billing unresolved'
    tracker.add_usage(usage, Cost(provenance=unresolved))


def anthropic_message(
    client: Any, tracker: BudgetTracker, prices: TokenPrices | None, **request: Any
) -> dict[str, Any]:
    """Stream one Anthropic response; return it whole, or what arrived of it marked incomplete.

    Every event is consumed and only a received message_stop completes a response. An
    unfinished one is never acted on (see :func:`anthropic_incomplete`): the usage it
    reported is kept, its billing unresolved, and it comes back as the partial message
    (empty if none started) with ``status`` ``incomplete`` and the reason in
    ``incomplete_details``, as OpenAI marks its own, with the text of what ended the
    stream as ``detail``. A message that didn't stop for tool use also carries
    ``input_json``: each tool call's input exactly as it streamed, by content index, since
    the SDK's parse of an unfinished one drops what doesn't parse.

    Args:
        client: The Anthropic client.
        tracker: The episode's accounting.
        prices: Caller rates, if any.
        **request: ``messages.stream`` keyword arguments.

    Raises:
        anthropic.APIError: If the request itself fails.
    """
    with client.messages.stream(**request) as stream:
        stopped = False
        failure: Exception | None = None
        fragments: dict[str, str] = {}
        try:
            for event in stream:
                stopped = stopped or event.type == 'message_stop'
                if event.type == 'content_block_delta' and event.delta.type == 'input_json_delta':
                    index = str(event.index)
                    fragments[index] = fragments.get(index, '') + event.delta.partial_json
        # Anything raised while reading (httpx or httpx2 transport errors, by SDK
        # version, or an error event) ends this response.
        except Exception as exc:  # noqa: BLE001
            failure = exc
        snapshot = _snapshot(stream)
    message = snapshot.model_dump(exclude_none=True) if snapshot is not None else {}
    if fragments and message.get('stop_reason') != 'tool_use':
        message['input_json'] = fragments
    if failure is not None or not stopped or snapshot is None:
        _incomplete(tracker, message, prices)
        details = {'reason': 'Response stream ended before completion'}
        if failure is not None:
            # Its text too: the SDK's error for a delta it can't parse quotes that delta,
            # which never reaches the snapshot or the fragments above.
            details = {
                'reason': f'Response stream failed: {type(failure).__name__}',
                'detail': str(failure),
            }
        return {**message, 'status': 'incomplete', 'incomplete_details': details}
    return message


def anthropic_incomplete(message: dict[str, Any]) -> str | None:
    """Why :func:`anthropic_message` returned *message* incomplete, or None if it completed."""
    if message.get('status') != 'incomplete':
        return None
    return message['incomplete_details']['reason']


def anthropic_unfinished_calls(message: dict[str, Any]) -> str | None:
    """Why a complete *message*'s tool calls must not run, or None if they may.

    Its tool calls are whole only if it stopped for tool use; one cut at max_tokens may
    hold a call whose input was parsed from a fragment.
    """
    stop = message.get('stop_reason')
    if stop == 'tool_use' or not any(part['type'] == 'tool_use' for part in message['content']):
        return None
    return f'Response stopped ({stop}) before its tool calls were complete'


def openai_incomplete(response: dict[str, Any]) -> str | None:
    """Why an OpenAI response did not complete, or None if it did.

    Never act on, or blindly continue, a partial response; keep its reason.
    """
    if response.get('status') == 'completed':
        return None
    details = response.get('incomplete_details') or {}
    reason = details.get('reason') or response.get('status')
    return f'Response did not complete: {reason}'


def account(
    tracker: BudgetTracker,
    data: dict[str, Any] | None,
    prices: TokenPrices | None,
    provider: Literal['anthropic', 'openai'],
) -> None:
    """Add a response's usage and cost, then stop if a bound is spent.

    Raises:
        ValueError: If the provider returned no usage (billing is then unknown).
        BudgetExceeded: If time or spend is exhausted.
    """
    try:
        if not isinstance(data, dict):
            raise ValueError('provider did not return token usage')
        usage = _anthropic_usage(data) if provider == 'anthropic' else _openai_usage(data)
    except (KeyError, TypeError, ValueError):
        tracker.add_usage(
            Usage(), Cost(provenance=f'{provider} usage unavailable; billing unknown')
        )
        raise
    tracker.add_usage(
        usage,
        prices.estimate(usage)
        if prices
        else Cost(
            provenance='provider token usage; prices not supplied',
        ),
    )
    tracker.check()


def validate_budget(budget: Budget, prices: TokenPrices | None) -> None:
    """Refuse a USD budget without explicit prices to measure it.

    Raises:
        ValueError: If *budget* has a USD limit and *prices* is None.
    """
    if budget.max_cost_usd is not None and prices is None:
        raise ValueError('A USD budget requires explicit TokenPrices with source provenance')


def _anthropic_usage(data: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=data['input_tokens'],
        output_tokens=data['output_tokens'],
        cache_read_tokens=data.get('cache_read_input_tokens') or 0,
        cache_write_tokens=data.get('cache_creation_input_tokens') or 0,
    )


def _openai_usage(data: dict[str, Any]) -> Usage:
    details = data.get('input_tokens_details') or {}
    cached = details.get('cached_tokens') or 0
    written = details.get('cache_write_tokens') or 0
    return Usage(
        input_tokens=data['input_tokens'] - cached - written,
        output_tokens=data['output_tokens'],
        cache_read_tokens=cached,
        cache_write_tokens=written,
    )


def api_error(exc: Exception) -> str:
    """Describe a provider API error by class, status and the provider's own validation message.

    Only the structured error body is used, never the exception text, which may echo request
    content.
    """
    status = getattr(exc, 'status_code', None)
    body = getattr(exc, 'body', None)
    error = body.get('error', body) if isinstance(body, dict) else None
    detail = error.get('message') if isinstance(error, dict) else None
    status_text = f' {status}' if status else ''
    detail_text = f': {str(detail)[:300]}' if detail else ''
    return f'{type(exc).__name__}{status_text}{detail_text}'
