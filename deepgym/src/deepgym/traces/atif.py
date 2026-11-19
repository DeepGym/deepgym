"""Convert traces to ATIF, Harbor's Agent Trajectory Interchange Format (ATIF-v1.7).

Spec: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md.
ATIF publishes no JSON Schema; its reference validator is Harbor's Pydantic models
(``harbor.models.trajectories.Trajectory``), which the tests use when Harbor is installed.

Mapping, one ATIF document per trace:

* ``session_id`` and ``trajectory_id`` are the trace id. ``agent`` is the caller's
  :class:`AgentMeta` (``version`` is 'unknown' when not supplied; sampling goes in
  ``agent.extra.sampling``).
* Step 1 is a ``user`` step with the task. If the session started, a ``system`` step
  carries the first observation (an environment reset, in ATIF's terms).
* Each model turn (:attr:`Trace.turns`) becomes, in order: one step per recorded model
  message before the reply (``system`` -> system, ``user`` -> user, ``tool`` -> user
  with ``extra.role='tool'``, ``assistant`` -> agent with ``llm_call_count=1``), then one
  ``agent`` step for its actions. That step's ``message`` is the assistant reply (the
  last message, when it's the assistant's; '' otherwise), and it has one tool call per
  action (``function_name`` = action type, ``arguments`` = the action's non-default
  fields), in order, each with its own observation result: what the action produced, or
  the error if it produced nothing. Exit code, done flag, ``applied``, error, step
  reward and step verifier output go in that result's ``extra``.
* CLI launches remain one tool call; their raw nested events are preserved in the
  observation result's ``extra.cli_events``, never translated into executable actions.
* Assistant token ids and logprobs become ``metrics.completion_token_ids`` and
  ``metrics.logprobs``; token data on other roles has no ATIF field and is dropped.
* The provider usage of a turn's steps (:attr:`TraceStep.usage`), summed, goes in the
  metrics of its ``agent`` step: ``prompt_tokens`` counts every input token, cache reads
  and writes included, ``cached_tokens`` the cache reads, ``completion_tokens`` the output
  (reasoning included, so it overrides the reply's token-id count), ``cost_usd`` the cost,
  and ``extra`` the cache writes and reasoning tokens, which ATIF has no field for. That
  usage covers every reply in the turn, those that issued no action included, so the
  agent step's ``llm_call_count`` is the number of assistant messages in the turn (None
  when the turn records no messages, as a computer agent's steps don't), and their own
  steps carry neither ``completion_tokens`` nor ``llm_call_count``: step sums never count
  a call twice. The trace's totals go in ``final_metrics`` the same way
  (``total_prompt_tokens``, ...), so they include :attr:`Trace.unattached_usage`, which
  is on no step; the root ``notes`` say so when there is some, and say when usage is
  incomplete.
* Screenshots are PNG files under ``images/<trace_id>/`` next to the document, referenced
  from ``image`` content parts, as the spec's storage convention says.
* Root ``extra.deepgym`` holds the trace schema version, provenance, final result and
  any ``prior_results``, status, ``has_verdict``, error and finish time.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import JsonValue

from deepgym.exceptions import DeepGymError
from deepgym.models import Observation
from deepgym.traces.schema import (
    Trace,
    TraceMessage,
    TraceStep,
    Usage,
    check_unique_ids,
    json_text,
    sum_usage,
)

ATIF_VERSION = 'ATIF-v1.7'

Content = str | list[dict[str, JsonValue]]
_ROLE_SOURCE = {'system': 'system', 'user': 'user', 'tool': 'user', 'assistant': 'agent'}


def to_atif(trace: Trace) -> tuple[dict[str, JsonValue], dict[str, bytes]]:
    """Convert *trace* to an ATIF document.

    Args:
        trace: The trace to convert.

    Returns:
        The document, and the screenshots it references as {relative path: PNG bytes}.

    Raises:
        binascii.Error: If a screenshot isn't valid base64.
    """
    images: dict[str, bytes] = {}

    def content(observation: Observation, name: str) -> Content:
        if not observation.screenshot:
            return observation.content
        path = f'images/{trace.trace_id}/{name}.png'
        images[path] = base64.b64decode(observation.screenshot, validate=True)
        return [
            {'type': 'text', 'text': observation.content},
            {'type': 'image', 'source': {'media_type': 'image/png', 'path': path}},
        ]

    steps: list[dict[str, JsonValue]] = [{'source': 'user', 'message': trace.task}]
    if trace.initial_observation is not None:
        reset = {'results': [_result(content(trace.initial_observation, 'reset'))]}
        steps.append({'source': 'system', 'message': 'Environment reset', 'observation': reset})
    for turn in trace.turns:
        steps.extend(_turn([trace.steps[index - 1] for index in turn], content))
    agent = trace.provenance.agent
    document = _compact({
        'schema_version': ATIF_VERSION,
        'session_id': trace.trace_id,
        'trajectory_id': trace.trace_id,
        'agent': _compact({
            'name': agent.name,
            'version': agent.version or 'unknown',
            'model_name': agent.model,
            'extra': {'sampling': agent.sampling} if agent.sampling else None,
        }),
        'steps': [{'step_id': number, **step} for number, step in enumerate(steps, 1)],
        'notes': _notes(trace),
        'final_metrics': _usage_metrics(trace.usage, trace.cost_usd, 'total_') or None,
        'extra': {
            'deepgym': trace.model_dump(
                mode='json', exclude={'task', 'initial_observation', 'steps', 'turns'}
            )
        },
    })  # fmt: skip
    return document, images


def write_atif(traces: Sequence[Trace], out: Path) -> int:
    """Write each trace as ``<out>/<trace_id>.json``, with its screenshots under ``images/``.

    Returns:
        The number of documents written.

    Raises:
        ValueError: If two traces share an id (one document would replace the other).
        DeepGymError: If *out* already has files: documents left from an earlier export
            would be read as part of this one.
    """
    check_unique_ids(traces)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise DeepGymError(f'{out} is not empty; export ATIF into a new directory')
    # Files are created exclusively ('x'), so nothing written by this export is replaced.
    for trace in traces:
        document, images = to_atif(trace)
        with (out / f'{trace.trace_id}.json').open('x', encoding='utf-8') as file:
            file.write(json_text(document, indent=2) + '\n')
        for relative, png in images.items():
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as file:
                file.write(png)
    return len(traces)


def _turn(
    steps: Sequence[TraceStep], content: Callable[[Observation, str], Content]
) -> list[dict[str, JsonValue]]:
    """Return the ATIF steps for one model turn: context messages, then its actions."""
    messages = steps[0].messages
    has_reply = bool(messages) and messages[-1].role == 'assistant'
    context, reply = (messages[:-1], messages[-1]) if has_reply else (messages, None)
    calls: list[JsonValue] = []
    results: list[JsonValue] = []
    for step in steps:
        call_id = f'call_{step.index}'
        calls.append({
            'tool_call_id': call_id,
            'function_name': step.action.action_type,
            'arguments': step.action.model_dump(
                mode='json', exclude={'action_type', 'events'}, exclude_defaults=True
            ),
        })  # fmt: skip
        observed = (
            (step.error or '')
            if step.observation is None
            else content(step.observation, f'step_{step.index}')
        )
        results.append(_result(observed, call_id, _compact({
            'exit_code': step.exit_code,
            'done': None if step.observation is None else step.observation.done,
            'applied': step.applied,
            'error': step.error,
            'reward': step.reward,
            'step_verifier': step.step_verifier,
            'cli_events': list(step.action.events) if step.action.action_type == 'cli' else None,
        })))  # fmt: skip
    usage, cost_usd = sum_usage((step.usage, step.cost_usd) for step in steps)
    if usage is None:
        calls_made = None if reply is None else 1
    else:  # the turn's usage covers every reply in it, those before the last included
        calls_made = sum(message.role == 'assistant' for message in messages) or None
    action = _compact({
        'timestamp': steps[0].timestamp.isoformat(),
        'source': 'agent',
        'message': '' if reply is None else reply.content,
        'tool_calls': calls,
        'observation': {'results': results},
        'metrics': {**_token_metrics(reply), **_usage_metrics(usage, cost_usd)} or None,
        'llm_call_count': calls_made,
    })  # fmt: skip
    counted = usage is not None
    return [*(_message_step(message, counted) for message in context), action]


def _message_step(message: TraceMessage, counted: bool) -> dict[str, JsonValue]:
    """Return the ATIF step for a model message that isn't the action's reply.

    When the turn's usage is *counted* on its agent step, an assistant message's call is
    counted there, so its own step has no ``completion_tokens`` or ``llm_call_count``.
    """
    source = _ROLE_SOURCE[message.role]
    agent = source == 'agent'
    metrics = _token_metrics(message, count=not counted) if agent else {}
    return _compact({
        'source': source,
        'message': message.content,
        'metrics': metrics or None,
        'llm_call_count': 1 if agent and not counted else None,
        'extra': {'role': 'tool'} if message.role == 'tool' else None,
    })  # fmt: skip


def _token_metrics(message: TraceMessage | None, *, count: bool = True) -> dict[str, JsonValue]:
    """Return ATIF metrics for an assistant message's token data, with their count unless
    not *count*; {} without any."""
    if message is None:
        return {}
    ids = message.token_ids
    return _compact({
        'completion_tokens': len(ids) if count and ids is not None else None,
        'completion_token_ids': None if ids is None else list(ids),
        'logprobs': None if message.logprobs is None else list(message.logprobs),
    })  # fmt: skip


def _notes(trace: Trace) -> str:
    """Return the document's notes: where DeepGym's own data is, and what usage covers."""
    notes = ['Exported from a DeepGym trace; provenance and the verdict are in extra.deepgym.']
    if trace.unattached_usage is not None:
        notes.append(
            'final_metrics include usage recorded on no step (extra.deepgym.unattached_usage).'
        )
    if not trace.usage_complete:
        notes.append(
            'Some model calls reported no usage, so token totals are a lower bound and the '
            'total cost is unknown.'
        )
    return ' '.join(notes)


def _usage_metrics(
    usage: Usage | None, cost_usd: float | None, prefix: str = ''
) -> dict[str, JsonValue]:
    """Return ATIF's token and cost fields for *usage*, named with *prefix*; {} without it."""
    if usage is None:
        return {}
    prompt = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    return _compact({
        f'{prefix}prompt_tokens': prompt,
        f'{prefix}completion_tokens': usage.output_tokens,
        f'{prefix}cached_tokens': usage.cache_read_tokens,
        f'{prefix}cost_usd': cost_usd,
        'extra': _compact({
            'cache_write_tokens': usage.cache_write_tokens,
            'reasoning_tokens': usage.reasoning_tokens,
        }),
    })  # fmt: skip


def _result(
    content: Content, call_id: str | None = None, extra: dict[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    """Return one ATIF observation result."""
    return _compact({'source_call_id': call_id, 'content': content, 'extra': extra or None})


def _compact(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Drop None values: ATIF omits optional fields rather than nulling them."""
    return {key: value for key, value in fields.items() if value is not None}
