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
from deepgym.traces.schema import Trace, TraceMessage, TraceStep, check_unique_ids, json_text

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
    document = {
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
        'notes': 'Exported from a DeepGym trace; provenance and the verdict are in extra.deepgym.',
        'extra': {
            'deepgym': trace.model_dump(
                mode='json', exclude={'task', 'initial_observation', 'steps', 'turns'}
            )
        },
    }  # fmt: skip
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
    action = _compact({
        'timestamp': steps[0].timestamp.isoformat(),
        'source': 'agent',
        'message': '' if reply is None else reply.content,
        'tool_calls': calls,
        'observation': {'results': results},
        'metrics': None if reply is None else _metrics(reply),
        'llm_call_count': None if reply is None else 1,
    })  # fmt: skip
    return [*map(_message_step, context), action]


def _message_step(message: TraceMessage) -> dict[str, JsonValue]:
    """Return the ATIF step for a model message that isn't the action's reply."""
    source = _ROLE_SOURCE[message.role]
    agent = source == 'agent'
    return _compact({
        'source': source,
        'message': message.content,
        'metrics': _metrics(message) if agent else None,
        'llm_call_count': 1 if agent else None,
        'extra': {'role': 'tool'} if message.role == 'tool' else None,
    })  # fmt: skip


def _metrics(message: TraceMessage) -> dict[str, JsonValue] | None:
    """Return ATIF metrics for an assistant message's token data, or None without any."""
    metrics = _compact({
        'completion_tokens': None if message.token_ids is None else len(message.token_ids),
        'completion_token_ids': None if message.token_ids is None else list(message.token_ids),
        'logprobs': None if message.logprobs is None else list(message.logprobs),
    })  # fmt: skip
    return metrics or None


def _result(
    content: Content, call_id: str | None = None, extra: dict[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    """Return one ATIF observation result."""
    return _compact({'source_call_id': call_id, 'content': content, 'extra': extra or None})


def _compact(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Drop None values: ATIF omits optional fields rather than nulling them."""
    return {key: value for key, value in fields.items() if value is not None}
