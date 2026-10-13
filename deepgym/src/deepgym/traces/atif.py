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
* Each trace step becomes, in order: one step per recorded model message before the
  reply (``system`` -> system, ``user`` -> user, ``tool`` -> user with
  ``extra.role='tool'``, ``assistant`` -> agent with ``llm_call_count=1``), then one
  ``agent`` step for the action. That step's ``message`` is the assistant reply (the
  last message, when it's the assistant's; '' otherwise), its single tool call is the
  action (``function_name`` = action type, ``arguments`` = the action's non-default
  fields) and its observation is what the action produced, or the error if it produced
  nothing. Exit code, error and done flag go in the observation result's ``extra``; the
  step reward and step verifier output in the step's ``extra``.
* Assistant token ids and logprobs become ``metrics.completion_token_ids`` and
  ``metrics.logprobs``; token data on other roles has no ATIF field and is dropped.
* Screenshots are PNG files under ``images/<trace_id>/`` next to the document, referenced
  from ``image`` content parts, as the spec's storage convention says.
* Root ``extra.deepgym`` holds the trace schema version, provenance, final result,
  status, ``has_verdict``, error and finish time.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import JsonValue

from deepgym.exceptions import DeepGymError
from deepgym.models import Observation
from deepgym.traces.schema import Trace, TraceMessage, TraceStep, json_text

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
        reset = content(trace.initial_observation, 'reset')
        steps.append(
            {'source': 'system', 'message': 'Environment reset', 'observation': _results(reset)}
        )
    for step in trace.steps:
        steps.extend(_turn(step, content))
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
                mode='json', exclude={'task', 'initial_observation', 'steps'}
            )
        },
    }  # fmt: skip
    return document, images


def write_atif(traces: Sequence[Trace], out: Path) -> int:
    """Write each trace as ``<out>/<trace_id>.json``, with its screenshots under ``images/``.

    Returns:
        The number of documents written.

    Raises:
        DeepGymError: If *out* already has files: documents left from an earlier export
            would be read as part of this one.
    """
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise DeepGymError(f'{out} is not empty; export ATIF into a new directory')
    for trace in traces:
        document, images = to_atif(trace)
        (out / f'{trace.trace_id}.json').write_text(
            json_text(document, indent=2) + '\n', encoding='utf-8'
        )
        for relative, png in images.items():
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(png)
    return len(traces)


def _turn(
    step: TraceStep, content: Callable[[Observation, str], Content]
) -> list[dict[str, JsonValue]]:
    """Return the ATIF steps for one trace step: context messages, then the action."""
    messages = step.messages
    has_reply = bool(messages) and messages[-1].role == 'assistant'
    context, reply = (messages[:-1], messages[-1]) if has_reply else (messages, None)
    turn = [_message_step(message) for message in context]
    call_id = f'call_{step.index}'
    observed = (
        (step.error or '')
        if step.observation is None
        else content(step.observation, f'step_{step.index}')
    )
    turn.append(
        _compact({
            'timestamp': step.timestamp.isoformat(),
            'source': 'agent',
            'message': '' if reply is None else reply.content,
            'tool_calls': [{
                'tool_call_id': call_id,
                'function_name': step.action.action_type,
                'arguments': step.action.model_dump(
                    mode='json', exclude={'action_type'}, exclude_defaults=True
                ),
            }],
            'observation': _results(observed, call_id, _compact({
                'exit_code': step.exit_code,
                'done': None if step.observation is None else step.observation.done,
                'error': step.error,
            })),
            'metrics': None if reply is None else _metrics(reply),
            'llm_call_count': None if reply is None else 1,
            'extra': None if step.reward is None else _compact({
                'reward': step.reward, 'step_verifier': step.step_verifier,
            }),
        })
    )  # fmt: skip
    return turn


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


def _results(
    content: Content, call_id: str | None = None, extra: dict[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    """Return an ATIF observation holding one result."""
    return {
        'results': [
            _compact({'source_call_id': call_id, 'content': content, 'extra': extra or None})
        ]
    }


def _compact(fields: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Drop None values: ATIF omits optional fields rather than nulling them."""
    return {key: value for key, value in fields.items() if value is not None}
