"""Read a queue item's content from the console database: what a reviewer grades, and
nothing that reveals a grade (see :class:`deepgym.review_queues.ItemContent`).

One reader per target kind takes a connection, so the caller reads the content in the same
transaction as its access check.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence

from pydantic import JsonValue

from deepgym.console.db import TRACE_ID, TRACE_KEY
from deepgym.console.episode_store import select_held_traces
from deepgym.console.schemas import Frame
from deepgym.models import Action
from deepgym.registry import environment_dir, load_environment
from deepgym.review_queues import (
    ContentAction,
    ContentMessage,
    ContentStep,
    ItemContent,
    QueueItem,
)
from deepgym.traces import Trace, TraceStep
from deepgym.traces.schema import environment_sha256

# What a CLI agent's JSON events report about usage and cost (see
# deepgym.agents.cli_events): any key named here, or ending in a token count's suffix.
_USAGE_KEYS = frozenset({'usage', 'modelUsage', 'total_cost_usd', 'cost', 'cost_usd', 'costUSD'})
_TOKEN_SUFFIXES = ('_tokens', 'Tokens')


class UnreadableRecordingError(Exception):
    """A stored run, episode or trace that doesn't parse, or a file its content needs that
    can't be read. The cause is chained; the message never quotes stored data."""


class ConflictingRecordingError(ValueError):
    """A trace whose stored copies disagree, or whose id another trace's matches but for
    case, so the console can't tell which recording to show; the trace routes answer the
    same 409. The message names trace ids only."""


def item_content(conn: sqlite3.Connection, item: QueueItem) -> ItemContent | None:
    """Return *item*'s content, read in *conn*; None if the console holds no recording for
    its target.

    Raises:
        ConflictingRecordingError: If a trace's stored copies don't read as one trace.
        UnreadableRecordingError: If the recording or a file it needs can't be read.
    """
    readers: dict[str, Callable[[sqlite3.Connection, QueueItem], ItemContent | None]] = {
        'run': run_content,
        'episode': episode_content,
        'trace': trace_content,
    }
    try:
        return readers[item.target.kind](conn, item)
    except ConflictingRecordingError:
        raise
    except (ValueError, OSError) as exc:  # pydantic's ValidationError is a ValueError
        raise UnreadableRecordingError(
            f"The console can't read the stored {item.target.kind} of item {item.id}"
        ) from exc


def run_content(conn: sqlite3.Connection, item: QueueItem) -> ItemContent | None:
    """Return a run item's content: its solution, and its task when the registry still
    holds the environment it ran."""
    row = conn.execute(
        'SELECT environment, environment_sha256, solution FROM runs WHERE id = ?',
        (item.target.id,),
    ).fetchone()
    if row is None:
        return None
    task = _run_task(row['environment'], row['environment_sha256'])
    return ItemContent(
        item=item, task=task, answer=row['solution'], initial_observation=None, steps=[]
    )


def episode_content(conn: sqlite3.Connection, item: QueueItem) -> ItemContent | None:
    """Return an episode item's content: its replay frames, and the task of the trace it was
    recorded from, if any."""
    row = conn.execute(
        'SELECT frames, trace FROM episodes WHERE id = ?', (item.target.id,)
    ).fetchone()
    if row is None:
        return None
    # Validated here, not read with json_extract, so a corrupt trace is a ValueError.
    task = None if row['trace'] is None else Trace.model_validate_json(row['trace']).task
    frames = [Frame.model_validate(frame) for frame in json.loads(row['frames'])]
    # Frame i shows what action i-1 produced, so its exit code is action i-1's.
    before = [None, *(frame.action for frame in frames)]
    after = [*frames[1:], None]
    steps = [
        ContentStep(
            index=frame.index,
            action=_action(frame.action),
            observation=_output(frame.content, produced_by),
            exit_code=None
            if frame.action is None or then is None
            else _exit_code((then.metadata or {}).get('exit_code')),
            has_screenshot=frame.has_screenshot,
            applied=True,
            error=None,
            messages=[],
        )
        for frame, produced_by, then in zip(frames, before, after)
    ]
    return ItemContent(item=item, task=task, answer=None, initial_observation=None, steps=steps)


def trace_content(conn: sqlite3.Connection, item: QueueItem) -> ItemContent | None:
    """Return a trace item's content, from its stored copies merged as the trace routes
    merge them (``episode_store.select_held_traces``): the one that got furthest."""
    trace_id = item.target.id
    held = select_held_traces(conn, trace_id)
    if conflict := next((c for c in held.conflicts if c.trace_id == trace_id), None):
        raise ConflictingRecordingError(conflict.detail)
    trace = next((trace for trace in held.traces if trace.trace_id == trace_id), None)
    if trace is None:
        # Only a copy stored under this exact id makes it unreadable; one whose id matches
        # but for case is another trace, so this one isn't held (404, as the trace route).
        if _stored_under(conn, held.unreadable, trace_id):
            raise ValueError(f'No stored copy of trace {trace_id} reads as a trace')
        return None
    # A step verifier ran if the fingerprint names one or any step kept its output or reward,
    # so an imported trace without the fingerprint still has its step errors cut.
    verified = trace.provenance.env.step_verifier_sha256 is not None or any(
        step.step_verifier is not None or step.reward is not None for step in trace.steps
    )
    steps = [
        ContentStep(
            index=step.index,
            action=_action(step.action),
            observation=None
            if step.observation is None
            else _output(step.observation.content, step.action),
            exit_code=_exit_code(step.exit_code),
            has_screenshot=False,
            applied=step.applied,
            error=_step_error(step, verified=verified),
            messages=[ContentMessage(role=m.role, content=m.content) for m in step.messages],
        )
        for step in trace.steps
    ]
    first = trace.initial_observation
    return ItemContent(
        item=item, task=trace.task, answer=None,
        initial_observation=None if first is None else first.content, steps=steps,
    )  # fmt: skip


def _action(action: Action | None) -> ContentAction | None:
    """Return *action* with only the fields :class:`ContentAction` lists."""
    if action is None:
        return None
    return ContentAction.model_validate(action.model_dump(include=set(ContentAction.model_fields)))


def _exit_code(code: object) -> int | None:
    """Return an exit code read from observation metadata, if it is an integer: anything
    else there is left out rather than failing the whole item."""
    return code if isinstance(code, int) else None


def _stored_under(conn: sqlite3.Connection, episode_ids: Sequence[str], trace_id: str) -> bool:
    """Return whether one of *episode_ids* stores a trace whose id is exactly *trace_id*."""
    if not episode_ids:
        return False
    marks = ', '.join('?' * len(episode_ids))
    row = conn.execute(
        f'SELECT 1 FROM episodes WHERE {TRACE_KEY} = lower(?) AND {TRACE_ID} = ? '
        f'AND id IN ({marks}) LIMIT 1',
        (trace_id, trace_id, *episode_ids),
    ).fetchone()
    return row is not None


def _step_error(step: TraceStep, *, verified: bool) -> str | None:
    """Return a trace step's error as a reviewer may read it.

    The whole text when nothing graded could be in it: the step wasn't applied (refused
    before anything ran) or the trace has no step verifier. Otherwise only the exception's
    name (``deepgym.traces.capture`` writes ``<name>: <message>``), since the message of an
    error raised while the step verifier ran, a sandbox dying among them, can quote what it
    printed; None when the text names no exception.
    """
    if step.error is None or not step.applied or not verified:
        return step.error
    name = step.error.partition(':')[0]
    return name if name.isidentifier() else None


def _output(text: str, produced_by: Action | None) -> str:
    """Return an observation's *text*, without usage and cost when a CLI action produced it.

    A CLI agent's stdout (Claude Code stream-json, Codex ``--json``) has one JSON event per
    line; each line that is a JSON object or array loses its usage, token count and cost
    keys, at any depth. A line that isn't, or that has none of them, is kept as it is.
    """
    if produced_by is None or produced_by.action_type != 'cli':
        return text
    lines = text.split('\n')
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict | list) and (kept := _without_usage(event)) != event:
            lines[index] = json.dumps(kept, ensure_ascii=False)
    return '\n'.join(lines)


def _without_usage(value: JsonValue) -> JsonValue:
    """Return *value* without usage, token count and cost keys, at any depth."""
    if isinstance(value, dict):
        return {
            key: _without_usage(item)
            for key, item in value.items()
            if key not in _USAGE_KEYS and not key.endswith(_TOKEN_SUFFIXES)
        }
    if isinstance(value, list):
        return [_without_usage(item) for item in value]
    return value


def _run_task(environment: str, sha256: str | None) -> str | None:
    """Return the task of the registry environment a run ran, if it still hashes to
    *sha256*; None when the run kept no hash, or the environment changed, is gone or no
    longer loads or hashes, or picks its task per instance when the run starts.

    The registry builds in code only the benchmark adapters that pick an instance at run
    time (``swebench_pro``, ``terminal_bench_2``): their ``task`` is a generic instruction,
    not the instance's problem the run was graded on, so they have no task here.

    Raises:
        OSError: If a file the environment names can't be read while loading it.
    """
    if sha256 is None:
        return None
    try:
        if environment_dir(environment) is None:
            return None
        env = load_environment(environment)
        same = environment_sha256(env) == sha256
    except ValueError:  # unknown or invalid environment, or one that can't be hashed
        return None
    return env.task if same else None
