"""Persist recorded episodes for replay; screenshots are stored apart as PNG bytes.

An episode recorded from a :class:`Trace` also keeps the whole trace, for
``deepgym traces export``.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from deepgym.console.db import (
    DEFAULT_DB_PATH,
    connect,
    connect_read_only,
    init_db,
    table_columns,
    utc_now,
    verdict_columns,
)
from deepgym.console.schemas import (
    EpisodeCreate,
    EpisodeDetail,
    EpisodePage,
    EpisodeSummary,
    Frame,
    TraceConflict,
)
from deepgym.models import RunResult, Trajectory
from deepgym.traces import Trace, merge_copies

_SUMMARY_COLUMNS = (
    'id, environment, steps, total_reward, score, passed, has_verdict, error_type, created_at'
)
# The rows whose stored trace's id is the one given ignoring case, so a case collision
# shows; CASE keeps json_extract off rows that aren't JSON, which would fail the query.
_COPIES_OF = (
    'SELECT id, trace, 1 AS valid FROM episodes WHERE CASE WHEN json_valid(trace) '
    "THEN lower(json_extract(trace, '$.trace_id')) END = lower(?) ORDER BY rowid"
)


class HeldTraces(BaseModel):
    """The traces the console holds, one copy each, and the stored traces that don't read
    as one."""

    model_config = ConfigDict(frozen=True)

    traces: tuple[Trace, ...]
    """One copy of each trace, the one that got furthest
    (:func:`~deepgym.traces.merge_copies`), in the order first stored."""
    episodes: dict[str, tuple[str, ...]]
    """The episodes holding each kept trace's copies, by trace id."""
    untraced: int
    """Episodes recorded without a trace."""
    conflicts: tuple[TraceConflict, ...]
    unreadable: tuple[str, ...]
    """Episodes whose stored trace doesn't read as a trace."""

    @property
    def merged(self) -> int:
        """Copies of kept traces merged into the copy kept."""
        return sum(len(ids) - 1 for ids in self.episodes.values())

    @property
    def conflicting(self) -> int:
        """Copies of the conflicting traces."""
        return sum(len(conflict.episodes) for conflict in self.conflicts)


class EpisodeStore:
    """Episodes and screenshots tables of the console database."""

    def __init__(self, path: Path) -> None:
        self._path = path
        init_db(path)

    def add(self, episode: EpisodeCreate, trace: Trace | None = None) -> EpisodeSummary:
        """Store an episode, and the *trace* it came from if any.

        Its screenshots are decoded and kept per frame.

        Raises:
            binascii.Error: If an observation's screenshot isn't valid base64.
        """
        frames, screenshots = split_frames(episode.trajectory)
        result = episode.result
        score, passed, error_type = (
            (None, None, None) if result is None else verdict_columns(result)
        )
        summary = EpisodeSummary(
            id=uuid4().hex[:12],
            environment=episode.environment,
            steps=len(episode.trajectory.steps),
            total_reward=episode.trajectory.total_reward,
            score=score,
            passed=passed,
            has_verdict=None if result is None else result.has_verdict,
            error_type=error_type,
            created_at=utc_now(),
        )
        with connect(self._path) as conn:
            conn.execute(
                'INSERT INTO episodes (id, environment, steps, total_reward, score, passed, '
                'has_verdict, error_type, frames, result, created_at, trace) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (summary.id, summary.environment, summary.steps, summary.total_reward,
                 summary.score, summary.passed, summary.has_verdict, summary.error_type,
                 json.dumps([frame.model_dump(mode='json') for frame in frames]),
                 None if result is None else result.model_dump_json(), summary.created_at,
                 None if trace is None else trace.model_dump_json()),
            )  # fmt: skip
            conn.executemany(
                'INSERT INTO screenshots (episode_id, frame, png) VALUES (?, ?, ?)',
                [(summary.id, index, png) for index, png in screenshots.items()],
            )
        return summary

    def page(self, environment: str | None, offset: int, limit: int | None) -> EpisodePage:
        """Return one page of the episodes, newest first, and how many there are in all.

        Args:
            environment: Only this environment's episodes, if given.
            offset: How many of the newest episodes to skip.
            limit: Most episodes to return; None returns every one after *offset*.

        Returns:
            The page, with the total count of matching episodes.
        """
        where, params = ('WHERE environment = ?', (environment,)) if environment else ('', ())
        with connect(self._path) as conn:
            total = conn.execute(f'SELECT COUNT(*) FROM episodes {where}', params).fetchone()[0]
            rows = conn.execute(
                f'SELECT {_SUMMARY_COLUMNS} FROM episodes {where} '
                'ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?',
                (*params, -1 if limit is None else limit, offset),  # SQLite: LIMIT -1 is none
            ).fetchall()
        episodes = [EpisodeSummary.model_validate(dict(row)) for row in rows]
        return EpisodePage(episodes=episodes, total=total)

    def traces(self) -> tuple[list[Trace], int]:
        """Return the stored traces, and how many episodes were recorded without one."""
        return read_traces(self._path)

    def held_traces(self, trace_id: str | None = None) -> HeldTraces:
        """Return :func:`read_held_traces` of this database."""
        return read_held_traces(self._path, trace_id)

    def get(self, episode_id: str) -> EpisodeDetail | None:
        """Return one episode with its frames and result, or None if unknown."""
        with connect(self._path) as conn:
            row = conn.execute(
                f'SELECT {_SUMMARY_COLUMNS}, frames, result FROM episodes WHERE id = ?',
                (episode_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data['frames'] = json.loads(data['frames'])
        data['result'] = json.loads(data['result']) if data['result'] else None
        return EpisodeDetail.model_validate(data)

    def screenshot(self, episode_id: str, frame: int) -> bytes | None:
        """Return the PNG shown at *frame*, or None if that frame had no screenshot."""
        with connect(self._path) as conn:
            row = conn.execute(
                'SELECT png FROM screenshots WHERE episode_id = ? AND frame = ?',
                (episode_id, frame),
            ).fetchone()
        return None if row is None else row['png']


def read_traces(path: Path) -> tuple[list[Trace], int]:
    """Return the traces stored in the console database at *path*, in the order stored,
    and how many episodes were recorded without one.

    *path* is opened read-only, so it is never created or migrated; in a database from
    before episodes kept their trace, every episode is one without.
    """
    with connect_read_only(path) as conn:
        columns = table_columns(conn, 'episodes')
        if not columns:
            return [], 0
        if 'trace' not in columns:
            return [], conn.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
        rows = conn.execute('SELECT trace FROM episodes ORDER BY rowid').fetchall()
    traces = [Trace.model_validate_json(row['trace']) for row in rows if row['trace']]
    return traces, len(rows) - len(traces)


def read_held_traces(path: Path, trace_id: str | None = None) -> HeldTraces:
    """Return the traces stored in the console database at *path*, one copy of each, and
    what doesn't read as one trace: copies that disagree, ids that differ only in case
    (both left out as conflicts), and stored traces that don't parse (left out as
    unreadable). Nothing raises for one bad trace.

    *path* is opened read-only, so it is never created or migrated.

    Args:
        path: The console database.
        trace_id: Read only the copies of this trace and of ids differing only in case
            (their ``untraced`` is 0); None reads every stored trace.
    """
    with connect_read_only(path) as conn:
        columns = table_columns(conn, 'episodes')
        if 'trace' not in columns:
            rows = (
                conn.execute('SELECT id, NULL AS trace FROM episodes').fetchall() if columns else []
            )
        elif trace_id is None:
            rows = conn.execute(
                'SELECT id, trace, json_valid(trace) AS valid FROM episodes ORDER BY rowid'
            ).fetchall()
        else:
            rows = conn.execute(_COPIES_OF, (trace_id,)).fetchall()
    stored = [row for row in rows if row['trace'] is not None]
    return _held(stored, untraced=len(rows) - len(stored))


def _held(rows: Sequence[sqlite3.Row], *, untraced: int) -> HeldTraces:
    """Parse each row's trace, merge each trace's copies, and set aside what won't.

    A trace that SQLite's ``json_valid`` refuses (such as one holding ``NaN``) is unreadable
    even where pydantic would parse it, so the one-trace query, which can only select valid
    JSON, agrees with reading them all.
    """
    copies: dict[str, list[tuple[str, Trace]]] = {}
    unreadable: list[str] = []
    for row in rows:
        trace = _parsed(row['trace']) if row['valid'] else None
        if trace is None:
            unreadable.append(row['id'])
            continue
        copies.setdefault(trace.trace_id, []).append((row['id'], trace))
    kept: dict[str, Trace] = {}
    conflicts: list[TraceConflict] = []
    for trace_id, held in copies.items():
        episodes = tuple(episode_id for episode_id, _ in held)
        try:
            (kept[trace_id],) = merge_copies([trace for _, trace in held])
        except ValueError as exc:
            conflicts.append(
                TraceConflict(
                    trace_id=trace_id, reason='copies_disagree', episodes=episodes, detail=str(exc)
                )
            )
    folded: dict[str, list[str]] = {}
    for trace_id in kept:
        folded.setdefault(trace_id.casefold(), []).append(trace_id)
    for ids in (ids for ids in folded.values() if len(ids) > 1):
        detail = f'Trace ids {ids} differ only in case, which exports cannot tell apart'
        for trace_id in ids:
            del kept[trace_id]
            episodes = tuple(episode_id for episode_id, _ in copies[trace_id])
            conflicts.append(
                TraceConflict(
                    trace_id=trace_id, reason='case_collision', episodes=episodes, detail=detail
                )
            )
    return HeldTraces(
        traces=tuple(kept.values()),
        episodes={
            trace_id: tuple(episode_id for episode_id, _ in copies[trace_id]) for trace_id in kept
        },
        untraced=untraced,
        conflicts=tuple(conflicts),
        unreadable=tuple(unreadable),
    )


def _parsed(text: str) -> Trace | None:
    """Return the trace *text* holds, or None if it doesn't read as one."""
    try:
        return Trace.model_validate_json(text)
    except ValidationError:
        return None


def read_trace_ids(path: Path) -> dict[str, str]:
    """Return the trace id of each episode recorded from a trace, by episode id.

    *path* is opened read-only, so it is never created or migrated; a database from before
    episodes kept their trace gives none.
    """
    with connect_read_only(path) as conn:
        if 'trace' not in table_columns(conn, 'episodes'):
            return {}
        rows = conn.execute('SELECT id, trace FROM episodes WHERE trace IS NOT NULL').fetchall()
    return {row['id']: json.loads(row['trace'])['trace_id'] for row in rows}


def split_frames(trajectory: Trajectory) -> tuple[list[Frame], dict[int, bytes]]:
    """Turn a trajectory into replay frames plus each frame's decoded screenshot.

    Frame *i* is the observation the agent acted on at step *i*, that action and its
    reward; the final observation, if any, is one more frame with no action.

    Raises:
        binascii.Error: If a screenshot isn't valid base64.
    """
    shown = [*trajectory.steps]
    if trajectory.final_observation is not None:
        shown.append((trajectory.final_observation, None))
    frames: list[Frame] = []
    screenshots: dict[int, bytes] = {}
    for index, (observation, action) in enumerate(shown):
        if observation.screenshot:
            screenshots[index] = base64.b64decode(observation.screenshot, validate=True)
        has_reward = action is not None and index < len(trajectory.step_rewards)
        frames.append(
            Frame(
                index=index,
                content=observation.content,
                done=observation.done,
                metadata=observation.metadata,
                action=action,
                reward=trajectory.step_rewards[index] if has_reward else None,
                has_screenshot=bool(observation.screenshot),
            )
        )
    return frames, screenshots


def record_episode(
    trajectory: Trajectory | Trace,
    result: RunResult | None = None,
    *,
    environment: str = '',
    db_path: Path | None = None,
) -> str:
    """Store an episode so ``deepgym console`` can replay it.

    Args:
        trajectory: The episode's steps, e.g. from ``dg.run_episode()``, or a whole
            :class:`Trace` (e.g. ``tr.trace``), which is also kept for export.
        result: The final verifier result, if the episode was scored. A trace
            carries its own.
        environment: Environment name shown in the console; defaults to the trace's.
        db_path: Console database; defaults to ``~/.deepgym/console.db``.

    Returns:
        The new episode's id.

    Raises:
        ValueError: If *result* is given with a trace.

    Example:
        >>> trajectory, result = dg.run_episode(load_environment('todo_list'), agent)
        >>> record_episode(trajectory, result, environment='todo_list')
    """
    store = EpisodeStore(db_path or DEFAULT_DB_PATH)
    if not isinstance(trajectory, Trace):
        episode = EpisodeCreate(environment=environment, trajectory=trajectory, result=result)
        return store.add(episode).id
    if result is not None:
        raise ValueError('A trace carries its own result; pass the trace alone')
    episode = EpisodeCreate(
        environment=environment or trajectory.provenance.env.name or '',
        trajectory=trajectory.to_trajectory(),
        result=trajectory.result,
    )
    return store.add(episode, trajectory).id
