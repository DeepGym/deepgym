"""Persist recorded episodes for replay; screenshots are stored apart as PNG bytes.

An episode recorded from a :class:`Trace` also keeps the whole trace, for
``deepgym traces export``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from uuid import uuid4

from deepgym.console.db import DEFAULT_DB_PATH, connect, init_db, utc_now
from deepgym.console.schemas import EpisodeCreate, EpisodeDetail, EpisodeSummary, Frame
from deepgym.models import RunResult, Trajectory
from deepgym.traces import Trace

_SUMMARY_COLUMNS = 'id, environment, steps, total_reward, score, passed, created_at'


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
        summary = EpisodeSummary(
            id=uuid4().hex[:12],
            environment=episode.environment,
            steps=len(episode.trajectory.steps),
            total_reward=episode.trajectory.total_reward,
            score=None if result is None else result.score,
            passed=None if result is None else result.passed,
            created_at=utc_now(),
        )
        with connect(self._path) as conn:
            conn.execute(
                'INSERT INTO episodes (id, environment, steps, total_reward, score, passed, '
                'frames, result, created_at, trace) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (summary.id, summary.environment, summary.steps, summary.total_reward,
                 summary.score, summary.passed,
                 json.dumps([frame.model_dump(mode='json') for frame in frames]),
                 None if result is None else result.model_dump_json(), summary.created_at,
                 None if trace is None else trace.model_dump_json()),
            )  # fmt: skip
            conn.executemany(
                'INSERT INTO screenshots (episode_id, frame, png) VALUES (?, ?, ?)',
                [(summary.id, index, png) for index, png in screenshots.items()],
            )
        return summary

    def list(self, environment: str | None = None, limit: int = 200) -> list[EpisodeSummary]:
        """Return the newest episodes, optionally only those of *environment*."""
        where, params = ('WHERE environment = ?', (environment,)) if environment else ('', ())
        with connect(self._path) as conn:
            rows = conn.execute(
                f'SELECT {_SUMMARY_COLUMNS} FROM episodes {where} '
                'ORDER BY created_at DESC, rowid DESC LIMIT ?',
                (*params, limit),
            ).fetchall()
        return [EpisodeSummary.model_validate(dict(row)) for row in rows]

    def traces(self) -> tuple[list[Trace], int]:
        """Return the stored traces, and how many episodes were recorded without one."""
        with connect(self._path) as conn:
            rows = conn.execute('SELECT trace FROM episodes ORDER BY rowid').fetchall()
        traces = [Trace.model_validate_json(row['trace']) for row in rows if row['trace']]
        return traces, len(rows) - len(traces)

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
