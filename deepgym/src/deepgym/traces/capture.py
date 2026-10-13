"""Record a session's trace to disk while the episode runs.

Each trace is an append-only JSONL file of events: ``start`` (provenance and task),
``reset`` (the first observation), one ``step`` per action, ``result`` (the final
verdict) and ``end``. Every event is flushed when it happens, so a crash loses at most
the record being written. :func:`load_trace` rebuilds a :class:`Trace` from whatever
reached the disk; a trace that was never scored is incomplete.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO, TypeVar
from uuid import uuid4

from deepgym.exceptions import DeepGymError, VerifierError
from deepgym.models import Action, MultiTurnEnvironment, Observation, RunResult, is_verdict
from deepgym.session import Session, StepResult
from deepgym.traces.schema import (
    TRACE_SCHEMA_VERSION,
    Provenance,
    Trace,
    TraceMessage,
    TraceStep,
    json_text,
)

_T = TypeVar('_T')

UNFINISHED = 'trace ended before the episode was closed (process crashed or was killed?)'


def traces_dir() -> Path:
    """Directory traces are written to: ``$DEEPGYM_TRACES_DIR``, default ``~/.deepgym/traces``."""
    return Path(os.environ.get('DEEPGYM_TRACES_DIR') or Path.home() / '.deepgym' / 'traces')


class TracedSession:
    """Drive a :class:`Session` and record its trace as each event happens.

    Usage::

        with dg.trace(env, AgentMeta(name='react', model='Qwen/Qwen3-8B')) as tr:
            obs = tr.start()
            while not tr.done:
                action, messages = policy(obs)   # messages: list[TraceMessage]
                obs = tr.step(action, messages).observation
            tr.score()
        tr.trace.has_verdict

    Leaving the block closes the session. An exception, or leaving before ``score()``,
    keeps everything recorded so far and marks the trace incomplete. The trace keeps
    its own copies of actions, observations and results, so changing the objects
    passed in or returned never changes it. Concurrent calls run one at a time.
    """

    def __init__(
        self,
        session: Session,
        env: MultiTurnEnvironment,
        provenance: Provenance,
        path: Path | None = None,
    ) -> None:
        self.trace_id = uuid4().hex
        self.path = path or traces_dir() / f'{self.trace_id}.jsonl'
        self._session = session
        self._env = env
        self._provenance = provenance.model_copy(deep=True)
        self._initial: Observation | None = None
        self._steps: list[TraceStep] = []
        self._result: RunResult | None = None
        self._error: str | None = None
        self._finished_at: datetime | None = None
        self._file: TextIO | None = None
        self._lock = threading.Lock()

    @property
    def done(self) -> bool:
        """Whether the episode has ended."""
        return self._session.done

    @property
    def trace(self) -> Trace:
        """A copy of everything recorded so far."""
        with self._lock:
            return Trace(
                trace_id=self.trace_id,
                task=self._env.task,
                provenance=self._provenance,
                initial_observation=self._initial,
                steps=tuple(self._steps),
                result=self._result,
                error=self._error,
                finished_at=self._finished_at,
            ).model_copy(deep=True)

    def start(self) -> Observation:
        """Start the session and record its first observation.

        The trace file is created first, so even a failed setup leaves a trace.

        Raises:
            DeepGymError: If the trace is closed, or as :meth:`Session.start` (the error
                is recorded first).
            FileExistsError: If the trace file already exists.
        """
        with self._lock:
            self._check_open()
            self._open()
            observation = self._recorded(self._session.start)
            self._initial = observation.model_copy(deep=True)
            self._write({'event': 'reset', 'observation': self._initial})
            return observation

    def step(self, action: Action, messages: Sequence[TraceMessage] = ()) -> StepResult:
        """Apply *action* and record it with the model *messages* that produced it.

        A step that raises is recorded with its error (and its observation, when the
        session applied the action before failing), then the exception propagates.

        Args:
            action: The agent's action.
            messages: Model messages new at this step, ending with the assistant reply
                that produced *action* (see :attr:`TraceStep.messages`).

        Returns:
            The session's step result.

        Raises:
            DeepGymError: If the trace is closed or the episode isn't running (not
                started, ended or already scored: a verdict must match the state the
                trace ends in). Nothing is recorded then.
        """
        with self._lock:
            self._check_open()
            if self._result is not None:
                raise DeepGymError('The episode was already scored; open a new trace to act')
            if self._initial is None or self._session.done:
                raise DeepGymError('The episode is not running; call start() or open a new trace')
            trajectory = self._session.trajectory
            applied_before = len(trajectory.steps)
            scored = bool(self._env.step_verifier_code)
            record = {
                'index': len(self._steps) + 1,
                'action': action.model_copy(deep=True),
                'messages': tuple(messages),
                'timestamp': datetime.now(timezone.utc),
            }
            start = time.perf_counter()
            try:
                result = self._session.step(action)
            except Exception as exc:
                applied = len(trajectory.steps) > applied_before
                # The session records a step once its step verifier scored it, or failed
                # (VerifierError); a later failure, such as a screenshot, keeps the score.
                verified = applied and scored and not isinstance(exc, VerifierError)
                observation = trajectory.final_observation if applied else None
                self._add_step(
                    TraceStep(
                        **record,
                        **(self._step_score(trajectory.step_rewards[-1]) if verified else {}),
                        applied=applied,
                        observation=observation and observation.model_copy(deep=True),
                        error=_describe(exc),
                        duration_ms=_elapsed_ms(start),
                    )
                )
                raise
            self._add_step(
                TraceStep(
                    **record,
                    **(self._step_score(result.reward) if scored else {}),
                    observation=result.observation.model_copy(deep=True),
                    duration_ms=_elapsed_ms(start),
                )
            )
            return result

    def score(self) -> RunResult:
        """Run the final verifier and record its result.

        Raises:
            DeepGymError: If the trace is closed or the episode never started.
            VerifierError: As :meth:`Session.score`; the error is recorded first.
        """
        with self._lock:
            self._check_open()
            if self._initial is None:
                raise DeepGymError('The episode never started; call start() first')
            result = self._recorded(self._session.score)
            self._result = result.model_copy(deep=True)
            self._write({'event': 'result', 'result': self._result})
            return result

    def close(self) -> None:
        """Finish the trace file, if there is one, then close the session. Safe to call twice."""
        with self._lock:
            if self._finished_at is not None:
                return
            self._finished_at = datetime.now(timezone.utc)
            try:
                if self._file is not None:
                    with self._file:
                        finished = self._finished_at.isoformat()
                        self._write({'event': 'end', 'error': self._error, 'finished_at': finished})
            finally:
                self._session.close()

    def __enter__(self) -> TracedSession:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, *_: object
    ) -> None:
        if exc is not None and self._error is None:
            self._error = _describe(exc)
        self.close()

    def _step_score(self, reward: float) -> dict[str, object]:
        """Return the step verifier's output, with *reward* only if that output is a verdict.

        A step verifier that reports its own failure (exit code 2, or an ``error_type``
        that isn't the model's fault) scores 0 in the session, but that 0 judged nothing;
        so does one whose ``error_type`` isn't even a string.
        """
        output = self._session.step_verdict
        error = None if output is None else output.get('error_type')
        judged = (
            output is not None
            and (error is None or isinstance(error, str))
            and is_verdict(output['exit_code'], error)
        )
        return {'reward': reward if judged else None, 'step_verifier': output}

    def _check_open(self) -> None:
        """Refuse to touch the session once the trace is closed."""
        if self._finished_at is not None:
            raise DeepGymError('The trace is closed; open a new one')

    def _recorded(self, call: Callable[[], _T]) -> _T:
        """Return ``call()``, remembering its error for the trace before re-raising it."""
        try:
            return call()
        except Exception as exc:
            self._error = _describe(exc)
            raise

    def _add_step(self, step: TraceStep) -> None:
        self._steps.append(step)
        self._write({'event': 'step', 'step': step})

    def _open(self) -> TextIO:
        """Return the trace file, creating it with its start event on first use.

        The file is created exclusively, so a trace never overwrites another.
        """
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open('x', encoding='utf-8')
            self._write({
                'event': 'start',
                'schema_version': TRACE_SCHEMA_VERSION,
                'trace_id': self.trace_id,
                'task': self._env.task,
                'provenance': self._provenance,
            })  # fmt: skip
        return self._file

    def _write(self, event: dict[str, object]) -> None:
        """Append *event* to the trace file and flush it."""
        file = self._open()
        # ponytail: flush, not fsync: survives a crash or kill, not a power cut.
        file.write(json_text(event) + '\n')
        file.flush()


def load_trace(path: Path) -> Trace:
    """Rebuild a trace from its event file.

    A record cut short by a crash (bytes after the last newline) is dropped; a trace
    that never reached ``end`` gets an error saying so.

    Args:
        path: A file written by :class:`TracedSession`.

    Returns:
        The trace, incomplete if it was never scored.

    Raises:
        DeepGymError: If the file isn't a trace or a complete record is corrupt.
        pydantic.ValidationError: If a record doesn't match the trace schema.
    """
    # Split bytes, not text: a crash can cut the last record inside a UTF-8 character.
    *lines, _partial = path.read_bytes().split(b'\n')
    try:
        events = [json.loads(line) for line in lines]
    except ValueError as exc:  # JSONDecodeError or UnicodeDecodeError
        raise DeepGymError(f'{path}: corrupt trace record: {exc}') from exc
    if not events or not _is_start(events[0]):
        raise DeepGymError(f'{path} is not a deepgym trace file')
    head, *rest = events
    fields = {key: head[key] for key in ('schema_version', 'trace_id', 'task', 'provenance')}
    fields['error'] = UNFINISHED
    steps = []
    for event in rest:
        match event.get('event') if isinstance(event, dict) else None:
            case 'reset':
                fields['initial_observation'] = event['observation']
            case 'step':
                steps.append(event['step'])
            case 'result':
                fields['result'] = event['result']
            case 'end':
                fields['error'] = event['error']
                fields['finished_at'] = event['finished_at']
            case other:
                raise DeepGymError(f'{path}: unknown trace event {other!r}')
    return Trace.model_validate({**fields, 'steps': steps})


def is_trace_file(path: Path) -> bool:
    """Whether *path* begins with a trace's ``start`` event, as no export or other JSONL does."""
    with path.open('rb') as file:
        first = file.readline()
    try:
        return _is_start(json.loads(first))
    except ValueError:
        return False


def _is_start(event: object) -> bool:
    return isinstance(event, dict) and event.get('event') == 'start'


def _describe(exc: BaseException) -> str:
    return f'{type(exc).__name__}: {exc}'


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
