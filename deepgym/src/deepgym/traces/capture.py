"""Record a session's trace to disk while the episode runs.

Each trace is an append-only JSONL file of events: ``start`` (provenance and task),
``reset`` (the first observation), one ``step`` per action, ``usage`` for usage on no
step and ``usage_incomplete`` once a model call reported none, one ``result`` per grade
(the last is the verdict) and ``end``. Every event is flushed when it happens, so a crash
loses at most the record being written. :func:`load_trace` rebuilds a :class:`Trace` from
whatever reached the disk; a trace that was never scored is incomplete.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TextIO, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from deepgym.cli_runtime import CliExecution, CliExecutionError, CliRequest, EventObserver
from deepgym.exceptions import DeepGymError, SandboxError, SessionClosedError
from deepgym.models import (
    Action,
    MultiTurnEnvironment,
    Observation,
    RunResult,
    Trajectory,
    is_verdict,
)
from deepgym.session import Session, StepResult
from deepgym.tools import HttpTool
from deepgym.traces.schema import (
    TRACE_SCHEMA_VERSION,
    Provenance,
    Trace,
    TraceMessage,
    TraceStep,
    Usage,
    Usd,
    json_text,
    sum_usage,
)

_T = TypeVar('_T')

logger = logging.getLogger(__name__)

UNFINISHED = 'trace ended before the episode was closed (process crashed or was killed?)'


class _UsageEvent(BaseModel):
    """A ``usage`` event: model calls that led to no action, and what they cost."""

    model_config = ConfigDict(frozen=True)
    usage: Usage
    cost_usd: Usd | None


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

    When one reply issues several actions, step each one, passing the reply's messages
    and ``usage=`` with the first only: the others continue its turn (see
    :attr:`Trace.turns`), and the reply's tokens are counted once.

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
        self._results: list[RunResult] = []
        self._unattached: list[tuple[Usage, float | None]] = []
        self._usage_complete = True
        self._error: str | None = None
        self._finished_at: datetime | None = None
        self._file: TextIO | None = None
        self._session_closed = False
        self._lock = threading.Lock()

    @property
    def step_count(self) -> int:
        """How many steps the trace has recorded."""
        with self._lock:
            return len(self._steps)

    @property
    def done(self) -> bool:
        """Whether the episode has ended."""
        return self._session.done

    @property
    def tools(self) -> tuple[HttpTool, ...]:
        """The API tools the environment declares, for tool-calling agents."""
        return self._session.tools

    @property
    def trajectory(self) -> Trajectory:
        """Return a copy of the session trajectory for an agent's result."""
        with self._lock:
            return self._session.trajectory.model_copy(deep=True)

    @property
    def trace(self) -> Trace:
        """A copy of everything recorded so far."""
        with self._lock:
            usage, cost_usd = sum_usage(self._unattached)
            return Trace(
                trace_id=self.trace_id,
                task=self._env.task,
                provenance=self._provenance,
                initial_observation=self._initial,
                steps=tuple(self._steps),
                result=self._results[-1] if self._results else None,
                prior_results=tuple(self._results[:-1]),
                error=self._error,
                unattached_usage=usage,
                unattached_cost_usd=cost_usd,
                usage_complete=self._usage_complete,
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

    def step(
        self,
        action: Action,
        messages: Sequence[TraceMessage] = (),
        *,
        usage: Usage | None = None,
        cost_usd: float | None = None,
        timeout: float | None = None,
    ) -> StepResult:
        """Apply *action* and record it with the model *messages* that produced it.

        A step that raises is recorded with its error (and its observation, when the
        session applied the action before failing), then the exception propagates.

        Args:
            action: The agent's action.
            messages: Model messages new at this step, ending with the assistant reply
                that produced *action* (see :attr:`TraceStep.messages`).
            usage: Provider-reported tokens of the model calls since the previous step;
                pass a reply's usage with its first action only (see :attr:`TraceStep.usage`).
            cost_usd: What *usage* cost from your rates (the agents use ``TokenPrices``);
                None without rates.
            timeout: Optional action deadline forwarded to the session.

        Returns:
            The session's step result.

        Raises:
            DeepGymError: If the trace is closed or the episode isn't running (not
                started, ended or already scored: a verdict must match the state the
                trace ends in). Nothing is recorded then.
            pydantic.ValidationError: If *messages*, *usage* or *cost_usd* can't be
                recorded (a cost without usage, say); the action is not applied.
        """
        return self._capture(
            action,
            messages,
            lambda: self._session.step(action, timeout=timeout),
            usage=usage,
            cost_usd=cost_usd,
        )

    def run_cli(self, request: CliRequest, on_event: EventObserver) -> CliExecution:
        """Run and capture one CLI launch, with its events nested under that action.

        Args:
            request: The sandbox CLI and explicit credential-exposure opt-in.
            on_event: The agent's budget observer; never replays parsed commands.

        Returns:
            The supervised process result after its descendants have stopped.

        Raises:
            DeepGymError: If the trace cannot accept actions or CLI execution fails.
        """
        return self._capture(
            Action(action_type='cli', content=shlex.join(request.argv)),
            (),
            lambda: self._session.run_cli(request, on_event),
        )

    def record_unattached_usage(self, usage: Usage, cost_usd: float | None = None) -> None:
        """Record usage that belongs to no step; the trace's totals include it.

        The agents in :mod:`deepgym.agents` record it when they finish: an API agent's
        responses after its last action (a closing answer, a refusal, a reply cut off, one
        after which a budget ran out), and a CLI agent's whole run, whose one step is
        recorded before the CLI reports its usage. Calls add up in
        :attr:`Trace.unattached_usage`.

        Args:
            usage: Provider-reported tokens.
            cost_usd: What *usage* cost: from your rates (the agents use ``TokenPrices``),
                or a CLI's own reported total (Claude Code's ``total_cost_usd``); None
                without either.

        Raises:
            SessionClosedError: If the trace is closed.
            pydantic.ValidationError: If *cost_usd* isn't a finite amount of at least 0;
                nothing is recorded then.
        """
        with self._lock:
            self._check_open()
            event = _UsageEvent(usage=usage, cost_usd=cost_usd)
            self._unattached.append((event.usage, event.cost_usd))
            self._write({'event': 'usage', **event.model_dump()})

    def mark_usage_incomplete(self) -> None:
        """Record that some model call reported no usage, or only part of it.

        The trace's token totals are then a lower bound and its cost total None (see
        :attr:`Trace.usage_complete`).

        Raises:
            SessionClosedError: If the trace is closed.
        """
        with self._lock:
            self._check_open()
            self._usage_complete = False
            self._write({'event': 'usage_incomplete'})

    def abandon(self, error_type: Literal['provider_error', 'setup_error'], detail: str) -> None:
        """End the episode with no verdict, as :meth:`Session.abandon`; :meth:`score` records it."""
        with self._lock:
            self._check_open()
            self._session.abandon(error_type, detail)

    def cut_short(self) -> None:
        """Mark the episode as cut short, as :meth:`Session.cut_short`."""
        with self._lock:
            self._check_open()
            self._session.cut_short()

    def finalize_cli(self) -> None:
        """Finalize CLI containment without inventing an action in the trace.

        Raises:
            DeepGymError: If containment cleanup fails; the trace records the error.
        """
        with self._lock:
            self._check_open()
            self._recorded(self._session.finalize_cli)

    def _capture(
        self,
        action: Action,
        messages: Sequence[TraceMessage],
        call: Callable[[], _T],
        *,
        usage: Usage | None = None,
        cost_usd: float | None = None,
    ) -> _T:
        """Record the one real Session action produced by *call*, even on failure."""
        with self._lock:
            self._check_open()
            if self._results:
                raise DeepGymError('The episode was already scored; open a new trace to act')
            if self._initial is None or self._session.done:
                raise DeepGymError('The episode is not running; call start() or open a new trace')
            trajectory = self._session.trajectory
            applied_before = len(trajectory.steps)
            record = {
                'index': len(self._steps) + 1,
                'action': action.model_copy(deep=True),
                'messages': tuple(messages),
                'timestamp': datetime.now(timezone.utc),
                'usage': usage,
                'cost_usd': cost_usd,
            }
            # Refuse a step that can't be recorded before acting: an action taken is recorded.
            TraceStep(**record, duration_ms=0)
            start = time.perf_counter()
            try:
                result = call()
            except BaseException as exc:
                if isinstance(exc, CliExecutionError):
                    self._cli_version(exc.cli_version)
                # Only the session knows whether a failed operation changed the sandbox.
                applied = len(trajectory.steps) > applied_before
                observation = trajectory.final_observation if applied else None
                if applied:
                    record['action'] = trajectory.steps[-1][1].model_copy(deep=True)
                self._add_step(
                    TraceStep(
                        **record,
                        applied=applied,
                        observation=observation and observation.model_copy(deep=True),
                        error=_describe(exc),
                        duration_ms=_elapsed_ms(start),
                    )
                )
                raise
            record['action'] = trajectory.steps[-1][1].model_copy(deep=True)
            if isinstance(result, CliExecution) and result.cli_version is not None:
                self._cli_version(result.cli_version)
            observation = (
                result.observation
                if isinstance(result, StepResult)
                else trajectory.final_observation
            )
            assert observation is not None
            self._add_step(
                TraceStep(
                    **record,
                    **(
                        self._step_score(trajectory.step_rewards[-1])
                        if self._env.step_verifier_code
                        else {}
                    ),
                    observation=observation.model_copy(deep=True),
                    duration_ms=_elapsed_ms(start),
                )
            )
            return result

    def _cli_version(self, version: str | None) -> None:
        if version is not None:
            self._provenance = self._provenance.model_copy(
                update={'agent': self._provenance.agent.model_copy(update={'version': version})}
            )
            self._write({'event': 'agent_version', 'version': version})

    def score(self) -> RunResult:
        """Run the final verifier and record its result.

        Calling it again grades the same final state again. Every result is kept: the
        last is the trace's verdict, the others its ``prior_results``.

        Raises:
            DeepGymError: If the trace is closed or the episode never started.
            VerifierError: As :meth:`Session.score`; the error is recorded first.
        """
        with self._lock:
            self._check_open()
            if self._initial is None:
                raise DeepGymError('The episode never started; call start() first')
            result = self._recorded(self._session.score)
            self._results.append(result.model_copy(deep=True))
            self._write({'event': 'result', 'result': self._results[-1]})
            return result

    def close(self) -> None:
        """Finish the trace file, if there is one, then close the session. Safe to call twice;
        after a failed sandbox delete, the next call tries the delete again."""
        with self._lock:
            try:
                if self._finished_at is None:
                    self._finished_at = datetime.now(timezone.utc)
                    if self._file is not None:
                        with self._file:
                            finished = self._finished_at.isoformat()
                            event = {'event': 'end', 'error': self._error, 'finished_at': finished}
                            self._write(event)
            finally:
                if not self._session_closed:
                    self._session.close()
                    self._session_closed = True

    def __enter__(self) -> TracedSession:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, *_: object
    ) -> None:
        """Close the trace and session; a failed sandbox delete is logged, never raised over
        the body's outcome (an explicit close() raises it)."""
        if exc is not None and self._error is None:
            self._error = _describe(exc)
        try:
            self.close()
        except SandboxError:
            logger.warning(
                'Could not delete sandbox %s; it may still exist',
                self._session.sandbox_id,
                exc_info=True,
            )

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
            raise SessionClosedError('The trace is closed; open a new one')

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
    results = []
    unattached = []
    for event in rest:
        match event.get('event') if isinstance(event, dict) else None:
            case 'reset':
                fields['initial_observation'] = event['observation']
            case 'step':
                steps.append(event['step'])
            case 'agent_version':
                fields['provenance']['agent']['version'] = event['version']
            case 'result':
                results.append(event['result'])
            case 'usage':
                record = _UsageEvent.model_validate(event)
                unattached.append((record.usage, record.cost_usd))
            case 'usage_incomplete':
                fields['usage_complete'] = False
            case 'end':
                fields['error'] = event['error']
                fields['finished_at'] = event['finished_at']
            case other:
                raise DeepGymError(f'{path}: unknown trace event {other!r}')
    if results:
        fields['result'], fields['prior_results'] = results[-1], results[:-1]
    fields['unattached_usage'], fields['unattached_cost_usd'] = sum_usage(unattached)
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
