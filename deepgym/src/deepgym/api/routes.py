"""API route definitions for DeepGym."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from functools import partial
from typing import Annotated

try:
    from fastapi import APIRouter, Depends, HTTPException, Request, status
except ImportError as _exc:
    raise ImportError(
        'FastAPI is required for the server. Install with: pip install "deepgym[server]"'
    ) from _exc
from pydantic_core import PydanticSerializationError

from deepgym.api.deps import ServerDeepGym, get_deepgym
from deepgym.api.schemas import (
    APIEnvironment,
    BatchJobResponse,
    BatchRunRequest,
    BenchmarkAuditReport,
    BenchmarkAuditRequest,
    CapabilitiesResponse,
    CreateEnvironmentResponse,
    CreateSessionRequest,
    CreateSnapshotRequest,
    EvalRequest,
    HealthResponse,
    JobResponse,
    RegisteredRunRequest,
    RegisteredVerifierAuditRequest,
    RunRequest,
    SessionResponse,
    StepResponse,
    VerifierAuditReport,
    VerifierAuditRequest,
)
from deepgym.benchmark_ops import build_benchmark_audit
from deepgym.core import DeepGym
from deepgym.exceptions import DeepGymError, SandboxError, VerifierError
from deepgym.models import (
    Action,
    BatchJob,
    BatchResult,
    Environment,
    EvalResult,
    Job,
    JobStatus,
    MultiTurnEnvironment,
    RunResult,
)
from deepgym.registry import load_environment
from deepgym.reward_qa import RewardAuditor
from deepgym.session import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Health router
# ---------------------------------------------------------------------------

health_router = APIRouter(tags=['health'])


@health_router.get('/health', response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return service health status, version, and how many sandboxes cleanup gave up on."""
    return HealthResponse(leaked_sandboxes=len(_leaked_sandboxes))


# ---------------------------------------------------------------------------
# V1 router
# ---------------------------------------------------------------------------

v1_router = APIRouter(prefix='/v1', tags=['v1'])

# In-memory store capacity limits.
_MAX_ENVIRONMENTS = 1000
_MAX_JOBS = 10_000  # queued or running at once
_MAX_JOB_BYTES = 256 * 1024 * 1024  # request bodies of the jobs queued or running
_MAX_FINISHED_JOBS = 10_000  # kept for polling; the oldest finished job goes first
_MAX_FINISHED_BYTES = 256 * 1024 * 1024  # finished jobs kept, as JSON; the newest always stays

# In-memory environment store (swap for a real DB later).
_environments: dict[str, Environment] = {}


@v1_router.post('/run', response_model=RunResult)
def run_episode(
    body: RunRequest,
    dg: Annotated[DeepGym, Depends(get_deepgym)],
) -> RunResult:
    """Run a single model output against an environment's verifier.

    The solution is executed inside an isolated Daytona sandbox and scored
    by the environment's verifier script.
    """
    return _run_environment(dg, body.environment.to_environment(), body.model_output)


@v1_router.get('/capabilities', response_model=CapabilitiesResponse)
def get_capabilities() -> CapabilitiesResponse:
    """Return the server features exposed for remote runtimes."""
    return CapabilitiesResponse()


@v1_router.post('/environments/{env_id}/run', response_model=RunResult)
def run_registered_environment(
    env_id: str,
    body: RegisteredRunRequest,
    dg: Annotated[DeepGym, Depends(get_deepgym)],
) -> RunResult:
    """Run a model output against a previously registered environment."""
    env = _get_registered_environment(env_id)
    return _run_environment(dg, env, body.model_output)


@contextmanager
def _http_errors(
    fallback_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> Iterator[None]:
    """Translate DeepGym errors into HTTP exceptions."""
    try:
        yield
    except VerifierError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Verifier error: {exc}',
        ) from exc
    except SandboxError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f'Sandbox error: {exc}',
        ) from exc
    except DeepGymError as exc:
        raise HTTPException(status_code=fallback_status, detail=f'DeepGym error: {exc}') from exc


def _run_environment(dg: DeepGym, environment: Environment, model_output: str) -> RunResult:
    """Execute a run and normalize backend errors into HTTP exceptions."""
    with _http_errors():
        return dg.run(environment, model_output)


@v1_router.post('/audit/verifier', response_model=VerifierAuditReport)
def audit_verifier(
    body: VerifierAuditRequest,
    dg: Annotated[DeepGym, Depends(get_deepgym)],
) -> VerifierAuditReport:
    """Audit an inline environment verifier for reward hacking risk."""
    auditor = RewardAuditor(dg)
    return auditor.audit(
        body.environment.to_environment(),
        verifier_id=body.verifier_id or '',
        benchmark=body.benchmark,
        strategies=body.strategies,
        persist=body.persist,
    )


@v1_router.post('/environments/{env_id}/audit', response_model=VerifierAuditReport)
def audit_registered_environment(
    env_id: str,
    body: RegisteredVerifierAuditRequest,
    dg: Annotated[DeepGym, Depends(get_deepgym)],
) -> VerifierAuditReport:
    """Audit a registered environment verifier."""
    env = _get_registered_environment(env_id)
    auditor = RewardAuditor(dg)
    return auditor.audit(
        env,
        verifier_id=body.verifier_id or env_id,
        benchmark=body.benchmark,
        strategies=body.strategies,
        persist=body.persist,
    )


@v1_router.post('/benchmarks/audit', response_model=BenchmarkAuditReport)
def audit_registered_benchmark(body: BenchmarkAuditRequest) -> BenchmarkAuditReport:
    """Audit split leakage across a set of registered environments."""
    missing = sorted(env_id for env_id in body.environment_ids if env_id not in _environments)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Unknown environment ids: {missing}',
        )

    environments = {env_id: _environments[env_id] for env_id in body.environment_ids}
    return build_benchmark_audit(
        environments,
        benchmark=body.benchmark,
        provenance='registered',
        seed=body.seed,
        public_eval_ratio=body.public_eval_ratio,
        holdout_ratio=body.holdout_ratio,
        canary_ratio=body.canary_ratio,
        split_overrides=body.split_overrides,
    )


@v1_router.post('/run/batch', response_model=BatchResult)
def run_batch(
    body: BatchRunRequest,
    dg: Annotated[DeepGym, Depends(get_deepgym)],
) -> BatchResult:
    """Run multiple model outputs against an environment and aggregate results.

    Solutions are executed in parallel up to *max_parallel* concurrency.
    """
    try:
        return dg.run_batch(
            body.environment.to_environment(), body.outputs, max_parallel=body.max_parallel
        )
    except VerifierError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Verifier error: {exc}',
        ) from exc
    except SandboxError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f'Sandbox error: {exc}',
        ) from exc
    except DeepGymError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'DeepGym error: {exc}',
        ) from exc


@v1_router.post('/eval', response_model=EvalResult)
def run_eval(
    body: EvalRequest,
    dg: Annotated[DeepGym, Depends(get_deepgym)],
) -> EvalResult:
    """Run an evaluation suite and return aggregated metrics.

    Each task in the suite is scored independently. Results include per-task
    scores and an overall pass rate.
    """
    try:
        return dg.eval(body.suite, body.model_outputs, max_parallel=body.max_parallel)
    except VerifierError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Verifier error: {exc}',
        ) from exc
    except SandboxError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f'Sandbox error: {exc}',
        ) from exc
    except DeepGymError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'DeepGym error: {exc}',
        ) from exc


@v1_router.post(
    '/environments', response_model=CreateEnvironmentResponse, status_code=status.HTTP_201_CREATED
)
def create_environment(body: APIEnvironment) -> CreateEnvironmentResponse:
    """Create and register a new environment.

    The environment is stored in memory and assigned a unique identifier. It takes the
    same fields as an inline environment, so runs against it score exactly like inline runs.

    Raises:
        HTTPException: 429 if the in-memory store has reached capacity.
    """
    if len(_environments) >= _MAX_ENVIRONMENTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail='Environment limit reached',
        )
    env_id = uuid.uuid4().hex[:12]
    _environments[env_id] = body.to_environment()
    return CreateEnvironmentResponse(id=env_id, created=True)


def _env_view(env_id: str, env: Environment) -> dict:
    """Return a registered environment with its id and env_vars removed (they may hold secrets).

    Args:
        env_id: The identifier the environment was registered under.
        env: The environment model to serialize.

    Returns:
        Dict representation with ``id`` added and env_vars set to None.
    """
    return {'id': env_id, **env.model_dump(), 'env_vars': None}


def _get_registered_environment(env_id: str) -> Environment:
    """Resolve a stored environment or raise 404."""
    env = _environments.get(env_id)
    if env is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment '{env_id}' not found.",
        )
    return env


@v1_router.get('/environments')
def list_environments() -> list[dict]:
    """List all registered environments with their ids (env_vars stripped)."""
    return [_env_view(env_id, env) for env_id, env in _environments.items()]


@v1_router.get('/environments/{env_id}')
def get_environment(env_id: str) -> dict:
    """Retrieve a specific environment by its identifier (env_vars stripped)."""
    return _env_view(env_id, _get_registered_environment(env_id))


@v1_router.post('/snapshots', status_code=status.HTTP_501_NOT_IMPLEMENTED)
def create_snapshot(_body: CreateSnapshotRequest) -> None:
    """Create a pre-built sandbox snapshot with dependencies pre-installed.

    Not yet implemented. Returns 501.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail='Snapshot creation not yet implemented',
    )


# ---------------------------------------------------------------------------
# Live multi-turn sessions
# ---------------------------------------------------------------------------
# ponytail: in-memory store, single process. Move to a shared store if the API scales out.

_MAX_SESSIONS = 100
_SESSION_IDLE_TTL_S = 30 * 60
SESSION_SWEEP_INTERVAL_S = 60
_sessions: dict[str, Session] = {}
_last_used: dict[str, float] = {}
# Sessions taken out of use but not closed yet: closing now, or after a close that failed.
_closing: dict[str, Session] = {}
# Failed closes so far of each session in _closing; at _MAX_CLOSE_ATTEMPTS it is given up on.
_close_failures: dict[str, int] = {}
_MAX_CLOSE_ATTEMPTS = 10
# Sandbox ids of the sessions given up on (the newest 1000); they hold no slot.
_leaked_sandboxes: deque[str] = deque(maxlen=1000)
# Counts live, booting and unclosed sessions, so the cap bounds the sandboxes that may exist.
_session_slots = threading.BoundedSemaphore(_MAX_SESSIONS)
# Guards the dicts above so a lookup's touch and a sweep's idle check can't interleave.
_store_lock = threading.Lock()


def _get_session(session_id: str) -> Session:
    """Resolve a live session (marking it used) or raise 404."""
    with _store_lock:
        session = _sessions.get(session_id)
        if session is not None:
            _last_used[session_id] = time.monotonic()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return session


def _close(session_id: str) -> None:
    """Close a session taken out of use; once it has closed, forget it and free its slot.

    A session whose close fails keeps its slot, so a retried DELETE or the next sweep can
    try again, until it has failed _MAX_CLOSE_ATTEMPTS times. Then it is given up on: its
    sandbox id is logged as an error and kept in _leaked_sandboxes, and its slot is freed
    so failing deletes can't fill the server.

    Raises:
        Exception: Whatever closing raised.
    """
    with _store_lock:
        session = _closing.get(session_id)
    if session is None:
        return  # A concurrent close got there first.
    try:
        session.close()
    except Exception:
        with _store_lock:
            failures = _close_failures[session_id] = _close_failures.get(session_id, 0) + 1
        if failures >= _MAX_CLOSE_ATTEMPTS and _forget(session_id):
            _leaked_sandboxes.append(session.sandbox_id)
            logger.error(
                'Gave up deleting sandbox %s (session %s) after %d failed attempts; '
                'it may still be running: delete it with the provider',
                session.sandbox_id,
                session_id,
                failures,
            )
            _session_slots.release()
        raise
    if _forget(session_id):
        _session_slots.release()


def _forget(session_id: str) -> bool:
    """Stop tracking a session's cleanup; return True if this call is the one that did."""
    with _store_lock:
        _close_failures.pop(session_id, None)
        return _closing.pop(session_id, None) is not None


def _close_logged(session_id: str) -> None:
    """Close a session, logging a failure instead of raising (the next sweep retries it)."""
    try:
        _close(session_id)
    except Exception:
        logger.exception('Closing session %s failed', session_id)


def sweep_sessions(max_idle_s: float | None = None) -> None:
    """Close sessions unused for *max_idle_s* seconds and retry closes that failed.

    *max_idle_s* defaults to the idle TTL; ``sweep_sessions(0)`` closes every session, as
    at shutdown. Sessions close in parallel; one that fails is logged and keeps its slot
    for the next sweep.
    """
    idle_for = _SESSION_IDLE_TTL_S if max_idle_s is None else max_idle_s
    cutoff = time.monotonic() - idle_for
    with _store_lock:
        for session_id in [sid for sid, used in _last_used.items() if used <= cutoff]:
            del _last_used[session_id]
            _closing[session_id] = _sessions.pop(session_id)
        pending = list(_closing)
    with ThreadPoolExecutor() as pool:
        pool.map(_close_logged, pending)


async def sweep_sessions_forever() -> None:
    """Sweep sessions every SESSION_SWEEP_INTERVAL_S seconds until cancelled."""
    while True:
        await asyncio.sleep(SESSION_SWEEP_INTERVAL_S)
        try:
            await asyncio.to_thread(sweep_sessions)
        except Exception:
            logger.exception('Session sweep failed; retrying next interval')


@v1_router.post('/sessions', response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    body: CreateSessionRequest,
    dg: Annotated[DeepGym, Depends(get_deepgym)],
) -> SessionResponse:
    """Open a live sandbox for one episode and return the first observation."""
    env = body.environment
    if isinstance(env, str):
        try:
            env = load_environment(env, variant=body.variant)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if not isinstance(env, MultiTurnEnvironment):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Sessions need a multi-turn environment; use /v1/run for single-turn ones.',
        )
    if not _session_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail='Session limit reached'
        )
    try:
        with _http_errors(fallback_status=status.HTTP_400_BAD_REQUEST):
            session = dg.session(env)
            try:
                observation = session.start()
            except BaseException:
                session.close()
                raise
    except BaseException:
        _session_slots.release()
        raise
    session_id = uuid.uuid4().hex
    with _store_lock:
        _sessions[session_id] = session
        _last_used[session_id] = time.monotonic()
    return SessionResponse(
        session_id=session_id, sandbox_id=session.sandbox_id, observation=observation
    )


@v1_router.post('/sessions/{session_id}/step', response_model=StepResponse)
def step_session(session_id: str, action: Action) -> StepResponse:
    """Apply one action and return the new observation, step reward, and done flag."""
    session = _get_session(session_id)
    with _http_errors(fallback_status=status.HTTP_409_CONFLICT):
        observation, reward, done = session.step(action)
    return StepResponse(observation=observation, reward=reward, done=done)


@v1_router.post('/sessions/{session_id}/score', response_model=RunResult)
def score_session(session_id: str) -> RunResult:
    """Run the final verifier against the session's current state."""
    with _http_errors():
        return _get_session(session_id).score()


@v1_router.delete('/sessions/{session_id}', status_code=status.HTTP_204_NO_CONTENT)
def close_session(session_id: str) -> None:
    """Close the session and delete its sandbox.

    If the delete fails (502), the session keeps its id and its slot, and a retried
    DELETE or the periodic sweep tries again.
    """
    with _store_lock:
        session = _sessions.pop(session_id, None)
        if session is not None:
            del _last_used[session_id]
            _closing[session_id] = session
        known = session_id in _closing
    if not known:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    try:
        _close(session_id)
    except Exception as exc:
        logger.exception('Deleting session %s failed', session_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not delete session '{session_id}': {exc}. Retry the DELETE.",
        ) from exc


# ---------------------------------------------------------------------------
# Async job endpoints
# ---------------------------------------------------------------------------
# ponytail: jobs live in this process's memory and a restart loses them. Move to a durable
# store and worker queue before running more than one API process.

# Jobs wait on runs for minutes, so they get their own threads. At most _JOB_THREADS run at
# once and the rest wait in _queued, where a cancel drops them with their request: the
# executor is never handed more work than it has threads, so nothing waits inside it.
_JOB_THREADS = min(32, (os.cpu_count() or 1) + 4)
_job_executor = ThreadPoolExecutor(max_workers=_JOB_THREADS, thread_name_prefix='deepgym-job')

_jobs: dict[str, Job | BatchJob] = {}
# Jobs waiting for a thread, oldest first: their work, cancel flag and request size.
_queued: dict[str, tuple[Callable[[], RunResult | BatchResult], threading.Event, int]] = {}
# Jobs on a thread: their cancel flag and request size, kept until the thread hands back.
_running: dict[str, tuple[threading.Event, int]] = {}
# Request bytes of the queued and running jobs.
_job_bytes = 0
# Finished jobs, oldest first, with each one's size as JSON; _finished_bytes is their sum.
_finished: deque[tuple[str, int]] = deque()
_finished_bytes = 0


def _start_job(
    job: Job | BatchJob,
    cancelled: threading.Event,
    work: Callable[[], RunResult | BatchResult],
    size: int,
) -> None:
    """Queue *job*, submitted in a *size*-byte request, to run *work* on a job thread.

    Raises:
        HTTPException: 429 if _MAX_JOBS jobs are already queued or running, or their
            requests would pass _MAX_JOB_BYTES.
    """
    global _job_bytes
    if len(_queued) + len(_running) >= _MAX_JOBS or _job_bytes + size > _MAX_JOB_BYTES:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail='Job limit reached',
        )
    _jobs[job.id] = job
    _queued[job.id] = (work, cancelled, size)
    _job_bytes += size
    loop = asyncio.get_running_loop()
    loop.call_soon(_dispatch, loop)


def _dispatch(loop: asyncio.AbstractEventLoop) -> None:
    """Give free job threads to the oldest queued jobs."""
    while _queued and len(_running) < _JOB_THREADS:
        job_id = next(iter(_queued))
        work, cancelled, size = _queued.pop(job_id)
        _running[job_id] = (cancelled, size)
        job = _jobs[job_id]
        job.status = JobStatus.running
        job.updated_at = datetime.now(timezone.utc)
        _job_executor.submit(work).add_done_callback(partial(_hand_back, loop, job_id))


def _hand_back(
    loop: asyncio.AbstractEventLoop, job_id: str, future: Future[RunResult | BatchResult]
) -> None:
    """On the job thread, pass the finished work back to the event loop."""
    with suppress(RuntimeError):  # The loop is closed: the server has stopped.
        loop.call_soon_threadsafe(_finish_job, loop, job_id, future)


def _finish_job(
    loop: asyncio.AbstractEventLoop, job_id: str, future: Future[RunResult | BatchResult]
) -> None:
    """Record how a job ended, unless it was cancelled, and free its thread and bytes."""
    global _job_bytes
    cancelled, size = _running.pop(job_id)
    _job_bytes -= size
    try:
        job = _jobs[job_id]
        if not cancelled.is_set():
            if (error := future.exception()) is not None:
                job.status = JobStatus.failed
                job.error = str(error)
            else:
                job.status = JobStatus.completed
                job.result = future.result()
                if isinstance(job.result, BatchResult):
                    job.completed_count = job.result.total
            job.updated_at = datetime.now(timezone.utc)
        _keep_finished(job_id)
    finally:
        _dispatch(loop)  # Whatever went wrong above, the thread goes to the next job.


def _keep_finished(job_id: str) -> None:
    """Keep a finished job for polling, then drop the oldest while more than
    _MAX_FINISHED_JOBS or _MAX_FINISHED_BYTES are kept; the newest always stays.

    A result that can't be sent as JSON (a lone surrogate in what the verifier returned,
    say) fails the job instead.
    """
    global _finished_bytes
    job = _jobs[job_id]
    try:
        size = len(job.model_dump_json().encode())
    except PydanticSerializationError as exc:
        job.result = None
        job.status = JobStatus.failed
        job.error = f'The result could not be serialized: {exc}'
        size = len(job.model_dump_json().encode())
    _finished.append((job_id, size))
    _finished_bytes += size
    while len(_finished) > 1 and (
        len(_finished) > _MAX_FINISHED_JOBS or _finished_bytes > _MAX_FINISHED_BYTES
    ):
        old_id, old_size = _finished.popleft()
        del _jobs[old_id]
        _finished_bytes -= old_size


async def _body_size(request: Request) -> int:
    """Return the size of the request body, which FastAPI has already read."""
    return len(await request.body())


@v1_router.post('/jobs/run', response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_run_job(
    body: RunRequest,
    dg: Annotated[ServerDeepGym, Depends(get_deepgym)],
    size: Annotated[int, Depends(_body_size)],
) -> Job:
    """Submit an async run job.

    Returns immediately with a job handle. Poll GET /v1/jobs/{job_id} for
    status and results.

    Raises:
        HTTPException: 429 if the queued and running jobs are at their limit.
    """
    cancelled = threading.Event()
    job = Job()
    work = partial(dg.for_job(cancelled).run, body.environment.to_environment(), body.model_output)
    _start_job(job, cancelled, work, size)
    return job


@v1_router.post(
    '/jobs/batch', response_model=BatchJobResponse, status_code=status.HTTP_202_ACCEPTED
)
async def submit_batch_job(
    body: BatchRunRequest,
    dg: Annotated[ServerDeepGym, Depends(get_deepgym)],
    size: Annotated[int, Depends(_body_size)],
) -> BatchJob:
    """Submit an async batch run job.

    Returns immediately with a job handle. Poll GET /v1/jobs/{job_id} for
    status and results.

    Raises:
        HTTPException: 429 if the queued and running jobs are at their limit.
    """
    cancelled = threading.Event()
    job = BatchJob(total=len(body.outputs))
    work = partial(
        dg.for_job(cancelled).run_batch,
        body.environment.to_environment(),
        body.outputs,
        max_parallel=body.max_parallel,
    )
    _start_job(job, cancelled, work, size)
    return job


@v1_router.get('/jobs/{job_id}', response_model=JobResponse | BatchJobResponse)
async def get_job(job_id: str) -> Job | BatchJob:
    """Get the current status and result of a job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job


@v1_router.post('/jobs/{job_id}/cancel', response_model=JobResponse | BatchJobResponse)
async def cancel_job(job_id: str) -> Job | BatchJob:
    """Cancel a queued or running job; it stays cancelled.

    A queued job is dropped with its request and never starts, and a running batch starts
    none of its remaining outputs. Runs already executing finish within their
    environment's timeout, their sandboxes are deleted as usual, and their results are
    dropped; the job's request counts against the job limits until then.
    """
    global _job_bytes
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    if job.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job '{job_id}' is already in terminal state '{job.status.value}'.",
        )
    job.status = JobStatus.cancelled
    job.updated_at = datetime.now(timezone.utc)
    if job_id in _queued:
        _, cancelled, size = _queued.pop(job_id)  # Its work and request go with the entry.
        _job_bytes -= size
        _keep_finished(job_id)
    else:
        cancelled = _running[job_id][0]  # Its thread finishes the runs already going.
    cancelled.set()
    return job
