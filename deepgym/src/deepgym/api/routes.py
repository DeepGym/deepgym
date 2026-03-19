"""API route definitions for DeepGym."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Annotated

try:
    from fastapi import APIRouter, Depends, HTTPException, status
except ImportError as _exc:
    raise ImportError(
        'FastAPI is required for the server. Install with: pip install "deepgym[server]"'
    ) from _exc

from deepgym.api.deps import get_deepgym
from deepgym.api.schemas import (
    BatchJobResponse,
    BatchRunRequest,
    CreateEnvironmentRequest,
    CreateEnvironmentResponse,
    CreateSnapshotRequest,
    EvalRequest,
    HealthResponse,
    JobResponse,
    RunRequest,
)
from deepgym.core import DeepGym
from deepgym.exceptions import DeepGymError, SandboxError, VerifierError
from deepgym.models import (
    BatchJob,
    BatchResult,
    Environment,
    EvalResult,
    Job,
    JobStatus,
    RunResult,
)

# ---------------------------------------------------------------------------
# Health router
# ---------------------------------------------------------------------------

health_router = APIRouter(tags=['health'])


@health_router.get('/health', response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return service health status and version."""
    return HealthResponse()


# ---------------------------------------------------------------------------
# V1 router
# ---------------------------------------------------------------------------

v1_router = APIRouter(prefix='/v1', tags=['v1'])

# In-memory store capacity limits.
_MAX_ENVIRONMENTS = 1000
_MAX_JOBS = 10_000

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
    try:
        return dg.run(body.environment.to_environment(), body.model_output)
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
def create_environment(body: CreateEnvironmentRequest) -> CreateEnvironmentResponse:
    """Create and register a new environment.

    The environment is stored in memory and assigned a unique identifier.

    Raises:
        HTTPException: 429 if the in-memory store has reached capacity.
    """
    if len(_environments) >= _MAX_ENVIRONMENTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail='Environment limit reached',
        )
    env_id = uuid.uuid4().hex[:12]
    env = Environment(**body.model_dump())
    _environments[env_id] = env
    return CreateEnvironmentResponse(id=env_id, created=True)


def _strip_env_vars(env: Environment) -> dict:
    """Return environment data with env_vars removed to prevent secret leakage.

    Args:
        env: The environment model to serialize.

    Returns:
        Dict representation with env_vars set to None.
    """
    return {**env.model_dump(), 'env_vars': None}


@v1_router.get('/environments')
def list_environments() -> list[dict]:
    """List all registered environments (env_vars stripped)."""
    return [_strip_env_vars(env) for env in _environments.values()]


@v1_router.get('/environments/{env_id}')
def get_environment(env_id: str) -> dict:
    """Retrieve a specific environment by its identifier (env_vars stripped)."""
    env = _environments.get(env_id)
    if env is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment '{env_id}' not found.",
        )
    return _strip_env_vars(env)


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
# Async job endpoints
# ---------------------------------------------------------------------------
# NOTE: This MVP uses an in-memory dict and asyncio.create_task for background
# execution. Production would use Postgres + a worker queue (e.g. Celery, arq,
# or Temporal) for persistence, retries, and horizontal scaling.

_jobs: dict[str, Job | BatchJob] = {}


def _check_job_capacity() -> None:
    """Raise 429 if the in-memory job store has reached capacity."""
    if len(_jobs) >= _MAX_JOBS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail='Job limit reached',
        )


async def _run_job_background(
    job_id: str, dg: DeepGym, environment: Environment, model_output: str
) -> None:
    """Execute a single run in the background and update the job store."""
    job = _jobs[job_id]
    assert isinstance(job, Job)
    try:
        job.status = JobStatus.running
        job.updated_at = datetime.now(UTC)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, dg.run, environment, model_output)

        if job.status == JobStatus.cancelled:
            return  # Don't overwrite cancelled status
        job.status = JobStatus.completed
        job.result = result
    except Exception as exc:
        if job.status == JobStatus.cancelled:
            return  # Don't overwrite cancelled status
        job.status = JobStatus.failed
        job.error = str(exc)
    finally:
        job.updated_at = datetime.now(UTC)


async def _run_batch_job_background(
    job_id: str, dg: DeepGym, environment: Environment, outputs: list[str], max_parallel: int
) -> None:
    """Execute a batch run in the background and update the job store."""
    job = _jobs[job_id]
    assert isinstance(job, BatchJob)
    try:
        job.status = JobStatus.running
        job.updated_at = datetime.now(UTC)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: dg.run_batch(environment, outputs, max_parallel=max_parallel)
        )

        if job.status == JobStatus.cancelled:
            return  # Don't overwrite cancelled status
        job.status = JobStatus.completed
        job.completed_count = result.total
        job.result = result
    except Exception as exc:
        if job.status == JobStatus.cancelled:
            return  # Don't overwrite cancelled status
        job.status = JobStatus.failed
        job.error = str(exc)
    finally:
        job.updated_at = datetime.now(UTC)


@v1_router.post('/jobs/run', response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_run_job(
    body: RunRequest,
    dg: Annotated[DeepGym, Depends(get_deepgym)],
) -> Job:
    """Submit an async run job.

    Returns immediately with a job handle. Poll GET /v1/jobs/{job_id} for
    status and results.

    Raises:
        HTTPException: 429 if the in-memory job store has reached capacity.
    """
    _check_job_capacity()
    job = Job()
    _jobs[job.id] = job
    asyncio.create_task(
        _run_job_background(job.id, dg, body.environment.to_environment(), body.model_output)
    )
    return job


@v1_router.post(
    '/jobs/batch', response_model=BatchJobResponse, status_code=status.HTTP_202_ACCEPTED
)
async def submit_batch_job(
    body: BatchRunRequest,
    dg: Annotated[DeepGym, Depends(get_deepgym)],
) -> BatchJob:
    """Submit an async batch run job.

    Returns immediately with a job handle. Poll GET /v1/jobs/{job_id} for
    status and results.

    Raises:
        HTTPException: 429 if the in-memory job store has reached capacity.
    """
    _check_job_capacity()
    job = BatchJob(total=len(body.outputs))
    _jobs[job.id] = job
    asyncio.create_task(
        _run_batch_job_background(
            job.id, dg, body.environment.to_environment(), body.outputs, body.max_parallel
        )
    )
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
    """Cancel a queued or running job.

    Sets job status to cancelled. Note: in this MVP implementation the
    background task is not forcibly interrupted; it will finish but its
    result will be discarded on the next status check.
    """
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
    job.updated_at = datetime.now(UTC)
    return job
