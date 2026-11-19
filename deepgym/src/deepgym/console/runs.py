"""Run routes: start runs or a small eval in the background, then list, inspect and compare them."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic_core import PydanticSerializationError

from deepgym.compare import Attempt, CompareReport, compare
from deepgym.console.deps import RunnerDep, RunStoreDep, strict_query
from deepgym.console.environments import reference_solution
from deepgym.console.run_store import RunStore, run_attempt
from deepgym.console.schemas import (
    RunConfig,
    RunCreate,
    RunDetail,
    RunPage,
    RunSpec,
    RunStats,
    RunStatus,
    RunSummary,
)
from deepgym.exceptions import DeepGymError
from deepgym.models import Environment
from deepgym.registry import load_environment
from deepgym.traces.schema import environment_sha256

router = APIRouter(prefix='/runs', tags=['runs'])


@router.post('', response_model=list[RunSummary], status_code=status.HTTP_202_ACCEPTED)
def start_runs(body: RunCreate, runner: RunnerDep) -> list[RunSummary]:
    """Queue one run per environment; nothing is queued if any environment is unusable."""
    try:
        dg = runner.client(body.mode)
        runner.check_allowed(dg, custom_code=body.solution is not None)
    except DeepGymError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    jobs = [_plan(name, body.solution) for name in body.environments]
    return runner.submit(dg, body.mode, jobs)


def _plan(name: str, solution: str | None) -> tuple[RunSpec, Environment]:
    """Load *name* and pick its solution: the given one, or else its reference.

    Raises:
        HTTPException: 404 for an unknown environment, 400 if it can't be run this way or
            its definition can't be hashed (a file it names is unreadable, say).
    """
    try:
        env = load_environment(name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if not isinstance(env, Environment):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'{name} is a {env.type} environment: it needs an agent, not a solution. '
            'Record its episodes with deepgym.console.record_episode instead.',
        )
    code = solution if solution is not None else reference_solution(env)
    if code is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'{name} has no reference solution; pass a solution',
        )
    try:
        fingerprint = environment_sha256(env)
    except PydanticSerializationError as exc:
        reason = exc.__cause__ or exc  # the serializer's own error, without pydantic's wrapper
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Can't hash {name}'s environment definition: {reason}",
        ) from exc
    spec = RunSpec(
        environment=name,
        solution=code,
        reference=solution is None,
        environment_sha256=fingerprint,
    )
    return spec, env


@router.get('/config', response_model=RunConfig)
def run_config(runner: RunnerDep) -> RunConfig:
    """Tell the UI whether custom solutions may run on this machine."""
    return RunConfig(local_custom_code=runner.allow_local_exec)


def run_filters(
    run_status: RunStatus | None = Query(None, alias='status'),
    environment: str | None = None,
    provider: str | None = None,
    group_id: str | None = None,
) -> dict[str, str | None]:
    """Collect the query filters shared by the run list and its stats."""
    return {
        'status': run_status,
        'environment': environment,
        'provider': provider,
        'group_id': group_id,
    }


RunFiltersDep = Annotated[dict[str, str | None], Depends(run_filters)]


@router.get('', response_model=RunPage)
def list_runs(
    runs: RunStoreDep,
    filters: RunFiltersDep,
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1),
) -> RunPage:
    """List the newest matching runs a page at a time (every one without *limit*) and the total."""
    return runs.page(filters, offset, limit)


@router.get('/stats', response_model=RunStats)
def run_stats(
    runs: RunStoreDep,
    filters: RunFiltersDep,
    hours: int | None = Query(None, ge=1, le=24 * 365),
) -> RunStats:
    """Aggregate runs matching every filter, started in the last *hours* (or ever)."""
    return runs.stats(filters, hours)


@router.get(
    '/compare', response_model=CompareReport, dependencies=[Depends(strict_query('a', 'b'))]
)
def compare_groups(runs: RunStoreDep, a: str, b: str) -> CompareReport:
    """Compare run group *b* against run group *a*, environment by environment.

    Raises:
        HTTPException: 404 if a group has no runs, 409 while one still has runs queued or
            running, 422 for a missing, unknown or repeated query parameter.
    """
    sides = [_finished_group(runs, group_id) for group_id in (a, b)]
    return compare(*sides)


def _finished_group(runs: RunStore, group_id: str) -> list[Attempt]:
    """Return the runs of *group_id* as attempts, once every one of them has finished.

    Raises:
        HTTPException: 404 if the group has no runs, 409 if any is queued or running.
    """
    group = runs.page({'group_id': group_id}, 0, None).runs
    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f'Run group {group_id} not found'
        )
    if busy := sum(run.status in ('queued', 'running') for run in group):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f'Run group {group_id} has {busy} runs queued or running; '
            'compare it once they finish',
        )
    return [run_attempt(run) for run in group]


@router.get('/{run_id}', response_model=RunDetail)
def get_run(run_id: str, runs: RunStoreDep) -> RunDetail:
    """Return a run with its solution, error and per-case verifier results."""
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Run {run_id} not found')
    return run
