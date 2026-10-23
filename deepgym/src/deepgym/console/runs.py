"""Run routes: start runs or a small eval in the background, then list and inspect them."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from deepgym.console.deps import RunnerDep, RunStoreDep
from deepgym.console.environments import reference_solution
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
        HTTPException: 404 for an unknown environment, 400 if it can't be run this way.
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
    return RunSpec(environment=name, solution=code, reference=solution is None), env


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


@router.get('/{run_id}', response_model=RunDetail)
def get_run(run_id: str, runs: RunStoreDep) -> RunDetail:
    """Return a run with its solution, error and per-case verifier results."""
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'Run {run_id} not found')
    return run
