"""Environment routes: the registry with filters, and one environment in full."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from deepgym.console.schemas import EnvironmentDetail, EnvironmentSummary, EnvKind
from deepgym.models import Environment, MultiTurnEnvironment
from deepgym.registry import entry_label, list_environments, load_environment

router = APIRouter(prefix='/environments', tags=['environments'])

_KINDS: dict[str, EnvKind] = {'computer-use': 'computer-use', 'multi-turn': 'multi-turn'}
# Fields of an environment not in the registry that stand in for its registry entry.
_ENTRY_FIELDS = {'name', 'type', 'difficulty', 'domain', 'tags'}


def summarize(entry: dict) -> EnvironmentSummary:
    """Build a table row from a registry entry; coding and tool-use envs are single-turn."""
    env_type = str(entry.get('type', 'coding'))
    label = entry_label(entry)
    return EnvironmentSummary(
        id=label,
        name=str(entry.get('name') or label),
        kind=_KINDS.get(env_type, 'single-turn'),
        type=env_type,
        difficulty=str(entry.get('difficulty', 'medium')),
        domain=str(entry.get('domain', 'coding')),
        family=str(entry.get('family', '')),
        tags=[str(tag) for tag in entry.get('tags', [])],
        variants=int(entry.get('variants', 1)),
    )


def reference_solution(env: Environment | MultiTurnEnvironment) -> str | None:
    """Return *env*'s reference solution, if it is single-turn and has one."""
    return env.reference_solution if isinstance(env, Environment) else None


@router.get('', response_model=list[EnvironmentSummary])
def list_envs(
    domain: str | None = None,
    difficulty: str | None = None,
    kind: EnvKind | None = None,
) -> list[EnvironmentSummary]:
    """List registry environments matching every given filter."""
    rows = [summarize(entry) for entry in list_environments()]
    return [
        row
        for row in rows
        if domain in (None, row.domain)
        and difficulty in (None, row.difficulty)
        and kind in (None, row.kind)
    ]


@router.get('/{name:path}', response_model=EnvironmentDetail)
def get_env(name: str, variant: int = Query(0, ge=0)) -> EnvironmentDetail:
    """Return an environment's task, verifier source, reference solution and metadata."""
    try:
        env = load_environment(name, variant=variant)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    entry = next((e for e in list_environments() if entry_label(e) == name), None)
    summary = summarize(entry or {'path': name, **env.model_dump(include=_ENTRY_FIELDS)})
    verifier = (
        env.final_verifier_code if isinstance(env, MultiTurnEnvironment) else env.verifier_source
    )
    return EnvironmentDetail(
        **summary.model_dump(),
        variant=variant,
        task=env.task,
        verifier=verifier,
        reference=reference_solution(env),
        metadata=entry or {},
    )
