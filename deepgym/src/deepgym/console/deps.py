"""FastAPI dependencies: the stores and runner that create_app puts on app.state, and a
strict query-parameter check."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from deepgym.console.episode_store import EpisodeStore
from deepgym.console.label_store import LabelStore
from deepgym.console.run_store import RunStore
from deepgym.console.runner import Runner


def strict_query(*names: str) -> Callable[[Request], None]:
    """Return a dependency that refuses query parameters other than *names*, and any given
    twice, with 422, so a mistyped or repeated parameter fails instead of being ignored."""
    allowed = frozenset(names)

    def check(request: Request) -> None:
        counts = Counter(name for name, _ in request.query_params.multi_items())
        unknown = sorted(counts.keys() - allowed)
        repeated = sorted(name for name, count in counts.items() if count > 1)
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f'Unknown query parameters {unknown}; known: {sorted(allowed)}',
            )
        if repeated:
            raise HTTPException(
                status_code=422, detail=f'Query parameters given more than once: {repeated}'
            )

    return check


def get_run_store(request: Request) -> RunStore:
    """Return the app's run store."""
    return request.app.state.runs


def get_episode_store(request: Request) -> EpisodeStore:
    """Return the app's episode store."""
    return request.app.state.episodes


def get_label_store(request: Request) -> LabelStore:
    """Return the app's label store."""
    return request.app.state.labels


def get_runner(request: Request) -> Runner:
    """Return the app's background runner."""
    return request.app.state.runner


def get_qa_dir(request: Request) -> Path:
    """Return the directory QA reports are read from."""
    return request.app.state.qa_dir


RunStoreDep = Annotated[RunStore, Depends(get_run_store)]
EpisodeStoreDep = Annotated[EpisodeStore, Depends(get_episode_store)]
LabelStoreDep = Annotated[LabelStore, Depends(get_label_store)]
RunnerDep = Annotated[Runner, Depends(get_runner)]
QADirDep = Annotated[Path, Depends(get_qa_dir)]
