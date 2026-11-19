"""FastAPI dependencies: the stores and runner that create_app puts on app.state."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request

from deepgym.console.episode_store import EpisodeStore
from deepgym.console.label_store import LabelStore
from deepgym.console.run_store import RunStore
from deepgym.console.runner import Runner


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
