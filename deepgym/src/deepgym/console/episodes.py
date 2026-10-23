"""Episode routes: store finished trajectories and replay them frame by frame."""

from __future__ import annotations

import binascii

from fastapi import APIRouter, HTTPException, Query, Response, status

from deepgym.console.deps import EpisodeStoreDep
from deepgym.console.schemas import EpisodeCreate, EpisodeDetail, EpisodePage, EpisodeSummary

router = APIRouter(prefix='/episodes', tags=['episodes'])


@router.post('', response_model=EpisodeSummary, status_code=status.HTTP_201_CREATED)
def create_episode(body: EpisodeCreate, episodes: EpisodeStoreDep) -> EpisodeSummary:
    """Store a trajectory (and optional final result) for replay."""
    try:
        return episodes.add(body)
    except binascii.Error as exc:
        raise HTTPException(
            status_code=422,
            detail=f'A screenshot is not valid base64: {exc}',
        ) from exc


@router.get('', response_model=EpisodePage)
def list_episodes(
    episodes: EpisodeStoreDep,
    environment: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1),
) -> EpisodePage:
    """List the newest episodes a page at a time (every one without *limit*), with the total."""
    return episodes.page(environment, offset, limit)


@router.get('/{episode_id}', response_model=EpisodeDetail)
def get_episode(episode_id: str, episodes: EpisodeStoreDep) -> EpisodeDetail:
    """Return an episode's frames (observation text, action, reward) and result."""
    episode = episodes.get(episode_id)
    if episode is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f'Episode {episode_id} not found'
        )
    return episode


@router.get('/{episode_id}/frames/{frame}/screenshot', response_class=Response)
def get_screenshot(episode_id: str, frame: int, episodes: EpisodeStoreDep) -> Response:
    """Return the PNG screenshot shown at one frame."""
    png = episodes.screenshot(episode_id, frame)
    if png is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No screenshot')
    return Response(content=png, media_type='image/png')
