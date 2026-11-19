"""Review queue routes: create queues and add items, hand items to reviewers, serve what they
grade, take their reviews, and report progress."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status

from deepgym.console.deps import QueueStoreDep, strict_query
from deepgym.console.item_content import UnreadableRecordingError
from deepgym.console.queue_store import NotReviewerError, QueueInputError, QueueNotFoundError
from deepgym.console.schemas import ReviewerBody, ReviewSubmit
from deepgym.labels import SQLITE_MAX_INT, Name
from deepgym.review_queues import (
    Assignment,
    ItemContent,
    ItemCreate,
    ItemPage,
    Queue,
    QueueCreate,
    QueueItem,
    QueueProgress,
    QueueUpdate,
    ReviewView,
)

router = APIRouter(prefix='/queues', tags=['queues'])
_NO_QUERY = [Depends(strict_query())]


@contextmanager
def _refusals() -> Iterator[None]:
    """Answer the store's refusals: 404 for an unknown queue or item, 403 for a reviewer
    the queue doesn't list, 422 for input its settings or the label rules refuse (each
    problem's ``loc`` from the body), 409 for the rest; and 500, with fixed text that quotes
    nothing stored, for a recording whose content can't be read.

    Raises:
        HTTPException: For each refusal.
    """
    try:
        yield
    except QueueNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NotReviewerError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except QueueInputError as exc:
        detail = [{**error, 'loc': ('body', *error['loc'])} for error in exc.errors]
        raise HTTPException(status_code=422, detail=detail) from exc
    except UnreadableRecordingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The console can't read this item's stored recording",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post('', response_model=Queue, status_code=status.HTTP_201_CREATED, dependencies=_NO_QUERY)
def create_queue(body: QueueCreate, queues: QueueStoreDep) -> Queue:
    """Create an open queue; a retry with the same client id and settings returns it."""
    with _refusals():
        return queues.create(body)


@router.get('', response_model=list[Queue], dependencies=_NO_QUERY)
def list_queues(queues: QueueStoreDep) -> list[Queue]:
    """List every queue, oldest first."""
    return queues.queues()


@router.get('/{queue_id}', response_model=Queue, dependencies=_NO_QUERY)
def get_queue(queue_id: str, queues: QueueStoreDep) -> Queue:
    """Return one queue."""
    with _refusals():
        return queues.get(queue_id)


@router.patch('/{queue_id}', response_model=Queue, dependencies=_NO_QUERY)
def update_queue(queue_id: str, body: QueueUpdate, queues: QueueStoreDep) -> Queue:
    """Close or reopen a queue, add reviewers, or change its description, instructions,
    reservation time or (until a review is submitted) reveal policy; fields left out keep
    their value."""
    with _refusals():
        return queues.update(queue_id, body)


@router.post(
    '/{queue_id}/items', response_model=list[QueueItem], status_code=status.HTTP_201_CREATED,
    dependencies=_NO_QUERY,
)  # fmt: skip
def add_items(queue_id: str, body: list[ItemCreate], queues: QueueStoreDep) -> list[QueueItem]:
    """Add items in order, or none if any is refused; a target already queued is kept."""
    with _refusals():
        return queues.add_items(queue_id, body)


@router.get(
    '/{queue_id}/items',
    response_model=ItemPage,
    dependencies=[Depends(strict_query('offset', 'limit'))],
)
def list_items(
    queue_id: str,
    queues: QueueStoreDep,
    offset: int = Query(0, ge=0, le=SQLITE_MAX_INT),
    limit: int | None = Query(None, ge=1, le=SQLITE_MAX_INT),
) -> ItemPage:
    """List a queue's items with their review counts (every one without *limit*)."""
    with _refusals():
        return queues.items(queue_id, offset, limit)


@router.get('/{queue_id}/progress', response_model=QueueProgress, dependencies=_NO_QUERY)
def get_progress(queue_id: str, queues: QueueStoreDep) -> QueueProgress:
    """Report how far a queue's reviews are, overall and per reviewer."""
    with _refusals():
        return queues.progress(queue_id)


@router.post('/{queue_id}/next', response_model=ReviewView | None, dependencies=_NO_QUERY)
def next_item(queue_id: str, body: ReviewerBody, queues: QueueStoreDep) -> ReviewView | None:
    """Reserve the next item for a reviewer (or extend the one they hold); null if none."""
    with _refusals():
        return queues.next_item(queue_id, body.reviewer)


@router.get(
    '/{queue_id}/items/{item_id}',
    response_model=ReviewView,
    dependencies=[Depends(strict_query('reviewer'))],
)
def view_item(queue_id: str, item_id: str, reviewer: Name, queues: QueueStoreDep) -> ReviewView:
    """Return an item as a reviewer may see it."""
    with _refusals():
        return queues.view(queue_id, item_id, reviewer)


@router.get(
    '/{queue_id}/items/{item_id}/content',
    response_model=ItemContent,
    dependencies=[Depends(strict_query('reviewer'))],
)
def item_content(queue_id: str, item_id: str, reviewer: Name, queues: QueueStoreDep) -> ItemContent:
    """Return what a reviewer who could submit a review of the item, or has, needs to grade
    it, and nothing that reveals a grade."""
    with _refusals():
        return queues.content(queue_id, item_id, reviewer)


@router.get(
    '/{queue_id}/items/{item_id}/frames/{frame}/screenshot',
    response_class=Response,
    dependencies=[Depends(strict_query('reviewer'))],
)
def item_screenshot(
    queue_id: str,
    item_id: str,
    frame: Annotated[int, Path(ge=0, le=SQLITE_MAX_INT)],
    reviewer: Name,
    queues: QueueStoreDep,
) -> Response:
    """Return the PNG shown at one frame of an episode item, to a reviewer who may read its
    content."""
    with _refusals():
        png = queues.screenshot(queue_id, item_id, reviewer, frame)
    return Response(content=png, media_type='image/png')


@router.post(
    '/{queue_id}/items/{item_id}/submit', response_model=ReviewView,
    status_code=status.HTTP_201_CREATED, dependencies=_NO_QUERY,
)  # fmt: skip
def submit_review(
    queue_id: str, item_id: str, body: ReviewSubmit, queues: QueueStoreDep
) -> ReviewView:
    """Store a review of an item; nothing is stored if any of it is refused."""
    with _refusals():
        return queues.submit(queue_id, item_id, body.reviewer, body.labels)


@router.post(
    '/{queue_id}/items/{item_id}/release', response_model=Assignment, dependencies=_NO_QUERY
)
def release_item(
    queue_id: str, item_id: str, body: ReviewerBody, queues: QueueStoreDep
) -> Assignment:
    """Hand back a held item; it may be handed to the same reviewer again."""
    with _refusals():
        return queues.release(queue_id, item_id, body.reviewer)


@router.post('/{queue_id}/items/{item_id}/skip', response_model=Assignment, dependencies=_NO_QUERY)
def skip_item(queue_id: str, item_id: str, body: ReviewerBody, queues: QueueStoreDep) -> Assignment:
    """Hand back a held item, never to be handed to the same reviewer again."""
    with _refusals():
        return queues.skip(queue_id, item_id, body.reviewer)
