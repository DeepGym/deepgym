"""Label routes: store labels singly or in a batch, list them, read current and excluded ones."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from deepgym.console.deps import LabelStoreDep, strict_query
from deepgym.console.label_store import LabelStore
from deepgym.console.schemas import LabelPage, LabelQuery
from deepgym.labels import SQLITE_MAX_INT, Label, LabelCreate, LabelTarget, TargetKind

router = APIRouter(prefix='/labels', tags=['labels'])


_NO_QUERY = [Depends(strict_query())]


@router.post('', response_model=Label, status_code=status.HTTP_201_CREATED, dependencies=_NO_QUERY)
def create_label(body: LabelCreate, labels: LabelStoreDep) -> Label:
    """Store one label; a retry with the same client id and body returns the stored one."""
    return _add(labels, [body])[0]


@router.post(
    '/batch', response_model=list[Label], status_code=status.HTTP_201_CREATED,
    dependencies=_NO_QUERY,
)  # fmt: skip
def create_labels(body: list[LabelCreate], labels: LabelStoreDep) -> list[Label]:
    """Store every label of *body* in order, or none of them if any is refused."""
    return _add(labels, body)


def _add(labels: LabelStore, body: Sequence[LabelCreate]) -> list[Label]:
    """Store *body*.

    Raises:
        HTTPException: 409 if the store refuses a record (``LabelStore.add_many``).
    """
    try:
        return labels.add_many(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get(
    '',
    response_model=LabelPage,
    dependencies=[Depends(strict_query(*LabelQuery.model_fields, 'offset', 'limit'))],
)
def list_labels(
    labels: LabelStoreDep,
    query: Annotated[LabelQuery, Depends()],
    offset: int = Query(0, ge=0, le=SQLITE_MAX_INT),
    limit: int | None = Query(None, ge=1, le=SQLITE_MAX_INT),
) -> LabelPage:
    """List matching labels a page at a time, in ``seq`` order (every one without *limit*)."""
    return labels.page(query, offset, limit)


@router.get(
    '/current',
    response_model=list[Label],
    dependencies=[Depends(strict_query('target_kind', 'target_id'))],
)
def get_current(target_kind: TargetKind, target_id: str, labels: LabelStoreDep) -> list[Label]:
    """Return a target's records still in force, its steps' included."""
    return labels.current(target_kind, target_id)


@router.get(
    '/excluded',
    response_model=list[LabelTarget],
    response_model_exclude_none=True,
    dependencies=_NO_QUERY,
)
def get_excluded(labels: LabelStoreDep) -> list[LabelTarget]:
    """Return the targets a current invalidation excludes, as ``{"kind": ..., "id": ...}``."""
    targets = sorted(labels.excluded())
    return [LabelTarget.model_construct(kind=kind, id=target_id) for kind, target_id in targets]
