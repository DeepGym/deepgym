"""Calibration routes: a judge's stored verdicts on runs against the experts', from stored
labels only, with no judge call."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from deepgym.console.calibration_labels import StoredAgreement, StoredJudge, stored_agreement
from deepgym.console.deps import LabelStoreDep, QueueStoreDep, RunStoreDep, strict_query
from deepgym.console.label_store import LabelStore
from deepgym.console.queue_store import QueueNotFoundError, QueueStore
from deepgym.console.schemas import LabelQuery
from deepgym.labels import Name

router = APIRouter(prefix='/calibration', tags=['calibration'])

# Every LabelQuery filter but target_kind, which is always 'run', prefixed so that its
# source, author and rubric_sha256 don't clash with the judge's.
_LABEL_PARAMS = tuple(f'label_{name}' for name in LabelQuery.model_fields if name != 'target_kind')
_SELECT = 'Select the runs one way: run_ids, queue_id, or label_* filters'


def _label_filter(request: Request) -> LabelQuery | None:
    """Return the ``label_*`` query parameters as a query for labels on runs; None without any.

    Raises:
        RequestValidationError: For a value ``LabelQuery`` refuses (422).
    """
    given = {
        key.removeprefix('label_'): value
        for key, value in request.query_params.items()
        if key in _LABEL_PARAMS
    }
    if not given:
        return None
    try:
        return LabelQuery.model_validate({**given, 'target_kind': 'run'})
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        raise RequestValidationError(
            [{**error, 'loc': ('query', f'label_{error["loc"][0]}', *error['loc'][1:])}
             for error in errors]
        ) from exc  # fmt: skip


@router.get(
    '/agreement',
    response_model=StoredAgreement,
    dependencies=[
        Depends(
            strict_query(
                'run_ids', 'queue_id', 'judge_author', 'judge_source', 'judge_rubric_version',
                'rubric_sha256', *_LABEL_PARAMS,
            )
        )
    ],
)  # fmt: skip
def agreement(
    labels: LabelStoreDep,
    runs: RunStoreDep,
    queues: QueueStoreDep,
    label_query: Annotated[LabelQuery | None, Depends(_label_filter)],
    judge_author: Name,
    judge_source: Literal['judge', 'verifier'] = 'judge',
    judge_rubric_version: Name | None = None,
    rubric_sha256: str | None = Query(None, pattern=r'^[0-9a-f]{64}$'),
    run_ids: str | None = None,
    queue_id: str | None = None,
) -> StoredAgreement:
    """Measure a judge's stored verdicts on the runs selected against the experts'.

    Raises:
        HTTPException: 404 for an unknown queue, 422 unless the runs are selected exactly
            one way, or for an empty id in ``run_ids``.
    """
    selected = _selected_runs(labels, queues, run_ids, queue_id, label_query)
    judge = StoredJudge(
        author=judge_author, source=judge_source, rubric_version=judge_rubric_version
    )
    return stored_agreement(labels, runs, selected, judge, rubric_sha256=rubric_sha256)


def _selected_runs(
    labels: LabelStore,
    queues: QueueStore,
    run_ids: str | None,
    queue_id: str | None,
    label_query: LabelQuery | None,
) -> list[str]:
    """Return the run ids *run_ids* lists (comma-separated), the queue's run items hold, or
    *label_query*'s labels are on, in that order; exactly one of them must be given.

    Raises:
        HTTPException: 404 for an unknown queue, 422 for no selection or several, or an
            empty id in *run_ids*.
    """
    if [run_ids, queue_id, label_query].count(None) != 2:
        raise HTTPException(status_code=422, detail=_SELECT)
    if run_ids is not None:
        selected = run_ids.split(',')
        if '' in selected:
            raise HTTPException(status_code=422, detail=f'run_ids has an empty id: {run_ids!r}')
        return selected
    if queue_id is not None:
        try:
            items = queues.items(queue_id).items
        except QueueNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return [each.item.target.id for each in items if each.item.target.kind == 'run']
    if label_query is None:
        raise HTTPException(status_code=422, detail=_SELECT)
    return [label.target.id for label in labels.page(label_query, 0, None).labels]
