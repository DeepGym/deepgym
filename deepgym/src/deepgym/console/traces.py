"""Trace routes: list the traces recorded episodes kept, and read one with its labels.

A stored trace that doesn't read as one trace (copies that disagree, ids differing only in
case, a row that doesn't parse) is left out and reported; it never fails the others.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from deepgym.console.deps import EpisodeStoreDep, LabelStoreDep, strict_query
from deepgym.console.schemas import TraceDetail, TracePage, TraceQuery, TraceSummary
from deepgym.labels import SQLITE_MAX_INT
from deepgym.traces import summary_row

router = APIRouter(prefix='/traces', tags=['traces'])


@router.get(
    '',
    response_model=TracePage,
    dependencies=[Depends(strict_query(*TraceQuery.model_fields, 'offset', 'limit'))],
)
def list_traces(
    episodes: EpisodeStoreDep,
    query: Annotated[TraceQuery, Depends()],
    offset: int = Query(0, ge=0, le=SQLITE_MAX_INT),
    limit: int | None = Query(None, ge=1, le=SQLITE_MAX_INT),
) -> TracePage:
    """List matching traces a page at a time, newest first (every one without *limit*),
    with every stored trace left out, whatever the filters."""
    # ponytail: parses every stored trace per request; add summary columns if this gets slow.
    held = episodes.held_traces()
    found = sorted(
        (trace for trace in held.traces if query.matches(trace)),
        key=lambda trace: (trace.provenance.started_at, trace.trace_id),
        reverse=True,
    )
    page = found[offset:] if limit is None else found[offset : offset + limit]
    return TracePage(
        traces=[TraceSummary.model_validate(summary_row(trace)) for trace in page],
        total=len(found),
        conflicts=list(held.conflicts),
        unreadable=list(held.unreadable),
    )


@router.get('/{trace_id}', response_model=TraceDetail, dependencies=[Depends(strict_query())])
def get_trace(trace_id: str, episodes: EpisodeStoreDep, labels: LabelStoreDep) -> TraceDetail:
    """Return a whole trace, with its usage and cost totals, its current labels and those on
    the episodes recorded from it.

    Raises:
        HTTPException: 404 if no copy of it reads as a trace, 409 if its copies disagree or
            another trace's id is the same but for case.
    """
    held = episodes.held_traces(trace_id)
    if conflict := next((c for c in held.conflicts if c.trace_id == trace_id), None):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=conflict.detail)
    trace = next((trace for trace in held.traces if trace.trace_id == trace_id), None)
    if trace is None:
        unreadable = f'; episodes {list(held.unreadable)} hold copies that do not read'
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Trace {trace_id} not found{unreadable if held.unreadable else ""}',
        )
    episode_labels = [
        label
        for episode_id in held.episodes[trace_id]
        for label in labels.current('episode', episode_id)
    ]
    return TraceDetail(
        trace=trace,
        labels=labels.current('trace', trace_id),
        episode_labels=sorted(episode_labels, key=lambda label: label.seq),
        unreadable=list(held.unreadable),
    )
