"""Scan routes: run rule scanners over the console's traces, and store what they found as
labels. Reports aren't kept; the labels are the durable record.

Each scan runs in a new Python process, so a client's regex can't hold the server's GIL or
event loop, and the process is killed when the client disconnects. There is no time limit.
The process is an isolated interpreter (``python -I -c``, in deepgym's own folder): it
never runs this process's ``__main__``, never imports from the server's working directory
or ``PYTHON*`` variables, and takes its ``sys.path`` and job as JSON on stdin.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from pathlib import Path

import anyio
import anyio.lowlevel
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, ValidationError

from deepgym.console.deps import EpisodeStoreDep, LabelStoreDep, strict_query
from deepgym.console.schemas import ScanCreate, ScanResult
from deepgym.scanners import (
    PatternScanner,
    ProtectedPathScanner,
    RuleScanner,
    ScanReport,
    finding_labels,
    scan_traces,
)
from deepgym.traces import Trace

router = APIRouter(prefix='/scans', tags=['scans'])

_POLL_SECONDS = 0.05
"""How often a request waiting on its scan process checks whether the client left."""

SCAN_COMMAND = [
    sys.executable, '-I', '-c',
    'import json, sys; sys.path[:0] = json.loads(sys.stdin.buffer.readline()); '
    'from deepgym.console.scans import _child; _child()',
]  # fmt: skip
"""How a scan process starts: an isolated interpreter (no working directory or user site on
``sys.path``, ``PYTHON*`` variables ignored) that takes the first stdin line as the paths to
import from, then imports only this module."""

_SCAN_FOLDER = Path(__file__).resolve().parent.parent
"""The scan process's working directory: deepgym's own package folder."""

CLIENT_CLOSED = 499
"""The status a scan stopped because its client disconnected answers (nobody reads it)."""


@router.post('', response_model=ScanResult, dependencies=[Depends(strict_query())])
async def scan(
    body: ScanCreate, request: Request, episodes: EpisodeStoreDep, labels: LabelStoreDep
) -> ScanResult:
    """Scan the traces asked for (every one that reads by default) and return the report
    with the stored traces left out; with ``write_labels``, also store its labels in one
    transaction.

    Raises:
        HTTPException: 404 for a trace the console doesn't hold, 409 for one named whose
            stored copies don't read as one trace, or if the label store refuses a label,
            and see :func:`scan_in_process`.
    """
    held = await run_in_threadpool(episodes.held_traces)
    traces = list(held.traces)
    if body.trace_ids is not None:
        wanted = list(dict.fromkeys(body.trace_ids))
        if bad := [conflict.detail for conflict in held.conflicts if conflict.trace_id in wanted]:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='; '.join(bad))
        found = {trace.trace_id: trace for trace in traces}
        if missing := [trace_id for trace_id in wanted if trace_id not in found]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f'Traces {missing} not found'
            )
        traces = [found[trace_id] for trace_id in wanted]
    report = await scan_in_process(traces, body.scanners, request.is_disconnected)
    if body.run_id is not None:
        report = report.model_copy(update={'run_id': body.run_id})
    if body.write_labels:
        try:
            await run_in_threadpool(labels.add_many, finding_labels(report))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ScanResult(
        **report.model_dump(), conflicts=list(held.conflicts), unreadable=list(held.unreadable)
    )


class _ScanJob(BaseModel):
    """What the scan process is sent."""

    model_config = ConfigDict(frozen=True)

    traces: list[Trace]
    scanners: list[RuleScanner]


class _ScanOutcome(BaseModel):
    """What the scan process sends back: the report, or why ``scan_traces`` refused."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    report: ScanReport | None = None
    refused: str | None = None


async def scan_in_process(
    traces: Sequence[Trace],
    scanners: Sequence[PatternScanner | ProtectedPathScanner],
    disconnected: Callable[[], Awaitable[bool]],
) -> ScanReport:
    """Run :func:`~deepgym.scanners.scan_traces` in a new Python process and wait for its
    report without blocking the event loop, killing the process once *disconnected* says
    the client left. The process is always reaped, even when this is cancelled.

    Everything crosses as JSON, so *scanners* are sent as their JSON config: a regex keeps
    the flags written inline (``(?i)``), as it does when read from JSON.

    Raises:
        HTTPException: 422 if ``scan_traces`` refuses the scanners (shared or verdict names)
            or the traces, :data:`CLIENT_CLOSED` once the client left, 500 if the process
            ended without a report it could read (with its exit code).
    """
    start = partial(
        subprocess.Popen, SCAN_COMMAND, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        cwd=_SCAN_FOLDER,
    )  # fmt: skip
    process = await anyio.to_thread.run_sync(start)
    left = False
    output = b''

    async def watch() -> None:
        nonlocal left
        try:
            while not await disconnected():
                await anyio.sleep(_POLL_SECONDS)
            left = True
        finally:  # the client left, the scan ended, or the request was cancelled
            process.kill()

    job = _ScanJob(traces=list(traces), scanners=list(scanners))
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(watch)
            output = await anyio.to_thread.run_sync(_communicate, process, job)
            group.cancel_scope.cancel()
    finally:
        if process.poll() is None:
            process.kill()
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(process.wait)
    await anyio.lowlevel.checkpoint_if_cancelled()  # a cancelled request ends here, not as a 500
    if left:
        raise HTTPException(
            status_code=CLIENT_CLOSED, detail='The client closed the request; the scan was stopped'
        )
    try:
        outcome = _ScanOutcome.model_validate_json(output) if process.returncode == 0 else None
    except ValidationError:
        outcome = None
    if outcome is None or (outcome.report is None) is (outcome.refused is None):
        raise HTTPException(
            status_code=500,
            detail=f'The scan process ended with exit code {process.returncode} without a '
            'report it could read',
        )
    if outcome.report is None:
        raise HTTPException(status_code=422, detail=outcome.refused)
    return outcome.report


def _communicate(process: subprocess.Popen[bytes], job: _ScanJob) -> bytes:
    """Send the scan process the paths to import from, one JSON line, then its job as JSON;
    return everything it wrote back."""
    paths = [path for path in sys.path if os.path.isabs(path)]  # a relative one isn't ours
    sent = f'{json.dumps(paths)}\n{job.model_dump_json()}'.encode()
    output, _ = process.communicate(sent)
    return output


def _child() -> None:
    """In the scan process: read the job (JSON) on stdin, and write ``scan_traces``'s report,
    or why it refused the input, as JSON on stdout."""
    out = sys.stdout.buffer
    sys.stdout = sys.stderr  # a stray print must not corrupt the answer
    job = _ScanJob.model_validate_json(sys.stdin.buffer.read())
    try:
        outcome = _ScanOutcome(report=scan_traces(job.traces, job.scanners))
    except ValueError as exc:
        outcome = _ScanOutcome(refused=str(exc))
    out.write(outcome.model_dump_json().encode())
    out.flush()
