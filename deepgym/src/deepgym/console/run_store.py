"""Persist console runs: queued when submitted, then updated by the runner."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from deepgym.console.db import connect, init_db, utc_now
from deepgym.console.schemas import (
    EnvironmentStats,
    ErrorTypeCount,
    RunDetail,
    RunMode,
    RunSpec,
    RunStats,
    RunSummary,
    RunTotals,
)
from deepgym.models import RunResult

_SUMMARY_COLUMNS = (
    'id, group_id, environment, mode, provider, reference, status, score, passed, '
    'error_type, duration_ms, created_at, finished_at'
)
# Columns list() and stats() may filter on; keys are code-defined, values are bound parameters.
_FILTER_COLUMNS = frozenset({'status', 'environment', 'provider', 'group_id'})
_FINISHED = "status IN ('completed', 'failed')"
# Select list of RunTotals; SUM over no rows is NULL, hence the COALESCEs. A finished run
# without a score got no verdict: it counts 0 in mean_score and is left out of the judged_*.
_TOTALS = (
    'COUNT(*) AS runs, '
    f'COALESCE(SUM({_FINISHED}), 0) AS finished, '
    'COALESCE(SUM(passed = 1), 0) AS passed, '
    'COALESCE(SUM(error_type IS NOT NULL), 0) AS errors, '
    f'AVG(CASE WHEN {_FINISHED} THEN COALESCE(score, 0) END) AS mean_score, '
    'AVG(duration_ms) AS mean_duration_ms, '
    f'COALESCE(SUM({_FINISHED} AND score IS NOT NULL), 0) AS judged_count, '
    f'COALESCE(SUM({_FINISHED} AND score IS NULL), 0) AS no_verdict_count, '
    f'AVG(CASE WHEN {_FINISHED} THEN score END) AS judged_mean_score'
)


class RunStore:
    """Runs table of the console database."""

    def __init__(self, path: Path) -> None:
        self._path = path
        init_db(path)

    def create_group(
        self, specs: Sequence[RunSpec], mode: RunMode, provider: str
    ) -> list[RunSummary]:
        """Queue one run per spec under a shared group id.

        Args:
            specs: What each run executes.
            mode: The provider mode the caller asked for.
            provider: Where the runs will execute (``DeepGym.backend``).

        Returns:
            The queued runs, in spec order.
        """
        group_id = uuid4().hex[:12]
        now = utc_now()
        runs = [
            RunSummary(
                id=uuid4().hex[:12],
                group_id=group_id,
                environment=spec.environment,
                mode=mode,
                provider=provider,
                reference=spec.reference,
                status='queued',
                created_at=now,
            )
            for spec in specs
        ]
        with connect(self._path) as conn:
            conn.executemany(
                'INSERT INTO runs (id, group_id, environment, mode, provider, reference, '
                'solution, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [
                    (run.id, group_id, run.environment, mode, provider, spec.reference,
                     spec.solution, 'queued', now)
                    for run, spec in zip(runs, specs)
                ],
            )  # fmt: skip
        return runs

    def start(self, run_id: str) -> None:
        """Mark a queued run as running."""
        with connect(self._path) as conn:
            conn.execute("UPDATE runs SET status = 'running' WHERE id = ?", (run_id,))

    def complete(self, run_id: str, result: RunResult, duration_ms: float) -> None:
        """Record the verifier result of a run that finished.

        A result without a verdict (``RunResult.has_verdict`` False) keeps no score or
        pass: the mean score counts it as 0, the judged totals leave it out, and its error
        type counts it as an error.
        """
        judged = result.has_verdict
        with connect(self._path) as conn:
            conn.execute(
                "UPDATE runs SET status = 'completed', score = ?, passed = ?, error_type = ?, "
                'result = ?, duration_ms = ?, finished_at = ? WHERE id = ?',
                (result.score if judged else None, result.passed if judged else None,
                 result.error_type or (None if judged else 'verifier_error'),
                 result.model_dump_json(), duration_ms, utc_now(), run_id),
            )  # fmt: skip

    def fail(self, run_id: str, error_type: str, error: str, duration_ms: float) -> None:
        """Record a run that raised instead of producing a verifier result; it has no score."""
        with connect(self._path) as conn:
            conn.execute(
                "UPDATE runs SET status = 'failed', error_type = ?, "
                'error = ?, duration_ms = ?, finished_at = ? WHERE id = ?',
                (error_type, error, duration_ms, utc_now(), run_id),
            )

    def fail_unfinished(self) -> int:
        """Fail runs a previous console process left queued or running; return how many.

        Returns:
            Number of runs marked failed with ``error_type='interrupted'``.
        """
        # ponytail: assumes one console process per database file.
        with connect(self._path) as conn:
            cursor = conn.execute(
                "UPDATE runs SET status = 'failed', error_type = 'interrupted', "
                "error = 'The console stopped before this run finished.', finished_at = ? "
                "WHERE status IN ('queued', 'running')",
                (utc_now(),),
            )
            return cursor.rowcount

    def list(self, filters: dict[str, str | None], limit: int = 200) -> list[RunSummary]:
        """Return the newest runs matching every non-None filter.

        Args:
            filters: Column -> value, for status, environment, provider or group_id.
            limit: Maximum number of runs to return.

        Returns:
            Runs, newest first.
        """
        where, params = _where(filters)
        with connect(self._path) as conn:
            rows = conn.execute(
                f'SELECT {_SUMMARY_COLUMNS} FROM runs WHERE {where} '
                'ORDER BY created_at DESC, rowid DESC LIMIT ?',
                (*params, limit),
            ).fetchall()
        return [RunSummary.model_validate(dict(row)) for row in rows]

    def stats(self, filters: dict[str, str | None], hours: int | None) -> RunStats:
        """Aggregate the runs matching every non-None filter, started in the last *hours*.

        Args:
            filters: Column -> value, as for :meth:`list`.
            hours: Window length; the window just before it is totalled too, for deltas.
                None aggregates every run and has no previous window.

        Returns:
            Totals, score histogram, error types and per-environment totals of the window.
        """
        where, params = _where(filters)
        previous: tuple[str, list[str]] | None = None
        if hours is not None:
            start = datetime.now(timezone.utc) - timedelta(hours=hours)
            earlier = start - timedelta(hours=hours)
            previous = (
                f'{where} AND created_at >= ? AND created_at < ?',
                [*params, earlier.isoformat(), start.isoformat()],
            )
            where, params = f'{where} AND created_at >= ?', [*params, start.isoformat()]
        with connect(self._path) as conn:
            environments = conn.execute(
                f'SELECT environment, {_TOTALS} FROM runs WHERE {where} '
                'GROUP BY environment ORDER BY environment',
                params,
            ).fetchall()
            error_types = conn.execute(
                f'SELECT error_type, COUNT(*) AS count FROM runs WHERE {where} '
                'AND error_type IS NOT NULL GROUP BY error_type ORDER BY count DESC, error_type',
                params,
            ).fetchall()
            bins = [0] * 10
            for row in conn.execute(
                'SELECT MIN(CAST(score * 10 AS INTEGER), 9) AS bin, COUNT(*) AS count FROM runs '
                f'WHERE {where} AND {_FINISHED} AND score IS NOT NULL GROUP BY bin',
                params,
            ):
                bins[row['bin']] = row['count']
            return RunStats(
                hours=hours,
                current=_totals(conn, where, params),
                previous=_totals(conn, *previous) if previous else None,
                score_bins=bins,
                error_types=[ErrorTypeCount.model_validate(dict(row)) for row in error_types],
                environments=[EnvironmentStats.model_validate(dict(row)) for row in environments],
            )

    def get(self, run_id: str) -> RunDetail | None:
        """Return one run with its solution and verifier result, or None if unknown."""
        with connect(self._path) as conn:
            row = conn.execute('SELECT * FROM runs WHERE id = ?', (run_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data['result'] = json.loads(data['result']) if data['result'] else None
        return RunDetail.model_validate(data)


def _where(filters: dict[str, str | None]) -> tuple[str, list[str]]:
    """Build a WHERE clause (and its parameters) matching every non-None filter.

    Raises:
        ValueError: If a filter names a column runs can't be filtered by.
    """
    active = {key: value for key, value in filters.items() if value is not None}
    unknown = active.keys() - _FILTER_COLUMNS
    if unknown:
        raise ValueError(f'Cannot filter runs by {sorted(unknown)}')
    return ' AND '.join(f'{key} = ?' for key in active) or '1', list(active.values())


def _totals(conn: sqlite3.Connection, where: str, params: Sequence[str]) -> RunTotals:
    """Total the runs matching *where*."""
    row = conn.execute(f'SELECT {_TOTALS} FROM runs WHERE {where}', params).fetchone()
    return RunTotals.model_validate(dict(row))
