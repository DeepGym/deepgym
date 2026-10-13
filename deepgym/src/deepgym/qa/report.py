"""QA report model, per-environment JSON files, and the CLI table."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

QAStatus = Literal['pass', 'fail', 'inconclusive', 'skip']
"""A verdict. 'inconclusive': the check ran but couldn't decide (e.g. an attack crashed).
'skip': the check doesn't apply; it's neutral, never a pass."""

SKIPPED = 'skipped'
"""Name of the single check a report carries when the environment isn't checkable."""

_COLUMNS = {
    'reference_passes': 'ref',
    'deterministic': 'det',
    'empty_fails': 'empty',
    'stub_fails': 'stub',
    'hack_resistant': 'hack',
    'difficulty': 'diff',
}
_CELLS: dict[QAStatus, str] = {'pass': 'ok', 'fail': 'FAIL', 'inconclusive': '?', 'skip': 'skip'}


def verdict(ok: bool) -> QAStatus:
    """Return 'pass' if *ok*, else 'fail'."""
    return 'pass' if ok else 'fail'


def overall(statuses: Iterable[QAStatus]) -> QAStatus:
    """Combine verdicts: any fail fails, then any inconclusive; skips count only if alone."""
    found = set(statuses)
    return next((s for s in ('fail', 'inconclusive', 'pass') if s in found), 'skip')


class QACheck(BaseModel):
    """Outcome of one quality check on one environment."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: QAStatus
    details: str

    @computed_field
    @property
    def passed(self) -> bool:
        """True only when the check passed."""
        return self.status == 'pass'


class QAReport(BaseModel):
    """Quality-gate verdict for one environment, saved as ``<env>.json``."""

    model_config = ConfigDict(frozen=True)

    environment: str
    checks: list[QACheck]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def status(self) -> QAStatus:
        """The environment's verdict: see :func:`overall`."""
        return overall(check.status for check in self.checks)

    @computed_field
    @property
    def passed(self) -> bool:
        """True only on a pass: not when it failed, was inconclusive or was skipped."""
        return self.status == 'pass'


def report_dir() -> Path:
    """Directory QA reports live in: ``$DEEPGYM_QA_DIR``, default ``~/.deepgym/qa``."""
    return Path(os.environ.get('DEEPGYM_QA_DIR') or Path.home() / '.deepgym' / 'qa')


def write_report(report: QAReport) -> Path:
    """Write *report* to ``<report_dir()>/<env>.json``.

    Slashes in the environment name become underscores in the file name.

    Returns:
        Path of the written file.
    """
    directory = report_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{report.environment.replace("/", "_")}.json'
    path.write_text(report.model_dump_json(indent=2) + '\n', encoding='utf-8')
    return path


def format_table(reports: Sequence[QAReport]) -> str:
    """Render one row per environment, then details of undecided, failed and difficulty checks."""
    width = max(len('environment'), *(len(report.environment) for report in reports))
    rows = [['environment'.ljust(width), *_COLUMNS.values(), 'result']]
    notes: list[str] = []
    for report in reports:
        by_name = {check.name: check for check in report.checks}
        rows.append(
            [
                report.environment.ljust(width),
                *(_CELLS[by_name[name].status] if name in by_name else '-' for name in _COLUMNS),
                report.status.upper(),
            ]
        )
        notes.extend(
            f'{report.environment}: {check.name}: {check.details}'
            for check in report.checks
            if check.status in ('fail', 'inconclusive') or check.name == 'difficulty'
        )
    table = '\n'.join('  '.join(f'{cell:<5}' for cell in row).rstrip() for row in rows)
    statuses = [report.status for report in reports]
    summary = (
        f'{statuses.count("pass")} passed, {statuses.count("fail")} failed, '
        f'{statuses.count("inconclusive")} inconclusive, {statuses.count("skip")} skipped'
    )
    return '\n\n'.join(section for section in (table, '\n'.join(notes), summary) if section)
