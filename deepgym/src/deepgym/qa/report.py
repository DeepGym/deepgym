"""QA report model, per-environment JSON files, and the CLI table."""

from __future__ import annotations

import contextlib
import os
import tempfile
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

MUTANTS_CHECK = 'mutants_killed'
"""Name of the mutation-testing check (see :mod:`deepgym.qa.mutants`)."""

DIFFICULTY_CHECK = 'difficulty'
"""Name of the model difficulty probe's check (see :mod:`deepgym.qa.difficulty`)."""

_ALWAYS_NOTED = frozenset({DIFFICULTY_CHECK, MUTANTS_CHECK})
"""Checks whose details the table prints even when they pass: they carry a score."""

_COLUMNS = {
    'reference_passes': 'ref',
    'deterministic': 'det',
    'empty_fails': 'empty',
    'stub_fails': 'stub',
    'busy_fails': 'busy',
    'random_fails': 'rand',
    'hack_resistant': 'hack',
    MUTANTS_CHECK: 'mut',
    DIFFICULTY_CHECK: 'diff',
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

    @computed_field
    @property
    def ran(self) -> list[str]:
        """Checks that ran: each passed, failed or couldn't decide."""
        return [check.name for check in self.checks if check.status != 'skip']

    @computed_field
    @property
    def not_run(self) -> list[str]:
        """Checks that were skipped: they count neither way, so a pass never rests on them."""
        return [check.name for check in self.checks if check.status == 'skip']


def write_atomically(path: Path, text: str, *, exclusive: bool = False) -> None:
    """Write *text* to *path* so readers see the old file or the whole new one, never half.

    The text goes to a temporary file in the same directory, flushed to disk, then
    takes *path*'s place in one step. With *exclusive*, an existing *path* is never
    replaced.

    Raises:
        FileExistsError: If *exclusive* and *path* already exists.
    """
    with tempfile.NamedTemporaryFile(
        'w', dir=path.parent, prefix=f'.{path.name}.', delete=False, encoding='utf-8'
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink()
            raise
    try:
        if exclusive:
            os.link(temporary, path)  # fails if path exists, and is atomic when it doesn't
        else:
            os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def report_dir() -> Path:
    """Directory QA reports live in: ``$DEEPGYM_QA_DIR``, default ``~/.deepgym/qa``."""
    return Path(os.environ.get('DEEPGYM_QA_DIR') or Path.home() / '.deepgym' / 'qa')


def write_report(report: QAReport) -> Path:
    """Write *report* to ``<report_dir()>/<env>.json``.

    Slashes in the environment name become underscores in the file name. The write is
    atomic (see :func:`write_atomically`): a crash or a concurrent writer never leaves a
    truncated report.

    Returns:
        Path of the written file.
    """
    directory = report_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{report.environment.replace("/", "_")}.json'
    write_atomically(path, report.model_dump_json(indent=2) + '\n')
    return path


def format_table(reports: Sequence[QAReport]) -> str:
    """Render one row per environment, then details of undecided, failed and scored checks."""
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
        noted = (
            ('fail', 'inconclusive')
            if report.status == 'skip'
            else ('fail', 'inconclusive', 'skip')
        )
        notes.extend(
            f'{report.environment}: {check.name}: '
            + ('not run: ' if check.status == 'skip' else '')
            + check.details
            for check in report.checks
            if check.status in noted or check.name in _ALWAYS_NOTED
        )
    table = '\n'.join('  '.join(f'{cell:<5}' for cell in row).rstrip() for row in rows)
    statuses = [report.status for report in reports]
    summary = (
        f'{statuses.count("pass")} passed, {statuses.count("fail")} failed, '
        f'{statuses.count("inconclusive")} inconclusive, {statuses.count("skip")} skipped'
    )
    return '\n\n'.join(section for section in (table, '\n'.join(notes), summary) if section)


def format_checks(checks: Sequence[QACheck]) -> str:
    """Render one environment's checks, one line each, then the combined result."""
    width = max(len(check.name) for check in checks)
    lines = [f'{_CELLS[c.status]:<5} {c.name.ljust(width)}  {c.details}' for c in checks]
    return '\n'.join([*lines, '', f'result: {overall(c.status for c in checks).upper()}'])
