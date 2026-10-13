"""Approval records: the evidence that an environment folder passed QA as it is now.

``deepgym qa <folder> --approve --by NAME`` writes ``approval.json`` once, read-only:
the full QA report, a sha256 of every file in the folder, the deepgym version and
the time. ``deepgym check`` recomputes the hashes and flags an approval that no
longer matches, so a delivered env pack carries proof of exactly what was tested.

The advisory verifier lint's warnings ride along under ``lint``, marked advisory:
they inform the reader and are never evidence that the verifier isolates the solution.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deepgym import __version__
from deepgym.exceptions import DeepGymError
from deepgym.judge import local_runs_root
from deepgym.qa.report import (
    DIFFICULTY_CHECK,
    MUTANTS_CHECK,
    QACheck,
    QAReport,
    write_atomically,
)
from deepgym.verifier_lint import ADVISORY_NOTE

APPROVAL_FILE = 'approval.json'

_REQUIRED = ('reference_passes', 'deterministic')
"""Checks every approval needs passed: a working reference that scores the same each run."""

_ASKED_FOR = (MUTANTS_CHECK, DIFFICULTY_CHECK)
"""Checks a report holds only when asked for (--mutants, --model); then they must pass too."""


class LintRecord(BaseModel):
    """The advisory verifier lint's warnings, kept for the reader; never safety evidence."""

    model_config = ConfigDict(frozen=True)

    advisory: Literal[True] = True
    note: str = ADVISORY_NOTE
    warnings: list[str]


class Approval(BaseModel):
    """What approval.json holds. Provenance has no defaults, so a record read back never
    gains a time, version or lint result it didn't store."""

    model_config = ConfigDict(frozen=True)

    environment: str
    approved_by: str = Field(min_length=1)
    approved_at: datetime
    deepgym_version: str
    hashes: dict[str, str]
    """Everything the env is made of, by relative POSIX path: '<sha256> <octal mode>'
    for a file, 'directory <octal mode>' for a directory, 'symlink to <target>'."""
    report: QAReport
    lint: LintRecord

    @field_validator('report', mode='before')
    @classmethod
    def _report_keeps_its_time(cls, value: object) -> object:
        """Refuse a stored report without created_at rather than date it now."""
        if isinstance(value, dict) and 'created_at' not in value:
            raise ValueError('the QA report has no created_at')
        return value


def _support_files(env_dir: Path) -> list[str]:
    """The support_files metadata.json lists, or none if it can't be read."""
    try:
        metadata = json.loads((env_dir / 'metadata.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    listed = metadata.get('support_files', []) if isinstance(metadata, dict) else []
    return [rel for rel in listed if isinstance(rel, str)] if isinstance(listed, list) else []


def missing_evidence(report: QAReport) -> list[str]:
    """Return the checks an approval of *report* needs that didn't pass.

    That is reference_passes and deterministic, plus mutants_killed and difficulty when
    the report has them (they were asked for): a skipped one approves nothing.
    """
    status = {check.name: check.status for check in report.checks}
    needed = [*_REQUIRED, *(name for name in _ASKED_FOR if name in status)]
    return [name for name in needed if status.get(name) != 'pass']


def _digest(path: Path) -> str:
    """sha256 of *path*'s bytes and its permission bits in octal (a build copies both)."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f'{digest} {stat.S_IMODE(path.stat().st_mode):o}'


def content_hashes(env_dir: Path) -> dict[str, str]:
    """Hash every file the env is made of: task, verifier, reference, metadata and the rest.

    That is every file under *env_dir*, hidden or not, ``__pycache__`` and nested
    ``.git`` directories included (a loader such as Harbor's uploads whole
    directories), except approval.json and the folder's own top-level ``.git``; plus
    each support file metadata.json lists, even one outside the folder (keyed by its
    listed path). A symlink, to a file or a directory, is recorded as a link, never
    followed: approved folders hold none (see :func:`snapshot`), so adding one stales
    the approval.
    """
    hashes = {}
    for path in sorted(env_dir.rglob('*')):
        rel = path.relative_to(env_dir)
        if rel == Path(APPROVAL_FILE) or rel.parts[0] == '.git':
            continue
        if path.is_symlink():
            hashes[rel.as_posix()] = f'symlink to {os.readlink(path)}'
        elif path.is_dir():
            hashes[rel.as_posix()] = f'directory {stat.S_IMODE(path.stat().st_mode):o}'
        elif path.is_file():
            hashes[rel.as_posix()] = _digest(path)
    root = env_dir.resolve()
    for listed in _support_files(env_dir):
        path = (env_dir / listed).resolve()
        if path.is_file():
            key = path.relative_to(root).as_posix() if path.is_relative_to(root) else listed
            hashes.setdefault(key, _digest(path))
    return hashes


@contextlib.contextmanager
def snapshot(env_dir: Path) -> Iterator[Path]:
    """Yield a private copy of *env_dir* (same folder name) for QA to run on.

    Approval hashes come from this copy, so they describe exactly the files QA tested,
    even if the folder itself changes while QA runs. The copy is what gets checked, so
    nothing changes between the check and the run. It sits under the hidden runs root,
    so no solution QA runs locally can read the reference or verifier in it. Removed
    afterwards.

    Raises:
        DeepGymError: If the folder holds a symlink (QA would read a live file outside
            the copy), or a support file metadata.json lists lies outside the folder:
            an approval is evidence for a self-contained folder.
    """
    with tempfile.TemporaryDirectory(prefix='deepgym_approve_', dir=local_runs_root()) as parent:
        copy = Path(parent, env_dir.resolve().name)
        shutil.copytree(env_dir, copy, symlinks=True)
        (copy / APPROVAL_FILE).unlink(missing_ok=True)
        links = sorted(p.relative_to(copy).as_posix() for p in copy.rglob('*') if p.is_symlink())
        if links:
            raise DeepGymError(
                f'approval needs a self-contained folder without symlinks, but {env_dir} has '
                f'{links}; replace them with the files'
            )
        root = copy.resolve()
        support = _support_files(copy)
        outside = [rel for rel in support if not (copy / rel).resolve().is_relative_to(root)]
        if outside:
            raise DeepGymError(
                f'approval needs a self-contained folder, but support files {outside} are '
                f'outside {env_dir}; move them in'
            )
        yield copy


def approve(
    env_dir: Path, report: QAReport, by: str, hashes: dict[str, str], lint: Sequence[str]
) -> Path:
    """Write a read-only approval.json for *report*, a QA run over the files in *hashes*.

    Args:
        env_dir: The environment folder.
        report: Its QA report; it must have passed.
        by: Who approves.
        hashes: :func:`content_hashes` of the :func:`snapshot` QA ran on.
        lint: The advisory verifier lint's warnings, recorded as advisory.

    Returns:
        Path of the written approval.json.

    Raises:
        DeepGymError: If the report didn't pass or lacks evidence an approval needs (see
            :func:`missing_evidence`), the files differ from what QA tested, or an
            approval already exists (approvals are never overwritten).
    """
    if report.status != 'pass':
        raise DeepGymError(f'QA result is {report.status}; only a passing report can be approved')
    if missing := missing_evidence(report):
        raise DeepGymError(f'an approval needs {", ".join(missing)} to pass; they did not run')
    if content_hashes(env_dir) != hashes:
        raise DeepGymError(f'files in {env_dir} differ from what QA tested; run QA again')
    path = env_dir / APPROVAL_FILE
    approval = Approval(
        environment=report.environment,
        approved_by=by,
        approved_at=datetime.now(timezone.utc),
        deepgym_version=__version__,
        hashes=hashes,
        report=report,
        lint=LintRecord(warnings=list(lint)),
    )
    try:
        write_atomically(path, approval.model_dump_json(indent=2) + '\n', exclusive=True)
    except FileExistsError as exc:
        raise DeepGymError(
            f'{path} already exists and approvals are never overwritten; '
            'delete it to approve the current files'
        ) from exc
    path.chmod(0o444)
    return path


def approval_check(env_dir: Path) -> QACheck:
    """Check that approval.json, if any, still matches every file in *env_dir*."""
    path = env_dir / APPROVAL_FILE
    if not path.is_file():
        details = f'not approved yet: deepgym qa {env_dir} --approve --by NAME once it passes'
        return QACheck(name='approval', status='skip', details=details)
    try:
        approval = Approval.model_validate_json(path.read_text(encoding='utf-8'))
    except ValueError as exc:
        return QACheck(name='approval', status='fail', details=f'{APPROVAL_FILE} is invalid: {exc}')
    if approval.report.status != 'pass':
        details = (
            f'{APPROVAL_FILE} holds a QA report that did not pass ({approval.report.status}), '
            'so it approves nothing'
        )
        return QACheck(name='approval', status='fail', details=details)
    if missing := missing_evidence(approval.report):
        details = f'{APPROVAL_FILE} holds a report where {", ".join(missing)} did not pass'
        return QACheck(name='approval', status='fail', details=details)
    now, then = content_hashes(env_dir), approval.hashes
    changes = {
        'changed': sorted(name for name in then.keys() & now.keys() if then[name] != now[name]),
        'added': sorted(now.keys() - then.keys()),
        'removed': sorted(then.keys() - now.keys()),
    }
    if any(changes.values()):
        listed = '; '.join(f'{kind}: {", ".join(rels)}' for kind, rels in changes.items() if rels)
        details = (
            f'{APPROVAL_FILE} no longer matches the files ({listed}); re-run QA, delete '
            f'{APPROVAL_FILE} and approve again'
        )
        return QACheck(name='approval', status='fail', details=details)
    details = (
        f'approved by {approval.approved_by} on {approval.approved_at:%Y-%m-%d} with deepgym '
        f'{approval.deepgym_version}; all {len(now)} file hashes match'
    )
    if approval.report.not_run:
        details += f'; checks that did not run: {", ".join(approval.report.not_run)}'
    return QACheck(name='approval', status='pass', details=details)
