"""``deepgym check``: a fast authoring loop over one environment folder.

Static checks first (files, metadata schema, verifier source); if they hold, the
reference runs twice, an empty and a stub solution run once (a multi-turn folder
replays its reference actions and an agent that quits at once), and the runs'
protocol output is checked. Last, an existing approval.json is checked against the
files. Hack attacks and mutants are left to ``deepgym qa``.

:func:`lint_env` adds advisory warnings about the judge code (see
:mod:`deepgym.verifier_lint`); they never change the result.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from deepgym.factory.approval import approval_check
from deepgym.models import (
    REFERENCE_ACTIONS,
    REFERENCE_FILES,
    EnvMetadata,
    MultiTurnEnvironment,
    RunResult,
)
from deepgym.qa.checks import (
    episode_results,
    run_verdicts,
    single_turn_results,
    single_turn_verdicts,
)
from deepgym.qa.report import QACheck
from deepgym.registry import load_environment_dir
from deepgym.verifier_lint import lint_verifier
from deepgym.verifier_template import detect_verifier_mode

if TYPE_CHECKING:
    from deepgym.core import DeepGym

_RERUNS = 1
"""One extra reference run: enough to catch nondeterminism while authoring."""

_PROTOCOL_ERRORS = frozenset({'verifier_error', 'invalid_score'})
"""error_type values meaning the verifier's output broke the protocol."""


def check_env(env_dir: Path, dg: DeepGym) -> list[QACheck]:
    """Run every authoring check on the environment folder *env_dir*.

    Args:
        env_dir: The folder, as ``deepgym new`` lays it out.
        dg: Client whose backend runs the verifier.

    Returns:
        files, metadata and verifier_source; then, if those pass, reference_passes,
        deterministic, empty_fails, stub_fails (single-turn) and protocol; then approval.
        A desktop env off the Daytona backend reports a single inconclusive
        reference_passes instead of the run checks: nothing here can replay it.
    """
    if (env_dir / 'task.toml').is_file():
        return [_harbor_check(env_dir), approval_check(env_dir)]
    if (env_dir / 'test_patch.diff').is_file():
        details = (
            'image-based repo patch env (test_patch.diff): it runs only in its image; '
            'deepgym check covers verifier.py folders such as deepgym new --kind repo-patch'
        )
        return [QACheck(name='files', status='skip', details=details), approval_check(env_dir)]
    metadata, metadata_check = _metadata_check(env_dir)
    files_check = _files_check(env_dir, metadata)
    checks = [files_check, metadata_check, _verifier_check(env_dir)]
    if all(check.status == 'pass' for check in checks):
        checks += _run_checks(env_dir, dg)
    return [*checks, approval_check(env_dir)]


def _harbor_check(env_dir: Path) -> QACheck:
    """Load a Harbor task folder (task.toml) through its loader; deepgym qa runs it."""
    try:
        load_environment_dir(env_dir)
    except (ValueError, OSError) as exc:
        return QACheck(name='files', status='fail', details=f'Harbor task does not load: {exc}')
    details = 'Harbor task (task.toml) loads; deepgym qa runs its reference and checks'
    return QACheck(name='files', status='pass', details=details)


def _metadata_check(env_dir: Path) -> tuple[EnvMetadata | None, QACheck]:
    """Parse metadata.json against :class:`EnvMetadata`."""
    path = env_dir / 'metadata.json'
    if not path.is_file():
        details = 'metadata.json is missing; it needs at least {"name": "<env name>"}'
        return None, QACheck(name='metadata', status='fail', details=details)
    try:
        metadata = EnvMetadata.model_validate(json.loads(path.read_text(encoding='utf-8')))
    except ValidationError as exc:
        errors = '; '.join(
            f'{".".join(map(str, error["loc"])) or "metadata"}: {error["msg"]}'
            for error in exc.errors()
        )
        if any(error['type'] == 'extra_forbidden' for error in exc.errors()):
            errors += f' (known keys: {", ".join(EnvMetadata.model_fields)})'
        return None, QACheck(name='metadata', status='fail', details=f'metadata.json: {errors}')
    except ValueError as exc:
        details = f'metadata.json is not valid JSON: {exc}'
        return None, QACheck(name='metadata', status='fail', details=details)
    details = f'{metadata.type} env {metadata.name!r}, difficulty {metadata.difficulty}'
    return metadata, QACheck(name='metadata', status='pass', details=details)


def _files_check(env_dir: Path, metadata: EnvMetadata | None) -> QACheck:
    """Check the folder contract: the files every environment needs."""
    problems = []
    task = env_dir / 'task.md'
    if not task.is_file() or not task.read_text(encoding='utf-8').strip():
        problems.append('task.md is missing or empty (it is the prompt the model sees)')
    if not (env_dir / 'verifier.py').is_file():
        problems.append('verifier.py is missing (it scores every solution)')
    if metadata is not None and metadata.type in ('multi-turn', 'computer-use'):
        if not (env_dir / REFERENCE_ACTIONS).is_file():
            problems.append(f'{REFERENCE_ACTIONS} is missing (the episode that must score 1.0)')
    elif not any((env_dir / name).is_file() for name in REFERENCE_FILES):
        problems.append(f'{" or ".join(REFERENCE_FILES)} is missing (it must score 1.0)')
    root = env_dir.resolve()
    for rel in metadata.support_files if metadata else []:
        path = (env_dir / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            problems.append(f'support file {rel!r} is not a file inside the folder')
    if problems:
        return QACheck(name='files', status='fail', details='; '.join(problems))
    return QACheck(name='files', status='pass', details='task, verifier, reference and metadata')


def _verifier_check(env_dir: Path) -> QACheck:
    """Parse verifier.py and name its shape. Says nothing about isolation (see lint_env)."""
    path = env_dir / 'verifier.py'
    if not path.is_file():
        return QACheck(name='verifier_source', status='skip', details='no verifier.py')
    source = path.read_text(encoding='utf-8')
    try:
        ast.parse(source)
    except SyntaxError as exc:
        details = f'verifier.py:{exc.lineno}: {exc.msg}'
        return QACheck(name='verifier_source', status='fail', details=details)
    details = f'{detect_verifier_mode(source)} verifier'
    return QACheck(name='verifier_source', status='pass', details=details)


def lint_env(env_dir: Path) -> list[str]:
    """Advisory warnings about the judge code in *env_dir*: verifier.py and its support.

    Never a check: findings don't change ``deepgym check``'s result, and an empty list
    proves nothing about isolation (see :mod:`deepgym.verifier_lint`).
    """
    path = env_dir / 'verifier.py'
    if not path.is_file():
        return []
    metadata, _ = _metadata_check(env_dir)
    support = {
        rel: (env_dir / rel).read_text(encoding='utf-8')
        for rel in (metadata.support_files if metadata else [])
        if (env_dir / rel).is_file()
    }
    prepended = metadata is not None and metadata.type in ('multi-turn', 'computer-use')
    try:
        return lint_verifier(path.read_text(encoding='utf-8'), support, prepended=prepended)
    except SyntaxError as exc:
        return [f'{exc.filename}:{exc.lineno}: {exc.msg}; the lint could not read it']


def _run_checks(env_dir: Path, dg: DeepGym) -> list[QACheck]:
    """Run the reference, empty and stub solutions (or episodes) and judge them.

    A desktop env off the Daytona backend reports a single inconclusive
    ``reference_passes``: no local session can drive a desktop, so there is no evidence.
    """
    try:
        env = load_environment_dir(env_dir)
    except (ValueError, OSError) as exc:
        return [QACheck(name='load', status='fail', details=str(exc))]
    runs = 1 + _RERUNS
    if isinstance(env, MultiTurnEnvironment):
        try:
            ast.parse(env.final_verifier_code)
        except SyntaxError as exc:
            details = f'verifier.py with its support files prepended: line {exc.lineno}: {exc.msg}'
            return [QACheck(name='load', status='fail', details=details)]
        if env.desktop and dg.backend != 'daytona':
            details = 'a computer-use env needs a desktop: run with --mode daytona'
            return [QACheck(name='reference_passes', status='inconclusive', details=details)]
        results = episode_results(env, env.reference_actions or [], dg, _RERUNS)
        return [*run_verdicts(results, runs), protocol_check(results)]
    results = single_turn_results(env, env.reference_solution or '', dg, _RERUNS)
    return [*single_turn_verdicts(results, _RERUNS), protocol_check(results)]


def protocol_check(results: Sequence[RunResult]) -> QACheck:
    """Check the verifier's output: valid protocol, per-case results, reasons on failures.

    *results* starts with a reference run. Every run must parse as protocol JSON
    with a matching exit code, the reference run must report its cases, and every
    failed case must say why (``error`` or ``actual_summary``).
    """
    broken = [result for result in results if result.error_type in _PROTOCOL_ERRORS]
    if broken:
        details = f'verifier output breaks the protocol: {broken[0].output.strip()}'
        return QACheck(name='protocol', status='fail', details=details)
    if not results[0].cases:
        details = (
            "the reference run reports no per-case results; return a 'cases' list "
            '(id, passed, error) so every failure says why'
        )
        return QACheck(name='protocol', status='fail', details=details)
    silent = [
        case.id or '?'
        for result in results
        for case in result.cases or []
        if not case.passed and not (case.error or case.actual_summary)
    ]
    if silent:
        details = (
            f'failed cases without a reason: {", ".join(silent[:5])}; '
            "set 'error' (or 'actual_summary') on every failed case"
        )
        return QACheck(name='protocol', status='fail', details=details)
    details = f'{len(results[0].cases)} cases per run; every failed case says why'
    return QACheck(name='protocol', status='pass', details=details)
