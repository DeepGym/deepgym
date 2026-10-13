"""Tests for deepgym.factory: deepgym new, deepgym check and approval records."""

import ast
import json
import runpy
import shutil
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

import deepgym.qa
from deepgym import __version__, solution_runtime
from deepgym.cli import main
from deepgym.core import DeepGym
from deepgym.exceptions import DeepGymError
from deepgym.factory.approval import (
    APPROVAL_FILE,
    Approval,
    approval_check,
    approve,
    content_hashes,
    missing_evidence,
    snapshot,
)
from deepgym.factory.check import check_env, lint_env, protocol_check
from deepgym.factory.scaffold import KINDS, scaffold
from deepgym.integrations.zai import ZaiChatClient
from deepgym.judge import local_runs_root
from deepgym.models import CaseResult, RunResult
from deepgym.qa import QACheck, QAOptions, QAReport, format_table, run_qa
from deepgym.registry import load_environment_dir
from deepgym.verifier_lint import ADVISORY_NOTE, lint_verifier

_FILES = {
    'function': {'task.md', 'verifier.py', 'reference_solution.py', 'variants.py', 'metadata.json'},
    'script': {'task.md', 'verifier.py', 'reference_solution.py', 'variants.py', 'metadata.json'},
    'repo-patch': {
        'task.md',
        'verifier.py',
        'reference_solution.diff',
        'variants.py',
        'metadata.json',
        'repo/runs.py',
    },
    'multi-turn': {
        'task.md',
        'verifier.py',
        'reference_actions.json',
        'variants.py',
        'metadata.json',
        'runs.py',
    },
}

_UNSAFE_CALLS = {'exec', 'eval', 'compile', '__import__'}


def _files(path: Path) -> set[str]:
    return {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}


def _statuses(checks: list[QACheck]) -> dict[str, str]:
    return {check.name: check.status for check in checks}


def _cli(*args: str) -> int:
    """Run the deepgym CLI; return its exit code (0 when it returns normally)."""
    with patch.object(sys, 'argv', ['deepgym', *args]):
        try:
            main()
        except SystemExit as exc:
            return int(exc.code or 0)
    return 0


# ---------------------------------------------------------------------------
# deepgym new
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', KINDS)
def test_scaffold_renders_every_template(kind: str, tmp_path: Path) -> None:
    path = scaffold('word_stats', kind, tmp_path)

    assert path == tmp_path / 'word_stats'
    assert _files(path) == _FILES[kind]
    assert (path / 'task.md').read_text(encoding='utf-8').startswith('# Word Stats\n')
    assert json.loads((path / 'metadata.json').read_text(encoding='utf-8'))['name'] == 'word_stats'
    assert not any('$' in (path / rel).read_text(encoding='utf-8') for rel in _files(path))


@pytest.mark.parametrize('kind', KINDS)
def test_generated_verifier_never_imports_or_runs_the_solution(kind: str, tmp_path: Path) -> None:
    tree = ast.parse((scaffold('env', kind, tmp_path) / 'verifier.py').read_text(encoding='utf-8'))

    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert 'deepgym_solution' in imported
    assert not imported & {'importlib', 'runpy', 'solution', 'runs'}
    assert not calls & _UNSAFE_CALLS
    assert lint_env(tmp_path / 'env') == []


@pytest.mark.parametrize('name', ['Bad', '1env', 'my-env', ''])
def test_scaffold_rejects_names_that_are_not_snake_case(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='snake_case'):
        scaffold(name, 'function', tmp_path)


def test_scaffold_never_overwrites_a_folder(tmp_path: Path) -> None:
    path = scaffold('env', 'function', tmp_path)
    (path / 'task.md').write_text('mine', encoding='utf-8')

    with pytest.raises(ValueError, match='already exists'):
        scaffold('env', 'script', tmp_path)
    assert (path / 'task.md').read_text(encoding='utf-8') == 'mine'


def test_cli_new_lists_the_files(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert _cli('new', 'word_stats', '--kind', 'script', '--dir', str(tmp_path)) == 0

    out = capsys.readouterr().out
    assert f'Created {tmp_path / "word_stats"}:' in out
    assert '  verifier.py\n' in out
    assert f'deepgym check {tmp_path / "word_stats"}' in out
    assert _cli('new', 'word_stats', '--kind', 'script', '--dir', str(tmp_path)) == 1


@pytest.mark.parametrize('command', ['new', 'check'])
def test_cli_help(command: str, capsys: pytest.CaptureFixture) -> None:
    assert _cli(command, '--help') == 0
    assert 'usage: deepgym' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# deepgym check
# ---------------------------------------------------------------------------


def _no_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PATH', '')


def _disk_full(monkeypatch: pytest.MonkeyPatch) -> None:
    def mkdtemp(*args: object, **kwargs: object) -> str:
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(solution_runtime.tempfile, 'mkdtemp', mkdtemp)


def _launcher_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DEEPGYM_PYTHON', '/nonexistent/python')  # started, never ready


@pytest.mark.parametrize('kind', ['function', 'script', 'repo-patch', 'multi-turn'])
@pytest.mark.parametrize('breakage', [_no_processes, _disk_full, _launcher_fails])
def test_templates_end_the_run_when_the_solution_cannot_be_started(
    kind: str,
    breakage: Callable[[pytest.MonkeyPatch], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = scaffold('env', kind, tmp_path)
    for support in (path / 'repo').glob('*'):  # the judge finds support files by name
        shutil.copy(support, path / support.name)
    monkeypatch.setitem(sys.modules, 'deepgym_solution', solution_runtime)
    monkeypatch.delitem(sys.modules, 'variants', raising=False)
    monkeypatch.syspath_prepend(str(path))
    monkeypatch.chdir(path)
    verify = runpy.run_path(str(path / 'verifier.py'))['verify']
    reference = [str(next(path.glob('reference_solution.*')))] if kind != 'multi-turn' else []
    breakage(monkeypatch)  # the machine's fault, never the solution's 0

    with pytest.raises(solution_runtime.LaunchError):
        verify(*reference)


@pytest.mark.parametrize('kind', KINDS)
def test_every_scaffold_passes_check(kind: str, tmp_path: Path, local_dg: DeepGym) -> None:
    statuses = _statuses(check_env(scaffold('env', kind, tmp_path), local_dg))

    assert statuses.pop('approval') == 'skip'
    single_turn = {'stub_fails': 'skip' if kind == 'repo-patch' else 'pass'}
    assert statuses == {
        'files': 'pass',
        'metadata': 'pass',
        'verifier_source': 'pass',
        'reference_passes': 'pass',
        'deterministic': 'pass',
        'empty_fails': 'pass',
        **({} if kind == 'multi-turn' else single_turn),
        'protocol': 'pass',
    }


def test_check_reports_missing_files_and_bad_metadata(tmp_path: Path, local_dg: DeepGym) -> None:
    path = scaffold('env', 'function', tmp_path)
    (path / 'reference_solution.py').unlink()
    (path / 'metadata.json').write_text('{"name": "env", "dificulty": "easy"}', encoding='utf-8')

    checks = {check.name: check for check in check_env(path, local_dg)}

    assert checks['files'].status == 'fail'
    assert 'reference_solution.py or reference_solution.diff is missing' in checks['files'].details
    assert checks['metadata'].status == 'fail'
    assert 'dificulty: Extra inputs are not permitted' in checks['metadata'].details
    assert 'known keys: id, name, type, difficulty' in checks['metadata'].details
    assert 'reference_passes' not in checks


_LINTED = (
    'import os, subprocess, sys\n'
    'from deepgym_solution import load_solution\n\n'
    'def verify(solution_path, test_cases_path=None):\n'
    '    load_solution(solution_path)\n'
    '    {body}\n'
    '    return 1.0\n'
)


@pytest.mark.parametrize(
    ('body', 'found'),
    [
        # The documented dodges: each must come back as a warning, never silently pass.
        ('getattr(__import__("sub" + "process"), "run")([solution_path])', 'uses __import__'),
        ('import importlib; importlib.import_module("sol" + "ution")', 'imports importlib'),
        ('import base64; exec(base64.b64decode(b"cHJpbnQoMSk="))', 'uses exec'),
        ('import runpy; runpy.run_path(solution_path)', 'imports runpy'),
        ('code = compile(source, "s", "exec")', 'uses compile'),
        ('subprocess.run([sys.executable, solution_path])', 'uses subprocess.run'),
        ('exec(open(solution_path).read())', 'passes the solution path to open()'),
        ('text = open(solution_path).read()', 'passes the solution path to open()'),
        ('import pathlib; pathlib.Path(solution_path).read_text()', 'solution path to Path()'),
        ('import imp; imp.load_source("s", solution_path)', 'imports imp'),
        # Aliases and indirection.
        ('run_it = eval', 'uses eval'),
        ('sys.path.insert(0, os.path.dirname(solution_path))', 'uses sys.path'),
        ('import solution', 'imports solution'),
        ('import subprocess as sp; sp.call([solution_path])', 'uses subprocess.call'),
        ('from subprocess import Popen as P; P([solution_path])', 'uses subprocess.Popen'),
        ('import builtins; builtins.exec("")', 'uses builtins.exec'),
        ('os.execv(sys.executable, [solution_path])', 'uses os.execv'),
        ('import asyncio; asyncio.create_subprocess_exec(solution_path)', 'uses asyncio.create'),
        ('getattr(subprocess, "r" + "un")([solution_path])', 'passes subprocess around'),
        ('sp = subprocess; sp.run([solution_path])', 'passes subprocess around'),
        ('sys.modules["subprocess"].run([solution_path])', 'uses sys.modules'),
        ('import asyncio; asyncio.get_event_loop().subprocess_exec()', 'uses subprocess_exec'),
        ('import timeit; timeit.timeit("pass")', 'imports timeit'),
        ('from cProfile import run as r; r("")', 'imports cProfile'),
        ('from logging import config', 'imports config from logging'),
    ],
)
def test_the_lint_warns_about_documented_dodges(body: str, found: str) -> None:
    warnings = lint_verifier(_LINTED.format(body=body), {}, prepended=False)

    assert any(found in warning for warning in warnings), warnings


def test_the_lint_wants_the_judge_to_use_a_deepgym_solution_helper() -> None:
    reads_text = 'def verify(solution_path, test_cases_path=None):\n    return 1.0\n'

    assert lint_verifier(reads_text, {}, prepended=False) == [
        'verifier.py never calls a deepgym_solution helper '
        '(load_files, load_solution, run_solution, solution_source)'
    ]
    assert lint_verifier(_LINTED.format(body='pass'), {}, prepended=False) == []


def test_check_prints_lint_warnings_without_failing(
    tmp_path: Path, local_dg: DeepGym, capsys: pytest.CaptureFixture
) -> None:
    path = scaffold('env', 'function', tmp_path)
    verifier = path / 'verifier.py'
    docstring = '    """Score the solution at *solution_path* on every generated case."""\n'
    source = verifier.read_text()
    assert docstring in source
    verifier.write_text(source.replace(docstring, docstring + '    open(solution_path).close()\n'))

    statuses = _statuses(check_env(path, local_dg))

    assert set(statuses.values()) == {'pass', 'skip'}  # warnings never change the result
    assert _cli('check', str(path)) == 0
    out = capsys.readouterr().out
    assert f'warnings ({ADVISORY_NOTE})' in out
    assert 'passes the solution path to open()' in out


def test_an_alias_shadowed_in_another_scope_still_counts() -> None:
    verifier = (
        'import subprocess as sp\n\n'
        'def helper():\n    import math as sp\n\n'
        'def verify(solution_path, test_cases_path=None):\n    sp.run([solution_path])\n'
    )

    warnings = lint_verifier(verifier, {}, prepended=False)

    assert {'verifier.py:1 imports subprocess as sp', 'verifier.py:7 uses subprocess.run'} <= set(
        warnings
    )


@pytest.mark.parametrize(
    ('helper', 'found'),
    [
        ('from subprocess import run\n', 'helper.py:1 imports run from subprocess'),
        ('from subprocess import *\n', 'helper.py:1 imports * from subprocess'),
        ('from builtins import exec as run\n', 'helper.py:1 imports exec from builtins'),
        ('import subprocess as run\n', 'helper.py:1 imports subprocess as run'),
        ('from asyncio import subprocess as run\n', 'helper.py:1 imports subprocess from asyncio'),
    ],
)
def test_launchers_re_exported_by_a_support_module_count(helper: str, found: str) -> None:
    verifier = 'from helper import run\n\ndef verify(solution_path):\n    run([solution_path])\n'

    assert found in lint_verifier(verifier, {'helper.py': helper}, prepended=False)


def test_the_lint_follows_imported_packages_but_not_fixtures() -> None:
    verifier = 'from helpers import load\n'
    support = {
        'helpers/__init__.py': 'from . import loader\nload = loader.load\n',
        'helpers/loader.py': 'import importlib\n',
        'repo/cli.py': 'import subprocess\nsubprocess.run(["ls"])\n',  # solution-side fixture
    }

    warnings = lint_verifier(verifier, support, prepended=False)

    assert 'helpers/loader.py:1 imports importlib' in warnings
    assert not any('repo/cli.py' in warning for warning in warnings)
    assert 'repo/cli.py:2 uses subprocess.run' in lint_verifier(verifier, support, prepended=True)


def test_prepended_support_files_are_linted_whatever_their_name() -> None:
    verifier = 'from deepgym_solution import load_solution\nload_solution("runs.py")\n'
    support = {'helper.txt': 'exec(open("runs.py").read())\n'}

    assert 'helper.txt:1 uses exec' in lint_verifier(verifier, support, prepended=True)


def test_lint_env_covers_support_modules_the_verifier_imports(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    path = scaffold('env', 'repo-patch', tmp_path)
    variants = path / 'variants.py'
    variants.write_text('import importlib.util\n' + variants.read_text(), encoding='utf-8')
    fixture = path / 'repo' / 'runs.py'  # read as text by the judge, run by the solution
    fixture.write_text('import subprocess\n' + fixture.read_text(), encoding='utf-8')

    warnings = lint_env(path)

    assert warnings == ['variants.py:1 imports importlib.util']
    assert {c.name: c for c in check_env(path, local_dg)}['verifier_source'].status == 'pass'


@pytest.mark.parametrize('kind', ['function', 'repo-patch'])
def test_a_solution_whose_entry_cannot_load_scores_zero(
    kind: str, tmp_path: Path, local_dg: DeepGym
) -> None:
    path = scaffold('env', kind, tmp_path)
    env = load_environment_dir(path)
    loop = 'longest_run = []; longest_run.append(longest_run)\n'  # imports, can't be sent
    patch = f'--- a/runs.py\n+++ b/runs.py\n@@ -13,0 +14,1 @@\n+{loop}'  # after the def
    answer = loop if kind == 'function' else patch

    result = local_dg.run(env, answer)

    assert (result.score, result.has_verdict) == (0.0, True)
    assert 'failed to load' in result.output


def test_check_fails_a_wrong_reference(tmp_path: Path, local_dg: DeepGym) -> None:
    path = scaffold('env', 'function', tmp_path)
    reference = path / 'reference_solution.py'
    reference.write_text(reference.read_text().replace('value > previous', 'value >= previous'))

    check = {c.name: c for c in check_env(path, local_dg)}['reference_passes']

    assert check.status == 'fail'
    assert 'cases passed' in check.details


def test_check_marks_a_verifier_error_on_the_empty_solution_inconclusive(
    tmp_path: Path, local_dg: DeepGym, capsys: pytest.CaptureFixture
) -> None:
    path = scaffold('env', 'function', tmp_path)
    (path / 'verifier.py').write_text(
        'import os\n'
        'from deepgym_solution import load_solution\n\n'
        'def verify(solution_path, test_cases_path=None):\n'
        '    if not os.path.getsize(solution_path):  # a bug when the file is empty\n'
        "        raise IndexError('no first line')\n"
        '    got = load_solution(solution_path).longest_run([1, 2, 3])\n'
        "    case = {'id': 'c0', 'passed': got == 3, 'actual_summary': repr(got)}\n"
        "    return {'score': float(got == 3), 'passed': got == 3, 'cases': [case]}\n",
        encoding='utf-8',
    )

    statuses = _statuses(check_env(path, local_dg))

    # The verifier raises on the empty solution before running it: no verdict.
    assert statuses['empty_fails'] == 'inconclusive'
    assert statuses['stub_fails'] == 'pass'
    assert _cli('check', str(path)) == 2
    assert 'the verifier errored instead of judging the solution' in capsys.readouterr().out


def test_check_needs_per_case_results(tmp_path: Path, local_dg: DeepGym) -> None:
    path = scaffold('env', 'function', tmp_path)
    verifier = path / 'verifier.py'
    verifier.write_text(verifier.read_text().replace("        'cases': cases,\n", ''))

    check = {c.name: c for c in check_env(path, local_dg)}['protocol']

    assert check.status == 'fail'
    assert "return a 'cases' list" in check.details


def _result(**kwargs: object) -> RunResult:
    defaults = {
        'score': 0.0,
        'passed': False,
        'output': '',
        'stderr': '',
        'exit_code': 1,
        'execution_time_ms': 1.0,
        'sandbox_id': 'local',
    }
    return RunResult(**{**defaults, **kwargs})


def test_protocol_check_flags_broken_output_and_silent_failures() -> None:
    good = _result(score=1.0, passed=True, exit_code=0, cases=[CaseResult(id='a', passed=True)])
    silent = _result(cases=[CaseResult(id='b', passed=False)])
    broken = _result(error_type='verifier_error', output='Verifier did not produce valid JSON')

    assert protocol_check([good, silent]).details.startswith('failed cases without a reason: b')
    assert 'breaks the protocol: Verifier did not produce' in protocol_check([good, broken]).details
    assert protocol_check([good, _result(cases=[])]).status == 'pass'


def test_multi_turn_verifiers_keep_future_imports_first(tmp_path: Path, local_dg: DeepGym) -> None:
    path = scaffold('env', 'multi-turn', tmp_path)
    verifier = path / 'verifier.py'
    verifier.write_text('from __future__ import annotations\n\n' + verifier.read_text())

    statuses = _statuses(check_env(path, local_dg))

    assert statuses['reference_passes'] == 'pass'
    assert 'load' not in statuses


def test_check_rejects_invalid_reference_actions(tmp_path: Path, local_dg: DeepGym) -> None:
    path = scaffold('env', 'multi-turn', tmp_path)
    (path / 'reference_actions.json').write_text('[{"action_type": "fly"}]', encoding='utf-8')

    check = {c.name: c for c in check_env(path, local_dg)}['load']

    assert check.status == 'fail'
    assert check.details.startswith('reference_actions.json: ')


def test_check_skips_image_based_repo_patch_folders(tmp_path: Path, local_dg: DeepGym) -> None:
    (tmp_path / 'test_patch.diff').write_text('', encoding='utf-8')

    assert _statuses(check_env(tmp_path, local_dg)) == {'files': 'skip', 'approval': 'skip'}


def test_cli_check_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = scaffold('env', 'function', tmp_path)

    assert _cli('check', str(path)) == 0
    assert capsys.readouterr().out.rstrip().endswith('result: PASS')
    (path / 'task.md').write_text('', encoding='utf-8')
    assert _cli('check', str(path)) == 1
    assert _cli('check', str(tmp_path / 'missing')) == 1


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------


def test_qa_approve_writes_a_read_only_record_that_check_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_dg: DeepGym
) -> None:
    monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path / 'reports'))
    path = scaffold('env', 'function', tmp_path)

    assert _cli('qa', str(path), '--reruns', '1', '--approve', '--by', 'Jane Doe') == 0

    record = path / APPROVAL_FILE
    data = json.loads(record.read_text(encoding='utf-8'))
    assert stat.S_IMODE(record.stat().st_mode) == 0o444
    assert (data['approved_by'], data['deepgym_version']) == ('Jane Doe', __version__)
    assert set(data['hashes']) == _FILES['function']
    assert (data['report']['environment'], data['report']['status']) == ('env', 'pass')
    assert data['lint'] == {'advisory': True, 'note': ADVISORY_NOTE, 'warnings': []}
    assert approval_check(path).status == 'pass'

    (path / 'verifier.py').write_text('# changed\n', encoding='utf-8')
    (path / 'notes.txt').write_text('new', encoding='utf-8')
    stale = approval_check(path)
    assert stale.status == 'fail'
    assert '(changed: verifier.py; added: notes.txt)' in stale.details


def test_qa_json_output_stays_parseable_when_approving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path / 'reports'))
    path = scaffold('env', 'function', tmp_path)

    assert _cli('qa', str(path), '--reruns', '1', '--json', '--approve', '--by', 'Jane') == 0

    out = capsys.readouterr()
    assert json.loads(out.out)[0]['status'] == 'pass'
    assert f'Approved: {path / APPROVAL_FILE}' in out.err


def test_qa_does_not_approve_a_failing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path / 'reports'))
    path = scaffold('env', 'function', tmp_path)
    (path / 'reference_solution.py').write_text('def longest_run(values):\n    return 1\n')

    assert _cli('qa', str(path), '--reruns', '1', '--approve', '--by', 'Jane') == 1

    assert 'Not approved: QA result is fail' in capsys.readouterr().err
    assert not (path / APPROVAL_FILE).exists()


@pytest.mark.parametrize(
    ('others', 'flags'), [([], ['--approve']), (['reverse_string'], ['--approve', '--by', 'J'])]
)
def test_qa_approve_needs_a_name_and_one_folder(
    others: list[str], flags: list[str], tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = scaffold('env', 'function', tmp_path)

    assert _cli('qa', str(path), *others, *flags) == 1
    assert '--approve needs --by NAME and exactly one environment folder' in capsys.readouterr().err


def test_qa_approve_needs_a_rerun_to_check_determinism(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = scaffold('env', 'function', tmp_path)

    assert _cli('qa', str(path), '--reruns', '0', '--approve', '--by', 'Jane') == 1

    assert '--approve needs --reruns 1 or more' in capsys.readouterr().err
    assert not (path / APPROVAL_FILE).exists()


def test_qa_approves_nothing_when_a_requested_check_did_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path / 'reports'))
    path = scaffold('env', 'multi-turn', tmp_path)  # a list of actions has no mutants

    assert _cli('qa', str(path), '--reruns', '1', '--mutants', '--approve', '--by', 'J') == 1

    assert 'Not approved: needed checks did not run: mutants_killed' in capsys.readouterr().err
    assert not (path / APPROVAL_FILE).exists()


def _passing_report(*extra: QACheck) -> QAReport:
    names = ('reference_passes', 'deterministic')
    checks = [QACheck(name=name, status='pass', details='') for name in names]
    return QAReport(environment='env', checks=[*checks, *extra])


@pytest.mark.parametrize(
    ('checks', 'missing'),
    [
        ([QACheck(name='reference_passes', status='pass', details='')], 'deterministic'),
        ([QACheck(name='mutants_killed', status='skip', details='')], 'mutants_killed'),
        ([QACheck(name='difficulty', status='skip', details='')], 'difficulty'),
    ],
)
def test_approvals_need_determinism_and_every_check_asked_for(
    checks: list[QACheck], missing: str, tmp_path: Path
) -> None:
    path = scaffold('env', 'function', tmp_path)
    report = QAReport(environment='env', checks=checks)
    if checks[0].name != 'reference_passes':
        report = _passing_report(*checks)  # passing, but an asked-for check didn't run
    assert report.passed

    with pytest.raises(DeepGymError, match=f'an approval needs {missing} to pass'):
        approve(path, report, 'Jane', content_hashes(path), [])
    record = Approval(
        environment='env',
        approved_by='Jane',
        approved_at='2026-10-13T00:00:00Z',
        deepgym_version=__version__,
        hashes=content_hashes(path),
        report=report,
        lint={'warnings': []},
    )
    (path / APPROVAL_FILE).write_text(record.model_dump_json(), encoding='utf-8')
    check = approval_check(path)
    assert check.status == 'fail'
    assert f'holds a report where {missing} did not pass' in check.details


def test_approve_refuses_failures_changes_and_overwrites(tmp_path: Path) -> None:
    path = scaffold('env', 'function', tmp_path)
    hashes = content_hashes(path)
    failed = QAReport(environment='env', checks=[QACheck(name='c', status='fail', details='')])

    with pytest.raises(DeepGymError, match='only a passing report'):
        approve(path, failed, 'Jane', hashes, [])
    with pytest.raises(DeepGymError, match='differ from what QA tested'):
        approve(path, _passing_report(), 'Jane', {**hashes, 'task.md': 'old'}, [])
    approve(path, _passing_report(), 'Jane', hashes, [])
    with pytest.raises(DeepGymError, match='never overwritten'):
        approve(path, _passing_report(), 'Jane', hashes, [])


def test_an_approval_holding_a_failed_report_never_passes(tmp_path: Path) -> None:
    path = scaffold('env', 'function', tmp_path)
    failed = QAReport(environment='env', checks=[QACheck(name='c', status='fail', details='')])
    record = Approval(
        environment='env',
        approved_by='Jane',
        approved_at='2026-10-13T00:00:00Z',
        deepgym_version=__version__,
        hashes=content_hashes(path),
        report=failed,
        lint={'warnings': []},
    )
    (path / APPROVAL_FILE).write_text(record.model_dump_json(), encoding='utf-8')

    check = approval_check(path)

    assert check.status == 'fail'
    assert 'holds a QA report that did not pass (fail)' in check.details


@pytest.mark.parametrize(
    'drop',
    [
        lambda record: record.pop('approved_at'),
        lambda record: record.pop('deepgym_version'),
        lambda record: record['lint'].pop('warnings'),
        lambda record: record['report'].pop('created_at'),
    ],
    ids=['approved_at', 'deepgym_version', 'lint.warnings', 'report.created_at'],
)
def test_an_approval_missing_provenance_is_invalid(
    drop: Callable[[dict], object], tmp_path: Path
) -> None:
    path = scaffold('env', 'function', tmp_path)
    written = approve(path, _passing_report(), 'Jane', content_hashes(path), [])
    record = json.loads(written.read_text(encoding='utf-8'))
    drop(record)  # never filled in with today's date or the installed version
    written.chmod(0o644)
    written.write_text(json.dumps(record), encoding='utf-8')

    check = approval_check(path)

    assert check.status == 'fail'
    assert f'{APPROVAL_FILE} is invalid' in check.details


def test_permissions_and_directories_are_part_of_the_approval(tmp_path: Path) -> None:
    path = scaffold('env', 'function', tmp_path)
    approve(path, _passing_report(), 'Jane', content_hashes(path), [])

    (path / 'variants.py').chmod(0o600)  # a build copies modes: another user can't read it
    (path / 'data').mkdir()  # a build copies an empty directory too

    check = approval_check(path)
    assert check.status == 'fail'
    assert '(changed: variants.py; added: data)' in check.details


def test_adding_any_symlink_stales_an_approval(tmp_path: Path) -> None:
    path = scaffold('env', 'function', tmp_path)
    approve(path, _passing_report(), 'Jane', content_hashes(path), [])
    (tmp_path / 'real').mkdir()
    (path / 'alias').symlink_to(tmp_path / 'real')  # a build would copy the link

    check = approval_check(path)
    assert check.status == 'fail'
    assert 'added: alias' in check.details


def test_check_revalidates_an_approved_harbor_task(tmp_path: Path, local_dg: DeepGym) -> None:
    fixtures = Path(__file__).resolve().parent / 'fixtures' / 'harbor'
    path = shutil.copytree(fixtures / 'tasks' / 'hello', tmp_path / 'hello')
    broken = shutil.copytree(fixtures / 'invalid' / 'bad-toml', tmp_path / 'bad-toml')
    assert [c.status for c in check_env(path, local_dg)] == ['pass', 'skip']

    approve(path, _passing_report(), 'Jane', content_hashes(path), [])
    assert [c.status for c in check_env(path, local_dg)] == ['pass', 'pass']
    (path / 'tests' / 'test.sh').write_text('exit 0\n', encoding='utf-8')
    assert [c.status for c in check_env(path, local_dg)] == ['pass', 'fail']
    assert check_env(broken, local_dg)[0].details.startswith('Harbor task does not load')


def test_approval_records_keep_lint_warnings_marked_advisory(tmp_path: Path) -> None:
    path = scaffold('env', 'function', tmp_path)

    record = approve(path, _passing_report(), 'Jane', content_hashes(path), ['verifier.py:9 x'])

    lint = json.loads(record.read_text(encoding='utf-8'))['lint']
    assert lint == {'advisory': True, 'note': ADVISORY_NOTE, 'warnings': ['verifier.py:9 x']}
    assert 'never evidence' in lint['note']


def test_content_hashes_cover_every_input_but_the_record_and_own_git(tmp_path: Path) -> None:
    path = scaffold('env', 'function', tmp_path)
    before = content_hashes(path)
    (path / '.git').mkdir()
    (path / '.git' / 'HEAD').write_text('ref: refs/heads/main\n', encoding='utf-8')
    (path / APPROVAL_FILE).write_text('{}', encoding='utf-8')
    assert content_hashes(path) == before

    (path / '.helper.py').write_text('X = 1\n', encoding='utf-8')  # could be a support file
    (path / '.DS_Store').write_bytes(b'\0')
    (path / 'tests' / '__pycache__').mkdir(parents=True)  # a loader may upload whole dirs
    (path / 'tests' / '__pycache__' / 'grader.cpython-312.pyc').write_bytes(b'\0')
    (path / 'tests' / '.git').mkdir()
    (path / 'tests' / '.git' / 'grader.sh').write_text('exit 0\n', encoding='utf-8')

    added = {
        '.helper.py',
        '.DS_Store',
        'tests',
        'tests/__pycache__',
        'tests/__pycache__/grader.cpython-312.pyc',
        'tests/.git',
        'tests/.git/grader.sh',
    }
    assert set(content_hashes(path)) == {*before, *added}


def test_qa_approves_the_snapshot_it_tested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path / 'reports'))
    path = scaffold('env', 'function', tmp_path)
    verifier = path / 'verifier.py'
    original = verifier.read_text(encoding='utf-8')
    real_run_qa = deepgym.qa.run_qa
    tested = []

    def edit_during_qa(target: Path, *args: object) -> QAReport:
        tested.append(target)
        verifier.write_text('# B\n', encoding='utf-8')  # A -> B while QA runs ...
        report = real_run_qa(target, *args)
        verifier.write_text(original, encoding='utf-8')  # ... and back to A
        return report

    monkeypatch.setattr('deepgym.qa.run_qa', edit_during_qa)

    assert _cli('qa', str(path), '--reruns', '1', '--approve', '--by', 'Jane') == 0

    assert tested[0] != path and tested[0].name == path.name  # QA ran on a private copy
    assert approval_check(path).status == 'pass'


def test_qa_approves_nothing_when_qa_changed_its_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path / 'reports'))
    path = scaffold('env', 'function', tmp_path)
    real_run_qa = deepgym.qa.run_qa

    def qa_that_writes(target: Path, *args: object) -> QAReport:
        report = real_run_qa(target, *args)
        (target / 'cache.txt').write_text('written during QA\n', encoding='utf-8')
        return report

    monkeypatch.setattr('deepgym.qa.run_qa', qa_that_writes)

    assert _cli('qa', str(path), '--reruns', '1', '--approve', '--by', 'Jane') == 1

    assert 'changed during QA' in capsys.readouterr().err
    assert not (path / APPROVAL_FILE).exists()


def test_a_python_graded_env_that_looks_like_a_diff_is_still_attacked(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    path = scaffold('env', 'function', tmp_path)
    reference = path / 'reference_solution.py'
    looks_like_a_diff = '"""Example:\n--- a/runs.py\n+++ b/runs.py\n@@ -1 +1 @@\n"""\n\n'
    reference.write_text(looks_like_a_diff + reference.read_text(), encoding='utf-8')

    report = run_qa(path, local_dg, QAOptions(reruns=1))

    hack = {check.name: check for check in report.checks}['hack_resistant']
    assert (hack.status, report.status) == ('pass', 'pass')
    assert hack.details.startswith('5 attacks, max score')
    assert report.not_run == []


def test_a_skipped_attack_check_is_never_reported_as_a_pass(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    # The repo-patch verifier rejects every Python attack as a malformed patch.
    path = scaffold('env', 'repo-patch', tmp_path)

    report = run_qa(path, local_dg, QAOptions(reruns=1))

    hack = {check.name: check for check in report.checks}['hack_resistant']
    assert hack.status == 'skip' and not hack.passed
    assert hack.details.startswith('the verifier rejected every attack as a malformed answer')
    assert report.not_run == ['stub_fails', 'hack_resistant']
    assert 'env: hack_resistant: not run: the verifier rejected' in format_table([report])
    approve(path, report, 'Jane', content_hashes(path), [])
    assert 'checks that did not run: stub_fails, hack_resistant' in approval_check(path).details


def test_approval_needs_a_self_contained_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = scaffold('env', 'function', tmp_path)
    (tmp_path / 'shared.py').write_text('X = 1\n', encoding='utf-8')
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    metadata['support_files'].append('../shared.py')
    (path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')

    assert _cli('qa', str(path), '--reruns', '1', '--approve', '--by', 'Jane') == 1

    assert 'approval needs a self-contained folder' in capsys.readouterr().err
    assert not (path / APPROVAL_FILE).exists()


def test_approval_refuses_folders_with_symlinks(tmp_path: Path) -> None:
    path = scaffold('env', 'function', tmp_path)
    with snapshot(path) as copy:
        assert copy.name == path.name and copy != path
        assert copy.is_relative_to(local_runs_root())  # no local solution can read it
    outside = tmp_path / 'reference.py'
    outside.write_text((path / 'reference_solution.py').read_text(), encoding='utf-8')
    (path / 'reference_solution.py').unlink()
    (path / 'reference_solution.py').symlink_to(outside)  # QA would read a live file

    with pytest.raises(DeepGymError, match='without symlinks'), snapshot(path):
        pass


def test_a_multi_turn_agent_never_gets_reference_answers(tmp_path: Path) -> None:
    path = scaffold('env', 'multi-turn', tmp_path)
    (path / 'reference_solution.py').write_text('ANSWER = 42\n', encoding='utf-8')

    env = load_environment_dir(path)

    assert 'runs.py' in env.files
    assert not {'reference_solution.py', 'reference_actions.json'} & set(env.files)


def test_a_multi_turn_folder_never_stages_a_file_that_links_outside(tmp_path: Path) -> None:
    path = scaffold('env', 'multi-turn', tmp_path)
    secret = tmp_path / 'secret.txt'
    secret.write_text('host credentials', encoding='utf-8')
    (path / 'notes.txt').symlink_to(secret)  # would be uploaded into the agent's workspace

    with pytest.raises(ValueError, match='notes.txt links outside env'):
        load_environment_dir(path)


def test_multi_turn_reports_name_every_check_that_did_not_run(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    path = scaffold('env', 'multi-turn', tmp_path)
    client = ZaiChatClient('unused', base_url='http://127.0.0.1:9', model='unused')

    report = run_qa(path, local_dg, QAOptions(reruns=1, client=client, mutants=True))

    assert report.status == 'pass'
    assert report.not_run == ['stub_fails', 'hack_resistant', 'mutants_killed', 'difficulty']
    assert missing_evidence(report) == ['mutants_killed', 'difficulty']


def test_content_hashes_cover_every_listed_support_file(tmp_path: Path) -> None:
    path = scaffold('env', 'function', tmp_path)
    (tmp_path / 'shared.py').write_text('X = 1\n', encoding='utf-8')
    (path / '__pycache__').mkdir()
    (path / '__pycache__' / 'scoring.py').write_text('Y = 1\n', encoding='utf-8')
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    metadata['support_files'] += ['../shared.py', '__pycache__/scoring.py']
    (path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    approve(path, _passing_report(), 'Jane', content_hashes(path), [])

    (tmp_path / 'shared.py').write_text('X = 2\n', encoding='utf-8')
    (path / '__pycache__' / 'scoring.py').write_text('Y = 2\n', encoding='utf-8')

    assert 'changed: ../shared.py, __pycache__/scoring.py' in approval_check(path).details
    assert approval_check(path).status == 'fail'  # an unreadable record never passes
