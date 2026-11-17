"""Controls: the answers an env ships that its verifier must pass or fail."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deepgym.core import DeepGym
from deepgym.factory.scaffold import scaffold
from deepgym.models import Environment
from deepgym.qa import QAOptions, run_qa
from deepgym.qa.controls import controls_check
from deepgym.registry import load_environment_dir

_ADD = 'def add(a, b):\n    return a + b\n'
_SWAPPED = 'def add(a, b):\n    return b + a\n'
_HARDCODED = 'def add(a, b):\n    return 5\n'
_VERIFIER = (
    'import os\n'
    'if os.path.getsize(solution_path) > 200:  # a verifier bug on long answers\n'
    "    raise RuntimeError('verifier bug')\n"
    'from deepgym_solution import load_solution\n'
    'sol = load_solution(solution_path)\n'
    'wrong = [(a, b) for a, b in [(2, 3), (-1, 4)] if sol.add(a, b) != a + b]\n'
    "details = ', '.join(f'add{case} is wrong' for case in wrong) or 'all right'\n"
    "return {'score': 0.0 if wrong else 1.0, 'passed': not wrong, 'details': details}\n"
)
_ONE_CASE = (
    'from deepgym_solution import load_solution\n'
    'return 1.0 if load_solution(solution_path).add(2, 3) == 5 else 0.0\n'
)
_SOURCE_MATCH = "return 1.0 if 'a + b' in open(solution_path).read() else 0.0\n"


def _env(verifier: str) -> Environment:
    return Environment(task='Write `add(a, b)` returning the sum.', verifier_code=verifier)


def _folder(tmp_path: Path, controls: dict[str, str]) -> Path:
    """An env folder holding *controls*, keyed by path under controls/."""
    for rel, text in controls.items():
        path = tmp_path / 'controls' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return tmp_path


def test_good_controls_pass(tmp_path: Path, local_dg: DeepGym) -> None:
    folder = _folder(tmp_path, {'must_pass/swapped.py': _SWAPPED, 'must_fail/five.py': _HARDCODED})

    check = controls_check(_env(_VERIFIER), folder, local_dg, runs=2)

    assert (check.status, check.details) == (
        'pass',
        '1 must_pass and 1 must_fail controls, 2 runs each: every run as stated',
    )


def test_a_must_fail_control_the_grader_passes_fails_the_check(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    folder = _folder(tmp_path, {'must_fail/five.py': _HARDCODED})

    check = controls_check(_env(_ONE_CASE), folder, local_dg, runs=2)

    assert check.status == 'fail'
    assert check.details.startswith(
        'must_fail/five.py: 2 of 2 runs wrong, the first scored full marks: score 1.00'
    )


def test_a_must_pass_control_the_grader_fails_fails_the_check(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    folder = _folder(tmp_path, {'must_pass/swapped.py': _SWAPPED})

    check = controls_check(_env(_SOURCE_MATCH), folder, local_dg, runs=1)

    assert check.status == 'fail'
    assert check.details.startswith(
        'must_pass/swapped.py: 1 of 1 runs wrong, the first scored below full marks'
    )


def test_a_control_without_a_verdict_fails_the_check(tmp_path: Path, local_dg: DeepGym) -> None:
    padding = '# padding\n' * 30  # trips the verifier bug before it judges the answer
    folder = _folder(
        tmp_path,
        {
            'must_pass/long.py': _SWAPPED + padding,
            'must_fail/long.py': _HARDCODED + padding,  # its 0 is no verdict, not a fail
            'must_fail/x.py': '',
        },
    )

    check = controls_check(_env(_VERIFIER), folder, local_dg, runs=1)

    assert check.status == 'fail'  # never a pass, never merely inconclusive
    for label in ('must_fail/long.py', 'must_pass/long.py'):
        assert f'{label}: 1 of 1 runs wrong, the first got no verdict' in check.details
    assert 'must_fail/x.py' not in check.details  # the empty answer was judged: a real 0


def test_a_missing_or_empty_controls_folder_never_fails(tmp_path: Path, local_dg: DeepGym) -> None:
    assert controls_check(_env(_VERIFIER), tmp_path, local_dg, runs=1) is None

    (tmp_path / 'controls' / 'must_fail').mkdir(parents=True)
    (tmp_path / 'controls' / 'must_fail' / '.DS_Store').write_bytes(b'\0')
    check = controls_check(_env(_VERIFIER), tmp_path, local_dg, runs=1)

    assert check.status == 'skip'


def test_a_stray_path_under_controls_fails(tmp_path: Path, local_dg: DeepGym) -> None:
    folder = _folder(tmp_path, {'must-fail/five.py': _HARDCODED, 'must_pass/swapped.py': _SWAPPED})

    check = controls_check(_env(_VERIFIER), folder, local_dg, runs=1)

    assert check.status == 'fail'
    assert check.details.startswith('controls/ holds must-fail, must-fail/five.py;')


def test_an_expect_line_must_appear_in_every_run_and_never_reaches_the_verifier(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    # Gives 0 to any answer holding the expect line, so a kept line fails must_pass.
    verifier = "if '# expect' in open(solution_path).read():\n    return 0.0\n" + _VERIFIER
    folder = _folder(
        tmp_path,
        {
            'must_pass/swapped.py': f'# expect: all right\n{_SWAPPED}',
            'must_fail/five.py': f'# expect: add(-1, 4) is wrong\n{_HARDCODED}',
            'must_fail/typo.py': f'# expect: add(-1, 4) is wrong\n{_ADD}+',  # a SyntaxError
        },
    )

    check = controls_check(_env(verifier), folder, local_dg, runs=1)

    assert check.status == 'fail'
    assert check.details.startswith(
        "must_fail/typo.py: 1 of 1 runs wrong, the first did not say 'add(-1, 4) is wrong'"
    )
    assert 'swapped.py' not in check.details and 'five.py' not in check.details


def test_run_qa_reports_the_controls_a_folder_ships(tmp_path: Path, local_dg: DeepGym) -> None:
    path = scaffold('env', 'function', tmp_path)  # ships one must_pass and one must_fail

    report = run_qa(path, local_dg, QAOptions(reruns=1))

    assert report.status == 'pass', report.checks
    assert 'controls' in report.ran
    (path / 'controls' / 'must_fail' / 'equal_neighbors.py').write_text(
        (path / 'reference_solution.py').read_text(encoding='utf-8'), encoding='utf-8'
    )
    failed = {check.name: check for check in run_qa(path, local_dg, QAOptions(reruns=0)).checks}
    assert failed['controls'].status == 'fail'
    assert failed['controls'].details.startswith('must_fail/equal_neighbors.py: 1 of 1 runs wrong')


def test_multi_turn_controls_replay_as_episodes_and_never_reach_the_agent(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    path = scaffold('env', 'multi-turn', tmp_path)
    reference = (path / 'reference_actions.json').read_text(encoding='utf-8')
    _folder(path, {'must_pass/reference.json': reference, 'must_fail/quits.json': '[]'})

    report = run_qa(path, local_dg, QAOptions(reruns=0))

    checks = {check.name: check for check in report.checks}
    assert checks['controls'].status == 'pass', checks['controls']
    assert not any(rel.startswith('controls/') for rel in load_environment_dir(path).files)
    (path / 'controls').rename(path / 'Controls')  # same folder on a case-insensitive disk
    assert not any(rel.startswith('Controls/') for rel in load_environment_dir(path).files)
    (path / 'Controls').rename(path / 'controls')
    (path / 'controls' / 'must_fail' / 'quits.json').write_text(json.dumps({'not': 'a list'}))
    broken = {check.name: check for check in run_qa(path, local_dg, QAOptions(reruns=0)).checks}
    assert broken['controls'].status == 'fail'
    assert broken['controls'].details.startswith(
        'must_fail/quits.json is not an episode like reference_actions.json'
    )


def test_a_named_env_runs_the_controls_in_its_registered_folder(
    tmp_path: Path, local_dg: DeepGym, monkeypatch: pytest.MonkeyPatch
) -> None:
    """By name, as `deepgym qa NAME` and --all run it, not only by folder."""
    import deepgym.registry as registry_module

    path = scaffold('named_env', 'multi-turn', tmp_path)
    _folder(path, {'must_pass/quits.json': '[]'})  # wrong: quitting earns nothing
    monkeypatch.setattr(registry_module, '_get_cache_dir', lambda: tmp_path)

    checks = {
        check.name: check for check in run_qa('named_env', local_dg, QAOptions(reruns=0)).checks
    }

    assert checks['controls'].status == 'fail', checks['controls']


@pytest.mark.skipif(not hasattr(os, 'symlink'), reason='needs symlinks')
def test_a_link_to_a_control_never_reaches_the_agent(tmp_path: Path) -> None:
    path = scaffold('env', 'multi-turn', tmp_path)
    reference = (path / 'reference_actions.json').read_text(encoding='utf-8')
    _folder(path, {'must_pass/reference.json': reference})
    (path / 'answer.json').symlink_to(Path('controls/must_pass/reference.json'))

    assert 'answer.json' not in load_environment_dir(path).files
