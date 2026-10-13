"""Tests for deepgym.qa -- the environment quality gate."""

from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from deepgym.core import DeepGym
from deepgym.exceptions import DeepGymError
from deepgym.integrations.zai import ZaiChatClient
from deepgym.models import Environment
from deepgym.qa import QACheck, QAReport, client_from_env, format_table, run_qa, write_report
from deepgym.qa.checks import hack_resistant, single_turn_checks, stub_solution
from deepgym.qa.difficulty import difficulty_check
from tests.test_adversarial import CRASHING_VERIFIER

_ADD_TASK = 'Write `add(a, b)` returning the sum.'
_ADD_REFERENCE = 'def add(a, b):\n    return a + b\n'
_ADD_VERIFIER = (
    'from deepgym_solution import load_solution\n'
    'sol = load_solution(solution_path)\n'
    'return 1.0 if hasattr(sol, "add") and sol.add(2, 3) == 5 else 0.0\n'
)


def _env(verifier_body: str) -> Environment:
    return Environment(task=_ADD_TASK, verifier_code=verifier_body)


def _by_name(checks: list[QACheck]) -> dict[str, QACheck]:
    return {check.name: check for check in checks}


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def test_stub_solution_defines_every_top_level_function_returning_none() -> None:
    reference = 'import os\n\ndef solve(x):\n    return x\n\nasync def helper(x):\n    pass\n'

    stub = stub_solution(reference)

    namespace: dict = {}
    exec(stub, namespace)
    assert namespace['solve'](1, k=2) is None
    assert namespace['helper']() is None


@pytest.mark.parametrize('reference', ['print("done")\n', '{"path": "/x"}', 'not python at all ('])
def test_stub_solution_is_empty_without_top_level_functions(reference: str) -> None:
    assert stub_solution(reference) == ''


def test_single_turn_checks_pass_for_a_sound_env(local_dg: DeepGym) -> None:
    checks = single_turn_checks(_env(_ADD_VERIFIER), _ADD_REFERENCE, local_dg, reruns=2)

    assert [check.name for check in checks] == [
        'reference_passes',
        'deterministic',
        'empty_fails',
        'stub_fails',
    ]
    assert all(check.passed for check in checks), checks
    assert _by_name(checks)['deterministic'].details == '3 runs all scored 1.00'


def test_single_turn_checks_catch_a_verifier_that_passes_anything(local_dg: DeepGym) -> None:
    checks = _by_name(single_turn_checks(_env('return 1.0\n'), _ADD_REFERENCE, local_dg, 1))

    assert checks['reference_passes'].passed
    assert not checks['empty_fails'].passed
    assert not checks['stub_fails'].passed


def test_single_turn_checks_catch_nondeterministic_scores(local_dg: DeepGym) -> None:
    env = _env('import random\nreturn random.random()\n')

    checks = _by_name(single_turn_checks(env, _ADD_REFERENCE, local_dg, reruns=3))

    assert not checks['deterministic'].passed
    assert checks['deterministic'].details.startswith('scores differ across 4 runs')


def test_single_turn_checks_fail_without_a_reference(local_dg: DeepGym) -> None:
    checks = single_turn_checks(_env(_ADD_VERIFIER), None, local_dg, reruns=3)

    assert [(c.name, c.passed) for c in checks] == [
        ('reference_passes', False),
        ('empty_fails', True),
    ]
    assert checks[0].details == 'no reference_solution.py found'


def test_single_turn_checks_skip_the_stub_of_a_script_reference(local_dg: DeepGym) -> None:
    checks = single_turn_checks(_env(_ADD_VERIFIER), 'print(5)\n', local_dg, reruns=0)

    assert _by_name(checks)['stub_fails'].status == 'skip'


def test_hack_resistant(local_dg: DeepGym) -> None:
    assert hack_resistant(_env(_ADD_VERIFIER), local_dg).passed

    exploited = hack_resistant(_env('return 1.0\n'), local_dg)

    assert not exploited.passed
    assert exploited.details.startswith('exploited by ')


def test_hack_resistant_is_inconclusive_when_attacks_cannot_run(local_dg: DeepGym) -> None:
    check = hack_resistant(_env(CRASHING_VERIFIER), local_dg)

    assert not check.passed
    assert check.status == 'inconclusive'
    assert check.details.startswith('5 of 5 attacks could not run: empty (Attack could not run')


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def test_run_qa_passes_a_builtin_env(local_dg: DeepGym) -> None:
    report = run_qa('reverse_string', local_dg, reruns=1)

    assert report.passed
    assert [check.name for check in report.checks] == [
        'reference_passes',
        'deterministic',
        'empty_fails',
        'stub_fails',
        'hack_resistant',
    ]


@pytest.mark.parametrize('name', ['signup_form', 'swebench_pro'])
def test_run_qa_skips_envs_it_cannot_check(name: str, local_dg: DeepGym) -> None:
    report = run_qa(name, local_dg)

    assert not report.passed
    assert report.status == 'skip'
    assert [(check.name, check.status) for check in report.checks] == [('skipped', 'skip')]


def test_run_qa_reports_an_unloadable_env(local_dg: DeepGym) -> None:
    report = run_qa('no_such_env', local_dg)

    assert report.status == 'fail'
    assert report.checks[0].name == 'load'
    assert 'not found' in report.checks[0].details


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_write_report_uses_the_console_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path))
    report = QAReport(
        environment='cyber/toy',
        checks=[QACheck(name='reference_passes', status='fail', details='score 0.00')],
    )

    path = write_report(report)

    assert path == tmp_path / 'cyber_toy.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    assert set(data) == {'environment', 'status', 'passed', 'checks', 'created_at'}
    assert data['environment'] == 'cyber/toy'
    assert (data['status'], data['passed']) == ('fail', False)
    assert data['checks'] == [
        {'name': 'reference_passes', 'status': 'fail', 'passed': False, 'details': 'score 0.00'}
    ]
    assert datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')).tzinfo is not None


@pytest.mark.parametrize(
    ('statuses', 'expected'),
    [
        (['pass', 'skip'], 'pass'),
        (['pass', 'inconclusive', 'skip'], 'inconclusive'),
        (['inconclusive', 'fail', 'pass'], 'fail'),
        (['skip'], 'skip'),
    ],
)
def test_report_status_is_the_worst_verdict_and_skips_are_neutral(
    statuses: list[str], expected: str
) -> None:
    checks = [QACheck(name=f'c{i}', status=status, details='') for i, status in enumerate(statuses)]

    report = QAReport(environment='e', checks=checks)

    assert (report.status, report.passed) == (expected, expected == 'pass')


def test_format_table_lists_failures_and_a_summary() -> None:
    reports = [
        QAReport(environment=env, checks=[QACheck(name=check, status=status, details=details)])
        for env, check, status, details in [
            ('good', 'empty_fails', 'pass', ''),
            ('bad', 'stub_fails', 'fail', 'x'),
            ('odd', 'hack_resistant', 'inconclusive', 'z'),
            ('gui', 'skipped', 'skip', 'y'),
        ]
    ]

    table = format_table(reports)

    assert 'bad: stub_fails: x' in table
    assert 'odd: hack_resistant: z' in table
    assert 'gui: ' not in table
    assert 'good: ' not in table
    assert table.endswith('1 passed, 1 failed, 1 inconclusive, 1 skipped')


# ---------------------------------------------------------------------------
# difficulty
# ---------------------------------------------------------------------------

_RIGHT = '```python\ndef add(a, b):\n    return a + b\n```'
_WRONG = 'Sure:\n```python\ndef add(a, b):\n    return a - b\n```'


def _client(answers: list[str], seen: list[dict]) -> ZaiChatClient:
    replies = itertools.cycle(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={'choices': [{'message': {'content': next(replies)}}]})

    return ZaiChatClient(
        'key', base_url='https://llm.test/v1', model='m', transport=httpx.MockTransport(handler)
    )


@pytest.mark.parametrize(
    ('answers', 'passed', 'details'),
    [
        ([_RIGHT, _WRONG], True, 'pass@1=0.50 pass@4=1.00 (2/4 solved by m)'),
        ([_RIGHT], False, 'too_easy: pass@1=1.00 pass@4=1.00 (4/4 solved by m)'),
        ([_WRONG], False, 'too_hard: pass@1=0.00 pass@4=0.00 (0/4 solved by m)'),
    ],
)
def test_difficulty_check(
    answers: list[str], passed: bool, details: str, local_dg: DeepGym
) -> None:
    seen: list[dict] = []

    check = difficulty_check(_env(_ADD_VERIFIER), local_dg, _client(answers, seen), samples=4)

    assert (check.passed, check.details) == (passed, details)
    assert len(seen) == 4
    assert seen[0]['messages'] == [{'role': 'user', 'content': _ADD_TASK}]
    assert seen[0]['temperature'] == 1.0
    assert 'thinking' not in seen[0]


def test_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(DeepGymError, match='OPENAI_API_KEY'):
        client_from_env('m')

    monkeypatch.setenv('OPENAI_API_KEY', 'key')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    assert client_from_env('m').base_url == 'https://api.openai.com/v1'

    monkeypatch.setenv('OPENAI_BASE_URL', 'http://localhost:8000/v1/')
    client = client_from_env('m')
    assert (client.base_url, client.model) == ('http://localhost:8000/v1', 'm')
