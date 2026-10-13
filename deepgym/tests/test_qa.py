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
from deepgym.qa import (
    QACheck,
    QAOptions,
    QAReport,
    client_from_env,
    format_table,
    run_qa,
    write_report,
)
from deepgym.qa.checks import hack_resistant, single_turn_checks, stub_solution
from deepgym.qa.difficulty import difficulty_check
from tests.test_adversarial import CRASHING_VERIFIER, RAISING_VERIFIER

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


def test_reference_runs_without_a_verdict_prove_nothing(local_dg: DeepGym) -> None:
    # Full marks, but the verifier reported its own error: not a verdict.
    body = "return {'score': 1.0, 'passed': True, 'error_type': 'runtime_error'}\n"

    checks = _by_name(single_turn_checks(_env(body), _ADD_REFERENCE, local_dg, reruns=1))

    assert checks['reference_passes'].status == 'inconclusive'
    assert checks['deterministic'].status == 'inconclusive'


@pytest.mark.parametrize('reward', [0.25, 0.01])
def test_negative_controls_that_earn_any_reward_fail(reward: float, local_dg: DeepGym) -> None:
    body = f"return 1.0 if 'a + b' in open(solution_path).read() else {reward}\n"

    checks = _by_name(single_turn_checks(_env(body), _ADD_REFERENCE, local_dg, reruns=1))

    assert checks['reference_passes'].passed
    for name in ('empty_fails', 'stub_fails'):
        assert checks[name].status == 'fail'
        assert checks[name].details.endswith('earned reward without doing the task')


def test_one_reference_run_says_nothing_about_determinism(local_dg: DeepGym) -> None:
    checks = _by_name(single_turn_checks(_env(_ADD_VERIFIER), _ADD_REFERENCE, local_dg, reruns=0))

    assert checks['deterministic'].status == 'skip'
    assert checks['deterministic'].details.startswith('one run says nothing about determinism')


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


def test_a_verifier_error_on_empty_or_stub_is_inconclusive_not_a_pass(local_dg: DeepGym) -> None:
    # sol.add raises AttributeError in the judge for the empty solution: no verdict on it.
    strict = (
        'from deepgym_solution import load_solution\n'
        'sol = load_solution(solution_path)\n'
        'return 1.0 if sol.add(2, 3) == 5 else 0.0\n'
    )

    checks = _by_name(single_turn_checks(_env(strict), _ADD_REFERENCE, local_dg, reruns=0))

    assert checks['empty_fails'].status == 'inconclusive'
    assert checks['empty_fails'].details.startswith('the verifier errored instead of judging')
    assert checks['stub_fails'].status == 'pass'  # the stub's add() returns None: a real 0


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
    assert check.details.startswith('5 of 5 attacks got no verdict: empty (Attack could not run')


def test_hack_resistant_is_inconclusive_when_the_verifier_errors(local_dg: DeepGym) -> None:
    check = hack_resistant(_env(RAISING_VERIFIER), local_dg)

    assert (check.status, check.passed) == ('inconclusive', False)
    assert 'runtime_error' in check.details


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def test_run_qa_passes_a_builtin_env(local_dg: DeepGym) -> None:
    report = run_qa('reverse_string', local_dg, QAOptions(reruns=1))

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
    assert set(data) == {
        'environment',
        'status',
        'passed',
        'ran',
        'not_run',
        'checks',
        'created_at',
    }
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
    scored = QACheck(name='mutants_killed', status='pass', details='mutation score 3/3')
    assert 'e: mutants_killed: mutation score 3/3' in format_table(
        [QAReport(environment='e', checks=[scored])]
    )
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
    ('answers', 'details'),
    [
        ([_RIGHT, _WRONG], 'pass@1=0.50 pass@4=1.00, mean reward 0.50, spread 0.50'),
        ([_RIGHT], 'too_easy, no_variance: pass@1=1.00 pass@4=1.00, mean reward 1.00, spread 0.00'),
        ([_WRONG], 'too_hard, no_variance: pass@1=0.00 pass@4=0.00, mean reward 0.00, spread 0.00'),
    ],
)
def test_difficulty_is_reported_never_gated(
    answers: list[str], details: str, local_dg: DeepGym
) -> None:
    seen: list[dict] = []

    check = difficulty_check(_env(_ADD_VERIFIER), local_dg, _client(answers, seen), samples=4)

    assert check.passed  # too easy or too hard is information, not a failed environment
    assert check.details.startswith(details)
    assert len(seen) == 4
    assert seen[0]['messages'] == [{'role': 'user', 'content': _ADD_TASK}]
    assert seen[0]['temperature'] == 1.0
    assert 'thinking' not in seen[0]


def test_difficulty_is_inconclusive_when_a_sample_gets_no_verdict(local_dg: DeepGym) -> None:
    body = (
        'source = open(solution_path).read()\n'
        "if 'BOOM' in source:\n"
        "    raise RuntimeError('a verifier bug')\n"
        "return 1.0 if 'a + b' in source else 0.0\n"
    )
    boom = 'Here:\n```python\nBOOM = 1\n```'

    check = difficulty_check(_env(body), local_dg, _client([_RIGHT, boom], []), samples=2)

    assert check.status == 'inconclusive'  # never a pass@1 of 0.50 from one real verdict
    assert check.details == '1/2 samples got no verdict (the verifier errored)'


def test_partial_rewards_are_variance_not_too_hard(local_dg: DeepGym) -> None:
    body = "return 0.2 if 'low' in open(solution_path).read() else 0.8\n"
    answers = ['```python\nlow = 1\n```', '```python\nhigh = 1\n```']

    check = difficulty_check(_env(body), local_dg, _client(answers, []), samples=2)

    assert check.details.startswith('pass@1=0.00 pass@2=0.00, mean reward 0.50, spread 0.30')


def _flaky_client(statuses: list[int], seen: list[int]) -> ZaiChatClient:
    """A client whose endpoint answers with *statuses* in turn, then a solved answer."""
    replies = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(replies, 200)
        seen.append(status)
        if status != 200:
            return httpx.Response(status, headers={'retry-after': '0'}, json={})
        return httpx.Response(200, json={'choices': [{'message': {'content': _RIGHT}}]})

    transport = httpx.MockTransport(handler)
    return ZaiChatClient('key', base_url='https://llm.test/v1', model='m', transport=transport)


def test_a_rate_limited_sample_is_retried(
    local_dg: DeepGym, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('deepgym.qa.difficulty.time.sleep', lambda _: None)
    seen: list[int] = []

    check = difficulty_check(_env(_ADD_VERIFIER), local_dg, _flaky_client([429, 503], seen), 2)

    assert check.status == 'pass'
    assert seen[:2] == [429, 503] and seen.count(200) == 2


def test_an_endpoint_that_stays_rate_limited_makes_difficulty_inconclusive(
    local_dg: DeepGym, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('deepgym.qa.difficulty.time.sleep', lambda _: None)

    check = difficulty_check(_env(_ADD_VERIFIER), local_dg, _flaky_client([429] * 99, []), 2)

    assert check.status == 'inconclusive'  # never a model failure, never an aborted run
    assert '2/2 samples never came back from the model endpoint after 5 retries' in check.details


def test_an_endpoint_that_refuses_the_request_stops_the_run(local_dg: DeepGym) -> None:
    with pytest.raises(DeepGymError, match='refused the request'):
        difficulty_check(_env(_ADD_VERIFIER), local_dg, _flaky_client([401] * 9, []), 2)


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


def test_reports_are_written_whole_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepgym.qa import report as report_module

    target = tmp_path / 'r.json'
    report_module.write_atomically(target, 'old\n')

    def crash(*args: object) -> None:
        raise OSError('disk gone')

    monkeypatch.setattr(report_module.os, 'replace', crash)
    with pytest.raises(OSError, match='disk gone'):
        report_module.write_atomically(target, 'new\n')

    assert target.read_text() == 'old\n'  # the previous evidence survives a failed write
    assert [path.name for path in tmp_path.iterdir()] == ['r.json']  # no stray temp file
    with pytest.raises(FileExistsError):
        report_module.write_atomically(target, 'new\n', exclusive=True)
