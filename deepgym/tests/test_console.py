"""Tests for the console API: environments, runs, sandboxes and QA reports."""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deepgym.console.db import connect
from deepgym.console.run_store import RunStore
from deepgym.console.schemas import RunSpec
from deepgym.models import RunResult
from deepgym.qa.report import QACheck, QAReport, write_report
from deepgym.sandboxes.apple import AppleContainer
from deepgym.sandboxes.docker import DockerContainer
from tests.console_client import make_client

REVERSE_WRONG = 'def reverse_string(s: str) -> str:\n    return s\n'


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """Console allowed to run custom solutions locally."""
    with make_client(tmp_path, allow_local_exec=True) as test_client:
        yield test_client


def _wait_finished(client: TestClient, run_id: str, timeout_s: float = 60) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        run = client.get(f'/api/runs/{run_id}').json()
        if run['status'] in ('completed', 'failed'):
            return run
        time.sleep(0.2)
    raise AssertionError(f'run {run_id} did not finish in {timeout_s}s')


class TestEnvironments:
    def test_list_has_every_kind(self, client: TestClient) -> None:
        envs = client.get('/api/environments').json()
        assert len(envs) >= 31
        assert {env['kind'] for env in envs} >= {'single-turn', 'computer-use'}

    @pytest.mark.parametrize(
        ('query', 'field', 'value'),
        [('kind=computer-use', 'kind', 'computer-use'), ('difficulty=easy', 'difficulty', 'easy'),
         ('domain=browser', 'domain', 'browser')],
    )  # fmt: skip
    def test_filters(self, client: TestClient, query: str, field: str, value: str) -> None:
        envs = client.get(f'/api/environments?{query}').json()
        assert envs
        assert all(env[field] == value for env in envs)

    def test_detail_has_task_verifier_and_reference(self, client: TestClient) -> None:
        env = client.get('/api/environments/coin_change').json()
        assert env['name'] == 'Coin Change'
        assert env['kind'] == 'single-turn'
        assert 'coin' in env['task'].lower()
        assert 'def ' in env['verifier']
        assert env['reference']
        assert env['metadata']['id'] == 'env-012'

    def test_detail_of_nested_env(self, client: TestClient) -> None:
        response = client.get('/api/environments/cyber/web_path_traversal_toy')
        assert response.status_code == 200
        assert response.json()['domain'] == 'cyber'

    def test_computer_use_variant(self, client: TestClient) -> None:
        env = client.get('/api/environments/signup_form?variant=3').json()
        assert env['kind'] == 'computer-use'
        assert env['variants'] == 8
        assert env['variant'] == 3
        assert env['reference'] is None

    def test_unknown_env_is_404(self, client: TestClient) -> None:
        assert client.get('/api/environments/no_such_env_xyz').status_code == 404


class TestRuns:
    def test_reference_run_passes(self, client: TestClient) -> None:
        response = client.post('/api/runs', json={'environments': ['reverse_string']})
        assert response.status_code == 202
        (queued,) = response.json()
        assert queued['reference'] is True

        run = _wait_finished(client, queued['id'])

        assert run['status'] == 'completed'
        assert run['passed'] is True
        assert run['score'] == 1.0
        assert run['provider'] == 'local'
        assert run['duration_ms'] > 0
        assert run['solution']
        assert len(run['result']['cases']) > 1

    def test_wrong_solution_scores_partially(self, client: TestClient) -> None:
        body = {'environments': ['reverse_string'], 'solution': REVERSE_WRONG, 'mode': 'local'}
        (queued,) = client.post('/api/runs', json=body).json()

        run = _wait_finished(client, queued['id'])

        assert run['status'] == 'completed'
        assert run['passed'] is False
        assert 0 < run['score'] < 1
        assert run['reference'] is False
        assert any(not case['passed'] for case in run['result']['cases'])

    def test_small_eval_shares_a_group(self, client: TestClient) -> None:
        body = {'environments': ['reverse_string', 'fizzbuzz']}
        queued = client.post('/api/runs', json=body).json()
        assert len({run['group_id'] for run in queued}) == 1

        for run in queued:
            _wait_finished(client, run['id'])
        listed = client.get(f'/api/runs?group_id={queued[0]["group_id"]}').json()

        assert sorted(run['environment'] for run in listed) == ['fizzbuzz', 'reverse_string']
        assert client.get('/api/runs?status=completed').json()

    @pytest.mark.parametrize(
        ('body', 'code'),
        [({'environments': ['signup_form']}, 400), ({'environments': ['no_such_env']}, 404),
         ({'environments': []}, 422), ({'environments': ['fizzbuzz'], 'mode': 'nope'}, 422)],
    )  # fmt: skip
    def test_rejected_requests(self, client: TestClient, body: dict, code: int) -> None:
        assert client.post('/api/runs', json=body).status_code == code
        assert client.get('/api/runs').json() == []

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get('/api/runs/nope').status_code == 404

    def test_restart_fails_unfinished_runs(self, tmp_path: Path) -> None:
        spec = RunSpec(environment='fizzbuzz', solution='pass', reference=False)
        (run,) = RunStore(tmp_path / 'console.db').create_group([spec], 'local', 'local')

        detail = make_client(tmp_path).get(f'/api/runs/{run.id}').json()

        assert detail['status'] == 'failed'
        assert detail['error_type'] == 'interrupted'


def _seed_runs(db: Path) -> None:
    """Five finished runs: three in the last day, one the day before, one four days ago."""
    store = RunStore(db)
    # environment, score, error_type, duration_ms, hours ago
    plan = [
        ('fizzbuzz', 1.0, None, 100.0, 1),
        ('fizzbuzz', 0.4, None, 300.0, 2),
        ('two_sum', None, 'timeout', 50.0, 3),
        ('fizzbuzz', 1.0, None, 100.0, 30),
        ('two_sum', 1.0, None, 100.0, 100),
    ]
    specs = [RunSpec(environment=env, solution='pass', reference=True) for env, *_ in plan]
    runs = store.create_group(specs, 'local', 'local')
    now = datetime.now(timezone.utc)
    for run, (_, score, error_type, duration_ms, hours_ago) in zip(runs, plan):
        if error_type:
            store.fail(run.id, error_type, 'took too long', duration_ms)
        else:
            result = RunResult(
                score=score,
                passed=score == 1.0,
                output='',
                stderr='',
                exit_code=0,
                execution_time_ms=duration_ms,
                sandbox_id='local',
            )
            store.complete(run.id, result, duration_ms)
        with connect(db) as conn:
            created_at = (now - timedelta(hours=hours_ago)).isoformat()
            conn.execute('UPDATE runs SET created_at = ? WHERE id = ?', (created_at, run.id))


class TestRunStats:
    @pytest.fixture()
    def stats_client(self, tmp_path: Path) -> TestClient:
        _seed_runs(tmp_path / 'console.db')
        return make_client(tmp_path)

    def test_window_totals_and_previous_window(self, stats_client: TestClient) -> None:
        stats = stats_client.get('/api/runs/stats?hours=24').json()

        current = stats['current']
        assert [current[key] for key in ('runs', 'finished', 'passed', 'errors')] == [3, 3, 1, 1]
        # The timeout has no verdict: 0 in the mean, left out of the judged mean.
        assert current['mean_score'] == pytest.approx((1.0 + 0.4 + 0.0) / 3)
        assert current['judged_mean_score'] == pytest.approx((1.0 + 0.4) / 2)
        assert (current['judged_count'], current['no_verdict_count']) == (2, 1)
        assert current['mean_duration_ms'] == pytest.approx(150.0)
        assert (stats['previous']['runs'], stats['previous']['passed']) == (1, 1)
        assert stats['hours'] == 24

    def test_window_breakdowns(self, stats_client: TestClient) -> None:
        stats = stats_client.get('/api/runs/stats?hours=24').json()

        assert stats['score_bins'] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        assert stats['error_types'] == [{'error_type': 'timeout', 'count': 1}]
        by_env = {env['environment']: env for env in stats['environments']}
        assert by_env.keys() == {'fizzbuzz', 'two_sum'}
        assert (by_env['fizzbuzz']['runs'], by_env['fizzbuzz']['passed']) == (2, 1)
        assert by_env['fizzbuzz']['mean_duration_ms'] == pytest.approx(200.0)
        assert by_env['two_sum']['errors'] == 1

    def test_without_hours_covers_every_run(self, stats_client: TestClient) -> None:
        stats = stats_client.get('/api/runs/stats').json()
        assert stats['current']['runs'] == 5
        assert stats['previous'] is None

    def test_filters_apply(self, stats_client: TestClient) -> None:
        stats = stats_client.get('/api/runs/stats?hours=24&environment=fizzbuzz').json()
        assert stats['current']['runs'] == 2
        assert stats['error_types'] == []

    def test_empty_window(self, tmp_path: Path) -> None:
        stats = make_client(tmp_path).get('/api/runs/stats?hours=24').json()
        assert stats['current'] == {'runs': 0, 'finished': 0, 'passed': 0, 'errors': 0,
                                    'mean_score': None, 'mean_duration_ms': None,
                                    'judged_count': 0, 'no_verdict_count': 0,
                                    'judged_mean_score': None}  # fmt: skip
        assert stats['score_bins'] == [0] * 10

    def test_run_without_a_verdict_counts_zero_and_never_raises_the_mean(
        self, tmp_path: Path
    ) -> None:
        store = RunStore(tmp_path / 'console.db')
        specs = [RunSpec(environment='fizzbuzz', solution='pass', reference=True)] * 3
        judged, crashed, silent = store.create_group(specs, 'local', 'local')
        result = RunResult(score=1.0, passed=True, output='', stderr='', exit_code=0,
                           execution_time_ms=1.0, sandbox_id='local')  # fmt: skip
        unjudged = {'score': 0.0, 'passed': False, 'exit_code': 2}
        store.complete(judged.id, result, 1.0)
        store.complete(
            crashed.id, result.model_copy(update={**unjudged, 'error_type': 'boom'}), 1.0
        )
        store.complete(silent.id, result.model_copy(update=unjudged), 1.0)

        stats = store.stats({}, hours=None)
        current = stats.current

        assert current.mean_score == pytest.approx(1 / 3)  # not 1.0: the two count as 0
        assert (current.judged_mean_score, current.judged_count, current.no_verdict_count) == (
            1.0, 1, 2,
        )  # fmt: skip
        assert current.errors == 2
        assert {count.error_type for count in stats.error_types} == {'boom', 'verifier_error'}
        assert (store.get(crashed.id).score, store.get(crashed.id).passed) == (None, None)

    @pytest.mark.parametrize('query', ['hours=0', 'hours=-1', 'status=nope'])
    def test_rejects_bad_queries(self, stats_client: TestClient, query: str) -> None:
        assert stats_client.get(f'/api/runs/stats?{query}').status_code == 422


class TestLocalExecPolicy:
    """Without --allow-local-exec only shipped reference solutions run on this machine."""

    def test_custom_local_solution_is_refused(self, tmp_path: Path) -> None:
        client = make_client(tmp_path)
        body = {'environments': ['reverse_string'], 'mode': 'local', 'solution': REVERSE_WRONG}

        response = client.post('/api/runs', json=body)

        assert response.status_code == 400
        assert '--allow-local-exec' in response.json()['detail']
        assert 'docker' in response.json()['detail']
        assert client.get('/api/runs').json() == []

    def test_reference_still_runs_locally(self, tmp_path: Path) -> None:
        client = make_client(tmp_path)
        response = client.post('/api/runs', json={'environments': ['fizzbuzz'], 'mode': 'local'})
        assert response.status_code == 202

    @pytest.mark.parametrize('allowed', [True, False])
    def test_config_reports_policy(self, tmp_path: Path, allowed: bool) -> None:
        config = make_client(tmp_path, allow_local_exec=allowed).get('/api/runs/config').json()
        assert config == {'local_custom_code': allowed}


class TestSandboxes:
    def test_lists_local_providers(self, client: TestClient) -> None:
        listing = client.get('/api/sandboxes').json()
        assert [p['provider'] for p in listing] == ['apple', 'docker']
        for provider in listing:
            assert provider['available'] or provider['error']

    @pytest.mark.parametrize('name', ['not-ours', 'deepgym-x;rm', 'deepgym-', 'my-deepgym-ab12'])
    def test_kill_refuses_other_names(self, client: TestClient, name: str) -> None:
        response = client.delete(f'/api/sandboxes/docker/{name}')
        assert response.status_code == 400
        assert 'not a DeepGym sandbox' in response.json()['detail']

    def test_kill_refuses_unknown_provider(self, client: TestClient) -> None:
        assert client.delete('/api/sandboxes/modal/deepgym-ab12').status_code == 422

    def test_parse_docker_listing(self) -> None:
        line = {'Names': 'deepgym-ab12', 'Image': 'python:3.12-slim', 'State': 'running',
                'CreatedAt': '2026-10-12 10:00:00 +0000 UTC'}  # fmt: skip
        (box,) = DockerContainer._parse_listing(json.dumps(line) + '\n')
        assert (box.id, box.image, box.state) == ('deepgym-ab12', 'python:3.12-slim', 'running')

    def test_parse_apple_listing(self) -> None:
        rows = [
            {'id': name, 'status': {'state': 'running', 'startedDate': '2026-10-12T10:00:00Z'},
             'configuration': {'image': {'reference': 'docker.io/library/python:3.12-slim'}}}
            for name in ('deepgym-ab12', 'someone-else')
        ]  # fmt: skip
        boxes = AppleContainer._parse_listing(json.dumps(rows))
        assert [box.id for box in boxes] == ['deepgym-ab12', 'someone-else']
        assert boxes[0].started == '2026-10-12T10:00:00Z'


class TestQA:
    def test_reads_reports_written_by_deepgym_qa(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('DEEPGYM_QA_DIR', str(tmp_path / 'qa'))
        for env, day, ok in [('fizzbuzz', 1, True), ('two_sum', 3, False), ('cyber/x', 2, True)]:
            checks = [
                QACheck(name='reference_passes', status='pass', details='1.00 x3'),
                QACheck(name='empty_fails', status='pass' if ok else 'fail', details='0.00'),
            ]
            created_at = datetime(2026, 10, day, tzinfo=timezone.utc)
            write_report(QAReport(environment=env, checks=checks, created_at=created_at))
        (tmp_path / 'qa' / 'broken.json').write_text('{not json')

        reports = client.get('/api/qa').json()
        fizzbuzz = client.get('/api/qa?environment=fizzbuzz').json()

        assert [r['environment'] for r in reports] == ['two_sum', 'cyber/x', 'fizzbuzz']
        assert [(r['status'], r['passed']) for r in reports] == [
            ('fail', False),
            ('pass', True),
            ('pass', True),
        ]
        assert reports[0]['checks'][1] == {
            'name': 'empty_fails',
            'status': 'fail',
            'passed': False,
            'details': '0.00',
        }
        assert [r['environment'] for r in fizzbuzz] == ['fizzbuzz']

    def test_missing_dir_is_empty(self, client: TestClient) -> None:
        assert client.get('/api/qa').json() == []
