"""Tests for comparing two console run groups: the run-store adapter and GET /api/runs/compare."""

import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from deepgym.compare import Attempt
from deepgym.console import runs as runs_module
from deepgym.console.run_store import RunStore, run_attempt
from deepgym.console.runner import Runner
from deepgym.console.schemas import RunSpec, RunSummary
from deepgym.console.security import TOKEN_HEADER
from deepgym.harbor import HarborEnvironment
from deepgym.models import Environment, RunResult
from deepgym.registry import load_environment
from deepgym.traces.schema import environment_sha256
from tests.console_client import make_client

HARBOR_HELLO = Path(__file__).resolve().parent / 'fixtures' / 'harbor' / 'tasks' / 'hello'


def _result(score: float, **fields: object) -> RunResult:
    base = {'score': score, 'passed': score == 1.0, 'output': '', 'stderr': '', 'exit_code': 0,
            'execution_time_ms': 1.0, 'sandbox_id': 'local'}  # fmt: skip
    return RunResult.model_validate(base | fields)


def _group(store: RunStore, plan: list[tuple[str, object]], sha: str = 'v1') -> str:
    """Queue one run per (environment, outcome) and finish it as *outcome* says.

    An outcome is a score, a RunResult, ``'fail'`` (the run raised) or ``'queued'``.
    """
    specs = [RunSpec(environment=env, solution='pass', reference=True,
                     environment_sha256=sha) for env, _ in plan]  # fmt: skip
    runs = store.create_group(specs, 'local', 'local')
    for run, (_, outcome) in zip(runs, plan):
        if outcome == 'fail':
            store.fail(run.id, 'sandbox_error', 'boom', 1.0)
        elif isinstance(outcome, RunResult):
            store.complete(run.id, outcome, 1.0)
        elif outcome != 'queued':
            store.complete(run.id, _result(outcome), 1.0)
    return runs[0].group_id


def _attempts(store: RunStore, group_id: str) -> dict[str, Attempt]:
    return {
        run.environment: run_attempt(run)
        for run in store.page({'group_id': group_id}, 0, None).runs
    }


def test_run_attempts(tmp_path: Path) -> None:
    store = RunStore(tmp_path / 'console.db')
    group = _group(
        store,
        [('a', 1.0), ('b', 0.4), ('c', _result(0.0, exit_code=2)), ('d', 'fail'),
         ('e', 'queued'), ('f', _result(0.0, error_type='solution_timeout')),
         ('g', _result(0.6, passed=True)), ('h', _result(1.0, passed=False))],
    )  # fmt: skip

    attempts = _attempts(store, group)

    fields = {key: (a.score, a.has_verdict, a.solved, a.error_type) for key, a in attempts.items()}
    assert fields == {
        'a': (1.0, True, True, None),
        'b': (0.4, True, False, None),
        'c': (0.0, False, False, 'verifier_error'),
        'd': (0.0, False, False, 'sandbox_error'),
        'e': (0.0, False, False, 'queued'),
        'f': (0.0, True, False, 'solution_timeout'),  # the model's fault: a judged 0
        'g': (0.6, True, True, None),  # solved is the verifier's pass, as the console counts it
        'h': (1.0, True, False, None),
    }
    assert {attempt.fingerprint for attempt in attempts.values()} == {'v1'}


def test_only_a_completed_run_has_a_verdict() -> None:
    run = RunSummary(id='r', group_id='g', environment='fizzbuzz', mode='local',
                     provider='local', reference=True, status='failed', score=1.0, passed=True,
                     created_at='t')  # fmt: skip

    attempt = run_attempt(run)

    assert (attempt.has_verdict, attempt.solved, attempt.error_type) == (False, False, 'failed')


def test_compare_round_trip(tmp_path: Path) -> None:
    store = RunStore(tmp_path / 'console.db')
    a = _group(store, [('fizzbuzz', 0.0), ('fizzbuzz', 0.0), ('two_sum', 1.0),
                       ('coin_change', 1.0), ('reverse_string', 1.0)])  # fmt: skip
    b = _group(store, [('fizzbuzz', 1.0), ('fizzbuzz', 1.0), ('two_sum', 0.0),
                       ('coin_change', 'queued'), ('word_count', 1.0)])  # fmt: skip
    client = make_client(tmp_path)  # fails the queued run as interrupted

    response = client.get(f'/api/runs/compare?a={a}&b={b}')

    assert response.status_code == 200
    report = response.json()
    rows = {row['key']: row for row in report['tasks']}
    assert (rows['fizzbuzz']['flip'], rows['fizzbuzz']['b']['attempts']) == ('improved', 2)
    assert rows['two_sum']['flip'] == 'regressed'
    assert report['uncomparable'] == ['coin_change']
    assert (report['only_a'], report['only_b']) == (['reverse_string'], ['word_count'])
    assert report['b']['no_verdict'] == {'interrupted': 1}
    assert report['significance']['test'] == 'exact_stratified_permutation'
    assert (report['k_limit'], report['k_limit_tasks']) == (1, ['two_sum'])
    assert report['a']['passk']['pass_at_k'] == {'1': 0.5}


def test_compare_flags_a_changed_environment(tmp_path: Path) -> None:
    store = RunStore(tmp_path / 'console.db')
    a = _group(store, [('fizzbuzz', 1.0)], sha='v1')
    b = _group(store, [('fizzbuzz', 1.0)], sha='v2')

    report = make_client(tmp_path).get(f'/api/runs/compare?a={a}&b={b}').json()

    assert (report['changed'], report['compared'], report['significance']) == (['fizzbuzz'], 0,
                                                                              None)  # fmt: skip


def test_compare_needs_two_known_groups(tmp_path: Path) -> None:
    group = _group(RunStore(tmp_path / 'console.db'), [('fizzbuzz', 1.0)])
    client = make_client(tmp_path)

    unknown = client.get(f'/api/runs/compare?a={group}&b=nope')

    assert unknown.status_code == 404
    assert unknown.json()['detail'] == 'Run group nope not found'
    assert client.get(f'/api/runs/compare?a={group}&b={group}').json()['compared'] == 1


@pytest.mark.parametrize(
    'query',
    ['a={g}', 'a={g}&b={g}&c=1', 'a={g}&a={g}&b={g}', 'a={g}&B={g}'],
    ids=['missing', 'unknown', 'repeated', 'miscased'],
)
def test_compare_refuses_a_bad_query_with_422(tmp_path: Path, query: str) -> None:
    group = _group(RunStore(tmp_path / 'console.db'), [('fizzbuzz', 1.0)])

    response = make_client(tmp_path).get(f'/api/runs/compare?{query.format(g=group)}')

    assert response.status_code == 422


def test_compare_waits_for_unfinished_groups(tmp_path: Path) -> None:
    client = make_client(tmp_path)  # started first, so it leaves these runs queued
    store = RunStore(tmp_path / 'console.db')
    done = _group(store, [('fizzbuzz', 1.0)])
    busy = _group(store, [('fizzbuzz', 1.0), ('two_sum', 'queued')])

    response = client.get(f'/api/runs/compare?a={done}&b={busy}')

    assert response.status_code == 409
    assert response.json()['detail'] == (
        f'Run group {busy} has 1 runs queued or running; compare it once they finish'
    )


def test_compare_needs_the_launch_token(tmp_path: Path) -> None:
    response = make_client(tmp_path).get('/api/runs/compare?a=x&b=y', headers={TOKEN_HEADER: ''})
    assert response.status_code == 403


def test_started_runs_keep_the_environment_hash(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        (queued,) = client.post('/api/runs', json={'environments': ['fizzbuzz']}).json()

    expected = environment_sha256(load_environment('fizzbuzz'))
    assert queued['environment_sha256'] == expected
    group = _attempts(RunStore(tmp_path / 'console.db'), queued['group_id'])
    assert group['fizzbuzz'].fingerprint == expected


def test_harbor_task_with_a_binary_test_file_is_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_dir = shutil.copytree(HARBOR_HELLO, tmp_path / 'hello')
    (task_dir / 'tests' / 'blob.bin').write_bytes(b'\xff\xfe\x00\x80')
    env = HarborEnvironment.from_dir(task_dir)
    monkeypatch.setattr(runs_module, 'load_environment', lambda name: env)

    with make_client(tmp_path) as client:
        response = client.post('/api/runs', json={'environments': ['hello']})

    assert response.status_code == 202
    assert response.json()[0]['environment_sha256'] == environment_sha256(env)


def test_an_environment_that_cannot_be_hashed_is_a_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = tmp_path / 'verifier.py'
    verifier.write_text('def verify(output):\n    return 1.0\n')
    env = Environment(name='gone', task='t', verifier_path=verifier)
    verifier.unlink()  # the hash reads the verifier's source
    monkeypatch.setattr(runs_module, 'load_environment', lambda name: env)

    with make_client(tmp_path, allow_local_exec=True) as client:
        response = client.post('/api/runs', json={'environments': ['gone'], 'solution': 'x'})
        listed = client.get('/api/runs').json()

    assert response.status_code == 400
    detail = response.json()['detail']
    assert detail.startswith("Can't hash gone's environment definition: [Errno 2]"), detail
    assert 'Error calling function' not in detail  # pydantic's wrapper is left out
    assert listed['total'] == 0


def test_a_run_stores_the_hash_of_what_it_ran(tmp_path: Path) -> None:
    store = RunStore(tmp_path / 'console.db')
    runner = Runner(store, allow_local_exec=False)
    env = load_environment('fizzbuzz')
    # As if the verifier changed while the run waited in the queue.
    spec = RunSpec(environment='fizzbuzz', solution=env.reference_solution, reference=True,
                   environment_sha256='hash-when-queued')  # fmt: skip
    try:
        (run,) = runner.submit(runner.client('local'), 'local', [(spec, env)])
        deadline = time.monotonic() + 60
        while store.get(run.id).status not in ('completed', 'failed'):
            assert time.monotonic() < deadline, 'the run did not finish'
            time.sleep(0.1)
    finally:
        runner.shutdown()

    assert store.get(run.id).environment_sha256 == environment_sha256(env)


class _Client:
    """Stands in for DeepGym in the runner: scores every run 1.0, editing the verifier
    file first when asked, as if someone changed it mid-run."""

    backend = 'local'

    def __init__(self, verifier: Path, edit: bool) -> None:
        self._verifier, self._edit = verifier, edit

    def run(self, env: Environment, solution: str) -> RunResult:
        if self._edit:
            self._verifier.write_text('def verify(output):\n    return 0.0\n')
        return _result(1.0)


@pytest.mark.parametrize('edit', [False, True])
def test_a_run_whose_environment_changes_while_it_runs_keeps_no_hash(
    tmp_path: Path, edit: bool
) -> None:
    verifier = tmp_path / 'verifier.py'
    verifier.write_text('def verify(output):\n    return 1.0\n')
    env = Environment(name='edited', task='t', verifier_path=verifier)
    before = environment_sha256(env)
    store = RunStore(tmp_path / 'console.db')
    runner = Runner(store, allow_local_exec=False)
    spec = RunSpec(environment='edited', solution='x', reference=False,
                   environment_sha256=before)  # fmt: skip
    try:
        client = _Client(verifier, edit)
        (run,) = runner.submit(client, 'local', [(spec, env)])  # type: ignore[arg-type]
        deadline = time.monotonic() + 60
        while store.get(run.id).status != 'completed':
            assert time.monotonic() < deadline, 'the run did not finish'
            time.sleep(0.05)
    finally:
        runner.shutdown()

    assert store.get(run.id).environment_sha256 == (None if edit else before)


def test_old_database_gains_the_hash_column_and_stores_verdicts_anew(tmp_path: Path) -> None:
    db = tmp_path / 'console.db'
    judged, unjudged = _result(1.0), _result(0.0, exit_code=2)
    with sqlite3.connect(db) as conn:
        conn.execute(
            'CREATE TABLE runs (id TEXT PRIMARY KEY, group_id TEXT NOT NULL, '
            'environment TEXT NOT NULL, mode TEXT NOT NULL, provider TEXT NOT NULL, '
            'reference INTEGER NOT NULL, solution TEXT NOT NULL, status TEXT NOT NULL, '
            'score REAL, passed INTEGER, error_type TEXT, error TEXT, duration_ms REAL, '
            'result TEXT, created_at TEXT NOT NULL, finished_at TEXT)'
        )
        conn.executemany(
            "INSERT INTO runs VALUES (?, 'g1', ?, 'local', 'local', 1, 'pass', ?, ?, ?, ?, "
            "NULL, 1.0, ?, 't1', 't2')",
            [
                ('r1', 'fizzbuzz', 'completed', 1.0, 1, None, judged.model_dump_json()),
                # Stored before results without a verdict kept no score.
                ('r2', 'two_sum', 'completed', 0.0, 0, None, unjudged.model_dump_json()),
                ('r3', 'coin_change', 'failed', 0.0, 0, 'sandbox_error', None),
                # Rows a whole RunResult no longer validates are read field by field.
                ('r4', 'word_count', 'completed', 1.0, 1, None, 'not json'),
                (
                    'r5',
                    'reverse_string',
                    'completed',
                    0.5,
                    1,
                    None,
                    '{"score": 0.5, "passed": true, "exit_code": 0, "execution_time_ms": "?"}',
                ),
            ],
        )
    conn.close()

    store = RunStore(db)

    attempts = _attempts(store, 'g1')
    fields = {key: (a.has_verdict, a.solved, a.error_type, a.fingerprint)
              for key, a in attempts.items()}  # fmt: skip
    assert fields == {
        'fizzbuzz': (True, True, None, None),
        'two_sum': (False, False, 'verifier_error', None),
        'coin_change': (False, False, 'sandbox_error', None),
        'word_count': (False, False, 'verifier_error', None),
        'reverse_string': (True, True, None, None),
    }
    assert [(store.get(r).score, store.get(r).passed) for r in ('r2', 'r3')] == [(None, None)] * 2
