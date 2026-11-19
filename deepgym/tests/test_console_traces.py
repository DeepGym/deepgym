"""Tests for the console's trace and scan API: list and read traces, scan them, store findings."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from deepgym.console import record_episode, scans
from deepgym.console.db import connect
from deepgym.console.episode_store import read_held_traces
from deepgym.console.label_store import LabelStore
from deepgym.console.scans import CLIENT_CLOSED, scan_in_process
from deepgym.labels import LabelCreate
from deepgym.models import Trajectory
from deepgym.scanners import PatternScanner, ScanReport, finding_labels, scan_traces
from deepgym.traces import Trace
from tests.console_client import make_client
from tests.test_dataset import _trace, _unjudged
from tests.trace_samples import graded_trace

ECHO = {'kind': 'pattern', 'name': 'echo', 'patterns': [{'regex': 'echo', 'message': 'echoes'}]}
RUNAWAY = {**ECHO, 'patterns': [{'regex': '(a+)+$', 'message': 'backtracks'}]}
"""A regex that backtracks for ever on 'aaa...ab'."""


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / 'console.db'


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path)


@pytest.fixture()
def started(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[Any]]:
    """Every process started while the test runs."""
    processes: list[subprocess.Popen[Any]] = []
    real = subprocess.Popen

    def recorded(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        processes.append(real(*args, **kwargs))
        return processes[-1]

    monkeypatch.setattr(subprocess, 'Popen', recorded)
    return processes


def _record(db: Path, *traces: Trace) -> None:
    for trace in traces:
        record_episode(trace, db_path=db)


def _three(db: Path) -> None:
    """t1 and t2 are judged, t3 isn't; t3 started last and t1 first."""
    _record(
        db,
        _trace('t1', 'a', started=1),
        _trace('t2', 'b', started=2, env_name='other', agent='coder', model='m2'),
        _unjudged('t3', 'c', started=3),
    )


def _ids(response: object) -> list[str]:
    return [trace['trace_id'] for trace in response.json()['traces']]  # type: ignore[attr-defined]


def _corrupt(db: Path, trace: Trace, stored: str) -> str:
    """Record *trace*, then overwrite its stored copy with *stored*; return the episode id."""
    episode_id = record_episode(trace, db_path=db)
    with connect(db) as conn:
        conn.execute('UPDATE episodes SET trace = ? WHERE id = ?', (stored, episode_id))
    return episode_id


def test_list_traces_newest_first_with_filters_and_totals(client: TestClient, db: Path) -> None:
    _three(db)
    record_episode(Trajectory(), environment='no_trace', db_path=db)  # an episode, no trace

    every = client.get('/api/traces').json()

    assert [trace['trace_id'] for trace in every['traces']] == ['t3', 't2', 't1']
    assert every['total'] == 3
    first = every['traces'][2]
    assert (first['env_name'], first['agent'], first['model']) == (
        'write_notes',
        'react-shell',
        'm1',
    )
    assert (first['has_verdict'], first['passed'], first['score']) == (True, True, 1.0)
    assert (every['traces'][0]['has_verdict'], every['traces'][0]['score']) == (False, None)
    assert _ids(client.get('/api/traces', params={'environment': 'other'})) == ['t2']
    assert _ids(client.get('/api/traces', params={'agent': 'react-shell'})) == ['t3', 't1']
    assert _ids(client.get('/api/traces', params={'model': 'm2'})) == ['t2']
    assert _ids(client.get('/api/traces', params={'has_verdict': 'false'})) == ['t3']
    page = client.get('/api/traces', params={'has_verdict': 'true', 'offset': 1, 'limit': 5})
    assert page.json()['total'] == 2
    assert _ids(page) == ['t1']
    assert client.get('/api/traces', params={'offset': 3}).json() == {
        'traces': [],
        'total': 3,
        'conflicts': [],
        'unreadable': [],
    }


def test_get_trace_with_usage_totals_and_current_labels(client: TestClient, db: Path) -> None:
    trace = graded_trace()
    _record(db, trace)
    store = LabelStore(db)
    target = {'kind': 'trace', 'id': trace.trace_id}
    first, step = store.add_many(
        [
            LabelCreate.model_validate({'target': target, 'name': 'tone', 'value': 'curt',
                                        'source': 'human', 'author': 'ana'}),
            LabelCreate.model_validate({'target': {**target, 'step': 2}, 'name': 'mark',
                                        'value': 'rejected', 'source': 'human', 'author': 'ana'}),
        ]
    )  # fmt: skip
    fixed = store.add(
        LabelCreate.model_validate({'target': target, 'name': 'tone', 'value': 'fine',
                                    'source': 'human', 'author': 'ana', 'supersedes': first.id})
    )  # fmt: skip

    body = client.get(f'/api/traces/{trace.trace_id}').json()

    assert body['trace'] == json.loads(trace.model_dump_json())
    assert body['trace']['usage'] == trace.usage.model_dump(mode='json')  # type: ignore[union-attr]
    assert body['trace']['cost_usd'] == trace.cost_usd
    assert [label['id'] for label in body['labels']] == [step.id, fixed.id]
    current = {'target_kind': 'trace', 'target_id': trace.trace_id}
    assert body['labels'] == client.get('/api/labels/current', params=current).json()
    assert (body['episode_labels'], body['unreadable']) == ([], [])
    assert client.get('/api/traces/nope').status_code == 404


def test_trace_detail_has_the_labels_on_its_episodes(client: TestClient, db: Path) -> None:
    trace = _trace('t1')
    first = record_episode(
        trace.model_copy(update={'result': None, 'finished_at': None}), db_path=db
    )
    second = record_episode(trace, db_path=db)
    record_episode(_trace('t2'), db_path=db)
    store = LabelStore(db)
    on_frame, on_other = store.add_many(
        [
            LabelCreate.model_validate({'target': {'kind': 'episode', 'id': second, 'step': 1},
                                        'name': 'mark', 'value': 'slow', 'source': 'human',
                                        'author': 'ana'}),
            LabelCreate.model_validate({'target': {'kind': 'episode', 'id': first},
                                        'name': 'mark', 'value': 'early', 'source': 'human',
                                        'author': 'bo'}),
        ]
    )  # fmt: skip

    body = client.get('/api/traces/t1').json()

    assert [label['id'] for label in body['episode_labels']] == [on_frame.id, on_other.id]
    assert body['labels'] == []


def test_one_trace_is_read_from_its_own_rows(db: Path) -> None:
    _three(db)
    _corrupt(db, _trace('t4', 'd'), '{"not": "json"')

    held = read_held_traces(db, 't2')

    assert [trace.trace_id for trace in held.traces] == ['t2']
    assert (held.untraced, held.conflicts, held.unreadable) == (0, (), ())


@pytest.mark.parametrize(
    ('path', 'params'),
    [
        ('/api/traces', {'env': 'other'}),
        ('/api/traces', [('agent', 'a'), ('agent', 'b')]),
        ('/api/traces', {'has_verdict': 'maybe'}),
        ('/api/traces', {'limit': 0}),
        ('/api/traces', {'offset': -1}),
        ('/api/traces/t1', {'labels': 'true'}),
    ],
)
def test_trace_queries_are_strict(client: TestClient, db: Path, path: str, params: object) -> None:
    _three(db)
    assert client.get(path, params=params).status_code == 422  # type: ignore[arg-type]


def test_disagreeing_copies_are_left_out_and_reported(client: TestClient, db: Path) -> None:
    trace = _trace('t1')
    first = record_episode(trace, db_path=db)
    second = record_episode(trace.model_copy(update={'task': 'Another task'}), db_path=db)
    _record(db, _trace('t2', 'b'))

    listed = client.get('/api/traces').json()
    detail = client.get('/api/traces/t1')
    other = client.get('/api/traces/t2')
    scanned = client.post('/api/scans', json={'scanners': [ECHO]})
    named = client.post('/api/scans', json={'scanners': [ECHO], 'trace_ids': ['t1', 't2']})

    assert (_ids(client.get('/api/traces')), listed['total']) == (['t2'], 1)
    (conflict,) = listed['conflicts']
    assert (conflict['trace_id'], conflict['reason'], conflict['episodes']) == (
        't1',
        'copies_disagree',
        [first, second],
    )
    assert (detail.status_code, other.status_code) == (409, 200)
    assert "['t1'] disagree" in detail.json()['detail']
    assert scanned.status_code == 200
    assert (scanned.json()['traces'], scanned.json()['conflicts']) == (['t2'], [conflict])
    assert named.status_code == 409


def test_ids_differing_only_in_case_are_left_out(client: TestClient, db: Path) -> None:
    _record(db, _trace('Ab', 'a'), _trace('ab', 'b'), _trace('t1', 'c'))

    listed = client.get('/api/traces').json()

    assert _ids(client.get('/api/traces')) == ['t1']
    assert [(c['trace_id'], c['reason']) for c in listed['conflicts']] == [
        ('Ab', 'case_collision'),
        ('ab', 'case_collision'),
    ]
    assert client.get('/api/traces/Ab').status_code == 409
    assert client.get('/api/traces/AB').status_code == 404


def test_a_stored_trace_that_does_not_parse_is_left_out(client: TestClient, db: Path) -> None:
    _three(db)
    broken = _corrupt(db, _trace('t4', 'd'), '{"trace_id": "t4"')
    wrong = _corrupt(db, _trace('t5', 'e'), '{"trace_id": "t5", "task": 1}')
    # pydantic reads NaN, SQLite's json_valid doesn't: every route calls it unreadable.
    text = _trace('t6', 'f').model_dump_json().replace('"exit_code":0', '"exit_code":NaN', 1)
    nan = _corrupt(db, _trace('t6', 'f'), text)

    listed = client.get('/api/traces').json()
    missing = client.get('/api/traces/t5')
    scanned = client.post('/api/scans', json={'scanners': [ECHO]}).json()

    assert (listed['total'], listed['unreadable']) == (3, [broken, wrong, nan])
    assert client.get('/api/traces/t1').status_code == 200
    assert missing.status_code == 404
    assert missing.json()['detail'] == (
        f"Trace t5 not found; episodes ['{wrong}'] hold copies that do not read"
    )
    assert client.get('/api/traces/t6').json() == {'detail': 'Trace t6 not found'}
    assert (sorted(scanned['traces']), scanned['unreadable']) == (
        ['t1', 't2', 't3'],
        [broken, wrong, nan],
    )


def test_an_unfinished_copy_merges_into_the_finished_one(client: TestClient, db: Path) -> None:
    trace = _trace('t1')
    _record(db, trace.model_copy(update={'result': None, 'finished_at': None}), trace)

    assert client.get('/api/traces').json()['total'] == 1
    assert client.get('/api/traces/t1').json()['trace']['has_verdict'] is True


def test_scan_returns_the_report_and_stores_nothing(client: TestClient, db: Path) -> None:
    _three(db)

    response = client.post('/api/scans', json={'scanners': [ECHO]})

    assert response.status_code == 200
    report = response.json()
    assert report['traces'] == ['t1', 't2', 't3']
    assert [(finding['trace_id'], finding['step'], finding['quote'])
            for finding in report['findings']] == [('t1', 1, 'echo'), ('t2', 1, 'echo'),
                                                    ('t3', 1, 'echo')]  # fmt: skip
    assert report['errors'] == []
    assert client.get('/api/labels').json()['total'] == 0


def test_scan_chosen_traces_each_once(client: TestClient, db: Path) -> None:
    _three(db)

    report = client.post('/api/scans', json={'scanners': [ECHO], 'trace_ids': ['t2', 't1', 't2']})
    missing = client.post('/api/scans', json={'scanners': [ECHO], 'trace_ids': ['t1', 'nope']})

    assert report.json()['traces'] == ['t2', 't1']
    assert missing.status_code == 404
    assert "['nope']" in missing.json()['detail']


def test_scan_writes_the_labels_finding_labels_gives(client: TestClient, db: Path) -> None:
    _three(db)

    report = client.post('/api/scans', json={'scanners': [ECHO], 'write_labels': True}).json()

    stored = client.get('/api/labels').json()
    expected = finding_labels(ScanReport.model_validate(report))
    assert stored['total'] == len(expected) == 3
    assert [{name: label[name] for name in LabelCreate.model_fields}
            for label in stored['labels']] == [
        json.loads(label.model_dump_json()) for label in expected
    ]  # fmt: skip
    assert {label['group_id'] for label in stored['labels']} == {report['run_id']}


def test_a_scan_retried_with_its_run_id_stores_nothing_new(client: TestClient, db: Path) -> None:
    _three(db)
    body = {'scanners': [ECHO], 'write_labels': True, 'run_id': 'nightly-1'}

    first = client.post('/api/scans', json=body)
    again = client.post('/api/scans', json=body)

    assert (first.status_code, again.status_code) == (200, 200)
    assert first.json()['run_id'] == again.json()['run_id'] == 'nightly-1'
    stored = client.get('/api/labels').json()
    assert stored['total'] == 3
    assert {label['id'] for label in stored['labels']} == {f'nightly-1-{n}' for n in range(3)}


def test_scan_process_report_matches_scan_traces() -> None:
    traces = [_trace('t1'), _trace('t2', 'b')]
    scanners = [PatternScanner.model_validate(ECHO)]

    async def connected() -> bool:
        return False

    report = anyio.run(scan_in_process, traces, scanners, connected)

    expected = scan_traces(traces, scanners)
    assert (report.traces, report.findings, report.errors) == (
        expected.traces,
        expected.findings,
        expected.errors,
    )


def test_a_runaway_scan_is_killed_when_its_client_leaves(
    started: list[subprocess.Popen[Any]],
) -> None:
    """A regex that backtracks for ever runs in its own process: the event loop keeps
    turning, and the process is killed once the client is gone."""
    traces = [_trace('t1', command='a' * 64 + 'b')]
    ticks = 0

    async def left() -> bool:
        return ticks >= 3

    async def tick() -> None:
        nonlocal ticks
        while ticks < 3:
            await anyio.sleep(0.05)
            ticks += 1

    async def main() -> HTTPException:
        async with anyio.create_task_group() as group:
            group.start_soon(tick)
            try:
                await scan_in_process(traces, [PatternScanner.model_validate(RUNAWAY)], left)
            except HTTPException as exc:
                return exc
        raise AssertionError('the scan was never stopped')

    stopped = anyio.run(main)

    assert stopped.status_code == CLIENT_CLOSED
    assert ticks == 3
    (process,) = started
    assert process.returncode == -9  # killed, and reaped


def test_a_cancelled_scan_still_reaps_its_process(started: list[subprocess.Popen[Any]]) -> None:
    traces = [_trace('t1', command='a' * 64 + 'b')]

    async def stays() -> bool:
        return False

    async def main() -> None:
        with anyio.move_on_after(0.5):
            await scan_in_process(traces, [PatternScanner.model_validate(RUNAWAY)], stays)

    anyio.run(main)

    (process,) = started
    assert process.returncode == -9


def test_a_scan_process_that_dies_is_a_server_error(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _three(db)
    monkeypatch.setattr(scans, 'SCAN_COMMAND', [sys.executable, '-c', 'raise SystemExit(3)'])

    response = client.post('/api/scans', json={'scanners': [ECHO], 'write_labels': True})

    assert (response.status_code, response.json()['detail']) == (
        500,
        'The scan process ended with exit code 3 without a report it could read',
    )
    assert client.get('/api/labels').json()['total'] == 0


@pytest.mark.parametrize('written', ['not json', '{}', '{"report": {"traces": 1}}'])
def test_a_scan_process_writing_nonsense_is_a_server_error(
    client: TestClient, db: Path, monkeypatch: pytest.MonkeyPatch, written: str
) -> None:
    _three(db)
    code = f'import sys; sys.stdin.buffer.read(); sys.stdout.write({written!r})'
    monkeypatch.setattr(scans, 'SCAN_COMMAND', [sys.executable, '-I', '-c', code])

    response = client.post('/api/scans', json={'scanners': [ECHO], 'write_labels': True})

    assert (response.status_code, response.json()['detail']) == (
        500,
        'The scan process ended with exit code 0 without a report it could read',
    )
    assert client.get('/api/labels').json()['total'] == 0


def test_the_scan_process_ignores_modules_planted_in_the_working_directory(
    client: TestClient, db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Modules named like the standard library and deepgym's dependencies, in the server's
    working directory and on PYTHONPATH, are never imported by the scan process."""
    _three(db)
    planted = tmp_path / 'planted'
    planted.mkdir()
    for name in ('json', 'pydantic', 'anyio', 'deepgym'):
        (planted / f'{name}.py').write_text(
            'from pathlib import Path\n'
            f"Path(__file__).with_name('{name}.imported').write_text('imported')\n"
            f"raise ImportError('planted {name} was imported')\n",
            encoding='utf-8',
        )
    monkeypatch.chdir(planted)
    monkeypatch.setenv('PYTHONPATH', str(planted))

    response = client.post('/api/scans', json={'scanners': [ECHO]})

    assert response.status_code == 200, response.text
    assert len(response.json()['findings']) == 3
    assert sorted(path.name for path in planted.glob('*.imported')) == []


def test_a_scan_never_runs_the_callers_main_module_again(tmp_path: Path) -> None:
    """A script that scans at import time (as one building the app at import would) runs
    once: the scan process imports only deepgym.console.scans."""
    marker = tmp_path / 'imports.txt'
    script = tmp_path / 'scan_at_import.py'
    script.write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'import anyio\n'
        'from deepgym.console.scans import scan_in_process\n'
        'from deepgym.scanners import PatternScanner\n'
        "with Path(sys.argv[1]).open('a') as file:\n"
        "    file.write('imported\\n')\n"
        'async def connected():\n'
        '    return False\n'
        f'scanner = PatternScanner.model_validate({ECHO!r})\n'
        'print(anyio.run(scan_in_process, [], [scanner], connected).traces)\n',
        encoding='utf-8',
    )
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(path for path in sys.path if path)}

    done = subprocess.run(
        [sys.executable, str(script), str(marker)], env=env, capture_output=True, text=True
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout == '()\n'
    assert marker.read_text(encoding='utf-8') == 'imported\n'


def test_scan_labels_are_stored_all_or_none(client: TestClient, db: Path) -> None:
    _three(db)
    taken = LabelCreate.model_validate({'id': 'run-1', 'name': 'other', 'value': 1,
                                        'target': {'kind': 'trace', 'id': 't1'},
                                        'source': 'human', 'author': 'ana'})  # fmt: skip
    LabelStore(db).add(taken)

    response = client.post(
        '/api/scans', json={'scanners': [ECHO], 'write_labels': True, 'run_id': 'run'}
    )

    assert response.status_code == 409
    assert 'stored with another body' in response.json()['detail']
    assert [label['id'] for label in client.get('/api/labels').json()['labels']] == [taken.id]


@pytest.mark.parametrize(
    'body',
    [
        {'scanners': []},
        {'scanners': [{**ECHO, 'kind': 'llm'}]},
        {'scanners': [{**ECHO, 'patterns': [{'regex': '(', 'message': 'm'}]}]},
        {'scanners': [ECHO, ECHO]},
        {'scanners': [{**ECHO, 'name': 'passed'}]},
        {'scanners': [ECHO], 'trace_ids': []},
        {'scanners': [ECHO], 'extra': 1},
    ],
)
def test_refused_scans(client: TestClient, db: Path, body: dict) -> None:
    _three(db)
    assert client.post('/api/scans', json=body).status_code == 422
    assert client.get('/api/labels').json()['total'] == 0


def test_scan_takes_no_query_parameters(client: TestClient) -> None:
    response = client.post('/api/scans', params={'write_labels': 'true'}, json={'scanners': [ECHO]})
    assert response.status_code == 422
