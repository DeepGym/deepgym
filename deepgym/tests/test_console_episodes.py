"""Tests for recording and replaying episodes in the console."""

import base64
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deepgym.console import record_episode
from deepgym.console.episode_store import EpisodeStore
from deepgym.console.schemas import EpisodeCreate
from deepgym.models import Action, Observation, RunResult, Trajectory
from tests.console_client import make_client

PNG = b'\x89PNG\r\n\x1a\n fake image bytes'


def _trajectory() -> Trajectory:
    shot = base64.b64encode(PNG).decode()
    first = Observation(content='form loaded', step=0, screenshot=shot)
    second = Observation(content='name typed', step=1, screenshot=shot)
    final = Observation(content='saved', step=2, done=True, metadata={'exit_code': 0})
    return Trajectory(
        steps=[
            (first, Action(action_type='click', x=10, y=20)),
            (second, Action(action_type='done')),
        ],
        final_observation=final,
        total_reward=0.5,
        step_rewards=[0.25, 0.25],
    )


def _result(**changes: object) -> RunResult:
    fields = {
        'score': 1.0, 'passed': True, 'output': '3/3', 'stderr': '', 'exit_code': 0,
        'execution_time_ms': 5.0, 'sandbox_id': 'sb-1',
    }  # fmt: skip
    return RunResult.model_validate({**fields, **changes})


# A provider or verifier that failed gave no verdict; a 0 the policy earned, a model fault
# included, is one.
PROVIDER_ERROR = _result(
    score=0.0, passed=False, output='provider unreachable', exit_code=1, error_type='provider_error'
)
VERIFIER_CRASH = _result(score=0.0, passed=False, output='Traceback', exit_code=2)
SCORED_ZERO = _result(score=0.0, passed=False, output='0/3', exit_code=1)
MODEL_FAULT = _result(
    score=0.0, passed=False, output='timed out', exit_code=1, error_type='solution_timeout'
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / 'console.db'


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path)


def test_post_and_replay(client: TestClient) -> None:
    body = {
        'environment': 'signup_form',
        'trajectory': _trajectory().model_dump(mode='json'),
        'result': _result().model_dump(mode='json'),
    }

    response = client.post('/api/episodes', json=body)

    assert response.status_code == 201
    summary = response.json()
    assert (summary['steps'], summary['total_reward'], summary['passed']) == (2, 0.5, True)
    detail = client.get(f'/api/episodes/{summary["id"]}').json()
    frames = detail['frames']
    assert [f['content'] for f in frames] == ['form loaded', 'name typed', 'saved']
    assert frames[0]['action'] == {**frames[0]['action'], 'action_type': 'click', 'x': 10}
    assert [f['reward'] for f in frames] == [0.25, 0.25, None]
    assert [f['has_screenshot'] for f in frames] == [True, True, False]
    assert frames[2]['action'] is None
    assert detail['result']['output'] == '3/3'


def test_screenshots_are_served_as_png(client: TestClient) -> None:
    body = {'environment': 'signup_form', 'trajectory': _trajectory().model_dump(mode='json')}
    episode_id = client.post('/api/episodes', json=body).json()['id']

    shot = client.get(f'/api/episodes/{episode_id}/frames/1/screenshot')

    assert shot.status_code == 200
    assert shot.headers['content-type'] == 'image/png'
    assert shot.content == PNG
    assert client.get(f'/api/episodes/{episode_id}/frames/2/screenshot').status_code == 404


def test_bad_screenshot_is_rejected(client: TestClient) -> None:
    trajectory = _trajectory().model_dump(mode='json')
    trajectory['steps'][0][0]['screenshot'] = 'not base64!'

    response = client.post('/api/episodes', json={'trajectory': trajectory})

    assert response.status_code == 422
    assert client.get('/api/episodes').json() == {'episodes': [], 'total': 0}


def test_record_episode_helper(client: TestClient, db: Path) -> None:
    episode_id = record_episode(_trajectory(), _result(), environment='todo_list', db_path=db)

    listed = client.get('/api/episodes?environment=todo_list').json()

    assert [e['id'] for e in listed['episodes']] == [episode_id]
    assert (listed['total'], listed['episodes'][0]['score']) == (1, 1.0)
    assert client.get('/api/episodes?environment=other').json() == {'episodes': [], 'total': 0}


@pytest.mark.parametrize(
    ('result', 'verdict'),
    [
        (PROVIDER_ERROR, (None, None, False, 'provider_error')),
        (VERIFIER_CRASH, (None, None, False, 'verifier_error')),
        (SCORED_ZERO, (0.0, False, True, None)),
        (MODEL_FAULT, (0.0, False, True, 'solution_timeout')),
        (None, (None, None, None, None)),
    ],
    ids=['provider_error', 'verifier_crash', 'scored_zero', 'model_fault', 'unscored'],
)
def test_episode_without_a_verdict_is_not_a_scored_zero(
    client: TestClient,
    db: Path,
    result: RunResult | None,
    verdict: tuple[float | None, bool | None, bool | None, str | None],
) -> None:
    record_episode(_trajectory(), result, environment='todo_list', db_path=db)

    (listed,) = client.get('/api/episodes').json()['episodes']
    detail = client.get(f'/api/episodes/{listed["id"]}').json()

    for episode in (listed, detail):
        fields = (
            episode['score'],
            episode['passed'],
            episode['has_verdict'],
            episode['error_type'],
        )
        assert fields == verdict
    assert detail['result'] == (None if result is None else result.model_dump(mode='json'))


def test_old_database_gains_verdicts_from_stored_results(tmp_path: Path, db: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            'CREATE TABLE episodes (id TEXT PRIMARY KEY, environment TEXT NOT NULL, '
            'steps INTEGER NOT NULL, total_reward REAL NOT NULL, score REAL, passed INTEGER, '
            'frames TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL, trace TEXT)'
        )
        conn.executemany(
            "INSERT INTO episodes VALUES (?, 'e', 0, 0, ?, ?, '[]', ?, ?, NULL)",
            [
                ('crashed', 0.0, False, PROVIDER_ERROR.model_dump_json(), 't3'),
                ('zero', 0.0, False, SCORED_ZERO.model_dump_json(), 't2'),
                ('unscored', None, None, None, 't1'),
            ],
        )
    conn.close()

    listed = make_client(tmp_path).get('/api/episodes').json()['episodes']

    assert [(e['id'], e['score'], e['has_verdict'], e['error_type']) for e in listed] == [
        ('crashed', None, False, 'provider_error'),
        ('zero', 0.0, True, None),
        ('unscored', None, None, None),
    ]


def test_every_episode_is_reachable_a_page_at_a_time(client: TestClient, db: Path) -> None:
    store = EpisodeStore(db)
    episode = EpisodeCreate(environment='todo_list', trajectory=Trajectory())
    newest_first = [store.add(episode).id for _ in range(201)][::-1]

    everything = client.get('/api/episodes').json()
    first = client.get('/api/episodes?limit=25').json()
    last = client.get('/api/episodes?offset=200&limit=25').json()

    assert everything['total'] == 201
    assert [e['id'] for e in everything['episodes']] == newest_first
    assert (first['total'], [e['id'] for e in first['episodes']]) == (201, newest_first[:25])
    assert (last['total'], [e['id'] for e in last['episodes']]) == (201, newest_first[200:])


def test_unknown_episode_is_404(client: TestClient) -> None:
    assert client.get('/api/episodes/nope').status_code == 404
