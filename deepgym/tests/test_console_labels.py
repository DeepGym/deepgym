"""Tests for the console's label API: store, correct, retract, list, current and excluded."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.console_client import make_client

STEP = {'kind': 'trace', 'id': 't1', 'step': 3, 'field': 'observation', 'span': [10, 14],
        'quote': 'diff'}  # fmt: skip
TRACE = {'target_kind': 'trace', 'target_id': 't1'}


def _body(**changes: object) -> dict:
    """An expert's passed verdict on trace t1, with *changes*."""
    fields = {
        'target': {'kind': 'trace', 'id': 't1'},
        'name': 'passed',
        'value': True,
        'source': 'human',
        'author': 'ana',
        'comment': 'Solves it.',
    }
    return {**fields, **changes}


def _whole(kind: str, comment: str, **changes: object) -> dict:
    """An invalidation or retraction body on trace t1."""
    return _body(type=kind, name=None, value=None, comment=comment, **changes)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path)


def test_post_correct_retract_and_read_current(client: TestClient) -> None:
    response = client.post('/api/labels', json=_body(blind=True, comment='clean solution'))
    assert response.status_code == 201
    first = response.json()
    assert (first['seq'], first['value'], first['blind']) == (1, True, True)
    assert first['id'] and first['created_at']

    step = client.post(
        '/api/labels', json=_body(target=STEP, name='mark', value='wrong-file', comment=None)
    ).json()
    fixed = client.post(
        '/api/labels', json=_body(value=False, comment='misses a case', supersedes=first['id'])
    ).json()
    retraction = client.post(
        '/api/labels', json=_whole('retract', 'meant t2', supersedes=step['id'])
    ).json()

    assert client.get('/api/labels/current', params=TRACE).json() == [fixed]
    assert client.get('/api/labels').json() == {
        'labels': [first, step, fixed, retraction],
        'total': 4,
    }
    assert client.get('/api/labels', params={'current': 'true'}).json() == {
        'labels': [fixed],
        'total': 1,
    }
    assert client.get('/api/labels', params={'after_seq': 2}).json()['labels'] == [
        fixed,
        retraction,
    ]


def test_excluded_targets(client: TestClient) -> None:
    run = {'kind': 'run', 'id': 'r1'}
    by_bot = client.post(
        '/api/labels',
        json=_whole('invalidate', 'flaky sandbox', source='verifier', author='qa-bot'),
    ).json()
    client.post('/api/labels', json=_whole('invalidate', 'stale image', target=run))

    assert client.get('/api/labels/excluded').json() == [
        {'kind': 'run', 'id': 'r1'},
        {'kind': 'trace', 'id': 't1'},
    ]

    lifted = client.post(
        '/api/labels',
        json=_whole('retract', 'reran it: fine', author='lead', supersedes=by_bot['id']),
    )

    assert lifted.status_code == 201
    assert client.get('/api/labels/excluded').json() == [{'kind': 'run', 'id': 'r1'}]


@pytest.mark.parametrize(
    ('body', 'code'),
    [
        (_body(comment=None), 422),
        (_body(extra=1), 422),
        (_body(name='c1', value='MET'), 422),
        (_body(target=STEP), 422),
        (_body(blind='true'), 422),
        (_body(author='ana '), 422),
        (_body(author='a\x00na'), 422),
        (_body(target={'kind': 'trace', 'id': 't1', 'step': 2**63}, name='mark'), 422),
        (_whole('invalidate', 'broken', target={'kind': 'trace', 'id': 't1', 'step': 1}), 422),
        (_body(supersedes='nope'), 409),
    ],
)  # fmt: skip
def test_refused_labels(client: TestClient, body: dict, code: int) -> None:
    assert client.post('/api/labels', json=body).status_code == code
    assert client.get('/api/labels').json()['total'] == 0


def test_corrections_stay_with_their_author(client: TestClient) -> None:
    first = client.post('/api/labels', json=_body()).json()

    other = client.post(
        '/api/labels', json=_body(author='lead', value=False, supersedes=first['id'])
    )

    assert other.status_code == 409
    assert "another labeler's view is a new label" in other.json()['detail']


def test_a_second_verdict_must_supersede(client: TestClient) -> None:
    first = client.post('/api/labels', json=_body()).json()

    again = client.post('/api/labels', json=_body(value=False, comment='misses a case'))
    other = client.post('/api/labels', json=_body(author='bo', value=False, comment='no'))

    assert again.status_code == 409
    assert 'supersede it' in again.json()['detail']
    assert other.status_code == 201
    assert client.get('/api/labels', params={'current': 'true', 'author': 'ana'}).json() == {
        'labels': [first],
        'total': 1,
    }


def test_a_retry_with_the_same_id_stores_once(client: TestClient) -> None:
    first = client.post('/api/labels', json=_body(id='ana-1'))
    again = client.post('/api/labels', json=_body(id='ana-1'))
    changed = client.post('/api/labels', json=_body(id='ana-1', value=False))

    assert (first.status_code, again.status_code) == (201, 201)
    assert again.json() == first.json()
    assert first.json()['id'] == 'ana-1'
    assert changed.status_code == 409
    assert 'another body' in changed.json()['detail']
    assert client.get('/api/labels').json()['total'] == 1


def test_batch_is_all_or_nothing(client: TestClient) -> None:
    good = [_body(name=f'c{n}') for n in (1, 2)]

    invalid = client.post('/api/labels/batch', json=[*good, _body(author='')])
    conflict = client.post('/api/labels/batch', json=[*good, _body(supersedes='nope')])
    twice = client.post('/api/labels/batch', json=[*good, _body(name='c1', value=False)])

    assert invalid.status_code == 422
    assert (conflict.status_code, twice.status_code) == (409, 409)
    assert 'not stored' in conflict.json()['detail']
    assert client.get('/api/labels').json() == {'labels': [], 'total': 0}

    stored = client.post('/api/labels/batch', json=good)

    assert stored.status_code == 201
    assert [label['seq'] for label in stored.json()] == [1, 2]
    assert client.get('/api/labels').json()['total'] == 2


def test_list_filters_and_pages(client: TestClient) -> None:
    sha = 'cd' * 32
    judged = {'source': 'judge', 'author': 'judge-a', 'rubric_sha256': sha, 'group_id': 'g1',
              'rubric_version': 'rubric-judge/1', 'comment': None}  # fmt: skip
    stored = [
        client.post('/api/labels', json=body).json()
        for body in (
            _body(blind=True),
            _body(target={'kind': 'run', 'id': 'r1'}, name='score', value=0.5,
                  source='verifier', author='fizzbuzz', comment=None),
            _body(name='c1', blind=False, **judged),
            _body(target=STEP, name='c1-note', value='typo'),
            _whole('invalidate', 'stale image', target={'kind': 'run', 'id': 'r1'}),
        )
    ]  # fmt: skip

    def listed(**params: object) -> tuple[list[int], int]:
        page = client.get('/api/labels', params=params).json()
        return [label['seq'] for label in page['labels']], page['total']

    assert listed(target_kind='trace', source='judge') == ([3], 1)
    assert listed(source='human', author='ana') == ([1, 4, 5], 3)
    assert listed(target_kind='run', target_id='r1') == ([2, 5], 2)
    assert listed(type='invalidate') == ([5], 1)
    assert listed(step=3) == ([4], 1)
    assert listed(blind='true') == ([1], 1)
    assert listed(blind='false') == ([3], 1)
    assert listed(rubric_sha256=sha, rubric_version='rubric-judge/1', group_id='g1') == ([3], 1)
    assert listed(after_seq=3) == ([4, 5], 2)
    assert listed(current='true', target_kind='run') == ([2, 5], 2)
    assert listed(offset=1, limit=2) == ([2, 3], 5)
    assert listed(since=stored[2]['created_at']) == ([3, 4, 5], 3)
    assert listed(until=stored[2]['created_at']) == ([1, 2], 2)


@pytest.mark.parametrize(
    'query',
    [
        'target_kind=job',
        'source=expert',
        'type=delete',
        'blind=maybe',
        'current=maybe',
        'since=2026-11-19T10:00:00',
        'since=0001-01-01T00:00:00%2B05:00',
        'until=9999-12-31T23:59:59-10:00',
        'limit=0',
        'offset=-1',
        f'offset={2**63}',
        f'limit={2**63}',
        f'after_seq={2**63}',
        'stepp=1',
        'target=t1',
        'name=c1&name=c2',
        'author=ana&author=ana',
    ],
)
def test_bad_query_is_refused(client: TestClient, query: str) -> None:
    assert client.get(f'/api/labels?{query}').status_code == 422


def test_largest_page_bounds_are_accepted(client: TestClient) -> None:
    bound = 2**63 - 1
    params = {'offset': bound, 'limit': bound, 'after_seq': bound}
    assert client.get('/api/labels', params=params).json() == {'labels': [], 'total': 0}


@pytest.mark.parametrize(
    ('path', 'query'),
    [
        ('/api/labels/current', 'target_kind=trace'),
        ('/api/labels/current', 'target_kind=trace&target_id=t1&step=1'),
        ('/api/labels/current', 'target_kind=trace&target_id=t1&target_id=t2'),
        ('/api/labels/excluded', 'target_kind=trace'),
    ],
)
def test_current_and_excluded_take_only_their_parameters(
    client: TestClient, path: str, query: str
) -> None:
    assert client.get(f'{path}?{query}').status_code == 422


@pytest.mark.parametrize(
    ('path', 'body'), [('/api/labels', _body()), ('/api/labels/batch', [_body()])]
)
def test_posts_take_no_query_parameters(client: TestClient, path: str, body: object) -> None:
    assert client.post(f'{path}?dry_run=1', json=body).status_code == 422
    assert client.get('/api/labels').json()['total'] == 0


def test_writes_need_token_and_origin(client: TestClient) -> None:
    no_token = client.post('/api/labels', json=_body(), headers={'X-DeepGym-Token': 'wrong'})
    foreign = client.post('/api/labels', json=_body(), headers={'Origin': 'https://evil.example'})

    assert (no_token.status_code, foreign.status_code) == (403, 403)
    assert client.get('/api/labels').json()['total'] == 0
