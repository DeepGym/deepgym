"""Tests for the console's review queue API: queues, items, reservations, reviews, progress."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deepgym.console.queue_store import QueueStore
from tests.console_client import make_client

PASSED = {'name': 'passed', 'value': False, 'comment': 'Opens port 80 instead.'}


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path)


def _queue(client: TestClient, items: int = 2, **settings: object) -> str:
    """Create a queue of traces t1 to t<items> asking for a passed verdict; return its id."""
    body = {'name': 'calibration', 'label_names': ['passed'], **settings}
    queue_id = client.post('/api/queues', json=body).json()['id']
    targets = [{'target': {'kind': 'trace', 'id': f't{n}'}} for n in range(1, items + 1)]
    client.post(f'/api/queues/{queue_id}/items', json=targets)
    return queue_id


def test_review_round_trip(client: TestClient) -> None:
    created = client.post(
        '/api/queues',
        json={'id': 'q1', 'name': 'calibration', 'instructions': 'Read the diff.',
              'label_names': ['passed', 'c1'], 'overlap': 2, 'reviewers': ['ana', 'bo']},
    )  # fmt: skip
    added = client.post(
        '/api/queues/q1/items',
        json=[{'target': {'kind': 'run', 'id': 'r1'}, 'context': 'Fix the port.'},
              {'target': {'kind': 'trace', 'id': 't1'}}],
    )  # fmt: skip
    client.post(
        '/api/labels',
        json={'target': {'kind': 'run', 'id': 'r1'}, 'name': 'passed', 'value': True,
              'source': 'judge', 'author': 'judge-a'},
    )  # fmt: skip

    assert created.status_code == 201
    queue = created.json()
    assert (queue['id'], queue['overlap'], queue['blind']) == ('q1', 2, True)
    assert queue['closed'] is False
    assert queue['label_names'] == ['passed', 'c1']
    assert client.get('/api/queues/q1').json() == queue
    assert client.get('/api/queues').json() == [queue]
    assert added.status_code == 201
    first, second = added.json()
    assert (first['position'], first['context'], second['position']) == (1, 'Fix the port.', 2)

    held = client.post('/api/queues/q1/next', json={'reviewer': 'ana'})

    assert held.status_code == 200
    view = held.json()
    assert view['item'] == first
    assert view['assignment']['status'] == 'reserved'
    assert view['labels'] == []
    item = f'/api/queues/q1/items/{first["id"]}'
    assert client.get(item, params={'reviewer': 'ana'}).json() == view

    done = client.post(
        f'{item}/submit',
        json={'reviewer': 'ana', 'labels': [PASSED, {'name': 'c1', 'value': True}]},
    )

    assert done.status_code == 201
    review = done.json()
    assert review['assignment']['status'] == 'submitted'
    assert [(label['author'], label['name']) for label in review['labels']] == [
        ('judge-a', 'passed'),
        ('ana', 'passed'),
        ('ana', 'c1'),
    ]
    group = review['assignment']['id']
    stored = client.get('/api/labels', params={'group_id': group, 'blind': 'true'}).json()
    assert stored['total'] == 2
    progress = client.get('/api/queues/q1/progress').json()
    assert progress == {
        'queue_id': 'q1', 'items': 2, 'complete': 0, 'in_progress': 1, 'reviews': 1,
        'reviews_needed': 4, 'reserved': 0, 'blocked': 0,
        'reviewers': [{'reviewer': 'ana', 'submitted': 1, 'reserved': 0, 'skipped': 0,
                       'released': 0, 'expired': 0}],
    }  # fmt: skip
    page = client.get('/api/queues/q1/items', params={'offset': 0, 'limit': 1}).json()
    assert page == {
        'items': [{'item': first, 'reviews': 1, 'needed': 2, 'reserved': 0}],
        'total': 2,
    }
    again = client.post(
        f'{item}/submit',
        json={'reviewer': 'ana', 'labels': [PASSED, {'name': 'c1', 'value': True}]},
    )
    assert (again.status_code, again.json()) == (201, review)
    assert client.get('/api/labels').json()['total'] == 3


def test_release_skip_and_none_left(client: TestClient) -> None:
    queue_id = _queue(client, items=1)
    view = client.post(f'/api/queues/{queue_id}/next', json={'reviewer': 'ana'}).json()
    item = f'/api/queues/{queue_id}/items/{view["item"]["id"]}'

    released = client.post(f'{item}/release', json={'reviewer': 'ana'})
    client.post(f'/api/queues/{queue_id}/next', json={'reviewer': 'ana'})
    skipped = client.post(f'{item}/skip', json={'reviewer': 'ana'})
    none_left = client.post(f'/api/queues/{queue_id}/next', json={'reviewer': 'ana'})

    assert (released.status_code, released.json()['status']) == (200, 'released')
    assert (skipped.status_code, skipped.json()['status']) == (200, 'skipped')
    assert (none_left.status_code, none_left.json()) == (200, None)


def test_close_and_reopen(client: TestClient) -> None:
    queue_id = _queue(client)
    held = client.post(f'/api/queues/{queue_id}/next', json={'reviewer': 'bo'}).json()

    closed = client.patch(f'/api/queues/{queue_id}', json={'closed': True})
    refused = client.post(f'/api/queues/{queue_id}/next', json={'reviewer': 'ana'})
    submitted = client.post(
        f'/api/queues/{queue_id}/items/{held["item"]["id"]}/submit',
        json={'reviewer': 'bo', 'labels': [PASSED]},
    )
    reopened = client.patch(f'/api/queues/{queue_id}', json={'closed': False})

    assert closed.json()['closed'] is True
    assert (refused.status_code, refused.json()['detail']) == (409, f'Queue {queue_id} is closed')
    assert submitted.status_code == 201
    assert reopened.json()['closed'] is False
    assert client.post(f'/api/queues/{queue_id}/next', json={'reviewer': 'ana'}).json()


def test_patch_changes_what_may_change(client: TestClient) -> None:
    queue_id = _queue(client, reviewers=['ana'], description='v1')

    patched = client.patch(
        f'/api/queues/{queue_id}',
        json={'reviewers': ['ana', 'bo'], 'instructions': 'Read the diff.', 'description': None,
              'reservation_seconds': 60},
    )  # fmt: skip
    dropped = client.patch(f'/api/queues/{queue_id}', json={'reviewers': ['bo']})
    frozen = [
        client.patch(f'/api/queues/{queue_id}', json=body)
        for body in ({'overlap': 2}, {'blind': False}, {'label_names': []}, {'closed': None})
    ]

    assert patched.status_code == 200
    queue = patched.json()
    assert (queue['reviewers'], queue['instructions'], queue['description']) == (
        ['ana', 'bo'],
        'Read the diff.',
        None,
    )
    assert queue['reservation_seconds'] == 60
    assert client.get(f'/api/queues/{queue_id}').json() == queue
    assert dropped.status_code == 409
    assert [response.status_code for response in frozen] == [422] * 4
    assert client.post(f'/api/queues/{queue_id}/next', json={'reviewer': 'bo'}).json()


def test_refusals(client: TestClient) -> None:
    queue_id = _queue(client, reviewers=['ana', 'bo'])
    view = client.post(f'/api/queues/{queue_id}/next', json={'reviewer': 'ana'}).json()
    item = f'/api/queues/{queue_id}/items/{view["item"]["id"]}'

    def submit(reviewer: str, *labels: dict) -> object:
        return client.post(f'{item}/submit', json={'reviewer': reviewer, 'labels': list(labels)})

    other_target = {'target': {'kind': 'trace', 'id': 't2'}, 'name': 'note', 'value': 'x'}
    cases = {
        'unknown queue': client.get('/api/queues/nope'),
        'unknown queue items': client.get('/api/queues/nope/items'),
        'unknown item': client.post(
            f'/api/queues/{queue_id}/items/nope/release', json={'reviewer': 'ana'}
        ),
        'not listed next': client.post(f'/api/queues/{queue_id}/next', json={'reviewer': 'cy'}),
        'not listed view': client.get(item, params={'reviewer': 'cy'}),
        'not listed submit': submit('cy', PASSED),
        'not listed release': client.post(f'{item}/release', json={'reviewer': 'cy'}),
        'not listed skip': client.post(f'{item}/skip', json={'reviewer': 'cy'}),
        "someone else's": submit('bo', PASSED),
        'no labels': submit('ana'),
        'missing name': submit('ana', {'name': 'c1', 'value': True}),
        'other target': submit('ana', PASSED, other_target),
        'label rule': submit('ana', {'name': 'passed', 'value': True}),
        'repeated id': submit(
            'ana', {**PASSED, 'id': 'a'}, {'name': 'note', 'value': 'x', 'id': 'a'}
        ),
        'other settings': client.post(
            '/api/queues', json={'id': queue_id, 'name': 'calibration', 'overlap': 2}
        ),
    }

    assert {name: response.status_code for name, response in cases.items()} == {
        'unknown queue': 404, 'unknown queue items': 404, 'unknown item': 404,
        'not listed next': 403, 'not listed view': 403, 'not listed submit': 403,
        'not listed release': 403, 'not listed skip': 403, "someone else's": 409,
        'no labels': 422, 'missing name': 422, 'other target': 422, 'label rule': 422,
        'repeated id': 422, 'other settings': 409,
    }  # fmt: skip
    locs = {name: cases[name].json()['detail'][0]['loc'] for name in
            ('no labels', 'missing name', 'other target', 'label rule', 'repeated id')}  # fmt: skip
    assert locs == {
        'no labels': ['body', 'labels'],
        'missing name': ['body', 'labels'],
        'other target': ['body', 'labels', 1, 'target'],
        'label rule': ['body', 'labels', 0],
        'repeated id': ['body', 'labels', 1, 'id'],
    }
    assert 'need a comment' in cases['label rule'].json()['detail'][0]['msg']
    assert client.get(item, params={'reviewer': 'ana'}).json()['assignment']['status'] == (
        'reserved'
    )
    assert client.get('/api/labels').json()['total'] == 0


@pytest.mark.parametrize(
    ('path', 'body'),
    [
        ('/api/queues', {'name': 'q', 'overlap': 0}),
        ('/api/queues', {'name': 'q', 'reviewers': ['ana'], 'overlap': 2}),
        ('/api/queues', {'name': 'q', 'extra': 1}),
        ('/api/queues/{q}', {'closed': 'yes'}),
        ('/api/queues/{q}/items', [{'target': {'kind': 'job', 'id': 'j1'}}]),
        ('/api/queues/{q}/items', [{'target': {'kind': 'trace', 'id': 't9', 'step': 1}}]),
        ('/api/queues/{q}/next', {'reviewer': 'ana '}),
        ('/api/queues/{q}/next', {}),
        ('/api/queues/{q}/items/i/submit', {'reviewer': 'ana', 'labels': [{'name': 'passed'}]}),
    ],
)
def test_bad_bodies_get_a_422(client: TestClient, path: str, body: object) -> None:
    queue_id = _queue(client)
    url = path.format(q=queue_id)

    send = client.patch if url == f'/api/queues/{queue_id}' else client.post
    assert send(url, json=body).status_code == 422


@pytest.mark.parametrize(
    'url',
    [
        '/api/queues?limit=1',
        '/api/queues/{q}/items?stepp=1',
        '/api/queues/{q}/items?limit=0',
        '/api/queues/{q}/items?offset=1&offset=2',
        '/api/queues/{q}/progress?reviewer=ana',
        '/api/queues/{q}/items/i',
        '/api/queues/{q}/items/i?reviewer=ana&reviewer=bo',
        '/api/queues/{q}/items/i?reviewer=%20ana',
    ],
)
def test_bad_queries_get_a_422(client: TestClient, url: str) -> None:
    queue_id = _queue(client)

    assert client.get(url.format(q=queue_id)).status_code == 422


def test_posts_take_no_query_parameters(client: TestClient) -> None:
    queue_id = _queue(client)

    response = client.post(f'/api/queues/{queue_id}/next?reviewer=ana', json={'reviewer': 'ana'})

    assert response.status_code == 422
    assert client.get(f'/api/queues/{queue_id}/progress').json()['reserved'] == 0


def test_writes_need_token_and_origin(client: TestClient) -> None:
    no_token = client.post('/api/queues', json={'name': 'q'}, headers={'X-DeepGym-Token': 'x'})
    foreign = client.post(
        '/api/queues', json={'name': 'q'}, headers={'Origin': 'https://e.example'}
    )

    assert (no_token.status_code, foreign.status_code) == (403, 403)
    assert client.get('/api/queues').json() == []


def test_only_the_stores_lookups_are_404s(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: object) -> None:
        raise KeyError('a bug, not an unknown queue')

    monkeypatch.setattr(QueueStore, 'progress', broken)

    with pytest.raises(KeyError):
        client.get(f'/api/queues/{_queue(client)}/progress')
