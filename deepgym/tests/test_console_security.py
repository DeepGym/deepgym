"""Tests that other websites can't drive the console: token, same-origin writes, JSON, hosts."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deepgym.console import app as console_app
from deepgym.console.app import create_app
from deepgym.console.security import TOKEN_HEADER
from tests.console_client import TOKEN, make_client

RUN = {'environments': ['fizzbuzz'], 'mode': 'local'}


@pytest.fixture()
def built_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in static export with an index and a sub-page."""
    static = tmp_path / 'static'
    (static / 'runs').mkdir(parents=True)
    (static / 'index.html').write_text(
        '<html><head><title>c</title></head><body>console</body></html>'
    )
    (static / 'runs' / 'index.html').write_text('<html><head></head><body>runs</body></html>')
    (static / 'app.js').write_text('console.log(1)')
    monkeypatch.setattr(console_app, 'STATIC_DIR', static)
    return static


@pytest.mark.parametrize('headers', [{}, {TOKEN_HEADER: 'wrong'}, {TOKEN_HEADER: ''}])
def test_api_needs_the_launch_token(tmp_path: Path, headers: dict) -> None:
    client = TestClient(create_app(tmp_path / 'c.db', tmp_path / 'qa', token=TOKEN))
    assert client.get('/api/qa', headers=headers).status_code == 403
    assert client.post('/api/runs', json=RUN, headers=headers).status_code == 403


def test_cross_origin_writes_are_refused(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post('/api/runs', json=RUN, headers={'Origin': 'https://evil.example'})
    assert response.status_code == 403
    assert (
        client.delete(
            '/api/sandboxes/docker/deepgym-ab12', headers={'Origin': 'https://evil.example'}
        ).status_code
        == 403
    )
    assert client.get('/api/runs').json() == []


def test_same_origin_writes_pass(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post('/api/runs', json=RUN, headers={'Origin': 'http://testserver'})
    assert response.status_code == 202


@pytest.mark.parametrize('content_type', ['text/plain', 'application/x-www-form-urlencoded', ''])
def test_writes_must_be_json(tmp_path: Path, content_type: str) -> None:
    response = make_client(tmp_path).post(
        '/api/runs',
        content=b'{"environments": ["fizzbuzz"]}',
        headers={'Content-Type': content_type},
    )
    assert response.status_code == 415


def test_html_carries_the_token(tmp_path: Path, built_ui: Path) -> None:
    client = TestClient(create_app(tmp_path / 'c.db', tmp_path / 'qa', token=TOKEN))

    index = client.get('/')
    page = client.get('/runs/')

    assert f'<meta name="deepgym-token" content="{TOKEN}"/></head>' in index.text
    assert 'runs' in page.text and TOKEN in page.text
    assert index.headers['cache-control'] == 'no-store'
    assert client.get('/app.js').text == 'console.log(1)'


def test_each_launch_gets_a_new_token(tmp_path: Path, built_ui: Path) -> None:
    first = TestClient(create_app(tmp_path / 'a.db', tmp_path / 'qa')).get('/').text
    second = TestClient(create_app(tmp_path / 'b.db', tmp_path / 'qa')).get('/').text
    assert first != second


def test_missing_ui_explains_how_to_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(console_app, 'STATIC_DIR', tmp_path / 'static')
    response = make_client(tmp_path).get('/')
    assert response.status_code == 503
    assert 'bun run build' in response.text


def test_rejects_foreign_host_headers(tmp_path: Path) -> None:
    app = create_app(tmp_path / 'c.db', tmp_path / 'qa', allowed_hosts=['127.0.0.1'], token=TOKEN)
    headers = {TOKEN_HEADER: TOKEN}
    assert (
        TestClient(app, base_url='http://evil.example').get('/api/qa', headers=headers).status_code
        == 400
    )
    assert (
        TestClient(app, base_url='http://127.0.0.1').get('/api/qa', headers=headers).status_code
        == 200
    )
