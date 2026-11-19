"""Tests that other websites can't drive the console: token, same-origin writes, JSON, hosts."""

import re
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
    index_html = '<html><head><title>c</title></head><body>console</body></html>'
    (static / 'index.html').write_text(index_html)
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
    assert client.get('/api/runs').json() == {'runs': [], 'total': 0}


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


def test_serves_a_ui_dir_in_place_of_the_bundled_ui(tmp_path: Path, built_ui: Path) -> None:
    ui = tmp_path / 'my-ui'
    (ui / 'runs').mkdir(parents=True)
    (ui / 'index.html').write_text('<html><head></head><body>mine</body></html>')
    (ui / 'runs' / 'index.html').write_text('<html><head></head><body>my runs</body></html>')
    default = TestClient(create_app(tmp_path / 'a.db', tmp_path / 'qa', token=TOKEN))
    client = TestClient(create_app(tmp_path / 'b.db', tmp_path / 'qa', token=TOKEN, static_dir=ui))

    index = client.get('/')
    page = client.get('/runs/')

    assert 'console' in default.get('/').text
    assert 'mine' in index.text and 'console' not in index.text
    assert f'<meta name="deepgym-token" content="{TOKEN}"/></head>' in index.text
    assert 'my runs' in page.text and TOKEN in page.text
    assert index.headers['cache-control'] == 'no-store'
    assert client.get('/api/qa').status_code == 403
    assert client.get('/api/qa', headers={TOKEN_HEADER: TOKEN}).status_code == 200


def test_ui_dir_404_page_and_uppercase_html_carry_the_token(tmp_path: Path) -> None:
    ui = tmp_path / 'my-ui'
    ui.mkdir()
    (ui / 'index.html').write_text('<html><head></head></html>')
    (ui / '404.html').write_text('<html><head></head><body>not here</body></html>')
    (ui / 'PAGE.HTM').write_text('<HTML><HEAD></HEAD><BODY>loud</BODY></HTML>')
    (tmp_path / 'secret.txt').write_text('secret')
    (ui / 'secret.txt').symlink_to(tmp_path / 'secret.txt')
    client = TestClient(create_app(tmp_path / 'c.db', tmp_path / 'qa', token=TOKEN, static_dir=ui))
    meta = f'<meta name="deepgym-token" content="{TOKEN}"/>'

    missing = client.get('/runs/nope/')
    loud = client.get('/PAGE.HTM')
    linked_out = client.get('/secret.txt')

    assert missing.status_code == 404
    assert 'not here' in missing.text and f'{meta}</head>' in missing.text
    assert missing.headers['cache-control'] == 'no-store'
    assert f'{meta}</HEAD>' in loud.text and loud.headers['cache-control'] == 'no-store'
    assert linked_out.status_code == 404 and 'secret' not in linked_out.text


@pytest.mark.parametrize(
    ('setup', 'problem'),
    [
        ('missing', 'does not exist'),
        ('empty', 'has no index.html'),
        ('linked_out', 'has an index.html that links outside it'),
    ],
)
def test_ui_dir_needs_its_own_index_html(tmp_path: Path, setup: str, problem: str) -> None:
    ui = tmp_path / 'my-ui'
    if setup != 'missing':
        ui.mkdir()
    if setup == 'linked_out':
        (tmp_path / 'index.html').write_text('<html><head></head></html>')
        (ui / 'index.html').symlink_to(tmp_path / 'index.html')
    with pytest.raises(FileNotFoundError, match=re.escape(f'UI folder {ui} {problem}')):
        create_app(tmp_path / 'c.db', tmp_path / 'qa', static_dir=ui)


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
