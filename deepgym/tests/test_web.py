"""Tests for deepgym.web -- web debugging UI API endpoints."""

import pytest
from fastapi.testclient import TestClient

from deepgym.web import RunRequest, RunResponse, create_web_app


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a TestClient for the web debugging app."""
    monkeypatch.setenv('DEEPGYM_ALLOW_LOCAL_EXEC', 'true')
    app = create_web_app()
    return TestClient(app)


class TestWebUI:
    """Verify the web debugging UI serves HTML."""

    def test_index_returns_html(self, client: TestClient) -> None:
        response = client.get('/')
        assert response.status_code == 200
        assert 'text/html' in response.headers['content-type']
        assert 'DeepGym' in response.text


class TestWebAPIEnvironments:
    """Verify the /api/environments endpoints."""

    def test_list_environments(self, client: TestClient) -> None:
        response = client.get('/api/environments')
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 24

    def test_get_environment_details(self, client: TestClient) -> None:
        response = client.get('/api/environments/coin_change')
        assert response.status_code == 200
        data = response.json()
        assert 'task' in data
        assert 'difficulty' in data
        assert len(data['task']) > 10

    def test_get_nonexistent_environment_returns_404(self, client: TestClient) -> None:
        response = client.get('/api/environments/nonexistent_env_xyz')
        assert response.status_code == 404


class TestWebAPIRun:
    """Verify the /api/run endpoint."""

    def test_run_with_valid_env_and_code(self, client: TestClient) -> None:
        response = client.post(
            '/api/run',
            json={
                'environment': 'reverse_string',
                'code': ('def reverse_string(s: str) -> str:\n    return s[::-1]\n'),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert 'score' in data
        assert 'passed' in data
        assert isinstance(data['score'], float)
        assert data['score'] >= 0.0

    def test_run_with_bad_code_still_returns_200(self, client: TestClient) -> None:
        response = client.post(
            '/api/run',
            json={
                'environment': 'reverse_string',
                'code': '# bad solution\npass\n',
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data['score'] < 1.0


class TestWebAPIHistory:
    """Verify the /api/history endpoint."""

    def test_history_returns_list(self, client: TestClient) -> None:
        response = client.get('/api/history')
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_history_grows_after_run(self, client: TestClient) -> None:
        # Get initial history count.
        initial = len(client.get('/api/history').json())

        # Run something.
        client.post(
            '/api/run',
            json={'environment': 'reverse_string', 'code': 'pass\n'},
        )

        after = len(client.get('/api/history').json())
        assert after == initial + 1


class TestWebModels:
    """Verify the Pydantic models used by the web module."""

    def test_run_request_validation(self) -> None:
        req = RunRequest(environment='test', code='print(1)')
        assert req.environment == 'test'
        assert req.code == 'print(1)'

    def test_run_response_with_defaults(self) -> None:
        resp = RunResponse(
            score=0.5,
            passed=False,
            output='out',
            execution_time_ms=100.0,
        )
        assert resp.reward_components is None
        assert resp.error is None
