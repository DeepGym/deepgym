"""Tests for the DeepGym FastAPI application."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from deepgym.api.app import app
from deepgym.api.deps import get_deepgym
from deepgym.api.routes import _environments
from deepgym.core import DeepGym

# A standalone verifier that works in subprocess (uses sys.exit, not os._exit).
STANDALONE_PASSING_VERIFIER = (
    'import sys, json\n'
    'if __name__ == "__main__":\n'
    '    print(json.dumps({"schema_version":"1.0","score":1.0,"passed":True,'
    '"details":None,"truncated":False}))\n'
    '    sys.exit(0)\n'
)


@pytest.fixture(autouse=True)
def _clear_env_store():
    """Clear the in-memory environment store before each test."""
    _environments.clear()
    yield
    _environments.clear()


@pytest.fixture(autouse=True)
def _clear_deepgym_cache():
    """Clear the lru_cache on get_deepgym so each test gets a fresh override."""
    get_deepgym.cache_clear()
    yield
    get_deepgym.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    """Return a TestClient wired to local mode with no API key required."""

    def _local_dg() -> DeepGym:
        return DeepGym(mode='local')

    app.dependency_overrides[get_deepgym] = _local_dg
    with patch.dict(os.environ, {'DEEPGYM_NO_AUTH': 'true'}, clear=False):
        # Opt out of auth for test convenience.
        os.environ.pop('DEEPGYM_API_KEY', None)
        yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    """Test the /health endpoint."""

    def test_health_returns_200_and_version(self, client: TestClient) -> None:
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.json()
        assert data['status'] == 'ok'
        assert 'version' in data


# ---------------------------------------------------------------------------
# POST /v1/run
# ---------------------------------------------------------------------------


class TestRunEndpoint:
    """Test POST /v1/run with a valid environment."""

    def test_run_with_valid_environment(self, client: TestClient) -> None:
        payload = {
            'environment': {
                'task': 'Return 1.0',
                'verifier_code': STANDALONE_PASSING_VERIFIER,
            },
            'model_output': '# empty\n',
        }
        resp = client.post('/v1/run', json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data['score'] == 1.0
        assert data['passed'] is True


# ---------------------------------------------------------------------------
# Environments CRUD
# ---------------------------------------------------------------------------


class TestEnvironmentsEndpoints:
    """Test environment creation, listing, and retrieval."""

    def test_create_environment(self, client: TestClient) -> None:
        payload = {
            'task': 'Sort a list',
            'verifier_code': 'return True\n',
        }
        resp = client.post('/v1/environments', json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data['created'] is True
        assert 'id' in data

    def test_list_environments(self, client: TestClient) -> None:
        # Create one first.
        client.post(
            '/v1/environments',
            json={
                'task': 'Task A',
                'verifier_code': 'return 1.0\n',
            },
        )
        resp = client.get('/v1/environments')
        assert resp.status_code == 200
        envs = resp.json()
        assert len(envs) == 1
        assert envs[0]['task'] == 'Task A'

    def test_get_environment_not_found(self, client: TestClient) -> None:
        resp = client.get('/v1/environments/nonexistent')
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    """Test X-API-Key authentication middleware."""

    def _make_client(self) -> TestClient:
        """Build a TestClient with local DG override (no fixture to avoid env leakage)."""

        def _local_dg() -> DeepGym:
            return DeepGym(mode='local')

        app.dependency_overrides[get_deepgym] = _local_dg
        return TestClient(app)

    def test_rejected_with_wrong_key(self) -> None:
        env_patch = {'DEEPGYM_API_KEY': 'secret123'}
        with patch.dict(os.environ, env_patch, clear=False):
            os.environ.pop('DEEPGYM_NO_AUTH', None)
            c = self._make_client()
            resp = c.post(
                '/v1/run',
                json={
                    'environment': {
                        'task': 't',
                        'verifier_code': STANDALONE_PASSING_VERIFIER,
                    },
                    'model_output': '#\n',
                },
                headers={'X-API-Key': 'wrong'},
            )
            assert resp.status_code == 401
        app.dependency_overrides.clear()

    def test_passes_with_correct_key(self) -> None:
        env_patch = {'DEEPGYM_API_KEY': 'secret123'}
        with patch.dict(os.environ, env_patch, clear=False):
            os.environ.pop('DEEPGYM_NO_AUTH', None)
            c = self._make_client()
            resp = c.post(
                '/v1/run',
                json={
                    'environment': {
                        'task': 't',
                        'verifier_code': STANDALONE_PASSING_VERIFIER,
                    },
                    'model_output': '#\n',
                },
                headers={'X-API-Key': 'secret123'},
            )
            assert resp.status_code == 200
        app.dependency_overrides.clear()

    def test_passes_when_no_auth_mode(self, client: TestClient) -> None:
        # client fixture sets DEEPGYM_NO_AUTH=true and unsets DEEPGYM_API_KEY.
        resp = client.post(
            '/v1/run',
            json={
                'environment': {
                    'task': 't',
                    'verifier_code': STANDALONE_PASSING_VERIFIER,
                },
                'model_output': '#\n',
            },
        )
        assert resp.status_code == 200

    def test_rejected_when_no_key_and_no_opt_out(self) -> None:
        """Without DEEPGYM_API_KEY or DEEPGYM_NO_AUTH, requests are rejected."""

        def _local_dg() -> DeepGym:
            return DeepGym(mode='local')

        app.dependency_overrides[get_deepgym] = _local_dg
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('DEEPGYM_API_KEY', None)
            os.environ.pop('DEEPGYM_NO_AUTH', None)
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.post(
                '/v1/run',
                json={
                    'environment': {
                        'task': 't',
                        'verifier_code': STANDALONE_PASSING_VERIFIER,
                    },
                    'model_output': '#\n',
                },
            )
            assert resp.status_code == 401
        app.dependency_overrides.clear()
