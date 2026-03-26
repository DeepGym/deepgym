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

    def test_capabilities_returns_expected_flags(self, client: TestClient) -> None:
        resp = client.get('/v1/capabilities')
        assert resp.status_code == 200
        data = resp.json()
        assert data['named_environment_run'] is True
        assert data['verifier_audit'] is True
        assert data['benchmark_audit'] is True


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

    def test_run_registered_environment(self, client: TestClient) -> None:
        create = client.post(
            '/v1/environments',
            json={
                'task': 'Return 1.0',
                'verifier_code': STANDALONE_PASSING_VERIFIER,
            },
        )
        env_id = create.json()['id']

        resp = client.post(f'/v1/environments/{env_id}/run', json={'model_output': '# empty\n'})
        assert resp.status_code == 200
        data = resp.json()
        assert data['score'] == 1.0
        assert data['passed'] is True


class TestAuditEndpoints:
    """Test inline and registered verifier audit endpoints."""

    def test_audit_inline_environment(self, client: TestClient) -> None:
        payload = {
            'environment': {
                'task': 'Do anything',
                'verifier_code': STANDALONE_PASSING_VERIFIER,
            },
            'verifier_id': 'weak-api',
            'strategies': ['empty'],
        }
        resp = client.post('/v1/audit/verifier', json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data['verifier_id'] == 'weak-api'
        assert data['exploitable'] is True
        assert data['risk_level'] in {'high', 'critical'}

    def test_audit_registered_environment(self, client: TestClient) -> None:
        create = client.post(
            '/v1/environments',
            json={
                'task': 'Return 1.0',
                'verifier_code': STANDALONE_PASSING_VERIFIER,
            },
        )
        env_id = create.json()['id']

        resp = client.post(f'/v1/environments/{env_id}/audit', json={'strategies': ['empty']})
        assert resp.status_code == 200
        data = resp.json()
        assert data['verifier_id'] == env_id
        assert data['exploitable'] is True


class TestBenchmarkAuditEndpoint:
    """Test benchmark split leakage auditing for registered environments."""

    def test_benchmark_audit_detects_public_private_leak(self, client: TestClient) -> None:
        env_ids: list[str] = []
        for task in ('Shared task', 'Shared task', 'Unique task'):
            resp = client.post(
                '/v1/environments',
                json={
                    'task': task,
                    'verifier_code': 'return 1.0\n',
                },
            )
            env_ids.append(resp.json()['id'])

        payload = {
            'environment_ids': env_ids,
            'benchmark': 'registered-demo',
            'split_overrides': {
                env_ids[0]: 'public_train',
                env_ids[1]: 'private_holdout',
                env_ids[2]: 'public_eval',
            },
        }
        resp = client.post('/v1/benchmarks/audit', json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data['benchmark'] == 'registered-demo'
        assert data['contamination_risk'] is True
        assert len(data['leaks']) >= 1


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
