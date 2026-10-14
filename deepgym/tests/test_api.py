"""Tests for the DeepGym FastAPI application."""

import asyncio
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from deepgym import __version__
from deepgym.api import app as app_module
from deepgym.api import deps, routes
from deepgym.api.app import app
from deepgym.api.deps import ServerDeepGym, get_deepgym
from deepgym.api.routes import _environments, _sessions
from deepgym.api.schemas import APIEnvironment, RunRequest
from deepgym.async_core import AsyncDeepGym
from deepgym.core import DeepGym
from deepgym.exceptions import SandboxError
from deepgym.models import Environment, Job, JobStatus, RunResult
from deepgym.session import Session

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

# A standalone verifier that works in subprocess (uses sys.exit, not os._exit).
STANDALONE_PASSING_VERIFIER = (
    'import sys, json\n'
    'if __name__ == "__main__":\n'
    '    print(json.dumps({"schema_version":"1.0","score":1.0,"passed":True,'
    '"details":None,"truncated":False}))\n'
    '    sys.exit(0)\n'
)
PASSING_ENV = {'task': 'Return 1.0', 'verifier_code': STANDALONE_PASSING_VERIFIER}
RUN_BODY = {'environment': PASSING_ENV, 'model_output': '#\n'}
FAKE_RESULT = RunResult(
    score=1.0,
    passed=True,
    output='',
    stderr='',
    exit_code=0,
    execution_time_ms=0.0,
    sandbox_id='fake',
)


@pytest.fixture(autouse=True)
def _clear_env_store():
    """Clear the in-memory environment store before each test."""
    _environments.clear()
    yield
    _environments.clear()


@pytest.fixture(autouse=True)
def _close_sessions():
    """Close any sessions a test left open or unclosed."""
    yield
    for session in [*_sessions.values(), *routes._closing.values()]:
        session.close()
    _sessions.clear()
    routes._closing.clear()
    routes._close_failures.clear()
    routes._leaked_sandboxes.clear()
    routes._last_used.clear()
    routes._session_slots = threading.BoundedSemaphore(routes._MAX_SESSIONS)


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Forget the jobs a test submitted."""
    yield
    routes._jobs.clear()
    routes._queued.clear()
    routes._running.clear()
    routes._finished.clear()
    routes._job_bytes = routes._finished_bytes = 0


@pytest.fixture(autouse=True)
def _clear_deepgym_cache():
    """Clear the lru_cache on get_deepgym so each test gets a fresh override."""
    get_deepgym.cache_clear()
    yield
    get_deepgym.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    """Return a TestClient wired to local mode with no API key required."""

    def _local_dg() -> ServerDeepGym:
        return ServerDeepGym(mode='local')

    app.dependency_overrides[get_deepgym] = _local_dg
    with patch.dict(os.environ, {'DEEPGYM_NO_AUTH': 'true'}, clear=False):
        # Opt out of auth for test convenience.
        os.environ.pop('DEEPGYM_API_KEY', None)
        yield TestClient(app)
    app.dependency_overrides.clear()


def _app_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient for the real app and client in local mode; enter it to run the lifespan."""
    monkeypatch.setenv('DEEPGYM_NO_AUTH', 'true')
    monkeypatch.setenv('DEEPGYM_ALLOW_LOCAL_EXEC', 'true')
    monkeypatch.delenv('DEEPGYM_API_KEY', raising=False)
    monkeypatch.delenv('DAYTONA_API_KEY', raising=False)
    monkeypatch.setattr(routes, 'SESSION_SWEEP_INTERVAL_S', 0.05, raising=False)
    return TestClient(app)


@pytest.fixture()
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A running app: its lifespan, session sweeper and job tasks live until the test ends."""
    with _app_client(monkeypatch) as app_client:
        yield app_client


def _eventually(check: Callable[[], object], timeout: float = 10) -> bool:
    """Poll *check* until it is truthy or *timeout* seconds pass; return whether it held."""
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def _finished_job(app_client: TestClient, job_id: str) -> dict:
    """Poll a job until it reaches a terminal status and return it."""
    job: dict = {}

    def done() -> bool:
        job.update(app_client.get(f'/v1/jobs/{job_id}').json())
        return job['status'] in ('completed', 'failed', 'cancelled')

    assert _eventually(done, timeout=30)
    return job


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

    def test_async_client_sends_a_file_backed_verifier_inline(
        self,
        client: TestClient,
        sorting_env: Environment,
        sorting_solution: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_client = httpx.AsyncClient
        transport = httpx.ASGITransport(app=app)
        monkeypatch.setattr(httpx, 'AsyncClient', lambda: real_client(transport=transport))

        remote = AsyncDeepGym(api_url='http://testserver')
        result = asyncio.run(remote.run(sorting_env, sorting_solution))

        assert (result.score, result.passed) == (1.0, True)


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

    def test_audit_rejects_an_unknown_strategy(self, client: TestClient) -> None:
        payload = {
            'environment': {'task': 'Do anything', 'verifier_code': STANDALONE_PASSING_VERIFIER},
            'strategies': ['overlfow'],
        }
        assert client.post('/v1/audit/verifier', json=payload).status_code == 422


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


# ---------------------------------------------------------------------------
# Live sessions
# ---------------------------------------------------------------------------

FILE_CHECK_ENV = {
    'type': 'multi-turn',
    'task': 'Write hello to out.txt',
    'allowed_actions': ['bash', 'done'],
    'final_verifier_code': (
        'import json, pathlib\n'
        'ok = pathlib.Path("out.txt").read_text().strip() == "hello"\n'
        'print(json.dumps({"score": float(ok), "passed": ok}))\n'
    ),
}


def _open_session(app_client: TestClient) -> str:
    """Open a FILE_CHECK_ENV session and return its id."""
    created = app_client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV})
    assert created.status_code == 201
    return created.json()['session_id']


def _fail_first_close(monkeypatch: pytest.MonkeyPatch, session: Session) -> None:
    """Make *session*'s first close fail like a provider outage; later ones really close."""
    real_close = session.close
    calls: list[int] = []

    def close() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise SandboxError('Could not delete sandbox: provider unavailable')
        real_close()

    monkeypatch.setattr(session, 'close', close)


class TestSessions:
    """Test the /v1/sessions endpoints."""

    def test_full_episode(self, client: TestClient) -> None:
        created = client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV})
        assert created.status_code == 201
        session_id = created.json()['session_id']
        assert created.json()['observation']['step'] == 0

        step = client.post(
            f'/v1/sessions/{session_id}/step',
            json={'action_type': 'bash', 'content': 'echo hello > out.txt'},
        )
        assert step.status_code == 200
        assert step.json()['done'] is False

        score = client.post(f'/v1/sessions/{session_id}/score')
        assert score.status_code == 200
        assert score.json()['passed'] is True

        assert client.delete(f'/v1/sessions/{session_id}').status_code == 204
        assert client.post(f'/v1/sessions/{session_id}/score').status_code == 404

    def test_disallowed_action_is_conflict(self, client: TestClient) -> None:
        session_id = client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).json()[
            'session_id'
        ]
        resp = client.post(
            f'/v1/sessions/{session_id}/step', json={'action_type': 'code', 'content': 'print(1)'}
        )
        assert resp.status_code == 409

    def test_invalid_action_is_rejected(self, client: TestClient) -> None:
        session_id = client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).json()[
            'session_id'
        ]
        resp = client.post(f'/v1/sessions/{session_id}/step', json={'action_type': 'click'})
        assert resp.status_code == 422

    def test_single_turn_env_rejected(self, client: TestClient) -> None:
        resp = client.post('/v1/sessions', json={'environment': 'coin_change'})
        assert resp.status_code == 400

    def test_unknown_env_is_404(self, client: TestClient) -> None:
        resp = client.post('/v1/sessions', json={'environment': 'no_such_env_xyz'})
        assert resp.status_code == 404

    @pytest.mark.parametrize('field', ['image', 'snapshot'])
    def test_caller_chosen_runtime_is_refused(self, client: TestClient, field: str) -> None:
        env = {**FILE_CHECK_ENV, field: 'attacker-runtime'}
        resp = client.post('/v1/sessions', json={'environment': env})
        assert resp.status_code == 422

    def test_run_requests_never_carry_an_image_or_snapshot(self) -> None:
        """The run, batch, job and environment endpoints drop both fields."""
        spec = {
            'task': 't',
            'verifier_code': 'return 1.0',
            'image': 'attacker/image:latest',
            'snapshot': 'attacker-snapshot',
        }
        env = APIEnvironment.model_validate(spec).to_environment()
        assert (env.image, env.snapshot) == (None, None)

    def test_session_cap_counts_booting_and_live_sessions(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(routes, '_session_slots', threading.BoundedSemaphore(1))
        first = client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV})
        assert first.status_code == 201
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 429

        session_id = first.json()['session_id']
        assert client.delete(f'/v1/sessions/{session_id}').status_code == 204
        assert client.delete(f'/v1/sessions/{session_id}').status_code == 404
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 201

    def test_failed_start_frees_its_slot(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(routes, '_session_slots', threading.BoundedSemaphore(1))
        assert client.post('/v1/sessions', json={'environment': 'signup_form'}).status_code == 400
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 201

    def test_inline_env_without_type_is_multi_turn(self, client: TestClient) -> None:
        spec = {key: value for key, value in FILE_CHECK_ENV.items() if key != 'type'}
        created = client.post('/v1/sessions', json={'environment': spec})
        assert created.status_code == 201
        step = client.post(
            f'/v1/sessions/{created.json()["session_id"]}/step',
            json={'action_type': 'bash', 'content': 'echo hello > out.txt'},
        )
        assert step.status_code == 200

    def test_idle_sessions_are_evicted(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(routes, '_session_slots', threading.BoundedSemaphore(1))
        stale = _open_session(client)
        routes._last_used[stale] -= routes._SESSION_IDLE_TTL_S + 1

        routes.sweep_sessions()

        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 201
        assert client.post(f'/v1/sessions/{stale}/score').status_code == 404

    def test_slot_stays_held_until_sandbox_closes(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(routes, '_session_slots', threading.BoundedSemaphore(1))
        stale = _open_session(client)
        session = routes._sessions[stale]
        closing, release = threading.Event(), threading.Event()
        real_close = session.close
        monkeypatch.setattr(
            session, 'close', lambda: (closing.set(), release.wait(5), real_close())
        )
        routes._last_used[stale] -= routes._SESSION_IDLE_TTL_S + 1
        sweep = threading.Thread(target=routes.sweep_sessions)
        sweep.start()
        assert closing.wait(5)

        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 429

        release.set()
        sweep.join(5)
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 201

    def test_idle_sessions_close_without_new_traffic(
        self, served: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S1: the sweeper closes idle sessions on its own; no request has to come in."""
        monkeypatch.setattr(routes, '_SESSION_IDLE_TTL_S', 0.2)
        session_id = _open_session(served)

        assert _eventually(lambda: session_id not in routes._sessions)
        assert served.post(f'/v1/sessions/{session_id}/score').status_code == 404

    def test_shutdown_closes_open_sessions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """S1: stopping the server closes every session, so no sandbox outlives it."""
        closed: list[str] = []
        with _app_client(monkeypatch) as app_client:
            session_id = _open_session(app_client)
            session = routes._sessions[session_id]
            real_close = session.close
            monkeypatch.setattr(session, 'close', lambda: (closed.append(session_id), real_close()))

        assert closed == [session_id]
        assert not routes._sessions and not routes._closing

    def test_failed_delete_keeps_the_slot_until_a_retry_succeeds(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S2/H7: a failed delete keeps the session's id and slot; DELETE can be retried."""
        monkeypatch.setattr(routes, '_session_slots', threading.BoundedSemaphore(1))
        session_id = _open_session(client)
        _fail_first_close(monkeypatch, routes._sessions[session_id])

        assert client.delete(f'/v1/sessions/{session_id}').status_code == 502
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 429

        assert client.delete(f'/v1/sessions/{session_id}').status_code == 204
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 201

    def test_sweep_retries_a_failed_close(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S1/S2: a close that fails during a sweep keeps its slot and is retried next sweep."""
        monkeypatch.setattr(routes, '_session_slots', threading.BoundedSemaphore(1))
        session_id = _open_session(client)
        _fail_first_close(monkeypatch, routes._sessions[session_id])
        routes._last_used[session_id] -= routes._SESSION_IDLE_TTL_S + 1

        routes.sweep_sessions()
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 429

        routes.sweep_sessions()
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 201

    def test_a_close_that_keeps_failing_is_given_up_and_frees_its_slot(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A delete that never succeeds can't hold its slot forever: after the last attempt the
        sandbox is reported as leaked and the slot is freed."""
        monkeypatch.setattr(routes, '_session_slots', threading.BoundedSemaphore(1))
        monkeypatch.setattr(routes, '_MAX_CLOSE_ATTEMPTS', 3)
        session_id = _open_session(client)
        session = routes._sessions[session_id]
        real_close = session.close

        def refuse() -> None:
            raise SandboxError('Could not delete sandbox: provider unavailable')

        monkeypatch.setattr(session, 'close', refuse)

        assert client.delete(f'/v1/sessions/{session_id}').status_code == 502
        routes.sweep_sessions()
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 429
        assert client.get('/health').json()['leaked_sandboxes'] == 0

        routes.sweep_sessions()  # the third failed attempt: given up

        assert list(routes._leaked_sandboxes) == [session.sandbox_id]
        assert client.get('/health').json()['leaked_sandboxes'] == 1
        assert client.delete(f'/v1/sessions/{session_id}').status_code == 404
        assert client.post('/v1/sessions', json={'environment': FILE_CHECK_ENV}).status_code == 201
        real_close()

    def test_bad_variant_is_404(self, client: TestClient) -> None:
        resp = client.post('/v1/sessions', json={'environment': 'todo_list', 'variant': 99})
        assert resp.status_code == 404

    def test_desktop_env_needs_daytona(self, client: TestClient) -> None:
        resp = client.post('/v1/sessions', json={'environment': 'signup_form'})
        assert resp.status_code == 400
        assert not _sessions


# ---------------------------------------------------------------------------
# Registered environments behave exactly like inline ones
# ---------------------------------------------------------------------------

# Every execution field set away from its default; the verifier needs its support file.
PARITY_ENV = {
    'task': 't',
    'verifier_code': 'from scoring import half\n\ndef verify(solution_path):\n    return half()\n',
    'verifier_mode': 'verify_function',
    'case_timeout': 7,
    'support_files': {'scoring.py': 'def half():\n    return 0.5\n'},
}
BAD_ENV = {**PASSING_ENV, 'support_files': {'../outside': 'x'}}


class TestEnvironmentParity:
    """S3, S4, S8: registration keeps every field, validates first, and lists usable ids."""

    def test_registered_env_keeps_every_field_and_scores_like_inline(
        self, client: TestClient
    ) -> None:
        inline = client.post('/v1/run', json={'environment': PARITY_ENV, 'model_output': ''})
        env_id = client.post('/v1/environments', json=PARITY_ENV).json()['id']

        stored = client.get(f'/v1/environments/{env_id}').json()
        registered = client.post(f'/v1/environments/{env_id}/run', json={'model_output': ''})

        assert {key: stored[key] for key in PARITY_ENV} == PARITY_ENV
        assert inline.json()['score'] == registered.json()['score'] == 0.5

    def test_listed_environments_carry_ids_that_run(self, client: TestClient) -> None:
        created = client.post('/v1/environments', json=PASSING_ENV).json()

        (listed,) = client.get('/v1/environments').json()

        assert listed['id'] == created['id']
        run = client.post(f'/v1/environments/{listed["id"]}/run', json={'model_output': '#'})
        assert run.json()['passed'] is True

    @pytest.mark.parametrize(
        ('path', 'body'),
        [
            ('/v1/run', {'environment': BAD_ENV, 'model_output': ''}),
            ('/v1/run/batch', {'environment': BAD_ENV, 'outputs': ['']}),
            ('/v1/jobs/run', {'environment': BAD_ENV, 'model_output': ''}),
            ('/v1/jobs/batch', {'environment': BAD_ENV, 'outputs': ['']}),
            ('/v1/environments', BAD_ENV),
        ],
    )
    def test_invalid_environment_is_rejected_before_any_work(
        self, served: TestClient, path: str, body: dict
    ) -> None:
        resp = served.post(path, json=body)

        assert resp.status_code == 422
        assert 'support file' in resp.text
        assert not routes._jobs and not _environments


class TestRequestLimits:
    """Request bodies are capped before they are read into memory."""

    def test_an_oversized_body_is_refused_with_413(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(app_module, 'MAX_BODY_BYTES', 1000)
        big = json.dumps({**RUN_BODY, 'model_output': 'x' * 2000}).encode()
        headers = {'content-type': 'application/json'}

        declared = client.post('/v1/run', content=big, headers=headers)
        chunked = client.post('/v1/run', content=iter([big[:600], big[600:]]), headers=headers)

        assert declared.status_code == 413
        assert chunked.status_code == 413
        assert client.post('/v1/run', json=RUN_BODY).status_code == 200


class TestVersion:
    """F7: every version the API reports is the package's."""

    def test_reported_version_is_the_package_version(self, client: TestClient) -> None:
        pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'

        assert client.get('/health').json()['version'] == __version__
        assert client.get('/v1/capabilities').json()['version'] == __version__
        assert app.openapi()['info']['version'] == __version__
        assert tomllib.loads(pyproject.read_text(encoding='utf-8'))['project']['version'] == (
            __version__
        )


# ---------------------------------------------------------------------------
# Async jobs
# ---------------------------------------------------------------------------


class TestJobs:
    """S5/H5, H6, H8: retention, a shared run limit, and cancellation that sticks."""

    def test_finished_jobs_free_capacity_and_the_oldest_expire(
        self, served: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(routes, '_MAX_JOBS', 1)
        monkeypatch.setattr(routes, '_MAX_FINISHED_JOBS', 2, raising=False)
        job_ids = []
        for _ in range(3):
            submitted = served.post('/v1/jobs/run', json=RUN_BODY)
            assert submitted.status_code == 202
            job_ids.append(submitted.json()['id'])
            assert _finished_job(served, job_ids[-1])['status'] == 'completed'

        assert served.get(f'/v1/jobs/{job_ids[0]}').status_code == 404
        assert all(served.get(f'/v1/jobs/{job_id}').status_code == 200 for job_id in job_ids[1:])

    def test_finished_jobs_are_kept_within_a_byte_budget(
        self, served: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(routes, '_MAX_FINISHED_BYTES', 1)  # room for the newest job only
        monkeypatch.setattr(DeepGym, 'run', lambda self, env, model_output, **_: FAKE_RESULT)
        first = served.post('/v1/jobs/run', json=RUN_BODY).json()['id']
        _finished_job(served, first)
        second = served.post('/v1/jobs/run', json=RUN_BODY).json()['id']
        _finished_job(served, second)

        assert _eventually(lambda: served.get(f'/v1/jobs/{first}').status_code == 404)
        assert served.get(f'/v1/jobs/{second}').json()['status'] == 'completed'

    def test_queued_job_payloads_have_a_byte_budget(
        self, served: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = threading.Event()

        def blocking_run(
            self: DeepGym, env: Environment, model_output: str, **_: object
        ) -> RunResult:
            release.wait(10)
            return FAKE_RESULT

        monkeypatch.setattr(DeepGym, 'run', blocking_run)
        first = served.post('/v1/jobs/run', json=RUN_BODY).json()['id']
        monkeypatch.setattr(routes, '_MAX_JOB_BYTES', routes._job_bytes * 3 // 2)

        assert served.post('/v1/jobs/run', json=RUN_BODY).status_code == 429

        release.set()
        _finished_job(served, first)
        assert _eventually(lambda: routes._job_bytes == 0)
        assert served.post('/v1/jobs/run', json=RUN_BODY).status_code == 202

    def test_a_cancelled_queued_job_lets_go_of_its_request(
        self, served: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With every job thread busy, a cancelled job leaves nothing behind in the executor
        while its bytes stop counting."""
        release = threading.Event()
        calls: list[str] = []

        def blocking_run(
            self: DeepGym, env: Environment, model_output: str, **_: object
        ) -> RunResult:
            calls.append(model_output)
            release.wait(10)
            return FAKE_RESULT

        monkeypatch.setattr(DeepGym, 'run', blocking_run)
        monkeypatch.setattr(routes, '_JOB_THREADS', 1)
        busy = served.post('/v1/jobs/run', json=RUN_BODY).json()['id']
        assert _eventually(lambda: busy in routes._running)
        one_request = routes._job_bytes
        waiting = served.post('/v1/jobs/run', json=RUN_BODY).json()['id']
        assert served.get(f'/v1/jobs/{waiting}').json()['status'] == 'queued'

        assert served.post(f'/v1/jobs/{waiting}/cancel').json()['status'] == 'cancelled'

        assert routes._job_bytes == one_request and waiting not in routes._queued
        assert routes._job_executor._work_queue.empty()  # nothing waits inside the executor
        release.set()
        assert _finished_job(served, busy)['status'] == 'completed'
        assert served.get(f'/v1/jobs/{waiting}').json()['status'] == 'cancelled'
        assert len(calls) == 1

    def test_an_unserializable_result_fails_its_job_without_stalling_the_queue(
        self, served: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unsendable = FAKE_RESULT.model_copy(update={'metrics': {'text': '\ud800'}})
        results = iter([unsendable, FAKE_RESULT])
        monkeypatch.setattr(DeepGym, 'run', lambda self, env, model_output, **_: next(results))
        monkeypatch.setattr(routes, '_JOB_THREADS', 1)  # the second job waits for the first

        first = served.post('/v1/jobs/run', json=RUN_BODY).json()['id']
        second = served.post('/v1/jobs/run', json=RUN_BODY).json()['id']

        failed = _finished_job(served, first)
        assert failed['status'] == 'failed' and 'could not be serialized' in failed['error']
        assert _finished_job(served, second)['status'] == 'completed'

    def test_finished_job_budget_counts_bytes_not_characters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wide = FAKE_RESULT.model_copy(update={'output': 'é' * 1000})  # 2 bytes a character
        jobs = [Job(status=JobStatus.completed, result=wide) for _ in range(2)]
        for job in jobs:
            routes._jobs[job.id] = job
        # Both jobs fit in this many characters, but not in this many bytes.
        monkeypatch.setattr(routes, '_MAX_FINISHED_BYTES', 2 * len(jobs[0].model_dump_json()))

        for job in jobs:
            routes._keep_finished(job.id)

        assert jobs[0].id not in routes._jobs and jobs[1].id in routes._jobs

    def test_concurrent_requests_share_one_run_limit(
        self, served: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(deps, '_run_slots', threading.BoundedSemaphore(2), raising=False)
        lock = threading.Lock()
        running = peak = 0

        def counting_run(
            self: DeepGym, env: Environment, model_output: str, **_: object
        ) -> RunResult:
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.2)
            with lock:
                running -= 1
            return FAKE_RESULT

        monkeypatch.setattr(DeepGym, 'run', counting_run)
        body = {'environment': PASSING_ENV, 'outputs': ['#'] * 4, 'max_parallel': 4}

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: served.post('/v1/run/batch', json=body), range(2)))

        assert [resp.status_code for resp in responses] == [200, 200]
        assert peak == 2

    async def test_cancelling_a_queued_job_sticks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('DAYTONA_API_KEY', raising=False)
        ran: list[str] = []

        def recording_run(
            self: DeepGym, env: Environment, model_output: str, **_: object
        ) -> RunResult:
            ran.append(model_output)
            return FAKE_RESULT

        monkeypatch.setattr(DeepGym, 'run', recording_run)

        body = RunRequest.model_validate(RUN_BODY)
        job = await routes.submit_run_job(body, get_deepgym(), len(body.model_dump_json()))
        await routes.cancel_job(job.id)
        await asyncio.sleep(0.5)

        assert job.status is JobStatus.cancelled
        assert not ran

    def test_cancelling_a_running_batch_starts_no_more_outputs(
        self, served: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started: list[str] = []
        release, batch_done = threading.Event(), threading.Event()

        def blocking_run(
            self: DeepGym, env: Environment, model_output: str, **_: object
        ) -> RunResult:
            started.append(model_output)
            release.wait(10)
            return FAKE_RESULT

        real_run_batch = DeepGym.run_batch

        def tracked_run_batch(self: DeepGym, *args: object, **kwargs: object) -> object:
            try:
                return real_run_batch(self, *args, **kwargs)
            finally:
                batch_done.set()

        monkeypatch.setattr(DeepGym, 'run', blocking_run)
        monkeypatch.setattr(DeepGym, 'run_batch', tracked_run_batch)
        body = {'environment': PASSING_ENV, 'outputs': ['a', 'b', 'c'], 'max_parallel': 1}
        job_id = served.post('/v1/jobs/batch', json=body).json()['id']
        assert _eventually(lambda: started)

        cancelled = served.post(f'/v1/jobs/{job_id}/cancel')
        release.set()

        assert cancelled.json()['status'] == 'cancelled'
        assert batch_done.wait(10)
        assert started == ['a']
        assert served.get(f'/v1/jobs/{job_id}').json()['status'] == 'cancelled'
