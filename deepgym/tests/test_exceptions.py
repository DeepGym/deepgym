"""Tests for is_infrastructure_failure: what trainers keep out, and what they score.

Only a failure of the network or the sandbox provider is kept out of training. Anything
that follows from the policy's own actions is scored, so a policy can't escape a 0 by
breaking its sandbox.
"""

from __future__ import annotations

import builtins

import httpx
import pytest

from deepgym.exceptions import SandboxError, VerifierError, is_infrastructure_failure
from deepgym.exceptions import TimeoutError as DeepGymTimeout
from deepgym.sandboxes import SandboxTimeoutError

_REQUEST = httpx.Request('POST', 'https://provider.test/process/exec')


def _status(code: int) -> httpx.HTTPStatusError:
    response = httpx.Response(code, request=_REQUEST)
    return httpx.HTTPStatusError(f'HTTP {code}', request=_REQUEST, response=response)


def _caused_by(error: BaseException, cause: BaseException) -> BaseException:
    error.__cause__ = cause
    return error


class _SdkError(Exception):
    """An SDK error that carries its HTTP status, as Daytona's do."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f'HTTP {status_code}')
        self.status_code = status_code


@pytest.mark.parametrize(
    'error',
    [
        ConnectionResetError('connection reset by peer'),
        builtins.TimeoutError('the provider call timed out'),
        httpx.ConnectError('connection refused', request=_REQUEST),
        httpx.ReadTimeout('no response', request=_REQUEST),
        _status(502),
        _SdkError(503),
        _SdkError(429),
        _status(401),  # our credentials
        _SdkError(403),
        # DeepGym wraps provider errors; the cause decides.
        _caused_by(SandboxError('click failed'), ConnectionRefusedError()),
    ],
    ids=[
        'reset',
        'call_timeout',
        'refused',
        'read_timeout',
        '502',
        'sdk_503',
        'rate_limit',
        'unauthorized',
        'forbidden',
        'wrapped',
    ],
)
def test_network_and_provider_failures_are_infrastructure(error: BaseException) -> None:
    assert is_infrastructure_failure(error)


@pytest.mark.parametrize(
    'error',
    [
        SandboxError('Failed to write out.txt: Is a directory'),
        SandboxError('Sandbox 7f3a is not running'),
        _SdkError(404),
        _SdkError(409),
        _status(400),
        DeepGymTimeout('Command exceeded 30s timeout'),
        # A command that ran out of time hung on the policy, even when a transport timeout
        # is what noticed.
        _caused_by(SandboxTimeoutError('command exceeded 30s'), builtins.TimeoutError()),
        VerifierError('Verifier did not produce valid JSON'),
        IsADirectoryError('out.txt'),
        ValueError('garbage'),
    ],
    ids=[
        'failed_write',
        'sandbox_stopped',
        'sdk_404',
        'sdk_409',
        '400',
        'command_timeout',
        'command_timeout_over_transport',
        'verifier_error',
        'local_os_error',
        'garbage',
    ],
)
def test_failures_that_follow_the_policy_are_not(error: BaseException) -> None:
    assert not is_infrastructure_failure(error)


def test_daytona_errors_are_told_apart() -> None:
    """A command timeout is the policy's; a dropped connection or a spot eviction isn't."""
    daytona = pytest.importorskip('daytona')

    assert is_infrastructure_failure(daytona.DaytonaConnectionError('connection dropped'))
    assert is_infrastructure_failure(daytona.DaytonaSpotEvictedError('evicted'))
    assert is_infrastructure_failure(daytona.DaytonaError('down', status_code=500))
    assert not is_infrastructure_failure(daytona.DaytonaProcessExecutionTimeoutError('slow'))
    assert not is_infrastructure_failure(daytona.DaytonaNotFoundError('gone', status_code=404))


def test_e2b_errors_are_told_apart() -> None:
    """e2b reports an unreachable sandbox, a busy service or a 503 as its own; not a dead one."""
    e2b = pytest.importorskip('e2b.exceptions')

    assert is_infrastructure_failure(e2b.SandboxUnreachableException('connection lost'))
    assert is_infrastructure_failure(e2b.ServiceBusyException('no capacity'))
    assert is_infrastructure_failure(e2b.RateLimitException('slow down'))
    assert is_infrastructure_failure(e2b.SandboxException('503: unavailable'))
    assert not is_infrastructure_failure(e2b.SandboxNotRunningException('was not found'))
    assert not is_infrastructure_failure(e2b.SandboxException('400: bad request'))
    assert not is_infrastructure_failure(e2b.SandboxException('disk full'))


def test_e2b_dropped_connections_turn_on_whether_the_sandbox_still_runs() -> None:
    """e2b checks the sandbox after a dropped connection, and its answer decides.

    Built with e2b's own handlers, wrapped as DeepGym's e2b sandbox wraps them.
    """
    pytest.importorskip('e2b')
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError
    from e2b.envd.api import handle_envd_api_transport_exception
    from e2b.envd.rpc import handle_rpc_exception
    from pyqwest import ReadError

    command = _caused_by(ConnectError(Code.UNAVAILABLE, 'stream reset'), ReadError('reset'))
    still_running = handle_rpc_exception(command, sandbox_running=True)
    command_stopped = handle_rpc_exception(command, sandbox_running=False)
    file_stopped = handle_envd_api_transport_exception(
        httpx.ReadError('dropped', request=_REQUEST), sandbox_running=False
    )

    assert is_infrastructure_failure(still_running)
    assert not is_infrastructure_failure(_caused_by(SandboxError('gone'), command_stopped))
    assert not is_infrastructure_failure(_caused_by(SandboxError('gone'), file_stopped))


def test_modal_errors_are_told_apart() -> None:
    """Modal's server errors and quotas are its own; a sandbox it reports dead isn't."""
    modal = pytest.importorskip('modal')

    assert is_infrastructure_failure(modal.exception.InternalError('gRPC INTERNAL'))
    assert is_infrastructure_failure(modal.exception.ResourceExhaustedError('rate limited'))
    assert is_infrastructure_failure(modal.exception.ConnectionError('no route'))
    assert not is_infrastructure_failure(modal.exception.SandboxTerminatedError('terminated'))
    assert not is_infrastructure_failure(modal.exception.ExecTimeoutError('slow'))


@pytest.mark.parametrize(
    ('module', 'name'),
    [
        ('e2b.exceptions', 'AuthenticationException'),
        ('modal.exception', 'AuthError'),
        ('modal.exception', 'PermissionDeniedError'),
    ],
)
def test_provider_auth_errors_without_an_http_status_are_infrastructure(
    monkeypatch: pytest.MonkeyPatch, module: str, name: str
) -> None:
    """Our credentials refused by an SDK whose error carries no integer status (E2B's has
    none, Modal's is an Enum): never the policy's zero."""
    import sys
    import types

    error = type(name, (Exception,), {})
    monkeypatch.setitem(sys.modules, module, types.SimpleNamespace(**{name: error}))

    assert is_infrastructure_failure(_caused_by(SandboxError('exec failed'), error('denied')))
