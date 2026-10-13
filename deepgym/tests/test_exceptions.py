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
