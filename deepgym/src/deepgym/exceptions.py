"""DeepGym exception hierarchy."""

import builtins
import sys

_TRANSPORT_ERRORS = (
    ('httpx', ('TransportError',)),
    (
        'urllib3.exceptions',
        ('ProtocolError', 'NewConnectionError', 'MaxRetryError', 'TimeoutError'),
    ),
    ('requests.exceptions', ('ConnectionError', 'Timeout')),
    ('aiohttp', ('ClientConnectionError',)),
    ('daytona', ('DaytonaConnectionError', 'DaytonaSpotEvictedError', 'DaytonaQueueTimeoutError')),
    ('modal.exception', ('ConnectionError', 'InternalFailure', 'ServiceError')),
)
"""Exception classes, by module, that mean the network or the sandbox provider failed.

A module that was never imported can't have raised one, so none is imported here.
"""


class DeepGymError(Exception):
    """Base exception for all DeepGym errors."""


class VerifierError(DeepGymError):
    """Raise when verifier output is not valid JSON or the verifier failed to run."""


class InvalidScoreError(VerifierError):
    """Raise when a verifier reports a score that is NaN, infinite, or outside [0, 1]."""


class SandboxError(DeepGymError):
    """Raise when sandbox creation or execution fails."""


class TimeoutError(DeepGymError):
    """Raise when execution exceeds the configured timeout."""


class ActionError(DeepGymError):
    """Raise when the environment rejects an agent's action (not allowed, unknown key)."""


class NoVerdictError(DeepGymError):
    """Raise when a run got no verdict (its sandbox or verifier failed), so it has no reward."""


def error_type_for(exc: BaseException) -> str:
    """Name the failure for a zero-score result."""
    if isinstance(exc, TimeoutError):
        return 'timeout'
    if isinstance(exc, InvalidScoreError):
        return 'invalid_score'
    if isinstance(exc, SandboxError):
        return 'sandbox_error'
    if isinstance(exc, DeepGymError):
        return 'verifier_error'
    return 'unknown_error'


def is_infrastructure_failure(exc: BaseException) -> bool:
    """Return True if *exc* is a failure of the network or the sandbox provider.

    That is a connection or transport error, a provider call timing out, a provider HTTP
    5xx or 429, or the provider evicting the sandbox, anywhere along the exception's
    ``__cause__`` chain. Such a run got no verdict, and trainers keep it out of training.

    Anything else is the policy's doing and is scored: a command or verifier that ran out
    of time (DeepGym's ``TimeoutError``, which ends the walk), a sandbox the provider reports
    gone or stopped after the policy acted, a failed write, garbage output.

    ponytail: a provider 5xx caused by a policy that exhausted its sandbox still reads as
    infrastructure; the masked-rate warning in :mod:`deepgym.integrations.reward` shows it.
    """
    transport = tuple(
        error
        for module, names in _TRANSPORT_ERRORS
        if module in sys.modules
        for name in names
        if isinstance(error := getattr(sys.modules[module], name, None), type)
    )
    while exc is not None:
        if isinstance(exc, TimeoutError):
            return False
        if isinstance(exc, (ConnectionError, builtins.TimeoutError, *transport)):
            return True
        status = _status_code(exc)
        if status is not None and (status >= 500 or status == 429):
            return True
        exc = exc.__cause__
    return False


def _status_code(exc: BaseException) -> int | None:
    """Return the HTTP status *exc* carries, as provider SDK, httpx and aiohttp errors do."""
    response = getattr(exc, 'response', None)
    for status in (getattr(exc, 'status_code', None), getattr(response, 'status_code', None)):
        if isinstance(status, int):
            return status
    status = getattr(exc, 'status', None)  # aiohttp
    return status if isinstance(status, int) else None
