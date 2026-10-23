"""DeepGym exception hierarchy."""

import builtins
import re
import sys

_E2B_STATUS = re.compile(r'(\d{3}): ')
"""How e2b's envd client words an HTTP status it has no exception class for."""

_TRANSPORT_ERRORS = (
    ('httpx', ('TransportError',)),
    ('httpx2', ('TransportError',)),  # the Anthropic and OpenAI SDKs' transport, newer releases
    ('anthropic', ('APIConnectionError',)),
    ('openai', ('APIConnectionError',)),
    (
        'urllib3.exceptions',
        ('ProtocolError', 'NewConnectionError', 'MaxRetryError', 'TimeoutError'),
    ),
    ('requests.exceptions', ('ConnectionError', 'Timeout')),
    ('aiohttp', ('ClientConnectionError',)),
    ('daytona', ('DaytonaConnectionError', 'DaytonaSpotEvictedError', 'DaytonaQueueTimeoutError')),
    (
        'e2b.exceptions',
        (
            'SandboxUnreachableException',
            'ServiceBusyException',
            'RateLimitException',
            'AuthenticationException',  # our credentials refused; it carries no status
        ),
    ),
    ('pyqwest', ('ReadError', 'WriteError', 'StreamError')),  # e2b's RPC transport
    (
        'modal.exception',
        (
            'ConnectionError',
            'InternalError',
            'InternalFailure',
            'ResourceExhaustedError',
            'ServiceError',
            'AuthError',  # our credentials refused; Modal's status is an Enum, not an int
            'PermissionDeniedError',
        ),
    ),
)
"""Exception classes, by module, that mean the network or the sandbox provider failed.

A module that was never imported can't have raised one, so none is imported here.
"""

_SANDBOX_GONE_ERRORS = (('e2b.exceptions', ('SandboxNotRunningException',)),)
"""Exception classes, by module, of a provider confirming the sandbox stopped.

e2b raises one with the dropped connection that led it to check as its cause.
"""


class DeepGymError(Exception):
    """Base exception for all DeepGym errors."""


class VerifierError(DeepGymError):
    """Raise when verifier output is not valid JSON or the verifier failed to run."""


class InvalidScoreError(VerifierError):
    """Raise when a verifier reports a score that is NaN, infinite, or outside [0, 1]."""


class SandboxError(DeepGymError):
    """Raise when sandbox creation or execution fails."""


class SandboxDiedError(SandboxError):
    """Raise when the sandbox itself stopped during a command (terminated, killed, out of
    memory), as opposed to the provider failing to answer. Code running in it may be why.

    Args:
        message: What happened.
        output: What the command printed first, where the provider returns it, else ''.
    """

    def __init__(self, message: str, output: str = '') -> None:
        super().__init__(message)
        self.output = output


class SandboxUnreachableError(SandboxError):
    """Raise when the provider gave no answer about a command, or none to trust: its daemon
    hung or can't be reached, before the command started or while it ran. Infrastructure,
    never the policy's doing (see :func:`is_infrastructure_failure`)."""


class SandboxExpiredError(SandboxError):
    """Raise when the provider retired the sandbox at the end of its lifetime, whatever ran
    in it. Not the policy's doing unless its own actions ran the clock out (see
    :func:`deepgym.session.failure_result`)."""


class TimeoutError(DeepGymError):
    """Raise when execution exceeds the configured timeout."""


class ActionError(DeepGymError):
    """Raise when the environment rejects an agent's action (not allowed, unknown key)."""


class NoVerdictError(DeepGymError):
    """Raise when a run got no verdict (its sandbox or verifier failed), so it has no reward.

    Args:
        reason: What failed, kept as ``reason``.
        advice: What the caller can do about it, added to the message.
    """

    def __init__(self, reason: str, advice: str = '') -> None:
        super().__init__(f'{reason} {advice}'.strip())
        self.reason = reason


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
    5xx or 429, the provider refusing our credentials (401, 403), or the provider evicting
    the sandbox, anywhere along the exception's ``__cause__`` chain. Such a run got no
    verdict, and trainers keep it out of training.

    Anything else is the policy's doing and is scored: a command or verifier that ran out
    of time, a sandbox the provider reports gone or stopped after the policy acted, a
    failed write, garbage output. DeepGym's ``TimeoutError`` and a provider confirming the
    sandbox stopped (:class:`SandboxDiedError`, or the provider's own exception) end the
    walk, whatever their causes.

    ponytail: a provider 5xx caused by a policy that exhausted its sandbox still reads as
    infrastructure; the masked-rate warning in :mod:`deepgym.integrations.reward` shows it.
    """
    gone = _loaded(_SANDBOX_GONE_ERRORS)
    transport = _loaded(_TRANSPORT_ERRORS)
    while exc is not None:
        if isinstance(exc, (TimeoutError, SandboxDiedError, *gone)):
            return False
        if isinstance(exc, (ConnectionError, builtins.TimeoutError, SandboxUnreachableError)):
            return True
        if isinstance(exc, transport):
            return True
        status = _status_code(exc)
        if status is not None and (status >= 500 or status in (401, 403, 429)):
            return True
        exc = exc.__cause__
    return False


def _loaded(table: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[type, ...]:
    """Return the classes *table* names from modules already imported."""
    return tuple(
        error
        for module, names in table
        if module in sys.modules
        for name in names
        if isinstance(error := getattr(sys.modules[module], name, None), type)
    )


def _status_code(exc: BaseException) -> int | None:
    """Return the HTTP status *exc* carries, as provider SDK, httpx and aiohttp errors do.

    e2b gives an envd status it has no class for (a 503, say) only in the message.
    """
    response = getattr(exc, 'response', None)
    for status in (getattr(exc, 'status_code', None), getattr(response, 'status_code', None)):
        if isinstance(status, int):
            return status
    status = getattr(exc, 'status', None)  # aiohttp
    if isinstance(status, int):
        return status
    if type(exc).__module__.startswith('e2b') and (match := _E2B_STATUS.match(str(exc))):
        return int(match.group(1))
    return None
