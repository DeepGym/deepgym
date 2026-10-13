"""DeepGym exception hierarchy."""


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
