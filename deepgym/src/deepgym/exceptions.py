"""DeepGym exception hierarchy."""


class DeepGymError(Exception):
    """Base exception for all DeepGym errors."""


class VerifierError(DeepGymError):
    """Raise when verifier output is not valid JSON or the verifier failed to run."""


class SandboxError(DeepGymError):
    """Raise when sandbox creation or execution fails."""


class TimeoutError(DeepGymError):
    """Raise when execution exceeds the configured timeout."""
