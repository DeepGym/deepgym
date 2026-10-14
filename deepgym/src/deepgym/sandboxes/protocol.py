"""What a sandbox provider implements so DeepGym's judge and shell sessions can run on it.

A sandbox is one throwaway Linux machine. ``exec`` runs a shell command as root,
because the judge creates its own unprivileged users inside (see
:data:`deepgym.judge.SANDBOX_SETUP`), and ``upload`` writes a file as root.
Untrusted code never runs as root: the judge and sessions drop to those users.

``exec`` starts ``/bin/sh`` by absolute path, so the image's PATH, which may include
folders the agent writes, never picks the shell that runs as root. Commands that run as
root and must not trust the image's environment wrap themselves in
:data:`TRUSTED_PATH` (see :func:`deepgym.session.trusted`).

``network_blocked`` says the provider itself stops all egress. When it doesn't,
DeepGym puts untrusted code in its own offline network namespace instead, and
refuses to run if the sandbox can't make one.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import NamedTuple, Protocol

from deepgym.exceptions import SandboxDiedError, SandboxError, SandboxExpiredError, TimeoutError

TRUSTED_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
"""Root-owned system folders only; no agent work directory may hold or sit in them."""

TIMEOUT_EXIT_CODES = frozenset({124, 137})
"""Exit codes of coreutils ``timeout``: 124 after SIGTERM, 137 after the follow-up SIGKILL."""


class ExecResult(NamedTuple):
    """Exit code and combined stdout/stderr of a shell command."""

    exit_code: int
    output: str


class SandboxImage(NamedTuple):
    """The image a sandbox booted, as its provider reports it."""

    name: str | None
    """Its tag, or the snapshot or template name; None if the provider names none."""

    digest: str | None
    """Content digest (``sha256:...``) the provider resolved it to."""

    unavailable: str | None
    """Why there is no digest."""


class SandboxTimeoutError(TimeoutError):
    """Raised when a sandbox command outruns its timeout.

    Args:
        message: What happened.
        output: What the command printed first, where the provider returns it, else ''.
    """

    def __init__(self, message: str, output: str = '') -> None:
        super().__init__(message)
        self.output = output


def stopped_error(message: str, expires_at: float, output: str = '') -> SandboxError:
    """Return the error for a sandbox its provider reports stopped.

    From *expires_at* on (``time.monotonic()``, taken before the sandbox was created, so
    never after the provider's own deadline), the provider retired it at the end of its
    lifetime, whatever ran in it: :class:`~deepgym.exceptions.SandboxExpiredError`. Before
    then something in it stopped it: :class:`SandboxDiedError`.

    Args:
        message: What happened.
        expires_at: When the provider's lifetime for the sandbox runs out.
        output: What the command printed first, for :class:`SandboxDiedError`.
    """
    if time.monotonic() >= expires_at:
        return SandboxExpiredError(f'{message} (its provider lifetime was up)')
    return SandboxDiedError(message, output)


class Sandbox(Protocol):
    """A running sandbox, deleted by :meth:`delete`."""

    id: str
    network_blocked: bool

    def booted_image(self) -> SandboxImage:
        """Return the image the sandbox booted and its digest; never raises.

        A provider that can't say gives the reason in ``unavailable`` instead of a guess.
        """
        ...

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        """Run *command* with ``/bin/sh -c`` as root, in the image's environment.

        Raises:
            SandboxTimeoutError: If the command outruns *timeout* seconds.
            SandboxDiedError: If the sandbox itself stopped during the command.
        """
        ...

    def upload(self, content: bytes, path: str) -> None:
        """Write *content* to the absolute *path* as root, creating parent directories.

        Raises:
            SandboxDiedError: If the sandbox itself has stopped.
        """
        ...

    def delete(self) -> bool:
        """Stop and remove the sandbox; never raises.

        Returns:
            False if the provider reported a failure, so the sandbox may still exist; True
            once it is gone, including when it was gone already.
        """
        ...
