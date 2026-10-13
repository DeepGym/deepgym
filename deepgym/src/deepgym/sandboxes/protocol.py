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

from collections.abc import Mapping
from typing import NamedTuple, Protocol

from deepgym.exceptions import TimeoutError

TRUSTED_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
"""Root-owned system folders only; no agent work directory may hold or sit in them."""

TIMEOUT_EXIT_CODES = frozenset({124, 137})
"""Exit codes of coreutils ``timeout``: 124 after SIGTERM, 137 after the follow-up SIGKILL."""


class ExecResult(NamedTuple):
    """Exit code and combined stdout/stderr of a shell command."""

    exit_code: int
    output: str


class SandboxTimeoutError(TimeoutError):
    """Raised when a sandbox command outruns its timeout."""


class Sandbox(Protocol):
    """A running sandbox, deleted by :meth:`delete`."""

    id: str
    network_blocked: bool

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
        """
        ...

    def upload(self, content: bytes, path: str) -> None:
        """Write *content* to the absolute *path* as root, creating parent directories."""
        ...

    def delete(self) -> bool:
        """Stop and remove the sandbox; never raises.

        Returns:
            False if the provider reported a failure, so the sandbox may still exist; True
            once it is gone, including when it was gone already.
        """
        ...
