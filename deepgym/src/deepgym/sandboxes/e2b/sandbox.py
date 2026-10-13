"""E2B (extra ``e2b``): Firecracker microVMs on E2B's cloud, created with
``allow_internet_access=False`` so E2B itself stops all egress. Needs E2B_API_KEY.

The default template's user is ``user``; every call here passes ``user='root'``.
"""

from __future__ import annotations

import logging
import shlex
import time
from collections.abc import Mapping

from e2b import (
    CommandExitException,
    SandboxNotRunningException,
    SandboxUnreachableException,
    TimeoutException,
)
from e2b import Sandbox as E2BClient

from deepgym.exceptions import SandboxDiedError, SandboxError
from deepgym.sandboxes.protocol import (
    TIMEOUT_EXIT_CODES,
    ExecResult,
    SandboxImage,
    SandboxTimeoutError,
)

logger = logging.getLogger(__name__)

# E2B deletes a sandbox this long after creation even if delete() never runs.
LIFETIME_S = 3600
# How much longer than the command's own deadline the SDK waits for it to end.
_WAIT_GRACE_S = 15


class E2BSandbox:
    """A running E2B sandbox with no network."""

    network_blocked = True

    def __init__(self, template: str | None = None) -> None:
        self._template = template
        self._sandbox = E2BClient.create(
            template=template, timeout=LIFETIME_S, allow_internet_access=False
        )
        self.id = self._sandbox.sandbox_id

    def booted_image(self) -> SandboxImage:
        """Return the template the sandbox booted (E2B's default if none was given); no digest."""
        template = self._template or E2BClient.default_template
        return SandboxImage(template, None, 'E2B boots templates, which have no image digest')

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        """Run *command* as root; output is stdout followed by stderr.

        E2B's own timeout only stops waiting and leaves the command running, so the
        deadline is enforced in the sandbox by coreutils ``timeout``, which kills the
        command's process group before this returns.

        Raises:
            SandboxTimeoutError: If the command outruns *timeout* seconds.
            SandboxDiedError: If the sandbox stopped running during the command.
            SandboxError: If E2B can't reach the sandbox.
        """
        if timeout:
            command = f'/usr/bin/timeout --kill-after=2 {timeout} /bin/sh -c {shlex.quote(command)}'
        start = time.monotonic()
        try:
            result = self._sandbox.commands.run(
                command,
                user='root',
                cwd=cwd,
                envs=dict(env) if env else None,
                timeout=timeout + _WAIT_GRACE_S if timeout else LIFETIME_S,
            )
        except CommandExitException as exc:
            # E2B raises on any non-zero exit; that's an ordinary result here.
            result = exc
        # Both subclass TimeoutException, but mean the sandbox stopped or can't be reached.
        except SandboxNotRunningException as exc:
            raise SandboxDiedError(f'E2B sandbox {self.id} stopped running: {exc}') from exc
        except SandboxUnreachableException as exc:
            raise SandboxError(f'E2B sandbox {self.id} is not reachable: {exc}') from exc
        except TimeoutException as exc:
            raise SandboxTimeoutError(f'command exceeded {timeout}s') from exc
        if timeout and result.exit_code in TIMEOUT_EXIT_CODES:
            if time.monotonic() - start >= timeout:
                raise SandboxTimeoutError(f'command exceeded {timeout}s')
        return ExecResult(result.exit_code, result.stdout + result.stderr)

    def upload(self, content: bytes, path: str) -> None:
        """Write *content* to *path* as root (E2B creates parent directories)."""
        self._sandbox.files.write(path, content, user='root')

    def delete(self) -> bool:
        """Kill the sandbox; errors are logged. Return False if E2B reported a failure.

        E2B's kill answers False, without raising, for a sandbox that is already gone.
        """
        try:
            self._sandbox.kill()
        except Exception:
            logger.warning('Failed to delete e2b sandbox %s', self.id)
            return False
        return True
