"""Modal Sandboxes (extra ``modal``): containers on Modal's cloud, created with
``block_network=True`` so Modal itself stops all egress. Credentials come from
``modal token set`` or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET.
"""

from __future__ import annotations

import logging
import posixpath
import shlex
import time
from collections.abc import Mapping

import modal

from deepgym.sandboxes.protocol import (
    TRUSTED_PATH,
    ExecResult,
    SandboxImage,
    SandboxTimeoutError,
    stopped_error,
)

logger = logging.getLogger(__name__)

APP_NAME = 'deepgym'
# Modal deletes a sandbox this long after creation even if delete() never runs.
LIFETIME_S = 3600


class ModalSandbox:
    """A running Modal sandbox with no network."""

    network_blocked = True

    def __init__(self, image: str | None = None) -> None:
        self._image = image
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        self._expires_at = time.monotonic() + LIFETIME_S
        self._sandbox = modal.Sandbox.create(
            'sleep', 'infinity',
            app=app,
            image=(
                modal.Image.from_registry(image)
                if image
                else modal.Image.debian_slim(python_version='3.12')
            ),
            block_network=True,
            timeout=LIFETIME_S,
        )  # fmt: skip
        self.id = self._sandbox.object_id

    def booted_image(self) -> SandboxImage:
        """Return the registry image given, or None for Modal's own Debian image; no digest."""
        if self._image is None:
            reason = 'Modal built its own Debian image (Python 3.12) and reports no digest'
        else:
            reason = 'Modal reports no digest for the image it booted'
        return SandboxImage(self._image, None, reason)

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        """Run *command* with sh as root; output is stdout followed by stderr.

        Raises:
            SandboxTimeoutError: If the command outruns *timeout* seconds.
            SandboxDiedError: If Modal terminated the sandbox during the command.
            SandboxExpiredError: If Modal retired it at the end of its lifetime (see
                :func:`~deepgym.sandboxes.protocol.stopped_error`).
        """
        start = time.monotonic()
        try:
            process = self._sandbox.exec(
                '/bin/sh', '-c', command, workdir=cwd, env=dict(env) if env else None,
                timeout=timeout,
            )  # fmt: skip
            output = process.stdout.read() + process.stderr.read()
            exit_code = process.wait()
        except modal.exception.Error as exc:
            # Modal words a stopped sandbox several ways (terminated, not found); poll() tells.
            if isinstance(exc, modal.exception.SandboxTerminatedError) or self._stopped():
                message = f'Modal sandbox {self.id} was terminated: {exc}'
                raise stopped_error(message, self._expires_at) from exc
            raise
        if exit_code != 0 and self._stopped():
            message = f'Modal sandbox {self.id} was terminated: {output}'
            raise stopped_error(message, self._expires_at, output)
        # Modal kills an exec at its timeout and reports only a failed exit.
        if timeout and exit_code != 0 and time.monotonic() - start >= timeout:
            raise SandboxTimeoutError(f'command exceeded {timeout}s', output)
        return ExecResult(exit_code, output)

    def _stopped(self) -> bool:
        """Whether the sandbox itself has finished; False if Modal can't say."""
        try:
            return self._sandbox.poll() is not None
        except modal.exception.Error:
            return False

    def upload(self, content: bytes, path: str) -> None:
        """Write *content* to *path*, creating parent directories."""
        folder = shlex.quote(posixpath.dirname(path))
        self.exec(f'PATH={TRUSTED_PATH} mkdir -p {folder}', timeout=60)
        self._sandbox.filesystem.write_bytes(content, path)

    def delete(self) -> bool:
        """Terminate the sandbox (a no-op once it has finished); errors are logged.

        Returns:
            False if Modal reported a failure.
        """
        try:
            self._sandbox.terminate()
        except modal.exception.Error:
            logger.warning('Failed to delete modal sandbox %s', self.id)
            return False
        return True
