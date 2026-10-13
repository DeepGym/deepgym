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

from deepgym.sandboxes.protocol import TRUSTED_PATH, ExecResult, SandboxTimeoutError

logger = logging.getLogger(__name__)

APP_NAME = 'deepgym'
# Modal deletes a sandbox this long after creation even if delete() never runs.
LIFETIME_S = 3600


class ModalSandbox:
    """A running Modal sandbox with no network."""

    network_blocked = True

    def __init__(self, image: str | None = None) -> None:
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
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
        """
        start = time.monotonic()
        process = self._sandbox.exec(
            '/bin/sh', '-c', command, workdir=cwd, env=dict(env) if env else None, timeout=timeout
        )
        output = process.stdout.read() + process.stderr.read()
        exit_code = process.wait()
        # Modal kills an exec at its timeout and reports only a failed exit.
        if timeout and exit_code != 0 and time.monotonic() - start >= timeout:
            raise SandboxTimeoutError(f'command exceeded {timeout}s')
        return ExecResult(exit_code, output)

    def upload(self, content: bytes, path: str) -> None:
        """Write *content* to *path*, creating parent directories."""
        folder = shlex.quote(posixpath.dirname(path))
        self.exec(f'PATH={TRUSTED_PATH} mkdir -p {folder}', timeout=60)
        self._sandbox.filesystem.write_bytes(content, path)

    def delete(self) -> None:
        """Terminate the sandbox; errors are logged (it may already be gone)."""
        try:
            self._sandbox.terminate()
        except modal.exception.Error:
            logger.warning('Failed to delete modal sandbox %s', self.id)
