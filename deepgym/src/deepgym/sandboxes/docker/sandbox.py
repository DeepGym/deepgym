"""Docker: one container per sandbox, started with ``--network none`` so it has
loopback only. Needs a running Docker daemon (Docker Desktop, OrbStack, colima, ...).

Containers share the Docker host's kernel, so this is weaker isolation than the
VM-based providers (Apple container, E2B, Daytona). A process limit and
no-new-privileges keep a fork bomb or a setuid binary from reaching past the
container.
"""

from __future__ import annotations

import json

from deepgym.sandboxes.cli import CliContainer, RunningSandbox
from deepgym.sandboxes.protocol import SandboxImage


class DockerContainer(CliContainer):
    """A running Docker container with no network."""

    cli = 'docker'
    run_flags = ('--network', 'none', '--pids-limit', '512', '--security-opt', 'no-new-privileges')
    delete_args = ('rm', '--force')
    list_args = ('ps', '--filter', 'label=deepgym=1', '--format', '{{json .}}')
    network_blocked = True

    def _inspect_image(self) -> SandboxImage:
        """Read the registry digest of the image the container was created from.

        Docker keeps one per repository the image came through; an image built here has none.
        """
        image_id = self._answer('inspect', '--format', '{{.Image}}', self.id).strip()
        listed = self._answer('image', 'inspect', '--format', '{{json .RepoDigests}}', image_id)
        digests = sorted({entry.partition('@')[2] for entry in json.loads(listed) or ()})
        if len(digests) == 1:
            return SandboxImage(self.image, digests[0], None)
        if digests:
            reason = f'image {image_id} has several registry digests: {", ".join(digests)}'
        else:
            reason = f'image {image_id} has no registry digest (built locally, or never pulled)'
        return SandboxImage(self.image, None, reason)

    @staticmethod
    def _parse_listing(output: str) -> list[RunningSandbox]:
        """Parse ``docker ps --format '{{json .}}'``: one JSON object per line."""
        rows = [json.loads(line) for line in output.splitlines() if line.strip()]
        return [
            RunningSandbox(
                id=row['Names'], image=row['Image'], state=row['State'], started=row['CreatedAt']
            )
            for row in rows
        ]
