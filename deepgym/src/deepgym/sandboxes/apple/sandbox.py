"""Apple ``container`` (https://github.com/apple/container): one lightweight Linux VM per
sandbox on macOS. Needs the CLI with its service running (``container system start``).

Apple's CLI has no ``--network none``, so the provider doesn't block egress. The VM
gets CAP_SYS_ADMIN and CAP_NET_ADMIN only so root can build an offline network
namespace (with loopback up) for untrusted code, which always runs as a non-root
user with every capability dropped.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from deepgym.sandboxes.cli import CliContainer, RunningSandbox
from deepgym.sandboxes.protocol import SandboxImage


class AppleContainer(CliContainer):
    """A running Apple container VM."""

    cli = 'container'
    run_flags = ('--cap-add', 'CAP_SYS_ADMIN', '--cap-add', 'CAP_NET_ADMIN')
    delete_args = ('delete', '--force')
    list_args = ('ls', '--format', 'json')
    network_blocked = False

    @classmethod
    def available(cls) -> bool:
        """Return True if the container CLI is installed and its service is running."""
        if shutil.which(cls.cli) is None:
            return False
        try:
            status = subprocess.run(
                [cls.cli, 'system', 'status'], capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return status.returncode == 0 and 'running' in status.stdout

    def _inspect_image(self) -> SandboxImage:
        """Read the reference and digest of the image the container was created from."""
        image = json.loads(self._answer('inspect', self.id))[0]['configuration']['image']
        return SandboxImage(image['reference'], image['descriptor']['digest'], None)

    @staticmethod
    def _parse_listing(output: str) -> list[RunningSandbox]:
        """Parse ``container ls --format json``: a JSON array of container records."""
        return [
            RunningSandbox(
                id=row['id'],
                image=row['configuration']['image']['reference'],
                state=row['status']['state'],
                started=row['status'].get('startedDate', ''),
            )
            for row in json.loads(output or '[]')
        ]
