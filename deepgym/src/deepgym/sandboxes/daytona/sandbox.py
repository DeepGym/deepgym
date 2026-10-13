"""Daytona (extra ``daytona``): cloud or self-hosted sandboxes.

:class:`DaytonaSandbox` is the provider the judge and live sessions use, and
:class:`~deepgym.sandboxes.daytona.desktop.DaytonaDesktop` drives its desktop.
``create_sandbox`` and ``cleanup_sandbox`` hand out raw Daytona sandboxes for code
that needs Daytona's full API (SWE-bench Pro).
"""

from __future__ import annotations

import logging
import posixpath
import re
import shlex
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING
from uuid import uuid4

from deepgym.exceptions import SandboxDiedError, SandboxError
from deepgym.models import Environment, MultiTurnEnvironment
from deepgym.sandboxes.protocol import (
    TRUSTED_PATH,
    ExecResult,
    SandboxImage,
    SandboxTimeoutError,
)

if TYPE_CHECKING:
    from daytona import (
        AsyncSandbox,
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        Daytona,
        Sandbox,
    )
    from daytona._sync.computer_use import ComputerUse

logger = logging.getLogger(__name__)


def _sdk() -> ModuleType:
    """Import the daytona SDK.

    Raises:
        SandboxError: If it isn't installed.
    """
    try:
        import daytona
    except ImportError as exc:
        raise SandboxError(
            "daytona is required. Install it with: pip install 'deepgym[daytona]'"
        ) from exc
    return daytona


def create_params(
    env: Environment,
) -> CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams:
    """Return params for an ephemeral judge sandbox: ``env.snapshot``, else ``env.image``,
    else a Debian Python 3.12 image.

    Raises:
        SandboxError: If the daytona SDK is missing.
    """
    sdk = _sdk()
    if env.snapshot:
        return sdk.CreateSandboxFromSnapshotParams(snapshot=env.snapshot, ephemeral=True)
    image = sdk.Image.base(env.image) if env.image else sdk.Image.debian_slim('3.12')
    return sdk.CreateSandboxFromImageParams(image=image, ephemeral=True)


def _session_params(
    env: MultiTurnEnvironment,
) -> CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams:
    """Return params for an ephemeral, network-blocked session sandbox: ``env.image``
    unless a snapshot is set, else ``env.snapshot`` or Daytona's default snapshot (the
    one with a desktop).

    Raises:
        SandboxError: If the daytona SDK is missing.
    """
    sdk = _sdk()
    options = {'ephemeral': True, 'network_block_all': True}
    if env.image and not env.snapshot:
        return sdk.CreateSandboxFromImageParams(image=sdk.Image.base(env.image), **options)
    return sdk.CreateSandboxFromSnapshotParams(snapshot=env.snapshot, **options)


def create_sandbox(env: Environment, daytona: Daytona) -> Sandbox:
    """Create a raw Daytona sandbox from :func:`create_params`.

    Raises:
        SandboxError: If the daytona SDK is missing or creation fails.
    """
    params = create_params(env)
    try:
        return daytona.create(params)
    except Exception as exc:
        raise SandboxError(f'Failed to create sandbox: {exc}') from exc


def cleanup_sandbox(sandbox: Sandbox, daytona: Daytona, *, wait: bool = False) -> bool:
    """Delete a raw Daytona sandbox, logging rather than raising errors.

    Returns:
        False if Daytona reported a failure; True once the sandbox is gone, including when
        Daytona no longer knows it (404).
    """
    try:
        daytona.delete(sandbox, wait=wait)
    except Exception as exc:
        if getattr(exc, 'status_code', None) == 404:
            return True
        logger.warning('Failed to clean up sandbox %s', getattr(sandbox, 'id', '?'))
        return False
    return True


def _delete_labelled(daytona: Daytona, label: dict[str, str]) -> None:
    """Delete any sandbox carrying *label*, so a failed create doesn't leak one."""
    try:
        for sandbox in daytona.list(_sdk().ListSandboxesQuery(labels=label)):
            cleanup_sandbox(sandbox, daytona)
    except Exception:
        logger.warning('Could not check for leaked sandboxes labelled %s', label)


def snapshot_digest(ref: str | None) -> str | None:
    """Return the image digest a Daytona snapshot reference carries, or None if it carries none.

    Daytona stores a snapshot of a registry image as ``<registry>/daytona-<hex>:daytona``,
    where ``<hex>`` is that image's registry digest (checked against Docker Hub for
    ``daytonaio/sandbox:0.8.0`` and three other images).
    """
    # ponytail: an observed Daytona naming scheme, not a documented API; if Daytona drops it,
    # traces fall back to "no digest". Use a digest field once Daytona's API has one.
    match = re.fullmatch(r'[^@]+/daytona-([0-9a-f]{64}):daytona', ref or '')
    return f'sha256:{match[1]}' if match else None


def is_timeout(exc: Exception) -> bool:
    """Return True if a Daytona exec failed because the command outran its timeout."""
    try:
        from daytona import DaytonaProcessExecutionTimeoutError
    except ImportError:
        return False
    return isinstance(exc, DaytonaProcessExecutionTimeoutError)


def has_stopped(exc: Exception, sandbox: Sandbox | AsyncSandbox) -> bool:
    """Return True if a call that failed with *exc* means the sandbox itself stopped: the
    freshly read *sandbox* isn't ``started``, and Daytona didn't evict it or time it out in
    its queue (its own doing), as the error or the sandbox's record says."""
    from daytona import DaytonaQueueTimeoutError, DaytonaSpotEvictedError

    if isinstance(exc, DaytonaSpotEvictedError | DaytonaQueueTimeoutError):
        return False
    evicted = sandbox.spot_evicted_at or sandbox.queue_timed_out_at
    return sandbox.state != 'started' and not evicted


class DaytonaSandbox:
    """A Daytona sandbox, deleted by :meth:`delete`.

    ``exec`` and ``upload`` act as root, as the :class:`~deepgym.sandboxes.Sandbox`
    protocol requires. Daytona runs both as the image's default user, which on its
    default snapshot is a passwordless sudoer, so there commands go through ``sudo``
    and uploads land in that user's private directory before root moves them.

    ``network_blocked`` stays False even with ``network_block_all``: Daytona serves its
    toolbox API (run any command as that sudoer) on loopback, so untrusted code must
    always get its own network namespace here.
    """

    network_blocked = False

    def __init__(self, daytona: Daytona, env: Environment | MultiTurnEnvironment) -> None:
        """Create a sandbox for *env*.

        A single run gets :func:`create_params`. A live session (a multi-turn *env*)
        gets its image or Daytona's default snapshot, which has the desktop, with the
        network blocked and ``setup_timeout`` to boot; deleting it waits until Daytona
        confirms it is gone, because the HTTP API frees a session slot on close.

        Raises:
            SandboxError: If creation fails or the sandbox can't be reached.
        """
        self._daytona = daytona
        self._image = env.image
        self._session = isinstance(env, MultiTurnEnvironment)
        params = _session_params(env) if self._session else create_params(env)
        # A label lets us find and delete the sandbox if create() fails after allocating it.
        label = {'deepgym_sandbox': uuid4().hex}
        try:
            self._sandbox = daytona.create(
                params.model_copy(update={'labels': label}),
                timeout=env.setup_timeout if self._session else 60,
            )
        except Exception as exc:
            _delete_labelled(daytona, label)
            raise SandboxError(f'Failed to create sandbox: {exc}') from exc
        self.id = str(getattr(self._sandbox, 'id', 'unknown'))
        self._sudo = ''
        self._stage = ''
        try:
            probe = self.exec('[ "$(id -u)" = 0 ] && echo root || mktemp -d', timeout=30)
        except Exception as exc:
            self.delete()
            raise SandboxError(f'Sandbox {self.id} is not reachable: {exc}') from exc
        stage = probe.output.strip()
        if probe.exit_code != 0 or not stage:
            self.delete()
            raise SandboxError(f'Could not prepare sandbox {self.id}: {probe.output[-500:]}')
        if stage != 'root':
            self._sudo, self._stage = '/usr/bin/sudo -n -E ', stage

    @property
    def computer_use(self) -> ComputerUse:
        """Daytona's desktop API for this sandbox."""
        return self._sandbox.computer_use

    def booted_image(self) -> SandboxImage:
        """Return the image the sandbox booted and its digest, as Daytona reports them.

        Only a snapshot of a registry image has a digest: Daytona's default snapshot, or one
        the environment names. A sandbox Daytona built, from the environment's image or a
        Dockerfile, comes with a build reference but no digest of what it pulled.
        """
        name = self._sandbox.snapshot
        if not name:
            return SandboxImage(
                self._image, None, 'Daytona built the sandbox and reports no digest'
            )
        try:
            snapshot = self._daytona.snapshot.get(name)
        except Exception as exc:
            return SandboxImage(name, None, f'could not look up Daytona snapshot {name}: {exc}')
        digest = None if snapshot.build_info else snapshot_digest(snapshot.ref)
        if snapshot.image_name and digest:
            return SandboxImage(snapshot.image_name, digest, None)
        reason = f'Daytona reports no registry digest for snapshot {name}'
        return SandboxImage(snapshot.image_name or name, None, reason)

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        """Run *command* with sh as root.

        Raises:
            SandboxTimeoutError: If the command outruns *timeout* seconds.
            SandboxDiedError: If the sandbox stopped during the command.
        """
        try:
            response = self._sandbox.process.exec(
                f'{self._sudo}/bin/sh -c {shlex.quote(command)}',
                cwd=cwd,
                env=dict(env) if env else None,
                timeout=timeout,
            )
        except Exception as exc:
            if is_timeout(exc):
                raise SandboxTimeoutError(f'command exceeded {timeout}s') from exc
            self._raise_if_stopped(exc)
            raise
        return ExecResult(response.exit_code, response.result)

    def _raise_if_stopped(self, exc: Exception) -> None:
        """Raise what a call that failed with *exc* means if the sandbox stopped: Daytona's
        own error when its record says it evicted the sandbox or timed it out in its queue (a
        provider failure, whatever the call raised), else :class:`SandboxDiedError`. Return
        if it still runs or Daytona can't say."""
        try:
            self._sandbox.refresh_data()
        except Exception:
            return
        from daytona import DaytonaQueueTimeoutError, DaytonaSpotEvictedError

        if self._sandbox.spot_evicted_at:
            raise DaytonaSpotEvictedError(f'Daytona evicted sandbox {self.id}: {exc}') from exc
        if self._sandbox.queue_timed_out_at:
            message = f'Daytona timed out sandbox {self.id} in its queue: {exc}'
            raise DaytonaQueueTimeoutError(message) from exc
        if has_stopped(exc, self._sandbox):
            raise SandboxDiedError(f'Daytona sandbox {self.id} stopped: {exc}') from exc

    def upload(self, content: bytes, path: str) -> None:
        """Write *content* to *path* as root, creating parent directories.

        Raises:
            SandboxError: If root can't move the staged file into place.
        """
        if not self._stage:
            self._sandbox.fs.upload_file(content, path)
            return
        staged = f'{self._stage}/{uuid4().hex}'
        self._sandbox.fs.upload_file(content, staged)
        src, dst = shlex.quote(staged), shlex.quote(path)
        moved = self.exec(
            f'PATH={TRUSTED_PATH}; mkdir -p {shlex.quote(posixpath.dirname(path))} && '
            f'cat {src} > {dst}; '
            f'status=$?; rm -f {src}; exit $status',
            timeout=60,
        )
        if moved.exit_code != 0:
            raise SandboxError(f'Failed to upload {path}: {moved.output[-500:]}')

    def delete(self) -> bool:
        """Delete the sandbox; errors are logged. Return False if Daytona reported a failure."""
        return cleanup_sandbox(self._sandbox, self._daytona, wait=self._session)
