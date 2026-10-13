"""Sandbox providers: one folder per provider, each implementing :class:`Sandbox`.

``open_sandbox('docker', image)`` starts a fresh sandbox by provider name, from the
environment's image or the provider's default. Daytona is opened by
:class:`~deepgym.core.DeepGym` itself because it needs a client and the environment's
snapshot. ``build_image`` builds a Dockerfile for the container CLIs (apple, docker).
Provider SDKs are optional extras, imported only when that provider is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from deepgym.exceptions import DeepGymError
from deepgym.sandboxes.protocol import TRUSTED_PATH, ExecResult, Sandbox, SandboxTimeoutError

Provider = Literal['apple', 'docker', 'modal', 'e2b']
PROVIDERS: tuple[Provider, ...] = ('apple', 'docker', 'modal', 'e2b')

__all__ = [
    'PROVIDERS',
    'ExecResult',
    'Provider',
    'Sandbox',
    'SandboxTimeoutError',
    'TRUSTED_PATH',
    'build_image',
    'check_provider',
    'open_sandbox',
]


def open_sandbox(provider: Provider, image: str | None = None) -> Sandbox:
    """Start a fresh sandbox from *provider*.

    Args:
        provider: Which provider to use.
        image: :attr:`~deepgym.models.Environment.image`, or None for the provider's default.

    Raises:
        SandboxError: If the sandbox can't be started.
    """
    match provider:
        case 'apple':
            from deepgym.sandboxes.apple import AppleContainer

            return AppleContainer(image)
        case 'docker':
            from deepgym.sandboxes.docker import DockerContainer

            return DockerContainer(image)
        case 'modal':
            from deepgym.sandboxes.modal import ModalSandbox

            return ModalSandbox(image)
        case 'e2b':
            from deepgym.sandboxes.e2b import E2BSandbox

            return E2BSandbox(image)
    raise DeepGymError(f'Unknown sandbox provider {provider!r}; choose from {PROVIDERS}')


def build_image(backend: str, context: Path) -> str:
    """Build the Dockerfile in *context* for *backend*; return the image to boot.

    Args:
        backend: A client's backend (``DeepGym.backend``): a provider, 'daytona' or 'local'.
        context: The build context, holding the Dockerfile.

    Raises:
        DeepGymError: If *backend* boots registry images only (everything but apple and
            docker), or the build fails.
    """
    if backend not in ('apple', 'docker'):
        raise DeepGymError(
            f'{backend} sandboxes boot registry images only; give the task a prebuilt image'
        )
    from deepgym.sandboxes.apple import AppleContainer
    from deepgym.sandboxes.docker import DockerContainer

    return (AppleContainer if backend == 'apple' else DockerContainer).build(context)


def check_provider(provider: Provider) -> None:
    """Fail fast if *provider* can't be used on this machine.

    Raises:
        DeepGymError: If its CLI isn't running or its SDK isn't installed.
    """
    if provider in ('apple', 'docker'):
        from deepgym.sandboxes.apple import AppleContainer
        from deepgym.sandboxes.docker import DockerContainer

        cls = AppleContainer if provider == 'apple' else DockerContainer
        if not cls.available():
            hint = '`container system start`' if provider == 'apple' else 'start the Docker daemon'
            raise DeepGymError(f"mode='{provider}' needs the {cls.cli} CLI running ({hint})")
        return
    try:
        __import__(f'deepgym.sandboxes.{provider}')
    except ImportError as exc:
        raise DeepGymError(
            f"mode='{provider}' needs its SDK: pip install 'deepgym[{provider}]'"
        ) from exc
