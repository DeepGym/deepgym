"""Pin supported CLI releases in one place for adapters and image builders."""

import shlex
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict


class CliVersion(BaseModel):
    """Describe one tested release; the supported range is this exact version."""

    model_config = ConfigDict(frozen=True)
    command: str
    package: str
    version: str


SUPPORTED_CLIS: Mapping[Literal['anthropic', 'openai'], CliVersion] = MappingProxyType(
    {
        'anthropic': CliVersion(
            command='claude', package='@anthropic-ai/claude-code', version='2.1.302'
        ),
        'openai': CliVersion(command='codex', package='@openai/codex', version='0.162.1'),
    }
)


def cli_install_command() -> str:
    """Return the pinned npm installation command for sandbox image builders.

    It installs under /usr/local, so the binaries land on the standard PATH the sandbox
    agent gets even when the image's Node comes from nvm or another custom prefix.

    Returns:
        A shell command installing both supported CLI releases; nothing is executed.
    """
    return shlex.join(
        [
            'npm',
            'install',
            '-g',
            '--prefix',
            '/usr/local',
            *(f'{release.package}@{release.version}' for release in SUPPORTED_CLIS.values()),
        ]
    )
