"""Check pinned CLI flags locally and replay version-specific stream samples."""

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from deepgym._cli_worker import cli_version_supported, parse_cli_version
from deepgym.agents.base import Budget, BudgetTracker
from deepgym.agents.cli import ClaudeCodeAgent, CodexAgent
from deepgym.agents.cli_events import CliEvents
from deepgym.agents.versions import SUPPORTED_CLIS, cli_install_command


def test_install_command_uses_supported_releases() -> None:
    assert shlex.split(cli_install_command()) == [
        'npm',
        'install',
        '-g',
        '--prefix',
        '/usr/local',
        *(f'{release.package}@{release.version}' for release in SUPPORTED_CLIS.values()),
    ]


@pytest.mark.parametrize('provider', SUPPORTED_CLIS)
def test_installed_cli_supports_adapter_flags(provider: str, tmp_path: Path) -> None:
    release = SUPPORTED_CLIS[provider]
    executable = shutil.which(release.command)
    if executable is None:
        pytest.skip(f'{release.command} is not installed locally')

    config = tmp_path / release.command
    config.mkdir()
    env = dict(os.environ)
    env['CLAUDE_CONFIG_DIR' if provider == 'anthropic' else 'CODEX_HOME'] = str(config)
    version = subprocess.run(
        [executable, '--version'],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
        env=env,
    )
    assert cli_version_supported(parse_cli_version(version.stdout), release.version), (
        f'Unsupported local {release.command}: {version.stdout.strip()}. '
        f'Install the tested CLIs with `{cli_install_command()}`, or bump SUPPORTED_CLIS, '
        'record new stream fixtures, run the compat test, and rebuild images.'
    )
    agent = (ClaudeCodeAgent if provider == 'anthropic' else CodexAgent)(model='compat-only')
    budget = Budget(max_cost_usd=1 if provider == 'anthropic' else None)
    argv = agent._argv('compat-only', budget)
    subcommand = list(argv[1:2]) if provider == 'openai' else []
    help_result = subprocess.run(
        [executable, *subcommand, '--help'],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
        env=env,
    )
    flags = {arg for arg in argv[1 : argv.index('--')] if arg.startswith('--')}
    missing = sorted(
        flag
        for flag in flags
        if not re.search(rf'(?<![\w-]){re.escape(flag)}(?![\w-])', help_result.stdout)
    )
    assert not missing, f'{release.command} {release.version} help omits adapter flags: {missing}'


@pytest.mark.parametrize('provider', SUPPORTED_CLIS)
def test_supported_release_stream_fixture(provider: str) -> None:
    release = SUPPORTED_CLIS[provider]
    fixture = (
        Path(__file__).parent / 'fixtures' / release.command / release.version / 'stream.jsonl'
    )
    assert fixture.is_file(), (
        f'Missing compatibility fixture: {fixture}. Record the new release stream before '
        'accepting the version bump; see fixtures/README.md for provenance requirements.'
    )
    parser = CliEvents(provider, BudgetTracker(Budget()), None)
    for line in fixture.read_text().splitlines():
        assert parser.consume(json.loads(line)) is None
    assert parser.finish(0, None) == 'completed'
    assert parser.final_text
    assert parser.tracker.usage.output_tokens > 0
