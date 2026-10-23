"""Check version admission without running a sandbox or an inference command."""

import json
import subprocess
from types import SimpleNamespace
from typing import cast

import pytest

from deepgym import _cli_worker
from deepgym.agents import SUPPORTED_CLIS, Budget, ClaudeCodeAgent
from deepgym.cli_runtime import CliExecution, CliExecutionError
from deepgym.models import Observation
from deepgym.session import Session
from tests.agents.test_cli import RecordingSession


@pytest.mark.parametrize('provider', SUPPORTED_CLIS)
def test_version_probe_is_unprivileged_keyless_and_bounded(monkeypatch, provider: str) -> None:
    release = SUPPORTED_CLIS[provider]
    captured = {}

    def run(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return SimpleNamespace(stdout=f'{release.command} {release.version}', returncode=0)

    monkeypatch.setattr(_cli_worker.subprocess, 'run', run)
    monkeypatch.setattr(_cli_worker.time, 'monotonic', lambda: 100)
    status = {}
    config = {
        'prefix': ['nsenter', '--net=isolated', '--', 'setpriv', '--reuid=997', '--'],
        'argv': [release.command, '--model', 'unused'],
        'cwd': '/workspace',
        'version_env': {'PATH': '/usr/bin'},
        'env': {'ANTHROPIC_API_KEY': 'must-not-reach-probe'},
        'supported_version': release.version,
    }
    _cli_worker._check_version(config, status, 102)
    assert captured['argv'] == [*config['prefix'], release.command, '--version']
    assert captured['env'] == config['version_env']
    assert captured['timeout'] == 2
    assert status == {'cli_version': release.version}


@pytest.mark.parametrize('output', ['unversioned', '1.2.3 4.5.6', '1.2.3-beta', 'v1.2.3'])
def test_malformed_or_ambiguous_versions_fail_closed(output: str) -> None:
    assert _cli_worker.parse_cli_version(output) is None


@pytest.mark.parametrize('failure', [FileNotFoundError(), subprocess.TimeoutExpired('cli', 1)])
def test_unavailable_version_has_launch_diagnostic(monkeypatch, failure: Exception) -> None:
    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(_cli_worker.subprocess, 'run', run)
    monkeypatch.setattr(_cli_worker.time, 'monotonic', lambda: 0)
    release = SUPPORTED_CLIS['anthropic']
    status = {}
    _cli_worker._check_version(
        {
            'prefix': [],
            'argv': [release.command],
            'cwd': '/',
            'version_env': {},
            'supported_version': release.version,
        },
        status,
        10,
    )
    assert status['cli_version'] is None
    assert 'Cannot start' in status['version_error']
    assert 'Unsupported' not in status['version_error']
    assert 'cli_install_command()' in status['version_error']
    assert '/usr/local/bin/node' in status['version_error']


def test_supervisor_rejects_wrong_version_before_egress_or_inference(monkeypatch, tmp_path) -> None:
    release = SUPPORTED_CLIS['anthropic']
    config = {
        'provider': 'anthropic',
        'timeout': 30,
        'uid': 997,
        'prefix': ['setpriv'],
        'argv': [release.command],
        'cwd': str(tmp_path),
        'version_env': {},
        'supported_version': release.version,
    }
    (tmp_path / 'request.json').write_text(json.dumps(config))
    server = SimpleNamespace(
        serve_forever=lambda: None, shutdown=lambda: None, server_close=lambda: None
    )
    monkeypatch.setattr(_cli_worker.socketserver, 'ThreadingUnixStreamServer', lambda *args: server)
    monkeypatch.setattr(
        _cli_worker.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda: None)
    )
    monkeypatch.setattr(_cli_worker.os, 'pidfd_open', lambda pid: 123, raising=False)
    monkeypatch.setattr(_cli_worker.signal, 'pidfd_send_signal', lambda *args: None, raising=False)
    monkeypatch.setattr(_cli_worker.os, 'close', lambda fd: None)
    monkeypatch.setattr(
        _cli_worker.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(stdout='9.9.9', returncode=0),
    )
    monkeypatch.setattr(
        _cli_worker.subprocess,
        'Popen',
        lambda *args, **kwargs: pytest.fail('Version mismatch must not launch bridge/inference'),
    )
    _cli_worker._supervise(tmp_path)
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['cli_version'] == '9.9.9'
    assert 'Unsupported' in status['version_error']


@pytest.mark.parametrize('exit_code', [126, 127])
def test_missing_binary_or_node_is_not_reported_as_version_mismatch(monkeypatch, exit_code) -> None:
    monkeypatch.setattr(
        _cli_worker.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(stdout='', returncode=exit_code),
    )
    monkeypatch.setattr(_cli_worker.time, 'monotonic', lambda: 0)
    status = {}
    release = SUPPORTED_CLIS['anthropic']
    _cli_worker._check_version(
        {
            'prefix': [],
            'argv': [release.command],
            'cwd': '/',
            'version_env': {},
            'supported_version': release.version,
        },
        status,
        10,
    )
    assert 'executable or interpreter' in status['version_error']
    assert 'agent PATH' in status['version_error']
    assert 'Unsupported' not in status['version_error']


def test_rejected_version_reaches_result_with_zero_spend() -> None:
    class UnsupportedSession(RecordingSession):
        def run_cli(self, request, consume):
            return CliExecution(
                events=(),
                output='',
                exit_code=-1,
                cli_version='9.9.9',
                version_error='Unsupported version; bump SUPPORTED_CLIS and rebuild images',
            )

    session = UnsupportedSession('claude')
    result = ClaudeCodeAgent(model='m', api_key='fixture', allow_model_credentials=True).run(
        cast(Session, session),
        Observation(content='', step=0),
        task='task',
        budget=Budget(),
    )
    # The model never started: nothing the policy did, so the episode gets no verdict.
    assert result.termination_reason == 'setup_error'
    assert [kind for kind, _ in session.abandoned] == ['setup_error']
    assert result.cli_version == '9.9.9'
    assert 'rebuild images' in result.error
    assert result.cost.usd == 0


def test_later_supervision_failure_keeps_observed_version() -> None:
    version = SUPPORTED_CLIS['anthropic'].version

    class FailedSession(RecordingSession):
        def run_cli(self, request, consume):
            raise CliExecutionError('supervision failed', version)

    result = ClaudeCodeAgent(model='m', api_key='fixture', allow_model_credentials=True).run(
        cast(Session, FailedSession('claude')),
        Observation(content='', step=0),
        task='task',
        budget=Budget(),
    )
    assert result.termination_reason == 'error'
    assert result.cli_version == version


def test_version_probe_deadline_preserves_budget_reason() -> None:
    class ExpiredSession(RecordingSession):
        def run_cli(self, request, consume):
            return CliExecution(
                events=(),
                output='',
                exit_code=-1,
                stop_reason='time_budget',
                version_error='Wall budget exhausted before CLI version check',
            )

    result = ClaudeCodeAgent(model='m', api_key='fixture', allow_model_credentials=True).run(
        cast(Session, ExpiredSession('claude')),
        Observation(content='', step=0),
        task='task',
        budget=Budget(),
    )
    assert result.termination_reason == 'time_budget'


@pytest.mark.parametrize(
    ('version', 'tested', 'supported'),
    [
        ('2.1.302', '2.1.302', True),
        ('2.1.309', '2.1.302', True),  # a later release in the major line
        ('2.4.0', '2.1.302', True),
        ('2.1.301', '2.1.302', False),  # older than tested
        ('3.0.0', '2.1.302', False),  # a new major may change flags and events
        ('0.162.4', '0.162.1', True),
        ('0.163.0', '0.162.1', False),  # below 1.0 a minor release may break them
        (None, '2.1.302', False),
    ],
)
def test_supported_cli_versions(version, tested, supported) -> None:
    assert _cli_worker.cli_version_supported(version, tested) is supported
