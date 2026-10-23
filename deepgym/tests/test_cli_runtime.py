"""Offline checks for CLI supervision, retained isolation and root service ordering."""

import base64
import errno
import json
import os
import socket
from types import SimpleNamespace
from urllib.request import proxy_bypass_environment

import pytest
from pydantic import ValidationError

from deepgym import _cli_worker, cli_runtime
from deepgym.cli_runtime import CliExecution, CliRequest
from deepgym.exceptions import ActionError, SandboxError, VerifierError
from deepgym.models import Action, MultiTurnEnvironment
from deepgym.sandboxes import ExecResult
from deepgym.session import LocalBackend, SandboxBackend, Session
from tests.test_sandboxes import _USERS, FakeSandbox


def request(**changes: object) -> CliRequest:
    return CliRequest(
        **{
            'argv': ('claude', '-p', 'task'),
            'env': {'ANTHROPIC_API_KEY': 'test-key'},
            'provider': 'anthropic',
            'timeout': 5,
            'allow_model_credentials': True,
            **changes,
        }
    )


def test_cli_refuses_unacknowledged_key_exposure() -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    with pytest.raises(SandboxError, match='model-readable'):
        backend.run_cli(request(allow_model_credentials=False), lambda event: None)


def test_cli_refuses_to_remove_namespace_for_provider_egress() -> None:
    backend = SandboxBackend(FakeSandbox(True, [_USERS]))
    with pytest.raises(SandboxError, match='retained isolated namespace'):
        backend.run_cli(request(), lambda event: None)


@pytest.mark.parametrize(
    'env',
    [
        {'ANTHROPIC_API_KEY': 'key', 'LD_PRELOAD': '/tmp/evil.so'},
        {'OPENAI_API_KEY': 'key'},
        {'ANTHROPIC_API_KEY': ''},
    ],
)
def test_only_selected_provider_credential_is_injected(env: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match='only a non-empty'):
        request(env=env)


@pytest.mark.parametrize(
    'address', ['127.0.0.1', '169.254.169.254', '10.0.0.1', '::1', '224.0.0.1']
)
def test_provider_dns_cannot_open_toolbox_or_metadata(monkeypatch, address: str) -> None:
    monkeypatch.setattr(
        socket,
        'getaddrinfo',
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 443))],
    )
    with pytest.raises(ValueError, match='non-public'):
        _cli_worker.public_addresses('api.anthropic.com')


def test_provider_connect_uses_the_validated_numeric_address(monkeypatch) -> None:
    monkeypatch.setattr(
        socket,
        'getaddrinfo',
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('1.1.1.1', 443))],
    )
    assert _cli_worker.public_addresses('api.anthropic.com') == [(socket.AF_INET, ('1.1.1.1', 443))]


@pytest.mark.parametrize('target', ['127.0.0.1:3000', 'api.openai.com:443', 'api.anthropic.com:80'])
def test_connect_rejects_every_other_destination(target: str) -> None:
    class Connection:
        def __init__(self) -> None:
            self.incoming = bytearray(f'CONNECT {target} HTTP/1.1\r\n\r\n'.encode())
            self.sent = b''

        def settimeout(self, timeout: float) -> None:
            pass

        def recv(self, amount: int) -> bytes:
            result = bytes(self.incoming[:amount])
            del self.incoming[:amount]
            return result

        def sendall(self, data: bytes) -> None:
            self.sent += data

    connection = Connection()
    _cli_worker._proxy_handler('api.anthropic.com')(connection, None, None)
    assert connection.sent.startswith(b'HTTP/1.1 403')


def test_cli_supervisor_retains_namespace_and_collects_events_once(monkeypatch) -> None:
    box = FakeSandbox(False, [_USERS])
    uploads = {}
    monkeypatch.setattr(box, 'upload', lambda content, path: uploads.update({path: content}))
    backend = SandboxBackend(box)
    packet = {
        'data': base64.b64encode(b'{"type":"assistant"}\nDONE\n').decode(),
        'status': {'exit_code': 0, 'stop_reason': None},
    }
    monkeypatch.setattr(cli_runtime, '_poll', lambda *args: packet)
    observed = []
    result = backend.run_cli(request(), lambda event: observed.append(event))
    config = json.loads(
        next(value for key, value in uploads.items() if key.endswith('request.json'))
    )
    assert config['namespace'] == ['/usr/bin/nsenter', '--net=/opt/deepgym-netns', '--']
    assert '--reuid=997' in config['prefix']
    assert '--no-new-privs' in config['prefix']
    assert config['prefix'].index('/usr/bin/unshare') < config['prefix'].index('--reuid=997')
    assert ['--pid', '--fork', '--mount-proc'] == config['prefix'][
        config['prefix'].index('/usr/bin/unshare') + 1 : config['prefix'].index('/usr/bin/unshare')
        + 4
    ]
    assert '--pid=/opt/deepgym-staging/agent-processes/pid' in config['prefix']
    assert config['env']['ANTHROPIC_API_KEY'] == 'test-key'
    assert observed == [{'type': 'assistant'}]
    assert result.events == ({'type': 'assistant'},)
    assert result.stop_reason is None
    assert 'DONE' in result.output
    assert all('test-key' not in command for command in box.commands)
    assert any('rm -rf' in command for command in box.commands)
    assert any('control.py stop' in command for command in box.commands)


def test_callback_budget_stop_reaches_root_supervisor(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    packets = iter(
        [
            {'data': base64.b64encode(b'{"type":"assistant"}\n').decode(), 'status': None},
            {'data': '', 'status': {'exit_code': -9, 'stop_reason': 'turn_budget'}},
        ]
    )
    monkeypatch.setattr(cli_runtime, '_poll', lambda *args: next(packets))
    monkeypatch.setattr(cli_runtime.time, 'sleep', lambda seconds: None)
    result = backend.run_cli(request(), lambda event: 'turn_budget')
    assert result.stop_reason == 'turn_budget'
    assert any('/stop' in command for command in backend._sandbox.commands)


def test_launch_then_rpc_failure_independently_destroys_containment(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    operations = []
    original = backend.exec_trusted

    def execute(command: str, timeout: int) -> ExecResult:
        if 'supervise' in command and 'setsid' in command:
            operations.append('launch-took-effect')
            raise SandboxError('launch RPC response lost')
        if 'control.py stop' in command:
            operations.append('kernel-teardown')
        if '/stop' in command and 'write_text' in command:
            operations.append('stop-marker')
        if 'rm -rf' in command:
            operations.append('remove-credentials')
        return original(command, timeout)

    monkeypatch.setattr(backend, 'exec_trusted', execute)
    monkeypatch.setattr(
        cli_runtime,
        '_poll',
        lambda *args: {'data': '', 'status': {'exit_code': -9, 'stop_reason': 'cancelled'}},
    )
    with pytest.raises(SandboxError, match='launch RPC response lost'):
        backend.run_cli(request(), lambda event: None)
    assert operations == [
        'launch-took-effect',
        'kernel-teardown',
        'stop-marker',
        'remove-credentials',
    ]


@pytest.mark.parametrize('failed_file', ['worker.py', 'request.json'])
def test_upload_failure_still_removes_staged_credentials(monkeypatch, failed_file: str) -> None:
    box = FakeSandbox(False, [_USERS])
    backend = SandboxBackend(box)
    uploaded = []

    def upload(content: bytes, path: str) -> None:
        uploaded.append(path)
        if path.endswith(failed_file):
            raise SandboxError('upload took effect, response lost')

    monkeypatch.setattr(box, 'upload', upload)
    with pytest.raises(SandboxError, match='upload took effect'):
        backend.run_cli(request(), lambda event: None)
    assert uploaded
    assert any('rm -rf -- /opt/deepgym-staging/cli-' in command for command in box.commands)
    assert not any('supervise' in command for command in box.commands)


def test_containment_cleanup_failure_refuses_grading(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    monkeypatch.setattr(
        cli_runtime,
        '_poll',
        lambda *args: {'data': '', 'status': {'exit_code': 0, 'stop_reason': None}},
    )

    def stop() -> None:
        raise SandboxError('namespace survived')

    monkeypatch.setattr(backend, 'stop_agent', stop)
    monkeypatch.setattr(backend, 'verify', lambda *args, **kwargs: pytest.fail('unsafe grading'))
    session = Session(
        MultiTurnEnvironment(task='task', final_verifier_code='print(1)', cli_provider_egress=True),
        backend,
    )
    session.start()
    with pytest.raises(SandboxError, match='containment cleanup failed'):
        session.run_cli(request(), lambda event: None)
    assert session.failed_result is not None
    assert session.score() == session.failed_result
    assert session.score().score == 0 and not session.score().passed
    assert any('rm -rf' in command for command in backend._sandbox.commands)


def test_setup_and_desktop_share_containment_but_root_services_do_not() -> None:
    box = FakeSandbox(False, [_USERS])
    backend = SandboxBackend(box)
    assert '--pid=/opt/deepgym-staging/agent-processes/pid' in backend._as_agent
    backend.exec('setup-command', 10)
    assert '--pid=/opt/deepgym-staging/agent-processes/pid' in box.commands[-1]
    backend.exec_root('trusted-service', 10, offline=True, env={})
    assert '--pid=' not in box.commands[-1]


def test_staging_timeout_stops_setup_descendants_before_step_grading(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    now = [0.0]
    operations = []
    monkeypatch.setattr(cli_runtime.time, 'monotonic', lambda: now[0])

    def upload(content: bytes, path: str) -> None:
        if path.endswith('worker.py'):
            now[0] = 2.0

    def grade(output: str) -> float:
        assert operations == ['kernel-teardown']
        operations.append('step-grade')
        return 0.0

    monkeypatch.setattr(backend._sandbox, 'upload', upload)
    monkeypatch.setattr(backend, 'stop_agent', lambda: operations.append('kernel-teardown'))
    session = Session(
        MultiTurnEnvironment(task='task', step_verifier_code='print(0)', cli_provider_egress=True),
        backend,
    )
    monkeypatch.setattr(session, '_step_reward', grade)
    session.start()
    result = session.run_cli(request(timeout=1), lambda event: None)
    assert result.stop_reason == 'time_budget'
    assert operations == ['kernel-teardown', 'step-grade']
    assert not any('supervise' in command for command in backend._sandbox.commands)


def test_local_backend_refuses_root_service_before_running_anything() -> None:
    backend = LocalBackend()
    try:
        session = Session(MultiTurnEnvironment(task='task', root_service_start='true'), backend)
        with pytest.raises(SandboxError, match='isolated SandboxBackend'):
            session.start()
    finally:
        backend.close()


def test_root_service_runs_before_desktop_and_setup(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    calls = []
    monkeypatch.setattr(
        backend,
        'exec_root',
        lambda command, timeout, **kwargs: calls.append(('root', kwargs)) or ExecResult(0, 'ready'),
    )
    monkeypatch.setattr(backend, 'start_desktop', lambda: calls.append(('desktop', {})))
    monkeypatch.setattr(
        backend, 'exec', lambda *args: calls.append(('setup', {})) or ExecResult(0, '')
    )
    monkeypatch.setattr(backend, 'screenshot', lambda: 'png')
    session = Session(
        MultiTurnEnvironment(
            task='task',
            root_service_start='bash /opt/probe/start.sh',
            desktop=True,
            setup_script='browser',
        ),
        backend,
    )
    session.start()
    assert [name for name, _ in calls] == ['root', 'desktop', 'setup']
    assert calls[0][1]['offline'] is True
    assert set(calls[0][1]['env']) == {'PATH'}


def test_cli_is_one_action_and_never_replays_commands(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    event = {'type': 'tool', 'command': 'touch already-ran'}
    execution = CliExecution(events=(event,), output='DONE\n', exit_code=0)
    monkeypatch.setattr(backend, 'run_cli', lambda request, observer: execution)
    session = Session(MultiTurnEnvironment(task='task', cli_provider_egress=True), backend)
    session.start()
    count = len(backend._sandbox.commands)
    assert session.run_cli(request(), lambda event: None) == execution
    assert session.done
    assert len(session.trajectory.steps) == 1
    action = session.trajectory.steps[0][1]
    assert action.action_type == 'cli'
    assert action.events == (event,)
    assert len(backend._sandbox.commands) == count
    assert 'test-key' not in action.model_dump_json()


def test_cli_action_cannot_be_replayed_through_step() -> None:
    backend = LocalBackend()
    try:
        session = Session(MultiTurnEnvironment(task='task', allowed_actions={'cli'}), backend)
        session.start()
        with pytest.raises(ActionError, match='Session.run_cli'):
            session.step(Action(action_type='cli', content='touch unsafe'))
    finally:
        backend.close()


def test_events_and_output_redact_known_key_before_observer(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    raw = b'{"type":"assistant","nested":[{"text":"test-key"}],"tokens":12}\nkey=test-key\n'
    packet = {
        'data': base64.b64encode(raw).decode(),
        'status': {'exit_code': 0, 'stop_reason': None},
    }
    monkeypatch.setattr(cli_runtime, '_poll', lambda *args: packet)
    observed = []
    result = backend.run_cli(request(), lambda event: observed.append(event))
    assert 'test-key' not in result.model_dump_json()
    assert observed[0]['nested'] == [{'text': '[REDACTED]'}]
    assert observed[0]['tokens'] == 12


def test_cli_failure_keeps_already_observed_events(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    event = {'type': 'assistant', 'text': 'partial'}

    def fail(request, observer):
        observer(event)
        raise SandboxError('transport lost')

    monkeypatch.setattr(backend, 'run_cli', fail)
    session = Session(MultiTurnEnvironment(task='task', cli_provider_egress=True), backend)
    session.start()
    with pytest.raises(SandboxError, match='transport lost'):
        session.run_cli(request(), lambda event: None)
    assert session.trajectory.steps[0][1].events == (event,)
    assert session.done


def test_step_budget_restricts_command_timeout(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    timeouts = []
    monkeypatch.setattr(
        backend, 'exec', lambda command, timeout: timeouts.append(timeout) or ExecResult(0, '')
    )
    session = Session(MultiTurnEnvironment(task='task', timeout_per_step=30), backend)
    session.start()
    session.step(Action(action_type='bash', content='true'), timeout=0.25)
    session.step(Action(action_type='bash', content='true'), timeout=100)
    assert 0 < timeouts[0] <= 0.25
    assert timeouts[1] == 30


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_step_rejects_invalid_budget_timeout(timeout: float) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    session = Session(MultiTurnEnvironment(task='task'), backend)
    session.start()
    with pytest.raises(ValueError, match='finite and positive'):
        session.step(Action(action_type='bash', content='true'), timeout=timeout)


def test_observer_accounts_for_trailing_events_after_budget_stop(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    raw = b'{"type":"tool"}\n{"type":"result","usage":{"tokens":123}}\n'
    packet = {
        'data': base64.b64encode(raw).decode(),
        'status': {'exit_code': 0, 'stop_reason': None},
    }
    monkeypatch.setattr(cli_runtime, '_poll', lambda *args: packet)
    observed = []

    def observer(event: dict) -> str:
        observed.append(event)
        return 'turn_budget'

    result = backend.run_cli(request(), observer)
    assert observed[-1]['usage'] == {'tokens': 123}
    assert result.stop_reason == 'turn_budget'
    assert sum('/stop' in command for command in backend._sandbox.commands) == 1


def test_gui_action_and_screenshot_share_remaining_deadline(monkeypatch) -> None:
    backend = SandboxBackend(FakeSandbox(False, [_USERS]))
    monkeypatch.setattr(backend, 'start_desktop', lambda: None)
    now = [100.0]
    timeouts = []

    def act(action: Action, *, timeout: float) -> None:
        timeouts.append(('action', timeout))
        now[0] += 0.6

    def screenshot(*, timeout: float | None = None) -> str:
        if timeout is not None:
            timeouts.append(('screenshot', timeout))
        return 'png'

    monkeypatch.setattr(backend, 'act', act)
    monkeypatch.setattr(backend, 'screenshot', screenshot)
    monkeypatch.setattr('deepgym.session.time.monotonic', lambda: now[0])
    session = Session(
        MultiTurnEnvironment(
            task='task',
            desktop=True,
            allowed_actions={'click'},
            timeout_per_step=120,
        ),
        backend,
    )
    session.start()
    session.step(Action(action_type='click', x=1, y=2), timeout=1.0)
    assert timeouts[0] == ('action', 1.0)
    assert timeouts[1][0] == 'screenshot'
    assert timeouts[1][1] == pytest.approx(0.4)


@pytest.mark.parametrize('bridge_ready', [False, True])
@pytest.mark.parametrize('stop_reason', ['time_budget', 'turn_budget'])
def test_worker_does_not_launch_cli_after_setup_exhausts_budget(
    monkeypatch, tmp_path, bridge_ready: bool, stop_reason: str
) -> None:
    config = {
        'provider': 'anthropic',
        'timeout': 1,
        'uid': 997,
        'namespace': ['nsenter'],
        'prefix': ['setpriv'],
        'argv': ['claude'],
        'env': {},
        'cwd': str(tmp_path),
    }
    (tmp_path / 'request.json').write_text(json.dumps(config))
    now = [0.0]
    calls = []

    def popen(argv, **kwargs):
        calls.append(argv)
        if bridge_ready:
            (tmp_path / 'port').write_text('12345')
        if stop_reason == 'time_budget':
            now[0] = 2.0
        else:
            (tmp_path / 'stop').write_text(stop_reason)
        return SimpleNamespace(pid=999, poll=lambda: None, wait=lambda **kwargs: -9)

    server = SimpleNamespace(
        serve_forever=lambda: None, shutdown=lambda: None, server_close=lambda: None
    )
    monkeypatch.setattr(_cli_worker.socketserver, 'ThreadingUnixStreamServer', lambda *args: server)
    monkeypatch.setattr(
        _cli_worker.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda: None)
    )
    monkeypatch.setattr(_cli_worker.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(_cli_worker.subprocess, 'Popen', popen)
    monkeypatch.setattr(_cli_worker.os, 'pidfd_open', lambda pid: 123, raising=False)
    monkeypatch.setattr(_cli_worker.signal, 'pidfd_send_signal', lambda *args: None, raising=False)
    monkeypatch.setattr(_cli_worker.os, 'close', lambda fd: None)
    monkeypatch.setattr(_cli_worker.os, 'killpg', lambda *args: None)
    _cli_worker._supervise(tmp_path)
    assert len(calls) == 1
    assert 'bridge' in calls[0]
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['stop_reason'] == stop_reason
    assert status['exit_code'] == -1


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
def test_worker_passes_loopback_only_bypass_to_cli_and_child_environment(
    monkeypatch, tmp_path, provider: str
) -> None:
    config = {
        'provider': provider,
        'timeout': 10,
        'namespace': ['nsenter'],
        'prefix': ['setpriv'],
        'argv': ['fixture-cli'],
        'env': {'NO_PROXY': '*', 'no_proxy': '*'},
        'cwd': str(tmp_path),
    }
    (tmp_path / 'request.json').write_text(json.dumps(config))
    captured = {}

    def popen(argv, **kwargs):
        if 'bridge' in argv:
            (tmp_path / 'port').write_text('12345')
        else:
            captured.update(kwargs['env'])
        return SimpleNamespace(pid=999, poll=lambda: 0, wait=lambda **kwargs: 0)

    server = SimpleNamespace(
        serve_forever=lambda: None, shutdown=lambda: None, server_close=lambda: None
    )
    monkeypatch.setattr(_cli_worker.socketserver, 'ThreadingUnixStreamServer', lambda *args: server)
    monkeypatch.setattr(
        _cli_worker.threading, 'Thread', lambda **kwargs: SimpleNamespace(start=lambda: None)
    )
    monkeypatch.setattr(_cli_worker.subprocess, 'Popen', popen)
    monkeypatch.setattr(_cli_worker.os, 'pidfd_open', lambda pid: 123, raising=False)
    monkeypatch.setattr(_cli_worker.signal, 'pidfd_send_signal', lambda *args: None, raising=False)
    monkeypatch.setattr(_cli_worker.os, 'close', lambda fd: None)
    _cli_worker._supervise(tmp_path)

    assert captured['NO_PROXY'] == captured['no_proxy']
    bypass = {'no': captured['no_proxy']}
    for host in (
        'localhost:8000',
        'localhost.:8000',
        '127.0.0.1:8000',
        '::1',
        '[::1]:8000',
        '0:0:0:0:0:0:0:1',
        '[0:0:0:0:0:0:0:1]:8000',
    ):
        assert proxy_bypass_environment(host, bypass), host
    for host in (
        'api.anthropic.com:443',
        'api.openai.com:443',
        'example.com:80',
        'localhost.example.com:8000',
        '10.0.0.1:8000',
        '169.254.169.254:80',
        '[2001:4860:4860::8888]:443',
        '127.0.0.2:8000',
        '[::ffff:127.0.0.1]:8000',
    ):
        assert not proxy_bypass_environment(host, bypass), host
    for variable in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        assert captured[variable] == 'http://127.0.0.1:12345'


@pytest.mark.parametrize('acted', [False, True])
def test_a_verifier_upload_the_agent_may_have_blocked(monkeypatch, acted: bool) -> None:
    """The verifier's folder is on the disk the agent shares: once it acted, a failed upload
    (the disk it filled) is its judged 0, like any write; before, no verdict."""
    box = FakeSandbox(False, [_USERS])
    backend = SandboxBackend(box)

    def disk_full(content: bytes, path: str) -> None:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    session = Session(MultiTurnEnvironment(task='task', final_verifier_code='print(1)'), backend)
    session.start()
    if acted:
        session.step(Action(action_type='bash', content='true'))
    monkeypatch.setattr(box, 'upload', disk_full)

    if not acted:
        with pytest.raises(VerifierError, match='Could not upload the verifier'):
            session.score()
        return
    result = session.score()
    assert (result.has_verdict, result.error_type) == (True, 'policy_sandbox_failure')
