"""Sandbox providers and the code that runs on them, against fakes (no provider needed).

Live runs on real providers are in test_sandboxes_live.py.
"""

import contextlib
import importlib
import subprocess
import sys
import types
from collections.abc import Mapping

import pytest

from deepgym import ComputerUseEnvironment, DeepGym
from deepgym.exceptions import DeepGymError, SandboxError, VerifierError
from deepgym.judge import JudgeJob
from deepgym.models import Action, Environment, MultiTurnEnvironment
from deepgym.sandbox import build_run_result, run_judge_in_sandbox
from deepgym.sandboxes import ExecResult, SandboxTimeoutError, open_sandbox
from deepgym.sandboxes.apple import AppleContainer
from deepgym.sandboxes.cli import DEFAULT_IMAGE
from deepgym.sandboxes.docker import DockerContainer
from deepgym.session import SandboxBackend


class FakeSandbox:
    """Records commands; answers from a script of (substring, result) pairs."""

    def __init__(self, network_blocked: bool, answers: list[tuple[str, str]]) -> None:
        self.id = 'fake'
        self.network_blocked = network_blocked
        self.answers = answers
        self.commands: list[str] = []
        self.uploads: list[str] = []
        self.deleted = False

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        self.commands.append(command)
        for needle, output in self.answers:
            if needle in command:
                return ExecResult(0, output)
        return ExecResult(0, '')

    def upload(self, content: bytes, path: str) -> None:
        self.uploads.append(path)

    def delete(self) -> None:
        self.deleted = True


class FakeDaytona:
    """Stands in for the Daytona client. Its sandboxes run commands as a sudoer, like
    Daytona's default snapshot, and answer from a script of (substring, result) pairs."""

    def __init__(self, answers: list[tuple[str, str]]) -> None:
        self.answers = answers
        self.boxes: list[types.SimpleNamespace] = []

    def create(self, params: object, timeout: float = 60) -> types.SimpleNamespace:
        events: list[str] = []
        uploads: dict[str, bytes] = {}

        def exec_(command: str, **kwargs: object) -> types.SimpleNamespace:
            events.append(command)
            output = next((out for needle, out in self.answers if needle in command), '')
            return types.SimpleNamespace(exit_code=0, result=output)

        mouse = types.SimpleNamespace(click=lambda *a, **k: events.append('click'))
        box = types.SimpleNamespace(
            id='fake-daytona',
            params=params,
            events=events,
            uploads=uploads,
            process=types.SimpleNamespace(exec=exec_),
            fs=types.SimpleNamespace(
                upload_file=lambda data, path: uploads.update({path: data}),
                create_folder=lambda path, mode: None,
            ),
            computer_use=types.SimpleNamespace(
                start=lambda **kwargs: events.append('computer_use.start'),
                mouse=mouse,
                screenshot=types.SimpleNamespace(
                    take_compressed=lambda *a, **k: types.SimpleNamespace(screenshot='png')
                ),
            ),
        )
        self.boxes.append(box)
        return box

    def delete(self, sandbox: object, timeout: float = 60, wait: bool = False) -> None:
        sandbox.events.append('deleted')


# ---------------------------------------------------------------------------
# Container CLIs
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b'out', b'')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    return calls


def test_docker_runs_offline_and_removes_with_rm(cli_calls: list[list[str]]) -> None:
    box = DockerContainer(image='img')
    box.exec('echo hi', cwd='/w', env={'A': '1'}, timeout=5)
    box.upload(b'x', '/p/f')
    box.delete()
    run, exec_, upload, delete = cli_calls
    assert run[:2] == ['docker', 'run'] and {'--network', '--pids-limit', '--memory'} <= set(run)
    assert exec_ == [
        'docker', 'exec', '--workdir', '/w', '--env', 'A=1', box.id,
        '/usr/bin/timeout', '--kill-after=2', '5', '/bin/sh', '-c', 'echo hi',
    ]  # fmt: skip
    assert upload[:3] == ['docker', 'exec', '--interactive']
    assert delete == ['docker', 'rm', '--force', box.id]
    assert box.network_blocked


def test_apple_adds_netns_caps_and_deletes(cli_calls: list[list[str]]) -> None:
    box = AppleContainer(image='img')
    box.delete()
    assert '--cap-add' in cli_calls[0] and 'CAP_SYS_ADMIN' in cli_calls[0]
    assert cli_calls[1] == ['container', 'delete', '--force', box.id]
    assert not box.network_blocked


def test_cli_boots_the_given_image_or_the_default(cli_calls: list[list[str]]) -> None:
    open_sandbox('docker', 'python:3.11-slim')
    open_sandbox('apple')
    assert cli_calls[0][-3:] == ['python:3.11-slim', 'sleep', 'infinity']
    assert cli_calls[1][-3:] == [DEFAULT_IMAGE, 'sleep', 'infinity']


def test_cli_timeout_exit_becomes_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, 124 if argv[1] == 'exec' else 0, b'', b'')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    box = DockerContainer(image='img')
    clock = iter([0.0, 10.0])
    monkeypatch.setattr('deepgym.sandboxes.cli.time.monotonic', lambda: next(clock))
    with pytest.raises(SandboxTimeoutError):
        box.exec('sleep 99', timeout=5)


def test_cli_start_timeout_removes_the_container(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(argv)
        if argv[1] == 'run':
            raise subprocess.TimeoutExpired(argv, 1)
        return subprocess.CompletedProcess(argv, 0, b'', b'')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    with pytest.raises(SandboxError, match='did not start'):
        DockerContainer(image='img')
    assert calls[-1][:3] == ['docker', 'rm', '--force']


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(DeepGymError, match='Unknown sandbox provider'):
        open_sandbox('nope')  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Modal and E2B, against stand-ins for their SDKs
# ---------------------------------------------------------------------------


def _install(monkeypatch: pytest.MonkeyPatch, name: str, attrs: dict) -> None:
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_modal_blocks_network_and_maps_results(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict = {}

    class Process:
        stdout = types.SimpleNamespace(read=lambda: 'out')
        stderr = types.SimpleNamespace(read=lambda: 'err')

        def wait(self) -> int:
            return 3

    class Sandbox:
        object_id = 'sb-1'
        filesystem = types.SimpleNamespace(write_bytes=lambda data, path: created.update(w=path))

        @staticmethod
        def create(*args: str, **kwargs: object) -> 'Sandbox':
            created.update(args=args, **kwargs)
            return Sandbox()

        def exec(self, *args: str, **kwargs: object) -> Process:
            created.setdefault('execs', []).append(args)
            return Process()

        def terminate(self) -> None:
            created['terminated'] = True

    _install(monkeypatch, 'modal', {
        'App': types.SimpleNamespace(lookup=lambda name, create_if_missing: name),
        'Image': types.SimpleNamespace(
            debian_slim=lambda python_version: 'image', from_registry=lambda tag: f'reg:{tag}'
        ),
        'Sandbox': Sandbox,
        'exception': types.SimpleNamespace(Error=Exception),
    })  # fmt: skip
    monkeypatch.delitem(sys.modules, 'deepgym.sandboxes.modal.sandbox', raising=False)
    module = importlib.import_module('deepgym.sandboxes.modal.sandbox')

    box = module.ModalSandbox()
    assert created['block_network'] is True and created['args'] == ('sleep', 'infinity')
    assert created['image'] == 'image'
    assert box.exec('false') == ExecResult(3, 'outerr')
    box.upload(b'x', '/a/b/c.txt')
    box.delete()
    assert created['execs'][-1][:2] == ('/bin/sh', '-c')
    assert 'mkdir -p /a/b' in created['execs'][-1][2]
    assert created['w'] == '/a/b/c.txt' and created['terminated']
    module.ModalSandbox('python:3.11-slim')
    assert created['image'] == 'reg:python:3.11-slim'


def test_e2b_runs_as_root_offline_and_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExitError(Exception):
        exit_code, stdout, stderr = 2, 'o', 'e'

    class SlowError(Exception):
        pass

    class GoneError(SlowError):
        pass

    seen: dict = {}

    class Commands:
        def run(self, cmd: str, **kwargs: object) -> object:
            seen.update(kwargs)
            if cmd == 'fail':
                raise ExitError()
            if cmd == 'slow':
                raise SlowError()
            if cmd == 'gone':
                raise GoneError()
            return types.SimpleNamespace(exit_code=0, stdout='ok', stderr='')

    class Client:
        sandbox_id = 'e2b-1'
        commands = Commands()

        @staticmethod
        def create(**kwargs: object) -> 'Client':
            seen['create'] = kwargs
            return Client()

    _install(monkeypatch, 'e2b', {
        'Sandbox': Client,
        'CommandExitException': ExitError,
        'TimeoutException': SlowError,
        'SandboxNotRunningException': GoneError,
        'SandboxUnreachableException': GoneError,
    })  # fmt: skip
    monkeypatch.delitem(sys.modules, 'deepgym.sandboxes.e2b.sandbox', raising=False)
    module = importlib.import_module('deepgym.sandboxes.e2b.sandbox')

    box = module.E2BSandbox()
    assert seen['create']['allow_internet_access'] is False
    assert box.exec('ok') == ExecResult(0, 'ok') and seen['user'] == 'root'
    assert box.exec('fail') == ExecResult(2, 'oe')
    with pytest.raises(SandboxTimeoutError):
        box.exec('slow', timeout=1)
    with pytest.raises(SandboxError, match='not reachable'):
        box.exec('gone', timeout=1)
    module.E2BSandbox('my-template')
    assert seen['create']['template'] == 'my-template'


# ---------------------------------------------------------------------------
# Judge and sessions on any provider
# ---------------------------------------------------------------------------

_JOB = JudgeJob.for_environment(Environment(task='t', verifier_code='return 1.0'), '')


def test_judge_fails_closed_when_it_cannot_go_offline() -> None:
    box = FakeSandbox(network_blocked=False, answers=[('useradd', '0 999 987 /usr/bin/python3 0')])
    with pytest.raises(SandboxError, match='network'):
        run_judge_in_sandbox(box, _JOB)


def test_judge_trusts_a_provider_that_blocks_the_network() -> None:
    box = FakeSandbox(
        network_blocked=True,
        answers=[
            ('useradd', '0 999 987 /usr/bin/python3 0'),
            ('DEEPGYM_DEADLINE', '{"score": 1.0, "passed": true}'),
        ],
    )
    assert run_judge_in_sandbox(box, _JOB).verifier_result.score == 1.0


# agent and verifier uid/gid; 0: no sudo, so no deepgym_solution for verifiers; the
# setpriv, nsenter and sudo paths; then the work directory as realpath prints it.
_USERS = (
    'useradd',
    '997 997 996 996 0 /usr/bin/setpriv /usr/bin/nsenter /usr/bin/sudo\n/tmp/deepgym',
)


def test_session_runs_as_agent_and_joins_netns_only_when_needed() -> None:
    blocked = FakeSandbox(network_blocked=True, answers=[_USERS])
    SandboxBackend(blocked).exec('ls', timeout=5)
    assert blocked.commands[-1].startswith('/usr/bin/setpriv --reuid=997 --regid=997')
    assert '--no-new-privs' in blocked.commands[-1]

    open_ = FakeSandbox(network_blocked=False, answers=[_USERS])
    SandboxBackend(open_).exec('ls', timeout=5)
    assert any('mount --bind /proc/$pid/ns/net /opt/deepgym-netns' in c for c in open_.commands)
    assert open_.commands[-1].startswith(
        '/usr/bin/nsenter --net=/opt/deepgym-netns -- /usr/bin/setpriv'
    )


def test_session_verifier_runs_as_its_own_user_from_a_private_file() -> None:
    box = FakeSandbox(network_blocked=True, answers=[_USERS])
    SandboxBackend(box).verify('print(1)', timeout=5, quiet=True)
    script = box.uploads[-1]
    assert script.startswith('/opt/deepgym-verifier/') and script.endswith('.py')
    assert box.commands[-1].startswith('/usr/bin/env -i ')
    assert '/usr/bin/setpriv --reuid=996 --regid=996' in box.commands[-1]
    assert f'python3 -I {script} 2>/dev/null' in box.commands[-1]
    assert 'print(1)' not in box.commands[-1]


def test_session_write_file_is_staged_then_written_as_agent() -> None:
    box = FakeSandbox(network_blocked=True, answers=[_USERS])
    SandboxBackend(box).write_file('a/b.py', b'x')
    staged = box.uploads[-1]
    assert staged.startswith('/opt/deepgym-staging/')
    assert box.commands[-1].startswith('/usr/bin/env -i ')
    assert '/usr/bin/setpriv --reuid=997' in box.commands[-1]
    assert f'< {staged}' in box.commands[-1] and '/tmp/deepgym/a/b.py' in box.commands[-1]


def test_session_closes_sandbox_when_setup_fails() -> None:
    box = FakeSandbox(network_blocked=True, answers=[])
    box.exec = lambda command, **kw: ExecResult(1, 'setpriv missing')  # type: ignore[method-assign]
    with pytest.raises(SandboxError, match='session users'):
        SandboxBackend(box)
    assert box.deleted


def test_async_daytona_judge_fails_closed_when_it_cannot_go_offline() -> None:
    import asyncio

    from deepgym.async_core import AsyncDeepGym

    class Process:
        async def exec(self, command: str, **kwargs: object) -> object:
            return types.SimpleNamespace(exit_code=0, result='0 999 987 /usr/bin/python3 0')

    class Files:
        async def upload_file(self, content: bytes, path: str) -> None:
            pass

    sandbox = types.SimpleNamespace(process=Process(), fs=Files())
    client = AsyncDeepGym(mode='local')
    with pytest.raises(SandboxError, match='network'):
        asyncio.run(client._run_judge(sandbox, _JOB))


def test_judge_killed_after_printing_a_pass_is_an_error() -> None:
    box = FakeSandbox(network_blocked=True, answers=[('useradd', '0 999 987 /usr/bin/python3 0')])
    answer = box.exec

    def exec_(command: str, **kwargs: object) -> ExecResult:
        if 'DEEPGYM_DEADLINE' in command:
            return ExecResult(137, '{"score": 1.0, "passed": true}')
        return answer(command, **kwargs)

    box.exec = exec_  # type: ignore[method-assign]
    executed = run_judge_in_sandbox(box, _JOB)
    with pytest.raises(VerifierError, match='137'):
        build_run_result(executed.verifier_result, 1.0, box.id, exit_code=executed.exit_code)


# ---------------------------------------------------------------------------
# Daytona sessions: the same users and network namespace as any other provider
# ---------------------------------------------------------------------------

# Daytona's default snapshot runs commands as a sudoer; this is its private upload staging dir.
_STAGE = ('mktemp -d', '/tmp/tmp.stage')
_DAYTONA_USERS = (
    'useradd',
    '997 997 996 996 1 /usr/bin/setpriv /usr/bin/nsenter /usr/bin/sudo\n/tmp/deepgym',
)


def _daytona_client(monkeypatch: pytest.MonkeyPatch) -> tuple[DeepGym, FakeDaytona]:
    pytest.importorskip('daytona')
    client = FakeDaytona([_STAGE, _DAYTONA_USERS])
    monkeypatch.setattr(DeepGym, '_init_daytona', staticmethod(lambda *args: client))
    return DeepGym(mode='daytona'), client


def test_daytona_session_runs_agent_and_verifier_as_separate_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dg, client = _daytona_client(monkeypatch)
    final = 'print(\'{"score": 0.0, "passed": false}\')  # SECRET'
    with dg.session(MultiTurnEnvironment(task='t', final_verifier_code=final)) as session:
        session.start()
        session.step(Action(action_type='bash', content='echo step-marker'))
        with contextlib.suppress(VerifierError):
            session.score()
    box = client.boxes[0]
    step = next(c for c in box.events if 'step-marker' in c)
    assert step.startswith('/usr/bin/sudo -n -E /bin/sh -c')
    assert '/usr/bin/nsenter --net=/opt/deepgym-netns -- /usr/bin/setpriv --reuid=997' in step
    assert not any('SECRET' in c for c in box.events), 'verifier code must not be in a command'
    score = next(c for c in box.events if '/opt/deepgym-verifier/' in c and 'python3' in c)
    assert '--reuid=996' in score
    staged = [path for path, data in box.uploads.items() if b'SECRET' in data]
    assert staged and all(path.startswith('/tmp/tmp.stage/') for path in staged)
    assert box.events[-1] == 'deleted'


def test_daytona_desktop_runs_as_the_agent_and_is_locked_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dg, client = _daytona_client(monkeypatch)
    with dg.session(ComputerUseEnvironment(task='t')) as session:
        session.start()
    events = client.boxes[0].events
    started = events.index('computer_use.start')
    before = ' '.join(events[:started])
    assert '/usr/local/bin/xfce4-session' in before, 'the desktop must not run as the sudoer'
    prepare = next(c for c in events[:started] if '/usr/local/bin/xfce4-session' in c)
    assert prepare.startswith('/usr/bin/sudo -n -E /bin/sh -c')
    assert '--reuid=997' in prepare and '/opt/deepgym-netns' in prepare
    for blocked in ('thunar-uca', 'Thunar/uca.xml', 'gio-launch-desktop', 'xfce4-terminal', 'idle'):
        assert blocked in prepare, f'{blocked} must be locked down before the desktop starts'
    for x11vnc_option in ('novncconnect', 'noremote', 'nocmds', 'safer'):
        assert x11vnc_option in prepare
