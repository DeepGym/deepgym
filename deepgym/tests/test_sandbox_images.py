"""An environment's ``image`` reaches the sandbox it runs in, against fakes.

Provider constructors are covered in test_sandboxes.py and live runs in
test_sandboxes_live.py.
"""

import contextlib

import pytest

from deepgym import DeepGym
from deepgym.exceptions import SandboxError, VerifierError
from deepgym.models import Environment, MultiTurnEnvironment
from tests.test_sandboxes import FakeSandbox, _daytona_client

_ENV = Environment(task='t', verifier_code='return 1.0', image='python:3.11-slim')


def test_client_opens_runs_and_sessions_from_the_envs_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[tuple[str, str | None]] = []

    def fake_open(provider: str, image: str | None = None) -> FakeSandbox:
        opened.append((provider, image))
        return FakeSandbox(
            network_blocked=True,
            answers=[
                (
                    'realpath',
                    '997 997 996 996 0 - - /usr/bin/setpriv /usr/bin/nsenter -\n/tmp/deepgym',
                ),
                ('useradd', '0 999 987 /usr/bin/python3 0'),
                ('DEEPGYM_DEADLINE', '{"score": 1.0, "passed": true}'),
            ],
        )

    monkeypatch.setattr('deepgym.core.check_provider', lambda provider: None)
    monkeypatch.setattr('deepgym.core.open_sandbox', fake_open)
    dg = DeepGym(mode='docker')

    assert dg.run(_ENV, '').score == 1.0
    with dg.session(MultiTurnEnvironment(task='t', image='debian:12')):
        pass

    assert opened == [('docker', 'python:3.11-slim'), ('docker', 'debian:12')]


def test_daytona_judge_boots_the_image_unless_a_snapshot_is_set() -> None:
    daytona = pytest.importorskip('daytona')
    from deepgym.sandboxes.daytona import create_params

    params = create_params(_ENV)
    snapshot = create_params(_ENV.model_copy(update={'snapshot': 'snap'}))

    assert isinstance(params, daytona.CreateSandboxFromImageParams)
    assert params.image.dockerfile() == 'FROM python:3.11-slim\n'
    assert isinstance(snapshot, daytona.CreateSandboxFromSnapshotParams)
    assert snapshot.snapshot == 'snap'


def test_daytona_session_boots_the_image_offline() -> None:
    daytona = pytest.importorskip('daytona')
    from deepgym.sandboxes.daytona import DaytonaSandbox

    seen: list[object] = []

    class Client:
        def create(self, params: object, timeout: float) -> None:
            seen.append(params)
            raise RuntimeError('stop after create')

        def list(self, query: object) -> list[object]:
            return []

    with pytest.raises(SandboxError, match='stop after create'):
        DaytonaSandbox(Client(), MultiTurnEnvironment(task='t', image='python:3.11-slim'))

    params = seen[0]
    assert isinstance(params, daytona.CreateSandboxFromImageParams)
    assert params.image.dockerfile() == 'FROM python:3.11-slim\n'
    assert params.network_block_all is True
    assert params.labels  # so a create that fails after allocating can be found and deleted


@pytest.mark.parametrize('runtime', [{}, {'snapshot': 'cli-snapshot'}, {'image': 'debian:12'}])
@pytest.mark.parametrize('egress', [False, True])
def test_daytona_session_network_requires_explicit_cli_opt_in(runtime, egress: bool) -> None:
    pytest.importorskip('daytona')
    from deepgym.sandboxes.daytona.sandbox import DaytonaSandbox, _session_params

    environment = MultiTurnEnvironment(task='task', cli_provider_egress=egress, **runtime)
    params = _session_params(environment)
    assert params.network_block_all is (not egress)
    assert params.ephemeral
    # Provider-level egress must never remove the namespace shielding the toolbox.
    assert DaytonaSandbox.network_blocked is False


def test_cli_provider_egress_is_default_off_and_not_for_gui_only() -> None:
    from pydantic import ValidationError

    from deepgym.computer_use import ComputerUseEnvironment

    assert MultiTurnEnvironment(task='task').cli_provider_egress is False
    with pytest.raises(ValidationError, match='requires bash or cli'):
        ComputerUseEnvironment(task='task', cli_provider_egress=True)


def test_cli_egress_keeps_agent_root_services_and_verifier_offline(monkeypatch) -> None:
    client, daytona = _daytona_client(monkeypatch)
    environment = MultiTurnEnvironment(
        task='task',
        cli_provider_egress=True,
        root_service_start='root-service-marker',
        setup_script='agent-setup-marker',
        final_verifier_code='return 0.0',
    )
    with client.session(environment) as session:
        session.start()
        with contextlib.suppress(VerifierError):
            session.score()
    box = daytona.boxes[0]
    assert box.params.network_block_all is False
    root = next(command for command in box.events if 'root-service-marker' in command)
    agent = next(command for command in box.events if 'agent-setup-marker' in command)
    verifier = next(
        command
        for command in box.events
        if '/opt/deepgym-verifier/' in command and 'python3' in command
    )
    for command in (root, agent, verifier):
        assert '/usr/bin/nsenter --net=/opt/deepgym-netns --' in command
    assert '--reuid=997' in agent
    assert '--reuid=996' in verifier


def test_daytona_params_mount_declared_volumes() -> None:
    pytest.importorskip('daytona')
    from deepgym.models import VolumeMount
    from deepgym.sandboxes.daytona import create_params

    mount = VolumeMount(volume_id='vol', mount_path='/srv/a', subpath='x')
    env = MultiTurnEnvironment(task='t', snapshot='snap', volumes=(mount,))
    params = create_params(env)
    assert [(v.volume_id, v.mount_path, v.subpath) for v in params.volumes] == [
        ('vol', '/srv/a', 'x')
    ]
    assert not create_params(MultiTurnEnvironment(task='t', snapshot='snap')).volumes
