"""An environment's ``image`` reaches the sandbox it runs in, against fakes.

Provider constructors are covered in test_sandboxes.py and live runs in
test_sandboxes_live.py.
"""

import pytest

from deepgym import DeepGym
from deepgym.exceptions import SandboxError
from deepgym.models import Environment, MultiTurnEnvironment
from tests.test_sandboxes import FakeSandbox

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
