"""Reject privileged inline session configuration before creating a backend."""

import pytest
from pydantic import ValidationError

from deepgym.api.schemas import CreateSessionRequest
from deepgym.computer_use import ComputerUseEnvironment
from deepgym.models import MultiTurnEnvironment


@pytest.mark.parametrize('environment_type', ['multi-turn', 'computer-use'])
@pytest.mark.parametrize('field', ['image', 'snapshot', 'root_service_start'])
def test_inline_session_rejects_privileged_configuration(
    environment_type: str,
    field: str,
) -> None:
    with pytest.raises(ValidationError, match=field):
        CreateSessionRequest.model_validate(
            {
                'environment': {'task': 't', 'type': environment_type, field: 'untrusted'},
            }
        )


@pytest.mark.parametrize('model', [MultiTurnEnvironment, ComputerUseEnvironment])
def test_inline_environment_instances_cannot_bypass_root_start_rejection(
    model: type[MultiTurnEnvironment],
) -> None:
    environment = model(task='t', root_service_start='start-trusted-service')
    with pytest.raises(ValidationError, match='root_service_start'):
        CreateSessionRequest(environment=environment)


def test_empty_inline_root_start_remains_allowed() -> None:
    request = CreateSessionRequest.model_validate(
        {
            'environment': {'task': 't', 'root_service_start': ''},
        }
    )
    assert isinstance(request.environment, MultiTurnEnvironment)
    assert request.environment.root_service_start == ''


def test_trusted_registry_reference_remains_allowed() -> None:
    request = CreateSessionRequest(environment='trusted-vendor-snapshot', variant=2)
    assert request.environment == 'trusted-vendor-snapshot'
    assert request.variant == 2


def test_trusted_environment_can_still_specify_root_start() -> None:
    environment = MultiTurnEnvironment(task='t', root_service_start='start-trusted-service')
    assert environment.root_service_start == 'start-trusted-service'


@pytest.mark.parametrize('as_model', [False, True])
def test_inline_api_cannot_open_trusted_root_egress(as_model: bool) -> None:
    environment = {'task': 'task', 'cli_provider_egress': True}
    if as_model:
        environment = MultiTurnEnvironment(**environment)
    with pytest.raises(ValidationError, match='cli_provider_egress cannot be set over the API'):
        CreateSessionRequest(environment=environment)


@pytest.mark.parametrize('as_model', [False, True])
def test_inline_api_cannot_mount_volumes(as_model: bool) -> None:
    environment = {'task': 'task', 'volumes': [{'volume_id': 'v', 'mount_path': '/data'}]}
    if as_model:
        environment = MultiTurnEnvironment(**environment)
    with pytest.raises(ValidationError, match='volumes cannot be set over the API'):
        CreateSessionRequest(environment=environment)
