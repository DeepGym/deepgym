"""Opt-in GUI and CLI roundtrips; default CI needs no credentials or provider SDKs."""

import json
import os
from uuid import uuid4

import pytest

_GUI_REQUIRED = (
    'DAYTONA_API_KEY',
    'ANTHROPIC_API_KEY',
    'ITOP_ADMIN_PASSWORD',
    'DEEPGYM_AGENT_MODEL',
)
pytestmark = pytest.mark.live_agents


@pytest.mark.skipif(
    os.getenv('DEEPGYM_LIVE_AGENTS') != '1' or not all(os.getenv(key) for key in _GUI_REQUIRED),
    reason='set DEEPGYM_LIVE_AGENTS=1 and Daytona/Anthropic keys, iTop password and model',
)
def test_anthropic_itop_snapshot() -> None:
    """Create a unique organization through the GUI and verify the live iTop database."""
    from deepgym import DeepGym
    from deepgym.agents import AnthropicComputerAgent, Budget
    from deepgym.computer_use import ComputerUseEnvironment

    organization = f'DeepGym smoke {uuid4().hex[:12]}'
    password = os.environ['ITOP_ADMIN_PASSWORD']
    verifier = f"""
import json
import urllib.parse
import urllib.request
query = {{'operation': 'core/get', 'class': 'Organization',
          'key': "SELECT Organization WHERE name = '{organization}'", 'output_fields': 'name'}}
body = urllib.parse.urlencode({{'auth_user': 'admin', 'auth_pwd': {json.dumps(password)},
                               'json_data': json.dumps(query)}}).encode()
with urllib.request.urlopen('http://127.0.0.1:8000/webservices/rest.php?version=1.3',
                            body, timeout=60) as response:
    result = json.load(response)
found = result.get('code') == 0 and len(result.get('objects') or {{}}) == 1
print(json.dumps({{'score': float(found), 'passed': found}}))
"""
    env = ComputerUseEnvironment(
        name='itop-agent-smoke',
        snapshot='img-itop-3.3.0-v1',
        task=f'Sign in to iTop as admin with password {password}. '
        f'Create a new Organization named {organization}. Then stop.',
        root_service_start='bash /opt/probe/start.sh',
        setup_script=(
            'setsid nohup chromium --no-first-run --no-default-browser-check '
            '--password-store=basic --disable-gpu --no-sandbox --test-type '
            '--disable-session-crashed-bubble --disable-features=Translate '
            '--user-data-dir="$HOME/.chrome" --start-maximized '
            'http://localhost:8000/pages/UI.php >/tmp/chrome.log 2>&1 </dev/null & '
            'sleep 12; pgrep -u "$(id -u)" chromium >/dev/null'
        ),
        final_verifier_code=verifier,
        max_steps=240,
        timeout_per_step=120,
        setup_timeout=600,
    )
    agent = AnthropicComputerAgent(model=os.environ['DEEPGYM_AGENT_MODEL'])
    result, grade = DeepGym(mode='daytona').run_episode(
        env, agent, budget=Budget(max_turns=30, max_seconds=300)
    )
    assert result.termination_reason == 'completed', result.termination_reason
    assert result.usage.output_tokens > 0
    assert grade.passed and grade.score == 1.0, grade.output


@pytest.mark.parametrize('provider', ['anthropic', 'openai'])
def test_cli_snapshot_roundtrip(provider: str) -> None:
    """Require a model/tool file write and verify it after supervised CLI teardown."""
    key = 'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY'
    model = 'DEEPGYM_CLAUDE_MODEL' if provider == 'anthropic' else 'DEEPGYM_CODEX_MODEL'
    required = ('DAYTONA_API_KEY', 'DEEPGYM_CLI_SNAPSHOT', key, model)
    if os.getenv('DEEPGYM_LIVE_CLI_AGENTS') != '1' or not all(os.getenv(name) for name in required):
        pytest.skip(f'set DEEPGYM_LIVE_CLI_AGENTS=1 and {", ".join(required)}')

    from deepgym import DeepGym
    from deepgym.agents import Budget, ClaudeCodeAgent, CodexAgent
    from deepgym.models import MultiTurnEnvironment

    filename = f'cli-smoke-{uuid4().hex}.txt'
    content = f'deepgym-roundtrip-{uuid4().hex}'
    verifier = f"""
import json
from pathlib import Path
path = Path({filename!r})
passed = path.is_file() and path.read_text() == {content!r}
print(json.dumps({{'score': float(passed), 'passed': passed}}))
"""
    env = MultiTurnEnvironment(
        name=f'{provider}-cli-smoke',
        snapshot=os.environ['DEEPGYM_CLI_SNAPSHOT'],
        cli_provider_egress=True,
        allowed_actions=frozenset({'cli'}),
        task=f'Create {filename} in your current working directory, containing exactly '
        f'{content!r} with no newline. Use your file-writing or shell tool and make the '
        'file readable by other users (mode 0644). Then stop.',
        final_verifier_code=verifier,
        max_steps=1,
        timeout_per_step=120,
    )
    agent_type = ClaudeCodeAgent if provider == 'anthropic' else CodexAgent
    agent = agent_type(model=os.environ[model], allow_model_credentials=True)
    result, grade = DeepGym(mode='daytona').run_episode(
        env, agent, budget=Budget(max_turns=20, max_seconds=120)
    )
    assert result.termination_reason == 'completed', result.error or result.termination_reason
    assert result.usage.output_tokens > 0
    assert [action.action_type for _, action in result.trajectory.steps] == ['cli']
    assert grade.passed and grade.score == 1.0, grade.output
