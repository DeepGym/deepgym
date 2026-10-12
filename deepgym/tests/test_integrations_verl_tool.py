"""DeepGym verl integration against a stand-in for verl (verl itself is not installed here).

The stand-in ToolAgentLoop does what verl's does: create -> execute -> release around
every single tool call. State must still carry across calls within one rollout.
"""

import asyncio
import importlib
import sys
import types
from typing import Any

import pytest

from deepgym.exceptions import DeepGymError
from deepgym.models import MultiTurnEnvironment

ENV = MultiTurnEnvironment(
    task='Write hello to out.txt',
    allowed_actions={'bash', 'done'},
    max_steps=10,
    final_verifier_code=(
        'import json, pathlib\n'
        'p = pathlib.Path("out.txt")\n'
        'ok = p.exists() and p.read_text().strip() == "hello"\n'
        'print(json.dumps({"score": float(ok), "passed": ok}))\n'
    ),
)


class _ToolResponse:
    def __init__(self, text: str | None = None, image: list | None = None) -> None:
        self.text = text
        self.image = image


class _Schema:
    def __init__(self, **kwargs: object) -> None:
        self.data = kwargs


class _BaseTool:
    def __init__(self, config: dict, tool_schema: _Schema) -> None:
        self.config = config
        self.tool_schema = tool_schema
        self.name = 'computer'


class _ToolAgentLoop:
    """Mimic verl's ToolAgentLoop: per-call tool lifecycle, scripted tool calls."""

    def __init__(self, tools: list, calls: list[dict]) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.calls = calls
        self.responses: list[str | None] = []

    async def run(self, sampling_params: dict, **kwargs: object) -> types.SimpleNamespace:
        tools_kwargs = kwargs.get('tools_kwargs') or {}
        for args in self.calls:
            tool = self.tools['computer']
            create_kwargs = tools_kwargs.get('computer', {}).get('create_kwargs', {})
            instance_id, _ = await tool.create(create_kwargs=create_kwargs)
            try:
                response, _, _ = await tool.execute(instance_id, args, agent_data=None)
                self.responses.append(response.text)
            finally:
                await tool.release(instance_id)
        return types.SimpleNamespace(reward_score=None)


@pytest.fixture()
def verl(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    modules = {
        'verl.tools.base_tool': {
            'BaseTool': _BaseTool,
            'OpenAIFunctionToolSchema': _Schema,
            'ToolResponse': _ToolResponse,
        },
        'verl.experimental.agent_loop.agent_loop': {
            'AgentLoopOutput': types.SimpleNamespace,
            'register': lambda name: lambda cls: cls,
        },
        'verl.experimental.agent_loop.tool_agent_loop': {'ToolAgentLoop': _ToolAgentLoop},
    }
    for name in ('verl', 'verl.tools', 'verl.experimental', 'verl.experimental.agent_loop'):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, 'deepgym.integrations.verl_tool', raising=False)
    module = importlib.import_module('deepgym.integrations.verl_tool')
    monkeypatch.setattr(module, 'load_environment', lambda name, variant=0: ENV)
    return module


def _tool(verl: types.ModuleType) -> Any:
    return verl.DeepGymSessionTool({'environment': 'x', 'mode': 'local'}, None)


def _rollout(verl: types.ModuleType, calls: list[dict]) -> tuple[Any, Any, list]:
    tool = _tool(verl)
    loop = verl.DeepGymAgentLoop([tool], calls)
    output = asyncio.run(loop.run({}, tools_kwargs={}))
    return tool, output, loop.responses


def test_state_carries_across_calls_and_reward_is_set(verl: types.ModuleType) -> None:
    tool, output, responses = _rollout(
        verl,
        [
            {'action_type': 'screenshot'},
            {'action_type': 'bash', 'content': 'echo hello > out.txt && echo wrote'},
            {'action_type': 'done'},
        ],
    )

    assert responses[0] == '(no output)'
    assert responses[1].strip() == 'wrote'
    assert output.reward_score == 1.0
    assert not tool._sessions


def test_first_call_starts_episode_without_applying_action(verl: types.ModuleType) -> None:
    _, output, responses = _rollout(
        verl, [{'action_type': 'bash', 'content': 'echo hello > out.txt'}, {'action_type': 'done'}]
    )
    assert responses[0].startswith('Episode started')
    assert output.reward_score == 0.0


def test_bad_calls_come_back_as_text(verl: types.ModuleType) -> None:
    _, _, responses = _rollout(
        verl,
        [
            {'action_type': 'screenshot'},
            {'action_type': 'click'},
            {'action_type': 'code', 'content': 'print(1)'},
        ],
    )
    assert responses[1].startswith('Invalid action')
    assert 'not allowed' in responses[2]


def test_failed_rollout_still_deletes_sandbox(verl: types.ModuleType) -> None:
    tool = _tool(verl)
    real_execute = tool.execute

    async def execute(instance_id: str, parameters: dict, **kwargs: object) -> tuple:
        if parameters['action_type'] == 'done':
            raise RuntimeError('rollout crashed')
        return await real_execute(instance_id, parameters, **kwargs)

    tool.execute = execute
    loop = verl.DeepGymAgentLoop([tool], [{'action_type': 'screenshot'}, {'action_type': 'done'}])
    with pytest.raises(RuntimeError):
        asyncio.run(loop.run({}, tools_kwargs={}))
    assert not tool._sessions


def test_plain_tool_agent_loop_is_rejected(verl: types.ModuleType) -> None:
    with pytest.raises(DeepGymError, match='deepgym_agent'):
        asyncio.run(_tool(verl).create(create_kwargs={}))


def test_default_schema_lists_every_action(verl: types.ModuleType) -> None:
    enum = _tool(verl).tool_schema.data['function']['parameters']['properties']['action_type'][
        'enum'
    ]
    assert {'click', 'type', 'key', 'bash', 'done'} <= set(enum)


def test_parquet_nulls_fall_back_to_the_tool_default(verl: types.ModuleType) -> None:
    tool = _tool(verl)
    create_kwargs = {'episode': 'e', 'environment': None, 'variant': None}
    asyncio.run(tool.create(create_kwargs=create_kwargs))
    assert tool._specs['e'] == ('x', 0)
