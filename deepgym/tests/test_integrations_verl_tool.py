"""DeepGym verl integration against a stand-in for verl (verl itself is not installed here).

The stand-in ToolAgentLoop does what verl's does: create -> execute -> release around
every single tool call, and an exception from the tool comes back to the model as
tool-error text while the rollout goes on (ToolAgentLoop._call_tool at verl 76e837e).
State must still carry across calls within one rollout. Sessions are real (local).
"""

import asyncio
import importlib
import logging
import re
import sys
import types
from typing import Any

import pytest

from deepgym.exceptions import DeepGymError, SandboxError
from deepgym.exceptions import TimeoutError as DeepGymTimeout
from deepgym.models import MODEL_FAULT_ERRORS, MultiTurnEnvironment
from deepgym.sandboxes import SandboxTimeoutError
from deepgym.session import LocalBackend, Session

WRITES_HELLO = (
    'import json, pathlib\n'
    'p = pathlib.Path("out.txt")\n'
    'ok = p.exists() and p.read_text().strip() == "hello"\n'
    'print(json.dumps({"score": float(ok), "passed": ok}))\n'
)
ENV = MultiTurnEnvironment(
    task='Write hello to out.txt',
    allowed_actions={'bash', 'done'},
    max_steps=10,
    final_verifier_code=WRITES_HELLO,
)
WRITE_AND_FINISH = [
    {'action_type': 'screenshot'},
    {'action_type': 'bash', 'content': 'echo hello > out.txt && echo wrote'},
    {'action_type': 'done'},
]


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

    def __init__(self, tools: list, calls: list[dict], use_v1: bool = True) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.calls = calls
        self.responses: list[str | None] = []
        self.config = types.SimpleNamespace(trainer={'use_v1': use_v1})

    async def run(self, sampling_params: dict, **kwargs: object) -> types.SimpleNamespace:
        tools_kwargs = kwargs.get('tools_kwargs') or {}
        for args in self.calls:
            self.responses.append((await self._call_tool(args, tools_kwargs)).text)
        return types.SimpleNamespace(reward_score=None, response_mask=[1, 1, 1], extra_fields={})

    async def _call_tool(self, args: dict, tools_kwargs: dict) -> _ToolResponse:
        """As verl's _call_tool: a tool exception becomes text for the model."""
        tool, instance_id = self.tools['computer'], None
        try:
            create_kwargs = tools_kwargs.get('computer', {}).get('create_kwargs', {})
            instance_id, _ = await tool.create(create_kwargs=create_kwargs)
            response, _, _ = await tool.execute(instance_id, args, agent_data=None)
        except Exception as exc:
            return _ToolResponse(text=f"Error executing tool 'computer': {exc}")
        finally:
            if instance_id:
                await tool.release(instance_id)
        return response


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
        'verl.experimental.agent_loop.single_turn_agent_loop': {'SingleTurnAgentLoop': object},
        'verl.trainer.ppo.reward': {'get_custom_reward_fn': lambda config: None},
    }
    for name in (
        'verl',
        'verl.tools',
        'verl.experimental',
        'verl.experimental.agent_loop',
        'verl.trainer',
        'verl.trainer.ppo',
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    for name in ('deepgym.integrations.verl_tool', 'deepgym.integrations.verl_mask'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module('deepgym.integrations.verl_tool')
    monkeypatch.setattr(module, 'load_environment', lambda name, variant=0: ENV)
    return module


def _tool(verl: types.ModuleType) -> Any:
    return verl.DeepGymSessionTool({'environment': 'x', 'mode': 'local'}, None)


def _rollout(
    verl: types.ModuleType, calls: list[dict], use_v1: bool = True
) -> tuple[Any, Any, list]:
    tool = _tool(verl)
    loop = verl.DeepGymAgentLoop([tool], calls, use_v1)
    output = asyncio.run(loop.run({}, tools_kwargs={}))
    return tool, output, loop.responses


def test_state_carries_across_calls_and_reward_is_set(verl: types.ModuleType) -> None:
    tool, output, responses = _rollout(verl, WRITE_AND_FINISH)

    assert responses[0] == '(no output)'
    assert responses[1].strip() == 'wrote'
    assert output.reward_score == 1.0
    assert output.extra_fields['reward_extra_info'] == {'score': 1.0, 'passed': 1.0, 'error': 0.0}
    assert output.response_mask == [1, 1, 1]
    assert not tool._sessions


def test_first_call_starts_episode_without_applying_action(verl: types.ModuleType) -> None:
    _, output, responses = _rollout(
        verl, [{'action_type': 'bash', 'content': 'echo hello > out.txt'}, {'action_type': 'done'}]
    )
    assert responses[0].startswith('Episode started')
    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info']['error'] == 0.0


def test_a_rollout_without_tool_calls_scores_zero(verl: types.ModuleType) -> None:
    """The model never acted, so the task isn't done: a real 0, trained on."""
    _, output, _ = _rollout(verl, [])

    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info'] == {'score': 0.0, 'passed': 0.0, 'error': 0.0}
    assert output.response_mask == [1, 1, 1]


@pytest.fixture()
def failing_start(verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make the sandbox's setup fail (a SandboxError from session.start); count the starts."""
    env = ENV.model_copy(update={'setup_script': 'echo starting >> starts.log; exit 3'})
    starts: list[str] = []

    def load(name: str, variant: int = 0) -> MultiTurnEnvironment:
        starts.append(name)
        return env

    monkeypatch.setattr(verl, 'load_environment', load)
    return starts


def test_a_sandbox_that_fails_to_start_drops_the_rollout_under_v1(
    verl: types.ModuleType, failing_start: list[str]
) -> None:
    """verl swallows the startup error into text; the rollout must not end as a plain 0.0."""
    tool = _tool(verl)
    loop = verl.DeepGymAgentLoop([tool], WRITE_AND_FINISH, True)

    with pytest.raises(DeepGymError, match='sandbox_error'):
        asyncio.run(loop.run({}, tools_kwargs={}))

    assert loop.responses[0].startswith("Error executing tool 'computer'")
    assert loop.responses[1:] == [verl._FAILED_NOTE] * 2
    assert failing_start == ['x']
    assert not tool._failed


def test_a_sandbox_that_fails_to_start_is_masked_under_the_legacy_trainer(
    verl: types.ModuleType, failing_start: list[str]
) -> None:
    _, output, _ = _rollout(verl, WRITE_AND_FINISH, use_v1=False)

    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info']['error'] == 1.0
    assert output.response_mask == [0, 0, 0]
    assert failing_start == ['x']


@pytest.mark.parametrize(
    'final_verifier',
    [
        # Reports its own failure (exit 2): score() returns a result without a verdict.
        'import json\n'
        'print(json.dumps({"score": 0.0, "passed": False, "error_type": "verifier_error"}))\n'
        'raise SystemExit(2)\n',
        # Prints nothing usable: score() raises.
        'print("not json")\n',
    ],
)
def test_a_final_verifier_failure_is_not_a_zero(
    verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch, final_verifier: str
) -> None:
    env = ENV.model_copy(update={'final_verifier_code': final_verifier})
    monkeypatch.setattr(verl, 'load_environment', lambda name, variant=0: env)

    tool, output, _ = _rollout(verl, WRITE_AND_FINISH, use_v1=False)

    assert output.extra_fields['reward_extra_info']['error'] == 1.0
    assert output.response_mask == [0, 0, 0]
    assert not tool._sessions
    with pytest.raises(DeepGymError, match='No verdict'):
        _rollout(verl, WRITE_AND_FINISH, use_v1=True)


def test_a_step_verifier_failure_is_not_a_zero(
    verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """verl shows the step's VerifierError to the model; score() then reports no verdict."""
    env = ENV.model_copy(update={'step_verifier_code': 'raise SystemExit(3)\n'})
    monkeypatch.setattr(verl, 'load_environment', lambda name, variant=0: env)

    _, output, responses = _rollout(verl, WRITE_AND_FINISH, use_v1=False)

    assert responses[1].startswith("Error executing tool 'computer'")
    assert output.extra_fields['reward_extra_info']['error'] == 1.0
    assert output.response_mask == [0, 0, 0]


@pytest.fixture()
def masked(verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """What the loop reports to the masked-rate warning, one entry per rollout."""
    reports: list[bool] = []
    recorder = types.SimpleNamespace(record=lambda *, masked: reports.append(masked))
    monkeypatch.setattr(verl, '_MASKED', recorder)
    return reports


def _fail_writing_out_txt(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Make the model's command that writes out.txt fail with *error*."""
    real_exec = LocalBackend.exec

    def exec_(backend: LocalBackend, command: str, timeout: int, env: Any = None) -> Any:
        if 'out.txt' in command:
            raise error
        return real_exec(backend, command, timeout, env)

    monkeypatch.setattr(LocalBackend, 'exec', exec_)


@pytest.mark.parametrize('use_v1', [True, False])
def test_a_provider_outage_during_an_action_is_not_a_zero(
    verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch, masked: list[bool], use_v1: bool
) -> None:
    """An outage blocks the model's correct command; the state it left is not a verdict.

    verl shows the error to the model and the rollout goes on to 'done', so the tool must
    remember the failure, or the final verifier would grade the unwritten file as a 0.
    """
    _fail_writing_out_txt(monkeypatch, ConnectionError('provider unavailable'))
    tool = _tool(verl)
    loop = verl.DeepGymAgentLoop([tool], WRITE_AND_FINISH, use_v1)

    if use_v1:
        with pytest.raises(DeepGymError, match='No verdict'):
            asyncio.run(loop.run({}, tools_kwargs={}))
    else:
        output = asyncio.run(loop.run({}, tools_kwargs={}))
        assert output.extra_fields['reward_extra_info']['error'] == 1.0
        assert output.response_mask == [0, 0, 0]
    assert loop.responses[1:] == ["Error executing tool 'computer': provider unavailable"] + [
        verl._FAILED_NOTE
    ]
    assert not tool._sessions
    assert masked == [True]


def test_an_action_the_policy_broke_is_scored(
    verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch, masked: list[bool]
) -> None:
    """A failure that follows from the policy's own action is shown to it and scores 0.

    Masking it would let a policy escape its 0 by breaking its sandbox.
    """
    _fail_writing_out_txt(monkeypatch, SandboxError('Sandbox local-session is not running'))

    _, output, responses = _rollout(verl, WRITE_AND_FINISH)

    assert responses[1] == "Error executing tool 'computer': Sandbox local-session is not running"
    assert responses[2] == 'The episode is over.'  # the failed step ended it
    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info']['error'] == 0.0
    assert output.response_mask == [1, 1, 1]
    assert masked == [False]


class _RefusedError(Exception):
    """A provider refusing our credentials, as its SDK reports it."""

    status_code = 401


@pytest.mark.parametrize(
    ('failure', 'error_type'),
    [
        (lambda mark: ConnectionError('provider unavailable'), 'unknown_error'),
        (lambda mark: _RefusedError('HTTP 401'), 'unknown_error'),
        # It timed out before it printed the consulted mark: it never read the policy's output.
        (
            lambda mark: SandboxTimeoutError('exceeded 30s', f'\n{mark}:start\nloading fixtures'),
            'verifier_error',
        ),
        # Nothing it printed came back, so nothing shows it hadn't read the policy's output.
        (lambda mark: SandboxTimeoutError('exceeded 30s'), 'policy_sandbox_failure'),
        (lambda mark: SandboxTimeoutError('exceeded 30s', f'\n{mark}\n'), 'solution_timeout'),
        (
            lambda mark: SandboxError('Sandbox local-session is not running'),
            'policy_sandbox_failure',
        ),
        (lambda mark: DeepGymTimeout('Verifier exceeded 30s'), 'policy_sandbox_failure'),
    ],
    ids=[
        'provider_outage',
        'auth',
        'timed_out_before_reading',
        'timed_out_with_output_lost',
        'timed_out_reading',
        'sandbox_broken',
        'sandbox_hung',
    ],
)
def test_a_final_verifier_that_cannot_run(
    verl: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    masked: list[bool],
    failure: Any,
    error_type: str,
) -> None:
    """Only a failure that is ours for certain keeps the rollout out; the rest scores 0."""
    kept_out = error_type not in MODEL_FAULT_ERRORS

    def verify(backend: LocalBackend, code: str, timeout: int, env: Any = None) -> Any:
        raise failure(re.search(r'DEEPGYM_CONSULTED"\] = \'(\w+)\'', code)[1])

    monkeypatch.setattr(LocalBackend, 'verify', verify)
    real_finish = verl.DeepGymSessionTool.finish
    results: list = []

    async def finish(tool: Any, episode: str, *, score: bool) -> Any:
        results.append(await real_finish(tool, episode, score=score))
        return results[-1]

    monkeypatch.setattr(verl.DeepGymSessionTool, 'finish', finish)

    _, output, _ = _rollout(verl, WRITE_AND_FINISH, use_v1=False)

    assert output.reward_score == 0.0
    assert output.extra_fields['reward_extra_info']['error'] == float(kept_out)
    assert output.response_mask == ([0, 0, 0] if kept_out else [1, 1, 1])
    assert results[0].has_verdict is not kept_out
    assert results[0].error_type == error_type
    assert masked == [kept_out]


def test_a_failed_sandbox_delete_keeps_the_score(
    verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A provider that reports a failed delete must not cost the episode its verdict."""
    real_close = Session.close

    def close(session: Session) -> None:
        real_close(session)
        raise SandboxError('delete failed: provider returned 500')

    monkeypatch.setattr(Session, 'close', close)

    with caplog.at_level(logging.WARNING, logger='deepgym.integrations.verl_tool'):
        tool, output, _ = _rollout(verl, WRITE_AND_FINISH)

    assert output.reward_score == 1.0
    assert output.extra_fields['reward_extra_info']['error'] == 0.0
    assert 'Could not delete sandbox' in caplog.text
    assert not tool._sessions


def test_bad_calls_come_back_as_text(verl: types.ModuleType) -> None:
    """A bad call is the model's own doing: the episode still gets its verdict."""
    _, output, responses = _rollout(
        verl,
        [
            {'action_type': 'screenshot'},
            {'action_type': 'click'},
            {'action_type': 'code', 'content': 'print(1)'},
        ],
    )
    assert responses[1].startswith('Invalid action')
    assert 'not allowed' in responses[2]
    assert output.extra_fields['reward_extra_info']['error'] == 0.0


def test_failed_rollout_still_deletes_sandbox(
    verl: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _ToolAgentLoop.run

    async def crashing(self: Any, sampling_params: dict, **kwargs: object) -> None:
        await original(self, sampling_params, **kwargs)
        raise RuntimeError('rollout crashed')

    monkeypatch.setattr(_ToolAgentLoop, 'run', crashing)
    tool = _tool(verl)
    loop = verl.DeepGymAgentLoop([tool], [{'action_type': 'screenshot'}])

    with pytest.raises(RuntimeError, match='rollout crashed'):
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


def test_reward_extra_info_keys_match_compute_score(verl: types.ModuleType) -> None:
    """verl batches reward_extra_info by the first sample's keys, so every loop must agree."""
    from deepgym.integrations.verl import compute_score

    _, output, _ = _rollout(verl, WRITE_AND_FINISH)
    row = compute_score('coin_change', 'pass', mode='local')

    assert output.extra_fields['reward_extra_info'].keys() == row.keys()
