"""Test actual provider loops with in-memory SDK responses and Session stepping."""

import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from deepgym.agents.actions import anthropic_actions, openai_actions
from deepgym.agents.base import Budget, TokenPrices
from deepgym.agents.computer import AnthropicComputerAgent, OpenAIComputerAgent
from deepgym.models import Action, Observation, Trajectory
from deepgym.session import StepResult


@pytest.mark.parametrize(
    'name,data,kind',
    [
        ('screenshot', {}, 'screenshot'),
        ('left_click', {'coordinate': [2, 3]}, 'click'),
        ('right_click', {'coordinate': [2, 3]}, 'right_click'),
        ('double_click', {'coordinate': [2, 3]}, 'double_click'),
        ('mouse_move', {'coordinate': [2, 3]}, 'move'),
        ('left_click_drag', {'start_coordinate': [1, 1], 'coordinate': [2, 3]}, 'drag'),
        ('scroll', {'coordinate': [2, 3], 'scroll_direction': 'down'}, 'scroll'),
        ('type', {'text': 'hello'}, 'type'),
        ('key', {'text': 'Control+Return'}, 'key'),
        ('wait', {'duration': 1.5}, 'wait'),
    ],
)
def test_anthropic_mapping(name, data, kind):
    assert anthropic_actions(name, data)[0].action_type == kind


@pytest.mark.parametrize(
    'name,data',
    [
        ('middle_click', {'coordinate': [1, 2]}),
        ('left_mouse_down', {}),
        ('left_mouse_up', {}),
        ('cursor_position', {}),
        ('hold_key', {'text': 'a'}),
        ('zoom', {}),
        ('triple_click', {'coordinate': [1, 2]}),
        ('left_click', {}),
        ('left_click', {'coordinate': [1, 2], 'text': 'shift'}),
        ('left_click', {'coordinate': [1.5, 2]}),
        ('left_click', {'coordinate': [True, 2]}),
        ('scroll', {'coordinate': [1, 2], 'scroll_direction': 'down', 'scroll_amount': 100}),
        ('key', {'text': 'a', 'repeat': 0}),
        ('key', {'text': 'ctrl++a'}),
        ('wait', {'duration': float('nan')}),
        ('unknown', {}),
    ],
)
def test_anthropic_rejects_without_changing_meaning(name, data):
    with pytest.raises(ValueError):
        anthropic_actions(name, data)


@pytest.mark.parametrize(
    'data,kind',
    [
        ({'type': 'click', 'x': 2, 'y': 3, 'button': 'left'}, 'click'),
        ({'type': 'click', 'x': 2, 'y': 3, 'button': 'right'}, 'right_click'),
        ({'type': 'double_click', 'x': 2, 'y': 3}, 'double_click'),
        ({'type': 'move', 'x': 2, 'y': 3}, 'move'),
        ({'type': 'drag', 'path': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]}, 'drag'),
        ({'type': 'scroll', 'x': 2, 'y': 3, 'scroll_y': -200}, 'scroll'),
        ({'type': 'scroll', 'x': 2, 'y': 3, 'scroll_x': 200}, 'scroll'),
        ({'type': 'type', 'text': 'hello'}, 'type'),
        ({'type': 'keypress', 'keys': ['CTRL', 'A']}, 'key'),
        ({'type': 'wait'}, 'wait'),
        ({'type': 'screenshot'}, 'screenshot'),
    ],
)
def test_openai_mapping(data, kind):
    assert openai_actions(data)[0].action_type == kind


@pytest.mark.parametrize(
    'data',
    [
        {'type': 'click', 'x': 1, 'y': 2, 'button': 'middle'},
        {'type': 'click', 'x': 1, 'y': 2, 'keys': ['shift']},
        {'type': 'scroll', 'x': 1, 'y': 2, 'scroll_x': 10, 'scroll_y': 20},
        {'type': 'drag', 'path': [{'x': 1, 'y': 2}] * 3},
        {'type': 'keypress', 'keys': []},
        {'type': 'unknown'},
    ],
)
def test_openai_rejects_unsupported(data):
    with pytest.raises(ValueError):
        openai_actions(data)


class FakeSession:
    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps
        self.done = False
        self.trajectory = Trajectory()
        self.observation = Observation(content='', step=0, screenshot='aW1hZ2U=')

    def step(self, action: Action, *, timeout=None):
        assert timeout > 0
        self.trajectory.steps.append((self.observation, action))
        step = self.observation.step + 1
        self.done = self.max_steps is not None and step >= self.max_steps
        self.observation = self.observation.model_copy(
            update={
                'step': step,
                'done': self.done,
                'screenshot': f'image-{step}',
            }
        )
        self.trajectory.step_rewards.append(0)
        self.trajectory.final_observation = self.observation
        return StepResult(self.observation, 0, self.done)


class FakeClient:
    def __init__(self, responses):
        self.responses = self.messages = self
        self.results = iter(responses)
        self.requests = []
        self.closed = False

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = next(self.results)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(model_dump=lambda **_: response)

    def stream(self, **kwargs):
        return _Stream(self.create(**kwargs))  # raises like a failed request

    def close(self):
        self.closed = True


class _Stream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):  # a complete fake stream ends with message_stop
        yield SimpleNamespace(type='message_stop')

    @property
    def current_message_snapshot(self):
        return self.response


@pytest.fixture
def sdk(monkeypatch):
    def install(provider, responses):
        client = FakeClient(responses)

        def constructor(**kwargs):
            assert kwargs['max_retries'] == 4
            return client

        module = SimpleNamespace(
            APIError=RuntimeError,
            **{'Anthropic' if provider == 'anthropic' else 'OpenAI': constructor},
        )
        monkeypatch.setitem(sys.modules, provider, module)
        return client

    return install


def anthropic_response(content, **kwargs):
    return {
        'usage': {'input_tokens': 100, 'output_tokens': 10},
        'content': content,
        'stop_reason': 'end_turn',
        **kwargs,
    }


def openai_response(output, **kwargs):
    return {
        'id': 'response-1',
        'usage': {
            'input_tokens': 100,
            'output_tokens': 10,
            'input_tokens_details': {'cached_tokens': 20},
        },
        'output': output,
        'status': 'completed',
        **kwargs,
    }


def test_anthropic_real_loop_tracks_actions_and_usage(sdk):
    client = sdk(
        'anthropic',
        [
            anthropic_response(
                [
                    {
                        'type': 'tool_use',
                        'id': 'call-1',
                        'name': 'left_click',
                        'input': {'coordinate': [10, 20]},
                    }
                ]
            ),
            anthropic_response([{'type': 'text', 'text': 'Finished'}]),
        ],
    )
    session = FakeSession()
    result = AnthropicComputerAgent('test-model').run(
        session, session.observation, task='Do it', budget=Budget()
    )
    assert result.termination_reason == 'completed'
    assert result.final_text == 'Finished'
    assert result.turns == 2
    assert result.usage.input_tokens == 200
    assert result.cost.usd is None
    assert [action.action_type for _, action in result.trajectory.steps] == ['click']
    assert client.closed
    assert 0 < client.requests[0]['timeout'] <= 300


def test_anthropic_bad_action_skips_rest_of_batch_and_reports_error(sdk):
    client = sdk(
        'anthropic',
        [
            anthropic_response(
                [
                    {'type': 'tool_use', 'id': 'bad', 'name': 'hold_key', 'input': {'text': 'a'}},
                    {
                        'type': 'tool_use',
                        'id': 'next',
                        'name': 'type',
                        'input': {'text': 'must not type'},
                    },
                ]
            ),
            anthropic_response([]),
        ],
    )
    session = FakeSession()
    result = AnthropicComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget()
    )
    assert [a.action_type for _, a in result.trajectory.steps] == ['screenshot']
    content = client.requests[1]['messages'][-2]['content']
    results = [item for item in content if item['type'] == 'tool_result']
    assert all(item['is_error'] and isinstance(item['content'], str) for item in results)
    assert content[-1]['type'] == 'image'


def test_openai_real_loop_preserves_response_chain(sdk):
    client = sdk(
        'openai',
        [
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call-1',
                        'actions': [
                            {'type': 'click', 'x': 5, 'y': 6, 'button': 'left'},
                            {'type': 'type', 'text': 'hello'},
                        ],
                    }
                ]
            ),
            openai_response(
                [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Finished'}]}]
            ),
        ],
    )
    session = FakeSession()
    result = OpenAIComputerAgent('test').run(
        session, session.observation, task='Do it', budget=Budget()
    )
    assert result.termination_reason == 'completed'
    assert result.usage.input_tokens == 160
    assert result.usage.cache_read_tokens == 40
    assert [a.action_type for _, a in result.trajectory.steps] == ['click', 'type']
    assert client.requests[1]['previous_response_id'] == 'response-1'
    assert client.requests[1]['input'][0]['call_id'] == 'call-1'


def test_openai_safety_checks_block_all_actions(sdk):
    sdk(
        'openai',
        [
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call-1',
                        'pending_safety_checks': [
                            {'id': 'check-1', 'code': 'malicious_instructions'}
                        ],
                        'action': {'type': 'click', 'x': 5, 'y': 6},
                    }
                ]
            )
        ],
    )
    session = FakeSession()
    result = OpenAIComputerAgent('test').run(
        session, session.observation, task='Do it', budget=Budget()
    )
    assert result.termination_reason == 'safety_check'
    assert result.trajectory.steps == []


def test_anthropic_output_limit_continues_within_turn_budget(sdk):
    client = sdk('anthropic', [anthropic_response([], stop_reason='max_tokens')])
    session = FakeSession()
    result = AnthropicComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget(max_turns=1)
    )
    assert result.termination_reason == 'turn_budget'
    assert len(client.requests) == 1


CLICK = {'type': 'computer_call', 'call_id': 'call', 'action': {'type': 'click', 'x': 1, 'y': 2}}


@pytest.mark.parametrize('output', [[], [CLICK]])
@pytest.mark.parametrize(
    'details', [{'reason': 'max_output_tokens'}, {'reason': 'content_filter'}, None]
)
def test_openai_incomplete_response_stops_without_acting(sdk, output, details):
    response = openai_response(output, status='incomplete')
    if details:
        response['incomplete_details'] = details
    client = sdk('openai', [response])
    session = FakeSession()
    result = OpenAIComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget(max_turns=5)
    )
    assert result.termination_reason == 'error'
    reason = (details or {}).get('reason', 'incomplete')
    assert result.error == f'Response did not complete: {reason}'
    assert len(client.requests) == 1
    assert 'max_output_tokens' not in client.requests[0] and client.requests[0]['reasoning'] == {
        'effort': 'max'
    }
    assert not result.trajectory.steps
    assert result.usage.output_tokens == 10


@pytest.mark.parametrize('agent', [AnthropicComputerAgent, OpenAIComputerAgent])
def test_unknown_prices_reject_usd_budget_before_sdk_import(agent):
    session = FakeSession()
    with pytest.raises(ValueError, match='TokenPrices'):
        agent('unknown').run(
            session, session.observation, task='Task', budget=Budget(max_cost_usd=1)
        )


def test_cost_cap_stops_before_applying_returned_actions(sdk):
    prices = TokenPrices(
        input_per_million=100,
        output_per_million=100,
        cache_read_per_million=100,
        cache_write_per_million=100,
        source='test rates',
    )
    sdk(
        'openai',
        [
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call',
                        'action': {'type': 'click', 'x': 1, 'y': 2},
                    }
                ]
            )
        ],
    )
    session = FakeSession()
    result = OpenAIComputerAgent('test', prices).run(
        session, session.observation, task='Task', budget=Budget(max_cost_usd=0.001)
    )
    assert result.termination_reason == 'cost_budget'
    assert result.cost.usd == pytest.approx(0.011)
    assert not result.trajectory.steps


def test_api_failure_cost_unknown_and_no_retry(sdk):
    client = sdk('anthropic', [RuntimeError('secret must not appear in result')])
    session = FakeSession()
    result = AnthropicComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget()
    )
    assert result.termination_reason == 'error'
    assert result.cost.usd is None
    assert 'secret' not in result.error
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    'agent,provider,response',
    [
        (AnthropicComputerAgent, 'anthropic', anthropic_response([], usage=None)),
        (OpenAIComputerAgent, 'openai', openai_response([], usage={'output_tokens': 1})),
    ],
)
def test_missing_provider_usage_never_reports_zero_cost(sdk, agent, provider, response):
    sdk(provider, [response])
    session = FakeSession()
    result = agent('test').run(session, session.observation, task='Task', budget=Budget())
    assert result.termination_reason == 'error'
    assert result.cost.usd is None
    assert 'billing unknown' in result.cost.provenance


def test_failed_openai_response_cannot_execute_actions(sdk):
    sdk(
        'openai',
        [
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call',
                        'action': {'type': 'click', 'x': 1, 'y': 2},
                    }
                ],
                status='failed',
            )
        ],
    )
    session = FakeSession()
    result = OpenAIComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget()
    )
    assert result.termination_reason == 'error'
    assert not result.trajectory.steps


def test_deadline_after_response_prevents_tool_execution(sdk, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('deepgym.agents.base.time.monotonic', lambda: clock[0])
    client = sdk(
        'openai',
        [
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call',
                        'action': {'type': 'click', 'x': 1, 'y': 2},
                    }
                ]
            )
        ],
    )
    create = client.create

    def delayed(**kwargs):
        clock[0] = 10
        return create(**kwargs)

    client.create = delayed
    session = FakeSession()
    result = OpenAIComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget(max_seconds=5)
    )
    assert result.termination_reason == 'time_budget'
    assert not result.trajectory.steps
    assert result.usage.input_tokens == 80


@pytest.mark.parametrize(
    'agent,provider,response',
    [
        (
            AnthropicComputerAgent,
            'anthropic',
            anthropic_response(
                [
                    {
                        'type': 'tool_use',
                        'id': 'call',
                        'name': 'left_click',
                        'input': {'coordinate': [1, 2]},
                    }
                ]
            ),
        ),
        (
            OpenAIComputerAgent,
            'openai',
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call',
                        'action': {'type': 'click', 'x': 1, 'y': 2},
                    }
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize('expired', [False, True])
def test_session_failure_preserves_accounting(sdk, monkeypatch, agent, provider, response, expired):
    from deepgym.exceptions import SandboxError, TimeoutError

    clock = [0.0]
    monkeypatch.setattr('deepgym.agents.base.time.monotonic', lambda: clock[0])
    client = sdk(provider, [response])
    session = FakeSession()

    def failed_step(action, *, timeout):
        session.trajectory.steps.append((session.observation, action))
        session.done = True
        if expired:
            clock[0] = 10
            raise TimeoutError('sensitive backend context')
        raise SandboxError('sensitive backend context')

    session.step = failed_step
    prices = TokenPrices(
        input_per_million=100,
        output_per_million=100,
        cache_read_per_million=100,
        cache_write_per_million=100,
        source='test rates',
    )
    result = agent('test', prices).run(
        session, session.observation, task='Task', budget=Budget(max_seconds=5)
    )
    assert result.termination_reason == ('time_budget' if expired else 'error')
    assert result.turns == 1
    assert result.cost.usd == pytest.approx(0.011)
    assert result.usage.output_tokens == 10
    assert len(result.trajectory.steps) == 1
    assert result.error == f'Session step failed: {"TimeoutError" if expired else "SandboxError"}'
    assert client.closed


@pytest.mark.parametrize(
    'agent,provider,response',
    [
        (
            AnthropicComputerAgent,
            'anthropic',
            anthropic_response(
                [
                    {
                        'type': 'tool_use',
                        'id': 'call',
                        'name': 'left_click',
                        'input': {'coordinate': [1, 2]},
                    }
                ]
            ),
        ),
        (
            OpenAIComputerAgent,
            'openai',
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call',
                        'action': {'type': 'click', 'x': 1, 'y': 2},
                    }
                ]
            ),
        ),
    ],
)
def test_two_step_session_allows_two_click_turns(sdk, agent, provider, response):
    client = sdk(provider, [response, response])
    session = FakeSession(max_steps=2)
    result = agent('test').run(session, session.observation, task='Task', budget=Budget())
    assert result.termination_reason == 'session_done'
    assert len(client.requests) == 2
    assert result.turns == 2
    assert [action.action_type for _, action in result.trajectory.steps] == ['click', 'click']
    assert len(result.trajectory.step_rewards) == 2
    assert result.trajectory.final_observation.step == 2
    if provider == 'anthropic':
        results = client.requests[1]['messages'][2]['content']
        assert results[-1]['content'][-1]['source']['data'] == 'image-1'
    else:
        assert client.requests[1]['input'][0]['output']['image_url'].endswith('image-1')


@pytest.mark.parametrize(
    'agent,provider,response,completed',
    [
        (
            AnthropicComputerAgent,
            'anthropic',
            anthropic_response(
                [
                    {
                        'type': 'tool_use',
                        'id': 'first',
                        'name': 'left_click',
                        'input': {'coordinate': [1, 2]},
                    },
                    {'type': 'tool_use', 'id': 'bad', 'name': 'hold_key', 'input': {'text': 'a'}},
                    {
                        'type': 'tool_use',
                        'id': 'skip',
                        'name': 'type',
                        'input': {'text': 'must not type'},
                    },
                ]
            ),
            anthropic_response([]),
        ),
        (
            OpenAIComputerAgent,
            'openai',
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call',
                        'actions': [
                            {'type': 'click', 'x': 1, 'y': 2},
                            {'type': 'unsupported'},
                            {'type': 'type', 'text': 'must not type'},
                        ],
                    }
                ]
            ),
            openai_response([]),
        ),
    ],
)
def test_rejected_later_action_reuses_last_successful_observation(
    sdk,
    agent,
    provider,
    response,
    completed,
):
    client = sdk(provider, [response, completed])
    session = FakeSession()
    result = agent('test').run(session, session.observation, task='Task', budget=Budget())
    assert result.termination_reason == 'completed'
    assert [action.action_type for _, action in result.trajectory.steps] == ['click']
    if provider == 'anthropic':
        results = client.requests[1]['messages'][2]['content']
        assert results[0]['content'][-1]['source']['data'] == 'image-1'
        assert results[1]['is_error'] and results[2]['is_error']
        assert isinstance(results[1]['content'], str) and isinstance(results[2]['content'], str)
    else:
        assert client.requests[1]['input'][0]['output']['image_url'].endswith('image-1')
        assert 'Action rejected' in client.requests[1]['input'][1]['content']


@pytest.mark.parametrize(
    'agent,provider,response,completed',
    [
        (
            AnthropicComputerAgent,
            'anthropic',
            anthropic_response(
                [
                    {
                        'type': 'tool_use',
                        'id': 'call',
                        'name': 'left_click',
                        'input': {'coordinate': [1, 2]},
                    }
                ]
            ),
            anthropic_response([]),
        ),
        (
            OpenAIComputerAgent,
            'openai',
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call',
                        'action': {'type': 'click', 'x': 1, 'y': 2},
                    }
                ]
            ),
            openai_response([]),
        ),
    ],
)
def test_missing_action_screenshot_gets_one_fresh_observation(
    sdk,
    agent,
    provider,
    response,
    completed,
):
    client = sdk(provider, [response, completed])
    session = FakeSession()
    original_step = session.step

    def missing_first_screenshot(action, *, timeout):
        result = original_step(action, timeout=timeout)
        if result.observation.step == 1:
            return StepResult(result.observation.model_copy(update={'screenshot': None}), 0, False)
        return result

    session.step = missing_first_screenshot
    result = agent('test').run(session, session.observation, task='Task', budget=Budget())
    assert result.termination_reason == 'completed'
    assert [action.action_type for _, action in result.trajectory.steps] == ['click', 'screenshot']
    if provider == 'anthropic':
        results = client.requests[1]['messages'][2]['content']
        assert results[-1]['content'][-1]['source']['data'] == 'image-2'
    else:
        assert client.requests[1]['input'][0]['output']['image_url'].endswith('image-2')


def test_openai_rejected_batch_gets_one_fresh_observation(sdk):
    client = sdk(
        'openai',
        [
            openai_response(
                [
                    {
                        'type': 'computer_call',
                        'call_id': 'call',
                        'actions': [
                            {'type': 'unsupported'},
                            {'type': 'type', 'text': 'must not type'},
                        ],
                    }
                ]
            ),
            openai_response([]),
        ],
    )
    session = FakeSession()
    result = OpenAIComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget()
    )
    assert result.termination_reason == 'completed'
    assert [action.action_type for _, action in result.trajectory.steps] == ['screenshot']
    assert client.requests[1]['input'][0]['output']['image_url'].endswith('image-1')


def test_screenshot_never_attached_to_an_error_result(sdk):
    failing = {
        'type': 'tool_use',
        'id': 'bad',
        'name': 'middle_click',
        'input': {'coordinate': [1, 1]},
    }
    client = sdk(
        'anthropic',
        [anthropic_response([failing], stop_reason='tool_use'), anthropic_response([])],
    )
    session = FakeSession()
    AnthropicComputerAgent('test').run(session, session.observation, task='Task', budget=Budget())
    follow_up = client.requests[1]['messages'][2]['content']
    result = follow_up[0]
    assert result['type'] == 'tool_result' and result['is_error'] is True
    assert isinstance(result['content'], str)
    assert follow_up[-1]['type'] == 'image'


@pytest.mark.parametrize('effort', ['high', None])
def test_anthropic_requests_cache_the_prefix_and_set_thinking(sdk, effort):
    client = sdk('anthropic', [anthropic_response([])])
    session = FakeSession()
    AnthropicComputerAgent('test', effort=effort).run(
        session, session.observation, task='Task', budget=Budget()
    )
    request = client.requests[0]
    assert request['cache_control'] == {'type': 'ephemeral'}
    if effort:
        assert request['thinking']['type'] == 'adaptive'
        assert request['output_config'] == {'effort': 'high'}
    else:
        assert 'thinking' not in request and 'output_config' not in request


def _sdk_transport():
    """The HTTP package the installed anthropic SDK uses (httpx, or httpx2 from 1.15 on)."""
    base = importlib.import_module('anthropic._base_client')
    return getattr(base, 'httpx2', None) or base.httpx


def _sse(*events: dict) -> bytes:
    return b''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n'.encode() for e in events)


START = {
    'type': 'message_start',
    'message': {
        'id': 'msg',
        'type': 'message',
        'role': 'assistant',
        'model': 'test',
        'content': [],
        'stop_reason': None,
        'stop_sequence': None,
        'usage': {'input_tokens': 100, 'output_tokens': 1},
    },
}
OUT = {'output_tokens': 5}
CLICK_EVENTS = (
    {
        'type': 'content_block_start',
        'index': 0,
        'content_block': {'type': 'tool_use', 'id': 'c1', 'name': 'left_click', 'input': {}},
    },
    {
        'type': 'content_block_delta',
        'index': 0,
        'delta': {'type': 'input_json_delta', 'partial_json': '{"coordinate": [10, 20]}'},
    },
    {'type': 'content_block_stop', 'index': 0},
)


DELTA = {
    'type': 'message_delta',
    'delta': {'stop_reason': 'tool_use'},
    'usage': {'output_tokens': 5},
}


@pytest.mark.parametrize('failure', ['eof', 'disconnect', 'eof_after_delta', 'empty'])
def test_anthropic_unfinished_stream_never_acts(monkeypatch, failure):
    anthropic = pytest.importorskip('anthropic')
    httpx = _sdk_transport()
    body = {  # none of these streams reaches message_stop
        'eof': _sse(START, *CLICK_EVENTS),
        'disconnect': _sse(START, *CLICK_EVENTS),
        'eof_after_delta': _sse(START, *CLICK_EVENTS, DELTA),
        'empty': b'',
    }[failure]

    class Cut(httpx.SyncByteStream):
        def __iter__(self):
            yield body
            if failure == 'disconnect':
                raise httpx.RemoteProtocolError('peer closed connection')

    def handler(request):
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Cut())

    real = anthropic.Anthropic

    def client(**kwargs):
        transport = httpx.MockTransport(handler)
        return real(api_key='test', http_client=httpx.Client(transport=transport), max_retries=0)

    monkeypatch.setattr(anthropic, 'Anthropic', client)
    session = FakeSession()
    result = AnthropicComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget()
    )
    assert result.termination_reason == 'error'
    assert not result.trajectory.steps  # the half-received click was not executed
    assert result.usage.input_tokens == (
        0 if failure == 'empty' else 100
    )  # observed usage kept once
    assert result.cost.usd is None and 'billing unresolved' in result.cost.provenance


def test_anthropic_completed_stream_acts(monkeypatch):
    anthropic = pytest.importorskip('anthropic')
    httpx = _sdk_transport()
    done = (
        {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}, 'usage': OUT},
        {'type': 'message_stop'},
    )
    text = (
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Ok'}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': OUT},
        {'type': 'message_stop'},
    )
    bodies = iter([_sse(START, *CLICK_EVENTS, *done), _sse(START, *text)])

    def handler(request):
        return httpx.Response(
            200, headers={'content-type': 'text/event-stream'}, content=next(bodies)
        )

    real = anthropic.Anthropic

    def client(**kwargs):
        transport = httpx.MockTransport(handler)
        return real(api_key='test', http_client=httpx.Client(transport=transport), max_retries=0)

    monkeypatch.setattr(anthropic, 'Anthropic', client)
    session = FakeSession()
    result = AnthropicComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget()
    )
    assert result.termination_reason == 'completed'
    assert [action.action_type for _, action in result.trajectory.steps] == ['click']


def test_unlimited_budget_sends_no_deadline(sdk):
    client = sdk('anthropic', [anthropic_response([{'type': 'text', 'text': 'Done'}])])
    session = FakeSession()
    result = AnthropicComputerAgent('test').run(
        session, session.observation, task='Task', budget=Budget(max_turns=None, max_seconds=None)
    )
    assert result.termination_reason == 'completed'
    assert client.requests[0]['timeout'] is None
