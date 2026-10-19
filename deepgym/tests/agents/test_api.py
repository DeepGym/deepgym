"""API tool agents: real SDK requests over mock transports, tool calls against a loopback app.

No provider is called and no sandbox is created: the SDKs talk to in-memory transports and
each tool call runs through a local Session to a real HTTP server on 127.0.0.1.
"""

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from deepgym.agents import AnthropicToolAgent, Budget, OpenAIToolAgent, TokenPrices
from deepgym.agents.api import _NO_CALL, _SKIPPED
from deepgym.agents.providers import ANTHROPIC_MAX_TOKENS, SDK_RETRIES
from deepgym.models import MultiTurnEnvironment
from deepgym.session import LocalBackend, Session
from deepgym.traces import AgentMeta, EnvFingerprint, Provenance, TracedSession, load_trace
from tests.http_app import API_KEY, BIG, App, app_env, app_session

UNLIMITED = Budget(max_turns=None, max_seconds=None)
NO_TIMEOUT = {'connect': None, 'read': None, 'write': None, 'pool': None}


def _transport(sdk: str):
    """The HTTP package the installed SDK uses (httpx, or httpx2 in newer releases)."""
    base = importlib.import_module(f'{sdk}._base_client')
    return getattr(base, 'httpx2', None) or base.httpx


def _install(monkeypatch, sdk: str, respond) -> list:
    """Route the SDK's client through a mock transport; return the requests it sends."""
    module = pytest.importorskip(sdk)
    httpx = _transport(sdk)
    sent: list = []

    def handler(request):
        sent.append({'body': json.loads(request.content), 'timeout': request.extensions['timeout']})
        return respond(httpx, len(sent) - 1)

    name = 'Anthropic' if sdk == 'anthropic' else 'OpenAI'
    real = getattr(module, name)

    def client(**kwargs):
        assert kwargs == {'max_retries': SDK_RETRIES}
        transport = httpx.MockTransport(handler)
        return real(api_key='test', http_client=httpx.Client(transport=transport), max_retries=0)

    monkeypatch.setattr(module, name, client)
    return sent


# -- Anthropic: streamed messages ------------------------------------------------------------

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


def text(value: str) -> dict:
    return {'type': 'text', 'text': value}


def use(name: str, arguments: dict, ident: str = 'call-1') -> dict:
    return {'type': 'tool_use', 'id': ident, 'name': name, 'input': arguments}


def stream(*blocks: dict, stop: str = 'tool_use', finished: bool = True) -> bytes:
    """An Anthropic SSE body with *blocks*, ending at message_stop when *finished*."""
    events: list[dict] = [START]
    for index, block in enumerate(blocks):
        if block['type'] == 'text':
            start = {'type': 'text', 'text': ''}
            delta = {'type': 'text_delta', 'text': block['text']}
        else:
            start = {**block, 'input': {}}
            delta = {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}
        events += [
            {'type': 'content_block_start', 'index': index, 'content_block': start},
            {'type': 'content_block_delta', 'index': index, 'delta': delta},
            {'type': 'content_block_stop', 'index': index},
        ]
    events.append(
        {'type': 'message_delta', 'delta': {'stop_reason': stop}, 'usage': {'output_tokens': 5}}
    )
    if finished:
        events.append({'type': 'message_stop'})
    return b''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n'.encode() for e in events)


def anthropic(monkeypatch, *bodies: bytes) -> list:
    def respond(httpx, index):
        headers = {'content-type': 'text/event-stream'}
        return httpx.Response(200, headers=headers, content=bodies[index])

    return _install(monkeypatch, 'anthropic', respond)


# -- OpenAI: Responses API -------------------------------------------------------------------


def function_call(name: str, arguments: dict | str, ident: str = 'call-1') -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        'type': 'function_call',
        'id': f'fc-{ident}',
        'call_id': ident,
        'name': name,
        'arguments': raw,
        'status': 'completed',
    }


def message(value: str) -> dict:
    return {
        'type': 'message',
        'id': 'msg-1',
        'role': 'assistant',
        'status': 'completed',
        'content': [{'type': 'output_text', 'text': value, 'annotations': []}],
    }


def response(*output: dict, status: str = 'completed', **extra: object) -> dict:
    return {
        'id': 'resp-1',
        'object': 'response',
        'created_at': 0,
        'model': 'test',
        'status': status,
        'output': list(output),
        'usage': {
            'input_tokens': 100,
            'output_tokens': 10,
            'total_tokens': 110,
            'input_tokens_details': {'cached_tokens': 20},
            'output_tokens_details': {'reasoning_tokens': 4},
        },
        **extra,
    }


def openai(monkeypatch, *bodies: dict) -> list:
    def respond(httpx, index):
        return httpx.Response(200, json={**bodies[index], 'id': f'resp-{index + 1}'})

    return _install(monkeypatch, 'openai', respond)


GET_BUG = {'method': 'GET', 'path': '/rest/bug/1'}


def started(app: App) -> tuple[Session, object]:
    session = app_session(app.url)
    return session, session.start()


def test_anthropic_tool_call_returns_its_full_result_then_done_completes(app, monkeypatch):
    sent = anthropic(
        monkeypatch,
        stream(use('tracker', GET_BUG)),
        stream(text('Triaged.'), use('done', {}, 'call-2')),
    )
    session, initial = started(app)
    with session:
        result = AnthropicToolAgent('test').run(session, initial, task='Triage', budget=UNLIMITED)

    assert result.termination_reason == 'completed'
    assert result.final_text == 'Triaged.'
    assert [action.action_type for _, action in result.trajectory.steps] == ['tool', 'done']
    assert session.done
    observation = result.trajectory.steps[1][0]  # what the tool call produced
    (tool_result,) = sent[1]['body']['messages'][2]['content']
    assert tool_result == {
        'type': 'tool_result',
        'tool_use_id': 'call-1',
        'content': observation.content,
    }
    assert json.loads(json.loads(observation.content)['response']['body'])['bugs'][0][
        'summary'
    ] == BIG  # fmt: skip
    assert [request['method'] for request in app.requests] == ['GET']
    assert result.usage.input_tokens == 200 and result.turns == 2


def test_openai_tool_call_returns_its_full_result_then_done_completes(app, monkeypatch):
    sent = openai(
        monkeypatch,
        response(function_call('tracker', GET_BUG)),
        response(message('Triaged.'), function_call('done', {}, 'call-2')),
    )
    session, initial = started(app)
    with session:
        result = OpenAIToolAgent('test').run(session, initial, task='Triage', budget=UNLIMITED)

    assert result.termination_reason == 'completed'
    assert result.final_text == 'Triaged.'
    assert [action.action_type for _, action in result.trajectory.steps] == ['tool', 'done']
    observation = result.trajectory.steps[1][0]
    assert sent[1]['body']['previous_response_id'] == 'resp-1'
    assert sent[1]['body']['input'] == [
        {'type': 'function_call_output', 'call_id': 'call-1', 'output': observation.content}
    ]
    assert BIG in observation.content
    assert result.usage.input_tokens == 160 and result.usage.cache_read_tokens == 40


def test_requests_carry_only_declared_tools_and_done(app, monkeypatch):
    claude = anthropic(monkeypatch, stream(use('done', {})))
    gpt = openai(monkeypatch, response(function_call('done', {})))
    for agent in (AnthropicToolAgent('test'), OpenAIToolAgent('test')):
        session, initial = started(app)
        with session:
            assert agent.run(session, initial, task='T', budget=UNLIMITED).termination_reason == (
                'completed'
            )

    claude_body, gpt_body = claude[0]['body'], gpt[0]['body']
    assert [tool['name'] for tool in claude_body['tools']] == ['tracker', 'done']
    assert all(
        set(tool) == {'name', 'description', 'input_schema'} for tool in claude_body['tools']
    )
    assert set(claude_body) <= {
        'model', 'max_tokens', 'cache_control', 'thinking', 'output_config', 'system', 'tools',
        'messages', 'stream',
    }  # fmt: skip
    assert [(tool['type'], tool['name']) for tool in gpt_body['tools']] == [
        ('function', 'tracker'),
        ('function', 'done'),
    ]
    assert set(gpt_body) <= {
        'model', 'instructions', 'tools', 'input', 'reasoning', 'previous_response_id',
    }  # fmt: skip
    for body in (claude_body, gpt_body):
        text_of = json.dumps(body)
        for hosted in ('web_search', 'web_fetch', 'code_execution', 'computer', 'mcp', 'bash'):
            assert hosted not in text_of
        assert API_KEY not in text_of  # the app's credential never goes to the provider


def test_defaults_are_uncapped(app, monkeypatch):
    claude = anthropic(monkeypatch, stream(use('done', {})))
    gpt = openai(monkeypatch, response(function_call('done', {})))
    for agent in (AnthropicToolAgent('test'), OpenAIToolAgent('test')):
        session, initial = started(app)
        with session:
            agent.run(session, initial, task='T', budget=UNLIMITED)

    claude_body = claude[0]['body']
    assert claude_body['max_tokens'] == ANTHROPIC_MAX_TOKENS
    assert claude_body['thinking'] == {'type': 'adaptive', 'display': 'summarized'}
    assert claude_body['output_config'] == {'effort': 'max'}
    assert gpt[0]['body']['reasoning'] == {'effort': 'max'}
    assert 'max_output_tokens' not in gpt[0]['body']
    assert claude[0]['timeout'] == NO_TIMEOUT and gpt[0]['timeout'] == NO_TIMEOUT


def test_injection_shaped_calls_are_rejected_and_the_rest_of_the_batch_skipped(app, monkeypatch):
    hostile = {'method': 'GET', 'path': '/rest/bug/1;touch pwned', 'query': {'q': '$(id)'}}
    sent = anthropic(
        monkeypatch,
        stream(use('tracker', hostile), use('tracker', GET_BUG, 'call-2'), use('sh', {}, 'c3')),
        stream(use('done', {})),
    )
    session, initial = started(app)
    with session:
        workdir = Path(session._backend._dir)
        result = AnthropicToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)
        assert not list(workdir.glob('pwned*'))

    results = sent[1]['body']['messages'][2]['content']
    assert results[0]['is_error'] and results[0]['content'].startswith('Tool call rejected')
    assert [r['content'] for r in results[1:]] == [_SKIPPED, _SKIPPED]
    assert all(r['is_error'] for r in results)
    assert app.requests == []
    assert result.termination_reason == 'completed'
    assert [action.action_type for _, action in result.trajectory.steps] == ['done']


def test_openai_non_json_arguments_are_rejected(app, monkeypatch):
    sent = openai(
        monkeypatch,
        response(function_call('tracker', '"; rm -rf / #')),
        response(function_call('done', {})),
    )
    session, initial = started(app)
    with session:
        OpenAIToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)

    (output,) = sent[1]['body']['input']
    assert output['output'].startswith('Tool call rejected: arguments must be a JSON object')
    assert app.requests == []


@pytest.mark.parametrize('failure', ['eof', 'disconnect', 'eof_after_delta'])
def test_anthropic_unfinished_stream_never_acts(app, monkeypatch, failure):
    anthropic_sdk = pytest.importorskip('anthropic')
    httpx = _transport('anthropic')
    body = stream(use('tracker', GET_BUG), finished=False)
    if failure != 'eof_after_delta':
        body = body.rsplit(b'event: message_delta', 1)[0]

    class Cut(httpx.SyncByteStream):
        def __iter__(self):
            yield body
            if failure == 'disconnect':
                raise httpx.RemoteProtocolError('peer closed connection')

    real = anthropic_sdk.Anthropic

    def client(**kwargs):
        def handler(request):
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Cut())

        transport = httpx.MockTransport(handler)
        return real(api_key='test', http_client=httpx.Client(transport=transport), max_retries=0)

    monkeypatch.setattr(anthropic_sdk, 'Anthropic', client)
    session, initial = started(app)
    with session:
        result = AnthropicToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)

    assert result.termination_reason == 'error'
    assert not result.trajectory.steps and app.requests == []  # the received call never ran
    assert result.usage.input_tokens == 100  # the partial usage is kept
    assert result.cost.usd is None and 'billing unresolved' in result.cost.provenance
    partial = result.messages[-1]  # what arrived is kept, marked incomplete
    assert partial.role == 'assistant'
    payload = json.loads(partial.content)
    assert payload['status'] == 'incomplete'
    assert payload['incomplete_details'] == {'reason': result.error}
    assert payload['usage']['input_tokens'] == 100
    assert payload['content'][0]['name'] == 'tracker'


@pytest.mark.parametrize('details', [{'reason': 'max_output_tokens'}, None])
def test_openai_incomplete_response_never_acts(app, monkeypatch, details):
    body = response(function_call('tracker', GET_BUG), status='incomplete')
    if details:
        body['incomplete_details'] = details
    sent = openai(monkeypatch, body)
    session, initial = started(app)
    with session:
        result = OpenAIToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)

    assert result.termination_reason == 'error'
    assert (
        result.error == f'Response did not complete: {(details or {}).get("reason", "incomplete")}'
    )
    assert len(sent) == 1 and not result.trajectory.steps and app.requests == []
    assert result.usage.output_tokens == 10
    payload = json.loads(result.messages[-1].content)  # what arrived is kept, marked incomplete
    assert payload['status'] == 'incomplete' and payload['output'][0]['name'] == 'tracker'


def test_reply_without_tool_calls_is_reminded_not_completed(app, monkeypatch):
    sent = anthropic(
        monkeypatch, stream(text('All done, I think.'), stop='end_turn'), stream(use('done', {}))
    )
    session, initial = started(app)
    with session:
        result = AnthropicToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)

    assert sent[1]['body']['messages'][-1] == {'role': 'user', 'content': _NO_CALL}
    assert result.termination_reason == 'completed' and result.turns == 2


def test_done_text_in_a_tool_result_does_not_end_the_episode(app, monkeypatch):
    sent = openai(
        monkeypatch,
        response(function_call('tracker', {'method': 'GET', 'path': '/rest/done'})),
        response(function_call('done', {})),
    )
    session, initial = started(app)
    with session:
        result = OpenAIToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)

    assert 'DONE' in sent[1]['body']['input'][0]['output']
    assert len(sent) == 2
    assert [action.action_type for _, action in result.trajectory.steps] == ['tool', 'done']


def test_lost_write_is_recorded_uncertain_and_never_retried(app, monkeypatch):
    write = {'method': 'POST', 'path': '/rest/lost', 'body': {'status': 'RESOLVED'}}
    sent = openai(
        monkeypatch, response(function_call('tracker', write)), response(function_call('done', {}))
    )
    session, initial = started(app)
    with session:
        result = OpenAIToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)

    record = json.loads(sent[1]['body']['input'][0]['output'])
    assert record['uncertain'] is True and record['request']['body'] == write['body']
    assert [request['method'] for request in app.requests] == ['POST']
    assert result.termination_reason == 'completed'


def test_traced_steps_and_result_carry_every_message(app, monkeypatch, tmp_path):
    anthropic(
        monkeypatch,
        stream(use('tracker', GET_BUG)),
        stream(text('Triaged.'), use('done', {}, 'call-2')),
    )
    env = app_env(app.url)
    provenance = Provenance(
        env=EnvFingerprint.of(env),
        agent=AgentMeta(name='api-tool', model='test'),
        provider='local',
        deepgym_version='test',
        started_at=datetime.now(timezone.utc),
    )
    with TracedSession(Session(env, LocalBackend()), env, provenance, tmp_path / 't.jsonl') as tr:
        initial = tr.start()
        result = AnthropicToolAgent('test').run(tr, initial, task=env.task, budget=UNLIMITED)
        tr.score()

    assert API_KEY not in (tmp_path / 't.jsonl').read_text()
    assert all(API_KEY not in message.content for message in result.messages)
    trace = load_trace(tmp_path / 't.jsonl')
    first, done = trace.steps
    assert [m.role for m in first.messages] == ['system', 'user', 'assistant']
    assert json.loads(first.messages[-1].content)['usage']['input_tokens'] == 100
    assert first.action.tool == 'tracker' and first.action.arguments == GET_BUG
    assert BIG in first.observation.content
    assert [m.role for m in done.messages] == ['tool', 'assistant']
    assert done.action.action_type == 'done'
    assert result.messages == (*first.messages, *done.messages)
    assert json.loads(done.messages[0].content)['content'] == first.observation.content


def test_cost_cap_stops_before_running_returned_calls_and_keeps_the_reply(app, monkeypatch):
    prices = TokenPrices(
        input_per_million=100,
        output_per_million=100,
        cache_read_per_million=100,
        cache_write_per_million=100,
        source='test rates',
    )
    anthropic(monkeypatch, stream(use('tracker', GET_BUG)))
    session, initial = started(app)
    with session:
        result = AnthropicToolAgent('test', prices).run(
            session, initial, task='T', budget=Budget(max_cost_usd=0.001)
        )

    assert result.termination_reason == 'cost_budget'
    assert app.requests == [] and not result.trajectory.steps
    assert json.loads(result.messages[-1].content)['content'][0]['name'] == 'tracker'


def test_turn_budget_is_the_callers(app, monkeypatch):
    sent = anthropic(monkeypatch, stream(use('tracker', GET_BUG)))
    session, initial = started(app)
    with session:
        result = AnthropicToolAgent('test').run(
            session, initial, task='T', budget=Budget(max_turns=1, max_seconds=None)
        )

    assert result.termination_reason == 'turn_budget' and len(sent) == 1
    assert [action.action_type for _, action in result.trajectory.steps] == ['tool']


@pytest.mark.parametrize('agent', [AnthropicToolAgent, OpenAIToolAgent])
def test_environment_without_tools_is_refused(agent):
    session = Session(MultiTurnEnvironment(task='T'), LocalBackend())
    with session, pytest.raises(ValueError, match='declares tools'):
        agent('test').run(session, session.start(), task='T', budget=UNLIMITED)
