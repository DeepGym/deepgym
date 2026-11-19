"""API tool agents: real SDK requests over mock transports, tool calls against a loopback app.

No provider is called and no sandbox is created: the SDKs talk to in-memory transports and
each tool call runs through a local Session to a real HTTP server on 127.0.0.1.
"""

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from deepgym.agents import AnthropicToolAgent, Budget, OpenAIToolAgent, TokenPrices, Usage, api
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
        grade = session.score()

    # The provider's stream was cut: nothing the policy did, so no verdict, never a 0.
    assert result.termination_reason == 'provider_error'
    assert not grade.has_verdict and grade.error_type == 'provider_error'
    assert not result.trajectory.steps and app.requests == []  # the received call never ran
    assert result.usage.input_tokens == 100  # the partial usage is kept
    assert result.cost.usd is None and 'billing unresolved' in result.cost.provenance
    partial = result.messages[-1]  # what arrived is kept, marked incomplete
    assert partial.role == 'assistant'
    payload = json.loads(partial.content)
    assert payload['status'] == 'incomplete'
    assert payload['incomplete_details']['reason'] == result.error
    assert payload['usage']['input_tokens'] == 100
    assert payload['content'][0]['name'] == 'tracker'


@pytest.mark.parametrize('finished', [True, False])
def test_anthropic_tool_calls_cut_short_never_run(app, monkeypatch, finished):
    raw = '{"method": "POST", "path": "/rest/bug/1/comment", "body": {"comment": "unfini'
    events = [
        START,
        {'type': 'content_block_start', 'index': 0, 'content_block': use('tracker', {})},
        {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {'type': 'input_json_delta', 'partial_json': raw},
        },
    ]
    if finished:  # a whole message, stopped at the output limit mid-call
        events += [
            {'type': 'content_block_stop', 'index': 0},
            {
                'type': 'message_delta',
                'delta': {'stop_reason': 'max_tokens'},
                'usage': {'output_tokens': 5},
            },
            {'type': 'message_stop'},
        ]
    anthropic(
        monkeypatch,
        b''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n'.encode() for e in events),
    )
    session, initial = started(app)
    with session:
        result = AnthropicToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)

    # Stopped at the model's own output limit: the policy's, scored. Cut: the provider's.
    assert result.termination_reason == ('error' if finished else 'provider_error')
    assert app.requests == [] and not result.trajectory.steps  # the cut call never ran
    payload = json.loads(result.messages[-1].content)
    assert payload['input_json'] == {'0': raw}  # every fragment received, kept
    if finished:
        assert payload['stop_reason'] == 'max_tokens' and 'max_tokens' in result.error
        assert payload['usage']['output_tokens'] == 5 and result.usage.output_tokens == 5
    else:
        assert payload['status'] == 'incomplete'


def test_anthropic_overloaded_mid_stream_is_no_verdict(app, monkeypatch):
    """A provider error event arrives under the stream's 200; it is still the provider's."""
    error = {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}
    body = f'event: message_start\ndata: {json.dumps(START)}\n\n'
    body += f'event: error\ndata: {json.dumps(error)}\n\n'
    anthropic(monkeypatch, body.encode())
    session, initial = started(app)
    with session:
        result = AnthropicToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)
        grade = session.score()

    assert result.termination_reason == 'provider_error'
    assert not grade.has_verdict and grade.error_type == 'provider_error'


def test_anthropic_delta_the_sdk_cannot_parse_is_kept_and_never_runs(app, monkeypatch):
    first = '{"method": "POST", "path": "/rest/bug/1/comment", '
    events = [
        START,
        {'type': 'content_block_start', 'index': 0, 'content_block': use('tracker', {})},
        {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {'type': 'input_json_delta', 'partial_json': first},
        },
        {
            'type': 'content_block_delta',
            'index': 0,
            'delta': {'type': 'input_json_delta', 'partial_json': 'BROKEN_LAST_DELTA_MARKER}'},
        },
        {'type': 'content_block_stop', 'index': 0},
        {
            'type': 'message_delta',
            'delta': {'stop_reason': 'tool_use'},
            'usage': {'output_tokens': 5},
        },
        {'type': 'message_stop'},
    ]
    anthropic(
        monkeypatch,
        b''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n'.encode() for e in events),
    )
    session, initial = started(app)
    with session:
        result = AnthropicToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)

    assert result.termination_reason == 'error'
    assert app.requests == [] and not result.trajectory.steps  # nothing it held ran
    payload = json.loads(result.messages[-1].content)
    assert payload['status'] == 'incomplete'
    assert 'BROKEN_LAST_DELTA_MARKER' in payload['incomplete_details']['detail']


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


def _traced(app: App, path: Path) -> TracedSession:
    """A traced local session of the loopback app's environment, written to *path*."""
    env = app_env(app.url)
    provenance = Provenance(
        env=EnvFingerprint.of(env),
        agent=AgentMeta(name='api-tool', model='test'),
        provider='local',
        deepgym_version='test',
        started_at=datetime.now(timezone.utc),
    )
    return TracedSession(Session(env, LocalBackend()), env, provenance, path)


def test_traced_steps_and_result_carry_every_message(app, monkeypatch, tmp_path):
    anthropic(
        monkeypatch,
        stream(use('tracker', GET_BUG)),
        stream(text('Triaged.'), use('done', {}, 'call-2')),
    )
    with _traced(app, tmp_path / 't.jsonl') as tr:
        initial = tr.start()
        result = AnthropicToolAgent('test').run(tr, initial, task='Triage', budget=UNLIMITED)
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


PRICES = TokenPrices(
    input_per_million=3,
    output_per_million=15,
    cache_read_per_million=0.3,
    cache_write_per_million=3.75,
    source='test rates',
)
REPLY = Usage(input_tokens=100, output_tokens=5)  # what each fake Anthropic response reports
REPLY_COST = PRICES.estimate(REPLY).usd


def test_traced_usage_is_on_each_replys_first_action_and_sums_to_the_result(
    app, monkeypatch, tmp_path
):
    anthropic(
        monkeypatch,
        stream(use('tracker', GET_BUG), use('tracker', GET_BUG, 'call-2')),
        stream(use('done', {}, 'call-3')),
    )
    with _traced(app, tmp_path / 't.jsonl') as tr:
        initial = tr.start()
        result = AnthropicToolAgent('test', PRICES).run(tr, initial, task='T', budget=UNLIMITED)
        tr.score()

    trace = load_trace(tmp_path / 't.jsonl')
    assert [step.action.action_type for step in trace.steps] == ['tool', 'tool', 'done']
    # Two calls of one reply: its usage is on the first only, so nothing counts twice.
    assert [(step.usage, step.cost_usd) for step in trace.steps] == [
        (REPLY, REPLY_COST),
        (None, None),
        (REPLY, REPLY_COST),
    ]
    assert trace.unattached_usage is None  # every response led to an action
    assert trace.usage == result.usage == REPLY + REPLY
    assert trace.cost_usd == result.cost.usd


@pytest.mark.parametrize(
    'last,budget,reason',
    [
        # A closing answer without done: reminded, then the turn budget ends the run.
        (
            stream(text('Triaged.'), stop='end_turn'),
            Budget(max_turns=2, max_seconds=None),
            'turn_budget',
        ),
        (stream(text('I will not.'), stop='refusal'), UNLIMITED, 'refusal'),
        # The reply that crosses the USD cap: its call never runs.
        (
            stream(use('tracker', GET_BUG, 'call-2')),
            Budget(max_turns=None, max_seconds=None, max_cost_usd=1.5 * REPLY_COST),
            'cost_budget',
        ),
    ],
)
def test_traced_usage_totals_are_the_agents_when_the_last_reply_takes_no_action(
    app, monkeypatch, tmp_path, last, budget, reason
):
    anthropic(monkeypatch, stream(use('tracker', GET_BUG)), last)
    with _traced(app, tmp_path / 't.jsonl') as tr:
        initial = tr.start()
        result = AnthropicToolAgent('test', PRICES).run(tr, initial, task='T', budget=budget)
        tr.score()

    trace = load_trace(tmp_path / 't.jsonl')
    assert result.termination_reason == reason
    assert [(step.usage, step.cost_usd) for step in trace.steps] == [(REPLY, REPLY_COST)]
    # The last reply is on no step: it is recorded outside them, and counted.
    assert (trace.unattached_usage, trace.unattached_cost_usd) == (REPLY, REPLY_COST)
    assert trace.usage == result.usage == REPLY + REPLY
    assert trace.cost_usd == result.cost.usd


def test_a_run_without_model_calls_has_no_trace_usage(app, monkeypatch, tmp_path):
    sent = anthropic(monkeypatch)
    with _traced(app, tmp_path / 't.jsonl') as tr:
        initial = tr.start()
        budget = Budget(max_turns=None, max_seconds=None, max_cost_usd=0)
        result = AnthropicToolAgent('test', PRICES).run(tr, initial, task='T', budget=budget)

    trace = load_trace(tmp_path / 't.jsonl')
    assert (result.termination_reason, sent) == ('cost_budget', [])
    # Nothing was reported: the trace says unknown, while the agent's result counts zeros.
    assert (trace.usage, trace.cost_usd) == (None, None)
    assert (result.usage, result.cost.usd) == (Usage(reasoning_tokens=0), 0)


def test_traced_usage_survives_an_uncaught_error(app, monkeypatch, tmp_path):
    anthropic(monkeypatch, stream(text('Looking.'), stop='end_turn'))
    real = api.anthropic_message
    calls: list[int] = []

    def fails_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError('disk full')
        return real(*args, **kwargs)

    monkeypatch.setattr(api, 'anthropic_message', fails_second)
    with pytest.raises(OSError, match='disk full'), _traced(app, tmp_path / 't.jsonl') as tr:
        AnthropicToolAgent('test', PRICES).run(tr, tr.start(), task='T', budget=UNLIMITED)

    trace = load_trace(tmp_path / 't.jsonl')
    # The reply before the error took no action; the agent's finally recorded it, and the
    # request in flight when the error escaped left the usage incomplete.
    assert (trace.steps, trace.unattached_usage, trace.unattached_cost_usd) == (
        (),
        REPLY,
        REPLY_COST,
    )
    assert trace.usage_complete is False
    assert (trace.usage, trace.cost_usd) == (REPLY, None)


def test_a_failed_request_marks_trace_usage_incomplete(app, monkeypatch, tmp_path):
    def respond(httpx, index):
        if index == 0:
            headers = {'content-type': 'text/event-stream'}
            return httpx.Response(200, headers=headers, content=stream(use('tracker', GET_BUG)))
        error = {'type': 'error', 'error': {'type': 'api_error', 'message': 'boom'}}
        return httpx.Response(500, json=error)

    _install(monkeypatch, 'anthropic', respond)
    with _traced(app, tmp_path / 't.jsonl') as tr:
        initial = tr.start()
        result = AnthropicToolAgent('test', PRICES).run(tr, initial, task='T', budget=UNLIMITED)

    trace = load_trace(tmp_path / 't.jsonl')
    assert result.termination_reason == 'provider_error'
    # The failed request's usage is unknown: not counted as 0, and flagged.
    assert trace.usage_complete is False
    assert trace.usage == result.usage == REPLY  # a lower bound
    assert trace.cost_usd is None and result.cost.usd is None
    assert trace.unattached_usage is None


def test_traced_usage_of_a_reply_without_calls_goes_on_the_next_action(app, monkeypatch, tmp_path):
    openai(monkeypatch, response(message('Looking.')), response(function_call('done', {})))
    with _traced(app, tmp_path / 't.jsonl') as tr:
        initial = tr.start()
        result = OpenAIToolAgent('test').run(tr, initial, task='T', budget=UNLIMITED)
        tr.score()

    (done,) = load_trace(tmp_path / 't.jsonl').steps
    # Each response: 100 input of which 20 cached, 10 output of which 4 reasoning.
    reply = Usage(input_tokens=80, output_tokens=10, cache_read_tokens=20, reasoning_tokens=4)
    assert [message.role for message in done.messages].count('assistant') == 2
    assert done.usage == result.usage == reply + reply
    assert done.cost_usd is None  # no prices: unknown, never a guess


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


@pytest.mark.parametrize('usage', [None, 'counted'])
def test_openai_failed_response_is_no_verdict_with_or_without_usage(app, monkeypatch, usage):
    """A failed response may carry no usage; it is still the provider's failure, never a 0."""
    body = response(
        function_call('tracker', GET_BUG),
        status='failed',
        error={'code': 'server_error', 'message': 'The server had an error'},
    )
    if usage is None:
        body['usage'] = None
    openai(monkeypatch, body)
    session, initial = started(app)
    with session:
        result = OpenAIToolAgent('test').run(session, initial, task='T', budget=UNLIMITED)
        grade = session.score()

    assert result.termination_reason == 'provider_error'
    assert not grade.has_verdict and grade.error_type == 'provider_error'
    assert app.requests == [] and not result.trajectory.steps  # the failed call never ran
    assert 'billing unresolved' in result.cost.provenance
