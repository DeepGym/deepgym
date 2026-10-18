"""Tests for declared HTTP API tools run as Session tool actions, against a loopback app."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym.exceptions import ActionError
from deepgym.models import Action, MultiTurnEnvironment
from deepgym.qa.agents import do_everything, random_agent
from deepgym.sandboxes import ExecResult
from deepgym.session import Session
from deepgym.tools import HttpRoute, HttpTool, run_command
from tests.http_app import API_KEY, BIG, App, app_env, app_session, tool


def call(path: str, method: str = 'GET', **arguments: object) -> Action:
    return Action(
        action_type='tool', tool='tracker', arguments={'method': method, 'path': path, **arguments}
    )


def record(session: Session, action: Action) -> dict:
    return json.loads(session.step(action).observation.content)


def test_call_returns_the_full_request_and_response(app: App) -> None:
    with app_session(app.url) as session:
        session.start()

        step = session.step(call('/rest/bug/1', query={'include': ['id', 'summary'], 'n': 2}))

        result = json.loads(step.observation.content)
        assert step.observation.metadata == {'exit_code': 0}
        assert result['request'] == {
            'method': 'GET',
            'url': f'{app.url}/rest/bug/1?include=id&include=summary&n=2',
            'headers': {'X-API-KEY': '[redacted]'},
        }
        response = result['response']
        assert response['status'] == 200
        assert json.loads(response['body'])['bugs'][0]['summary'] == BIG  # never clipped
        assert ['Link', f'<{app.url}/rest/bug?page=2>; rel="next"'] in response['headers']
        assert result['elapsed_ms'] >= 0
        assert API_KEY not in step.observation.content
        (seen,) = app.requests
        assert seen['headers']['x-api-key'] == API_KEY
        assert seen['path'] == '/rest/bug/1?include=id&include=summary&n=2'
        assert [a.action_type for _, a in session.trajectory.steps] == ['tool']


def test_write_sends_its_json_body(app: App) -> None:
    with app_session(app.url) as session:
        session.start()

        result = record(session, call('/rest/bug/1/comment', 'POST', body={'comment': 'ok'}))

        assert result['response']['status'] == 201
        assert result['request']['body'] == {'comment': 'ok'}
        assert result['request']['headers']['Content-Type'] == 'application/json'
        assert json.loads(app.requests[0]['body']) == {'comment': 'ok'}


def test_binary_body_comes_back_whole_as_base64(app: App) -> None:
    with app_session(app.url) as session:
        session.start()
        assert record(session, call('/rest/binary'))['response']['body_base64'] == '/wD+'


@pytest.mark.parametrize(
    'arguments',
    [
        {'method': 'GET', 'path': 'http://evil.example/rest/bug/1'},
        {'method': 'GET', 'path': '//evil.example/rest/bug/1'},
        {'method': 'GET', 'path': '/rest/bug/1?x=1'},
        {'method': 'GET', 'path': '/rest/bug/1#frag'},
        {'method': 'GET', 'path': '/rest/bug/../../admin'},
        {'method': 'GET', 'path': '/rest/bug/%2e%2e'},
        {'method': 'GET', 'path': '/rest/bug/a%2F..%2F..%2Fadmin'},
        {'method': 'GET', 'path': '/rest/bug/a%5Cb'},
        {'method': 'GET', 'path': '/rest/bug/1 2'},
        {'method': 'GET', 'path': '/admin/run_script'},
        {'method': 'GET', 'path': '/rest/bug/1/'},
        {'method': 'DELETE', 'path': '/rest/bug/1'},
        {'method': 'GET', 'path': '/rest/bug/1', 'body': {'a': 1}},
        {'method': 'POST', 'path': '/rest/bug/1/comment', 'body': 'text'},
        {'method': 'GET', 'path': '/rest/bug/1', 'headers': {'Host': 'evil.example'}},
        {'method': 'GET', 'path': '/rest/bug/1', 'url': 'http://evil.example/'},
        {'method': 'GET', 'path': '/rest/bug/1', 'query': {'q': {'nested': 1}}},
        {'method': 'GET'},
    ],
)
def test_off_origin_or_undeclared_calls_are_rejected_unsent(app: App, arguments: dict) -> None:
    with app_session(app.url) as session:
        session.start()

        with pytest.raises(ActionError, match='rejected the call'):
            session.step(Action(action_type='tool', tool='tracker', arguments=arguments))

        assert app.requests == []
        assert not session.done
        assert session.trajectory.steps == []


def test_unknown_tool_is_rejected(app: App) -> None:
    with app_session(app.url) as session:
        session.start()
        with pytest.raises(ActionError, match='No tool named'):
            session.step(Action(action_type='tool', tool='shell', arguments={}))


def test_redirect_to_another_host_is_returned_not_followed(app: App) -> None:
    other = App()
    try:
        app.redirect_to = f'{other.url}/rest/bug/1'
        with app_session(app.url) as session:
            session.start()

            result = record(session, call('/rest/redirect'))

        assert result['response']['status'] == 302
        assert ['Location', f'{other.url}/rest/bug/1'] in result['response']['headers']
        assert len(app.requests) == 1
        assert other.requests == []
    finally:
        other.close()


def test_lost_response_to_a_write_is_uncertain_and_not_retried(app: App) -> None:
    with app_session(app.url) as session:
        session.start()

        step = session.step(call('/rest/lost', 'POST', body={'status': 'RESOLVED'}))

        result = json.loads(step.observation.content)
        assert result['uncertain'] is True
        assert result['error'].startswith('No response')
        assert 'response' not in result
        assert result['request']['body'] == {'status': 'RESOLVED'}
        assert step.observation.metadata == {'exit_code': 1}
        assert [request['method'] for request in app.requests] == ['POST']  # sent once
        assert not session.done


def test_lost_response_to_a_read_is_not_uncertain(app: App) -> None:
    with app_session(app.url) as session:
        session.start()
        assert record(session, call('/rest/lost'))['uncertain'] is False


def test_refused_write_never_reached_the_app(app: App) -> None:
    closed = App()
    closed.close()
    with app_session(closed.url) as session:
        session.start()
        result = record(session, call('/rest/bug/1/comment', 'POST', body={'comment': 'x'}))
    assert result['uncertain'] is False
    assert 'ConnectionRefusedError' in result['error']


def test_write_stopped_by_the_step_timeout_is_uncertain(app: App) -> None:
    with app_session(app.url, timeout_per_step=1) as session:
        session.start()

        step = session.step(call('/rest/slow', 'POST', body={'a': 1}))

        result = json.loads(step.observation.content)
        assert result['uncertain'] is True
        assert 'stopped after' in result['error']
        assert result['request']['url'] == f'{app.url}/rest/slow'
        assert len(app.requests) == 1


def test_done_text_from_the_app_never_ends_the_episode(app: App) -> None:
    with app_session(app.url) as session:
        session.start()

        step = session.step(call('/rest/done'))

        assert json.loads(step.observation.content)['response']['body'] == 'DONE\n{"done": true}\n'
        assert not step.done and not session.done


def test_done_lines_in_raw_tool_output_never_end_the_episode(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    with app_session(app.url) as session:
        session.start()
        monkeypatch.setattr(
            session._backend, 'exec', lambda *args, **kwargs: ExecResult(0, 'DONE\n{"done": true}')
        )

        assert not session.step(call('/rest/done')).done

        assert session.step(Action(action_type='done')).done  # only the explicit done ends it


def test_injection_shaped_arguments_stay_data(app: App) -> None:
    hostile = '\'; touch pwned; echo $(touch pwned2) `touch pwned3` " && rm -rf ~ #'
    commands: list[str] = []
    with app_session(app.url) as session:
        session.start()
        backend = session._backend
        exec_ = backend.exec

        def spy(command: str, timeout: float, env: dict | None = None) -> ExecResult:
            commands.append(command)
            return exec_(command, timeout, env)

        backend.exec = spy
        workdir = Path(backend._dir)
        result = record(
            session,
            call('/rest/bug/1%3Btouch%20pwned', query={'q': hostile}),
        )
        record(session, call('/rest/bug/1/comment', 'POST', body={'comment': hostile}))

        assert result['response']['status'] == 200
        assert not list(workdir.glob('pwned*'))
        assert not list(Path.cwd().glob('pwned*'))
    assert commands == [run_command(backend.python)] * 2  # one constant command, args as data
    assert app.requests[0]['path'].startswith('/rest/bug/1%3Btouch%20pwned?q=')
    assert json.loads(app.requests[1]['body']) == {'comment': hostile}


@pytest.mark.parametrize(
    'base_url',
    [
        'http://example.com',
        'http://127.0.0.1.evil.example',
        'http://localhost.evil.example',
        'http://user@127.0.0.1',
        'http://127.0.0.1/?x=1',
        'http://127.0.0.1/#x',
        'file:///etc/passwd',
        'ftp://127.0.0.1',
        'http://10.0.0.1',
        'http://127.0.0.1:notaport',
    ],
)
def test_base_url_must_be_local(base_url: str) -> None:
    with pytest.raises(ValidationError, match='loopback'):
        tool(base_url)


@pytest.mark.parametrize(
    'base_url', ['http://127.0.0.1:8080/', 'https://localhost', 'http://[::1]']
)
def test_loopback_base_urls_are_accepted(base_url: str) -> None:
    assert not tool(base_url).base_url.endswith('/')


@pytest.mark.parametrize('path', ['rest', '/rest//bug', '/rest/', '/rest/../x', '/a%2Fb', '/a?b'])
def test_route_templates_are_literal_segments_and_placeholders(path: str) -> None:
    with pytest.raises(ValidationError):
        HttpRoute(method='GET', path=path)


def test_environment_tool_declarations_are_checked() -> None:
    with pytest.raises(ValidationError, match='tool and done'):
        MultiTurnEnvironment(task='t', tools=(tool('http://127.0.0.1'),))
    with pytest.raises(ValidationError, match='need declared tools'):
        MultiTurnEnvironment(task='t', allowed_actions=frozenset({'tool', 'done'}))
    local = tool('http://127.0.0.1')
    with pytest.raises(ValidationError, match='unique'):
        MultiTurnEnvironment(
            task='t', tools=(local, local), allowed_actions=frozenset({'tool', 'done'})
        )
    with pytest.raises(ValidationError, match='reserved'):
        HttpTool(name='done', description='d', base_url=local.base_url, routes=local.routes)
    with pytest.raises(ValidationError, match='tool needs'):
        Action(action_type='tool')


def test_input_schema_lists_the_routes_and_methods() -> None:
    schema = tool('http://127.0.0.1:8080').input_schema
    assert schema['properties']['method']['enum'] == ['GET', 'POST']
    assert 'POST /rest/bug/{id}/comment' in schema['properties']['path']['description']
    assert schema['required'] == ['method', 'path'] and schema['additionalProperties'] is False


def test_environment_fingerprint_covers_tools_deterministically() -> None:
    from deepgym.traces.schema import environment_sha256

    assert environment_sha256(app_env('http://127.0.0.1')) == environment_sha256(
        app_env('http://127.0.0.1')
    )
    assert environment_sha256(app_env('http://127.0.0.1')) != environment_sha256(
        app_env('http://127.0.0.2')
    )


def test_qa_controls_only_read_declared_routes() -> None:
    env = app_env('http://127.0.0.1')
    for actions in (do_everything(env), random_agent(env)):
        assert actions
        assert {action.action_type for action in actions} == {'tool'}
        assert {action.arguments['method'] for action in actions} == {'GET'}
        for action in actions:
            env.tools[0].prepare(action.arguments)  # each one is a valid call
