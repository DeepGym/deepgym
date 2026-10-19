"""Tests for declared HTTP API tools run as Session tool actions, against a loopback app."""

import base64
import html
import json
import shlex
from pathlib import Path
from urllib.parse import quote, quote_plus

import pytest
from pydantic import SecretStr, ValidationError

from deepgym.exceptions import ActionError
from deepgym.models import Action, MultiTurnEnvironment, RunResult
from deepgym.qa.agents import do_everything, random_agent
from deepgym.sandboxes import ExecResult
from deepgym.session import LocalBackend, SandboxBackend, Session
from deepgym.tools import REDACTED, HttpRoute, HttpTool, run_command
from tests.http_app import API_KEY, BIG, ROUTES, App, app_env, app_session, tool


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
        {'method': 'GET', 'path': '/rest/bug/..;'},
        {'method': 'GET', 'path': '/rest/bug/%2e%2e;x'},
        {'method': 'GET', 'path': '/rest/bug/%252e%252e'},
        {'method': 'GET', 'path': '/rest/bug/%EF%BC%8E%EF%BC%8E'},
        {'method': 'GET', 'path': '/rest/bug/...'},
        {'method': 'GET', 'path': '/rest/bug/%00'},
        {'method': 'GET', 'path': '/rest/bug/1%0d%0aX-Injected:%201'},
        {'method': 'GET', 'path': '/rest/bug/%zz'},
        {'method': 'GET', 'path': '/rest/bug/%2'},
        {'method': 'GET', 'path': '/rest/bug/\u00e9'},
        {'method': 'GET', 'path': '/rest/bug/1\t'},
        {'method': 'GET', 'path': '/rest/bug/%EF%BC%BC'},
        {'method': 'POST', 'path': '/rest/bug/1/comment', 'body': {'n': float('nan')}},
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


@pytest.mark.parametrize('path', ['/rest/partial', '/rest/reset'])
def test_incomplete_response_keeps_what_arrived(app: App, path: str) -> None:
    with app_session(app.url) as session:
        session.start()
        step = session.step(call(path))

    result = json.loads(step.observation.content)
    assert step.observation.metadata == {'exit_code': 1}  # no complete response
    assert result['error'].startswith('Incomplete response: ')
    assert result['uncertain'] is False
    response = result['response']
    assert response['incomplete'] is True
    assert response['status'] == 200 and ['Content-Length', '100'] in response['headers']
    assert response['body'] == '{"id": 7, "sta'  # every byte received


def verifier(score: float) -> str:
    return f'import json\nprint(json.dumps({{"score": {score}, "passed": {score == 1}}}))\n'


def scored(session: Session, *actions: Action) -> RunResult:
    session.start()
    for action in actions:
        session.step(action)
    session.step(Action(action_type='done'))
    return session.score()


SETTLED_VERIFIER = (
    'import json, pathlib\n'
    'ok = pathlib.Path("settled").exists()\n'
    'print(json.dumps({"score": float(ok), "passed": ok}))\n'
)


@pytest.mark.parametrize('score', [1.0, 0.0])
@pytest.mark.parametrize(
    ('path', 'settings'),
    [
        ('/rest/bug/1/comment', {}),  # answered 201: the app may still be applying it
        ('/rest/lost', {}),
        ('/rest/slow', {'timeout_per_step': 2}),
    ],
)
def test_a_write_never_settled_gets_no_verdict_whatever_its_score(
    app: App, path: str, settings: dict, score: float
) -> None:
    write = call(path, 'POST', body={'status': 'RESOLVED'})
    with app_session(app.url, final_verifier_code=verifier(score), **settings) as session:
        result = scored(session, write)

    assert not result.has_verdict and result.error_type == 'unsettled_write'
    assert result.score == 0 and not result.passed
    assert f'{app.url}{path}' in result.output and 'no root_tool_settle' in result.output


@pytest.mark.parametrize(('settle', 'graded'), [('touch settled', True), ('exit 3', False)])
def test_writes_are_settled_before_grading(app: App, settle: str, graded: bool) -> None:
    write = call('/rest/bug/1/comment', 'POST', body={'comment': 'x'})
    with app_session(
        app.url, final_verifier_code=SETTLED_VERIFIER, root_tool_settle=settle
    ) as session:
        result = scored(session, write)
    assert result.has_verdict is graded and result.passed is graded
    assert graded or 'root_tool_settle exited 3' in result.output


def test_step_verifiers_cannot_run_beside_write_tools() -> None:
    with pytest.raises(ValidationError, match='step verifier'):
        app_env('http://127.0.0.1', step_verifier_code='print(1)')
    reads = tool('http://127.0.0.1').model_copy(
        update={'routes': tuple(r for r in ROUTES if r.method == 'GET')}
    )
    MultiTurnEnvironment(  # a tool that only reads leaves nothing to settle
        task='t',
        tools=(reads,),
        allowed_actions=frozenset({'tool', 'done'}),
        step_verifier_code='print(1)',
    )


def test_reads_and_writes_that_never_arrived_leave_nothing_to_settle(app: App) -> None:
    closed = App()
    closed.close()
    for base_url, action in (
        (app.url, call('/rest/lost')),
        (closed.url, call('/rest/bug/1/comment', 'POST', body={'comment': 'x'})),
    ):
        with app_session(base_url, final_verifier_code=verifier(1.0)) as session:
            assert scored(session, action).passed


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
    sent: list[tuple[str, str]] = []
    with app_session(app.url) as session:
        session.start()
        backend = session._backend
        exec_input = backend.exec_input

        def spy(command: str, data: str, timeout: float) -> ExecResult:
            sent.append((command, data))
            return exec_input(command, data, timeout)

        backend.exec_input = spy
        workdir = Path(backend._dir)
        result = record(session, call('/rest/bug/1%3Btouch%20pwned', query={'q': hostile}))
        record(session, call('/rest/bug/1/comment', 'POST', body={'comment': hostile}))

        assert result['response']['status'] == 200
        assert not list(workdir.glob('pwned*'))
        assert not list(Path.cwd().glob('pwned*'))
    # One constant command; the arguments travel only as JSON data on its stdin.
    assert [command for command, _ in sent] == [run_command(backend.python)] * 2
    assert all(hostile not in command for command, _ in sent)
    assert json.loads(sent[1][1])['body'] == json.dumps({'comment': hostile}, ensure_ascii=False)
    assert app.requests[0]['path'].startswith('/rest/bug/1%3Btouch%20pwned?q=')
    assert json.loads(app.requests[1]['body']) == {'comment': hostile}


def test_credentials_stay_out_of_files_results_and_errors(app: App) -> None:
    with app_session(app.url) as session:
        session.start()
        workdir = Path(session._backend._dir)

        echoed = session.step(call('/rest/echo')).observation.content
        with pytest.raises(ActionError) as rejected:
            session.step(call('/rest/bug/1', body={'x': 1}))
        files = [path.read_bytes() for path in workdir.rglob('*') if path.is_file()]

    assert app.requests[0]['headers']['x-api-key'] == API_KEY  # the app did get it
    assert API_KEY not in echoed and '[redacted]' in json.loads(echoed)['response']['body']
    assert API_KEY not in str(rejected.value)
    assert not any(API_KEY.encode() in content for content in files)


def test_sandbox_stdin_is_staged_root_only_and_never_on_a_command_line() -> None:
    from tests.test_sandboxes import _CONTAINED_AGENT_PREFIX, _USERS, FakeSandbox

    box = FakeSandbox(network_blocked=False, answers=[_USERS])
    uploads: dict[str, bytes] = {}
    box.upload = lambda content, path: uploads.update({path: content})  # type: ignore[method-assign]
    backend = SandboxBackend(box)

    backend.exec_input(run_command(backend.python), f'{{"key": "{API_KEY}"}}', timeout=5)

    command = box.commands[-1]
    (staged,) = [path for path, content in uploads.items() if API_KEY.encode() in content]
    assert staged.startswith('/opt/deepgym-staging/')
    assert command.startswith('/usr/bin/env -i -- PATH=')
    inner = shlex.split(command)[-1]  # root's sh opens the root-only file, then drops to the agent
    assert inner.startswith(_CONTAINED_AGENT_PREFIX)
    assert f'< {staged}; status=$?; rm -f {staged}; exit $status' in inner
    assert API_KEY not in command


SECRET = 'k3y/with+special&"chars<>'
PASSWORD = 's3cret-pass:word'


def test_echoed_credentials_are_redacted_in_every_form(app: App) -> None:
    basic = base64.b64encode(f'admin:{PASSWORD}'.encode()).decode()
    env = app_env(app.url).model_copy(
        update={
            'tools': (
                tool(app.url).model_copy(
                    update={
                        'headers': {
                            'X-API-KEY': SecretStr(SECRET),
                            'Authorization': SecretStr(f'Basic {basic}'),
                        }
                    }
                ),
            )
        }
    )
    with Session(env, LocalBackend()) as session:
        session.start()

        content = session.step(call('/rest/echo')).observation.content

    seen = app.requests[0]['headers']
    assert seen['x-api-key'] == SECRET  # the app did get both credentials
    assert seen['authorization'] == f'Basic {basic}'
    for secret in (SECRET, f'Basic {basic}', basic, f'admin:{PASSWORD}', PASSWORD):
        for form in (
            secret,
            secret.upper(),
            json.dumps(secret)[1:-1],
            html.escape(secret),
            quote(secret, safe=''),
            quote(secret, safe='').lower(),
            quote_plus(secret, safe=''),
        ):
            assert form not in content, form
    response = json.loads(content)['response']
    headers = dict(response['headers'])
    assert headers['Set-Cookie'] == headers['x-api-key'] == headers['X-Seen'] == REDACTED
    assert headers['Location'] == f'/next?key={REDACTED}'
    assert 'issued-session-id' not in content
    for prefix in (b'', b'x', b'xy'):  # the 12 middle characters depend on the key alone
        encoded = base64.b64encode(prefix + SECRET.encode() + b'!').decode()
        assert encoded[4:16] not in content
    assert json.loads(response['body'])['basic'] == {'pair': REDACTED, 'password': REDACTED}


def test_redaction_keeps_the_record_json_and_its_protocol_types(app: App) -> None:
    env = app_env(app.url).model_copy(
        update={
            'tools': (
                tool(app.url).model_copy(update={'headers': {'X-API-KEY': SecretStr('200')}}),
            )
        }
    )
    with Session(env, LocalBackend()) as session:
        session.start()
        step = session.step(call('/rest/echo'))

    assert step.observation.metadata == {'exit_code': 0}
    result = json.loads(step.observation.content)  # still a record, the credential gone
    response = result['response']
    assert response['status'] == 200 and isinstance(result['elapsed_ms'], float)
    assert dict(response['headers'])['X-Seen'] == REDACTED
    assert json.loads(response['body'])['raw']['X-API-KEY'] == REDACTED


def test_every_tools_credentials_are_redacted_from_every_record(app: App) -> None:
    app.redirect_to = f'http://127.0.0.1:9/?k={SECRET}'  # the app hands back the other key
    admin = tool(app.url).model_copy(
        update={'name': 'admin', 'headers': {'X-Admin-Key': SecretStr(SECRET)}}
    )
    env = app_env(app.url).model_copy(update={'tools': (tool(app.url), admin)})
    with Session(env, LocalBackend()) as session:
        session.start()
        content = session.step(call('/rest/redirect')).observation.content

    assert json.dumps(SECRET)[1:-1] not in content and quote(SECRET, safe='') not in content
    headers = dict(json.loads(content)['response']['headers'])
    assert headers['Location'] == f'http://127.0.0.1:9/?k={REDACTED}'


def test_credentials_in_a_binary_body_are_redacted_in_its_bytes(app: App) -> None:
    env = app_env(app.url).model_copy(
        update={
            'tools': (
                tool(app.url).model_copy(update={'headers': {'X-API-KEY': SecretStr('200')}}),
            )
        }
    )
    with Session(env, LocalBackend()) as session:
        session.start()
        result = record(session, call('/rest/binary-key'))

    body = base64.b64decode(result['response']['body_base64'], validate=True)
    assert body == b'\xff' + REDACTED.encode() + b'\x80'


def test_credentials_are_masked_in_dumps_fingerprints_and_errors() -> None:
    from deepgym.traces.schema import environment_sha256

    one, other = (
        app_env('http://127.0.0.1').model_copy(
            update={'tools': (tool('http://127.0.0.1').model_copy(update={'headers': headers}),)}
        )
        for headers in ({'X-API-KEY': SecretStr(SECRET)}, {'X-API-KEY': SecretStr('other')})
    )
    assert SECRET not in repr(one) and SECRET not in one.model_dump_json()
    # A fingerprint hash of a secret could be brute-forced; the secret is not in it.
    assert environment_sha256(one) == environment_sha256(other)
    for headers in ({'Host': SECRET}, {'X-Key': f'{SECRET}\n'}, {'X-Key': SECRET, 'x-key': SECRET}):
        with pytest.raises(ValidationError) as invalid:
            HttpTool(
                name='t',
                description='d',
                base_url='http://127.0.0.1',
                routes=ROUTES,
                headers=headers,
            )
        assert SECRET not in str(invalid.value)


@pytest.mark.parametrize('path', ['/rest/bug/%C0%AE%C0%AE', '/rest/bug/%C0%AF', '/rest/bug/%FF'])
def test_escapes_that_are_not_utf8_are_rejected(app: App, path: str) -> None:
    with app_session(app.url) as session:
        session.start()
        with pytest.raises(ActionError, match='no dot segments'):
            session.step(call(path))
    assert app.requests == []


@pytest.mark.parametrize(
    ('path', 'method', 'body'),
    [
        ('/rest/bug/%C3%A9', 'GET', None),
        ('/rest/bug/a;b', 'GET', None),
        ('/rest/bug/1/comment', 'POST', {'comment': 'x'}),
    ],
)
def test_the_request_line_is_the_validated_target(
    app: App, path: str, method: str, body: dict | None
) -> None:
    with app_session(app.url) as session:
        session.start()
        arguments = {'query': {'q': 'a b'}} | ({'body': body} if body else {})
        result = record(session, call(path, method, **arguments))

    (seen,) = app.requests
    target = result['request']['url'].removeprefix(app.url)
    assert target == f'{path}?q=a+b'
    assert seen['line'] == f'{method} {target} HTTP/1.1'  # exactly what was validated


@pytest.mark.parametrize('method', ['get', 'Get', 'GET ', 'HEAD', 'OPTIONS', 'TRACE', 'CONNECT'])
def test_method_must_be_a_declared_one_exactly(app: App, method: str) -> None:
    with app_session(app.url) as session:
        session.start()
        with pytest.raises(ActionError):
            session.step(call('/rest/bug/1', method))
    assert app.requests == []


@pytest.mark.parametrize(
    'arguments',
    [
        {'query': {'_method': 'DELETE'}},
        {'query': {'_METHOD': 'DELETE'}},
        {'query': {'X-HTTP-Method-Override': 'DELETE'}},
        {'query': {'x-method-override': 'PUT'}},
        {'body': {'_method': 'DELETE', 'comment': 'x'}},
        {'body': {'X-Http-Method': 'DELETE'}},
    ],
)
def test_method_overrides_are_rejected(app: App, arguments: dict) -> None:
    method = 'POST' if 'body' in arguments else 'GET'
    path = '/rest/bug/1/comment' if 'body' in arguments else '/rest/bug/1'
    with app_session(app.url) as session:
        session.start()
        with pytest.raises(ActionError, match='override the request method'):
            session.step(call(path, method, **arguments))
    assert app.requests == []


@pytest.mark.parametrize(
    'headers',
    [
        {'X-HTTP-Method-Override': 'DELETE'},
        {'x-http-method': 'DELETE'},
        {'Host': 'evil.example'},
        {'Content-Length': '0'},
        {'Transfer-Encoding': 'chunked'},
        {'X-Key': 'a\r\nX-Injected: 1'},
        {'X-Key': ''},
        {'Bad Name': 'x'},
        {'Content-Type': 'text/plain'},
        {'X-Key': 'a', 'x-key': 'b'},
    ],
)
def test_tool_headers_cannot_reroute_or_reframe_requests(headers: dict) -> None:
    with pytest.raises(ValidationError, match='header'):
        HttpTool(
            name='t', description='d', base_url='http://127.0.0.1', routes=ROUTES, headers=headers
        )


@pytest.mark.parametrize(
    'path',
    [
        '/rest/bug/1@evil.example',
        '/rest/bug/a;b=c',
        "/rest/bug/!$&'()*+,=:~_-.x",
        '/rest/bug/%41%42',
        '/rest/file/src%2Fmain.c',
    ],
)
def test_the_app_receives_exactly_the_recorded_target(app: App, path: str) -> None:
    with app_session(app.url) as session:
        session.start()
        result = record(session, call(path, query={'q': 'a b/é&x=1'}))

    (seen,) = app.requests
    assert result['request']['url'] == f'{app.url}{seen["path"]}'  # byte for byte
    assert seen['path'].startswith(f'{path}?q=')


def test_an_encoded_slash_needs_a_path_placeholder(app: App) -> None:
    with app_session(app.url) as session:
        session.start()
        for path in ('/rest/bug/a%2Fcomment', '/rest/bug/a%252Fb', '/rest/bug/%EF%BC%8F'):
            with pytest.raises(ActionError, match='fits no route'):
                session.step(call(path))
        assert record(session, call('/rest/file/a%2Fb'))['response']['status'] == 404
    assert [request['path'] for request in app.requests] == ['/rest/file/a%2Fb']


@pytest.mark.parametrize(
    ('base_url', 'normal'),
    [
        ('http://LOCALHOST:8080/api/', 'http://localhost:8080/api'),
        ('http://localhost', 'http://localhost:80'),
        ('https://127.0.0.2', 'https://127.0.0.2:443'),
        ('http://[::1]:9000', 'http://[::1]:9000'),
    ],
)
def test_base_url_is_rebuilt_from_its_checked_parts(base_url: str, normal: str) -> None:
    assert tool(base_url).base_url == normal


def test_tools_allow_no_other_agent_action() -> None:
    for extra in ('bash', 'code', 'write_file'):
        with pytest.raises(ValidationError, match='exactly tool and done'):
            MultiTurnEnvironment(
                task='t',
                tools=(tool('http://127.0.0.1'),),
                allowed_actions=frozenset({'tool', 'done', extra}),
            )


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
        'http://local\thost',
        'http://127.0.0.1\\@evil.example',
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
