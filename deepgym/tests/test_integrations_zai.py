"""Tests for the Z.ai / GLM integration client."""

from __future__ import annotations

import json

import httpx
import pytest

from deepgym.integrations.zai import ZaiChatClient


def test_zai_client_posts_openai_compatible_payload() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen['url'] = str(request.url)
        seen['auth'] = request.headers.get('Authorization')
        seen['payload'] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={'choices': [{'message': {'content': '{"ok": true}'}}]},
        )

    client = ZaiChatClient(
        'test-key',
        base_url='https://api.test/v4/',
        model='glm-5.1',
        transport=httpx.MockTransport(handler),
    )

    result = client.complete_prompt('solve task', thinking='disabled', temperature=0.0)

    assert result.text == '{"ok": true}'
    assert seen['url'] == 'https://api.test/v4/chat/completions'
    assert seen['auth'] == 'Bearer test-key'
    assert seen['payload']['model'] == 'glm-5.1'
    assert seen['payload']['thinking'] == {'type': 'disabled'}
    assert seen['payload']['messages'][-1] == {'role': 'user', 'content': 'solve task'}


def test_zai_client_from_env_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ZAI_API_KEY', raising=False)

    with pytest.raises(RuntimeError, match='ZAI_API_KEY'):
        ZaiChatClient.from_env()


def test_zai_client_omits_thinking_when_none() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen['payload'] = json.loads(request.content.decode())
        return httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}}]})

    client = ZaiChatClient('test-key', transport=httpx.MockTransport(handler))

    assert client.complete_prompt('solve task', thinking=None).text == 'ok'
    assert 'thinking' not in seen['payload']


@pytest.mark.parametrize(
    ('content', 'finish_reason', 'text', 'truncated'),
    [
        ('def f(): return 1', 'stop', 'def f(): return 1', False),
        ('def f(): return 1', 'length', 'def f(): return 1', True),  # same text, cut off
        (None, 'length', '', True),  # the whole output went to reasoning: no answer at all
    ],
)
def test_zai_client_sends_no_output_limit_and_returns_how_generation_ended(
    content: str | None, finish_reason: str, text: str, truncated: bool
) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen['payload'] = json.loads(request.content.decode())
        choice = {'message': {'content': content}, 'finish_reason': finish_reason}
        return httpx.Response(200, json={'choices': [choice]})

    client = ZaiChatClient('test-key', transport=httpx.MockTransport(handler))

    completion = client.complete_prompt('solve task')

    assert 'max_tokens' not in seen['payload']  # the model's own maximum applies
    assert (completion.text, completion.finish_reason) == (text, finish_reason)
    assert completion.truncated is truncated


def test_zai_client_sends_an_output_limit_only_when_asked() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen['payload'] = json.loads(request.content.decode())
        return httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}}]})

    client = ZaiChatClient('test-key', transport=httpx.MockTransport(handler))

    assert client.complete_prompt('solve task', max_tokens=50).finish_reason is None
    assert seen['payload']['max_tokens'] == 50


def test_a_cached_completion_keeps_its_finish_reason(tmp_path) -> None:
    """A reply cut off at the output limit stays marked so from the cache; one cached as
    text alone, before finish reasons were kept, comes back unknown, not whole."""
    from deepgym.integrations.zai import Completion, cached_completion

    cut = Completion(text='partial answer', finish_reason='length')
    path = tmp_path / 'task.txt'

    assert cached_completion(path, lambda: cut) == (cut, 'model')
    assert cached_completion(path, lambda: pytest.fail('cached')) == (cut, 'cache')
    assert cached_completion(path, lambda: cut, refresh=True) == (cut, 'model')

    path.write_text('an old cache: text only', encoding='utf-8')
    legacy, source = cached_completion(path, lambda: pytest.fail('cached'))
    assert (legacy.text, legacy.finish_reason, source) == ('an old cache: text only', None, 'cache')
