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

    assert result == '{"ok": true}'
    assert seen['url'] == 'https://api.test/v4/chat/completions'
    assert seen['auth'] == 'Bearer test-key'
    assert seen['payload']['model'] == 'glm-5.1'
    assert seen['payload']['thinking'] == {'type': 'disabled'}
    assert seen['payload']['messages'][-1] == {'role': 'user', 'content': 'solve task'}


def test_zai_client_from_env_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ZAI_API_KEY', raising=False)

    with pytest.raises(RuntimeError, match='ZAI_API_KEY'):
        ZaiChatClient.from_env()
