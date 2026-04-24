"""Z.ai / GLM chat integration for DeepGym smoke tests and baselines.

The client intentionally reads credentials from environment variables instead
of files or arguments printed by scripts.  Use it for baseline completions and
trajectory generation; keep DeepGym verifiers deterministic as the reward
source of truth.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import httpx

_DEFAULT_BASE_URL = 'https://api.z.ai/api/paas/v4'
_DEFAULT_MODEL = 'glm-5.1'


class ZaiChatClient:
    """Small OpenAI-compatible Z.ai chat client built on the existing httpx dependency."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError('api_key must be non-empty')
        self._api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._transport = transport

    @classmethod
    def from_env(cls) -> ZaiChatClient:
        """Build a client from ZAI_* environment variables.

        Required:
            ZAI_API_KEY

        Optional:
            ZAI_API_BASE, ZAI_MODEL
        """
        api_key = os.getenv('ZAI_API_KEY')
        if not api_key:
            raise RuntimeError('ZAI_API_KEY is not set')
        return cls(
            api_key,
            base_url=os.getenv('ZAI_API_BASE', _DEFAULT_BASE_URL),
            model=os.getenv('ZAI_MODEL', _DEFAULT_MODEL),
        )

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        thinking: str = 'enabled',
    ) -> str:
        """Return the assistant message content for a chat-completions request."""
        payload: dict[str, Any] = {
            'model': self.model,
            'messages': list(messages),
            'max_tokens': max_tokens,
            'temperature': temperature,
            'thinking': {'type': thinking},
        }
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            response = client.post(
                f'{self.base_url}/chat/completions',
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        return _extract_message_content(data)

    def complete_prompt(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        thinking: str = 'enabled',
    ) -> str:
        """Convenience wrapper for a single user prompt."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        return self.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
        )


def _extract_message_content(data: dict[str, Any]) -> str:
    """Extract assistant content from an OpenAI-compatible response payload."""
    try:
        content = data['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f'Unexpected Z.ai response shape: {data!r}') from exc
    if content is None:
        return ''
    return str(content)
