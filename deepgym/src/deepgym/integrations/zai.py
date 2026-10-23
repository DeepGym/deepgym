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
from pydantic import BaseModel, ConfigDict

_DEFAULT_BASE_URL = 'https://api.z.ai/api/paas/v4'
_DEFAULT_MODEL = 'glm-5.1'


class Completion(BaseModel):
    """The assistant's reply to one chat-completions request, and why generation stopped."""

    model_config = ConfigDict(frozen=True)

    text: str
    """The reply's content; empty if the endpoint returned none."""
    finish_reason: str | None
    """The endpoint's ``finish_reason``: ``'stop'`` at a natural end, ``'length'`` when an
    output limit cut the reply off. None if the endpoint gave none."""

    @property
    def truncated(self) -> bool:
        """Whether an output limit cut the reply off before the model finished it."""
        return self.finish_reason == 'length'


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
        max_tokens: int | None = None,
        temperature: float = 0.2,
        thinking: str | None = 'enabled',
    ) -> Completion:
        """Return the assistant's reply to a chat-completions request, with its finish reason.

        No output limit is sent unless *max_tokens* is given, so the model may answer up to
        its own maximum; check ``Completion.truncated`` for a reply an endpoint cut off.
        Pass ``thinking=None`` to omit the Z.ai-specific field for other
        OpenAI-compatible endpoints.
        """
        payload: dict[str, Any] = {
            'model': self.model,
            'messages': list(messages),
            'temperature': temperature,
        }
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
        if thinking is not None:
            payload['thinking'] = {'type': thinking}
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
        return _completion(data)

    def complete_prompt(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        thinking: str | None = 'enabled',
    ) -> Completion:
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


def _completion(data: dict[str, Any]) -> Completion:
    """Read the assistant's reply and finish reason from an OpenAI-compatible response."""
    try:
        choice = data['choices'][0]
        content = choice['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f'Unexpected Z.ai response shape: {data!r}') from exc
    text = '' if content is None else str(content)
    return Completion(text=text, finish_reason=choice.get('finish_reason'))
