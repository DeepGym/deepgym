"""OpenRLHF reward server integration.

OpenRLHF supports 'verifiable rewards' (function-based rewards) natively.
This module provides a reward server router that OpenRLHF can call for scores.

Usage (custom app)::

    from fastapi import FastAPI
    from deepgym.core import DeepGym
    from deepgym.integrations.openrlhf import create_openrlhf_router
    from deepgym.registry import load_environment

    app = FastAPI()
    env = load_environment('coin_change')
    dg = DeepGym(mode='local')
    app.include_router(create_openrlhf_router(env, dg))

    # Run with: uvicorn app:app --port 8000
    # Train with: --remote_rm_url http://localhost:8000/reward/score

The last fenced code block of each response (or all of it, if none) is what runs,
as in the verl and TRL adapters.

Note: This router is NOT automatically mounted by ``deepgym serve``.
Create a custom FastAPI app to use it.

Sources:
    - https://github.com/OpenRLHF/OpenRLHF (openrlhf/utils/agent.py,
      openrlhf/cli/serve_rm.py)
"""

from __future__ import annotations

import logging
from typing import Annotated

try:
    from fastapi import APIRouter
except ImportError as _exc:
    raise ImportError(
        'FastAPI is required for the OpenRLHF integration. '
        'Install with: pip install "deepgym[server]"'
    ) from _exc
from pydantic import BaseModel, Field, model_validator

from deepgym.core import DeepGym
from deepgym.integrations.verl import extract_code
from deepgym.models import Environment

logger = logging.getLogger(__name__)

PROMPT_MISMATCH = 'prompt_mismatch'
"""error_type of a query whose prompt couldn't be found: it scores 0 instead of risking
scoring prompt text, such as a fenced example in the prompt."""

MAX_TEXT_CHARS = 2_000_000
"""Longest query, prompt or output accepted (about 500k tokens); longer ones get a 422."""
MAX_BATCH = 1024
"""Most samples in one request."""
MAX_REQUEST_CHARS = 64_000_000
"""Most characters in one request, over every query, prompt and output."""

_TAIL = 256
"""Characters at the end of a prompt that locate the response when its front was cut."""

_Text = Annotated[str, Field(max_length=MAX_TEXT_CHARS)]


def _response(query: str, prompt: str) -> str | None:
    """Return the response part of *query*, or None if its prompt can't be located.

    The query is the prompt and the response decoded together, so it normally starts
    with the prompt. When the prompt is too long, OpenRLHF drops tokens from its front
    and keeps its end, so the response starts after the last occurrence of the prompt's
    final ``_TAIL`` characters. That never takes in prompt text; a response that repeats
    those characters only loses what comes before the repeat. Linear time: a prefix
    check and one ``rfind`` of a bounded needle.
    """
    if query.startswith(prompt):
        return query[len(prompt) :]
    tail = prompt[-_TAIL:]
    end = query.rfind(tail)
    return None if end == -1 else query[end + len(tail) :]


class OpenRLHFRewardRequest(BaseModel):
    """Reward request in either shape OpenRLHF clients send.

    OpenRLHF's agent client sends ``query`` (each prompt and its response decoded
    together, special tokens included), ``prompts`` and ``labels`` (unused here).
    Older clients send ``outputs``, the responses alone. Bodies are untrusted, so sizes
    are capped (``MAX_TEXT_CHARS``, ``MAX_BATCH``, ``MAX_REQUEST_CHARS``).
    """

    prompts: list[_Text] = Field(max_length=MAX_BATCH)
    query: list[_Text] | None = Field(default=None, max_length=MAX_BATCH)
    outputs: list[_Text] | None = Field(default=None, max_length=MAX_BATCH)

    @model_validator(mode='after')
    def check_responses(self) -> OpenRLHFRewardRequest:
        """Require ``query`` or ``outputs``, one prompt per query, and a bounded total size."""
        if self.outputs is None and self.query is None:
            raise ValueError('send query (prompt + response, as OpenRLHF does) or outputs')
        if self.outputs is None and len(self.query) != len(self.prompts):
            raise ValueError('query and prompts must have the same length')
        texts = (*self.prompts, *(self.query or ()), *(self.outputs or ()))
        if sum(map(len, texts)) > MAX_REQUEST_CHARS:
            raise ValueError(f'request should have at most {MAX_REQUEST_CHARS} characters')
        return self

    @property
    def responses(self) -> list[str | None]:
        """Each sample's response: ``outputs``, or each query after its prompt.

        None where a query's prompt can't be found (see :func:`_response`).
        """
        if self.outputs is not None:
            return list(self.outputs)
        return [_response(query, prompt) for query, prompt in zip(self.query, self.prompts)]


class OpenRLHFRewardResponse(BaseModel):
    """Response format compatible with OpenRLHF's reward API.

    OpenRLHF reads ``rewards``; ``error_types`` says, per sample, why there was no
    verdict (None when there was one), so a caller can tell a real 0 from a failure.
    """

    rewards: list[float]
    error_types: list[str | None] = Field(default_factory=list)


def create_openrlhf_router(env: Environment, dg: DeepGym) -> APIRouter:
    """Create a FastAPI router compatible with OpenRLHF's reward server protocol.

    OpenRLHF sends one request per rollout, concurrently. The handler is synchronous, so
    FastAPI runs each request in its worker thread pool (40 threads by default) and a
    slow verification never blocks the event loop the other requests arrive on.
    """
    router = APIRouter(prefix='/reward')

    @router.post('/score')
    def score(request: OpenRLHFRewardRequest) -> OpenRLHFRewardResponse:
        """Score a batch of model responses and return rewards."""
        responses = request.responses
        found = [response for response in responses if response is not None]
        if len(found) < len(responses):
            logger.warning(
                'OpenRLHF: %d of %d queries do not contain their prompt; scored 0 (%s)',
                len(responses) - len(found),
                len(responses),
                PROMPT_MISMATCH,
            )
        solutions = [extract_code(response) for response in found]
        results = iter(
            dg.run_batch(env, solutions, max_parallel=min(len(solutions), 32)).results
            if solutions
            else ()
        )
        rewards: list[float] = []
        error_types: list[str | None] = []
        for response in responses:
            result = None if response is None else next(results)
            rewards.append(0.0 if result is None else result.score)
            error_types.append(PROMPT_MISMATCH if result is None else result.error_type)
        return OpenRLHFRewardResponse(rewards=rewards, error_types=error_types)

    return router
