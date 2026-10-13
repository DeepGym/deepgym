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

_PROBE = 64
"""Leading query characters that must match the prompt's tail when the prompt was cut."""


def _response(query: str, prompt: str) -> str | None:
    """Return the response part of *query*, or None if its prompt can't be located.

    The query is the prompt and the response decoded together. When the prompt is too
    long, OpenRLHF drops tokens from its front, so the query starts with the prompt's
    tail instead, and the response starts after the longest tail of *prompt* that the
    query starts with (at least ``_PROBE`` characters of it).
    """
    if query.startswith(prompt):
        return query[len(prompt) :]
    query = query.lstrip('\ufffd')  # leftover bytes of a character the cut split
    probe = query[:_PROBE]
    start = prompt.find(probe)
    while start != -1:
        if query.startswith(prompt[start:]):
            return query[len(prompt) - start :]
        start = prompt.find(probe, start + 1)
    return None


class OpenRLHFRewardRequest(BaseModel):
    """Reward request in either shape OpenRLHF clients send.

    OpenRLHF's agent client sends ``query`` (each prompt and its response decoded
    together, special tokens included), ``prompts`` and ``labels`` (unused here).
    Older clients send ``outputs``, the responses alone.
    """

    prompts: list[str]
    query: list[str] | None = None
    outputs: list[str] | None = None

    @model_validator(mode='after')
    def check_responses(self) -> OpenRLHFRewardRequest:
        """Require ``query`` or ``outputs``, with one prompt per query."""
        if self.outputs is None and self.query is None:
            raise ValueError('send query (prompt + response, as OpenRLHF does) or outputs')
        if self.outputs is None and len(self.query) != len(self.prompts):
            raise ValueError('query and prompts must have the same length')
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
    """Create a FastAPI router compatible with OpenRLHF's reward server protocol."""
    router = APIRouter(prefix='/reward')

    @router.post('/score')
    async def score(request: OpenRLHFRewardRequest) -> OpenRLHFRewardResponse:
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
