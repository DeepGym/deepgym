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

try:
    from fastapi import APIRouter
except ImportError as _exc:
    raise ImportError(
        'FastAPI is required for the OpenRLHF integration. '
        'Install with: pip install "deepgym[server]"'
    ) from _exc
from pydantic import BaseModel, model_validator

from deepgym.core import DeepGym
from deepgym.integrations.verl import extract_code
from deepgym.models import Environment


class OpenRLHFRewardRequest(BaseModel):
    """Reward request in either shape OpenRLHF clients send.

    OpenRLHF's agent client sends ``query`` (each prompt and its response decoded
    together, special tokens included), ``prompts`` and ``labels`` (unused here).
    Older clients send ``outputs``, the responses alone. A query that doesn't start
    with its prompt (a tokenizer that doesn't round-trip it) is scored whole.
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
    def responses(self) -> list[str]:
        """Each sample's response: ``outputs``, or each query with its prompt removed."""
        if self.outputs is not None:
            return self.outputs
        return [query.removeprefix(prompt) for query, prompt in zip(self.query, self.prompts)]


class OpenRLHFRewardResponse(BaseModel):
    """Response format compatible with OpenRLHF's reward API."""

    rewards: list[float]


def create_openrlhf_router(env: Environment, dg: DeepGym) -> APIRouter:
    """Create a FastAPI router compatible with OpenRLHF's reward server protocol."""
    router = APIRouter(prefix='/reward')

    @router.post('/score')
    async def score(request: OpenRLHFRewardRequest) -> OpenRLHFRewardResponse:
        """Score a batch of model responses and return rewards."""
        solutions = [extract_code(response) for response in request.responses]
        if not solutions:
            return OpenRLHFRewardResponse(rewards=[])
        batch = dg.run_batch(env, solutions, max_parallel=min(len(solutions), 32))
        return OpenRLHFRewardResponse(rewards=[r.score for r in batch.results])

    return router
