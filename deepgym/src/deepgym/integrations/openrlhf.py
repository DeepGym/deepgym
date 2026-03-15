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

Note: This router is NOT automatically mounted by ``deepgym serve``.
Create a custom FastAPI app to use it.
"""

from __future__ import annotations

try:
    from fastapi import APIRouter
except ImportError as _exc:
    raise ImportError(
        'FastAPI is required for the OpenRLHF integration. '
        'Install with: pip install "deepgym[server]"'
    ) from _exc
from pydantic import BaseModel

from deepgym.core import DeepGym
from deepgym.models import Environment


class OpenRLHFRewardRequest(BaseModel):
    """Request format compatible with OpenRLHF's reward API."""

    prompts: list[str]
    outputs: list[str]


class OpenRLHFRewardResponse(BaseModel):
    """Response format compatible with OpenRLHF's reward API."""

    rewards: list[float]


def create_openrlhf_router(env: Environment, dg: DeepGym) -> APIRouter:
    """Create a FastAPI router compatible with OpenRLHF's reward server protocol."""
    router = APIRouter(prefix='/reward')

    @router.post('/score')
    async def score(request: OpenRLHFRewardRequest) -> OpenRLHFRewardResponse:
        """Score a batch of model outputs and return rewards."""
        if not request.outputs:
            return OpenRLHFRewardResponse(rewards=[])
        batch = dg.run_batch(env, request.outputs, max_parallel=min(len(request.outputs), 32))
        return OpenRLHFRewardResponse(rewards=[r.score for r in batch.results])

    return router
