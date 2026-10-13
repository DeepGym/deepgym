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

OpenRLHF trains on the ``rewards`` the response carries and can't skip a sample. A
failed request doesn't help either: OpenRLHF retries it, then the rollout has no reward
and the training step fails, so a policy that breaks its runs could stop training. The
only way OpenRLHF drops samples is ``--algo.dynamic_filtering_enable``, which drops a
prompt's whole group when the mean of its scores falls outside
``--algo.dynamic_filtering_range``; a NaN always does.

So the endpoint never fails a request over a run. A judged run gets its score, and a
failure the model caused is judged and scores 0 (``RunResult.has_verdict`` says which).
A run the sandbox provider or the network failed is run again after a short pause, and
no request waits on scoring past 150 s, so it answers inside the 180 s OpenRLHF gives it.
A run that still fails that way, one still going at that point, or a query that doesn't
contain its prompt is rewarded the router's ``infra_failure_score``:

- ``'zero'`` (the default): 0.0, logged as a warning with its error type and counted
  toward the masked-rate warning (:class:`~deepgym.integrations.reward.MaskedRate`).
  OpenRLHF trains on it as a failed answer; that is the limit of its API.
- ``'nan'``: NaN, so dynamic filtering drops the sample's group, and is counted the same
  way. Use it only with ``--algo.dynamic_filtering_enable``: without it OpenRLHF trains
  on the NaN and the run breaks.

Any other run without a verdict (the verifier failed, or the run timed out) may follow
from the model's output, so it scores 0 in either mode, with a warning.

Note: This router is NOT automatically mounted by ``deepgym serve``.
Create a custom FastAPI app to use it.

Sources:
    - https://github.com/OpenRLHF/OpenRLHF, 0.11.2 (openrlhf/utils/agent.py:
      SingleTurnAgentExecutor; openrlhf/trainer/ppo_utils/samples_generator.py and
      experience_maker.py)
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import time
from typing import Annotated, Literal

try:
    from fastapi import APIRouter, Response
except ImportError as _exc:
    raise ImportError(
        'FastAPI is required for the OpenRLHF integration. '
        'Install with: pip install "deepgym[server]"'
    ) from _exc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepgym.core import DeepGym
from deepgym.integrations.reward import MaskedRate
from deepgym.integrations.verl import extract_code
from deepgym.models import Environment, RunResult

logger = logging.getLogger(__name__)

PROMPT_MISMATCH = 'prompt_mismatch'
"""error_type of a query whose prompt couldn't be found: rather than risk scoring prompt
text, such as a fenced example in the prompt, it gets the router's infra_failure_score."""

DEADLINE_EXCEEDED = 'deadline_exceeded'
"""error_type of a sample still being scored when its request had to answer; it gets the
router's infra_failure_score (a run the model hangs ends at its own timeout, well before)."""

_RETRYABLE = frozenset({'sandbox_error', 'unknown_error'})
"""error_type of a run the sandbox provider or the network failed (see error_type_for)."""
_NOT_THE_MODELS = _RETRYABLE | {PROMPT_MISMATCH, DEADLINE_EXCEEDED}
"""Why a sample may get no verdict without the model's output being the cause."""
_RETRY_DELAYS_S = (1.0, 2.0)
"""Pause before each re-run of a provider or network failure."""
_REQUEST_BUDGET_S = 150.0
"""Most time a request may take, inside the 180 s OpenRLHF gives each attempt."""

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

    OpenRLHF reads ``rewards``. ``error_types`` names, per sample, a judged failure that is
    the model's own (such as ``invalid_patch``), or why it got no verdict, in which case its
    reward is the router's ``infra_failure_score``; None for a clean verdict.
    """

    # A NaN reward goes out as the JSON constant NaN, which OpenRLHF's parser reads.
    model_config = ConfigDict(ser_json_inf_nan='constants')

    rewards: list[float]
    error_types: list[str | None] = Field(default_factory=list)


def _score_by(
    deadline: float, dg: DeepGym, env: Environment, solutions: list[str]
) -> list[RunResult] | None:
    """Score *solutions*, or stop waiting at *deadline* (``time.monotonic``) and return None.

    Runs still going then finish in the background and delete their own sandboxes.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    batch = pool.submit(dg.run_batch, env, solutions, max_parallel=min(len(solutions), 32))
    pool.shutdown(wait=False)
    try:
        return list(batch.result(timeout=max(0.0, deadline - time.monotonic())).results)
    except concurrent.futures.TimeoutError:
        return None


def _score_retrying(dg: DeepGym, env: Environment, solutions: list[str]) -> list[RunResult | None]:
    """Score *solutions*, running again those the provider or the network left unjudged.

    Each re-run waits its ``_RETRY_DELAYS_S`` pause. Nothing is waited for past
    ``_REQUEST_BUDGET_S``, so the request always answers before OpenRLHF gives up on it; a
    sample still being scored then is None.
    """
    deadline = time.monotonic() + _REQUEST_BUDGET_S
    results: list[RunResult | None] = [None] * len(solutions)
    pending = list(range(len(solutions)))
    for delay in (0.0, *_RETRY_DELAYS_S):
        if not pending or time.monotonic() + delay >= deadline:
            break
        time.sleep(delay)
        scored = _score_by(deadline, dg, env, [solutions[i] for i in pending])
        if scored is None:
            break
        for i, result in zip(pending, scored, strict=True):
            results[i] = result
        pending = [i for i in pending if results[i].error_type in _RETRYABLE]
    return results


def create_openrlhf_router(
    env: Environment, dg: DeepGym, infra_failure_score: Literal['zero', 'nan'] = 'zero'
) -> APIRouter:
    """Create a FastAPI router compatible with OpenRLHF's reward server protocol.

    OpenRLHF sends one request per rollout, concurrently. The handler is synchronous, so
    FastAPI runs each request in its worker thread pool (40 threads by default) and a
    slow verification never blocks the event loop the other requests arrive on.

    Args:
        env: The environment every response is scored with.
        dg: The DeepGym client that scores them.
        infra_failure_score: The reward of a sample that got no verdict: ``'zero'``, or
            ``'nan'`` for OpenRLHF runs with dynamic filtering (see the module docstring).
    """
    router = APIRouter(prefix='/reward')
    masked = MaskedRate('OpenRLHF reward server')
    fallback = 0.0 if infra_failure_score == 'zero' else math.nan

    @router.post('/score', response_model=OpenRLHFRewardResponse)
    def score(request: OpenRLHFRewardRequest) -> Response:
        """Score a batch of model responses and return a reward for every one."""
        responses = request.responses
        found = [i for i, response in enumerate(responses) if response is not None]
        solutions = [extract_code(responses[i]) for i in found]
        results = dict(zip(found, _score_retrying(dg, env, solutions), strict=True))
        rewards: list[float] = []
        error_types: list[str | None] = []
        for i in range(len(responses)):
            result = results.get(i)
            if result is not None and result.has_verdict:
                masked.record(masked=False)
                rewards.append(result.score)
                error_types.append(result.error_type)
                continue
            if i not in results:
                error_type = PROMPT_MISMATCH
            elif result is None:
                error_type = DEADLINE_EXCEEDED
            else:
                error_type = result.error_type or 'no_verdict'
            # Any other failure may be the model's output breaking the verifier: it scores 0.
            infra = error_type in _NOT_THE_MODELS
            masked.record(masked=infra)
            reward = fallback if infra else 0.0
            logger.warning(
                'OpenRLHF: a sample got no verdict (%s); rewarded %s', error_type, reward
            )
            rewards.append(reward)
            error_types.append(error_type)
        body = OpenRLHFRewardResponse(rewards=rewards, error_types=error_types)
        # FastAPI's own rendering writes a NaN as null, or refuses it, by version.
        return Response(body.model_dump_json(), media_type='application/json')

    return router
