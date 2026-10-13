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
    app.include_router(create_openrlhf_router(env, dg, infra_failure_score='nan'))

    # Run with: uvicorn app:app --port 8000
    # Train with: --remote_rm_url http://localhost:8000/reward/score \
    #   --algo.dynamic_filtering_enable

The last fenced code block of each response (or all of it, if none) is what runs,
as in the verl and TRL adapters.

OpenRLHF trains on the ``rewards`` the response carries and can't skip a sample. A
failed request doesn't help either: OpenRLHF retries it, then the rollout has no reward
and the training step fails, so a policy that breaks its runs could stop training. The
only way OpenRLHF drops samples is ``--algo.dynamic_filtering_enable``, which drops a
prompt's whole group when the mean of its scores falls outside
``--algo.dynamic_filtering_range``; a NaN always does.

So the endpoint never fails a request over a run. A judged run gets its score. A run
without a verdict for a reason the model may have caused (the verifier failed, the run
timed out, the sandbox broke) scores 0 in either mode, with a warning, and is never run
again. A run the sandbox provider or the network failed (see
:func:`~deepgym.exceptions.is_infrastructure_failure`) is graded once more after a short
pause. Every run, first or repeated, takes one of the router's ``max_parallel`` slots,
and no request waits past 150 s, inside the 180 s OpenRLHF gives it; a first run still
going then scores 0, since the model's code may be what slows it. If the second grading
fails too, can't start in time or is still going then, or the first one never started,
or a query doesn't contain its prompt, the sample is rewarded the router's
``infra_failure_score``, never a score from a run that didn't finish. The router has no
default for it, so the choice is made when it's set up, where no policy can reach it:

- ``'nan'`` (recommended): NaN, so dynamic filtering drops the sample's group. It needs
  ``--algo.dynamic_filtering_enable``: without it OpenRLHF trains on the NaN and the run
  breaks.
- ``'zero'``: 0.0, so OpenRLHF trains on DeepGym's failure as a failed answer, an
  accepted leak.

Either way the sample is logged as a warning with its error type and counted toward the
masked-rate warning (:class:`~deepgym.integrations.reward.MaskedRate`).

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
from deepgym.exceptions import error_type_for, is_infrastructure_failure
from deepgym.integrations.reward import MaskedRate
from deepgym.integrations.verl import extract_code
from deepgym.models import Environment, RunResult
from deepgym.sandbox import error_result

logger = logging.getLogger(__name__)

PROMPT_MISMATCH = 'prompt_mismatch'
"""error_type of a query whose prompt couldn't be found: rather than risk scoring prompt
text, such as a fenced example in the prompt, it gets the router's infra_failure_score."""

DEADLINE_EXCEEDED = 'deadline_exceeded'
"""error_type of a sample not graded when its request had to answer. If its run never
started, it gets the router's infra_failure_score; if it was still going, it scores 0,
since the model's code may be what slowed it."""

_RETRY_DELAY_S = 1.0
"""Pause before the one re-run of a provider or network failure."""
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


def _wait(run: concurrent.futures.Future[RunResult], deadline: float) -> RunResult | str:
    """Return *run*'s result, or why the provider, the network or the clock stopped it.

    A run the provider or the network failed (see
    :func:`~deepgym.exceptions.is_infrastructure_failure`), or one still waiting for a slot
    at *deadline* (``time.monotonic``), which then never starts, comes back as its
    error_type: there is no score to give. Anything else the model may have caused, so it
    comes back as a result without a verdict, scored 0: a run that raised for any other
    reason, or one still going at the deadline (it keeps its slot until it ends, and may
    be slow because of the model's code).
    """
    try:
        return run.result(timeout=max(0.0, deadline - time.monotonic()))
    except concurrent.futures.TimeoutError:
        if run.cancel():
            return DEADLINE_EXCEEDED
        unfinished = error_result(TimeoutError('still running at the request deadline'))
        return unfinished.model_copy(update={'error_type': DEADLINE_EXCEEDED})
    except Exception as exc:
        if is_infrastructure_failure(exc):
            logger.warning('OpenRLHF: the provider or network failed a run: %s', exc)
            return error_type_for(exc)
        return error_result(exc)


def create_openrlhf_router(
    env: Environment,
    dg: DeepGym,
    *,
    infra_failure_score: Literal['zero', 'nan'] | None = None,
    max_parallel: int = 32,
) -> APIRouter:
    """Create a FastAPI router compatible with OpenRLHF's reward server protocol.

    OpenRLHF sends one request per rollout, concurrently. The handler is synchronous, so
    FastAPI runs each request in its worker thread pool (40 threads by default) and a
    slow verification never blocks the event loop the other requests arrive on.

    Args:
        env: The environment every response is scored with.
        dg: The DeepGym client that scores them.
        infra_failure_score: Required. The reward of a sample DeepGym couldn't grade
            through no fault of the model: ``'nan'`` (recommended, with OpenRLHF's dynamic
            filtering) or ``'zero'`` (see the module docstring).
        max_parallel: Most runs going at once across every request, first gradings and
            second ones alike.

    Raises:
        ValueError: If *infra_failure_score* isn't ``'nan'`` or ``'zero'``.
    """
    if infra_failure_score not in ('nan', 'zero'):
        raise ValueError(
            "Choose infra_failure_score, the reward of a sample DeepGym couldn't grade because "
            "the sandbox provider or the network failed (OpenRLHF can't skip a sample): 'nan' "
            '(recommended) drops its prompt group, and needs --algo.dynamic_filtering_enable, '
            "without which the NaN breaks training; 'zero' trains on it as a failed answer, "
            'an accepted leak that is logged and counted toward the masked-rate warning.'
        )
    router = APIRouter(prefix='/reward')
    slots = concurrent.futures.ThreadPoolExecutor(max_parallel, 'deepgym-openrlhf')
    masked = MaskedRate('OpenRLHF reward server')
    fallback = 0.0 if infra_failure_score == 'zero' else math.nan

    @router.post('/score', response_model=OpenRLHFRewardResponse)
    def score(request: OpenRLHFRewardRequest) -> Response:
        """Score a batch of model responses and return a reward for every one."""
        deadline = time.monotonic() + _REQUEST_BUDGET_S
        responses = request.responses
        codes = {i: extract_code(r) for i, r in enumerate(responses) if r is not None}
        runs = {i: slots.submit(dg.run, env, code) for i, code in codes.items()}
        outcomes = {i: _wait(run, deadline) for i, run in runs.items()}
        # A provider or network failure is graded once more, in the same slots and time.
        retry = [i for i, o in outcomes.items() if isinstance(o, str) and o != DEADLINE_EXCEEDED]
        if retry and time.monotonic() + _RETRY_DELAY_S < deadline:
            time.sleep(_RETRY_DELAY_S)
            reruns = {i: slots.submit(dg.run, env, codes[i]) for i in retry}
            for i, rerun in reruns.items():
                outcome = _wait(rerun, deadline)
                if isinstance(outcome, RunResult) and outcome.has_verdict:
                    outcomes[i] = outcome
        rewards: list[float] = []
        error_types: list[str | None] = []
        for i in range(len(responses)):
            outcome = outcomes.get(i, PROMPT_MISMATCH)
            masked.record(masked=isinstance(outcome, str))
            if isinstance(outcome, str):
                logger.warning('OpenRLHF: no verdict (%s); rewarded %s', outcome, fallback)
                rewards.append(fallback)
                error_types.append(outcome)
                continue
            if not outcome.has_verdict:
                logger.warning(
                    'OpenRLHF: no verdict (%s), maybe the model output; rewarded 0',
                    outcome.error_type,
                )
            rewards.append(outcome.score if outcome.has_verdict else 0.0)
            error_types.append(outcome.error_type)
        body = OpenRLHFRewardResponse(rewards=rewards, error_types=error_types)
        # FastAPI's own rendering writes a NaN as null, or refuses it, by version.
        return Response(body.model_dump_json(), media_type='application/json')

    return router
