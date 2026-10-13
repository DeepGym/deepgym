"""Difficulty probe: sample a model on the task and report pass@1 and pass@k."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from deepgym.exceptions import DeepGymError
from deepgym.integrations.verl import extract_code
from deepgym.integrations.zai import ZaiChatClient
from deepgym.models import Environment
from deepgym.qa.checks import solved
from deepgym.qa.report import QACheck, verdict

if TYPE_CHECKING:
    from deepgym.core import DeepGym

_DEFAULT_BASE_URL = 'https://api.openai.com/v1'


def client_from_env(model: str) -> ZaiChatClient:
    """Build an OpenAI-compatible chat client for *model* from the environment.

    Reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` (default: the OpenAI API).
    For a local server without auth, set ``OPENAI_API_KEY`` to any placeholder.

    Raises:
        DeepGymError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise DeepGymError('--model needs OPENAI_API_KEY (and OPENAI_BASE_URL if not OpenAI)')
    base_url = os.getenv('OPENAI_BASE_URL') or _DEFAULT_BASE_URL
    return ZaiChatClient(api_key, base_url=base_url, model=model)


def difficulty_check(env: Environment, dg: DeepGym, client: ZaiChatClient, samples: int) -> QACheck:
    """Score *samples* completions of ``env.task`` and flag tasks with no reward variance.

    The last fenced code block of each completion is the solution. A sample counts
    as solved at full marks. When every sample is solved (too_easy) or none is
    (too_hard), group-relative RL gets no signal from this env for this model, so
    the check fails.

    Raises:
        httpx.HTTPError: If the model endpoint rejects a request.
    """

    def sample(_: int) -> str:
        return client.complete_prompt(env.task, temperature=1.0, thinking=None)

    with ThreadPoolExecutor(max_workers=samples) as pool:
        completions = list(pool.map(sample, range(samples)))
    results = dg.run_batch(env, [extract_code(text) for text in completions]).results
    count = sum(solved(result) for result in results)
    flag = 'too_easy: ' if count == samples else 'too_hard: ' if count == 0 else ''
    details = (
        f'{flag}pass@1={count / samples:.2f} pass@{samples}={float(count > 0):.2f} '
        f'({count}/{samples} solved by {client.model})'
    )
    return QACheck(name='difficulty', status=verdict(not flag), details=details)
