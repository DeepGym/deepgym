"""Difficulty probe: sample a model on the task and report pass@1 and pass@k."""

from __future__ import annotations

import email.utils
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from deepgym.exceptions import DeepGymError
from deepgym.integrations.verl import extract_code
from deepgym.integrations.zai import ZaiChatClient
from deepgym.models import SCORE_TOLERANCE, Environment
from deepgym.qa.checks import solved
from deepgym.qa.report import DIFFICULTY_CHECK, QACheck

if TYPE_CHECKING:
    from deepgym.core import DeepGym

_DEFAULT_BASE_URL = 'https://api.openai.com/v1'

_RETRIES = 5
"""Retries for a sample the endpoint rate-limited (429), timed out or failed (5xx)."""

_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504})
_BASE_WAIT_S = 2.0
_MAX_WAIT_S = 60.0

_IN_FLIGHT = 4
"""Requests at once: a shared or rate-limited key answers a burst with 429s."""


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
    """Score *samples* completions of ``env.task`` and report how the model does.

    Reported, never gated: the check passes once it has measured, and its details give
    pass@1, pass@k, and the mean and spread of the rewards. They flag a task every
    sample solves (too_easy), one where no sample earns any reward (too_hard), and one
    where every sample earns the same reward (no_variance), since group-relative RL
    learns nothing from equal rewards. Partial rewards count as variance: [0.2, 0.8] is
    neither too hard nor flat.

    The last fenced code block of each completion is the solution; a completion with
    none is an empty answer, the model's own failure. The endpoint's failures are not
    the model's: a request it rate-limits (429), times out or fails (5xx) is retried
    with backoff, and if it never comes back, or the verifier can't judge a sample,
    the check is inconclusive.

    Raises:
        DeepGymError: If the endpoint refuses a request outright (a bad key or model).
    """
    with ThreadPoolExecutor(max_workers=min(samples, _IN_FLIGHT)) as pool:
        completions = list(pool.map(lambda _: _complete(client, env.task), range(samples)))
    if missing := [error for error in completions if isinstance(error, Exception)]:
        details = (
            f'{len(missing)}/{samples} samples never came back from the model endpoint '
            f'after {_RETRIES} retries: {missing[0]}'
        )
        return QACheck(name=DIFFICULTY_CHECK, status='inconclusive', details=details)
    results = dg.run_batch(env, [extract_code(str(text)) for text in completions]).results
    if errored := sum(not result.has_verdict for result in results):
        details = f'{errored}/{samples} samples got no verdict (the verifier errored)'
        return QACheck(name=DIFFICULTY_CHECK, status='inconclusive', details=details)
    scores = [result.score for result in results]
    count = sum(solved(result) for result in results)
    spread = statistics.pstdev(scores)
    flags = [
        flag
        for flag, raised in (
            ('too_easy', count == samples),
            ('too_hard', max(scores) <= SCORE_TOLERANCE),
            ('no_variance', spread <= SCORE_TOLERANCE),
        )
        if raised
    ]
    details = (
        (f'{", ".join(flags)}: ' if flags else '')
        + f'pass@1={count / samples:.2f} pass@{samples}={float(count > 0):.2f}, '
        f'mean reward {statistics.fmean(scores):.2f}, spread {spread:.2f} '
        f'({count}/{samples} solved by {client.model})'
    )
    return QACheck(name=DIFFICULTY_CHECK, status='pass', details=details)


def _complete(client: ZaiChatClient, prompt: str) -> str | Exception:
    """Return one completion, retrying the endpoint's own failures with backoff.

    Waits as long as a Retry-After header asks, else a random time up to 2s, 4s, 8s ...
    (capped at a minute): full jitter, so samples that hit a limit together don't retry
    together. Returns the last error instead of raising once the retries run out, so
    a busy endpoint makes the check inconclusive rather than aborting the run.

    Raises:
        DeepGymError: If the endpoint refuses the request outright (4xx but 408/409/429).
    """
    error: Exception | None = None
    for attempt in range(_RETRIES + 1):
        if attempt:
            ceiling = min(_MAX_WAIT_S, _BASE_WAIT_S * 2 ** (attempt - 1))
            asked = _retry_after(error)
            time.sleep(min(asked, _MAX_WAIT_S) if asked is not None else random.uniform(0, ceiling))
        try:
            return client.complete_prompt(prompt, temperature=1.0, thinking=None)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRYABLE:
                raise DeepGymError(f'the model endpoint refused the request: {exc}') from exc
            error = exc
        except httpx.TransportError as exc:
            error = exc
    return error or RuntimeError('no attempt was made')


def _retry_after(error: Exception | None) -> float | None:
    """Seconds a 429 or 503 asks to wait (Retry-After as seconds or an HTTP date)."""
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    value = error.response.headers.get('retry-after', '').strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):  # absent or not a date
        return None
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
