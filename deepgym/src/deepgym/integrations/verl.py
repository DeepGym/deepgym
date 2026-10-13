"""verl integration.

No-code setup: verl loads DeepGym's reward function straight from the package,
and each dataset row names its environment::

    reward:
      custom_reward_function:
        path: pkg://deepgym.integrations.verl
        name: compute_score
        reward_kwargs: {mode: daytona}   # optional: auto (default), daytona, local, apple

Build the rows with :func:`verl_rows` and save them with
``pandas.DataFrame(rows).to_parquet(path)``.

A failed sandbox or verifier scores 0.0 with ``error: 1.0``, and verl trains on it.
To keep those samples out of the policy loss, use the agent loop in
:mod:`deepgym.integrations.verl_mask` instead of the reward function.

For a fixed environment in your own reward module, use
:func:`make_verl_compute_score`; for custom training loops,
:func:`make_verl_reward_fn`.

Sources:
    - https://verl.readthedocs.io/en/latest/preparation/reward_function.html
    - https://github.com/volcengine/verl (verl/trainer/ppo/reward.py,
      verl/experimental/reward_loop/reward_manager/naive.py)
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from deepgym.core import DeepGym
from deepgym.models import Environment, RunResult
from deepgym.registry import load_environment
from deepgym.sandboxes import Provider

_FENCED_BLOCK = re.compile(r'```[\w+-]*\n(.*?)```', re.DOTALL)
# End-of-turn/EOS tokens a decoder leaves when it keeps special tokens: <|im_end|>,
# <|endoftext|>, <|eot_id|>, <|end|>, </s>, <eos>, <end_of_turn>, DeepSeek's <｜...｜>.
_SPECIAL_TOKEN = re.compile(r'<\|[\w.-]+\|>|<｜[^｜<>]+｜>|</s>|<eos>|<end_of_turn>')


def _drop_trailing_special_tokens(response: str) -> str:
    """Return *response* without the special tokens (and whitespace) at its end.

    Walks back by index, so it stays linear in the response length however many
    tokens there are.
    """
    cut = None
    end = len(response.rstrip())
    while (start := response.rfind('<', 0, end)) != -1 and _SPECIAL_TOKEN.fullmatch(
        response, start, end
    ):
        cut = end = start
        while end and response[end - 1].isspace():
            end -= 1
    return response if cut is None else response[:cut]


def extract_code(response: str) -> str:
    """Return the last fenced code block in *response*, or the whole response if none.

    Special tokens at the end (e.g. ``<|im_end|>``, ``</s>``), which decoding with
    ``skip_special_tokens=False`` leaves, are dropped first.
    """
    response = _drop_trailing_special_tokens(response)
    blocks = _FENCED_BLOCK.findall(response)
    return blocks[-1] if blocks else response


@functools.cache
def _client(mode: str) -> DeepGym:
    return DeepGym(mode=mode)  # type: ignore[arg-type]


@functools.cache
def _environment(name: str, variant: int) -> Environment:
    return load_environment(name, variant=variant)  # type: ignore[return-value]


def _score_one(dg: DeepGym, env: Environment, response: str) -> RunResult:
    # run_batch turns verifier and sandbox errors into a zero score instead of crashing training.
    return dg.run_batch(env, [extract_code(response)], max_parallel=1).results[0]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: dict | None = None,
    mode: Literal['auto', 'daytona', 'local'] | Provider = 'auto',
    **kwargs: Any,
) -> dict[str, float]:
    """Score one verl response with the DeepGym environment its dataset row names.

    The environment is ``extra_info['environment']`` (with optional
    ``extra_info['variant']``), else *data_source*. The last fenced code block
    in the response is what gets run. Extra keyword arguments verl passes (e.g.
    ``reward_router_address``) are ignored.

    Args:
        data_source: verl's dataset name; the environment name if extra_info has none.
        solution_str: The model's decoded response.
        ground_truth: Unused; the environment's verifier holds the answers.
        extra_info: The row's extra_info.
        mode: DeepGym backend, set from verl's ``reward_kwargs``.
        **kwargs: Anything else verl passes.

    Returns:
        The same keys for every row, as verl requires to batch them: ``score``,
        ``passed`` (1.0 or 0.0), and ``error``, which is 1.0 when no verdict was
        produced (sandbox or verifier failure, timeout, invalid score; see
        ``RunResult.has_verdict``). The score is then 0.0;
        :mod:`deepgym.integrations.verl_mask` masks such samples out of the loss.
        A patch that is malformed or doesn't apply is the model's fault, so it scores
        0.0 with ``error`` 0.0 and is trained on.

    Raises:
        DeepGymError: If the named environment doesn't exist.
    """
    info = extra_info or {}
    env = _environment(info.get('environment') or data_source, int(info.get('variant') or 0))
    result = _score_one(_client(mode), env, solution_str)
    return {
        'score': result.score,
        'passed': float(result.passed),
        'error': float(not result.has_verdict),
    }


def verl_rows(names: Sequence[str]) -> list[dict[str, Any]]:
    """Build verl dataset rows, one per environment name, for :func:`compute_score`.

    Each row has verl's standard fields (``data_source``, ``prompt``, ``ability``,
    ``reward_model``, ``extra_info``). Repeat rows to train on an environment more often.

    Raises:
        DeepGymError: If a named environment doesn't exist.
    """
    return [
        {
            'data_source': name,
            'prompt': [{'role': 'user', 'content': load_environment(name).task}],
            'ability': 'coding',
            'reward_model': {'style': 'rule', 'ground_truth': ''},
            'extra_info': {'environment': name},
        }
        for name in names
    ]


def make_verl_compute_score(
    env: Environment,
    dg: DeepGym | None = None,
) -> Callable[..., float]:
    """Create a reward function for one fixed environment, matching verl's interface.

    verl calls ``compute_score(data_source=..., solution_str=..., ground_truth=...,
    extra_info=..., **kwargs)``. Point ``reward.custom_reward_function`` at the
    module that holds the returned function.

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).

    Returns:
        A function returning a score between 0.0 and 1.0; 0.0 if the verifier
        failed or gave an invalid score.
    """
    _dg = dg or DeepGym(mode='auto')

    def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: Any = None,
        extra_info: Any = None,
        **kwargs: Any,
    ) -> float:
        """Score the last fenced code block of *solution_str* (or all of it)."""
        return _score_one(_dg, env, solution_str).score

    return compute_score


def make_verl_reward_fn(
    env: Environment,
    dg: DeepGym | None = None,
) -> Callable[[dict], list[float]]:
    """Create a batch reward function for verl custom training loops.

    For use when you have direct access to the training loop and want
    to score batches of outputs at once (more efficient than per-sample).

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).

    Returns:
        Callable taking a data batch dict and returning list of scores.
    """
    _dg = dg or DeepGym(mode='auto')

    def reward_fn(data_batch: dict) -> list[float]:
        """Score a verl data batch against the DeepGym environment.

        Args:
            data_batch: Dict with 'responses' or 'completions' key
                        containing decoded output strings.

        Returns:
            List of scores, one per output.
        """
        outputs = data_batch.get('responses', data_batch.get('completions', []))
        if not outputs:
            return []
        if isinstance(outputs[0], list):
            raise ValueError('Token ID inputs not yet supported. Pass decoded strings.')
        batch = _dg.run_batch(env, outputs, max_parallel=min(len(outputs), 32))
        return [r.score for r in batch.results]

    return reward_fn
