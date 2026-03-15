"""verl (ByteDance) integration.

verl supports function-based verifiable rewards via a specific interface:
    def compute_score(data_source, solution_str, ground_truth, extra_info=None):
        return score_value

This module provides reward functions matching that signature.

Usage (single function):
    # In your reward module file:
    from deepgym.integrations.verl import make_verl_compute_score

    env = Environment(task='...', verifier_code='...')
    compute_score = make_verl_compute_score(env=env)

    # In verl config:
    # custom_reward_function.path = "path/to/your/reward_module.py"
    # custom_reward_function.name = "compute_score"  (optional if named compute_score)

Usage (batch, for custom training loops):
    from deepgym.integrations.verl import make_verl_reward_fn
    reward_fn = make_verl_reward_fn(env=env)
    scores = reward_fn(data_batch)

Sources:
    - https://verl.readthedocs.io/en/latest/preparation/reward_function.html
    - https://github.com/verl-project/verl
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from deepgym.core import DeepGym
from deepgym.models import Environment


def make_verl_compute_score(
    env: Environment,
    dg: DeepGym | None = None,
) -> Callable[[str, str, str, Any], float]:
    """Create a reward function matching verl's compute_score interface.

    verl calls: compute_score(data_source, solution_str, ground_truth, extra_info)

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).

    Returns:
        Callable matching verl's compute_score(data_source, solution_str,
        ground_truth, extra_info=None) -> float signature.
    """
    _dg = dg or DeepGym(mode='auto')

    def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: Any = None,
    ) -> float:
        """Score a single solution against the DeepGym environment.

        Args:
            data_source: Dataset name (unused by DeepGym, passed by verl).
            solution_str: The model's decoded response string.
            ground_truth: Expected output (unused — DeepGym verifier handles this).
            extra_info: Optional context (unused).

        Returns:
            Score between 0.0 and 1.0.
        """
        result = _dg.run(env, model_output=solution_str)
        return result.score

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
