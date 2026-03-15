"""TRL (HuggingFace) GRPOTrainer integration.

TRL's GRPOTrainer accepts reward functions via the reward_funcs parameter.
Reward functions receive keyword arguments and return list[float].

Supported signatures (from TRL docs):
    def reward_func(completions, **kwargs) -> list[float]
    def reward_func(completions, prompts, **kwargs) -> list[float]
    def reward_func(completions, ground_truth, **kwargs) -> list[float]
    async def reward_func(completions, **kwargs) -> list[float]

Usage:
    from deepgym.integrations.trl import make_trl_reward_fn

    reward_fn = make_trl_reward_fn(env=env)

    trainer = GRPOTrainer(
        model='Qwen/Qwen2-0.5B-Instruct',
        reward_funcs=[reward_fn],
        train_dataset=dataset,
    )

Sources:
    - https://huggingface.co/docs/trl/main/en/grpo_trainer
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from deepgym.core import DeepGym
from deepgym.models import Environment


def make_trl_reward_fn(env: Environment, dg: DeepGym | None = None) -> Callable[..., list[float]]:
    """Create a sync reward function compatible with TRL's GRPOTrainer.

    TRL calls: reward_func(completions=..., prompts=..., **kwargs)
    where completions is a list of decoded model output strings.

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).

    Returns:
        Callable matching TRL's reward_funcs interface.
    """
    _dg = dg or DeepGym(mode='auto')

    def reward_fn(completions: list[str], **kwargs: object) -> list[float]:
        """Score completions against the DeepGym environment.

        Args:
            completions: List of decoded model output strings.
            **kwargs: Additional TRL args (prompts, ground_truth, etc.)

        Returns:
            List of scores between 0.0 and 1.0.
        """
        if not completions:
            return []
        batch = _dg.run_batch(env, completions, max_parallel=min(len(completions), 32))
        return [r.score for r in batch.results]

    return reward_fn


def make_trl_async_reward_fn(
    env: Environment, dg: DeepGym | None = None
) -> Callable[..., list[float]]:
    """Create an async reward function for TRL's GRPOTrainer.

    Async reward functions run concurrently when multiple are provided,
    reducing latency for I/O-bound verification (e.g., Daytona sandboxes).

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).

    Returns:
        Async callable matching TRL's reward_funcs interface.
    """
    _dg = dg or DeepGym(mode='auto')

    async def reward_fn(completions: list[str], **kwargs: object) -> list[float]:
        """Score completions asynchronously against the DeepGym environment.

        Args:
            completions: List of decoded model output strings.
            **kwargs: Additional TRL args.

        Returns:
            List of scores between 0.0 and 1.0.
        """
        if not completions:
            return []
        loop = asyncio.get_running_loop()
        batch = await loop.run_in_executor(
            None,
            lambda: _dg.run_batch(env, completions, max_parallel=min(len(completions), 32)),
        )
        return [r.score for r in batch.results]

    return reward_fn
