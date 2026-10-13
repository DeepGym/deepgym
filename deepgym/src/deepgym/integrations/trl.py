"""TRL (HuggingFace) GRPOTrainer integration.

TRL's GRPOTrainer accepts reward functions via the reward_funcs parameter.
Reward functions receive keyword arguments and return one reward per completion.

Completions arrive as plain strings (standard datasets) or as lists of messages
(conversational datasets, with tool results and several assistant turns when the
model calls tools). Either way the last assistant turn is the answer, and its last
fenced code block (or all of it, if none) is what runs, as in the verl adapter.

A completion that got no verdict (the sandbox or verifier failed) is rewarded None
instead of 0.0. TRL 1.6+ leaves a completion that every reward function scored None
out of its group's baseline and gives it zero advantage, so a failed sandbox neither
penalizes the completion nor shifts its siblings' advantages. (Older TRL treats the
None as 0.0.)

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
    - https://github.com/huggingface/trl (trl/trainer/grpo_trainer.py: _calculate_rewards,
      and the unscorable-completion mask in _generate_and_score_completions)
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from deepgym.core import DeepGym
from deepgym.integrations.reward import Completion, reward, solutions
from deepgym.models import Environment


def make_trl_reward_fn(
    env: Environment,
    dg: DeepGym | None = None,
    max_parallel: int = 32,
    component: str | None = None,
) -> Callable[..., list[float | None]]:
    """Create a sync reward function compatible with TRL's GRPOTrainer.

    TRL calls: reward_func(completions=..., prompts=..., **kwargs)
    where completions are decoded strings or conversations (lists of message dicts).

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).
        max_parallel: Most sandbox runs at once.
        component: Reward component to return instead of the score (a result without
            it falls back to its score), for one reward function per component.

    Returns:
        Callable matching TRL's reward_funcs interface: a score between 0.0 and 1.0
        per completion, or None for one that got no verdict.
    """
    _dg = dg or DeepGym(mode='auto')

    def reward_fn(completions: Sequence[Completion], **kwargs: object) -> list[float | None]:
        """Score completions against the DeepGym environment.

        Args:
            completions: Decoded strings or conversations, one per sample.
            **kwargs: Additional TRL args (prompts, ground_truth, etc.)

        Returns:
            One score between 0.0 and 1.0 per completion; None where there was no verdict.
        """
        if not completions:
            return []
        batch = _dg.run_batch(
            env,
            solutions(completions),
            max_parallel=min(len(completions), max_parallel),
            **kwargs,
        )
        return [reward(result, component) for result in batch.results]

    return reward_fn


def make_trl_async_reward_fn(
    env: Environment,
    dg: DeepGym | None = None,
    max_parallel: int = 32,
) -> Callable[..., Awaitable[list[float | None]]]:
    """Create an async reward function for TRL's GRPOTrainer.

    Async reward functions run concurrently when multiple are provided,
    reducing latency for I/O-bound verification (e.g., Daytona sandboxes).

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).
        max_parallel: Most sandbox runs at once.

    Returns:
        Async callable matching TRL's reward_funcs interface, scoring as
        :func:`make_trl_reward_fn` does.
    """
    score = make_trl_reward_fn(env, dg, max_parallel)

    async def reward_fn(completions: Sequence[Completion], **kwargs: object) -> list[float | None]:
        """Score completions in a worker thread; see :func:`make_trl_reward_fn`."""
        return await asyncio.to_thread(score, completions, **kwargs)

    return reward_fn
