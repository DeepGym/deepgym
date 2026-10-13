"""TRL (HuggingFace) GRPOTrainer integration.

TRL's GRPOTrainer accepts reward functions via the reward_funcs parameter.
Reward functions receive keyword arguments and return list[float].

Completions arrive as plain strings (standard datasets) or as lists of messages
(conversational datasets, with tool results and several assistant turns when the
model calls tools). Either way the last assistant turn is the answer, and its last
fenced code block (or all of it, if none) is what runs, as in the verl adapter.

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
from collections.abc import Awaitable, Callable, Mapping, Sequence

from deepgym.core import DeepGym
from deepgym.integrations.verl import extract_code
from deepgym.models import Environment

Completion = str | Sequence[Mapping[str, object]]
"""One TRL completion: decoded text, or a conversation of message dicts."""


def _answer_text(completion: Completion) -> str:
    """Return the last assistant turn's text; content given as parts has its text parts joined."""
    if isinstance(completion, str):
        return completion
    turns = [message for message in completion if message.get('role') == 'assistant']
    content = turns[-1].get('content') if turns else None
    if isinstance(content, str):
        return content
    return ''.join(part['text'] for part in content or () if part.get('type') == 'text')


def _solutions(completions: Sequence[Completion]) -> list[str]:
    """Return the code to run for each completion."""
    return [extract_code(_answer_text(completion)) for completion in completions]


def make_trl_reward_fn(env: Environment, dg: DeepGym | None = None) -> Callable[..., list[float]]:
    """Create a sync reward function compatible with TRL's GRPOTrainer.

    TRL calls: reward_func(completions=..., prompts=..., **kwargs)
    where completions are decoded strings or conversations (lists of message dicts).

    Args:
        env: DeepGym environment with verifier.
        dg: Optional DeepGym client (auto-created if None).

    Returns:
        Callable matching TRL's reward_funcs interface.
    """
    _dg = dg or DeepGym(mode='auto')

    def reward_fn(completions: Sequence[Completion], **kwargs: object) -> list[float]:
        """Score completions against the DeepGym environment.

        Args:
            completions: Decoded strings or conversations, one per sample.
            **kwargs: Additional TRL args (prompts, ground_truth, etc.)

        Returns:
            List of scores between 0.0 and 1.0.
        """
        if not completions:
            return []
        batch = _dg.run_batch(
            env,
            _solutions(completions),
            max_parallel=min(len(completions), 32),
            **kwargs,
        )
        return [r.score for r in batch.results]

    return reward_fn


def make_trl_async_reward_fn(
    env: Environment, dg: DeepGym | None = None
) -> Callable[..., Awaitable[list[float]]]:
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

    async def reward_fn(completions: Sequence[Completion], **kwargs: object) -> list[float]:
        """Score completions asynchronously against the DeepGym environment.

        Args:
            completions: Decoded strings or conversations, one per sample.
            **kwargs: Additional TRL args.

        Returns:
            List of scores between 0.0 and 1.0.
        """
        if not completions:
            return []
        solutions = _solutions(completions)
        loop = asyncio.get_running_loop()
        batch = await loop.run_in_executor(
            None,
            lambda: _dg.run_batch(
                env,
                solutions,
                max_parallel=min(len(completions), 32),
                **kwargs,
            ),
        )
        return [r.score for r in batch.results]

    return reward_fn
