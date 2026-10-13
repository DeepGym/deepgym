"""Universal reward function interface for RL training frameworks.

DeepGym produces reward signals (score 0.0-1.0) that plug into any RL algorithm:
GRPO, DAPO, PPO, RLOO, REINFORCE++, Dr.GRPO, MaxRL, ScaleRL, etc.

The training framework handles advantage computation, clipping, and loss aggregation.
DeepGym handles execution, verification, and scoring.

The reward functions here and in the TRL, Axolotl and DAPO adapters share two rules:

- What runs is the last fenced code block of the completion's last assistant turn, or
  all of it if there is none (:func:`solutions`).
- A completion that got no verdict (the sandbox or verifier failed, see
  ``RunResult.has_verdict``) is rewarded None, not 0.0. Leave it out of the loss and out
  of its group's baseline; TRL 1.6+ does both when every reward function returns None
  for a completion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from deepgym.async_core import AsyncDeepGym
from deepgym.core import DeepGym
from deepgym.integrations.verl import extract_code
from deepgym.models import BatchResult, Environment, RunResult

Completion = str | Sequence[Mapping[str, object]]
"""One completion: decoded text, or a conversation of message dicts (as TRL passes them)."""


def _answer_text(completion: Completion) -> str:
    """Return the last assistant turn's text; content given as parts has its text parts joined."""
    if isinstance(completion, str):
        return completion
    turns = [message for message in completion if message.get('role') == 'assistant']
    content = turns[-1].get('content') if turns else None
    if isinstance(content, str):
        return content
    return ''.join(part['text'] for part in content or () if part.get('type') == 'text')


def solutions(completions: Sequence[Completion]) -> list[str]:
    """Return the code to run for each completion."""
    return [extract_code(_answer_text(completion)) for completion in completions]


def reward(result: RunResult, component: str | None = None) -> float | None:
    """Return *result*'s score, or its reward *component* if it has one; None without a verdict."""
    if not result.has_verdict:
        return None
    if component is None:
        return result.score
    return (result.reward_components or {}).get(component, result.score)


def _shaped(result: RunResult) -> dict[str, float] | None:
    """Return *result*'s reward components (or its score); None without a verdict."""
    if not result.has_verdict:
        return None
    return result.reward_components or {'score': result.score}


def _per_test(result: RunResult) -> dict[str, float] | None:
    """Return *result*'s per-case scores plus 'overall'; None without a verdict."""
    if not result.has_verdict:
        return None
    rewards = {c.id or f'case_{i}': c.score for i, c in enumerate(result.cases or ())}
    rewards['overall'] = result.score
    return rewards


class RewardFunction:
    """Drop-in reward function for any RL training framework.

    Usage:
        reward_fn = RewardFunction(env=env, max_parallel=100)
        scores = reward_fn(model_outputs)  # list[float | None]
    """

    def __init__(
        self,
        env: Environment,
        dg: DeepGym | None = None,
        max_parallel: int = 10,
    ) -> None:
        self._env = env
        self._dg = dg or DeepGym(mode='auto')
        self._max_parallel = max_parallel

    def __call__(self, outputs: Sequence[Completion], **kwargs: object) -> list[float | None]:
        """Score a batch of model outputs: a score in [0, 1] each, None without a verdict."""
        return [reward(r) for r in self.call_with_details(outputs, **kwargs).results]

    def call_with_details(self, outputs: Sequence[Completion], **kwargs: object) -> BatchResult:
        """Score with full details (reward_components, metrics, has_verdict, etc.)."""
        return self._dg.run_batch(
            self._env,
            solutions(outputs),
            max_parallel=self._max_parallel,
            **kwargs,
        )

    def shaped_rewards(
        self, outputs: Sequence[Completion], **kwargs: object
    ) -> list[dict[str, float] | None]:
        """Return shaped reward components for each output; None without a verdict.

        Useful for GRPO/DAPO which benefit from multi-dimensional rewards:
        e.g. {"correctness": 0.8, "efficiency": 0.9, "style": 0.7}
        """
        return [_shaped(r) for r in self.call_with_details(outputs, **kwargs).results]

    def per_test_rewards(
        self, outputs: Sequence[Completion], **kwargs: object
    ) -> list[dict[str, float] | None]:
        """Return per-test-case score breakdown for each output.

        Each dict maps test case IDs to their individual scores plus an
        'overall' key with the aggregate score. When verifiers emit a
        ``cases`` list, this enables fine-grained reward shaping for GRPO
        training where individual test outcomes inform the reward signal.

        Args:
            outputs: Model completions (text or conversations).

        Returns:
            List of dicts, one per output. Each dict contains per-test scores
            like ``{'test_0': 1.0, 'test_1': 0.0, 'overall': 0.5}``.
            Falls back to ``{'overall': <score>}`` when no per-test cases
            are available, and is None for an output that got no verdict.
        """
        return [_per_test(r) for r in self.call_with_details(outputs, **kwargs).results]


class AsyncRewardFunction:
    """Async version of RewardFunction for frameworks using asyncio."""

    def __init__(
        self,
        env: Environment,
        dg: AsyncDeepGym | None = None,
        max_parallel: int = 10,
    ) -> None:
        self._env = env
        self._dg = dg or AsyncDeepGym(mode='auto')
        self._max_parallel = max_parallel

    async def __call__(self, outputs: Sequence[Completion]) -> list[float | None]:
        """Score a batch of model outputs: a score in [0, 1] each, None without a verdict."""
        return [reward(r) for r in (await self.call_with_details(outputs)).results]

    async def call_with_details(self, outputs: Sequence[Completion]) -> BatchResult:
        """Score with full details (reward_components, metrics, has_verdict, etc.)."""
        return await self._dg.run_batch(
            self._env, solutions(outputs), max_parallel=self._max_parallel
        )

    async def shaped_rewards(self, outputs: Sequence[Completion]) -> list[dict[str, float] | None]:
        """Return shaped reward components for each output; None without a verdict."""
        return [_shaped(r) for r in (await self.call_with_details(outputs)).results]

    async def per_test_rewards(
        self, outputs: Sequence[Completion]
    ) -> list[dict[str, float] | None]:
        """Return per-test-case score breakdown for each output.

        Async version of :meth:`RewardFunction.per_test_rewards`. Each dict
        maps test case IDs to their individual scores plus an 'overall' key.

        Args:
            outputs: Model completions (text or conversations).

        Returns:
            List of dicts, one per output, with per-test scores and 'overall';
            None for an output that got no verdict.
        """
        return [_per_test(r) for r in (await self.call_with_details(outputs)).results]
