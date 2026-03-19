"""Universal reward function interface for RL training frameworks.

DeepGym produces reward signals (score 0.0-1.0) that plug into any RL algorithm:
GRPO, DAPO, PPO, RLOO, REINFORCE++, Dr.GRPO, MaxRL, ScaleRL, etc.

The training framework handles advantage computation, clipping, and loss aggregation.
DeepGym handles execution, verification, and scoring.
"""

from __future__ import annotations

from deepgym.async_core import AsyncDeepGym
from deepgym.core import DeepGym
from deepgym.models import BatchResult, Environment


class RewardFunction:
    """Drop-in reward function for any RL training framework.

    Usage:
        reward_fn = RewardFunction(env=env, max_parallel=100)
        scores = reward_fn(model_outputs)  # list[float]
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

    def __call__(self, outputs: list[str]) -> list[float]:
        """Score a batch of model outputs. Return list of scores 0.0-1.0."""
        if not outputs:
            return []
        batch = self._dg.run_batch(self._env, outputs, max_parallel=self._max_parallel)
        return [r.score for r in batch.results]

    def call_with_details(self, outputs: list[str]) -> BatchResult:
        """Score with full details (reward_components, metrics, etc.)."""
        if not outputs:
            return BatchResult(
                results=[], total=0, passed=0, failed=0, avg_score=0.0, execution_time_ms=0.0
            )
        return self._dg.run_batch(self._env, outputs, max_parallel=self._max_parallel)

    def shaped_rewards(self, outputs: list[str]) -> list[dict[str, float]]:
        """Return shaped reward components for each output.

        Useful for GRPO/DAPO which benefit from multi-dimensional rewards:
        e.g. {"correctness": 0.8, "efficiency": 0.9, "style": 0.7}
        """
        if not outputs:
            return []
        batch = self._dg.run_batch(self._env, outputs, max_parallel=self._max_parallel)
        return [r.reward_components or {'score': r.score} for r in batch.results]

    def per_test_rewards(self, outputs: list[str]) -> list[dict[str, float]]:
        """Return per-test-case score breakdown for each output.

        Each dict maps test case IDs to their individual scores plus an
        'overall' key with the aggregate score. When verifiers emit a
        ``cases`` list, this enables fine-grained reward shaping for GRPO
        training where individual test outcomes inform the reward signal.

        Args:
            outputs: List of model-generated solution source code strings.

        Returns:
            List of dicts, one per output. Each dict contains per-test scores
            like ``{'test_0': 1.0, 'test_1': 0.0, 'overall': 0.5}``.
            Falls back to ``{'overall': <score>}`` when no per-test cases
            are available.
        """
        if not outputs:
            return []
        batch = self._dg.run_batch(self._env, outputs, max_parallel=self._max_parallel)
        result: list[dict[str, float]] = []
        for r in batch.results:
            if r.cases:
                rewards = {c.id or f'case_{i}': c.score for i, c in enumerate(r.cases)}
                rewards['overall'] = r.score
                result.append(rewards)
            else:
                result.append({'overall': r.score})
        return result


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

    async def __call__(self, outputs: list[str]) -> list[float]:
        """Score a batch of model outputs asynchronously. Return list of scores 0.0-1.0."""
        if not outputs:
            return []
        batch = await self._dg.run_batch(self._env, outputs, max_parallel=self._max_parallel)
        return [r.score for r in batch.results]

    async def call_with_details(self, outputs: list[str]) -> BatchResult:
        """Score with full details (reward_components, metrics, etc.)."""
        if not outputs:
            return BatchResult(
                results=[], total=0, passed=0, failed=0, avg_score=0.0, execution_time_ms=0.0
            )
        return await self._dg.run_batch(self._env, outputs, max_parallel=self._max_parallel)

    async def shaped_rewards(self, outputs: list[str]) -> list[dict[str, float]]:
        """Return shaped reward components for each output."""
        if not outputs:
            return []
        batch = await self._dg.run_batch(self._env, outputs, max_parallel=self._max_parallel)
        return [r.reward_components or {'score': r.score} for r in batch.results]

    async def per_test_rewards(self, outputs: list[str]) -> list[dict[str, float]]:
        """Return per-test-case score breakdown for each output.

        Async version of :meth:`RewardFunction.per_test_rewards`. Each dict
        maps test case IDs to their individual scores plus an 'overall' key.

        Args:
            outputs: List of model-generated solution source code strings.

        Returns:
            List of dicts, one per output, with per-test scores and 'overall'.
        """
        if not outputs:
            return []
        batch = await self._dg.run_batch(self._env, outputs, max_parallel=self._max_parallel)
        result: list[dict[str, float]] = []
        for r in batch.results:
            if r.cases:
                rewards = {c.id or f'case_{i}': c.score for i, c in enumerate(r.cases)}
                rewards['overall'] = r.score
                result.append(rewards)
            else:
                result.append({'overall': r.score})
        return result
