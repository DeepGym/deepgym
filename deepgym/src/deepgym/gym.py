"""Gymnasium-compatible environment interface for DeepGym.

Provide the standard RL API that all training frameworks expect::

    env = DeepGymEnv(environment)
    obs = env.reset()
    obs, reward, done, info = env.step(action)

Compatible with: OpenEnv, ARES, Gymnasium, TRL, Unsloth, verl, OpenRLHF.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from deepgym.async_core import AsyncDeepGym
from deepgym.core import DeepGym
from deepgym.models import CaseResult, Environment, RunResult


class GymObservation(BaseModel):
    """Represent what the agent sees at each step."""

    task: str
    """The task description / prompt."""

    feedback: str = ''
    """Feedback from previous step (score, error messages, etc.)."""

    step: int = 0
    """Current step number."""

    done: bool = False
    """Whether the episode is finished."""


class GymInfo(BaseModel):
    """Additional info returned with each step."""

    score: float = 0.0
    passed: bool = False
    execution_time_ms: float = 0.0
    details: str = ''
    reward_components: dict[str, float] | None = None
    cases: list[CaseResult] | None = None
    """Per-test-case breakdown for fine-grained reward shaping."""


def _result_to_info(result: RunResult) -> GymInfo:
    """Convert a RunResult to a GymInfo.

    Args:
        result: The run result from the DeepGym client.

    Returns:
        A GymInfo with score, pass/fail, timing, and details.
    """
    return GymInfo(
        score=result.score,
        passed=result.passed,
        execution_time_ms=result.execution_time_ms,
        details=result.output,
        reward_components=result.reward_components,
        cases=result.cases,
    )


def _result_to_obs(task: str, result: RunResult, step: int, done: bool) -> GymObservation:
    """Convert a RunResult to a GymObservation.

    Args:
        task: The task description.
        result: The run result from the DeepGym client.
        step: Current step number.
        done: Whether the episode is finished.

    Returns:
        A GymObservation with task, feedback, step, and done flag.
    """
    return GymObservation(
        task=task,
        feedback=result.output,
        step=step,
        done=done,
    )


_StepResult = tuple[GymObservation, float, bool, GymInfo]


class DeepGymEnv:
    """Gymnasium-style environment wrapper for DeepGym.

    Provide the standard reset()/step() interface that all RL training
    frameworks expect.

    Usage::

        from deepgym.gym import DeepGymEnv
        from deepgym import load_environment

        env = DeepGymEnv(load_environment('coin_change'))
        obs = env.reset()

        # Agent generates code
        action = model.generate(obs.task)

        # Execute and get reward
        obs, reward, done, info = env.step(action)
        print(f'Reward: {reward}, Done: {done}')

    For batch/GRPO (multiple samples per prompt)::

        obs = env.reset()
        results = env.step_batch(['solution1', 'solution2', ...])
        rewards = [r[1] for r in results]  # list of floats

    Args:
        environment: The DeepGym environment spec.
        dg: Optional DeepGym client. Auto-created if None.
        max_steps: Maximum steps per episode (1 for single-turn coding).
    """

    def __init__(
        self,
        environment: Environment,
        dg: DeepGym | None = None,
        max_steps: int = 1,
    ) -> None:
        """Create a Gymnasium-style DeepGym environment.

        Args:
            environment: The DeepGym environment spec.
            dg: Optional DeepGym client. Auto-created if None.
            max_steps: Maximum steps per episode (1 for single-turn coding).
        """
        self._env = environment
        self._dg = dg or DeepGym(mode='auto')
        self._max_steps = max_steps
        self._step = 0
        self._done = False
        self._last_result: RunResult | None = None

    def reset(self) -> GymObservation:
        """Reset the environment and return initial observation.

        Returns:
            GymObservation with the task description.
        """
        self._step = 0
        self._done = False
        self._last_result = None
        return GymObservation(
            task=self._env.task,
            feedback='',
            step=0,
            done=False,
        )

    def step(self, action: str) -> _StepResult:
        """Execute an action (code submission) and return result.

        Args:
            action: The model's generated code.

        Returns:
            Tuple of (observation, reward, done, info).

        Raises:
            RuntimeError: If called after the episode is done without reset.
        """
        if self._done:
            raise RuntimeError('Episode is done. Call reset() first.')

        self._step += 1
        result = self._dg.run(self._env, model_output=action)
        self._last_result = result
        self._done = self._step >= self._max_steps or result.passed

        obs = _result_to_obs(self._env.task, result, self._step, self._done)
        info = _result_to_info(result)
        return obs, result.score, self._done, info

    def step_batch(self, actions: Sequence[str], max_parallel: int = 10) -> list[_StepResult]:
        """Execute multiple actions in parallel (for GRPO/batch training).

        Args:
            actions: List of model-generated code solutions.
            max_parallel: Maximum concurrent executions.

        Returns:
            List of (observation, reward, done, info) tuples, one per action.
        """
        batch = self._dg.run_batch(self._env, actions, max_parallel=max_parallel)
        results: list[_StepResult] = []
        for r in batch.results:
            obs = _result_to_obs(self._env.task, r, self._step, done=True)
            info = _result_to_info(r)
            results.append((obs, r.score, True, info))
        return results

    @property
    def state(self) -> dict[str, Any]:
        """Return current environment state (OpenEnv compatible)."""
        return {
            'step': self._step,
            'done': self._done,
            'last_score': self._last_result.score if self._last_result else None,
            'task': self._env.task,
        }

    def close(self) -> None:
        """Clean up resources."""

    def __enter__(self) -> DeepGymEnv:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager."""
        self.close()


class AsyncDeepGymEnv:
    """Async Gymnasium-style environment for high-throughput training.

    Provide the same interface as :class:`DeepGymEnv` but uses
    ``async``/``await`` throughout for non-blocking execution.

    Args:
        environment: The DeepGym environment spec.
        dg: Optional AsyncDeepGym client. Auto-created if None.
        max_steps: Maximum steps per episode (1 for single-turn coding).
    """

    def __init__(
        self,
        environment: Environment,
        dg: AsyncDeepGym | None = None,
        max_steps: int = 1,
    ) -> None:
        """Create an async Gymnasium-style DeepGym environment.

        Args:
            environment: The DeepGym environment spec.
            dg: Optional AsyncDeepGym client. Auto-created if None.
            max_steps: Maximum steps per episode (1 for single-turn coding).
        """
        self._env = environment
        self._dg = dg or AsyncDeepGym(mode='auto')
        self._max_steps = max_steps
        self._step = 0
        self._done = False
        self._last_result: RunResult | None = None

    async def reset(self) -> GymObservation:
        """Reset the environment and return initial observation.

        Returns:
            GymObservation with the task description.
        """
        self._step = 0
        self._done = False
        self._last_result = None
        return GymObservation(
            task=self._env.task,
            feedback='',
            step=0,
            done=False,
        )

    async def step(self, action: str) -> _StepResult:
        """Execute an action (code submission) and return result.

        Args:
            action: The model's generated code.

        Returns:
            Tuple of (observation, reward, done, info).

        Raises:
            RuntimeError: If called after the episode is done without reset.
        """
        if self._done:
            raise RuntimeError('Episode is done. Call reset() first.')

        self._step += 1
        result = await self._dg.run(self._env, model_output=action)
        self._last_result = result
        self._done = self._step >= self._max_steps or result.passed

        obs = _result_to_obs(self._env.task, result, self._step, self._done)
        info = _result_to_info(result)
        return obs, result.score, self._done, info

    async def step_batch(self, actions: Sequence[str], max_parallel: int = 10) -> list[_StepResult]:
        """Execute multiple actions in parallel (for GRPO/batch training).

        Args:
            actions: List of model-generated code solutions.
            max_parallel: Maximum concurrent executions.

        Returns:
            List of (observation, reward, done, info) tuples, one per action.
        """
        batch = await self._dg.run_batch(self._env, actions, max_parallel=max_parallel)
        results: list[_StepResult] = []
        for r in batch.results:
            obs = _result_to_obs(self._env.task, r, self._step, done=True)
            info = _result_to_info(r)
            results.append((obs, r.score, True, info))
        return results

    @property
    def state(self) -> dict[str, Any]:
        """Return current environment state (OpenEnv compatible)."""
        return {
            'step': self._step,
            'done': self._done,
            'last_score': self._last_result.score if self._last_result else None,
            'task': self._env.task,
        }

    async def close(self) -> None:
        """Clean up resources."""

    async def __aenter__(self) -> AsyncDeepGymEnv:
        """Enter the async context manager."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the async context manager."""
        await self.close()
