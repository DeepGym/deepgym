"""Tests for the Gymnasium-compatible DeepGymEnv and AsyncDeepGymEnv."""

import pytest
import pytest_asyncio

from deepgym.core import DeepGym
from deepgym.async_core import AsyncDeepGym
from deepgym.gym import (
    AsyncDeepGymEnv,
    DeepGymEnv,
    GymInfo,
    GymObservation,
)
from deepgym.models import Environment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_dg() -> DeepGym:
    """Return a DeepGym client in local mode."""
    return DeepGym(mode='local')


@pytest.fixture()
def async_dg() -> AsyncDeepGym:
    """Return an AsyncDeepGym client in local mode."""
    return AsyncDeepGym(mode='local')


@pytest.fixture()
def simple_env() -> Environment:
    """Return a simple inline-verifier environment."""
    return Environment(
        task='Write a function `solve(x)` that returns x * 2.',
        verifier_code=(
            'import importlib.util\n'
            'spec = importlib.util.spec_from_file_location("solution", solution_path)\n'
            'mod = importlib.util.module_from_spec(spec)\n'
            'spec.loader.exec_module(mod)\n'
            'if hasattr(mod, "solve") and mod.solve(2) == 4:\n'
            '    return 1.0\n'
            'return 0.0\n'
        ),
        difficulty='easy',
    )


@pytest.fixture()
def correct_solution() -> str:
    """Return a correct solution for the simple_env."""
    return 'def solve(x):\n    return x * 2\n'


@pytest.fixture()
def wrong_solution() -> str:
    """Return a wrong solution for the simple_env."""
    return 'def solve(x):\n    return x + 1\n'


# ---------------------------------------------------------------------------
# GymObservation / GymInfo model tests
# ---------------------------------------------------------------------------


class TestGymModels:
    """Test the Pydantic models for the gym interface."""

    def test_gym_observation_defaults(self) -> None:
        """Verify default values on GymObservation."""
        obs = GymObservation(task='Do something')
        assert obs.task == 'Do something'
        assert obs.feedback == ''
        assert obs.step == 0
        assert obs.done is False

    def test_gym_info_defaults(self) -> None:
        """Verify default values on GymInfo."""
        info = GymInfo()
        assert info.score == 0.0
        assert info.passed is False
        assert info.execution_time_ms == 0.0
        assert info.details == ''
        assert info.reward_components is None


# ---------------------------------------------------------------------------
# DeepGymEnv (sync) tests
# ---------------------------------------------------------------------------


class TestDeepGymEnv:
    """Test the synchronous Gymnasium-style wrapper."""

    def test_reset_returns_observation(self, simple_env: Environment, local_dg: DeepGym) -> None:
        """Reset returns a GymObservation with task text."""
        env = DeepGymEnv(simple_env, dg=local_dg)
        obs = env.reset()
        assert isinstance(obs, GymObservation)
        assert obs.task == simple_env.task
        assert obs.step == 0
        assert obs.done is False

    def test_step_correct_solution(
        self,
        simple_env: Environment,
        local_dg: DeepGym,
        correct_solution: str,
    ) -> None:
        """Step with a correct solution returns reward=1.0 and done=True."""
        env = DeepGymEnv(simple_env, dg=local_dg)
        env.reset()
        obs, reward, done, info = env.step(correct_solution)

        assert isinstance(obs, GymObservation)
        assert reward == 1.0
        assert done is True
        assert isinstance(info, GymInfo)
        assert info.score == 1.0
        assert info.passed is True

    def test_step_wrong_solution(
        self,
        simple_env: Environment,
        local_dg: DeepGym,
        wrong_solution: str,
    ) -> None:
        """Step with a wrong solution returns reward=0.0."""
        env = DeepGymEnv(simple_env, dg=local_dg)
        env.reset()
        obs, reward, done, info = env.step(wrong_solution)

        assert reward == 0.0
        assert info.passed is False
        # max_steps=1 so done should be True
        assert done is True

    def test_step_after_done_raises(
        self,
        simple_env: Environment,
        local_dg: DeepGym,
        correct_solution: str,
    ) -> None:
        """Step after episode is done raises RuntimeError."""
        env = DeepGymEnv(simple_env, dg=local_dg)
        env.reset()
        env.step(correct_solution)
        with pytest.raises(RuntimeError, match='Episode is done'):
            env.step(correct_solution)

    def test_reset_clears_done(
        self,
        simple_env: Environment,
        local_dg: DeepGym,
        correct_solution: str,
    ) -> None:
        """Reset after done allows stepping again."""
        env = DeepGymEnv(simple_env, dg=local_dg)
        env.reset()
        env.step(correct_solution)
        obs = env.reset()
        assert obs.done is False
        # Should not raise
        env.step(correct_solution)

    def test_state_property(
        self,
        simple_env: Environment,
        local_dg: DeepGym,
        correct_solution: str,
    ) -> None:
        """State property reflects current episode state."""
        env = DeepGymEnv(simple_env, dg=local_dg)
        env.reset()

        state = env.state
        assert state['step'] == 0
        assert state['done'] is False
        assert state['last_score'] is None
        assert state['task'] == simple_env.task

        env.step(correct_solution)
        state = env.state
        assert state['step'] == 1
        assert state['done'] is True
        assert state['last_score'] == 1.0

    def test_context_manager(self, simple_env: Environment, local_dg: DeepGym) -> None:
        """DeepGymEnv works as a context manager."""
        with DeepGymEnv(simple_env, dg=local_dg) as env:
            obs = env.reset()
            assert isinstance(obs, GymObservation)

    def test_multi_step_episode(
        self,
        simple_env: Environment,
        local_dg: DeepGym,
        wrong_solution: str,
        correct_solution: str,
    ) -> None:
        """Multi-step episode: wrong then correct."""
        env = DeepGymEnv(simple_env, dg=local_dg, max_steps=3)
        env.reset()

        obs, reward, done, info = env.step(wrong_solution)
        assert reward == 0.0
        assert done is False  # max_steps=3, step=1, not passed

        obs, reward, done, info = env.step(correct_solution)
        assert reward == 1.0
        assert done is True  # passed=True triggers done

    def test_step_batch(
        self,
        simple_env: Environment,
        local_dg: DeepGym,
        correct_solution: str,
        wrong_solution: str,
    ) -> None:
        """step_batch returns results for all actions."""
        env = DeepGymEnv(simple_env, dg=local_dg)
        env.reset()
        results = env.step_batch([correct_solution, wrong_solution])

        assert len(results) == 2
        # First should pass
        assert results[0][1] == 1.0  # reward
        assert results[0][2] is True  # done
        # Second should fail
        assert results[1][1] == 0.0
        assert results[1][2] is True


# ---------------------------------------------------------------------------
# AsyncDeepGymEnv tests
# ---------------------------------------------------------------------------


class TestAsyncDeepGymEnv:
    """Test the asynchronous Gymnasium-style wrapper."""

    @pytest.mark.asyncio
    async def test_reset_returns_observation(
        self, simple_env: Environment, async_dg: AsyncDeepGym
    ) -> None:
        """Async reset returns a GymObservation with task text."""
        env = AsyncDeepGymEnv(simple_env, dg=async_dg)
        obs = await env.reset()
        assert isinstance(obs, GymObservation)
        assert obs.task == simple_env.task
        assert obs.step == 0

    @pytest.mark.asyncio
    async def test_step_correct_solution(
        self,
        simple_env: Environment,
        async_dg: AsyncDeepGym,
        correct_solution: str,
    ) -> None:
        """Async step with correct solution returns reward=1.0."""
        env = AsyncDeepGymEnv(simple_env, dg=async_dg)
        await env.reset()
        obs, reward, done, info = await env.step(correct_solution)

        assert reward == 1.0
        assert done is True
        assert info.passed is True

    @pytest.mark.asyncio
    async def test_step_after_done_raises(
        self,
        simple_env: Environment,
        async_dg: AsyncDeepGym,
        correct_solution: str,
    ) -> None:
        """Async step after done raises RuntimeError."""
        env = AsyncDeepGymEnv(simple_env, dg=async_dg)
        await env.reset()
        await env.step(correct_solution)
        with pytest.raises(RuntimeError, match='Episode is done'):
            await env.step(correct_solution)

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self, simple_env: Environment, async_dg: AsyncDeepGym
    ) -> None:
        """AsyncDeepGymEnv works as an async context manager."""
        async with AsyncDeepGymEnv(simple_env, dg=async_dg) as env:
            obs = await env.reset()
            assert isinstance(obs, GymObservation)

    @pytest.mark.asyncio
    async def test_state_property(
        self,
        simple_env: Environment,
        async_dg: AsyncDeepGym,
        correct_solution: str,
    ) -> None:
        """Async state property reflects current episode state."""
        env = AsyncDeepGymEnv(simple_env, dg=async_dg)
        await env.reset()
        assert env.state['step'] == 0

        await env.step(correct_solution)
        assert env.state['step'] == 1
        assert env.state['last_score'] == 1.0
