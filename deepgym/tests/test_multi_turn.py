"""Tests for deepgym.multi_turn -- multi-turn environment execution."""

import pytest

from deepgym.exceptions import DeepGymError
from deepgym.models import (
    Action,
    MultiTurnEnvironment,
    Observation,
    RunResult,
    Trajectory,
)
from deepgym.multi_turn import MultiTurnRunner


@pytest.fixture()
def runner() -> MultiTurnRunner:
    """Return a MultiTurnRunner with safe_mode enabled."""
    return MultiTurnRunner(safe_mode=True)


@pytest.fixture()
def unsafe_runner() -> MultiTurnRunner:
    """Return a MultiTurnRunner with safe_mode disabled."""
    return MultiTurnRunner(safe_mode=False)


class TestMultiTurnModels:
    """Verify Pydantic model construction for multi-turn types."""

    def test_observation_defaults(self) -> None:
        obs = Observation(content='hello', step=0)
        assert obs.done is False
        assert obs.metadata is None

    def test_observation_with_done(self) -> None:
        obs = Observation(content='done', step=5, done=True)
        assert obs.done is True

    def test_action_defaults(self) -> None:
        action = Action(content='print(1)')
        assert action.action_type == 'code'

    def test_action_bash_type(self) -> None:
        action = Action(content='ls -la', action_type='bash')
        assert action.action_type == 'bash'

    def test_trajectory_defaults(self) -> None:
        traj = Trajectory()
        assert traj.steps == []
        assert traj.total_reward == 0.0
        assert traj.step_rewards == []
        assert traj.final_observation is None

    def test_multi_turn_environment_defaults(self) -> None:
        env = MultiTurnEnvironment(task='do something')
        assert env.max_steps == 10
        assert env.timeout_per_step == 30
        assert env.type == 'multi-turn'

    def test_multi_turn_environment_custom_steps(self) -> None:
        env = MultiTurnEnvironment(task='test', max_steps=5, timeout_per_step=10)
        assert env.max_steps == 5
        assert env.timeout_per_step == 10

    def test_multi_turn_environment_min_steps_validation(self) -> None:
        with pytest.raises(Exception):
            MultiTurnEnvironment(task='test', max_steps=0)


class TestMultiTurnRunnerCreation:
    """Verify MultiTurnRunner construction."""

    def test_creation_with_safe_mode(self) -> None:
        runner = MultiTurnRunner(safe_mode=True)
        assert runner._safe_mode is True

    def test_creation_without_safe_mode(self) -> None:
        runner = MultiTurnRunner(safe_mode=False)
        assert runner._safe_mode is False


class TestSafeModeBlocksBash:
    """Verify safe_mode blocks bash actions."""

    def test_bash_action_raises_in_safe_mode(self, runner: MultiTurnRunner) -> None:
        env = MultiTurnEnvironment(
            task='test',
            max_steps=1,
            final_verifier_code='',
        )

        def bash_agent(obs: Observation) -> Action:
            return Action(content='echo hello', action_type='bash')

        with pytest.raises(DeepGymError, match='Bash actions are disabled'):
            runner.run(env, bash_agent)


class TestCheckDone:
    """Verify the done-detection logic."""

    def test_done_marker(self) -> None:
        assert MultiTurnRunner._check_done('some output\nDONE\n') is True

    def test_done_json(self) -> None:
        assert MultiTurnRunner._check_done('{"done": true}') is True

    def test_not_done(self) -> None:
        assert MultiTurnRunner._check_done('some regular output') is False

    def test_done_false_json(self) -> None:
        assert MultiTurnRunner._check_done('{"done": false}') is False

    def test_empty_output(self) -> None:
        assert MultiTurnRunner._check_done('') is False

    def test_invalid_json_not_done(self) -> None:
        assert MultiTurnRunner._check_done('{not valid json}') is False


class TestMultiTurnExecution:
    """Verify end-to-end multi-turn episode execution."""

    def test_simple_code_episode(self, runner: MultiTurnRunner) -> None:
        env = MultiTurnEnvironment(
            task='Print numbers',
            setup_code='',
            max_steps=3,
            timeout_per_step=10,
            final_verifier_code=(
                'import json\n'
                'result = {"schema_version": "1.0", "score": 1.0, '
                '"passed": True, "details": "ok"}\n'
                'print(json.dumps(result))\n'
            ),
        )

        step_count = 0

        def agent(obs: Observation) -> Action:
            nonlocal step_count
            step_count += 1
            if step_count >= 2:
                return Action(content='print("DONE")')
            return Action(content='print("step")')

        trajectory, result = runner.run(env, agent)
        assert isinstance(trajectory, Trajectory)
        assert isinstance(result, RunResult)
        assert result.score == 1.0
        assert result.passed is True
        assert len(trajectory.steps) >= 1

    def test_episode_without_final_verifier(self, runner: MultiTurnRunner) -> None:
        env = MultiTurnEnvironment(
            task='No verifier test',
            max_steps=1,
            final_verifier_code='',
        )

        def agent(obs: Observation) -> Action:
            return Action(content='print("DONE")')

        trajectory, result = runner.run(env, agent)
        assert result.score == 0.0
        assert result.passed is False

    def test_episode_respects_max_steps(self, runner: MultiTurnRunner) -> None:
        env = MultiTurnEnvironment(
            task='Test max steps',
            max_steps=2,
            timeout_per_step=5,
            final_verifier_code=(
                'import json\n'
                'print(json.dumps({"schema_version": "1.0", "score": 0.5, '
                '"passed": False, "details": "hit max"}))\n'
            ),
        )

        call_count = 0

        def agent(obs: Observation) -> Action:
            nonlocal call_count
            call_count += 1
            return Action(content='print("working")')

        trajectory, result = runner.run(env, agent)
        assert call_count == 2
        assert len(trajectory.steps) == 2

    def test_bash_allowed_without_safe_mode(self, unsafe_runner: MultiTurnRunner) -> None:
        env = MultiTurnEnvironment(
            task='Bash test',
            max_steps=1,
            timeout_per_step=5,
            final_verifier_code=(
                'import json\n'
                'print(json.dumps({"schema_version": "1.0", "score": 1.0, '
                '"passed": True, "details": "ok"}))\n'
            ),
        )

        def agent(obs: Observation) -> Action:
            return Action(content='echo DONE', action_type='bash')

        trajectory, result = unsafe_runner.run(env, agent)
        assert result.passed is True
