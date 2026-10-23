"""Exercise the public runner without optional SDKs or remote sandboxes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from deepgym import Action, DeepGym, MultiTurnEnvironment
from deepgym.agents.base import AgentResult, Budget
from deepgym.models import Observation, Trajectory
from deepgym.session import Session


class DoneAgent:
    def run(
        self, session: Session, initial: Observation, *, task: str, budget: Budget
    ) -> AgentResult:
        assert initial.step == 0
        assert task == 'finish'
        assert budget.max_turns == 2
        session.step(Action(action_type='done'))
        return AgentResult(trajectory=session.trajectory, termination_reason='completed')


def test_public_import_needs_no_provider_sdk() -> None:
    source = str(Path(__file__).resolve().parents[2] / 'src')
    checked = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import deepgym.agents; '
            'assert "anthropic" not in sys.modules; assert "openai" not in sys.modules',
        ],
        env={**os.environ, 'PYTHONPATH': source, 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert checked.returncode == 0, checked.stderr


def test_harness_runner_returns_agent_result_and_grade() -> None:
    result, grade = DeepGym(mode='local').run_episode(
        MultiTurnEnvironment(task='finish'), DoneAgent(), budget=Budget(max_turns=2)
    )
    assert isinstance(result, AgentResult)
    assert result.trajectory.final_observation.done
    assert grade.score == 0


def test_legacy_callable_keeps_trajectory_return() -> None:
    result, _ = DeepGym(mode='local').run_episode(
        MultiTurnEnvironment(task='finish'), lambda obs: Action(action_type='done')
    )
    assert isinstance(result, Trajectory)


def test_legacy_callable_cannot_silently_ignore_budget() -> None:
    with pytest.raises(ValueError, match='budget requires'):
        DeepGym(mode='local').run_episode(
            MultiTurnEnvironment(task='finish'),
            lambda obs: Action(action_type='done'),
            budget=Budget(),
        )


class StoppedAgent:
    """An agent of the caller's own that builds its AgentResult directly."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def run(
        self, session: Session, initial: Observation, *, task: str, budget: Budget
    ) -> AgentResult:
        session.step(Action(action_type='done'))
        return AgentResult(
            trajectory=session.trajectory, termination_reason=self.reason, error='stopped'
        )


@pytest.mark.parametrize('reason', ['provider_error', 'setup_error'])
def test_any_agent_that_stopped_through_no_policy_fault_gets_no_verdict(reason: str) -> None:
    """Enforced where every agent's run is scored, not only for agents that use our helper."""
    env = MultiTurnEnvironment(task='finish', final_verifier_code='print(1.0)')

    _, grade = DeepGym(mode='local').run_episode(env, StoppedAgent(reason), budget=Budget())

    assert not grade.has_verdict and grade.error_type == reason


def test_a_run_the_harness_cut_short_is_graded_as_truncated() -> None:
    passes = 'import json\nprint(json.dumps({"score": 1.0, "passed": True}))'
    env = MultiTurnEnvironment(task='finish', final_verifier_code=passes)

    _, grade = DeepGym(mode='local').run_episode(env, StoppedAgent('output_limit'), budget=Budget())

    assert grade.has_verdict and grade.truncated
