"""Multi-turn QA: the reference episode replays at full marks, scripted agents earn nothing."""

from pathlib import Path

import pytest

from deepgym.core import DeepGym
from deepgym.factory.scaffold import scaffold
from deepgym.models import Action, MultiTurnEnvironment, RunResult, Trajectory
from deepgym.qa import QAOptions, run_qa
from deepgym.qa.agents import _COMMANDS, do_everything, random_agent
from deepgym.registry import load_environment
from tests.computer_use_trajectories import NAMES, reference, variants


def _by_name(report) -> dict:
    return {check.name: check for check in report.checks}


def test_a_multi_turn_folder_runs_the_reference_and_three_idle_agents(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    path = scaffold('env', 'multi-turn', tmp_path)

    report = run_qa(path, local_dg, QAOptions(reruns=1))

    checks = _by_name(report)
    assert report.status == 'pass'
    assert report.ran == [
        'reference_passes',
        'deterministic',
        'empty_fails',
        'busy_fails',
        'random_fails',
    ]
    assert report.not_run == ['stub_fails', 'hack_resistant']
    for name in ('empty_fails', 'busy_fails', 'random_fails'):
        assert checks[name].details.startswith('score 0.00')


# Pays 0.1 per file in the work directory beyond the starting one: rewards activity.
_PAYS_FOR_ACTIVITY = """
import json, os
files = [name for name in os.listdir('.') if name != 'start.txt']
done = os.path.exists('answer.txt') and open('answer.txt').read() == '42'
score = 1.0 if done else min(0.9, 0.1 * len(files))
print(json.dumps({'schema_version': '1.0', 'score': score, 'passed': done}))
"""


def test_agents_that_are_busy_or_random_must_earn_nothing(local_dg: DeepGym) -> None:
    env = MultiTurnEnvironment(
        task='Write 42 to answer.txt.',
        files={'start.txt': 'go\n'},
        final_verifier_code=_PAYS_FOR_ACTIVITY,
        reference_actions=[Action(action_type='write_file', path='answer.txt', content='42')],
    )

    checks = run_qa_on(env, local_dg)

    assert checks['reference_passes'].passed and checks['empty_fails'].passed
    for name in ('busy_fails', 'random_fails'):
        assert checks[name].status == 'fail'
        assert checks[name].details.endswith('earned reward without doing the task')


def run_qa_on(env: MultiTurnEnvironment, dg: DeepGym) -> dict:
    """Run the multi-turn checks on an env built in memory (run_qa loads by name or folder)."""
    from deepgym.qa.gate import _multi_turn_checks

    return {check.name: check for check in _multi_turn_checks(env, dg, QAOptions(reruns=1))}


def test_scripted_agents_use_every_allowed_action_harmlessly() -> None:
    env = load_environment('signup_form')
    shell = MultiTurnEnvironment(task='t', final_verifier_code='print(1)')

    busy = do_everything(env)
    assert len(busy) == env.max_steps
    assert {action.action_type for action in busy} == env.allowed_actions - {'done'}
    assert random_agent(env) == random_agent(env)  # seeded: the same episode every run
    assert all(0 <= (a.x or 0) < 1024 and 0 <= (a.y or 0) < 768 for a in random_agent(env))
    actions = do_everything(shell) + random_agent(shell)
    commands = [a.content for a in actions if a.action_type == 'bash']
    assert commands and set(commands) <= set(_COMMANDS)


def test_computer_use_envs_carry_their_reference_episode_but_never_hand_it_out() -> None:
    for name in NAMES:
        last = len(variants(name)) - 1
        env = load_environment(name, variant=last)
        assert env.reference_actions == reference(name, last)
        assert 'reference_actions' not in env.model_dump()  # never serialized
        assert 'reference_actions.json' not in env.files  # never staged for the agent


def test_computer_use_qa_needs_a_desktop(local_dg: DeepGym) -> None:
    report = run_qa('signup_form', local_dg)

    assert report.status == 'skip'
    assert report.checks[0].details.startswith('computer-use env needs a desktop')


def test_computer_use_qa_replays_on_a_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stand-in desktop backend that judges the episode by whether it is the reference.
    dg = DeepGym(mode='local')
    expected = load_environment('todo_list').reference_actions
    monkeypatch.setattr(DeepGym, 'backend', property(lambda self: 'daytona'))

    def run_episode(env, agent):
        actions = []
        while (action := agent(None)).action_type != 'done' and len(actions) < env.max_steps:
            actions.append(action)
        score = 1.0 if actions + [action] == expected else 0.0
        result = RunResult(
            score=score,
            passed=score == 1.0,
            output='',
            stderr='',
            exit_code=0 if score else 1,
            execution_time_ms=1.0,
            sandbox_id='fake-desktop',
        )
        return Trajectory(steps=[]), result

    monkeypatch.setattr(dg, 'run_episode', run_episode)

    report = run_qa('todo_list', dg, QAOptions(reruns=1))

    assert report.status == 'pass', report.checks
    assert set(report.ran) >= {'reference_passes', 'busy_fails', 'random_fails'}
