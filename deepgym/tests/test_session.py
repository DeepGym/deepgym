"""Tests for deepgym.session -- live multi-turn episodes on the local backend."""

import asyncio
import os
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym import AsyncDeepGym, ComputerUseEnvironment, DeepGym
from deepgym.exceptions import DeepGymError, SandboxError, VerifierError
from deepgym.models import Action, MultiTurnEnvironment, Observation, Trajectory
from deepgym.session import _signals_done

PASS_VERIFIER = (
    'import json\n'
    'print(json.dumps({"schema_version": "1.0", "score": 1.0, "passed": True, "details": "ok"}))\n'
)

FILE_VERIFIER = (
    'import json, pathlib\n'
    'ok = pathlib.Path("notes/out.txt").read_text() == "hello"\n'
    'print(json.dumps({"score": float(ok), "passed": ok}))\n'
)


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    """Local DeepGym client."""
    return DeepGym(mode='local')


def _scripted(*actions: Action):
    """Return an agent that plays *actions* in order."""
    queue = list(actions)
    return lambda obs: queue.pop(0)


class TestActionValidation:
    """Actions must carry the arguments their type needs."""

    def test_default_is_code(self) -> None:
        assert Action(content='print(1)').action_type == 'code'

    def test_click_needs_coordinates(self) -> None:
        with pytest.raises(ValidationError, match='needs x and y'):
            Action(action_type='click', x=10)

    def test_drag_needs_end(self) -> None:
        with pytest.raises(ValidationError, match='to_x and to_y'):
            Action(action_type='drag', x=1, y=2)

    def test_scroll_needs_direction(self) -> None:
        with pytest.raises(ValidationError, match='scroll content'):
            Action(action_type='scroll', x=1, y=2, content='sideways')

    def test_key_needs_content(self) -> None:
        with pytest.raises(ValidationError, match='key needs content'):
            Action(action_type='key')

    @pytest.mark.parametrize('path', [None, '/etc/passwd', '../escape.txt', 'a/../../b', '.', ''])
    def test_write_file_rejects_unsafe_paths(self, path: str | None) -> None:
        with pytest.raises(ValidationError, match='relative path'):
            Action(action_type='write_file', path=path, content='x')

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Action(action_type='teleport')


class TestEnvironmentModels:
    """Episode environment defaults and invariants."""

    def test_multi_turn_defaults(self) -> None:
        env = MultiTurnEnvironment(task='t')
        assert env.max_steps == 10
        assert env.desktop is False
        assert env.allowed_actions == {'code', 'bash', 'write_file', 'done'}

    def test_gui_actions_need_desktop(self) -> None:
        with pytest.raises(ValidationError, match='desktop=True'):
            MultiTurnEnvironment(task='t', allowed_actions={'click', 'done'})

    def test_computer_use_is_gui_only(self) -> None:
        env = ComputerUseEnvironment(task='t')
        assert env.type == 'computer-use'
        assert env.desktop is True
        assert 'bash' not in env.allowed_actions
        assert {'click', 'type', 'key', 'done'} <= env.allowed_actions

    def test_min_steps_validation(self) -> None:
        with pytest.raises(ValidationError):
            MultiTurnEnvironment(task='t', max_steps=0)


class TestSignalsDone:
    """Output-based done detection."""

    @pytest.mark.parametrize(
        ('output', 'expected'),
        [
            ('some output\nDONE\n', True),
            ('{"done": true}', True),
            ('{"done": false}', False),
            ('regular output', False),
            ('', False),
            ('{not valid json}', False),
        ],
    )
    def test_signals_done(self, output: str, expected: bool) -> None:
        assert _signals_done(output) is expected


class TestLocalEpisodes:
    """End-to-end episodes on the local backend."""

    def test_done_marker_ends_episode(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=5, final_verifier_code=PASS_VERIFIER)
        agent = _scripted(Action(content='print("step")'), Action(content='print("DONE")'))

        trajectory, result = dg.run_episode(env, agent)

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory.steps) == 2
        assert trajectory.final_observation is not None
        assert trajectory.final_observation.done is True
        assert result.passed is True

    def test_done_action_ends_episode(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=5, final_verifier_code=PASS_VERIFIER)
        trajectory, _ = dg.run_episode(env, _scripted(Action(action_type='done')))
        assert len(trajectory.steps) == 1

    def test_max_steps_ends_episode(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=2, final_verifier_code=PASS_VERIFIER)
        calls: list[Observation] = []

        def agent(obs: Observation) -> Action:
            calls.append(obs)
            return Action(content='print("working")')

        trajectory, _ = dg.run_episode(env, agent)
        assert len(calls) == 2
        assert len(trajectory.steps) == 2

    def test_no_final_verifier_scores_zero(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=1)
        _, result = dg.run_episode(env, _scripted(Action(action_type='done')))
        assert result.score == 0.0
        assert result.passed is False

    def test_verifier_sees_state_left_by_actions(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=3, final_verifier_code=FILE_VERIFIER)
        agent = _scripted(
            Action(action_type='write_file', path='notes/out.txt', content='hel'),
            Action(action_type='bash', content='printf hello > notes/out.txt'),
            Action(action_type='done'),
        )
        _, result = dg.run_episode(env, agent)
        assert result.passed is True

    def test_files_and_setup_run_before_first_observation(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(
            task='t',
            files={'data/in.txt': 'seed'},
            setup_script='cat data/in.txt',
            final_verifier_code=PASS_VERIFIER,
        )
        with dg.session(env) as session:
            obs = session.start()
        assert obs.content.strip() == 'seed'
        assert obs.step == 0

    def test_planted_module_cannot_hijack_verifier(self, dg: DeepGym) -> None:
        fake = 'print(\'{"score": 1.0, "passed": true}\')\nraise SystemExit(0)\n'
        honest = 'import json\nprint(json.dumps({"score": 0.0, "passed": False}))\n'
        env = MultiTurnEnvironment(task='t', max_steps=2, final_verifier_code=honest)
        agent = _scripted(
            Action(action_type='write_file', path='json.py', content=fake),
            Action(action_type='done'),
        )
        _, result = dg.run_episode(env, agent)
        assert result.passed is False

    def test_step_verifier_rewards(self, dg: DeepGym) -> None:
        step_verifier = (
            'import os, pathlib\n'
            'out = pathlib.Path(os.environ["STEP_OUTPUT_PATH"]).read_text()\n'
            'print(0.5 if "good" in out else 0.0)\n'
        )
        env = MultiTurnEnvironment(
            task='t',
            max_steps=2,
            step_verifier_code=step_verifier,
            final_verifier_code=PASS_VERIFIER,
        )
        agent = _scripted(Action(content='print("good")'), Action(content='print("bad")'))
        trajectory, _ = dg.run_episode(env, agent)
        assert trajectory.step_rewards == [0.5, 0.0]
        assert trajectory.total_reward == 0.5

    def test_failing_setup_raises(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', setup_script='echo broken >&2; exit 3')
        with dg.session(env) as session:
            with pytest.raises(SandboxError, match='exited 3'):
                session.start()

    def test_command_exit_code_in_observation(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=3)
        with dg.session(env) as session:
            session.start()
            failed = session.step(Action(action_type='bash', content='false')).observation
            wrote = session.step(Action(action_type='write_file', path='a.txt', content='x'))
        assert failed.metadata == {'exit_code': 1}
        assert wrote.observation.metadata is None

    def test_broken_step_verifier_raises(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=2, step_verifier_code='print("nope")')
        with dg.session(env) as session:
            session.start()
            with pytest.raises(VerifierError, match='unparseable'):
                session.step(Action(content='print(1)'))

    def test_step_verifier_stderr_is_ignored(self, dg: DeepGym) -> None:
        noisy = 'import sys\nprint("DeprecationWarning: x", file=sys.stderr)\nprint(0.5)\n'
        env = MultiTurnEnvironment(task='t', max_steps=1, step_verifier_code=noisy)
        trajectory, _ = dg.run_episode(env, lambda obs: Action(content='print(1)'))
        assert trajectory.step_rewards == [0.5]

    def test_failed_step_verifier_ends_episode_consistently(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=5, step_verifier_code='print("nope")')
        with dg.session(env) as session:
            session.start()
            with pytest.raises(VerifierError):
                session.step(Action(content='print(1)'))
            assert session.done
            assert len(session.trajectory.steps) == 1
            assert session.trajectory.step_rewards == []  # no verdict, so no 0 to train on

    @pytest.mark.parametrize(
        'step_verifier',
        [
            # Prints a failed verdict, then crashes: the guard exits 2.
            'print(\'{"score": 0.0, "passed": false}\', flush=True)\nraise RuntimeError("bug")\n',
            # Reports its own infrastructure failure with a normal "failed" exit.
            'import sys\nprint(\'{"score": 0.0, "passed": false, "error_type": "sandbox_error"}\')'
            '\nsys.exit(1)\n',
            'print(\'{"score": 0.5, "error_type": []}\')\n',
        ],
    )
    def test_step_verifier_without_a_verdict_ends_the_episode(
        self, dg: DeepGym, step_verifier: str
    ) -> None:
        env = MultiTurnEnvironment(
            task='t',
            max_steps=3,
            step_verifier_code=step_verifier,
            final_verifier_code=PASS_VERIFIER,
        )
        with dg.session(env) as session:
            session.start()
            with pytest.raises(VerifierError, match='no verdict'):
                session.step(Action(content='print(1)'))
            result = session.score()
        assert session.done and session.trajectory.step_rewards == []
        assert (result.has_verdict, result.error_type) == (False, 'verifier_error')

    def test_step_verifier_model_fault_is_a_reward(self, dg: DeepGym) -> None:
        model_fault = 'print(\'{"score": 0.0, "passed": false, "error_type": "invalid_patch"}\')\n'
        env = MultiTurnEnvironment(task='t', max_steps=1, step_verifier_code=model_fault)
        trajectory, _ = dg.run_episode(env, lambda obs: Action(content='print(1)'))
        assert trajectory.step_rewards == [0.0]

    def test_setup_has_its_own_timeout(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(
            task='t', setup_script='sleep 2; echo ready', timeout_per_step=1, setup_timeout=10
        )
        with dg.session(env) as session:
            assert session.start().content.strip() == 'ready'

    def test_concurrent_steps_are_serialized(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=20)
        with dg.session(env) as session:
            session.start()

            def worker() -> None:
                for _ in range(5):
                    session.step(Action(action_type='bash', content='sleep 0.02'))

            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            steps = [obs.step for obs, _ in session.trajectory.steps]
        assert steps == list(range(15))

    def test_background_processes_die_with_session(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=2)
        with dg.session(env) as session:
            session.start()
            obs = session.step(
                Action(action_type='bash', content='sleep 600 >/dev/null 2>&1 & echo $!')
            ).observation
        pid = int(obs.content.strip())
        time.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    def test_step_score_printed_before_a_crash_is_an_error(self, dg: DeepGym) -> None:
        crashes = 'print(1.0, flush=True)\nraise RuntimeError("verifier bug")\n'
        env = MultiTurnEnvironment(task='t', max_steps=2, step_verifier_code=crashes)
        with dg.session(env) as session:
            session.start()
            with pytest.raises(VerifierError):
                session.step(Action(content='print(1)'))
        assert session.trajectory.total_reward == 0.0

    def test_final_score_printed_before_a_crash_is_an_error(self, dg: DeepGym) -> None:
        crashes = PASS_VERIFIER + 'import sys\nsys.stdout.flush()\nraise RuntimeError("bug")\n'
        env = MultiTurnEnvironment(task='t', max_steps=1, final_verifier_code=crashes)
        with pytest.raises(VerifierError):
            dg.run_episode(env, _scripted(Action(action_type='done')))

    def test_protocol_fail_exit_keeps_partial_credit(self, dg: DeepGym) -> None:
        failed = 'import sys\nprint(\'{"score": 0.4, "passed": false}\')\nsys.exit(1)\n'
        env = MultiTurnEnvironment(task='t', max_steps=1, final_verifier_code=failed)
        _, result = dg.run_episode(env, _scripted(Action(action_type='done')))
        assert result.score == 0.4

    def test_step_timeout_becomes_observation(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=1, timeout_per_step=1)
        with dg.session(env) as session:
            session.start()
            obs, _, done = session.step(Action(action_type='bash', content='sleep 5'))
        assert 'timed out' in obs.content
        assert done is True


DEBUG_FIX = Path(__file__).resolve().parent.parent / 'src/deepgym/envs/multi_turn/debug_fix'
ALWAYS_EQUAL = (
    'class AlwaysEqual:\n'
    '    def __eq__(self, other):\n'
    '        return True\n\n'
    'def sum_evens(numbers):\n'
    '    return AlwaysEqual()\n'
)
FIXED = 'def sum_evens(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n'


def debug_fix_score(dg: DeepGym, buggy: str) -> float:
    """Score a debug_fix episode whose agent writes *buggy* as buggy.py."""
    env = MultiTurnEnvironment(
        task=(DEBUG_FIX / 'task.md').read_text(),
        files={'buggy.py': (DEBUG_FIX / 'buggy.py').read_text()},
        final_verifier_code=(DEBUG_FIX / 'verifier.py').read_text(),
        max_steps=2,
    )
    agent = _scripted(
        Action(action_type='write_file', path='buggy.py', content=buggy),
        Action(action_type='done'),
    )
    return dg.run_episode(env, agent)[1].score


class TestDebugFixEnvironment:
    """The debug_fix verifier runs the agent's code outside the judge."""

    def test_fixed_code_passes(self, dg: DeepGym) -> None:
        assert debug_fix_score(dg, FIXED) == 1.0

    def test_always_equal_object_fails(self, dg: DeepGym) -> None:
        assert debug_fix_score(dg, ALWAYS_EQUAL) < 1.0

    def test_verifier_never_imports_the_agents_code(self) -> None:
        source = (DEBUG_FIX / 'verifier.py').read_text()
        assert 'load_solution' in source
        assert 'import buggy' not in source and 'from buggy' not in source


class TestSessionGuards:
    """Misuse fails loudly."""

    def test_disallowed_action_raises(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(task='t', allowed_actions={'code', 'done'})
        with dg.session(env) as session:
            session.start()
            with pytest.raises(DeepGymError, match='not allowed'):
                session.step(Action(action_type='bash', content='ls'))

    def test_step_before_start_raises(self, dg: DeepGym) -> None:
        with dg.session(MultiTurnEnvironment(task='t')) as session:
            with pytest.raises(DeepGymError, match='not running'):
                session.step(Action(action_type='done'))

    def test_double_start_raises(self, dg: DeepGym) -> None:
        with dg.session(MultiTurnEnvironment(task='t')) as session:
            session.start()
            with pytest.raises(DeepGymError, match='already started'):
                session.start()

    def test_desktop_needs_daytona(self, dg: DeepGym) -> None:
        with dg.session(ComputerUseEnvironment(task='t')) as session:
            with pytest.raises(DeepGymError, match='Daytona'):
                session.start()

    def test_run_rejects_multi_turn_env(self, dg: DeepGym) -> None:
        with pytest.raises(DeepGymError, match='run_episode'):
            dg.run(MultiTurnEnvironment(task='t'), model_output='')

    def test_async_run_rejects_multi_turn_env(self) -> None:
        with pytest.raises(DeepGymError, match='run_episode'):
            asyncio.run(AsyncDeepGym(mode='local').run(MultiTurnEnvironment(task='t'), ''))

    def test_eval_rejects_agent_suites(self, dg: DeepGym) -> None:
        with pytest.raises(DeepGymError, match='needs an agent'):
            dg.eval('computer-use', {})
