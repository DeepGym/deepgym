"""Tests for deepgym.session -- live multi-turn episodes on the local backend."""

import asyncio
import errno
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym import AsyncDeepGym, ComputerUseEnvironment, DeepGym
from deepgym import session as session_module
from deepgym.cli_runtime import CliRequest
from deepgym.exceptions import (
    ActionError,
    DeepGymError,
    SandboxError,
    SandboxExpiredError,
    VerifierError,
)
from deepgym.models import Action, MultiTurnEnvironment, Observation, Trajectory
from deepgym.sandboxes import SandboxDiedError, SandboxTimeoutError
from deepgym.session import Session, _signals_done, failure_result, run_episode

PASS_VERIFIER = (
    'import json\n'
    'print(json.dumps({"schema_version": "1.0", "score": 1.0, "passed": True, "details": "ok"}))\n'
)

FILE_VERIFIER = (
    'import json, pathlib\n'
    'ok = pathlib.Path("notes/out.txt").read_text() == "hello"\n'
    'print(json.dumps({"score": float(ok), "passed": ok}))\n'
)

CLI_REQUEST = CliRequest(
    argv=('claude', '-p', 'task'),
    env={'ANTHROPIC_API_KEY': 'test-key'},
    provider='anthropic',
    timeout=5,
    allow_model_credentials=True,
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

    def test_final_verdict_comes_from_stdout_and_stderr_is_kept(self, dg: DeepGym) -> None:
        verifier = (
            'import json, sys\n'
            'print(json.dumps({"score": 0.0, "passed": False}))\n'
            'print(json.dumps({"score": 1.0, "passed": True}), file=sys.stderr)\n'
        )
        env = MultiTurnEnvironment(task='t', max_steps=1, final_verifier_code=verifier)

        _, result = dg.run_episode(env, _scripted(Action(action_type='done')))

        assert (result.score, result.passed) == (0.0, False)
        assert result.stderr == '{"score": 1.0, "passed": true}'

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

    def test_step_verifier_outage_ends_the_episode_unjudged(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = MultiTurnEnvironment(
            task='t',
            max_steps=1,
            step_verifier_code='print(1.0)',
            final_verifier_code=PASS_VERIFIER,
        )
        first = Action(action_type='bash', content='echo one')
        with dg.session(env) as session:
            session.start()

            def outage(*args: object, **kwargs: object) -> None:
                raise SandboxError('provider unreachable') from ConnectionError('reset')

            monkeypatch.setattr(session._backend, 'verify', outage)
            with pytest.raises(SandboxError, match='unreachable'):
                session.step(first)
            # A retry would apply a second action past the budget and score it.
            with pytest.raises(DeepGymError, match='not running'):
                session.step(Action(action_type='bash', content='echo two'))
            result = session.score()

        trajectory = session.trajectory
        assert session.done and [action for _, action in trajectory.steps] == [first]
        assert (trajectory.step_rewards, trajectory.total_reward) == ([], 0.0)
        assert trajectory.final_observation.content.strip() == 'one'
        assert (result.has_verdict, result.error_type) == (False, 'sandbox_error')

    def test_a_write_the_agents_own_files_block_is_rejected_and_the_episode_goes_on(
        self, dg: DeepGym
    ) -> None:
        with dg.session(MultiTurnEnvironment(task='t', max_steps=3)) as session:
            session.start()
            session.step(Action(action_type='bash', content='mkdir taken'))
            with pytest.raises(ActionError, match='Failed to write taken'):
                session.step(Action(action_type='write_file', path='taken', content='x'))
            session.step(Action(action_type='bash', content='echo still running'))
        assert [action.action_type for _, action in session.trajectory.steps] == ['bash', 'bash']

    @pytest.mark.parametrize(
        'blocker',
        [
            'mkdir _step_output.txt',
            'ln -s missing_dir/output _step_output.txt',
            'mkfifo _step_output.txt',
        ],
        ids=['folder', 'dangling_link', 'fifo'],
    )
    def test_blocking_the_step_output_ends_the_episode_with_a_judged_zero(
        self, dg: DeepGym, blocker: str
    ) -> None:
        env = MultiTurnEnvironment(
            task='t',
            max_steps=3,
            step_verifier_code='print(1.0)',
            final_verifier_code=PASS_VERIFIER,
        )
        block = Action(action_type='bash', content=blocker)
        with dg.session(env) as session:
            session.start()
            with pytest.raises(ActionError):
                session.step(block)
            result = session.score()
        assert session.done and [action for _, action in session.trajectory.steps] == [block]
        assert session.trajectory.step_rewards == []
        assert (result.score, result.has_verdict) == (0.0, True)
        assert result.error_type == 'policy_sandbox_failure'

    @pytest.mark.parametrize('when', ['step', 'final_verifier'])
    def test_a_sandbox_that_dies_once_the_agent_acted_is_a_judged_zero(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch, when: str
    ) -> None:
        env = MultiTurnEnvironment(task='t', max_steps=3, final_verifier_code=PASS_VERIFIER)

        def dies(*args: object, **kwargs: object) -> None:
            raise SandboxDiedError('the sandbox ran out of memory')

        with dg.session(env) as session:
            session.start()
            if when == 'step':
                monkeypatch.setattr(session._backend, 'exec', dies)
                with pytest.raises(SandboxDiedError):
                    session.step(Action(action_type='bash', content='echo hog'))
            else:
                session.step(Action(action_type='done'))
                monkeypatch.setattr(session._backend, 'verify', dies)
            result = session.score()

        assert session.done and len(session.trajectory.steps) == 1
        assert (result.score, result.has_verdict) == (0.0, True)
        assert result.error_type == 'policy_sandbox_failure'

    def test_malformed_agent_output_is_the_agents_zero(self, dg: DeepGym) -> None:
        parse = 'import json\njson.loads(open("answer.json").read())\nprint(1.0)\n'
        env = MultiTurnEnvironment(
            task='t', max_steps=2, step_verifier_code=parse, final_verifier_code=parse
        )
        with dg.session(env) as session:
            session.start()
            step = session.step(Action(action_type='bash', content='echo oops > answer.json'))
            result = session.score()

        assert (step.reward, session.step_verdict['error_type']) == (0.0, 'solution_error')
        assert (result.has_verdict, result.score, result.error_type) == (
            True,
            0.0,
            'solution_error',
        )

    def test_run_episode_returns_the_result_a_failed_step_fixed(self, dg: DeepGym) -> None:
        env = MultiTurnEnvironment(
            task='t',
            max_steps=3,
            step_verifier_code='print(1.0)',
            final_verifier_code=PASS_VERIFIER,
        )
        actions = iter([Action(action_type='bash', content='mkdir _step_output.txt')])

        trajectory, result = dg.run_episode(env, lambda obs: next(actions))

        assert len(trajectory.steps) == 1  # the agent blocked the harness: its own 0
        assert (result.has_verdict, result.error_type) == (True, 'policy_sandbox_failure')

    @pytest.mark.parametrize('code', [errno.EIO, errno.ENOSPC, errno.EDQUOT])
    def test_a_step_output_write_that_fails_is_the_agents_zero(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        """The agent can fill the disk, or plant a link the write follows, and free space
        again before anything checks: no write failure after it acted is ours for certain."""
        env = MultiTurnEnvironment(
            task='t',
            max_steps=3,
            step_verifier_code='print(1.0)',
            final_verifier_code=PASS_VERIFIER,
        )
        real_open = os.open

        def disk_error(path: object, *args: object, **kwargs: object) -> int:
            if str(path).endswith('_step_output.txt'):
                raise OSError(code, os.strerror(code))
            return real_open(path, *args, **kwargs)

        with dg.session(env) as session:
            session.start()
            monkeypatch.setattr(os, 'open', disk_error)
            with pytest.raises(SandboxError, match=os.strerror(code)):
                session.step(Action(action_type='bash', content='echo harmless'))
            monkeypatch.undo()
            result = session.score()
        assert session.done and len(session.trajectory.steps) == 1
        assert (result.has_verdict, result.error_type) == (True, 'policy_sandbox_failure')

    def test_a_verifier_the_agents_output_hangs_is_its_zero(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A verifier stuck reading what the agent made (a FIFO) timed out on the agent."""
        monkeypatch.setattr('deepgym.session.DEADLINE_GRACE_S', 0)
        reads = 'print(float(open("answer.txt").read() == "42\\n"))\n'
        env = MultiTurnEnvironment(
            task='t',
            max_steps=3,
            timeout_per_step=1,
            step_verifier_code=reads,
            final_verifier_code=reads,
        )
        with dg.session(env) as session:
            session.start()
            answered = session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            hung = session.step(
                Action(action_type='bash', content='rm answer.txt; mkfifo answer.txt')
            )
            verdict = session.step_verdict
            result = session.score()

        assert answered.reward == 1.0  # the plain-float form still parses
        assert (hung.reward, hung.done, verdict['error_type']) == (0.0, False, 'solution_timeout')
        assert (result.score, result.has_verdict, result.error_type) == (
            0.0,
            True,
            'solution_timeout',
        )

    def test_a_verifier_that_hangs_before_reading_the_agents_output_has_no_verdict(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('deepgym.session.DEADLINE_GRACE_S', 0)
        hangs = 'import time\ntime.sleep(30)\n'
        step_env = MultiTurnEnvironment(
            task='t', max_steps=3, timeout_per_step=1, step_verifier_code=hangs
        )
        final_env = MultiTurnEnvironment(
            task='t', max_steps=3, timeout_per_step=1, final_verifier_code=hangs
        )
        with dg.session(step_env) as session:
            session.start()
            with pytest.raises(VerifierError, match='timed out'):
                session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            result = session.score()
        with dg.session(final_env) as session:
            session.start()
            session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            with pytest.raises(VerifierError, match='before it read'):
                session.score()

        assert (result.has_verdict, result.error_type) == (False, 'verifier_error')

    @pytest.mark.parametrize(
        'consults',
        [
            'import os\nos.listdir(".")\n',
            'import glob\nglob.glob("**/*.txt", recursive=True)\n',
            'import subprocess\nsubprocess.run(["true"])\n',
            'import os, subprocess\n'
            'subprocess.run(["cat", os.path.abspath("answer.txt")], cwd="/", stdout=-3)\n',
            'import contextlib, io\nwith contextlib.redirect_stdout(io.StringIO()):\n'
            '    open("answer.txt").close()\n',
        ],
        ids=[
            'lists_its_folder',
            'globs_its_folder',
            'runs_a_program_there',
            'runs_a_program_on_its_file',
            'captures_stdout',
        ],
    )
    def test_a_verifier_can_not_be_kept_from_marking_what_it_consulted(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch, consults: str
    ) -> None:
        """Listing the agent's folder or running a program in it consults the agent, and
        capturing stdout can't hide the mark: a hang after that is the agent's 0."""
        monkeypatch.setattr('deepgym.session.DEADLINE_GRACE_S', 0)
        hangs = consults + 'import time\ntime.sleep(30)\n'
        env = MultiTurnEnvironment(
            task='t', max_steps=2, timeout_per_step=1, final_verifier_code=hangs
        )
        with dg.session(env) as session:
            session.start()
            session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            result = session.score()

        assert (result.score, result.has_verdict, result.error_type) == (
            0.0,
            True,
            'solution_timeout',
        )

    @pytest.mark.parametrize('stalls', [True, False], ids=['agent_stalls', 'host_pauses'])
    def test_a_provider_lifetime_the_agent_ran_out_is_its_zero(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch, stalls: bool
    ) -> None:
        """An expiry is ours only when the agent's own actions didn't take most of the time."""
        env = MultiTurnEnvironment(task='t', max_steps=2, final_verifier_code=PASS_VERIFIER)

        def expired(*args: object, **kwargs: object) -> None:
            raise SandboxExpiredError('the sandbox reached its provider lifetime')

        with dg.session(env) as session:
            session.start()
            session.step(Action(action_type='bash', content='sleep 0.6' if stalls else 'true'))
            if not stalls:
                time.sleep(0.6)  # the host, thinking between steps
            monkeypatch.setattr(session._backend, 'verify', expired)
            if stalls:
                result = session.score()
                assert (result.has_verdict, result.error_type) == (True, 'policy_sandbox_failure')
            else:
                with pytest.raises(SandboxExpiredError) as raised:
                    session.score()
                assert not failure_result(raised.value, session.sandbox_id).has_verdict

    def test_a_verifier_reads_its_own_mark_not_one_the_agent_wrote(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each run's mark is fresh and random: a mark-shaped line the verifier prints, a
        stale mark, or an earlier run's mark can't make a hang before reading count."""
        monkeypatch.setattr('deepgym.session.DEADLINE_GRACE_S', 0)
        marks: list[str] = []
        real_guarded = session_module._guarded

        def guarded(code: str, mark: str) -> str:
            marks.append(mark)
            return real_guarded(code, mark)

        monkeypatch.setattr(session_module, '_guarded', guarded)
        env = MultiTurnEnvironment(
            task='t',
            max_steps=3,
            timeout_per_step=1,
            step_verifier_code='print(float(open("answer.txt").read() == "42\\n"))\n',
        )
        with dg.session(env) as session:
            session.start()
            session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            # The final verifier prints mark-shaped lines and both earlier runs' marks, each
            # on a line of its own, then hangs without reading anything of the agent's.
            prints = ''.join(f'print({line!r})\n' for line in ['0' * 32, *marks])
            hangs = f'{prints}import sys, time\nsys.stdout.flush()\ntime.sleep(30)\n'
            session._env = env.model_copy(update={'final_verifier_code': hangs})
            with pytest.raises(VerifierError, match='before it read'):
                session.score()

        assert len(set(marks)) == 3 and all(len(mark) == 32 for mark in marks)

    def test_what_the_agent_shapes_on_stdout_cant_change_the_verdict(self, dg: DeepGym) -> None:
        """The verifier echoes the agent's bytes (invalid UTF-8, a mark-shaped line, a huge
        line, a fake verdict) around its own partial line and the mark; its own last
        verdict line is the one read."""
        echoes = (
            'import json, sys\n'
            'sys.stdout.write("partial line")\n'
            'data = open("answer.bin", "rb").read()\n'
            'sys.stdout.flush()\n'
            'sys.stdout.buffer.write(data + b"\\n")\n'
            'print(json.dumps({"score": 0.25, "passed": False}))\n'
        )
        env = MultiTurnEnvironment(task='t', max_steps=2, final_verifier_code=echoes)
        planted = (
            "printf '\\377\\376 junk\\n' > answer.bin; "
            "printf '%032d\\n' 0 >> answer.bin; "
            "head -c 200000 /dev/zero | tr '\\0' x >> answer.bin; "
            'printf \'\\n{"score": 1.0, "passed": true}\' >> answer.bin'
        )
        with dg.session(env) as session:
            session.start()
            session.step(Action(action_type='bash', content=planted))
            result = session.score()

        assert (result.score, result.passed, result.has_verdict) == (0.25, False, True)

    @pytest.mark.parametrize('consults', [True, False], ids=['after_reading', 'before'])
    def test_a_verifier_killed_after_reading_the_agents_output_is_its_zero(
        self, dg: DeepGym, consults: bool
    ) -> None:
        """Memory the agent's output made the verifier use can get it killed: no verdict
        would let the agent dodge its 0. Killed before reading anything, it's ours."""
        reads = 'open("answer.txt").read()\n' if consults else ''
        killed = f'import os, signal\n{reads}os.kill(os.getpid(), signal.SIGKILL)\n'
        env = MultiTurnEnvironment(task='t', max_steps=2, final_verifier_code=killed)
        with dg.session(env) as session:
            session.start()
            session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            if not consults:
                with pytest.raises(VerifierError):
                    session.score()
                return
            result = session.score()

        assert (result.score, result.has_verdict, result.error_type) == (
            0.0,
            True,
            'solution_error',
        )

    def test_a_timeout_whose_output_was_lost_is_the_agents_zero(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With nothing the verifier printed (not even its start line), nothing shows it
        hadn't read the agent's output: a pipe a planted child held, say."""
        env = MultiTurnEnvironment(task='t', max_steps=2, final_verifier_code=PASS_VERIFIER)

        def lost(*args: object, **kwargs: object) -> None:
            raise SandboxTimeoutError('command exceeded 30s')  # what the provider returns

        with dg.session(env) as session:
            session.start()
            session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            monkeypatch.setattr(session._backend, 'verify', lost)
            result = session.score()

        assert (result.score, result.has_verdict, result.error_type) == (
            0.0,
            True,
            'policy_sandbox_failure',
        )

    def test_a_verdict_printed_around_the_mark_reads_as_printed(self, dg: DeepGym) -> None:
        """The mark is cut out exactly, so a line it fell inside is whole again."""
        split = (
            'import sys\n'
            'sys.stdout.write(\'{"score":\')\n'
            'open("answer.txt").close()\n'
            'print(\' 0.5, "passed": false}\')\n'
        )
        env = MultiTurnEnvironment(task='t', max_steps=2, final_verifier_code=split)
        with dg.session(env) as session:
            session.start()
            session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            result = session.score()

        assert (result.score, result.passed, result.has_verdict) == (0.5, False, True)

    @pytest.mark.parametrize(
        'fails',
        [
            'import sys\nsys.exit(2)\n',
            'print(\'{"score": 5}\')\n',
            'print("not a verdict")\n',
            'print(\'{"score": 0, "passed": false, "error_type": "runtime_error"}\')\n'
            'import sys\nsys.exit(2)\n',
        ],
        ids=['exits_silently', 'invalid_score', 'no_verdict_line', 'reports_its_own_error'],
    )
    def test_a_verifier_that_fails_after_reading_the_agents_output_is_its_zero(
        self, dg: DeepGym, fails: str
    ) -> None:
        """Whatever the agent's output made the verifier do, no verdict would let it dodge."""
        reads = 'open("answer.txt").read()\n'
        env = MultiTurnEnvironment(
            task='t',
            max_steps=3,
            step_verifier_code=reads + fails,
            final_verifier_code=reads + fails,
        )
        with dg.session(env) as session:
            session.start()
            step = session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            verdict = session.step_verdict
            result = session.score()

        assert (step.reward, verdict['error_type']) == (0.0, 'solution_error')
        assert (result.score, result.has_verdict, result.error_type) == (
            0.0,
            True,
            'solution_error',
        )

    def test_the_agents_code_failing_to_launch_has_no_verdict_even_after_reading(
        self, dg: DeepGym
    ) -> None:
        """The crash guard's launch failure is the machine's, whatever the verifier read."""
        launch = (
            'open("answer.txt").read()\n'
            'print(\'{"score": 0, "passed": false, "error_type": "sandbox_error"}\')\n'
            'import sys\nsys.exit(2)\n'
        )
        env = MultiTurnEnvironment(task='t', max_steps=2, final_verifier_code=launch)
        with dg.session(env) as session:
            session.start()
            session.step(Action(action_type='bash', content='echo 42 > answer.txt'))
            result = session.score()

        assert (result.has_verdict, result.error_type) == (False, 'sandbox_error')

    def test_a_failure_the_rule_cant_read_is_the_agents_zero(self) -> None:
        """An exception that breaks the classifier is never a reason to mask."""

        class UnreadableError(Exception):
            @property
            def status_code(self) -> int:
                raise RuntimeError('broken SDK error')

            def __str__(self) -> str:
                raise RuntimeError('no message either')

        result = failure_result(UnreadableError(), 'sb-1')

        assert (result.score, result.has_verdict) == (0.0, True)
        assert result.error_type == 'policy_sandbox_failure'

    def test_output_that_isnt_utf8_is_shown_not_a_failure(self, dg: DeepGym) -> None:
        with dg.session(MultiTurnEnvironment(task='t', max_steps=2)) as session:
            session.start()
            obs = session.step(Action(action_type='bash', content="printf '\\377ok'")).observation
        assert obs.content == '\ufffdok' and not session.done

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

    def test_final_score_printed_before_a_crash_is_not_kept(self, dg: DeepGym) -> None:
        crashes = PASS_VERIFIER + 'import sys\nsys.stdout.flush()\nraise RuntimeError("bug")\n'
        env = MultiTurnEnvironment(task='t', max_steps=1, final_verifier_code=crashes)
        _, result = dg.run_episode(env, _scripted(Action(action_type='done')))
        assert (result.score, result.has_verdict) == (0.0, False)
        assert result.error_type == 'runtime_error'

    def test_final_verifier_crash_after_loading_the_agents_code_is_a_judged_zero(
        self, dg: DeepGym
    ) -> None:
        crashes = (
            'from deepgym_solution import load_solution\n'
            'load_solution("agent.py")\n' + PASS_VERIFIER + 'raise RuntimeError("bug")\n'
        )
        env = MultiTurnEnvironment(task='t', max_steps=2, final_verifier_code=crashes)
        agent = _scripted(
            Action(action_type='write_file', path='agent.py', content='x = 1\n'),
            Action(action_type='done'),
        )
        _, result = dg.run_episode(env, agent)
        assert (result.score, result.passed, result.has_verdict) == (0.0, False, True)

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

    @pytest.mark.parametrize(
        'operation',
        [
            lambda session: session.start(),
            lambda session: session.step(Action(action_type='bash', content='echo hi')),
            lambda session: session.run_cli(CLI_REQUEST, lambda event: None),
            lambda session: session.finalize_cli(),
            lambda session: session.score(),
        ],
        ids=['start', 'step', 'run_cli', 'finalize_cli', 'score'],
    )
    def test_a_closed_session_refuses_to_touch_its_sandbox(
        self, dg: DeepGym, operation: Callable[[Session], object]
    ) -> None:
        """A5: a call queued behind close() would hit a sandbox that may be deleted and be
        scored as the agent's failure; it is refused instead, and nothing is recorded."""
        session = dg.session(MultiTurnEnvironment(task='t', final_verifier_code=PASS_VERIFIER))
        session.start()
        session.close()

        with pytest.raises(DeepGymError, match='closed'):
            operation(session)

        assert not session.trajectory.steps and session.failed_result is None

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


def _fail_delete(monkeypatch: pytest.MonkeyPatch, session: Session) -> None:
    """Make *session*'s backend close the way a provider that reports a failed delete does."""
    real_close = session._backend.close

    def close() -> None:
        real_close()
        raise SandboxError(f'Could not delete sandbox {session.sandbox_id}')

    monkeypatch.setattr(session._backend, 'close', close)


class TestCloseOnExit:
    """Leaving a session's with-block never trades its outcome for a failed sandbox delete."""

    def test_run_episode_keeps_its_result_when_the_delete_fails(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = dg.session(MultiTurnEnvironment(task='t', final_verifier_code=PASS_VERIFIER))
        _fail_delete(monkeypatch, session)

        trajectory, result = run_episode(session, _scripted(Action(action_type='done')))

        assert result.passed and len(trajectory.steps) == 1
        assert f'Could not delete sandbox {session.sandbox_id}' in caplog.text

    def test_a_body_error_still_propagates_when_the_delete_fails(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = dg.session(MultiTurnEnvironment(task='t'))
        _fail_delete(monkeypatch, session)

        with pytest.raises(RuntimeError, match='agent crashed'), session:
            raise RuntimeError('agent crashed')

    def test_an_explicit_close_still_raises(
        self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = dg.session(MultiTurnEnvironment(task='t'))
        _fail_delete(monkeypatch, session)

        with pytest.raises(SandboxError, match='Could not delete'):
            session.close()


def test_a_local_session_result_records_its_unsandboxed_actions(dg: DeepGym) -> None:
    """A local session runs the agent's actions as plain host processes: its results say
    so, fixed ones too, whatever its verifier isolates."""
    env = MultiTurnEnvironment(task='t', max_steps=1, final_verifier_code=PASS_VERIFIER)

    _, graded = dg.run_episode(env, lambda obs: Action(action_type='done'))
    with dg.session(env) as session:
        session.start()
        session.abandon('provider_error', 'lost')
        fixed = session.score()

    assert graded.isolation == fixed.isolation == 'none'
