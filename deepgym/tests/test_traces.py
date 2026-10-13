"""Tests for deepgym.traces: schema, provenance hashing, crash-safe capture, console storage."""

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym import ComputerUseEnvironment, DeepGym
from deepgym.console.episode_store import EpisodeStore, record_episode
from deepgym.exceptions import ActionError, DeepGymError, VerifierError
from deepgym.models import Action, MultiTurnEnvironment, Trajectory
from deepgym.traces import (
    AgentMeta,
    EnvFingerprint,
    Trace,
    TraceMessage,
    export_traces,
    load_trace,
)
from deepgym.traces.capture import UNFINISHED
from deepgym.traces.schema import environment_sha256
from tests.trace_samples import ENV, crashed_trace, graded_trace, unjudged_trace

SRC = Path(__file__).resolve().parent.parent / 'src'

# Step verifier: half credit once out.txt exists, with details for the trace.
STEP_VERIFIER = (
    'import json, os, pathlib\n'
    'made = pathlib.Path("out.txt").exists()\n'
    'print(json.dumps({"score": 0.5 if made else 0.0, "passed": made,'
    ' "details": "step " + os.environ["STEP_NUMBER"]}))\n'
)
FINAL_VERIFIER = (
    'import json, pathlib\n'
    'ok = pathlib.Path("out.txt").read_text().strip() == "hello"\n'
    'print(json.dumps({"score": float(ok), "passed": ok, "details": "checked out.txt"}))\n'
)
WRITE_ENV = MultiTurnEnvironment(
    name='write_out',
    task='Write hello to out.txt',
    step_verifier_code=STEP_VERIFIER,
    final_verifier_code=FINAL_VERIFIER,
    max_steps=4,
)
AGENT = AgentMeta(name='scripted', version='1', model='none', sampling={'temperature': 0.0})


@pytest.fixture(autouse=True)
def traces_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write traces into a temp directory."""
    directory = tmp_path / 'traces'
    monkeypatch.setenv('DEEPGYM_TRACES_DIR', str(directory))
    return directory


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    """Local DeepGym client."""
    return DeepGym(mode='local')


def _reply(content: str) -> list[TraceMessage]:
    return [TraceMessage(role='assistant', content=content)]


class TestSchema:
    """The trace schema round-trips and rejects inconsistent data."""

    @pytest.mark.parametrize('make', [graded_trace, crashed_trace, unjudged_trace])
    def test_json_round_trip(self, make) -> None:
        trace = make()

        restored = Trace.model_validate_json(trace.model_dump_json())

        assert restored == trace

    def test_verdict_and_status(self) -> None:
        assert (graded_trace().status, graded_trace().has_verdict) == ('complete', True)
        assert (crashed_trace().status, crashed_trace().has_verdict) == ('incomplete', False)
        assert (unjudged_trace().status, unjudged_trace().has_verdict) == ('complete', False)

    def test_dump_carries_flags_and_exit_codes(self) -> None:
        data = graded_trace().model_dump(mode='json')

        assert (data['status'], data['has_verdict']) == ('complete', True)
        assert [step['exit_code'] for step in data['steps']] == [0, None, None]

    def test_models_are_frozen(self) -> None:
        with pytest.raises(ValidationError):
            graded_trace().task = 'other'  # type: ignore[misc]

    def test_token_ids_and_logprobs_must_align(self) -> None:
        with pytest.raises(ValidationError, match='same length'):
            TraceMessage(role='assistant', content='x', token_ids=(1, 2), logprobs=(-0.1,))

    @pytest.mark.parametrize('trace_id', ['../escape', 'a/b', '', 'x' * 65])
    def test_trace_id_is_path_safe(self, trace_id: str) -> None:
        with pytest.raises(ValidationError):
            Trace.model_validate({**graded_trace().model_dump(), 'trace_id': trace_id})

    def test_unknown_schema_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Trace.model_validate({**graded_trace().model_dump(), 'schema_version': '2.0'})

    def test_naive_timestamps_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Trace.model_validate(
                {**graded_trace().model_dump(), 'finished_at': datetime(2026, 1, 1)}
            )

    def test_steps_must_be_in_order(self) -> None:
        trace = graded_trace()

        with pytest.raises(ValidationError, match='step indexes'):
            trace.model_validate({**trace.model_dump(), 'steps': trace.model_dump()['steps'][::-1]})

    def test_to_trajectory_skips_steps_without_observation(self) -> None:
        trace = graded_trace()

        trajectory = trace.to_trajectory()

        assert [action.action_type for _, action in trajectory.steps] == ['bash', 'done']
        assert trajectory.steps[0][0] == trace.initial_observation
        assert trajectory.final_observation == trace.steps[2].observation
        assert trajectory.step_rewards == [0.5, 0.5]


class TestProvenance:
    """Content hashes are stable and track what defines the environment."""

    def test_fingerprint_fields(self) -> None:
        fingerprint = EnvFingerprint.of(ENV, variant=2, seed=7, reference='ref')

        assert (fingerprint.name, fingerprint.variant, fingerprint.seed) == ('write_notes', 2, 7)
        assert fingerprint.image_digest == 'sha256:' + 'ab' * 32
        assert fingerprint.reference_sha256 is not None
        assert fingerprint.step_verifier_sha256 is not None

    def test_environment_hash_is_pinned(self) -> None:
        # Changing this hash invalidates every recorded trace's fingerprint: don't, lightly.
        assert environment_sha256(ENV) == (
            '7f142979bcdf5748ac3baa872a5db2a58d701da9bb29b7451d747e8a3777b6f9'
        )

    def test_hash_ignores_string_hash_seed(self) -> None:
        code = (
            'from deepgym import ComputerUseEnvironment\n'
            'from deepgym.traces.schema import environment_sha256\n'
            "files = {'b': '2', 'a': '1'}\n"
            "env = ComputerUseEnvironment(task='t', final_verifier_code='v', files=files)\n"
            'print(environment_sha256(env))\n'
        )
        hashes = {
            subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, 'PYTHONHASHSEED': seed, 'PYTHONPATH': str(SRC)},
            ).stdout.strip()
            for seed in ('0', '1', '2')
        }

        expected = ComputerUseEnvironment(
            task='t', final_verifier_code='v', files={'a': '1', 'b': '2'}
        )
        assert hashes == {environment_sha256(expected)}

    def test_hash_tracks_content_not_spelling(self) -> None:
        base = environment_sha256(ENV)

        assert environment_sha256(ENV.model_copy(update={'max_steps': 10})) == base
        assert environment_sha256(ENV.model_copy(update={'setup_script': 'true'})) != base
        assert environment_sha256(ENV.model_copy(update={'final_verifier_code': 'x'})) != base

    def test_subclass_defaults_change_the_hash(self) -> None:
        shell = MultiTurnEnvironment(task='t')
        desktop = ComputerUseEnvironment(task='t')

        assert environment_sha256(shell) != environment_sha256(desktop)


class TestCapture:
    """dg.trace records as the episode runs and keeps partial traces."""

    def test_local_episode_end_to_end(self, dg: DeepGym, traces_dir: Path) -> None:
        with dg.trace(WRITE_ENV, AGENT, variant=0, seed=3) as tr:
            obs = tr.start()
            prompt = [TraceMessage(role='user', content=f'{WRITE_ENV.task}\n{obs.content}')]
            tr.step(Action(action_type='bash', content='echo hello > out.txt'), [
                *prompt,
                TraceMessage(role='assistant', content='echo', token_ids=(5,), logprobs=(-0.5,)),
            ])  # fmt: skip
            tr.step(Action(action_type='done'), _reply('done'))
            result = tr.score()

        trace = load_trace(tr.path)
        assert trace == tr.trace
        assert tr.path.parent == traces_dir
        assert result.passed and trace.has_verdict and trace.error is None
        assert trace.provenance.provider == 'local'
        assert trace.provenance.env.seed == 3
        assert trace.provenance.env.verifier_sha256 == EnvFingerprint.of(WRITE_ENV).verifier_sha256
        assert [step.reward for step in trace.steps] == [0.5, 0.5]
        assert trace.steps[0].step_verifier == {'score': 0.5, 'passed': True, 'details': 'step 1'}
        assert trace.steps[0].exit_code == 0
        assert trace.steps[0].messages[-1].token_ids == (5,)
        assert trace.steps[1].observation is not None and trace.steps[1].observation.done

    def test_no_step_verifier_means_no_step_reward(self, dg: DeepGym) -> None:
        env = WRITE_ENV.model_copy(update={'step_verifier_code': ''})

        with dg.trace(env, AGENT) as tr:
            tr.start()
            tr.step(Action(action_type='done'))

        step = load_trace(tr.path).steps[0]
        assert (step.reward, step.step_verifier) == (None, None)

    def test_exception_keeps_partial_trace(self, dg: DeepGym) -> None:
        with pytest.raises(RuntimeError, match='policy crashed'):
            with dg.trace(WRITE_ENV, AGENT) as tr:
                tr.start()
                tr.step(Action(action_type='bash', content='echo hello > out.txt'))
                raise RuntimeError('policy crashed')

        trace = load_trace(tr.path)
        assert trace.status == 'incomplete' and not trace.has_verdict
        assert trace.error == 'RuntimeError: policy crashed'
        assert len(trace.steps) == 1 and trace.finished_at is not None

    def test_rejected_action_is_recorded(self, dg: DeepGym) -> None:
        env = WRITE_ENV.model_copy(update={'allowed_actions': frozenset({'bash', 'done'})})

        with dg.trace(env, AGENT) as tr:
            tr.start()
            with pytest.raises(ActionError):
                tr.step(Action(action_type='code', content='print(1)'), _reply('print(1)'))
            tr.step(Action(action_type='done'), _reply('done'))

        rejected, done = load_trace(tr.path).steps
        assert rejected.observation is None and rejected.error.startswith('ActionError')
        assert (rejected.index, done.index) == (1, 2)
        assert done.error is None

    def test_failing_step_verifier_is_recorded(self, dg: DeepGym) -> None:
        env = WRITE_ENV.model_copy(update={'step_verifier_code': 'print("not a score")'})

        with pytest.raises(VerifierError), dg.trace(env, AGENT) as tr:
            tr.start()
            tr.step(Action(action_type='bash', content='echo hi'))

        trace = load_trace(tr.path)
        step = trace.steps[0]
        assert step.error.startswith('VerifierError') and step.reward is None
        assert step.observation is not None and step.observation.done
        assert trace.error.startswith('VerifierError') and trace.status == 'incomplete'

    def test_failing_final_verifier_is_recorded(self, dg: DeepGym) -> None:
        env = WRITE_ENV.model_copy(update={'final_verifier_code': 'print("garbage")'})

        with dg.trace(env, AGENT) as tr:
            tr.start()
            tr.step(Action(action_type='done'))
            with pytest.raises(VerifierError):
                tr.score()

        trace = load_trace(tr.path)
        assert trace.result is None and trace.error.startswith('VerifierError')

    def test_killed_process_leaves_an_incomplete_trace(self, traces_dir: Path) -> None:
        code = (
            'import os\n'
            'from deepgym import DeepGym\n'
            'from deepgym.models import Action, MultiTurnEnvironment\n'
            'from deepgym.traces import AgentMeta\n'
            "env = MultiTurnEnvironment(task='t', final_verifier_code='print(1)')\n"
            "tr = DeepGym(mode='local').trace(env, AgentMeta(name='a'))\n"
            'tr.start()\n'
            "tr.step(Action(action_type='bash', content='echo one'))\n"
            'os._exit(9)\n'
        )

        proc = subprocess.run(
            [sys.executable, '-c', code],
            env={**os.environ, 'PYTHONPATH': str(SRC)},
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 9, proc.stderr
        (path,) = traces_dir.glob('*.jsonl')
        trace = load_trace(path)
        assert trace.status == 'incomplete' and trace.error == UNFINISHED
        assert trace.steps[0].observation.content.strip() == 'one'

    def test_record_cut_mid_write_is_dropped(self, dg: DeepGym) -> None:
        with dg.trace(WRITE_ENV, AGENT) as tr:
            tr.start()
            tr.step(Action(action_type='bash', content='echo hello > out.txt'))
        lines = tr.path.read_text().splitlines()
        tr.path.write_text('\n'.join(lines[:3]) + '\n' + lines[3][:20])

        trace = load_trace(tr.path)

        assert len(trace.steps) == 1 and trace.error == UNFINISHED

    def test_corrupt_record_raises(self, dg: DeepGym) -> None:
        with dg.trace(WRITE_ENV, AGENT) as tr:
            tr.start()
        lines = tr.path.read_text().splitlines()
        tr.path.write_text('\n'.join([lines[0], '{broken', *lines[1:]]) + '\n')

        with pytest.raises(DeepGymError, match='corrupt trace record'):
            load_trace(tr.path)

    def test_not_a_trace_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'rows.jsonl'
        path.write_text('{"event": "step"}\n')

        with pytest.raises(DeepGymError, match='not a deepgym trace'):
            load_trace(path)

    def test_traces_never_overwrite(self, dg: DeepGym, tmp_path: Path) -> None:
        first = dg.trace(WRITE_ENV, AGENT)
        first.path = tmp_path / 'same.jsonl'
        first.path.write_text('taken\n')

        with pytest.raises(FileExistsError):
            first.start()
        first.close()

        assert first.path.read_text() == 'taken\n'


class TestConsoleStorage:
    """record_episode stores traces next to the replay frames, backward compatibly."""

    def test_record_trace_keeps_replay_and_trace(self, tmp_path: Path) -> None:
        db = tmp_path / 'console.db'
        trace = graded_trace()

        episode_id = record_episode(trace, db_path=db)

        store = EpisodeStore(db)
        detail = store.get(episode_id)
        assert detail is not None and detail.environment == 'write_notes'
        assert [frame.content for frame in detail.frames] == ['$ ', '', '']
        assert detail.result == trace.result
        assert store.screenshot(episode_id, 0) is not None
        assert store.traces() == ([trace], 0)

    def test_trajectory_still_records(self, tmp_path: Path) -> None:
        db = tmp_path / 'console.db'

        record_episode(Trajectory(), environment='legacy', db_path=db)

        assert EpisodeStore(db).traces() == ([], 1)

    def test_trace_with_separate_result_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='carries its own result'):
            record_episode(graded_trace(), graded_trace().result, db_path=tmp_path / 'c.db')

    def test_old_database_gains_trace_column(self, tmp_path: Path) -> None:
        db = tmp_path / 'console.db'
        with sqlite3.connect(db) as conn:
            conn.execute(
                'CREATE TABLE episodes (id TEXT PRIMARY KEY, environment TEXT NOT NULL, '
                'steps INTEGER NOT NULL, total_reward REAL NOT NULL, score REAL, passed INTEGER, '
                'frames TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL)'
            )
            conn.execute(
                "INSERT INTO episodes VALUES ('old', 'e', 0, 0, NULL, NULL, '[]', NULL, 't')"
            )
        conn.close()

        record_episode(graded_trace(), db_path=db)

        store = EpisodeStore(db)
        assert store.traces() == ([graded_trace()], 1)
        assert store.get('old') is not None


def test_exported_local_episode_is_trainable(dg: DeepGym, tmp_path: Path) -> None:
    """A real local episode goes from capture to an RL row with its rewards."""
    with dg.trace(WRITE_ENV, AGENT) as tr:
        tr.start()
        prompt = TraceMessage(role='user', content=WRITE_ENV.task)
        command = 'echo hello > out.txt'
        obs = tr.step(Action(action_type='bash', content=command), [prompt, *_reply(command)])[0]
        seen = TraceMessage(role='user', content=f'exit {obs.metadata["exit_code"]}')
        tr.step(Action(action_type='done'), [seen, *_reply('done')])
        tr.score()

    report = export_traces([load_trace(tr.path)], 'jsonl', tmp_path / 'rl.jsonl', rows='rl')

    (row,) = [json.loads(line) for line in (tmp_path / 'rl.jsonl').read_text().splitlines()]
    assert report.written == 1
    assert (row['reward'], row['passed'], row['step_rewards']) == (1.0, True, [0.5, 0.5])
    assert [(m['role'], m['content']) for m in row['messages']] == [
        ('user', WRITE_ENV.task),
        ('assistant', command),
        ('user', 'exit 0'),
        ('assistant', 'done'),
    ]
