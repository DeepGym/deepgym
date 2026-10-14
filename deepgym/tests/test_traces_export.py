"""Tests for trace exporters (golden files) and ``deepgym traces export``.

Regenerate the golden files after an intended format change with
``DEEPGYM_UPDATE_GOLDEN=1 pytest tests/test_traces_export.py`` and review the diff.
"""

import base64
import json
import os
import sys
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.cli import main
from deepgym.console.episode_store import record_episode
from deepgym.exceptions import DeepGymError
from deepgym.models import Action, MultiTurnEnvironment, Trajectory
from deepgym.traces import (
    AgentMeta,
    Trace,
    TraceMessage,
    export_traces,
    load_trace,
    to_atif,
    write_atif,
)
from tests.trace_samples import PNG, crashed_trace, graded_trace, unjudged_trace

GOLDEN = Path(__file__).resolve().parent / 'golden' / 'traces'
UPDATE = os.environ.get('DEEPGYM_UPDATE_GOLDEN') == '1'

# One model turn issues two bash commands, the next a write_file tool call plus a bash
# command, the last finishes. The step verifier pays step N a reward of N/10, so a reward
# moved onto any other action shows.
THREE_FILES = MultiTurnEnvironment(
    name='three_files',
    task='Create a.txt, b.txt and c.txt.',
    step_verifier_code='import os\nprint(int(os.environ["STEP_NUMBER"]) / 10)\n',
    final_verifier_code='import json\nprint(json.dumps({"score": 1.0, "passed": True}))\n',
)
REPLIES = ('run two commands', 'write c.txt, then read it', 'done')
ACTIONS = (
    Action(action_type='bash', content='touch a.txt && echo A'),
    Action(action_type='bash', content='touch b.txt && echo B'),
    Action(action_type='write_file', path='c.txt', content='c'),
    Action(action_type='bash', content='cat c.txt'),
    Action(action_type='done'),
)
# (turn, action, observation, step reward) of every action, in order.
EXPECTED = [
    (1, ACTIONS[0], 'A\n', 0.1),
    (1, ACTIONS[1], 'B\n', 0.2),
    (2, ACTIONS[2], 'wrote c.txt', 0.3),
    (2, ACTIONS[3], 'c', 0.4),
    (3, ACTIONS[4], '', 0.5),
]


def _samples() -> list[Trace]:
    return [graded_trace(), crashed_trace(), unjudged_trace()]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def _rows(out: Path, fmt: str) -> list[dict]:
    """Read back the rows of a JSONL or Parquet export."""
    if fmt == 'jsonl':
        return _jsonl(out)
    return pytest.importorskip('pyarrow.parquet').read_table(out).to_pylist()


def _reply(content: str) -> TraceMessage:
    return TraceMessage(role='assistant', content=content)


@pytest.fixture
def multi_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Trace, Trajectory]:
    """A local episode of THREE_FILES, and the trajectory its session recorded."""
    monkeypatch.setenv('DEEPGYM_TRACES_DIR', str(tmp_path / 'traces'))
    with DeepGym(mode='local').trace(THREE_FILES, AgentMeta(name='scripted')) as tr:
        tr.start()
        prompt = TraceMessage(role='user', content=THREE_FILES.task)
        first = tr.step(ACTIONS[0], [prompt, _reply(REPLIES[0])]).observation
        second = tr.step(ACTIONS[1]).observation  # the same reply: no new messages
        seen = [TraceMessage(role='tool', content=obs.content) for obs in (first, second)]
        third = tr.step(ACTIONS[2], [*seen, _reply(REPLIES[1])]).observation
        fourth = tr.step(ACTIONS[3]).observation
        seen = [TraceMessage(role='tool', content=obs.content) for obs in (third, fourth)]
        tr.step(ACTIONS[4], [*seen, _reply(REPLIES[2])])
        tr.score()
    return load_trace(tr.path), tr._session.trajectory


def _trace_records(data: dict) -> list[tuple]:
    """(turn, action, observation, reward) of each step of a whole exported trace."""
    turn = {index: number for number, steps in enumerate(data['turns'], 1) for index in steps}
    return [
        (turn[step['index']], Action.model_validate(step['action']),
         step['observation'] and step['observation']['content'], step['reward'])
        for step in data['steps']
    ]  # fmt: skip


def _rl_records(row: dict) -> list[tuple]:
    """(turn, action, observation, reward) of each step of an RL row; a turn is its reply."""
    replies = sorted({step['message_index'] for step in row['steps']})
    return [
        (replies.index(step['message_index']) + 1, Action.model_validate(step['action']),
         step['observation'], step['reward'])
        for step in row['steps']
    ]  # fmt: skip


def _atif_records(document: dict) -> list[tuple]:
    """(turn, action, observation, reward) of each tool call; a turn is one agent step."""
    records = []
    calls = [step for step in document['steps'] if step.get('tool_calls')]
    for turn, step in enumerate(calls, 1):
        results = {result['source_call_id']: result for result in step['observation']['results']}
        for call in step['tool_calls']:
            result = results[call['tool_call_id']]
            action = Action.model_validate(
                {'action_type': call['function_name'], **call['arguments']}
            )
            records.append((turn, action, result['content'], result['extra'].get('reward')))
    return records


def _check_golden(actual: Path, name: str) -> None:
    """Compare *actual* with the golden file *name* (as JSON documents or JSONL rows)."""
    golden = GOLDEN / name
    if UPDATE:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(actual.read_bytes())
    if name.endswith('.jsonl'):
        assert _jsonl(actual) == _jsonl(golden)
    else:
        assert json.loads(actual.read_text()) == json.loads(golden.read_text())


class TestExporters:
    """Each format matches its golden file and flags or drops traces without a verdict."""

    def test_trace_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / 'traces.jsonl'

        report = export_traces(_samples(), 'jsonl', out)

        assert (report.written, report.skipped) == (3, {})
        _check_golden(out, 'traces.jsonl')
        assert [Trace.model_validate(row) for row in _jsonl(out)] == _samples()

    def test_rl_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / 'rl.jsonl'

        report = export_traces(_samples(), 'jsonl', out, rows='rl')

        assert (report.written, report.skipped) == (2, {'no_model_messages': 1})
        _check_golden(out, 'rl.jsonl')
        graded, unjudged = _jsonl(out)
        assert (graded['reward'], graded['has_verdict']) == (1.0, True)
        assert (unjudged['reward'], unjudged['passed'], unjudged['has_verdict']) == (
            None,
            None,
            False,
        )

    @pytest.mark.parametrize('fmt', ['jsonl', 'parquet'])
    def test_rl_rows_keep_screenshots(self, tmp_path: Path, fmt: str) -> None:
        trace = graded_trace()
        shot = base64.b64encode(PNG).decode()
        first = trace.steps[0]
        seen = first.model_copy(
            update={'observation': first.observation.model_copy(update={'screenshot': shot})}
        )
        trace = trace.model_copy(update={'steps': (seen, *trace.steps[1:])})

        export_traces([trace], fmt, tmp_path / f'rl.{fmt}', rows='rl')

        (row,) = _rows(tmp_path / f'rl.{fmt}', fmt)
        assert [step['screenshot'] for step in row['steps']] == [shot, None, None]
        assert [step['done'] for step in row['steps']] == [False, None, True]
        # The screenshot the first action answered to.
        start = trace.initial_observation
        assert (row['initial_observation'], row['initial_screenshot']) == (
            start.content,
            start.screenshot,
        )

    def test_only_valid_drops_unjudged_traces(self, tmp_path: Path) -> None:
        out = tmp_path / 'valid.jsonl'

        report = export_traces(_samples(), 'jsonl', out, only_valid=True)

        assert (report.written, report.skipped) == (1, {'incomplete': 1, 'no_verdict': 1})
        assert [row['trace_id'] for row in _jsonl(out)] == ['trace0001']

    def test_parquet_rows(self, tmp_path: Path) -> None:
        pq = pytest.importorskip('pyarrow.parquet')
        summary, rl = tmp_path / 'summary.parquet', tmp_path / 'rl.parquet'

        export_traces(_samples(), 'parquet', summary)
        export_traces(_samples(), 'parquet', rl, rows='rl')

        summary_rows = pq.read_table(summary).to_pylist()
        readable = tmp_path / 'summary.jsonl'
        readable.write_text(
            ''.join(
                json.dumps({**row, 'trace': json.loads(row['trace'])}) + '\n'
                for row in summary_rows
            )
        )
        _check_golden(readable, 'summary.jsonl')
        assert [Trace.model_validate_json(row['trace']) for row in summary_rows] == _samples()
        rl_rows = pq.read_table(rl).to_pylist()
        golden_rl = _jsonl(GOLDEN / 'rl.jsonl')
        assert [row['trace_id'] for row in rl_rows] == [row['trace_id'] for row in golden_rl]
        message = rl_rows[0]['messages'][2]
        assert (message['token_ids'], message['logprobs']) == ([11, 12, 13], [-0.25, -0.5, -0.125])
        assert rl_rows[0]['messages'][0]['token_ids'] is None
        assert (rl_rows[1]['reward'], rl_rows[1]['step_rewards']) == (None, [0.5, None, 0.5])

    def test_atif_document(self, tmp_path: Path) -> None:
        out = tmp_path / 'atif'

        report = export_traces([graded_trace()], 'atif', out)

        assert report.written == 1
        _check_golden(out / 'trace0001.json', 'atif/trace0001.json')
        assert (out / 'images' / 'trace0001' / 'reset.png').read_bytes() == PNG

    @pytest.mark.parametrize('make', [graded_trace, crashed_trace, unjudged_trace])
    def test_atif_validates_against_harbor(self, make) -> None:
        harbor = pytest.importorskip('harbor.models.trajectories')

        document, _ = to_atif(make())

        harbor.Trajectory.model_validate(document)

    def test_schema_1_0_traces_still_load_and_export(self, tmp_path: Path) -> None:
        pq = pytest.importorskip('pyarrow.parquet')
        # Written by schema 1.0: an event log and a JSONL export of the same trace.
        old = load_trace(GOLDEN / 'v1_0' / 'events.jsonl')
        (row,) = _jsonl(GOLDEN / 'v1_0' / 'export.jsonl')

        export_traces([old], 'jsonl', tmp_path / 'out.jsonl')
        export_traces([old], 'parquet', tmp_path / 'out.parquet')
        export_traces([old], 'atif', tmp_path / 'atif')

        assert Trace.model_validate(row) == old
        assert old.schema_version == '1.0'
        provenance = old.provenance
        assert (provenance.deepgym_commit, provenance.env.digest_unavailable) == (None, None)
        assert old.prior_results == ()
        (parquet,) = pq.read_table(tmp_path / 'out.parquet').to_pylist()
        atif = json.loads((tmp_path / 'atif' / 'trace0001.json').read_text())
        for exported in (*_jsonl(tmp_path / 'out.jsonl'), json.loads(parquet['trace'])):
            assert Trace.model_validate(exported) == old
        assert atif['extra']['deepgym']['schema_version'] == '1.0'

    def test_atif_needs_whole_traces(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='whole traces'):
            export_traces(_samples(), 'atif', tmp_path, rows='rl')

    def test_atif_keeps_unjudged_step_verifier_output(self) -> None:
        trace = graded_trace()
        unjudged = trace.steps[0].model_copy(
            update={'reward': None, 'step_verifier': {'score': 0, 'exit_code': 2}}
        )
        trace = trace.model_copy(update={'steps': (unjudged, *trace.steps[1:])})

        document, _ = to_atif(trace)

        action = next(step for step in document['steps'] if step.get('tool_calls'))
        (result,) = action['observation']['results']
        assert result['extra']['step_verifier'] == {'score': 0, 'exit_code': 2}
        assert 'reward' not in result['extra']

    def test_duplicate_trace_ids_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='share ids'):
            export_traces([graded_trace(), graded_trace()], 'atif', tmp_path / 'atif')
        with pytest.raises(ValueError, match='share ids'):
            write_atif([graded_trace(), graded_trace()], tmp_path / 'direct')
        upper = graded_trace().model_copy(update={'trace_id': 'TRACE0001'})
        with pytest.raises(ValueError, match='share ids'):
            write_atif([graded_trace(), upper], tmp_path / 'cased')

    def test_atif_refuses_a_populated_directory(self, tmp_path: Path) -> None:
        export_traces(_samples(), 'atif', tmp_path / 'atif')

        with pytest.raises(DeepGymError, match='not empty'):
            export_traces(_samples(), 'atif', tmp_path / 'atif', only_valid=True)


class TestMultiActionTurns:
    """A turn that issues several actions keeps each one, in its turn, in every export."""

    @pytest.mark.parametrize('fmt', ['jsonl', 'parquet'])
    def test_rl_rows(self, multi_action: tuple, tmp_path: Path, fmt: str) -> None:
        trace, _ = multi_action
        out = tmp_path / f'rl.{fmt}'

        report = export_traces([trace], fmt, out, rows='rl')

        (row,) = _rows(out, fmt)
        assert (report.written, report.skipped) == (1, {})
        assert _rl_records(row) == EXPECTED
        assert all(set(step['action']) == set(Action.model_fields) for step in row['steps'])
        assert row['step_rewards'] == [reward for *_, reward in EXPECTED]
        assert [(m['role'], m['content']) for m in row['messages']] == [
            ('user', THREE_FILES.task),
            ('assistant', REPLIES[0]),
            ('tool', 'A\n'),
            ('tool', 'B\n'),
            ('assistant', REPLIES[1]),
            ('tool', 'wrote c.txt'),
            ('tool', 'c'),
            ('assistant', REPLIES[2]),
        ]
        assert [row['messages'][step['message_index']]['content'] for step in row['steps']] == [
            REPLIES[0], REPLIES[0], REPLIES[1], REPLIES[1], REPLIES[2]
        ]  # fmt: skip

    @pytest.mark.parametrize('fmt', ['jsonl', 'parquet'])
    def test_trace_rows(self, multi_action: tuple, tmp_path: Path, fmt: str) -> None:
        trace, _ = multi_action
        out = tmp_path / f'traces.{fmt}'

        export_traces([trace], fmt, out)

        (row,) = _rows(out, fmt)
        data = json.loads(row['trace']) if fmt == 'parquet' else row
        assert Trace.model_validate(data) == trace
        assert data['turns'] == [[1, 2], [3, 4], [5]]
        assert _trace_records(data) == EXPECTED

    def test_atif(self, multi_action: tuple, tmp_path: Path) -> None:
        trace, _ = multi_action

        export_traces([trace], 'atif', tmp_path / 'atif')

        document = json.loads((tmp_path / 'atif' / f'{trace.trace_id}.json').read_text())
        assert _atif_records(document) == EXPECTED
        agent = [step for step in document['steps'] if step['source'] == 'agent']
        assert [step['message'] for step in agent] == list(REPLIES)
        assert [step['llm_call_count'] for step in agent] == [1, 1, 1]

    def test_atif_validates_against_harbor(self, multi_action: tuple) -> None:
        harbor = pytest.importorskip('harbor.models.trajectories')

        harbor.Trajectory.model_validate(to_atif(multi_action[0])[0])

    def test_to_trajectory_matches_the_session(self, multi_action: tuple) -> None:
        trace, recorded = multi_action

        trajectory = trace.to_trajectory()

        assert trajectory == recorded
        assert [action for _, action in trajectory.steps] == list(ACTIONS)
        shown = [obs.content for obs, _ in trajectory.steps[1:]]
        assert [*shown, trajectory.final_observation.content] == [obs for _, _, obs, _ in EXPECTED]
        assert trajectory.step_rewards == [reward for *_, reward in EXPECTED]


class TestCli:
    """``deepgym traces export`` reads the console database and trace files."""

    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
        monkeypatch.setattr(sys, 'argv', ['deepgym', 'traces', 'export', *args])
        main()

    def test_exports_from_db_and_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        db = tmp_path / 'console.db'
        graded = graded_trace()
        # Recorded mid-episode; the trace file below got further.
        early = graded.model_copy(update={'result': None, 'steps': (), 'finished_at': None})
        record_episode(early, db_path=db)
        record_episode(crashed_trace(), db_path=db)
        record_episode(unjudged_trace(), db_path=db)
        record_episode(Trajectory(), environment='legacy', db_path=db)
        files = tmp_path / 'traces'
        files.mkdir()
        (files / 'a.jsonl').write_text(_event_log(graded))
        out = tmp_path / 'out.jsonl'

        self._run(monkeypatch, '--format', 'jsonl', '--out', str(out), '--db', str(db),
                  '--traces', str(files), '--only-valid')  # fmt: skip

        assert [Trace.model_validate(row) for row in _jsonl(out)] == [graded]
        printed = capsys.readouterr().out
        assert 'Wrote 1 of 3 traces' in printed
        assert 'skipped 1: incomplete' in printed and 'skipped 1: no_verdict' in printed
        assert '1 console episodes were recorded without a trace' in printed

    def test_conflicting_copies_are_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        db = tmp_path / 'console.db'
        graded = graded_trace()
        record_episode(graded, db_path=db)
        # The same trace id on disk, graded differently: neither copy can be picked.
        failed = graded.result.model_copy(update={'score': 0.0, 'passed': False})
        (tmp_path / 'other.jsonl').write_text(
            _event_log(graded.model_copy(update={'result': failed}))
        )
        out = tmp_path / 'out.jsonl'

        with pytest.raises(SystemExit) as exit_info:
            self._run(monkeypatch, '--format', 'jsonl', '--out', str(out), '--db', str(db),
                      '--traces', str(tmp_path / 'other.jsonl'))  # fmt: skip

        assert exit_info.value.code == 1
        assert 'trace0001' in capsys.readouterr().err
        assert not out.exists()

    def test_identical_copies_are_merged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / 'console.db'
        graded = graded_trace()
        record_episode(graded, db_path=db)
        (tmp_path / 'same.jsonl').write_text(_event_log(graded))
        out = tmp_path / 'out.jsonl'

        self._run(monkeypatch, '--format', 'jsonl', '--out', str(out), '--db', str(db),
                  '--traces', str(tmp_path / 'same.jsonl'))  # fmt: skip

        assert [Trace.model_validate(row) for row in _jsonl(out)] == [graded]

    def test_defaults_read_traces_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('DEEPGYM_TRACES_DIR', str(tmp_path / 'traces'))
        monkeypatch.setattr('deepgym.console.db.DEFAULT_DB_PATH', tmp_path / 'missing.db')
        (tmp_path / 'traces').mkdir()
        (tmp_path / 'traces' / 'g.jsonl').write_text(_event_log(graded_trace()))
        out = tmp_path / 'atif'

        self._run(monkeypatch, '--format', 'atif', '--out', str(out))

        assert (out / 'trace0001.json').exists()

    def test_missing_source_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            self._run(monkeypatch, '--format', 'jsonl', '--out', str(tmp_path / 'o.jsonl'),
                      '--db', str(tmp_path / 'nope.db'))  # fmt: skip

        assert exit_info.value.code == 1
        assert 'not found' in capsys.readouterr().err

    def test_refuses_to_overwrite_an_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        source = tmp_path / 'trace.jsonl'
        source.write_text(_event_log(graded_trace()))

        with pytest.raises(SystemExit):
            self._run(monkeypatch, '--format', 'jsonl', '--rows', 'rl', '--out', str(source),
                      '--traces', str(source))  # fmt: skip

        assert 'is one of the inputs' in capsys.readouterr().err
        assert source.read_text() == _event_log(graded_trace())

    def test_exports_in_a_traces_directory_are_not_read_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        traces = tmp_path / 'traces'
        traces.mkdir()
        (traces / 'trace.jsonl').write_text(_event_log(graded_trace()))
        first = traces / 'export.jsonl'
        self._run(monkeypatch, '--format', 'jsonl', '--out', str(first), '--traces',
                  str(traces / 'trace.jsonl'))  # fmt: skip

        self._run(monkeypatch, '--format', 'jsonl', '--out', str(tmp_path / 'again.jsonl'),
                  '--traces', str(traces))  # fmt: skip

        assert _jsonl(tmp_path / 'again.jsonl') == _jsonl(first)
        assert '1 .jsonl files in the traces directories are not traces' in capsys.readouterr().out


def _event_log(trace: Trace) -> str:
    """Write *trace* as the event file :class:`TracedSession` would have recorded."""
    events = [
        {
            'event': 'start',
            'schema_version': trace.schema_version,
            'trace_id': trace.trace_id,
            'task': trace.task,
            'provenance': trace.provenance.model_dump(mode='json'),
        },
        {'event': 'reset', 'observation': trace.initial_observation.model_dump(mode='json')},
        *({'event': 'step', 'step': step.model_dump(mode='json')} for step in trace.steps),
    ]
    if trace.result is not None:
        for result in (*trace.prior_results, trace.result):
            events.append({'event': 'result', 'result': result.model_dump(mode='json')})
    events.append({'event': 'end', 'error': trace.error, 'finished_at': '2026-10-01T12:00:04Z'})
    return ''.join(json.dumps(event) + '\n' for event in events)
