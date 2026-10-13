"""Tests for trace exporters (golden files) and ``deepgym traces export``.

Regenerate the golden files after an intended format change with
``DEEPGYM_UPDATE_GOLDEN=1 pytest tests/test_traces_export.py`` and review the diff.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from deepgym.cli import main
from deepgym.console.episode_store import record_episode
from deepgym.models import Trajectory
from deepgym.traces import Trace, export_traces, to_atif
from tests.trace_samples import PNG, crashed_trace, graded_trace, unjudged_trace

GOLDEN = Path(__file__).resolve().parent / 'golden' / 'traces'
UPDATE = os.environ.get('DEEPGYM_UPDATE_GOLDEN') == '1'


def _samples() -> list[Trace]:
    return [graded_trace(), crashed_trace(), unjudged_trace()]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


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

    def test_atif_needs_whole_traces(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='whole traces'):
            export_traces(_samples(), 'atif', tmp_path, rows='rl')


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
        record_episode(graded.model_copy(update={'result': None, 'steps': ()}), db_path=db)
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
        events.append({'event': 'result', 'result': trace.result.model_dump(mode='json')})
    events.append({'event': 'end', 'error': trace.error, 'finished_at': '2026-10-01T12:00:04Z'})
    return ''.join(json.dumps(event) + '\n' for event in events)
