"""Export traces as training data: JSONL, Parquet or ATIF, as whole traces or SFT/RL rows.

Row shapes (``rows=``):

* ``trace``: the whole trace. A JSONL line is the full trace. A Parquet row has
  summary columns plus the full trace as JSON in ``trace``, because traces hold
  free-form dicts (verifier metrics, observation metadata) that Parquet can't type.
* ``rl``: one conversation per trace, for SFT and RL trainers. ``messages`` concatenates
  the model messages recorded at each step (TRL's conversational format, the
  ``messages`` key of verl's multi-turn SFT dataset), with token ids and logprobs where
  recorded; ``reward`` and ``passed`` come from the final verifier; ``step_rewards``
  are the step verifier's scores (null where none ran). ``steps`` holds every action in
  order with what it produced (text, base64 PNG screenshot, done flag, exit code, or the
  error if nothing), its step reward, and ``message_index``: where in ``messages`` the
  reply that produced it is, shared by the actions of one turn. A trace whose first
  action has no model messages has no conversation to train on and is skipped.

Every row carries ``has_verdict``, and its ``score``/``reward`` and ``passed`` are null
without one, so a failed sandbox or verifier never reads as a scored 0. ``only_valid``
drops those traces instead. ATIF writes one document per trace (see :mod:`.atif`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from deepgym.exceptions import DeepGymError
from deepgym.traces.atif import write_atif
from deepgym.traces.schema import Trace, check_unique_ids, json_text

if TYPE_CHECKING:
    import pyarrow as pa

ExportFormat = Literal['jsonl', 'parquet', 'atif']
RowShape = Literal['trace', 'rl']
Row = dict[str, JsonValue]


class ExportReport(BaseModel):
    """What an export wrote and what it left out."""

    model_config = ConfigDict(frozen=True)

    written: int
    skipped: dict[str, int]
    """Traces left out, by reason: 'incomplete', 'no_verdict' (with ``only_valid``) or
    'no_model_messages' (``rl`` rows)."""


def rl_row(trace: Trace) -> Row | None:
    """Return *trace* as one SFT/RL row, or None if its first action has no model messages.

    A later action without messages came from the reply before it (see :attr:`Trace.turns`).
    """
    if not trace.steps or not trace.steps[0].messages:
        return None
    verdict = trace.result if trace.has_verdict else None
    provenance = trace.provenance
    messages: list[JsonValue] = []
    steps: list[JsonValue] = []
    for step in trace.steps:
        messages += [
            message.model_dump(mode='json', exclude_none=True) for message in step.messages
        ]
        seen = step.observation
        steps.append({
            'message_index': len(messages) - 1,
            'action': step.action.model_dump(mode='json'),
            'observation': None if seen is None else seen.content,
            'screenshot': None if seen is None else seen.screenshot,
            'done': None if seen is None else seen.done,
            'exit_code': step.exit_code,
            'error': step.error,
            'reward': step.reward,
        })  # fmt: skip
    return {
        'trace_id': trace.trace_id,
        'data_source': provenance.env.name or '',
        'agent': provenance.agent.name,
        'model': provenance.agent.model,
        'messages': messages,
        'reward': None if verdict is None else verdict.score,
        'passed': None if verdict is None else verdict.passed,
        'step_rewards': [step.reward for step in trace.steps],
        'steps': steps,
        'has_verdict': trace.has_verdict,
    }


def summary_row(trace: Trace) -> Row:
    """Return *trace* as a Parquet row: summary columns plus the full trace as JSON."""
    verdict = trace.result if trace.has_verdict else None
    provenance = trace.provenance
    return {
        'trace_id': trace.trace_id,
        'env_name': provenance.env.name,
        'env_variant': provenance.env.variant,
        'agent': provenance.agent.name,
        'model': provenance.agent.model,
        'provider': provenance.provider,
        'status': trace.status,
        'has_verdict': trace.has_verdict,
        'score': None if verdict is None else verdict.score,
        'passed': None if verdict is None else verdict.passed,
        'steps': len(trace.steps),
        'started_at': provenance.started_at.isoformat(),
        'trace': trace.model_dump_json(),
    }


def export_traces(
    traces: Sequence[Trace],
    fmt: ExportFormat,
    out: Path,
    *,
    rows: RowShape = 'trace',
    only_valid: bool = False,
) -> ExportReport:
    """Write *traces* to *out* in *fmt*.

    Args:
        traces: Traces to export.
        fmt: 'jsonl' or 'parquet' (a file), or 'atif' (a directory of documents).
        out: Output file, or directory for ATIF.
        rows: 'trace' for whole traces, 'rl' for SFT/RL rows (JSONL or Parquet only).
        only_valid: Leave out traces without a verdict instead of flagging them.

    Returns:
        How many records were written and why the others were skipped.

    Raises:
        ValueError: If ATIF is asked for RL rows, or two traces share an id.
        DeepGymError: If Parquet is asked for without pyarrow installed.
    """
    if fmt == 'atif' and rows == 'rl':
        raise ValueError("ATIF documents hold whole traces; use rows='trace'")
    check_unique_ids(traces)
    skipped: Counter[str] = Counter()
    kept: list[Trace] = []
    for trace in traces:
        if only_valid and not trace.has_verdict:
            skipped['incomplete' if trace.result is None else 'no_verdict'] += 1
        else:
            kept.append(trace)
    if fmt == 'atif':
        return ExportReport(written=write_atif(kept, out), skipped=+skipped)
    if rows == 'rl':
        records = [row for row in map(rl_row, kept) if row is not None]
        skipped['no_model_messages'] += len(kept) - len(records)
    elif fmt == 'jsonl':
        records = [trace.model_dump(mode='json') for trace in kept]
    else:
        records = [summary_row(trace) for trace in kept]
    if fmt == 'jsonl':
        _write_jsonl(records, out)
    else:
        _write_parquet(records, out, rows)
    return ExportReport(written=len(records), skipped=+skipped)


def _write_jsonl(records: Sequence[Row], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as file:
        file.writelines(json_text(record) + '\n' for record in records)


def _write_parquet(records: Sequence[Row], out: Path, rows: RowShape) -> None:
    """Write *records* to *out* as Parquet, with a fixed schema per row shape.

    Raises:
        DeepGymError: If pyarrow isn't installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DeepGymError("Parquet export needs pyarrow: pip install 'deepgym[traces]'") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(records), schema=_schema(rows)), out)


def _schema(rows: RowShape) -> pa.Schema:
    """Return the Parquet schema of a row shape (pyarrow must be importable)."""
    import pyarrow as pa

    if rows == 'trace':
        return pa.schema([
            ('trace_id', pa.string()),
            ('env_name', pa.string()),
            ('env_variant', pa.int64()),
            ('agent', pa.string()),
            ('model', pa.string()),
            ('provider', pa.string()),
            ('status', pa.string()),
            ('has_verdict', pa.bool_()),
            ('score', pa.float64()),
            ('passed', pa.bool_()),
            ('steps', pa.int64()),
            ('started_at', pa.string()),
            ('trace', pa.string()),
        ])  # fmt: skip
    message = pa.struct([
        ('role', pa.string()),
        ('content', pa.string()),
        ('token_ids', pa.list_(pa.int64())),
        ('logprobs', pa.list_(pa.float64())),
    ])  # fmt: skip
    action = pa.struct([
        ('action_type', pa.string()),
        ('content', pa.string()),
        ('x', pa.int64()),
        ('y', pa.int64()),
        ('to_x', pa.int64()),
        ('to_y', pa.int64()),
        ('path', pa.string()),
        ('amount', pa.int64()),
    ])  # fmt: skip
    step = pa.struct([
        ('message_index', pa.int64()),
        ('action', action),
        ('observation', pa.string()),
        ('screenshot', pa.string()),
        ('done', pa.bool_()),
        ('exit_code', pa.int64()),
        ('error', pa.string()),
        ('reward', pa.float64()),
    ])  # fmt: skip
    return pa.schema([
        ('trace_id', pa.string()),
        ('data_source', pa.string()),
        ('agent', pa.string()),
        ('model', pa.string()),
        ('messages', pa.list_(message)),
        ('reward', pa.float64()),
        ('passed', pa.bool_()),
        ('step_rewards', pa.list_(pa.float64())),
        ('steps', pa.list_(step)),
        ('has_verdict', pa.bool_()),
    ])  # fmt: skip
