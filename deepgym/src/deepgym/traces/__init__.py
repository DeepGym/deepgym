"""Graded agent traces: capture episodes with provenance and export them as training data.

``dg.trace(env, AgentMeta(...))`` records an episode to ``~/.deepgym/traces`` (or
``$DEEPGYM_TRACES_DIR``) as it runs; ``export_traces`` and ``deepgym traces export``
turn traces into JSONL, Parquet, ATIF or SFT/RL rows.
"""

from deepgym.traces.atif import to_atif, write_atif
from deepgym.traces.capture import TracedSession, is_trace_file, load_trace, traces_dir
from deepgym.traces.export import ExportReport, export_traces, rl_row, summary_row
from deepgym.traces.schema import (
    TRACE_SCHEMA_VERSION,
    AgentMeta,
    EnvFingerprint,
    Provenance,
    Trace,
    TraceMessage,
    TraceStep,
)

__all__ = [
    'TRACE_SCHEMA_VERSION',
    'AgentMeta',
    'EnvFingerprint',
    'ExportReport',
    'Provenance',
    'Trace',
    'TraceMessage',
    'TraceStep',
    'TracedSession',
    'export_traces',
    'is_trace_file',
    'load_trace',
    'rl_row',
    'summary_row',
    'to_atif',
    'traces_dir',
    'write_atif',
]
