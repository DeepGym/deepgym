"""Environment quality gate (``deepgym qa``).

Checks that an environment is fit to train on: its reference solution scores
1.0 every time, empty and stub solutions don't, the heuristic reward-hack attacks
fail, (optionally) every mutant of the reference fails, and (optionally) a model's
pass rate is neither 0 nor 1.

Modules:
    checks: reference, determinism, empty/stub and reward-hack checks.
    mutants: mutation testing of the verifier with AST mutants of the reference.
    difficulty: pass@1 / pass@k from an OpenAI-compatible model endpoint.
    report: QAReport model, per-environment JSON files, CLI table.
    gate: run_qa, which runs every check on one environment.
"""

from deepgym.qa.difficulty import client_from_env
from deepgym.qa.gate import QAOptions, run_qa
from deepgym.qa.report import QACheck, QAReport, format_checks, format_table, write_report

__all__ = [
    'QACheck',
    'QAOptions',
    'QAReport',
    'client_from_env',
    'format_checks',
    'format_table',
    'run_qa',
    'write_report',
]
