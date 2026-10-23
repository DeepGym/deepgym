"""Smoke test: every built-in single-turn environment passes the QA gate."""

import json
from pathlib import Path

import pytest

from deepgym.core import DeepGym
from deepgym.judge import local_isolation_kind
from deepgym.qa import QAOptions, run_qa

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'

_BUILTIN = json.loads((ENVS_DIR / 'registry.json').read_text(encoding='utf-8'))['environments']
# cli_task's verifier runs the solution with subprocesses (git), which macOS refuses.
_NEEDS_SUBPROCESSES = pytest.mark.skipif(
    local_isolation_kind() == 'macos', reason='allow_subprocess needs bwrap or a sandbox provider'
)
_SINGLE_TURN_ENVS = [
    pytest.param(entry['path'], marks=_NEEDS_SUBPROCESSES if entry['path'] == 'cli_task' else ())
    for entry in _BUILTIN
    if entry.get('type') != 'computer-use'
]


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    """Module-scoped local DeepGym client."""
    return DeepGym(mode='local')


@pytest.mark.parametrize('env_name', _SINGLE_TURN_ENVS)
def test_builtin_env_passes_qa_gate(env_name: str, dg: DeepGym) -> None:
    """The reference scores 1.0 on every run; empty, stub and attack solutions don't."""
    report = run_qa(env_name, dg, QAOptions(reruns=2))

    failed = [f'{check.name}: {check.details}' for check in report.checks if not check.passed]
    assert report.passed, f'{env_name} failed the QA gate: {failed}'
