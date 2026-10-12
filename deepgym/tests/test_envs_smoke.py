"""Smoke test: every built-in single-turn environment passes the QA gate."""

import json
from pathlib import Path

import pytest

from deepgym.core import DeepGym
from deepgym.qa import run_qa
from deepgym.registry import load_environment

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'

_BUILTIN = json.loads((ENVS_DIR / 'registry.json').read_text(encoding='utf-8'))['environments']
_SINGLE_TURN_ENVS = [entry['path'] for entry in _BUILTIN if entry.get('type') != 'computer-use']


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    """Module-scoped local DeepGym client."""
    return DeepGym(mode='local')


@pytest.mark.parametrize('env_name', _SINGLE_TURN_ENVS)
def test_builtin_env_passes_qa_gate(env_name: str, dg: DeepGym) -> None:
    """The reference scores 1.0 on every run; empty, stub and attack solutions don't."""
    report = run_qa(env_name, dg, reruns=2)

    failed = [f'{check.name}: {check.details}' for check in report.checks if not check.passed]
    assert report.passed, f'{env_name} failed the QA gate: {failed}'


# ---------------------------------------------------------------------------
# The negative direction: a do-nothing solution must never score 1.0.
# ---------------------------------------------------------------------------

_EMPTY_SOLUTION = ''

#: Defines whatever the verifier calls, but computes nothing.  A verifier that
#: rewards the shape of the answer instead of the answer would hand this 1.0.
_STUB_SOLUTION = """\
def solve(*args, **kwargs):
    return None


def main(*args, **kwargs):
    return None


def reverse_string(*args, **kwargs):
    return None


class Solution:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return None
"""

_DO_NOTHING_SOLUTIONS = [_EMPTY_SOLUTION, _STUB_SOLUTION]


@pytest.mark.parametrize('env_name', _SINGLE_TURN_ENVS)
@pytest.mark.parametrize('solution', _DO_NOTHING_SOLUTIONS, ids=['empty', 'stub-returns-none'])
def test_do_nothing_solution_scores_below_one(env_name: str, solution: str, dg: DeepGym) -> None:
    """A do-nothing solution must score below 1.0 on every built-in environment.

    ``test_builtin_env_passes_qa_gate`` only checks that the *reference* passes.
    A verifier that always returns 1.0 would satisfy that and still be useless
    for training.  This is the complementary check: an empty file, and a file
    that defines the expected function but returns ``None``, must both fall
    short of full marks.
    """
    result = dg.run(load_environment(env_name), solution)

    assert result.score < 1.0, (
        f'{env_name} scored {result.score:.2f} for a do-nothing solution '
        f'(exit_code={result.exit_code}); its verifier rewards the wrong answer'
    )
