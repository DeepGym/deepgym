"""Tests for deepgym.integrations.lm_eval against the real lm-evaluation-harness.

The tasks run through ``lm_eval.simple_evaluate`` with a scripted model, so lm-eval
builds the requests (including the ``metadata`` keyword it passes to
``construct_requests``), calls ``process_results`` and aggregates the metrics itself.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from deepgym.core import DeepGym
from deepgym.integrations.lm_eval import make_lm_eval_task
from deepgym.models import Environment

REVERSE_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs' / 'reverse_string'
GOOD_SOLUTION = 'def reverse_string(s: str) -> str:\n    return s[::-1]\n'


@pytest.fixture()
def env() -> Environment:
    """Return reverse_string environment for lm-eval tests."""
    return Environment(
        task=(REVERSE_DIR / 'task.md').read_text(encoding='utf-8'),
        verifier_path=REVERSE_DIR / 'verifier.py',
    )


@pytest.fixture()
def lm_eval() -> ModuleType:
    """The installed lm_eval package; the test is skipped without it."""
    return pytest.importorskip('lm_eval')


def test_raises_import_error_when_lm_eval_missing(env: Environment) -> None:
    with patch('builtins.__import__', side_effect=ImportError('no lm_eval')):
        with pytest.raises(ImportError, match='lm-evaluation-harness'):
            make_lm_eval_task(env, task_name='reverse_string')


def _evaluate(lm_eval: ModuleType, reply: str, *tasks: object) -> tuple[dict, list]:
    """Run *tasks* through lm_eval.simple_evaluate with a model that always says *reply*."""
    from lm_eval.api.model import LM
    from lm_eval.tasks import TaskManager

    class Scripted(LM):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list = []

        def generate_until(self, requests: list, disable_tqdm: bool = False) -> list[str]:
            self.requests.extend(requests)
            return [reply] * len(requests)

        def loglikelihood(self, requests: list, disable_tqdm: bool = False) -> list:
            raise NotImplementedError

        def loglikelihood_rolling(self, requests: list, disable_tqdm: bool = False) -> list:
            raise NotImplementedError

    model = Scripted()
    # Our tasks are objects; skip indexing lm-eval's thousands of built-in task configs.
    task_manager = TaskManager(include_defaults=False)
    out = lm_eval.simple_evaluate(
        model=model, tasks=list(tasks), task_manager=task_manager, bootstrap_iters=0
    )
    return out['results'], model.requests


def test_fenced_answer_is_scored_by_the_verifier(
    lm_eval: ModuleType, env: Environment, local_dg: DeepGym
) -> None:
    """The task builds, lm-eval queries the model, and the fenced code inside is what runs."""
    results, requests = _evaluate(
        lm_eval,
        f'Sure:\n```python\n{GOOD_SOLUTION}```',
        make_lm_eval_task(env, 'reverse_string', local_dg),
    )

    metrics = results['deepgym_reverse_string']
    assert metrics['deepgym_score,none'] == 1.0
    assert metrics['deepgym_pass,none'] == 1.0
    assert metrics['deepgym_verdict,none'] == 1.0
    prompt, generation_kwargs = requests[0].args
    assert prompt == env.task
    # No stop strings: code has blank lines, so only EOS or max_gen_toks may end it.
    assert generation_kwargs['until'] == []


def test_a_verifier_failure_is_not_a_wrong_answer(
    lm_eval: ModuleType, env: Environment, local_dg: DeepGym
) -> None:
    broken = Environment(task='t', verifier_code='raise SystemExit(3)\n', verifier_mode='script')

    results, _ = _evaluate(
        lm_eval,
        GOOD_SOLUTION,
        make_lm_eval_task(env, 'reverse_string', local_dg),
        make_lm_eval_task(broken, 'broken', local_dg),
    )

    assert results['deepgym_reverse_string']['deepgym_score,none'] == 1.0
    failed = results['deepgym_broken']
    assert math.isnan(failed['deepgym_score,none'])
    assert math.isnan(failed['deepgym_pass,none'])
    assert failed['deepgym_verdict,none'] == 0.0


def test_a_wrong_answer_scores_zero_with_a_verdict(
    lm_eval: ModuleType, env: Environment, local_dg: DeepGym
) -> None:
    results, _ = _evaluate(lm_eval, 'pass\n', make_lm_eval_task(env, 'reverse_string', local_dg))

    metrics = results['deepgym_reverse_string']
    assert metrics['deepgym_score,none'] == 0.0
    assert metrics['deepgym_pass,none'] == 0.0
    assert metrics['deepgym_verdict,none'] == 1.0
