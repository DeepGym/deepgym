"""lm-evaluation-harness integration for DeepGym environments.

Wraps DeepGym environments as lm-eval tasks so any model evaluated with
EleutherAI's ``lm_eval`` can be scored on DeepGym coding environments.

Usage::

    import lm_eval
    from deepgym import load_environment
    from deepgym.integrations.lm_eval import make_lm_eval_task

    tasks = [
        make_lm_eval_task(load_environment(name), name, max_gen_toks=8192)
        for name in ('coin_change', 'two_sum')
    ]
    results = lm_eval.simple_evaluate(
        model='hf', model_args='pretrained=Qwen/Qwen2.5-0.5B-Instruct', tasks=tasks
    )

lm-eval keeps no registry of Python tasks (``register_task`` went away in 0.4.10), so
the tasks are passed to ``simple_evaluate`` as objects; ``lm_eval --tasks`` can't name them.

Task interface:
    - Each DeepGym environment becomes one lm-eval task named ``deepgym_{name}``, whose
      one document is the environment's ``task`` string, given to the model as the prompt.
    - The last fenced code block of the generation (or all of it) is run by the
      environment's verifier, as in the TRL and verl adapters.
    - Generation runs to the model's own maximum output, which the caller passes as
      ``max_gen_toks`` (lm-eval needs a number, and its backends would default to 256).
      lm-eval hands ``process_results`` the text alone, so a generation that reached it
      can't be told from one that ended, and is scored as written.
    - Metrics: ``deepgym_score`` (0.0-1.0) and ``deepgym_pass`` (0 or 1), averaged over the
      documents that got a verdict, and ``deepgym_verdict``, the fraction that did. A
      sandbox or verifier failure counts in neither score nor pass, instead of as a wrong
      answer.

Sources:
    - https://github.com/EleutherAI/lm-evaluation-harness (lm_eval/api/task.py:
      ConfigurableTask; lm_eval/config/task.py: TaskConfig; lm_eval/api/metrics.py: nanmean)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from deepgym.core import DeepGym
from deepgym.integrations.verl import extract_code
from deepgym.models import Environment

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask

logger = logging.getLogger(__name__)


def make_lm_eval_task(
    env: Environment, task_name: str, dg: DeepGym | None = None, *, max_gen_toks: int
) -> ConfigurableTask:
    """Create an lm-eval task for a DeepGym environment.

    Pass it to ``lm_eval.simple_evaluate(tasks=[...])``.

    Args:
        env: DeepGym environment with task prompt and verifier.
        task_name: Short name used in lm-eval (e.g. ``'coin_change'``).
            The task is named ``deepgym_{task_name}``.
        dg: DeepGym client that runs the verifier (auto-created if None).
        max_gen_toks: The model's maximum output, in tokens. lm-eval needs a number (its
            backends would otherwise cut every answer at 256), and only the caller knows
            the model's real maximum, so there is no default.

    Returns:
        A ConfigurableTask (lm-evaluation-harness 0.4.9+) with one document.

    Raises:
        ImportError: If ``lm_eval`` is not installed.
    """
    try:
        import datasets  # type: ignore[import-untyped]
        from lm_eval.api.task import ConfigurableTask  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            'lm-evaluation-harness is required for this integration. '
            "Install with: pip install 'deepgym[lm_eval]'"
        ) from exc

    name = f'deepgym_{task_name}'
    _dg = dg or DeepGym(mode='auto')

    def process_results(doc: dict, results: list[str]) -> dict[str, float]:
        """Run the generation's code through the verifier; NaN score and pass if no verdict."""
        result = _dg.run_batch(env, [extract_code(results[0])], max_parallel=1).results[0]
        if not result.has_verdict:
            logger.warning('%s got no verdict (%s): %s', name, result.error_type, result.output)
            return {'deepgym_score': math.nan, 'deepgym_pass': math.nan, 'deepgym_verdict': 0.0}
        return {
            'deepgym_score': result.score,
            'deepgym_pass': float(result.passed),
            'deepgym_verdict': 1.0,
        }

    return ConfigurableTask(
        config={
            'task': name,
            'custom_dataset': lambda **kwargs: {
                'test': datasets.Dataset.from_list([{'task': env.task}])
            },
            'test_split': 'test',
            'doc_to_text': 'task',
            'doc_to_target': '',
            'output_type': 'generate_until',
            # No stop strings (code has blank lines): generation ends at EOS or the model's
            # own maximum, which the caller states.
            'generation_kwargs': {'until': [], 'max_gen_toks': max_gen_toks, 'do_sample': False},
            'process_results': process_results,
            'metric_list': [
                {'metric': 'deepgym_score', 'aggregation': 'nanmean', 'higher_is_better': True},
                {'metric': 'deepgym_pass', 'aggregation': 'nanmean', 'higher_is_better': True},
                {'metric': 'deepgym_verdict', 'aggregation': 'mean', 'higher_is_better': True},
            ],
            'metadata': {'version': 3},
        }
    )
