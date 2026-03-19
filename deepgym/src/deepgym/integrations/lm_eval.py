"""lm-evaluation-harness integration for DeepGym environments.

Wraps DeepGym environments as lm-eval tasks so any model evaluated with
EleutherAI's `lm_eval` can be scored on DeepGym coding environments.

Usage (register and run via lm-eval CLI)::

    from deepgym.integrations.lm_eval import register_deepgym_tasks
    register_deepgym_tasks()

    # Then run from CLI:
    # lm_eval --model hf --model_args pretrained=Qwen/Qwen2-0.5B-Instruct \\
    #         --tasks deepgym_coin_change,deepgym_two_sum \\
    #         --num_fewshot 0

Usage (programmatic)::

    from deepgym.integrations.lm_eval import make_lm_eval_task, register_deepgym_tasks
    from deepgym import load_environment

    env = load_environment('coin_change')
    TaskClass = make_lm_eval_task(env, task_name='coin_change')

    # Register all built-in environments (prefixed deepgym_*)
    register_deepgym_tasks()

Task interface:
    - Each DeepGym environment becomes one lm-eval task named ``deepgym_{env_name}``.
    - The model is given the environment's ``task`` string as the prompt.
    - The generated code is evaluated by the DeepGym verifier.
    - Metric: ``deepgym_score`` (0.0-1.0) and ``deepgym_pass`` (0 or 1).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepgym.core import DeepGym
from deepgym.models import Environment

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def make_lm_eval_task(env: Environment, task_name: str) -> type:
    """Create an lm-eval Task class for a DeepGym environment.

    The returned class can be registered with lm-eval's task registry and
    used via the ``--tasks`` CLI flag or programmatically.

    Args:
        env: DeepGym environment with task prompt and verifier.
        task_name: Short name used in lm-eval (e.g. ``'coin_change'``).
            The registered task name will be ``deepgym_{task_name}``.

    Returns:
        A Task subclass compatible with lm-evaluation-harness v0.4+.

    Raises:
        ImportError: If ``lm_eval`` is not installed.
    """
    try:
        from lm_eval.api.metrics import mean  # type: ignore[import-untyped]
        from lm_eval.api.task import Task  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            'lm-evaluation-harness is required for this integration. '
            "Install with: pip install 'deepgym[lm_eval]'"
        ) from exc

    _env = env
    _task_name = f'deepgym_{task_name}'
    _dg = DeepGym(mode='auto')

    class DeepGymTask(Task):
        """lm-eval task backed by a DeepGym environment."""

        VERSION = 1
        DATASET_PATH = None  # type: ignore[assignment]
        DATASET_NAME = None  # type: ignore[assignment]

        def has_training_docs(self) -> bool:
            """Return False — DeepGym tasks have no training split."""
            return False

        def has_validation_docs(self) -> bool:
            """Return False — DeepGym tasks have no validation split."""
            return False

        def has_test_docs(self) -> bool:
            """Return True — the environment task IS the test document."""
            return True

        def test_docs(self) -> list[dict]:
            """Return a single-document list containing the environment task."""
            return [{'task': _env.task, 'env_name': task_name}]

        def doc_to_text(self, doc: dict) -> str:
            """Convert document to prompt text.

            Args:
                doc: Document dict from :meth:`test_docs`.

            Returns:
                The environment's task description as the model prompt.
            """
            return doc['task']

        def doc_to_target(self, doc: dict) -> str:
            """Return empty string — targets are verified, not matched.

            Args:
                doc: Document dict from :meth:`test_docs`.

            Returns:
                Empty string (target is implicit in the verifier).
            """
            return ''

        def construct_requests(self, doc: dict, ctx: str):  # type: ignore[override]
            """Construct a generation request for the document.

            Args:
                doc: Document dict.
                ctx: Context/prompt string.

            Returns:
                A single greedy generation request.
            """
            from lm_eval.api.instance import Instance  # type: ignore[import-untyped]

            return [
                Instance(
                    request_type='generate_until',
                    doc=doc,
                    arguments=(ctx, {'until': [], 'max_gen_toks': 1024}),
                    idx=0,
                )
            ]

        def process_results(self, doc: dict, results: list) -> dict:
            """Run the generated code through the DeepGym verifier.

            Args:
                doc: Document dict from :meth:`test_docs`.
                results: List of generated strings from the model.

            Returns:
                Dict with ``deepgym_score`` (float) and ``deepgym_pass`` (int).
            """
            generated_code = results[0] if results else ''
            try:
                run_result = _dg.run(_env, model_output=generated_code)
                return {
                    'deepgym_score': run_result.score,
                    'deepgym_pass': int(run_result.passed),
                }
            except Exception as exc:
                logger.warning('DeepGym task %r evaluation failed: %s', _task_name, exc)
                return {'deepgym_score': 0.0, 'deepgym_pass': 0}

        def aggregation(self) -> dict:
            """Return aggregation functions for each metric.

            Returns:
                Dict mapping metric names to aggregation callables.
            """
            return {
                'deepgym_score': mean,
                'deepgym_pass': mean,
            }

        def higher_is_better(self) -> dict:
            """Return whether higher metric values are better.

            Returns:
                Dict mapping metric names to booleans.
            """
            return {
                'deepgym_score': True,
                'deepgym_pass': True,
            }

    DeepGymTask.__name__ = f'DeepGymTask_{task_name}'
    DeepGymTask.__qualname__ = f'DeepGymTask_{task_name}'
    return DeepGymTask


def register_deepgym_tasks(env_names: list[str] | None = None) -> list[str]:
    """Register DeepGym environments as lm-eval tasks.

    Registers each environment as a task named ``deepgym_{env_name}``.

    Args:
        env_names: List of environment names to register. If None, registers
            all built-in environments.

    Returns:
        List of registered task names (e.g. ``['deepgym_coin_change', ...]``).

    Raises:
        ImportError: If ``lm_eval`` is not installed.
    """
    try:
        from lm_eval.api.registry import register_task  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            'lm-evaluation-harness is required for this integration. '
            "Install with: pip install 'deepgym[lm_eval]'"
        ) from exc

    from deepgym.registry import list_environments, load_environment

    names = env_names or list_environments()
    registered: list[str] = []

    for name in names:
        try:
            env = load_environment(name)
            task_class = make_lm_eval_task(env, task_name=name)
            register_task(f'deepgym_{name}')(task_class)
            registered.append(f'deepgym_{name}')
        except Exception as exc:
            logger.warning('Failed to register lm-eval task for %r: %s', name, exc)

    logger.info('Registered %d DeepGym lm-eval tasks', len(registered))
    return registered
