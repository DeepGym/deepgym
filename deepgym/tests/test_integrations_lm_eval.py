"""Tests for deepgym.integrations.lm_eval — lm-evaluation-harness task adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepgym.integrations.lm_eval import make_lm_eval_task, register_deepgym_tasks
from deepgym.models import Environment

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'
REVERSE_DIR = ENVS_DIR / 'reverse_string'

GOOD_SOLUTION = 'def reverse_string(s: str) -> str:\n    return s[::-1]\n'
BAD_SOLUTION = 'pass\n'


@pytest.fixture()
def env() -> Environment:
    """Return reverse_string environment for lm-eval tests."""
    return Environment(
        task=(REVERSE_DIR / 'task.md').read_text(encoding='utf-8'),
        verifier_path=REVERSE_DIR / 'verifier.py',
    )


def _make_mock_lm_eval() -> MagicMock:
    """Build a minimal mock of the lm_eval module tree."""
    mock_task_base = MagicMock()
    mock_task_base.__name__ = 'Task'

    # Task base class: allow subclassing
    class FakeTask:
        VERSION = 0
        DATASET_PATH = None
        DATASET_NAME = None

    mock_lm_eval = MagicMock()
    mock_lm_eval.api.task.Task = FakeTask
    mock_lm_eval.api.metrics.mean = sum  # simple stand-in

    return mock_lm_eval


class TestMakeLmEvalTask:
    """Verify make_lm_eval_task returns a usable Task subclass."""

    def test_raises_import_error_when_lm_eval_missing(self, env: Environment) -> None:
        with patch('builtins.__import__', side_effect=ImportError('no lm_eval')):
            with pytest.raises(ImportError, match='lm-evaluation-harness'):
                make_lm_eval_task(env, task_name='reverse_string')

    def test_returns_class(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')
        assert isinstance(TaskClass, type)

    def test_task_name_embedded_in_class_name(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')
        assert 'reverse_string' in TaskClass.__name__

    def test_task_methods_present(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')

        instance = TaskClass.__new__(TaskClass)
        assert hasattr(instance, 'has_test_docs')
        assert hasattr(instance, 'doc_to_text')
        assert hasattr(instance, 'process_results')
        assert hasattr(instance, 'aggregation')
        assert hasattr(instance, 'higher_is_better')

    def test_has_test_docs_returns_true(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')

        instance = TaskClass.__new__(TaskClass)
        assert instance.has_test_docs() is True
        assert instance.has_training_docs() is False
        assert instance.has_validation_docs() is False

    def test_test_docs_returns_single_doc(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')

        instance = TaskClass.__new__(TaskClass)
        docs = instance.test_docs()
        assert len(docs) == 1
        assert docs[0]['env_name'] == 'reverse_string'
        assert env.task in docs[0]['task']

    def test_doc_to_text_returns_task_prompt(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')

        instance = TaskClass.__new__(TaskClass)
        doc = {'task': env.task, 'env_name': 'reverse_string'}
        assert instance.doc_to_text(doc) == env.task

    def test_process_results_good_solution_scores_high(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')

        instance = TaskClass.__new__(TaskClass)
        doc = {'task': env.task, 'env_name': 'reverse_string'}
        metrics = instance.process_results(doc, [GOOD_SOLUTION])
        assert metrics['deepgym_score'] >= 0.9
        assert metrics['deepgym_pass'] == 1

    def test_process_results_bad_solution_scores_zero(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')

        instance = TaskClass.__new__(TaskClass)
        doc = {'task': env.task, 'env_name': 'reverse_string'}
        metrics = instance.process_results(doc, [BAD_SOLUTION])
        assert metrics['deepgym_score'] == 0.0
        assert metrics['deepgym_pass'] == 0

    def test_process_results_empty_results_returns_zero(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')

        instance = TaskClass.__new__(TaskClass)
        doc = {'task': env.task, 'env_name': 'reverse_string'}
        metrics = instance.process_results(doc, [])
        assert metrics['deepgym_score'] == 0.0

    def test_aggregation_keys_match_higher_is_better(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.instance': MagicMock(),
        }):
            TaskClass = make_lm_eval_task(env, task_name='reverse_string')

        instance = TaskClass.__new__(TaskClass)
        assert set(instance.aggregation().keys()) == set(instance.higher_is_better().keys())
        assert all(instance.higher_is_better().values())


class TestRegisterDeepGymTasks:
    """Verify register_deepgym_tasks calls register_task for each environment."""

    def test_raises_import_error_when_lm_eval_missing(self) -> None:
        with patch('builtins.__import__', side_effect=ImportError('no lm_eval')):
            with pytest.raises(ImportError, match='lm-evaluation-harness'):
                register_deepgym_tasks()

    def test_registers_specified_envs(self, env: Environment) -> None:
        mock_lm_eval = _make_mock_lm_eval()
        registered: list[str] = []

        def fake_register_task(name: str):
            registered.append(name)
            return lambda cls: cls

        mock_lm_eval.api.registry = MagicMock()
        mock_lm_eval.api.registry.register_task = fake_register_task

        with patch.dict('sys.modules', {
            'lm_eval': mock_lm_eval,
            'lm_eval.api': mock_lm_eval.api,
            'lm_eval.api.task': mock_lm_eval.api.task,
            'lm_eval.api.metrics': mock_lm_eval.api.metrics,
            'lm_eval.api.registry': mock_lm_eval.api.registry,
            'lm_eval.api.instance': MagicMock(),
        }):
            result = register_deepgym_tasks(env_names=['reverse_string', 'two_sum'])

        assert 'deepgym_reverse_string' in result
        assert 'deepgym_two_sum' in result
        assert len(result) == 2
