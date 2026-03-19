"""Tests for deepgym.integrations.hf — HuggingFace Hub integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepgym.integrations.hf import (
    environment_from_dict,
    environment_to_dict,
    load_all_environments_from_hub,
    load_environment_from_hub,
    push_environment_to_hub,
    push_environments_to_hub,
    push_results_to_hub,
)
from deepgym.models import Environment

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'
REVERSE_DIR = ENVS_DIR / 'reverse_string'


@pytest.fixture()
def simple_env() -> Environment:
    """Return a minimal environment for serialisation tests."""
    return Environment(
        task='Write a function add(a, b) that returns a + b.',
        verifier_code='return 1.0 if hasattr(mod, "add") and mod.add(1, 2) == 3 else 0.0',
    )


@pytest.fixture()
def env_with_test_cases() -> Environment:
    """Return an environment with test_cases set."""
    return Environment(
        task='Write a function add(a, b) that returns a + b.',
        verifier_code='return float(mod.add(a, b) == a + b)',
        test_cases=[{'a': 1, 'b': 2, 'expected': 3}, {'a': 0, 'b': 0, 'expected': 0}],
    )


@pytest.fixture()
def env_with_verifier_path() -> Environment:
    """Return an environment backed by a real verifier file."""
    return Environment(
        task=(REVERSE_DIR / 'task.md').read_text(encoding='utf-8'),
        verifier_path=REVERSE_DIR / 'verifier.py',
    )


class TestEnvironmentToDict:
    """Verify environment_to_dict serialisation."""

    def test_basic_fields_present(self, simple_env: Environment) -> None:
        row = environment_to_dict(simple_env, env_name='add')
        assert row['env_name'] == 'add'
        assert row['task'] == simple_env.task
        assert row['verifier_code'] == simple_env.verifier_code
        assert row['test_cases'] == ''
        assert row['timeout'] == simple_env.timeout
        assert row['schema_version'] == '1.0'

    def test_test_cases_serialised_as_json(self, env_with_test_cases: Environment) -> None:
        row = environment_to_dict(env_with_test_cases)
        parsed = json.loads(row['test_cases'])
        assert parsed == env_with_test_cases.test_cases

    def test_verifier_path_read_to_code(self, env_with_verifier_path: Environment) -> None:
        row = environment_to_dict(env_with_verifier_path)
        expected = REVERSE_DIR.joinpath('verifier.py').read_text(encoding='utf-8')
        assert row['verifier_code'] == expected

    def test_empty_env_name_default(self, simple_env: Environment) -> None:
        row = environment_to_dict(simple_env)
        assert row['env_name'] == ''


class TestEnvironmentFromDict:
    """Verify environment_from_dict deserialisation."""

    def test_round_trip(self, simple_env: Environment) -> None:
        row = environment_to_dict(simple_env, env_name='add')
        recovered = environment_from_dict(row)
        assert recovered.task == simple_env.task
        assert recovered.verifier_code == simple_env.verifier_code
        assert recovered.test_cases is None

    def test_round_trip_with_test_cases(self, env_with_test_cases: Environment) -> None:
        row = environment_to_dict(env_with_test_cases)
        recovered = environment_from_dict(row)
        assert recovered.test_cases == env_with_test_cases.test_cases

    def test_invalid_test_cases_json_returns_none(self, simple_env: Environment) -> None:
        row = environment_to_dict(simple_env)
        row['test_cases'] = '{not valid json'
        recovered = environment_from_dict(row)
        assert recovered.test_cases is None

    def test_default_timeout_applied(self, simple_env: Environment) -> None:
        row = environment_to_dict(simple_env)
        del row['timeout']
        recovered = environment_from_dict(row)
        assert recovered.timeout == 30


class TestPushEnvironmentToHub:
    """Verify push_environment_to_hub delegates correctly to datasets."""

    def test_raises_import_error_when_datasets_missing(self, simple_env: Environment) -> None:
        with patch('builtins.__import__', side_effect=ImportError('no datasets')):
            with pytest.raises(ImportError, match='huggingface_hub and datasets'):
                push_environment_to_hub(simple_env, repo_id='org/repo')

    def test_calls_push_to_hub(self, simple_env: Environment) -> None:
        mock_dataset_cls = MagicMock()
        mock_dataset_instance = MagicMock()
        mock_dataset_cls.from_list.return_value = mock_dataset_instance
        mock_datasets_module = MagicMock()
        mock_datasets_module.Dataset = mock_dataset_cls

        with patch('deepgym.integrations.hf._require_datasets', return_value=mock_datasets_module):
            push_environment_to_hub(simple_env, repo_id='org/repo', env_name='add')

        mock_dataset_cls.from_list.assert_called_once()
        mock_dataset_instance.push_to_hub.assert_called_once_with(
            'org/repo', private=False, token=None
        )


class TestPushEnvironmentsToHub:
    """Verify push_environments_to_hub handles multiple environments."""

    def test_calls_push_with_all_rows(self, simple_env: Environment) -> None:
        mock_dataset_cls = MagicMock()
        mock_dataset_instance = MagicMock()
        mock_dataset_cls.from_list.return_value = mock_dataset_instance
        mock_datasets_module = MagicMock()
        mock_datasets_module.Dataset = mock_dataset_cls

        envs = {'add': simple_env, 'sub': simple_env}
        with patch('deepgym.integrations.hf._require_datasets', return_value=mock_datasets_module):
            push_environments_to_hub(envs, repo_id='org/repo')

        rows = mock_dataset_cls.from_list.call_args[0][0]
        assert len(rows) == 2
        assert {r['env_name'] for r in rows} == {'add', 'sub'}


class TestLoadEnvironmentFromHub:
    """Verify load_environment_from_hub deserialises correctly."""

    def _make_mock_datasets(self, rows: list[dict]) -> MagicMock:
        mock_datasets_module = MagicMock()
        mock_datasets_module.load_dataset.return_value = rows
        return mock_datasets_module

    def test_loads_first_row_when_no_name(self, simple_env: Environment) -> None:
        row = environment_to_dict(simple_env, env_name='add')
        mock_datasets = self._make_mock_datasets([row])

        with patch('deepgym.integrations.hf._require_datasets', return_value=mock_datasets):
            env = load_environment_from_hub('org/repo')

        assert env.task == simple_env.task

    def test_filters_by_env_name(self, simple_env: Environment) -> None:
        rows = [
            environment_to_dict(simple_env, env_name='add'),
            environment_to_dict(simple_env, env_name='sub'),
        ]
        mock_datasets = self._make_mock_datasets(rows)

        with patch('deepgym.integrations.hf._require_datasets', return_value=mock_datasets):
            env = load_environment_from_hub('org/repo', env_name='sub')

        assert env.task == simple_env.task

    def test_raises_value_error_when_name_not_found(self, simple_env: Environment) -> None:
        row = environment_to_dict(simple_env, env_name='add')
        mock_datasets = self._make_mock_datasets([row])

        with patch('deepgym.integrations.hf._require_datasets', return_value=mock_datasets):
            with pytest.raises(ValueError, match='missing'):
                load_environment_from_hub('org/repo', env_name='missing')


class TestLoadAllEnvironmentsFromHub:
    """Verify load_all_environments_from_hub returns all rows keyed by name."""

    def test_returns_all_envs_keyed_by_name(self, simple_env: Environment) -> None:
        rows = [
            environment_to_dict(simple_env, env_name='add'),
            environment_to_dict(simple_env, env_name='sub'),
        ]
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = rows

        with patch('deepgym.integrations.hf._require_datasets', return_value=mock_datasets):
            envs = load_all_environments_from_hub('org/repo')

        assert set(envs.keys()) == {'add', 'sub'}

    def test_unnamed_rows_keyed_by_index(self, simple_env: Environment) -> None:
        rows = [
            environment_to_dict(simple_env, env_name=''),
            environment_to_dict(simple_env, env_name=''),
        ]
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = rows

        with patch('deepgym.integrations.hf._require_datasets', return_value=mock_datasets):
            envs = load_all_environments_from_hub('org/repo')

        assert '0' in envs
        assert '1' in envs


class TestPushResultsToHub:
    """Verify push_results_to_hub formats rows correctly."""

    def test_pushes_one_row_per_env(self) -> None:
        mock_dataset_cls = MagicMock()
        mock_dataset_instance = MagicMock()
        mock_dataset_cls.from_list.return_value = mock_dataset_instance
        mock_datasets_module = MagicMock()
        mock_datasets_module.Dataset = mock_dataset_cls

        results = {'coin_change': 1.0, 'two_sum': 0.8}
        with patch('deepgym.integrations.hf._require_datasets', return_value=mock_datasets_module):
            push_results_to_hub(results, repo_id='org/lb', model_name='Qwen/Qwen2')

        rows = mock_dataset_cls.from_list.call_args[0][0]
        assert len(rows) == 2
        assert all(r['model_name'] == 'Qwen/Qwen2' for r in rows)
        env_names = {r['env_name'] for r in rows}
        assert env_names == {'coin_change', 'two_sum'}
