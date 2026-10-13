"""Tests for deepgym.integrations.hf — HuggingFace Hub integration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from deepgym.benchmark_envs import RepoPatchEnvironment
from deepgym.core import DeepGym
from deepgym.integrations.hf import (
    environment_from_dict,
    environment_to_dict,
    load_all_environments_from_hub,
    load_environment_from_hub,
    push_environment_to_hub,
    push_environments_to_hub,
    push_results_to_hub,
)
from deepgym.judge import JudgeJob
from deepgym.models import Environment

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'
REVERSE_DIR = ENVS_DIR / 'reverse_string'
ADD = 'def add(a, b):\n    return a + b\n'


@pytest.fixture()
def simple_env() -> Environment:
    """Return a minimal environment for serialisation tests."""
    return Environment(
        task='Write a function add(a, b) that returns a + b.',
        verifier_code='return 1.0 if hasattr(mod, "add") and mod.add(1, 2) == 3 else 0.0',
    )


@pytest.fixture()
def full_env() -> Environment:
    """An environment that sets every field a round trip has to keep."""
    return Environment(
        name='add',
        task='Write add(a, b).',
        type='tool-use',
        verifier_code=(
            'from deepgym_solution import load_solution\n'
            'from scoring import expected\n\n'
            'def verify(solution_path, test_cases_path=None):\n'
            '    return float(load_solution(solution_path).add(1, 2) == expected())\n'
        ),
        verifier_mode='verify_function',
        support_files={'scoring.py': 'def expected():\n    return 3\n'},
        timeout=7,
        case_timeout=2,
        difficulty='easy',
        domain='math',
        tags=['arithmetic'],
        test_cases=[{'a': 1, 'b': 2, 'expected': 3}],
        snapshot='deepgym-python',
        image='python:3.12-slim',
    )


@pytest.fixture()
def env_with_verifier_path() -> Environment:
    """Return an environment backed by a real verifier file."""
    return Environment(
        task=(REVERSE_DIR / 'task.md').read_text(encoding='utf-8'),
        verifier_path=REVERSE_DIR / 'verifier.py',
    )


def _crafted(env: Environment, **fields: object) -> dict[str, str]:
    """A schema 2.0 row for *env* with *fields* overwritten, as a hostile dataset could."""
    row = environment_to_dict(env)
    row['environment'] = json.dumps({**json.loads(row['environment']), **fields})
    return row


class TestRoundTrip:
    """A pushed row must load back as the environment that was pushed."""

    def test_every_field_survives(self, full_env: Environment) -> None:
        row = environment_to_dict(full_env, env_name='add')

        assert row['schema_version'] == '2.0'
        assert row['env_name'] == 'add'
        assert environment_from_dict(row, trust_image=True) == full_env

    def test_the_judge_gets_the_same_job(self, full_env: Environment) -> None:
        loaded = environment_from_dict(environment_to_dict(full_env), trust_image=True)

        assert JudgeJob.for_environment(loaded, ADD) == JudgeJob.for_environment(full_env, ADD)

    def test_the_loaded_environment_scores_the_same(
        self, full_env: Environment, local_dg: DeepGym
    ) -> None:
        """The helper module and verifier mode come back, so the verifier still runs."""
        loaded = environment_from_dict(environment_to_dict(full_env), trust_image=True)

        assert local_dg.run(full_env, ADD).score == local_dg.run(loaded, ADD).score == 1.0

    def test_nested_support_files_survive(self, simple_env: Environment) -> None:
        env = simple_env.model_copy(
            update={'support_files': {'lib/__init__.py': '', 'lib/scoring.py': 'X = 1\n'}}
        )

        assert environment_from_dict(environment_to_dict(env)) == env

    def test_verifier_path_is_inlined(self, env_with_verifier_path: Environment) -> None:
        loaded = environment_from_dict(environment_to_dict(env_with_verifier_path))

        assert loaded.verifier_path is None
        assert loaded.verifier_code == (REVERSE_DIR / 'verifier.py').read_text(encoding='utf-8')

    def test_subclasses_are_refused(self) -> None:
        repo_patch = RepoPatchEnvironment(task='t', repo_dir='/repo', test_command='pytest')

        with pytest.raises(TypeError, match='RepoPatchEnvironment'):
            environment_to_dict(repo_patch)

    @pytest.mark.parametrize(
        'fields',
        [
            {'env_vars': {'API_KEY': 'sk-123'}},
            {'support_files': {'./scoring.py': 'X = 1\n'}},
            {'verifier_mode': 'script'},
        ],
        ids=['env_vars', 'support_file', 'verifier_mode'],
    )
    def test_a_push_refuses_what_a_load_refuses(
        self, simple_env: Environment, fields: dict
    ) -> None:
        """Both directions share one check, so nothing pushes that can't load."""
        with pytest.raises(ValueError):
            environment_to_dict(simple_env.model_copy(update=fields))
        with pytest.raises(ValueError):
            environment_from_dict(_crafted(simple_env, **fields))


class TestHubRowsAreUntrusted:
    """A Hub dataset is untrusted input: a crafted row can't change how its env runs."""

    @pytest.mark.parametrize('fields', [{'timeout': '30'}, {'tags': 'math'}, {'case_timeout': 2.5}])
    def test_wrong_types_are_refused(self, simple_env: Environment, fields: dict) -> None:
        with pytest.raises(ValueError):
            environment_from_dict(_crafted(simple_env, **fields))

    def test_unknown_fields_are_refused(self, simple_env: Environment) -> None:
        """A newer deepgym's field would otherwise be dropped silently."""
        with pytest.raises(ValueError, match='sandbox_cpu'):
            environment_from_dict(_crafted(simple_env, sandbox_cpu=4))

    def test_a_verifier_path_is_refused(self, simple_env: Environment, tmp_path: Path) -> None:
        """A path would have the loading machine's file read and uploaded as the verifier."""
        local = tmp_path / 'secret.py'
        local.write_text('API_KEY = "sk-123"\n')

        with pytest.raises(ValueError, match='verifier_path'):
            environment_from_dict(_crafted(simple_env, verifier_path=str(local)))

    def test_env_vars_are_refused(self, simple_env: Environment) -> None:
        """They would reach the verifier and the solution, e.g. as LD_PRELOAD."""
        with pytest.raises(ValueError, match='env_vars'):
            environment_from_dict(_crafted(simple_env, env_vars={'LD_PRELOAD': '/tmp/x.so'}))

    @pytest.mark.parametrize('mode', ['script', 'verify_function'])
    def test_a_verifier_mode_the_verifier_does_not_read_as_is_refused(
        self, simple_env: Environment, mode: str
    ) -> None:
        with pytest.raises(ValueError, match='verifier_mode'):
            environment_from_dict(_crafted(simple_env, verifier_mode=mode))

    @pytest.mark.parametrize(
        'name',
        [
            '../escape.py',
            '/etc/cron.d/x',
            './scoring.py',
            'lib//scoring.py',
            'lib/./scoring.py',
            'lib/',
            'lib\\scoring.py',
            'verifier.py',
            'deepgym_solution.py',
        ],
    )
    def test_support_file_names_must_be_plain_relative_paths(
        self, simple_env: Environment, name: str
    ) -> None:
        with pytest.raises(ValueError):
            environment_from_dict(_crafted(simple_env, support_files={name: 'x = 1\n'}))

    @pytest.mark.parametrize('field', ['image', 'snapshot'])
    def test_an_image_needs_trust_image(self, simple_env: Environment, field: str) -> None:
        row = _crafted(simple_env, **{field: 'attacker/miner:latest'})

        with pytest.raises(ValueError, match='trust_image'):
            environment_from_dict(row)
        loaded = environment_from_dict(row, trust_image=True)
        assert getattr(loaded, field) == 'attacker/miner:latest'


class TestEnvironmentFromDict:
    """Verify environment_from_dict deserialisation."""

    def test_schema_1_rows_still_load(self) -> None:
        row = {
            'env_name': 'add',
            'task': 'Write add.',
            'verifier_code': 'return 1.0',
            'test_cases': json.dumps([{'a': 1}]),
            'timeout': 12,
            'schema_version': '1.0',
        }

        env = environment_from_dict(row)

        assert (env.task, env.verifier_code, env.test_cases, env.timeout) == (
            'Write add.',
            'return 1.0',
            [{'a': 1}],
            12,
        )

    def test_invalid_test_cases_json_raises(self) -> None:
        """Loading without its test cases would change what the verifier checks."""
        row = {'task': 't', 'verifier_code': 'return 1.0', 'test_cases': '{not valid json'}

        with pytest.raises(ValueError):
            environment_from_dict(row)

    def test_an_unknown_schema_version_raises(self, simple_env: Environment) -> None:
        row = {**environment_to_dict(simple_env), 'schema_version': '3.0'}

        with pytest.raises(ValueError, match='upgrade deepgym'):
            environment_from_dict(row)


def _datasets_module(rows: list[dict]) -> MagicMock:
    """A stand-in for the datasets module: load_dataset returns *rows*, push returns a commit."""
    module = MagicMock()
    module.load_dataset.return_value = rows
    module.Dataset.from_list.return_value.push_to_hub.return_value = SimpleNamespace(oid='abc123')
    return module


class TestPushEnvironmentToHub:
    """Verify push_environment_to_hub delegates correctly to datasets."""

    def test_raises_import_error_when_datasets_missing(self, simple_env: Environment) -> None:
        with patch('builtins.__import__', side_effect=ImportError('no datasets')):
            with pytest.raises(ImportError, match='huggingface_hub and datasets'):
                push_environment_to_hub(simple_env, repo_id='org/repo')

    def test_returns_the_commit_to_pin(self, simple_env: Environment) -> None:
        module = _datasets_module([])

        with (
            patch('deepgym.integrations.hf._require_datasets', return_value=module),
            patch('deepgym.integrations.hf._replace_stale_features', return_value=None),
        ):
            revision = push_environment_to_hub(simple_env, repo_id='org/repo', env_name='add')

        assert revision == 'abc123'
        module.Dataset.from_list.return_value.push_to_hub.assert_called_once_with(
            'org/repo', private=False, token=None
        )

    def test_pushes_one_row_per_environment(self, simple_env: Environment) -> None:
        module = _datasets_module([])

        with (
            patch('deepgym.integrations.hf._require_datasets', return_value=module),
            patch('deepgym.integrations.hf._replace_stale_features', return_value=None),
        ):
            push_environments_to_hub({'add': simple_env, 'sub': simple_env}, repo_id='org/repo')

        rows = module.Dataset.from_list.call_args[0][0]
        assert {r['env_name'] for r in rows} == {'add', 'sub'}


class TestPushOverAnExistingRepo:
    """push_to_hub keeps the features an earlier push wrote into the card for the config."""

    def _repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> list:
        """A local stand-in for a Hub repo whose card describes *rows*; returns its commits.

        push_to_hub writes the parquet, keeps the card and resolves the repo id to the
        user's namespace; hf_hub_download reads the card; create_commit writes it.
        """
        datasets = pytest.importorskip('datasets')
        import huggingface_hub
        from datasets.info import DatasetInfosDict
        from huggingface_hub import DatasetCard, HfApi

        earlier = datasets.Dataset.from_list(rows)
        card = DatasetCard('---\n---\n')
        info = DatasetInfosDict({'default': datasets.DatasetInfo(features=earlier.features)})
        info.to_dataset_card_data(card.data)
        card.data['configs'] = [
            {'config_name': 'default', 'data_files': [{'split': 'train', 'path': 'data/*'}]}
        ]
        (tmp_path / 'README.md').write_text(str(DatasetCard(f'---\n{card.data.to_yaml()}\n---\n')))
        (tmp_path / 'data').mkdir()
        commits: list = []

        def push_to_hub(dataset: object, repo_id: str, **kwargs: object) -> SimpleNamespace:
            dataset.to_parquet(tmp_path / 'data' / 'train-00000-of-00001.parquet')
            commits.append(('data', f'someone/{repo_id}'))
            return SimpleNamespace(
                oid='data-commit', repo_url=SimpleNamespace(repo_id=f'someone/{repo_id}')
            )

        def hf_hub_download(repo_id: str, filename: str, **kwargs: object) -> str:
            assert (repo_id, filename, kwargs['revision']) == (
                'someone/envs',
                'README.md',
                'data-commit',
            )
            return str(tmp_path / 'README.md')

        def create_commit(api: object, repo_id: str, operations: list, **kwargs: object) -> object:
            (tmp_path / 'README.md').write_bytes(operations[0].path_or_fileobj)
            commits.append(('card', repo_id, kwargs['parent_commit']))
            return SimpleNamespace(oid='card-commit')

        monkeypatch.setattr(datasets.Dataset, 'push_to_hub', push_to_hub)
        monkeypatch.setattr(huggingface_hub, 'hf_hub_download', hf_hub_download)
        monkeypatch.setattr(HfApi, 'create_commit', create_commit)
        return commits

    def test_a_push_over_schema_1_rows_rewrites_the_card(
        self, simple_env: Environment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fix lands in the repo the push resolved, on top of the push's own commit."""
        old_row = {
            'env_name': 'old',
            'task': 't',
            'verifier_code': 'return 1.0',
            'test_cases': '',
            'timeout': 30,
            'schema_version': '1.0',
        }
        commits = self._repo(tmp_path, monkeypatch, [old_row])

        revision = push_environment_to_hub(simple_env, repo_id='envs', env_name='add')

        assert revision == 'card-commit'
        assert commits == [('data', 'someone/envs'), ('card', 'someone/envs', 'data-commit')]
        assert load_environment_from_hub(str(tmp_path), revision=revision) == simple_env

    def test_a_push_over_current_rows_needs_no_card_commit(
        self, simple_env: Environment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        commits = self._repo(tmp_path, monkeypatch, [environment_to_dict(simple_env, 'add')])

        revision = push_environment_to_hub(simple_env, repo_id='envs', env_name='add')

        assert (revision, commits) == ('data-commit', [('data', 'someone/envs')])


class TestLoadEnvironmentFromHub:
    """Verify load_environment_from_hub deserialises correctly."""

    def test_loads_first_row_when_no_name(self, simple_env: Environment) -> None:
        module = _datasets_module([environment_to_dict(simple_env, env_name='add')])

        with patch('deepgym.integrations.hf._require_datasets', return_value=module):
            env = load_environment_from_hub('org/repo', revision='abc123')

        assert env == simple_env

    def test_filters_by_env_name(self, simple_env: Environment, full_env: Environment) -> None:
        rows = [
            environment_to_dict(simple_env, env_name='add'),
            environment_to_dict(full_env, env_name='full'),
        ]

        module = _datasets_module(rows)
        with patch('deepgym.integrations.hf._require_datasets', return_value=module):
            env = load_environment_from_hub(
                'org/repo', env_name='full', revision='abc123', trust_image=True
            )

        assert env == full_env

    def test_raises_value_error_when_name_not_found(self, simple_env: Environment) -> None:
        module = _datasets_module([environment_to_dict(simple_env, env_name='add')])

        with patch('deepgym.integrations.hf._require_datasets', return_value=module):
            with pytest.raises(ValueError, match='missing'):
                load_environment_from_hub('org/repo', env_name='missing', revision='abc123')

    @pytest.mark.parametrize('load', [load_environment_from_hub, load_all_environments_from_hub])
    def test_the_revision_is_pinned(self, simple_env: Environment, load: object) -> None:
        module = _datasets_module([environment_to_dict(simple_env, env_name='add')])

        with patch('deepgym.integrations.hf._require_datasets', return_value=module):
            load('org/repo', revision='abc123', token='t')  # type: ignore[operator]

        module.load_dataset.assert_called_once_with(
            'org/repo', split='train', revision='abc123', token='t'
        )

    def test_an_unpinned_load_warns(
        self, simple_env: Environment, caplog: pytest.LogCaptureFixture
    ) -> None:
        module = _datasets_module([environment_to_dict(simple_env, env_name='add')])

        with patch('deepgym.integrations.hf._require_datasets', return_value=module):
            with caplog.at_level(logging.WARNING, logger='deepgym.integrations.hf'):
                load_environment_from_hub('org/repo')

        assert 'changes with every push' in caplog.text


class TestLoadAllEnvironmentsFromHub:
    """Verify load_all_environments_from_hub returns all rows keyed by name."""

    def test_returns_all_envs_keyed_by_name(self, simple_env: Environment) -> None:
        rows = [
            environment_to_dict(simple_env, env_name='add'),
            environment_to_dict(simple_env, env_name='sub'),
        ]

        module = _datasets_module(rows)
        with patch('deepgym.integrations.hf._require_datasets', return_value=module):
            envs = load_all_environments_from_hub('org/repo', revision='abc123')

        assert set(envs.keys()) == {'add', 'sub'}

    def test_unnamed_rows_keyed_by_index(self, simple_env: Environment) -> None:
        rows = [environment_to_dict(simple_env), environment_to_dict(simple_env)]

        module = _datasets_module(rows)
        with patch('deepgym.integrations.hf._require_datasets', return_value=module):
            envs = load_all_environments_from_hub('org/repo', revision='abc123')

        assert set(envs) == {'0', '1'}


def test_rows_survive_a_real_parquet_round_trip(
    full_env: Environment, simple_env: Environment, tmp_path: Path
) -> None:
    """Write the rows as push_to_hub uploads them (parquet) and load them with datasets."""
    datasets = pytest.importorskip('datasets')
    rows = [environment_to_dict(full_env, 'full'), environment_to_dict(simple_env, 'simple')]
    (tmp_path / 'data').mkdir()
    datasets.Dataset.from_list(rows).to_parquet(tmp_path / 'data' / 'train-00000-of-00001.parquet')

    envs = load_all_environments_from_hub(str(tmp_path), trust_image=True)

    assert envs == {'full': full_env, 'simple': simple_env}


class TestPushResultsToHub:
    """Verify push_results_to_hub formats rows correctly."""

    def test_pushes_one_row_per_env(self) -> None:
        module = _datasets_module([])

        results = {'coin_change': 1.0, 'two_sum': 0.8}
        with patch('deepgym.integrations.hf._require_datasets', return_value=module):
            push_results_to_hub(results, repo_id='org/lb', model_name='Qwen/Qwen2')

        rows = module.Dataset.from_list.call_args[0][0]
        assert len(rows) == 2
        assert all(r['model_name'] == 'Qwen/Qwen2' for r in rows)
        assert {r['env_name'] for r in rows} == {'coin_change', 'two_sum'}
