"""Tests for deepgym.registry -- environment registry loading."""

import json
from pathlib import Path

import pytest

from deepgym import registry as registry_module
from deepgym.models import Environment, MultiTurnEnvironment
from deepgym.registry import (
    _read_registry,
    list_environments,
    load_environment,
    load_suite,
)


class TestListEnvironments:
    """Verify the registry lists all built-in environments."""

    def test_returns_list(self) -> None:
        envs = list_environments()
        assert isinstance(envs, list)

    def test_at_least_24_environments(self) -> None:
        envs = list_environments()
        assert len(envs) >= 24

    def test_each_entry_has_required_keys(self) -> None:
        envs = list_environments()
        for entry in envs:
            assert 'name' in entry
            assert 'path' in entry
            assert 'difficulty' in entry


class TestLoadEnvironment:
    """Verify loading individual environments by name."""

    def test_load_coin_change(self) -> None:
        env = load_environment('coin_change')
        assert isinstance(env, Environment)
        assert env.task  # non-empty task
        assert env.verifier_path is not None
        assert env.verifier_path.exists()

    def test_load_file_organizer_is_coding(self) -> None:
        env = load_environment('file_organizer')
        assert isinstance(env, Environment)
        assert env.type == 'coding'

    def test_load_nonexistent_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match='not found'):
            load_environment('nonexistent_env_that_does_not_exist')

    def test_load_reverse_string(self) -> None:
        env = load_environment('reverse_string')
        assert isinstance(env, Environment)
        assert env.difficulty == 'easy'

    def test_load_by_path_returns_valid_task(self) -> None:
        env = load_environment('binary_search')
        assert 'binary' in env.task.lower() or len(env.task) > 10

    def test_load_humaneval_benchmark_alias_from_nested_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / 'environments'
        env_dir = cache_dir / 'humaneval' / 'HumanEval_0'
        env_dir.mkdir(parents=True)
        (env_dir / 'task.md').write_text('Write solve().\n', encoding='utf-8')
        (env_dir / 'verifier.py').write_text('return 1.0\n', encoding='utf-8')
        (env_dir / 'metadata.json').write_text(
            '{"id":"HumanEval_0","benchmark":"humaneval","difficulty":"medium"}\n',
            encoding='utf-8',
        )
        (cache_dir / 'humaneval' / 'registry.json').write_text(
            (
                '[{"id":"HumanEval_0","name":"solve","benchmark":"humaneval",'
                '"difficulty":"medium","path":"environments/humaneval/HumanEval_0"}]\n'
            ),
            encoding='utf-8',
        )

        monkeypatch.setattr(registry_module, '_get_cache_dir', lambda: cache_dir)

        env = load_environment('humaneval')
        assert isinstance(env, Environment)
        assert env.name == 'HumanEval_0'

    def test_metadata_image_reaches_the_environment(self, tmp_path: Path) -> None:
        (tmp_path / 'task.md').write_text('Write solve().\n', encoding='utf-8')
        (tmp_path / 'verifier.py').write_text('return 1.0\n', encoding='utf-8')
        (tmp_path / 'metadata.json').write_text('{"image": "python:3.11-slim"}', encoding='utf-8')

        env = registry_module._load_env_from_dir(tmp_path)

        assert env.image == 'python:3.11-slim'

    def test_multi_turn_metadata_declares_tools_and_their_settling(self, tmp_path: Path) -> None:
        tools = [
            {
                'name': 'tracker',
                'description': 'Bug tracker REST API.',
                'base_url': 'http://127.0.0.1:8080',
                'routes': [{'method': 'POST', 'path': '/rest/bug/{id}/comment'}],
                'headers': {'X-API-KEY': 'app-key'},
            }
        ]
        metadata = {
            'type': 'multi-turn',
            'allowed_actions': ['tool', 'done'],
            'tools': tools,
            'root_tool_settle': 'service tracker stop',
        }
        (tmp_path / 'task.md').write_text('Triage bug 1.\n', encoding='utf-8')
        (tmp_path / 'verifier.py').write_text('print(1.0)\n', encoding='utf-8')
        (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')

        env = registry_module._load_env_from_dir(tmp_path)

        assert isinstance(env, MultiTurnEnvironment)
        assert env.allowed_actions == {'tool', 'done'}
        assert env.root_tool_settle == 'service tracker stop'
        (declared,) = env.tools
        assert declared.name == 'tracker'
        assert declared.headers['X-API-KEY'].get_secret_value() == 'app-key'
        assert 'metadata.json' not in env.files  # the key never reaches the work directory

    def test_load_humaneval_task_id_with_slash_alias(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / 'environments'
        env_dir = cache_dir / 'humaneval' / 'HumanEval_0'
        env_dir.mkdir(parents=True)
        (env_dir / 'task.md').write_text('Write solve().\n', encoding='utf-8')
        (env_dir / 'verifier.py').write_text('return 1.0\n', encoding='utf-8')
        (cache_dir / 'humaneval' / 'registry.json').write_text(
            (
                '[{"id":"HumanEval_0","benchmark":"humaneval",'
                '"path":"environments/humaneval/HumanEval_0"}]\n'
            ),
            encoding='utf-8',
        )

        monkeypatch.setattr(registry_module, '_get_cache_dir', lambda: cache_dir)

        env = load_environment('HumanEval/0')
        assert env.name == 'HumanEval_0'


class TestLoadSuite:
    """Verify suite loading by difficulty, type, and family."""

    def test_load_easy_suite(self) -> None:
        envs = load_suite('easy')
        assert len(envs) > 0
        for env in envs:
            assert env.difficulty == 'easy'

    def test_load_medium_suite(self) -> None:
        envs = load_suite('medium')
        assert len(envs) > 0
        for env in envs:
            assert env.difficulty == 'medium'

    def test_load_coding_suite(self) -> None:
        envs = load_suite('coding')
        assert len(envs) > 0
        for env in envs:
            assert env.type == 'coding'

    def test_load_computer_use_suite(self) -> None:
        envs = load_suite('computer-use')
        assert len(envs) == 5
        assert all(env.type == 'computer-use' for env in envs)

    def test_load_tool_use_suite(self) -> None:
        envs = load_suite('tool-use')
        assert len(envs) == 2

    def test_load_all_suite(self) -> None:
        all_envs = load_suite('all')
        registry = list_environments()
        # 'all' should return at most as many envs as the registry has entries.
        assert len(all_envs) <= len(registry)
        assert len(all_envs) >= 24

    def test_load_invalid_suite_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match='No environments match'):
            load_suite('totally_invalid_suite_name_xyz')


class TestReadRegistry:
    """Verify internal registry reading."""

    def test_returns_list_of_dicts(self) -> None:
        data = _read_registry()
        assert isinstance(data, list)
        assert all(isinstance(e, dict) for e in data)


class TestBenchmarkRouting:
    """humaneval and humaneval_plus share task ids; the requested benchmark must win."""

    @pytest.fixture()
    def caches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        cache_dir = tmp_path / 'environments'
        for benchmark in ('humaneval', 'humaneval_plus'):
            env_dir = cache_dir / benchmark / 'HumanEval_0'
            env_dir.mkdir(parents=True)
            (env_dir / 'task.md').write_text(f'{benchmark} task\n', encoding='utf-8')
            (env_dir / 'verifier.py').write_text('return 1.0\n', encoding='utf-8')
            (cache_dir / benchmark / 'registry.json').write_text(
                f'[{{"id":"HumanEval_0","name":"solve","benchmark":"{benchmark}",'
                f'"path":"environments/{benchmark}/HumanEval_0"}}]\n',
                encoding='utf-8',
            )
        monkeypatch.setattr(registry_module, '_get_cache_dir', lambda: cache_dir)
        return cache_dir

    @pytest.mark.parametrize(
        ('name', 'benchmark'),
        [
            ('humaneval_plus', 'humaneval_plus'),
            ('humaneval_plus/HumanEval_0', 'humaneval_plus'),
            ('humaneval_plus/HumanEval/0', 'humaneval_plus'),
            ('environments/humaneval_plus/HumanEval_0', 'humaneval_plus'),
            ('humaneval', 'humaneval'),
            ('humaneval/HumanEval_0', 'humaneval'),
            ('humaneval/HumanEval/0', 'humaneval'),
            ('environments/humaneval/HumanEval_0', 'humaneval'),
        ],
    )
    def test_requested_benchmark_wins(self, caches: Path, name: str, benchmark: str) -> None:
        assert load_environment(name).task == f'{benchmark} task\n'

    @pytest.mark.parametrize('name', ['HumanEval_0', 'HumanEval/0', 'solve'])
    def test_ambiguous_name_lists_candidates(self, caches: Path, name: str) -> None:
        with pytest.raises(ValueError, match='ambiguous') as exc:
            load_environment(name)
        assert 'humaneval_plus/HumanEval_0' in str(exc.value)

    @pytest.mark.parametrize('name', ['mbpp/HumanEval_0', 'humaneval_pls/HumanEval_0'])
    def test_unknown_namespace_does_not_fall_back_to_the_basename(
        self, caches: Path, name: str
    ) -> None:
        with pytest.raises(ValueError, match='not found'):
            load_environment(name)


@pytest.mark.parametrize('name', ['nonexistent_benchmark/coin_change', 'cyber/coin_change'])
def test_qualified_name_never_loads_a_builtin_by_basename(name: str) -> None:
    with pytest.raises(ValueError, match='not found'):
        load_environment(name)


def test_nested_builtin_still_loads_by_its_path() -> None:
    assert load_environment('tool_use/data_pipeline').name == 'data_pipeline'
