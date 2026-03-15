"""Tests for deepgym.registry -- environment registry loading."""

import pytest

from deepgym.models import Environment
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

    def test_load_file_organizer_is_computer_use(self) -> None:
        env = load_environment('file_organizer')
        assert isinstance(env, Environment)
        assert env.type == 'computer-use'

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
        assert len(envs) == 2

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
