"""Tests for benchmark-backed environments and custom execution routing."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from deepgym.benchmark_envs import (
    MixedEnvironment,
    PatchVerifier,
    SWEBenchProEnvironment,
    TerminalBenchEnvironment,
    _split_batch_kwargs,
)
from deepgym.integrations.trl import make_trl_reward_fn
from deepgym.models import Environment
from deepgym.registry import load_environment


def _git(repo_dir: Path, *args: str) -> str:
    return subprocess.check_output(
        ['git', '-C', str(repo_dir), *args],
        text=True,
    ).strip()


@pytest.fixture()
def swebench_repo(tmp_path: Path) -> dict[str, object]:
    repo_dir = tmp_path / 'swebench-fixture'
    repo_dir.mkdir()
    (repo_dir / 'tests').mkdir()

    _git_init = [
        ['git', '-C', str(repo_dir), 'init'],
        ['git', '-C', str(repo_dir), 'config', 'user.email', 'tests@example.com'],
        ['git', '-C', str(repo_dir), 'config', 'user.name', 'DeepGym Tests'],
    ]
    for command in _git_init:
        subprocess.run(command, check=True, capture_output=True, text=True)

    (repo_dir / 'calculator.py').write_text(
        'def add(a, b):\n'
        '    return a - b\n',
        encoding='utf-8',
    )
    (repo_dir / 'tests' / 'test_calc.py').write_text(
        'from calculator import add\n\n'
        'def test_add():\n'
        '    assert add(1, 2) == 3\n',
        encoding='utf-8',
    )
    subprocess.run(
        ['git', '-C', str(repo_dir), 'add', '.'],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ['git', '-C', str(repo_dir), 'commit', '-m', 'base'],
        check=True,
        capture_output=True,
        text=True,
    )
    base_commit = _git(repo_dir, 'rev-parse', 'HEAD')

    (repo_dir / 'calculator.py').write_text(
        'def add(a, b):\n'
        '    return a + b\n',
        encoding='utf-8',
    )
    patch_text = _git(repo_dir, 'diff', 'HEAD')
    subprocess.run(
        ['git', '-C', str(repo_dir), 'checkout', '--', 'calculator.py'],
        check=True,
        capture_output=True,
        text=True,
    )

    return {
        'repo_dir': repo_dir,
        'base_commit': base_commit,
        'patch_text': patch_text,
        'row': {
            'instance_id': 'fixture-1',
            'repo': 'fixture/repo',
            'repo_source': str(repo_dir),
            'base_commit': base_commit,
            'problem_statement': 'Fix add() so it returns the correct sum.',
            'test_patch': '',
            'fail_to_pass': ['tests/test_calc.py::test_add'],
            'pass_to_pass': [],
            'selected_test_files_to_run': ['tests/test_calc.py'],
            'test_commands': ['PYTHONPATH=. pytest -q tests/test_calc.py'],
        },
    }


@pytest.fixture()
def terminal_task_root(tmp_path: Path) -> Path:
    task_root = tmp_path / 'terminal-bench'
    task_dir = task_root / 'echo-task'
    environment_dir = task_dir / 'environment'
    tests_dir = task_dir / 'tests'
    environment_dir.mkdir(parents=True)
    tests_dir.mkdir()

    (task_dir / 'instruction.md').write_text(
        'Write a terminal command that creates output.txt with the value "done".\n',
        encoding='utf-8',
    )
    (task_dir / 'task.toml').write_text(
        '[verifier]\n'
        'timeout_sec = 60\n',
        encoding='utf-8',
    )
    (tests_dir / 'test.sh').write_text(
        '#!/bin/sh\n'
        'mkdir -p logs/verifier\n'
        'if [ "$(cat environment/output.txt 2>/dev/null)" = "done" ]; then\n'
        '  echo 1 > logs/verifier/reward.txt\n'
        '  exit 0\n'
        'fi\n'
        'echo 0 > logs/verifier/reward.txt\n'
        'exit 1\n',
        encoding='utf-8',
    )
    subprocess.run(
        ['chmod', '+x', str(tests_dir / 'test.sh')],
        check=True,
        capture_output=True,
        text=True,
    )
    return task_root


class TestPatchVerifier:
    """Verify patch extraction and local repo scoring."""

    def test_extracts_fenced_diff(self, swebench_repo: dict[str, object]) -> None:
        patch_text = swebench_repo['patch_text']
        extracted = PatchVerifier.extract_patch(f'```diff\n{patch_text}\n```')
        assert extracted.startswith('diff --git')

    def test_scores_local_patch_repo(
        self,
        local_dg,
        swebench_repo: dict[str, object],
    ) -> None:
        env = SWEBenchProEnvironment()
        result = local_dg.run(env, swebench_repo['patch_text'], task=swebench_repo['row'])
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert result.reward_components == {'apply': 0.3, 'tests': 0.7}


class TestSWEBenchEnvironment:
    """Verify SWE-bench Pro loading and TRL integration."""

    def test_registry_load_returns_special_env(self) -> None:
        env = load_environment('swebench_pro')
        assert isinstance(env, SWEBenchProEnvironment)

    def test_trl_reward_fn_uses_raw_dataset_columns(
        self,
        local_dg,
        swebench_repo: dict[str, object],
    ) -> None:
        env = SWEBenchProEnvironment()
        reward_fn = make_trl_reward_fn(env=env, dg=local_dg)
        row = swebench_repo['row']
        scores = reward_fn(
            completions=[f'```diff\n{swebench_repo["patch_text"]}\n```'],
            repo=[row['repo']],
            repo_source=[row['repo_source']],
            base_commit=[row['base_commit']],
            test_patch=[row['test_patch']],
            fail_to_pass=[row['fail_to_pass']],
            pass_to_pass=[row['pass_to_pass']],
            selected_test_files_to_run=[row['selected_test_files_to_run']],
            test_commands=[row['test_commands']],
        )
        assert scores == pytest.approx([1.0])


class TestTerminalBenchEnvironment:
    """Verify terminal task execution in local mode."""

    def test_registry_load_returns_terminal_env(self) -> None:
        env = load_environment('terminal_bench_2')
        assert isinstance(env, TerminalBenchEnvironment)

    def test_runs_local_terminal_task(
        self,
        local_dg,
        terminal_task_root: Path,
    ) -> None:
        env = TerminalBenchEnvironment(task_root=terminal_task_root, default_task_id='echo-task')
        result = local_dg.run(env, 'printf done > output.txt\n', task_id='echo-task')
        assert result.score == pytest.approx(1.0)
        assert result.passed is True


class TestMixedEnvironment:
    """Verify mixed routing and sampling across heterogeneous envs."""

    def test_sample_batch_returns_child_envs(
        self,
        sorting_env: Environment,
        terminal_task_root: Path,
    ) -> None:
        named_sorting = sorting_env.model_copy(update={'name': 'python_sorting'})
        terminal_env = TerminalBenchEnvironment(
            task_root=terminal_task_root,
            default_task_id='echo-task',
        )
        mixed = MixedEnvironment(environments=[(named_sorting, 0.4), (terminal_env, 0.6)], seed=7)
        sampled = mixed.sample_batch(5)
        assert len(sampled) == 5
        assert all(env.name in {'python_sorting', 'terminal_bench_2'} for env in sampled)

    def test_split_batch_kwargs_rejects_mismatched_sequence_lengths(self) -> None:
        with pytest.raises(ValueError, match='batch size is 2'):
            _split_batch_kwargs(2, {'task_type': ['coding']})

    def test_ambiguous_task_type_route_raises(
        self,
        sorting_env: Environment,
        string_env: Environment,
    ) -> None:
        named_sorting = sorting_env.model_copy(update={'name': 'python_sorting'})
        named_string = string_env.model_copy(update={'name': 'string_manipulation'})
        mixed = MixedEnvironment(environments=[(named_sorting, 0.5), (named_string, 0.5)], seed=3)

        with pytest.raises(ValueError, match='could not route'):
            mixed.prepare_batch_requests(['def solve():\n    pass\n'], task_type=['coding'])

    def test_explicit_environment_object_route_wins(
        self,
        sorting_env: Environment,
        sorting_solution: str,
        local_dg,
    ) -> None:
        named_sorting = sorting_env.model_copy(update={'name': 'python_sorting'})
        mixed = MixedEnvironment(environments=[(named_sorting, 1.0)], seed=5)

        reward_fn = make_trl_reward_fn(env=mixed, dg=local_dg)
        scores = reward_fn(completions=[sorting_solution], environment=[named_sorting])
        assert scores[0] >= 0.9

    def test_trl_reward_routes_to_matching_child_env(
        self,
        local_dg,
        sorting_env: Environment,
        sorting_solution: str,
        terminal_task_root: Path,
    ) -> None:
        named_sorting = sorting_env.model_copy(update={'name': 'python_sorting'})
        terminal_env = TerminalBenchEnvironment(
            task_root=terminal_task_root,
            default_task_id='echo-task',
        )
        mixed = MixedEnvironment(environments=[(named_sorting, 0.5), (terminal_env, 0.5)], seed=11)
        reward_fn = make_trl_reward_fn(env=mixed, dg=local_dg)

        scores = reward_fn(
            completions=[sorting_solution, 'printf done > output.txt\n'],
            environment_name=['python_sorting', 'terminal_bench_2'],
            task_id=[None, 'echo-task'],
        )

        assert scores[0] >= 0.9
        assert scores[1] == pytest.approx(1.0)
