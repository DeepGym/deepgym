"""Smoke test: every built-in coding environment passes with its reference solution."""

from pathlib import Path

import pytest

from deepgym.core import DeepGym
from deepgym.registry import list_environments, load_environment

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'

# Nested subdirectory mappings for locating reference solutions.
_NESTED_SUBDIRS = {
    'computer_use': ['file_organizer', 'cli_task'],
    'tool_use': ['api_request', 'data_pipeline'],
    'multi_turn': ['debug_fix'],
}


def _find_reference_solution(env_path: str) -> Path | None:
    """Locate the reference_solution.py for an environment by registry path.

    Args:
        env_path: The 'path' field from the registry entry.

    Returns:
        Path to reference_solution.py or None if not found.
    """
    # Top-level envs directory.
    candidate = ENVS_DIR / env_path / 'reference_solution.py'
    if candidate.exists():
        return candidate

    # Check nested subdirectories.
    for subdir, names in _NESTED_SUBDIRS.items():
        if env_path in names:
            nested = ENVS_DIR / subdir / env_path / 'reference_solution.py'
            if nested.exists():
                return nested

    return None


def _get_testable_envs() -> list[str]:
    """Return registry paths for environments that have reference solutions."""
    result = []
    for entry in list_environments():
        path = entry['path']
        if _find_reference_solution(path) is not None:
            result.append(path)
    return result


_TESTABLE_ENVS = _get_testable_envs()


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    """Module-scoped local DeepGym client."""
    return DeepGym(mode='local')


@pytest.mark.parametrize('env_name', _TESTABLE_ENVS)
def test_builtin_env_with_reference_solution(env_name: str, dg: DeepGym) -> None:
    """Every built-in environment should pass with its reference solution."""
    env = load_environment(env_name)
    solution_path = _find_reference_solution(env_name)
    assert solution_path is not None, f'No reference solution for {env_name}'

    solution_code = solution_path.read_text(encoding='utf-8')
    result = dg.run(env, solution_code)
    assert result.passed is True, (
        f'{env_name}: expected passed=True, got score={result.score}. Output: {result.output!r}'
    )
