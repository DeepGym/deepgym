"""Shared fixtures for the DeepGym test suite."""

from pathlib import Path

import pytest

from deepgym.core import DeepGym
from deepgym.models import Environment

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'

SORTING_DIR = ENVS_DIR / 'python_sorting'
TWO_SUM_DIR = ENVS_DIR / 'two_sum'
STRING_DIR = ENVS_DIR / 'string_manipulation'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_dg() -> DeepGym:
    """Return a DeepGym client in local mode."""
    return DeepGym(mode='local')


@pytest.fixture()
def sorting_env() -> Environment:
    """Return the python_sorting example environment."""
    return Environment(
        task=(SORTING_DIR / 'task.md').read_text(encoding='utf-8'),
        verifier_path=SORTING_DIR / 'verifier.py',
        difficulty='easy',
        domain='coding',
        tags=['sorting'],
    )


@pytest.fixture()
def sorting_solution() -> str:
    """Return the reference solution for the python_sorting example."""
    return (SORTING_DIR / 'reference_solution.py').read_text(encoding='utf-8')


@pytest.fixture()
def two_sum_env() -> Environment:
    """Return the two_sum example environment."""
    return Environment(
        task=(TWO_SUM_DIR / 'task.md').read_text(encoding='utf-8'),
        verifier_path=TWO_SUM_DIR / 'verifier.py',
        difficulty='medium',
        domain='coding',
        tags=['arrays', 'hash-map'],
    )


@pytest.fixture()
def two_sum_solution() -> str:
    """Return the reference solution for the two_sum example."""
    return (TWO_SUM_DIR / 'reference_solution.py').read_text(encoding='utf-8')


@pytest.fixture()
def string_env() -> Environment:
    """Return the string_manipulation example environment."""
    return Environment(
        task=(STRING_DIR / 'task.md').read_text(encoding='utf-8'),
        verifier_path=STRING_DIR / 'verifier.py',
        difficulty='easy',
        domain='coding',
        tags=['strings'],
    )


@pytest.fixture()
def string_solution() -> str:
    """Return the reference solution for the string_manipulation example."""
    return (STRING_DIR / 'reference_solution.py').read_text(encoding='utf-8')


@pytest.fixture()
def sample_verifier_code() -> str:
    """Return a simple verifier body that checks for a 'solve' function."""
    return (
        'import importlib.util\n'
        'spec = importlib.util.spec_from_file_location("solution", solution_path)\n'
        'mod = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(mod)\n'
        'if hasattr(mod, "solve") and mod.solve(2) == 4:\n'
        '    return 1.0\n'
        'return 0.0\n'
    )


@pytest.fixture()
def sample_solution_code() -> str:
    """Return a trivial solution that defines solve(x) -> x*2."""
    return 'def solve(x):\n    return x * 2\n'
