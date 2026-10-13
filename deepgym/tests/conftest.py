"""Shared fixtures for the DeepGym test suite."""

import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pytest

from deepgym import solution_runtime
from deepgym.core import DeepGym
from deepgym.models import Environment
from deepgym.sandboxes import ExecResult

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


@pytest.fixture(autouse=True)
def fresh_solution_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test as a fresh verifier: no solution code run yet, no preflight done,
    no watcher started."""
    monkeypatch.setattr(solution_runtime, '_consulted', False)
    monkeypatch.setattr(solution_runtime, '_preflight_passed', False)
    monkeypatch.setattr(solution_runtime, '_watcher', None)


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


# ---------------------------------------------------------------------------
# Fake sandbox for benchmark runners
# ---------------------------------------------------------------------------


class RunnerSandbox:
    """Fake provider for benchmark runners (SWE-bench Pro, Terminal-Bench, CyberGym, MiMo).

    Those runners refuse local mode, so tests reach them through the sandbox code
    path. Uploads land in a temp dir and the runner runs from there, under *python* as
    the image's ``python3``; only the tests' own fixture repos and patches ever run this
    way. Like a real provider, it runs the command with sh and returns stdout and stderr
    as one stream.
    """

    network_blocked = True

    def __init__(self, python: str = sys.executable) -> None:
        self.root = Path(tempfile.mkdtemp(prefix='deepgym_fake_sandbox_'))
        self.id = f'fake-{self.root.name}'
        self.python = python

    def _path(self, sandbox_path: str) -> Path:
        return self.root / sandbox_path.lstrip('/')

    def upload(self, content: bytes, path: str) -> None:
        self._path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path(path).write_bytes(content)

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> ExecResult:
        assert 'python3 /home/user/' in command  # python3 <runner> <task> [patch] [archive]
        command = command.replace('/home/user/', f'{self.root}/home/user/')
        command = command.replace('python3 ', f'{shlex.quote(self.python)} ', 1)
        proc = subprocess.run(
            ['sh', '-c', command], capture_output=True, text=True, timeout=timeout
        )
        return ExecResult(proc.returncode, proc.stdout + proc.stderr)

    def delete(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture()
def sandbox_dg(monkeypatch: pytest.MonkeyPatch) -> DeepGym:
    """A sandbox-mode client whose benchmark runners run in a RunnerSandbox."""
    monkeypatch.setattr('deepgym.core.check_provider', lambda provider: None)
    monkeypatch.setattr('deepgym.core.open_sandbox', lambda provider, image: RunnerSandbox())
    return DeepGym(mode='docker')
