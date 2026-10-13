"""Tests for deepgym.qa.mutants: AST mutants of the reference expose weak verifiers."""

import ast
import os
import signal
import time
from pathlib import Path

import pytest

from deepgym.core import DeepGym
from deepgym.exceptions import SandboxError
from deepgym.factory.scaffold import scaffold
from deepgym.judge import local_isolation_kind
from deepgym.models import Environment
from deepgym.qa import QAOptions, run_qa
from deepgym.qa.mutants import MAX_MUTANTS, mutants, mutation_check

needs_sandbox = pytest.mark.skipif(
    local_isolation_kind() == 'none', reason='mutants only run with sandbox-exec or bwrap'
)


@pytest.fixture
def mutants_run_here(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let mutation_check use this machine's sandbox even on macOS.

    It refuses macOS for want of process teardown; these tests' references start no
    processes, so the confined path still runs (and stays covered) under sandbox-exec.
    """
    monkeypatch.setattr('deepgym.qa.mutants.local_isolation_kind', lambda: 'bwrap')


_REFERENCE = '''\
"""Module docstring."""


def longest_run(values: list[int]) -> int:
    """Return the length of the longest strictly increasing run of neighbors."""
    if not values:
        return 0
    best = current = 1
    for previous, value in zip(values, values[1:]):
        if value > previous:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def sign(x) -> bool:
    return x >= 0 if x else False


if __name__ == '__main__':
    print(longest_run([1, 2]))
'''

_WEAK_VERIFIER = '''\
from deepgym_solution import load_solution

def verify(solution_path, test_cases_path=None):
    """Checks one example only."""
    solution = load_solution(solution_path)
    got = solution.longest_run([1, 2, 3])
    return {'score': float(got == 3), 'passed': got == 3, 'details': f'got {got!r}'}
'''


def _descriptions(reference: str) -> list[str]:
    return [mutant.description for mutant in mutants(reference)]


def test_mutants_cover_every_operator_and_compile() -> None:
    found = mutants(_REFERENCE)
    described = '\n'.join(mutant.description for mutant in found)

    assert 'line 7: off-by-one 0 -> 1' in described
    assert 'line 10: > -> >=' in described
    assert 'line 15: return best -> 0' in described
    assert 'line 10: swapped the branches of `value > previous`' in described
    assert 'line 6: negated `if not values`' in described
    assert 'line 19: swapped the branches of `x`' in described  # a if c else b
    assert 'line 14: dropped `current = 1`' in described
    assert 'line 19: return x >= 0 if x else False -> True' in described
    original = ast.unparse(ast.parse(_REFERENCE))
    sources = [mutant.source for mutant in found]
    assert len(set(sources)) == len(sources) and original not in sources
    for source in sources:
        compile(source, '<mutant>', 'exec')


def test_mutants_leave_docstrings_imports_defs_and_the_main_guard_alone() -> None:
    described = '\n'.join(_descriptions(_REFERENCE + 'import os\n'))

    assert 'docstring' not in described.lower() and 'dropped `"""' not in described
    assert 'dropped `def' not in described and 'dropped `import' not in described
    assert '__main__' not in described


@pytest.mark.parametrize(
    ('reference', 'replacement'),
    [
        ('def f(x) -> list[int]:\n    return sorted(x)\n', '-> []'),
        ('def f(x) -> str:\n    return x.strip()\n', "-> ''"),
        ('def f(x):\n    return {k: 1 for k in x}\n', '-> {}'),
        ('def f(x):\n    return len(x)\n', '-> 0'),
        ('def f(x):\n    return x.strip()\n', '-> None'),
    ],
)
def test_constant_returns_keep_the_kind_of_value(reference: str, replacement: str) -> None:
    assert any(d.endswith(replacement) for d in _descriptions(reference)), _descriptions(reference)


def test_non_python_references_give_no_mutants_and_long_ones_are_sampled() -> None:
    long_reference = ''.join(f'X{i} = {i}\n' for i in range(200))

    assert mutants('not python (') == []
    assert mutants('{"path": "/x", "depth": 2}') == []  # JSON data parses as Python
    assert len(mutants(long_reference)) == MAX_MUTANTS


def test_lines_marked_no_mutate_are_left_alone() -> None:
    reference = (
        'def f(n):\n    if n <= 2:  # pragma: no mutate\n        return n\n    return n * 2\n'
    )

    assert not [d for d in _descriptions(reference) if d.startswith('line 2:')]
    assert [d for d in _descriptions(reference) if d.startswith('line 3:')]


def _function_env(tmp_path: Path, verifier: str | None = None) -> tuple[Environment, str]:
    """The scaffolded function env (its strong verifier, or *verifier*) and its reference."""
    path = scaffold('env', 'function', tmp_path)
    if verifier is not None:
        (path / 'verifier.py').write_text(verifier, encoding='utf-8')
    reference = (path / 'reference_solution.py').read_text(encoding='utf-8')
    support = {'variants.py': (path / 'variants.py').read_text(encoding='utf-8')}
    env = Environment(task='t', verifier_path=path / 'verifier.py', support_files=support)
    return env, reference


@needs_sandbox
@pytest.mark.usefixtures('mutants_run_here')
def test_a_weak_verifier_fails_the_mutation_check(tmp_path: Path, local_dg: DeepGym) -> None:
    env, reference = _function_env(tmp_path, _WEAK_VERIFIER)

    check = mutation_check(env, reference, local_dg)

    assert check.status == 'fail'
    assert 'survived: ' in check.details
    assert 'line 6: dropped `if not values:`' in check.details  # [1, 2, 3] never tests []


@needs_sandbox
@pytest.mark.usefixtures('mutants_run_here')
def test_a_strong_verifier_kills_every_mutant(tmp_path: Path, local_dg: DeepGym) -> None:
    env, reference = _function_env(tmp_path)

    check = mutation_check(env, reference, local_dg)

    count = len(mutants(reference))
    assert (check.status, check.details) == (
        'pass',
        f'mutation score {count}/{count} (100%) of {count} mutants',
    )


@needs_sandbox
@pytest.mark.usefixtures('mutants_run_here')
def test_mutants_the_verifier_errors_on_are_inconclusive(local_dg: DeepGym) -> None:
    # It crashes before running a mutant; a crash after running one would score it 0.
    verifier = (
        'from deepgym_solution import load_solution, solution_source\n\n'
        'def verify(solution_path, test_cases_path=None):\n'
        "    if 'x + 1' not in solution_source(solution_path):\n"
        "        raise ValueError('a verifier that crashes instead of scoring')\n"
        '    return float(load_solution(solution_path).f(1) == 2)\n'
    )
    env = Environment(task='t', verifier_code=verifier)

    check = mutation_check(env, 'def f(x):\n    return x + 1\n', local_dg)

    assert check.status == 'inconclusive'
    assert check.details == 'mutation score 0/0 of 4 mutants; 4 got no verdict (verifier error)'


@needs_sandbox
@pytest.mark.usefixtures('mutants_run_here')
def test_a_mutant_without_a_verdict_never_counts_as_surviving(local_dg: DeepGym) -> None:
    verifier = (
        'from deepgym_solution import load_solution\n\n'
        'def verify(solution_path, test_cases_path=None):\n'
        '    if load_solution(solution_path).f(1) != 2:  # full marks, but its own error\n'
        "        return {'score': 1.0, 'passed': True, 'error_type': 'runtime_error'}\n"
        '    return 1.0\n'
    )
    env = Environment(task='t', verifier_code=verifier)

    check = mutation_check(env, 'def f(x):\n    return x + 1\n', local_dg)

    assert check.status == 'inconclusive'
    assert check.details == 'mutation score 0/0 of 4 mutants; 4 got no verdict (verifier error)'


@needs_sandbox
@pytest.mark.usefixtures('mutants_run_here')
def test_a_verifier_that_reads_formatting_cannot_be_judged(local_dg: DeepGym) -> None:
    verifier = (
        'from deepgym_solution import solution_source\n\n'
        'def verify(solution_path, test_cases_path=None):\n'
        "    return 1.0 if '# fast' in solution_source(solution_path) else 0.0\n"
    )
    env = Environment(task='t', verifier_code=verifier)

    check = mutation_check(env, 'def f(x):  # fast\n    return x + 1\n', local_dg)

    assert check.status == 'inconclusive'
    assert 'depends on formatting' in check.details


@pytest.mark.skipif(local_isolation_kind() != 'macos', reason='sandbox-exec only')
def test_macos_runs_no_local_mutants(tmp_path: Path, local_dg: DeepGym) -> None:
    env, reference = _function_env(tmp_path)
    local_dg.run_batch = lambda *_: pytest.fail('mutants ran')  # type: ignore[method-assign]

    check = mutation_check(env, reference, local_dg)

    assert check.status == 'inconclusive'
    assert "macOS's sandbox-exec can't take down processes" in check.details


def test_mutants_never_run_without_local_isolation(
    tmp_path: Path, local_dg: DeepGym, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, reference = _function_env(tmp_path)
    monkeypatch.setattr('deepgym.qa.mutants.local_isolation_kind', lambda: 'none')
    monkeypatch.setattr(local_dg, 'run_batch', lambda *_: pytest.fail('mutants ran'))

    check = mutation_check(env, reference, local_dg)

    assert check.status == 'inconclusive'
    assert check.details.startswith('local mutation runs need bwrap (Linux)')


@pytest.mark.parametrize('backend', ['apple', 'daytona'])
def test_mutants_never_run_where_the_judge_stays_online(
    backend: str, tmp_path: Path, local_dg: DeepGym, monkeypatch: pytest.MonkeyPatch
) -> None:
    env, reference = _function_env(tmp_path)
    monkeypatch.setattr(DeepGym, 'backend', property(lambda self: backend))
    monkeypatch.setattr(local_dg, 'run_batch', lambda *_: pytest.fail('mutants ran'))

    check = mutation_check(env, reference, local_dg)

    assert check.status == 'inconclusive'
    assert check.details.startswith(f'{backend} sandboxes keep the judge itself online')


_IN_PROCESS = (
    'def verify(solution_path, test_cases_path=None):\n'
    '    namespace = {}\n'
    '    exec(open(solution_path).read(), namespace)  # runs the solution inside the judge\n'
    "    return 1.0 if namespace['f'](1) == 2 else 0.0\n"
)


def _writes(marker: Path) -> str:
    """A solution whose f() works, after writing *marker* on the way."""
    return (
        'import pathlib\n\n'
        f'def f(x):\n    pathlib.Path({str(marker)!r}).write_text("x")\n    return x + 1\n'
    )


@needs_sandbox
def test_a_confined_judge_contains_a_verifier_that_runs_code_in_process(tmp_path: Path) -> None:
    marker = tmp_path / 'escaped.txt'
    env = Environment(task='t', verifier_code=_IN_PROCESS)
    assert DeepGym(mode='local').run(env, _writes(marker)).score == 1.0
    assert marker.exists()  # unconfined, code the verifier ran in-process reached the host
    marker.unlink()

    confined = DeepGym(mode='local', confine_local_judge=True).run(env, _writes(marker))

    assert not marker.exists()
    assert not confined.has_verdict  # the write was refused, so the verifier errored


@needs_sandbox
@pytest.mark.usefixtures('mutants_run_here')
def test_mutants_stay_contained_when_the_verifier_runs_them_in_process(
    tmp_path: Path, local_dg: DeepGym
) -> None:
    marker = tmp_path / 'escaped.txt'
    env = Environment(task='t', verifier_code=_IN_PROCESS)

    check = mutation_check(env, _writes(marker), local_dg)

    assert not marker.exists()
    assert check.status == 'inconclusive'
    assert 'cannot be judged' in check.details


@needs_sandbox
def test_a_confined_run_takes_down_what_the_judge_left_running() -> None:
    # A mutant that dropped worker.terminate(), run in-process by the verifier.
    verifier = (
        'def verify(solution_path, test_cases_path=None):\n'
        '    namespace = {}\n'
        '    exec(open(solution_path).read(), namespace)\n'
        "    return {'score': 1.0, 'passed': True, 'details': str(namespace['start']())}\n"
    )
    worker = 'import subprocess\n\ndef start():\n    return subprocess.Popen(["sleep", "60"]).pid\n'
    env = Environment(task='t', verifier_code=verifier)

    result = DeepGym(mode='local', confine_local_judge=True).run(env, worker)

    pid = int(result.output)
    for _ in range(50):  # the kill is sent before run() returns; give the OS a moment
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail(f'worker {pid} outlived its confined run')


@pytest.mark.skipif(local_isolation_kind() != 'macos', reason='sandbox-exec only')
def test_code_a_confined_judge_launches_starts_no_processes_on_macos() -> None:
    # sandbox-exec can't take down a child that calls setsid(), so none may start.
    verifier = (
        'from deepgym_solution import load_solution\n\n'
        'def verify(solution_path, test_cases_path=None):\n'
        '    try:\n'
        '        pid = load_solution(solution_path).start()\n'
        '    except OSError as exc:\n'
        "        return {'score': 0.0, 'passed': False, 'details': type(exc).__name__}\n"
        "    return {'score': 1.0, 'passed': True, 'details': str(pid)}\n"
    )
    worker = (
        'import subprocess\n\n'
        'def start():\n'
        '    return subprocess.Popen(["sleep", "60"], start_new_session=True).pid\n'
    )
    env = Environment(task='t', verifier_code=verifier)

    result = DeepGym(mode='local', confine_local_judge=True).run(env, worker)

    assert (result.score, result.output) == (0.0, 'BlockingIOError')


@pytest.mark.skipif(local_isolation_kind() != 'macos', reason='sandbox-exec only')
def test_a_confined_macos_script_starts_no_processes_even_when_allowed() -> None:
    verifier = (
        'from deepgym_solution import run_solution\n\n'
        'def verify(solution_path, test_cases_path=None):\n'
        '    proc = run_solution(solution_path, allow_subprocess=True)\n'
        '    ok = proc.returncode == 0\n'
        "    return {'score': float(ok), 'passed': ok, 'details': proc.stderr[-300:]}\n"
    )
    script = 'import subprocess\nsubprocess.Popen(["sleep", "60"], start_new_session=True)\n'
    env = Environment(task='t', verifier_code=verifier)

    result = DeepGym(mode='local', confine_local_judge=True).run(env, script)

    assert result.score == 0.0
    assert 'BlockingIOError' in result.output


def test_a_confined_judge_needs_a_local_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('deepgym.judge.local_isolation_kind', lambda: 'none')
    env = Environment(task='t', verifier_code='return 1.0\n')

    with pytest.raises(SandboxError, match='needs sandbox-exec or bwrap'):
        DeepGym(mode='local', confine_local_judge=True).run(env, '')


def test_mutation_check_skips_a_reference_that_is_not_python(local_dg: DeepGym) -> None:
    env = Environment(task='t', verifier_code='return 0.0\n')

    diff = '--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n'

    assert mutation_check(env, diff, local_dg).status == 'skip'


@needs_sandbox
@pytest.mark.usefixtures('mutants_run_here')
def test_run_qa_adds_the_mutation_check_only_when_asked(local_dg: DeepGym) -> None:
    plain = run_qa('reverse_string', local_dg, QAOptions(reruns=0))
    mutated = run_qa('reverse_string', local_dg, QAOptions(reruns=0, mutants=True))

    assert 'mutants_killed' not in [check.name for check in plain.checks]
    assert mutated.checks[-1].name == 'mutants_killed'
    assert mutated.checks[-1].details.startswith('mutation score ')
