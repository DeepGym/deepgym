"""The solution runs isolated from the judge: reward hacks fail, slow cases fail alone.

Runs locally. File-isolation checks need sandbox-exec (macOS) or bwrap (Linux)
and are skipped where neither works.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from deepgym import DeepGym, load_environment
from deepgym.exceptions import SandboxError
from deepgym.judge import local_isolation_kind, sandbox_isolation
from deepgym.models import Environment

needs_sandbox = pytest.mark.skipif(
    local_isolation_kind() == 'none', reason='no sandbox-exec or bwrap on this machine'
)

WRONG = 'def coin_change(coins, amount):\n    return 0\n'

_LINGERING = (
    'import threading, time\n\n'
    'def start():\n    threading.Thread(target=time.sleep, args=(60,)).start()\n\n'
    'def hang():\n    time.sleep(60)\n'
)


def _server(solution: Path, **pipes: int) -> subprocess.Popen:
    """Start a load_solution server for *solution* directly, as the judge would."""
    from deepgym import solution_runtime

    argv = [sys.executable, '-I', '-B', solution_runtime.__file__, '--serve', str(solution)]
    return subprocess.Popen(argv, stdin=subprocess.PIPE, **pipes)


def _call(name: str, call_id: int) -> bytes:
    message = {
        'op': 'call',
        'id': call_id,
        'target': {'name': name},
        'args': [],
        'kwargs': {'$d': []},
    }
    return (json.dumps(message) + '\n').encode()


def test_a_server_exits_with_its_pipe_even_if_the_solution_left_threads(tmp_path: Path) -> None:
    solution = tmp_path / 'solution.py'
    solution.write_text(_LINGERING, encoding='utf-8')
    server = _server(solution, stdout=subprocess.PIPE)
    server.stdin.write(b'{"op": "load", "id": 1}\n' + _call('start', 2))
    server.stdin.flush()
    ready, *replies = (server.stdout.readline() for _ in range(3))
    assert ready == b'{"ready": true}\n'
    assert all(b'"ok": true' in reply for reply in replies)

    server.stdin.close()  # the judge is done

    assert server.wait(timeout=10) == 0  # a non-daemon thread can't hold it for 60s


_PARTIAL = (
    'from deepgym_solution import LaunchError, load_solution\n\n'
    'def verify(solution_path, test_cases_path=None):\n'
    '    solution = load_solution(solution_path)\n'
    '    passed = 0\n'
    '    for values in ([], [1], [2]):\n'
    '        try:\n'
    '            passed += solution.f(values) == len(values)\n'
    '        except LaunchError:\n'
    '            raise\n'
    '        except Exception:\n'
    '            pass\n'
    "    return {'score': passed / 3, 'passed': passed == 3, 'details': f'{passed}/3'}\n"
)


def test_a_restart_with_the_run_out_of_time_keeps_the_partial_score() -> None:
    # Case 2 hangs through the whole run, so restarting for case 3 finds no time left:
    # the solution's doing (TimeoutError), never a launch failure that voids the run.
    solution = (
        'import time\n\ndef f(values):\n    if values:\n        time.sleep(60)\n    return 0\n'
    )
    env = Environment(task='t', verifier_code=_PARTIAL, timeout=1, case_timeout=1)

    result = DeepGym(mode='local').run(env, solution)

    assert result.has_verdict
    assert result.output == '1/3'


def test_run_solution_tells_a_launch_failure_from_the_script_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepgym import solution_runtime

    script = tmp_path / 'solution.py'
    script.write_text('import sys\nsys.exit(127)\n', encoding='utf-8')

    # The script's own exit code, however launcher-like, is its result to score.
    assert solution_runtime.run_solution(str(script)).returncode == 127
    monkeypatch.setenv('DEEPGYM_PYTHON', '/nonexistent/python')  # env starts, Python never does
    with pytest.raises(solution_runtime.LaunchError):
        solution_runtime.run_solution(str(script))


@pytest.mark.skipif(shutil.which('python3', path='/usr/bin') is None, reason='no python3')
def test_run_solution_finds_a_bare_interpreter_name_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepgym import solution_runtime

    script = tmp_path / 'solution.py'
    script.write_text('print("hi")\n', encoding='utf-8')
    monkeypatch.setenv('DEEPGYM_PYTHON', 'python3')  # as exported Harbor tasks set it

    proc = solution_runtime.run_solution(str(script))

    assert (proc.returncode, proc.stdout) == (0, 'hi\n')


def test_no_start_signal_goes_through_sudo(monkeypatch: pytest.MonkeyPatch) -> None:
    from deepgym import solution_runtime

    spec = {'kind': 'user', 'prefix': ['/usr/bin/sudo', '-n', '-u', 'solver', '--']}
    monkeypatch.setenv('DEEPGYM_ISOLATION', json.dumps(spec))

    with solution_runtime._start_signal() as (prefix, fds, started):
        assert (prefix, fds, started()) == ([], (), True)  # sudo would close the pipe


def test_a_reader_that_cannot_start_is_a_launch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepgym import solution_runtime

    solution = tmp_path / 'solution.py'
    solution.write_text('def f():\n    return 1\n', encoding='utf-8')
    started: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def popen(*args: object, **kwargs: object) -> subprocess.Popen:
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    def no_threads(self: object) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(solution_runtime.subprocess, 'Popen', popen)
    monkeypatch.setattr(solution_runtime.threading.Thread, 'start', no_threads)

    with pytest.raises(solution_runtime.LaunchError, match="can't start new thread"):
        solution_runtime.load_solution(str(solution))
    assert started[0].poll() is not None  # the server it started is gone


def test_local_sandboxes_hide_host_bus_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    import deepgym.judge

    monkeypatch.setattr(deepgym.judge, 'local_isolation_kind', lambda: 'bwrap')
    monkeypatch.setattr(deepgym.judge.os.path, 'exists', lambda path: True)

    hide = deepgym.judge.local_isolation([])['hide']

    assert {'/run', os.path.realpath('/var/run')} <= set(hide)
    assert len(hide) == len(set(hide))


_LAUNCHER = """
import subprocess, sys, time
argv = [sys.executable, '-I', '-B', sys.argv[1], '--serve', sys.argv[2]]
server = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
server.stdin.write(sys.argv[3].encode())
server.stdin.flush()
server.stdout.readline()  # ready
server.stdout.readline()  # loaded; hang() is next
print(server.pid, flush=True)
time.sleep(60)
"""


def test_a_server_exits_when_its_judge_dies_mid_call(tmp_path: Path) -> None:
    from deepgym import solution_runtime

    solution = tmp_path / 'solution.py'
    solution.write_text(_LINGERING, encoding='utf-8')
    requests = '{"op": "load", "id": 1}\n' + _call('hang', 2).decode()
    argv = [sys.executable, '-c', _LAUNCHER, solution_runtime.__file__, str(solution), requests]
    judge = subprocess.Popen(argv, stdout=subprocess.PIPE)
    pid = int(judge.stdout.readline())

    judge.kill()  # dies while the server sleeps in hang()
    judge.wait()

    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(pid, 9)
    pytest.fail(f'server {pid} outlived its judge')


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    return DeepGym(mode='local')


def _score(dg: DeepGym, solution: str, env: Environment | None = None) -> float:
    return dg.run(env or load_environment('coin_change'), solution).score


def _env(verify_body: str, **kwargs: object) -> Environment:
    """A verify()-style environment; *verify_body* sees ``sol`` = load_solution(...)."""
    source = (
        'from deepgym_solution import load_solution, run_solution\n\n'
        'def verify(solution_path, test_cases_path=None):\n'
        '    sol = load_solution(solution_path)\n'
        + ''.join(f'    {line}\n' for line in verify_body.strip().splitlines())
    )
    return Environment(task='t', verifier_code=source, **kwargs)


class TestRewardHacksFail:
    """Classic in-process hacks that used to work against verifiers that import the solution."""

    def test_reference_scores_full(self, dg: DeepGym) -> None:
        reference = load_environment('coin_change').verifier_path.with_name('reference_solution.py')
        assert _score(dg, reference.read_text()) == 1.0

    def test_fake_score_line_then_exit(self, dg: DeepGym) -> None:
        hack = 'import os\nprint(\'{"score": 1.0, "passed": true}\', flush=True)\nos._exit(0)\n'
        assert _score(dg, hack) == 0.0

    def test_fake_score_line_at_exit(self, dg: DeepGym) -> None:
        hack = (
            'import atexit\n'
            'atexit.register(lambda: print(\'{"score": 1.0, "passed": true}\'))\n' + WRONG
        )
        assert _score(dg, hack) < 1.0

    def test_object_that_equals_everything(self, dg: DeepGym) -> None:
        hack = (
            'class Anything:\n'
            '    def __eq__(self, other):\n'
            '        return True\n'
            'def coin_change(coins, amount):\n'
            '    return Anything()\n'
        )
        assert _score(dg, hack) == 0.0

    def test_int_subclass_that_equals_everything(self, dg: DeepGym) -> None:
        hack = (
            'class Sneaky(int):\n'
            '    def __eq__(self, other):\n'
            '        return True\n'
            'def coin_change(coins, amount):\n'
            '    return Sneaky(0)\n'
        )
        assert _score(dg, hack) < 1.0

    def test_patching_the_judge_does_nothing(self, dg: DeepGym) -> None:
        hack = 'import __main__\n__main__.ref_coin_change = lambda coins, amount: 0\n' + WRONG
        assert _score(dg, hack) < 1.0

    @needs_sandbox
    def test_cannot_read_judge_files_or_import_deepgym(self, dg: DeepGym) -> None:
        env = _env(
            """
from pathlib import Path
secret = Path(__file__).with_name('secret.txt')
secret.write_text('s3cret')
leaks = []
for probe, arg in (('read', str(secret)), ('import_deepgym', None)):
    try:
        getattr(sol, probe)(*([arg] if arg else []))
        leaks.append(probe)
    except Exception:
        pass
return {'score': 0.0 if leaks else 1.0, 'passed': not leaks, 'details': str(leaks)}
"""
        )
        solution = (
            'def read(path):\n'
            '    return open(path).read()\n'
            'def import_deepgym():\n'
            '    import deepgym\n'
        )
        result = dg.run(env, solution)
        assert result.passed, result.output

    def test_host_secrets_not_visible(self, dg: DeepGym, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('DG_TEST_SECRET', 'hunter2')
        env = _env("return {'score': 0.0 if sol.leaked() else 1.0, 'passed': not sol.leaked()}")
        solution = 'import os\ndef leaked():\n    return "DG_TEST_SECRET" in os.environ\n'
        assert dg.run(env, solution).passed


class TestRuntime:
    """Calls behave like local calls, with failures contained per case."""

    def test_slow_case_fails_alone(self, dg: DeepGym) -> None:
        env = _env(
            """
ok = 0
for n in (1, 2, 3):
    try:
        ok += sol.double(n) == 2 * n
    except TimeoutError:
        pass
return ok / 3
""",
            case_timeout=1,
            timeout=20,
        )
        solution = 'def double(n):\n    while n == 2:\n        pass\n    return 2 * n\n'
        assert dg.run(env, solution).score == pytest.approx(2 / 3)

    def test_deadline_ends_run_with_partial_credit(self, dg: DeepGym) -> None:
        env = _env(
            """
ok = 0
for n in range(20):
    try:
        ok += sol.spin(n) == n
    except TimeoutError:
        pass
return {'score': ok / 20, 'passed': False}
""",
            case_timeout=5,
            timeout=2,
        )
        start = time.monotonic()
        result = dg.run(env, 'def spin(n):\n    while True:\n        pass\n')
        assert result.score == 0.0
        assert time.monotonic() - start < 12

    def test_request_to_a_solution_that_stopped_reading_fails_alone(self, dg: DeepGym) -> None:
        """Sending the request counts against the case timeout, not just awaiting the reply."""
        env = _env(
            """
import time
time.sleep(1.5)
try:
    sol.size('x' * (1 << 20))
except TimeoutError:
    return 1.0
return 0.0
""",
            case_timeout=1,
            timeout=8,
        )
        # Once loaded, the solution stops its own process: it never reads the 1 MiB request.
        solution = (
            'import os, signal, threading\n'
            'threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGSTOP)).start()\n'
            'def size(text):\n'
            '    return len(text)\n'
        )
        start = time.monotonic()
        assert dg.run(env, solution).score == 1.0
        assert time.monotonic() - start < 6

    def test_crash_in_one_case_recovers(self, dg: DeepGym) -> None:
        env = _env(
            """
ok = 0
for n in (1, 2, 3):
    try:
        ok += sol.f(n) == n
    except Exception:
        pass
return ok / 3
"""
        )
        solution = 'import os\ndef f(n):\n    if n == 2:\n        os._exit(1)\n    return n\n'
        assert dg.run(env, solution).score == pytest.approx(2 / 3)

    def test_builtin_exceptions_keep_their_type(self, dg: DeepGym) -> None:
        env = _env(
            """
try:
    sol.f()
except ValueError as exc:
    return str(exc) == 'bad input'
return False
"""
        )
        assert dg.run(env, 'def f():\n    raise ValueError("bad input")\n').passed

    def test_plain_data_round_trips(self, dg: DeepGym) -> None:
        env = _env(
            """
value = {'t': (1, 2.5, None), 's': {1, 2}, 'b': bytes([0, 255]), 'inf': float('inf'), 7: [True]}
return sol.echo(value) == value
"""
        )
        assert dg.run(env, 'def echo(x):\n    return x\n').passed

    def test_objects_stay_remote(self, dg: DeepGym) -> None:
        env = _env(
            """
counter = sol.Counter(5)
counter.add(2)
return counter.value() == 7 and sol.LIMIT == 10
"""
        )
        solution = (
            'LIMIT = 10\n'
            'class Counter:\n'
            '    def __init__(self, start):\n'
            '        self.n = start\n'
            '    def add(self, k):\n'
            '        self.n += k\n'
            '    def value(self):\n'
            '        return self.n\n'
        )
        assert dg.run(env, solution).passed

    def test_solution_prints_do_not_break_calls(self, dg: DeepGym) -> None:
        env = _env('return sol.f(3) == 9')
        solution = 'def f(x):\n    print("noise " * 1000)\n    return x * x\n'
        assert dg.run(env, solution).passed

    def test_missing_function_is_attribute_error(self, dg: DeepGym) -> None:
        env = _env("return not hasattr(sol, 'nope') and hasattr(sol, 'f')")
        assert dg.run(env, 'def f():\n    pass\n').passed

    def test_import_error_surfaces(self, dg: DeepGym) -> None:
        env = _env('return 1.0')
        result = dg.run(env, 'def broken(:\n')
        assert result.score == 0.0
        assert 'SyntaxError' in result.output

    def test_run_solution_writes_only_in_cwd(self, dg: DeepGym) -> None:
        verifier = (
            'import tempfile\n'
            'from pathlib import Path\n'
            'from deepgym_solution import run_solution\n\n'
            'def verify(solution_path):\n'
            '    work = tempfile.mkdtemp()\n'
            "    proc = run_solution(solution_path, cwd=work, env={'OUT': work})\n"
            "    inside = Path(work, 'ok.txt').read_text() == 'ok'\n"
            "    outside = Path(__file__).with_name('pwned.txt').exists()\n"
            "    return inside and not outside and 'done' in proc.stdout\n"
        )
        solution = (
            'import os, pathlib, sys\n'
            'pathlib.Path(os.environ["OUT"], "ok.txt").write_text("ok")\n'
            'try:\n'
            '    pathlib.Path(sys.argv[0]).parent.parent.joinpath("pwned.txt").write_text("x")\n'
            'except OSError:\n'
            '    pass\n'
            'print("done")\n'
        )
        assert dg.run(Environment(task='t', verifier_code=verifier), solution).passed


class TestVerifierModes:
    """Verifier shape comes from parsing (or an explicit mode), not text search."""

    def test_script_with_helper_printer_and_future_import(self, dg: DeepGym) -> None:
        source = (
            'from __future__ import annotations\n'
            'import json, sys\n'
            'def emit(result: dict) -> None:\n'
            '    sys.stdout.write(json.JSONEncoder().encode(result) + "\\n")\n'
            'if __name__ == "__main__":\n'
            '    emit({"score": 1.0, "passed": True})\n'
        )
        env = Environment(task='t', verifier_code=source)
        assert dg.run(env, '').passed

    def test_body_with_future_import(self, dg: DeepGym) -> None:
        env = Environment(
            task='t', verifier_code='from __future__ import annotations\nreturn 1.0\n'
        )
        assert dg.run(env, '').passed

    def test_explicit_mode_wins(self, dg: DeepGym) -> None:
        env = Environment(task='t', verifier_code='return 0.25\n', verifier_mode='body')
        assert dg.run(env, '').score == 0.25


class TestSupportFiles:
    """Verifiers can import shared helper modules."""

    def test_helper_module_is_importable(self, dg: DeepGym) -> None:
        env = Environment(
            task='t',
            verifier_code=(
                'from scoring import half\n\ndef verify(solution_path):\n    return half()\n'
            ),
            support_files={'scoring.py': 'def half():\n    return 0.5\n'},
        )
        assert dg.run(env, '').score == 0.5

    def test_nested_helper_package(self, dg: DeepGym) -> None:
        env = Environment(
            task='t',
            verifier_code=(
                'from lib.scoring import full\n\ndef verify(solution_path):\n    return full()\n'
            ),
            support_files={
                'lib/__init__.py': '',
                'lib/scoring.py': 'def full():\n    return 1.0\n',
            },
        )
        assert dg.run(env, '').passed

    @needs_sandbox
    def test_solution_cannot_read_support_files(self, dg: DeepGym) -> None:
        env = Environment(
            task='t',
            verifier_code=(
                'from pathlib import Path\n'
                'from deepgym_solution import load_solution\n\n'
                'def verify(solution_path):\n'
                '    sol = load_solution(solution_path)\n'
                '    try:\n'
                '        sol.read(str(Path(__file__).with_name("answers.py")))\n'
                '        return 0.0\n'
                '    except Exception:\n'
                '        return 1.0\n'
            ),
            support_files={'answers.py': 'ANSWER = 42\n'},
        )
        assert dg.run(env, 'def read(p):\n    return open(p).read()\n').passed


class TestRegistrySupportFiles:
    """metadata.json can list shared helpers; paths can't escape the environments root."""

    def _bench(self, root, support: list[str]) -> None:
        import json

        (root / '_shared').mkdir(parents=True, exist_ok=True)
        (root / '_shared' / 'scoring.py').write_text('def score(x):\n    return float(x == 4)\n')
        env_dir = root / 'env1'
        env_dir.mkdir(exist_ok=True)
        (env_dir / 'task.md').write_text('square 2')
        (env_dir / 'verifier.py').write_text(
            'from deepgym_solution import load_solution\n'
            'from scoring import score\n\n'
            'def verify(solution_path):\n'
            '    return score(load_solution(solution_path).sq(2))\n'
        )
        (env_dir / 'metadata.json').write_text(
            json.dumps({'support_files': support, 'case_timeout': 3})
        )

    def test_shared_helper_loads(
        self, dg: DeepGym, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('deepgym.registry._get_cache_dir', lambda: tmp_path)
        self._bench(tmp_path / 'bench', ['../_shared/scoring.py'])
        env = load_environment('bench/env1')
        assert env.case_timeout == 3
        assert dg.run(env, 'def sq(x):\n    return x * x\n').passed

    def test_escaping_the_root_is_rejected(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('deepgym.registry._get_cache_dir', lambda: tmp_path / 'cache')
        (tmp_path / 'cache').mkdir()
        (tmp_path / 'outside.py').write_text('SECRET = 1\n')
        self._bench(tmp_path / 'cache' / 'bench', ['../../../outside.py'])
        with pytest.raises(ValueError, match='not a file under'):
            load_environment('bench/env1')


class TestJudgeSideHardening:
    """Things the judge reads or runs after the solution are safe from tampering."""

    @needs_sandbox
    def test_solution_cannot_swap_its_own_source(self, dg: DeepGym) -> None:
        env = _env(
            """
from deepgym_solution import solution_source
return 'cheat' in solution_source(solution_path)
"""
        )
        solution = (
            'try:\n'
            '    open(__file__, "w").write("def f():\\n    return 1  # innocent\\n")\n'
            'except OSError:\n'
            '    pass\n'
            'def f():\n'
            '    return 2  # cheat\n'
        )
        assert dg.run(env, solution).passed

    def test_symlinked_output_is_refused(self, dg: DeepGym) -> None:
        verifier = (
            'import tempfile\n'
            'from pathlib import Path\n'
            'from deepgym_solution import read_output, run_solution\n\n'
            'def verify(solution_path):\n'
            '    secret = Path(__file__).with_name("answers.txt")\n'
            '    secret.write_text("EXPECTED")\n'
            '    work = tempfile.mkdtemp()\n'
            '    run_solution(solution_path, cwd=work, env={"OUT": work, "TARGET": str(secret)})\n'
            '    try:\n'
            '        read_output(str(Path(work, "out.txt")), work)\n'
            '        return 0.0\n'
            '    except OSError:\n'
            '        return 1.0\n'
        )
        solution = (
            'import os\n'
            'os.symlink(os.environ["TARGET"], os.path.join(os.environ["OUT"], "out.txt"))\n'
        )
        assert dg.run(Environment(task='t', verifier_code=verifier), solution).passed

    def test_planted_git_config_does_not_run_in_judge(self, dg: DeepGym, tmp_path) -> None:
        marker = tmp_path / 'pwned'
        solution = (
            'import os, subprocess\n'
            'os.makedirs("my_repo")\n'
            'run = lambda *a: subprocess.run(a, cwd="my_repo", capture_output=True)\n'
            'run("git", "init", "-q")\n'
            f'run("git", "config", "core.fsmonitor", "touch {marker}")\n'
            'print("done")\n'
        )
        dg.run(load_environment('cli_task'), solution)
        assert not marker.exists()


class TestReviewFixes:
    """Regressions found in review of the isolation change."""

    @needs_sandbox
    def test_parallel_runs_cannot_read_each_other(self, dg: DeepGym) -> None:
        env = _env(
            """
import time
from pathlib import Path
Path(__file__).with_name('secret.txt').write_text('mine')
time.sleep(1)
found = sol.scan()
return {'score': 0.0 if found else 1.0, 'passed': not found, 'details': str(found)}
"""
        )
        solution = (
            'from pathlib import Path\n'
            'def scan():\n'
            '    root = Path(__file__).resolve().parents[2]\n'
            '    found = []\n'
            '    try:\n'
            '        for p in root.glob("*/*/*"):\n'
            '            try:\n'
            '                p.read_text()\n'
            '                found.append(str(p))\n'
            '            except OSError:\n'
            '                pass\n'
            '    except OSError:\n'
            '        pass\n'
            '    own = str(Path(__file__).resolve().parents[1])\n'
            '    return [f for f in found if not f.startswith(own)]\n'
        )
        batch = dg.run_batch(env, [solution] * 4, max_parallel=4)
        assert all(r.passed for r in batch.results), [r.output for r in batch.results]

    def test_builtin_subclasses_arrive_as_plain_values(self, dg: DeepGym) -> None:
        env = _env(
            """
counts, color, table = sol.values()
return counts == {'a': 2} and color == 1 and table == {'k': [1]}
"""
        )
        solution = (
            'import collections, enum\n'
            'class Color(enum.IntEnum):\n'
            '    RED = 1\n'
            'def values():\n'
            '    table = collections.defaultdict(list)\n'
            '    table["k"].append(1)\n'
            '    return collections.Counter("aa"), Color.RED, table\n'
        )
        assert dg.run(env, solution).passed

    def test_solution_source_falls_back_to_whole_file(self, dg: DeepGym) -> None:
        env = _env(
            """
from deepgym_solution import solution_source
return 'lambda' in solution_source(solution_path, 'two_sum')
"""
        )
        assert dg.run(env, 'two_sum = lambda nums, target: [0, 1]\n').passed

    def test_body_with_local_verify_helper(self, dg: DeepGym) -> None:
        body = 'def verify(path):\n    return 0.75\nreturn verify(solution_path)\n'
        assert dg.run(Environment(task='t', verifier_code=body), '').score == 0.75

    def test_namespace_skips_values_it_cannot_send(self, dg: DeepGym) -> None:
        env = _env(
            """
from deepgym_solution import solution_namespace
ns = solution_namespace(sol)
exec('result = f(2)', ns)
return ns['result'] == 4 and 'DEEP' not in ns
"""
        )
        solution = (
            'DEEP = []\n'
            'node = DEEP\n'
            'for _ in range(300):\n'
            '    node.append([])\n'
            '    node = node[0]\n'
            'def f(x):\n'
            '    return x * x\n'
        )
        assert dg.run(env, solution).passed


class TestSecurityFollowups:
    """Follow-ups from the security review of the isolation change."""

    def test_symlinked_parent_directory_is_refused(self, dg: DeepGym) -> None:
        verifier = (
            'import tempfile\n'
            'from pathlib import Path\n'
            'from deepgym_solution import read_output, run_solution\n\n'
            'def verify(solution_path):\n'
            '    Path(__file__).with_name("answers.txt").write_text("EXPECTED")\n'
            '    work = tempfile.mkdtemp()\n'
            '    target = str(Path(__file__).parent)\n'
            '    run_solution(solution_path, cwd=work, env={"OUT": work, "TARGET": target})\n'
            '    try:\n'
            '        read_output(str(Path(work, "out", "answers.txt")), work)\n'
            '        return 0.0\n'
            '    except OSError:\n'
            '        return 1.0\n'
        )
        solution = (
            'import os\nos.symlink(os.environ["TARGET"], os.path.join(os.environ["OUT"], "out"))\n'
        )
        assert dg.run(Environment(task='t', verifier_code=verifier), solution).passed

    def test_cli_task_never_runs_git_in_judge(self, dg: DeepGym, tmp_path) -> None:
        marker = tmp_path / 'pwned'
        solution = (
            'import os, subprocess\n'
            'os.makedirs("my_repo")\n'
            'run = lambda *a: subprocess.run(a, cwd="my_repo", capture_output=True)\n'
            'run("git", "init", "-q")\n'
            f'for key in ("core.fsmonitor", "core.pager", "core.alternateRefsCommand"):\n'
            f'    run("git", "config", key, "touch {marker}")\n'
            'print("done")\n'
        )
        dg.run(load_environment('cli_task'), solution)
        assert not marker.exists()

    @needs_sandbox
    def test_home_directory_is_hidden(self, dg: DeepGym) -> None:
        import uuid
        from pathlib import Path

        secret = Path.home() / f'.deepgym_test_secret_{uuid.uuid4().hex}'
        secret.write_text('home secret')
        try:
            env = _env(f'return not sol.read({str(secret)!r})')
            solution = (
                'def read(path):\n'
                '    try:\n'
                '        return open(path).read()\n'
                '    except OSError:\n'
                '        return None\n'
            )
            assert dg.run(env, solution).passed
        finally:
            secret.unlink()


macos_only = pytest.mark.skipif(
    local_isolation_kind() != 'macos', reason='fork limits apply to the macOS sandbox'
)


class TestForkLimits:
    """On macOS, a detached child could outlive the solution, so forking is opt-in."""

    @macos_only
    def test_functions_cannot_fork(self, dg: DeepGym) -> None:
        env = _env(
            """
try:
    sol.spawn()
    return 0.0
except Exception:
    return 1.0
"""
        )
        solution = 'import subprocess\ndef spawn():\n    subprocess.run(["true"], check=True)\n'
        assert dg.run(env, solution).passed

    @macos_only
    def test_scripts_fork_only_when_allowed(self, dg: DeepGym) -> None:
        verifier = (
            'from deepgym_solution import run_solution\n\n'
            'def verify(solution_path):\n'
            '    blocked = run_solution(solution_path).returncode != 0\n'
            '    allowed = run_solution(solution_path, allow_subprocess=True).returncode == 0\n'
            '    return blocked and allowed\n'
        )
        solution = 'import subprocess\nsubprocess.run(["true"], check=True)\n'
        assert dg.run(Environment(task='t', verifier_code=verifier), solution).passed


LOOPBACK_PROBE_VERIFIER = """
import socket
import subprocess
import sys
from deepgym_solution import run_solution

def verify(solution_path):
    allowed, other = socket.create_server(('127.0.0.1', 0)), socket.create_server(('127.0.0.1', 0))
    ports = [str(server.getsockname()[1]) for server in (allowed, other)]
    out = ' '.join(run_solution(solution_path, ports, loopback_port=int(ports[0])).stdout.split())
    return {'score': float(out == 'open closed closed'), 'passed': True, 'details': out}
"""

LOOPBACK_PROBE = """
import socket, sys

for host, port in [('127.0.0.1', sys.argv[1]), ('127.0.0.1', sys.argv[2]), ('1.1.1.1', 80)]:
    try:
        socket.create_connection((host, int(port)), timeout=3).close()
        print('open')
    except OSError:
        print('closed')
"""


@needs_sandbox
def test_solution_reaches_only_the_judges_loopback_port(dg: DeepGym) -> None:
    """loopback_port opens that one judge port; other ports and egress stay shut."""
    env = Environment(task='t', verifier_code=LOOPBACK_PROBE_VERIFIER)

    assert dg.run(env, LOOPBACK_PROBE).output == 'open closed closed'


def test_overlapping_loopback_setups_in_a_shared_workdir_stay_judge_owned(monkeypatch) -> None:
    """Two overlapping calls sharing a workdir with a planted symlink and directory each get
    their own unpredictable, judge-owned socket that relays to their own port, and neither
    loosens the judge's umask nor chmods anything by path.
    """
    from deepgym import solution_runtime

    base = Path(tempfile.mkdtemp(dir='/tmp'))  # AF_UNIX paths must stay short
    servers = [socket.create_server(('127.0.0.1', 0)) for _ in range(2)]
    try:
        target = base / 'socket.py'  # stands in for a module root imports
        target.write_text('# root imports this')
        target.chmod(0o644)
        runtime = base / 'runtime' / 'deepgym_solution.py'
        runtime.parent.mkdir()
        runtime.write_text('')
        work = base / 'work'
        work.mkdir()
        (work / 'loopback.sock').symlink_to(target)
        (work / 'tmpplanted').mkdir()
        planted = sorted(p.name for p in work.iterdir())

        calls: list[tuple[str, object]] = []
        umask, chmod = os.umask, os.chmod
        monkeypatch.setattr(os, 'umask', lambda m: calls.append(('umask', m)) or umask(m))
        monkeypatch.setattr(
            os, 'chmod', lambda *a, **k: calls.append(('chmod', a)) or chmod(*a, **k)
        )
        isolation = {'kind': 'user', 'prefix': [], 'own_network': True}
        monkeypatch.setenv('DEEPGYM_ISOLATION', json.dumps(isolation))
        monkeypatch.setenv('DEEPGYM_RUNTIME', str(runtime))
        ports = [server.getsockname()[1] for server in servers]

        with (
            solution_runtime._loopback(ports[0]) as first,
            solution_runtime._loopback(ports[1]) as second,
        ):
            paths = [Path(first[-1]), Path(second[-1])]
            assert paths[0].parent != paths[1].parent
            for path, server in zip(paths, servers):
                assert path.parent.parent == runtime.parent
                assert oct(path.parent.stat().st_mode & 0o777) == oct(0o711)
                assert oct(path.stat().st_mode & 0o777) == oct(0o777)
                server.settimeout(5)
                with socket.socket(socket.AF_UNIX) as client:
                    client.connect(str(path))
                    server.accept()[0].close()  # relayed to this call's own port

        assert calls == []
        assert not any(path.parent.exists() for path in paths)
        assert oct(target.stat().st_mode & 0o777) == oct(0o644)
        assert sorted(p.name for p in work.iterdir()) == planted
    finally:
        for server in servers:
            server.close()
        shutil.rmtree(base)


def test_root_loopback_bootstrap_ignores_modules_planted_in_cwd_or_env(tmp_path) -> None:
    from deepgym.judge import _LOOPBACK_UP_THEN_EXEC

    marker = tmp_path / 'pwned'
    for name in ('socket', 'fcntl', 'struct', 'sitecustomize'):
        (tmp_path / f'{name}.py').write_text(f'open({str(marker)!r}, "w").write("{name}")\n')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path), 'PYTHONSTARTUP': str(tmp_path / 'fcntl.py')}

    # Not root (and maybe not Linux), so the ioctl fails; any planted import would run first.
    subprocess.run(
        [sys.executable, '-I', '-S', '-c', _LOOPBACK_UP_THEN_EXEC, 'true'],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        check=False,
    )

    assert not marker.exists()


def test_sharing_a_workdir_never_follows_symlinks(tmp_path) -> None:
    from deepgym.solution_runtime import _share_with_solution

    judge_file = tmp_path / 'judge_secret.txt'
    judge_file.write_text('EXPECTED')
    judge_file.chmod(0o600)
    work = tmp_path / 'work'
    (work / 'sub').mkdir(parents=True)
    (work / 'input.csv').write_text('a,b\n')
    (work / 'input.csv').chmod(0o600)
    (work / 'link').symlink_to(judge_file)
    (work / 'sub' / 'dirlink').symlink_to(tmp_path)

    _share_with_solution(str(work))

    assert judge_file.stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o777 != 0o777
    assert (work / 'input.csv').stat().st_mode & 0o044 == 0o044
    assert (work / 'sub').stat().st_mode & 0o777 == 0o777


def test_sandbox_isolation_goes_offline_or_fails_closed() -> None:
    offline = sandbox_isolation('0 999 987 /usr/bin/python3 1')[0]['prefix']
    assert offline[:3] == ['unshare', '--net', '--']
    # Root brings loopback up from the solution's cwd: no cwd, site or env on its path.
    assert offline[3:7] == ['/usr/bin/python3', '-I', '-S', '-c']
    assert "{'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'}" in offline[7]
    assert '--bounding-set=-all' in offline
    assert sandbox_isolation('0 999 987 /usr/bin/python3 0')[0]['prefix'][0] == 'setpriv'
    with pytest.raises(SandboxError, match='network'):
        sandbox_isolation('0 999 987 /usr/bin/python3 0', require_offline=True)
