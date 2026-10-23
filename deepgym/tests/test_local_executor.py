"""Tests for deepgym.sandbox.LocalExecutor."""

import asyncio
import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.exceptions import DeepGymError, SandboxError, TimeoutError, VerifierError
from deepgym.judge import JudgeJob
from deepgym.models import Environment, VerifierResult
from deepgym.sandbox import AsyncLocalExecutor, ExecutionResult, LocalExecutor, judge_verdict

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs'


@pytest.fixture()
def executor() -> LocalExecutor:
    """Return a fresh LocalExecutor."""
    return LocalExecutor()


def _standalone_verifier(return_expr: str) -> str:
    """Build a standalone verifier script that returns the given expression.

    Unlike wrap_verifier, standalone scripts use sys.exit() so stdout is
    properly flushed before process exit.
    """
    return (
        '#!/usr/bin/env python3\n'
        'import sys, json\n'
        'def _run(solution_path, test_cases_path=None):\n'
        f'    {return_expr}\n'
        'if __name__ == "__main__":\n'
        '    sp = sys.argv[1] if len(sys.argv) > 1 else "solution.py"\n'
        '    result = _run(sp)\n'
        '    if isinstance(result, (int, float)):\n'
        '        out = {"schema_version":"1.0","score":float(result),'
        '"passed":float(result)>=0.5,"details":None,"truncated":False}\n'
        '    elif isinstance(result, dict):\n'
        '        out = {"schema_version":"1.0","score":float(result.get("score",0)),'
        '"passed":bool(result.get("passed",False)),'
        '"details":result.get("details"),"truncated":False}\n'
        '    else:\n'
        '        out = {"schema_version":"1.0","score":0.0,"passed":False,'
        '"details":"bad type","truncated":False}\n'
        '    out["score"] = max(0.0, min(1.0, out["score"]))\n'
        '    print(json.dumps(out))\n'
        '    sys.exit(0 if out["passed"] else 1)\n'
    )


def _make_ready_verifier(verifier_path: Path) -> str:
    """Read a standalone verifier file and return it as-is."""
    return verifier_path.read_text(encoding='utf-8')


class TestLocalExecutorRun:
    """Core run behaviour of LocalExecutor."""

    def test_passing_solution(self, executor: LocalExecutor) -> None:
        verifier_code = _standalone_verifier('return 1.0')
        result = executor.run(
            JudgeJob(
                verifier=verifier_code,
                solution='# anything\n',
                timeout=10,
            )
        )
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.verifier_result, VerifierResult)
        assert result.verifier_result.score == 1.0
        assert result.verifier_result.passed is True
        assert isinstance(result.stderr, str)
        assert isinstance(result.exit_code, int)

    def test_failing_solution(self, executor: LocalExecutor) -> None:
        verifier_code = _standalone_verifier('return 0.0')
        result = executor.run(
            JudgeJob(
                verifier=verifier_code,
                solution='# anything\n',
                timeout=10,
            )
        )
        assert isinstance(result, ExecutionResult)
        assert result.verifier_result.score == 0.0
        assert result.verifier_result.passed is False

    def test_timeout(self, executor: LocalExecutor) -> None:
        slow_verifier = (
            '#!/usr/bin/env python3\n'
            'import sys, time, json\n'
            'time.sleep(60)\n'
            'print(json.dumps({"schema_version":"1.0","score":1.0,"passed":True,"truncated":False}))\n'
        )
        with pytest.raises(TimeoutError):
            executor.run(
                JudgeJob(
                    verifier=slow_verifier,
                    solution='# slow\n',
                    timeout=1,
                )
            )

    def test_invalid_verifier_no_json_output(self, executor: LocalExecutor) -> None:
        bad_verifier = (
            '#!/usr/bin/env python3\nimport sys\nprint("this is not json")\nsys.exit(0)\n'
        )
        with pytest.raises(VerifierError):
            executor.run(
                JudgeJob(
                    verifier=bad_verifier,
                    solution='# anything\n',
                    timeout=10,
                )
            )

    @pytest.mark.parametrize(
        'ending',
        ['os.kill(os.getpid(), signal.SIGKILL)', 'os._exit(137)', 'os._exit(1)'],
        ids=['killed', 'exit_137', 'exit_1_claiming_pass'],
    )
    def test_pass_printed_before_a_bad_exit_is_an_error(self, ending: str) -> None:
        verifier = (
            'import os, signal\n'
            'if __name__ == "__main__":\n'
            '    print(\'{"score": 1.0, "passed": true}\', flush=True)\n'
            f'    {ending}\n'
        )
        env = Environment(task='t', verifier_code=verifier)
        with pytest.raises(VerifierError, match='exited'):
            DeepGym(mode='local').run(env, '')

    def test_result_is_execution_result_instance(self, executor: LocalExecutor) -> None:
        verifier_code = _standalone_verifier(
            'return {"score": 0.7, "passed": True, "details": "ok"}'
        )
        result = executor.run(
            JudgeJob(
                verifier=verifier_code,
                solution='# x\n',
                timeout=10,
            )
        )
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.verifier_result, VerifierResult)
        assert result.verifier_result.details == 'ok'

    def test_invalid_env_var_key_rejected(self, executor: LocalExecutor) -> None:
        verifier_code = _standalone_verifier('return 1.0')
        with pytest.raises(SandboxError, match='Invalid environment variable name'):
            executor.run(
                JudgeJob(
                    verifier=verifier_code,
                    solution='# x\n',
                    timeout=10,
                    env_vars={'; rm -rf /': 'bad'},
                )
            )


async def test_cancelled_async_run_kills_and_reaps_the_verifier(tmp_path: Path) -> None:
    pid_file = tmp_path / 'pids'
    # The verifier starts a child that holds its output pipes, as a helper process might.
    verifier = (
        'import os, subprocess, time\n'
        'child = subprocess.Popen(["sleep", "60"])\n'
        'path = os.environ["PID_FILE"]\n'
        'open(path + ".tmp", "w").write(f"{os.getpid()} {child.pid}")\n'
        'os.replace(path + ".tmp", path)\n'
        'time.sleep(60)\n'
    )
    job = JudgeJob(verifier=verifier, solution='', timeout=60, env_vars={'PID_FILE': str(pid_file)})
    run = asyncio.ensure_future(AsyncLocalExecutor().run(job))
    while not pid_file.exists():
        await asyncio.sleep(0.05)
    pids = [int(pid) for pid in pid_file.read_text().split()]
    try:
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, timeout=3)
        await asyncio.sleep(0.2)  # the child is reparented to init, which reaps it
        for pid in pids:  # killed and reaped: not even a zombie is left
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize('use_async', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('reads_it', [False, True], ids=['before_the_solution', 'after_reading_it'])
async def test_an_outer_timeout_is_the_solutions_once_its_output_is_in_play(
    use_async: bool, reads_it: bool
) -> None:
    """A verifier stuck on what the solution gave it (a regex it made backtrack, say)."""
    read = 'open(sys.argv[1]).read()\n    ' if reads_it else ''
    verifier = f'import sys\n\nif __name__ == "__main__":\n    {read}while True:\n        pass\n'
    env = Environment(task='t', verifier_code=verifier, verifier_mode='script', timeout=1)
    job = JudgeJob.for_environment(env, 'x = 1\n')
    if use_async:
        run = AsyncLocalExecutor().run(job)
    else:
        run = asyncio.to_thread(LocalExecutor().run, job)

    if not reads_it:
        with pytest.raises(TimeoutError):
            await run
        return
    executed = await run
    verdict = executed.verifier_result
    assert (verdict.score, verdict.error_type, executed.exit_code) == (0.0, 'solution_timeout', 1)


def test_a_timed_out_run_kills_and_reaps_the_verifier_and_its_children(tmp_path: Path) -> None:
    pid_file = tmp_path / 'pids'
    verifier = (
        'import os, subprocess, time\n'
        'child = subprocess.Popen(["sleep", "60"])\n'
        'path = os.environ["PID_FILE"]\n'
        'open(path + ".tmp", "w").write(f"{os.getpid()} {child.pid}")\n'
        'os.replace(path + ".tmp", path)\n'
        'time.sleep(60)\n'
    )
    job = JudgeJob(verifier=verifier, solution='', timeout=1, env_vars={'PID_FILE': str(pid_file)})
    with pytest.raises(TimeoutError):
        LocalExecutor().run(job)
    pids = [int(pid) for pid in pid_file.read_text().split()]
    try:
        time.sleep(0.2)  # the child is reparented to init, which reaps it
        for pid in pids:  # killed and reaped: not even a zombie is left
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


# A C loop holds the interpreter lock, so no thread in that process (a watchdog, say) runs.
_HOG = 'import itertools; sum(itertools.repeat(1))'
# As a module (load_solution) f() hogs; as a script (run_solution) it starts a busy child
# that is told the script's path, so it can be found, and then hogs too.
_HOGGING_SOLUTION = (
    'import itertools, subprocess, sys\n\n'
    'def f():\n'
    '    sum(itertools.repeat(1))\n\n'
    'if __name__ == "__main__":\n'
    f'    subprocess.Popen([sys.executable, "-c", {_HOG!r}, sys.argv[0]])\n'
    '    sum(itertools.repeat(1))\n'
)
_ENDS_WHILE_SOLUTION_RUNS = """\
import os, sys, threading, time
from deepgym_solution import load_solution, run_solution

def start():
    if os.environ["CALL"] == "load_solution":
        load_solution(sys.argv[1]).f()
    else:
        run_solution(sys.argv[1], allow_subprocess=True, timeout=60)

threading.Thread(target=start, daemon=True).start()
time.sleep(1)  # the solution and its child are running
with open(os.environ["MARK"], "w") as mark:
    mark.write(sys.argv[1])
if os.environ["ENDING"] == "crash":
    raise RuntimeError("the verifier crashes while the solution runs")
if os.environ["ENDING"] in ("timeout", "cancel"):
    time.sleep(60)
print('{"score": 1.0, "passed": true}')
"""


def _processes_running(solution_path: str) -> list[str]:
    """The processes whose command line names *solution_path*: its server, script, child."""
    listing = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True)
    return [line for line in listing.stdout.splitlines() if solution_path in line]


@pytest.mark.parametrize('call', ['load_solution', 'run_solution'])
@pytest.mark.parametrize('ending', ['normal', 'crash', 'timeout', 'cancel'])
async def test_no_solution_process_outlives_its_run(tmp_path: Path, call: str, ending: str) -> None:
    """However a run ends, nothing the solution started keeps running: not a server or
    script hogging the interpreter lock, nor a busy child it started."""
    mark = tmp_path / 'solution_path'
    env_vars = {'CALL': call, 'ENDING': ending, 'MARK': str(mark)}
    job = JudgeJob(
        verifier=_ENDS_WHILE_SOLUTION_RUNS,
        solution=_HOGGING_SOLUTION,
        timeout=2,
        env_vars=env_vars,
    )
    if ending == 'cancel':
        run = asyncio.ensure_future(AsyncLocalExecutor().run(job))
        while not (mark.exists() and mark.read_text()):  # created before it is written
            await asyncio.sleep(0.05)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
    else:
        with contextlib.suppress(DeepGymError):  # a crash or timeout has no verdict
            await asyncio.to_thread(LocalExecutor().run, job)
    solution_path = mark.read_text()
    assert solution_path  # an empty one would match, and kill, every process
    try:
        deadline = time.monotonic() + 5
        while _processes_running(solution_path) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        assert _processes_running(solution_path) == []
    finally:  # never leave a busy process behind, even when the test fails
        for line in _processes_running(solution_path):
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(line.split()[0]), signal.SIGKILL)


class TestExampleEnvironments:
    """Ensure all 3 example environments pass with their reference solutions."""

    def test_sorting_example(self, executor: LocalExecutor) -> None:
        verifier_code = _make_ready_verifier(ENVS_DIR / 'python_sorting' / 'verifier.py')
        solution_code = (ENVS_DIR / 'python_sorting' / 'reference_solution.py').read_text(
            encoding='utf-8'
        )
        result = executor.run(
            JudgeJob(
                verifier=verifier_code,
                solution=solution_code,
                timeout=30,
            )
        )
        assert result.verifier_result.passed is True
        assert result.verifier_result.score == 1.0

    def test_two_sum_example(self, executor: LocalExecutor) -> None:
        verifier_code = _make_ready_verifier(ENVS_DIR / 'two_sum' / 'verifier.py')
        solution_code = (ENVS_DIR / 'two_sum' / 'reference_solution.py').read_text(encoding='utf-8')
        result = executor.run(
            JudgeJob(
                verifier=verifier_code,
                solution=solution_code,
                timeout=30,
            )
        )
        assert result.verifier_result.passed is True
        assert result.verifier_result.score == 1.0

    def test_string_manipulation_example(self, executor: LocalExecutor) -> None:
        verifier_code = _make_ready_verifier(ENVS_DIR / 'string_manipulation' / 'verifier.py')
        solution_code = (ENVS_DIR / 'string_manipulation' / 'reference_solution.py').read_text(
            encoding='utf-8'
        )
        result = executor.run(
            JudgeJob(
                verifier=verifier_code,
                solution=solution_code,
                timeout=30,
            )
        )
        assert result.verifier_result.passed is True
        assert result.verifier_result.score == 1.0


_JUDGE_DIES = """\
import os, signal
from deepgym_solution import load_solution

def verify(solution_path, test_cases_path=None):
    if os.environ['WHEN'] == 'after':
        load_solution(solution_path).f()
    os.kill(os.getpid(), signal.SIGKILL)
"""


@pytest.mark.parametrize('executor', [LocalExecutor, AsyncLocalExecutor])
def test_a_judge_that_dies_after_the_solution_ran_is_its_judged_zero(executor) -> None:
    """Killed or starved through what the solution ran, the judge printed no verdict: the
    solution's 0, or ending the judge would dodge it. Before the solution ran, no verdict."""

    def run(when: str) -> ExecutionResult:
        env = Environment(task='t', verifier_code=_JUDGE_DIES, env_vars={'WHEN': when})
        job = JudgeJob.for_environment(env, 'def f():\n    return 1\n')
        result = executor().run(job)
        return asyncio.run(result) if asyncio.iscoroutine(result) else result

    after = run('after').verifier_result
    assert (after.score, after.error_type) == (0.0, 'solution_error')
    with pytest.raises(VerifierError):
        run('before')


def test_a_judge_killed_after_an_earlier_report_is_still_the_solutions_zero() -> None:
    """A report printed before a later solution call can't make the judge's death after it
    anyone else's: the abnormal exit decides, whatever was printed."""
    stdout = 'mark\n{"score": 1.0, "passed": true}\n'

    killed = judge_verdict(stdout, '', -9, 'mark').verifier_result

    assert (killed.score, killed.error_type) == (0.0, 'solution_error')
    assert judge_verdict(stdout, '', 0, 'mark').verifier_result.score == 1.0
    with pytest.raises(VerifierError):
        judge_verdict('', '', -9, 'mark')  # nothing announced: the solution never ran
