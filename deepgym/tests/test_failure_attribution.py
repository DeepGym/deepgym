"""Whose fault a failure is comes out the same on every path that grades a solution.

The scenarios walk the outcome table in deepgym/solution_runtime.py: everything the
solution does wrong, or a resource it exhausts once it runs, is a judged 0 (a hang past
its time limit with error_type solution_timeout); the runtime failing its preflight or
launch before any solution code runs, or the verifier failing first, has no verdict; and
a crash never keeps a score printed before it. That must hold for every verifier shape and
auto detection, sync and async runs, and session step and final verifiers. These run
locally; test_sandboxes_live.py reruns them on every sandbox provider it can reach,
through its own ``mode`` fixture.
"""

import asyncio
import contextlib
import textwrap

import pytest

from deepgym import AsyncDeepGym, DeepGym
from deepgym.exceptions import DeepGymError, VerifierError
from deepgym.models import Action, Environment, MultiTurnEnvironment, RunResult
from deepgym.sandbox import error_result
from deepgym.session import Session

GOOD = 'def double(n):\n    return 2 * n\n'
RAISES = 'def double(n):\n    raise ValueError("solution bug")\n'
HANGS = 'def double(n):\n    while True:\n        pass\n'
# Garbage on the pipe the runtime answers through (the only writable pipe it holds).
_SCRIBBLE = (
    'import os, stat\n\n'
    'def double(n):\n'
    '    for fd in range(3, 64):\n'
    '        try:\n'
    '            if stat.S_ISFIFO(os.fstat(fd).st_mode):\n'
    '                os.write(fd, b"not json\\n")\n'
    '        except OSError:\n'
    '            pass\n'
    '    return 2 * n\n'
)
# Uses the answer (+ 0), so garbage from the solution crashes the verifier's own code.
_CALL = 'from deepgym_solution import load_solution\nok = load_solution({path}).double(2) + 0 == 4'
_LOAD = 'from deepgym_solution import load_solution\nload_solution({path})\n'
_PRINT_THEN_CRASH = (
    'import json\nprint(json.dumps({"score": 0.75, "passed": False}), flush=True)\n'
    'raise RuntimeError("verifier bug")'
)
# A partial line left unterminated must not swallow the crash report printed after it.
_PRINT_PARTIAL_LINE_THEN_CRASH = _PRINT_THEN_CRASH.replace(
    'raise', 'print("checking", end="", flush=True)\nraise'
)
# Once the solution has run, a resource it exhausted makes the verifier's own I/O fail.
_EXHAUSTED = '\nimport errno\nraise {error}(errno.{code}, "exhausted")'
# A later launch, after the solution ran, finds the disk full.
_LATER_LAUNCH_ON_A_FULL_DISK = (
    'import errno, deepgym_solution as runtime\n'
    'runtime.load_solution({path}).double(2)\n'
    'def full(*args, **kwargs):\n'
    '    raise OSError(errno.ENOSPC, "No space left on device")\n'
    'runtime._scratch_dir = full\n'
    'runtime.run_solution({path})\n'
    'ok = True'
)
# The sandbox is short of space before the solution runs: the preflight's to catch.
_DISK_ALREADY_FULL = (
    'import os\n'
    'os.statvfs = lambda path: os.statvfs_result((4096, 4096, 100, 0, 0, 100, 0, 0, 0, 255))\n'
)
# The verifier's own isolation launcher can't be found, or can't start code.
_LAUNCHER = (
    'import json, os\n'
    'os.environ["DEEPGYM_ISOLATION"] = json.dumps({{"kind": "user", "prefix": ["{0}"]}})\n'
)

# Verifier statements over a solution at {path}, ending with `ok` set (or a crash), the
# solution they grade, and the outcome every path must give: one row per outcome in the
# table in deepgym/solution_runtime.py.
SCENARIOS = {
    'works': (_CALL, GOOD, 'pass'),
    # Solution faults: a judged 0.
    'solution_raises': (_CALL, RAISES, 'judged_zero'),
    'solution_exits': (_CALL, 'import os\n\ndef double(n):\n    os._exit(3)\n', 'judged_zero'),
    'solution_exits_early': (_CALL, 'import os\nos._exit(0)\n', 'judged_zero'),
    'solution_is_killed': (
        _CALL,
        'import os, signal\n\ndef double(n):\n    os.kill(os.getpid(), signal.SIGKILL)\n',
        'judged_zero',
    ),
    'solution_hangs': (_CALL, HANGS, 'judged_zero:solution_timeout'),
    'solution_runs_out_of_memory': (
        _CALL,
        'def double(n):\n    raise MemoryError("out of memory")\n',
        'judged_zero',
    ),
    'solution_returns_garbage': (_CALL, 'def double(n):\n    return "four"\n', 'judged_zero'),
    'solution_breaks_the_protocol': (_CALL, _SCRIBBLE, 'judged_zero'),
    'solution_closes_its_pipe': (
        _CALL,
        'import os\nos.closerange(3, 64)\n\ndef double(n):\n    return 2 * n\n',
        'judged_zero',
    ),
    'solution_exhausts_the_disk': (
        _CALL + _EXHAUSTED.format(error='OSError', code='ENOSPC'),
        GOOD,
        'judged_zero',
    ),
    'solution_exhausts_inodes': (
        _CALL + _EXHAUSTED.format(error='OSError', code='EDQUOT'),
        GOOD,
        'judged_zero',
    ),
    'solution_exhausts_processes': (
        _CALL + _EXHAUSTED.format(error='BlockingIOError', code='EAGAIN'),
        GOOD,
        'judged_zero',
    ),
    'solution_exhausts_file_descriptors': (
        _CALL + _EXHAUSTED.format(error='OSError', code='EMFILE'),
        GOOD,
        'judged_zero',
    ),
    'later_launch_on_a_full_disk': (_LATER_LAUNCH_ON_A_FULL_DISK, GOOD, 'judged_zero'),
    'crash_after_the_solution_runs': (_LOAD + _PRINT_THEN_CRASH, GOOD, 'judged_zero'),
    'partial_line_then_crash': (_LOAD + _PRINT_PARTIAL_LINE_THEN_CRASH, GOOD, 'judged_zero'),
    # Infra, or the verifier failing before the solution runs: no verdict.
    'disk_full_before_the_solution_runs': (_DISK_ALREADY_FULL + _CALL, GOOD, 'no_verdict'),
    'launcher_is_missing': (_LAUNCHER.format('/nonexistent/launcher') + _CALL, GOOD, 'no_verdict'),
    'launcher_cannot_start_code': (_LAUNCHER.format('/bin/false') + _CALL, GOOD, 'no_verdict'),
    'setup_fails_first': (f'open("missing_cases.json").read()\n{_CALL}', GOOD, 'no_verdict'),
    'crash_before_the_solution_runs': (_PRINT_THEN_CRASH, GOOD, 'no_verdict'),
}
# (verifier shape, Environment.verifier_mode): each shape named, and found by auto detection.
SHAPES = [
    ('script', 'script'),
    ('script', 'auto'),
    ('verify_function', 'verify_function'),
    ('verify_function', 'auto'),
    ('body', 'body'),
    ('body', 'auto'),
]


@pytest.fixture(scope='module')
def mode() -> str:
    """Where runs happen; test_sandboxes_live.py overrides this with each provider."""
    return 'local'


def _outcome(result: RunResult) -> str:
    if not result.has_verdict:
        return 'no_verdict'
    if result.passed and result.score == 1.0:
        return 'pass'
    if (result.score, result.passed) != (0.0, False):
        return str(result)
    return 'judged_zero' + (f':{result.error_type}' if result.error_type else '')


def _single_turn(statements: str, shape: str) -> str:
    """The verifier *statements* (over solution_path) as a verifier of *shape*."""
    body = statements.replace('{path}', 'solution_path') + '\nreturn 1.0 if ok else 0.0'
    if shape == 'body':
        return body
    return _with_main(
        f'def verify(solution_path, test_cases_path=None):\n{textwrap.indent(body, "    ")}\n',
        shape,
    )


def _with_main(verify: str, shape: str) -> str:
    """*verify* (a verify_function verifier), with a main block that reports it if *shape*
    is a script."""
    if shape != 'script':
        return verify
    return (
        f'import json, sys\n\n{verify}\n'
        'if __name__ == "__main__":\n'
        '    score = verify(sys.argv[1])\n'
        '    print(json.dumps({"score": score, "passed": score == 1.0}))\n'
        '    sys.exit(0 if score == 1.0 else 1)\n'
    )


def _session_verifier(statements: str) -> str:
    """The verifier *statements* as a session verifier grading the agent's agent.py."""
    return (
        statements.replace('{path}', '"agent.py"')
        + '\nimport json\nprint(json.dumps({"score": float(ok), "passed": ok}))\n'
    )


def _grade(mode: str, use_async: bool, env: Environment, solution: str) -> RunResult:
    """Grade *solution* as a caller would: a run that raises gets no verdict."""
    try:
        if use_async:
            return asyncio.run(AsyncDeepGym(mode=mode).run(env, solution))
        return DeepGym(mode=mode).run(env, solution)
    except DeepGymError as exc:
        return error_result(exc)


@pytest.mark.parametrize('use_async', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize(('shape', 'verifier_mode'), SHAPES, ids=[f'{s}-{m}' for s, m in SHAPES])
@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
def test_single_runs_attribute_failures_alike(
    mode: str, scenario: str, shape: str, verifier_mode: str, use_async: bool
) -> None:
    statements, solution, expected = SCENARIOS[scenario]
    env = Environment(
        task='t',
        verifier_code=_single_turn(statements, shape),
        verifier_mode=verifier_mode,
        case_timeout=1,
    )

    assert _outcome(_grade(mode, use_async, env, solution)) == expected


# A verifier that leaves a solution's timeout to the harness, as reported against main.
_TIMEOUT_REPRO = (
    'from deepgym_solution import load_solution\n\n\n'
    'def verify(p, t=None):\n'
    '    return float(load_solution(p).solve() == 42)\n'
)
_TIMEOUT_REPRO_MODES = {
    'verify_function': _TIMEOUT_REPRO,
    'auto': _TIMEOUT_REPRO,
    'script': _with_main(_TIMEOUT_REPRO, 'script'),
    'body': 'from deepgym_solution import load_solution\n'
    'return float(load_solution(solution_path).solve() == 42)\n',
}


@pytest.mark.parametrize('use_async', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('verifier_mode', sorted(_TIMEOUT_REPRO_MODES))
def test_an_uncaught_solution_timeout_is_a_judged_zero(
    mode: str, verifier_mode: str, use_async: bool
) -> None:
    env = Environment(
        task='t',
        verifier_code=_TIMEOUT_REPRO_MODES[verifier_mode],
        verifier_mode=verifier_mode,
        case_timeout=1,
    )

    result = _grade(mode, use_async, env, 'def solve():\n    while True:\n        pass\n')

    assert (result.score, result.has_verdict, result.error_type) == (0.0, True, 'solution_timeout')


def _write_agent(code: str) -> Action:
    return Action(action_type='write_file', path='agent.py', content=code)


def _session_env(solution: str, **verifiers: str) -> MultiTurnEnvironment:
    """A two-step episode; a hanging solution gets a short step limit, which bounds the code
    a session verifier loads, so the test stays quick."""
    limit = 2 if solution is HANGS else 30
    return MultiTurnEnvironment(task='t', max_steps=2, timeout_per_step=limit, **verifiers)


def _skip_unless_it_loads_code(session: Session, expected: str) -> None:
    """Every judged outcome here needs the session verifier to load the agent's code, which
    SandboxBackend allows only on images with sudo (see its docstring)."""
    if expected != 'no_verdict' and getattr(session._backend, '_runtime_env', None) == {}:
        pytest.skip("no sudo in this image, so session verifiers can't load the agent's code")


@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
def test_final_verifiers_attribute_failures_alike(mode: str, scenario: str) -> None:
    statements, agent_code, expected = SCENARIOS[scenario]
    env = _session_env(agent_code, final_verifier_code=_session_verifier(statements))

    with DeepGym(mode=mode).session(env) as session:
        session.start()
        _skip_unless_it_loads_code(session, expected)
        session.step(_write_agent(agent_code))
        try:
            result = session.score()
        except DeepGymError as exc:
            result = error_result(exc)

    assert _outcome(result) == expected


@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
def test_step_verifiers_attribute_failures_alike(mode: str, scenario: str) -> None:
    statements, agent_code, expected = SCENARIOS[scenario]
    env = _session_env(agent_code, step_verifier_code=_session_verifier(statements))

    with DeepGym(mode=mode).session(env) as session:
        session.start()
        _skip_unless_it_loads_code(session, expected)
        with contextlib.suppress(VerifierError):
            session.step(_write_agent(agent_code))
        rewards = session.trajectory.step_rewards
        unjudged = session.done and not session.score().has_verdict

    outcome = {(): 'no_verdict', (1.0,): 'pass', (0.0,): 'judged_zero'}.get(tuple(rewards))
    if outcome == 'judged_zero' and (error := (session.step_verdict or {}).get('error_type')):
        outcome += f':{error}'
    assert outcome == expected
    assert unjudged is (expected == 'no_verdict')
