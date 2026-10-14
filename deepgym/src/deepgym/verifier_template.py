"""Turn any verifier source into a runnable script that prints the JSON protocol.

Three verifier shapes are supported, picked by ``Environment.verifier_mode``
or detected by parsing the source (never by text search):

* ``script``: a full program with an ``if __name__ == '__main__':`` block that
  prints protocol JSON itself. Run as-is.
* ``verify_function``: a module defining ``verify(solution_path, test_cases_path=None)``
  without a main block. Its return value is normalized.
* ``body``: the body of a function taking ``(solution_path, test_cases_path)``.
  Its return value is normalized; ``from __future__`` imports are hoisted.

Normalized output: one JSON object on stdout; exit 0 passed, 1 failed, 2 verifier error.
A return value that isn't a float, bool or dict, or a dict whose ``passed`` or
``truncated`` isn't a real bool, is a verifier error, or the solution's judged 0
(``solution_error``) once its output is in play. In every shape, an uncaught exception
overrides whatever was printed before it (see :func:`guard_crashes`).
"""

from __future__ import annotations

import ast
import textwrap
from typing import Literal

from deepgym.models import VerifierMode

ResolvedMode = Literal['script', 'verify_function', 'body']

_REPORT = textwrap.dedent('''\
    import json as _dg_json
    import math as _dg_math
    import os as _dg_os
    import sys as _dg_sys
    import traceback as _dg_traceback


    class _DgInvalidScore(ValueError):
        """The verifier returned NaN, inf, or a score or case score outside [0, 1]."""


    def _dg_sub_score(value, what, bounded):
        """*value* as a float: a finite number, within [0, 1] (up to float noise) if *bounded*.

        The float is what gets printed, so the host never reads the value differently.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise _DgInvalidScore(f"{what} {value!r} is not a number") from None
        if not _dg_math.isfinite(number) or bounded and not -1e-6 <= number <= 1 + 1e-6:
            raise _DgInvalidScore(f"invalid {what} {value!r}")
        return number


    def _dg_report(run):
        """Run the verifier, normalize its result to the protocol, print it, and exit.

        A crash inside *run* is left to the crash guard, which knows whether the solution ran.
        """
        solution_path = _dg_sys.argv[1] if len(_dg_sys.argv) > 1 else "solution.py"
        test_cases_path = _dg_sys.argv[2] if len(_dg_sys.argv) > 2 else None
        if test_cases_path is None:
            result = run(solution_path)
        else:
            result = run(solution_path, test_cases_path)
        exit_code = 0
        try:
            # bool before int/float: bool is a subclass of int.
            if isinstance(result, bool):
                output = {"score": 1.0 if result else 0.0, "passed": result, "details": None}
            elif isinstance(result, (int, float)):
                output = {
                    "score": float(result),
                    "passed": float(result) >= 0.5,
                    "details": None,
                }
            elif isinstance(result, dict):
                output = {
                    "score": float(result.get("score", 0.0)),
                    "passed": result.get("passed", False),
                    "details": result.get("details", None),
                    "truncated": result.get("truncated", False),
                }
                # bool("false") is True, so only a real bool may claim a pass or a cut-off.
                for flag in ("passed", "truncated"):
                    if not isinstance(output[flag], bool):
                        raise TypeError(f"{flag} must be True or False, not {output[flag]!r}")
                for key in ("reward_components", "metrics", "seed", "cases", "error_type"):
                    if key in result:
                        output[key] = result[key]
            else:
                raise TypeError(
                    f"the verifier returned {type(result).__name__}; "
                    "expected a float, bool or dict"
                )
            output["schema_version"] = "1.0"
            # Clamp only float noise. NaN fails this check too; never turn it into reward.
            if not -1e-6 <= output["score"] <= 1 + 1e-6:
                raise _DgInvalidScore(f"invalid score {output['score']!r}; must be in [0, 1]")
            # Sub-scores too, as the host checks them, so one the solution's output made
            # invalid is its judged 0 here rather than a run with no verdict there.
            for case in output.get("cases") or ():
                if not isinstance(case, dict):
                    raise _DgInvalidScore(f"case {case!r} is not an object")
                if "score" in case:
                    case["score"] = _dg_sub_score(case["score"], "case score", True)
            if output.get("reward_components") is not None:
                output["reward_components"] = {
                    name: _dg_sub_score(value, f"reward component {name!r}", False)
                    for name, value in output["reward_components"].items()
                }
            output["score"] = max(0.0, min(1.0, output["score"]))
            output.setdefault("truncated", False)
            # A line of its own, even after a partial line the verifier left unterminated.
            print("\\n" + _dg_json.dumps(output))
            exit_code = 0 if output["passed"] else 1
        except Exception as e:
            if _dg_consulted():  # the solution's output made the result invalid: its fault
                report = {"details": f"Invalid result from the solution: {e}"}
                report, exit_code = {**report, "error_type": "solution_error"}, 1
            elif isinstance(e, _DgInvalidScore):
                report = {"details": f"Verifier error: {e}", "error_type": "invalid_score"}
                exit_code = 2
            else:
                details = (
                    f"Verifier error: {type(e).__name__}: {e}\\n{_dg_traceback.format_exc()}"
                )
                report, exit_code = {"details": details, "error_type": "runtime_error"}, 2
            output = {"schema_version": "1.0", "score": 0.0, "passed": False, "truncated": False}
            print("\\n" + _dg_json.dumps({**output, **report}))
        # os._exit skips buffer flushing.
        _dg_sys.stdout.flush()
        _dg_os._exit(exit_code)
''')

_BODY_TEMPLATE = '''\
{future}#!/usr/bin/env python3
"""DeepGym verifier wrapper. Ensures structured JSON output."""
import os
import sys
import json
import traceback


def _run_verifier(solution_path, test_cases_path=None):
    """User verification logic."""
{body}


{report}

if __name__ == "__main__":
    _dg_report(_run_verifier)
'''

_VERIFY_FUNCTION_TEMPLATE = """\
{source}


{report}

if __name__ == "__main__":
    _dg_report(verify)
"""

# Python exits 1 on an uncaught exception, which is also the protocol's "failed" code, so a
# verifier that printed a score and then crashed would keep that score. This hook never
# trusts what was printed. Once the policy's output is in play (a solution process started,
# or the verifier opened one of the policy's files, see DEEPGYM_POLICY_DIR), the crash came
# from handling it, so it is the policy's failure: a judged 0 (exit 1; error_type
# solution_timeout when the solution ran out of time, else solution_error), or a solution
# could dodge its score by breaking the verifier. Before that, it is the verifier's own
# failure, or the machine's (a LaunchError: sandbox_error): no verdict (exit 2). Either way
# the hook prints the last protocol line. SystemExit raised from an excepthook still runs
# atexit handlers; a deliberate sys.exit() never reaches it.
_CRASH_GUARD = '''\
import json as _dg_json
import os as _dg_os
import sys as _dg_sys
import traceback as _dg_traceback

# Where the policy's output lives (the solution, a session's work directory): opening a
# file there, listing a folder there or starting a program there (a session verifier's own
# working folder is the agent's) counts as consulting it, as starting the solution through
# deepgym_solution does.
_dg_policy = tuple(
    _dg_os.path.join(path, "")
    for entry in _dg_os.environ.get("DEEPGYM_POLICY_DIR", "").split(_dg_os.pathsep)
    if entry
    for path in {entry, _dg_os.path.realpath(entry)}
)
_dg_touched = False
# Audit events that read the path in their first argument (None: the working folder).
_dg_reads = {"open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2"}
# Audit events that start a program, which works in the working folder (subprocess.Popen's
# cwd argument when given), with where its program and argument list are (os.system has none).
_dg_starts = {
    "subprocess.Popen": (0, 1),
    "os.system": (),
    "os.posix_spawn": (0, 1),
    "os.exec": (0, 1),
    "os.spawn": (1, 2),
}


def _dg_announce():
    """Tell the host on stdout that the policy's output is in play (DEEPGYM_CONSULTED).

    Written to the stdout file itself, so a verifier that captures sys.stdout can't hide it,
    on a line of its own after whatever the verifier printed before.
    """
    mark = _dg_os.environ.get("DEEPGYM_CONSULTED")
    if mark:
        try:
            if _dg_sys.__stdout__ is not None:
                _dg_sys.__stdout__.flush()
            _dg_os.write(1, ("\\n" + mark + "\\n").encode())
        except (OSError, ValueError):
            pass  # an audit hook that raises would fail the verifier's own call


def _dg_inside(target):
    """Whether *target* (a path; None for the working folder) is in the policy's folders."""
    try:
        path = _dg_os.path.abspath(_dg_os.fsdecode("." if target is None else target))
    except (TypeError, ValueError):
        return False  # a file descriptor, say
    return _dg_os.path.join(path, "").startswith(_dg_policy)


def _dg_audit(event, args):
    """Count reading one of the policy's files or folders, or starting a program in one or
    on one (an absolute path of the policy's among its program and arguments), as
    consulting it."""
    global _dg_touched
    if _dg_touched or not _dg_policy:
        return
    if event in _dg_reads:
        hit = _dg_inside(args[2] if event == "glob.glob/2" and args[2] is not None else args[0])
    elif event in _dg_starts:
        named = []
        for index in _dg_starts[event]:
            value = args[index]
            named += value if isinstance(value, (list, tuple)) else [value]
        absolute = []
        for arg in named:
            try:
                text = _dg_os.fsdecode(arg)  # a str, bytes or path object
            except TypeError:
                continue
            if text.startswith("/"):
                absolute.append(text)
        cwd = args[2] if event == "subprocess.Popen" else None
        hit = _dg_inside(cwd) or any(_dg_inside(arg) for arg in absolute)
    else:
        return
    if hit:
        _dg_touched = True
        _dg_announce()


_dg_sys.addaudithook(_dg_audit)


def _dg_consulted():
    """Whether the policy's output is in play: the verifier opened it or started it."""
    runtime = _dg_sys.modules.get("deepgym_solution")
    return _dg_touched or (runtime is not None and runtime.consulted())


def _dg_crashed(kind, value, tb):
    """Report an uncaught exception: a judged 0 once the policy's output is in play, else a
    verifier error."""
    trace = "".join(_dg_traceback.format_exception(kind, value, tb))
    runtime = _dg_sys.modules.get("deepgym_solution")
    launch = runtime is not None and issubclass(kind, runtime.LaunchError)
    if _dg_consulted() and not launch:
        details = f"The verifier crashed handling the solution: {kind.__name__}: {value}"
        timeout = runtime is not None and isinstance(value, runtime.SolutionTimeoutError)
        error_type = "solution_timeout" if timeout else "solution_error"  # still a verdict
        report, code = {"details": details, "error_type": error_type}, 1
    else:
        error_type = (
            "import_error" if issubclass(kind, ImportError)
            else "timeout" if issubclass(kind, TimeoutError)
            else "sandbox_error" if launch
            else "runtime_error"
        )
        report = {"details": f"Verifier error: {kind.__name__}: {value}\\n{trace}"}
        report, code = {**report, "error_type": error_type}, 2
    output = {"schema_version": "1.0", "score": 0.0, "passed": False, "truncated": False}
    try:
        # On a line of its own, even after a partial line the verifier left unterminated.
        print("\\n" + _dg_json.dumps({**output, **report}), flush=True)
    except (OSError, ValueError):
        _dg_os._exit(2)  # no verdict line can get out, so nothing printed before counts
    try:
        _dg_sys.stderr.write(trace)
        _dg_sys.stderr.flush()
    except (AttributeError, OSError, ValueError):
        # A stderr the solution broke (a full disk) must not change the verdict or exit code.
        _dg_sys.stderr = None
    raise SystemExit(code)


_dg_sys.excepthook = _dg_crashed
'''


def is_main_guard(node: ast.stmt) -> bool:
    """Return True for a top-level ``if __name__ == '__main__':`` statement."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    sides = (test.left, test.comparators[0])
    has_name = any(isinstance(side, ast.Name) and side.id == '__name__' for side in sides)
    has_main = any(isinstance(side, ast.Constant) and side.value == '__main__' for side in sides)
    return has_name and has_main


def detect_verifier_mode(source: str) -> ResolvedMode:
    """Classify verifier source by parsing it.

    Returns:
        'script' if it has a main block, 'verify_function' if it defines a
        top-level ``verify`` function, otherwise 'body'.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 'body'
    if any(is_main_guard(node) for node in tree.body):
        return 'script'
    if any(isinstance(node, ast.Return) for node in tree.body):
        return 'body'
    if any(isinstance(node, ast.FunctionDef) and node.name == 'verify' for node in tree.body):
        return 'verify_function'
    return 'body'


def split_future_imports(source: str) -> tuple[str, str]:
    """Return (future import lines, rest of source); __future__ imports must come first."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return '', source
    lines = source.splitlines()
    future_lines: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            future_lines.update(range(node.lineno - 1, (node.end_lineno or node.lineno)))
    future = ''.join(lines[i] + '\n' for i in sorted(future_lines))
    rest = '\n'.join(line for i, line in enumerate(lines) if i not in future_lines)
    return future, rest


def wrap_verifier(user_code: str) -> str:
    """Wrap a verifier function body in the standard template.

    The body receives ``solution_path`` and ``test_cases_path`` and returns a
    float in [0, 1], a bool, or a dict with ``score``/``passed``/``details``.

    Example user code::

        content = open(solution_path).read()
        return 1.0 if "def sort" in content else 0.0
    """
    future, body = split_future_imports(user_code)
    return guard_crashes(
        _BODY_TEMPLATE.format(
            future=future,
            body=textwrap.indent(body.rstrip() or 'pass', '    '),
            report=_REPORT,
        )
    )


def guard_crashes(source: str, prelude: str = '') -> str:
    """Return *source* with :data:`_CRASH_GUARD` installed before any of it runs.

    The guard goes right after *source*'s ``from __future__`` imports, which must stay
    first, and *prelude* (more setup code) right after the guard. The script keeps running
    as ``__main__`` with its own ``__file__``, so pickling, multiprocessing and paths
    relative to the script behave as before.
    """
    future, rest = split_future_imports(source)
    return f'{future}{_CRASH_GUARD}{prelude}{rest}'


def prepare_verifier(source: str, mode: VerifierMode = 'auto') -> str:
    """Return a runnable verifier script for *source* in the given (or detected) mode.

    Every shape gets :func:`guard_crashes`, so a crash never keeps a printed score.
    """
    resolved = detect_verifier_mode(source) if mode == 'auto' else mode
    if resolved == 'body':
        return wrap_verifier(source)
    if resolved == 'verify_function':
        source = _VERIFY_FUNCTION_TEMPLATE.format(source=source.rstrip(), report=_REPORT)
    return guard_crashes(source)
