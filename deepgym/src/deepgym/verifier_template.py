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
``truncated`` isn't a real bool, is a verifier error. In every shape, an uncaught
exception overrides whatever was printed before it (see :func:`guard_crashes`).
"""

from __future__ import annotations

import ast
import textwrap
from typing import Literal

from deepgym.models import VerifierMode

ResolvedMode = Literal['script', 'verify_function', 'body']

_REPORT = textwrap.dedent('''\
    import json as _dg_json
    import os as _dg_os
    import sys as _dg_sys
    import traceback as _dg_traceback


    class _DgInvalidScore(ValueError):
        """The verifier returned NaN, inf, or a score outside [0, 1]."""


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
            output["score"] = max(0.0, min(1.0, output["score"]))
            output.setdefault("truncated", False)
            # A line of its own, even after a partial line the verifier left unterminated.
            print("\\n" + _dg_json.dumps(output))
            exit_code = 0 if output["passed"] else 1
        except Exception as e:
            print(
                "\\n"
                + _dg_json.dumps(
                    {
                        "schema_version": "1.0",
                        "score": 0.0,
                        "passed": False,
                        "details": f"Verifier error: {e}"
                        if isinstance(e, _DgInvalidScore)
                        else (
                            f"Verifier error: {type(e).__name__}: {e}"
                            f"\\n{_dg_traceback.format_exc()}"
                        ),
                        "truncated": False,
                        "error_type": "invalid_score"
                        if isinstance(e, _DgInvalidScore)
                        else "runtime_error",
                    }
                )
            )
            exit_code = 2
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
# trusts what was printed. Once a solution process has started, the crash came from handling
# the solution, so it is the solution's failure: a judged 0 (exit 1; error_type
# solution_timeout when the solution ran out of time), or a solution could dodge its score
# by breaking the verifier. Before that, it is the verifier's own failure, or the machine's
# (a LaunchError: sandbox_error): no verdict (exit 2). Either way the hook prints the last
# protocol line. SystemExit raised from an
# excepthook still runs atexit handlers; a deliberate sys.exit() never reaches it.
_CRASH_GUARD = '''\
import json as _dg_json
import os as _dg_os
import sys as _dg_sys
import traceback as _dg_traceback


def _dg_crashed(kind, value, tb):
    """Report an uncaught exception: a judged 0 once the solution ran, else a verifier error."""
    trace = "".join(_dg_traceback.format_exception(kind, value, tb))
    runtime = _dg_sys.modules.get("deepgym_solution")
    if runtime is not None and runtime.consulted():
        details = f"The verifier crashed handling the solution: {kind.__name__}: {value}"
        report, code = {"details": details}, 1
        if isinstance(value, runtime.SolutionTimeoutError):
            report["error_type"] = "solution_timeout"  # a model fault: still a verdict
    else:
        error_type = (
            "import_error" if issubclass(kind, ImportError)
            else "timeout" if issubclass(kind, TimeoutError)
            else "sandbox_error" if runtime is not None and issubclass(kind, runtime.LaunchError)
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
