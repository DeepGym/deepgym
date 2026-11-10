#!/usr/bin/env python3
"""Import BigCodeBench benchmark into DeepGym environment format.

Downloads the 1,140 BigCodeBench problems from HuggingFace and generates
a DeepGym-compatible environment directory for each one, including:
  - task.md (the prompt)
  - reference_solution.py (prompt + canonical solution)
  - verifier.py (wraps the BigCodeBench unittest test code)
  - metadata.json

Also generates environments/bigcodebench/registry.json indexing all problems.
"""

import ast
import json
import textwrap
from pathlib import Path


# Runs inside the isolated solution process: loads the solution, runs the
# BigCodeBench unittest code against it, prints one result line signed with the
# judge's key. The tests share the solution's process, so everything here that runs
# after the solution is imported is bound or snapshotted before it.
_RUNNER_SOURCE = """
import ast
import builtins
import collections
import hashlib
import importlib.util
import io
import json
import os
import sys
import unittest


def main():
    # Read the judge's key before any solution code runs (stdin is then empty) and bind
    # what is used after the import: patching a module or this script's globals can't
    # reach it, and the key only ever goes to C code.
    key = sys.stdin.buffer.readline().strip()
    sign, write, StringIO, deque = hashlib.blake2b, os.write, io.StringIO, collections.deque
    TestCase, TestLoader, TestSuite = unittest.TestCase, unittest.TestLoader, unittest.TestSuite
    TextTestResult, TextTestRunner = unittest.TextTestResult, unittest.TextTestRunner

    def report(kind, details):
        # A line the solution prints, or rewrites in its stdout file, has no valid tag.
        line = (kind + " " + " ".join(details.split())).encode("utf-8", "replace")
        write(1, b"\\n" + sign(line, key=key).hexdigest().encode() + b" " + line + b"\\n")

    # The methods of unittest, builtins and the test classes must keep their values
    # while the tests run: an assertion that never fails or a result that drops
    # failures is the solution's doing, and scores 0.
    watched, scopes = [], []

    def behaves(value):
        return callable(value) or hasattr(type(value), "__get__")

    def watch(*objects):
        for obj in objects:
            label, names = getattr(obj, "__qualname__", obj.__name__), vars(obj)
            watched.extend((label, names, k, v) for k, v in names.items() if behaves(v))
            is_class = isinstance(obj, type)
            # A new name there would hide one of these: a base's method, or a builtin.
            parents = obj.__mro__[1:] if is_class else () if obj is builtins else (builtins,)
            hidden = {k for parent in parents for k, v in vars(parent).items() if behaves(v)}
            mro = obj.__mro__ if is_class else None
            scopes.append((label, obj, mro, names, set(names), hidden))

    def tampered():
        found = [
            f"{label}.{name}"
            for label, names, name, value in watched
            if names.get(name, watched) is not value
        ]
        for label, obj, mro, names, keys, hidden in scopes:
            if mro is not None and obj.__mro__ is not mro:
                found.append(f"{label}.__bases__")
            found += [f"{label}.{name}" for name in sorted((names.keys() - keys) & hidden)]
        return found

    started, changes = [], []

    class Result(TextTestResult):
        def startTest(self, test):
            started.append(test)
            super().startTest(test)

        def stopTest(self, test):
            super().stopTest(test)
            changes.append(tampered())

    with open(sys.argv[1]) as handle:
        spec_data = json.load(handle)
    entry = spec_data["entry_point"]
    sys.stdout, sys.stderr = StringIO(), StringIO()

    # The tests and their imports are trusted: check them before the solution runs,
    # so a module the sandbox lacks is no verdict instead of the solution's 0.
    # ponytail: only top-level imports are checked; imports inside test methods still
    # fail as tests, since running other test code needs names the solution defines.
    try:
        tree = ast.parse(spec_data["test_code"])
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=imports, type_ignores=[]), "<tests>", "exec"), {})
    except Exception as e:
        return report("setup", f"Test setup error: {type(e).__name__}: {e}")

    modules = [m for n, m in list(sys.modules.items()) if n.partition(".")[0] == "unittest"]
    classes = {v for m in modules for v in vars(m).values() if isinstance(v, type)}
    watch(builtins, Result, *modules, *(c for c in classes if c.__module__.startswith("unittest")))

    try:
        spec = importlib.util.spec_from_file_location("solution", sys.argv[2])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return report("fail", f"Import error: {type(e).__name__}: {e}")
    if changed := tampered():
        return report("fail", f"The solution changed {', '.join(changed)}")
    if not hasattr(mod, entry):
        return report("fail", f"Missing function: {entry}")
    task = getattr(mod, entry)

    def guarded(*args, **kwargs):
        # A change the call makes counts even if undone before its test ends (a patch
        # that restores itself on first use); a test's own mocks are there before and after.
        before = tampered()
        try:
            value = task(*args, **kwargs)
        finally:
            if changed := sorted(set(tampered()) ^ set(before)):
                changes.append(changed)
        # The tests compare what it returns, and an object of a class the solution
        # defines can claim to equal anything: such a value fails the call.
        stack, seen = [value], set()
        while stack:
            item = stack.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            if type(item).__module__ == mod.__name__:
                kind = type(item).__name__
                raise TypeError(f"{entry} returned a {kind}, a class the solution defines")
            if isinstance(item, dict):
                stack.extend([*item, *item.values()])
            elif isinstance(item, (tuple, list, set, frozenset, deque)):
                stack.extend(item)
        return value

    ns = dict(vars(mod))
    ns[entry] = guarded
    solution_names = dict(ns)
    try:
        exec(spec_data["test_code"], ns)
        # Only the classes the tests define: the solution's own would run its own tests.
        cases = [
            v for k, v in ns.items()
            if v is not solution_names.get(k) and isinstance(v, type)
            and issubclass(v, TestCase) and v is not TestCase
        ]
        watch(*cases)
        loader, suite = TestLoader(), TestSuite()
        for case in cases:
            suite.addTests(loader.loadTestsFromTestCase(case))
        result = TextTestRunner(stream=StringIO(), verbosity=0, resultclass=Result).run(suite)
    except Exception as e:
        return report("fail", f"Error: {type(e).__name__}: {e}")
    if changed := next((c for c in [*changes, tampered()] if c), None):
        return report("fail", f"The solution changed {', '.join(changed)}")

    def name(test):
        test = getattr(test, "test_case", test)  # a subtest counts for its method
        return f"{type(test).__qualname__}.{getattr(test, '_testMethodName', '')}"

    failed = {name(test) for test, _ in result.failures + result.errors}
    skipped = {name(test) for test, _ in result.skipped} - failed
    outcomes = {}
    for test in map(name, started):
        outcomes[test] = "fail" if test in failed else "skip" if test in skipped else "pass"
    report("ran", " ".join(f"{test}={outcome}" for test, outcome in outcomes.items()))


main()
"""


def test_names(test_code: str) -> list[str]:
    """``Class.method`` for each test method of the TestCase classes *test_code* defines."""
    return [
        f'{node.name}.{item.name}'
        for node in ast.parse(test_code).body
        if isinstance(node, ast.ClassDef)
        and any(ast.unparse(base).endswith('TestCase') for base in node.bases)
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name.startswith('test')
    ]


def generate_verifier(task_id: str, entry_point: str, test_code: str, libs: list) -> str:
    """Generate a DeepGym-compatible verifier from BigCodeBench test code.

    BigCodeBench tests inspect rich objects (DataFrames, plots), so the tests
    and the solution must share a process: the isolated solution process, away
    from the judge and the host. There the runner signs its one result line with a
    key only it holds, scores a run where unittest, builtins or a test class changed
    0, and fails a call that returns an object of a class the solution defines (one
    that could equal anything). The judge counts the test methods it reads from the
    test source, so hidden, skipped or extra tests can't raise the score.

    A solution that reads the runner's memory (its frames, or the key) can still
    forge the result, as can one that hides a lying object where the runner doesn't
    look (inside a DataFrame, a generator, a mock's call arguments). Prefer
    function-level benchmarks (HumanEval, MBPP, EvalPlus) for adversarial RL.

    Raises:
        ValueError: If *test_code* defines no test methods.
    """
    tests = test_names(test_code)
    if not tests:
        raise ValueError(f'{task_id}: the test code defines no test methods')
    return f'''#!/usr/bin/env python3
"""DeepGym verifier for BigCodeBench {task_id}.

Tests run with the solution in the isolated process; see the importer docstring
for what keeps that result honest, and what a determined model can still forge.
"""
import hashlib
import json
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from deepgym_solution import run_solution

_TEST_CODE = {test_code!r}
_ENTRY_POINT = {entry_point!r}
# Every test method the task defines: one the result doesn't report as passed fails.
_TESTS = {tests!r}
_RUNNER = {_RUNNER_SOURCE!r}


def verify(solution_path):
    """Run the BigCodeBench tests with the solution in the isolated process."""
    key = secrets.token_hex(16)
    work = tempfile.mkdtemp(prefix="deepgym_bcb_")
    try:
        Path(work, "runner.py").write_text(_RUNNER, encoding="utf-8")
        spec = {{"test_code": _TEST_CODE, "entry_point": _ENTRY_POINT}}
        Path(work, "tests.json").write_text(json.dumps(spec), encoding="utf-8")
        try:
            proc = run_solution(
                str(Path(work, "runner.py")),
                [str(Path(work, "tests.json")), solution_path],
                cwd=work,
                input=key + "\\n",
                timeout=120,
                allow_subprocess=True,
            )
        except subprocess.TimeoutExpired:
            return {{"score": 0.0, "passed": False, "details": "Tests timed out"}}
        # Only the runner holds the key: any other line is the solution's.
        reports = []
        for line in proc.stdout.split("\\n"):
            tag, _, payload = line.partition(" ")
            if tag == hashlib.blake2b(payload.encode(), key=key.encode()).hexdigest():
                reports.append(payload)
        if len(reports) != 1:
            return {{"score": 0.0, "passed": False, "details": f"No result: {{proc.stderr[-500:]}}"}}
        kind, _, details = reports[0].partition(" ")
        if kind == "setup":
            raise RuntimeError(details)  # the tests failed to load before the solution ran
        if kind != "ran":
            return {{"score": 0.0, "passed": False, "details": details}}
        outcomes = dict(pair.partition("=")[::2] for pair in details.split())
        ok = sum(outcomes.get(test) == "pass" for test in _TESTS)
        skipped = sum(outcomes.get(test) == "skip" for test in _TESTS)
        if skipped == len(_TESTS):
            raise RuntimeError(f"all {{skipped}} tests were skipped")  # nothing checked: no verdict
        # A skipped or unreported test proved nothing, so it counts against the score.
        details = f"{{ok}}/{{len(_TESTS)}} tests passed, {{skipped}} skipped"
        return {{"score": ok / len(_TESTS), "passed": ok == len(_TESTS), "details": details}}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        result = verify(sys.argv[1])
    except Exception as e:
        # The judge failed, not the solution: report a verifier error, never a score.
        print(json.dumps({{"score": 0.0, "passed": False, "details": f"Verifier error: {{type(e).__name__}}: {{e}}", "error_type": "runtime_error"}}))
        sys.exit(2)
    print(json.dumps(result))
'''


def main():
    from datasets import load_dataset

    output_dir = Path.home() / ".deepgym" / "environments" / "bigcodebench"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BigCodeBench dataset from HuggingFace...")
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    print(f"Loaded {len(ds)} problems.")

    registry = []
    skipped = []

    for row in ds:
        task_id = row["task_id"].replace("/", "_")
        entry_point = row["entry_point"]
        complete_prompt = row["complete_prompt"]
        canonical_solution = row["canonical_solution"]
        test_code = row["test"]
        libs_raw = row.get("libs", "[]")
        # libs field is a string repr of a list, e.g. "['random', 'itertools']"
        if isinstance(libs_raw, str):
            try:
                libs = ast.literal_eval(libs_raw)
            except (ValueError, SyntaxError):
                libs = []
        else:
            libs = libs_raw if libs_raw else []

        env_dir = output_dir / task_id
        env_dir.mkdir(exist_ok=True)

        try:
            # task.md -- the problem prompt
            (env_dir / "task.md").write_text(complete_prompt, encoding="utf-8")

            # reference_solution.py -- prompt + canonical solution (complete runnable file)
            (env_dir / "reference_solution.py").write_text(
                complete_prompt + canonical_solution, encoding="utf-8"
            )

            # verifier.py
            verifier_src = generate_verifier(task_id, entry_point, test_code, libs)
            (env_dir / "verifier.py").write_text(verifier_src, encoding="utf-8")

            # metadata.json
            metadata = {
                "id": task_id,
                "name": entry_point,
                "difficulty": "hard",
                "domain": "coding",
                "benchmark": "bigcodebench",
                "libs": libs,
                "tags": ["bigcodebench", "function-completion"] + libs[:3],
            }
            (env_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )

            registry.append({**metadata, "path": f"environments/bigcodebench/{task_id}"})
        except Exception as e:
            skipped.append({"id": task_id, "reason": str(e)})
            print(f"  SKIP {task_id}: {e}")

    # registry.json
    (output_dir / "registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nGenerated {len(registry)} BigCodeBench environments in {output_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} problems:")
        for s in skipped:
            print(f"  {s['id']}: {s['reason']}")


if __name__ == "__main__":
    main()
