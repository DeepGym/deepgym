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
import importlib
import importlib.util
import io
import json
import os
import sys
import types
import unittest

# Headless plots, before the tests or the solution import matplotlib: a GUI backend's
# show() can block an honest answer until the timeout.
os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    # Read the judge's key before any solution code runs (stdin is then empty) and bind
    # what is used after the import: patching a module or this script's globals can't
    # reach it, and the key only ever goes to C code.
    key = sys.stdin.buffer.readline().strip()
    sign, write, StringIO, deque = hashlib.blake2b, os.write, io.StringIO, collections.deque
    threads, gettrace, getprofile = sys._current_frames, sys.gettrace, sys.getprofile
    monitoring = getattr(sys, "monitoring", None)
    Function, Wrapped = types.FunctionType, (classmethod, staticmethod)
    native = (
        types.WrapperDescriptorType, types.MethodDescriptorType, types.GetSetDescriptorType,
        types.MemberDescriptorType, types.ClassMethodDescriptorType,
    )
    TestCase, TestLoader, TestSuite = unittest.TestCase, unittest.TestLoader, unittest.TestSuite
    TextTestResult, TextTestRunner = unittest.TextTestResult, unittest.TextTestRunner
    solution_file, workdir, gone = sys.argv[2], os.getcwd() + os.sep, object()
    HEAP, IMMUTABLE = 1 << 9, 1 << 8  # type flags: a class made at run time, one that can't change

    def report(kind, details):
        # A line the solution prints, or rewrites in its stdout file, has no valid tag.
        line = (kind + " " + " ".join(details.split())).encode("utf-8", "replace")
        write(1, b"\\n" + sign(line, key=key).hexdigest().encode() + b" " + line + b"\\n")

    # What the solution must leave alone: the methods of builtins, unittest, the tests'
    # testing helpers and the test classes (their code, defaults, closures and
    # attributes, not just which object they are); unittest's skip and expected-failure
    # flags; the running test's own attributes; and the process's threads and trace,
    # profile and monitoring hooks. A change is the solution's doing and scores 0.
    FLAGS = ("__unittest_skip__", "__unittest_skip_why__", "__unittest_expecting_failure__")
    watched, scopes, methods, current = [], [], {}, []

    def behaves(value):
        return callable(value) or hasattr(type(value), "__get__")

    def fingerprint(value, seen):
        # The objects that decide what calling *value* runs, through its wrappers.
        if type(value) in Wrapped:
            value = value.__func__
        if type(value) is property:
            return [x for f in (value.fget, value.fset, value.fdel) for x in fingerprint(f, seen)]
        if type(value) is not Function or id(value) in seen:
            return [value]
        seen.add(id(value))
        state = [value, value.__code__, value.__defaults__]
        for k, v in [*(value.__kwdefaults__ or {}).items(), *vars(value).items()]:
            state += [k, *fingerprint(v, seen)]
        for cell in value.__closure__ or ():
            try:
                state += fingerprint(cell.cell_contents, seen)
            except ValueError:  # an empty cell
                state.append(gone)
        return state

    def same(a, b):
        return len(a) == len(b) and all(x is y for x, y in zip(a, b))

    def watch(*objects):
        for obj in objects:
            label, names = getattr(obj, "__qualname__", obj.__name__), vars(obj)
            for k, v in list(names.items()):
                if behaves(v):
                    watched.append((label, names, k, v, fingerprint(v, set())))
            is_class = isinstance(obj, type)
            # A new name there would hide one of these: a base's method, or a builtin.
            parents = obj.__mro__[1:] if is_class else () if obj is builtins else (builtins,)
            hidden = {k for parent in parents for k, v in vars(parent).items() if behaves(v)}
            flags = [names.get(flag, gone) for flag in FLAGS]
            mro = obj.__mro__ if is_class else None
            scopes.append((label, obj, mro, names, set(names), hidden, flags))
            if is_class:
                methods[obj] = hidden | {k for k, v in names.items() if behaves(v)}

    def tampered():
        found = [
            f"{label}.{name}"
            for label, names, name, value, state in watched
            if names.get(name, gone) is not value or not same(fingerprint(value, set()), state)
        ]
        for label, obj, mro, names, keys, hidden, flags in scopes:
            if mro is not None and obj.__mro__ is not mro:
                found.append(f"{label}.__bases__")
            found += [f"{label}.{name}" for name in sorted((names.keys() - keys) & hidden)]
            found += [
                f"{label}.{flag}"
                for flag, old in zip(FLAGS, flags)
                if names.get(flag, gone) is not old
            ]
        if current:  # the running test's own attributes may not hide its methods
            test, keys = current
            hiding = (vars(test).keys() - keys) & methods.get(type(test), set())
            found += [f"{type(test).__qualname__}().{name}" for name in sorted(hiding)]
        found += [f"a thread ({ident})" for ident in threads() if ident not in first_threads]
        if gettrace() is not None or getprofile() is not None:
            found.append("a trace or profile function")
        if monitoring is not None and [monitoring.get_tool(i) for i in range(6)] != tools:
            found.append("sys.monitoring")
        return found

    started, succeeded, changes = [], [], []

    class Result(TextTestResult):
        def startTest(self, test):
            started.append(test)
            current[:] = [test, set(vars(test))]
            super().startTest(test)

        def addSuccess(self, test):
            succeeded.append(test)
            super().addSuccess(test)

        def stopTest(self, test):
            super().stopTest(test)
            changes.append(tampered())
            current.clear()

    def name(test):
        test = getattr(test, "test_case", test)  # a subtest counts for its method
        return f"{type(test).__qualname__}.{getattr(test, '_testMethodName', '')}"

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
    # The testing helpers of the packages they import (numpy.testing, pandas.testing).
    packages = [a.name for n in imports if isinstance(n, ast.Import) for a in n.names]
    packages += [n.module for n in imports if isinstance(n, ast.ImportFrom) and n.module]
    for package in {package.partition(".")[0] for package in packages}:
        try:
            importlib.import_module(package + ".testing")
        except Exception:
            pass  # most have none

    def helpers(name):
        parts = name.split(".")
        return parts[0] == "unittest" or "testing" in parts or "_testing" in parts

    modules = [m for n, m in list(sys.modules.items()) if m is not None and helpers(n)]
    names = {m.__name__ for m in modules}
    classes = {v for m in modules for v in vars(m).values() if isinstance(v, type)}
    first_threads = set(threads())
    tools = [monitoring.get_tool(i) for i in range(6)] if monitoring is not None else None
    watch(builtins, Result, *modules, *(c for c in classes if c.__module__ in names))
    for module in modules:  # the module itself can't be swapped, where it's imported from
        parent, _, child = module.__name__.rpartition(".")
        watched.append(("sys.modules", sys.modules, module.__name__, module, [module]))
        if parent in sys.modules and vars(sys.modules[parent]).get(child) is module:
            watched.append((parent, vars(sys.modules[parent]), child, module, [module]))

    try:
        spec = importlib.util.spec_from_file_location("solution", solution_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return report("fail", f"Import error: {type(e).__name__}: {e}")
    if changed := tampered():
        return report("fail", f"Tampering by the solution: {', '.join(changed)}")
    if not hasattr(mod, entry):
        return report("fail", f"Missing function: {entry}")
    task = getattr(mod, entry)

    # Special methods that build or show an object; any other one (comparison, hash,
    # truth, arithmetic, attribute lookup) can make a returned value lie to a test.
    SHOWN = {
        "__init__", "__new__", "__repr__", "__str__", "__format__", "__init_subclass__",
        "__getnewargs__", "__getnewargs_ex__", "__reduce__", "__reduce_ex__", "__getstate__",
        "__setstate__", "__copy__", "__deepcopy__", "__del__", "__set_name__", "__class_getitem__",
    }
    verdicts = {}

    def library(function):
        # Code from a file the solution didn't write, or that a library compiled into its
        # own module (a dataclass's __eq__): not the solution's, nor code it compiled.
        filename, scope = function.__code__.co_filename, function.__globals__
        if scope is vars(mod) or filename == solution_file or filename.startswith(workdir):
            return False
        if not filename.startswith("<"):
            return True
        module = sys.modules.get(scope.get("__name__"))
        return module is not None and vars(module) is scope

    def trusted(value):
        if type(value) in Wrapped:
            value = value.__func__
        if type(value) is property:
            return all(f is None or trusted(f) for f in (value.fget, value.fset, value.fdel))
        if type(value) is Function:
            return library(value)
        kind = type(value)  # a C slot like tuple.__eq__, or an object of a library class
        return kind in native or kind.__flags__ & (HEAP | IMMUTABLE) == HEAP and honest(kind)

    def honest(cls):
        # Whether the special methods of cls, its bases and its metaclass are library code.
        if id(cls) not in verdicts:
            verdicts[id(cls)] = (cls, True)  # met again while checking it: decided below
            verdict = all(
                not c.__flags__ & HEAP or all(
                    trusted(v) for k, v in vars(c).items()
                    if k[:2] == "__" == k[-2:] and k not in SHOWN and behaves(v)
                )
                for c in (type(cls), *cls.__mro__)
            )
            verdicts[id(cls)] = (cls, verdict)
        return verdicts[id(cls)][1]

    def guarded(*args, **kwargs):
        # A change the call makes counts even if undone before its test ends (a patch
        # that restores itself on first use); a test's own mocks are there before and after.
        before = set(tampered())
        try:
            value = task(*args, **kwargs)
        finally:
            if changed := sorted(set(tampered()) - before):
                changes.append(changed)
        # The tests compare what it returns: a value whose comparisons the solution wrote
        # can claim to equal anything, so it fails the call.
        stack, seen = [value], set()
        while stack:
            item = stack.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            if not honest(type(item)):
                kind = type(item).__name__
                raise TypeError(f"{entry} returned a {kind}, whose comparisons the solution wrote")
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
        # Skipped by the tests' own decorators, before any solution call: no verdict.
        declared = {
            name(test) for test in suite
            if getattr(type(test), "__unittest_skip__", False)
            or getattr(getattr(test, test._testMethodName), "__unittest_skip__", False)
        }
        result = TextTestRunner(stream=StringIO(), verbosity=0, resultclass=Result).run(suite)
    except Exception as e:
        return report("fail", f"Error: {type(e).__name__}: {e}")
    if changed := next((c for c in [*changes, tampered()] if c), None):
        return report("fail", f"Tampering by the solution: {', '.join(changed)}")

    # Only an explicit success passes: an expected failure, an unexpected success or a
    # skip the tests didn't declare is a failure.
    failed = {name(test) for test, _ in result.failures + result.errors + result.expectedFailures}
    failed |= {name(test) for test in result.unexpectedSuccesses}
    skipped = {name(test) for test, _ in result.skipped}
    passed = {name(test) for test in succeeded}

    def outcome(test):
        if test in failed:
            return "fail"
        if test in skipped:
            return "skip" if test in declared else "fail"
        return "pass" if test in passed else "fail"

    report("ran", " ".join(f"{test}={outcome(test)}" for test in dict.fromkeys(map(name, started))))


main()
"""


def declared_tests(test_code: str) -> list[str]:
    """``Class.method`` for each test method of the TestCase classes *test_code* defines.

    A name defined twice is one test, as Python keeps only the last definition.
    """
    return list(
        dict.fromkeys(
            f'{node.name}.{item.name}'
            for node in ast.parse(test_code).body
            if isinstance(node, ast.ClassDef)
            and any(ast.unparse(base).endswith('TestCase') for base in node.bases)
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name.startswith('test')
        )
    )


def generate_verifier(task_id: str, entry_point: str, test_code: str, libs: list) -> str:
    """Generate a DeepGym-compatible verifier from BigCodeBench test code.

    BigCodeBench tests inspect rich objects (DataFrames, plots), so the tests
    and the solution must share a process: the isolated solution process, away
    from the judge and the host. There the runner signs its one result line with a
    key only it holds. It scores 0 a run where the solution changed unittest,
    builtins, the tests' testing helpers (numpy.testing, pandas.testing) or a test
    class, function code included, or left a thread, trace or profile hook behind,
    and fails a call that returns a value whose comparisons the solution wrote. The
    judge reads the test methods from the test source and scores 1.0 only when every
    one passed (BigCodeBench's pass@1), with the fraction in the details; a skip the
    tests' own decorators don't declare is a failure.

    The ceiling: a solution that reads the runner's memory (its frames, or the key)
    can still forge the result, as can one that hides a lying object where the
    runner doesn't look (inside a DataFrame, a generator, a mock it configures, a
    mock's call arguments). Prefer function-level benchmarks (HumanEval, MBPP,
    EvalPlus) for adversarial RL.

    Raises:
        ValueError: If *test_code* defines no test methods.
    """
    tests = declared_tests(test_code)
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
        if skipped := sum(outcomes.get(test) == "skip" for test in _TESTS):
            # The tests' own decorators skipped these: unchecked, whatever the solution did.
            raise RuntimeError(f"the tests skip {{skipped}} of {{len(_TESTS)}} cases")
        # All or nothing, as BigCodeBench's pass@1; an unreported test failed.
        ok = sum(outcomes.get(test) == "pass" for test in _TESTS)
        passed = ok == len(_TESTS)
        details = f"{{ok}}/{{len(_TESTS)}} tests passed"
        return {{"score": float(passed), "passed": passed, "details": details}}
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
