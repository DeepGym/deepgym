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
# BigCodeBench unittest code against it, prints one JSON result line.
_RUNNER_SOURCE = """
import ast
import importlib.util
import io
import json
import sys
import unittest


def main():
    with open(sys.argv[1]) as handle:
        spec_data = json.load(handle)
    real_stdout = sys.stdout
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()

    def report(score, passed, details, **extra):
        real_stdout.write(json.dumps({"score": score, "passed": passed, "details": details, **extra}) + "\\n")
        real_stdout.flush()

    # The tests and their imports are trusted: check them before the solution runs,
    # so a module the sandbox lacks is no verdict instead of the solution's 0.
    # ponytail: only top-level imports are checked; imports inside test methods still
    # fail as tests, since running other test code needs names the solution defines.
    try:
        tree = ast.parse(spec_data["test_code"])
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=imports, type_ignores=[]), "<tests>", "exec"), {})
    except Exception as e:
        return report(0.0, False, f"Test setup error: {type(e).__name__}: {e}", error_type="test_setup_error")

    try:
        spec = importlib.util.spec_from_file_location("solution", sys.argv[2])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return report(0.0, False, f"Import error: {type(e).__name__}: {e}")
    if not hasattr(mod, spec_data["entry_point"]):
        return report(0.0, False, f"Missing function: {spec_data['entry_point']}")
    ns = dict(vars(mod))
    try:
        exec(spec_data["test_code"], ns)
        loader, suite = unittest.TestLoader(), unittest.TestSuite()
        for obj in list(ns.values()):
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
                suite.addTests(loader.loadTestsFromTestCase(obj))
        if suite.countTestCases() == 0:
            # A real BigCodeBench task always has assertions. An empty suite means
            # the solution hid its cases (the loader is the only way they are
            # found), so this is not a pass.
            return report(0.0, False, "No tests discovered")
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        # Count test methods, like testsRun: outcomes are recorded once per subTest.
        def method(test):
            return getattr(test, "test_case", test).id()
        failed = {method(test) for test, _ in result.failures + result.errors}
        skipped = {method(test) for test, _ in result.skipped} - failed
        ok = result.testsRun - len(failed) - len(skipped)
        if not ok and not failed:
            return report(0.0, False, f"all {len(skipped)} tests were skipped", error_type="no_tests_run")
        # A skipped test proved nothing, so it counts against the score like a failure.
        report(ok / result.testsRun, ok == result.testsRun, f"{ok}/{result.testsRun} tests passed, {len(skipped)} skipped")
    except AssertionError as e:
        report(0.0, False, f"Assertion failed: {e}")
    except Exception as e:
        report(0.0, False, f"Error: {type(e).__name__}: {e}")


main()
"""


def generate_verifier(task_id: str, entry_point: str, test_code: str, libs: list) -> str:
    """Generate a DeepGym-compatible verifier from BigCodeBench test code.

    BigCodeBench tests inspect rich objects (DataFrames, plots), so the tests
    and the solution must share a process. The verifier runs both together in
    the isolated solution process (away from the judge and the host) and reads
    back one JSON result line. A determined model can still forge that line,
    so prefer function-level benchmarks (HumanEval, MBPP, EvalPlus) for
    adversarial RL.
    """
    return f'''#!/usr/bin/env python3
"""DeepGym verifier for BigCodeBench {task_id}.

Tests run with the solution in the isolated process; see the importer docstring
for why this result is forgeable by a determined model.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from deepgym_solution import run_solution

_TEST_CODE = {test_code!r}
_ENTRY_POINT = {entry_point!r}
_RUNNER = {_RUNNER_SOURCE!r}


def verify(solution_path):
    """Run the BigCodeBench tests with the solution in the isolated process."""
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
                timeout=120,
                allow_subprocess=True,
            )
        except subprocess.TimeoutExpired:
            return {{"score": 0.0, "passed": False, "details": "Tests timed out"}}
        results = [line for line in proc.stdout.splitlines() if line.startswith("{{")]
        if not results:
            return {{"score": 0.0, "passed": False, "details": f"No result: {{proc.stderr[-500:]}}"}}
        result = json.loads(results[-1])
        if result.get("error_type"):
            raise RuntimeError(result["details"])  # e.g. every test skipped: no verdict
        return result
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
