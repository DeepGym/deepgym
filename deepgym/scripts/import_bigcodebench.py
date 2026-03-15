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

from datasets import load_dataset


def generate_verifier(task_id: str, entry_point: str, test_code: str, libs: list) -> str:
    """Generate a DeepGym-compatible verifier from BigCodeBench test code.

    BigCodeBench tests define unittest.TestCase subclasses that exercise the
    solution's entry-point function.  The verifier loads the solution module,
    injects all its public names into the test namespace, discovers TestCase
    classes, runs them, and reports results via the DeepGym JSON protocol.

    Unlike HumanEval (which uses a simple ``check(candidate)`` pattern),
    BigCodeBench tests directly call the function by name, so we inject all
    solution-level names into the exec namespace.
    """
    # Store test code safely using repr so all quotes/escapes are handled.
    test_code_repr = repr(test_code)

    return f'''#!/usr/bin/env python3
"""DeepGym verifier for BigCodeBench {task_id}."""
import sys
import json
import importlib.util
import unittest
import io
import os


# BigCodeBench test code (stored as string, executed with solution namespace).
_TEST_CODE = {test_code_repr}


def verify(solution_path):
    """Load the solution and run BigCodeBench tests against it."""
    # Load solution module
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return {{"score": 0.0, "passed": False, "details": f"Import error: {{type(e).__name__}}: {{e}}"}}

    if not hasattr(mod, {entry_point!r}):
        return {{"score": 0.0, "passed": False, "details": "Missing function: {entry_point}"}}

    # Build a namespace with all solution-level names so the test code
    # can reference the entry-point and any helpers/imports from the solution.
    ns = dict(vars(mod))

    # Suppress stdout/stderr during test execution to prevent stray prints
    # from corrupting the JSON protocol output.
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        # Execute the test code (defines TestCase classes) in the solution namespace
        exec(_TEST_CODE, ns)

        # Discover and run unittest TestCase classes
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for name, obj in ns.items():
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
                suite.addTests(loader.loadTestsFromTestCase(obj))

        if suite.countTestCases() == 0:
            # No unittest classes found -- test code uses direct assertions
            # If we got here without exception, they passed.
            return {{"score": 1.0, "passed": True, "details": "Direct assertion tests passed"}}

        # Run tests
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0)
        result = runner.run(suite)

        total = result.testsRun
        failed = len(result.failures) + len(result.errors)
        passed = total - failed
        score = passed / total if total > 0 else 0.0

        details = f"{{passed}}/{{total}} tests passed"
        if result.failures:
            details += f". Failures: {{len(result.failures)}}"
        if result.errors:
            details += f". Errors: {{len(result.errors)}}"

        return {{"score": score, "passed": score >= 0.8, "details": details}}

    except AssertionError as e:
        return {{"score": 0.0, "passed": False, "details": f"Assertion failed: {{e}}"}}
    except Exception as e:
        return {{"score": 0.0, "passed": False, "details": f"Error: {{type(e).__name__}}: {{e}}"}}
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({{"score": 0.0, "passed": False, "details": "Usage: verifier.py <solution_path>"}}))
        sys.exit(1)

    try:
        result = verify(sys.argv[1])
    except Exception as e:
        result = {{"score": 0.0, "passed": False, "details": f"Verifier error: {{type(e).__name__}}: {{e}}"}}

    print(json.dumps(result))
'''


def main():
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
