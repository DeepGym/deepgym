#!/usr/bin/env python3
"""Import MBPP benchmark into DeepGym environment format.

Downloads the 500 MBPP test problems from HuggingFace and generates
a DeepGym-compatible environment directory for each one, including:
  - task.md (the problem description)
  - reference_solution.py (the canonical code)
  - verifier.py (runs the assert-based test cases)
  - metadata.json

Also generates environments/mbpp/registry.json indexing all problems.
"""

import json
import re
from pathlib import Path

from datasets import load_dataset


def extract_entry_point(code: str) -> str:
    """Extract the first function name from the reference code."""
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else "solution"


def generate_verifier(
    task_id: str,
    entry_point: str,
    test_list: list[str],
    challenge_test_list: list[str],
    test_setup_code: str,
) -> str:
    """Generate a DeepGym-compatible verifier from MBPP test assertions.

    MBPP tests are simple assert strings (e.g. ``assert foo(1) == 2``).
    The verifier loads the solution module, injects its namespace, and
    executes each assertion.
    """
    # Combine regular and challenge tests
    all_tests = list(test_list)
    if challenge_test_list:
        all_tests.extend(challenge_test_list)

    all_tests_repr = repr(all_tests)
    test_setup_repr = repr(test_setup_code)

    return f'''#!/usr/bin/env python3
"""DeepGym verifier for MBPP_{task_id}."""
import sys
import json
import importlib.util


# MBPP test assertions
_TESTS = {all_tests_repr}
_TEST_SETUP = {test_setup_repr}


def verify(solution_path):
    """Load the solution and run MBPP test assertions against it."""
    # Load solution module
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return {{"score": 0.0, "passed": False, "details": f"Import error: {{type(e).__name__}}: {{e}}"}}

    if not hasattr(mod, {entry_point!r}):
        return {{"score": 0.0, "passed": False, "details": "Missing function: {entry_point}"}}

    # Build namespace from the solution module
    ns = dict(vars(mod))

    # Run optional test setup code
    if _TEST_SETUP:
        try:
            exec(_TEST_SETUP, ns)
        except Exception as e:
            return {{"score": 0.0, "passed": False, "details": f"Setup error: {{type(e).__name__}}: {{e}}"}}

    passed = 0
    total = len(_TESTS)
    details = []

    for i, test_code in enumerate(_TESTS):
        try:
            exec(test_code, ns)
            passed += 1
        except AssertionError as e:
            details.append(f"Test {{i}}: assertion failed: {{e}}")
        except Exception as e:
            details.append(f"Test {{i}}: {{type(e).__name__}}: {{e}}")

    score = passed / total if total > 0 else 0.0
    summary = f"{{passed}}/{{total}} passed"
    if details:
        summary += ". " + "; ".join(details)

    return {{"score": score, "passed": passed == total, "details": summary}}


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
    output_dir = Path.home() / ".deepgym" / "environments" / "mbpp"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MBPP dataset from HuggingFace...")
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    print(f"Loaded {len(ds)} problems.")

    registry = []

    for row in ds:
        task_id = row["task_id"]
        text = row["text"]
        code = row["code"]
        test_list = row["test_list"]
        test_setup_code = row.get("test_setup_code", "")
        challenge_test_list = row.get("challenge_test_list", [])

        entry_point = extract_entry_point(code)
        dir_name = f"MBPP_{task_id}"

        env_dir = output_dir / dir_name
        env_dir.mkdir(exist_ok=True)

        # task.md -- the problem description
        (env_dir / "task.md").write_text(text + "\n", encoding="utf-8")

        # reference_solution.py -- the canonical code
        (env_dir / "reference_solution.py").write_text(code, encoding="utf-8")

        # verifier.py
        verifier_src = generate_verifier(
            task_id, entry_point, test_list, challenge_test_list, test_setup_code
        )
        (env_dir / "verifier.py").write_text(verifier_src, encoding="utf-8")

        # metadata.json
        metadata = {
            "id": dir_name,
            "name": entry_point,
            "difficulty": "medium",
            "domain": "coding",
            "benchmark": "mbpp",
            "tags": ["mbpp", "function-completion"],
        }
        (env_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        registry.append({**metadata, "path": f"environments/mbpp/{dir_name}"})

    # registry.json
    (output_dir / "registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(registry)} MBPP environments in {output_dir}")


if __name__ == "__main__":
    main()
