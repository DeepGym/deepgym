"""Verifier for the Python sorting task.

Checks both output correctness AND implementation approach.
Solutions that simply call sorted(), list.sort(), or similar built-ins
receive a heavy penalty since the task requires implementing the algorithm.
"""
import inspect
import sys
import json
import importlib.util
import time
import random


_BANNED_PATTERNS = ['sorted(', '.sort(', 'heapq.nsmallest', 'heapq.nlargest', 'heapq.merge']


def _check_implementation(source: str) -> tuple[bool, str]:
    """Check if the solution implements sorting vs using a built-in.

    Returns
    -------
    tuple[bool, str]
        (is_genuine, reason) -- True when the solution implements an actual
        sorting algorithm rather than delegating to a built-in.
    """
    for pattern in _BANNED_PATTERNS:
        if pattern in source:
            return False, f'Uses built-in: {pattern}'

    # A genuine sorting implementation must contain comparison operators
    # and loop constructs (for/while).
    has_comparison = any(op in source for op in ['<', '>', '<=', '>='])
    has_loop = 'for ' in source or 'while ' in source
    if not (has_comparison and has_loop):
        return False, 'No comparison/loop logic found'

    return True, 'Implements algorithm'


def verify(solution_path, test_cases_path=None):
    """Verify a sorting solution for correctness and implementation quality."""
    # Load the solution module
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    if not hasattr(solution, 'sort_list'):
        return {"score": 0.0, "passed": False, "details": "Missing sort_list function"}

    # --- Implementation analysis ---
    # Inspect the entire module source so helper functions (e.g. _merge) are
    # included in the analysis, not just the entry-point function.
    source = inspect.getsource(solution)
    is_genuine, impl_reason = _check_implementation(source)

    # Test cases
    test_cases = [
        ([], []),
        ([1], [1]),
        ([3, 1, 2], [1, 2, 3]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([1, 1, 1], [1, 1, 1]),
        ([-3, -1, -2], [-3, -2, -1]),
        ([0, -1, 1, -2, 2], [-2, -1, 0, 1, 2]),
        (list(range(100, 0, -1)), list(range(1, 101))),
    ]

    # Add random large test cases to resist hardcoded outputs
    rng = random.Random(int(time.time()))
    large = rng.sample(range(10000), 1000)
    test_cases.append((large, sorted(large)))

    medium = [rng.randint(-500, 500) for _ in range(200)]
    test_cases.append((medium, sorted(medium)))

    passed = 0
    total = len(test_cases)
    details = []

    for i, (input_list, expected) in enumerate(test_cases):
        try:
            start = time.monotonic()
            result = solution.sort_list(input_list.copy())
            elapsed = time.monotonic() - start

            if elapsed > 5.0:
                details.append(f"Test {i}: exceeded 5s time limit ({elapsed:.2f}s)")
                continue

            if not isinstance(result, list):
                details.append(f"Test {i}: returned {type(result).__name__}, expected list")
                continue

            if result == expected:
                passed += 1
            else:
                preview_exp = str(expected[:5]) + ("..." if len(expected) > 5 else "")
                preview_got = str(result[:5]) + ("..." if len(result) > 5 else "")
                details.append(f"Test {i}: expected {preview_exp}, got {preview_got}")
        except Exception as e:
            details.append(f"Test {i}: {type(e).__name__}: {e}")

    correctness_score = passed / total

    # Combine correctness (70%) and implementation quality (30%).
    # Trivial solutions that use built-ins are capped at 0.3.
    if is_genuine:
        impl_score = 1.0
    else:
        impl_score = 0.0
        details.append(f"Implementation check failed: {impl_reason}")

    score = round(correctness_score * 0.7 + impl_score * 0.3, 4)
    if not is_genuine:
        score = min(score, 0.3)

    return {
        "score": score,
        "passed": score >= 0.9,
        "details": (
            f"{passed}/{total} correctness tests passed. "
            f"Implementation: {impl_reason}. "
            + ("; ".join(details) if details else "")
        ),
        "reward_components": {
            "correctness": correctness_score,
            "implementation_quality": impl_score,
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"score": 0.0, "passed": False, "details": "Usage: verifier.py <solution_path> [test_cases_path]"}))
        sys.exit(1)

    solution_path = sys.argv[1]
    test_cases_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = verify(solution_path, test_cases_path)
    except Exception as e:
        result = {"score": 0.0, "passed": False, "details": f"Verifier error: {type(e).__name__}: {e}"}

    result.setdefault("schema_version", "1.0")
    print(json.dumps(result))
