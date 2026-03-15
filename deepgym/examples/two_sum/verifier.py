"""Verifier for the Two Sum task.

Checks both output correctness AND implementation quality.
Solutions that return hardcoded values, lack function logic, or use
brute-force O(n^2) nested loops are penalised.
"""
import inspect
import sys
import json
import importlib.util
import time
import random


def _check_implementation(source: str) -> tuple[bool, str]:
    """Check if the solution has genuine algorithmic logic.

    Returns
    -------
    tuple[bool, str]
        (is_genuine, reason).
    """
    lines = [ln.strip() for ln in source.splitlines() if ln.strip() and not ln.strip().startswith('#')]

    # Must have a loop (iterating through nums)
    has_loop = any(kw in source for kw in ['for ', 'while '])
    if not has_loop:
        return False, 'No loop found -- likely hardcoded returns'

    # Check for hardcoded return pattern: multiple "return [" with literal ints
    literal_returns = [ln for ln in lines if ln.startswith('return [') and any(c.isdigit() for c in ln)]
    if len(literal_returns) >= 3:
        return False, 'Multiple hardcoded return statements detected'

    # Check for brute-force O(n^2) nested loops (two for/while on the list)
    # Count loop depth: if there are nested "for" both iterating over indices/range
    nested_for_count = 0
    in_outer_for = False
    for ln in lines:
        if ln.startswith('for ') or ln.startswith('while '):
            if in_outer_for:
                nested_for_count += 1
            else:
                in_outer_for = True
        elif not ln.startswith(('if ', 'elif ', 'else:', 'return', '#')) and ':' not in ln:
            in_outer_for = False

    # A more reliable nested-loop check: look for the classic O(n^2) pattern
    if 'for ' in source and source.count('for ') >= 2:
        import re
        # Pattern: for i in range(...) ... for j in range(i+1, ...)
        if re.search(r'for\s+\w+\s+in\s+range\(.+\).*\n\s+for\s+\w+\s+in\s+range\(\s*\w+', source):
            return False, 'O(n^2) brute-force nested loops detected'

    return True, 'Implements proper algorithm'


def verify(solution_path, test_cases_path=None):
    """Verify a two_sum solution for correctness and implementation quality."""
    # Load the solution module
    try:
        spec = importlib.util.spec_from_file_location("solution", solution_path)
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)
    except Exception as e:
        return {"score": 0.0, "passed": False, "details": f"Import error: {type(e).__name__}: {e}"}

    if not hasattr(solution, 'two_sum'):
        return {"score": 0.0, "passed": False, "details": "Missing two_sum function"}

    # --- Implementation analysis ---
    source = inspect.getsource(solution.two_sum)
    is_genuine, impl_reason = _check_implementation(source)

    # Static test cases: (nums, target, expected_indices)
    test_cases = [
        # Basic cases
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        # Duplicate values
        ([3, 3], 6, [0, 1]),
        # No solution
        ([1, 2, 3], 10, []),
        # Negative numbers
        ([-1, -2, -3, -4, -5], -8, [2, 4]),
        # Mixed positive and negative
        ([-3, 4, 3, 90], 0, [0, 2]),
        # Zero involved
        ([0, 4, 3, 0], 0, [0, 3]),
        # Single element (no pair possible)
        ([5], 5, []),
        # Empty list
        ([], 10, []),
        # Larger values
        ([1000000, -1000000, 3, 7], 0, [0, 1]),
    ]

    # Add random test cases to resist hardcoded outputs
    rng = random.Random(int(time.time()))
    for _ in range(3):
        size = rng.randint(10, 100)
        nums = [rng.randint(-500, 500) for _ in range(size)]
        # Pick two random distinct indices and set a known target
        idx_a, idx_b = sorted(rng.sample(range(size), 2))
        target = nums[idx_a] + nums[idx_b]
        # We validate by checking the pair sums to target, not exact indices
        # (multiple valid pairs may exist)
        test_cases.append((nums, target, None))  # None means check dynamically

    passed = 0
    total = len(test_cases)
    details = []

    for i, (nums, target, expected) in enumerate(test_cases):
        try:
            start = time.monotonic()
            result = solution.two_sum(nums[:], target)
            elapsed = time.monotonic() - start

            if elapsed > 5.0:
                details.append(f"Test {i}: exceeded 5s time limit ({elapsed:.2f}s)")
                continue

            if not isinstance(result, list):
                details.append(f"Test {i}: returned {type(result).__name__}, expected list")
                continue

            if expected is not None:
                # Static test: exact match
                if result == expected:
                    passed += 1
                else:
                    details.append(f"Test {i}: expected {expected}, got {result}")
            else:
                # Dynamic test: verify the answer is valid
                if len(result) == 0:
                    # Check if there really is no solution
                    has_pair = False
                    for a in range(len(nums)):
                        for b in range(a + 1, len(nums)):
                            if nums[a] + nums[b] == target:
                                has_pair = True
                                break
                        if has_pair:
                            break
                    if not has_pair:
                        passed += 1
                    else:
                        details.append(f"Test {i}: returned [] but a valid pair exists")
                elif len(result) == 2:
                    ia, ib = result
                    if (isinstance(ia, int) and isinstance(ib, int)
                            and 0 <= ia < ib < len(nums)
                            and nums[ia] + nums[ib] == target):
                        passed += 1
                    else:
                        details.append(f"Test {i}: invalid result {result} for target {target}")
                else:
                    details.append(f"Test {i}: expected 0 or 2 indices, got {len(result)}")
        except Exception as e:
            details.append(f"Test {i}: {type(e).__name__}: {e}")

    correctness_score = passed / total

    # Combine correctness (70%) and implementation quality (30%).
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
