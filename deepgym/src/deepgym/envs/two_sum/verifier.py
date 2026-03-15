"""Verifier for the Two Sum task.

Checks both output correctness AND implementation quality.
Solutions that return hardcoded values, lack function logic, or use
brute-force O(n^2) nested loops are penalised.
"""

import importlib.util
import inspect
import json
import random
import sys
import time


def _check_implementation(source: str) -> tuple[bool, str]:
    """Check if the solution has genuine algorithmic logic.

    Returns
    -------
    tuple[bool, str]
        (is_genuine, reason).
    """
    lines = [
        ln.strip() for ln in source.splitlines() if ln.strip() and not ln.strip().startswith('#')
    ]

    # Must have a loop (iterating through nums)
    has_loop = any(kw in source for kw in ['for ', 'while '])
    if not has_loop:
        return False, 'No loop found -- likely hardcoded returns'

    # Check for hardcoded return pattern: multiple "return [" with literal ints
    literal_returns = [
        ln for ln in lines if ln.startswith('return [') and any(c.isdigit() for c in ln)
    ]
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
        spec = importlib.util.spec_from_file_location('solution', solution_path)
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)
    except Exception as e:
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': f'Import error: {type(e).__name__}: {e}',
        }

    if not hasattr(solution, 'two_sum'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing two_sum function',
        }

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
    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        size = rng.randint(10, 200)
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
    cases = []

    for i, (nums, target, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': f'nums=[{len(nums)} items], target={target}'[:500],
            'expected_summary': str(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.two_sum(nums[:], target)
            elapsed = time.monotonic() - start
            case['execution_time_ms'] = elapsed * 1000

            if elapsed > 5.0:
                details.append(f'Test {i}: exceeded 5s time limit ({elapsed:.2f}s)')
                case['passed'] = False
                case['score'] = 0.0
                case['error'] = 'exceeded 5s time limit'
            elif not isinstance(result, list):
                details.append(f'Test {i}: returned {type(result).__name__}, expected list')
                case['passed'] = False
                case['score'] = 0.0
                case['error'] = f'returned {type(result).__name__}, expected list'
            elif expected is not None:
                # Static test: exact match
                if result == expected:
                    passed += 1
                    case['passed'] = True
                    case['score'] = 1.0
                    case['actual_summary'] = str(result)[:500]
                else:
                    details.append(f'Test {i}: expected {expected}, got {result}')
                    case['passed'] = False
                    case['score'] = 0.0
                    case['actual_summary'] = str(result)[:500]
            else:
                # Dynamic test: verify the answer is valid
                ok = False
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
                        ok = True
                    else:
                        details.append(f'Test {i}: returned [] but a valid pair exists')
                elif len(result) == 2:
                    ia, ib = result
                    if (
                        isinstance(ia, int)
                        and isinstance(ib, int)
                        and 0 <= ia < ib < len(nums)
                        and nums[ia] + nums[ib] == target
                    ):
                        ok = True
                    else:
                        details.append(f'Test {i}: invalid result {result} for target {target}')
                else:
                    details.append(f'Test {i}: expected 0 or 2 indices, got {len(result)}')

                if ok:
                    passed += 1
                    case['passed'] = True
                    case['score'] = 1.0
                else:
                    case['passed'] = False
                    case['score'] = 0.0
                case['actual_summary'] = str(result)[:500]
        except Exception as e:
            details.append(f'Test {i}: {type(e).__name__}: {e}')
            case['passed'] = False
            case['score'] = 0.0
            case['error'] = f'{type(e).__name__}: {e}'

        cases.append(case)

    correctness_score = passed / total

    # Combine correctness (70%) and implementation quality (30%).
    if is_genuine:
        impl_score = 1.0
    else:
        impl_score = 0.0
        details.append(f'Implementation check failed: {impl_reason}')

    score = round(correctness_score * 0.7 + impl_score * 0.3, 4)
    if not is_genuine:
        score = min(score, 0.3)

    return {
        'schema_version': '1.0',
        'score': score,
        'passed': score == 1.0,
        'details': (
            f'{passed}/{total} correctness tests passed. '
            f'Implementation: {impl_reason}. ' + ('; '.join(details) if details else '')
        ),
        'reward_components': {
            'correctness': correctness_score,
            'implementation_quality': impl_score,
        },
        'cases': cases,
        'seed': seed,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    'schema_version': '1.0',
                    'score': 0.0,
                    'passed': False,
                    'cases': [],
                    'details': 'Usage: verifier.py <solution_path> [test_cases_path]',
                }
            )
        )
        sys.exit(1)

    solution_path = sys.argv[1]
    test_cases_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = verify(solution_path, test_cases_path)
    except Exception as e:
        result = {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': f'Verifier error: {type(e).__name__}: {e}',
        }

    print(json.dumps(result))
