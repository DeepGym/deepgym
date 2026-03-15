"""Verifier for the Python sorting task.

Checks both output correctness AND implementation approach.
Solutions that simply call sorted(), list.sort(), or similar built-ins
receive a heavy penalty since the task requires implementing the algorithm.
"""

import importlib.util
import inspect
import json
import random
import sys
import time

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
    spec = importlib.util.spec_from_file_location('solution', solution_path)
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    if not hasattr(solution, 'sort_list'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing sort_list function',
        }

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

    # Add random test cases to resist hardcoded outputs
    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    large = rng.sample(range(10000), 1000)
    test_cases.append((large, sorted(large)))

    medium = [rng.randint(-500, 500) for _ in range(200)]
    test_cases.append((medium, sorted(medium)))

    for _ in range(13):
        size = rng.randint(10, 500)
        nums = [rng.randint(-1000, 1000) for _ in range(size)]
        test_cases.append((nums, sorted(nums)))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (input_list, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': f'list[{len(input_list)} items]'[:500],
            'expected_summary': str(expected[:5])[:500] + ('...' if len(expected) > 5 else ''),
        }
        try:
            start = time.monotonic()
            result = solution.sort_list(input_list.copy())
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
            elif result == expected:
                passed += 1
                case['passed'] = True
                case['score'] = 1.0
                case['actual_summary'] = str(result[:5])[:500] + ('...' if len(result) > 5 else '')
            else:
                preview_exp = str(expected[:5]) + ('...' if len(expected) > 5 else '')
                preview_got = str(result[:5]) + ('...' if len(result) > 5 else '')
                details.append(f'Test {i}: expected {preview_exp}, got {preview_got}')
                case['passed'] = False
                case['score'] = 0.0
                case['actual_summary'] = preview_got[:500]
        except Exception as e:
            details.append(f'Test {i}: {type(e).__name__}: {e}')
            case['passed'] = False
            case['score'] = 0.0
            case['error'] = f'{type(e).__name__}: {e}'

        cases.append(case)

    correctness_score = passed / total

    # Combine correctness (70%) and implementation quality (30%).
    # Trivial solutions that use built-ins are capped at 0.3.
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
