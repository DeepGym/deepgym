"""Verifier for the Top K Frequent Elements task."""

import json
import random
import sys
import time
from collections import Counter

from deepgym_solution import load_solution


def ref_top_k(nums, k):
    count = Counter(nums)
    sorted_items = sorted(count.keys(), key=lambda x: (-count[x], x))
    return sorted_items[:k]


def verify(solution_path, test_cases_path=None):
    try:
        solution = load_solution(solution_path)
    except Exception as e:
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': f'Import error: {type(e).__name__}: {e}',
        }

    if not hasattr(solution, 'top_k_frequent'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing top_k_frequent function',
        }

    test_cases = [
        ([1, 1, 1, 2, 2, 3], 2, [1, 2]),
        ([1], 1, [1]),
        ([1, 2, 2, 3, 3, 3], 1, [3]),
        ([4, 4, 4, 1, 1, 2, 2, 3], 3, [4, 1, 2]),
        ([1, 1, 2, 2, 3, 3], 3, [1, 2, 3]),
        ([5, 5, 5, 5], 1, [5]),
        ([-1, -1, 2, 2, 3], 2, [-1, 2]),
        ([1, 2, 3, 4, 5], 5, [1, 2, 3, 4, 5]),
    ]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        size = rng.randint(20, 500)
        nums = [rng.randint(-100, 100) for _ in range(size)]
        k = rng.randint(1, min(10, len(set(nums))))
        test_cases.append((nums, k, ref_top_k(nums, k)))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (nums, k, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': f'nums={str(nums)[:500]}, k={k}',
            'expected_summary': str(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.top_k_frequent(nums[:], k)
            elapsed = time.monotonic() - start
            case['execution_time_ms'] = elapsed * 1000

            if elapsed > 5.0:
                details.append(f'Test {i}: exceeded 5s time limit')
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
                case['actual_summary'] = str(result)[:500]
            else:
                details.append(f'Test {i}: expected {expected}, got {result}')
                case['passed'] = False
                case['score'] = 0.0
                case['actual_summary'] = str(result)[:500]
        except Exception as e:
            details.append(f'Test {i}: {type(e).__name__}: {e}')
            case['passed'] = False
            case['score'] = 0.0
            case['error'] = f'{type(e).__name__}: {e}'

        cases.append(case)

    score = passed / total
    return {
        'schema_version': '1.0',
        'score': score,
        'passed': score == 1.0,
        'details': f'{passed}/{total} passed. ' + '; '.join(details)
        if details
        else f'{passed}/{total} passed',
        'cases': cases,
        'seed': seed,
    }


if __name__ == '__main__':
    try:
        result = verify(sys.argv[1])
    except Exception as e:
        # The judge failed, not the solution: report a verifier error, never a score.
        print(
            json.dumps(
                {
                    'schema_version': '1.0',
                    'score': 0.0,
                    'passed': False,
                    'details': f'Verifier error: {type(e).__name__}: {e}',
                    'error_type': 'runtime_error',
                }
            )
        )
        sys.exit(2)
    print(json.dumps(result))
