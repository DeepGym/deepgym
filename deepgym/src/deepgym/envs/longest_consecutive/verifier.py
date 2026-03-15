"""Verifier for the Longest Consecutive Sequence task."""

import importlib.util
import json
import random
import sys
import time


def ref_longest_consecutive(nums):
    if not nums:
        return 0
    num_set = set(nums)
    best = 0
    for n in num_set:
        if n - 1 not in num_set:
            length = 1
            while n + length in num_set:
                length += 1
            best = max(best, length)
    return best


def verify(solution_path, test_cases_path=None):
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

    if not hasattr(solution, 'longest_consecutive'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing longest_consecutive function',
        }

    test_cases = [
        ([100, 4, 200, 1, 3, 2], 4),
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9),
        ([], 0),
        ([1], 1),
        ([1, 2, 3, 4, 5], 5),
        ([10, 20, 30], 1),
        ([1, 1, 1, 1], 1),
        ([-1, 0, 1, 2], 4),
        ([5, 4, 3, 2, 1, 0, -1], 7),
        ([9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6], 7),
    ]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        size = rng.randint(10, 500)
        nums = [rng.randint(-1000, 1000) for _ in range(size)]
        test_cases.append((nums, ref_longest_consecutive(nums)))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (nums, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': f'nums={str(nums)[:500]}',
            'expected_summary': str(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.longest_consecutive(nums[:])
            elapsed = time.monotonic() - start
            case['execution_time_ms'] = elapsed * 1000

            if elapsed > 5.0:
                details.append(f'Test {i}: exceeded 5s time limit')
                case['passed'] = False
                case['score'] = 0.0
                case['error'] = 'exceeded 5s time limit'
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
    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    'schema_version': '1.0',
                    'score': 0.0,
                    'passed': False,
                    'cases': [],
                    'details': 'Usage: verifier.py <solution_path>',
                }
            )
        )
        sys.exit(1)

    try:
        result = verify(sys.argv[1])
    except Exception as e:
        result = {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': f'Verifier error: {type(e).__name__}: {e}',
        }

    print(json.dumps(result))
