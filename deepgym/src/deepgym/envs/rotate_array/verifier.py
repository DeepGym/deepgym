"""Verifier for the Rotate Array task."""

import importlib.util
import json
import random
import sys
import time


def ref_rotate(nums, k):
    if not nums:
        return nums
    k = k % len(nums)
    return nums[-k:] + nums[:-k] if k else nums[:]


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

    if not hasattr(solution, 'rotate_array'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing rotate_array function',
        }

    test_cases = [
        ([1, 2, 3, 4, 5, 6, 7], 3, [5, 6, 7, 1, 2, 3, 4]),
        ([-1, -100, 3, 99], 2, [3, 99, -1, -100]),
        ([1, 2], 0, [1, 2]),
        ([1], 5, [1]),
        ([], 3, []),
        ([1, 2, 3], 3, [1, 2, 3]),
        ([1, 2, 3], 1, [3, 1, 2]),
        ([1, 2, 3, 4], 6, [3, 4, 1, 2]),
    ]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        size = rng.randint(5, 500)
        nums = [rng.randint(-1000, 1000) for _ in range(size)]
        k = rng.randint(0, size * 3)
        test_cases.append((nums, k, ref_rotate(nums, k)))

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
            result = solution.rotate_array(nums[:], k)
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
                details.append(f'Test {i}: expected {str(expected)[:500]}, got {str(result)[:500]}')
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
