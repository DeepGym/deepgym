"""Verifier for the Merge Intervals task."""

import importlib.util
import json
import random
import sys
import time


def ref_merge(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0][:]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


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

    if not hasattr(solution, 'merge'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing merge function',
        }

    test_cases = [
        ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
        ([[1, 4], [4, 5]], [[1, 5]]),
        ([], []),
        ([[1, 1]], [[1, 1]]),
        ([[1, 4], [0, 4]], [[0, 4]]),
        ([[1, 4], [2, 3]], [[1, 4]]),
        ([[1, 10], [2, 3], [4, 5], [6, 7]], [[1, 10]]),
        ([[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]),
        ([[1, 3], [0, 2], [2, 4]], [[0, 4]]),
        ([[2, 3], [4, 5], [6, 7], [8, 9], [1, 10]], [[1, 10]]),
    ]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        size = rng.randint(5, 100)
        intervals = []
        for _ in range(size):
            a = rng.randint(0, 200)
            b = a + rng.randint(0, 50)
            intervals.append([a, b])
        test_cases.append((intervals, ref_merge([iv[:] for iv in intervals])))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (intervals, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': str(intervals)[:500],
            'expected_summary': str(expected)[:500],
        }
        try:
            inp = [iv[:] for iv in intervals]
            start = time.monotonic()
            result = solution.merge(inp)
            elapsed = time.monotonic() - start
            case['execution_time_ms'] = elapsed * 1000

            if elapsed > 5.0:
                details.append(f'Test {i}: exceeded 5s time limit')
                case['passed'] = False
                case['score'] = 0.0
                case['error'] = 'exceeded 5s time limit'
            else:
                # Normalize result to list of lists
                normalized = [list(iv) for iv in result]

                if normalized == expected:
                    passed += 1
                    case['passed'] = True
                    case['score'] = 1.0
                    case['actual_summary'] = str(normalized)[:500]
                else:
                    details.append(
                        f'Test {i}: expected {str(expected)[:60]}, got {str(normalized)[:60]}'
                    )
                    case['passed'] = False
                    case['score'] = 0.0
                    case['actual_summary'] = str(normalized)[:500]
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
