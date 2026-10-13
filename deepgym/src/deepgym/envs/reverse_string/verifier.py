"""Verifier for the Reverse String task."""

import json
import random
import sys
import time

from deepgym_solution import load_solution


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

    if 'reverse_string' not in dir(solution):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing reverse_string function',
        }

    test_cases = [
        ('hello', 'olleh'),
        ('Python', 'nohtyP'),
        ('', ''),
        ('a', 'a'),
        ('abcdef', 'fedcba'),
        ('racecar', 'racecar'),
        ('12345', '54321'),
        ('  spaces  ', '  secaps  '),
    ]

    # Randomized test cases
    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        length = rng.randint(5, 200)
        s = ''.join(
            rng.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ', k=length)
        )
        test_cases.append((s, s[::-1]))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (inp, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': repr(inp)[:500],
            'expected_summary': repr(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.reverse_string(inp)
            elapsed = time.monotonic() - start
            case['execution_time_ms'] = elapsed * 1000

            if elapsed > 5.0:
                details.append(f'Test {i}: exceeded 5s time limit ({elapsed:.2f}s)')
                case['passed'] = False
                case['score'] = 0.0
                case['error'] = 'exceeded 5s time limit'
            elif result == expected:
                passed += 1
                case['passed'] = True
                case['score'] = 1.0
                case['actual_summary'] = repr(result)[:500]
            else:
                details.append(f'Test {i}: expected {expected!r:.50}, got {result!r:.50}')
                case['passed'] = False
                case['score'] = 0.0
                case['actual_summary'] = repr(result)[:500]
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
