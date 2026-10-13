"""Verifier for the Palindrome Check task."""

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

    if 'is_palindrome' not in dir(solution):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing is_palindrome function',
        }

    test_cases = [
        ('racecar', True),
        ('A man, a plan, a canal: Panama', True),
        ('hello', False),
        ('', True),
        ('a', True),
        (' ', True),
        ('ab', False),
        ('Was it a car or a cat I saw?', True),
        ("No 'x' in Nixon", True),
        ('0P', False),
    ]

    # Randomized palindrome test cases
    def ref_is_palindrome(s):
        cleaned = ''.join(c.lower() for c in s if c.isalnum())
        return cleaned == cleaned[::-1]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        length = rng.randint(5, 100)
        half = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        palindrome = half + half[::-1]
        test_cases.append((palindrome, True))
        # Generate a guaranteed non-palindrome
        non_pal = half + 'ab'
        test_cases.append((non_pal, ref_is_palindrome(non_pal)))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (inp, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': repr(inp)[:500],
            'expected_summary': str(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.is_palindrome(inp)
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
                details.append(f'Test {i}: expected {expected}, got {result} for input {inp!r:.40}')
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
