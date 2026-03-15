"""Verifier for the Valid Parentheses task."""

import importlib.util
import json
import random
import sys
import time


def ref_is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    return not stack


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

    if not hasattr(solution, 'is_valid'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing is_valid function',
        }

    test_cases = [
        ('()', True),
        ('()[]{}', True),
        ('(]', False),
        ('([)]', False),
        ('{[]}', True),
        ('', True),
        ('(', False),
        (')', False),
        ('((()))', True),
        ('{[()]}', True),
        ('((', False),
        (']}', False),
    ]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    pairs = [('(', ')'), ('[', ']'), ('{', '}')]
    for _ in range(15):
        # Generate a shuffled string (may or may not be valid)
        s = []
        depth = rng.randint(5, 50)
        for _ in range(depth):
            p = rng.choice(pairs)
            s.append(p[0])
            s.append(p[1])
        rng.shuffle(s)
        result_str = ''.join(s)
        test_cases.append((result_str, ref_is_valid(result_str)))

        # Generate a known-valid nested string
        valid = ''
        for _ in range(rng.randint(3, 20)):
            p = rng.choice(pairs)
            valid = p[0] + valid + p[1]
        test_cases.append((valid, True))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (s, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': repr(s)[:500],
            'expected_summary': str(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.is_valid(s)
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
                details.append(f'Test {i}: expected {expected}, got {result} for {s!r:.40}')
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
