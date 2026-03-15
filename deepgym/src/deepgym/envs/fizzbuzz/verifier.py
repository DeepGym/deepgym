"""Verifier for the FizzBuzz with Custom Rules task."""

import importlib.util
import json
import random
import sys
import time


def ref_fizzbuzz(n, rules):
    result = []
    for i in range(1, n + 1):
        s = ''
        for divisor, word in rules:
            if i % divisor == 0:
                s += word
        result.append(s if s else str(i))
    return result


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

    if not hasattr(solution, 'fizzbuzz'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing fizzbuzz function',
        }

    test_cases = [
        (5, [(3, 'Fizz'), (5, 'Buzz')], ['1', '2', 'Fizz', '4', 'Buzz']),
        (
            15,
            [(3, 'Fizz'), (5, 'Buzz')],
            [
                '1',
                '2',
                'Fizz',
                '4',
                'Buzz',
                'Fizz',
                '7',
                '8',
                'Fizz',
                'Buzz',
                '11',
                'Fizz',
                '13',
                '14',
                'FizzBuzz',
            ],
        ),
        (1, [(2, 'Even')], ['1']),
        (4, [(2, 'Even')], ['1', 'Even', '3', 'Even']),
        (6, [(2, 'A'), (3, 'B')], ['1', 'A', 'B', 'A', '5', 'AB']),
        (3, [], ['1', '2', '3']),
        (1, [(1, 'All')], ['All']),
        (10, [(7, 'Seven')], ['1', '2', '3', '4', '5', '6', 'Seven', '8', '9', '10']),
    ]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    words = ['Fizz', 'Buzz', 'Jazz', 'Pop', 'Boom']
    for _ in range(15):
        n = rng.randint(10, 200)
        num_rules = rng.randint(1, 4)
        rules = []
        for _ in range(num_rules):
            d = rng.randint(2, 10)
            w = rng.choice(words)
            rules.append((d, w))
        test_cases.append((n, rules, ref_fizzbuzz(n, rules)))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (n, rules, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': f'n={n}, rules={str(rules)[:500]}',
            'expected_summary': str(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.fizzbuzz(n, rules[:])
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
                details.append(f'Test {i}: mismatch at n={n}')
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
