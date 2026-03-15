"""Verifier for the Longest Common Subsequence task."""

import importlib.util
import json
import random
import sys
import time


def ref_lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


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

    if not hasattr(solution, 'longest_common_subsequence'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing longest_common_subsequence function',
        }

    test_cases = [
        ('abcde', 'ace', 3),
        ('abc', 'abc', 3),
        ('abc', 'def', 0),
        ('', '', 0),
        ('a', '', 0),
        ('', 'b', 0),
        ('abcba', 'abcbcba', 5),
        ('oxcpqrsvwf', 'shmtulqrypy', 2),
        ('bsbininm', 'jmjkbkjkv', 1),
        ('aaaa', 'aa', 2),
    ]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        len1 = rng.randint(5, 80)
        len2 = rng.randint(5, 80)
        t1 = ''.join(rng.choices('abcdefghij', k=len1))
        t2 = ''.join(rng.choices('abcdefghij', k=len2))
        test_cases.append((t1, t2, ref_lcs(t1, t2)))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (t1, t2, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': f't1={t1!r:.30}, t2={t2!r:.30}'[:500],
            'expected_summary': str(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.longest_common_subsequence(t1, t2)
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
