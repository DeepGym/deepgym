"""Verifier for the Group Anagrams task."""

import importlib.util
import json
import random
import sys
import time
from collections import defaultdict


def ref_group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    result = [sorted(g) for g in groups.values()]
    result.sort(key=lambda g: g[0])
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

    if not hasattr(solution, 'group_anagrams'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing group_anagrams function',
        }

    test_cases = [
        (
            ['eat', 'tea', 'tan', 'ate', 'nat', 'bat'],
            [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']],
        ),
        ([''], [['']]),
        (['a'], [['a']]),
        ([], []),
        (['abc', 'bca', 'cab', 'xyz', 'zyx'], [['abc', 'bca', 'cab'], ['xyz', 'zyx']]),
        (['aa', 'aa'], [['aa', 'aa']]),
        (['ab', 'ba', 'cd', 'dc', 'ef'], [['ab', 'ba'], ['cd', 'dc'], ['ef']]),
        (['a', 'b', 'c'], [['a'], ['b'], ['c']]),
    ]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        strs = []
        for _ in range(rng.randint(5, 50)):
            length = rng.randint(1, 10)
            s = ''.join(rng.choices('abcdefghij', k=length))
            strs.append(s)
        test_cases.append((strs, ref_group_anagrams(strs)))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (strs, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': str(strs)[:500],
            'expected_summary': str(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.group_anagrams(strs[:])
            elapsed = time.monotonic() - start
            case['execution_time_ms'] = elapsed * 1000

            if elapsed > 5.0:
                details.append(f'Test {i}: exceeded 5s time limit')
                case['passed'] = False
                case['score'] = 0.0
                case['error'] = 'exceeded 5s time limit'
            else:
                # Normalize: sort each group, then sort groups by first element
                try:
                    normalized = [sorted(g) for g in result]
                    normalized.sort(key=lambda g: g[0])
                except Exception:
                    details.append(f'Test {i}: result not in expected format')
                    case['passed'] = False
                    case['score'] = 0.0
                    case['error'] = 'result not in expected format'
                    cases.append(case)
                    continue

                if normalized == expected:
                    passed += 1
                    case['passed'] = True
                    case['score'] = 1.0
                    case['actual_summary'] = str(normalized)[:500]
                else:
                    details.append(f'Test {i}: grouping mismatch')
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
