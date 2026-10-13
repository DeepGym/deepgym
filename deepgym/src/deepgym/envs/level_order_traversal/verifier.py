"""Verifier for the Binary Tree Level Order Traversal task."""

import json
import random
import sys
import time

from deepgym_solution import load_solution


def ref_level_order(tree):
    if not tree or tree[0] is None:
        return []
    result = []
    level_start = 0
    level_size = 1
    while level_start < len(tree):
        level = []
        for i in range(level_start, min(level_start + level_size, len(tree))):
            if tree[i] is not None:
                level.append(tree[i])
        if not level:
            break
        result.append(level)
        level_start += level_size
        level_size *= 2
    return result


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

    if not hasattr(solution, 'level_order'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing level_order function',
        }

    test_cases = [
        ([3, 9, 20, None, None, 15, 7], [[3], [9, 20], [15, 7]]),
        ([1], [[1]]),
        ([], []),
        ([1, 2, 3, 4, 5, 6, 7], [[1], [2, 3], [4, 5, 6, 7]]),
        ([1, None, 2], [[1], [2]]),
        ([1, 2, None], [[1], [2]]),
        ([5, 3, 8, 1, 4, 7, 9], [[5], [3, 8], [1, 4, 7, 9]]),
        ([1, 2, 3, None, None, None, None], [[1], [2, 3]]),
        ([10], [[10]]),
        ([1, 2, 3, 4, None, None, 7], [[1], [2, 3], [4, 7]]),
    ]

    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        depth = rng.randint(2, 5)
        size = (1 << depth) - 1  # full tree size
        tree = []
        for j in range(size):
            if rng.random() < 0.75:
                tree.append(rng.randint(1, 200))
            else:
                tree.append(None)
        if tree:
            tree[0] = rng.randint(1, 200)  # root must exist
        for j in range(1, size):
            if tree[(j - 1) // 2] is None:
                tree[j] = None  # a missing node has no children
        test_cases.append((tree, ref_level_order(tree)))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (tree, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': str(tree)[:500],
            'expected_summary': str(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.level_order(tree[:])
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
