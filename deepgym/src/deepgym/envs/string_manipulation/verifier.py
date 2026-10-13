"""Verifier for the String Manipulation task.

Scores output correctness only: the task allows any implementation, so an
idiomatic split-and-slice solution earns the same reward as a loop.
"""

import json
import random
import string
import sys
import time

from deepgym_solution import load_solution


def _reference_transform(s: str) -> str:
    """Reference implementation for validation."""
    # Split on space boundaries while preserving spacing structure
    # We process each character sequence between spaces
    result = []
    word = []
    for ch in s:
        if ch == ' ':
            if word:
                result.append(''.join(reversed(word)))
                word = []
            result.append(' ')
        else:
            word.append(ch)
    if word:
        result.append(''.join(reversed(word)))
    return ''.join(result)


def verify(solution_path, test_cases_path=None):
    """Verify a string manipulation solution for correctness."""
    # Load the solution module
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

    if 'transform' not in dir(solution):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing transform function',
        }

    # Static test cases: (input, expected_output)
    test_cases = [
        ('hello world', 'olleh dlrow'),
        ('Python is fun', 'nohtyP si nuf'),
        ('', ''),
        ('a', 'a'),
        ('ab', 'ba'),
        ('  hello  world  ', '  olleh  dlrow  '),
        ('already', 'ydaerla'),
        ('racecar level', 'racecar level'),  # palindromes stay the same
        ('123 456', '321 654'),
        ('Hello World', 'olleH dlroW'),  # case preservation
        (' ', ' '),
        ('   ', '   '),
    ]

    # Add random test cases to resist hardcoded outputs
    # Use a fixed seed so random tests are deterministic and batch-comparable (GRPO)
    seed = 42
    rng = random.Random(seed)
    for _ in range(15):
        num_words = rng.randint(1, 15)
        words = []
        for _ in range(num_words):
            wlen = rng.randint(1, 20)
            word = ''.join(rng.choices(string.ascii_letters + string.digits, k=wlen))
            words.append(word)
        # Random spacing
        parts = []
        for j, w in enumerate(words):
            if j > 0:
                parts.append(' ' * rng.randint(1, 4))
            parts.append(w)
        s = ''.join(parts)
        test_cases.append((s, _reference_transform(s)))

    passed = 0
    total = len(test_cases)
    details = []
    cases = []

    for i, (input_str, expected) in enumerate(test_cases):
        case = {
            'id': f'test_{i}',
            'input_summary': repr(input_str)[:500],
            'expected_summary': repr(expected)[:500],
        }
        try:
            start = time.monotonic()
            result = solution.transform(input_str)
            elapsed = time.monotonic() - start
            case['execution_time_ms'] = elapsed * 1000

            if elapsed > 5.0:
                details.append(f'Test {i}: exceeded 5s time limit ({elapsed:.2f}s)')
                case['passed'] = False
                case['score'] = 0.0
                case['error'] = 'exceeded 5s time limit'
            elif not isinstance(result, str):
                details.append(f'Test {i}: returned {type(result).__name__}, expected str')
                case['passed'] = False
                case['score'] = 0.0
                case['error'] = f'returned {type(result).__name__}, expected str'
            elif result == expected:
                passed += 1
                case['passed'] = True
                case['score'] = 1.0
                case['actual_summary'] = repr(result)[:500]
            else:
                details.append(
                    f'Test {i}: input={input_str!r}, expected={expected!r}, got={result!r}'
                )
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
    test_cases_path = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        result = verify(sys.argv[1], test_cases_path)
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
