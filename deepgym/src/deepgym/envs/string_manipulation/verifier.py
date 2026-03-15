"""Verifier for the String Manipulation task.

Checks both output correctness AND implementation approach.
Solutions that rely on trivial slice reversal ([::-1]) with split/join
are penalised since the task expects loop-based character-level logic.
"""

import importlib.util
import inspect
import json
import random
import string
import sys
import time


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


def _check_implementation(source: str) -> tuple[bool, str]:
    """Check if the solution uses a genuine implementation vs trivial shortcuts.

    Returns
    -------
    tuple[bool, str]
        (is_genuine, reason).
    """
    # Count non-comment, non-blank lines in the function body
    body_lines = [
        ln.strip()
        for ln in source.splitlines()
        if ln.strip()
        and not ln.strip().startswith('#')
        and not ln.strip().startswith('def ')
        and not ln.strip().startswith('"""')
        and not ln.strip().startswith("'''")
    ]

    # Trivial one/two-liner using [::-1] with split/join
    uses_slice_reverse = '[::-1]' in source
    uses_split = '.split(' in source
    if uses_slice_reverse and uses_split and len(body_lines) <= 4:
        return False, 'Trivial split+slice-reverse one-liner'

    # Pure [::-1] without any loop-based logic
    if uses_slice_reverse and 'for ' not in source and 'while ' not in source:
        return False, 'Uses [::-1] without loop-based reversal'

    # Must have some iterative logic (loop or character-by-character processing)
    has_loop = 'for ' in source or 'while ' in source
    if not has_loop and len(body_lines) <= 3:
        return False, 'No iterative logic found -- likely a trivial shortcut'

    return True, 'Implements character-level logic'


def verify(solution_path, test_cases_path=None):
    """Verify a string manipulation solution for correctness and implementation quality."""
    # Load the solution module
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

    if not hasattr(solution, 'transform'):
        return {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': 'Missing transform function',
        }

    # --- Implementation analysis ---
    source = inspect.getsource(solution.transform)
    is_genuine, impl_reason = _check_implementation(source)

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

    correctness_score = passed / total

    # Combine correctness (70%) and implementation quality (30%).
    if is_genuine:
        impl_score = 1.0
    else:
        impl_score = 0.0
        details.append(f'Implementation check failed: {impl_reason}')

    score = round(correctness_score * 0.7 + impl_score * 0.3, 4)
    if not is_genuine:
        score = min(score, 0.3)

    return {
        'schema_version': '1.0',
        'score': score,
        'passed': score == 1.0,
        'details': (
            f'{passed}/{total} correctness tests passed. '
            f'Implementation: {impl_reason}. ' + ('; '.join(details) if details else '')
        ),
        'reward_components': {
            'correctness': correctness_score,
            'implementation_quality': impl_score,
        },
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
                    'details': 'Usage: verifier.py <solution_path> [test_cases_path]',
                }
            )
        )
        sys.exit(1)

    solution_path = sys.argv[1]
    test_cases_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = verify(solution_path, test_cases_path)
    except Exception as e:
        result = {
            'schema_version': '1.0',
            'score': 0.0,
            'passed': False,
            'cases': [],
            'details': f'Verifier error: {type(e).__name__}: {e}',
        }

    print(json.dumps(result))
