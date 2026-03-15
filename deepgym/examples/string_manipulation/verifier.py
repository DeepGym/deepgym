"""Verifier for the String Manipulation task.

Checks both output correctness AND implementation approach.
Solutions that rely on trivial slice reversal ([::-1]) with split/join
are penalised since the task expects loop-based character-level logic.
"""
import inspect
import sys
import json
import importlib.util
import time
import random
import string


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
        ln.strip() for ln in source.splitlines()
        if ln.strip() and not ln.strip().startswith('#') and not ln.strip().startswith('def ')
        and not ln.strip().startswith('"""') and not ln.strip().startswith("'''")
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
        spec = importlib.util.spec_from_file_location("solution", solution_path)
        solution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution)
    except Exception as e:
        return {"score": 0.0, "passed": False, "details": f"Import error: {type(e).__name__}: {e}"}

    if not hasattr(solution, 'transform'):
        return {"score": 0.0, "passed": False, "details": "Missing transform function"}

    # --- Implementation analysis ---
    source = inspect.getsource(solution.transform)
    is_genuine, impl_reason = _check_implementation(source)

    # Static test cases: (input, expected_output)
    test_cases = [
        ("hello world", "olleh dlrow"),
        ("Python is fun", "nohtyP si nuf"),
        ("", ""),
        ("a", "a"),
        ("ab", "ba"),
        ("  hello  world  ", "  olleh  dlrow  "),
        ("already", "ydaerla"),
        ("racecar level", "racecar level"),  # palindromes stay the same
        ("123 456", "321 654"),
        ("Hello World", "olleH dlroW"),  # case preservation
        (" ", " "),
        ("   ", "   "),
    ]

    # Add random test cases to resist hardcoded outputs
    rng = random.Random(int(time.time()))
    for _ in range(4):
        num_words = rng.randint(1, 10)
        words = []
        for _ in range(num_words):
            wlen = rng.randint(1, 15)
            word = ''.join(rng.choices(string.ascii_letters + string.digits, k=wlen))
            words.append(word)
        # Random spacing
        parts = []
        for j, w in enumerate(words):
            if j > 0:
                parts.append(' ' * rng.randint(1, 3))
            parts.append(w)
        s = ''.join(parts)
        test_cases.append((s, _reference_transform(s)))

    passed = 0
    total = len(test_cases)
    details = []

    for i, (input_str, expected) in enumerate(test_cases):
        try:
            start = time.monotonic()
            result = solution.transform(input_str)
            elapsed = time.monotonic() - start

            if elapsed > 5.0:
                details.append(f"Test {i}: exceeded 5s time limit ({elapsed:.2f}s)")
                continue

            if not isinstance(result, str):
                details.append(f"Test {i}: returned {type(result).__name__}, expected str")
                continue

            if result == expected:
                passed += 1
            else:
                details.append(f"Test {i}: input={input_str!r}, expected={expected!r}, got={result!r}")
        except Exception as e:
            details.append(f"Test {i}: {type(e).__name__}: {e}")

    correctness_score = passed / total

    # Combine correctness (70%) and implementation quality (30%).
    if is_genuine:
        impl_score = 1.0
    else:
        impl_score = 0.0
        details.append(f"Implementation check failed: {impl_reason}")

    score = round(correctness_score * 0.7 + impl_score * 0.3, 4)
    if not is_genuine:
        score = min(score, 0.3)

    return {
        "score": score,
        "passed": score >= 0.9,
        "details": (
            f"{passed}/{total} correctness tests passed. "
            f"Implementation: {impl_reason}. "
            + ("; ".join(details) if details else "")
        ),
        "reward_components": {
            "correctness": correctness_score,
            "implementation_quality": impl_score,
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"score": 0.0, "passed": False, "details": "Usage: verifier.py <solution_path> [test_cases_path]"}))
        sys.exit(1)

    solution_path = sys.argv[1]
    test_cases_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = verify(solution_path, test_cases_path)
    except Exception as e:
        result = {"score": 0.0, "passed": False, "details": f"Verifier error: {type(e).__name__}: {e}"}

    result.setdefault("schema_version", "1.0")
    print(json.dumps(result))
