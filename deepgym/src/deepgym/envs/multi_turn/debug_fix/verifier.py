"""Final verifier for the debug_fix multi-turn environment.

Check that buggy.py has been fixed so all tests pass. The agent's buggy.py runs
through ``load_solution``, in its own isolated process, so it can't patch this
verifier or hand back an object that equals everything.

NOTE: This environment needs a multi-turn session and cannot be used via the
standard single-turn dg.run() path. It reads buggy.py from the session's
working directory. It is intentionally NOT registered in registry.json.
"""

import json

from deepgym_solution import load_solution

TESTS = [
    ([], 0),
    ([2, 4, 6], 12),
    ([1, 3, 5], 0),
    ([1, 2, 3, 4, 5, 6], 12),
    ([-2, -3, -4], -6),
]


def run_tests():
    """Run every test against the agent's sum_evens; return (passed, total, details)."""
    try:
        solution = load_solution('buggy.py')
    except Exception as e:
        return 0, len(TESTS), f'Import error: {type(e).__name__}: {e}'
    passed = 0
    for nums, expected in TESTS:
        try:
            passed += solution.sum_evens(nums) == expected
        except Exception:
            pass
    return passed, len(TESTS), f'{passed}/{len(TESTS)} tests passed'


if __name__ == '__main__':
    passed, total, details = run_tests()
    output = {
        'schema_version': '1.0',
        'score': passed / total,
        'passed': passed == total,
        'cases': [],
        'details': details,
    }
    print(json.dumps(output))
