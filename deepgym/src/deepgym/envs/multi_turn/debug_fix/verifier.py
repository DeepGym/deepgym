"""Final verifier for the debug_fix multi-turn environment.

Check that buggy.py has been fixed so all tests pass.

NOTE: This environment requires MultiTurnRunner and cannot be used via the
standard single-turn dg.run() path. The verifier imports sibling files
(buggy.py, test_buggy.py) that must be present in the working directory.
It is intentionally NOT registered in registry.json.
"""

import json
import sys
from pathlib import Path

work_dir = Path(__file__).parent


def run_tests():
    """Import and run all tests, return (passed, total)."""
    # Re-import buggy from the working directory.
    sys.path.insert(0, str(work_dir))

    # Force reimport in case it was cached.
    if 'buggy' in sys.modules:
        del sys.modules['buggy']

    from buggy import sum_evens

    tests = [
        ([], 0),
        ([2, 4, 6], 12),
        ([1, 3, 5], 0),
        ([1, 2, 3, 4, 5, 6], 12),
        ([-2, -3, -4], -6),
    ]
    passed = 0
    for nums, expected in tests:
        if sum_evens(nums) == expected:
            passed += 1
    return passed, len(tests)


if __name__ == '__main__':
    passed, total = run_tests()
    score = passed / total if total > 0 else 0.0
    output = {
        'schema_version': '1.0',
        'score': score,
        'passed': passed == total,
        'cases': [],
        'details': f'{passed}/{total} tests passed',
    }
    print(json.dumps(output))
