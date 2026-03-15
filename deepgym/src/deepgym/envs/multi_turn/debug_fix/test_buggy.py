"""Tests for the sum_evens function."""

import sys
from pathlib import Path

# Add the working directory so buggy.py can be imported.
sys.path.insert(0, str(Path(__file__).parent))

from buggy import sum_evens


def test_empty():
    assert sum_evens([]) == 0


def test_all_even():
    assert sum_evens([2, 4, 6]) == 12


def test_all_odd():
    assert sum_evens([1, 3, 5]) == 0


def test_mixed():
    assert sum_evens([1, 2, 3, 4, 5, 6]) == 12


def test_negative():
    assert sum_evens([-2, -3, -4]) == -6


if __name__ == '__main__':
    tests = [test_empty, test_all_even, test_all_odd, test_mixed, test_negative]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError:
            pass
    print(f'{passed}/{len(tests)} tests passed')
