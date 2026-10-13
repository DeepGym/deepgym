"""BigCodeBench importer: skipped tests are not passes, and passing means every test passed."""

import sys
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.models import Environment

scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from import_bigcodebench import generate_verifier  # noqa: E402

_TESTS = """
import unittest


class TestCases(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(task_func(0), 0)

    def test_two(self):
        self.assertEqual(task_func(2), 4)

    def test_three(self):
        self.assertEqual(task_func(3), 9)

    def test_negative(self):
        self.assertEqual(task_func(-4), 16)

    def test_large(self):
        self.assertEqual(task_func(100), 10000)
"""

_SKIP = 'import unittest\n\n\ndef task_func(x):\n'
_SKIP_NEGATIVE = _SKIP + '    if x < 0:\n        raise unittest.SkipTest("no")\n    return x * x\n'


@pytest.fixture(scope='module')
def env() -> Environment:
    """A five-test BigCodeBench-style task: square a number."""
    return Environment(task='t', verifier_code=generate_verifier('t', 'task_func', _TESTS, []))


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    """Module-scoped local DeepGym client."""
    return DeepGym(mode='local')


@pytest.mark.parametrize(
    ('solution', 'score', 'passed'),
    [
        ('def task_func(x):\n    return x ** 2\n', 1.0, True),
        ('def task_func(x):\n    return abs(x) * x\n', 0.8, False),  # 4 of 5 is no longer a pass
        (_SKIP_NEGATIVE, 0.8, False),  # skipping the hard case used to count it as a pass
    ],
)
def test_score_counts_only_tests_that_passed(
    env: Environment, dg: DeepGym, solution: str, score: float, passed: bool
) -> None:
    result = dg.run(env, solution)

    assert result.score == pytest.approx(score), result.output
    assert result.passed is passed


_SUBTESTS = """
import unittest


class TestCases(unittest.TestCase):
    def test_small(self):
        for n in range(5):
            with self.subTest(n=n):
                self.assertEqual(task_func(n), n * n)

    def test_minus_two(self):
        self.assertEqual(task_func(-2), 4)

    def test_minus_three(self):
        self.assertEqual(task_func(-3), 9)
"""


def test_skipped_subtests_count_once_per_test(dg: DeepGym) -> None:
    """Five skipped subtests of one method used to give 3 - 5 = -2 passing tests."""
    env = Environment(task='t', verifier_code=generate_verifier('t', 'task_func', _SUBTESTS, []))
    solution = _SKIP + '    if x >= 0:\n        raise unittest.SkipTest("no")\n    return x * x\n'

    result = dg.run(env, solution)

    assert result.score == pytest.approx(2 / 3)
    assert result.passed is False


def test_suite_that_ran_nothing_has_no_verdict(env: Environment, dg: DeepGym) -> None:
    """Every test skipped used to score 1.0; nothing was checked, so there is no verdict."""
    result = dg.run(env, _SKIP + '    raise unittest.SkipTest("no")\n')

    assert result.exit_code == 2
    assert result.has_verdict is False
