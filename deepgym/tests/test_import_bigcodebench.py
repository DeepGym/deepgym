"""BigCodeBench importer: a run scores 1.0 only when every test passes, a skip the solution
causes is a failure, and a solution that rigs the result line or unittest scores 0."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.judge import ALLOW_UNSANDBOXED_ENV, local_isolation_kind
from deepgym.models import Environment

scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from import_bigcodebench import declared_tests, generate_verifier  # noqa: E402

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
def dg() -> Iterator[DeepGym]:
    """Module-scoped local DeepGym client.

    The verifier lets the solution start subprocesses, which the macOS sandbox refuses, so
    there it runs unsandboxed: every forgery here happens inside the solution's own process,
    whatever contains it.
    """
    with pytest.MonkeyPatch.context() as patch:
        if local_isolation_kind() == 'macos':
            patch.setattr('deepgym.judge.local_isolation_kind', lambda: 'none')
            patch.setenv(ALLOW_UNSANDBOXED_ENV, '1')
        yield DeepGym(mode='local')


@pytest.mark.parametrize(
    ('solution', 'details'),
    [
        ('def task_func(x):\n    return x ** 2\n', '5/5 tests passed'),
        ('def task_func(x):\n    return abs(x) * x\n', '4/5 tests passed'),
        (_SKIP_NEGATIVE, '4/5 tests passed'),  # skipping the hard case used to count as a pass
    ],
)
def test_only_a_full_pass_scores(
    env: Environment, dg: DeepGym, solution: str, details: str
) -> None:
    """All or nothing, like BigCodeBench's pass@1; the fraction stays in the details."""
    result = dg.run(env, solution)

    assert result.output == details
    assert (result.score, result.passed) == (
        (1.0, True) if details == '5/5 tests passed' else (0.0, False)
    )


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

    assert (result.score, result.output) == (0.0, '2/3 tests passed')


def test_skips_the_solution_raises_score_zero(env: Environment, dg: DeepGym) -> None:
    """Raising SkipTest from its own call used to mask a failure as no verdict."""
    result = dg.run(env, _SKIP + '    raise unittest.SkipTest("no")\n')

    assert (result.score, result.passed, result.has_verdict) == (0.0, False, True), result.output


def test_skips_the_tests_declare_have_no_verdict(dg: DeepGym) -> None:
    """A test its own decorator skips checked nothing, whatever the solution does."""
    tests = _TESTS.replace('class TestCases', '@unittest.skip("needs a display")\nclass TestCases')
    env = Environment(task='t', verifier_code=generate_verifier('t', 'task_func', tests, []))

    result = dg.run(env, 'def task_func(x):\n    return x ** 2\n')

    assert (result.exit_code, result.has_verdict) == (2, False)


_HIDE_TESTS = (
    _SKIP + '    return x * x\n'
    '\n'
    '\n'
    '# The runner finds the cases through this loader, so stubbing it out\n'
    '# leaves an empty suite.\n'
    'unittest.TestLoader.loadTestsFromTestCase = lambda self, case: unittest.TestSuite()\n'
)


def test_hidden_test_cases_score_zero(env: Environment, dg: DeepGym) -> None:
    """Hiding the tests used to score 1.0 for "direct assertion tests passed"."""
    result = dg.run(env, _HIDE_TESTS)

    assert result.score == 0.0
    assert result.passed is False


def test_missing_test_dependency_has_no_verdict(dg: DeepGym) -> None:
    """A module the tests import but the sandbox lacks used to score a correct solution 0."""
    tests = 'import deepgym_no_such_module\n' + _TESTS
    env = Environment(task='t', verifier_code=generate_verifier('t', 'task_func', tests, []))

    result = dg.run(env, 'def task_func(x):\n    return x ** 2\n')

    assert result.exit_code == 2
    assert result.has_verdict is False


_WRONG = 'def task_func(x):\n    return -1\n'

# Each one used to score above 0 with a wrong answer (most a full 1.0): the tests share the
# solution's process, so the solution could write the result line or rig unittest itself.
_FORGERIES = {
    # A fake result line, on the real stdout after the runner's own.
    'fake line at exit': (
        'import atexit, json, sys\n'
        'atexit.register(lambda: (sys.__stdout__.write(json.dumps('
        '{"score": 1.0, "passed": True}) + "\\n"), sys.__stdout__.flush()))\n' + _WRONG
    ),
    # A fake line on every descriptor it holds (a private report pipe would be one of
    # them), then exit before the runner reports.
    'fake line on every fd': (
        'import os\n'
        'for fd in range(256):\n'
        '    try:\n'
        '        os.write(fd, b\'\\n{"score": 1.0, "passed": true}\\n\')\n'
        '    except OSError:\n'
        '        pass\n'
        'os._exit(0)\n'
    ),
    # Stdout is a file the solution can read back: rewrite the runner's own report.
    'rewrite the report': (
        'import atexit, os\n'
        'def rewrite():\n'
        '    os.lseek(1, 0, os.SEEK_SET)\n'
        '    data = b""\n'
        '    while chunk := os.read(1, 65536):\n'
        '        data += chunk\n'
        '    for old, new in [(b\'"score": 0.0\', b\'"score": 1.0\'),\n'
        '                     (b\'"passed": false\', b\'"passed": true\'), (b"=fail", b"=pass")]:\n'
        '        data = data.replace(old, new)\n'
        '    os.lseek(1, 0, os.SEEK_SET)\n'
        '    os.write(1, data)\n'
        'atexit.register(rewrite)\n' + _WRONG
    ),
    # Run only the test this answer passes.
    'loader hides failing tests': (
        'import unittest\n'
        "unittest.TestLoader.getTestCaseNames = lambda self, case: ['test_zero']\n"
        'def task_func(x):\n    return 0\n'
    ),
    'runner reports a clean run': (
        'import unittest\n'
        'def run(self, suite):\n'
        '    result = unittest.TestResult()\n'
        '    result.testsRun = suite.countTestCases()\n'
        '    return result\n'
        'unittest.TextTestRunner.run = run\n' + _WRONG
    ),
    'assertion that never fails': (
        'import unittest\n'
        'unittest.TestCase.assertEqual = lambda self, first, second, msg=None: None\n' + _WRONG
    ),
    'result that drops failures': (
        'import unittest\nunittest.TestResult.addFailure = lambda self, test, err: None\n' + _WRONG
    ),
    'test methods emptied as classes are made': (
        'import unittest\n'
        'def hook(cls, **kwargs):\n'
        '    cls._class_cleanups = []  # what the hook it replaces sets\n'
        '    for name in dir(cls):\n'
        "        if name.startswith('test'):\n"
        '            setattr(cls, name, lambda self: None)\n'
        'unittest.TestCase.__init_subclass__ = classmethod(hook)\n' + _WRONG
    ),
    'test methods emptied mid-run': (
        'import unittest\n'
        'def task_func(x):\n'
        '    for cls in unittest.TestCase.__subclasses__():\n'
        '        for name in list(vars(cls)):\n'
        "            if name.startswith('test'):\n"
        '                setattr(cls, name, lambda self: None)\n'
        '    return -1\n'
    ),
    'object that equals everything': (
        'class Anything:\n'
        '    def __eq__(self, other):\n'
        '        return True\n'
        'def task_func(x):\n'
        '    return Anything()\n'
    ),
    'int that equals everything': (
        'class Sneaky(int):\n'
        '    def __eq__(self, other):\n'
        '        return True\n'
        'def task_func(x):\n'
        '    return Sneaky(0)\n'
    ),
    # The runner is __main__: swap the unittest it uses for one with a forging runner.
    'patching the judge': (
        'import __main__, types, unittest\n'
        'class Runner:\n'
        '    def __init__(self, **kwargs):\n'
        '        pass\n'
        '    def run(self, suite):\n'
        '        result = unittest.TestResult()\n'
        '        result.testsRun = suite.countTestCases()\n'
        '        return result\n'
        "fake = types.SimpleNamespace(**{**vars(unittest), 'TextTestRunner': Runner})\n"
        '__main__.unittest = fake\n' + _WRONG
    ),
    # Its own passing tests used to dilute the task's failing ones (95 of 100).
    'extra passing tests': (
        'import unittest\n'
        'class Extra(unittest.TestCase):\n'
        '    pass\n'
        'for i in range(95):\n'
        "    setattr(Extra, f'test_{i}', lambda self: None)\n" + _WRONG
    ),
    # The same function object, running other code.
    'assertion code swapped': (
        'import unittest\n'
        'noop = lambda self, first, second, msg=None: None\n'
        'unittest.TestCase.assertEqual.__code__ = noop.__code__\n' + _WRONG
    ),
    # Failed assertions become expected failures, which scoring used to ignore.
    'every test marked an expected failure': (
        'import unittest\nunittest.TestCase.__unittest_expecting_failure__ = True\n' + _WRONG
    ),
    'equals everything under a stdlib module name': (
        'class Anything:\n'
        "    __module__ = 'builtins'\n"
        '    def __eq__(self, other):\n'
        '        return True\n'
        'def task_func(x):\n'
        '    return Anything()\n'
    ),
    'equals everything through a C callable': (
        'import functools, operator\n'
        'class Anything:\n'
        "    __module__ = 'builtins'\n"
        '    __eq__ = functools.partial(operator.is_not, object())\n'
        'def task_func(x):\n'
        '    return Anything()\n'
    ),
    # A trace function rewrites each assertion's expected value to the answer.
    'trace function': (
        'import sys, unittest\n'
        'target = unittest.TestCase.assertEqual.__code__\n'
        'def tracer(frame, event, arg):\n'
        "    if event == 'call' and frame.f_code is target:\n"
        "        frame.f_locals['second'] = frame.f_locals['first']\n"
        'def task_func(x):\n'
        '    sys.settrace(tracer)\n'
        '    return -1\n'
    ),
}


@pytest.mark.parametrize('solution', _FORGERIES.values(), ids=_FORGERIES.keys())
def test_a_forged_result_scores_zero(env: Environment, dg: DeepGym, solution: str) -> None:
    result = dg.run(env, solution)

    assert (result.score, result.passed) == (0.0, False), result.output
    assert result.has_verdict


# The usual BigCodeBench shape: call first, assert on the result after.
_CALL_THEN_ASSERT = """
import unittest


class TestCases(unittest.TestCase):
    def test_two(self):
        result = task_func(2)
        self.assertEqual(result, {"square": 4})

    def test_three(self):
        result = task_func(3)
        self.assertEqual(result, {"square": 9})
"""

_LATE_FORGERIES = {
    # Patched inside the call and restored by the next assertion, before any test ends.
    'assertion patched for one use': (
        'import unittest\n'
        'original = unittest.TestCase.assertEqual\n'
        'def once(self, first, second, msg=None):\n'
        '    unittest.TestCase.assertEqual = original\n'
        'def task_func(x):\n'
        '    unittest.TestCase.assertEqual = once\n'
        '    return {}\n'
    ),
    # Its own class, inside a library dict subclass.
    'equals everything inside a defaultdict': (
        'import collections\n'
        'class Anything:\n'
        '    def __eq__(self, other):\n'
        '        return True\n'
        'def task_func(x):\n'
        '    return collections.defaultdict(int, square=Anything())\n'
    ),
    # The test's own instance, reached through the stack.
    'assertion hidden on the test instance': (
        'import sys, unittest\n'
        'def task_func(x):\n'
        '    frame = sys._getframe(1)\n'
        "    while not isinstance(frame.f_locals.get('self'), unittest.TestCase):\n"
        '        frame = frame.f_back\n'
        "    frame.f_locals['self'].assertEqual = lambda *args, **kwargs: None\n"
        '    return {}\n'
    ),
}


@pytest.mark.parametrize('solution', _LATE_FORGERIES.values(), ids=_LATE_FORGERIES.keys())
def test_a_forgery_after_the_call_scores_zero(dg: DeepGym, solution: str) -> None:
    env = Environment(
        task='t', verifier_code=generate_verifier('t', 'task_func', _CALL_THEN_ASSERT, [])
    )
    assert dg.run(env, 'def task_func(x):\n    return {"square": x * x}\n').score == 1.0

    result = dg.run(env, solution)

    assert (result.score, result.passed) == (0.0, False), result.output


def test_a_thread_left_running_scores_zero(env: Environment, dg: DeepGym) -> None:
    """A thread that outlives its call could change unittest between the runner's checks."""
    solution = (
        'import threading, time\n'
        'threading.Thread(target=time.sleep, args=(60,), daemon=True).start()\n'
        'def task_func(x):\n'
        '    return x * x\n'
    )

    result = dg.run(env, solution)

    assert (result.score, result.passed) == (0.0, False), result.output


def test_a_patched_testing_helper_scores_zero(dg: DeepGym) -> None:
    """numpy.testing and pandas.testing are watched like unittest."""
    pytest.importorskip('numpy')
    tests = (
        'import unittest\nimport numpy as np\n\n\n'
        'class TestCases(unittest.TestCase):\n'
        '    def test_square(self):\n'
        '        np.testing.assert_array_equal(task_func([1, 2]), [1, 4])\n'
    )
    env = Environment(task='t', verifier_code=generate_verifier('t', 'task_func', tests, []))
    solution = (
        'import numpy.testing\n'
        'numpy.testing.assert_array_equal = lambda *args, **kwargs: None\n'
        'def task_func(xs):\n'
        '    return xs\n'
    )

    result = dg.run(env, solution)

    assert (result.score, result.passed) == (0.0, False), result.output


# BigCodeBench/96's tests, verbatim: a list of (word, count) tuples.
_TASK_96_TESTS = """
import unittest
from unittest.mock import patch, mock_open
class TestCases(unittest.TestCase):
    def test_return_type(self):
        with patch('builtins.open', mock_open(read_data="word1,word2,word1")):
            result = task_func('dummy_path.csv', ',')
        self.assertIsInstance(result, list)
    def test_tuple_structure(self):
        with patch('builtins.open', mock_open(read_data="word1,word2,word1")):
            result = task_func('dummy_path.csv', ',')
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
    def test_word_count(self):
        with patch('builtins.open', mock_open(read_data="word1\\nword2\\nword1")):
            result = task_func('dummy_path.csv', ',')
        self.assertIn(('word1', 2), result)
        self.assertIn(('word2', 1), result)
    def test_empty_file(self):
        with patch('builtins.open', mock_open(read_data="")):
            result = task_func('dummy_path.csv', ',')
        self.assertEqual(len(result), 0)
    def test_no_repeated_words(self):
        with patch('builtins.open', mock_open(read_data="word1,word2,word3")):
            result = task_func('dummy_path.csv', ',')
        expected_counts = {('word1', 1), ('word2', 1), ('word3', 1)}
        self.assertTrue(all(pair in expected_counts for pair in result))
    def test_custom_delimiter(self):
        with patch('builtins.open', mock_open(read_data="word1;word2;word1")):
            result = task_func('dummy_path.csv', ';')
        self.assertIn(('word1', 2), result)
        self.assertIn(('word2', 1), result)
"""


def test_a_namedtuple_answer_still_passes(dg: DeepGym) -> None:
    """A tuple subclass that keeps tuple's comparisons is a fair answer (BigCodeBench/96)."""
    env = Environment(
        task='t', verifier_code=generate_verifier('t', 'task_func', _TASK_96_TESTS, [])
    )
    solution = (
        'import csv\n'
        'from collections import Counter, namedtuple\n'
        "Pair = namedtuple('Pair', 'word count')\n"
        'def task_func(csv_file, csv_delimiter):\n'
        '    with open(csv_file) as f:\n'
        '        words = [word for row in csv.reader(f, delimiter=csv_delimiter) for word in row]\n'
        '    return [Pair(*pair) for pair in Counter(words).most_common()]\n'
    )

    result = dg.run(env, solution)

    assert (result.score, result.output) == (1.0, '6/6 tests passed')


def test_a_redefined_test_method_counts_once() -> None:
    """Python keeps only the last definition of a name (BigCodeBench/804 has 8 for 5 tests)."""
    tests = (
        'class TestCases(unittest.TestCase):\n'
        '    def test_a(self): pass\n'
        '    def test_b(self): pass\n'
        '    def test_a(self): pass\n'
    )

    assert declared_tests(tests) == ['TestCases.test_a', 'TestCases.test_b']
