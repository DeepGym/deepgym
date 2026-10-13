"""MBPP importer: tests take only the names they call from the solution, never a builtin."""

import sys
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.models import Environment

scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from import_mbpp import generate_verifier  # noqa: E402

# MBPP task 2 (google-research-datasets/mbpp).
_SIMILAR_TESTS = [
    'assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))',
    'assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))',
]
_SIMILAR_REF = 'def similar_elements(a, b):\r\n  return tuple(set(a) & set(b))'
# MBPP task 18, shortened: the tests call the reference's second function, not its first.
_DIRTY_TESTS = [
    'assert remove_dirty_chars("probasscurve", "pros") == \'bacuve\'',
    'assert remove_dirty_chars("digitalindia", "talent") == \'digiidi\'',
]
_DIRTY_REF = (
    'def str_to_list(string): \r\n\treturn list(string) \r\n'
    'def remove_dirty_chars(string, second_string): \r\n'
    "\treturn ''.join(c for c in str_to_list(string) if c not in second_string) "
)
# MBPP task 126: the function under test shadows a builtin.
_SUM_TESTS = ['assert sum(10,15) == 6', 'assert sum(100,150) == 93', 'assert sum(4,6) == 3']
_SUM_REF = (
    'def sum(a,b): \r\n    sum = 0\r\n    for i in range (1,min(a,b)): \r\n'
    '        if (a % i == 0 and b % i == 0): \r\n            sum += i \r\n    return sum'
)
# Shaped like the sanitized MBPP split, whose tests call math from the reference's import.
_SPHERE_TESTS = ['assert math.isclose(volume_sphere(10), 4188.790204786391, rel_tol=0.001)']
_SPHERE_REF = 'import math\ndef volume_sphere(r):\n  return (4/3)*math.pi*r*r*r'

_COMMON = 'def similar_elements(a, b):\n    return tuple(x for x in a if x in b)\n'
_FIRST_ONLY = 'def similar_elements(a, b):\n    return tuple(a)\n'
_FAKE_SET = 'def set(*args):\n    return 1\n\n\ndef similar_elements(a, b):\n    return ()\n'
_FAKE_MATH = (
    'import math\n\nmath.isclose = lambda *args, **kwargs: True\n\n\n'
    'def volume_sphere(r):\n    return 0\n'
)
_SPHERE = 'from math import pi\n\n\ndef volume_sphere(r):\n    return 4 / 3 * pi * r**3\n'
_CLEAN = "def remove_dirty_chars(s, bad):\n    return ''.join(c for c in s if c not in bad)\n"
_UNCLEANED = 'def remove_dirty_chars(s, bad):\n    return s\n'
_GCD_DIVISORS = (
    'import builtins\nimport math\n\n\ndef sum(a, b):\n    g = math.gcd(a, b)\n'
    '    return builtins.sum(d for d in range(1, g + 1) if g % d == 0 and d < min(a, b))\n'
)


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    """Module-scoped local DeepGym client."""
    return DeepGym(mode='local')


@pytest.mark.parametrize(
    ('tests', 'reference', 'solution', 'score'),
    [
        (_SIMILAR_TESTS, _SIMILAR_REF, _COMMON, 1.0),
        (_SIMILAR_TESTS, _SIMILAR_REF, _FIRST_ONLY, 0.0),
        # A solution's own builtin or patched module used to replace the test's: both passed.
        (_SIMILAR_TESTS, _SIMILAR_REF, _FAKE_SET, 0.0),
        (_SPHERE_TESTS, _SPHERE_REF, _FAKE_MATH, 0.0),
        (_SPHERE_TESTS, _SPHERE_REF, _SPHERE, 1.0),
        # A correct solution without the reference's first helper used to fail as "missing".
        (_DIRTY_TESTS, _DIRTY_REF, _CLEAN, 1.0),
        (_DIRTY_TESTS, _DIRTY_REF, _UNCLEANED, 0.0),
        # The tested function itself may be named like a builtin.
        (_SUM_TESTS, _SUM_REF, _GCD_DIVISORS, 1.0),
    ],
)
def test_tests_resolve_names_from_trusted_code(
    dg: DeepGym, tests: list[str], reference: str, solution: str, score: float
) -> None:
    env = Environment(task='t', verifier_code=generate_verifier('t', tests, [], '', reference))

    result = dg.run(env, solution)

    assert result.score == score, result.output
    assert result.has_verdict is True
