"""HumanEval importer: tests see the prompt's helpers and the entry point, nothing else."""

import sys
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.models import Environment

scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from import_humaneval import generate_verifier  # noqa: E402

# HumanEval/38 (openai/openai_humaneval): the test calls the prompt's encode_cyclic in a loop.
_PROMPT_38 = '''
def encode_cyclic(s: str):
    """returns encoded string by cycling groups of three characters."""
    # split string to groups. Each of length 3.
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    # cycle elements in each group. Unless group has fewer elements than 3.
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)


def decode_cyclic(s: str):
    """takes as input string encoded with encode_cyclic function. Returns decoded string."""
'''
_TEST_38 = """
METADATA = {}


def check(candidate):
    from random import randint, choice
    import string

    letters = string.ascii_lowercase
    for _ in range(100):
        str = ''.join(choice(letters) for i in range(randint(10, 20)))
        encoded_str = encode_cyclic(str)
        assert candidate(encoded_str) == str
"""
# HumanEval/33 (shortened): the test also calls the entry point by name.
_PROMPT_33 = '''
def sort_third(l: list):
    """Sort the values at indices divisible by three, keep the rest."""
'''
_TEST_33 = """
def check(candidate):
    assert tuple(candidate([1, 2, 3])) == tuple(sort_third([1, 2, 3]))
    assert tuple(candidate([5, 6, 3, 4, 8, 9, 2])) == tuple([2, 6, 3, 4, 8, 9, 5])
    assert tuple(candidate([5, 6, 3, 4, 8, 9, 2, 1])) == tuple([2, 6, 3, 4, 8, 9, 5, 1])
"""

_ROTATE_RIGHT = (
    'def decode_cyclic(s):\n'
    '    groups = [s[i:i + 3] for i in range(0, len(s), 3)]\n'
    "    return ''.join(g[-1] + g[:-1] if len(g) == 3 else g for g in groups)\n"
)
_FORGOT_TO_DECODE = 'def decode_cyclic(s):\n    return s\n'
_OWN_ENCODER = 'def encode_cyclic(s):\n    return s\n\n\ndef decode_cyclic(s):\n    return s\n'
_EMPTY_RANGE = 'def range(*args):\n    return []\n\n\ndef decode_cyclic(s):\n    return s\n'
# Fetching this entry point fails in the solution process (too deep to send to the judge).
_UNSENDABLE = 'decode_cyclic = []\ndecode_cyclic.append(decode_cyclic)\n'


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    """Module-scoped local DeepGym client."""
    return DeepGym(mode='local')


@pytest.mark.parametrize(
    ('solution', 'score'),
    [
        (_ROTATE_RIGHT, 1.0),
        (_FORGOT_TO_DECODE, 0.0),
        # Exporting a helper or a builtin used to replace the test's own: both passed.
        (_OWN_ENCODER, 0.0),
        (_EMPTY_RANGE, 0.0),
        (_UNSENDABLE, 0.0),  # the solution's failure, not the judge's
    ],
)
def test_solution_cannot_supply_the_tests_helpers(dg: DeepGym, solution: str, score: float) -> None:
    env = Environment(
        task=_PROMPT_38,
        verifier_code=generate_verifier('HumanEval/38', 'decode_cyclic', _PROMPT_38, _TEST_38),
    )

    result = dg.run(env, solution)

    assert result.score == score, result.output
    assert result.has_verdict is True


@pytest.mark.parametrize(
    ('solution', 'score'),
    [
        ('def sort_third(l):\n    l = list(l)\n    l[::3] = sorted(l[::3])\n    return l\n', 1.0),
        ('def sort_third(l):\n    return sorted(l)\n', 0.0),
    ],
)
def test_test_code_calls_the_candidate_by_name(dg: DeepGym, solution: str, score: float) -> None:
    env = Environment(
        task=_PROMPT_33,
        verifier_code=generate_verifier('HumanEval/33', 'sort_third', _PROMPT_33, _TEST_33),
    )

    assert dg.run(env, solution).score == score
