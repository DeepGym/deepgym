"""Unit tests for EvalPlus importer verifier generator and large-integer handling."""

import sys
from pathlib import Path

import pytest

from deepgym import DeepGym
from deepgym.models import Environment

# Ensure scripts directory is importable
scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from import_evalplus import (  # noqa: E402
    _has_large_int,
    compute_expected_outputs,
    generate_evalplus_verifier,
    import_benchmark,
)


def _verifier_env(entry_point: str, inputs: list, outputs: list) -> Environment:
    code = generate_evalplus_verifier('t', entry_point, inputs, [], outputs, atol=0)
    return Environment(task='t', verifier_code=code)


class TestEvalPlusScoring:
    """Imported cases stay true to the reference, and passing means every scored test passed."""

    def test_mutating_reference_keeps_the_embedded_inputs(self) -> None:
        inputs = [[[1, 2]], [[5, 6, 7]]]
        outputs = compute_expected_outputs(
            '', 'def pop_first(xs):\n    return xs.pop(0)\n', 'pop_first', inputs
        )

        env = _verifier_env('pop_first', inputs, outputs)
        dg = DeepGym(mode='local')
        correct = dg.run(env, 'def pop_first(xs):\n    return xs[0]\n')
        wrong = dg.run(env, 'def pop_first(xs):\n    return xs[-1]\n')

        assert inputs == [[[1, 2]], [[5, 6, 7]]]
        assert outputs == [1, 5]
        assert correct.score == 1.0
        assert wrong.score == 0.0

    def test_partial_credit_is_not_a_pass(self) -> None:
        inputs = [[n] for n in range(5)]
        env = _verifier_env('square', inputs, [n * n for n in range(5)])

        result = DeepGym(mode='local').run(
            env, 'def square(n):\n    return n * n if n < 4 else 0\n'
        )

        assert result.score == pytest.approx(0.8)
        assert result.passed is False

    def test_unsendable_entry_point_is_a_scored_failure(self) -> None:
        env = _verifier_env('square', [[2]], [4])

        result = DeepGym(mode='local').run(env, 'square = []\nsquare.append(square)\n')

        assert result.score == 0.0
        assert result.has_verdict is True

    @pytest.mark.parametrize('outputs', [[1, '__REFERENCE_ERROR__'], ['__REFERENCE_ERROR__'] * 2])
    def test_incomplete_reference_outputs_are_rejected(self, outputs: list) -> None:
        with pytest.raises(ValueError, match='reference'):
            _verifier_env('f', [[1], [2]], outputs)

    @pytest.mark.parametrize('linked', [False, True])
    def test_task_with_a_reference_error_is_not_published(
        self, tmp_path: Path, linked: bool
    ) -> None:
        """A reference failing on input 1 used to drop that case, so answering 0 scored 1.0."""
        canonical = 'def f(x):\n    if x == 1:\n        import no_such_module\n    return x\n'
        problem = {
            'entry_point': 'f',
            'prompt': '',
            'canonical_solution': canonical,
            'base_input': [[0], [1]],
            'plus_input': [],
        }
        out = tmp_path / 'out'
        old = tmp_path / 'old' if linked else out / 'T_0'  # an earlier import's copy
        old.mkdir(parents=True)
        (old / 'verifier.py').write_text('')
        if linked:
            out.mkdir()
            (out / 'T_0').symlink_to(old)

        registry = import_benchmark('humaneval_plus', {'T/0': problem}, out)

        assert registry == []
        assert not (out / 'T_0').exists()
        assert not (out / 'T_0').is_symlink()


class TestEvalPlusLargeIntHandling:
    """Test that verifiers with large integer literals (>4300 digits) compress cleanly."""

    def test_has_large_int_detection(self) -> None:
        """Verify _has_large_int accurately identifies integers with >4300 digits."""
        assert not _has_large_int(42)
        assert not _has_large_int(True)
        assert not _has_large_int(False)
        assert not _has_large_int('42')
        assert not _has_large_int([1, 2, [3, 4]])
        assert not _has_large_int({'a': 10**100, 'b': [1, 2]})

        # Exactly 4200 digits: under 4300 limit
        small_huge = 10**4200
        assert not _has_large_int(small_huge)

        # 5000 digits: over 4300 limit
        large_huge = 10**5000 + 7
        assert _has_large_int(large_huge)
        assert _has_large_int(-large_huge)

        # Nested in structures
        assert _has_large_int([1, (2, {'deep': large_huge})])
        assert _has_large_int({large_huge: 'key_is_huge'})

    def test_generate_evalplus_verifier_normal_data_uncompressed(self) -> None:
        """Normal small data is embedded directly as readable repr."""
        v_code = generate_evalplus_verifier(
            task_id='test/normal',
            entry_point='add',
            base_inputs=[[1, 2]],
            plus_inputs=[[3, 4]],
            expected_outputs=[3, 7],
            atol=0,
        )
        assert '_COMPRESSED_DATA' not in v_code
        assert 'TEST_INPUTS = [[1, 2], [3, 4]]' in v_code
        assert 'EXPECTED_OUTPUTS = [3, 7]' in v_code

        # Compiles without issue
        compiled = compile(v_code, '<verifier>', 'exec')
        assert compiled is not None

    def test_generate_evalplus_verifier_large_int_compiles_and_scores(self) -> None:
        """Verifier with 5,000-digit int compresses to avoid SyntaxError and scores 1.0."""
        large_int = 10**5000 + 7
        v_code = generate_evalplus_verifier(
            task_id='test/large_int',
            entry_point='add_one',
            base_inputs=[[large_int]],
            plus_inputs=[],
            expected_outputs=[large_int + 1],
            atol=0,
        )

        # Must use compressed form rather than raw integer literals
        assert '_COMPRESSED_DATA' in v_code
        assert str(large_int) not in v_code

        # Verifier must compile cleanly without:
        # SyntaxError: Exceeds the limit (4300 digits) for integer string conversion
        compiled = compile(v_code, '<verifier>', 'exec')
        assert compiled is not None

        # Execute verifier against reference solution and assert 1.0 score
        dg = DeepGym(mode='local')
        env = Environment(task='test/large_int', verifier_code=v_code)
        solution = 'def add_one(x):\n    return x + 1\n'
        result = dg.run(env, solution)

        assert result.score == 1.0
        assert result.passed is True
        assert result.exit_code == 0
