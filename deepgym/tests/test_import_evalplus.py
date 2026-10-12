"""Unit tests for EvalPlus importer verifier generator and large-integer handling."""

import sys
from pathlib import Path

from deepgym import DeepGym
from deepgym.models import Environment

# Ensure scripts directory is importable
scripts_dir = Path(__file__).resolve().parent.parent / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from import_evalplus import _has_large_int, generate_evalplus_verifier  # noqa: E402


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
