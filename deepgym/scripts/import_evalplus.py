#!/usr/bin/env python3
"""Import EvalPlus (HumanEval+ and MBPP+) into DeepGym format.

EvalPlus provides 80x more tests than HumanEval and 35x more than MBPP.
This makes the verifiers significantly harder to exploit through reward hacking.

Each verifier is fully self-contained: expected outputs are computed at import
time by running the reference solution, then embedded in the verifier file.
"""

import copy
import json
import sys
import traceback
from pathlib import Path

# Some EvalPlus reference outputs are very large integers (HumanEval/83, /139)
sys.set_int_max_str_digits(0)


def _has_large_int(obj, max_digits=4300):
    """Check recursively if obj contains any integer with more than max_digits digits."""
    if isinstance(obj, bool):
        return False
    if isinstance(obj, int):
        # 10^4300 has ~14285 bits
        if obj.bit_length() < 14284:
            return False
        try:
            return len(str(abs(obj))) > max_digits
        except ValueError:
            return True
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_has_large_int(x, max_digits) for x in obj)
    if isinstance(obj, dict):
        return any(
            _has_large_int(k, max_digits) or _has_large_int(v, max_digits)
            for k, v in obj.items()
        )
    return False



def compute_expected_outputs(prompt, canonical_solution, entry_point, inputs):
    """Run the reference solution against all inputs to get expected outputs.

    Returns a list of expected outputs (one per input). If the reference
    solution itself errors on an input, that entry is set to the sentinel
    string '__REFERENCE_ERROR__' so the verifier can skip it.
    """
    ns = {}
    code = prompt + canonical_solution
    try:
        exec(compile(code, "<reference>", "exec"), ns)
    except Exception:
        # Reference solution won't even compile — return all errors
        return ["__REFERENCE_ERROR__"] * len(inputs)

    func = ns.get(entry_point)
    if func is None:
        return ["__REFERENCE_ERROR__"] * len(inputs)

    outputs = []
    for inp in inputs:
        try:
            # Copies: a reference that mutates its arguments (or returns a value it
            # mutates later) must not change the inputs and outputs we embed.
            result = func(*copy.deepcopy(inp))
            outputs.append(copy.deepcopy(result))
        except Exception:
            outputs.append("__REFERENCE_ERROR__")
    return outputs


def generate_evalplus_verifier(task_id, entry_point, base_inputs, plus_inputs,
                                expected_outputs, atol):
    """Generate a self-contained verifier with embedded expected outputs.

    The verifier tests a submitted solution against both base and plus
    (augmented) test inputs. The plus inputs are the 80x augmented tests
    that make exploitation significantly harder.

    For tasks with very large data (huge integers), test data is stored as
    compressed pickle to keep verifier files manageable.
    """
    import pickle as _pickle
    import zlib as _zlib
    import base64 as _base64

    all_inputs = base_inputs + plus_inputs
    num_base = len(base_inputs)
    num_plus = len(plus_inputs)
    total = len(all_inputs)

    # Use compressed pickle if any integer exceeds the Python compilation string digit limit
    # (4300 digits), which fails at compile time regardless of runtime sys.set_int_max_str_digits()
    use_compressed = _has_large_int(all_inputs) or _has_large_int(expected_outputs)

    if not use_compressed:
        # Use repr() for embedding — handles tuples, complex, sets, large ints
        inputs_repr = repr(all_inputs)
        outputs_repr = repr(expected_outputs)
        # If the repr is too large (>500KB), use pickle+zlib+base64 instead
        use_compressed = len(inputs_repr) + len(outputs_repr) > 500_000

    if use_compressed:
        data_blob = _pickle.dumps((all_inputs, expected_outputs))
        compressed = _zlib.compress(data_blob, 9)
        encoded = _base64.b64encode(compressed).decode('ascii')
        data_section = f"""import pickle as _pickle
import zlib as _zlib
import base64 as _base64

_COMPRESSED_DATA = {encoded!r}
_data = _pickle.loads(_zlib.decompress(_base64.b64decode(_COMPRESSED_DATA)))
TEST_INPUTS = _data[0]
EXPECTED_OUTPUTS = _data[1]
del _data, _COMPRESSED_DATA
ATOL = {atol}"""
    else:
        data_section = f"""TEST_INPUTS = {inputs_repr}
EXPECTED_OUTPUTS = {outputs_repr}
ATOL = {atol}"""

    return f'''#!/usr/bin/env python3
"""DeepGym verifier for EvalPlus {task_id}.
Tests: {num_base} base + {num_plus} augmented = {total} total.
"""
import sys
sys.set_int_max_str_digits(0)
import json
import math

from deepgym_solution import load_solution

{data_section}
NUM_BASE = {num_base}
NUM_PLUS = {num_plus}
ENTRY_POINT = {entry_point!r}


def _compare(result, expected, atol):
    """Compare result to expected, handling floats with tolerance."""
    if expected == "__REFERENCE_ERROR__":
        return True  # skip inputs where reference itself errored
    if isinstance(expected, float):
        return isinstance(result, (int, float)) and math.isclose(
            result, expected, abs_tol=atol if atol else 1e-6
        )
    if isinstance(expected, list) and isinstance(result, list):
        if len(result) != len(expected):
            return False
        return all(_compare(r, e, atol) for r, e in zip(result, expected))
    if isinstance(expected, tuple) and isinstance(result, tuple):
        if len(result) != len(expected):
            return False
        return all(_compare(r, e, atol) for r, e in zip(result, expected))
    return result == expected


def verify(solution_path):
    """Load solution and run all EvalPlus test cases against it."""
    # The solution runs in its own isolated process; calls go through a proxy.
    try:
        mod = load_solution(solution_path)
    except Exception as e:
        return {{"score": 0.0, "passed": False,
                "details": f"Import error: {{type(e).__name__}}: {{e}}"}}

    try:  # fetching it can run solution code, so a failure here is the solution's
        func = getattr(mod, ENTRY_POINT)
    except Exception as e:
        return {{"score": 0.0, "passed": False,
                "details": f"Missing function: {{ENTRY_POINT}} ({{type(e).__name__}})"}}

    passed = 0
    base_passed = 0
    plus_passed = 0
    total = len(TEST_INPUTS)
    skipped = 0
    details = []

    for i, (inp, expected) in enumerate(zip(TEST_INPUTS, EXPECTED_OUTPUTS)):
        if expected == "__REFERENCE_ERROR__":
            skipped += 1
            continue
        try:
            result = func(*inp)
            if _compare(result, expected, ATOL):
                passed += 1
                if i < NUM_BASE:
                    base_passed += 1
                else:
                    plus_passed += 1
            else:
                if len(details) < 5:
                    details.append(
                        f"Test {{i}}: expected {{repr(expected)[:80]}}, "
                        f"got {{repr(result)[:80]}}"
                    )
        except Exception as e:
            if len(details) < 5:
                details.append(f"Test {{i}}: {{type(e).__name__}}: {{e}}")

    effective_total = total - skipped
    if effective_total == 0:
        raise RuntimeError("the reference errored on every input, so nothing can be scored")
    score = passed / effective_total
    effective_base = NUM_BASE - sum(
        1 for e in EXPECTED_OUTPUTS[:NUM_BASE] if e == "__REFERENCE_ERROR__"
    )
    effective_plus = NUM_PLUS - sum(
        1 for e in EXPECTED_OUTPUTS[NUM_BASE:] if e == "__REFERENCE_ERROR__"
    )

    # The score keeps partial credit; passing means every scored test passed.
    return {{
        "score": score,
        "passed": passed == effective_total,
        "details": (
            f"{{passed}}/{{effective_total}} passed "
            f"(base: {{base_passed}}/{{effective_base}}, "
            f"plus: {{plus_passed}}/{{effective_plus}}). "
            + "; ".join(details)
        ),
        "reward_components": {{
            "base_score": base_passed / effective_base if effective_base > 0 else 0.0,
            "plus_score": plus_passed / effective_plus if effective_plus > 0 else 0.0,
        }}
    }}


if __name__ == "__main__":
    try:
        result = verify(sys.argv[1])
    except Exception as e:
        # The judge failed, not the solution: report a verifier error, never a score.
        print(json.dumps({{"score": 0.0, "passed": False,
                          "details": f"Verifier error: {{type(e).__name__}}: {{e}}",
                          "error_type": "runtime_error"}}))
        sys.exit(2)
    print(json.dumps(result))
'''


def import_benchmark(name, data, output_dir):
    """Import a single EvalPlus benchmark (HumanEval+ or MBPP+)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = []
    errors = []

    for idx, (task_id, problem) in enumerate(data.items()):
        safe_id = task_id.replace("/", "_")
        entry_point = problem["entry_point"]
        prompt = problem["prompt"]
        canonical = problem["canonical_solution"]
        base_inputs = problem.get("base_input", [])
        plus_inputs = problem.get("plus_input", [])
        # Some tasks have dict instead of list for empty plus_inputs
        if not isinstance(base_inputs, list):
            base_inputs = []
        if not isinstance(plus_inputs, list):
            plus_inputs = []
        atol = problem.get("atol", 0)

        # Compute expected outputs by running the reference solution
        all_inputs = base_inputs + plus_inputs
        expected_outputs = compute_expected_outputs(
            prompt, canonical, entry_point, all_inputs
        )

        ref_errors = sum(1 for o in expected_outputs if o == "__REFERENCE_ERROR__")
        if ref_errors > 0:
            errors.append(f"  {safe_id}: {ref_errors}/{len(all_inputs)} reference errors")

        env_dir = output_dir / safe_id
        env_dir.mkdir(exist_ok=True)

        # task.md
        (env_dir / "task.md").write_text(prompt, encoding="utf-8")

        # reference_solution.py
        # For MBPP, the prompt is a docstring, not code — prepend it as a comment
        if name == "mbpp_plus":
            ref_code = prompt.rstrip() + "\n" + canonical
        else:
            ref_code = prompt + canonical
        (env_dir / "reference_solution.py").write_text(ref_code, encoding="utf-8")

        # verifier.py
        verifier_src = generate_evalplus_verifier(
            task_id, entry_point, base_inputs, plus_inputs,
            expected_outputs, atol
        )
        (env_dir / "verifier.py").write_text(verifier_src, encoding="utf-8")

        # metadata.json
        total_tests = len(all_inputs)
        metadata = {
            "id": safe_id,
            "name": entry_point,
            "difficulty": "medium",
            "domain": "coding",
            "benchmark": name,
            "num_base_tests": len(base_inputs),
            "num_plus_tests": len(plus_inputs),
            "total_tests": total_tests,
            "tags": ["evalplus", name, "augmented-tests"],
        }
        (env_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        registry.append({**metadata, "path": f"environments/{name}/{safe_id}"})

        if (idx + 1) % 50 == 0:
            print(f"  [{name}] {idx + 1}/{len(data)} processed...")

    # registry.json
    (output_dir / "registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )

    if errors:
        print(f"  Reference errors in {len(errors)} tasks:")
        for e in errors[:10]:
            print(e)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return registry


def main():
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError:
        print('evalplus is required to run the importer: pip install evalplus')
        sys.exit(1)

    cache_root = Path.home() / ".deepgym" / "environments"
    stats = {}

    # HumanEval+
    print("Loading HumanEval+...")
    he_plus = get_human_eval_plus()
    print(f"Loaded {len(he_plus)} HumanEval+ problems. Computing expected outputs...")
    output_dir = cache_root / "humaneval_plus"
    registry_he = import_benchmark("humaneval_plus", he_plus, output_dir)
    print(f"Generated {len(registry_he)} HumanEval+ environments in {output_dir}")
    stats["humaneval_plus"] = registry_he

    # MBPP+
    print("\nLoading MBPP+...")
    mbpp_plus = get_mbpp_plus()
    print(f"Loaded {len(mbpp_plus)} MBPP+ problems. Computing expected outputs...")
    output_dir = cache_root / "mbpp_plus"
    registry_mbpp = import_benchmark("mbpp_plus", mbpp_plus, output_dir)
    print(f"Generated {len(registry_mbpp)} MBPP+ environments in {output_dir}")
    stats["mbpp_plus"] = registry_mbpp

    # Summary
    print("\n=== Import Summary ===")
    for bench, reg in stats.items():
        total_tests = sum(e["total_tests"] for e in reg)
        avg_tests = total_tests / len(reg) if reg else 0
        print(f"{bench}: {len(reg)} environments, "
              f"{total_tests} total tests, "
              f"{avg_tests:.0f} avg tests/env")


if __name__ == "__main__":
    main()
