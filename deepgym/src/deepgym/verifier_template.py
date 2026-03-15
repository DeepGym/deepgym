"""Standard verifier wrapper template.

The wrapper guarantees that **any** user-supplied verification function
produces structured JSON on stdout, regardless of whether the user returns a
``float``, a ``bool``, or a full ``dict``.

Output protocol
---------------
The wrapper always prints exactly one JSON object to stdout::

    {"score": <float 0-1>, "passed": <bool>, "details": <str|null>}

Exit codes:
  * ``0`` — verifier passed (``passed is True``)
  * ``1`` — verifier ran but the solution failed (``passed is False``)
  * ``2`` — verifier itself crashed
"""

from __future__ import annotations

import textwrap

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

VERIFIER_WRAPPER: str = textwrap.dedent('''\
    #!/usr/bin/env python3
    """DeepGym verifier wrapper. Ensures structured JSON output."""
    import os
    import sys
    import json
    import traceback

    def _run_verifier(solution_path, test_cases_path=None):
        """User verification logic — injected at runtime."""
    {verifier_code}

    if __name__ == "__main__":
        solution_path = sys.argv[1] if len(sys.argv) > 1 else "solution.py"
        test_cases_path = sys.argv[2] if len(sys.argv) > 2 else None

        exit_code = 0
        try:
            result = _run_verifier(solution_path, test_cases_path)

            # Normalize result into the standard protocol dict.
            # Check bool before int/float because bool is a subclass of int.
            if isinstance(result, bool):
                output = {{
                    "schema_version": "1.0",
                    "score": 1.0 if result else 0.0,
                    "passed": result,
                    "details": None,
                }}
            elif isinstance(result, (int, float)):
                output = {{
                    "schema_version": "1.0",
                    "score": float(result),
                    "passed": float(result) >= 0.5,
                    "details": None,
                }}
            elif isinstance(result, dict):
                output = {{
                    "schema_version": "1.0",
                    "score": float(result.get("score", 0.0)),
                    "passed": bool(result.get("passed", False)),
                    "details": result.get("details", None),
                }}
                # Pass through optional rich fields if the user returned them.
                if "reward_components" in result:
                    output["reward_components"] = result["reward_components"]
                if "metrics" in result:
                    output["metrics"] = result["metrics"]
                if "seed" in result:
                    output["seed"] = result["seed"]
                if "cases" in result:
                    output["cases"] = result["cases"]
            else:
                output = {{
                    "schema_version": "1.0",
                    "score": 0.0,
                    "passed": False,
                    "details": f"Unexpected return type: {{type(result).__name__}}",
                }}

            # Clamp score to [0, 1].
            output["score"] = max(0.0, min(1.0, output["score"]))
            output["truncated"] = False
            print(json.dumps(output))
            exit_code = 0 if output["passed"] else 1

        except Exception as e:
            error_type = type(e).__name__
            # Map common exceptions to standard error categories.
            if isinstance(e, ImportError):
                error_category = "import_error"
            elif isinstance(e, TimeoutError):
                error_category = "timeout"
            elif isinstance(e, OSError):
                error_category = "runtime_error"
            else:
                error_category = "runtime_error"

            error_output = {{
                "schema_version": "1.0",
                "score": 0.0,
                "passed": False,
                "details": (
                    f"Verifier error: {{error_type}}: {{str(e)}}"
                    f"\\n{{traceback.format_exc()}}"
                ),
                "truncated": False,
                "error_type": error_category,
            }}
            print(json.dumps(error_output))
            exit_code = 2

        # Flush stdout before os._exit (which does not flush buffers).
        sys.stdout.flush()
        os._exit(exit_code)
''')


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def wrap_verifier(user_code: str) -> str:
    """Wrap user verification code in the standard DeepGym template.

    The *user_code* should define the body of a function that accepts
    ``(solution_path, test_cases_path=None)`` and returns one of:

    * A ``float`` in [0, 1] representing the score.
    * A ``bool`` — ``True`` maps to score 1.0, ``False`` to 0.0.
    * A ``dict`` with keys ``"score"`` (float), ``"passed"`` (bool), and
      optionally ``"details"`` (str).

    Example user code::

        content = open(solution_path).read()
        if "def sort" in content:
            return 1.0
        return 0.0

    The wrapper indents the user code so it becomes the body of
    ``_run_verifier`` inside the template.

    Parameters
    ----------
    user_code:
        Python source that will be placed inside ``_run_verifier``.

    Returns
    -------
    str
        A complete, executable Python script.
    """
    # Indent every line of user_code by 4 spaces so it sits inside the
    # function body of _run_verifier (which is already indented by 4).
    indented = textwrap.indent(user_code.rstrip(), '    ')
    return VERIFIER_WRAPPER.format(verifier_code=indented)
