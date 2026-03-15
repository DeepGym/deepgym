#!/usr/bin/env python3
"""Validate BigCodeBench environments by running verifiers against reference solutions."""

import json
import random
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_DIR = PROJECT_ROOT / "environments" / "bigcodebench"

def validate_env(task_dir: Path, timeout: int = 30) -> dict:
    """Run verifier.py against reference_solution.py, return result dict."""
    verifier = task_dir / "verifier.py"
    solution = task_dir / "reference_solution.py"
    task_id = task_dir.name

    if not verifier.exists() or not solution.exists():
        return {"id": task_id, "status": "missing_files"}

    try:
        proc = subprocess.run(
            [PYTHON, str(verifier), str(solution)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(task_dir),
        )
        if proc.returncode != 0 and not proc.stdout.strip():
            return {
                "id": task_id,
                "status": "error",
                "details": proc.stderr[:500] if proc.stderr else "non-zero exit, no output",
            }
        result = json.loads(proc.stdout.strip())
        return {"id": task_id, "status": "ok", **result}
    except subprocess.TimeoutExpired:
        return {"id": task_id, "status": "timeout"}
    except json.JSONDecodeError:
        return {"id": task_id, "status": "bad_json", "details": proc.stdout[:300]}
    except Exception as e:
        return {"id": task_id, "status": "exception", "details": str(e)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    all_dirs = sorted(d for d in ENV_DIR.iterdir() if d.is_dir())
    print(f"Total environments: {len(all_dirs)}")

    random.seed(42)
    sample = random.sample(all_dirs, min(n, len(all_dirs)))

    passed = 0
    failed = 0
    errors = 0
    results = []

    for task_dir in sample:
        r = validate_env(task_dir)
        results.append(r)
        status_str = r.get("status", "?")
        score = r.get("score", "?")
        details = r.get("details", "")
        is_pass = r.get("passed", False)

        if status_str == "ok" and is_pass:
            passed += 1
            print(f"  PASS {r['id']}: score={score} {details}")
        elif status_str == "ok":
            failed += 1
            print(f"  FAIL {r['id']}: score={score} {details}")
        else:
            errors += 1
            print(f"  ERR  {r['id']}: {status_str} -- {details[:200] if details else ''}")

    print(f"\n{'='*60}")
    print(f"Validated {len(sample)} environments")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Errors:  {errors}")


if __name__ == "__main__":
    main()
