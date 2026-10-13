# Write a verifier

This page covers writing a verifier: the shapes it can take, the `deepgym_solution`
helpers it must use to run the solution, how the solution is isolated from it, and the
JSON protocol it prints.

```python
from deepgym import DeepGym, Environment

dg = DeepGym(mode='local')
env = Environment(
    task='Write a function `add(a, b)` that returns the sum of two numbers.',
    verifier_code=(
        'from deepgym_solution import load_solution\n'
        'sol = load_solution(solution_path)\n'
        'return 1.0 if hasattr(sol, "add") and sol.add(2, 3) == 5 else 0.0\n'
    ),
)

result = dg.run(env, model_output='def add(a, b):\n    return a + b\n')
print(result.score)   # 1.0
```

For a full environment folder with seeded cases and a reference, start from
`deepgym new` instead (see [new, check, qa](new-check-qa.md)).

## Running the solution

Never import or run the solution yourself. Use the helpers in `deepgym_solution`:

- `load_solution(path)` starts it in a separate, isolated process and returns a proxy:
  `sol.add(2, 3)` runs there and returns plain data.
- `run_solution(path, cwd=..., env=...)` runs it as a script, writing only inside `cwd`.
  Pass `allow_subprocess=True` if the script needs to start other programs (macOS blocks
  forking otherwise), and `loopback_port=port` to let it reach a server the verifier runs
  on `127.0.0.1:port` (nothing else on the network).
- `read_output(path, cwd)` reads its files back and refuses symlinks.
- `solution_source(path, name=None)` reads the source for static checks without running
  it.
- `apply_patch(diff, files)` applies a unified diff to a dict of file texts in memory, and
  `load_files(files, entry)` imports the result like `load_solution`. Together they score
  repo-patch answers.

When the machine can't start the solution at all (out of processes, say), these raise
`LaunchError`. Let it propagate: the run ends as a verifier error with no verdict, instead
of scoring the solution 0.

Each call is bounded by `case_timeout` (default 10 s) and by the run's `timeout` (default
30 s), so a slow case fails on its own and the run still returns partial credit.

## Isolation

- **Sandboxes (Daytona, Modal, E2B, Docker, Apple container):** the solution runs as a
  separate Linux user that can't read the judge's files, touch the judge's process, or
  write to its output, with no network.
- **Locally:** it runs under `sandbox-exec` on macOS or `bwrap` on Linux, with the judge
  directory, the deepgym package, your home directory and `/run` (daemon sockets) hidden.
  If neither is available, you still get process separation and a warning.

What this blocks:

- Only plain data crosses the boundary, so an object that claims to equal everything
  can't fake a pass.
- A fake score line printed by the solution never reaches the judge's output.
- Patching the verifier at runtime has no effect.

## Verifier shapes

Detected by parsing, or set explicitly with `verifier_mode`:

- `script`: has a `__main__` block that prints protocol JSON.
- `verify_function`: defines `verify(solution_path, test_cases_path=None)`.
- `body` (the example above): the body of that function.

Return a float, bool or dict; the wrapper turns it into the protocol. Any other return
value, or a `passed` or `truncated` that isn't a real bool (wrap numpy bools in `bool()`),
is a verifier error (exit 2). A NaN, infinite or out-of-range score is a verifier error
too, never reward. `from __future__` imports are fine in any mode.

Shared helper code goes in `support_files` (name to content). These files sit next to the
verifier, are importable by it, and are hidden from the solution. In a registry env, list
paths in `metadata.json`, for example `"support_files": ["../_shared/scoring.py"]`.

## Output protocol

Verifiers print JSON to stdout:

```json
{
  "schema_version": "1.0",
  "score": 0.85,
  "passed": true,
  "details": "12/14 tests passed",
  "cases": [
    {"id": "test_0", "passed": true, "score": 1.0, "input_summary": "coins=[1,2,5] amount=11"},
    {"id": "test_1", "passed": false, "score": 0.0, "error": "expected 3, got -1"}
  ],
  "seed": 42
}
```

- `score` is in [0, 1]. Optional fields: `reward_components` (shaped rewards such as
  `{"correctness": 0.8}`), `metrics`, `truncated` and `error_type`.
- `cases` tells you which tests passed and failed, so your training loop gets a denser
  signal than 0 or 1. A failed case without an explicit `score` counts 0.
- Exit codes back the result: 0 passed, 1 failed (it keeps its partial score but can't
  claim a pass), 2 verifier error. Any other exit (killed, timed out, crashed) means the
  verifier never finished, whatever it printed first.
- Only stdout counts, so a JSON line on stderr can't replace the verdict; stderr is kept
  in `RunResult.stderr`.

A verifier that returns a float or bool is wrapped into this format automatically.
