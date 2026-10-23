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
- `run_solution(path, cwd=..., env=...)` runs it as a script and returns a
  `subprocess.CompletedProcess`; check its `returncode`. Locally it can write only inside
  `cwd`; in a sandbox it runs as the solver user, which can still write world-writable
  paths.
  Pass `allow_subprocess=True` if the script needs to start other programs. Locally on
  macOS that is refused (the run ends with no verdict), since nothing there contains a
  child that detaches: run such environments on Linux or a sandbox provider. Pass
  `loopback_port=port` to let it reach a server the verifier runs on `127.0.0.1:port`
  (nothing else on the network). Output past 64 MiB, printed or read back with
  `read_output_bytes`, is the solution's failure, never cut to a prefix.
- `read_output(path, cwd)` reads its files back and refuses symlinks.
- `solution_source(path, name=None)` reads the source for static checks without running
  it.
- `apply_patch(diff, files)` applies a unified diff to a dict of file texts in memory, and
  `load_files(files, entry)` imports the result like `load_solution`. Together they score
  repo-patch answers.

Whatever a `load_solution` call does (raise, exit, hang, die, return garbage) reaches the
verifier as a `deepgym_solution.SolutionError`, still of the builtin type the solution raised, so
`except ValueError` keeps working. If the verifier doesn't handle it, the run is a judged
0 (`solution_error`, or `solution_timeout` for a hang past the time limit), not a verifier
error, so a solution can't dodge its score by breaking the verifier. Load test data before
starting the solution: a verifier that crashes after that scores 0 too.

When the verifier itself can't judge (a fixture it needs is missing, a service it checks
against is down), it must raise, never return 0. Raised before it starts the solution, the
run ends with no verdict, which trainers mask; a returned 0 is a real verdict that training
learns from. So check what you need before starting the solution: after that, a crash is
the solution's 0, as above.

When the machine can't start the solution before any solution code has run (out of
processes, say), these raise `LaunchError`. It isn't an `Exception`, so `except Exception`
doesn't catch it. Let it propagate: the run ends with `sandbox_error` and no verdict. Once
solution code has run, the same failure is the solution's (it may have used up the disk
or processes) and reaches the verifier as a `SolutionError`.

Each call is bounded by `case_timeout` (default 10 s) and by the run's `timeout` (default
30 s); loading the solution has its own 30 s limit. A slow case raises a `SolutionError`
that is a `TimeoutError`: catch it per case to keep partial credit, since an uncaught one
makes the whole run a judged 0.

## Isolation

- **Sandboxes (Daytona, Modal, E2B, Docker, Apple container):** the solution runs as a
  separate Linux user that can't read the judge's files, touch the judge's process, or
  write to its output, with no network.
- **Locally:** it runs under `sandbox-exec` on macOS or `bwrap` on Linux and can read only
  the operating system, the software trees a Python links against (Homebrew's, say), the
  Python installation and its own run's files. Your home directory, temp directories
  (another run's verifier, an answer key under `/tmp`) and daemon sockets aren't
  readable, and no deepgym package is. Where neither sandbox works, local runs refuse
  unless you set `DEEPGYM_ALLOW_UNSANDBOXED=1`; the solution can then read your files.

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
value, a `passed` or `truncated` that isn't a real bool (wrap numpy bools in `bool()`), or
a NaN, infinite or out-of-range score is never reward: it is a verifier error (exit 2), or
a judged 0 (`solution_error`) once the solution's output is in play. `from __future__`
imports are fine in any mode.

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
  claim a pass), 2 verifier error. Any other exit (killed by a signal, say) has no
  verdict, whatever it printed first.
- An uncaught exception overrides anything printed before it, in every verifier shape:
  before the solution's output is in play it is a verifier error with no verdict (such
  as `runtime_error`), after it a judged 0 (`solution_error`).
- Only stdout counts, so a JSON line on stderr can't replace the verdict; stderr is kept
  in `RunResult.stderr`.

A verifier that returns a float or bool is wrapped into this format automatically.
