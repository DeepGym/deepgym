# The QA gate

This page covers `deepgym qa`, which checks an environment before you train on it: the
checks for single-turn and multi-turn environments, your own controls, mutation testing,
the difficulty probe, statuses and exit codes, and the report it writes.

```bash
deepgym qa coin_change two_sum          # or --all
deepgym qa ./word_stats --mutants       # an environment folder
deepgym qa --all --mode daytona --reruns 5   # computer-use envs too: they need a desktop
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
  deepgym qa coin_change --mode docker --model Qwen/Qwen3-8B --samples 8
```

## Single-turn checks

Each single-turn environment must pass:

- `reference_passes`: its `reference_solution.py` scores 1.0.
- `deterministic`: the reference scores the same on every rerun (`--reruns`, default 3).
  Skipped with `--reruns 0`, since one run proves nothing.
- `empty_fails`, `stub_fails`: an empty file, and a stub of the reference's functions
  that return `None`, both earn nothing: any reward above float noise fails, since a
  verifier that pays for doing nothing teaches a policy to do nothing. `stub_fails` is
  skipped when the reference is a script with no top-level function. A verifier that
  crashes on them instead of scoring them is inconclusive, not a pass.
- `hack_resistant`: none of the heuristic reward-hack attacks behind `deepgym audit`
  earns reward above float noise, since a policy learns to farm whatever a shortcut
  earns. The attacks define the env's own entry point (from the reference, the functions
  the verifier calls, or the task) returning a constant, nothing, NaN or its input. If
  none got through but some crashed or couldn't run, the check is inconclusive, not
  passed. The attacks are Python solutions, so the check is skipped where they can't
  apply: a Harbor task runs bash, and a verifier that rejects every attack as a malformed
  answer (a repo-patch env takes diffs) was never attacked. Self-contained patch tasks
  such as `mimo_code` get `patch_tampering` attacks instead.
- `controls` (when the folder has `controls/`): the answers you ship score as they say
  (see below).
- `mutants_killed` (with `--mutants`): no mutant of the reference passes or scores full
  marks (see below).
- `difficulty` (with `--model`): see below.

A skipped check counts neither way, and the report lists it under `not_run`. Harbor tasks
with a `solution/solve.sh` use it as the reference, and an empty script must fail; their
`hack_resistant` and `mutants_killed` checks are skipped, since both run Python
solutions.

## Multi-turn checks

A multi-turn environment with a `reference_actions.json` (the computer-use built-ins ship
one per variant) replays it for `reference_passes` and `deterministic`, then runs three
scripted agents that act without working on the task, each of which must earn nothing:

- `empty_fails` quits at once.
- `busy_fails` uses every allowed action in turn, for the whole step budget.
- `random_fails` acts at random (seeded), for the whole step budget.

A verifier that pays for activity instead of the outcome fails. Computer-use envs need a
desktop, so they run with `--mode daytona` and are skipped elsewhere. Multi-turn envs
without a reference, and benchmark-backed ones such as `swebench_pro`, are skipped:
neither passed nor failed.

## Controls

The checks above try answers the gate makes up. Controls are answers you know about: a
wrong answer your verifier once paid for, or a right one it once scored 0. Put each in
the environment folder, in the same form as the reference (a solution `.py`, a diff for
repo-patch, a JSON list of actions for multi-turn):

```
word_stats/controls/must_fail/nan_string.py     # the string "NaN" where a number belongs
word_stats/controls/must_fail/off_by_one.py
word_stats/controls/must_pass/other_order.py    # right, in an order the reference doesn't use
```

Each control runs exactly as the reference does, as many times (`1 + --reruns`). The
`controls` check fails if any run of a `must_pass` control scores below 1.0, any run of a
`must_fail` control scores 1.0, or the verifier errors on a run instead of judging it,
since a crash's 0 says nothing about the answer. Every failure names the control.

A control can start with `# expect: TEXT`. Every run's details, or one of its cases'
errors, must then contain TEXT, so a `must_fail` control that fails for some other reason
(a typo in it, say) fails the check instead of passing it. The line is removed before the
control runs, so it works in a JSON control too.

A folder without `controls/` doesn't get the check. One whose `controls/` holds anything
but files directly in `must_pass/` and `must_fail/` fails it, so a misnamed folder can't
go unrun. Controls are never copied into a multi-turn session, and an approval covers
them like every other file. `deepgym new --kind function` ships one of each to copy.
Controls are read from the folder you pass, or for an env you name, from the folder the
registry loads it from. A file that links into `controls/` is kept from the agent too.

## Mutation testing

`--mutants` edits the reference one small step at a time with the `ast` module (an
integer off by one, `<` for `<=`, a computed return replaced by a constant, an if/else
swapped, a statement dropped) and runs each mutant like a model solution. A mutant
survives if it passes or scores full marks, and one survivor fails the check: it points at
behavior the verifier never checks. The report gives the mutation score and the first 8
survivors by line, then how many more. A mutant
that behaves exactly like the reference (a redundant guard dropped, say) can't fail; mark
its line `# pragma: no mutate`.

Where mutants can run:

- Locally, mutation runs need `bwrap` (Linux): each mutant's whole judge run goes inside
  its sandbox, so mutants stay contained even if a verifier runs them in-process, and its
  PID namespace takes down everything the run started. Sandboxes don't nest, so within
  the run a mutant isn't kept from the judge's files; mutants are edits of your
  reference, not attacks.
- macOS's `sandbox-exec` can't reap a process a mutant detaches, so on macOS, or without
  `bwrap`, use `--mode docker`, `modal` or `e2b`, which keep the whole sandbox offline.
- `apple` and `daytona` leave the judge online, so there the check is inconclusive and no
  mutant runs.

## Difficulty

`--model` draws n samples (`--samples`, default 8) from an OpenAI-compatible endpoint
(`OPENAI_BASE_URL`, `OPENAI_API_KEY`) and reports the unbiased pass@k (at least one of k
attempts solves the task) and pass^k (all k do) for k = 1, each power of 2 below n and n
itself, and the mean and spread of the rewards. It is reported, never gated: it passes once
measured and flags `too_easy` (every sample solved), `too_hard` (no sample earned
anything), `no_variance` (every sample earned the same), since group-relative RL learns
nothing from equal rewards, and `outside_band` (solved by under 10% or over 90% of the
samples). Partial rewards are variance: `[0.2, 0.8]` is neither too hard nor flat.

For your own sampled runs, `deepgym.passk` has the same estimators (`pass_at_k(n, c, k)`,
`pass_hat_k(n, c, k)`), the band of one task (`task_difficulty`) and their average over
many tasks with counts per band (`summarize`).

The probe sends no output limit. If the endpoint still cuts a sample off at its own
(`finish_reason` `length`), that sample is no answer, so the check is inconclusive
instead of reading it as a failure. Rate limits (429), timeouts and 5xx are retried with
backoff; if a sample never comes back, or the verifier couldn't judge one, the check is
inconclusive. A refused request (a bad key or model) stops the run.

`--model` runs model-generated code, so with `--mode local` (the default) or `auto` (which
can fall back to local) it refuses to start unless you pass `--allow-local-exec`. Use a
sandboxed mode instead.

## Statuses, exit codes and reports

Each check and each environment gets a `status`: `pass`, `fail`, `inconclusive` or
`skip`. An environment fails if any check fails, is inconclusive if any check is, is
`skip` if every check was skipped, and passes otherwise. Every report lists the checks
that `ran` and those `not_run`.

The command exits 0 when the checked environments pass (skipped ones alongside are fine),
1 if any fails, and 2 if none fails but some are inconclusive or none could be checked.
`--json` prints the reports as JSON.

It writes one report per environment to `~/.deepgym/qa/<env>.json` (set `DEEPGYM_QA_DIR`
to change it); the [console](../console.md) shows them.

```json
{
  "environment": "coin_change",
  "status": "pass",
  "passed": true,
  "ran": ["reference_passes", "deterministic", "empty_fails", "stub_fails", "hack_resistant"],
  "not_run": [],
  "checks": [
    {"name": "reference_passes", "status": "pass", "passed": true, "details": "score 1.00 (25/25 passed)"}
  ],
  "created_at": "2026-10-12T16:45:34Z"
}
```

To record that a folder passed, see [approvals](approvals.md). From Python, use
`deepgym.qa.run_qa(target, dg, QAOptions(...))`.
