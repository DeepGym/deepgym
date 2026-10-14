# Write an environment: new, check, qa

This page covers going from an idea to a QA-approved environment folder: scaffolding it
with `deepgym new`, the files it contains, the four kinds, validating it with
`deepgym check`, and gating it with `deepgym qa`.

```bash
deepgym new word_stats --kind function   # or script, repo-patch, multi-turn
deepgym check word_stats                 # passes as generated
# edit word_stats/: task.md, the reference, expected() in verifier.py, variants.py
deepgym check word_stats
deepgym qa word_stats --mutants          # adds reward-hack attacks and mutation testing
deepgym qa word_stats --mutants --mode docker --approve --by "Jane Doe"
```

`deepgym new` writes a working example (the longest strictly increasing run in a list)
that passes every check, so you start green and change one thing at a time. `--dir`
picks the parent folder.

## The files

| File | What it is |
|---|---|
| `task.md` | The prompt the model sees. |
| `verifier.py` | Scores a solution. It runs the solution only through `deepgym_solution`, computes each expected answer with its own `expected()`, and reports every case with a reason. |
| `reference_solution.py` | Must score 1.0. A `.diff` for repo-patch, `reference_actions.json` for multi-turn. |
| `variants.py` | `generate(seed, count)`: edge cases plus seeded random inputs. The fixed seed keeps runs reproducible. |
| `metadata.json` | Name, type, difficulty, tags and the `support_files` the verifier imports. Unknown keys are rejected. |

How to write the verifier itself, and what it may and may not do with the solution, is
in [verifiers](verifiers.md).

## Kinds

- `function`: the verifier calls a function through `load_solution`.
- `script`: the verifier runs a stdin/stdout script through `run_solution`.
- `repo-patch`: the model answers with a unified diff. The verifier applies it to the
  files in `repo/` with `apply_patch` (in memory, only to files you list; a diff that
  creates, deletes, renames or copies a file is refused) and imports the result with
  `load_files`, the way Python would import it from a checkout. It isn't tagged `patch`:
  that tag adds attacks whose diffs add test files, which this verifier can never apply;
  tag it once you let patches add files.
- `multi-turn`: an agent works in a directory that starts with the folder's other files
  (here `runs.py`), then the final verifier tests the result. Only fixes earn reward:
  cases the shipped `runs.py` already gets right must keep passing (breaking one zeroes
  the score), and the score is the share of the cases it got wrong that now pass, so an
  agent that changes nothing earns nothing. `reference_actions.json` is the scripted
  episode that must score 1.0. A file that links outside the folder is refused rather
  than copied in. Local sessions aren't isolated; use a sandbox mode for untrusted
  agents. The final verifier loads the agent's code with `load_solution`, which a sandbox
  session provides only on an image with `sudo` (a separate user runs that code).

## deepgym check

`deepgym check PATH` validates the files, the metadata schema and that the verifier
parses. Then it runs the reference twice and an empty and a stub solution once, and
checks the output: protocol JSON, a `cases` list on the reference run and a reason on
every failed case. It exits 0, 1 if a check failed, or 2 if one is inconclusive (say, the
verifier crashed on the empty solution). `--mode` picks where it runs, as for `qa`.

```
ok    files             task, verifier, reference and metadata
ok    metadata          coding env 'word_stats', difficulty easy
ok    verifier_source   verify_function verifier
ok    reference_passes  score 1.00 (46/46 cases passed)
ok    deterministic     2 runs all scored 1.00
ok    empty_fails       score 0.00 (solution does not define a function named longest_run)
ok    stub_fails        score 0.00 (0/46 cases passed)
ok    protocol          46 cases per run; every failed case says why
skip  approval          not approved yet: deepgym qa word_stats --approve --by NAME once it passes

result: PASS
```

It also prints warnings from an advisory lint over the judge code: a verifier that never
calls `load_solution`, `run_solution`, `load_files` or `solution_source`, that hands the
solution path to anything else (such as `open`), or that uses something able to run code
itself (`exec`, `eval`, `importlib`, `runpy`, `pickle`, `timeit`, subprocesses,
`sys.path`). Warnings never change the result, and a clean lint proves nothing: a pattern
check can always be dodged. The isolation is the judge reaching the solution only through
those helpers.

Once the folder is approved, `deepgym check` also fails when its files no longer match the
approval (see [approvals](approvals.md)).

## deepgym qa

`deepgym qa` takes environment folders as well as registered names. It runs the reference
several times, checks that empty and stub answers earn nothing and that no reward-hack
answer reaches 0.5, and with `--mutants` checks that no small mutation of the reference
passes or scores full marks. Multi-turn folders
replay `reference_actions.json` and run scripted agents that must earn nothing. Every check
is described in [the QA gate](qa-gate.md).

**Note:** locally, `--mutants` needs `bwrap` (Linux). On macOS, run it with
`--mode docker`, `modal` or `e2b`; otherwise the mutation check is inconclusive and no
mutant runs.

`--approve --by NAME` writes `approval.json` once everything passes; see
[approvals](approvals.md).

## Use it

A folder loads like any environment:

```python
from pathlib import Path

from deepgym import DeepGym
from deepgym.registry import load_environment_dir

env = load_environment_dir(Path('word_stats'))
result = DeepGym(mode='docker').run(env, Path('word_stats/reference_solution.py').read_text())
print(result.score)   # 1.0
```

To ship it with the package, add the folder under `src/deepgym/envs/` and an entry in
`src/deepgym/envs/registry.json`; see the
[contributing guide](https://github.com/DeepGym/deepgym/blob/main/.github/CONTRIBUTING.md).
