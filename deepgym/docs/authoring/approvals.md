# Approvals

This page covers `approval.json`: how `deepgym qa --approve` records that an environment
folder passed, what the file holds, and how `deepgym check` tells when the folder has
changed since.

```bash
deepgym qa ./word_stats --mutants --approve --by "Jane Doe"
deepgym check ./word_stats   # fails once the files no longer match the approval
```

## When it is written

`--approve --by NAME` writes `approval.json` once QA passes with at least one rerun and
every check you asked for (`--mutants`, `--model`) passed. Checks that didn't run are
listed with it, never counted as passed.

QA runs on a private copy of the folder, so the hashes describe exactly the files it
tested. The folder must be self-contained: no symlinks, and no support files outside it.

## What it holds

- The full QA report.
- A sha256 and the permission bits of every file and directory in the folder, hidden ones
  too, so even a `.DS_Store` Finder adds makes it stale.
- The deepgym version and the time.
- The lint warnings from `deepgym check`, under `lint` and marked `"advisory": true`:
  information for the reader, never evidence that the verifier isolates the solution.

It is written once, read-only. Delete it to approve again.

## Checking an approved folder

`deepgym check` fails when the files no longer match the approval, Harbor task folders
included. So an environment pack carries proof of exactly what was tested, and anyone can
confirm the files they received are those files.
