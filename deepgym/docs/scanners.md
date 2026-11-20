# Scan traces

This page covers scanning recorded [traces](traces.md) for what the verifier didn't
catch, from Python or with `deepgym traces scan`: rule and model scanners, their
findings, storing them as [labels](labels.md), and dropping flagged traces from a
[dataset](datasets.md).

`deepgym.scanners` reads recorded traces for behaviour the verifier didn't catch, such as
an agent changing files the task marks as off-limits, or a pattern you have learned to
watch for. Each finding cites where it happened, as a label cites text.

```python
from deepgym.console.db import DEFAULT_DB_PATH
from deepgym.console.label_store import LabelStore, read_labels
from deepgym.dataset import DatasetSpec
from deepgym.rubric import openai_judge
from deepgym.scanners import (
    HumanMark, LLMScanner, PatternScanner, ProtectedPathScanner,
    finding_labels, group_findings, scan_traces, scanner_agreement, scanner_forbids,
)

scanners = [
    ProtectedPathScanner(name='protected', globs=('tests', 'conftest.py')),
    PatternScanner(name='skips', patterns=[
        {'regex': r'pytest\.skip|@unittest\.skip', 'message': 'Skips a test', 'severity': 'high'},
    ]),
    LLMScanner('tamper', 'Find any step where the agent weakens or bypasses the tests.',
               openai_judge('judge-model', base_url=..., reasoning_effort='high'),
               judge_id='judge-model, reasoning_effort high'),
]
report = scan_traces(traces, scanners, max_parallel=8)
report.findings   # Finding(scanner, trace_id, step, field, span, quote, message, severity)
report.errors     # ScanError(scanner, trace_id, error): that trace was not cleared
report.scanners   # name -> ScannerInfo(source, version, instruction_sha256, judge_id)
for group in group_findings(report.findings):
    print(group.scanner, group.message, len(group.findings), group.traces)
LabelStore(DEFAULT_DB_PATH).add_many(finding_labels(report))
scanner_agreement(report, read_labels(DEFAULT_DB_PATH),
                  {'tamper': HumanMark(name='weakens_tests')})
spec = DatasetSpec(forbid_labels=(
    *scanner_forbids('protected', source='verifier'),
    *scanner_forbids('tamper', source='judge', values=('high',)),
))
```

```bash
deepgym traces scan --config scanners.json --out report.json --write-labels
```

- **A finding** names its scanner and trace, the step (`TraceStep.index`, or null for
  the whole trace), the text (`action` for `Action.content`, `observation`, or
  `message:<i>` for `TraceStep.messages[i]`), optionally a `(start, end)` span of that
  text in Unicode code points with the quoted text, a one-line message, and a severity
  (`low`, `medium` or `high`). It follows the label target rules
  (`deepgym.labels.LabelTarget`).
- **A scanner** is anything with a `name`, a `source` (`verifier` for rules, `judge` for
  a model), a one-line `version`, an `instruction_sha256` (None unless it follows a
  written instruction), a `judge_id` (None unless it asks a judge the caller named) and
  `scan(trace)` returning findings. `scan_traces` takes one or more scanners and runs
  each on each trace, `max_parallel` at once (threads, worth raising for a model
  scanner), and records each scanner's source, version, instruction hash and judge in
  `report.scanners`. A scanner that raises, returns anything but an iterable of findings
  (a string, bytes or a mapping included, even an empty one), or returns a finding of
  another trace or one citing a step, text or quote the trace doesn't hold, is a
  `ScanError` for that trace, never "no findings". Scanner names are unique, and can't be
  a verdict name (`passed` or a rubric check id).
- **`PatternScanner`** flags each match of each regex (Python `re`, flags inline such as
  `(?i)`) in the `fields` chosen: `action` (the default), `observation`, or the messages
  of a role (`system`, `user`, `assistant`, `tool`). Each pattern has its message and
  severity (`medium` by default). An empty match has nothing to quote and is skipped. It
  doesn't read `Action.events`, where a CLI agent's own tool calls are: for a `cli`
  action it sees only the launch command and the CLI's output (`ProtectedPathScanner`
  reads the edits and commands in the events). Its version is `pattern/1+config-sha256:`
  and the SHA-256 of its JSON config with each regex's flags (which the JSON leaves out);
  `ProtectedPathScanner`'s is `protected-paths/6+config-sha256:...`. The part before `+`
  names how the scanner reads a trace and changes whenever that does.
- **`ProtectedPathScanner`** flags each applied action whose text names a path matching
  one of its `globs` as one it writes, deletes or moves (severity `high` by default).
  Nothing is run, and **only these forms are read**: a `write_file` path; in `bash`,
  redirection targets, the operands of `rm`, `rmdir`, `unlink`, `shred`, `touch`,
  `truncate`, `tee`, `mkdir`, `chmod`, `chown` and `chgrp`, the destination of `cp`,
  `install`, `ln`, `mv` and `rsync` and what `mv` moves away (a destination that is a
  folder by its form, `.`, a `..` chain or ending in `/`, receives each source by its
  name, so `cp /tmp/data.csv .` writes `./data.csv`; so does `DIR` with `-t DIR`, also
  bundled as `-rt DIR` or `-tDIR`, or `--target-directory[=]DIR`, though not for `rsync`,
  whose `-t` keeps times; with `-T` (`--no-target-directory`), or a source the shell
  expands (`*?[{$`, a backtick, `~`), the folder itself is the destination; with
  `install -d`, every operand; long options match by unique prefix, as GNU's do, such as
  `--target-dir`; unrecognised options, a prefix that matches several, and a destination
  that can't be told, such as `-t` with no DIR, two DIRs or a single operand, make the
  scanner report every operand and option value;
  option values are always reported too, so a mistake in the option tables can't hide a
  path), the files of `sed`, `perl` and `ruby` with `-i` (not the script: the first
  operand unless `-e`, `-f` or `--expression` gives it), the paths named after
  `git rm`, `git mv`, `git checkout`, `git restore` and `git clean` (which deletes
  untracked files), and `find ... -delete` (`find . -delete` is flagged: `.` holds every
  relative path, and nothing in the text says which it deletes), also after `if`, `then`,
  `do`, `!`, `{` and
  the like, `VAR=value`, and `sudo`, `env`, `nice`, `timeout`, `nohup`, `time`,
  `command` or `exec` with their options, and inside `bash -c`; in `code` (Python),
  literal paths given to `open` with a write mode (`file=` and `mode=` too), to `os` and
  `shutil` calls that change files (`path=`, `src=` and `dst=` too) and to `Path(...)`
  methods that do, and literal shell commands given to `os.system` or `subprocess` (a
  `shutil.copy`, `copy2` or `move` destination that is a folder by its form receives a
  literal source by its name; with any other source, the folder itself is the
  destination), with the code's imports resolved (`from pathlib import Path
  as P`, `import shutil as sh`, `import os.path as osp`; regardless of scope or order, a
  name bound in several places takes every meaning, and `os`, `shutil`, `subprocess`,
  `io`, `builtins` and builtins such as `open` keep their own, so a conditional or
  function-local `import foo as os` can't hide `os.remove`) and a literal counted through
  calls that keep the path (`str`, `os.fspath`, `os.path.abspath`, `realpath`,
  `normpath`, `Path(...).resolve()`) or join literals (`Path('a') / 'b'`,
  `Path('a', 'b')`, `os.path.join('a', 'b')`, `joinpath`); methods of anything else,
  such as `str(...).replace`, change no file; in `cli`, the files Claude Code's Write,
  Edit, MultiEdit and NotebookEdit calls and Codex's file changes touched, and the shell
  commands each ran, a Codex command seen only starting (a run cut off) included. An edit
  the tool reported failing is still reported, as an attempt: "tried to change <path>
  matching <glob>; the tool reported failure", with the same severity. A heredoc's body
  is data.
- **So no finding means none of those forms named a protected path, not that no protected
  file changed.** Not read: commands that change the whole work tree without naming a
  path (`git checkout` of a branch, `git reset --hard`, `git stash`, `git clean` without
  paths), patches (`git apply`, `patch`), paths held in variables or built at run time,
  shell globs (read as written, never expanded: `rm t*/a.py` isn't `tests/a.py`),
  scripts run from a file or heredoc, a `cd` before a relative path, `tool` actions (calls
  to the environment's HTTP tools), and text typed into a GUI. When you need to
  know every file that changed, diff the final workspace against the starting one. A
  rejected action changed nothing. A `code` action that doesn't parse as Python here is
  an error unless its observation shows Python refused it (then it changed nothing).
- **Globs** are normalised first: `./tests/**` is `tests/**`, and `tests/` is `tests`.
  `**` matches any run of folders, `*` and `?` stay within one name, and `[...]` and
  `{a,b}` are literal text. A glob that matches a folder covers everything under it, so
  `tests`, `tests/` and `./tests/**` all protect `tests/test_x.py`. A glob starting with
  `/` matches absolute paths only, never a relative one, so never a `write_file` path
  (always relative). Any other matches a path's names wherever they are: `tests/**`
  matches `/repo/tests/a.py`, `conftest.py` any `conftest.py`, and `tests` a folder named
  `tests` anywhere (`src/pkg/tests/x.py` too), and also `.` and `..`, which hold every
  relative path (`rm -rf .`, `git checkout .`), though not `/`. A folder holding a glob's
  fixed leading part, after any leading `**/`, counts too: `rm -r tests` changes
  `tests/conftest.py` and `**/tests/**`. Each glob is one regex a path matches whole, one match per
  path and glob; a glob with several `**` can still slow down on a very deep path that
  repeats its folder names.
  A finding quotes the path where the action's text has it; for `write_file` and `cli`,
  its message names the path instead.
- **`LLMScanner(name, instruction, judge, scope='trace', judge_id=None)`** takes any
  `deepgym.rubric.JudgeClient` (`openai_judge`, `anthropic_judge` or your own); the client
  doesn't say which model it asks, so name it in `judge_id` for the report. The judge
  gets a fixed system prompt (`SCAN_PROMPT`, version `SCAN_PROMPT_VERSION`,
  `trace-scan/2`) and one JSON user message: the instruction, the task, the initial
  observation's text, `Trace.error`, the scope, and each step's number, action (every
  field set), whether it was applied, its error, exit code, observation text and
  messages (role and content). Not sent: screenshots, other observation metadata, token
  ids and logprobs, usage and cost, timing, step rewards and step verifier output, the
  grades and provenance. Nothing sent is cut. Its reply must be exactly
  `{"findings": [...]}`, bare or in one `json` fence, each finding with exactly `step`,
  `field`, `quote`, `message` and `severity`. Quotes come only from a step's action
  content, observation or messages, and are located by their first occurrence; evidence
  elsewhere (another action key, an error, the initial observation) has a null quote. A
  judge failure, any other reply, a step that wasn't sent, or a quote its text doesn't
  hold is a `ScanError`, never clean. `scope='step'` sends one step per call, and a
  finding must name that step (null is refused); a trace without steps gets one call
  with none. Any other scope is refused. The scanner's version, on its labels as
  `rubric_version`, names the prompt, the scope and the judge when given:
  `trace-scan/2+scope:step+judge:<judge_id>`. With `scope='trace'`, a trace too long for the
  judge's context is an error that says so.
- **Groups.** `group_findings(findings, key=None)` clusters findings by scanner and
  message, casefolded with runs of whitespace as one space, and by `key(finding)` when
  given, largest first; each group keeps every finding as its citations and counts its
  traces.
- **Labels.** `finding_labels(report)` returns a `LabelCreate` per finding, a mark on its
  exact target (trace, step, field, span and quote) valued by the severity with the
  message as the comment, and one per error, a mark on the whole trace valued
  `scan_error` (`SCAN_ERROR`) with the error as the comment, so a trace a scanner
  couldn't clear never reads as clean; a trace no scanner ever scanned has no mark.
  Each is named and authored by its scanner, with the scanner's source, its version as
  `rubric_version`, its instruction's SHA-256 (model scanners) as `rubric_sha256`, the
  run's `ScanReport.run_id` as `group_id`, and the id `<run_id>-<n>`: storing the same
  report twice is a retry that stores nothing new. Marks aren't verdicts, and review
  queues can take them.
- **Dropping flagged traces.** `scanner_forbids(name, source=None, values=None)` returns
  the `DatasetSpec.forbid_labels` entries for one scanner: its marks (only those of the
  severities in `values`, when given) and always its `scan_error` marks, so a value
  filter never keeps a trace the scanner failed on, as
  `LabelMatch(name='tamper', value='high')` alone would. Give human marks other names
  than scanners (here `weakens_tests`, not `tamper`), so a `LabelMatch` by name never
  mixes the two.
- **Scanning again** is a new run with new labels in a new group; filter label queries
  by `group_id` to read one run. `forbid_labels` can't filter by group, so every current
  mark counts, old ones included: a trace flagged, or failed on, by an earlier run stays
  dropped until that mark is retracted, by a record of the same source and author:
  `LabelCreate(type='retract', target={'kind': 'trace', 'id': 't1'}, source='verifier',
  author='protected', supersedes='<the mark id>', comment='Rescanned cleanly in run
  <run_id>.')`.
- **Agreement.** `scanner_agreement(report, labels, marks)` compares each scanner named
  in `marks` with a human bool label (`HumanMark(name, flagged=True)`: the value that
  means the scanner should flag; `HumanMark(name='passed', flagged=False)` uses failed
  verdicts). It counts current human labels on `trace` targets the report scanned,
  outside traces a current invalidation excludes, as a `Confusion` (TPR, TNR, precision,
  F1, kappa; a flag is the positive): `traces` from whole-trace labels against any
  finding on the trace, `steps` from labels on a step against that step's findings. A
  target no human labeled never counts as clean; targets whose human labels disagree
  (`contested`) and those on traces the scanner failed on (`scan_errors`) are left out.
- **`deepgym traces scan`** reads traces as `deepgym traces export` does: from the
  console database and the traces directory, or the `--db` and `--traces` named.
  `--config` is a JSON list of rule scanners, each `{"kind": "pattern", "name": ...,
  "patterns": [...], "fields": [...]}` or `{"kind": "protected_paths", "name": ...,
  "globs": [...]}` (one at least; an unknown key is refused). It prints every group with
  every citation
  and each error on stderr; `--out` writes the whole report as JSON, and
  `--write-labels` stores the labels of the findings and errors in the console database:
  `--db`, which must exist since traces are read from it too, else
  `~/.deepgym/console.db`, created if missing. It exits 3 when a scanner failed on a
  trace (2 is a usage error). Model scanners run from Python only.
