# Traces

This page covers recording graded agent episodes as traces with `dg.trace()`, what a
trace holds, exporting traces as training data in JSONL, Parquet (verl and TRL rows)
or ATIF, comparing two runs task by task, scanning traces for what the verifier didn't
catch, and building versioned datasets from traces.

## Record an episode

`dg.trace()` records a multi-turn episode while it runs, as training data. It works like
[`dg.session()`](environments/sessions.md), and each step also takes the model messages
behind the action:

```python
from deepgym import load_environment
from deepgym.traces import AgentMeta

agent = AgentMeta(name='react-shell', model='Qwen/Qwen3-8B', sampling={'temperature': 0.7})
env = load_environment('todo_list', variant=3)
with dg.trace(env, agent, variant=3) as tr:   # variant here only labels the provenance
    obs = tr.start()
    while not tr.done:
        action, messages = my_agent(obs)   # messages: this turn's new TraceMessages
        obs = tr.step(action, messages).observation
    tr.score()
print(tr.trace.has_verdict)
```

- `TraceMessage(role=..., content=..., token_ids=..., logprobs=...)` holds one model
  message; token ids and logprobs are optional.
- When one model reply issues several actions, step each one and pass the reply's
  messages with the first only; the others belong to the same turn (`trace.turns`).
- `tr.step(action, messages, usage=Usage(...), cost_usd=...)` also records the provider
  token usage of the model calls since the previous step (`deepgym.traces.Usage`: input,
  output, cache-read, cache-write and reasoning tokens, reasoning being part of output)
  and, if you price it, what it cost in USD. Like messages, pass a reply's usage with its
  first action only, so totals count it once. Under `dg.trace()` the agents in
  `deepgym.agents` do this for you. Their costs come from the `TokenPrices` you give them,
  or, for a CLI run, the CLI's own reported total (Claude Code's `total_cost_usd`, which
  takes precedence over `TokenPrices`); without either, the cost is null, never a guess.
- A traced episode refuses actions after `score()`, so its verdict always matches the
  state it ends in.
- `variant`, `seed` and `reference` (the reference solution's source) only go into the
  provenance; load the environment with the same variant yourself.

## What a trace holds

- **Provenance:** environment name, variant and seed; SHA-256 hashes of the environment
  definition, task, verifiers and reference; the image the sandbox booted with the digest
  its provider resolved, or why there is none (`digest_unavailable`); the deepgym version
  and the git commit of the deepgym checkout (null for an installed package), with whether
  it had local changes; the provider; and the agent, model and sampling config you pass.
- **Every step:** the action, the model messages behind it with optional token ids and
  logprobs, the provider token usage and USD cost of the model calls behind it, what came
  back, the step reward with the step verifier's output, exit code and error.
- **Usage outside any step:** a model call that led to no action (a closing answer, a
  refusal, a reply cut off, or one after which a budget ran out) is on no step, and nor is
  a CLI agent's run, whose one step is recorded before the CLI reports its usage. Their
  usage and cost are in `trace.unattached_usage` and `trace.unattached_cost_usd`.
  `tr.record_unattached_usage(usage, cost_usd)` records it; the agents do when they finish,
  including when an error ends the run.
- **Incomplete usage:** `trace.usage_complete` is true when no model call was flagged
  incomplete. It is false when some call reported no usage, or only part of it (a failed
  request, a response cut off, an error that ended the run mid-call, an interrupted CLI
  run). Unknown usage is never counted as 0 tokens: the token totals then count only what
  was reported, a lower bound, and the cost total is null. `tr.mark_usage_incomplete()`
  flags it.
- **Usage totals:** `trace.usage` and `trace.cost_usd` sum the steps and the unattached
  usage, and are null when nothing reported usage, so unknown never reads as 0; the cost
  is also null when any part of the usage has no cost. For the agents in
  `deepgym.agents` they match `AgentResult.usage` and `AgentResult.cost` up to float
  rounding (the trace sums costs exactly, the agent adds them one by one), with two
  exceptions. When nothing reported usage, `AgentResult.usage` has zero tokens while the
  trace's totals are null; `AgentResult.cost` is $0 only if no call was made (a run whose
  only request failed has a null cost in both). And an incomplete CLI run keeps its
  partial CLI cost in `AgentResult.cost`, labelled incomplete, while the trace's cost
  total is null.
- **Every result of `score()`:** the last is the verdict, with `has_verdict`, and earlier
  ones stay in `prior_results`.

Traces use a versioned, frozen schema (`deepgym.traces.Trace`, schema 1.2); schema 1.0
and 1.1 traces still load and export, without usage. Docker and Apple container report
the image digest from the container, and Daytona from its snapshot; images Daytona
builds, Modal, E2B and the local backend give a `digest_unavailable` reason instead.

## On disk

Each event is appended to `~/.deepgym/traces/<trace_id>.jsonl` (or
`$DEEPGYM_TRACES_DIR`) and flushed as it happens, so an exception or a killed process
keeps the steps taken so far, and a trace that was never scored is `incomplete`.
`deepgym.traces.load_trace(path)` reads one back; `path` is a `pathlib.Path`.

`record_episode(tr.trace)` from `deepgym.console` also stores it in the
[console](console.md), where episodes can be replayed step by step. Scores and expert
marks on a trace or one of its steps are stored as [labels](console.md#labels).

## Export

```bash
deepgym traces export --format jsonl --out traces.jsonl             # whole traces
deepgym traces export --format parquet --rows rl --out rl.parquet   # verl / TRL rows
deepgym traces export --format atif --out atif/ --only-valid        # ATIF documents
```

- Export reads the console database and the traces directory, or the sources named with
  `--db` and `--traces`. A trace stored in both is exported once, and refused if the
  copies disagree.
- A trace without a verdict is exported with `has_verdict: false`, or left out with
  `--only-valid`. `--rows rl` rows and trace-shaped Parquet rows give it a null reward;
  whole-trace JSONL and ATIF (under `extra.deepgym`) keep the result as recorded, with
  its placeholder score of 0.0.
- `--rows rl` writes one conversation per trace: `messages`, `reward`, `step_rewards`,
  the unattached usage and cost, the `usage` and `cost_usd` totals, `usage_complete`, and
  `steps` (every action with its observation text and screenshot, done flag, exit code,
  error, step reward, usage, cost and the index of the reply that produced it). It skips
  traces with a turn that has no model messages.
- Trace-shaped Parquet rows have `usage`, `cost_usd` and `usage_complete` columns.
- `--format atif` writes one ATIF v1.7 (Agent Trajectory Interchange Format) document per
  trace, with one agent step per turn and a tool call and result per action. A turn's
  usage goes in its agent step `metrics`, whose `llm_call_count` is the number of replies
  in the turn (null for a turn with no recorded messages, as a computer agent's), and the
  totals in `final_metrics`; when some usage is on no step, the root `notes` say
  `final_metrics` include it. ATIF's prompt tokens count every input token, cached and cache-written ones
  too, and cache writes and reasoning tokens, which ATIF has no field for, go in
  `metrics.extra`. It refuses a non-empty directory, so stale documents never mix in. The
  mapping is documented in the `deepgym.traces.atif` module.
- Exports are strict JSON: a NaN in verifier output becomes null.
- Parquet needs `pip install "deepgym[traces]"`.

From Python, `deepgym.traces.export_traces(traces, fmt, out, rows=..., only_valid=...)`
does the same for a list of traces; `out` is a `pathlib.Path`.

## Compare two runs

`deepgym.compare` compares two runs of the same tasks, such as model A against model B or
one checkpoint against another:

```python
from deepgym.compare import attempts_from_traces, compare

report = compare(attempts_from_traces(traces_a), attempts_from_traces(traces_b))
print(report.flips, report.significance)
for task in report.tasks:
    print(task.key, task.status, task.flip, task.delta)
```

- Each side is a list of `Attempt(key, score, has_verdict, solved, error_type,
  fingerprint)`, one per graded sample; solved means the verifier passed it (`passed`)
  with a verdict, as eval pass rates count it, so a rubric pass with a partial score is
  solved.
- `attempts_from_traces` keys a trace by `json.dumps([name, variant])` of its environment,
  e.g. `["todo_list", 3]`, with variant 0 when the trace recorded none (`dg.trace`
  records none unless you pass one; `load_environment` loads variant 0); an unnamed
  environment adds its task hash, `[null, variant, "<task sha256>"]`. The seed is not part
  of the key, so traces that differ only in seed are attempts at one task. The
  fingerprint is the environment hash, which covers the task and the verifiers, so a
  trace that ran another variant than its key says is still caught. Two traces with the
  same id are refused.
- Per task: each side's judged attempts, solves, solve rate (the task's pass@1) and mean
  score, B's change in both, and a flip: `improved` (every judged attempt failed on A and
  passed on B), `regressed` (the reverse), `both_solved`, `both_failed`, or `mixed` (a
  side solved only some of its judged attempts).
- Overall: the tasks on one side only (`only_a`, `only_b`), counts per flip, and per side
  the mean pass@k and pass^k with band counts over the compared tasks
  (`deepgym.passk.summarize`), for k = 1, each power of 2 below `k_limit`, and `k_limit`:
  the fewest judged attempts a compared task has on either side, set by the tasks in
  `k_limit_tasks`.
- `significance` is a two-sided p-value from a stratified permutation test (the exact
  conditional test behind Cochran-Mantel-Haenszel): if A and B were the same model, each
  task's judged attempts would be interchangeable between the sides, so the test takes B's
  total solves against every way of dealing each task's attempts between A and B. It
  holds with different attempt counts per task or side, and with one judged attempt per
  task and side it equals exact McNemar. It is computed in floating point (164 tasks
  sampled 100 times a side take well under a second): accurate to float precision down to
  about 1e-290, drifting below that, and read as 0 only once it is under about 1e-300.
- An attempt without a verdict is left out of every rate and counted per side by error
  type (`no_verdict`). A task whose attempts on one side all lack a verdict is
  `uncomparable`, not a fail. A task whose environment hash on A differs from B's is
  `changed`, and one with more than one hash on a side is `ambiguous`. None of these
  counts in the flips, the rates or the test. An attempt without a hash is no change, but
  its compared task is listed in `unverified`.
- The environment hash counts files by content, not path: a verifier read from a file by
  its source, and a Harbor task by its tests, its solution and, when it builds its image,
  its `environment/` folder: every entry's path, type and mode (all but owner and group write), each file's content, each
  link's target (not followed) and every folder, empty ones included. The files under a
  Terminal-Bench `task_root` and CyberGym `artifacts` still count by path only.

The [console](console.md) compares two run groups the same way.

## Scan traces

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
  `ProtectedPathScanner`'s is `protected-paths/3+config-sha256:...`. The part before `+`
  names how the scanner reads a trace and changes whenever that does.
- **`ProtectedPathScanner`** flags each applied action whose text names a path matching
  one of its `globs` as one it writes, deletes or moves (severity `high` by default).
  Nothing is run, and **only these forms are read**: a `write_file` path; in `bash`,
  redirection targets, the operands of `rm`, `rmdir`, `unlink`, `shred`, `touch`,
  `truncate`, `tee`, `mkdir`, `chmod`, `chown` and `chgrp`, the destination of `cp`,
  `install`, `ln`, `mv` and `rsync` and what `mv` moves away (with `-t DIR`, also
  bundled as `-rt DIR` or `-tDIR`, or `--target-directory[=]DIR`, `DIR/<name>` of each
  source; with `install -d`, every operand; long options match by unique prefix, as GNU's
  do, such as `--target-dir`; unrecognised options, a prefix that matches several, and a
  destination that can't be told, such as `-t` with no DIR, two DIRs or a single
  operand, make the scanner report every operand and option value), the files of
  `sed`, `perl` and `ruby` with `-i` (not the script: the first operand unless `-e`, `-f`
  or `--expression` gives it), the paths named after
  `git rm`, `git mv`, `git checkout`, `git restore` and `git clean` (which deletes
  untracked files), and `find ... -delete`, also after `if`, `then`, `do`, `!`, `{` and
  the like, `VAR=value`, and `sudo`, `env`, `nice`, `timeout`, `nohup`, `time`,
  `command` or `exec` with their options, and inside `bash -c`; in `code` (Python),
  literal paths given to `open` with a write mode (`file=` and `mode=` too), to `os` and
  `shutil` calls that change files (`path=`, `src=` and `dst=` too) and to `Path(...)`
  methods that do, and literal shell commands given to
  `os.system` or `subprocess`, with the code's imports resolved (`from pathlib import Path
  as P`, `import shutil as sh`, `import os.path as osp`; regardless of scope or order, a
  name bound in several places takes every meaning, and `os`, `shutil`, `subprocess`,
  `io`, `builtins` and builtins such as `open` keep their own, so a conditional or
  function-local `import foo as os` can't hide `os.remove`) and a literal counted through
  calls that keep the path (`str`, `os.fspath`, `os.path.abspath`, `realpath`,
  `normpath`, `Path(...).resolve()`) or join literals (`Path('a') / 'b'`,
  `Path('a', 'b')`, `os.path.join('a', 'b')`, `joinpath`); methods of anything else,
  such as `str(...).replace`, change no file; in `cli`, the files Claude Code's Write,
  Edit, MultiEdit and NotebookEdit calls and Codex's file changes touched, and the shell
  commands each ran, a Codex command seen only starting (a run cut off) included. An edit the tool reported failing is still reported, as an attempt:
  "tried to change <path> matching <glob>; the tool reported failure", with the same
  severity. A heredoc's body is data.
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
  relative path (`rm -rf .`, `git checkout .`). A folder holding a glob's fixed leading
  part counts too (`rm -r tests` changes `tests/conftest.py`). Each glob is one regex a
  path matches whole, so matching takes one pass per path and glob, however deep the path.
  A finding quotes the path where the action's text has it; for `write_file` and `cli`,
  its message names the path instead.
- **`LLMScanner(name, instruction, judge, scope='trace', judge_id=None)`** takes any
  `deepgym.rubric.JudgeClient` (`openai_judge`, `anthropic_judge` or your own); the client
  doesn't say which model it asks, so name it in `judge_id` for the report. The judge
  gets a fixed system prompt (`SCAN_PROMPT`; `SCAN_PROMPT_VERSION`, `trace-scan/2`, is the
  scanner's version) and one JSON user message: the instruction, the task, the initial
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
  with none. Any other scope is refused. With `scope='trace'`, a trace too long for the
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

## Datasets

`deepgym.dataset` turns traces into a versioned dataset: it keeps the traces a spec asks
for, drops duplicate trajectories, splits the rest into train, val and test by prompt,
and writes them to a folder with a manifest.

```python
from pathlib import Path

from deepgym.console.db import DEFAULT_DB_PATH
from deepgym.console.episode_store import read_trace_ids
from deepgym.console.label_store import read_labels
from deepgym.dataset import DatasetSpec, LabelMatch, build_dataset, verify_dataset, write_dataset

spec = DatasetSpec(
    models=('Qwen/Qwen3-8B',),
    bands=('learnable',),
    forbid_labels=(LabelMatch(name='mark', value='reward-hack'),),
    max_cost_usd=0.5,
)
labels = read_labels(DEFAULT_DB_PATH)        # every record, read-only
episodes = read_trace_ids(DEFAULT_DB_PATH)   # console episode id -> trace id
dataset = build_dataset(traces, spec, labels, episodes)
dataset.counts    # traces given, kept, per split, dropped per reason, unmatched labels
dataset.dropped   # {'no_verdict': ('t3',), 'band': ('t1', 't7'), ...}
sha256 = write_dataset(dataset, Path('notes-v1'), name='notes', version='1.0.0',
                       formats=['jsonl', 'parquet'])
verify_dataset(Path('notes-v1'), sha256=sha256)
```

```bash
deepgym dataset build --spec spec.json --name notes --version 1.0.0 --out notes-v1 \
    --format jsonl parquet
deepgym dataset verify notes-v1 --sha256 <the sha256 build printed>
```

The spec (`DatasetSpec`) is JSON, and an unknown key is refused:

| Field | Keeps |
|---|---|
| `require_verdict` (default `true`) | Traces with a verdict, the verifier's or a human's |
| `min_score`, `max_score` | Traces the verifier scored in that range, both ends included |
| `passed` | `true`: passed verdicts only; `false`: failed ones only |
| `agents`, `models` | Traces of these agents (`AgentMeta.name`) and models (`AgentMeta.model`) |
| `require_labels`, `forbid_labels` | Traces with a current label matching each `{name, value, source, author}` given (fields left out match anything), and without one matching any forbidden |
| `env_names`, `env_variants` | Traces of these environment names and registry variants |
| `require_usage_complete` | Traces whose model calls all reported usage |
| `max_cost_usd` | Traces that cost at most this; a trace with an unknown cost is dropped |
| `bands` | Traces whose task is `never_solved`, `learnable` or `always_solved` |
| `min_samples` (default 1) | Verdicts a task needs to have a band; one with fewer is dropped as `no_band` by a band filter |
| `band_range` (default `[0.1, 0.9]`), `inside_band` | With `inside_band`, traces whose task's solve rate is in `band_range` |
| `seed`, `test_ratio`, `val_ratio` (default 0, 0.1, 0.1) | Not a filter: how prompts are split |

- **Copies.** Traces of one environment definition with the same task text and
  trajectory are copies of one trajectory: the same initial observation text, and each
  step's action, message roles and texts, observation text, whether it was applied and its
  error; not times, usage, token data, screenshots or verdicts. The copies that pool their
  verdicts and labels are those in the population asked for (`agents`, `models`,
  `env_names`, `env_variants`) that aren't invalidated: their human `passed` labels make
  one verdict for every copy, and their labels count for `require_labels` on every copy. A
  label `forbid_labels` rules out, on any copy, drops every copy. An invalidation stays on
  its own trace: it discredits one run (its sandbox, setup or grade), not what the agent
  did, so another run of the same trajectory keeps its own grade; and so that a
  discredited run can't vouch for its copies, its human verdict and labels reach none of
  them (its forbidden marks still do).
- **Verdicts.** A trace's verdict is the pooled human `passed` verdict when there is one,
  which overrides the verifier, else its verifier's. Pooled copies whose human labels
  disagree are dropped as `contested`, and pooled copies the verifier both passed and
  failed, with no human verdict, as `inconsistent`. Their verifier scores that stand (not
  reversed by a human) agree when they are within `deepgym.models.SCORE_TOLERANCE` (1e-6)
  of each other; when they don't, the verdict still ships but with no score
  (`scores_disagree`), so a pass or fail filter keeps it and a score filter drops it as
  `no_score`. Judges' labels never override.
- **The copy kept** is, of those the other checks keep, one with a verdict, then one whose
  verifier no human reversed, then one with a standing score, then the highest score, and
  only then the earliest by start time and id; the rest are dropped as `duplicate`. So
  the verdict, score and `overridden` that ship never depend on which copy ran first;
  which of several equally judged copies ships still follows start time. `duplicates`
  lists every copy of each trajectory, the one kept (or none), and `verdicts_disagree`
  when the pooled humans disagree, the pooled verifier verdicts differ, or their scores
  disagree.
- **No verdict.** A trace without one never counts as a fail. It is dropped as
  `incomplete` (never scored) or `no_verdict` unless `require_verdict` is false, and
  always when a pass filter is set. A score filter needs a verifier score that stands: a
  trace only a human judged, whose verifier a human reversed, or whose copies' scores
  disagree is dropped as `no_score`.
- **Reasons.** Each dropped trace is listed once, under the first check it fails, in this
  order: `agent`, `model`, `env_name`, `env_variant`, `invalidated`, `contested`,
  `inconsistent`, `forbidden_label`, `incomplete`, `no_verdict`, `usage_incomplete`,
  `cost_unknown`, `over_cost`, `no_score`, `score`, `passed`, `failed`, `no_band`, `band`,
  `outside_band`, `missing_label`, `duplicate`. Kept and dropped traces add up to the
  traces given. `counts.overridden` counts kept traces whose verifier verdict, which their
  files hold, a human reversed (a human verdict where the verifier gave none isn't one),
  `counts.unread` what the sources held that never became a trace (console episodes
  recorded without one, `.jsonl` files that aren't traces, copies of a trace merged into
  another), and `counts.unmatched_labels` the current label records that matched none of
  the traces.
- **Labels.** Labels on `trace` targets count, and a label on a console episode, or on one
  of its replay frames, counts on the trace the episode was recorded from (`episodes`);
  run labels match no trace. A current invalidation drops the trace, and a retracted or
  superseded label no longer counts (`deepgym.labels.current_labels`). A label match looks
  at the whole trace and its steps, and a bool value never matches a number.
- **Bands.** A task is one environment definition (`EnvFingerprint.environment_sha256`).
  Its band (`deepgym.passk.task_difficulty`) counts the verdicts of the traces in the
  population asked for that aren't marked bad (invalidated, contested, inconsistent, or
  with a label `forbid_labels` rules out), whatever the later checks drop. `require_labels` selects among them after the bands are counted, so
  requiring an expert's `passed: true` keeps the expert-passed traces of a task the model
  solves half the time as `learnable`. The manifest records each task's samples, solved
  count, band and `outside_band` under `bands`.
- **Splits** are by prompt (`EnvFingerprint.task_sha256`), so traces with the same prompt
  land in the same split even in different environments. A prompt goes to test when
  `u = int(sha256(f'{seed}:{task_sha256}'), 16) / 2**256` is below `test_ratio`, to val
  when it is below `test_ratio + val_ratio`, and to train otherwise
  (`DatasetSpec.split_of`), so adding prompts never moves another. The ratios are chances,
  not exact shares. A build where no trace lands in train while some land in val or test
  is refused; lower the ratios or change the seed.

The folder holds, for each split with traces, `<split>.verdicts.jsonl` and, for each
format asked for, `<split>.jsonl`, `<split>.parquet` or `atif/<split>/` (whole traces, as
`deepgym traces export` writes them; ATIF always comes with JSONL, so the traces' hashes
can be checked), and `manifest.json`. `write_dataset` checks the name, version (one line
each, no whitespace at either end), formats and pyarrow first, writes into a new folder
next to the output (created under your umask), verifies it with `verify_dataset`, then
renames it to the output, and refuses an output that isn't a new or empty directory.

- **Read verdicts and rewards from `<split>.verdicts.jsonl`**, one JSON object per kept
  trace: `trace_id`, `passed`, `verdict_source` (`verifier` or `human`), `score` (the
  verifier's score, or null when it gave none, a human reversed it, or the copies' scores
  disagree) and `overridden`
  (whether a human reversed the verifier's verdict). The exported traces keep the
  verifier's result as it was recorded, so where a human reversed it or filled in a
  missing one, their `result`, `reward` and `passed` disagree with the dataset.
- The manifest has the name, version, `created_at`, `deepgym_version`, the formats, the
  spec, the counts, each kept trace's split, `environment_sha256`, `task_sha256`,
  `content_sha256` (its trajectory; actions count without their default fields, so a
  field added later doesn't change it), `sha256`, verdict, `verdict_source`, `score`,
  `scores_disagree` and `overridden` (`entries`), the dropped ids by reason, the duplicate
  groups, the task bands, the SHA-256 of every other file (`files`) and `content_sha256`.
- **The identifier to publish is the SHA-256 of the `manifest.json` file**, which
  `write_dataset` returns and `deepgym dataset build` prints (`sha256sum manifest.json`
  gives the same). Through `files`, it pins every file in the folder. `created_at` is the
  time of writing, so rebuilding the same dataset never reproduces a published SHA-256;
  to compare two builds, compare their manifests without `created_at` and
  `content_sha256`.
- A trace's `sha256` is the SHA-256 of its line in `<split>.jsonl`, without the newline;
  lines end at `\n` only (a trace's text can hold other Unicode line separators, so split
  on `\n` alone). It is also the SHA-256 of the trace's `trace` column in Parquet.
- `content_sha256` is the SHA-256 of every other manifest field as Python's
  `json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)` writes it.
  It catches an accidental edit, but anyone can recompute it.
- `verify_dataset(folder, sha256=...)` and `deepgym dataset verify` check that
  `manifest.json` has the SHA-256 given (in either case); that the manifest is consistent:
  its `content_sha256`, counts that match its entries and dropped ids, no trace both kept
  and dropped, each entry in the split its prompt and the spec give, no prompt in two
  splits, one kept copy per trajectory, duplicate groups that match the kept and
  duplicate traces, each task's band and `outside_band` recomputed from its counts and
  the spec, and no task keeping more verdicts or passes than its band counts; that the
  folder holds exactly the files listed, as regular files with
  their hashes, and no symlink, other file or empty directory (regular `.DS_Store` files,
  which macOS Finder writes, are skipped); that those include every file the formats and
  the verdicts need for each split with traces; that each JSONL and Parquet file holds
  exactly its split's traces, each once, each matching its entry (hash, trajectory,
  prompt and environment hashes, and the verifier's verdict and score the entry records;
  in Parquet, a `trace_id` column that is the trace's, never null) and passing the spec's
  checks that need no labels (agents, models, environments, verdict, usage, cost, score
  range, and band by the manifest's `bands`); that each JSONL and verdict file ends with
  a newline and each `<split>.verdicts.jsonl` is, byte for byte, the lines its entries
  give; and that each ATIF folder has one document per
  trace of its split. They raise listing every problem, and refuse a manifest with a key
  given twice.
- Without `sha256`, verifying shows only that the folder is self-consistent: someone who
  rewrote the files and the manifest together passes. It doesn't check the summary
  columns of Parquet rows or the content of ATIF documents beyond their file hashes, so
  check the published SHA-256.
- `deepgym dataset build` checks the output folder, name, version and formats before it
  reads anything. It reads traces from the sources `deepgym traces export` reads, and
  labels and episodes from that console database, the default one
  (`~/.deepgym/console.db`) when it exists, and `--labels`
  files written by `LabelStore.export_jsonl` (`deepgym.console.label_store.load_jsonl`
  reads one back, and refuses another file). It warns when it read no label records at
  all, lists on stderr each invalidation, human verdict and label the spec requires or
  forbids that matched no trace, and prints how many kept traces a human reversed.
- Both commands open console databases read-only, never creating or migrating one
  (`deepgym.console.episode_store.read_traces` and `read_trace_ids`,
  `deepgym.console.label_store.read_labels`); an older database reads as having none of
  what it lacks. The console keeps its database in WAL mode: in a folder you can't write,
  one with no uncommitted `-wal` content is read as it is on disk, and one with some can't
  be read (open it once where you can write, or copy it).
