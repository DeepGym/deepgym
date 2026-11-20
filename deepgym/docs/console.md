# Console

This page covers `deepgym console`, the local web console for the environments, runs,
episodes, sandboxes and QA reports on your machine: what each page shows, where data is
stored, the trace, scan, dataset and judge agreement routes, how the console protects
itself, and how to build or replace its UI. Labels on runs, episodes and traces and
review queues for experts have their own pages: [Labels](labels.md) and
[Review queues](review-queues.md).

```bash
pip install "deepgym[server]"
deepgym console --port 8080 --allow-local-exec   # http://127.0.0.1:8080
```

`deepgym web` is an alias.

## Pages

- **Environments:** search and filter the registry; open one to read its task, verifier,
  reference solution and metadata.
- **Runs:** score a pasted solution or the reference against one environment or a small
  eval, on any provider (`local`, `docker`, `apple`, `daytona`, `modal`, `e2b`). Runs
  execute in the background; the run page shows per-case results and a diff against the
  reference. A run without a verdict keeps no score and counts as an error, and the totals
  show a "No verdict" count with the judged pass rate and score beside the headline ones.
  `GET /api/runs/compare?a=<group_id>&b=<group_id>` compares two run groups environment
  by environment, as [`deepgym.compare`](compare.md) does; the key is the
  environment name. It answers 404 for a group with no runs, 409 while a group still has
  runs queued or running, and 422 when `a` or `b` is missing or given twice, or another
  query parameter is passed. Each run keeps the hash of the environment it ran, taken
  again when it starts, so a changed environment is flagged. A run whose environment
  changed while it ran keeps no hash, and neither do runs stored before the console kept
  one; both are listed as `unverified`. Starting a run whose
  environment can't be hashed (a file it names is unreadable, say) answers 400.
- **Episodes:** replay recorded trajectories step by step, with screenshots for
  computer-use tasks. Record one from your agent loop:

  ```python
  from deepgym.console import record_episode

  trajectory, result = dg.run_episode(env, agent)
  record_episode(trajectory, result, environment='todo_list')
  ```

  `record_episode(tr.trace)` stores a [trace](traces.md) the same way, and keeps it for
  `deepgym traces export`.
- **Sandboxes:** live `deepgym-*` containers from Docker and Apple `container`, with kill.
  Cloud sandboxes are not listed.
- **QA:** the reports `deepgym qa` writes to `~/.deepgym/qa/` (or `$DEEPGYM_QA_DIR`).

## Storage and execution

Runs, episodes, labels and review queues are stored in `~/.deepgym/console.db`, and
datasets are built into `~/.deepgym/datasets` (`--datasets-dir` to choose another). Reference
solutions may run locally, but a pasted solution only runs on your machine with
`--allow-local-exec` (or `DEEPGYM_ALLOW_LOCAL_EXEC=true`); otherwise pick a sandbox
provider.

## Traces

The console keeps the trace of each episode recorded from one (`record_episode(tr.trace)`).
A trace stored more than once, such as mid-episode and again finished, is read as one copy,
the one that got furthest. A stored trace that doesn't read as one trace never fails the
others: it is left out, and the trace list, scan and dataset build routes report every one,
whatever was asked, in two lists:

- `conflicts`: a trace id whose copies disagree (`copies_disagree`, as
  `deepgym.traces.merge_copies` refuses them), or whose id another trace's matches but for
  case (`case_collision`, which exports can't tell apart), with the `episodes` holding its
  copies and a `detail`.
- `unreadable`: the episodes whose stored trace doesn't parse as a trace, including JSON
  that SQLite's `json_valid` refuses (such as `NaN`), which every route treats alike.

Routes:

- `GET /api/traces` lists the traces newest first, by start time, as
  `{"traces": [...], "total": N, "conflicts": [...], "unreadable": [...]}`. Each has the
  summary columns of a Parquet export: `trace_id`, `env_name`, `env_variant`, `agent`,
  `model`, `provider`, `status`, `has_verdict`, `score` and `passed` (null without a
  verdict), `steps`, `usage`, `cost_usd`, `usage_complete` and `started_at`. Filters:
  `environment`, `agent`, `model` and `has_verdict`, with `offset` and `limit` (none by
  default).
- `GET /api/traces/{trace_id}` reads only the rows of that trace id (and of ids differing
  from it only in case) and returns `{"trace", "labels", "episode_labels", "unreadable"}`:
  the whole trace with its `usage` and `cost_usd` totals; its current labels, its steps'
  included, as `/api/labels/current` returns them; the current labels on the console
  episodes recorded from it, in `seq` order; and the episodes holding a copy that doesn't
  parse, which is merged without them. An episode numbers its replay frames, not the
  trace's steps: frame k shows step k's observation (the initial one for k = 0) and step
  k + 1's action, when no action was rejected. It answers 409 for an id in `conflicts`,
  and 404 when no copy of it parses.

## Scans

`POST /api/scans` runs rule scanners over the console's traces, as `deepgym traces scan`
does:

```json
{
  "scanners": [{"kind": "protected_paths", "name": "tests", "globs": ["tests/**"]}],
  "trace_ids": ["trace0001"],
  "write_labels": true,
  "run_id": "nightly-2026-11-20"
}
```

- `scanners` is the JSON list `--config` takes: `pattern` and `protected_paths` scanners.
  The API doesn't run LLM scanners.
- `trace_ids` picks traces, each scanned once; left out, every trace that reads is
  scanned. An unknown id gets 404, and one in `conflicts` 409.
- Each scan runs in a new, isolated Python process (`python -I -c`, started in deepgym's
  package folder), so a regex that backtracks for a long time holds neither the console's
  event loop nor its other requests. It never runs your script's `__main__`, never imports
  from the console's working directory, the user site or `PYTHON*` variables (it is sent the
  console's absolute `sys.path` entries instead), and gets its job and returns its report
  as JSON. There is no time
  limit: a scan runs until it finishes or its client leaves. When the client disconnects,
  the process is killed and the request answers 499 (no client is left to read it); a
  request cancelled another way, such as by the server stopping, kills and reaps it too. A
  scan process that ends without a report it can read, such as one killed by the system,
  answers 500 with its exit code, and stores no labels. Starting the process adds a fraction of a
  second to each scan.
- It returns the `ScanReport` (200), with `conflicts` and `unreadable` as above: its
  `run_id`, each scanner's source and version, the traces scanned, the findings and the
  scan errors. A trace a scanner failed on is in `errors`, and is not cleared.
- With `write_labels: true` it also stores a label per finding and per scan error
  (`finding_labels`) in one transaction: if the label store refuses one, none is stored and
  the request gets 409. They share the report's `run_id` as `group_id`, so
  `GET /api/labels?group_id=<run_id>` reads them back, and their ids are
  `<run_id>-<n>`. Give `run_id` yourself to retry safely: the same scan of the same traces
  under the same `run_id` stores nothing new, while one that found something else is
  refused with 409.
- The console doesn't keep scan reports; the labels are the durable record. Keep the
  response if you need the report itself.
- Scanners that share a name or are named like a verdict (`passed`, `c1`, ...) get 422, and
  so do an invalid regex and any other `kind`.

## Datasets

The console builds datasets into one directory chosen at launch: `~/.deepgym/datasets`, or
`deepgym console --datasets-dir PATH` (`create_app(datasets_dir=...)`), with `~` expanded
and the path resolved once at launch; an empty `--datasets-dir` is refused. A dataset is the
folder `<name>-<version>` in it. Requests name that folder, never a path: a name or version
holding `/`, `\` or `:`, a folder name starting with `.`, or one longer than 221 bytes in
UTF-8 (`deepgym.dataset.manifest.MAX_FOLDER_BYTES`: the 255 bytes file systems allow, less
what the build's staging folder adds) gets 422. No answer holds a path on the server:
paths are relative to the directory, which reads `<datasets>`.

- `POST /api/datasets/build` with `{"spec": {...}, "name": "notes", "version": "1",
  "formats": ["jsonl"]}` builds a dataset from every trace the console holds that reads as
  one and every label record in its database, as `deepgym dataset build` does, and writes
  it to `notes-1` (201). `spec` is a `DatasetSpec`, and `formats` defaults to `["jsonl"]`.
  It returns `{"dataset": "notes-1", "sha256", "counts", "conflicts", "unreadable"}`: the
  SHA-256 of `manifest.json` to publish, and the counts (kept, per split, dropped per
  reason, overridden, unmatched labels, and `unread`: episodes recorded without a trace,
  merged copies, `conflicting_copy` for each copy of a trace in `conflicts` and
  `unreadable_trace` for each stored trace that doesn't parse), which the manifest
  records too. It gets 409 when the folder exists and isn't empty or another build created
  it meanwhile, the datasets directory can't be written, Parquet is asked for without
  pyarrow, no stored trace reads as one, or the build refuses the traces (no trace landed
  in train while some landed in val or test).
- `GET /api/datasets` lists the folders of the directory by name as
  `{"datasets": [...], "total": N}`, with `offset` and `limit` (none by default). Each has
  `dataset`, `sha256` (of its `manifest.json`), `manifest` as read, not verified, and
  `error`: why a folder has no manifest that reads (none, a symlink, one that isn't a
  manifest, or one that can't be read). Hidden folders, such as a build's staging folder,
  and symlinks are left out. A directory that can't be read gets 409.
- `POST /api/datasets/verify` with `{"dataset": "notes-1", "sha256": "..."}` checks the
  folder against its manifest, and the manifest against `sha256` (either case) when given,
  as `deepgym dataset verify` does. It returns `{"dataset", "valid", "sha256", "manifest",
  "problems"}`: a folder that doesn't match is `valid: false` with every problem listed
  (`DatasetMismatchError.problems`), not an error, and so is one whose manifest is missing,
  a symlink, not a manifest, or can't be read. Without `sha256`, `valid` only means the
  folder is self-consistent. A folder the directory doesn't hold, or a symlink, gets 404,
  and a datasets directory that can't be read 409.

## Judge agreement

`GET /api/calibration/agreement` measures a judge's stored verdicts on runs against the
experts', from labels only: it calls no judge.

- Pick the runs one way: `run_ids` (comma-separated), `queue_id` (the queue's run items;
  other items are left out), or label filters: the `GET /api/labels` filters prefixed with
  `label_`, such as `label_author=ana&label_current=true`, matched against labels on runs.
  No selection, or more than one, gets 422; an unknown queue 404.
- `judge_author` (required) and `judge_source` (`judge` by default, or `verifier`) say whose
  stored verdicts are the judge's. A console run's own verifier result lives in the run
  (its `score` and `passed`), not in labels: `judge_source=verifier` reads only verifier
  labels someone stored, such as through `POST /api/labels`.
- `rubric_sha256` leaves out verdicts recorded against another rubric hash, the experts'
  and the judge's; those recording none are kept. Without it, verdicts against different
  rubrics are pooled: `rubric_hashes` and `judge_rubric_hashes` show whether they are.
  `judge_rubric_version` keeps only the judge's verdicts recorded with that
  `rubric_version` (a judge prompt version, say); those recording another or none are left
  out.
- The experts' side is `labeled_answers`: the experts' majority on each verdict. Each judge
  call, the judge's verdicts on a run sharing a `group_id`, is one judged run: it passes
  when its `passed` is true, and a check is MET when the judge marked it true.
- It returns the numbers a `CalibrationReport` computes from those: `overall`, `blind` and
  `assisted` (the judge's pass against the experts', with TPR, TNR, agreement, precision,
  F1 and kappa), `check_agreement` per check and `check_summary`, `self_consistency` and
  `check_consistency` over repeated calls, `blind_baseline` and `assisted_baseline`,
  `tied_answers`, `check_ids`, and `answers`, each run kept with the experts' labels and the
  judge's calls. `expert_only_checks` lists the check ids the experts gave that the judge
  never did, which count in no judge number. `judged_calls`, `no_passed_calls` and
  `no_judge_runs` count the calls used, the calls left out and the runs left out.
- `skipped` lists every run, expert, verdict and judge call left out, in the order of the
  runs, with the reasons `labeled_answers` gives and, for the judge, `rubric_mismatch`
  (another rubric hash or version), `no_passed` (a call without `passed`), `no_judge` (no
  call with `passed` on the run, which is left out) and `missing_checks` (a call that gave
  some check ids but not every one the judge gave on these runs: its check verdicts are left
  out, its `passed` kept).
- `rubric_hashes` and `no_rubric_hash` count the rubric hashes on the experts' verdicts
  used; `judge_rubric_hashes`, `judge_no_rubric_hash`, `judge_rubric_versions` and
  `judge_no_rubric_version` count the hashes and versions on the judge's.

An unknown or repeated query parameter is refused with 422 on every route above.

## Security

The console has no login, so other websites can't drive it:

- Every `/api` call needs the per-launch token the console writes into its pages, sent as
  `X-DeepGym-Token`.
- Writes from other origins are refused, and so are write bodies that aren't JSON.
- Bound to `127.0.0.1` or `localhost` (the default), it only answers requests addressed to
  localhost. Bound to any other host, it doesn't check the Host header.

## Building the UI

The UI lives in `console-ui/` (Next.js static export, built with bun) and ships prebuilt in
wheels. From a clone, build it once with `cd console-ui && bun install && bun run build`.

## Serving your own UI build

To run a different static export against the same API, pass its folder:

```bash
deepgym console --ui-dir ./my-ui/out   # or DEEPGYM_CONSOLE_UI_DIR=./my-ui/out
```

The folder must hold an `index.html`; if it doesn't exist or has none, the console refuses
to start rather than serve the bundled UI. It is served the way the bundled UI is: every
HTML page gets `<meta name="deepgym-token" content="...">` before its first `</head>` and is
never cached, and other files are served as they are. Your pages read the token from that
tag and send it as `X-DeepGym-Token` on every `/api` call; the rules under
[Security](#security) apply unchanged. From Python, pass `create_app(static_dir=...)`.

Point it only at a folder you trust and that only you can write to. Everything in it is
served as it is on disk, without the token, to anything that can reach the port (dotfiles
too; symlinks out of the folder are refused). Any page or script in it runs on the
console's origin, can read the token and can call the whole API, which with
`--allow-local-exec` runs code on this machine.
