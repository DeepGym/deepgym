# Datasets

This page covers building a versioned dataset from recorded [traces](traces.md) and
their [labels](labels.md): the spec that filters it, how copies, verdicts and splits are
decided, the folder and manifest it writes, and how to verify a published one.

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
renames it to the output, and refuses an output that isn't a new or empty directory, or
one another build created while it ran.

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
  trace of its split. A folder that doesn't match raises `DatasetMismatchError` (a
  `DeepGymError`, from `deepgym.dataset`) whose `problems` lists every problem, one string
  each, and whose message lists them too; it pickles with them. They refuse a manifest with
  a key given twice, and a `manifest.json` that is a symlink (it is never followed, nor by
  `manifest_sha256`).
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
