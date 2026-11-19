# Console

This page covers `deepgym console`, the local web console for the environments, runs,
episodes, sandboxes and QA reports on your machine: what each page shows, where data is
stored, labels on runs, episodes and traces, review queues for experts, the trace, scan,
dataset and judge agreement routes, and how the console protects itself.

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
  by environment, as [`deepgym.compare`](traces.md#compare-two-runs) does; the key is the
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

## Labels

A label scores one thing about a run, an episode or a trace, or one step of it: a verifier
score, a judge's verdict, or an expert's mark with the reason. Labels are kept in the
console database as an append-only log.

```python
from deepgym.console.db import DEFAULT_DB_PATH
from deepgym.console.label_store import LabelStore
from deepgym.labels import LabelCreate, LabelTarget

store = LabelStore(DEFAULT_DB_PATH)
trace = LabelTarget(kind='trace', id=tr.trace.trace_id)
store.add(LabelCreate(
    target=trace, name='passed', value=False, source='human', author='ana', blind=True,
    comment='Opens port 80 instead of fixing the config it was asked to fix.',
))
store.add(LabelCreate(
    target=LabelTarget(kind='trace', id=tr.trace.trace_id, step=3, field='action',
                       span=(0, 25), quote='sed -i s/8080/80/ app.ini'),
    name='mark', value='wrong-file', source='human', author='ana',
))
```

- `target`: `kind` (`run`, `episode` or `trace`) and `id`, then optionally:
  - `step`: a trace numbers steps by `TraceStep.index`, from 1, and an episode by replay
    frame, from 0; a run has no steps. A console episode has its own id, even when it was
    recorded from a trace. When that trace has no rejected actions (the replay leaves them
    out), frame k shows trace step k's observation (the initial observation for k = 0)
    and trace step k + 1's action and reward.
  - `field`: which text. `answer` is a run's solution, the text it was graded on;
    episodes and traces have no answer text. With a step: `action` (`Action.content`),
    `observation` (`Observation.content`) or `message:<i>`
    (`TraceStep.messages[i].content`, traces only).
  - `span`: `(start, end)` offsets into that text, end exclusive, counted in Unicode code
    points (Python string indexes; a UI that counts UTF-16 units converts). A span needs a
    `field`.
  - `quote`: the spanned text as the labeler saw it, exactly `end - start` code points, so
    a span that no longer matches its text can be found.
- `name`: what is scored. `value`: `true` or `false`, a finite number, or a one-line
  category.
- `source`: `verifier`, `judge` or `human`. `author`: who labeled.
- `comment`: why. Required on a human's `passed` verdict, a right answer's too, and on
  every invalidation and retraction; optional, and worth offering, on every other label.
- Optional: `blind` (`true` when labeled without seeing other labels or judge output),
  `rubric_version`, `rubric_sha256`, `group_id` (shared by the labels of one judge call or
  one repeat), and `id`: a client id for safe retries, generated when left out.
- The store adds `seq` and `created_at`. `seq` increases with each label stored, and
  `created_at` never runs back, so they sort the same.
- Ids, names, authors and categories are one line that doesn't start or end with
  whitespace, with no control or format characters. A comment or quote may be any valid
  Unicode text.

### Verdicts

Experts and judges give verdicts the same way. The verdict on an answer is a bool named
`passed`, and the verdict on one rubric check a bool named by its check id (`c1`, `c2`,
...), `true` meaning MET. These names take only bools, and only on a whole target: a mark
on a step or a span of text takes another name, such as `mark`. Each source and author
holds one current verdict per target, name and `group_id`: to change it, supersede it.
Each repeat of a judge gets its own `group_id`.

`deepgym.console.calibration_labels.labeled_answers` turns experts' verdicts on runs into
`rubric_calibration.LabeledAnswer` items for judge calibration, with the run's solution as
the answer text; see [From stored labels](training/rubric-judge.md#from-stored-labels).

### Corrections, retractions and invalidations

Records are never edited, and there are three types:

- `label` (the default) scores the target.
- `invalidate` excludes the whole target, its steps included, with the reason in
  `comment`. It has no name, value, step or field.
- `retract` withdraws the record named in `supersedes`, with the reason in `comment`, and
  has no name, value, step or field. Retracting a correction withdraws its whole chain,
  the records it corrected included. Retracting an invalidation lifts it.

A correction is a new record of the same type whose `supersedes` names the one it
replaces. A correction or retraction must be on the same target, from the same source and
author, except that any human may retract an invalidation, whoever made it. Each record
is superseded at most once, and a retraction is final. A different labeler's view is a new
label, so disagreement stays visible.

### Reading labels

`LabelQuery` is in `deepgym.console.schemas`.
`store.page(LabelQuery(...), offset, limit)` filters by `type`, `target_kind`,
`target_id`, `step`, `name`, `source`, `author`, `blind`, `rubric_version`,
`rubric_sha256`, `group_id`, `since` (inclusive), `until` (exclusive), `after_seq` (labels
stored after that one, for syncing) and `current`, in `seq` order, and returns the total;
`limit=None` returns every match. `current=True` keeps only records still in force (no
retraction, and nothing supersedes them), checked against the whole store, so it combines
with any other filter: query what you need with it instead of downloading everything.
`store.current(kind, id)` returns one target's current records, steps included, and
`store.excluded()` the `(kind, id)` of each target a current invalidation excludes; drop
those targets before using labels. For records already in hand, such as an export,
`deepgym.labels.current_labels(labels)` and `deepgym.labels.excluded(labels)` do the same,
given every record of the targets in question.

Records read back as they were written: a rule added since a record was stored never
makes it unreadable. `store.add_many(labels)` stores a batch in one transaction: one
refused record stores none. `store.export_jsonl(path)` writes every record, superseded ones included, one JSON
object per line.

The console serves the same under `/api/labels`:

- `POST /api/labels` stores one label, `POST /api/labels/batch` a list. An invalid record
  is refused with 422. A record the store refuses gets 409, when:
  - the record it supersedes is not stored, is on another target, is from another source
    or author (a human retracting an invalidation excepted), is a retraction, has another
    type than a correction of it, or is already superseded;
  - it is a verdict its source and author already hold on that target, name and
    `group_id`;
  - its id is stored with another body.

  Posting the same id and body again returns the stored record.
- `GET /api/labels` takes the filters above as query parameters, with `offset` and
  `limit` (none by default), and returns `{"labels": [...], "total": N}`.
- `GET /api/labels/current?target_kind=trace&target_id=...` returns a target's current
  records.
- `GET /api/labels/excluded` returns the excluded targets as `{"kind": ..., "id": ...}`.
- An unknown or repeated query parameter is refused with 422.

## Review queues

A review queue hands items to expert reviewers and stores their reviews as labels: to
validate a judge against experts, or to label traces. An item is a run, an episode or a
trace, or one step or text of one (a `LabelTarget`), with optional context such as the
task text.
Each item needs `overlap` reviews from different reviewers, and no reviewer reviews an item
twice. Queues are kept in the console database.

```python
from deepgym.console.db import DEFAULT_DB_PATH
from deepgym.console.queue_store import QueueStore
from deepgym.labels import LabelTarget
from deepgym.review_queues import ItemCreate, QueueCreate, ReviewLabel

queues = QueueStore(DEFAULT_DB_PATH)
queue = queues.create(QueueCreate(
    name='judge v3 calibration', instructions='Pass only if the config is fixed.',
    label_names=('passed', 'c1', 'c2'), overlap=3, blind=True, reveal='on_close',
    reservation_seconds=1800,
))
queues.add_items(queue.id, [
    ItemCreate(target=LabelTarget(kind='run', id=run_id), context=task_text)
    for run_id, task_text in runs
])

view = queues.next_item(queue.id, 'ana')  # None when nothing is left for ana
content = queues.content(queue.id, view.item.id, 'ana')  # task, answer, steps; no grades
queues.submit(queue.id, view.item.id, 'ana', [
    ReviewLabel(name='passed', value=False, comment='Opens port 80 instead.'),
    ReviewLabel(name='c1', value=True),
    ReviewLabel(name='c2', value=False),
])
```

### Settings

- `name`, and optional `description` and `instructions` (what reviewers are asked to do,
  for the UI to show with every item).
- `label_names`: names every review must give a label on the item's target, such as
  `passed` and the rubric's check ids. A review may give other labels too, such as a
  `mark` on a step or a span. A queue that asks for a verdict (`passed` or a check id)
  takes only whole targets as items, since verdicts are on whole targets.
- `overlap` (default 1): how many different reviewers review each item.
- `blind` (default `true`): see [Blind review](#blind-review).
- `reveal` (default `after_submit`): which labels a reviewer sees once the queue would show
  them; see [Reveal policy](#reveal-policy).
- `reservation_seconds` (default 3600): how long a reviewer holds an item they were
  handed before it is free again.
- `reviewers`: who may ask for, view and review items; empty (the default) lets anyone
  with console access.
- `id`: optional client id. Creating the same id with the same settings again returns the
  stored queue; other settings are refused.

`update(queue_id, QueueUpdate(...))` changes what may change: `closed`, `description`,
`instructions`, `reservation_seconds` (for reservations made or extended from then on) and
`reviewers`, which takes the whole new list and only adds to it; a queue that lets anyone
review keeps doing so. `reveal` changes only until it has decided what a reviewer saw: in
a blind queue until a review is submitted, and in one that isn't until an item is handed
out, since reviewers there see labels from their first turn. `overlap`, `blind` and
`label_names` never change. A closed queue hands out no items; a reviewer who
still holds a live reservation may submit it.

### Items and reservations

`add_items` adds items in order, in one transaction. An item whose target the queue
already holds is kept once: adding it again with the same context returns the stored
item, and with another context it is refused, and so is the whole batch.

`next_item(queue_id, reviewer)` reserves the first item, in the order added, that the
reviewer hasn't reviewed or skipped and whose reviews plus live reservations are fewer
than `overlap`; items the reviewer released come after the others. A reviewer holds one
reservation per queue at a time: asking again returns the item they hold, its time
renewed (never shortened, if `reservation_seconds` was lowered). Reservations run out
after `reservation_seconds`, and the item is then free for
someone else. It returns a `ReviewView`: the item, the reviewer's latest turn at it
(`assignment`), and the labels they may see.

A turn ends one of these ways (`assignment.status`):

- `submitted`: `submit(queue_id, item_id, reviewer, labels)` stores the review's labels in
  the label store and ends the turn, in one transaction, so either both are stored or
  neither is. Each label is a `ReviewLabel` (`name`, `value`, optional `comment`,
  `target`, `id`, `rubric_version`, `rubric_sha256`); the queue sets `source='human'`,
  `author` to the reviewer, `blind` (see below) and `group_id` to the assignment's id, so
  one review's labels are `group_id=<assignment id>`. A label's `target` defaults to the
  item's and may be inside it: the same kind and id, the same step and field where the
  item has one, and a span inside the item's. The usual label rules apply: a `passed`
  verdict needs a comment. Two labels of one review can't share a client `id`. Submitting
  the same labels again (the same bodies in order, and
  the same ids where given) returns the stored review; other labels are refused.
- `released`: `release(...)` hands the item back unreviewed; it may be handed to the same
  reviewer again, after the items they haven't released.
- `skipped`: `skip(...)` hands it back, never to be handed to that reviewer again.
- `expired`: the reservation ran out.

A review is taken on a live reservation, or on one that expired or was released while the
item still has a free slot (fewer reviews and live reservations than `overlap`) and the
queue is open. It is refused on a skipped turn, or when the reviewer never had one. Every
write takes the database's write lock first and reads the time under it, so reservations
never exceed an item's overlap, across threads and processes.

`view(queue_id, item_id, reviewer)` returns an item as that reviewer may see it.

Labels added to the label store after a review change nothing in the queue: retracting a
review's labels doesn't reopen its item, which still counts the review. An item whose
target is invalidated (`store.excluded()`) is still handed out; close the queue, or skip
the item.

### Item content

`content(queue_id, item_id, reviewer)` returns what a reviewer needs to grade an item, as
an `ItemContent`:

- `item`: the queue item, with its `context`.
- `task`: a trace's task, and an episode's when it was recorded from a trace; a run's,
  read from the registry, when its environment still hashes to what the run ran.
  Otherwise none: give the task as the item's `context`.
- `answer`: a run's solution, the text it was graded on.
- `initial_observation` (a trace's) and `steps`: every step of the recording, whatever
  step the item is. Each step has its `index`, as the item's target numbers it; its
  `action` (an `Action` without a CLI agent's `events`); its `observation` text; the
  `exit_code` of its own action (for an episode, read from the next frame, which shows
  what the action produced); `has_screenshot`; `applied` (false for a trace step whose
  action was rejected); a trace step's `error`; and a trace step's `messages` (role and
  content).

It never returns a score, pass, reward, step verifier output, verifier result, error type,
verifier source, usage, cost or label: the response model lists every field it returns.
Two fields are cut where they could carry one:

- A trace step's `error` is the whole text only when the step wasn't applied (refused
  before anything ran) or the trace has no step verifier. Otherwise it is only the
  exception's name, such as `SandboxDiedError`: the message of an error raised while the
  step verifier ran can quote what it printed. A text that names no exception gives none.
- Output a CLI action produced (Claude Code stream-json, Codex `--json`) loses the usage,
  token count and cost keys of each line that is JSON, at any depth; other lines are kept
  as stored. A span marked on a changed line counts in the text shown, not the stored one.

Content is otherwise what the agent saw and did, so an environment that showed the agent
its reward in an observation or message shows it to the reviewer too. A trace stored more
than once is read as `GET /api/traces/{trace_id}` reads it: its copies merged into the one
that got furthest. Copies that disagree, or another trace whose id matches but for case,
get 409, as there.

`screenshot(queue_id, item_id, reviewer, frame)` returns the PNG shown at one frame of an
episode item, the steps whose `has_screenshot` is true, under the same rules as `content`.
A trace item's steps have no screenshots there: to review a computer-use trace's
screenshots, queue the episode recorded from it, which is the same recording for blind
review.

A reviewer may read an item's content and screenshots once they have reviewed it, and
while they could submit a review of it: a live reservation, or one that expired or was
released while the item has a free slot in an open queue (see above). Anyone else is
refused (409 over REST), and a reviewer the queue doesn't list gets 403. An item whose
run, episode or trace the console doesn't hold (a trace kept only in a traces directory,
say) gets 404, and so does a frame with no screenshot or an item that isn't an episode. A
trace whose copies don't read as one trace gets 409 (above). A stored recording that
doesn't parse, or a file its content needs that can't be read, gets 500 with a fixed
message that quotes nothing stored.

### Blind review

Blind review works per recording: a run, a console episode or a trace, by its kind and id,
every step and span included. A console episode recorded from a trace
(`record_episode(tr.trace)`) is on that trace's recording, so an item on the episode and one
on the trace are treated as one recording below. The link is read from the episode when the
item is added; an episode the console doesn't hold yet is its own recording.

In a blind queue, a reviewer sees no label on the item's kind and id until they have
submitted their review of it. Then they see only their review's own labels while they may
still review another item on the same recording in a blind queue that lets them review:
one they haven't reviewed or skipped, that still needs reviews, and that is in an open queue
or held by them (a live reservation) in a closed one. Once none is left, they see every
current label on the item's kind and id. Outside blind mode, a reviewer sees every current
label once they have had a turn at the item.

Close a blind queue that won't be finished: while its items stay open to a reviewer, their
reviews of the same recordings in other blind queues show them only their own labels.

Whenever a queue response (`next_item`, `view` or `submit`) returns a label that isn't the
reviewer's own (a judge's, a verifier's or another expert's), the console records that
they have seen labels on that recording, and when they first did. A review's labels are
`blind: true` only in a blind queue, and only if no such exposure is recorded for the
reviewer on the item's recording and they had no turn at an item on it in a queue that isn't
blind. So a reviewer who works through every step of a trace in a blind step queue, and
only ever sees their own labels, reviews every step blind. Exposures are recorded only for
what queue responses return: labels seen anywhere else (below) are not. The scope decision,
the label read and the exposure are made in the request's own transaction.

### Reveal policy

`reveal` narrows which labels `next_item`, `view` and `submit` return once the rules
above would show them:

- `after_submit` (the default): as above.
- `humans_only`: only humans' labels, never a judge's or a verifier's.
- `on_close`: only the reviewer's own review's labels while the queue is open; once it is
  closed, as `after_submit`.

Use `on_close` to validate a judge: while the queue is open, its responses show no
reviewer the judge's verdict, the verifier's score or another expert's labels. Use
`humans_only` when reviewers may compare notes with each other but must never anchor on a
judge or verifier, and `after_submit` when seeing the judge after submitting is wanted
feedback, such as when training reviewers.

Exposures follow what a response returns, under every policy: a judge's, a verifier's or
another expert's label shown is an exposure. Under `humans_only` a reviewer is exposed once
another human's label is shown to them; under `on_close` nothing is exposed while the queue
is open. In a queue that isn't blind, `humans_only` shows only humans' labels from the
reviewer's first turn and `on_close` only their own review's until the queue is closed.

A reviewer in an `on_close` queue can still be exposed, and their later reviews of that
recording are then marked not blind:

- by a turn at an item on the same recording in a queue that isn't blind, whatever it
  showed them;
- by another queue's response on the same recording that shows labels, such as an
  `after_submit` queue;
- by viewing an item of the `on_close` queue while it is closed: reopening it then leaves
  them exposed for the items still open to them.

### What blind review doesn't cover

Blind mode, the reveal policy and the content route cover what the queue routes return, and
nothing else. The local console has no login: any client can pass another reviewer's name,
and a client that calls the general routes can still see grades:

- `GET /api/labels` and `GET /api/labels/current` list labels, the judge's included.
- `GET /api/runs` lists runs with their score and pass, `GET /api/runs/stats` totals them
  with score bins, `GET /api/runs/compare` compares groups' solve rates, and
  `GET /api/runs/{id}` returns a run's score, pass and verifier result.
- `GET /api/episodes` lists episodes with their score and pass, and `GET /api/episodes/{id}`
  returns an episode's score, rewards and result.
- `GET /api/environments/{name}` returns the verifier's source and the reference solution.

A reviewer UI should call only the queue routes (`next`, the item view, `content`, the
frame screenshots, `submit`, `release` and `skip`). A hosted deployment must enforce this
server-side: give reviewers access to the queue routes only, and tie each reviewer name to
a login.

### Progress

`progress(queue_id)` returns `items`, `complete` (every review they need), `in_progress`
(a review or a reservation, but not complete), `reviews` submitted, `reviews_needed`
(items times overlap), `reserved` (reservations held now), `blocked` (items that can no
longer get every review they need because too many of the queue's listed reviewers skipped
them; adding reviewers unblocks them), and per reviewer the count of turns `submitted`,
`reserved`, `skipped`, `released` and `expired`.
`items(queue_id, offset, limit)` lists the items in order with their `reviews`, `needed`
and `reserved` counts, and the total; `limit=None` (the default) returns every one.

### REST

The console serves the same under `/api/queues`:

- `POST /api/queues` creates a queue (201); `GET /api/queues` lists every queue;
  `GET /api/queues/{id}` returns one; `PATCH /api/queues/{id}` takes the `QueueUpdate`
  fields, such as `{"closed": true}`; fields left out keep their value.
- `POST /api/queues/{id}/items` adds a list of `{"target": ..., "context": ...}` (201);
  `GET /api/queues/{id}/items` lists them with their counts, with `offset` and `limit`
  (none by default), as `{"items": [...], "total": N}`.
- `GET /api/queues/{id}/progress` returns the progress.
- `POST /api/queues/{id}/next` with `{"reviewer": "ana"}` returns a `ReviewView`, or `null`
  when nothing is left for that reviewer now.
- `GET /api/queues/{id}/items/{item_id}?reviewer=ana` returns the item as ana may see it,
  recording an exposure when it shows her someone else's label.
- `GET /api/queues/{id}/items/{item_id}/content?reviewer=ana` returns the item's content
  while ana could submit a review of it or once she has (see
  [Item content](#item-content)), 404 when the console doesn't hold its run, episode or
  trace, 409 when a trace's copies don't read as one trace, and 500 when the stored
  recording can't be read.
  `GET /api/queues/{id}/items/{item_id}/frames/{frame}/screenshot?reviewer=ana` returns
  the PNG shown at that frame of an episode item, under the same rules, and 404 when it
  has none.
- `POST /api/queues/{id}/items/{item_id}/submit` with `{"reviewer": "ana", "labels":
  [...]}` stores a review (201). `.../release` and `.../skip` take `{"reviewer": "ana"}`.

An unknown queue or item gets 404, and a reviewer missing from the queue's `reviewers` 403.
Input that breaks a rule gets 422, with each problem's `loc` in the body: an invalid body,
a review with no labels, missing a `label_names` label, with two labels sharing an `id`, or
with a label outside the item's target or breaking a label rule (`loc`
`["body", "labels", i, ...]`), and a step or text item in a queue that asks for a verdict.
Any other refusal gets 409: a closed queue, a turn that can't be submitted, content asked
for by a reviewer who couldn't submit a review of the item and hasn't, an item already
reviewed with other labels, a verdict the label store refuses, a reviewer list that drops
someone, a `reveal` change once it has decided what a reviewer saw, or a client id stored
with other settings. An unknown or repeated query parameter is refused with 422.

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
