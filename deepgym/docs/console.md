# Console

This page covers `deepgym console`, the local web console for the environments, runs,
episodes, sandboxes and QA reports on your machine: what each page shows, where data is
stored, labels on runs, episodes and traces, review queues for experts, and how the console
protects itself.

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

Runs, episodes, labels and review queues are stored in `~/.deepgym/console.db`. Reference
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
    label_names=('passed', 'c1', 'c2'), overlap=3, blind=True, reservation_seconds=1800,
))
queues.add_items(queue.id, [
    ItemCreate(target=LabelTarget(kind='run', id=run_id), context=task_text)
    for run_id, task_text in runs
])

view = queues.next_item(queue.id, 'ana')  # None when nothing is left for ana
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
- `reservation_seconds` (default 3600): how long a reviewer holds an item they were
  handed before it is free again.
- `reviewers`: who may ask for, view and review items; empty (the default) lets anyone
  with console access.
- `id`: optional client id. Creating the same id with the same settings again returns the
  stored queue; other settings are refused.

`update(queue_id, QueueUpdate(...))` changes what may change: `closed`, `description`,
`instructions`, `reservation_seconds` (for reservations made or extended from then on) and
`reviewers`, which takes the whole new list and only adds to it; a queue that lets anyone
review keeps doing so. `overlap`, `blind` and `label_names` never change. A closed queue
hands out no items; a reviewer who still holds a live reservation may submit it.

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

Blind mode covers what the queue routes return, and nothing else:

- `/api/labels` lists every label.
- The run and episode routes show the verifier's score and pass, and an episode's rewards.
- The console has no login: any client can pass another reviewer's name.

A reviewer UI for a blind queue should read labels through the queue routes only, and
show nothing of the target that the queue doesn't.

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
- `POST /api/queues/{id}/items/{item_id}/submit` with `{"reviewer": "ana", "labels":
  [...]}` stores a review (201). `.../release` and `.../skip` take `{"reviewer": "ana"}`.

An unknown queue or item gets 404, and a reviewer missing from the queue's `reviewers` 403.
Input that breaks a rule gets 422, with each problem's `loc` in the body: an invalid body,
a review with no labels, missing a `label_names` label, with two labels sharing an `id`, or
with a label outside the item's target or breaking a label rule (`loc`
`["body", "labels", i, ...]`), and a step or text item in a queue that asks for a verdict.
Any other refusal gets 409: a closed queue, a turn that can't be submitted, an item already
reviewed with other labels, a verdict the label store refuses, a reviewer list that drops
someone, or a client id stored with other settings. An unknown or repeated query parameter
is refused with 422.

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
