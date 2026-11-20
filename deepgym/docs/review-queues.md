# Review queues

This page covers review queues in the [console](console.md): their settings, how items
are reserved and reviewed, what a reviewer is shown, blind review and the reveal
policy, progress, and the `/api/queues` routes. Reviews are stored as
[labels](labels.md).

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

## Settings

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

## Items and reservations

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

## Item content

`content(queue_id, item_id, reviewer)` returns what a reviewer needs to grade an item, as
an `ItemContent`:

- `item`: the queue item, with its `context`.
- `task`: a trace's task, and an episode's when it was recorded from a trace; a run's,
  read from the registry, when its environment still hashes to what the run ran. A run
  of a benchmark adapter that picks its task per instance when the run starts
  (`swebench_pro`, `terminal_bench_2`) has none: the adapter's own task is a generic
  instruction, not the instance's problem. Otherwise none too: give the task as the
  item's `context`.
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

## Blind review

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

## Reveal policy

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

## What blind review doesn't cover

Blind mode, the reveal policy and the content route cover what the queue routes return, and
nothing else. The local console has no login: any client can pass another reviewer's name,
and a client that calls any of the general routes can still see grades. For example:

- `GET /api/traces/{trace_id}` returns a whole trace with its verdict, step rewards, step
  verifier output and labels, the most direct way to see a trace item's grade, and
  `GET /api/traces` lists traces with their verdicts.
- `GET /api/labels` and `GET /api/labels/current` list labels, the judge's included, and
  `GET /api/calibration/agreement` compares the judge's verdicts with the experts'.
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

## Progress

`progress(queue_id)` returns `items`, `complete` (every review they need), `in_progress`
(a review or a reservation, but not complete), `reviews` submitted, `reviews_needed`
(items times overlap), `reserved` (reservations held now), `blocked` (items that can no
longer get every review they need because too many of the queue's listed reviewers skipped
them; adding reviewers unblocks them), and per reviewer the count of turns `submitted`,
`reserved`, `skipped`, `released` and `expired`.
`items(queue_id, offset, limit)` lists the items in order with their `reviews`, `needed`
and `reserved` counts, and the total; `limit=None` (the default) returns every one.

## REST

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
