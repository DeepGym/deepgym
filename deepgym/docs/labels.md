# Labels

This page covers labels in the [console](console.md) database: what a label records and
targets, verdicts, corrections, retractions and invalidations, and reading labels from
Python or over `/api/labels`. [Review queues](review-queues.md) store expert reviews as
labels.

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

## Verdicts

Experts and judges give verdicts the same way. The verdict on an answer is a bool named
`passed`, and the verdict on one rubric check a bool named by its check id (`c1`, `c2`,
...), `true` meaning MET. These names take only bools, and only on a whole target: a mark
on a step or a span of text takes another name, such as `mark`. Each source and author
holds one current verdict per target, name and `group_id`: to change it, supersede it.
Each repeat of a judge gets its own `group_id`.

`deepgym.console.calibration_labels.labeled_answers` turns experts' verdicts on runs into
`rubric_calibration.LabeledAnswer` items for judge calibration, with the run's solution as
the answer text; see [From stored labels](training/rubric-judge.md#from-stored-labels).

## Corrections, retractions and invalidations

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

## Reading labels

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
