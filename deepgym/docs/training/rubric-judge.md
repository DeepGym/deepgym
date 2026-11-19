# Rubric-judged rewards

This page covers `deepgym.rubric`, a host-side reward for open-ended answers that have no
code to run: an LLM judge marks each check of a rubric `MET` or `UNMET`.

## A rubric

A rubric is a JSON list of checks, each `[requirement, weight]` or
`[requirement, weight, tag]`:

```json
[
  ["Recommends raising the basic plan price", 2, "must"],
  ["Quotes the churn figure from the brief", 1],
  ["Uses last year's price list", -1, "fatal"]
]
```

A `must` check needs a weight above 0. A `fatal` check describes a mistake and needs a
weight below 0. Tags are exactly `"must"` or `"fatal"`, lower case. `Rubric.load` takes
JSON text or the same list as Python data, and raises one `RubricError` that lists every
problem: an unknown tag, a weight that isn't a finite number, a check with the wrong
number of items, an empty or duplicate requirement, a misweighted `must` or `fatal`, or no
positive weight at all.

## Scoring

1. An answer that leaves a `must` check unmet or meets a `fatal` check scores 0.
2. Any other answer scores the weight of the checks it meets divided by the total positive
   weight, clamped to [0, 1].

Points earned elsewhere never lift a wrong answer above 0, and right answers are still
ranked against each other. `metrics` holds `verdicts`, `gates` (each check that zeroed the
score, with its tag), `point_share` (unclamped, kept even when a gate fired, never in the
reward) and `rubric_prompt_version`.

## Use it

```python
from pathlib import Path

from deepgym.rubric import Rubric, RubricReward, openai_judge

rubric = Rubric.load(Path('rubric.json').read_text())
judge = openai_judge('judge-model', base_url='https://judge.example/v1',
                     key_env='JUDGE_API_KEY', reasoning_effort='high')
reward_fn = RubricReward('Should we change the basic plan price?', rubric, judge)
rewards = reward_fn(completions)  # one float in [0, 1] per completion, or None
```

`anthropic_judge(model, base_url=..., key_env=..., **settings)` does the same for an
Anthropic-compatible endpoint (needs `deepgym[anthropic]`). Settings such as reasoning
effort or thinking go into the request as given. Any callable
`(system_prompt, user_message) -> reply_text` also works as a judge.

Bounds:

- `openai_judge` sends no output limit and has no read timeout. Connecting times out
  after 30 seconds.
- `anthropic_judge` has no timeout. The API requires `max_tokens`, so it sends 128,000,
  or the `max_tokens` setting if you give one, and lowers it when the API reports a
  smaller maximum for the model.

## Verdicts

The judge gets the fixed prompt `RUBRIC_PROMPT` and one user message: a JSON object with
`task`, `checks` and `answer`. Its reply must be a JSON object with exactly one `"MET"` or
`"UNMET"` per check id, bare or inside a single ```` ```json ```` fence with only
whitespace around it. Only a reply that finished normally counts: `finish_reason` `stop`,
or stop reason `end_turn` or `stop_sequence`.

When the judge refuses a request as too long for its context, the same request, with the
same settings (the same `max_tokens` too, never lowered), is sent once more with an empty
answer. That probe counts as accepted on any 2xx response, whatever the reply says.

| What happened | Result |
|---|---|
| The judge refused the input as too long, and accepted the probe | Judged 0 (`solution_error`): the answer caused it. |
| The judge refused the probe as too long too | No verdict (`judge_error`). |
| The probe failed any other way, or would have needed a smaller `max_tokens` | No verdict (`judge_error` or `provider_error`). |
| Any other reply, a refusal, a reply cut off or stopped for another reason | No verdict (`judge_error`). |
| A provider outage, a rate limit or refused credentials | No verdict (`provider_error`). |
| Any other bad request, such as an unsupported setting | No verdict (`judge_error`). |

A custom judge client gets the probe by raising `ContextOverflowError(message, accepts)`;
without `accepts`, an overflow gets no verdict. A run without a verdict is rewarded `None`,
as in [the reward contract](reward-contract.md).

## Calibrate the judge

Before training on a judge's reward, measure how often it agrees with expert labels on
answers that were not used to write the rubric:

```python
from deepgym.rubric_calibration import LabeledAnswer, calibrate

answers = [
    LabeledAnswer(id='a17', answer=text, passed=True, blind=True, checks={'c1': True}),
    ...
]
report = calibrate(reward_fn, answers, judge_id='judge-model, reasoning_effort=high',
                   built_from=['a01', 'a02'])
print(report.meets(min_tpr=0.9, min_tnr=0.9))
Path('calibration.json').write_text(report.to_json())
```

- Each answer is graded `runs` times (3 by default), with at most the reward's
  `max_parallel` judge calls at once. The judge passes an answer whose score is at least
  `threshold` (1.0 by default).
- `built_from` lists the ids of the answers used to write the rubric. If any of them is
  among the labeled answers, `calibrate` raises `ValueError` naming them before calling the
  judge. It also refuses duplicate ids, labels for checks the rubric doesn't have, and
  `runs` below 1.
- `blind` says whether the expert labeled the answer without seeing the judge's output.
  `checks` holds the expert's per-check verdicts, where given: `True` is the judge's `MET`,
  the answer meets the requirement as written (for a `fatal` check, it makes the mistake).
- `experts` optionally holds each expert's own label by author
  (`ExpertLabel(passed=..., blind=..., checks=...)`). An answer two or more experts labeled
  is in the human baselines below; `passed`, `checks` and `blind` are then the one label the
  judge is measured against, which you choose. `passed=None` is allowed only when as many
  experts pass the answer as fail it: the answer is then left out of `overall`, `blind`,
  `assisted` and the `min_kappa` bar, while its agreed `checks` still count in
  `check_agreement` and its experts' labels in the baselines. `tied_answers` counts them,
  and the TPR, TNR and kappa reasons of `meets` say how many were left out. An odd number
  of experts per answer avoids ties on `passed`.
- `calibrate` validates every answer again before any judge call, so an answer changed with
  `model_copy` (which skips validation) is refused. To change one, rebuild it:
  `LabeledAnswer.model_validate({**answer.model_dump(), 'passed': True})`.

The report keeps every run of every answer (`answers`: score, check verdicts and
`error_type` per run) and computes everything else from them:

| Field | What it is |
|---|---|
| `judge_id`, `prompt_version`, `rubric_sha256` | The judge as you named it, the prompt version, a SHA-256 of the task and the checks. |
| `judged_runs`, `no_verdict_runs`, `no_verdict_by_type` | Runs with and without a verdict, the latter per `error_type`. |
| `overall`, `blind`, `assisted` | Judge pass against expert pass over judged runs: `tp`, `fn`, `tn`, `fp`, `tpr`, `tnr`, `agreement`, `precision`, `f1`, `kappa`. |
| `self_consistency` | Answers whose judged runs all pass or all fail, out of those judged at least twice. |
| `check_consistency` | Per check, answers whose runs all gave the same verdict. |
| `check_agreement` | Per check an expert labeled, the judge's `MET` against the expert's, with the same numbers. |
| `check_summary` | `check_agreement` averaged over the checks, unweighted: `macro_f1` (the F1 of the `MET` call, not of `MET` and `UNMET` averaged) over `f1_checks` checks, and `mean_kappa` over `kappa_checks` checks, each leaving out checks where it is `null`. |
| `tied_answers` | Answers whose experts tie on `passed`, left out of `overall`, `blind`, `assisted` and `min_kappa`. |
| `blind_baseline`, `assisted_baseline` | The human baseline from blind labels, and apart from it the one from labels given after seeing the judge's output. `null` without an answer to use. |

A run without a verdict is left out of every number above, never counted as a fail. A rate
with nothing to measure is `null`. `blind` and `assisted` are kept apart because an expert
who saw the judge's output may lean toward it.

`kappa` is Cohen's kappa: agreement corrected for the agreement chance would give, each
side keeping its own pass rate. 1 is full agreement, 0 is what chance gives, and below 0 is
worse. It is `null` when nothing was compared, and when both sides made one and the same
call every time (all pass or all fail): chance then explains all the agreement. Kappa
depends on how often the experts pass: of 100 answers, judge and expert agreeing on 90 and
splitting the other 10 evenly gives 0.80 when each passes 50, and 0.44 when each passes 90.
Compare kappas on the same answers, as the baseline does. `f1` is the F1 of the pass (or
`MET`), `null` when neither side passed anything.

### Human baseline

The baseline answers whether the judge agrees with experts about as well as experts agree
with each other. It uses answers two or more experts labeled with the same blind flag and
the judge judged at least once. Each pair of experts is compared on the answers both
labeled (`experts[a][b]`), and the judge against each of the two on the judged runs of the
same answers (`judge[a][b]` against `a`, `judge[b][a]` against `b`). Every table is kept
with its counts.

`expert_kappa` and `judge_kappa` pool the tables of each side into one kappa
(`pooled_kappa`): the summed agreement less the summed agreement chance gives, over the
summed comparisons less that, each table keeping its own chance agreement. A table weighs
by its comparisons, so a pair of experts sharing one answer barely moves the number, and
one pair alone gives its own Cohen's kappa. This equals the pooled kappa of De Vries et al.
(2008) only when the tables are the same size, since that one weighs tables equally. On
the experts' side an answer counts once per pair of experts that labeled it. On the
judge's side each judged run counts once per expert per pair, so an answer with fewer
judged runs (some without a verdict) weighs less. `expert_agreement` and
`judge_agreement` are the shares of `expert_comparisons` and `judge_comparisons` that
agree. `deepgym.rubric_calibration.pair_agreement(rater, reference)` compares any two score
sources the same way, given each one's marks per target
(`{'r1': {'passed': True, 'c2': False}, ...}`).

### Bars

`report.meets(min_tpr, min_tnr)` checks the overall rates and that no more than
`max_no_verdict` of all runs (0.05 by default, the share above which reward functions warn)
got no verdict, and returns `passed` with one reason per bar; a `null` rate fails. Two more
bars apply only when given, and a `null` kappa or a missing baseline fails its bar:

- `min_kappa=` on `overall.kappa`: the judge against each answer's one expert pass mark
  (with stored labels, the majority; ties left out), blind and assisted labels alike.
- `max_kappa_gap=`: how far `blind_baseline.judge_kappa` may fall below
  `blind_baseline.expert_kappa`, within 1e-9 for float rounding. Its reason gives the
  comparison counts. It also fails when the baseline is uninformative: an `expert_kappa`
  that is `null` or at most 0 means the experts agree no better than chance, so there is
  nothing to come close to.
- `min_baseline_answers=`, with `max_kappa_gap`: the gap bar also fails when the blind
  baseline has fewer answers than this. Off by default; a handful of answers gives a kappa
  too noisy to compare.

The two measure different things: `min_kappa` compares the judge with one reference label
per answer, while `judge_kappa` compares it with each expert's own label, beside how well
the same experts agree. A majority label is less noisy than one expert's, so a judge can
reach one bar and miss the other.

`CalibrationReport.model_validate_json` reads a saved report back, including reports saved
before these numbers existed.

### From stored labels

Expert labels in the console's label store (see [the console](../console.md)) become
calibration input with `labeled_answers`:

```python
from deepgym.console.calibration_labels import labeled_answers
from deepgym.console.label_store import LabelStore
from deepgym.console.run_store import RunStore
from deepgym.rubric_calibration import rubric_sha256

db = Path.home() / '.deepgym' / 'console.db'
labeled = labeled_answers(LabelStore(db), RunStore(db), run_ids,
                          rubric_sha256=rubric_sha256(reward_fn.task, reward_fn.rubric))
for skip in labeled.skipped:
    print(skip.run_id, skip.reason, skip.detail)
report = calibrate(reward_fn, labeled.answers, judge_id='...',
                   built_from=rubric_run_ids)  # ids of the runs used to write the rubric
```

- Each run is one answer, its solution the answer text and its id the answer id. Only
  current human verdicts count: `passed` is the pass mark and check ids the per-check
  marks. Superseded and retracted labels are ignored, and judge and verifier labels are
  left out, since `calibrate` runs the judge itself.
- With `rubric_sha256`, verdicts recorded against another rubric hash are left out, and
  those recording no hash are kept. With or without it, `labeled.rubric_hashes` counts the
  verdicts behind the answers per rubric hash, and `labeled.no_rubric_hash` those with
  none; only verdicts in an expert label that made it into an answer count. More than one
  hash means the labels span rubric versions. Check ids are positions in the rubric
  (`c1` is the first check), so editing a rubric renumbers them: a `c2` labeled before
  the edit may name another requirement now.
- An expert's label is blind only when every one of their verdicts on the run is labeled
  blind; an unrecorded `blind` counts as not blind.
- The judge is measured against the experts' majority, on `passed` and on each check, blind
  only when every expert's is. On a tie, `passed` is `None` or the check is left out. Every
  expert's own label is in `experts`. To choose another way, rebuild the answers with
  `LabeledAnswer.model_validate`.
- `skipped` lists what was left out and why:
  - `excluded`: a current invalidation excludes the run (its comment is the detail).
  - `no_run`: no such run, so no answer text.
  - `rubric_mismatch`: an expert's verdicts recorded against another rubric hash.
  - `conflicting_author`: an expert holds both true and false for one verdict on the run,
    across groups. That verdict is left out of their label; if it is `passed`, the expert
    is left out of the run.
  - `no_passed`: either an expert gave check verdicts but no `passed` (none, or only ones
    left out above) and is left out of the run, or no expert is left with a `passed`
    (none was given, or each was left out above) and the run is left out.
  - `tie`: as many experts mark a verdict true as false; `passed` is `None` or the check
    is left out of `checks`.
- A run given twice in `run_ids` is read once.
- For the runs a label query matches, pass
  `dict.fromkeys(label.target.id for label in store.page(query, 0, None).labels)` with
  `target_kind='run'` in the query.
