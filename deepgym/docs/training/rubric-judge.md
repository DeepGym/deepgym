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

The report keeps every run of every answer (`answers`: score, check verdicts and
`error_type` per run) and computes everything else from them:

| Field | What it is |
|---|---|
| `judge_id`, `prompt_version`, `rubric_sha256` | The judge as you named it, the prompt version, a SHA-256 of the task and the checks. |
| `judged_runs`, `no_verdict_runs`, `no_verdict_by_type` | Runs with and without a verdict, the latter per `error_type`. |
| `overall`, `blind`, `assisted` | Judge pass against expert pass over judged runs: `tp`, `fn`, `tn`, `fp`, `tpr`, `tnr`, `agreement`. |
| `self_consistency` | Answers whose judged runs all pass or all fail, out of those judged at least twice. |
| `check_consistency` | Per check, answers whose runs all gave the same verdict. |
| `check_agreement` | Per check an expert labeled, the judge's `MET` against the expert's. |

A run without a verdict is left out of every number above, never counted as a fail. A rate
with nothing to measure is `null`. `blind` and `assisted` are kept apart because an expert
who saw the judge's output may lean toward it. `report.meets(min_tpr, min_tnr)` checks the
overall rates and that no more than `max_no_verdict` of all runs (0.05 by default, the share
above which reward functions warn) got no verdict, and returns `passed` with one reason per
bar; a `null` rate fails.
`CalibrationReport.model_validate_json` reads a saved report back.
