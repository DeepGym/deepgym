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

Checking a judge against answers with known verdicts is planned as a separate feature.
