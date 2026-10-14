# OpenRLHF

This page covers DeepGym's reward endpoint for OpenRLHF's remote reward model option:
setup, how requests are graded, and what a sample gets when DeepGym can't grade it.

## Setup

```python
from fastapi import FastAPI
from deepgym import DeepGym, load_environment
from deepgym.integrations.openrlhf import create_openrlhf_router

env = load_environment('coin_change')
dg = DeepGym(mode='docker')

app = FastAPI()
app.include_router(create_openrlhf_router(env, dg, infra_failure_score='nan'))
# Run with: uvicorn app:app --port 8000
# Train with: --remote_rm_url http://localhost:8000/reward/score --algo.dynamic_filtering_enable
```

The router serves `POST /reward/score`. It accepts the payload OpenRLHF's agent client
sends (`query`, `prompts`, `labels`; `outputs` also works), strips each prompt from its
query and runs the response's last fenced code block. A query that doesn't contain its
prompt is never scored as text. Requests are capped at 2,000,000 characters per text,
1,024 samples and 64,000,000 characters in all (422 above).

## How requests are graded

OpenRLHF sends one request per rollout, all at once. A request waits for its runs without
holding a thread, so a slow verification doesn't hold up the others.

OpenRLHF can't skip a sample: it trains on whatever `rewards` holds, and a failed request
would stop its training step, so the endpoint never fails a well-formed request over a
run. A malformed or oversized body gets 422, and `prompts` is required even when the
client sends `outputs`.

- A failure the model caused is judged and scores 0, and so does any run without a
  verdict that the model's output may have caused (a verifier failure, a timeout); those
  are never run again.
- A run the sandbox provider or the network failed
  (`deepgym.exceptions.is_infrastructure_failure`) is graded once more after a short
  pause.
- Every run, first or second, takes one of the router's `max_parallel` slots (32 by
  default), and no request waits past 150 s (OpenRLHF gives up at 180 s). A first run
  still going then scores 0, since the model's code may be what slows it.

## Samples DeepGym can't grade

If the second grading fails too or doesn't finish in time, the first one never got a slot
in time, or a query doesn't contain its prompt, the sample gets the router's
`infra_failure_score`, never a score from a run that didn't finish. It has no default:
creating the router without it raises an error, so the choice is made at setup, where no
policy can reach it.

- `'nan'` (recommended): NaN. With `--algo.dynamic_filtering_enable`, OpenRLHF drops the
  sample's whole prompt group (the group's mean score is NaN, which is outside any range).
  Without dynamic filtering it trains on the NaN and the run breaks, so use it only with
  dynamic filtering on.
- `'zero'`: 0.0. OpenRLHF trains on DeepGym's failure as a failed answer, an accepted
  leak.

Either way the sample is logged as a warning with its error type, and counts toward a
warning logged when more than 5 samples in a block of 100 get `infra_failure_score`. Runs
without a verdict that score 0 (above) aren't counted. See
[the reward contract](reward-contract.md).
