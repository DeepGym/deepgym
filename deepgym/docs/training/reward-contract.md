# The reward contract

This page covers how DeepGym decides whether a run was judged, what each trainer
integration does with a run that wasn't, and how the totals DeepGym reports stay
conservative.

## Verdicts

Every `RunResult` either has a verdict or it doesn't:

```python
batch = dg.run_batch(env, outputs)
rewards = [r.score if r.has_verdict else None for r in batch.results]   # None: leave it out
```

A run has no verdict when its `error_type` is anything but a model fault, or the verifier
exited 2 (verifier error). Its score means nothing about the answer, so training on it
would punish the policy for an outage.

`dg.run` returns such a result when the verifier reports its own failure. It raises
instead (`SandboxError`, `TimeoutError`, `VerifierError`) when the provider fails, when
the sandbox dies or the time limit runs out before the answer is in play, or when the
verifier prints no result. `run_batch` and `eval` turn each raise into a result without a
verdict, so use them in training loops.

Some failures are the model's own doing, and they are verdicts: a real 0 to learn from.
They are listed in `deepgym.models.MODEL_FAULT_ERRORS`:

| `error_type` | Meaning |
|---|---|
| `invalid_patch` | The answer contains no unified diff. |
| `patch_apply_failed` | The diff doesn't apply. |
| `test_infra_tampering` | The diff touches tests or test infrastructure. |
| `solution_error` | The verifier crashed, or returned a result it can't report, after the solution's output came into play. |
| `solution_timeout` | The solution, or the verifier handling its output, ran past the time limit. |
| `policy_sandbox_failure` | The policy broke its own grading: code its patch added stopped the test run, the sandbox died after the solution started, or a session's agent broke its sandbox or a step. |

Everything else that carries an `error_type`, such as `sandbox_error`, `runtime_error`,
`verifier_error`, `timeout` or `invalid_score`, has no verdict. A NaN, infinite or
out-of-range score is never reward.

## Infrastructure or policy

A failure is dated by when the policy's output comes into play: a solution process
starts, the verifier opens the solution's file, or a session verifier opens a file in the
agent's work directory.

- Before that, a verifier crash, a timeout or a dead sandbox has no verdict.
- After it, the same failure came from handling the policy's output, so it is a judged 0:
  `solution_error`, `solution_timeout` or `policy_sandbox_failure`. A score the verifier
  printed before crashing is never kept.
- A provider or network failure has no verdict at any point. In a single run, so does a
  sandbox that dies on Daytona, or in an E2B or Modal call that raises, since those
  return nothing the judge printed.

In agent episodes, `deepgym.session.failure_result(exc, sandbox_id)` draws the line once the
agent has acted. Only a failure that is ours for certain has no verdict: what
`deepgym.exceptions.is_infrastructure_failure(exc)` calls infrastructure (a connection or
transport error, a provider call timing out, a provider HTTP 5xx or 429, the provider
refusing our credentials with 401 or 403, or the provider evicting the sandbox), a
verifier that couldn't run or timed out before it read the agent's output, or the
provider retiring the sandbox at the end of its lifetime (`SandboxExpiredError`), unless the
agent's own actions took at least half the session's time. Anything else, a
`SandboxDiedError` or a failed write included, is the agent's 0, and so is a failure the
rule can't read. Sessions, verl's
multi-turn agent loop and Harbor's agent phase use this rule; the OpenRLHF endpoint
grades once more what `is_infrastructure_failure` calls infrastructure.

## What each trainer receives

| Integration | A run without a verdict |
|---|---|
| TRL, Axolotl and DAPO reward functions, `RewardFunction` | `None`. TRL 1.6+ leaves a completion out of its group's baseline when every reward function returned `None` for it. |
| verl `compute_score` | Raises `NoVerdictError`. verl's v1 trainer drops the sample; the legacy trainer stops, so use the `deepgym_single_turn` agent loop there. |
| verl agent loops (`deepgym_single_turn`, `deepgym_agent`) | Dropped under verl's v1 trainer, masked under the legacy trainer, logged as `error` 1.0. |
| verl `make_verl_reward_fn` (custom loops) | `None`. |
| OpenRLHF endpoint | Graded once more if the provider or the network failed it; any sample still without a verdict gets the router's `infra_failure_score`. |
| lm-eval task | Left out of `deepgym_score` and `deepgym_pass`; `deepgym_verdict` is the share that got one. |
| `DeepGymEnv` | `info.has_verdict` is false; mask the sample. `step` raises where `dg.run` does; `step_batch` returns those runs without a verdict. |

Details are on each trainer's page: [verl](verl.md), [TRL](trl.md),
[OpenRLHF](openrlhf.md), [lm-eval](lm-eval.md) and [Axolotl](axolotl.md).

A policy that learns to trigger failures that get masked escapes their penalty. The verl
agent loops and the OpenRLHF endpoint log a warning when more than 5 samples in a block of
100 got no verdict (`deepgym.integrations.reward.MaskedRate`); the OpenRLHF endpoint counts
only the samples it gave `infra_failure_score`. A rate that climbs means a provider
outage, or a policy that found such a failure.

## Totals

`BatchResult` (from `run_batch`) and `EvalResult` (from `eval`) keep their headline numbers
conservative: `passed`, `failed`, `avg_score` and `pass_rate` count a run without a
verdict as a failed 0, so a run that breaks its verifier or sandbox can only lower them. A
run without a verdict never counts as passed, even if it reported a pass.

Beside them sit the judged-only numbers:

| Field | Meaning |
|---|---|
| `no_verdict_count` | Runs whose verifier or sandbox failed instead of judging them. |
| `judged_count` | Runs with a verdict. |
| `judged_pass_rate` | Share of the judged runs that passed (0.0 if none was judged). |
| `judged_mean_score` | Mean score of the judged runs (0.0 if none was judged). |

`EvalResult.env_names` names the environment of each result, in suite order.

The CLI prints both, for example `Pass Rate: 63.00% (judged 70.00%, 10.00% no verdict)`,
and shows a run without a verdict as `ERROR (<error_type>)` with its cause. The console
keeps no score for such a run and shows a "No verdict" count with the judged pass rate
and score beside the headline ones.

## Your own training loop

`RewardFunction` applies the same rules for any framework: it runs the last fenced code
block of each completion (text or a conversation) and returns `None` where there was no
verdict.

```python
from deepgym import RewardFunction

reward_fn = RewardFunction(env=env, max_parallel=64)
rewards = reward_fn(completions)                 # list[float | None]
shaped = reward_fn.shaped_rewards(completions)   # reward_components per completion
per_test = reward_fn.per_test_rewards(completions)  # {'test_0': 1.0, ..., 'overall': 0.5}
```

`AsyncRewardFunction` is the same with an `AsyncDeepGym` client.

For GRPO, compute each group's baseline from the judged samples only:

```python
batch = dg.run_batch(env, solutions, max_parallel=8)
judged = [r.score for r in batch.results if r.has_verdict]
if judged:
    mean = sum(judged) / len(judged)
    std = (sum((s - mean) ** 2 for s in judged) / len(judged)) ** 0.5
    advantages = [
        (r.score - mean) / (std + 1e-8) if r.has_verdict else None   # None: leave it out
        for r in batch.results
    ]
```
