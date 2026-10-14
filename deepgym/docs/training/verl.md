# verl

This page covers training with verl on DeepGym environments: the reward function verl
loads from the package, the agent loops that keep failed runs out of training, multi-turn
agent RL on live sandboxes, and DAPO recipes.

## Reward function

verl loads DeepGym's reward function from the package. Each dataset row names its
environment, so one run can mix environments:

```python
import pandas as pd
from deepgym.integrations.verl import verl_rows

pd.DataFrame(verl_rows(['coin_change', 'two_sum']) * 64).to_parquet('train.parquet')
```

```yaml
reward:
  custom_reward_function:
    path: pkg://deepgym.integrations.verl
    name: compute_score
    reward_kwargs: {mode: daytona}   # or auto, local, modal, e2b, docker, apple
```

- `verl_rows` writes verl's standard fields (`data_source`, `prompt`, `ability`,
  `reward_model`, `extra_info`); repeat rows to train on an environment more often.
- `compute_score` runs the last fenced code block of each response (end-of-turn tokens
  such as `<|im_end|>` are dropped first) and logs `score`, `passed` and `error`.
- For one fixed environment in your own module, use `make_verl_compute_score(env)`. For a
  custom training loop, `make_verl_reward_fn(env)` scores a batch and returns `None` for
  a run without a verdict.

## Failed runs

A verl reward function can't mark a sample as unscored, so when the sandbox or verifier
fails, DeepGym's raises `NoVerdictError` instead of returning a 0. verl's v1 trainer
(`trainer.use_v1`, the default since verl 0.9) scores inside the sample's agent-loop
session, so the raise drops the sample: verl records its prompt group as failed and
computes the group's GRPO mean and std from the other samples (the async trainers replace
the whole group). verl's legacy trainer can't mask a sample a reward function scored, so
there the raise stops training, and the error says so.

With the legacy trainer, let DeepGym's single-turn agent loop score the rows (the
generated DAPO config already does), so the sample is masked instead:

```yaml
# actor_rollout_ref.rollout.agent.agent_loop_config_path
- name: deepgym_single_turn
  _target_: deepgym.integrations.verl_mask.DeepGymSingleTurnAgentLoop
  mode: daytona   # used without reward.custom_reward_function: auto, local, modal, ...
```

Then set `actor_rollout_ref.rollout.agent.default_agent_loop=deepgym_single_turn`. The
rows are the same `verl_rows`.

- The loop calls the reward function above (`reward.custom_reward_function`, as verl
  would; DeepGym's `compute_score` if none is set), and verl skips its own reward step
  because the score is already set.
- Under the legacy trainer the loop zeroes a failed sample's response mask and logs
  `error` 1.0. That sample gets no gradient and adds no tokens to the loss, but its 0
  still counts toward its group's mean and std. Under v1 the loop drops the sample like
  the reward function does.
- A patch that is malformed, doesn't apply or touches test infrastructure is the model's
  fault, so it scores 0 and is trained on.

A policy that learns to trigger failures escapes their penalty either way, so keep
watching `error` and the failed samples; the loop logs a warning when more than 5
samples in a block of 100 are kept out. See [the reward contract](reward-contract.md).

## Multi-turn agents

For multi-turn agent RL, use DeepGym's agent loop and session tool. verl's stock
`ToolAgentLoop` creates and releases a tool around every call, so the `deepgym_agent` loop
keeps one sandbox per rollout, sets `reward_score` from the final verifier, and deletes
the sandbox. Screenshots come back as images.

```yaml
# agent_loop_config_path
- name: deepgym_agent
  _target_: deepgym.integrations.verl_tool.DeepGymAgentLoop

# multi_turn.tool_config_path
tools:
  - class_name: deepgym.integrations.verl_tool.DeepGymSessionTool
    config: {type: native, environment: signup_form, mode: daytona}
```

- Dataset rows set `agent_name: deepgym_agent`.
- The first tool call starts the episode and returns the starting screen, so prompt the
  model to begin with a screenshot.
- Pick the environment and variant per row with
  `extra_info.tools_kwargs.computer.create_kwargs: {environment: todo_list, variant: 3}`.
- verl cuts tool output to 256 characters by default; raise
  `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` for shell output.
- `mode: local` runs the model's commands on the training host; use a sandbox mode for
  real training.
- Written against verl 0.9.1. CI runs the loop against a stand-in for verl's; it was also
  run on Daytona sandboxes through verl 0.9.1's own `ToolAgentLoop._call_tool`.

Only failures that are ours for certain get no verdict and are kept out of training the
same way (`error` 1.0): a sandbox that fails to start, a provider, network or auth
failure, a provider lifetime that ran out, and a step or final verifier that couldn't
run or timed out before it read the agent's output (`deepgym.session.failure_result`);
an expiry only if the policy's own actions didn't take most of the time. Everything else
is scored:

- An invalid or disallowed action comes back to the model as text, and the rollout goes
  on.
- Any other failed action (a failed write, a sandbox the policy killed) comes back as an
  error and ends the episode with a 0, `error_type` `policy_sandbox_failure`, as does a
  final verifier that can't run in a sandbox the policy killed or hung.
- A verifier that times out after reading the agent's output scores 0 with
  `solution_timeout`.
- A rollout where the model never calls the tool scores 0.

This loop also warns when more than 5 samples in a block of 100 are kept out.

## DAPO

For verl-based DAPO recipes, generate a reward module and a minimal config snippet:

```python
from deepgym.integrations.dapo import (
    generate_dapo_reward_module,
    generate_dapo_verl_config,
)

reward_module = generate_dapo_reward_module('coin_change')
config_yaml = generate_dapo_verl_config(
    train_files='data/train.parquet',
    reward_module_path='reward_module.py',
)
```

The config scores through the `deepgym_single_turn` agent loop above, listed in a file
that ships with deepgym, so a sample without a verdict is dropped or masked rather than
trained on as a 0. verl's reward manager, and with it DAPO's overlong penalty, doesn't run
on samples the loop scored. For DAPO with TRL, see [TRL](trl.md#dapo).
