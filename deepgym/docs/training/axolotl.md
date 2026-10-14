# Axolotl

This page covers GRPO training with Axolotl on DeepGym environments, from Python or from
a generated config with no code, plus the process reward model config.

Axolotl trains GRPO with TRL, so `make_axolotl_reward_fn(env)` is the
[TRL reward function](trl.md): it runs the last fenced code block of each completion,
plain or conversational, and returns `None` for a completion that got no verdict.

## A config with no code

Save `verl_rows` as JSON lines and point `trl.reward_funcs` at
`deepgym.integrations.axolotl.row_reward_fn`, which scores each row with the environment
it names:

```python
from pathlib import Path

import pandas as pd
from deepgym.integrations.axolotl import generate_axolotl_config
from deepgym.integrations.verl import verl_rows

Path('data').mkdir(exist_ok=True)
rows = pd.DataFrame(verl_rows(['coin_change', 'two_sum']) * 64)
rows.to_json('data/train.jsonl', orient='records', lines=True)
config = generate_axolotl_config(base_model='Qwen/Qwen2.5-Coder-7B-Instruct')
# write config to axolotl.yaml, then: axolotl train axolotl.yaml
```

- The environment of a row is `extra_info['environment']` (with optional `variant`), else
  its `data_source`.
- `row_reward_fn` scores in Daytona when `DAYTONA_API_KEY` is set, otherwise locally. If
  the key is set but `deepgym[daytona]` isn't installed, it scores locally with only a
  warning.
- The generated config sets `eval_batch_size: 8`, so each eval batch holds whole groups of
  TRL's 8 generations per prompt on any number of GPUs.
- `generate_axolotl_config` also takes `dataset_path`, `sequence_len`,
  `micro_batch_size`, `gradient_accumulation_steps`, `num_epochs` and `learning_rate`.

## Process reward models

`generate_axolotl_config(method='prm')` writes a process reward model config for a
`stepwise_supervised` dataset (prompt, completions, labels) that you provide. DeepGym
doesn't write that format: [`deepgym traces export --rows rl`](../traces.md#export) gives
real step rewards from multi-turn episodes as `step_rewards`, but turning them into
labels is up to you.
