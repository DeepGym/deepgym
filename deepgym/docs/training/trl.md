# TRL

This page covers DeepGym's reward functions for TRL's `GRPOTrainer`: setup, what runs,
how failed runs are kept out of training, and the DAPO and mixed-benchmark variants.

## Reward function

```python
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from deepgym import DeepGym, load_environment
from deepgym.integrations.trl import make_trl_reward_fn

env = load_environment('coin_change')
reward_fn = make_trl_reward_fn(env, DeepGym(mode='docker'))

trainer = GRPOTrainer(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    reward_funcs=[reward_fn],
    args=GRPOConfig(output_dir='grpo-coin-change'),
    train_dataset=Dataset.from_list([{'prompt': env.task}] * 64),
)
trainer.train()
```

- It runs the last fenced code block of each completion, or all of it if there is none.
  Completions can be plain text or conversations (lists of message dicts, including
  tool-calling turns); for a conversation, the last assistant turn is the answer.
- A completion whose sandbox or verifier failed gets `None`, not 0.0. When every reward
  function returns `None` for a completion, TRL 1.6+ leaves it out of its group's
  baseline and gives it no advantage, so an outage neither penalizes it nor shifts its
  siblings' advantages. Older TRL treats the `None` as 0.0. See
  [the reward contract](reward-contract.md).
- `max_parallel` (default 32) caps the sandbox runs at once.
- `component='correctness'` returns one reward component instead of the score (a result
  without it falls back to its score), for one reward function per component.
- `make_trl_async_reward_fn(env)` is the async version.

Without a client, the reward function uses `DeepGym(mode='auto')`: Daytona when
`DAYTONA_API_KEY` is set, otherwise local. If the key is set but `deepgym[daytona]` isn't
installed, it runs locally with only a warning. Pass a client to pick the provider, such
as `DeepGym(mode='daytona')`.

## DAPO

`make_dapo_reward_fn` is the TRL reward function under another name, and
`make_dapo_shaped_reward_fn` takes `component=`:

```python
from deepgym.integrations.dapo import make_dapo_reward_fn

reward_fn = make_dapo_reward_fn(env)
scores = reward_fn(completions=['def solve(x): return x'])
```

The last fenced code block runs, and a completion that got no verdict scores `None`, so
leave it out of the loss and of its group's statistics. For verl-based DAPO recipes, see
[verl](verl.md#dapo).

## Mixed benchmarks

`MixedEnvironment` puts several environments behind one reward function:

```python
from deepgym import MixedEnvironment, load_environment
from deepgym.integrations.trl import make_trl_reward_fn

mixed = MixedEnvironment(environments=[
    (load_environment('swebench_pro'), 0.6),
    (load_environment('terminal_bench_2'), 0.2),
    (load_environment('coin_change'), 0.2),
])
tasks = mixed.sample_batch(64)   # environments drawn by weight, for building a dataset

reward_fn = make_trl_reward_fn(mixed)
```

The weights only decide what `sample_batch` draws. When scoring, each completion goes to
the environment its own metadata names: `env_name` (or `environment_name`), `benchmark`
or `task_type`, or the benchmark's task fields such as `instance_id` or `task_id`. TRL
passes the dataset's columns to the reward function, so give each row an `env_name`
column. A completion that matches no environment raises `ValueError`.

See [importers](../environments/importers.md) for the benchmark environments.
