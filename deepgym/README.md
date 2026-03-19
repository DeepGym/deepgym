# DeepGym

RL training environments with verifiable rewards for coding agents.

You give it model-generated code. It runs the code in a sandbox, checks it against
a verifier, and hands back a score. That score plugs straight into TRL, verl,
OpenRLHF, or whatever you're using for GRPO/DAPO/PPO.

## Quick Start

```bash
pip install deepgym
```

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='local')
env = load_environment('coin_change')

solution = '''
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
'''

result = dg.run(env, model_output=solution)
print(f'Score: {result.score}, Passed: {result.passed}')
```

## How it works

```
prompt --> model --> DeepGym sandbox --> verifier --> score --> training loop
                        |                  |
                     Daytona /          JSON protocol:
                     local subprocess   score, passed, reward_components
```

Model writes code. DeepGym runs it. Verifier checks it. You get back a score
and per-test-case breakdown showing exactly which tests passed and which didn't.

## What's in the box

- 24 built-in coding environments (ship with pip install)
- 2,350+ importable benchmarks (HumanEval, MBPP, BigCodeBench, EvalPlus)
- Per-test-case reward traces (not just pass/fail -- you see which tests broke)
- Deterministic seeding (same input, same score, every time)
- Three runtime modes: local subprocess, self-hosted Daytona, cloud Daytona
- Drop-in reward functions for TRL, verl, OpenRLHF
- lm-evaluation-harness task adapter (evaluate with `lm_eval --tasks deepgym_*`)
- HuggingFace Hub integration (share environments as HF datasets)
- Batch scoring for GRPO (score N completions in parallel)
- Gymnasium-style API if you prefer reset/step/state

## Usage

### Score a single solution

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='local')
env = load_environment('two_sum')

result = dg.run(env, model_output='def two_sum(nums, target): ...')
print(result.score, result.passed, result.reward_components)
```

### Batch scoring for GRPO

```python
solutions = [model.generate(prompt) for _ in range(8)]
batch = dg.run_batch(env, solutions, max_parallel=8)

scores = [r.score for r in batch.results]
# GRPO advantage: (r - mean) / std
mean = sum(scores) / len(scores)
std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
advantages = [(s - mean) / (std + 1e-8) for s in scores]
```

### Drop into TRL

```python
from deepgym.integrations.trl import make_trl_reward_fn
from trl import GRPOTrainer

reward_fn = make_trl_reward_fn(env)
trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn])
trainer.train()
```

### Drop into verl

```python
from deepgym.integrations.verl import make_verl_compute_score

compute_score = make_verl_compute_score(env)
# In verl config: custom_reward_function.path = "your_reward_module.py"
```

### Drop into OpenRLHF

```python
from fastapi import FastAPI
from deepgym.integrations.openrlhf import create_openrlhf_router

app = FastAPI()
app.include_router(create_openrlhf_router(env, dg))
# Run with: uvicorn app:app --port 8000
```

### Use with lm-evaluation-harness

```python
from deepgym.integrations.lm_eval import register_deepgym_tasks
register_deepgym_tasks()  # registers deepgym_* tasks

# lm_eval --model hf --model_args pretrained=Qwen/Qwen2-0.5B-Instruct \
#         --tasks deepgym_coin_change,deepgym_two_sum
```

### Share environments on HuggingFace Hub

```python
from deepgym.integrations.hf import push_environment_to_hub, load_environment_from_hub

# Push to HF Hub
push_environment_to_hub(env, repo_id='your-org/deepgym-coin-change', env_name='coin_change')

# Load from anywhere
env = load_environment_from_hub('your-org/deepgym-coin-change')
```

### Write your own verifier

```python
from deepgym import DeepGym, Environment

dg = DeepGym(mode='local')
env = Environment(
    task='Write a function `add(a, b)` that returns the sum of two numbers.',
    verifier_code=(
        'import importlib.util, sys\n'
        'spec = importlib.util.spec_from_file_location("sol", solution_path)\n'
        'mod = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(mod)\n'
        'return 1.0 if hasattr(mod, "add") and mod.add(2, 3) == 5 else 0.0\n'
    ),
)

result = dg.run(env, model_output='def add(a, b):\n    return a + b\n')
```

The `verifier_code` string becomes the body of a function that gets
`(solution_path, test_cases_path=None)`. Return a float, bool, or dict.
The wrapper handles the rest.

## Environments

### Built-in (24, ship with pip install)

**Coding (20):**
- Array/String: reverse_string, palindrome_check, anagram_check, max_subarray, rotate_array, remove_duplicates, valid_parentheses
- Hash Map: group_anagrams, longest_consecutive, top_k_frequent
- DP: climbing_stairs, coin_change, longest_common_subsequence, house_robber
- Graph/Tree: binary_search, merge_intervals, level_order_traversal
- Practical: fizzbuzz, roman_to_integer, matrix_spiral

**Computer-use (2):** file_organizer, cli_task

**Tool-use (2):** api_request, data_pipeline

Load by name:

```python
from deepgym import load_environment
env = load_environment('coin_change')
```

### Importable benchmarks

Run the import scripts to pull in standard benchmarks:

```bash
python scripts/import_humaneval.py      # 164 problems
python scripts/import_evalplus.py       # HumanEval+ (80x more tests) + MBPP+
python scripts/import_mbpp.py           # 500 problems
python scripts/import_bigcodebench.py   # 1,140 problems
```

After import, they're available through `load_environment()`.

## Verifier protocol

Verifiers output JSON to stdout:

```json
{
  "schema_version": "1.0",
  "score": 0.85,
  "passed": true,
  "details": "12/14 tests passed",
  "cases": [
    {"id": "test_0", "passed": true, "score": 1.0, "input_summary": "coins=[1,2,5] amount=11"},
    {"id": "test_1", "passed": false, "score": 0.0, "error": "expected 3, got -1"}
  ],
  "seed": 42
}
```

The `cases` field is the interesting part -- it tells you exactly which tests
passed and failed, so your training loop gets a denser signal than just 0 or 1.

Simple verifiers that return a float or bool get auto-wrapped to this format.

## Architecture

```
Training Framework (verl / OpenRLHF / TRL)
    |
    v
DeepGym (environments + verifiers + scoring)
    |
    v
Daytona sandbox / local subprocess
```

Three modes: **local** (subprocess, no deps, no isolation), **daytona**
(real container isolation), **auto** (tries Daytona, falls back to local).

Local mode is fine for development. For anything shared or untrusted, use Daytona.

## CLI

```bash
deepgym run --task task.md --verifier verifier.py --solution solution.py

# Dev mode
DEEPGYM_NO_AUTH=true deepgym serve --host 127.0.0.1 --port 8000 --allow-local-exec

# Production
DEEPGYM_API_KEY=your-key DAYTONA_API_KEY=your-key deepgym serve --port 8000
```

The server won't start without auth configured. Set `DEEPGYM_API_KEY` for
production or `DEEPGYM_NO_AUTH=true` for local development.
`--allow-local-exec` is required when running without Daytona.

## Development

```bash
pip install -e ".[dev]"   # install with test deps
pytest                    # 227 tests
ruff check src/           # lint
ruff format src/          # format
```

### Daytona setup

```bash
git clone https://github.com/daytonaio/daytona
docker compose -f docker/docker-compose.yaml up -d
# Set DAYTONA_API_URL and DAYTONA_API_KEY
```

Or use Daytona cloud: get a key from [app.daytona.io](https://app.daytona.io).

## License

MIT
