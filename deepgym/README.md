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
- SWE-bench Pro support for repo-level patch RL tasks
- Terminal-Bench 2.0 support for shell/terminal RL tasks
- CyberBench/CyberGym support for local, artifact-backed cyber patch tasks
- MixedEnvironment routing for multi-benchmark training in one reward function
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

### Drop into DAPO

```python
from deepgym.integrations.dapo import make_dapo_reward_fn

reward_fn = make_dapo_reward_fn(env)
scores = reward_fn(completions=['def solve(x): return x'])
```

For verl-style DAPO recipes, DeepGym also exposes thin helpers to generate a
reward module and a minimal config snippet:

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

### Train on repo patches with SWE-bench Pro

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='auto')
env = load_environment('swebench_pro')

result = dg.run(
    env,
    model_output='''```diff\n... unified diff ...\n```''',
    repo='owner/repo',
    base_commit='abc123',
    test_patch='diff --git ...',
    fail_to_pass=['tests/test_bug.py::test_fix'],
    pass_to_pass=['tests/test_smoke.py::test_smoke'],
)
print(result.score)
```

### Train on terminal tasks with Terminal-Bench 2.0

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='auto')
env = load_environment('terminal_bench_2')

result = dg.run(
    env,
    model_output='python solve.py --input data.txt > output.txt',
    task_id='regex-log',
)
print(result.score)
```

### Train a cyber model with CyberBench/CyberGym

DeepGym can turn CyberGym Hugging Face metadata and artifacts into
CyberBench-style RL tasks for defensive cyber training. Use it for local
patch-repair, log-triage, forensics, and synthetic CTF-style tasks where the
reward comes from a deterministic verifier, not from live targets.

Install the optional Hugging Face and Daytona dependencies:

```bash
pip install "deepgym[hf,daytona]"
```

Keep provider keys local and untracked:

```bash
cp .env.example .env
# edit .env with DAYTONA_API_KEY, ZAI_API_KEY, ZAI_API_BASE, ZAI_MODEL
```

Generate safe CyberBench seed specs from CyberGym metadata. Add `--use-zai` if
you want GLM/Z.ai to enrich the seed plans while preserving local-only safety
constraints:

```bash
python scripts/inspect_cybergym_hf.py --repo-id sunblaze-ucb/cybergym --limit 100
python scripts/generate_cyberbench_seeds.py --count 100 --use-zai
```

For RL reward diagnostics, prefer artifact-backed patch tasks over
metadata-only prompts. The runner downloads the vulnerable repository archive
and reference patch, applies a model-produced unified diff in a sandbox, then
scores application, touched-file overlap, changed-line similarity, minimality,
and safety scope.

```bash
# Local verifier smoke/diagnostics
python scripts/run_cybergym_artifact_eval.py --count 100 --mode local --max-parallel 12

# Daytona-isolated execution for untrusted model outputs
python scripts/run_cybergym_artifact_eval.py --count 20 --mode daytona --max-parallel 4

# Ask GLM for candidate patches when rate limits allow
python scripts/run_cybergym_artifact_eval.py --count 20 --answer-source glm --fallback-to-reference
```

Inside a trainer, load a CyberGym row, build a `CyberGymPatchEnvironment`, and
score model completions exactly like any other DeepGym environment:

```python
from deepgym import DeepGym
from deepgym.cybergym_artifacts import CyberGymPatchEnvironment, load_cybergym_rows

dg = DeepGym(mode='auto')  # Daytona when configured, local fallback otherwise
row = load_cybergym_rows(count=1)[0]
env = CyberGymPatchEnvironment.from_row(row)

candidate_patch = model.generate(env.task)  # return a unified diff
result = dg.run(env, model_output=candidate_patch)
reward = result.score
```

Use the generated `data/cyberbench/*.jsonl` files as curriculum inputs for
TRL/verl/OpenRLHF. Keep training and evaluation splits separate, use Daytona for
untrusted completions, and do not train on metadata-only scores; materialize
seed specs into artifact-backed environments with deterministic verifiers first.
See `docs/cyberbench.md` for the full workflow and current diagnostic numbers.

### Mix multiple benchmarks behind one reward function

```python
from deepgym import MixedEnvironment, load_environment
from deepgym.integrations.trl import make_trl_reward_fn

swe_env = load_environment('swebench_pro')
terminal_env = load_environment('terminal_bench_2')
humaneval_env = load_environment('coin_change')

mixed = MixedEnvironment([
    (swe_env, 0.6),
    (terminal_env, 0.2),
    (humaneval_env, 0.2),
])

reward_fn = make_trl_reward_fn(mixed)
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

### Benchmark-backed special environments

These names resolve directly through `load_environment()` and use custom execution paths:

- `swebench_pro`: repo clone -> checkout -> patch apply -> test run -> score by pass fraction
- `terminal_bench_2`: execute terminal commands in a task sandbox -> verify expected output/state

`MixedEnvironment` lets you combine these with built-in or imported coding environments while keeping the same reward-function surface.

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
pytest                    # 291 tests
ruff check src/           # lint
ruff format src/          # format
```

### Release

PyPI publishing is tag-driven in GitHub Actions.

```bash
git tag v0.3.0
git push origin v0.3.0
```

Pushing a normal branch commit runs CI only. Pushing a `v*` tag runs the publish job and uploads the package to PyPI.

### Daytona setup

```bash
git clone https://github.com/daytonaio/daytona
docker compose -f docker/docker-compose.yaml up -d
# Set DAYTONA_API_URL and DAYTONA_API_KEY
```

Or use Daytona cloud: get a key from [app.daytona.io](https://app.daytona.io).

## Built with Daytona

DeepGym is part of the [Daytona Startup Grid](https://www.daytona.io/startups).

We use Daytona to power fast, isolated execution for modern agent training
and evaluation workflows.

## License

MIT
