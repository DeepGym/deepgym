<p align="center">
  <h1 align="center">DeepGym</h1>
  <p align="center">Reward signals for RL code training. Sandbox it, verify it, score it.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/deepgym/"><img src="https://img.shields.io/pypi/v/deepgym" alt="PyPI"></a>
  <a href="https://pypi.org/project/deepgym/"><img src="https://img.shields.io/pypi/pyversions/deepgym" alt="Python"></a>
  <a href="https://github.com/DeepGym/deepgym/blob/main/LICENSE"><img src="https://img.shields.io/github/license/DeepGym/deepgym" alt="License"></a>
  <a href="https://github.com/DeepGym/deepgym/wiki"><img src="https://img.shields.io/badge/docs-wiki-blue" alt="Wiki"></a>
</p>

---

Your model writes code. DeepGym runs it in an isolated sandbox, executes tests against it, and returns a structured reward signal -- per-test-case scores, shaped reward components, execution metrics -- that plugs straight into TRL, verl, OpenRLHF, or your own GRPO/DAPO/PPO loop.

DeepSeek-R1 [deliberately avoided neural reward models](https://arxiv.org/abs/2501.12948) for code because they're susceptible to reward hacking at scale. DAPO, QwQ-32B, and Open-R1 followed the same path: rule-based, execution-verified rewards. That's what DeepGym provides -- deterministic, execution-based scoring with per-test granularity, running in sandboxed containers so untrusted model outputs can't touch your infrastructure.

```
                          reward signal
               +------------------------------------+
               |                                    |
               v                                    |
           +-------+     +----------+     +--------------------+
           | Model | --> | DeepGym  | --> |      Sandbox       |
           +-------+     +----------+     | (Daytona / local)  |
               ^              |           +--------------------+
               |              |                    |
               |              v                    v
               |         +-----------+       +----------+
               |         |  RunResult |<-----| Verifier |
               |         +-----------+       +----------+
               |           |                       |
               |           | score: 0.85           | JSON stdout
               |           | passed: false         | per-test cases
               |           | cases: [...]          | reward components
               |           v
           +-------------------+
           |   Training Loop   |
           | (TRL/verl/ORLHF)  |
           +-------------------+
```

## Install

```bash
pip install deepgym
```

<details>
<summary>More install options</summary>

```bash
# With Daytona sandbox support
pip install deepgym[daytona]

# With HuggingFace Hub integration
pip install deepgym[hf]

# With lm-evaluation-harness
pip install deepgym[lm-eval]

# Everything (dev + daytona + hf + lm-eval)
pip install deepgym[all]

# From source
git clone https://github.com/DeepGym/deepgym.git
cd deepgym
pip install -e ".[all]"
```

</details>

## Quick Start

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
print(result.score)    # 1.0
print(result.passed)   # True
print(result.cases)    # per-test breakdown: which tests passed, which failed
```

## How it works

```
  Model            DeepGym             Sandbox              Verifier
    |                 |                   |                     |
    |  solution code  |                   |                     |
    |---------------->|                   |                     |
    |                 |  create sandbox   |                     |
    |                 |------------------>|                     |
    |                 |  upload files     |                     |
    |                 |------------------>|                     |
    |                 |                   |  python verifier.py |
    |                 |                   |-------------------->|
    |                 |                   |                     | run tests
    |                 |                   |                     | (seeded)
    |                 |                   |  JSON stdout        |
    |                 |                   |<--------------------|
    |                 |  stdout + stderr  |                     |
    |                 |<------------------|                     |
    |                 |  parse JSON       |                     |
    |  RunResult      |                   |                     |
    |<----------------|                   |                     |
    |                 |                   |                     |
```

The verifier returns structured JSON: a 0.0-1.0 score, pass/fail, per-test-case breakdown, and optional shaped reward components (correctness, efficiency, style -- whatever you define). The per-test granularity is what makes this useful for training. Binary pass/fail is a sparse signal. Knowing that 12 out of 14 tests passed, and specifically which two failed, gives the optimizer something to work with -- this is the same approach used by [CodePRM](https://aclanthology.org/2025.findings-acl.428/), [PRIME](https://github.com/PRIME-RL/PRIME), and [Posterior-GRPO](https://arxiv.org/html/2508.05170v1), but without needing a separate process reward model.

## Why execution-based rewards

The field has largely converged here. [A Practitioner's Guide to Multi-Turn Agentic RL](https://arxiv.org/abs/2510.01132) found execution-based unit tests hit 22% success on SWE-Gym vs 4.2% for sparse binary and 7-9% for model-based judges (including GPT-4.1). DeepSeek-R1, DAPO, and QwQ-32B all use rule-based execution rewards rather than neural reward models.

The catch is infrastructure. You need sandboxed execution (you can't run untrusted model output on your training nodes), deterministic scoring (GRPO computes advantages across completions -- non-determinism breaks this), and structured output (binary pass/fail is too sparse for GRPO/DAPO to learn from). DeepGym handles all three.

## What you get

- **Execution-based verification** -- the approach DeepSeek-R1, DAPO, and QwQ-32B converged on, not neural reward models
- **Per-test reward signals** -- test-case-level scores like [CodePRM](https://aclanthology.org/2025.findings-acl.428/) and [PRIME](https://github.com/PRIME-RL/PRIME) provide, without training a separate PRM
- **Shaped reward components** -- `reward_components` dict for multi-signal composition (correctness + efficiency + style), similar to [Posterior-GRPO](https://arxiv.org/html/2508.05170v1)'s gated reward approach
- **Deterministic seeded scoring** -- same solution, same score, every time. GRPO and DAPO both require this
- **Sandboxed execution via Daytona** -- container isolation for untrusted code, same pattern as [verl's Sandbox Fusion](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html) and [DeepSWE](https://www.together.ai/blog/deepswe)'s 512-container setup
- **Reward hack detection** -- 6 adversarial attack strategies. [Anthropic's Nov 2025 paper](https://arxiv.org/abs/2511.18397) showed reward hacking during RL causes emergent misalignment. Check your verifiers before you train
- **24 built-in environments** + 2,350+ importable benchmarks (HumanEval, MBPP, EvalPlus, BigCodeBench)
- **Drop-in integrations** -- Axolotl, TRL `GRPOTrainer`, verl `compute_score`, OpenRLHF reward server, lm-eval tasks, HF Hub
- **PRM data generation** -- convert per-test results into [Axolotl-compatible](https://axolotlai.substack.com/p/process-reward-models) stepwise supervision datasets via `deepgym generate-prm`
- **Batch scoring** -- N completions in parallel with `run_batch()`, async client with semaphore-based concurrency
- **Gymnasium API** -- `reset()` / `step()` for multi-turn agent training, same interface as [Agent-R1](https://github.com/0russwest0/Agent-R1) and [VerlTool](https://arxiv.org/html/2509.01055v1)
- **REST API** -- FastAPI server with async jobs and API key auth

## Usage

### Score a single solution

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='local')
env = load_environment('two_sum')

result = dg.run(env, model_output='def two_sum(nums, target): ...')
print(result.score)              # 0.85
print(result.passed)             # False
print(result.reward_components)  # {'correctness': 0.85, 'efficiency': 0.9}
```

### Batch scoring for GRPO

Generate N completions, score them all, compute advantages:

```python
solutions = [model.generate(prompt) for _ in range(8)]
batch = dg.run_batch(env, solutions, max_parallel=8)

scores = [r.score for r in batch.results]
mean = sum(scores) / len(scores)
std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
advantages = [(s - mean) / (std + 1e-8) for s in scores]
```

### TRL

```python
from deepgym.integrations.trl import make_trl_reward_fn
from trl import GRPOTrainer

reward_fn = make_trl_reward_fn(env)
trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn])
trainer.train()
```

### verl

```python
from deepgym.integrations.verl import make_verl_compute_score

compute_score = make_verl_compute_score(env)
# In verl config: custom_reward_function.path = "your_reward_module.py"
```

### OpenRLHF

```python
from fastapi import FastAPI
from deepgym.integrations.openrlhf import create_openrlhf_router

app = FastAPI()
app.include_router(create_openrlhf_router(env, dg))
# uvicorn app:app --port 8000
# POST /reward/score {"prompts": [...], "outputs": [...]} -> {"rewards": [...]}
```

### lm-evaluation-harness

```bash
python -c "from deepgym.integrations.lm_eval import register_deepgym_tasks; register_deepgym_tasks()"

lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2-0.5B-Instruct \
  --tasks deepgym_coin_change,deepgym_two_sum
```

### HuggingFace Hub

```python
from deepgym.integrations.hf import push_environment_to_hub, load_environment_from_hub

push_environment_to_hub(env, repo_id='your-org/deepgym-coin-change', env_name='coin_change')

# load from anywhere
env = load_environment_from_hub('your-org/deepgym-coin-change')
```

### Axolotl

Axolotl wraps TRL's GRPOTrainer under the hood. DeepGym's reward functions work with it directly -- point your config at a DeepGym environment and completions get scored in sandboxes during training. No glue code.

```python
from deepgym.integrations.axolotl import make_axolotl_reward_fn
from deepgym import load_environment

env = load_environment('coin_change')
reward_fn = make_axolotl_reward_fn(env)

# Use in your Axolotl training script with GRPOTrainer
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model='Qwen/Qwen3.5-Coder-7B',
    reward_funcs=[reward_fn],
    train_dataset=dataset,
)
trainer.train()
```

Multi-signal training -- use separate reward functions for each component:

```python
from deepgym.integrations.axolotl import make_axolotl_shaped_reward_fn

correctness_fn = make_axolotl_shaped_reward_fn(env, component='correctness')
efficiency_fn = make_axolotl_shaped_reward_fn(env, component='efficiency')

trainer = GRPOTrainer(
    model='Qwen/Qwen3.5-Coder-7B',
    reward_funcs=[correctness_fn, efficiency_fn],
    train_dataset=dataset,
)
```

Generate an Axolotl YAML config:

```python
from deepgym.integrations.axolotl import generate_axolotl_config

config = generate_axolotl_config(
    base_model='Qwen/Qwen3.5-Coder-7B',
    method='grpo',
    dataset_path='data/train.jsonl',
)

with open('axolotl_grpo.yaml', 'w') as f:
    f.write(config)
# axolotl train axolotl_grpo.yaml
```

### PRM data generation

Process Reward Models score each reasoning step individually rather than giving one pass/fail for the whole solution. DeepGym already returns per-test-case verdicts, so the mapping is mechanical: each test case is a step, each pass/fail is a label. You get PRM training data from any DeepGym environment without hand-labeling.

```python
from deepgym.integrations.axolotl import generate_prm_dataset, write_prm_dataset
from deepgym import DeepGym, load_environment
from pathlib import Path

dg = DeepGym(mode='local')
env = load_environment('coin_change')

# Score solutions and extract per-test step labels
solutions = [open(f).read() for f in sorted(Path('solutions/').glob('*.py'))]
records = generate_prm_dataset(env, solutions, dg=dg)
write_prm_dataset(records, Path('prm_data.jsonl'))
```

Each record follows Axolotl's `stepwise_supervised` format:

```json
{
  "prompt": "Given coins [1,2,5] and amount 11, find minimum coins needed.",
  "completions": ["Input: [1,2,5], 11 | Expected: 3 | Got: 3", "Input: [], 0 | Expected: 0 | Got: -1"],
  "labels": [true, false]
}
```

Train a PRM with Axolotl on the generated data:

```yaml
# prm_config.yaml
base_model: Qwen/Qwen3.5-Coder-7B
model_type: AutoModelForTokenClassification
num_labels: 2
process_reward_model: true

datasets:
  - path: prm_data.jsonl
    type: stepwise_supervised
    step_separator: "\n\n"
    split: train
```

```bash
axolotl train prm_config.yaml
```

Or do it all from the CLI:

```bash
# Generate PRM dataset + Axolotl config in one shot
deepgym generate-prm \
  --env coin_change \
  --solutions-dir ./solutions/ \
  -o prm_data.jsonl \
  --axolotl-config prm_config.yaml \
  --base-model Qwen/Qwen3.5-Coder-7B

# Train
axolotl train prm_config.yaml
```

DeepGym runs code in sandboxes and reports which tests passed. Axolotl trains on that signal. The PRM piece goes a step further: instead of "this solution scored 0.85," the model sees "tests 1 through 10 passed, test 11 failed on the empty-list edge case." That's the same per-step granularity that [CodePRM](https://aclanthology.org/2025.findings-acl.428/) and [PRIME](https://github.com/PRIME-RL/PRIME) get from trained reward models -- but here it comes from execution directly, no reward model needed.

## Advanced Examples

### Custom verifiers

Write your own verifier inline. The string becomes the body of a function that gets `(solution_path, test_cases_path=None)`. Return a float, bool, or dict -- the wrapper normalizes it to JSON.

```python
from deepgym import DeepGym, Environment

dg = DeepGym(mode='local')
env = Environment(
    task='Write a function `add(a, b)` that returns the sum of two numbers.',
    verifier_code=(
        'import importlib.util\n'
        'spec = importlib.util.spec_from_file_location("sol", solution_path)\n'
        'mod = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(mod)\n'
        'cases = [(2, 3, 5), (0, 0, 0), (-1, 1, 0), (100, 200, 300)]\n'
        'passed = sum(1 for a, b, exp in cases if mod.add(a, b) == exp)\n'
        'return passed / len(cases)\n'
    ),
)

result = dg.run(env, model_output='def add(a, b):\n    return a + b\n')
# score: 1.0, passed: True
```

### Per-test reward shaping

Instead of just a single number, you get scores for each individual test case. Useful for denser training signals.

```python
result = dg.run(env, model_output=solution)

for case in result.cases:
    print(f"{case.id}: {'PASS' if case.passed else 'FAIL'} "
          f"(input: {case.input_summary}, expected: {case.expected_summary})")

# or through the reward function
from deepgym.integrations.reward import RewardFunction
reward_fn = RewardFunction(env, max_parallel=8)

per_test = reward_fn.per_test_rewards(solutions)
# [{'test_0': 1.0, 'test_1': 0.0, 'test_2': 1.0, 'overall': 0.67}, ...]

shaped = reward_fn.shaped_rewards(solutions)
# [{'correctness': 0.8, 'efficiency': 0.9}, ...]
```

### Async batch processing

When you need throughput, use the async client:

```python
import asyncio
from deepgym import AsyncDeepGym, load_environment

async def score_all():
    dg = AsyncDeepGym(mode='daytona')
    envs = ['coin_change', 'two_sum', 'climbing_stairs']

    tasks = [
        dg.run(load_environment(name), solutions[name])
        for name in envs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for name, result in zip(envs, results):
        if isinstance(result, Exception):
            print(f'{name}: ERROR')
        else:
            print(f'{name}: {result.score:.2f}')

asyncio.run(score_all())
```

### Gymnasium-style API

```python
from deepgym.gym import DeepGymEnv

gym_env = DeepGymEnv(environment=env, max_steps=3)
obs = gym_env.reset()
obs, reward, done, info = gym_env.step('def coin_change(coins, amount): ...')
```

### Audit verifiers for reward hacking

[Anthropic found](https://arxiv.org/abs/2511.18397) that models which learn to reward-hack during RL generalize to alignment faking and sabotage. [Lilian Weng's analysis](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) documents models rewriting unit tests, modifying reward-computing code, and gaming complexity metrics. Check your verifiers before training:

```python
from deepgym.adversarial import AdversarialTester

tester = AdversarialTester(dg, pass_threshold=0.5)
report = tester.test(env, strategies=['empty', 'hardcoded', 'trivial', 'overflow'])

print(f'Exploits found: {report.exploits_found}/{report.attacks_run}')
print(f'Robust: {report.is_robust}')
```

```bash
deepgym audit --verifier verifier.py --task "..." --strategies empty hardcoded trivial
```

Six attack strategies: empty/null code, hardcoded outputs, trivial placeholders, numeric overflow (NaN/Inf), pattern matching against test structure, and LLM-generated adversarial code. The auditor also analyzes verifier source for anti-patterns (static inputs, few test cases, no type validation) and assigns a risk score.

## Environments

### Built-in (24)

**Coding (20):**
| Difficulty | Environments |
|-----------|-------------|
| Easy | `fizzbuzz`, `reverse_string`, `palindrome_check`, `anagram_check`, `valid_parentheses`, `python_sorting`, `string_manipulation`, `two_sum` |
| Medium | `coin_change`, `climbing_stairs`, `house_robber`, `rotate_array`, `remove_duplicates`, `max_subarray`, `roman_to_integer`, `matrix_spiral`, `longest_consecutive`, `group_anagrams`, `top_k_frequent`, `merge_intervals`, `binary_search` |
| Hard | `longest_common_subsequence`, `level_order_traversal` |

**Computer-use (2):** `file_organizer`, `cli_task`

**Tool-use (2):** `api_request`, `data_pipeline`

### Importable benchmarks (2,350+)

```bash
python scripts/import_humaneval.py      # 164 problems
python scripts/import_mbpp.py           # 500 problems
python scripts/import_evalplus.py       # HumanEval+ (80x more tests) + MBPP+
python scripts/import_bigcodebench.py   # 1,140 problems
```

## Verifier protocol

Verifiers are standalone scripts that print JSON to stdout. No SDK, no imports from DeepGym, any language works. This is deliberate -- same philosophy as DeepSeek-R1's rule-based rewards. Keep the verifier simple and auditable.

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
  "reward_components": {"correctness": 0.85, "efficiency": 0.92},
  "seed": 42
}
```

Three levels of reward signal, depending on how much you want from your verifier:

1. **Binary** -- just `score` and `passed`. Equivalent to what most RLVR setups use.
2. **Per-test** -- add `cases` for test-case-level granularity. The model learns which tests it gets right, not just whether everything passed. This is what [PRIME](https://github.com/PRIME-RL/PRIME) and [CodePRM](https://aclanthology.org/2025.findings-acl.428/) provide through process reward models, but here it comes directly from execution.
3. **Multi-signal** -- add `reward_components` for shaped rewards. Compose correctness, efficiency, and style signals with custom weights, like [Posterior-GRPO](https://arxiv.org/html/2508.05170v1)'s format + rule + thinking reward composition.

Simple verifiers that return a float or bool get auto-wrapped to this format. Full spec in the [wiki](https://github.com/DeepGym/deepgym/wiki/Verifier-Protocol).

## Architecture

```
+------------------------------------------------------------------+
|                      TRAINING FRAMEWORKS                          |
| Axolotl | TRL (HuggingFace) | verl (ByteDance) | OpenRLHF | ... |
+------------------------------------------------------------------+
                               |
                      completions (code)
                               |
                               v
+------------------------------------------------------------------+
|                           DEEPGYM                                 |
|                                                                   |
|  +---------------------+    +----------------------------------+  |
|  | Python Client       |    | Environment Registry             |  |
|  |   DeepGym (sync)    |    |   24 built-in envs               |  |
|  |   AsyncDeepGym      |    |   HumanEval / MBPP / EvalPlus   |  |
|  +---------------------+    |   BigCodeBench / HF Hub          |  |
|            |                 +----------------------------------+  |
|            v                                                      |
|  +---------------------+    +----------------------------------+  |
|  | Verifier Engine      |    | Adversarial Tester              |  |
|  |   template wrapping  |    |   6 attack strategies           |  |
|  |   JSON protocol      |    |   reward hack detection         |  |
|  +---------------------+    +----------------------------------+  |
|            |                                                      |
+------------------------------------------------------------------+
             |
             v
+------------------------------------------------------------------+
|                         EXECUTION LAYER                           |
|                                                                   |
|   +-------------------------+  +------------------------------+   |
|   | LocalExecutor           |  | DaytonaSandbox               |   |
|   | (subprocess, no isolation) | (container, full isolation)  |   |
|   +-------------------------+  +------------------------------+   |
|                                                                   |
+------------------------------------------------------------------+
             |
             v
     RunResult { score, passed, cases, reward_components }
```

Three modes: **local** (subprocess, no deps, no isolation), **daytona** (container isolation), **auto** (tries Daytona, falls back to local). Use local for dev, Daytona for anything untrusted. The same Daytona infrastructure [runs 500 sandboxes in parallel for TRL GRPO training](https://www.daytona.io/docs/en/guides/reinforcement-learning/trl-grpo-training/) with sub-200ms cold starts.

## Where DeepGym fits

```
You're probably using one of these:        DeepGym is this layer:

+-------------------------------------+
| Training Framework                  |    Axolotl, TRL GRPOTrainer, verl,
| (policy optimization)               |    OpenRLHF, rLLM, or your own loop
+-------------------------------------+
                  |
                  | "score these N completions"
                  v
+-------------------------------------+
| Reward Infrastructure               |    <-- DeepGym
| (execution, verification, scoring)  |
+-------------------------------------+
                  |
                  | sandbox lifecycle
                  v
+-------------------------------------+
| Compute Isolation                   |    Daytona containers, local subprocess
| (run untrusted code safely)         |
+-------------------------------------+
```

Other projects in this space: verl uses [Sandbox Fusion](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html) for code verification. [DeepSWE](https://www.together.ai/blog/deepswe) runs 512 Docker containers via rLLM. [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) and [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) provide execution-based environments for SWE tasks. DeepGym wraps the same pattern -- sandboxed execution + structured reward output -- into a single `pip install` with drop-in reward functions for the major frameworks.

## CLI

```bash
# Run one environment
deepgym run --task task.md --verifier verifier.py --solution solution.py

# Batch eval
deepgym eval --suite medium --solutions-dir ./solutions/ --max-parallel 100

# Audit a verifier
deepgym audit --verifier verifier.py --task "..." --strategies empty hardcoded trivial

# API server (dev)
DEEPGYM_NO_AUTH=true deepgym serve --host 127.0.0.1 --port 8000 --allow-local-exec

# API server (production)
DEEPGYM_API_KEY=your-key DAYTONA_API_KEY=your-key deepgym serve --port 8000
```

## Daytona setup

<details>
<summary>Self-hosted (local Docker)</summary>

```bash
git clone https://github.com/daytonaio/daytona
cd daytona
docker compose -f docker/docker-compose.yaml up -d
# Dashboard: http://localhost:3000 (dev@daytona.io / password)
```

```bash
export DAYTONA_API_URL=http://localhost:3000
export DAYTONA_API_KEY=your-local-key
```

</details>

<details>
<summary>Daytona Cloud</summary>

1. Sign up at [app.daytona.io](https://app.daytona.io)
2. Grab your API key from the dashboard

```bash
export DAYTONA_API_KEY=your-cloud-key
```

</details>

## Development

```bash
pip install -e ".[all]"
pytest                      # 227 tests
ruff check src/             # lint
ruff format src/            # format
```

## Docs

Full docs on the [GitHub Wiki](https://github.com/DeepGym/deepgym/wiki):

- [Getting Started](https://github.com/DeepGym/deepgym/wiki/Getting-Started) -- install and first run
- [Core API Reference](https://github.com/DeepGym/deepgym/wiki/Core-API-Reference) -- classes, methods, models
- [Environments](https://github.com/DeepGym/deepgym/wiki/Environments) -- built-in + importable benchmarks
- [Verifier Protocol](https://github.com/DeepGym/deepgym/wiki/Verifier-Protocol) -- JSON spec, writing verifiers
- [Integrations](https://github.com/DeepGym/deepgym/wiki/Integrations) -- Axolotl, TRL, verl, OpenRLHF, lm-eval, HF Hub
- [Sandbox Modes](https://github.com/DeepGym/deepgym/wiki/Sandbox-Modes) -- local vs Daytona vs auto
- [Adversarial Testing](https://github.com/DeepGym/deepgym/wiki/Adversarial-Testing) -- reward hack detection
- [Advanced Usage](https://github.com/DeepGym/deepgym/wiki/Advanced-Usage) -- Gymnasium API, multi-turn, shaped rewards
- [Architecture](https://github.com/DeepGym/deepgym/wiki/Architecture) -- system design, module map

## License

MIT

---

<p align="center">
  <sub>Runs on <a href="https://www.daytona.io">Daytona</a> sandboxes</sub>
</p>
