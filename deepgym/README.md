# DeepGym

[![PyPI](https://img.shields.io/pypi/v/deepgym)](https://pypi.org/project/deepgym/)
[![CI](https://github.com/DeepGym/deepgym/actions/workflows/deepgym-ci.yml/badge.svg)](https://github.com/DeepGym/deepgym/actions/workflows/deepgym-ci.yml)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue)](https://github.com/DeepGym/deepgym/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/DeepGym/deepgym?style=social)](https://github.com/DeepGym/deepgym)

RL training environments with verifiable rewards for coding and computer-use agents.

DeepGym runs model output in a sandbox, checks it with a verifier and returns a
score. The score plugs into TRL, verl, OpenRLHF or your own GRPO/DAPO/PPO loop.

For agents, it keeps one sandbox alive for the whole episode (a shell or a full
desktop), applies actions step by step, then scores the state the agent left behind.

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

Optional extras: `deepgym[daytona]`, `deepgym[modal]` or `deepgym[e2b]` for those sandboxes, `deepgym[server]` for
`deepgym serve`, `deepgym[hf]` for the Hugging Face Hub, `deepgym[lm_eval]` for
lm-evaluation-harness, `deepgym[all]` for everything.

## What's in the box

- 31 built-in environments that ship with the package: coding, tool-use, cyber
  and 5 computer-use tasks
- Computer-use episodes on a real Daytona desktop: screenshots in, mouse and keyboard out
- Live multi-turn sessions from Python (`dg.session()`) or HTTP (`/v1/sessions`)
- Importers for HumanEval, MBPP, BigCodeBench and EvalPlus
- SWE-bench Pro (repo-level patches), Terminal-Bench 2.0 (shell tasks) and
  CyberGym (artifact-backed cyber patches)
- `MixedEnvironment` to train on several benchmarks behind one reward function
- Per-test-case results, not just pass/fail, so you see which tests broke
- Seeded verifiers: the same input gets the same score
- Batch scoring for GRPO (score N completions in parallel)
- Reward functions for TRL, verl, OpenRLHF and Axolotl, plus a verl multi-turn agent loop
- lm-evaluation-harness tasks (`lm_eval --tasks deepgym_*`) and Hugging Face Hub sharing
- Gymnasium-style `reset`/`step`/`state` API
- Runs locally or in sandboxes from Daytona, Modal, E2B, Docker or Apple container

## DeepGym and verl

verl trains the model. DeepGym gives it tasks and scores the answers. Use them
together: verl loads DeepGym's reward function with one config line (see [verl](#verl)).

| | verl | DeepGym |
|---|---|---|
| What it is | RL training framework | RL environments and verifiable rewards |
| Does the training | Yes: rollouts on vLLM/SGLang, updates on FSDP/Megatron, PPO, GRPO, DAPO and more | No, it plugs into verl, TRL, OpenRLHF or your own loop |
| Tasks | You bring the dataset and write the reward function | 31 built-in environments, HumanEval/MBPP/BigCodeBench/EvalPlus importers, SWE-bench Pro, Terminal-Bench 2.0 |
| Running model code | Your reward function, or a remote Sandbox Fusion service | Every run in a sandbox: Daytona, Modal, E2B, Docker, Apple container, or locally under sandbox-exec/bwrap |
| Reward hacking | Up to your reward function | The solution runs apart from the verifier with no network, so it can't read the tests or forge the score; `deepgym qa` checks that empty, stub and hack solutions fail before you train |
| Multi-turn agents | Agent loop and tool interface | One live sandbox per episode (shell or desktop), with a verl agent loop and session tool |
| Computer use | Not built in | Desktop tasks, scored on the state the agent leaves behind |
| Failed runs | A reward function can't drop a sample | A verl agent loop masks failed runs out of the loss |
| Managing it | Training logs (wandb, TensorBoard) | `deepgym console`: environments, runs, episode replay, sandboxes, QA reports |

## For teams and enterprises

DeepGym is free for personal, research and other noncommercial use. Any commercial
use, including inside a company or in a product you sell, needs a commercial license
from us. We can also help with:

- Custom environments and verifiers for your tasks
- Private or self-hosted deployment
- Setting up and running RL training

Email [contact@deepgym.io](mailto:contact@deepgym.io) or visit [deepgym.io](https://deepgym.io).

## How it works

```
prompt --> model --> DeepGym sandbox --> verifier --> score --> training loop
                        |                  |
                     Daytona / Modal /  JSON protocol:
                     E2B / Docker /     score, passed, cases
                     Apple / local
```

The verifier prints JSON with a score and a per-test breakdown
(see Output protocol below).

### Runtime modes

- `local`: a subprocess on your machine. Fine for development. Repo, terminal and
  cyber benchmark tasks refuse it (see below).
- `daytona`: a Daytona sandbox, cloud or self-hosted.
- `modal`: a Modal sandbox (`pip install 'deepgym[modal]'`, then `modal token set`).
- `e2b`: an E2B sandbox (`pip install 'deepgym[e2b]'`, set `E2B_API_KEY`).
- `docker`: a local Docker container with no network.
- `apple`: a throwaway Linux VM from Apple's container CLI on macOS.
- `auto` (default): Daytona when `DAYTONA_API_KEY` is set, local otherwise.

```python
dg = DeepGym(mode='modal')   # same for AsyncDeepGym and dg.session()
```

Every sandbox mode gives the same isolation, for single runs and shell sessions.
For anything shared or untrusted, use one of them instead of `local`.
Computer-use (desktop) environments need Daytona.

Each provider is one folder in `src/deepgym/sandboxes/` implementing a small
`Sandbox` protocol (run a command as root, upload a file, delete). Adding a
provider means adding a folder.

### Daytona setup

Cloud: get a key from [app.daytona.io](https://app.daytona.io) and set `DAYTONA_API_KEY`.

Self-hosted:

```bash
git clone https://github.com/daytonaio/daytona
cd daytona
docker compose -f docker/docker-compose.yaml up -d
# Set DAYTONA_API_URL and DAYTONA_API_KEY
```

### Docker and Apple container (local)

Each run or session gets a throwaway container (Docker) or Linux VM (Apple's
[container](https://github.com/apple/container) CLI on macOS). Same isolation as
the cloud sandboxes, no API key. For Docker, start the daemon. For Apple container:

```bash
brew install container
container system start
```

```python
dg = DeepGym(mode='apple')   # or AsyncDeepGym(mode='apple')
dg.run(env, solution)
```

- Image: `python:3.12-slim`, override with `DEEPGYM_CONTAINER_IMAGE` or per
  environment (below).
- Sessions run offline; servers inside the sandbox still work on localhost.
- Stop Apple's service with `container system stop`.

### Per-environment images

Set `image` on an environment, or in its `metadata.json`, to boot its sandbox from
an image with the task's dependencies already installed:

```python
env = Environment(task=..., verifier_code=..., image='python:3.11-slim')
```

- Docker, Apple container, Modal and Daytona pull it from a registry; E2B treats it
  as a template id. On Daytona a `snapshot` wins.
- The image must run as root and have python3, bash, useradd and util-linux
  (setpriv, unshare, nsenter, mount). Debian and Ubuntu images do; Alpine doesn't.
- Root in the sandbox runs the image's binaries, so only use images you trust. The
  HTTP API refuses caller-chosen images.

## Scoring and training

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

### TRL

```python
from deepgym.integrations.trl import make_trl_reward_fn
from trl import GRPOTrainer

reward_fn = make_trl_reward_fn(env)
trainer = GRPOTrainer(model=model, reward_funcs=[reward_fn])
trainer.train()
```

### verl

verl loads DeepGym's reward function from the package. Each dataset row names
its environment, so one run can mix environments:

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

It runs the last fenced code block of each response and logs `score`, `passed`
and `error` (1.0 when the sandbox or verifier failed and the score is a forced 0).
For one fixed environment in your own module, use `make_verl_compute_score(env)`.

A verl reward function or reward manager can't drop a sample, so a failed run
trains as a 0. To keep those samples out of the policy loss, let DeepGym's
single-turn agent loop score the rows instead. It zeroes the response mask of any
sample whose `error` is 1.0, so the sample gets no gradient and adds no tokens to
the loss, and verl skips its own reward step because the score is already set.

```yaml
# actor_rollout_ref.rollout.agent.agent_loop_config_path
- name: deepgym_single_turn
  _target_: deepgym.integrations.verl_mask.DeepGymSingleTurnAgentLoop
  mode: daytona   # or auto, local, modal, e2b, docker, apple
```

Then set `actor_rollout_ref.rollout.agent.default_agent_loop=deepgym_single_turn`.
The rows are the same `verl_rows`. A masked sample's 0 still counts toward its
GRPO group's mean and std, and a policy that learns to trigger failures escapes
their penalty, so keep watching `error`.

For multi-turn agent RL, use DeepGym's agent loop and session tool. verl's stock
`ToolAgentLoop` creates and releases a tool around every call, so the
`deepgym_agent` loop keeps one sandbox per rollout, sets `reward_score` from the
final verifier, and deletes the sandbox. Screenshots come back as images.

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
- The first tool call starts the episode and returns the starting screen, so
  prompt the model to begin with a screenshot.
- Pick the env and variant per row with
  `extra_info.tools_kwargs.computer.create_kwargs: {environment: todo_list, variant: 3}`.
- verl cuts tool output to 256 characters by default; raise
  `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` for shell output.
- Written against verl v0.9.1 and tested against a stand-in for its loop.

### DAPO

```python
from deepgym.integrations.dapo import make_dapo_reward_fn

reward_fn = make_dapo_reward_fn(env)
scores = reward_fn(completions=['def solve(x): return x'])
```

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

### OpenRLHF

```python
from fastapi import FastAPI
from deepgym.integrations.openrlhf import create_openrlhf_router

app = FastAPI()
app.include_router(create_openrlhf_router(env, dg))
# Run with: uvicorn app:app --port 8000
```

### lm-evaluation-harness

```python
from deepgym.integrations.lm_eval import register_deepgym_tasks
register_deepgym_tasks()  # registers deepgym_* tasks

# lm_eval --model hf --model_args pretrained=Qwen/Qwen2-0.5B-Instruct \
#         --tasks deepgym_coin_change,deepgym_two_sum
```

### Hugging Face Hub

```python
from deepgym.integrations.hf import push_environment_to_hub, load_environment_from_hub

push_environment_to_hub(env, repo_id='your-org/deepgym-coin-change', env_name='coin_change')
env = load_environment_from_hub('your-org/deepgym-coin-change')
```

## Repo, terminal and cyber tasks

These tasks apply the model's patch, or run its commands, and then run tests, all
outside the judge isolation that `deepgym_solution` gives single-file solutions. So
they only run in a sandbox: with `mode='local'` (or `auto` without Daytona) they raise
`SandboxError`. Use `docker`, `apple`, `modal`, `e2b` or `daytona`.

### SWE-bench Pro

Clones the repo, checks out the base commit, applies the model's patch, runs the
tests and scores by pass fraction.

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='daytona')  # clones the repo, so the sandbox needs network
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

### MiMo-V2.6-RL code tasks

[XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)
(Apache-2.0) is the RL data behind MiMo-V2.6. Its code subset has 2,698 repo-level
tasks, each with its own Docker image, hidden test patch and test command.

```bash
pip install datasets
python scripts/import_mimo_rl.py   # writes ~/.deepgym/environments/mimo_code/
docker pull docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457
```

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='docker')
env = load_environment('mimo_code/format-code-task-001457')
result = dg.run(env, '''```diff\n... unified diff ...\n```''')
```

- The sandbox boots from the task's image. The model's diff is applied, every file
  the hidden test patch touches is put back as it was, and then the hidden tests are
  applied. The score is 1.0 if the test command exits 0, else 0.0, as in MiMo's
  training. A diff can't edit the hidden tests, but it can still change the code they
  run, so the reward is only as strong as the tests.
- The tests run as root in that sandbox because the images expect root's toolchains
  and caches, so the sandbox, not `deepgym_solution`, is what isolates them.
- Images are amd64 and up to about 3.4 GB. Pull them first: a sandbox waits at most
  5 minutes for its image. Docker sandboxes are capped at 2 CPUs and 2 GB, which some
  builds exceed.
- The other subsets aren't imported because their rewards can't be reproduced here:
  cyber grades with a submit script inside each image, general with LLM-judged
  rubrics against mock apps, webdev with a visual grader, and music with MiMo's
  statistical style scorer.

### Terminal-Bench 2.0

Runs terminal commands in a task sandbox and checks the expected output or state.

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='daytona')
env = load_environment('terminal_bench_2')

result = dg.run(
    env,
    model_output='python solve.py --input data.txt > output.txt',
    task_id='regex-log',
)
print(result.score)
```

### Mix benchmarks behind one reward function

```python
from deepgym import MixedEnvironment, load_environment
from deepgym.integrations.trl import make_trl_reward_fn

mixed = MixedEnvironment(environments=[
    (load_environment('swebench_pro'), 0.6),
    (load_environment('terminal_bench_2'), 0.2),
    (load_environment('coin_change'), 0.2),
])

reward_fn = make_trl_reward_fn(mixed)
```

### CyberBench/CyberGym

DeepGym turns CyberGym Hugging Face metadata and artifacts into CyberBench-style
RL tasks for defensive cyber training: local patch repair, log triage, forensics
and synthetic CTF-style tasks. The reward comes from a deterministic verifier,
not from live targets.

The scripts below run from a clone of this repo, in the `deepgym/` folder:

```bash
pip install "deepgym[hf,daytona]"
cp .env.example .env
# edit .env with DAYTONA_API_KEY, ZAI_API_KEY, ZAI_API_BASE, ZAI_MODEL
```

Generate CyberBench seed specs from CyberGym metadata. `--use-zai` lets GLM/Z.ai
enrich the seed plans while keeping the local-only safety constraints:

```bash
python scripts/inspect_cybergym_hf.py --repo-id sunblaze-ucb/cybergym --limit 100
python scripts/generate_cyberbench_seeds.py --count 100 --use-zai
```

For RL, use artifact-backed patch tasks rather than metadata-only prompts. The
runner downloads the vulnerable repository archive and reference patch, applies
a model-produced unified diff in a sandbox, then scores application, touched-file
overlap, changed-line similarity, minimality and safety scope.

```bash
# Daytona-isolated execution for model outputs
python scripts/run_cybergym_artifact_eval.py --count 20 --mode daytona --max-parallel 4

# Ask GLM for candidate patches when rate limits allow
python scripts/run_cybergym_artifact_eval.py --count 20 --answer-source glm --fallback-to-reference
```

Inside a trainer, score completions like any other environment:

```python
from deepgym import DeepGym
from deepgym.cybergym_artifacts import CyberGymPatchEnvironment, load_cybergym_rows

dg = DeepGym(mode='daytona')
row = load_cybergym_rows(count=1)[0]
env = CyberGymPatchEnvironment.from_row(row)

candidate_patch = model.generate(env.task)  # return a unified diff
result = dg.run(env, model_output=candidate_patch)
reward = result.score
```

Use the generated `data/cyberbench/*.jsonl` files as curriculum inputs for
TRL/verl/OpenRLHF. Keep training and evaluation splits separate, use Daytona for
untrusted completions, and don't train on metadata-only scores. See
[docs/cyberbench.md](https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/cyberbench.md)
for the full workflow and current diagnostic numbers.

## Agents and sessions

### Computer-use agents

Computer-use environments boot Daytona's desktop (xfce + Chromium), run a setup
script, and hand the agent a screenshot. Actions are clicks, typing, keys, scrolls
and drags in screenshot pixels. The verifier runs in the same sandbox and checks
what the agent changed (saved app state, files), never pixels. Needs `DAYTONA_API_KEY`.

```python
from deepgym import Action, DeepGym, load_environment

dg = DeepGym(mode='daytona')
env = load_environment('settings_tabs')

with dg.session(env) as session:
    obs = session.start()                     # obs.screenshot is a base64 PNG
    while not session.done:
        action = my_agent(obs)                # e.g. Action(action_type='click', x=376, y=121)
        obs, reward, done = session.step(action)
    result = session.score()
print(result.score, result.output)            # "3/3 changes made"
```

Or in one call: `trajectory, result = dg.run_episode(env, my_agent)`.

Each task has 8 seeded variants (different names, values, files) so a policy
can't memorize one answer: `load_environment('todo_list', variant=3)`. The count
is in `list_environments()` under `variants`.

Verifiers split checks into *progress* (the changes the task asked for) and
*guards* (work was saved, nothing else broke, no duplicate rows). Any failed
guard scores 0, so neither doing nothing nor clicking Save on untouched state
earns reward. Setup locks the desktop down before the agent starts: no terminal,
run dialog or browser devtools, and browser tasks get no file manager.
Known gap: Thunar's "Open with custom command" still runs commands in the
file-manager task.

The same session API runs shell-based multi-turn tasks (`MultiTurnEnvironment`),
and the local backend works for those without Daytona.

### Sessions over HTTP

`deepgym serve` exposes sessions to trainers in any language:

```
POST   /v1/sessions                {"environment": "todo_list", "variant": 3}  -> session_id, observation
POST   /v1/sessions/{id}/step      {"action_type": "type", "content": "Pay rent"}
POST   /v1/sessions/{id}/score     -> RunResult
DELETE /v1/sessions/{id}
```

`environment` can also be an inline `MultiTurnEnvironment` spec. Idle sessions
are closed after 30 minutes; at most 100 run at once.

## Environments

### Built-in (31, ship with the package)

**Coding (22):**
- Array/String: reverse_string, palindrome_check, anagram_check, max_subarray, rotate_array, remove_duplicates, valid_parentheses
- Hash Map: group_anagrams, longest_consecutive, top_k_frequent
- DP: climbing_stairs, coin_change, longest_common_subsequence, house_robber
- Graph/Tree: binary_search, merge_intervals, level_order_traversal
- Practical: fizzbuzz, roman_to_integer, matrix_spiral
- Filesystem/CLI scripting: file_organizer, cli_task

**Computer-use (5, multi-turn, Daytona desktop):** signup_form, todo_list, settings_tabs, expense_sheet, file_manager

**Tool-use (2):** api_request, data_pipeline

**Cyber (2):** cyber/cve_2021_44228_log_triage, cyber/web_path_traversal_toy

```python
from deepgym import load_environment
env = load_environment('coin_change')
```

### Importable benchmarks

From a clone of this repo, in the `deepgym/` folder:

```bash
python scripts/import_humaneval.py      # 164 problems
python scripts/import_evalplus.py       # HumanEval+ and MBPP+
python scripts/import_mbpp.py           # 500 problems
python scripts/import_bigcodebench.py   # 1,140 problems
python scripts/import_mimo_rl.py        # 2,698 MiMo-V2.6-RL code tasks
```

Imported problems are then available through `load_environment()`.
`swebench_pro` and `terminal_bench_2` load the same way (see
Repo, terminal and cyber tasks above).

## Write your own verifier

```python
from deepgym import DeepGym, Environment

dg = DeepGym(mode='local')
env = Environment(
    task='Write a function `add(a, b)` that returns the sum of two numbers.',
    verifier_code=(
        'from deepgym_solution import load_solution\n'
        'sol = load_solution(solution_path)\n'
        'return 1.0 if hasattr(sol, "add") and sol.add(2, 3) == 5 else 0.0\n'
    ),
)

result = dg.run(env, model_output='def add(a, b):\n    return a + b\n')
```

### Running the solution

Never import or run the solution yourself. Use the helpers in `deepgym_solution`:

- `load_solution(path)` starts it in a separate, isolated process and returns a
  proxy: `sol.add(2, 3)` runs there and returns plain data.
- `run_solution(path, cwd=..., env=...)` runs it as a script, writing only inside
  `cwd`. Pass `allow_subprocess=True` if the script needs to start other programs
  (macOS blocks forking otherwise).
- `read_output(path, cwd)` reads its files back and refuses symlinks.
- `solution_source(path, name=None)` reads the source for static checks without running it.

Each call is bounded by `case_timeout` (default 10s) and by the run's `timeout`,
so a slow case fails on its own and the run still returns partial credit.

### Isolation

- **Sandboxes (Daytona, Modal, E2B, Docker, Apple container):** the solution runs as
  a separate Linux user that can't read the judge's files, touch the judge's
  process, or write to its output, with no network.
- **Locally:** it runs under `sandbox-exec` on macOS or `bwrap` on Linux, with the
  judge directory and the deepgym package hidden. If neither is available, you
  still get process separation and a warning.

What this blocks:

- Only plain data crosses the boundary, so an object that claims to equal
  everything can't fake a pass.
- A fake score line printed by the solution never reaches the judge's output.
- Patching the verifier at runtime has no effect.

### Verifier shapes

Detected by parsing, or set explicitly with `verifier_mode`:

- `script`: has a `__main__` block that prints protocol JSON.
- `verify_function`: defines `verify(solution_path, test_cases_path=None)`.
- `body` (the example above): the body of that function.

Return a float, bool or dict; the wrapper normalizes it. `from __future__`
imports are fine in any mode.

Shared helper code goes in `support_files` (name to content). These files sit
next to the verifier, are importable by it, and are hidden from the solution. In
a registry env, list paths in `metadata.json`, for example
`"support_files": ["../_shared/scoring.py"]`.

### Output protocol

Verifiers print JSON to stdout:

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

`cases` tells you which tests passed and failed, so your training loop gets a
denser signal than 0 or 1. A verifier that returns a float or bool is wrapped
into this format automatically.

## Quality gate

Check environments before you train on them:

```bash
deepgym qa coin_change two_sum          # or --all
deepgym qa --all --mode daytona --reruns 5
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
  deepgym qa coin_change --mode docker --model Qwen/Qwen3-8B --samples 8
```

Each single-turn environment must pass:

- `reference_passes`: its `reference_solution.py` scores 1.0.
- `deterministic`: the reference scores the same on every rerun.
- `empty_fails`, `stub_fails`: an empty file, and a stub of the reference's
  functions that return `None`, both score below 1.0.
- `hack_resistant`: none of the heuristic reward-hack attacks behind
  `deepgym audit` reaches the pass threshold.
- `difficulty` (with `--model`): pass@1 and pass@k over k samples from an
  OpenAI-compatible endpoint. Fails as `too_easy` or `too_hard` when every or no
  sample is solved, because then group-relative RL gets no signal from the task.

`--model` runs model-generated code, so with `--mode local` (the default) or
`auto` (which can fall back to local) it refuses to start unless you pass
`--allow-local-exec`. Use a sandboxed mode instead.

Multi-turn and computer-use environments are reported as skipped. The command
exits 1 if any environment fails and writes one report per environment to
`~/.deepgym/qa/<env>.json` (set `DEEPGYM_QA_DIR` to change it):

```json
{
  "environment": "coin_change",
  "passed": true,
  "checks": [{"name": "reference_passes", "passed": true, "details": "score 1.00 (25/25 passed)"}],
  "created_at": "2026-10-12T16:45:34Z"
}
```

## CLI

```bash
deepgym run --task task.md --verifier verifier.py --solution solution.py

# Dev server
DEEPGYM_NO_AUTH=true deepgym serve --host 127.0.0.1 --port 8000 --allow-local-exec

# Production server
DEEPGYM_API_KEY=your-key DAYTONA_API_KEY=your-key deepgym serve --port 8000
```

The server won't start without auth configured. Set `DEEPGYM_API_KEY` for
production or `DEEPGYM_NO_AUTH=true` for local development.
`--allow-local-exec` is required when running without Daytona.

Other commands: `run-batch`, `eval`, `create`, `audit`, `qa`, `benchmark-audit`,
`generate-prm` and `console`. Run `deepgym <command> --help` for options.

## Console

A local web console for the environments, runs, episodes, sandboxes and QA reports
on your machine (`pip install "deepgym[server]"`):

```bash
deepgym console --port 8080 --allow-local-exec   # http://127.0.0.1:8080
```

- **Environments:** search and filter the registry; open one to read its task,
  verifier, reference solution and metadata.
- **Runs:** score a pasted solution or the reference against one environment or a
  small eval, on any provider (`local`, `docker`, `apple`, `daytona`, `modal`, `e2b`).
  Runs execute in the background; the run page shows per-case results and a diff
  against the reference.
- **Episodes:** replay recorded trajectories step by step, with screenshots for
  computer-use tasks. Record one from your agent loop:

  ```python
  from deepgym.console import record_episode
  trajectory, result = dg.run_episode(env, agent)
  record_episode(trajectory, result, environment='todo_list')
  ```

- **Sandboxes:** live `deepgym-*` containers from Docker and Apple `container`,
  with kill. Cloud sandboxes are not listed.
- **QA:** the verdicts `deepgym qa` writes to `~/.deepgym/qa/` (or `$DEEPGYM_QA_DIR`).

Runs and episodes are stored in `~/.deepgym/console.db`. Reference solutions may
run locally, but a pasted solution only runs on your machine with
`--allow-local-exec`; otherwise pick a sandbox provider. The console has no login,
so other websites can't drive it: every `/api` call needs the per-launch token the
console writes into its pages (`X-DeepGym-Token`), writes from other origins are
refused, and when bound to `127.0.0.1` it only answers requests addressed to
localhost. `deepgym web` is an alias.

The UI lives in `console-ui/` (Next.js static export, built with bun) and ships
prebuilt in wheels. From a clone, build it once with
`cd console-ui && bun install && bun run build`.

## Development

From the `deepgym/` folder of a clone:

```bash
pip install -e ".[dev]"   # install with test deps
pytest
ruff check src/           # lint
ruff format src/          # format
```

### Release

PyPI publishing is tag-driven in GitHub Actions. Bump `version` in
`pyproject.toml`, then push a `v*` tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Branch pushes and PRs run CI only. A `v*` tag also runs the publish job.

## Contributing

New environments, verifiers, benchmark importers and integrations are all welcome.
Start with a [good first issue](https://github.com/DeepGym/deepgym/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
and see [CONTRIBUTING.md](https://github.com/DeepGym/deepgym/blob/main/.github/CONTRIBUTING.md).
We're in Hacktoberfest: merged PRs get the `hacktoberfest-accepted` label.

If DeepGym is useful to you, a star helps other people find it.

## License

[PolyForm Noncommercial License 1.0.0](https://github.com/DeepGym/deepgym/blob/main/LICENSE).
Commercial use needs a license: email [contact@deepgym.io](mailto:contact@deepgym.io).

We changed the license after 0.4.0 to protect the new work going into DeepGym.
Releases up to 0.4.0 stay under MIT. The vendored UI kits in
`console-ui/components/base` and `console-ui/components/primitives` keep their own
MIT licenses.

---

Need custom environments, a private deployment or help with RL training?
Email [contact@deepgym.io](mailto:contact@deepgym.io) or visit [deepgym.io](https://deepgym.io).
