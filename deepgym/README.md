# DeepGym

[![PyPI](https://img.shields.io/pypi/v/deepgym?label=pypi)](https://pypi.org/project/deepgym/)
[![CI](https://github.com/DeepGym/deepgym/actions/workflows/deepgym-ci.yml/badge.svg)](https://github.com/DeepGym/deepgym/actions/workflows/deepgym-ci.yml)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue)](https://github.com/DeepGym/deepgym/blob/main/LICENSE)

**RL environments with verifiable rewards you can trust.**

DeepGym runs a model's answer (code, a patch, shell commands or desktop actions) in a
sandbox and scores it with a verifier kept apart from that answer. When the infrastructure
fails instead of judging the answer, the run gets no verdict and can be kept out of
training instead of counting as a 0. It's for teams training and evaluating coding and
computer-use agents with verl, TRL, OpenRLHF, Axolotl or their own GRPO loop.

## Install

```bash
pip install deepgym
```

Python 3.10 or later. Optional extras:

| Extra | Adds |
|---|---|
| `deepgym[daytona]`, `deepgym[modal]`, `deepgym[e2b]` | Those sandbox providers |
| `deepgym[server]` | `deepgym serve` and `deepgym console` |
| `deepgym[hf]` | Hugging Face Hub sharing and dataset loaders |
| `deepgym[lm_eval]` | lm-evaluation-harness tasks |
| `deepgym[traces]` | Parquet trace export |
| `deepgym[all]` | Everything above except `lm_eval` |

## Quickstart

Score a solution to a built-in task. This runs as is, on your machine:

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
print(f'Score: {result.score}, Passed: {result.passed}')   # Score: 1.0, Passed: True

wrong = dg.run(env, model_output='def coin_change(coins, amount):\n    return -1\n')
print(wrong.score, wrong.passed)   # 0.12 False: 3 of 25 cases pass, each in wrong.cases
```

`mode='local'` is for trying things out. For model-generated code, switch to a sandbox
provider such as `mode='docker'`; nothing else changes.

## Core concepts

- **Environment:** a task prompt plus a verifier. Single-turn (`Environment`: the model
  answers once), multi-turn (`MultiTurnEnvironment`: one live sandbox per episode) or
  computer use (a desktop). Load one by name with `load_environment`, or scaffold your
  own with `deepgym new`.
- **Verifier:** Python that scores an answer and prints JSON: a score in [0, 1], a pass
  flag, and optionally each test case with a reason. It reaches the answer only through
  the `deepgym_solution` helpers, which run it in a separate, isolated process with no
  network.
- **Sandbox providers:** where answers run, picked with `mode`. In every sandbox mode, code
  run through the `deepgym_solution` helpers and shell sessions runs as a separate Linux
  user with no network; repo-patch tests run the patched code as root inside the sandbox.
  Docker shares the host's kernel, so it isolates less than the VM-based providers.

  | `mode` | Runs on | Setup |
  |---|---|---|
  | `local` | This machine, under `sandbox-exec` (macOS) or `bwrap` (Linux) | None; for development only, and repo-patch, Harbor and CyberGym tasks refuse it |
  | `docker` | A throwaway container with no network | A running Docker daemon |
  | `apple` | A throwaway Linux VM from Apple's `container` CLI (macOS) | `brew install container`, then `container system start` |
  | `daytona` | A Daytona sandbox, cloud or self-hosted; desktops need it | `deepgym[daytona]` and `DAYTONA_API_KEY` (self-hosted: also `DAYTONA_API_URL`) |
  | `modal` | A Modal sandbox | `deepgym[modal]`, then `modal token set` |
  | `e2b` | An E2B sandbox | `deepgym[e2b]` and `E2B_API_KEY` |
  | `auto` (default) | Daytona when `DAYTONA_API_KEY` is set, otherwise `local`; also `local`, with a warning, if `deepgym[daytona]` isn't installed | |

- **Reward contract:**
  - *Infrastructure failures give no verdict.* When the sandbox or verifier fails before
    the answer is in play, or the provider or network fails, `result.has_verdict` is
    false, and trainer integrations keep that sample out of training where the trainer
    allows it.
  - *Failures caused by the policy are scored.* Failing tests, a malformed patch, a patch
    that touches test infrastructure, a solution that crashes or hangs the verifier, or a
    sandbox the policy killed is a real 0. A score printed before a crash is never kept.
  - *Headline totals stay conservative.* Batch and eval totals count a run without a
    verdict as a failed 0, and judged-only numbers sit beside them.
- **Traces:** `dg.trace()` records an episode with its provenance as it runs, and
  `deepgym traces export` writes it as JSONL, Parquet rows for verl and TRL, or ATIF.

More in [core concepts][concepts] and [the reward contract][reward-contract].

## What's in the box

- **31 built-in environments:** 22 coding, 5 computer-use, 2 tool-use and 2 cyber tasks
  ([list][built-ins]).
- **Benchmark importers:** HumanEval, MBPP, EvalPlus, BigCodeBench, CyberGym, SWE-bench
  Pro, MiMo-V2.6-RL code tasks and Harbor datasets such as Terminal-Bench 2.0
  ([importers][importers]).
- **Computer use:** desktop tasks with 8 seeded variants each; screenshots in, mouse and
  keyboard out, scored on the state the agent leaves ([computer use][computer-use]).
- **Multi-turn sessions:** one live sandbox per episode, from Python (`dg.session()`) or
  HTTP (`/v1/sessions`) ([sessions][sessions]).
- **QA gate:** `deepgym qa` checks that the reference passes, empty and stub answers earn
  nothing and no reward-hack answer reaches 0.5; with `--mutants` it also checks that no
  mutant of the reference passes ([QA gate][qa-gate]).
- **Environment authoring:** `deepgym new`, `deepgym check` and `deepgym qa --approve`,
  from an idea to an approved environment folder ([authoring][authoring],
  [approvals][approvals]).
- **Trace export:** JSONL, Parquet (verl and TRL rows) and ATIF ([traces][traces]).
- **Trainer integrations:** [verl][verl] (reward function and agent loops), [TRL][trl],
  [OpenRLHF][openrlhf], [lm-evaluation-harness][lm-eval] and [Axolotl][axolotl].
- **Sandboxes:** Docker, Apple container, Daytona, Modal and E2B ([sandboxes][sandboxes]).
- **Also:** Hugging Face Hub sharing, `MixedEnvironment` for several benchmarks behind one
  reward function, a Gymnasium-style `reset`/`step` API, and a local web console.

## Examples

### Score a batch for GRPO

Score a group of completions in parallel and keep runs without a verdict out of the group
statistics (continues the quickstart):

```python
completions = [solution, 'def coin_change(coins, amount):\n    return -1\n']
batch = dg.run_batch(env, completions, max_parallel=8)

rewards = [r.score if r.has_verdict else None for r in batch.results]   # [1.0, 0.12]
print(batch.avg_score, batch.judged_count, batch.no_verdict_count)      # 0.56 2 0
```

`batch.avg_score` and `batch.passed` count a run without a verdict as a failed 0;
`judged_mean_score` and `judged_pass_rate` leave it out. `RewardFunction(env)` wraps the
same thing for any framework. More in [the reward contract][reward-contract].

### Train with TRL

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

The reward function runs the last fenced code block of each completion, plain text or
conversation, and returns `None` for a completion whose sandbox or verifier failed. TRL
1.6+ leaves a completion out of the group's baseline when every reward function returned
`None` for it. More in [TRL][trl].

### Train with verl

verl loads DeepGym's reward function from the package, and each dataset row names its
environment:

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
    reward_kwargs: {mode: docker}   # or daytona, modal, e2b, apple, auto, local
```

A run without a verdict raises `NoVerdictError`, which verl's v1 trainer turns into a
dropped sample. For the legacy trainer and for multi-turn agents, DeepGym ships agent
loops; see [verl][verl].

### Serve rewards to OpenRLHF

```python
from fastapi import FastAPI
from deepgym import DeepGym, load_environment
from deepgym.integrations.openrlhf import create_openrlhf_router

app = FastAPI()
app.include_router(create_openrlhf_router(
    load_environment('coin_change'), DeepGym(mode='docker'), infra_failure_score='nan'
))
# uvicorn app:app --port 8000
# OpenRLHF: --remote_rm_url http://localhost:8000/reward/score --algo.dynamic_filtering_enable
```

OpenRLHF can't skip a sample, so `infra_failure_score` decides what a sample DeepGym
couldn't grade gets: `'nan'` (dynamic filtering then drops its group) or `'zero'`. More in
[OpenRLHF][openrlhf].

### Evaluate with lm-evaluation-harness

```python
import lm_eval
from deepgym import load_environment
from deepgym.integrations.lm_eval import make_lm_eval_task

tasks = [make_lm_eval_task(load_environment(n), n) for n in ['coin_change', 'two_sum']]
results = lm_eval.simple_evaluate(
    model='hf', model_args='pretrained=Qwen/Qwen2.5-0.5B-Instruct', tasks=tasks
)
```

Each environment is one task with one document: `deepgym_score` and `deepgym_pass` are
NaN without a verdict, and `deepgym_verdict` says whether it got one. Average across tasks
skipping NaN. More in [lm-eval][lm-eval].

### Run a multi-turn session

One sandbox stays alive for the whole episode. This one runs locally as is:

```python
from deepgym import Action, DeepGym, MultiTurnEnvironment

env = MultiTurnEnvironment(
    task='Write the word hello into notes.txt, then send done.',
    final_verifier_code=(
        'import json, pathlib\n'
        'p = pathlib.Path("notes.txt")\n'
        'ok = p.is_file() and p.read_text().strip() == "hello"\n'
        'print(json.dumps({"score": float(ok), "passed": ok}))\n'
    ),
)

dg = DeepGym(mode='local')   # local sessions aren't isolated: use a sandbox for real agents
with dg.session(env) as session:
    obs = session.start()
    obs, reward, done = session.step(Action(action_type='bash', content='echo hello > notes.txt'))
    obs, reward, done = session.step(Action(action_type='done'))
    result = session.score()
print(result.score)   # 1.0
```

Agents act with `code`, `bash`, `write_file` and `done`; an optional step verifier rewards
each step. `dg.run_episode(env, agent)` runs a whole episode in one call. More in
[sessions][sessions].

### Drive a desktop

Computer-use tasks boot a locked-down desktop on Daytona and score what the agent changed,
never pixels:

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='daytona')
env = load_environment('settings_tabs', variant=3)

trajectory, result = dg.run_episode(env, my_agent)   # my_agent(obs) -> Action(...)
print(result.score, result.output)                   # e.g. "3/3 changes made"
```

`obs.screenshot` is a base64 PNG, and actions are clicks, typing, keys, scrolls and drags
in screenshot pixels. More in [computer use][computer-use].

### Record and export traces

Record an episode as training data while it runs, then export it (continues the session
example):

```python
from deepgym.traces import AgentMeta, TraceMessage

agent = AgentMeta(name='react-shell', model='Qwen/Qwen3-8B', sampling={'temperature': 0.7})
with dg.trace(env, agent) as tr:
    obs = tr.start()
    reply = [TraceMessage(role='assistant', content='echo hello > notes.txt')]
    tr.step(Action(action_type='bash', content='echo hello > notes.txt'), reply)
    tr.step(Action(action_type='done'))
    tr.score()
```

```bash
deepgym traces export --format jsonl --out traces.jsonl             # whole traces
deepgym traces export --format parquet --rows rl --out rl.parquet   # verl / TRL rows
deepgym traces export --format atif --out atif/ --only-valid        # ATIF documents
```

Each event is written to `~/.deepgym/traces/` as it happens, so a crash keeps the steps
taken so far. A trace without a verdict is flagged, or left out with `--only-valid`.
Parquet needs `deepgym[traces]`. More in [traces][traces].

### Write an environment

```bash
deepgym new word_stats --kind function   # or script, repo-patch, multi-turn
deepgym check word_stats                 # passes as generated
# edit task.md, the reference, expected() in verifier.py and variants.py, then:
deepgym check word_stats
deepgym qa word_stats                    # reference, determinism, empty, stub, reward hacks
deepgym qa word_stats --mutants --mode docker --approve --by "Jane Doe"
```

`deepgym new` writes a working example that passes every check, so you start green and
change one thing at a time. `--mutants` fails a verifier that passes any small mutation
of the reference (locally it needs `bwrap` on Linux; elsewhere use `docker`, `modal` or
`e2b`). `--approve` writes a read-only `approval.json` with the QA report and a hash of
every file, and `deepgym check` fails once the folder changes. More in
[authoring][authoring], [the QA gate][qa-gate] and [approvals][approvals].

### Write a verifier

```python
from deepgym import DeepGym, Environment

env = Environment(
    task='Write a function `add(a, b)` that returns the sum of two numbers.',
    verifier_code=(
        'from deepgym_solution import load_solution\n'
        'sol = load_solution(solution_path)\n'
        'return 1.0 if hasattr(sol, "add") and sol.add(2, 3) == 5 else 0.0\n'
    ),
)
result = DeepGym(mode='local').run(env, model_output='def add(a, b):\n    return a + b\n')
print(result.score)   # 1.0
```

`load_solution` runs the answer in a separate, isolated process and only plain data comes
back, so in a sandbox, or locally under `sandbox-exec` or `bwrap`, it can't read the
expected answers, print a fake score or patch the verifier. More in [verifiers][verifiers].

### Import a code benchmark

From a clone of this repo, in the `deepgym/` folder (`pip install datasets`):

```bash
python scripts/import_humaneval.py      # 164 problems
python scripts/import_mbpp.py           # 500 problems
python scripts/import_bigcodebench.py   # 1,140 problems
python scripts/import_evalplus.py       # HumanEval+ and MBPP+ (pip install evalplus)
```

```python
env = load_environment('humaneval/HumanEval_0')   # a qualified name picks one problem
```

Imported problems land in `~/.deepgym/environments/` and load like built-ins. More in
[importers][importers].

### Score a repo-level patch

SWE-bench Pro and MiMo tasks apply the model's diff to a real repository and run its
tests, so they need a sandbox provider:

```python
dg = DeepGym(mode='daytona')   # clones the repo, so the sandbox needs network
env = load_environment('swebench_pro')

result = dg.run(
    env,
    model_output='''```diff\n... unified diff ...\n```''',
    repo='owner/repo',
    base_commit='abc123',
    test_patch='diff --git ...',
    test_commands=['pytest -q tests/test_bug.py', 'pytest -q tests/test_smoke.py'],
)
```

The hidden tests are restored after the diff, and a diff that touches tests or test
infrastructure scores 0 with `error_type='test_infra_tampering'`. More in
[importers][importers].

### Import Harbor tasks

Harbor task folders (`instruction.md`, `task.toml`, `tests/test.sh`, an image or
Dockerfile) run as environments whose answer is a bash script:

```bash
deepgym harbor export coin_change two_sum --out ./harbor-tasks   # built-ins as Harbor tasks
deepgym harbor import ./harbor-tasks --register mine             # validate, load as mine/<task>
deepgym qa mine/coin_change --mode docker                        # solve.sh passes, a no-op fails
deepgym harbor import harborframework/terminal-bench-2.0 --limit 5   # or a dataset from the Hub
```

The script runs as an unprivileged user with no network, and a fresh sandbox from the same
image grades a copy of its work directory, so nothing the agent leaves in its sandbox is in
the tests or the reward folder. More in [importers][importers-harbor].

## DeepGym and verl

verl trains the model. DeepGym gives it tasks and scores the answers. Use them together:
verl loads DeepGym's reward function with one config line (see
[Train with verl](#train-with-verl)).

| | verl | DeepGym |
|---|---|---|
| What it is | RL training framework | RL environments and verifiable rewards |
| Does the training | Yes: rollouts on vLLM/SGLang, updates on FSDP/Megatron, PPO, GRPO, DAPO and more | No, it plugs into verl, TRL, OpenRLHF or your own loop |
| Tasks | You bring the dataset and write the reward function | 31 built-in environments; importers for HumanEval, MBPP, EvalPlus, BigCodeBench, SWE-bench Pro, MiMo, CyberGym and Harbor datasets such as Terminal-Bench 2.0 |
| Running model code | Your reward function, or a remote code sandbox service | Daytona, Modal, E2B, Docker, Apple container, or locally under sandbox-exec/bwrap (without them, local runs get process separation and a warning) |
| Reward hacking | Up to your reward function | The solution runs apart from the verifier with no network, so it can't read the expected answers or print a fake score; `deepgym qa` checks that empty and stub solutions earn nothing and reward-hack ones stay below 0.5 before you train |
| Multi-turn agents | Agent loop and tool interface | One live sandbox per episode (shell or desktop), with a verl agent loop and session tool |
| Computer use | Not built in | Desktop tasks, scored on the state the agent leaves behind |
| Failed runs | A reward function returns a score; it can't mark a sample as unscored | A run whose sandbox or verifier failed has no verdict: verl's v1 trainer drops the sample, and DeepGym's agent loops drop or mask it |
| Managing it | Training logs (wandb, TensorBoard) | `deepgym console`: environments, runs, episode replay, sandboxes, QA reports |

## For teams and labs

DeepGym is free for personal, research and other noncommercial use. Commercial use,
including inside a company or in a product you sell, needs a commercial license. We also
build private environments and verifiers for your tasks, deploy DeepGym privately or
self-hosted, and help set up and run RL training. For DeepGym Pro access or a custom RL
environment, email [contact@deepgym.io](mailto:contact@deepgym.io) or visit
[deepgym.io](https://deepgym.io).

## CLI

| Command | What it does |
|---|---|
| `deepgym run`, `run-batch` | Score one solution, or a folder of them, against a task and verifier |
| `deepgym eval` | Score solutions against a suite of built-in environments |
| `deepgym qa` | Quality-gate environments before training on them |
| `deepgym new`, `check` | Scaffold and validate an environment folder |
| `deepgym audit`, `benchmark-audit` | Audit a verifier for reward hacking, or a benchmark for split leakage |
| `deepgym harbor import`, `export` | Import Harbor tasks, or export environments as Harbor tasks |
| `deepgym traces export` | Export recorded traces as training data |
| `deepgym create` | Print an environment as JSON for `POST /v1/environments` |
| `deepgym serve` | Start the HTTP API |
| `deepgym console` | Launch the local web console (alias `deepgym web`) |

```bash
deepgym run --task task.md --verifier verifier.py --solution solution.py

# Dev server
DEEPGYM_NO_AUTH=true deepgym serve --host 127.0.0.1 --port 8000 --allow-local-exec

# Production server
DEEPGYM_API_KEY=your-key DAYTONA_API_KEY=your-key deepgym serve --host 0.0.0.0 --port 8000
```

The server won't start without auth configured: set `DEEPGYM_API_KEY` for production or
`DEEPGYM_NO_AUTH=true` for local development. Without `DAYTONA_API_KEY` it runs code on
the host, so it also needs `--allow-local-exec`. Run `deepgym <command> --help` for
options, and see [the CLI reference][cli] and [the API server][api-server].

## Console

A local web console for the environments, runs, episodes, sandboxes and QA reports on
your machine (`pip install "deepgym[server]"`):

```bash
deepgym console --port 8080 --allow-local-exec   # http://127.0.0.1:8080
```

- **Environments:** search the registry and read each task, verifier, reference and
  metadata.
- **Runs:** score a pasted solution or the reference against one environment or a small
  eval, on any provider, with per-case results and a diff against the reference.
- **Episodes:** replay recorded trajectories and traces step by step, with screenshots
  for computer-use tasks. Store one from your agent loop with
  `deepgym.console.record_episode(trajectory, result, environment=...)` or
  `record_episode(tr.trace)`.
- **Sandboxes:** live `deepgym-*` containers from Docker and Apple `container`, with kill.
- **QA:** the reports `deepgym qa` writes.

Runs and episodes are stored in `~/.deepgym/console.db`. A pasted solution only runs on
your machine with `--allow-local-exec`; otherwise pick a sandbox provider. The console has
no login, so every `/api` call needs a per-launch token the console writes into its pages,
and writes from other origins are refused. More in [the console docs][console].

## Development

From the `deepgym/` folder of a clone:

```bash
pip install -e ".[dev]"     # install with test deps
pytest
ruff check src/ tests/      # lint
ruff format src/ tests/     # format
```

The console UI lives in `console-ui/` (Next.js static export, built with bun) and ships
prebuilt in wheels. From a clone, build it once with
`cd console-ui && bun install && bun run build`.

Releases are tag-driven: bump `version` in `pyproject.toml`, then push a `v*` tag
(`git tag vX.Y.Z && git push origin vX.Y.Z`). Branch pushes and PRs run CI only; a tag
also publishes to PyPI, and fails if the tag isn't the built version or that version is
already on PyPI.

## Contributing

New environments, verifiers, benchmark importers and integrations are all welcome. Start
with a [good first issue](https://github.com/DeepGym/deepgym/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
and see [CONTRIBUTING.md](https://github.com/DeepGym/deepgym/blob/main/.github/CONTRIBUTING.md).

## License

[PolyForm Noncommercial License 1.0.0](https://github.com/DeepGym/deepgym/blob/main/LICENSE).
Commercial use needs a license: [contact@deepgym.io](mailto:contact@deepgym.io).

[concepts]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/concepts.md
[reward-contract]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/training/reward-contract.md
[built-ins]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/environments/built-ins.md
[importers]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/environments/importers.md
[importers-harbor]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/environments/importers.md#harbor-tasks
[computer-use]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/environments/computer-use.md
[sessions]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/environments/sessions.md
[authoring]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/authoring/new-check-qa.md
[verifiers]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/authoring/verifiers.md
[qa-gate]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/authoring/qa-gate.md
[approvals]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/authoring/approvals.md
[verl]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/training/verl.md
[trl]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/training/trl.md
[openrlhf]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/training/openrlhf.md
[lm-eval]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/training/lm-eval.md
[axolotl]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/training/axolotl.md
[traces]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/traces.md
[sandboxes]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/sandboxes.md
[cli]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/cli.md
[api-server]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/api-server.md
[console]: https://github.com/DeepGym/deepgym/blob/main/deepgym/docs/console.md
