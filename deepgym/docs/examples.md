# Examples

This page has worked examples for scoring batches, each trainer integration, sessions,
agents, traces, authoring and benchmark imports. Some continue the
[quickstart](../README.md#quickstart) and reuse its `dg`, `env` and `solution`.

## Score a batch for GRPO

Score a group of completions in parallel and keep runs without a verdict out of the group
statistics (continues the quickstart):

```python
completions = [solution, 'def coin_change(coins, amount):\n    return -1\n']
batch = dg.run_batch(env, completions, max_parallel=8)

rewards = [r.score if r.has_verdict else None for r in batch.results]   # [1.0, 0.0]
print(batch.avg_score, batch.judged_count, batch.no_verdict_count)      # 0.5 2 0
```

`batch.avg_score` and `batch.passed` count a run without a verdict as a failed 0;
`judged_mean_score` and `judged_pass_rate` leave it out. `RewardFunction(env)` wraps the
same thing for any framework. More in [the reward contract][reward-contract].

## Train with TRL

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

## Train with verl

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

## Serve rewards to OpenRLHF

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

## Evaluate with lm-evaluation-harness

```python
import lm_eval
from deepgym import load_environment
from deepgym.integrations.lm_eval import make_lm_eval_task

# max_gen_toks: your model's maximum output (Qwen2.5 generates up to 8K tokens).
tasks = [
    make_lm_eval_task(load_environment(n), n, max_gen_toks=8192)
    for n in ['coin_change', 'two_sum']
]
results = lm_eval.simple_evaluate(
    model='hf', model_args='pretrained=Qwen/Qwen2.5-0.5B-Instruct', tasks=tasks
)
```

Each environment is one task with one document: `deepgym_score` and `deepgym_pass` are
NaN without a verdict, and `deepgym_verdict` says whether it got one. Average across tasks
skipping NaN. More in [lm-eval][lm-eval].

## Run a multi-turn session

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

## Drive a desktop with a computer-use agent

Computer-use tasks boot a locked-down desktop on Daytona and score what the agent changed,
never pixels (`pip install "deepgym[daytona,anthropic]"`, plus `DAYTONA_API_KEY` and
`ANTHROPIC_API_KEY`):

```python
from deepgym import DeepGym, load_environment
from deepgym.agents import AnthropicComputerAgent, Budget

dg = DeepGym(mode='daytona')
env = load_environment('settings_tabs', variant=3)

result, grade = dg.run_episode(
    env,
    AnthropicComputerAgent(model='claude-opus-5-5'),
    budget=Budget(max_turns=60, max_seconds=1800),
)
print(grade.score, result.termination_reason, result.turns)
```

`OpenAIComputerAgent` works the same way. Your own agent can be any callable that maps an
observation (`obs.screenshot` is a base64 PNG) to an `Action`: clicks, typing, keys,
scrolls and drags in screenshot pixels. More in [computer use][computer-use].

## Run Claude Code or Codex in the sandbox

`ClaudeCodeAgent` drives the Claude Code CLI (`claude -p`), not the Claude Agent SDK. The
CLI runs headless inside the session's sandbox as an unprivileged user, and DeepGym scores
the state it leaves:

```python
from deepgym import DeepGym, MultiTurnEnvironment
from deepgym.agents import Budget, ClaudeCodeAgent

env = MultiTurnEnvironment(
    task='Fix the failing test in the repo, then stop.',
    snapshot='my-snapshot-with-claude-code',   # the CLI must be installed in the image
    allowed_actions=frozenset({'cli', 'done'}),
    cli_provider_egress=True,                   # lets the provider proxy reach its API
    final_verifier_code=verifier,
)
agent = ClaudeCodeAgent(model='claude-opus-5-5', allow_model_credentials=True)
result, grade = DeepGym(mode='daytona').run_episode(
    env, agent, budget=Budget(max_turns=100, max_seconds=1800)
)
```

The agent's network reaches only the provider's API host, through a proxy outside its
namespace, and every process it started is stopped before grading. The provider key is
readable by the model, which is why `allow_model_credentials` must be set. `CodexAgent`
works the same way. Supported CLI versions live in one place
(`deepgym.agents.SUPPORTED_CLIS`), and `cli_install_command()` gives the install line for
your image. More in [agents][agents].

## Record and export traces

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

## Write an environment

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

## Write a verifier

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

## Import a code benchmark

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

## Score a repo-level patch

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

## Import Harbor tasks

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

[reward-contract]: training/reward-contract.md
[trl]: training/trl.md
[verl]: training/verl.md
[openrlhf]: training/openrlhf.md
[lm-eval]: training/lm-eval.md
[sessions]: environments/sessions.md
[computer-use]: environments/computer-use.md
[agents]: agents.md
[traces]: traces.md
[authoring]: authoring/new-check-qa.md
[qa-gate]: authoring/qa-gate.md
[approvals]: authoring/approvals.md
[verifiers]: authoring/verifiers.md
[importers]: environments/importers.md
[importers-harbor]: environments/importers.md#harbor-tasks
