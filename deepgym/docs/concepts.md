# Core concepts

This page covers the five ideas DeepGym is built on (environments, verifiers, sandbox
providers, the reward contract and traces) and how one run moves through them. Each
section links to the page with the details.

## How a run works

```
prompt --> model --> DeepGym sandbox --> verifier --> score --> training loop
                        |                  |
                     Daytona / Modal /  JSON protocol:
                     E2B / Docker /     score, passed, cases
                     Apple / local
```

DeepGym takes the model's answer, runs it in a sandbox, and lets the environment's verifier
score it. The verifier prints one JSON line with a score in [0, 1], a pass flag and a
per-test breakdown. DeepGym checks that line and returns it as a `RunResult`, which a
reward function hands to your trainer.

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='local')
env = load_environment('two_sum')

result = dg.run(env, model_output='def two_sum(nums, target): ...')
print(result.score, result.passed, result.has_verdict, result.cases)
```

`dg.run_batch(env, outputs, max_parallel=8)` scores many answers to one environment in
parallel, as GRPO needs. `AsyncDeepGym` has the same `run`, `run_batch` and `eval`
methods as coroutines, and with `api_url=` it sends them to a remote `deepgym serve` (see
the [API server](api-server.md)).

## Environments

An environment is a task prompt plus a verifier. There are three kinds:

- **`Environment`** (single-turn): the model answers once, with code, a patch or a shell
  script, and the verifier scores that answer. Most built-ins and every benchmark
  importer are single-turn.
- **`MultiTurnEnvironment`**: one live sandbox per episode. The agent runs commands and
  writes files step by step, an optional step verifier rewards each step, and a final
  verifier scores the state the agent left. See [sessions](environments/sessions.md).
- **`ComputerUseEnvironment`**: a multi-turn environment on a desktop, driven with
  screenshots, mouse and keyboard. See [computer use](environments/computer-use.md).

Load environments by name:

```python
from deepgym import list_environments, load_environment, load_suite

env = load_environment('coin_change')
desktop = load_environment('todo_list', variant=3)   # computer-use tasks have 8 variants
coding = load_suite('coding')   # also easy, medium, hard, tool-use, all, or a family
print(len(list_environments()))
```

Each environment lives in a folder: `task.md` (the prompt), `verifier.py`, a reference
solution that must score 1.0, and `metadata.json`. The [built-ins](environments/built-ins.md)
ship with the package, the [importers](environments/importers.md) bring in public
benchmarks, and `deepgym new` scaffolds your own (see
[authoring](authoring/new-check-qa.md)).

An environment can name the image its sandbox boots from, in `metadata.json` or in code,
so the task's dependencies are already installed:

```python
from deepgym import Environment

env = Environment(task=..., verifier_code=..., image='python:3.11-slim')
```

- Docker, Apple container, Modal and Daytona pull it from a registry; E2B treats it as a
  template id. On Daytona a `snapshot` wins.
- The image must run as root and have python3, bash, useradd and util-linux (setpriv,
  unshare, nsenter, mount). Debian and Ubuntu images do; Alpine doesn't.
- Root in the sandbox runs the image's binaries, so only use images you trust. The HTTP
  API refuses caller-chosen images.

## Verifiers

A verifier is Python that scores one answer. It never imports or runs the answer itself:
the helpers in `deepgym_solution` (`load_solution`, `run_solution` and a few more) start
it in a separate, isolated process with no network, and only plain data crosses back. So
the answer can't read the expected outputs, print a fake score or patch the verifier.

The verifier reports every test case with a reason, which gives a trainer a denser signal
than pass or fail. Its exit code backs the result: 0 passed, 1 failed, 2 verifier error.
Verifiers are seeded, so the same answer gets the same score, and `deepgym qa` checks
that. See [writing verifiers](authoring/verifiers.md).

## Sandbox providers

The `mode` you give `DeepGym` picks where answers run: `local`, `docker`, `apple`,
`daytona`, `modal`, `e2b`, or `auto` (the default: Daytona when `DAYTONA_API_KEY` is set,
otherwise local).

```python
dg = DeepGym(mode='docker')   # same for AsyncDeepGym and dg.session()
```

- Every sandbox mode gives the same isolation, for single runs and shell sessions: the
  answer runs as a separate Linux user that can't read the judge's files, touch its
  process or reach the network. Use one for anything shared or untrusted.
- `local` runs a subprocess under `sandbox-exec` (macOS) or `bwrap` (Linux). It is for
  development, and repo, terminal and cyber tasks refuse it.
- Docker and Apple container give each run a throwaway container or Linux VM, with no
  API key. The default image is `python:3.12-slim` (override with
  `DEEPGYM_CONTAINER_IMAGE` or per environment), and sessions run offline while servers
  inside the sandbox still work on localhost. Stop Apple's service with
  `container system stop`.
- Desktop (computer-use) environments need Daytona.
- Each provider is one folder in `src/deepgym/sandboxes/` that implements a small
  `Sandbox` protocol (run a command as root, upload a file, delete), so adding a provider
  means adding a folder.

Setup for each provider is in [sandboxes](sandboxes.md).

## The reward contract

A run either has a verdict or it doesn't (`result.has_verdict`):

- **Infrastructure failures give no verdict.** When the sandbox, the provider or the
  verifier fails instead of judging the answer, the result's score is a placeholder 0.0.
  Every trainer integration keeps such a sample out of training instead of training on
  the 0.
- **Failures caused by the policy are scored.** Failing tests, a malformed patch or a
  patch that touches test infrastructure is a real 0 to learn from.
- **Headline totals stay conservative.** `passed`, `avg_score` and `pass_rate` count a
  run without a verdict as a failed 0, so a broken run can only lower them. The
  `judged_*` fields beside them count only the runs that got a verdict.

See [the reward contract](training/reward-contract.md).

## Traces

`dg.trace()` records a multi-turn episode while it runs: provenance (environment hashes,
image digest, deepgym version and commit, agent and sampling config), every action with
the model messages behind it, every step reward, and the final verdict. Each event is
appended to disk as it happens, so a crash keeps the steps taken so far.
`deepgym traces export` turns traces into JSONL, Parquet rows for verl and TRL, or ATIF
documents. See [traces](traces.md).

## Gymnasium-style API

`DeepGymEnv` wraps a single-turn environment in `reset`/`step`/`state`:

```python
from deepgym import DeepGymEnv, load_environment

env = DeepGymEnv(load_environment('coin_change'))
obs = env.reset()
obs, reward, done, info = env.step(model_output)   # info.has_verdict, info.cases
results = env.step_batch(outputs)   # one (obs, reward, done, info) per output
```

When `info.has_verdict` is false, the reward is a placeholder 0.0: mask that sample.
