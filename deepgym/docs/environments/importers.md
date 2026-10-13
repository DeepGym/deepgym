# Benchmark importers

This page covers the public benchmarks DeepGym can load or import: the code benchmarks
(HumanEval, MBPP, EvalPlus, BigCodeBench), repo-level patch tasks (SWE-bench Pro, MiMo),
shell tasks (Terminal-Bench 2.0 and any Harbor dataset), CyberGym, mixing benchmarks
behind one reward function, and sharing environments on the Hugging Face Hub.

## Code benchmarks

The import scripts run from a clone of this repo, in the `deepgym/` folder. They need
`pip install datasets`, and the EvalPlus one `pip install evalplus`:

```bash
python scripts/import_humaneval.py      # 164 problems
python scripts/import_evalplus.py       # HumanEval+ and MBPP+
python scripts/import_mbpp.py           # 500 problems
python scripts/import_bigcodebench.py   # 1,140 problems
```

They write environments to `~/.deepgym/environments/`, which `load_environment()` then
finds. A benchmark name loads its first problem, and a qualified name loads one problem
(an ambiguous name raises instead of loading the wrong benchmark):

```python
from deepgym import load_environment

env = load_environment('humaneval_plus/HumanEval_0')
```

- HumanEval and MBPP tests run against the dataset's own helpers and only the names the
  tests call, so a solution that redefines `range`, `set` or a helper can't rig the
  asserts.
- EvalPlus and BigCodeBench pass only when every scored test passes, and keep partial
  credit as the score.
- BigCodeBench counts a skipped test as a failure. A fully skipped suite has no verdict,
  and an empty one scores 0.
- BigCodeBench tests need the solution's real objects, so they run inside the solution's
  isolated process, and a determined model can forge that result.
- Re-run the import scripts after upgrading DeepGym to pick up verifier fixes.

## Repo, terminal and cyber tasks need a sandbox

These tasks apply the model's patch, or run its commands, and then run tests, all outside
the judge isolation that `deepgym_solution` gives single-file solutions. So they only run
in a sandbox: with `mode='local'` (or `auto` without Daytona) they raise `SandboxError`.
Use `docker`, `apple`, `modal`, `e2b` or `daytona`.

## SWE-bench Pro

Clones the repo, checks out the base commit, applies the model's patch, runs the tests
and scores the result.

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

Pass `instance_id=...` instead of the row fields to load that task from the SWE-bench Pro
dataset on the Hugging Face Hub (needs `deepgym[hf]`).

- The model's diff goes first. Every path the hidden test patch touches is put back to its
  state before the diff, and the test patch goes last, so a diff can't edit the hidden
  tests.
- A patch that applies scores 0.3, plus 0.7 times the share of the task's test commands
  that pass; `passed` needs all of them. A diff that is missing (`invalid_patch`) or
  doesn't apply (`patch_apply_failed`) scores 0.
- A task with no test runner (no `test_commands`, and a language other than Python or
  JS/TS) gets `error_type='unsupported_runner'`: no verdict, so trainers mask it.

### Test infrastructure guard

SWE-bench Pro and MiMo tasks check every other path the diff touches before running the
tests. A diff that touches a test or its helper code, conftest, ini or runner config, the
test sections of `pyproject.toml`, `setup.cfg`, `package.json` or Cargo files, build
files that run code (`Makefile`, `pom.xml`, `build.gradle`, `build.rs`), the installed
harness (`node_modules`, `site-packages`, `vendor`, virtualenvs, `.cargo`), or that creates
a symlink or an unusual file mode, scores 0 with `error_type='test_infra_tampering'`. That
is the model's fault, so trainers train on it.

- So that a correct fix never gets a false 0, the guard allows data fixtures under a test
  directory, dependency and version edits, lockfiles, and ordinary files that only share
  a word with a test name.
- An edit the guard can't classify still runs the tests. A failure is the normal 0; a pass
  gets `error_type='validation_uncertain'` with no verdict, so trainers mask it.
  `validation_timeout` and `validation_error` (git couldn't list the paths in time, or
  couldn't find the repository) have no verdict either.
- The guard is a denylist, so it can't be exhaustive, and a diff to the code under test
  can still force the test command to exit 0. The reward is only as strong as the tests.

`deepgym qa` runs a `patch_tampering` attack on self-contained patch tasks such as
`mimo_code`.

## MiMo-V2.6-RL code tasks

[XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)
(Apache-2.0) is the RL data behind MiMo-V2.6. Its code subset has 2,698 repo-level tasks,
each with its own Docker image, hidden test patch and test command.

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

- The sandbox boots from the task's image. The model's diff is applied, every file the
  hidden test patch touches is put back as it was, and then the hidden tests are applied.
  The score is 1.0 if the test command exits 0, else 0.0, as in MiMo's training.
- An image that can't run the test command at all (its repo, runner or toolchain is
  missing, checked before the diff is applied) gives no verdict instead; after the diff,
  every outcome is scored. Code the diff added that stops the test run scores 0 with
  `error_type='policy_sandbox_failure'`.
- A diff can't edit the hidden tests, and the [guard](#test-infrastructure-guard) above
  refuses other test infrastructure; data fixtures the tests read may change. It can still
  change the code the tests run, so the reward is only as strong as the tests.
- The tests run as root in that sandbox because the images expect root's toolchains and
  caches, so the sandbox, not `deepgym_solution`, is what isolates them.
- Images are amd64 and up to about 3.4 GB. Pull them first: a sandbox waits at most 5
  minutes for its image. Docker sandboxes are capped at 2 CPUs and 2 GB, which some
  builds exceed.
- The other subsets aren't imported because their rewards can't be reproduced here: cyber
  grades with a submit script inside each image, general with LLM-judged rubrics against
  mock apps, webdev with a visual grader, and music with MiMo's statistical style scorer.

## Terminal-Bench 2.0

Terminal-Bench 2.0 is a Harbor dataset, so each run works like the Harbor tasks below:
the model's shell commands run in the task's image and its tests grade the result. Its
tests download their tools while grading and its images are amd64-only, so run it on
Daytona.

```python
from deepgym import DeepGym, load_environment

dg = DeepGym(mode='daytona')
env = load_environment('terminal_bench_2')

result = dg.run(
    env,
    model_output='python solve.py --input data.txt > output.txt',
    task_id='log-summary-date-ranges',
)
print(result.score)
```

`env.list_task_ids()` lists the tasks. They load from the Hugging Face Hub, with GitHub as
a fallback.

## Harbor tasks

[Harbor](https://harborframework.com) tasks are folders with `instruction.md`,
`task.toml`, `tests/test.sh` and an image or `environment/Dockerfile`. Import a dataset
from a folder, a git URL or a Hugging Face dataset; `--register` makes the valid tasks
loadable by name:

```bash
deepgym harbor import harborframework/terminal-bench-2.0 --limit 5   # list and validate
deepgym harbor import ./my-tasks --register mine
deepgym qa mine/hello-world --mode docker   # reference solve.sh passes, no-op fails
```

Remote datasets are shallow-cloned once into `~/.deepgym/benchmarks/harbor/` and reused.
Each task is an environment whose model output is a bash script:

```python
from deepgym import DeepGym
from deepgym.harbor import HarborEnvironment, load_tasks

env = HarborEnvironment.from_task(load_tasks('./my-tasks')[0])
result = DeepGym(mode='apple').run(env, 'echo hello > /app/hello.txt')
```

The script runs as an unprivileged user with no network, in a sandbox from the task's
image. The grader is a fresh sandbox from the same image that gets only a copy of the
work directory; `tests/` is uploaded there at grading time and the reward is read from its
`/logs/verifier/` after `test.sh` exits. Nothing the agent writes can reach the tests or
the reward file. This holds on every provider.

- Docker and Apple container also build `environment/Dockerfile` tasks; Modal, E2B and
  Daytona need a prebuilt `docker_image`.
- Tests get the network unless the task says `no-network`, but Docker, Modal and E2B
  sandboxes never have one.
- On Apple container and Daytona the image needs python3 and util-linux to cut the agent
  off the network.
- The agent owns the work directory (`[environment].workdir`, else the image's WORKDIR),
  so it can't be `/`, `/tmp`, `/root` or a system folder, however it is spelled or linked.
- Every command DeepGym runs as root clears the whole image environment at the provider,
  so no image variable (`LD_PRELOAD`, `BASH_ENV`, `GCONV_PATH`, ...) can point root at a
  file the agent wrote. `test.sh` gets only PATH entries that resolve, in the sandbox,
  outside the work directory, plus HOME, the image's locale and TERM, and
  `[verifier].env` from task.toml. The agent keeps the image PATH and locale but no loader
  variable. As in Harbor, `reward.json` wins over `reward.txt`; tests that run the agent's
  code as root trust it.
- `mode='local'` runs on your machine without isolation, for tasks that use relative
  paths only.

Export built-in environments as Harbor tasks. The agent writes `/app/solution.py` as a
non-root user and a separate offline verifier runs the DeepGym verifier on it:

```bash
deepgym harbor export coin_change two_sum --out ./harbor-tasks
```

## CyberGym

DeepGym turns CyberGym Hugging Face metadata and artifacts into CyberBench-style RL tasks
for defensive cyber training: local patch repair, log triage, forensics and synthetic
CTF-style tasks. The reward comes from a deterministic verifier, not from live targets.

The scripts below run from a clone of this repo, in the `deepgym/` folder:

```bash
pip install "deepgym[hf,daytona]"
cp .env.example .env
# edit .env with DAYTONA_API_KEY, ZAI_API_KEY, ZAI_API_BASE, ZAI_MODEL
```

Generate CyberBench seed specs from CyberGym metadata. `--use-zai` lets GLM/Z.ai enrich
the seed plans while keeping the local-only safety constraints:

```bash
python scripts/inspect_cybergym_hf.py --repo-id sunblaze-ucb/cybergym --limit 100
python scripts/generate_cyberbench_seeds.py --count 100 --use-zai
```

For RL, use artifact-backed patch tasks rather than metadata-only prompts. The runner
downloads the vulnerable repository archive and reference patch, applies a model-produced
unified diff in a sandbox, then scores application, touched-file overlap, changed-line
similarity, minimality and safety scope. Overlap and similarity are measured on what the
patch changed on disk, not on its text: changed lines outside a hunk, paths outside the
repository and patches that change nothing score 0.

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
TRL, verl or OpenRLHF. Keep training and evaluation splits separate, use a sandbox for
untrusted completions, and don't train on metadata-only scores. See
[CyberBench](../cyberbench.md) for the full workflow and diagnostic numbers.

## Mix benchmarks behind one reward function

```python
from deepgym import MixedEnvironment, load_environment

mixed = MixedEnvironment(environments=[
    (load_environment('swebench_pro'), 0.6),
    (load_environment('terminal_bench_2'), 0.2),
    (load_environment('coin_change'), 0.2),
])
```

The weights decide what `mixed.sample_batch(n)` draws; each scored completion goes to the
environment its metadata names. See [TRL](../training/trl.md#mixed-benchmarks).

## Hugging Face Hub

Push an environment to a Hub dataset and load it back:

```python
from deepgym.integrations.hf import load_environment_from_hub, push_environment_to_hub

revision = push_environment_to_hub(
    env, repo_id='your-org/deepgym-coin-change', env_name='coin_change'
)
env = load_environment_from_hub('your-org/deepgym-coin-change', revision=revision)
```

A round trip gives back the environment that was pushed: verifier source, support files,
test cases, timeouts, verifier mode, image. Pass `revision` (the commit a push returns);
without it you get whatever `main` holds now, with a warning. Only a plain `Environment`
round-trips; subclasses (repo-patch, benchmark and mixed environments) are refused. Every
environment needs an `env_name` of its own: a push refuses a blank one, and loading by
name or loading all refuses a dataset with a blank or repeated name rather than drop an
environment. Needs `pip install "deepgym[hf]"`.

A Hub dataset is untrusted, so pushes and loads run the same checks and a row can't
change how its environment runs: fields must be known and of the right type, the verifier
is inlined (no host path), there are no `env_vars` (they reach the verifier and the
solution, and may hold credentials), `verifier_mode` must match the verifier, support file
names are plain relative paths, and a custom `image` or `snapshot` loads only with
`trust_image=True`.
