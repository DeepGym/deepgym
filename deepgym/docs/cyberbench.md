# CyberBench-style DeepGym environments

DeepGym cyber tasks should be **sandbox-first, deterministic, and reward-shaped**.
Use live models such as GLM-5.1 only as candidate policies; rewards come from
local verifiers.

## Safe scope

Allowed benchmark shapes:

- defensive CVE-style log triage and incident response;
- CTF-style exploitation of synthetic local services;
- vulnerable-repository patch tasks;
- malware-free forensics over provided artifacts;
- local-only terminal tasks with explicit flags and constraints.

Do not point tasks at public targets, real credentials, persistence, evasion,
malware behavior, or uncontrolled network exploitation.

## Built-in layout

```text
src/deepgym/envs/cyber/<task_id>/
  task.md
  verifier.py
  metadata.json
  reference_solution.py
```

Register each task in `src/deepgym/envs/registry.json` with
`domain: "cyber"` and `family: "cyberbench"`.  Then it works with:

```python
from deepgym import DeepGym, load_environment, load_suite

dg = DeepGym(mode='local')
env = load_environment('cyber/cve_2021_44228_log_triage')
cyber_suite = load_suite('cyberbench')
```

## Z.ai / GLM smoke testing

Create a local untracked `.env` from `.env.example` and add rotated keys:

```bash
cp .env.example .env
# edit .env locally; never commit it
python scripts/smoke_zai_deepgym.py --env cyber/cve_2021_44228_log_triage
```

Relevant variables:

```bash
DAYTONA_API_KEY=...
ZAI_API_KEY=...
ZAI_API_BASE=https://api.z.ai/api/paas/v4
ZAI_MODEL=glm-5.1
```

## Quality bar for new CVE-style tasks

1. Use synthetic/local artifacts or intentionally vulnerable local targets.
2. Include shaped reward components, not only pass/fail.
3. Include `cases` for per-check feedback.
4. Use train/eval/holdout variants before training.
5. Keep a reference solution for smoke tests.
6. Add canary details that are not copied from public exploit writeups.

## CyberGym HF bridge

The seed-generation bridge defaults to `sunblaze-ucb/cybergym`, which exposes a
`tasks` split with metadata fields such as `task_id`, `project_name`,
`project_language`, `project_main_repo`, `vulnerability_description`, and
`task_difficulty`. The bridge converts those rows into safe DeepGym seed specs.

```bash
python scripts/inspect_cybergym_hf.py --repo-id sunblaze-ucb/cybergym --limit 100
python scripts/generate_cyberbench_seeds.py --repo-id sunblaze-ucb/cybergym --count 100
```

To ask GLM-5.1 to enrich each seed spec while preserving safe/local constraints:

```bash
python scripts/generate_cyberbench_seeds.py --count 100 --use-zai
```

The generated JSONL is written to `data/cyberbench/seed_specs.jsonl` by default.
Materialization into runnable DeepGym environments is a separate step: each seed
needs a synthetic target, a deterministic verifier, and a reference solution.

## Artifact-backed CyberGym patch evaluation

For real RL diagnostics, use artifact-backed patch tasks instead of metadata-only
prompts. The runner downloads `repo-vul.tar.gz` and `patch.diff`, creates a
`CyberGymPatchEnvironment`, applies a candidate unified diff inside DeepGym, and
returns shaped rewards:

- `apply` — candidate patch applies to the extracted vulnerable repo;
- `file_overlap` — candidate touches files expected by the reference patch;
- `line_similarity` — changed-line similarity to the reference repair;
- `minimality` — candidate is not much broader than the reference patch;
- `safety` — candidate avoids obviously unsafe payload patterns.

Validate top-100 importer/verifier behavior locally:

```bash
python scripts/run_cybergym_artifact_eval.py --count 100 --mode local --max-parallel 12
```

Run through Daytona for sandbox-backed execution:

```bash
python scripts/run_cybergym_artifact_eval.py --count 100 --mode daytona --max-parallel 10
```

Use GLM patch attempts when rate limits allow:

```bash
python scripts/run_cybergym_artifact_eval.py --count 20 --answer-source glm --fallback-to-reference
```

Current diagnostic from the first 100 CyberGym rows with reference patches:

- local artifact verifier: 69/100 pass, average score 0.69;
- Daytona smoke: 4/5 pass, average score 0.80.

The remaining failures are useful: they identify patch application edge cases
where the pure-Python patch applier or CyberGym archive alignment needs more
work before this becomes a production RL benchmark.
