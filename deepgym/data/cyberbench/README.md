# CyberBench seed data

`seed_specs.jsonl` contains safe DeepGym CyberBench-RL task plans generated from
CyberGym-style vulnerability metadata. These are curriculum seeds, not final
weaponized exploit tasks.

Regenerate deterministic seeds from Hugging Face metadata:

```bash
python scripts/generate_cyberbench_seeds.py --count 100
```

Use Z.ai/GLM enrichment after setting a rotated local key in `.env`:

```bash
python scripts/generate_cyberbench_seeds.py --count 100 --use-zai
```

Inspect CyberGym source distribution:

```bash
python scripts/inspect_cybergym_hf.py --repo-id sunblaze-ucb/cybergym --limit 200
```

## Artifact-backed evaluation outputs

- `top100_artifact_eval_local.jsonl` — first 100 CyberGym rows evaluated locally
  with reference patches through the artifact-backed verifier.
- `top100_artifact_eval_local.summary.json` — summary of local artifact-backed
  verifier coverage.
- `artifact_eval_daytona_5.jsonl` — Daytona smoke run for artifact-backed patch
  tasks.
