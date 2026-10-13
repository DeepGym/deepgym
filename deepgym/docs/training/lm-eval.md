# lm-evaluation-harness

This page covers evaluating a model on DeepGym environments with lm-evaluation-harness
(0.4.9 or later): building the tasks, what runs, and the metrics.

```bash
pip install "deepgym[lm_eval]"
```

```python
import lm_eval
from deepgym import load_environment
from deepgym.integrations.lm_eval import make_lm_eval_task

tasks = [make_lm_eval_task(load_environment(n), n) for n in ['coin_change', 'two_sum']]
results = lm_eval.simple_evaluate(
    model='hf', model_args='pretrained=Qwen/Qwen2.5-0.5B-Instruct', tasks=tasks
)
```

- Each environment becomes one task named `deepgym_<name>`, whose one document is the
  environment's prompt.
- lm-eval has no registry for Python tasks, so pass the task objects; `lm_eval --tasks`
  can't name them.
- The last fenced code block of the generation runs, or all of it if there is none.
- Pass `dg=` to pick the client that runs the verifier; without one, the task uses
  `DeepGym(mode='auto')`.

## Metrics

| Metric | Meaning |
|---|---|
| `deepgym_score` | Mean score (0.0 to 1.0) over the environments that got a verdict. |
| `deepgym_pass` | Share of those that passed. |
| `deepgym_verdict` | Share of environments that got a verdict at all. |

A sandbox or verifier failure counts in neither score nor pass, so it doesn't read as a
wrong answer; it lowers `deepgym_verdict` instead. See
[the reward contract](reward-contract.md).
