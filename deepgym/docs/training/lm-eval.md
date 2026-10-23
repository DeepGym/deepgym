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

# max_gen_toks: the model's maximum output (Qwen2.5 generates up to 8K tokens).
tasks = [
    make_lm_eval_task(load_environment(n), n, max_gen_toks=8192)
    for n in ['coin_change', 'two_sum']
]
results = lm_eval.simple_evaluate(
    model='hf', model_args='pretrained=Qwen/Qwen2.5-0.5B-Instruct', tasks=tasks
)
```

- Each environment becomes one task named `deepgym_<name>`, whose one document is the
  environment's prompt.
- lm-eval has no registry for Python tasks, so pass the task objects; `lm_eval --tasks`
  can't name them.
- The last fenced code block of the generation runs, or all of it if there is none.
- Generation runs to the model's maximum output, which you pass as `max_gen_toks`: lm-eval
  needs a number, its backends would otherwise cut every answer at 256, and only you know
  your model's real maximum, so there is no default. lm-eval gives the scorer only the
  text, so a generation that reached the maximum is scored as written.
- Pass `dg=` to pick the client that runs the verifier; without one, the task uses
  `DeepGym(mode='auto')`.

## Metrics

Each task has one document, so its metrics describe one run:

| Metric | Meaning |
|---|---|
| `deepgym_score` | The score (0.0 to 1.0), or NaN without a verdict. |
| `deepgym_pass` | 1.0 if it passed, 0.0 if not, NaN without a verdict. |
| `deepgym_verdict` | 1.0 if it got a verdict, else 0.0. |

For a number across environments, average each metric over the tasks, skipping NaN. A
sandbox or verifier failure then counts in neither score nor pass, so it doesn't read as
a wrong answer; it lowers the mean `deepgym_verdict` instead. See
[the reward contract](reward-contract.md).
