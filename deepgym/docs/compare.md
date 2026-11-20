# Compare two runs

This page covers comparing two runs of the same tasks from their recorded
[traces](traces.md): how attempts are keyed per task, which tasks improved or regressed,
pass@k and pass^k per side, and the significance test.

`deepgym.compare` compares two runs of the same tasks, such as model A against model B or
one checkpoint against another:

```python
from deepgym.compare import attempts_from_traces, compare

report = compare(attempts_from_traces(traces_a), attempts_from_traces(traces_b))
print(report.flips, report.significance)
for task in report.tasks:
    print(task.key, task.status, task.flip, task.delta)
```

- Each side is a list of `Attempt(key, score, has_verdict, solved, error_type,
  fingerprint)`, one per graded sample; solved means the verifier passed it (`passed`)
  with a verdict, as eval pass rates count it, so a rubric pass with a partial score is
  solved.
- `attempts_from_traces` keys a trace by `json.dumps([name, variant])` of its environment,
  e.g. `["todo_list", 3]`, with variant 0 when the trace recorded none (`dg.trace`
  records none unless you pass one; `load_environment` loads variant 0); an unnamed
  environment adds its task hash, `[null, variant, "<task sha256>"]`. The seed is not part
  of the key, so traces that differ only in seed are attempts at one task. The
  fingerprint is the environment hash, which covers the task and the verifiers, so a
  trace that ran another variant than its key says is still caught. Two traces with the
  same id are refused.
- Per task: each side's judged attempts, solves, solve rate (the task's pass@1) and mean
  score, B's change in both, and a flip: `improved` (every judged attempt failed on A and
  passed on B), `regressed` (the reverse), `both_solved`, `both_failed`, or `mixed` (a
  side solved only some of its judged attempts).
- Overall: the tasks on one side only (`only_a`, `only_b`), counts per flip, and per side
  the mean pass@k and pass^k with band counts over the compared tasks
  (`deepgym.passk.summarize`), for k = 1, each power of 2 below `k_limit`, and `k_limit`:
  the fewest judged attempts a compared task has on either side, set by the tasks in
  `k_limit_tasks`.
- `significance` is a two-sided p-value from a stratified permutation test (the exact
  conditional test behind Cochran-Mantel-Haenszel): if A and B were the same model, each
  task's judged attempts would be interchangeable between the sides, so the test takes B's
  total solves against every way of dealing each task's attempts between A and B. It
  holds with different attempt counts per task or side, and with one judged attempt per
  task and side it equals exact McNemar. It is computed in floating point (164 tasks
  sampled 100 times a side take well under a second): accurate to float precision down to
  about 1e-290, drifting below that, and read as 0 only once it is under about 1e-300.
- An attempt without a verdict is left out of every rate and counted per side by error
  type (`no_verdict`). A task whose attempts on one side all lack a verdict is
  `uncomparable`, not a fail. A task whose environment hash on A differs from B's is
  `changed`, and one with more than one hash on a side is `ambiguous`. None of these
  counts in the flips, the rates or the test. An attempt without a hash is no change, but
  its compared task is listed in `unverified`.
- The environment hash counts files by content, not path: a verifier read from a file by
  its source, and a Harbor task by its tests, its solution and, when it builds its image,
  its `environment/` folder: every entry's path, type and mode (all but owner and group write), each file's content, each
  link's target (not followed) and every folder, empty ones included. The files under a
  Terminal-Bench `task_root` and CyberGym `artifacts` still count by path only.

The [console](console.md) compares two run groups the same way.
