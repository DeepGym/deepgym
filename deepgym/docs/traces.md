# Traces

This page covers recording graded agent episodes as traces with `dg.trace()`, what a
trace holds, and exporting traces as training data in JSONL, Parquet (verl and TRL rows)
or ATIF.

## Record an episode

`dg.trace()` records a multi-turn episode while it runs, as training data. It works like
[`dg.session()`](environments/sessions.md), and each step also takes the model messages
behind the action:

```python
from deepgym import load_environment
from deepgym.traces import AgentMeta

agent = AgentMeta(name='react-shell', model='Qwen/Qwen3-8B', sampling={'temperature': 0.7})
env = load_environment('todo_list', variant=3)
with dg.trace(env, agent, variant=3) as tr:   # variant here only labels the provenance
    obs = tr.start()
    while not tr.done:
        action, messages = my_agent(obs)   # messages: this turn's new TraceMessages
        obs = tr.step(action, messages).observation
    tr.score()
print(tr.trace.has_verdict)
```

- `TraceMessage(role=..., content=..., token_ids=..., logprobs=...)` holds one model
  message; token ids and logprobs are optional.
- When one model reply issues several actions, step each one and pass the reply's
  messages with the first only; the others belong to the same turn (`trace.turns`).
- A traced episode refuses actions after `score()`, so its verdict always matches the
  state it ends in.
- `variant`, `seed` and `reference` (the reference solution's source) only go into the
  provenance; load the environment with the same variant yourself.

## What a trace holds

- **Provenance:** environment name, variant and seed; SHA-256 hashes of the environment
  definition, task, verifiers and reference; the image the sandbox booted with the digest
  its provider resolved, or why there is none (`digest_unavailable`); the deepgym version
  and the git commit of the deepgym checkout (null for an installed package), with whether
  it had local changes; the provider; and the agent, model and sampling config you pass.
- **Every step:** the action, the model messages behind it with optional token ids and
  logprobs, what came back, the step reward with the step verifier's output, exit code
  and error.
- **Every result of `score()`:** the last is the verdict, with `has_verdict`, and earlier
  ones stay in `prior_results`.

Traces use a versioned, frozen schema (`deepgym.traces.Trace`, schema 1.1); schema 1.0
traces still load and export. Docker and Apple container report the image digest from the
container, and Daytona from its snapshot; images Daytona builds, Modal, E2B and the local
backend give a `digest_unavailable` reason instead.

## On disk

Each event is appended to `~/.deepgym/traces/<trace_id>.jsonl` (or
`$DEEPGYM_TRACES_DIR`) and flushed as it happens, so an exception or a killed process
keeps the steps taken so far, and a trace that was never scored is `incomplete`.
`deepgym.traces.load_trace(path)` reads one back; `path` is a `pathlib.Path`.

`record_episode(tr.trace)` from `deepgym.console` also stores it in the
[console](console.md), where episodes can be replayed step by step.

## Export

```bash
deepgym traces export --format jsonl --out traces.jsonl             # whole traces
deepgym traces export --format parquet --rows rl --out rl.parquet   # verl / TRL rows
deepgym traces export --format atif --out atif/ --only-valid        # ATIF documents
```

- Export reads the console database and the traces directory, or the sources named with
  `--db` and `--traces`. A trace stored in both is exported once, and refused if the
  copies disagree.
- A trace without a verdict is exported with `has_verdict: false`, or left out with
  `--only-valid`. `--rows rl` rows and trace-shaped Parquet rows give it a null reward;
  whole-trace JSONL and ATIF (under `extra.deepgym`) keep the result as recorded, with
  its placeholder score of 0.0.
- `--rows rl` writes one conversation per trace: `messages`, `reward`, `step_rewards`,
  and `steps` (every action with its observation text and screenshot, done flag, exit
  code, error, step reward and the index of the reply that produced it). It skips traces
  with a turn that has no model messages.
- `--format atif` writes one ATIF v1.7 (Agent Trajectory Interchange Format) document per
  trace, with one agent step per turn and a tool call and result per action. It refuses a
  non-empty directory, so stale documents never mix in. The mapping is documented in the
  `deepgym.traces.atif` module.
- Exports are strict JSON: a NaN in verifier output becomes null.
- Parquet needs `pip install "deepgym[traces]"`.

From Python, `deepgym.traces.export_traces(traces, fmt, out, rows=..., only_valid=...)`
does the same for a list of traces; `out` is a `pathlib.Path`.
