# Traces

This page covers recording graded agent episodes as traces with `dg.trace()`, what a
trace holds, where traces are kept on disk, and exporting traces as training data in
JSONL, Parquet (verl and TRL rows) or ATIF. Comparing two runs task by task, scanning
traces for what the verifier didn't catch and building versioned datasets from traces
have their own pages: [Compare two runs](compare.md), [Scan traces](scanners.md) and
[Datasets](datasets.md).

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
- `tr.step(action, messages, usage=Usage(...), cost_usd=...)` also records the provider
  token usage of the model calls since the previous step (`deepgym.traces.Usage`: input,
  output, cache-read, cache-write and reasoning tokens, reasoning being part of output)
  and, if you price it, what it cost in USD. Like messages, pass a reply's usage with its
  first action only, so totals count it once. Under `dg.trace()` the agents in
  `deepgym.agents` do this for you. Their costs come from the `TokenPrices` you give them,
  or, for a CLI run, the CLI's own reported total (Claude Code's `total_cost_usd`, which
  takes precedence over `TokenPrices`); without either, the cost is null, never a guess.
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
  logprobs, the provider token usage and USD cost of the model calls behind it, what came
  back, the step reward with the step verifier's output, exit code and error.
- **Usage outside any step:** a model call that led to no action (a closing answer, a
  refusal, a reply cut off, or one after which a budget ran out) is on no step, and nor is
  a CLI agent's run, whose one step is recorded before the CLI reports its usage. Their
  usage and cost are in `trace.unattached_usage` and `trace.unattached_cost_usd`.
  `tr.record_unattached_usage(usage, cost_usd)` records it; the agents do when they finish,
  including when an error ends the run.
- **Incomplete usage:** `trace.usage_complete` is true when no model call was flagged
  incomplete. It is false when some call reported no usage, or only part of it (a failed
  request, a response cut off, an error that ended the run mid-call, an interrupted CLI
  run). Unknown usage is never counted as 0 tokens: the token totals then count only what
  was reported, a lower bound, and the cost total is null. `tr.mark_usage_incomplete()`
  flags it.
- **Usage totals:** `trace.usage` and `trace.cost_usd` sum the steps and the unattached
  usage, and are null when nothing reported usage, so unknown never reads as 0; the cost
  is also null when any part of the usage has no cost. For the agents in
  `deepgym.agents` they match `AgentResult.usage` and `AgentResult.cost` up to float
  rounding (the trace sums costs exactly, the agent adds them one by one), with two
  exceptions. When nothing reported usage, `AgentResult.usage` has zero tokens while the
  trace's totals are null; `AgentResult.cost` is $0 only if no call was made (a run whose
  only request failed has a null cost in both). And an incomplete CLI run keeps its
  partial CLI cost in `AgentResult.cost`, labelled incomplete, while the trace's cost
  total is null.
- **Every result of `score()`:** the last is the verdict, with `has_verdict`, and earlier
  ones stay in `prior_results`.

Traces use a versioned, frozen schema (`deepgym.traces.Trace`, schema 1.2); schema 1.0
and 1.1 traces still load and export, without usage. Docker and Apple container report
the image digest from the container, and Daytona from its snapshot; images Daytona
builds, Modal, E2B and the local backend give a `digest_unavailable` reason instead.

## On disk

Each event is appended to `~/.deepgym/traces/<trace_id>.jsonl` (or
`$DEEPGYM_TRACES_DIR`) and flushed as it happens, so an exception or a killed process
keeps the steps taken so far, and a trace that was never scored is `incomplete`.
`deepgym.traces.load_trace(path)` reads one back; `path` is a `pathlib.Path`.

`record_episode(tr.trace)` from `deepgym.console` also stores it in the
[console](console.md), where episodes can be replayed step by step. Scores and expert
marks on a trace or one of its steps are stored as [labels](labels.md).

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
  the unattached usage and cost, the `usage` and `cost_usd` totals, `usage_complete`, and
  `steps` (every action with its observation text and screenshot, done flag, exit code,
  error, step reward, usage, cost and the index of the reply that produced it). It skips
  traces with a turn that has no model messages.
- Trace-shaped Parquet rows have `usage`, `cost_usd` and `usage_complete` columns.
- `--format atif` writes one ATIF v1.7 (Agent Trajectory Interchange Format) document per
  trace, with one agent step per turn and a tool call and result per action. A turn's
  usage goes in its agent step `metrics`, whose `llm_call_count` is the number of replies
  in the turn (null for a turn with no recorded messages, as a computer agent's), and the
  totals in `final_metrics`; when some usage is on no step, the root `notes` say
  `final_metrics` include it. ATIF's prompt tokens count every input token, cached and cache-written ones
  too, and cache writes and reasoning tokens, which ATIF has no field for, go in
  `metrics.extra`. It refuses a non-empty directory, so stale documents never mix in. The
  mapping is documented in the `deepgym.traces.atif` module.
- Exports are strict JSON: a NaN in verifier output becomes null.
- Parquet needs `pip install "deepgym[traces]"`.

From Python, `deepgym.traces.export_traces(traces, fmt, out, rows=..., only_valid=...)`
does the same for a list of traces; `out` is a `pathlib.Path`.
