# Console

This page covers `deepgym console`, the local web console for the environments, runs,
episodes, sandboxes and QA reports on your machine: what each page shows, where data is
stored, and how the console protects itself.

```bash
pip install "deepgym[server]"
deepgym console --port 8080 --allow-local-exec   # http://127.0.0.1:8080
```

`deepgym web` is an alias.

## Pages

- **Environments:** search and filter the registry; open one to read its task, verifier,
  reference solution and metadata.
- **Runs:** score a pasted solution or the reference against one environment or a small
  eval, on any provider (`local`, `docker`, `apple`, `daytona`, `modal`, `e2b`). Runs
  execute in the background; the run page shows per-case results and a diff against the
  reference. A run without a verdict keeps no score and counts as an error, and the totals
  show a "No verdict" count with the judged pass rate and score beside the headline ones.
- **Episodes:** replay recorded trajectories step by step, with screenshots for
  computer-use tasks. Record one from your agent loop:

  ```python
  from deepgym.console import record_episode

  trajectory, result = dg.run_episode(env, agent)
  record_episode(trajectory, result, environment='todo_list')
  ```

  `record_episode(tr.trace)` stores a [trace](traces.md) the same way, and keeps it for
  `deepgym traces export`.
- **Sandboxes:** live `deepgym-*` containers from Docker and Apple `container`, with kill.
  Cloud sandboxes are not listed.
- **QA:** the reports `deepgym qa` writes to `~/.deepgym/qa/` (or `$DEEPGYM_QA_DIR`).

## Storage and execution

Runs and episodes are stored in `~/.deepgym/console.db`. Reference solutions may run
locally, but a pasted solution only runs on your machine with `--allow-local-exec` (or
`DEEPGYM_ALLOW_LOCAL_EXEC=true`); otherwise pick a sandbox provider.

## Security

The console has no login, so other websites can't drive it:

- Every `/api` call needs the per-launch token the console writes into its pages, sent as
  `X-DeepGym-Token`.
- Writes from other origins are refused, and so are write bodies that aren't JSON.
- Bound to `127.0.0.1` or `localhost` (the default), it only answers requests addressed to
  localhost. Bound to any other host, it doesn't check the Host header.

## Building the UI

The UI lives in `console-ui/` (Next.js static export, built with bun) and ships prebuilt in
wheels. From a clone, build it once with `cd console-ui && bun install && bun run build`.
