# CLI reference

This page lists every `deepgym` command with its main flags. Run
`deepgym <command> --help` for the full list.

Flags that take a task, verifier or solution (`--task`, `--verifier`, `--solution`) accept
a file path or the text itself. `run`, `run-batch` and `eval` use the default client
(Daytona when `DAYTONA_API_KEY` is set, otherwise local); `qa` and `check` take `--mode`.

## Scoring

| Command | What it does | Main flags |
|---|---|---|
| `deepgym run` | Score one solution against a task and verifier | `--task`, `--verifier`, `--solution` (required); `--timeout` (default 30 s), `--snapshot` |
| `deepgym run-batch` | Score every `.py` file in a folder | `--task`, `--verifier`, `--solutions-dir` (required); `--max-parallel` (default 10), `--timeout`, `--snapshot` |
| `deepgym eval` | Score solutions against a suite of built-in environments | `--suite` (`easy`, `medium`, `hard`, `coding`, `computer-use`, `tool-use`, `all` or a family), `--solutions-dir` with one `<env>.py` per environment; `--max-parallel` (default 100) |

```bash
deepgym run --task task.md --verifier verifier.py --solution solution.py
deepgym eval --suite dynamic-programming --solutions-dir solutions/
```

A run without a verdict prints as `ERROR (<error_type>)` with its cause, and totals show
the judged numbers beside the headline ones, for example
`Pass Rate: 62.00% (judged 70.00%, 12.00% no verdict)`.

## Environments

| Command | What it does | Main flags |
|---|---|---|
| `deepgym new NAME` | Scaffold a working environment folder | `--kind` (`function`, `script`, `repo-patch`, `multi-turn`; required), `--dir` |
| `deepgym check PATH` | Validate a folder and run its reference, empty and stub solutions | `--mode` (default `local`) |
| `deepgym qa [ENV ...]` | Quality-gate registered environments or folders | `--all`, `--mode`, `--reruns` (default 3), `--mutants`, `--model` with `--samples` (default 8), `--allow-local-exec`, `--approve --by NAME`, `--json` |
| `deepgym create` | Print an environment as JSON for `POST /v1/environments` (it isn't saved) | `--name`, `--task`, `--verifier` (required); `--difficulty`, `--domain`, `--tags` |
| `deepgym harbor import SOURCE` | Read and validate Harbor tasks from a folder, git URL or Hugging Face dataset | `--limit N`, `--register NAME` (load them as `NAME/<task>`) |
| `deepgym harbor export ENV ...` | Write environments as Harbor task folders | `--out` (required) |

`check` exits 0 when every check passes, 1 if any fails, and 2 if some are inconclusive.
`qa` exits 0 when the checked environments pass, 1 if any fails, and 2 if none fails but
some are inconclusive or none could be checked. See [new, check, qa](authoring/new-check-qa.md)
and [the QA gate](authoring/qa-gate.md).

## Audits

| Command | What it does | Main flags |
|---|---|---|
| `deepgym audit` | Run the reward-hack attacks against a verifier, locally | `--task`, `--verifier` (required); `--strategies`, `--benchmark`, `--verifier-id`, `--persist`, `--db-path`, `--json` |
| `deepgym benchmark-audit` | Check a folder of environments for split leakage and holdouts | `--env-dir` (required); `--benchmark`, `--seed`, `--public-eval-ratio`, `--holdout-ratio`, `--canary-ratio`, `--split ENV=SPLIT`, `--json` |

## Traces

| Command | What it does | Main flags |
|---|---|---|
| `deepgym traces export` | Export recorded traces as training data | `--format` (`jsonl`, `parquet`, `atif`) and `--out` (required); `--rows` (`trace` or `rl`), `--only-valid`, `--db`, `--traces` |

See [traces](traces.md).

## Servers

| Command | What it does | Main flags |
|---|---|---|
| `deepgym serve` | Start the HTTP API | `--host` (default `127.0.0.1`), `--port` (default 8000), `--allow-local-exec`, `--no-auth`, `--reload` |
| `deepgym console` | Launch the local web console (alias `deepgym web`) | `--host` (default `127.0.0.1`), `--port` (default 8080), `--allow-local-exec` |

Both need `pip install "deepgym[server]"`. See [the API server](api-server.md) and
[the console](console.md).
