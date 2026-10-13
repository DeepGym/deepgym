# API server

This page covers `deepgym serve`, the HTTP API for scoring from any language: starting it,
authentication, the endpoints, and the limits it enforces.

## Start it

```bash
pip install "deepgym[server]"

# Dev server
DEEPGYM_NO_AUTH=true deepgym serve --host 127.0.0.1 --port 8000 --allow-local-exec

# Production server
DEEPGYM_API_KEY=your-key DAYTONA_API_KEY=your-key deepgym serve --port 8000
```

- The server won't start without auth configured: set `DEEPGYM_API_KEY` for production,
  or `DEEPGYM_NO_AUTH=true` (or `--no-auth`) for local development. Clients send the key
  in the `X-API-Key` header.
- It runs code on Daytona when `DAYTONA_API_KEY` is set; a Daytona that fails to start is
  fatal rather than a silent fall back. Without a key it runs code on the host, so it needs
  `--allow-local-exec` (or `DEEPGYM_ALLOW_LOCAL_EXEC=true`).
- `GET /health` needs no key. The OpenAPI schema is served at `/docs`.

From Python, `AsyncDeepGym(api_url='http://host:8000', api_key=...)` sends its `run`,
`run_batch` and `eval` calls to the server.

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | Status, package version and `leaked_sandboxes` |
| `GET /v1/capabilities` | What this server supports |
| `POST /v1/run` | Score one output against an inline environment |
| `POST /v1/run/batch` | Score many outputs against one environment |
| `POST /v1/eval` | Score outputs against a suite |
| `POST /v1/environments`, `GET /v1/environments`, `GET /v1/environments/{id}` | Register and list environments |
| `POST /v1/environments/{id}/run` | Score one output against a registered environment |
| `POST /v1/audit/verifier`, `POST /v1/environments/{id}/audit` | Audit a verifier for reward hacking |
| `POST /v1/benchmarks/audit` | Audit a benchmark for split leakage |
| `POST /v1/jobs/run`, `POST /v1/jobs/batch` | Queue a run or a batch; returns a job id |
| `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel` | Poll or cancel a job |
| `POST /v1/sessions` and `/v1/sessions/{id}/step`, `/score`, `DELETE /v1/sessions/{id}` | Live multi-turn sessions; see [sessions](environments/sessions.md) |

Registering an environment takes the same fields as an inline run. An invalid environment
is a 422 before anything is queued, and the API refuses caller-chosen images.

## Limits

- Request bodies over 32 MiB get 413 before they are read.
- All requests and jobs share one limit of 100 concurrent runs, so `max_parallel` in a
  request splits the server's slots and never adds to them.
- Jobs are `queued` until a job thread is free, then `running`. They count against the
  job limit only while queued or running, and their request bodies may take at most
  256 MiB in all (429 above).
- Finished jobs stay readable up to the newest 10,000 or 256 MiB of results.
- A cancelled job stays cancelled: a queued one is dropped, a cancelled batch starts none
  of its remaining outputs, and runs already going finish within their timeout with their
  results dropped.
- At most 100 sessions run at once; idle ones close after 30 minutes, and every session
  closes on shutdown.
