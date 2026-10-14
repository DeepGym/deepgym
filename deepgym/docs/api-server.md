# API server

This page covers `deepgym serve`, the HTTP API for scoring from any language: starting it,
authentication, the endpoints, and the limits it enforces.

## Start it

```bash
pip install "deepgym[server]"

# Dev server
DEEPGYM_NO_AUTH=true deepgym serve --host 127.0.0.1 --port 8000 --allow-local-exec

# Production server
DEEPGYM_API_KEY=your-key DAYTONA_API_KEY=your-key deepgym serve --host 0.0.0.0 --port 8000
```

- The server won't start without auth configured: set `DEEPGYM_API_KEY` for production,
  or `DEEPGYM_NO_AUTH=true` (or `--no-auth`) for local development. Clients send the key
  in the `X-API-Key` header.
- It runs code on Daytona when `DAYTONA_API_KEY` is set; a Daytona that fails to start is
  fatal rather than a silent fall back. Without a key it runs code on the host, so it needs
  `--allow-local-exec` (or `DEEPGYM_ALLOW_LOCAL_EXEC=true`).
- `--host` defaults to `127.0.0.1`; bind `0.0.0.0` to serve other machines.
- `GET /health` needs no key. Interactive docs are at `/docs` and the OpenAPI schema at
  `/openapi.json`.

From Python, `AsyncDeepGym(api_url='http://host:8000', api_key=...)` sends each run to
the server as one `POST /v1/run`. Its `run_batch` and `eval` fan out on the client, one
request per output; they don't call `/v1/run/batch` or `/v1/eval`.

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
is a 422 before anything is queued. Environments sent to the run, batch, job and
environment endpoints have no `image` or `snapshot` field, and one sent anyway is
dropped. A session request whose inline environment sets an `image` gets 422; a
`snapshot` is accepted.

## Limits

- Request bodies over 32 MiB get 413: before they are read when `Content-Length` says so,
  else once the first 32 MiB have been read.
- All requests and jobs share one limit of 100 concurrent runs, so `max_parallel` in a
  request splits the server's slots and never adds to them.
- Jobs are `queued` until a job thread is free, then `running`. They count against the
  job limit only while queued or running, and their request bodies may take at most
  256 MiB in all (429 above).
- Finished jobs stay readable up to the newest 10,000 or 256 MiB of results.
- A cancelled job stays cancelled: a queued one is dropped, a cancelled batch starts none
  of its remaining outputs, and runs already going finish within their timeout with their
  results dropped.
- At most 100 sessions run at once; idle ones close after 30 minutes, and the server
  tries to close every session on shutdown.
