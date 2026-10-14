# Sandboxes

This page covers where DeepGym runs answers: the providers, how to set each one up, the
isolation the sandbox modes give, and per-environment images.

## Providers

Pick a provider with `mode`. It works the same for `DeepGym`, `AsyncDeepGym` and
`dg.session()`:

```python
from deepgym import DeepGym

dg = DeepGym(mode='docker')
```

| `mode` | Runs on | Setup |
|---|---|---|
| `local` | A subprocess on this machine, under `sandbox-exec` (macOS) or `bwrap` (Linux) | None |
| `docker` | A throwaway container with no network | A running Docker daemon |
| `apple` | A throwaway Linux VM from Apple's [container](https://github.com/apple/container) CLI (macOS) | `brew install container`, then `container system start` |
| `daytona` | A Daytona sandbox, cloud or self-hosted | `pip install "deepgym[daytona]"`, then `DAYTONA_API_KEY` (see below) |
| `modal` | A Modal sandbox with no network | `pip install "deepgym[modal]"`, then `modal token set` |
| `e2b` | An E2B sandbox with no network | `pip install "deepgym[e2b]"`, then `E2B_API_KEY` |
| `auto` (default) | Daytona when `DAYTONA_API_KEY` is set, otherwise `local` | |

- `auto` also falls back to `local`, with only a warning, if `DAYTONA_API_KEY` is set but
  `deepgym[daytona]` isn't installed. Pass `mode='daytona'` to make that an error.
- `local` is for development. Repo-patch, Harbor (terminal) and CyberGym artifact tasks
  refuse it and raise `SandboxError`, and a local session runs the agent's actions without
  isolation.
- Desktop (computer-use) environments need Daytona.
- A provider that isn't ready fails fast: the Docker or Apple CLI not running, or the
  provider's SDK not installed.

### Daytona

Cloud: get a key from [app.daytona.io](https://app.daytona.io) and set `DAYTONA_API_KEY`.

Self-hosted:

```bash
git clone https://github.com/daytonaio/daytona
cd daytona
docker compose -f docker/docker-compose.yaml up -d
# Set DAYTONA_API_URL and DAYTONA_API_KEY for this instance
```

### Docker and Apple container

Each run or session gets a throwaway container (Docker) or Linux VM (Apple container),
with no API key. Docker containers share the host's kernel, so Docker isolates less than
the VM-based providers (Apple container, E2B, Daytona).

- The default image is `python:3.12-slim`; override it with `DEEPGYM_CONTAINER_IMAGE` or
  per environment (below).
- Each sandbox is capped at 2 CPUs and 2 GB, and Docker also at 512 processes.
- Sessions run offline; servers inside the sandbox still work on localhost.
- Stop Apple's service with `container system stop`.

## Isolation

Every sandbox mode applies the boundaries below, for single runs and shell sessions. Use
one for anything shared or untrusted.

- **Single runs:** an answer run through the `deepgym_solution` helpers runs as a separate
  Linux user that can't read the judge's files, touch the judge's process, or write to its
  output, with no network. Repo-patch tests (SWE-bench Pro, MiMo) run the patched code as
  root inside the sandbox. Only plain
  data crosses back to the verifier (see [verifiers](authoring/verifiers.md)).
- **Sessions:** agent commands run as an unprivileged user in a loopback-only network
  namespace, and verifiers as another user, from a file only that user can read (see
  [sessions](environments/sessions.md)). Setup refuses an image where anyone but root can
  change a folder on root's PATH (`/usr/local/sbin` to `/bin`), a folder above one, or a
  program in one.
- **Network:** Docker, Modal and E2B take the whole sandbox offline. Apple container
  doesn't block egress, and Daytona, though it blocks the network, serves its toolbox API
  (run any command as its sudoer) on loopback, so on both DeepGym gives untrusted code its
  own offline network namespace; that needs util-linux in the image, and a judge run fails
  rather than run online without it.
- **Locally:** the answer runs under `sandbox-exec` (macOS) or `bwrap` (Linux), with the
  judge directory, the deepgym package, your home directory and `/run` (daemon sockets)
  hidden. If neither is available, you still get process separation and a warning.
  `DeepGym(mode='local', confine_local_judge=True)` also runs each whole judge inside the
  local sandbox, for verifiers that might run code in-process; `deepgym qa --mutants`
  uses it. Sandboxes don't nest, so the answer then runs as plain child processes that
  can read the judge's files: use it only for code that isn't adversarial, such as
  mutants of a reference.

Each provider is one folder in `src/deepgym/sandboxes/` implementing a small `Sandbox`
protocol (run a command as root, upload a file, delete). Adding one also means
registering it in `src/deepgym/sandboxes/__init__.py` and the console's `RunMode`.

## Per-environment images

Set `image` on an environment, or in its `metadata.json`, to boot its sandbox from an image
with the task's dependencies already installed:

```python
from deepgym import Environment

env = Environment(task=..., verifier_code=..., image='python:3.11-slim')
```

- Docker, Apple container, Modal and Daytona pull it from a registry; E2B treats it as a
  template id. On Daytona a `snapshot` wins.
- The image must run as root and have python3, bash, useradd and util-linux (setpriv,
  unshare, nsenter, mount). The official `python` images do; plain `debian` and `ubuntu`
  images lack python3, and Alpine lacks more.
- Root in the sandbox runs the image's binaries, so only use images you trust. What the
  HTTP API accepts is in [the API server](api-server.md) docs.
- Docker and Apple container can also build a Dockerfile, as Harbor tasks with an
  `environment/Dockerfile` use (see [importers](environments/importers.md#harbor-tasks)).
