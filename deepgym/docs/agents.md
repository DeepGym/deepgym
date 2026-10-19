# Agent harnesses

`deepgym.agents` supplies `AnthropicComputerAgent`, `OpenAIComputerAgent`,
`AnthropicToolAgent`, `OpenAIToolAgent`, `ClaudeCodeAgent`, and `CodexAgent`. Each implements
`run(session, initial, *, task, budget) -> AgentResult`. Start the Session first;
the caller owns grading and cleanup. `DeepGym.run_episode(env, agent, budget=...)`
does that lifecycle and returns `(AgentResult, RunResult)`. Its existing callable
interface still returns `(Trajectory, RunResult)`.

Computer agents need the `anthropic` or `openai` optional extra, respectively.
Imports stay lazy. Choose a model supporting Anthropic's
`computer_toolset_20260801` or OpenAI's Responses `computer` tool. Provider keys stay
in the host process for these two agents. Unsupported mouse buttons, modifiers,
curved drag paths, and other unrepresentable actions produce explicit tool errors.
Pending OpenAI safety checks stop the run without executing their actions.

```python
from deepgym import DeepGym
from deepgym.agents import AnthropicComputerAgent, Budget

# env is an already-defined ComputerUseEnvironment.
result, grade = DeepGym(mode='daytona').run_episode(
    env,
    AnthropicComputerAgent(model='your-computer-capable-model'),
    budget=Budget(max_turns=30, max_seconds=300),
)
```

`Budget` always supplies finite positive turns and wall time. For computer agents,
a turn is one provider request. For Claude Code it is a unique assistant message;
for Codex it is a completed activity item (message, reasoning, or tool), because
Codex's JSON `turn` covers the entire invocation. These units are deliberately
different and are not suitable for cross-provider model-call comparisons.

`TokenPrices` requires all four token-category rates and a dated source. No price
is guessed from a mutable model name. Costs distinguish caller-rate estimates,
CLI-reported totals, and unavailable accounting. USD caps stop work when observed
usage crosses the threshold; one in-flight response and polling latency can
overshoot it. Computer agents let the SDK retry connection drops, 429s and 5xx responses up to 4 times with backoff; a retried request may be billed twice. Failed requests may have unknown
billing. Wall limits include provider time and actions; transport timeouts and
cleanup introduce bounded overhead rather than a precise real-time interrupt.

Codex rejects every finite `max_cost_usd` before launch: usage arrives only at the
end of its task. Claude accepts a dollar cap only with explicit `TokenPrices` and
stops from per-message usage; missing usage fails closed. Its native dollar-budget
flag is an additional control. Turn caps are enforced by the library; the adapter
does not rely on undocumented CLI flags. Interrupted accounting may be incomplete.

Codex standalone `error` frames do not terminate a run: an older local exec exporter
forwards retry notifications without their retry flag. The exact pinned release's
retry emission is **not confirmed**. Terminal events/process exit decide failure,
with the existing wall deadline as the bound. Terminal failures with recognized
structured provider codes use `provider_error`; message-only or unknown-code errors
remain `error`, regardless of their prose. Evidence and fixture limits are recorded
in `tests/agents/fixtures/README.md`.

## API tool agents

`AnthropicToolAgent` and `OpenAIToolAgent` give the model only the environment's
declared API tools plus a `done` tool. No hosted or server-side tool (web search, code
execution, browsing) is requested, and provider keys stay on the host. An environment
declares its tools as trusted configuration:

```python
from deepgym.models import MultiTurnEnvironment
from deepgym.tools import HttpRoute, HttpTool

env = MultiTurnEnvironment(
    task='your task',
    snapshot='your-app-snapshot',
    root_service_start='your trusted service start command',
    tools=(
        HttpTool(
            name='bugzilla',
            description='Bugzilla 5.0 REST API. Put the API docs the task needs here.',
            base_url='http://127.0.0.1:8080',
            routes=(
                HttpRoute(method='GET', path='/rest/bug'),
                HttpRoute(method='GET', path='/rest/bug/{id}/comment'),
                HttpRoute(method='POST', path='/rest/bug/{id}/comment'),
            ),
            headers={'X-BUGZILLA-API-KEY': app_key},
        ),
    ),
    allowed_actions=frozenset({'tool', 'done'}),
    max_steps=None,
)
```

`base_url` must be on a loopback host. A call names a method and a path that fits a
declared route; `{name}` matches one path segment and `{name:path}` also takes an
encoded `/` (GitLab project paths and file paths). The model supplies only the method,
path, query and JSON body. Each call is a Session `tool` action: the host checks it,
then a constant runner sends it with http.client from inside the sandbox, as the agent
user in the offline network namespace. Its request arrives on stdin from a root-only
file. Redirects are returned, never followed.
Header values are credentials: masked in dumps, environment fingerprints and validation
errors, and every tool's are redacted from every call's record and result, including
forms an app may echo (escaped, percent-encoded, base64, a Basic pair or password, any
letter case) and the cookies it sets. A body that isn't UTF-8 comes back base64-encoded,
with its bytes redacted. Header values must be ASCII, a Basic pair inside one too, so the
bytes an app sees match what redaction looks for. Pattern redaction cannot catch every
transformation, so declare only routes that never return credentials. An environment with tools allows exactly
`tool` and `done`. Tools are trusted configuration, so an inline environment sent to the
sessions API cannot declare any; a registered environment can, in code or in its
folder's `metadata.json` (`tools`, `allowed_actions`, `root_tool_settle`).

A route allows exactly a method and a path; the body is checked only as JSON, not
against the app's operations. A declared GraphQL endpoint (`POST /graphql`) or any
other endpoint that dispatches on its body (a JSON-RPC method, a batch call) therefore
allows every operation the tool's credentials permit. Give such a tool an app account
whose role allows only what the task needs.

The observation, which the model gets back whole, records the full request, the full
response (status, every header, the body) and timing. A response cut short keeps every
byte that arrived, marked `"incomplete": true`, and counts as a failed call. A write
whose response was lost, cut short or stopped by the step timeout is marked
`"uncertain": true` and is never retried.

An app may still change after it answers a write (a queued job, say) or after its runner
was stopped, so the session holds every write it sends as unsettled; only a call that
never connected is cleared. Before the final verifier runs, the environment's trusted
`root_tool_settle` command (root, offline namespace, `setup_timeout`) must settle them:
exit 0 only once the app has applied or abandoned every request it received, for example
by stopping its server or waiting for its queue to drain. Without it, or if it fails,
the episode gets no verdict (`error_type='unsettled_write'`) whatever its score, and
counts toward the masked rate. Writes leave no settled state between steps, so an
environment whose tools have any route other than `GET` cannot have a step verifier.

Only the `done` tool ends the episode; app output that reads `DONE` does not. A reply
without tool calls gets a reminder. Defaults are uncapped: maximum effort, Anthropic's
model-maximum `max_tokens`, no OpenAI output limit, and only the turn, time and USD
limits the caller's `Budget` sets. `AgentResult.messages` holds every model message, full
responses with their usage included. A provider response that never completed is kept
too, as what arrived marked `"status": "incomplete"`, and its tool calls never run. Nor
do the calls of an Anthropic response that stopped for any reason but tool use
(`max_tokens` mid-call, say): the run ends with an error, and the response is kept with
its `stop_reason`, usage and `input_json`, each call's input exactly as it streamed. Under
`dg.trace`, each step also carries the messages that produced it.

## CLI isolation and its exposure

CLIs must already be installed in the sandbox image; the library installs nothing.
Supported releases are pinned in `deepgym.agents.SUPPORTED_CLIS`. Each run checks
the installed version inside the sandbox before launching the task; unsupported
versions are refused. The observed version is retained in result and trace
provenance. Their flags must match the adapters. The environment
must allow `bash` or `cli`, and its backend must retain a separate network
namespace. LocalBackend and provider-only network blocking without that namespace
are refused. GUI-only environments cannot gain shell access via an agent adapter.

```python
import os

from deepgym.agents import CodexAgent
from deepgym.models import MultiTurnEnvironment

env = MultiTurnEnvironment(
    task='your task',
    snapshot=os.environ['DEEPGYM_CLI_SNAPSHOT'],
    cli_provider_egress=True,
    allowed_actions=frozenset({'cli'}),
)
agent = CodexAgent(model='your-codex-model', allow_model_credentials=True)
# OPENAI_API_KEY is read on the host; the CLI receives its CODEX_API_KEY equivalent.
```

`MultiTurnEnvironment.cli_provider_egress=True` is a separate, required opt-in for
CLI transport. By default Daytona blocks all sandbox network traffic; the provider
tunnel cannot reach its upstream API through that outer block. This flag removes
Daytona's sandbox-wide network block so the trusted proxy can connect outward.
**Trusted root processes outside the offline namespace then have unrestricted
egress**, not provider-only or HTTPS-route-limited access. The agent still runs in
its loopback-only network namespace and PID containment; root service startup and
the verifier remain inside the offline network namespace. Missing opt-in is
rejected before CLI launch, and inline API environment specs reject the override.
Use it only on trusted CLI snapshots.

`allow_model_credentials=True` explicitly permits the model and its commands to
read the real provider key. A root-owned CONNECT proxy outside the namespace and
a loopback bridge inside it permit only the selected provider hostname on port
443. DNS results must be public addresses; connections use the validated numeric
address. Daytona's privileged loopback toolbox remains outside the agent's
namespace. This is destination filtering, not HTTPS-route filtering or account
spend enforcement: arbitrary calls to that provider can bypass CLI accounting.
The CLI and its children inherit matching `NO_PROXY`/`no_proxy` values for
`localhost` (including its trailing-dot form), `127.0.0.1`, and compressed/expanded
`::1` with or without URL brackets. Local app traffic therefore bypasses the provider
proxy inside the isolated namespace. Other 127/8 addresses and IPv4-mapped IPv6
forms are not explicitly bypassed; remote destinations retain the HTTP(S) proxy.
TLS is opaque to the tunnel, so shared CDN addresses may also serve other hosts;
the CONNECT allowlist is not an application-layer endpoint guarantee.
Exact credential strings are redacted from capture; transformed secrets and task
contents can still appear in traces, which should be treated as sensitive.

The CLI runs as `dgagent` with capabilities and privilege escalation disabled.
CLI-capable sessions create a persistent PID namespace before any agent execution,
including setup and desktop processes. Each CLI invocation runs as PID 1 inside a
nested PID namespace, created before dropping privileges. Teardown kills the
persistent namespace's pinned init through a pidfd and waits for kernel destruction
before verifying that no live agent-UID process remains. The UID scan never serves
as the termination mechanism. Root services and verifiers stay outside containment.
CLI finalization also covers zero-dollar budgets and expired preflight deadlines,
before returning a result for grading; failed cleanup leaves the session ungradable.
Ordinary GUI scoring does not finalize CLI containment or stop the desktop.
Linux pidfds and util-linux PID/mount namespace support are required; unavailable
containment fails closed. This avoids relying on unverified cgroup delegation.
Root captures stdout outside agent-writable directories. One `cli` Session action
contains its nested events; events are never replayed through `Session.step`.
All processes with that episode's agent UID, including detached children, stop
before grading. This also stops an agent-owned browser/desktop: CLI grading must
inspect persisted state or separately owned services, not a live agent browser.
The verifier retains its separate unprivileged identity.

`MultiTurnEnvironment.root_service_start` starts trusted image services as root
inside the offline namespace before files, desktop, and ordinary setup. It uses
`setup_timeout`; the command must fail if readiness fails. Never populate this
field from model output. It supports snapshots whose database/web services need
root startup, without making the agent or verifier root.
Inline API specifications reject nonempty `root_service_start`; trusted registry
environments can retain it.

## Upgrading the CLIs

`deepgym.agents.SUPPORTED_CLIS` is the sole source of supported versions and npm
package names. Phase 1 supports one exact release per CLI. Image builders can
generate the installation command without maintaining their own pins:

```python
from deepgym.agents import SUPPORTED_CLIS, cli_install_command

print(cli_install_command())  # npm install -g, with both packages pinned
print({name: release.version for name, release in SUPPORTED_CLIS.items()})
```

To upgrade, change the version in that mapping, record sanitized stream fixtures
under `tests/agents/fixtures/<command>/<version>/stream.jsonl` with capture
provenance, run `python -m pytest tests/agents/test_cli_compat.py tests/agents/test_cli.py -q`,
then rebuild sandbox images using `cli_install_command()`. The compatibility test
checks the installed CLI's version and every flag supplied by the adapter against
its help; it skips a CLI absent locally and never invokes a model. Initial stream
fixtures remain schema-derived, as documented in their README; recording model
runs requires separate authorization. Versioned parser fixtures catch missing
upgrade coverage but do not turn schema samples into live compatibility evidence.

## Offline verification and live smoke

The offline suite covers mappings, provider loops, event streams, budgets,
namespace and identity invariants, capture/ATIF/Parquet preservation, and startup
ordering. Stream fixtures are schema-derived samples, not claimed live recordings.
No performance improvement or live sandbox isolation proof is claimed by those
tests. The opt-in GUI test is pinned to `img-itop-3.3.0-v1`: it creates a uniquely named
organization through the GUI and verifies that organization through iTop's API.

With `DAYTONA_API_KEY`, `ANTHROPIC_API_KEY`, and `ITOP_ADMIN_PASSWORD` exported, run
from the package directory:

```bash
DEEPGYM_LIVE_AGENTS=1 DEEPGYM_AGENT_MODEL=claude-opus-5-5 python -m pytest tests/agents/test_live.py::test_anthropic_itop_snapshot -q -s
```

The CLI smoke is independently gated by `DEEPGYM_LIVE_CLI_AGENTS=1`. Set
`DEEPGYM_CLI_SNAPSHOT` to your trusted snapshot with the supported CLIs already
installed. It creates a unique file through the model's tools, then verifies its
exact contents after process teardown; no iTop credentials are needed. The test
explicitly enables `cli_provider_egress` and model-readable credentials. With
`DAYTONA_API_KEY` and the corresponding provider key exported, run either test:

```bash
DEEPGYM_LIVE_CLI_AGENTS=1 DEEPGYM_CLI_SNAPSHOT="$YOUR_CLI_SNAPSHOT" DEEPGYM_CLAUDE_MODEL="$YOUR_CLAUDE_MODEL" python -m pytest 'tests/agents/test_live.py::test_cli_snapshot_roundtrip[anthropic]' -q -s
DEEPGYM_LIVE_CLI_AGENTS=1 DEEPGYM_CLI_SNAPSHOT="$YOUR_CLI_SNAPSHOT" DEEPGYM_CODEX_MODEL="$YOUR_CODEX_MODEL" python -m pytest 'tests/agents/test_live.py::test_cli_snapshot_roundtrip[openai]' -q -s
```

Each case skips unless its own snapshot, model, provider key, and Daytona key are
present. These commands create paid model calls and Daytona sandboxes when enabled;
the CLI roundtrips have not been run during this change.

Phase 2 replaces the transport at `Session.run_cli` with the hosted key-injecting
relay, revocable episode credentials, inference-route restrictions, and live dollar
metering. Phase 1 does not pretend to provide those guarantees.

Implementation status: all four adapters, startup and transport hooks, optional
extras, and offline checks are present. The live smoke remains unrun; no sandbox,
provider request, or performance benchmark was used to validate this change.

Validation on 2026-10-18: `ruff check src/` passed; `ruff format --check src/`
reported 173 files already formatted. The loopback-proxy pytest run with four workers
passed 230 tests and skipped 3, including CLI environment bypass checks, namespace
invariants, and both local CLI help checks. All three live tests were skipped:

```bash
python -m pytest tests/agents tests/test_cli_runtime.py tests/test_pid_namespace.py -n 4 -q
```

A broader check found `test_computer_use_envs.py::test_computer_use_suite_is_separate`
failing because its expected list omits 182 registered environments. The same failure
was reproduced from clean starting commit `62d39f532a8944b20d3264fc09d2c44ab25c5472`;
that unrelated test and registry were left unchanged.
