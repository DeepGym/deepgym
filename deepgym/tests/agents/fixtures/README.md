Version compatibility streams live at `<command>/<version>/stream.jsonl`. Tests
resolve the command and version through `deepgym.agents.SUPPORTED_CLIS`; a version
bump fails until its fixture exists. The directory names identify compatibility
targets, not evidence that the CLI produced these initial samples. The separate
`claude.jsonl` and `codex.jsonl` schema samples remain fixed inputs for detailed
accounting and malformed-event unit tests, so new recordings need not reproduce
their particular token counts, identifiers, or text.

The initial JSONL samples are hand-authored from the documented CLI event shapes
and the HUD reference parser schemas. They are **schema-derived fixtures**, not
captured model runs. No SDK or provider API is used to replay them. Authentic
recordings remain required before claiming end-to-end compatibility for a release.

- `claude/*/stream.jsonl`: stream-json assistant/tool/result lifecycle, cumulative usage.
- `codex/*/stream.jsonl`: exec --json command/message/task lifecycle, cached input usage.

For an upgrade, record a harmless completed task from the pinned CLI in an
authorized environment, including a tool call, final response, and token usage.
Sanitize credentials, user paths, and identifiers without changing event shape or
usage fields. Add a provenance note beside the fixture with the command, capture
date, and sanitizations. Never relabel a schema-derived sample as a recording.
The compatibility parser check accepts any completed task with nonempty final
text and reported output-token usage. Run
`python -m pytest tests/agents/test_cli_compat.py tests/agents/test_cli.py -q`.

Commands in these samples must only be preserved as nested observations. Tests
fail if an adapter tries to execute them again through `Session.step`.

## Codex error evidence (2026-10-18)

The installed npm package at `/opt/homebrew/lib/node_modules/@openai/codex/package.json`
matches the pinned release in `SUPPORTED_CLIS`; its `codex --version` agrees.
`codex exec --json --help` documents JSONL output but does not define error/retry fields.
Exact pinned-release retry emission is **not confirmed**; no model call was made.

Local source `/Users/abhishek/Desktop/typesafe/research/codex-jev` imports upstream
`7498521d288b9b3b96ffba4eedf089d8d6e06a84` (an older snapshot, not release proof):

- `codex-rs/app-server/src/bespoke_event_handling.rs:1054-1070` emits intermediate
  stream errors with `will_retry: true`.
- `codex-rs/exec/src/event_processor_with_jsonl_output.rs:447-457` converts these
  notifications to message-only `error` events and keeps running, dropping retry
  and structured error metadata. Lines 525-557 handle terminal turn outcomes.
- `codex-rs/exec/src/exec_events.rs:90-94` defines `ThreadErrorEvent` with only
  `message`; `exec/tests/event_processor_with_json_output.rs:1691-1740` also keeps
  running after an error notification with `will_retry: false`.
- `codex-rs/app-server-protocol/src/protocol/v2/shared.rs:70-120` defines the
  structured provider codes used only when a terminal error actually retains them.

The versioned `retry-then-success.jsonl` is a source-derived regression scenario,
not a captured run. `terminal-provider-error.jsonl` tests retained structured fields;
their presence in the pinned `exec --json` output is **not confirmed**. Message-only
terminal failures stay generic `error`. Standalone errors cannot prove terminality,
so the parser waits for a terminal event or process exit, bounded by the existing
wall-time budget. Neither success nor provider attribution is inferred from prose.
