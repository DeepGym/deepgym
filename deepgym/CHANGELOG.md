# Changelog

## Unreleased

- **Apple container backend:** `DeepGym(mode='apple')` / `AsyncDeepGym(mode='apple')` run judges and multi-turn sessions in throwaway Linux VMs from Apple's `container` CLI on macOS. Same isolation as Daytona: separate solver user, no network for the solution or session commands, all capabilities dropped. Image set by `DEEPGYM_CONTAINER_IMAGE`. Desktop envs still need Daytona. On Daytona the solution now also runs without network when the sandbox allows it.
- **Rewards:** NaN, inf and out-of-range scores are verifier errors, never reward. A single run raises `InvalidScoreError`; batches and trainer reward functions score that sample 0 with `error_type='invalid_score'`. Non-finite reward components are zeroed. Before, NaN scored 1.0.
- **Benchmark-backed envs:** SWE-bench Pro, Terminal-Bench and `MixedEnvironment` now score correctly in `AsyncDeepGym` and in single sync runs. Before, both silently scored a 0.0 placeholder.
- **Benchmark names:** a fully qualified name such as `humaneval_plus/HumanEval_0` beats loose matches, and an ambiguous name raises instead of loading the wrong benchmark. Before, HumanEval+ requests could load HumanEval.
- **Isolation hardening:**
  - Local runs hide the home directory and every other run; only the Python install and the run's own directories are visible.
  - On Daytona the judge's temp files are private.
  - On macOS solutions can't fork unless `run_solution(..., allow_subprocess=True)`.
  - Scripts' leftover processes are killed when the script exits.
  - `cli_task` reads `.git` without running git.
  - `read_output` refuses symlinked paths.

- **Security: the solution no longer runs inside the judge.**
  - Before, single-turn verifiers imported the model's code into their own process, and in both local and Daytona mode it shared a folder with the verifier and test cases. It could print a fake score line, patch the verifier, read the expected answers, import the reference solution from the installed package, or return an object that equals everything.
  - Now verifiers call `load_solution()` / `run_solution()` from `deepgym_solution`, which run the solution in a separate isolated process:
    - Daytona: a dedicated Linux user.
    - Local: `sandbox-exec` (macOS) or `bwrap` (Linux), with the judge, the deepgym package and `~/.deepgym` hidden.
  - Only plain data crosses the boundary. All built-in verifiers and the HumanEval, MBPP and EvalPlus importers are migrated; re-run the import scripts for cached benchmarks.
  - BigCodeBench tests run with the solution inside the sandbox, because they need rich objects. That result is still forgeable by a determined model.
- Per-call `case_timeout` (default 10s), capped by the run's `timeout`: a slow case fails alone and the run returns partial credit instead of timing out.
- Verifier shape is detected by parsing the source (`script`, `verify_function`, `body`) or set with `verifier_mode`, replacing a text search that misclassified verifiers. `from __future__` imports are hoisted when wrapping.
- `support_files` on `Environment` (and `metadata.json`) for multi-file verifiers and shared helpers across environments.
- `cli_task` now works in its own directory (`my_repo` in the working dir) instead of a shared `/tmp/my_repo`.
- Breaking: `LocalExecutor.run()` / `AsyncLocalExecutor.run()` take a `JudgeJob`. `is_standalone_verifier`, `setup_sandbox` and `execute_verifier` are replaced by `prepare_verifier` and `run_judge_in_sandbox`.

- Added live multi-turn sessions (`dg.session()`, `dg.run_episode()`, `Session`): one sandbox per episode, actions applied step by step, final verifier run against the resulting state. Local and Daytona backends.
- Added real computer-use support on Daytona's desktop: typed GUI actions (click, type, key, scroll, drag, ...), full-resolution screenshots in observations, GUI-only action sets by default
- Added 5 computer-use environments (`signup_form`, `todo_list`, `settings_tabs`, `expense_sheet`, `file_manager`) with progress/guard verifiers and scripted reference runs checked live on Daytona
- Added 8 seeded variants per computer-use env (`load_environment(name, variant=i)`), each with a scripted reference run verified live
- Desktop lockdown during setup: Chromium devtools and `javascript:`/`chrome:`/`file:` URLs blocked, terminals and the run dialog disabled, file manager removed from browser tasks
- Added `/v1/sessions` HTTP endpoints (create, step, score, delete) with a race-safe session cap and 30-minute idle eviction
- Added verl multi-turn support: `DeepGymAgentLoop` (`agent_name: deepgym_agent`) keeps one sandbox per rollout, since verl's stock loop creates and releases tools per call, and sets `reward_score` from the final verifier; `DeepGymSessionTool` is its tool
- Verifiers in sessions run with `python -I` and are passed inline, never written to the sandbox, so the agent can't swap them or hijack them with planted modules
- Setup scripts that exit non-zero raise `SandboxError`; shell steps report `exit_code` in observation metadata; both backends run commands under bash
- Step verifiers that print garbage now raise `VerifierError` instead of scoring 0
- CI: ruff skips markdown task prompts and JSON payloads stored as `.py`, which formatting would change
- `ActionError` for actions the environment rejects (not allowed, unknown key name); the verl tool returns these to the model as text and raises everything else
- Sessions serialize concurrent step/score/close calls; a failed step verifier ends the episode with a consistent trajectory; step-verifier stderr is ignored
- `setup_timeout` (default 120s) bounds setup separately from per-step actions
- Daytona sessions create their work dir up front and tell command timeouts apart from connection timeouts via the SDK's typed errors
- `dg.eval()` and `AsyncDeepGym.run()` reject multi-turn envs with a clear error instead of crashing
- One shared `parse_verifier_output()` replaces five copies of verifier JSON parsing
- Daytona sessions wait for sandbox deletion before freeing an API slot, bound every desktop call with a request timeout, and delete sandboxes left behind by a failed create (found via a per-session label)
- Session API store is locked so idle eviction can't race a session that was just used
- Computer-use app state saves are atomic and applied in order (todo app chains its saves)
- Local backend runs each command in its own process group and kills leftovers on timeout and close
- Breaking: removed `MultiTurnRunner` (use `dg.run_episode()`), `ScreenshotVerifier`, and `ToolUseEnvironment`; `ComputerUseEnvironment` is now a desktop `MultiTurnEnvironment`
- Breaking: `file_organizer` and `cli_task` are now type `coding` (they were single-shot scripts, not computer use); computer-use envs only appear in the `computer-use` suite
- Switched the Daytona dependency from the deprecated `daytona-sdk` to `daytona`; Daytona session sandboxes block outbound network

## 0.3.0 (2026-04-02)

- Added a thin native DAPO adapter with reward/config helpers for verl-style training recipes
- Hardened imported benchmark loading so nested HumanEval/MBPP-style benchmark registries resolve predictably
- Fixed benchmark alias and task-id routing for imported suites such as `humaneval` and `HumanEval/0`
- Made mixed benchmark routing stricter and rejected mismatched per-sample batch metadata instead of silently broadcasting it
- Restored Python 3.10 compatibility in the release/test matrix

## 0.2.0 (2026-04-02)

- Added `SWEBenchProEnvironment` for repo-level patch tasks backed by ScaleAI/SWE-bench_Pro
- Added `TerminalBenchEnvironment` for shell-task evaluation backed by Terminal-Bench 2.0
- Added `MixedEnvironment` for ratio-based multi-benchmark reward routing through existing TRL/reward APIs
- Added reusable `PatchVerifier` for unified-diff parsing, apply checks, and repo test scoring
- Extended reward integrations so per-sample benchmark metadata can flow through existing batch interfaces

## 0.1.0 (2026-03-17)

Initial release.

- Core SDK: DeepGym client with run(), run_batch(), eval()
- 25 built-in environments
- 2,350+ importable benchmarks
- Gymnasium-style API
- Framework integrations: TRL, verl, OpenRLHF
- Multi-turn environment support
- Web debugging UI
- FastAPI server with async jobs
- CLI: run, eval, serve, web, create
