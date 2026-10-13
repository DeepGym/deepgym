# Changelog

## Unreleased

- **Security: repo, terminal and cyber tasks no longer run on the host.** SWE-bench Pro, Terminal-Bench, CyberGym patch tasks and MiMo code tasks apply the model's patch or run its commands, then run tests, with no judge isolation. In `mode='local'` (and `auto` without Daytona) they used to do that on your machine; now they raise `SandboxError` and ask for `docker`, `apple`, `modal`, `e2b` or `daytona`. There is no opt-out. They also start their sandbox from the client's provider instead of always Daytona, and `scripts/run_cybergym_artifact_eval.py` drops `--mode local`.
- **Security: patches can't edit hidden tests.** The SWE-bench Pro patch runner used to apply the hidden test patch first and the model's diff after it. A diff could then still edit a hidden test file, for example through the source of a renamed test. Now the model's diff goes first. Every path the test patch touches, as listed by git's own parser (`git apply --numstat -z`, forwards and reversed), is put back to its state before the diff, and the test patch goes last. A diff that fails `git apply --check` still scores 0.
- **SWE-bench Pro: no stand-in tests.** A task with no test runner (no `test_commands`, and a language other than Python or JS/TS) now returns `error_type='unsupported_runner'` with score 0, so trainers mask it. Before, a `git diff --stat` stand-in exited 0 and a still-broken patch scored 1.0. A task with no tests no longer earns the 0.3 apply credit either, and the JS `npm install` step no longer counts as a passing test.
- **Patch extraction:** a diff whose last line is a blank context line (a single space) is no longer corrupted by whitespace stripping, which failed correct patches.
- **MiMo-V2.6-RL code tasks:** `scripts/import_mimo_rl.py` imports the 2,698 code tasks of `XiaomiMiMo/MiMo-V2.6-RL-oss` (Apache-2.0) as `mimo_code/<instance_id>` environments. Each is a `RepoPatchEnvironment`: its sandbox boots from the task's own image, the patch runner patches the repo in place, and the score is 1.0 if the test command exits 0, else 0.0. The cyber, general, webdev and music subsets aren't imported (see the README).
- **verl:** a patch that is malformed or doesn't apply (`invalid_patch`, `patch_apply_failed`) now scores 0 and is trained on instead of being masked out as a missing verdict, so models are penalised for broken patches.
- **Session reward integrity:** Daytona sessions now run through `SandboxBackend`, like every other provider: agent commands as `dgagent` in a loopback-only network namespace, verifiers as `dgverifier` from a root-only file. Before, the agent and verifier shared Daytona's passwordless sudoer, and the agent could reach Daytona's toolbox API on loopback, so it could forge any score. Desktop sessions run the whole xfce session and its apps as the agent and lock the desktop down before setup (no Thunar custom actions, "Open With" or custom-command launches, terminals, IDLE, or x11vnc remote control); setup scripts no longer get sudo, and Chromium runs with `--no-sandbox` as the unprivileged agent. Verifier scores only count after a clean exit (0, or 1 for a failed run): a verifier killed (e.g. exit 137) or crashing after printing a score is a `VerifierError`, never reward. Session verifiers can use `load_solution`/`run_solution` (locally, and where the image has sudo as a separate `dgsolver` user); the multi-turn `debug_fix` verifier now does, so an always-equal object no longer passes.
- **`api_request` env:** the verifier serves randomized slideshow documents, drawn from a fresh seed every run (reported as `seed`), from a local HTTP server and checks that the solution, run with each URL as `sys.argv[1]`, prints that document's `slideshow.title`. Before, it scored source keywords and output against a mocked response, so comments plus a fixed print scored 1.0 and passed the QA gate. The task now passes the URL as an argument and asks for the standard library only.
- **Loopback for solutions:** `run_solution(..., loopback_port=port)` lets the solution connect to a server the judge runs on `127.0.0.1:port` and nothing else: macOS allows that one port in the sandbox profile, and where the solution has its own network namespace (bwrap, sandbox users) a forwarder pipes the port through a Unix socket in a judge-owned directory the solution can enter but not change. In sandboxes the solution's offline namespace now has loopback up; root does that with `python -I -S` and a fixed `PATH`, so nothing in the solution's working directory or the environment can reach it.
- **Per-case scores:** a failed case without an explicit `score` now counts 0. Before, it got the passing default of 1.0, so `per_test_rewards` paid full credit for failed tests.
- **TRL:** the reward functions accept conversational completions (lists of message dicts, including tool-calling turns) and run the last fenced code block of the final assistant turn, like the verl adapter. Before, correct code scored 1 as plain text but 0 as a conversation, and fenced answers in plain text scored 0.
- **OpenRLHF:** the reward router accepts the payload OpenRLHF's agent client sends (`query`, `prompts`, `labels`), strips each prompt from its query and runs the response's last fenced code block. When OpenRLHF cut a long prompt's front tokens, the response starts after the prompt's tail; a query that doesn't contain its prompt scores 0 with `error_types` `prompt_mismatch`, so prompt text is never scored. Matching is linear in the text length, and requests are capped at 2M characters per text, 1024 samples and 64M characters in all (422 above). `outputs` still works. Before, OpenRLHF's requests were rejected with 422.
- **Code extraction:** `extract_code`, which the verl, TRL and OpenRLHF adapters and the QA difficulty probe share, drops end-of-turn tokens left at the end of a response (`<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `</s>`, `<end_of_turn>`, ...). Before, valid unfenced code decoded with `skip_special_tokens=False` failed with a SyntaxError and scored 0.
- **Reward-hack attacks:** an attack whose run fails (timeout, sandbox or verifier error), whose result reports a verifier error (an `error_type` other than a model fault such as `invalid_patch`, or exit code 2, e.g. a verifier that raised inside the wrapper), or that never runs (LLM attack without a key or parseable exploits) is now `inconclusive` instead of counted as resisted. The rule is `RunResult.has_verdict`, which the verl reward's `error` flag now uses too, so it also masks exit-code-2 runs. `AdversarialReport` gains an `inconclusive` count, and `is_robust` is true only when every attack ran and none got through. `RewardAuditor` flags a verifier `exploitable` only when an exploit was found, and saved attack records carry an `inconclusive` flag. `ExploitDB.save` refuses an inconclusive record (no exploits, some attacks without a verdict, for example after a provider failure or without an Anthropic key) and returns False, so `RewardAuditor.audit(persist=True)` reports `stored=False` and `scripts/run_exploit_scan.py`, which now builds records the same way, tests that verifier again next time. Before, such audits were stored (the script even saved them as EXPLOITABLE) and later scans skipped them for good.
- **Quality gate:** every check and report has a `status`: `pass`, `fail`, `inconclusive` or `skip`, and `passed` is true only for `pass`. Multi-turn, computer-use and benchmark-backed environments are `skip` (neither passed nor failed) instead of passing; `hack_resistant` is `inconclusive` when no attack got through but some crashed; `stub_fails` is `skip` for a script reference. `deepgym qa` exits 0 when the checked environments pass, 1 if any fails, and 2 if none fails but some are inconclusive or none could be checked. The table and the console show the new statuses. Reports written by older versions don't load in the console; rerun `deepgym qa` to refresh them.

## 0.4.1 (2026-10-13)

- **License:** DeepGym is now under the PolyForm Noncommercial License 1.0.0. Commercial use needs a license (contact@deepgym.io). Releases up to 0.4.0 stay MIT.

## 0.4.0 (2026-10-13)

- **Console:** `deepgym console` (alias `deepgym web`) serves a management UI and a JSON API under `/api`: searchable environments with task/verifier/reference views, background runs and small evals on any provider with per-case results, episode replay with screenshots, live Docker/Apple sandboxes with kill, and QA reports from `~/.deepgym/qa/`. Runs and episodes persist in `~/.deepgym/console.db`; `deepgym.console.record_episode(trajectory, result)` stores an episode from any agent loop. The API needs a per-launch token written into the served pages, refuses cross-origin writes and non-JSON bodies, and runs pasted solutions locally only with `--allow-local-exec`. Wheels ship the built UI. Replaces the single-page `deepgym web` debug UI (`deepgym.web` is removed).
- **verl masking:** `deepgym.integrations.verl_mask.DeepGymSingleTurnAgentLoop` scores single-turn rows itself and zeroes the response mask of samples whose sandbox or verifier failed (`error: 1.0`), so they get no gradient instead of training as a 0. A verl reward function or reward manager can't drop a sample.
- **Per-environment images:** set `image` on an `Environment`, a `MultiTurnEnvironment` or in `metadata.json` to boot its sandbox from that image. Docker, Apple container, Modal and Daytona pull it from a registry (a Daytona `snapshot` wins); E2B uses it as a template id. The image must run as root and have python3, bash, useradd and util-linux. The HTTP API refuses caller-chosen images.
- **Quality gate:** `deepgym qa [ENV ...] [--all]` checks single-turn environments before training: the reference solution scores 1.0 on every rerun, empty and stub solutions score below 1.0, and no heuristic reward-hack attack reaches the pass threshold. `--model` adds pass@1/pass@k from an OpenAI-compatible endpoint and flags tasks that are too easy or too hard. Writes one JSON report per environment to `~/.deepgym/qa/` and exits 1 if any environment fails. `--model` with `--mode local` or `auto` needs `--allow-local-exec`. All 26 built-in single-turn environments pass; the environment smoke test now runs the gate.
- **Reward-hack attacks** (`AdversarialTester`, `RewardAuditor`, the audit API) run through the `DeepGym` client they're given instead of always running locally, so a Daytona or container client runs them in its sandboxes.
- **Sandbox providers:** `mode='modal'`, `'e2b'` and `'docker'` join `'daytona'` and `'apple'` for single runs, `AsyncDeepGym` and shell sessions, all with the same isolation. Each provider is a folder in `deepgym/sandboxes/` implementing one small `Sandbox` protocol; Modal and E2B are optional extras. Judge runs fail closed when a sandbox can't cut the solution off the network. In sandbox sessions, verifiers run as their own user from a file the agent can't read, so leftover agent processes can't read or forge the score; agent and solver drops add no-new-privs; container sandboxes are capped at 2 CPUs, 2 GB and (Docker) 512 processes. `deepgym.apple_container` moved to `deepgym.sandboxes.apple`, `AppleContainerBackend` is now the provider-agnostic `SandboxBackend`, and `create_sandbox`/`cleanup_sandbox` moved to `deepgym.sandboxes.daytona`.
- **Apple container backend:** `DeepGym(mode='apple')` / `AsyncDeepGym(mode='apple')` run judges and multi-turn sessions in throwaway Linux VMs from Apple's `container` CLI on macOS. Same isolation as Daytona: the solution and session commands run as non-root users with all capabilities dropped and no network; a judge run fails rather than run online if the image lacks util-linux. Image set by `DEEPGYM_CONTAINER_IMAGE`. Desktop envs still need Daytona. On Daytona the solution now also runs without network when the sandbox allows it.
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
