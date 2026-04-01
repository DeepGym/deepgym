# Changelog

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
