# Contributing to DeepGym

Thanks for helping. New environments, verifiers, benchmark importers, integrations
and docs are all welcome.

## Hacktoberfest

DeepGym takes part in Hacktoberfest. Pick an issue labelled
[`hacktoberfest`](https://github.com/DeepGym/deepgym/issues?q=is%3Aissue+is%3Aopen+label%3Ahacktoberfest)
or [`good first issue`](https://github.com/DeepGym/deepgym/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22),
comment that you're taking it, and open a PR. Merged PRs get `hacktoberfest-accepted`.
PRs that only reformat code, fix one typo, or are AI-generated without being run
are closed as spam.

## Setup

```bash
git clone https://github.com/DeepGym/deepgym.git
cd deepgym/deepgym          # the Python package lives in deepgym/
pip install -e ".[dev]"
pytest                      # no Daytona or API keys needed
ruff check src/ && ruff format --check src/
```

## Add an environment

The fastest way to contribute. Each built-in environment is a folder in
`src/deepgym/envs/<name>/`:

- `task.md`: the prompt the model sees.
- `verifier.py`: scores a solution. Use `load_solution` / `run_solution` from
  `deepgym_solution` so the solution runs isolated from the judge; never import
  or exec it yourself. See `src/deepgym/envs/coin_change/verifier.py`.
- `reference_solution.py`: a correct solution. It must score 1.0.
- `metadata.json`: id, name, difficulty, domain, tags.

Then add an entry to `src/deepgym/envs/registry.json`. `tests/test_envs_smoke.py`
runs every registered environment against its reference solution, so `pytest`
tells you if it works. Also check that an obviously wrong solution scores below 1.0.

## Code style

- Python 3.10+, type hints on public functions, docstrings on public classes and methods.
- Ruff for linting and formatting, single quotes.
- Tests for new behaviour, in `tests/`.

## Submitting changes

1. Fork the repo and create a branch (`feat/...`, `fix/...`, `docs/...`).
2. Make the change with tests.
3. Run `pytest` and `ruff`.
4. Open a PR that says what changed and how you tested it.

Before we can merge your first PR, you'll need to sign our
[Contributor Assignment Agreement](CLA.md). A bot comments on your PR with a link.
Signing takes a minute and covers all your PRs.
