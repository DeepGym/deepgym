# DeepGym

Managed sandboxed execution, scoring, and evaluation infrastructure for RL and agent training loops. Models take actions, we execute them in Daytona sandboxes, run verification, and return reward signals.

## Project structure

```
src/deepgym/
├── models.py            # All Pydantic models (Environment, RunResult, VerifierResult, Job, etc.)
├── core.py              # Sync DeepGym client (mode: auto/daytona/local)
├── async_core.py        # Async client with semaphore-based concurrency
├── sandbox.py           # Daytona sandbox lifecycle + LocalExecutor fallback
├── verifier.py          # Verifier model + protocol validation
├── verifier_template.py # Wrapper normalizing any verifier to JSON protocol
├── adversarial.py       # Reward-hack detection (5 attack strategies)
├── exceptions.py        # DeepGymError hierarchy
├── cli.py               # CLI: run, eval, serve, create
└── api/
    ├── app.py           # FastAPI app + API key auth middleware
    ├── routes.py        # Sync + async job endpoints
    ├── schemas.py       # Request/response Pydantic models
    └── deps.py          # Dependency injection
examples/                # 3 example environments with verifiers + solutions
```

## Commands

```bash
# Install (local mode, no Daytona needed)
pip install -e .

# Install with Daytona support
pip install -e ".[daytona]"

# Install everything (dev + daytona)
pip install -e ".[all]"

# Run the API server
deepgym serve --host 127.0.0.1 --port 8000

# Run a single environment
deepgym run --task task.md --verifier verifier.py --solution solution.py

# Run linter
ruff check src/

# Run tests
pytest
```

## Daytona setup

Self-hosted (local):
```bash
git clone https://github.com/daytonaio/daytona
docker compose -f docker/docker-compose.yaml up -d
# Dashboard: http://localhost:3000 (dev@daytona.io / password)
# Set DAYTONA_API_URL and DAYTONA_API_KEY for the local instance
```

Cloud: set `DAYTONA_API_KEY` from app.daytona.io.

## Core principles

All code MUST be fully optimized:
- Maximize algorithmic big-O efficiency for memory and runtime.
- Use parallelization and vectorization where appropriate.
- Follow DRY — maximize code reuse, no duplicated logic.
- No extra code beyond what is necessary. Zero technical debt.
- If code is not fully optimized, do another pass before finishing.

## Code standards

### Python version and types

- Target Python 3.10+. Use `X | Y` union syntax, not `Union[X, Y]` or `Optional[X]`.
- Use `from __future__ import annotations` only when needed for forward refs.
- Every public function and method has type annotations on all parameters and return type.
- Never use `Any` type unless absolutely necessary — prefer specific types.
- Use `Literal` for constrained string values, not bare `str`.
- Prefer `Sequence` over `list` in function signatures when the function only reads from the collection.
- Use `is` for comparing with `None`, `True`, `False`.

### Pydantic

- All data models use Pydantic `BaseModel`, not `dataclass`.
- Use `Field()` for validation constraints (`ge=`, `le=`, `min_length=`, etc.).
- Use `model_validator` for cross-field validation, not `__post_init__`.
- Immutable models: set `model_config = ConfigDict(frozen=True)` where the model should not be mutated after creation.
- Never use `dict()` on models — use `model_dump()`.

### Error handling

- Use the exception hierarchy in `exceptions.py`: `DeepGymError` > `VerifierError`, `SandboxError`, `TimeoutError`.
- Never silently swallow errors. If a verifier fails to parse, raise `VerifierError` with context, don't return a zero score.
- Always include the original exception as `raise XError(...) from e`.
- Never use bare `except:` clauses. Catch specific exceptions.
- Sandbox cleanup goes in `finally` blocks. Cleanup errors are logged but not raised.
- Use context managers (`with` statements) for resource cleanup.
- Provide meaningful error messages with context.
- Use `logger.error()` not `print()` for error reporting.

### Verifier protocol

Every verifier outputs JSON to stdout:
```json
{
  "schema_version": "1.0",
  "score": 0.85,
  "passed": true,
  "details": "8/10 tests passed",
  "reward_components": {"correctness": 0.8, "efficiency": 0.9},
  "metrics": {"execution_time_ms": 142, "memory_mb": 24},
  "seed": 42,
  "truncated": false,
  "error_type": null
}
```
- `score` is always 0.0-1.0, clamped.
- User verifiers return float, bool, or dict — the wrapper template normalizes to this schema.
- Exit codes: 0 = passed, 1 = failed, 2 = verifier error.

### Async patterns

- Use `AsyncDaytona` and `asyncio.Semaphore` for parallel execution, never raw thread spawning.
- The sync `DeepGym` client uses `ThreadPoolExecutor` for `run_batch` only.
- All async methods are prefixed with `async def`, never wrap sync code in `asyncio.to_thread` unless interfacing with sync-only libraries.
- Use `asyncio.gather(*tasks, return_exceptions=True)` for batch operations — don't let one failure kill the batch.

### Function and class design

- Keep functions focused on a single responsibility.
- Never use mutable objects (lists, dicts) as default argument values. Use `Field(default_factory=...)` or `None`.
- Limit function parameters to 5 or fewer. Use a config/params object for more.
- Return early to reduce nesting.
- Keep classes focused on a single responsibility.
- Keep `__init__` simple — avoid complex logic.
- Prefer composition over inheritance.
- Use `@property` for computed attributes.
- Use list comprehensions and generator expressions where clearer than loops.
- Use `enumerate()` instead of manual counter variables.
- Use f-strings for string formatting.

### Documentation

- Docstrings on all public classes, functions, and methods.
- Use imperative mood: "Create a sandbox" not "Creates a sandbox".
- Document function parameters, return values, and exceptions raised (Args/Returns/Raises).
- Keep comments up-to-date with code changes.
- Include examples in docstrings for complex functions.

```python
def run(self, env: Environment, model_output: str) -> RunResult:
    """Run a model output against an environment verifier in a sandbox.

    Args:
        env: The environment specification.
        model_output: Model-generated solution source code.

    Returns:
        RunResult with score, pass/fail, timing, and verifier details.

    Raises:
        VerifierError: If verifier output is not valid JSON.
        SandboxError: If sandbox creation fails.
    """
```

### Code style

- Max line length: 100 (configured in pyproject.toml ruff).
- Imports: stdlib, then third-party, then local. Enforced by ruff `I` rules.
- No wildcard imports. No `import *`.
- Prefer early returns over deep nesting.
- No dead code. No commented-out code. No TODO without a linked issue.
- Single quotes for strings unless the string contains a single quote.
- Use `pathlib.Path` for file operations, not `os.path`.
- Use `logging` module, not `print()`, for any operational output. `print()` is only for CLI user-facing output.
- Use snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants.
- Never use emoji or unicode that emulates emoji in code or output.

### Testing

- Tests go in `tests/` mirroring `src/deepgym/` structure.
- Use `pytest` with `pytest-asyncio` for async tests.
- Write unit tests for all new functions and classes.
- Test the verifier protocol contract explicitly — verify JSON output shape.
- Use `LocalExecutor` for tests, never require Daytona in CI.
- No mocking Daytona in tests — use `LocalExecutor` or skip with `@pytest.mark.skipif`.
- Follow Arrange-Act-Assert pattern.
- Never commit commented-out tests.
- Save test files before running them.
- Ensure test output folders are in `.gitignore`.

### Security

- Sandbox network isolation is ON by default.
- Never run user-provided code on the host. Always in sandbox or subprocess with timeout.
- API key auth is required in production (`DEEPGYM_API_KEY` env var). Dev mode (unset) skips auth.
- Never store secrets, API keys, or passwords in code. Use `.env` files (ensure `.env` is in `.gitignore`).
- Never log or print API keys, tokens, PII, sandbox contents, or user code at INFO level. DEBUG only.
- Never log URLs containing API keys.
- Use environment variables for all sensitive configuration.
- Verifier code is untrusted — always run with resource limits (timeout, memory).

### Git

- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- One logical change per commit.
- Branch naming: `feat/xxx`, `fix/xxx`, `refactor/xxx`.
- Never commit commented-out code — delete it.
- Never commit debug print statements or breakpoints.
- Never commit credentials or sensitive data.

### Before committing checklist

- All tests pass (`pytest`).
- Linter and formatter pass (`ruff check src/ && ruff format src/`).
- All functions have docstrings and type hints.
- No commented-out code or debug statements.
- No hardcoded credentials.

### Maintainability

Long term maintainability is a core priority. If you add new functionality, first check if there is shared logic that can be extracted to a separate module. Duplicate logic across multiple files is a code smell and should be avoided. Don't be afraid to change existing code. Don't take shortcuts by just adding local logic to solve a problem.

- Before adding new code, search for existing utilities that do the same thing.
- If you find yourself writing the same pattern in 2+ places, extract it immediately.
- Prefer modifying existing modules over creating new ones when the functionality is related.
- Keep module responsibilities clear and documented in docstrings.
- When refactoring, update all callers — don't leave dead imports or compatibility shims.

### What NOT to do

- Don't add abstractions until there are 3+ concrete uses. Three similar lines > premature abstraction.
- Don't add optional parameters "for future use." Add them when needed.
- Don't use `Any` in type annotations unless interfacing with untyped external code.
- Don't add logging, metrics, or config for things that aren't built yet.
- Don't write defensive code against impossible states. Trust the type system and Pydantic validation.
- Don't use global mutable state. Pass dependencies explicitly or use FastAPI's DI.
