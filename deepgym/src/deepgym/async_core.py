"""Asynchronous DeepGym client.

NOTE: Daytona SDK import names may vary across versions. Verify the import
paths against the installed daytona-sdk version.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from deepgym.exceptions import DeepGymError, SandboxError, TimeoutError, VerifierError
from deepgym.models import (
    BatchResult,
    Environment,
    EvalResult,
    RunResult,
    VerifierResult,
)
from deepgym.sandbox import (
    AsyncLocalExecutor,
    _extract_json_line,
    _shell_escape,
    _validate_env_vars,
    build_run_result,
    is_standalone_verifier,
)
from deepgym.verifier_template import wrap_verifier

if TYPE_CHECKING:
    from daytona import AsyncDaytona, Sandbox

logger = logging.getLogger(__name__)

_VERIFIER_PATH = '/home/user/verifier.py'
_SOLUTION_PATH = '/home/user/solution.py'
_TEST_CASES_PATH = '/home/user/test_cases.json'


class AsyncDeepGym:
    """Asynchronous client for running RL evaluations in sandboxed environments.

    Provides the same interface as :class:`~deepgym.core.DeepGym` but uses
    ``async``/``await`` throughout.

    Args:
        api_key: Optional API key for authentication.
        api_url: If set, use HTTP mode and talk to a remote DeepGym server.
        default_timeout: Default execution timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        default_timeout: int = 30,
        mode: Literal['auto', 'daytona', 'local'] = 'auto',
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._default_timeout = default_timeout
        self._daytona: AsyncDaytona | None = None
        self._local_executor: AsyncLocalExecutor | None = None

        # Read env vars for configuration.
        daytona_api_key = api_key or os.getenv('DAYTONA_API_KEY')
        daytona_api_url = os.getenv('DAYTONA_API_URL')

        if api_url is not None:
            logger.info('Initialising AsyncDeepGym in HTTP mode: %s', api_url)
        elif mode == 'local':
            logger.info('Initialising AsyncDeepGym in local mode (no Daytona)')
            self._local_executor = AsyncLocalExecutor()
        elif mode == 'daytona':
            self._daytona = self._init_daytona(daytona_api_key, daytona_api_url)
        else:
            # Auto mode: try Daytona, fall back to local.
            if daytona_api_key:
                try:
                    self._daytona = self._init_daytona(daytona_api_key, daytona_api_url)
                    logger.info('Initialised AsyncDeepGym with Daytona backend')
                except (SandboxError, DeepGymError):
                    logger.warning('Daytona init failed, falling back to local mode')
                    self._local_executor = AsyncLocalExecutor()
            else:
                logger.info('No DAYTONA_API_KEY set, using async local executor')
                self._local_executor = AsyncLocalExecutor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, env: Environment, model_output: str) -> RunResult:
        """Execute a model's output against an environment asynchronously.

        Lifecycle mirrors :meth:`DeepGym.run` but uses async Daytona calls.

        Args:
            env: The environment specification.
            model_output: The model-generated solution source code.

        Returns:
            A RunResult with score, timing, and execution details.

        Raises:
            SandboxError: If sandbox operations fail.
            VerifierError: If verifier output is invalid.
            TimeoutError: If execution exceeds the timeout.
        """
        if self._api_url is not None:
            return await self._run_http(env, model_output)

        if self._local_executor is not None:
            return await self._run_local(env, model_output)

        if self._daytona is None:
            raise SandboxError('AsyncDaytona client not initialised')

        sandbox: Sandbox | None = None
        start = time.perf_counter()

        try:
            sandbox = await self._create_sandbox(env)
            sandbox_id = getattr(sandbox, 'id', 'unknown')

            await self._setup_sandbox(sandbox, env, model_output)

            timeout = env.timeout or self._default_timeout
            verifier_result = await self._execute_verifier(
                sandbox,
                timeout,
                has_test_cases=env.test_cases is not None,
                env_vars=env.env_vars,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            return build_run_result(verifier_result, elapsed_ms, str(sandbox_id))

        except DeepGymError:
            raise

        except Exception as exc:
            raise SandboxError(f'Unexpected error during async run: {exc}') from exc

        finally:
            if sandbox is not None and self._daytona is not None:
                await self._cleanup_sandbox(sandbox)

    async def run_batch(
        self,
        env: Environment,
        outputs: Sequence[str],
        max_parallel: int = 10,
    ) -> BatchResult:
        """Run multiple solutions concurrently with a semaphore-based limit.

        Args:
            env: The environment specification.
            outputs: List of model-generated solutions.
            max_parallel: Maximum number of concurrent sandbox executions.

        Returns:
            Aggregated BatchResult.
        """
        if not outputs:
            return BatchResult(
                results=[], total=0, passed=0, failed=0, avg_score=0.0, execution_time_ms=0.0
            )

        start = time.perf_counter()
        semaphore = asyncio.Semaphore(max_parallel)

        async def _guarded_run(output: str) -> RunResult:
            async with semaphore:
                return await self.run(env, output)

        tasks = [asyncio.ensure_future(_guarded_run(out)) for out in outputs]
        settled = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[RunResult] = []
        for i, item in enumerate(settled):
            if isinstance(item, VerifierError | TimeoutError):
                # Model's code caused verifier failure or timeout.
                error_type = 'timeout' if isinstance(item, TimeoutError) else 'verifier_error'
                logger.warning('Async run %d failed (%s): %s', i, error_type, item)
                results.append(
                    RunResult(
                        score=0.0,
                        passed=False,
                        output=str(item),
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0.0,
                        sandbox_id='error',
                        error_type=error_type,
                    )
                )
            elif isinstance(item, SandboxError):
                # Sandbox infrastructure failure.
                logger.error('Async run %d failed (sandbox_error): %s', i, item)
                results.append(
                    RunResult(
                        score=0.0,
                        passed=False,
                        output=str(item),
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0.0,
                        sandbox_id='error',
                        error_type='sandbox_error',
                    )
                )
            elif isinstance(item, BaseException):
                # Unknown failure.
                logger.error('Async run %d failed (unknown_error): %s', i, item)
                results.append(
                    RunResult(
                        score=0.0,
                        passed=False,
                        output=str(item),
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0.0,
                        sandbox_id='error',
                        error_type='unknown_error',
                    )
                )
            else:
                results.append(item)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        passed = sum(1 for r in results if r.passed)

        return BatchResult(
            results=results,
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            avg_score=sum(r.score for r in results) / max(len(results), 1),
            execution_time_ms=elapsed_ms,
        )

    async def eval(
        self,
        suite: str,
        model_outputs: dict[str, str] | None = None,
        *,
        model: Callable[[str], Awaitable[str]] | None = None,
        max_parallel: int = 100,
    ) -> EvalResult:
        """Evaluate against a suite of environments.

        Provide either *model_outputs* (a dict mapping environment name to
        solution code, matching the sync :meth:`DeepGym.eval` signature) or
        *model* (an async callable that generates solutions from task prompts).

        Args:
            suite: Name of the evaluation suite to load.
            model_outputs: Mapping of environment name to model solution code.
                Mutually exclusive with *model*.
            model: An async callable that takes a task prompt and returns a
                solution string. Mutually exclusive with *model_outputs*.
            max_parallel: Maximum number of concurrent evaluations.

        Returns:
            An EvalResult with per-task results and aggregate statistics.

        Raises:
            ValueError: If neither or both of *model_outputs* and *model*
                are provided.
        """
        if model_outputs is None and model is None:
            raise ValueError('Provide either model_outputs or model')
        if model_outputs is not None and model is not None:
            raise ValueError('Provide either model_outputs or model, not both')

        environments = self._load_suite(suite)
        semaphore = asyncio.Semaphore(max_parallel)

        if model is not None:
            return await self._eval_with_model(suite, environments, model, semaphore)
        assert model_outputs is not None
        return await self._eval_with_outputs(suite, environments, model_outputs, semaphore)

    async def _eval_with_model(
        self,
        suite: str,
        environments: list[Environment],
        model: Callable[[str], Awaitable[str]],
        semaphore: asyncio.Semaphore,
    ) -> EvalResult:
        """Evaluate using an async model callable that generates solutions.

        Args:
            suite: Suite name for the result metadata.
            environments: List of environments to evaluate.
            model: Async callable producing solution code from a task prompt.
            semaphore: Concurrency limiter.

        Returns:
            Aggregated EvalResult.
        """

        async def _eval_one(env: Environment) -> RunResult:
            async with semaphore:
                output = await model(env.task)
                return await self.run(env, output)

        tasks = [asyncio.ensure_future(_eval_one(e)) for e in environments]
        settled = await asyncio.gather(*tasks, return_exceptions=True)

        results = self._collect_eval_results(settled)
        return self._build_eval_result(suite, results, getattr(model, '__name__', 'unknown'))

    async def _eval_with_outputs(
        self,
        suite: str,
        environments: list[Environment],
        model_outputs: dict[str, str],
        semaphore: asyncio.Semaphore,
    ) -> EvalResult:
        """Evaluate using pre-computed model outputs (mirrors sync eval).

        Args:
            suite: Suite name for the result metadata.
            environments: List of environments to evaluate.
            model_outputs: Mapping of environment name to solution code.
            semaphore: Concurrency limiter.

        Returns:
            Aggregated EvalResult.
        """

        async def _eval_one(env: Environment, solution: str) -> RunResult:
            async with semaphore:
                return await self.run(env, solution)

        tasks: list[asyncio.Task[RunResult]] = []
        skipped: list[RunResult] = []

        for env in environments:
            env_name = env.verifier_path.parent.name if env.verifier_path else ''
            solution = model_outputs.get(env_name)
            if solution is None:
                logger.warning('No model output for environment %r, skipping', env_name)
                skipped.append(
                    RunResult(
                        score=0.0,
                        passed=False,
                        output=f'No model output provided for {env_name}',
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0.0,
                        sandbox_id='skipped',
                    )
                )
                continue
            tasks.append(asyncio.ensure_future(_eval_one(env, solution)))

        settled = await asyncio.gather(*tasks, return_exceptions=True)
        results = skipped + self._collect_eval_results(settled)
        return self._build_eval_result(suite, results, 'unknown')

    @staticmethod
    def _collect_eval_results(
        settled: list[RunResult | BaseException],
    ) -> list[RunResult]:
        """Convert gather results into a list of RunResult, logging errors.

        Args:
            settled: Raw results from asyncio.gather with return_exceptions=True.

        Returns:
            List of RunResult (failures converted to zero-score results).
        """
        results: list[RunResult] = []
        for i, item in enumerate(settled):
            if isinstance(item, VerifierError | TimeoutError):
                # Model's code caused verifier failure or timeout.
                error_type = 'timeout' if isinstance(item, TimeoutError) else 'verifier_error'
                logger.warning('Eval task %d failed (%s): %s', i, error_type, item)
                results.append(
                    RunResult(
                        score=0.0,
                        passed=False,
                        output=str(item),
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0.0,
                        sandbox_id='error',
                        error_type=error_type,
                    )
                )
            elif isinstance(item, SandboxError):
                # Sandbox infrastructure failure.
                logger.error('Eval task %d failed (sandbox_error): %s', i, item)
                results.append(
                    RunResult(
                        score=0.0,
                        passed=False,
                        output=str(item),
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0.0,
                        sandbox_id='error',
                        error_type='sandbox_error',
                    )
                )
            elif isinstance(item, BaseException):
                # Unknown failure.
                logger.error('Eval task %d failed (unknown_error): %s', i, item)
                results.append(
                    RunResult(
                        score=0.0,
                        passed=False,
                        output=str(item),
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0.0,
                        sandbox_id='error',
                        error_type='unknown_error',
                    )
                )
            else:
                results.append(item)
        return results

    @staticmethod
    def _build_eval_result(
        suite: str,
        results: list[RunResult],
        model_name: str,
    ) -> EvalResult:
        """Build an EvalResult from collected run results.

        Args:
            suite: Suite name.
            results: List of per-environment RunResult objects.
            model_name: Name to include in the result metadata.

        Returns:
            Aggregated EvalResult.
        """
        passed = sum(1 for r in results if r.passed)
        total = len(results)

        return EvalResult(
            suite=suite,
            model_name=model_name,
            pass_rate=passed / max(total, 1),
            results=results,
            total=total,
            passed=passed,
            avg_score=sum(r.score for r in results) / max(total, 1),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _run_local(self, env: Environment, model_output: str) -> RunResult:
        """Execute using the async local subprocess executor."""
        assert self._local_executor is not None

        if env.verifier_path is not None:
            verifier_source = env.verifier_path.read_text(encoding='utf-8')
        else:
            verifier_source = env.verifier_code

        # Standalone scripts run directly; simple function bodies get wrapped.
        if is_standalone_verifier(verifier_source):
            wrapped = verifier_source
        else:
            wrapped = wrap_verifier(verifier_source)

        timeout = env.timeout or self._default_timeout
        start = time.perf_counter()
        exec_result = await self._local_executor.run(
            verifier_code=wrapped,
            solution_code=model_output,
            test_cases=env.test_cases,
            timeout=timeout,
            env_vars=env.env_vars,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return build_run_result(
            exec_result.verifier_result,
            elapsed_ms,
            'local',
            stderr=exec_result.stderr,
            exit_code=exec_result.exit_code,
        )

    @staticmethod
    def _init_daytona(api_key: str | None, api_url: str | None = None) -> AsyncDaytona:
        """Initialise the async Daytona SDK client."""
        try:
            from daytona import AsyncDaytona, DaytonaConfig
        except ImportError as exc:
            raise SandboxError(
                "daytona-sdk is required. Install with: pip install 'deepgym[daytona]'"
            ) from exc

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs['api_key'] = api_key
        if api_url:
            kwargs['api_url'] = api_url
        config = DaytonaConfig(**kwargs) if kwargs else None
        return AsyncDaytona(config) if config else AsyncDaytona()

    async def _create_sandbox(self, env: Environment) -> Sandbox:
        """Create an ephemeral sandbox from snapshot or image."""
        try:
            from daytona import (
                CreateSandboxFromImageParams,
                CreateSandboxFromSnapshotParams,
                Image,
            )
        except ImportError as exc:
            raise SandboxError('daytona-sdk is required but could not be imported.') from exc

        assert self._daytona is not None

        try:
            if env.snapshot:
                params = CreateSandboxFromSnapshotParams(
                    snapshot=env.snapshot,
                    ephemeral=True,
                )
                return await self._daytona.create(params)
            else:
                params = CreateSandboxFromImageParams(
                    image=Image.debian_slim('3.12'),
                    ephemeral=True,
                )
                return await self._daytona.create(params)
        except Exception as exc:
            raise SandboxError(f'Failed to create async sandbox: {exc}') from exc

    async def _setup_sandbox(
        self,
        sandbox: Sandbox,
        env: Environment,
        model_output: str,
    ) -> None:
        """Upload verifier, solution, and test cases."""
        try:
            verifier_source = (
                env.verifier_path.read_text(encoding='utf-8')
                if env.verifier_path is not None
                else env.verifier_code
            )

            # Wrap simple verifiers that don't already produce JSON protocol output.
            if is_standalone_verifier(verifier_source):
                ready_code = verifier_source
            else:
                ready_code = wrap_verifier(verifier_source)

            await sandbox.fs.upload_file(ready_code.encode('utf-8'), _VERIFIER_PATH)
            await sandbox.fs.upload_file(model_output.encode('utf-8'), _SOLUTION_PATH)

            if env.test_cases is not None:
                payload = json.dumps(env.test_cases, ensure_ascii=False)
                await sandbox.fs.upload_file(payload.encode('utf-8'), _TEST_CASES_PATH)

        except Exception as exc:
            raise SandboxError(f'Failed to set up async sandbox: {exc}') from exc

    async def _execute_verifier(
        self,
        sandbox: Sandbox,
        timeout: int,
        has_test_cases: bool = False,
        env_vars: dict[str, str] | None = None,
    ) -> VerifierResult:
        """Run the verifier and parse its JSON output.

        Args:
            sandbox: A sandbox with verifier and solution already uploaded.
            timeout: Maximum execution time in seconds.
            has_test_cases: Whether test_cases.json was uploaded.
            env_vars: Optional env vars to inject into the verifier command.

        Returns:
            Parsed VerifierResult.

        Raises:
            VerifierError: If verifier output is not valid JSON.
            TimeoutError: If execution exceeds timeout.
        """
        _validate_env_vars(env_vars)

        env_prefix = ''
        if env_vars:
            env_prefix = ' '.join(f'{k}={_shell_escape(v)}' for k, v in env_vars.items()) + ' '

        cmd = f'{env_prefix}python {_VERIFIER_PATH} {_SOLUTION_PATH}'
        if has_test_cases:
            cmd += f' {_TEST_CASES_PATH}'

        try:
            response = await sandbox.process.exec(cmd, timeout=timeout)
        except Exception as exc:
            err_str = str(exc).lower()
            if 'timeout' in err_str or 'timed out' in err_str:
                raise TimeoutError(f'Verifier execution exceeded {timeout}s timeout') from exc
            raise VerifierError(f'Verifier execution failed: {exc}') from exc

        stdout = response.result.strip() if hasattr(response, 'result') else ''
        exit_code = getattr(response, 'exit_code', -1)

        # Exit codes 0 (passed) and 1 (failed) are normal verifier outcomes.
        # Parse JSON regardless of exit code; only raise if no valid JSON found.
        json_str = _extract_json_line(stdout)
        if json_str is None:
            stderr = getattr(response, 'stderr', '') or ''
            raise VerifierError(
                f'Verifier did not produce valid JSON. '
                f'exit_code={exit_code}, stdout: {stdout!r}, stderr: {stderr!r}'
            )

        try:
            return VerifierResult.model_validate_json(json_str)
        except Exception as exc:
            raise VerifierError(f'Failed to parse verifier JSON: {exc}. Raw: {json_str!r}') from exc

    async def _cleanup_sandbox(self, sandbox: Sandbox) -> None:
        """Safely destroy a sandbox, swallowing errors."""
        try:
            assert self._daytona is not None
            await self._daytona.delete(sandbox)
        except Exception:
            logger.debug(
                'Failed to clean up async sandbox %s (ignored)',
                getattr(sandbox, 'id', '?'),
            )

    async def _run_http(self, env: Environment, model_output: str) -> RunResult:
        """Execute a run via the remote DeepGym API server."""
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f'{self._api_url}/v1/run',
                json={
                    'environment': env.model_dump(mode='json'),
                    'model_output': model_output,
                },
                headers=({'X-API-Key': self._api_key} if self._api_key else {}),
                timeout=env.timeout + 30,
            )
            resp.raise_for_status()
            return RunResult.model_validate(resp.json())

    @staticmethod
    def _load_suite(suite: str) -> list[Environment]:
        """Load an evaluation suite by name from the built-in registry.

        Args:
            suite: Suite name ('easy', 'medium', 'hard', 'all', or a family name).

        Returns:
            List of Environment objects matching the suite criteria.
        """
        from deepgym.registry import load_suite

        return load_suite(suite)
