"""Core DeepGym client — orchestrates sandbox lifecycle for RL evaluation."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from deepgym.exceptions import DeepGymError, SandboxError, TimeoutError, VerifierError
from deepgym.models import BatchResult, Environment, EvalResult, RunResult
from deepgym.sandbox import (
    LocalExecutor,
    build_run_result,
    cleanup_sandbox,
    create_sandbox,
    execute_verifier,
    is_standalone_verifier,
    setup_sandbox,
)
from deepgym.verifier_template import wrap_verifier

logger = logging.getLogger(__name__)


class DeepGym:
    """Provide the main client for running RL environments in sandboxed execution.

    Manages Daytona sandbox creation, verifier execution, and result
    aggregation for single runs, batches, and full evaluation suites.
    """

    def __init__(
        self,
        api_key: str | None = None,
        mode: Literal['auto', 'daytona', 'local'] = 'auto',
    ) -> None:
        self._daytona = None
        self._local_executor: LocalExecutor | None = None

        # Read env vars for configuration.
        daytona_api_key = api_key or os.getenv('DAYTONA_API_KEY')
        daytona_api_url = os.getenv('DAYTONA_API_URL')

        if mode == 'local':
            logger.info('Initialising DeepGym in local mode (no Daytona)')
            self._local_executor = LocalExecutor()
        elif mode == 'daytona':
            self._daytona = self._init_daytona(daytona_api_key, daytona_api_url)
        else:
            # Auto mode: try Daytona, fall back to local.
            if daytona_api_key:
                try:
                    self._daytona = self._init_daytona(daytona_api_key, daytona_api_url)
                    logger.info('Initialised DeepGym with Daytona backend')
                except DeepGymError:
                    logger.warning('Daytona init failed, falling back to local mode')
                    self._local_executor = LocalExecutor()
            else:
                logger.info('No DAYTONA_API_KEY set, using local executor')
                self._local_executor = LocalExecutor()

    @staticmethod
    def _init_daytona(api_key: str | None, api_url: str | None = None):
        """Initialise the Daytona SDK client."""
        try:
            from daytona import Daytona, DaytonaConfig
        except ImportError as exc:
            raise DeepGymError(
                "daytona-sdk is required. Install it with: pip install 'deepgym[daytona]'"
            ) from exc

        kwargs = {}
        if api_key:
            kwargs['api_key'] = api_key
        if api_url:
            kwargs['api_url'] = api_url
        config = DaytonaConfig(**kwargs) if kwargs else None
        return Daytona(config) if config else Daytona()

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run(self, env: Environment, model_output: str) -> RunResult:
        """Run a model's output against an environment's verifier in a sandbox.

        Args:
            env: The environment specification.
            model_output: Model-generated solution source code.

        Returns:
            A RunResult with score, pass/fail, stdout, stderr, timing, and
            sandbox id.
        """
        if self._local_executor is not None:
            return self._run_local(env, model_output)
        return self._run_daytona(env, model_output)

    def _run_local(self, env: Environment, model_output: str) -> RunResult:
        """Execute using the local subprocess executor."""
        assert self._local_executor is not None

        # Resolve verifier source.
        if env.verifier_path is not None:
            verifier_source = env.verifier_path.read_text(encoding='utf-8')
        else:
            verifier_source = env.verifier_code

        # Standalone scripts run directly; simple function bodies get wrapped.
        if is_standalone_verifier(verifier_source):
            ready_code = verifier_source
        else:
            ready_code = wrap_verifier(verifier_source)

        start = time.perf_counter()
        exec_result = self._local_executor.run(
            verifier_code=ready_code,
            solution_code=model_output,
            test_cases=env.test_cases,
            timeout=env.timeout,
            env_vars=env.env_vars,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return build_run_result(
            exec_result.verifier_result,
            elapsed_ms,
            'local',
            stderr=exec_result.stderr,
            exit_code=exec_result.exit_code,
        )

    def _run_daytona(self, env: Environment, model_output: str) -> RunResult:
        """Execute using the Daytona sandbox."""
        sandbox = create_sandbox(env, self._daytona)
        sandbox_id = getattr(sandbox, 'id', 'unknown')
        start = time.perf_counter()

        try:
            setup_sandbox(sandbox, env, model_output)
            verifier_result = execute_verifier(
                sandbox,
                env.timeout,
                has_test_cases=env.test_cases is not None,
                env_vars=env.env_vars,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            return build_run_result(verifier_result, elapsed_ms, sandbox_id)
        finally:
            cleanup_sandbox(sandbox, self._daytona)

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def run_batch(
        self,
        env: Environment,
        outputs: Sequence[str],
        *,
        max_parallel: int = 10,
    ) -> BatchResult:
        """Run multiple solutions against one environment in parallel.

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
        results: list[RunResult | None] = [None] * len(outputs)

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(self.run, env, output): i for i, output in enumerate(outputs)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except (VerifierError, TimeoutError) as exc:
                    # Model's code caused verifier failure or timeout.
                    error_type = 'timeout' if isinstance(exc, TimeoutError) else 'verifier_error'
                    logger.warning('Run %d failed (%s): %s', idx, error_type, exc)
                    results[idx] = RunResult(
                        score=0.0,
                        passed=False,
                        output=str(exc),
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0,
                        sandbox_id='error',
                        error_type=error_type,
                    )
                except SandboxError as exc:
                    # Sandbox infrastructure failure.
                    logger.error('Run %d failed (sandbox_error): %s', idx, exc)
                    results[idx] = RunResult(
                        score=0.0,
                        passed=False,
                        output=str(exc),
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0,
                        sandbox_id='error',
                        error_type='sandbox_error',
                    )
                except Exception as exc:
                    # Unknown failure.
                    logger.error('Run %d failed (unknown_error): %s', idx, exc)
                    results[idx] = RunResult(
                        score=0.0,
                        passed=False,
                        output=str(exc),
                        stderr='',
                        exit_code=1,
                        execution_time_ms=0,
                        sandbox_id='error',
                        error_type='unknown_error',
                    )

        # Every slot must be filled — assert rather than silently drop.
        assert all(r is not None for r in results), 'Not all batch slots were filled'
        final_results: list[RunResult] = results  # type: ignore[assignment]

        elapsed_ms = (time.perf_counter() - start) * 1000
        passed = sum(1 for r in final_results if r.passed)
        avg_score = (
            sum(r.score for r in final_results) / len(final_results) if final_results else 0.0
        )

        return BatchResult(
            results=final_results,
            total=len(final_results),
            passed=passed,
            failed=len(final_results) - passed,
            avg_score=avg_score,
            execution_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Evaluation suite
    # ------------------------------------------------------------------

    def eval(
        self,
        suite: str,
        model_outputs: dict[str, str],
        *,
        max_parallel: int = 100,
    ) -> EvalResult:
        """Evaluate model outputs against a named suite of environments.

        Args:
            suite: Suite name ('easy', 'medium', 'hard', 'all', or a family name).
            model_outputs: Mapping of environment name to model solution code.
            max_parallel: Maximum concurrent sandbox executions.

        Returns:
            Aggregated EvalResult with per-environment breakdown.
        """
        from deepgym.registry import load_suite

        envs = load_suite(suite)
        results: list[RunResult] = []

        # Build a lookup from environment directory name to solution.
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {}
            for env in envs:
                # Match by verifier_path parent directory name.
                env_name = env.verifier_path.parent.name if env.verifier_path else ''
                solution = model_outputs.get(env_name)
                if solution is None:
                    logger.warning('No model output for environment %r, skipping', env_name)
                    results.append(
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
                futures[pool.submit(self.run, env, solution)] = env_name

            for future in as_completed(futures):
                env_name = futures[future]
                try:
                    results.append(future.result())
                except DeepGymError as exc:
                    logger.warning('Eval run for %r failed: %s', env_name, exc)
                    results.append(
                        RunResult(
                            score=0.0,
                            passed=False,
                            output='',
                            stderr=str(exc),
                            exit_code=1,
                            execution_time_ms=0.0,
                            sandbox_id='error',
                        )
                    )

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        return EvalResult(
            suite=suite,
            model_name='unknown',
            pass_rate=passed / max(total, 1),
            results=results,
            total=total,
            passed=passed,
            avg_score=sum(r.score for r in results) / max(total, 1),
        )
