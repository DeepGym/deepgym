"""Core DeepGym client — orchestrates sandbox lifecycle for RL evaluation."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from deepgym.benchmark_envs import build_run_requests, route_single_run
from deepgym.exceptions import (
    DeepGymError,
    SandboxError,
    TimeoutError,
    VerifierError,
    error_type_for,
)
from deepgym.judge import JudgeJob
from deepgym.models import (
    Action,
    BatchResult,
    Environment,
    EvalResult,
    MultiTurnEnvironment,
    Observation,
    RunResult,
    Trajectory,
)
from deepgym.sandbox import LocalExecutor, build_run_result, run_judge_in_sandbox
from deepgym.sandboxes import PROVIDERS, Provider, Sandbox, check_provider, open_sandbox
from deepgym.sandboxes.daytona import DaytonaDesktop, DaytonaSandbox
from deepgym.session import LocalBackend, SandboxBackend, Session, run_episode

logger = logging.getLogger(__name__)


class DeepGym:
    """Provide the main client for running RL environments in sandboxed execution.

    Manages Daytona sandbox creation, verifier execution, and result
    aggregation for single runs, batches, and full evaluation suites.
    """

    def __init__(
        self,
        api_key: str | None = None,
        mode: Literal['auto', 'daytona', 'local'] | Provider = 'auto',
    ) -> None:
        self._daytona = None
        self._local_executor: LocalExecutor | None = None
        self._provider: Provider | None = mode if mode in PROVIDERS else None

        # Read env vars for configuration.
        daytona_api_key = api_key or os.getenv('DAYTONA_API_KEY')
        daytona_api_url = os.getenv('DAYTONA_API_URL')

        if self._provider:
            check_provider(self._provider)
            logger.info('Initialising DeepGym with %s sandboxes', self._provider)
        elif mode == 'local':
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

    @property
    def backend(self) -> str:
        """Where runs execute: 'local', 'daytona', or the sandbox provider's name."""
        if self._provider:
            return self._provider
        return 'local' if self._local_executor is not None else 'daytona'

    @staticmethod
    def _init_daytona(api_key: str | None, api_url: str | None = None):
        """Initialise the Daytona SDK client."""
        try:
            from daytona import Daytona, DaytonaConfig
        except ImportError as exc:
            raise DeepGymError(
                "daytona is required. Install it with: pip install 'deepgym[daytona]'"
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

    def run(self, env: Environment, model_output: str, **kwargs: object) -> RunResult:
        """Run a model's output against an environment's verifier in a sandbox.

        Args:
            env: The environment specification.
            model_output: Model-generated solution source code.

        Returns:
            A RunResult with score, pass/fail, stdout, stderr, timing, and
            sandbox id.
        """
        if isinstance(env, MultiTurnEnvironment):
            raise DeepGymError('Multi-turn environments need an agent: use dg.run_episode()')
        env = route_single_run(env, kwargs)
        if hasattr(env, 'run_with_deepgym'):
            return env.run_with_deepgym(self, model_output, **kwargs)  # type: ignore[return-value]

        if self._provider:
            return self._run_in(open_sandbox(self._provider, env.image), env, model_output)
        if self._local_executor is not None:
            return self._run_local(env, model_output)
        return self._run_in(DaytonaSandbox(self._daytona, env), env, model_output)

    def _run_local(self, env: Environment, model_output: str) -> RunResult:
        """Execute using the local subprocess executor."""
        assert self._local_executor is not None
        start = time.perf_counter()
        exec_result = self._local_executor.run(JudgeJob.for_environment(env, model_output))
        return build_run_result(
            exec_result.verifier_result,
            (time.perf_counter() - start) * 1000,
            'local',
            stderr=exec_result.stderr,
            exit_code=exec_result.exit_code,
        )

    def _run_in(self, sandbox: Sandbox, env: Environment, model_output: str) -> RunResult:
        """Execute in a fresh *sandbox*, deleted afterwards."""
        start = time.perf_counter()
        try:
            exec_result = run_judge_in_sandbox(sandbox, JudgeJob.for_environment(env, model_output))
            return build_run_result(
                exec_result.verifier_result,
                (time.perf_counter() - start) * 1000,
                sandbox.id,
                exit_code=exec_result.exit_code,
            )
        finally:
            sandbox.delete()

    # ------------------------------------------------------------------
    # Multi-turn episodes
    # ------------------------------------------------------------------

    def session(self, env: MultiTurnEnvironment) -> Session:
        """Open a live session for one episode of *env* on this client's backend.

        Args:
            env: The multi-turn or computer-use environment.

        Returns:
            An unstarted Session. Call start(), then step() until done, then score().

        Raises:
            SandboxError: If the sandbox cannot be created.
        """
        if self._provider:
            return Session(env, SandboxBackend(open_sandbox(self._provider, env.image)))
        if self._local_executor is not None:
            return Session(env, LocalBackend())
        box = DaytonaSandbox(self._daytona, env)
        desktop = DaytonaDesktop(box, env.timeout_per_step, env.setup_timeout)
        return Session(env, SandboxBackend(box, desktop if env.desktop else None))

    def run_episode(
        self, env: MultiTurnEnvironment, agent: Callable[[Observation], Action]
    ) -> tuple[Trajectory, RunResult]:
        """Run one full episode of *env* driven by *agent* and score it.

        Args:
            env: The multi-turn or computer-use environment.
            agent: Callable mapping an observation to the next action.

        Returns:
            Tuple of (trajectory, final_result).
        """
        return run_episode(self.session(env), agent)

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def run_batch(
        self,
        env: Environment,
        outputs: Sequence[str],
        *,
        max_parallel: int = 10,
        **kwargs: object,
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

        run_requests = build_run_requests(env, outputs, dict(kwargs))

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(self.run, request.env, request.output, **request.kwargs): i
                for i, request in enumerate(run_requests)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except (VerifierError, TimeoutError) as exc:
                    # Model's code caused verifier failure or timeout.
                    error_type = error_type_for(exc)
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
        if any(isinstance(env, MultiTurnEnvironment) for env in envs):
            raise DeepGymError(f'Suite {suite!r} needs an agent: use dg.run_episode() per env')
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
