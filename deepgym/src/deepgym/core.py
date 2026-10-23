"""Core DeepGym client — orchestrates sandbox lifecycle for RL evaluation."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, overload

from deepgym import __version__
from deepgym.benchmark_envs import _infer_environment_name, build_run_requests, route_single_run
from deepgym.exceptions import DeepGymError, SandboxError
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
    VerdictTotals,
)
from deepgym.sandbox import LocalExecutor, build_run_result, error_result, run_judge_in_sandbox
from deepgym.sandboxes import PROVIDERS, Provider, Sandbox, check_provider, open_sandbox
from deepgym.sandboxes.daytona import DaytonaDesktop, DaytonaSandbox
from deepgym.session import LocalBackend, SandboxBackend, Session, run_episode
from deepgym.traces import AgentMeta, EnvFingerprint, Provenance, TracedSession
from deepgym.traces.checkout import deepgym_checkout

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deepgym.agents.base import Agent, AgentResult, Budget


class DeepGym:
    """Provide the main client for running RL environments in sandboxed execution.

    Manages Daytona sandbox creation, verifier execution, and result
    aggregation for single runs, batches, and full evaluation suites.
    """

    def __init__(
        self,
        api_key: str | None = None,
        mode: Literal['auto', 'daytona', 'local'] | Provider = 'auto',
        *,
        confine_local_judge: bool = False,
    ) -> None:
        """Pick the backend: a sandbox provider, Daytona, or local subprocesses.

        Args:
            api_key: Daytona API key; DAYTONA_API_KEY if None.
            mode: 'auto' (Daytona if configured, else local), 'daytona', 'local', or a
                sandbox provider name.
            confine_local_judge: In local mode, run each whole judge inside the local
                sandbox too, for verifiers that might run code in-process (mutation
                testing); needs sandbox-exec or bwrap.
        """
        self._daytona = None
        local = LocalExecutor(confined=confine_local_judge)
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
            self._local_executor = local
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
                    self._local_executor = local
            else:
                logger.info('No DAYTONA_API_KEY set, using local executor')
                self._local_executor = local

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

        if self._local_executor is not None:
            return self._run_local(env, model_output)
        return self._run_in(self.open_sandbox(env), env, model_output)

    def open_sandbox(self, env: Environment) -> Sandbox:
        """Start a fresh sandbox for *env* on this client's provider or Daytona.

        Raises:
            SandboxError: In local mode, which has none, or if the sandbox can't be started.
        """
        if self._local_executor is not None:
            raise SandboxError(
                f'{type(env).__name__} needs a sandbox and local mode has none: '
                "use DeepGym(mode='docker'), 'apple', 'modal', 'e2b' or 'daytona'."
            )
        if self._provider:
            return open_sandbox(self._provider, env.image)
        return DaytonaSandbox(self._daytona, env)

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
                stderr=exec_result.stderr,
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
        return Session(env, SandboxBackend(box, desktop=desktop if env.desktop else None))

    @overload
    def run_episode(
        self, env: MultiTurnEnvironment, agent: Callable[[Observation], Action]
    ) -> tuple[Trajectory, RunResult]: ...

    @overload
    def run_episode(
        self, env: MultiTurnEnvironment, agent: Agent, *, budget: Budget | None = None
    ) -> tuple[AgentResult, RunResult]: ...

    def run_episode(
        self,
        env: MultiTurnEnvironment,
        agent: Callable[[Observation], Action] | Agent,
        *,
        budget: Budget | None = None,
    ) -> tuple[Trajectory | AgentResult, RunResult]:
        """Run one full episode of *env* driven by *agent* and score it.

        Args:
            env: The multi-turn or computer-use environment.
            agent: An agent harness, or a callable mapping observations to actions.
            budget: Harness limits; omitted uses Budget defaults.

        Returns:
            Tuple of (AgentResult, final_result) for a harness, or
            (trajectory, final_result) for a legacy callable. An agent that stopped through
            nothing the policy did (``provider_error``, ``setup_error``) gets no verdict;
            one the harness cut short (``output_limit``) is graded with ``truncated`` set.

        Raises:
            ValueError: If a budget is supplied to a legacy callable.
        """
        from deepgym.agents.base import Agent, Budget

        if isinstance(agent, Agent):
            with self.session(env) as session:
                initial = session.start()
                result = agent.run(session, initial, task=env.task, budget=budget or Budget())
                reason = result.termination_reason
                if reason in ('provider_error', 'setup_error'):  # any Agent, not only ours
                    session.abandon(reason, result.error or reason)
                grade = session.score()
                if reason == 'output_limit':  # the harness cut the run short
                    grade = grade.model_copy(update={'truncated': True})
                return result, grade
        if budget is not None:
            raise ValueError('budget requires an Agent harness with a run method')
        return run_episode(self.session(env), agent)

    def trace(
        self,
        env: MultiTurnEnvironment,
        agent: AgentMeta,
        *,
        variant: int | None = None,
        seed: int | None = None,
        reference: str | None = None,
    ) -> TracedSession:
        """Open a session for one episode of *env* that records its trace as it runs.

        The trace is written to ``~/.deepgym/traces/<trace_id>.jsonl`` (or
        ``$DEEPGYM_TRACES_DIR``) event by event, so a crash keeps everything up to it.

        Args:
            env: The multi-turn or computer-use environment.
            agent: The agent, model and sampling config acting in the episode.
            variant: Registry variant *env* was loaded with, for provenance.
            seed: Episode seed, for provenance.
            reference: Reference solution source; its hash goes in the provenance.

        Returns:
            An unstarted TracedSession: start(), step() until done, then score().

        Raises:
            SandboxError: If the sandbox cannot be created.

        Example:
            >>> with dg.trace(env, AgentMeta(name='react', model='Qwen/Qwen3-8B')) as tr:
            ...     obs = tr.start()
            ...     while not tr.done:
            ...         action, messages = policy(obs)
            ...         obs = tr.step(action, messages).observation
            ...     tr.score()
        """
        # Run the environment that was fingerprinted, even if the caller changes theirs.
        env = env.model_copy(deep=True)
        commit, dirty = deepgym_checkout()
        provenance = Provenance(
            env=EnvFingerprint.of(env, variant=variant, seed=seed, reference=reference),
            agent=agent,
            provider=self.backend,
            deepgym_version=__version__,
            deepgym_commit=commit,
            deepgym_dirty=dirty,
            started_at=datetime.now(timezone.utc),
        )
        session = self.session(env)
        booted = provenance.env.booted(session.booted_image())
        return TracedSession(session, env, provenance.model_copy(update={'env': booted}))

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
            Aggregated BatchResult (see :func:`build_batch_result`).
        """
        if not outputs:
            return build_batch_result([], 0.0)

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
                except Exception as exc:
                    logger.warning('Run %d failed: %s', idx, exc)
                    results[idx] = error_result(exc)

        # Every slot must be filled — assert rather than silently drop.
        assert all(r is not None for r in results), 'Not all batch slots were filled'
        final_results: list[RunResult] = results  # type: ignore[assignment]
        return build_batch_result(final_results, (time.perf_counter() - start) * 1000)

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
            Aggregated EvalResult; ``results`` follow the suite's order, named in ``env_names``.
        """
        from deepgym.registry import load_suite

        envs = load_suite(suite)
        if any(isinstance(env, MultiTurnEnvironment) for env in envs):
            raise DeepGymError(f'Suite {suite!r} needs an agent: use dg.run_episode() per env')
        names = [_infer_environment_name(env) for env in envs]
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [
                pool.submit(self.run, env, model_outputs[name]) if name in model_outputs else None
                for env, name in zip(envs, names)
            ]
            outcomes = [
                None if future is None else future.exception() or future.result()
                for future in futures
            ]
        return build_eval_result(suite, 'unknown', names, outcomes)


def build_eval_result(
    suite: str,
    model_name: str,
    names: Sequence[str],
    outcomes: Sequence[RunResult | BaseException | None],
) -> EvalResult:
    """Aggregate one outcome per suite environment into an EvalResult, keeping their order.

    Args:
        suite: Suite name.
        model_name: Identifier of the evaluated model.
        names: The environments' names, in suite order.
        outcomes: Per environment, its RunResult; the exception its run raised, which
            becomes a no-verdict result (see :func:`~deepgym.sandbox.error_result`); or
            None if the model gave no output for it, a plain 0.

    Returns:
        EvalResult whose ``results`` and ``env_names`` follow *names*. Its pass rate and
        mean score count an environment without a verdict as a failed 0; the ``judged_*``
        fields leave it out.
    """
    results = [_suite_entry(name, outcome) for name, outcome in zip(names, outcomes, strict=True)]
    passed, score, judged = _verdict_totals(results)
    total = len(results)
    return EvalResult(
        suite=suite,
        model_name=model_name,
        pass_rate=passed / max(total, 1),
        results=results,
        total=total,
        passed=passed,
        avg_score=score / max(total, 1),
        env_names=list(names),
        **judged.model_dump(),
    )


def build_batch_result(results: Sequence[RunResult], execution_time_ms: float) -> BatchResult:
    """Aggregate a batch's results.

    Args:
        results: One result per output, in output order.
        execution_time_ms: Wall-clock time of the whole batch.

    Returns:
        BatchResult whose pass count and mean score count a run without a verdict as a
        failed 0; the ``judged_*`` fields leave it out.
    """
    passed, score, judged = _verdict_totals(results)
    return BatchResult(
        results=list(results),
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        avg_score=score / max(len(results), 1),
        execution_time_ms=execution_time_ms,
        **judged.model_dump(),
    )


def _verdict_totals(results: Sequence[RunResult]) -> tuple[int, float, VerdictTotals]:
    """Return (passes, score sum, judged-only totals); a run without a verdict adds 0 to both."""
    judged = [r for r in results if r.has_verdict]
    passed = sum(1 for r in judged if r.passed)
    score = sum(r.score for r in judged)
    return (
        passed,
        score,
        VerdictTotals(
            no_verdict_count=len(results) - len(judged),
            judged_count=len(judged),
            judged_pass_rate=passed / max(len(judged), 1),
            judged_mean_score=score / max(len(judged), 1),
        ),
    )


def _suite_entry(name: str, outcome: RunResult | BaseException | None) -> RunResult:
    """Return suite environment *name*'s result from its outcome (see build_eval_result)."""
    if outcome is None:
        logger.warning('No model output for environment %r, skipping', name)
        return RunResult(
            score=0.0,
            passed=False,
            output=f'No model output provided for {name}',
            stderr='',
            exit_code=1,
            execution_time_ms=0.0,
            sandbox_id='skipped',
        )
    if isinstance(outcome, BaseException):
        logger.warning('Eval run for %r failed: %s', name, outcome)
        return error_result(outcome)
    return outcome
