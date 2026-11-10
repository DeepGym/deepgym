"""Asynchronous DeepGym client."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import httpx

from deepgym.benchmark_envs import (
    ResolvedRun,
    _infer_environment_name,
    build_run_requests,
    route_single_run,
)
from deepgym.core import build_batch_result, build_eval_result
from deepgym.exceptions import DeepGymError, SandboxError
from deepgym.judge import DEADLINE_GRACE_S, SANDBOX_SETUP, JudgeJob, sandbox_judge, sandbox_uploads
from deepgym.models import (
    BatchResult,
    Environment,
    EvalResult,
    MultiTurnEnvironment,
    RunResult,
)
from deepgym.sandbox import (
    AsyncLocalExecutor,
    ExecutionResult,
    _validate_env_vars,
    build_run_result,
    error_result,
    judge_timeout,
    local_run_result,
    parse_wrapped_output,
    wrap_stderr,
)
from deepgym.sandboxes import PROVIDERS, Provider, SandboxTimeoutError
from deepgym.sandboxes.daytona import create_params
from deepgym.sandboxes.daytona.sandbox import LEAK_SWEEP_EVERY_S, LEAK_SWEEP_S, exec_async

if TYPE_CHECKING:
    from daytona import AsyncDaytona, Sandbox

    from deepgym.core import DeepGym

logger = logging.getLogger(__name__)


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
        mode: Literal['auto', 'daytona', 'local'] | Provider = 'auto',
    ) -> None:
        self._mode = mode
        self._api_key = api_key
        self._api_url = api_url
        self._default_timeout = default_timeout
        self._daytona: AsyncDaytona | None = None
        # Sweeps for sandboxes a failed create may still leave; aclose() waits for them.
        self._sweeps: set[asyncio.Task[bool | None]] = set()
        self._local_executor: AsyncLocalExecutor | None = None
        self._sync: DeepGym | None = None

        # Read env vars for configuration.
        daytona_api_key = api_key or os.getenv('DAYTONA_API_KEY')
        daytona_api_url = os.getenv('DAYTONA_API_URL')

        if api_url is not None:
            logger.info('Initialising AsyncDeepGym in HTTP mode: %s', api_url)
        elif mode in PROVIDERS:
            # Provider runs go through the sync client in worker threads; creating it
            # now fails fast if the provider can't be used here.
            self._sync_client()
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

    async def __aenter__(self) -> AsyncDeepGym:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Wait for any sweep still deleting a sandbox a failed create left (up to
        LEAK_SWEEP_S), then close the Daytona client's connections."""
        if self._sweeps:
            await asyncio.gather(*self._sweeps, return_exceptions=True)
        if self._daytona is not None:
            await self._daytona.close()
            self._daytona = None

    async def run(self, env: Environment, model_output: str, **kwargs: Any) -> RunResult:
        """Execute a model's output against an environment asynchronously.

        Lifecycle mirrors :meth:`DeepGym.run` but uses async Daytona calls.

        Args:
            env: The environment specification.
            model_output: The model-generated solution source code.
            **kwargs: Per-sample routing/task metadata for benchmark-backed and
                mixed environments (same as :meth:`DeepGym.run`).

        Returns:
            A RunResult with score, timing, and execution details.

        Raises:
            SandboxError: If sandbox operations fail.
            VerifierError: If verifier output is invalid.
            TimeoutError: If execution exceeds the timeout.
            DeepGymError: If *env* is multi-turn (it needs an agent, not one output),
                or benchmark-backed while using the HTTP API.
        """
        if isinstance(env, MultiTurnEnvironment):
            raise DeepGymError('Multi-turn environments need an agent: use dg.run_episode()')
        env = route_single_run(env, kwargs)
        if hasattr(env, 'run_with_deepgym'):
            if self._api_url is not None:
                raise DeepGymError(
                    'Benchmark-backed environments run their own scoring and cannot go '
                    'through the HTTP API; use a local or Daytona client.'
                )
            # Their runners are sync-only, so run them on a sync client in a thread.
            return await asyncio.to_thread(
                env.run_with_deepgym, self._sync_client(), model_output, **kwargs
            )
        if self._api_url is not None:
            return await self._run_http(env, model_output)

        if self._mode in PROVIDERS:
            return await asyncio.to_thread(self._sync_client().run, env, model_output)

        if self._local_executor is not None:
            return await self._run_local(env, model_output)

        if self._daytona is None:
            raise SandboxError('AsyncDaytona client not initialised')

        sandbox: Sandbox | None = None
        start = time.perf_counter()

        try:
            sandbox = await self._create_sandbox(env)
            exec_result = await self._run_judge(
                sandbox, JudgeJob.for_environment(env, model_output)
            )
            return build_run_result(
                exec_result.verifier_result,
                (time.perf_counter() - start) * 1000.0,
                str(getattr(sandbox, 'id', 'unknown')),
                stderr=exec_result.stderr,
                exit_code=exec_result.exit_code,
            )

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
        **kwargs: Any,
    ) -> BatchResult:
        """Run multiple solutions concurrently with a semaphore-based limit.

        Args:
            env: The environment specification.
            outputs: List of model-generated solutions.
            max_parallel: Maximum number of concurrent sandbox executions.
            **kwargs: Per-sample metadata (scalars or one value per output), routed
                like :meth:`DeepGym.run_batch`.

        Returns:
            Aggregated BatchResult (see :func:`~deepgym.core.build_batch_result`).

        Raises:
            ValueError: If *max_parallel* is below 1 (no run could ever start).
        """
        if max_parallel < 1:
            raise ValueError(f'max_parallel must be at least 1, got {max_parallel}')
        if not outputs:
            return build_batch_result([], 0.0)

        start = time.perf_counter()
        semaphore = asyncio.Semaphore(max_parallel)

        async def _guarded_run(request: ResolvedRun) -> RunResult:
            async with semaphore:
                return await self.run(request.env, request.output, **request.kwargs)

        requests = build_run_requests(env, outputs, dict(kwargs))
        tasks = [asyncio.ensure_future(_guarded_run(request)) for request in requests]
        settled = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[RunResult] = []
        for i, item in enumerate(settled):
            if isinstance(item, BaseException):
                logger.warning('Async run %d failed: %s', i, item)
                item = error_result(item)
            results.append(item)
        return build_batch_result(results, (time.perf_counter() - start) * 1000.0)

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
            An EvalResult with aggregate statistics; ``results`` follow the suite's
            order, named in ``env_names``.

        Raises:
            ValueError: If neither or both of *model_outputs* and *model*
                are provided, or if *max_parallel* is below 1.
        """
        if model_outputs is None and model is None:
            raise ValueError('Provide either model_outputs or model')
        if model_outputs is not None and model is not None:
            raise ValueError('Provide either model_outputs or model, not both')
        if max_parallel < 1:
            raise ValueError(f'max_parallel must be at least 1, got {max_parallel}')

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

        settled = await asyncio.gather(
            *(_eval_one(env) for env in environments), return_exceptions=True
        )
        names = [_infer_environment_name(env) for env in environments]
        return build_eval_result(suite, getattr(model, '__name__', 'unknown'), names, settled)

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

        async def _eval_one(env: Environment, name: str) -> RunResult | None:
            if name not in model_outputs:
                return None
            async with semaphore:
                return await self.run(env, model_outputs[name])

        names = [_infer_environment_name(env) for env in environments]
        settled = await asyncio.gather(
            *(_eval_one(env, name) for env, name in zip(environments, names)),
            return_exceptions=True,
        )
        return build_eval_result(suite, 'unknown', names, settled)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _run_local(self, env: Environment, model_output: str) -> RunResult:
        """Execute using the async local subprocess executor."""
        assert self._local_executor is not None
        start = time.perf_counter()
        exec_result = await self._local_executor.run(JudgeJob.for_environment(env, model_output))
        return local_run_result(exec_result, (time.perf_counter() - start) * 1000.0)

    def _sync_client(self) -> DeepGym:
        """A sync client on the same backend, for runners that are sync-only."""
        if self._sync is None:
            from deepgym.core import DeepGym

            if self._mode in PROVIDERS:
                mode = self._mode
            else:
                mode = 'local' if self._local_executor is not None else 'daytona'
            self._sync = DeepGym(api_key=self._api_key, mode=mode)
        return self._sync

    @staticmethod
    def _init_daytona(api_key: str | None, api_url: str | None = None) -> AsyncDaytona:
        """Initialise the async Daytona SDK client."""
        try:
            from daytona import AsyncDaytona, DaytonaConfig
        except ImportError as exc:
            raise SandboxError(
                "daytona is required. Install it with: pip install 'deepgym[daytona]'"
            ) from exc

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs['api_key'] = api_key
        if api_url:
            kwargs['api_url'] = api_url
        config = DaytonaConfig(**kwargs) if kwargs else None
        return AsyncDaytona(config) if config else AsyncDaytona()

    async def _create_sandbox(self, env: Environment) -> Sandbox:
        """Create an ephemeral sandbox from the env's snapshot or image.

        Labelled, as the sync client's are, so one a failed create still allocated is found
        and deleted rather than left running.
        """
        assert self._daytona is not None
        label = {'deepgym_sandbox': uuid4().hex}
        params = create_params(env).model_copy(update={'labels': label})
        try:
            return await self._daytona.create(params)
        except Exception as exc:
            if await self._delete_labelled(label) is not True:
                # Daytona may still be making it: keep looking (see LEAK_SWEEP_S).
                sweep = asyncio.create_task(self._sweep_labelled(label))
                self._sweeps.add(sweep)
                sweep.add_done_callback(self._sweeps.discard)
            raise SandboxError(f'Failed to create async sandbox: {exc}') from exc

    async def _delete_labelled(self, label: dict[str, str]) -> bool | None:
        """Delete every sandbox carrying *label*.

        Returns:
            None if no sandbox carries it, True once every one that does is deleted, False
            if the listing or a delete failed.
        """
        from daytona import ListSandboxesQuery

        assert self._daytona is not None
        try:
            query = ListSandboxesQuery(labels=label)
            found = [sandbox async for sandbox in self._daytona.list(query)]
        except Exception:
            return False
        if not found:
            return None
        return all([await self._cleanup_sandbox(sandbox) for sandbox in found])

    async def _sweep_labelled(
        self,
        label: dict[str, str],
        seconds: float = LEAK_SWEEP_S,
        every: float = LEAK_SWEEP_EVERY_S,
    ) -> bool | None:
        """Async twin of :func:`deepgym.sandboxes.daytona.sweep_labelled`."""
        end = time.monotonic() + seconds
        outcome = await self._delete_labelled(label)
        while outcome is not True and time.monotonic() < end:
            await asyncio.sleep(every)
            outcome = await self._delete_labelled(label)
        if outcome is False:
            logger.warning('Could not delete a sandbox labelled %s; Daytona auto-stops it', label)
        return outcome

    async def _run_judge(self, sandbox: Sandbox, job: JudgeJob) -> ExecutionResult:
        """Async twin of :func:`deepgym.sandbox.run_judge_in_sandbox`: a run that times out
        after the judge announced the solution's start is a judged 0 (see
        :func:`~deepgym.sandbox.judge_timeout`).

        Raises:
            SandboxError: If setting up users or uploading files fails, or if Daytona
                fails to run the verifier.
            TimeoutError: If the verifier outruns its time budget before the solution starts.
            VerifierError: If the verifier prints no protocol JSON.
        """
        _validate_env_vars(job.env_vars)
        try:
            setup = await sandbox.process.exec(f'bash -c {shlex.quote(SANDBOX_SETUP)}', timeout=60)
            if setup.exit_code != 0:
                raise SandboxError(f'Judge setup failed: {setup.result}')
            for content, path in sandbox_uploads(job):
                await sandbox.fs.upload_file(content, path)
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f'Failed to set up the judge: {exc}') from exc
        # Daytona doesn't block the network itself, so refuse to judge online.
        command, env = sandbox_judge(job, setup.result, require_offline=True)
        command, marker = wrap_stderr(command)
        timeout = job.timeout + DEADLINE_GRACE_S
        try:
            response = await exec_async(sandbox, command, env=env, timeout=timeout)
        except SandboxTimeoutError as exc:
            return judge_timeout(job, exc)
        except Exception as exc:
            # A failed call returns nothing the verifier printed, so even a sandbox that died
            # can't be shown to have died after the solution started: no verdict.
            raise SandboxError(f'Verifier execution failed: {exc}') from exc
        return parse_wrapped_output(response.output, marker, response.exit_code, job.consulted_mark)

    async def _cleanup_sandbox(self, sandbox: Sandbox) -> bool:
        """Safely destroy a sandbox, swallowing errors.

        Returns:
            False if Daytona reported a failure; True once the sandbox is gone, including
            when Daytona no longer knows it (404).
        """
        try:
            assert self._daytona is not None
            await self._daytona.delete(sandbox)
        except Exception as exc:
            if getattr(exc, 'status_code', None) == 404:
                return True
            logger.warning('Failed to clean up async sandbox %s', getattr(sandbox, 'id', '?'))
            return False
        return True

    async def _run_http(self, env: Environment, model_output: str) -> RunResult:
        """Execute a run via the remote DeepGym API server."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f'{self._api_url}/v1/run',
                json={
                    # The API takes no file paths, so a file-backed verifier goes inline.
                    'environment': {
                        **env.model_dump(mode='json', exclude={'verifier_path'}),
                        'verifier_code': env.verifier_source,
                    },
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

        envs = load_suite(suite)
        if any(isinstance(env, MultiTurnEnvironment) for env in envs):
            raise DeepGymError(f'Suite {suite!r} needs an agent: use dg.run_episode() per env')
        return envs
