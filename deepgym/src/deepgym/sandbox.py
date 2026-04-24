"""Sandbox lifecycle management for DeepGym.

Handles creation, setup, execution, and teardown of Daytona sandboxes.

NOTE: Daytona SDK import names may vary across versions. Verify the import
paths (Daytona, CreateSandboxFromSnapshotParams, CreateSandboxFromImageParams,
Image, Sandbox) against the installed daytona-sdk version.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from deepgym.exceptions import SandboxError, TimeoutError, VerifierError
from deepgym.models import Environment, RunResult, VerifierResult
from deepgym.verifier_template import wrap_verifier

if TYPE_CHECKING:
    try:
        from daytona import Daytona, Sandbox
    except ImportError:
        from daytona_sdk import Daytona, Sandbox

logger = logging.getLogger(__name__)

_VERIFIER_PATH = '/home/user/verifier.py'
_SOLUTION_PATH = '/home/user/solution.py'
_TEST_CASES_PATH = '/home/user/test_cases.json'

_ENV_KEY_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _validate_env_vars(env_vars: dict[str, str] | None) -> None:
    """Validate environment variable keys are safe shell identifiers.

    Args:
        env_vars: Mapping of env var names to values. None is a no-op.

    Raises:
        SandboxError: If any key is not a valid shell identifier.
    """
    if not env_vars:
        return
    for key in env_vars:
        if not _ENV_KEY_PATTERN.match(key):
            raise SandboxError(
                f'Invalid environment variable name: {key!r}. '
                'Names must match [A-Za-z_][A-Za-z0-9_]*'
            )


def is_standalone_verifier(source: str) -> bool:
    """Check if verifier source is a standalone script with JSON protocol output.

    Standalone verifiers have their own ``__name__`` guard and produce JSON
    directly via ``json.dumps``.  Non-standalone verifiers are function bodies
    that need wrapping with :func:`~deepgym.verifier_template.wrap_verifier`.

    Args:
        source: Python source code of the verifier.

    Returns:
        True if the verifier is standalone, False if it needs wrapping.
    """
    return '__name__' in source and 'json.dumps' in source


class ExecutionResult:
    """Bundle verifier output with process-level diagnostics.

    Attributes:
        verifier_result: Parsed structured verifier output.
        stderr: Captured stderr from the verifier process.
        exit_code: Process exit code (0 = passed, 1 = failed, 2 = verifier error).
    """

    __slots__ = ('verifier_result', 'stderr', 'exit_code')

    def __init__(
        self,
        verifier_result: VerifierResult,
        stderr: str = '',
        exit_code: int = 0,
    ) -> None:
        self.verifier_result = verifier_result
        self.stderr = stderr
        self.exit_code = exit_code


def build_run_result(
    verifier_result: VerifierResult,
    elapsed_ms: float,
    sandbox_id: str,
    stderr: str = '',
    exit_code: int = 0,
) -> RunResult:
    """Build a RunResult from a VerifierResult with timing info.

    Args:
        verifier_result: Parsed verifier output.
        elapsed_ms: Wall-clock execution time in milliseconds.
        sandbox_id: Identifier for the sandbox (or 'local').
        stderr: Captured stderr from verifier execution.
        exit_code: Process exit code from verifier execution.

    Returns:
        A fully populated RunResult.
    """
    return RunResult(
        score=verifier_result.score,
        passed=verifier_result.passed,
        output=verifier_result.details or '',
        stderr=stderr,
        exit_code=exit_code,
        execution_time_ms=elapsed_ms,
        sandbox_id=sandbox_id,
        reward_components=verifier_result.reward_components,
        metrics=verifier_result.metrics,
        seed=verifier_result.seed,
        truncated=verifier_result.truncated,
        error_type=verifier_result.error_type,
        cases=verifier_result.cases,
    )


def create_sandbox(env: Environment, daytona: Daytona) -> Sandbox:
    """Create a Daytona sandbox configured for the given environment.

    Uses a pre-built snapshot when ``env.snapshot`` is set, otherwise falls
    back to a default Python image.

    Args:
        env: The environment specification.
        daytona: An initialised Daytona client.

    Returns:
        A running Sandbox instance.

    Raises:
        SandboxError: If sandbox creation fails.
    """
    # NOTE: Import here so module-level import failures don't break the
    # entire package when daytona-sdk is not installed.
    try:
        from daytona import (
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            Image,
        )
    except ImportError:
        try:
            from daytona_sdk import (
                CreateSandboxFromImageParams,
                CreateSandboxFromSnapshotParams,
                Image,
            )
        except ImportError as exc:
            raise SandboxError(
                'daytona-sdk is required but could not be imported. '
                'Install it with: pip install daytona-sdk'
            ) from exc

    try:
        if env.snapshot:
            params = CreateSandboxFromSnapshotParams(
                snapshot=env.snapshot,
                ephemeral=True,
            )
            sandbox = daytona.create(params)
        else:
            params = CreateSandboxFromImageParams(
                image=Image.debian_slim('3.12'),
                ephemeral=True,
            )
            sandbox = daytona.create(params)
    except Exception as exc:
        raise SandboxError(f'Failed to create sandbox: {exc}') from exc

    return sandbox


def setup_sandbox(sandbox: Sandbox, env: Environment, model_output: str) -> None:
    """Upload verifier, solution, and test cases into the sandbox.

    Args:
        sandbox: The target sandbox.
        env: The environment specification (contains verifier code / path).
        model_output: The model-generated solution source code.

    Raises:
        SandboxError: If any file upload fails.
    """
    try:
        # Resolve verifier source.
        if env.verifier_path is not None:
            verifier_source = env.verifier_path.read_text(encoding='utf-8')
        else:
            verifier_source = env.verifier_code

        # Wrap simple verifiers that don't already produce JSON protocol output.
        if is_standalone_verifier(verifier_source):
            ready_code = verifier_source
        else:
            ready_code = wrap_verifier(verifier_source)

        # Daytona SDK: upload_file(src: bytes, dst: str) — content first, path second
        sandbox.fs.upload_file(ready_code.encode('utf-8'), _VERIFIER_PATH)
        sandbox.fs.upload_file(model_output.encode('utf-8'), _SOLUTION_PATH)

        if env.test_cases is not None:
            payload = json.dumps(env.test_cases, ensure_ascii=False)
            sandbox.fs.upload_file(payload.encode('utf-8'), _TEST_CASES_PATH)

    except Exception as exc:
        raise SandboxError(f'Failed to set up sandbox files: {exc}') from exc


def execute_verifier(
    sandbox: Sandbox,
    timeout: int,
    has_test_cases: bool = False,
    env_vars: dict[str, str] | None = None,
) -> VerifierResult:
    """Run the verifier inside the sandbox and parse its JSON output.

    The verifier is invoked as::

        python verifier.py solution.py [test_cases.json]

    Environment variables are passed inline so they persist for the verifier
    process (``export`` in a prior exec call is lost between processes).

    Args:
        sandbox: A sandbox with verifier and solution already uploaded.
        timeout: Maximum execution time in seconds.
        has_test_cases: Whether test_cases.json was uploaded.
        env_vars: Optional env vars to inject into the verifier command.

    Returns:
        Parsed VerifierResult.

    Raises:
        VerifierError: If verifier output is not valid JSON or cannot be parsed.
        TimeoutError: If execution exceeds *timeout* seconds.
    """
    _validate_env_vars(env_vars)

    env_prefix = ''
    if env_vars:
        env_prefix = ' '.join(f'{k}={_shell_escape(v)}' for k, v in env_vars.items()) + ' '

    cmd = f'{env_prefix}python {_VERIFIER_PATH} {_SOLUTION_PATH}'
    if has_test_cases:
        cmd += f' {_TEST_CASES_PATH}'

    try:
        response = sandbox.process.exec(cmd, timeout=timeout)
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
        result = VerifierResult.model_validate_json(json_str)
    except Exception as exc:
        raise VerifierError(
            f'Failed to parse verifier JSON: {exc}. Raw output: {json_str!r}'
        ) from exc

    return result


def cleanup_sandbox(sandbox: Sandbox, daytona: Daytona) -> None:
    """Safely destroy a sandbox, swallowing any errors.

    Args:
        sandbox: The sandbox to destroy.
        daytona: The Daytona client that owns the sandbox.
    """
    try:
        daytona.delete(sandbox)
    except Exception:
        logger.debug('Failed to clean up sandbox %s (ignored)', getattr(sandbox, 'id', '?'))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_json_line(stdout: str) -> str | None:
    """Return the last line of *stdout* that looks like a JSON object."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            return stripped
    return None


def _shell_escape(value: str) -> str:
    """Wrap *value* in single quotes for safe shell interpolation."""
    return "'" + value.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# Local executor (no Daytona required)
# ---------------------------------------------------------------------------


class LocalExecutor:
    """Run verifiers locally via subprocess. For dev/testing only — no isolation."""

    def run(
        self,
        verifier_code: str,
        solution_code: str,
        test_cases: list[dict] | None = None,
        timeout: int = 30,
        env_vars: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute verifier against solution in a temporary directory.

        Args:
            verifier_code: Complete Python verifier script.
            solution_code: The model-generated solution source.
            test_cases: Optional list of test-case dicts.
            timeout: Maximum execution time in seconds.
            env_vars: Optional env vars to inject into the subprocess.

        Returns:
            ExecutionResult with parsed VerifierResult, stderr, and exit_code.

        Raises:
            TimeoutError: If execution exceeds *timeout*.
            VerifierError: If verifier output is invalid or execution fails.
        """
        _validate_env_vars(env_vars)

        tmp_dir = tempfile.mkdtemp(prefix='deepgym_local_')
        try:
            verifier_path = Path(tmp_dir) / 'verifier.py'
            solution_path = Path(tmp_dir) / 'solution.py'
            test_cases_path = Path(tmp_dir) / 'test_cases.json'

            verifier_path.write_text(verifier_code, encoding='utf-8')
            solution_path.write_text(solution_code, encoding='utf-8')

            cmd = [sys.executable, str(verifier_path), str(solution_path)]

            if test_cases is not None:
                test_cases_path.write_text(
                    json.dumps(test_cases, ensure_ascii=False), encoding='utf-8'
                )
                cmd.append(str(test_cases_path))

            # Merge env_vars into a copy of the current environment.
            proc_env = None
            if env_vars:
                proc_env = {**os.environ, **env_vars}

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmp_dir,
                    env=proc_env,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f'Local verifier execution exceeded {timeout}s timeout') from exc

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            json_str = _extract_json_line(stdout)
            if json_str is None:
                raise VerifierError(
                    f'Local verifier did not produce valid JSON. '
                    f'stdout: {stdout!r}, stderr: {stderr!r}'
                )

            try:
                result = VerifierResult.model_validate_json(json_str)
            except Exception as exc:
                raise VerifierError(
                    f'Failed to parse local verifier JSON: {exc}. Raw: {json_str!r}'
                ) from exc

            return ExecutionResult(
                verifier_result=result,
                stderr=stderr,
                exit_code=proc.returncode,
            )

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class AsyncLocalExecutor:
    """Async version of LocalExecutor using asyncio.subprocess."""

    async def run(
        self,
        verifier_code: str,
        solution_code: str,
        test_cases: list[dict] | None = None,
        timeout: int = 30,
        env_vars: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute verifier against solution asynchronously in a temp directory.

        Args:
            verifier_code: Complete Python verifier script.
            solution_code: The model-generated solution source.
            test_cases: Optional list of test-case dicts.
            timeout: Maximum execution time in seconds.
            env_vars: Optional env vars to inject into the subprocess.

        Returns:
            ExecutionResult with parsed VerifierResult, stderr, and exit_code.

        Raises:
            TimeoutError: If execution exceeds *timeout*.
            VerifierError: If verifier output is invalid or execution fails.
        """
        _validate_env_vars(env_vars)

        tmp_dir = tempfile.mkdtemp(prefix='deepgym_async_local_')
        try:
            verifier_path = Path(tmp_dir) / 'verifier.py'
            solution_path = Path(tmp_dir) / 'solution.py'
            test_cases_path = Path(tmp_dir) / 'test_cases.json'

            verifier_path.write_text(verifier_code, encoding='utf-8')
            solution_path.write_text(solution_code, encoding='utf-8')

            cmd = [sys.executable, str(verifier_path), str(solution_path)]

            if test_cases is not None:
                test_cases_path.write_text(
                    json.dumps(test_cases, ensure_ascii=False), encoding='utf-8'
                )
                cmd.append(str(test_cases_path))

            # Merge env_vars into a copy of the current environment.
            proc_env = None
            if env_vars:
                proc_env = {**os.environ, **env_vars}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmp_dir,
                env=proc_env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise TimeoutError(
                    f'Async local verifier execution exceeded {timeout}s timeout'
                ) from exc

            stdout = stdout_bytes.decode('utf-8', errors='replace').strip()
            stderr = stderr_bytes.decode('utf-8', errors='replace').strip()
            exit_code = proc.returncode if proc.returncode is not None else -1

            json_str = _extract_json_line(stdout)
            if json_str is None:
                raise VerifierError(
                    f'Async local verifier did not produce valid JSON. '
                    f'stdout: {stdout!r}, stderr: {stderr!r}'
                )

            try:
                result = VerifierResult.model_validate_json(json_str)
            except Exception as exc:
                raise VerifierError(
                    f'Failed to parse async local verifier JSON: {exc}. Raw: {json_str!r}'
                ) from exc

            return ExecutionResult(
                verifier_result=result,
                stderr=stderr,
                exit_code=exit_code,
            )

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
