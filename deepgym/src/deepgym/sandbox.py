"""Verifier execution for DeepGym.

Runs verifier jobs (see :mod:`deepgym.judge`) locally or in a sandbox from any
provider (see :mod:`deepgym.sandboxes`), with the solution isolated from the
judge in both cases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import subprocess

from deepgym.exceptions import InvalidScoreError, SandboxError, TimeoutError, VerifierError
from deepgym.judge import (
    DEADLINE_GRACE_S,
    SANDBOX_SETUP,
    JudgeJob,
    local_judge,
    sandbox_judge,
    sandbox_uploads,
)
from deepgym.models import RunResult, VerifierResult, checked_score
from deepgym.sandboxes import Sandbox, SandboxTimeoutError

logger = logging.getLogger(__name__)

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


def check_exit(exit_code: int, score: float, passed: bool | None) -> None:
    """Accept a verifier's result only if its exit code backs it.

    Protocol exit codes: 0 passed, 1 failed, 2 verifier error. A failed run keeps
    its partial score but can't claim a pass, a reported verifier error carries no
    reward, and any other code (killed, timed out, crashed) means the verifier
    never finished, whatever it printed first. *passed* is None for a bare float,
    which only exit 0 backs.

    Raises:
        VerifierError: If the exit code doesn't back the result.
    """
    if exit_code == 0:
        return
    if passed is False and (exit_code == 1 or (exit_code == 2 and score == 0.0)):
        return
    raise VerifierError(
        f'Verifier exited {exit_code} after reporting score={score}, passed={passed}; '
        'only a clean exit (0, or 1 for a failed run) counts'
    )


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

    Raises:
        VerifierError: If *exit_code* doesn't back the result (see :func:`check_exit`).
    """
    check_exit(exit_code, verifier_result.score, verifier_result.passed)
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


def run_judge_in_sandbox(sandbox: Sandbox, job: JudgeJob) -> ExecutionResult:
    """Run *job*'s verifier in *sandbox*.

    The solution runs as a separate, unprivileged user with no capabilities and no
    network: the provider blocks it, or the solution gets its own offline network
    namespace.

    Raises:
        SandboxError: If setting up users or uploading files fails, or if the
            solution can't be cut off the network.
        TimeoutError: If the verifier outruns its time budget.
        VerifierError: If the verifier fails to run or prints no protocol JSON.
    """
    _validate_env_vars(job.env_vars)
    try:
        setup = sandbox.exec(f'bash -c {shlex.quote(SANDBOX_SETUP)}', timeout=60)
        if setup.exit_code != 0:
            raise SandboxError(f'Judge setup failed: {setup.output[-2000:]}')
        for content, path in sandbox_uploads(job):
            sandbox.upload(content, path)
    except SandboxError:
        raise
    except Exception as exc:
        raise SandboxError(f'Failed to set up the judge: {exc}') from exc
    command, env = sandbox_judge(job, setup.output, require_offline=not sandbox.network_blocked)
    try:
        response = sandbox.exec(command, env=env, timeout=job.timeout + DEADLINE_GRACE_S)
    except SandboxTimeoutError as exc:
        raise TimeoutError(f'Verifier exceeded {job.timeout}s') from exc
    except Exception as exc:
        raise VerifierError(f'Verifier execution failed: {exc}') from exc
    # Exit codes 0 (passed) and 1 (failed) are normal verifier outcomes, so parse regardless.
    return ExecutionResult(
        parse_verifier_output(response.output.strip(), f'exit_code={response.exit_code}'),
        exit_code=response.exit_code,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def parse_verifier_output(stdout: str, context: str = '') -> VerifierResult:
    """Parse the last JSON line of verifier stdout into a VerifierResult.

    Args:
        stdout: Verifier stdout.
        context: Extra diagnostics (exit code, stderr) for the error message.

    Raises:
        InvalidScoreError: If the score is NaN, infinite, or outside [0, 1].
        VerifierError: If there is no JSON line or it does not match the protocol.
    """
    json_str = _extract_json_line(stdout)
    if json_str is None:
        raise VerifierError(f'Verifier did not produce valid JSON. stdout: {stdout!r} {context}')
    try:
        data = json.loads(json_str)  # accepts NaN/Infinity, so they're caught below
    except ValueError as exc:
        raise VerifierError(f'Failed to parse verifier JSON: {exc}. Raw: {json_str!r}') from exc
    if not isinstance(data, dict):
        raise VerifierError(f'Verifier JSON is not an object. Raw: {json_str!r}')
    if data.get('error_type') == 'invalid_score':
        raise InvalidScoreError(data.get('details') or 'Verifier returned an invalid score')
    if 'score' in data:
        data['score'] = checked_score(data['score'])
    try:
        return VerifierResult.model_validate(data)
    except ValueError as exc:
        raise VerifierError(f'Failed to parse verifier JSON: {exc}. Raw: {json_str!r}') from exc


def _extract_json_line(stdout: str) -> str | None:
    """Return the last line of *stdout* that looks like a JSON object."""
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            return stripped
    return None


# ---------------------------------------------------------------------------
# Local executor (no Daytona required)
# ---------------------------------------------------------------------------


class LocalExecutor:
    """Run verifiers locally, with the solution in a sandboxed subprocess where available."""

    def run(self, job: JudgeJob) -> ExecutionResult:
        """Run *job*'s verifier as a local subprocess.

        Raises:
            TimeoutError: If the verifier outruns its time budget.
            VerifierError: If the verifier prints no protocol JSON.
        """
        _validate_env_vars(job.env_vars)
        with local_judge(job) as (argv, env, cwd):
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=job.timeout + DEADLINE_GRACE_S,
                    cwd=cwd,
                    env=env,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f'Local verifier exceeded {job.timeout}s') from exc
        stderr = proc.stderr.strip()
        return ExecutionResult(
            parse_verifier_output(proc.stdout.strip(), f'stderr: {stderr!r}'),
            stderr=stderr,
            exit_code=proc.returncode,
        )


class AsyncLocalExecutor:
    """Async version of LocalExecutor using asyncio.subprocess."""

    async def run(self, job: JudgeJob) -> ExecutionResult:
        """Run *job*'s verifier as a local subprocess without blocking the event loop.

        Raises:
            TimeoutError: If the verifier outruns its time budget.
            VerifierError: If the verifier prints no protocol JSON.
        """
        _validate_env_vars(job.env_vars)
        with local_judge(job) as (argv, env, cwd):
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=job.timeout + DEADLINE_GRACE_S
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f'Async local verifier exceeded {job.timeout}s') from exc
        stderr_text = stderr.decode('utf-8', errors='replace').strip()
        return ExecutionResult(
            parse_verifier_output(
                stdout.decode('utf-8', errors='replace').strip(), f'stderr: {stderr_text!r}'
            ),
            stderr=stderr_text,
            exit_code=proc.returncode if proc.returncode is not None else -1,
        )
