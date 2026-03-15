"""Multi-turn environment execution.

WARNING: Local mode runs agent code on the host without isolation.
Use Daytona sandboxes for untrusted agents.

Support environments where agents interact over multiple steps.
Each step: agent receives observation, produces action, action executes,
new observation returned. Final score computed by verifier.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from deepgym.exceptions import DeepGymError, TimeoutError, VerifierError
from deepgym.models import (
    Action,
    MultiTurnEnvironment,
    Observation,
    RunResult,
    Trajectory,
    VerifierResult,
)
from deepgym.sandbox import _extract_json_line

logger = logging.getLogger(__name__)


class MultiTurnRunner:
    """Run multi-turn episodes using local subprocess execution.

    WARNING: This runner always uses local subprocess execution regardless
    of the DeepGym client's configured backend. Daytona sandbox support
    for multi-turn episodes is not yet implemented.

    WARNING: Local mode runs agent code on the host without isolation.
    Use safe_mode=True (default) to restrict execution to Python code only.

    Args:
        safe_mode: If True (default), refuse to execute bash actions and
            only allow code actions. Set to False only when the agent is
            trusted or execution is sandboxed.
    """

    def __init__(self, *, safe_mode: bool = True) -> None:
        self._safe_mode = safe_mode

    def run(
        self,
        env: MultiTurnEnvironment,
        agent: Callable[[Observation], Action],
    ) -> tuple[Trajectory, RunResult]:
        """Run a complete multi-turn episode.

        Args:
            env: Multi-turn environment spec.
            agent: Callable that takes an Observation and returns an Action.

        Returns:
            Tuple of (trajectory, final_result).

        Raises:
            TimeoutError: If a step exceeds its timeout.
            VerifierError: If final verifier output is invalid.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix='deepgym_multiturn_'))
        try:
            return self._run_episode(env, agent, tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _run_episode(
        self,
        env: MultiTurnEnvironment,
        agent: Callable[[Observation], Action],
        work_dir: Path,
    ) -> tuple[Trajectory, RunResult]:
        """Execute the full episode loop in a temporary directory.

        Args:
            env: Multi-turn environment spec.
            agent: Agent callable.
            work_dir: Temporary working directory for execution.

        Returns:
            Tuple of (trajectory, final_result).
        """
        trajectory = Trajectory()

        # Run setup code if provided.
        setup_output = ''
        if env.setup_code:
            setup_output = self._execute_code(env.setup_code, work_dir, env.timeout_per_step)

        # Initial observation.
        obs = Observation(content=setup_output, step=0)

        for step in range(1, env.max_steps + 1):
            action = agent(obs)
            stdout = self._execute_step(action, work_dir, env.timeout_per_step)

            # Compute step reward if step verifier is provided.
            step_reward = 0.0
            if env.step_verifier_code:
                step_reward = self._run_step_verifier(
                    env.step_verifier_code, stdout, step, work_dir, env.timeout_per_step
                )

            trajectory.steps.append((obs, action))
            trajectory.step_rewards.append(step_reward)
            trajectory.total_reward += step_reward

            # Check if the action output signals completion.
            done = self._check_done(stdout)
            obs = Observation(content=stdout, step=step, done=done)

            if done:
                break

        trajectory.final_observation = obs

        # Run final verifier.
        result = self._run_final_verifier(env, work_dir)

        return trajectory, result

    def _execute_step(self, action: Action, work_dir: Path, timeout: int) -> str:
        """Execute a single step action and return output.

        Args:
            action: The action to execute.
            work_dir: Working directory for execution.
            timeout: Maximum execution time in seconds.

        Returns:
            Combined stdout from executing the action.

        Raises:
            DeepGymError: If bash actions are attempted in safe_mode.
            TimeoutError: If execution exceeds *timeout*.
        """
        if action.action_type == 'bash':
            if self._safe_mode:
                raise DeepGymError(
                    'Bash actions are disabled in safe_mode. '
                    'Use safe_mode=False only with trusted agents or '
                    'Daytona sandboxes.'
                )
            return self._execute_bash(action.content, work_dir, timeout)
        return self._execute_code(action.content, work_dir, timeout)

    def _execute_code(self, code: str, work_dir: Path, timeout: int) -> str:
        """Execute Python code in the working directory.

        Args:
            code: Python source code to execute.
            work_dir: Working directory.
            timeout: Maximum execution time in seconds.

        Returns:
            Captured stdout.

        Raises:
            TimeoutError: If execution exceeds *timeout*.
        """
        script_path = work_dir / '_step.py'
        script_path.write_text(code, encoding='utf-8')

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(work_dir),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f'Step execution exceeded {timeout}s timeout') from exc

        output = proc.stdout
        if proc.stderr:
            output += '\n' + proc.stderr
        return output

    def _execute_bash(self, command: str, work_dir: Path, timeout: int) -> str:
        """Execute a bash command in the working directory.

        Args:
            command: Shell command string.
            work_dir: Working directory.
            timeout: Maximum execution time in seconds.

        Returns:
            Captured stdout.

        Raises:
            TimeoutError: If execution exceeds *timeout*.
        """
        try:
            proc = subprocess.run(
                ['bash', '-c', command],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(work_dir),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f'Bash step execution exceeded {timeout}s timeout') from exc

        output = proc.stdout
        if proc.stderr:
            output += '\n' + proc.stderr
        return output

    def _run_step_verifier(
        self,
        verifier_code: str,
        step_output: str,
        step: int,
        work_dir: Path,
        timeout: int,
    ) -> float:
        """Run the step verifier and return a reward for this step.

        The step verifier receives the step output and step number as
        arguments via environment variables.

        Args:
            verifier_code: Python source for the step verifier.
            step_output: Stdout from the current step.
            step: Current step number.
            work_dir: Working directory.
            timeout: Maximum execution time in seconds.

        Returns:
            Reward value for this step (0.0 on verifier failure).
        """
        verifier_path = work_dir / '_step_verifier.py'
        verifier_path.write_text(verifier_code, encoding='utf-8')

        step_output_path = work_dir / '_step_output.txt'
        step_output_path.write_text(step_output, encoding='utf-8')

        env = {
            'STEP_OUTPUT_PATH': str(step_output_path),
            'STEP_NUMBER': str(step),
            'PATH': '/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin',
        }

        try:
            proc = subprocess.run(
                [sys.executable, str(verifier_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(work_dir),
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.warning('Step verifier timed out at step %d', step)
            return 0.0

        json_str = _extract_json_line(proc.stdout.strip())
        if json_str is None:
            # Try parsing stdout as a plain float.
            try:
                return max(0.0, min(1.0, float(proc.stdout.strip())))
            except ValueError:
                return 0.0

        try:
            data = json.loads(json_str)
            return max(0.0, min(1.0, float(data.get('score', 0.0))))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    def _run_final_verifier(
        self,
        env: MultiTurnEnvironment,
        work_dir: Path,
    ) -> RunResult:
        """Run the final verifier and return a RunResult.

        Args:
            env: The multi-turn environment spec.
            work_dir: Working directory containing all artifacts.

        Returns:
            RunResult from the final verifier.

        Raises:
            VerifierError: If verifier output is invalid.
        """
        if not env.final_verifier_code:
            return RunResult(
                score=0.0,
                passed=False,
                output='No final verifier provided',
                stderr='',
                exit_code=0,
                execution_time_ms=0.0,
                sandbox_id='local-multiturn',
            )

        verifier_path = work_dir / '_final_verifier.py'
        verifier_path.write_text(env.final_verifier_code, encoding='utf-8')

        try:
            proc = subprocess.run(
                [sys.executable, str(verifier_path)],
                capture_output=True,
                text=True,
                timeout=env.timeout_per_step,
                cwd=str(work_dir),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f'Final verifier exceeded {env.timeout_per_step}s timeout') from exc

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        json_str = _extract_json_line(stdout)

        if json_str is None:
            raise VerifierError(
                f'Final verifier did not produce valid JSON. stdout: {stdout!r}, stderr: {stderr!r}'
            )

        try:
            vr = VerifierResult.model_validate_json(json_str)
        except Exception as exc:
            raise VerifierError(
                f'Failed to parse final verifier JSON: {exc}. Raw: {json_str!r}'
            ) from exc

        return RunResult(
            score=vr.score,
            passed=vr.passed,
            output=vr.details or '',
            stderr=stderr,
            exit_code=proc.returncode,
            execution_time_ms=0.0,
            sandbox_id='local-multiturn',
            reward_components=vr.reward_components,
            metrics=vr.metrics,
            seed=vr.seed,
            truncated=vr.truncated,
            error_type=vr.error_type,
        )

    @staticmethod
    def _check_done(output: str) -> bool:
        """Check if the step output indicates the episode is done.

        Look for a JSON line containing a 'done' key set to true, or
        the literal marker 'DONE' on its own line.

        Args:
            output: Stdout from the step execution.

        Returns:
            True if the episode should end.
        """
        for line in output.strip().splitlines():
            stripped = line.strip()
            if stripped == 'DONE':
                return True
            if stripped.startswith('{') and stripped.endswith('}'):
                try:
                    data = json.loads(stripped)
                    if data.get('done') is True:
                        return True
                except (json.JSONDecodeError, AttributeError):
                    pass
        return False
