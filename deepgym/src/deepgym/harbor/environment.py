"""A Harbor task as a single-turn DeepGym environment."""

from __future__ import annotations

import base64
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from deepgym.exceptions import DeepGymError
from deepgym.harbor.agent import run_agent
from deepgym.harbor.grading import grade
from deepgym.harbor.task import HarborTask, pack_files
from deepgym.models import Environment, RunResult
from deepgym.sandbox import build_run_result
from deepgym.sandboxes import build_image

if TYPE_CHECKING:
    from deepgym.core import DeepGym


class HarborEnvironment(Environment):
    """One Harbor task: the model output is a bash script, and tests/test.sh scores it.

    With a sandbox backend, ``dg.run(env, script)`` runs the script as an unprivileged,
    offline user in a sandbox from the task's image (:func:`~deepgym.harbor.agent.run_agent`),
    then grades a copy of the work directory in a fresh sandbox from the same image
    (:func:`~deepgym.harbor.grading.grade`). Local mode has no sandbox, so it refuses.

    Example:
        >>> env = HarborEnvironment.from_dir(Path('tasks/hello-world'))
        >>> DeepGym(mode='docker').run(env, 'echo hello > /app/hello.txt').score
        1.0
    """

    verifier_code: str = 'return 0.0'
    """Placeholder the base model requires; tests/test.sh does the scoring."""

    harbor_task: HarborTask
    """The task folder this environment runs."""

    @classmethod
    def from_task(cls, task: HarborTask) -> HarborEnvironment:
        """Wrap *task*, named after its folder."""
        return cls(
            name=task.name,
            task=task.instruction,
            type='tool-use',
            domain='terminal',
            tags=['harbor'],
            image=task.docker_image,
            timeout=math.ceil(task.verifier_timeout_sec),
            harbor_task=task,
        )

    @classmethod
    def from_dir(cls, path: Path) -> HarborEnvironment:
        """Read the task folder at *path* and wrap it.

        Raises:
            ValueError: If the folder isn't a valid Harbor task.
        """
        return cls.from_task(HarborTask.from_dir(path))

    @property
    def reference_solution(self) -> str | None:
        """A script that unpacks solution/ into a temp folder and runs its solve.sh there.

        The agent can't write /solution, where Harbor puts it, so a solve.sh that reads its
        files from that absolute path won't find them.
        """
        if not self.harbor_task.solution:
            return None
        archive = base64.b64encode(pack_files(self.harbor_task.solution)).decode('ascii')
        return (
            'solution=$(mktemp -d)\n'
            f'echo \'{archive}\' | base64 -d | tar -xzf - -C "$solution"\n'
            'bash "$solution/solve.sh"\n'
        )

    def run_with_deepgym(self, dg: DeepGym, model_output: str, **kwargs: object) -> RunResult:
        """Run *model_output* as the agent's bash script, then grade the result.

        Args:
            dg: Client whose backend runs the agent and the grader.
            model_output: The bash script.
            **kwargs: Ignored; benchmark routing passes them to every environment.

        Raises:
            DeepGymError: If no image can be booted for the task on this backend.
            SandboxError: In local mode, or if a sandbox fails.
            TimeoutError: If tests/test.sh outruns the verifier timeout.
            VerifierError: If tests/test.sh writes no reward or an invalid one.
        """
        start = time.perf_counter()
        env = self if self.image else self.model_copy(update={'image': self._built_image(dg)})
        agent = run_agent(dg.open_sandbox(env), self.harbor_task, model_output)
        grader = dg.open_sandbox(env)
        result = grade(grader, self.harbor_task, agent.workspace, agent.workdir)
        return build_run_result(
            result.verifier_result.model_copy(update={'truncated': agent.timed_out}),
            (time.perf_counter() - start) * 1000,
            grader.id,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    def _built_image(self, dg: DeepGym) -> str:
        """Build the task's Dockerfile for *dg*'s backend.

        Raises:
            DeepGymError: If the task has no Dockerfile, or the backend can't build one.
        """
        dockerfile = self.harbor_task.dockerfile
        if dockerfile is None:
            raise DeepGymError(
                f'{self.name} has neither [environment].docker_image nor environment/Dockerfile'
            )
        try:
            context = self.harbor_task.build_context
        except ValueError as exc:
            raise DeepGymError(f'{self.name}: {exc}') from exc
        return build_image(dg.backend, context)
