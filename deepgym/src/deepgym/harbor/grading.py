"""Grade an agent's work directory in a fresh sandbox the agent never touched.

All the agent's sandbox hands over is an archive of its work directory. The grader boots
a new sandbox from the same image, unpacks the archive as an unprivileged user, uploads
tests/ to /tests only now, empties /logs/verifier, runs tests/test.sh as root (Harbor's
default verifier user), then reads the reward with a separate command. Nothing the agent
wrote can be in /tests or /logs/verifier, and none of its processes is alive.

The harness's own commands here run with an empty environment and only root-owned
folders on PATH. tests/test.sh gets an allowlisted environment: the image's PATH entries
that, resolved in the grader after the restore, lie outside the work directory; HOME;
the image's locale and TERM; and the task's ``[verifier].env``. Nothing else from the
image reaches it, so no startup variable (BASH_ENV, PYTHONSTARTUP, ...) can name a file
the agent left in the work directory.

Reward precedence follows Harbor: reward.json wins over reward.txt, and both must be
written during this run, since the folder is emptied first. Tests usually run the agent's
code as root; whatever that code writes while they run, including a reward file, is the
task's to guard against, as with any Harbor verifier. Tasks DeepGym exports write
reward.json last, after every solution process is gone, so theirs is always the harness's.
"""

from __future__ import annotations

import json
import math
import shlex
import statistics

from deepgym.exceptions import SandboxError, VerifierError
from deepgym.harbor.task import STAGING_DIR, HarborTask, pack_files
from deepgym.models import SCORE_TOLERANCE, VerifierResult, checked_score
from deepgym.sandbox import ExecutionResult
from deepgym.sandboxes import ExecResult, Sandbox
from deepgym.session import SandboxBackend, resolve_outside

TESTS_DIR = '/tests'
REWARD_DIR = '/logs/verifier'
_TESTS_ARCHIVE = f'{STAGING_DIR}/tests.tgz'
# Image variables tests/test.sh may see, read before the restore with shell builtins.
_PASSED_ON = ('PATH', 'LANG', 'LC_ALL', 'TERM')
_IMAGE_ENV = "printf '%s\\0' " + ' '.join(f'"${{{name}-}}"' for name in _PASSED_ON)


def grade(sandbox: Sandbox, task: HarborTask, workspace: bytes, workdir: str) -> ExecutionResult:
    """Grade *workspace*, an archive of the agent's *workdir*, with *task*'s tests.

    Args:
        sandbox: A fresh sandbox from the task's image; deleted afterwards.
        task: The task whose tests/test.sh grades.
        workspace: The agent's work directory, from ``SandboxBackend.archive_workdir``.
        workdir: Where to unpack it; tests/test.sh runs there.

    Returns:
        The reward as a verifier result, the tests' whole output, and their exit code.

    Raises:
        SandboxError: If setting up the sandbox fails.
        TimeoutError: If tests/test.sh outruns the verifier timeout.
        VerifierError: If tests/test.sh writes no reward or an invalid one.
    """
    backend = SandboxBackend(sandbox, workdir)  # deletes the sandbox if setup fails
    try:
        image = sandbox.exec(_IMAGE_ENV, timeout=30).output.split('\0')
        backend.restore_workdir(workspace)
        sandbox.upload(pack_files(task.tests), _TESTS_ARCHIVE)
        staged = backend.exec_trusted(
            f'rm -rf {TESTS_DIR} {REWARD_DIR} && mkdir -p {TESTS_DIR} {REWARD_DIR} && '
            f'tar -xzf {_TESTS_ARCHIVE} -C {TESTS_DIR} && rm -f {_TESTS_ARCHIVE} && '
            f'chmod +x {TESTS_DIR}/test.sh',
            timeout=120,
        )
        if staged.exit_code != 0:
            raise SandboxError(f'Failed to stage the tests: {staged.output}')
        tests = backend.exec_root(
            f'{TESTS_DIR}/test.sh',
            math.ceil(task.verifier_timeout_sec),
            offline=task.verifier_network == 'no-network',
            env=_test_env(backend, task, dict(zip(_PASSED_ON, image, strict=False))),
        )
        listing = backend.exec_trusted(read_reward_command(REWARD_DIR), timeout=60)
        return ExecutionResult(
            reward_result(listing.output, tests),
            stderr=tests.output,
            exit_code=tests.exit_code,
        )
    finally:
        sandbox.delete()  # Never raises; a failed delete is logged, not traded for the grade.


def _test_env(backend: SandboxBackend, task: HarborTask, image: dict[str, str]) -> dict[str, str]:
    """The only variables tests/test.sh gets (see the module docstring)."""
    basics = {name: value for name, value in image.items() if value and name != 'PATH'}
    return {
        **basics,
        'HOME': '/root',
        **task.verifier_env,
        'PATH': resolve_outside(backend, image.get('PATH', '')),
    }


def read_reward_command(directory: str) -> str:
    """Shell that prints the reward file's name, then its contents; ``none`` if missing.

    reward.json wins over reward.txt, as in Harbor (see the module docstring).
    """
    folder = shlex.quote(directory)
    return (
        f'for name in reward.json reward.txt; do if [ -f {folder}/"$name" ]; then '
        f'echo "$name"; cat {folder}/"$name"; exit 0; fi; done; echo none'
    )


def reward_result(listing: str, tests: ExecResult) -> VerifierResult:
    """Turn the output of :func:`read_reward_command` into a verifier result.

    reward.txt holds one number. reward.json maps labels to numbers: its ``reward`` is
    the score if present, else the mean of all of them, and each becomes a reward component.

    Raises:
        InvalidScoreError: If the score isn't a finite number in [0, 1].
        VerifierError: If tests/test.sh wrote no reward, or an unreadable one.
    """
    name, _, body = listing.partition('\n')
    components: dict[str, float] | None = None
    try:
        if name == 'reward.txt':
            score = float(body)
        elif name == 'reward.json':
            components = {str(key): float(value) for key, value in json.loads(body).items()}
            score = (
                components['reward']
                if 'reward' in components
                else statistics.fmean(components.values())
            )
        else:
            raise VerifierError(
                f'tests/test.sh wrote no reward (exit {tests.exit_code}): {tests.output}'
            )
    except (ValueError, TypeError, AttributeError) as exc:
        raise VerifierError(f'Unreadable {name}: {body!r}') from exc
    score = checked_score(score)
    return VerifierResult(
        score=score,
        passed=score >= 1 - SCORE_TOLERANCE,
        details=f'{name}: {score:g} (tests/test.sh exit {tests.exit_code})',
        reward_components=components,
        metrics={'tests_exit_code': tests.exit_code},
    )
