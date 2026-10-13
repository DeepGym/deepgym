"""The agent phase: run the model's bash script in a sandbox, then pack up its work."""

from __future__ import annotations

import math
from typing import NamedTuple

from deepgym.exceptions import DeepGymError, TimeoutError
from deepgym.harbor.task import HARNESS_PATHS, STAGING_DIR, HarborTask
from deepgym.sandboxes import Sandbox
from deepgym.session import SandboxBackend, check_workdir

DEFAULT_AGENT_TIMEOUT_S = 1800
"""Seconds the agent's script may run when task.toml sets no ``[agent].timeout_sec``."""

# Root-owned and readable, outside the work directory so it isn't graded.
_SCRIPT = f'{STAGING_DIR}/agent.sh'


class AgentRun(NamedTuple):
    """What the agent left behind."""

    workspace: bytes
    """The work directory as a gzipped tar."""

    workdir: str
    """Where the work directory is."""

    timed_out: bool
    """Whether the script was killed at the agent timeout."""


def run_agent(sandbox: Sandbox, task: HarborTask, script: str) -> AgentRun:
    """Run *script* with bash in *task*'s work directory, then archive that directory.

    The script runs as an unprivileged user with no capabilities and no network
    (:class:`~deepgym.session.SandboxBackend`), and owns the work directory. Its processes
    are killed before the archive is made. A timeout ends the script, not the run.

    Args:
        sandbox: A fresh sandbox from the task's image; deleted afterwards.
        task: The task; its workdir, else the image's WORKDIR, is the work directory.
        script: The model's output.

    Raises:
        DeepGymError: If the image's WORKDIR can't be the agent's (``/``, /root, /tmp ...).
        SandboxError: If setting up the sandbox or archiving fails.
    """
    workdir = task.workdir or _image_workdir(sandbox, task.name)
    backend = SandboxBackend(sandbox, workdir)  # deletes the sandbox if setup fails
    try:
        sandbox.upload(script.encode('utf-8'), _SCRIPT)
        timeout = math.ceil(task.agent_timeout_sec or DEFAULT_AGENT_TIMEOUT_S)
        try:
            backend.exec(f'bash {_SCRIPT}', timeout)
            timed_out = False
        except TimeoutError:
            timed_out = True
        return AgentRun(backend.archive_workdir(), workdir, timed_out)
    finally:
        backend.close()


def _image_workdir(sandbox: Sandbox, name: str) -> str:
    """Return the folder exec starts in (the image's WORKDIR); delete *sandbox* if unusable.

    Raises:
        DeepGymError: If that folder can't be the agent's.
    """
    try:
        return check_workdir(sandbox.exec('pwd', timeout=30).output.strip(), HARNESS_PATHS)
    except ValueError as exc:
        sandbox.delete()
        raise DeepGymError(f'{name}: {exc}; set [environment].workdir') from exc
    except BaseException:
        sandbox.delete()
        raise
