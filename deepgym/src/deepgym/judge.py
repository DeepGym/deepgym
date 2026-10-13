"""Lay out a verifier run so the solution executes isolated from the judge.

Every run uses three directories:

* judge (private): verifier.py, deepgym_solution.py, test_cases.json, support files
* solution: solution.py (the model's output)
* runtime: the solution-side server, readable by the isolated solution

The verifier reaches the solution only through ``deepgym_solution``
(``load_solution`` / ``run_solution``), which starts it in a separate, isolated
process: a dedicated Linux user in Daytona, ``sandbox-exec`` on macOS, or
``bwrap`` on Linux. The solution can't read the judge directory, the deepgym
package, or the judge's memory, and can't write to the judge's stdout.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from deepgym.exceptions import SandboxError
from deepgym.models import Environment
from deepgym.verifier_template import prepare_verifier

logger = logging.getLogger(__name__)

RUNTIME_FILE = 'deepgym_solution.py'
RUNTIME_SOURCE = Path(__file__).with_name('solution_runtime.py').read_text(encoding='utf-8')
# Extra seconds past the run's deadline for the verifier to finish and print.
DEADLINE_GRACE_S = 5

SANDBOX_ROOT = '/opt/deepgym'
"""Where judge, runtime and solution files go in a sandbox (see :data:`SANDBOX_SETUP`)."""
SOLVER_USER = 'dgsolver'

# Run once per sandbox (Daytona or Apple container), as its default user (root or a sudoer).
SANDBOX_SETUP = f"""
set -e
if [ "$(id -u)" = 0 ]; then SUDO=; else SUDO="sudo -n"; fi
id {SOLVER_USER} >/dev/null 2>&1 || \\
  $SUDO useradd --system --no-create-home --shell /usr/sbin/nologin {SOLVER_USER}
$SUDO mkdir -p {SANDBOX_ROOT}/judge {SANDBOX_ROOT}/runtime {SANDBOX_ROOT}/solution \\
  {SANDBOX_ROOT}/tmp
$SUDO chown -R "$(id -u):$(id -g)" {SANDBOX_ROOT}
chmod 755 {SANDBOX_ROOT} {SANDBOX_ROOT}/runtime
chmod 700 {SANDBOX_ROOT}/judge
chmod 755 {SANDBOX_ROOT}/solution
# Judge temp files: the solution user may enter known subdirectories, never list.
chmod 711 {SANDBOX_ROOT}/tmp
# 1 if the solution can get its own (offline) network namespace here.
netns=0
command -v setpriv >/dev/null && $SUDO unshare --net true 2>/dev/null && netns=1
echo "$(id -u) $(id -u {SOLVER_USER}) $(id -g {SOLVER_USER}) $(command -v python3) $netns"
"""


class JudgeJob(BaseModel):
    """Everything one verifier run needs."""

    model_config = ConfigDict(frozen=True)

    verifier: str
    """Runnable verifier script (already prepared from the environment's source)."""

    solution: str
    """The model's output, written to solution.py."""

    test_cases: list[dict] | None = None
    support_files: dict[str, str] = Field(default_factory=dict)
    timeout: int = Field(default=30, ge=1)
    case_timeout: int = Field(default=10, ge=1)
    env_vars: dict[str, str] | None = None

    @classmethod
    def for_environment(cls, env: Environment, model_output: str) -> JudgeJob:
        """Build the job for running *model_output* against *env*'s verifier."""
        return cls(
            verifier=prepare_verifier(env.verifier_source, env.verifier_mode),
            solution=model_output,
            test_cases=env.test_cases,
            support_files=env.support_files,
            timeout=env.timeout,
            case_timeout=env.case_timeout,
            env_vars=env.env_vars,
        )

    def judge_files(self) -> dict[str, str]:
        """Files for the private judge directory, keyed by relative path."""
        files = {**self.support_files, 'verifier.py': self.verifier, RUNTIME_FILE: RUNTIME_SOURCE}
        if self.test_cases is not None:
            files['test_cases.json'] = json.dumps(self.test_cases, ensure_ascii=False)
        return files

    def runtime_env(self, isolation: dict, runtime_path: str, python: str) -> dict[str, str]:
        """Variables telling deepgym_solution how to launch and bound the solution."""
        return {
            **(self.env_vars or {}),
            'DEEPGYM_ISOLATION': json.dumps(isolation),
            'DEEPGYM_RUNTIME': runtime_path,
            'DEEPGYM_PYTHON': python,
            'DEEPGYM_CASE_TIMEOUT': str(self.case_timeout),
        }

    def verifier_args(self, judge_dir: str, solution_path: str) -> list[str]:
        """Arguments after ``python``: verifier, solution, and test cases if any."""
        args = [f'{judge_dir}/verifier.py', solution_path]
        if self.test_cases is not None:
            args.append(f'{judge_dir}/test_cases.json')
        return args


# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------


@functools.cache
def local_isolation_kind() -> Literal['macos', 'bwrap', 'none']:
    """Pick the strongest sandbox this machine offers for local runs."""
    probes = {
        'macos': ['sandbox-exec', '-p', '(version 1)(allow default)', 'true'],
        'bwrap': ['bwrap', '--ro-bind', '/', '/', '--unshare-all', 'true'],
    }
    for kind, probe in probes.items():
        if shutil.which(probe[0]) is None:
            continue
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            if subprocess.run(probe, capture_output=True, timeout=10).returncode == 0:
                return kind  # type: ignore[return-value]
    logger.warning(
        'No sandbox-exec or bwrap available: local solutions run in a separate process but '
        'can read judge files. Use Daytona for untrusted code.'
    )
    return 'none'


_runs_root_lock = threading.Lock()
_runs_root: list[str] = []


def _local_runs_root() -> str:
    """One parent for every local run in this process, hidden from all solutions.

    Locked: parallel batch runs must share it, or each would hide only its own.
    """
    with _runs_root_lock:
        if not _runs_root:
            root = os.path.realpath(tempfile.mkdtemp(prefix='deepgym_runs_'))
            atexit.register(shutil.rmtree, root, ignore_errors=True)
            _runs_root.append(root)
        return _runs_root[0]


def local_isolation(expose: list[str]) -> dict:
    """Isolation spec for a local run.

    Hides the home directory (repo clones, other worktrees, ~/.deepgym), all
    runs, and the deepgym package. Re-exposes the Python installation and this
    run's own *expose* directories; the package is hidden again even if it sits
    inside the Python installation.
    """
    kind = local_isolation_kind()
    if kind == 'none':
        return {'kind': 'none'}
    package = os.path.realpath(Path(__file__).resolve().parent)
    hide = [str(Path.home()), _local_runs_root()]
    pythons = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    return {
        'kind': kind,
        'hide': [os.path.realpath(p) for p in hide if os.path.exists(p)],
        'expose': [*sorted(pythons), *expose],
        'rehide': [package],
    }


@contextlib.contextmanager
def local_judge(job: JudgeJob) -> Iterator[tuple[list[str], dict[str, str], str]]:
    """Materialize *job* under the hidden runs root; yield (argv, env, cwd) for the judge.

    Sibling runs (e.g. other rollouts in a GRPO batch) can't read each other's
    judge files, solutions, or scratch: they all live under the hidden root.
    """
    run_dir = tempfile.mkdtemp(dir=_local_runs_root())
    judge_dir, runtime_dir, solution_dir, tmp_dir = (
        f'{run_dir}/{part}' for part in ('judge', 'runtime', 'solution', 'tmp')
    )
    for path in (judge_dir, runtime_dir, solution_dir, tmp_dir):
        os.mkdir(path)
    try:
        for rel, content in job.judge_files().items():
            target = Path(judge_dir, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
        runtime_path = f'{runtime_dir}/{RUNTIME_FILE}'
        Path(runtime_path).write_text(RUNTIME_SOURCE, encoding='utf-8')
        solution_path = f'{solution_dir}/solution.py'
        Path(solution_path).write_text(job.solution, encoding='utf-8')
        isolation = local_isolation([runtime_dir, solution_dir])
        env = {
            **os.environ,
            **job.runtime_env(isolation, runtime_path, sys.executable),
            'DEEPGYM_DEADLINE': str(time.time() + job.timeout),
            # Verifier temp files (and solution scratch dirs) stay inside this run.
            'TMPDIR': tmp_dir,
        }
        yield [sys.executable, *job.verifier_args(judge_dir, solution_path)], env, judge_dir
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sandboxes (Daytona, Apple container)
# ---------------------------------------------------------------------------


# Runs as root in the solution's fresh network namespace: bring loopback up (it starts
# down and slim images have no `ip`), then exec the rest of the command, which drops root.
# Its working directory is the solution's, so it runs with `python -I -S` (no cwd, site
# or PYTHON* variables on the import path) and execs with a fixed PATH and nothing else.
_LOOPBACK_UP_THEN_EXEC = """
import fcntl, os, socket, struct, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
flags = struct.unpack('16sH', fcntl.ioctl(s, 0x8913, struct.pack('16sH', b'lo', 0)))[1]
fcntl.ioctl(s, 0x8914, struct.pack('16sH', b'lo', flags | 1))
os.execvpe(sys.argv[1], sys.argv[1:], {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'})
"""


def sandbox_isolation(setup_output: str, *, require_offline: bool = False) -> tuple[dict, str]:
    """Turn SANDBOX_SETUP's output into (isolation spec, python path).

    The solution runs as the unprivileged solver user with every capability
    dropped and no-new-privs (a setuid binary can't regain root) and, where the
    sandbox allows it, in its own network namespace (loopback only: no internet,
    no host). ``own_network`` in the spec says it got one.

    Raises:
        SandboxError: If *require_offline* is set but the sandbox can't make that namespace.
    """
    judge_uid, uid, gid, python, netns = setup_output.split()[-5:]
    if require_offline and netns != '1':
        raise SandboxError(
            'Cannot cut the solution off the network here; the image needs util-linux '
            '(unshare, setpriv)'
        )
    drop = ['setpriv', f'--reuid={uid}', f'--regid={gid}', '--clear-groups']
    drop += ['--inh-caps=-all', '--bounding-set=-all', '--no-new-privs', '--']
    offline: list[str] = []
    if netns == '1':
        offline = ['unshare', '--net', '--', python, '-I', '-S', '-c', _LOOPBACK_UP_THEN_EXEC]
    if judge_uid == '0':
        prefix = [*offline, *drop]
    elif offline:
        prefix = ['sudo', '-n', *offline, *drop]
    else:
        prefix = ['sudo', '-n', '-u', SOLVER_USER, '--']
    return {'kind': 'user', 'prefix': prefix, 'own_network': netns == '1'}, python


def sandbox_uploads(job: JudgeJob) -> list[tuple[bytes, str]]:
    """(content, sandbox path) pairs for the judge, runtime, and solution files."""
    uploads = [
        (content.encode('utf-8'), f'{SANDBOX_ROOT}/judge/{rel}')
        for rel, content in job.judge_files().items()
    ]
    uploads.append((RUNTIME_SOURCE.encode('utf-8'), f'{SANDBOX_ROOT}/runtime/{RUNTIME_FILE}'))
    uploads.append((job.solution.encode('utf-8'), f'{SANDBOX_ROOT}/solution/solution.py'))
    return uploads


def sandbox_judge(
    job: JudgeJob, setup_output: str, *, require_offline: bool = False
) -> tuple[str, dict[str, str]]:
    """Return (bash command, env) that run the verifier inside the sandbox.

    Raises:
        SandboxError: If *require_offline* is set but the solution can't be cut off the network.
    """
    isolation, python = sandbox_isolation(setup_output, require_offline=require_offline)
    judge_dir = f'{SANDBOX_ROOT}/judge'
    args = job.verifier_args(judge_dir, f'{SANDBOX_ROOT}/solution/solution.py')
    command = (
        f'umask 077 && cd {judge_dir} && DEEPGYM_DEADLINE=$(( $(date +%s) + {job.timeout} )) '
        f'exec {shlex.quote(python)} {shlex.join(args)}'
    )
    env = job.runtime_env(isolation, f'{SANDBOX_ROOT}/runtime/{RUNTIME_FILE}', python)
    # Private temp dir + umask 077: the judge's own temp files stay unreadable to the
    # solution; run_solution shares only the directories it hands over.
    env['TMPDIR'] = f'{SANDBOX_ROOT}/tmp'
    return f'bash -c {shlex.quote(command)}', env
