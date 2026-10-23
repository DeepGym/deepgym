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
from uuid import uuid4

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


def ids_are_exclusive(*users: str) -> str:
    """Return a shell check that each of *users* that exists holds ids no other account has.

    A user's uid and gid must be nonzero, that account's alone, and its only group, so
    dropping to them grants that account's rights and nobody else's. Ruling out root isn't
    enough: a uid or primary gid shared with any other account, or a supplementary group
    the image put the user in (a stray ``shadow:x:42:dgsolver`` line), would carry that
    account's or group's rights too. The passwd gid check catches an account whose primary
    gid has no group entry. The check prints the first user that fails and exits 1.
    """
    return (
        f'for u in {shlex.join(users)}; do '
        'id "$u" >/dev/null 2>&1 || continue; '
        'uid=$(id -u "$u"); gid=$(id -g "$u"); '
        '[ "$uid" != 0 ] && [ "$gid" != 0 ] && '
        '[ "$(getent passwd | cut -d: -f3 | grep -Fcx "$uid")" = 1 ] && '
        '[ "$(getent passwd | cut -d: -f4 | grep -Fcx "$gid")" = 1 ] && '
        '[ "$(getent group | cut -d: -f3 | grep -Fcx "$gid")" = 1 ] && '
        '[ "$(id -Gn "$u")" = "$u" ] '
        '|| { echo "$u is root or shares an id or a group with another account"; exit 1; }; '
        'done'
    )


# Run once per sandbox (Daytona or Apple container), as its default user (root or a sudoer).
# A solver account the image already has is used only if its ids are its own.
SANDBOX_SETUP = f"""
set -e
if [ "$(id -u)" = 0 ]; then SUDO=; else SUDO="sudo -n"; fi
id {SOLVER_USER} >/dev/null 2>&1 || \\
  $SUDO useradd --system --no-create-home --shell /usr/sbin/nologin {SOLVER_USER}
{ids_are_exclusive(SOLVER_USER)}
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
    consulted_mark: str = Field(default_factory=lambda: uuid4().hex)
    """A token the judge prints on stdout once the solution's output is in play (it read
    the solution or started it), so a run that then times out or loses its sandbox is
    the solution's doing. Random per run, so the solution can't print it first."""

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

    def runtime_env(
        self, isolation: dict, runtime_path: str, python: str, solution_dir: str
    ) -> dict[str, str]:
        """Variables telling deepgym_solution how to launch and bound the solution, and the
        verifier's crash guard where the solution is and how to announce consulting it."""
        return {
            **(self.env_vars or {}),
            'DEEPGYM_ISOLATION': json.dumps(isolation),
            'DEEPGYM_RUNTIME': runtime_path,
            'DEEPGYM_PYTHON': python,
            'DEEPGYM_CASE_TIMEOUT': str(self.case_timeout),
            'DEEPGYM_POLICY_DIR': solution_dir,
            'DEEPGYM_CONSULTED': self.consulted_mark,
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
    return 'none'


ALLOW_UNSANDBOXED_ENV = 'DEEPGYM_ALLOW_UNSANDBOXED'
"""Set to 1 to run local solutions with no sandbox, where none works; they can then read
every file this user can, verifiers and answer keys included."""


def unsandboxed_allowed() -> bool:
    """Whether the user opted in to local runs with no sandbox (:data:`ALLOW_UNSANDBOXED_ENV`)."""
    return os.environ.get(ALLOW_UNSANDBOXED_ENV, '').lower() in ('1', 'true', 'yes')


# What a local solution may read besides the Python installation and its own run: the
# operating system and the software trees a Python links against (Homebrew's, MacPorts',
# Nix's, Apple's developer tools). Nothing else is readable, so nothing a user or another
# process wrote is: home, temp directories (other runs and their verifiers), mounted
# volumes, daemon sockets in /run and /var/run.
_SYSTEM_READ = {
    'darwin': (
        '/System',
        '/usr',
        '/bin',
        '/sbin',
        '/dev',
        '/private/etc',
        '/private/var/db/timezone',
        '/private/var/select',
        '/private/var/db/xcode_select_link',
        '/Library/Frameworks',
        '/Library/Developer/CommandLineTools',
        '/Applications/Xcode.app/Contents/Developer',
        '/opt/homebrew',
        '/opt/local',
        '/nix/store',
    ),
    'linux': (
        '/usr',
        '/bin',
        '/sbin',
        '/lib',
        '/lib32',
        '/lib64',
        '/libx32',
        '/etc',
        '/sys',
        '/nix/store',
        '/gnu/store',
    ),
}
# Service data inside those trees (databases, logs), hidden again.
_SYSTEM_HIDE = ('/opt/homebrew/var', '/opt/local/var', '/usr/local/var')


_runs_root_lock = threading.Lock()
_runs_root: list[str] = []
_RUNS_PREFIX = 'deepgym_runs_'
_RUNS_LOCK = '.owner.lock'


def local_runs_root() -> str:
    """One parent for every local run in this process, unreadable to all solutions.

    Anything else no local solution may read (an approval's snapshot) goes here too.
    It is removed at exit; this process holds a lock in it for life, so a later process
    can tell a root left by one that was killed and remove it (see :func:`_remove_stale_roots`).

    Locked: parallel batch runs must share it.
    """
    import fcntl  # Unix only, like the local sandboxes

    with _runs_root_lock:
        if not _runs_root:
            _remove_stale_roots()
            root = os.path.realpath(tempfile.mkdtemp(prefix=_RUNS_PREFIX))
            # Locked before it takes the name another process looks for, so it is never
            # seen unlocked while this process lives. The kernel drops the lock however
            # this process ends; the descriptor stays open until then.
            fd = os.open(f'{root}/.locking', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.rename(f'{root}/.locking', f'{root}/{_RUNS_LOCK}')
            atexit.register(shutil.rmtree, root, ignore_errors=True)
            _runs_root.append(root)
        return _runs_root[0]


def _remove_stale_roots() -> None:
    """Remove this user's runs roots whose process is gone: killed, it never ran its exit
    cleanup. A root whose lock another process holds, or that has none, is left alone."""
    import fcntl

    for root in Path(tempfile.gettempdir()).glob(f'{_RUNS_PREFIX}*'):
        try:
            if root.is_symlink() or root.stat().st_uid != os.getuid():
                continue
            fd = os.open(root / _RUNS_LOCK, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue  # its process is alive
        finally:
            os.close(fd)
        shutil.rmtree(root, ignore_errors=True)


def _deepgym_packages() -> list[str]:
    """Every deepgym package a solution's Python could import: this one, and any other
    copy on the import path (an installed wheel next to a source checkout, say)."""
    here = os.path.realpath(Path(__file__).resolve().parent)
    found = (os.path.realpath(Path(entry or os.curdir, 'deepgym')) for entry in sys.path)
    others = [path for path in found if Path(path, '__init__.py').is_file()]
    return list(dict.fromkeys([here, *others]))


def sandbox_python() -> str:
    """This Python, by a path a local sandbox has (see :func:`local_isolation`).

    The sandbox holds the installation's real directories, so an alias outside them (a
    ``~/bin`` link, a profile link) goes by its target. A venv keeps its own directory, by
    its real path, so the venv and its packages stay in use.
    """
    executable = os.path.abspath(sys.executable)
    folder = os.path.realpath(os.path.dirname(executable))
    prefixes = {os.path.realpath(p) for p in (sys.prefix, sys.exec_prefix)}
    if any(Path(folder).is_relative_to(prefix) for prefix in prefixes):
        return os.path.join(folder, os.path.basename(executable))
    return os.path.realpath(executable)


@functools.cache
def _warn_unsandboxed() -> None:
    """Say once per process that local solutions run unsandboxed, so a stray
    :data:`ALLOW_UNSANDBOXED_ENV` never goes unnoticed (results also record it)."""
    logger.warning(
        '%s is set and no sandbox works: local solutions run unsandboxed and can read '
        'every file you can, verifiers included',
        ALLOW_UNSANDBOXED_ENV,
    )


def local_isolation(expose: list[str]) -> dict:
    """Isolation spec for a local run: what its solution may read.

    Only the operating system and the software trees a Python links against (see
    :data:`_SYSTEM_READ`), the Python installation and this run's own *expose*
    directories. Every deepgym package stays unreadable even inside those.

    Raises:
        SandboxError: If this machine has no working sandbox-exec or bwrap, unless
            :data:`ALLOW_UNSANDBOXED_ENV` is set.
    """
    kind = local_isolation_kind()
    if kind == 'none':
        if unsandboxed_allowed():
            _warn_unsandboxed()
            return {'kind': 'none'}
        raise SandboxError(
            'No working sandbox-exec or bwrap on this machine, so a local solution could '
            'read every file you can, its verifier included. Install bubblewrap (Linux), '
            f'use a sandbox provider (mode="docker", "daytona", ...), or set '
            f'{ALLOW_UNSANDBOXED_ENV}=1 to run it unsandboxed anyway.'
        )
    system = _SYSTEM_READ['darwin' if kind == 'macos' else 'linux']
    pythons = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    hidden = [path for path in _SYSTEM_HIDE if os.path.exists(path)]
    return {
        'kind': kind,
        # Unresolved: bwrap recreates a symlinked tree (/bin -> usr/bin) as the link.
        'read': [path for path in system if os.path.lexists(path)] + sorted(pythons) + expose,
        'hide': [*_deepgym_packages(), *hidden],
    }


@contextlib.contextmanager
def local_judge(
    job: JudgeJob, *, confined: bool = False
) -> Iterator[tuple[list[str], dict[str, str], str]]:
    """Materialize *job* under the hidden runs root; yield (argv, env, cwd) for the judge.

    Sibling runs (e.g. other rollouts in a GRPO batch) can't read each other's
    judge files, solutions, or scratch: they all live under the hidden root.

    *confined* also sandboxes the judge itself, for verifiers that might run code
    in-process (mutation testing). The whole run goes inside one sandbox-exec or bwrap
    sandbox: it writes only to the run's temp directory, sees nothing of the host
    beyond the Python install and this run, has no network and gets a minimal
    environment. The solution then runs in plain child processes inside it, since
    sandbox-exec can't nest: the host is protected from everything in the run, but
    the solution is not kept from the judge's files, so use this only for code that
    isn't adversarial, such as mutants of a reference. The run's processes are torn
    down with it and CPU-capped (see the runtime's ``--confine``).

    Raises:
        SandboxError: If *confined* but this machine has no sandbox-exec or bwrap.
    """
    run_dir = tempfile.mkdtemp(dir=local_runs_root())
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
        judge = [sys.executable, *job.verifier_args(judge_dir, solution_path)]
        deadline = {'DEEPGYM_DEADLINE': str(time.time() + job.timeout), 'TMPDIR': tmp_dir}
        if confined:
            if local_isolation_kind() == 'none':  # never unsandboxed, opt-in or not
                raise SandboxError('confining the judge needs sandbox-exec or bwrap')
            outer = local_isolation([run_dir])
            python = sandbox_python()
            env = {
                'PATH': os.environ.get('PATH', os.defpath),
                **job.runtime_env(outer, runtime_path, python, solution_dir),
                **deadline,
            }
            confine = [sys.executable, '-I', '-B', runtime_path, '--confine', tmp_dir]
            yield [*confine, python, *judge[1:]], env, judge_dir
            return
        isolation = local_isolation([runtime_dir, solution_dir])
        # Verifier temp files (and solution scratch dirs) stay inside this run.
        runtime_env = job.runtime_env(isolation, runtime_path, sandbox_python(), solution_dir)
        env = {**os.environ, **runtime_env, **deadline}
        yield judge, env, judge_dir
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
        SandboxError: If the solver's uid or gid is root or unreadable, or if
            *require_offline* is set but the sandbox can't make that namespace.
    """
    judge_uid, uid, gid, python, netns = setup_output.split()[-5:]
    # SANDBOX_SETUP proved the ids exclusive; this guards the parse before it reaches setpriv.
    # Only sandbox_judge_by_name passes the solver's name instead.
    if not all(v == SOLVER_USER or (v.isdecimal() and int(v) > 0) for v in (uid, gid)):
        raise SandboxError(f'Refusing to run the solution as root or an unknown id: {uid}:{gid}')
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
    runtime_path = f'{SANDBOX_ROOT}/runtime/{RUNTIME_FILE}'
    env = job.runtime_env(isolation, runtime_path, python, f'{SANDBOX_ROOT}/solution')
    # Private temp dir + umask 077: the judge's own temp files stay unreadable to the
    # solution; run_solution shares only the directories it hands over.
    env['TMPDIR'] = f'{SANDBOX_ROOT}/tmp'
    return f'bash -c {shlex.quote(command)}', env


def sandbox_judge_by_name(job: JudgeJob) -> tuple[str, dict[str, str]]:
    """Return :func:`sandbox_judge`'s (command, env) for a root shell that runs
    SANDBOX_SETUP itself, so its output isn't known here (an exported Harbor task).

    The solver is named rather than numbered (setpriv takes either) and gets no network
    namespace of its own: the sandbox itself must be offline.
    """
    return sandbox_judge(job, f'0 {SOLVER_USER} {SOLVER_USER} python3 0')
