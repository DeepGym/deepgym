"""Harbor runs against fake sandboxes: what reaches the agent, what reaches the grader.

The fake records every command and upload; real containers are in test_harbor_live.py.
"""

import base64
import contextlib
import io
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from deepgym.benchmark_envs import TerminalBenchEnvironment
from deepgym.exceptions import DeepGymError, SandboxError, VerifierError
from deepgym.harbor import HarborEnvironment, HarborTask
from deepgym.harbor.agent import run_agent
from deepgym.harbor.grading import grade
from deepgym.harbor.task import pack_files
from deepgym.models import Environment
from deepgym.sandboxes import TRUSTED_PATH, SandboxTimeoutError, build_image
from deepgym.sandboxes.docker import DockerContainer
from deepgym.session import SandboxBackend
from tests.test_harbor import BUILT, HELLO
from tests.test_sandboxes import FakeSandbox

_IDS = '997 997 996 996 0 /usr/bin/setpriv /usr/bin/nsenter -'
_WORKSPACE = pack_files({'hello.txt': b'hello\n'})
_ENCODED = ('base64 /opt', base64.b64encode(_WORKSPACE).decode())
_PASS = ('for name in reward.json', 'reward.txt\n1\n')
_AS_AGENT = '/usr/bin/setpriv --reuid=997 --regid=997'
_RUN_TESTS = '/bin/bash -c /tests/test.sh'


def _box(*answers: tuple[str, str], workdir: str = '/app', blocked: bool = True) -> FakeSandbox:
    """A fake sandbox whose setup reports the session users and *workdir* as its realpath."""
    users = ('useradd', f'{_IDS}\n{workdir}')
    return FakeSandbox(network_blocked=blocked, answers=[users, *answers])


def _inner(command: str) -> str:
    """The command a :func:`deepgym.session.trusted` wrapper runs, else *command* itself."""
    return shlex.split(command)[-1] if command.startswith('/usr/bin/env -i ') else command


def _index(box: FakeSandbox, needle: str) -> int:
    return next(i for i, command in enumerate(box.commands) if needle in command)


# ---------------------------------------------------------------------------
# Agent phase
# ---------------------------------------------------------------------------


def test_agent_runs_unprivileged_and_hands_over_only_its_work_directory() -> None:
    box = _box(_ENCODED)

    run = run_agent(box, HarborTask.from_dir(HELLO), 'echo hello > /app/hello.txt')

    assert run.workspace == _WORKSPACE and run.workdir == '/app' and not run.timed_out
    script = _index(box, 'bash /opt/deepgym-harbor/agent.sh')
    assert box.commands[script].startswith(_AS_AGENT)
    packed = _index(box, '-czf')
    assert packed > script
    assert box.commands[packed].startswith('/usr/bin/env -i ')
    assert _inner(box.commands[packed]).startswith(_AS_AGENT)
    inner = _inner(box.commands[packed])
    assert (
        inner.index('kill -9 -1') < inner.index('"$(realpath /app)" = /app') < inner.index('tar -C')
    )
    assert "--no-new-privs -- sh -c 'kill -9 -1'" in _inner(box.commands[packed])
    assert box.uploads == ['/opt/deepgym-harbor/agent.sh']
    assert not [command for command in box.commands if '/tests' in command or '/logs' in command]
    assert box.deleted


def test_agent_workdir_defaults_to_the_image_workdir_and_refuses_root() -> None:
    task = HarborTask.from_dir(BUILT)
    box = _box(('pwd', '/srv/app'), _ENCODED, workdir='/srv/app')
    assert run_agent(box, task, '').workdir == '/srv/app'

    for workdir in ('/', '/root', '/tmp'):
        box = _box(('pwd', workdir))
        with pytest.raises(DeepGymError, match='set \\[environment\\].workdir'):
            run_agent(box, task, '')
        assert box.deleted and box.commands == ['pwd']


def test_agent_timeout_ends_the_script_not_the_run() -> None:
    box = _box(_ENCODED)
    real_exec = box.exec

    def exec_(command: str, **kwargs: object) -> object:
        if 'agent.sh' in command:
            raise SandboxTimeoutError('command exceeded 60s')
        return real_exec(command, **kwargs)

    box.exec = exec_  # type: ignore[method-assign]
    assert run_agent(box, HarborTask.from_dir(HELLO), 'sleep 999').timed_out


def test_a_corrupted_archive_is_a_sandbox_error() -> None:
    box = _box(('base64 /opt', 'H4sI\nnot base64: warning from stderr\n'))
    with pytest.raises(SandboxError, match='read the archive'):
        run_agent(box, HarborTask.from_dir(HELLO), '')
    assert box.deleted


def test_a_failed_delete_costs_neither_the_run_nor_the_grade() -> None:
    agent_box, grader_box = _box(_ENCODED), _box(_PASS)
    agent_box.delete_result = grader_box.delete_result = False

    run = run_agent(agent_box, HarborTask.from_dir(HELLO), 'echo hello > /app/hello.txt')
    result = grade(grader_box, HarborTask.from_dir(HELLO), _WORKSPACE, '/app')

    assert run.workspace == _WORKSPACE
    assert result.verifier_result.score == 1.0
    assert agent_box.deleted and grader_box.deleted


def test_agent_sandbox_is_deleted_when_archiving_fails() -> None:
    box = _box()
    real_exec = box.exec
    box.exec = lambda command, **kw: (  # type: ignore[method-assign]
        real_exec(command, **kw)._replace(exit_code=2)
        if '-czf' in command
        else real_exec(command, **kw)
    )
    with pytest.raises(SandboxError, match='archive'):
        run_agent(box, HarborTask.from_dir(HELLO), '')
    assert box.deleted


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def test_grader_restores_then_uploads_tests_then_reads_the_reward_separately() -> None:
    box = _box(_PASS)

    result = grade(box, HarborTask.from_dir(HELLO), _WORKSPACE, '/app')

    assert result.verifier_result.score == 1.0
    restored = _index(box, 'tar -xzpf')
    assert _AS_AGENT in box.commands[restored]
    assert 'tar -xzpf - -C /app < ' in box.commands[restored]
    assert box.commands[restored].index('chown -hR 0:0 /app') > box.commands[restored].index('tar')
    assert box.commands[restored].index('chmod -R a-st /app') > box.commands[restored].index(
        'chown'
    )
    staged = _index(box, 'rm -rf /tests /logs/verifier')
    tests = _index(box, _RUN_TESTS)
    reward = _index(box, 'for name in reward.json')
    assert restored < staged < tests < reward
    assert 'test.sh' not in box.commands[reward]
    assert not box.commands[tests].startswith('setpriv')
    assert box.uploads[-1] == '/opt/deepgym-harbor/tests.tgz'
    assert box.deleted


def test_grader_without_a_reward_is_a_verifier_error_and_still_cleans_up() -> None:
    box = _box(('for name in reward.json', 'none'))
    with pytest.raises(VerifierError, match='no reward'):
        grade(box, HarborTask.from_dir(HELLO), _WORKSPACE, '/app')
    assert box.deleted


def test_offline_tasks_run_their_tests_in_the_offline_namespace(tmp_path: Path) -> None:
    online = _box(_PASS, blocked=False)
    grade(online, HarborTask.from_dir(HELLO), _WORKSPACE, '/app')
    assert online.commands[_index(online, _RUN_TESTS)].startswith('/usr/bin/nsenter --net=')

    public = HarborTask.from_dir(HELLO).model_copy(update={'verifier_network': 'public'})
    box = _box(_PASS, blocked=False)
    grade(box, public, _WORKSPACE, '/app')
    assert box.commands[_index(box, _RUN_TESTS)].startswith('/usr/bin/env -i ')


def test_shell_syntax_in_the_workdir_is_always_quoted(tmp_path: Path) -> None:
    workdir = '/app/$(touch pwned) `id`'
    task = HarborTask.from_dir(HELLO).model_copy(update={'workdir': workdir})
    agent, grader = _box(_ENCODED, workdir=workdir), _box(_PASS, workdir=workdir)

    run = run_agent(agent, task, 'true')
    grade(grader, task, run.workspace, run.workdir)

    quoted = shlex.quote(workdir)
    mentions = [_inner(c) for c in agent.commands + grader.commands if 'touch pwned' in c]
    assert mentions and all(quoted in command for command in mentions)
    assert all(command.count('touch pwned') == command.count(quoted) for command in mentions)


class _EnvBox(FakeSandbox):
    """A fake sandbox that also records the environment each command gets."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.envs: list[dict[str, str] | None] = []

    def exec(self, command: str, *, env: dict[str, str] | None = None, **kwargs: object) -> object:
        self.envs.append(dict(env) if env else None)
        return super().exec(command, env=env, **kwargs)


def _env_box(*answers: tuple[str, str], blocked: bool = True, workdir: str = '/app') -> _EnvBox:
    """An _EnvBox whose setup reports the session users and *workdir*."""
    users = ('useradd', f'{_IDS}\n{workdir}')
    return _EnvBox(network_blocked=blocked, answers=[users, *answers])


def test_tests_run_with_an_allowlisted_env_and_a_resolved_path() -> None:
    task = HarborTask.from_dir(HELLO).model_copy(update={'verifier_env': {'MODE': 'strict'}})
    agent = _box(_ENCODED)
    grader = _env_box(
        # The grader's own image env (session capture), then the probe grading reads for the
        # tests: PATH //app/bin:/usr/bin:.:/opt/tools, LANG, no LC_ALL, TERM.
        ('/usr/bin/env -0', 'PATH=/usr/bin\0LD_PRELOAD=/app/x.so\0'),
        ("printf '%s", '//app/bin:/usr/bin:.:/opt/tools\0C.UTF-8\0\0xterm'),
        # realpath resolves //app/bin and the /opt/tools link into /app; /usr/bin stays.
        ('realpath -ez', '/app/bin\0/usr/bin\0/app/tools\0'),
        _PASS,
    )

    run = run_agent(agent, task, 'true')
    grade(grader, task, run.workspace, run.workdir)

    tests = shlex.split(grader.commands[_index(grader, _RUN_TESTS)])
    start = tests.index('/usr/bin/env') + 3  # past `env -i --`, after any nsenter prefix
    assigned = dict(item.split('=', 1) for item in tests[start : tests.index('/bin/bash')])
    assert assigned == {
        'LANG': 'C.UTF-8',
        'TERM': 'xterm',
        'HOME': '/root',
        'MODE': 'strict',
        'PATH': '/usr/bin',
    }


def test_env_ends_its_options_before_any_variable() -> None:
    box = _box()
    backend = SandboxBackend(box, '/app')

    backend.exec_root('true', 5, offline=False, env={'HOME': '/root'})
    backend.exec_trusted('true', 5)

    # `--` comes first, so env reads every NAME=VALUE as a variable, never as an option.
    root, wrapped = (shlex.split(command) for command in box.commands[-2:])
    assert root[:4] == ['/usr/bin/env', '-i', '--', 'HOME=/root']
    assert wrapped[:4] == ['/usr/bin/env', '-i', '--', f'PATH={TRUSTED_PATH}']


def test_backend_clears_the_whole_image_env_for_root_commands() -> None:
    box = _env_box(('/usr/bin/env -0', 'PATH=/app/bin:/usr/bin\0LD_PRELOAD=/app/x.so\0LANG=C\0'))
    backend = SandboxBackend(box, '/app')

    with contextlib.suppress(SandboxError):
        backend.archive_workdir()  # a privileged root command
    backend.exec('true', timeout=5)

    cleared = {'PATH': '', 'LD_PRELOAD': '', 'LANG': ''}
    # Every root command names each image variable but empties it, so nothing it sets
    # (a loader path, a startup file) reaches the first binary.
    privileged = [env for cmd, env in zip(box.commands, box.envs) if '-czf ' in cmd]
    assert privileged and all(env == cleared for env in privileged)
    # The agent keeps the image PATH and locale, with every loader variable emptied.
    agent_env = box.envs[-1]
    assert agent_env['PATH'] == '/app/bin:/usr/bin' and agent_env['LANG'] == 'C'
    assert agent_env['LD_PRELOAD'] == '' and agent_env['HOME'] == '/home/dgagent'


def test_backend_owns_where_the_workdir_really_leads() -> None:
    linked = _box(workdir='/usr/bin')
    with pytest.raises(SandboxError, match="'/app' leads to '/usr/bin'"):
        SandboxBackend(linked, '/app')
    assert linked.deleted and not [c for c in linked.commands if 'chown' in c]

    spelled = _box()
    with pytest.raises(SandboxError, match='overlaps /usr'):
        SandboxBackend(spelled, '//usr/bin')
    assert spelled.deleted and not spelled.commands


# ---------------------------------------------------------------------------
# The environment: two sandboxes per run
# ---------------------------------------------------------------------------


class _FakeClient:
    """Stands in for DeepGym: hands out a new fake sandbox per call."""

    backend = 'docker'

    def __init__(self) -> None:
        self.boxes: list[FakeSandbox] = []
        self.images: list[str | None] = []

    def open_sandbox(self, env: Environment) -> FakeSandbox:
        self.images.append(env.image)
        box = _box(_ENCODED, _PASS, ('pwd', '/app'))  # every task here works in /app
        box.id = f'fake-{len(self.boxes)}'
        self.boxes.append(box)
        return box


def test_each_run_uses_a_fresh_grader_from_the_same_image() -> None:
    client = _FakeClient()
    env = HarborEnvironment.from_dir(HELLO)

    result = env.run_with_deepgym(client, 'echo hello > /app/hello.txt')  # type: ignore[arg-type]

    agent, grader = client.boxes
    assert client.images == ['python:3.12-slim', 'python:3.12-slim']
    assert result.score == 1.0 and result.sandbox_id == grader.id
    assert '/opt/deepgym-harbor/tests.tgz' not in agent.uploads
    assert '/opt/deepgym-harbor/agent.sh' not in grader.uploads
    assert agent.deleted and grader.deleted


def test_a_dockerfile_task_boots_the_image_built_for_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        'deepgym.harbor.environment.build_image',
        lambda provider, context: built.append((provider, context)) or 'deepgym-build:abc',
    )
    client = _FakeClient()

    HarborEnvironment.from_dir(BUILT).run_with_deepgym(client, 'echo hello > hello.txt')  # type: ignore[arg-type]

    assert built == [('docker', BUILT / 'environment')]
    assert client.images == ['deepgym-build:abc', 'deepgym-build:abc']


def test_shell_syntax_in_a_task_name_or_instruction_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    instruction = 'Say hello. `touch pwned-tick` $(touch pwned-dollar)\n'
    task = tmp_path / 'tasks' / 'x$(touch pwned-name)`touch pwned-tick2`'
    shutil.copytree(HELLO, task)
    (task / 'instruction.md').write_text(instruction, encoding='utf-8')
    client = _FakeClient()
    env = TerminalBenchEnvironment(task_root=task.parent)

    result = env.run_with_deepgym(client, 'echo hello > /app/hello.txt', task_id=task.name)  # type: ignore[arg-type]

    assert result.score == 1.0
    assert env.get_task(task.name).instruction == instruction
    assert not [command for box in client.boxes for command in box.commands if 'touch' in command]
    assert not list(tmp_path.rglob('pwned*'))


def test_reference_solution_carries_the_whole_solution_folder() -> None:
    script = HarborEnvironment.from_dir(HELLO).reference_solution

    assert script is not None and script.endswith('bash "$solution/solve.sh"\n')
    (encoded,) = re.findall(r"echo '([A-Za-z0-9+/=]+)' \| base64 -d", script)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded))) as archive:
        assert sorted(archive.getnames()) == ['answer.txt', 'solve.sh']
    assert HarborEnvironment.from_dir(BUILT).reference_solution is None


# ---------------------------------------------------------------------------
# Building Dockerfiles and SandboxBackend's work directory
# ---------------------------------------------------------------------------


def test_build_reuses_an_identical_image_and_builds_a_changed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = tmp_path / 'ctx $(touch pwned)'
    context.mkdir()
    (context / 'Dockerfile').write_text('FROM python:3.12-slim\n', encoding='utf-8')
    calls: list[list[str]] = []
    have: set[str] = set()

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert isinstance(argv, list) and not kwargs.get('shell')
        calls.append(argv)
        missing = argv[1:3] == ['image', 'inspect'] and argv[3] not in have
        if argv[1] == 'build':
            have.add(argv[3])
        return subprocess.CompletedProcess(argv, int(missing), b'', b'')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    tag = DockerContainer.build(context)
    assert DockerContainer.build(context) == tag and tag.startswith('deepgym-build:')
    assert [argv[1] for argv in calls] == ['image', 'build', 'image']
    assert calls[1] == ['docker', 'build', '--tag', tag, str(context)]
    (context / 'Dockerfile').write_text('FROM python:3.11-slim\n', encoding='utf-8')
    changed = DockerContainer.build(context)
    assert changed != tag

    (context / 'Dockerfile').chmod(0o600)
    moded = DockerContainer.build(context)
    assert moded != changed
    (context / 'link').symlink_to('Dockerfile')
    linked = DockerContainer.build(context)
    (context / 'link').unlink()
    (context / 'link').symlink_to('elsewhere')
    assert DockerContainer.build(context) not in (moded, linked)


def test_only_container_clis_build_images(tmp_path: Path) -> None:
    for provider in ('modal', 'e2b', 'daytona'):
        with pytest.raises(DeepGymError, match='registry images only'):
            build_image(provider, tmp_path)


def test_backend_takes_over_a_given_workdir_and_refuses_its_own_paths() -> None:
    box = _box(workdir='/srv/my app')
    SandboxBackend(box, '/srv/my app').exec('ls', timeout=5)
    assert _inner(box.commands[1]) == "chown -R dgagent: '/srv/my app'"

    for workdir in ('/', '/opt', '/usr/local/bin', '/root'):
        refused = _box()
        with pytest.raises(SandboxError, match='Work directory'):
            SandboxBackend(refused, workdir)
        assert refused.deleted and not refused.commands
