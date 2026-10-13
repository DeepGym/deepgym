"""Harbor runs against fake sandboxes: what reaches the agent, what reaches the grader.

The fake records every command and upload; real containers are in test_harbor_live.py.
"""

import base64
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
from deepgym.sandboxes import SandboxTimeoutError, build_image
from deepgym.sandboxes.docker import DockerContainer
from deepgym.session import SandboxBackend
from tests.test_harbor import BUILT, HELLO
from tests.test_sandboxes import FakeSandbox

_USERS = ('useradd', '997 997 996 996')
_WORKSPACE = pack_files({'hello.txt': b'hello\n'})
_ENCODED = ('base64 /opt', base64.b64encode(_WORKSPACE).decode())
_PASS = ('for name in reward.json', 'reward.txt\n1\n')
_AS_AGENT = 'setpriv --reuid=997 --regid=997'


def _box(*answers: tuple[str, str]) -> FakeSandbox:
    return FakeSandbox(network_blocked=True, answers=[_USERS, *answers])


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
    assert (
        box.commands[packed].startswith(f'{_AS_AGENT}')
        and "sh -c 'kill -9 -1'" in box.commands[packed]
    )
    assert box.uploads == ['/opt/deepgym-harbor/agent.sh']
    assert not [command for command in box.commands if '/tests' in command or '/logs' in command]
    assert box.deleted


def test_agent_workdir_defaults_to_the_image_workdir_and_refuses_root() -> None:
    task = HarborTask.from_dir(BUILT)
    assert run_agent(_box(('pwd', '/srv/app'), _ENCODED), task, '').workdir == '/srv/app'

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
    staged = _index(box, 'rm -rf /tests /logs/verifier')
    tests = _index(box, '/tests/test.sh')
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
    online = FakeSandbox(network_blocked=False, answers=[_USERS, _PASS])
    grade(online, HarborTask.from_dir(HELLO), _WORKSPACE, '/app')
    assert online.commands[_index(online, '/tests/test.sh')].startswith('nsenter --net=')

    public = HarborTask.from_dir(HELLO).model_copy(update={'verifier_network': 'public'})
    box = FakeSandbox(network_blocked=False, answers=[_USERS, _PASS])
    grade(box, public, _WORKSPACE, '/app')
    assert box.commands[_index(box, '/tests/test.sh')].startswith('bash -c')


def test_shell_syntax_in_the_workdir_is_always_quoted(tmp_path: Path) -> None:
    workdir = '/app/$(touch pwned) `id`'
    task = HarborTask.from_dir(HELLO).model_copy(update={'workdir': workdir})
    agent, grader = _box(_ENCODED), _box(_PASS)

    run = run_agent(agent, task, 'true')
    grade(grader, task, run.workspace, run.workdir)

    quoted = shlex.quote(workdir)
    mentions = [c for c in agent.commands + grader.commands if 'touch pwned' in c]
    assert mentions and all(quoted in command for command in mentions)
    assert all(command.count('touch pwned') == command.count(quoted) for command in mentions)


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
        box = _box(_ENCODED, _PASS, ('pwd', '/app'))
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
    assert DockerContainer.build(context) != tag


def test_only_container_clis_build_images(tmp_path: Path) -> None:
    for provider in ('modal', 'e2b', 'daytona'):
        with pytest.raises(DeepGymError, match='registry images only'):
            build_image(provider, tmp_path)


def test_backend_takes_over_a_given_workdir_and_refuses_its_own_paths() -> None:
    box = _box()
    SandboxBackend(box, '/srv/my app').exec('ls', timeout=5)
    assert "chown -R dgagent: '/srv/my app'" in box.commands[0]

    for workdir in ('/', '/opt', '/usr/local/bin', '/root'):
        refused = _box()
        with pytest.raises(SandboxError, match='Work directory'):
            SandboxBackend(refused, workdir)
        assert refused.deleted and not refused.commands
