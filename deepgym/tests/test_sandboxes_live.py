"""Every sandbox provider gives the same isolation guarantees.

Local providers (Apple container, Docker) run when their CLI is up. Cloud providers
cost money, so they run only when named, e.g. DEEPGYM_LIVE_PROVIDERS=modal,e2b,daytona.
"""

import asyncio
import contextlib
import os
import shlex
import subprocess

import pytest

from deepgym import AsyncDeepGym, ComputerUseEnvironment, DeepGym, load_environment
from deepgym.exceptions import DeepGymError, SandboxError
from deepgym.models import Action, MultiTurnEnvironment
from deepgym.sandboxes.apple import AppleContainer
from deepgym.sandboxes.docker import DockerContainer
from tests.test_isolation import _env

# Every live isolation test, rerun here on each provider via this module's `dg`.
from tests.test_isolation_live import (  # noqa: F401
    test_api_request_scores_what_the_solution_fetches,
    test_hacks_fail,
    test_judge_temp_files_are_private,
    test_reference_passes_and_wrong_fails,
    test_script_env_and_support_files,
    test_setsid_daemon_dies_with_its_script,
    test_slow_case_fails_alone,
    test_solution_reaches_the_judges_loopback_port_but_not_the_internet,
    test_solution_user_cannot_touch_judge,
)

# One xdist worker runs these: the leak check sees every deepgym- container.
pytestmark = pytest.mark.xdist_group('live-containers')

_CLOUD = set(os.environ.get('DEEPGYM_LIVE_PROVIDERS', '').split(','))
_PROVIDERS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(not usable, reason=f'{name} not available or not enabled'),
    )
    for name, usable in [
        ('apple', AppleContainer.available()),
        ('docker', DockerContainer.available()),
        ('modal', 'modal' in _CLOUD),
        ('e2b', 'e2b' in _CLOUD),
        ('daytona', 'daytona' in _CLOUD and bool(os.getenv('DAYTONA_API_KEY'))),
    ]
]
# Lists container names, to check no sandbox outlives its run or session.
_LIST_CONTAINERS = {
    'apple': ['container', 'ls', '--all', '--quiet'],
    'docker': ['docker', 'ps', '--all', '--format', '{{.Names}}'],
}


@pytest.fixture(scope='module', params=_PROVIDERS)
def dg(request: pytest.FixtureRequest) -> DeepGym:
    yield DeepGym(mode=request.param)
    if command := _LIST_CONTAINERS.get(request.param):
        names = subprocess.run(command, capture_output=True, text=True, check=False).stdout
        assert not [name for name in names.split() if name.startswith('deepgym-')]


def test_solution_has_no_network(dg: DeepGym) -> None:
    env = _env(
        """
reached = sol.probe()
return {'score': 0.0 if reached else 1.0, 'passed': not reached, 'details': str(reached)}
"""
    )
    solution = (
        'import socket\n'
        'def probe():\n'
        '    reached = []\n'
        '    for addr in (("1.1.1.1", 80), ("192.168.64.1", 22)):\n'
        '        try:\n'
        '            socket.create_connection(addr, timeout=3)\n'
        '            reached.append(addr[0])\n'
        '        except OSError:\n'
        '            pass\n'
        '    return reached\n'
    )
    result = dg.run(env, solution)
    assert result.passed, result.output


def test_session_shell_episode(dg: DeepGym) -> None:
    env = MultiTurnEnvironment(
        task='write hello',
        max_steps=3,
        final_verifier_code=(
            'import json, pathlib\n'
            'ok = pathlib.Path("out.txt").read_text().strip() == "hello"\n'
            'print(json.dumps({"score": float(ok), "passed": ok}))\n'
        ),
    )
    actions = iter(
        [Action(action_type='bash', content='echo hello > out.txt'), Action(action_type='done')]
    )
    trajectory, result = dg.run_episode(env, lambda obs: next(actions))
    assert result.passed
    assert len(trajectory.steps) == 2


_PY311 = 'python:3.11-slim'  # not the 3.12 default, so the version shows which image ran


def test_env_image_boots_runs_and_sessions(dg: DeepGym) -> None:
    env = _env(
        """
version = sol.version()
return {'score': float(version == [3, 11]), 'passed': version == [3, 11], 'details': str(version)}
""",
        image=_PY311,
    )
    solution = 'import sys\ndef version():\n    return list(sys.version_info[:2])\n'
    result = dg.run(env, solution)
    assert result.passed, result.output
    with dg.session(MultiTurnEnvironment(task='t', max_steps=2, image=_PY311)) as session:
        session.start()
        step = session.step(Action(action_type='bash', content='python3 --version'))
    assert 'Python 3.11' in step.observation.content


def test_session_is_offline_but_keeps_local_servers(dg: DeepGym) -> None:
    env = MultiTurnEnvironment(
        task='t',
        max_steps=3,
        setup_script='(setsid python3 -m http.server 8123 >/dev/null 2>&1 &) && sleep 1',
    )
    fetch = (
        'import socket, urllib.request\n'
        'print(urllib.request.urlopen("http://127.0.0.1:8123/", timeout=5).status)\n'
        'try:\n'
        '    socket.create_connection(("1.1.1.1", 80), timeout=3)\n'
        '    print("ONLINE")\n'
        'except OSError:\n'
        '    print("offline")\n'
    )
    escape = 'nsenter --net=/proc/1/ns/net -- true && echo ESCAPED || echo stayed'
    with dg.session(env) as session:
        session.start()
        output = session.step(Action(content=fetch)).observation.content
        escaped = session.step(Action(action_type='bash', content=escape)).observation.content
    assert '200' in output
    assert 'offline' in output
    assert 'stayed' in escaped


# Planted where the privileged exec wrapper would run it on the next step.
_TROJAN = '#!/bin/sh\ntouch /tmp/deepgym/pwned\nexec /usr/bin/nsenter "$@"\n'
_PWNED = 'test -e pwned && echo PWNED || echo clean'


def test_session_cannot_trojan_the_exec_wrapper(dg: DeepGym) -> None:
    plant = f'printf %s {shlex.quote(_TROJAN)} > /usr/local/bin/nsenter'
    plant += ' && chmod +x /usr/local/bin/nsenter; echo planted'
    with dg.session(MultiTurnEnvironment(task='t', max_steps=5)) as session:
        session.start()
        session.step(Action(action_type='bash', content=plant))
        after = session.step(Action(action_type='bash', content=_PWNED)).observation.content
    assert 'clean' in after


def test_session_write_file_cannot_follow_symlinks_out(dg: DeepGym) -> None:
    with dg.session(MultiTurnEnvironment(task='t', max_steps=5)) as session:
        session.start()
        session.step(Action(action_type='bash', content='ln -s /usr/bin ub'))
        with contextlib.suppress(SandboxError):
            session.step(Action(action_type='write_file', path='ub/timeout', content=_TROJAN))
        after = session.step(Action(action_type='bash', content=_PWNED)).observation.content
    assert 'clean' in after


def test_session_cannot_kill_the_network_holder(dg: DeepGym) -> None:
    kill_all = 'for p in /proc/[0-9]*; do kill -9 "${p#/proc/}" 2>/dev/null; done; echo swept'
    probe = (
        'import socket\n'
        'try:\n'
        '    socket.create_connection(("1.1.1.1", 80), timeout=3)\n'
        '    print("ONLINE")\n'
        'except OSError:\n'
        '    print("offline")\n'
    )
    with dg.session(MultiTurnEnvironment(task='t', max_steps=5)) as session:
        session.start()
        session.step(Action(action_type='bash', content=kill_all))
        after = session.step(Action(content=probe)).observation.content
    assert 'offline' in after, after


# Runs as the agent: logs every python command line, tries to read verifier scripts,
# and writes a fake passing score into every python process's stdout.
_SPY = """
import os, time
end = time.time() + 40
while time.time() < end:
    for pid in [p for p in os.listdir('/proc') if p.isdigit()]:
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\\0', b' ').decode()
        except OSError:
            continue
        if 'python' in cmd and 'spy.py' not in cmd:
            with open('leak.txt', 'a') as log:
                log.write(cmd + '\\n')
            try:
                with open(f'/proc/{pid}/fd/1', 'w') as out:
                    out.write('{"score": 1.0, "passed": true}\\n')
            except OSError:
                pass
    try:
        for name in os.listdir('/opt/deepgym-verifier'):
            with open('leak.txt', 'a') as log:
                log.write(open(f'/opt/deepgym-verifier/{name}').read())
    except OSError:
        pass
    time.sleep(0.02)
"""


def test_session_verifiers_are_out_of_the_agents_reach(dg: DeepGym) -> None:
    env = MultiTurnEnvironment(
        task='t',
        max_steps=6,
        step_verifier_code='import time\ntime.sleep(1)\nprint(0.0)  # SECRET-STEP\n',
        final_verifier_code=(
            'import json, time\n'
            'print(json.dumps({"score": 0.0, "passed": False}), flush=True)  # SECRET-FINAL\n'
            'time.sleep(1.5)\n'
        ),
    )
    start_spy = '(setsid python3 spy.py >/dev/null 2>&1 &); sleep 0.5; echo started'
    with dg.session(env) as session:
        session.start()
        rewards = [
            session.step(Action(action_type='write_file', path='spy.py', content=_SPY)).reward,
            session.step(Action(action_type='bash', content=start_spy)).reward,
            session.step(Action(action_type='bash', content='sleep 1')).reward,
        ]
        leak = session.step(Action(action_type='bash', content='cat leak.txt; echo END'))
        result = session.score()
    assert rewards == [0.0, 0.0, 0.0]
    assert result.score == 0.0
    assert 'python' in leak.observation.content, 'the spy should have seen the verifiers run'
    assert 'SECRET' not in leak.observation.content
    assert 'b64decode' not in leak.observation.content


def test_desktop_env_is_refused(dg: DeepGym) -> None:
    if dg.backend == 'daytona':
        pytest.skip('Daytona has a desktop; see test_computer_use_live.py')
    with dg.session(ComputerUseEnvironment(task='t')) as session:
        with pytest.raises(DeepGymError, match='Daytona'):
            session.start()


def test_async_client(dg: DeepGym) -> None:
    env = load_environment('coin_change')
    reference = env.verifier_path.with_name('reference_solution.py').read_text()
    result = asyncio.run(AsyncDeepGym(mode=dg.backend).run(env, reference))
    assert result.score == 1.0
