"""Apple container backend: same isolation guarantees as Daytona, run locally on macOS.

Skipped unless Apple's container CLI is installed and its service is running.
"""

import asyncio
import contextlib
import shlex
import subprocess

import pytest

from deepgym import AsyncDeepGym, ComputerUseEnvironment, DeepGym, load_environment
from deepgym.apple_container import apple_container_available
from deepgym.exceptions import DeepGymError, SandboxError
from deepgym.models import Action, MultiTurnEnvironment
from tests.test_isolation import _env

# Every live isolation test, rerun here against Apple containers via this module's `dg`.
from tests.test_isolation_live import (  # noqa: F401
    test_hacks_fail,
    test_judge_temp_files_are_private,
    test_reference_passes_and_wrong_fails,
    test_script_env_and_support_files,
    test_setsid_daemon_dies_with_its_script,
    test_slow_case_fails_alone,
    test_solution_user_cannot_touch_judge,
)

pytestmark = pytest.mark.skipif(
    not apple_container_available(), reason='needs Apple container CLI with service running'
)


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    yield DeepGym(mode='apple')
    left = subprocess.run(
        ['container', 'ls', '--all', '--quiet'], capture_output=True, text=True, check=False
    )
    assert not [line for line in left.stdout.split() if line.startswith('deepgym-')]


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


def test_desktop_env_is_refused(dg: DeepGym) -> None:
    with dg.session(ComputerUseEnvironment(task='t')) as session:
        with pytest.raises(DeepGymError, match='Daytona'):
            session.start()


def test_async_client(dg: DeepGym) -> None:
    env = load_environment('coin_change')
    reference = env.verifier_path.with_name('reference_solution.py').read_text()
    result = asyncio.run(AsyncDeepGym(mode='apple').run(env, reference))
    assert result.score == 1.0
