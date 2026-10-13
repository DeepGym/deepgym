"""Live computer-use episodes on Daytona: reference runs pass, doing nothing scores 0."""

import os

import pytest

from deepgym import Action, DeepGym, load_environment
from tests.computer_use_trajectories import DONE, REFERENCE, variants
from tests.test_sandboxes_live import _SPY

pytestmark = pytest.mark.skipif(
    not os.getenv('DAYTONA_API_KEY'), reason='needs DAYTONA_API_KEY for live sandboxes'
)

# First and last variant of each env; the full sweep is too slow for a test run.
CASES = [(name, v) for name in sorted(REFERENCE) for v in (0, len(variants(name)) - 1)]


@pytest.fixture(scope='module')
def dg() -> DeepGym:
    return DeepGym(mode='daytona')


@pytest.mark.parametrize(('name', 'variant'), CASES)
def test_reference_trajectory_passes(dg: DeepGym, name: str, variant: int) -> None:
    actions = iter(REFERENCE[name](variants(name)[variant]))
    env = load_environment(name, variant=variant)
    trajectory, result = dg.run_episode(env, lambda obs: next(actions))
    assert result.passed, result.output
    assert trajectory.steps[0][0].screenshot


def test_doing_nothing_scores_zero(dg: DeepGym) -> None:
    _, result = dg.run_episode(load_environment('settings_tabs'), lambda obs: DONE)
    assert result.score == 0.0


# Each prints a marker only if that way out of the GUI or up to root still works.
_ESCAPES = {
    'sudo': 'sudo -n true 2>/dev/null && echo ESCAPED',
    'toolbox': 'curl -s -m 3 -o /dev/null http://127.0.0.1:2280/version && echo ESCAPED',
    'custom_command': (
        "printf '[Desktop Entry]\\nType=Application\\nName=x\\nExec=touch /tmp/deepgym/ran\\n'"
        ' > x.desktop; gio launch ./x.desktop 2>/dev/null; sleep 1; test -e ran && echo ESCAPED'
    ),
    'custom_actions': 'test -r /usr/lib/x86_64-linux-gnu/thunarx-3/thunar-uca.so && echo ESCAPED',
    'vnc_remote': 'timeout 10 x11vnc -display :0 -Q ping 2>&1 | grep -q ans=ping && echo ESCAPED',
    # The desktop runs as the agent, not as the sudoer that runs Daytona's daemon (pid 1).
    'desktop_user': (
        'u=$(ps -o user= -C xfce4-session | head -1); '
        '[ "$u" = "$(id -un)" ] && [ "$u" != "$(ps -o user= -p 1)" ] || echo ESCAPED'
    ),
}


def test_a_shell_out_of_the_desktop_is_the_unprivileged_agent(dg: DeepGym) -> None:
    """Bash actions stand in for any way out of the GUI: they run where a GUI escape would."""
    env = load_environment('file_manager')
    env = env.model_copy(update={'allowed_actions': env.allowed_actions | {'bash', 'write_file'}})
    start_spy = '(setsid python3 spy.py >/dev/null 2>&1 &); sleep 0.5'
    with dg.session(env) as session:
        session.start()
        escaped = {
            name: session.step(Action(action_type='bash', content=probe)).observation.content
            for name, probe in _ESCAPES.items()
        }
        session.step(Action(action_type='write_file', path='spy.py', content=_SPY))
        session.step(Action(action_type='bash', content=start_spy))
        result = session.score()
    assert not [name for name, out in escaped.items() if 'ESCAPED' in out], escaped
    assert result.score == 0.0, 'the spy must not forge the untouched desktop to a pass'


# Root-side stand-in for any loopback service outside the agent's network namespace.
_LISTENER = (
    '(setsid python3 -c "import socket; s = socket.socket(); '
    "s.bind(('127.0.0.1', 5999)); s.listen(); s.accept(); "
    "open('/root/vnc_hit', 'w').write('hit')\" >/dev/null 2>&1 &); sleep 1"
)


def test_x11vnc_ignores_connect_requests_from_x_clients(dg: DeepGym) -> None:
    """x11vnc runs as the sudoer outside the agent's network namespace, and any X client
    can set root-window properties, so it must not act on VNC_CONNECT."""
    env = load_environment('file_manager')
    env = env.model_copy(update={'allowed_actions': env.allowed_actions | {'bash'}})
    request = 'xprop -root -f VNC_CONNECT 8s -set VNC_CONNECT 127.0.0.1:5999; sleep 5'
    with dg.session(env) as session:
        session.start()
        root = session._backend._sandbox  # root exec, outside the agent's namespace
        root.exec(_LISTENER, timeout=30)
        session.step(Action(action_type='bash', content=request))
        hit = root.exec('test -e /root/vnc_hit && echo CONNECTED || echo none', timeout=30)
    assert 'none' in hit.output
