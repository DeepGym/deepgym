"""Scripted agents for multi-turn QA: they act, but never work on the task.

A verifier must give them nothing. One that pays for being busy, or for random
clicks and commands, rewards a policy for activity instead of the outcome.
Their shell actions are harmless (``ls``, ``print``, notes files), because a local
session runs them on the host; their tool calls only read (GET on a declared route).
"""

from __future__ import annotations

import random

from deepgym.models import Action, ActionType, MultiTurnEnvironment
from deepgym.tools import PLACEHOLDER

_SCREEN = (1024, 768)
"""Daytona's default desktop, in screenshot pixels."""

_SEED = 0
"""Fixed, so the random agent replays the same episode every time."""

_COMMANDS = ('ls', 'pwd', 'echo working', 'true')
_CODE = ('print("working")', 'total = sum(range(10))')
_TEXT = ('hello', 'test', '42')
_KEYS = ('Return', 'Tab', 'Escape', 'ctrl+a')
_DIRECTIONS = ('up', 'down', 'left', 'right')


def do_everything(env: MultiTurnEnvironment) -> list[Action]:
    """Every allowed action type in turn, with fixed arguments, for the whole step budget."""
    kinds = _kinds(env)
    return [
        _action(env, kinds[step % len(kinds)], step, None)
        for step in range(_control_steps(env, kinds))
    ]


def random_agent(env: MultiTurnEnvironment) -> list[Action]:
    """Seeded random actions of the allowed types, for the whole step budget."""
    rng = random.Random(_SEED)
    kinds = _kinds(env)
    return [
        _action(env, rng.choice(kinds), step, rng) for step in range(_control_steps(env, kinds))
    ]


def _control_steps(env: MultiTurnEnvironment, kinds: list[ActionType]) -> int:
    """Length of a scripted QA control: the env's step budget, or one round of every allowed
    action type when the env sets no step limit (fixed controls must end; never a model run)."""
    return env.max_steps if env.max_steps is not None else len(kinds)


def _kinds(env: MultiTurnEnvironment) -> list[ActionType]:
    """The action types the env allows, except 'done', in a fixed order."""
    return sorted(env.allowed_actions - {'done'}) or ['done']


def _action(
    env: MultiTurnEnvironment, kind: ActionType, step: int, rng: random.Random | None
) -> Action:
    """One harmless action of *kind*: fixed arguments without *rng*, random ones with it."""

    def pick(options: tuple[str, ...]) -> str:
        return rng.choice(options) if rng else options[step % len(options)]

    def read() -> Action:
        """A GET on a declared route, each placeholder filled with 1; done if none reads."""
        reads = [(t.name, r.path) for t in env.tools for r in t.routes if r.method == 'GET']
        if not reads:
            return Action(action_type='done')
        name, path = rng.choice(reads) if rng else reads[step % len(reads)]
        arguments = {'method': 'GET', 'path': PLACEHOLDER.sub('1', path)}
        return Action(action_type=kind, tool=name, arguments=arguments)

    def point() -> tuple[int, int]:
        if rng is None:
            return 10 + step, 10 + step
        return rng.randrange(_SCREEN[0]), rng.randrange(_SCREEN[1])

    x, y = point()
    match kind:
        case 'bash':
            return Action(action_type=kind, content=pick(_COMMANDS))
        case 'code':
            return Action(action_type=kind, content=pick(_CODE))
        case 'write_file':
            return Action(action_type=kind, path=f'notes_{step}.txt', content='notes\n')
        case 'type':
            return Action(action_type=kind, content=pick(_TEXT))
        case 'key':
            return Action(action_type=kind, content=pick(_KEYS))
        case 'scroll':
            return Action(action_type=kind, content=pick(_DIRECTIONS), x=x, y=y)
        case 'drag':
            to_x, to_y = point()
            return Action(action_type=kind, x=x, y=y, to_x=to_x, to_y=to_y)
        case 'click' | 'double_click' | 'right_click' | 'move':
            return Action(action_type=kind, x=x, y=y)
        case 'tool':
            return read()
        case _:  # screenshot, wait, done
            return Action(action_type=kind)
