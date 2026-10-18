"""Translate provider computer actions without silently dropping semantics."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from deepgym.models import Action

_KEY_ALIASES = {
    'return': 'enter',
    'kp_enter': 'enter',
    'esc': 'escape',
    'back_space': 'backspace',
    'del': 'delete',
    'page_down': 'pagedown',
    'page_up': 'pageup',
    'pgdn': 'pagedown',
    'pgup': 'pageup',
    'arrowdown': 'down',
    'arrowup': 'up',
    'arrowleft': 'left',
    'arrowright': 'right',
    'control': 'ctrl',
    'super': 'cmd',
    'meta': 'cmd',
    'win': 'cmd',
    'option': 'alt',
    'spacebar': 'space',
    ' ': 'space',
}


def _key(value: str) -> str:
    return _KEY_ALIASES.get(value.lower(), value.lower() if len(value) > 1 else value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be an integer')
    if not math.isfinite(value) or int(value) != value:
        raise ValueError(f'{label} must be a finite integer')
    return int(value)


def _point(value: object) -> dict[str, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('a coordinate with exactly two integers is required')
    return {'x': _integer(value[0], 'x'), 'y': _integer(value[1], 'y')}


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError('nonempty text is required')
    return value


def anthropic_actions(name: str, data: Mapping[str, Any]) -> list[Action]:
    """Map an Anthropic tool call; reject actions the Session cannot represent.

    Args:
        name: Computer tool name, or ``computer`` for legacy input.action calls.
        data: Provider tool input.

    Returns:
        Actions preserving the supported operation.

    Raises:
        ValueError: For unsupported operations, modifiers, or malformed arguments.
    """
    if name == 'computer':
        name = str(data.get('action', ''))
    pointer = {
        'left_click',
        'right_click',
        'double_click',
        'mouse_move',
        'scroll',
        'left_click_drag',
    }
    if name in pointer and data.get('text'):
        raise ValueError('mouse modifier keys are unsupported by Session')
    if name == 'screenshot':
        return [Action(action_type='screenshot')]
    if name in pointer:
        point = _point(data.get('coordinate'))
        if name == 'scroll':
            return [
                Action(
                    action_type='scroll',
                    content=_text(data.get('scroll_direction')),
                    amount=_integer(data.get('scroll_amount', 3), 'scroll_amount'),
                    **point,
                )
            ]
        if name == 'left_click_drag':
            start = _point(data.get('start_coordinate'))
            return [Action(action_type='drag', **start, to_x=point['x'], to_y=point['y'])]
        kind = {
            'left_click': 'click',
            'right_click': 'right_click',
            'double_click': 'double_click',
            'mouse_move': 'move',
        }[name]
        return [Action(action_type=kind, **point)]
    if name == 'type':
        return [Action(action_type='type', content=_text(data.get('text')))]
    if name == 'key':
        keys = _text(data.get('text')).split('+')
        if any(not key.strip() for key in keys):
            raise ValueError('keypress contains an empty key')
        repeat = _integer(data.get('repeat', 1), 'repeat')
        if not 1 <= repeat <= 50:
            raise ValueError('key repeat must be between 1 and 50')
        return [
            Action(action_type='key', content='+'.join(_key(key.strip()) for key in keys))
            for _ in range(repeat)
        ]
    if name == 'wait':
        seconds = data.get('duration', 1)
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ValueError('wait duration must be numeric')
        if not math.isfinite(seconds) or not 0 < seconds <= 60:
            raise ValueError('wait duration must be between 0 and 60 seconds')
        return [Action(action_type='wait', amount=math.ceil(seconds))]
    raise ValueError(f'Anthropic computer action {name!r} is unsupported by Session')


def openai_actions(data: Mapping[str, Any]) -> list[Action]:
    """Map a Responses computer action; reject unrepresentable buttons and paths.

    Args:
        data: Provider action serialized with model_dump.

    Returns:
        Actions for Session.step. Pixel scroll distances become wheel clicks at 100px/click.

    Raises:
        ValueError: For unsupported actions, modifiers, malformed inputs, or curved drags.
    """
    name = data.get('type')
    if data.get('keys') and name != 'keypress':
        raise ValueError('mouse modifier keys are unsupported by Session')
    if name in {'screenshot', 'wait'}:
        return [Action(action_type=name, amount=2 if name == 'wait' else 1)]
    if name == 'type':
        return [Action(action_type='type', content=_text(data.get('text')))]
    if name == 'keypress':
        keys = data.get('keys')
        if not isinstance(keys, (list, tuple)) or not keys:
            raise ValueError('keypress needs a nonempty keys list')
        return [Action(action_type='key', content='+'.join(_key(_text(key)) for key in keys))]
    if name == 'drag':
        path = data.get('path')
        if not isinstance(path, Sequence) or len(path) != 2:
            raise ValueError('Session drag requires exactly two points; curved paths unsupported')
        if not all(isinstance(point, Mapping) for point in path):
            raise ValueError('drag points must contain x and y')
        start = _point([path[0].get('x'), path[0].get('y')])
        end = _point([path[1].get('x'), path[1].get('y')])
        return [Action(action_type='drag', **start, to_x=end['x'], to_y=end['y'])]
    if name in {'click', 'double_click', 'move', 'scroll'}:
        point = _point([data.get('x'), data.get('y')])
        if name == 'click':
            button = data.get('button', 'left')
            if button not in {'left', 'right'}:
                raise ValueError(f'Session cannot represent mouse button {button!r}')
            return [Action(action_type='click' if button == 'left' else 'right_click', **point)]
        if name == 'scroll':
            horizontal = _integer(data.get('scroll_x', 0), 'scroll_x')
            vertical = _integer(data.get('scroll_y', 0), 'scroll_y')
            if horizontal and vertical:
                raise ValueError('Session cannot represent simultaneous diagonal scrolling')
            if not horizontal and not vertical:
                raise ValueError('scroll requires a nonzero distance')
            distance = vertical or horizontal
            direction = (
                ('down' if distance > 0 else 'up')
                if vertical
                else ('right' if distance > 0 else 'left')
            )
            return [
                Action(
                    action_type='scroll',
                    content=direction,
                    amount=max(1, round(abs(distance) / 100)),
                    **point,
                )
            ]
        return [Action(action_type=name, **point)]
    raise ValueError(f'OpenAI computer action {name!r} is unsupported by Session')
