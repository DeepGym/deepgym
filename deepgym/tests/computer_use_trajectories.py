"""Scripted reference trajectories for the built-in computer-use envs, per variant.

Each env folder's reference_actions.json holds them, one episode per variant in
variants.json order; ``deepgym qa`` replays the same files. Coordinates are
screenshot pixels on Daytona's default 1024x768 desktop.
"""

import json
from pathlib import Path

from deepgym import Action

ENVS_DIR = Path(__file__).resolve().parent.parent / 'src' / 'deepgym' / 'envs' / 'computer_use'
NAMES = ('expense_sheet', 'file_manager', 'settings_tabs', 'signup_form', 'todo_list')
DONE = Action(action_type='done')


def variants(name: str) -> list[dict]:
    return json.loads((ENVS_DIR / name / 'variants.json').read_text())


def reference(name: str, variant: int) -> list[Action]:
    episodes = json.loads((ENVS_DIR / name / 'reference_actions.json').read_text())
    return [Action.model_validate(action) for action in episodes[variant]]
