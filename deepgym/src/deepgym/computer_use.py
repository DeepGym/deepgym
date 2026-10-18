"""Computer-use environments: agents drive a real desktop in a Daytona sandbox.

The agent sees screenshots and acts with mouse and keyboard. The final verifier
runs inside the same sandbox and checks the resulting state (files, app state,
browser URL via Chromium's debug port), never pixels.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from deepgym.models import GUI_ACTIONS, ActionType, MultiTurnEnvironment


class ComputerUseEnvironment(MultiTurnEnvironment):
    """A multi-turn environment with a desktop and GUI-only actions by default."""

    type: Literal['computer-use'] = 'computer-use'  # type: ignore[assignment]
    """Environment type discriminator."""

    desktop: bool = True
    """Computer-use environments always get a desktop."""

    allowed_actions: frozenset[ActionType] = frozenset({*GUI_ACTIONS, 'done'})
    """GUI only, so the agent can't skip the interface with a shell."""

    max_steps: int | None = Field(default=30, ge=1)  # None: no step limit
    """GUI tasks take more steps than shell tasks."""
