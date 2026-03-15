"""FastAPI dependency providers for the DeepGym API."""

from __future__ import annotations

import os
from functools import lru_cache

from deepgym.core import DeepGym


@lru_cache
def get_deepgym() -> DeepGym:
    """Return a singleton DeepGym client instance.

    In server mode the client is configured explicitly: if DAYTONA_API_KEY is
    set, mode is forced to ``'daytona'`` so that a Daytona init failure is
    fatal rather than silently falling back to local execution.  Otherwise
    local mode is used (already gated by ``--allow-local-exec``).

    Returns:
        A DeepGym client in the appropriate mode.
    """
    if os.getenv('DAYTONA_API_KEY'):
        return DeepGym(mode='daytona')
    return DeepGym(mode='local')
