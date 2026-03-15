"""Main FastAPI application for the DeepGym API."""

from __future__ import annotations

import logging
import os
import secrets
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

try:
    from fastapi import Depends, FastAPI, HTTPException, Security
    from fastapi.security import APIKeyHeader
except ImportError as _exc:
    raise ImportError(
        'FastAPI is required for the server. Install with: pip install "deepgym[server]"'
    ) from _exc

from deepgym.api.routes import health_router, v1_router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)


def _is_no_auth() -> bool:
    """Return True when the operator has explicitly opted out of authentication.

    The opt-out is signalled by setting ``DEEPGYM_NO_AUTH=true`` (or ``1`` /
    ``yes``) in the environment.
    """
    return os.getenv('DEEPGYM_NO_AUTH', '').lower() in ('1', 'true', 'yes')


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate the X-API-Key header against DEEPGYM_API_KEY env var.

    If ``DEEPGYM_NO_AUTH=true`` is set, all requests are allowed (dev mode).
    """
    if _is_no_auth():
        return  # Operator explicitly opted out of auth.
    expected = os.getenv('DEEPGYM_API_KEY', '')
    if not expected or not secrets.compare_digest(expected, api_key or ''):
        raise HTTPException(status_code=401, detail='Invalid API key')


# ---------------------------------------------------------------------------
# Local execution safety check
# ---------------------------------------------------------------------------


def _check_auth_config() -> None:
    """Refuse to start without authentication unless explicitly opted out.

    Require DEEPGYM_API_KEY for production. If the operator wants to run
    without auth (e.g. local development), they must set
    ``DEEPGYM_NO_AUTH=true`` or pass ``--no-auth``.
    """
    has_key = bool(os.getenv('DEEPGYM_API_KEY'))
    no_auth = _is_no_auth()

    if not has_key and not no_auth:
        print(
            'ERROR: DEEPGYM_API_KEY is not set.\n'
            'Set DEEPGYM_API_KEY for production, or set DEEPGYM_NO_AUTH=true '
            'for development.\n',
            file=sys.stderr,
        )
        sys.exit(1)


def _check_local_exec_safety() -> None:
    """Refuse to start if local execution is enabled without explicit opt-in.

    When no Daytona backend is configured (local executor mode), the server
    can execute arbitrary code on the host. Require the operator to
    explicitly acknowledge this via --allow-local-exec or
    DEEPGYM_ALLOW_LOCAL_EXEC=true.
    """
    has_daytona = bool(os.getenv('DAYTONA_API_KEY'))
    if has_daytona:
        return  # Daytona sandboxes are configured; safe.

    allow_local = os.getenv('DEEPGYM_ALLOW_LOCAL_EXEC', '').lower() in (
        'true',
        '1',
        'yes',
    )
    if not allow_local:
        print(
            'ERROR: Running in local executor mode without Daytona sandboxes.\n'
            'This allows unauthenticated remote code execution on the host.\n\n'
            'To start the server anyway, pass --allow-local-exec or set\n'
            'DEEPGYM_ALLOW_LOCAL_EXEC=true in your environment.\n',
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run safety checks on startup and clean up on shutdown."""
    _check_auth_config()
    _check_local_exec_safety()

    if _is_no_auth():
        logger.warning(
            'DEEPGYM_NO_AUTH is set. The API is running without '
            'authentication. Do NOT expose this server to untrusted networks.'
        )

    # Eagerly create the client to fail fast on misconfig.
    from deepgym.api.deps import get_deepgym

    try:
        client = get_deepgym()
        if os.getenv('DAYTONA_API_KEY') and client._local_executor is not None:
            logger.error('DAYTONA_API_KEY set but client fell back to local mode')
            sys.exit(1)
    except Exception as exc:
        logger.error('Failed to initialize DeepGym client: %s', exc)
        sys.exit(1)

    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title='DeepGym API',
    description='RL training and evaluation infrastructure',
    version='0.1.0',
    lifespan=_lifespan,
)

app.include_router(health_router)
app.include_router(v1_router, dependencies=[Depends(verify_api_key)])
