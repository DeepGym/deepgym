"""Web debugging UI for DeepGym environments.

Run with:
    deepgym web --port 8080

Opens a browser-based interface for testing environments interactively.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from importlib.resources import files

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
except ImportError as _exc:
    raise ImportError(
        'FastAPI is required for the web UI. Install with: pip install "deepgym[server]"'
    ) from _exc
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    """Request body for the /api/run endpoint."""

    environment: str
    code: str


class RunResponse(BaseModel):
    """Response body for the /api/run endpoint."""

    score: float
    passed: bool
    output: str
    execution_time_ms: float
    reward_components: dict[str, float] | None = None
    error: str | None = None


def create_web_app() -> FastAPI:
    """Create the web debugging application.

    Returns:
        Configured FastAPI app with the debugging UI and API endpoints.

    Raises:
        RuntimeError: If running without Daytona and without the local-exec
            safety flag, to prevent accidental host code execution.
    """
    allow_local = os.getenv('DEEPGYM_ALLOW_LOCAL_EXEC', '').lower() in ('1', 'true', 'yes')
    has_daytona = bool(os.getenv('DAYTONA_API_KEY'))

    if not has_daytona and not allow_local:
        raise RuntimeError(
            'Web UI requires --allow-local-exec flag or DEEPGYM_ALLOW_LOCAL_EXEC=true '
            'when running without Daytona. This prevents accidental host code execution.'
        )

    from deepgym.core import DeepGym

    app = FastAPI(title='DeepGym Web UI', docs_url='/docs')
    dg = DeepGym(mode='auto')

    # In-memory run history (bounded to last 200 entries).
    _history: list[dict] = []
    _max_history = 200

    @app.get('/', response_class=HTMLResponse)
    async def index() -> str:
        """Serve the main debugging UI page."""
        return _load_web_ui_html()

    @app.get('/api/environments')
    async def list_envs() -> list[dict]:
        """List all available environments from the registry."""
        from deepgym.registry import list_environments

        return list_environments()

    @app.get('/api/environments/{name}')
    async def get_env(name: str) -> dict:
        """Load an environment and return its task description and metadata.

        Args:
            name: Environment path/name from the registry.
        """
        from fastapi import HTTPException

        from deepgym.registry import load_environment

        try:
            env = load_environment(name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            'task': env.task,
            'difficulty': env.difficulty,
            'domain': env.domain,
            'tags': env.tags,
        }

    @app.post('/api/run', response_model=RunResponse)
    async def run_code(request: RunRequest) -> RunResponse:
        """Run submitted code against an environment verifier.

        Args:
            request: Contains environment name and solution code.
        """
        from deepgym.registry import load_environment

        start = time.perf_counter()
        try:
            env = load_environment(request.environment)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: dg.run(env, model_output=request.code)
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            entry = {
                'environment': request.environment,
                'score': result.score,
                'passed': result.passed,
                'execution_time_ms': round(elapsed_ms, 1),
                'timestamp': time.strftime('%H:%M:%S'),
            }
            _history.append(entry)
            if len(_history) > _max_history:
                del _history[: len(_history) - _max_history]

            return RunResponse(
                score=result.score,
                passed=result.passed,
                output=result.output or '',
                execution_time_ms=round(elapsed_ms, 1),
                reward_components=result.reward_components,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error('Run failed: %s', exc, exc_info=True)

            entry = {
                'environment': request.environment,
                'score': 0.0,
                'passed': False,
                'execution_time_ms': round(elapsed_ms, 1),
                'timestamp': time.strftime('%H:%M:%S'),
                'error': str(exc),
            }
            _history.append(entry)
            if len(_history) > _max_history:
                del _history[: len(_history) - _max_history]

            return RunResponse(
                score=0.0,
                passed=False,
                output='',
                execution_time_ms=round(elapsed_ms, 1),
                error=str(exc),
            )

    @app.get('/api/history')
    async def get_history() -> list[dict]:
        """Return the run history (most recent first)."""
        return list(reversed(_history))

    return app


def _load_web_ui_html() -> str:
    """Load the web UI HTML from the static resource file.

    Returns:
        The HTML content for the debugging single-page app.
    """
    return (files('deepgym') / 'static' / 'web_ui.html').read_text(encoding='utf-8')
