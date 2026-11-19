"""The console app: a JSON API under /api plus the built UI from ./static or a given folder."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import APIRouter, Depends, FastAPI, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, PlainTextResponse
    from starlette.middleware.trustedhost import TrustedHostMiddleware
except ImportError as _exc:
    raise ImportError(
        'FastAPI is required for the console. Install with: pip install "deepgym[server]"'
    ) from _exc

from deepgym.console import environments, episodes, labels, qa, queues, runs, sandboxes
from deepgym.console.db import DEFAULT_DB_PATH
from deepgym.console.episode_store import EpisodeStore
from deepgym.console.label_store import LabelStore
from deepgym.console.queue_store import QueueStore
from deepgym.console.run_store import RunStore
from deepgym.console.runner import Runner
from deepgym.console.security import TokenedStaticFiles, require_console_access
from deepgym.qa.report import report_dir

STATIC_DIR = Path(__file__).parent / 'static'
MISSING_UI_MESSAGE = (
    'The console UI is not built, so only the API under /api is served.\n'
    'Build it from a clone of the repo (needs bun):\n'
    '  cd deepgym/console-ui && bun install && bun run build\n'
    'Wheels from PyPI ship it prebuilt.'
)


def ui_built() -> bool:
    """Return True if the UI's static export is in place."""
    return (STATIC_DIR / 'index.html').is_file()


def create_app(
    db_path: Path = DEFAULT_DB_PATH,
    qa_dir: Path | None = None,
    *,
    allow_local_exec: bool = False,
    allowed_hosts: Sequence[str] | None = None,
    token: str | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Create the console application.

    Args:
        db_path: SQLite file for runs, episodes, labels and review queues.
        qa_dir: Directory of QA report JSON files; defaults to ``deepgym qa``'s.
        allow_local_exec: Allow custom solutions to run on this machine (mode 'local',
            or 'auto' without Daytona). Reference solutions may run locally regardless.
        allowed_hosts: Host headers to accept; None accepts any. Pass the loopback
            names when bound to loopback so other sites can't reach it by DNS rebinding.
        token: Secret every /api call must send as X-DeepGym-Token; it is written into
            the served HTML. Defaults to $DEEPGYM_CONSOLE_TOKEN, else a fresh random one.
        static_dir: Folder of a UI build to serve instead of the bundled one, the same way
            (token in every HTML page, never cached). It must hold an index.html.

    Returns:
        The FastAPI app. Runs left unfinished by a previous process are marked failed.

    Raises:
        FileNotFoundError: *static_dir* doesn't exist or has no index.html of its own.
    """
    if static_dir is not None:
        _check_ui_dir(static_dir)
    run_store = RunStore(db_path)
    run_store.fail_unfinished()
    runner = Runner(run_store, allow_local_exec=allow_local_exec)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        runner.shutdown()

    app = FastAPI(title='DeepGym Console', lifespan=lifespan)
    app.state.runs = run_store
    app.state.episodes = EpisodeStore(db_path)
    app.state.labels = LabelStore(db_path)
    app.state.queues = QueueStore(db_path)
    app.state.runner = runner
    app.state.qa_dir = qa_dir or report_dir()
    app.state.token = token or os.environ.get('DEEPGYM_CONSOLE_TOKEN') or secrets.token_urlsafe(32)
    app.add_exception_handler(RequestValidationError, _validation_error)
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    api = APIRouter(prefix='/api', dependencies=[Depends(require_console_access)])
    for module in (environments, runs, episodes, labels, queues, sandboxes, qa):
        api.include_router(module.router)
    app.include_router(api)

    if static_dir is not None or ui_built():
        app.mount('/', TokenedStaticFiles(static_dir or STATIC_DIR, app.state.token), name='ui')
    else:
        app.add_api_route('/', _missing_ui, include_in_schema=False)
    return app


def _check_ui_dir(static_dir: Path) -> None:
    """Refuse a folder the server couldn't serve index.html from (it never leaves the folder)."""
    index = static_dir / 'index.html'
    if not static_dir.exists():
        problem = 'does not exist'
    elif not index.is_file():
        problem = 'has no index.html'
    elif not index.resolve().is_relative_to(static_dir.resolve()):
        problem = 'has an index.html that links outside it'
    else:
        return
    raise FileNotFoundError(f'UI folder {static_dir} {problem}')


class _AsciiJSONResponse(JSONResponse):
    """JSON escaped to ASCII, so text that isn't valid Unicode (a lone surrogate) renders."""

    def render(self, content: object) -> bytes:
        """Serialize *content* as compact, ASCII-only JSON."""
        return json.dumps(content, allow_nan=False, separators=(',', ':')).encode('ascii')


async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Answer a refused request as FastAPI does (422), with its inputs escaped to ASCII.

    FastAPI's own renderer writes UTF-8, which fails, as a 500, on a refused input
    holding a lone surrogate.
    """
    return _AsciiJSONResponse(status_code=422, content={'detail': jsonable_encoder(exc.errors())})


def _missing_ui() -> PlainTextResponse:
    return PlainTextResponse(MISSING_UI_MESSAGE, status_code=503)
