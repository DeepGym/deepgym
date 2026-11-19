"""Guards that keep other websites from driving the console through the user's browser.

The console has no login. Every API call must carry a per-launch token that only
the served HTML holds; other origins can't read that HTML, and the custom header
forces a CORS preflight the console never answers (it has no CORS middleware).
Writes must also come from the console's own origin and carry JSON bodies.
"""

from __future__ import annotations

import hmac
import os
import re
from html import escape
from pathlib import Path

from fastapi import HTTPException, Request, Response, status
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.types import Scope

TOKEN_HEADER = 'X-DeepGym-Token'
TOKEN_META = 'deepgym-token'
_WRITES = frozenset({'POST', 'PUT', 'PATCH', 'DELETE'})
_BODY_WRITES = frozenset({'POST', 'PUT', 'PATCH'})


def require_console_access(request: Request) -> None:
    """Reject API calls without the launch token, cross-origin writes and non-JSON bodies.

    Raises:
        HTTPException: 403 for a missing or wrong token or a foreign Origin,
            415 for a write whose body isn't JSON.
    """
    sent = request.headers.get(TOKEN_HEADER, '').encode()
    if not hmac.compare_digest(sent, request.app.state.token.encode()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f'Missing or wrong {TOKEN_HEADER}; reload the console page.',
        )
    if request.method not in _WRITES:
        return
    origin = request.headers.get('origin')
    own_origin = f'{request.url.scheme}://{request.headers.get("host", "")}'
    if origin is not None and origin != own_origin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail='Cross-origin writes are not allowed.'
        )
    content_type = request.headers.get('content-type', '').split(';')[0].strip().lower()
    if request.method in _BODY_WRITES and content_type != 'application/json':
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail='Send JSON with Content-Type: application/json.',
        )


class TokenedStaticFiles(StaticFiles):
    """The UI's static export, with the launch token written into every HTML page."""

    def __init__(self, directory: Path, token: str) -> None:
        super().__init__(directory=directory, html=True)
        self._meta = f'<meta name="{TOKEN_META}" content="{escape(token)}"/>'.encode()

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve as StaticFiles does, with its 404.html fallback going through file_response."""
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse) and response.status_code == 404:
            return self.file_response(response.path, response.stat_result, scope, 404)
        return response

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: dict,
        status_code: int = 200,
    ) -> Response:
        """Serve HTML with the token meta tag before its first </head>, never cached."""
        if not str(full_path).lower().endswith(('.html', '.htm')):
            return super().file_response(full_path, stat_result, scope, status_code)
        page = re.sub(
            rb'</head>',
            lambda end: self._meta + end[0],
            Path(full_path).read_bytes(),
            count=1,
            flags=re.IGNORECASE,
        )
        return Response(
            page,
            status_code=status_code,
            media_type='text/html',
            headers={'Cache-Control': 'no-store'},
        )
