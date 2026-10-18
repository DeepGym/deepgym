"""HTTP API tools an environment declares for tool-calling agents, and how a session runs them.

A tool is trusted environment configuration, never model output: its name, base URL, routes
and headers come from the environment's author. A model supplies only a call's arguments
(method, path, query, body). :meth:`HttpTool.prepare` checks them against the declared
routes, and the session writes the request to a file in the work directory. The command that
sends it is a constant (:func:`run_command`), so nothing the model writes reaches a shell; it
runs as the agent user in the sandbox's offline network namespace, where only the episode's
own services answer, and it follows no redirect (see :mod:`deepgym._http_request`).
"""

from __future__ import annotations

import ipaddress
import json
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlencode, urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

HttpMethod = Literal['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
QueryScalar = str | int | float | bool

DONE_TOOL = 'done'
"""Reserved tool name: agents offer it so the model ends the episode with a ``done`` action."""

REQUEST_FILE = '_tool_request.json'
"""Where in the work directory the session writes a call's request for the runner."""

_RUNNER = Path(__file__).with_name('_http_request.py').read_text(encoding='utf-8')
_BODY_METHODS = frozenset({'POST', 'PUT', 'PATCH'})
# RFC 3986 path characters; a model percent-encodes anything else, so nothing it sends can
# start a query, a fragment or another authority.
_SEGMENT = re.compile(r"[A-Za-z0-9\-._~!$&'()*+,;=:@]+")
_PATH = re.compile(r"/[A-Za-z0-9\-._~!$&'()*+,;=:@%/]*")
_PLACEHOLDER = re.compile(r'\{\w+\}')
_REDACTED = '[redacted]'


class HttpRoute(BaseModel):
    """One endpoint a tool may call: a method and a path template.

    ``{name}`` in the template stands for exactly one non-empty path segment; every other
    segment must match literally. An endpoint no route names (an admin page, a script
    runner) is unreachable.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')
    method: HttpMethod
    path: str

    @field_validator('path')
    @classmethod
    def check_path(cls, value: str) -> str:
        """Accept '/' or '/'-separated literal segments and ``{name}`` placeholders."""
        segments = value.split('/')
        literal = [segment for segment in segments[1:] if not _PLACEHOLDER.fullmatch(segment)]
        if value != '/' and (
            segments[0]
            or not all(_SEGMENT.fullmatch(part) and part not in ('.', '..') for part in literal)
        ):
            raise ValueError(
                f'route path {value!r} must start with "/" and hold only non-empty literal '
                'segments (no dot segments or percent signs) and {name} placeholders'
            )
        return value

    def matches(self, path: str) -> bool:
        """Whether request *path* fits this template, segment by segment."""
        want, got = self.path.split('/'), path.split('/')
        return len(want) == len(got) and all(
            expected == actual or (bool(actual) and bool(_PLACEHOLDER.fullmatch(expected)))
            for expected, actual in zip(want, got)
        )


class _Call(BaseModel):
    """A tool call's arguments, as the model sent them."""

    model_config = ConfigDict(extra='forbid', strict=True)
    method: HttpMethod
    path: str
    query: dict[str, QueryScalar | list[QueryScalar]] = Field(default_factory=dict)
    body: dict[str, JsonValue] | list[JsonValue] | None = None


class HttpTool(BaseModel):
    """An API tool: HTTP requests to declared routes under one loopback base URL.

    The model sees *name*, *description* (the place for the API docs a task needs) and the
    routes. *headers* go with every request (an API key for the local app, say) and are
    recorded with their values redacted. Responses come back whole: status, every header
    (pagination links included) and the full body.

    Example::

        HttpTool(
            name='bugzilla',
            description='Bugzilla 5.0 REST API: search bugs, read comments, add a comment.',
            base_url='http://127.0.0.1:8080',
            routes=(
                HttpRoute(method='GET', path='/rest/bug'),
                HttpRoute(method='GET', path='/rest/bug/{id}/comment'),
                HttpRoute(method='POST', path='/rest/bug/{id}/comment'),
            ),
            headers={'X-BUGZILLA-API-KEY': api_key},
        )
    """

    model_config = ConfigDict(frozen=True, extra='forbid')
    name: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    description: str = Field(min_length=1)
    base_url: str
    routes: tuple[HttpRoute, ...] = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator('name')
    @classmethod
    def check_name(cls, value: str) -> str:
        """Keep the done tool's name for the done tool."""
        if value == DONE_TOOL:
            raise ValueError(f'{DONE_TOOL!r} is reserved for the tool that ends the episode')
        return value

    @field_validator('base_url')
    @classmethod
    def check_base_url(cls, value: str) -> str:
        """Accept an http(s) URL on a loopback host, without credentials, query or fragment.

        Returns:
            The URL without a trailing slash, so a call's path appends to it.
        """
        parts = urlsplit(value)
        try:
            host = parts.hostname or ''
            loopback = host == 'localhost' or ipaddress.ip_address(host).is_loopback
            parts.port  # raises ValueError on a malformed port
        except ValueError:
            loopback = False
        if (
            parts.scheme not in ('http', 'https')
            or not loopback
            or '@' in parts.netloc
            or '?' in value
            or '#' in value
            or (parts.path and not _PATH.fullmatch(parts.path))
        ):
            raise ValueError(
                f'base_url {value!r} must be an http(s) URL on a loopback host '
                '(localhost, 127.0.0.0/8, ::1) with no credentials, query or fragment'
            )
        return value.rstrip('/')

    @property
    def input_schema(self) -> dict[str, JsonValue]:
        """The JSON schema of a call's arguments, as tool-calling APIs take it."""
        scalar: dict[str, JsonValue] = {'type': ['string', 'number', 'boolean']}
        return {
            'type': 'object',
            'properties': {
                'method': {'type': 'string', 'enum': sorted({r.method for r in self.routes})},
                'path': {
                    'type': 'string',
                    'description': (
                        f'Path under {self.base_url}, fitting one of these routes: '
                        f'{self._listing}. {{name}} stands for one path segment; '
                        'percent-encode reserved characters in it. Query parameters go in '
                        '"query", not in the path.'
                    ),
                },
                'query': {
                    'type': 'object',
                    'description': 'Query parameters; a list value repeats its parameter.',
                    'additionalProperties': {'anyOf': [scalar, {'type': 'array', 'items': scalar}]},
                },
                'body': {
                    'type': ['object', 'array'],
                    'description': 'JSON request body, for POST, PUT and PATCH only.',
                },
            },
            'required': ['method', 'path'],
            'additionalProperties': False,
        }

    @property
    def _listing(self) -> str:
        return ', '.join(f'{route.method} {route.path}' for route in self.routes)

    def prepare(self, arguments: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Check a call's *arguments*; return the request the runner sends, with its record.

        Args:
            arguments: The model's arguments: method, path, optional query and body.

        Returns:
            The request (method, url, headers, body text), whether it is a mutation (any
            method but GET), and its record: the same with header values redacted.

        Raises:
            ValueError: If an argument is unknown or of the wrong type, the path has a
                character to percent-encode or a dot segment, no declared route fits the
                method and path, or the method takes no body and one was given.
        """
        call = _Call.model_validate(arguments)
        decoded = unquote(call.path)
        if (
            not _PATH.fullmatch(call.path)
            or '\\' in decoded
            or any(part in ('.', '..') for part in decoded.split('/'))
        ):
            raise ValueError(
                f'path {call.path!r} must start with "/", percent-encode reserved '
                'characters and have no dot segments'
            )
        if not any(r.method == call.method and r.matches(call.path) for r in self.routes):
            raise ValueError(
                f'{call.method} {call.path} fits no route of this tool: {self._listing}'
            )
        if call.body is not None and call.method not in _BODY_METHODS:
            raise ValueError(f'{call.method} takes no body')
        pairs = [
            (key, item if isinstance(item, str) else json.dumps(item))
            for key, value in call.query.items()
            for item in (value if isinstance(value, list) else [value])
        ]
        url = self.base_url + call.path + (f'?{urlencode(pairs)}' if pairs else '')
        headers = dict(self.headers)
        record: dict[str, JsonValue] = {'method': call.method, 'url': url}
        if call.body is not None:
            headers['Content-Type'] = 'application/json'
            record['body'] = call.body
        record['headers'] = {
            name: _REDACTED if name in self.headers else value for name, value in headers.items()
        }
        return {
            'method': call.method,
            'url': url,
            'headers': headers,
            'body': None if call.body is None else json.dumps(call.body, ensure_ascii=False),
            'mutation': call.method != 'GET',
            'record': record,
        }


def run_command(python: str) -> str:
    """The constant command that sends the request in :data:`REQUEST_FILE` with *python*."""
    return f'{python} -I -c {shlex.quote(_RUNNER)} {REQUEST_FILE}'


def lost_response(request: Mapping[str, JsonValue], reason: str) -> str:
    """Record a call whose runner was stopped before it reported, as the runner would.

    A mutation is uncertain: it may have been applied before the response was lost.
    """
    record = {
        'request': request['record'],
        'error': f'No response: {reason}',
        'uncertain': request['mutation'],
    }
    return json.dumps(record, ensure_ascii=False)
