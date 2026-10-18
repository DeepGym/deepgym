"""HTTP API tools an environment declares for tool-calling agents, and how a session runs them.

A tool is trusted environment configuration, never model output: its name, base URL, routes
and headers come from the environment's author. A model supplies only a call's arguments
(method, path, query, body). :meth:`HttpTool.prepare` checks them against the declared
routes and builds the request from validated parts: host, port and the exact request
target. The command that sends it is a constant (:func:`run_command`) reading the request
on stdin, so nothing the model writes reaches a shell or a file it could read or change; it
runs as the agent user in the sandbox's offline network namespace, where only the episode's
own services answer, with http.client, which parses no URL and follows no redirect (see
:mod:`deepgym._http_request`).
"""

from __future__ import annotations

import ipaddress
import json
import re
import shlex
import unicodedata
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlencode, urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

HttpMethod = Literal['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
QueryScalar = str | int | float | bool

DONE_TOOL = 'done'
"""Reserved tool name: agents offer it so the model ends the episode with a ``done`` action."""

REDACTED = '[redacted]'
"""What stands in for a tool header's value in every record and result."""

_RUNNER = Path(__file__).with_name('_http_request.py').read_text(encoding='utf-8')
_BODY_METHODS = frozenset({'POST', 'PUT', 'PATCH'})
# RFC 3986 path characters; a model percent-encodes anything else, so nothing it sends can
# start a query, a fragment or another authority, and no space or control character is sent.
_SEGMENT = re.compile(r"[A-Za-z0-9\-._~!$&'()*+,;=:@]+")
_PATH = re.compile(r"/[A-Za-z0-9\-._~!$&'()*+,;=:@%/]*")
_STRAY_PERCENT = re.compile(r'%(?![0-9A-Fa-f]{2})')
PLACEHOLDER = re.compile(r'\{\w+(:path)?\}')
"""A route template's ``{name}`` or ``{name:path}`` segment."""
# '.', '..', '...', '. .': segments servers resolve as dot segments, however spelled.
_DOTS = re.compile(r'[. ]*\.[. ]*')
_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
# Names frameworks (Rack, Laravel, Symfony, Express) read to replace the request's method.
_METHOD_OVERRIDES = frozenset(
    {'_method', 'x-http-method-override', 'x-http-method', 'x-method-override'}
)
_RESERVED_HEADERS = _METHOD_OVERRIDES | {'host', 'content-length', 'transfer-encoding'}
_PORTS = {'http': 80, 'https': 443}


def _readings(text: str) -> Iterator[str]:
    """*text* and each further percent-decoding of it until none changes, NFKC-normalized:
    every way an app that decodes once, twice, ... or folds compatibility forms may read it.
    Each decoding is shorter than the last, so this ends."""
    while True:
        yield unicodedata.normalize('NFKC', text)
        decoded = unquote(text)
        if decoded == text:
            return
        text = decoded


def _unsafe(path: str) -> bool:
    """Whether a reading of *path* has a dot segment (``..;x`` too), a backslash or a
    control character."""
    return any(
        '\\' in form
        or any(char < ' ' or char == '\x7f' for char in form)
        or any(_DOTS.fullmatch(part.split(';', 1)[0]) for part in form.split('/'))
        for form in _readings(path)
    )


class HttpRoute(BaseModel):
    """One endpoint a tool may call: a method and a path template.

    ``{name}`` in the template stands for exactly one non-empty path segment, which may not
    hold an encoded ``/`` either, since an app that decodes before routing would read more
    segments; ``{name:path}`` allows one (GitLab's ``group%2Fproject`` ids and file paths).
    Every other segment must match literally. An endpoint no route names (an admin page, a
    script runner) is unreachable.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')
    method: HttpMethod
    path: str

    @field_validator('path')
    @classmethod
    def check_path(cls, value: str) -> str:
        """Accept '/' or '/'-separated literal segments and placeholders."""
        segments = value.split('/')
        literal = [segment for segment in segments[1:] if not PLACEHOLDER.fullmatch(segment)]
        if value != '/' and (
            segments[0]
            or not all(_SEGMENT.fullmatch(part) and not _DOTS.fullmatch(part) for part in literal)
        ):
            raise ValueError(
                f'route path {value!r} must start with "/" and hold only non-empty literal '
                'segments (no dot segments or percent signs) and {name} placeholders'
            )
        return value

    def matches(self, path: str) -> bool:
        """Whether request *path* fits this template, segment by segment."""
        want, got = self.path.split('/'), path.split('/')
        return len(want) == len(got) and all(map(_fits, want, got))


def _fits(template: str, segment: str) -> bool:
    """Whether request path *segment* fits one *template* segment."""
    if not PLACEHOLDER.fullmatch(template):
        return template == segment
    if template.endswith(':path}'):
        return bool(segment)
    return bool(segment) and not any('/' in form for form in _readings(segment))


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
    routes. *headers* go with every request and are treated as credentials (an API key for
    the local app, say): their values appear in no record, result, trace or error, and
    never on disk where the agent user could read them. Responses come back whole: status,
    every header (pagination links included) and the full body.

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
            The URL rebuilt from its checked parts: scheme, host, explicit port and path
            without a trailing slash, so a call's path appends to it.
        """
        parts = urlsplit(value)
        try:
            host = parts.hostname or ''
            loopback = host == 'localhost' or ipaddress.ip_address(host).is_loopback
            port = parts.port or _PORTS.get(parts.scheme)
        except ValueError:
            loopback = False
        path = parts.path.rstrip('/')
        if (
            parts.scheme not in _PORTS
            or not loopback
            or not value.isascii()
            or not value.isprintable()
            or any(mark in value for mark in '@?#\\')
            or (path and (not _PATH.fullmatch(path) or _unsafe(path)))
        ):
            raise ValueError(
                f'base_url {value!r} must be an http(s) URL on a loopback host '
                '(localhost, 127.0.0.0/8, ::1) with no credentials, query or fragment'
            )
        netloc = f'[{host}]' if ':' in host else host
        return f'{parts.scheme}://{netloc}:{port}{path}'

    @field_validator('headers')
    @classmethod
    def check_headers(cls, value: dict[str, str]) -> dict[str, str]:
        """Accept token names, none that reroutes or reframes a request, with plain values."""
        for name, text in value.items():
            if (
                not _TOKEN.fullmatch(name)
                or name.lower() in _RESERVED_HEADERS
                or not text
                or any((char < ' ' and char != '\t') or char == '\x7f' for char in text)
            ):
                raise ValueError(
                    f'header {name!r} must be a token name other than '
                    f'{sorted(_RESERVED_HEADERS)}, with a non-empty value and no control '
                    'characters'
                )
        return value

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
            The request from validated parts (scheme, host, port, the exact request target,
            method, headers, body text), whether it is a mutation (any method but GET), and
            its record: the URL it sends to, with header values redacted.

        Raises:
            ValueError: If an argument is unknown or of the wrong type; the path has a
                character to percent-encode, a stray percent sign, or a reading with a dot
                segment, backslash or control character; no declared route fits the method
                and path; a body comes with a method that takes none, is not plain JSON, or
                it or the query names a method override.
        """
        call = _Call.model_validate(arguments)
        if not _PATH.fullmatch(call.path) or _STRAY_PERCENT.search(call.path) or _unsafe(call.path):
            raise ValueError(
                f'path {call.path!r} must start with "/", percent-encode reserved characters '
                'and have no dot segments, backslashes or control characters, encoded or not'
            )
        if not any(r.method == call.method and r.matches(call.path) for r in self.routes):
            raise ValueError(
                f'{call.method} {call.path} fits no route of this tool: {self._listing}'
            )
        if call.body is not None and call.method not in _BODY_METHODS:
            raise ValueError(f'{call.method} takes no body')
        keys = [*call.query, *(call.body if isinstance(call.body, dict) else ())]
        if overrides := [key for key in keys if key.lower() in _METHOD_OVERRIDES]:
            raise ValueError(f'{overrides} would override the request method')
        pairs = [
            (key, item if isinstance(item, str) else json.dumps(item))
            for key, value in call.query.items()
            for item in (value if isinstance(value, list) else [value])
        ]
        base = urlsplit(self.base_url)
        target = base.path + call.path + (f'?{urlencode(pairs)}' if pairs else '')
        headers = dict(self.headers)
        record: dict[str, JsonValue] = {
            'method': call.method,
            'url': f'{base.scheme}://{base.netloc}{target}',
        }
        body = None
        if call.body is not None:
            body = json.dumps(call.body, ensure_ascii=False, allow_nan=False)
            headers['Content-Type'] = 'application/json'
            record['body'] = call.body
        record['headers'] = {
            name: REDACTED if name in self.headers else value for name, value in headers.items()
        }
        return {
            'scheme': base.scheme,
            'host': base.hostname,
            'port': base.port,
            'target': target,
            'method': call.method,
            'headers': headers,
            'body': body,
            'mutation': call.method != 'GET',
            'record': record,
        }

    def redact(self, text: str) -> str:
        """*text* with each header value, as is and JSON-escaped, replaced by REDACTED, in
        case the app echoes one back."""
        for value in self.headers.values():
            for form in {value, json.dumps(value)[1:-1]}:
                text = text.replace(form, REDACTED)
        return text


def run_command(python: str) -> str:
    """The constant command that sends the request it reads on stdin, with *python*."""
    return f'{python} -I -c {shlex.quote(_RUNNER)}'


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
