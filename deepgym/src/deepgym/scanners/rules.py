"""Rule scanners the user configures: regexes over chosen texts, and protected paths.

Both are JSON models told apart by ``kind`` (:data:`RuleScanner`, read by
:func:`load_rule_scanners`); their findings are ``verifier`` marks, and their version is a
hash of their config.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterator
from typing import Annotated, ClassVar, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter

from deepgym.labels import Name
from deepgym.scanners.core import Finding, ScannerSource, Severity
from deepgym.scanners.writes import action_writes
from deepgym.traces.schema import Trace, TraceStep, json_text, sha256_text

ScanField = Literal['action', 'observation', 'system', 'user', 'assistant', 'tool']
"""A step's text to scan: its action, its observation, or its messages of one role."""


class Pattern(BaseModel):
    """One regex, and what a match of it means."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    regex: re.Pattern[str]
    """Python ``re`` syntax; flags go inline, e.g. ``(?i)``."""
    message: Name
    severity: Severity = 'medium'


class _Rule(BaseModel):
    """What the rule scanners share: a JSON config, the ``verifier`` source, no instruction."""

    model_config = ConfigDict(frozen=True, extra='forbid')
    reads: ClassVar[str]
    """The version of how the scanner reads a trace; bump it whenever that changes."""

    name: Name

    @property
    def source(self) -> ScannerSource:
        """``verifier``: a rule's findings are labeled as a verifier's."""
        return 'verifier'

    @property
    def instruction_sha256(self) -> None:
        """None: a rule follows no written instruction."""
        return None

    @property
    def judge_id(self) -> None:
        """None: a rule asks no judge."""
        return None

    @property
    def version(self) -> str:
        """The version of the scanner's reading logic, ``+config-sha256:`` and the SHA-256 of
        its config as JSON (with each regex's flags, which the JSON leaves out)."""
        return f'{self.reads}+config-sha256:{sha256_text(json_text(self.config()))}'

    def config(self) -> dict[str, object]:
        """Return what decides the scanner's findings besides its reading logic."""
        return self.model_dump(mode='json')


class PatternScanner(_Rule):
    """Flag every match of each pattern in the chosen texts of every step.

    A match is cited by its code point span and text; an empty match has nothing to quote
    and is skipped.
    """

    reads: ClassVar[str] = 'pattern/1'

    kind: Literal['pattern'] = 'pattern'
    patterns: tuple[Pattern, ...] = Field(min_length=1)
    fields: tuple[ScanField, ...] = Field(default=('action',), min_length=1)

    def config(self) -> dict[str, object]:
        """Return the JSON config with each regex's flags."""
        return {**super().config(), 'flags': [pattern.regex.flags for pattern in self.patterns]}

    def scan(self, trace: Trace) -> list[Finding]:
        """Return a finding per match."""
        return [
            Finding(
                scanner=self.name,
                trace_id=trace.trace_id,
                step=step.index,
                field=field,
                span=match.span(),
                quote=match.group(),
                message=pattern.message,
                severity=pattern.severity,
            )
            for step in trace.steps
            for field, text in _texts(step, self.fields)
            for pattern in self.patterns
            for match in pattern.regex.finditer(text)
            if match.end() > match.start()
        ]


def _texts(step: TraceStep, fields: tuple[ScanField, ...]) -> Iterator[tuple[str, str]]:
    """Yield the ``(field, text)`` of each of *step*'s texts *fields* chooses."""
    if 'action' in fields:
        yield 'action', step.action.content
    if 'observation' in fields and step.observation is not None:
        yield 'observation', step.observation.content
    for number, message in enumerate(step.messages):
        if message.role in fields:
            yield f'message:{number}', message.content


class _Glob(NamedTuple):
    """A protected-path glob, compiled to match a normalised path whole (see :func:`_compile`)."""

    text: str
    """As configured."""
    regex: re.Pattern[str]


class ProtectedPathScanner(_Rule):
    """Flag each applied action that names a protected path as one it writes, deletes or
    moves.

    Only the forms :mod:`deepgym.scanners.writes` lists are read, so no finding means none
    of them named a protected path, not that no protected file changed: diff the final
    workspace for that. A rejected action changed nothing.

    Globs are normalised first (``./tests/**`` is ``tests/**``, ``tests/`` is ``tests``).
    ``**`` matches any run of folders, ``*`` and ``?`` stay within one name, and ``[...]``
    and ``{a,b}`` are literal text. A glob that matches a folder covers everything under it,
    so ``tests`` protects ``tests/test_x.py``. A glob starting with ``/`` matches absolute
    paths only, never a relative one (so never a ``write_file`` path). Any other matches
    the path's names wherever they are: ``tests/**`` matches ``/repo/tests/a.py``,
    ``conftest.py`` any ``conftest.py``, ``tests`` a folder named ``tests`` anywhere, and
    each also ``.`` and ``..`` (``rm -rf .``). A folder holding a glob's fixed leading part
    counts too: ``rm -r tests`` changes ``tests/conftest.py``. A finding quotes the path
    where the action's text has it; where it doesn't (``write_file``, ``cli``), its message
    names the path. An edit a CLI's tool reported failing is still flagged, as an attempt:
    "tried to change <path> ...; the tool reported failure", with the same severity.
    """

    reads: ClassVar[str] = 'protected-paths/4'
    """Covers :mod:`deepgym.scanners.writes`: bump it whenever what that reads changes."""

    kind: Literal['protected_paths'] = 'protected_paths'
    globs: tuple[Name, ...] = Field(min_length=1)
    severity: Severity = 'high'
    _globs: list[_Glob] = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        """Compile each glob."""
        self._globs = [_Glob(glob, _compile(glob)) for glob in self.globs]

    def scan(self, trace: Trace) -> list[Finding]:
        """Return a finding per path an applied action changes, or tried to, that a glob
        matches."""
        findings: list[Finding] = []
        for step in trace.steps:
            if not step.applied:
                continue
            for write in action_writes(step):
                norm = posixpath.normpath(write.path) if write.path else ''
                glob = next((g.text for g in self._globs if g.regex.fullmatch(norm)), None)
                if not norm or glob is None:
                    continue
                quote = None if write.span is None else step.action.content[slice(*write.span)]
                if write.failed:
                    message = (
                        f'tried to change {write.path} matching {glob}; the tool reported failure'
                    )
                else:
                    message = f'changes {"a path" if quote else write.path} matching {glob}'
                findings.append(
                    Finding(
                        scanner=self.name,
                        trace_id=trace.trace_id,
                        step=step.index,
                        field='action',
                        span=write.span,
                        quote=quote,
                        message=message,
                        severity=self.severity,
                    )
                )
        return findings


def _compile(glob: str) -> re.Pattern[str]:
    """Compile *glob* to one regex a normalised path matches whole when the glob matches it
    or a folder it is in, or it is a folder holding the glob's fixed leading part.

    ``**/`` is any run of folders, ``**`` anything, ``*`` and ``?`` part of one name. A glob
    not starting with ``/`` may start after any folder (``(?:.*/)?``, so its own leading
    ``**/`` adds nothing and is dropped), and anything under a match counts
    (``(?:/.*)?``). Such a glob also matches ``.`` and ``..``, folders that hold every
    relative path (``rm -rf .``). One match per path, never a match per slice of it.
    """
    norm = posixpath.normpath(glob)
    anchored = norm.startswith('/')
    body = norm if anchored else re.sub(r'^(?:\*\*/)+', '', norm)
    wild = {'**/': '(?:.*/)?', '**': '.*', '*': '[^/]*', '?': '[^/]'}
    parts = re.findall(r'\*\*/|\*\*|\*|\?|[^*?]+', body)
    pattern = ''.join(wild.get(part) or re.escape(part) for part in parts)
    fixed = re.split(r'[*?]', norm)[0]
    folders = [re.escape(fixed[:at] or '/') for at, char in enumerate(fixed) if char == '/']
    if not anchored:  # the work directory, or a folder above it, holds every relative path
        folders.append(r'\.|\.\.(?:/\.\.)*')
    alternatives = '|'.join([f'{pattern}(?:/.*)?', *folders])
    return re.compile(f'{"" if anchored else "(?:.*/)?"}(?:{alternatives})', re.DOTALL)


RuleScanner = Annotated[PatternScanner | ProtectedPathScanner, Field(discriminator='kind')]

_RULE_SCANNERS = TypeAdapter(Annotated[list[RuleScanner], Field(min_length=1)])


def load_rule_scanners(text: str) -> list[PatternScanner | ProtectedPathScanner]:
    """Read a JSON list of one or more rule scanners, each told apart by ``kind``.

    Raises:
        pydantic.ValidationError: If the JSON doesn't describe one or more valid scanners.
    """
    return _RULE_SCANNERS.validate_json(text)
