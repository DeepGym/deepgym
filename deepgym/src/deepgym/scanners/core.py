"""Findings, scan runs, groups, labels and agreement with human reviewers.

A :class:`Finding` cites what a scanner flagged: a trace, a step or the whole trace, a text
of that step and optionally a code point span of it with the quoted text, as
:class:`~deepgym.labels.LabelTarget` points at text. :func:`scan_traces` runs scanners
over traces; a scanner that raises, or returns anything but findings that fit the trace,
is an error for that trace, never "no findings". :func:`group_findings` clusters findings,
:func:`finding_labels` turns findings and errors into label records (marks, not verdicts),
and :func:`scanner_agreement` compares each scanner with the human labels on the same
traces.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from deepgym.dataset import LabelMatch
from deepgym.labels import (
    Label,
    LabelCreate,
    LabelTarget,
    Name,
    Offset,
    Text,
    current_labels,
    excluded,
    is_verdict,
)
from deepgym.rubric_calibration import Confusion
from deepgym.traces.schema import Trace, check_unique_ids

Severity = Literal['low', 'medium', 'high']
ScannerSource = Literal['verifier', 'judge']
"""The label source of a scanner's findings: ``verifier`` for rules, ``judge`` for a model."""

SCAN_ERROR = 'scan_error'
"""The value of the whole-trace mark :func:`finding_labels` gives a trace a scanner failed on."""


class Finding(BaseModel):
    """One thing a scanner flagged in a trace, cited as a label target cites text.

    ``field`` is ``'action'`` (``Action.content``), ``'observation'``
    (``Observation.content``) or ``'message:<i>'`` (``TraceStep.messages[i].content``); a
    span is ``(start, end)`` in Unicode code points, end exclusive, and ``quote`` the text
    it spans.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    scanner: Name
    trace_id: Name
    step: Offset | None = None
    """``TraceStep.index`` (from 1); None for the whole trace."""
    field: str | None = None
    span: tuple[Offset, Offset] | None = None
    quote: Text | None = None
    message: Name
    """One short line saying what was found."""
    severity: Severity = 'medium'

    @model_validator(mode='after')
    def check_target(self) -> Finding:
        """Hold the citation to the label target rules (see :class:`LabelTarget`)."""
        _ = self.target
        return self

    @property
    def target(self) -> LabelTarget:
        """The label target this finding cites."""
        return LabelTarget(
            kind='trace',
            id=self.trace_id,
            step=self.step,
            field=self.field,
            span=self.span,
            quote=self.quote,
        )


class Scanner(Protocol):
    """Anything that reads one trace and returns what it flagged in it."""

    @property
    def name(self) -> str:
        """Names the scanner in findings, errors and labels (their name and author)."""
        ...

    @property
    def source(self) -> ScannerSource:
        """The source its findings are labeled with."""
        ...

    @property
    def version(self) -> str:
        """What decides its findings besides the trace (a prompt version, a config hash);
        one line."""
        ...

    @property
    def instruction_sha256(self) -> str | None:
        """SHA-256 of the user-written instruction it follows, if it takes one."""
        ...

    @property
    def judge_id(self) -> str | None:
        """Names the judge model it asks, if it asks one and the caller said which."""
        ...

    def scan(self, trace: Trace) -> Iterable[Finding]:
        """Return what *trace* holds that this scanner looks for; raise if it can't tell."""
        ...


class ScannerInfo(BaseModel):
    """What a scan report records about one scanner."""

    model_config = ConfigDict(frozen=True)

    source: ScannerSource
    version: Name
    instruction_sha256: str | None = None
    judge_id: str | None = None


class ScanError(BaseModel):
    """A scanner that failed on a trace: it neither found nor cleared anything there."""

    model_config = ConfigDict(frozen=True)

    scanner: str
    trace_id: str
    error: str


class ScanReport(BaseModel):
    """What one :func:`scan_traces` run found, and where its scanners failed."""

    model_config = ConfigDict(frozen=True)

    run_id: Name = Field(default_factory=lambda: uuid4().hex)
    """The labels of this run share it as their ``group_id``, and their ids start with it."""
    scanners: dict[str, ScannerInfo]
    """Each scanner by name."""
    traces: tuple[str, ...]
    """The ids of the traces scanned."""
    findings: tuple[Finding, ...]
    errors: tuple[ScanError, ...]

    def failed(self, scanner: str) -> set[str]:
        """Return the traces *scanner* failed on."""
        return {error.trace_id for error in self.errors if error.scanner == scanner}


def field_text(trace: Trace, step: int, field: str) -> str:
    """Return the text *field* names on *trace*'s *step* (see :class:`Finding`).

    Raises:
        ValueError: If the trace has no such step or text.
    """
    if not 1 <= step <= len(trace.steps):
        raise ValueError(f'trace {trace.trace_id} has no step {step}')
    found = trace.steps[step - 1]
    if field == 'action':
        return found.action.content
    if field == 'observation':
        if found.observation is None:
            raise ValueError(f'step {step} of trace {trace.trace_id} has no observation')
        return found.observation.content
    number = int(field.removeprefix('message:'))
    if number >= len(found.messages):
        raise ValueError(f'step {step} of trace {trace.trace_id} has no message {number}')
    return found.messages[number].content


def _check_citation(trace: Trace, scanner: Scanner, finding: Finding) -> None:
    """Refuse a finding of another scanner or trace, or one citing text the trace lacks.

    Raises:
        ValueError: Saying what doesn't fit.
    """
    if (finding.scanner, finding.trace_id) != (scanner.name, trace.trace_id):
        raise ValueError(
            f'a finding of scanner {finding.scanner} on trace {finding.trace_id} came back '
            f'from scanner {scanner.name} on trace {trace.trace_id}'
        )
    step = finding.step
    if step is None:
        return
    text = field_text(trace, step, finding.field or 'action')  # the step exists
    if finding.span is not None and text[slice(*finding.span)] != finding.quote:
        raise ValueError(
            f'step {finding.step} {finding.field} {list(finding.span)} is '
            f'{text[slice(*finding.span)]!r}, not the quote {finding.quote!r}'
        )


def _scan_one(trace: Trace, scanner: Scanner) -> list[Finding] | ScanError:
    """Run *scanner* on *trace*: its checked findings, or the error that stopped it."""
    try:
        result = scanner.scan(trace)
        if isinstance(result, (str, bytes, Mapping)):  # iterable, but not findings
            raise TypeError(f'scan returned a {type(result).__name__}, not findings')
        found = list(result)
        for finding in found:
            if not isinstance(finding, Finding):
                raise TypeError(f'scan returned a {type(finding).__name__}, not a Finding')
            _check_citation(trace, scanner, finding)
    except Exception as exc:  # noqa: BLE001 - a failed scan is an error, never "no findings"
        return ScanError(
            scanner=scanner.name, trace_id=trace.trace_id, error=f'{type(exc).__name__}: {exc}'
        )
    return found


def scan_traces(
    traces: Sequence[Trace], scanners: Sequence[Scanner], max_parallel: int = 1
) -> ScanReport:
    """Run every scanner on every trace.

    Args:
        traces: The traces, each id once.
        scanners: The scanners, each name once; a name may not be a verdict name
            (``passed`` or a rubric check id), since findings are labeled by it.
        max_parallel: Most scans at once (threads: worth raising for a model scanner).

    Returns:
        The findings, and an error for each trace a scanner failed on.

    Raises:
        ValueError: If no scanner is given, two traces share an id, two scanners a name,
            or a scanner is named like a verdict.
    """
    if not scanners:
        raise ValueError('scan_traces needs at least one scanner')
    check_unique_ids(traces)
    names = [scanner.name for scanner in scanners]
    if len(set(names)) < len(names):
        raise ValueError(f'scanners share names: {names}')
    if verdicts := [name for name in names if is_verdict(name)]:
        raise ValueError(f'scanner names {verdicts} are verdict names; findings are marks')
    jobs = [(trace, scanner) for trace in traces for scanner in scanners]
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        outcomes = list(pool.map(lambda job: _scan_one(*job), jobs))
    return ScanReport(
        scanners={
            scanner.name: ScannerInfo(
                source=scanner.source,
                version=scanner.version,
                instruction_sha256=scanner.instruction_sha256,
                judge_id=scanner.judge_id,
            )
            for scanner in scanners
        },
        traces=tuple(trace.trace_id for trace in traces),
        findings=tuple(found for each in outcomes if isinstance(each, list) for found in each),
        errors=tuple(each for each in outcomes if isinstance(each, ScanError)),
    )


class FindingGroup(BaseModel):
    """Findings of one scanner with the same normalised message (and key, if one is used)."""

    model_config = ConfigDict(frozen=True)

    scanner: str
    message: str
    """The normalised message: casefolded, runs of whitespace as one space."""
    key: str | None
    findings: tuple[Finding, ...]
    """Every finding of the group, in the order given: the citations."""

    @property
    def traces(self) -> int:
        """How many traces the group's findings are on."""
        return len({finding.trace_id for finding in self.findings})


def _normalised(message: str) -> str:
    """Return *message* casefolded, with each run of whitespace as one space."""
    return ' '.join(message.casefold().split())


def group_findings(
    findings: Sequence[Finding], key: Callable[[Finding], str] | None = None
) -> list[FindingGroup]:
    """Cluster *findings* by scanner and normalised message, and by *key* when given.

    Returns:
        The groups, the largest first, then by scanner, message and key.
    """
    groups: dict[tuple[str, str, str | None], list[Finding]] = {}
    for finding in findings:
        about = (finding.scanner, _normalised(finding.message), key(finding) if key else None)
        groups.setdefault(about, []).append(finding)
    ordered = sorted(
        groups.items(), key=lambda item: (-len(item[1]), item[0][:2], item[0][2] or '')
    )
    return [
        FindingGroup(scanner=scanner, message=message, key=group_key, findings=tuple(found))
        for (scanner, message, group_key), found in ordered
    ]


def finding_labels(report: ScanReport) -> list[LabelCreate]:
    """Return a label record for each finding and each error of *report*, to store in a
    :class:`~deepgym.console.label_store.LabelStore`.

    A finding's label marks its target (trace, step, field, span and quote), valued by its
    severity, with its message as the comment. An error's label marks the whole trace the
    scanner failed on, valued :data:`SCAN_ERROR`, with the error as the comment, so a trace
    the scanner couldn't clear never reads as clean. Each is named and authored by its scanner,
    with the scanner's source, its version as ``rubric_version``, its instruction's SHA-256
    (if any) as ``rubric_sha256``, the run's id as ``group_id`` and the id
    ``<run_id>-<n>``, so storing the same report again is a retry that stores nothing new.
    A mark is no verdict, so it works with a dataset's ``forbid_labels`` (see
    :func:`scanner_forbids`) and review queues.
    """
    marks = [
        (finding.scanner, finding.target, finding.severity, finding.message)
        for finding in report.findings
    ]
    marks += [
        (error.scanner, LabelTarget(kind='trace', id=error.trace_id), SCAN_ERROR, error.error)
        for error in report.errors
    ]
    return [
        LabelCreate(
            id=f'{report.run_id}-{number}',
            target=target,
            name=scanner,
            value=value,
            source=report.scanners[scanner].source,
            author=scanner,
            comment=comment,
            rubric_version=report.scanners[scanner].version,
            rubric_sha256=report.scanners[scanner].instruction_sha256,
            group_id=report.run_id,
        )
        for number, (scanner, target, value, comment) in enumerate(marks)
    ]


def scanner_forbids(
    name: str, *, source: ScannerSource | None = None, values: Sequence[Severity] | None = None
) -> tuple[LabelMatch, ...]:
    """Return the ``DatasetSpec.forbid_labels`` entries that drop the traces scanner *name*
    flagged (only findings of the severities in *values*, when given) and always those it
    failed on (:data:`SCAN_ERROR`), so a value filter never keeps a trace nobody cleared.
    """
    flagged = [None] if values is None else list(values)
    return tuple(
        LabelMatch(name=name, source=source, value=value) for value in [*flagged, SCAN_ERROR]
    )


class HumanMark(BaseModel):
    """The human label a scanner is measured against: a bool label named ``name``."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: Name
    flagged: StrictBool = True
    """The value that means the scanner should flag the target: True for a mark such as
    ``reward_hack``, False for ``passed`` (a failed answer is the problem)."""


class ScannerAgreement(BaseModel):
    """A scanner against human reviewers; a flag is the positive."""

    model_config = ConfigDict(frozen=True)

    traces: Confusion
    """On whole traces the humans labeled: flagged if any finding is on the trace."""
    steps: Confusion
    """On steps the humans labeled: flagged if a finding is on the step."""
    contested: int
    """Traces and steps left out because their human labels disagree."""
    scan_errors: int
    """Traces and steps humans labeled, left out because the scanner failed on the trace."""


def scanner_agreement(
    report: ScanReport, labels: Sequence[Label], marks: Mapping[str, HumanMark]
) -> dict[str, ScannerAgreement]:
    """Compare each scanner in *marks* with the current human labels on the scanned traces.

    Only ``trace`` targets the report scanned count, outside traces a current invalidation
    excludes. A whole-trace label is compared with the trace's findings, a label on a step
    with that step's; a target the humans never labeled is never counted as clean.

    Args:
        report: What the scanners found.
        labels: Label records, whole target histories (see
            :func:`~deepgym.labels.current_labels`).
        marks: Per scanner name, the human label it should agree with.

    Returns:
        Per scanner name in *marks*, its agreement.

    Raises:
        ValueError: If *marks* names a scanner the report doesn't have.
    """
    if unknown := marks.keys() - report.scanners.keys():
        raise ValueError(f'the report has no scanners {sorted(unknown)}')
    scanned = set(report.traces) - {key for kind, key in excluded(labels) if kind == 'trace'}
    human = [
        label
        for label in current_labels(labels)
        if label.source == 'human'
        and label.type == 'label'
        and label.target.kind == 'trace'
        and label.target.id in scanned
        and isinstance(label.value, bool)
    ]
    return {
        name: _agreement(report, name, mark, [label for label in human if label.name == mark.name])
        for name, mark in marks.items()
    }


def _agreement(
    report: ScanReport, scanner: str, mark: HumanMark, labels: Sequence[Label]
) -> ScannerAgreement:
    """Compare *scanner*'s findings with the human *labels* of its mark."""
    reference: dict[tuple[str, int | None], set[bool]] = {}
    for label in labels:
        reference.setdefault((label.target.id, label.target.step), set()).add(
            label.value == mark.flagged
        )
    findings = [finding for finding in report.findings if finding.scanner == scanner]
    flagged = {(finding.trace_id, None) for finding in findings}
    flagged |= {(finding.trace_id, finding.step) for finding in findings}
    failed = report.failed(scanner)
    pairs: dict[bool, list[tuple[bool, bool]]] = {True: [], False: []}  # by whole trace
    contested = errors = 0
    for (trace_id, step), values in reference.items():
        if trace_id in failed:
            errors += 1
        elif len(values) > 1:
            contested += 1
        else:
            (value,) = values
            pairs[step is None].append(((trace_id, step) in flagged, value))
    return ScannerAgreement(
        traces=Confusion.of(pairs[True]),
        steps=Confusion.of(pairs[False]),
        contested=contested,
        scan_errors=errors,
    )
