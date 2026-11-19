"""Request and response models for the console API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from deepgym.dataset import DatasetCounts, DatasetManifest, DatasetSpec
from deepgym.dataset.manifest import MAX_FOLDER_BYTES
from deepgym.labels import SQLITE_MAX_INT, Label, LabelSource, LabelType, Name, TargetKind
from deepgym.models import Action, RunResult, Trajectory
from deepgym.review_queues import ReviewLabel
from deepgym.sandboxes.cli import RunningSandbox
from deepgym.scanners import RuleScanner, ScanReport
from deepgym.traces import Trace, Usage
from deepgym.traces.export import ExportFormat

RunMode = Literal['local', 'daytona', 'auto', 'apple', 'docker', 'modal', 'e2b']
RunStatus = Literal['queued', 'running', 'completed', 'failed']
EnvKind = Literal['single-turn', 'multi-turn', 'computer-use']
LocalProvider = Literal['apple', 'docker']


class EnvironmentSummary(BaseModel):
    """One registry entry, as listed in the environments table."""

    id: str
    """Name to pass to load_environment, e.g. 'coin_change' or 'cyber/web_path_traversal_toy'."""

    name: str
    kind: EnvKind
    type: str
    difficulty: str
    domain: str
    family: str = ''
    tags: list[str] = Field(default_factory=list)
    variants: int = 1


class EnvironmentDetail(EnvironmentSummary):
    """An environment with its task, verifier source and reference solution."""

    variant: int = 0
    task: str
    verifier: str
    reference: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class RunCreate(BaseModel):
    """Start one run per environment, all with the same solution and provider."""

    environments: list[str] = Field(min_length=1, max_length=50)
    mode: RunMode = 'local'
    solution: str | None = None
    """Solution source; None runs each environment's reference solution."""


class RunConfig(BaseModel):
    """What the console allows when starting runs."""

    local_custom_code: bool
    """Whether custom solutions may run on this machine (--allow-local-exec)."""


class RunSpec(BaseModel):
    """What one run executes: an environment and the solution to score against it."""

    model_config = ConfigDict(frozen=True)

    environment: str
    solution: str
    reference: bool
    environment_sha256: str
    """Hash of the environment definition (:func:`deepgym.traces.schema.environment_sha256`)."""


class RunSummary(BaseModel):
    """A run as listed in the runs table."""

    id: str
    group_id: str
    """Shared by runs started together (a small eval)."""

    environment: str
    mode: RunMode
    provider: str
    """Where the run executed: 'local', 'daytona' or a sandbox provider."""

    reference: bool
    status: RunStatus
    score: float | None = None
    passed: bool | None = None
    error_type: str | None = None
    duration_ms: float | None = None
    created_at: str
    finished_at: str | None = None
    environment_sha256: str | None = None
    """Hash of the environment definition the run was scored against; None for runs
    stored before the console recorded it."""


class RunPage(BaseModel):
    """One page of the runs table, newest first."""

    runs: list[RunSummary]
    total: int
    """Matching runs across every page."""


class RunDetail(RunSummary):
    """A run with its solution, error and full verifier result."""

    solution: str
    error: str | None = None
    result: RunResult | None = None


class RunTotals(BaseModel):
    """Counts and means over a set of runs."""

    runs: int
    finished: int
    """Runs that completed or failed."""

    passed: int
    errors: int
    """Runs with an error_type: they raised, were interrupted, or the verifier reported one."""

    mean_score: float | None = None
    """Mean score of the finished runs, a run without a verdict counting 0."""

    mean_duration_ms: float | None = None

    judged_count: int = 0
    """Finished runs with a verdict (a score)."""

    no_verdict_count: int = 0
    """Finished runs without a verdict: they raised, were interrupted, or the verifier or
    sandbox failed instead of judging them."""

    judged_mean_score: float | None = None
    """Mean score of the judged runs only."""


class EnvironmentStats(RunTotals):
    """Totals of one environment's runs."""

    environment: str


class ErrorTypeCount(BaseModel):
    """How many runs ended with one error_type."""

    error_type: str
    count: int


class RunStats(BaseModel):
    """Aggregates of the runs started in a time window, for the overview and runs pages."""

    hours: int | None
    """Window length in hours; None covers every run."""

    current: RunTotals
    previous: RunTotals | None = None
    """The window of the same length just before, for deltas; None when hours is None."""

    score_bins: list[int]
    """Finished runs per score tenth: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]."""

    error_types: list[ErrorTypeCount]
    """Most common first."""

    environments: list[EnvironmentStats]


class EpisodeCreate(BaseModel):
    """A finished episode to store for replay."""

    environment: str = ''
    trajectory: Trajectory
    result: RunResult | None = None


class EpisodeSummary(BaseModel):
    """An episode as listed in the episodes table."""

    id: str
    environment: str
    steps: int
    total_reward: float
    score: float | None = None
    """None when the episode was recorded without a result, or its result has no verdict."""

    passed: bool | None = None
    has_verdict: bool | None = None
    """Whether the verifier judged the episode (``RunResult.has_verdict``); None when it was
    recorded without a result."""

    error_type: str | None = None
    """The result's error type; always set when it has no verdict."""

    created_at: str


class EpisodePage(BaseModel):
    """One page of the episodes table, newest first."""

    episodes: list[EpisodeSummary]
    total: int
    """Matching episodes across every page."""


class Frame(BaseModel):
    """One replay frame: an observation and the action taken from it (none for the last)."""

    index: int
    content: str
    done: bool = False
    metadata: dict[str, object] | None = None
    action: Action | None = None
    reward: float | None = None
    has_screenshot: bool = False


class EpisodeDetail(EpisodeSummary):
    """An episode with every frame (screenshots are served separately) and its result."""

    frames: list[Frame]
    result: RunResult | None = None


class ProviderSandboxes(BaseModel):
    """Live DeepGym sandboxes of one local container CLI."""

    provider: LocalProvider
    available: bool
    error: str | None = None
    sandboxes: list[RunningSandbox] = Field(default_factory=list)


def _in_utc(moment: datetime) -> datetime:
    """Return *moment* in UTC.

    Raises:
        ValueError: If it falls outside the years a datetime holds once in UTC.
    """
    try:
        return moment.astimezone(timezone.utc)
    except OverflowError as exc:
        raise ValueError('is out of range once converted to UTC') from exc


UtcTime = Annotated[AwareDatetime, AfterValidator(_in_utc)]


class LabelQuery(BaseModel):
    """Which labels to list: those matching every filter given."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    type: LabelType | None = None
    target_kind: TargetKind | None = None
    target_id: str | None = None
    step: int | None = Field(default=None, ge=0, le=SQLITE_MAX_INT)
    name: str | None = None
    source: LabelSource | None = None
    author: str | None = None
    blind: bool | None = None
    rubric_version: str | None = None
    rubric_sha256: str | None = None
    group_id: str | None = None
    since: UtcTime | None = None
    """Created at or after this time."""

    until: UtcTime | None = None
    """Created before this time."""

    after_seq: int | None = Field(default=None, ge=0, le=SQLITE_MAX_INT)
    """Only labels stored after the one at this ``seq``: what a sync hasn't seen yet."""

    current: bool = False
    """Only records still in force: nothing supersedes them, and they aren't retractions
    (``deepgym.labels.current_labels``). Safe to combine with any other filter."""


class LabelPage(BaseModel):
    """One page of labels, oldest first."""

    labels: list[Label]
    total: int
    """Matching labels across every page."""


class ReviewerBody(BaseModel):
    """Who asks for an item, or hands one back."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    reviewer: Name


class ReviewSubmit(ReviewerBody):
    """A reviewer's review of an item they hold."""

    labels: list[ReviewLabel] = Field(min_length=1)


class TraceQuery(BaseModel):
    """Which traces to list: those matching every filter given."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    environment: str | None = None
    """``EnvFingerprint.name``."""

    agent: str | None = None
    """``AgentMeta.name``."""

    model: str | None = None
    """``AgentMeta.model``."""

    has_verdict: bool | None = None

    def matches(self, trace: Trace) -> bool:
        """Return whether *trace* matches every filter given."""
        provenance = trace.provenance
        actual = {
            'environment': provenance.env.name,
            'agent': provenance.agent.name,
            'model': provenance.agent.model,
            'has_verdict': trace.has_verdict,
        }
        return all(wanted is None or actual[name] == wanted for name, wanted in self)


class TraceSummary(BaseModel):
    """A trace as listed: :func:`deepgym.traces.summary_row` without the trace itself."""

    trace_id: str
    env_name: str | None
    env_variant: int | None
    agent: str
    model: str | None
    provider: str
    status: Literal['complete', 'incomplete']
    has_verdict: bool
    score: float | None
    """None without a verdict."""

    passed: bool | None
    steps: int
    usage: Usage | None
    cost_usd: float | None
    usage_complete: bool
    started_at: AwareDatetime


class TraceConflict(BaseModel):
    """A trace id whose stored copies don't read as one trace, so it is left out."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    reason: Literal['copies_disagree', 'case_collision']
    """``copies_disagree``: keeping either copy would drop the other
    (:func:`deepgym.traces.merge_copies`). ``case_collision``: another trace's id is the same
    but for case, which exports can't tell apart."""
    episodes: tuple[str, ...]
    """The console episodes holding its copies."""
    detail: str


class StoredProblems(BaseModel):
    """Stored traces a route left out: whatever was asked, these never read as one trace."""

    conflicts: list[TraceConflict] = Field(default_factory=list)
    unreadable: list[str] = Field(default_factory=list)
    """Episodes whose stored trace doesn't read as a trace."""


class TracePage(StoredProblems):
    """One page of the traces the console holds, newest first, and every stored trace left out."""

    traces: list[TraceSummary]
    total: int
    """Matching traces across every page."""


class TraceDetail(BaseModel):
    """A whole trace, its usage and cost totals included, with its current labels."""

    trace: Trace
    labels: list[Label]
    """The trace's records still in force, its steps' included (``/api/labels/current``)."""
    episode_labels: list[Label]
    """The records in force on the console episodes recorded from it, in ``seq`` order. An
    episode numbers its replay frames, not the trace's steps: frame k shows step k's
    observation (the initial one for k = 0) and step k + 1's action, when no action was
    rejected."""
    unreadable: list[str]
    """Episodes holding a copy of it (or of an id differing only in case) that doesn't
    read as a trace; the copies that do read are merged without them."""


class ScanCreate(BaseModel):
    """Rule scanners to run over the console's traces."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    scanners: list[RuleScanner] = Field(min_length=1)
    """What ``deepgym traces scan --config`` reads: ``pattern`` and ``protected_paths``."""

    trace_ids: list[Name] | None = Field(default=None, min_length=1)
    """The traces to scan, each read once; None scans every one."""

    write_labels: bool = False
    """Store a label per finding and per scan error, all in one transaction."""

    run_id: Name | None = None
    """The report's ``run_id``, so the labels' ids and ``group_id``; a retry with the same
    one stores nothing new. Generated when None."""


class ScanResult(ScanReport, StoredProblems):
    """A scan report, and the stored traces left out of it because they don't read as one."""


def _dataset_folder(folder: str) -> str:
    """Refuse a dataset folder name that isn't one visible folder in the datasets directory.

    Raises:
        ValueError: If it holds ``/``, ``\\`` or ``:``, starts with a dot, or is longer than
            :data:`~deepgym.dataset.manifest.MAX_FOLDER_BYTES` in UTF-8.
    """
    if any(char in folder for char in '/\\:') or folder.startswith('.'):
        raise ValueError(
            f'{folder!r} is not a folder name in the datasets directory: it may not hold '
            '"/", "\\" or ":", or start with "."'
        )
    if (size := len(folder.encode())) > MAX_FOLDER_BYTES:
        raise ValueError(
            f'the folder name is {size} bytes in UTF-8; at most {MAX_FOLDER_BYTES} fit'
        )
    return folder


DatasetFolder = Annotated[Name, AfterValidator(_dataset_folder)]
"""A dataset's folder in the console's datasets directory: ``<name>-<version>``."""


class DatasetBuild(BaseModel):
    """A dataset to build from the console's traces and labels."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    spec: DatasetSpec
    name: Name
    version: Name
    formats: tuple[ExportFormat, ...] = Field(default=('jsonl',), min_length=1)

    @property
    def folder(self) -> str:
        """The folder it is written to in the datasets directory."""
        return f'{self.name}-{self.version}'

    @model_validator(mode='after')
    def check_folder(self) -> DatasetBuild:
        """Refuse a name and version that would leave the datasets directory."""
        _dataset_folder(self.folder)
        return self


class DatasetBuilt(StoredProblems):
    """A dataset written into the datasets directory."""

    dataset: str
    """Its folder, ``<name>-<version>``."""

    sha256: str
    """SHA-256 of its ``manifest.json``: the identifier to publish."""

    counts: DatasetCounts


class DatasetListing(BaseModel):
    """A folder of the datasets directory, with its manifest as read (not verified)."""

    dataset: str
    sha256: str | None
    """SHA-256 of its ``manifest.json``; None without one."""

    manifest: DatasetManifest | None
    error: str | None = None
    """Why there is no manifest: none, or one that doesn't read as a manifest."""


class DatasetPage(BaseModel):
    """One page of the datasets directory, by folder name."""

    datasets: list[DatasetListing]
    total: int
    """Folders across every page."""


class DatasetVerify(BaseModel):
    """A dataset of the datasets directory to check against its manifest."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    dataset: DatasetFolder
    sha256: str | None = Field(default=None, pattern=r'^[0-9a-fA-F]{64}$')
    """The SHA-256 it was published with; without one, only self-consistency is checked."""


class DatasetVerification(BaseModel):
    """What :func:`deepgym.dataset.verify_dataset` found."""

    dataset: str
    valid: bool
    sha256: str | None
    """SHA-256 of its ``manifest.json`` as it is now; None without one."""

    manifest: DatasetManifest | None
    """The manifest, when the folder matches it."""

    problems: list[str]
    """Every problem found; empty when valid."""
