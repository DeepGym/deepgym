"""Request and response models for the console API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from deepgym.models import Action, RunResult, Trajectory
from deepgym.sandboxes.cli import RunningSandbox

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
