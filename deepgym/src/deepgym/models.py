"""Pydantic data models for DeepGym."""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from deepgym.exceptions import InvalidScoreError
from deepgym.tools import HttpTool

logger = logging.getLogger(__name__)

# Float noise this far outside [0, 1] is clamped; anything further is a verifier bug.
SCORE_TOLERANCE = 1e-6

MODEL_FAULT_ERRORS = frozenset(
    {
        'invalid_patch',
        'patch_apply_failed',
        'test_infra_tampering',
        'policy_sandbox_failure',
        'solution_timeout',
        'solution_error',
    }
)
"""error_type values that are the model's own fault (a patch that is malformed, doesn't
apply, or touches test infrastructure; code it wrote that killed the grading harness once
its tests ran, or a sandbox it broke, hung or exhausted, see
:func:`~deepgym.exceptions.is_infrastructure_failure`; a solution that ran past its time
limit, or failed or broke the verifier any other way once its output was in play): a real 0
to learn from, not a missing verdict. A timeout on DeepGym's side is
``timeout``, which has no verdict."""


def is_verdict(exit_code: int | None, error_type: str | None) -> bool:
    """False when a verifier or sandbox failed instead of judging the solution.

    That is any *error_type* other than a model fault (see ``MODEL_FAULT_ERRORS``), or
    the verifier-error exit code 2.
    """
    return exit_code != 2 and (error_type is None or error_type in MODEL_FAULT_ERRORS)


def checked_score(value: object) -> float:
    """Return *value* as a reward in [0, 1], clamping only float noise.

    NaN, infinities, and scores clearly outside [0, 1] are verifier bugs; letting
    them through (or clamping them) would hand the policy free reward.

    Raises:
        InvalidScoreError: If *value* isn't a finite number within [0, 1].
    """
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise InvalidScoreError(f'Verifier returned a non-numeric score: {value!r}') from exc
    if not -SCORE_TOLERANCE <= score <= 1 + SCORE_TOLERANCE:  # also False for NaN
        logger.warning('Verifier returned invalid score %r; scoring 0', score)
        raise InvalidScoreError(f'Verifier returned invalid score {score!r}; must be in [0, 1]')
    return min(1.0, max(0.0, score))


def finite_components(components: dict[str, float] | None) -> dict[str, float] | None:
    """Return *components* if every one is finite.

    A NaN or infinite component is a verifier bug, like an invalid score: zeroing it
    would hand a trainer a fabricated 0 under an otherwise valid verdict.

    Raises:
        InvalidScoreError: If a component is NaN or infinite.
    """
    bad = [key for key, value in (components or {}).items() if not math.isfinite(value)]
    if bad:
        raise InvalidScoreError(f'Verifier returned non-finite reward components {bad}')
    return components


VerifierMode = Literal['auto', 'script', 'verify_function', 'body']

# Files the judge directory always contains; support files can't shadow them.
RESERVED_JUDGE_FILES = frozenset({'verifier.py', 'deepgym_solution.py', 'test_cases.json'})

REFERENCE_FILES = ('reference_solution.py', 'reference_solution.diff')
"""Where an env folder keeps the model output that must score 1.0 (a diff for repo patches)."""

REFERENCE_ACTIONS = 'reference_actions.json'
"""Where a multi-turn env folder keeps the scripted episode that must score 1.0: a JSON list
of actions, or for a computer-use env one such list per variant, in variants.json order."""


class Environment(BaseModel):
    """Define a task environment for RL evaluation.

    The environment specifies the task prompt, verifier logic, execution
    constraints, and optional pre-built sandbox snapshot.
    """

    name: str | None = None
    """Optional stable environment identifier used for routing and reporting."""

    task: str
    """The task description / prompt presented to the model."""

    type: Literal['coding', 'computer-use', 'tool-use'] = 'coding'
    """Environment type: coding, computer-use, or tool-use."""

    verifier_code: str = ''
    """Python source code of the verifier script."""

    verifier_path: Path | None = None
    """Optional filesystem path to a verifier script (alternative to inline code)."""

    @model_validator(mode='after')
    def check_verifier_source(self) -> Environment:
        """Ensure a verifier is given and support file names are safe."""
        if not self.verifier_code and self.verifier_path is None:
            raise ValueError('Either verifier_code or verifier_path must be provided')
        for name in self.support_files:
            rel = PurePosixPath(name)
            if not rel.parts or rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'support file {name!r} must be a relative path without ".."')
            if name in RESERVED_JUDGE_FILES:
                raise ValueError(f'support file {name!r} would replace a judge file')
        return self

    @property
    def verifier_source(self) -> str:
        """The verifier's source code, from verifier_path or verifier_code."""
        if self.verifier_path is not None:
            return self.verifier_path.read_text(encoding='utf-8')
        return self.verifier_code

    @property
    def reference_solution(self) -> str | None:
        """A model output that should score 1.0: the first of REFERENCE_FILES next to the
        verifier (reference_solution.py, or reference_solution.diff for a repo patch)."""
        if self.verifier_path is None:
            return None
        folder = self.verifier_path.parent
        path = next((folder / name for name in REFERENCE_FILES if (folder / name).is_file()), None)
        return path.read_text(encoding='utf-8') if path else None

    language: str = 'python'
    """Programming language for the solution."""

    timeout: int = Field(default=30, ge=1)
    """Total time budget for the verifier run, in seconds."""

    case_timeout: int = Field(default=10, ge=1)
    """Seconds one call into the solution may take; a slow case fails on its own."""

    verifier_mode: VerifierMode = 'auto'
    """How to run the verifier: 'script' (has a main block), 'verify_function'
    (defines verify()), 'body' (function body), or 'auto' to detect by parsing."""

    support_files: dict[str, str] = Field(default_factory=dict)
    """Extra files placed next to the verifier and importable by it (name -> content)."""

    difficulty: Literal['easy', 'medium', 'hard'] = 'medium'
    """Task difficulty level."""

    domain: str = 'coding'
    """Task domain (e.g. coding, math, reasoning)."""

    tags: list[str] = Field(default_factory=list)
    """Freeform tags for filtering and categorization."""

    test_cases: list[dict] | None = None
    """Optional list of test-case dicts passed to the verifier."""

    snapshot: str | None = None
    """Pre-built Daytona snapshot name with dependencies pre-installed."""

    image: str | None = None
    """Image to boot the sandbox from, e.g. 'python:3.11-slim': a registry image for docker,
    apple, modal and daytona (a snapshot wins), a template id for e2b. None uses the
    provider's default. It must run as root and have python3, bash, useradd and util-linux
    (setpriv, unshare, nsenter, mount), as Debian and Ubuntu images do."""

    env_vars: dict[str, str] | None = None
    """Environment variables to inject into the sandbox."""


class EnvMetadata(BaseModel):
    """Schema of an environment folder's metadata.json, as ``deepgym check`` validates it.

    Unknown keys are rejected so a typo ('dificulty') fails instead of silently
    falling back to a default.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    id: str | None = None
    name: str = Field(min_length=1)
    type: Literal['coding', 'computer-use', 'tool-use', 'multi-turn'] = 'coding'
    difficulty: Literal['easy', 'medium', 'hard'] = 'medium'
    domain: str = 'coding'
    family: str | None = None
    tags: list[str] = Field(default_factory=list)
    support_files: list[str] = Field(default_factory=list)
    """Files the verifier imports, relative to the folder (single-turn: placed next to it;
    multi-turn: prepended to it, since it runs where the agent can rewrite files)."""
    verifier_mode: VerifierMode = 'auto'
    timeout: int = Field(default=30, ge=1)
    case_timeout: int = Field(default=10, ge=1)
    image: str | None = None
    snapshot: str | None = None
    """Multi-turn and computer-use envs: the Daytona snapshot their sessions boot."""
    max_steps: int | None = Field(default=10, ge=1)  # None: no step limit
    timeout_per_step: int = Field(default=30, ge=1)
    repo_dir: str | None = None
    """Image-based repo patch envs (test_patch.diff): where the repo lives in the image."""
    test_command: str | None = None
    """Image-based repo patch envs: the command whose exit code scores the patch."""


class CaseResult(BaseModel):
    """Result of a single test case within a verifier.

    Enable fine-grained reward shaping for GRPO training by reporting
    which specific tests passed or failed, along with diagnostic summaries.
    """

    id: str = ''
    """Identifier for this test case (e.g. 'test_0', 'edge_case_empty')."""

    passed: bool
    """Whether this specific test case passed."""

    score: float = Field(default=1.0, ge=0.0, le=1.0)
    """Score for this test case (0.0-1.0). Defaults to 1.0 if passed, 0.0 if failed."""

    input_summary: str = ''
    """Brief summary of the test input (truncated for large inputs)."""

    expected_summary: str = ''
    """Brief summary of the expected output."""

    actual_summary: str = ''
    """Brief summary of the actual output produced."""

    error: str | None = None
    """Error message if the test case raised an exception."""

    execution_time_ms: float = 0.0
    """Execution time for this specific test case in milliseconds."""

    @field_validator('execution_time_ms')
    @classmethod
    def zero_non_finite_time(cls, value: float) -> float:
        """Zero a NaN or infinite timing, which JSON can't carry and would fail to reload."""
        return value if math.isfinite(value) else 0.0

    @model_validator(mode='before')
    @classmethod
    def check_score(cls, data: object) -> object:
        """Clamp float noise; reject a NaN, infinite or out-of-range score.

        Raises:
            InvalidScoreError: Never a 0: that would be a trainable reward the verifier
                didn't give, under an otherwise valid verdict.
        """
        if isinstance(data, dict) and 'score' in data:
            return {**data, 'score': checked_score(data['score'])}
        return data

    @model_validator(mode='after')
    def no_credit_for_unscored_failure(self) -> CaseResult:
        """Score a failed case 0 unless the verifier gave it an explicit partial score."""
        if not self.passed and 'score' not in self.model_fields_set:
            self.score = 0.0
        return self


class VerifierResult(BaseModel):
    """Represent structured output from a verifier script.

    Verifiers must print a JSON object to stdout conforming to this schema.
    """

    schema_version: str = '1.0'
    """Protocol version for forward compatibility."""

    score: float = Field(ge=0.0, le=1.0)
    """Numeric score between 0 and 1."""

    passed: bool
    """Whether the solution passed the verifier's acceptance criteria."""

    details: str | None = None
    """Optional human-readable explanation."""

    reward_components: dict[str, float] | None = None
    """Breakdown of shaped rewards (e.g. correctness, style, efficiency)."""

    @field_validator('reward_components')
    @classmethod
    def check_components(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        """Reject NaN/inf components before they reach a trainer."""
        return finite_components(value)

    metrics: dict[str, Any] | None = None
    """Execution metrics (e.g. execution_time, memory_used)."""

    seed: int | None = None
    """Random seed used, for reproducibility."""

    truncated: bool = False
    """Whether execution was cut short (e.g. timeout)."""

    error_type: str | None = None
    """Category of error if one occurred (timeout, runtime_error, import_error, etc.)."""

    cases: list[CaseResult] | None = None
    """Per-test-case breakdown. Enable fine-grained reward shaping for GRPO training."""


class RunResult(BaseModel):
    """Represent the result of a single environment run."""

    score: float = Field(ge=0, le=1)
    """Normalized score from the verifier."""

    passed: bool
    """Whether the solution passed."""

    output: str
    """Verifier details/summary string."""

    stderr: str
    """Captured stderr from verifier execution."""

    exit_code: int
    """Process exit code."""

    execution_time_ms: float
    """Wall-clock execution time in milliseconds."""

    sandbox_id: str
    """Identifier of the sandbox used for this run."""

    reward_components: dict[str, float] | None = None
    """Shaped reward breakdown passed through from the verifier."""

    @field_validator('reward_components')
    @classmethod
    def check_components(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        """Reject NaN/inf components before they reach a trainer."""
        return finite_components(value)

    metrics: dict[str, Any] | None = None
    """Execution metrics passed through from the verifier."""

    seed: int | None = None
    """Random seed used by the verifier, for reproducibility."""

    truncated: bool = False
    """Whether execution was cut short."""

    error_type: str | None = None
    """Category of error if one occurred."""

    cases: list[CaseResult] | None = None
    """Per-test-case breakdown passed through from the verifier."""

    @property
    def has_verdict(self) -> bool:
        """False when the verifier or sandbox failed instead of judging the solution.

        That is any ``error_type`` other than a model fault (see ``MODEL_FAULT_ERRORS``),
        or the verifier-error exit code 2. The score is then 0.0 but means nothing.
        """
        return is_verdict(self.exit_code, self.error_type)


class Observation(BaseModel):
    """Represent what the agent sees at each step."""

    content: str
    """Text content (stdout, page content, file listing, etc.)."""

    step: int
    """Current step number."""

    done: bool = False
    """Whether the episode is finished."""

    screenshot: str | None = None
    """Base64-encoded PNG of the desktop, at display resolution (desktop sessions only)."""

    metadata: dict[str, Any] | None = None
    """Optional structured metadata for the observation."""


ShellActionType = Literal['code', 'bash', 'write_file']
GuiActionType = Literal[
    'click',
    'double_click',
    'right_click',
    'move',
    'drag',
    'type',
    'key',
    'scroll',
    'screenshot',
    'wait',
]
ActionType = Literal[ShellActionType, GuiActionType, 'done', 'cli', 'tool']

GUI_ACTIONS: frozenset[GuiActionType] = frozenset(get_args(GuiActionType))

_POINTER_ACTIONS = frozenset({'click', 'double_click', 'right_click', 'move', 'scroll', 'drag'})
_TEXT_ACTIONS = frozenset({'type', 'key'})
_SCROLL_DIRECTIONS = frozenset({'up', 'down', 'left', 'right'})


class Action(BaseModel):
    """Represent an agent action.

    Shell actions (code, bash, write_file) run in the session work directory.
    GUI actions drive the desktop; coordinates are screenshot pixels.
    Tool actions call one of the environment's declared API tools (see :mod:`deepgym.tools`).
    """

    action_type: ActionType = 'code'
    """What to do. ``done`` ends the episode."""

    content: str = ''
    """Code, shell command, file contents, text to type, key chord ('ctrl+s'),
    or scroll direction (up/down/left/right)."""

    events: tuple[dict[str, JsonValue], ...] = ()
    """Nested CLI stdout events, recorded once after execution; never replayed as actions."""

    x: int | None = Field(default=None, ge=0)
    """Pointer x for click/move/scroll, or drag start."""

    y: int | None = Field(default=None, ge=0)
    """Pointer y for click/move/scroll, or drag start."""

    to_x: int | None = Field(default=None, ge=0)
    """Drag end x."""

    to_y: int | None = Field(default=None, ge=0)
    """Drag end y."""

    path: str | None = None
    """Target path for write_file, relative to the work directory."""

    amount: int = Field(default=1, ge=1, le=60)
    """Scroll clicks, or seconds for wait."""

    tool: str | None = None
    """Name of the declared tool a tool action calls."""

    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    """A tool action's arguments; the session checks them against the tool's routes."""

    @model_validator(mode='after')
    def check_arguments(self) -> Action:
        """Reject actions missing the arguments their type needs."""
        kind = self.action_type
        if kind in _POINTER_ACTIONS and (self.x is None or self.y is None):
            raise ValueError(f'{kind} needs x and y')
        if kind == 'drag' and (self.to_x is None or self.to_y is None):
            raise ValueError('drag needs to_x and to_y')
        if kind == 'scroll' and self.content not in _SCROLL_DIRECTIONS:
            raise ValueError(f'scroll content must be one of {sorted(_SCROLL_DIRECTIONS)}')
        if kind in _TEXT_ACTIONS and not self.content:
            raise ValueError(f'{kind} needs content')
        if kind == 'tool' and not self.tool:
            raise ValueError('tool needs the name of a declared tool')
        if kind == 'write_file':
            rel = PurePosixPath(self.path or '')
            if not rel.parts or rel.is_absolute() or '..' in rel.parts:
                raise ValueError('write_file needs a relative path without ".."')
        return self


class Trajectory(BaseModel):
    """Record of a complete multi-turn episode."""

    steps: list[tuple[Observation, Action]] = Field(default_factory=list)
    """Sequence of (observation, action) pairs."""

    final_observation: Observation | None = None
    """The last observation after the final action or termination."""

    total_reward: float = 0.0
    """Cumulative reward for the episode."""

    step_rewards: list[float] = Field(default_factory=list)
    """Per-step reward values. A step whose verifier gave no verdict ends the episode
    without one, so a failed episode has one reward fewer than steps."""


class VolumeMount(BaseModel):
    """One provider volume mounted at an absolute path inside the sandbox."""

    model_config = ConfigDict(frozen=True, extra='forbid')
    volume_id: str = Field(min_length=1)
    mount_path: str = Field(pattern=r'^/')
    subpath: str | None = None


class MultiTurnEnvironment(BaseModel):
    """Define a multi-turn environment where agents take multiple steps.

    One episode runs in one live sandbox: files are written, setup runs,
    the agent acts step by step, then the final verifier inspects the state.
    """

    name: str | None = None
    """Optional stable environment identifier used for routing and reporting."""

    task: str
    """The task description / prompt presented to the agent."""

    files: dict[str, str] = Field(default_factory=dict)
    """Text files written into the work directory before setup (relative path -> content)."""

    setup_script: str = ''
    """Bash run in the work directory after files are written (start servers, open apps)."""

    root_service_start: str = ''
    """Trusted image service startup, as root in the offline namespace before desktop/setup.
    Uses setup_timeout. Only SandboxBackend supports it; never accept this from an agent.
    The command must start the service itself and exit non-zero if readiness fails.
    """

    volumes: tuple[VolumeMount, ...] = ()
    """Provider volumes mounted into the sandbox (Daytona), for evidence too large for the image,
    such as attachment files. Trusted environment configuration only; never accept from an agent.
    A volume is shared and outlives each sandbox, so whatever one episode writes, the next reads:
    mount it where untrusted processes cannot write (e.g. a root-only path re-exposed through a
    read-only bind mount in root_service_start) and check its contents against a pinned manifest."""

    cli_provider_egress: bool = Field(default=False, strict=True)
    """Opt in to sandbox-level egress for the trusted CLI provider proxy (Daytona).
    Agent commands, root service startup and verification remain in the offline namespace.
    Other trusted root processes outside that namespace also gain network access.
    """

    setup_code: str = ''
    """Python run after setup_script. Its output is the first observation."""

    step_verifier_code: str = ''
    """Optional verifier run after each step (for intermediate rewards)."""

    final_verifier_code: str = ''
    """Verifier run at the end to compute final score."""

    max_steps: int | None = Field(default=10, ge=1)  # None: no step limit
    """Maximum number of steps before forced termination."""

    timeout_per_step: int = Field(default=30, ge=1)
    """Timeout per individual step in seconds."""

    setup_timeout: int = Field(default=120, ge=1)
    """Timeout for setup_script and setup_code each, in seconds (desktop apps boot slowly)."""

    allowed_actions: frozenset[ActionType] = frozenset({'code', 'bash', 'write_file', 'done'})
    """Action types the agent may use. Anything else is rejected."""

    tools: tuple[HttpTool, ...] = ()
    """API tools that tool actions call, run in the sandbox's offline namespace (see
    :mod:`deepgym.tools`). Trusted environment configuration only; never accept from an agent.
    Declaring any needs ``tool`` and ``done`` in allowed_actions: a tool-calling agent ends
    the episode only by an explicit ``done``."""

    desktop: bool = False
    """Start a desktop (Xvfb + xfce) and return screenshots. Needs the Daytona backend."""

    snapshot: str | None = None
    """Daytona snapshot to boot. None uses Daytona's default image (has xfce + Chromium)."""

    image: str | None = None
    """Image to boot the sandbox from; see :attr:`Environment.image`."""

    difficulty: Literal['easy', 'medium', 'hard'] = 'medium'
    """Task difficulty level."""

    domain: str = 'agentic'
    """Task domain (e.g. browser, desktop, terminal)."""

    tags: list[str] = Field(default_factory=list)
    """Freeform tags for filtering and categorization."""

    type: Literal['multi-turn'] = 'multi-turn'
    """Environment type discriminator."""

    reference_actions: list[Action] | None = Field(default=None, exclude=True)
    """A scripted episode that should score 1.0 (the folder's reference_actions.json), for
    QA. Never serialized, so it reaches neither the agent nor a stored env."""

    @model_validator(mode='after')
    def check_desktop_actions(self) -> MultiTurnEnvironment:
        """Reject GUI actions without a desktop and CLI egress without shell actions."""
        if not self.desktop and self.allowed_actions & GUI_ACTIONS:
            raise ValueError('GUI actions need desktop=True')
        if self.cli_provider_egress and not self.allowed_actions & {'bash', 'cli'}:
            raise ValueError('cli_provider_egress requires bash or cli in allowed_actions')
        return self

    @model_validator(mode='after')
    def check_tools(self) -> MultiTurnEnvironment:
        """Require unique tool names, and tools exactly when tool actions are allowed."""
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError(f'tool names must be unique, got {names}')
        if self.tools and not {'tool', 'done'} <= self.allowed_actions:
            raise ValueError('tools need tool and done in allowed_actions')
        if 'tool' in self.allowed_actions and not self.tools:
            raise ValueError('tool actions need declared tools')
        return self


class VerdictTotals(BaseModel):
    """Totals over only the runs that got a verdict (``RunResult.has_verdict``).

    They sit next to the headline totals, which count a run without a verdict as a failed
    0, so a run that breaks its verifier or sandbox can never raise the headline.
    """

    no_verdict_count: int = 0
    """Runs whose verifier or sandbox failed instead of judging them."""
    judged_count: int = 0
    """Runs with a verdict."""
    judged_pass_rate: float = 0.0
    """Fraction of the judged runs that passed; 0.0 if none was judged."""
    judged_mean_score: float = 0.0
    """Mean score of the judged runs; 0.0 if none was judged."""


class BatchResult(VerdictTotals):
    """Aggregate results from running multiple solutions against one environment."""

    results: list[RunResult]
    total: int
    passed: int
    """Runs judged as passed."""
    failed: int
    """Every other run: judged as failed, or without a verdict."""
    avg_score: float
    """Mean score over every run, a run without a verdict counting 0."""
    execution_time_ms: float
    """Total wall-clock time for the entire batch."""


class EvalResult(VerdictTotals):
    """Results from evaluating a model against an entire suite."""

    suite: str
    """Name of the evaluation suite."""

    model_name: str
    """Identifier for the model being evaluated."""

    pass_rate: float
    """Fraction of all tasks passed (0.0 - 1.0); a task without a verdict counts as failed."""

    results: list[RunResult]
    total: int
    passed: int
    """Tasks judged as passed."""
    avg_score: float
    """Mean score over every task, a task without a verdict counting 0."""

    env_names: list[str] | None = None
    """Environment of each entry in ``results``, in the same (suite) order."""


# ---------------------------------------------------------------------------
# Job models (async job execution)
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Status of an async job."""

    queued = 'queued'
    starting = 'starting'
    running = 'running'
    scoring = 'scoring'
    completed = 'completed'
    failed = 'failed'
    cancelled = 'cancelled'


class Job(BaseModel):
    """Represent an async single-run job."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: RunResult | None = None
    error: str | None = None


class BatchJob(BaseModel):
    """Represent an async batch-run job."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total: int = 0
    completed_count: int = 0
    result: BatchResult | None = None
    error: str | None = None
