"""Request and response schemas for the DeepGym API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from deepgym import __version__
from deepgym.adversarial import AttackStrategy
from deepgym.benchmark_ops import BenchmarkAuditReport
from deepgym.computer_use import ComputerUseEnvironment
from deepgym.models import Environment, MultiTurnEnvironment, Observation, VerifierMode
from deepgym.reward_qa import VerifierAuditReport

# ---------------------------------------------------------------------------
# API-safe environment model (no filesystem paths)
# ---------------------------------------------------------------------------


class APIEnvironment(BaseModel):
    """Environment specification for API requests. No filesystem paths allowed.

    Unlike :class:`~deepgym.models.Environment`, this model does not accept
    ``verifier_path``, ``snapshot`` or ``image`` fields, preventing remote callers from
    reading arbitrary files on the server or choosing what the sandbox boots.
    """

    task: str = Field(..., max_length=100_000)
    verifier_code: str = Field(..., max_length=500_000)
    language: str = Field(default='python', max_length=50)
    timeout: int = Field(default=30, ge=1, le=3600)
    case_timeout: int = Field(default=10, ge=1, le=3600)
    verifier_mode: VerifierMode = 'auto'
    support_files: dict[str, str] = Field(default_factory=dict, max_length=50)
    difficulty: Literal['easy', 'medium', 'hard'] = 'medium'
    domain: str = Field(default='coding', max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=20)
    test_cases: list[dict] | None = Field(default=None, max_length=100)
    env_vars: dict[str, str] | None = Field(default=None, max_length=50)

    @field_validator('env_vars')
    @classmethod
    def validate_env_var_sizes(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        """Enforce key and value length limits on env vars."""
        if v is None:
            return v
        for key, value in v.items():
            if len(key) > 100:
                raise ValueError(f'Env var key too long: {len(key)} chars')
            if len(value) > 10_000:
                raise ValueError(f'Env var value for {key!r} too long')
        return v

    @model_validator(mode='after')
    def check_environment(self) -> APIEnvironment:
        """Reject what Environment would (bad support file names, ...) as a 422, before any work."""
        self.to_environment()
        return self

    def to_environment(self) -> Environment:
        """Convert to an internal Environment model.

        Returns:
            An Environment instance safe for server-side execution.
        """
        return Environment(
            task=self.task,
            verifier_code=self.verifier_code,
            language=self.language,
            timeout=self.timeout,
            case_timeout=self.case_timeout,
            verifier_mode=self.verifier_mode,
            support_files=self.support_files,
            difficulty=self.difficulty,
            domain=self.domain,
            tags=self.tags,
            test_cases=self.test_cases,
            env_vars=self.env_vars,
        )


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Request body for a single episode run."""

    environment: APIEnvironment
    model_output: str = Field(
        ..., max_length=500_000, description='Model-generated solution source code.'
    )


class RegisteredRunRequest(BaseModel):
    """Request body for running against a previously registered environment."""

    model_output: str = Field(
        ..., max_length=500_000, description='Model-generated solution source code.'
    )


class BatchRunRequest(BaseModel):
    """Request body for running multiple solutions against one environment."""

    environment: APIEnvironment
    outputs: list[str] = Field(
        ..., max_length=1000, description='List of model-generated solutions.'
    )
    max_parallel: int = Field(
        default=10, ge=1, le=100, description='Maximum concurrent sandbox executions.'
    )

    @field_validator('outputs')
    @classmethod
    def _cap_output_length(cls, v: list[str]) -> list[str]:
        """Enforce per-item length limit on model outputs."""
        for i, item in enumerate(v):
            if len(item) > 500_000:
                raise ValueError(f'outputs[{i}] exceeds 500,000 character limit (got {len(item)})')
        return v


class EvalRequest(BaseModel):
    """Request body for running a full evaluation suite."""

    suite: str = Field(
        ...,
        description=(
            "Suite name: 'easy', 'medium', 'hard' (difficulty), "
            "'coding', 'computer-use', 'tool-use' (type), 'all', "
            "or a family name like 'dynamic-programming'."
        ),
    )
    model_outputs: dict[str, str] = Field(
        ...,
        max_length=500,
        description='Mapping of task id to model-generated solution source code.',
    )
    max_parallel: int = Field(
        default=100, ge=1, le=100, description='Maximum concurrent sandbox executions.'
    )

    @field_validator('model_outputs')
    @classmethod
    def validate_model_output_sizes(cls, v: dict[str, str]) -> dict[str, str]:
        """Enforce per-value length limit on model outputs."""
        for key, value in v.items():
            if len(value) > 500_000:
                raise ValueError(f'Model output for {key!r} exceeds 500,000 char limit')
        return v


class CreateSnapshotRequest(BaseModel):
    """Request body for creating a pre-built sandbox snapshot."""

    name: str = Field(..., max_length=200, description='Snapshot name.')
    image: str = Field(default='python:3.12-slim', max_length=200, description='Base Docker image.')
    packages: list[str] = Field(
        default_factory=list, max_length=50, description='Packages to pre-install.'
    )


class VerifierAuditRequest(BaseModel):
    """Request body for auditing an inline environment verifier."""

    environment: APIEnvironment
    verifier_id: str | None = Field(default=None, max_length=200)
    benchmark: str = Field(default='custom', max_length=200)
    strategies: list[AttackStrategy] | None = Field(default=None, max_length=20)
    persist: bool = False


class RegisteredVerifierAuditRequest(BaseModel):
    """Request body for auditing a registered environment verifier."""

    verifier_id: str | None = Field(default=None, max_length=200)
    benchmark: str = Field(default='registered', max_length=200)
    strategies: list[AttackStrategy] | None = Field(default=None, max_length=20)
    persist: bool = False


class BenchmarkAuditRequest(BaseModel):
    """Request body for auditing benchmark split leakage."""

    environment_ids: list[str] = Field(..., min_length=1, max_length=10_000)
    benchmark: str = Field(default='registered', max_length=200)
    seed: int = 0
    public_eval_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    holdout_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    canary_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    split_overrides: (
        dict[str, Literal['public_train', 'public_eval', 'private_holdout', 'canary']] | None
    ) = None


class CreateSessionRequest(BaseModel):
    """Request body for POST /v1/sessions."""

    environment: str | MultiTurnEnvironment | ComputerUseEnvironment = Field(
        union_mode='left_to_right'
    )
    """Registry name (e.g. 'signup_form') or an inline spec; no 'type' means multi-turn."""

    variant: int = Field(default=0, ge=0)
    """Parameter set to load for a registry computer-use env (ignored for inline specs)."""

    @field_validator('environment')
    @classmethod
    def reject_image(
        cls, v: str | MultiTurnEnvironment | ComputerUseEnvironment
    ) -> str | MultiTurnEnvironment | ComputerUseEnvironment:
        """Refuse a caller-chosen image: root in the sandbox runs the image's binaries."""
        if not isinstance(v, str) and v.image is not None:
            raise ValueError('image cannot be set over the API')
        return v


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str = 'ok'
    version: str = __version__
    leaked_sandboxes: int = 0
    """Sandboxes whose delete kept failing and that cleanup gave up on (the newest 1000)."""


class CreateEnvironmentResponse(BaseModel):
    """Response for the POST /v1/environments endpoint."""

    id: str
    created: bool


class CapabilitiesResponse(BaseModel):
    """Response describing server capabilities exposed to remote runtimes."""

    version: str = __version__
    sync_run: bool = True
    batch_run: bool = True
    async_jobs: bool = True
    registered_environments: bool = True
    named_environment_run: bool = True
    verifier_audit: bool = True
    benchmark_audit: bool = True
    sessions: bool = True


class SessionResponse(BaseModel):
    """Response for POST /v1/sessions: the live session and its first observation."""

    session_id: str
    sandbox_id: str
    observation: Observation


class StepResponse(BaseModel):
    """Response for POST /v1/sessions/{id}/step."""

    observation: Observation
    reward: float
    done: bool


class CreateSnapshotResponse(BaseModel):
    """Response for the POST /v1/snapshots endpoint."""

    name: str
    created: bool


# ---------------------------------------------------------------------------
# Job response schemas (async endpoints)
# ---------------------------------------------------------------------------

# Job and BatchJob are used directly as response models; re-export here for
# convenience so the routes module can import everything from schemas.
from deepgym.models import BatchJob as BatchJobResponse  # noqa: E402
from deepgym.models import Job as JobResponse  # noqa: E402

__all__ = [  # noqa: F822
    'APIEnvironment',
    'BenchmarkAuditReport',
    'BenchmarkAuditRequest',
    'RunRequest',
    'RegisteredRunRequest',
    'BatchRunRequest',
    'EvalRequest',
    'CreateSessionRequest',
    'CreateSnapshotRequest',
    'VerifierAuditRequest',
    'RegisteredVerifierAuditRequest',
    'HealthResponse',
    'CapabilitiesResponse',
    'CreateEnvironmentResponse',
    'CreateSnapshotResponse',
    'SessionResponse',
    'StepResponse',
    'VerifierAuditReport',
    'JobResponse',
    'BatchJobResponse',
]
