"""Request and response schemas for the DeepGym API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from deepgym.models import Environment

# ---------------------------------------------------------------------------
# API-safe environment model (no filesystem paths)
# ---------------------------------------------------------------------------


class APIEnvironment(BaseModel):
    """Environment specification for API requests. No filesystem paths allowed.

    Unlike :class:`~deepgym.models.Environment`, this model does not accept
    ``verifier_path`` or ``snapshot`` fields, preventing remote callers from
    reading arbitrary files on the server or specifying Daytona snapshots.
    """

    task: str = Field(..., max_length=100_000)
    verifier_code: str = Field(..., max_length=500_000)
    language: str = Field(default='python', max_length=50)
    timeout: int = Field(default=30, ge=1, le=3600)
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


class CreateEnvironmentRequest(BaseModel):
    """Request body for creating/registering a new environment."""

    task: str = Field(..., max_length=100_000, description='Task description / prompt.')
    verifier_code: str = Field(
        ..., max_length=500_000, description='Python source code of the verifier script.'
    )
    language: str = Field(
        default='python', max_length=50, description='Programming language for the solution.'
    )
    timeout: int = Field(
        default=30, ge=1, le=3600, description='Maximum execution time in seconds.'
    )
    difficulty: Literal['easy', 'medium', 'hard'] = Field(
        default='medium', description='Task difficulty level.'
    )
    domain: str = Field(
        default='coding', max_length=100, description='Task domain (e.g. coding, math, reasoning).'
    )
    tags: list[str] = Field(
        default_factory=list, max_length=20, description='Freeform tags for filtering.'
    )
    test_cases: list[dict] | None = Field(
        default=None,
        max_length=100,
        description='Optional list of test-case dicts passed to the verifier.',
    )
    env_vars: dict[str, str] | None = Field(
        default=None, max_length=50, description='Environment variables to inject into the sandbox.'
    )

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


class CreateSnapshotRequest(BaseModel):
    """Request body for creating a pre-built sandbox snapshot."""

    name: str = Field(..., max_length=200, description='Snapshot name.')
    image: str = Field(default='python:3.12-slim', max_length=200, description='Base Docker image.')
    packages: list[str] = Field(
        default_factory=list, max_length=50, description='Packages to pre-install.'
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str = 'ok'
    version: str = '0.1.0'


class CreateEnvironmentResponse(BaseModel):
    """Response for the POST /v1/environments endpoint."""

    id: str
    created: bool


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
    'RunRequest',
    'BatchRunRequest',
    'EvalRequest',
    'CreateEnvironmentRequest',
    'CreateSnapshotRequest',
    'HealthResponse',
    'CreateEnvironmentResponse',
    'CreateSnapshotResponse',
    'JobResponse',
    'BatchJobResponse',
]
