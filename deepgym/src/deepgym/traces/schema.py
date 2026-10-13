"""Versioned trace schema: one graded agent episode with its provenance.

A :class:`Trace` is what an episode leaves behind for training: where it ran
(environment, verifier and image fingerprints, deepgym version, provider), who acted
(agent, model and sampling config, as the caller describes them), every step (the
action, the model messages behind it, what came back, the step reward and its
verifier output) and the final verdict.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
    model_validator,
)

from deepgym.models import Action, MultiTurnEnvironment, Observation, RunResult, Trajectory

TRACE_SCHEMA_VERSION = '1.0'

SamplingValue = str | int | float | bool | None


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def environment_sha256(env: MultiTurnEnvironment) -> str:
    """Fingerprint everything that defines *env*: task, files, setup, verifiers, limits.

    Fields left at their defaults are omitted, so adding a new field with a default
    doesn't change existing fingerprints; ``type`` is always kept, because subclasses
    have different defaults. Keys and the action set are sorted, so the hash doesn't
    depend on dict order or string hash seeds.
    """
    data = env.model_dump(mode='json', exclude_defaults=True) | {'type': env.type}
    if 'allowed_actions' in data:
        data['allowed_actions'] = sorted(data['allowed_actions'])
    return sha256_text(json.dumps(data, sort_keys=True, separators=(',', ':')))


class AgentMeta(BaseModel):
    """Who acted in an episode, as the caller describes it."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    """Agent system, e.g. 'react-shell'."""

    version: str | None = None
    """Agent system version."""

    model: str | None = None
    """Model id the agent called, e.g. 'Qwen/Qwen3-8B'."""

    sampling: dict[str, SamplingValue] = Field(default_factory=dict)
    """Sampling config, e.g. {'temperature': 0.7, 'top_p': 0.95, 'max_tokens': 2048}."""


class EnvFingerprint(BaseModel):
    """Which environment an episode ran, by name and by content hash."""

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    variant: int | None = None
    """Registry variant (``load_environment(name, variant=...)``), if any."""

    seed: int | None = None
    """Episode seed supplied by the caller, if any."""

    environment_sha256: str
    """Hash of the whole environment definition (see :func:`environment_sha256`)."""

    task_sha256: str
    verifier_sha256: str
    """Hash of the final verifier's source."""

    step_verifier_sha256: str | None = None
    reference_sha256: str | None = None
    """Hash of the reference solution the caller supplied, if any."""

    image: str | None = None
    image_digest: str | None = None
    """Content digest of the image, known when the image reference pins one (``@sha256:``)."""

    snapshot: str | None = None

    @classmethod
    def of(
        cls,
        env: MultiTurnEnvironment,
        *,
        variant: int | None = None,
        seed: int | None = None,
        reference: str | None = None,
    ) -> EnvFingerprint:
        """Fingerprint *env*.

        Args:
            env: The environment the episode runs.
            variant: Registry variant it was loaded with.
            seed: Episode seed, if the caller uses one.
            reference: Reference solution source, hashed when given.

        Returns:
            The environment's fingerprint.
        """
        image = env.image
        return cls(
            name=env.name,
            variant=variant,
            seed=seed,
            environment_sha256=environment_sha256(env),
            task_sha256=sha256_text(env.task),
            verifier_sha256=sha256_text(env.final_verifier_code),
            step_verifier_sha256=sha256_text(env.step_verifier_code)
            if env.step_verifier_code
            else None,
            reference_sha256=None if reference is None else sha256_text(reference),
            image=image,
            image_digest=image.rpartition('@')[2] if image and '@' in image else None,
            snapshot=env.snapshot,
        )


class Provenance(BaseModel):
    """Where, how and by whom an episode was produced."""

    model_config = ConfigDict(frozen=True)

    env: EnvFingerprint
    agent: AgentMeta
    provider: str
    """Where the episode ran: 'local', 'daytona', 'docker', ..."""

    deepgym_version: str
    started_at: AwareDatetime


class TraceMessage(BaseModel):
    """One model message, optionally with the token ids and logprobs the model produced."""

    model_config = ConfigDict(frozen=True)

    role: Literal['system', 'user', 'assistant', 'tool']
    content: str
    token_ids: tuple[int, ...] | None = None
    logprobs: tuple[float, ...] | None = None

    @model_validator(mode='after')
    def check_token_alignment(self) -> TraceMessage:
        """Reject token ids and logprobs of different lengths."""
        if (
            self.token_ids is not None
            and self.logprobs is not None
            and len(self.token_ids) != len(self.logprobs)
        ):
            raise ValueError('token_ids and logprobs must have the same length')
        return self


class TraceStep(BaseModel):
    """One agent action and what it produced."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=1)
    """1-based position in the trace, counting failed attempts."""

    action: Action
    messages: tuple[TraceMessage, ...] = ()
    """Model messages new at this step, ending with the assistant reply that produced the
    action: the prompt on the first step, the rendered observation and reply after that.
    Concatenated over all steps they are the whole conversation. Empty when not supplied."""

    observation: Observation | None = None
    """What the action produced; None if it failed before producing anything."""

    reward: float | None = None
    """Step verifier score; None when the environment has no step verifier or it failed."""

    step_verifier: dict[str, JsonValue] | None = None
    """The step verifier's parsed output (score, passed, details, ...)."""

    error: str | None = None
    """Why the step failed (rejected action, step verifier or sandbox error)."""

    timestamp: AwareDatetime
    duration_ms: float = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exit_code(self) -> int | None:
        """Exit code of a code or bash action."""
        if self.observation is None or not self.observation.metadata:
            return None
        return self.observation.metadata.get('exit_code')


class Trace(BaseModel):
    """A recorded episode: provenance, steps and the final verdict."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal['1.0'] = TRACE_SCHEMA_VERSION
    trace_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    task: str
    provenance: Provenance
    initial_observation: Observation | None = None
    steps: tuple[TraceStep, ...] = ()
    result: RunResult | None = None
    """The final verifier's result; None if the episode was never scored."""

    error: str | None = None
    """What ended the episode early, if anything did."""

    finished_at: AwareDatetime | None = None

    @model_validator(mode='after')
    def check_step_order(self) -> Trace:
        """Reject steps that aren't numbered 1, 2, 3, ... in order."""
        if [step.index for step in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError('step indexes must run 1, 2, 3, ... in order')
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> Literal['complete', 'incomplete']:
        """'complete' once the final verifier scored the episode, else 'incomplete'."""
        return 'incomplete' if self.result is None else 'complete'

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_verdict(self) -> bool:
        """Whether the final verifier judged the episode (see ``RunResult.has_verdict``)."""
        return self.result is not None and self.result.has_verdict

    def to_trajectory(self) -> Trajectory:
        """Return the episode as a session :class:`Trajectory`, for console replay.

        Steps that failed before producing an observation are left out, and steps
        without a step reward count 0, as in a session's own trajectory.
        """
        applied = [step for step in self.steps if step.observation is not None]
        if self.initial_observation is None or not applied:
            return Trajectory(final_observation=self.initial_observation)
        shown = [self.initial_observation, *(step.observation for step in applied)]
        rewards = [step.reward or 0.0 for step in applied]
        return Trajectory(
            steps=list(zip(shown, (step.action for step in applied))),
            final_observation=shown[-1],
            total_reward=sum(rewards),
            step_rewards=rewards,
        )
