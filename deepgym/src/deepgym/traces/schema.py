"""Versioned trace schema: one graded agent episode with its provenance.

A :class:`Trace` is what an episode leaves behind for training: where it ran
(environment, verifier and image fingerprints, deepgym version and commit, provider),
who acted (agent, model and sampling config, as the caller describes them), every step
(the action, the model messages behind it, what came back, the step reward and its
verifier output) and every grade, the last of them the verdict.

Schema 1.1 added the deepgym commit, the image digest the provider resolved (or why
there is none) and every grade but the last. Traces written as 1.0 still load, with
those fields empty.
"""

from __future__ import annotations

import functools
import hashlib
import json
import operator
import os
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    computed_field,
    model_validator,
)
from pydantic_core import to_json

from deepgym.models import Action, MultiTurnEnvironment, Observation, RunResult, Trajectory
from deepgym.sandboxes import SandboxImage

TRACE_SCHEMA_VERSION = '1.1'

NOT_RESOLVED = 'not resolved: no sandbox reported which image it booted'
UNFINISHED = 'trace ended before the episode was closed (process crashed or was killed?)'

SamplingValue = str | int | FiniteFloat | bool | None


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def json_text(value: object, *, indent: int | None = None) -> str:
    """Serialize *value*, models included, as strict JSON: NaN and infinities become null.

    Verifier output can carry NaN, which ``json.dumps`` writes as the invalid token ``NaN``.
    """
    return to_json(value, indent=indent, inf_nan_mode='null').decode()


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


def git_checkout(package: Path, since: float | None = None) -> tuple[str | None, bool | None]:
    """Return the git commit *package* is checked out at, and whether its files differ from it.

    The package must be tracked by the repository around it, so an install inside some
    other project's checkout (a virtualenv in it, say) never reports that project's commit.
    ``GIT_*`` variables are dropped, so they can't point git at another repository, and
    git takes no optional locks, so it never blocks a commit running at the same time.

    Args:
        package: The package folder.
        since: A time; a tracked file modified after it also counts as a difference.

    Returns:
        ``(commit, dirty)``; ``(None, None)`` when *package* isn't tracked in a git
        checkout, or git can't tell.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ['git', '--no-optional-locks', '-C', str(package), *args],
                capture_output=True, text=True, timeout=10, env=env, check=False,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    tracked = git('ls-files', '-z', '--', '.')
    if not tracked:
        return None, None
    head = git('rev-parse', '--verify', 'HEAD')
    status = git('status', '--porcelain', '--untracked-files=normal', '--', '.')
    if head is None or status is None:
        return None, None
    names = filter(None, tracked.split('\0'))
    touched = since is not None and any(_modified_after(package / name, since) for name in names)
    return head.strip(), bool(status.strip()) or touched


def _modified_after(path: Path, since: float) -> bool:
    try:
        return path.lstat().st_mtime > since
    except OSError:
        return True


_PACKAGE = Path(__file__).resolve().parents[1]
_IMPORTED_AT = time.time()


def deepgym_checkout() -> tuple[str | None, bool | None]:
    """Return the commit of the deepgym checkout this code runs from, and whether it's dirty.

    Dirty means deepgym's files differ from the commit, or changed after deepgym was
    imported: code loaded before that keeps running, so they may no longer show it.

    Returns:
        ``(commit, dirty)``, or ``(None, None)`` when not running from a git checkout.
    """
    return git_checkout(_PACKAGE, since=_IMPORTED_AT)


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
    """The image the sandbox booted (its tag, snapshot or template), as the provider names it."""

    image_digest: str | None = None
    """Content digest (``sha256:...``) of that image, as the provider resolved it; before
    the sandbox boots (:meth:`of`), the digest the environment's image reference pins."""

    digest_unavailable: str | None = None
    """Why there is no ``image_digest``; traces before schema 1.1 have neither."""

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
            The environment's fingerprint. Its image is the environment's own until
            :meth:`booted` says what the sandbox booted.
        """
        image = env.image
        digest = image.rpartition('@')[2] if image and '@' in image else None
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
            image_digest=digest,
            digest_unavailable=None if digest else NOT_RESOLVED,
            snapshot=env.snapshot,
        )

    def booted(self, image: SandboxImage) -> EnvFingerprint:
        """Return this fingerprint with the image its sandbox booted, as the provider reports it.

        Only the provider can tell a digest is what booted: an image it built from a
        pinned reference is another image, so the pinned digest never stands in.
        """
        unavailable = None if image.digest else image.unavailable or NOT_RESOLVED
        return self.model_copy(
            update={
                'image': image.name,
                'image_digest': image.digest,
                'digest_unavailable': unavailable,
            }
        )


class Provenance(BaseModel):
    """Where, how and by whom an episode was produced."""

    model_config = ConfigDict(frozen=True)

    env: EnvFingerprint
    agent: AgentMeta
    provider: str
    """Where the episode ran: 'local', 'daytona', 'docker', ..."""

    deepgym_version: str
    deepgym_commit: str | None = None
    """Git commit of the deepgym checkout that ran the episode (see :func:`deepgym_checkout`);
    None when it ran from an installed package, and in traces before schema 1.1."""

    deepgym_dirty: bool | None = None
    """Whether deepgym's files in that checkout differed from the commit, or had changed
    since deepgym was imported; None without a commit."""

    started_at: AwareDatetime


class TraceMessage(BaseModel):
    """One model message, optionally with the token ids and logprobs the model produced."""

    model_config = ConfigDict(frozen=True)

    role: Literal['system', 'user', 'assistant', 'tool']
    content: str
    token_ids: tuple[int, ...] | None = None
    logprobs: tuple[FiniteFloat, ...] | None = None

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
    action: the prompt on the first step, the rendered observations and reply after that.
    Concatenated over all steps they are the whole conversation. Since schema 1.1, empty
    on the second and later actions of one reply, which continue its turn (see
    :attr:`Trace.turns`); also empty when not supplied."""

    observation: Observation | None = None
    """What the action produced; None if it failed before producing anything."""

    applied: bool = True
    """Whether the session recorded the action as taken; False if it was rejected, or
    failed before the session recorded it."""

    reward: float | None = Field(default=None, ge=0.0, le=1.0)
    """Step verifier score; None without a step verifier, or when it failed or reported
    its own failure (see :func:`deepgym.models.is_verdict`)."""

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

    schema_version: Literal['1.0', '1.1'] = TRACE_SCHEMA_VERSION
    trace_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    task: str
    provenance: Provenance
    initial_observation: Observation | None = None
    steps: tuple[TraceStep, ...] = ()
    result: RunResult | None = None
    """The final verifier's last result, the verdict; None if the episode was never scored."""

    prior_results: tuple[RunResult, ...] = ()
    """Earlier results of the final verifier, oldest first, each replaced by a later grade
    of the same final state: the episode's grades are ``(*prior_results, result)``."""

    error: str | None = None
    """What ended the episode early, if anything did."""

    finished_at: AwareDatetime | None = None

    @model_validator(mode='after')
    def check_step_order(self) -> Trace:
        """Reject steps that aren't numbered 1, 2, 3, ... in order."""
        if [step.index for step in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError('step indexes must run 1, 2, 3, ... in order')
        return self

    @model_validator(mode='after')
    def check_grades(self) -> Trace:
        """Reject earlier grades without a last one."""
        if self.prior_results and self.result is None:
            raise ValueError('prior_results need a final result')
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def turns(self) -> list[list[int]]:
        """Step indexes by model turn: the actions one assistant reply produced, in order.

        Since schema 1.1 a step without messages belongs to the turn before it. One before
        any model messages, and any step of a 1.0 trace (where no messages meant none were
        recorded), is a group of its own.
        """
        turns: list[list[int]] = []
        for step in self.steps:
            continues = (
                self.schema_version != '1.0'
                and turns
                and not step.messages
                and self.steps[turns[-1][0] - 1].messages
            )
            if continues:
                turns[-1].append(step.index)
            else:
                turns.append([step.index])
        return turns

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

        Steps that failed before producing an observation are left out. Rewards match a
        session's own trajectory: each step counts 0 when the environment has no step
        verifier, and a step whose step verifier failed has no step reward at all, so
        nothing is appended for it (it ended the episode without a verdict). Rewards stay
        on their own steps: older traces could go on past such a step, and their later
        rewards are left out rather than shifted onto earlier actions.
        """
        applied = [step for step in self.steps if step.observation is not None]
        if self.initial_observation is None or not applied:
            return Trajectory(final_observation=self.initial_observation)
        shown = [self.initial_observation, *(step.observation for step in applied)]
        if self.provenance.env.step_verifier_sha256 is None:
            rewards = [0.0] * len(applied)
        else:
            rewards = []
            for step in applied:
                if step.reward is None:
                    break
                rewards.append(step.reward)
        return Trajectory(
            steps=list(zip(shown, (step.action for step in applied))),
            final_observation=shown[-1],
            # The session's running total: sum() rounds differently from Python 3.12 on.
            total_reward=functools.reduce(operator.add, rewards, 0.0),
            step_rewards=rewards,
        )


def check_unique_ids(traces: Sequence[Trace]) -> None:
    """Refuse traces that share an id: an export would keep one and silently drop the rest.

    Ids are compared ignoring case, because ATIF names files after them and macOS and
    Windows file systems ignore case.

    Raises:
        ValueError: If two traces have the same ``trace_id``, ignoring case.
    """
    ids = Counter(trace.trace_id.casefold() for trace in traces)
    repeated = [key for key, count in ids.items() if count > 1]
    if repeated:
        raise ValueError(f'Traces share ids {repeated}; keep one copy of each')


def merge_copies(traces: Sequence[Trace]) -> list[Trace]:
    """Keep one copy of each trace, the one that got furthest, if its copies agree.

    A trace can be stored twice, e.g. in the console mid-episode and on disk once it
    ended. Copies agree when each is the furthest one cut short: the same start, steps
    and grades that begin the furthest one's, and, once finished, nothing different.

    Raises:
        ValueError: If copies of a trace disagree: keeping either would drop the other.
    """
    copies: dict[str, list[Trace]] = {}
    for trace in traces:
        copies.setdefault(trace.trace_id, []).append(trace)
    kept: list[Trace] = []
    conflicts: list[str] = []
    for trace_id, group in copies.items():
        furthest = max(group, key=_progress)
        if all(_cut_short(copy, furthest) for copy in group):
            kept.append(furthest)
        else:
            conflicts.append(trace_id)
    if conflicts:
        raise ValueError(f'Copies of traces {conflicts} disagree; keep the right copy of each')
    return kept


def _grades(trace: Trace) -> tuple[RunResult, ...]:
    return () if trace.result is None else (*trace.prior_results, trace.result)


def _progress(trace: Trace) -> tuple[int, int, bool, bool, int]:
    """How far a copy got: steps, grades, whether it started and finished, then what it says
    stopped it (a real error over the loader's guess that the process died)."""
    error = trace.error
    return (
        len(trace.steps),
        len(_grades(trace)),
        trace.initial_observation is not None,
        trace.finished_at is not None,
        0 if error is None else 1 if error == UNFINISHED else 2,
    )


def _cut_short(copy: Trace, furthest: Trace) -> bool:
    """Whether *copy* is *furthest* recorded earlier, or the same trace.

    A traced episode takes no action once scored, so a graded copy has every step, and it
    never clears an error, so one the copy records stays (or a later one replaces it).
    """
    if copy.schema_version == '1.0' and copy.result is not None and not copy.prior_results:
        # Schema 1.0 kept only the last grade in memory, so a console copy can lack the
        # earlier grades its event log has: take those from the furthest copy.
        history = _grades(furthest)
        if copy.result not in history:
            return False
        finished = copy.finished_at is not None
        position = len(history) - 1 if finished else history.index(copy.result)
        copy = copy.model_copy(update={'prior_results': history[:position]})
    if copy.finished_at is not None:
        return copy == furthest
    if copy.error != furthest.error and _progress(copy) == _progress(furthest):
        return False  # neither got further, so neither error is known to be the later one
    grades = _grades(copy)
    steps = furthest.steps if grades else furthest.steps[: len(copy.steps)]
    erred = copy.error not in (None, UNFINISHED)
    return (
        (copy.schema_version, copy.task, copy.provenance)
        == (furthest.schema_version, furthest.task, furthest.provenance)
        and copy.initial_observation in (None, furthest.initial_observation)
        and copy.steps == steps
        and grades == _grades(furthest)[: len(grades)]
        and not (erred and furthest.error is None)
    )
