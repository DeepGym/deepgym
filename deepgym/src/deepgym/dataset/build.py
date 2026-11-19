"""Choose a dataset's traces: filter, deduplicate and split them.

:func:`build_dataset` keeps the traces a :class:`DatasetSpec` asks for and records why it
dropped each other one.

Traces of one environment definition with the same task text and trajectory are copies of
one trajectory (see :class:`DuplicateGroup`), and labels about a trajectory reach its
copies. Of the copies in the population asked for (agents, models and environments) that
aren't invalidated, the current human ``passed`` labels are pooled into one verdict for
all copies, which overrides the verifier, and their labels count for ``require_labels``
on every copy. Humans who disagree make every copy ``contested``. A label ``forbid_labels``
rules out, on any copy, drops every copy. An invalidation stays on its own trace: it
discredits one run (its sandbox, setup or grade), not what the agent did, so another run
of the same trajectory keeps its own grade; so that a discredited run can't vouch for its
copies, its human verdict and labels reach none of them (its forbidden marks do).

A trace's verdict is the pooled human verdict when there is one, else its verifier's.
Pooled copies the verifier both passed and failed, with no human verdict, are all
``inconsistent``. Their verifier scores that stand (not reversed by a human) agree when
they are within :data:`deepgym.models.SCORE_TOLERANCE` of each other; when they don't, the
verdict still ships, but with no score, so a score filter drops it as ``no_score``.

The copy kept is, of those the other checks keep, one with a verdict, then one whose
verifier no human reversed, then one with a standing score, then the highest score, and
only then the earliest by start time and id. So the verdict, score and ``overridden`` that
ship never depend on which copy ran first; which of several equally judged copies ships
still follows start time. Labels count on ``trace`` targets, and a label on a console
episode (or one of its replay frames) counts on the trace the episode was recorded from.

Each dropped trace has one reason, the first check it fails, in this order:

* ``agent``, ``model``: not one of the agents or models asked for.
* ``env_name``, ``env_variant``: not one of the environments asked for.
* ``invalidated``: a current invalidation excludes it.
* ``contested``: the pooled human ``passed`` labels on its copies disagree.
* ``inconsistent``: the verifier passed some of its copies and failed others, and no
  human judged them.
* ``forbidden_label``: it or a copy has a current label ``forbid_labels`` rules out.
* ``incomplete`` (never scored) or ``no_verdict`` (its verifier or sandbox failed instead
  of judging it, and no human judged it): with ``require_verdict`` or a pass filter, since
  a trace without a verdict never counts as a fail.
* ``usage_incomplete``: some model call reported no usage, with ``require_usage_complete``.
* ``cost_unknown``, ``over_cost``: with ``max_cost_usd``, its cost is unknown or above it.
* ``no_score``, ``score``: with a score filter, it has no verifier score that stands (none,
  a human reversed the verifier, or its copies' scores disagree), or that score is outside
  ``[min_score, max_score]``.
* ``passed``, ``failed``: it passed when only failures are asked for, or the reverse.
* ``no_band``: its task has fewer verdicts than ``min_samples``, with a band filter.
* ``band``, ``outside_band``: its task's band isn't one of ``bands``, or its solve rate is
  outside ``band_range`` with ``inside_band``.
* ``missing_label``: neither it nor a copy has a current label ``require_labels`` asks for.
* ``duplicate``: another copy of its trajectory is kept.

A task's band (:func:`deepgym.passk.task_difficulty`) is per environment definition
(``EnvFingerprint.environment_sha256``), from the verdicts of the traces in the population
asked for that aren't marked bad (invalidated, contested, inconsistent or with a
forbidden label), whatever the later checks drop: ``require_labels`` selects among the
samples, so requiring an expert's pass still keeps a task the model solves half the time
``learnable``. Splits are by prompt (``EnvFingerprint.task_sha256``): each prompt lands in
one split, by its own hash, so adding prompts never moves another.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from deepgym.labels import PASS_MARK, Label, LabelSource, LabelValue, Name, current_labels
from deepgym.models import SCORE_TOLERANCE
from deepgym.passk import Band, task_difficulty
from deepgym.traces.schema import Trace, Usd, check_unique_ids, json_text, sha256_text

Split = Literal['train', 'val', 'test']
DropReason = Literal[
    'agent', 'model', 'env_name', 'env_variant', 'invalidated', 'contested', 'inconsistent',
    'forbidden_label', 'incomplete', 'no_verdict', 'usage_incomplete', 'cost_unknown',
    'over_cost', 'no_score', 'score', 'passed', 'failed', 'no_band', 'band', 'outside_band',
    'missing_label', 'duplicate',
]  # fmt: skip
UnreadReason = Literal['episode_without_trace', 'not_a_trace_file', 'merged_copy']
"""Why a source held something that never reached :func:`build_dataset` as a trace."""

VerdictSource = Literal['verifier', 'human']
Sha256 = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Ratio = Annotated[FiniteFloat, Field(ge=0, lt=1)]
Score = Annotated[FiniteFloat, Field(ge=0, le=1)]


class LabelMatch(BaseModel):
    """A current label to require or forbid on a trace, on the whole trace or one step.

    Fields left None match anything; a value matches only a value of the same kind, so
    ``true`` never matches the number 1.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    name: Name
    value: LabelValue | None = None
    source: LabelSource | None = None
    author: Name | None = None

    def matches(self, label: Label) -> bool:
        """Return whether *label* has this name and every other field given."""
        value = self.value
        same_value = value is None or (
            isinstance(value, bool) is isinstance(label.value, bool) and value == label.value
        )
        return (
            label.name == self.name
            and same_value
            and self.source in (None, label.source)
            and self.author in (None, label.author)
        )


class DatasetSpec(BaseModel):
    """Which traces a dataset keeps and how it splits them; JSON-serialisable.

    Every filter left at its default keeps everything except traces without a verdict.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    require_verdict: bool = True
    """Drop traces without a verdict, the verifier's or a human's (they never count as fails)."""

    min_score: Score | None = None
    max_score: Score | None = None
    """Keep traces whose verifier score stands and is in ``[min_score, max_score]``."""

    passed: bool | None = None
    """True keeps only passed verdicts, False only failed ones."""

    agents: tuple[str, ...] | None = Field(default=None, min_length=1)
    models: tuple[str, ...] | None = Field(default=None, min_length=1)
    """Keep traces of these agents (``AgentMeta.name``) and models (``AgentMeta.model``)."""

    require_labels: tuple[LabelMatch, ...] = ()
    """Keep traces with a current label matching each of these, on them or a copy."""

    forbid_labels: tuple[LabelMatch, ...] = ()
    """Drop traces with a current label matching any of these, on them or any copy."""

    env_names: tuple[str, ...] | None = Field(default=None, min_length=1)
    env_variants: tuple[int, ...] | None = Field(default=None, min_length=1)
    """Keep traces of these environments (``EnvFingerprint.name``) and registry variants."""

    require_usage_complete: bool = False
    """Drop traces where some model call reported no usage (``Trace.usage_complete``)."""

    max_cost_usd: Usd | None = None
    """Drop traces that cost more, or whose cost is unknown (``Trace.cost_usd`` is None)."""

    bands: tuple[Band, ...] | None = Field(default=None, min_length=1)
    """Keep traces whose task is in one of these difficulty bands."""

    min_samples: int = Field(default=1, ge=1)
    """Verdicts a task needs to have a band."""

    band_range: tuple[Score, Score] = (0.1, 0.9)
    """Solve rates inside the band, both inclusive (``TaskBand.outside_band``)."""

    inside_band: bool = False
    """Drop traces whose task's solve rate is outside ``band_range``."""

    seed: int = 0
    test_ratio: Ratio = 0.1
    val_ratio: Ratio = 0.1
    """Chance that a prompt lands in test, and in val; the rest go to train."""

    @model_validator(mode='after')
    def check_ranges(self) -> DatasetSpec:
        """Reject an empty score or band range, and split ratios adding up to 1 or more."""
        low, high = self.min_score, self.max_score
        if low is not None and high is not None and low > high:
            raise ValueError('min_score must be at most max_score')
        if self.band_range[0] > self.band_range[1]:
            raise ValueError('band_range is (low, high) with low <= high')
        if self.test_ratio + self.val_ratio >= 1:
            raise ValueError('test_ratio + val_ratio must be below 1')
        return self

    def split_of(self, task_sha256: str) -> Split:
        """Place a prompt by its own hash, so adding another never moves it.

        ``u = int(sha256(f'{seed}:{task_sha256}'), 16) / 2**256`` goes to test below
        ``test_ratio``, to val below ``test_ratio + val_ratio``, and to train otherwise.
        """
        u = int(hashlib.sha256(f'{self.seed}:{task_sha256}'.encode()).hexdigest(), 16) / 2**256
        if u < self.test_ratio:
            return 'test'
        return 'val' if u < self.test_ratio + self.val_ratio else 'train'


class TaskBand(BaseModel):
    """One task's verdicts among the band samples, and its band."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    samples: int
    solved: int
    band: Band | None
    """None when there are fewer samples than ``min_samples``."""

    outside_band: bool | None
    """Whether the solve rate is outside ``band_range``; None without a band."""


class DatasetEntry(BaseModel):
    """One kept trace."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    split: Split
    environment_sha256: str
    """The task its band is counted on."""

    task_sha256: str
    """The prompt it is split by."""

    content_sha256: Sha256
    """Its trajectory, shared by its copies (:func:`copy_sha256`)."""

    sha256: Sha256
    """SHA-256 of the trace as a whole-trace JSONL export writes it, the line without its
    newline (``json_text(trace)``)."""

    passed: bool | None
    """Its verdict: the pooled human one, when a human judged it or a copy, else the
    verifier's; None without one."""

    verdict_source: VerdictSource | None
    score: float | None
    """The verifier's score, or None when it gave none, a human reversed its verdict, or
    the scores of the trace's copies disagree."""

    scores_disagree: bool
    """Whether the verifier scores that stand on the trace's pooled copies differ by more
    than ``SCORE_TOLERANCE``, so no score ships."""

    overridden: bool
    """Whether a human reversed the verdict the verifier gave, which the exported trace
    holds; a human verdict where the verifier gave none isn't one."""

    @classmethod
    def of(
        cls, trace: Trace, spec: DatasetSpec, human: bool | None, sha256: str, disputed: bool
    ) -> DatasetEntry:
        """Describe *trace*, given the human verdict on it (None without one), its hash and
        whether its copies' scores disagree."""
        env = trace.provenance.env
        passed, source, score, overridden = judged(trace, human)
        return cls(
            split=spec.split_of(env.task_sha256),
            environment_sha256=env.environment_sha256,
            task_sha256=env.task_sha256,
            content_sha256=copy_sha256(trace),
            sha256=sha256,
            passed=passed,
            verdict_source=source,
            score=None if disputed else score,
            scores_disagree=disputed,
            overridden=overridden,
        )

    @property
    def human(self) -> bool | None:
        """The human verdict this entry was built with, or None."""
        return self.passed if self.verdict_source == 'human' else None


class DuplicateGroup(BaseModel):
    """Copies of one trajectory: traces of one environment definition with the same task
    text and trajectory.

    The trajectory is the initial observation's text and each step's action, model
    messages (role and text), observation text, whether it was applied and its error;
    times, usage, token data, screenshots and verdicts don't count. Of the copies the other
    checks keep, one is kept (see :mod:`deepgym.dataset.build`) and the rest are dropped as
    ``duplicate``.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    content_sha256: Sha256
    copies: tuple[str, ...]
    """Every copy among the traces given, by id."""

    kept: str | None
    """The copy kept; None when the checks dropped them all."""

    verdicts_disagree: bool
    """Whether the pooled human verdicts disagree, or the verifier's verdicts on the pooled
    copies differ, or their standing scores do by more than ``SCORE_TOLERANCE``."""


class DatasetCounts(BaseModel):
    """How many traces came in, were kept, landed in each split and were dropped."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    sources: int
    """Traces given to :func:`build_dataset`: those kept and those dropped."""

    kept: int
    splits: dict[Split, int]
    dropped: dict[DropReason, int]
    overridden: int
    """Kept traces whose verifier verdict, which their files hold, a human reversed."""

    unread: dict[UnreadReason, int]
    """What the sources held that never reached the builder as a trace."""

    unmatched_labels: int
    """Current label records that matched none of the traces given."""

    @classmethod
    def of(
        cls,
        entries: Mapping[str, DatasetEntry],
        dropped: Mapping[DropReason, Sequence[str]],
        unread: Mapping[UnreadReason, int],
        unmatched_labels: int,
    ) -> DatasetCounts:
        """Count *entries* per split and *dropped* per reason, with what is recorded as is."""
        splits = Counter(entry.split for entry in entries.values())
        lost: dict[DropReason, int] = {reason: len(ids) for reason, ids in dropped.items()}
        return cls(
            sources=len(entries) + sum(lost.values()),
            kept=len(entries),
            splits={split: splits[split] for split in get_args(Split)},
            dropped=lost,
            overridden=sum(entry.overridden for entry in entries.values()),
            unread=dict(unread),
            unmatched_labels=unmatched_labels,
        )


class Dataset(BaseModel):
    """What :func:`build_dataset` kept and dropped; every trace given is in one or the other."""

    model_config = ConfigDict(frozen=True)

    spec: DatasetSpec
    traces: tuple[Trace, ...] = Field(exclude=True, repr=False)
    """The kept traces, by start time, then id."""

    entries: dict[str, DatasetEntry]
    """Each kept trace by id."""

    dropped: dict[DropReason, tuple[str, ...]]
    """The ids of the dropped traces, by reason."""

    duplicates: tuple[DuplicateGroup, ...]
    """Every trajectory with more than one copy among the traces given."""

    bands: dict[str, TaskBand]
    """Each task among the band samples, by ``environment_sha256``."""

    unread: dict[UnreadReason, int]
    unmatched: tuple[Label, ...] = Field(exclude=True, repr=False)
    """The current label records that matched none of the traces given."""

    @property
    def counts(self) -> DatasetCounts:
        """How many traces were kept, per split and dropped per reason."""
        return DatasetCounts.of(self.entries, self.dropped, self.unread, len(self.unmatched))


class _Facts(BaseModel):
    """What the labels and verifier grades say about the traces given and their copies."""

    model_config = ConfigDict(frozen=True)

    group: dict[str, str]
    """The copy hash of each trace (:func:`copy_sha256`)."""

    invalid: frozenset[str]
    human: dict[str, frozenset[bool]]
    """The current human ``passed`` values on each trajectory's pooled copies."""

    marks: dict[str, tuple[Label, ...]]
    """The current labels (not invalidations) on each trajectory's pooled copies."""

    forbidden: frozenset[str]
    """The trajectories a copy of which has a label ``forbid_labels`` rules out."""

    status: dict[str, Literal['contested', 'inconsistent']]
    verdicts: dict[str, frozenset[bool]]
    """The verifier's verdicts on each trajectory's pooled copies."""

    disputed: frozenset[str]
    """The trajectories whose pooled copies' standing scores differ beyond tolerance."""

    unmatched: tuple[Label, ...]


def build_dataset(
    traces: Sequence[Trace],
    spec: DatasetSpec,
    labels: Sequence[Label] = (),
    episodes: Mapping[str, str] | None = None,
    unread: Mapping[UnreadReason, int] | None = None,
) -> Dataset:
    """Filter, deduplicate and split *traces* as *spec* says.

    Args:
        traces: The candidate traces, one copy of each.
        spec: The filters and split settings.
        labels: Label records, every record of each target labeled (as
            :func:`deepgym.labels.current_labels` needs).
        episodes: The trace id of each console episode recorded from a trace, by episode
            id (``episode_store.read_trace_ids``), so episode labels reach their trace.
        unread: What the sources held that never became one of *traces*, recorded as is.

    Returns:
        The kept traces with their split, keys, hash and verdict, the ids of the dropped
        ones by reason, the copies of each trajectory and the task bands. The result
        doesn't depend on the order of *traces*.

    Raises:
        ValueError: If two traces share an id, or no kept trace landed in train while
            some landed in val or test.
    """
    check_unique_ids(traces)
    ordered = sorted(traces, key=lambda trace: (trace.provenance.started_at, trace.trace_id))
    facts = _facts(ordered, spec, labels, episodes or {})
    marked: dict[str, DropReason | None] = {
        trace.trace_id: _marked(trace, spec, facts) for trace in ordered
    }
    bands = _bands((trace for trace in ordered if marked[trace.trace_id] is None), facts, spec)
    dropped: dict[DropReason, list[str]] = {}
    survivors: dict[str, list[Trace]] = {}
    for trace in ordered:
        passed, _, score, _ = _shipped(trace, facts)
        reason = (
            marked[trace.trace_id]
            or trace_failure(trace, spec, passed, score, bands)
            or _missing_label(trace, spec, facts)
        )
        if reason is None:
            survivors.setdefault(facts.group[trace.trace_id], []).append(trace)
        else:
            dropped.setdefault(reason, []).append(trace.trace_id)
    kept: list[Trace] = []
    for copies in survivors.values():
        first, *rest = sorted(copies, key=lambda trace: _preference(trace, facts))
        kept.append(first)
        if rest:
            dropped.setdefault('duplicate', []).extend(trace.trace_id for trace in rest)
    kept.sort(key=lambda trace: (trace.provenance.started_at, trace.trace_id))
    entries = {
        trace.trace_id: DatasetEntry.of(
            trace,
            spec,
            _human(trace, facts),
            sha256_text(json_text(trace)),
            facts.group[trace.trace_id] in facts.disputed,
        )
        for trace in kept
    }
    splits = Counter(entry.split for entry in entries.values())
    if entries and not splits['train']:
        raise ValueError(
            f'No trace landed in train ({splits["val"]} in val, {splits["test"]} in test): '
            'lower test_ratio and val_ratio, or change the seed'
        )
    return Dataset(
        spec=spec,
        traces=tuple(kept),
        entries=dict(sorted(entries.items())),
        dropped={
            reason: tuple(sorted(dropped[reason]))
            for reason in get_args(DropReason)
            if reason in dropped
        },
        duplicates=_duplicates(ordered, {trace.trace_id for trace in kept}, facts),
        bands=dict(sorted(bands.items())),
        unread=dict(unread or {}),
        unmatched=facts.unmatched,
    )


def trace_failure(
    trace: Trace,
    spec: DatasetSpec,
    passed: bool | None,
    score: float | None,
    bands: Mapping[str, TaskBand],
) -> DropReason | None:
    """Return the first check *trace* fails that needs no labels, given its verdict and the
    verifier score that stands: population, verdict, usage, cost, score and band."""
    return (
        _population_failure(trace, spec)
        or _quality_failure(trace, spec, passed, score)
        or _band_failure(trace, spec, bands)
    )


def judged(
    trace: Trace, human: bool | None
) -> tuple[bool | None, VerdictSource | None, float | None, bool]:
    """Return *trace*'s verdict, whose it is, the verifier score that stands and whether a
    human reversed the verifier's verdict, given the human verdict on it."""
    result = trace.result if trace.has_verdict else None
    if human is not None:
        reversed_ = result is not None and result.passed is not human
        return human, 'human', None if result is None or reversed_ else result.score, reversed_
    if result is not None:
        return result.passed, 'verifier', result.score, False
    return None, None, None, False


def copy_sha256(trace: Trace) -> str:
    """Hash *trace*'s environment definition, task text and trajectory, which its copies
    share (see :class:`DuplicateGroup`). Actions leave out fields at their defaults, so a
    field added with a default doesn't change published hashes."""
    start = trace.initial_observation
    steps = [
        {
            'action': step.action.model_dump(mode='json', exclude_defaults=True),
            'messages': [[message.role, message.content] for message in step.messages],
            'observation': None if step.observation is None else step.observation.content,
            'applied': step.applied,
            'error': step.error,
        }
        for step in trace.steps
    ]
    return canonical_sha256({
        'environment_sha256': trace.provenance.env.environment_sha256,
        'task': trace.task,
        'initial_observation': None if start is None else start.content,
        'steps': steps,
    })  # fmt: skip


def canonical_sha256(value: object) -> str:
    """SHA-256 of *value* as compact JSON with sorted keys, as Python's ``json`` writes it."""
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return sha256_text(text)


def _facts(
    traces: Sequence[Trace], spec: DatasetSpec, labels: Sequence[Label], episodes: Mapping[str, str]
) -> _Facts:
    """Group *traces* into copies and pool what the current labels and the verifier say
    about each trajectory, counting the labels about none of *traces*."""
    group = {trace.trace_id: copy_sha256(trace) for trace in traces}
    about = [(label, _trace_of(label, episodes)) for label in current_labels(labels)]
    unmatched = tuple(label for label, trace_id in about if trace_id not in group)
    invalid = group.keys() & {trace_id for label, trace_id in about if label.type == 'invalidate'}
    pooled = {
        trace.trace_id
        for trace in traces
        if trace.trace_id not in invalid and _population_failure(trace, spec) is None
    }
    human: dict[str, set[bool]] = {}
    marks: dict[str, list[Label]] = {}
    forbidden: set[str] = set()
    for label, trace_id in about:
        if trace_id not in group or label.type == 'invalidate':
            continue
        content = group[trace_id]
        if any(match.matches(label) for match in spec.forbid_labels):
            forbidden.add(content)
        if trace_id not in pooled:
            continue
        marks.setdefault(content, []).append(label)
        if label.source == 'human' and label.name == PASS_MARK:
            if isinstance(label.value, bool) and label.target.step is None:
                human.setdefault(content, set()).add(label.value)
    verdicts: dict[str, set[bool]] = {}
    standing: dict[str, list[float]] = {}
    for trace in traces:
        result = trace.result if trace.has_verdict else None
        if result is None or trace.trace_id not in pooled:
            continue
        content = group[trace.trace_id]
        verdicts.setdefault(content, set()).add(result.passed)
        values = human.get(content, set())
        score = judged(trace, next(iter(values)) if len(values) == 1 else None)[2]
        if score is not None:
            standing.setdefault(content, []).append(score)
    status: dict[str, Literal['contested', 'inconsistent']] = {}
    for content in verdicts.keys() | human.keys():
        if len(human.get(content, ())) > 1:
            status[content] = 'contested'
        elif content not in human and len(verdicts.get(content, ())) > 1:
            status[content] = 'inconsistent'
    disputed = {
        content
        for content, scores in standing.items()
        if max(scores) - min(scores) > SCORE_TOLERANCE
    }
    return _Facts(
        group=group,
        invalid=frozenset(invalid),
        human={content: frozenset(values) for content, values in human.items()},
        marks={content: tuple(found) for content, found in marks.items()},
        forbidden=frozenset(forbidden),
        status=status,
        verdicts={content: frozenset(found) for content, found in verdicts.items()},
        disputed=frozenset(disputed),
        unmatched=unmatched,
    )


def _trace_of(label: Label, episodes: Mapping[str, str]) -> str | None:
    """Return the trace *label* is about: its trace, or for a label on a console episode
    or one of its replay frames, the trace the episode was recorded from."""
    target = label.target
    if target.kind == 'trace':
        return target.id
    return episodes.get(target.id) if target.kind == 'episode' else None


def _human(trace: Trace, facts: _Facts) -> bool | None:
    """The pooled human verdict on *trace*'s copies, or None without one or when humans
    disagree."""
    values = facts.human.get(facts.group[trace.trace_id], frozenset())
    return next(iter(values)) if len(values) == 1 else None


def _preference(trace: Trace, facts: _Facts) -> tuple[bool, bool, bool, float, datetime, str]:
    """Order copies for keeping: one with a verdict, its verifier not reversed, with a
    standing score, the highest score; then the earliest, then by id."""
    passed, _, score, overridden = judged(trace, _human(trace, facts))
    started = trace.provenance.started_at
    return passed is None, overridden, score is None, -(score or 0.0), started, trace.trace_id


def _shipped(
    trace: Trace, facts: _Facts
) -> tuple[bool | None, VerdictSource | None, float | None, bool]:
    """What ships for *trace*: :func:`judged`, with no score when its copies' disagree."""
    passed, source, score, overridden = judged(trace, _human(trace, facts))
    disputed = facts.group[trace.trace_id] in facts.disputed
    return passed, source, None if disputed else score, overridden


def _task(trace: Trace) -> str:
    return trace.provenance.env.environment_sha256


def _population_failure(trace: Trace, spec: DatasetSpec) -> DropReason | None:
    """Return the first agent, model or environment check *trace* fails, or None."""
    agent, env = trace.provenance.agent, trace.provenance.env
    failed: list[tuple[DropReason, bool]] = [
        ('agent', spec.agents is not None and agent.name not in spec.agents),
        ('model', spec.models is not None and agent.model not in spec.models),
        ('env_name', spec.env_names is not None and env.name not in spec.env_names),
        ('env_variant', spec.env_variants is not None and env.variant not in spec.env_variants),
    ]
    return next((reason for reason, fails in failed if fails), None)


def _marked(trace: Trace, spec: DatasetSpec, facts: _Facts) -> DropReason | None:
    """Return the first check that also keeps *trace* out of the band samples, or None:
    the population asked for, and the marks of a bad trace."""
    content = facts.group[trace.trace_id]
    failed: list[tuple[DropReason, bool]] = [
        ('invalidated', trace.trace_id in facts.invalid),
        ('contested', facts.status.get(content) == 'contested'),
        ('inconsistent', facts.status.get(content) == 'inconsistent'),
        ('forbidden_label', content in facts.forbidden),
    ]
    return _population_failure(trace, spec) or next(
        (reason for reason, fails in failed if fails), None
    )


def _quality_failure(
    trace: Trace, spec: DatasetSpec, passed: bool | None, score: float | None
) -> DropReason | None:
    """Return the first verdict, usage, cost or score check *trace* fails, or None."""
    missing = (spec.require_verdict or spec.passed is not None) and passed is None
    scored = spec.min_score is not None or spec.max_score is not None
    low = 0.0 if spec.min_score is None else spec.min_score
    high = 1.0 if spec.max_score is None else spec.max_score
    cost, max_cost = trace.cost_usd, spec.max_cost_usd
    failed: list[tuple[DropReason, bool]] = [
        ('incomplete', missing and trace.result is None),
        ('no_verdict', missing),
        ('usage_incomplete', spec.require_usage_complete and not trace.usage_complete),
        ('cost_unknown', max_cost is not None and cost is None),
        ('over_cost', max_cost is not None and cost is not None and cost > max_cost),
        ('no_score', scored and score is None),
        ('score', scored and score is not None and not low <= score <= high),
        ('passed', spec.passed is False and passed is True),
        ('failed', spec.passed is True and passed is False),
    ]
    return next((reason for reason, fails in failed if fails), None)


def _band_failure(
    trace: Trace, spec: DatasetSpec, bands: Mapping[str, TaskBand]
) -> DropReason | None:
    """Return the first band check *trace*'s task fails, or None."""
    if spec.bands is None and not spec.inside_band:
        return None
    task = bands.get(_task(trace))
    if task is None or task.band is None:
        return 'no_band'
    if spec.bands is not None and task.band not in spec.bands:
        return 'band'
    if spec.inside_band and task.outside_band:
        return 'outside_band'
    return None


def _missing_label(trace: Trace, spec: DatasetSpec, facts: _Facts) -> DropReason | None:
    """Return 'missing_label' if the pooled labels of *trace*'s copies lack one
    ``require_labels`` asks for."""
    marks = facts.marks.get(facts.group[trace.trace_id], ())
    return None if all(any(map(m.matches, marks)) for m in spec.require_labels) else 'missing_label'


def _bands(traces: Iterable[Trace], facts: _Facts, spec: DatasetSpec) -> dict[str, TaskBand]:
    """Count each task's verdicts among *traces* and place it in a band."""
    samples: Counter[str] = Counter()
    solved: Counter[str] = Counter()
    for trace in traces:
        passed = judged(trace, _human(trace, facts))[0]
        if passed is not None:
            samples[_task(trace)] += 1
            solved[_task(trace)] += passed
    low, high = spec.band_range
    bands: dict[str, TaskBand] = {}
    for task, n in samples.items():
        difficulty = task_difficulty(n, solved[task], low, high) if n >= spec.min_samples else None
        bands[task] = TaskBand(
            samples=n,
            solved=solved[task],
            band=None if difficulty is None else difficulty.band,
            outside_band=None if difficulty is None else difficulty.outside_band,
        )
    return bands


def _duplicates(
    traces: Sequence[Trace], kept: set[str], facts: _Facts
) -> tuple[DuplicateGroup, ...]:
    """Describe every trajectory with more than one copy among *traces*."""
    groups: dict[str, list[str]] = {}
    for trace in traces:
        groups.setdefault(facts.group[trace.trace_id], []).append(trace.trace_id)
    duplicates = [
        DuplicateGroup(
            content_sha256=content,
            copies=tuple(sorted(copies)),
            kept=next((trace_id for trace_id in copies if trace_id in kept), None),
            verdicts_disagree=len(facts.human.get(content, ())) > 1
            or len(facts.verdicts.get(content, ())) > 1
            or content in facts.disputed,
        )
        for content, copies in groups.items()
        if len(copies) > 1
    ]
    return tuple(sorted(duplicates, key=lambda group: group.copies))
