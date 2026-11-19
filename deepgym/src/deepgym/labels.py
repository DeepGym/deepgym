"""Labels: scores and marks on runs, episodes and trace steps, kept as an append-only log.

A :class:`Label` scores one thing about a target (a console run, a console episode or a
trace, optionally one step of it and a character span of one of its texts): a verifier
score, a judge's verdict or an expert's mark with the reason. Records are never edited. A
correction is a new record that supersedes an earlier one, a ``retract`` record withdraws
one, and an ``invalidate`` record excludes a whole target, giving the reason.
:func:`current_labels` keeps the records still in force, and :func:`excluded` names the
targets to leave out.

One convention covers experts and judges alike, holding what a
``rubric_calibration.LabeledAnswer`` needs (there is no adapter yet): the verdict on an
answer is a bool label named ``passed``, and the verdict on one rubric check a bool label
named by its check id (``c1``, ``c2``, ...), True meaning MET. These verdict names take
only bools, on a whole target, and an author keeps one current verdict per name and
``group_id`` on a target: changing it means superseding it.

The console keeps labels in its database (:class:`deepgym.console.label_store.LabelStore`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)

TargetKind = Literal['run', 'episode', 'trace']
LabelSource = Literal['verifier', 'judge', 'human']
LabelType = Literal['label', 'invalidate', 'retract']

SQLITE_MAX_INT = 2**63 - 1
"""Largest integer SQLite holds; a bigger one overflows in the driver instead of matching."""

PASS_MARK = 'passed'
"""Name of the bool label that passes or fails a whole answer."""

_CHECK_ID = re.compile(r'c[1-9][0-9]*')  # deepgym.rubric.Rubric.ids

_HIDDEN = r'\p{Cc}\p{Cf}\p{Zl}\p{Zp}'  # control and format characters, line separators
Name = Annotated[
    str, StringConstraints(pattern=rf'^[^\s{_HIDDEN}](?:[^{_HIDDEN}]*[^\s{_HIDDEN}])?$')
]
"""Text that names something: one line, not empty, no whitespace at either end, and no
control or format characters (NUL, bidi controls, U+2028, ...)."""

Offset = Annotated[StrictInt, Field(ge=0, le=SQLITE_MAX_INT)]
"""A step number or text offset."""

LabelValue = StrictBool | FiniteFloat | Name
"""A pass or MET mark, a number such as a score or rating, or a one-line category."""


class LabelTarget(BaseModel):
    """What a label is about: a run, an episode or a trace, or one step of it.

    ``field`` names one text of the target, and ``span`` a part of that text, in Unicode
    code points (Python ``str`` indexes; a UI counting UTF-16 units converts).
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: TargetKind
    id: Name
    """A console run or episode id, or a ``Trace.trace_id``. An episode recorded from a
    trace has its own id. Ids aren't checked against any store: a trace may live only in
    the traces directory."""

    step: Offset | None = None
    """The step, numbered as its target numbers it: ``TraceStep.index`` for a trace (from 1),
    the replay frame index for an episode (from 0). A run has no steps. Frame k of an
    episode recorded from a trace is trace step k + 1 when the trace has no rejected
    actions (the replay leaves those out)."""

    field: str | None = Field(
        default=None, pattern=r'^(answer|action|observation|message:(0|[1-9][0-9]*))$'
    )
    """Which text: ``'answer'`` (without a step) is the final text the target was graded
    on, such as a run's solution or an episode's or trace's final answer; with a step,
    ``'action'`` is ``Action.content``, ``'observation'`` is ``Observation.content``, and
    ``'message:<i>'`` is ``TraceStep.messages[i].content`` (traces only)."""

    span: tuple[Offset, Offset] | None = None
    """``(start, end)`` code point offsets into ``field``'s text, end exclusive."""

    quote: str | None = None
    """The spanned text as the labeler saw it, so a span that no longer fits shows."""

    @model_validator(mode='after')
    def check_step(self) -> LabelTarget:
        """Reject a step on a run, and trace step 0."""
        if self.step is not None and self.kind == 'run':
            raise ValueError('a run has no steps')
        if self.step == 0 and self.kind == 'trace':
            raise ValueError('trace steps count from 1')
        return self

    @model_validator(mode='after')
    def check_text(self) -> LabelTarget:
        """Require a field that fits the step for a span, and a quote as long as the span."""
        if self.field is not None and (self.field == 'answer') != (self.step is None):
            raise ValueError("field 'answer' takes no step; the other fields need one")
        if self.field is not None and self.field.startswith('message:') and self.kind != 'trace':
            raise ValueError('only trace steps have messages')
        if self.span is not None and self.field is None:
            raise ValueError('a span needs a field naming the text it is in')
        if self.span is not None and self.span[0] >= self.span[1]:
            raise ValueError('a span is (start, end) with start < end')
        if self.quote is not None and self.span is None:
            raise ValueError('a quote needs a span')
        if self.quote is not None and len(self.quote) != self.span[1] - self.span[0]:
            raise ValueError('the quote must be the spanned text: end - start code points')
        return self


class LabelCreate(BaseModel):
    """A label as its author gives it; storing it adds seq and time (:class:`Label`)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    target: LabelTarget
    type: LabelType = 'label'
    """'label' scores the target. 'invalidate' excludes the whole target, steps included.
    'retract' withdraws the record it supersedes, and with it the chain that record
    corrected; superseding a retraction is refused."""

    id: Name | None = None
    """Optional client id, for safe retries: storing the same id and body again returns the
    stored record, and the same id with another body is refused. Generated when None."""

    name: Name | None = None
    """What is scored, e.g. ``'passed'``, a check id such as ``'c2'``, or a mark such as
    ``'tone'``; None unless a label."""

    value: LabelValue | None = None
    """True or False, a finite number, or a one-line category; None unless a label."""

    source: LabelSource
    author: Name
    """Who labeled: an expert's handle, or the judge or verifier that did."""

    comment: str | None = None
    """Why: required on a human's ``passed`` verdict (a right answer too), an invalidation and
    a retraction; optional on every other label."""

    blind: StrictBool | None = None
    """True when the labeler saw no other label or judge output first; None if unrecorded."""

    rubric_version: Name | None = None
    """Version of the rubric or judge prompt, e.g. ``deepgym.rubric.RUBRIC_PROMPT_VERSION``."""

    rubric_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    """Hash of the task and rubric labeled against (``rubric_calibration.rubric_sha256``)."""

    group_id: Name | None = None
    """Ties together the labels of one judge call, or of one repeat of a labeling pass."""

    supersedes: Name | None = None
    """Id of the record this one corrects or retracts. It must be on the same target, from
    the same source and author, not a retraction, and not already superseded; a correction
    keeps its type. Someone else's view is a new label, so disagreement stays visible. Any
    human may retract an invalidation, whoever made it."""

    @model_validator(mode='after')
    def check_type(self) -> LabelCreate:
        """Require what each type needs: a mark on a label, a whole target otherwise."""
        is_label = self.type == 'label'
        if is_label and (self.name is None or self.value is None):
            raise ValueError('a label needs a name and a value')
        if not is_label and (self.name is not None or self.value is not None):
            raise ValueError(f'a {self.type} record has no name or value')
        if not is_label and (self.target.step is not None or self.target.field is not None):
            raise ValueError(f'a {self.type} record is about a whole target: no step or field')
        if self.type == 'retract' and self.supersedes is None:
            raise ValueError('a retraction names what it retracts in supersedes')
        human_pass = self.source == 'human' and self.name == PASS_MARK
        if (human_pass or not is_label) and not (self.comment or '').strip():
            raise ValueError(
                "a human's passed verdict, an invalidation and a retraction need a comment "
                'saying why'
            )
        return self

    @model_validator(mode='after')
    def check_verdict(self) -> LabelCreate:
        """Hold a verdict (``passed`` or a check id) to a bool on a whole target."""
        if not is_verdict(self.name):
            return self
        if not isinstance(self.value, bool):
            raise ValueError(f"'{self.name}' is a verdict: its value is true or false")
        if self.target.step is not None or self.target.field is not None:
            raise ValueError(
                f"'{self.name}' is a verdict on a whole target; give a mark on a step or "
                'text another name'
            )
        return self


class Label(LabelCreate):
    """A stored label record; it never changes."""

    seq: int = Field(ge=1)
    """Position in the store, increasing with each record stored; a sync cursor (``after_seq``)."""

    id: Name
    created_at: AwareDatetime
    """Never earlier than the record stored before it."""


def is_verdict(name: str | None) -> bool:
    """Return whether *name* names a verdict: ``passed`` or a rubric check id."""
    return name == PASS_MARK or _CHECK_ID.fullmatch(name or '') is not None


def current_labels(labels: Sequence[Label]) -> list[Label]:
    """Return the records of *labels* still in force, in order.

    A record is in force unless another one supersedes it; retractions are left out too.
    Retracting a correction so withdraws its whole chain.
    Pass whole target histories (every record of each target): what supersedes a record is
    always on its target, but a subset filtered another way (one name, one source) can
    miss it and keep what it replaced.
    """
    replaced = {label.supersedes for label in labels}
    return [label for label in labels if label.id not in replaced and label.type != 'retract']


def excluded(labels: Sequence[Label]) -> set[tuple[TargetKind, str]]:
    """Return the ``(kind, id)`` of each target a current invalidation in *labels* excludes.

    Drop those targets, with every label on them and their steps, before using the labels.
    Pass whole target histories, as for :func:`current_labels`.
    """
    return {
        (label.target.kind, label.target.id)
        for label in current_labels(labels)
        if label.type == 'invalidate'
    }
