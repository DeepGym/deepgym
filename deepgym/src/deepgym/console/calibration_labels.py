"""Judge calibration input from the console's expert labels on runs.

:func:`labeled_answers` turns runs that experts labeled (see :mod:`deepgym.labels`) into
:class:`~deepgym.rubric_calibration.LabeledAnswer` items for
:func:`~deepgym.rubric_calibration.calibrate`, and lists what it left out and why.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from deepgym.console.label_store import LabelStore
from deepgym.console.run_store import RunStore
from deepgym.labels import PASS_MARK, Label, is_verdict
from deepgym.rubric_calibration import ExpertLabel, LabeledAnswer

SkipReason = Literal[
    'excluded', 'no_run', 'rubric_mismatch', 'no_passed', 'conflicting_author', 'tie'
]


class Skipped(BaseModel):
    """Labels left out of the calibration input, and why."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    reason: SkipReason
    """``excluded``: a current invalidation excludes the run. ``no_run``: the run store has
    no such run, so no answer text. ``rubric_mismatch``: an expert's verdicts recorded
    against another rubric hash than the one asked for; they are left out.
    ``conflicting_author``: one expert holds both true and false for a verdict on the run
    (in different groups); that verdict is left out of their label, and if it is
    ``passed``, the expert is left out of the run. ``no_passed``: an expert gave check
    verdicts but no ``passed`` (none, or only ones left out above) and is left out of the
    run; or no expert is left with a ``passed`` (none was given, or each was left out for
    a reason here), and the run is left out. ``tie``: as many experts mark a verdict true
    as false; it is left out of the label the judge is measured against: ``passed`` is
    None, which leaves the answer out of ``overall``, ``blind``, ``assisted`` and the
    ``min_kappa`` bar, or the check is left out of ``checks``. Every expert's own label
    stays in ``experts`` for the human baselines."""
    detail: str


class LabeledRuns(BaseModel):
    """Calibration input built from stored labels, and what was left out."""

    model_config = ConfigDict(frozen=True)

    answers: tuple[LabeledAnswer, ...]
    skipped: tuple[Skipped, ...]
    no_rubric_hash: int
    """Expert verdicts read (on runs neither excluded nor missing) that record no rubric
    hash: kept, though they can't be checked against ``rubric_sha256``."""


def labeled_answers(
    labels: LabelStore, runs: RunStore, run_ids: Iterable[str], *, rubric_sha256: str | None = None
) -> LabeledRuns:
    """Build calibration input from the experts' current verdicts on *run_ids*.

    Each run becomes one answer whose text is the run's solution. Only current human
    verdicts count (``passed`` and check ids): superseded and retracted records are
    ignored, a run a current invalidation excludes is skipped, and judge and verifier
    labels are left out, since ``calibrate`` runs the judge itself. Each expert (label
    author) gives one :class:`~deepgym.rubric_calibration.ExpertLabel`, blind only when
    every one of their verdicts on the run is labeled blind (unrecorded counts as not
    blind). With *rubric_sha256*, verdicts recorded against another rubric hash are left
    out; those recording none are kept and counted in ``no_rubric_hash``.

    The label the judge is measured against is the experts' majority, on ``passed`` and on
    each check, blind only when every expert's is. On a tie ``passed`` is None, or the
    check is left out. Every expert's own label is in ``experts``; to choose the judge's
    reference another way, rebuild the answers with ``LabeledAnswer.model_validate``.

    Args:
        labels: The label store.
        runs: The run store holding the runs' solutions.
        run_ids: The runs, in the order the answers should take. For the runs a label
            query matches, pass ``dict.fromkeys(label.target.id for label in
            labels.page(query, 0, None).labels)`` with ``target_kind='run'`` in the query.
        rubric_sha256: The hash of the task and rubric being calibrated
            (``rubric_calibration.rubric_sha256(reward_fn.task, reward_fn.rubric)``).

    Returns:
        The answers, and each run, expert or verdict left out, with the reason.
    """
    answers: list[LabeledAnswer] = []
    skipped: list[Skipped] = []
    no_rubric_hash = 0
    for run_id in run_ids:
        current = labels.current('run', run_id)
        if why := [label.comment or '' for label in current if label.type == 'invalidate']:
            skipped.append(Skipped(run_id=run_id, reason='excluded', detail='; '.join(why)))
            continue
        run = runs.get(run_id)
        if run is None:
            skipped.append(Skipped(run_id=run_id, reason='no_run', detail='no such run'))
            continue
        experts, unhashed = _experts(run_id, current, rubric_sha256, skipped)
        no_rubric_hash += unhashed
        if not experts:
            detail = 'no expert passed or failed it; the run is left out'
            skipped.append(Skipped(run_id=run_id, reason='no_passed', detail=detail))
            continue
        answers.append(_majority(run_id, run.solution, experts, skipped))
    return LabeledRuns(
        answers=tuple(answers), skipped=tuple(skipped), no_rubric_hash=no_rubric_hash
    )


def _experts(
    run_id: str, current: Sequence[Label], rubric_sha256: str | None, skipped: list[Skipped]
) -> tuple[dict[str, ExpertLabel], int]:
    """Return each expert's label from their current verdicts on the run, and how many of
    those verdicts record no rubric hash, noting in *skipped* what is left out."""
    values: defaultdict[str, defaultdict[str, set[bool]]] = defaultdict(lambda: defaultdict(set))
    blind: dict[str, bool] = {}
    other_rubric: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    unhashed = 0
    for label in current:
        name, digest = label.name, label.rubric_sha256
        if label.source != 'human' or name is None or not is_verdict(name):
            continue
        if rubric_sha256 is not None and digest is not None and digest != rubric_sha256:
            other_rubric[label.author, digest].append(name)
            continue
        unhashed += digest is None
        values[label.author][name].add(label.value is True)
        blind[label.author] = blind.get(label.author, True) and label.blind is True
    for (author, digest), names in other_rubric.items():
        detail = f'{author} {", ".join(names)} labeled against {digest}'
        skipped.append(Skipped(run_id=run_id, reason='rubric_mismatch', detail=detail))
    experts: dict[str, ExpertLabel] = {}
    for author, marks in values.items():
        for name in (name for name, given in marks.items() if len(given) > 1):
            left_out = f'{author} is left out of the run'
            if name != PASS_MARK:
                left_out = f"it is left out of {author}'s label"
            detail = f'{author} marks {name} both true and false; {left_out}'
            skipped.append(Skipped(run_id=run_id, reason='conflicting_author', detail=detail))
        checks = {name: next(iter(given)) for name, given in marks.items() if len(given) == 1}
        if len(marks.get(PASS_MARK, ())) > 1:  # contradicts its own pass mark
            continue
        if PASS_MARK not in checks:
            detail = f'{author} gave check verdicts but no passed; left out'
            skipped.append(Skipped(run_id=run_id, reason='no_passed', detail=detail))
            continue
        passed = checks.pop(PASS_MARK)
        experts[author] = ExpertLabel(passed=passed, blind=blind[author], checks=checks)
    return experts, unhashed


def _majority(
    run_id: str, answer: str, experts: dict[str, ExpertLabel], skipped: list[Skipped]
) -> LabeledAnswer:
    """Return the answer with the experts' majority on each verdict, noting each tie in
    *skipped*."""
    votes: defaultdict[str, dict[bool, list[str]]] = defaultdict(lambda: {True: [], False: []})
    for author, label in experts.items():
        for name, value in label.marks.items():
            votes[name][value].append(author)
    chosen: dict[str, bool] = {}
    for name, by_value in votes.items():
        yes, no = by_value[True], by_value[False]
        if len(yes) != len(no):
            chosen[name] = len(yes) > len(no)
            continue
        left_out = (
            'passed is None: the answer is left out of overall, blind, assisted and min_kappa'
        )
        if name != PASS_MARK:
            left_out = f'{name} is left out of its checks'
        detail = f'{name}: {", ".join(yes)} true, {", ".join(no)} false; {left_out}'
        skipped.append(Skipped(run_id=run_id, reason='tie', detail=detail))
    return LabeledAnswer(
        id=run_id,
        answer=answer,
        passed=chosen.pop(PASS_MARK, None),
        blind=all(label.blind for label in experts.values()),
        checks=chosen,
        experts=experts,
    )
