"""Judge calibration input from the console's expert labels on runs.

:func:`labeled_answers` turns runs that experts labeled (see :mod:`deepgym.labels`) into
:class:`~deepgym.rubric_calibration.LabeledAnswer` items for
:func:`~deepgym.rubric_calibration.calibrate`, and lists what it left out and why.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
    rubric_hashes: dict[str, int]
    """Each rubric hash recorded on the verdicts behind ``answers``, with how many: more
    than one means the labels span rubric versions."""
    no_rubric_hash: int
    """Verdicts behind ``answers`` that record no rubric hash, so can't be checked against
    ``rubric_sha256``. Only verdicts in an expert label that made it into an answer count."""


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
    out; those recording none are kept. Either way the result counts the rubric hashes on
    the verdicts used (``rubric_hashes``, ``no_rubric_hash``). Check ids are positions in
    the rubric, so editing a rubric renumbers them: a ``c2`` labeled before the edit may
    be another requirement now.

    The label the judge is measured against is the experts' majority, on ``passed`` and on
    each check, blind only when every expert's is. On a tie ``passed`` is None, or the
    check is left out. Every expert's own label is in ``experts``; to choose the judge's
    reference another way, rebuild the answers with ``LabeledAnswer.model_validate``.

    Args:
        labels: The label store.
        runs: The run store holding the runs' solutions.
        run_ids: The runs, in the order the answers should take; a run given twice is
            read once. For the runs a label
            query matches, pass ``dict.fromkeys(label.target.id for label in
            labels.page(query, 0, None).labels)`` with ``target_kind='run'`` in the query.
        rubric_sha256: The hash of the task and rubric being calibrated
            (``rubric_calibration.rubric_sha256(reward_fn.task, reward_fn.rubric)``).

    Returns:
        The answers, and each run, expert or verdict left out, with the reason.
    """
    answers: list[LabeledAnswer] = []
    skipped: list[Skipped] = []
    hashes: Counter[str | None] = Counter()
    for run_id in dict.fromkeys(run_ids):
        current = labels.current('run', run_id)
        if why := [label.comment or '' for label in current if label.type == 'invalidate']:
            skipped.append(Skipped(run_id=run_id, reason='excluded', detail='; '.join(why)))
            continue
        run = runs.get(run_id)
        if run is None:
            skipped.append(Skipped(run_id=run_id, reason='no_run', detail='no such run'))
            continue
        experts, used = _experts(run_id, current, rubric_sha256, skipped)
        if not experts:
            detail = 'no expert passed or failed it; the run is left out'
            skipped.append(Skipped(run_id=run_id, reason='no_passed', detail=detail))
            continue
        answers.append(_majority(run_id, run.solution, experts, skipped))
        hashes.update(used)
    return LabeledRuns(
        answers=tuple(answers),
        skipped=tuple(skipped),
        rubric_hashes={digest: hashes[digest] for digest in sorted(filter(None, hashes))},
        no_rubric_hash=hashes[None],
    )


def _experts(
    run_id: str, current: Sequence[Label], rubric_sha256: str | None, skipped: list[Skipped]
) -> tuple[dict[str, ExpertLabel], Counter[str | None]]:
    """Return each expert's label from their current verdicts on the run, and the rubric
    hash (or None) of each verdict in those labels, noting in *skipped* what is left out."""
    given: defaultdict[str, defaultdict[str, list[tuple[bool, str | None]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    blind: dict[str, bool] = {}
    other_rubric: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for label in current:
        name, digest = label.name, label.rubric_sha256
        if label.source != 'human' or name is None or not is_verdict(name):
            continue
        if rubric_sha256 is not None and digest is not None and digest != rubric_sha256:
            other_rubric[label.author, digest].append(name)
            continue
        given[label.author][name].append((label.value is True, digest))
        blind[label.author] = blind.get(label.author, True) and label.blind is True
    for (author, digest), names in other_rubric.items():
        detail = f'{author} {", ".join(names)} labeled against {digest}'
        skipped.append(Skipped(run_id=run_id, reason='rubric_mismatch', detail=detail))
    experts: dict[str, ExpertLabel] = {}
    used: Counter[str | None] = Counter()
    for author, records in given.items():
        marks = {name: {value for value, _ in each} for name, each in records.items()}
        for name in (name for name, values in marks.items() if len(values) > 1):
            left_out = f'{author} is left out of the run'
            if name != PASS_MARK:
                left_out = f"it is left out of {author}'s label"
            detail = f'{author} marks {name} both true and false; {left_out}'
            skipped.append(Skipped(run_id=run_id, reason='conflicting_author', detail=detail))
        checks = {name: next(iter(values)) for name, values in marks.items() if len(values) == 1}
        if len(marks.get(PASS_MARK, ())) > 1:  # contradicts its own pass mark
            continue
        if PASS_MARK not in checks:
            detail = f'{author} gave check verdicts but no passed; left out'
            skipped.append(Skipped(run_id=run_id, reason='no_passed', detail=detail))
            continue
        used.update(digest for name in checks for _, digest in records[name])
        passed = checks.pop(PASS_MARK)
        experts[author] = ExpertLabel(passed=passed, blind=blind[author], checks=checks)
    return experts, used


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
