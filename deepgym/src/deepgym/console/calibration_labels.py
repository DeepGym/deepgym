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
from deepgym.labels import PASS_MARK, Label, Name, is_verdict
from deepgym.rubric_calibration import (
    AnswerRow,
    CalibrationReport,
    CheckSummary,
    Confusion,
    Consistency,
    ExpertLabel,
    HumanBaseline,
    JudgeRun,
    LabeledAnswer,
)

SkipReason = Literal[
    'excluded', 'no_run', 'rubric_mismatch', 'no_passed', 'conflicting_author', 'tie',
    'no_judge', 'missing_checks',
]  # fmt: skip


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
    stays in ``experts`` for the human baselines.

    Only :func:`stored_agreement` gives these, about the judge's stored verdicts:
    ``rubric_mismatch`` as for an expert, and for verdicts recorded with another
    ``rubric_version`` than the one asked for (or none); ``no_passed`` per judge call
    (``group_id``) without one;
    ``no_judge``: the judge gave no ``passed`` on the run, which is left out;
    ``missing_checks``: a judge call gave some check ids but not every one the judge gave
    on the runs, so its check verdicts are left out and its ``passed`` kept."""
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


class StoredJudge(BaseModel):
    """Whose stored verdicts stand in for a judge's runs: one source and author."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    author: Name
    source: Literal['judge', 'verifier'] = 'judge'
    rubric_version: Name | None = None
    """Only its verdicts recorded with this ``rubric_version``, such as a judge prompt
    version; None takes every one."""


class StoredAgreement(BaseModel):
    """A judge's stored verdicts against the experts', as a
    :class:`~deepgym.rubric_calibration.CalibrationReport` computes them: each judge call
    (its verdicts sharing a ``group_id``) is one judged run, passing when its ``passed`` is
    true, and each check id it marks true is MET.

    Without ``rubric_sha256``, verdicts against different rubric hashes are pooled: read
    ``rubric_hashes`` and ``judge_rubric_hashes`` to see whether they span rubric versions.
    """

    model_config = ConfigDict(frozen=True)

    judge: StoredJudge
    rubric_sha256: str | None
    check_ids: tuple[str, ...]
    """The check ids the judge gave on the runs."""
    expert_only_checks: tuple[str, ...]
    """Check ids the experts gave on the runs kept that the judge never gave: in no
    judge-versus-expert number, only in the human baselines."""
    answers: tuple[AnswerRow, ...]
    """Each run left in, with the experts' labels and the judge's calls."""
    overall: Confusion
    blind: Confusion
    assisted: Confusion
    check_agreement: dict[str, Confusion]
    check_summary: CheckSummary
    self_consistency: Consistency
    check_consistency: dict[str, Consistency]
    blind_baseline: HumanBaseline | None
    assisted_baseline: HumanBaseline | None
    tied_answers: int
    judged_calls: int
    """Judge calls in ``answers``."""
    no_passed_calls: int
    """Judge calls left out for giving no ``passed``."""
    no_judge_runs: int
    """Runs left out for having no judge call with ``passed``."""
    skipped: tuple[Skipped, ...]
    rubric_hashes: dict[str, int]
    """As in :class:`LabeledRuns`: the rubric hashes on the experts' verdicts used."""
    no_rubric_hash: int
    judge_rubric_hashes: dict[str, int]
    """Each rubric hash on the judge's verdicts used, with how many."""
    judge_no_rubric_hash: int
    judge_rubric_versions: dict[str, int]
    """Each ``rubric_version`` on the judge's verdicts used, with how many: more than one
    means its calls span judge versions."""
    judge_no_rubric_version: int


_REPORTED = (
    'check_ids', 'answers', 'overall', 'blind', 'assisted', 'check_agreement', 'check_summary',
    'self_consistency', 'check_consistency', 'blind_baseline', 'assisted_baseline',
    'tied_answers',
)  # fmt: skip

_Call = dict[str, Label]
"""One judge call's verdicts on a run, by name."""


def stored_agreement(
    labels: LabelStore,
    runs: RunStore,
    run_ids: Iterable[str],
    judge: StoredJudge,
    *,
    rubric_sha256: str | None = None,
) -> StoredAgreement:
    """Measure *judge*'s current verdicts on *run_ids* against the experts', calling no model.

    The experts' side is :func:`labeled_answers`. The judge's is its current verdicts on
    each run, grouped into calls by ``group_id``; with *rubric_sha256*, those recorded
    against another rubric hash are left out, as the experts' are, and with
    ``judge.rubric_version``, those recorded with another version or none. A call without
    ``passed`` is left out, and a run with no call left is too; each is listed in
    ``skipped`` with the reason, in the order of the runs.

    Args:
        labels: The label store.
        runs: The run store holding the runs' solutions.
        run_ids: The runs, as :func:`labeled_answers` takes them.
        judge: Whose verdicts are the judge's.
        rubric_sha256: The hash of the task and rubric the verdicts should be against.

    Returns:
        The agreement numbers, every run kept with its labels, and what was left out.
    """
    position = {run_id: at for at, run_id in enumerate(dict.fromkeys(run_ids))}
    labeled = labeled_answers(labels, runs, position, rubric_sha256=rubric_sha256)
    skipped = list(labeled.skipped)
    calls: dict[str, list[_Call]] = {}
    no_passed = 0
    for answer in labeled.answers:
        every = _judge_calls(answer.id, labels, judge, rubric_sha256, skipped)
        calls[answer.id] = [call for call in every if PASS_MARK in call]
        for call in (call for call in every if PASS_MARK not in call):
            group_id = next(iter(call.values())).group_id
            detail = f'{judge.author} call {group_id} gave check verdicts but no passed; left out'
            skipped.append(Skipped(run_id=answer.id, reason='no_passed', detail=detail))
            no_passed += 1
    named = {name for each in calls.values() for call in each for name in call} - {PASS_MARK}
    check_ids = tuple(sorted(named, key=_check_order))
    rows: list[AnswerRow] = []
    used: list[Label] = []
    for answer in labeled.answers:
        if not calls[answer.id]:
            detail = f'{judge.source} {judge.author} gave no passed on it; the run is left out'
            skipped.append(Skipped(run_id=answer.id, reason='no_judge', detail=detail))
            continue
        judged = [_judge_run(answer.id, call, check_ids, skipped) for call in calls[answer.id]]
        rows.append(AnswerRow(**answer.model_dump(), runs=tuple(run for run, _ in judged)))
        used += [label for _, verdicts in judged for label in verdicts]
    # Only the computed numbers are read: the metadata fields describe a live judge run.
    report = CalibrationReport(
        judge_id=judge.author,
        prompt_version='',
        rubric_sha256=rubric_sha256 or '',
        check_ids=check_ids,
        runs=0,
        threshold=1.0,
        answers=tuple(rows),
    )
    expert_checks = {
        name for row in rows for expert in row.experts.values() for name in expert.checks
    }
    hashes = Counter(label.rubric_sha256 for label in used)
    versions = Counter(label.rubric_version for label in used)
    return StoredAgreement(
        judge=judge,
        rubric_sha256=rubric_sha256,
        **{name: getattr(report, name) for name in _REPORTED},
        expert_only_checks=tuple(sorted(expert_checks - named, key=_check_order)),
        judged_calls=report.judged_runs,
        no_passed_calls=no_passed,
        no_judge_runs=sum(skip.reason == 'no_judge' for skip in skipped),
        skipped=tuple(sorted(skipped, key=lambda skip: position[skip.run_id])),
        rubric_hashes=labeled.rubric_hashes,
        no_rubric_hash=labeled.no_rubric_hash,
        judge_rubric_hashes=_known(hashes),
        judge_no_rubric_hash=hashes[None],
        judge_rubric_versions=_known(versions),
        judge_no_rubric_version=versions[None],
    )


def _check_order(name: str) -> tuple[int, str]:
    """Sort key putting c2 before c10."""
    return len(name), name


def _known(counts: Counter[str | None]) -> dict[str, int]:
    """*counts* without the None key, sorted."""
    return {key: counts[key] for key in sorted(filter(None, counts))}


def _judge_calls(
    run_id: str,
    labels: LabelStore,
    judge: StoredJudge,
    rubric_sha256: str | None,
    skipped: list[Skipped],
) -> list[_Call]:
    """Return the judge's current verdicts on *run_id*, one call per ``group_id``, noting in
    *skipped* those another rubric hash or version leaves out."""
    calls: dict[str | None, _Call] = {}
    left_out: list[str] = []
    for label in labels.current('run', run_id):
        name, digest, version = label.name, label.rubric_sha256, label.rubric_version
        mine = (label.source, label.author) == (judge.source, judge.author)
        if not mine or name is None or not is_verdict(name):
            continue
        if rubric_sha256 is not None and digest is not None and digest != rubric_sha256:
            left_out.append(f'{name} against {digest}')
        elif judge.rubric_version is not None and version != judge.rubric_version:
            left_out.append(f'{name} with rubric_version {version}')
        else:
            calls.setdefault(label.group_id, {})[name] = label
    if left_out:
        detail = f'{judge.author} labeled {", ".join(left_out)}'
        skipped.append(Skipped(run_id=run_id, reason='rubric_mismatch', detail=detail))
    return list(calls.values())


def _judge_run(
    run_id: str, call: _Call, check_ids: tuple[str, ...], skipped: list[Skipped]
) -> tuple[JudgeRun, list[Label]]:
    """Return one judge call as a judged run, with the verdicts it uses, noting in
    *skipped* check verdicts left out."""
    checks = {name: label for name, label in call.items() if name != PASS_MARK}
    complete = checks.keys() == set(check_ids)
    if checks and not complete:
        detail = f'a call gave {sorted(checks)}, not every check {list(check_ids)}; left out'
        skipped.append(Skipped(run_id=run_id, reason='missing_checks', detail=detail))
    verdicts: dict[str, Literal['MET', 'UNMET']] = {
        name: 'MET' if label.value is True else 'UNMET' for name, label in checks.items()
    }
    run = JudgeRun(
        score=1.0 if call[PASS_MARK].value is True else 0.0,
        verdicts=verdicts if complete else None,
        error_type=None,
    )
    return run, [call[PASS_MARK], *(checks.values() if complete else ())]
