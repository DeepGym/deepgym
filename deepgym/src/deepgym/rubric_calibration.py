"""Judge calibration: how often a :class:`~deepgym.rubric.RubricReward` judge agrees with
expert labels, measured before its reward is trusted for training.

:func:`calibrate` grades each held-out, expert-labeled answer several times and keeps every
run in a :class:`CalibrationReport`, whose numbers are all computed from those runs. The
judge passes an answer when its score reaches the threshold. A run without a verdict is
counted by ``error_type`` and left out of every agreement number, never counted as a fail,
as in the reward contract.

Agreement is reported as rates, Cohen's kappa and F1 (:class:`Confusion`), per rubric check
with their averages (:class:`CheckSummary`), and, where two or more experts labeled an
answer, against a human baseline: how well the experts agree with each other, beside how
well the judge agrees with the same experts on the same answers (:class:`HumanBaseline`),
from blind and assisted labels apart. :func:`pair_agreement` compares any two score
sources the same way.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import chain, combinations
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictStr, computed_field, model_validator

from deepgym.integrations.reward import MASKED_WARN_RATE, reward
from deepgym.labels import PASS_MARK
from deepgym.models import RunResult
from deepgym.rubric import RUBRIC_PROMPT_VERSION, Rubric, RubricReward
from deepgym.traces.schema import sha256_text

Marks = Mapping[str, bool]
"""One score source's verdicts on one target by name: ``passed`` and check ids, True for a
pass or MET (as in :mod:`deepgym.labels`)."""

Target = TypeVar('Target', bound=Hashable)

KAPPA_GAP_TOLERANCE = 1e-9
"""Slack on the ``max_kappa_gap`` bar, so float rounding can't fail a gap equal to the bar."""


def _rate(hits: int, total: int) -> float | None:
    """*hits* over *total*, or None when *total* is 0."""
    return hits / total if total else None


def _mean(values: Iterable[float | None]) -> float | None:
    """Mean of the values that aren't None, or None when every one is."""
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


class ExpertLabel(BaseModel):
    """One expert's own label on an answer; the fields are as in :class:`LabeledAnswer`."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    blind: bool
    checks: dict[str, bool] = Field(default_factory=dict)

    @property
    def marks(self) -> dict[str, bool]:
        """``passed`` and ``checks`` as one source's :data:`Marks`."""
        return {PASS_MARK: self.passed, **self.checks}


class LabeledAnswer(BaseModel):
    """One answer an expert labeled, held out from writing the rubric."""

    model_config = ConfigDict(frozen=True)

    id: StrictStr
    answer: str
    passed: bool | None
    """Whether the expert passes the answer. None only when its ``experts`` tie on it: the
    answer is then left out of ``overall``, ``blind``, ``assisted`` and the ``min_kappa``
    bar, while its ``checks`` still count in ``check_agreement`` and its experts' own labels
    in the human baselines."""
    blind: bool
    """True when the expert labeled it without seeing the judge's output."""
    checks: dict[str, bool] = Field(default_factory=dict)
    """The expert's verdict per check id, where given. True means the answer meets the
    requirement as written (for a fatal check: makes the mistake), like the judge's MET."""
    experts: dict[str, ExpertLabel] = Field(default_factory=dict)
    """Each expert's own label by author, where known. An answer two or more experts labeled
    is in the report's human baselines, and ``passed``, ``checks`` and ``blind`` above are
    the one label the judge is measured against, chosen by the caller
    (:func:`deepgym.console.calibration_labels.labeled_answers` takes the majority)."""

    @model_validator(mode='after')
    def _no_pass_mark_only_on_a_tie(self) -> LabeledAnswer:
        """Allow ``passed`` None only when as many experts pass the answer as fail it."""
        votes = Counter(label.passed for label in self.experts.values())
        if self.passed is None and (not votes or votes[True] != votes[False]):
            raise ValueError(f'answer {self.id}: passed may be None only when its experts tie')
        return self


class Confusion(BaseModel):
    """A rater against a reference, such as the judge against an expert over judged runs, or
    one expert against another. Positive is the reference's pass, or MET."""

    model_config = ConfigDict(frozen=True)

    tp: int
    """Both pass."""
    fn: int
    """The reference passes, the rater fails."""
    tn: int
    """Both fail."""
    fp: int
    """The reference fails, the rater passes."""

    @classmethod
    def of(cls, pairs: Iterable[tuple[bool, bool]]) -> Confusion:
        """Tally ``(rater, reference)`` pairs, such as ``(judge, expert)``."""
        n = Counter(pairs)
        return cls(tp=n[True, True], fn=n[False, True], tn=n[False, False], fp=n[True, False])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tpr(self) -> float | None:
        """Share of the reference's passes the rater passes too (recall); None without any."""
        return _rate(self.tp, self.tp + self.fn)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tnr(self) -> float | None:
        """Share of the reference's fails the rater fails too; None without any."""
        return _rate(self.tn, self.tn + self.fp)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def agreement(self) -> float | None:
        """Share of pairs where rater and reference agree; None without any."""
        return _rate(self.tp + self.tn, self.tp + self.fn + self.tn + self.fp)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> float | None:
        """Share of the rater's passes the reference passes too; None without any."""
        return _rate(self.tp, self.tp + self.fp)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def f1(self) -> float | None:
        """F1 of the pass (or MET), the harmonic mean of ``precision`` and ``tpr``; None when
        neither side passes anything."""
        return _rate(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def kappa(self) -> float | None:
        """Cohen's kappa: ``agreement`` corrected for the agreement chance would give, each
        side keeping its own pass rate. 1 is full agreement, 0 what chance gives, below 0
        less than chance.

        None without any pair, and when both sides make one and the same call on every
        pair (all pass or all fail): chance then explains all the agreement, and kappa is
        0/0, not 1.
        """
        tp, fn, tn, fp = self.tp, self.fn, self.tn, self.fp
        return _rate(2 * (tp * tn - fn * fp), (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))


def _size(table: Confusion) -> int:
    """Pairs tallied in *table*."""
    return table.tp + table.fn + table.tn + table.fp


def macro_f1(tables: Iterable[Confusion]) -> float | None:
    """Unweighted mean F1 of the pass or MET call (the positive class) over *tables*, one
    per rubric check, say, leaving out each whose F1 is undefined; None when every one is."""
    return _mean(table.f1 for table in tables)


def pooled_kappa(tables: Iterable[Confusion]) -> float | None:
    """One kappa over several tables, each keeping its own chance agreement: the summed
    agreement less the summed agreement chance gives, over the summed pairs less that.

    A table weighs by its pairs, so one sharing a single answer barely moves it, and one
    table alone gives its own ``kappa``. With tables of one size this is the pooled kappa
    of De Vries et al. (2008), which weighs tables equally; with sizes that differ it is
    not. The order of each table's two sides doesn't matter. None when nothing was
    compared, or chance explains all the agreement in every table.
    """
    agreed = chance = pairs = Fraction(0)
    for table in tables:
        if n := _size(table):
            agreed += table.tp + table.tn
            pass_both = (table.tp + table.fp) * (table.tp + table.fn)
            fail_both = (table.tn + table.fn) * (table.tn + table.fp)
            chance += Fraction(pass_both + fail_both, n)
            pairs += n
    return float((agreed - chance) / (pairs - chance)) if pairs != chance else None


class CheckSummary(BaseModel):
    """Per-check tables averaged over the checks, unweighted, each number over the checks
    where it is defined."""

    model_config = ConfigDict(frozen=True)

    macro_f1: float | None
    """Mean F1 of the MET call (the positive class), not of MET and UNMET averaged."""
    f1_checks: int
    """Checks in ``macro_f1``."""
    mean_kappa: float | None
    """Mean kappa."""
    kappa_checks: int
    """Checks in ``mean_kappa``."""

    @classmethod
    def of(cls, tables: Iterable[Confusion]) -> CheckSummary:
        """Average *tables*, one per check."""
        tables = list(tables)
        return cls(
            macro_f1=macro_f1(tables),
            f1_checks=sum(table.f1 is not None for table in tables),
            mean_kappa=_mean(table.kappa for table in tables),
            kappa_checks=sum(table.kappa is not None for table in tables),
        )


class PairAgreement(BaseModel):
    """One score source against another, over the targets and verdicts both gave."""

    model_config = ConfigDict(frozen=True)

    passed: Confusion
    """On ``passed``."""
    checks: dict[str, Confusion]
    """Per other verdict name both gave on some target (check ids), in check id order."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def check_summary(self) -> CheckSummary:
        """``checks`` averaged."""
        return CheckSummary.of(self.checks.values())


def pair_agreement(
    rater: Mapping[Target, Marks], reference: Mapping[Target, Marks]
) -> PairAgreement:
    """Compare two score sources (a judge, an expert, a verifier) per verdict name, over the
    targets both labeled and, on each, the names both gave.

    Args:
        rater: Its marks per target, such as an answer id.
        reference: The other source's marks per target; its pass is the positive.

    Returns:
        A :class:`Confusion` per verdict name, ``passed`` apart.
    """
    pairs: dict[str, list[tuple[bool, bool]]] = {}
    for target in rater.keys() & reference.keys():
        mine, theirs = rater[target], reference[target]
        for name in mine.keys() & theirs.keys():
            pairs.setdefault(name, []).append((mine[name], theirs[name]))
    passed = Confusion.of(pairs.pop(PASS_MARK, ()))
    in_order = sorted(pairs, key=lambda name: (len(name), name))  # c2 before c10
    return PairAgreement(
        passed=passed, checks={name: Confusion.of(pairs[name]) for name in in_order}
    )


class Consistency(BaseModel):
    """Answers whose judged runs all gave the same call, out of those judged at least twice
    (one run can't disagree with itself)."""

    model_config = ConfigDict(frozen=True)

    same: int
    answers: int

    @classmethod
    def of(cls, calls: Iterable[Sequence[object]]) -> Consistency:
        """Count, per answer, whether its judged runs' *calls* all agree."""
        repeated = [set(each) for each in calls if len(each) >= 2]
        return cls(same=sum(len(each) == 1 for each in repeated), answers=len(repeated))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def share(self) -> float | None:
        """``same`` over ``answers``; None without any."""
        return _rate(self.same, self.answers)


class JudgeRun(BaseModel):
    """One judge run of an answer."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    score: float | None
    """None without a verdict."""
    verdicts: dict[str, Literal['MET', 'UNMET']] | None
    """Check id to MET or UNMET; None without a verdict, or for an answer too long for the
    judge (a judged 0 with no check verdicts)."""
    error_type: str | None


class AnswerRow(LabeledAnswer):
    """A labeled answer with each of its judge runs."""

    runs: tuple[JudgeRun, ...]


class BarCheck(BaseModel):
    """Whether a judge reached a bar, with one reason per rate."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    reasons: tuple[str, ...]


def _refuse_repeated(ids: Iterable[str]) -> None:
    """Raise ValueError naming each id given more than once."""
    if repeated := sorted(i for i, n in Counter(ids).items() if n > 1):
        raise ValueError(f'answer ids must be unique; given more than once: {repeated}')


def _passed(pairs: dict[str, dict[str, PairAgreement]]) -> list[Confusion]:
    """The ``passed`` table of each pair in *pairs*."""
    return [pair.passed for row in pairs.values() for pair in row.values()]


class HumanBaseline(BaseModel):
    """How well the experts agree with each other, and the judge with the same experts on
    the same answers, over the answers two or more experts labeled (all blind, or all
    assisted) that have a judged run.

    Each pair of experts is compared on the answers both labeled, and the judge against each
    of the two on the judged runs of those answers. Each side's tables are pooled
    (:func:`pooled_kappa`), weighing each by its comparisons, so a pair sharing one answer
    barely moves the numbers. On the experts' side an answer counts once per pair that
    labeled it; on the judge's side each judged run counts once per expert per pair, so an
    answer with fewer judged runs weighs less.
    """

    model_config = ConfigDict(frozen=True)

    answers: int
    """Answers in the baseline."""
    experts: dict[str, dict[str, PairAgreement]]
    """``experts[a][b]``: expert *a* against expert *b* over the answers both labeled, for
    each pair of experts sharing one, *a* first by name."""
    judge: dict[str, dict[str, PairAgreement]]
    """``judge[e][o]``: the judge against expert *e* over the judged runs of the answers *e*
    and *o* both labeled."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def expert_comparisons(self) -> int:
        """Comparisons of two experts' ``passed``: one per pair per answer both labeled."""
        return sum(map(_size, _passed(self.experts)))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def expert_agreement(self) -> float | None:
        """Share of ``expert_comparisons`` that agree."""
        return _rate(sum(t.tp + t.tn for t in _passed(self.experts)), self.expert_comparisons)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def expert_kappa(self) -> float | None:
        """:func:`pooled_kappa` of the pairs of experts on ``passed``."""
        return pooled_kappa(_passed(self.experts))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def judge_comparisons(self) -> int:
        """Comparisons of a judged run's pass with an expert's ``passed``."""
        return sum(map(_size, _passed(self.judge)))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def judge_agreement(self) -> float | None:
        """Share of ``judge_comparisons`` that agree."""
        return _rate(sum(t.tp + t.tn for t in _passed(self.judge)), self.judge_comparisons)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def judge_kappa(self) -> float | None:
        """:func:`pooled_kappa` of the judge against the experts on ``passed``."""
        return pooled_kappa(_passed(self.judge))


class CalibrationReport(BaseModel):
    """Every judge run of every labeled answer, and the agreement numbers computed from them."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    judge_id: str
    """The judge model and its settings, as the caller named them."""
    prompt_version: str
    """:data:`~deepgym.rubric.RUBRIC_PROMPT_VERSION` the judge ran with."""
    rubric_sha256: str
    """SHA-256 of the task and the rubric's checks as canonical JSON (see
    :func:`rubric_sha256`): what the judge is shown besides the answer."""
    check_ids: tuple[str, ...]
    runs: int
    """Judge runs per answer."""
    threshold: float
    """The judge passes an answer whose score is at least this."""
    answers: tuple[AnswerRow, ...]

    @model_validator(mode='after')
    def _answer_ids_are_unique(self) -> CalibrationReport:
        """Each answer is in the report once."""
        _refuse_repeated(row.id for row in self.answers)
        return self

    @model_validator(mode='after')
    def _verdicts_cover_every_check(self) -> CalibrationReport:
        """A run's check verdicts, when it has them, name exactly the report's checks."""
        expected = set(self.check_ids)
        for row in self.answers:
            for run in row.runs:
                if run.verdicts is not None and set(run.verdicts) != expected:
                    raise ValueError(
                        f'answer {row.id}: a run has verdicts for {sorted(run.verdicts)}, '
                        f'not the checks {sorted(expected)}'
                    )
        return self

    def _confusion(self, blind: bool | None) -> Confusion:
        """Judge pass against expert pass over judged runs, of answers with an expert pass
        mark labeled *blind* (any when None)."""
        return Confusion.of(
            (run.score >= self.threshold, row.passed)
            for row in self.answers
            if row.passed is not None and (blind is None or row.blind is blind)
            for run in row.runs
            if run.score is not None
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def judged_runs(self) -> int:
        """Runs with a verdict."""
        return sum(run.score is not None for row in self.answers for run in row.runs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def no_verdict_runs(self) -> int:
        """Runs without a verdict, left out of every agreement number."""
        return sum(run.score is None for row in self.answers for run in row.runs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def no_verdict_by_type(self) -> dict[str, int]:
        """Runs without a verdict per ``error_type``."""
        return dict(
            Counter(
                str(run.error_type) for row in self.answers for run in row.runs if run.score is None
            )
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tied_answers(self) -> int:
        """Answers whose experts tie on ``passed``, left out of ``overall``, ``blind``,
        ``assisted`` and the ``min_kappa`` bar."""
        return sum(row.passed is None for row in self.answers)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overall(self) -> Confusion:
        """Over every labeled answer."""
        return self._confusion(None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blind(self) -> Confusion:
        """Over answers the expert labeled without seeing the judge's output."""
        return self._confusion(True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def assisted(self) -> Confusion:
        """Over answers the expert labeled after seeing the judge's output."""
        return self._confusion(False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def self_consistency(self) -> Consistency:
        """Answers whose judged runs all pass or all fail."""
        return Consistency.of(
            [run.score >= self.threshold for run in row.runs if run.score is not None]
            for row in self.answers
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def check_consistency(self) -> dict[str, Consistency]:
        """Per check, answers whose runs with check verdicts all gave the same one."""
        return {
            cid: Consistency.of(
                [run.verdicts[cid] for run in row.runs if run.verdicts and run.score is not None]
                for row in self.answers
            )
            for cid in self.check_ids
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def check_agreement(self) -> dict[str, Confusion]:
        """Per check an expert labeled, the judge's MET against the expert's, over runs with
        check verdicts."""
        return {
            cid: Confusion.of(
                (run.verdicts[cid] == 'MET', row.checks[cid])
                for row in self.answers
                if cid in row.checks
                for run in row.runs
                if run.verdicts and run.score is not None
            )
            for cid in self.check_ids
            if any(cid in row.checks for row in self.answers)
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def check_summary(self) -> CheckSummary:
        """``check_agreement`` averaged over the checks: MET-class macro F1 and mean kappa."""
        return CheckSummary.of(self.check_agreement.values())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blind_baseline(self) -> HumanBaseline | None:
        """From blind expert labels, the baseline ``max_kappa_gap`` checks; None without
        any answer it can use."""
        return self._baseline(blind=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def assisted_baseline(self) -> HumanBaseline | None:
        """From expert labels given after seeing the judge's output, which may lean toward
        it; None without any answer it can use."""
        return self._baseline(blind=False)

    def _baseline(self, blind: bool) -> HumanBaseline | None:
        """The human baseline from expert labels whose blind flag is *blind*."""
        experts: dict[str, dict[str, dict[str, bool]]] = {}  # answer -> author -> marks
        judged: dict[str, list[dict[str, bool]]] = {}  # answer -> each judged run's marks
        for row in self.answers:
            labels = {a: e.marks for a, e in row.experts.items() if e.blind is blind}
            runs = [
                {
                    PASS_MARK: run.score >= self.threshold,
                    **{cid: verdict == 'MET' for cid, verdict in (run.verdicts or {}).items()},
                }
                for run in row.runs
                if run.score is not None
            ]
            if len(labels) >= 2 and runs:
                experts[row.id], judged[row.id] = labels, runs
        if not experts:
            return None
        shared: dict[tuple[str, str], list[str]] = {}
        for answer_id, labels in experts.items():
            for pair in combinations(sorted(labels), 2):
                shared.setdefault(pair, []).append(answer_id)
        pairs: dict[str, dict[str, PairAgreement]] = {}
        judge: dict[str, dict[str, PairAgreement]] = {}
        for (a, b), ids in sorted(shared.items()):
            pairs.setdefault(a, {})[b] = pair_agreement(
                {i: experts[i][a] for i in ids}, {i: experts[i][b] for i in ids}
            )
            runs_of = {(i, k): marks for i in ids for k, marks in enumerate(judged[i])}
            for e, o in ((a, b), (b, a)):
                judge.setdefault(e, {})[o] = pair_agreement(
                    runs_of, {key: experts[key[0]][e] for key in runs_of}
                )
        return HumanBaseline(answers=len(experts), experts=pairs, judge=judge)

    def to_json(self) -> str:
        """The report as JSON, computed numbers included; ``model_validate_json`` reads it back."""
        return self.model_dump_json(indent=2)

    def meets(
        self,
        min_tpr: float,
        min_tnr: float,
        max_no_verdict: float = MASKED_WARN_RATE,
        *,
        min_kappa: float | None = None,
        max_kappa_gap: float | None = None,
        min_baseline_answers: int | None = None,
    ) -> BarCheck:
        """Whether the overall TPR and TNR reach *min_tpr* and *min_tnr*, with no more than
        *max_no_verdict* of all runs left without a verdict (by default the share above
        which reward functions warn). A rate with no judged run to measure it fails.

        Two more bars apply only when given: *min_kappa* on the overall kappa (the judge
        against each answer's one expert pass mark, tied answers left out), and
        *max_kappa_gap*, how far the judge's pooled kappa with the experts may fall below
        the experts' with each other, from blind labels only (``blind_baseline``), within
        :data:`KAPPA_GAP_TOLERANCE`. An undefined kappa fails its bar, and so does a
        missing baseline. The gap bar also fails when the baseline is uninformative (the
        experts' kappa is undefined or at most 0: no agreement beyond chance to come close
        to) and, with *min_baseline_answers*, when it has fewer answers than that.

        Tied answers (``tied_answers``) are left out of TPR, TNR and kappa, and their
        reasons say how many.

        Raises:
            ValueError: When a bar isn't a finite number, or *min_baseline_answers* is
                below 1 or given without *max_kappa_gap*.
        """
        given = [bar for bar in (min_kappa, max_kappa_gap) if bar is not None]
        if not all(math.isfinite(bar) for bar in (min_tpr, min_tnr, max_no_verdict, *given)):
            raise ValueError(
                f'bars must be finite numbers, got {min_tpr=}, {min_tnr=}, {max_no_verdict=}, '
                f'{min_kappa=}, {max_kappa_gap=}'
            )
        if min_baseline_answers is not None and (max_kappa_gap is None or min_baseline_answers < 1):
            raise ValueError(
                'min_baseline_answers is a count of at least 1 for the max_kappa_gap bar, got '
                f'{min_baseline_answers=}, {max_kappa_gap=}'
            )
        overall = self.overall
        total_runs = self.judged_runs + self.no_verdict_runs
        share = self.no_verdict_runs / total_runs if total_runs else 0.0
        passed = share <= max_no_verdict
        reasons = [
            f'no verdict {self.no_verdict_runs}/{total_runs} ({share:.3f}) '
            + ('is within ' if passed else 'is above ')
            + str(max_no_verdict)
        ]
        ties = self._ties_left_out()
        for name, hits, total, bar, labeled in (
            ('TPR', overall.tp, overall.tp + overall.fn, min_tpr, 'passed'),
            ('TNR', overall.tn, overall.tn + overall.fp, min_tnr, 'failed'),
        ):
            if not total:
                passed = False
                reasons.append(
                    f'{name} is missing: no judged run of an answer the expert {labeled}{ties}'
                )
            elif hits / total < bar:
                passed = False
                reasons.append(f'{name} {hits}/{total} ({hits / total:.3f}) is below {bar}{ties}')
            else:
                reasons.append(f'{name} {hits}/{total} ({hits / total:.3f}) reaches {bar}{ties}')
        optional = []
        if min_kappa is not None:
            optional.append(self._kappa_bar(min_kappa))
        if max_kappa_gap is not None:
            optional.append(self._gap_bar(max_kappa_gap, min_baseline_answers))
        for reached, reason in optional:
            passed = passed and reached
            reasons.append(reason)
        return BarCheck(passed=passed, reasons=tuple(reasons))

    def _ties_left_out(self) -> str:
        """A note naming the tied answers left out, for a bar's reason; empty without any."""
        tied = self.tied_answers
        return f'; {tied} tied answer{"" if tied == 1 else "s"} left out' if tied else ''

    def _kappa_bar(self, min_kappa: float) -> tuple[bool, str]:
        """Whether the overall kappa reaches *min_kappa*, and why."""
        overall, ties = self.overall, self._ties_left_out()
        if overall.kappa is None:
            why = (
                'the judge and the experts made one and the same call on every judged run'
                if _size(overall)
                else 'no judged run of an answer with an expert pass mark'
            )
            return False, f'kappa is undefined: {why}{ties}'
        over = f'over {_size(overall)} judged runs'
        if overall.kappa < min_kappa:
            return False, f'kappa {overall.kappa:.3f} {over} is below {min_kappa}{ties}'
        return True, f'kappa {overall.kappa:.3f} {over} reaches {min_kappa}{ties}'

    def _gap_bar(self, max_gap: float, min_answers: int | None) -> tuple[bool, str]:
        """Whether the judge's pooled kappa with the blind experts is within *max_gap* below
        the experts' with each other, on a baseline of at least *min_answers* answers where
        the experts agree beyond chance, and why."""
        baseline = self.blind_baseline
        if baseline is None:
            return False, (
                'blind baseline is missing: no answer with a judged run that two or more '
                'experts labeled blind'
            )
        experts, judge = baseline.expert_kappa, baseline.judge_kappa
        counts = (
            f'{baseline.judge_comparisons} judged-run comparisons; experts '
            f'{baseline.expert_comparisons} comparisons on {baseline.answers} answers'
        )
        if min_answers is not None and baseline.answers < min_answers:
            return False, (
                f'blind baseline has {baseline.answers} answers, fewer than {min_answers} '
                f'({counts})'
            )
        if experts is None or judge is None or experts <= 0:  # judge is None only if experts is
            kappa = 'is undefined' if experts is None else f'{experts:.3f} is not above 0'
            return False, (
                f"blind baseline is uninformative: the experts' kappa {kappa}, no agreement "
                f'beyond chance to come close to ({counts})'
            )
        if judge < experts - max_gap - KAPPA_GAP_TOLERANCE:
            return False, (
                f"judge kappa {judge:.3f} is more than {max_gap} below the experts' "
                f'{experts:.3f} (blind; {counts})'
            )
        return True, (
            f"judge kappa {judge:.3f} is within {max_gap} of the experts' {experts:.3f} "
            f'(blind; {counts})'
        )


def rubric_sha256(task: str, rubric: Rubric) -> str:
    """SHA-256 of *task* and *rubric*'s checks as JSON with sorted keys and no spaces."""
    data = {'task': task, 'checks': rubric.model_dump(mode='json')['checks']}
    return sha256_text(json.dumps(data, sort_keys=True, separators=(',', ':')))


def _judge_run(result: RunResult) -> JudgeRun:
    return JudgeRun(
        score=reward(result),
        verdicts=(result.metrics or {}).get('verdicts'),
        error_type=result.error_type,
    )


def calibrate(
    reward_fn: RubricReward,
    answers: Sequence[LabeledAnswer],
    *,
    judge_id: str,
    built_from: Iterable[str],
    runs: int = 3,
    threshold: float = 1.0,
) -> CalibrationReport:
    """Grade each held-out answer *runs* times with *reward_fn* and compare with the experts.

    Judge calls run in parallel, at most the reward's ``max_parallel`` at once.

    Args:
        reward_fn: The task, rubric and judge to calibrate.
        answers: Expert-labeled answers, none of them used to write the rubric.
        judge_id: The judge model and its settings, kept in the report.
        built_from: Ids of the answers used to write the rubric.
        runs: Judge calls per answer, for self-consistency.
        threshold: The judge passes an answer whose score is at least this.

    Raises:
        ValueError: Before any judge call, when an answer's id is in *built_from*, two
            answers share an id, an expert labels a check the rubric doesn't have, or
            *runs* is below 1, or *threshold* isn't finite.
        pydantic.ValidationError: Before any judge call, when an answer isn't valid, such
            as one changed with ``model_copy``, which skips validation.
        TypeError: When *built_from* is one string, or holds an id that isn't a string.
    """
    if runs < 1:
        raise ValueError(f'runs must be at least 1, got {runs}')
    if not math.isfinite(threshold):
        raise ValueError(f'threshold must be a finite number, got {threshold}')
    if isinstance(built_from, str):  # its characters would be checked, not the id
        raise TypeError(f'built_from is a list of answer ids, got the string {built_from!r}')
    built = set(built_from)
    if not_ids := [i for i in built if not isinstance(i, str)]:
        raise TypeError(f'built_from ids must be strings, got {not_ids!r}')
    fields = set(LabeledAnswer.model_fields)
    answers = [LabeledAnswer.model_validate(a.model_dump(include=fields)) for a in answers]
    _refuse_repeated(answer.id for answer in answers)
    if used := sorted({answer.id for answer in answers} & built):
        raise ValueError(
            f'answers used to write the rubric can not validate it; hold them out: {used}'
        )
    check_ids = reward_fn.rubric.ids
    if unknown := [
        f'{a.id}: {c}'
        for a in answers
        for c in dict.fromkeys(chain(a.checks, *(e.checks for e in a.experts.values())))
        if c not in check_ids
    ]:
        raise ValueError(f'expert labels for checks the rubric does not have: {unknown}')
    texts = [answer.answer for answer in answers for _ in range(runs)]
    with ThreadPoolExecutor(max_workers=reward_fn.max_parallel) as pool:
        results = [_judge_run(result) for result in pool.map(reward_fn.grade, texts)]
    rows = tuple(
        AnswerRow(
            **answer.model_dump(include=fields),
            runs=tuple(results[i * runs : (i + 1) * runs]),
        )
        for i, answer in enumerate(answers)
    )
    return CalibrationReport(
        judge_id=judge_id,
        prompt_version=RUBRIC_PROMPT_VERSION,
        rubric_sha256=rubric_sha256(reward_fn.task, reward_fn.rubric),
        check_ids=check_ids,
        runs=runs,
        threshold=threshold,
        answers=rows,
    )
