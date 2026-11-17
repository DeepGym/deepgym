"""Judge calibration: how often a :class:`~deepgym.rubric.RubricReward` judge agrees with
expert labels, measured before its reward is trusted for training.

:func:`calibrate` grades each held-out, expert-labeled answer several times and keeps every
run in a :class:`CalibrationReport`, whose numbers are all computed from those runs. The
judge passes an answer when its score reaches the threshold. A run without a verdict is
counted by ``error_type`` and left out of every agreement number, never counted as a fail,
as in the reward contract.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, computed_field, model_validator

from deepgym.integrations.reward import MASKED_WARN_RATE, reward
from deepgym.models import RunResult
from deepgym.rubric import RUBRIC_PROMPT_VERSION, Rubric, RubricReward
from deepgym.traces.schema import sha256_text


def _rate(hits: int, total: int) -> float | None:
    """*hits* over *total*, or None when *total* is 0."""
    return hits / total if total else None


class LabeledAnswer(BaseModel):
    """One answer an expert labeled, held out from writing the rubric."""

    model_config = ConfigDict(frozen=True)

    id: StrictStr
    answer: str
    passed: bool
    """Whether the expert passes the answer."""
    blind: bool
    """True when the expert labeled it without seeing the judge's output."""
    checks: dict[str, bool] = Field(default_factory=dict)
    """The expert's verdict per check id, where given. True means the answer meets the
    requirement as written (for a fatal check: makes the mistake), like the judge's MET."""


class Confusion(BaseModel):
    """Judge against expert over judged runs. Positive is the expert's pass, or MET."""

    model_config = ConfigDict(frozen=True)

    tp: int
    """Both pass."""
    fn: int
    """The expert passes, the judge fails."""
    tn: int
    """Both fail."""
    fp: int
    """The expert fails, the judge passes."""

    @classmethod
    def of(cls, pairs: Iterable[tuple[bool, bool]]) -> Confusion:
        """Tally ``(judge, expert)`` pairs."""
        n = Counter(pairs)
        return cls(tp=n[True, True], fn=n[False, True], tn=n[False, False], fp=n[True, False])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tpr(self) -> float | None:
        """Share of the expert's passes the judge passes too; None without any."""
        return _rate(self.tp, self.tp + self.fn)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tnr(self) -> float | None:
        """Share of the expert's fails the judge fails too; None without any."""
        return _rate(self.tn, self.tn + self.fp)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def agreement(self) -> float | None:
        """Share of runs where judge and expert agree; None without any."""
        return _rate(self.tp + self.tn, self.tp + self.fn + self.tn + self.fp)


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
        """Judge pass against expert pass over judged runs, of answers labeled *blind* (any
        when None)."""
        return Confusion.of(
            (run.score >= self.threshold, row.passed)
            for row in self.answers
            if blind is None or row.blind is blind
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

    def to_json(self) -> str:
        """The report as JSON, computed numbers included; ``model_validate_json`` reads it back."""
        return self.model_dump_json(indent=2)

    def meets(
        self, min_tpr: float, min_tnr: float, max_no_verdict: float = MASKED_WARN_RATE
    ) -> BarCheck:
        """Whether the overall TPR and TNR reach *min_tpr* and *min_tnr*, with no more than
        *max_no_verdict* of all runs left without a verdict (by default the share above
        which reward functions warn). A rate with no judged run to measure it fails.

        Raises:
            ValueError: When a bar isn't a finite number.
        """
        if not all(math.isfinite(bar) for bar in (min_tpr, min_tnr, max_no_verdict)):
            raise ValueError(
                f'bars must be finite numbers, got {min_tpr=}, {min_tnr=}, {max_no_verdict=}'
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
        for name, hits, total, bar, labeled in (
            ('TPR', overall.tp, overall.tp + overall.fn, min_tpr, 'passed'),
            ('TNR', overall.tn, overall.tn + overall.fp, min_tnr, 'failed'),
        ):
            if not total:
                passed = False
                reasons.append(
                    f'{name} is missing: no judged run of an answer the expert {labeled}'
                )
            elif hits / total < bar:
                passed = False
                reasons.append(f'{name} {hits}/{total} ({hits / total:.3f}) is below {bar}')
            else:
                reasons.append(f'{name} {hits}/{total} ({hits / total:.3f}) reaches {bar}')
        return BarCheck(passed=passed, reasons=tuple(reasons))


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
    ids = Counter(answer.id for answer in answers)
    if repeated := sorted(i for i, n in ids.items() if n > 1):
        raise ValueError(f'answer ids must be unique; given more than once: {repeated}')
    if used := sorted(ids.keys() & built):
        raise ValueError(
            f'answers used to write the rubric can not validate it; hold them out: {used}'
        )
    check_ids = reward_fn.rubric.ids
    if unknown := [f'{a.id}: {c}' for a in answers for c in a.checks if c not in check_ids]:
        raise ValueError(f'expert labels for checks the rubric does not have: {unknown}')
    texts = [answer.answer for answer in answers for _ in range(runs)]
    with ThreadPoolExecutor(max_workers=reward_fn.max_parallel) as pool:
        results = [_judge_run(result) for result in pool.map(reward_fn.grade, texts)]
    rows = tuple(
        AnswerRow(
            **answer.model_dump(include=set(LabeledAnswer.model_fields)),
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
