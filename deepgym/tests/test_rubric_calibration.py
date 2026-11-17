"""Tests for judge calibration; every judge here is a stub, no model is called."""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Callable

import httpx
import pytest
from pydantic import ValidationError

from deepgym.rubric import (
    RUBRIC_PROMPT_VERSION,
    ContextOverflowError,
    JudgeClient,
    Rubric,
    RubricReward,
)
from deepgym.rubric_calibration import (
    CalibrationReport,
    Confusion,
    Consistency,
    JudgeRun,
    LabeledAnswer,
    calibrate,
    rubric_sha256,
)
from tests.test_rubric import RUBRIC, TASK, evidence_judge, replying

TEXTS = {
    'good': 'Raise it to $12; churn is 4.1%.',  # every check right: 1.0
    'partial': 'Raise it to $12.',  # no churn figure: 2/3
    'wrong': 'Keep the price; churn is 4.1%.',  # the must check unmet: 0
    'stale': 'Raise it to $12; churn is 4.1%, per the 2023 price list.',  # the fatal check: 0
}
EXPERT = {'good': True, 'partial': False, 'wrong': False, 'stale': False}
"""The expert's call on each answer, the same as a correct judge's at threshold 1.0."""


def labeled(**checks: dict[str, bool]) -> list[LabeledAnswer]:
    """Every answer with the expert's blind call, and per-check labels where given."""
    return [
        LabeledAnswer(
            id=key, answer=text, passed=EXPERT[key], blind=True, checks=checks.get(key, {})
        )
        for key, text in TEXTS.items()
    ]


def run(judge: JudgeClient, answers: list[LabeledAnswer], **kwargs: object) -> CalibrationReport:
    return calibrate(
        RubricReward(TASK, RUBRIC, judge),
        answers,
        judge_id='stub-judge',
        built_from=['the rubric author draft'],
        **kwargs,  # type: ignore[arg-type]
    )


def by_call(reply: Callable[[str, int, str], str]) -> JudgeClient:
    """A judge whose reply to its *n*-th call about an answer (from 0) is ``reply(answer, n,
    right)``, *right* being a correct judge's reply, whatever order the threads call in."""
    calls: Counter[str] = Counter()
    lock = threading.Lock()

    def judge(system: str, message: str) -> str:
        answer = json.loads(message)['answer']
        with lock:
            n = calls[answer]
            calls[answer] += 1
        return reply(answer, n, evidence_judge(system, message))

    return judge


def test_a_perfect_judge_agrees_on_every_run() -> None:
    report = run(evidence_judge, labeled())

    assert report.overall == Confusion(tp=3, fn=0, tn=9, fp=0)
    assert (report.overall.tpr, report.overall.tnr, report.overall.agreement) == (1.0, 1.0, 1.0)
    assert (report.judged_runs, report.no_verdict_runs, report.no_verdict_by_type) == (12, 0, {})
    assert report.self_consistency == Consistency(same=4, answers=4)
    assert {cid: c.share for cid, c in report.check_consistency.items()} == {
        'c1': 1.0,
        'c2': 1.0,
        'c3': 1.0,
    }
    assert report.check_agreement == {}  # no expert labeled a check
    assert (report.judge_id, report.prompt_version) == ('stub-judge', RUBRIC_PROMPT_VERSION)
    assert report.rubric_sha256 == rubric_sha256(TASK, RUBRIC)
    assert len(report.rubric_sha256) == 64
    assert report.answers[1].id == 'partial'
    assert (
        report.answers[1].runs
        == (
            JudgeRun(
                score=2 / 3, verdicts={'c1': 'MET', 'c2': 'UNMET', 'c3': 'UNMET'}, error_type=None
            ),
        )
        * 3
    )
    assert report.meets(1.0, 1.0).passed


def test_the_threshold_decides_the_judges_pass() -> None:
    report = run(evidence_judge, labeled(), threshold=0.5)

    assert report.overall == Confusion(tp=3, fn=0, tn=6, fp=3)  # the 2/3 answer passes


def test_an_always_met_judge_passes_everything() -> None:
    rubric = Rubric.load([['Recommends a price', 1, 'must'], ['Quotes churn', 1]])
    answers = [
        LabeledAnswer(id='a', answer='right', passed=True, blind=True),
        LabeledAnswer(id='b', answer='wrong', passed=False, blind=True),
        LabeledAnswer(id='c', answer='empty', passed=False, blind=True),
    ]
    reward_fn = RubricReward(TASK, rubric, replying('{"c1": "MET", "c2": "MET"}'))

    report = calibrate(reward_fn, answers, judge_id='stub', built_from=[])

    assert report.overall == Confusion(tp=3, fn=0, tn=0, fp=6)
    assert (report.overall.tpr, report.overall.tnr) == (1.0, 0.0)
    bar = report.meets(min_tpr=0.9, min_tnr=0.9)
    assert not bar.passed
    assert bar.reasons == (
        'no verdict 0/9 (0.000) is within 0.05',
        'TPR 3/3 (1.000) reaches 0.9',
        'TNR 0/6 (0.000) is below 0.9',
    )


def _flip_c2(reply: str) -> str:
    verdicts = json.loads(reply)
    verdicts['c2'] = 'UNMET' if verdicts['c2'] == 'MET' else 'MET'
    return json.dumps(verdicts)


def test_a_flaky_judge_shows_in_self_consistency() -> None:
    """Every second call about an answer flips the churn check."""
    flaky = by_call(lambda answer, n, right: _flip_c2(right) if n % 2 else right)

    report = run(flaky, labeled())

    # good: pass, fail, pass. partial: fail, pass, fail. wrong and stale are gated to 0.
    assert report.overall == Confusion(tp=2, fn=1, tn=8, fp=1)
    assert report.self_consistency == Consistency(same=2, answers=4)
    assert report.self_consistency.share == 0.5
    assert report.check_consistency == {
        'c1': Consistency(same=4, answers=4),
        'c2': Consistency(same=0, answers=4),
        'c3': Consistency(same=4, answers=4),
    }


def test_runs_without_a_verdict_are_counted_by_type_and_never_a_fail() -> None:
    def failing(answer: str, n: int, right: str) -> str:
        if n == 0 and answer == TEXTS['good']:
            raise httpx.ConnectError('refused')
        if n == 0 and answer == TEXTS['wrong']:
            return 'not json'
        if n == 0 and answer == TEXTS['stale']:  # too long for the judge: a judged 0
            raise ContextOverflowError('input too long', lambda message: True)
        return right

    report = run(by_call(failing), labeled())

    assert (report.judged_runs, report.no_verdict_runs) == (10, 2)
    assert report.no_verdict_by_type == {'provider_error': 1, 'judge_error': 1}
    assert report.overall == Confusion(tp=2, fn=0, tn=8, fp=0)  # fn stays 0
    good, _, wrong, stale = report.answers
    assert good.runs[0] == JudgeRun(score=None, verdicts=None, error_type='provider_error')
    assert wrong.runs[0].error_type == 'judge_error'
    assert stale.runs[0] == JudgeRun(score=0.0, verdicts=None, error_type='solution_error')
    assert report.self_consistency == Consistency(same=4, answers=4)
    assert report.check_consistency['c3'] == Consistency(same=4, answers=4)
    assert not report.meets(0.9, 0.9).passed  # 2 of 12 runs without a verdict
    assert report.meets(0.9, 0.9, max_no_verdict=0.2).reasons[0] == (
        'no verdict 2/12 (0.167) is within 0.2'
    )


def test_one_run_per_answer_measures_no_self_consistency() -> None:
    report = run(evidence_judge, labeled(), runs=1)

    assert report.self_consistency == Consistency(same=0, answers=0)
    assert report.self_consistency.share is None


def _labels(**changes: object) -> list[LabeledAnswer]:
    answers = labeled()
    return [answers[0].model_copy(update=changes), *answers[1:]]


@pytest.mark.parametrize(
    ('kwargs', 'error', 'message'),
    [
        (
            {'built_from': ['partial', 'draft', 'good']},
            ValueError,
            "can not validate it; hold them out: ['good', 'partial']",
        ),
        (
            {'built_from': 'good'},
            TypeError,
            "built_from is a list of answer ids, got the string 'good'",
        ),
        (
            {'answers': labeled() + labeled()},
            ValueError,
            "given more than once: ['good', 'partial'",
        ),
        ({'answers': _labels(checks={'c9': True})}, ValueError, "does not have: ['good: c9']"),
        ({'runs': 0}, ValueError, 'runs must be at least 1, got 0'),
    ],
)
def test_bad_input_is_refused_before_any_judge_call(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    seen: list[tuple[str, str]] = []
    args: dict[str, object] = {'answers': labeled(), 'judge_id': 'stub', 'built_from': [], **kwargs}

    with pytest.raises(error) as raised:
        calibrate(RubricReward(TASK, RUBRIC, replying('{}', seen)), **args)  # type: ignore[arg-type]

    assert message in str(raised.value)
    assert seen == []


def test_blind_and_assisted_labels_are_reported_apart() -> None:
    """The blind expert passes the partial answer the judge fails; assisted labels agree."""
    answers = [
        answer.model_copy(update={'blind': True, 'passed': True})
        if answer.id == 'partial'
        else answer.model_copy(update={'blind': False})
        for answer in labeled()
    ]

    report = run(evidence_judge, answers)

    assert report.blind == Confusion(tp=0, fn=3, tn=0, fp=0)
    assert (report.blind.tpr, report.blind.tnr) == (0.0, None)
    assert report.assisted == Confusion(tp=3, fn=0, tn=6, fp=0)
    assert report.overall == Confusion(tp=3, fn=3, tn=6, fp=0)


def test_per_check_agreement_uses_the_experts_check_labels() -> None:
    answers = labeled(
        stale={'c3': True},
        good={'c1': True, 'c2': True},
        partial={'c2': True},  # the judge marks c2 UNMET here
    )

    report = run(evidence_judge, answers)

    assert list(report.check_agreement) == ['c1', 'c2', 'c3']  # rubric order
    assert report.check_agreement['c1'] == Confusion(tp=3, fn=0, tn=0, fp=0)
    assert report.check_agreement['c2'] == Confusion(tp=3, fn=3, tn=0, fp=0)
    assert report.check_agreement['c2'].agreement == 0.5
    assert report.check_agreement['c3'] == Confusion(tp=3, fn=0, tn=0, fp=0)


def test_the_report_round_trips_through_json() -> None:
    def failing(answer: str, n: int, right: str) -> str:
        if n == 0 and answer == TEXTS['good']:
            raise httpx.ConnectError('refused')
        return right

    report = run(by_call(failing), labeled(good={'c1': True}))

    text = report.to_json()

    assert CalibrationReport.model_validate_json(text) == report
    data = json.loads(text)
    assert data['overall'] == {
        'tp': 2,
        'fn': 0,
        'tn': 9,
        'fp': 0,
        'tpr': 1.0,
        'tnr': 1.0,
        'agreement': 1.0,
    }
    assert data['no_verdict_by_type'] == {'provider_error': 1}
    assert data['answers'][0]['runs'][0] == {
        'score': None,
        'verdicts': None,
        'error_type': 'provider_error',
    }
    with pytest.raises(ValidationError):
        report.runs = 5  # type: ignore[misc]


def test_a_rate_with_nothing_to_measure_fails_the_bar() -> None:
    answers = [answer for answer in labeled() if not answer.passed]

    bar = run(evidence_judge, answers).meets(min_tpr=0.9, min_tnr=0.9)

    assert not bar.passed
    assert bar.reasons == (
        'no verdict 0/9 (0.000) is within 0.05',
        'TPR is missing: no judged run of an answer the expert passed',
        'TNR 9/9 (1.000) reaches 0.9',
    )
