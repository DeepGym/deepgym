"""Tests for judge calibration; every judge here is a stub, no model is called."""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path

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
    CheckSummary,
    Confusion,
    Consistency,
    ExpertLabel,
    JudgeRun,
    LabeledAnswer,
    calibrate,
    macro_f1,
    pair_agreement,
    pooled_kappa,
    rubric_sha256,
)
from tests.test_rubric import RUBRIC, TASK, evidence_judge, raising, replying

GOLDEN = Path(__file__).resolve().parent / 'golden' / 'calibration' / 'report_0_6_7.json'
"""A report as 0.6.7 saved it, before kappa, F1, experts and the human baseline."""

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


def expert(passed: bool, blind: bool = True, **checks: bool) -> ExpertLabel:
    return ExpertLabel(passed=passed, blind=blind, checks=checks)


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
        ({'threshold': float('nan')}, ValueError, 'threshold must be a finite number, got nan'),
        ({'built_from': [b'good']}, TypeError, "built_from ids must be strings, got [b'good']"),
        (
            {'answers': _labels(experts={'ana': expert(True, c9=True), 'bo': expert(True)})},
            ValueError,
            "does not have: ['good: c9']",
        ),
        (  # model_copy skips validation; calibrate validates again
            {'answers': _labels(passed=None)},
            ValidationError,
            'passed may be None only when its experts tie',
        ),
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


def first_good_call_refused(answer: str, n: int, right: str) -> str:
    """The provider refuses the first call about the good answer; every other reply is right."""
    if n == 0 and answer == TEXTS['good']:
        raise httpx.ConnectError('refused')
    return right


def test_the_report_round_trips_through_json() -> None:
    report = run(by_call(first_good_call_refused), labeled(good={'c1': True}))

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
        'precision': 1.0,
        'f1': 1.0,
        'kappa': 1.0,
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


@pytest.mark.parametrize(
    'bars', [(float('nan'), 0.9), (0.9, float('nan')), (0.9, 0.9, float('inf'))]
)
def test_a_bar_that_is_not_a_finite_number_is_refused(bars: tuple[float, ...]) -> None:
    report = run(replying('{"c1": "UNMET", "c2": "UNMET", "c3": "MET"}'), labeled())

    with pytest.raises(ValueError, match='bars must be finite numbers'):
        report.meets(*bars)


def test_an_answer_id_must_be_a_string() -> None:
    with pytest.raises(ValidationError):
        LabeledAnswer(id=b'good', answer='x', passed=True, blind=True)  # type: ignore[arg-type]


def test_a_stored_report_never_holds_a_non_finite_number() -> None:
    data = json.loads(run(evidence_judge, labeled()).to_json())
    data['answers'][0]['runs'][0]['score'] = float('inf')

    with pytest.raises(ValidationError):
        CalibrationReport.model_validate(data)


def test_check_verdicts_of_a_run_without_a_verdict_are_left_out() -> None:
    report = run(evidence_judge, labeled(good={'c1': True}), runs=2)
    good, *rest = report.answers
    judged = good.runs[0]
    flipped = {**(judged.verdicts or {}), 'c1': 'UNMET'}
    unjudged = JudgeRun(score=None, verdicts=flipped, error_type='judge_error')  # type: ignore[arg-type]
    stored = report.model_copy(
        update={'answers': (good.model_copy(update={'runs': (judged, unjudged)}), *rest)}
    )

    assert stored.check_agreement['c1'] == Confusion(tp=1, fn=0, tn=0, fp=0)
    assert stored.check_consistency['c1'] == Consistency(same=3, answers=3)


def test_answers_from_a_saved_report_can_be_calibrated_again() -> None:
    saved = run(evidence_judge, labeled(), runs=1)

    again = run(evidence_judge, list(saved.answers), runs=2)

    assert [len(row.runs) for row in again.answers] == [2, 2, 2, 2]
    assert again.overall == Confusion(tp=2, fn=0, tn=6, fp=0)


def test_a_stored_run_must_mark_every_check() -> None:
    data = json.loads(run(evidence_judge, labeled()).to_json())
    data['answers'][0]['runs'][0]['verdicts'] = {'c1': 'MET'}

    with pytest.raises(ValidationError, match='not the checks'):
        CalibrationReport.model_validate(data)


@pytest.mark.parametrize(
    ('table', 'kappa'),
    [
        (Confusion(tp=3, fn=0, tn=2, fp=0), 1.0),  # perfect
        (Confusion(tp=1, fn=1, tn=1, fp=1), 0.0),  # chance: po = pe = 0.5
        (Confusion(tp=2, fn=0, tn=0, fp=2), 0.0),  # the rater passes all: pe = po = 0.5
        (Confusion(tp=0, fn=2, tn=0, fp=2), -1.0),  # always the opposite call
        (Confusion(tp=1, fn=2, tn=1, fp=2), -1 / 3),  # po 1/3, pe 1/2
        (Confusion(tp=20, fn=10, tn=15, fp=5), 0.4),  # po 0.7, pe 0.5*0.6 + 0.5*0.4 = 0.5
        (Confusion(tp=4, fn=0, tn=0, fp=0), None),  # both pass all: pe = 1, 0/0
        (Confusion(tp=0, fn=0, tn=4, fp=0), None),  # both fail all
        (Confusion(tp=0, fn=0, tn=0, fp=0), None),  # nothing compared
    ],
)
def test_kappa_against_hand_computed_values(table: Confusion, kappa: float | None) -> None:
    expected = None if kappa is None else pytest.approx(kappa)

    assert table.kappa == expected
    assert pooled_kappa([table]) == expected  # one table pools to its own kappa


def test_pooled_kappa_weighs_each_table_by_its_pairs() -> None:
    many = Confusion(tp=10, fn=0, tn=10, fp=0)  # agree 20, chance 10
    one = Confusion(tp=0, fn=0, tn=0, fp=1)  # agree 0, chance 0
    flipped = Confusion(tp=0, fn=1, tn=0, fp=0)

    assert one.kappa == 0.0  # a mean of the two kappas would be 0.5
    assert pooled_kappa([many, one]) == pytest.approx(10 / 11)  # (20 - 10) / (21 - 10)
    assert pooled_kappa([many, flipped]) == pooled_kappa([many, one])  # sides' order is moot
    assert pooled_kappa([]) is None


def test_precision_f1_macro_f1_and_the_check_summary() -> None:
    table = Confusion(tp=20, fn=10, tn=15, fp=5)

    assert (table.precision, table.tpr, table.f1) == (0.8, pytest.approx(2 / 3), 40 / 55)
    no_pass = Confusion(tp=0, fn=0, tn=3, fp=0)
    assert (no_pass.precision, no_pass.f1) == (None, None)
    assert Confusion(tp=0, fn=0, tn=2, fp=1).f1 == 0.0
    passes, half, undefined = (
        Confusion(tp=1, fn=0, tn=0, fp=0),
        Confusion(tp=1, fn=1, tn=0, fp=0),
        Confusion(tp=0, fn=0, tn=2, fp=0),
    )
    assert macro_f1([passes, half, undefined]) == pytest.approx((1 + 2 / 3) / 2)
    assert macro_f1([undefined]) is None
    assert macro_f1([]) is None
    assert CheckSummary.of([passes, half, undefined, table]).model_dump() == {
        'macro_f1': pytest.approx((1 + 2 / 3 + 40 / 55) / 3),
        'f1_checks': 3,
        'mean_kappa': pytest.approx((0.0 + 0.4) / 2),  # half: 0.0; table: 0.4
        'kappa_checks': 2,
    }


def test_pair_agreement_compares_the_targets_and_names_both_gave() -> None:
    ana = {
        'a': {'passed': True, 'c1': True},
        'b': {'passed': False, 'c2': True, 'c10': False},
        'c': {'passed': True},  # bo never labeled c
    }
    bo = {
        'a': {'passed': True, 'c1': False},
        'b': {'passed': True, 'c2': True, 'c10': False},
        'd': {'passed': False},
    }

    pair = pair_agreement(ana, bo)

    assert pair.passed == Confusion(tp=1, fn=1, tn=0, fp=0)
    assert pair.checks == {
        'c1': Confusion(tp=0, fn=0, tn=0, fp=1),
        'c2': Confusion(tp=1, fn=0, tn=0, fp=0),
        'c10': Confusion(tp=0, fn=0, tn=1, fp=0),
    }
    assert list(pair.checks) == ['c1', 'c2', 'c10']
    # c1: F1 0.0 and kappa 0.0; c2: F1 1.0, kappa undefined; c10: both undefined
    assert pair.check_summary == CheckSummary(
        macro_f1=0.5, f1_checks=2, mean_kappa=0.0, kappa_checks=1
    )
    assert pair_agreement(bo, ana).passed == Confusion(tp=1, fn=0, tn=0, fp=1)


def test_the_report_gives_kappa_and_macro_f1_per_check() -> None:
    answers = labeled(
        stale={'c3': True},
        good={'c1': True, 'c2': True},
        partial={'c2': True},  # the judge marks c2 UNMET here
    )

    report = run(evidence_judge, answers)

    assert (report.overall.kappa, report.overall.f1) == (1.0, 1.0)
    assert {cid: c.kappa for cid, c in report.check_agreement.items()} == {
        'c1': None,  # the judge and the expert mark it MET on every run
        'c2': 0.0,  # the expert marks it MET throughout: nothing beyond chance
        'c3': None,
    }
    assert report.check_agreement['c2'].f1 == 6 / 9
    assert report.check_summary.model_dump() == {
        'macro_f1': pytest.approx((1 + 6 / 9 + 1) / 3),
        'f1_checks': 3,
        'mean_kappa': 0.0,
        'kappa_checks': 1,
    }


def _two_experts() -> list[LabeledAnswer]:
    """ana and bo split on the partial answer, which the judge fails, but agree it meets c1;
    one expert on stale."""
    return [
        LabeledAnswer(
            id='good',
            answer=TEXTS['good'],
            passed=True,
            blind=True,
            experts={'ana': expert(True, c2=True), 'bo': expert(True, c2=True)},
        ),
        LabeledAnswer(
            id='partial',
            answer=TEXTS['partial'],
            passed=None,
            blind=True,
            checks={'c1': True},
            experts={'ana': expert(False, c1=True), 'bo': expert(True, c1=True)},
        ),
        LabeledAnswer(
            id='wrong',
            answer=TEXTS['wrong'],
            passed=False,
            blind=True,
            experts={'ana': expert(False), 'bo': expert(False)},
        ),
        LabeledAnswer(id='stale', answer=TEXTS['stale'], passed=False, blind=True),
    ]


def test_the_human_baseline_puts_experts_beside_the_judge() -> None:
    report = run(evidence_judge, _two_experts())

    baseline = report.blind_baseline
    assert baseline is not None
    assert report.assisted_baseline is None
    assert baseline.answers == 3
    assert list(baseline.experts) == ['ana']
    assert baseline.experts['ana']['bo'].passed == Confusion(tp=1, fn=1, tn=1, fp=0)
    assert baseline.experts['ana']['bo'].checks['c1'] == Confusion(tp=1, fn=0, tn=0, fp=0)
    # The judge against each of the two, on the judged runs of the answers both labeled.
    assert baseline.judge['ana']['bo'].passed == Confusion(tp=3, fn=0, tn=6, fp=0)
    assert baseline.judge['bo']['ana'].passed == Confusion(tp=3, fn=3, tn=3, fp=0)
    assert (baseline.expert_comparisons, baseline.judge_comparisons) == (3, 18)
    assert baseline.expert_kappa == pytest.approx(0.4)  # one pair: its own Cohen's kappa
    assert baseline.expert_agreement == pytest.approx(2 / 3)
    # agreed 9 + 6, chance 5 + 4, of 18: (15 - 9) / (18 - 9)
    assert baseline.judge_kappa == pytest.approx(2 / 3)
    assert baseline.judge_agreement == pytest.approx(15 / 18)
    # The tie is left out of overall, but its agreed c1 counts per check.
    assert report.overall == Confusion(tp=3, fn=0, tn=6, fp=0)
    assert report.check_agreement == {'c1': Confusion(tp=3, fn=0, tn=0, fp=0)}
    bar = report.meets(1.0, 1.0, max_kappa_gap=0.0)
    assert bar.passed
    assert bar.reasons[-1] == (
        "judge kappa 0.667 is within 0.0 of the experts' 0.400 "
        '(blind; 18 judged-run comparisons; experts 3 comparisons on 3 answers)'
    )


def test_a_pair_sharing_one_answer_barely_moves_the_baseline() -> None:
    """ana and bo agree on 20 answers; cy shares one with ana and disagrees."""
    answers = [
        LabeledAnswer(
            id=f'{key}{i}',
            answer=TEXTS[key],
            passed=key == 'good',
            blind=True,
            experts={'ana': expert(key == 'good'), 'bo': expert(key == 'good')},
        )
        for key in ('good', 'wrong')
        for i in range(10)
    ]
    answers.append(
        LabeledAnswer(
            id='x',
            answer=TEXTS['good'],
            passed=None,
            blind=True,
            experts={'ana': expert(True), 'cy': expert(False)},
        )
    )

    baseline = run(evidence_judge, answers, runs=1).blind_baseline

    assert baseline is not None
    assert baseline.experts['ana']['cy'].passed.kappa == 0.0
    assert baseline.experts['ana']['bo'].passed.kappa == 1.0
    assert baseline.expert_kappa == pytest.approx(10 / 11)  # not the pairs' mean, 0.5
    # judge vs ana and vs bo: agreed 20 each, chance 10 each; vs ana on x: 1 and 1; vs cy: 0
    assert baseline.judge_kappa == pytest.approx(20 / 21)
    assert (baseline.answers, baseline.expert_comparisons, baseline.judge_comparisons) == (
        21,
        21,
        42,
    )


def test_blind_and_assisted_labels_make_separate_baselines() -> None:
    answers = [
        LabeledAnswer(
            id='good',
            answer=TEXTS['good'],
            passed=True,
            blind=True,
            experts={'ana': expert(True), 'bo': expert(True)},
        ),
        LabeledAnswer(
            id='partial',
            answer=TEXTS['partial'],
            passed=None,
            blind=True,
            experts={'ana': expert(False), 'bo': expert(True)},
        ),
        LabeledAnswer(
            id='wrong',
            answer=TEXTS['wrong'],
            passed=False,
            blind=False,
            experts={'ana': expert(False, blind=False), 'bo': expert(False, blind=False)},
        ),
        LabeledAnswer(  # one blind and one assisted label: in neither baseline
            id='stale',
            answer=TEXTS['stale'],
            passed=False,
            blind=False,
            experts={'ana': expert(False, blind=False), 'bo': expert(False)},
        ),
    ]

    report = run(evidence_judge, answers)

    assert report.blind_baseline is not None
    assert report.assisted_baseline is not None
    assert report.blind_baseline.answers == 2
    assert report.blind_baseline.experts['ana']['bo'].passed == Confusion(tp=1, fn=1, tn=0, fp=0)
    assert report.assisted_baseline.answers == 1
    assert report.assisted_baseline.experts['ana']['bo'].passed == Confusion(tp=0, fn=0, tn=1, fp=0)


def test_an_answer_without_a_judged_run_is_left_out_of_both_sides() -> None:
    def refuse_wrong(answer: str, n: int, right: str) -> str:
        if answer == TEXTS['wrong']:
            raise httpx.ConnectError('refused')
        return right

    baseline = run(by_call(refuse_wrong), _two_experts()).blind_baseline

    assert baseline is not None
    assert baseline.answers == 2
    assert baseline.experts['ana']['bo'].passed == Confusion(tp=1, fn=1, tn=0, fp=0)


def _experts_who_agree() -> CalibrationReport:
    """Both experts pass the partial answer the judge fails."""
    answers = [
        LabeledAnswer(
            id=key,
            answer=TEXTS[key],
            passed=key != 'wrong',
            blind=True,
            experts={'ana': expert(key != 'wrong'), 'bo': expert(key != 'wrong')},
        )
        for key in ('good', 'partial', 'wrong')
    ]
    return run(evidence_judge, answers)


def test_the_judge_falls_short_of_experts_who_agree() -> None:
    report = _experts_who_agree()

    bar = report.meets(0.5, 0.5, min_kappa=0.3, max_kappa_gap=0.5)

    assert not bar.passed
    assert bar.reasons[-2:] == (
        'kappa 0.400 over 9 judged runs reaches 0.3',
        "judge kappa 0.400 is more than 0.5 below the experts' 1.000 "
        '(blind; 18 judged-run comparisons; experts 3 comparisons on 3 answers)',
    )
    assert report.meets(0.5, 0.5, min_kappa=0.5).reasons[-1] == (
        'kappa 0.400 over 9 judged runs is below 0.5'
    )


def test_a_gap_at_the_bar_passes_within_float_rounding() -> None:
    report = _experts_who_agree()  # experts 1.0, judge 0.4

    assert report.meets(0.5, 0.5, max_kappa_gap=0.6 - 1e-12).passed
    assert not report.meets(0.5, 0.5, max_kappa_gap=0.6 - 1e-6).passed


@pytest.mark.parametrize(
    ('judge', 'answers', 'kwargs', 'reason'),
    [
        (
            evidence_judge,
            [a for a in labeled() if not a.passed],
            {'min_kappa': 0.5},
            'kappa is undefined: the judge and the experts made one and the same call on '
            'every judged run',
        ),
        (
            raising(httpx.ConnectError('refused')),
            labeled(),
            {'min_kappa': 0.5},
            'kappa is undefined: no judged run of an answer with an expert pass mark',
        ),
        (
            evidence_judge,
            labeled(),
            {'max_kappa_gap': 0.1},
            'blind baseline is missing: no answer with a judged run that two or more experts '
            'labeled blind',
        ),
        (
            evidence_judge,
            [
                LabeledAnswer(
                    id='good',
                    answer=TEXTS['good'],
                    passed=True,
                    blind=False,
                    experts={'ana': expert(True, blind=False), 'bo': expert(True, blind=False)},
                )
            ],
            {'max_kappa_gap': 0.1},
            'blind baseline is missing: no answer with a judged run that two or more experts '
            'labeled blind',
        ),
        (
            evidence_judge,
            [
                LabeledAnswer(
                    id='good',
                    answer=TEXTS['good'],
                    passed=True,
                    blind=True,
                    experts={'ana': expert(True), 'bo': expert(True)},
                )
            ],
            {'max_kappa_gap': 0.1},
            "blind baseline is uninformative: the experts' kappa is undefined, no agreement "
            'beyond chance to come close to (6 judged-run comparisons; experts 1 comparisons '
            'on 1 answers)',
        ),
    ],
)
def test_an_undefined_kappa_or_missing_baseline_fails_its_bar(
    judge: JudgeClient, answers: list[LabeledAnswer], kwargs: dict[str, float], reason: str
) -> None:
    bar = run(judge, answers).meets(0.0, 0.0, max_no_verdict=1.0, **kwargs)

    assert not bar.passed
    assert bar.reasons[-1] == reason


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'min_kappa': float('nan')}, 'bars must be finite numbers'),
        ({'max_kappa_gap': float('inf')}, 'bars must be finite numbers'),
        ({'min_baseline_answers': 2}, 'for the max_kappa_gap bar'),
        ({'max_kappa_gap': 0.1, 'min_baseline_answers': 0}, 'a count of at least 1'),
    ],
)
def test_a_bad_kappa_bar_is_refused(kwargs: dict[str, float], message: str) -> None:
    report = run(evidence_judge, labeled())

    with pytest.raises(ValueError, match=message):
        report.meets(0.9, 0.9, **kwargs)


def test_experts_who_agree_no_better_than_chance_make_no_baseline() -> None:
    """ana and bo give opposite calls on both answers: kappa -1."""
    answers = [
        LabeledAnswer(
            id=key,
            answer=TEXTS[key],
            passed=None,
            blind=True,
            experts={'ana': expert(key == 'good'), 'bo': expert(key != 'good')},
        )
        for key in ('good', 'wrong')
    ]
    report = run(evidence_judge, answers)

    bar = report.meets(0.0, 0.0, max_no_verdict=1.0, max_kappa_gap=1.0)

    assert not bar.passed
    assert bar.reasons[-1] == (
        "blind baseline is uninformative: the experts' kappa -1.000 is not above 0, no "
        'agreement beyond chance to come close to (12 judged-run comparisons; experts 2 '
        'comparisons on 2 answers)'
    )


def test_a_baseline_with_too_few_answers_fails_the_gap_bar() -> None:
    report = run(evidence_judge, _two_experts())  # 3 baseline answers

    assert report.meets(1.0, 1.0, max_kappa_gap=0.0, min_baseline_answers=3).passed
    bar = report.meets(1.0, 1.0, max_kappa_gap=0.0, min_baseline_answers=4)
    assert not bar.passed
    assert bar.reasons[-1] == (
        'blind baseline has 3 answers, fewer than 4 '
        '(18 judged-run comparisons; experts 3 comparisons on 3 answers)'
    )


def test_bar_reasons_say_how_many_tied_answers_were_left_out() -> None:
    report = run(evidence_judge, _two_experts())

    bar = report.meets(1.0, 1.0, min_kappa=0.5)

    assert report.tied_answers == 1
    assert bar.reasons[1:] == (
        'TPR 3/3 (1.000) reaches 1.0; 1 tied answer left out',
        'TNR 6/6 (1.000) reaches 1.0; 1 tied answer left out',
        'kappa 1.000 over 9 judged runs reaches 0.5; 1 tied answer left out',
    )
    good, partial, wrong, stale = _two_experts()
    split = wrong.model_dump() | {
        'passed': None,
        'experts': {'ana': expert(False), 'bo': expert(True)},
    }
    two_tied = run(evidence_judge, [good, partial, LabeledAnswer.model_validate(split), stale])
    assert two_tied.tied_answers == 2
    assert two_tied.meets(1.0, 1.0).reasons[1] == (
        'TPR 3/3 (1.000) reaches 1.0; 2 tied answers left out'
    )


def test_a_stored_report_must_not_repeat_an_answer() -> None:
    data = json.loads(run(evidence_judge, labeled()).to_json())
    data['answers'].append(data['answers'][0])

    with pytest.raises(ValidationError, match="given more than once: \\['good'\\]"):
        CalibrationReport.model_validate(data)


@pytest.mark.parametrize(
    'experts',
    [{}, {'ana': expert(True)}, {'ana': expert(False), 'bo': expert(False)}],
)
def test_no_pass_mark_only_when_the_experts_tie(experts: dict[str, ExpertLabel]) -> None:
    with pytest.raises(ValidationError, match='only when its experts tie'):
        LabeledAnswer(id='a', answer='x', passed=None, blind=True, experts=experts)

    tied = {'ana': expert(True), 'bo': expert(False)}
    assert LabeledAnswer(id='a', answer='x', passed=None, blind=True, experts=tied).passed is None


def test_a_report_with_experts_round_trips_through_json() -> None:
    report = run(evidence_judge, _two_experts())

    text = report.to_json()

    assert CalibrationReport.model_validate_json(text) == report
    data = json.loads(text)
    assert data['answers'][1]['passed'] is None
    assert data['answers'][0]['experts']['ana'] == {
        'passed': True,
        'blind': True,
        'checks': {'c2': True},
    }
    assert data['blind_baseline']['expert_kappa'] == pytest.approx(0.4)
    assert data['blind_baseline']['judge']['bo']['ana']['passed']['kappa'] == pytest.approx(0.4)
    assert data['assisted_baseline'] is None
    assert data['check_summary'] == {
        'macro_f1': 1.0,
        'f1_checks': 1,
        'mean_kappa': None,
        'kappa_checks': 0,
    }


def _kept(old: object, new: object) -> bool:
    """Whether every key of *old* is in *new* with the same value, at every depth."""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(k in new and _kept(v, new[k]) for k, v in old.items())
    if isinstance(old, list):
        return isinstance(new, list) and len(old) == len(new) and all(map(_kept, old, new))
    return old == new


def test_a_report_saved_before_kappa_and_experts_still_loads() -> None:
    text = GOLDEN.read_text()

    report = CalibrationReport.model_validate_json(text)

    expected = run(
        by_call(first_good_call_refused),
        labeled(good={'c1': True}, partial={'c2': False}),
        runs=2,
    )
    assert report == expected
    assert all(row.experts == {} for row in report.answers)
    assert (report.blind_baseline, report.assisted_baseline) == (None, None)
    new = json.loads(report.to_json())
    assert _kept(json.loads(text), new)  # every number saved then is still there
    assert new['overall']['kappa'] == 1.0
    assert new['check_summary'] == {
        'macro_f1': 1.0,
        'f1_checks': 1,
        'mean_kappa': None,
        'kappa_checks': 0,
    }
