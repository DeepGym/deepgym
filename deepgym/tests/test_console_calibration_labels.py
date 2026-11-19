"""Tests for calibration input built from stored labels, on real stores in a temporary db."""

from pathlib import Path

import pytest

from deepgym.console.calibration_labels import LabeledRuns, Skipped, labeled_answers
from deepgym.console.label_store import LabelStore
from deepgym.console.run_store import RunStore
from deepgym.console.schemas import RunSpec
from deepgym.labels import LabelCreate
from deepgym.rubric import RubricReward
from deepgym.rubric_calibration import (
    Confusion,
    ExpertLabel,
    LabeledAnswer,
    calibrate,
    rubric_sha256,
)
from tests.test_rubric import RUBRIC, TASK, evidence_judge

PARTIAL = 'Raise it to $12.'
GOOD = 'Raise it to $12; churn is 4.1%.'
THIS_RUBRIC = rubric_sha256(TASK, RUBRIC)
OLD_RUBRIC = 'cd' * 32


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / 'console.db'


def _runs(db: Path, *solutions: str) -> list[str]:
    specs = [
        RunSpec(environment='pricing', solution=text, reference=False, environment_sha256='ab' * 32)
        for text in solutions
    ]
    return [run.id for run in RunStore(db).create_group(specs, 'local', 'local')]


def _verdict(run_id: str, author: str, name: str, value: bool, **changes: object) -> LabelCreate:
    """An expert's blind verdict on a run, with *changes*."""
    fields = {
        'target': {'kind': 'run', 'id': run_id},
        'name': name,
        'value': value,
        'source': 'human',
        'author': author,
        'comment': 'why',
        'blind': True,
    }
    return LabelCreate.model_validate({**fields, **changes})


def _whole(run_id: str, kind: str, **changes: object) -> LabelCreate:
    """An invalidation or retraction on a run."""
    fields = {'target': {'kind': 'run', 'id': run_id}, 'type': kind, 'source': 'human'}
    return LabelCreate.model_validate({**fields, 'author': 'ana', **changes})


def _read(db: Path, run_ids: list[str], rubric: str | None = None) -> LabeledRuns:
    return labeled_answers(LabelStore(db), RunStore(db), run_ids, rubric_sha256=rubric)


def test_one_experts_verdicts_on_a_run_become_its_label(db: Path) -> None:
    (run_id,) = _runs(db, PARTIAL)
    LabelStore(db).add_many(
        [
            _verdict(run_id, 'ana', 'passed', False),
            _verdict(run_id, 'ana', 'c1', True),
            _verdict(run_id, 'ana', 'c2', False),
            _verdict(run_id, 'judge-model', 'passed', True, source='judge', comment=None),
        ]
    )

    result = _read(db, [run_id])

    ana = ExpertLabel(passed=False, blind=True, checks={'c1': True, 'c2': False})
    assert result.answers == (
        LabeledAnswer(
            id=run_id,
            answer=PARTIAL,
            passed=False,
            blind=True,
            checks={'c1': True, 'c2': False},
            experts={'ana': ana},
        ),
    )
    assert result.skipped == ()
    assert (result.rubric_hashes, result.no_rubric_hash) == ({}, 3)  # the judge's isn't read
    assert _read(db, [run_id, run_id]) == result  # a run given twice is read once


def test_corrections_count_and_retracted_labels_do_not(db: Path) -> None:
    (run_id,) = _runs(db, PARTIAL)
    store = LabelStore(db)
    first, by_bo = store.add_many(
        [_verdict(run_id, 'ana', 'passed', True), _verdict(run_id, 'bo', 'passed', True)]
    )
    store.add_many(
        [
            _verdict(run_id, 'ana', 'passed', False, supersedes=first.id),
            _whole(run_id, 'retract', author='bo', comment='wrong run', supersedes=by_bo.id),
        ]
    )

    (answer,) = _read(db, [run_id]).answers

    assert (answer.passed, list(answer.experts)) == (False, ['ana'])


def test_an_excluded_run_is_skipped_until_the_exclusion_is_lifted(db: Path) -> None:
    excluded, lifted = _runs(db, PARTIAL, GOOD)
    store = LabelStore(db)
    store.add_many([_verdict(run_id, 'ana', 'passed', True) for run_id in (excluded, lifted)])
    store.add(_whole(excluded, 'invalidate', comment='flaky sandbox'))
    invalidation = store.add(_whole(lifted, 'invalidate', comment='looked flaky'))
    store.add(_whole(lifted, 'retract', comment='it was fine', supersedes=invalidation.id))

    result = _read(db, [excluded, lifted])

    assert [answer.id for answer in result.answers] == [lifted]
    assert result.skipped == (Skipped(run_id=excluded, reason='excluded', detail='flaky sandbox'),)


def test_runs_without_text_or_a_pass_mark_are_skipped_with_the_reason(db: Path) -> None:
    unlabeled, checks_only = _runs(db, PARTIAL, GOOD)
    LabelStore(db).add_many(
        [_verdict('no-such-run', 'ana', 'passed', True), _verdict(checks_only, 'bo', 'c1', True)]
    )

    result = _read(db, ['no-such-run', unlabeled, checks_only])

    left_out = 'no expert passed or failed it; the run is left out'
    assert result.answers == ()
    assert result.skipped == (
        Skipped(run_id='no-such-run', reason='no_run', detail='no such run'),
        Skipped(run_id=unlabeled, reason='no_passed', detail=left_out),
        Skipped(
            run_id=checks_only,
            reason='no_passed',
            detail='bo gave check verdicts but no passed; left out',
        ),
        Skipped(run_id=checks_only, reason='no_passed', detail=left_out),
    )


def test_only_the_verdicts_an_expert_contradicts_are_left_out(db: Path) -> None:
    (run_id,) = _runs(db, PARTIAL)
    LabelStore(db).add_many(
        [
            _verdict(run_id, 'ana', 'passed', True, group_id='pass-1'),
            _verdict(run_id, 'ana', 'passed', False, group_id='pass-2'),
            _verdict(run_id, 'bo', 'passed', True, group_id='pass-1'),
            _verdict(run_id, 'bo', 'passed', True, group_id='pass-2'),  # agrees with itself
            _verdict(run_id, 'bo', 'c1', True, group_id='pass-1'),
            _verdict(run_id, 'bo', 'c1', False, group_id='pass-2'),
            _verdict(run_id, 'bo', 'c2', True, group_id='pass-1'),
        ]
    )

    result = _read(db, [run_id])

    (answer,) = result.answers
    assert answer.experts == {'bo': ExpertLabel(passed=True, blind=True, checks={'c2': True})}
    assert result.no_rubric_hash == 3  # bo's two passed and c2; not ana's, nor bo's c1
    assert result.skipped == (
        Skipped(
            run_id=run_id,
            reason='conflicting_author',
            detail='ana marks passed both true and false; ana is left out of the run',
        ),
        Skipped(
            run_id=run_id,
            reason='conflicting_author',
            detail="bo marks c1 both true and false; it is left out of bo's label",
        ),
    )


def test_verdicts_against_another_rubric_are_left_out(db: Path) -> None:
    (run_id,) = _runs(db, PARTIAL)
    LabelStore(db).add_many(
        [
            _verdict(run_id, 'ana', 'passed', True, rubric_sha256=THIS_RUBRIC),
            _verdict(run_id, 'ana', 'c1', True, rubric_sha256=OLD_RUBRIC),
            _verdict(run_id, 'bo', 'passed', False, rubric_sha256=OLD_RUBRIC),
            _verdict(run_id, 'bo', 'c1', True),  # no hash recorded: kept, counted
            _verdict(run_id, 'cy', 'passed', True),
        ]
    )

    checked = _read(db, [run_id], rubric=THIS_RUBRIC)

    (answer,) = checked.answers
    assert answer.experts == {
        'ana': ExpertLabel(passed=True, blind=True),
        'cy': ExpertLabel(passed=True, blind=True),
    }
    assert checked.skipped == (
        Skipped(
            run_id=run_id, reason='rubric_mismatch', detail=f'ana c1 labeled against {OLD_RUBRIC}'
        ),
        Skipped(
            run_id=run_id,
            reason='rubric_mismatch',
            detail=f'bo passed labeled against {OLD_RUBRIC}',
        ),
        Skipped(
            run_id=run_id,
            reason='no_passed',
            detail='bo gave check verdicts but no passed; left out',
        ),
    )
    # bo's c1 records no hash but bo is left out, so only cy's passed counts
    assert (checked.rubric_hashes, checked.no_rubric_hash) == ({THIS_RUBRIC: 1}, 1)
    unchecked = _read(db, [run_id])
    assert list(unchecked.answers[0].experts) == ['ana', 'bo', 'cy']
    assert unchecked.skipped == ()
    # Without a filter the mix of rubric versions still shows.
    assert unchecked.rubric_hashes == {THIS_RUBRIC: 1, OLD_RUBRIC: 2}
    assert unchecked.no_rubric_hash == 2


def test_the_majority_is_the_judges_reference_and_ties_are_reported(db: Path) -> None:
    majority, tied = _runs(db, PARTIAL, GOOD)
    LabelStore(db).add_many(
        [
            _verdict(majority, 'ana', 'passed', True),
            _verdict(majority, 'ana', 'c1', True),
            _verdict(majority, 'ana', 'c2', True),
            _verdict(majority, 'bo', 'passed', True),
            _verdict(majority, 'bo', 'c1', False, blind=False),  # bo's label isn't blind
            _verdict(majority, 'cy', 'passed', False, blind=None),  # unrecorded: not blind
            _verdict(tied, 'ana', 'passed', True),
            _verdict(tied, 'bo', 'passed', False),
        ]
    )

    result = _read(db, [majority, tied])

    first, second = result.answers
    assert (first.passed, first.blind, first.checks) == (True, False, {'c2': True})
    assert first.experts == {
        'ana': ExpertLabel(passed=True, blind=True, checks={'c1': True, 'c2': True}),
        'bo': ExpertLabel(passed=True, blind=False, checks={'c1': False}),
        'cy': ExpertLabel(passed=False, blind=False),
    }
    assert (second.passed, second.blind, list(second.experts)) == (None, True, ['ana', 'bo'])
    assert result.skipped == (
        Skipped(
            run_id=majority,
            reason='tie',
            detail='c1: ana true, bo false; c1 is left out of its checks',
        ),
        Skipped(
            run_id=tied,
            reason='tie',
            detail='passed: ana true, bo false; passed is None: the answer is left out of '
            'overall, blind, assisted and min_kappa',
        ),
    )


def test_stored_labels_calibrate_a_judge_with_a_human_baseline(db: Path) -> None:
    """ana and bo agree on the good answer and split on the partial one the judge fails."""
    good, partial = _runs(db, GOOD, PARTIAL)
    LabelStore(db).add_many(
        [
            _verdict(run_id, author, 'passed', value, rubric_sha256=THIS_RUBRIC)
            for run_id, author, value in (
                (good, 'ana', True),
                (good, 'bo', True),
                (partial, 'ana', False),
                (partial, 'bo', True),
            )
        ]
    )
    reward_fn = RubricReward(TASK, RUBRIC, evidence_judge)
    labeled = _read(db, [good, partial], rubric=rubric_sha256(reward_fn.task, reward_fn.rubric))

    report = calibrate(reward_fn, labeled.answers, judge_id='stub', built_from=[])

    assert (labeled.rubric_hashes, labeled.no_rubric_hash) == ({THIS_RUBRIC: 4}, 0)
    assert report.overall == Confusion(tp=3, fn=0, tn=0, fp=0)  # the tie is left out
    baseline = report.blind_baseline
    assert baseline is not None
    assert baseline.experts['ana']['bo'].passed == Confusion(tp=1, fn=1, tn=0, fp=0)
    assert baseline.judge['ana']['bo'].passed == Confusion(tp=3, fn=0, tn=3, fp=0)
    assert baseline.judge['bo']['ana'].passed == Confusion(tp=3, fn=3, tn=0, fp=0)
