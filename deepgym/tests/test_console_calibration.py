"""Tests for the console's calibration API: a judge's stored verdicts against the experts'."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deepgym.console.calibration_labels import labeled_answers
from deepgym.console.label_store import LabelStore
from deepgym.console.queue_store import QueueStore
from deepgym.console.run_store import RunStore
from deepgym.labels import LabelCreate, LabelTarget
from deepgym.review_queues import ItemCreate, QueueCreate
from deepgym.rubric_calibration import AnswerRow, CalibrationReport, Confusion, JudgeRun
from tests.console_client import make_client
from tests.test_console_calibration_labels import OLD_RUBRIC, THIS_RUBRIC, _runs, _verdict

JUDGE = 'judge-v1'
REPORTED = (
    'overall', 'blind', 'assisted', 'check_agreement', 'check_summary', 'self_consistency',
    'check_consistency', 'blind_baseline', 'assisted_baseline', 'tied_answers', 'check_ids',
)  # fmt: skip


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / 'console.db'


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path)


def _judge(
    run_id: str, group: str, name: str, value: bool, author: str = JUDGE, **changes: object
) -> LabelCreate:
    """A judge call's verdict on a run."""
    fields = {'source': 'judge', 'comment': None, 'blind': None, 'group_id': group, **changes}
    return _verdict(run_id, author, name, value, **fields)


def _call(passed: bool, c1: bool) -> JudgeRun:
    return JudgeRun(
        score=1.0 if passed else 0.0, verdicts={'c1': 'MET' if c1 else 'UNMET'}, error_type=None
    )


def _labeled(db: Path) -> list[str]:
    """r1 and r2 two blind experts labeled; r3 one assisted expert, with no judge call."""
    r1, r2, r3 = _runs(db, 'one', 'two', 'three')
    LabelStore(db).add_many(
        [
            _verdict(r1, 'ana', 'passed', True), _verdict(r1, 'ana', 'c1', True),
            _verdict(r1, 'bo', 'passed', True), _verdict(r1, 'bo', 'c1', True),
            _verdict(r2, 'ana', 'passed', False), _verdict(r2, 'ana', 'c1', False),
            _verdict(r2, 'bo', 'passed', False), _verdict(r2, 'bo', 'c1', True),
            _verdict(r3, 'cy', 'passed', True, blind=False),
            _judge(r1, 'g1', 'passed', True), _judge(r1, 'g1', 'c1', True),
            _judge(r1, 'g2', 'passed', False), _judge(r1, 'g2', 'c1', False),
            _judge(r2, 'g1', 'passed', False), _judge(r2, 'g1', 'c1', True),
            _judge(r2, 'g1', 'passed', True, author='judge-v2'),
        ]
    )  # fmt: skip
    return [r1, r2, r3]


def _expected(db: Path, r1: str, r2: str) -> CalibrationReport:
    """The report the library computes from the same answers and judge calls."""
    answers = labeled_answers(LabelStore(db), RunStore(db), [r1, r2]).answers
    calls = {r1: (_call(True, True), _call(False, False)), r2: (_call(False, True),)}
    rows = tuple(AnswerRow(**answer.model_dump(), runs=calls[answer.id]) for answer in answers)
    return CalibrationReport(
        judge_id=JUDGE, prompt_version='', rubric_sha256='', check_ids=('c1',), runs=0,
        threshold=1.0, answers=rows,
    )  # fmt: skip


def _agreement(client: TestClient, **params: str) -> dict:
    response = client.get('/api/calibration/agreement', params={'judge_author': JUDGE, **params})
    assert response.status_code == 200, response.text
    return response.json()


def test_agreement_matches_the_library(client: TestClient, db: Path) -> None:
    r1, r2, r3 = _labeled(db)

    body = _agreement(client, run_ids=f'{r1},{r2},{r3}')

    expected = json.loads(_expected(db, r1, r2).model_dump_json())
    assert {name: body[name] for name in REPORTED} == {name: expected[name] for name in REPORTED}
    assert [answer['id'] for answer in body['answers']] == [r1, r2]
    assert body['answers'] == expected['answers']
    tie = labeled_answers(LabelStore(db), RunStore(db), [r1, r2]).skipped
    assert [(skip.run_id, skip.reason) for skip in tie] == [(r2, 'tie')]  # on c1
    assert body['skipped'] == [
        *(skip.model_dump() for skip in tie),
        {'run_id': r3, 'reason': 'no_judge',
         'detail': f'judge {JUDGE} gave no passed on it; the run is left out'},
    ]  # fmt: skip
    assert (body['judge'], body['rubric_sha256']) == (
        {'author': JUDGE, 'source': 'judge', 'rubric_version': None},
        None,
    )
    assert (body['judged_calls'], body['no_passed_calls'], body['no_judge_runs']) == (3, 0, 1)
    assert body['judged_calls'] == expected['judged_runs']
    assert body['expert_only_checks'] == []


def test_agreement_numbers(client: TestClient, db: Path) -> None:
    r1, r2, _ = _labeled(db)

    body = _agreement(client, run_ids=f'{r1},{r2}')

    overall = Confusion(tp=1, fn=1, tn=1, fp=0)  # r1: judge passes once, fails once; r2: fails
    assert body['overall'] == overall.model_dump(mode='json')
    assert body['overall']['kappa'] == pytest.approx(0.4)
    assert body['blind'] == body['overall']
    assert body['assisted']['agreement'] is None
    # The experts tie on r2's c1, so only r1's two calls count.
    assert body['check_agreement']['c1'] == Confusion(tp=1, fn=1, tn=0, fp=0).model_dump()
    assert body['blind_baseline']['answers'] == 2
    assert body['blind_baseline']['expert_kappa'] == pytest.approx(1.0)
    assert body['assisted_baseline'] is None
    assert body['self_consistency'] == {'same': 0, 'answers': 1, 'share': 0.0}


def test_select_runs_by_queue_or_label_filter(client: TestClient, db: Path) -> None:
    r1, r2, r3 = _labeled(db)
    queue = QueueStore(db).create(QueueCreate(name='calibration', label_names=('passed',)))
    QueueStore(db).add_items(
        queue.id,
        [ItemCreate(target=LabelTarget(kind='run', id=run_id)) for run_id in (r2, r1)]
        + [ItemCreate(target=LabelTarget(kind='trace', id='t1'))],
    )

    by_ids = _agreement(client, run_ids=f'{r2},{r1}')
    by_queue = _agreement(client, queue_id=queue.id)
    by_labels = _agreement(client, label_author='bo', label_name='passed', label_current='true')

    assert by_queue == by_ids
    assert [answer['id'] for answer in by_labels['answers']] == [r1, r2]
    assert by_labels['overall'] == by_ids['overall']
    every = _agreement(client, label_source='human')
    assert [answer['id'] for answer in every['answers']] == [r1, r2]
    assert [(skip['run_id'], skip['reason']) for skip in every['skipped']] == [
        (r2, 'tie'),
        (r3, 'no_judge'),
    ]


def test_another_judge_and_the_verifier_are_chosen_by_source_and_author(
    client: TestClient, db: Path
) -> None:
    r1, r2, _ = _labeled(db)
    LabelStore(db).add(_verdict(r1, 'unit-tests', 'passed', True, source='verifier', comment=None))

    other = _agreement(client, run_ids=f'{r1},{r2}', judge_author='judge-v2')
    verifier = _agreement(
        client, run_ids=f'{r1},{r2}', judge_author='unit-tests', judge_source='verifier'
    )

    assert [answer['id'] for answer in other['answers']] == [r2]
    assert other['overall'] == Confusion(tp=0, fn=0, tn=0, fp=1).model_dump(mode='json')
    assert [answer['id'] for answer in verifier['answers']] == [r1]
    assert verifier['check_ids'] == []
    assert verifier['overall']['tp'] == 1


def test_rubric_hash_and_incomplete_judge_calls(client: TestClient, db: Path) -> None:
    r1, r2, r3 = _runs(db, 'one', 'two', 'three')
    LabelStore(db).add_many(
        [
            _verdict(r1, 'ana', 'passed', True, rubric_sha256=THIS_RUBRIC),
            _verdict(r2, 'ana', 'passed', False), _verdict(r3, 'ana', 'passed', True),
            _judge(r1, 'g1', 'passed', True, rubric_sha256=OLD_RUBRIC),
            _judge(r1, 'g2', 'passed', True), _judge(r1, 'g2', 'c1', True),
            _judge(r1, 'g2', 'c2', False),
            _judge(r2, 'g1', 'passed', False), _judge(r2, 'g1', 'c1', False),
            _judge(r3, 'g1', 'c1', True),
        ]
    )  # fmt: skip

    body = _agreement(client, run_ids=f'{r1},{r2},{r3}', rubric_sha256=THIS_RUBRIC)

    assert body['check_ids'] == ['c1', 'c2']
    assert [answer['id'] for answer in body['answers']] == [r1, r2]
    assert [len(answer['runs']) for answer in body['answers']] == [1, 1]
    assert body['answers'][1]['runs'][0]['verdicts'] is None
    assert [(skip['run_id'], skip['reason']) for skip in body['skipped']] == [
        (r1, 'rubric_mismatch'),
        (r2, 'missing_checks'),
        (r3, 'no_passed'),
        (r3, 'no_judge'),
    ]
    assert (body['rubric_hashes'], body['no_rubric_hash']) == ({THIS_RUBRIC: 1}, 2)
    assert body['rubric_sha256'] == THIS_RUBRIC


def test_judge_rubric_hashes_versions_and_version_filter(client: TestClient, db: Path) -> None:
    r1, r2 = _runs(db, 'one', 'two')
    LabelStore(db).add_many(
        [
            _verdict(r1, 'ana', 'passed', True), _verdict(r1, 'ana', 'c1', True),
            _verdict(r1, 'ana', 'c3', False), _verdict(r2, 'ana', 'passed', False),
            _judge(r1, 'g1', 'passed', True, rubric_version='v1', rubric_sha256=THIS_RUBRIC),
            _judge(r1, 'g1', 'c1', True, rubric_version='v1', rubric_sha256=THIS_RUBRIC),
            _judge(r1, 'g2', 'passed', False, rubric_version='v2'),
            _judge(r1, 'g2', 'c1', False, rubric_version='v2'),
            _judge(r2, 'g1', 'passed', False, rubric_version='v1', rubric_sha256=OLD_RUBRIC),
            _judge(r2, 'g2', 'c1', True, rubric_version='v2'),
        ]
    )  # fmt: skip

    pooled = _agreement(client, run_ids=f'{r1},{r2}')
    v1 = _agreement(client, run_ids=f'{r1},{r2}', judge_rubric_version='v1')

    assert pooled['judge_rubric_hashes'] == {OLD_RUBRIC: 1, THIS_RUBRIC: 2}
    assert pooled['judge_no_rubric_hash'] == 2  # r1's g2; r2's g2 gave no passed
    assert (pooled['judge_rubric_versions'], pooled['judge_no_rubric_version']) == (
        {'v1': 3, 'v2': 2},
        0,
    )
    assert (pooled['judged_calls'], pooled['no_passed_calls'], pooled['no_judge_runs']) == (3, 1, 0)
    assert pooled['expert_only_checks'] == ['c3']
    assert pooled['check_consistency'] == {'c1': {'same': 0, 'answers': 1, 'share': 0.0}}
    assert (v1['judged_calls'], v1['judge_rubric_versions']) == (2, {'v1': 3})
    assert [(skip['run_id'], skip['reason']) for skip in v1['skipped']] == [
        (r1, 'rubric_mismatch'),
        (r2, 'rubric_mismatch'),
    ]
    assert v1['judge']['rubric_version'] == 'v1'


def test_excluded_and_unknown_runs_are_skipped(client: TestClient, db: Path) -> None:
    r1, r2, _ = _labeled(db)
    LabelStore(db).add(
        LabelCreate.model_validate({'target': {'kind': 'run', 'id': r2}, 'type': 'invalidate',
                                    'source': 'human', 'author': 'ana', 'comment': 'bad run'})
    )  # fmt: skip

    body = _agreement(client, run_ids=f'{r1},{r2},nope')

    assert [answer['id'] for answer in body['answers']] == [r1]
    assert [(skip['run_id'], skip['reason']) for skip in body['skipped']] == [
        (r2, 'excluded'),
        ('nope', 'no_run'),
    ]


def test_agreement_writes_nothing(client: TestClient, db: Path) -> None:
    r1, r2, _ = _labeled(db)
    before = client.get('/api/labels').json()

    _agreement(client, run_ids=f'{r1},{r2}')

    assert client.get('/api/labels').json() == before


@pytest.mark.parametrize(
    'params',
    [
        {'judge_author': JUDGE},
        {'judge_author': JUDGE, 'run_ids': 'a', 'queue_id': 'q'},
        {'judge_author': JUDGE, 'run_ids': 'a', 'label_author': 'ana'},
        {'judge_author': JUDGE, 'run_ids': 'a,,b'},
        {'run_ids': 'a'},
        {'judge_author': ' judge', 'run_ids': 'a'},
        {'judge_author': JUDGE, 'judge_source': 'human', 'run_ids': 'a'},
        {'judge_author': JUDGE, 'rubric_sha256': 'AB' * 32, 'run_ids': 'a'},
        {'judge_author': JUDGE, 'judge_rubric_version': ' v1', 'run_ids': 'a'},
        {'judge_author': JUDGE, 'label_target_kind': 'trace'},
        {'judge_author': JUDGE, 'label_type': 'vote'},
        {'judge_author': JUDGE, 'run_id': 'a'},
        [('judge_author', JUDGE), ('run_ids', 'a'), ('run_ids', 'b')],
    ],
)
def test_refused_selections(client: TestClient, params: object) -> None:
    response = client.get('/api/calibration/agreement', params=params)  # type: ignore[arg-type]
    assert response.status_code == 422


def test_a_refused_label_filter_names_its_parameter(client: TestClient) -> None:
    response = client.get(
        '/api/calibration/agreement', params={'judge_author': JUDGE, 'label_step': '-1'}
    )

    assert response.status_code == 422
    assert response.json()['detail'][0]['loc'] == ['query', 'label_step']


def test_unknown_queue_is_not_found(client: TestClient) -> None:
    response = client.get(
        '/api/calibration/agreement', params={'judge_author': JUDGE, 'queue_id': 'nope'}
    )
    assert response.status_code == 404


def test_no_runs_selected_gives_empty_numbers(client: TestClient) -> None:
    body = _agreement(client, label_author='nobody')

    assert body['answers'] == []
    assert body['overall'] == Confusion(tp=0, fn=0, tn=0, fp=0).model_dump(mode='json')
