"""Tests for review queues: settings, items, reservations, reviews, blind mode and progress."""

import multiprocessing
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from multiprocessing.synchronize import Barrier
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym.console import queue_store
from deepgym.console.db import connect, iso_utc
from deepgym.console.episode_store import record_episode
from deepgym.console.label_store import LabelStore
from deepgym.console.queue_store import (
    NotReviewerError,
    QueueInputError,
    QueueNotFoundError,
    QueueStore,
)
from deepgym.console.schemas import LabelQuery
from deepgym.labels import Label, LabelCreate, LabelTarget
from deepgym.review_queues import ItemCreate, Queue, QueueCreate, QueueUpdate, ReviewLabel
from tests.trace_samples import graded_trace

PASSED = ReviewLabel(name='passed', value=True, comment='Solves it.')
T1 = LabelTarget(kind='trace', id='t1')
STEP_2 = LabelTarget(kind='trace', id='t1', step=2)
STEP_3 = LabelTarget(kind='trace', id='t1', step=3)


def _queue(store: QueueStore, items: int = 3, **settings: object) -> Queue:
    """A queue of *items* traces, t1 to t<items>, asking for a passed verdict."""
    settings = {'name': 'calibration', 'label_names': ('passed',), **settings}
    queue = store.create(QueueCreate.model_validate(settings))
    store.add_items(queue.id, [_item(f't{n}') for n in range(1, items + 1)])
    return queue


def _item(trace_id: str, context: str | None = None) -> ItemCreate:
    return ItemCreate(target=LabelTarget(kind='trace', id=trace_id), context=context)


def _review(store: QueueStore, queue: Queue, reviewer: str) -> str | None:
    """Reserve and submit the next item for *reviewer*; return its target id."""
    view = store.next_item(queue.id, reviewer)
    if view is None:
        return None
    store.submit(queue.id, view.item.id, reviewer, [PASSED])
    return view.item.target.id


def _given(store: QueueStore, target: LabelTarget, author: str, **changes: object) -> Label:
    """Store a judge's 'quality' mark by *author* on *target*, outside any queue."""
    fields = {'target': target, 'name': 'quality', 'value': 'good', 'source': 'judge',
              'author': author, **changes}  # fmt: skip
    return LabelStore(store._path).add(LabelCreate.model_validate(fields))


def _assignments(path: Path) -> list[tuple[str, str, str]]:
    """Every stored (target id, reviewer, status), in storing order."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT json_extract(i.target, '$.id'), a.reviewer, a.status "
            'FROM queue_assignments AS a JOIN queue_items AS i ON i.id = a.item_id ORDER BY a.rowid'
        ).fetchall()
    return [tuple(row) for row in rows]


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / 'console.db'


@pytest.fixture()
def store(db: Path) -> QueueStore:
    return QueueStore(db)


class _Clock:
    """The time queue_store reads, moved by the test."""

    def __init__(self) -> None:
        self.at = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)

    def advance(self, seconds: float) -> None:
        self.at += timedelta(seconds=seconds)


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()

    class Frozen(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return clock.at

    monkeypatch.setattr(queue_store, 'datetime', Frozen)
    return clock


class TestSettings:
    @pytest.mark.parametrize(
        'changes',
        [
            {'overlap': 0},
            {'overlap': True},
            {'overlap': '2'},
            {'reservation_seconds': 0},
            {'reservation_seconds': 10**9 + 1},
            {'label_names': ('passed', 'passed')},
            {'label_names': ('passed ',)},
            {'reviewers': ('ana', 'ana')},
            {'reviewers': ('ana',), 'overlap': 2},
            {'blind': 'yes'},
            {'extra': 1},
        ],
    )
    def test_bad(self, changes: dict) -> None:
        with pytest.raises(ValidationError):
            QueueCreate.model_validate({'name': 'q', **changes})

    def test_defaults(self) -> None:
        queue = QueueCreate(name='q')

        assert (queue.overlap, queue.blind, queue.reservation_seconds) == (1, True, 3600)
        assert (queue.label_names, queue.reviewers) == ((), ())

    def test_create_reads_back(self, store: QueueStore) -> None:
        created = store.create(
            QueueCreate(
                name='calibration', description='Judge v3 check', instructions='Read the diff.',
                label_names=('passed', 'c1'), overlap=2, blind=False, reservation_seconds=60,
                reviewers=('ana', 'bo'),
            )
        )  # fmt: skip

        assert store.get(created.id) == created
        assert store.queues() == [created]
        assert not created.closed

    def test_a_retry_with_the_same_id_creates_once(self, store: QueueStore) -> None:
        first = store.create(QueueCreate(id='q1', name='calibration'))

        assert store.create(QueueCreate(id='q1', name='calibration')) == first
        with pytest.raises(ValueError, match='other settings'):
            store.create(QueueCreate(id='q1', name='calibration', overlap=2))
        assert store.queues() == [first]

    def test_unknown_queue(self, store: QueueStore) -> None:
        with pytest.raises(QueueNotFoundError, match='not found'):
            store.get('nope')
        with pytest.raises(QueueNotFoundError):
            store.next_item('nope', 'ana')
        with pytest.raises(QueueNotFoundError):
            store.add_items('nope', [_item('t1')])


class TestUpdate:
    def test_changes_what_may_change(self, store: QueueStore) -> None:
        queue = _queue(store, description='v1', reviewers=('ana',), reservation_seconds=60)

        updated = store.update(
            queue.id,
            QueueUpdate(description=None, instructions='Read the diff.', reviewers=('ana', 'bo'),
                        reservation_seconds=600, closed=True),
        )  # fmt: skip

        assert store.get(queue.id) == updated
        assert (updated.description, updated.instructions) == (None, 'Read the diff.')
        assert (updated.reviewers, updated.reservation_seconds, updated.closed) == (
            ('ana', 'bo'),
            600,
            True,
        )
        assert (updated.overlap, updated.blind, updated.label_names) == (1, True, ('passed',))
        assert store.update(queue.id, QueueUpdate()) == updated

    def test_a_new_reservation_time_applies_from_then_on(
        self, store: QueueStore, clock: _Clock
    ) -> None:
        queue = _queue(store, reservation_seconds=60)
        ana = store.next_item(queue.id, 'ana')

        store.update(queue.id, QueueUpdate(reservation_seconds=600))
        bo = store.next_item(queue.id, 'bo')

        assert ana.assignment.expires_at == clock.at + timedelta(seconds=60)
        assert bo.assignment.expires_at == clock.at + timedelta(seconds=600)
        assert store.view(queue.id, ana.item.id, 'ana').assignment == ana.assignment

    def test_reviewers_are_only_added(self, store: QueueStore) -> None:
        listed = _queue(store, reviewers=('ana', 'bo'))
        anyone = _queue(store)

        with pytest.raises(ValueError, match=r"drops \['bo'\]"):
            store.update(listed.id, QueueUpdate(reviewers=('ana', 'cy')))
        with pytest.raises(ValueError, match='lets anyone review'):
            store.update(anyone.id, QueueUpdate(reviewers=('ana',)))
        assert store.get(listed.id).reviewers == ('ana', 'bo')

    @pytest.mark.parametrize(
        'changes',
        [
            {'overlap': 2},
            {'blind': False},
            {'label_names': ['c1']},
            {'closed': None},
            {'reservation_seconds': None},
            {'reservation_seconds': 0},
            {'reviewers': None},
            {'reviewers': ['ana', 'ana']},
        ],
    )
    def test_bad(self, changes: dict) -> None:
        with pytest.raises(ValidationError):
            QueueUpdate.model_validate(changes)


class TestItems:
    def test_added_in_order_once_per_target(self, store: QueueStore) -> None:
        queue = store.create(QueueCreate(name='q'))
        first = store.add_items(queue.id, [_item('t1', 'Fix the port.'), _item('t2')])

        again = store.add_items(queue.id, [_item('t3'), _item('t1', 'Fix the port.')])

        assert [item.position for item in first] == [1, 2]
        assert [item.position for item in again] == [3, 1]
        assert again[1] == first[0]
        assert [entry.item for entry in store.items(queue.id).items] == [*first, again[0]]

    def test_another_context_adds_none(self, store: QueueStore) -> None:
        queue = _queue(store, items=1)

        with pytest.raises(ValueError, match='another context'):
            store.add_items(queue.id, [_item('t9'), _item('t1', 'other task')])

        assert store.items(queue.id).total == 1

    def test_steps_and_spans_are_items(self, store: QueueStore) -> None:
        queue = store.create(QueueCreate(name='q', label_names=('quality',)))
        span = LabelTarget(kind='trace', id='t1', step=2, field='action', span=(0, 3), quote='sed')

        stored = store.add_items(queue.id, [ItemCreate(target=span), ItemCreate(target=STEP_2)])

        assert [entry.item.target for entry in store.items(queue.id).items] == [span, STEP_2]
        assert len({item.id for item in stored}) == 2

    def test_a_verdict_queue_takes_whole_targets_only(self, store: QueueStore) -> None:
        queue = _queue(store, items=0, label_names=('passed', 'tone'))
        answer = LabelTarget(kind='run', id='r1', field='answer')
        items = [ItemCreate(target=T1), ItemCreate(target=STEP_2), ItemCreate(target=answer)]

        with pytest.raises(QueueInputError) as refused:
            store.add_items(queue.id, items)

        assert [error['loc'] for error in refused.value.errors] == [(1, 'target'), (2, 'target')]
        assert "verdicts ['passed']" in refused.value.errors[0]['msg']
        assert store.items(queue.id).total == 0

    def test_pages(self, store: QueueStore) -> None:
        queue = _queue(store, items=5)

        page = store.items(queue.id, 1, 2)

        assert [entry.item.target.id for entry in page.items] == ['t2', 't3']
        assert page.total == 5
        assert len(store.items(queue.id, 3).items) == 2


class TestNextItem:
    def test_overlap_hands_each_item_to_n_reviewers(self, store: QueueStore) -> None:
        queue = _queue(store, overlap=2)

        handed = {reviewer: _review(store, queue, reviewer) for reviewer in ('ana', 'bo', 'cy')}

        assert handed == {'ana': 't1', 'bo': 't1', 'cy': 't2'}

    def test_live_reservations_by_others_count(self, store: QueueStore) -> None:
        queue = _queue(store, items=2, overlap=2)

        held = [store.next_item(queue.id, reviewer) for reviewer in ('ana', 'bo', 'cy')]

        assert [view.item.target.id for view in held] == ['t1', 't1', 't2']

    def test_no_reviewer_reviews_an_item_twice(self, store: QueueStore) -> None:
        queue = _queue(store, overlap=2)

        reviewed = [_review(store, queue, 'ana') for _ in range(4)]

        assert reviewed == ['t1', 't2', 't3', None]
        assert _review(store, queue, 'bo') == 't1'

    def test_asking_again_extends_the_held_item(self, store: QueueStore, clock: _Clock) -> None:
        queue = _queue(store, reservation_seconds=60)
        first = store.next_item(queue.id, 'ana')
        clock.advance(50)

        again = store.next_item(queue.id, 'ana')

        assert again.assignment.id == first.assignment.id
        assert again.assignment.expires_at == clock.at + timedelta(seconds=60)
        clock.advance(50)
        assert store.view(queue.id, first.item.id, 'ana').assignment.status == 'reserved'
        assert [status for *_, status in _assignments(store._path)] == ['reserved']

    def test_a_renewal_never_shortens_the_reservation(
        self, store: QueueStore, clock: _Clock
    ) -> None:
        queue = _queue(store, reservation_seconds=600)
        first = store.next_item(queue.id, 'ana')
        store.update(queue.id, QueueUpdate(reservation_seconds=60))
        clock.advance(10)

        again = store.next_item(queue.id, 'ana')

        assert again.assignment.expires_at == first.assignment.expires_at
        clock.advance(590)
        renewed = store.next_item(queue.id, 'ana')
        assert renewed.assignment.expires_at == clock.at + timedelta(seconds=60)
        assert renewed.assignment.id != first.assignment.id

    def test_released_items_come_back_last(self, store: QueueStore) -> None:
        queue = _queue(store)
        handed = []

        for _ in range(4):
            view = store.next_item(queue.id, 'ana')
            handed.append(view.item.target.id)
            released = store.release(queue.id, view.item.id, 'ana')

        assert handed == ['t1', 't2', 't3', 't1']
        assert (released.status, released.finished_at is not None) == ('released', True)
        assert store.next_item(queue.id, 'bo').item.target.id == 't1'

    def test_a_skipped_item_never_comes_back(self, store: QueueStore) -> None:
        queue = _queue(store, items=2)

        for _ in range(2):
            view = store.next_item(queue.id, 'ana')
            assert store.skip(queue.id, view.item.id, 'ana').status == 'skipped'

        assert store.next_item(queue.id, 'ana') is None
        assert store.next_item(queue.id, 'bo').item.target.id == 't1'

    def test_only_listed_reviewers(self, store: QueueStore) -> None:
        queue = _queue(store, reviewers=('ana', 'bo'))
        held = store.next_item(queue.id, 'ana')
        item_id = held.item.id

        for call in (
            lambda: store.next_item(queue.id, 'cy'),
            lambda: store.view(queue.id, item_id, 'cy'),
            lambda: store.submit(queue.id, item_id, 'cy', [PASSED]),
            lambda: store.release(queue.id, item_id, 'cy'),
            lambda: store.skip(queue.id, item_id, 'cy'),
        ):
            with pytest.raises(NotReviewerError, match='cy is not a reviewer'):
                call()
        assert store.next_item(queue.id, 'bo') is not None

    def test_a_reviewer_is_a_name(self, store: QueueStore) -> None:
        queue = _queue(store)

        with pytest.raises(ValidationError):
            store.next_item(queue.id, ' ana')

    def test_a_closed_queue_hands_out_nothing_but_takes_held_reviews(
        self, store: QueueStore
    ) -> None:
        queue = _queue(store)
        held = store.next_item(queue.id, 'ana')

        closed = store.update(queue.id, QueueUpdate(closed=True))

        assert closed.closed and store.get(queue.id).closed
        for reviewer in ('ana', 'bo'):
            with pytest.raises(ValueError, match='closed'):
                store.next_item(queue.id, reviewer)
        done = store.submit(queue.id, held.item.id, 'ana', [PASSED])
        assert done.assignment.status == 'submitted'
        assert not store.update(queue.id, QueueUpdate(closed=False)).closed
        assert store.next_item(queue.id, 'bo').item.target.id == 't2'


class TestExpiry:
    def test_an_expired_reservation_is_reclaimed(self, store: QueueStore, clock: _Clock) -> None:
        queue = _queue(store, items=1, reservation_seconds=60)
        held = store.next_item(queue.id, 'ana')
        clock.advance(59)

        assert store.next_item(queue.id, 'bo') is None

        clock.advance(1)
        bo = store.next_item(queue.id, 'bo')

        assert bo.item == held.item
        assert bo.assignment.expires_at == clock.at + timedelta(seconds=60)
        with pytest.raises(ValueError, match='expired at 2026-11-19T12:01:00.*every review'):
            store.submit(queue.id, held.item.id, 'ana', [PASSED])
        ana = store.view(queue.id, held.item.id, 'ana').assignment
        assert (ana.status, ana.finished_at) == ('expired', held.assignment.expires_at)
        assert _assignments(store._path) == [('t1', 'ana', 'expired'), ('t1', 'bo', 'reserved')]

    def test_a_timed_out_reservation_reads_as_expired(
        self, store: QueueStore, clock: _Clock
    ) -> None:
        queue = _queue(store, items=1, reservation_seconds=60)
        held = store.next_item(queue.id, 'ana')
        clock.advance(60)

        with pytest.raises(ValueError, match='expired'):
            store.release(queue.id, held.item.id, 'ana')
        assert store.view(queue.id, held.item.id, 'ana').assignment.status == 'expired'
        assert store.progress(queue.id).reviewers[0].expired == 1
        assert store.next_item(queue.id, 'ana').item == held.item

    def test_a_late_review_takes_a_free_slot(self, store: QueueStore, clock: _Clock) -> None:
        queue = _queue(store, items=1, reservation_seconds=60)
        held = store.next_item(queue.id, 'ana')
        clock.advance(61)

        done = store.submit(queue.id, held.item.id, 'ana', [PASSED])

        assert (done.assignment.id, done.assignment.status) == (held.assignment.id, 'submitted')
        assert _assignments(store._path) == [('t1', 'ana', 'submitted')]
        assert store.next_item(queue.id, 'bo') is None

    def test_a_released_turn_can_be_submitted_while_a_slot_is_free(self, store: QueueStore) -> None:
        queue = _queue(store, items=1)
        held = store.next_item(queue.id, 'ana')
        store.release(queue.id, held.item.id, 'ana')

        assert store.submit(queue.id, held.item.id, 'ana', [PASSED]).assignment.status == (
            'submitted'
        )

    def test_late_reviews_are_refused_when_closed_or_skipped(
        self, store: QueueStore, clock: _Clock
    ) -> None:
        queue = _queue(store, items=2, reservation_seconds=60)
        skipped = store.next_item(queue.id, 'ana')
        store.skip(queue.id, skipped.item.id, 'ana')
        late = store.next_item(queue.id, 'ana')
        clock.advance(61)
        store.update(queue.id, QueueUpdate(closed=True))

        with pytest.raises(ValueError, match='ana skipped item'):
            store.submit(queue.id, skipped.item.id, 'ana', [PASSED])
        with pytest.raises(ValueError, match='is closed, and .* expired'):
            store.submit(queue.id, late.item.id, 'ana', [PASSED])


class TestSubmit:
    def test_labels_are_the_reviewers_with_blind_and_group(self, store: QueueStore) -> None:
        queue = _queue(store, label_names=('passed', 'c1'))
        held = store.next_item(queue.id, 'ana')
        mark = LabelTarget(kind='trace', id=held.item.target.id, step=2)

        done = store.submit(
            queue.id, held.item.id, 'ana',
            [PASSED, ReviewLabel(name='c1', value=False, id='ana-c1'),
             ReviewLabel(target=mark, name='mark', value='wrong-file')],
        )  # fmt: skip

        assert done.assignment.status == 'submitted'
        assert [(label.name, label.target) for label in done.labels] == [
            ('passed', held.item.target),
            ('c1', held.item.target),
            ('mark', mark),
        ]
        assert {(label.source, label.author, label.blind) for label in done.labels} == {
            ('human', 'ana', True)
        }
        assert {label.group_id for label in done.labels} == {held.assignment.id}
        assert done.labels[1].id == 'ana-c1'

    @pytest.mark.parametrize(
        ('labels', 'loc', 'match'),
        [
            ([], ('labels',), 'at least one label'),
            ([ReviewLabel(name='c1', value=True)], ('labels',), r"no \['passed'\]"),
            (
                [ReviewLabel(target=STEP_2, name='passed', value=True, comment='step')],
                ('labels', 0),
                'verdict on a whole target',
            ),
            (
                [PASSED, ReviewLabel(target=LabelTarget(kind='trace', id='t2'), name='m',
                                     value=1)],
                ('labels', 1, 'target'),
                'or inside it',
            ),
            ([ReviewLabel(name='passed', value=True)], ('labels', 0), 'need a comment'),
            (
                [ReviewLabel(name='passed', value=True, comment='ok', id='a'),
                 ReviewLabel(name='note', value='x', id='a')],
                ('labels', 1, 'id'),
                "repeats an earlier label's id",
            ),
        ],
    )  # fmt: skip
    def test_bad_input_stores_nothing(
        self, store: QueueStore, labels: list[ReviewLabel], loc: tuple, match: str
    ) -> None:
        queue = _queue(store)
        held = store.next_item(queue.id, 'ana')

        with pytest.raises(QueueInputError, match=match) as refused:
            store.submit(queue.id, held.item.id, 'ana', labels)

        assert refused.value.errors[0]['loc'] == loc
        assert LabelStore(store._path).current('trace', 't1') == []
        assert store.view(queue.id, held.item.id, 'ana').assignment.status == 'reserved'

    def test_a_refused_label_stores_nothing(self, store: QueueStore) -> None:
        queue = _queue(store)
        held = store.next_item(queue.id, 'ana')

        with pytest.raises(ValueError, match='already has a current'):
            store.submit(queue.id, held.item.id, 'ana', [PASSED, PASSED])

        assert LabelStore(store._path).current('trace', 't1') == []
        assert store.view(queue.id, held.item.id, 'ana').assignment.status == 'reserved'

    @pytest.mark.parametrize(
        ('target', 'inside'),
        [
            (None, True),
            ({'step': 2, 'field': 'action', 'span': (0, 10)}, True),
            ({'step': 2, 'field': 'action', 'span': (2, 5), 'quote': 'abc'}, True),
            ({'step': 2, 'field': 'action', 'span': (5, 12)}, False),
            ({'step': 2, 'field': 'action'}, False),
            ({'step': 2, 'field': 'observation', 'span': (0, 10)}, False),
            ({'step': 3, 'field': 'action', 'span': (0, 10)}, False),
            ({'step': 2}, False),
            ({}, False),
        ],
    )
    def test_labels_are_on_the_item_or_inside_it(
        self, store: QueueStore, target: dict | None, inside: bool
    ) -> None:
        span = LabelTarget(kind='trace', id='t1', step=2, field='action', span=(0, 10))
        queue = store.create(QueueCreate(name='spans', label_names=('quality',)))
        item = store.add_items(queue.id, [ItemCreate(target=span)])[0]
        store.next_item(queue.id, 'ana')
        mark = ReviewLabel(name='quality', value='good')
        given = None if target is None else LabelTarget(kind='trace', id='t1', **target)
        labels = [mark, ReviewLabel(target=given, name='note', value='typo')]

        if inside:
            assert len(store.submit(queue.id, item.id, 'ana', labels).labels) == 2
        else:
            with pytest.raises(QueueInputError, match='or inside it'):
                store.submit(queue.id, item.id, 'ana', labels)

    def test_labels_and_assignment_commit_together(
        self, store: QueueStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        queue = _queue(store)
        held = store.next_item(queue.id, 'ana')

        def fail(*_args: object) -> None:
            raise RuntimeError('disk full')

        monkeypatch.setattr(queue_store, '_finish', fail)
        with pytest.raises(RuntimeError):
            store.submit(queue.id, held.item.id, 'ana', [PASSED])

        assert LabelStore(store._path).current('trace', 't1') == []
        assert _assignments(store._path) == [('t1', 'ana', 'reserved')]

    def test_a_repeated_submit_returns_the_stored_review(self, store: QueueStore) -> None:
        queue = _queue(store, items=1, label_names=('passed', 'c1'))
        held = store.next_item(queue.id, 'ana')
        labels = [PASSED, ReviewLabel(name='c1', value=False, id='ana-c1')]
        done = store.submit(queue.id, held.item.id, 'ana', labels)

        again = store.submit(queue.id, held.item.id, 'ana', labels)
        unnamed = store.submit(
            queue.id, held.item.id, 'ana', [PASSED, ReviewLabel(name='c1', value=False)]
        )

        assert again == unnamed == done
        assert len(LabelStore(store._path).current('trace', 't1')) == 2
        for changed in (
            [PASSED, ReviewLabel(name='c1', value=True)],
            [PASSED, ReviewLabel(name='c1', value=False, id='other')],
            [*labels, ReviewLabel(name='note', value='typo')],
        ):
            with pytest.raises(ValueError, match='ana already reviewed'):
                store.submit(queue.id, held.item.id, 'ana', changed)

    def test_only_a_holder_submits(self, store: QueueStore) -> None:
        queue = _queue(store, items=1)
        held = store.next_item(queue.id, 'ana')

        with pytest.raises(ValueError, match='bo holds no reservation'):
            store.submit(queue.id, held.item.id, 'bo', [PASSED])
        with pytest.raises(ValueError, match='bo holds no reservation'):
            store.release(queue.id, held.item.id, 'bo')
        with pytest.raises(QueueNotFoundError, match='has no item'):
            store.submit(queue.id, 'nope', 'ana', [PASSED])
        store.submit(queue.id, held.item.id, 'ana', [PASSED])
        with pytest.raises(ValueError, match='ana already reviewed'):
            store.release(queue.id, held.item.id, 'ana')

        assert len(LabelStore(store._path).current('trace', 't1')) == 1

    def test_a_review_counts_as_a_new_group(self, store: QueueStore) -> None:
        labels = LabelStore(store._path)
        labels.add(
            LabelCreate(target=T1, name='passed', value=False, source='human', author='ana',
                        comment='earlier pass')
        )  # fmt: skip
        queue = _queue(store, items=1)

        assert _review(store, queue, 'ana') == 't1'
        assert len(labels.current('trace', 't1')) == 2


class TestBlind:
    def _judged(self, store: QueueStore, **settings: object) -> Queue:
        """A one-item queue whose trace a judge labeled and bo reviewed."""
        queue = _queue(store, items=1, overlap=2, **settings)
        _given(store, T1, 'judge-a', name='passed', value=True)
        _review(store, queue, 'bo')
        return queue

    def test_hidden_until_submitted(self, store: QueueStore) -> None:
        queue = self._judged(store)

        held = store.next_item(queue.id, 'ana')
        viewed = store.view(queue.id, held.item.id, 'ana')
        done = store.submit(queue.id, held.item.id, 'ana', [PASSED])

        assert held.labels == viewed.labels == []
        assert [(label.author, label.blind) for label in done.labels] == [
            ('judge-a', None),
            ('bo', True),
            ('ana', True),
        ]
        assert store.view(queue.id, held.item.id, 'ana').labels == done.labels
        assert len(store.view(queue.id, held.item.id, 'bo').labels) == 3
        assert store.view(queue.id, held.item.id, 'cy').labels == []

    def test_shown_outside_blind_mode_after_a_turn(self, store: QueueStore) -> None:
        queue = self._judged(store, blind=False)

        held = store.next_item(queue.id, 'ana')
        done = store.submit(queue.id, held.item.id, 'ana', [PASSED])

        assert [label.author for label in held.labels] == ['judge-a', 'bo']
        assert done.labels[-1].blind is False
        assert store.view(queue.id, held.item.id, 'cy').labels == []

    def test_sibling_steps_show_only_own_labels_and_stay_blind(self, store: QueueStore) -> None:
        queue = store.create(QueueCreate(name='steps', label_names=('quality',)))
        step_2, step_3 = store.add_items(
            queue.id, [ItemCreate(target=STEP_2), ItemCreate(target=STEP_3)]
        )
        _given(store, STEP_2, 'judge-a')
        _given(store, STEP_3, 'judge-a')
        _given(store, T1, 'judge-a', name='passed', value=True)
        _given(store, LabelTarget(kind='trace', id='t1', step=2, field='action', span=(0, 2)),
               'judge-a')  # fmt: skip
        mark = ReviewLabel(name='quality', value='poor')
        own_span = ReviewLabel(
            target=LabelTarget(kind='trace', id='t1', step=2, field='action', span=(3, 5)),
            name='note', value='typo',
        )  # fmt: skip

        store.next_item(queue.id, 'ana')
        first = store.submit(queue.id, step_2.id, 'ana', [mark, own_span])
        store.next_item(queue.id, 'ana')
        second = store.submit(queue.id, step_3.id, 'ana', [mark])

        assert [(label.author, label.target) for label in first.labels] == [
            ('ana', STEP_2),
            ('ana', own_span.target),
        ]
        assert [label.blind for label in second.labels if label.author == 'ana'] == [True] * 3
        assert len(second.labels) == 7
        assert len(store.view(queue.id, step_2.id, 'ana').labels) == 7
        later = _queue(store, items=1)
        assert _review(store, later, 'ana') == 't1'
        assert LabelStore(store._path).current('trace', 't1')[-1].blind is False

    @pytest.mark.parametrize(
        ('other', 'shown'),
        [
            ({}, ['ana']),
            ({'closed': True}, ['judge-a', 'ana']),
            ({'reviewers': ('bo',)}, ['judge-a', 'ana']),
            ({'blind': False}, ['judge-a', 'ana']),
            ({'reviewed_by': 'bo'}, ['bo', 'judge-a', 'ana']),
            ({'skipped': True}, ['judge-a', 'ana']),
            ({'held': True, 'closed': True}, ['ana']),
            ({'held': True, 'closed': True, 'expired': True}, ['judge-a', 'ana']),
        ],
    )
    def test_siblings_in_other_blind_queues_count_while_reachable(
        self, store: QueueStore, clock: _Clock, other: dict, shown: list[str]
    ) -> None:
        settings = {key: other[key] for key in ('reviewers', 'blind') if key in other}
        elsewhere = _queue(store, items=1, reservation_seconds=60, **settings)
        if other.get('reviewed_by'):
            _review(store, elsewhere, other['reviewed_by'])
        if other.get('skipped'):
            store.skip(elsewhere.id, store.next_item(elsewhere.id, 'ana').item.id, 'ana')
        if other.get('held'):
            store.next_item(elsewhere.id, 'ana')
        if other.get('closed'):
            store.update(elsewhere.id, QueueUpdate(closed=True))
        if other.get('expired'):
            clock.advance(60)
        _given(store, T1, 'judge-a', name='passed', value=True)
        here = _queue(store, items=1)

        held = store.next_item(here.id, 'ana')
        done = store.submit(here.id, held.item.id, 'ana', [PASSED])

        assert [label.author for label in done.labels] == shown

    def test_the_exposure_commits_with_the_response(
        self, store: QueueStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _given(store, T1, 'judge-a', name='passed', value=True)
        queue = _queue(store, items=1)
        held = store.next_item(queue.id, 'ana')
        monkeypatch.setattr(queue_store, '_EXPOSE', 'INSERT INTO missing VALUES (1)')

        with pytest.raises(sqlite3.OperationalError):
            store.submit(queue.id, held.item.id, 'ana', [PASSED])

        assert [label.author for label in LabelStore(store._path).current('trace', 't1')] == [
            'judge-a'
        ]
        assert store.view(queue.id, held.item.id, 'ana').assignment.status == 'reserved'

    def test_only_the_first_exposure_is_written(self, store: QueueStore, clock: _Clock) -> None:
        _given(store, T1, 'judge-a', name='passed', value=True)
        queue = _queue(store, items=1)
        held = store.next_item(queue.id, 'ana')
        store.submit(queue.id, held.item.id, 'ana', [PASSED])
        first = clock.at
        clock.advance(60)

        again = store.view(queue.id, held.item.id, 'ana')

        assert [label.author for label in again.labels] == ['judge-a', 'ana']
        with connect(store._path) as conn:
            rows = [tuple(row) for row in conn.execute('SELECT * FROM queue_exposures')]
        assert rows == [('ana', 'trace', 't1', iso_utc(first))]

    def test_only_shown_labels_unblind(self, store: QueueStore) -> None:
        first = _queue(store, items=1)
        alone = store.submit(first.id, store.next_item(first.id, 'ana').item.id, 'ana', [PASSED])
        _given(store, T1, 'judge-a', name='passed', value=True)
        second = _queue(store, items=1)

        held = store.next_item(second.id, 'ana')
        done = store.submit(second.id, held.item.id, 'ana', [PASSED])
        third = _queue(store, items=1)
        _review(store, third, 'ana')

        assert [label.author for label in alone.labels] == ['ana']
        assert [label.author for label in done.labels] == ['ana', 'judge-a', 'ana']
        assert done.labels[-1].blind is True
        assert [label.blind for label in LabelStore(store._path).current('trace', 't1')][
            -1
        ] is False

    def test_an_episode_and_its_trace_are_one_recording(self, store: QueueStore) -> None:
        trace = graded_trace()
        episode = LabelTarget(kind='episode', id=record_episode(trace, db_path=store._path))
        recorded = LabelTarget(kind='trace', id=trace.trace_id)
        _given(store, episode, 'judge-a', name='passed', value=True)
        _given(store, recorded, 'judge-a', name='passed', value=True)
        triage = store.create(QueueCreate(name='triage', label_names=('passed',), blind=False))
        store.add_items(triage.id, [ItemCreate(target=episode)])
        blind = store.create(QueueCreate(name='blind', label_names=('passed',), overlap=2))
        store.add_items(blind.id, [ItemCreate(target=recorded)])
        pending = store.create(QueueCreate(name='pending', label_names=('passed',)))
        store.add_items(pending.id, [ItemCreate(target=episode)])

        bo = store.submit(blind.id, store.next_item(blind.id, 'bo').item.id, 'bo', [PASSED])
        seen = store.next_item(triage.id, 'ana')
        ana = store.submit(blind.id, store.next_item(blind.id, 'ana').item.id, 'ana', [PASSED])

        assert [label.author for label in bo.labels] == ['bo']
        assert bo.labels[0].blind is True
        assert [label.author for label in seen.labels] == ['judge-a']
        assert {label.author: label.blind for label in ana.labels if label.author == 'ana'} == {
            'ana': False
        }

    def test_skipped_siblings_dont_hide(self, store: QueueStore) -> None:
        queue = store.create(QueueCreate(name='steps', label_names=('quality',)))
        step_2, step_3 = store.add_items(
            queue.id, [ItemCreate(target=STEP_2), ItemCreate(target=STEP_3)]
        )
        _given(store, STEP_3, 'judge-a')

        store.next_item(queue.id, 'ana')
        store.skip(queue.id, step_2.id, 'ana')
        store.next_item(queue.id, 'ana')
        done = store.submit(queue.id, step_3.id, 'ana', [ReviewLabel(name='quality', value='ok')])

        assert len(done.labels) == 2

    def test_a_triage_turn_unblinds_later_reviews(self, store: QueueStore) -> None:
        _given(store, T1, 'judge-a', name='passed', value=True)
        triage = _queue(store, items=1, blind=False)
        seen = store.next_item(triage.id, 'ana')
        store.release(triage.id, seen.item.id, 'ana')
        blind = _queue(store, items=1, overlap=2)

        reviews = {reviewer: store.next_item(blind.id, reviewer) for reviewer in ('ana', 'bo')}
        done = {
            reviewer: store.submit(blind.id, view.item.id, reviewer, [PASSED])
            for reviewer, view in reviews.items()
        }

        assert [label.author for label in seen.labels] == ['judge-a']
        blind_flags = {
            label.author: label.blind for label in done['bo'].labels if label.source == 'human'
        }
        assert blind_flags == {'ana': False, 'bo': True}


def test_progress(store: QueueStore, clock: _Clock) -> None:
    queue = _queue(store, items=4, overlap=2, reservation_seconds=60)
    for reviewer in ('ana', 'bo'):
        _review(store, queue, reviewer)  # t1 complete
    _review(store, queue, 'ana')  # t2 one review
    held = store.next_item(queue.id, 'cy')  # t2 reserved
    store.skip(queue.id, held.item.id, 'cy')
    store.next_item(queue.id, 'cy')  # t3 reserved
    store.next_item(queue.id, 'di')  # t2 reserved
    clock.advance(30)
    store.next_item(queue.id, 'ed')  # t3 reserved; t4 untouched
    clock.advance(31)  # cy's and di's run out

    progress = store.progress(queue.id)

    assert progress.model_dump(exclude={'reviewers'}) == {
        'queue_id': queue.id, 'items': 4, 'complete': 1, 'in_progress': 2, 'reviews': 3,
        'reviews_needed': 8, 'reserved': 1, 'blocked': 0,
    }  # fmt: skip
    assert [tuple(row.model_dump().values()) for row in progress.reviewers] == [
        ('ana', 2, 0, 0, 0, 0),
        ('bo', 1, 0, 0, 0, 0),
        ('cy', 0, 0, 1, 0, 1),
        ('di', 0, 0, 0, 0, 1),
        ('ed', 0, 1, 0, 0, 0),
    ]
    page = store.items(queue.id)
    assert [(e.item.target.id, e.reviews, e.needed, e.reserved) for e in page.items] == [
        ('t1', 2, 2, 0),
        ('t2', 1, 2, 0),
        ('t3', 0, 2, 1),
        ('t4', 0, 2, 0),
    ]


def test_items_skipped_by_too_many_listed_reviewers_are_blocked(store: QueueStore) -> None:
    queue = _queue(store, items=2, overlap=2, reviewers=('ana', 'bo'))
    held = store.next_item(queue.id, 'ana')

    store.skip(queue.id, held.item.id, 'ana')

    assert store.progress(queue.id).blocked == 1
    store.update(queue.id, QueueUpdate(reviewers=('ana', 'bo', 'cy')))
    assert store.progress(queue.id).blocked == 0
    open_queue = _queue(store, items=1, overlap=2)
    store.skip(open_queue.id, store.next_item(open_queue.id, 'ana').item.id, 'ana')
    assert store.progress(open_queue.id).blocked == 0


def _check_reviews(path: Path, items: int, overlap: int) -> None:
    """Every item was reviewed by exactly *overlap* different reviewers, each once, and
    reservation times follow the order the reservations were stored in."""
    rows = _assignments(path)
    submitted = [(target, who) for target, who, status in rows if status == 'submitted']
    assert len(set(submitted)) == len(submitted)
    assert Counter(target for target, _ in submitted) == {
        f't{n}': overlap for n in range(1, items + 1)
    }
    assert [status for *_, status in rows] == ['submitted'] * len(rows)
    assert LabelStore(path).page(LabelQuery(), 0, None).total == items * overlap
    with connect(path) as conn:
        times = [row[0] for row in conn.execute('SELECT reserved_at FROM queue_assignments')]
    assert times == sorted(times)


def test_threads_never_reserve_beyond_overlap(db: Path) -> None:
    queue = _queue(QueueStore(db), items=1, overlap=2)
    barrier = threading.Barrier(10)
    got: list[str | None] = []

    def ask(reviewer: str) -> None:
        store = QueueStore(db)
        barrier.wait()
        view = store.next_item(queue.id, reviewer)
        got.append(None if view is None else view.item.target.id)

    threads = [threading.Thread(target=ask, args=(f'r{n}',)) for n in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert Counter(got) == {'t1': 2, None: 8}


def test_threads_review_every_item_n_times(db: Path) -> None:
    queue = _queue(QueueStore(db), items=10, overlap=3)

    def work(reviewer: str) -> None:
        store = QueueStore(db)
        while _review(store, queue, reviewer) is not None:
            pass

    threads = [threading.Thread(target=work, args=(f'r{n}',)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    _check_reviews(db, items=10, overlap=3)


def _work_in_process(db: str, queue_id: str, reviewer: str, barrier: Barrier) -> None:
    """Review every item left for *reviewer*, from another process."""
    store = QueueStore(Path(db))
    queue = store.get(queue_id)
    barrier.wait()
    while _review(store, queue, reviewer) is not None:
        pass


def test_processes_review_every_item_n_times(db: Path) -> None:
    queue = _queue(QueueStore(db), items=12, overlap=3)
    context = multiprocessing.get_context('spawn')
    barrier = context.Barrier(4)
    processes = [
        context.Process(target=_work_in_process, args=(str(db), queue.id, f'r{n}', barrier))
        for n in range(4)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)

    assert [process.exitcode for process in processes] == [0, 0, 0, 0]
    _check_reviews(db, items=12, overlap=3)
