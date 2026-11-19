"""Tests for review queues: settings, items, reservations, reviews, blind mode and progress."""

import json
import multiprocessing
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from multiprocessing.synchronize import Barrier
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepgym.console import episode_store, item_content, queue_store
from deepgym.console.db import TRACE_KEY, connect, iso_utc
from deepgym.console.episode_store import record_episode
from deepgym.console.item_content import ConflictingRecordingError
from deepgym.console.label_store import LabelStore
from deepgym.console.queue_store import (
    NotReviewerError,
    QueueInputError,
    QueueNotFoundError,
    QueueStore,
)
from deepgym.console.schemas import LabelQuery
from deepgym.labels import Label, LabelCreate, LabelTarget
from deepgym.models import Action, Observation
from deepgym.registry import load_environment
from deepgym.review_queues import (
    ItemContent,
    ItemCreate,
    Queue,
    QueueCreate,
    QueueUpdate,
    ReviewLabel,
    ReviewView,
)
from deepgym.traces import Trace
from deepgym.traces.schema import environment_sha256
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


def _exposed(path: Path) -> list[str]:
    """Every reviewer with a recorded exposure, by name."""
    with connect(path) as conn:
        rows = conn.execute('SELECT reviewer FROM queue_exposures ORDER BY reviewer')
        return [row[0] for row in rows]


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
            {'reveal': 'never'},
            {'extra': 1},
        ],
    )
    def test_bad(self, changes: dict) -> None:
        with pytest.raises(ValidationError):
            QueueCreate.model_validate({'name': 'q', **changes})

    def test_defaults(self) -> None:
        queue = QueueCreate(name='q')

        assert (queue.overlap, queue.blind, queue.reservation_seconds) == (1, True, 3600)
        assert (queue.label_names, queue.reviewers, queue.reveal) == ((), (), 'after_submit')

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
        for other in ({'overlap': 2}, {'reveal': 'on_close'}):
            with pytest.raises(ValueError, match='other settings'):
                store.create(QueueCreate(id='q1', name='calibration', **other))
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
            {'reveal': None},
            {'reveal': 'never'},
        ],
    )
    def test_bad(self, changes: dict) -> None:
        with pytest.raises(ValidationError):
            QueueUpdate.model_validate(changes)

    def test_reveal_changes_only_until_a_review_is_submitted(self, store: QueueStore) -> None:
        queue = _queue(store, items=2)
        held = store.next_item(queue.id, 'ana')

        reserved = store.update(queue.id, QueueUpdate(reveal='on_close'))
        store.submit(queue.id, held.item.id, 'ana', [PASSED])

        assert reserved.reveal == 'on_close'
        with pytest.raises(ValueError, match="submitted reviews under reveal 'on_close'"):
            store.update(queue.id, QueueUpdate(reveal='after_submit', closed=True))
        assert (store.get(queue.id).reveal, store.get(queue.id).closed) == ('on_close', False)
        assert store.update(queue.id, QueueUpdate(reveal='on_close', closed=True)).closed

    def test_reveal_freezes_at_the_first_turn_outside_blind_mode(self, store: QueueStore) -> None:
        queue = _queue(store, items=1, blind=False)

        store.next_item(queue.id, 'ana')

        with pytest.raises(ValueError, match="handed out items under reveal 'after_submit'"):
            store.update(queue.id, QueueUpdate(reveal='on_close'))
        assert store.get(queue.id).reveal == 'after_submit'


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


class TestReveal:
    def _reviewed(
        self, store: QueueStore, reveal: str, **settings: object
    ) -> tuple[Queue, ReviewView, ReviewView]:
        """A one-item queue on t1, which a judge labeled and bo reviewed; ana's view of the
        item when handed it, and when she submitted her review."""
        queue = _queue(store, items=1, overlap=2, reveal=reveal, **settings)
        _given(store, T1, 'judge-a', name='passed', value=True)
        _review(store, queue, 'bo')
        held = store.next_item(queue.id, 'ana')
        return queue, held, store.submit(queue.id, held.item.id, 'ana', [PASSED])

    @pytest.mark.parametrize(
        ('reveal', 'shown', 'exposed'),
        [
            ('after_submit', ['judge-a', 'bo', 'ana'], ['ana', 'bo']),
            ('humans_only', ['bo', 'ana'], ['ana']),
            ('on_close', ['ana'], []),
        ],
    )
    def test_what_a_reviewer_sees_once_submitted(
        self, store: QueueStore, reveal: str, shown: list[str], exposed: list[str]
    ) -> None:
        queue, held, done = self._reviewed(store, reveal)
        viewed = store.view(queue.id, held.item.id, 'ana')
        exposures = _exposed(store._path)
        later = _queue(store, items=1)

        _review(store, later, 'ana')

        assert held.labels == []
        assert [label.author for label in done.labels] == shown
        assert [label.author for label in viewed.labels] == shown
        assert exposures == exposed
        assert LabelStore(store._path).current('trace', 't1')[-1].blind is ('ana' not in exposed)

    def test_on_close_shows_every_label_once_closed(self, store: QueueStore) -> None:
        queue, held, _ = self._reviewed(store, 'on_close')
        bo = store.view(queue.id, held.item.id, 'bo')

        store.update(queue.id, QueueUpdate(closed=True))
        closed = store.view(queue.id, held.item.id, 'ana')
        later = _queue(store, items=1)
        _review(store, later, 'ana')

        assert [label.author for label in bo.labels] == ['bo']
        assert [label.author for label in closed.labels] == ['judge-a', 'bo', 'ana']
        assert _exposed(store._path) == ['ana']
        assert LabelStore(store._path).current('trace', 't1')[-1].blind is False

    def test_humans_only_never_shows_a_judge_or_a_verifier(self, store: QueueStore) -> None:
        _given(store, T1, 'judge-a', name='passed', value=True)
        _given(store, T1, 'tests', source='verifier', name='score', value=1.0)
        queue = _queue(store, items=1, reveal='humans_only')
        held = store.next_item(queue.id, 'ana')

        done = store.submit(queue.id, held.item.id, 'ana', [PASSED])
        exposures = _exposed(store._path)
        _review(store, _queue(store, items=1), 'ana')

        assert [label.author for label in done.labels] == ['ana']
        assert exposures == []
        assert LabelStore(store._path).current('trace', 't1')[-1].blind is True

    @pytest.mark.parametrize(
        ('reveal', 'handed', 'submitted', 'exposed'),
        [
            ('after_submit', ['judge-a', 'bo'], ['judge-a', 'bo', 'ana'], ['ana', 'bo']),
            ('humans_only', ['bo'], ['bo', 'ana'], ['ana']),
            ('on_close', [], ['ana'], []),
        ],
    )
    def test_outside_blind_mode(
        self,
        store: QueueStore,
        reveal: str,
        handed: list[str],
        submitted: list[str],
        exposed: list[str],
    ) -> None:
        _, held, done = self._reviewed(store, reveal, blind=False)

        assert [label.author for label in held.labels] == handed
        assert [label.author for label in done.labels] == submitted
        assert _exposed(store._path) == exposed
        assert done.labels[-1].blind is False


def _without_step_verifier(trace: Trace) -> Trace:
    """*trace* as if its environment had no step verifier: no fingerprint of one, and no
    step reward or step verifier output."""
    env = trace.provenance.env.model_copy(update={'step_verifier_sha256': None})
    provenance = trace.provenance.model_copy(update={'env': env})
    steps = [
        step.model_copy(update={'reward': None, 'step_verifier': None}) for step in trace.steps
    ]
    return trace.model_copy(update={'provenance': provenance, 'steps': tuple(steps)})


def _with_steps(trace: Trace, **changes: dict) -> Trace:
    """*trace* with each step's fields changed, by step index: ``s1={'error': ...}``."""
    steps = [step.model_copy(update=changes.get(f's{step.index}', {})) for step in trace.steps]
    return trace.model_copy(update={'steps': tuple(steps)})


class TestContent:
    def _read(self, store: QueueStore, target: LabelTarget) -> ItemContent:
        """The content of a one-item queue on *target*, as ana, who holds it."""
        queue = store.create(QueueCreate(name='q'))
        item = store.add_items(queue.id, [ItemCreate(target=target)])[0]
        store.next_item(queue.id, 'ana')
        return store.content(queue.id, item.id, 'ana')

    def test_readable_while_a_review_could_be_submitted(
        self, store: QueueStore, clock: _Clock
    ) -> None:
        record_episode(graded_trace(), db_path=store._path)
        queue = store.create(
            QueueCreate(name='q', label_names=('passed',), overlap=2, reservation_seconds=60,
                        reviewers=('ana', 'bo', 'cy'))
        )  # fmt: skip
        item = store.add_items(queue.id, [_item('trace0001')])[0]

        def read(reviewer: str) -> str:
            return store.content(queue.id, item.id, reviewer).item.target.id

        def refused(reviewer: str, match: str) -> None:
            with pytest.raises(ValueError, match=match):
                read(reviewer)

        refused('ana', 'ana holds no reservation')
        store.next_item(queue.id, 'ana')
        assert read('ana') == 'trace0001'
        refused('bo', 'bo holds no reservation')
        store.release(queue.id, item.id, 'ana')
        assert read('ana') == 'trace0001'  # released, with a slot free: a submit is taken
        store.next_item(queue.id, 'ana')
        clock.advance(60)
        assert read('ana') == 'trace0001'  # expired, with a slot free
        store.next_item(queue.id, 'bo')
        store.next_item(queue.id, 'cy')
        refused('ana', 'every review and reservation it needs')
        store.submit(queue.id, item.id, 'bo', [PASSED])
        store.skip(queue.id, item.id, 'cy')
        refused('cy', 'cy skipped item')
        store.next_item(queue.id, 'ana')
        store.update(queue.id, QueueUpdate(closed=True))
        assert read('bo') == read('ana') == 'trace0001'
        with pytest.raises(NotReviewerError):
            read('di')
        with pytest.raises(QueueNotFoundError, match='has no item'):
            store.content(queue.id, 'nope', 'ana')

    def test_a_steps_error_is_cut_to_its_name_where_the_step_verifier_ran(
        self, store: QueueStore
    ) -> None:
        died = 'SandboxDiedError: Container c1 stopped: {"score": 0.2, "passed": false}'
        rejected = "ActionError: Action 'write_file' not allowed here."
        unnamed = 'the step verifier printed 0.2'
        errors = {'s1': {'error': died}, 's2': {'error': rejected}, 's3': {'error': unnamed}}
        verified = _with_steps(graded_trace(), **errors)
        unverified = _without_step_verifier(verified)
        traces = {
            'fingerprint': verified,
            'none': unverified,
            # No fingerprint (an imported trace, say), but a step kept a reward or the step
            # verifier's output: a step verifier ran all the same.
            'reward': _with_steps(unverified, s3={'reward': 0.5}),
            'output': _with_steps(unverified, s3={'step_verifier': {'details': 'ran'}}),
        }
        for key, trace in traces.items():
            record_episode(trace.model_copy(update={'trace_id': key}), db_path=store._path)

        errors_by = {
            key: [step.error for step in self._read(store, LabelTarget(kind='trace', id=key)).steps]
            for key in traces
        }

        cut = ['SandboxDiedError', rejected, None]
        assert errors_by == {
            'fingerprint': cut, 'none': [died, rejected, unnamed], 'reward': cut, 'output': cut,
        }  # fmt: skip
        shown = self._read(store, LabelTarget(kind='trace', id='fingerprint'))
        assert '0.2' not in json.dumps([step.model_dump() for step in shown.steps])

    def test_cli_output_loses_usage_and_cost(self, store: QueueStore) -> None:
        events = [
            {'type': 'assistant', 'message': {'id': 'm1', 'content': [
                {'type': 'text', 'text': 'Writing hello'}],
                'usage': {'input_tokens': 913, 'output_tokens': 41}}},
            {'type': 'result', 'subtype': 'success', 'result': 'done', 'total_cost_usd': 0.0173,
             'usage': {'input_tokens': 1913}, 'modelUsage': {'opus': {'inputTokens': 1913,
                                                                    'costUSD': 0.0173}}},
            {'type': 'turn.completed', 'usage': {'cached_input_tokens': 7, 'output_tokens': 2}},
        ]  # fmt: skip
        stream = '\n'.join([json.dumps(events[0]), 'progress: café {not json', *map(
            json.dumps, events[1:])])  # fmt: skip
        not_cli = '{"usage": "kept: a done action produced this"}'
        trace = _with_steps(
            graded_trace(),
            s1={'action': Action(action_type='cli', content='Write hello'),
                'observation': Observation(content=stream, step=1, metadata={'exit_code': 0})},
            s3={'observation': Observation(content=not_cli, step=2, done=True)},
        )  # fmt: skip
        episode_id = record_episode(trace, db_path=store._path)

        recorded = self._read(store, LabelTarget(kind='trace', id=trace.trace_id))
        episode = self._read(store, LabelTarget(kind='episode', id=episode_id))

        for content, cli, done in ((recorded, 0, 2), (episode, 1, 2)):
            shown = content.steps[cli].observation or ''
            assert shown.split('\n') == [
                json.dumps({'type': 'assistant', 'message': {'id': 'm1', 'content': [
                    {'type': 'text', 'text': 'Writing hello'}]}}),
                'progress: café {not json',
                json.dumps({'type': 'result', 'subtype': 'success', 'result': 'done'}),
                json.dumps({'type': 'turn.completed'}),
            ]  # fmt: skip
            assert content.steps[done].observation == not_cli
            text = json.dumps([step.model_dump() for step in content.steps])
            for spent in ('913', '1913', '0.0173', 'tokens', 'Tokens', 'cost'):
                assert spent not in text

    def test_an_episodes_exit_code_is_its_own_actions(self, store: QueueStore) -> None:
        trace = _with_steps(
            graded_trace(),
            s3={'observation': Observation(content='', step=2, done=True,
                                           metadata={'exit_code': 3})},
        )  # fmt: skip
        episode_id = record_episode(trace, db_path=store._path)

        recorded = self._read(store, LabelTarget(kind='trace', id=trace.trace_id))
        episode = self._read(store, LabelTarget(kind='episode', id=episode_id))

        acted = [step.exit_code for step in episode.steps if step.action is not None]
        assert acted == [step.exit_code for step in recorded.steps if step.applied] == [0, 3]
        assert episode.steps[-1].exit_code is None

    def test_a_trace_stored_twice_is_read_as_the_trace_routes_read_it(
        self, store: QueueStore
    ) -> None:
        trace = graded_trace()
        cut_short = trace.model_copy(
            update={'steps': trace.steps[:1], 'result': None, 'finished_at': None}
        )
        for copy in (trace, cut_short, trace.model_copy(update={'trace_id': 'other'})):
            record_episode(copy, db_path=store._path)
        record_episode(trace.model_copy(update={'trace_id': 'OTHER'}), db_path=store._path)

        merged = self._read(store, LabelTarget(kind='trace', id='trace0001'))

        assert [step.index for step in merged.steps] == [1, 2, 3]  # the copy that got furthest
        record_episode(trace.model_copy(update={'task': 'Another task'}), db_path=store._path)
        for trace_id, match in (('trace0001', 'disagree'), ('other', 'differ only in case')):
            with pytest.raises(ConflictingRecordingError, match=match):
                self._read(store, LabelTarget(kind='trace', id=trace_id))

    def test_a_trace_is_found_through_its_index(self, store: QueueStore) -> None:
        with connect(store._path) as conn:
            plan = conn.execute(
                f'EXPLAIN QUERY PLAN {episode_store._COPIES_OF}', ('trace0001',)
            ).fetchall()

        assert any('USING INDEX episodes_trace_id' in row['detail'] for row in plan)

    def test_a_runs_task_is_read_only_while_its_environment_is_unchanged(self) -> None:
        env = load_environment('coin_change')
        digest = environment_sha256(env)

        assert item_content._run_task('coin_change', digest) == env.task
        for name, sha256 in (('coin_change', '0' * 64), ('coin_change', None), ('nope', digest)):
            assert item_content._run_task(name, sha256) is None

    @pytest.mark.parametrize('name', ['swebench_pro', 'terminal_bench_2'])
    def test_an_adapter_that_picks_its_instance_at_run_time_has_no_task(self, name: str) -> None:
        env = load_environment(name)

        assert env.task  # a generic instruction, not the instance's problem
        assert item_content._run_task(name, environment_sha256(env)) is None

    # Recording the trace warns that its exit_code isn't an int: that odd value is the point.
    @pytest.mark.filterwarnings('ignore:Pydantic serializer warnings')
    def test_an_odd_exit_code_is_left_out(self, store: QueueStore) -> None:
        odd = Observation(content='', step=1, metadata={'exit_code': 'killed'})
        trace = _with_steps(graded_trace(), s1={'observation': odd})
        episode_id = record_episode(trace, db_path=store._path)

        recorded = self._read(store, LabelTarget(kind='trace', id=trace.trace_id))
        episode = self._read(store, LabelTarget(kind='episode', id=episode_id))

        assert recorded.steps[0].exit_code is None
        assert episode.steps[0].exit_code is None
        assert recorded.steps[0].action == episode.steps[0].action

    def test_an_unreadable_copy_counts_only_under_its_exact_id(self, store: QueueStore) -> None:
        record_episode(
            graded_trace().model_copy(update={'trace_id': 'TRACE0001'}), db_path=store._path
        )
        with connect(store._path) as conn:  # valid JSON, so the lookup finds it, but no trace
            conn.execute('UPDATE episodes SET trace = \'{"trace_id": "TRACE0001"}\'')

        with pytest.raises(QueueNotFoundError, match='holds no trace trace0001'):
            self._read(store, LabelTarget(kind='trace', id='trace0001'))
        with pytest.raises(item_content.UnreadableRecordingError):
            self._read(store, LabelTarget(kind='trace', id='TRACE0001'))


def test_an_older_database_gets_the_default_reveal_and_the_trace_index(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            'CREATE TABLE episodes (id TEXT PRIMARY KEY, environment TEXT NOT NULL, '
            'steps INTEGER NOT NULL, total_reward REAL NOT NULL, score REAL, passed INTEGER, '
            'frames TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL)'
        )
        conn.execute(
            'CREATE TABLE queues (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, '
            'instructions TEXT, label_names TEXT NOT NULL, overlap INTEGER NOT NULL, '
            'blind INTEGER NOT NULL, reservation_seconds INTEGER NOT NULL, '
            'reviewers TEXT NOT NULL, closed INTEGER NOT NULL, created_at TEXT NOT NULL)'
        )
        conn.execute(
            "INSERT INTO queues VALUES ('q1', 'calibration', NULL, NULL, '[\"passed\"]', 1, 1, "
            "3600, '[]', 0, '2026-11-19T12:00:00.000000+00:00')"
        )
    conn.close()

    store = QueueStore(db)

    assert store.get('q1').reveal == 'after_submit'
    assert store.update('q1', QueueUpdate(reveal='on_close')).reveal == 'on_close'
    assert store.get('q1').reveal == 'on_close'
    assert store.create(QueueCreate(name='new')).reveal == 'after_submit'
    with connect(db) as conn:
        indexes = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'episodes_trace_id'")
        assert TRACE_KEY in indexes.fetchone()['sql']


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
