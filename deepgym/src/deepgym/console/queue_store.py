"""Persist review queues in the console database (see :mod:`deepgym.review_queues`).

Every write takes the database's write lock first (``BEGIN IMMEDIATE``) and reads the time
under it, so reservations never exceed an item's overlap, across threads and processes
alike. A reservation whose time ran out is marked expired by the next request for an item of
its queue; until then it reads back as expired and counts as free. An item's content is read
from the runs and episodes in the same database.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, TypedDict, cast
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from deepgym.console.db import TRACE_ID, connect, init_db, iso_utc
from deepgym.console.episode_store import select_screenshot
from deepgym.console.item_content import item_content
from deepgym.console.label_store import insert_labels, select_labels, stored_target
from deepgym.console.schemas import LabelQuery
from deepgym.labels import Label, LabelCreate, LabelFields, LabelTarget, Name, is_verdict
from deepgym.review_queues import (
    Assignment,
    ItemContent,
    ItemCreate,
    ItemPage,
    ItemProgress,
    Queue,
    QueueCreate,
    QueueItem,
    QueueProgress,
    QueueUpdate,
    ReviewerProgress,
    ReviewLabel,
    ReviewView,
)

_QUEUE_COLUMNS = (
    'id', 'name', 'description', 'instructions', 'label_names', 'overlap', 'blind',
    'reservation_seconds', 'reviewers', 'closed', 'created_at', 'reveal',
)  # fmt: skip
_INSERT_QUEUE = (
    f'INSERT INTO queues ({", ".join(_QUEUE_COLUMNS)}) '
    f'VALUES ({", ".join(f":{column}" for column in _QUEUE_COLUMNS)})'
)
_UPDATE_QUEUE = (
    'UPDATE queues SET closed = :closed, description = :description, '
    'instructions = :instructions, reservation_seconds = :reservation_seconds, '
    'reviewers = :reviewers, reveal = :reveal WHERE id = :id'
)
_INSERT_ITEM = (
    'INSERT INTO queue_items (id, queue_id, position, target, recording_kind, recording_id, '
    'context, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
)
_INSERT_ASSIGNMENT = (
    'INSERT INTO queue_assignments (id, queue_id, item_id, reviewer, status, reserved_at, '
    'expires_at) VALUES (:id, :queue_id, :item_id, :reviewer, :status, :reserved_at, '
    ':expires_at)'
)
# The first item, in position order and after those the reviewer released, that they haven't
# reviewed or skipped and that has fewer reviews and live reservations than the overlap. Run
# after expiring the queue's timed-out reservations, so every 'reserved' row is live.
# ponytail: scans every item of the queue on each call; keep an open-slot count per item if
# queues grow past ~10^5 items.
_NEXT = """
SELECT * FROM queue_items AS i
WHERE i.queue_id = :queue_id
  AND NOT EXISTS (
    SELECT 1 FROM queue_assignments AS mine
    WHERE mine.item_id = i.id AND mine.reviewer = :reviewer
      AND mine.status IN ('submitted', 'skipped'))
  AND (SELECT COUNT(*) FROM queue_assignments AS taken
       WHERE taken.item_id = i.id AND taken.status IN ('reserved', 'submitted')) < :overlap
ORDER BY EXISTS (
    SELECT 1 FROM queue_assignments AS gave
    WHERE gave.item_id = i.id AND gave.reviewer = :reviewer AND gave.status = 'released'),
  i.position
LIMIT 1
"""
# Reviews and live reservations of one item at :now.
_TAKEN = """
SELECT COUNT(*) FROM queue_assignments
WHERE item_id = :item_id AND (status = 'submitted' OR (status = 'reserved' AND expires_at > :now))
"""
# Whether the reviewer may still review another item on this item's recording in a blind
# queue: one in a queue that is open (or that they hold a live reservation of), that lets
# them review, that hasn't every review it needs, and that they haven't reviewed or skipped.
_PENDING = """
SELECT 1 FROM queue_items AS this
JOIN queue_items AS i
  ON i.recording_kind = this.recording_kind AND i.recording_id = this.recording_id
  AND i.id != this.id
JOIN queues AS q ON q.id = i.queue_id
WHERE this.id = :item_id AND q.blind = 1
  AND (q.closed = 0 OR EXISTS (
    SELECT 1 FROM queue_assignments AS held
    WHERE held.item_id = i.id AND held.reviewer = :reviewer AND held.status = 'reserved'
      AND held.expires_at > :now))
  AND (q.reviewers = '[]'
       OR EXISTS (SELECT 1 FROM json_each(q.reviewers) WHERE value = :reviewer))
  AND NOT EXISTS (
    SELECT 1 FROM queue_assignments AS a
    WHERE a.item_id = i.id AND a.reviewer = :reviewer AND a.status IN ('submitted', 'skipped'))
  AND (SELECT COUNT(*) FROM queue_assignments AS done
       WHERE done.item_id = i.id AND done.status = 'submitted') < q.overlap
LIMIT 1
"""
# Whether the reviewer may have seen others' labels on this item's recording: a queue showed
# them some, or they had a turn at an item on it in a queue that isn't blind.
_SHOWN = """
SELECT 1 FROM queue_items AS this
WHERE this.id = :item_id AND (
  EXISTS (
    SELECT 1 FROM queue_exposures AS e
    WHERE e.reviewer = :reviewer AND e.recording_kind = this.recording_kind
      AND e.recording_id = this.recording_id)
  OR EXISTS (
    SELECT 1 FROM queue_assignments AS a
    JOIN queue_items AS i ON i.id = a.item_id
    JOIN queues AS q ON q.id = a.queue_id
    WHERE a.reviewer = :reviewer AND q.blind = 0
      AND i.recording_kind = this.recording_kind AND i.recording_id = this.recording_id))
"""
# Record the reviewer's first exposure to the item's recording; a later one writes nothing.
_EXPOSE = """
INSERT INTO queue_exposures (reviewer, recording_kind, recording_id, shown_at)
SELECT :reviewer, i.recording_kind, i.recording_id, :shown_at FROM queue_items AS i
WHERE i.id = :item_id AND NOT EXISTS (
  SELECT 1 FROM queue_exposures AS e
  WHERE e.reviewer = :reviewer AND e.recording_kind = i.recording_kind
    AND e.recording_id = i.recording_id)
"""
# Each item of a queue with its submitted reviews, skips and live reservations at :now.
_ITEM_COUNTS = """
SELECT i.*,
  COALESCE(SUM(a.status = 'submitted'), 0) AS reviews,
  COALESCE(SUM(a.status = 'skipped'), 0) AS skipped,
  COALESCE(SUM(a.status = 'reserved' AND a.expires_at > :now), 0) AS reserved
FROM queue_items AS i LEFT JOIN queue_assignments AS a ON a.item_id = i.id
WHERE i.queue_id = :queue_id GROUP BY i.id
"""
# Only listed reviewers take turns in a queue that lists them, and each skips an item at most
# once, so :listed - skipped is how many reviewers could still review the item.
_TOTALS = f"""
SELECT COUNT(*) AS items,
  COALESCE(SUM(reviews >= :overlap), 0) AS complete,
  COALESCE(SUM(reviews < :overlap AND (reviews > 0 OR reserved > 0)), 0) AS in_progress,
  COALESCE(SUM(reviews), 0) AS reviews,
  COALESCE(SUM(reserved), 0) AS reserved,
  COALESCE(SUM(reviews < :overlap AND :listed > 0 AND :listed - skipped < :overlap), 0)
    AS blocked
FROM ({_ITEM_COUNTS})
"""
_REVIEWER_COUNTS = """
SELECT reviewer,
  SUM(status = 'submitted') AS submitted,
  SUM(status = 'reserved' AND expires_at > :now) AS reserved,
  SUM(status = 'skipped') AS skipped,
  SUM(status = 'released') AS released,
  SUM(status = 'expired' OR (status = 'reserved' AND expires_at <= :now)) AS expired
FROM queue_assignments WHERE queue_id = :queue_id GROUP BY reviewer ORDER BY reviewer
"""
_REVIEWER = TypeAdapter(Name)
_RETRY_FIELDS = set(LabelFields.model_fields) - {'id', 'blind'}

Scope = Literal['none', 'own', 'all']
"""Which labels on an item's kind and id a reviewer may see (see ``ReviewView.labels``)."""


class ErrorDetail(TypedDict):
    """One problem with an input: where it is, and what is wrong."""

    loc: tuple[str | int, ...]
    """Path from the argument it is in, e.g. ``('labels', 0, 'target')``."""

    msg: str
    type: str


class QueueNotFoundError(LookupError):
    """An unknown queue, an item its queue doesn't hold, or a run, episode or trace the
    console doesn't hold for an item's content."""


class NotReviewerError(PermissionError):
    """A reviewer the queue doesn't list."""


class QueueInputError(ValueError):
    """Input that the queue's settings or the label rules refuse; ``errors`` lists each problem."""

    def __init__(self, errors: list[ErrorDetail]) -> None:
        super().__init__(
            '; '.join(f'{".".join(map(str, error["loc"]))}: {error["msg"]}' for error in errors)
        )
        self.errors = errors


class QueueStore:
    """Review queues, their items and reviewers' reservations, in the console database.

    An unknown queue or item raises :class:`QueueNotFoundError`, a reviewer the queue
    doesn't list :class:`NotReviewerError`, input its settings or the label rules refuse
    :class:`QueueInputError`, and any other refusal ``ValueError``.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        init_db(path)

    def create(self, queue: QueueCreate) -> Queue:
        """Store a new, open queue.

        Raises:
            ValueError: If its id is stored with other settings. The same id and settings
                return the stored queue.
        """
        with connect(self._path) as conn:
            now = _lock(conn)
            if queue.id is not None:
                row = conn.execute('SELECT * FROM queues WHERE id = ?', (queue.id,)).fetchone()
                if row is not None:
                    stored = _queue(row)
                    if _settings(stored) != _settings(queue):
                        raise ValueError(
                            f'Queue {queue.id} is stored with other settings; use a new id'
                        )
                    return stored
            stored = Queue.model_validate(
                {**queue.model_dump(), 'id': queue.id or uuid4().hex, 'created_at': now}
            )
            conn.execute(_INSERT_QUEUE, _queue_row(stored))
        return stored

    def get(self, queue_id: str) -> Queue:
        """Return one queue."""
        with connect(self._path) as conn:
            return _get_queue(conn, queue_id)

    def queues(self) -> list[Queue]:
        """Return every queue, oldest first."""
        with connect(self._path) as conn:
            rows = conn.execute('SELECT * FROM queues ORDER BY created_at, rowid').fetchall()
        return [_queue(row) for row in rows]

    def update(self, queue_id: str, update: QueueUpdate) -> Queue:
        """Apply *update*'s given fields to a queue, and return it.

        Raises:
            ValueError: If the new reviewers leave out a current one, or the queue lets
                anyone review; or *update* changes ``reveal`` once a blind queue has a
                submitted review, or one that isn't blind has handed out an item.
        """
        changes = update.model_dump(exclude_unset=True)
        with connect(self._path) as conn:
            _lock(conn)
            queue = _get_queue(conn, queue_id)
            if 'reviewers' in changes:
                _check_added(queue, changes['reviewers'])
            if changes.get('reveal', queue.reveal) != queue.reveal:
                _check_reveal_unused(conn, queue)
            updated = queue.model_copy(update=changes)
            conn.execute(_UPDATE_QUEUE, _queue_row(updated))
        return updated

    def add_items(self, queue_id: str, items: Sequence[ItemCreate]) -> list[QueueItem]:
        """Add *items* after the queue's last, in this order, in one transaction.

        An item whose target the queue already holds, with the same context, is a retry:
        the stored item is returned and nothing new is added for it.

        Returns:
            The stored items, in the order of *items*.

        An episode recorded from a trace is on the same recording as that trace: blind
        review treats the two as one (see :class:`deepgym.review_queues.ReviewView`).

        Raises:
            QueueInputError: If the queue asks for a verdict and an item is a step or a
                text, which can't get one. Then none of *items* is added.
            ValueError: If the queue holds an item's target with another context; then
                none of *items* is added.
        """
        with connect(self._path) as conn:
            now = _lock(conn)
            queue = _get_queue(conn, queue_id)
            _check_whole(queue, items)
            position = conn.execute(
                'SELECT COALESCE(MAX(position), 0) FROM queue_items WHERE queue_id = ?',
                (queue_id,),
            ).fetchone()[0]
            stored: list[QueueItem] = []
            for item in items:
                key = item.target.model_dump_json()
                row = conn.execute(
                    'SELECT * FROM queue_items WHERE queue_id = ? AND target = ?', (queue_id, key)
                ).fetchone()
                if row is not None:
                    held = _item(row)
                    if held.context != item.context:
                        raise ValueError(
                            f'Queue {queue_id} holds {key} as item {held.id} with another context'
                        )
                    stored.append(held)
                    continue
                position += 1
                added = QueueItem(
                    target=item.target, context=item.context, id=uuid4().hex, queue_id=queue_id,
                    position=position, created_at=now,
                )  # fmt: skip
                conn.execute(
                    _INSERT_ITEM,
                    (added.id, queue_id, position, key, *_recording(conn, item.target),
                     added.context, iso_utc(now)),
                )  # fmt: skip
                stored.append(added)
        return stored

    def items(self, queue_id: str, offset: int = 0, limit: int | None = None) -> ItemPage:
        """Return one page of a queue's items in position order, with their review counts.

        Args:
            queue_id: The queue.
            offset: How many of the first items to skip.
            limit: Most items to return; None returns every one after *offset*.
        """
        with connect(self._path) as conn:
            now = _snapshot(conn)
            queue = _get_queue(conn, queue_id)
            total = conn.execute(
                'SELECT COUNT(*) FROM queue_items WHERE queue_id = ?', (queue_id,)
            ).fetchone()[0]
            rows = conn.execute(
                f'{_ITEM_COUNTS} ORDER BY i.position LIMIT :limit OFFSET :offset',
                {'queue_id': queue_id, 'now': iso_utc(now), 'offset': offset,
                 'limit': -1 if limit is None else limit},  # SQLite: LIMIT -1 is none
            ).fetchall()  # fmt: skip
        items = [
            ItemProgress(
                item=_item(row), reviews=row['reviews'], needed=queue.overlap,
                reserved=row['reserved'],
            )
            for row in rows
        ]  # fmt: skip
        return ItemPage(items=items, total=total)

    def progress(self, queue_id: str) -> QueueProgress:
        """Return how far a queue's reviews are, overall and per reviewer."""
        with connect(self._path) as conn:
            now = _snapshot(conn)
            queue = _get_queue(conn, queue_id)
            params = {'queue_id': queue_id, 'now': iso_utc(now)}
            totals = conn.execute(
                _TOTALS, {**params, 'overlap': queue.overlap, 'listed': len(queue.reviewers)}
            ).fetchone()
            reviewers = conn.execute(_REVIEWER_COUNTS, params).fetchall()
        return QueueProgress(
            queue_id=queue_id,
            **dict(totals),
            reviews_needed=totals['items'] * queue.overlap,
            reviewers=[ReviewerProgress(**dict(row)) for row in reviewers],
        )

    def next_item(self, queue_id: str, reviewer: str) -> ReviewView | None:
        """Reserve the next item for *reviewer*, or extend the reservation they hold.

        A reviewer holds one reservation per queue at a time; asking again returns it, its
        time renewed (never shortened, if the reservation time was lowered). The next item
        is the first, in position order and after the items *reviewer* released, that they
        haven't reviewed or skipped and whose reviews and live reservations by others are
        fewer than the overlap.

        Returns:
            The item as *reviewer* may see it, or None if no item is left for them now.

        Raises:
            ValueError: If the queue is closed.
            NotReviewerError: If the queue lists its reviewers and *reviewer* isn't one.
        """
        _REVIEWER.validate_python(reviewer)
        with connect(self._path) as conn:
            now = _lock(conn)
            queue = _get_queue(conn, queue_id)
            _check_reviewer(queue, reviewer)
            if queue.closed:
                raise ValueError(f'Queue {queue_id} is closed')
            conn.execute(
                "UPDATE queue_assignments SET status = 'expired', finished_at = expires_at "
                "WHERE queue_id = ? AND status = 'reserved' AND expires_at <= ?",
                (queue_id, iso_utc(now)),
            )
            expires_at = now + timedelta(seconds=queue.reservation_seconds)
            held = conn.execute(
                'SELECT * FROM queue_assignments WHERE queue_id = ? AND reviewer = ? '
                "AND status = 'reserved'",
                (queue_id, reviewer),
            ).fetchone()
            if held is not None:
                assignment = _assignment(held, now)
                expires_at = max(assignment.expires_at, expires_at)
                assignment = assignment.model_copy(update={'expires_at': expires_at})
                conn.execute(
                    'UPDATE queue_assignments SET expires_at = ? WHERE id = ?',
                    (iso_utc(expires_at), assignment.id),
                )
                item = _get_item(conn, queue_id, assignment.item_id)
            else:
                row = conn.execute(
                    _NEXT, {'queue_id': queue_id, 'reviewer': reviewer, 'overlap': queue.overlap}
                ).fetchone()
                if row is None:
                    return None
                item = _item(row)
                assignment = Assignment(
                    id=uuid4().hex, queue_id=queue_id, item_id=item.id, reviewer=reviewer,
                    status='reserved', reserved_at=now, expires_at=expires_at,
                )  # fmt: skip
                conn.execute(_INSERT_ASSIGNMENT, _assignment_row(assignment))
            view = _view(conn, queue, item, assignment, now)
        return view

    def view(self, queue_id: str, item_id: str, reviewer: str) -> ReviewView:
        """Return an item as *reviewer* may see it, with their latest turn at it.

        Raises:
            NotReviewerError: If the queue lists its reviewers and *reviewer* isn't one.
        """
        with connect(self._path) as conn:
            now = _lock(conn)  # the view may record an exposure
            queue = _get_queue(conn, queue_id)
            _check_reviewer(queue, reviewer)
            item = _get_item(conn, queue_id, item_id)
            assignment = _latest(conn, item_id, reviewer, now)
            view = _view(conn, queue, item, assignment, now)
        return view

    def content(self, queue_id: str, item_id: str, reviewer: str) -> ItemContent:
        """Return what *reviewer* needs to grade an item, and nothing that reveals a grade
        (see :class:`deepgym.review_queues.ItemContent`).

        A reviewer may read it while they could submit a review of the item (see
        :meth:`submit`), and once they have.

        Raises:
            NotReviewerError: If the queue lists its reviewers and *reviewer* isn't one.
            ValueError: If *reviewer* hasn't reviewed the item and couldn't submit a review
                of it now.
            QueueNotFoundError: Also if the console holds no run, episode or trace for the
                item's target.
            ConflictingRecordingError: A ``ValueError``, if a trace's stored copies disagree
                or another trace's id matches but for case.
            UnreadableRecordingError: If the stored recording, or a file its content needs,
                can't be read.
        """
        with connect(self._path) as conn:
            item = _readable(conn, queue_id, item_id, reviewer)
            content = item_content(conn, item)
        if content is None:
            raise QueueNotFoundError(f'The console holds no {item.target.kind} {item.target.id}')
        return content

    def screenshot(self, queue_id: str, item_id: str, reviewer: str, frame: int) -> bytes:
        """Return the PNG shown at *frame* of an episode item, to the reviewers who may read
        its content (see :meth:`content`).

        Raises:
            NotReviewerError: If the queue lists its reviewers and *reviewer* isn't one.
            ValueError: If *reviewer* may not read the item's content.
            QueueNotFoundError: Also if the item isn't an episode, or that frame of it has no
                screenshot.
        """
        with connect(self._path) as conn:
            target = _readable(conn, queue_id, item_id, reviewer).target
            png = None if target.kind != 'episode' else select_screenshot(conn, target.id, frame)
        if png is None:
            raise QueueNotFoundError(
                f'The console holds no screenshot of {target.kind} {target.id} at frame {frame}'
            )
        return png

    def submit(
        self, queue_id: str, item_id: str, reviewer: str, labels: Sequence[ReviewLabel]
    ) -> ReviewView:
        """Store *reviewer*'s review of an item, and end their turn at it.

        The reviewer must hold a live reservation of the item, or have one that expired or
        that they released while the item still has a free slot (a closed queue takes only
        live ones). The labels and the submitted turn are stored in one transaction: both,
        or neither if anything is refused. Submitting the same labels again (the same
        bodies, in order, and the same ids where given) returns the review as stored.

        Returns:
            The item as *reviewer* sees it now; the review's labels are those whose
            ``group_id`` is the assignment's id.

        Raises:
            NotReviewerError: If the queue lists its reviewers and *reviewer* isn't one.
            QueueInputError: If the review gives no label, a label outside the item's target or
                one breaking a label rule (``LabelCreate``), or misses one of the queue's
                ``label_names`` on the item's target.
            ValueError: If *reviewer* holds no usable turn at the item (none, skipped,
                already reviewed with other labels, or expired or released with no free
                slot), the queue is closed and their turn isn't live, or the label store
                refuses a label (``LabelStore.add_many``).
        """
        with connect(self._path) as conn:
            now = _lock(conn)
            queue = _get_queue(conn, queue_id)
            _check_reviewer(queue, reviewer)
            item = _get_item(conn, queue_id, item_id)
            latest = _latest(conn, item_id, reviewer, now)
            if latest is not None and latest.status == 'submitted':
                records = _review_labels(queue, item, latest, labels, blind=queue.blind)
                stored = select_labels(conn, LabelQuery(group_id=latest.id))
                if not _is_retry(stored, records):
                    raise ValueError(f'{reviewer} already reviewed item {item_id}')
                assignment = latest
            else:
                if latest is None:
                    raise ValueError(
                        f'{reviewer} holds no reservation of item {item_id}; ask for the next item'
                    )
                assignment = _submittable(conn, queue, item, latest, now)
                insert_labels(
                    conn,
                    _review_labels(
                        queue, item, assignment, labels, blind=_blind(conn, queue, item, assignment)
                    ),
                )
                assignment = _finish(conn, assignment, 'submitted', now)
            view = _view(conn, queue, item, assignment, now)
        return view

    def release(self, queue_id: str, item_id: str, reviewer: str) -> Assignment:
        """Hand back a held item unreviewed; it may be handed to *reviewer* again later.

        Raises:
            NotReviewerError: If the queue lists its reviewers and *reviewer* isn't one.
            ValueError: If *reviewer* holds no live reservation of the item.
        """
        return self._end(queue_id, item_id, reviewer, 'released')

    def skip(self, queue_id: str, item_id: str, reviewer: str) -> Assignment:
        """Hand back a held item unreviewed, never to be handed to *reviewer* again.

        Raises:
            NotReviewerError: If the queue lists its reviewers and *reviewer* isn't one.
            ValueError: If *reviewer* holds no live reservation of the item.
        """
        return self._end(queue_id, item_id, reviewer, 'skipped')

    def _end(
        self, queue_id: str, item_id: str, reviewer: str, status: Literal['released', 'skipped']
    ) -> Assignment:
        """End *reviewer*'s live reservation of an item with *status*."""
        with connect(self._path) as conn:
            now = _lock(conn)
            queue = _get_queue(conn, queue_id)
            _check_reviewer(queue, reviewer)
            item = _get_item(conn, queue_id, item_id)
            return _finish(conn, _held(conn, item, reviewer, now), status, now)


def _lock(conn: sqlite3.Connection) -> datetime:
    """Take the write lock, and return the time read under it."""
    conn.execute('BEGIN IMMEDIATE')
    return datetime.now(timezone.utc)


def _snapshot(conn: sqlite3.Connection) -> datetime:
    """Start a read transaction, so every read in it sees one state, and return the time."""
    conn.execute('BEGIN')
    return datetime.now(timezone.utc)


def _get_queue(conn: sqlite3.Connection, queue_id: str) -> Queue:
    """Return a stored queue, or raise :class:`QueueNotFoundError`."""
    row = conn.execute('SELECT * FROM queues WHERE id = ?', (queue_id,)).fetchone()
    if row is None:
        raise QueueNotFoundError(f'Queue {queue_id} not found')
    return _queue(row)


def _get_item(conn: sqlite3.Connection, queue_id: str, item_id: str) -> QueueItem:
    """Return a stored item of a queue, or raise :class:`QueueNotFoundError`."""
    row = conn.execute(
        'SELECT * FROM queue_items WHERE id = ? AND queue_id = ?', (item_id, queue_id)
    ).fetchone()
    if row is None:
        raise QueueNotFoundError(f'Queue {queue_id} has no item {item_id}')
    return _item(row)


def _readable(conn: sqlite3.Connection, queue_id: str, item_id: str, reviewer: str) -> QueueItem:
    """Start a read transaction, and return the item if *reviewer* may read its content:
    the queue lets them review, and they have reviewed it or could submit a review of it now
    (``_submittable``), so the two rules never drift apart.

    Raises:
        NotReviewerError: If the queue lists its reviewers and *reviewer* isn't one.
        ValueError: If they never had a turn at it, or ``_submittable`` refuses theirs.
    """
    now = _snapshot(conn)
    queue = _get_queue(conn, queue_id)
    _check_reviewer(queue, reviewer)
    item = _get_item(conn, queue_id, item_id)
    latest = _latest(conn, item_id, reviewer, now)
    if latest is None:
        raise ValueError(
            f'{reviewer} holds no reservation of item {item_id}; ask for the next item'
        )
    if latest.status != 'submitted':
        _submittable(conn, queue, item, latest, now)
    return item


def _check_reviewer(queue: Queue, reviewer: str) -> None:
    """Refuse a reviewer the queue doesn't list, if it lists any."""
    if queue.reviewers and reviewer not in queue.reviewers:
        raise NotReviewerError(f'{reviewer} is not a reviewer of queue {queue.id}')


def _check_added(queue: Queue, reviewers: Sequence[str]) -> None:
    """Refuse a new reviewer list that drops a current reviewer or limits an open queue."""
    if not queue.reviewers:
        raise ValueError(f'Queue {queue.id} lets anyone review; it keeps doing so')
    dropped = [reviewer for reviewer in queue.reviewers if reviewer not in reviewers]
    if dropped:
        raise ValueError(f'Reviewers can only be added; the new list drops {dropped}')


def _check_reveal_unused(conn: sqlite3.Connection, queue: Queue) -> None:
    """Refuse a change of reveal policy once it has decided what a reviewer saw: in a blind
    queue once a review is submitted, and in one that isn't once an item was handed out,
    since a reviewer there sees labels from their first turn."""
    used = conn.execute(
        'SELECT 1 FROM queue_assignments WHERE queue_id = :queue_id '
        "AND (:any_turn OR status = 'submitted') LIMIT 1",
        {'queue_id': queue.id, 'any_turn': not queue.blind},
    ).fetchone()
    if used is not None:
        done = 'submitted reviews' if queue.blind else 'handed out items'
        raise ValueError(
            f'Queue {queue.id} has {done} under reveal {queue.reveal!r}; '
            'its reveal policy no longer changes'
        )


def _check_whole(queue: Queue, items: Sequence[ItemCreate]) -> None:
    """Refuse items that are a step or a text when the queue asks for a verdict."""
    verdicts = [name for name in queue.label_names if is_verdict(name)]
    errors: list[ErrorDetail] = [
        _error(
            (index, 'target'),
            f'queue {queue.id} asks for verdicts {verdicts}, which are on whole targets: '
            'an item that is a step or a text cannot get one',
        )
        for index, item in enumerate(items)
        if verdicts and (item.target.step is not None or item.target.field is not None)
    ]
    if errors:
        raise QueueInputError(errors)


def _latest(
    conn: sqlite3.Connection, item_id: str, reviewer: str, now: datetime
) -> Assignment | None:
    """Return *reviewer*'s latest turn at an item, or None if they never had one."""
    row = conn.execute(
        'SELECT * FROM queue_assignments WHERE item_id = ? AND reviewer = ? '
        'ORDER BY rowid DESC LIMIT 1',
        (item_id, reviewer),
    ).fetchone()
    return None if row is None else _assignment(row, now)


def _held(conn: sqlite3.Connection, item: QueueItem, reviewer: str, now: datetime) -> Assignment:
    """Return *reviewer*'s live reservation of *item*.

    Raises:
        ValueError: If it expired, they already reviewed the item, or they hold none.
    """
    assignment = _latest(conn, item.id, reviewer, now)
    if assignment is not None:
        if assignment.status == 'reserved':
            return assignment
        if assignment.status == 'expired':
            raise ValueError(
                f"{reviewer}'s reservation of item {item.id} expired at "
                f'{iso_utc(assignment.expires_at)}'
            )
        if assignment.status == 'submitted':
            raise ValueError(f'{reviewer} already reviewed item {item.id}')
    raise ValueError(f'{reviewer} holds no reservation of item {item.id}; ask for the next item')


def _submittable(
    conn: sqlite3.Connection, queue: Queue, item: QueueItem, latest: Assignment, now: datetime
) -> Assignment:
    """Return the turn *latest*, a review not yet submitted, if a review may be submitted on it.

    Raises:
        ValueError: If it was skipped, or it isn't live and either the queue is closed or
            the item has every review and reservation it needs.
    """
    if latest.status == 'reserved':
        return latest
    if latest.status == 'skipped':
        raise ValueError(f'{latest.reviewer} skipped item {item.id}')
    ended = f"{latest.reviewer}'s reservation of item {item.id} {latest.status}"
    if queue.closed:
        raise ValueError(f'Queue {queue.id} is closed, and {ended}')
    taken = conn.execute(_TAKEN, {'item_id': item.id, 'now': iso_utc(now)}).fetchone()[0]
    if taken >= queue.overlap:
        # Not reserved (that returned above), so the turn has ended and has a finished_at.
        finished_at = cast(datetime, latest.finished_at)
        raise ValueError(
            f'{ended} at {iso_utc(finished_at)}, and the item has every review and '
            'reservation it needs; ask for the next item'
        )
    return latest


def _view(
    conn: sqlite3.Connection,
    queue: Queue,
    item: QueueItem,
    assignment: Assignment | None,
    now: datetime,
) -> ReviewView:
    """Return *item* with the labels its reviewer may see, read in *conn*'s transaction
    (which holds the write lock), and record in it their first exposure to the recording
    when the labels include someone else's."""
    scope = _scope(conn, queue, item, assignment, now)
    labels: list[Label] = []
    if scope != 'none':
        query = LabelQuery(target_kind=item.target.kind, target_id=item.target.id, current=True)
        labels = select_labels(conn, query)
    if scope == 'own' and assignment is not None:
        labels = [label for label in labels if label.group_id == assignment.id]
    if queue.reveal == 'humans_only':
        labels = [label for label in labels if label.source == 'human']
    if assignment is not None and any(not _own(label, assignment.reviewer) for label in labels):
        conn.execute(
            _EXPOSE,
            {'reviewer': assignment.reviewer, 'item_id': item.id, 'shown_at': iso_utc(now)},
        )
    return ReviewView(item=item, assignment=assignment, labels=labels)


def _scope(
    conn: sqlite3.Connection,
    queue: Queue,
    item: QueueItem,
    assignment: Assignment | None,
    now: datetime,
) -> Scope:
    """Return which labels on *item*'s kind and id the reviewer of *assignment* may see:
    every one in a queue that isn't blind; in a blind one, none before they submit, then
    only their review's own while they may still review another item on the recording in a
    blind queue (see ``_PENDING``), and every one after that. Under ``reveal='on_close'``,
    never more than their review's own while the queue is open."""
    if assignment is None:
        return 'none'
    if queue.blind and assignment.status != 'submitted':
        return 'none'
    if queue.reveal == 'on_close' and not queue.closed:
        return 'own'
    if not queue.blind:
        return 'all'
    pending = conn.execute(
        _PENDING, {'item_id': item.id, 'reviewer': assignment.reviewer, 'now': iso_utc(now)}
    ).fetchone()
    return 'own' if pending else 'all'


def _blind(conn: sqlite3.Connection, queue: Queue, item: QueueItem, assignment: Assignment) -> bool:
    """Return whether the reviewer of *assignment* reviews *item* blind: the queue is blind,
    no queue response showed them someone else's label on its recording, and they had no
    turn at an item on it in a queue that isn't blind."""
    shown = conn.execute(_SHOWN, {'item_id': item.id, 'reviewer': assignment.reviewer})
    return queue.blind and shown.fetchone() is None


def _own(label: Label, reviewer: str) -> bool:
    """Return whether *label* is *reviewer*'s own."""
    return label.source == 'human' and label.author == reviewer


def _recording(conn: sqlite3.Connection, target: LabelTarget) -> tuple[str, str]:
    """Return the kind and id of the recording *target* is on: for a console episode
    recorded from a trace, that trace; for anything else, its own."""
    if target.kind == 'episode':
        row = conn.execute(f'SELECT {TRACE_ID} FROM episodes WHERE id = ?', (target.id,)).fetchone()
        if row is not None and row[0] is not None:
            return 'trace', row[0]
    return target.kind, target.id


def _review_labels(
    queue: Queue,
    item: QueueItem,
    assignment: Assignment,
    labels: Sequence[ReviewLabel],
    *,
    blind: bool,
) -> list[LabelCreate]:
    """Return the label records of a review.

    Raises:
        QueueInputError: If there are none, one repeats another's id, is outside the item's
            target or breaks a label rule, or one of the queue's ``label_names`` is missing
            on the item's target.
    """
    if not labels:
        raise QueueInputError([_error(('labels',), 'A review gives at least one label')])
    records: list[LabelCreate] = []
    errors: list[ErrorDetail] = []
    ids: set[str] = set()
    for index, label in enumerate(labels):
        if label.id is not None:
            if label.id in ids:
                errors.append(_error(('labels', index, 'id'), "repeats an earlier label's id"))
            ids.add(label.id)
        target = label.target or item.target
        if not _within(target, item.target):
            scope = item.target.model_dump(exclude_none=True)
            errors.append(
                _error(
                    ('labels', index, 'target'), f"is not the item's target {scope} or inside it"
                )
            )
            continue
        fields = {
            **label.model_dump(exclude={'target'}), 'target': target, 'source': 'human',
            'author': assignment.reviewer, 'blind': blind, 'group_id': assignment.id,
        }  # fmt: skip
        try:
            records.append(LabelCreate.model_validate(fields))
        except ValidationError as exc:
            errors += [
                _error(('labels', index, *error['loc']), error['msg'], error['type'])
                for error in exc.errors()
            ]
    if errors:
        raise QueueInputError(errors)
    given = {record.name for record in records if record.target == item.target}
    missing = [name for name in queue.label_names if name not in given]
    if missing:
        raise QueueInputError(
            [_error(('labels',), f"gives no {missing} on the item's target; queue {queue.id} "
                                 f'asks for {list(queue.label_names)}')]
        )  # fmt: skip
    return records


def _within(target: LabelTarget, scope: LabelTarget) -> bool:
    """Return whether *target* is *scope* or inside it: the same kind and id, the same step
    and field where *scope* has one, and a span inside *scope*'s where it has one."""
    span = scope.span is None or (
        target.span is not None
        and scope.span[0] <= target.span[0] <= target.span[1] <= scope.span[1]
    )
    return (
        (target.kind, target.id) == (scope.kind, scope.id)
        and scope.step in (None, target.step)
        and scope.field in (None, target.field)
        and span
    )


def _is_retry(stored: Sequence[Label], records: Sequence[LabelCreate]) -> bool:
    """Return whether *records* repeat a review's *stored* labels: the same bodies in order,
    and the same ids where *records* give one. ``blind`` may differ; it was set when stored."""
    return len(stored) == len(records) and all(
        label.model_dump(include=_RETRY_FIELDS) == record.model_dump(include=_RETRY_FIELDS)
        and record.id in (None, label.id)
        for label, record in zip(stored, records)
    )


def _error(loc: tuple[str | int, ...], msg: str, kind: str = 'value_error') -> ErrorDetail:
    """Return one :class:`QueueInputError` problem."""
    return {'loc': loc, 'msg': msg, 'type': kind}


def _finish(
    conn: sqlite3.Connection, assignment: Assignment, status: str, now: datetime
) -> Assignment:
    """End a turn with *status* at *now*."""
    conn.execute(
        'UPDATE queue_assignments SET status = ?, finished_at = ? WHERE id = ?',
        (status, iso_utc(now), assignment.id),
    )
    return assignment.model_copy(update={'status': status, 'finished_at': now})


def _settings(queue: QueueCreate) -> dict[str, object]:
    """Return a queue's settings, without its id."""
    return queue.model_dump(include=set(QueueCreate.model_fields) - {'id'})


def _queue_row(queue: Queue) -> dict[str, object]:
    """Return the column values that store *queue*."""
    return {
        **queue.model_dump(),
        'label_names': json.dumps(queue.label_names),
        'reviewers': json.dumps(queue.reviewers),
        'created_at': iso_utc(queue.created_at),
    }


def _queue(row: sqlite3.Row) -> Queue:
    """Rebuild a stored queue without checking today's rules."""
    return Queue.model_construct(
        **{
            **dict(row),
            'label_names': tuple(json.loads(row['label_names'])),
            'reviewers': tuple(json.loads(row['reviewers'])),
            'blind': bool(row['blind']),
            'closed': bool(row['closed']),
            'created_at': datetime.fromisoformat(row['created_at']),
        }
    )


def _item(row: sqlite3.Row) -> QueueItem:
    """Rebuild a stored item without checking today's rules."""
    return QueueItem.model_construct(
        id=row['id'],
        queue_id=row['queue_id'],
        position=row['position'],
        target=stored_target(json.loads(row['target'])),
        context=row['context'],
        created_at=datetime.fromisoformat(row['created_at']),
    )


def _assignment_row(assignment: Assignment) -> dict[str, object]:
    """Return the column values that store a new reservation."""
    return {
        **assignment.model_dump(exclude={'finished_at'}),
        'reserved_at': iso_utc(assignment.reserved_at),
        'expires_at': iso_utc(assignment.expires_at),
    }


def _assignment(row: sqlite3.Row, now: datetime) -> Assignment:
    """Rebuild a stored turn as of *now*: a reservation past its time reads as expired."""
    expires_at = datetime.fromisoformat(row['expires_at'])
    status, finished_at = row['status'], row['finished_at']
    if status == 'reserved' and expires_at <= now:
        status, finished_at = 'expired', row['expires_at']
    return Assignment(
        id=row['id'],
        queue_id=row['queue_id'],
        item_id=row['item_id'],
        reviewer=row['reviewer'],
        status=status,
        reserved_at=datetime.fromisoformat(row['reserved_at']),
        expires_at=expires_at,
        finished_at=None if finished_at is None else datetime.fromisoformat(finished_at),
    )
