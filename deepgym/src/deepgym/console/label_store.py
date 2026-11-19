"""Persist labels in the console database, append-only (see :mod:`deepgym.labels`)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import AwareDatetime, TypeAdapter

from deepgym.console.db import connect, connect_read_only, init_db, iso_utc, table_columns
from deepgym.console.schemas import LabelPage, LabelQuery
from deepgym.labels import (
    Label,
    LabelCreate,
    LabelFields,
    LabelTarget,
    TargetKind,
    is_verdict,
)
from deepgym.labels import excluded as excluded_targets

_COLUMNS = (
    'seq', 'id', 'type', 'target_kind', 'target_id', 'step', 'name', 'source', 'author',
    'blind', 'rubric_version', 'rubric_sha256', 'group_id', 'supersedes', 'created_at',
    'record',
)  # fmt: skip
_INSERT = (
    f'INSERT INTO labels ({", ".join(_COLUMNS)}) '
    f'VALUES ({", ".join(f":{column}" for column in _COLUMNS)})'
)
# LabelQuery fields matched by equality; each is also the column's name.
_EQUAL_FILTERS = (
    'type', 'target_kind', 'target_id', 'step', 'name', 'source', 'author', 'blind',
    'rubric_version', 'rubric_sha256', 'group_id',
)  # fmt: skip
# Records in force (deepgym.labels.current_labels): no retraction, and nothing supersedes them.
_CURRENT = (
    "type != 'retract' AND "
    'NOT EXISTS (SELECT 1 FROM labels AS later WHERE later.supersedes = labels.id)'
)
_TIME = TypeAdapter(AwareDatetime)
# Keys every stored record has (deepgym.labels.Label without its defaults).
_RECORD_KEYS = {'seq', 'id', 'target', 'type', 'source', 'author', 'created_at'}


class LabelStore:
    """Labels table of the console database."""

    def __init__(self, path: Path) -> None:
        self._path = path
        init_db(path)

    def add(self, label: LabelCreate) -> Label:
        """Store one label; see :meth:`add_many`."""
        return self.add_many([label])[0]

    def add_many(self, labels: Sequence[LabelCreate]) -> list[Label]:
        """Store *labels* in one transaction: every one of them, or none if any is refused.

        See :func:`insert_labels`.
        """
        with connect(self._path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            return insert_labels(conn, labels)

    def page(self, query: LabelQuery, offset: int, limit: int | None) -> LabelPage:
        """Return one page of the labels matching *query*, in ``seq`` order.

        The count and the page are read in one transaction, so they agree.

        Args:
            query: Filters; every one given must match.
            offset: How many of the earliest matching labels to skip.
            limit: Most labels to return; None returns every one after *offset*.

        Returns:
            The page, with the total count of matching labels.
        """
        where, params = _where(query)
        with connect(self._path) as conn:
            conn.execute('BEGIN')
            total = conn.execute(f'SELECT COUNT(*) FROM labels WHERE {where}', params).fetchone()[0]
            rows = conn.execute(
                f'SELECT record FROM labels WHERE {where} ORDER BY seq LIMIT ? OFFSET ?',
                (*params, -1 if limit is None else limit, offset),  # SQLite: LIMIT -1 is none
            ).fetchall()
        return LabelPage(labels=[_stored(row['record']) for row in rows], total=total)

    def current(self, kind: TargetKind, target_id: str) -> list[Label]:
        """Return the target's records still in force, its steps' too, in ``seq`` order.

        See :func:`deepgym.labels.current_labels`. A current invalidation is returned like
        any label; :meth:`excluded` names the targets to leave out.
        """
        query = LabelQuery(target_kind=kind, target_id=target_id, current=True)
        with connect(self._path) as conn:
            return select_labels(conn, query)

    def excluded(self) -> set[tuple[TargetKind, str]]:
        """Return the ``(kind, id)`` of each target a current invalidation excludes."""
        query = LabelQuery(type='invalidate', current=True)
        return excluded_targets(self.page(query, 0, None).labels)

    def export_jsonl(self, out: Path) -> int:
        """Write every record, superseded ones included, to *out* as JSON Lines in ``seq`` order.

        Returns:
            How many records were written.
        """
        out.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with connect(self._path) as conn, out.open('w', encoding='utf-8') as file:
            for row in conn.execute('SELECT record FROM labels ORDER BY seq'):
                file.write(f'{row["record"]}\n')
                written += 1
        return written


def read_labels(path: Path) -> list[Label]:
    """Return every label record in the console database at *path*, in ``seq`` order.

    *path* is opened read-only, so it is never created or migrated; a database from before
    labels has none.
    """
    with connect_read_only(path) as conn:
        if 'record' not in table_columns(conn, 'labels'):
            return []
        rows = conn.execute('SELECT record FROM labels ORDER BY seq').fetchall()
    return [_stored(row['record']) for row in rows]


def load_jsonl(path: Path) -> list[Label]:
    """Read the records :meth:`LabelStore.export_jsonl` wrote, as they were written.

    Raises:
        ValueError: If a line isn't a stored label record.
    """
    labels: list[Label] = []
    with path.open(encoding='utf-8') as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'{path} line {number} is not JSON: {exc}') from exc
            fits = (
                isinstance(data, dict) and _RECORD_KEYS <= data.keys() <= Label.model_fields.keys()
            )
            if not fits or not isinstance(data['target'], dict):
                raise ValueError(
                    f'{path} line {number} is not a label record, as LabelStore.export_jsonl '
                    'writes them'
                )
            labels.append(_constructed(data))
    return labels


def select_labels(conn: sqlite3.Connection, query: LabelQuery) -> list[Label]:
    """Return every label matching *query*, read on *conn* (so inside its transaction, if
    it holds one), in ``seq`` order."""
    where, params = _where(query)
    rows = conn.execute(f'SELECT record FROM labels WHERE {where} ORDER BY seq', params)
    return [_stored(row['record']) for row in rows]


def insert_labels(conn: sqlite3.Connection, labels: Sequence[LabelCreate]) -> list[Label]:
    """Store *labels* on *conn*, inside the write transaction (``BEGIN IMMEDIATE``) it holds.

    The time and positions are taken under the write lock, and the time never goes
    back, so ``seq`` order is also ``created_at`` order across processes. A label
    whose client id is stored with the same body is a retry: its stored record is
    returned, and nothing new is stored for it. Nothing is stored if the caller rolls back.

    Args:
        conn: A connection holding the write lock.
        labels: The labels, stored in this order under one creation time.

    Returns:
        The stored records, in the order of *labels*.

    Raises:
        ValueError: If a label's id is stored with another body, it supersedes a record
            that isn't stored or doesn't fit it (see ``LabelCreate.supersedes``), or it
            is a verdict its author already holds on that target (supersede that one).
    """
    seq, now = _last_position(conn)
    stored: list[Label] = []
    for label in labels:
        retried = _retried(conn, label)
        if retried is not None:
            stored.append(retried)
            continue
        seq += 1
        record = Label.model_validate(
            {**_body(label), 'seq': seq, 'id': label.id or uuid4().hex, 'created_at': now}
        )
        _check_supersedes(conn, record)
        _check_verdict(conn, record)
        try:
            conn.execute(_INSERT, _row(record))
        except sqlite3.IntegrityError as exc:  # the unique index on supersedes
            raise ValueError(f'Label {record.supersedes} is already superseded') from exc
        stored.append(record)
    return stored


def _row(label: Label) -> dict[str, object]:
    """Return the column values that store *label*."""
    return {
        'seq': label.seq, 'id': label.id, 'type': label.type, 'target_kind': label.target.kind,
        'target_id': label.target.id, 'step': label.target.step, 'name': label.name,
        'source': label.source, 'author': label.author, 'blind': label.blind,
        'rubric_version': label.rubric_version, 'rubric_sha256': label.rubric_sha256,
        'group_id': label.group_id, 'supersedes': label.supersedes,
        'created_at': iso_utc(label.created_at), 'record': label.model_dump_json(),
    }  # fmt: skip


def _stored(record: str) -> Label:
    """Rebuild a stored record as it was written.

    Only its JSON types are parsed; today's rules aren't checked again, so a rule added
    since it was stored can't make it unreadable.
    """
    return _constructed(json.loads(record))


def _constructed(data: dict) -> Label:
    """Build a :class:`Label` from a stored record's parsed JSON (see :func:`_stored`)."""
    return Label.model_construct(
        **{
            **data,
            'target': stored_target(data['target']),
            'created_at': _TIME.validate_python(data['created_at']),
        }
    )


def stored_target(target: dict[str, Any]) -> LabelTarget:
    """Rebuild a stored target from its JSON object, without checking today's rules."""
    if target.get('span') is not None:
        target = {**target, 'span': tuple(target['span'])}
    return LabelTarget.model_construct(**target)


def _body(label: LabelFields) -> dict[str, object]:
    """Return what *label*'s author gave, without a stored record's seq and time."""
    return label.model_dump(include=set(LabelFields.model_fields))


def _last_position(conn: sqlite3.Connection) -> tuple[int, datetime]:
    """Return the last stored seq (0 if none) and the time for the next records: now, or
    the last record's time if the clock reads earlier."""
    row = conn.execute('SELECT seq, created_at FROM labels ORDER BY seq DESC LIMIT 1').fetchone()
    now = datetime.now(timezone.utc)
    if row is None:
        return 0, now
    return row['seq'], max(now, datetime.fromisoformat(row['created_at']))


def _retried(conn: sqlite3.Connection, label: LabelCreate) -> Label | None:
    """Return the stored record *label* repeats, or None if its id is new or not given.

    Raises:
        ValueError: If *label*'s id is stored with another body.
    """
    if label.id is None:
        return None
    row = conn.execute('SELECT record FROM labels WHERE id = ?', (label.id,)).fetchone()
    if row is None:
        return None
    stored = _stored(row['record'])
    if _body(stored) != _body(label):
        raise ValueError(f'Label {label.id} is stored with another body; use a new id')
    return stored


def _check_supersedes(conn: sqlite3.Connection, label: Label) -> None:
    """Refuse *label* unless the record it supersedes, if any, is stored and fits it.

    Raises:
        ValueError: If that record isn't stored, is on another target, is a retraction, is
            from another source or author (unless a human retracts an invalidation), or
            has another type than a correction of it.
    """
    if label.supersedes is None:
        return
    row = conn.execute(
        'SELECT target_kind, target_id, source, author, type FROM labels WHERE id = ?',
        (label.supersedes,),
    ).fetchone()
    if row is None:
        raise ValueError(f'Label {label.supersedes} is not stored')
    if tuple(row)[:2] != (label.target.kind, label.target.id):
        raise ValueError(f'Label {label.supersedes} is on another target')
    if row['type'] == 'retract':
        raise ValueError(f'Label {label.supersedes} is a retraction, which is final')
    lifting = label.type == 'retract' and row['type'] == 'invalidate' and label.source == 'human'
    if not lifting and tuple(row)[2:4] != (label.source, label.author):
        raise ValueError(
            f'Label {label.supersedes} is from another source or author; '
            "another labeler's view is a new label"
        )
    if label.type != 'retract' and label.type != row['type']:
        raise ValueError(
            f'Label {label.supersedes} is a {row["type"]} record; a correction keeps its '
            'type, and a retraction withdraws it'
        )


def _check_verdict(conn: sqlite3.Connection, label: Label) -> None:
    """Refuse a verdict its source and author already hold on the target, name and group.

    Raises:
        ValueError: If *label* is a verdict and a current one of the same target, name,
            source, author and ``group_id`` exists that *label* doesn't supersede.
    """
    if not is_verdict(label.name):
        return
    rows = conn.execute(
        'SELECT id FROM labels WHERE target_kind = ? AND target_id = ? AND name = ? '
        f'AND source = ? AND author = ? AND group_id IS ? AND {_CURRENT}',
        (label.target.kind, label.target.id, label.name, label.source, label.author,
         label.group_id),
    ).fetchall()  # fmt: skip
    held = [row['id'] for row in rows if row['id'] != label.supersedes]
    if held:
        raise ValueError(
            f"{label.source} {label.author} already has a current '{label.name}' on "
            f'{label.target.kind} {label.target.id} (label {held[0]}); supersede it to '
            'change it, or give a repeat its own group_id'
        )


def _where(query: LabelQuery) -> tuple[str, list[object]]:
    """Build a WHERE clause (and its parameters) matching every filter *query* gives."""
    terms = [(f'{column} = ?', getattr(query, column)) for column in _EQUAL_FILTERS]
    terms += [
        ('created_at >= ?', None if query.since is None else iso_utc(query.since)),
        ('created_at < ?', None if query.until is None else iso_utc(query.until)),
        ('seq > ?', query.after_seq),
    ]
    active = [(sql, value) for sql, value in terms if value is not None]
    clauses = [sql for sql, _ in active] + ([_CURRENT] if query.current else [])
    return ' AND '.join(clauses) or '1', [value for _, value in active]
