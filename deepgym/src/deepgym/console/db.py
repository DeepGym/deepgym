"""SQLite file behind the console: runs, recorded episodes, their screenshots and labels."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from deepgym.models import RunResult, is_verdict

DEFAULT_DB_PATH = Path.home() / '.deepgym' / 'console.db'

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL,
    environment TEXT NOT NULL,
    mode TEXT NOT NULL,
    provider TEXT NOT NULL,
    reference INTEGER NOT NULL,
    solution TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL,
    passed INTEGER,
    error_type TEXT,
    error TEXT,
    duration_ms REAL,
    result TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    environment_sha256 TEXT
);
CREATE INDEX IF NOT EXISTS runs_created_at ON runs (created_at);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    environment TEXT NOT NULL,
    steps INTEGER NOT NULL,
    total_reward REAL NOT NULL,
    score REAL,
    passed INTEGER,
    frames TEXT NOT NULL,
    result TEXT,
    created_at TEXT NOT NULL,
    trace TEXT,
    has_verdict INTEGER,
    error_type TEXT
);
CREATE INDEX IF NOT EXISTS episodes_created_at ON episodes (created_at);
CREATE TABLE IF NOT EXISTS screenshots (
    episode_id TEXT NOT NULL,
    frame INTEGER NOT NULL,
    png BLOB NOT NULL,
    PRIMARY KEY (episode_id, frame)
);
CREATE TABLE IF NOT EXISTS labels (
    seq INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    step INTEGER,
    name TEXT,
    source TEXT NOT NULL,
    author TEXT NOT NULL,
    blind INTEGER,
    rubric_version TEXT,
    rubric_sha256 TEXT,
    group_id TEXT,
    supersedes TEXT,
    created_at TEXT NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS labels_target ON labels (target_kind, target_id);
-- A label is superseded at most once, so its corrections never fork.
CREATE UNIQUE INDEX IF NOT EXISTS labels_supersedes ON labels (supersedes);
"""


def init_db(path: Path) -> None:
    """Create the console tables in *path* if they don't exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.executescript(_SCHEMA)
        # Older databases lack the newer columns. Holding the write lock makes
        # check-then-add atomic when two processes open an old database at once.
        conn.execute('BEGIN IMMEDIATE')
        if 'environment_sha256' not in _columns(conn, 'runs'):
            _add_hash_column(conn)
        columns = _columns(conn, 'episodes')
        if 'trace' not in columns:
            conn.execute('ALTER TABLE episodes ADD COLUMN trace TEXT')
        if 'has_verdict' not in columns:
            _add_verdict_columns(conn)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the column names of *table* (a name from this module, never user input)."""
    return {row['name'] for row in conn.execute(f'PRAGMA table_info({table})')}


def _add_hash_column(conn: sqlite3.Connection) -> None:
    """Add the runs' environment_sha256, and store each older run's verdict as runs do now.

    A run stored before it may keep a 0 score for a result without a verdict, or a score
    on a run that failed: each stored result's verdict is stored again (see
    :func:`_stored_verdict`), and a failed run keeps no score or pass.
    """
    conn.execute('ALTER TABLE runs ADD COLUMN environment_sha256 TEXT')
    rows = conn.execute('SELECT id, result FROM runs WHERE result IS NOT NULL').fetchall()
    conn.executemany(
        'UPDATE runs SET score = ?, passed = ?, error_type = ? WHERE id = ?',
        [(*_stored_verdict(row['result']), row['id']) for row in rows],
    )
    conn.execute("UPDATE runs SET score = NULL, passed = NULL WHERE status = 'failed'")


def _stored_verdict(raw: str) -> tuple[float | None, bool | None, str | None]:
    """Return :func:`verdict_columns` for a stored result's JSON, read field by field.

    Old rows aren't validated as a whole :class:`RunResult`, so one that no longer parses
    can't stop the console opening. A row that isn't a JSON object, or lacks a numeric
    score or a boolean pass, has no verdict.
    """
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return _columns_of(None, None, judged=False, error_type=None)
    score, passed, error_type = data.get('score'), data.get('passed'), data.get('error_type')
    error_type = error_type if isinstance(error_type, str) else None
    judged = (
        is_verdict(data.get('exit_code'), error_type)
        and isinstance(score, (int, float))
        and isinstance(passed, bool)
    )
    return _columns_of(score, passed, judged=judged, error_type=error_type)


def _add_verdict_columns(conn: sqlite3.Connection) -> None:
    """Add the episodes' has_verdict and error_type, filled in from each stored result.

    Episodes stored before them kept a result without a verdict as a 0 score; that
    score and pass are cleared, as :func:`verdict_columns` stores them now.
    """
    conn.execute('ALTER TABLE episodes ADD COLUMN has_verdict INTEGER')
    conn.execute('ALTER TABLE episodes ADD COLUMN error_type TEXT')
    rows = conn.execute('SELECT id, result FROM episodes WHERE result IS NOT NULL').fetchall()
    results = [(row['id'], RunResult.model_validate_json(row['result'])) for row in rows]
    conn.executemany(
        'UPDATE episodes SET score = ?, passed = ?, error_type = ?, has_verdict = ? WHERE id = ?',
        [
            (*verdict_columns(result), result.has_verdict, episode_id)
            for episode_id, result in results
        ],
    )


def verdict_columns(result: RunResult) -> tuple[float | None, bool | None, str | None]:
    """Return the score, passed and error_type to store for *result*.

    A result without a verdict (``RunResult.has_verdict`` False) keeps no score or pass,
    and always has an error type: its own, else ``'verifier_error'``.
    """
    return _columns_of(
        result.score, result.passed, judged=result.has_verdict, error_type=result.error_type
    )


def _columns_of(
    score: float | None, passed: bool | None, *, judged: bool, error_type: str | None
) -> tuple[float | None, bool | None, str | None]:
    """Score, passed and error_type to store; see :func:`verdict_columns`."""
    return (
        score if judged else None,
        passed if judged else None,
        error_type or (None if judged else 'verifier_error'),
    )


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open *path*, commit on success (roll back on error), and always close.

    One connection per call keeps the store safe to use from the run thread pool.
    """
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
