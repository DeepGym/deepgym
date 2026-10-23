"""SQLite file behind the console: runs, recorded episodes and their screenshots."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from deepgym.models import RunResult

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
    finished_at TEXT
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
"""


def init_db(path: Path) -> None:
    """Create the console tables in *path* if they don't exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.executescript(_SCHEMA)
        # Older databases lack the newer episode columns. Holding the write lock makes
        # check-then-add atomic when two processes open an old database at once.
        conn.execute('BEGIN IMMEDIATE')
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(episodes)')}
        if 'trace' not in columns:
            conn.execute('ALTER TABLE episodes ADD COLUMN trace TEXT')
        if 'has_verdict' not in columns:
            _add_verdict_columns(conn)


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
    judged = result.has_verdict
    return (
        result.score if judged else None,
        result.passed if judged else None,
        result.error_type or (None if judged else 'verifier_error'),
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
