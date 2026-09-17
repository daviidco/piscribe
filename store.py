"""SQLite-backed history of pipeline runs and per-file results.

``pipeline.py`` writes here as it works; the Telegram bot only reads. One file
at ``DB_PATH``, WAL mode, no server. Timestamps are ``YYYY-MM-DD HH:MM:SS`` UTC
strings so SQLite ``datetime()`` math works on them directly.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger      TEXT NOT NULL,
    requested_by TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL,
    files_total  INTEGER NOT NULL DEFAULT 0,
    files_ok     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    log_path     TEXT
);
CREATE TABLE IF NOT EXISTS files (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER NOT NULL REFERENCES runs(id),
    filename           TEXT NOT NULL,
    kind               TEXT NOT NULL,
    transcript_path    TEXT,
    summary            TEXT,
    status             TEXT NOT NULL,
    error              TEXT,
    duration_ms        INTEGER,
    created_at         TEXT NOT NULL,
    transcribe_backend TEXT,
    summarize_backend  TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_run ON files(run_id);
CREATE INDEX IF NOT EXISTS idx_files_id ON files(id DESC);
"""

# Columns added after the initial release: CREATE TABLE above only applies to a
# brand-new DB, so an existing files table (e.g. already deployed on the Pi)
# needs these added explicitly.
_FILES_MIGRATIONS = {
    "transcribe_backend": "TEXT",
    "summarize_backend": "TEXT",
}


def _ensure_columns(conn, table, columns):
    """ALTER TABLE ``table`` to add any of ``columns`` (name -> SQL type) missing."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, col_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)  # all IF NOT EXISTS; cheap when the DB is set up
    _ensure_columns(conn, "files", _FILES_MIGRATIONS)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Ensure the schema exists. Optional — every connection also ensures it."""
    with _connect():
        pass


# --- writes (pipeline.py) --------------------------------------------------

def start_run(trigger, requested_by=None, log_path=None):
    """Insert a ``running`` run row and return its id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (trigger, requested_by, started_at, status, log_path) "
            "VALUES (?, ?, ?, 'running', ?)",
            (trigger, requested_by, _utcnow(), str(log_path) if log_path else None),
        )
        return cur.lastrowid


def record_file(run_id, filename, kind, status, *, transcript_path=None,
                summary=None, error=None, duration_ms=None,
                transcribe_backend=None, summarize_backend=None):
    """Insert one per-file result for ``run_id``.

    ``transcribe_backend``/``summarize_backend`` record which engine produced
    the transcript/summary, e.g. ``"Groq · whisper-large-v3"`` or
    ``"local · Ollama qwen3:1.7b"`` — provenance for ``/recap``, ``/stats``, and
    debugging the Groq-first/local-fallback resilience.
    """
    with _connect() as conn:
        conn.execute(
            "INSERT INTO files (run_id, filename, kind, transcript_path, summary, "
            "status, error, duration_ms, created_at, transcribe_backend, summarize_backend) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, filename, kind, transcript_path, summary, status, error,
             duration_ms, _utcnow(), transcribe_backend, summarize_backend),
        )


def finish_run(run_id, status, files_total, files_ok, error=None):
    """Mark a run finished with its final status and counts."""
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET finished_at=?, status=?, files_total=?, files_ok=?, error=? "
            "WHERE id=?",
            (_utcnow(), status, files_total, files_ok, error, run_id),
        )


# --- reads (bot) --------------------------------------------------

def last_run():
    """Most recent run row as a dict, or ``None``."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def run_by_id(run_id):
    """A specific run by its id (the ``#N`` shown everywhere else — /history,
    /status, the post-run report, the run's own log lines), or ``None``."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def recent_runs(limit=10):
    """Up to ``limit`` most recent runs, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def last_ok_at():
    """``finished_at`` of the most recent ``ok``/``partial`` run, or ``None``."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT finished_at FROM runs WHERE status IN ('ok', 'partial') "
            "AND finished_at IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["finished_at"] if row else None


def latest_file():
    """The single most recently processed file, or ``None``."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM files ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def file_by_id(file_id):
    """A specific file by its id (the ``#N`` shown by ``/find``), or ``None``."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None


def files_for_run(run_id):
    """All file rows recorded for ``run_id``, in processing order."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM files WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def file_by_name(name):
    """Most recent file row whose ``filename`` matches ``name``, or ``None``."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM files WHERE filename = ? ORDER BY id DESC LIMIT 1", (name,)
        ).fetchone()
        return dict(row) if row else None


def search(term, limit=10):
    """File rows whose filename or summary contains ``term``, newest first."""
    like = f"%{term}%"
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM files WHERE filename LIKE ? OR summary LIKE ? "
            "ORDER BY id DESC LIMIT ?", (like, like, limit)
        ).fetchall()
        return [dict(r) for r in rows]


def stats():
    """Rolling counts for ``/stats`` (last 24 h and 7 d, UTC)."""
    with _connect() as conn:
        def one(sql):
            return conn.execute(sql).fetchone()[0]

        day = "datetime('now', '-1 day')"
        week = "datetime('now', '-7 day')"
        return {
            "files_24h": one(f"SELECT COUNT(*) FROM files WHERE created_at >= {day}"),
            "files_7d": one(f"SELECT COUNT(*) FROM files WHERE created_at >= {week}"),
            "errors_7d": one(
                f"SELECT COUNT(*) FROM files WHERE status='error' AND created_at >= {week}"
            ),
            "avg_ms_7d": one(
                "SELECT AVG(duration_ms) FROM files WHERE status='ok' "
                f"AND duration_ms IS NOT NULL AND created_at >= {week}"
            ),
            "runs_7d": one(f"SELECT COUNT(*) FROM runs WHERE started_at >= {week}"),
        }
