"""Tests for the SQLite run/file history store."""

import sqlite3

import store


def test_run_lifecycle_and_reads():
    """A run written start -> file -> file -> finish reads back consistently."""
    store.init_db()
    run_id = store.start_run("cron", log_path="/tmp/run.log")

    store.record_file(run_id, "a.mp4", "video", "ok",
                      transcript_path="/t/a.txt", summary="resumen A", duration_ms=1200)
    store.record_file(run_id, "b.txt", "text", "error", error="boom")
    store.finish_run(run_id, "partial", files_total=2, files_ok=1)

    last = store.last_run()
    assert last["id"] == run_id
    assert last["status"] == "partial"
    assert (last["files_ok"], last["files_total"]) == (1, 2)
    assert last["log_path"] == "/tmp/run.log"

    assert store.nth_file(1)["filename"] == "b.txt"
    assert store.nth_file(2)["filename"] == "a.mp4"
    assert store.nth_file(2)["summary"] == "resumen A"
    assert store.file_by_name("a.mp4")["transcript_path"] == "/t/a.txt"
    assert store.file_by_name("nope") is None


def test_last_ok_at_only_counts_successful_runs():
    """last_ok_at ignores running/error runs."""
    store.init_db()
    assert store.last_ok_at() is None

    bad = store.start_run("cron")
    store.finish_run(bad, "error", 0, 0, error="x")
    assert store.last_ok_at() is None

    good = store.start_run("manual", requested_by="1")
    store.finish_run(good, "ok", 1, 1)
    assert store.last_ok_at() is not None


def test_run_by_id_fetches_the_exact_run_regardless_of_how_many_exist():
    """run_by_id looks up by the literal id, not by recency/position."""
    store.init_db()
    first_id = store.start_run("cron")
    store.finish_run(first_id, "ok", 0, 0)
    for _ in range(5):
        store.finish_run(store.start_run("cron"), "ok", 0, 0)

    assert store.run_by_id(first_id)["id"] == first_id
    assert store.run_by_id(999999) is None


def test_recent_runs_is_newest_first():
    """recent_runs returns rows in descending id order, capped at the limit."""
    store.init_db()
    ids = [store.start_run("cron") for _ in range(5)]
    for run_id in ids:
        store.finish_run(run_id, "ok", 0, 0)

    rows = store.recent_runs(limit=3)
    assert [r["id"] for r in rows] == sorted(ids, reverse=True)[:3]


def test_search_matches_filename_and_summary():
    """search() finds by filename fragment or summary fragment, newest first."""
    store.init_db()
    rid = store.start_run("cron")
    store.record_file(rid, "reunion-abril.mp4", "video", "ok", summary="acuerdo importante")
    store.record_file(rid, "nota.txt", "text", "ok", summary="tema presupuesto")
    store.finish_run(rid, "ok", 2, 2)

    assert [r["filename"] for r in store.search("abril")] == ["reunion-abril.mp4"]
    assert [r["filename"] for r in store.search("presupuesto")] == ["nota.txt"]
    assert store.search("nada-de-esto") == []


def test_stats_counts_recent_activity():
    """stats() aggregates files, errors, runs and average duration."""
    store.init_db()
    rid = store.start_run("cron")
    store.record_file(rid, "a.mp4", "video", "ok", duration_ms=2000)
    store.record_file(rid, "b.mp4", "video", "ok", duration_ms=4000)
    store.record_file(rid, "c.txt", "text", "error", error="x")
    store.finish_run(rid, "partial", 3, 2)

    data = store.stats()
    assert data["files_24h"] == 3
    assert data["files_7d"] == 3
    assert data["errors_7d"] == 1
    assert data["runs_7d"] == 1
    assert data["avg_ms_7d"] == 3000


def test_record_file_stores_backend_provenance():
    """record_file keeps which engine transcribed/summarized a file."""
    store.init_db()
    rid = store.start_run("cron")
    store.record_file(rid, "reunion.mp4", "video", "ok",
                      transcribe_backend="Groq · whisper-large-v3",
                      summarize_backend="local · Ollama qwen3:1.7b")

    row = store.nth_file(1)
    assert row["transcribe_backend"] == "Groq · whisper-large-v3"
    assert row["summarize_backend"] == "local · Ollama qwen3:1.7b"


def test_migration_adds_backend_columns_to_a_pre_existing_files_table():
    """A files table created before the Groq columns existed still works.

    Simulates the Pi's already-deployed piscribe.db: the old schema is
    created by hand (no transcribe_backend/summarize_backend), then the
    normal store API is used — _connect() must ALTER TABLE them in rather
    than fail with "no such column".
    """
    conn = sqlite3.connect(store.DB_PATH)
    conn.executescript("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger TEXT NOT NULL,
            requested_by TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            files_total INTEGER NOT NULL DEFAULT 0,
            files_ok INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            log_path TEXT
        );
        CREATE TABLE files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            kind TEXT NOT NULL,
            transcript_path TEXT,
            summary TEXT,
            status TEXT NOT NULL,
            error TEXT,
            duration_ms INTEGER,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

    run_id = store.start_run("cron")
    store.record_file(run_id, "old.mp4", "video", "ok",
                      transcribe_backend="Groq · whisper-large-v3",
                      summarize_backend="local · Ollama qwen3:1.7b")

    row = store.nth_file(1)
    assert row["transcribe_backend"] == "Groq · whisper-large-v3"
    assert row["summarize_backend"] == "local · Ollama qwen3:1.7b"
