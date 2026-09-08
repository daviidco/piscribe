"""Tests for the SQLite run/file history store."""

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
