#!/usr/bin/env python3
"""One pass of the piscribe pipeline.

The default mode processes every processable file in the Drive pending folder:
download it, move it to the processed folder, extract its text (transcribing
video with ``whisper.cpp`` or reading text directly), archive that text,
summarize it in Spanish with a local Qwen model, and post the summary to
Telegram. A per-file failure is logged and does not stop the rest.

``main()`` is what cron runs: it takes a cross-process lock, then dispatches on
environment variables the bot sets when it spawns this script:

* ``PISCRIBE_MODE``     ``run`` (default) | ``retry`` | ``resummarize``
* ``PISCRIBE_TRIGGER``  ``cron`` (default) | ``manual``
* ``PISCRIBE_BY``       Telegram id that asked for a manual run
* ``PISCRIBE_ONLY``     restrict ``run`` to one pending file; the target for
  ``retry`` (re-fetched from the processed folder) and ``resummarize``
  (re-run only the summary over the stored transcript)

Every pass is recorded in the SQLite store and gets its own log file. A
``SIGTERM`` (from the bot's ``/cancel``) unwinds cleanly and marks the run
``cancelled``.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import store
from config import (
    LOCAL_DIR,
    PAUSE_FLAG,
    PROCESSED_FOLDER,
    RUN_LOG_DIR,
    TRANSCRIPTIONS_DIR,
    VIDEO_EXTENSIONS,
)
from drive import download_file, list_pending_files, move_in_drive
from runlock import RunLockBusy, run_lock
from summary import generate_summary
from telegram_api import send_telegram_message
from text import process_text
from utils import log, run_log
from video import process_video


@dataclass
class FileResult:
    """Outcome of processing one file."""

    filename: str
    kind: str
    status: str  # "ok" | "error"
    transcript_path: str | None = None
    summary: str | None = None
    error: str | None = None
    duration_ms: int | None = None


@dataclass
class RunResult:
    """Outcome of one pipeline pass."""

    run_id: int
    trigger: str
    status: str  # "ok" | "partial" | "error" | "skipped" | "cancelled"
    files: list = field(default_factory=list)
    error: str | None = None

    @property
    def files_ok(self):
        """How many files in this run succeeded."""
        return sum(1 for f in self.files if f.status == "ok")


def _kind_of(filename):
    """Return ``"video"`` or ``"text"`` from a filename's extension."""
    return "video" if Path(filename).suffix.lower() in VIDEO_EXTENSIONS else "text"


def _new_run(trigger, requested_by):
    """Make the runtime dirs, open a run row, and return (id, log_path, result)."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    store.init_db()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = RUN_LOG_DIR / f"run-{stamp}-{trigger}.log"
    run_id = store.start_run(trigger, requested_by, log_path=log_path)
    return run_id, log_path, RunResult(run_id=run_id, trigger=trigger, status="ok")


def _record(run_id, file_result):
    """Persist one :class:`FileResult`."""
    store.record_file(
        run_id, file_result.filename, file_result.kind, file_result.status,
        transcript_path=file_result.transcript_path, summary=file_result.summary,
        error=file_result.error, duration_ms=file_result.duration_ms,
    )


def process_file(filename, *, source="pending"):
    """Fetch one file, transcribe/read it, summarize it, and send the summary.

    Args:
        filename: Name of the file.
        source: ``"pending"`` (default) — download from the pending folder and
            move it to processed; ``"processed"`` — re-fetch an already-handled
            file (used by ``retry``), leaving Drive untouched.

    Returns:
        A successful :class:`FileResult`.
    """
    started = time.monotonic()
    log(f"Processing: {filename} (source={source})")
    local_path = LOCAL_DIR / filename
    kind = _kind_of(filename)
    transcript = TRANSCRIPTIONS_DIR / f"{Path(filename).stem}.txt"

    if source == "pending":
        download_file(filename, LOCAL_DIR)
        move_in_drive(filename)
    else:
        download_file(filename, LOCAL_DIR, folder=PROCESSED_FOLDER)

    try:
        if kind == "video":
            text = process_video(filename, local_path)
        else:
            text = process_text(local_path)
            transcript.write_text(text, encoding="utf-8")

        log("Generating summary with Qwen...")
        summary = generate_summary(text)
        send_telegram_message(f"📋 Summary: {filename}\n\n{summary}")
    finally:
        local_path.unlink(missing_ok=True)

    log(f"Done: {filename}")
    return FileResult(
        filename=filename,
        kind=kind,
        status="ok",
        transcript_path=str(transcript) if transcript.is_file() else None,
        summary=summary,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _resummarize(filename):
    """Re-run only the summary step over ``filename``'s stored transcript."""
    started = time.monotonic()
    row = store.file_by_name(filename)
    if not row or not row.get("transcript_path"):
        raise RuntimeError(f"no stored transcript for {filename}")
    path = Path(row["transcript_path"])
    if not path.is_file():
        raise RuntimeError(f"transcript file missing: {path}")

    text = path.read_text(encoding="utf-8")
    log("Re-generating summary with Qwen...")
    summary = generate_summary(text)
    send_telegram_message(f"📋 Summary (re): {filename}\n\n{summary}")
    log(f"Done: {filename}")
    return FileResult(
        filename=filename, kind=row["kind"], status="ok", transcript_path=str(path),
        summary=summary, duration_ms=int((time.monotonic() - started) * 1000),
    )


def _single_file_run(trigger, requested_by, filename, worker, what):
    """Run ``worker(filename) -> FileResult`` as its own one-file recorded run."""
    if not filename:
        raise ValueError(f"{what} needs a filename")
    run_id, log_path, result = _new_run(trigger, requested_by)
    with run_log(log_path):
        log(f"Run {run_id}: {what} {filename}.")
        try:
            file_result = worker(filename)
        except KeyboardInterrupt:
            log(f"Run {run_id}: cancelled.")
            store.finish_run(run_id, "cancelled", 0, 0)
            result.status = "cancelled"
            return result
        except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
            log(f"ERROR ({what}) {filename}: {e}")
            file_result = FileResult(filename, _kind_of(filename), "error", error=str(e))
        result.files.append(file_result)
        _record(run_id, file_result)
        status = "ok" if file_result.status == "ok" else "error"
        store.finish_run(run_id, status, 1, 1 if status == "ok" else 0)
        result.status = status
        log(f"Run {run_id} finished: {status}.")
    return result


def retry_file(filename, trigger="manual", requested_by=None):
    """Reprocess ``filename`` from the processed folder as its own run."""
    return _single_file_run(
        trigger, requested_by, filename,
        lambda name: process_file(name, source="processed"), "retry",
    )


def resummarize_file(filename, trigger="manual", requested_by=None):
    """Re-run only the summary for ``filename`` as its own run."""
    return _single_file_run(trigger, requested_by, filename, _resummarize, "resummarize")


def run_pipeline(trigger, requested_by=None, only=None):
    """Process the pending folder once, recording the run in the store.

    Per-file failures are isolated; a failure while listing the pending folder
    ends the run without a traceback. If the pause flag is set, the run is
    recorded as ``skipped``. If ``only`` is given, just that pending file is
    processed.

    Args:
        trigger: ``"cron"`` or ``"manual"``.
        requested_by: Telegram id that asked for a manual run, if any.
        only: Restrict the pass to this one pending filename.

    Returns:
        A :class:`RunResult`.
    """
    run_id, log_path, result = _new_run(trigger, requested_by)

    with run_log(log_path):
        if PAUSE_FLAG.exists():
            log(f"Run {run_id}: pipeline is paused ({PAUSE_FLAG}); skipping.")
            store.finish_run(run_id, "skipped", 0, 0)
            result.status = "skipped"
            return result

        log(f"Run {run_id} started (trigger={trigger}).")
        try:
            files = list_pending_files()
        except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
            log(f"ERROR listing pending files: {e}")
            store.finish_run(run_id, "error", 0, 0, error=str(e))
            result.status = "error"
            result.error = str(e)
            return result

        if only is not None:
            files = [f for f in files if f == only]
            if not files:
                log(f"Run {run_id}: '{only}' is not in the pending folder.")
                store.finish_run(run_id, "error", 0, 0, error=f"{only} not pending")
                result.status = "error"
                return result

        if not files:
            log("No new files.")
            store.finish_run(run_id, "ok", 0, 0)
            return result

        try:
            for filename in files:
                try:
                    file_result = process_file(filename)
                except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
                    log(f"ERROR processing {filename}: {e}")
                    file_result = FileResult(
                        filename=filename, kind=_kind_of(filename), status="error", error=str(e)
                    )
                result.files.append(file_result)
                _record(run_id, file_result)
        except KeyboardInterrupt:
            log(f"Run {run_id}: cancelled.")
            store.finish_run(run_id, "cancelled", len(result.files), result.files_ok)
            result.status = "cancelled"
            return result

        total = len(result.files)
        ok = result.files_ok
        result.status = "ok" if ok == total else ("partial" if ok else "error")
        store.finish_run(run_id, result.status, total, ok)
        log(f"Run {run_id} finished: {ok}/{total} ok ({result.status}).")

    return result


def _raise_keyboard_interrupt(signum, frame):  # noqa: ARG001
    """SIGTERM handler: turn it into KeyboardInterrupt so cleanup runs."""
    raise KeyboardInterrupt()


def main():
    """Take the run lock and dispatch on the ``PISCRIBE_*`` environment.

    If another run already holds the lock, exit quietly instead of stacking work.
    """
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    trigger = os.environ.get("PISCRIBE_TRIGGER", "cron")
    requested_by = os.environ.get("PISCRIBE_BY") or None
    only = os.environ.get("PISCRIBE_ONLY") or None
    mode = os.environ.get("PISCRIBE_MODE", "run")

    try:
        with run_lock():
            if mode == "retry":
                retry_file(only, trigger, requested_by)
            elif mode == "resummarize":
                resummarize_file(only, trigger, requested_by)
            else:
                run_pipeline(trigger, requested_by, only=only)
    except RunLockBusy as e:
        log(f"Skipping {trigger} run: {e}")
    except KeyboardInterrupt:
        log("Run interrupted (SIGTERM).")


if __name__ == "__main__":
    main()
