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
    RUN_LOG_RETENTION_DAYS,
    TRANSCRIPTIONS_DIR,
    VIDEO_EXTENSIONS,
)
from drive import download_file, list_pending_files, move_in_drive
from runlock import RunLockBusy, run_lock
from summary import generate_summary
from telegram_api import redact, send_telegram_message
from text import process_text
from utils import log, log_error, run_log
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
    transcribe_backend: str | None = None  # e.g. "Groq · whisper-large-v3"
    summarize_backend: str | None = None  # e.g. "local · Ollama qwen3:1.7b"


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


def _signed_message(filename, summary, transcribe_backend, summarize_backend, label="Resumen"):
    """Build the outbound Telegram text, signed with the engine(s) that produced it."""
    lines = [f"📋 {label}: {filename}", "", summary, ""]
    if transcribe_backend:
        lines.append(f"_transcripción: {transcribe_backend}_")
    lines.append(f"_resumen: {summarize_backend}_")
    return "\n".join(lines)


def _notify(text):
    """Post a run-progress/result message — identical for cron and manual /run.

    ``/retry`` and ``/resummarize`` don't call this: they're single-file,
    spawned via ``_single_file_run`` rather than ``run_pipeline``, and still
    get their own announce/report from ``handlers._spawn_and_report``.
    """
    send_telegram_message(text)


def _prune_old_run_logs():
    """Delete per-run log files older than ``RUN_LOG_RETENTION_DAYS``.

    ``RUN_LOG_DIR`` gets a new file every run and nothing else ever removes
    them; this keeps the directory bounded without a separate cron job. Any
    file that can't be removed (permissions, a race) is silently skipped.
    """
    cutoff = time.time() - RUN_LOG_RETENTION_DAYS * 86400
    for path in RUN_LOG_DIR.glob("run-*.log"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def _new_run(trigger, requested_by):
    """Make the runtime dirs, open a run row, and return (id, log_path, result)."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    store.init_db()
    _prune_old_run_logs()
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
        transcribe_backend=file_result.transcribe_backend,
        summarize_backend=file_result.summarize_backend,
    )


def process_file(filename, *, source="pending", notify_stages=False, position=None):
    """Fetch one file, transcribe/read it, summarize it, and send the summary.

    Args:
        filename: Name of the file.
        source: ``"pending"`` (default) — download from the pending folder and
            move it to processed; ``"processed"`` — re-fetch an already-handled
            file (used by ``retry``), leaving Drive untouched.
        notify_stages: If true, post a short Telegram message before each
            stage (download, transcription, summary) — set by ``run_pipeline``
            for both cron and manual ``/run`` passes.
        position: Optional ``(index, total)``, 1-based, appended to stage
            messages so a multi-file run reads as "(2/5)" and so on.

    Returns:
        A successful :class:`FileResult`.
    """
    started = time.monotonic()
    log(f"Processing: {filename} (source={source})")
    tag = f" ({position[0]}/{position[1]})" if position else ""

    def stage(text):
        if notify_stages:
            send_telegram_message(f"{text}{tag}")

    local_path = LOCAL_DIR / filename
    kind = _kind_of(filename)
    transcript = TRANSCRIPTIONS_DIR / f"{Path(filename).stem}.txt"

    stage(f"⬇️ descargando {filename}…")
    if source == "pending":
        download_file(filename, LOCAL_DIR)
        move_in_drive(filename)
    else:
        download_file(filename, LOCAL_DIR, folder=PROCESSED_FOLDER)

    transcribe_backend = None
    try:
        if kind == "video":
            stage(f"🎙️ transcribiendo {filename}…")
            text, transcribe_backend = process_video(filename, local_path)
        else:
            text = process_text(local_path)
            transcript.write_text(text, encoding="utf-8")

        stage(f"🧠 generando resumen de {filename}…")
        summary, summarize_backend = generate_summary(text)
        send_telegram_message(
            _signed_message(filename, summary, transcribe_backend, summarize_backend)
        )
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
        transcribe_backend=transcribe_backend,
        summarize_backend=summarize_backend,
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
    summary, summarize_backend = generate_summary(text)
    transcribe_backend = row.get("transcribe_backend")
    send_telegram_message(_signed_message(
        filename, summary, transcribe_backend, summarize_backend, label="Resumen (re)"
    ))
    log(f"Done: {filename}")
    return FileResult(
        filename=filename, kind=row["kind"], status="ok", transcript_path=str(path),
        summary=summary, duration_ms=int((time.monotonic() - started) * 1000),
        transcribe_backend=transcribe_backend, summarize_backend=summarize_backend,
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
            log_error(f"({what}) {filename}: {e}")
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
            _notify("⏸️ pipeline en pausa; corrida omitida.")
            store.finish_run(run_id, "skipped", 0, 0)
            result.status = "skipped"
            return result

        log(f"Run {run_id} started (trigger={trigger}).")
        try:
            files = list_pending_files()
        except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
            log_error(f"listing pending files: {e}")
            _notify(f"❌ no pude listar Drive: {redact(str(e))}.")
            store.finish_run(run_id, "error", 0, 0, error=str(e))
            result.status = "error"
            result.error = str(e)
            return result

        if only is not None:
            files = [f for f in files if f == only]
            if not files:
                log_error(f"Run {run_id}: '{only}' is not in the pending folder.")
                _notify(f"❌ '{only}' no está en la carpeta pendiente de Drive.")
                store.finish_run(run_id, "error", 0, 0, error=f"{only} not pending")
                result.status = "error"
                return result

        if not files:
            log("No new files.")
            _notify("🚀 sin archivos pendientes.")
            store.finish_run(run_id, "ok", 0, 0)
            return result

        _notify(f"🚀 iniciando: {len(files)} archivo(s) pendiente(s).")

        try:
            for index, filename in enumerate(files, start=1):
                try:
                    file_result = process_file(
                        filename, notify_stages=True, position=(index, len(files))
                    )
                except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
                    log_error(f"processing {filename}: {e}")
                    _notify(
                        f"❌ error procesando {filename} ({index}/{len(files)}): "
                        f"{redact(str(e))}"
                    )
                    file_result = FileResult(
                        filename=filename, kind=_kind_of(filename), status="error", error=str(e)
                    )
                result.files.append(file_result)
                _record(run_id, file_result)
        except KeyboardInterrupt:
            log(f"Run {run_id}: cancelled.")
            _notify(f"🛑 corrida cancelada ({result.files_ok}/{len(files)} completados).")
            store.finish_run(run_id, "cancelled", len(result.files), result.files_ok)
            result.status = "cancelled"
            return result

        total = len(result.files)
        ok = result.files_ok
        result.status = "ok" if ok == total else ("partial" if ok else "error")
        store.finish_run(run_id, result.status, total, ok)
        log(f"Run {run_id} finished: {ok}/{total} ok ({result.status}).")
        icon = "✅" if ok == total else ("❌" if ok == 0 else "⚠️")
        _notify(f"{icon} corrida finalizada: {ok}/{total} ok.")

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
