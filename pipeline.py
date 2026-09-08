#!/usr/bin/env python3
"""Entry point that runs one pass of the piscribe pipeline.

For each processable file in the Drive pending folder this script downloads it,
moves it to the processed folder, extracts its text (transcribing video with
``whisper.cpp`` or reading text files directly), summarizes it in Spanish with a
local Qwen model, and posts the summary to Telegram. A failure on one file is
logged and does not stop the rest.

Run a single pass with ``python pipeline.py`` (typically from cron).
"""

from pathlib import Path

from config import LOCAL_DIR, TRANSCRIPTIONS_DIR, VIDEO_EXTENSIONS
from drive import download_file, list_pending_files, move_in_drive
from summary import generate_summary
from telegram import send_telegram_message
from text import process_text
from utils import log
from video import process_video


def process_file(filename):
    """Run the full pipeline for a single pending file.

    Downloads the file, moves it to the processed folder in Drive, extracts its
    text (transcription for video, direct read for text), generates a Spanish
    summary, and sends it to Telegram. The local copy is always deleted, even if
    a later step fails.

    Args:
        filename: Name of the file in the Drive pending folder.
    """
    log(f"Processing: {filename}")
    local_path = LOCAL_DIR / filename

    download_file(filename, LOCAL_DIR)
    move_in_drive(filename)

    try:
        extension = Path(filename).suffix.lower()
        if extension in VIDEO_EXTENSIONS:
            text = process_video(filename, local_path)
        else:
            text = process_text(local_path)

        log("Generating summary with Qwen...")
        summary = generate_summary(text)
        send_telegram_message(f"📋 Summary: {filename}\n\n{summary}")
    finally:
        local_path.unlink(missing_ok=True)
    log(f"Done: {filename}")


def main():
    """Process every pending file once, then exit.

    Ensures the local working directories exist and iterates over the pending
    files, isolating per-file failures so one bad file does not abort the batch.
    A failure while listing the pending folder (e.g. rclone missing or offline)
    is logged and ends the run without a traceback.
    """
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        files = list_pending_files()
    except Exception as e:
        log(f"ERROR listing pending files: {e}")
        return

    if not files:
        log("No new files.")
        return

    for filename in files:
        try:
            process_file(filename)
        except Exception as e:
            log(f"ERROR processing {filename}: {e}")


if __name__ == "__main__":
    main()
