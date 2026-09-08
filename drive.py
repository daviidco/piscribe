"""Google Drive access through the ``rclone`` command-line tool.

Every Drive operation shells out to ``rclone`` against the remote named by
``RCLONE_REMOTE``. This module also exposes :func:`run`, the generic subprocess
wrapper reused by other modules for logged, checked command execution.
"""

import subprocess
from pathlib import Path

from config import (
    PENDING_FOLDER,
    PROCESSED_FOLDER,
    RCLONE_REMOTE,
    TEXT_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from utils import log


def run(cmd):
    """Run a command, logging it first and raising on a non-zero exit.

    Args:
        cmd: The command as a list of arguments (strings or path-likes).

    Returns:
        The completed ``subprocess.CompletedProcess`` with captured text output.

    Raises:
        RuntimeError: If the command exits with a non-zero status; the message
            includes the command and its stderr.
    """
    log(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result


def list_pending_files():
    """List processable files waiting in the Drive pending folder.

    Returns:
        A list of file names (not full paths) whose extension is in
        ``VIDEO_EXTENSIONS`` or ``TEXT_EXTENSIONS``. Sub-directories and any
        other entry in the folder are ignored.
    """
    result = run(["rclone", "lsf", "--files-only", f"{RCLONE_REMOTE}:{PENDING_FOLDER}"])
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    valid_exts = VIDEO_EXTENSIONS | TEXT_EXTENSIONS
    return [f for f in files if Path(f).suffix.lower() in valid_exts]


def download_file(filename, dest_dir):
    """Copy a file from the Drive pending folder into a local directory.

    Args:
        filename: Name of the file inside the pending folder.
        dest_dir: Local directory to copy the file into.
    """
    run([
        "rclone", "copy",
        f"{RCLONE_REMOTE}:{PENDING_FOLDER}/{filename}",
        str(dest_dir),
    ])


def move_in_drive(filename):
    """Move a file from the pending folder to the processed folder in Drive.

    Done as soon as a file is picked up so it is not handled twice on a later
    run.

    Args:
        filename: Name of the file to move.
    """
    run([
        "rclone", "moveto",
        f"{RCLONE_REMOTE}:{PENDING_FOLDER}/{filename}",
        f"{RCLONE_REMOTE}:{PROCESSED_FOLDER}/{filename}",
    ])
