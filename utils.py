"""Small shared helpers for the pipeline."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from config import LOG_FILE

# Mutable module state (the active per-run log path), not a constant.
_run_log_path = None  # pylint: disable=invalid-name


@contextmanager
def run_log(path):
    """Also mirror every :func:`log` line into ``path`` for the duration.

    Used by ``run_pipeline`` so each run gets its own retrievable log file on
    top of the shared, ever-growing ``LOG_FILE``.
    """
    global _run_log_path  # pylint: disable=global-statement
    previous = _run_log_path
    _run_log_path = Path(path)
    try:
        yield
    finally:
        _run_log_path = previous


def log(msg):
    """Print a timestamped message and append it to the log file(s).

    The line goes to stdout, to the shared ``LOG_FILE``, and — while a
    :func:`run_log` context is active — to that run's own log file too.

    Args:
        msg: The message to record.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if _run_log_path is not None:
        with open(_run_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
