"""Small shared helpers for the pipeline."""

from datetime import datetime

from config import LOG_FILE


def log(msg):
    """Print a timestamped message and append it to the log file.

    The line is written to both stdout (useful when run interactively) and
    ``LOG_FILE`` (useful when run from cron), prefixed with a
    ``[YYYY-MM-DD HH:MM:SS]`` timestamp.

    Args:
        msg: The message to record.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
