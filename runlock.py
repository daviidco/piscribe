"""Cross-process run lock backed by ``flock``.

Both the cron job and the Telegram bot's ``/run`` end up executing
``pipeline.py``; this lock guarantees only one pass runs at a time. The holder
writes its PID into the lock file so ``/cancel`` (and ``/status``) can find the
running process regardless of who started it.
"""

import fcntl
import os
from contextlib import contextmanager

from config import LOCK_PATH


class RunLockBusy(RuntimeError):
    """Raised when another process already holds the run lock."""


@contextmanager
def run_lock():
    """Acquire the run lock for the duration of the block.

    Raises:
        RunLockBusy: If another process holds it. The message includes the
            holder's PID when it can be read.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_PATH, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RunLockBusy(_busy_message()) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def current_run_pid():
    """Return the PID recorded in the lock file, or ``None`` if unlocked/stale.

    A PID is returned only if that process is currently alive.
    """
    try:
        raw = LOCK_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not raw.isdigit():
        return None
    pid = int(raw)
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _busy_message():
    try:
        raw = LOCK_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        raw = ""
    return f"run already in progress (pid {raw})" if raw else "run already in progress"
