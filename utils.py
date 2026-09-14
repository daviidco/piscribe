"""Logging for the pipeline: one shared, level-aware, UTC-timestamped format.

``log()``/``log_warning()``/``log_error()`` write to stdout and to the shared,
size-rotated ``LOG_FILE``; ``run_log()`` additionally mirrors into one run's own
log file for the duration of the block. ``bot.py`` reuses ``LOG_FORMAT``/
``LOG_DATEFMT``/``configure_utc_formatter`` so pipeline logs (``/logs``) and bot
logs (``journalctl``) read the same way.
"""

import logging
import sys
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

from config import LOG_FILE

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Bound log.txt's growth instead of letting it accumulate forever: rotates to
# log.txt.1, .2, .3 once it passes 5 MB, ~20 MB total across all of them.
_LOG_MAX_BYTES = 5_000_000
_LOG_BACKUP_COUNT = 3


def configure_utc_formatter():
    """Return the shared formatter, set to render timestamps in UTC.

    Used by both the pipeline logger below and ``bot.py``'s, so a line from
    ``/logs`` and a line from ``journalctl`` look the same and match the UTC
    timestamps already used in ``store.py``.
    """
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    formatter.converter = time.gmtime
    return formatter


logger = logging.getLogger("piscribe")
logger.setLevel(logging.INFO)
logger.propagate = False

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(configure_utc_formatter())
logger.addHandler(_stream_handler)

_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT, encoding="utf-8"
)
_file_handler.setFormatter(configure_utc_formatter())
logger.addHandler(_file_handler)


@contextmanager
def run_log(path):
    """Also mirror every log line into ``path`` for the duration of the block.

    Used by ``run_pipeline`` so each run gets its own retrievable log file on
    top of the shared, rotated ``LOG_FILE``.
    """
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(configure_utc_formatter())
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def log(msg):
    """Log an informational message (the common case)."""
    logger.info(msg)


def log_warning(msg):
    """Log a message for a recovered problem (e.g. a fallback was used)."""
    logger.warning(msg)


def log_error(msg):
    """Log a message for a failure (e.g. a file or a run did not complete)."""
    logger.error(msg)
