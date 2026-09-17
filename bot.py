#!/usr/bin/env python3
"""piscribe Telegram control bot.

A persistent long-polling process (run under systemd) that answers status /
history / recap / transcript / logs / stats / find queries and can trigger work
with ``/run``, ``/retry``, ``/resummarize``, ``/cancel`` and ``/pause``. It never
runs the pipeline in process — those spawn ``pipeline.py`` just like cron — and
it only reads the SQLite store that ``pipeline.py`` writes.

Long polling means there is no inbound port: the process holds an open
``getUpdates`` request to Telegram that returns as soon as a command arrives.
"""

import logging

from telegram.ext import Application, CommandHandler

import handlers
import store
from config import TG_TOKEN
from utils import LOG_DATEFMT, LOG_FORMAT, configure_utc_formatter

HEALTH_INTERVAL_SECONDS = 900
# These log at INFO by default and would otherwise drown out real bot events
# (every long-poll and every sendMessage) in `journalctl`.
_QUIET_LOGGERS = ("httpx", "httpcore", "apscheduler")


def build_app():
    """Wire up the command handlers, the health check, and the startup ping.

    ``concurrent_updates(True)``: python-telegram-bot's default
    (``False``) processes updates one at a time — the next command isn't
    even picked off the queue until the current handler's coroutine fully
    returns. ``/run``/``/runcron``/``/retry``/``/resummarize`` await a
    spawned ``pipeline.py`` that can run for minutes, so without this,
    *every other command — including `/cancel` — is unresponsive for that
    whole time*, exactly when `/cancel` is most likely to be needed. Running
    updates concurrently fixes that; the risk of two `/run`-family commands
    racing into `_spawn_and_report`'s PID check is already covered by
    ``pipeline.py``'s own cross-process file lock (``runlock.py``), which
    lets a second process notice the lock is held and exit quietly.
    """
    store.init_db()
    app = (
        Application.builder()
        .token(TG_TOKEN)
        .concurrent_updates(True)
        .post_init(handlers.post_init)
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], handlers.help_))
    app.add_handler(CommandHandler("whoami", handlers.whoami))
    app.add_handler(CommandHandler("version", handlers.version_cmd))
    app.add_handler(CommandHandler("status", handlers.status))
    app.add_handler(CommandHandler(["recap", "last"], handlers.recap))
    app.add_handler(CommandHandler("transcript", handlers.transcript))
    app.add_handler(CommandHandler("logs", handlers.logs))
    app.add_handler(CommandHandler("history", handlers.history))
    app.add_handler(CommandHandler("pending", handlers.pending))
    app.add_handler(CommandHandler("stats", handlers.stats))
    app.add_handler(CommandHandler("find", handlers.find))
    app.add_handler(CommandHandler("run", handlers.run))
    app.add_handler(CommandHandler("runcron", handlers.runcron))
    app.add_handler(CommandHandler("retry", handlers.retry))
    app.add_handler(CommandHandler("resummarize", handlers.resummarize))
    app.add_handler(CommandHandler("cancel", handlers.cancel))
    app.add_handler(CommandHandler("pause", handlers.pause))
    app.add_handler(CommandHandler("resume", handlers.resume))
    app.add_error_handler(handlers.on_error)
    app.job_queue.run_repeating(
        handlers.deadman_check, interval=HEALTH_INTERVAL_SECONDS, first=HEALTH_INTERVAL_SECONDS
    )
    return app


def _configure_logging():
    """Set up logging so a bot line reads like a pipeline line (same UTC format).

    Third-party libraries that log routine traffic at INFO (every long-poll,
    every outgoing Telegram request) are quieted to WARNING so real bot events
    (unauthorized commands, job failures) aren't buried in `journalctl`.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(configure_utc_formatter())
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT,
                         handlers=[handler])
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def main():
    """Configure logging and start long polling (blocks until stopped)."""
    _configure_logging()
    build_app().run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
