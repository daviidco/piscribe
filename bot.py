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

HEALTH_INTERVAL_SECONDS = 900


def build_app():
    """Wire up the command handlers, the health check, and the startup ping."""
    store.init_db()
    app = (
        Application.builder()
        .token(TG_TOKEN)
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


def main():
    """Configure logging and start long polling (blocks until stopped)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    build_app().run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
