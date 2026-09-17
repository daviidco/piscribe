"""Tests for bot.py's Application wiring."""

import bot


def test_build_app_enables_concurrent_updates():
    """Without this, python-telegram-bot processes commands one at a time —
    the next command isn't even dequeued until the current handler's
    coroutine fully returns. /run et al. await a spawned pipeline.py that
    can run for minutes, so without concurrent updates every other command
    — including /cancel, exactly when it's most needed — is unresponsive
    for that whole time. (The library's default maps to an internal value
    of 1; concurrent_updates(True) raises it well past that.)
    """
    app = bot.build_app()

    assert app.concurrent_updates > 1
