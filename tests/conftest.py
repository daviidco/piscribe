"""Shared test setup.

The environment bootstrap below runs before any test module imports ``config``:
it points ``$HOME`` at a throwaway directory and fills in the variables
``config.py`` requires. Every pipeline path — runtime dirs, the SQLite store, the
run lock — therefore lands inside the sandbox instead of the real
``~/whisper.cpp``. ``config`` is imported straight after that bootstrap, hence
the deliberate non-top-level position.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

_SANDBOX = Path(tempfile.mkdtemp(prefix="piscribe-tests-"))
os.environ["HOME"] = str(_SANDBOX)
os.environ.setdefault("RCLONE_REMOTE", "gdrive")
os.environ.setdefault("PENDING_FOLDER", "pendings")
os.environ.setdefault("PROCESSED_FOLDER", "processed")
os.environ.setdefault("QWEN_MODEL", "test-qwen")
os.environ.setdefault("TG_TOKEN", "test-token")
os.environ.setdefault("TG_CHAT_IDS", "1,2")
# Force-disabled by default: without this, config.py's load_dotenv() would pick
# up the real GROQ_API_KEY from the repo's own .env (load_dotenv doesn't
# override already-set vars, so setting it here — even to "" — wins first).
os.environ["GROQ_API_KEY"] = ""
(_SANDBOX / "whisper.cpp").mkdir(parents=True, exist_ok=True)

import config  # noqa: E402  pylint: disable=wrong-import-position
import telegram_api  # noqa: E402  pylint: disable=wrong-import-position


def pytest_unconfigure():
    """Remove the sandbox once the whole test session is done."""
    shutil.rmtree(_SANDBOX, ignore_errors=True)


@pytest.fixture(autouse=True)
def _no_real_telegram_calls(monkeypatch):
    """Never let a test reach the real Telegram API, even indirectly.

    ``pipeline.run_pipeline`` now sends progress messages on every trigger
    (cron and manual), so any test that exercises it would otherwise shell
    out to a real ``curl`` against api.telegram.org with a fake token. This
    patches the one low-level chokepoint (``telegram_api._post``) so that
    happens nowhere, regardless of which module's ``send_telegram_message``
    binding gets called. A test that wants to inspect what was sent still
    monkeypatches ``send_telegram_message`` itself, which takes precedence.
    """
    monkeypatch.setattr(telegram_api, "_post", lambda method, fields: None)


@pytest.fixture(autouse=True)
def clean_state():
    """Wipe the store, run lock, pause flag and run logs before every test.

    ``config.LOG_FILE`` is truncated rather than unlinked: utils.py's
    RotatingFileHandler opens it once at import time and keeps that file
    descriptor for the whole test session, so deleting the path would leave
    it writing into an unlinked inode that ``read_log()`` can never see again.
    """
    for path in (config.DB_PATH, config.LOCK_PATH, config.PAUSE_FLAG,
                 config.DB_PATH.with_suffix(".db-wal"),
                 config.DB_PATH.with_suffix(".db-shm")):
        Path(path).unlink(missing_ok=True)
    if config.LOG_FILE.exists():
        config.LOG_FILE.write_text("", encoding="utf-8")
    shutil.rmtree(config.RUN_LOG_DIR, ignore_errors=True)
    yield


@pytest.fixture
def work_dirs():
    """Create the runtime dirs for one test, wipe them after.

    Yields the ``config`` module so tests can reach the sandbox paths.
    """
    for directory in (config.LOCAL_DIR, config.TRANSCRIPTIONS_DIR, config.RUN_LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    yield config
    for directory in (config.LOCAL_DIR, config.TRANSCRIPTIONS_DIR, config.RUN_LOG_DIR):
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def read_log():
    """Return a callable that reads the current shared log file contents."""

    def _read():
        """Read the log file, returning an empty string if it does not exist yet."""
        try:
            return config.LOG_FILE.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    return _read
