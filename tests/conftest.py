"""Shared test setup.

The module-level code here runs before any test module imports ``config``, so it
points ``$HOME`` at a throwaway directory and fills in the environment variables
``config.py`` requires. Every path the pipeline computes therefore lands inside
the sandbox instead of the real ``~/whisper.cpp``.
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
(_SANDBOX / "whisper.cpp").mkdir(parents=True, exist_ok=True)


def pytest_unconfigure(config):
    """Remove the sandbox once the whole test session is done."""
    shutil.rmtree(_SANDBOX, ignore_errors=True)


@pytest.fixture
def work_dirs():
    """Create ``LOCAL_DIR`` / ``TRANSCRIPTIONS_DIR`` for one test, wipe them after.

    Yields the ``config`` module so tests can reach the sandbox paths.
    """
    import config

    for directory in (config.LOCAL_DIR, config.TRANSCRIPTIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    yield config
    for directory in (config.LOCAL_DIR, config.TRANSCRIPTIONS_DIR):
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def read_log():
    """Return a callable that reads the current log file contents."""
    import config

    def _read():
        try:
            return config.LOG_FILE.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    return _read
