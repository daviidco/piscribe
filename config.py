"""Runtime paths and environment configuration for the piscribe pipeline.

Most filesystem locations live under ``~/whisper.cpp/`` so the pipeline shares a
directory with the ``whisper.cpp`` build it depends on. Secrets and deployment
specific values are read from the environment (loaded from a local ``.env`` file
via ``python-dotenv``); a missing required variable raises ``KeyError`` at import
time so misconfiguration fails fast. Infrastructure paths (DB, run logs, lock)
have sensible defaults and only need an env override in unusual setups.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_DIR = Path(__file__).resolve().parent

HOME = Path.home()
WHISPER_DIR = HOME / "whisper.cpp"
LOCAL_DIR = WHISPER_DIR / "local_pending"
TRANSCRIPTIONS_DIR = WHISPER_DIR / "transcriptions"
WHISPER_MODEL = WHISPER_DIR / "models" / "ggml-small.bin"
WHISPER_CLI = WHISPER_DIR / "build" / "bin" / "whisper-cli"
LOG_FILE = WHISPER_DIR / "log.txt"


def _path_env(name, default):
    """Return an env-configured path, falling back to ``default``."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# History store, per-run logs, the cross-process run lock, and the pause flag.
DB_PATH = _path_env("DB_PATH", WHISPER_DIR / "piscribe.db")
RUN_LOG_DIR = _path_env("RUN_LOG_DIR", WHISPER_DIR / "runs")
LOCK_PATH = _path_env("LOCK_PATH", WHISPER_DIR / "piscribe.lock")
PAUSE_FLAG = _path_env("PAUSE_FLAG", WHISPER_DIR / "piscribe.paused")
DEADMAN_HOURS = float(os.environ.get("DEADMAN_HOURS", "5"))

QWEN_MODEL = os.environ["QWEN_MODEL"]
RCLONE_REMOTE = os.environ["RCLONE_REMOTE"]
PENDING_FOLDER = os.environ["PENDING_FOLDER"]
PROCESSED_FOLDER = os.environ["PROCESSED_FOLDER"]
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT_IDS = [c.strip() for c in os.environ["TG_CHAT_IDS"].split(",") if c.strip()]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
TEXT_EXTENSIONS = {".txt", ".md"}
