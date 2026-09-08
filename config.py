"""Runtime paths and environment configuration for the piscribe pipeline.

All filesystem locations live under ``~/whisper.cpp/`` so the pipeline shares a
directory with the ``whisper.cpp`` build it depends on. Secrets and deployment
specific values are read from the environment (loaded from a local ``.env`` file
via ``python-dotenv``); a missing required variable raises ``KeyError`` at import
time so misconfiguration fails fast.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

HOME = Path.home()
WHISPER_DIR = HOME / "whisper.cpp"
LOCAL_DIR = WHISPER_DIR / "local_pending"
TRANSCRIPTIONS_DIR = WHISPER_DIR / "transcriptions"
WHISPER_MODEL = WHISPER_DIR / "models" / "ggml-small.bin"
WHISPER_CLI = WHISPER_DIR / "build" / "bin" / "whisper-cli"
LOG_FILE = WHISPER_DIR / "log.txt"

QWEN_MODEL = os.environ["QWEN_MODEL"]
RCLONE_REMOTE = os.environ["RCLONE_REMOTE"]
PENDING_FOLDER = os.environ["PENDING_FOLDER"]
PROCESSED_FOLDER = os.environ["PROCESSED_FOLDER"]
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT_IDS = os.environ["TG_CHAT_IDS"].split(",")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
TEXT_EXTENSIONS = {".txt", ".md"}
