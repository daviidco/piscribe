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


def _read_version():
    """App version from the ``VERSION`` file at the repo root (see CHANGELOG.md)."""
    try:
        return (REPO_DIR / "VERSION").read_text(encoding="utf-8").strip() or "0.0.0"
    except FileNotFoundError:
        return "0.0.0"


VERSION = _read_version()

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
# Per-run log files older than this are pruned at the start of each run (see
# pipeline._prune_old_run_logs); the shared LOG_FILE rotates by size instead
# (see utils.py) and DB_PATH is not pruned at all.
RUN_LOG_RETENTION_DAYS = int(os.environ.get("RUN_LOG_RETENTION_DAYS", "30"))

QWEN_MODEL = os.environ["QWEN_MODEL"]
RCLONE_REMOTE = os.environ["RCLONE_REMOTE"]
PENDING_FOLDER = os.environ["PENDING_FOLDER"]
PROCESSED_FOLDER = os.environ["PROCESSED_FOLDER"]
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT_IDS = [c.strip() for c in os.environ["TG_CHAT_IDS"].split(",") if c.strip()]

# Optional cloud-first transcription/summary via Groq; empty key disables it and
# every run falls back to the local whisper.cpp / Ollama path (see video.py,
# summary.py). Not required at import time, unlike the variables above.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3")
GROQ_TIMEOUT_SECONDS = float(os.environ.get("GROQ_TIMEOUT_SECONDS", "20"))
GROQ_AUDIO_TIMEOUT_SECONDS = float(os.environ.get("GROQ_AUDIO_TIMEOUT_SECONDS", "60"))
GROQ_MAX_AUDIO_MB = float(os.environ.get("GROQ_MAX_AUDIO_MB", "24"))
# The detailed prompt can need a long answer for a dense, multi-participant
# meeting; a response cut short by this cap is treated as a failure and falls
# back to local Ollama rather than delivered incomplete (see summary.py).
GROQ_MAX_COMPLETION_TOKENS = int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", "8192"))
# When Groq rejects a request for exceeding its free-tier OTPM cap
# (RateLimitError — see summary.py's generate_summary), the transcript is
# retried split into chunks of this size instead of falling back to local
# right away: each chunk needs far less output than one request for the
# whole transcript would (see summary.py's _summarize_groq_chunked for the
# actual OTPM trade-off this does and doesn't solve).
GROQ_CHUNK_CHARS = int(os.environ.get("GROQ_CHUNK_CHARS", "6000"))
GROQ_CHUNK_MAX_COMPLETION_TOKENS = int(os.environ.get("GROQ_CHUNK_MAX_COMPLETION_TOKENS", "500"))
# Each chunk after the first repeats this many characters from the end of the
# previous one, so a point made right at a cut isn't lost to whichever side
# didn't get it (see summary.py's _split_into_chunks).
GROQ_CHUNK_OVERLAP_CHARS = int(os.environ.get("GROQ_CHUNK_OVERLAP_CHARS", "300"))

# RAG (/ask): Groq has no embeddings API, so indexing and question-embedding
# always run through local Ollama (embeddings.py) regardless of GROQ_API_KEY —
# only the answer itself (rag.py) gets the Groq-first/local-fallback treatment.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
# Below this cosine similarity, /ask answers "no encontré información" without
# calling any LLM — the retrieved chunks aren't relevant enough to ground an
# answer, and guessing anyway risks the model inventing one (see rag.py).
RAG_MIN_SIMILARITY = float(os.environ.get("RAG_MIN_SIMILARITY", "0.35"))
RAG_CHUNK_CHARS = int(os.environ.get("RAG_CHUNK_CHARS", "1200"))
# Same rationale as GROQ_CHUNK_OVERLAP_CHARS: a chunk boundary shouldn't cut a
# sentence's meaning in half for retrieval purposes.
RAG_CHUNK_OVERLAP_CHARS = int(os.environ.get("RAG_CHUNK_OVERLAP_CHARS", "150"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
TEXT_EXTENSIONS = {".txt", ".md"}
