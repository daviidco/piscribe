"""Outbound Telegram Bot API helpers used by the pipeline.

Kept dependency-free (shells out to ``curl``) and separate from ``bot.py``,
which owns the inbound long-polling side with ``python-telegram-bot``. Named
``telegram_api`` rather than ``telegram`` to avoid shadowing that package.
"""

import re
import subprocess

import config
from config import TG_CHAT_IDS, TG_TOKEN
from utils import log_warning

# Telegram rejects a sendMessage text longer than 4096 UTF-16 code units; stay
# safely under that so long summaries are split instead of dropped.
TELEGRAM_MAX_CHARS = 4000


def redact(text):
    """Blank out any configured secret if it ever appears in outbound text.

    Reads secrets off the ``config`` module (not as bound names) so tests can
    monkeypatch ``config.GROQ_API_KEY``/``config.TG_TOKEN`` and have it apply.
    """
    out = text or ""
    for secret in (config.TG_TOKEN, config.GROQ_API_KEY):
        if secret:
            out = out.replace(secret, "***")
    return out


def split_message(text, limit=TELEGRAM_MAX_CHARS):
    """Yield chunks of ``text`` no longer than ``limit``, breaking on newlines.

    Falls back to a hard character cut when a single line exceeds ``limit``.
    """
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        yield text[:cut]
        text = text[cut:].lstrip("\n")
    if text:
        yield text


def _post(method, fields, chat_ids):
    """POST form ``fields`` to a Bot API ``method`` for each of ``chat_ids``."""
    for chat_id in chat_ids:
        args = ["curl", "-s", "-X", "POST",
                f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
                "-F", f"chat_id={chat_id}"]
        for key, value in fields:
            args += ["-F", f"{key}={value}"]
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if '"ok":true' not in result.stdout:
            log_warning(f"possible error calling {method} for {chat_id}: {result.stdout}")


def send_telegram_message(text, chat_ids=None):
    """Send a Markdown message to ``chat_ids`` (default: every configured chat).

    ``**bold**`` is rewritten to Telegram's ``*bold*`` before sending. Messages
    over the Telegram length limit are split into several parts.

    Args:
        text: The message body. Sent with ``parse_mode=Markdown``.
        chat_ids: Chat ids to send to. ``None`` (the default) broadcasts to
            every chat in ``TG_CHAT_IDS`` — pass a narrower list (e.g. a
            single requester's id) to keep an on-demand message private.
    """
    tg_text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', redact(text))
    targets = TG_CHAT_IDS if chat_ids is None else chat_ids
    for part in split_message(tg_text):
        _post("sendMessage", [("parse_mode", "Markdown"), ("text", part)], targets)


def send_document(path, caption="", chat_ids=None):
    """Upload a local file to ``chat_ids`` (default: every configured chat).

    Args:
        path: Path to the file to send.
        caption: Optional caption shown with the document.
        chat_ids: Chat ids to send to; ``None`` broadcasts to every configured chat.
    """
    targets = TG_CHAT_IDS if chat_ids is None else chat_ids
    _post("sendDocument", [("document", f"@{path}"), ("caption", caption)], targets)
