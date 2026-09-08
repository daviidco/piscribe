"""Outbound Telegram Bot API helpers used by the pipeline.

Kept dependency-free (shells out to ``curl``) and separate from ``bot.py``,
which owns the inbound long-polling side with ``python-telegram-bot``. Named
``telegram_api`` rather than ``telegram`` to avoid shadowing that package.
"""

import re
import subprocess

from config import TG_CHAT_IDS, TG_TOKEN
from utils import log

# Telegram rejects a sendMessage text longer than 4096 UTF-16 code units; stay
# safely under that so long summaries are split instead of dropped.
TELEGRAM_MAX_CHARS = 4000


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


def _post(method, fields):
    """POST form ``fields`` to a Bot API ``method`` for every configured chat."""
    for chat_id in TG_CHAT_IDS:
        args = ["curl", "-s", "-X", "POST",
                f"https://api.telegram.org/bot{TG_TOKEN}/{method}",
                "-F", f"chat_id={chat_id}"]
        for key, value in fields:
            args += ["-F", f"{key}={value}"]
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if '"ok":true' not in result.stdout:
            log(f"WARNING: possible error calling {method} for {chat_id}: {result.stdout}")


def send_telegram_message(text):
    """Send a Markdown message to every configured Telegram chat.

    ``**bold**`` is rewritten to Telegram's ``*bold*`` before sending. Messages
    over the Telegram length limit are split into several parts.

    Args:
        text: The message body. Sent with ``parse_mode=Markdown``.
    """
    tg_text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    for part in split_message(tg_text):
        _post("sendMessage", [("parse_mode", "Markdown"), ("text", part)])


def send_document(path, caption=""):
    """Upload a local file to every configured Telegram chat.

    Args:
        path: Path to the file to send.
        caption: Optional caption shown with the document.
    """
    _post("sendDocument", [("document", f"@{path}"), ("caption", caption)])
