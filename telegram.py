"""Delivery of summaries to Telegram chats through the Bot API."""

import re
import subprocess

from config import TG_CHAT_IDS, TG_TOKEN
from utils import log

# Telegram rejects a sendMessage text longer than 4096 UTF-16 code units; stay
# safely under that so long summaries are split instead of dropped.
TELEGRAM_MAX_CHARS = 4000


def _split_message(text, limit=TELEGRAM_MAX_CHARS):
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


def send_telegram_message(text):
    """Send a Markdown message to every configured Telegram chat.

    ``**bold**`` is rewritten to Telegram's ``*bold*`` before sending. Messages
    over the Telegram length limit are split into several parts. A failure for
    one recipient is logged as a warning and does not stop the others.

    Args:
        text: The message body. Sent with ``parse_mode=Markdown``.
    """
    tg_text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    for chat_id in TG_CHAT_IDS:
        for part in _split_message(tg_text):
            result = subprocess.run(
                ["curl", "-s", "-X", "POST", f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                 "-d", f"chat_id={chat_id}", "-d", "parse_mode=Markdown",
                 "--data-urlencode", f"text={part}"],
                capture_output=True, text=True, check=False
            )
            if '"ok":true' not in result.stdout:
                log(f"WARNING: possible error sending message to {chat_id}: {result.stdout}")
