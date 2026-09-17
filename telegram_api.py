"""Outbound Telegram Bot API helpers used by the pipeline.

Separate from ``bot.py``, which owns the inbound long-polling side with
``python-telegram-bot``. Named ``telegram_api`` rather than ``telegram`` to
avoid shadowing that package.

Uses ``httpx`` directly rather than ``python-telegram-bot``'s own (async)
``Bot`` object: this module is called from ``pipeline.py``, a plain
synchronous script invoked fresh by cron or on demand, not the long-lived
async bot process. An earlier version of this module shelled out to ``curl``
instead of using an HTTP library at all — that hand-rolled approach was the
source of a real bug: curl's multipart ``-F`` syntax treats a bare ``;``
inside a field's value as the start of an extra parameter clause
(``;type=...``), silently truncating the field right there, which ordinary
LLM-written prose (summaries routinely contain semicolons) hit constantly.
``httpx`` encodes form/multipart bodies correctly by construction.
"""

import re
from pathlib import Path

import httpx

import config
from config import TG_CHAT_IDS, TG_TOKEN
from utils import log_warning

# Telegram rejects a sendMessage text longer than 4096 UTF-16 code units; stay
# safely under that so long summaries are split instead of dropped.
TELEGRAM_MAX_CHARS = 4000

# Module-level and reused across calls: cheap connection reuse for the several
# messages one pipeline.py run can send (stage pings, the summary, its
# signature), and for bot.py's entire process lifetime if it ever calls in here.
_client = httpx.Client(timeout=10.0)


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


def to_telegram_markdown(text):
    """Redact secrets, then rewrite ``**bold**`` to Telegram's own ``*bold*``.

    Shared by :func:`send_telegram_message` and any bot command (``/recap``)
    that wants its reply to render the same way the pipeline's original
    delivery does — send the result with ``parse_mode="Markdown"``; sending
    it as plain text (the default for a bot's own ``reply_text``) would show
    the raw ``_..._``/``*...*`` markers literally instead of rendering them.
    """
    return re.sub(r'\*\*(.+?)\*\*', r'*\1*', redact(text))


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


def _post(method, data, chat_ids, files=None):
    """POST ``data`` (and optional ``files``) to a Bot API ``method`` for each
    of ``chat_ids``.

    A failure (network error, non-2xx response, or a JSON body without
    ``"ok": true``) is only logged, never raised — a failed notification
    shouldn't crash a pipeline run.

    Args:
        method: Bot API method name, e.g. ``"sendMessage"``.
        data: Form fields as a dict (``chat_id`` is added automatically).
        chat_ids: Chat ids to send this to, one request per id.
        files: Optional ``httpx``-style files dict for a multipart upload,
            e.g. ``{"document": (filename, bytes_content)}`` — see
            :func:`send_document`. Passing bytes rather than an open file
            handle matters here: a handle would be exhausted after the first
            of possibly several chat ids.
    """
    url = f"https://api.telegram.org/bot{TG_TOKEN}/{method}"
    for chat_id in chat_ids:
        payload = {"chat_id": chat_id, **data}
        try:
            response = _client.post(url, data=payload, files=files)
            ok = response.is_success and response.json().get("ok")
        except httpx.HTTPError as e:
            ok, response = False, e
        if not ok:
            log_warning(f"possible error calling {method} for {chat_id}: {response}")


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
    tg_text = to_telegram_markdown(text)
    targets = TG_CHAT_IDS if chat_ids is None else chat_ids
    for part in split_message(tg_text):
        _post("sendMessage", {"parse_mode": "Markdown", "text": part}, targets)


def send_document(path, caption="", chat_ids=None):
    """Upload a local file to ``chat_ids`` (default: every configured chat).

    Args:
        path: Path to the file to send.
        caption: Optional caption shown with the document.
        chat_ids: Chat ids to send to; ``None`` broadcasts to every configured chat.
    """
    targets = TG_CHAT_IDS if chat_ids is None else chat_ids
    content = Path(path).read_bytes()
    _post(
        "sendDocument", {"caption": caption}, targets,
        files={"document": (Path(path).name, content)},
    )
