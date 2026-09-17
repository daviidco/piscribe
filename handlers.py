"""Command handlers for the piscribe Telegram control bot.

Every command except ``/whoami`` is restricted to the ids in ``TG_CHAT_IDS``
(unauthorized updates are logged and ignored). Handlers never run the pipeline
in process: ``/run``, ``/retry`` and ``/resummarize`` spawn ``pipeline.py`` with
``PISCRIBE_*`` environment variables, exactly like cron.
"""

# python-telegram-bot calls every handler as ``(update, context)`` whether or
# not the handler needs both.
# pylint: disable=unused-argument

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import config
import store
from drive import list_pending_files
from runlock import current_run_pid
from telegram_api import redact, split_message, to_telegram_markdown

log = logging.getLogger("piscribe.bot")

# Spanish labels for the raw run/file status values stored in the DB, so they
# never leak into an outbound message as bare English words.
_STATUS_LABELS = {
    "ok": "completado",
    "partial": "parcial",
    "error": "error",
    "cancelled": "cancelado",
    "skipped": "omitido",
    "running": "en curso",
}


def _status_label(raw_status):
    """Spanish display label for a raw run/file status value."""
    return _STATUS_LABELS.get(raw_status, raw_status)


# Matches utils.LOG_FORMAT's "[<timestamp>] LEVEL message" — anchored to the
# level field itself, not a loose substring, so a normal message that happens
# to contain the word "error" doesn't false-positive in /logs errors.
_LOG_LEVEL_RE = re.compile(r"^\[[^\]]*\]\s+(WARNING|ERROR)\b")


HELP_TEXT = (
    "Piscribe Control bot\n\n"
    "«#id» es siempre un id real de la base — nunca una posición.\n"
    "El de una corrida sale de /history; el de un archivo, de /find.\n"
    "«n» en /history es la cantidad de corridas a mostrar, no un id.\n\n"
    "consulta:\n"
    "  /status — estado y último run\n"
    "  /recap [#id|archivo] — resumen\n"
    "  /transcript [#id|archivo] — transcripción\n"
    "  /logs [#id|errors] — log de la corrida\n"
    "  /history [n] — las últimas n corridas\n"
    "  /pending — archivos en la carpeta de Drive\n"
    "  /stats — métricas de los últimos 7 días\n"
    "  /find <texto> — buscar en resúmenes/archivos\n"
    "  /version — versión y modelo\n"
    "  /whoami — tu id de Telegram\n\n"
    "control:\n"
    "  /run [archivo] — ejecutar ahora (avisos solo para vos)\n"
    "  /runcron [archivo] — igual que /run, pero avisa a todo el equipo\n"
    "  /retry [#id|archivo] — reprocesar desde 'processed'\n"
    "  /resummarize [#id|archivo] — rehacer solo el resumen\n"
    "  /cancel — detener el run en curso\n"
    "  /pause /resume — pausar/reactivar los runs"
)


def version():
    """Short git SHA of the checkout, or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(config.REPO_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def authorized(func):
    """Ignore (and log) commands from ids outside ``TG_CHAT_IDS``."""

    @wraps(func)
    async def wrapper(update, context):
        user = update.effective_user
        if user is None or str(user.id) not in config.TG_CHAT_IDS:
            log.warning("Unauthorized /%s from id=%s", func.__name__, getattr(user, "id", "?"))
            return None
        return await func(update, context)

    return wrapper


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_exit(pid, seconds):
    """Block up to ``seconds`` for ``pid`` to disappear."""
    for _ in range(seconds * 10):
        if not _pid_alive(pid):
            return
        time.sleep(0.1)


def resolve_file(arg):
    """Map a command arg (None -> latest, digit -> that file's id, else
    filename) to a row. A file's id is the ``#N`` shown by ``/find``."""
    if not arg:
        return store.latest_file()
    if arg.isdigit():
        return store.file_by_id(int(arg))
    return store.file_by_name(arg)


def _target_filename(context):
    """Resolve a /retry|/resummarize argument to a filename (arg, a file id,
    or the latest file)."""
    arg = context.args[0] if context.args else None
    if arg and not arg.isdigit():
        return arg
    row = resolve_file(arg)
    return row["filename"] if row else None


async def _spawn_and_report(update, env_extra, announce=None, *, own_messages=False):
    """Spawn pipeline.py with ``env_extra``, then post the result.

    Defaults to ``PISCRIBE_TRIGGER=manual``; ``env_extra`` is merged in last,
    so ``/runcron`` overriding it to ``"cron"`` (to broadcast instead of
    notifying only the requester) works without any special-casing here.

    ``own_messages=True`` (used by ``/run``) skips both the pre-announce and
    the post-run report: ``run_pipeline`` posts the same start/stage/failure/
    end messages a cron pass would, so the bot adding its own here would just
    duplicate them. ``/retry`` and ``/resummarize`` go through
    ``_single_file_run`` instead, which has no such messages yet, so they
    still pass an ``announce`` and rely on the report below.
    """
    if _pid_alive(current_run_pid()):
        last = store.last_run()
        trg = last["trigger"] if last else "?"
        await update.message.reply_text(f"⏳ run en curso ({trg}), probá en un rato.")
        return

    if announce:
        await update.message.reply_text(announce)
    env = {
        **os.environ,
        "PISCRIBE_TRIGGER": "manual",
        "PISCRIBE_BY": str(update.effective_user.id),
        **env_extra,
    }
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, same command as cron
        [sys.executable, str(config.REPO_DIR / "pipeline.py")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True, env=env,
    )
    await asyncio.get_running_loop().run_in_executor(None, proc.wait)

    if own_messages:
        return

    last = store.last_run()
    if last and last["status"] != "running":
        icon = "✅" if last["status"] == "ok" else "⚠️"
        msg = (
            f"{icon} run #{last['id']}: {_status_label(last['status'])} "
            f"({last['files_ok']}/{last['files_total']})"
        )
        if last["error"]:
            msg += f"\n{redact(last['error'])}"
    else:
        msg = f"pipeline terminó (rc={proc.returncode})."
    await update.message.reply_text(msg)


# --- query commands --------------------------------------------------

async def help_(update, context):
    """/help and /start — list the commands (allowlisted ids only)."""
    user = update.effective_user
    if user is None or str(user.id) not in config.TG_CHAT_IDS:
        return
    await update.message.reply_text(HELP_TEXT)


async def whoami(update, context):
    """/whoami — echo the caller's ids (unauthenticated, for allowlist setup)."""
    user = update.effective_user
    chat = update.effective_chat
    await update.message.reply_text(
        f"id de usuario: {user.id}\nid de chat: {chat.id}\nusuario: @{user.username}"
        if user else "no hay usuario en este update"
    )


@authorized
async def version_cmd(update, context):
    """/version — app version, checkout SHA, and every configured model (cloud + local)."""
    lines = [
        f"piscribe v{config.VERSION} ({version()})",
        f"resumen local: Ollama {config.QWEN_MODEL}",
    ]
    if config.GROQ_API_KEY:
        lines.append(f"resumen: Groq · {config.GROQ_MODEL}")
        lines.append(f"transcripción: Groq · {config.GROQ_WHISPER_MODEL}")
    else:
        lines.append("Groq: desactivado (sin GROQ_API_KEY)")
    await update.message.reply_text("\n".join(lines))


@authorized
async def status(update, context):
    """/status — run in progress, pause state, last run, last success, free disk."""
    lines = []
    if config.PAUSE_FLAG.exists():
        lines.append("⏸️ EN PAUSA")
    pid = current_run_pid()
    lines.append(f"🟢 run en curso (pid {pid})" if pid else "⚪ sin run en curso")
    last = store.last_run()
    if last:
        lines.append(
            f"último: #{last['id']} {last['trigger']} → {last['status']} "
            f"({last['files_ok']}/{last['files_total']}) {last['started_at']}"
        )
        if last["error"]:
            lines.append(f"  error: {redact(last['error'])}")
    lines.append(f"último OK: {store.last_ok_at() or 'nunca'}")
    try:
        free_mb = shutil.disk_usage(config.WHISPER_DIR).free // (1024 * 1024)
        lines.append(f"disco libre: {free_mb} MB")
    except OSError:
        pass
    await update.message.reply_text("\n".join(lines))


@authorized
async def recap(update, context):
    """/recap [#id|archivo] — the summary of a processed file, signed with its engine(s)."""
    row = resolve_file(context.args[0] if context.args else None)
    if not row:
        await update.message.reply_text("no encuentro ese archivo.")
        return
    if not row["summary"]:
        await update.message.reply_text(
            f"{row['filename']}: sin resumen (status {row['status']})."
        )
        return
    summary_text = to_telegram_markdown(f"📋 {row['filename']}\n\n{row['summary']}")
    for part in split_message(summary_text):
        await update.message.reply_text(part, parse_mode="Markdown")

    # Sent as its own message, after the summary — same reasoning as the
    # pipeline's original delivery (see pipeline._signature_message): a long
    # or malformed summary can't take the signature down with it.
    sig_lines = []
    if row["transcribe_backend"]:
        sig_lines.append(f"_transcripción: {row['transcribe_backend']}_")
    if row["summarize_backend"]:
        sig_lines.append(f"_resumen: {row['summarize_backend']}_")
    if sig_lines:
        await update.message.reply_text(
            to_telegram_markdown("\n".join(sig_lines)), parse_mode="Markdown"
        )


@authorized
async def transcript(update, context):
    """/transcript [#id|archivo] — the original transcript, as a document."""
    row = resolve_file(context.args[0] if context.args else None)
    if not row:
        await update.message.reply_text("no encuentro ese archivo.")
        return
    path = row["transcript_path"]
    if not path or not Path(path).is_file():
        await update.message.reply_text(f"{row['filename']}: no hay transcripción guardada.")
        return
    with open(path, "rb") as handle:
        await update.message.reply_document(
            handle, filename=Path(path).name, caption=f"transcripción: {row['filename']}"
        )


@authorized
async def logs(update, context):
    """/logs [id|errors] — a run's log file by its id (see /history), or the
    latest run's if no id is given, or just its error/warning lines."""
    arg = context.args[0].lower() if context.args else None
    only_errors = arg == "errors"
    row = store.run_by_id(int(arg)) if (arg and arg.isdigit()) else store.last_run()
    if not row or not row["log_path"]:
        await update.message.reply_text("no hay log para esa ejecución.")
        return
    path = Path(row["log_path"])
    if not path.is_file():
        await update.message.reply_text("el archivo de log ya no existe.")
        return
    text = redact(path.read_text(encoding="utf-8", errors="replace"))
    if only_errors:
        hits = [ln for ln in text.splitlines() if _LOG_LEVEL_RE.match(ln)]
        for part in split_message("\n".join(hits) or "(sin errores ni advertencias)"):
            await update.message.reply_text(part)
        return
    buf = io.BytesIO(text.encode("utf-8"))
    buf.name = path.name
    await update.message.reply_document(
        buf, filename=path.name, caption=f"log run #{row['id']} ({row['trigger']})"
    )


@authorized
async def history(update, context):
    """/history [n] — a compact list of recent runs."""
    n = int(context.args[0]) if (context.args and context.args[0].isdigit()) else 10
    rows = store.recent_runs(min(n, 30))
    if not rows:
        await update.message.reply_text("sin ejecuciones registradas.")
        return
    lines = [
        f"#{r['id']} {r['started_at']} {r['trigger']:<6} {r['status']:<9} "
        f"{r['files_ok']}/{r['files_total']}"
        for r in rows
    ]
    await update.message.reply_text("\n".join(lines))


@authorized
async def pending(update, context):
    """/pending — list what is waiting in the Drive pending folder."""
    try:
        files = await asyncio.get_running_loop().run_in_executor(None, list_pending_files)
    except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
        await update.message.reply_text(f"error listando: {redact(str(e))}")
        return
    await update.message.reply_text(
        "pendientes:\n" + ("\n".join(files) if files else "(vacío)")
    )


@authorized
async def stats(update, context):
    """/stats — rolling counts from the run history."""
    data = store.stats()
    avg = f"{data['avg_ms_7d'] / 1000:.1f}s" if data["avg_ms_7d"] else "—"
    await update.message.reply_text(
        f"archivos 24h: {data['files_24h']}\n"
        f"archivos 7d: {data['files_7d']} (errores: {data['errors_7d']})\n"
        f"runs 7d: {data['runs_7d']}\n"
        f"duración media (ok, 7d): {avg}"
    )


@authorized
async def find(update, context):
    """/find <texto> — search filenames and summaries."""
    if not context.args:
        await update.message.reply_text("uso: /find <texto>")
        return
    term = " ".join(context.args)
    rows = store.search(term)
    if not rows:
        await update.message.reply_text(f"sin resultados para «{term}».")
        return
    lines = [
        f"#{r['id']} {r['filename']} — {(r['summary'] or '').replace(chr(10), ' ')[:80]}"
        for r in rows
    ]
    await update.message.reply_text("\n".join(lines))


# --- control commands --------------------------------------------------

@authorized
async def run(update, context):
    """/run [archivo] — process the pending folder now (or one pending file).

    No bot-side announce/report here: pipeline.py posts the same
    start/stage/failure/end messages a cron pass would. Every message from
    this run goes ONLY to whoever asked (see pipeline._targets) — use
    /runcron instead if the result should reach the whole team.
    """
    extra = {"PISCRIBE_ONLY": context.args[0]} if context.args else {}
    await _spawn_and_report(update, extra, own_messages=True)


@authorized
async def runcron(update, context):
    """/runcron [archivo] — like /run, but broadcasts to every configured
    chat (TG_CHAT_IDS) instead of only to whoever asked.

    Forces ``PISCRIBE_TRIGGER=cron`` so pipeline._targets treats this exactly
    like an automated cron pass for audience purposes (see also /history,
    where it shows up indistinguishable from a real cron run — that's the
    point). ``PISCRIBE_BY`` is still recorded, so the run's requester is not
    lost from the store even though the trigger reads "cron".
    """
    extra = {"PISCRIBE_TRIGGER": "cron"}
    if context.args:
        extra["PISCRIBE_ONLY"] = context.args[0]
    await _spawn_and_report(update, extra, own_messages=True)


@authorized
async def retry(update, context):
    """/retry [#id|archivo] — reprocess a file from the processed folder."""
    name = _target_filename(context)
    if not name:
        await update.message.reply_text("no encuentro ese archivo.")
        return
    await _spawn_and_report(
        update, {"PISCRIBE_MODE": "retry", "PISCRIBE_ONLY": name},
        f"🔁 reprocesando {name}…",
    )


@authorized
async def resummarize(update, context):
    """/resummarize [#id|archivo] — re-run only the summary over a stored transcript."""
    name = _target_filename(context)
    if not name:
        await update.message.reply_text("no encuentro ese archivo.")
        return
    await _spawn_and_report(
        update, {"PISCRIBE_MODE": "resummarize", "PISCRIBE_ONLY": name},
        f"📝 rehaciendo el resumen de {name}…",
    )


@authorized
async def cancel(update, context):
    """/cancel — stop the pipeline run in progress (cron or manual)."""
    pid = current_run_pid()
    if not pid:
        await update.message.reply_text("no hay run en curso.")
        return
    await update.message.reply_text(f"🛑 deteniendo el run (pid {pid})…")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        await update.message.reply_text(f"no pude señalizar el proceso: {e}")
        return
    await asyncio.get_running_loop().run_in_executor(None, _wait_for_exit, pid, 10)
    if _pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
        await update.message.reply_text("no respondió a SIGTERM; SIGKILL enviado.")
    else:
        await update.message.reply_text("run detenido.")


@authorized
async def pause(update, context):
    """/pause — runs record as 'skipped' until /resume."""
    config.PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    config.PAUSE_FLAG.touch()
    await update.message.reply_text("⏸️ pipeline en pausa. Las corridas van a quedar omitidas.")


@authorized
async def resume(update, context):
    """/resume — clear the pause flag."""
    config.PAUSE_FLAG.unlink(missing_ok=True)
    await update.message.reply_text("▶️ pipeline reactivado.")


# --- background / lifecycle --------------------------------------------------

async def deadman_check(context):
    """Alert the chats when no run has succeeded for too long, and on recovery."""
    ok_at = store.last_ok_at()
    if not ok_at:
        return
    last_ok = datetime.fromisoformat(ok_at).replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - last_ok).total_seconds() / 3600

    if age_h > config.DEADMAN_HOURS:
        if not context.bot_data.get("deadman_alerted"):
            await _broadcast(context, f"⚠️ piscribe: sin ejecución OK hace {age_h:.1f} h.")
            context.bot_data["deadman_alerted"] = True
    elif context.bot_data.get("deadman_alerted"):
        await _broadcast(context, "✅ piscribe: recuperado, hubo un run OK.")
        context.bot_data["deadman_alerted"] = False


async def post_init(application):
    """Announce startup to the chats once the bot is up."""
    for chat_id in config.TG_CHAT_IDS:
        try:
            await application.bot.send_message(
                chat_id, f"🤖 piscribe-bot online (v{config.VERSION}, {version()})"
            )
        except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
            log.warning("startup ping to %s failed: %s", chat_id, e)


async def _broadcast(context, text):
    for chat_id in config.TG_CHAT_IDS:
        try:
            await context.bot.send_message(chat_id, text)
        except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
            log.warning("broadcast to %s failed: %s", chat_id, e)


async def on_error(update, context):
    """Reply with a redacted error instead of letting a handler crash the bot."""
    log.exception("handler error", exc_info=context.error)
    try:
        message = getattr(update, "effective_message", None)
        if message is not None:
            await message.reply_text(f"⚠️ comando falló: {redact(str(context.error))}")
    except Exception:  # noqa: BLE001  pylint: disable=broad-exception-caught
        pass  # never raise from the error handler
