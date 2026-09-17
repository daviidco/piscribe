"""Tests for the Telegram bot command handlers.

Only the parts that do not need a real ``python-telegram-bot`` runtime: the sync
helpers and the authorization gate, driven with hand-rolled fake updates.
"""

# Tests reach into the handlers module's own helpers, and use small fake classes.
# pylint: disable=protected-access,too-few-public-methods

import asyncio
from types import SimpleNamespace

import config
import handlers
import store


class _Msg:
    """Fake ``update.message`` that records replies instead of sending them."""

    def __init__(self):
        self.texts = []
        self.parse_modes = []
        self.documents = []

    async def reply_text(self, text, **kw):
        """Record a text reply, and the parse_mode it was sent with (if any)."""
        self.texts.append(text)
        self.parse_modes.append(kw.get("parse_mode"))

    async def reply_document(self, _doc, filename=None, caption=None, **_kw):
        """Record a document reply (filename + caption only)."""
        self.documents.append((filename, caption))


def _update(user_id):
    """Build a fake ``(update, message)`` pair for ``user_id``."""
    msg = _Msg()
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id, username="tester"),
        effective_chat=SimpleNamespace(id=user_id),
        effective_message=msg,
        message=msg,
    )
    return update, msg


def _ctx(args=None):
    """Build a fake handler context."""
    return SimpleNamespace(args=args or [], bot_data={}, bot=None, error=None)


def test_redact_blanks_the_token():
    """redact replaces the bot token and tolerates None."""
    assert handlers.redact(f"leaked {config.TG_TOKEN} here") == "leaked *** here"
    assert handlers.redact(None) == ""


def test_redact_blanks_the_groq_key_too(monkeypatch):
    """redact also strips GROQ_API_KEY when one is configured."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_secret123")
    assert handlers.redact("error: gsk_secret123 invalid") == "error: *** invalid"


def test_status_label_translates_known_values_and_passes_through_others():
    """_status_label maps DB status words to Spanish, unknown values pass through."""
    assert handlers._status_label("partial") == "parcial"
    assert handlers._status_label("cancelled") == "cancelado"
    assert handlers._status_label("skipped") == "omitido"
    assert handlers._status_label("weird") == "weird"


def test_whoami_uses_spanish_field_labels():
    """/whoami answers in Spanish, not with English field names."""
    upd, msg = _update(42)
    asyncio.run(handlers.whoami(upd, _ctx()))
    assert "id de usuario: 42" in msg.texts[0]
    assert "user id" not in msg.texts[0]


def test_version_reports_groq_status(monkeypatch):
    """/version shows whether Groq is configured, and which models are in play."""
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    upd, msg = _update(_uid())
    asyncio.run(handlers.version_cmd(upd, _ctx()))
    assert any("desactivado" in t for t in msg.texts)

    monkeypatch.setattr(config, "GROQ_API_KEY", "fake-key")
    msg.texts.clear()
    asyncio.run(handlers.version_cmd(upd, _ctx()))
    assert any("Groq ·" in t for t in msg.texts)


def test_version_reports_the_app_version_from_the_version_file(monkeypatch):
    """/version shows config.VERSION (read from the VERSION file), not just the git SHA."""
    monkeypatch.setattr(config, "VERSION", "9.9.9")
    upd, msg = _update(_uid())
    asyncio.run(handlers.version_cmd(upd, _ctx()))
    assert any("v9.9.9" in t for t in msg.texts)


def test_authorized_ignores_unknown_ids():
    """A caller outside TG_CHAT_IDS gets no reply at all."""
    upd, msg = _update(999999)  # not in the "1,2" allowlist
    asyncio.run(handlers.status(upd, _ctx()))
    assert not msg.texts


def test_authorized_allows_listed_ids():
    """A caller in TG_CHAT_IDS gets a /status reply."""
    store.init_db()
    upd, msg = _update(int(config.TG_CHAT_IDS[0]))
    asyncio.run(handlers.status(upd, _ctx()))
    assert msg.texts and "run en curso" in msg.texts[0]


def test_help_sends_markdown_with_a_code_block():
    """/help renders with parse_mode=Markdown — needed for the ``` block to
    actually show up monospace, since plain spaces don't align anything in
    Telegram's normal proportional-font messages."""
    upd, msg = _update(int(config.TG_CHAT_IDS[0]))
    asyncio.run(handlers.help_(upd, _ctx()))

    assert msg.texts and msg.parse_modes == ["Markdown"]
    assert msg.texts[0].count("```") == 2


def test_help_command_columns_are_aligned():
    """Every command is padded to the same width, so every description
    starts at the same column regardless of how long its command is."""
    width = max(len(cmd) for _, rows in handlers._HELP_COMMANDS for cmd, _ in rows)
    block = handlers._format_help_commands()
    command_lines = [line for line in block.splitlines() if line.startswith("  /")]

    assert len(command_lines) >= 10  # sanity: didn't accidentally match nothing
    assert all(line[2 + width:2 + width + 2] == "  " for line in command_lines)


def test_recap_returns_the_stored_summary():
    """/recap with no args sends the most recent file's summary."""
    store.init_db()
    run_id = store.start_run("cron")
    store.record_file(run_id, "reunion.mp4", "video", "ok", summary="puntos clave: X")
    store.finish_run(run_id, "ok", 1, 1)

    upd, msg = _update(int(config.TG_CHAT_IDS[0]))
    asyncio.run(handlers.recap(upd, _ctx()))

    assert any("puntos clave: X" in t for t in msg.texts)


def test_recap_signs_the_summary_with_its_engines():
    """/recap appends the transcription/summary engines, same as the original Telegram send."""
    store.init_db()
    run_id = store.start_run("cron")
    store.record_file(
        run_id, "reunion.mp4", "video", "ok", summary="puntos clave: X",
        transcribe_backend="Groq · whisper-large-v3",
        summarize_backend="local · Ollama test-qwen",
    )
    store.finish_run(run_id, "ok", 1, 1)

    upd, msg = _update(int(config.TG_CHAT_IDS[0]))
    asyncio.run(handlers.recap(upd, _ctx()))

    text = "\n".join(msg.texts)
    assert "_transcripción: Groq · whisper-large-v3_" in text
    assert "_resumen: local · Ollama test-qwen_" in text
    # Both replies must render as Markdown, or the "_..._" signature shows up
    # as literal underscores instead of italics — reply_text defaults to
    # plain text unless parse_mode is passed explicitly.
    assert msg.parse_modes == ["Markdown", "Markdown"]


def test_recap_omits_the_transcription_line_for_text_files():
    """A text-kind file has no transcription engine, so only the summary line shows."""
    store.init_db()
    run_id = store.start_run("cron")
    store.record_file(
        run_id, "nota.txt", "text", "ok", summary="acuerdos",
        summarize_backend="local · Ollama test-qwen",
    )
    store.finish_run(run_id, "ok", 1, 1)

    upd, msg = _update(int(config.TG_CHAT_IDS[0]))
    asyncio.run(handlers.recap(upd, _ctx()))

    text = "\n".join(msg.texts)
    assert "_resumen: local · Ollama test-qwen_" in text
    assert "_transcripción:" not in text


def test_resolve_file_by_id_and_name():
    """resolve_file maps None/digit/name to the right file rows — a digit is
    the file's actual database id (the #N shown by /find), not a position."""
    store.init_db()
    run_id = store.start_run("cron")
    store.record_file(run_id, "one.txt", "text", "ok", summary="1")
    store.record_file(run_id, "two.txt", "text", "ok", summary="2")
    one_id = store.file_by_name("one.txt")["id"]

    assert handlers.resolve_file(None)["filename"] == "two.txt"
    assert handlers.resolve_file(str(one_id))["filename"] == "one.txt"
    assert handlers.resolve_file("one.txt")["summary"] == "1"
    assert handlers.resolve_file("missing.txt") is None
    assert handlers.resolve_file("999999") is None
    assert handlers.resolve_file(f"#{one_id}")["filename"] == "one.txt"


class _FakePopen:
    """Records the argv/env of a spawned process; wait() is a no-op."""

    last = None

    def __init__(self, argv, **kw):
        self.argv = argv
        self.env = kw.get("env")
        self.returncode = 0
        _FakePopen.last = self

    def wait(self):
        """Pretend the process finished immediately."""
        return 0


def _uid():
    return int(config.TG_CHAT_IDS[0])


def test_history_lists_recent_runs():
    """/history renders one line per run."""
    store.init_db()
    rid = store.start_run("cron")
    store.finish_run(rid, "ok", 2, 2)

    upd, msg = _update(_uid())
    asyncio.run(handlers.history(upd, _ctx()))

    assert msg.texts and f"#{rid}" in msg.texts[0] and "2/2" in msg.texts[0]


def test_logs_sends_the_run_log_as_a_document(tmp_path):
    """/logs with no arg attaches the LATEST run's log file."""
    store.init_db()
    logf = tmp_path / "run-x.log"
    logf.write_text("[t] Run 1 started\n[t] Done\n", encoding="utf-8")
    rid = store.start_run("cron", log_path=str(logf))
    store.finish_run(rid, "ok", 1, 1)

    upd, msg = _update(_uid())
    asyncio.run(handlers.logs(upd, _ctx()))

    assert msg.documents and msg.documents[0][0] == "run-x.log"


def test_logs_with_an_id_fetches_that_exact_run_not_the_nth_most_recent(tmp_path):
    """/logs <id> looks up the run whose id is <id> — the same id shown by
    /history, /status and every run's own log lines — not "the id-th most
    recent run" (a real point of confusion: a cron run numbered e.g. #31
    should be reachable as `/logs 31`, regardless of how many runs exist)."""
    store.init_db()
    old_log = tmp_path / "run-old.log"
    old_log.write_text("primera corrida", encoding="utf-8")
    new_log = tmp_path / "run-new.log"
    new_log.write_text("corrida mas reciente", encoding="utf-8")

    old_id = store.start_run("cron", log_path=str(old_log))
    store.finish_run(old_id, "ok", 1, 1)
    new_id = store.start_run("cron", log_path=str(new_log))
    store.finish_run(new_id, "ok", 1, 1)
    assert new_id != old_id

    upd, msg = _update(_uid())
    asyncio.run(handlers.logs(upd, _ctx([str(old_id)])))

    assert msg.documents and msg.documents[0][0] == "run-old.log"


def test_logs_with_a_hash_prefixed_id_matches_the_bare_digit_form(tmp_path):
    """/logs #<id> — /history and /find display ids with a leading '#', so
    typing one back exactly as shown must resolve the same run as the bare
    digit form (`_parse_id` strips the '#' before parsing)."""
    store.init_db()
    logf = tmp_path / "run-hash.log"
    logf.write_text("corrida buscada por hash", encoding="utf-8")
    rid = store.start_run("cron", log_path=str(logf))
    store.finish_run(rid, "ok", 1, 1)

    upd, msg = _update(_uid())
    asyncio.run(handlers.logs(upd, _ctx([f"#{rid}"])))

    assert msg.documents and msg.documents[0][0] == "run-hash.log"


def test_logs_errors_is_not_misparsed_as_an_id(tmp_path):
    """/logs errors must still filter the latest run's log, not be treated as
    (and fail to parse as) an id."""
    store.init_db()
    logf = tmp_path / "run-err.log"
    logf.write_text("[t] INFO fine\n[t] ERROR boom\n", encoding="utf-8")
    rid = store.start_run("cron", log_path=str(logf))
    store.finish_run(rid, "error", 1, 0)

    upd, msg = _update(_uid())
    asyncio.run(handlers.logs(upd, _ctx(["errors"])))

    assert msg.texts and "ERROR boom" in msg.texts[0] and "INFO fine" not in msg.texts[0]


def test_transcript_sends_the_archived_file(tmp_path):
    """/transcript attaches the stored transcript."""
    store.init_db()
    tf = tmp_path / "nota.txt"
    tf.write_text("contenido", encoding="utf-8")
    rid = store.start_run("cron")
    store.record_file(rid, "nota.txt", "text", "ok", transcript_path=str(tf), summary="s")

    upd, msg = _update(_uid())
    asyncio.run(handlers.transcript(upd, _ctx()))

    assert msg.documents and msg.documents[0][0] == "nota.txt"


def test_run_spawns_pipeline_with_manual_env(monkeypatch):
    """/run launches pipeline.py with PISCRIBE_TRIGGER=manual and the caller id."""
    store.init_db()
    monkeypatch.setattr(handlers.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(handlers, "current_run_pid", lambda: None)

    upd, msg = _update(_uid())
    asyncio.run(handlers.run(upd, _ctx(["reunion.mp4"])))

    env = _FakePopen.last.env
    assert env["PISCRIBE_TRIGGER"] == "manual"
    assert env["PISCRIBE_BY"] == str(_uid())
    assert env["PISCRIBE_ONLY"] == "reunion.mp4"
    # /run posts no announce/report of its own: pipeline.py's run_pipeline
    # already sends the same start/stage/failure/end messages cron gets.
    assert not msg.texts


def test_runcron_spawns_pipeline_with_trigger_forced_to_cron(monkeypatch):
    """/runcron overrides PISCRIBE_TRIGGER to 'cron' (so pipeline._targets
    broadcasts to every configured chat instead of just the caller) while
    still recording who actually asked for it via PISCRIBE_BY."""
    store.init_db()
    monkeypatch.setattr(handlers.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(handlers, "current_run_pid", lambda: None)

    upd, msg = _update(_uid())
    asyncio.run(handlers.runcron(upd, _ctx(["reunion.mp4"])))

    env = _FakePopen.last.env
    assert env["PISCRIBE_TRIGGER"] == "cron"
    assert env["PISCRIBE_BY"] == str(_uid())
    assert env["PISCRIBE_ONLY"] == "reunion.mp4"
    assert not msg.texts  # same own_messages=True behavior as /run


def test_runcron_with_no_args_still_forces_cron_and_has_no_target_file(monkeypatch):
    """/runcron with no filename still forces the cron trigger, and doesn't
    set PISCRIBE_ONLY at all (processes the whole pending folder)."""
    store.init_db()
    monkeypatch.setattr(handlers.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(handlers, "current_run_pid", lambda: None)

    upd, _msg = _update(_uid())
    asyncio.run(handlers.runcron(upd, _ctx()))

    env = _FakePopen.last.env
    assert env["PISCRIBE_TRIGGER"] == "cron"
    assert "PISCRIBE_ONLY" not in env


def test_retry_report_shows_a_spanish_status_label_not_the_raw_value(monkeypatch):
    """The post-/retry report translates the DB status word (e.g. 'partial') to Spanish.

    Unlike /run, /retry still gets its own bot-side report (it goes through
    _single_file_run, which has no per-stage Telegram messages of its own).
    """
    store.init_db()
    rid = store.start_run("manual")
    store.record_file(rid, "a.mp4", "video", "ok")
    store.finish_run(rid, "partial", 2, 1)

    monkeypatch.setattr(handlers.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(handlers, "current_run_pid", lambda: None)

    upd, msg = _update(_uid())
    asyncio.run(handlers.retry(upd, _ctx()))

    final = msg.texts[-1]
    assert "parcial" in final
    assert "partial" not in final


def test_retry_resolves_a_filename_and_sets_mode(monkeypatch):
    """/retry with no arg targets the latest file and sets PISCRIBE_MODE=retry."""
    store.init_db()
    rid = store.start_run("cron")
    store.record_file(rid, "latest.mp4", "video", "error", error="x")
    monkeypatch.setattr(handlers.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(handlers, "current_run_pid", lambda: None)

    upd, _msg = _update(_uid())
    asyncio.run(handlers.retry(upd, _ctx()))

    env = _FakePopen.last.env
    assert env["PISCRIBE_MODE"] == "retry"
    assert env["PISCRIBE_ONLY"] == "latest.mp4"


def test_retry_with_a_digit_targets_that_files_id_not_a_position(monkeypatch):
    """/retry <digit> resolves the digit as the file's actual database id
    (the #N shown by /find) — NOT "the n-th most recent file". The older,
    position-based file.mp4 here has the LOWER id despite being requested
    second, so an id-based lookup and a position-based one would disagree."""
    store.init_db()
    rid = store.start_run("cron")
    store.record_file(rid, "older.mp4", "video", "error", error="x")
    store.record_file(rid, "newer.mp4", "video", "error", error="y")
    older_id = store.file_by_name("older.mp4")["id"]
    monkeypatch.setattr(handlers.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(handlers, "current_run_pid", lambda: None)

    upd, _msg = _update(_uid())
    asyncio.run(handlers.retry(upd, _ctx([str(older_id)])))

    env = _FakePopen.last.env
    assert env["PISCRIBE_ONLY"] == "older.mp4"


def test_retry_with_a_hash_prefixed_id_targets_that_files_id(monkeypatch):
    """/retry #<id> — the exact form /find shows — must resolve by id, not be
    treated as a literal filename (the regression behind the 'processed/#44
    directory not found' rclone failure)."""
    store.init_db()
    rid = store.start_run("cron")
    store.record_file(rid, "hashed.mp4", "video", "error", error="x")
    hashed_id = store.file_by_name("hashed.mp4")["id"]
    monkeypatch.setattr(handlers.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(handlers, "current_run_pid", lambda: None)

    upd, _msg = _update(_uid())
    asyncio.run(handlers.retry(upd, _ctx([f"#{hashed_id}"])))

    env = _FakePopen.last.env
    assert env["PISCRIBE_ONLY"] == "hashed.mp4"


def test_run_is_refused_while_a_run_holds_the_lock(monkeypatch):
    """/run does not spawn a second pipeline while one is in progress."""
    _FakePopen.last = None
    monkeypatch.setattr(handlers.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(handlers, "current_run_pid", lambda: 4242)
    monkeypatch.setattr(handlers, "_pid_alive", lambda pid: True)

    upd, msg = _update(_uid())
    asyncio.run(handlers.run(upd, _ctx()))

    assert _FakePopen.last is None
    assert any("run en curso" in t for t in msg.texts)


def test_pause_and_resume_toggle_the_flag():
    """/pause creates the flag, /resume removes it."""
    upd, _msg = _update(_uid())
    asyncio.run(handlers.pause(upd, _ctx()))
    assert config.PAUSE_FLAG.exists()
    asyncio.run(handlers.resume(upd, _ctx()))
    assert not config.PAUSE_FLAG.exists()


def test_cancel_with_no_run_in_progress():
    """/cancel says so when nothing is running."""
    upd, msg = _update(_uid())
    asyncio.run(handlers.cancel(upd, _ctx()))
    assert msg.texts == ["no hay run en curso."]


def test_find_reports_matches():
    """/find searches summaries and filenames."""
    store.init_db()
    rid = store.start_run("cron")
    store.record_file(rid, "abril.mp4", "video", "ok", summary="acuerdo clave")

    upd, msg = _update(_uid())
    asyncio.run(handlers.find(upd, _ctx(["acuerdo"])))
    assert any("abril.mp4" in t for t in msg.texts)


def test_deadman_recovery_notice():
    """deadman_check sends a recovery message once the streak is broken."""
    store.init_db()
    rid = store.start_run("cron")
    store.finish_run(rid, "ok", 1, 1)

    sent = []

    class _Bot:
        async def send_message(self, _chat_id, text):
            """Record a broadcast message."""
            sent.append(text)

    ctx = SimpleNamespace(bot=_Bot(), bot_data={"deadman_alerted": True})
    asyncio.run(handlers.deadman_check(ctx))

    assert any("recuperado" in t for t in sent)
    assert ctx.bot_data["deadman_alerted"] is False
