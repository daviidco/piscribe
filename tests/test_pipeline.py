"""Regression tests for the piscribe pipeline.

These mirror the scenarios from the manual end-to-end simulation: every external
tool or client (rclone, ffmpeg, whisper, Ollama, the Telegram HTTP client) is
stubbed, so nothing here touches the network or any path outside the test sandbox.
"""

# A test that receives a fixture by name necessarily shadows the fixture
# function; that is the intended pytest pattern, not a mistake. Reaching into
# pipeline._targets is deliberate too: it's the one function this suite needs
# to verify directly, not through some public wrapper.
# pylint: disable=redefined-outer-name,protected-access

from types import SimpleNamespace

import httpx
import pytest

import drive
import pipeline
import store
import telegram_api
import video
from runlock import run_lock

# Captured at collection time, before conftest's autouse `_no_real_telegram_calls`
# fixture overwrites telegram_api._post for the duration of each test — the two
# _post tests below need the real implementation, not that no-op stand-in.
_real_telegram_post = telegram_api._post


# ---------------------------------------------------------------------------
# telegram_api.split_message
# ---------------------------------------------------------------------------

def test_split_message_short_text_is_one_part():
    """Text under the limit is returned unchanged as a single part."""
    assert list(telegram_api.split_message("hola")) == ["hola"]


def test_split_message_respects_limit_and_keeps_content():
    """A long, newline-rich message splits into within-limit parts, losing nothing."""
    text = "linea de prueba\n" * 800  # ~12.8k chars, newline every ~16
    parts = list(telegram_api.split_message(text, limit=4000))

    assert len(parts) > 1
    assert all(len(p) <= 4000 for p in parts)
    assert "\n".join(parts).replace("\n", "") == text.replace("\n", "")


def test_split_message_hard_cuts_a_line_with_no_newline():
    """With no newline to break on, the splitter falls back to hard character cuts."""
    text = "x" * 9000
    parts = list(telegram_api.split_message(text, limit=4000))

    assert [len(p) for p in parts] == [4000, 4000, 1000]
    assert "".join(parts) == text


# ---------------------------------------------------------------------------
# telegram_api._post
# ---------------------------------------------------------------------------

class _FakeHttpxResponse:  # pylint: disable=too-few-public-methods
    """Stand-in for httpx.Response — just enough for _post's success check."""

    is_success = True

    def json(self):
        """Always report Telegram-style success."""
        return {"ok": True}


def test_post_sends_a_semicolon_containing_value_intact(monkeypatch):
    """A semicolon in a field's value must reach httpx's payload unmodified.

    This was never guaranteed with the previous curl-based _post: curl's -F
    multipart syntax treats a bare ';' inside a value as the start of an
    extra parameter clause and silently truncates the field there —
    confirmed against a real curl invocation (`-F "text=a; b"` reached the
    server as just "a", `b` gone, no error anywhere). httpx has no such
    ambiguity since it isn't building a hand-rolled command line at all."""
    captured = []
    monkeypatch.setattr(
        telegram_api._client, "post",
        lambda url, data=None, files=None: captured.append((url, data, files))
        or _FakeHttpxResponse(),
    )

    _real_telegram_post(
        "sendMessage", {"parse_mode": "Markdown", "text": "hola; sigue despues"}, ["123"]
    )

    assert len(captured) == 1
    _url, data, files = captured[0]
    assert data["text"] == "hola; sigue despues"
    assert data["chat_id"] == "123"
    assert files is None


def test_post_sends_a_file_upload_via_files_not_data(monkeypatch):
    """A file upload (send_document's use case) goes through httpx's files=
    parameter as raw bytes, not folded into the plain data= fields."""
    captured = []
    monkeypatch.setattr(
        telegram_api._client, "post",
        lambda url, data=None, files=None: captured.append((url, data, files))
        or _FakeHttpxResponse(),
    )

    _real_telegram_post(
        "sendDocument", {"caption": "a; b"}, ["123"],
        files={"document": ("run-x.log", b"contenido del log")},
    )

    assert len(captured) == 1
    _url, data, files = captured[0]
    assert data == {"chat_id": "123", "caption": "a; b"}
    assert files == {"document": ("run-x.log", b"contenido del log")}


def test_post_never_raises_and_logs_a_warning_on_a_network_failure(monkeypatch):
    """A network error (or a non-'ok' response) is logged, never raised — a
    failed notification shouldn't crash a pipeline run."""

    def boom(_url, data=None, files=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(telegram_api._client, "post", boom)
    warnings = []
    monkeypatch.setattr(telegram_api, "log_warning", warnings.append)

    _real_telegram_post("sendMessage", {"text": "hola"}, ["123"])  # must not raise

    assert warnings and "sendMessage" in warnings[0]


# ---------------------------------------------------------------------------
# telegram_api.to_telegram_markdown
# ---------------------------------------------------------------------------

def test_to_telegram_markdown_converts_a_heading_to_bold():
    """Telegram has no heading syntax in any of its Markdown modes — a raw
    '## Heading' line would show up completely unstyled, '##' and all."""
    assert telegram_api.to_telegram_markdown("## Resumen") == "*Resumen*"


def test_to_telegram_markdown_collapses_every_heading_level_the_same_way():
    """Telegram can't distinguish heading levels either, so # through ######
    all become the same bold treatment."""
    for level in range(1, 7):
        heading = "#" * level + " Título"
        assert telegram_api.to_telegram_markdown(heading) == "*Título*"


def test_to_telegram_markdown_handles_headings_and_inline_bold_together():
    """A heading and an inline **bold** span in the same text both convert,
    independently of each other."""
    text = "# Resumen\n**importante** y texto normal\n\n## Puntos clave\n- x"
    expected = "*Resumen*\n*importante* y texto normal\n\n*Puntos clave*\n- x"
    assert telegram_api.to_telegram_markdown(text) == expected


def test_to_telegram_markdown_leaves_a_mid_line_hash_alone():
    """A '#' that isn't the very first thing on a line — e.g. a stray ticket
    reference like "ticket #3619" — is not a heading and must stay literal."""
    text = "El ticket #3619 quedó bloqueado."
    assert telegram_api.to_telegram_markdown(text) == text


def test_to_telegram_markdown_leaves_ordinary_prose_and_bullets_alone():
    """Plain prose and '- bullet' list items pass through unchanged."""
    text = "Texto normal.\n- primer punto\n- segundo punto"
    assert telegram_api.to_telegram_markdown(text) == text


# ---------------------------------------------------------------------------
# drive.list_pending_files
# ---------------------------------------------------------------------------

def test_list_pending_files_filters_by_extension(monkeypatch):
    """Only video/text extensions pass; images and sub-directories are dropped."""
    listing = "meeting.mp4\nnotes.txt\nreadme.md\nphoto.png\nsubdir/\n\n"
    monkeypatch.setattr(drive, "run", lambda cmd: SimpleNamespace(stdout=listing))

    assert drive.list_pending_files() == ["meeting.mp4", "notes.txt", "readme.md"]


# ---------------------------------------------------------------------------
# pipeline.process_file
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub the side-effecting calls in process_file; collect Telegram sends."""
    sent = []
    monkeypatch.setattr(pipeline, "download_file", lambda name, dest: None)
    monkeypatch.setattr(pipeline, "move_in_drive", lambda name: None)
    monkeypatch.setattr(
        pipeline, "generate_summary",
        lambda text: (f"SUMMARY<<{text.strip()}>>", "local · Ollama test-qwen"),
    )
    monkeypatch.setattr(
        pipeline, "send_telegram_message", lambda text, chat_ids=None: sent.append(text)
    )
    # Otherwise process_file/_resummarize would call the real embeddings.index_file,
    # which reaches out to a local Ollama server no test environment has running.
    monkeypatch.setattr(pipeline.embeddings, "index_file", lambda *a, **kw: None)
    return sent


def test_process_file_text_note(work_dirs, stub_pipeline):
    """A .txt note is summarised and archived; the summary, its engine
    signature, and the /ask indexing outcome are sent as three separate
    messages (summary/signature first, so indexing never delays them), local
    copy removed."""
    note = work_dirs.LOCAL_DIR / "nota.txt"
    note.write_text("acuerdos de la reunion", encoding="utf-8")

    result = pipeline.process_file("nota.txt")

    assert len(stub_pipeline) == 3
    assert "nota.txt" in stub_pipeline[0]
    assert "SUMMARY<<acuerdos de la reunion>>" in stub_pipeline[0]
    assert "_resumen:" not in stub_pipeline[0]  # signature is its own message
    assert "embedding" in stub_pipeline[2] and "nota.txt" in stub_pipeline[2]
    assert "_resumen: local · Ollama test-qwen_" in stub_pipeline[1]
    assert "_transcripción:" not in stub_pipeline[1]  # no transcription stage for text input
    assert not note.exists()  # local copy cleaned up

    assert result.status == "ok"
    assert result.kind == "text"
    assert result.summary == "SUMMARY<<acuerdos de la reunion>>"  # signature is not baked in
    assert result.summarize_backend == "local · Ollama test-qwen"
    assert result.transcribe_backend is None
    archived = work_dirs.TRANSCRIPTIONS_DIR / "nota.txt"
    assert archived.read_text(encoding="utf-8") == "acuerdos de la reunion"
    assert result.transcript_path == str(archived)


@pytest.mark.usefixtures("stub_pipeline")
def test_process_file_cleans_local_copy_when_a_later_step_fails(work_dirs, monkeypatch):
    """The finally block deletes the local copy even when summarising raises."""
    note = work_dirs.LOCAL_DIR / "boom.txt"
    note.write_text("contenido", encoding="utf-8")

    def explode(_text):
        """Fake generate_summary that always fails."""
        raise RuntimeError("ollama down")

    monkeypatch.setattr(pipeline, "generate_summary", explode)

    with pytest.raises(RuntimeError, match="ollama down"):
        pipeline.process_file("boom.txt")

    assert not note.exists()  # finally-block cleanup still ran


# ---------------------------------------------------------------------------
# video.process_video
# ---------------------------------------------------------------------------

def test_process_video_reads_transcript_and_removes_wav(work_dirs, monkeypatch):
    """process_video falls back to local whisper (no GROQ_API_KEY in tests),
    returns its transcript and a backend label, and cleans up the temp .wav."""
    monkeypatch.setattr(video, "extract_audio", lambda src, dst: None)

    def fake_transcribe(_audio_path, output_base):
        """Fake whisper run that writes the expected ``<base>.txt``."""
        output_base.with_name(f"{output_base.name}.txt").write_text(
            "transcripcion simulada", encoding="utf-8"
        )

    monkeypatch.setattr(video, "transcribe_local", fake_transcribe)

    local_path = work_dirs.LOCAL_DIR / "clip.mp4"
    local_path.write_bytes(b"fake")

    text, backend = video.process_video("clip.mp4", local_path)

    assert text == "transcripcion simulada"
    assert backend == "local · whisper.cpp (ggml-small)"
    assert not (work_dirs.LOCAL_DIR / "clip.wav").exists()


def test_process_video_raises_when_whisper_writes_nothing(work_dirs, monkeypatch):
    """A clean whisper exit that produces no file surfaces as a RuntimeError."""
    monkeypatch.setattr(video, "extract_audio", lambda src, dst: None)
    monkeypatch.setattr(video, "transcribe_local", lambda a, b: None)  # writes no .txt

    local_path = work_dirs.LOCAL_DIR / "clip.mp4"
    local_path.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="no transcript"):
        video.process_video("clip.mp4", local_path)


# ---------------------------------------------------------------------------
# pipeline.main
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("work_dirs")
def test_main_continues_after_one_file_fails(monkeypatch, read_log):
    """One failing file is logged, the batch continues, and the run is recorded."""
    monkeypatch.setattr(pipeline, "list_pending_files", lambda: ["bad.txt", "good.txt"])
    handled = []
    sent = []
    monkeypatch.setattr(
        pipeline, "send_telegram_message", lambda text, chat_ids=None: sent.append(text)
    )

    def fake_process(name, **_kwargs):
        """Fake process_file that fails only for bad.txt."""
        handled.append(name)
        if name == "bad.txt":
            raise RuntimeError("kaboom")
        return pipeline.FileResult(filename=name, kind="text", status="ok")

    monkeypatch.setattr(pipeline, "process_file", fake_process)

    pipeline.main()  # must not raise

    assert handled == ["bad.txt", "good.txt"]
    assert "ERROR processing bad.txt: kaboom" in read_log()

    last = store.last_run()
    assert last["trigger"] == "cron"
    assert last["status"] == "partial"
    assert (last["files_ok"], last["files_total"]) == (1, 2)

    # An immediate failure notice plus start/end messages — same for cron and
    # manual /run, see _notify.
    assert any("bad.txt" in m and "kaboom" in m for m in sent)
    assert any("iniciando" in m for m in sent)
    assert any("finalizada" in m and "1/2" in m for m in sent)


@pytest.mark.usefixtures("work_dirs")
def test_main_logs_listing_failure_without_traceback(monkeypatch, read_log):
    """A failure while listing the pending folder is logged, not raised."""
    sent = []
    monkeypatch.setattr(
        pipeline, "send_telegram_message", lambda text, chat_ids=None: sent.append(text)
    )

    def boom():
        """Fake list_pending_files that fails as if rclone were missing."""
        raise FileNotFoundError("[Errno 2] No such file or directory: 'rclone'")

    monkeypatch.setattr(pipeline, "list_pending_files", boom)

    pipeline.main()  # must not raise

    assert "ERROR listing pending files" in read_log()
    assert store.last_run()["status"] == "error"
    assert any("rclone" in m for m in sent)  # cron is told listing failed, too


@pytest.mark.usefixtures("work_dirs")
def test_main_skips_when_run_lock_is_held(monkeypatch, read_log):
    """A second cron invocation while a run holds the lock exits without working."""
    called = []
    monkeypatch.setattr(pipeline, "list_pending_files", lambda: called.append(1) or [])

    with run_lock():                # simulate a run already in progress
        pipeline.main()

    assert not called               # run_pipeline never got to list files
    assert store.last_run() is None
    assert "Skipping cron run" in read_log()


# ---------------------------------------------------------------------------
# run_pipeline(only=...) / manual trigger / retry / resummarize / cancel
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("work_dirs")
def test_run_pipeline_only_filters_to_one_pending_file(monkeypatch):
    """`only` restricts the pass to a single pending filename."""
    monkeypatch.setattr(pipeline, "list_pending_files", lambda: ["a.txt", "b.txt"])
    seen = []

    def fake_process(name, **_kwargs):
        seen.append(name)
        return pipeline.FileResult(filename=name, kind="text", status="ok")

    monkeypatch.setattr(pipeline, "process_file", fake_process)

    result = pipeline.run_pipeline("manual", requested_by="7", only="b.txt")

    assert seen == ["b.txt"]
    assert result.status == "ok"
    assert store.last_run()["requested_by"] == "7"


@pytest.mark.usefixtures("work_dirs")
def test_main_reads_trigger_and_requested_by_from_env(monkeypatch):
    """/run's PISCRIBE_* env vars land on the recorded run."""
    monkeypatch.setenv("PISCRIBE_TRIGGER", "manual")
    monkeypatch.setenv("PISCRIBE_BY", "42")
    monkeypatch.setattr(pipeline, "list_pending_files", list)

    pipeline.main()

    last = store.last_run()
    assert last["trigger"] == "manual"
    assert last["requested_by"] == "42"


@pytest.mark.usefixtures("work_dirs")
def test_retry_file_reprocesses_from_processed(monkeypatch):
    """retry_file runs process_file with source='processed' as its own run."""
    calls = {}

    def fake_process(name, *, source="pending", **_kwargs):
        calls["name"] = name
        calls["source"] = source
        return pipeline.FileResult(filename=name, kind="video", status="ok", summary="s")

    monkeypatch.setattr(pipeline, "process_file", fake_process)

    result = pipeline.retry_file("clip.mp4", requested_by="9")

    assert calls == {"name": "clip.mp4", "source": "processed"}
    assert result.status == "ok"
    row = store.last_run()
    assert row["trigger"] == "manual" and row["files_ok"] == 1


@pytest.mark.usefixtures("work_dirs")
def test_resummarize_file_uses_the_stored_transcript(monkeypatch):
    """resummarize_file re-runs only the summary over the archived transcript."""
    transcript = pipeline.TRANSCRIPTIONS_DIR / "nota.txt"
    transcript.write_text("texto original de la reunion", encoding="utf-8")
    rid = store.start_run("cron")
    store.record_file(rid, "nota.txt", "text", "ok", transcript_path=str(transcript))
    store.finish_run(rid, "ok", 1, 1)

    sent = []
    monkeypatch.setattr(
        pipeline, "generate_summary",
        lambda text: (f"RE<<{text}>>", "local · Ollama test-qwen"),
    )
    monkeypatch.setattr(
        pipeline, "send_telegram_message", lambda text, chat_ids=None: sent.append(text)
    )
    monkeypatch.setattr(pipeline.embeddings, "index_file", lambda *a, **kw: None)

    result = pipeline.resummarize_file("nota.txt")

    assert result.status == "ok"
    assert len(sent) == 3
    assert "RE<<texto original de la reunion>>" in sent[0]
    assert "_resumen:" not in sent[0]  # signature is its own message
    assert "_resumen: local · Ollama test-qwen_" in sent[1]
    assert "embedding" in sent[2] and "nota.txt" in sent[2]
    assert store.latest_file()["summary"] == "RE<<texto original de la reunion>>"
    assert store.latest_file()["summarize_backend"] == "local · Ollama test-qwen"


@pytest.mark.usefixtures("work_dirs")
def test_run_pipeline_marks_cancelled_on_sigterm(monkeypatch):
    """A KeyboardInterrupt mid-batch (SIGTERM) finishes the run as 'cancelled'."""
    monkeypatch.setattr(pipeline, "list_pending_files", lambda: ["x.txt"])

    def boom(_name, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline, "process_file", boom)

    result = pipeline.run_pipeline("manual")

    assert result.status == "cancelled"
    assert store.last_run()["status"] == "cancelled"


@pytest.mark.usefixtures("work_dirs")
def test_run_pipeline_cron_notifies_on_cancel(monkeypatch):
    """A run cancelled mid-batch (e.g. via /cancel) tells Telegram."""
    monkeypatch.setattr(pipeline, "list_pending_files", lambda: ["x.txt"])
    sent = []
    monkeypatch.setattr(
        pipeline, "send_telegram_message", lambda text, chat_ids=None: sent.append(text)
    )

    def boom(_name, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline, "process_file", boom)

    result = pipeline.run_pipeline("cron")

    assert result.status == "cancelled"
    assert any("cancelada" in m for m in sent)


@pytest.mark.usefixtures("work_dirs")
def test_run_pipeline_cron_notifies_when_no_pending_files(monkeypatch):
    """An empty cron pass still pings Telegram, so silence never means 'stuck'."""
    monkeypatch.setattr(pipeline, "list_pending_files", list)
    sent = []
    monkeypatch.setattr(
        pipeline, "send_telegram_message", lambda text, chat_ids=None: sent.append(text)
    )

    result = pipeline.run_pipeline("cron")

    assert result.status == "ok"
    assert any("sin archivos pendientes" in m for m in sent)


@pytest.mark.usefixtures("work_dirs")
def test_run_pipeline_notifies_when_paused(monkeypatch):
    """A run that hits the pause flag tells Telegram it was skipped, not just the log."""
    pipeline.PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    pipeline.PAUSE_FLAG.touch()
    sent = []
    monkeypatch.setattr(
        pipeline, "send_telegram_message", lambda text, chat_ids=None: sent.append(text)
    )

    result = pipeline.run_pipeline("cron")

    assert result.status == "skipped"
    assert any("pausa" in m for m in sent)


@pytest.mark.usefixtures("work_dirs")
def test_run_pipeline_notifies_when_only_target_is_not_pending(monkeypatch):
    """/run <file> for a file that isn't actually pending reports back, not just logs."""
    monkeypatch.setattr(pipeline, "list_pending_files", lambda: ["other.txt"])
    sent = []
    monkeypatch.setattr(
        pipeline, "send_telegram_message", lambda text, chat_ids=None: sent.append(text)
    )

    result = pipeline.run_pipeline("manual", only="missing.txt")

    assert result.status == "error"
    assert any("missing.txt" in m for m in sent)


@pytest.mark.usefixtures("work_dirs")
def test_run_pipeline_manual_trigger_gets_the_same_notifications_as_cron(monkeypatch):
    """A manual /run posts the same start/failure/end messages a cron pass would.

    pipeline.py no longer distinguishes triggers for these; the bot skips its
    own announce/report for /run specifically to avoid duplicating them (see
    handlers.run / handlers._spawn_and_report).
    """
    monkeypatch.setattr(pipeline, "list_pending_files", lambda: ["a.txt", "bad.txt"])
    sent = []
    monkeypatch.setattr(
        pipeline, "send_telegram_message", lambda text, chat_ids=None: sent.append(text)
    )

    def fake_process(name, **_kwargs):
        if name == "bad.txt":
            raise RuntimeError("kaboom")
        return pipeline.FileResult(filename=name, kind="text", status="ok")

    monkeypatch.setattr(pipeline, "process_file", fake_process)

    pipeline.run_pipeline("manual")

    assert any("iniciando" in m for m in sent)
    assert any("bad.txt" in m and "kaboom" in m for m in sent)
    assert any("finalizada" in m and "1/2" in m for m in sent)


def test_process_file_notify_stages_reports_each_stage_with_position(
    work_dirs, stub_pipeline
):
    """notify_stages posts a download/transcribe/summary message per file, tagged."""
    note = work_dirs.LOCAL_DIR / "nota.txt"
    note.write_text("contenido", encoding="utf-8")

    pipeline.process_file("nota.txt", notify_stages=True, position=(2, 3))

    stages = stub_pipeline[:-3]  # last three entries: summary, signature, embedding outcome
    assert any("descargando" in m and "(2/3)" in m for m in stages)
    assert any("generando resumen" in m and "(2/3)" in m for m in stages)
    assert not any("transcribiendo" in m for m in stages)  # text file, no transcription stage


# ---------------------------------------------------------------------------
# Message audience: cron broadcasts to every chat, on-demand triggers (/run,
# /retry, /resummarize) go ONLY to whoever asked — same wording either way,
# only the recipient list differs. See pipeline._targets.
# ---------------------------------------------------------------------------

def _capture_sent(monkeypatch):
    """Monkeypatch pipeline.send_telegram_message to record (text, chat_ids) pairs."""
    calls = []
    monkeypatch.setattr(
        pipeline, "send_telegram_message",
        lambda text, chat_ids=None: calls.append((text, chat_ids)),
    )
    return calls


def test_targets_broadcasts_for_cron_but_narrows_to_the_requester_for_manual():
    """_targets: cron -> None (broadcast); an on-demand trigger -> [requested_by]."""
    assert pipeline._targets("cron", None) is None
    assert pipeline._targets("cron", "42") is None  # cron never has a real requester anyway
    assert pipeline._targets("manual", "42") == ["42"]
    assert pipeline._targets("manual", None) is None  # no requester to narrow to


@pytest.mark.usefixtures("work_dirs")
def test_run_pipeline_cron_messages_broadcast_to_every_chat(monkeypatch):
    """A cron pass's messages go out with chat_ids=None (telegram_api broadcasts)."""
    monkeypatch.setattr(pipeline, "list_pending_files", list)
    sent = _capture_sent(monkeypatch)

    pipeline.run_pipeline("cron")

    assert sent  # the "sin archivos pendientes" message, at least
    assert all(chat_ids is None for _text, chat_ids in sent)


@pytest.mark.usefixtures("work_dirs")
def test_run_pipeline_manual_messages_go_only_to_the_requester(monkeypatch):
    """A manual /run's messages (start, stage, failure, end) all target just
    whoever asked — not a broadcast to every configured chat."""
    monkeypatch.setattr(pipeline, "list_pending_files", lambda: ["a.txt", "bad.txt"])
    sent = _capture_sent(monkeypatch)

    def fake_process(name, **_kwargs):
        if name == "bad.txt":
            raise RuntimeError("kaboom")
        return pipeline.FileResult(filename=name, kind="text", status="ok")

    monkeypatch.setattr(pipeline, "process_file", fake_process)

    pipeline.run_pipeline("manual", requested_by="123")

    assert sent  # start + failure + end, at least
    assert all(chat_ids == ["123"] for _text, chat_ids in sent)


@pytest.mark.usefixtures("work_dirs")
def test_retry_file_targets_only_the_requester(monkeypatch):
    """retry_file's process_file call — and everything it sends — is scoped
    to the id that asked for the retry, not broadcast."""
    seen = {}

    # source is part of process_file's real signature but irrelevant here.
    def fake_process(  # pylint: disable=unused-argument
        name, *, source="pending", chat_ids=None, **_kwargs
    ):
        seen["chat_ids"] = chat_ids
        return pipeline.FileResult(filename=name, kind="video", status="ok", summary="s")

    monkeypatch.setattr(pipeline, "process_file", fake_process)

    pipeline.retry_file("clip.mp4", requested_by="9")

    assert seen["chat_ids"] == ["9"]


@pytest.mark.usefixtures("work_dirs")
def test_resummarize_file_targets_only_the_requester(monkeypatch):
    """resummarize_file's summary/signature messages are scoped to the
    requester, not broadcast to every configured chat."""
    transcript = pipeline.TRANSCRIPTIONS_DIR / "nota.txt"
    transcript.write_text("texto original", encoding="utf-8")
    rid = store.start_run("cron")
    store.record_file(rid, "nota.txt", "text", "ok", transcript_path=str(transcript))
    store.finish_run(rid, "ok", 1, 1)

    monkeypatch.setattr(pipeline, "generate_summary", lambda text: ("re", "local · test"))
    monkeypatch.setattr(pipeline.embeddings, "index_file", lambda *a, **kw: None)
    sent = _capture_sent(monkeypatch)

    pipeline.resummarize_file("nota.txt", requested_by="55")

    assert sent
    assert all(chat_ids == ["55"] for _text, chat_ids in sent)


# ---------------------------------------------------------------------------
# RAG indexing (embeddings.index_file) — called from process_file/_resummarize
# so /ask always reflects the last content actually delivered.
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("stub_pipeline")
def test_process_file_indexes_transcript_and_summary_for_rag(work_dirs, monkeypatch):
    """process_file indexes both the transcript text and the summary."""
    note = work_dirs.LOCAL_DIR / "nota.txt"
    note.write_text("acuerdos de la reunion", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(
        pipeline.embeddings, "index_file",
        lambda filename, transcript=None, summary=None: seen.update(
            filename=filename, transcript=transcript, summary=summary
        ),
    )

    pipeline.process_file("nota.txt")

    assert seen == {
        "filename": "nota.txt",
        "transcript": "acuerdos de la reunion",
        "summary": "SUMMARY<<acuerdos de la reunion>>",
    }


def test_process_file_indexing_failure_does_not_break_delivery(
    work_dirs, stub_pipeline, monkeypatch, read_log
):
    """A RAG indexing failure (e.g. Ollama down) is logged, not raised — the
    summary was already generated and must still be delivered."""
    note = work_dirs.LOCAL_DIR / "nota.txt"
    note.write_text("contenido", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(pipeline.embeddings, "index_file", boom)

    result = pipeline.process_file("nota.txt")

    assert result.status == "ok"
    assert len(stub_pipeline) == 3  # summary + signature still sent, plus the failure notice
    assert "falló" in stub_pipeline[2] and "nota.txt" in stub_pipeline[2]
    assert "indexing nota.txt for RAG failed" in read_log()


@pytest.mark.usefixtures("work_dirs")
def test_resummarize_indexes_only_the_summary_not_the_transcript(monkeypatch):
    """_resummarize (via resummarize_file) re-indexes only the summary kind,
    leaving the transcript's indexed chunks untouched (it didn't change)."""
    transcript = pipeline.TRANSCRIPTIONS_DIR / "nota.txt"
    transcript.write_text("texto original", encoding="utf-8")
    rid = store.start_run("cron")
    store.record_file(rid, "nota.txt", "text", "ok", transcript_path=str(transcript))
    store.finish_run(rid, "ok", 1, 1)

    monkeypatch.setattr(pipeline, "generate_summary", lambda text: ("re", "local · test"))
    monkeypatch.setattr(pipeline, "send_telegram_message", lambda text, chat_ids=None: None)
    seen = {}
    monkeypatch.setattr(
        pipeline.embeddings, "index_file",
        lambda filename, transcript=None, summary=None: seen.update(
            filename=filename, transcript=transcript, summary=summary
        ),
    )

    pipeline.resummarize_file("nota.txt")

    assert seen == {"filename": "nota.txt", "transcript": None, "summary": "re"}


@pytest.mark.usefixtures("work_dirs")
def test_resummarize_indexing_failure_does_not_break_delivery(monkeypatch, read_log):
    """Same resilience as process_file: an indexing failure during /resummarize
    doesn't stop the (re)summary from being delivered."""
    transcript = pipeline.TRANSCRIPTIONS_DIR / "nota.txt"
    transcript.write_text("texto original", encoding="utf-8")
    rid = store.start_run("cron")
    store.record_file(rid, "nota.txt", "text", "ok", transcript_path=str(transcript))
    store.finish_run(rid, "ok", 1, 1)

    monkeypatch.setattr(pipeline, "generate_summary", lambda text: ("re", "local · test"))
    sent = _capture_sent(monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(pipeline.embeddings, "index_file", boom)

    result = pipeline.resummarize_file("nota.txt")

    assert result.status == "ok"
    assert len(sent) == 3
    assert "falló" in sent[2][0] and "nota.txt" in sent[2][0]
    assert "indexing nota.txt for RAG failed" in read_log()
