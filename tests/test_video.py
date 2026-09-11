"""Tests for video.py: Groq-first transcription, local whisper.cpp fallback."""

import video


def _writes_bytes(size):
    """Fake extract_audio_compact(src, dst) that writes `size` placeholder bytes."""
    return lambda _src, dst: dst.write_bytes(b"x" * size)


def test_groq_transcription_used_when_it_succeeds(work_dirs, monkeypatch):
    """Groq succeeds: its text is used directly, local whisper never runs."""
    monkeypatch.setattr(video, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(video, "extract_audio_compact", _writes_bytes(100))
    monkeypatch.setattr(video, "transcribe_groq", lambda audio_path: "transcripcion de groq")

    def boom(*_a, **_kw):
        raise AssertionError("local whisper must not run when Groq succeeds")

    monkeypatch.setattr(video, "extract_audio", boom)
    monkeypatch.setattr(video, "transcribe_local", boom)

    local_path = work_dirs.LOCAL_DIR / "clip.mp4"
    local_path.write_bytes(b"fake")

    text, backend = video.process_video("clip.mp4", local_path)

    assert text == "transcripcion de groq"
    assert backend == "Groq · whisper-large-v3"


def test_groq_failure_falls_back_to_local(work_dirs, monkeypatch, read_log):
    """Any Groq exception falls back to local whisper.cpp; the reason is logged."""
    monkeypatch.setattr(video, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(video, "extract_audio_compact", _writes_bytes(100))

    def raise_conn_error(_audio_path):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(video, "transcribe_groq", raise_conn_error)
    monkeypatch.setattr(video, "extract_audio", lambda src, dst: None)

    def fake_local(_audio_path, output_base):
        output_base.with_name(f"{output_base.name}.txt").write_text(
            "transcripcion local", encoding="utf-8"
        )

    monkeypatch.setattr(video, "transcribe_local", fake_local)

    local_path = work_dirs.LOCAL_DIR / "clip.mp4"
    local_path.write_bytes(b"fake")

    text, backend = video.process_video("clip.mp4", local_path)

    assert text == "transcripcion local"
    assert backend == "local · whisper.cpp (ggml-small)"
    log_text = read_log()
    assert "Groq transcription failed" in log_text
    assert "network unreachable" in log_text


def test_no_api_key_skips_groq_entirely(work_dirs, monkeypatch):
    """Without GROQ_API_KEY (the sandbox default) Groq is never attempted."""
    assert video.GROQ_API_KEY == ""

    def boom(*_a, **_kw):
        raise AssertionError("the Groq path must not run without an API key")

    monkeypatch.setattr(video, "extract_audio_compact", boom)
    monkeypatch.setattr(video, "transcribe_groq", boom)
    monkeypatch.setattr(video, "extract_audio", lambda src, dst: None)

    def fake_local(_audio_path, output_base):
        output_base.with_name(f"{output_base.name}.txt").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(video, "transcribe_local", fake_local)

    local_path = work_dirs.LOCAL_DIR / "clip.mp4"
    local_path.write_bytes(b"fake")

    text, backend = video.process_video("clip.mp4", local_path)

    assert text == "ok"
    assert backend.startswith("local ·")


def test_oversized_compressed_audio_skips_groq(work_dirs, monkeypatch, read_log):
    """A compressed file over GROQ_MAX_AUDIO_MB is never uploaded."""
    monkeypatch.setattr(video, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(video, "GROQ_MAX_AUDIO_MB", 0.0001)  # ~100 bytes
    monkeypatch.setattr(video, "extract_audio_compact", _writes_bytes(10000))

    def boom(_audio_path):
        raise AssertionError("must not upload an oversized file to Groq")

    monkeypatch.setattr(video, "transcribe_groq", boom)
    monkeypatch.setattr(video, "extract_audio", lambda src, dst: None)

    def fake_local(_audio_path, output_base):
        output_base.with_name(f"{output_base.name}.txt").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(video, "transcribe_local", fake_local)

    local_path = work_dirs.LOCAL_DIR / "clip.mp4"
    local_path.write_bytes(b"fake")

    text, backend = video.process_video("clip.mp4", local_path)

    assert text == "ok"
    assert backend.startswith("local ·")
    assert "Skipping Groq transcription" in read_log()


def test_groq_attempt_cleans_up_compact_audio(work_dirs, monkeypatch):
    """The temporary compressed mp3 is removed after a successful Groq attempt."""
    created = {}

    def fake_extract_compact(_src, dst):
        dst.write_bytes(b"x" * 10)
        created["path"] = dst

    monkeypatch.setattr(video, "GROQ_API_KEY", "fake-key")
    monkeypatch.setattr(video, "extract_audio_compact", fake_extract_compact)
    monkeypatch.setattr(video, "transcribe_groq", lambda audio_path: "listo")

    local_path = work_dirs.LOCAL_DIR / "clip.mp4"
    local_path.write_bytes(b"fake")

    video.process_video("clip.mp4", local_path)

    assert not created["path"].exists()
