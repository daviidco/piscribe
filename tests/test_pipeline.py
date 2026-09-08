"""Regression tests for the piscribe pipeline.

These mirror the scenarios from the manual end-to-end simulation: every external
tool (rclone, ffmpeg, whisper, Ollama, curl) is stubbed, so nothing here touches
the network or any path outside the test sandbox.
"""

import pytest

import drive
import pipeline
import telegram
import video


# ---------------------------------------------------------------------------
# telegram._split_message
# ---------------------------------------------------------------------------

def test_split_message_short_text_is_one_part():
    assert list(telegram._split_message("hola")) == ["hola"]


def test_split_message_respects_limit_and_keeps_content():
    text = "linea de prueba\n" * 800  # ~12.8k chars, newline every ~16
    parts = list(telegram._split_message(text, limit=4000))

    assert len(parts) > 1
    assert all(len(p) <= 4000 for p in parts)
    assert "\n".join(parts).replace("\n", "") == text.replace("\n", "")


def test_split_message_hard_cuts_a_line_with_no_newline():
    text = "x" * 9000
    parts = list(telegram._split_message(text, limit=4000))

    assert [len(p) for p in parts] == [4000, 4000, 1000]
    assert "".join(parts) == text


# ---------------------------------------------------------------------------
# drive.list_pending_files
# ---------------------------------------------------------------------------

def test_list_pending_files_filters_by_extension(monkeypatch):
    class FakeResult:
        stdout = "meeting.mp4\nnotes.txt\nreadme.md\nphoto.png\nsubdir/\n\n"

    monkeypatch.setattr(drive, "run", lambda cmd: FakeResult())

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
    monkeypatch.setattr(pipeline, "generate_summary", lambda text: f"SUMMARY<<{text.strip()}>>")
    monkeypatch.setattr(pipeline, "send_telegram_message", sent.append)
    return sent


def test_process_file_text_note(work_dirs, stub_pipeline):
    note = work_dirs.LOCAL_DIR / "nota.txt"
    note.write_text("acuerdos de la reunion", encoding="utf-8")

    pipeline.process_file("nota.txt")

    assert len(stub_pipeline) == 1
    assert "nota.txt" in stub_pipeline[0]
    assert "SUMMARY<<acuerdos de la reunion>>" in stub_pipeline[0]
    assert not note.exists()  # local copy cleaned up


def test_process_file_cleans_local_copy_when_a_later_step_fails(work_dirs, stub_pipeline, monkeypatch):
    note = work_dirs.LOCAL_DIR / "boom.txt"
    note.write_text("contenido", encoding="utf-8")

    def explode(_text):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(pipeline, "generate_summary", explode)

    with pytest.raises(RuntimeError, match="ollama down"):
        pipeline.process_file("boom.txt")

    assert not note.exists()  # finally-block cleanup still ran


# ---------------------------------------------------------------------------
# video.process_video
# ---------------------------------------------------------------------------

def test_process_video_reads_transcript_and_removes_wav(work_dirs, monkeypatch):
    monkeypatch.setattr(video, "extract_audio", lambda src, dst: None)

    def fake_transcribe(_audio_path, output_base):
        output_base.with_name(f"{output_base.name}.txt").write_text(
            "transcripcion simulada", encoding="utf-8"
        )

    monkeypatch.setattr(video, "transcribe", fake_transcribe)

    local_path = work_dirs.LOCAL_DIR / "clip.mp4"
    local_path.write_bytes(b"fake")

    assert video.process_video("clip.mp4", local_path) == "transcripcion simulada"
    assert not (work_dirs.LOCAL_DIR / "clip.wav").exists()


def test_process_video_raises_when_whisper_writes_nothing(work_dirs, monkeypatch):
    monkeypatch.setattr(video, "extract_audio", lambda src, dst: None)
    monkeypatch.setattr(video, "transcribe", lambda a, b: None)  # writes no .txt

    local_path = work_dirs.LOCAL_DIR / "clip.mp4"
    local_path.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="no transcript"):
        video.process_video("clip.mp4", local_path)


# ---------------------------------------------------------------------------
# pipeline.main
# ---------------------------------------------------------------------------

def test_main_continues_after_one_file_fails(work_dirs, monkeypatch, read_log):
    monkeypatch.setattr(pipeline, "list_pending_files", lambda: ["bad.txt", "good.txt"])
    handled = []

    def fake_process(name):
        handled.append(name)
        if name == "bad.txt":
            raise RuntimeError("kaboom")

    monkeypatch.setattr(pipeline, "process_file", fake_process)

    pipeline.main()  # must not raise

    assert handled == ["bad.txt", "good.txt"]
    assert "ERROR processing bad.txt: kaboom" in read_log()


def test_main_logs_listing_failure_without_traceback(work_dirs, monkeypatch, read_log):
    def boom():
        raise FileNotFoundError("[Errno 2] No such file or directory: 'rclone'")

    monkeypatch.setattr(pipeline, "list_pending_files", boom)

    pipeline.main()  # must not raise

    assert "ERROR listing pending files" in read_log()
