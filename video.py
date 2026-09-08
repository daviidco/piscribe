"""Video handling: extract audio with ``ffmpeg`` and transcribe with ``whisper.cpp``."""

from pathlib import Path

from config import TRANSCRIPTIONS_DIR, WHISPER_CLI, WHISPER_MODEL
from drive import run


def extract_audio(video_path, audio_path):
    """Extract mono 16 kHz PCM WAV audio from a video file.

    The format (16 kHz, mono, signed 16-bit little-endian) is what
    ``whisper.cpp`` expects as input.

    Args:
        video_path: Path to the source video.
        audio_path: Path where the ``.wav`` file is written.
    """
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(audio_path), "-loglevel", "error",
    ])


def transcribe(audio_path, output_base):
    """Transcribe an audio file to text with the ``whisper-cli`` binary.

    Language is auto-detected. The transcript is written to
    ``{output_base}.txt``.

    Args:
        audio_path: Path to the ``.wav`` file to transcribe.
        output_base: Output path without extension; whisper appends ``.txt``.
    """
    run([
        str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-f", str(audio_path),
        "-l", "auto", "-otxt", "-of", str(output_base),
    ])


def process_video(filename, local_path):
    """Turn a downloaded video into its transcript text.

    Extracts the audio to a temporary ``.wav`` next to the video, transcribes it
    into ``TRANSCRIPTIONS_DIR`` and reads the transcript back. The temporary
    audio file is always removed, even if transcription fails.

    Args:
        filename: Original file name, used to derive output names.
        local_path: Path to the downloaded video on disk.

    Returns:
        The transcript as a string.

    Raises:
        RuntimeError: If ``whisper`` exits cleanly but writes no transcript file.
    """
    stem = Path(filename).stem
    local_audio = local_path.parent / f"{stem}.wav"
    txt_output = TRANSCRIPTIONS_DIR / stem

    extract_audio(local_path, local_audio)
    try:
        transcribe(local_audio, txt_output)
        transcript = Path(f"{txt_output}.txt")
        if not transcript.is_file():
            raise RuntimeError(f"whisper produced no transcript at {transcript}")
        return transcript.read_text(encoding="utf-8")
    finally:
        local_audio.unlink(missing_ok=True)
