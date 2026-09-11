"""Video handling: extract audio, then transcribe it.

Transcription tries Groq's hosted ``whisper-large-v3`` first (skipped entirely
when no ``GROQ_API_KEY`` is configured) and falls back to the local
``whisper.cpp`` binary on any failure — network, rate limit, oversized audio,
or anything else. Every attempt and fallback is logged.
"""

from pathlib import Path

from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_AUDIO_TIMEOUT_SECONDS,
    GROQ_MAX_AUDIO_MB,
    GROQ_WHISPER_MODEL,
    TRANSCRIPTIONS_DIR,
    WHISPER_CLI,
    WHISPER_MODEL,
)
from drive import run
from utils import log


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


def extract_audio_compact(video_path, audio_path):
    """Extract mono 16 kHz MP3 audio, small enough to upload quickly.

    Used only for the Groq attempt — cuts a ~1 h meeting from ~108 MB (WAV)
    to ~14 MB, well under Groq's audio size limit and much faster to upload
    over a home connection.

    Args:
        video_path: Path to the source video.
        audio_path: Path where the ``.mp3`` file is written.
    """
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", "16000", "-ac", "1", "-b:a", "32k",
        str(audio_path), "-loglevel", "error",
    ])


def transcribe_local(audio_path, output_base):
    """Transcribe an audio file to text with the local ``whisper-cli`` binary.

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


def transcribe_groq(audio_path):
    """Transcribe an audio file with Groq's hosted Whisper.

    Args:
        audio_path: Path to the audio file to upload.

    Returns:
        The transcript text.

    Raises:
        Exception: Any failure from the Groq client (network, timeout, rate
            limit, oversized file, invalid key, ...). Callers are expected to
            fall back to :func:`transcribe_local`.
    """
    client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_AUDIO_TIMEOUT_SECONDS)
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(Path(audio_path).name, f.read()),
            model=GROQ_WHISPER_MODEL,
            temperature=0,
            response_format="verbose_json",
        )
    return response.text.strip()


def _try_groq_transcription(local_path, stem):
    """Attempt the Groq transcription path; return the text, or None to fall back."""
    if not GROQ_API_KEY:
        return None
    compact_audio = local_path.parent / f"{stem}.groq.mp3"
    try:
        extract_audio_compact(local_path, compact_audio)
        size_mb = compact_audio.stat().st_size / (1024 * 1024)
        if size_mb > GROQ_MAX_AUDIO_MB:
            log(f"Skipping Groq transcription: {size_mb:.1f} MB exceeds "
                f"GROQ_MAX_AUDIO_MB={GROQ_MAX_AUDIO_MB}.")
            return None
        log(f"Transcribing with Groq ({GROQ_WHISPER_MODEL}, {size_mb:.1f} MB)...")
        text = transcribe_groq(compact_audio)
        log("Groq transcription succeeded.")
        return text
    except Exception as e:  # noqa: BLE001  pylint: disable=broad-exception-caught
        log(f"Groq transcription failed ({e}); falling back to local whisper.cpp.")
        return None
    finally:
        compact_audio.unlink(missing_ok=True)


def process_video(filename, local_path):
    """Turn a downloaded video into its transcript text.

    Tries Groq's ``whisper-large-v3`` first; any failure (or no
    ``GROQ_API_KEY`` configured) falls back to the local ``whisper.cpp`` pass.
    Temporary audio files are always removed.

    Args:
        filename: Original file name, used to derive output names.
        local_path: Path to the downloaded video on disk.

    Returns:
        A ``(text, backend_label)`` tuple, e.g. ``(text, "Groq · whisper-large-v3")``
        or ``(text, "local · whisper.cpp (ggml-small)")``.

    Raises:
        RuntimeError: If Groq was unavailable/failed and local whisper exits
            cleanly but writes no transcript file.
    """
    stem = Path(filename).stem

    text = _try_groq_transcription(local_path, stem)
    if text is not None:
        return text, f"Groq · {GROQ_WHISPER_MODEL}"

    local_audio = local_path.parent / f"{stem}.wav"
    txt_output = TRANSCRIPTIONS_DIR / stem
    extract_audio(local_path, local_audio)
    try:
        transcribe_local(local_audio, txt_output)
        transcript = Path(f"{txt_output}.txt")
        if not transcript.is_file():
            raise RuntimeError(f"whisper produced no transcript at {transcript}")
        return transcript.read_text(encoding="utf-8"), f"local · whisper.cpp ({WHISPER_MODEL.stem})"
    finally:
        local_audio.unlink(missing_ok=True)
