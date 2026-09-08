# piscribe

A lightweight pipeline that watches a Google Drive folder for meeting recordings or
notes, transcribes them locally, generates a concise Spanish summary with a local
LLM, and delivers the result to Telegram. Built to run unattended from cron on a
Raspberry Pi.

## Overview

`piscribe` is designed to run unattended on a Raspberry Pi (e.g. from a cron job).
On each run it:

1. Lists the files waiting in a Google Drive "pending" folder via [`rclone`](https://rclone.org/).
2. Downloads each file locally and moves it to a "processed" folder in Drive so it
   is not handled twice.
3. Extracts the content:
   - **Video** (`.mp4`, `.mov`, `.mkv`, `.avi`) &rarr; audio is extracted with
     `ffmpeg` and transcribed with [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp).
   - **Text** (`.txt`, `.md`) &rarr; read directly.
4. Sends the transcript to a local [Qwen](https://ollama.com/library/qwen3) model
   served by [Ollama](https://ollama.com/) and asks for a clear, concise summary in
   Spanish highlighting key points, decisions, and open items.
5. Posts the summary to one or more Telegram chats through the Bot API.
6. Cleans up local temp files and appends progress to a log file.

Everything runs on your machine — transcription and summarization never leave the
host.

## Features

- **Drive-driven queue** — drop a file in a folder, get a summary back. No UI needed.
- **Local transcription** with `whisper.cpp` and automatic language detection.
- **Local summarization** with a small Qwen model via Ollama (no API keys, no cloud).
- **Always-Spanish summaries** regardless of the source language.
- **Multi-recipient Telegram delivery** with Markdown formatting.
- **Idempotent processing** — files are moved to a processed folder as soon as they
  are picked up.
- **Resilient batch runs** — a failure on one file is logged and does not stop the rest.
- **Timestamped logging** to both stdout and `~/whisper.cpp/log.txt`.

## Project layout

| File | Responsibility |
| --- | --- |
| [pipeline.py](pipeline.py) | Entry point; orchestrates the full pipeline |
| [config.py](config.py) | Paths and environment configuration |
| [drive.py](drive.py) | `rclone` wrappers: list, download, move files in Drive |
| [video.py](video.py) | Audio extraction (`ffmpeg`) and transcription (`whisper.cpp`) |
| [text.py](text.py) | Reads plain-text / Markdown inputs |
| [summary.py](summary.py) | Summary generation via Ollama / Qwen |
| [telegram.py](telegram.py) | Sends messages through the Telegram Bot API |
| [utils.py](utils.py) | Timestamped logging helper |
| [install.sh](install.sh) | Sets up the virtualenv, runtime dirs, and `.env` |
| [requirements.txt](requirements.txt) / [requirements-dev.txt](requirements-dev.txt) | Pinned runtime / test dependencies |
| [tests/](tests/) | `pytest` suite (external tools stubbed) |
| [docs/google-drive-setup.md](docs/google-drive-setup.md) | One-time `rclone` + Google Drive OAuth setup |

## Requirements

- Python 3.9+ with `python3-venv` (`sudo apt install python3-venv` on Raspberry Pi OS);
  developed and tested on **Python 3.13.5**
- Python packages from [requirements.txt](requirements.txt) (`python-dotenv`, `ollama`),
  installed into a virtualenv by `install.sh`
- [`git`](https://git-scm.com/)
- [`rclone`](https://rclone.org/) with a Google Drive remote — see
  [docs/google-drive-setup.md](docs/google-drive-setup.md)
- [`ffmpeg`](https://ffmpeg.org/)
- `curl` (used to call the Telegram Bot API)
- `cmake` and a C++ toolchain (to build `whisper.cpp`)
- [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp) built at `~/whisper.cpp/`
  with a model at `~/whisper.cpp/models/ggml-small.bin`
- The [Ollama](https://ollama.com/) service installed and running, with the Qwen
  model pulled (this is the native Ollama daemon/CLI, separate from the `ollama`
  Python package the pipeline uses to talk to it)
- A Telegram bot token (from [@BotFather](https://t.me/BotFather)) and the target chat IDs

## Setup

The external tools (`rclone`, `whisper.cpp`, Ollama) are set up once by hand;
`install.sh` then handles the Python side and the `.env`.

1. **Configure `rclone`** with a Google Drive remote. Follow
   [docs/google-drive-setup.md](docs/google-drive-setup.md) — it creates the
   Google OAuth client, runs `rclone config`, and creates the `pendings` /
   `processed` folders in Drive.

2. **Build `whisper.cpp`** and download a model:

   ```bash
   git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp
   cd ~/whisper.cpp && cmake -B build && cmake --build build --config Release
   sh ./models/download-ggml-model.sh small
   ```

3. **Install Ollama and pull the Qwen model.** This is the native Ollama
   service, not a Python package — no virtualenv involved:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh   # installs and starts the service
   ollama pull qwen3:1.7b
   ```

4. **Clone this project and run the installer:**

   ```bash
   git clone <repo-url> ~/piscribe
   cd ~/piscribe
   bash install.sh
   ```

   `install.sh` runs five steps: it fast-forwards the checkout, creates a
   virtualenv at `.venv/`, installs [requirements.txt](requirements.txt) into it,
   creates the runtime directories under `~/whisper.cpp/`, and then **prompts for
   each `.env` value**. For each variable it offers the current value (if `.env`
   already exists) or the default from `.env.example`; press Enter to accept it,
   or type a new value. An existing `.env` is backed up to `.env.bak` first.

   | Variable | Description |
   | --- | --- |
   | `TG_TOKEN` | Telegram bot token from BotFather (required) |
   | `TG_CHAT_IDS` | Comma-separated chat IDs to deliver summaries to (required) |
   | `RCLONE_REMOTE` | Name of the configured `rclone` remote (e.g. `gdrive`) |
   | `PENDING_FOLDER` | Drive folder to watch for new files |
   | `PROCESSED_FOLDER` | Drive folder that processed files are moved to |
   | `QWEN_MODEL` | Ollama model name used for summarization (e.g. `qwen3:1.7b`) |

   > `install.sh` only touches the virtualenv, `~/whisper.cpp/`'s runtime
   > directories, and `.env`. The code stays in the checkout; re-run the script
   > any time to update dependencies or reconfigure `.env`.

## Usage

`pipeline.py` imports the other modules as top-level modules and `config.py`
loads `.env` from the current directory, so always run it from the checkout with
the virtualenv's Python.

Run a single pass over the pending folder:

```bash
cd ~/piscribe && .venv/bin/python pipeline.py
```

Schedule it (for example, every 15 minutes) with cron — use absolute paths:

```cron
*/15 * * * * cd /home/pi/piscribe && /home/pi/piscribe/.venv/bin/python pipeline.py >> ~/whisper.cpp/cron.log 2>&1
```

Then just upload a recording or a note to the Drive `pendings` folder and wait for
the summary to arrive in Telegram.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The suite in [tests/](tests/) stubs every external tool (rclone, ffmpeg, whisper,
Ollama, curl) and runs against a throwaway sandbox directory, so it needs no
network, no Drive, and no models. It covers the message splitter, the extension
filter, local-file cleanup on failure, the transcript guard, and batch
resilience.

## How it works

```
Google Drive (pendings/)
        │  rclone lsf / copy / moveto
        ▼
  local download  ──►  video? ──► ffmpeg ──► whisper.cpp ──► transcript
        │                text? ─────────────────────────────► file contents
        ▼
   Ollama + Qwen  ──►  Spanish summary
        │
        ▼
   Telegram Bot API  ──►  chat(s)
```

## Notes & limitations

- Only `.mp4`, `.mov`, `.mkv`, `.avi`, `.txt`, and `.md` files are picked up; anything
  else in the folder is ignored.
- The summary prompt and output language are hard-coded to Spanish in
  [summary.py](summary.py).
- There is no locking; run one instance at a time.
- Secrets live in `.env`, which is git-ignored. Rotate any token that has been
  committed or shared.
