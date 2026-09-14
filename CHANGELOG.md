# Changelog

All notable changes to piscribe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The current version lives in [`VERSION`](VERSION) and is shown by the bot's
`/version` command.

## [Unreleased]

## [1.0.0] - 2026-09-14

First tagged version. Reconstructed from the project's git history rather than
tracked release-by-release, so it summarizes everything shipped so far as a
single baseline.

### Added

- Google Drive-driven pipeline: watches a "pending" folder via `rclone`,
  downloads new files, moves them to "processed", and never handles the same
  file twice.
- Video transcription (`.mp4`, `.mov`, `.mkv`, `.avi`) via
  [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp); text notes
  (`.txt`, `.md`) are read directly.
- Structured, always-Spanish meeting summaries (synthesis, key points,
  decisions with owner/deadline, risks, open items) via a
  [Qwen](https://ollama.com/library/qwen3) model served by
  [Ollama](https://ollama.com/), attributing statements to a speaker only when
  the transcript makes that identifiable.
- Telegram delivery of the summary to one or more chats, with Markdown
  formatting and automatic splitting of messages over Telegram's length limit.
- Telegram control bot (`bot.py`, `handlers.py`) with query commands
  (`/status`, `/recap`, `/transcript`, `/logs`, `/history`, `/pending`,
  `/stats`, `/find`, `/version`, `/whoami`) and control commands (`/run`,
  `/retry`, `/resummarize`, `/cancel`, `/pause`, `/resume`), restricted to an
  allowlist of chat ids.
- SQLite-backed run history (`store.py`) recording every run and file outcome,
  queryable from Telegram.
- Cross-process run lock (`runlock.py`) so cron and a bot-triggered `/run`
  never overlap; `/cancel` signals the in-progress run via its PID.
- Dead-man's-switch alert: the bot pings the configured chats if no run has
  succeeded within `DEADMAN_HOURS`, and again once it recovers.
- Groq-first, local-fallback architecture for both transcription (hosted
  `whisper-large-v3`) and summarization (hosted Qwen chat model): tries Groq
  first and falls back automatically to `whisper.cpp` / Ollama on any failure
  (missing key, network error, rate limit, oversized file, truncated
  response, ...), or runs fully local if no `GROQ_API_KEY` is configured.
- Engine provenance: every summary is signed with which backend transcribed
  and summarized it (e.g. `_resumen: Groq · qwen/qwen3.8-27b_` vs `_resumen:
  local · Ollama qwen3:1.7b_`), and recorded in the SQLite history.
- A dedicated, shorter Groq-only summary prompt so dense meetings stay within
  Groq's free-tier output-tokens-per-minute limit, while the local path keeps
  the original, more detailed prompt.
- UTC, level-aware logging (`utils.py`) via Python's stdlib `logging`: a
  rotating shared log, a per-run log file, and `/logs errors` to filter a
  run's log down to just its `WARNING`/`ERROR` lines. Per-run log files are
  pruned automatically after `RUN_LOG_RETENTION_DAYS` (default 30).
- Live Telegram progress messages for every pipeline pass, cron or manual
  `/run` alike: the run starting, each file's download/transcription/summary
  stage, an immediate notice if a file fails, and a wrap-up with the ok/total
  count — so a failure is never silent and both triggers read the same way.
- `install.sh`: a git-clone + venv installer with an interactive, idempotent
  `.env` prompt and automatic cron scheduling (Mon-Fri, every 2h, 10:00-16:00).
- systemd user service for the Telegram bot
  (`systemd/piscribe-bot.service`).
- Full pytest suite (`tests/`) running against an isolated sandbox
  `$HOME`/config so nothing touches the real Raspberry Pi paths or the
  network, plus a `.pylintrc` kept at a clean 10.00/10.
- `VERSION` file and `CHANGELOG.md` as the project's version of record,
  surfaced by `/version`.

### Changed

- Secrets (`TG_TOKEN`, `GROQ_API_KEY`) are now redacted centrally in
  `telegram_api.send_telegram_message`, so any outbound message — including
  the newer error notifications — can't leak a credential.
- `/run`'s Telegram feedback now comes entirely from the pipeline's own
  progress messages (identical to a cron run) instead of a separate bot-side
  announce/report, so the two no longer duplicate each other.
- Cron's stdout is no longer duplicated into `cron.log` on top of the shared
  log file; only stderr (a crash before logging is even configured) lands
  there now.

### Fixed

- Groq-transcribed files are archived to `TRANSCRIPTIONS_DIR` just like local
  transcriptions, so `/transcript` and `/resummarize` work regardless of which
  backend transcribed the file.
- Groq summaries are no longer silently truncated: a response with
  `finish_reason == "length"` now falls back to the local model instead of
  being delivered incomplete.
- A test-suite log-truncation bug where `RotatingFileHandler`'s persistently
  open file descriptor kept writing into an unlinked inode once a test
  fixture deleted the log file out from under it, silently losing every
  subsequent test's log output.
