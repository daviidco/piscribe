# Changelog

All notable changes to piscribe are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The current version lives in [`VERSION`](VERSION) and is shown by the bot's
`/version` command.

## [Unreleased]

### Added

- `/runcron [file]` — same as `/run`, but forces `PISCRIBE_TRIGGER=cron` so
  the run broadcasts its progress/result to every chat in `TG_CHAT_IDS`
  (`pipeline._targets`) instead of only to whoever asked, and shows up in
  `/history` indistinguishable from a real automated run. `/run` stays
  requester-only; use `/runcron` when the outcome matters to the whole team.
  The requester's id is still recorded (`PISCRIBE_BY`), just not surfaced by
  the trigger field.
- `/version` now shows `config.VERSION` (from the `VERSION` file) alongside
  the git SHA and configured models.
- `/recap` now signs the summary with which engine transcribed and
  summarized it (e.g. `_resumen: Groq · qwen/qwen3.8-27b_`), matching the
  signature already sent with the original Telegram delivery — previously
  this was only visible in that original message or by reading `/logs`.
- **Chunked Groq summarization**, retried when a single request gets rejected
  for exceeding Groq's free-tier output-tokens-per-minute (OTPM) limit
  (`summary.py`'s `_summarize_groq_chunked`, new `GROQ_CHUNK_CHARS`/
  `GROQ_CHUNK_MAX_COMPLETION_TOKENS` config). `generate_summary` first tries
  the normal single-shot concise prompt; only on a `groq.RateLimitError` —
  Groq's own signal that the request's estimated output is too large — does
  it retry with the transcript split into `GROQ_CHUNK_CHARS`-sized chunks,
  each summarized into terse notes with its own small output budget, then
  combined with one more Groq call into the final structured summary. (An
  earlier version of this pre-guessed whether to chunk from the transcript's
  character count; that guess turned out wrong in production — a transcript
  well under the guessed threshold was still rejected, because Groq's
  pre-flight estimate tracks content, not just length. Reacting to the actual
  rejection instead of predicting it sidesteps that entirely.) Any other Groq
  failure, or a chunked retry that also fails, falls back to local Ollama
  with the FULL original transcript, same as before — never a mix of partial
  Groq content and a local summary. This raises the ceiling on how much a
  single meeting can ask of Groq; it does not remove the per-minute cap,
  since enough chunks fired within the same minute can still exhaust it.
- **Chunk-level retry with backoff**: a single chunk (or the synthesis call)
  rejected for a transient OTPM `RateLimitError` now gets up to two retries
  with backoff (5s, then 15s — `_CHUNK_RETRY_DELAYS_SECONDS`) before the whole
  chunked attempt is aborted. OTPM is a per-minute budget, not a per-request
  wall, so a small chunk request rejected right now is quite likely to fit
  once the window has partially refreshed — previously any single chunk
  failure discarded every already-summarized chunk and fell all the way back
  to local (observed taking ~7 minutes), for what's often just a few seconds
  of transient contention. Any non-`RateLimitError` failure (missing headers,
  network error, ...) is still not retried, since waiting wouldn't fix those.
- **Per-call temperature**: chunk note-extraction keeps `temperature=0.3`, but
  the synthesis call (merging/deduplicating already-extracted notes rather
  than generating from the transcript) now uses `0.1` — more deterministic,
  favoring fidelity to what the chunks actually said over the synthesis call
  editorializing. Previously every Groq call shared the same hardcoded `0.3`.

### Changed

- The Groq required-section check (`_groq_chat`'s `require_headers`) now
  matches header text case-, heading-level-, and trailing-colon-insensitively
  (`## Puntos Clave`, `### puntos clave:`, and `## Puntos clave` all count) —
  it was an exact substring match before, so a model that wrote a perfectly
  complete response with a trivially different heading style (capitalization,
  a colon, `#` vs `##`) was wrongly flagged as having skipped the section and
  sent to a needless fallback.

- On-demand runs (`/run`, `/retry`, `/resummarize`) now send every message —
  progress, per-file failures, the summary itself, its signature — ONLY to
  the Telegram id that asked for them, instead of broadcasting to every chat
  in `TG_CHAT_IDS`. Cron is unaffected: with no requester of its own, it still
  broadcasts to everyone, and the wording is identical either way — only the
  audience changes. `telegram_api.send_telegram_message`/`send_document` gained
  an optional `chat_ids` parameter (default `None` = every configured chat) to
  make this possible; `pipeline._targets(trigger, requested_by)` decides which
  to use (a private Telegram chat shares its id with the user, so the
  requester's user id doubles as the target chat id).
- The engine signature (`_transcripción: ..._` / `_resumen: ..._`) is now
  sent as its own Telegram message, right after the summary, instead of
  appended to the same message — for both the original delivery and
  `/recap`. This way a very long summary that Telegram splits into several
  parts, or a summary whose Markdown fails to parse, can't take the
  signature down with it; the two are independent `sendMessage` calls.
- `/recap`, `/transcript`, `/retry` and `/resummarize`'s numeric argument is
  now a file's actual database id (the `#N` shown by `/find`) instead of a
  *position* counting back from the most recent file (1 = latest, 2 = second
  latest, ...) — the same confusion `/logs` had before it was switched to
  real run ids, and the same fix: `store.nth_file(n)` is gone, replaced by
  `store.file_by_id(file_id)` (plus a plain `store.latest_file()` for the
  no-argument case). `/help`'s command list and the README's table now spell
  out, up front, that `#id` is always a real database id — a run's from
  `/history`, a file's from `/find` — never a position, and that `/history`'s
  own `n` is an unrelated row count.

### Fixed

- **The bot was unresponsive to every other command while `/run`,
  `/runcron`, `/retry` or `/resummarize` was in progress — including
  `/cancel`, exactly when it's most needed.** `python-telegram-bot` defaults
  to `concurrent_updates=False`: it processes updates one at a time and
  won't even dequeue the next one until the current handler's coroutine
  fully returns, regardless of `_spawn_and_report` awaiting the spawned
  `pipeline.py` via a non-blocking `run_in_executor`. Since a run can take
  minutes, `/status`/`/help`/anything else — and especially `/cancel` for a
  run that's stuck or taking too long — would just sit queued until the run
  finished on its own. Fixed with `.concurrent_updates(True)` on the
  `Application.builder()` in `bot.py`. Verified safe: `store.py` opens a
  fresh SQLite connection per call (WAL mode, built for concurrent
  readers/writers) rather than sharing one, so there's no in-memory state a
  concurrent handler could corrupt; the one theoretical race (two
  `/run`-family commands both passing `_spawn_and_report`'s PID check before
  either spawns) is already covered by `pipeline.py`'s own cross-process
  file lock (`runlock.py`), which just makes the second process notice the
  lock is held and exit quietly.
- `/recap` sent its replies via `update.message.reply_text` with no
  `parse_mode`, so the `_transcripción: ..._`/`_resumen: ..._` signature (and
  any Markdown in the summary itself, like `**bold**`) showed up as literal
  underscores/asterisks instead of rendering — unlike the original pipeline
  delivery, which explicitly sends with `parse_mode="Markdown"`. Both of
  `/recap`'s replies now go through the same `to_telegram_markdown` step
  (new shared helper in `telegram_api.py`, extracted from
  `send_telegram_message`) and are sent with `parse_mode="Markdown"`, so a
  file looks the same whether you're reading its original delivery or
  pulling it back up later with `/recap`.
- **Critical**: `telegram_api._post` sent every outbound field (the summary
  text, its Markdown signature, a log's caption, ...) via curl's `-F`
  (multipart), which treats a bare `;` inside a value as the start of an
  extra parameter clause (`;type=...`, `;filename=...`) and silently
  truncates the field right there — no error anywhere, the run just finishes
  `ok`. Confirmed against a real curl invocation: `-F "text=a; b"` reaches
  the server as `text=a`, `b` gone without a trace. This is almost certainly
  the real cause behind several "the summary arrived cut off" reports this
  session that were chased down other paths (Groq's OTPM limit,
  missing-section detection, local's language drift) without success,
  because a semicolon in ordinary prose — extremely common in any real
  meeting summary — was truncating the message on its way out regardless of
  how complete the generated text actually was (confirmed via `/recap`,
  which reads the same stored summary through a different send path and
  printed it in full). Fixed by removing `curl`/`subprocess` from
  `telegram_api.py` entirely in favor of `httpx` (already a resolved
  dependency, now a direct one — see `requirements.txt`): plain fields go
  through `httpx`'s own form encoding, immune to this whole class of bug by
  construction, and the one real file upload (`send_document`) uses `httpx`'s
  `files=` with the content read into memory once, rather than a `curl -F
  @path` reference — both verified end-to-end against a local echo server
  with semicolons in every field, including the upload's caption (which an
  interim `-F`-for-uploads-only fix, since superseded, would not have
  covered either).
- The local fallback prompt (`_PROMPT_TEMPLATE`, used by Ollama's `qwen3:1.7b`
  when Groq is unavailable or rejects the request) now states the "always
  respond in Spanish" requirement twice — once up front, once again in the
  format section — after a real production run came back as a freeform
  English summary with the model's own headings, ignoring the single
  "profesional y español" mention that was there before. Observed on a long,
  dense transcript after Groq hit its OTPM rate limit and fell back to local.
- `/logs <n>` looked up the `n`-th most recent run instead of the run whose
  id is literally `n` — confusing, since `/history`, `/status`, the post-run
  report and every run's own log lines all display the literal id (e.g.
  `Run 31 started`). `/logs <id>` now looks up that exact id (`store.run_by_id`,
  replacing `run_by_offset`); `/logs` with no argument still means "the latest
  run".
- Both prompts now explicitly require ticket numbers to be written as one
  whole 4-digit number (e.g. `3619`), never split with a dot or a slash
  (`36.19`, `36/19`) — observed in a real summary.
- A Groq summary could come back visibly incomplete (missing entire sections)
  while Groq reported everything as normal. `_summarize_groq` now checks two
  independent signals instead of trusting `finish_reason` alone:
  - `completion_tokens` landing within 95% of `GROQ_MAX_COMPLETION_TOKENS`
    is treated as truncated even when `finish_reason="stop"` (the reasoning
    trace and the visible answer share one budget, so a squeezed answer
    isn't always reported as `"length"`).
  - The response is required to contain all five section headers the Groq
    prompt asks for; if one is missing, it's treated as incomplete and
    falls back to local Ollama. This is the one that actually caught the
    real production case — logs showed `finish_reason='stop'` with only
    510/8192 tokens used (plenty of budget left), so no token-based check
    could have caught it: the model was simply skipping sections on its own.
  Every attempt now logs its `finish_reason` and `completion_tokens` too,
  for easier diagnosis via `/logs`.

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
