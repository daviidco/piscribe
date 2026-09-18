# piscribe — project instructions for Claude Code

## What this is

A Raspberry Pi pipeline that watches a Google Drive folder for meeting
recordings/notes, transcribes them (Groq first, local `whisper.cpp` fallback),
summarizes them in Spanish (Groq first, local Ollama fallback), and delivers
the result via Telegram. `bot.py`/`handlers.py` is a companion Telegram
control bot for querying and controlling the pipeline on demand.

- `pipeline.py` is a plain **synchronous** script, re-invoked fresh by cron
  or on demand (never long-running).
- `bot.py`/`handlers.py` is a long-lived **async** process
  (`python-telegram-bot`). It never runs the pipeline in-process — `/run`,
  `/retry`, `/resummarize` spawn `pipeline.py` as a subprocess with
  `PISCRIBE_*` env vars, exactly like cron would.
- `store.py` is SQLite (WAL mode) — a fresh connection per call, which is
  why concurrent bot updates (`concurrent_updates(True)` in `bot.py`) are
  safe: there's no shared connection object to race on.
- Cross-process serialization of actual pipeline runs is via `runlock.py`
  (`fcntl`-based file lock), independent of anything at the bot layer.
- App logs and everything in `store.py` use **UTC** explicitly
  (`utils.py`'s `formatter.converter = time.gmtime`, `store._utcnow()`).
  Cron and `rclone`'s own log lines use the Pi's **local** system clock
  (currently UTC−5). Keep this straight when reasoning about timestamps
  across the two.

## Conventions to follow

- **Never run `git add`/`commit`/`push` unless the user explicitly asks
  that specific time.** Approving one commit does not authorize the next.
- Every fix/feature goes through the same verification ritual before being
  reported done:
  1. `python3 -m py_compile *.py tests/*.py`
  2. `.venv/bin/pytest -q`
  3. `.venv/bin/pylint *.py tests/*.py` — must stay at **10.00/10**
     (`.pylintrc`: `max-args=10`, `max-attributes=10`, `max-statements=55`)
  4. A `CHANGELOG.md` entry under `## [Unreleased]` in Keep a Changelog
     format (`### Added`/`### Changed`/`### Fixed`).
- Default to **no code comments**. Only add one when the *why* is
  genuinely non-obvious (a hidden constraint, a subtle invariant, a
  workaround for a specific bug) — never to restate what the code does.
- `#id` (a file id from `/find`/`/history`, or a run id from `/history`) is
  always a real database id, never a position — don't reintroduce
  position-based (`nth_file`-style) lookups. IDs may be typed back with a
  leading `#` exactly as displayed (`handlers._parse_id` tolerates that).
- Deployment is manual and out of band: after a change is verified, remind
  the user to `git pull` + `systemctl --user restart piscribe-bot` (or the
  cron path, if relevant) on the Pi — never attempt this yourself.
- Prefer `httpx` over shelling out to `curl`/`subprocess` for HTTP calls —
  `telegram_api.py`'s earlier `curl -F` implementation had a real,
  hard-to-spot bug (a bare `;` in a field value silently truncated it).
