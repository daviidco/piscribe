# Telegram bot setup

`piscribe` uses one Telegram bot for **both** directions:

* **outbound** — the pipeline posts each summary to the chats in `TG_CHAT_IDS`
  (works with just a token, even without `bot.py` running);
* **inbound** — the optional `bot.py` daemon answers `/status`, `/recap`,
  `/run`, … from those same chats.

Do this once, before `install.sh` (or re-run `install.sh` afterwards to fill in
`.env`).

---

## 1. Create the bot

1. In Telegram, open a chat with [@BotFather](https://t.me/BotFather).
2. Send `/newbot`, pick a display name and a username ending in `bot`.
3. BotFather replies with a **token** like `8538051576:AAH...`. That is
   `TG_TOKEN`. Keep it secret — anyone with it controls the bot.

## 2. Register the command menu (optional but nice)

Still in BotFather: `/setcommands`, pick your bot, paste:

```text
status - estado y último run
recap - resumen del archivo n (default 1)
transcript - transcripción original (archivo)
logs - log de la ejecución n (o "errors")
history - últimas n ejecuciones
pending - archivos en la carpeta de Drive
stats - métricas de los últimos 7 días
find - buscar en resúmenes/archivos
run - ejecutar el pipeline ahora
retry - reprocesar desde processed
resummarize - rehacer solo el resumen
cancel - detener el run en curso
pause - pausar los runs
resume - reactivar los runs
version - versión y modelo
whoami - tu id de Telegram
```

Telegram will then show these in the `/` menu.

## 3. Find your chat id

`TG_CHAT_IDS` is a comma-separated list of the **user or chat ids** allowed to
receive summaries and use the bot.

- **Quick way:** message [@userinfobot](https://t.me/userinfobot); it replies
  with your numeric id.
- **From piscribe itself:** put any placeholder in `TG_CHAT_IDS`, start the bot,
  send it `/whoami` (that command answers anyone), copy the `user id`, then set
  the real value and restart.

For a **group**, add the bot to the group and use the group's id (starts with
`-`). Group ids also show via `/whoami`. If plain commands are ignored in a
group, disable the bot's privacy mode in BotFather (`/setprivacy` → Disable) or
address commands as `/status@yourbot`.

## 4. Put the values in `.env`

`install.sh` step 5 prompts for these; or edit `.env` directly:

```ini
TG_TOKEN=8538051576:AAH...
TG_CHAT_IDS=967342406,-1001234567890
```

## 5. Run the bot

`bot.py` must stay running to answer commands. Install it as a user service:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/piscribe-bot.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now piscribe-bot
loginctl enable-linger "$USER"   # keep it running without an active login
```

On startup the bot posts `🤖 piscribe-bot online (<sha>)` to every chat.

```bash
systemctl --user status piscribe-bot
journalctl --user -u piscribe-bot -f
systemctl --user restart piscribe-bot   # after a git pull
```

It uses **long polling** — an outbound connection held open to Telegram — so no
inbound port, no public URL, no TLS cert. Nothing to open on the router.

## 6. Security

- Only ids in `TG_CHAT_IDS` are answered; every other update is logged and
  ignored. `/whoami` is the sole exception (it has to work before you are on the
  list).
- **There is no admin tier** — everyone in `TG_CHAT_IDS` can use every command,
  `/run`, `/cancel` and `/pause` included. Use a private chat and a short list.
- Bot replies can contain meeting content (`/recap`, `/transcript`) and internal
  paths (`/logs`). The bot redacts the token from its output; nothing else.

## Troubleshooting

- **Bot never replies** — service not running (`systemctl --user status
  piscribe-bot`), or your id is not in `TG_CHAT_IDS` (check `journalctl` for
  `Unauthorized /...`).
- **`/whoami` works but nothing else** — you are not allowlisted; add the id it
  printed to `TG_CHAT_IDS` and `systemctl --user restart piscribe-bot`.
- **Summaries arrive but the bot is dead** — that is expected without `bot.py`;
  outbound delivery only needs `TG_TOKEN` + `TG_CHAT_IDS`.
- **`Conflict: terminated by other getUpdates request`** — two bot instances are
  polling the same token. Stop the extra one.
