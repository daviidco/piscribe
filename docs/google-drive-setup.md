# Google Drive setup for `rclone`

`piscribe` reaches Google Drive only through [`rclone`](https://rclone.org/). This
guide creates your own Google OAuth client and configures an `rclone` remote for
it. Do this once, before running `install.sh`.

> **Why your own OAuth client?** `rclone` ships with a shared default client ID
> that is heavily rate-limited. For something that polls Drive 24/7 you want your
> own — it is free and takes about ten minutes.

The credentials produced here live in `rclone`'s own config file
(`~/.config/rclone/rclone.conf`), **not** in `piscribe`'s `.env`. The only
Drive-related values in `.env` are the remote name and the two folder names.

---

## 1. Create a Google Cloud project

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Project picker (top bar) → **New Project** → name it (e.g. `piscribe`) →
   **Create**, then select it.

## 2. Enable the Drive API

1. **APIs & Services → Library**.
2. Search for **Google Drive API** → open it → **Enable**.

## 3. Configure the OAuth consent screen

1. **APIs & Services → OAuth consent screen**.
2. **User type: External** → **Create**.
3. Fill the required fields: app name, user support email, developer contact
   email. Everything else can stay blank. **Save and continue**.
4. **Scopes**: nothing to add here — `rclone` requests the Drive scope itself.
   **Save and continue**.
5. **Test users**: add the Google account that owns the Drive. **Save and
   continue**.
6. Back on the OAuth consent screen, click **Publish app** → confirm, to move it
   from *Testing* to *In production*.

   > While the app is in *Testing*, refresh tokens expire after **7 days** and the
   > pipeline stops working until you re-authorize. *In production* they do not
   > expire. Google will show an "unverified app" warning during authorization —
   > that is expected for a personal app; click **Advanced → Go to piscribe
   > (unsafe)** to continue.

## 4. Create the OAuth client ID

1. **APIs & Services → Credentials → Create credentials → OAuth client ID**.
2. **Application type: Desktop app** → name it → **Create**.
3. Copy the **Client ID** and **Client secret**. You need both in the next step.

## 5. Run `rclone config`

Run this on any machine with `rclone` installed. The prompts:

```text
n) New remote
name> gdrive
Storage> drive                     # "Google Drive"
client_id> <paste your Client ID>
client_secret> <paste your Client secret>
scope> 1                           # 1 = full access ("drive")
service_account_file>              # leave blank
Edit advanced config? > n
Use web browser to automatically authenticate? >
```

- **On a desktop with a browser**: answer **y**. A browser tab opens; approve
  access (see the unverified-app note above).
- **On the Raspberry Pi (headless)**: answer **n**. `rclone` prints a command
  like `rclone authorize "drive" "..."`. Run *that* command on a desktop machine
  that has `rclone` and a browser; it prints a JSON token. Paste the token back
  into the Pi's prompt.

Then:

```text
Configure this as a Shared Drive (Team Drive)? > n
Keep this "gdrive" remote? > y
q) Quit config
```

## 6. Create the folders and test

```bash
rclone mkdir gdrive:pendings
rclone mkdir gdrive:processed
rclone lsd gdrive:            # should list both folders
```

## 7. Matching `.env` values

When `install.sh` prompts in step 5, these are the Drive-related answers:

| Variable | Value from this guide |
| --- | --- |
| `RCLONE_REMOTE` | `gdrive` (the remote name from step 5) |
| `PENDING_FOLDER` | `pendings` (folder from step 6) |
| `PROCESSED_FOLDER` | `processed` (folder from step 6) |

## Troubleshooting

- **`Error 403: access_denied`** — your account is not a Test user, or the app is
  in *Testing* and you did not publish it. See step 3.
- **Auth works, then breaks after ~a week** — app still in *Testing*; publish it
  (step 3.6) and re-run `rclone config` → `e` (edit remote) to re-authorize.
- **`Rate Limit Exceeded` / `userRateLimitExceeded`** — you are on the shared
  default client ID; make sure `client_id` and `client_secret` are set on the
  remote (`rclone config show gdrive`).
