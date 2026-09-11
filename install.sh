#!/usr/bin/env bash
# piscribe installer: sets up the virtualenv, runtime directories and .env.
# Run it from inside the cloned repository:
#   git clone <repo-url> ~/piscribe && cd ~/piscribe && bash install.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$HOME/whisper.cpp"
VENV_DIR="$REPO_DIR/.venv"
ENV_FILE="$REPO_DIR/.env"
ENV_EXAMPLE="$REPO_DIR/.env.example"

# Order of the variables written to .env.
ENV_VARS="TG_TOKEN TG_CHAT_IDS RCLONE_REMOTE PENDING_FOLDER PROCESSED_FOLDER QWEN_MODEL GROQ_API_KEY"
# Variables that have no sensible default and must be provided by the user.
ENV_REQUIRED=" TG_TOKEN TG_CHAT_IDS "
# Variables that may be left blank (blank = feature disabled, not "invalid").
# Kept in ENV_VARS (not just documented) so a re-run of this script doesn't wipe
# a value the user set by hand, since step 5 rewrites .env from ENV_VARS only.
ENV_OPTIONAL=" GROQ_API_KEY "

# Read the value of KEY from an env file (prints nothing if absent).
env_lookup() {
    local file="$1" key="$2"
    [ -f "$file" ] || return 0
    sed -n "s/^${key}=//p" "$file" | head -n1
}

# --- Step 1/6: update the code ------------------------------------------------
echo "==> Step 1/6: updating the repository"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only || echo "    skipped: could not fast-forward"
else
    echo "    skipped: not a git checkout"
fi

# --- Step 2/6: create the virtualenv ----------------------------------------
echo "==> Step 2/6: creating the virtualenv at $VENV_DIR"
if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "    python3-venv is missing. Install it with: sudo apt install python3-venv" >&2
    exit 1
fi
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade --quiet pip

# --- Step 3/6: install Python dependencies ---------------------------------
echo "==> Step 3/6: installing Python dependencies"
"$VENV_DIR/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt"

# --- Step 4/6: create runtime directories ---------------------------------
echo "==> Step 4/6: creating runtime directories under $WHISPER_DIR"
mkdir -p "$WHISPER_DIR/local_pending" "$WHISPER_DIR/transcriptions" "$WHISPER_DIR/runs"

# --- Step 5/6: configure .env --------------------------------------------------
echo "==> Step 5/6: configuring $ENV_FILE"
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.bak"
    echo "    existing .env backed up to $ENV_FILE.bak"
fi

# Ask for one variable and store the resolved value in ENV_ANSWER.
# Precedence for the pre-filled suggestion: current .env value, then (for
# required-or-defaulted variables) the .env.example default, otherwise no
# suggestion. Variables in ENV_OPTIONAL accept a blank answer (feature stays
# disabled); their .env.example placeholder is never offered as a default,
# since it would look like a real value. Returns non-zero if no interactive
# input is available.
ENV_ANSWER=""
ask_env_var() {
    local key="$1" current default suggestion label optional=false
    current="$(env_lookup "$ENV_FILE" "$key")"
    default="$(env_lookup "$ENV_EXAMPLE" "$key")"
    case "$ENV_OPTIONAL" in *" $key "*) optional=true ;; esac

    if [ -n "$current" ]; then
        suggestion="$current"
        label="  $key [Enter to keep current: $current]: "
    elif [ "$optional" = true ]; then
        suggestion=""
        label="  $key (optional, Enter to leave disabled): "
    elif [ -n "$default" ] && [ "${ENV_REQUIRED#*" $key "}" = "$ENV_REQUIRED" ]; then
        suggestion="$default"
        label="  $key [Enter for default: $default]: "
    else
        suggestion=""
        label="  $key (required): "
    fi

    while true; do
        if ! read -r -p "$label" ENV_ANSWER </dev/tty; then
            echo >&2
            echo "    aborted: no interactive input available" >&2
            return 1
        fi
        ENV_ANSWER="${ENV_ANSWER:-$suggestion}"
        if [ -n "$ENV_ANSWER" ] || [ "$optional" = true ]; then
            return 0
        fi
        echo "    a value is required, try again" >&2
    done
}

tmp_env="$(mktemp)"
trap 'rm -f "$tmp_env"' EXIT
for key in $ENV_VARS; do
    ask_env_var "$key" || exit 1
    printf '%s=%s\n' "$key" "$ENV_ANSWER" >> "$tmp_env"
done
mv "$tmp_env" "$ENV_FILE"
trap - EXIT

# --- Step 6/6: schedule the cron job ----------------------------------------
echo "==> Step 6/6: scheduling the cron job (Mon-Fri, 10:00-16:00, every 2h)"
cron_line="0 10-17/2 * * 1-5 cd $REPO_DIR && $VENV_DIR/bin/python pipeline.py >> $WHISPER_DIR/cron.log 2>&1 # piscribe"
if command -v crontab >/dev/null 2>&1; then
    { crontab -l 2>/dev/null | grep -Fv '# piscribe' || true; echo "$cron_line"; } | crontab -
    echo "    installed: $cron_line"
else
    echo "    crontab not found; add this line manually (crontab -e):" >&2
    echo "    $cron_line" >&2
fi

echo
echo "Done. Configuration written to $ENV_FILE"
echo
echo "Run a single pass now:      cd $REPO_DIR && $VENV_DIR/bin/python pipeline.py"
echo "Enable the Telegram bot:    see systemd/piscribe-bot.service (user service)"
