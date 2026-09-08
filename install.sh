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
ENV_VARS="TG_TOKEN TG_CHAT_IDS RCLONE_REMOTE PENDING_FOLDER PROCESSED_FOLDER QWEN_MODEL"
# Variables that have no sensible default and must be provided by the user.
ENV_REQUIRED=" TG_TOKEN TG_CHAT_IDS "

# Read the value of KEY from an env file (prints nothing if absent).
env_lookup() {
    local file="$1" key="$2"
    [ -f "$file" ] || return 0
    sed -n "s/^${key}=//p" "$file" | head -n1
}

# --- Step 1/5: update the code ------------------------------------------------
echo "==> Step 1/5: updating the repository"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only || echo "    skipped: could not fast-forward"
else
    echo "    skipped: not a git checkout"
fi

# --- Step 2/5: create the virtualenv ----------------------------------------
echo "==> Step 2/5: creating the virtualenv at $VENV_DIR"
if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "    python3-venv is missing. Install it with: sudo apt install python3-venv" >&2
    exit 1
fi
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade --quiet pip

# --- Step 3/5: install Python dependencies ---------------------------------
echo "==> Step 3/5: installing Python dependencies"
"$VENV_DIR/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt"

# --- Step 4/5: create runtime directories ---------------------------------
echo "==> Step 4/5: creating runtime directories under $WHISPER_DIR"
mkdir -p "$WHISPER_DIR/local_pending" "$WHISPER_DIR/transcriptions"

# --- Step 5/5: configure .env --------------------------------------------------
echo "==> Step 5/5: configuring $ENV_FILE"
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.bak"
    echo "    existing .env backed up to $ENV_FILE.bak"
fi

# Ask for one variable and store the resolved value in ENV_ANSWER.
# Precedence for the pre-filled suggestion: current .env value, then .env.example
# default (unless the variable is required), otherwise no suggestion.
# Returns non-zero if no interactive input is available.
ENV_ANSWER=""
ask_env_var() {
    local key="$1" current default suggestion label
    current="$(env_lookup "$ENV_FILE" "$key")"
    default="$(env_lookup "$ENV_EXAMPLE" "$key")"

    if [ -n "$current" ]; then
        suggestion="$current"
        label="  $key [Enter to keep current: $current]: "
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
        [ -n "$ENV_ANSWER" ] && return 0
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

echo
echo "Done. Configuration written to $ENV_FILE"
echo "Run a single pass with:"
echo "  cd $REPO_DIR && $VENV_DIR/bin/python pipeline.py"
