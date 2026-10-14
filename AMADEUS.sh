#!/usr/bin/env bash
# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
# Set up and launch macOS / Ubuntu / WSL2; dependency logic is in tools/setup_environment.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_version_status() {
    if [ ! -f "$SCRIPT_DIR/VERSION" ]; then
        return 0
    fi

    current="$(tr -d '[:space:]' < "$SCRIPT_DIR/VERSION")"

    latest_url=""
    if command -v curl >/dev/null 2>&1; then
        latest_url="$(curl -fsSIL -o /dev/null -w '%{url_effective}' \
            --connect-timeout 1 --max-time 2 \
            "https://github.com/jpmyrmecol/AMADEUS/releases/latest" 2>/dev/null || true)"
    elif command -v wget >/dev/null 2>&1; then
        latest_url="$(wget --spider --server-response --timeout=2 --tries=1 \
            "https://github.com/jpmyrmecol/AMADEUS/releases/latest" 2>&1 | \
            awk '/^[[:space:]]*Location:/ { url=$2; sub(/\r$/, "", url) } END { print url }' || true)"
    fi

    latest=""
    case "$latest_url" in
        https://github.com/jpmyrmecol/AMADEUS/releases/tag/*)
            latest="${latest_url#https://github.com/jpmyrmecol/AMADEUS/releases/tag/}"
            latest="${latest%%\?*}"
            latest="${latest%/}"
            latest="${latest#v}"
            latest="${latest#V}"
            ;;
    esac

    if [[ ! "$current" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
       [[ ! "$latest" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        return 0
    fi

    IFS=. read -r current_major current_minor current_patch <<< "$current"
    IFS=. read -r latest_major latest_minor latest_patch <<< "$latest"
    if (( latest_major > current_major ||
          (latest_major == current_major && latest_minor > current_minor) ||
          (latest_major == current_major && latest_minor == current_minor && latest_patch > current_patch) )); then
        echo "[AMADEUS] Update available: v$current -> v$latest"
    elif (( latest_major == current_major && latest_minor == current_minor && latest_patch == current_patch )); then
        :
    else
        echo "[AMADEUS] Latest release: v$latest (installed version is newer)"
    fi
}

show_version_status
echo "[AMADEUS] Location: $SCRIPT_DIR"

# Report unsupported hosts before downloading dependencies.
case "$(uname -s)" in
    Darwin)
        if [ "$(uname -m)" != "arm64" ]; then
            echo "[ERROR] This release requires native Apple Silicon on macOS (no Intel/Rosetta)." >&2
            exit 1
        fi
        ;;
    Linux)
        if [ -z "${DISPLAY:-}" ]; then
            echo "[ERROR] No graphical display found. Use a desktop session or WSLg, then retry." >&2
            exit 1
        fi
        ;;
    *)
        echo "[ERROR] Use this launcher on macOS or Linux; use AMADEUS.bat on Windows." >&2
        exit 1
        ;;
esac

export AMADEUS_VENV="${AMADEUS_VENV:-$SCRIPT_DIR/.venv}"
if [ "$(uname -s)" = Darwin ]; then
    source "$SCRIPT_DIR/tools/macos_python.sh"
    if amadeus_ensure_macos_python; then
        :
    else
        status=$?
        [ "$status" -eq 2 ] && exit 0
        exit "$status"
    fi
fi

# --- pinned uv (tests split AMADEUS.sh at this marker) ---------------------
# uv resolves uv.lock, so AMADEUS pins the uv executable separately in
# [tool.uv].required-version. pyproject.toml is the single source of truth shared
# by all launchers, setup, and Colab. An installed uv is used only when it matches;
# otherwise AMADEUS installs its own copy under .uv and leaves other installs alone.
if [ ! -f "$SCRIPT_DIR/pyproject.toml" ]; then
    echo "[ERROR] pyproject.toml is missing from $SCRIPT_DIR; the AMADEUS folder is incomplete." >&2
    exit 1
fi
UV_REQUIREMENT="$(
    awk '
        /^\[tool\.uv\][[:space:]]*$/ { in_uv = 1; next }
        /^\[/ { in_uv = 0 }
        in_uv && /^[[:space:]]*required-version[[:space:]]*=/ {
            value = $0
            sub(/^[^=]*=[[:space:]]*"/, "", value)
            sub(/"[[:space:]]*$/, "", value)
            print value
            exit
        }
    ' "$SCRIPT_DIR/pyproject.toml"
)"
case "$UV_REQUIREMENT" in
    ==?*) UV_VERSION="${UV_REQUIREMENT#==}" ;;
    *)
        echo "[ERROR] [tool.uv].required-version in pyproject.toml must be an exact == version pin." >&2
        exit 1
        ;;
esac
AMADEUS_UV_DIR="$SCRIPT_DIR/.uv"

uv_version_of() {
    # "uv 1.2.3 (abc1234 2026-01-01)" -> "1.2.3"
    [ -x "$1" ] || return 1
    "$1" --version 2>/dev/null | awk 'NR == 1 { print $2 }'
}

find_pinned_uv() {
    path_uv="$(command -v uv 2>/dev/null || true)"
    for candidate in "$AMADEUS_UV_DIR/bin/uv" "$AMADEUS_UV_DIR/uv" "$path_uv" \
                     "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        [ -n "$candidate" ] || continue
        if [ "$(uv_version_of "$candidate" || true)" = "$UV_VERSION" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

UV_EXE="$(find_pinned_uv || true)"

if [ -z "$UV_EXE" ]; then
    echo "[AMADEUS] Installing uv $UV_VERSION into $AMADEUS_UV_DIR ..."
    echo "[AMADEUS] Any other uv on this machine is left untouched."
    installer_url="https://astral.sh/uv/$UV_VERSION/install.sh"
    if command -v curl >/dev/null 2>&1; then
        fetch_installer() { curl -LsSf "$1"; }
    elif command -v wget >/dev/null 2>&1; then
        fetch_installer() { wget -qO- "$1"; }
    else
        echo "[ERROR] Neither curl nor wget is available, so uv cannot be installed automatically." >&2
        echo "[ERROR] Install curl (e.g. 'sudo apt install curl') and re-run, or install uv $UV_VERSION manually: https://docs.astral.sh/uv/" >&2
        exit 1
    fi
    # INSTALLER_NO_MODIFY_PATH keeps the user's shell profile and PATH as they are.
    if ! fetch_installer "$installer_url" |
        env UV_INSTALL_DIR="$AMADEUS_UV_DIR" INSTALLER_NO_MODIFY_PATH=1 sh; then
        echo "[ERROR] Installing uv $UV_VERSION failed. Check your network connection and retry." >&2
        exit 1
    fi
    UV_EXE="$(find_pinned_uv || true)"
fi

if [ -z "$UV_EXE" ]; then
    echo "[ERROR] uv $UV_VERSION could not be installed or found." >&2
    echo "[ERROR] Install it manually and re-run: https://docs.astral.sh/uv/" >&2
    exit 1
fi

# Use a persistent interpreter for setup. The setup procedure can replace
# .venv, so it must never run inside .venv or uv's ephemeral `run` environment.
setup_python=""
if [ "$(uname -s)" = Darwin ]; then
    # macos_python.sh has already validated the requested Python and Tk.
    setup_python="$("$AMADEUS_PYTHON" -c 'import sys; print(sys._base_executable)')"
else
    for minor in 12 11 10; do
        for candidate in "/usr/bin/python3.$minor" "/usr/local/bin/python3.$minor" \
                         "$(command -v "python3.$minor" || true)"; do
            [ -n "$candidate" ] && [ -x "$candidate" ] || continue
            resolved="$("$candidate" -c 'import sys; print(sys._base_executable)' 2>/dev/null || true)"
            case "$resolved" in
                */builds-v0/*|*/.tmp*|*/tmp/*) continue ;;
            esac
            if [ -n "$resolved" ] && [ -x "$resolved" ]; then
                setup_python="$resolved"
                break 2
            fi
        done
    done
    if [ -z "$setup_python" ]; then
        # uv-managed Python installations are persistent; `uv run` build
        # environments are not. Never select a Python from .cache/uv/builds-v0.
        setup_python="$("$UV_EXE" python find --system 3.12)" || {
            echo "[ERROR] No stable Python 3.10-3.12 could be found for setup." >&2
            exit 1
        }
    fi
fi

if [ ! -x "$setup_python" ]; then
    echo "[ERROR] Setup Python is unavailable: $setup_python" >&2
    exit 1
fi
echo "[AMADEUS] Setup Python (persistent): $setup_python"
if ! env -u VIRTUAL_ENV "$setup_python" tools/setup_environment.py --uv "$UV_EXE"; then
    echo "[ERROR] AMADEUS environment setup failed. See the diagnostic above." >&2
    exit 1
fi
if [ ! -x "$AMADEUS_VENV/bin/amadeus" ]; then
    echo "[ERROR] The AMADEUS command was not found in $AMADEUS_VENV." >&2
    exit 1
fi

ensure_posix_command_path() {
    system="$(uname -s)"
    case "$system" in
        Darwin|Linux) ;;
        *) return 0 ;;
    esac

    command_dir="$HOME/.local/bin"
    case ":${PATH}:" in
        *":$command_dir:"*) return 0 ;;
    esac

    case "${SHELL:-}" in
        */zsh) rc_file="$HOME/.zshrc" ;;
        */bash)
            if [ "$system" = Darwin ]; then
                rc_file="$HOME/.bash_profile"
            else
                rc_file="$HOME/.bashrc"
            fi
            ;;
        *)
            if [ "$system" = Darwin ]; then
                rc_file="$HOME/.zshrc"
            else
                rc_file="$HOME/.profile"
            fi
            ;;
    esac
    path_line='export PATH="$HOME/.local/bin:$PATH"'

    if [ ! -f "$rc_file" ] || ! grep -Fqx "$path_line" "$rc_file"; then
        {
            printf '\n# AMADEUS user commands\n'
            printf '%s\n' "$path_line"
        } >> "$rc_file"
        echo "[AMADEUS] Added $HOME/.local/bin to PATH in $rc_file."
    fi

    # Make the commands available to the remainder of this launcher too.
    export PATH="$command_dir:$PATH"
}

ensure_posix_command_path
set +e
"$AMADEUS_VENV/bin/amadeus" "$@"
status=$?
set -e

if (( status == 42 )); then
    echo "[AMADEUS] Update accepted. The updater will restart AMADEUS automatically."
    exit 0
fi
if (( status == 43 )); then
    echo "[AMADEUS] Uninstall accepted. A separate terminal will remove AMADEUS."
    exit 0
fi
if (( status != 0 )); then
    echo "[ERROR] AMADEUS exited with an error." >&2
    exit "$status"
fi

echo "[AMADEUS] GUI closed. Type \"amadeus\" or \"amade\" to reopen."
