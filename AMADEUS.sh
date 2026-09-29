#!/usr/bin/env bash
# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
# Set up and launch macOS / Ubuntu / WSL2; dependency logic is in tools/setup_environment.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_version_status() {
    if [ ! -f "$SCRIPT_DIR/VERSION" ]; then
        echo "[AMADEUS] Version: unavailable"
        return 0
    fi

    current="$(tr -d '[:space:]' < "$SCRIPT_DIR/VERSION")"
    echo "[AMADEUS] Version: v$current"

    latest=""
    if command -v curl >/dev/null 2>&1; then
        latest="$(curl -fsSL --connect-timeout 1 --max-time 2 \
            "https://raw.githubusercontent.com/jpmyrmecol/AMADEUS/main/VERSION" 2>/dev/null || true)"
    elif command -v wget >/dev/null 2>&1; then
        latest="$(wget -qO- --timeout=2 --tries=1 \
            "https://raw.githubusercontent.com/jpmyrmecol/AMADEUS/main/VERSION" 2>/dev/null || true)"
    fi
    latest="$(printf '%s' "$latest" | tr -d '[:space:]')"

    if [[ ! "$current" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
       [[ ! "$latest" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "[AMADEUS] Latest version: unavailable (offline or update check failed)"
        return 0
    fi

    IFS=. read -r current_major current_minor current_patch <<< "$current"
    IFS=. read -r latest_major latest_minor latest_patch <<< "$latest"
    if (( latest_major > current_major ||
          (latest_major == current_major && latest_minor > current_minor) ||
          (latest_major == current_major && latest_minor == current_minor && latest_patch > current_patch) )); then
        echo "[AMADEUS] Update available: v$current -> v$latest"
    elif (( latest_major == current_major && latest_minor == current_minor && latest_patch == current_patch )); then
        echo "[AMADEUS] Latest version: v$latest (up to date)"
    else
        echo "[AMADEUS] Latest version: v$latest (installed version is newer)"
    fi
}

show_version_status

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

echo "[AMADEUS] Using uv $UV_VERSION: $UV_EXE"

# The bootstrap interpreter only runs setup; setup selects and checks the GUI Python.
# AMADEUS_VENV explicitly selects an alternate environment. Do not infer it from activation.
if [ "$(uname -s)" = Darwin ]; then
    setup_python="$AMADEUS_PYTHON"
else
    setup_python=3.10
fi
if ! "$UV_EXE" run --no-project --python "$setup_python" python tools/setup_environment.py --uv "$UV_EXE"; then
    echo "[ERROR] AMADEUS environment setup failed. See the diagnostic above." >&2
    exit 1
fi
if [ ! -x "$AMADEUS_VENV/bin/amadeus" ]; then
    echo "[ERROR] The AMADEUS command was not found in $AMADEUS_VENV." >&2
    exit 1
fi

ensure_posix_command_path() {\n    system="$(uname -s)"\n    case "$system" in\n        Darwin|Linux) ;;\n        *) return 0 ;;\n    esac\n\n    command_dir="$HOME/.local/bin"\n    case ":${PATH}:" in\n        *":$command_dir:"*) return 0 ;;\n    esac\n\n    case "${SHELL:-}" in\n        */zsh) rc_file="$HOME/.zshrc" ;;\n        */bash)\n            if [ "$system" = Darwin ]; then\n                rc_file="$HOME/.bash_profile"\n            else\n                rc_file="$HOME/.bashrc"\n            fi\n            ;;\n        *)\n            if [ "$system" = Darwin ]; then\n                rc_file="$HOME/.zshrc"\n            else\n                rc_file="$HOME/.profile"\n            fi\n            ;;\n    esac\n    path_line='export PATH="$HOME/.local/bin:$PATH"'\n\n    if [ ! -f "$rc_file" ] || ! grep -Fqx "$path_line" "$rc_file"; then\n        {\n            printf '\n# AMADEUS user commands\n'\n            printf '%s\n' "$path_line"\n        } >> "$rc_file"\n        echo "[AMADEUS] Added $HOME/.local/bin to PATH in $rc_file."\n    fi\n\n    # Make the commands available to the remainder of this launcher too.\n    export PATH="$command_dir:$PATH"\n}\n\nensure_posix_command_path\necho "[AMADEUS] Environment is ready. Starting the GUI..."
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

echo "[AMADEUS] The GUI has closed."
echo "[AMADEUS] Type \"amadeus\" or \"amade\" from any terminal to start AMADEUS directly next time."
