#!/usr/bin/env bash
# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
# Set up and launch macOS / Ubuntu / WSL2; dependency logic is in tools/setup_environment.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
# uv resolves uv.lock, so its version is pinned like any other dependency:
# UV_VERSION is the single source of truth, shared with AMADEUS.bat, Colab and
# tools/setup_environment.py. A uv already installed on this machine is used
# only when it matches; otherwise AMADEUS installs its own copy under .uv and
# leaves the existing installation alone.
if [ ! -f "$SCRIPT_DIR/UV_VERSION" ]; then
    echo "[ERROR] UV_VERSION is missing from $SCRIPT_DIR; the AMADEUS folder is incomplete." >&2
    exit 1
fi
UV_VERSION="$(tr -d '[:space:]' < "$SCRIPT_DIR/UV_VERSION")"
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

case ":${PATH}:" in
    *":$HOME/.local/bin:"*) ;;
    *)
        case "${SHELL:-}" in
            */zsh) rc_file="~/.zshrc" ;;
            *) rc_file="~/.bashrc" ;;
        esac
        echo "[AMADEUS] NOTE: $HOME/.local/bin is not on your PATH."
        echo "[AMADEUS] Add it to use the \"amadeus\"/\"amade\" commands directly next time, e.g.:"
        echo "[AMADEUS]   echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> $rc_file && source $rc_file"
        ;;
esac

echo "[AMADEUS] Environment is ready. Starting the GUI..."
if ! "$AMADEUS_VENV/bin/amadeus" "$@"; then
    echo "[ERROR] AMADEUS exited with an error." >&2
    exit 1
fi

echo "[AMADEUS] The GUI has closed."
echo "[AMADEUS] Type \"amadeus\" or \"amade\" from any terminal to start AMADEUS directly next time."
