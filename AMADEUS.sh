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

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

UV_EXE="$(find_uv || true)"

if [ -z "$UV_EXE" ]; then
    echo "[AMADEUS] uv not found; installing via the official installer..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "[ERROR] Neither curl nor wget is available, so uv cannot be installed automatically." >&2
        echo "[ERROR] Install curl (e.g. 'sudo apt install curl') and re-run, or install uv manually: https://docs.astral.sh/uv/" >&2
        exit 1
    fi
    UV_EXE="$(find_uv || true)"
fi

if [ -z "$UV_EXE" ]; then
    echo "[ERROR] uv could not be installed or found." >&2
    exit 1
fi

echo "[AMADEUS] Using uv: $UV_EXE"

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
