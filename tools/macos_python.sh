#!/usr/bin/env bash
# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
# Also embedded verbatim in AMADEUS-site/AMADEUS-Setup.command so the
# downloaded, standalone installer can check Python before any downloads.

amadeus_check_python() {
    "$1" -I - <<'PY' >/dev/null 2>&1
import platform
import sys
assert sys.platform == 'darwin'
assert (3, 10) <= sys.version_info[:2] < (3, 13)
assert platform.machine() == 'arm64'
import tkinter as tk
assert tk.TclVersion == 8.6 and tk.TkVersion == 8.6
root = tk.Tk()
root.withdraw()
assert str(root.tk.call('info', 'patchlevel')).startswith('8.6.')
assert str(root.tk.call('package', 'require', 'Tk')).startswith('8.6.')
root.update()
root.destroy()
PY
}

amadeus_python_candidates() {
    local minor candidate
    for minor in 12 11 10; do
        printf '%s\n' "/Library/Frameworks/Python.framework/Versions/3.$minor/bin/python3.$minor"
        printf '%s\n' "/opt/homebrew/opt/python@3.$minor/bin/python3.$minor"
        command -v "python3.$minor" || true
    done
    candidate="$(command -v python3 || true)"
    # Apple's stub can open a Command Line Tools installation dialog.
    if [ "$candidate" != /usr/bin/python3 ]; then printf '%s\n' "$candidate"; fi
}

amadeus_open_python_installer() (
    # Clean up failures, but retain the pkg while Installer.app uses it.
    local temp_dir pkg digest
    temp_dir="$(mktemp -d)" || return 1
    trap 'rm -rf "$temp_dir"' EXIT
    pkg="$temp_dir/python.pkg"
    echo '[AMADEUS] Downloading Python 3.12.10 from python.org...'
    /usr/bin/curl --fail --location --proto '=https' --proto-redir '=https' \
        --retry 2 --connect-timeout 30 \
        https://www.python.org/ftp/python/3.12.10/python-3.12.10-macos11.pkg \
        --output "$pkg" || return 1
    # SHA-256 from the release's python.org .sigstore messageDigest.
    digest="$(/usr/bin/shasum -a 256 "$pkg")" || return 1
    if [ "${digest%% *}" != 8373e58da4ea146b3eb1c1f9834f19a319440b6b679b06050b1f9ee3237aa8e4 ]; then
        echo '[ERROR] Python package checksum mismatch; installation stopped.' >&2
        return 1
    fi
    /usr/sbin/pkgutil --check-signature "$pkg" || return 1
    /usr/sbin/spctl --assess --type install "$pkg" || return 1
    /usr/bin/open -a /System/Library/CoreServices/Installer.app "$pkg" || return 1
    trap - EXIT
    echo "[AMADEUS] Python package opened in Installer.app: $pkg"
    echo '[AMADEUS] Complete the Python installation in Installer.app, then run AMADEUS Setup again.'
    echo '[AMADEUS] Setup is exiting now. The package can be deleted after installation.'
)

# Returns 0 for a compatible Python, 1 for failure, or 2 to stop normally
# after cancellation or handing the package to Installer.app.
amadeus_ensure_macos_python() {
    local candidate existing
    case "${AMADEUS_VENV:?AMADEUS_VENV must be set}" in
        /*) ;;
        *) export AMADEUS_VENV="$PWD/$AMADEUS_VENV" ;;
    esac
    echo '[AMADEUS] Checking for compatible Apple Silicon Python and Tcl/Tk...'
    # An explicit selection or existing environment must never be silently replaced.
    if [ -n "${AMADEUS_PYTHON:-}" ]; then
        if ! amadeus_check_python "$AMADEUS_PYTHON"; then
            echo '[ERROR] AMADEUS_PYTHON is incompatible. Correct or unset it, then retry.' >&2
            return 1
        fi
        return 0
    fi
    existing="${AMADEUS_VENV:?AMADEUS_VENV must be set}/bin/python"
    if [ -e "$AMADEUS_VENV" ]; then
        if [ -x "$existing" ] && amadeus_check_python "$existing"; then
            export AMADEUS_PYTHON="$existing"
            return 0
        fi
        echo '[ERROR] Existing environment is incomplete or incompatible and has been preserved.' >&2
        echo '[ERROR] Set AMADEUS_VENV to a new directory, then retry setup.' >&2
        return 1
    fi
    while IFS= read -r candidate; do
        if [ -x "$candidate" ] && amadeus_check_python "$candidate"; then
            export AMADEUS_PYTHON="$candidate"
            echo "[AMADEUS] Using Python: $AMADEUS_PYTHON"
            return 0
        fi
    done < <(amadeus_python_candidates)

    echo 'No compatible Python was found (native arm64, Python 3.10-3.12, Tcl/Tk 8.6).'
    if ! /usr/bin/osascript -e 'display dialog "No compatible Python was found. Download the official Python 3.12.10 package from python.org and open it in Installer.app?\n\nInstaller.app will handle installation and any administrator authentication. It installs a system-wide Python and may replace an existing python.org 3.12 installation.\n\nAfter completing the Python installation, run AMADEUS Setup again." with title "AMADEUS Setup" buttons {"Cancel", "Download Python"} default button "Download Python" cancel button "Cancel"' >/dev/null; then
        echo '[AMADEUS] Python download was not approved; setup stopped.'
        return 2
    fi
    if ! amadeus_open_python_installer; then
        echo '[ERROR] Could not prepare or open the Python installer. Retry AMADEUS Setup.' >&2
        return 1
    fi
    return 2
}
