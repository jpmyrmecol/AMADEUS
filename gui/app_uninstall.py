# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Safe handoff for removing the complete AMADEUS installation folder."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from tkinter import messagebox
from typing import Callable

try:
    from .project_paths import PROJECT_ROOT
    from .update_support import terminate_other_amadeus_processes
except ImportError:
    from project_paths import PROJECT_ROOT
    from update_support import terminate_other_amadeus_processes


UNINSTALL_EXIT_CODE = 43


def _validate_project_root(root: Path) -> Path:
    root = root.resolve()
    if root == root.parent or root == Path.home().resolve():
        raise RuntimeError(f"Refusing to uninstall an unsafe path: {root}")
    required = (
        root / "VERSION",
        root / "pyproject.toml",
        root / "gui" / "gui_home.py",
        root / "main",
    )
    if not all(path.exists() for path in required):
        raise RuntimeError(
            "The AMADEUS installation folder could not be verified, so nothing was deleted."
        )
    try:
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not verify the AMADEUS installation: {exc}") from exc
    if 'name = "amadeus"' not in pyproject:
        raise RuntimeError(
            "The selected folder does not identify itself as the AMADEUS project."
        )
    return root


def _write_windows_uninstaller(project_root: Path) -> Path:
    script = Path(tempfile.gettempdir()) / f"AMADEUS-uninstall-{uuid.uuid4().hex}.ps1"
    root_literal = str(project_root).replace("'", "''")
    script.write_text(
        r"""param(
    [Parameter(Mandatory = $true)]
    [int]$ParentPid
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [IO.Path]::GetFullPath('""" + root_literal + r"""')
$HomePath = [IO.Path]::GetFullPath([Environment]::GetFolderPath("UserProfile"))
$DriveRoot = [IO.Path]::GetPathRoot($ProjectRoot)

function Test-AmadeusRoot([string]$Root) {
    if ($Root -eq $DriveRoot -or $Root -eq $HomePath) {
        throw "Refusing to uninstall an unsafe path: $Root"
    }
    $required = @(
        (Join-Path $Root "VERSION"),
        (Join-Path $Root "pyproject.toml"),
        (Join-Path $Root "gui\gui_home.py"),
        (Join-Path $Root "main")
    )
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "AMADEUS installation markers are missing: $path"
        }
    }
    $pyproject = Get-Content -Raw -LiteralPath (Join-Path $Root "pyproject.toml")
    if ($pyproject -notmatch '(?m)^name\s*=\s*"amadeus"\s*$') {
        throw "Target folder is not an AMADEUS installation."
    }
}

function Remove-WithRetry([string]$PathToRemove) {
    if (-not (Test-Path -LiteralPath $PathToRemove)) { return }
    $lastError = $null
    for ($i = 0; $i -lt 90; $i++) {
        try {
            Get-ChildItem -LiteralPath $PathToRemove -File -Recurse -Force -ErrorAction SilentlyContinue |
                ForEach-Object { try { $_.IsReadOnly = $false } catch {} }
            Remove-Item -LiteralPath $PathToRemove -Recurse -Force -ErrorAction Stop
        } catch {
            $lastError = $_
        }
        if (-not (Test-Path -LiteralPath $PathToRemove)) { return }
        Start-Sleep -Seconds 1
    }
    throw "Could not remove $PathToRemove. $lastError"
}

function Remove-AmadeusPathEntry([string]$CommandDir) {
    $current = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($null -eq $current) { return }
    $target = [IO.Path]::GetFullPath($CommandDir).TrimEnd("\")
    $kept = New-Object System.Collections.Generic.List[string]
    $changed = $false
    foreach ($raw in ($current -split ";")) {
        $entry = $raw.Trim()
        if (-not $entry) { continue }
        try {
            $expanded = [Environment]::ExpandEnvironmentVariables($entry)
            $normalized = [IO.Path]::GetFullPath($expanded).TrimEnd("\")
        } catch {
            $normalized = $entry.TrimEnd("\")
        }
        if ($normalized.Equals($target, [StringComparison]::OrdinalIgnoreCase)) {
            $changed = $true
            continue
        }
        $kept.Add($entry)
    }
    if ($changed) {
        $newPath = ($kept -join ";")
        if ($newPath) { $newPath += ";" }
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    }
}

try {
    Write-Host "[AMADEUS] Waiting for AMADEUS to close..."
    Wait-Process -Id $ParentPid -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 750

    Test-AmadeusRoot $ProjectRoot

    $LocalAmadeus = Join-Path $env:LOCALAPPDATA "AMADEUS"
    $CommandDir = Join-Path $LocalAmadeus "bin"
    Remove-AmadeusPathEntry $CommandDir

    $rootNorm = $ProjectRoot.TrimEnd("\")
    $localNorm = ([IO.Path]::GetFullPath($LocalAmadeus)).TrimEnd("\")
    $rootInsideLocal = $rootNorm.StartsWith(
        $localNorm + "\",
        [StringComparison]::OrdinalIgnoreCase
    ) -or $rootNorm.Equals($localNorm, [StringComparison]::OrdinalIgnoreCase)

    if ($rootInsideLocal) {
        Write-Host "[AMADEUS] Removing $LocalAmadeus ..."
        Remove-WithRetry $LocalAmadeus
    } else {
        Write-Host "[AMADEUS] Removing $LocalAmadeus ..."
        Remove-WithRetry $LocalAmadeus
        Write-Host "[AMADEUS] Removing $ProjectRoot ..."
        Remove-WithRetry $ProjectRoot
    }

    Write-Host ""
    Write-Host "[AMADEUS] Uninstall complete."
    Write-Host "[AMADEUS] The AMADEUS folder and installed command files were removed."
    Start-Sleep -Seconds 3
    exit 0
} catch {
    Write-Host ""
    Write-Host "[ERROR] AMADEUS uninstall failed."
    Write-Host "[ERROR] $($_.Exception.Message)"
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
} finally {
    try { Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue } catch {}
}
""",
        encoding="utf-8",
    )
    return script


def _write_posix_uninstaller(project_root: Path) -> Path:
    script = Path(tempfile.gettempdir()) / f"AMADEUS-uninstall-{uuid.uuid4().hex}.sh"
    root_literal = shlex.quote(str(project_root))
    script.write_text(
        """#!/usr/bin/env bash
set -u

PROJECT_ROOT=__AMADEUS_PROJECT_ROOT__
PARENT_PID="$1"

fail() {
    echo
    echo "[ERROR] AMADEUS uninstall failed."
    echo "[ERROR] $1"
    echo
    if [ -t 0 ]; then
        printf "Press Enter to close..."
        read -r _unused
    fi
    exit 1
}

echo "[AMADEUS] Waiting for AMADEUS to close..."
while kill -0 "$PARENT_PID" 2>/dev/null; do
    sleep 0.25
done
sleep 0.75

case "$PROJECT_ROOT" in
    ""|"/"|"$HOME")
        fail "Refusing to uninstall an unsafe path: $PROJECT_ROOT"
        ;;
esac

[ -f "$PROJECT_ROOT/VERSION" ] || fail "AMADEUS VERSION marker is missing."
[ -f "$PROJECT_ROOT/pyproject.toml" ] || fail "AMADEUS pyproject.toml is missing."
[ -f "$PROJECT_ROOT/gui/gui_home.py" ] || fail "AMADEUS Home GUI marker is missing."
[ -d "$PROJECT_ROOT/main" ] || fail "AMADEUS main directory is missing."
grep -Eq '^name[[:space:]]*=[[:space:]]*"amadeus"[[:space:]]*$' "$PROJECT_ROOT/pyproject.toml" \
    || fail "Target folder is not an AMADEUS installation."

echo "[AMADEUS] Removing installed amadeus/amade commands..."
rm -f -- "$HOME/.local/bin/amadeus" "$HOME/.local/bin/amade" \
    || fail "Could not remove the installed command files."

echo "[AMADEUS] Removing $PROJECT_ROOT ..."
chmod -R u+w "$PROJECT_ROOT" 2>/dev/null || true
rm -rf -- "$PROJECT_ROOT"
[ ! -e "$PROJECT_ROOT" ] || fail "Could not completely remove $PROJECT_ROOT."

echo
echo "[AMADEUS] Uninstall complete."
echo "[AMADEUS] The AMADEUS folder and installed command files were removed."
SELF="$0"
rm -f -- "$SELF" 2>/dev/null || true
sleep 3
exit 0
""".replace("__AMADEUS_PROJECT_ROOT__", root_literal),
        encoding="utf-8",
    )
    script.chmod(0o700)
    return script


def _launch_uninstaller(project_root: Path) -> None:
    temp_dir = str(Path(tempfile.gettempdir()).resolve())

    if os.name == "nt":
        powershell_script = _write_windows_uninstaller(project_root)
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(powershell_script),
                "-ParentPid",
                str(os.getpid()),
            ],
            cwd=temp_dir,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            close_fds=True,
        )
        return

    shell_script = _write_posix_uninstaller(project_root)
    command = ["/bin/bash", str(shell_script), str(os.getpid())]

    if sys.platform == "darwin":
        launcher = Path(tempfile.gettempdir()) / f"AMADEUS-uninstall-launch-{uuid.uuid4().hex}.command"
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            + "exec "
            + " ".join(shlex.quote(part) for part in command)
            + "\n",
            encoding="utf-8",
        )
        launcher.chmod(0o700)
        subprocess.Popen(
            ["open", "-a", "Terminal", str(launcher)],
            cwd=temp_dir,
            start_new_session=True,
            close_fds=True,
        )
        return

    terminals = (
        ("x-terminal-emulator", ["-e"]),
        ("gnome-terminal", ["--"]),
        ("konsole", ["-e"]),
        ("xterm", ["-e"]),
    )
    for executable, prefix in terminals:
        resolved = shutil.which(executable)
        if not resolved:
            continue
        subprocess.Popen(
            [resolved, *prefix, *command],
            cwd=temp_dir,
            start_new_session=True,
            close_fds=True,
        )
        return

    subprocess.Popen(
        command,
        cwd=temp_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def start_uninstall(root, *, on_uninstall_start: Callable[[], None]) -> None:
    try:
        project_root = _validate_project_root(PROJECT_ROOT)
    except Exception as exc:
        messagebox.showerror("Uninstall AMADEUS", str(exc), parent=root)
        return

    accepted = messagebox.askyesno(
        "Uninstall AMADEUS",
        "Permanently uninstall AMADEUS?\n\n"
        "The entire folder below will be deleted, including the virtual "
        "environment, scripts, Git metadata, and any other files stored inside it:\n\n"
        f"{project_root}\n\n"
        "Tracking sessions or result folders stored elsewhere are not deleted.\n\n"
        "Continue?",
        parent=root,
        default=messagebox.NO,
    )
    if not accepted:
        return

    try:
        _stopped, remaining = terminate_other_amadeus_processes(
            project_root,
            current_pid=os.getpid(),
        )
    except Exception as exc:
        messagebox.showerror(
            "Uninstall AMADEUS",
            f"Could not stop running AMADEUS processes. Nothing was deleted.\n\n{exc}",
            parent=root,
        )
        return

    if remaining:
        shown = "\n".join(f"- {name} (PID {pid})" for pid, name in remaining[:6])
        if len(remaining) > 6:
            shown += f"\n- ... and {len(remaining) - 6} more"
        messagebox.showerror(
            "Uninstall AMADEUS",
            "Some AMADEUS processes could not be stopped. Nothing was deleted.\n\n"
            f"Still running:\n{shown}",
            parent=root,
        )
        return

    try:
        _launch_uninstaller(project_root)
    except Exception as exc:
        messagebox.showerror(
            "Uninstall AMADEUS",
            f"Could not start the uninstaller. Nothing was deleted.\n\n{exc}",
            parent=root,
        )
        return

    on_uninstall_start()
