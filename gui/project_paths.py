# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import MutableMapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUI_DIR = PROJECT_ROOT / "gui"
MAIN_DIR = PROJECT_ROOT / "main"
ASSETS_DIR = PROJECT_ROOT / "assets"


def gui_asset(name: str) -> Path:
    """Locate an asset in the single shared assets directory."""
    return ASSETS_DIR / name


def gui_script(name: str) -> Path:
    return GUI_DIR / name


def main_script(name: str) -> Path:
    return MAIN_DIR / name


def ensure_import_paths(*directories: Path) -> None:
    """Add the script directories needed by directly executed GUIs."""
    for directory in reversed(directories):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)


def with_pythonpath(
    environment: MutableMapping[str, str],
    *directories: Path,
) -> MutableMapping[str, str]:
    """Prepend canonical paths to a subprocess environment's PYTHONPATH."""
    requested = [str(directory) for directory in directories]
    existing = environment.get("PYTHONPATH", "")
    if existing:
        requested.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(requested)
    return environment


def open_results_directory(session_path: str) -> Path:
    """Create and open the session results directory in the OS file manager.

    On Linux, ``xdg-open`` is preferred. If it is unavailable or cannot open
    the directory, ``explorer.exe`` is tried as a fallback for WSL2.
    """
    session = str(session_path or "").strip()
    if not session:
        raise ValueError("Session path is empty.")
    results = Path(session).expanduser() / "results"
    results.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(results))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(results)])
    else:
        xdg_open = shutil.which("xdg-open")
        if xdg_open:
            try:
                completed = subprocess.run(
                    [xdg_open, str(results)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5,
                )
                if completed.returncode == 0:
                    return results
            except (OSError, subprocess.TimeoutExpired):
                pass

        explorer = shutil.which("explorer.exe") or "explorer.exe"
        windows_path = str(results)
        wslpath = shutil.which("wslpath")
        if wslpath:
            try:
                converted = subprocess.run(
                    [wslpath, "-w", str(results)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=5,
                )
                if converted.returncode == 0 and converted.stdout.strip():
                    windows_path = converted.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            subprocess.Popen([explorer, windows_path])
        except OSError as exc:
            raise FileNotFoundError(
                "Could not open the results folder. Install xdg-utils or use "
                "WSL2 with explorer.exe available."
            ) from exc
    return results
