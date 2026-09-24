# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from tkinter import messagebox
from typing import Callable

try:
    from .project_paths import PROJECT_ROOT
except ImportError:  # Preserve direct execution with: python gui/gui_home.py
    from project_paths import PROJECT_ROOT


VERSION_URL = "https://raw.githubusercontent.com/jpmyrmecol/AMADEUS/main/VERSION"
ARCHIVE_URL = "https://github.com/jpmyrmecol/AMADEUS/archive/refs/heads/main.zip"
REQUEST_TIMEOUT_SECONDS = 30
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_UNPACKED_BYTES = 4 * 1024 * 1024 * 1024
VERSION_PATTERN = re.compile(
    r"v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z", re.IGNORECASE
)


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = VERSION_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Unsupported AMADEUS version format: {value!r}")
    return tuple(int(part) for part in match.groups())


def _version_label(value: str) -> str:
    return f"v{value.strip().removeprefix('v').removeprefix('V')}"


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": "AMADEUS-Updater", "Accept": "application/octet-stream"},
    )


def _fetch_latest_version() -> str:
    with urllib.request.urlopen(_request(VERSION_URL), timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = response.read(129)
    if len(payload) > 128:
        raise ValueError("GitHub returned an unexpectedly large VERSION file.")
    version = payload.decode("utf-8").strip()
    _version_tuple(version)
    return version


def _download_archive(archive_path: Path) -> None:
    with urllib.request.urlopen(_request(ARCHIVE_URL), timeout=REQUEST_TIMEOUT_SECONDS) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_ARCHIVE_BYTES:
            raise ValueError("The AMADEUS update archive exceeds the 2 GiB safety limit.")
        total = 0
        with archive_path.open("wb") as destination:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise ValueError("The AMADEUS update archive exceeds the 2 GiB safety limit.")
                destination.write(chunk)


def _extract_archive(archive_path: Path, staging_root: Path, expected_version: str) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        roots = {
            PurePosixPath(entry.filename).parts[0]
            for entry in entries
            if PurePosixPath(entry.filename).parts
        }
        if len(roots) != 1:
            raise ValueError("The downloaded archive does not have the expected single project folder.")

        archive_root = next(iter(roots))
        if PurePosixPath(archive_root).is_absolute():
            raise ValueError("The update archive contains an absolute file path.")
        if sum(entry.file_size for entry in entries) > MAX_UNPACKED_BYTES:
            raise ValueError("The AMADEUS update archive expands beyond the 4 GiB safety limit.")

        for entry in entries:
            name = entry.filename
            if "\\" in name or "\x00" in name:
                raise ValueError("The update archive contains an invalid file path.")
            parts = PurePosixPath(name).parts
            if not parts or parts[0] != archive_root:
                raise ValueError("The update archive contains a path outside the project folder.")
            if any(part in ("", ".", "..") or ":" in part for part in parts):
                raise ValueError("The update archive contains an unsafe file path.")
            if len(parts) == 1 or entry.is_dir():
                continue
            relative_parts = parts[1:]
            if relative_parts[0] in {".git", ".venv", ".uv", "__pycache__"}:
                continue

            mode = (entry.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(mode)
            if stat.S_ISLNK(mode) or file_type not in (0, stat.S_IFREG):
                raise ValueError("The update archive contains a non-regular file.")

            destination = staging_root.joinpath(*relative_parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            permissions = mode & 0o777
            if permissions:
                destination.chmod(permissions)

    required_files = (
        "VERSION",
        "pyproject.toml",
        "uv.lock",
        "gui/gui_home.py",
        "tools/setup_environment.py",
    )
    for required in required_files:
        if not (staging_root / required).is_file():
            raise ValueError(f"The update archive is missing {required}.")

    archive_version = (staging_root / "VERSION").read_text(encoding="utf-8").strip()
    if archive_version != expected_version:
        raise ValueError(
            "The GitHub version changed while the update was downloading. "
            "Please check for updates again."
        )


def _app_python() -> str:
    if os.name == "nt":
        candidate = Path(sys.prefix) / "Scripts" / "python.exe"
    else:
        candidate = Path(sys.prefix) / "bin" / "python"
    return str(candidate if candidate.is_file() else Path(sys.executable))


def start_update_check(
    root: tk.Misc,
    *,
    on_status: Callable[[str], None],
    on_update_start: Callable[[], None],
) -> None:
    """Check GitHub and, after confirmation, download and launch the updater."""
    if getattr(root, "_amadeus_update_busy", False):
        return
    root._amadeus_update_busy = True

    def status(text: str) -> None:
        try:
            on_status(text)
        except tk.TclError:
            pass

    def finish() -> None:
        try:
            root._amadeus_update_busy = False
            status("Update")
        except tk.TclError:
            pass

    def schedule(callback: Callable[[], None]) -> None:
        try:
            root.after(0, callback)
        except tk.TclError:
            pass

    try:
        current_version = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
        current_tuple = _version_tuple(current_version)
    except (OSError, UnicodeError, ValueError) as exc:
        messagebox.showerror("AMADEUS Update", f"Could not read the installed version.\n\n{exc}", parent=root)
        finish()
        return

    status("Checking...")

    def check_worker() -> None:
        try:
            latest_version = _fetch_latest_version()
        except Exception as exc:
            schedule(lambda error=exc: show_check_error(error))
            return
        schedule(lambda: show_update_choice(latest_version))

    def show_check_error(exc: Exception) -> None:
        try:
            messagebox.showerror(
                "AMADEUS Update",
                f"Could not check for updates. Please check your internet connection and try again.\n\n{exc}",
                parent=root,
            )
        finally:
            finish()

    def show_update_choice(latest_version: str) -> None:
        latest_tuple = _version_tuple(latest_version)
        if latest_tuple == current_tuple:
            messagebox.showinfo(
                "AMADEUS Update",
                f"AMADEUS is up to date. ({_version_label(current_version)})",
                parent=root,
            )
            finish()
            return
        if latest_tuple < current_tuple:
            messagebox.showinfo(
                "AMADEUS Update",
                f"This installation ({_version_label(current_version)}) is newer than the version on GitHub "
                f"({_version_label(latest_version)}). No update was made.",
                parent=root,
            )
            finish()
            return

        accepted = messagebox.askyesno(
            "AMADEUS Update",
            "A new version of AMADEUS is available.\n\n"
            f"{_version_label(current_version)} → {_version_label(latest_version)}\n\n"
            "Download and install this update now?",
            parent=root,
            default=messagebox.NO,
        )
        if not accepted:
            finish()
            return

        status("Downloading")
        threading.Thread(
            target=download_worker,
            args=(latest_version,),
            name="AMADEUS update download",
            daemon=True,
        ).start()

    def download_worker(latest_version: str) -> None:
        temporary_root: Path | None = None
        try:
            temporary_root = Path(tempfile.mkdtemp(prefix="amadeus-update-"))
            archive_path = temporary_root / "main.zip"
            staging_root = temporary_root / "source"
            staging_root.mkdir()
            _download_archive(archive_path)
            _extract_archive(archive_path, staging_root, latest_version)
        except Exception as exc:
            if temporary_root is not None:
                shutil.rmtree(temporary_root, ignore_errors=True)
            schedule(lambda error=exc: show_download_error(error))
            return

        schedule(lambda: launch_updater(temporary_root, staging_root, latest_version))

    def show_download_error(exc: Exception) -> None:
        try:
            messagebox.showerror(
                "AMADEUS Update",
                f"The update could not be downloaded or verified. AMADEUS has not been changed.\n\n{exc}",
                parent=root,
            )
        finally:
            finish()

    def launch_updater(temporary_root: Path, staging_root: Path, latest_version: str) -> None:
        log_path = temporary_root / "updater.log"
        python = _app_python()
        command = [
            python,
            "-m",
            "tools.update_worker",
            "--project-root",
            str(PROJECT_ROOT),
            "--staging-root",
            str(staging_root),
            "--temporary-root",
            str(temporary_root),
            "--expected-version",
            latest_version,
            "--parent-pid",
            str(os.getpid()),
            "--python",
            python,
        ]
        kwargs: dict[str, object] = {"cwd": str(PROJECT_ROOT), "close_fds": True}
        log_stream = log_path.open("a", encoding="utf-8")
        kwargs.update({"stdin": subprocess.DEVNULL, "stdout": log_stream, "stderr": subprocess.STDOUT})
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        try:
            subprocess.Popen(command, **kwargs)
        except OSError as exc:
            if log_stream is not None:
                log_stream.close()
            messagebox.showerror(
                "AMADEUS Update",
                f"The separate updater could not be started. AMADEUS has not been changed.\n\n{exc}",
                parent=root,
            )
            shutil.rmtree(temporary_root, ignore_errors=True)
            finish()
            return
        if log_stream is not None:
            log_stream.close()

        on_update_start()

    threading.Thread(target=check_worker, name="AMADEUS update check", daemon=True).start()
