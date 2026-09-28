# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import atexit
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from gui.update_support import (
    adopt_update_lock,
    find_other_amadeus_processes,
    prepare_managed_update,
    release_update_lock,
    write_managed_state,
)


REPOSITORY_URL = "https://github.com/jpmyrmecol/AMADEUS.git"
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


def _log(log, message: str) -> None:
    print(message, flush=True)
    log.write(message + "\n")
    log.flush()


def _run(
    command: list[str],
    *,
    cwd: Path,
    log,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    _log(log, "$ " + subprocess.list2cmdline(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
        log.write(completed.stdout)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
        log.write(completed.stderr)
    log.flush()
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)
    return completed


def _wait_for_process(pid: int) -> None:
    if os.name == "nt":
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if handle:
            try:
                kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
                return
            finally:
                kernel32.CloseHandle(handle)

    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.25)


def _validate_paths(root: Path, staging: Path, temporary: Path) -> None:
    root_resolved = root.resolve()
    temporary_resolved = temporary.resolve()
    staging_resolved = staging.resolve()
    try:
        temporary_resolved.relative_to(Path(tempfile.gettempdir()).resolve())
        staging_resolved.relative_to(temporary_resolved)
    except ValueError as exc:
        raise ValueError("The updater received a path outside its temporary update folder.") from exc
    if temporary_resolved == root_resolved or root_resolved.is_relative_to(temporary_resolved):
        raise ValueError("The updater temporary folder cannot contain the AMADEUS installation.")


def _git_update(root: Path, expected_version: str, log) -> str:
    branch = _run(
        ["git", "-C", str(root), "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=root,
        log=log,
    ).stdout.strip()
    if branch != "main":
        raise RuntimeError("This Git installation is not on the main branch. Switch to main before updating.")

    changes = _run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        log=log,
    ).stdout.strip()
    if changes:
        raise RuntimeError(
            "This Git installation has local changes to project files. Commit or save those changes, then retry the update."
        )

    old_head = _run(["git", "-C", str(root), "rev-parse", "HEAD"], cwd=root, log=log).stdout.strip()
    _run(["git", "-C", str(root), "fetch", "--no-tags", REPOSITORY_URL, "main"], cwd=root, log=log)
    remote_version = _run(
        ["git", "-C", str(root), "show", "FETCH_HEAD:VERSION"],
        cwd=root,
        log=log,
    ).stdout.strip()
    if remote_version != expected_version:
        raise RuntimeError(
            "GitHub's version changed while the update was being prepared. Please check for updates again."
        )
    _run(["git", "-C", str(root), "merge", "--ff-only", "FETCH_HEAD"], cwd=root, log=log)
    return old_head


def _apply_archive(
    root: Path,
    staging: Path,
    temporary: Path,
    records: list[tuple[Path, Path | None]],
    expected_version: str,
) -> None:
    backup_root = temporary / "backup"
    root_resolved = root.resolve()
    managed_files = prepare_managed_update(root, staging, backup_root, records)
    for source in sorted(path for path in staging.rglob("*") if path.is_file()):
        relative = source.relative_to(staging)
        target = root / relative

        candidate = root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise RuntimeError(f"The update target contains a symbolic link: {relative}")
        try:
            target.resolve(strict=False).relative_to(root_resolved)
        except ValueError as exc:
            raise RuntimeError(f"The update target escapes the AMADEUS folder: {relative}") from exc
        if target.exists() and not target.is_file():
            raise RuntimeError(f"Cannot replace a non-file AMADEUS path: {relative}")

        backup: Path | None = None
        if target.is_file():
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        records.append((target, backup))

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_target = target.with_name(f".{target.name}.update-{os.getpid()}.tmp")
        try:
            shutil.copy2(source, temporary_target)
            os.replace(temporary_target, target)
        finally:
            temporary_target.unlink(missing_ok=True)

    write_managed_state(root, managed_files, expected_version, backup_root, records)


def _restore_archive(root: Path, records: list[tuple[Path, Path | None]], log) -> None:
    root_resolved = root.resolve()
    for target, backup in reversed(records):
        try:
            target.resolve(strict=False).relative_to(root_resolved)
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary_target = target.with_name(f".{target.name}.restore-{os.getpid()}.tmp")
                try:
                    shutil.copy2(backup, temporary_target)
                    os.replace(temporary_target, target)
                finally:
                    temporary_target.unlink(missing_ok=True)
        except Exception as exc:
            _log(log, f"[ERROR] Could not restore {target}: {exc}")
            raise


def _restart(root: Path, python: str) -> None:
    """Hand off to a fresh launcher after this updater has fully exited."""
    environment = os.environ.copy()
    environment.pop("AMADEUS_SPLASH_TOKEN", None)
    environment.pop("AMADEUS_SPLASH_SECONDS", None)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("PYTHONHOME", None)

    if os.name == "nt":
        root_literal = str(root).replace("'", "''")
        script = (
            f"$root='{root_literal}'; "
            "$lock=Join-Path $root '.amadeus-update.lock'; "
            "Write-Host '[AMADEUS] Source update completed. Waiting to restart...'; "
            "$deadline=(Get-Date).AddMinutes(10); "
            "while ((Test-Path -LiteralPath $lock) -and ((Get-Date) -lt $deadline)) "
            "{ Start-Sleep -Milliseconds 250 }; "
            "if (Test-Path -LiteralPath $lock) { "
            "Write-Host '[ERROR] The updater did not finish within 10 minutes.'; "
            "Read-Host 'Press Enter to close'; exit 1 }; "
            "Write-Host '[AMADEUS] Restarting updated AMADEUS...'; "
            "& (Join-Path $root 'AMADEUS.bat'); exit $LASTEXITCODE"
        )
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            cwd=root,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            close_fds=True,
        )
        return

    handoff_script = r"""
root="$1"
lock="$root/.amadeus-update.lock"
deadline=$((SECONDS + 600))
while [ -e "$lock" ] && (( SECONDS < deadline )); do
    sleep 0.25
done
if [ -e "$lock" ]; then
    exit 1
fi

if [ "$(uname -s)" = "Darwin" ]; then
    exec open -a Terminal "$root/AMADEUS.command"
fi

exec /bin/bash "$root/AMADEUS.sh"
"""
    subprocess.Popen(
        ["/bin/bash", "-c", handoff_script, "amadeus-update-handoff", str(root)],
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

def _show_error(message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        window = tk.Tk()
        window.withdraw()
        messagebox.showerror("AMADEUS Update", message, parent=window)
        window.destroy()
    except Exception:
        print(message, file=sys.stderr, flush=True)


def _is_git_checkout(root: Path) -> bool:
    return (root / ".git").exists()


def run(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    staging = Path(args.staging_root).resolve()
    temporary = Path(args.temporary_root).resolve()
    _validate_paths(root, staging, temporary)
    expected = args.expected_version
    _version_tuple(expected)
    staged_version = (staging / "VERSION").read_text(encoding="utf-8").strip()
    if staged_version != expected:
        raise RuntimeError("The staged source version did not match the version approved for installation.")

    adopt_update_lock(root, args.lock_token)
    atexit.register(release_update_lock, root, args.lock_token)

    log_path = temporary / "updater.log"
    records: list[tuple[Path, Path | None]] = []
    old_head: str | None = None
    failure: Exception | None = None
    rollback_ok = True

    with log_path.open("a", encoding="utf-8") as log:
        _log(log, "[AMADEUS] Waiting for the Home window to close.")
        _wait_for_process(args.parent_pid)
        try:
            running = find_other_amadeus_processes(root)
            if running:
                shown = ", ".join(f"{name} (PID {pid})" for pid, name in running[:6])
                if len(running) > 6:
                    shown += f", and {len(running) - 6} more"
                raise RuntimeError(
                    "Another AMADEUS process is still running after the Home window closed. "
                    f"Close it and retry the update. Detected: {shown}"
                )

            if _is_git_checkout(root):
                _log(log, "[AMADEUS] Updating the clean main Git checkout.")
                old_head = _git_update(root, expected, log)
            else:
                _log(log, "[AMADEUS] Installing the verified source archive.")
                _apply_archive(root, staging, temporary, records, expected)

            installed_version = (root / "VERSION").read_text(encoding="utf-8").strip()
            if installed_version != expected:
                raise RuntimeError("The installed source version does not match the version approved for installation.")

            _log(
                log,
                "[AMADEUS] Source update installed. A fresh platform launcher will "
                "prepare the updated environment after this updater exits.",
            )
        except Exception as exc:
            failure = exc
            _log(log, f"[ERROR] Update failed: {exc}")
            traceback.print_exc(file=log)
            log.flush()

            if old_head is not None:
                try:
                    changes = subprocess.run(
                        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=True,
                    ).stdout.strip()
                    if changes:
                        raise RuntimeError("The Git checkout changed during rollback; refusing to discard those changes.")
                    _run(["git", "-C", str(root), "reset", "--hard", old_head], cwd=root, log=log)
                except Exception as rollback_error:
                    rollback_ok = False
                    _log(log, f"[ERROR] Could not restore the previous Git source: {rollback_error}")
            elif records:
                try:
                    _restore_archive(root, records, log)
                except Exception:
                    rollback_ok = False

        if failure is not None:
            if rollback_ok:
                message = "The update failed. The previous source files were restored."
            else:
                message = (
                    "The update failed and automatic recovery was incomplete. "
                    "Run the AMADEUS launcher to repair the installation."
                )
            message += f"\n\nDetails: {failure}\n\nUpdate log: {log_path}"
            log.flush()
            _show_error(message)
            if rollback_ok:
                try:
                    _restart(root, args.python)
                except OSError as restart_error:
                    _show_error(
                        "The previous AMADEUS version was restored but could not be restarted.\n\n"
                        f"{restart_error}"
                    )
            return 1

        _log(
            log,
            f"[AMADEUS] Source updated to {_version_label(expected)}; handing off to a fresh launcher.",
        )

    try:
        _restart(root, args.python)
    except OSError as exc:
        _show_error(
            f"AMADEUS was updated to {_version_label(expected)}, but could not be restarted automatically.\n\n{exc}"
        )
        return 1

    shutil.rmtree(temporary, ignore_errors=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a user-approved AMADEUS update after the GUI exits.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--temporary-root", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--python", required=True, help="The Python executable used to restart AMADEUS.")
    parser.add_argument("--lock-token", required=True, help="Token for the installation update lock.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[ERROR] AMADEUS updater could not start: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        _show_error(f"AMADEUS updater could not start.\n\n{exc}")
        raise SystemExit(1)
