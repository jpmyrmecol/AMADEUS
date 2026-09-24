# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path


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
) -> None:
    backup_root = temporary / "backup"
    root_resolved = root.resolve()
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


def _required_uv_version(root: Path) -> str:
    in_tool_uv = False
    for raw_line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_tool_uv = line == "[tool.uv]"
            continue
        if not in_tool_uv:
            continue
        key, separator, value = line.partition("=")
        if key.strip() != "required-version" or not separator:
            continue
        match = re.fullmatch(r'"==([0-9]+(?:\.[0-9]+){1,3})"', value.strip())
        if match:
            return match.group(1)
        break
    raise RuntimeError("[tool.uv].required-version must be an exact == version pin.")


def _uv_version(executable: Path) -> str:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return ""
    fields = completed.stdout.strip().splitlines()
    parts = fields[0].split() if fields else []
    return parts[1] if completed.returncode == 0 and len(parts) > 1 else ""


def _find_pinned_uv(root: Path, required: str) -> Path | None:
    names = ("uv.exe", "uv") if os.name == "nt" else ("uv",)
    candidates = [root / ".uv" / name for name in names]
    candidates.extend(root / ".uv" / "bin" / name for name in names)
    on_path = shutil.which("uv")
    if on_path:
        candidates.append(Path(on_path))
    candidates.extend(Path.home() / ".local" / "bin" / name for name in names)
    candidates.extend(Path.home() / ".cargo" / "bin" / name for name in names)
    seen: set[str] = set()
    for candidate in candidates:
        identity = os.path.normcase(str(candidate.resolve(strict=False)))
        if identity in seen:
            continue
        seen.add(identity)
        if candidate.is_file() and _uv_version(candidate) == required:
            return candidate
    return None


def _bootstrap_uv(root: Path, required: str, log) -> Path:
    if os.name == "nt":
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if not powershell:
            raise RuntimeError("PowerShell is required to prepare the pinned uv version on Windows.")
        completed = _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(root / "tools" / "uv_bootstrap.ps1"),
                "-ProjectRoot",
                str(root),
            ],
            cwd=root,
            log=log,
        )
        resolved = Path(completed.stdout.strip().splitlines()[-1])
        if _uv_version(resolved) != required:
            raise RuntimeError(f"Could not prepare the required uv {required}.")
        return resolved

    installer_url = f"https://astral.sh/uv/{required}/install.sh"
    request = urllib.request.Request(installer_url, headers={"User-Agent": "AMADEUS-Updater"})
    with urllib.request.urlopen(request, timeout=60) as response:
        installer = response.read(2 * 1024 * 1024 + 1)
    if len(installer) > 2 * 1024 * 1024:
        raise RuntimeError("The pinned uv installer exceeded the 2 MiB safety limit.")

    environment = os.environ.copy()
    environment["UV_INSTALL_DIR"] = str(root / ".uv")
    environment["INSTALLER_NO_MODIFY_PATH"] = "1"
    _log(log, f"[AMADEUS] Installing pinned uv {required} into {root / '.uv'}")
    completed = subprocess.run(
        ["sh"],
        cwd=root,
        env=environment,
        input=installer.decode("utf-8"),
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
        raise RuntimeError(f"Installing the pinned uv {required} failed.")

    resolved = _find_pinned_uv(root, required)
    if resolved is None:
        raise RuntimeError(f"uv {required} could not be found after installation.")
    return resolved


def _setup_environment(root: Path, python: str, log) -> None:
    required = _required_uv_version(root)
    uv = _find_pinned_uv(root, required)
    if uv is None:
        _log(log, f"[AMADEUS] The required uv {required} is not installed; preparing it.")
        uv = _bootstrap_uv(root, required, log)

    command = [python, str(root / "tools" / "setup_environment.py"), "--uv", str(uv)]
    environment = os.environ.copy()
    environment.setdefault("UV_PROJECT_ENVIRONMENT", str(root / ".venv"))
    _log(log, "[AMADEUS] Syncing locked dependencies and checking the installed runtime.")
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Dependency setup exited with code {return_code}.")


def _restart(root: Path, python: str) -> None:
    environment = os.environ.copy()
    environment.pop("AMADEUS_SPLASH_TOKEN", None)
    environment.pop("AMADEUS_SPLASH_SECONDS", None)
    kwargs: dict[str, object] = {"cwd": str(root), "env": environment, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([python, "-m", "gui.gui_home"], **kwargs)


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

    log_path = temporary / "updater.log"
    records: list[tuple[Path, Path | None]] = []
    old_head: str | None = None
    environment_attempted = False
    failure: Exception | None = None
    rollback_ok = True

    with log_path.open("a", encoding="utf-8") as log:
        _log(log, "[AMADEUS] Waiting for the Home window to close.")
        _wait_for_process(args.parent_pid)
        try:
            if _is_git_checkout(root):
                _log(log, "[AMADEUS] Updating the clean main Git checkout.")
                old_head = _git_update(root, expected, log)
            else:
                _log(log, "[AMADEUS] Installing the verified source archive.")
                _apply_archive(root, staging, temporary, records)

            installed_version = (root / "VERSION").read_text(encoding="utf-8").strip()
            if installed_version != expected:
                raise RuntimeError("The installed source version does not match the version approved for installation.")

            environment_attempted = True
            _setup_environment(root, args.python, log)
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

            if rollback_ok and environment_attempted:
                try:
                    _log(log, "[AMADEUS] Restoring dependencies for the previous source version.")
                    _setup_environment(root, args.python, log)
                except Exception as rollback_error:
                    rollback_ok = False
                    _log(log, f"[ERROR] Could not restore the previous Python environment: {rollback_error}")

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

        _log(log, f"[AMADEUS] Updated to {_version_label(expected)}; restarting AMADEUS.")

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
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"[ERROR] AMADEUS updater could not start: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        _show_error(f"AMADEUS updater could not start.\n\n{exc}")
        raise SystemExit(1)
