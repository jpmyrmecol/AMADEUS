# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path, PurePosixPath

import psutil


LOCK_FILENAME = ".amadeus-update.lock"
MANAGED_STATE_FILENAME = ".amadeus-managed-files.json"
BOOTSTRAP_MANIFEST = PurePosixPath("tools/update_manifest.txt")
MANAGED_SCHEMA = 1


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _lock_path(root: Path) -> Path:
    return root / LOCK_FILENAME


def _read_lock(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def acquire_update_lock(root: Path) -> str:
    """Reserve one installation for a single updater process."""
    root = root.resolve()
    path = _lock_path(root)
    token = secrets.token_hex(16)
    payload = json.dumps({"pid": os.getpid(), "token": token}, sort_keys=True).encode("utf-8")

    for _ in range(3):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            state = _read_lock(path)
            owner_pid = state.get("pid") if state else None
            if isinstance(owner_pid, int) and _pid_exists(owner_pid):
                raise RuntimeError("Another AMADEUS update is already running.")
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue

        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
        return token

    raise RuntimeError("Could not acquire the AMADEUS update lock.")


def adopt_update_lock(root: Path, token: str) -> None:
    """Transfer lock ownership from the Home process to the updater worker."""
    root = root.resolve()
    path = _lock_path(root)
    state = _read_lock(path)
    if not state or state.get("token") != token:
        raise RuntimeError("The AMADEUS update lock is missing or belongs to another updater.")

    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps({"pid": os.getpid(), "token": token}, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def release_update_lock(root: Path, token: str) -> None:
    path = _lock_path(root.resolve())
    state = _read_lock(path)
    if state and state.get("token") == token:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _path_within(value: str | os.PathLike[str] | None, root: Path) -> bool:
    if not value:
        return False
    try:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            return False
        candidate.resolve(strict=False).relative_to(root)
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _is_amadeus_runtime_process(
    name: str,
    executable: str | None,
    cmdline: list[str],
) -> bool:
    names = [name]
    if executable:
        names.append(Path(executable).name)
    if cmdline:
        names.append(Path(cmdline[0]).name)
    return any(
        value.lower().startswith(("python", "amadeus"))
        for value in names
        if value
    )


def find_other_amadeus_processes(
    project_root: Path,
    *,
    current_pid: int | None = None,
) -> list[tuple[int, str]]:
    """Return other Python/AMADEUS processes associated with this installation."""
    root = project_root.resolve()
    current_pid = os.getpid() if current_pid is None else current_pid
    matches: list[tuple[int, str]] = []

    for process in psutil.process_iter(["pid", "name", "exe", "cwd", "cmdline"]):
        try:
            info = process.info
            pid = int(info.get("pid") or 0)
            if pid <= 0 or pid == current_pid:
                continue
            name = str(info.get("name") or "process")
            executable = info.get("exe")
            cwd = info.get("cwd")
            cmdline = [str(part) for part in (info.get("cmdline") or []) if part]
        except (psutil.Error, OSError, ValueError, TypeError):
            continue

        if not _is_amadeus_runtime_process(name, executable, cmdline):
            continue
        if (
            _path_within(executable, root)
            or any(_path_within(argument, root) for argument in cmdline)
            or _path_within(cwd, root)
        ):
            matches.append((pid, name))

    matches.sort()
    return matches


def _normalize_relative_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid managed path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(
        part in ("", ".", "..") or ":" in part for part in path.parts
    ):
        raise ValueError(f"Invalid managed path: {value!r}")
    return path.as_posix()


def _state_path(root: Path) -> Path:
    return root / MANAGED_STATE_FILENAME


def read_managed_paths(root: Path) -> set[str] | None:
    """Read the exact source-file set managed by the previous archive update."""
    state_path = _state_path(root)
    if state_path.is_file():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read {MANAGED_STATE_FILENAME}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != MANAGED_SCHEMA:
            raise RuntimeError(f"Unsupported {MANAGED_STATE_FILENAME} format.")
        files = payload.get("files")
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise RuntimeError(f"Invalid {MANAGED_STATE_FILENAME} file list.")
        return {_normalize_relative_path(item) for item in files}

    bootstrap = root.joinpath(*BOOTSTRAP_MANIFEST.parts)
    if bootstrap.is_file():
        try:
            lines = bootstrap.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"Could not read {BOOTSTRAP_MANIFEST}: {exc}") from exc
        return {
            _normalize_relative_path(line.strip())
            for line in lines
            if line.strip()
        }
    return None


def staged_managed_paths(staging: Path) -> set[str]:
    return {
        _normalize_relative_path(source.relative_to(staging).as_posix())
        for source in staging.rglob("*")
        if source.is_file()
    }


def _validate_target(root: Path, relative: str) -> Path:
    relative_path = PurePosixPath(relative)
    target = root.joinpath(*relative_path.parts)
    root_resolved = root.resolve()

    candidate = root
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RuntimeError(f"The update target contains a symbolic link: {relative}")
    try:
        target.resolve(strict=False).relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"The update target escapes the AMADEUS folder: {relative}") from exc
    return target


def prepare_managed_update(
    root: Path,
    staging: Path,
    backup_root: Path,
    records: list[tuple[Path, Path | None]],
) -> set[str]:
    """Back up and remove only previously managed files absent from the new archive."""
    old_managed = read_managed_paths(root)
    new_managed = staged_managed_paths(staging)
    if old_managed is None:
        return new_managed

    for relative in sorted(old_managed - new_managed):
        if relative in {LOCK_FILENAME, MANAGED_STATE_FILENAME}:
            continue
        target = _validate_target(root, relative)
        if not target.exists():
            continue
        if not target.is_file():
            raise RuntimeError(f"Cannot remove a non-file AMADEUS path: {relative}")

        backup = backup_root / "deleted" / PurePosixPath(relative)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        records.append((target, backup))
        target.unlink()

    return new_managed


def write_managed_state(
    root: Path,
    managed_files: set[str],
    version: str,
    backup_root: Path,
    records: list[tuple[Path, Path | None]],
) -> None:
    """Persist the installed archive file set so later updates can remove obsolete files safely."""
    target = _state_path(root)
    if target.exists() and not target.is_file():
        raise RuntimeError(f"Cannot replace non-file {MANAGED_STATE_FILENAME}.")

    backup: Path | None = None
    if target.is_file():
        backup = backup_root / "state" / MANAGED_STATE_FILENAME
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
    records.append((target, backup))

    payload = {
        "schema": MANAGED_SCHEMA,
        "version": version,
        "files": sorted(_normalize_relative_path(item) for item in managed_files),
    }
    temporary = target.with_name(f".{target.name}.update-{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
