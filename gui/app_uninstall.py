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
import textwrap
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


def _base_python() -> str:
    project_root = PROJECT_ROOT.resolve()
    candidates = [
        Path(getattr(sys, "_base_executable", "") or ""),
        Path(shutil.which("python3") or ""),
        Path(shutil.which("python") or ""),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError:
            return str(resolved)
    raise RuntimeError(
        "No Python interpreter outside the AMADEUS folder is available to complete uninstall."
    )


def _write_external_uninstaller() -> Path:
    script = Path(tempfile.gettempdir()) / f"AMADEUS-uninstall-{uuid.uuid4().hex}.py"
    script.write_text(
        textwrap.dedent(
            r'''
            import ctypes
            import os
            import shutil
            import sys
            import time
            from pathlib import Path

            def validate(root: Path) -> Path:
                root = root.resolve()
                if root == root.parent or root == Path.home().resolve():
                    raise RuntimeError(f"Unsafe uninstall path: {root}")
                required = (
                    root / "VERSION",
                    root / "pyproject.toml",
                    root / "gui" / "gui_home.py",
                    root / "main",
                )
                if not all(path.exists() for path in required):
                    raise RuntimeError("AMADEUS installation markers are missing.")
                pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
                if 'name = "amadeus"' not in pyproject:
                    raise RuntimeError("Target folder is not an AMADEUS installation.")
                return root

            def wait_for_pid(pid: int) -> None:
                if os.name == "nt":
                    from ctypes import wintypes
                    kernel32 = ctypes.windll.kernel32
                    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                    kernel32.OpenProcess.restype = wintypes.HANDLE
                    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                    handle = kernel32.OpenProcess(0x00100000, False, pid)
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

            def main() -> int:
                root = validate(Path(sys.argv[1]))
                parent_pid = int(sys.argv[2])
                print("[AMADEUS] Waiting for AMADEUS to close...", flush=True)
                wait_for_pid(parent_pid)
                time.sleep(0.75)
                print(f"[AMADEUS] Removing {root}", flush=True)

                last_error = None
                for _ in range(90):
                    try:
                        shutil.rmtree(root)
                    except FileNotFoundError:
                        last_error = None
                        break
                    except OSError as exc:
                        last_error = exc
                        time.sleep(1.0)
                        continue
                    last_error = None
                    break

                if root.exists():
                    print("", flush=True)
                    print("[ERROR] AMADEUS could not be completely removed.", flush=True)
                    print(f"[ERROR] Close any program or terminal still using: {root}", flush=True)
                    if last_error is not None:
                        print(f"[ERROR] {last_error}", flush=True)
                    input("Press Enter to close...")
                    return 1

                print("", flush=True)
                print("[AMADEUS] Uninstall complete.", flush=True)
                print("[AMADEUS] The AMADEUS folder and its contents were removed.", flush=True)
                try:
                    Path(__file__).unlink()
                except OSError:
                    pass
                time.sleep(3.0)
                return 0

            if __name__ == "__main__":
                raise SystemExit(main())
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    return script


def _launch_uninstaller(script: Path, project_root: Path) -> None:
    python = _base_python()
    command = [python, str(script), str(project_root), str(os.getpid())]
    temp_dir = str(Path(tempfile.gettempdir()).resolve())

    if os.name == "nt":
        quoted = subprocess.list2cmdline(command)
        subprocess.Popen(
            ["cmd.exe", "/c", quoted],
            cwd=temp_dir,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            close_fds=True,
        )
        return

    if sys.platform == "darwin":
        launcher = Path(tempfile.gettempdir()) / f"AMADEUS-uninstall-{uuid.uuid4().hex}.command"
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            + "status=0\n"
            + " ".join(shlex.quote(part) for part in command)
            + " || status=$?\n"
            + 'rm -f -- "$0"\n'
            + 'exit "$status"\n',
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
        if resolved:
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

    script = None
    try:
        script = _write_external_uninstaller()
        _launch_uninstaller(script, project_root)
    except Exception as exc:
        messagebox.showerror(
            "Uninstall AMADEUS",
            f"Could not start the uninstaller. Nothing was deleted.\n\n{exc}",
            parent=root,
        )
        if script is not None:
            try:
                script.unlink(missing_ok=True)
            except OSError:
                pass
        return

    on_uninstall_start()
