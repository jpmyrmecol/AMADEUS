# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Native canvas file-drop support shared by GUI input screens."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES, COPY, REFUSE_DROP
    _TKINTERDND2_IMPORT_ERROR = None
except Exception as exc:
    TkinterDnD = None
    DND_FILES = None
    COPY = "copy"
    REFUSE_DROP = "refuse_drop"
    _TKINTERDND2_IMPORT_ERROR = exc


@dataclass
class CanvasFileDropState:
    enabled: bool = False
    canvas_registered: bool = False
    diagnostics: list[str] = field(default_factory=list)


def accepted_drop_paths(
    paths: Iterable[str],
    allowed_suffixes: Iterable[str],
    accept_path: Callable[[str], bool] | None = None,
) -> tuple[str, ...]:
    """Return existing, normalized paths accepted by the supplied filter."""
    suffixes = frozenset(str(suffix).lower() for suffix in allowed_suffixes)
    accepted: list[str] = []
    for raw_path in paths:
        normalized = os.path.normpath(str(raw_path))
        if not os.path.isfile(normalized):
            continue
        if suffixes and os.path.splitext(normalized)[1].lower() not in suffixes:
            continue
        if accept_path is not None and not accept_path(normalized):
            continue
        accepted.append(normalized)
    return tuple(accepted)


def _add_tkdnd_package_path(window: tk.Misc, log: Callable[[str], None]) -> None:
    """Expose the bundled native tkdnd directory using Tcl-safe separators."""
    if TkinterDnD is None:
        return
    module_path = getattr(TkinterDnD, "__file__", None)
    if not module_path:
        return

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        process_machine = os.environ.get("PROCESSOR_ARCHITECTURE", platform.machine()).upper()
        platform_name = {
            "AMD64": "win-x64",
            "ARM64": "win-arm64",
            "X86": "win-x86",
        }.get(process_machine)
    elif system == "Darwin":
        platform_name = {"arm64": "osx-arm64", "x86_64": "osx-x64"}.get(machine)
    elif system == "Linux":
        platform_name = {"aarch64": "linux-arm64", "x86_64": "linux-x64"}.get(machine)
    else:
        platform_name = None
    if platform_name is None:
        return

    tkdnd_root = Path(module_path).parent / "tkdnd"
    native_path = tkdnd_root / platform_name
    try:
        if int(str(window.tk.call("info", "tclversion")).split(".", 1)[0]) >= 9:
            tcl9_path = tkdnd_root / f"{platform_name}-tcl9"
            if tcl9_path.is_dir():
                native_path = tcl9_path
    except Exception:
        pass
    if native_path.is_dir():
        tcl_path = native_path.as_posix()
        if system == "Windows" and len(tcl_path) >= 2 and tcl_path[1] == ":":
            tcl_path = f"//?/{tcl_path}"
        window.tk.call("lappend", "auto_path", tcl_path)
        log(f"tkdnd package path={tcl_path}")


def install_canvas_file_drop(
    window: tk.Misc,
    canvas: tk.Canvas,
    *,
    allowed_suffixes: Iterable[str],
    on_path: Callable[[str], None],
    accept_path: Callable[[str], bool] | None = None,
    status_callback: Callable[[str], None] | None = None,
    label: str = "file",
) -> CanvasFileDropState:
    """Register one-file native drops on a canvas when tkdnd is available."""
    state = CanvasFileDropState()

    def log(message: str) -> None:
        entry = f"[DND] {message}"
        state.diagnostics.append(entry)
        print(entry, flush=True)

    def reject(message: str) -> str:
        log(f"rejected: {message}")
        if status_callback is not None:
            status_callback(message)
        return REFUSE_DROP

    try:
        dnd_version = importlib.metadata.version("tkinterdnd2")
    except importlib.metadata.PackageNotFoundError:
        dnd_version = "unavailable"
    except Exception as exc:
        dnd_version = f"unavailable ({exc})"
    try:
        tcl_version = str(window.tk.call("info", "patchlevel"))
    except Exception as exc:
        tcl_version = f"unavailable ({exc})"
    log(
        "startup "
        f"tkinterdnd2={dnd_version} platform={platform.platform()} "
        f"machine={platform.machine()} tcl={tcl_version}"
    )

    if TkinterDnD is None:
        log(f"initialization failed: {_TKINTERDND2_IMPORT_ERROR}")
        log("canvas registered=False")
        return state

    try:
        _add_tkdnd_package_path(window, log)
        tkdnd_version = TkinterDnD.require(window)
        log(f"tkdnd={tkdnd_version}")
        canvas.drop_target_register(DND_FILES)
        state.canvas_registered = True
        log("canvas registered=True")

        def on_enter(_event):
            return COPY

        def on_leave(_event):
            return None

        def on_drop(event):
            raw_data = str(getattr(event, "data", ""))
            log(f"drop raw={raw_data!r}")
            try:
                parsed_paths = tuple(str(path) for path in window.tk.splitlist(raw_data))
            except Exception as exc:
                log(f"drop parse failed: {exc}")
                return reject(f"Unable to read the dropped {label} list.")
            log(f"drop parsed={parsed_paths!r}")
            paths = accepted_drop_paths(parsed_paths, allowed_suffixes, accept_path)
            log(f"drop accepted={paths!r}")
            if len(paths) > 1:
                return reject(f"Please drop one {label} at a time.")
            if not paths:
                return reject(f"Please drop one {label} file.")
            selected_path = paths[0]
            window.after_idle(lambda path=selected_path: on_path(path))
            return COPY

        canvas.dnd_bind("<<DropEnter>>", on_enter)
        canvas.dnd_bind("<<DropLeave>>", on_leave)
        canvas.dnd_bind("<<Drop>>", on_drop)
        state.enabled = True
        log("initialization succeeded")
    except Exception as exc:
        log(f"initialization failed: {exc}")
        log(f"canvas registered={state.canvas_registered}")
    return state
