# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Small runtime probe for tkinterdnd2 with the application's root pattern."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import tkinter as tk
from pathlib import Path

import customtkinter as ctk
from tkinterdnd2 import COPY, DND_FILES, REFUSE_DROP, TkinterDnD


def _add_tkdnd_package_path(app: ctk.CTk) -> None:
    module_path = getattr(TkinterDnD, "__file__", None)
    if not module_path:
        return
    machine = platform.machine().lower()
    if platform.system() == "Windows":
        platform_name = {
            "AMD64": "win-x64",
            "ARM64": "win-arm64",
            "X86": "win-x86",
        }.get(os.environ.get("PROCESSOR_ARCHITECTURE", platform.machine()).upper())
    elif platform.system() == "Darwin":
        platform_name = {"arm64": "osx-arm64", "x86_64": "osx-x64"}.get(machine)
    elif platform.system() == "Linux":
        platform_name = {"aarch64": "linux-arm64", "x86_64": "linux-x64"}.get(machine)
    else:
        platform_name = None
    if platform_name is None:
        return
    path = Path(module_path).parent / "tkdnd" / platform_name
    if int(str(app.tk.call("info", "tclversion")).split(".", 1)[0]) >= 9:
        tcl9_path = path.parent / f"{path.name}-tcl9"
        if tcl9_path.is_dir():
            path = tcl9_path
    if path.is_dir():
        tcl_path = path.as_posix()
        if platform.system() == "Windows" and len(tcl_path) >= 2 and tcl_path[1] == ":":
            tcl_path = f"//?/{tcl_path}"
        app.tk.call("lappend", "auto_path", tcl_path)


def run_probe() -> None:
    app = ctk.CTk()
    try:
        _add_tkdnd_package_path(app)
        tkdnd_version = TkinterDnD.require(app)
        canvas = tk.Canvas(app, width=160, height=80)
        canvas.pack()
        canvas.drop_target_register(DND_FILES)
        canvas.dnd_bind("<<DropEnter>>", lambda _event: COPY)
        canvas.dnd_bind("<<DropLeave>>", lambda _event: None)
        canvas.dnd_bind("<<Drop>>", lambda _event: REFUSE_DROP)
        app.update_idletasks()
        print(
            "tkinterdnd2 probe succeeded: "
            f"version={importlib.metadata.version('tkinterdnd2')} "
            f"tkdnd={tkdnd_version} platform={platform.platform()} "
            f"machine={platform.machine()} tcl={app.tk.call('info', 'patchlevel')} "
            f"tk={app.tk.call('package', 'provide', 'Tk')}",
            flush=True,
        )
    finally:
        app.destroy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="create, register, update, and destroy the probe root")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("--smoke is required for the non-interactive probe")
    try:
        run_probe()
    except Exception as exc:
        print(f"tkinterdnd2 probe failed: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
