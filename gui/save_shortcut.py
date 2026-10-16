# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Save keyboard shortcuts scoped to one AMADEUS GUI window."""

from __future__ import annotations

import sys
import tkinter as tk
from typing import Callable


def bind_save_shortcut(window: tk.Misc, save_callback: Callable[[], object]) -> None:
    """Map Ctrl+S (also Command+S on macOS) to the window's existing Save action.

    Tk's bind_all reaches focused child Entries, canvases and spinboxes.
    Ignore events from other toplevels, including modal dialogs, so one GUI
    cannot accidentally save another window's configuration.
    """
    saving = False

    def on_save(event):
        nonlocal saving
        try:
            if event.widget.winfo_toplevel() is not window:
                return None
        except (AttributeError, tk.TclError):
            return None

        if saving:
            return "break"

        saving = True
        try:
            save_callback()
        finally:
            saving = False
        return "break"

    window.bind_all("<Control-s>", on_save, add="+")
    if sys.platform == "darwin":
        window.bind_all("<Command-s>", on_save, add="+")
