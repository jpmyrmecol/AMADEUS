# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Right-click copying for non-editable GUI text, including dynamically added widgets."""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

import customtkinter as ctk


_TEXT_WIDGETS = (
    tk.Label, tk.Button, tk.Checkbutton, tk.Radiobutton, tk.Menubutton,
    tk.LabelFrame,
    ttk.Label, ttk.Button, ttk.Checkbutton, ttk.Radiobutton, ttk.Menubutton,
    ttk.Labelframe,
    ctk.CTkLabel, ctk.CTkButton, ctk.CTkCheckBox, ctk.CTkRadioButton,
    ctk.CTkSwitch,
)
_EDITABLE_WIDGETS = (
    tk.Entry, tk.Spinbox, tk.Text, tk.Listbox,
    ttk.Entry, ttk.Combobox,
    ctk.CTkEntry, ctk.CTkTextbox, ctk.CTkComboBox,
)


def _widget_text(widget: tk.Misc) -> str | None:
    """Read the displayed text, including textvariable-backed status labels."""
    try:
        value = widget.cget("text")
    except (AttributeError, tk.TclError, ValueError):
        value = None
    if value is not None and str(value):
        return str(value)
    try:
        variable = widget.cget("textvariable")
    except (AttributeError, tk.TclError, ValueError):
        return None
    if not variable:
        return None
    try:
        value = variable.get() if hasattr(variable, "get") else widget.getvar(str(variable))
    except (AttributeError, tk.TclError, ValueError):
        return None
    return str(value) if value is not None and str(value) else None


def _canvas_text_at(canvas: tk.Canvas, x: int, y: int) -> str | None:
    """Support labels explicitly drawn as Canvas text (e.g. Home)."""
    try:
        hits = canvas.find_overlapping(x, y, x, y)
        for item in reversed(hits):
            if canvas.type(item) != "text":
                continue
            text = canvas.itemcget(item, "text")
            if text:
                return str(text)
    except (AttributeError, tk.TclError):
        pass
    return None


def copyable_text_at(widget: tk.Misc, x: int = 0, y: int = 0) -> str | None:
    """Return only text under the pointer; do not hijack editable fields."""
    canvas = widget if isinstance(widget, tk.Canvas) else None
    current = widget
    while current is not None:
        if isinstance(current, _EDITABLE_WIDGETS):
            return None
        if isinstance(current, _TEXT_WIDGETS):
            text = _widget_text(current)
            if text:
                return text
        if isinstance(current, ctk.CTkOptionMenu):
            value = str(current.get())
            return value or None
        current = getattr(current, "master", None)
    if canvas is not None:
        return _canvas_text_at(canvas, x, y)
    return None


def install_right_click_copy(window: tk.Misc) -> None:
    """Install one application-wide context menu per Tk interpreter.

    Uses Tk's 'all' bind tag so labels created later (including those inside
    dialogs) also work.  Widget-specific right-click handlers take precedence;
    editable fields and unlabeled canvases keep their own behavior.
    """
    root = window._root()
    if getattr(root, "_amadeus_copy_text_installed", False):
        return

    menu = tk.Menu(root, tearoff=False)

    def copy_text(value: str) -> None:
        root.clipboard_clear()
        root.clipboard_append(value)

    def right_click(event: tk.Event) -> str | None:
        value = copyable_text_at(event.widget, event.x, event.y)
        if not value:
            return None
        menu.delete(0, tk.END)
        menu.add_command(label="Copy text", command=lambda text=value: copy_text(text))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    root.bind_all("<Button-3>", right_click, add="+")
    if sys.platform == "darwin":
        root.bind_all("<Control-Button-1>", right_click, add="+")
    # Retain the callback/menu for the lifetime of this Tk interpreter.
    root._amadeus_copy_text_handler = right_click
    root._amadeus_copy_text_menu = menu
    root._amadeus_copy_text_installed = True
