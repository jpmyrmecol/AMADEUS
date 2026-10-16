# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Consistent dark numeric spinboxes on macOS, Windows and Linux."""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk


_DARK_BG = "#242424"
_FIELD_BG = "#343638"
_FIELD_FG = "#DCE4EE"
_ARROW_BG = "#565B5E"
_ACCENT = "#1A7C48"

_TK_ONLY_OPTIONS = frozenset({
    "bg", "background", "fg", "foreground",
    "disabledbackground", "disabledforeground", "readonlybackground",
    "insertbackground", "buttonbackground",
    "highlightthickness", "highlightbackground", "highlightcolor",
    "relief", "bd", "borderwidth", "selectbackground", "selectforeground",
})


def configure_dark_ttk(master: tk.Misc) -> None:
    """Avoid native Aqua controls and use a coherent dark ttk palette."""
    if sys.platform != "darwin":
        return
    window = master.winfo_toplevel()
    if getattr(window, "_amadeus_dark_ttk_configured", False):
        return
    style = ttk.Style(window)
    style.theme_use("clam")
    style.configure(".", background=_DARK_BG, foreground=_FIELD_FG)
    style.configure(
        "AMADEUS.Dark.TSpinbox",
        background=_ARROW_BG,
        fieldbackground=_FIELD_BG,
        foreground=_FIELD_FG,
        insertcolor=_FIELD_FG,
        arrowcolor=_FIELD_FG,
        bordercolor=_ARROW_BG,
        lightcolor=_ARROW_BG,
        darkcolor=_FIELD_BG,
        selectbackground="#1F6AA5",
        selectforeground="#FFFFFF",
        arrowsize=10,
        padding=(3, 2),
    )
    style.map(
        "AMADEUS.Dark.TSpinbox",
        fieldbackground=[("disabled", _FIELD_BG)],
        foreground=[("disabled", "#929292")],
        arrowcolor=[("disabled", "#929292"), ("active", "#FFFFFF")],
        background=[("disabled", "#3A3A3A")],
    )
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor="#4A4D50",
        background=_ACCENT,
        bordercolor=_DARK_BG,
        lightcolor=_ACCENT,
        darkcolor=_ACCENT,
    )
    style.configure(
        "Treeview",
        background="#2B2B2B",
        fieldbackground="#2B2B2B",
        foreground=_FIELD_FG,
        bordercolor="#444444",
        rowheight=20,
    )
    style.map(
        "Treeview",
        background=[("selected", "#1F6AA5")],
        foreground=[("selected", "#FFFFFF")],
    )
    style.configure(
        "Treeview.Heading",
        background="#3A3A3A",
        foreground=_FIELD_FG,
        relief="flat",
    )
    style.map("Treeview.Heading", background=[("active", "#444444")])
    style.configure(
        "Vertical.TScrollbar",
        background="#3A3A3A",
        troughcolor="#2B2B2B",
        arrowcolor=_FIELD_FG,
        bordercolor="#2B2B2B",
    )
    window._amadeus_dark_ttk_configured = True


def dark_spinbox(master: tk.Misc, **options) -> tk.Spinbox | ttk.Spinbox:
    """Preserve native Tk spinbox behavior except for macOS Aqua rendering."""
    if sys.platform != "darwin":
        return tk.Spinbox(master, **options)
    configure_dark_ttk(master)
    # Classic Tk colors are configured through the ttk style on macOS.
    styled_options = {key: value for key, value in options.items()
                      if key not in _TK_ONLY_OPTIONS}
    styled_options["style"] = "AMADEUS.Dark.TSpinbox"
    return ttk.Spinbox(master, **styled_options)
