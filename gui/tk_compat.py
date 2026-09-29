# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Compatibility fixes for supported Tkinter runtimes."""

from __future__ import annotations

import sys
import tkinter as tk


def apply_tk_compatibility() -> bool:
    """Backport CPython GH-103685 for Python 3.10 running with Tk 9.

    Tk 9 returns an empty string for the index of an empty Menu. Python 3.10's
    tkinter.Menu.index only treats "none" as an empty result and passes the
    empty string to getint(), which raises TclError. Python 3.11+ handles both
    values. Keep the patch limited to the affected runtime combination.
    """
    if tk.TkVersion < 9.0 or sys.version_info >= (3, 11):
        return False

    current = tk.Menu.index
    if getattr(current, "_amadeus_tk9_compat", False):
        return False

    def _menu_index(self, index):
        value = self.tk.call(self._w, "index", index)
        return None if value in ("", "none") else self.tk.getint(value)

    _menu_index._amadeus_tk9_compat = True
    tk.Menu.index = _menu_index
    return True
