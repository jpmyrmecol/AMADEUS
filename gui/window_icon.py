# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import sys
import tempfile
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from typing import Any, List

import customtkinter as ctk
from PIL import Image, ImageTk

try:
    from .project_paths import gui_asset
except ImportError:  # Preserve direct execution with: python gui/<script>.py
    from project_paths import gui_asset


APP_USER_MODEL_ID = "AMADEUS.GUI"
WINDOW_ICON_FILE = "logo_amadeus_mini.png"
WINDOW_ICON_SIZES = (16, 32, 48, 256)
WINDOW_ICON_REFRESH_DELAYS_MS = (50, 250, 500, 1000, 2000)


def tk_font_spec(
    root: tk.Misc,
    family: str,
    point_size: int,
    weight: str = "normal",
):
    """Return a Linux font whose measured size and weight match the Home baseline.

    Windows and macOS keep their existing tuple behavior. On Linux, use the
    Roboto family bundled with CustomTkinter when available; its Xft metrics
    closely match the Windows Arial Home text. Keep Tk's native point sizing,
    except for the measured 13-point corner text, where Roboto 12 matches the
    Windows 19-pixel line height and 51-pixel "Update" width.
    """
    if not sys.platform.startswith("linux"):
        return family, point_size, weight

    linux_family = family
    linux_size = point_size
    if family.casefold() == "arial":
        families = {name.casefold(): name for name in tkfont.families(root)}
        if "roboto" in families:
            linux_family = families["roboto"]
            if point_size == 13 and weight == "normal":
                linux_size = 12

    return tkfont.Font(root=root, family=linux_family, size=linux_size, weight=weight)

def configure_taskbar_identity() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def configure_dpi_scaling(window: tk.Misc) -> None:
    """Apply the display's reported DPI to CustomTkinter and Tk on Linux.

    WSLg reports the effective X display DPI through Tk. Query it directly;
    launching Windows PowerShell to infer the host DPI adds startup work and
    cannot correct Tk's font backend or missing fontconfig support.
    """
    if not sys.platform.startswith("linux"):
        return

    try:
        dpi = float(window.winfo_fpixels("1i"))
    except (tk.TclError, TypeError, ValueError):
        return

    if not dpi or dpi <= 0:
        return

    factor = dpi / 96.0
    ctk.set_widget_scaling(factor)
    ctk.set_window_scaling(factor)

    try:
        window.tk.call("tk", "scaling", dpi / 72.0)
    except tk.TclError:
        pass

def install_window_icon(window: tk.Misc) -> None:
    """Apply AMADEUS's icon now, and again shortly after, once the window
    manager has actually mapped the window. A single application while the
    window is still withdrawn/unmapped (as it commonly is, e.g. behind a
    splash screen) is not reliably picked up by every window
    manager/taskbar bridge -- Windows fights CTk's own delayed default-icon
    override this way already; WSLg's icon bridge needs the same retry even
    though CTk itself never overrides the icon on Linux.
    """
    set_window_icon(window)
    if os.name == "nt" or sys.platform.startswith("linux"):
        for delay_ms in WINDOW_ICON_REFRESH_DELAYS_MS:
            try:
                window.after(delay_ms, set_window_icon, window)
            except tk.TclError:
                return


def set_window_icon(window: tk.Misc) -> None:
    """Apply the AMADEUS logo to a Tk/CTk title bar and taskbar."""
    try:
        if not window.winfo_exists():
            return
    except tk.TclError:
        return

    icon_path = gui_asset(WINDOW_ICON_FILE)
    if not icon_path.is_file():
        return

    try:
        with Image.open(icon_path) as source:
            source = source.convert("RGBA")
            content_box = source.getbbox()
            if content_box:
                source = source.crop(content_box)

            icons: List[ImageTk.PhotoImage] = []
            largest_square: Image.Image | None = None
            for size in WINDOW_ICON_SIZES:
                contained = source.copy()
                contained.thumbnail((size, size), Image.LANCZOS)
                square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
                offset = ((size - contained.width) // 2, (size - contained.height) // 2)
                square.alpha_composite(contained, offset)
                icons.append(ImageTk.PhotoImage(square, master=window))
                largest_square = square

        window.iconphoto(True, *icons)
        setattr(window, "AMADEUS_WINDOW_ICONS", icons)

        if os.name == "nt" and largest_square is not None:
            cache_dir = Path(tempfile.gettempdir()) / "AMADEUS"
            cache_dir.mkdir(parents=True, exist_ok=True)
            ico_path = cache_dir / f"{Path(WINDOW_ICON_FILE).stem}.ico"
            largest_square.save(
                ico_path,
                format="ICO",
                sizes=[(size, size) for size in WINDOW_ICON_SIZES],
            )

            # CTk schedules its blue default icon shortly after creating each
            # window. Mark the icon as user-defined and set both Tk and native
            # Windows icon handles so title bar and taskbar stay on AMADEUS.
            if hasattr(window, "_iconbitmap_method_called"):
                setattr(window, "_iconbitmap_method_called", True)
            try:
                window.iconbitmap(str(ico_path))
            except tk.TclError:
                pass
            try:
                window.iconbitmap(default=str(ico_path))
            except tk.TclError:
                pass
            _apply_windows_icon_handles(window, ico_path)
            setattr(window, "AMADEUS_WINDOW_ICON_PATH", str(ico_path))
    except (OSError, tk.TclError):
        pass


def _apply_windows_icon_handles(window: tk.Misc, ico_path: Path) -> None:
    if os.name != "nt":
        return

    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.GetParent.restype = ctypes.c_void_p
        user32.GetParent.argtypes = [ctypes.c_void_p]
        user32.LoadImageW.restype = ctypes.c_void_p
        user32.LoadImageW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.SendMessageW.restype = ctypes.c_void_p
        user32.SendMessageW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]

        hwnd_value = window.winfo_id()
        hwnd = ctypes.c_void_p(hwnd_value)
        parent_hwnd = user32.GetParent(hwnd)
        target_hwnd = ctypes.c_void_p(parent_hwnd or hwnd.value)
        handles: dict[int, Any] | None = getattr(window, "AMADEUS_WINDOW_ICON_HANDLES", None)

        if handles is None:
            image_icon = 1
            load_from_file = 0x00000010
            handles = {
                0: user32.LoadImageW(None, str(ico_path), image_icon, 16, 16, load_from_file),
                1: user32.LoadImageW(None, str(ico_path), image_icon, 32, 32, load_from_file),
            }
            handles = {icon_type: handle for icon_type, handle in handles.items() if handle}
            setattr(window, "AMADEUS_WINDOW_ICON_HANDLES", handles)

        wm_seticon = 0x0080
        for icon_type, handle in handles.items():
            user32.SendMessageW(target_hwnd, wm_seticon, icon_type, ctypes.c_void_p(handle))
    except (AttributeError, OSError, tk.TclError, TypeError, ValueError):
        pass
