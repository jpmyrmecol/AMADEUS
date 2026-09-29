# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
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
    family: str,
    point_size: int,
    weight: str = "normal",
) -> tuple[str, int, str]:
    """Return a Tk font tuple with stable Windows-equivalent sizing on Linux.

    Tk uses positive font sizes as points and negative sizes as pixels. On the
    current Linux/Tk 9 path used by WSLg, requested positive sizes can collapse
    to nearly the same small bitmap size. Windows at the 96-DPI baseline maps
    points to pixels by 96/72, so use that mapping explicitly on Linux while
    leaving Windows and macOS unchanged.
    """
    if sys.platform.startswith("linux"):
        pixel_size = max(1, round(point_size * 96.0 / 72.0))
        return family, -pixel_size, weight
    return family, point_size, weight


def configure_taskbar_identity() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def _is_wsl() -> bool:
    """Return whether this Linux process is running under WSL/WSLg."""
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(
            encoding="utf-8", errors="ignore"
        )
    except OSError:
        return False
    return "microsoft" in release.lower() or "wsl" in release.lower()


def _windows_host_dpi() -> float | None:
    """Read the Windows system DPI from WSL in a DPI-aware thread context.

    PowerShell itself can already be initialized as DPI-unaware, in which case
    process-level DPI-awareness calls are too late and Windows virtualizes DPI
    queries to 96. Set the current thread to PER_MONITOR_AWARE_V2 temporarily,
    query GetDpiForSystem(), then restore the previous thread context.
    """
    if not _is_wsl():
        return None

    powershell = shutil.which("powershell.exe")
    if not powershell:
        candidate = Path(
            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        )
        powershell = str(candidate) if candidate.is_file() else None
    if not powershell:
        return None

    command = r"""
$source = @'
using System;
using System.Runtime.InteropServices;

public static class AmadeusDpi {
    [DllImport("User32.dll")]
    private static extern IntPtr SetThreadDpiAwarenessContext(IntPtr dpiContext);

    [DllImport("User32.dll")]
    private static extern uint GetDpiForSystem();

    public static uint QuerySystemDpi() {
        // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4.
        IntPtr previous = SetThreadDpiAwarenessContext(new IntPtr(-4));
        try {
            return GetDpiForSystem();
        }
        finally {
            if (previous != IntPtr.Zero) {
                SetThreadDpiAwarenessContext(previous);
            }
        }
    }
}
'@

Add-Type -TypeDefinition $source -ErrorAction Stop
[Console]::Write([AmadeusDpi]::QuerySystemDpi())
"""

    try:
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
            check=False,
        )
        if completed.returncode == 0:
            dpi = float((completed.stdout or "").strip())
            if 48.0 <= dpi <= 768.0:
                return dpi
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    # Last-resort compatibility fallback for hosts where the native query
    # cannot run. This value may remain 96 on WSL even at higher host scaling.
    registry_command = (
        "$v=(Get-ItemProperty -LiteralPath "
        "'HKCU:\\Control Panel\\Desktop\\WindowMetrics' "
        "-Name AppliedDPI -ErrorAction SilentlyContinue).AppliedDPI; "
        "if ($null -ne $v) { [Console]::Write([int]$v) }"
    )
    try:
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                registry_command,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3.0,
            check=False,
        )
        if completed.returncode != 0:
            return None
        dpi = float((completed.stdout or "").strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

    return dpi if 48.0 <= dpi <= 768.0 else None


def configure_dpi_scaling(window: tk.Misc) -> None:
    """Match Linux GUI scaling to the effective display DPI.

    CustomTkinter does not perform its Windows-style per-monitor DPI handling
    on Linux. Under WSL/WSLg, XWayland can report 96 DPI even when the Windows
    host uses 125%, 150%, 200%, or another display scale. Prefer the Windows
    user's AppliedDPI in WSL, and fall back to Tk's display DPI on native Linux.

    Apply the same DPI to both CustomTkinter and Tk itself. The Tk scaling is
    required for AMADEUS widgets drawn with tkinter.Canvas.create_text(), which
    are not affected by CustomTkinter's widget scaling.
    """
    if not sys.platform.startswith("linux"):
        return

    dpi = _windows_host_dpi()
    if dpi is None:
        try:
            dpi = float(window.winfo_fpixels("1i"))
        except (tk.TclError, TypeError, ValueError):
            return

    if not dpi or dpi <= 0:
        return

    factor = dpi / 96.0
    ctk.set_widget_scaling(factor)
    ctk.set_window_scaling(factor)

    # Tk font sizes expressed as positive numbers are point sizes. Match Tk's
    # point-to-pixel conversion to the same DPI so plain Tk/Canvas text scales
    # with the CustomTkinter controls.
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
