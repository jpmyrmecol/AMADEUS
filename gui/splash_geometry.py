# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Choose a single monitor for the AMADEUS opening splash."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass


# Preserve the original splash logo sizing on all platforms.
SPLASH_LOGO_WIDTH_RATIO = 0.40


@dataclass(frozen=True)
class MonitorBounds:
    width: int
    height: int
    x: int
    y: int
    primary: bool = False


_XRANDR_MONITOR = re.compile(
    r"^\s*\d+:\s+(?P<flags>\S+)\s+"
    r"(?P<width>\d+)(?:/\d+)?x(?P<height>\d+)(?:/\d+)?"
    r"(?P<x>[+-]\d+)(?P<y>[+-]\d+)(?:\s|$)"
)


def parse_xrandr_monitors(output: str) -> tuple[MonitorBounds, ...]:
    """Parse \`xrandr --listmonitors\` without assuming a monitor arrangement."""
    monitors = []
    for line in output.splitlines():
        match = _XRANDR_MONITOR.match(line)
        if match is None:
            continue
        width = int(match["width"])
        height = int(match["height"])
        if width <= 0 or height <= 0:
            continue
        monitors.append(
            MonitorBounds(
                width=width,
                height=height,
                x=int(match["x"]),
                y=int(match["y"]),
                primary="*" in match["flags"],
            )
        )
    return tuple(monitors)


def linux_primary_monitor() -> MonitorBounds | None:
    """Return an X11 monitor, or defer placement to the window manager."""
    # XRandR may expose one combined XWayland surface instead of real outputs.
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        return None
    try:
        result = subprocess.run(
            ["xrandr", "--listmonitors"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    monitors = parse_xrandr_monitors(result.stdout)
    return next((monitor for monitor in monitors if monitor.primary), monitors[0] if monitors else None)


def configure_splash_window(window) -> tuple[int, int]:
    """Place a splash on one monitor and return its drawable dimensions."""
    if sys.platform.startswith("linux"):
        monitor = linux_primary_monitor()
        # Tk interprets negative geometry offsets relative to the far edge
        # of the virtual screen. Defer unusual negative layouts to the WM.
        if monitor is not None and monitor.x >= 0 and monitor.y >= 0:
            window.overrideredirect(True)
            window.geometry(
                f"{monitor.width}x{monitor.height}{monitor.x:+d}{monitor.y:+d}"
            )
            return monitor.width, monitor.height

        # A fullscreen request is monitor-scoped by the window manager. It is
        # safer than sizing a borderless window to the virtual desktop.
        window.attributes("-fullscreen", True)
        window.update()
        return max(1, window.winfo_width()), max(1, window.winfo_height())

    window.attributes("-fullscreen", True)
    return window.winfo_screenwidth(), window.winfo_screenheight()
