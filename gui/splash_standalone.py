# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

from splash_ipc import (
    closed_file_path,
    dismissed_file_path,
    record_closed,
    record_dismissed,
    record_started,
    started_file_path,
    stop_file_path,
)
from splash_reveal import MAX_DISPLAY_MS, reveal_from_left, reveal_total_ms

POLL_MS = 100
SAFETY_TIMEOUT_MS = 120_000
SPLASH_BG = "black"
SPLASH_LOGO_WIDTH_RATIO = 0.40  # logo width as a fraction of screen width


def main() -> None:
    if len(sys.argv) < 2:
        return
    safety_timeout_ms = min(SAFETY_TIMEOUT_MS, MAX_DISPLAY_MS)
    if len(sys.argv) >= 3:
        try:
            requested_timeout_ms = max(0, int(float(sys.argv[2]) * 1000))
            # Older installed launch commands pass 4 seconds explicitly.
            # Never let that legacy value cut off the post-reveal hold.
            safety_timeout_ms = min(
                SAFETY_TIMEOUT_MS,
                max(reveal_total_ms(), requested_timeout_ms),
            )
        except ValueError:
            pass
    token = sys.argv[1]
    marker = stop_file_path(token)
    started_marker = started_file_path(token)
    dismissed_marker = dismissed_file_path(token)
    closed_marker = closed_file_path(token)
    marker.parent.mkdir(parents=True, exist_ok=True)
    for runtime_marker in (marker, started_marker, dismissed_marker, closed_marker):
        runtime_marker.unlink(missing_ok=True)
    record_started(token)

    logo_path = Path(__file__).resolve().parent.parent / "assets" / "logo_amadeus_splash.png"

    root = tk.Tk()
    root.configure(bg=SPLASH_BG)
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    is_closed = False

    def close_splash() -> None:
        nonlocal is_closed
        if is_closed:
            return
        is_closed = True
        try:
            root.destroy()
        except tk.TclError:
            pass

    def dismiss_splash(_event=None) -> None:
        if is_closed:
            return
        try:
            record_dismissed(token)
        except OSError:
            pass
        close_splash()

    root.bind_all("<ButtonPress>", dismiss_splash, add="+")

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg=SPLASH_BG, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)

    if logo_path.is_file():
        source = Image.open(logo_path).convert("RGBA")
        target_w = max(1, round(screen_w * SPLASH_LOGO_WIDTH_RATIO))
        target_h = max(1, round(source.height * (target_w / source.width)))
        source = source.resize((target_w, target_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(source)
        canvas.AMADEUS_SPLASH_PHOTO = photo  # keep a reference alive
        w, h = photo.width(), photo.height()
        cx, cy = screen_w / 2, screen_h / 2
        canvas.create_image(cx, cy, image=photo)
        reveal_from_left(canvas, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, SPLASH_BG)
    else:
        canvas.create_text(screen_w / 2, screen_h / 2, text="AMADEUS", font=("Arial", 32, "bold"), fill="white")

    def check_stop() -> None:
        if marker.exists():
            close_splash()
            return
        root.after(POLL_MS, check_stop)

    root.after(POLL_MS, check_stop)
    root.after(safety_timeout_ms, close_splash)
    root.mainloop()
    try:
        record_closed(token)
    except OSError:
        pass


if __name__ == "__main__":
    main()
