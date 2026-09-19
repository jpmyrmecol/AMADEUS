# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import time
import tkinter as tk

HOLD_AFTER_REVEAL_MS = 800
MAX_DISPLAY_MS = 4000 + HOLD_AFTER_REVEAL_MS
_DURATION_MS = 3000
_DELAY_MS = 500
_FRAME_MS = 16


def _ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def reveal_total_ms() -> int:
    """Time to keep the splash visible, including a short post-reveal hold."""
    return MAX_DISPLAY_MS


def reveal_from_left(canvas: tk.Canvas, left: float, top: float, right: float, bottom: float, bg: str) -> None:
    """Hold on black briefly, then uncover a region left-to-right."""
    cover = canvas.create_rectangle(left, top, right, bottom, fill=bg, outline="")

    def step(started_at: float | None = None) -> None:
        try:
            if started_at is None:
                started_at = time.perf_counter()
            if _DURATION_MS <= 0:
                progress = 1.0
            else:
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                progress = min(1.0, elapsed_ms / _DURATION_MS)

            if progress >= 1.0:
                canvas.delete(cover)
                return

            x = left + (right - left) * _ease_out_cubic(progress)
            canvas.coords(cover, x, top, right, bottom)
            canvas.after(_FRAME_MS, step, started_at)
        except tk.TclError:
            pass  # the splash window was closed mid-animation

    canvas.after(_DELAY_MS, step)
