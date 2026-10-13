# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Subtle rotating white highlight along the native border of a CTkButton."""

import math
import time
import tkinter as tk

import customtkinter as ctk


class RotatingBorderHighlight:
    """Only animates a CTkButton canvas; does not change button size or behavior."""

    def __init__(self, master: tk.Misc, button: ctk.CTkButton):
        self.master = master
        self.button = button
        self.active = False
        self.hovered = False
        self.phase = 0.0
        self.last_time = time.monotonic()
        self.job = None
        self.size = None
        # A non-highlighted button must have the ordinary borderless CTk look.
        # Remember the configured highlight width before clearing it.
        self.highlight_border_width = max(3, int(button.cget("border_width") or 0))
        button.configure(border_width=0)
        self.items = [
            button._canvas.create_line(
                0, 0, 0, 0, width=2.6, capstyle=tk.ROUND,
                tags="rotating_border_highlight", state="hidden",
            ) for _ in range(96)
        ]
        button.bind("<Enter>", self._enter)
        button.bind("<Leave>", self._leave)
        button._canvas.bind(
            "<Configure>", lambda _event: master.after_idle(self._paint), add="+"
        )

    @staticmethod
    def _outline(width: int, height: int, count: int, corner: float, border: float):
        inset = border / 2
        left, top, right, bottom = inset, inset, width - inset, height - inset
        radius = max(1.0, min(corner, width / 2, height / 2) - inset)
        horizontal = right - left - 2 * radius
        vertical = bottom - top - 2 * radius
        arc = math.pi * radius / 2
        lengths = (horizontal, arc, vertical, arc, horizontal, arc, vertical, arc)
        perimeter = sum(lengths)
        points = []
        for i in range(count):
            distance = i * perimeter / count
            for part, length in enumerate(lengths):
                if distance < length:
                    t = distance / length
                    break
                distance -= length
            if part == 0:
                point = (left + radius + t * horizontal, top)
            elif part == 1:
                angle = -math.pi / 2 + t * math.pi / 2
                point = (right - radius + radius * math.cos(angle),
                         top + radius + radius * math.sin(angle))
            elif part == 2:
                point = (right, top + radius + t * vertical)
            elif part == 3:
                angle = t * math.pi / 2
                point = (right - radius + radius * math.cos(angle),
                         bottom - radius + radius * math.sin(angle))
            elif part == 4:
                point = (right - radius - t * horizontal, bottom)
            elif part == 5:
                angle = math.pi / 2 + t * math.pi / 2
                point = (left + radius + radius * math.cos(angle),
                         bottom - radius + radius * math.sin(angle))
            elif part == 6:
                point = (left, bottom - radius - t * vertical)
            else:
                angle = math.pi + t * math.pi / 2
                point = (left + radius + radius * math.cos(angle),
                         top + radius + radius * math.sin(angle))
            points.append(point)
        return points

    def _paint(self):
        if not self.active or not self.button.winfo_exists():
            return
        canvas = self.button._canvas
        size = (canvas.winfo_width(), canvas.winfo_height())
        if min(size) < 20:
            return
        count = len(self.items)
        if size != self.size:
            points = self._outline(
                *size, count,
                self.button._apply_widget_scaling(self.button.cget("corner_radius")),
                self.button._apply_widget_scaling(self.button.cget("border_width")),
            )
            for i, item in enumerate(self.items):
                canvas.coords(item, *points[i], *points[(i + 1) % count])
            self.size = size
        for i, item in enumerate(self.items):
            distance = abs((i / count - self.phase + 0.5) % 1 - 0.5)
            brightness = max(0.0, 1.0 - distance / 0.13) ** 2
            shade = round(119 + (250 - 119) * brightness)
            canvas.itemconfigure(item, fill=f"#{shade:02x}{shade:02x}{shade:02x}")
        canvas.tag_raise("rotating_border_highlight")

    def _tick(self):
        self.job = None
        if not self.active or self.hovered:
            return
        now = time.monotonic()
        self.phase = (self.phase + (now - self.last_time) / 3.0) % 1.0
        self.last_time = now
        self._paint()
        self.job = self.master.after(50, self._tick)

    def _enter(self, _event=None):
        self.hovered = True
        if self.job is not None:
            self.master.after_cancel(self.job)
            self.job = None

    def _leave(self, _event=None):
        self.master.after(25, self._check_hover)

    def _check_hover(self):
        if not self.hovered or not self.button.winfo_exists():
            return
        x, y = self.master.winfo_pointerx(), self.master.winfo_pointery()
        b = self.button
        if (b.winfo_rootx() <= x < b.winfo_rootx() + b.winfo_width()
                and b.winfo_rooty() <= y < b.winfo_rooty() + b.winfo_height()):
            return
        self.hovered = False
        self.last_time = time.monotonic()
        if self.active:
            self._tick()

    def set_active(self, active: bool):
        if self.active == active:
            return
        self.active = active
        if self.job is not None:
            self.master.after_cancel(self.job)
            self.job = None
        self.button._canvas.itemconfigure(
            "rotating_border_highlight", state="normal" if active else "hidden"
        )
        self.button.configure(
            border_width=self.highlight_border_width if active else 0
        )
        self.size = None
        if active:
            self.last_time = time.monotonic()
            self._paint()
            if not self.hovered:
                self._tick()

    def stop(self):
        self.set_active(False)
