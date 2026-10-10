# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Time-synchronized left/right arrow-key navigation for video previews."""

import math
import time
from typing import Callable


class ArrowKeyHoldPlayback:
    """Single-frame tap followed by source-FPS-paced, frame-skipping key hold.

    A short deferred key-release accommodates platforms where OS autorepeat
    produces a KeyRelease/KeyPress pair for each repeated keystroke.
    """

    def __init__(
        self,
        owner,
        *,
        current_frame: Callable[[], int],
        frame_bounds: Callable[[], tuple[int, int]],
        fps: Callable[[], float],
        speed: Callable[[], float],
        seek: Callable[[int], bool | None],
        busy: Callable[[], bool] = lambda: False,
        pause: Callable[[], None] = lambda: None,
    ):
        self.owner = owner
        self.current_frame = current_frame
        self.frame_bounds = frame_bounds
        self.fps = fps
        self.speed = speed
        self.seek = seek
        self.busy = busy
        self.pause = pause
        self.direction = 0
        self.repeating = False
        self._timer = None
        self._release_timer = None
        self._start_frame = 0
        self._progress = 0.0
        self._last_tick = 0.0

    def _cancel(self, name: str) -> None:
        job = getattr(self, name)
        if job is not None:
            setattr(self, name, None)
            try:
                self.owner.after_cancel(job)
            except Exception:
                pass

    def stop(self) -> None:
        self._cancel("_timer")
        self._cancel("_release_timer")
        self.direction = 0
        self.repeating = False
        self._progress = 0.0

    def press(self, direction: int) -> None:
        if direction not in (-1, 1):
            return
        if self.direction == direction:
            # Ignore native auto-repeat presses, but cancel the synthetic
            # KeyRelease on X11 so an uninterrupted hold stays active.
            self._cancel("_release_timer")
            return

        self.stop()
        lower, upper = self.frame_bounds()
        if lower > upper:
            return
        self.pause()
        origin = self.current_frame()
        target = min(upper, max(lower, origin + direction))
        self.direction = direction
        self._start_frame = target
        if target != origin and self.seek(target) is False:
            self.stop()
            return
        self._timer = self.owner.after(220, self._begin_repeat)

    def release(self, direction: int) -> None:
        if self.direction != direction:
            return
        self._cancel("_release_timer")
        self._release_timer = self.owner.after(35, self.stop)

    def _begin_repeat(self) -> None:
        self._timer = None
        if not self.direction:
            return
        self.repeating = True
        self._progress = 0.0
        self._last_tick = time.monotonic()
        self._tick()

    def _tick(self) -> None:
        self._timer = None
        if not self.direction:
            return
        now = time.monotonic()
        rate = max(1e-9, float(self.fps()) * float(self.speed()))
        self._progress += max(0.0, now - self._last_tick) * rate
        self._last_tick = now

        lower, upper = self.frame_bounds()
        if lower > upper:
            self.stop()
            return
        target = min(upper, max(lower, self._start_frame + self.direction * int(self._progress)))
        if not self.busy() and target != self.current_frame():
            if self.seek(target) is False:
                self.stop()
                return

        if target in (lower, upper) and (
            (self.direction < 0 and target == lower)
            or (self.direction > 0 and target == upper)
        ):
            return  # Remain held, but do not schedule more work at the boundary.

        if self.busy():
            delay_ms = 8
        else:
            # Time to the next source-frame boundary (not a fixed per-frame
            # delay), so expensive rendering cannot accumulate clock drift.
            remaining = (math.floor(self._progress) + 1 - self._progress) / rate
            delay_ms = max(1, int(remaining * 1000))
        self._timer = self.owner.after(delay_ms, self._tick)
