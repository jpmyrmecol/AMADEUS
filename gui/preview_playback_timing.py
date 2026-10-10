# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Retain estimated video-preview rendering capacity between play/pause cycles."""

import math


class AdaptivePreviewPacer:
    """Limit frame requests to an observed sustainable rate without slowing video time.

    The learned rendering interval is retained across pause/resume; only the
    scheduling clock is reset. The playback controller still calculates the
    requested source frame from wall-clock video time, skipping frames as needed.
    """

    def __init__(self) -> None:
        self.estimated_seconds: float | None = None
        self.last_request_time: float | None = None

    def reset(self) -> None:
        """Forget the estimate when a different video is loaded."""
        self.estimated_seconds = None
        self.resume()

    def resume(self) -> None:
        """Keep the estimate but discard the prior playback session's clock."""
        self.last_request_time = None

    def begin_frame(self, now: float) -> None:
        self.last_request_time = float(now)

    def observe_frame(self, elapsed_seconds: float) -> None:
        duration = float(elapsed_seconds)
        if not math.isfinite(duration) or duration <= 0:
            return
        duration = min(duration, 2.0)
        old = self.estimated_seconds
        if old is None:
            self.estimated_seconds = duration
        else:
            # Do not mistake a single OS/GUI stall for a lasting drop in
            # decoding capacity. Repeated slow frames still raise the estimate.
            duration = min(duration, max(0.04, old * 2.5))
            weight = 0.30 if duration > old else 0.10
            self.estimated_seconds = old + weight * (duration - old)

    def delay_ms(self, now: float, frame_rate: float) -> int:
        """Milliseconds until another frame should be requested."""
        if self.last_request_time is None:
            return 0
        desired = 1.0 / max(1e-9, float(frame_rate))
        learned = (self.estimated_seconds or 0.0) * 1.15
        interval = max(desired, learned)
        return max(0, math.ceil((self.last_request_time + interval - now) * 1000.0))
