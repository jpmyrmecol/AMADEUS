# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FRAME_COUNT_TAIL_PROBE = 512


@dataclass(frozen=True)
class VideoFrameInfo:
    path: str
    reported_frame_count: int
    usable_frame_count: int
    width: int
    height: int
    fps: float

    @property
    def adjusted(self) -> bool:
        return self.usable_frame_count != self.reported_frame_count


def _is_valid_frame(frame) -> bool:
    return frame is not None and getattr(frame, "size", 0) > 0


def _cv2() -> Any:
    import cv2

    return cv2


def _try_seek_read(cap: Any, frame_idx: int) -> bool:
    cv2 = _cv2()
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    except Exception:
        return False
    return bool(ok and _is_valid_frame(frame))


def detect_seekable_frame_count(
    video_path: str,
    reported_count: int | None = None,
    *,
    tail_probe: int = FRAME_COUNT_TAIL_PROBE,
) -> int:
    """Return the highest frame count that OpenCV can seek/read reliably.

    Some MOV/HEVC files report more frames than OpenCV can directly decode near
    the tail. Tracking scripts use direct frame addressing, so the stable upper
    bound is the last frame that succeeds after CAP_PROP_POS_FRAMES + read().
    """
    cv2 = _cv2()
    probe_cap = cv2.VideoCapture(str(video_path))
    if not probe_cap.isOpened():
        return max(0, int(reported_count or 0))
    try:
        if reported_count is None:
            reported_count = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        reported_count = max(0, int(reported_count))
        if reported_count <= 0:
            return reported_count

        if _try_seek_read(probe_cap, reported_count - 1):
            return reported_count

        tail_start = max(0, reported_count - max(1, int(tail_probe)))
        for frame_idx in range(reported_count - 2, tail_start - 1, -1):
            if _try_seek_read(probe_cap, frame_idx):
                return frame_idx + 1

        best = -1
        lo, hi = 0, tail_start - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _try_seek_read(probe_cap, mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best + 1 if best >= 0 else reported_count
    finally:
        probe_cap.release()


def read_video_frame_info(video_path: str) -> VideoFrameInfo:
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(str(video_path))
    try:
        reported_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    usable_count = detect_seekable_frame_count(video_path, reported_count)
    return VideoFrameInfo(
        path=str(video_path),
        reported_frame_count=reported_count,
        usable_frame_count=usable_count,
        width=width,
        height=height,
        fps=fps if fps > 0 else 0.0,
    )


def warn_if_frame_count_adjusted(info: VideoFrameInfo, *, label: str = "video") -> None:
    if not info.adjusted:
        return
    print(
        f"[WARN] {label}: limiting frame range to usable frames "
        f"{info.usable_frame_count} (reported {info.reported_frame_count})",
        flush=True,
    )


def clamp_frame_range_to_usable_count(
    first_frame: int,
    last_frame: int,
    usable_frame_count: int,
) -> tuple[int, int]:
    first_frame = max(0, int(first_frame))
    usable_frame_count = max(0, int(usable_frame_count))
    if usable_frame_count <= 0:
        raise RuntimeError("No usable video frames were found.")

    last_frame = int(last_frame)
    if last_frame < 0 or last_frame >= usable_frame_count:
        last_frame = usable_frame_count - 1
    if last_frame < first_frame:
        raise RuntimeError(f"Invalid frame range: first={first_frame}, last={last_frame}")
    return first_frame, last_frame
