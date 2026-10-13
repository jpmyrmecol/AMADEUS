# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Find spatially persistent blobs in complete, consecutive video-frame windows.

This is deliberately not a background mask: only observed source blobs in
qualifying windows are excluded. It does not guess whether an animal is alive.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import cv2
import numpy as np


@dataclass
class _Observation:
    frame: int
    blob_index: int
    x: float
    y: float
    length: float
    contour: np.ndarray


@dataclass
class _Track:
    last: _Observation
    started: int
    observations: deque[_Observation] = field(default_factory=deque)


def contour_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Exact filled-contour IoU in the shared minimal bounding rectangle."""
    if np.array_equal(left, right):
        return 1.0
    lx, ly, lw, lh = cv2.boundingRect(left)
    rx, ry, rw, rh = cv2.boundingRect(right)
    x = min(lx, rx)
    y = min(ly, ry)
    w = max(lx + lw, rx + rw) - x
    h = max(ly + lh, ry + rh) - y
    if (min(lx + lw, rx + rw) <= max(lx, rx)
            or min(ly + lh, ry + rh) <= max(ly, ry)):
        return 0.0
    a = np.zeros((h, w), dtype=np.uint8)
    b = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(a, [left - (x, y)], -1, 1, cv2.FILLED)
    cv2.drawContours(b, [right - (x, y)], -1, 1, cv2.FILLED)
    union = cv2.countNonZero(cv2.bitwise_or(a, b))
    return cv2.countNonZero(cv2.bitwise_and(a, b)) / union if union else 0.0


class PersistentStaticScanner:
    """Streaming detector returning frame-to-blob keys only after a full window.

    A qualifying window needs the requested frame occupancy, median IoU
    against its temporal midpoint, and maximum 2D center span normalized by
    median OBB major-axis length. Motion is never assessed across gaps by a
    global fixed-location mask. The per-window qualifying observations are
    included retrospectively in the output.
    """

    def __init__(self, window_frames: int, presence: float = 0.98,
                 min_iou: float = 0.8, max_displacement: float = 0.2):
        self.window = int(window_frames)
        self.presence = float(presence)
        self.min_iou = float(min_iou)
        self.max_displacement = float(max_displacement)
        if self.window < 2:
            raise ValueError("Frame window must be >= 2.")
        if not 0 < self.presence <= 1:
            raise ValueError("Presence ratio must be in (0, 1].")
        if not 0 <= self.min_iou <= 1 or not 0 <= self.max_displacement <= 10:
            raise ValueError("Invalid static-blob similarity thresholds.")
        self._tracks: list[_Track] = []
        self._last_frame: int | None = None
        self.excluded: dict[int, set[int]] = {}
        self.qualifying_windows = 0

    @staticmethod
    def _observation(fid: int, blob) -> _Observation | None:
        if bool(blob.is_crossing):
            return None
        cnt = np.asarray(blob.contour, dtype=np.int32).reshape(-1, 1, 2)
        if cnt.shape[0] < 3:
            return None
        (_cx, _cy), (w, h), _angle = cv2.minAreaRect(cnt.astype(np.float32))
        length = float(max(w, h))
        if not math.isfinite(length) or length <= 0:
            return None
        return _Observation(
            int(fid), int(blob.blob_index),
            float(blob.center_x), float(blob.center_y), length, cnt.copy(),
        )

    def add_frame(self, frame: int, blobs) -> None:
        fid = int(frame)
        if self._last_frame is not None and fid != self._last_frame + 1:
            raise ValueError("Static-blob search requires consecutive video frames.")
        self._last_frame = fid

        # A missing detection for longer than the allowed fraction cannot
        # qualify in any subsequent complete window.
        allowed_missing = self.window - math.ceil(self.presence * self.window - 1e-9)
        candidates = [obs for blob in blobs
                      if (obs := self._observation(fid, blob)) is not None]
        active = [track for track in self._tracks
                  if fid - track.last.frame <= max(1, allowed_missing + 1)]
        remaining = set(range(len(active)))
        # Greedy unique matching: nearby contours only; no global location
        # suppression, so a different moving animal can later cross this area.
        for obs in candidates:
            best = None
            score = float("inf")
            for index in remaining:
                last = active[index].last
                dist = math.hypot(obs.x - last.x, obs.y - last.y)
                tolerance = max(1.0, self.max_displacement * max(obs.length, last.length))
                if dist > tolerance:
                    continue
                length_ratio = min(obs.length, last.length) / max(obs.length, last.length)
                if length_ratio < 0.6:
                    continue
                metric = dist / tolerance + (1.0 - length_ratio)
                if metric < score:
                    best = index
                    score = metric
            if best is None:
                active.append(_Track(last=obs, started=fid, observations=deque([obs])))
            else:
                track = active[best]
                remaining.remove(best)
                track.last = obs
                track.observations.append(obs)

        cutoff = fid - self.window + 1
        for track in active:
            while track.observations and track.observations[0].frame < cutoff:
                track.observations.popleft()
            if fid - track.started + 1 < self.window:
                continue
            observations = list(track.observations)
            if len(observations) < math.ceil(self.presence * self.window - 1e-9):
                continue
            axis = float(np.median([o.length for o in observations]))
            xs = [o.x for o in observations]
            ys = [o.y for o in observations]
            center_span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if axis <= 0 or center_span > self.max_displacement * axis:
                continue
            reference = observations[len(observations) // 2].contour
            ious = [contour_iou(reference, obs.contour) for obs in observations]
            if float(np.median(ious)) < self.min_iou:
                continue
            self.qualifying_windows += 1
            for obs in observations:
                self.excluded.setdefault(obs.frame, set()).add(obs.blob_index)
        self._tracks = active

    @property
    def excluded_count(self) -> int:
        return sum(len(indices) for indices in self.excluded.values())
