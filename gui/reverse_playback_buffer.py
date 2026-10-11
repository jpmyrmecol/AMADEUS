# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Bounded reverse playback prepared by one forward-decoding worker."""

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import av


MAX_REVERSE_SAMPLES = 512
MAX_REVERSE_BYTES = 6 * 1024 ** 3
REVERSE_SAMPLE_FPS = 60.0
REVERSE_HISTORY_FRAMES = 128


@dataclass(frozen=True)
class PreparedFrame:
    pixels: np.ndarray | None = None
    native: "av.VideoFrame | None" = None
    rotation: int = 0

    @property
    def nbytes(self) -> int:
        if self.native is not None:
            return sum(plane.buffer_size for plane in self.native.planes)
        return max(self.pixels.nbytes, self.pixels.shape[0] * abs(self.pixels.strides[0]))

    def to_rgb(self) -> np.ndarray:
        if self.native is None:
            return self.pixels
        rgb = self.native.reformat(format="rgb24")
        plane = rgb.planes[0]
        pixels = np.ndarray(
            (rgb.height, rgb.width, 3), dtype=np.uint8, buffer=plane,
            strides=(plane.line_size, 3, 1),
        )
        rotations = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        if self.rotation in rotations:
            pixels = cv2.rotate(pixels, rotations[self.rotation])
        return pixels


class ReversePlaybackBuffer:
    """Keep consecutive source intervals for time-synchronized reverse display.

    ``decode_block`` yields ``(frame_index, PreparedFrame)`` in ascending order. It
    grabs every source frame but retrieves only indexes divisible by ``stride``
    and the last index. It checks ``is_cancelled`` between native codec calls.
    Each interval finishes independently of ordinary display-position updates.
    """

    def __init__(
        self,
        decode_block: Callable[
            [int, int, int, Callable[[], bool]], Iterable[tuple[int, PreparedFrame]]
        ],
        *,
        frame_count: int,
        fps: float,
        frame_bytes: int,
        seed_frames: Callable[[int], Mapping[int, PreparedFrame]] | None = None,
        target_fps: float = REVERSE_SAMPLE_FPS,
    ):
        if frame_count < 1 or not math.isfinite(fps) or fps <= 0 or frame_bytes < 1:
            raise ValueError("Invalid video metadata for reverse playback.")
        self.frame_count = int(frame_count)
        self.fps = float(fps)
        self.frame_bytes = int(frame_bytes)
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("Reverse display rate must be positive and finite.")
        self.target_fps = float(target_fps)
        self.capacity = min(MAX_REVERSE_SAMPLES, MAX_REVERSE_BYTES // self.frame_bytes)
        if self.capacity < 4:
            raise ValueError("A video frame exceeds the reverse playback memory budget.")
        self._history_source_frames = min(REVERSE_HISTORY_FRAMES, self.capacity // 4)
        self._decode_block = decode_block
        self._seed_frames = seed_frames
        self._cond = threading.Condition()
        self._frames: OrderedDict[int, PreparedFrame] = OrderedDict()
        self._bytes = 0
        self._closed = False
        self._active = False
        self._warm_blocks = 0
        self._generation = 0
        self._center = 0
        self._anchor: int | None = None
        self._stride = 1
        self._rate = self.fps
        self._chunk_span = 1
        self._bootstrap_span = 1
        self._ready_span = 1
        self._bootstrap_blocks = 0
        self._next_last = -1
        self._loading: tuple[int, int, int] | None = None
        self._error: Exception | None = None
        self._blocks_started = 0
        self._blocks_completed = 0
        self._blocks_cancelled = 0
        self._decode_failures = 0
        self._thread = threading.Thread(
            target=self._worker_loop, name="reverse-video-buffer", daemon=True
        )
        self._thread.start()

    def _configuration(self, speed: float) -> tuple[int, int, int, int]:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Reverse playback speed must be positive and finite.")
        rate = self.fps * speed
        block_samples = max(1, (self.capacity - self._history_source_frames - 2) // 2)
        stride = max(1, math.ceil(rate / self.target_fps))
        if rate > self.fps or rate > self.target_fps or block_samples < rate * 2.0:
            stride = max(stride, math.ceil(rate * 4.0 / block_samples))
        chunk_span = stride * min(block_samples, max(1, math.ceil(rate * 4.0 / stride)))
        bootstrap_span = min(chunk_span, stride * max(1, math.ceil(rate * 0.5 / stride)))
        ready_span = min(
            bootstrap_span * 2,
            stride * max(1, math.ceil(rate * 0.75 / stride)),
        )
        return stride, chunk_span, bootstrap_span, ready_span

    def _remove_locked(self, index: int) -> None:
        frame = self._frames.pop(index)
        self._bytes -= frame.nbytes

    def _discard_consumed_locked(self) -> None:
        for index in list(self._frames):
            if index > self._center + self._history_source_frames:
                self._remove_locked(index)

    def _contiguous_lower_locked(self, center: int) -> int | None:
        indexes = sorted((index for index in self._frames if index < center), reverse=True)
        if not indexes or center - 1 - indexes[0] > self._stride:
            return None
        lower = indexes[0]
        for index in indexes[1:]:
            if lower - index > self._stride * 2:
                break
            lower = index
        return lower

    def _store_locked(self, index: int, frame: PreparedFrame) -> bool:
        if index > self._center + self._history_source_frames or index < 0 or index >= self.frame_count:
            return False
        if index in self._frames:
            return True
        if len(self._frames) >= self.capacity or self._bytes + frame.nbytes > MAX_REVERSE_BYTES:
            return False
        if frame.pixels is not None:
            frame.pixels.setflags(write=False)
        self._frames[index] = frame
        self._bytes += frame.nbytes
        return True

    def prepare(self, center: int, speed: float = 1.0, continuous: bool = False) -> None:
        center = min(self.frame_count - 1, max(0, int(center)))
        configuration = self._configuration(float(speed))
        seeds = self._seed_frames(center) if self._seed_frames is not None else {}
        with self._cond:
            if self._closed:
                return
            same_configuration = configuration == (
                self._stride, self._chunk_span, self._bootstrap_span, self._ready_span
            )
            lower = self._contiguous_lower_locked(center) if same_configuration else None
            same_position = (
                self._anchor is not None
                and (abs(center - self._center) <= self._stride or lower is not None)
            )
            reuse = same_configuration and same_position and self._error is None
            if not reuse:
                self._generation += 1
                self._stride, self._chunk_span, self._bootstrap_span, self._ready_span = configuration
                self._anchor = center
                self._center = center
                self._active = False
                self._warm_blocks = 0
                self._error = None
                # A manual jump or a new sampling rate starts a new sequence.
                # Keep useful prepared images close to its initial playhead.
                oldest = max(0, center - self._chunk_span * 2)
                for index in list(self._frames):
                    if index < oldest or index > center + self._history_source_frames:
                        self._remove_locked(index)
                self._bootstrap_blocks = 1
            self._center = center
            self._rate = self.fps * float(speed)
            self._discard_consumed_locked()
            # Keep the seed indexes themselves instead of imposing our sample
            # grid on forward display's skipped frames. Thin dense seeds to the
            # sampling cadence so they cannot exhaust the decoding reservation.
            previous_seed = None
            for index, frame in sorted(seeds.items(), reverse=True):
                if not center - self._chunk_span * 2 <= index < center:
                    continue
                if previous_seed is None:
                    if center - 1 - index > self._stride:
                        break
                else:
                    gap = previous_seed - index
                    if gap < self._stride:
                        continue
                    if gap > self._stride * 2:
                        break
                if not self._store_locked(int(index), frame):
                    break
                previous_seed = int(index)
            if not reuse:
                lower = self._contiguous_lower_locked(center)
                self._next_last = center - 1 if lower is None else lower - 1
            if continuous:
                self._active = True
                self._warm_blocks = 0
            elif not self._active and not reuse:
                self._warm_blocks = 1
            self._cond.notify_all()

    def start(self, center: int, speed: float = 1.0) -> None:
        self.prepare(center, speed, continuous=True)

    def advance(self, center: int) -> None:
        with self._cond:
            if self._closed:
                return
            self._center = min(self.frame_count - 1, max(0, int(center)))
            self._discard_consumed_locked()
            self._cond.notify_all()

    def stop(self) -> None:
        """Cancel codec work without waiting, and preserve unused ready images."""
        with self._cond:
            self._active = False
            self._warm_blocks = 0
            self._generation += 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._active = False
            self._warm_blocks = 0
            self._generation += 1
            self._frames.clear()
            self._bytes = 0
            self._cond.notify_all()

    def get_frame(self, target: int, current: int) -> tuple[int, PreparedFrame] | None:
        """Return a real prepared frame near the clock, strictly behind current."""
        target = int(target)
        current = int(current)
        with self._cond:
            candidates = [
                index for index in self._frames
                if index < current and abs(index - target) <= self._stride
            ]
            if not candidates:
                return None
            index = min(candidates, key=lambda value: (abs(value - target), -value))
            return index, self._frames[index]

    def get_exact(self, index: int) -> PreparedFrame | None:
        """Return an exact source frame, including recently consumed neighbors."""
        with self._cond:
            return self._frames.get(int(index))

    def ready(self, center: int) -> bool:
        center = int(center)
        with self._cond:
            if self._closed or self._error is not None:
                return False
            if center <= 0:
                return True
            lower = self._contiguous_lower_locked(center)
            return lower is not None and center - lower >= min(center, self._ready_span)

    @property
    def error(self) -> Exception | None:
        with self._cond:
            return self._error

    def stats(self) -> dict:
        with self._cond:
            lower = self._contiguous_lower_locked(self._center)
            return {
                "cached_samples": len(self._frames),
                "bytes": self._bytes,
                "capacity": self.capacity,
                "history_source_frames": self._history_source_frames,
                "stride": self._stride,
                "target_fps": self.target_fps,
                "ready_source_frames": 0 if lower is None else self._center - lower,
                "required_source_frames": self._ready_span,
                "first": min(self._frames) if self._frames else None,
                "last": max(self._frames) if self._frames else None,
                "generation": self._generation,
                "active": self._active,
                "loading": self._loading,
                "blocks_started": self._blocks_started,
                "blocks_completed": self._blocks_completed,
                "blocks_cancelled": self._blocks_cancelled,
                "decode_failures": self._decode_failures,
            }

    def _sample_indexes(self, first: int, last: int) -> set[int]:
        first_sample = ((first + self._stride - 1) // self._stride) * self._stride
        indexes = set(range(first_sample, last + 1, self._stride))
        indexes.add(last)
        return indexes

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(
                    lambda: self._closed or (
                        self._error is None
                        and (self._active or self._warm_blocks > 0)
                        and self._next_last >= 0
                    )
                )
                if self._closed:
                    return
                generation = self._generation
                last = self._next_last
                if self._bootstrap_blocks > 0:
                    span = self._bootstrap_span
                else:
                    lower = self._contiguous_lower_locked(self._center)
                    coverage = 0 if lower is None else self._center - lower
                    # Grow intervals only after their preparation can be hidden
                    # by the queue. A short seed from forward navigation should
                    # keep moving while the next older interval is decoded.
                    desired_span = max(self._rate, coverage * 0.5)
                    span = min(
                        self._chunk_span,
                        self._stride * max(1, math.ceil(desired_span / self._stride)),
                    )
                first = max(0, last - span + 1)
                new_indexes = self._sample_indexes(first, last).difference(self._frames)
                # Reserve an entire interval. Never evict an image that reverse
                # display has yet to consume just to fit the loading interval.
                while (
                    len(self._frames) + len(new_indexes) > self.capacity
                    or self._bytes + len(new_indexes) * self.frame_bytes > MAX_REVERSE_BYTES
                ):
                    self._cond.wait()
                    if self._closed:
                        return
                    if generation != self._generation:
                        break
                    new_indexes = self._sample_indexes(first, last).difference(self._frames)
                if generation != self._generation:
                    continue
                self._loading = (generation, first, last)
                self._blocks_started += 1
                stride = self._stride
                expected_indexes = sorted(self._sample_indexes(first, last))

            def is_cancelled() -> bool:
                with self._cond:
                    return self._closed or generation != self._generation

            error: Exception | None = None
            complete = False
            try:
                received = 0
                for index, frame in self._decode_block(first, last, stride, is_cancelled):
                    with self._cond:
                        if self._closed or generation != self._generation:
                            break
                        if frame.nbytes > self.frame_bytes:
                            raise RuntimeError("Reverse decoded frame exceeds its memory reservation.")
                        if received >= len(expected_indexes) or index != expected_indexes[received]:
                            raise RuntimeError("Reverse decoder yielded an incomplete or unordered interval.")
                        if index <= self._center + self._history_source_frames and not self._store_locked(index, frame):
                            raise RuntimeError("Reverse decoder exceeded its reserved memory budget.")
                        received += 1
                        self._cond.notify_all()
                complete = received == len(expected_indexes) and not is_cancelled()
                if not complete and not is_cancelled():
                    raise RuntimeError(f"Reverse decoding ended before frame {last}.")
            except Exception as exc:
                error = exc
            with self._cond:
                self._loading = None
                if self._closed:
                    return
                if generation != self._generation:
                    self._blocks_cancelled += 1
                    self._cond.notify_all()
                    continue
                if error is not None:
                    self._error = error
                    self._decode_failures += 1
                    self._active = False
                    self._warm_blocks = 0
                elif complete:
                    self._blocks_completed += 1
                    self._next_last = first - 1
                    self._bootstrap_blocks = max(0, self._bootstrap_blocks - 1)
                    if not self._active:
                        self._warm_blocks = max(0, self._warm_blocks - 1)
                self._cond.notify_all()
