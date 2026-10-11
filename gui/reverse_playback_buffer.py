# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Bounded reverse playback prepared by one forward-decoding worker."""

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import threading
import time

import av
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


    def to_bgr(self) -> np.ndarray:
        """Decoded BGR image; used only by the two OpenCV-based preview GUIs."""
        if self.native is None:
            return self.pixels
        bgr = self.native.reformat(format="bgr24")
        plane = bgr.planes[0]
        pixels = np.ndarray(
            (bgr.height, bgr.width, 3), dtype=np.uint8, buffer=plane,
            strides=(plane.line_size, 3, 1),
        )
        rotations = {90: cv2.ROTATE_90_CLOCKWISE,
                     180: cv2.ROTATE_180,
                     270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        return cv2.rotate(pixels, rotations[self.rotation]) if self.rotation in rotations else pixels


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

    def prepare(
        self, center: int, speed: float = 1.0,
        continuous: bool = False, warm_blocks: int = 1,
    ) -> None:
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
                # Allow a longer speculative warm-up while a key hold is
                # being distinguished from a single-frame tap.
                self._warm_blocks = max(1, int(warm_blocks))
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


class SmoothPreviewPlayback:
    """Common asynchronous playback engine for Segmentation and Cropping.

    PyAV decodes reverse blocks into the existing bounded reverse buffer.
    A separate single worker performs forward decoding or reverse-frame color
    conversion. Tk never executes a decoder call during continuous playback.
    """

    def __init__(self, owner, *, reader, fps, frame_count, read_forward,
                 cached_frames, render, on_error, on_boundary,
                 lower_bound=0, upper_bound=None, rotation=0,
                 hold_is_active=None, on_hold_released=None):
        self.owner, self.reader = owner, reader
        self.fps = float(fps)
        self.frame_count = int(frame_count)
        self.read_forward, self.render = read_forward, render
        self.on_error, self.on_boundary = on_error, on_boundary
        # Optionally reject a stale held-key session before drawing another
        # frame, rather than relying exclusively on queued Tk KeyRelease events.
        self.hold_is_active = hold_is_active
        self.on_hold_released = on_hold_released
        self.lower_bound = int(lower_bound)
        self.upper_bound = self.frame_count - 1 if upper_bound is None else int(upper_bound)
        self.rotation = int(rotation) % 360
        self._container = av.open(reader.video_path)
        try:
            self._stream = self._container.streams.video[0]
            codec = self._stream.codec_context
            codec.thread_count = 4
            codec.thread_type = "AUTO"
            width, height = codec.width, codec.height
            native_bytes = math.ceil(
                (width + 64) * (height + 64) * codec.format.padded_bits_per_pixel / 8
            )
            self._reverse_lock = threading.Lock()
            self.reverse_buffer = ReversePlaybackBuffer(
                self._decode_reverse_block, frame_count=frame_count,
                fps=fps, frame_bytes=max((width + 64) * height * 3, native_bytes),
                seed_frames=cached_frames,
            )
        except Exception:
            self._container.close()
            raise

        self._cond = threading.Condition()
        self._closed = False
        self._request = None
        self._result = None
        self._generation = 0
        self._pending = False
        self._active = False
        self._direction = 1
        self._speed = 1.0
        self._timing = None
        self._waiting_reverse = False
        self._resume_clock = False
        self._timer = None
        self._worker = threading.Thread(
            target=self._decode_loop, name="preview-video-decoder", daemon=True,
        )
        self._worker.start()

    def _decode_reverse_block(self, first, last, stride, is_cancelled):
        with self._reverse_lock:
            if self._closed or is_cancelled():
                return
            stream = self._stream
            start = stream.start_time or 0
            self._container.seek(
                start + int(first / self.fps / stream.time_base),
                stream=stream, backward=True, any_frame=False,
            )
            expected = first
            for frame in self._container.decode(stream):
                if self._closed or is_cancelled():
                    return
                if frame.pts is None:
                    raise RuntimeError("Reverse frame has no presentation timestamp.")
                position = float((frame.pts - start) * stream.time_base) * self.fps
                index = round(position)
                if abs(index - position) > 0.1:
                    raise RuntimeError("Reverse preview requires constant frame timing.")
                if index < first:
                    continue
                if index != expected:
                    raise RuntimeError(f"Reverse decoding skipped frame {expected}.")
                if index % stride == 0 or index == last:
                    yield index, PreparedFrame(native=frame, rotation=self.rotation)
                if index == last:
                    return
                expected += 1

    def _decode_loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._closed or self._request is not None)
                if self._closed:
                    return
                generation, index, prepared, max_gap = self._request
                self._request = None
            try:
                if prepared is not None:
                    frame, actual = prepared.to_bgr(), index
                else:
                    frame, actual = self.read_forward(index, max_gap)
                if frame is None:
                    raise RuntimeError(f"Could not decode preview frame {index}.")
                result = (generation, actual, frame, None)
            except Exception as exc:
                result = (generation, index, None, exc)
            with self._cond:
                if not self._closed and generation == self._generation:
                    self._result = result

    def _schedule(self, delay=8):
        if self._timer is not None:
            try:
                self.owner.after_cancel(self._timer)
            except Exception:
                pass
            self._timer = None
        if not self._closed and (self._active or self._pending):
            self._timer = self.owner.after(max(1, int(delay)), self._tick)

    def _position(self, now):
        anchor, epoch, speed = self._timing
        return anchor + max(0.0, now - epoch) * self.fps * speed

    def start(self, direction, speed):
        if self._closed:
            return
        if direction not in (-1, 1):
            raise ValueError("Playback direction must be -1 or 1.")
        self._direction, self._speed = direction, float(speed)
        center = int(self.owner.current_frame)
        self._active = True
        self._timing = (float(center), time.perf_counter(), direction * self._speed)
        self._resume_clock = self._pending
        if direction < 0:
            self.reverse_buffer.start(center, self._speed)
            self._waiting_reverse = not self.reverse_buffer.ready(center)
        else:
            self.reverse_buffer.stop()
            self._waiting_reverse = False
        self._schedule(1)

    def stop(self, *, discard_pending: bool = False):
        """Stop playback; optionally discard a delayed, not-yet-rendered frame.

        Reverse key holds must stop at the visible frame when the key is
        released, even if the decoder is still converting the next one.
        Ordinary Play/Stop preserves the previous pending-frame behavior.
        """
        self._active = False
        self._timing = None
        self.reverse_buffer.stop()
        self._waiting_reverse = False
        self._resume_clock = False
        if discard_pending:
            with self._cond:
                self._generation += 1
                self._request = None
                self._result = None
                self._pending = False
                self._cond.notify_all()
        self._schedule(8)  # Flush an accepted frame unless discarded.

    def set_speed(self, speed):
        self._speed = float(speed)
        if not self._active or self._timing is None:
            return
        now = time.perf_counter()
        position = self._position(now)
        self._timing = (position, now, self._direction * self._speed)
        if self._direction < 0:
            center = int(self.owner.current_frame)
            self.reverse_buffer.start(center, self._speed)
            self._waiting_reverse = not self.reverse_buffer.ready(center)
        self._schedule(1)

    def request_frame(self, index: int) -> bool:
        """Queue a single arrow-key step without decoding on the Tk thread.

        A newer step supersedes an older one; the decoder worker itself is
        single-threaded, so late results cannot overwrite the current image.
        """
        if self._closed:
            return False
        if self._active:
            self.stop(discard_pending=True)
        self.invalidate()
        self._queue_frame(index)
        self._schedule(1)
        return True

    @property
    def pending(self) -> bool:
        return self._pending

    def invalidate(self):
        """Invalidate stale decoder results after manual seeking."""
        with self._cond:
            self._generation += 1
            self._request = None
            self._result = None
            self._pending = False
        self._schedule()

    def _receive(self):
        with self._cond:
            result, self._result = self._result, None
        if result is None:
            return
        _, frame_idx, image, error = result
        self._pending = False
        if error is not None:
            self.stop()
            self.on_error(error)
            return
        self.render(image, frame_idx)
        if self._active:
            if self._direction < 0:
                self.reverse_buffer.advance(frame_idx)
            if self._resume_clock:
                # A held arrow may begin while its initial one-frame tap is
                # still decoding. Anchor reverse prefetch and the playback
                # clock to the frame actually displayed, not the stale origin.
                if self._direction < 0:
                    self.reverse_buffer.start(frame_idx, self._speed)
                    self._waiting_reverse = not self.reverse_buffer.ready(frame_idx)
                self._timing = (float(frame_idx), time.perf_counter(),
                                self._direction * self._speed)
                self._resume_clock = False

    def _queue_frame(self, index, prepared=None):
        with self._cond:
            if self._closed or self._pending:
                return
            self._generation += 1
            gap = max(12, math.ceil(self.fps * max(1.0, self._speed) * 2))
            self._request = (self._generation, int(index), prepared, gap)
            self._pending = True
            self._cond.notify()

    def _tick(self):
        self._timer = None
        if self._closed:
            return
        if (self._active and self.hold_is_active is not None
                and not self.hold_is_active()):
            # On Windows, native auto-repeat can queue KeyPress events ahead
            # of KeyRelease while the canvas is being redrawn. Query the
            # physical state before *every* playback render, and let the GUI
            # tear down the key-held session immediately.
            if self.on_hold_released is not None:
                self.on_hold_released()
            else:
                self.stop(discard_pending=True)
            return
        self._receive()
        if not self._active or self._timing is None:
            self._schedule()
            return
        current = int(self.owner.current_frame)
        bound = self.lower_bound if self._direction < 0 else self.upper_bound
        if (current - bound) * self._direction >= 0:
            self.on_boundary()
            return
        if self._pending:
            self._schedule(8)
            return
        now = time.perf_counter()
        target = (math.ceil(self._position(now)) if self._direction < 0
                  else int(self._position(now)))
        target = min(self.upper_bound, max(self.lower_bound, target))
        if (target - current) * self._direction <= 0:
            rate = self.fps * self._speed
            anchor, epoch, signed_speed = self._timing
            due = epoch + (current + self._direction - anchor) / (self.fps * signed_speed)
            self._schedule(max(1, int((due - now) * 1000)))
            return

        prepared = None
        if self._direction < 0:
            error = self.reverse_buffer.error
            if error is not None:
                self.stop()
                self.on_error(error)
                return
            if self._waiting_reverse:
                if not self.reverse_buffer.ready(current):
                    self._schedule(8)
                    return
                self._timing = (float(current), now, -self._speed)
                self._waiting_reverse = False
                self._schedule(1)
                return
            choice = self.reverse_buffer.get_frame(target, current)
            if choice is None:
                self.reverse_buffer.advance(current)
                self._waiting_reverse = True
                self._schedule(8)
                return
            target, prepared = choice

        self._queue_frame(target, prepared)
        self._schedule(8)

    def close(self):
        if self._timer is not None:
            try:
                self.owner.after_cancel(self._timer)
            except Exception:
                pass
            self._timer = None
        self._active = False
        self._closed = True
        self.reverse_buffer.close()
        with self._cond:
            self._request = None
            self._result = None
            self._cond.notify_all()
        def cleanup():
            with self._reverse_lock:
                self._container.close()
        threading.Thread(target=cleanup, name="preview-ffmpeg-cleanup",
                         daemon=True).start()
