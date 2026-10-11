# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Regression coverage for refinement playback without a Tk display.

The controller and decoder methods are the application implementations. Only
the Tk scheduler/widgets and video backend are replaced with stateful fakes.
One test also decodes a temporary video through the real OpenCV backend.
"""

import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from gui.gui_refinement import (
    FRAME_CACHE_SIZE,
    UmaDirectionRefinementApp,
    VideoFrameReader,
)
from gui.preview_playback_timing import AdaptivePreviewPacer


class Value:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeCapture:
    """Model the backend's next-frame position, including grabs and seeks."""

    def __init__(self, frame_count=1000, fps=30.0, opened=True):
        self.frame_count = frame_count
        self.fps = fps
        self.opened = opened
        self.position = 0
        self.seeks = []
        self.grabs = 0
        self.reads = []
        self.released = threading.Event()

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_POS_FRAMES: self.position,
        }[prop]

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            raise AssertionError(f"Unexpected capture property: {prop}")
        self.position = int(value)
        self.seeks.append(self.position)
        return True

    def grab(self):
        if self.position >= self.frame_count:
            return False
        self.position += 1
        self.grabs += 1
        return True

    def read(self):
        if self.position >= self.frame_count:
            return False, None
        index = self.position
        self.position += 1
        self.reads.append(index)
        value = index % 200
        frame = np.empty((64, 64, 3), dtype=np.uint8)
        frame[:] = (value, value + 1, value + 2)
        return True, frame

    def release(self):
        self.released.set()


class NavigationReader:
    def __init__(self, fps=30.0):
        self.fps = fps
        self.cancel_count = 0
        self.prefetches = []

    def cancel_prefetch(self):
        self.cancel_count += 1

    def request_prefetch(self, center, **kwargs):
        self.prefetches.append(center)

    def get_cached(self, frame):
        return None


class ArrowKeys:
    repeating = False

    def stop(self):
        pass


class CanvasSize:
    def winfo_width(self):
        return 64

    def winfo_height(self):
        return 64


class ImageCanvas(CanvasSize):
    def __init__(self):
        self.items = {}
        self.created = 0
        self.deleted = []

    def create_image(self, x, y, anchor, image):
        self.created += 1
        self.items[self.created] = {"coords": (x, y), "anchor": anchor, "image": image}
        return self.created

    def coords(self, item, x, y):
        self.items[item]["coords"] = (x, y)

    def itemconfigure(self, item, image):
        self.items[item]["image"] = image

    def delete(self, item):
        self.deleted.append(item)
        if item != "all":
            raise AssertionError(f"Unexpected deleted item: {item}")
        self.items.clear()


class PhotoImage:
    """Keep pixels so reuse tests also prove that each displayed image changes."""

    def __init__(self, image):
        self.pixels = np.array(image, copy=True)
        self.pastes = 0

    def width(self):
        return self.pixels.shape[1]

    def height(self):
        return self.pixels.shape[0]

    def paste(self, image):
        pixels = np.array(image, copy=True)
        if pixels.shape != self.pixels.shape:
            raise AssertionError("PhotoImage.paste received a different size")
        self.pixels[:] = pixels
        self.pastes += 1


class HeadlessApp(UmaDirectionRefinementApp):
    """Exercise controller methods while making scheduled work explicit."""

    def __init__(self, reader=None, current=10, frame_count=1000):
        self.reader = reader if reader is not None else NavigationReader()
        self.frame_count = frame_count
        self.current_frame = current
        self.requested_frame = current
        self._last_displayed_frame = current
        self.playback_active = False
        self.playback_speed = 1.0
        self._playback_timing = None
        self._paused_decode_gen = None
        self.playback_job = None
        self.frame_render_job = None
        self.frame_render_interval_ms = 8
        self.play_button = None
        self.speed_button = None
        self._preview_pacer = AdaptivePreviewPacer()
        self._decode_gen = 0
        self._decode_request_queue = queue.Queue(maxsize=1)
        self._frame_queue = queue.Queue()
        self._decode_stop = threading.Event()
        self._closing = False
        self._companion_save_messages = queue.Queue()
        self._companion_save_active = False
        self.frame_var = Value(current)
        self.frame_scale = Value(current)
        self.frame_cache_rgb = None
        self.drag_mode = None
        self.drag_tid = None
        self.drag_last_canvas = None
        self.left_button_down = False
        self._arrow_key_play = ArrowKeys()
        self.jobs = {}
        self.cancelled = []
        self.draws = []
        self.info_updates = 0
        self.statuses = []
        self.next_job = 0

    def after(self, delay, callback):
        self.next_job += 1
        job = str(self.next_job)
        self.jobs[job] = (delay, callback)
        return job

    def after_cancel(self, job):
        self.cancelled.append(job)
        self.jobs.pop(job, None)

    def _draw_canvas(self, frame, preserve_view):
        self.draws.append((self.current_frame, frame, preserve_view))

    def update_info_panel(self):
        self.info_updates += 1

    def set_status(self, text, auto_clear=True):
        self.statuses.append(text)


class RefinementPlaybackTests(unittest.TestCase):
    def make_reader(self, frame_count=1000, fps=30.0):
        capture = FakeCapture(frame_count=frame_count, fps=fps)
        unavailable_prefetch = FakeCapture(opened=False)
        with patch("gui.gui_refinement.cv2.VideoCapture", side_effect=[capture, unavailable_prefetch]):
            reader = VideoFrameReader("test-video")

        def cleanup():
            reader.close()
            self.assertTrue(capture.released.wait(2.0), "Capture cleanup did not finish")

        self.addCleanup(cleanup)
        return reader, capture

    def decode_one_request(self, app):
        worker = threading.Thread(target=app._decode_worker_loop, daemon=True)
        worker.start()
        try:
            result = app._frame_queue.get(timeout=2.0)
        finally:
            app._decode_stop.set()
            app._drain_queue(app._decode_request_queue)
            app._decode_request_queue.put_nowait(None)
            worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive(), "Decode worker did not stop")
        self.assertIsNone(result[-1], f"Decode failed: {result[-1]}")
        return result

    def test_first_frame_matches_real_video_without_seek(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "indexed.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
            self.assertTrue(writer.isOpened(), "OpenCV MJPEG writer is required")
            for value in (20, 70, 120, 170):
                writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
            writer.release()
            original_capture = cv2.VideoCapture
            capture = original_capture(str(video))
            with patch("gui.gui_refinement.cv2.VideoCapture", side_effect=[capture, FakeCapture(opened=False)]):
                reader = VideoFrameReader(str(video))
            try:
                self.assertEqual(reader._cap_pos, 0)
                self.assertEqual(reader._prefetch_pos, 0)
                self.assertAlmostEqual(float(reader.read(0).mean()), 20.0, delta=3.0)
                self.assertAlmostEqual(float(reader.read(1).mean()), 70.0, delta=3.0)
                self.assertEqual(reader._cap_pos, 2)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_POS_FRAMES)), 2)
            finally:
                reader.close()
                # Close's asynchronous release must finish before deleting the file.
                with reader._cap_lock:
                    capture.release()

    def test_nearby_frames_skip_with_grab_and_manual_far_navigation_seeks(self):
        reader, capture = self.make_reader()
        np.testing.assert_array_equal(reader.read(0)[0, 0], [2, 1, 0])
        np.testing.assert_array_equal(reader.read(10)[0, 0], [12, 11, 10])
        self.assertEqual(capture.grabs, 9)
        self.assertEqual(capture.seeks, [])
        reader.read(100)
        self.assertEqual(capture.seeks, [100])
        reader.read(5)
        self.assertEqual(capture.seeks, [100, 5])
        self.assertEqual(capture.reads, [0, 10, 100, 5])

    def test_partial_grab_failure_invalidates_position_and_retry_seeks_correctly(self):
        reader, capture = self.make_reader()
        reader.read(0)
        original_grab = capture.grab
        attempts = 0

        def fail_after_progress():
            nonlocal attempts
            attempts += 1
            if attempts == 3:
                return False
            return original_grab()

        with patch.object(capture, "grab", side_effect=fail_after_progress):
            with self.assertRaisesRegex(RuntimeError, "Failed to skip to frame 8"):
                reader.read(8)
        self.assertEqual(capture.position, 3)
        self.assertEqual(reader._cap_pos, -1)
        self.assertIsNone(reader.get_cached(8))
        frame = reader.read(8)
        self.assertEqual(capture.seeks, [8])
        self.assertEqual(capture.reads, [0, 8])
        self.assertEqual(reader._cap_pos, 9)
        self.assertEqual(capture.position, 9)
        np.testing.assert_array_equal(frame[0, 0], [10, 9, 8])

    def test_playback_catches_up_sequentially_beyond_manual_seek_threshold(self):
        reader, capture = self.make_reader()
        reader.read(0)
        app = HeadlessApp(reader=reader, current=0)
        app.requested_frame = 1
        app.playback_active = True
        app._playback_timing = (0.0, 0.0, 2.0)
        app._enqueue_decode_request(0, reader, 1, False)
        with patch("gui.gui_refinement.time.perf_counter", return_value=1.0):
            result = self.decode_one_request(app)
        self.assertEqual(result[2], 60)
        self.assertEqual(capture.reads, [0, 60])
        self.assertEqual(capture.grabs, 59)
        self.assertEqual(capture.seeks, [])

    def test_playback_far_jump_remains_bounded(self):
        reader, capture = self.make_reader()
        reader.read(0)
        app = HeadlessApp(reader=reader, current=0)
        app.playback_active = True
        app.requested_frame = 1
        app._playback_timing = (0.0, 0.0, 1.0)
        app._enqueue_decode_request(0, reader, 1, False)
        with patch("gui.gui_refinement.time.perf_counter", return_value=10.0):
            result = self.decode_one_request(app)
        self.assertEqual(result[2], 300)
        self.assertEqual(capture.grabs, 0)
        self.assertEqual(capture.seeks, [300])

    def test_cache_is_shared_immutable_and_overlay_keeps_source_pixels(self):
        reader, capture = self.make_reader()
        rgb = reader.read(0)
        self.assertIs(reader.read(0), rgb)
        self.assertIs(reader.get_cached(0), rgb)
        self.assertEqual(capture.reads, [0])
        self.assertFalse(rgb.flags.writeable)
        original = rgb.copy()
        with self.assertRaises(ValueError):
            rgb[0, 0] = (100, 100, 100)
        app = HeadlessApp(reader=reader, current=0)
        app.view_initialized = True
        app.canvas = CanvasSize()
        app.scale = 1.0
        app.offset_x = app.offset_y = 0.0
        app.df_np = np.array([[45.0, 32.0, 15.0, 32.0]])
        app.num_ids = 1
        app._front_x_idx = [0]
        app._front_y_idx = [1]
        app._rear_x_idx = [2]
        app._rear_y_idx = [3]
        app.reverse_direction_var = Value(False)
        app.arrow_width_var = Value(2)
        app.label_size_var = Value(12)
        app.show_arrows_var = Value(True)
        app.show_labels_var = Value(False)
        app.selected_tids = []
        app.id_palette = [(255, 0, 0)]
        display = app._render_display_frame(rgb)
        self.assertFalse(np.shares_memory(display, rgb))
        self.assertTrue(np.any(display != original), "Actual arrow drawing did not alter the display")
        np.testing.assert_array_equal(reader.get_cached(0), original)

    def test_cache_eviction_is_bounded(self):
        reader, _ = self.make_reader()
        for index in range(FRAME_CACHE_SIZE + 3):
            reader._store_cached(index, np.zeros((1, 1, 3), dtype=np.uint8))
        self.assertEqual(len(reader.cache), FRAME_CACHE_SIZE)
        self.assertIsNone(reader.get_cached(0))
        self.assertIsNotNone(reader.get_cached(FRAME_CACHE_SIZE + 2))

    def presentation_app(self):
        app = HeadlessApp()
        app.canvas = ImageCanvas()
        app.view_initialized = False
        app.current_image_shape = None
        app.tk_image = None
        app._canvas_image_id = None
        return app

    def test_same_size_presentation_reuses_photo_and_canvas_item_with_new_pixels(self):
        app = self.presentation_app()
        first = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        second = np.flip(first, axis=1).copy()
        original_first, original_second = first.copy(), second.copy()
        first.setflags(write=False)
        second.setflags(write=False)
        with patch("gui.gui_refinement.ImageTk.PhotoImage", PhotoImage):
            app._present_image(first, preserve_view=True)
            photo, item = app.tk_image, app._canvas_image_id
            before = photo.pixels.copy()
            app._present_image(second, preserve_view=True)
        self.assertIs(app.tk_image, photo)
        self.assertEqual(app._canvas_image_id, item)
        self.assertEqual(app.canvas.created, 1)
        self.assertEqual(app.canvas.deleted, [])
        self.assertEqual(photo.pastes, 1)
        self.assertTrue(np.any(photo.pixels != before))
        np.testing.assert_array_equal(photo.pixels, cv2.resize(second, (64, 64), interpolation=cv2.INTER_LINEAR))
        np.testing.assert_array_equal(first, original_first)
        np.testing.assert_array_equal(second, original_second)

    def test_size_change_allocates_photo_and_updates_existing_canvas_item(self):
        app = self.presentation_app()
        first = np.full((32, 32, 3), (20, 40, 60), dtype=np.uint8)
        second = np.full((32, 64, 3), (80, 100, 120), dtype=np.uint8)
        original = second.copy()
        second.setflags(write=False)
        with patch("gui.gui_refinement.ImageTk.PhotoImage", PhotoImage):
            app._present_image(first, preserve_view=True)
            photo, item = app.tk_image, app._canvas_image_id
            app._present_image(second, preserve_view=True)
        self.assertIsNot(app.tk_image, photo)
        self.assertEqual((app.tk_image.width(), app.tk_image.height()), (64, 32))
        self.assertEqual(app._canvas_image_id, item)
        self.assertEqual(app.canvas.created, 1)
        self.assertEqual(app.canvas.deleted, [])
        self.assertEqual(app.canvas.items[item]["coords"], (0, 16))
        self.assertIs(app.canvas.items[item]["image"], app.tk_image)
        np.testing.assert_array_equal(app.tk_image.pixels, second)
        np.testing.assert_array_equal(second, original)

    def test_fully_offscreen_then_visible_creates_valid_new_canvas_item(self):
        app = self.presentation_app()
        rgb = np.full((32, 32, 3), (50, 100, 150), dtype=np.uint8)
        original = rgb.copy()
        rgb.setflags(write=False)
        with patch("gui.gui_refinement.ImageTk.PhotoImage", PhotoImage):
            app._present_image(rgb, preserve_view=True)
            photo, old_item = app.tk_image, app._canvas_image_id
            app.offset_x = 100.0
            app._present_image(rgb, preserve_view=True)
            self.assertIsNone(app._canvas_image_id)
            self.assertEqual(app.canvas.items, {})
            app.offset_x = 0.0
            app._present_image(rgb, preserve_view=True)
        self.assertIs(app.tk_image, photo)
        self.assertNotEqual(app._canvas_image_id, old_item)
        self.assertEqual(app.canvas.created, 2)
        self.assertEqual(app.canvas.deleted, ["all"])
        self.assertEqual(list(app.canvas.items), [app._canvas_image_id])
        self.assertEqual(app.canvas.items[app._canvas_image_id]["coords"], (0, 0))
        np.testing.assert_array_equal(photo.pixels, cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_LINEAR))
        np.testing.assert_array_equal(rgb, original)

    def test_stop_preserves_pending_decode_and_renders_it_once(self):
        reader, capture = self.make_reader()
        reader.read(10)
        app = HeadlessApp(reader=reader)
        app.playback_active = True
        app._playback_timing = (10.0, 0.0, 1.0)
        app.request_frame(15, immediate=True)
        generation = app._decode_gen
        app._schedule_playback(8)
        playback_job = app.playback_job
        app.stop_playback()
        self.assertEqual(app._decode_gen, generation)
        self.assertEqual(app._decode_request_queue.qsize(), 1)
        self.assertIn(playback_job, app.cancelled)
        self.assertIsNone(app.playback_job)
        result = self.decode_one_request(app)
        app._frame_queue.put(result)
        app._pump_frame_queue()
        app._pump_frame_queue()
        self.assertEqual([draw[0] for draw in app.draws], [15])
        self.assertEqual(app.current_frame, 15)
        self.assertEqual(app.requested_frame, 15)
        self.assertEqual(capture.position, 16)
        self.assertFalse(app.playback_active)
        self.assertEqual(reader._cap_pos, app.current_frame + 1)

    def test_stop_accepts_retargeted_inflight_frame_after_pause(self):
        app = HeadlessApp()
        app.playback_active = True
        app.request_frame(15, immediate=True)
        generation = app._decode_gen
        # The worker already retargeted this decode to frame 18 before Stop.
        result = (generation, app.reader, 18, False, np.zeros((1, 1, 3)), None)
        app.stop_playback()
        app._frame_queue.put(result)
        app._pump_frame_queue()
        self.assertEqual(app.current_frame, 18)
        self.assertEqual(app.requested_frame, 18)
        self.assertEqual([draw[0] for draw in app.draws], [18])
        self.assertEqual(app.reader.prefetches, [])

    def test_stop_completion_skips_prefetch_but_new_manual_navigation_prefetches(self):
        app = HeadlessApp()
        app.playback_active = True
        app.request_frame(15, immediate=True)
        paused_generation = app._decode_gen
        app.stop_playback()
        app._finish_frame_render(paused_generation, app.reader, 15, False, np.zeros((1, 1, 3)))
        self.assertEqual([draw[0] for draw in app.draws], [15])
        self.assertEqual(app.reader.prefetches, [])
        app.request_frame(3, immediate=True)
        self.assertGreater(app._decode_gen, paused_generation)
        # Calling Stop again while paused must keep manual navigation eligible.
        app.stop_playback()
        app._finish_frame_render(app._decode_gen, app.reader, 3, False, np.zeros((1, 1, 3)))
        self.assertEqual([draw[0] for draw in app.draws], [15, 3])
        self.assertEqual(app.reader.prefetches, [3])

    def test_rapid_restart_anchors_at_pending_request_without_invalidation(self):
        app = HeadlessApp()
        app.playback_active = True
        app.request_frame(17, immediate=True)
        generation = app._decode_gen
        app.stop_playback()
        with patch("gui.gui_refinement.time.perf_counter", return_value=8.0):
            app.start_playback()
        self.assertEqual(app._playback_timing, (17.0, 8.0, 1.0))
        self.assertEqual(app._decode_gen, generation)
        self.assertEqual(app._decode_request_queue.qsize(), 1)
        self.assertTrue(app.playback_active)
        self.assertEqual(app.reader.cancel_count, 1)
        self.assertIsNone(app._paused_decode_gen)

    def test_manual_navigation_discards_old_generation_and_keeps_newest_request(self):
        app = HeadlessApp()
        old_generation = app._decode_gen
        app._frame_queue.put((old_generation, app.reader, 15, False, np.zeros((1, 1, 3)), None))
        app.request_frame(100, immediate=True)
        app.request_frame(3, immediate=True)
        self.assertEqual(app._decode_request_queue.qsize(), 1)
        generation, reader, frame, reset = app._decode_request_queue.get_nowait()
        self.assertEqual(frame, 3)
        app._pump_frame_queue()
        self.assertEqual(app.draws, [])
        app._finish_frame_render(generation, reader, frame, reset, np.zeros((1, 1, 3)))
        self.assertEqual([draw[0] for draw in app.draws], [3])

    def test_pending_decode_is_not_replaced_by_repeated_playback_ticks(self):
        app = HeadlessApp()
        app.playback_active = True
        app._playback_timing = (10.0, 0.0, 1.0)
        app.request_frame(15, immediate=True)
        generation = app._decode_gen
        with patch("gui.gui_refinement.time.perf_counter", return_value=10.0):
            app._playback_tick()
            app._playback_tick()
        self.assertEqual(app._decode_gen, generation)
        self.assertEqual(app._decode_request_queue.qsize(), 1)
        self.assertEqual(app.requested_frame, 15)

    def test_past_slow_frame_does_not_delay_available_playback_work(self):
        app = HeadlessApp()
        app._preview_pacer.observe_frame(0.20)
        app._preview_pacer.begin_frame(1.0)
        app.playback_active = True
        app._playback_timing = (10.0, 0.0, 1.0)
        with patch("gui.gui_refinement.time.perf_counter", return_value=1.01):
            app._playback_tick()
        self.assertGreater(app.requested_frame, app.current_frame)
        self.assertEqual(app._decode_request_queue.qsize(), 1)
        self.assertLessEqual(app.jobs[app.playback_job][0], 8)

    def test_completed_frame_wakes_playback_immediately(self):
        app = HeadlessApp()
        app.playback_active = True
        app.requested_frame = 15
        app._schedule_playback(8)
        old_job = app.playback_job
        app._finish_frame_render(0, app.reader, 15, False, np.zeros((1, 1, 3)))
        self.assertEqual(app.current_frame, 15)
        self.assertIn(old_job, app.cancelled)
        self.assertEqual(app.jobs[app.playback_job][0], 1)
        self.assertEqual(app.reader.prefetches, [])

    def test_cached_playback_frame_keeps_immediate_next_tick(self):
        reader, _ = self.make_reader()
        reader.read(15)
        app = HeadlessApp(reader=reader)
        app.playback_active = True
        app._playback_timing = (15.0, 1.0, 1.0)
        with patch("gui.gui_refinement.time.perf_counter", return_value=1.0):
            app._playback_tick()
        self.assertEqual([draw[0] for draw in app.draws], [15])
        self.assertEqual(app._decode_request_queue.qsize(), 0)
        self.assertEqual(app.jobs[app.playback_job][0], 1)

    def test_result_queue_polls_fast_during_playback_and_slows_while_idle(self):
        app = HeadlessApp()
        app.playback_active = True
        app._pump_frame_queue()
        self.assertEqual(app.jobs[str(app.next_job)][0], 1)
        app.stop_playback()
        app._pump_frame_queue()
        self.assertEqual(app.jobs[str(app.next_job)][0], 8)

    def test_speed_change_preserves_position_and_clock_is_monotonic(self):
        app = HeadlessApp()
        app.playback_active = True
        app._playback_timing = (10.0, 1.0, 1.0)
        with patch("gui.gui_refinement.time.perf_counter", return_value=2.0):
            app.set_playback_speed(2.0)
        self.assertEqual(app._playback_timing, (40.0, 2.0, 2.0))
        self.assertEqual(app._playback_frame_at(2.5, app.reader, app._playback_timing), 70)
        self.assertEqual(app._playback_frame_at(1.0, app.reader, app._playback_timing), 40)
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            app.set_playback_speed(speed)
            self.assertEqual(app.playback_speed, speed)
        for invalid in (0.0, -1.0, 3.0, float("nan")):
            with self.assertRaises(ValueError):
                app.set_playback_speed(invalid)

    def test_end_of_video_finishes_pending_last_frame_then_stops(self):
        app = HeadlessApp(current=18, frame_count=20)
        app.playback_active = True
        app._playback_timing = (18.0, 0.0, 1.0)
        with patch("gui.gui_refinement.time.perf_counter", return_value=100.0):
            app._playback_tick()
        self.assertEqual(app.requested_frame, 19)
        app._finish_frame_render(app._decode_gen, app.reader, 19, False, np.zeros((1, 1, 3)))
        app._playback_tick()
        self.assertEqual(app.current_frame, 19)
        self.assertFalse(app.playback_active)
        self.assertIsNone(app._playback_timing)
        app.start_playback()
        self.assertFalse(app.playback_active)


if __name__ == "__main__":
    unittest.main()
