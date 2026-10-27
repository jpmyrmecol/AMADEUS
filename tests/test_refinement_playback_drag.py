"""Exercise refinement playback edits without opening a Tk window or video."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gui import gui_refinement as refinement


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


@pytest.fixture
def app():
    """Use the real controller and data writes, replacing only GUI and video IO."""
    app = object.__new__(refinement.UmaDirectionRefinementApp)
    frames = np.arange(24, dtype=float)
    app.df = pd.DataFrame({
        "frame": frames,
        "x0": 10 + frames,
        "y0": 20 + 2 * frames,
        "x1": 30 + frames,
        "y1": 40 + 2 * frames,
        "x2": 110 + frames,
        "y2": 120 + 2 * frames,
        "x3": 130 + frames,
        "y3": 140 + 2 * frames,
    })
    app.df_np = app.df.to_numpy(copy=True)
    app.refinement_pose_np = app.df_np.copy()
    app._front_x_idx = np.array([1, 5])
    app._front_y_idx = np.array([2, 6])
    app._rear_x_idx = np.array([3, 7])
    app._rear_y_idx = np.array([4, 8])
    app.reverse_direction_var = Value(False)
    app.frame_count = len(app.df)
    app.num_ids = 2
    app.current_id = 0
    app.current_frame = app.requested_frame = app._last_displayed_frame = 2
    app.frame_var = Value(2)
    app.frame_scale = Value(2)
    app.id_var = Value(0)
    app.frame_cache_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    app.edited_keys = set()
    app.position_edited_keys = set()
    app.flip_only_edited_keys = set()
    app.direction_reversed_keys = set()
    app._decode_gen = 0
    app._paused_decode_gen = None
    app._paused_decode_direction = 1
    app._playback_drag_gen = None
    app._pending_playback_drags = []
    app._decode_request_queue = queue.Queue(maxsize=1)
    app._frame_queue = queue.Queue()
    app._decode_stop = threading.Event()
    app.frame_render_job = None
    app.frame_render_interval_ms = 16
    app.playback_job = None
    app.playback_active = True
    app.playback_speed = 1.0
    app._playback_timing = (2.0, 0.0, 1.0)
    app._reverse_waiting_for_cache = False
    app._arrow_key_play = SimpleNamespace(direction=0)
    app._key_play_delta = 0
    app.play_button = app.speed_button = None
    app.scale = 1.0
    app.offset_x = app.offset_y = 0.0
    app.drag_mode = app.drag_tid = app.drag_last_canvas = None
    app.drag_start_front = app.drag_start_rear = app.drag_anchor_img = None
    app.left_button_down = False
    app._pan_start_canvas = app._pan_start_offset = None
    app.hover_tid = None
    app.canvas = Mock()
    app.focus_set = Mock()
    app._stop_key_play = Mock()
    app.update_info_panel = Mock()
    app.redraw_current_frame = Mock()
    app._schedule_refresh_candidate_tree = Mock()
    app._draw_canvas = Mock()
    app._schedule_playback = Mock()
    app._queue_prefetch = Mock()
    app.after_cancel = Mock()
    app.reader = SimpleNamespace(
        fps=10.0,
        get_cached=Mock(return_value=app.frame_cache_rgb),
        read=Mock(return_value=app.frame_cache_rgb),
        _store_cached=Mock(),
        cancel_prefetch=Mock(),
        prepare_forward_position=Mock(),
        reverse_buffer=SimpleNamespace(
            error=None,
            start=Mock(),
            stop=Mock(),
            ready=Mock(return_value=True),
            get_frame=Mock(),
            advance=Mock(),
        ),
    )
    return app


def hold_arrow(app, mode="whole", tid=0, pointer=(50.25, 60.5)):
    app.left_button_down = True
    app.drag_mode = mode
    app.drag_tid = tid
    state = app._get_arrow_state(app.current_frame, tid)
    app.drag_start_front = state.front.copy()
    app.drag_start_rear = state.rear.copy()
    app.drag_anchor_img = (state.front + state.rear) / 2
    app.drag_last_canvas = np.array(pointer, dtype=float)


def finish_queued(app, actual_frame=None):
    generation, reader, target, reset_view, prepared = app._decode_request_queue.get_nowait()
    if actual_frame is not None:
        target = actual_frame
    pixels = prepared.to_rgb() if prepared is not None else app.frame_cache_rgb
    app._finish_frame_render(generation, reader, target, reset_view, pixels)


def assert_edits(app, frames, tid=0):
    expected = {(frame, tid) for frame in frames}
    assert app.edited_keys == expected
    assert app.position_edited_keys == expected
    np.testing.assert_array_equal(app.df.to_numpy(), app.df_np)


def test_cached_playback_records_all_skipped_frames(app):
    hold_arrow(app)
    app._set_frame(6, playback=True)

    assert_edits(app, range(2, 7))
    np.testing.assert_allclose(
        app.df.loc[2:6, ["x0", "y0", "x1", "y1"]],
        np.tile([40.25, 50.5, 60.25, 70.5], (5, 1)),
    )
    np.testing.assert_array_equal(app.df_np[:, 5:], app.refinement_pose_np[:, 5:])
    assert app.current_frame == 6
    assert app._schedule_refresh_candidate_tree.call_count == 1


def test_decoder_catchup_records_beyond_requested_frame(app, monkeypatch):
    hold_arrow(app)
    app.reader.get_cached.return_value = None
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.0)
    app._set_frame(4, playback=True)
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.65)

    def read_frame(frame, **_kwargs):
        assert frame == 8
        app._decode_stop.set()
        return app.frame_cache_rgb

    app.reader.read.side_effect = read_frame
    app._decode_worker_loop()
    generation, reader, target, reset_view, pixels, error = app._frame_queue.get_nowait()
    assert error is None
    app._finish_frame_render(generation, reader, target, reset_view, pixels)

    assert_edits(app, range(2, 9))
    assert app.current_frame == 8


@pytest.mark.parametrize("mode", ["front", "rear"])
@pytest.mark.parametrize("reverse_display", [False, True])
def test_endpoint_drag_preserves_each_frames_other_endpoint(app, mode, reverse_display):
    app.reverse_direction_var.set(reverse_display)
    hold_arrow(app, mode)
    app._set_frame(6, playback=True)

    assert_edits(app, range(2, 7))
    moved_columns = [1, 2] if (mode == "front") != reverse_display else [3, 4]
    other_columns = [3, 4] if moved_columns == [1, 2] else [1, 2]
    np.testing.assert_allclose(
        app.df_np[2:7, moved_columns], np.tile([50.25, 60.5], (5, 1)),
    )
    np.testing.assert_array_equal(
        app.df_np[:, other_columns], app.refinement_pose_np[:, other_columns],
    )


def test_reverse_playback_records_prepared_frame_and_gap(app, monkeypatch):
    app.current_frame = app.requested_frame = app._last_displayed_frame = 10
    app._playback_timing = (10.0, 0.0, -1.0)
    hold_arrow(app)
    app.reader.reverse_buffer.get_frame.return_value = (
        7, refinement.PreparedFrame(pixels=app.frame_cache_rgb),
    )
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.35)

    app._playback_tick()
    finish_queued(app)

    assert_edits(app, range(7, 11))
    assert app.current_frame == 7
    app.reader.reverse_buffer.advance.assert_called_once_with(7)


@pytest.mark.parametrize("action", ["release", "stop"])
def test_pending_decode_does_not_extend_edit_past_release_or_stop(app, monkeypatch, action):
    hold_arrow(app)
    app.reader.get_cached.return_value = None
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.0)
    app._set_frame(4, playback=True)

    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.35)
    if action == "release":
        app.on_canvas_release()
    else:
        app.stop_playback(update_button=False)
    finish_queued(app, actual_frame=8)

    assert_edits(app, range(2, 6))
    np.testing.assert_array_equal(app.df_np[6:], app.refinement_pose_np[6:])


def test_regrab_another_id_while_decode_pending_keeps_edit_intervals_separate(app, monkeypatch):
    hold_arrow(app)
    app.reader.get_cached.return_value = None
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.0)
    app._set_frame(4, playback=True)
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.25)
    app.on_canvas_release()

    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.45)
    app._pick_drag_target = Mock(return_value=(1, "front"))
    app.on_canvas_press(SimpleNamespace(x=180, y=190))
    finish_queued(app, actual_frame=8)

    expected = {(frame, 0) for frame in range(2, 5)} | {(frame, 1) for frame in range(6, 9)}
    assert app.edited_keys == expected
    assert app.position_edited_keys == expected
    np.testing.assert_array_equal(app.df.to_numpy(), app.df_np)
    np.testing.assert_array_equal(app.df_np[5], app.refinement_pose_np[5])


@pytest.mark.parametrize("invalid", ["generation", "reader"])
def test_stale_decode_cannot_write_refinement_data(app, invalid):
    hold_arrow(app)
    app.reader.get_cached.return_value = None
    app._set_frame(4, playback=True)
    generation, reader, target, reset_view, _prepared = app._decode_request_queue.get_nowait()
    if invalid == "generation":
        generation -= 1
    else:
        reader = object()

    app._finish_frame_render(generation, reader, target, reset_view, app.frame_cache_rgb)

    assert not app.edited_keys
    assert not app.position_edited_keys
    np.testing.assert_array_equal(app.df_np, app.refinement_pose_np)


@pytest.mark.parametrize("mode", [None, "pan"])
def test_playback_without_an_arrow_drag_preserves_data(app, mode):
    app.drag_mode = mode
    app.left_button_down = mode == "pan"
    app._set_frame(6, playback=True)

    assert not app.edited_keys
    np.testing.assert_array_equal(app.df_np, app.refinement_pose_np)


def test_manual_seek_during_playback_does_not_fill_intermediate_frames(app):
    hold_arrow(app)
    app._set_frame(8)

    assert_edits(app, [2, 8])
    np.testing.assert_array_equal(app.df_np[3:8], app.refinement_pose_np[3:8])


@pytest.mark.parametrize("mode", ["whole", "front", "rear"])
def test_single_frame_key_step_still_records_held_position(app, mode):
    app.playback_active = False
    app._playback_timing = None
    hold_arrow(app, mode)
    app._key_play_delta = 1
    app._key_play_step()

    assert_edits(app, [2, 3])
    assert app.current_frame == 3


def test_pointer_motion_after_skipped_frames_does_not_accumulate_displacement(app):
    hold_arrow(app)
    app._set_frame(6, playback=True)
    app.on_canvas_drag(SimpleNamespace(x=55.25, y=62.5))
    app._set_frame(9, playback=True)

    np.testing.assert_allclose(
        app.df.loc[7:9, ["x0", "y0", "x1", "y1"]],
        np.tile([45.25, 52.5, 65.25, 72.5], (3, 1)),
    )
    np.testing.assert_array_equal(app.df.to_numpy(), app.df_np)


@pytest.mark.parametrize("mode", ["front", "rear"])
def test_endpoint_drag_after_stop_keeps_the_new_frames_other_endpoint(app, monkeypatch, mode):
    hold_arrow(app, mode)
    app.reader.get_cached.return_value = None
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.0)
    app._set_frame(4, playback=True)
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.35)
    app.stop_playback(update_button=False)
    finish_queued(app, actual_frame=8)

    app.on_canvas_drag(SimpleNamespace(x=70.25, y=80.5))

    other_columns = [3, 4] if mode == "front" else [1, 2]
    np.testing.assert_array_equal(
        app.df_np[8, other_columns], app.refinement_pose_np[8, other_columns],
    )


def test_space_starts_fast_playback_and_records_skipped_frames(app, monkeypatch):
    app.playback_active = False
    app._playback_timing = None
    app.playback_speed = 4.0
    app.focus_get = Mock(return_value=None)
    hold_arrow(app)
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.0)

    assert app.on_space_press(SimpleNamespace()) == "break"
    assert app.playback_active
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.175)
    app._playback_tick()

    assert_edits(app, range(2, 10))
    assert app.current_frame == 9


def test_video_only_playback_requires_no_refinement_data(app):
    app.df = app.df_np = app.refinement_pose_np = None
    app._set_frame(6, playback=True)

    assert app.current_frame == 6
    assert not app.edited_keys


def test_saved_csv_preserves_skipped_frame_edits(app, tmp_path):
    hold_arrow(app)
    app._set_frame(6, playback=True)
    expected = app.df_np.copy()
    app._companion_save_active = False
    app._companion_save_messages = queue.Queue()
    app._refined_source_overwrite_path = None
    app.refinement_source_paths = set()
    app.refinement_obb_df = None
    app.refinement_output_kind = "rear_front"
    app.save_count = 0
    app.set_status = Mock()
    output = tmp_path / "refined_rear_front.csv"

    assert app._save_refinement_csv(str(output), save_as=True)
    saved_paths, error = app._companion_save_messages.get(timeout=5)

    assert error is None
    assert all(path.is_file() for path in saved_paths)
    saved = refinement._convert.load_rear_front_csv(output)
    restored = refinement._convert.rear_front_to_legacy_wide(saved, index_col="frame")
    np.testing.assert_array_equal(restored.to_numpy(), expected)
    assert not app.edited_keys


def test_grab_ahead_of_slow_decode_starts_at_its_original_frame(app, monkeypatch):
    app.reader.get_cached.return_value = None
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.0)
    app._set_frame(4, playback=True)
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.85)
    app._pick_drag_target = Mock(return_value=(0, "front"))
    app.on_canvas_press(SimpleNamespace(x=50.25, y=60.5))
    finish_queued(app, actual_frame=5)
    assert not app.edited_keys

    app._set_frame(12, playback=True)
    finish_queued(app)

    assert_edits(app, range(10, 13))
    np.testing.assert_array_equal(app.df_np[:10], app.refinement_pose_np[:10])


def test_release_interval_survives_a_decode_that_has_not_reached_its_end(app, monkeypatch):
    hold_arrow(app)
    app.reader.get_cached.return_value = None
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.0)
    app._set_frame(4, playback=True)
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.65)
    app.on_canvas_release()
    finish_queued(app, actual_frame=5)
    assert_edits(app, range(2, 6))

    app._set_frame(12, playback=True)
    finish_queued(app)

    assert_edits(app, range(2, 9))
    np.testing.assert_array_equal(app.df_np[9:], app.refinement_pose_np[9:])


@pytest.mark.parametrize("mode", ["whole", "front", "rear"])
def test_display_swap_after_release_does_not_change_pending_saved_coordinates(app, monkeypatch, mode):
    hold_arrow(app, mode)
    app.reader.get_cached.return_value = None
    app._set_frame(4, playback=True)
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.35)
    app.on_canvas_release()
    app.reverse_direction_var.set(True)
    finish_queued(app, actual_frame=5)

    expected = app.refinement_pose_np.copy()
    if mode == "whole":
        expected[2:6, 1:5] = [40.25, 50.5, 60.25, 70.5]
    elif mode == "front":
        expected[2:6, 1:3] = [50.25, 60.5]
    else:
        expected[2:6, 3:5] = [50.25, 60.5]
    np.testing.assert_array_equal(app.df.to_numpy(), expected)
    np.testing.assert_array_equal(app.df_np, expected)


def test_stop_then_restart_before_decode_finishes_keeps_recording_the_held_arrow(app, monkeypatch):
    hold_arrow(app)
    app.reader.get_cached.return_value = None
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.0)
    app._set_frame(4, playback=True)
    monkeypatch.setattr(refinement.time, "perf_counter", lambda: 0.35)
    app.stop_playback(update_button=False)
    app.start_playback()
    finish_queued(app, actual_frame=8)

    app._set_frame(10, playback=True)
    finish_queued(app)

    assert_edits(app, range(2, 11))
