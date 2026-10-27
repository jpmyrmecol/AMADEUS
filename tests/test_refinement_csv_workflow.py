"""Exercise refined CSV loading and candidate navigation without a Tk window."""

import json
import queue
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gui import gui_refinement as refinement
from gui.refinement_identity import infer_refinement_identity


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class CandidateTree:
    def __init__(self):
        self.rows = {}
        self.selected = ()

    def get_children(self):
        return tuple(self.rows)

    def delete(self, *rows):
        for row in rows:
            self.rows.pop(row)
        self.selected = ()

    def insert(self, _parent, _position, *, iid, values):
        self.rows[iid] = values

    def selection(self):
        return self.selected

    def selection_set(self, iid):
        self.selected = (iid,)

    def see(self, _iid):
        pass


def make_wide(frames=8, ids=3):
    frame = np.arange(frames, dtype=float)
    columns = {"frame": frame}
    for tid in range(ids):
        columns[f"x{2 * tid}"] = 20 + tid * 100 + frame
        columns[f"y{2 * tid}"] = 40 + tid * 100 + frame * 2
        columns[f"x{2 * tid + 1}"] = 10 + tid * 100 + frame
        columns[f"y{2 * tid + 1}"] = 30 + tid * 100 + frame * 2
    return pd.DataFrame(columns)


def result_from_wide(wide):
    columns = {"frame": wide["frame"].to_numpy(copy=True)}
    for tid in range((len(wide.columns) - 1) // 4):
        front = wide[[f"x{2 * tid}", f"y{2 * tid}"]].to_numpy()
        rear = wide[[f"x{2 * tid + 1}", f"y{2 * tid + 1}"]].to_numpy()
        center = (front + rear) / 2
        delta = front - rear
        columns.update({
            f"cx{tid}": center[:, 0], f"cy{tid}": center[:, 1],
            f"w{tid}": np.linalg.norm(delta, axis=1),
            f"h{tid}": np.full(len(wide), 5.0),
            f"heading{tid}": np.degrees(np.arctan2(delta[:, 0], -delta[:, 1])) % 360,
        })
    return pd.DataFrame(columns)


def candidate(frame, tid, code=10, short="DIR-FLIP"):
    return {
        "frame": frame, "tid": tid, "source_tid": tid, "assign_code": code,
        "label_short": short, "label": short, "_severity": 0.0,
    }


def install_wide(app, wide):
    app.df = wide.copy(deep=True)
    app.df_np = wide.to_numpy(copy=True)
    app.refinement_pose_np = app.df_np.copy()
    app.frame_count = len(wide)
    app.num_ids = (len(wide.columns) - 1) // 4
    tids = np.arange(app.num_ids)
    app._front_x_idx, app._front_y_idx = 1 + 4 * tids, 2 + 4 * tids
    app._rear_x_idx, app._rear_y_idx = 3 + 4 * tids, 4 + 4 * tids


@pytest.fixture
def app(tmp_path):
    app = object.__new__(refinement.UmaDirectionRefinementApp)
    source = tmp_path / "source_rear_front.csv"
    wide = make_wide()
    refinement._convert.save_rear_front_csv(
        refinement._convert.legacy_wide_to_rear_front(wide, "front_rear"), source,
    )
    install_wide(app, wide)
    app._dataset_reference_wide = wide.copy(deep=True)
    app._dataset_reference_result = result_from_wide(wide)
    app.current_dataset = {
        "converted_csv_path": str(source),
        "obb_path": str(tmp_path / "original_obb.csv"),
        "direction_path": str(tmp_path / "original_direction.csv"),
    }
    app.csv_path_var = Value(str(source))
    app.refinement_csv_path_var = Value(str(source))
    app.refinement_csv_paths = [str(source)]
    app.refinement_csv_combo = Mock()
    app.refinement_source_path = str(source)
    app.refinement_source_paths = {str(source)}
    app.loaded_csv_path = str(source)
    app._refined_source_overwrite_path = None
    app._pending_refinement_source = None
    app.refinement_output_kind = "rear_front"
    app.refinement_obb_df = None
    app.current_id = 0
    app.current_frame = 0
    app.id_var = Value(0)
    app.selected_tids = []
    app.reverse_direction_var = Value(False)
    app.edited_keys = set()
    app.flip_only_edited_keys = set()
    app.position_edited_keys = set()
    app.direction_reversed_keys = set()
    app._source_candidates = []
    app.suspects = []
    app._candidate_id_map = None
    app._candidate_runs = []
    app.current_suspect_index = None
    app.candidate_labels_by_key = {}
    app.assign_type_history = {}
    app.provenance_lookup = {}
    app.candidate_tree = CandidateTree()
    app.mode_var = Value("amadeus")
    app.show_only_unedited_var = Value(False)
    app.enabled_assign_codes = {10.0, 11.0}
    app._type_panel_open = False
    app._type_filter_label_vars = {10.0: Value(""), 11.0: Value("")}
    app.tree_event_suppressed = False
    app.pending_candidate_select_job = None
    app._companion_save_active = False
    app._companion_save_messages = queue.Queue()
    app.save_count = 0
    app.last_save_path = None
    app.set_status = Mock()
    app.update_info_panel = Mock()
    app.redraw_current_frame = Mock()
    app.request_frame = Mock()
    app.focus_set = Mock()
    app._cancel_pending_candidate_select = Mock()
    app._suppress_candidate_tree_select_until_idle = Mock()
    return app


def test_saved_csv_path_is_selected_and_restored_with_dataset_history(app, tmp_path):
    output = tmp_path / "corrected_rear_front.csv"
    app.df.at[2, "x0"] += 3
    app.df_np[2, 1] += 3
    assert app._save_refinement_csv(str(output), save_as=True)
    saved_paths, error = app._companion_save_messages.get(timeout=5)
    assert error is None
    assert all(path.is_file() for path in saved_paths)
    assert app.refinement_csv_path_var.get() == str(output)
    assert app.refinement_csv_paths[0] == str(output)
    assert app.last_save_path == str(output)

    source = app.current_dataset["converted_csv_path"]
    app.refinement_csv_paths = []
    app._restore_refinement_csv_paths(source)

    assert app.refinement_csv_paths == [source, str(output)]
    assert app.refinement_csv_path_var.get() == source
    stored = json.loads(app._refinement_csv_history_path().read_text())
    assert stored["csv_paths"] == [str(output), source]
    wide, result, mapping = app._prepare_refinement_csv(
        str(output), app._dataset_reference_wide, app._dataset_reference_result,
    )
    assert result is None
    np.testing.assert_array_equal(wide.to_numpy(), app.df_np)
    np.testing.assert_array_equal(mapping.source_to_loaded, np.tile(np.arange(3), (8, 1)))


def test_path_history_deduplicates_repeated_save_destinations(app, tmp_path):
    first, second = str(tmp_path / "first.csv"), str(tmp_path / "second.csv")
    app._remember_refinement_csv(first)
    app._remember_refinement_csv(second)
    app._remember_refinement_csv(first)

    assert app.refinement_csv_paths == [first, second, app.current_dataset["converted_csv_path"]]
    assert app.refinement_csv_path_var.get() == first


def test_prepare_csv_detects_a_mid_video_swap_without_moving_source_candidates(app, tmp_path):
    loaded = app.df.copy(deep=True)
    loaded.iloc[3:, 1:5] = app.df.iloc[3:, 5:9].to_numpy()
    loaded.iloc[3:, 5:9] = app.df.iloc[3:, 1:5].to_numpy()
    loaded.at[4, "x2"] += 5  # One endpoint is manually corrected after the ID swap.
    path = tmp_path / "switched_rear_front.csv"
    refinement._convert.save_rear_front_csv(
        refinement._convert.legacy_wide_to_rear_front(loaded, "front_rear"), path,
    )

    wide, result, mapping = app._prepare_refinement_csv(str(path), app.df, app._dataset_reference_result)

    assert result is None
    np.testing.assert_array_equal(wide.to_numpy(), loaded.to_numpy())
    np.testing.assert_array_equal(mapping.source_to_loaded[:3], np.tile([0, 1, 2], (3, 1)))
    np.testing.assert_array_equal(mapping.source_to_loaded[3:], np.tile([1, 0, 2], (5, 1)))
    assert {(entry.start_frame, entry.end_frame) for entry in mapping.switches} == {(3, 3)}


@pytest.mark.parametrize("mismatch", ["frames", "ids"])
def test_prepare_csv_rejects_incompatible_frame_or_id_count(app, tmp_path, mismatch):
    wrong = make_wide(frames=7 if mismatch == "frames" else 8, ids=2 if mismatch == "ids" else 3)
    path = tmp_path / "wrong_rear_front.csv"
    refinement._convert.save_rear_front_csv(
        refinement._convert.legacy_wide_to_rear_front(wrong, "front_rear"), path,
    )

    with pytest.raises(ValueError, match="same frame count and ID count"):
        app._prepare_refinement_csv(str(path), app.df, app._dataset_reference_result)


def test_prepare_amadeus_result_uses_original_unrounded_geometry(app, tmp_path):
    frames = np.arange(8)
    columns = {"frame": frames}
    for tid in range(3):
        columns.update({
            f"cx{tid}": 50.123456 + frames + tid * 100,
            f"cy{tid}": 90.654321 + frames + tid * 100,
            f"w{tid}": np.full(8, 31.123456), f"h{tid}": np.full(8, 5.654321),
            f"heading{tid}": 17.876543 + frames,
        })
    original_result = pd.DataFrame(columns)
    source_wide = refinement._convert.rear_front_to_legacy_wide(
        refinement._convert.amadeus_result_to_rear_front(original_result), index_col="frame",
    )
    saved = refinement._convert.swap_amadeus_result_ids(original_result, 0, 1, 3, 8)
    path = tmp_path / "result_refined.csv"
    refinement._convert.save_amadeus_result_csv(saved, path)

    wide, result, mapping = app._prepare_refinement_csv(str(path), source_wide, original_result)

    assert result is not None
    assert len(wide) == 8
    np.testing.assert_array_equal(mapping.source_to_loaded[3:], np.tile([1, 0, 2], (5, 1)))


def test_finish_load_preserves_dataset_and_remaps_candidates_by_frame(app, tmp_path):
    sources = [candidate(frame, tid) for frame in (1, 4) for tid in (0, 1)]
    app._source_candidates = deepcopy(sources)
    loaded = app.df.copy(deep=True)
    loaded.iloc[3:, 1:5] = app.df.iloc[3:, 5:9].to_numpy()
    loaded.iloc[3:, 5:9] = app.df.iloc[3:, 1:5].to_numpy()
    mapping = infer_refinement_identity(app.df, loaded)
    dataset = app.current_dataset
    path = str(tmp_path / "reloaded.csv")

    def load_files(*, preserve_view, prepared_df):
        assert preserve_view
        install_wide(app, prepared_df)
        app._activate_refinement_source(app._pending_refinement_source, app.csv_path_var.get())
        app.last_save_path = app._refined_source_overwrite_path
        return True

    app.load_files = Mock(side_effect=load_files)
    assert app._finish_refinement_csv_load(path, loaded, None, mapping)

    assert app.current_dataset is dataset
    assert app._source_candidates == sources
    assert [(item["frame"], item["tid"]) for item in app.suspects] == [(1, 0), (1, 1), (4, 1), (4, 0)]
    assert app.refinement_csv_path_var.get() == path
    assert app.last_save_path == path
    assert app._refined_source_overwrite_path == path
    assert "ID switch frames: 3" in app.set_status.call_args.args[0]


def test_failed_finish_load_keeps_previous_csv_and_candidate_ids(app, tmp_path):
    app._source_candidates = [candidate(4, 0)]
    app._apply_candidate_id_map(None)
    before = deepcopy(app.suspects)
    old_path = app.csv_path_var.get()
    app.load_files = Mock(return_value=False)
    mapping = infer_refinement_identity(app.df, app.df)

    assert not app._finish_refinement_csv_load(str(tmp_path / "failed.csv"), app.df, None, mapping)

    assert app.csv_path_var.get() == old_path
    assert app.refinement_csv_path_var.get() == old_path
    assert app.suspects == before
    assert app._pending_refinement_source is None


def test_mapping_is_reapplied_from_source_and_original_ids_can_be_restored(app):
    sources = [candidate(1, 0), candidate(4, 0), candidate(4, 1)]
    app._source_candidates = deepcopy(sources)
    mapping = np.tile(np.arange(3), (8, 1))
    mapping[3:, [0, 1]] = mapping[3:, [1, 0]]
    app._apply_candidate_id_map(mapping)
    app._apply_candidate_id_map(mapping)

    assert [item["tid"] for item in app.suspects] == [0, 1, 0]
    assert app._source_candidates == sources
    app._apply_candidate_id_map(None)
    assert [item["tid"] for item in app.suspects] == [0, 0, 1]


def test_unknown_candidate_id_is_shown_without_selecting_a_real_id(app):
    app._source_candidates = [candidate(2, 0)]
    mapping = np.tile(np.arange(3), (8, 1))
    mapping[2, 0] = -1
    app._apply_candidate_id_map(mapping)
    app.refresh_candidate_tree()

    assert app.candidate_tree.rows["0"][1].startswith("?")
    app.selected_tids = [1]
    app._set_id = Mock()
    app.select_candidate(0)

    app._set_id.assert_not_called()
    assert app.selected_tids == []
    app.request_frame.assert_called_once_with(2, reset_view=False, immediate=False)


def test_candidate_history_follows_loaded_identity_at_each_frame(app):
    app.assign_type_history = {0: {1: [("fwd", 10)], 4: [("fwd", 11)]}, 1: {4: [("bwd", 12)]}}
    app._source_candidates = [candidate(1, 0), candidate(4, 0), candidate(4, 1)]
    mapping = np.tile(np.arange(3), (8, 1))
    mapping[3:, [0, 1]] = mapping[3:, [1, 0]]
    app._apply_candidate_id_map(mapping)

    assert app._assign_history_entries(0, 1) == [("fwd", 10)]
    assert app._assign_history_entries(1, 4) == [("fwd", 11)]
    assert app._assign_history_entries(0, 4) == [("bwd", 12)]


def test_candidate_display_navigation_and_counts_use_run_starts(app):
    app._source_candidates = [candidate(frame, 0) for frame in (1, 2, 3)]
    app._source_candidates += [candidate(frame, 1) for frame in (5, 6)]
    app._apply_candidate_id_map(None)
    app.refresh_candidate_tree()
    app._update_type_filter_counts()

    assert list(app.candidate_tree.rows) == ["0", "3"]
    assert [row[0] for row in app.candidate_tree.rows.values()] == [1, 5]
    assert app._type_filter_label_vars[10].get().endswith("(2)")
    app._step_candidate(1)
    assert app.current_suspect_index == 0
    app._step_candidate(1)
    assert app.current_suspect_index == 3
    app._step_candidate(-1)
    assert app.current_suspect_index == 0


def test_hide_edited_filters_runs_without_exposing_middle_frames(app):
    app._source_candidates = [candidate(frame, 0) for frame in (1, 2, 3)]
    app._source_candidates += [candidate(5, 1)]
    app._apply_candidate_id_map(None)
    app.show_only_unedited_var.set(True)
    app.edited_keys = {(1, 0)}
    app.refresh_candidate_tree()

    assert list(app.candidate_tree.rows) == ["0", "3"]
    assert app.candidate_tree.rows["0"][0] == 1
    assert app.candidate_tree.rows["0"][3] == "partial"
    app.edited_keys.update({(2, 0), (3, 0)})
    app.refresh_candidate_tree()
    assert list(app.candidate_tree.rows) == ["3"]


def test_in_session_swap_updates_candidate_mapping_from_the_current_frame(app):
    app._source_candidates = [candidate(frame, tid) for frame in (1, 4, 6) for tid in (0, 1, 2)]
    app._apply_candidate_id_map(None)
    app.current_frame = 3
    original = app.df.copy(deep=True)

    assert app._swap_ids_from_current_frame(0, 1) == 5

    assert [(item["frame"], item["tid"]) for item in app.suspects] == [
        (1, 0), (1, 1), (1, 2), (4, 1), (4, 0), (4, 2), (6, 1), (6, 0), (6, 2),
    ]
    np.testing.assert_array_equal(app.df.iloc[3:, 1:5], original.iloc[3:, 5:9])
    np.testing.assert_array_equal(app.df.to_numpy(), app.df_np)
    app.current_frame = 5
    app._swap_ids_from_current_frame(0, 2)
    assert [item["tid"] for item in app.suspects[-3:]] == [1, 2, 0]
    assert [item["tid"] for item in app._source_candidates[-3:]] == [0, 1, 2]


def test_selecting_original_path_reloads_dataset_without_refined_input_conversion(app):
    app.load_dataset_record = Mock()
    app.load_refinement_csv(app.current_dataset["converted_csv_path"])

    app.load_dataset_record.assert_called_once_with(app.current_dataset, preserve_view=True)


def test_saving_loaded_result_cannot_overwrite_original_dataset_companion(app, tmp_path, monkeypatch):
    original_path = tmp_path / "source_rear_front.csv"
    original_bytes = original_path.read_bytes()
    loaded_path = tmp_path / "source.csv"
    loaded_result = result_from_wide(app.df)
    loaded_result.at[2, "cx0"] += 20
    refinement._convert.save_amadeus_result_csv(loaded_result, loaded_path)
    app.refinement_obb_df = loaded_result
    app.refinement_output_kind = "amadeus_result"
    app.refinement_source_path = str(loaded_path)
    app._refined_source_overwrite_path = str(loaded_path)
    app.refinement_source_paths = {str(loaded_path), *app.current_dataset.values()}
    monkeypatch.setattr(refinement.messagebox, "showwarning", Mock())

    saved = app._save_refinement_csv(str(loaded_path))
    if saved:
        app._companion_save_messages.get(timeout=5)

    assert not saved
    assert original_path.read_bytes() == original_bytes


@pytest.mark.parametrize("delta", [1, -1])
def test_navigation_from_hidden_run_selects_the_immediately_adjacent_visible_run(app, delta):
    app._source_candidates = [candidate(frame, 0) for frame in (1, 2, 4, 5, 7)]
    app._apply_candidate_id_map(None)
    app.show_only_unedited_var.set(True)
    app.current_suspect_index = 0 if delta > 0 else 4
    app.edited_keys = {(1, 0), (2, 0)} if delta > 0 else {(7, 0)}
    app.refresh_candidate_tree()

    app._step_candidate(delta)

    assert app.current_suspect_index == 2
    app.request_frame.assert_called_once_with(4, reset_view=False, immediate=False)


def test_load_files_activates_prepared_data_and_refined_save_destination(app, tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.touch()
    csv = tmp_path / "prepared_rear_front.csv"
    prepared = app.df.copy(deep=True)
    prepared.at[3, "x0"] += 7
    refinement._convert.save_rear_front_csv(
        refinement._convert.legacy_wide_to_rear_front(prepared, "front_rear"), csv,
    )
    app.video_path_var = Value(str(video))
    app.csv_path_var.set(str(csv))
    app.current_frame = 4
    old_reader = SimpleNamespace(close=Mock())
    new_reader = SimpleNamespace(frame_count=20, fps=30, video_path=str(video), close=Mock())
    app.reader = old_reader
    app.prefetch_after_id = app.frame_render_job = None
    app._preview_pacer = Mock()
    app.canvas = Mock()
    app._invalidate_decode_requests = Mock()
    app._stop_key_play = Mock()
    app.stop_playback = Mock()
    app.on_canvas_release = Mock()
    app._update_play_button = Mock()
    app._close_swap_popup = Mock()
    app._reset_view_state = Mock()
    app._configure_controls = Mock()
    app.after_idle = Mock()
    app._set_frame = Mock(side_effect=lambda frame, **_kwargs: setattr(app, "current_frame", frame))
    app._pending_refinement_source = {
        "output_kind": "rear_front", "result_df": None,
        "source_path": str(csv), "source_paths": [str(csv)],
        "allow_refined_overwrite": True,
    }
    monkeypatch.setattr(refinement, "VideoFrameReader", Mock(return_value=new_reader))
    monkeypatch.setattr(refinement, "_start_maximized", Mock())

    assert app.load_files(preserve_view=True, prepared_df=prepared)

    old_reader.close.assert_called_once()
    assert app.reader is new_reader
    assert app.refinement_source_path == str(csv)
    assert app.loaded_csv_path == str(csv)
    assert app.last_save_path == str(csv)
    assert app._pending_refinement_source is None
    assert app.current_frame == 4
    assert app.frame_count == 8
    assert app.num_ids == 3
    np.testing.assert_array_equal(app.df.to_numpy(), prepared.to_numpy())
    np.testing.assert_array_equal(app.df_np, prepared.to_numpy())
    np.testing.assert_array_equal(app.refinement_pose_np, prepared.to_numpy())
    app._set_frame.assert_called_once_with(4, reset_view=False)
