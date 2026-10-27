"""Verify CSV identity correspondence with known edits and ambiguous boundaries."""

import numpy as np
import pandas as pd
import pytest

from gui.refinement_identity import IdentityInterval, IdentitySwitch, infer_refinement_identity


def trajectories(frames=12, ids=3):
    data = {"frame": np.arange(frames)}
    for tid in range(ids):
        data[f"x{2 * tid}"] = 100.0 * tid + np.arange(frames)
        data[f"y{2 * tid}"] = 20.0 + 7 * tid + 2 * np.arange(frames)
        data[f"x{2 * tid + 1}"] = data[f"x{2 * tid}"] + 10
        data[f"y{2 * tid + 1}"] = data[f"y{2 * tid}"] + 3
    return pd.DataFrame(data)


def swap(data, id_a, id_b, start=0, stop=None):
    stop = len(data) if stop is None else stop
    a = [f"x{2 * id_a}", f"y{2 * id_a}", f"x{2 * id_a + 1}", f"y{2 * id_a + 1}"]
    b = [f"x{2 * id_b}", f"y{2 * id_b}", f"x{2 * id_b + 1}", f"y{2 * id_b + 1}"]
    values = data.loc[start:stop - 1, a].to_numpy(copy=True)
    data.loc[start:stop - 1, a] = data.loc[start:stop - 1, b].to_numpy()
    data.loc[start:stop - 1, b] = values


def test_unchanged_csv_has_identity_mapping_without_switches():
    original = trajectories()
    result = infer_refinement_identity(original, original.copy())
    np.testing.assert_array_equal(result.source_to_loaded, np.tile(np.arange(3), (12, 1)))
    assert result.switches == ()
    assert result.unresolved_intervals == ()
    assert result.source_to_loaded.dtype == np.int32


def test_repeated_switches_are_found_at_exact_frames():
    original = trajectories()
    loaded = original.copy()
    swap(loaded, 0, 1, 4, 9)
    result = infer_refinement_identity(original, loaded)
    expected = np.tile(np.arange(3), (12, 1))
    expected[4:9] = [1, 0, 2]
    np.testing.assert_array_equal(result.source_to_loaded, expected)
    assert result.switches == (
        IdentitySwitch(0, 0, 1, 4, 4), IdentitySwitch(1, 1, 0, 4, 4),
        IdentitySwitch(0, 1, 0, 9, 9), IdentitySwitch(1, 0, 1, 9, 9),
    )


def test_three_way_permutation_from_first_frame():
    original = trajectories()
    loaded = original.copy()
    swap(loaded, 0, 1)
    swap(loaded, 1, 2)
    result = infer_refinement_identity(original, loaded)
    np.testing.assert_array_equal(result.source_to_loaded, np.tile([2, 0, 1], (12, 1)))
    assert len(result.switches) == 3
    assert all(s.start_frame == s.end_frame == 0 for s in result.switches)


def test_direction_flips_do_not_become_identity_switches():
    original = trajectories()
    loaded = original.copy()
    for tid in range(3):
        cols = [f"x{2 * tid}", f"y{2 * tid}", f"x{2 * tid + 1}", f"y{2 * tid + 1}"]
        loaded.loc[3:8, cols] = loaded.loc[3:8, cols].to_numpy()[:, [2, 3, 0, 1]]
    result = infer_refinement_identity(original, loaded)
    assert result.switches == ()
    assert result.unresolved_intervals == ()


def test_endpoint_edits_and_last_remaining_id_preserve_switched_correspondence():
    original = trajectories()
    loaded = original.copy()
    swap(loaded, 0, 1, 4)
    loaded.loc[4:, ["x2", "y2"]] += 500  # Original ID 0 keeps its rear endpoint.
    loaded.loc[4:, ["x0", "y0", "x1", "y1"]] += 700  # Remaining ID by elimination.
    result = infer_refinement_identity(original, loaded)
    np.testing.assert_array_equal(result.source_to_loaded[4:], np.tile([1, 0, 2], (8, 1)))
    assert result.unresolved_intervals == ()


def test_fully_edited_gap_uses_only_agreeing_neighbors():
    original = trajectories()
    loaded = original.copy()
    swap(loaded, 0, 1)
    loaded.loc[4:7, original.columns[1:9]] += 1000
    result = infer_refinement_identity(original, loaded)
    np.testing.assert_array_equal(result.source_to_loaded, np.tile([1, 0, 2], (12, 1)))
    assert result.unresolved_intervals == ()


def test_gap_at_switch_keeps_unknown_ids_and_an_interval_for_the_boundary():
    original = trajectories()
    loaded = original.copy()
    swap(loaded, 0, 1, 5)
    loaded.loc[4:6, original.columns[1:9]] += 1000
    result = infer_refinement_identity(original, loaded)
    np.testing.assert_array_equal(result.source_to_loaded[4:7], np.tile([-1, -1, 2], (3, 1)))
    assert result.switches == (IdentitySwitch(0, 0, 1, 4, 7), IdentitySwitch(1, 1, 0, 4, 7))
    assert result.unresolved_intervals == (IdentityInterval(0, 4, 6), IdentityInterval(1, 4, 6))


def test_unbounded_missing_intervals_are_not_extrapolated():
    original = trajectories()
    loaded = original.copy()
    loaded.loc[0:2, original.columns[1:9]] = np.nan
    loaded.loc[9:11, original.columns[1:9]] = np.nan
    result = infer_refinement_identity(original, loaded)
    np.testing.assert_array_equal(result.source_to_loaded[:3, :2], -np.ones((3, 2), dtype=int))
    np.testing.assert_array_equal(result.source_to_loaded[9:, :2], -np.ones((3, 2), dtype=int))
    assert len(result.unresolved_intervals) == 4


def test_identical_arrows_remain_ambiguous():
    original = trajectories()
    original.iloc[:, 5:9] = original.iloc[:, 1:5].to_numpy()
    result = infer_refinement_identity(original, original.copy())
    np.testing.assert_array_equal(result.source_to_loaded, np.tile([-1, -1, 2], (12, 1)))
    assert result.switches == ()


def test_gap_bridge_cannot_override_an_established_different_id():
    original = trajectories()
    loaded = original.copy()
    loaded.loc[3:7, original.columns[1:9]] += 1000
    # Source 2 is established at loaded ID 0 within the ambiguous gap.
    loaded.loc[3:7, original.columns[1:5]] = original.loc[3:7, original.columns[9:13]].to_numpy()
    loaded.loc[3:7, original.columns[9:13]] += 1000
    result = infer_refinement_identity(original, loaded)
    assert (result.source_to_loaded[3:8, 2] == 0).all()
    assert (result.source_to_loaded[3:8, 0] == -1).all()
    for row in result.source_to_loaded:
        known = row[row >= 0]
        assert len(known) == len(set(known))


def test_rounding_tolerance_does_not_assign_nearby_but_changed_coordinates():
    original = trajectories(frames=1)
    close = original.copy()
    close.iloc[:, 1:] += 0.001
    np.testing.assert_array_equal(infer_refinement_identity(original, close).source_to_loaded, [[0, 1, 2]])
    different = original.copy()
    different.iloc[:, 1:] += 0.01
    np.testing.assert_array_equal(infer_refinement_identity(original, different).source_to_loaded, [[-1, -1, -1]])


def test_single_id_cannot_have_an_identity_switch_even_after_complete_manual_edit():
    original = trajectories(ids=1)
    loaded = original.copy()
    loaded.iloc[:, 1:] += 1000
    np.testing.assert_array_equal(infer_refinement_identity(original, loaded).source_to_loaded, np.zeros((12, 1)))


@pytest.mark.parametrize("change", ["frames", "ids", "sequence", "infinity"])
def test_incompatible_inputs_are_rejected(change):
    original = trajectories()
    loaded = original.copy()
    if change == "frames":
        loaded = loaded.iloc[:-1]
    elif change == "ids":
        loaded = loaded.iloc[:, :9]
    elif change == "sequence":
        loaded["frame"] = loaded["frame"].astype(float)
        loaded.loc[3, "frame"] = 3.5
    else:
        loaded.loc[3, "x0"] = np.inf
    with pytest.raises(ValueError):
        infer_refinement_identity(original, loaded)


def test_inputs_are_not_modified():
    original = trajectories()
    loaded = original.copy()
    swap(loaded, 0, 1, 4)
    original_copy, loaded_copy = original.copy(), loaded.copy()
    infer_refinement_identity(original, loaded)
    pd.testing.assert_frame_equal(original, original_copy)
    pd.testing.assert_frame_equal(loaded, loaded_copy)


def test_conflicting_source_variants_stay_unknown_even_between_matching_frames():
    original = trajectories()
    variant = original.copy()
    variant.iloc[4:7, 1:5] = original.iloc[4:7, 5:9].to_numpy()
    result = infer_refinement_identity(original, original.copy(), source_variants=(variant,))
    np.testing.assert_array_equal(result.source_to_loaded[4:7], np.tile([-1, -1, 2], (3, 1)))
    assert result.unresolved_intervals == (IdentityInterval(0, 4, 6), IdentityInterval(1, 4, 6))
    # The inputs are equivalent sources, not an ordered list of fallback paths.
    reordered = infer_refinement_identity(variant, original.copy(), source_variants=(original,))
    np.testing.assert_array_equal(reordered.source_to_loaded, result.source_to_loaded)


@pytest.mark.parametrize("saved_format", ["amadeus_result", "rear_front"])
def test_source_variants_follow_real_csv_round_trips_with_different_endpoint_geometry(tmp_path, saved_format):
    from gui import convert

    obb_data = {"position": np.arange(4)}
    direction_data = {"position": np.arange(4)}
    for tid in range(2):
        cx = 100.0 + 200 * tid
        cy = 100.0
        corners = [(cx - 5, cy - 20), (cx + 5, cy - 20),
                   (cx + 5, cy + 20), (cx - 5, cy + 20)]
        for corner, (x, y) in enumerate(corners):
            obb_data[f"x{corner}{tid}"] = np.full(4, x)
            obb_data[f"y{corner}{tid}"] = np.full(4, y)
        # These headings intentionally differ from the OBB's long axis.
        direction_data[f"c{tid}"] = np.full(4, 20.0 + 10 * tid)
    obb, directions = pd.DataFrame(obb_data), pd.DataFrame(direction_data)
    original = convert.rear_front_to_legacy_wide(
        convert.convert_obb_direction_to_rear_front(obb, directions), index_col="frame",
    )
    result_original = convert.amadeus_result_from_obb_direction(obb, directions)
    canonical = convert.rear_front_to_legacy_wide(
        convert.amadeus_result_to_rear_front(result_original, round_digits=None), index_col="frame",
    )
    assert not np.allclose(original.iloc[:, 1:], canonical.iloc[:, 1:], atol=0.002)
    if saved_format == "amadeus_result":
        corrected = convert.swap_amadeus_result_ids(result_original, 0, 1, 2)
        path = tmp_path / "result_refined.csv"
        convert.save_amadeus_result_csv(corrected, path)
        outputs = convert.convert_tracking_outputs(path)
        loaded = convert.load_wide_for_gui(outputs.rear_front, standard_front_first=True)
    else:
        corrected = original.copy()
        swap(corrected, 0, 1, 2)
        path = tmp_path / "rear_front_refined.csv"
        convert.save_rear_front_csv(convert.legacy_wide_to_rear_front(corrected, "front_rear"), path)
        loaded = convert.load_wide_for_gui(path, standard_front_first=True)
    mapping = infer_refinement_identity(original, loaded, source_variants=(canonical,))
    np.testing.assert_array_equal(mapping.source_to_loaded, [[0, 1], [0, 1], [1, 0], [1, 0]])
    assert mapping.unresolved_intervals == ()
    assert mapping.switches == (IdentitySwitch(0, 0, 1, 2, 2), IdentitySwitch(1, 1, 0, 2, 2))


def test_source_variants_require_matching_frames_and_id_slots():
    original = trajectories()
    bad_sequence = original.copy()
    bad_sequence.loc[4, "frame"] = 44
    for variant in (original.iloc[:-1], original.iloc[:, :9], bad_sequence):
        with pytest.raises(ValueError):
            infer_refinement_identity(original, original.copy(), source_variants=(variant,))
