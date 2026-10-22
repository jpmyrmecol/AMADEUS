# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""OBB + arrow rendering and unsaved Refinement edits."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gui import convert
from gui.refinement_video_export import (
    draw_refined_obbs,
    export_refinement_video,
    obb_corners,
    prepare_refined_obb_rows,
)


class OBBGeometryTests(unittest.TestCase):
    def test_heading_zero_and_ninety_match_amadeus_axes(self):
        zero = obb_corners(np.array([60, 60, 50, 20, 0]))
        self.assertEqual(zero.shape, (4, 1, 2))
        np.testing.assert_array_equal(zero[:, 0, :].min(axis=0), [50, 35])
        np.testing.assert_array_equal(zero[:, 0, :].max(axis=0), [70, 85])

        right = obb_corners(np.array([60, 60, 50, 20, 90]))
        np.testing.assert_array_equal(right[:, 0, :].min(axis=0), [35, 50])
        np.testing.assert_array_equal(right[:, 0, :].max(axis=0), [85, 70])

    def test_invalid_or_missing_obb_does_not_draw(self):
        self.assertIsNone(obb_corners([60, 60, 50, np.nan, 0]))
        self.assertIsNone(obb_corners([60, 60, 0, 20, 0]))
        rgb = np.zeros((120, 120, 3), np.uint8)
        draw_refined_obbs(rgb, np.array([[60, 60, 50, np.nan, 0]]), [(0, 255, 0)])
        self.assertFalse(rgb.any())


class RefinedSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame({
            "frame": [0, 1, 2],
            "cx0": [50.0, 50.0, 50.0],
            "cy0": [60.0, 60.0, 60.0],
            "w0": [40.0, 40.0, 40.0],
            "h0": [20.0, 20.0, 20.0],
            "heading0": [0.0, 0.0, 0.0],
        })

    def test_unsaved_move_and_direction_flip_are_reflected(self):
        original = self.base.copy(deep=True)
        wide = convert.rear_front_to_legacy_wide(
            convert.amadeus_result_to_rear_front(self.base, round_digits=None)
        )
        # Both endpoints translated by 10 px; box width/height are unchanged.
        wide.loc[1, ["x0", "x1"]] += 10.0
        frames, obbs = prepare_refined_obb_rows(
            self.base, wide, 1, {(1, 0)}, {(2, 0)}
        )
        np.testing.assert_array_equal(frames, [0, 1, 2])
        self.assertAlmostEqual(obbs[1, 0, 0], 60.0)  # cx moved
        self.assertAlmostEqual(obbs[1, 0, 2], 40.0)  # w retained
        self.assertAlmostEqual(obbs[1, 0, 3], 20.0)  # h interpolated
        self.assertAlmostEqual(obbs[2, 0, 4], 180.0)
        pd.testing.assert_frame_equal(self.base, original)  # not a save

    def test_identity_swap_uses_swapped_obb_geometry(self):
        base = self.base.copy(deep=True)
        for name, values in {
            "cx1": [80, 81, 82], "cy1": [40, 40, 40],
            "w1": [30, 30, 30], "h1": [10, 10, 10],
            "heading1": [90, 90, 90],
        }.items():
            base[name] = values
        swapped = convert.swap_amadeus_result_ids(base, 0, 1, start_frame=1)
        frames, obbs = prepare_refined_obb_rows(swapped, None, 2, set(), set())
        np.testing.assert_array_equal(frames, [0, 1, 2])
        self.assertAlmostEqual(obbs[0, 0, 0], 50.0)
        self.assertAlmostEqual(obbs[1, 0, 0], 81.0)
        self.assertAlmostEqual(obbs[1, 1, 0], 50.0)

    def test_missing_obb_source_columns_raise(self):
        with self.assertRaisesRegex(ValueError, "OBB columns"):
            prepare_refined_obb_rows(self.base.drop(columns="h0"), None, 1, set(), set())


class FakeCapture:
    def __init__(self):
        self.index = 0
        self.frames = [np.zeros((120, 120, 3), dtype=np.uint8) for _ in range(3)]

    def isOpened(self):
        return True

    def set(self, _property, index):
        self.index = int(index)
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def release(self):
        pass


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class ExportTests(unittest.TestCase):
    def run_export(self, show_obbs):
        capture = FakeCapture()
        writer = FakeWriter()
        # Input frames 0 and 2 have tracks. Frame 1 is absent from the CSV.
        poses = np.array([
            [0, 60, 35, 60, 85],
            [2, 90, 35, 90, 85],
        ], dtype=float)
        frames = np.array([0, 2], dtype=np.int64)
        boxes = np.array([
            [[60, 60, 50, 20, 0]],
            [[90, 60, 50, 20, 0]],
        ], dtype=float)
        events = []
        with (
            patch("gui.refinement_video_export.cv2.VideoCapture", return_value=capture),
            patch("gui.refinement_video_export.cv2.VideoWriter", return_value=writer),
        ):
            export_refinement_video(
                "input.mp4", "output.mp4", 0, 2, 1.0, 30.0,
                poses, np.array([1]), np.array([2]), np.array([3]), np.array([4]),
                [(0, 255, 0)], 1, 2, 12, True, False, False,
                lambda done, total, msg: events.append((done, total, msg)),
                show_obbs=show_obbs, obb_frames=frames if show_obbs else None,
                obb_values=boxes if show_obbs else None,
            )
        self.assertTrue(writer.released)
        self.assertEqual(len(writer.frames), 3)
        self.assertEqual(events[-1][:2], (3, 3))
        return writer.frames

    def test_obb_plus_arrow_video_and_sparse_frame_alignment(self):
        frames = self.run_export(show_obbs=True)
        self.assertTrue(frames[0][60, 50].any())  # left OBB edge
        self.assertTrue(frames[0][60, 60].any())  # central arrow
        self.assertFalse(frames[1].any())  # no CSV at frame 1
        self.assertTrue(frames[2][60, 80].any())  # box of ID 0 at new center

    def test_arrow_only_mode_keeps_previous_export_behavior(self):
        frames = self.run_export(show_obbs=False)
        self.assertFalse(frames[0][60, 50].any())  # OBB omitted
        self.assertTrue(frames[0][60, 60].any())  # arrow remains
        self.assertFalse(frames[1].any())

    def test_obb_mode_rejects_missing_obb_data(self):
        with self.assertRaisesRegex(ValueError, "OBB export requires"):
            export_refinement_video(
                "input.mp4", "output.mp4", 0, 0, 1.0, 30.0,
                np.zeros((1, 5)), np.array([1]), np.array([2]),
                np.array([3]), np.array([4]), [], 1, 2, 12, True, False, False,
                None, show_obbs=True,
            )


if __name__ == "__main__":
    unittest.main()
