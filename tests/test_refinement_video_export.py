# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Mutually exclusive Refinement video styles and corrected OBB data."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gui import convert
from gui.refinement_video_export import (
    DEFAULT_CENTER_RADIUS,
    DEFAULT_ELLIPSE_WIDTH,
    DEFAULT_OBB_WIDTH,
    RENDER_MODES,
    draw_refined_shapes,
    export_refinement_video,
    obb_corners,
    prepare_refined_obb_rows,
)


class OBBGeometryTests(unittest.TestCase):
    def test_heading_zero_and_ninety_match_amadeus_axes(self):
        zero = obb_corners(np.array([60, 60, 50, 20, 0]))
        self.assertEqual(zero.shape, (4, 2))
        np.testing.assert_array_equal(zero.min(axis=0), [50, 35])
        np.testing.assert_array_equal(zero.max(axis=0), [70, 85])
        right = obb_corners(np.array([60, 60, 50, 20, 90]))
        np.testing.assert_array_equal(right.min(axis=0), [35, 50])
        np.testing.assert_array_equal(right.max(axis=0), [85, 70])

    def test_invalid_or_missing_obb_is_skipped(self):
        self.assertIsNone(obb_corners([60, 60, 50, np.nan, 0]))
        self.assertIsNone(obb_corners([60, 60, 0, 20, 0]))
        rgb = np.zeros((120, 120, 3), np.uint8)
        for mode in ("obb", "ellipse"):
            draw_refined_shapes(
                rgb, np.array([[60, 60, 50, np.nan, 0]]),
                [(0, 255, 0)], mode, 1,
            )
        self.assertFalse(rgb.any())

    def test_create_video_defaults(self):
        self.assertEqual(DEFAULT_OBB_WIDTH, 1)
        self.assertEqual(DEFAULT_ELLIPSE_WIDTH, 1)
        self.assertEqual(DEFAULT_CENTER_RADIUS, 3)
        self.assertEqual(RENDER_MODES, ("center", "arrow", "obb", "ellipse"))

    def test_obb_has_triangle_toward_head_and_reverses(self):
        raw = np.array([[60, 60, 50, 20, 0]])
        rgb = np.zeros((120, 120, 3), np.uint8)
        draw_refined_shapes(rgb, raw, [(0, 255, 0)], "obb", 1)
        self.assertTrue(rgb[25, 60].any())  # triangle extends beyond top OBB edge
        self.assertTrue(rgb[35, 50].any())  # OBB outline still present
        self.assertFalse(rgb[95, 60].any())  # no rear triangle
        self.assertFalse(rgb[60, 60].any())  # no central line from an arrow

        reversed_rgb = np.zeros_like(rgb)
        draw_refined_shapes(reversed_rgb, raw, [(0, 255, 0)], "obb", 1, reverse=True)
        self.assertFalse(reversed_rgb[25, 60].any())
        self.assertTrue(reversed_rgb[95, 60].any())
        self.assertEqual(float(raw[0, 4]), 0.0)  # exporting does not modify source

    def test_ellipse_has_create_video_arrowhead_and_reverses(self):
        raw = np.array([[60, 60, 50, 20, 0]])
        rgb = np.zeros((120, 120, 3), np.uint8)
        draw_refined_shapes(rgb, raw, [(0, 255, 0)], "ellipse", 1)
        self.assertTrue(rgb[25, 60].any())  # tangent-arc arrowhead
        self.assertTrue(rgb[60, 50].any())  # elliptical outline
        self.assertFalse(rgb[95, 60].any())  # no rear marker
        self.assertFalse(rgb[60, 60].any())

        reversed_rgb = np.zeros_like(rgb)
        draw_refined_shapes(reversed_rgb, raw, [(0, 255, 0)], "ellipse", 1, reverse=True)
        self.assertFalse(reversed_rgb[25, 60].any())
        self.assertTrue(reversed_rgb[95, 60].any())

    def test_obb_and_ellipse_thickness_is_configurable(self):
        for mode in ("obb", "ellipse"):
            thin = np.zeros((120, 120, 3), np.uint8)
            thick = np.zeros_like(thin)
            row = np.array([[60, 60, 50, 20, 0]])
            draw_refined_shapes(thin, row, [(0, 255, 0)], mode, 1)
            draw_refined_shapes(thick, row, [(0, 255, 0)], mode, 4)
            self.assertGreater(np.count_nonzero(thick), np.count_nonzero(thin))


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
        wide.loc[1, ["x0", "x1"]] += 10.0
        frames, obbs = prepare_refined_obb_rows(
            self.base, wide, 1, {(1, 0)}, {(2, 0)}
        )
        np.testing.assert_array_equal(frames, [0, 1, 2])
        self.assertAlmostEqual(obbs[1, 0, 0], 60.0)
        self.assertAlmostEqual(obbs[1, 0, 2], 40.0)
        self.assertAlmostEqual(obbs[1, 0, 3], 20.0)
        self.assertAlmostEqual(obbs[2, 0, 4], 180.0)
        pd.testing.assert_frame_equal(self.base, original)

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
            prepare_refined_obb_rows(
                self.base.drop(columns="h0"), None, 1, set(), set(),
            )


class FakeCapture:
    def __init__(self):
        self.index = 0
        self.frames = [
            np.zeros((120, 120, 3), dtype=np.uint8) for _ in range(3)
        ]

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
    def run_export(self, mode, width=2, *, reverse=False, show_labels=False):
        capture = FakeCapture()
        writer = FakeWriter()
        # Input frames 0 and 2 have tracking rows. Frame 1 has no tracking row.
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
                poses, np.array([1]), np.array([2]),
                np.array([3]), np.array([4]), [(0, 255, 0)], 1,
                width, 12, mode, show_labels, reverse,
                lambda done, total, msg: events.append((done, total, msg)),
                obb_frames=frames if mode in ("obb", "ellipse") else None,
                obb_values=boxes if mode in ("obb", "ellipse") else None,
            )
        self.assertTrue(writer.released)
        self.assertEqual(len(writer.frames), 3)
        self.assertEqual(events[-1][:2], (3, 3))
        self.assertFalse(writer.frames[1].any())
        return writer.frames

    def test_center_mode_draws_only_center_dots(self):
        frames = self.run_export("center", DEFAULT_CENTER_RADIUS, show_labels=True)
        self.assertTrue(frames[0][60, 60].any())
        self.assertFalse(frames[0][45, 60].any())
        self.assertFalse(frames[0][60, 50].any())
        self.assertTrue(frames[2][60, 90].any())
        self.assertLess(np.count_nonzero(frames[0]), 200)

    def test_center_radius_changes_point_size(self):
        small = self.run_export("center", 2)
        large = self.run_export("center", 7)
        self.assertGreater(np.count_nonzero(large[0]), np.count_nonzero(small[0]))

    def test_arrow_mode_keeps_gui_style_and_uses_selected_width(self):
        small = self.run_export("arrow", 1)
        thick = self.run_export("arrow", 5)
        self.assertFalse(small[0][60, 50].any())
        self.assertTrue(small[0][60, 60].any())
        self.assertGreater(np.count_nonzero(thick[0]), np.count_nonzero(small[0]))

    def test_obb_video_shows_triangle_without_center_arrow(self):
        frames = self.run_export("obb", 1)
        self.assertTrue(frames[0][25, 60].any())
        self.assertTrue(frames[0][35, 50].any())
        self.assertFalse(frames[0][60, 60].any())
        self.assertTrue(frames[2][60, 80].any())

    def test_ellipse_video_shows_arrowhead_without_center_arrow(self):
        frames = self.run_export("ellipse", 1)
        self.assertTrue(frames[0][25, 60].any())
        self.assertTrue(frames[0][60, 50].any())
        self.assertFalse(frames[0][60, 60].any())

    def test_obb_and_ellipse_require_amadeus_geometry(self):
        for mode in ("obb", "ellipse"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "require AMADEUS"):
                    export_refinement_video(
                        "input.mp4", "output.mp4", 0, 0, 1.0, 30.0,
                        np.zeros((1, 5)), np.array([1]), np.array([2]),
                        np.array([3]), np.array([4]), [], 1, 2, 12,
                        mode, False, False, None,
                    )

    def test_unknown_mode_and_zero_width_rejected(self):
        for mode, width in [("nonsense", 1), ("arrow", 0)]:
            with self.subTest(mode=mode, width=width):
                with self.assertRaises(ValueError):
                    export_refinement_video(
                        "input.mp4", "output.mp4", 0, 0, 1.0, 30.0,
                        np.zeros((1, 5)), np.array([1]), np.array([2]),
                        np.array([3]), np.array([4]), [], 1, width, 12,
                        mode, False, False, None,
                    )


if __name__ == "__main__":
    unittest.main()
