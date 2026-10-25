# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Polygon ROI vertex insertion, settings, and rasterization regression tests."""

import unittest

from main.roi import (
    build_single_roi_mask,
    insert_longest_edge_midpoint,
    insert_roi_vertex_at,
    load_roi_settings,
    normalize_roi_set,
)


class RoiVertexTests(unittest.TestCase):
    def test_longest_edge_midpoint_and_repeated_insertion(self):
        rectangle = [[0, 0], [20, 0], [20, 10], [0, 10]]
        five = insert_longest_edge_midpoint(rectangle)
        self.assertEqual(five, [[0, 0], [10, 0], [20, 0], [20, 10], [0, 10]])
        six = insert_longest_edge_midpoint(five)
        self.assertEqual(six, [[0, 0], [10, 0], [20, 0], [20, 10], [10, 10], [0, 10]])
        self.assertEqual(len(rectangle), 4)

    def test_closing_edge_can_be_split(self):
        polygon = [[0, 0], [2, 0], [2, 2], [0, 20]]
        self.assertEqual(
            insert_longest_edge_midpoint(polygon),
            [[0, 0], [2, 0], [2, 2], [0, 20], [0, 10]],
        )

    def test_too_short_to_split_at_integer_resolution(self):
        self.assertIsNone(insert_longest_edge_midpoint([[0, 0], [1, 0], [0, 1]]))

    def test_right_click_inserts_on_closest_edge(self):
        rectangle = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertEqual(
            insert_roi_vertex_at(rectangle, (8, 3)),
            [[0, 0], [10, 0], [8, 3], [10, 10], [0, 10]],
        )
        self.assertEqual(
            insert_roi_vertex_at(rectangle, (-3, 5)),
            [[0, 0], [10, 0], [10, 10], [0, 10], [-3, 5]],
        )

    def test_right_click_ignores_existing_vertices(self):
        rectangle = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertIsNone(insert_roi_vertex_at(rectangle, (0, 0)))

    def test_normalization_round_trip_for_more_than_four_vertices(self):
        polygon = [[0, 0], [10, 0], [12, 5], [10, 10], [0, 10], [0, 5]]
        roi = normalize_roi_set({"enabled": True, "points": polygon}, (20, 20))
        self.assertEqual(roi["points"], polygon)
        loaded, reverse = load_roi_settings({"roi_sets": [roi], "roi_reverse": True}, (20, 20))
        self.assertEqual(loaded[0]["points"], polygon)
        self.assertTrue(reverse)

    def test_three_vertices_or_empty_is_valid(self):
        self.assertEqual(len(normalize_roi_set({"points": [[0, 0], [5, 0], [0, 5]]})["points"]), 3)
        self.assertEqual(normalize_roi_set({"points": []})["points"], [])

    def test_incomplete_polygons_are_rejected(self):
        for vertices in ([[0, 0]], [[0, 0], [1, 1]]):
            with self.subTest(vertices=vertices):
                with self.assertRaises(ValueError):
                    normalize_roi_set({"points": vertices})

    def test_longest_edge_insertion_preserves_rectangle_mask(self):
        import numpy as np

        polygon = [[2, 3], [18, 3], [18, 17], [2, 17]]
        after = insert_longest_edge_midpoint(polygon)
        first = build_single_roi_mask((24, 24), {"points": polygon})
        second = build_single_roi_mask((24, 24), {"points": after})
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
