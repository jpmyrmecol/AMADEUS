# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Regression tests for high-sensitivity temporal merge detection."""

import unittest
from types import SimpleNamespace

import numpy as np

from main.temporal_merge_blobs import (
    TemporalMergeGraphBuilder,
    TemporalMergeParams,
    evaluate_temporal_merges,
    extract_frame_nodes,
)


def _blob(x0: int, y0: int, x1: int, y1: int):
    contour = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32)
    return SimpleNamespace(
        contour=contour,
        area=float((x1 - x0) * (y1 - y0)),
        center_x=(x0 + x1) / 2.0,
        center_y=(y0 + y1) / 2.0,
        is_crossing=False,
    )


def _graph(episodes):
    """Each episode is (duration, rectangles per frame)."""
    builder = TemporalMergeGraphBuilder((80, 80))
    fid = 0
    for duration, rectangles in episodes:
        for _ in range(duration):
            blobs = [_blob(*r) for r in rectangles]
            builder.add_frame(extract_frame_nodes(fid, blobs, (80, 80)))
            fid += 1
    return builder.finish()


class TemporalMergeHighSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.params = TemporalMergeParams(
            horizon_frames=30, confirm_frames=2, tau=0.30, include_one_sided=True
        )
        # Just one pixel between separate bodies, far below one body length.
        self.separate = ((20, 25, 30, 35), (32, 25, 42, 35))
        self.merged = ((20, 25, 42, 35),)

    def test_adjacent_animals_are_excluded_without_body_length_gap(self):
        graph = _graph([(60, self.separate), (10, self.merged), (60, self.separate)])
        result = evaluate_temporal_merges(graph, self.params)
        self.assertTrue(result.diagnostics["reliable"])
        for fid in range(60, 70):
            self.assertIn(0, result.excluded.get(fid, set()))
            self.assertEqual(result.level(fid, 0), 2)

    def test_one_sided_merge_is_excluded_near_end_of_scan(self):
        graph = _graph([(120, self.separate), (8, self.merged)])
        result = evaluate_temporal_merges(graph, self.params)
        self.assertTrue(result.diagnostics["reliable"])
        for fid in range(120, 128):
            self.assertIn(0, result.excluded.get(fid, set()))
            self.assertEqual(result.level(fid, 0), 1)

    def test_fragments_of_one_body_are_not_double_counted(self):
        whole = ((20, 25, 42, 35),)
        fragments = ((20, 25, 30, 35), (32, 25, 42, 35))
        graph = _graph([(60, whole), (6, fragments), (60, whole)])
        result = evaluate_temporal_merges(graph, self.params)
        self.assertEqual(result.excluded_count, 0)
        self.assertEqual(result.counts["high_blobs"], 0)
        self.assertEqual(result.counts["medium_blobs"], 0)


    def test_long_separation_overrides_old_split_ancestry(self):
        # A historical 1->2 split is not evidence that two long-lived,
        # separately segmented objects are fragments of one animal.
        graph = _graph([
            (60, self.merged),
            (60, self.separate),
            (10, self.merged),
            (60, self.separate),
        ])
        params = TemporalMergeParams(
            horizon_frames=200, confirm_frames=2, tau=0.30, include_one_sided=True
        )
        result = evaluate_temporal_merges(graph, params)
        self.assertTrue(result.diagnostics["reliable"])
        for fid in range(120, 130):
            self.assertIn(0, result.excluded.get(fid, set()))
            self.assertEqual(result.level(fid, 0), 2)

    def test_nearby_new_animal_does_not_clear_existing_merge(self):
        third = (44, 25, 54, 35)
        ab_and_c = (self.merged[0], third)
        abc = ((20, 25, 54, 35),)
        graph = _graph([
            (60, self.separate),
            (10, self.merged),
            (10, ab_and_c),
            (10, abc),
            (60, (self.separate[0], self.separate[1], third)),
        ])
        result = evaluate_temporal_merges(graph, self.params)
        self.assertTrue(result.diagnostics["reliable"])
        for fid in range(70, 80):
            self.assertIn(0, result.excluded.get(fid, set()))
        for fid in range(80, 90):
            self.assertIn(0, result.excluded.get(fid, set()))
        triple = [row for row in result.intervals if row["frame_start"] <= 80 <= row["frame_end"]]
        self.assertTrue(any(row["n_forward"] >= 3 for row in triple))

    def test_three_independent_blobs_merge_and_split(self):
        third = (44, 25, 54, 35)
        separate = (self.separate[0], self.separate[1], third)
        combined = ((20, 25, 54, 35),)
        graph = _graph([(70, separate), (12, combined), (70, separate)])
        result = evaluate_temporal_merges(graph, self.params)
        self.assertTrue(result.diagnostics["reliable"])
        for fid in range(70, 82):
            self.assertEqual(result.level(fid, 0), 2)
            self.assertIn(0, result.excluded.get(fid, set()))
        triple = [row for row in result.intervals if row["frame_start"] <= 70 <= row["frame_end"]]
        self.assertTrue(any(row["n_forward"] >= 3 and row["n_backward"] >= 3 for row in triple))

    def test_parting_animals_stay_single(self):
        graph = _graph([(60, self.separate), (10, self.merged), (60, self.separate)])
        result = evaluate_temporal_merges(graph, self.params)
        for fid in list(range(60)) + list(range(70, 130)):
            self.assertEqual(result.excluded.get(fid, set()), set())

    def test_group_split_leaves_pair_with_its_count(self):
        # Three animals merge, then part as a pair and a single: the pair
        # keeps two of the three known animals by its share of the mass.
        third = (44, 25, 54, 35)
        graph = _graph([
            (60, (self.separate[0], self.separate[1], third)),
            (10, ((20, 25, 54, 35),)),
            (70, (self.merged[0], third)),
        ])
        result = evaluate_temporal_merges(graph, self.params)
        self.assertTrue(result.diagnostics["reliable"])
        for fid in range(70, 140):
            self.assertIn(0, result.excluded.get(fid, set()))
            self.assertNotIn(1, result.excluded.get(fid, set()))

    def test_one_sided_merge_beyond_horizon_needs_its_mass(self):
        # One-sided evidence lasts beyond H while the blob keeps its mass;
        # shrinking to one body (an animal may have left unseen) ends it.
        shrunk = ((20, 25, 30, 35),)
        graph = _graph([(60, self.separate), (50, self.merged), (40, shrunk)])
        result = evaluate_temporal_merges(graph, self.params)
        self.assertTrue(result.diagnostics["reliable"])
        for fid in range(60, 110):
            self.assertIn(0, result.excluded.get(fid, set()))
        for fid in range(110, 150):
            self.assertNotIn(0, result.excluded.get(fid, set()))

    def test_contact_match_ignores_inflated_contact_frame(self):
        # A single body looks pair-sized in the frame before touching an
        # overlapped pair. Matching by that frame would hand it the pair's
        # later count; sizes next to the contact keep it single.
        pair, single = (20, 25, 33, 35), (20, 45, 30, 55)
        graph = _graph([
            (60, (pair, single)),
            (1, (pair, (20, 45, 39, 55))),
            (1, ((20, 25, 39, 55),)),
            (30, ((20, 25, 42, 35), single)),
            (60, (self.separate[0], self.separate[1], single)),
        ])
        result = evaluate_temporal_merges(graph, self.params)
        self.assertTrue(result.diagnostics["reliable"])
        for fid in range(61):
            self.assertNotIn(1, result.excluded.get(fid, set()))
        for fid in range(62, 92):
            self.assertIn(0, result.excluded.get(fid, set()))
            self.assertNotIn(1, result.excluded.get(fid, set()))
        for fid in range(92, 152):
            self.assertNotIn(2, result.excluded.get(fid, set()))

    def test_morphological_shrink_before_merge_retains_confirmation(self):
        shrunk = ((20, 25, 26, 35), self.separate[1])
        graph = _graph([
            (100, self.separate),
            (1, shrunk),
            (10, self.merged),
        ])
        result = evaluate_temporal_merges(graph, self.params)
        self.assertTrue(result.diagnostics["reliable"])
        for fid in range(101, 111):
            self.assertIn(0, result.excluded.get(fid, set()))


if __name__ == "__main__":
    unittest.main()
