# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Object pool donor selection defaults to all source frames."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main"))

from interaction_image_synthesis import donor_candidates_and_scope, sample_donors_from_candidates


class DonorScopeTests(unittest.TestCase):
    def setUp(self):
        self.frame = [{"pool_id": "a", "frame": 1}]
        self.pool = self.frame + [{"pool_id": "b", "frame": 2}]

    def test_default_samples_full_object_pool(self):
        candidates, scope = donor_candidates_and_scope({}, self.pool, self.frame)
        self.assertIs(candidates, self.pool)
        self.assertEqual(scope, "object_pool")
        self.assertEqual({obj["frame"] for obj in candidates}, {1, 2})
        self.assertEqual(len(sample_donors_from_candidates(random.Random(0), candidates, 2)), 2)

    def test_explicit_same_frame_preserves_previous_behavior(self):
        candidates, scope = donor_candidates_and_scope(
            {"PASTE_SAME_FRAME_ONLY": True}, self.pool, self.frame
        )
        self.assertIs(candidates, self.frame)
        self.assertEqual(scope, "same_frame")
        self.assertEqual({obj["frame"] for obj in candidates}, {1})

    def test_explicit_false_keeps_full_pool(self):
        candidates, scope = donor_candidates_and_scope(
            {"PASTE_SAME_FRAME_ONLY": False}, self.pool, self.frame
        )
        self.assertIs(candidates, self.pool)
        self.assertEqual(scope, "object_pool")


if __name__ == "__main__":
    unittest.main()
