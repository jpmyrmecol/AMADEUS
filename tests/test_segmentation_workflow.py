# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Regression coverage for Segmentation's one-time background preview gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parent.parent / "gui" / "segmentation_workflow.py"
SPEC = importlib.util.spec_from_file_location("amadeus_segmentation_workflow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
stage_after_background_change = MODULE.stage_after_background_change


class TestOneTimeBackgroundReview(unittest.TestCase):
    def test_new_video_starts_with_background_review(self):
        self.assertEqual(
            stage_after_background_change("background", needs_background=True, preview_seen=False),
            "background",
        )

    def test_first_background_preview_keeps_threshold_step(self):
        self.assertEqual(
            stage_after_background_change("threshold", needs_background=True, preview_seen=True),
            "threshold",
        )

    def test_switching_median_to_max_retains_analysis_step(self):
        self.assertEqual(
            stage_after_background_change("analysis", needs_background=True, preview_seen=True),
            "analysis",
        )

    def test_switching_max_back_to_median_does_not_regate_preview(self):
        for method in ("median", "max", "median", "min", "max"):
            with self.subTest(method=method):
                self.assertEqual(
                    stage_after_background_change("analysis", needs_background=True, preview_seen=True),
                    "analysis",
                )

    def test_completed_analysis_restarts_at_analysis_not_preview(self):
        for progressed in ("outlier", "processing"):
            with self.subTest(previous_stage=progressed):
                self.assertEqual(
                    stage_after_background_change(
                        progressed, needs_background=True, preview_seen=True
                    ),
                    "analysis",
                )

    def test_unreviewed_background_remains_gated(self):
        for progressed in ("threshold", "analysis", "outlier", "processing"):
            with self.subTest(previous_stage=progressed):
                self.assertEqual(
                    stage_after_background_change(
                        progressed, needs_background=True, preview_seen=False
                    ),
                    "background",
                )

    def test_modes_that_do_not_need_background_never_gate_it(self):
        self.assertEqual(
            stage_after_background_change("background", needs_background=False, preview_seen=False),
            "threshold",
        )
        self.assertEqual(
            stage_after_background_change("analysis", needs_background=False, preview_seen=False),
            "analysis",
        )

    def test_switching_back_to_background_mode_retains_review(self):
        self.assertEqual(
            stage_after_background_change("analysis", needs_background=True, preview_seen=True),
            "analysis",
        )

    def test_gui_records_preview_and_does_not_clear_on_background_invalidation(self):
        gui_source = (MODULE_PATH.parent / "gui_segmentation.py").read_text(encoding="utf-8")
        begin = gui_source.index("    def _invalidate_analysis_for_background_change(self):")
        end = gui_source.index("\n    def ", begin + 8)
        invalidation = gui_source[begin:end]
        self.assertIn("stage_after_background_change(", invalidation)
        self.assertIn("self.analysis_is_stale = True", invalidation)
        self.assertNotIn("self._background_preview_seen = False", invalidation)
        self.assertIn('"background_preview_seen": bool(self._background_preview_seen)', gui_source)
        self.assertIn("self._background_preview_seen = True", gui_source)


if __name__ == "__main__":
    unittest.main()
