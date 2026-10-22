# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Canvas events must be harmless before a segmentation preview is available."""

import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(
    importlib.util.find_spec("customtkinter") is not None,
    "The GUI dependencies are not installed",
)
class SegmentationCanvasTests(unittest.TestCase):
    def test_click_on_empty_canvas_does_not_access_numpy_or_roi(self):
        from gui.gui_segmentation import CrossingReviewApp

        def unexpected(*_args):
            raise AssertionError("An empty canvas must not process image coordinates")

        stub = SimpleNamespace(
            frame_bgr_cache=None,
            last_drag_canvas=(4, 5),
            _canvas_to_image=unexpected,
            _start_roi_drag=unexpected,
        )
        CrossingReviewApp.on_canvas_press(stub, SimpleNamespace(x=15, y=20))
        self.assertIsNone(stub.last_drag_canvas)


if __name__ == "__main__":
    unittest.main()
