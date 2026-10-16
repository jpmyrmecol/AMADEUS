"""Regression tests for Easy Tracking's YOLO training progress accounting.

Extract the real methods without importing Tk or GPU dependencies.
"""
import ast
from pathlib import Path
import re
import time
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "gui" / "gui_easy_tracking.py"
TRAIN_DESC = re.compile(
    r"^(\d+)/(\d+)\s+([\d.]+)G\s+\S+\s+\S+\s+\S+\s+\d+\s+(\d+)$"
)


def progress_harness():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    gui = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "EasyTrackingGUI")
    names = {
        "_reset_training_progress", "_set_training_epoch_metadata",
        "_training_progress", "_record_training_loss",
    }
    methods = [n for n in gui.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    cls = ast.ClassDef(
        name="ProgressHarness", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    ns = {
        "time": time,
        "_YOLO_TRAIN_DESC_PAT": TRAIN_DESC,
        "_TRAIN_FIELD_GAP": "   ",
    }
    mod = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(mod, str(SOURCE), "exec"), ns)
    return ns["ProgressHarness"]()


class EasyTrainingImagesProgressTests(unittest.TestCase):
    def setUp(self):
        self.h = progress_harness()
        self.h._reset_training_progress()
        self.h._batch_losses = []
        self.h._batch_loss_images = []
        self.h._refine_losses = []
        self.h._refine_loss_images = []
        self.display_events = []
        self.h._on_tqdm_progress = lambda *args: None
        self.h._post_ui_event = lambda *args: self.display_events.append(args)
        self.h._training_backend_label = "Apple MPS"

    def test_epoch_progress_fps_and_partial_batch(self):
        h = self.h
        desc = "1/50 0G 0.123 0.456 0.789 11 640"
        h._set_training_epoch_metadata(1, 50, 34, 16)
        h._train_epoch_started_at = time.monotonic() - 2
        progress, caption = h._training_progress(desc, 1, 3)
        self.assertAlmostEqual(progress, 1 / 3)
        self.assertEqual(h._train_processed_images, 16)
        self.assertIn("Epoch: 1/50", caption)
        self.assertIn("Apple MPS", caption)
        self.assertIn("640px", caption)
        self.assertIn("8fps", caption)

        progress, caption = h._training_progress(desc, 3, 3)
        self.assertEqual(progress, 1.0)
        self.assertEqual(h._train_processed_images, 34)  # 16 + 16 + 2

        h._set_training_epoch_metadata(2, 50, 34, 16)
        progress, _caption = h._training_progress(
            "2/50 0G 0.123 0.456 0.789 11 640", 0, 3,
        )
        self.assertEqual(progress, 0.0)
        self.assertEqual(h._train_processed_images, 34)
        progress, _caption = h._training_progress(
            "2/50 0G 0.123 0.456 0.789 11 640", 3, 3,
        )
        self.assertEqual(progress, 1.0)
        self.assertEqual(h._train_processed_images, 68)

    def test_loss_deduplicates_redraws_and_recovery(self):
        h = self.h
        h._record_training_loss(1.1, 16, refine=False)
        h._record_training_loss(1.2, 16, refine=False)
        self.assertEqual(h._batch_loss_images, [16])
        self.assertEqual(h._batch_losses, [1.2])
        h._record_training_loss(0.9, 32, refine=False)
        h._record_training_loss(0.8, 34, refine=False)
        h._record_training_loss(1.0, 20, refine=False)
        self.assertEqual(h._batch_loss_images, [16, 20])
        self.assertEqual(h._batch_losses, [1.2, 1.0])

    def test_refine_loss_uses_independent_image_axis(self):
        h = self.h
        h._record_training_loss(0.4, 16, refine=True)
        self.assertEqual(h._refine_loss_images, [16])
        self.assertEqual(h._refine_losses, [0.4])
        self.assertFalse(h._batch_loss_images)


if __name__ == "__main__":
    unittest.main()
