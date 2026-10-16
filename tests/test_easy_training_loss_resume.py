"""Dependency-free regression tests for Easy Tracking's resumed Loss preview."""
import ast
import csv
import glob
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "gui" / "gui_easy_tracking.py"


def build_harness():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    gui = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "EasyTrackingGUI")
    names = {
        "_read_yolo_results_csv", "_show_training_loss",
        "_set_training_epoch_metadata", "_record_training_loss",
    }
    methods = [node for node in gui.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(name="Harness", bases=[], keywords=[],
                           body=methods, decorator_list=[])],
        type_ignores=[],
    ))
    namespace = {
        "csv": csv, "glob": glob, "math": math, "os": os, "re": re,
        "OUTPUT_ROOT_DIR": "results", "TRAINING_STAGE_DIRS": ("training",),
        "time": __import__("time"),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    harness = namespace["Harness"]()
    harness._batch_losses = []
    harness._batch_loss_images = []
    harness._train_epoch = 0
    harness._train_epoch_images = 0
    harness._train_batch_size = 0
    harness._training_backend_label = "NVIDIA CUDA"
    harness._active_step = 7
    harness._post_ui_event = lambda *_args: None
    harness._on_tqdm_progress = lambda *_args: None
    harness._canvas_status = SimpleNamespace(configure=lambda **kwargs: setattr(
        harness, "status", kwargs["text"]
    ))
    harness._draw_training_monitor = lambda data: setattr(harness, "chart", data)
    harness._draw_placeholder = lambda text: setattr(harness, "placeholder", text)
    return harness


def write_csv(root, epochs):
    folder = Path(root) / "results" / "run" / "training" / "dataset" 
    folder.mkdir(parents=True)
    path = folder / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train/box_loss", "val/box_loss", "metrics/mAP50(B)",
                         "metrics/mAP50-95(B)"))
        for epoch, loss in epochs:
            writer.writerow((epoch, loss, 8.0, 0.5, 0.4))
    return path


class ResumedTrainingLossTests(unittest.TestCase):
    def test_resume_excludes_uncommitted_epoch_and_restores_history(self):
        with tempfile.TemporaryDirectory() as root:
            write_csv(root, [(1, 1.2), (2, 0.9), (3, 0.7), (4, 0.1)])
            h = build_harness()
            # Epoch 4 is being replayed; its old CSV row is not valid.
            h._set_training_epoch_metadata(4, 50, 10, 4)
            h._record_training_loss(0.65, 34, refine=False)
            h._record_training_loss(0.59, 38, refine=False)
            h._show_training_loss(root)
            self.assertEqual(h.chart["loss_series"][0][0],
                             [1.2, 0.9, 0.7, 0.65, 0.59])
            self.assertEqual(h.chart["loss_x_values"], [10, 20, 30, 34, 38])
            self.assertEqual(h.chart["loss_history_count"], 3)
            self.assertEqual(h.chart["loss_x_label"], "Images")
            self.assertEqual(len(h.chart["loss_series"]), 1)
            self.assertIn("Batch size 4", h.status)
            self.assertNotIn("val/box_loss", h.status)

            # Starting the next epoch discards previous live batches. The new
            # CSV row for the now-completed epoch takes their place.
            write_path = Path(root) / "results" / "run" / "training" / "dataset" / "results.csv"
            with write_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("epoch", "train/box_loss", "val/box_loss"))
                writer.writerows([(1, 1.2, 8), (2, 0.9, 8), (3, 0.7, 8),
                                  (4, 0.58, 8)])
            h._set_training_epoch_metadata(5, 50, 10, 4)
            h._show_training_loss(root)
            self.assertEqual(h.chart["loss_history_count"], 4)
            self.assertEqual(h.chart["loss_series"][0][0], [1.2, 0.9, 0.7, 0.58])
            self.assertEqual(h.chart["loss_x_values"], [10, 20, 30, 40])

    def test_new_gui_shows_csv_before_live_batch(self):
        with tempfile.TemporaryDirectory() as root:
            write_csv(root, [(1, 1.0), (2, 0.8)])
            h = build_harness()
            h._set_training_epoch_metadata(3, 50, 16, 8)
            h._show_training_loss(root)
            self.assertEqual(h.chart["loss_history_count"], 2)
            self.assertEqual(h.chart["loss_x_values"], [16, 32])
            self.assertEqual(h.chart["loss_series"][0][0], [1.0, 0.8])

    def test_live_plot_is_bounded_and_includes_last_batch(self):
        with tempfile.TemporaryDirectory() as root:
            write_csv(root, [(1, 1.0)])
            h = build_harness()
            h._set_training_epoch_metadata(2, 50, 5000, 1)
            h._batch_loss_images = list(range(5001, 7501))
            h._batch_losses = [0.8] * 2500
            h._show_training_loss(root)
            self.assertLessEqual(len(h.chart["loss_series"][0][0]), 1202)
            self.assertEqual(h.chart["loss_x_values"][-1], 7500)
            self.assertEqual(h.chart["loss_history_count"], 1)

    def test_loss_chart_legend_is_explicitly_disabled(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        gui = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "EasyTrackingGUI")
        render = next(node for node in gui.body
                      if isinstance(node, ast.FunctionDef) and
                      node.name == "_render_training_monitor")
        calls = [node for node in ast.walk(render)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "_render_multi_line_chart"]
        self.assertEqual(len(calls), 1)
        options = {k.arg: k.value for k in calls[0].keywords}
        self.assertIs(options["show_legend"].value, False)


if __name__ == "__main__":
    unittest.main()
