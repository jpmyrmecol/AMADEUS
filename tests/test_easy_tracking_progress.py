# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""The Training progress bar in Easy Tracking.

Ultralytics' tqdm row is fed through the real output handler and the resulting
bar text and fraction are checked, so the wiring is covered and not just the
formatter. Skips itself where Tk or a display is missing.
"""

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _missing_requirement() -> str:
    if not os.environ.get("DISPLAY") and sys.platform not in ("darwin", "win32"):
        return "no display"
    for module in ("tkinter", "customtkinter"):
        try:
            __import__(module)
        except ImportError:
            return f"{module} is unavailable"
    return ""


MISSING = _missing_requirement()

# Rows exactly as Ultralytics' DetectionTrainer prints them, with tqdm's bar.
# Columns: Epoch, GPU_mem, box_loss, cls_loss, dfl_loss, Instances, Size.
BAR = "█" * 10


def training_line(epoch: int, total: int, mem: str, batch: int, batches: int,
                  size: int = 1024, instances: int = 37) -> str:
    percent = int(round(100 * batch / batches))
    return (
        f"       {epoch}/{total}      {mem}     0.8756     0.7188      1.492    {instances:>10}"
        f"       {size}: {percent:3d}%|{BAR}| {batch}/{batches} [00:23<00:00,  1.95s/it]"
    )


VALIDATION_LINE = (
    "                 Class     Images  Instances      Box(P          R      mAP50  "
    "mAP50-95): 100%|" + BAR + "| 4/4 [00:02<00:00,  1.72it/s]"
)
SCAN_LINE = (
    "train: Scanning /data/labels.cache... 400 images, 0 backgrounds, 0 corrupt: "
    "100%|" + BAR + "| 400/400 [00:00<00:00, 900.00it/s]"
)


class FakeClock:
    """Stands in for the time module inside the GUI."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@unittest.skipIf(MISSING, MISSING)
class TrainingProgressTests(unittest.TestCase):
    TRAINING_STEP = 7  # STEPS[7] is "Training" / obb_detector_training

    @classmethod
    def setUpClass(cls):
        import gui.gui_easy_tracking as module

        cls.module = module

    def setUp(self):
        module = self.module
        self.app = module.EasyTrackingGUI()
        # The GUI schedules window-icon and maximise callbacks; cancel them
        # before teardown so Tcl does not complain about the destroyed window.
        self.addCleanup(self._destroy_app)
        self.app.withdraw()

        self.clock = FakeClock()
        self._real_time = module.time
        self._real_vram = module._total_vram_gib
        module.time = self.clock
        module._total_vram_gib = lambda: 12.0

        def restore():
            module.time = self._real_time
            module._total_vram_gib = self._real_vram

        self.addCleanup(restore)

        self.app._active_step = self.TRAINING_STEP
        self.app._step_start_times = {self.TRAINING_STEP: self.clock.time()}
        self.app._reset_training_progress()
        self.state = {"done_count": 0, "last_was_tqdm": False, "last_tqdm_len": 0}

    def _destroy_app(self) -> None:
        try:
            for job in self.app.tk.call("after", "info"):
                try:
                    self.app.after_cancel(job)
                except Exception:
                    pass
        except Exception:
            pass
        self.app.destroy()

    def feed(self, line: str) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.app._handle_batch_output_record(line, self.state)
        self.app.update()

    @property
    def text(self) -> str:
        return self.app._progress_text

    # -- the requested format ---------------------------------------------

    def test_the_first_epoch_shows_every_field_and_no_estimate_yet(self):
        self.clock.advance(128)  # 2m 08s since the step started
        self.feed(training_line(1, 50, "4.97G", 6, 12))
        self.assertEqual(
            self.text,
            "Epoch: 1/50   VRAM: 4.63/12.0 GiB   Image size: 1024 px   "
            "Time: 2m 08s / Calculating...",
        )

    def test_an_estimate_appears_once_an_epoch_has_finished(self):
        self.feed(training_line(1, 50, "4.97G", 1, 12))
        self.clock.advance(60)  # epoch 1 took a minute
        self.feed(training_line(2, 50, "4.97G", 1, 12))
        self.clock.advance(68)
        self.feed(training_line(2, 50, "4.97G", 6, 12))
        # 50 epochs * 60 s = 50 m, and the bar has been running 2m 08s.
        self.assertEqual(
            self.text,
            "Epoch: 2/50   VRAM: 4.63/12.0 GiB   Image size: 1024 px   Time: 2m 08s / ~50m",
        )

    def test_the_estimate_is_the_mean_of_finished_epochs_not_the_last_one(self):
        self.feed(training_line(1, 10, "4.97G", 1, 12))
        self.clock.advance(60)
        self.feed(training_line(2, 10, "4.97G", 1, 12))   # epoch 1: 60 s
        self.clock.advance(180)
        self.feed(training_line(3, 10, "4.97G", 1, 12))   # epoch 2: 180 s
        # A last-epoch estimate would read ~30m; the mean of 60 s and 180 s gives ~20m.
        self.assertIn("/ ~20m", self.text)

    def test_the_estimate_includes_the_setup_before_the_first_epoch(self):
        self.clock.advance(120)  # dataset scan and model build
        self.feed(training_line(1, 10, "4.97G", 1, 12))
        self.clock.advance(60)
        self.feed(training_line(2, 10, "4.97G", 1, 12))
        # 120 s of setup + 10 epochs of 60 s = 12 minutes.
        self.assertIn("/ ~12m", self.text)

    def test_hours_are_shown_for_a_long_run(self):
        self.feed(training_line(1, 100, "4.97G", 1, 12))
        self.clock.advance(120)
        self.feed(training_line(2, 100, "4.97G", 1, 12))
        self.assertIn("/ ~3h 20m", self.text)
        self.clock.advance(3600)
        self.feed(training_line(2, 100, "4.97G", 2, 12))
        self.assertIn("Time: 1h 02m 00s /", self.text)

    # -- VRAM ---------------------------------------------------------------

    def test_cpu_and_mps_runs_show_no_vram_field(self):
        # Ultralytics prints 0G whenever torch.cuda is unavailable.
        self.feed(training_line(1, 50, "0G", 6, 12))
        self.assertNotIn("VRAM", self.text)
        self.assertEqual(
            self.text,
            "Epoch: 1/50   Image size: 1024 px   Time: 0s / Calculating...",
        )

    def test_an_unreadable_gpu_total_still_shows_what_is_in_use(self):
        self.module._total_vram_gib = lambda: None
        self.feed(training_line(1, 50, "4.97G", 6, 12))
        self.assertIn("VRAM: 4.63 GiB", self.text)

    def test_both_halves_of_the_vram_field_are_the_same_unit(self):
        """Ultralytics reports bytes/1e9, nvidia-smi MiB; both are shown in GiB."""
        self.module._total_vram_gib = lambda: 12.0
        self.feed(training_line(1, 50, "8.00G", 6, 12))
        # 8.00e9 bytes is 7.45 GiB, not 8.00.
        self.assertIn("VRAM: 7.45/12.0 GiB", self.text)
        self.assertNotIn(" GB", self.text)

    # -- what must not be there ---------------------------------------------

    def test_losses_instances_and_batch_numbers_are_gone(self):
        # Values chosen so none of them can appear inside the fields that are
        # shown ("4.97/12.0" contains "7/12", for instance).
        self.feed(training_line(3, 50, "4.97G", 7, 341, instances=883))
        for noise in ("0.8756", "0.7188", "1.492", "883", "7/341"):
            self.assertNotIn(noise, self.text, f"{noise!r} should not be on the bar")

    # -- the bar itself ------------------------------------------------------

    def test_the_bar_runs_continuously_across_every_epoch(self):
        self.feed(training_line(1, 10, "4.97G", 0, 10))
        self.assertAlmostEqual(self.app._progress_pct, 0.0, places=6)
        self.feed(training_line(1, 10, "4.97G", 5, 10))
        self.assertAlmostEqual(self.app._progress_pct, 0.05, places=6)
        self.feed(training_line(6, 10, "4.97G", 5, 10))
        self.assertAlmostEqual(self.app._progress_pct, 0.55, places=6)
        self.feed(training_line(10, 10, "4.97G", 10, 10))
        self.assertAlmostEqual(self.app._progress_pct, 1.0, places=6)

    def test_the_live_loss_chart_still_receives_per_batch_losses(self):
        self.feed(training_line(1, 50, "4.97G", 1, 12))
        self.assertEqual(self.app._batch_losses, [0.8756])

    # -- everything else is untouched ---------------------------------------

    def test_validation_and_scanning_bars_keep_their_old_display(self):
        for line, expected_tail in (
            (VALIDATION_LINE, "4/4"),
            (SCAN_LINE, "400/400"),
        ):
            with self.subTest(line=line.split(":")[0]):
                self.feed(line)
                self.assertTrue(
                    self.app._progress_text.endswith(expected_tail),
                    self.app._progress_text,
                )
                self.assertNotIn("Epoch:", self.app._progress_text)

    def test_a_plain_tqdm_bar_from_another_step_is_unchanged(self):
        self.app._active_step = 0
        self.feed("Tracking blobs: 42%|" + BAR + "| 420/1000 [00:10<00:14, 40.0it/s]")
        self.assertEqual(self.app._progress_text, "Tracking blobs  420/1000")
        self.assertAlmostEqual(self.app._progress_pct, 0.42, places=6)

    def test_a_non_tqdm_line_does_not_touch_the_bar(self):
        self.feed(training_line(1, 50, "4.97G", 6, 12))
        before = self.app._progress_text
        self.feed("[INFO] writing results.csv")
        self.assertEqual(self.app._progress_text, before)


@unittest.skipIf(MISSING, MISSING)
class DurationFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gui.gui_easy_tracking as module

        cls.module = module

    def test_elapsed(self):
        for seconds, expected in ((0, "0s"), (8, "8s"), (128, "2m 08s"),
                                  (3600, "1h 00m 00s"), (3908, "1h 05m 08s")):
            with self.subTest(seconds=seconds):
                self.assertEqual(self.module._format_elapsed(seconds), expected)

    def test_estimate_is_rounded_to_whole_minutes(self):
        for seconds, expected in ((10, "~1m"), (3120, "~52m"), (3140, "~52m"),
                                  (5400, "~1h 30m"), (86400, "~24h 00m")):
            with self.subTest(seconds=seconds):
                self.assertEqual(self.module._format_estimate(seconds), expected)


if __name__ == "__main__":
    unittest.main()
