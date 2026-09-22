# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Drive the video-input dialog the way a user does, on a headless display.

Skips itself when Tk, CustomTkinter, OpenCV, a display or the pinned FFmpeg
build is missing, so it never blocks a run on a machine without them.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import video_compat as vc  # noqa: E402


def _missing_requirement() -> str:
    if not os.environ.get("DISPLAY") and sys.platform not in ("darwin", "win32"):
        return "no display"
    for module in ("tkinter", "customtkinter", "cv2"):
        try:
            __import__(module)
        except ImportError:
            return f"{module} is unavailable"
    try:
        vc.ffmpeg_executable()
    except vc.FfmpegUnavailableError:
        return "the pinned FFmpeg build is not installed"
    return ""


MISSING = _missing_requirement()


@unittest.skipIf(MISSING, MISSING)
class VideoConversionDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import customtkinter as ctk
        import tkinter as tk

        cls.ctk = ctk
        try:
            cls.root = ctk.CTk()
        except tk.TclError as exc:
            raise unittest.SkipTest(f"Tk could not start: {exc}")
        cls.root.withdraw()

        cls.tmp = tempfile.mkdtemp(prefix="amadeus-video-dialog-")
        cls.source = os.path.join(cls.tmp, "clip.mp4")
        completed = subprocess.run(
            [
                vc.ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", cls.source,
            ],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise unittest.SkipTest(f"could not build a test video: {completed.stderr}")

        from gui import video_input

        cls.video_input = video_input

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _assessment(self, **probe_overrides) -> vc.VideoAssessment:
        """A real assessment of the test video, nudged into a given verdict."""
        assessment = vc.assess_video(self.source)
        if not probe_overrides:
            return assessment
        from dataclasses import replace

        return replace(
            assessment,
            status=probe_overrides.pop("status", vc.STATUS_RECOMMENDED),
            reasons=probe_overrides.pop("reasons", ("Test reason.",)),
        )

    def _dialog(self, assessment, *, allow_source=True):
        dialog = self.video_input.VideoConversionDialog(
            self.root, assessment, allow_source=allow_source
        )
        self.root.update()
        return dialog

    def test_frame_range_defaults_to_the_whole_video(self):
        assessment = self._assessment(status=vc.STATUS_RECOMMENDED)
        dialog = self._dialog(assessment)
        try:
            self.assertEqual(dialog.start_var.get(), "0")
            self.assertEqual(dialog.end_var.get(), str(assessment.usable_frame_count - 1))
            self.assertIn("whole video", dialog.range_hint.cget("text"))
            self.assertEqual(dialog.convert_button.cget("state"), "normal")
        finally:
            dialog.destroy()

    def test_an_invalid_range_blocks_the_convert_button(self):
        dialog = self._dialog(self._assessment())
        try:
            for start, end, expected in (
                ("-1", "100", "negative"),
                ("0", "100000", "exceed"),
                ("90", "10", "after"),
                ("x", "10", "whole numbers"),
            ):
                with self.subTest(start=start, end=end):
                    dialog.start_var.set(start)
                    dialog.end_var.set(end)
                    self.root.update()
                    self.assertEqual(dialog.convert_button.cget("state"), "disabled")
                    self.assertIn(expected, dialog.range_hint.cget("text"))

            dialog.start_var.set("10")
            dialog.end_var.set("39")
            self.root.update()
            self.assertEqual(dialog.convert_button.cget("state"), "normal")
            self.assertIn("30 frames", dialog.range_hint.cget("text"))
        finally:
            dialog.destroy()

    def test_whole_video_button_restores_the_full_range(self):
        assessment = self._assessment()
        dialog = self._dialog(assessment)
        try:
            dialog.start_var.set("10")
            dialog.end_var.set("20")
            self.root.update()
            dialog._reset_range()
            self.root.update()
            self.assertEqual(dialog.start_var.get(), "0")
            self.assertEqual(dialog.end_var.get(), str(assessment.usable_frame_count - 1))
        finally:
            dialog.destroy()

    def test_cancelling_yields_nothing(self):
        dialog = self._dialog(self._assessment())
        dialog._on_close()
        self.root.update()
        self.assertIsNone(dialog.result)

    def test_using_the_original_returns_the_source_unconverted(self):
        dialog = self._dialog(self._assessment())
        dialog._use_source()
        self.root.update()
        self.assertIsNotNone(dialog.result)
        self.assertFalse(dialog.result.converted)
        self.assertEqual(dialog.result.path, self.source)
        self.assertEqual(dialog.result.config_record(), {})

    def test_a_required_conversion_offers_no_way_past_it(self):
        assessment = self._assessment(status=vc.STATUS_REQUIRED)
        dialog = self._dialog(assessment, allow_source=False)
        try:
            self.assertFalse(dialog.source_button.winfo_ismapped())
        finally:
            dialog.destroy()

    def test_converting_a_range_through_the_dialog(self):
        assessment = self._assessment()
        dialog = self._dialog(assessment)
        warnings: list = []
        original = self.video_input.messagebox.showwarning
        self.video_input.messagebox.showwarning = lambda *a, **k: warnings.append(a)
        try:
            dialog.start_var.set("30")
            dialog.end_var.set("59")
            self.root.update()
            dialog._start_conversion()

            deadline = time.monotonic() + 120
            while dialog.winfo_exists() and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.02)
            self.assertFalse(dialog.winfo_exists(), "the conversion did not finish in time")
        finally:
            self.video_input.messagebox.showwarning = original

        result = dialog.result
        self.assertIsNotNone(result)
        self.assertTrue(result.converted)
        self.assertTrue(os.path.isfile(result.path))
        self.assertNotEqual(result.path, self.source)
        self.assertTrue(os.path.isfile(result.metadata_path))

        record = result.config_record()
        self.assertEqual(record["source_video_path"], os.path.abspath(self.source))
        self.assertEqual(record["source_first_frame"], 30)
        self.assertEqual(record["source_last_frame"], 59)
        self.assertTrue(record["frame_mapping_exact"])

        converted = vc.assess_video(result.path)
        self.assertEqual(converted.status, vc.STATUS_OK)
        self.assertEqual(converted.usable_frame_count, 30)

    def test_a_readable_mp4_is_returned_without_any_dialog(self):
        """The existing MP4 workflow must not gain a prompt."""
        shown: list = []
        original = self.video_input.VideoConversionDialog
        self.video_input.VideoConversionDialog = lambda *a, **k: shown.append(a)
        try:
            prepared = self.video_input.prepare_analysis_video(self.root, self.source)
        finally:
            self.video_input.VideoConversionDialog = original
        self.assertEqual(shown, [], "a conversion dialog was shown for a plain MP4")
        self.assertIsNotNone(prepared)
        self.assertFalse(prepared.converted)
        self.assertEqual(prepared.path, self.source)


if __name__ == "__main__":
    unittest.main()
