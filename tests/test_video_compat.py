# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Regression coverage for seeking and publication of converted videos."""

import io
import json
import threading
import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from main import video_compat as compat


class FakeCapture:
    def __init__(self, bad_frames, frame_count=101):
        self.bad_frames = set(bad_frames)
        self.frame_count = frame_count
        self.position = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            0: self.position,
            1: self.frame_count,
            2: 640,
            3: 480,
            4: 30.0,
            5: 0,
        }[prop]

    def set(self, prop, value):
        assert prop == 0
        self.position = int(value)
        return True

    def read(self):
        index = self.position
        if index in self.bad_frames:
            return False, None
        self.position += 1
        return True, SimpleNamespace(size=1)

    def release(self):
        pass


class FakeCV2:
    CAP_PROP_POS_FRAMES = 0
    CAP_PROP_FRAME_COUNT = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FPS = 4
    CAP_PROP_FOURCC = 5

    def __init__(self, bad_frames):
        self.bad_frames = bad_frames

    def VideoCapture(self, _path):
        return FakeCapture(self.bad_frames)


class SeekingTests(unittest.TestCase):
    def test_unreadable_last_frame_is_excluded_if_other_seeks_work(self):
        with (
            patch.object(compat, "_cv2", return_value=FakeCV2({100})),
            patch.object(compat, "measure_frame_intervals", return_value=(30.0, 0.0)),
            patch.object(compat, "detect_seekable_frame_count", return_value=100) as tail,
        ):
            probe = compat.probe_opencv("example.mov")
        self.assertTrue(probe.random_seek_ok)
        self.assertEqual(probe.reported_frame_count, 101)
        self.assertEqual(probe.seekable_frame_count, 100)
        self.assertEqual(probe.seek_failed_frames, ())
        self.assertEqual(probe.seek_tested_frames, (0, 25, 50, 75))
        tail.assert_called_once_with("example.mov", 101)

        with (
            patch.object(compat, "probe_source", return_value=compat.SourceVideoInfo(codec="h264")),
            patch.object(compat, "probe_opencv", return_value=probe),
        ):
            assessment = compat.assess_video("example.mov")
        self.assertEqual(assessment.status, compat.STATUS_OK)
        self.assertTrue(any("final 1 frame" in note for note in assessment.notes))

    def test_internal_seek_failure_still_requires_conversion(self):
        with (
            patch.object(compat, "_cv2", return_value=FakeCV2({50, 100})),
            patch.object(compat, "measure_frame_intervals", return_value=(30.0, 0.0)),
            patch.object(compat, "detect_seekable_frame_count") as tail,
        ):
            probe = compat.probe_opencv("example.mov")
        self.assertFalse(probe.random_seek_ok)
        self.assertEqual(probe.seek_failed_frames, (50, 100))
        tail.assert_not_called()


class FakeProcess:
    def __init__(self, return_code=0):
        self.return_code = return_code
        self.stdout = io.StringIO("frame=4\n")
        self.stderr = io.StringIO("simulated encoder failure")

    def poll(self):
        return self.return_code

    def terminate(self):
        self.return_code = -15

    def kill(self):
        self.return_code = -9

    def wait(self):
        return self.return_code


class ConversionTests(unittest.TestCase):
    def exercise_conversion(self, *, return_code=0, cancel=False, verify_fails=False):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.mov"
            dest = Path(directory) / "converted" / "source_amadeus.mp4"
            source.write_bytes(b"original source")
            dest.parent.mkdir()
            dest.write_bytes(b"previous complete video")
            plan = compat.ConversionPlan(
                source_path=str(source),
                output_path=str(dest),
                first_frame=0,
                last_frame=3,
                target_frame_rate=Fraction(30, 1),
                tone_map=False,
                frame_mapping="exact",
            )
            metadata_path = Path(plan.metadata_path)
            metadata_path.write_text('{"old":true}', encoding="utf-8")
            stop = threading.Event()
            if cancel:
                stop.set()
            staged_paths = []

            def build_command(staged_plan):
                staged_paths.append(staged_plan.output_path)
                return ["ffmpeg", "-y", staged_plan.output_path]

            def start_process(command, **_kwargs):
                Path(command[-1]).write_bytes(b"new complete video")
                return FakeProcess(return_code=return_code)

            def verify(staged_plan):
                self.assertNotEqual(staged_plan.output_path, plan.output_path)
                self.assertEqual(Path(staged_plan.output_path).read_bytes(), b"new complete video")
                if verify_fails:
                    raise RuntimeError("frame seeking failed")
                return 4, []

            def metadata(_plan, **_kwargs):
                return {"converted_video": {"path": str(dest), "frames_written": 4}}

            with (
                patch.object(compat, "build_ffmpeg_command", side_effect=build_command),
                patch.object(compat.subprocess, "Popen", side_effect=start_process),
                patch.object(compat, "_verify_conversion", side_effect=verify),
                patch.object(compat, "conversion_metadata", side_effect=metadata),
            ):
                if cancel:
                    with self.assertRaises(compat.ConversionCancelled):
                        compat.run_conversion(plan, cancel=stop)
                elif return_code != 0 or verify_fails:
                    with self.assertRaises(RuntimeError):
                        compat.run_conversion(plan)
                else:
                    result = compat.run_conversion(plan)
                    self.assertEqual(result.output_path, str(dest))
                    self.assertEqual(result.frames_written, 4)

            self.assertEqual(len(staged_paths), 1)
            self.assertNotEqual(staged_paths[0], str(dest))
            self.assertTrue(staged_paths[0].endswith(".partial.mp4"))
            self.assertFalse(Path(staged_paths[0]).exists())
            self.assertEqual(source.read_bytes(), b"original source")
            if cancel or return_code != 0 or verify_fails:
                self.assertEqual(dest.read_bytes(), b"previous complete video")
                self.assertEqual(metadata_path.read_text(encoding="utf-8"), '{"old":true}')
            else:
                self.assertEqual(dest.read_bytes(), b"new complete video")
                self.assertEqual(
                    json.loads(metadata_path.read_text(encoding="utf-8"))["converted_video"]["frames_written"], 4
                )

    def test_success_publishes_verified_file(self):
        self.exercise_conversion()

    def test_cancel_preserves_previous_file_and_record(self):
        self.exercise_conversion(cancel=True)

    def test_encoder_failure_preserves_previous_file_and_record(self):
        self.exercise_conversion(return_code=1)

    def test_verification_failure_preserves_previous_file_and_record(self):
        self.exercise_conversion(verify_fails=True)


if __name__ == "__main__":
    unittest.main()
