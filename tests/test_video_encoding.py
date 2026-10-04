# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Video export contracts without a GPU, GUI, or scientific Python packages.

The shared detector is imported normally. Selected GUI and Create Video
definitions are compiled from their source AST so these tests exercise their
actual implementations without initializing Tk or importing tracking packages.
Real encoder availability and successful exports still require integration
tests against each platform's pinned FFmpeg and GPU driver.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import video_encoders
from tools import ffmpeg_runtime


def load_definitions(path, names, namespace, *, class_name=None):
    """Compile the named production definitions, preserving source locations."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source_nodes = tree.body
    if class_name is not None:
        source_nodes = next(
            node.body for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    definitions = [
        node for node in source_nodes
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    found = {node.name for node in definitions}
    if found != set(names):
        raise AssertionError(f"Missing definitions in {path}: {set(names) - found}")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def encoder_option(command, option):
    return command[command.index(option) + 1]


def load_video_encoder_imports(path, namespace):
    """Resolve production encoder imports to catch stale imported names."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "main.video_encoders"
    ]
    if not imports:
        raise AssertionError(f"Missing shared encoder import in {path}")
    exec(compile(ast.Module(body=imports, type_ignores=[]), str(path), "exec"), namespace)


class GpuSelectionTests(unittest.TestCase):
    def setUp(self):
        video_encoders.detect_gpu_video_encoder.cache_clear()
        self.addCleanup(video_encoders.detect_gpu_video_encoder.cache_clear)

    @contextlib.contextmanager
    def platform(self, platform_name, vendors=(), devices=(), nvidia=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(video_encoders, "os", SimpleNamespace(name="nt" if platform_name == "win32" else "posix")))
            stack.enter_context(patch.object(video_encoders, "sys", SimpleNamespace(platform=platform_name, stderr=io.StringIO())))
            stack.enter_context(patch.object(video_encoders, "_windows_gpu_vendors", return_value=vendors))
            stack.enter_context(patch.object(video_encoders, "_linux_gpu_devices", return_value=devices))
            stack.enter_context(patch.object(video_encoders, "query_nvidia_driver_info", return_value=nvidia))
            yield

    def test_windows_selects_only_detected_vendor(self):
        for vendor, encoder in (("10de", "h264_nvenc"), ("1002", "h264_amf"), ("8086", "h264_qsv")):
            with self.subTest(vendor=vendor), self.platform("win32", vendors=(vendor,)):
                self.assertEqual(video_encoders._gpu_encoder_candidates(), ((encoder, None),))

    def test_windows_nvidia_driver_and_pci_are_deduplicated(self):
        with self.platform("win32", vendors=("10de", "8086"), nvidia={"name": "RTX 3060"}):
            self.assertEqual(
                video_encoders._gpu_encoder_candidates(),
                (("h264_nvenc", None), ("h264_qsv", None)),
            )

    def test_linux_uses_nvenc_or_each_detected_vaapi_device(self):
        devices = (("8086", "/dev/dri/renderD130"), ("1002", "/dev/dri/renderD129"), ("10de", "/dev/dri/renderD128"))
        with self.platform("linux", devices=devices, nvidia={"name": "NVIDIA"}):
            self.assertEqual(
                video_encoders._gpu_encoder_candidates(),
                (("h264_nvenc", None), ("h264_vaapi", "/dev/dri/renderD129"), ("h264_vaapi", "/dev/dri/renderD130")),
            )

    def test_linux_unknown_vendor_is_not_probed(self):
        with self.platform("linux", devices=(("1234", "/dev/dri/renderD128"),)):
            self.assertEqual(video_encoders._gpu_encoder_candidates(), ())

    def test_macos_uses_videotoolbox(self):
        with self.platform("darwin"):
            self.assertEqual(video_encoders._gpu_encoder_candidates(), (("h264_videotoolbox", None),))

    def test_no_gpu_yields_no_probe(self):
        with self.platform("win32"), patch.object(video_encoders.subprocess, "run") as run:
            self.assertIsNone(video_encoders.detect_gpu_video_encoder("pinned-ffmpeg"))
            run.assert_not_called()

    def test_detected_nvidia_is_tested_with_pinned_ffmpeg_only(self):
        with patch.object(video_encoders, "_gpu_encoder_candidates", return_value=(("h264_nvenc", None),)), patch.object(
            video_encoders.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")
        ) as run:
            self.assertEqual(video_encoders.detect_gpu_video_encoder("pinned-ffmpeg"), ("h264_nvenc", None))
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[0], "pinned-ffmpeg")
        self.assertEqual(encoder_option(command, "-c:v"), "h264_nvenc")
        self.assertIn("lavfi", command)
        self.assertIn("-frames:v", command)
        self.assertNotIn("h264_qsv", command)
        self.assertNotIn("h264_amf", command)
        self.assertEqual(run.call_args.kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs["stderr"], subprocess.PIPE)
        self.assertGreater(run.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(run.call_args.kwargs["timeout"], 30)

    def test_failures_return_none_without_raising(self):
        failures = (
            SimpleNamespace(returncode=1, stderr="driver unavailable"),
            subprocess.TimeoutExpired("pinned-ffmpeg", 20),
            OSError("cannot execute FFmpeg"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                video_encoders.detect_gpu_video_encoder.cache_clear()
                run_mock = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
                with patch.object(video_encoders, "_gpu_encoder_candidates", return_value=(("h264_nvenc", None),)), patch.object(
                    video_encoders.subprocess, "run", run_mock
                ), contextlib.redirect_stderr(io.StringIO()):
                    self.assertIsNone(video_encoders.detect_gpu_video_encoder("pinned-ffmpeg"))

    def test_other_detected_gpu_can_succeed_after_first_failure(self):
        with patch.object(video_encoders, "_gpu_encoder_candidates", return_value=(("h264_nvenc", None), ("h264_qsv", None))), patch.object(
            video_encoders.subprocess, "run", side_effect=[SimpleNamespace(returncode=1, stderr="unavailable"), SimpleNamespace(returncode=0, stderr="")]
        ) as run, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(video_encoders.detect_gpu_video_encoder("pinned-ffmpeg"), ("h264_qsv", None))
        self.assertEqual([encoder_option(call.args[0], "-c:v") for call in run.call_args_list], ["h264_nvenc", "h264_qsv"])

    def test_validation_cache_separates_ffmpeg_and_quality_profile(self):
        with patch.object(video_encoders, "_gpu_encoder_candidates", return_value=(("h264_nvenc", None),)), patch.object(
            video_encoders.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")
        ) as run:
            for _ in range(2):
                video_encoders.detect_gpu_video_encoder("ffmpeg-a", profile="preprocess")
            video_encoders.detect_gpu_video_encoder("ffmpeg-b", profile="preprocess")
            video_encoders.detect_gpu_video_encoder("ffmpeg-a", profile="create_video")
        self.assertEqual(run.call_count, 3)
        self.assertEqual([encoder_option(call.args[0], "-cq:v") for call in run.call_args_list], ["12", "12", "23"])

    def test_vaapi_probe_uploads_to_the_selected_render_device(self):
        device = "/dev/dri/renderD129"
        with patch.object(video_encoders, "_gpu_encoder_candidates", return_value=(("h264_vaapi", device),)), patch.object(
            video_encoders.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")
        ) as run:
            self.assertEqual(video_encoders.detect_gpu_video_encoder("pinned-ffmpeg"), ("h264_vaapi", device))
        command = run.call_args.args[0]
        self.assertEqual(encoder_option(command, "-vaapi_device"), device)
        self.assertIn("format=nv12,hwupload", encoder_option(command, "-vf"))

    def test_videotoolbox_forbids_software_substitution(self):
        for profile in ("preprocess", "create_video"):
            with self.subTest(profile=profile):
                self.assertEqual(encoder_option(video_encoders.gpu_encoder_args("h264_videotoolbox", profile=profile), "-allow_sw"), "0")

    def test_cpu_encoder_is_rejected_by_gpu_argument_builder(self):
        with self.assertRaisesRegex(ValueError, "libx264"):
            video_encoders.gpu_encoder_args("libx264")


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.namespace = {
            "queue": queue, "sys": sys, "math": math, "OUTPUT_SPEED_MIN": 0.1,
            "ffmpeg_exe": Mock(return_value="pinned-ffmpeg"),
            "threading": SimpleNamespace(Thread=Mock(side_effect=lambda *, target, daemon: SimpleNamespace(start=target))),
        }
        source = ROOT / "gui" / "gui_preprocess.py"
        load_video_encoder_imports(source, self.namespace)
        self.namespace["detect_gpu_video_encoder"] = Mock()
        self.namespace["gpu_encoder_args"] = Mock(wraps=video_encoders.gpu_encoder_args)
        load_definitions(source, {"_video_encoder_args", "_clamp", "ExportCancelled"}, self.namespace)
        load_definitions(source, {
            "_start_gpu_encoder_detection", "_poll_gpu_encoder_detection", "_build_ffmpeg_command",
            "_expected_export_frame_count", "_export_worker",
        }, self.namespace, class_name="PreprocessApp")
        self.app = SimpleNamespace(
            gpu_encoder_queue=queue.Queue(), gpu_encoder_detection_complete=False,
            gpu_video_encoder=None, gpu_video_encoder_device=None, export_running=False,
            gpu_acceleration_var=SimpleNamespace(set=Mock()), gpu_acceleration_check=Mock(),
            _refresh_video_controls=Mock(), _schedule_crop_trimming_config_save=Mock(), after=Mock(),
        )
        self.app._poll_gpu_encoder_detection = lambda: self.namespace["_poll_gpu_encoder_detection"](self.app)

    def test_success_enables_gpu_acceleration_by_default(self):
        self.app.gpu_encoder_queue.put(("result", ("h264_nvenc", None)))
        self.app._poll_gpu_encoder_detection()
        self.assertTrue(self.app.gpu_encoder_detection_complete)
        self.assertEqual(self.app.gpu_video_encoder, "h264_nvenc")
        self.app.gpu_acceleration_var.set.assert_called_once_with(True)
        self.app.gpu_acceleration_check.configure.assert_called_once_with(text="GPU acceleration", state="normal")
        self.app._refresh_video_controls.assert_called_once()

    def test_validation_failure_keeps_cpu_export_controls_available(self):
        self.app.gpu_encoder_queue.put(("result", None))
        self.app._poll_gpu_encoder_detection()
        self.assertTrue(self.app.gpu_encoder_detection_complete)
        self.assertIsNone(self.app.gpu_video_encoder)
        self.app.gpu_acceleration_var.set.assert_called_once_with(False)
        self.app.gpu_acceleration_check.configure.assert_called_once_with(text="GPU acceleration", state="disabled")
        self.app._refresh_video_controls.assert_called_once()

    def test_detection_worker_uses_pinned_ffmpeg(self):
        self.namespace["detect_gpu_video_encoder"].return_value = ("h264_nvenc", None)
        self.namespace["_start_gpu_encoder_detection"](self.app)
        self.namespace["detect_gpu_video_encoder"].assert_called_once_with("pinned-ffmpeg")
        self.app._poll_gpu_encoder_detection()
        self.app.gpu_acceleration_var.set.assert_called_once_with(True)

    def test_detection_worker_exception_is_nonfatal(self):
        for callable_name in ("ffmpeg_exe", "detect_gpu_video_encoder"):
            with self.subTest(callable_name=callable_name):
                self.setUp()
                self.namespace[callable_name].side_effect = RuntimeError("runtime missing")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.namespace["_start_gpu_encoder_detection"](self.app)
                self.app._poll_gpu_encoder_detection()
                self.assertTrue(self.app.gpu_encoder_detection_complete)
                self.app.gpu_acceleration_var.set.assert_called_once_with(False)
                self.app._refresh_video_controls.assert_called_once()

    def test_poll_waits_without_blocking_when_detection_is_incomplete(self):
        self.app._poll_gpu_encoder_detection()
        self.assertFalse(self.app.gpu_encoder_detection_complete)
        self.app.after.assert_called_once_with(100, self.app._poll_gpu_encoder_detection)

    def test_cpu_export_never_passes_libx264_to_gpu_builder(self):
        self.namespace["gpu_encoder_args"].side_effect = AssertionError("CPU sent to GPU builder")
        job = SimpleNamespace(in_frame=0, out_frame=4, fps=30, output_speed=1, output_fps=30,
                              video_encoder="libx264", video_encoder_device=None, ffmpeg="pinned-ffmpeg", video_path="source.mp4")
        region = SimpleNamespace(rect=None, angle=0, brightness=0, contrast=1, color_mode="color", quarter_turns=0, output_path="output.mp4")
        command = self.namespace["_build_ffmpeg_command"](self.app, job, region)
        self.assertEqual(encoder_option(command, "-c:v"), "libx264")
        self.namespace["gpu_encoder_args"].assert_not_called()

    def test_vaapi_export_matches_probe_device_and_upload(self):
        job = SimpleNamespace(in_frame=0, out_frame=4, fps=30, output_speed=1, output_fps=30,
                              video_encoder="h264_vaapi", video_encoder_device="/dev/dri/renderD129", ffmpeg="pinned-ffmpeg", video_path="source.mp4")
        region = SimpleNamespace(rect=None, angle=0, brightness=0, contrast=1, color_mode="color", quarter_turns=0, output_path="output.mp4")
        command = self.namespace["_build_ffmpeg_command"](self.app, job, region)
        self.assertEqual(encoder_option(command, "-vaapi_device"), job.video_encoder_device)
        self.assertIn("format=nv12,hwupload", encoder_option(command, "-vf"))

    def test_export_worker_logs_cpu_and_gpu_labels_and_completes(self):
        for encoder, label in (("libx264", "CPU (libx264)"), ("h264_nvenc", "NVIDIA NVENC")):
            with self.subTest(encoder=encoder):
                reader = Mock()
                process = SimpleNamespace(stdout=io.StringIO("frame=5\nprogress=end\n"),
                                          stderr=io.StringIO(""), wait=Mock(return_value=0))
                self.namespace.update({
                    "VideoFrameReader": Mock(return_value=reader), "PROJECT_ROOT": ROOT,
                    "os": SimpleNamespace(name=os.name, makedirs=Mock()),
                    "time": SimpleNamespace(perf_counter=Mock(return_value=1.0)),
                    "subprocess": SimpleNamespace(Popen=Mock(return_value=process), PIPE=subprocess.PIPE,
                                                  CREATE_NO_WINDOW=getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                })
                self.app.export_queue = queue.Queue()
                self.app.export_cancel = SimpleNamespace(is_set=Mock(return_value=False))
                self.app.export_proc_lock = contextlib.nullcontext()
                self.app._last_progress_update = 0.0
                self.app._build_ffmpeg_command = lambda job, region: self.namespace["_build_ffmpeg_command"](self.app, job, region)
                self.app._expected_export_frame_count = lambda job: self.namespace["_expected_export_frame_count"](self.app, job)
                self.app._verify_export = Mock(return_value=[])
                self.app._delete_partial = Mock()
                region = SimpleNamespace(rect=None, angle=0, brightness=0, contrast=1, color_mode="color",
                                         quarter_turns=0, output_path="output.mp4", display_name="Full frame")
                job = SimpleNamespace(in_frame=0, out_frame=4, fps=30, output_speed=1, output_fps=30,
                                      video_encoder=encoder, video_encoder_device=None, ffmpeg="pinned-ffmpeg",
                                      video_path="source.mp4", output_folder="exports", regions=(region,))
                self.namespace["_export_worker"](self.app, job)
                events = []
                while not self.app.export_queue.empty():
                    events.append(self.app.export_queue.get_nowait())
                self.assertIn(("log", f"Video encoder: {label}"), events)
                self.assertEqual(events[-1], ("done", ("output.mp4",), ()))
                reader.close.assert_called_once()
                self.namespace["VideoFrameReader"].assert_called_once_with("source.mp4")
                command = self.namespace["subprocess"].Popen.call_args.args[0]
                self.assertEqual(encoder_option(command, "-c:v"), encoder)
                self.app._verify_export.assert_called_once_with(job, region, reader)
                self.app._delete_partial.assert_not_called()
                self.assertIsNone(self.app.export_proc)


class CreateVideoTests(unittest.TestCase):
    def namespaces(self):
        for relative in ("main/create_video.py", "main/without_direction_estimation/create_video.py"):
            namespace = {
                "os": os, "ffmpeg_executable": Mock(return_value="pinned-ffmpeg"),
                "subprocess": SimpleNamespace(run=Mock(), Popen=Mock(), PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL),
            }
            load_video_encoder_imports(ROOT / relative, namespace)
            namespace["detect_gpu_video_encoder"] = Mock(return_value=None)
            namespace["gpu_encoder_args"] = Mock(wraps=video_encoders.gpu_encoder_args)
            load_definitions(ROOT / relative, {
                "resolve_video_acceleration", "_ffmpeg_path", "_gpu_video_encoder", "_selected_video_encoder",
                "make_video_from_image_sequence", "_cpu_encoder_args", "FfmpegRawVideoWriter",
            }, namespace)
            yield relative, namespace

    def test_cpu_mode_skips_gpu_detection(self):
        for path, namespace in self.namespaces():
            with self.subTest(path=path):
                self.assertEqual(namespace["_selected_video_encoder"]("cpu", "h264"), ("libx264", None))
                namespace["detect_gpu_video_encoder"].assert_not_called()

    def test_auto_and_gpu_success_share_validation_profile(self):
        for path, namespace in self.namespaces():
            for mode in ("auto", "gpu"):
                with self.subTest(path=path, mode=mode):
                    detector = namespace["detect_gpu_video_encoder"]
                    detector.reset_mock()
                    detector.return_value = ("h264_nvenc", None)
                    self.assertEqual(namespace["_selected_video_encoder"](mode, "h264"), ("h264_nvenc", None))
                    detector.assert_called_once_with("pinned-ffmpeg", profile="create_video")

    def test_failed_validation_falls_back_and_explicit_gpu_warns(self):
        for path, namespace in self.namespaces():
            for mode in ("auto", "gpu"):
                with self.subTest(path=path, mode=mode), contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(namespace["_selected_video_encoder"](mode, "auto"), ("libx264", None))
                    if mode == "gpu":
                        self.assertIn("[WARN]", output.getvalue())
                        self.assertIn("libx264", output.getvalue())
                    else:
                        self.assertEqual(output.getvalue(), "")

    def test_mp4v_uses_cpu_and_skips_h264_probe(self):
        for path, namespace in self.namespaces():
            for mode in ("cpu", "auto", "gpu"):
                with self.subTest(path=path, mode=mode), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(namespace["_selected_video_encoder"](mode, "mp4v"), ("mpeg4", None))
                    namespace["detect_gpu_video_encoder"].assert_not_called()

    def test_unavailable_ffmpeg_is_not_passed_to_detector(self):
        for path, namespace in self.namespaces():
            with self.subTest(path=path):
                namespace["ffmpeg_executable"].side_effect = RuntimeError("missing")
                self.assertIsNone(namespace["_gpu_video_encoder"]())
                namespace["detect_gpu_video_encoder"].assert_not_called()

    def test_cpu_builders_reject_gpu_encoders(self):
        for path, namespace in self.namespaces():
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "h264_nvenc"):
                namespace["_cpu_encoder_args"]("h264_nvenc")

    def test_image_sequence_cpu_fallback_never_calls_gpu_builder(self):
        for path, namespace in self.namespaces():
            with self.subTest(path=path), contextlib.redirect_stdout(io.StringIO()):
                namespace["gpu_encoder_args"].side_effect = AssertionError("CPU sent to GPU builder")
                namespace["make_video_from_image_sequence"]("frames", 0, 30, "output.mp4", "png", "auto")
                command = namespace["subprocess"].run.call_args.args[0]
                self.assertEqual(command[0], "pinned-ffmpeg")
                self.assertEqual(encoder_option(command, "-c:v"), "libx264")
                namespace["gpu_encoder_args"].assert_not_called()

    def test_raw_video_cpu_encoders_never_call_gpu_builder(self):
        for path, namespace in self.namespaces():
            for encoder in ("libx264", "mpeg4"):
                with self.subTest(path=path, encoder=encoder):
                    namespace["gpu_encoder_args"].side_effect = AssertionError("CPU sent to GPU builder")
                    namespace["FfmpegRawVideoWriter"]("output.mp4", 30, (48, 64, 3), encoder)
                    command = namespace["subprocess"].Popen.call_args.args[0]
                    self.assertEqual(encoder_option(command, "-c:v"), encoder)
                    namespace["gpu_encoder_args"].assert_not_called()

    def test_vaapi_image_sequence_and_raw_writer_upload(self):
        for path, namespace in self.namespaces():
            with self.subTest(path=path), contextlib.redirect_stdout(io.StringIO()):
                namespace["detect_gpu_video_encoder"].return_value = ("h264_vaapi", "/dev/dri/renderD129")
                namespace["make_video_from_image_sequence"]("frames", 0, 30, "output.mp4", "png", "auto")
                namespace["FfmpegRawVideoWriter"]("output.mp4", 30, (48, 64, 3), "h264_vaapi", "/dev/dri/renderD129")
                for call in (namespace["subprocess"].run.call_args, namespace["subprocess"].Popen.call_args):
                    command = call.args[0]
                    self.assertEqual(encoder_option(command, "-vaapi_device"), "/dev/dri/renderD129")
                    self.assertIn("format=nv12,hwupload", encoder_option(command, "-vf"))
                self.assertEqual(namespace["gpu_encoder_args"].call_count, 2)
                namespace["gpu_encoder_args"].assert_called_with("h264_vaapi", profile="create_video")


class PinnedFFmpegInventoryTests(unittest.TestCase):
    def test_cpu_only_inventory_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "ffmpeg"
            executable.touch()
            with patch.object(ffmpeg_runtime.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=" V..... libx264 H.264\n", stderr="")):
                self.assertIn("libx264", ffmpeg_runtime._run_encoder_inventory(executable))

    def test_gpu_without_libx264_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "ffmpeg"
            executable.touch()
            with patch.object(ffmpeg_runtime.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=" V..... h264_nvenc NVIDIA\n", stderr="")):
                with self.assertRaisesRegex(RuntimeError, "libx264"):
                    ffmpeg_runtime._run_encoder_inventory(executable)


if __name__ == "__main__":
    unittest.main()
