# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Cover the video compatibility layer without needing a GUI.

The parsing, verdict and planning tests run anywhere. The conversion tests need
OpenCV and the pinned FFmpeg build, and skip themselves when either is missing.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import video_compat as vc  # noqa: E402


# Recorded "ffmpeg -i <file>" reports. Keeping them verbatim means the parser is
# tested against what FFmpeg actually prints, not against an idea of it.
HEVC_HDR_DUMP = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'media/hdr.mov':
  Metadata:
    major_brand     : qt
    encoder         : Lavf61.1.100
  Duration: 00:00:06.00, start: 0.000000, bitrate: 693 kb/s
  Stream #0:0[0x1]: Video: hevc (Main 10) (hvc1 / 0x31637668), yuv420p10le(tv, bt2020nc/bt2020/smpte2084, progressive), 640x480 [SAR 1:1 DAR 4:3], 685 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Metadata:
      handler_name    : VideoHandler
At least one output file must be specified
"""

MPEGTS_DUMP = """Input #0, mpegts, from 'media/clip.MTS':
  Duration: 00:00:06.00, start: 1.466667, bitrate: 1019 kb/s
  Program 1
  Stream #0:0[0x100]: Video: h264 (High) ([27][0][0][0] / 0x001B), yuv420p(progressive), 1920x1080 [SAR 1:1 DAR 16:9], 29.97 fps, 59.94 tbr, 90k tbn
  Stream #0:1[0x101]: Audio: ac3 ([129][0][0][0] / 0x0081), 48000 Hz, stereo, fltp, 256 kb/s
At least one output file must be specified
"""

ROTATED_DUMP = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'media/rot.mov':
  Duration: 00:00:06.00, start: 0.000000, bitrate: 960 kb/s
  Stream #0:0[0x1]: Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 640x480 [SAR 1:1 DAR 4:3], 957 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
    Side data:
      displaymatrix: rotation of -90.00 degrees
At least one output file must be specified
"""

PLAIN_DUMP = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'media/plain.mp4':
  Duration: 00:00:06.00, start: 0.000000, bitrate: 961 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 640x480 [SAR 1:1 DAR 4:3], 957 kb/s, 30 fps, 30 tbr, 15360 tbn (default)
At least one output file must be specified
"""

NO_VIDEO_DUMP = """Input #0, mp3, from 'media/sound.mp3':
  Duration: 00:00:06.00, start: 0.000000, bitrate: 128 kb/s
  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 128 kb/s
At least one output file must be specified
"""


def make_probe(**overrides) -> vc.OpenCvProbe:
    """An OpenCV probe of a plain, well-behaved 8-bit H.264 MP4."""
    defaults = dict(
        opened=True,
        fourcc="avc1",
        width=1920,
        height=1080,
        fps=30.0,
        reported_frame_count=900,
        seekable_frame_count=900,
        first_frame_decoded=True,
        random_seek_ok=True,
        measured_fps=30.0,
        irregular_interval_fraction=0.0,
        variable_frame_rate=False,
    )
    defaults.update(overrides)
    return vc.OpenCvProbe(**defaults)


def make_source(**overrides) -> vc.SourceVideoInfo:
    defaults = dict(
        path="/videos/plain.mp4",
        container="mov,mp4,m4a,3gp,3g2,mj2",
        codec="h264",
        profile="High",
        pix_fmt="yuv420p",
        bit_depth=8,
        width=1920,
        height=1080,
        nominal_fps=30.0,
        tbr=30.0,
        duration_seconds=30.0,
        rotation_degrees=0,
    )
    defaults.update(overrides)
    return vc.SourceVideoInfo(**defaults)


def assessment_for(path, source, probe) -> vc.VideoAssessment:
    """Run the real verdict logic over stubbed probes."""
    real_source, real_opencv, real_ffmpeg = vc.probe_source, vc.probe_opencv, vc.ffmpeg_is_available
    vc.probe_source = lambda _path: source
    vc.probe_opencv = lambda _path: probe
    vc.ffmpeg_is_available = lambda: True
    try:
        return vc.assess_video(path)
    finally:
        vc.probe_source, vc.probe_opencv, vc.ffmpeg_is_available = real_source, real_opencv, real_ffmpeg


class AcceptedInputTests(unittest.TestCase):
    def test_every_requested_container_is_selectable(self):
        for suffix in (".mp4", ".mov", ".avi", ".mts", ".m2ts", ".mkv", ".mpg", ".mpeg", ".webm"):
            with self.subTest(suffix=suffix):
                self.assertIn(suffix, vc.SUPPORTED_VIDEO_SUFFIXES)
                self.assertIn(suffix, vc.VIDEO_DROP_SUFFIXES)

    def test_file_dialog_pattern_lists_every_suffix(self):
        pattern = vc.VIDEO_FILETYPES[0][1]
        for suffix in vc.SUPPORTED_VIDEO_SUFFIXES:
            self.assertIn(f"*{suffix}", pattern)
        self.assertEqual(vc.VIDEO_FILETYPES[-1], ("All files", "*.*"))


class StreamDumpParsingTests(unittest.TestCase):
    def test_hevc_hdr_10bit(self):
        parsed = vc.parse_ffmpeg_stream_dump(HEVC_HDR_DUMP)
        self.assertEqual(parsed["codec"], "hevc")
        self.assertEqual(parsed["profile"], "Main 10")
        self.assertEqual(parsed["pix_fmt"], "yuv420p10le")
        self.assertEqual(parsed["bit_depth"], 10)
        self.assertEqual((parsed["width"], parsed["height"]), (640, 480))
        self.assertEqual(parsed["color_transfer"], "smpte2084")
        # "bt2020nc/bt2020/smpte2084" is colourspace/primaries/transfer, so the
        # primaries are bt2020; bt2020nc is the matrix.
        self.assertEqual(parsed["color_primaries"], "bt2020")
        self.assertAlmostEqual(parsed["duration_seconds"], 6.0)
        self.assertEqual(parsed["nominal_fps"], 30.0)

    def test_mpegts_with_audio(self):
        parsed = vc.parse_ffmpeg_stream_dump(MPEGTS_DUMP)
        self.assertEqual(parsed["container"], "mpegts")
        self.assertEqual(parsed["codec"], "h264")
        self.assertEqual(parsed["pix_fmt"], "yuv420p")
        self.assertEqual((parsed["width"], parsed["height"]), (1920, 1080))
        self.assertEqual(parsed["nominal_fps"], 29.97)
        self.assertEqual(parsed["tbr"], 59.94)
        self.assertEqual(parsed["audio_streams"], 1)

    def test_codec_tag_hex_is_not_read_as_a_resolution_or_pixel_format(self):
        parsed = vc.parse_ffmpeg_stream_dump(PLAIN_DUMP)
        self.assertEqual((parsed["width"], parsed["height"]), (640, 480))
        self.assertEqual(parsed["pix_fmt"], "yuv420p")
        self.assertEqual(parsed["bit_depth"], 8)
        self.assertEqual(parsed["color_transfer"], "bt709")

    def test_display_matrix_becomes_a_clockwise_angle(self):
        # FFmpeg prints the display-matrix angle; a portrait phone video shows
        # "-90.00 degrees" and is played rotated 90 degrees clockwise.
        parsed = vc.parse_ffmpeg_stream_dump(ROTATED_DUMP)
        self.assertEqual(parsed["rotation_degrees"], 90)

    def test_file_without_a_video_stream_is_reported(self):
        parsed = vc.parse_ffmpeg_stream_dump(NO_VIDEO_DUMP)
        self.assertIn("error", parsed)
        self.assertNotIn("codec", parsed)


class PixelFormatTests(unittest.TestCase):
    def test_bit_depth(self):
        cases = {
            "yuv420p": 8,
            "yuvj420p": 8,
            "yuv420p10le": 10,
            "yuv444p12le": 12,
            "p010le": 10,
            "p016le": 16,
            "p210le": 10,
            "gray": 8,
            "gray16le": 16,
            "": 0,
        }
        for pix_fmt, expected in cases.items():
            with self.subTest(pix_fmt=pix_fmt):
                self.assertEqual(vc._bit_depth_from_pix_fmt(pix_fmt), expected)


class FrameRateTests(unittest.TestCase):
    def test_broadcast_rates_stay_exact(self):
        # OpenCV derives CAP_PROP_FPS from the container's own rational, so these
        # are the values that actually reach rational_frame_rate().
        for numerator, denominator in ((30000, 1001), (24000, 1001), (60000, 1001),
                                       (25, 1), (30, 1), (50, 1), (60, 1)):
            with self.subTest(rate=f"{numerator}/{denominator}"):
                self.assertEqual(
                    vc.rational_frame_rate(numerator / denominator),
                    Fraction(numerator, denominator),
                )

    def test_a_rounded_rate_is_kept_as_the_exact_value_given(self):
        # 59.94 is not 60000/1001; guessing that it was meant to be would change
        # the recording's time base behind the user's back.
        self.assertEqual(vc.rational_frame_rate(59.94), Fraction(2997, 50))
        self.assertEqual(float(vc.rational_frame_rate(59.94)), 59.94)

    def test_zero_or_negative_is_rejected(self):
        for value in (0.0, -1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    vc.rational_frame_rate(value)


class FrameRangeValidationTests(unittest.TestCase):
    def test_accepts_a_range_inside_the_video(self):
        self.assertEqual(vc.validate_frame_range(0, 899, 900), "")
        self.assertEqual(vc.validate_frame_range(10, 10, 900), "")

    def test_rejects_impossible_ranges(self):
        self.assertIn("negative", vc.validate_frame_range(-1, 100, 900))
        self.assertIn("exceed", vc.validate_frame_range(0, 900, 900))
        self.assertIn("after", vc.validate_frame_range(500, 100, 900))
        self.assertIn("whole numbers", vc.validate_frame_range("a", 100, 900))
        self.assertIn("No usable frames", vc.validate_frame_range(0, 0, 0))


class AssessmentTests(unittest.TestCase):
    def test_plain_h264_mp4_is_left_alone(self):
        result = assessment_for("/videos/plain.mp4", make_source(), make_probe())
        self.assertEqual(result.status, vc.STATUS_OK)
        self.assertFalse(result.needs_dialog)
        self.assertEqual(result.reasons, ())

    def test_a_tiny_unreachable_tail_is_a_note_not_a_conversion(self):
        # video_frame_count.py already clamps a couple of tail frames; that must
        # not start asking the user about conversions.
        result = assessment_for(
            "/videos/plain.mp4", make_source(), make_probe(seekable_frame_count=899)
        )
        self.assertEqual(result.status, vc.STATUS_OK)
        self.assertTrue(any("excluded" in note for note in result.notes))

    def test_a_large_unreachable_tail_requires_conversion(self):
        result = assessment_for(
            "/videos/plain.mp4", make_source(), make_probe(seekable_frame_count=700)
        )
        self.assertEqual(result.status, vc.STATUS_REQUIRED)

    def test_failed_random_seek_requires_conversion(self):
        result = assessment_for(
            "/videos/clip.mts",
            make_source(container="mpegts", path="/videos/clip.mts"),
            make_probe(random_seek_ok=False, seekable_frame_count=0),
        )
        self.assertEqual(result.status, vc.STATUS_REQUIRED)
        self.assertTrue(any("Seeking" in reason for reason in result.reasons))
        # The container's own count is what a frame range can still span.
        self.assertEqual(result.usable_frame_count, 900)

    def test_undecodable_file_requires_conversion(self):
        result = assessment_for(
            "/videos/odd.mkv",
            make_source(container="matroska", path="/videos/odd.mkv"),
            vc.OpenCvProbe(error="OpenCV could not open the file."),
        )
        self.assertEqual(result.status, vc.STATUS_REQUIRED)

    def test_unusual_container_is_recommended_even_when_it_decodes(self):
        result = assessment_for(
            "/videos/clip.m2ts", make_source(path="/videos/clip.m2ts"), make_probe()
        )
        self.assertEqual(result.status, vc.STATUS_RECOMMENDED)
        self.assertTrue(any("container" in reason for reason in result.reasons))

    def test_hevc_hdr_ten_bit_is_recommended(self):
        result = assessment_for(
            "/videos/iphone.mov",
            make_source(
                path="/videos/iphone.mov",
                codec="hevc",
                profile="Main 10",
                pix_fmt="yuv420p10le",
                bit_depth=10,
                color_transfer="smpte2084",
                color_primaries="bt2020nc",
            ),
            make_probe(fourcc="hvc1"),
        )
        self.assertEqual(result.status, vc.STATUS_RECOMMENDED)
        self.assertTrue(result.is_hdr)
        joined = " ".join(result.reasons)
        self.assertIn("hevc", joined)
        self.assertIn("10-bit", joined)
        self.assertIn("HDR", joined)

    def test_variable_frame_rate_is_recommended(self):
        result = assessment_for(
            "/videos/iphone.mp4",
            make_source(path="/videos/iphone.mp4"),
            make_probe(variable_frame_rate=True, irregular_interval_fraction=0.4),
        )
        self.assertEqual(result.status, vc.STATUS_RECOMMENDED)
        self.assertTrue(result.is_variable_frame_rate)

    def test_rotation_metadata_is_recommended(self):
        result = assessment_for(
            "/videos/portrait.mov",
            make_source(path="/videos/portrait.mov", rotation_degrees=90),
            make_probe(),
        )
        self.assertEqual(result.status, vc.STATUS_RECOMMENDED)

    def test_summary_lists_the_facts_the_dialog_shows(self):
        result = assessment_for("/videos/plain.mp4", make_source(), make_probe())
        text = "\n".join(result.summary_lines())
        for expected in ("Container:", "Codec:", "Resolution:", "Pixel format:",
                         "Frame rate:", "Frame rate mode:", "Colour:",
                         "Rotation metadata:", "Frames:", "Random frame access:"):
            self.assertIn(expected, text)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.assessment = assessment_for(
            "/videos/clip.mts",
            make_source(path="/videos/clip.mts", container="mpegts"),
            make_probe(),
        )

    def test_default_range_is_the_whole_video(self):
        plan = vc.build_plan(self.assessment)
        self.assertEqual((plan.first_frame, plan.last_frame), (0, 899))
        self.assertEqual(plan.frame_count, 900)
        self.assertEqual(plan.frame_mapping, "exact")
        self.assertEqual(plan.target_frame_rate, Fraction(30, 1))

    def test_output_goes_to_the_session_directory_and_never_over_the_source(self):
        plan = vc.build_plan(self.assessment)
        self.assertEqual(
            Path(plan.output_path).parent,
            Path("/videos/amadeus_clip") / vc.CONVERTED_DIR_NAME,
        )
        self.assertTrue(plan.output_path.endswith(".mp4"))
        self.assertNotEqual(
            os.path.normcase(plan.output_path), os.path.normcase(plan.source_path)
        )

    def test_a_partial_range_gets_its_own_file_name(self):
        whole = vc.build_plan(self.assessment)
        part = vc.build_plan(self.assessment, 100, 199)
        self.assertNotEqual(whole.output_path, part.output_path)
        self.assertIn("f100-199", Path(part.output_path).name)
        self.assertEqual(part.frame_count, 100)

    def test_invalid_ranges_are_refused(self):
        with self.assertRaises(ValueError):
            vc.build_plan(self.assessment, 500, 100)
        with self.assertRaises(ValueError):
            vc.build_plan(self.assessment, 0, 900)

    def test_filters_trim_by_frame_and_force_a_constant_rate(self):
        filters = vc.build_video_filters(vc.build_plan(self.assessment, 100, 199))
        self.assertEqual(filters[0], "trim=start_frame=100:end_frame=200")
        self.assertIn("setpts=PTS-STARTPTS", filters)
        self.assertIn("fps=30/1", filters)
        self.assertIn(f"format={vc.ANALYSIS_PIX_FMT}", filters)
        # select= would leave the source's end-of-stream in place and the fps
        # filter would then pad the tail with duplicates of the last frame.
        self.assertFalse(any(item.startswith("select=") for item in filters))

    def test_hdr_plans_a_tone_mapping_chain_and_records_it(self):
        hdr = assessment_for(
            "/videos/hdr.mov",
            make_source(path="/videos/hdr.mov", codec="hevc", bit_depth=10,
                        color_transfer="smpte2084", color_primaries="bt2020nc"),
            make_probe(fourcc="hvc1"),
        )
        real = vc.can_tone_map
        vc.can_tone_map = lambda: True
        try:
            plan = vc.build_plan(hdr)
        finally:
            vc.can_tone_map = real
        self.assertTrue(plan.tone_map)
        chain = ",".join(vc.build_video_filters(plan))
        self.assertIn("zscale=t=linear", chain)
        self.assertIn("tonemap=tonemap=hable:desat=0", chain)
        self.assertTrue(any("tone mapped" in note for note in plan.notes))

    def test_hdr_without_the_filters_says_so_instead_of_pretending(self):
        hdr = assessment_for(
            "/videos/hdr.mov",
            make_source(path="/videos/hdr.mov", codec="hevc", bit_depth=10,
                        color_transfer="smpte2084", color_primaries="bt2020nc"),
            make_probe(fourcc="hvc1"),
        )
        real = vc.can_tone_map
        vc.can_tone_map = lambda: False
        try:
            plan = vc.build_plan(hdr)
        finally:
            vc.can_tone_map = real
        self.assertFalse(plan.tone_map)
        self.assertTrue(any("no zscale/tonemap" in note for note in plan.notes))

    def test_variable_frame_rate_plans_a_resample_at_the_dominant_rate(self):
        variable = assessment_for(
            "/videos/vfr.mp4",
            make_source(path="/videos/vfr.mp4"),
            make_probe(variable_frame_rate=True, fps=21.3, measured_fps=30.0),
        )
        plan = vc.build_plan(variable)
        self.assertEqual(plan.frame_mapping, "resampled")
        self.assertEqual(plan.target_frame_rate, Fraction(30, 1))

    def test_the_command_keeps_the_source_read_only_and_pins_the_encoder(self):
        plan = vc.build_plan(self.assessment, 10, 19)
        try:
            command = vc.build_ffmpeg_command(plan)
        except vc.FfmpegUnavailableError:
            self.skipTest("the pinned FFmpeg build is not installed")
        self.assertEqual(command[command.index("-i") + 1], plan.source_path)
        self.assertEqual(command[-1], plan.output_path)
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-crf") + 1], str(vc.ANALYSIS_CRF))
        self.assertEqual(command[command.index("-pix_fmt") + 1], vc.ANALYSIS_PIX_FMT)
        self.assertEqual(command[command.index("-g") + 1], str(vc.ANALYSIS_GOP))
        self.assertIn("-an", command)


class FrameMappingTests(unittest.TestCase):
    def test_converted_frame_zero_maps_to_the_first_selected_source_frame(self):
        metadata = {"frame_mapping": {"converted_frame_0_source_frame": 250}}
        self.assertEqual(vc.source_frame_for_converted_frame(metadata, 0), 250)
        self.assertEqual(vc.source_frame_for_converted_frame(metadata, 40), 290)

    def test_a_whole_video_conversion_maps_one_to_one(self):
        metadata = {"frame_mapping": {"converted_frame_0_source_frame": 0}}
        self.assertEqual(vc.source_frame_for_converted_frame(metadata, 123), 123)

    def test_metadata_records_the_source_the_range_and_the_conditions(self):
        assessment = assessment_for(
            "/videos/clip.mts",
            make_source(path="/videos/clip.mts", container="mpegts", codec="h264"),
            make_probe(),
        )
        plan = vc.build_plan(assessment, 100, 199)
        real_version, real_exe = vc.ffmpeg_version, vc.ffmpeg_executable
        vc.ffmpeg_version = lambda: "ffmpeg version 7.0.2-static"
        vc.ffmpeg_executable = lambda: "/pinned/ffmpeg"
        try:
            metadata = vc.conversion_metadata(plan, command=["ffmpeg"], frames_written=100)
        finally:
            vc.ffmpeg_version, vc.ffmpeg_executable = real_version, real_exe

        self.assertEqual(metadata["source_video"]["path"], plan.source_path)
        self.assertTrue(metadata["source_video"]["unmodified"])
        self.assertEqual(metadata["source_video"]["codec"], "h264")
        self.assertEqual(metadata["frame_range"]["source_first_frame"], 100)
        self.assertEqual(metadata["frame_range"]["source_last_frame"], 199)
        self.assertEqual(metadata["frame_mapping"]["converted_frame_0_source_frame"], 100)
        self.assertTrue(metadata["frame_mapping"]["exact"])
        self.assertEqual(metadata["converted_video"]["frame_rate_mode"], "constant")
        self.assertEqual(metadata["converted_video"]["crf"], vc.ANALYSIS_CRF)
        self.assertEqual(metadata["ffmpeg_pinned_requirement"], vc.PINNED_FFMPEG_REQUIREMENT)
        # json round-trip: the record has to survive being written to disk.
        self.assertEqual(json.loads(json.dumps(metadata))["frame_range"],
                         metadata["frame_range"])

    def test_config_record_summarises_the_conversion_for_a_gui_config(self):
        metadata = {
            "source_video": {"path": "/v/a.MTS", "codec": "h264", "container": "mpegts",
                             "nominal_fps": 29.97, "bit_depth": 8, "hdr": False},
            "converted_video": {"path": "/v/amadeus_a/converted/a_amadeus.mp4",
                                "fps": 29.97, "codec": "libx264"},
            "frame_range": {"source_first_frame": 10, "source_last_frame": 99},
            "frame_mapping": {"formula": "source_frame = converted_frame + 10", "exact": True},
            "conversion_notes": ["note"],
        }
        record = vc.config_conversion_record(metadata, "/v/amadeus_a/converted/a.json")
        self.assertEqual(record["source_video_path"], "/v/a.MTS")
        self.assertEqual(record["source_first_frame"], 10)
        self.assertEqual(record["conversion_metadata_path"], "/v/amadeus_a/converted/a.json")
        self.assertEqual(vc.config_conversion_record(None), {})


class FfmpegResolutionTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get(vc.FFMPEG_OVERRIDE_ENV_VAR)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(vc.FFMPEG_OVERRIDE_ENV_VAR, None)
        else:
            os.environ[vc.FFMPEG_OVERRIDE_ENV_VAR] = self._saved

    def test_the_pinned_binary_is_used_rather_than_one_on_PATH(self):
        os.environ.pop(vc.FFMPEG_OVERRIDE_ENV_VAR, None)
        try:
            resolved = vc.ffmpeg_executable()
        except vc.FfmpegUnavailableError:
            self.skipTest("the pinned FFmpeg build is not installed")
        self.assertIn("imageio_ffmpeg", resolved.replace(os.sep, "/"))
        on_path = shutil.which("ffmpeg")
        if on_path:
            self.assertNotEqual(os.path.normcase(resolved), os.path.normcase(on_path))

    def test_the_environment_override_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "ffmpeg")
            Path(fake).write_text("", encoding="utf-8")
            os.environ[vc.FFMPEG_OVERRIDE_ENV_VAR] = fake
            self.assertEqual(vc.ffmpeg_executable(), fake)

    def test_a_missing_override_fails_with_advice_not_a_bare_error(self):
        os.environ[vc.FFMPEG_OVERRIDE_ENV_VAR] = "/nowhere/ffmpeg"
        with self.assertRaises(vc.FfmpegUnavailableError) as caught:
            vc.ffmpeg_executable()
        message = str(caught.exception)
        self.assertIn(vc.PINNED_FFMPEG_REQUIREMENT, message)
        self.assertIn("launcher", message)
        self.assertIn("/nowhere/ffmpeg", message)


def _has_opencv() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


def _has_ffmpeg() -> bool:
    try:
        vc.ffmpeg_executable()
    except vc.FfmpegUnavailableError:
        return False
    return True


@unittest.skipUnless(_has_opencv() and _has_ffmpeg(), "OpenCV and the pinned FFmpeg are required")
class ConversionTests(unittest.TestCase):
    """End-to-end: build a video, convert it, and check what came out."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="amadeus-video-compat-")
        cls.source = os.path.join(cls.tmp, "source.mp4")
        completed = subprocess.run(
            [
                vc.ffmpeg_executable(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", cls.source,
            ],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0 or not os.path.isfile(cls.source):
            raise unittest.SkipTest(f"could not build a test video: {completed.stderr}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_a_plain_mp4_needs_no_conversion(self):
        assessment = vc.assess_video(self.source)
        self.assertEqual(assessment.status, vc.STATUS_OK)
        self.assertEqual(assessment.usable_frame_count, 120)

    def test_converting_a_frame_range_keeps_the_source_untouched(self):
        import cv2
        import numpy as np

        before = Path(self.source).read_bytes()
        assessment = vc.assess_video(self.source)
        plan = vc.build_plan(assessment, 30, 59)
        seen = []
        result = vc.run_conversion(plan, progress=lambda fraction, frames: seen.append(frames))

        self.assertEqual(Path(self.source).read_bytes(), before, "the source video was modified")
        self.assertTrue(os.path.isfile(result.output_path))
        self.assertTrue(seen, "no progress was reported")

        converted = vc.assess_video(result.output_path)
        self.assertEqual(converted.status, vc.STATUS_OK)
        self.assertEqual(converted.usable_frame_count, 30)
        self.assertEqual(converted.opencv.width, 320)
        self.assertEqual(converted.opencv.height, 240)
        self.assertAlmostEqual(converted.opencv.fps, 30.0, places=3)

        metadata = vc.read_conversion_metadata(result.output_path)
        self.assertEqual(metadata["frame_mapping"]["converted_frame_0_source_frame"], 30)

        def frame_at(path, index):
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = cap.read()
            cap.release()
            self.assertTrue(ok, f"frame {index} of {path} could not be read")
            return image.astype(np.int16)

        for converted_index in (0, 15, 29):
            source_index = vc.source_frame_for_converted_frame(metadata, converted_index)
            matched = np.mean(np.abs(frame_at(result.output_path, converted_index)
                                     - frame_at(self.source, source_index)))
            mismatched = np.mean(np.abs(frame_at(result.output_path, converted_index)
                                        - frame_at(self.source, source_index + 20)))
            self.assertLess(matched, 3.0, "converted frame does not match its source frame")
            self.assertGreater(mismatched, matched * 3,
                               "the frame mapping is not distinguishable from a wrong one")

    def test_an_existing_conversion_is_found_again(self):
        assessment = vc.assess_video(self.source)
        plan = vc.build_plan(assessment, 0, 9)
        vc.run_conversion(plan)
        found = vc.find_existing_conversions(self.source)
        self.assertTrue(found)
        self.assertTrue(
            any(item["converted_video"]["path"] == plan.output_path for item in found)
        )


if __name__ == "__main__":
    unittest.main()
