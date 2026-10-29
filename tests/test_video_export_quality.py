# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Check common, high-quality FFmpeg exports without source bitrate limits."""

import shutil

import cv2
import numpy as np
import pytest

from main import video_encoders


@pytest.mark.parametrize(
    ("quality", "qp", "videotoolbox", "mpeg4"),
    [
        ("standard", "23", "70", "3"),
        ("high", "18", "85", "2"),
    ],
)
def test_quality_modes_for_cpu_and_gpu(quality, qp, videotoolbox, mpeg4):
    cpu = video_encoders.quality_encoder_args("libx264", quality=quality)
    assert cpu[cpu.index("-crf") + 1] == qp
    assert cpu[cpu.index("-preset") + 1] == "medium"
    legacy = video_encoders.quality_encoder_args("mpeg4", quality=quality)
    assert legacy[legacy.index("-q:v") + 1] == mpeg4
    for encoder, option in (
        ("h264_nvenc", "-cq:v"),
        ("h264_qsv", "-global_quality:v"),
        ("h264_amf", "-qp_i"),
        ("h264_vaapi", "-qp"),
    ):
        args = video_encoders.quality_encoder_args(encoder, quality=quality)
        assert args[args.index(option) + 1] == qp
        assert "-maxrate:v" not in args
        assert "-bufsize:v" not in args
    videotoolbox_args = video_encoders.quality_encoder_args(
        "h264_videotoolbox", quality=quality
    )
    assert videotoolbox_args[videotoolbox_args.index("-q:v") + 1] == videotoolbox
    assert "-maxrate:v" not in cpu
    assert "-b:v" not in cpu


def test_default_preview_quality_is_standard():
    assert video_encoders.quality_encoder_args("libx264")[
        video_encoders.quality_encoder_args("libx264").index("-crf") + 1
    ] == "23"
    for encoder in (
        "h264_nvenc", "h264_qsv", "h264_amf", "h264_vaapi", "h264_videotoolbox",
    ):
        assert video_encoders.gpu_encoder_args(
            encoder, profile="preprocess"
        ) == video_encoders.gpu_encoder_args(encoder, profile="create_video")
        assert video_encoders.gpu_encoder_args(encoder) == (
            video_encoders.quality_encoder_args(encoder)
        )


@pytest.mark.parametrize("quality", ["unknown", "", "best", None])
def test_invalid_quality_rejected(quality):
    with pytest.raises(ValueError, match="Unsupported video quality"):
        video_encoders.quality_encoder_args("libx264", quality=quality)


def test_cropping_ffmpeg_command_respects_quality_choice():
    from gui.gui_preprocess import ExportJob, ExportRegion, PreprocessApp

    region = ExportRegion(
        display_name="Full image", token="full", rect=None,
        brightness=0.0, contrast=1.0, color_mode="color",
        output_path="output.mp4",
    )
    for quality, expected in (("standard", "23"), ("high", "18")):
        job = ExportJob(
            ffmpeg="ffmpeg", video_path="source.mp4", output_folder=".",
            fps=30.0, output_fps=30.0, output_speed=1.0,
            video_encoder="libx264", frame_count=30,
            source_width=320, source_height=240,
            in_frame=0, out_frame=29, regions=(region,),
            quality=quality,
        )
        cmd = PreprocessApp._build_ffmpeg_command(None, job, region)
        assert cmd[cmd.index("-crf") + 1] == expected
        assert "-maxrate:v" not in cmd
        assert "-bufsize:v" not in cmd


@pytest.mark.parametrize("extension", ["mp4", "avi"])
def test_shared_writer_preserves_frames_and_color_overlays(tmp_path, monkeypatch, extension):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("An FFmpeg executable is required for the integration test.")
    monkeypatch.setattr(video_encoders, "ffmpeg_executable", lambda: ffmpeg)
    out_path = tmp_path / f"tracking.{extension}"
    writer = video_encoders.FfmpegRawVideoWriter(
        str(out_path), 15.0, (119, 159, 3), "libx264",
    )
    try:
        for frame_idx in range(12):
            frame = np.full((119, 159, 3), 48, dtype=np.uint8)
            cv2.rectangle(frame, (10 + frame_idx, 20), (80, 75), (0, 255, 0), 2)
            writer.write(frame)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(out_path))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 120
        ok, decoded = cap.read()
        assert ok
        assert int(decoded[20, 50, 1]) > int(decoded[20, 50, 0]) + 60
    finally:
        cap.release()


@pytest.mark.parametrize(
    ("platform", "mode", "expected"),
    [
        ("darwin", "auto", False),
        ("darwin", "cpu", False),
        ("darwin", "gpu", True),
        ("win32", "auto", True),
        ("win32", "cpu", False),
        ("win32", "gpu", True),
        ("linux", "auto", True),
        ("linux", "cpu", False),
        ("linux", "gpu", True),
    ],
)
def test_platform_aware_hardware_selection(monkeypatch, platform, mode, expected):
    monkeypatch.setattr(video_encoders.sys, "platform", platform)
    assert video_encoders.prefer_hardware_video_encoder(mode) is expected
