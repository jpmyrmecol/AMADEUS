# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Check common, high-quality FFmpeg exports without source bitrate limits."""

import shutil

import cv2
import numpy as np
import pytest

from main import video_encoders


@pytest.mark.parametrize("encoder", [
    "h264_nvenc", "h264_qsv", "h264_amf", "h264_vaapi", "h264_videotoolbox",
])
def test_gpu_quality_same_in_all_export_profiles(encoder):
    preprocess = video_encoders.gpu_encoder_args(encoder, profile="preprocess")
    create_video = video_encoders.gpu_encoder_args(encoder, profile="create_video")
    assert preprocess == create_video == video_encoders.quality_encoder_args(encoder)
    assert "-maxrate:v" not in preprocess
    assert "-bufsize:v" not in preprocess
    if encoder != "h264_videotoolbox":
        assert "18" in preprocess
    if encoder == "h264_nvenc":
        assert preprocess[preprocess.index("-cq:v") + 1] == "18"
        assert preprocess[preprocess.index("-preset") + 1] == "p5"
        assert preprocess[preprocess.index("-tune") + 1] == "hq"
        assert preprocess[preprocess.index("-b:v") + 1] == "0"


def test_cpu_quality_has_no_bitrate_ceiling():
    args = video_encoders.quality_encoder_args("libx264")
    assert args[args.index("-crf") + 1] == "18"
    assert args[args.index("-preset") + 1] == "medium"
    assert "-maxrate:v" not in args
    assert "-b:v" not in args


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
