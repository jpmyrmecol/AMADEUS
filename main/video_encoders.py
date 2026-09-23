# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared hardware encoder selection for AMADEUS video exports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


HARDWARE_ENCODER_LABELS = {
    "h264_videotoolbox": "Apple VideoToolbox",
    "h264_nvenc": "NVIDIA NVENC",
    "h264_qsv": "Intel Quick Sync",
    "h264_amf": "AMD AMF",
    "h264_vaapi": "VAAPI (AMD/Intel)",
}

_ENCODER_CACHE: dict[str, tuple[str, str | None] | None] = {}


def hardware_encoder_candidates() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("h264_videotoolbox",)
    if os.name == "nt":
        return ("h264_nvenc", "h264_qsv", "h264_amf")
    return ("h264_nvenc", "h264_qsv", "h264_amf", "h264_vaapi")


def hardware_encoder_args(encoder: str, *, profile: str = "preprocess") -> list[str]:
    """Return encoder options, preserving each export path's quality target."""
    if encoder == "h264_nvenc":
        cq = "12" if profile == "preprocess" else "23"
        preset = "p5" if profile == "preprocess" else "fast"
        args = ["-c:v", encoder, "-preset", preset]
        if profile == "preprocess":
            args.extend(["-tune", "hq"])
        args.extend(["-rc:v", "vbr", "-cq:v", cq, "-b:v", "0", "-pix_fmt", "yuv420p"])
        return args
    if encoder == "h264_qsv":
        quality = "12" if profile == "preprocess" else "23"
        return ["-c:v", encoder, "-preset", "medium", "-global_quality:v", quality, "-pix_fmt", "nv12"]
    if encoder == "h264_amf":
        quality = "12" if profile == "preprocess" else "23"
        return ["-c:v", encoder, "-quality", "quality", "-rc:v", "cqp", "-qp_i", quality, "-qp_p", quality, "-qp_b", quality, "-pix_fmt", "yuv420p"]
    if encoder == "h264_vaapi":
        quality = "12" if profile == "preprocess" else "23"
        return ["-c:v", encoder, "-qp", quality]
    if encoder == "h264_videotoolbox":
        return ["-c:v", encoder, "-q:v", "85", "-allow_sw", "0", "-pix_fmt", "yuv420p"]
    raise ValueError(f"Unsupported hardware video encoder: {encoder}")


def _vaapi_device_candidates() -> tuple[str, ...]:
    if not sys.platform.startswith("linux"):
        return ()
    return tuple(
        sorted(str(path) for path in Path("/dev/dri").glob("renderD*") if path.is_char_device())
    )


def detect_hardware_video_encoder(ffmpeg: str) -> tuple[str, str | None] | None:
    """Return the first encoder/device pair that completes a real short encode."""
    cached = _ENCODER_CACHE.get(ffmpeg, ...)
    if cached is not ...:
        return cached

    for encoder in hardware_encoder_candidates():
        devices = _vaapi_device_candidates() if encoder == "h264_vaapi" else (None,)
        for device in devices:
            command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
            if device is not None:
                command.extend(["-vaapi_device", device])
            command.extend([
                "-f", "lavfi", "-i", "color=s=128x128:r=30:d=0.2", "-frames:v", "2",
            ])
            if encoder == "h264_vaapi":
                command.extend(["-vf", "format=nv12,hwupload"])
            command.extend([*hardware_encoder_args(encoder), "-f", "null", "-"])
            try:
                completed = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if completed.returncode == 0:
                result = (encoder, device)
                _ENCODER_CACHE[ffmpeg] = result
                return result

    _ENCODER_CACHE[ffmpeg] = None
    return None
