# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Select hardware video encoders from the GPUs detected by the operating system."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .compute_telemetry import query_nvidia_driver_info


HARDWARE_ENCODER_LABELS = {
    "h264_videotoolbox": "Apple VideoToolbox",
    "h264_nvenc": "NVIDIA NVENC",
    "h264_qsv": "Intel Quick Sync",
    "h264_amf": "AMD AMF",
    "h264_vaapi": "VAAPI (AMD/Intel)",
}

_WINDOWS_ENCODERS = {
    "10de": "h264_nvenc",
    "1002": "h264_amf",
    "8086": "h264_qsv",
}


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


def _windows_gpu_vendors() -> tuple[str, ...]:
    """Read PCI vendor IDs without depending on a PyTorch compute backend."""
    try:
        completed = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty PNPDeviceID",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[AMADEUS] GPU detection failed: {exc}", file=sys.stderr)
        return ()
    if completed.returncode != 0:
        print(f"[AMADEUS] GPU detection failed: {completed.stderr.strip()}", file=sys.stderr)
        return ()
    devices = completed.stdout.upper()
    return tuple(vendor for vendor in _WINDOWS_ENCODERS if f"VEN_{vendor.upper()}" in devices)


def _linux_gpu_devices() -> tuple[tuple[str, str], ...]:
    """Return PCI vendor IDs and their accessible DRM render devices."""
    devices: list[tuple[str, str]] = []
    for path in sorted(Path("/dev/dri").glob("renderD*")):
        if not path.is_char_device():
            continue
        vendor_path = Path("/sys/class/drm") / path.name / "device" / "vendor"
        try:
            vendor = vendor_path.read_text().strip().lower().removeprefix("0x")
        except OSError:
            continue
        devices.append((vendor, str(path)))
    return tuple(devices)


@lru_cache(maxsize=1)
def detect_hardware_video_encoder() -> tuple[str, str | None] | None:
    """Choose an encoder from GPU presence, without a trial FFmpeg encode."""
    if sys.platform == "darwin":
        return ("h264_videotoolbox", None)
    if query_nvidia_driver_info() is not None:
        return ("h264_nvenc", None)
    if os.name == "nt":
        vendors = _windows_gpu_vendors()
        for vendor, encoder in _WINDOWS_ENCODERS.items():
            if vendor in vendors:
                return (encoder, None)
    elif sys.platform.startswith("linux"):
        devices = _linux_gpu_devices()
        for vendor in ("10de", "1002", "8086"):
            for device_vendor, device in devices:
                if device_vendor == vendor:
                    return ("h264_nvenc", None) if vendor == "10de" else ("h264_vaapi", device)
    return None
