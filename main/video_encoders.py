# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Select and validate GPU encoders for AMADEUS's pinned FFmpeg exports."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .compute_telemetry import query_nvidia_driver_info


GPU_ENCODER_LABELS = {
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


def gpu_encoder_args(encoder: str, *, profile: str = "preprocess") -> list[str]:
    """Return encoder options, preserving each export path's quality target."""
    if profile not in {"preprocess", "create_video"}:
        raise ValueError(f"Unsupported video encoding profile: {profile}")
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
    raise ValueError(f"Unsupported GPU video encoder: {encoder}")


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


def _gpu_encoder_candidates() -> tuple[tuple[str, str | None], ...]:
    """Only consider encoder families for GPUs present on this operating system."""
    if sys.platform == "darwin":
        # VideoToolbox also supports some Intel Macs. The probe requires its
        # hardware implementation, independently of the PyTorch backend.
        return (("h264_videotoolbox", None),)
    if os.name != "nt" and not sys.platform.startswith("linux"):
        return ()
    candidates: list[tuple[str, str | None]] = []
    # NVIDIA can provide NVENC without a DRM render node (including WSL2).
    if query_nvidia_driver_info() is not None:
        candidates.append(("h264_nvenc", None))
    if os.name == "nt":
        vendors = _windows_gpu_vendors()
        for vendor, encoder in _WINDOWS_ENCODERS.items():
            if vendor in vendors:
                candidates.append((encoder, None))
    elif sys.platform.startswith("linux"):
        devices = _linux_gpu_devices()
        for vendor in ("10de", "1002", "8086"):
            for device_vendor, device in devices:
                if device_vendor == vendor:
                    candidates.append(("h264_nvenc", None) if vendor == "10de" else ("h264_vaapi", device))
    return tuple(dict.fromkeys(candidates))


@lru_cache(maxsize=16)
def detect_gpu_video_encoder(
    ffmpeg: str, *, profile: str = "preprocess"
) -> tuple[str, str | None] | None:
    """Return a detected GPU encoder that completes a short encode, or None.

    Cache by pinned executable and export profile. GPU presence and an encoder
    inventory alone cannot establish driver/runtime availability.
    """
    if profile not in {"preprocess", "create_video"}:
        raise ValueError(f"Unsupported video encoding profile: {profile}")
    for encoder, device in _gpu_encoder_candidates():
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if device is not None:
            command.extend(["-vaapi_device", device])
        # Tiny frames can be rejected by NVENC. Use software frames as both
        # export paths do, so conversion/upload is validated as well.
        command.extend([
            "-f", "lavfi", "-i", "color=s=640x480:r=30:d=0.1,format=bgr24",
            "-frames:v", "3", "-an",
        ])
        if encoder == "h264_vaapi":
            command.extend(["-vf", "format=nv12,hwupload"])
        command.extend([*gpu_encoder_args(encoder, profile=profile), "-f", "null", "-"])
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            detail = str(exc)
        else:
            if completed.returncode == 0:
                return (encoder, device)
            detail = completed.stderr.strip() or f"FFmpeg exited with code {completed.returncode}."
        device_label = f" ({device})" if device is not None else ""
        print(
            f"[AMADEUS] GPU acceleration test failed for {encoder}{device_label} "
            f"using {ffmpeg}: {detail}",
            file=sys.stderr,
        )
    return None
