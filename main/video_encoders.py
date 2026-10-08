# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Select and validate GPU encoders for AMADEUS's pinned FFmpeg exports."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from fractions import Fraction
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


class GpuVideoEncodingError(RuntimeError):
    """A selected GPU encoder failed during the actual video export."""


def video_stream_bitrate(ffmpeg: str, path: str, *, cancel_event=None) -> int:
    """Measure average encoded video bitrate without decoding or buffering the file."""
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", path, "-map", "0:v:0", "-c:v", "copy", "-f", "framecrc", "-",
    ]
    total_bytes = 0
    first_pts = None
    last_end = None
    time_base = None
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, text=True,
            encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Video bitrate measurement canceled.")
                if line.startswith("#tb 0:"):
                    time_base = Fraction(line.split(":", 1)[1].strip())
                elif line.strip() and not line.startswith("#"):
                    values = line.split(",")
                    pts, duration, size = (int(values[i]) for i in (2, 3, 4))
                    total_bytes += size
                    first_pts = pts if first_pts is None else min(first_pts, pts)
                    end = pts + duration
                    last_end = end if last_end is None else max(last_end, end)
            return_code = process.wait()
            if return_code != 0:
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Could not measure video bitrate: {detail}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if process.stdout is not None:
                process.stdout.close()
    if not time_base or first_pts is None or last_end is None or last_end <= first_pts or total_bytes <= 0:
        raise RuntimeError(f"Could not determine encoded video bitrate: {path}")
    return int(Fraction(total_bytes * 8, last_end - first_pts) / time_base)


def bitrate_limited_encoder_args(encoder: str, max_bitrate: int) -> list[str]:
    """Use variable bitrate with a ceiling for Cropping & Trimming."""
    maximum = int(max_bitrate)
    if maximum <= 0:
        raise ValueError("Video bitrate limit must be positive.")
    average = max(1, int(maximum * 0.9))
    limits = ["-b:v", str(average), "-maxrate:v", str(maximum), "-bufsize:v", str(maximum)]
    if encoder == "libx264":
        # Optional SEI packets include x264's encoder identification. Their
        # fixed overhead can exceed a low source bitrate for short trims.
        return ["-c:v", encoder, "-preset", "medium", "-crf", "18", *limits,
                "-pix_fmt", "yuv420p", "-bsf:v", "filter_units=remove_types=6"]
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, "-preset", "p5", "-tune", "hq", "-rc:v", "vbr", "-cq:v", "18", *limits, "-pix_fmt", "yuv420p"]
    if encoder == "h264_qsv":
        return ["-c:v", encoder, "-preset", "medium", *limits, "-pix_fmt", "nv12"]
    if encoder == "h264_amf":
        return ["-c:v", encoder, "-quality", "quality", "-rc:v", "vbr_peak", *limits, "-pix_fmt", "yuv420p"]
    if encoder == "h264_vaapi":
        return ["-c:v", encoder, "-rc_mode", "VBR", *limits]
    if encoder == "h264_videotoolbox":
        return ["-c:v", encoder, *limits, "-allow_sw", "0", "-pix_fmt", "yuv420p"]
    raise ValueError(f"Unsupported video encoder: {encoder}")


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
