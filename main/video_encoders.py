# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Select and validate GPU encoders for AMADEUS's pinned FFmpeg exports."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
from functools import lru_cache
from pathlib import Path

from .compute_telemetry import query_nvidia_driver_info
from .video_compat import ffmpeg_executable


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


VIDEO_QUALITY_QP = {"standard": "23", "high": "18"}
VIDEO_QUALITY_VIDEOTOOLBOX = {"standard": "70", "high": "85"}
VIDEO_QUALITY_MPEG4 = {"standard": "3", "high": "2"}


class GpuVideoEncodingError(RuntimeError):
    """A selected GPU encoder failed during the actual video export."""


def prefer_hardware_video_encoder(mode: str) -> bool:
    """Default to CPU on macOS, but allow explicit VideoToolbox acceleration."""
    return mode == "gpu" or (mode == "auto" and sys.platform != "darwin")


def normalize_video_quality(value: str) -> str:
    """Validate the two exported video-quality profiles."""
    quality = str(value).strip().lower()
    if quality not in VIDEO_QUALITY_QP:
        raise ValueError(f"Unsupported video quality: {value!r}. Use 'standard' or 'high'.")
    return quality


def gpu_encoder_args(
    encoder: str, *, profile: str = "create_video", quality: str = "standard",
) -> list[str]:
    """Return hardware encoder settings for the requested quality profile.

    The quality numbers are not equivalent between hardware encoder families.
    """
    if profile not in {"preprocess", "create_video"}:
        raise ValueError(f"Unsupported video encoding profile: {profile}")
    quality = normalize_video_quality(quality)
    qp = VIDEO_QUALITY_QP[quality]
    if encoder == "h264_nvenc":
        return ["-c:v", encoder, "-preset", "p5", "-tune", "hq", "-rc:v", "vbr",
                "-cq:v", qp, "-b:v", "0", "-pix_fmt", "yuv420p"]
    if encoder == "h264_qsv":
        return ["-c:v", encoder, "-preset", "medium",
                "-global_quality:v", qp, "-pix_fmt", "nv12"]
    if encoder == "h264_amf":
        return ["-c:v", encoder, "-quality", "quality", "-rc:v", "cqp",
                "-qp_i", qp, "-qp_p", qp, "-qp_b", qp, "-pix_fmt", "yuv420p"]
    if encoder == "h264_vaapi":
        return ["-c:v", encoder, "-qp", qp]
    if encoder == "h264_videotoolbox":
        return ["-c:v", encoder, "-q:v", VIDEO_QUALITY_VIDEOTOOLBOX[quality],
                "-allow_sw", "0", "-pix_fmt", "yuv420p"]
    raise ValueError(f"Unsupported GPU video encoder: {encoder}")


def quality_encoder_args(encoder: str, *, quality: str = "standard") -> list[str]:
    """Encode previews at Standard and input-preparation video at High quality."""
    quality = normalize_video_quality(quality)
    if encoder == "libx264":
        return ["-c:v", encoder, "-preset", "medium",
                "-crf", VIDEO_QUALITY_QP[quality], "-pix_fmt", "yuv420p"]
    if encoder == "mpeg4":
        return ["-c:v", encoder, "-q:v", VIDEO_QUALITY_MPEG4[quality],
                "-pix_fmt", "yuv420p"]
    return gpu_encoder_args(encoder, quality=quality)


class FfmpegRawVideoWriter:
    """Write rendered BGR frames through AMADEUS's fixed FFmpeg build."""

    def __init__(
        self,
        path: str,
        fps: float,
        frame_shape: tuple[int, int, int],
        encoder: str,
        device: str | None = None,
        *,
        quality: str = "standard",
    ) -> None:
        h, w = frame_shape[:2]
        ffmpeg = ffmpeg_executable()
        self.path = path
        self.encoder = encoder
        self._released = False
        self.width = int(w)
        self.height = int(h)
        filters = ["pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        if encoder == "h264_vaapi":
            if not device:
                raise RuntimeError("VAAPI requires a detected render device.")
            filters.append("format=nv12,hwupload")
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if device:
            command.extend(["-vaapi_device", device])
        command.extend([
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(float(fps)),
            "-i", "pipe:0",
            "-vf", ",".join(filters),
            *quality_encoder_args(encoder, quality=quality),
            path,
        ])
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("FFmpeg writer stdin is closed.")
        arr = np.ascontiguousarray(frame)
        if arr.shape[0] != self.height or arr.shape[1] != self.width or arr.shape[2] != 3:
            raise RuntimeError(f"Unexpected frame shape for FFmpeg writer: {arr.shape}")
        try:
            self.proc.stdin.write(arr.tobytes())
        except BrokenPipeError as exc:
            self.release()
            error_type = GpuVideoEncodingError if self.encoder not in {"libx264", "mpeg4"} else RuntimeError
            raise error_type("FFmpeg video writer closed its input unexpectedly.") from exc

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        err = self.proc.stderr.read() if self.proc.stderr is not None else b""
        rc = self.proc.wait()
        if self.proc.stderr is not None:
            self.proc.stderr.close()
        if rc != 0:
            if os.path.exists(self.path):
                os.remove(self.path)
            msg = err.decode("utf-8", errors="replace").strip()
            detail = f": {msg}" if msg else ""
            error_type = GpuVideoEncodingError if self.encoder not in {"libx264", "mpeg4"} else RuntimeError
            raise error_type(f"FFmpeg video writer failed with exit code {rc}{detail}")


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
