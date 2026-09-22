# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""One shared layer for every video AMADEUS reads.

AMADEUS addresses frames directly (``CAP_PROP_POS_FRAMES`` + ``read()``), so an
input video is only usable when OpenCV can open it, decode it, and land on an
arbitrary frame. Container and codec alone do not decide that, so this module
inspects the file itself:

* :func:`probe_source` reads container/codec/pixel-format/rotation/HDR metadata
  from FFmpeg's own stream dump. FFmpeg ships with AMADEUS (``imageio-ffmpeg``),
  so this works on Windows, macOS and Linux without a separate install, and
  without depending on ``ffprobe``, which that wheel does not contain.
* :func:`probe_opencv` measures what actually matters downstream: does frame 0
  decode, does a random seek land, how many frames can be addressed, and are the
  frame intervals constant.
* :func:`assess_video` turns both into one verdict plus human-readable reasons.
* :func:`build_plan` / :func:`run_conversion` normalize an awkward video into an
  analysis copy: same resolution, 8-bit SDR, constant frame rate, short GOP,
  H.264/MP4. The source file is only ever opened for reading.

Frame counting stays in :mod:`video_frame_count`; this module calls it instead of
repeating the MOV/HEVC tail-seek workaround.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # Imported as ``main.video_compat`` from the GUIs.
    from .video_frame_count import detect_seekable_frame_count
except ImportError:  # Imported as ``video_compat`` from main/ scripts.
    from video_frame_count import detect_seekable_frame_count


# --------------------------------------------------------------------------
# Accepted inputs
# --------------------------------------------------------------------------

# Offered in the file dialogs and accepted on drop. Membership here only decides
# what the user may *select*; whether a file can be used unchanged is decided by
# assess_video() after probing and decoding it.
SUPPORTED_VIDEO_SUFFIXES: tuple[str, ...] = (
    ".mp4",
    ".mov",
    ".avi",
    ".m4v",
    ".mts",
    ".m2ts",
    ".ts",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".webm",
)

VIDEO_DROP_SUFFIXES = frozenset(SUPPORTED_VIDEO_SUFFIXES)

VIDEO_FILETYPES: list[tuple[str, str]] = [
    ("Video files", " ".join(f"*{suffix}" for suffix in SUPPORTED_VIDEO_SUFFIXES)),
    ("All files", "*.*"),
]

# Containers OpenCV historically reads without help in AMADEUS. A file outside
# this set is normalized even when it happens to decode, because seeking in
# transport streams and Matroska is not dependable across OpenCV builds.
ROBUST_CONTAINER_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".m4v"})

# Codecs whose OpenCV support depends on the FFmpeg build behind cv2.
FRAGILE_CODECS = frozenset({"hevc", "h265", "av1", "vp9", "prores", "dnxhd", "mpeg2video"})

HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})

# --------------------------------------------------------------------------
# Analysis-copy encoding settings
# --------------------------------------------------------------------------

# Visually lossless for practical purposes: small-animal outlines, blob
# segmentation and detector input must not be softened by the normalization.
ANALYSIS_CRF = 12
ANALYSIS_PRESET = "medium"
ANALYSIS_PIX_FMT = "yuv420p"
# A short GOP keeps CAP_PROP_POS_FRAMES seeks cheap; AMADEUS seeks constantly.
ANALYSIS_GOP = 30
ANALYSIS_SUFFIX = ".mp4"

CONVERTED_DIR_NAME = "converted"
CONVERTED_NAME_SUFFIX = "_amadeus"
METADATA_EXTENSION = ".amadeus_conversion.json"

PROBE_TIMEOUT_SECONDS = 60
# Sequential frames sampled when measuring whether the frame rate is constant.
FRAME_INTERVAL_SAMPLE = 120
# An interval this far from the median counts as irregular.
FRAME_INTERVAL_TOLERANCE = 0.25
# Share of irregular intervals above which the video is treated as VFR.
VFR_IRREGULAR_FRACTION = 0.05

# The FFmpeg AMADEUS uses is pinned, not discovered: the analysis copies this
# module produces must be reproducible, so the encoder version has to be fixed
# by the lockfile rather than by whatever happens to be installed on the host.
PINNED_FFMPEG_REQUIREMENT = "imageio-ffmpeg==0.6.0"
FFMPEG_OVERRIDE_ENV_VAR = "AMADEUS_FFMPEG"

FFMPEG_NOT_FOUND_MESSAGE = (
    "The FFmpeg build AMADEUS depends on was not found.\n\n"
    f"AMADEUS uses the FFmpeg binary pinned by {PINNED_FFMPEG_REQUIREMENT}, which "
    "the installer places inside the AMADEUS environment. It is not on your PATH "
    "and is not meant to be: pinning the build keeps every converted analysis "
    "video reproducible.\n\n"
    "Without it, videos cannot be inspected in detail or converted.\n\n"
    "How to fix it:\n"
    "  1. Re-run the AMADEUS launcher (AMADEUS.bat on Windows, AMADEUS.command on "
    "macOS, AMADEUS.sh on Linux/WSL2) so the environment is reinstalled, or\n"
    "  2. Reinstall the dependency directly:  pip install "
    f"{PINNED_FFMPEG_REQUIREMENT}\n\n"
    f"To point AMADEUS at a specific FFmpeg build instead, set the "
    f"{FFMPEG_OVERRIDE_ENV_VAR} environment variable to its full path. The version "
    "actually used is recorded in every conversion's metadata file.\n\n"
    "Videos already in a format AMADEUS reads directly (8-bit H.264 MP4/MOV/AVI) "
    "can still be used without FFmpeg."
)


class FfmpegUnavailableError(RuntimeError):
    """Raised when no usable FFmpeg binary can be located."""

    def __init__(self, detail: str = ""):
        message = FFMPEG_NOT_FOUND_MESSAGE
        if detail:
            message = f"{message}\n\nDetail: {detail}"
        super().__init__(message)
        self.detail = detail


class ConversionCancelled(RuntimeError):
    """Raised when the user stops a conversion before it finishes."""


# --------------------------------------------------------------------------
# FFmpeg discovery
# --------------------------------------------------------------------------

def _subprocess_kwargs() -> dict:
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _pinned_ffmpeg_binary() -> str:
    """Locate the FFmpeg binary shipped inside the pinned imageio-ffmpeg wheel.

    ``imageio_ffmpeg.get_ffmpeg_exe()`` is deliberately not used: it falls back
    to a conda or PATH FFmpeg when the bundled binary is missing, which would
    silently swap the encoder for an unknown version. Each platform wheel of
    ``imageio-ffmpeg`` contains exactly one ``ffmpeg-<platform>-v<version>``
    binary, and that is the build AMADEUS is pinned to.
    """
    import importlib.resources

    try:
        binaries = importlib.resources.files("imageio_ffmpeg.binaries")
    except (ImportError, ModuleNotFoundError) as exc:
        raise FfmpegUnavailableError(str(exc)) from exc

    candidates = []
    for entry in binaries.iterdir():
        name = entry.name
        if name.startswith("ffmpeg-") and not name.endswith(".md"):
            candidates.append(str(entry))
    if not candidates:
        raise FfmpegUnavailableError(
            f"The imageio-ffmpeg installation at {binaries} contains no FFmpeg binary."
        )
    return sorted(candidates)[-1]


def ffmpeg_executable() -> str:
    """Return the pinned FFmpeg binary AMADEUS must use.

    ``AMADEUS_FFMPEG`` overrides it for sites that have to supply their own
    build; the version in use is written into every conversion's metadata either
    way, so the analysis stays reproducible.
    """
    override = os.environ.get(FFMPEG_OVERRIDE_ENV_VAR, "").strip()
    if override:
        if not os.path.isfile(override):
            raise FfmpegUnavailableError(
                f"{FFMPEG_OVERRIDE_ENV_VAR} points at {override!r}, which is not a file."
            )
        return override

    path = _pinned_ffmpeg_binary()
    if not os.path.isfile(path):
        raise FfmpegUnavailableError(f"The pinned FFmpeg binary is missing: {path}")
    if not os.access(path, os.X_OK):
        raise FfmpegUnavailableError(f"The pinned FFmpeg binary is not executable: {path}")
    return path


def ffmpeg_is_available() -> bool:
    try:
        ffmpeg_executable()
    except FfmpegUnavailableError:
        return False
    return True


def _run_ffmpeg(args: Sequence[str], *, timeout: int = PROBE_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ffmpeg_executable(), "-hide_banner", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        **_subprocess_kwargs(),
    )


_version_cache: dict[str, str] = {}


def ffmpeg_version() -> str:
    """Return the FFmpeg version banner line, cached per binary."""
    executable = ffmpeg_executable()
    cached = _version_cache.get(executable)
    if cached is not None:
        return cached
    completed = _run_ffmpeg(["-version"], timeout=30)
    first_line = (completed.stdout or "").splitlines()
    version = first_line[0].strip() if first_line else "unknown"
    _version_cache[executable] = version
    return version


_filters_cache: dict[str, frozenset[str]] = {}


def available_filters() -> frozenset[str]:
    """Return the filter names this FFmpeg build provides."""
    executable = ffmpeg_executable()
    cached = _filters_cache.get(executable)
    if cached is not None:
        return cached
    completed = _run_ffmpeg(["-filters"], timeout=30)
    names = set()
    for line in (completed.stdout or "").splitlines():
        match = re.match(r"^\s*[A-Z.]{3,}\s+(\S+)\s+\S+->\S+", line)
        if match:
            names.add(match.group(1))
    result = frozenset(names)
    _filters_cache[executable] = result
    return result


def can_tone_map() -> bool:
    """True when this FFmpeg build can convert HDR to SDR properly."""
    filters = available_filters()
    return "zscale" in filters and "tonemap" in filters


# --------------------------------------------------------------------------
# Source metadata, read from FFmpeg's stream dump
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceVideoInfo:
    path: str = ""
    container: str = ""
    codec: str = ""
    profile: str = ""
    pix_fmt: str = ""
    bit_depth: int = 0
    width: int = 0
    height: int = 0
    nominal_fps: float = 0.0
    tbr: float = 0.0
    duration_seconds: float = 0.0
    rotation_degrees: int = 0
    color_transfer: str = ""
    color_primaries: str = ""
    audio_streams: int = 0
    error: str = ""

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS

    @property
    def is_wide_gamut(self) -> bool:
        return self.color_primaries.startswith("bt2020")

    def as_dict(self) -> dict:
        data = {
            "path": self.path,
            "container": self.container,
            "codec": self.codec,
            "profile": self.profile,
            "pix_fmt": self.pix_fmt,
            "bit_depth": self.bit_depth,
            "width": self.width,
            "height": self.height,
            "nominal_fps": self.nominal_fps,
            "tbr": self.tbr,
            "duration_seconds": self.duration_seconds,
            "rotation_degrees": self.rotation_degrees,
            "color_transfer": self.color_transfer,
            "color_primaries": self.color_primaries,
            "hdr": self.is_hdr,
        }
        if self.error:
            data["probe_error"] = self.error
        return data


_PIX_FMT_RE = re.compile(
    r"\b("
    r"yuva?j?\d{3}[a-z0-9]*"
    r"|gbra?p[a-z0-9]*"
    r"|gray[a-z0-9]*"
    r"|nv\d{2}"
    r"|p\d{3}(?:le|be)"
    r"|(?:rgb|bgr|argb|abgr|rgba|bgra)[a-z0-9]*"
    r"|pal8"
    r"|monob|monow"
    r")\b"
)
_SEMI_PLANAR_RE = re.compile(r"^p\d{3}(?:le|be)$")
_TRAILING_DEPTH_RE = re.compile(r"(\d{1,2})(?:le|be)$")


_COLOUR_NAME_RE = re.compile(
    r"^(?:bt2020(?:nc|c)?|bt709|bt470bg|bt470m|smpte170m|smpte240m|smpte428|smpte431|"
    r"smpte432|smpte2084|arib-std-b67|linear|log100|log316|iec61966-2-1|bt1361e|"
    r"unknown|reserved|fcc|gbr|ycgco|chroma-derived-nc|chroma-derived-c|ictcp)$"
)


def _parse_colour_description(line: str, start: int) -> tuple[str, str]:
    """Read (primaries, transfer) from the parentheses after the pixel format.

    FFmpeg writes them as ``yuv420p10le(tv, bt2020nc/bt2020/smpte2084, progressive)``
    -- colourspace/primaries/transfer -- and collapses the three to a single name
    when they agree, as in ``yuv420p(tv, bt709, progressive)``.
    """
    if start >= len(line) or line[start] != "(":
        return "", ""
    end = line.find(")", start)
    if end < 0:
        return "", ""
    for field in line[start + 1:end].split(","):
        field = field.strip()
        if "/" in field:
            parts = [part.strip() for part in field.split("/")]
            if len(parts) >= 3 and _COLOUR_NAME_RE.match(parts[2]):
                return parts[1], parts[2]
            continue
        if _COLOUR_NAME_RE.match(field) and field not in {"unknown", "reserved"}:
            return field, field
    return "", ""


def _parse_duration(text: str) -> float:
    match = re.search(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", text)
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_rate(value: str) -> float:
    value = value.strip()
    multiplier = 1000.0 if value.endswith("k") else 1.0
    if multiplier != 1.0:
        value = value[:-1]
    try:
        return float(value) * multiplier
    except ValueError:
        return 0.0


def _bit_depth_from_pix_fmt(pix_fmt: str) -> int:
    """Bits per component for an FFmpeg pixel-format name (8 when unmarked)."""
    if not pix_fmt:
        return 0
    if _SEMI_PLANAR_RE.match(pix_fmt):
        # pXYZle: X is the chroma subsampling, YZ the bit depth (p010le, p216le).
        return int(pix_fmt[2:4])
    match = _TRAILING_DEPTH_RE.search(pix_fmt)
    if match:
        return int(match.group(1))
    return 8


def parse_ffmpeg_stream_dump(text: str) -> dict:
    """Extract video-stream facts from the report ``ffmpeg -i`` writes.

    Split out from :func:`probe_source` so the parser can be tested against
    recorded FFmpeg output without running FFmpeg.
    """
    result: dict[str, Any] = {}

    container = re.search(r"^Input #\d+,\s*(.+?),\s*from ", text, re.MULTILINE)
    if container:
        result["container"] = container.group(1).strip()

    result["duration_seconds"] = _parse_duration(text)

    stream = re.search(r"^\s*Stream #\d+:\d+.*?:\s*Video:\s*(.+)$", text, re.MULTILINE)
    if not stream:
        result["error"] = "No video stream was reported by FFmpeg."
        return result
    line = stream.group(1)

    codec = re.match(r"([A-Za-z0-9_]+)", line)
    if codec:
        result["codec"] = codec.group(1).lower()
    profile = re.match(r"[A-Za-z0-9_]+\s+\(([^)/]+)\)", line)
    if profile:
        result["profile"] = profile.group(1).strip()

    # Skip the codec tag -- "(hvc1 / 0x31637668)" -- before looking for the
    # pixel format, so its hex digits cannot be mistaken for one.
    scan_from = line.find(")") + 1 if profile else 0
    pix_fmt = _PIX_FMT_RE.search(line, scan_from)
    if pix_fmt:
        result["pix_fmt"] = pix_fmt.group(1)
        result["bit_depth"] = _bit_depth_from_pix_fmt(pix_fmt.group(1))

    size = re.search(r",\s*(\d{2,5})x(\d{2,5})\b", line)
    if size:
        result["width"] = int(size.group(1))
        result["height"] = int(size.group(2))

    fps = re.search(r"([\d.]+k?)\s+fps\b", line)
    if fps:
        result["nominal_fps"] = _parse_rate(fps.group(1))
    tbr = re.search(r"([\d.]+k?)\s+tbr\b", line)
    if tbr:
        result["tbr"] = _parse_rate(tbr.group(1))

    if pix_fmt:
        primaries, transfer = _parse_colour_description(line, pix_fmt.end())
        if primaries:
            result["color_primaries"] = primaries
        if transfer:
            result["color_transfer"] = transfer

    rotation = re.search(r"rotation of\s*(-?[\d.]+)\s*degrees", text)
    if rotation:
        # FFmpeg prints the display-matrix angle; AMADEUS records the clockwise
        # rotation a player applies, which is its negation.
        result["rotation_degrees"] = int(round(-float(rotation.group(1)))) % 360

    result["audio_streams"] = len(re.findall(r"^\s*Stream #\d+:\d+.*?:\s*Audio:", text, re.MULTILINE))
    return result


def probe_source(path: str) -> SourceVideoInfo:
    """Read container/codec/colour metadata for ``path`` using FFmpeg."""
    path = str(path)
    try:
        completed = _run_ffmpeg(["-i", path])
    except FfmpegUnavailableError as exc:
        return SourceVideoInfo(path=path, error=exc.detail or "FFmpeg is unavailable.")
    except subprocess.TimeoutExpired:
        return SourceVideoInfo(path=path, error="FFmpeg timed out while inspecting the file.")
    except OSError as exc:
        return SourceVideoInfo(path=path, error=str(exc))

    # "ffmpeg -i <file>" always exits non-zero because no output was requested;
    # the stream report we want is on stderr either way.
    parsed = parse_ffmpeg_stream_dump(completed.stderr or "")
    if "codec" not in parsed:
        if completed.returncode < 0:
            parsed["error"] = (
                f"FFmpeg stopped unexpectedly (signal {-completed.returncode}) while "
                "reading this file."
            )
        elif "error" not in parsed:
            parsed["error"] = "FFmpeg did not report a readable video stream."
    return SourceVideoInfo(path=path, **parsed)


# --------------------------------------------------------------------------
# What OpenCV can actually do with the file
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenCvProbe:
    opened: bool = False
    fourcc: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    reported_frame_count: int = 0
    seekable_frame_count: int = 0
    first_frame_decoded: bool = False
    random_seek_ok: bool = False
    measured_fps: float = 0.0
    irregular_interval_fraction: float | None = None
    variable_frame_rate: bool | None = None
    error: str = ""

    @property
    def unreadable_tail(self) -> int:
        return max(0, self.reported_frame_count - self.seekable_frame_count)

    @property
    def addressable_frame_count(self) -> int:
        """Frames a frame range may span.

        When seeking works this is the count proven reachable; when it does not,
        the container's own count is the only figure available -- and that file
        is on its way to being converted anyway, where frames are read in decode
        order rather than by seeking.
        """
        if self.random_seek_ok:
            return self.seekable_frame_count
        return self.reported_frame_count

    def as_dict(self) -> dict:
        data = {
            "opened": self.opened,
            "fourcc": self.fourcc,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "reported_frame_count": self.reported_frame_count,
            "seekable_frame_count": self.seekable_frame_count,
            "first_frame_decoded": self.first_frame_decoded,
            "random_seek_ok": self.random_seek_ok,
            "measured_fps": self.measured_fps,
            "variable_frame_rate": self.variable_frame_rate,
        }
        if self.irregular_interval_fraction is not None:
            data["irregular_interval_fraction"] = round(self.irregular_interval_fraction, 4)
        if self.error:
            data["error"] = self.error
        return data


def _cv2() -> Any:
    import cv2

    return cv2


def _fourcc_to_string(value: float) -> str:
    code = int(value or 0)
    if code <= 0:
        return ""
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(ch for ch in chars if ch.isprintable()).strip()


def measure_frame_intervals(cap: Any, sample: int = FRAME_INTERVAL_SAMPLE) -> tuple[float, float | None]:
    """Decode a short run and report (measured fps, irregular-interval share).

    Presentation timestamps expose a variable frame rate directly, which no
    container or codec name can. Returns ``(0.0, None)`` when too few frames
    decode to judge.
    """
    cv2 = _cv2()
    timestamps: list[float] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(max(2, int(sample))):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
    if len(timestamps) < 8:
        return 0.0, None

    intervals = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if len(intervals) < 6:
        return 0.0, None
    ordered = sorted(intervals)
    median = ordered[len(ordered) // 2]
    if median <= 0.0:
        return 0.0, None
    irregular = sum(1 for value in intervals if abs(value - median) / median > FRAME_INTERVAL_TOLERANCE)
    return 1000.0 / median, irregular / len(intervals)


def probe_opencv(path: str) -> OpenCvProbe:
    """Open ``path`` with OpenCV and test the access pattern AMADEUS relies on."""
    try:
        cv2 = _cv2()
    except ImportError as exc:
        return OpenCvProbe(error=str(exc))

    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return OpenCvProbe(error="OpenCV could not open the file.")
    try:
        reported = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fourcc = _fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))

        ok, frame = cap.read()
        first_ok = bool(ok and frame is not None and getattr(frame, "size", 0) > 0)

        measured_fps, irregular = measure_frame_intervals(cap)

        random_ok = False
        if first_ok and reported > 2:
            target = reported // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            random_ok = bool(ok and frame is not None and getattr(frame, "size", 0) > 0)
        elif first_ok:
            random_ok = True
    finally:
        cap.release()

    # detect_seekable_frame_count() probes the tail frame by frame, which is slow
    # and floods the console with decoder errors on a file OpenCV cannot seek in
    # at all. That file already needs converting, so skip the count.
    seekable = detect_seekable_frame_count(path, reported) if random_ok else 0

    variable = None
    if irregular is not None:
        variable = irregular > VFR_IRREGULAR_FRACTION

    return OpenCvProbe(
        opened=True,
        fourcc=fourcc,
        width=width,
        height=height,
        fps=fps if fps > 0 else 0.0,
        reported_frame_count=reported,
        seekable_frame_count=seekable,
        first_frame_decoded=first_ok,
        random_seek_ok=random_ok,
        measured_fps=measured_fps,
        irregular_interval_fraction=irregular,
        variable_frame_rate=variable,
    )


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

STATUS_OK = "ok"
STATUS_RECOMMENDED = "recommended"
STATUS_REQUIRED = "required"
STATUS_UNREADABLE = "unreadable"

_STATUS_ORDER = {STATUS_OK: 0, STATUS_RECOMMENDED: 1, STATUS_REQUIRED: 2, STATUS_UNREADABLE: 3}


@dataclass(frozen=True)
class VideoAssessment:
    path: str
    source: SourceVideoInfo
    opencv: OpenCvProbe
    status: str
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def needs_dialog(self) -> bool:
        return self.status != STATUS_OK

    @property
    def can_convert(self) -> bool:
        return self.status in (STATUS_REQUIRED, STATUS_RECOMMENDED, STATUS_UNREADABLE)

    @property
    def usable_frame_count(self) -> int:
        count = int(self.opencv.addressable_frame_count)
        if count <= 0 and self.source.duration_seconds > 0:
            fps = self.opencv.fps or self.source.nominal_fps
            if fps > 0:
                count = int(round(self.source.duration_seconds * fps))
        return max(0, count)

    @property
    def is_hdr(self) -> bool:
        return self.source.is_hdr

    @property
    def is_variable_frame_rate(self) -> bool:
        return bool(self.opencv.variable_frame_rate)

    def summary_lines(self) -> list[str]:
        source, probe = self.source, self.opencv
        width = source.width or probe.width
        height = source.height or probe.height
        fps = probe.fps or source.nominal_fps
        lines = [
            f"File: {os.path.basename(self.path)}",
            f"Container: {source.container or Path(self.path).suffix.lstrip('.') or 'unknown'}",
            f"Codec: {source.codec or probe.fourcc or 'unknown'}"
            + (f" ({source.profile})" if source.profile else ""),
            f"Resolution: {width} x {height}" if width and height else "Resolution: unknown",
            f"Pixel format: {source.pix_fmt or 'unknown'}"
            + (f" ({source.bit_depth}-bit)" if source.bit_depth else ""),
            f"Frame rate: {fps:.3f} fps" if fps else "Frame rate: unknown",
            "Frame rate mode: "
            + (
                "unknown"
                if probe.variable_frame_rate is None
                else ("variable (VFR)" if probe.variable_frame_rate else "constant (CFR)")
            ),
            "Colour: "
            + (
                f"HDR ({source.color_transfer})"
                if source.is_hdr
                else (source.color_transfer or "unknown")
            )
            + (f" / {source.color_primaries}" if source.color_primaries else ""),
            f"Rotation metadata: {source.rotation_degrees} deg",
            f"Frames: {probe.reported_frame_count} reported"
            + (
                f", {probe.seekable_frame_count} addressable"
                if probe.random_seek_ok and probe.seekable_frame_count != probe.reported_frame_count
                else ""
            ),
            "Random frame access: " + ("works" if probe.random_seek_ok else "fails"),
        ]
        if source.duration_seconds:
            lines.append(f"Duration: {source.duration_seconds:.3f} s")
        return lines


def assess_video(path: str) -> VideoAssessment:
    """Inspect ``path`` and decide whether AMADEUS can use it unchanged."""
    path = str(path)
    source = probe_source(path)
    probe = probe_opencv(path)

    reasons: list[str] = []
    notes: list[str] = []
    status = STATUS_OK

    def escalate(new_status: str) -> None:
        nonlocal status
        if _STATUS_ORDER[new_status] > _STATUS_ORDER[status]:
            status = new_status

    if not probe.opened or not probe.first_frame_decoded:
        escalate(STATUS_REQUIRED)
        reasons.append(
            "OpenCV cannot decode this file directly"
            + (f" ({probe.error})" if probe.error else "")
            + ", so AMADEUS cannot read frames from it as it is."
        )
    else:
        if not probe.random_seek_ok:
            escalate(STATUS_REQUIRED)
            reasons.append(
                "Seeking to an arbitrary frame failed. AMADEUS addresses frames "
                "directly during segmentation and tracking, so this video cannot "
                "be used unchanged."
            )
        tail = probe.unreadable_tail if probe.random_seek_ok else 0
        if tail > max(2, int(probe.reported_frame_count * 0.005)):
            escalate(STATUS_REQUIRED)
            reasons.append(
                f"The last {tail} of {probe.reported_frame_count} frames cannot be "
                "reached by seeking, so part of the video would be dropped."
            )
        elif tail:
            notes.append(
                f"The final {tail} frame(s) cannot be reached by seeking and are "
                "excluded from the usable range."
            )

    suffix = Path(path).suffix.lower()
    if suffix not in ROBUST_CONTAINER_SUFFIXES:
        escalate(STATUS_RECOMMENDED)
        reasons.append(
            f"The {suffix.lstrip('.') or 'source'} container is not one AMADEUS seeks "
            "reliably in; MP4 keeps frame addressing dependable."
        )

    codec = (source.codec or "").lower()
    if codec in FRAGILE_CODECS or probe.fourcc.lower() in {"hvc1", "hev1", "av01"}:
        escalate(STATUS_RECOMMENDED)
        reasons.append(
            f"The video is encoded with {source.codec or probe.fourcc}. OpenCV support "
            "for it depends on the build, and random frame access is often slow or "
            "inexact."
        )

    if source.bit_depth and source.bit_depth > 8:
        escalate(STATUS_RECOMMENDED)
        reasons.append(
            f"The video is {source.bit_depth}-bit. OpenCV hands AMADEUS 8-bit frames, "
            "so the conversion should be done once, explicitly and on record."
        )

    if source.is_hdr:
        escalate(STATUS_RECOMMENDED)
        reasons.append(
            f"The video is HDR ({source.color_transfer}). Decoded without tone mapping "
            "it looks flat and dark, which changes segmentation thresholds."
        )
    elif source.is_wide_gamut:
        escalate(STATUS_RECOMMENDED)
        reasons.append(
            "The video uses BT.2020 primaries, which OpenCV does not convert to the "
            "BT.709 range the rest of the workflow assumes."
        )

    if probe.variable_frame_rate:
        escalate(STATUS_RECOMMENDED)
        reasons.append(
            "The frame rate is variable. Frame indices and timestamps then disagree, "
            "so speeds and time axes derived from tracking would be wrong."
        )

    if source.rotation_degrees:
        escalate(STATUS_RECOMMENDED)
        reasons.append(
            f"The file carries {source.rotation_degrees} deg rotation metadata. Whether "
            "OpenCV applies it depends on the build, so the rotation is better baked in."
        )

    if source.error:
        notes.append(f"FFmpeg metadata was incomplete: {source.error}")

    if status != STATUS_OK and not ffmpeg_is_available():
        notes.append("FFmpeg is unavailable, so no conversion can be offered.")

    return VideoAssessment(
        path=path,
        source=source,
        opencv=probe,
        status=status,
        reasons=tuple(reasons),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# Conversion plan
# --------------------------------------------------------------------------

def session_directory_for(source_path: str) -> str:
    """Return the AMADEUS session directory derived from a source video."""
    source_path = os.path.abspath(str(source_path))
    stem = Path(source_path).stem or "video"
    return os.path.join(os.path.dirname(source_path), f"amadeus_{stem}")


def conversion_directory_for(source_path: str) -> str:
    return os.path.join(session_directory_for(source_path), CONVERTED_DIR_NAME)


def conversion_output_path(source_path: str, first_frame: int, last_frame: int, total_frames: int) -> str:
    """Name the analysis copy; a partial range is named after that range."""
    stem = Path(str(source_path)).stem or "video"
    whole = int(first_frame) <= 0 and int(last_frame) >= int(total_frames) - 1
    name = f"{stem}{CONVERTED_NAME_SUFFIX}"
    if not whole:
        name = f"{name}_f{int(first_frame)}-{int(last_frame)}"
    return os.path.join(conversion_directory_for(source_path), f"{name}{ANALYSIS_SUFFIX}")


def metadata_path_for(converted_path: str) -> str:
    root, _ = os.path.splitext(str(converted_path))
    return f"{root}{METADATA_EXTENSION}"


@dataclass(frozen=True)
class ConversionPlan:
    source_path: str
    output_path: str
    first_frame: int
    last_frame: int
    target_frame_rate: Fraction
    tone_map: bool
    frame_mapping: str  # "exact" or "resampled"
    assessment: VideoAssessment | None = None
    notes: tuple[str, ...] = ()

    @property
    def metadata_path(self) -> str:
        return metadata_path_for(self.output_path)

    @property
    def target_fps(self) -> float:
        return float(self.target_frame_rate)

    @property
    def frame_rate_argument(self) -> str:
        return f"{self.target_frame_rate.numerator}/{self.target_frame_rate.denominator}"

    @property
    def frame_count(self) -> int:
        return max(0, int(self.last_frame) - int(self.first_frame) + 1)

    @property
    def expected_output_frames(self) -> int:
        """Frames the analysis copy should hold.

        Exact for a constant-rate source; for a resampled one the selected range
        is re-timed at the average rate, so the count only lands nearby.
        """
        return max(1, self.frame_count)

    @property
    def frame_count_tolerance(self) -> float:
        return 0.01 if self.frame_mapping == "exact" else 0.10


# Denominator bound that still expresses every broadcast rate exactly
# (30000/1001 and friends) without inventing absurd ratios.
FRAME_RATE_DENOMINATOR_LIMIT = 1001


def rational_frame_rate(fps: float) -> Fraction:
    """Express a frame rate as the exact rational FFmpeg should be given.

    Handing FFmpeg a rounded decimal makes its fps filter duplicate or drop a
    frame every few thousand, which would slowly shift frame indices away from
    the source.
    """
    value = Fraction(float(fps)).limit_denominator(FRAME_RATE_DENOMINATOR_LIMIT)
    if value <= 0:
        raise ValueError(f"A frame rate of {fps!r} cannot be used.")
    return value


def validate_frame_range(first_frame: int, last_frame: int, total_frames: int) -> str:
    """Return an empty string when the range is usable, else why it is not."""
    try:
        first = int(first_frame)
        last = int(last_frame)
    except (TypeError, ValueError):
        return "Frame numbers must be whole numbers."
    total = int(total_frames)
    if total <= 0:
        return "No usable frames were found in this video."
    if first < 0:
        return "The start frame cannot be negative."
    if last > total - 1:
        return f"The end frame cannot exceed {total - 1} (the last usable frame)."
    if first > last:
        return "The start frame must not be after the end frame."
    return ""


def build_plan(
    assessment: VideoAssessment,
    first_frame: int = 0,
    last_frame: int = -1,
    *,
    output_path: str | None = None,
) -> ConversionPlan:
    """Decide how ``assessment``'s video should be normalized."""
    total = assessment.usable_frame_count
    if total <= 0:
        raise RuntimeError(
            "No addressable frames were found, so a frame range cannot be chosen. "
            "The file may be truncated or unsupported."
        )
    first = max(0, int(first_frame))
    last = total - 1 if int(last_frame) < 0 else int(last_frame)
    problem = validate_frame_range(first, last, total)
    if problem:
        raise ValueError(problem)

    probe = assessment.opencv
    if assessment.is_variable_frame_rate:
        # The dominant (median) interval is the rate most frames were shot at, so
        # resampling to it duplicates a few frames instead of dropping many.
        frame_mapping = "resampled"
        raw_fps = probe.measured_fps or probe.fps or assessment.source.nominal_fps
    else:
        # CAP_PROP_FPS comes from the container's own rational, so it is exact.
        frame_mapping = "exact"
        raw_fps = probe.fps or assessment.source.nominal_fps or probe.measured_fps

    if raw_fps <= 0.0:
        raise RuntimeError(
            "The frame rate of this video could not be determined, so it cannot be "
            "normalized to a constant frame rate."
        )
    target_rate = rational_frame_rate(raw_fps)

    notes: list[str] = []
    if frame_mapping == "resampled":
        notes.append(
            "The variable frame rate was resampled to a constant "
            f"{float(target_rate):.6f} fps, so converted frame indices follow time, "
            "not the source frame order."
        )

    tone_map = False
    if assessment.is_hdr:
        if can_tone_map():
            tone_map = True
            notes.append(
                "HDR was tone mapped to 8-bit BT.709 SDR (zscale linear -> Hable "
                "tonemap, desaturation 0). Absolute luminance is not preserved."
            )
        else:
            notes.append(
                "This FFmpeg build has no zscale/tonemap filter, so HDR could only be "
                "converted by matrix and range. Contrast may differ from the source."
            )
    elif assessment.source.bit_depth > 8:
        notes.append(
            f"{assessment.source.bit_depth}-bit video was reduced to 8-bit "
            f"{ANALYSIS_PIX_FMT}."
        )
    if assessment.source.rotation_degrees:
        notes.append(
            f"{assessment.source.rotation_degrees} deg rotation metadata was applied to "
            "the pixels; the analysis copy carries no rotation metadata."
        )

    if output_path is None:
        output_path = conversion_output_path(assessment.path, first, last, total)

    return ConversionPlan(
        source_path=os.path.abspath(assessment.path),
        output_path=os.path.abspath(output_path),
        first_frame=first,
        last_frame=last,
        target_frame_rate=target_rate,
        tone_map=tone_map,
        frame_mapping=frame_mapping,
        assessment=assessment,
        notes=tuple(notes),
    )


def build_video_filters(plan: ConversionPlan) -> list[str]:
    """Build the filter chain that turns the source into the analysis copy."""
    filters: list[str] = [
        # Trimming on frame numbers keeps the range frame-exact where a time-based
        # -ss would not be, and -- unlike select -- it moves the end of the stream
        # too, so the fps filter below does not pad the tail with duplicates.
        f"trim=start_frame={plan.first_frame}:end_frame={plan.last_frame + 1}",
        "setpts=PTS-STARTPTS",
    ]
    if plan.tone_map:
        filters.extend(
            [
                "zscale=t=linear:npl=100",
                "format=gbrpf32le",
                "zscale=p=bt709",
                "tonemap=tonemap=hable:desat=0",
                "zscale=t=bt709:m=bt709:r=tv",
            ]
        )
    elif plan.assessment is not None and plan.assessment.source.is_wide_gamut:
        filters.append("scale=in_color_matrix=bt2020:out_color_matrix=bt709")
    filters.append(f"fps={plan.frame_rate_argument}")
    filters.append(f"format={ANALYSIS_PIX_FMT}")
    # H.264 needs even dimensions; the source resolution is otherwise untouched.
    filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    return filters


def build_ffmpeg_command(plan: ConversionPlan) -> list[str]:
    """Return the full FFmpeg argument list for ``plan``."""
    return [
        ffmpeg_executable(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        plan.source_path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        ",".join(build_video_filters(plan)),
        "-c:v",
        "libx264",
        "-preset",
        ANALYSIS_PRESET,
        "-crf",
        str(ANALYSIS_CRF),
        "-pix_fmt",
        ANALYSIS_PIX_FMT,
        "-g",
        str(ANALYSIS_GOP),
        "-keyint_min",
        str(ANALYSIS_GOP),
        "-r",
        plan.frame_rate_argument,
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        "-progress",
        "pipe:1",
        "-nostats",
        plan.output_path,
    ]


# --------------------------------------------------------------------------
# Running the conversion
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ConversionResult:
    plan: ConversionPlan
    output_path: str
    metadata_path: str
    frames_written: int
    elapsed_seconds: float
    warnings: tuple[str, ...] = ()


def _amadeus_version() -> str:
    try:
        return (Path(__file__).resolve().parent.parent / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def conversion_metadata(
    plan: ConversionPlan,
    *,
    command: Sequence[str],
    frames_written: int,
    warnings: Sequence[str] = (),
) -> dict:
    """Everything needed to reproduce and audit one conversion."""
    assessment = plan.assessment
    source = assessment.source if assessment is not None else SourceVideoInfo(path=plan.source_path)
    probe = assessment.opencv if assessment is not None else OpenCvProbe()
    return {
        "tool": "AMADEUS video compatibility conversion",
        "amadeus_version": _amadeus_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ffmpeg_version": ffmpeg_version(),
        "ffmpeg_binary": ffmpeg_executable(),
        "ffmpeg_pinned_requirement": PINNED_FFMPEG_REQUIREMENT,
        "source_video": {
            **source.as_dict(),
            "path": plan.source_path,
            "opencv": probe.as_dict(),
            "unmodified": True,
        },
        "converted_video": {
            "path": plan.output_path,
            "container": "mp4",
            "codec": "libx264",
            "crf": ANALYSIS_CRF,
            "preset": ANALYSIS_PRESET,
            "pix_fmt": ANALYSIS_PIX_FMT,
            "bit_depth": 8,
            "gop": ANALYSIS_GOP,
            "fps": plan.target_fps,
            "frame_rate_rational": plan.frame_rate_argument,
            "frame_rate_mode": "constant",
            "frames_written": int(frames_written),
            "frames_expected": plan.expected_output_frames,
            "tone_mapped": plan.tone_map,
            "rotation_applied_degrees": source.rotation_degrees,
        },
        "frame_range": {
            "source_first_frame": plan.first_frame,
            "source_last_frame": plan.last_frame,
            "source_frame_count": plan.frame_count,
            "source_usable_frame_count": assessment.usable_frame_count if assessment else 0,
        },
        "frame_mapping": {
            "kind": plan.frame_mapping,
            "converted_frame_0_source_frame": plan.first_frame,
            "formula": (
                f"source_frame = converted_frame + {plan.first_frame}"
                if plan.frame_mapping == "exact"
                else (
                    f"source_time_s = converted_frame * {plan.target_frame_rate.denominator}"
                    f" / {plan.target_frame_rate.numerator}"
                    f" + source_frame_{plan.first_frame}_time"
                )
            ),
            "exact": plan.frame_mapping == "exact",
        },
        "conversion_notes": list(plan.notes),
        "assessment": {
            "status": assessment.status if assessment else "unknown",
            "reasons": list(assessment.reasons) if assessment else [],
            "notes": list(assessment.notes) if assessment else [],
        },
        "ffmpeg_command": list(command),
        "warnings": list(warnings),
    }


def write_conversion_metadata(plan: ConversionPlan, metadata: dict) -> str:
    path = plan.metadata_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def read_conversion_metadata(converted_path: str) -> dict | None:
    """Return the conversion record for an analysis copy, if there is one."""
    path = metadata_path_for(converted_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def source_frame_for_converted_frame(metadata: dict, converted_frame: int) -> int:
    """Map a frame of the analysis copy back onto the source video."""
    mapping = (metadata or {}).get("frame_mapping", {})
    offset = int(mapping.get("converted_frame_0_source_frame", 0) or 0)
    return offset + max(0, int(converted_frame))


def find_existing_conversions(source_path: str) -> list[dict]:
    """Return metadata for analysis copies already made from ``source_path``."""
    directory = conversion_directory_for(source_path)
    if not os.path.isdir(directory):
        return []
    source_key = os.path.normcase(os.path.abspath(str(source_path)))
    found: list[dict] = []
    for entry in sorted(os.listdir(directory)):
        if not entry.endswith(METADATA_EXTENSION):
            continue
        try:
            with open(os.path.join(directory, entry), "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        recorded = str(data.get("source_video", {}).get("path", ""))
        if os.path.normcase(os.path.abspath(recorded)) != source_key:
            continue
        converted = str(data.get("converted_video", {}).get("path", ""))
        if converted and os.path.isfile(converted):
            found.append(data)
    return found


def _verify_conversion(plan: ConversionPlan) -> tuple[int, list[str]]:
    """Open the analysis copy and confirm it behaves the way AMADEUS needs."""
    warnings: list[str] = []
    try:
        cv2 = _cv2()
    except ImportError:
        return 0, ["OpenCV is unavailable, so the converted video was not verified."]

    cap = cv2.VideoCapture(plan.output_path)
    if not cap.isOpened():
        cap.release()
        _delete_quietly(plan.output_path)
        raise RuntimeError(
            "The converted video could not be opened by OpenCV. The conversion did "
            "not produce a usable analysis video."
        )
    try:
        frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        _delete_quietly(plan.output_path)
        raise RuntimeError("The converted video's first frame could not be decoded.")

    expected = plan.expected_output_frames
    if frames and abs(frames - expected) > max(1, int(expected * plan.frame_count_tolerance)):
        warnings.append(
            f"The converted video holds {frames} frames; {expected} were expected."
        )

    seekable = detect_seekable_frame_count(plan.output_path, frames)
    if frames and seekable < frames:
        warnings.append(
            f"Only {seekable} of {frames} frames in the converted video can be reached "
            "by seeking."
        )
    return frames, warnings


def run_conversion(
    plan: ConversionPlan,
    *,
    progress: Callable[[float, int], None] | None = None,
    cancel: threading.Event | None = None,
    log: Callable[[str], None] | None = None,
) -> ConversionResult:
    """Produce the analysis copy described by ``plan``.

    ``progress`` receives ``(fraction, frames_done)``. Setting ``cancel`` stops
    FFmpeg and removes the partial file. The source video is never written to.
    """
    if os.path.normcase(plan.output_path) == os.path.normcase(plan.source_path):
        raise RuntimeError("The converted video would overwrite the source video.")

    command = build_ffmpeg_command(plan)
    os.makedirs(os.path.dirname(plan.output_path), exist_ok=True)
    if log is not None:
        log("FFmpeg: " + " ".join(command))

    expected = max(1, plan.expected_output_frames)
    started = time.perf_counter()
    frames_done = 0

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_subprocess_kwargs(),
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if cancel is not None and cancel.is_set():
                process.terminate()
                break
            line = line.strip()
            if line.startswith("frame="):
                try:
                    frames_done = max(frames_done, int(line.split("=", 1)[1]))
                except ValueError:
                    continue
                if progress is not None:
                    progress(min(1.0, frames_done / expected), frames_done)
        stderr_text = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    if cancel is not None and cancel.is_set():
        _delete_quietly(plan.output_path)
        raise ConversionCancelled("The conversion was stopped before it finished.")
    if return_code != 0:
        _delete_quietly(plan.output_path)
        detail = (stderr_text or "").strip() or f"FFmpeg exited with code {return_code}"
        raise RuntimeError(f"The video conversion failed.\n\n{detail}")

    frames_written, warnings = _verify_conversion(plan)
    metadata = conversion_metadata(
        plan, command=command, frames_written=frames_written or frames_done, warnings=warnings
    )
    metadata_path = write_conversion_metadata(plan, metadata)
    if progress is not None:
        progress(1.0, frames_written or frames_done)
    return ConversionResult(
        plan=plan,
        output_path=plan.output_path,
        metadata_path=metadata_path,
        frames_written=frames_written or frames_done,
        elapsed_seconds=time.perf_counter() - started,
        warnings=tuple(warnings),
    )


def _delete_quietly(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def config_conversion_record(metadata: dict | None, metadata_path: str = "") -> dict:
    """Condense a conversion record for storage inside a GUI config file."""
    if not metadata:
        return {}
    source = metadata.get("source_video", {})
    converted = metadata.get("converted_video", {})
    frame_range = metadata.get("frame_range", {})
    mapping = metadata.get("frame_mapping", {})
    return {
        "source_video_path": source.get("path", ""),
        "source_codec": source.get("codec", ""),
        "source_container": source.get("container", ""),
        "source_fps": source.get("nominal_fps", 0.0),
        "source_bit_depth": source.get("bit_depth", 0),
        "source_hdr": bool(source.get("hdr", False)),
        "converted_video_path": converted.get("path", ""),
        "converted_fps": converted.get("fps", 0.0),
        "converted_codec": converted.get("codec", ""),
        "source_first_frame": frame_range.get("source_first_frame", 0),
        "source_last_frame": frame_range.get("source_last_frame", 0),
        "frame_mapping": mapping.get("formula", ""),
        "frame_mapping_exact": bool(mapping.get("exact", True)),
        "conversion_metadata_path": str(metadata_path or ""),
        "conversion_notes": list(metadata.get("conversion_notes", [])),
    }
