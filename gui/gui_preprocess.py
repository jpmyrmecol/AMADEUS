# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

try:
    from .project_paths import PROJECT_ROOT, ensure_import_paths, gui_asset, gui_script
    from .canvas_file_drop import install_canvas_file_drop
    from .window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
except ImportError:
    from project_paths import PROJECT_ROOT, ensure_import_paths, gui_asset, gui_script
    from canvas_file_drop import install_canvas_file_drop
    from window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon

ensure_import_paths(PROJECT_ROOT)
from gui.color import CYAN_RGB, GREEN_RGB, ORANGE_RGB, WHITE_RGB
from gui.video_input import (
    VIDEO_DROP_SUFFIXES,
    ask_open_analysis_video,
    prepare_analysis_video,
    video_conversion_record,
)
from main.video_compat import ffmpeg_executable
from main.video_frame_count import detect_seekable_frame_count


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(str(gui_asset("deep_green.json")))

APP_TITLE = "AMADEUS: Preprocess"
WINDOW_W = 1600
WINDOW_H = 980
WINDOW_MIN_W = 1200
WINDOW_MIN_H = 820
CANVAS_BG = "black"

BRIGHTNESS_MIN = -1.0
BRIGHTNESS_MAX = 1.0
CONTRAST_MIN = 0.0
CONTRAST_MAX = 2.0
OUTPUT_SPEED_MIN = 0.1
OUTPUT_SPEED_MAX = 100.0
MIN_CROP_SIZE = 16
HANDLE_HIT_PX = 8
HANDLE_DRAW_PX = 7
FRAME_CACHE_SIZE = 12
VERIFY_MAD_THRESHOLD = 3.0
GRAYSCALE_MEAN_SPREAD_THRESHOLD = 1.0
GRAYSCALE_P99_SPREAD_THRESHOLD = 3.0

PANEL_FG = "#242424"
PANEL_BORDER = "#3d3d3d"
ENTRY_BG = "#343638"
TEXT_COLOR = "#dce4ee"
MUTED_TEXT = "#9aa4ad"
TIMELINE_BG = "#181818"
TIMELINE_RANGE = "#1f8040"
TIMELINE_LINE = "#f0a000"

cv2 = None  # type: ignore[assignment]
np = None  # type: ignore[assignment]
Image = None  # type: ignore[assignment]
ImageTk = None  # type: ignore[assignment]

_SPIN_CFG = dict(
    bg=ENTRY_BG,
    fg=TEXT_COLOR,
    insertbackground=TEXT_COLOR,
    buttonbackground="#565b5e",
    relief="flat",
    highlightthickness=2,
    highlightbackground="#565b5e",
    highlightcolor="#1f8040",
    selectbackground="#1f6aa5",
    selectforeground="white",
)


def _maximize_window_once(window: tk.Tk) -> None:
    try:
        window.state("zoomed")
        window.update_idletasks()
        if str(window.state()) == "zoomed":
            return
    except Exception:
        pass

    try:
        window.attributes("-zoomed", True)
        window.update_idletasks()
        return
    except Exception:
        pass

    try:
        screen_w = window.winfo_screenwidth()
        screen_h = window.winfo_screenheight()
        window.geometry(f"{screen_w}x{screen_h}+0+0")
    except Exception:
        pass


def _start_maximized(window: tk.Tk) -> None:
    _maximize_window_once(window)
    for delay_ms in (50, 250, 1000):
        window.after(delay_ms, lambda w=window: _maximize_window_once(w))


def _ensure_video_modules() -> None:
    global cv2, np, Image, ImageTk
    if cv2 is not None and np is not None and Image is not None and ImageTk is not None:
        return
    import cv2 as _cv2
    import numpy as _np
    from PIL import Image as _Image, ImageTk as _ImageTk

    cv2 = _cv2
    np = _np
    Image = _Image
    ImageTk = _ImageTk


def ffmpeg_exe() -> str:
    """The pinned FFmpeg build, shared with the video compatibility layer.

    Raises FfmpegUnavailableError, whose message explains the fix; start_export()
    shows it instead of letting a traceback end the export.
    """
    return ffmpeg_executable()


HARDWARE_ENCODER_LABELS = {
    "h264_videotoolbox": "Apple VideoToolbox",
    "h264_nvenc": "NVIDIA NVENC",
    "h264_qsv": "Intel Quick Sync",
    "h264_amf": "AMD AMF",
}


def _hardware_encoder_candidates() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("h264_videotoolbox",)
    if os.name == "nt":
        return ("h264_nvenc", "h264_qsv", "h264_amf")
    return ("h264_nvenc", "h264_qsv")


def _video_encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
            "-rc:v", "vbr", "-cq:v", "12", "-b:v", "0", "-pix_fmt", "yuv420p",
        ]
    if encoder == "h264_qsv":
        return [
            "-c:v", "h264_qsv", "-preset", "medium",
            "-global_quality:v", "12", "-pix_fmt", "nv12",
        ]
    if encoder == "h264_amf":
        return [
            "-c:v", "h264_amf", "-quality", "quality", "-rc:v", "cqp",
            "-qp_i", "12", "-qp_p", "12", "-qp_b", "12", "-pix_fmt", "yuv420p",
        ]
    if encoder == "h264_videotoolbox":
        return [
            "-c:v", "h264_videotoolbox", "-q:v", "85",
            "-allow_sw", "0", "-pix_fmt", "yuv420p",
        ]
    return ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]


def _hardware_ffmpeg_candidates() -> tuple[str, ...]:
    candidates: list[str] = []
    try:
        candidates.append(ffmpeg_exe())
    except Exception:
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        candidates.append(system_ffmpeg)

    unique: list[str] = []
    seen: set[str] = set()
    for executable in candidates:
        key = os.path.normcase(os.path.realpath(os.path.abspath(executable)))
        if key not in seen:
            seen.add(key)
            unique.append(executable)
    return tuple(unique)


def _detect_hardware_video_encoder() -> tuple[str, str] | None:
    executables = _hardware_ffmpeg_candidates()
    for encoder in _hardware_encoder_candidates():
        for executable in executables:
            cmd = [
                executable,
                "-hide_banner",
                "-loglevel", "error",
                "-f", "lavfi",
                "-i", "color=s=128x128:r=30:d=0.2",
                "-frames:v", "2",
                *_video_encoder_args(encoder),
                "-f", "null",
                "-",
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if completed.returncode == 0:
                return executable, encoder
    return None


def _rgb_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _fourcc_to_string(value: float) -> str:
    code = int(value or 0)
    if code <= 0:
        return "unknown"
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    text = "".join(ch for ch in chars if ch.isprintable()).strip()
    return text or "unknown"


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000.0))
    if ms >= 1000:
        whole += 1
        ms -= 1000
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _build_eq_lut(brightness: float, contrast: float) -> np.ndarray:
    _ensure_video_modules()
    brightness = _clamp(float(brightness), BRIGHTNESS_MIN, BRIGHTNESS_MAX)
    contrast = _clamp(float(contrast), CONTRAST_MIN, CONTRAST_MAX)
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.clip((x - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0) * 255.0
    return y.astype(np.uint8)


def _process_region_bgr(
    frame_bgr: np.ndarray,
    brightness: float,
    contrast: float,
    color_mode: str,
) -> np.ndarray:
    _ensure_video_modules()
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.LUT(yuv[:, :, 0], _build_eq_lut(brightness, contrast))
    if str(color_mode).lower() == "grayscale":
        return cv2.cvtColor(yuv[:, :, 0], cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def _pad_even_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    _ensure_video_modules()
    h, w = frame_bgr.shape[:2]
    pad_right = w % 2
    pad_bottom = h % 2
    if not pad_right and not pad_bottom:
        return frame_bgr
    return cv2.copyMakeBorder(
        frame_bgr,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def _sanitize_filename(value: str) -> str:
    value = os.path.basename(str(value).strip())
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value or "preprocess"


def _unique_output_path(folder: str, filename: str, reserved: set[str]) -> str:
    stem, ext = os.path.splitext(filename)
    if not ext:
        ext = ".mp4"
    if ext.lower() != ".mp4":
        filename = f"{stem}.mp4"
        stem, ext = os.path.splitext(filename)
    candidate = os.path.abspath(os.path.join(folder, filename))
    index = 1
    while os.path.exists(candidate) or os.path.normcase(candidate) in reserved:
        candidate = os.path.abspath(os.path.join(folder, f"{stem}_{index}{ext}"))
        index += 1
    reserved.add(os.path.normcase(candidate))
    return candidate


def _rotate_point(x, y, cx, cy, angle):
    radians = math.radians(angle)
    c, s = math.cos(radians), math.sin(radians)
    return cx + c * (x-cx) - s * (y-cy), cy + s * (x-cx) + c * (y-cy)


def _crop_corners(rect, angle):
    x, y, w, h = rect
    return [_rotate_point(px, py, x+w/2, y+h/2, angle)
            for px, py in ((x, y), (x+w, y), (x+w, y+h), (x, y+h))]


def _crop_quarter_turns(angle: float) -> int:
    # At exact 45-degree ties choose the clockwise orientation, without banker's rounding.
    return math.floor(angle / 90.0 + 0.5) % 4


def _crop_rotated_bgr(frame, rect, angle):
    x, y, w, h = rect
    if not angle:
        return frame[y:y+h, x:x+w]
    # Pixel-center convention matches FFmpeg's rotate filter.
    cx, cy = x + (w-1)/2, y + (h-1)/2
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    matrix[:, 2] += np.array([(w-1)/2-cx, (h-1)/2-cy])
    crop = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return np.ascontiguousarray(np.rot90(crop, -_crop_quarter_turns(angle)))


def _normalize_rect(
    rect: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    minimum: int = MIN_CROP_SIZE,
) -> tuple[int, int, int, int]:
    x, y, w, h = (int(round(v)) for v in rect)
    width = max(1, int(width))
    height = max(1, int(height))
    min_w = max(1, min(int(minimum), width))
    min_h = max(1, min(int(minimum), height))
    w = max(min_w, min(w, width))
    h = max(min_h, min(h, height))
    x = max(0, min(x, width - w))
    y = max(0, min(y, height - h))
    return x, y, w, h


def _normalize_square_rect(
    rect: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    size: int | None = None,
) -> tuple[int, int, int, int]:
    x, y, w, h = _normalize_rect(rect, width, height)
    max_size = max(1, min(int(width), int(height)))
    if size is None:
        size = min(w, h)
    size = max(1, min(int(round(size)), max_size))
    if width >= MIN_CROP_SIZE and height >= MIN_CROP_SIZE:
        size = max(MIN_CROP_SIZE, size)
    size = min(size, max_size)
    x = max(0, min(int(x), int(width) - size))
    y = max(0, min(int(y), int(height) - size))
    return x, y, size, size


def _even_crop_rect(rect: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = _normalize_rect(rect, width, height)
    w -= w % 2
    h -= h % 2
    w = max(2, w)
    h = max(2, h)
    if w < MIN_CROP_SIZE and width >= MIN_CROP_SIZE:
        w = MIN_CROP_SIZE
    if h < MIN_CROP_SIZE and height >= MIN_CROP_SIZE:
        h = MIN_CROP_SIZE
    w -= w % 2
    h -= h % 2
    if w <= 0:
        w = 2 if width >= 2 else 1
    if h <= 0:
        h = 2 if height >= 2 else 1
    x = max(0, min(x, width - w))
    y = max(0, min(y, height - h))
    if width >= 2 and height >= 2:
        assert w % 2 == 0 and h % 2 == 0
    return x, y, w, h


class VideoFrameReader:
    def __init__(self, video_path: str):
        _ensure_video_modules()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self._lock = threading.Lock()
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.raw_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.reported_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_count = detect_seekable_frame_count(video_path, self.reported_frame_count)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps_estimated = self.fps <= 0.0
        if self.fps_estimated:
            self.fps = 30.0
        self.codec = _fourcc_to_string(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.rotation_degrees: int | None = None
        self._manual_rotation_degrees = 0
        self.orientation_note = self._configure_orientation()

        first = self._read_uncached(0)
        self.height, self.width = first.shape[:2]
        if self.frame_count <= 0:
            self.frame_count = 1
        self.frame_count_adjusted = self.frame_count != self.reported_frame_count
        self._cache_put(0, first)

    def _configure_orientation(self) -> str:
        if not hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            return "Rotation metadata unavailable; preview uses raw OpenCV orientation."

        try:
            raw_angle = self.cap.get(cv2.CAP_PROP_ORIENTATION_META)
            angle = int(round(float(raw_angle))) % 360
        except Exception:
            return "Rotation metadata could not be read; preview uses raw OpenCV orientation."

        if angle not in {0, 90, 180, 270}:
            return f"Unsupported rotation metadata ({angle} deg); preview uses raw OpenCV orientation."

        self.rotation_degrees = angle
        if angle == 0:
            return "Rotation metadata: 0 deg."

        auto_enabled = False
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            try:
                self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
                auto_enabled = bool(round(float(self.cap.get(cv2.CAP_PROP_ORIENTATION_AUTO))))
            except Exception:
                auto_enabled = False

        if auto_enabled:
            return f"OpenCV orientation auto-rotation enabled ({angle} deg)."

        self._manual_rotation_degrees = angle
        return f"Applied preview rotation from metadata ({angle} deg)."

    def _apply_orientation(self, frame_bgr: np.ndarray) -> np.ndarray:
        angle = int(self._manual_rotation_degrees) % 360
        if angle == 90:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(frame_bgr, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame_bgr

    def _read_uncached(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(0, min(max(0, self.frame_count - 1), int(frame_idx)))
        with self._lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")
        return self._apply_orientation(frame)

    def _cache_put(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        self.cache[int(frame_idx)] = frame_bgr
        self.cache.move_to_end(int(frame_idx))
        while len(self.cache) > FRAME_CACHE_SIZE:
            self.cache.popitem(last=False)

    def read_bgr(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(0, min(max(0, self.frame_count - 1), int(frame_idx)))
        cached = self.cache.get(frame_idx)
        if cached is not None:
            self.cache.move_to_end(frame_idx)
            return cached.copy()
        frame = self._read_uncached(frame_idx)
        self._cache_put(frame_idx, frame)
        return frame.copy()

    def close(self) -> None:
        self.cap.release()


@dataclass
class Region:
    uid: int
    kind: str
    rect: tuple[int, int, int, int] | None = None
    brightness: float = 0.0
    contrast: float = 1.0
    color_mode: str = "color"
    square: bool = False
    export_enabled: bool = True
    angle: float = 0.0  # Clockwise degrees about the crop center in source coordinates.


@dataclass(frozen=True)
class ExportRegion:
    display_name: str
    token: str
    rect: tuple[int, int, int, int] | None
    brightness: float
    contrast: float
    color_mode: str
    output_path: str
    angle: float = 0.0
    quarter_turns: int = 0


@dataclass(frozen=True)
class ExportJob:
    ffmpeg: str
    video_path: str
    output_folder: str
    fps: float
    output_fps: float
    output_speed: float
    video_encoder: str
    frame_count: int
    source_width: int
    source_height: int
    in_frame: int
    out_frame: int
    regions: tuple[ExportRegion, ...]


class ExportCancelled(RuntimeError):
    pass


class PreprocessApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        configure_dpi_scaling(self)
        install_window_icon(self)
        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.minsize(WINDOW_MIN_W, WINDOW_MIN_H)

        self.video_path_var = tk.StringVar()
        self.meta_var = tk.StringVar(value="No video loaded.")
        self.status_var = tk.StringVar(value="Select a video file.")
        self.active_region_var = tk.StringVar(value="Full image")
        self.frame_label_var = tk.StringVar(value="frame 0 / 0   00:00.000")
        self.trim_label_var = tk.StringVar(value="in 0 / out 0 / length 0 frames")
        self.in_frame_var = tk.StringVar(value="0")
        self.out_frame_var = tk.StringVar(value="0")
        self.output_folder_var = tk.StringVar()
        self.template_var = tk.StringVar(value="{stem}_{region}")
        self.source_output_info_var = tk.StringVar(value="fps —  |  total frames —")
        self.output_speed_var = tk.DoubleVar(value=1.0)
        self.output_frame_count_var = tk.StringVar(value="total frames —")
        self.gpu_acceleration_var = tk.BooleanVar(value=False)
        self.brightness_var = tk.DoubleVar(value=0.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.color_mode_var = tk.StringVar(value="Color")
        self.output_fps_var = tk.DoubleVar(value=30.0)
        self.crop_label_var = tk.StringVar(value="No crop selected.")
        self.crop_x_var = tk.IntVar(value=0)
        self.crop_y_var = tk.IntVar(value=0)
        self.crop_w_var = tk.IntVar(value=0)
        self.crop_h_var = tk.IntVar(value=0)
        self.crop_angle_var = tk.DoubleVar(value=0.0)
        self.export_view_rotation_var = tk.BooleanVar(value=True)
        self.crop_square_var = tk.BooleanVar(value=False)
        self.progress_var = tk.DoubleVar(value=0.0)

        self.reader: VideoFrameReader | None = None
        self.frame_bgr_cache: np.ndarray | None = None
        self.current_frame = 0
        self.in_frame = 0
        self.out_frame = 0
        self.default_color_mode = "color"
        self.regions: list[Region] = [Region(uid=0, kind="full")]
        self.active_region_idx = 0
        self.editing_crop_idx: int | None = None
        self.next_region_uid = 1
        self.region_rows: list[ctk.CTkFrame] = []
        self.region_export_checks: list[ctk.CTkCheckBox] = []
        self.region_select_buttons: list[ctk.CTkButton] = []
        self.region_export_vars: list[tk.BooleanVar] = []

        self.tk_image: ImageTk.PhotoImage | None = None
        self.view_initialized = False
        self.zoom_scale = 1.0
        self.preview_turns = 0
        self.view_scale = 1.0
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0
        self.view_source_shape: tuple[int, int] | None = None
        self.drag_state: dict[str, object] | None = None
        self.cursor_position = None

        self.playback_active = False
        self.playback_job: str | None = None
        self.status_clear_job: str | None = None
        self.config_save_job: str | None = None
        self.export_queue: queue.Queue = queue.Queue()
        self.hardware_encoder_queue: queue.Queue = queue.Queue()
        self.hardware_video_encoder: str | None = None
        self.hardware_video_ffmpeg: str | None = None
        self.export_thread: threading.Thread | None = None
        self.export_cancel = threading.Event()
        self.export_proc: subprocess.Popen | None = None
        self.export_proc_lock = threading.Lock()
        self.export_poll_job: str | None = None
        self.export_running = False
        self._syncing_adjustments = False
        self._syncing_crop_controls = False
        self._syncing_trim_inputs = False
        self._last_progress_update = 0.0
        self.output_folder_var.trace_add("write", lambda *_: self._schedule_crop_trimming_config_save())
        self.template_var.trace_add("write", lambda *_: self._schedule_crop_trimming_config_save())

        self._build_ui()
        self._bind_events()
        self._refresh_region_list(select_index=0)
        self._sync_adjustment_controls_from_region()
        self._refresh_video_controls()
        self._start_hardware_encoder_detection()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        _start_maximized(self)

    def _maximize_window(self) -> None:
        _maximize_window_once(self)

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0, minsize=330)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=360)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)

        self.left_pane = ctk.CTkFrame(self, width=330, border_width=1, corner_radius=4)
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self.left_pane.grid_propagate(False)

        self.center_pane = ctk.CTkFrame(self, corner_radius=0)
        self.center_pane.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        self.center_pane.grid_columnconfigure(0, weight=1)
        self.center_pane.grid_rowconfigure(0, weight=1)

        self.right_pane = ctk.CTkFrame(self, width=360, border_width=1, corner_radius=4)
        self.right_pane.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)
        self.right_pane.grid_propagate(False)

        self.canvas = tk.Canvas(self.center_pane, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas_drop_state = install_canvas_file_drop(
            self,
            self.canvas,
            allowed_suffixes=VIDEO_DROP_SUFFIXES,
            on_path=self.select_video,
            status_callback=lambda text: self.set_status(text, auto_clear=False),
            label="video",
        )
        view_bar = ctk.CTkFrame(self.center_pane, corner_radius=0)
        view_bar.grid(row=1, column=0, sticky="ew")
        ctk.CTkButton(view_bar, text="Fit window", width=100, command=self.fit_canvas_to_window).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(view_bar, text="Screenshot", width=100, command=self.save_canvas_screenshot).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(view_bar, text="Rotate view 90°", width=120, command=self.rotate_preview).pack(side="left", padx=5, pady=5)
        self.export_view_rotation_check = ctk.CTkCheckBox(
            self.center_pane, text="Apply view rotation to exported videos",
            variable=self.export_view_rotation_var, command=self._schedule_crop_trimming_config_save)
        self.export_view_rotation_check.grid(row=2, column=0, sticky="w", padx=8, pady=(0, 5))

        self.timeline_pane = ctk.CTkFrame(self, corner_radius=0)
        self.timeline_pane.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 6))
        self.timeline_pane.grid_columnconfigure(11, weight=1)

        self.status_label = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w", text_color=MUTED_TEXT)
        self.status_label.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 6))

        self._build_left_pane()
        self._build_right_pane()
        self._build_timeline()

    def _build_left_pane(self) -> None:
        input_frame = ctk.CTkFrame(self.left_pane, corner_radius=6)
        input_frame.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(input_frame, text="Input", font=("TkDefaultFont", 13, "bold"), anchor="w").pack(
            fill="x", padx=8, pady=(7, 2)
        )
        row = ctk.CTkFrame(input_frame, corner_radius=0)
        row.pack(fill="x", padx=8, pady=(2, 6))
        self.video_entry = ctk.CTkEntry(row, textvariable=self.video_path_var)
        self.video_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.browse_video_button = ctk.CTkButton(row, text="Browse...", width=88, command=self.browse_video)
        self.browse_video_button.pack(side="left")
        self.meta_label = ctk.CTkLabel(
            input_frame,
            textvariable=self.meta_var,
            justify="left",
            anchor="w",
            text_color=MUTED_TEXT,
            wraplength=286,
        )
        self.meta_label.pack(fill="x", padx=8, pady=(0, 8))

        region_frame = ctk.CTkFrame(self.left_pane, corner_radius=6)
        region_frame.pack(fill="both", expand=True, padx=10, pady=6)
        ctk.CTkLabel(region_frame, text="Regions", font=("TkDefaultFont", 13, "bold"), anchor="w").pack(
            fill="x", padx=8, pady=(7, 2)
        )
        self.region_scroll = ctk.CTkScrollableFrame(
            region_frame,
            corner_radius=0,
            fg_color="#1e1e1e",
            border_width=1,
            border_color=PANEL_BORDER,
            height=190,
        )
        self.region_scroll.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        button_row = ctk.CTkFrame(region_frame, corner_radius=0)
        button_row.pack(fill="x", padx=8, pady=(0, 8))
        self.add_crop_button = ctk.CTkButton(button_row, text="Add Crop", command=self.add_crop)
        self.add_crop_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.delete_crop_button = ctk.CTkButton(button_row, text="Delete Crop", command=self.delete_crop)
        self.delete_crop_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

        crop_frame = ctk.CTkFrame(region_frame, corner_radius=0)
        crop_frame.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(crop_frame, text="Crop Geometry", font=("TkDefaultFont", 12, "bold"), anchor="w").pack(
            fill="x", pady=(2, 0)
        )
        ctk.CTkLabel(crop_frame, textvariable=self.crop_label_var, anchor="w", text_color=MUTED_TEXT).pack(
            fill="x", pady=(0, 4)
        )
        grid = ctk.CTkFrame(crop_frame, corner_radius=0)
        grid.pack(fill="x")
        self.crop_x_spin = self._pack_int_spinbox(grid, "X", self.crop_x_var, 0, self._on_crop_x_changed, row=0, col=0)
        self.crop_y_spin = self._pack_int_spinbox(grid, "Y", self.crop_y_var, 0, self._on_crop_y_changed, row=0, col=1)
        self.crop_w_spin = self._pack_int_spinbox(grid, "W", self.crop_w_var, MIN_CROP_SIZE, self._on_crop_w_changed, row=1, col=0)
        self.crop_h_spin = self._pack_int_spinbox(grid, "H", self.crop_h_var, MIN_CROP_SIZE, self._on_crop_h_changed, row=1, col=1)
        angle_cell = ctk.CTkFrame(grid, corner_radius=0)
        angle_cell.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        ctk.CTkLabel(angle_cell, text="Angle (°, clockwise)").pack(side="left")
        self.crop_angle_spin = tk.Spinbox(angle_cell, from_=-180, to=180, increment=0.1,
                                         textvariable=self.crop_angle_var, width=8, **_SPIN_CFG)
        self.crop_angle_spin.pack(side="left", fill="x", expand=True)
        self.crop_angle_spin.configure(command=self._on_crop_angle_changed)
        for event in ("<Return>", "<KP_Enter>", "<FocusOut>"):
            self.crop_angle_spin.bind(event, self._on_crop_angle_changed)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)
        self.crop_square_check = ctk.CTkCheckBox(
            crop_frame,
            text="Square",
            variable=self.crop_square_var,
            command=self._on_crop_square_changed,
        )
        self.crop_square_check.pack(anchor="w", pady=(6, 0))

    def _build_right_pane(self) -> None:
        adjust = ctk.CTkFrame(self.right_pane, corner_radius=6)
        adjust.pack(fill="x", padx=10, pady=(10, 6))
        ctk.CTkLabel(adjust, text="Adjustment", font=("TkDefaultFont", 13, "bold"), anchor="w").pack(
            fill="x", padx=8, pady=(7, 2)
        )
        ctk.CTkLabel(adjust, textvariable=self.active_region_var, anchor="w", text_color=MUTED_TEXT).pack(
            fill="x", padx=8, pady=(0, 4)
        )
        self.brightness_slider, self.brightness_entry = self._pack_float_slider(
            adjust,
            "Brightness",
            self.brightness_var,
            BRIGHTNESS_MIN,
            BRIGHTNESS_MAX,
            0.01,
            self._on_brightness_changed,
        )
        self.contrast_slider, self.contrast_entry = self._pack_float_slider(
            adjust,
            "Contrast",
            self.contrast_var,
            CONTRAST_MIN,
            CONTRAST_MAX,
            0.01,
            self._on_contrast_changed,
        )
        mode_row = ctk.CTkFrame(adjust, corner_radius=0)
        mode_row.pack(fill="x", padx=8, pady=(4, 4))
        ctk.CTkLabel(mode_row, text="Color mode", width=110, anchor="w").pack(side="left")
        self.color_mode_combo = ctk.CTkComboBox(
            mode_row,
            values=["Color", "Grayscale"],
            variable=self.color_mode_var,
            command=self._on_color_mode_changed,
            state="readonly",
            width=130,
        )
        self.color_mode_combo.pack(side="right")
        self.reset_adjust_button = ctk.CTkButton(adjust, text="Reset", command=self.reset_adjustment)
        self.reset_adjust_button.pack(fill="x", padx=8, pady=(4, 8))

        export = ctk.CTkFrame(self.right_pane, corner_radius=6)
        export.pack(fill="both", expand=True, padx=10, pady=6)
        ctk.CTkLabel(export, text="Export", font=("TkDefaultFont", 13, "bold"), anchor="w").pack(
            fill="x", padx=8, pady=(7, 2)
        )
        out_row = ctk.CTkFrame(export, corner_radius=0)
        out_row.pack(fill="x", padx=8, pady=(2, 6))
        self.output_entry = ctk.CTkEntry(out_row, textvariable=self.output_folder_var)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.output_button = ctk.CTkButton(out_row, text="Output folder...", width=118, command=self.browse_output_folder)
        self.output_button.pack(side="left")

        ctk.CTkLabel(export, text="Name template", anchor="w", text_color=MUTED_TEXT).pack(
            fill="x", padx=8, pady=(2, 0)
        )
        self.template_entry = ctk.CTkEntry(export, textvariable=self.template_var)
        self.template_entry.pack(fill="x", padx=8, pady=(0, 6))

        source_row = ctk.CTkFrame(export, corner_radius=0)
        source_row.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(source_row, text="Source", width=58, anchor="w").pack(side="left")
        ctk.CTkLabel(
            source_row,
            textvariable=self.source_output_info_var,
            anchor="e",
            text_color=MUTED_TEXT,
        ).pack(side="right", fill="x", expand=True)

        ctk.CTkLabel(export, text="↓", height=18, text_color=MUTED_TEXT).pack(fill="x", padx=8)

        fps_row = ctk.CTkFrame(export, corner_radius=0)
        fps_row.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(fps_row, text="Output", width=58, anchor="w").pack(side="left")
        self.output_speed_spin = tk.Spinbox(
            fps_row,
            from_=OUTPUT_SPEED_MIN,
            to=OUTPUT_SPEED_MAX,
            increment=0.1,
            width=6,
            format="%.2f",
            **_SPIN_CFG,
        )
        self.output_speed_spin.pack(side="left")
        self.output_speed_spin.delete(0, tk.END)
        self.output_speed_spin.insert(0, f"{self.output_speed_var.get():.2f}")
        self.output_speed_spin.configure(command=self._commit_output_speed)
        self.output_speed_spin.bind("<Return>", self._commit_output_speed, add="+")
        self.output_speed_spin.bind("<KP_Enter>", self._commit_output_speed, add="+")
        self.output_speed_spin.bind("<FocusOut>", self._commit_output_speed, add="+")
        ctk.CTkLabel(fps_row, text="x", width=16, anchor="w").pack(side="left", padx=(2, 4))
        ctk.CTkLabel(fps_row, text="fps", width=24, anchor="e").pack(side="left", padx=(0, 4))
        self.output_fps_spin = tk.Spinbox(
            fps_row,
            from_=1.0,
            to=240.0,
            increment=0.1,
            width=9,
            format="%.3f",
            **_SPIN_CFG,
        )
        self.output_fps_spin.pack(side="left")
        self.output_fps_spin.delete(0, tk.END)
        self.output_fps_spin.insert(0, f"{self.output_fps_var.get():.3f}")
        self.output_fps_spin.configure(command=self._commit_output_fps)
        self.output_fps_spin.bind("<Return>", self._commit_output_fps, add="+")
        self.output_fps_spin.bind("<KP_Enter>", self._commit_output_fps, add="+")
        self.output_fps_spin.bind("<FocusOut>", self._commit_output_fps, add="+")
        ctk.CTkLabel(
            fps_row,
            textvariable=self.output_frame_count_var,
            anchor="e",
            text_color=MUTED_TEXT,
        ).pack(side="right", fill="x", expand=True, padx=(6, 0))

        self.gpu_acceleration_check = ctk.CTkCheckBox(
            export,
            text="Detecting GPU acceleration...",
            variable=self.gpu_acceleration_var,
            command=self._schedule_crop_trimming_config_save,
            state="disabled",
        )
        self.gpu_acceleration_check.pack(fill="x", padx=8, pady=(0, 8))

        action_row = ctk.CTkFrame(export, corner_radius=0)
        action_row.pack(fill="x", padx=8, pady=(0, 8))
        self.export_button = ctk.CTkButton(action_row, text="Export", command=self.start_export)
        self.export_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.cancel_button = ctk.CTkButton(action_row, text="Cancel", command=self.cancel_export, state="disabled")
        self.cancel_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.progress_bar = ctk.CTkProgressBar(export, mode="determinate", variable=self.progress_var)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=8, pady=(0, 8))

        self.log_box = ctk.CTkTextbox(export, height=220, wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_box.configure(state="disabled")

    def _build_timeline(self) -> None:
        self.first_button = ctk.CTkButton(self.timeline_pane, text="|<", width=42, command=lambda: self.set_frame(0))
        self.first_button.grid(row=0, column=0, padx=(0, 4), pady=(7, 3))
        self.prev_button = ctk.CTkButton(self.timeline_pane, text="<", width=42, command=lambda: self.step_frame(-1))
        self.prev_button.grid(row=0, column=1, padx=4, pady=(7, 3))
        self.play_button = ctk.CTkButton(self.timeline_pane, text=">", width=54, command=self.toggle_playback)
        self.play_button.grid(row=0, column=2, padx=4, pady=(7, 3))
        self.next_button = ctk.CTkButton(self.timeline_pane, text=">", width=42, command=lambda: self.step_frame(1))
        self.next_button.grid(row=0, column=3, padx=4, pady=(7, 3))
        self.last_button = ctk.CTkButton(
            self.timeline_pane,
            text=">|",
            width=42,
            command=lambda: self.set_frame(self._last_frame_index()),
        )
        self.last_button.grid(row=0, column=4, padx=(4, 12), pady=(7, 3))
        self.set_in_button = ctk.CTkButton(self.timeline_pane, text="Set In", width=74, command=self.set_in_frame)
        self.set_in_button.grid(row=0, column=5, padx=4, pady=(7, 3))
        self.set_out_button = ctk.CTkButton(self.timeline_pane, text="Set Out", width=78, command=self.set_out_frame)
        self.set_out_button.grid(row=0, column=6, padx=(4, 10), pady=(7, 3))
        ctk.CTkLabel(self.timeline_pane, text="In frame", width=58, anchor="e").grid(
            row=0, column=7, padx=(0, 3), pady=(7, 3)
        )
        self.in_frame_spin = tk.Spinbox(
            self.timeline_pane,
            from_=0,
            to=0,
            increment=1,
            width=8,
            textvariable=self.in_frame_var,
            **_SPIN_CFG,
        )
        self.in_frame_spin.grid(row=0, column=8, padx=(0, 6), pady=(7, 3))
        ctk.CTkLabel(self.timeline_pane, text="Out frame", width=68, anchor="e").grid(
            row=0, column=9, padx=(0, 3), pady=(7, 3)
        )
        self.out_frame_spin = tk.Spinbox(
            self.timeline_pane,
            from_=0,
            to=0,
            increment=1,
            width=8,
            textvariable=self.out_frame_var,
            **_SPIN_CFG,
        )
        self.out_frame_spin.grid(row=0, column=10, padx=(0, 10), pady=(7, 3))
        for spinbox, which in (
            (self.in_frame_spin, "in"),
            (self.out_frame_spin, "out"),
        ):
            spinbox.configure(command=lambda k=which: self._commit_trim_frame_input(k))
            for event in ("<Return>", "<KP_Enter>", "<FocusOut>"):
                spinbox.bind(
                    event,
                    lambda _event, k=which: self._commit_trim_frame_input(k),
                    add="+",
                )
        self.frame_slider = ctk.CTkSlider(self.timeline_pane, orientation="horizontal", command=self.on_frame_slider)
        self.frame_slider.grid(row=0, column=11, sticky="ew", padx=(0, 8), pady=(7, 3))
        self.frame_label = ctk.CTkLabel(self.timeline_pane, textvariable=self.frame_label_var, width=180, anchor="e")
        self.frame_label.grid(row=0, column=12, sticky="e", pady=(7, 3))
        self.trim_canvas = tk.Canvas(self.timeline_pane, height=18, bg=TIMELINE_BG, highlightthickness=0)
        self.trim_canvas.grid(row=1, column=11, sticky="ew", padx=(0, 8), pady=(0, 7))
        self.trim_label = ctk.CTkLabel(self.timeline_pane, textvariable=self.trim_label_var, anchor="w", text_color=MUTED_TEXT)
        self.trim_label.grid(row=1, column=0, columnspan=11, sticky="ew", padx=(0, 10), pady=(0, 7))

    def _pack_float_slider(
        self,
        parent: ctk.CTkFrame,
        label: str,
        var: tk.DoubleVar,
        lo: float,
        hi: float,
        step: float,
        command,
    ) -> tuple[ctk.CTkSlider, tk.Spinbox]:
        default_value = _clamp(float(var.get()), lo, hi)
        outer = ctk.CTkFrame(parent, corner_radius=0)
        outer.pack(fill="x", padx=8, pady=4)
        row = ctk.CTkFrame(outer, corner_radius=0)
        row.pack(fill="x")
        ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
        spinbox = tk.Spinbox(row, from_=lo, to=hi, increment=step, width=8, format="%.2f", **_SPIN_CFG)
        spinbox.pack(side="right")
        slider = ctk.CTkSlider(outer, from_=lo, to=hi, orientation="horizontal", command=command)
        slider.set(float(var.get()))
        slider.pack(fill="x", pady=(2, 0))

        def commit(_event=None):
            try:
                value = float(spinbox.get().strip())
            except Exception:
                value = float(var.get())
            value = round(_clamp(value, lo, hi) / step) * step
            value = _clamp(value, lo, hi)
            var.set(value)
            slider.set(value)
            spinbox.delete(0, tk.END)
            spinbox.insert(0, f"{value:.2f}")
            command(value)
            return "break"

        def sync_from_var(*_):
            value = float(var.get())
            spinbox.delete(0, tk.END)
            spinbox.insert(0, f"{value:.2f}")
            slider.set(value)

        def reset_to_default(_event=None):
            try:
                if str(slider.cget("state")) == "disabled":
                    return "break"
            except Exception:
                pass
            value = default_value
            var.set(value)
            slider.set(value)
            spinbox.delete(0, tk.END)
            spinbox.insert(0, f"{value:.2f}")
            command(value)
            return "break"

        spinbox.configure(command=commit)
        spinbox.bind("<Return>", commit, add="+")
        spinbox.bind("<KP_Enter>", commit, add="+")
        spinbox.bind("<FocusOut>", commit, add="+")
        slider.bind("<Double-Button-1>", reset_to_default, add="+")
        var.trace_add("write", sync_from_var)
        sync_from_var()
        return slider, spinbox

    def _pack_int_spinbox(
        self,
        parent: ctk.CTkFrame,
        label: str,
        var: tk.IntVar,
        lo: int,
        command,
        *,
        row: int,
        col: int,
    ) -> tk.Spinbox:
        cell = ctk.CTkFrame(parent, corner_radius=0)
        cell.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 5, 0), pady=2)
        ctk.CTkLabel(cell, text=label, width=18, anchor="w").pack(side="left")
        spinbox = tk.Spinbox(cell, from_=lo, to=999999, increment=1, width=7, **_SPIN_CFG)
        spinbox.pack(side="left", fill="x", expand=True)

        def commit(_event=None):
            if self._syncing_crop_controls:
                return "break"
            try:
                value = int(round(float(spinbox.get().strip())))
            except Exception:
                value = int(var.get())
            var.set(max(int(lo), value))
            command()
            return "break"

        def sync_from_var(*_):
            state = str(spinbox.cget("state"))
            if state == "disabled":
                spinbox.configure(state="normal")
            spinbox.delete(0, tk.END)
            spinbox.insert(0, str(int(var.get())))
            if state == "disabled":
                spinbox.configure(state=state)

        spinbox.configure(command=commit)
        spinbox.bind("<Return>", commit, add="+")
        spinbox.bind("<KP_Enter>", commit, add="+")
        spinbox.bind("<FocusOut>", commit, add="+")
        var.trace_add("write", sync_from_var)
        sync_from_var()
        return spinbox

    def _commit_output_speed(self, _event=None):
        try:
            value = float(self.output_speed_spin.get().strip())
        except Exception:
            value = float(self.output_speed_var.get())
        value = _clamp(value, OUTPUT_SPEED_MIN, OUTPUT_SPEED_MAX)
        self.output_speed_var.set(value)
        self.output_speed_spin.delete(0, tk.END)
        self.output_speed_spin.insert(0, f"{value:.2f}")
        self._update_output_summary()
        self._schedule_crop_trimming_config_save()
        return "break"

    def _commit_output_fps(self, _event=None):
        try:
            value = float(self.output_fps_spin.get().strip())
        except Exception:
            value = float(self.output_fps_var.get())
        value = _clamp(value, 1.0, 240.0)
        self.output_fps_var.set(value)
        self.output_fps_spin.delete(0, tk.END)
        self.output_fps_spin.insert(0, f"{value:.3f}")
        self._update_output_summary()
        self._schedule_crop_trimming_config_save()
        return "break"

    def _update_output_summary(self) -> None:
        if self.reader is None:
            self.source_output_info_var.set("fps —  |  total frames —")
            self.output_frame_count_var.set("total frames —")
            return

        source_fps = max(1e-9, float(self.reader.fps))
        source_frames = max(1, int(self.reader.frame_count))
        output_fps = max(1e-9, float(self.output_fps_var.get()))
        output_speed = max(OUTPUT_SPEED_MIN, float(self.output_speed_var.get()))
        selected_frames = max(1, int(self.out_frame) - int(self.in_frame) + 1)
        source_duration = selected_frames / source_fps
        output_duration = source_duration / output_speed
        output_frames = max(1, int(round(output_duration * output_fps)))

        self.source_output_info_var.set(
            f"{source_fps:.3f} fps  |  total {source_frames:,} frames"
        )
        self.output_frame_count_var.set(f"total {output_frames:,} frames")

    def _start_hardware_encoder_detection(self) -> None:
        def worker() -> None:
            self.hardware_encoder_queue.put(_detect_hardware_video_encoder())

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._poll_hardware_encoder_detection)

    def _poll_hardware_encoder_detection(self) -> None:
        try:
            detected = self.hardware_encoder_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_hardware_encoder_detection)
            return

        if detected is None:
            self.hardware_video_encoder = None
            self.hardware_video_ffmpeg = None
            self.gpu_acceleration_var.set(False)
            self.gpu_acceleration_check.configure(
                text="GPU acceleration unavailable",
                state="disabled",
            )
            return

        executable, encoder = detected
        self.hardware_video_encoder = encoder
        self.hardware_video_ffmpeg = executable
        label = HARDWARE_ENCODER_LABELS.get(encoder, encoder)
        self.gpu_acceleration_check.configure(
            text=f"Use GPU acceleration ({label})",
            state="disabled" if self.export_running else "normal",
        )

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", lambda _event: self.fit_canvas_to_window())
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind(event, self.on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press, add="+")
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag, add="+")
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release, add="+")
        self.canvas.bind("<Motion>", self._on_canvas_motion, add="+")
        self.canvas.bind("<Leave>", self._on_canvas_leave, add="+")
        self.frame_slider.bind("<Double-Button-1>", self._on_frame_slider_double_click, add="+")
        self.trim_canvas.bind("<Configure>", lambda _event: self._draw_trim_markers(), add="+")

    def _last_frame_index(self) -> int:
        if self.reader is None:
            return 0
        return max(0, int(self.reader.frame_count) - 1)

    def _active_region(self) -> Region:
        return self.regions[max(0, min(self.active_region_idx, len(self.regions) - 1))]

    def _region_display_name(self, index: int) -> str:
        if index == 0:
            return "Full image"
        return f"Crop {index}"

    def _region_token(self, index: int) -> str:
        if index == 0:
            return "full"
        return f"crop{index}"

    def _refresh_region_list(self, select_index: int | None = None) -> None:
        select_index = self.active_region_idx if select_index is None else int(select_index)
        select_index = max(0, min(select_index, len(self.regions) - 1))
        self.active_region_idx = select_index
        for child in self.region_scroll.winfo_children():
            child.destroy()
        self.region_rows = []
        self.region_export_checks = []
        self.region_select_buttons = []
        self.region_export_vars = []
        state = "disabled" if self.export_running else "normal"
        for index, region in enumerate(self.regions):
            selected = index == select_index
            row_color = "#225c38" if selected else "#1e1e1e"
            button_color = "#1f8040" if selected else "#2b2b2b"
            row = ctk.CTkFrame(self.region_scroll, corner_radius=0, fg_color=row_color)
            row.pack(fill="x", padx=2, pady=1)
            var = tk.BooleanVar(value=bool(region.export_enabled))
            check = ctk.CTkCheckBox(
                row,
                text="",
                width=28,
                variable=var,
                command=lambda i=index, v=var: self._on_region_export_toggled(i, v),
                state=state,
            )
            check.pack(side="left", padx=(6, 2), pady=4)
            button = ctk.CTkButton(
                row,
                text=self._region_display_name(index),
                height=28,
                anchor="w",
                fg_color=button_color,
                hover_color="#2a8d50",
                command=lambda i=index: self._select_region(i),
                state=state,
            )
            button.pack(side="left", fill="x", expand=True, padx=(0, 6), pady=4)
            row.bind("<Button-1>", lambda _event, i=index: self._select_region(i), add="+")
            self.region_rows.append(row)
            self.region_export_checks.append(check)
            self.region_select_buttons.append(button)
            self.region_export_vars.append(var)
        self._sync_adjustment_controls_from_region()

    def _select_region(self, index: int) -> None:
        if not (0 <= int(index) < len(self.regions)):
            return
        self.view_initialized = False
        self.active_region_idx = int(index)
        if self.active_region_idx > 0:
            self.editing_crop_idx = self.active_region_idx
        else:
            self.editing_crop_idx = None
        self._refresh_region_list(select_index=self.active_region_idx)
        self.redraw_current_frame()
        self._refresh_video_controls()

    def _on_region_export_toggled(self, index: int, var: tk.BooleanVar) -> None:
        if not (0 <= int(index) < len(self.regions)):
            return
        self.regions[int(index)].export_enabled = bool(var.get())
        self._refresh_video_controls()
        self._schedule_crop_trimming_config_save()

    def _set_region_list_state(self, state: str) -> None:
        for widget in [*self.region_export_checks, *self.region_select_buttons]:
            widget.configure(state=state)

    def _apply_export_defaults_after_crop_change(self, *, force_crop_exports: bool = False) -> None:
        if len(self.regions) == 1:
            self.regions[0].export_enabled = True
            return
        self.regions[0].export_enabled = False
        if force_crop_exports:
            for region in self.regions[1:]:
                region.export_enabled = True
        elif not any(region.export_enabled for region in self.regions[1:]):
            self.regions[-1].export_enabled = True

    def _sync_adjustment_controls_from_region(self) -> None:
        region = self._adjustment_region()
        index = self._crop_control_index()
        self._syncing_adjustments = True
        try:
            self.active_region_var.set(self._region_display_name(
                self.active_region_idx if index is None else index))
            self.brightness_var.set(float(region.brightness))
            self.contrast_var.set(float(region.contrast))
            self.color_mode_var.set("Grayscale" if region.color_mode == "grayscale" else "Color")
        finally:
            self._syncing_adjustments = False
        self._sync_crop_controls_from_region()

    def _on_brightness_changed(self, value) -> None:
        if self._syncing_adjustments:
            return
        region = self._adjustment_region()
        region.brightness = _clamp(float(value), BRIGHTNESS_MIN, BRIGHTNESS_MAX)
        self.brightness_var.set(region.brightness)
        self.redraw_current_frame()
        self._schedule_crop_trimming_config_save()

    def _on_contrast_changed(self, value) -> None:
        if self._syncing_adjustments:
            return
        region = self._adjustment_region()
        region.contrast = _clamp(float(value), CONTRAST_MIN, CONTRAST_MAX)
        self.contrast_var.set(region.contrast)
        self.redraw_current_frame()
        self._schedule_crop_trimming_config_save()

    def _on_color_mode_changed(self, value) -> None:
        if self._syncing_adjustments:
            return
        mode = "grayscale" if str(value).lower().startswith("gray") else "color"
        self._adjustment_region().color_mode = mode
        self.redraw_current_frame()
        self._schedule_crop_trimming_config_save()

    def reset_adjustment(self) -> None:
        region = self._adjustment_region()
        region.brightness = 0.0
        region.contrast = 1.0
        self._sync_adjustment_controls_from_region()
        self.redraw_current_frame()
        self._schedule_crop_trimming_config_save()

    def _adjustment_region(self) -> Region:
        """Region the appearance controls edit: the same one the crop controls do.

        While the full view is shown with a crop selected, the crop is what gets
        exported, so brightness, contrast and colour mode have to reach it rather
        than the full-image region.
        """
        index = self._crop_control_index()
        return self.regions[index] if index is not None else self._active_region()

    def _crop_control_index(self) -> int | None:
        if self.active_region_idx > 0:
            return self.active_region_idx
        if self.editing_crop_idx is not None and 0 < self.editing_crop_idx < len(self.regions):
            return self.editing_crop_idx
        return None

    def _set_crop_controls_state(self, enabled: bool) -> None:
        state = "normal" if enabled and not self.export_running else "disabled"
        for spinbox in [self.crop_x_spin, self.crop_y_spin, self.crop_w_spin, self.crop_h_spin, self.crop_angle_spin]:
            spinbox.configure(state=state)
        self.crop_square_check.configure(state=state)

    def _sync_crop_controls_from_region(self) -> None:
        index = self._crop_control_index()
        self._syncing_crop_controls = True
        try:
            if index is None or self.reader is None or self.regions[index].rect is None:
                self.crop_label_var.set("No crop selected.")
                self.crop_x_var.set(0)
                self.crop_y_var.set(0)
                self.crop_w_var.set(0)
                self.crop_h_var.set(0)
                self.crop_angle_var.set(0.0)
                self.crop_square_var.set(False)
                self._set_crop_controls_state(False)
                return
            region = self.regions[index]
            x, y, w, h = _normalize_rect(region.rect, self.reader.width, self.reader.height)
            if region.square:
                x, y, w, h = _normalize_square_rect((x, y, w, h), self.reader.width, self.reader.height)
                region.rect = (x, y, w, h)
            self.crop_label_var.set(self._region_display_name(index))
            self.crop_x_var.set(x)
            self.crop_y_var.set(y)
            self.crop_w_var.set(w)
            self.crop_h_var.set(h)
            self.crop_angle_var.set(round(region.angle, 3))
            self.crop_square_var.set(bool(region.square))
            self._set_crop_controls_state(True)
        finally:
            self._syncing_crop_controls = False

    def _apply_crop_controls(self, changed: str) -> None:
        if self._syncing_crop_controls or self.reader is None:
            return
        index = self._crop_control_index()
        if index is None:
            return
        region = self.regions[index]
        if region.rect is None:
            return
        region.square = bool(self.crop_square_var.get())
        x = int(self.crop_x_var.get())
        y = int(self.crop_y_var.get())
        w = int(self.crop_w_var.get())
        h = int(self.crop_h_var.get())
        if region.square:
            if changed == "w":
                size = w
            elif changed == "h":
                size = h
            elif changed == "square":
                size = min(region.rect[2], region.rect[3])
            else:
                size = min(w, h) if w != h else w
            rect = _normalize_square_rect((x, y, size, size), self.reader.width, self.reader.height, size=size)
        else:
            rect = _normalize_rect((x, y, w, h), self.reader.width, self.reader.height)
        region.rect = rect
        self.editing_crop_idx = index
        self._sync_crop_controls_from_region()
        self.redraw_current_frame()
        self._schedule_crop_trimming_config_save()

    def _on_crop_angle_changed(self, _event=None):
        index = self._crop_control_index()
        if self._syncing_crop_controls or self.export_running or index is None:
            return
        try:
            angle = float(self.crop_angle_var.get())
            if not math.isfinite(angle):
                raise ValueError("Non-finite angle")
        except (ValueError, tk.TclError):
            self._sync_crop_controls_from_region()
            return "break"
        self.regions[index].angle = (angle + 180.0) % 360.0 - 180.0
        self._sync_crop_controls_from_region()
        self.redraw_current_frame()
        self._schedule_crop_trimming_config_save()
        return "break"

    def _on_crop_x_changed(self) -> None:
        self._apply_crop_controls("x")

    def _on_crop_y_changed(self) -> None:
        self._apply_crop_controls("y")

    def _on_crop_w_changed(self) -> None:
        self._apply_crop_controls("w")

    def _on_crop_h_changed(self) -> None:
        self._apply_crop_controls("h")

    def _on_crop_square_changed(self) -> None:
        self._apply_crop_controls("square")

    def set_status(self, text: str, auto_clear: bool = True) -> None:
        if self.status_clear_job is not None:
            try:
                self.after_cancel(self.status_clear_job)
            except tk.TclError:
                pass
            self.status_clear_job = None
        self.status_var.set(text)
        if auto_clear and text:
            self.status_clear_job = self.after(6000, self.clear_status)

    def clear_status(self) -> None:
        self.status_clear_job = None
        self.status_var.set("")

    def _log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert(tk.END, text.rstrip() + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.configure(state="disabled")

    def _crop_trimming_config_path(self) -> str:
        video_path = os.path.abspath(self.video_path_var.get().strip())
        folder = os.path.dirname(video_path)
        stem = Path(video_path).stem or "video"
        return os.path.join(folder, f"{stem}_cropping_trimming_config.yaml")

    def _crop_trimming_config_data(self) -> dict:
        if self.reader is None:
            raise RuntimeError("No video loaded.")
        regions: list[dict] = []
        for index, region in enumerate(self.regions):
            rect = None
            even_rect = None
            if region.rect is not None:
                rect = [int(v) for v in region.rect]
                even_rect = [int(v) for v in _even_crop_rect(region.rect, self.reader.width, self.reader.height)]
            regions.append(
                {
                    "name": self._region_display_name(index),
                    "token": self._region_token(index),
                    "kind": region.kind,
                    "export": bool(region.export_enabled),
                    "rect": rect,
                    "export_rect_even": even_rect,
                    "square": bool(region.square),
                    "angle_degrees_clockwise": float(region.angle),
                    "crop_orientation_degrees_clockwise": _crop_quarter_turns(region.angle) * 90,
                    "brightness": float(region.brightness),
                    "contrast": float(region.contrast),
                    "color_mode": str(region.color_mode),
                }
            )
        data = {
            "tool": "AMADEUS Cropping & Trimming",
            "video_path": self.video_path_var.get().strip(),
            "source_width": int(self.reader.width),
            "source_height": int(self.reader.height),
            "source_fps": float(self.reader.fps),
            "source_frame_count": int(self.reader.frame_count),
            "source_reported_frame_count": int(self.reader.reported_frame_count),
            "default_color_mode": self.default_color_mode,
            "trim": {
                "in_frame": int(self.in_frame),
                "out_frame": int(self.out_frame),
                "in_seconds": float(self.in_frame / max(1e-9, self.reader.fps)),
                "out_seconds_exclusive": float((self.out_frame + 1) / max(1e-9, self.reader.fps)),
            },
            "output": {
                "folder": self.output_folder_var.get().strip(),
                "template": self.template_var.get().strip() or "{stem}_{region}",
                "speed": float(self.output_speed_var.get()),
                "fps": float(self.output_fps_var.get()),
                "use_gpu_acceleration": bool(self.gpu_acceleration_var.get()),
                "video_encoder": (
                    self.hardware_video_encoder
                    if self.gpu_acceleration_var.get() and self.hardware_video_encoder
                    else "libx264"
                ),
                "apply_view_rotation": bool(self.export_view_rotation_var.get()),
                "view_rotation_degrees_clockwise": self.preview_turns * 90,
                "selected_regions": [
                    self._region_token(index)
                    for index, region in enumerate(self.regions)
                    if region.export_enabled
                ],
            },
            "regions": regions,
        }
        # Empty unless this video is an analysis copy; read back from the record
        # FFmpeg conversion left beside it, so it traces to the original recording.
        conversion = video_conversion_record(self.video_path_var.get().strip())
        if conversion:
            data["video_conversion"] = conversion
        return data

    def _write_crop_trimming_config(self, *, log_success: bool = False) -> str | None:
        if self.reader is None:
            return None
        path = self._crop_trimming_config_path()
        try:
            import yaml

            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = self._crop_trimming_config_data()
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            os.replace(tmp_path, path)
            if log_success:
                self._log(f"Saved cropping/trimming config: {path}")
            return path
        except Exception as exc:
            self._log(f"WARNING: failed to save cropping/trimming config: {exc}")
            return None

    def _schedule_crop_trimming_config_save(self) -> None:
        if self.reader is None:
            return
        if self.config_save_job is not None:
            try:
                self.after_cancel(self.config_save_job)
            except tk.TclError:
                pass
        self.config_save_job = self.after(500, self._run_scheduled_crop_trimming_config_save)

    def _run_scheduled_crop_trimming_config_save(self) -> None:
        self.config_save_job = None
        self._write_crop_trimming_config()

    def _flush_crop_trimming_config_save(self, *, log_success: bool = False) -> None:
        if self.config_save_job is not None:
            try:
                self.after_cancel(self.config_save_job)
            except tk.TclError:
                pass
            self.config_save_job = None
        self._write_crop_trimming_config(log_success=log_success)

    def _detect_default_color_mode(self, reader: VideoFrameReader) -> str:
        sample_indices = sorted({
            0,
            max(0, reader.frame_count // 4),
            max(0, reader.frame_count // 2),
            max(0, (reader.frame_count * 3) // 4),
            max(0, reader.frame_count - 1),
        })
        spreads: list[tuple[float, float]] = []
        for frame_idx in sample_indices:
            try:
                frame = reader.read_bgr(frame_idx)
            except Exception:
                continue
            if frame.ndim < 3 or frame.shape[2] < 3:
                return "grayscale"
            stride_y = max(1, frame.shape[0] // 240)
            stride_x = max(1, frame.shape[1] // 240)
            sample = frame[::stride_y, ::stride_x, :3].astype(np.int16)
            spread = sample.max(axis=2) - sample.min(axis=2)
            spreads.append((float(np.mean(spread)), float(np.percentile(spread, 99))))
        if spreads and all(
            mean <= GRAYSCALE_MEAN_SPREAD_THRESHOLD and p99 <= GRAYSCALE_P99_SPREAD_THRESHOLD
            for mean, p99 in spreads
        ):
            return "grayscale"
        return "color"

    def browse_video(self) -> None:
        prepared = ask_open_analysis_video(self, title="Select video", log=self._log)
        if prepared is not None:
            self.load_video(prepared.path)

    def select_video(self, path: str) -> None:
        """Handle a video the user chose: check it, then load what we can read.

        Only the browse and drop paths run the compatibility check; load_video
        itself stays free of it so restoring a session never re-asks.
        """
        prepared = prepare_analysis_video(self, path, log=self._log)
        if prepared is not None:
            self.load_video(prepared.path)

    def load_video(self, path: str) -> None:
        path = str(Path(path).expanduser())
        if not os.path.isfile(path):
            messagebox.showerror("Error", "Video file not found.")
            return
        try:
            reader = VideoFrameReader(path)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        self.stop_playback(update_button=False)
        if self.reader is not None:
            self.reader.close()
        self.view_initialized = False
        self.reader = reader
        self.video_path_var.set(path)
        self.output_folder_var.set(os.path.dirname(path))
        self.current_frame = 0
        self.in_frame = 0
        self.out_frame = reader.frame_count - 1
        self.frame_bgr_cache = None
        self.output_speed_var.set(1.0)
        self.output_speed_spin.delete(0, tk.END)
        self.output_speed_spin.insert(0, "1.00")
        self.output_fps_var.set(float(reader.fps))
        self.output_fps_spin.delete(0, tk.END)
        self.output_fps_spin.insert(0, f"{reader.fps:.3f}")
        self.default_color_mode = self._detect_default_color_mode(reader)
        self.regions = [Region(uid=0, kind="full", color_mode=self.default_color_mode)]
        self.active_region_idx = 0
        self.editing_crop_idx = None
        self.next_region_uid = 1
        self._refresh_region_list(select_index=0)
        self._configure_frame_slider()
        self._update_meta_label()
        self.set_frame(0)
        self._refresh_video_controls()
        self._log(reader.orientation_note)
        self._log(
            "Detected grayscale input; default color mode set to Grayscale."
            if self.default_color_mode == "grayscale"
            else "Detected color input; default color mode set to Color."
        )
        if reader.fps_estimated:
            self._log("FPS metadata was unavailable; timeline uses 30 fps.")
        if reader.frame_count_adjusted:
            self._log(
                f"Usable frame count adjusted to {reader.frame_count:,} "
                f"(container reported {reader.reported_frame_count:,})."
            )
        self._flush_crop_trimming_config_save(log_success=True)
        self.set_status(f"Loaded: {os.path.basename(path)}")
        _start_maximized(self)

    def _update_meta_label(self) -> None:
        if self.reader is None:
            self.meta_var.set("No video loaded.")
            return
        duration = self.reader.frame_count / max(1e-9, self.reader.fps)
        frame_count_text = f"{self.reader.frame_count:,}"
        if self.reader.frame_count_adjusted:
            frame_count_text += f" usable (reported {self.reader.reported_frame_count:,})"
        self.meta_var.set(
            f"{self.reader.width} x {self.reader.height}\n"
            f"fps: {self.reader.fps:.3f}\n"
            f"frames: {frame_count_text}\n"
            f"duration: {_format_seconds(duration)}\n"
            f"codec: {self.reader.codec}\n"
            f"default color mode: {'Grayscale' if self.default_color_mode == 'grayscale' else 'Color'}\n"
            f"{self.reader.orientation_note}"
        )

    def _configure_frame_slider(self) -> None:
        last = self._last_frame_index()
        self.frame_slider.configure(from_=0, to=max(0, last))
        self.in_frame_spin.configure(from_=0, to=max(0, last))
        self.out_frame_spin.configure(from_=0, to=max(0, last))
        self.frame_slider.set(0)
        self._update_timeline_labels()

    def _sync_trim_frame_inputs(self) -> None:
        self._syncing_trim_inputs = True
        try:
            self.in_frame_var.set(str(int(self.in_frame)))
            self.out_frame_var.set(str(int(self.out_frame)))
        finally:
            self._syncing_trim_inputs = False

    def _commit_trim_frame_input(self, which: str | None = None):
        """Apply the editable In/Out frame fields to the active trim range."""
        if self._syncing_trim_inputs or self.reader is None:
            return "break"
        try:
            in_frame = int(self.in_frame_var.get().strip())
            out_frame = int(self.out_frame_var.get().strip())
        except (AttributeError, TypeError, ValueError):
            self.set_status("Trim frame values must be integers.", auto_clear=False)
            self._sync_trim_frame_inputs()
            return "break"

        if which == "in" and in_frame > out_frame:
            out_frame = in_frame
        elif which == "out" and out_frame < in_frame:
            in_frame = out_frame

        last = self._last_frame_index()
        if not (0 <= in_frame <= out_frame <= last):
            self.set_status(
                f"Trim frame range must satisfy 0 <= in <= out <= {last}.",
                auto_clear=False,
            )
            self._sync_trim_frame_inputs()
            return "break"

        self.in_frame = int(in_frame)
        self.out_frame = int(out_frame)
        self._sync_trim_frame_inputs()
        self._update_timeline_labels()
        self._schedule_crop_trimming_config_save()
        return "break"

    def _refresh_video_controls(self) -> None:
        has_video = self.reader is not None and not self.export_running
        normal = "normal" if has_video else "disabled"
        for widget in [
            self.add_crop_button,
            self.first_button,
            self.prev_button,
            self.play_button,
            self.next_button,
            self.last_button,
            self.set_in_button,
            self.set_out_button,
            self.in_frame_spin,
            self.out_frame_spin,
            self.frame_slider,
        ]:
            widget.configure(state=normal)
        delete_state = "normal" if has_video and self.active_region_idx > 0 else "disabled"
        self.delete_crop_button.configure(state=delete_state)
        export_state = "normal" if has_video and self.output_folder_var.get().strip() else "disabled"
        if not any(region.export_enabled for region in self.regions):
            export_state = "disabled"
        self.export_button.configure(state=export_state if not self.export_running else "disabled")
        self._set_crop_controls_state(has_video and self._crop_control_index() is not None)
        self._set_region_list_state("normal" if not self.export_running else "disabled")

    def add_crop(self) -> None:
        if self.reader is None:
            messagebox.showwarning("No video", "Please load a video first.")
            return
        w = max(MIN_CROP_SIZE, self.reader.width // 3)
        h = max(MIN_CROP_SIZE, self.reader.height // 3)
        w = min(w, self.reader.width)
        h = min(h, self.reader.height)
        x = max(0, (self.reader.width - w) // 2)
        y = max(0, (self.reader.height - h) // 2)
        region = Region(uid=self.next_region_uid, kind="crop", rect=(x, y, w, h), color_mode=self.default_color_mode)
        self.next_region_uid += 1
        self.regions.append(region)
        self._apply_export_defaults_after_crop_change(force_crop_exports=True)
        self.editing_crop_idx = len(self.regions) - 1
        self.active_region_idx = 0
        self._refresh_region_list(select_index=0)
        self._sync_crop_controls_from_region()
        self.redraw_current_frame()
        self._refresh_video_controls()
        self._schedule_crop_trimming_config_save()
        self.set_status(f"{self._region_display_name(self.editing_crop_idx)} added.")

    def delete_crop(self) -> None:
        if self.reader is None:
            return
        if self.active_region_idx <= 0:
            messagebox.showwarning("Delete Crop", "Select a crop in the region list.")
            return
        name = self._region_display_name(self.active_region_idx)
        del self.regions[self.active_region_idx]
        if self.editing_crop_idx == self.active_region_idx:
            self.editing_crop_idx = None
        elif self.editing_crop_idx is not None and self.editing_crop_idx > self.active_region_idx:
            self.editing_crop_idx -= 1
        self.active_region_idx = 0
        self._apply_export_defaults_after_crop_change()
        self._refresh_region_list(select_index=0)
        self._sync_crop_controls_from_region()
        self.redraw_current_frame()
        self._refresh_video_controls()
        self._schedule_crop_trimming_config_save()
        self.set_status(f"Deleted {name}.")

    def on_frame_slider(self, value) -> None:
        if self.reader is None:
            return
        self.set_frame(int(round(float(value))))

    def _on_frame_slider_double_click(self, _event=None) -> None:
        if self.reader is None or self.export_running:
            return
        # Let the slider finish handling the click before reading its value.
        self.after_idle(self._jump_to_nearest_trim_boundary)

    def _jump_to_nearest_trim_boundary(self) -> None:
        if self.reader is None or self.export_running:
            return
        clicked_frame = max(
            0,
            min(self._last_frame_index(), int(round(float(self.frame_slider.get())))),
        )
        if abs(clicked_frame - self.in_frame) <= abs(clicked_frame - self.out_frame):
            target_frame = self.in_frame
            boundary_name = "In"
        else:
            target_frame = self.out_frame
            boundary_name = "Out"
        self.set_frame(target_frame)
        self.set_status(f"Moved to {boundary_name} frame {target_frame:,}.")

    def set_frame(self, frame_idx: int) -> None:
        if self.reader is None:
            return
        frame_idx = max(0, min(self._last_frame_index(), int(frame_idx)))
        self.current_frame = frame_idx
        try:
            self.frame_bgr_cache = self.reader.read_bgr(frame_idx)
        except Exception as exc:
            self.set_status(f"Frame decode failed: {exc}", auto_clear=False)
            return
        self.frame_slider.set(frame_idx)
        self._update_timeline_labels()
        self.redraw_current_frame()

    def step_frame(self, delta: int) -> None:
        if self.reader is None:
            return
        self.set_frame(self.current_frame + int(delta))

    def set_in_frame(self) -> None:
        if self.reader is None:
            return
        self.in_frame = int(self.current_frame)
        if self.in_frame > self.out_frame:
            self.out_frame = self.in_frame
        self._sync_trim_frame_inputs()
        self._update_timeline_labels()
        self._schedule_crop_trimming_config_save()

    def set_out_frame(self) -> None:
        if self.reader is None:
            return
        self.out_frame = int(self.current_frame)
        if self.out_frame < self.in_frame:
            self.in_frame = self.out_frame
        self._sync_trim_frame_inputs()
        self._update_timeline_labels()
        self._schedule_crop_trimming_config_save()

    def toggle_playback(self) -> None:
        if self.reader is None:
            return
        if self.playback_active:
            self.stop_playback()
            return
        if self.current_frame >= self.out_frame:
            self.set_frame(self.in_frame)
        self.playback_active = True
        self.play_button.configure(text="||")
        self._schedule_playback_step()

    def _schedule_playback_step(self) -> None:
        if self.reader is None or not self.playback_active:
            return
        interval = max(10, int(round(1000.0 / max(1e-9, self.reader.fps))))
        self.playback_job = self.after(interval, self._playback_step)

    def _playback_step(self) -> None:
        self.playback_job = None
        if self.reader is None or not self.playback_active:
            return
        if self.current_frame >= self.out_frame:
            self.stop_playback()
            return
        self.set_frame(self.current_frame + 1)
        self._schedule_playback_step()

    def stop_playback(self, update_button: bool = True) -> None:
        self.playback_active = False
        if self.playback_job is not None:
            try:
                self.after_cancel(self.playback_job)
            except tk.TclError:
                pass
            self.playback_job = None
        if update_button and hasattr(self, "play_button"):
            self.play_button.configure(text=">")

    def _update_timeline_labels(self) -> None:
        self._sync_trim_frame_inputs()
        self._update_output_summary()
        if self.reader is None:
            self.frame_label_var.set("frame 0 / 0   00:00.000")
            self.trim_label_var.set("in 0 / out 0 / length 0 frames")
            self._draw_trim_markers()
            return
        total = max(1, self.reader.frame_count)
        fps = max(1e-9, self.reader.fps)
        self.frame_label_var.set(
            f"frame {self.current_frame:,} / {total - 1:,}   {_format_seconds(self.current_frame / fps)}"
        )
        length = max(0, self.out_frame - self.in_frame + 1)
        self.trim_label_var.set(
            f"in {self.in_frame:,}  out {self.out_frame:,}  length {length:,} frames / {_format_seconds(length / fps)}"
        )
        self._draw_trim_markers()

    def _draw_trim_markers(self) -> None:
        canvas = self.trim_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.create_rectangle(0, 0, width, height, fill=TIMELINE_BG, outline="")
        if self.reader is None:
            return
        total = max(1, self.reader.frame_count)
        start_x = width * self.in_frame / total
        end_x = width * min(total, self.out_frame + 1) / total
        current_x = width * min(total - 1, self.current_frame) / max(1, total - 1)
        canvas.create_rectangle(start_x, 2, end_x, height - 2, fill=TIMELINE_RANGE, outline="")
        canvas.create_line(start_x, 0, start_x, height, fill=_rgb_hex(GREEN_RGB), width=2)
        canvas.create_line(end_x, 0, end_x, height, fill=_rgb_hex(GREEN_RGB), width=2)
        canvas.create_line(current_x, 0, current_x, height, fill=TIMELINE_LINE, width=2)

    def _build_preview_bgr(self) -> np.ndarray | None:
        if self.frame_bgr_cache is None:
            return None
        region = self._active_region()
        frame = self.frame_bgr_cache
        if self.active_region_idx > 0 and region.rect is not None:
            x, y, w, h = _normalize_rect(region.rect, frame.shape[1], frame.shape[0])
            frame = _crop_rotated_bgr(frame, (x, y, w, h), region.angle)
        adjust = self._adjustment_region()
        return _process_region_bgr(frame, adjust.brightness, adjust.contrast, adjust.color_mode)

    def redraw_current_frame(self) -> None:
        preview_bgr = self._build_preview_bgr()
        if preview_bgr is None:
            self.canvas.delete("all")
            w = max(1, self.canvas.winfo_width())
            h = max(1, self.canvas.winfo_height())
            self.canvas.create_text(w / 2, h / 2, text="No video loaded", fill=MUTED_TEXT, font=("Arial", 18))
            return
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        self._present_image(preview_rgb)
        if self.active_region_idx == 0:
            self._draw_crop_overlays()
        if self.cursor_position is not None:
            self._draw_edit_cursor(*self.cursor_position)

    def fit_canvas_to_window(self):
        self.view_initialized = False
        self.redraw_current_frame()

    def rotate_preview(self):
        self.drag_state = None
        self.preview_turns = (self.preview_turns + 1) % 4
        self.fit_canvas_to_window()
        self._schedule_crop_trimming_config_save()

    def on_mousewheel(self, event):
        if self.frame_bgr_cache is None or self.drag_state is not None:
            return
        direction = getattr(event, "delta", 0)
        factor = 1.12 if direction > 0 or getattr(event, "num", None) == 4 else 1 / 1.12
        zoom = _clamp(self.zoom_scale * factor, 0.2, 20.0)
        factor = zoom / self.zoom_scale
        self.zoom_scale = zoom
        self.view_offset_x = event.x - (event.x - self.view_offset_x) * factor
        self.view_offset_y = event.y - (event.y - self.view_offset_y) * factor
        self.view_scale *= factor
        self.redraw_current_frame()
        return "break"

    def _present_image(self, img_rgb: np.ndarray) -> None:
        self.unrotated_view_shape = img_rgb.shape[:2]
        img_rgb = np.rot90(img_rgb, -self.preview_turns)
        img_h, img_w = img_rgb.shape[:2]
        can_w = max(1, self.canvas.winfo_width())
        can_h = max(1, self.canvas.winfo_height())
        if not self.view_initialized or self.view_source_shape != (img_h, img_w):
            self.view_scale = min(can_w / img_w, can_h / img_h)
            self.zoom_scale = 1.0
            self.view_offset_x = (can_w - img_w * self.view_scale) / 2
            self.view_offset_y = (can_h - img_h * self.view_scale) / 2
            self.view_source_shape = (img_h, img_w)
            self.view_initialized = True
        # Render only the viewport, even at high zoom.
        matrix = np.array([[self.view_scale, 0, self.view_offset_x],
                           [0, self.view_scale, self.view_offset_y]], dtype=float)
        rendered = cv2.warpAffine(img_rgb, matrix, (can_w, can_h), flags=cv2.INTER_LINEAR)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(rendered))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def save_canvas_screenshot(self):
        frame = self._build_preview_bgr()
        if frame is None:
            return
        path = filedialog.asksaveasfilename(parent=self, title="Save screenshot",
            defaultextension=".png", filetypes=[("PNG image", "*.png")],
            initialdir=self.output_folder_var.get().strip() or os.getcwd(),
            initialfile=f"{Path(self.video_path_var.get()).stem}_frame{self.current_frame:06d}.png")
        if not path:
            return
        try:
            # Like segmentation, save the visible area at native image resolution.
            frame = frame.copy()
            if self.active_region_idx == 0:
                for index, region in enumerate(self.regions[1:], 1):
                    if region.rect is None:
                        continue
                    points = np.rint(_crop_corners(region.rect, region.angle)).astype(np.int32)
                    color = GREEN_RGB if index == self.editing_crop_idx else CYAN_RGB
                    cv2.polylines(frame, [points], True, tuple(reversed(color)), 2)
                    cv2.putText(frame, self._region_display_name(index), tuple(points[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(reversed(color)), 1, cv2.LINE_AA)
            frame = np.rot90(frame, -self.preview_turns)
            x0 = -self.view_offset_x / self.view_scale
            y0 = -self.view_offset_y / self.view_scale
            x1 = (self.canvas.winfo_width() - self.view_offset_x) / self.view_scale
            y1 = (self.canvas.winfo_height() - self.view_offset_y) / self.view_scale
            h, w = frame.shape[:2]
            frame = frame[int(_clamp(math.floor(y0), 0, h)):int(_clamp(math.ceil(y1), 0, h)),
                          int(_clamp(math.floor(x0), 0, w)):int(_clamp(math.ceil(x1), 0, w))]
            if frame.size == 0:
                raise ValueError("No image pixels are visible. Use Fit window first.")
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save screenshot: {exc}")
            return
        self.set_status(f"Screenshot saved: {path}")

    def source_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        h, w = self.unrotated_view_shape
        if self.preview_turns == 1:
            x, y = h-y, x
        elif self.preview_turns == 2:
            x, y = w-x, h-y
        elif self.preview_turns == 3:
            x, y = y, w-x
        return self.view_offset_x + float(x) * self.view_scale, self.view_offset_y + float(y) * self.view_scale

    def canvas_to_source(self, x: float, y: float) -> tuple[float, float]:
        scale = max(1e-9, self.view_scale)
        x, y = (float(x) - self.view_offset_x) / scale, (float(y) - self.view_offset_y) / scale
        h, w = self.unrotated_view_shape
        if self.preview_turns == 1:
            return y, h-x
        if self.preview_turns == 2:
            return w-x, h-y
        if self.preview_turns == 3:
            return w-y, x
        return x, y

    def _draw_crop_overlays(self) -> None:
        for index, region in enumerate(self.regions):
            if index == 0 or region.rect is None:
                continue
            x, y, w, h = _normalize_rect(region.rect, self.reader.width, self.reader.height)  # type: ignore[union-attr]
            selected = index == self.editing_crop_idx
            color = _rgb_hex(GREEN_RGB if selected else CYAN_RGB)
            width = 3 if selected else 2
            corners = [self.source_to_canvas(*point) for point in _crop_corners(region.rect, region.angle)]
            x0, y0 = corners[0]
            self.canvas.create_polygon(*[v for point in corners for v in point], fill="", outline=color, width=width)
            self.canvas.create_text(
                x0 + 6,
                y0 + 6,
                text=self._region_display_name(index),
                fill=color,
                anchor="nw",
                font=("Arial", 11, "bold"),
            )
            if selected:
                self._draw_handles(x, y, x+w, y+h, region.angle)

    def _draw_handles(self, x0: float, y0: float, x1: float, y1: float, angle: float = 0.0) -> None:
        for hx, hy in self._handle_points(x0, y0, x1, y1).values():
            hx, hy = _rotate_point(hx, hy, (x0 + x1) / 2, (y0 + y1) / 2, angle)
            hx, hy = self.source_to_canvas(hx, hy)
            r = HANDLE_DRAW_PX / 2
            self.canvas.create_rectangle(
                hx - r,
                hy - r,
                hx + r,
                hy + r,
                fill=_rgb_hex(ORANGE_RGB),
                outline=_rgb_hex(WHITE_RGB),
                width=1,
            )

    def _handle_points(self, x0: float, y0: float, x1: float, y1: float) -> dict[str, tuple[float, float]]:
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        return {
            "nw": (x0, y0),
            "n": (cx, y0),
            "ne": (x1, y0),
            "e": (x1, cy),
            "se": (x1, y1),
            "s": (cx, y1),
            "sw": (x0, y1),
            "w": (x0, cy),
        }

    def _hit_test_crop(self, canvas_x: float, canvas_y: float) -> tuple[int | None, str | None]:
        if self.reader is None or self.active_region_idx != 0:
            return None, None
        indices = list(range(1, len(self.regions)))
        if self.editing_crop_idx in indices:
            indices.remove(self.editing_crop_idx)
            indices.insert(0, self.editing_crop_idx)
        for index in indices:
            region = self.regions[index]
            if region.rect is None:
                continue
            x, y, w, h = _normalize_rect(region.rect, self.reader.width, self.reader.height)
            x0, y0, x1, y1 = x, y, x+w, y+h
            source_x, source_y = self.canvas_to_source(canvas_x, canvas_y)
            local_x, local_y = _rotate_point(source_x, source_y, x+w/2, y+h/2, -region.angle)
            tolerance = HANDLE_HIT_PX / self.view_scale
            for handle, (hx, hy) in self._handle_points(x0, y0, x1, y1).items():
                if abs(local_x - hx) <= tolerance and abs(local_y - hy) <= tolerance:
                    return index, handle
            for hx, hy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                outside_x = local_x < x0 if hx == x0 else local_x > x1
                outside_y = local_y < y0 if hy == y0 else local_y > y1
                distance = math.hypot(local_x - hx, local_y - hy)
                if outside_x and outside_y and tolerance < distance <= 32 / self.view_scale:
                    return index, "rotate"
            if x0 <= local_x <= x1 and y0 <= local_y <= y1:
                return index, "move"
        return None, None

    def _on_canvas_press(self, event) -> None:
        if self.reader is None or self.export_running:
            return
        index, mode = self._hit_test_crop(event.x, event.y)
        if index is None or mode is None:
            self.drag_state = {"mode": "pan", "canvas": (event.x, event.y)}
            return
        self.editing_crop_idx = index
        source_x, source_y = self.canvas_to_source(event.x, event.y)
        self.drag_state = {
            "index": index,
            "mode": mode,
            "start_source": (source_x, source_y),
            "start_rect": self.regions[index].rect,
            "start_angle": self.regions[index].angle,
        }
        # Picking a crop on the canvas moves the appearance controls with it.
        self._sync_adjustment_controls_from_region()
        self.redraw_current_frame()

    def _resize_square_rect(
        self,
        rect: tuple[int, int, int, int],
        mode: str,
        dx: float,
        dy: float,
    ) -> tuple[int, int, int, int]:
        if self.reader is None:
            return rect
        x, y, w, h = _normalize_square_rect(rect, self.reader.width, self.reader.height)
        width = float(self.reader.width)
        height = float(self.reader.height)
        min_size = float(min(MIN_CROP_SIZE, self.reader.width, self.reader.height))
        if mode == "se":
            size = max(w + dx, h + dy, min_size)
            size = min(size, width - x, height - y)
            return _normalize_square_rect((x, y, int(round(size)), int(round(size))), self.reader.width, self.reader.height, size=int(round(size)))
        if mode == "nw":
            right = x + w
            bottom = y + h
            size = max(w - dx, h - dy, min_size)
            size = min(size, right, bottom)
            return _normalize_square_rect((int(round(right - size)), int(round(bottom - size)), int(round(size)), int(round(size))), self.reader.width, self.reader.height, size=int(round(size)))
        if mode == "ne":
            left = x
            bottom = y + h
            size = max(w + dx, h - dy, min_size)
            size = min(size, width - left, bottom)
            return _normalize_square_rect((left, int(round(bottom - size)), int(round(size)), int(round(size))), self.reader.width, self.reader.height, size=int(round(size)))
        if mode == "sw":
            right = x + w
            top = y
            size = max(w - dx, h + dy, min_size)
            size = min(size, right, height - top)
            return _normalize_square_rect((int(round(right - size)), top, int(round(size)), int(round(size))), self.reader.width, self.reader.height, size=int(round(size)))
        if mode in {"e", "w"}:
            center_y = y + h * 0.5
            if mode == "e":
                fixed_x = x
                size = max(w + dx, min_size)
                size = min(size, width - fixed_x, center_y * 2.0, (height - center_y) * 2.0)
                new_x = fixed_x
            else:
                fixed_right = x + w
                size = max(w - dx, min_size)
                size = min(size, fixed_right, center_y * 2.0, (height - center_y) * 2.0)
                new_x = fixed_right - size
            new_y = center_y - size * 0.5
            return _normalize_square_rect((int(round(new_x)), int(round(new_y)), int(round(size)), int(round(size))), self.reader.width, self.reader.height, size=int(round(size)))
        if mode in {"n", "s"}:
            center_x = x + w * 0.5
            if mode == "s":
                fixed_y = y
                size = max(h + dy, min_size)
                size = min(size, height - fixed_y, center_x * 2.0, (width - center_x) * 2.0)
                new_y = fixed_y
            else:
                fixed_bottom = y + h
                size = max(h - dy, min_size)
                size = min(size, fixed_bottom, center_x * 2.0, (width - center_x) * 2.0)
                new_y = fixed_bottom - size
            new_x = center_x - size * 0.5
            return _normalize_square_rect((int(round(new_x)), int(round(new_y)), int(round(size)), int(round(size))), self.reader.width, self.reader.height, size=int(round(size)))
        return _normalize_square_rect(rect, self.reader.width, self.reader.height)

    def _on_canvas_drag(self, event) -> None:
        self.cursor_position = (event.x, event.y)
        if self.reader is None or self.drag_state is None:
            return
        if self.export_running:
            return
        if self.drag_state["mode"] == "pan":
            x, y = self.drag_state["canvas"]
            self.view_offset_x += event.x - x
            self.view_offset_y += event.y - y
            self.drag_state["canvas"] = (event.x, event.y)
            self.redraw_current_frame()
            return
        index = int(self.drag_state["index"])
        mode = str(self.drag_state["mode"])
        start_rect = self.drag_state["start_rect"]
        if start_rect is None:
            return
        sx, sy = self.drag_state["start_source"]
        px, py = self.canvas_to_source(event.x, event.y)
        dx = px - float(sx)
        dy = py - float(sy)
        x, y, w, h = start_rect
        angle = float(self.drag_state["start_angle"])
        if mode == "rotate":
            cx, cy = x + w/2, y + h/2
            delta = math.degrees(math.atan2(py-cy, px-cx) - math.atan2(float(sy)-cy, float(sx)-cx))
            self.regions[index].angle = (angle + delta + 180) % 360 - 180
            self._sync_crop_controls_from_region()
            self.redraw_current_frame()
            self._schedule_crop_trimming_config_save()
            return
        if mode != "move":
            dx, dy = _rotate_point(dx, dy, 0, 0, -angle)
        if mode == "move":
            new_rect = (int(round(x + dx)), int(round(y + dy)), w, h)
        elif self.regions[index].square:
            new_rect = self._resize_square_rect(start_rect, mode, dx, dy)
        else:
            left = float(x)
            top = float(y)
            right = float(x + w)
            bottom = float(y + h)
            if "w" in mode:
                left += dx
            if "e" in mode:
                right += dx
            if "n" in mode:
                top += dy
            if "s" in mode:
                bottom += dy
            left, top, right, bottom = self._clamp_resize_edges(left, top, right, bottom)
            new_rect = (
                int(round(left)),
                int(round(top)),
                int(round(right - left)),
                int(round(bottom - top)),
            )
        if mode != "move" and angle:
            nx, ny, nw, nh = new_rect
            cx, cy = _rotate_point(nx + nw/2, ny + nh/2, x + w/2, y + h/2, angle)
            new_rect = (round(cx - nw/2), round(cy - nh/2), nw, nh)
        self.regions[index].rect = _normalize_rect(new_rect, self.reader.width, self.reader.height)
        if self.regions[index].square:
            self.regions[index].rect = _normalize_square_rect(self.regions[index].rect, self.reader.width, self.reader.height)
        self._sync_crop_controls_from_region()
        self.redraw_current_frame()
        self._schedule_crop_trimming_config_save()

    def _clamp_resize_edges(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> tuple[float, float, float, float]:
        if self.reader is None:
            return left, top, right, bottom
        width = float(self.reader.width)
        height = float(self.reader.height)
        left = _clamp(left, 0.0, width - MIN_CROP_SIZE)
        top = _clamp(top, 0.0, height - MIN_CROP_SIZE)
        right = _clamp(right, left + MIN_CROP_SIZE, width)
        bottom = _clamp(bottom, top + MIN_CROP_SIZE, height)
        if right - left < MIN_CROP_SIZE:
            right = min(width, left + MIN_CROP_SIZE)
        if bottom - top < MIN_CROP_SIZE:
            bottom = min(height, top + MIN_CROP_SIZE)
        return left, top, right, bottom

    def _on_canvas_release(self, event) -> None:
        self.drag_state = None
        self._on_canvas_motion(event)

    def _on_canvas_leave(self, _event=None) -> None:
        self.cursor_position = None
        self.canvas.delete("edit_cursor")
        self.canvas.configure(cursor="")

    def _on_canvas_motion(self, event) -> None:
        self.cursor_position = (event.x, event.y)
        self._draw_edit_cursor(event.x, event.y)

    def _draw_edit_cursor(self, px: float, py: float) -> None:
        self.canvas.delete("edit_cursor")
        if self.reader is None or self.active_region_idx != 0 or self.export_running:
            self.canvas.configure(cursor="")
            return
        index, mode = self._hit_test_crop(px, py)
        if self.drag_state is not None and self.drag_state.get("mode") != "pan":
            index, mode = int(self.drag_state["index"]), str(self.drag_state["mode"])
        if mode is None or mode == "move":
            self.canvas.configure(cursor="fleur" if mode == "move" else "")
            return
        region = self.regions[index]
        if mode == "rotate":
            corners = [self.source_to_canvas(*point) for point in _crop_corners(region.rect, region.angle)]
            cx, cy = min(corners, key=lambda point: math.hypot(px-point[0], py-point[1]))
            # The arc bulges away from the corner; its arrows are tangential.
            direction = math.atan2(py-cy, px-cx)
            radius = 16.0
            center_x = px - radius * math.cos(direction)
            center_y = py - radius * math.sin(direction)
            points = []
            for step in range(21):
                angle = direction + math.radians(-55 + 110 * step / 20)
                points.extend((center_x + radius * math.cos(angle),
                               center_y + radius * math.sin(angle)))
        else:
            # Transform the resize axis by both crop and preview rotations.
            dx = -1 if "w" in mode else (1 if "e" in mode else 0)
            dy = -1 if "n" in mode else (1 if "s" in mode else 0)
            dx, dy = _rotate_point(dx, dy, 0, 0, region.angle + 90 * self.preview_turns)
            length = math.hypot(dx, dy)
            dx, dy = 12 * dx / length, 12 * dy / length
            points = [px-dx, py-dy, px+dx, py+dy]
        # A canvas cursor supports arbitrary angles consistently across platforms.
        self.canvas.configure(cursor="none")
        self.canvas.create_line(*points, fill="black", width=5, arrow="both",
                                arrowshape=(9, 11, 6), tags="edit_cursor")
        self.canvas.create_line(*points, fill="white", width=2, arrow="both",
                                arrowshape=(7, 9, 4), tags="edit_cursor")

    def browse_output_folder(self) -> None:
        initial = self.output_folder_var.get().strip() or os.getcwd()
        path = filedialog.askdirectory(parent=self, initialdir=initial, title="Select output folder")
        if path:
            self.output_folder_var.set(path)
            self._refresh_video_controls()

    def start_export(self) -> None:
        if self.export_running:
            return
        try:
            job = self._build_export_job()
        except Exception as exc:
            messagebox.showerror("Export", str(exc))
            return
        self.stop_playback()
        self.export_cancel.clear()
        self.export_running = True
        self._last_progress_update = 0.0
        self.progress_var.set(0.0)
        self.progress_bar.set(0.0)
        self._clear_log()
        self._flush_crop_trimming_config_save(log_success=True)
        self._set_export_ui_running(True)
        self.export_thread = threading.Thread(target=self._export_worker, args=(job,), daemon=True)
        self.export_thread.start()
        self.export_poll_job = self.after(80, self._poll_export_queue)

    def _build_export_job(self) -> ExportJob:
        if self.reader is None:
            raise RuntimeError("Please load a video first.")
        video_path = self.video_path_var.get().strip()
        if not video_path or not os.path.isfile(video_path):
            raise RuntimeError("Video file not found.")
        output_folder = self.output_folder_var.get().strip()
        if not output_folder:
            raise RuntimeError("Please select an output folder.")
        template = self.template_var.get().strip() or "{stem}_{region}"
        try:
            template.format(stem="stem", region="region")
        except Exception as exc:
            raise RuntimeError(f"Invalid output name template: {exc}") from exc
        if not (0 <= self.in_frame <= self.out_frame < self.reader.frame_count):
            raise RuntimeError("Invalid trim range: in must be less than or equal to out.")
        self._commit_output_speed()
        self._commit_output_fps()
        output_speed = float(self.output_speed_var.get())
        output_fps = float(self.output_fps_var.get())
        if not (OUTPUT_SPEED_MIN <= output_speed <= OUTPUT_SPEED_MAX):
            raise RuntimeError(
                f"Output speed must be between {OUTPUT_SPEED_MIN:.1f}x and {OUTPUT_SPEED_MAX:.1f}x."
            )
        if not (1.0 <= output_fps <= 240.0):
            raise RuntimeError("Output fps must be between 1.0 and 240.0.")
        video_encoder = "libx264"
        if self.gpu_acceleration_var.get():
            if self.hardware_video_encoder is None or self.hardware_video_ffmpeg is None:
                raise RuntimeError("GPU acceleration is selected, but no supported GPU encoder is available.")
            video_encoder = self.hardware_video_encoder
            export_ffmpeg = self.hardware_video_ffmpeg
        else:
            export_ffmpeg = ffmpeg_exe()

        stem = Path(video_path).stem
        reserved: set[str] = set()
        snapshots: list[ExportRegion] = []
        selected_indices = [index for index, region in enumerate(self.regions) if region.export_enabled]
        if not selected_indices:
            raise RuntimeError("Select at least one region to export.")
        for index in selected_indices:
            region = self.regions[index]
            if not (BRIGHTNESS_MIN <= float(region.brightness) <= BRIGHTNESS_MAX):
                raise RuntimeError(f"{self._region_display_name(index)} brightness is outside the slider range.")
            if not (CONTRAST_MIN <= float(region.contrast) <= CONTRAST_MAX):
                raise RuntimeError(f"{self._region_display_name(index)} contrast is outside the slider range.")
            if region.color_mode not in {"color", "grayscale"}:
                raise RuntimeError(f"{self._region_display_name(index)} color mode is invalid.")
            rect = None
            if index > 0:
                if region.rect is None:
                    raise RuntimeError(f"{self._region_display_name(index)} has no crop rectangle.")
                rect = _even_crop_rect(region.rect, self.reader.width, self.reader.height)
                x, y, w, h = rect
                if not (0 <= x < self.reader.width and 0 <= y < self.reader.height):
                    raise RuntimeError(f"{self._region_display_name(index)} crop is outside the frame.")
                if x + w > self.reader.width or y + h > self.reader.height:
                    raise RuntimeError(f"{self._region_display_name(index)} crop exceeds the frame.")
                assert w % 2 == 0 and h % 2 == 0
            token = self._region_token(index)
            rendered = template.format(stem=stem, region=token)
            filename = _sanitize_filename(rendered)
            if os.path.splitext(filename)[1].lower() != ".mp4":
                filename = f"{filename}.mp4"
            output_path = _unique_output_path(output_folder, filename, reserved)
            snapshots.append(
                ExportRegion(
                    display_name=self._region_display_name(index),
                    token=token,
                    rect=rect,
                    brightness=float(region.brightness),
                    contrast=float(region.contrast),
                    color_mode=str(region.color_mode),
                    output_path=output_path,
                    angle=float(region.angle),
                    quarter_turns=self.preview_turns if self.export_view_rotation_var.get() else 0,
                )
            )

        return ExportJob(
            ffmpeg=export_ffmpeg,
            video_path=video_path,
            output_folder=output_folder,
            fps=float(self.reader.fps),
            output_fps=output_fps,
            output_speed=output_speed,
            video_encoder=video_encoder,
            frame_count=int(self.reader.frame_count),
            source_width=int(self.reader.width),
            source_height=int(self.reader.height),
            in_frame=int(self.in_frame),
            out_frame=int(self.out_frame),
            regions=tuple(snapshots),
        )

    def _set_export_ui_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for widget in [
            self.browse_video_button,
            self.export_view_rotation_check,
            self.video_entry,
            self.add_crop_button,
            self.delete_crop_button,
            self.output_entry,
            self.output_button,
            self.template_entry,
            self.reset_adjust_button,
            self.brightness_slider,
            self.contrast_slider,
            self.frame_slider,
            self.first_button,
            self.prev_button,
            self.play_button,
            self.next_button,
            self.last_button,
            self.set_in_button,
            self.set_out_button,
            self.in_frame_spin,
            self.out_frame_spin,
        ]:
            widget.configure(state=state)
        self.brightness_entry.configure(state=state)
        self.contrast_entry.configure(state=state)
        self.color_mode_combo.configure(state="disabled" if running else "readonly")
        self.output_speed_spin.configure(state=state)
        self.output_fps_spin.configure(state=state)
        self.gpu_acceleration_check.configure(
            state="disabled" if running or self.hardware_video_encoder is None else "normal"
        )
        self._set_region_list_state(state)
        self._set_crop_controls_state(not running and self._crop_control_index() is not None)
        self.export_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")
        if not running:
            self._refresh_video_controls()

    def cancel_export(self) -> None:
        if not self.export_running:
            return
        self.export_cancel.set()
        with self.export_proc_lock:
            proc = self.export_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
        self.set_status("Canceling export...", auto_clear=False)

    def _expected_export_frame_count(self, job: ExportJob) -> int:
        duration_s = (
            (job.out_frame - job.in_frame + 1)
            / max(1e-9, job.fps)
            / max(OUTPUT_SPEED_MIN, job.output_speed)
        )
        return max(1, int(round(duration_s * max(1e-9, job.output_fps))))

    def _export_worker(self, job: ExportJob) -> None:
        outputs: list[str] = []
        warnings: list[str] = []
        current_output: str | None = None
        try:
            os.makedirs(job.output_folder, exist_ok=True)
            source_reader = VideoFrameReader(job.video_path)
            try:
                encoder_label = HARDWARE_ENCODER_LABELS.get(job.video_encoder, "CPU (libx264)")
                self.export_queue.put(("log", f"Video encoder: {encoder_label}"))
                total_regions = max(1, len(job.regions))
                for region_index, region in enumerate(job.regions):
                    if self.export_cancel.is_set():
                        raise ExportCancelled()
                    current_output = region.output_path
                    self.export_queue.put(("log", f"Exporting {region.display_name}: {region.output_path}"))
                    cmd = self._build_ffmpeg_command(job, region)
                    target_frames = self._expected_export_frame_count(job)
                    start_time = time.perf_counter()
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    with self.export_proc_lock:
                        self.export_proc = proc
                    last_frame = 0
                    try:
                        assert proc.stdout is not None
                        for line in proc.stdout:
                            if self.export_cancel.is_set():
                                try:
                                    proc.terminate()
                                except OSError:
                                    pass
                            line = line.strip()
                            if line.startswith("frame="):
                                try:
                                    last_frame = max(last_frame, int(line.split("=", 1)[1]))
                                except ValueError:
                                    pass
                            elif line.startswith("out_time_ms=") or line.startswith("out_time_us="):
                                try:
                                    micros = int(line.split("=", 1)[1])
                                    last_frame = max(last_frame, int(round((micros / 1_000_000.0) * job.output_fps)))
                                except ValueError:
                                    pass
                            ratio = _clamp(last_frame / max(1, target_frames), 0.0, 1.0)
                            overall = (region_index + ratio) / total_regions
                            now = time.perf_counter()
                            if now - self._last_progress_update > 0.08 or ratio >= 1.0:
                                self._last_progress_update = now
                                self.export_queue.put(
                                    (
                                        "progress",
                                        overall,
                                        f"{region.display_name}: {int(ratio * 100)}%",
                                    )
                                )
                        stderr_text = proc.stderr.read() if proc.stderr is not None else ""
                        return_code = proc.wait()
                    finally:
                        with self.export_proc_lock:
                            self.export_proc = None
                    if self.export_cancel.is_set():
                        raise ExportCancelled()
                    if return_code != 0:
                        self._delete_partial(current_output)
                        detail = stderr_text.strip() or f"ffmpeg exited with code {return_code}"
                        raise RuntimeError(f"Export failed for {region.display_name}:\n{detail}")
                    outputs.append(region.output_path)
                    elapsed = time.perf_counter() - start_time
                    self.export_queue.put(("log", f"Completed {region.display_name} in {elapsed:.1f}s"))
                    warnings.extend(self._verify_export(job, region, source_reader))
            finally:
                source_reader.close()
            self.export_queue.put(("progress", 1.0, "Completed"))
            self.export_queue.put(("done", tuple(outputs), tuple(warnings)))
        except ExportCancelled:
            if current_output is not None:
                self._delete_partial(current_output)
            for path in outputs:
                self._delete_partial(path)
            self.export_queue.put(("cancelled",))
        except Exception as exc:
            if current_output is not None:
                self._delete_partial(current_output)
            self.export_queue.put(("error", str(exc)))

    def _build_ffmpeg_command(self, job: ExportJob, region: ExportRegion) -> list[str]:
        filters: list[str] = [
            f"trim=start_frame={job.in_frame}:end_frame={job.out_frame + 1}",
            (
                f"setpts=N/({max(1e-9, job.fps):.12f}*"
                f"{max(OUTPUT_SPEED_MIN, job.output_speed):.12f}*TB)"
            ),
        ]
        if region.rect is not None:
            x, y, w, h = region.rect
            assert w % 2 == 0 and h % 2 == 0
            if region.angle:
                # Center the selected region before rotating; keep the requested output size.
                margin = math.ceil(math.hypot(w, h) / 2)
                filters.extend(["format=yuv444p",
                    f"pad=iw+{2*margin}:ih+{2*margin}:{margin}:{margin}:black",
                    f"crop={w+2*margin}:{h+2*margin}:{x}:{y}:exact=1",
                    f"rotate={-math.radians(region.angle):.12f}:ow={w}:oh={h}:c=black"])
            else:
                filters.append(f"crop={w}:{h}:{x}:{y}:exact=1")
        if abs(region.brightness) > 1e-9 or abs(region.contrast - 1.0) > 1e-9:
            # eq maps the stored luma, which for ordinary tv-range video spans
            # 16-235, while the preview maps the luma OpenCV hands back after
            # expanding that to 0-255.  Normalise the range so both apply the
            # same curve to the same values, then hand back a limited-range
            # frame for the encoder.  Skipped entirely when nothing is adjusted.
            filters.extend([
                "scale=in_range=auto:out_range=full",
                f"eq=brightness={region.brightness:.6f}:contrast={region.contrast:.6f}",
                "scale=in_range=full:out_range=limited",
            ])
        if region.color_mode == "grayscale":
            filters.append("format=gray")
        turns = (region.quarter_turns + (_crop_quarter_turns(region.angle) if region.rect is not None else 0)) % 4
        if turns == 1:
            filters.append("transpose=clock")
        elif turns == 2:
            filters.extend(["hflip", "vflip"])
        elif turns == 3:
            filters.append("transpose=cclock")
        filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        filters.append(f"fps=fps={job.output_fps:.6f}")

        cmd = [
            job.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            job.video_path,
            "-map",
            "0:v:0",
            "-an",
        ]
        cmd.extend(["-vf", ",".join(filters), *_video_encoder_args(job.video_encoder)])
        cmd.extend(["-progress", "pipe:1", region.output_path])
        return cmd

    def _verify_export(self, job: ExportJob, region: ExportRegion, source_reader: VideoFrameReader) -> list[str]:
        warnings: list[str] = []
        expected_frames = self._expected_export_frame_count(job)
        cap = cv2.VideoCapture(region.output_path)
        if not cap.isOpened():
            return [f"{region.display_name}: could not open exported file for verification."]
        try:
            actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if actual_frames > 0 and abs(actual_frames - expected_frames) > max(1, int(expected_frames * 0.02)):
                warnings.append(
                    f"{region.display_name}: frame count expected about {expected_frames}, got {actual_frames}."
                )
            elif actual_frames <= 0:
                warnings.append(f"{region.display_name}: exported frame count was unavailable.")

            output_frame_idx = max(0, expected_frames // 2)
            if actual_frames > 0:
                output_frame_idx = min(output_frame_idx, actual_frames - 1)
            source_offset_s = (
                output_frame_idx
                / max(1e-9, job.output_fps)
                * max(OUTPUT_SPEED_MIN, job.output_speed)
            )
            source_frame_idx = job.in_frame + int(round(source_offset_s * job.fps))
            source_frame_idx = max(job.in_frame, min(job.out_frame, source_frame_idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, output_frame_idx)
            ok, exported_bgr = cap.read()
            if not ok or exported_bgr is None:
                warnings.append(f"{region.display_name}: representative frame could not be read.")
                return warnings
        finally:
            cap.release()

        best_mad: float | None = None
        best_source_frame_idx: int | None = None
        expected_shape = None
        candidate_indices = range(
            max(job.in_frame, source_frame_idx - 2),
            min(job.out_frame, source_frame_idx + 2) + 1,
        )
        for candidate_idx in candidate_indices:
            try:
                expected_bgr = self._expected_export_frame_bgr(
                    source_reader.read_bgr(candidate_idx), region
                )
            except Exception:
                continue
            if expected_shape is None:
                expected_shape = expected_bgr.shape
            h = min(expected_bgr.shape[0], exported_bgr.shape[0])
            w = min(expected_bgr.shape[1], exported_bgr.shape[1])
            if h <= 0 or w <= 0:
                continue
            diff = np.abs(
                expected_bgr[:h, :w].astype(np.int16)
                - exported_bgr[:h, :w].astype(np.int16)
            )
            mad = float(np.mean(diff))
            if best_mad is None or mad < best_mad:
                best_mad = mad
                best_source_frame_idx = candidate_idx

        if expected_shape is None or best_mad is None:
            warnings.append(f"{region.display_name}: source verification frame could not be read.")
            return warnings
        if expected_shape != exported_bgr.shape:
            warnings.append(
                f"{region.display_name}: verification shape mismatch "
                f"expected {expected_shape[1]}x{expected_shape[0]}, "
                f"got {exported_bgr.shape[1]}x{exported_bgr.shape[0]}."
            )
        if best_mad > VERIFY_MAD_THRESHOLD:
            warnings.append(
                f"{region.display_name}: preview/export mean absolute difference {best_mad:.2f}/255 "
                f"exceeded {VERIFY_MAD_THRESHOLD:.2f}/255."
            )
        else:
            self.export_queue.put(
                (
                    "log",
                    f"{region.display_name}: verification MAD {best_mad:.2f}/255 "
                    f"(source frame {best_source_frame_idx})",
                )
            )
        return warnings

    def _expected_export_frame_bgr(self, source_bgr: np.ndarray, region: ExportRegion) -> np.ndarray:
        frame = source_bgr
        if region.rect is not None:
            x, y, w, h = region.rect
            frame = _crop_rotated_bgr(frame, (x, y, w, h), region.angle)
        frame = _process_region_bgr(frame, region.brightness, region.contrast, region.color_mode)
        if region.quarter_turns:
            frame = np.ascontiguousarray(np.rot90(frame, -region.quarter_turns))
        return _pad_even_bgr(frame)

    def _delete_partial(self, path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def _poll_export_queue(self) -> None:
        try:
            while True:
                item = self.export_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._log(str(item[1]))
                elif kind == "progress":
                    value = float(item[1])
                    self.progress_var.set(value)
                    self.progress_bar.set(value)
                    self.set_status(str(item[2]), auto_clear=False)
                elif kind == "done":
                    outputs = tuple(item[1])
                    warnings = tuple(item[2])
                    self._finish_export_ui()
                    self.set_status(f"Export completed: {len(outputs)} file(s)")
                    if warnings:
                        for warning in warnings:
                            self._log("WARNING: " + warning)
                        messagebox.showwarning(
                            "Export completed with warnings",
                            "Export completed, but verification reported warnings.\n\n"
                            + "\n".join(warnings[:8]),
                        )
                    else:
                        messagebox.showinfo("Completed", f"Exported {len(outputs)} file(s).")
                    return
                elif kind == "cancelled":
                    self._finish_export_ui()
                    self.set_status("Export canceled.")
                    messagebox.showinfo("Canceled", "Export was canceled.")
                    return
                elif kind == "error":
                    self._finish_export_ui()
                    self.set_status("Export failed.", auto_clear=False)
                    messagebox.showerror("Export failed", str(item[1]))
                    return
        except queue.Empty:
            pass
        if self.export_running:
            self.export_poll_job = self.after(100, self._poll_export_queue)

    def _finish_export_ui(self) -> None:
        self.export_running = False
        self.export_thread = None
        self.export_poll_job = None
        self._set_export_ui_running(False)
        with self.export_proc_lock:
            self.export_proc = None

    def on_close(self) -> None:
        if self.export_running:
            if not messagebox.askyesno("Export running", "Cancel export and close?"):
                return
            self.cancel_export()
            deadline = time.time() + 1.5
            while self.export_thread is not None and self.export_thread.is_alive() and time.time() < deadline:
                self.update()
                time.sleep(0.03)
        self.stop_playback(update_button=False)
        self._flush_crop_trimming_config_save()
        if self.status_clear_job is not None:
            try:
                self.after_cancel(self.status_clear_job)
            except tk.TclError:
                pass
        if self.export_poll_job is not None:
            try:
                self.after_cancel(self.export_poll_job)
            except tk.TclError:
                pass
        if self.reader is not None:
            self.reader.close()
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="")
    args = parser.parse_args()

    configure_taskbar_identity()
    app = PreprocessApp()
    if args.video and os.path.isfile(args.video):
        app.after(300, lambda: app.load_video(args.video))
    app.mainloop()


if __name__ == "__main__":
    main()
