# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import importlib.metadata
import platform
import sys
import csv
import hashlib
import json
import math
import pickle
import queue
import re
import shutil
import subprocess
import threading
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import customtkinter as ctk
import yaml

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES, COPY, REFUSE_DROP
    _TKINTERDND2_IMPORT_ERROR = None
except Exception as exc:
    TkinterDnD = None
    DND_FILES = None
    COPY = "copy"
    REFUSE_DROP = "refuse_drop"
    _TKINTERDND2_IMPORT_ERROR = exc

try:
    from .project_paths import PROJECT_ROOT, ensure_import_paths, gui_asset, gui_script
    from .window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
    from .dark_widgets import dark_spinbox
    from .save_shortcut import bind_save_shortcut
except ImportError:  # Preserve direct execution with: python gui/gui_segmentation.py
    from project_paths import PROJECT_ROOT, ensure_import_paths, gui_asset, gui_script
    from window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
    from dark_widgets import dark_spinbox
    from save_shortcut import bind_save_shortcut

ensure_import_paths(PROJECT_ROOT)
from gui.color import (
    OBB_COLOR, OUTLIER_COLOR, ROI_COLOR, WHITE,
    STATIC_OUTLIER_COLOR, TEMPORAL_MERGE_COLOR,
)
from gui.video_input import (
    VIDEO_DROP_SUFFIXES,
    VIDEO_FILETYPES,
    ask_open_analysis_video,
    prepare_analysis_video,
    video_conversion_record,
)
from main.path_utils import resolve_config_paths, resolve_path, to_relative_path
from main.roi import (
    default_roi_set, initial_roi_points,
    normalize_roi_set, load_roi_settings, roi_is_active, roi_signature,
    build_single_roi_mask as _build_single_roi_mask,
    build_roi_mask_for_frame as _build_roi_mask_for_frame,
    contour_mask_region as _blob_mask_region,
    convex_polygon_mask_region as _result_obb_mask,
    mask_region_overlap as _mask_region_overlap,
)
from main.video_frame_count import detect_seekable_frame_count
from gui.preview_key_navigation import ArrowKeyHoldPlayback
from gui.preview_playback_timing import AdaptivePreviewPacer
from gui.attention_border import RotatingBorderHighlight
from gui.reverse_playback_buffer import PreparedFrame, SmoothPreviewPlayback

APP_TITLE = "AMADEUS: Segmentation"
WINDOW_W = 1720
WINDOW_H = 980
WINDOW_MIN_W = 1200
WINDOW_MIN_H = 860
CANVAS_BG = "black"
CTK_THEME = str(gui_asset("deep_green.json"))

VIDEO_EXTS = VIDEO_FILETYPES
PKL_EXTS = [("Pickle files", "*.pickle *.pkl"), ("All files", "*.*")]
PNG_EXTS = [("PNG files", "*.png"), ("All files", "*.*")]
RESULT_CSV_EXTS = [("Result CSV files", "*.csv"), ("All files", "*.*")]

# Imported-result overlay: magenta stays readable next to the cyan/orange blobs.
RESULT_OBB_COLOR = (255, 0, 255)
_TEMPORAL_LEVEL_NAMES = {1: "medium", 2: "high", 3: "uncertain"}
TEMPORAL_EVENTS_CSV = "temporal_merge_events.csv"
# Sentinel: an argument that was not passed (None explicitly means disabled).
_UNSET = object()
# Fraction of a blob that its single imported OBB must cover for the blob to be
# accepted as one large animal instead of an area outlier.
DEFAULT_RESULT_OBB_COVERAGE = 0.7
# Stage 1 rejects OBB aspect ratios strictly below MIN_ASPECT (1.1 by default).
DEFAULT_MIN_OBB_ASPECT = 1.1

ZOOM_IN_FACTOR = 1.12
ZOOM_OUT_FACTOR = 1.0 / ZOOM_IN_FACTOR
MIN_ZOOM = 0.2
MAX_ZOOM = 20.0

FRAME_CACHE_SIZE = 128
# Segmentation keeps larger temporary masks and background stacks than the
# other GUIs. Bound the improved preview history independently of those jobs.
FRAME_CACHE_BYTES = 384 * 1024 * 1024
PREFETCH_FORWARD = 60
PREFETCH_BACKWARD = 60
FRAME_SEEK_BACKTRACK = 64

# Parallel export: one OpenCV thread per Python thread avoids internal contention.
_EXPORT_WORKERS = min(os.cpu_count() or 4, 8)
_BACKGROUND_READ_WORKERS = min(os.cpu_count() or 4, 4)
_ANALYSIS_WORKERS = min(_EXPORT_WORKERS, _BACKGROUND_READ_WORKERS)
BACKGROUND_METHODS = ("median", "max", "min")  # "mean" remains supported internally but is not exposed.
BACKGROUND_SAMPLE_COUNT = 100
# A 4K BGR frame is about 24 MiB.  Keeping 100 float32 frames (the old
# implementation) could require well over 10 GiB and make the GUI appear to
# hang or terminate.  Background frames stay uint8 and the sample count is
# capped so their shared stack remains bounded.
BACKGROUND_STACK_MAX_BYTES = 768 * 1024 * 1024
# Median selection is faster when the frame axis is contiguous.  Keep each
# temporary transposed tile small so it never recreates a second full stack.
BACKGROUND_MEDIAN_TILE_MAX_BYTES = 32 * 1024 * 1024
# Bound speculative forward decoding; widely separated samples still seek.
BACKGROUND_FORWARD_MAX_FRAMES = 256
# Analysis samples are usually farther apart than background samples, but the
# same forward decoder remains cheaper than repeated long-GOP seeks on videos
# whose samples are spread across the full duration.
ANALYSIS_FORWARD_MAX_FRAMES = 512
PROCESSING_BATCH_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_TRAINING_MAX_FRAME_INDEX = 19999
LONG_VIDEO_FRAME_COUNT_THRESHOLD = DEFAULT_TRAINING_MAX_FRAME_INDEX + 1
# Maximum IQR multiplier used for interactive area limits.
AREA_IQR_MULTIPLIER_MAX = 10.0
REGION_EXPAND_SLIDER_MAX_PX = 50
REGION_EXPAND_ENTRY_MAX_PX = 1000
# Handles and hit tolerance stay the same size on screen at any zoom level.
ROI_HANDLE_RADIUS_CANVAS_PX = 6
ROI_HANDLE_GRAB_CANVAS_PX = 10
# The Min blob area slider spans a range fixed once per video; the spinbox keeps
# the wider limit so a larger area can still be typed.
MIN_AREA_ENTRY_MAX = 100000
# One entry holds a full-resolution mask, and dragging the ROI produces a new
# segmentation signature per mouse position, so the cache needs a ceiling.
MASK_CACHE_MAX_ENTRIES = 64

# GPU acceleration via PyTorch -- initialized in a background thread to avoid blocking startup.
_torch = None  # type: ignore
_CUDA_AVAILABLE = False
_torch_init_started = False
cv2 = None  # type: ignore[assignment]
np = None  # type: ignore[assignment]
Image = None  # type: ignore[assignment]
ImageTk = None  # type: ignore[assignment]

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

def _init_torch():
    global _torch, _CUDA_AVAILABLE
    try:
        import torch as t
        _torch = t
        _CUDA_AVAILABLE = t.cuda.is_available()
    except ImportError:
        pass

def _start_torch_init():
    global _torch_init_started
    if _torch_init_started:
        return
    _torch_init_started = True
    threading.Thread(target=_init_torch, daemon=True).start()


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(CTK_THEME)

_SPIN_CFG = dict(
    bg="#343638", fg="#dce4ee",
    disabledbackground="#343638", disabledforeground="#929292",
    insertbackground="#dce4ee",
    buttonbackground="#565b5e",
    relief="flat",
    highlightthickness=2,
    highlightbackground="#565b5e",
    highlightcolor="#1f8040",
    selectbackground="#1f6aa5", selectforeground="white",
)

# Use the same dark palette for dialogs and primary controls on every OS.
_POPUP_BG = "#242424"
_POPUP_FG = "#DCE4EE"


def _popup_button(master, *, text: str, command) -> ctk.CTkButton:
    """Give modal actions an OS-independent dark fill with legible text."""
    return ctk.CTkButton(
        master, text=text, command=command, width=100, height=30,
        fg_color="#225e3b", hover_color="#30764e",
        text_color="#ffffff", corner_radius=5,
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


class VideoFrameReader:
    def __init__(self, video_path: str):
        _ensure_video_modules()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        self.reported_frame_count = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.frame_count = detect_seekable_frame_count(self.video_path, self.reported_frame_count)
        self.frame_count_adjusted = self.frame_count != self.reported_frame_count
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_bytes = max(1, self.width * self.height * 3)
        self.cache_capacity = max(2, min(FRAME_CACHE_SIZE, FRAME_CACHE_BYTES // frame_bytes))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else 0.0

        self.prefetch_cap = cv2.VideoCapture(video_path)
        if not self.prefetch_cap.isOpened():
            self.prefetch_cap = None

        from collections import OrderedDict
        self.cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_lock = threading.RLock()
        self._cap_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()
        self._cap_pos: int = -1
        self._prefetch_pos: int = -1
        self._closed = threading.Event()

    @staticmethod
    def _is_valid_frame(frame) -> bool:
        return frame is not None and getattr(frame, "size", 0) > 0

    def _decode_bgr_from_nearby(
        self, cap: cv2.VideoCapture, frame_idx: int
    ) -> "Optional[tuple[np.ndarray, int, int]]":
        if frame_idx <= 0:
            return None
        start_idx = max(0, frame_idx - FRAME_SEEK_BACKTRACK)
        if start_idx == frame_idx:
            return None

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_idx))
            last_frame = None
            last_idx = start_idx - 1
            for idx in range(start_idx, frame_idx + 1):
                ok, frame = cap.read()
                if not ok or not self._is_valid_frame(frame):
                    break
                last_frame = frame
                last_idx = idx
                if idx == frame_idx:
                    return frame, idx + 1, idx
        except Exception:
            return None

        if last_frame is not None and frame_idx >= max(0, self.frame_count - 1):
            self.frame_count = max(1, min(self.frame_count, last_idx + 1))
            self.frame_count_adjusted = self.frame_count != self.reported_frame_count
            return last_frame, last_idx + 1, last_idx
        return None

    def _decode_bgr(
        self, cap: cv2.VideoCapture, frame_idx: int, current_pos: int,
        *, max_sequential_gap: int = 12,
    ) -> "tuple[np.ndarray, int, int]":
        gap = frame_idx - current_pos
        if current_pos >= 0 and 0 < gap <= max_sequential_gap:
            for _ in range(gap):
                if not cap.grab():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    break
        elif current_pos != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or not self._is_valid_frame(frame):
            recovered = self._decode_bgr_from_nearby(cap, frame_idx)
            if recovered is not None:
                return recovered
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")
        return frame, frame_idx + 1, frame_idx

    def _cache_put(self, frame_idx: int, frame_bgr: np.ndarray):
        with self._cache_lock:
            if frame_idx in self.cache:
                self.cache.move_to_end(frame_idx)
            else:
                self.cache[frame_idx] = frame_bgr
                while len(self.cache) > self.cache_capacity:
                    self.cache.popitem(last=False)

    def _clamp_frame_idx(self, frame_idx: int) -> int:
        frame_idx = int(frame_idx)
        if self.frame_count > 0:
            frame_idx = max(0, min(self.frame_count - 1, frame_idx))
        return frame_idx

    def read_bgr_with_index(
        self, frame_idx: int, *, max_sequential_gap: int = 12,
    ) -> "tuple[np.ndarray, int]":
        if self._closed.is_set():
            raise RuntimeError("Video reader is closed.")
        frame_idx = self._clamp_frame_idx(frame_idx)
        with self._cache_lock:
            if frame_idx in self.cache:
                self.cache.move_to_end(frame_idx)
                return self.cache[frame_idx].copy(), frame_idx
        with self._cap_lock:
            with self._cache_lock:
                if frame_idx in self.cache:
                    self.cache.move_to_end(frame_idx)
                    return self.cache[frame_idx].copy(), frame_idx
            frame, self._cap_pos, actual_idx = self._decode_bgr(self.cap, frame_idx, self._cap_pos,
                                                                  max_sequential_gap=max_sequential_gap)
            self._cache_put(actual_idx, frame)
            return frame.copy(), actual_idx

    def read_bgr(self, frame_idx: int, *, max_sequential_gap: int = 12) -> np.ndarray:
        frame, _actual_idx = self.read_bgr_with_index(frame_idx, max_sequential_gap=max_sequential_gap)
        return frame

    def prefetch(self, frame_idx: int):
        if self._closed.is_set() or self.prefetch_cap is None:
            return
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return
        with self._cache_lock:
            if frame_idx in self.cache:
                return
        with self._prefetch_lock:
            with self._cache_lock:
                if frame_idx in self.cache:
                    return
            try:
                frame, self._prefetch_pos, actual_idx = self._decode_bgr(
                    self.prefetch_cap, frame_idx, self._prefetch_pos)
            except Exception:
                self._prefetch_pos = -1
                return
            self._cache_put(actual_idx, frame)

    def close(self):
        """Release captures after outstanding decoder calls finish, off Tk."""
        if self._closed.is_set():
            return
        self._closed.set()

        def cleanup():
            with self._cap_lock:
                self.cap.release()
            with self._prefetch_lock:
                if self.prefetch_cap is not None:
                    self.prefetch_cap.release()
                    self.prefetch_cap = None

        threading.Thread(target=cleanup, name="segmentation-video-cleanup",
                         daemon=True).start()

    def clear_cache(self):
        """Release preview frames before a memory-intensive batch operation."""
        with self._cap_lock:
            with self._prefetch_lock:
                with self._cache_lock:
                    self.cache.clear()


@dataclass
class BlobMetrics:
    frame: int
    blob_index: int
    contour: np.ndarray
    center_x: float
    center_y: float
    area: float
    bbox_area: float
    w: float
    h: float
    area_outlier: bool = False
    bbox_area_outlier: bool = False
    width_outlier: bool = False
    height_outlier: bool = False
    low_obb_aspect_outlier: bool = False
    persistent_static_outlier: bool = False
    temporal_merge_outlier: bool = False   # excluded by temporal merge evidence
    temporal_merge_animals: int = 0       # minimum animal count inferred by forward/backward evidence
    temporal_merge_level: int = 0          # 2 high, 1 medium candidate, 0 none
    manual_outlier: bool = False
    is_crossing: bool = False
    result_obb_count: int = 0        # imported OBBs owned by this blob
    result_obb_coverage: float = 0.0  # blob fraction covered by that single OBB
    result_single: bool = False       # area outlier rescued as one large animal
    result_missing: bool = False      # full imported OBB with no segmentation owner
    result_area_outlier: bool = False # rejected by the independent Result area bounds
    result_obb_indices: tuple[int, ...] = ()
    result_blob_area: Optional[int] = None  # original segment union inside OBB (Result threshold)
    result_generated: bool = False
    result_protected: bool = False
    result_source_index: Optional[int] = None
    result_source_frame: Optional[tuple] = field(default=None, repr=False, compare=False)


@dataclass
class _SegConfig:
    """Immutable snapshot of all segmentation parameters for thread-safe parallel processing."""
    mode: str
    ksize: int
    threshold: int
    dark_threshold: int
    bright_threshold: int
    diff_threshold: int
    min_area: float
    open_iter: int
    close_iter: int
    fill_holes: bool
    invert_mask: bool
    expand_px: int                         # outward growth of every blob outline, in pixels
    expand_merge_only: bool                # keep only the growth that merges separate regions
    roi_sets: list                         # polygon snapshots; masks use the shared ROI implementation
    roi_reverse: bool                      # reverse the completed union, never individual sets
    additional_outlier_sets: list          # segmentation settings; converted to blobs per frame
    background_bgr: Optional[np.ndarray]  # read-only shared reference
    additional_backgrounds: dict[str, np.ndarray] = field(default_factory=dict)
    # Bounds used to identify the normal non-outlier blobs whose pixels must be
    # protected from Additional Outlier segmentation.  During the first IQR
    # analysis pass this is None, so every normal blob is protected.
    protected_area_bounds: Optional[tuple[float, float]] = None
    single_blob_smoothing_enabled: bool = False
    single_blob_smoothing_level: int = 3
    # Backend chosen once per snapshot, so one run never switches between the
    # CPU and GPU paths when the deferred torch import completes mid-run.
    use_gpu: bool = False


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    # Keep OpenCV lazy-loaded at GUI startup; share the tracking implementation.
    from main.segmentation_core import fill_binary_holes
    return fill_binary_holes(mask)


_BACKGROUND_SEGMENTATION_MODES = {
    "background_diff",
    "dark_region_and_background_diff",
    "bright_region_and_background_diff",
}
def _mode_requires_background(mode: str) -> bool:
    return mode in _BACKGROUND_SEGMENTATION_MODES


def _segment_mask_with_settings(
    frame_bgr: np.ndarray,
    frame_idx: int,
    cfg: "_SegConfig",
    *,
    mode: str,
    threshold: int,
    dark_threshold: int,
    bright_threshold: int,
    diff_threshold: int,
    background_bgr: Optional[np.ndarray],
    expand: bool,
) -> np.ndarray:
    """Build one binary mask from a plain segmentation-settings snapshot."""
    h, w = frame_bgr.shape[:2]
    if _mode_requires_background(mode) and background_bgr is None:
        return np.zeros((h, w), dtype=np.uint8)

    local_cfg = replace(
        cfg,
        mode=mode,
        threshold=int(threshold),
        dark_threshold=int(dark_threshold),
        bright_threshold=int(bright_threshold),
        diff_threshold=int(diff_threshold),
        background_bgr=background_bgr,
    )
    if local_cfg.use_gpu and _torch is not None and _mode_requires_background(mode):
        mask = _segment_diff_gpu(frame_bgr, local_cfg)
    else:
        mask = _segment_cpu(frame_bgr, local_cfg)

    if local_cfg.invert_mask:
        mask = cv2.bitwise_not(mask)
    roi_mask = _build_roi_mask_for_frame(local_cfg.roi_sets, frame_bgr.shape[:2], frame_idx, local_cfg.roi_reverse)
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    if local_cfg.open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=local_cfg.open_iter)
    if local_cfg.close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=local_cfg.close_iter)

    if local_cfg.fill_holes:
        mask = _fill_binary_holes(mask)
        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, roi_mask)

    if expand and local_cfg.expand_px > 0:
        mask = _expand_mask(mask, local_cfg.expand_px, local_cfg.expand_merge_only)
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)
    return mask


def _extract_blob_metrics(
    mask: np.ndarray,
    frame_idx: int,
    min_area: float,
    *,
    manual_outlier: bool = False,
) -> list[BlobMetrics]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[BlobMetrics] = []
    for cnt in contours:
        area = contour_area(cnt)
        if area < float(min_area):
            continue
        _, _, bw, bh = cv2.boundingRect(cnt)
        cx, cy = contour_center(cnt)
        blobs.append(BlobMetrics(
            frame=int(frame_idx), blob_index=len(blobs),
            contour=cnt.astype(np.int32), center_x=cx, center_y=cy,
            area=area, bbox_area=float(bw * bh), w=float(bw), h=float(bh),
            manual_outlier=bool(manual_outlier),
            is_crossing=bool(manual_outlier),
        ))
    return blobs


def _draw_blob_mask(image_shape: tuple, blobs: list[BlobMetrics]) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if blobs:
        cv2.drawContours(mask, [blob.contour for blob in blobs], -1, 255, thickness=cv2.FILLED)
    return mask


def _draw_blob_union_mask(image_shape: tuple, blobs: list[BlobMetrics]) -> np.ndarray:
    """Draw overlapping blob contours as their foreground union."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for blob in blobs:
        cv2.drawContours(mask, [blob.contour], -1, 255, thickness=cv2.FILLED)
    return mask


def _blob_with_contour(blob: BlobMetrics, contour: np.ndarray) -> BlobMetrics:
    """Refresh geometry before the caller applies any post-smoothing classification."""
    _, _, w, h = cv2.boundingRect(contour)
    cx, cy = contour_center(contour)
    return replace(
        blob, contour=contour, center_x=cx, center_y=cy,
        area=contour_area(contour), bbox_area=float(w * h), w=float(w), h=float(h),
    )


def _smooth_classified_blobs(
    blobs: list[BlobMetrics], image_shape: tuple, cfg: _SegConfig,
) -> list[BlobMetrics]:
    """Smooth single-animal contours after classification and Additional Outlier.

    Raw masks and sampled statistics remain untouched. Removed pixels are
    background, even where an Additional Outlier overlaps a rescued animal.
    Callers reapply the existing area bounds to the returned geometry.
    """
    if not cfg.single_blob_smoothing_enabled:
        return blobs
    blobs = _restore_result_inputs(blobs)
    from main.segmentation_core import smooth_single_animal_contour, solid_contours_from_mask

    removed = np.zeros(image_shape[:2], dtype=np.uint8)
    smoothed = []
    for blob in blobs:
        if blob.is_crossing or blob.manual_outlier:
            smoothed.append(blob)
            continue
        contour = smooth_single_animal_contour(blob.contour, cfg.single_blob_smoothing_level)
        if np.array_equal(contour, blob.contour):
            smoothed.append(blob)
            continue
        x, y, w, h = cv2.boundingRect(blob.contour)
        before = np.zeros((h, w), dtype=np.uint8)
        after = np.zeros_like(before)
        cv2.drawContours(before, [blob.contour], -1, 255, cv2.FILLED, offset=(-x, -y))
        cv2.drawContours(after, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
        removed[y:y + h, x:x + w] |= cv2.bitwise_and(before, cv2.bitwise_not(after))
        smoothed.append(_blob_with_contour(blob, contour))

    if not cv2.countNonZero(removed):
        return smoothed
    output = []
    for blob in smoothed:
        if blob.result_missing:
            output.append(blob)
            continue
        if not (blob.is_crossing or blob.manual_outlier):
            output.append(blob)
            continue
        x, y, w, h = cv2.boundingRect(blob.contour)
        excluded = removed[y:y + h, x:x + w]
        if not cv2.countNonZero(excluded):
            output.append(blob)
            continue
        residual = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(residual, [blob.contour], -1, 255, cv2.FILLED, offset=(-x, -y))
        if not cv2.countNonZero(cv2.bitwise_and(residual, excluded)):
            output.append(blob)
            continue
        residual[excluded != 0] = 0
        offset = np.array([[[x, y]]], dtype=np.int32)
        for contour in solid_contours_from_mask(residual):
            output.append(_blob_with_contour(blob, contour + offset))
    return [replace(blob, blob_index=index) for index, blob in enumerate(output)]


def _build_protected_blue_mask(
    normal_blobs: list[BlobMetrics],
    image_shape: tuple,
    bounds: Optional[tuple[float, float]],
) -> np.ndarray:
    """Rasterize the normal non-outlier blobs that Additional Outlier cannot change."""
    protected = np.zeros(image_shape[:2], dtype=np.uint8)
    for blob in normal_blobs:
        if blob.manual_outlier:
            continue
        if bounds is not None and not inside_bounds(blob.area, bounds):
            continue
        cv2.drawContours(protected, [blob.contour], -1, 255, thickness=cv2.FILLED)
    return protected


def _build_additional_residual_mask_for_frame(
    frame_bgr: np.ndarray,
    cfg: "_SegConfig",
    frame_idx: int,
    protected_blue_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Segment enabled Additional Outlier Sets and remove protected blue pixels."""
    additional_mask: Optional[np.ndarray] = None
    for item in cfg.additional_outlier_sets:
        if not item.get("enabled", False):
            continue
        fs = int(item.get("frame_start", 0))
        fe = int(item.get("frame_end", -1))
        if frame_idx < fs or (fe >= 0 and frame_idx > fe):
            continue

        mode = str(item.get("segmentation_mode", "background_diff"))
        background = cfg.additional_backgrounds.get(str(item.get("bg_method", "median")))
        candidate = _segment_mask_with_settings(
            frame_bgr,
            frame_idx,
            cfg,
            mode=mode,
            threshold=int(item.get("threshold", 50)),
            dark_threshold=int(item.get("dark_threshold", 50)),
            bright_threshold=int(item.get("bright_threshold", 200)),
            diff_threshold=int(item.get("diff_threshold", 50)),
            background_bgr=background,
            expand=False,
        )
        if cv2.countNonZero(candidate) == 0:
            continue

        # Apply the Set's minimum area before and after protection.  The first
        # pass removes tiny candidates; the second pass removes tiny residuals
        # left after a candidate overlaps a protected blue blob.
        min_area = float(max(1, int(item.get("min_area", 10))))
        candidate = _mask_from_contours_above_area(candidate, min_area)
        residual = cv2.bitwise_and(candidate, cv2.bitwise_not(protected_blue_mask))
        residual = _mask_from_contours_above_area(residual, min_area)
        if cv2.countNonZero(residual) == 0:
            continue
        additional_mask = (
            residual
            if additional_mask is None
            else cv2.bitwise_or(additional_mask, residual)
        )
    return additional_mask


def _mask_from_contours_above_area(mask: np.ndarray, min_area: float) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    for contour in contours:
        if contour_area(contour) >= float(min_area):
            cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)
    return filtered


def _segment_with_config(frame_bgr: np.ndarray, frame_idx: int, cfg: "_SegConfig") -> "tuple[np.ndarray, list[BlobMetrics]]":
    """Segment normal blobs and append only Additional Outlier residual blobs."""
    normal_mask = _segment_mask_with_settings(
        frame_bgr,
        frame_idx,
        cfg,
        mode=cfg.mode,
        threshold=cfg.threshold,
        dark_threshold=cfg.dark_threshold,
        bright_threshold=cfg.bright_threshold,
        diff_threshold=cfg.diff_threshold,
        background_bgr=cfg.background_bgr,
        expand=True,
    )
    normal_blobs = _extract_blob_metrics(normal_mask, frame_idx, cfg.min_area)
    protected_blue_mask = _build_protected_blue_mask(
        normal_blobs,
        frame_bgr.shape[:2],
        cfg.protected_area_bounds,
    )
    additional_mask = _build_additional_residual_mask_for_frame(
        frame_bgr, cfg, frame_idx, protected_blue_mask
    )
    additional_blobs = (
        _extract_blob_metrics(additional_mask, frame_idx, 0, manual_outlier=True)
        if additional_mask is not None
        else []
    )
    blobs = normal_blobs + additional_blobs
    for index, blob in enumerate(blobs):
        blob.blob_index = index
    return _draw_blob_mask(frame_bgr.shape, blobs), blobs


def _segment_blobs_with_config(
    frame_bgr: np.ndarray,
    frame_idx: int,
    cfg: "_SegConfig",
) -> list[BlobMetrics]:
    """Segment one frame and release its full-size mask before returning."""
    _mask, blobs = _segment_with_config(frame_bgr, frame_idx, cfg)
    return blobs


def _segment_cpu(frame_bgr: np.ndarray, cfg: "_SegConfig") -> np.ndarray:
    """CPU implementation of the threshold/diff pipeline."""
    mode, ksize = cfg.mode, cfg.ksize
    if mode in {"dark_region", "bright_region"}:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if ksize > 1:
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        ttype = cv2.THRESH_BINARY_INV if mode == "dark_region" else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, cfg.threshold, 255, ttype)
    elif mode == "background_diff":
        diff_gray = cv2.cvtColor(cv2.absdiff(frame_bgr, cfg.background_bgr), cv2.COLOR_BGR2GRAY)
        if ksize > 1:
            diff_gray = cv2.GaussianBlur(diff_gray, (ksize, ksize), 0)
        _, mask = cv2.threshold(diff_gray, cfg.threshold, 255, cv2.THRESH_BINARY)
    else:  # hybrid modes
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.cvtColor(cv2.absdiff(frame_bgr, cfg.background_bgr), cv2.COLOR_BGR2GRAY)
        if ksize > 1:
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            diff_gray = cv2.GaussianBlur(diff_gray, (ksize, ksize), 0)
        ttype = cv2.THRESH_BINARY_INV if "dark" in mode else cv2.THRESH_BINARY
        tval = cfg.dark_threshold if "dark" in mode else cfg.bright_threshold
        _, imask = cv2.threshold(gray, tval, 255, ttype)
        _, dmask = cv2.threshold(diff_gray, cfg.diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(imask, dmask)
    return mask


# GPU helpers
# Pre-built Gaussian kernels and background tensors are cached per (ksize, device).
_gpu_gauss_cache: dict = {}
_gpu_bg_cache: dict = {}   # key: id(background_bgr array) -> cuda Tensor


def _get_gauss_kernel_gpu(ksize: int):
    if ksize not in _gpu_gauss_cache:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  # OpenCV default sigma
        ax = _torch.arange(ksize, dtype=_torch.float32) - ksize // 2
        g = _torch.exp(-ax ** 2 / (2 * sigma ** 2))
        k2d = g.outer(g)
        k2d /= k2d.sum()
        _gpu_gauss_cache[ksize] = k2d.view(1, 1, ksize, ksize).cuda()
    return _gpu_gauss_cache[ksize]


def _segment_diff_gpu(frame_bgr: np.ndarray, cfg: "_SegConfig") -> np.ndarray:
    """
    GPU-accelerated diff/threshold path using PyTorch CUDA.
    Falls back to CPU on any error (e.g. OOM).
    """
    try:
        import torch.nn.functional as F
        bg = cfg.background_bgr

        # Upload frame & background (background is cached as a GPU tensor).
        bg_key = id(bg)
        if bg_key not in _gpu_bg_cache:
            _gpu_bg_cache[bg_key] = _torch.from_numpy(bg).float().cuda()
        bg_t = _gpu_bg_cache[bg_key]                         # HxWx3
        fr_t = _torch.from_numpy(frame_bgr).float().cuda()   # HxWx3

        # Absolute diff, convert to grayscale (BGR weights: B=0.114 G=0.587 R=0.299)
        diff = _torch.abs(fr_t - bg_t)
        # BGR -> gray: dim 2 channels are B,G,R
        diff_gray = 0.114 * diff[:, :, 0] + 0.587 * diff[:, :, 1] + 0.299 * diff[:, :, 2]

        if cfg.mode in {"dark_region_and_background_diff", "bright_region_and_background_diff"}:
            gray = 0.114 * fr_t[:, :, 0] + 0.587 * fr_t[:, :, 1] + 0.299 * fr_t[:, :, 2]

        if cfg.ksize > 1:
            k = _get_gauss_kernel_gpu(cfg.ksize)
            pad = cfg.ksize // 2
            diff_gray = F.conv2d(diff_gray.unsqueeze(0).unsqueeze(0), k, padding=pad).squeeze()
            if cfg.mode in {"dark_region_and_background_diff", "bright_region_and_background_diff"}:
                gray = F.conv2d(gray.unsqueeze(0).unsqueeze(0), k, padding=pad).squeeze()

        # Round float32 to nearest integer before threshold comparison to match the
        # CPU/uint8 path: cv2.GaussianBlur on uint8 input produces uint8 output, so a
        # float32 value of 50.3 becomes uint8 50 on CPU (50 > 50 == False) but stays
        # 50.3 on GPU (50.3 > 50 == True), producing larger blobs and causing IQR
        # discrepancies between the GUI preview (CPU) and the pickle export (GPU).
        if cfg.mode == "background_diff":
            # background_diff uses cfg.threshold (not cfg.diff_threshold which is for hybrid modes)
            mask_t = (diff_gray.round() > cfg.threshold).to(_torch.uint8)
        else:
            diff_mask = (diff_gray.round() > cfg.diff_threshold).to(_torch.uint8)
            tval = cfg.dark_threshold if "dark" in cfg.mode else cfg.bright_threshold
            if "dark" in cfg.mode:
                int_mask = (gray.round() <= tval).to(_torch.uint8)
            else:
                int_mask = (gray.round() > tval).to(_torch.uint8)
            mask_t = int_mask & diff_mask

        # Download to CPU (mask_t is HxW uint8, values 0 or 1 -> *255 for OpenCV)
        return (mask_t * 255).byte().cpu().numpy()
    except Exception:
        return _segment_cpu(frame_bgr, cfg)


def contour_center(cnt: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) > 1e-8:
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    pts = cnt.reshape(-1, 2).astype(np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def contour_area(cnt: np.ndarray) -> float:
    return float(abs(cv2.contourArea(cnt)))


# Merge-only corridors below are the exact shortest connection between two
# regions; one pixel of slack is what keeps that set connected once it is
# rasterized. Distances there use DIST_MASK_5 rather than DIST_MASK_PRECISE:
# the precise transform is not bit-reproducible across identical calls, and a
# segmentation run has to give the same mask every time.
_BRIDGE_TOLERANCE_PX = 1.0
_BRIDGE_DIST_MASK = 5


def _draw_roi_polygons(base, roi_sets: list, frame_idx: int, reverse: bool = False):
    drawn = False
    for roi in roi_sets:
        if roi_is_active(roi, frame_idx) and roi["points"]:
            cv2.polylines(base, [np.asarray(roi["points"], dtype=np.int32)], True, ROI_COLOR, 2, cv2.LINE_AA)
            drawn = True
    if drawn and reverse:
        cv2.putText(base, "REV", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_COLOR, 1, cv2.LINE_AA)


def _disk_kernel(radius: int) -> np.ndarray:
    """Structuring element of every pixel within `radius` of the centre."""
    offsets = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    return (np.hypot(dy, dx) <= radius).astype(np.uint8)


def _expand_mask(mask: np.ndarray, expand_px: int, merge_only: bool) -> np.ndarray:
    """Grow every blob outline outward by expand_px pixels (Euclidean).

    With merge_only, only the expansion that actually joins separate regions
    is kept: inside each group of regions that the expansion merges, the
    shortest corridor is filled in for every edge of a minimum spanning tree
    over the gaps between them.  The group therefore ends up as one region
    exactly as it would under the full expansion, while isolated blobs keep
    their original outline -- the intended use is closing notches and
    unintended gaps without inflating every blob.
    """
    expand_px = int(expand_px)
    if expand_px <= 0 or cv2.countNonZero(mask) == 0:
        return mask
    expanded = cv2.dilate(mask, _disk_kernel(expand_px))
    if not merge_only:
        return expanded
    return _keep_merging_expansion(mask, expanded)


def _keep_merging_expansion(mask: np.ndarray, expanded: np.ndarray) -> np.ndarray:
    n_src, src_labels = cv2.connectedComponents(mask, connectivity=8)
    if n_src <= 2:  # background plus at most one region: nothing can merge
        return mask
    _n_dst, dst_labels, dst_stats, _centroids = cv2.connectedComponentsWithStats(expanded, connectivity=8)
    foreground = mask > 0
    pairs = np.unique(np.stack([dst_labels[foreground], src_labels[foreground]], axis=1), axis=0)
    groups: dict[int, list[int]] = {}
    for dst_label, src_label in pairs:
        groups.setdefault(int(dst_label), []).append(int(src_label))

    out = mask.copy()
    for dst_label, group in groups.items():
        if len(group) < 2:
            continue
        x0 = int(dst_stats[dst_label, cv2.CC_STAT_LEFT])
        y0 = int(dst_stats[dst_label, cv2.CC_STAT_TOP])
        x1 = x0 + int(dst_stats[dst_label, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(dst_stats[dst_label, cv2.CC_STAT_HEIGHT])
        inside = dst_labels[y0:y1, x0:x1] == dst_label
        src_roi = src_labels[y0:y1, x0:x1]
        dists = [
            cv2.distanceTransform(
                np.where(src_roi == src_label, 0, 255).astype(np.uint8),
                cv2.DIST_L2, _BRIDGE_DIST_MASK,
            )
            for src_label in group
        ]
        out[y0:y1, x0:x1][_merge_corridors(inside, src_roi, group, dists)] = 255
    return out


def _merge_corridors(inside, src_roi, group, dists) -> np.ndarray:
    """Corridors joining every region of one merged group.

    Edges come from a minimum spanning tree over the gaps between the regions,
    so the group is connected with the shortest corridors and nothing more.
    Corridor pixels that reach no region are dropped: they would otherwise
    show up as spurious extra blobs.
    """
    size = len(group)
    gaps = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i + 1, size):
            gaps[i, j] = gaps[j, i] = float(np.where(inside, dists[i] + dists[j], np.inf).min())

    bridge = np.zeros(inside.shape, dtype=bool)
    joined = [0]
    remaining = set(range(1, size))
    while remaining:
        i, j, gap = min(
            ((a, b, gaps[a, b]) for a in joined for b in sorted(remaining)),
            key=lambda edge: edge[2],
        )
        bridge |= inside & (dists[i] + dists[j] <= gap + _BRIDGE_TOLERANCE_PX)
        joined.append(j)
        remaining.discard(j)

    sources = np.isin(src_roi, group)
    _n_labels, labels = cv2.connectedComponents((sources | bridge).astype(np.uint8), connectivity=8)
    return bridge & np.isin(labels, np.unique(labels[sources]))


@dataclass
class _ResultObbImport:
    """OBB polygons per frame, read from a tracked result CSV."""
    path: str
    obbs_by_frame: dict          # frame index -> (N, 4, 2) float32 corner array
    track_ids: list
    frame_min: int
    frame_max: int
    obb_count: int


@dataclass
class _ResultMatchConfig:
    """Snapshot of the imported-result settings for thread-safe classification."""
    min_coverage: float
    obbs_by_frame: dict          # shared read-only reference
    area_bounds: tuple[float, float] = (0.0, float("inf"))


def _obb_corners_from_rect(cx: float, cy: float, w: float, h: float, heading_deg: float) -> np.ndarray:
    """Rebuild the four OBB corners stored as (cx, cy, w, h, heading) in a result CSV.

    ``w`` is the extent along the long axis, ``h`` along the short axis, and
    ``heading`` is measured in degrees clockwise from the upward image
    direction -- the same convention used by the imported-result analysis.
    """
    theta = math.radians(float(heading_deg))
    long_x, long_y = math.sin(theta), -math.cos(theta)
    short_x, short_y = -long_y, long_x
    half_w, half_h = 0.5 * float(w), 0.5 * float(h)
    return np.array([
        [cx - long_x * half_w - short_x * half_h, cy - long_y * half_w - short_y * half_h],
        [cx + long_x * half_w - short_x * half_h, cy + long_y * half_w - short_y * half_h],
        [cx + long_x * half_w + short_x * half_h, cy + long_y * half_w + short_y * half_h],
        [cx - long_x * half_w + short_x * half_h, cy - long_y * half_w + short_y * half_h],
    ], dtype=np.float32)


def _load_result_obb_csv(path: str) -> "_ResultObbImport":
    """Read a final tracking result CSV into per-frame OBB polygons."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name).strip() for name in (reader.fieldnames or [])]
        if "frame" not in fieldnames:
            raise ValueError("The selected CSV has no 'frame' column; it is not a tracking result CSV.")
        track_ids = sorted(
            int(match.group(1))
            for name in fieldnames
            for match in [re.fullmatch(r"cx(\d+)", name)]
            if match is not None
        )
        if not track_ids:
            raise ValueError("The selected CSV has no cx<ID> columns; it is not a tracking result CSV.")

        obbs_by_frame: dict[int, np.ndarray] = {}
        obb_count = 0
        for row in reader:
            try:
                frame_idx = int(float(row["frame"]))
            except (TypeError, ValueError, KeyError):
                continue
            corners = []
            for tid in track_ids:
                try:
                    cx, cy, w, h, heading = (
                        float(row[f"{name}{tid}"]) for name in ("cx", "cy", "w", "h", "heading")
                    )
                except (TypeError, ValueError, KeyError):
                    continue
                if not all(math.isfinite(v) for v in (cx, cy, w, h, heading)):
                    continue
                if w <= 0.0 or h <= 0.0:
                    continue
                corners.append(_obb_corners_from_rect(cx, cy, w, h, heading))
            if corners:
                obbs_by_frame[frame_idx] = np.stack(corners)
                obb_count += len(corners)

    if not obbs_by_frame:
        raise ValueError("The selected CSV contains no usable OBB rows.")
    frames = obbs_by_frame.keys()
    return _ResultObbImport(
        path=os.path.abspath(path),
        obbs_by_frame=obbs_by_frame,
        track_ids=track_ids,
        frame_min=min(frames),
        frame_max=max(frames),
        obb_count=obb_count,
    )


def _match_result_obbs_to_blobs(blobs: list, frame_idx: int, cfg: "_ResultMatchConfig", image_shape: tuple) -> list:
    """Attach the imported OBBs of one frame to the blobs of that same frame.

    First resolve all centre hits; two centres in a merged blob still give
    that blob a count of 2. Then match remaining OBBs to unclaimed blobs by
    largest raster overlap, without borrowing a blob owned by another OBB.
    The covered fraction distinguishes one animal from a merged blob whose
    second individual the tracker missed. Measure it for every real single-
    owned blob so Result area bounds can reject an otherwise ordinary inlier.
    Return the frame's canonical OBB mask regions for the subsequent area and
    outlier-expansion decisions.
    """
    for blob in blobs:
        blob.result_obb_count = 0
        blob.result_obb_coverage = 0.0
        blob.result_single = False
        blob.result_obb_indices = ()
    obbs = cfg.obbs_by_frame.get(int(frame_idx))
    if obbs is None:
        return []

    regions = [_result_obb_mask(poly, image_shape) for poly in obbs]
    blob_regions = [_blob_mask_region(blob.contour, image_shape) for blob in blobs]
    boxes = [cv2.boundingRect(blob.contour) for blob in blobs]
    owners: list[int] = [-1] * len(obbs)
    center_candidates = sorted(range(len(blobs)), key=lambda i: blobs[i].manual_outlier)
    for ri, poly in enumerate(obbs):
        center = (float(poly[:, 0].mean()), float(poly[:, 1].mean()))
        for index in center_candidates:
            blob = blobs[index]
            bx, by, bw, bh = boxes[index]
            if not (bx <= center[0] <= bx + bw and by <= center[1] <= by + bh):
                continue
            if cv2.pointPolygonTest(blob.contour, center, False) >= 0:
                owners[ri] = index
                break

    claimed = {owner for owner in owners if owner >= 0}
    available = [index for index in range(len(blobs)) if index not in claimed]
    candidates = []
    for ri, poly in enumerate(obbs):
        if owners[ri] >= 0:
            continue
        poly_key = tuple(sorted((float(x), float(y)) for x, y in poly))
        poly_area = float(cv2.contourArea(poly))
        for index, overlap in _result_obb_overlap_candidates(blob_regions, regions[ri], available):
            candidates.append((-overlap, poly_area, poly_key, boxes[index], ri, index))
    # Strongest overlap wins globally. Equal overlap prefers the tighter OBB;
    # geometry breaks ties so CSV ID/column order does not choose the owner.
    for _, _, _, _, ri, index in sorted(candidates):
        if owners[ri] < 0 and index not in claimed:
            owners[ri] = index
            claimed.add(index)
    for ri, owner in enumerate(owners):
        if owner >= 0:
            blobs[owner].result_obb_count += 1
            blobs[owner].result_obb_indices += (ri,)

    for index, blob in enumerate(blobs):
        if blob.result_obb_count != 1 or blob.manual_outlier:
            continue
        mask = blob_regions[index][2]
        blob_pixels = cv2.countNonZero(mask) if mask.size else 0
        covered_pixels = _mask_region_overlap(blob_regions[index], regions[owners.index(index)])
        if blob_pixels > 0:
            blob.result_obb_coverage = covered_pixels / blob_pixels
    return regions


def _restore_result_inputs(blobs: list[BlobMetrics]) -> list[BlobMetrics]:
    """Recover real frame inputs after a prior expansion/protection pass."""
    frames = {}
    for blob in blobs:
        if blob.result_source_frame is not None:
            frames.setdefault(blob.frame, blob.result_source_frame)
    if not frames:
        return blobs
    restored = [replace(b, contour=b.contour.copy(), result_source_frame=None)
                for sources in frames.values() for b in sources]
    restored.extend(b for b in blobs if b.frame not in frames)
    return restored


def _segmentation_pickle_blob(blob: BlobMetrics, source_index: Optional[int] = None) -> SimpleNamespace:
    """Keep exclusion provenance separate from the contour used for erasure."""
    item = SimpleNamespace(contour=blob.contour.astype(np.int32), is_outlier=bool(blob.is_crossing))
    if blob.result_protected:
        item.result_protected = True
    if blob.result_generated or blob.result_missing:
        exclusion = "synthetic"
    elif blob.manual_outlier:
        exclusion = "manual"
    elif blob.temporal_merge_outlier:
        exclusion = "temporal_merge"
    elif blob.persistent_static_outlier:
        exclusion = "persistent_static"
    elif blob.low_obb_aspect_outlier:
        exclusion = "low_obb_aspect"
    elif blob.result_obb_count > 1:
        exclusion = "result_multiple"
    elif not blob.is_crossing:
        exclusion = ""
    elif blob.result_area_outlier:
        exclusion = "result_area"
    elif blob.area_outlier or blob.bbox_area_outlier or blob.width_outlier or blob.height_outlier:
        exclusion = "statistical"
    else:
        exclusion = "result_overlap"
    if exclusion:
        item.segmentation_exclusion = exclusion
    if blob.temporal_merge_level:
        # Provenance survives a persistent_static primary reason; Medium
        # candidates are recorded but never excluded unless chosen.
        item.temporal_merge = _TEMPORAL_LEVEL_NAMES.get(int(blob.temporal_merge_level), "")
        item.temporal_merge_excluded = bool(blob.temporal_merge_outlier)
    if blob.result_obb_count:
        item.result_obb_count = int(blob.result_obb_count)
    if source_index is not None:
        item.segmentation_source_index = int(source_index)
    return item


@dataclass
class _TemporalSearchState:
    """Immutable outcome of one temporal-merge Search (graph kept for re-evaluation)."""
    geometry_key: tuple
    kinds_key: tuple
    eval_key: tuple
    reclassifiable: bool
    start: int
    end: int
    graph: object
    result: object
    params: object


@dataclass
class _FilterPlan:
    """Optional exclusion filters for one preview, Export or Processing run.

    ``mode="apply"`` uses Search results only on frames whose blob digest
    matches the Search. ``mode="processing"`` uses them only when every frame
    matches and otherwise recomputes both filters on Processing's own blobs,
    so an exclusion can never land on a different blob.
    ``static`` is (exclusions, digests, first frame, scanner arguments).
    """
    mode: str
    static: Optional[tuple] = None
    temporal: Optional[_TemporalSearchState] = None
    stats: dict = field(default_factory=dict)


def _apply_temporal_levels(blobs: list, result, fid: int, base_crossing: list[bool]) -> None:
    """Apply temporal merges before Outlier Extraction, regardless of area labels.

    All temporal candidates render violet; an exclusion is applied when
    the graph's reliability gate permits it. The legacy base_crossing
    argument cannot veto an earlier temporal decision.
    """
    levels = result.levels.get(int(fid))
    if not levels:
        return
    excluded = result.excluded.get(int(fid), ())
    for position, level in levels.items():
        if position >= len(blobs):
            continue
        blob = blobs[position]
        blob.temporal_merge_level = int(level)
        blob.temporal_merge_animals = int(result.animals.get(int(fid), {}).get(position, 0))
        if position in excluded:
            blob.temporal_merge_outlier = True
            blob.is_crossing = True


def _apply_filter_plan_to_frame(blobs: list, fid: int, plan: "_FilterPlan", apply_static) -> bool:
    """Apply verified Search results to one frame; True if a digest differed."""
    from main.frame_scan import frame_blob_digest

    base = [bool(b.is_crossing) for b in blobs]
    digest = None
    mismatch = False
    if plan.static is not None:
        exclusions, digests, start, _args = plan.static
        index = int(fid) - int(start)
        if digests is not None and 0 <= index < len(digests):
            digest = frame_blob_digest(blobs)
            if digest == int(digests[index]):
                apply_static(blobs, fid, exclusions, base)
            else:
                mismatch = True
    if plan.temporal is not None:
        state = plan.temporal
        index = int(fid) - int(state.start)
        if 0 <= index < state.graph.frame_count:
            if digest is None:
                digest = frame_blob_digest(blobs)
            if digest == int(state.graph.digests[index]):
                _apply_temporal_levels(blobs, state.result, fid, base)
            else:
                mismatch = True
    return mismatch


class _FrameContext:
    """ROI-edge pixels and segmentation context of each frame for temporal nodes.

    The context id changes where an ROI or Additional Outlier frame range
    starts or ends; the graph builder creates no edges across such a change.
    """

    def __init__(self, cfg: "_SegConfig", image_shape: tuple):
        self.cfg = cfg
        self.shape = (int(image_shape[0]), int(image_shape[1]))
        self._lock = threading.Lock()
        self._borders: dict = {}

    def _key(self, fid: int) -> tuple:
        rois = tuple(i for i, roi in enumerate(self.cfg.roi_sets) if roi_is_active(roi, fid))
        extra = tuple(
            i for i, item in enumerate(self.cfg.additional_outlier_sets)
            if item.get("enabled", False) and int(item.get("frame_start", 0)) <= fid
            and (int(item.get("frame_end", -1)) < 0 or fid <= int(item.get("frame_end", -1))))
        return rois, extra

    def nodes(self, fid: int, blobs: list, digest: int):
        from main.temporal_merge_blobs import extract_frame_nodes, roi_border_pixels

        rois, extra = self._key(int(fid))
        with self._lock:
            known = rois in self._borders
            border = self._borders.get(rois)
        if not known:
            mask = _build_roi_mask_for_frame(self.cfg.roi_sets, self.shape, int(fid), self.cfg.roi_reverse)
            border = roi_border_pixels(mask)
            with self._lock:
                self._borders[rois] = border
        return extract_frame_nodes(int(fid), blobs, self.shape, border_pixels=border,
                                   digest=digest, context=hash((rois, extra)))


def _result_region_covered_at_most_half(region, foreground: np.ndarray) -> bool:
    x, y, obb_mask = region
    if not obb_mask.size:
        return False
    h, w = obb_mask.shape
    obb_pixels = cv2.countNonZero(obb_mask)
    if not obb_pixels:
        return False
    foreground_pixels = cv2.countNonZero(obb_mask & foreground[y:y+h, x:x+w])
    return foreground_pixels * 2 <= obb_pixels


def _result_blob_pixel_areas(blobs: list[BlobMetrics], obbs, image_shape: tuple) -> list[int]:
    blobs = _restore_result_inputs(blobs)
    if obbs is None:
        areas = []
        for b in blobs:
            if b.manual_outlier:
                continue
            _, _, mask = _blob_mask_region(b.contour, image_shape)
            areas.append(cv2.countNonZero(mask) if mask.size else 0)
        return areas
    foreground = _draw_blob_union_mask(image_shape, blobs)
    areas = []
    for poly in obbs:
        x, y, mask = _result_obb_mask(poly, image_shape)
        h, w = mask.shape
        areas.append(cv2.countNonZero(mask & foreground[y:y+h, x:x+w]) if mask.size else 0)
    return areas


def _solid_region_blobs(template: BlobMetrics, mask: np.ndarray, x: int, y: int) -> list[BlobMetrics]:
    from main.segmentation_core import solid_contours_from_mask

    output = []
    for contour in solid_contours_from_mask(mask):
        contour = contour + np.array([[[x, y]]], dtype=np.int32)
        # Tracking OBB previews require >=3 points, including 1-pixel strips.
        # Repeated vertices preserve the exact filled raster footprint.
        if len(contour) < 3:
            contour = np.repeat(contour, 4 if len(contour) == 1 else 2, axis=0)
        output.append(_blob_with_contour(template, contour))
    return output


def _missing_result_obb_blob(
    poly: np.ndarray, frame_idx: int, blob_index: int, image_shape: tuple,
) -> Optional[BlobMetrics]:
    """Keep the complete OBB contour; raster masks clip it to the video frame."""
    contour = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    H, W = image_shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return None
    visible = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillConvexPoly(visible, contour.reshape(-1, 2) - np.array((x0, y0), dtype=np.int32), 1)
    if not cv2.countNonZero(visible):
        return None
    cx, cy = contour_center(contour)
    return BlobMetrics(
        frame=int(frame_idx), blob_index=blob_index, contour=contour,
        center_x=cx, center_y=cy, area=contour_area(contour),
        bbox_area=float(w * h), w=float(w), h=float(h),
        manual_outlier=True, is_crossing=True, result_obb_count=1,
        result_missing=True,
    )


def _result_obb_overlap_candidates(
    blob_regions: list, obb_region: tuple, available: list[int],
) -> list[tuple[int, int]]:
    """Positive raster overlaps with blobs not yet owned by a centre match."""
    candidates = []
    for index in available:
        overlap = _mask_region_overlap(blob_regions[index], obb_region)
        if overlap > 0:
            candidates.append((index, overlap))
    return candidates


def _blob_color(blob: BlobMetrics) -> tuple:
    """Temporal violet takes precedence; only newly excluded blobs are orange."""
    if blob.temporal_merge_level:
        return TEMPORAL_MERGE_COLOR
    if blob.persistent_static_outlier:
        return STATIC_OUTLIER_COLOR
    return OUTLIER_COLOR if blob.is_crossing else OBB_COLOR


def _draw_blob_overlays(base: np.ndarray, blobs: list, show_overlay: bool,
                        show_contours: bool, show_centers: bool) -> np.ndarray:
    """Shared by preview and Export: temporal violet, later outliers orange."""
    if show_overlay:
        overlay = np.zeros_like(base)
        for b in blobs:
            cv2.drawContours(overlay, [b.contour], -1, _blob_color(b), thickness=cv2.FILLED)
        base = cv2.addWeighted(base, 1.0, overlay, 0.35, 0)

    for b in blobs:
        if show_contours:
            cv2.drawContours(base, [b.contour], -1, _blob_color(b), 2, cv2.LINE_AA)
        if show_centers:
            cv2.circle(base, (int(round(b.center_x)), int(round(b.center_y))), 2, WHITE, -1, cv2.LINE_AA)
    return base


def _draw_result_obbs(image_bgr: np.ndarray, obbs: "Optional[np.ndarray]") -> None:
    if obbs is None:
        return
    for poly in obbs:
        points = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_bgr, [points], True, RESULT_OBB_COLOR, 1, cv2.LINE_AA)


def iqr_stats(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    return q1, q3, iqr


def inside_bounds(v: float, bounds: tuple[float, float]) -> bool:
    lo, hi = bounds
    lower_ok = True if np.isneginf(lo) else (v >= lo)
    upper_ok = True if np.isposinf(hi) else (v <= hi)
    return bool(lower_ok and upper_ok)


def _median_background_from_stack(stack: np.ndarray) -> np.ndarray:
    """Return the exact uint8 median without a second full-size frame stack."""
    count, height, width, channels = stack.shape
    upper = count // 2
    lower = upper - 1
    background = np.empty((height, width, channels), dtype=np.uint8)

    # A frame stack is laid out as (frame, y, x, channel), so the values for
    # one pixel are far apart in memory.  Partitioning a small transposed tile
    # makes that axis contiguous while leaving the source stack available for
    # the max/min reductions that follow.
    row_bytes = max(1, count * width * channels)
    tile_rows = max(1, min(height, BACKGROUND_MEDIAN_TILE_MAX_BYTES // row_bytes))
    for row_start in range(0, height, tile_rows):
        row_end = min(height, row_start + tile_rows)
        tile = np.ascontiguousarray(np.moveaxis(stack[:, row_start:row_end], 0, -1))
        if count % 2:
            tile.partition(upper, axis=-1)
            background[row_start:row_end] = tile[..., upper]
            continue

        tile.partition((lower, upper), axis=-1)
        median_tile = tile[..., lower].astype(np.uint16)
        median_tile += tile[..., upper]
        median_tile //= 2
        background[row_start:row_end] = median_tile
    return background


def _compute_background_method(stack: np.ndarray, method: str) -> np.ndarray:
    """Reduce one uint8 frame stack without creating a float-sized copy."""
    if stack.ndim != 4 or stack.shape[0] <= 0:
        raise RuntimeError("No frames are available for background extraction.")
    if method == "median":
        return _median_background_from_stack(stack)
    if method == "max":
        return np.max(stack, axis=0)
    if method == "min":
        return np.min(stack, axis=0)
    if method == "mean":
        return np.clip(np.mean(stack, axis=0), 0, 255).astype(np.uint8)
    raise RuntimeError(f"Unknown background method: {method}")


class CrossingReviewApp(ctk.CTk):
    def __init__(self, no_launch_tracking: bool = False):
        super().__init__()
        configure_dpi_scaling(self)
        install_window_icon(self)
        self._no_launch_tracking = no_launch_tracking
        self._closing = False
        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.minsize(WINDOW_MIN_W, WINDOW_MIN_H)

        self.video_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a video file.")
        self.preview_title_var = tk.StringVar(value="")
        self.background_ready_var = tk.StringVar(value="background: not computed")
        self.analysis_ready_var = tk.StringVar(value="analysis: not computed")

        self.bg_method_var = tk.StringVar(value="median")
        self.segmentation_mode_var = tk.StringVar(value="background_diff")

        self.threshold_var = tk.IntVar(value=50)
        self.dark_threshold_var = tk.IntVar(value=50)
        self.bright_threshold_var = tk.IntVar(value=200)
        self.diff_threshold_var = tk.IntVar(value=50)
        self.min_area_var = tk.IntVar(value=10)
        self.blur_ksize_var = tk.IntVar(value=5)
        self.open_iter_var = tk.IntVar(value=1)
        self.close_iter_var = tk.IntVar(value=1)
        self.fill_holes_var = tk.BooleanVar(value=True)
        self.invert_mask_var = tk.BooleanVar(value=False)
        self.roi_enabled_var = tk.BooleanVar(value=False)
        self.roi_edit_var = tk.BooleanVar(value=False)
        self.roi_reverse_var = tk.BooleanVar(value=False)
        self.roi_frame_start_var = tk.IntVar(value=0)
        self.roi_frame_end_var = tk.IntVar(value=-1)
        self.roi_sets: list[dict] = [default_roi_set()]
        self.roi_active_set_idx: int = 0
        self._switching_roi_set: bool = False
        self.roi_set_combo: Optional[ctk.CTkComboBox] = None

        self.region_expand_px_var = tk.IntVar(value=0)
        self.region_expand_merge_only_var = tk.BooleanVar(value=False)
        self.single_blob_smoothing_enabled_var = tk.BooleanVar(value=False)
        self.single_blob_smoothing_level_var = tk.IntVar(value=3)

        self.additional_outlier_enabled_var = tk.BooleanVar(value=False)
        self.additional_outlier_frame_start_var = tk.IntVar(value=0)
        self.additional_outlier_frame_end_var = tk.IntVar(value=-1)
        self.additional_outlier_segmentation_mode_var = tk.StringVar(value="background_diff")
        self.additional_outlier_bg_method_var = tk.StringVar(value="median")
        self.additional_outlier_threshold_var = tk.IntVar(value=50)
        self.additional_outlier_dark_threshold_var = tk.IntVar(value=50)
        self.additional_outlier_bright_threshold_var = tk.IntVar(value=200)
        self.additional_outlier_diff_threshold_var = tk.IntVar(value=50)
        self.additional_outlier_min_area_var = tk.IntVar(value=10)
        self.additional_outlier_sets: list[dict] = [self._default_additional_outlier_set()]
        self.additional_outlier_active_set_idx: int = 0
        self._switching_additional_outlier_set: bool = False
        self.additional_outlier_set_combo: Optional[ctk.CTkComboBox] = None
        self.additional_outlier_revision = 0

        self.analysis_sample_count_var = tk.IntVar(value=200)

        # Default 0.5 second window is converted to whole frames on video load.
        self.static_exclusion_enabled_var = tk.BooleanVar(value=False)
        self.static_frame_window_var = tk.IntVar(value=15)
        self.static_presence_var = tk.DoubleVar(value=0.98)
        self.static_iou_var = tk.DoubleVar(value=0.8)
        self.static_max_movement_var = tk.DoubleVar(value=0.2)
        self.static_search_status_var = tk.StringVar(value="Not searched")
        self._static_search_signature: Optional[tuple] = None
        self._static_exclusions: dict[int, set[int]] = {}
        self._static_digests: "Optional[np.ndarray]" = None
        self._static_scan_start = 0

        # High-sensitivity defaults. Search is required before Processing.
        # Confirmation is fixed at 2 frames in the temporal merge algorithm.
        self.temporal_merge_enabled_var = tk.BooleanVar(value=False)
        self.temporal_merge_mode_var = tk.StringVar(value="one_sided")
        self.temporal_merge_tau_manual_var = tk.BooleanVar(value=True)
        self.temporal_merge_tau_var = tk.DoubleVar(value=0.30)
        self.temporal_merge_share_rounding_var = tk.DoubleVar(value=0.25)
        self.temporal_merge_mass_ratio_var = tk.DoubleVar(value=0.60)
        self.temporal_search_status_var = tk.StringVar(value="Disabled")
        self._temporal_search: Optional[_TemporalSearchState] = None
        self._filter_preview_mismatch = False
        self._last_filter_stats: dict = {}

        self.result_import_enabled_var = tk.BooleanVar(value=False)
        self.result_import_path_var = tk.StringVar(value="")
        self.result_import_min_coverage_var = tk.DoubleVar(value=DEFAULT_RESULT_OBB_COVERAGE)
        self.result_import_area_min_var = tk.IntVar(value=0)
        self.result_import_area_max_var = tk.IntVar(value=1000)
        self._result_area_updating = False
        self._result_area_control = SimpleNamespace(
            area_absolute_min_var=self.result_import_area_min_var,
            area_absolute_max_var=self.result_import_area_max_var,
            _absolute_default_values=None, _absolute_defaults_pending=True,
        )
        self._result_area_reference_key = None
        self._result_area_reference_stats = (0.0, 0.0, 0.0, 0.0)
        self.result_import_show_obb_var = tk.BooleanVar(value=True)
        self.result_import_status_var = tk.StringVar(value="result OBB: not loaded")

        self.background_frame_start_var = tk.IntVar(value=0)
        self.background_frame_end_var = tk.IntVar(value=-1)
        self.training_frame_start_var = tk.IntVar(value=0)
        self.training_frame_end_var = tk.IntVar(value=-1)
        self.training_frame_interval_var = tk.IntVar(value=5)

        self.area_outlier_method_var = tk.StringVar(value="iqr")
        self.area_iqr_min_var = tk.DoubleVar(value=1.5)
        self.area_iqr_max_var = tk.DoubleVar(value=1.5)
        self.area_absolute_min_var = tk.DoubleVar(value=0.0)
        self.area_absolute_max_var = tk.DoubleVar(value=100000.0)
        self._absolute_defaults_pending = True
        self._absolute_default_values: Optional[tuple[int, int]] = None
        self.analysis_run_frame = None
        self.area_iqr_frame = None
        self.area_absolute_frame = None
        self.area_iqr_boxplot = None
        self.analysis_gate_frame = None
        self.outlier_section_revealed = False
        self._guidance_stage = "background"
        self._guidance_glows: dict[str, RotatingBorderHighlight] = {}

        self.optional_configurations_var = tk.BooleanVar(value=False)
        self.show_overlay_var = tk.BooleanVar(value=True)
        self.show_centers_var = tk.BooleanVar(value=True)
        self.show_contours_var = tk.BooleanVar(value=True)
        self.frame_var = tk.IntVar(value=0)

        self._collapsible_sections = {}

        self.reader: Optional[VideoFrameReader] = None
        self.frame_count = 0
        self.current_frame = 0
        self.frame_bgr_cache: Optional[np.ndarray] = None
        self.background_bgr: Optional[np.ndarray] = None
        self.active_background_method: Optional[str] = None

        self.current_mask_cache: dict[tuple, tuple[np.ndarray, list[BlobMetrics]]] = {}
        self._area_reference_frame: "np.ndarray | None" = None
        self._area_reference_blobs: list[BlobMetrics] = []
        self._area_reference_key = None
        self._area_reference_max = 0.0
        self._area_reference_stats = (0.0, 0.0, 0.0, 0.0)
        self._min_area_slider_cap: Optional[int] = None
        self.analysis_blobs_by_frame: dict[int, list[BlobMetrics]] = {}
        self.analysis_outlier_frames: list[int] = []
        self.analysis_frames: list[int] = []
        self.analysis_signature: Optional[tuple] = None
        self.analysis_bounds: dict[str, tuple[float, float]] = {}
        self.analysis_iqr_stats: dict[str, tuple[float, float, float]] = {}
        self.analysis_is_stale = True

        self.result_obb_import: Optional[_ResultObbImport] = None
        self.result_import_params_frame = None
        self._result_match: Optional[_ResultMatchConfig] = None

        self.tk_image = None
        self._canvas_image_id: int | None = None
        self.bg_preview_button = None
        self.frame_ranges_button = None
        self.screenshot_button = None
        self.progress_popup = None
        self.progress_bar = None
        self._progress_indeterminate = False
        self._progress_busy = False
        self._progress_poll_job: Optional[str] = None
        self._progress_close_requested = False
        self.progress_label_var = tk.StringVar(value="")
        self._progress_cancel = False
        self._progress_queue: queue.Queue = queue.Queue()
        self.bg_preview_popup = None
        self.bg_preview_image = None
        self.prefetch_after_id: Optional[str] = None
        self._prefetch_cancel = threading.Event()
        self.status_clear_job: Optional[str] = None
        self.playback_job: Optional[str] = None
        self.playback_active = False
        self.playback_speed = 1.0
        self._playback_timing: tuple[float, float, float] | None = None
        self._smooth_player: SmoothPreviewPlayback | None = None
        self._requested_arrow_frame: int | None = None
        self._space_key_down = False
        self._space_release_job: str | None = None
        self._preview_pacer = AdaptivePreviewPacer()
        self._arrow_key_play = ArrowKeyHoldPlayback(
            self,
            current_frame=lambda: (
                self._requested_arrow_frame if self._requested_arrow_frame is not None
                else self.current_frame
            ),
            frame_bounds=lambda: (0, self.frame_count - 1) if self.reader is not None else (0, -1),
            fps=self._playback_fps,
            speed=lambda: self.playback_speed,
            seek=self._queue_arrow_step,
            pause=self.stop_playback,
            pacer=lambda: self._preview_pacer,
            hold_start=lambda direction: self.start_playback(direction),
            hold_stop=self._stop_held_playback,
        )
        self.config_path: Optional[str] = None
        self.session_path: str = ""
        self.background_rebuild_job: Optional[str] = None
        self._applying_config = False
        self._loading_video = False
        self._background_building = False
        self._restore_temporal_after_background = False

        self._current_blobs: list[BlobMetrics] = []
        self._tooltip_window: Optional[tk.Toplevel] = None
        self._tooltip_label: Optional[tk.Label] = None

        self.base_fit_scale = 1.0
        self.zoom_scale = 1.0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.view_initialized = False
        self.current_image_shape: Optional[tuple[int, int]] = None
        self.last_drag_canvas: Optional[np.ndarray] = None
        self._roi_drag_mode: Optional[str] = None
        self._roi_drag_anchor: Optional[tuple[float, float]] = None
        self._roi_drag_vertex: Optional[int] = None
        self._roi_drag_points: list[list[int]] = []

        self._numeric_entry_commits = {}
        self._numeric_click_tag = f"SegmentationNumericCommit{id(self)}"
        for button in (1, 2, 3):
            self.bind_class(self._numeric_click_tag, f"<ButtonPress-{button}>", self._commit_numeric_on_click)
        self.bind_all("<Map>", self._tag_numeric_click_target, add="+")
        self._build_ui()
        self._bind_events()
        bind_save_shortcut(self, self._save_config_shortcut)
        self._bind_traces()
        self._refresh_guidance_ui()
        _start_maximized(self)
        self.after(1200, _start_torch_init)

    def _maximize_window(self) -> None:
        _maximize_window_once(self)

    def _tag_numeric_click_target(self, event):
        """Commit pending numeric input before the clicked widget's action."""
        widget = event.widget
        ancestor = widget
        while ancestor is not None and ancestor is not self:
            ancestor = getattr(ancestor, "master", None)
        if ancestor is self:
            tags = widget.bindtags()
            if self._numeric_click_tag not in tags:
                widget.bindtags((self._numeric_click_tag, *tags))

    def _register_numeric_entry(self, widget, commit):
        self._numeric_entry_commits[widget] = commit
        widget.bind("<Destroy>", lambda _e: self._numeric_entry_commits.pop(widget, None), add="+")

    def _commit_numeric_on_click(self, event):
        focused = self.focus_get()
        commit = self._numeric_entry_commits.get(focused)
        if commit is None or event.widget is focused:
            return
        commit()
        # Labels and canvases do not normally take focus when clicked. Moving
        # focus prevents a later edit from remaining attached to the old field.
        event.widget.focus_set()

    def _register_popup_integer_entry(self, widget):
        """Normalize a dialog value while retaining its Apply/Export action."""
        state = {"value": int(widget.get())}

        def commit(_event=None):
            if not widget.winfo_exists():
                return "break"
            try:
                value = int(widget.get().strip())
                value = max(int(widget.cget("from")), min(int(widget.cget("to")), value))
            except ValueError:
                value = state["value"]
            state["value"] = value
            widget.delete(0, tk.END)
            widget.insert(0, str(value))
            return "break"

        widget.configure(command=commit)
        for event in ("<Return>", "<KP_Enter>", "<FocusOut>"):
            widget.bind(event, commit, add="+")
        self._register_numeric_entry(widget, commit)

    def _build_ui(self):
        top = ctk.CTkFrame(self, corner_radius=0)
        top.pack(fill="x", padx=8, pady=8)

        ctk.CTkLabel(top, text="Video").grid(row=0, column=0, sticky="w")
        ctk.CTkEntry(top, textvariable=self.video_path_var).grid(row=0, column=1, sticky="ew", padx=8)
        self.video_reference_button = ctk.CTkButton(
            top, text="Reference", command=self.browse_video,
            border_width=3, border_color="#777777",
        )
        self.video_reference_button.grid(row=0, column=2)
        top.grid_columnconfigure(1, weight=1)

        main = ctk.CTkFrame(self, corner_radius=0)
        main.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        self.sidebar = ctk.CTkFrame(main, width=390, border_width=1, corner_radius=4)
        self.sidebar.pack(side="left", fill="y", padx=(0, 8))
        self.sidebar.pack_propagate(False)

        right = ctk.CTkFrame(main, corner_radius=0)
        right.pack(side="left", fill="both", expand=True)

        self._build_sidebar(self.sidebar)

        nav = ctk.CTkFrame(right, corner_radius=0)
        nav.pack(fill="x", pady=(0, 6))
        ctk.CTkButton(nav, text="|<", width=40, command=lambda: self.set_frame(0)).pack(side="left")
        ctk.CTkButton(nav, text="<", width=40, command=lambda: self.step_frame(-1)).pack(side="left", padx=4)
        self.play_button = ctk.CTkButton(nav, text="Play", width=70, command=self.toggle_playback)
        self.play_button.pack(side="left")
        self.playback_speed_button = ctk.CTkButton(
            nav, text="1×", width=48, command=self.cycle_playback_speed,
        )
        self.playback_speed_button.pack(side="left", padx=(4, 0))
        ctk.CTkButton(nav, text=">", width=40, command=lambda: self.step_frame(1)).pack(side="left", padx=4)
        ctk.CTkButton(nav, text=">|", width=40, command=lambda: self.set_frame(self.frame_count - 1)).pack(side="left", padx=(0, 12))
        ctk.CTkLabel(nav, text="frame").pack(side="left")
        self.frame_spin = dark_spinbox(nav, width=10, textvariable=self.frame_var, command=self.on_frame_spin, **_SPIN_CFG)
        self.frame_spin.pack(side="left", padx=(6, 12))
        self.frame_spin.bind("<Return>", self.on_frame_entry_commit, add="+")
        self.frame_spin.bind("<KP_Enter>", self.on_frame_entry_commit, add="+")
        self.frame_spin.bind("<FocusOut>", self.on_frame_entry_commit, add="+")
        self._register_numeric_entry(self.frame_spin, self.on_frame_entry_commit)
        self.frame_scale = ctk.CTkSlider(nav, orientation="horizontal", command=self.on_frame_scale)
        self.frame_scale.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(nav, text="Fit window", width=90, command=self.fit_canvas_to_window).pack(side="left", padx=(8, 0))

        self.canvas = tk.Canvas(right, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._initialize_video_drag_and_drop()


    def _build_sidebar(self, parent: ctk.CTkFrame):
        def section(title: str, *, expanded: bool = False) -> ctk.CTkFrame:
            return self._make_collapsible_section(parent, title, expanded=expanded)

        self.optional_configurations_checkbox = ctk.CTkCheckBox(
            parent,
            text="Show optional settings",
            variable=self.optional_configurations_var,
            command=self._on_optional_configurations_changed,
        )
        self.optional_configurations_checkbox.pack(anchor="w", padx=8, pady=(8, 4))

        self.seg_section = section("Background / Segmentation", expanded=True)
        self.roi_section = section("ROI (Optional)")
        self.region_expansion_section = section("Region Expansion (Optional, Beta)")
        # Temporal merges are detected before statistical Outlier Extraction.
        self.temporal_blob_section = section("Exclude temporal merges (Optional, Beta)")
        self.ana_section = section("Outlier Extraction", expanded=True)
        self.analysis_gate_frame = ctk.CTkFrame(parent, corner_radius=0)
        ctk.CTkLabel(
            self.analysis_gate_frame,
            text="│\n│\n│\n│\n▼",
            anchor="center",
            justify="center",
        ).pack(fill="x", pady=(0, 4))
        self.analysis_start_button = ctk.CTkButton(
            self.analysis_gate_frame, text="Analyze", command=self.analyze_sample_frames,
            border_width=3, border_color="#777777",
        )
        self.analysis_start_button.pack(fill="x", padx=6, pady=(0, 6))
        self.result_section = section("Result OBB Import (Optional, Beta)")
        self.additional_outlier_section = section("Additional Outlier (Optional, Beta)")
        self.single_blob_smoothing_section = section("Single-animal Smoothing (Optional, Beta)")
        self.static_blob_section = section("Exclude persistent static blobs (Optional, Beta)")

        seg = self.seg_section
        self._pack_combobox(seg, "Segmentation mode", self.segmentation_mode_var, ["background_diff", "dark_region", "bright_region", "dark_region_and_background_diff", "bright_region_and_background_diff"])

        self.bg_controls_frame = ctk.CTkFrame(seg, corner_radius=0)
        self.bg_controls_frame.pack(fill="x")
        self._pack_combobox(self.bg_controls_frame, "Background method", self.bg_method_var, list(BACKGROUND_METHODS))
        self.bg_preview_button = ctk.CTkButton(
            self.bg_controls_frame, text="Show background",
            command=self.show_background_popup, state="disabled",
            border_width=0, border_color="#777777",
        )
        self.bg_preview_button.pack(fill="x", padx=6, pady=(2, 6))
        self.frame_ranges_button = ctk.CTkButton(seg, text="Change Frame Ranges...", command=self.change_frame_ranges, state="disabled")
        self.frame_ranges_button.pack(fill="x", padx=6, pady=(2, 6))

        self.threshold_controls_frame = ctk.CTkFrame(seg, corner_radius=0)
        self.threshold_controls_frame.pack(fill="x")
        self.threshold_scale = self._pack_scale_entry(self.threshold_controls_frame, "Threshold", self.threshold_var, 0, 255, 1)

        self.dark_hybrid_threshold_controls_frame = ctk.CTkFrame(seg, corner_radius=0)
        self.dark_hybrid_threshold_controls_frame.pack(fill="x")
        self.dark_threshold_scale = self._pack_scale_entry(self.dark_hybrid_threshold_controls_frame, "Dark threshold", self.dark_threshold_var, 0, 255, 1)
        self.dark_diff_threshold_scale = self._pack_scale_entry(self.dark_hybrid_threshold_controls_frame, "Diff threshold", self.diff_threshold_var, 0, 255, 1)

        self.bright_hybrid_threshold_controls_frame = ctk.CTkFrame(seg, corner_radius=0)
        self.bright_hybrid_threshold_controls_frame.pack(fill="x")
        self.bright_threshold_scale = self._pack_scale_entry(self.bright_hybrid_threshold_controls_frame, "Bright threshold", self.bright_threshold_var, 0, 255, 1)
        self.bright_diff_threshold_scale = self._pack_scale_entry(self.bright_hybrid_threshold_controls_frame, "Diff threshold", self.diff_threshold_var, 0, 255, 1)

        self.min_area_scale = self._pack_scale_entry(seg, "Min blob area", self.min_area_var, 1, 100000, 1)
        self.threshold_done_button = ctk.CTkButton(
            seg, text="Done", command=self._confirm_threshold,
            border_width=3, border_color="#777777",
        )
        self.threshold_done_button.pack(fill="x", padx=6, pady=(6, 8))

        roi = self.roi_section
        _roi_set_row = ctk.CTkFrame(roi, corner_radius=0)
        _roi_set_row.pack(fill="x", padx=6, pady=(4, 2))
        ctk.CTkLabel(_roi_set_row, text="ROI Set", width=60, anchor="w").pack(side="left")
        self.roi_set_combo = ctk.CTkComboBox(_roi_set_row, values=["Set 1"], state="readonly", width=90, command=self._on_roi_set_selected)
        self.roi_set_combo.set("Set 1")
        self.roi_set_combo.pack(side="left", padx=(4, 2))
        ctk.CTkButton(_roi_set_row, text="+", width=28, command=self._add_roi_set).pack(side="left", padx=2)
        ctk.CTkButton(_roi_set_row, text="-", width=28, command=self._remove_roi_set).pack(side="left", padx=2)
        roi_use_row = ctk.CTkFrame(roi, corner_radius=0)
        roi_use_row.pack(fill="x", padx=6, pady=(4, 4))
        ctk.CTkCheckBox(roi_use_row, text="Use this ROI", variable=self.roi_enabled_var, command=self._toggle_roi_params).pack(side="left")
        self.roi_edit_button = ctk.CTkButton(roi_use_row, text="Edit ROI", width=110, command=self._toggle_roi_edit)
        self.roi_edit_button.pack(side="left", padx=(10, 0))
        self.roi_params_frame = ctk.CTkFrame(roi, corner_radius=0)
        self.roi_params_frame.pack(fill="x", padx=0, pady=(0, 4))
        ctk.CTkLabel(self.roi_params_frame, text="Edit: drag a vertex to reshape, inside to move.\nDrag outside the polygon to pan.",
                        justify="left", anchor="w", wraplength=340).pack(fill="x", padx=6, pady=(0, 4))
        self.roi_frame_start_scale = self._pack_scale_entry(self.roi_params_frame, "From frame", self.roi_frame_start_var, 0, 999999, 1)
        self.roi_frame_end_scale = self._pack_scale_entry(self.roi_params_frame, "To frame (-1=end)", self.roi_frame_end_var, -1, 999999, 1, slider_lo=0, end_sentinel=-1)
        ctk.CTkCheckBox(self.roi_params_frame, text="Reverse ROI", variable=self.roi_reverse_var).pack(anchor="w", padx=6, pady=(4, 6))
        self._toggle_roi_params()

        additional = self.additional_outlier_section
        additional_set_row = ctk.CTkFrame(additional, corner_radius=0)
        additional_set_row.pack(fill="x", padx=6, pady=(4, 2))
        ctk.CTkLabel(additional_set_row, text="Additional Outlier Set", width=162, anchor="w").pack(side="left")
        self.additional_outlier_set_combo = ctk.CTkComboBox(
            additional_set_row,
            values=["Set 1"],
            state="readonly",
            width=90,
            command=self._on_additional_outlier_set_selected,
        )
        self.additional_outlier_set_combo.set("Set 1")
        self.additional_outlier_set_combo.pack(side="left", padx=(4, 2))
        ctk.CTkButton(additional_set_row, text="+", width=28, command=self._add_additional_outlier_set).pack(side="left", padx=2)
        ctk.CTkButton(additional_set_row, text="-", width=28, command=self._remove_additional_outlier_set).pack(side="left", padx=2)
        ctk.CTkCheckBox(
            additional,
            text="Enabled",
            variable=self.additional_outlier_enabled_var,
            command=self._toggle_additional_outlier_params,
        ).pack(anchor="w", padx=6, pady=(2, 4))
        self.additional_outlier_params_frame = ctk.CTkFrame(additional, corner_radius=0)
        self.additional_outlier_frame_start_scale = self._pack_scale_entry(
            self.additional_outlier_params_frame, "From frame", self.additional_outlier_frame_start_var, 0, 999999, 1
        )
        self.additional_outlier_frame_end_scale = self._pack_scale_entry(
            self.additional_outlier_params_frame, "To frame (-1=end)", self.additional_outlier_frame_end_var, -1, 999999, 1,
            slider_lo=0, end_sentinel=-1,
        )
        self._pack_combobox(
            self.additional_outlier_params_frame,
            "Segmentation mode",
            self.additional_outlier_segmentation_mode_var,
            ["background_diff", "dark_region", "bright_region",
             "dark_region_and_background_diff", "bright_region_and_background_diff"],
        )
        self.additional_outlier_bg_controls_frame = ctk.CTkFrame(self.additional_outlier_params_frame, corner_radius=0)
        self._pack_combobox(
            self.additional_outlier_bg_controls_frame,
            "Background method",
            self.additional_outlier_bg_method_var,
            list(BACKGROUND_METHODS),
        )
        self.additional_outlier_threshold_controls_frame = ctk.CTkFrame(self.additional_outlier_params_frame, corner_radius=0)
        self._pack_scale_entry(
            self.additional_outlier_threshold_controls_frame,
            "Threshold", self.additional_outlier_threshold_var, 0, 255, 1,
        )
        self.additional_outlier_dark_hybrid_threshold_controls_frame = ctk.CTkFrame(
            self.additional_outlier_params_frame, corner_radius=0,
        )
        self._pack_scale_entry(
            self.additional_outlier_dark_hybrid_threshold_controls_frame,
            "Dark threshold", self.additional_outlier_dark_threshold_var, 0, 255, 1,
        )
        self._pack_scale_entry(
            self.additional_outlier_dark_hybrid_threshold_controls_frame,
            "Diff threshold", self.additional_outlier_diff_threshold_var, 0, 255, 1,
        )
        self.additional_outlier_bright_hybrid_threshold_controls_frame = ctk.CTkFrame(
            self.additional_outlier_params_frame, corner_radius=0,
        )
        self._pack_scale_entry(
            self.additional_outlier_bright_hybrid_threshold_controls_frame,
            "Bright threshold", self.additional_outlier_bright_threshold_var, 0, 255, 1,
        )
        self._pack_scale_entry(
            self.additional_outlier_bright_hybrid_threshold_controls_frame,
            "Diff threshold", self.additional_outlier_diff_threshold_var, 0, 255, 1,
        )
        self._pack_scale_entry(
            self.additional_outlier_params_frame,
            "Min blob area", self.additional_outlier_min_area_var, 1, 100000, 1,
        )
        self._refresh_additional_outlier_controls()
        self._toggle_additional_outlier_params()

        smoothing = self.single_blob_smoothing_section
        ctk.CTkCheckBox(
            smoothing, text="Smooth single-animal blobs",
            variable=self.single_blob_smoothing_enabled_var,
        ).pack(anchor="w", padx=6, pady=(4, 4))
        self.single_blob_smoothing_params_frame = ctk.CTkFrame(smoothing, corner_radius=0)
        self._pack_scale_entry(
            self.single_blob_smoothing_params_frame, "Smoothing level",
            self.single_blob_smoothing_level_var, 1, 20, 1,
        )
        ctk.CTkLabel(
            self.single_blob_smoothing_params_frame,
            text="Blue blobs only, after Outlier Extraction.\n"
                 "Higher levels remove thicker protrusions.\n"
                 "Area bounds are reapplied afterward.\n"
                 "Removed pixels become background.\n"
                 "Analyze first when using IQR.",
            anchor="w", justify="left", wraplength=280,
        ).pack(fill="x", padx=6, pady=(2, 6))
        self._toggle_single_blob_smoothing_params()

        static_section = self.static_blob_section
        ctk.CTkCheckBox(
            static_section, text="Enabled",
            variable=self.static_exclusion_enabled_var,
            command=self._on_static_search_changed,
        ).pack(anchor="w", padx=6, pady=(4, 4))
        self.static_options_frame = ctk.CTkFrame(static_section, corner_radius=0)
        self._pack_scale_entry(
            self.static_options_frame, "Frame window",
            self.static_frame_window_var, 2, 2000, 1,
        )
        self._pack_scale_entry(
            self.static_options_frame, "Presence ratio",
            self.static_presence_var, 0.01, 1.0, 0.01, is_float=True,
        )
        self._pack_scale_entry(
            self.static_options_frame, "Median contour IoU",
            self.static_iou_var, 0.0, 1.0, 0.01, is_float=True,
        )
        self._pack_scale_entry(
            self.static_options_frame, "Max center drift / body length",
            self.static_max_movement_var, 0.0, 1.0, 0.01, is_float=True,
        )
        self.static_search_button = ctk.CTkButton(
            self.static_options_frame, text="Search",
            command=self.search_persistent_static_blobs,
        )
        self.static_search_button.pack(fill="x", padx=6, pady=(4, 4))
        self._static_search_glow = RotatingBorderHighlight(self, self.static_search_button)
        self._static_search_glow.set_active(bool(self.static_exclusion_enabled_var.get()))
        ctk.CTkLabel(
            self.static_options_frame, textvariable=self.static_search_status_var,
            anchor="w", wraplength=330, justify="left",
        ).pack(fill="x", padx=6, pady=(0, 6))
        self._toggle_static_options()
        for var in (
            self.static_frame_window_var, self.static_presence_var,
            self.static_iou_var, self.static_max_movement_var,
        ):
            var.trace_add("write", lambda *_: self._on_static_parameters_changed())

        temporal_section = self.temporal_blob_section
        ctk.CTkCheckBox(
            temporal_section, text="Enabled",
            variable=self.temporal_merge_enabled_var,
            command=self._on_temporal_enabled_changed,
        ).pack(anchor="w", padx=6, pady=(4, 4))
        self.temporal_options_frame = ctk.CTkFrame(temporal_section, corner_radius=0)
        mode_row = ctk.CTkFrame(self.temporal_options_frame, corner_radius=0)
        mode_row.pack(fill="x", padx=6, pady=(2, 4))
        ctk.CTkLabel(mode_row, text="Exclude", width=60, anchor="w").pack(side="left")
        ctk.CTkRadioButton(
            mode_row, text="Bilateral only", variable=self.temporal_merge_mode_var, value="bilateral",
        ).pack(side="left", padx=(0, 10))
        ctk.CTkRadioButton(
            mode_row, text="Include one-sided", variable=self.temporal_merge_mode_var, value="one_sided",
        ).pack(side="left")
        # Search parameters; all thresholds can be entered directly.
        self._pack_scale_entry(
            self.temporal_options_frame, "Manual containment",
            self.temporal_merge_tau_var, 0.1, 0.9, 0.01, is_float=True,
            enable_var=self.temporal_merge_tau_manual_var, show_slider=False,
        )
        self._pack_scale_entry(
            self.temporal_options_frame, "Share rounding",
            self.temporal_merge_share_rounding_var, 0.0, 1.0, 0.01,
            is_float=True, show_slider=False,
        )
        self._pack_scale_entry(
            self.temporal_options_frame, "One-sided mass ratio",
            self.temporal_merge_mass_ratio_var, 0.01, 1.0, 0.01,
            is_float=True, show_slider=False,
        )
        search_row = ctk.CTkFrame(self.temporal_options_frame, corner_radius=0)
        search_row.pack(fill="x", padx=6, pady=(4, 2))
        self.temporal_search_button = ctk.CTkButton(
            search_row, text="Search", command=self.search_temporal_merges,
        )
        self.temporal_search_button.pack(side="left", expand=True, fill="x")
        # Jump to the previous/next High or Medium interval to inspect it.
        ctk.CTkButton(
            search_row, text="<", width=36, command=lambda: self._goto_temporal_event(-1),
        ).pack(side="left", padx=(6, 2))
        ctk.CTkButton(
            search_row, text=">", width=36, command=lambda: self._goto_temporal_event(1),
        ).pack(side="left")
        self._temporal_search_glow = RotatingBorderHighlight(self, self.temporal_search_button)
        self._temporal_search_glow.set_active(bool(self.temporal_merge_enabled_var.get()))
        ctk.CTkLabel(
            self.temporal_options_frame, textvariable=self.temporal_search_status_var,
            anchor="w", wraplength=360, justify="left",
        ).pack(fill="x", padx=6, pady=(0, 4))
        self._toggle_temporal_options()
        for var in (
            self.temporal_merge_mode_var, self.temporal_merge_tau_manual_var,
            self.temporal_merge_tau_var, self.temporal_merge_share_rounding_var,
            self.temporal_merge_mass_ratio_var,
        ):
            var.trace_add("write", lambda *_: self._on_temporal_parameters_changed())

        expansion = self.region_expansion_section
        self._pack_scale_entry(expansion, "Expand region (px)", self.region_expand_px_var,
                               0, REGION_EXPAND_ENTRY_MAX_PX, 1, slider_hi=REGION_EXPAND_SLIDER_MAX_PX)
        ctk.CTkCheckBox(
            expansion,
            text="Keep merging expansion only",
            variable=self.region_expand_merge_only_var,
        ).pack(anchor="w", padx=6, pady=(2, 6))

        ana = self.ana_section
        method_frame = ctk.CTkFrame(ana, corner_radius=0)
        method_frame.pack(fill="x", padx=6, pady=(2, 4))
        ctk.CTkLabel(method_frame, text="Area outlier method", width=144, anchor="w").pack(side="left")
        ctk.CTkRadioButton(
            method_frame, text="IQR", variable=self.area_outlier_method_var,
            value="iqr", command=self._on_area_outlier_method_changed,
        ).pack(side="left", padx=(0, 12))
        ctk.CTkRadioButton(
            method_frame, text="Absolute", variable=self.area_outlier_method_var,
            value="absolute", command=self._on_area_outlier_method_changed,
        ).pack(side="left")

        # Sampled-frame analysis exists only to fit the IQR bounds. Absolute
        # bounds are entered directly, so this block belongs to the IQR branch.
        self.analysis_run_frame = ctk.CTkFrame(ana, corner_radius=0)
        self._pack_scale_entry(self.analysis_run_frame, "Analysis samples", self.analysis_sample_count_var, 10, 5000, 1)
        analysis_nav_row = ctk.CTkFrame(self.analysis_run_frame, corner_radius=0)
        analysis_nav_row.pack(fill="x", padx=6, pady=(2, 6))
        ctk.CTkButton(
            analysis_nav_row, text="Analyze", command=self.analyze_sample_frames,
        ).pack(side="left", fill="x", expand=True)
        self.analysis_prev_outlier_button = ctk.CTkButton(
            analysis_nav_row, text="<", width=36, state="disabled",
            command=lambda: self._goto_sampled_outlier(-1),
        )
        self.analysis_prev_outlier_button.pack(side="left", padx=(6, 2))
        self.analysis_next_outlier_button = ctk.CTkButton(
            analysis_nav_row, text=">", width=36, state="disabled",
            command=lambda: self._goto_sampled_outlier(1),
        )
        self.analysis_next_outlier_button.pack(side="left")

        self.area_iqr_frame = ctk.CTkFrame(ana, corner_radius=0)
        self.area_absolute_frame = ctk.CTkFrame(ana, corner_radius=0)
        self._build_area_iqr_control(self.area_iqr_frame)
        self._build_area_absolute_control(self.area_absolute_frame)
        self._toggle_area_outlier_method_controls()
        self.outlier_done_button = ctk.CTkButton(
            ana, text="Done", command=self._confirm_outliers,
            border_width=3, border_color="#777777",
        )
        self.outlier_done_button.pack(side="bottom", fill="x", padx=6, pady=(6, 8))

        result = self.result_section
        ctk.CTkCheckBox(
            result,
            text="Use imported result OBB",
            variable=self.result_import_enabled_var,
            command=self._toggle_result_import_params,
        ).pack(anchor="w", padx=6, pady=(2, 4))
        self.result_import_params_frame = ctk.CTkFrame(result, corner_radius=0)
        _result_path_row = ctk.CTkFrame(self.result_import_params_frame, corner_radius=0)
        _result_path_row.pack(fill="x", padx=6, pady=(2, 2))
        ctk.CTkEntry(_result_path_row, textvariable=self.result_import_path_var).pack(
            side="left", fill="x", expand=True)
        ctk.CTkButton(
            _result_path_row, text="Reference", width=84, command=self.browse_result_obb_csv,
        ).pack(side="left", padx=(4, 0))
        ctk.CTkLabel(
            self.result_import_params_frame, textvariable=self.result_import_status_var, anchor="w",
        ).pack(fill="x", padx=6, pady=(2, 2))
        self._pack_scale_entry(
            self.result_import_params_frame, "Min OBB coverage",
            self.result_import_min_coverage_var, 0.0, 1.0, 0.1, is_float=True,
        )
        self._build_area_absolute_control(self.result_import_params_frame, result=True)
        ctk.CTkCheckBox(
            self.result_import_params_frame, text="Show imported OBB",
            variable=self.result_import_show_obb_var,
        ).pack(anchor="w", padx=6, pady=(2, 2))
        ctk.CTkButton(
            self.result_import_params_frame, text="Clear imported result",
            command=self.clear_result_obb_import,
        ).pack(fill="x", padx=6, pady=(2, 6))
        self._toggle_result_import_params()

        # Config changes auto-save; Save As remains available for making a copy.
        bottom = ctk.CTkFrame(self.sidebar, corner_radius=0)
        bottom.pack(side="bottom", fill="x", padx=6, pady=6)
        ctk.CTkButton(bottom, text="Save config", command=self.save_config).pack(side="left", expand=True, fill="x", padx=(0,4))
        ctk.CTkButton(bottom, text="Load config", command=self.load_config).pack(side="left", expand=True, fill="x", padx=(4,0))

        view = ctk.CTkFrame(self.sidebar, border_width=1, corner_radius=4)
        view.pack(side="bottom", fill="x", padx=8, pady=(0, 6))
        toggles = ctk.CTkFrame(view, corner_radius=0)
        toggles.pack(fill="x", padx=6, pady=(2, 6))
        ctk.CTkCheckBox(toggles, text="Show overlay", variable=self.show_overlay_var).pack(side="left")
        ctk.CTkCheckBox(toggles, text="Show white dots", variable=self.show_centers_var).pack(side="left", padx=(10, 0))
        ctk.CTkCheckBox(toggles, text="Show contours", variable=self.show_contours_var).pack(side="left", padx=(10, 0))
        export_row = ctk.CTkFrame(view, corner_radius=0)
        export_row.pack(fill="x", padx=6, pady=(2, 6))
        self.export_button = ctk.CTkButton(export_row, text="Export labeled video", command=self.export_labeled_video, state="disabled")
        self.export_button.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.screenshot_button = ctk.CTkButton(export_row, text="Screenshot", command=self.save_canvas_screenshot, state="disabled")
        self.screenshot_button.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # Independent of the overlay-toggle/export box below, so it stands
        # out instead of blending in with the checkboxes.
        self.processing_button = ctk.CTkButton(
            self.sidebar, text="Processing", command=self.run_processing, height=40,
            border_width=3, border_color="#777777",
        )
        self.processing_button.pack(side="bottom", fill="x", padx=8, pady=(0, 6))


        self._refresh_segmentation_controls()
        self._refresh_segmentation_controls()
        self._guidance_glows = {
            "video": RotatingBorderHighlight(self, self.video_reference_button),
            "background": RotatingBorderHighlight(self, self.bg_preview_button),
            "threshold": RotatingBorderHighlight(self, self.threshold_done_button),
            "analysis": RotatingBorderHighlight(self, self.analysis_start_button),
            "outlier": RotatingBorderHighlight(self, self.outlier_done_button),
            "processing": RotatingBorderHighlight(self, self.processing_button),
        }
        self._set_workflow_stage("background")

    def _make_collapsible_section(self, parent: ctk.CTkFrame, title: str, expanded: bool = True) -> ctk.CTkFrame:
        outer = ctk.CTkFrame(parent, corner_radius=0)
        header = ctk.CTkButton(
            outer,
            text=("v " if expanded else "> ") + title,
            anchor="w",
            fg_color="transparent",
            hover_color=("#d0d0d0", "#3a3a3a"),
            text_color=("black", "white"),
            corner_radius=4,
        )
        header.pack(fill="x")

        body = ctk.CTkFrame(outer, border_width=1, corner_radius=4)
        if expanded:
            body.pack(fill="x")

        self._collapsible_sections[body] = {
            "outer": outer,
            "header": header,
            "title": title,
            "expanded": bool(expanded),
        }
        header.configure(command=lambda b=body: self._toggle_collapsible_section(b))
        return body

    def _toggle_collapsible_section(self, body: ctk.CTkFrame):
        state = self._collapsible_sections.get(body)
        if state is None:
            return
        self._set_section_expanded(body, not state["expanded"])

    def _set_section_expanded(self, body: ctk.CTkFrame, expanded: bool):
        state = self._collapsible_sections.get(body)
        if state is None:
            return
        state["expanded"] = bool(expanded)
        title = state["title"]
        header = state["header"]
        if expanded:
            if not body.winfo_ismapped():
                body.pack(fill="x")
            header.configure(text="v " + title)
        else:
            body.pack_forget()
            header.configure(text="> " + title)

    def _pack_sidebar_section(self, body: ctk.CTkFrame):
        state = self._collapsible_sections.get(body)
        if state is None:
            body.pack(fill="x", padx=8, pady=6)
            return
        state["outer"].pack(fill="x", padx=8, pady=6)

    def _on_optional_configurations_changed(self):
        self._set_workflow_stage("optional_configurations")

    def _confirm_threshold(self):
        if self._guidance_stage == "threshold":
            self._guidance_stage = "analysis"
            self._set_workflow_stage("analysis")

    def _confirm_outliers(self):
        if self._guidance_stage == "outlier":
            self._guidance_stage = "processing"
            self._refresh_guidance_ui()

    def _refresh_guidance_ui(self):
        if not self._guidance_glows:
            return
        has_video_path = bool(self.video_path_var.get().strip())
        # RotatingBorderHighlight alone controls button border visibility.
        stage = self._guidance_stage if has_video_path else "video"
        if stage == "background" and not self._segmentation_needs_background():
            stage = self._guidance_stage = "threshold"
        if stage in ("outlier", "processing") and not self.outlier_section_revealed:
            stage = self._guidance_stage = "analysis"

        if stage == "threshold":
            if not self.threshold_done_button.winfo_manager():
                self.threshold_done_button.pack(fill="x", padx=6, pady=(6, 8))
        else:
            self.threshold_done_button.pack_forget()

        if stage == "outlier" and self.outlier_section_revealed:
            if not self.outlier_done_button.winfo_manager():
                self.outlier_done_button.pack(side="bottom", fill="x", padx=6, pady=(6, 8))
        else:
            self.outlier_done_button.pack_forget()

        # Processing remains hidden until the Outlier Extraction Done is pressed.
        if stage == "processing" and self.outlier_section_revealed:
            if not self.processing_button.winfo_manager():
                self.processing_button.pack(side="bottom", fill="x", padx=8, pady=(0, 6))
        else:
            self.processing_button.pack_forget()
        for name, glow in self._guidance_glows.items():
            glow.set_active(name == stage)

    def _set_workflow_stage(self, stage: str):
        """Show optional sections on request while preserving their original order."""
        show_optional = bool(self.optional_configurations_var.get())
        sections = [
            (getattr(self, "seg_section", None), False),
            (getattr(self, "roi_section", None), True),
            (getattr(self, "region_expansion_section", None), True),
            (getattr(self, "temporal_blob_section", None), True),
            (getattr(self, "ana_section", None), False),
            (getattr(self, "result_section", None), True),
            (getattr(self, "additional_outlier_section", None), True),
            (getattr(self, "single_blob_smoothing_section", None), True),
            (getattr(self, "static_blob_section", None), True),
        ]

        # Reset the pack order before restoring visible sections. This keeps
        # optional sections in their original positions after toggling them on.
        if self.analysis_gate_frame is not None:
            self.analysis_gate_frame.pack_forget()
        for section, _optional in sections:
            if section is None:
                continue
            state = self._collapsible_sections.get(section)
            if state is not None:
                state["outer"].pack_forget()

        for section, optional in sections:
            if section is None:
                continue
            # An enabled saved filter must expose its required Search button,
            # even when optional settings are otherwise collapsed.
            if section is getattr(self, "static_blob_section", None) and self.static_exclusion_enabled_var.get():
                optional = False
            if section is getattr(self, "temporal_blob_section", None) and self.temporal_merge_enabled_var.get():
                optional = False
            if optional and not show_optional:
                continue
            if section is self.ana_section and not self.outlier_section_revealed:
                if self._guidance_stage == "analysis" and self.analysis_gate_frame is not None:
                    self.analysis_gate_frame.pack(fill="x", padx=8, pady=6)
                continue
            self._pack_sidebar_section(section)
        self._update_analysis_dependent_buttons()
        self._refresh_guidance_ui()

    def _update_analysis_dependent_buttons(self):
        # Processing is revealed by the Outlier Extraction Done button.
        ready = self.outlier_section_revealed and not self.analysis_is_stale and (
            bool(self.analysis_iqr_stats) or not self._needs_area_analysis()
        )
        state = "normal" if ready else "disabled"
        if hasattr(self, "export_button") and self.export_button is not None:
            self.export_button.configure(state=state)
        outlier_nav_state = (
            "normal" if self.reader is not None and not self.analysis_is_stale
            and self.analysis_outlier_frames else "disabled"
        )
        for name in ("analysis_prev_outlier_button", "analysis_next_outlier_button"):
            button = getattr(self, name, None)
            if button is not None:
                button.configure(state=outlier_nav_state)
        if hasattr(self, "screenshot_button") and self.screenshot_button is not None:
            screenshot_state = "normal" if self.reader is not None and self.frame_bgr_cache is not None else "disabled"
            self.screenshot_button.configure(state=screenshot_state)

    def _config_vars(self) -> dict:
        return {
            "video_path": self.video_path_var,
            "bg_method": self.bg_method_var,
            "segmentation_mode": self.segmentation_mode_var,
            "threshold": self.threshold_var,
            "dark_threshold": self.dark_threshold_var,
            "bright_threshold": self.bright_threshold_var,
            "diff_threshold": self.diff_threshold_var,
            "min_area": self.min_area_var,
            "blur_ksize": self.blur_ksize_var,
            "open_iter": self.open_iter_var,
            "close_iter": self.close_iter_var,
            "fill_holes": self.fill_holes_var,
            "invert_mask": self.invert_mask_var,
            "background_frame_start": self.background_frame_start_var,
            "background_frame_end": self.background_frame_end_var,
            "training_frame_start": self.training_frame_start_var,
            "training_frame_end": self.training_frame_end_var,
            "training_frame_interval": self.training_frame_interval_var,
            "roi_reverse": self.roi_reverse_var,
            "additional_outlier_enabled": self.additional_outlier_enabled_var,
            "additional_outlier_frame_start": self.additional_outlier_frame_start_var,
            "additional_outlier_frame_end": self.additional_outlier_frame_end_var,
            "additional_outlier_segmentation_mode": self.additional_outlier_segmentation_mode_var,
            "additional_outlier_bg_method": self.additional_outlier_bg_method_var,
            "additional_outlier_threshold": self.additional_outlier_threshold_var,
            "additional_outlier_dark_threshold": self.additional_outlier_dark_threshold_var,
            "additional_outlier_bright_threshold": self.additional_outlier_bright_threshold_var,
            "additional_outlier_diff_threshold": self.additional_outlier_diff_threshold_var,
            "additional_outlier_min_area": self.additional_outlier_min_area_var,
            "region_expand_px": self.region_expand_px_var,
            "region_expand_merge_only": self.region_expand_merge_only_var,
            "single_blob_smoothing_enabled": self.single_blob_smoothing_enabled_var,
            "single_blob_smoothing_level": self.single_blob_smoothing_level_var,
            "analysis_sample_count": self.analysis_sample_count_var,
            "static_exclusion_enabled": self.static_exclusion_enabled_var,
            "static_frame_window": self.static_frame_window_var,
            "static_presence": self.static_presence_var,
            "static_min_iou": self.static_iou_var,
            "static_max_movement": self.static_max_movement_var,
            "temporal_merge_enabled": self.temporal_merge_enabled_var,
            "temporal_merge_mode": self.temporal_merge_mode_var,
            "temporal_merge_tau_manual": self.temporal_merge_tau_manual_var,
            "temporal_merge_tau": self.temporal_merge_tau_var,
            "temporal_merge_share_rounding": self.temporal_merge_share_rounding_var,
            "temporal_merge_one_sided_mass_ratio": self.temporal_merge_mass_ratio_var,
            "area_outlier_method": self.area_outlier_method_var,
            "area_iqr_min": self.area_iqr_min_var,
            "area_iqr_max": self.area_iqr_max_var,
            "area_absolute_min": self.area_absolute_min_var,
            "area_absolute_max": self.area_absolute_max_var,
            "result_import_enabled": self.result_import_enabled_var,
            "result_import_path": self.result_import_path_var,
            "result_import_min_coverage": self.result_import_min_coverage_var,
            "result_import_area_min": self.result_import_area_min_var,
            "result_import_area_max": self.result_import_area_max_var,
            "result_import_show_obb": self.result_import_show_obb_var,
            "show_overlay": self.show_overlay_var,
            "show_centers": self.show_centers_var,
            "show_contours": self.show_contours_var,
        }

    def _collect_config(self, path: Optional[str] = None) -> dict:
        self._sync_roi_vars_to_set(self.roi_active_set_idx)
        self._sync_additional_outlier_vars_to_set()
        settings = {}
        for key, var in self._config_vars().items():
            try:
                settings[key] = var.get()
            except Exception:
                pass
        settings["roi_sets"] = [normalize_roi_set(r) for r in self.roi_sets]
        settings["additional_outlier_sets"] = [dict(r) for r in self.additional_outlier_sets]
        config_dir = os.path.dirname(os.path.abspath(path or self.config_path or self._default_config_path()))
        settings["result_import_path"] = to_relative_path(settings.get("result_import_path", ""), config_dir)

        analysis_iqr_stats = {
            k: [float(x) for x in v]
            for k, v in self.analysis_iqr_stats.items()
        }
        analysis_bounds = {
            k: [float(v[0]), float(v[1])]
            for k, v in self.analysis_bounds.items()
        }

        return {
            "version": 5,
            "app": APP_TITLE,
            "settings": settings,
            # Empty unless this video is an analysis copy; read back from the
            # record FFmpeg conversion left beside it, so it is never stale.
            "video_conversion": video_conversion_record(self.video_path_var.get().strip()),
            "hidden_state": {
                "sampled_frame_indices": [int(x) for x in self.analysis_frames],
                "analysis_iqr_stats": analysis_iqr_stats,
                "analysis_bounds": analysis_bounds,
                "analysis_signature": repr(self.analysis_signature),
                "analysis_is_stale": bool(self.analysis_is_stale),
                "frame_count": int(self.frame_count),
                "current_frame": int(self.current_frame),
            },
        }

    @staticmethod
    def _default_frame_ranges(total_frames: int) -> dict[str, int]:
        last = max(0, int(total_frames) - 1)
        training_end = (
            -1  # covers the whole video; -1 means "last frame" everywhere else in this file
            if int(total_frames) < LONG_VIDEO_FRAME_COUNT_THRESHOLD
            else min(DEFAULT_TRAINING_MAX_FRAME_INDEX, last)
        )
        return {
            "background_frame_start": 0,
            "background_frame_end": last,
            "training_frame_start": 0,
            "training_frame_end": training_end,
            "training_frame_interval": 5,
        }

    def _set_frame_range_vars(self, values: dict[str, int]) -> None:
        self.background_frame_start_var.set(int(values["background_frame_start"]))
        self.background_frame_end_var.set(int(values["background_frame_end"]))
        self.training_frame_start_var.set(int(values["training_frame_start"]))
        self.training_frame_end_var.set(int(values["training_frame_end"]))
        self.training_frame_interval_var.set(max(1, int(values["training_frame_interval"])))

    def _resolved_frame_end(self, frame_end: int) -> int:
        if self.frame_count <= 0:
            return int(frame_end)
        end = int(frame_end)
        return self.frame_count - 1 if end == -1 else min(self.frame_count - 1, end)

    def _update_geometry_control_ranges(self) -> None:
        """Fit the ROI frame-range sliders to the loaded video."""
        if self.reader is None:
            return
        last = max(0, int(self.frame_count) - 1)
        for scale in (self.roi_frame_start_scale, self.additional_outlier_frame_start_scale):
            scale.set_value_range(0, last)
        for scale in (self.roi_frame_end_scale, self.additional_outlier_frame_end_scale):
            scale.set_value_range(-1, last, slider_lower=0)

    def _frame_range_label(self, frame_start: int, frame_end: int) -> str:
        start = int(frame_start)
        end = int(frame_end)
        if end == -1:
            return f"{start}-{self._resolved_frame_end(end)} (-1=last)"
        return f"{start}-{end}"

    def _validate_frame_range_values(
        self,
        background_start: int,
        background_end: int,
        training_start: int,
        training_end: int,
        training_interval: int,
    ) -> dict[str, int]:
        if self.frame_count <= 0:
            raise ValueError("No video has been loaded.")
        last = self.frame_count - 1
        values = {
            "background_frame_start": int(background_start),
            "background_frame_end": int(background_end),
            "training_frame_start": int(training_start),
            "training_frame_end": int(training_end),
            "training_frame_interval": int(training_interval),
        }
        for label, start_key, end_key in (
            ("Background estimation", "background_frame_start", "background_frame_end"),
            ("Training data preparation", "training_frame_start", "training_frame_end"),
        ):
            start = values[start_key]
            end = values[end_key]
            effective_end = self._resolved_frame_end(end)
            if end != -1 and not (0 <= end < self.frame_count):
                raise ValueError(
                    f"{label}: end must be -1 or satisfy 0 <= end < total frame count "
                    f"(got end={end}, valid end={last})."
                )
            if not (0 <= start <= effective_end):
                raise ValueError(
                    f"{label}: require 0 <= start <= end, where end may be -1 for the last frame "
                    f"(got start={start}, end={end}, valid end={last})."
                )
        if values["training_frame_interval"] < 1:
            raise ValueError("Training data preparation: Frame interval must be at least 1.")
        return values

    def _apply_frame_ranges_from_settings(self, settings: dict) -> None:
        if self.reader is None or self.frame_count <= 0:
            return

        raw = {
            "background_frame_start": settings["background_frame_start"],
            "background_frame_end": settings["background_frame_end"],
            "training_frame_start": settings["training_frame_start"],
            "training_frame_end": settings["training_frame_end"],
            "training_frame_interval": settings["training_frame_interval"],
        }

        last = self.frame_count - 1
        errors: list[str] = []

        def _coerce_range(label: str, start_key: str, end_key: str) -> tuple[int, int]:
            try:
                start = int(raw[start_key])
                raw_end = int(raw[end_key])
            except Exception:
                errors.append(f"{label}: invalid saved frame range; using full video range.")
                return 0, last
            if raw_end < -1:
                errors.append(f"{label}: saved frame range is invalid for this video; using full video range.")
                return 0, last
            end = -1 if raw_end == -1 else min(raw_end, last)
            effective_end = last if end == -1 else end
            if start < 0 or start > effective_end:
                errors.append(f"{label}: saved frame range is invalid for this video; using full video range.")
                return 0, last
            return start, end

        bg_start, bg_end = _coerce_range("Background estimation", "background_frame_start", "background_frame_end")
        tr_start, tr_end = _coerce_range("Training data preparation", "training_frame_start", "training_frame_end")
        try:
            interval = int(raw["training_frame_interval"])
            if interval < 1:
                errors.append("Training data preparation: saved frame interval is invalid; using 5.")
                interval = 5
        except Exception:
            errors.append("Training data preparation: invalid frame interval; using 5.")
            interval = 5

        self._set_frame_range_vars({
            "background_frame_start": bg_start,
            "background_frame_end": bg_end,
            "training_frame_start": tr_start,
            "training_frame_end": tr_end,
            "training_frame_interval": interval,
        })
        if errors:
            messagebox.showwarning("Frame Range Settings", "\n".join(errors))

    def _apply_config(self, config: dict):
        settings = config["settings"]
        hidden_state = config.get("hidden_state", {})
        video_path = str(settings.get("video_path", "") or "")
        vars_by_key = self._config_vars()

        if video_path and os.path.exists(video_path):
            self.video_path_var.set(video_path)
            self.load_video(video_path)
        elif video_path:
            self.video_path_var.set(video_path)
            messagebox.showwarning("Warning", f"Video file not found. Loading settings only.\n{video_path}")

        for key, value in settings.items():
            var = vars_by_key.get(key)
            if var is None:
                continue
            try:
                var.set(value)
            except Exception:
                pass
        if "show_centers" not in settings:
            self.show_centers_var.set(True)
        # Old configs must not inherit optional settings from another video.
        if "static_exclusion_enabled" not in settings:
            self.static_exclusion_enabled_var.set(False)
        if "static_presence" not in settings:
            self.static_presence_var.set(0.98)
        if "static_min_iou" not in settings:
            self.static_iou_var.set(0.8)
        if "static_max_movement" not in settings:
            self.static_max_movement_var.set(0.2)
        # Configs written before the temporal merge filter keep it disabled.
        self.temporal_merge_enabled_var.set(bool(settings.get("temporal_merge_enabled", False)))
        mode = str(settings.get("temporal_merge_mode", "one_sided"))
        self.temporal_merge_mode_var.set(mode if mode in ("bilateral", "one_sided") else "one_sided")
        self.temporal_merge_tau_manual_var.set(bool(settings.get("temporal_merge_tau_manual", True)))
        try:
            self.temporal_merge_tau_var.set(min(0.9, max(0.1, float(settings.get("temporal_merge_tau", 0.30)))))
        except (TypeError, ValueError):
            self.temporal_merge_tau_var.set(0.30)
        # Ignore legacy horizon/confirmation keys in saved configs; those
        # controls no longer exist. Restore the new parameters explicitly.
        for key, var, default, lower, upper in (
            ("temporal_merge_share_rounding", self.temporal_merge_share_rounding_var, 0.25, 0.0, 1.0),
            ("temporal_merge_one_sided_mass_ratio", self.temporal_merge_mass_ratio_var, 0.60, 0.01, 1.0),
        ):
            try:
                value = float(settings.get(key, default))
                if not math.isfinite(value):
                    raise ValueError("nonfinite")
                var.set(max(lower, min(upper, value)))
            except (TypeError, ValueError):
                var.set(default)
        # Loading an older session must not inherit smoothing from the last one.
        self.single_blob_smoothing_enabled_var.set(bool(settings.get("single_blob_smoothing_enabled", False)))
        self.single_blob_smoothing_level_var.set(max(1, min(20, int(settings.get("single_blob_smoothing_level", 3)))))
        # Missing bounds adopt real-blob defaults after the loaded segmentation
        # settings are applied; explicit saved limits remain user settings.
        self._result_area_control._absolute_defaults_pending = not any(
            key in settings for key in ("result_import_area_min", "result_import_area_max")
        )
        self.result_import_area_min_var.set(int(settings.get("result_import_area_min", 0)))
        self.result_import_area_max_var.set(int(settings.get("result_import_area_max", 1000)))
        self._current_result_area_bounds()

        self._apply_frame_ranges_from_settings(settings)
        if "static_frame_window" not in settings and self.reader is not None:
            fps = float(self.reader.fps) if self.reader.fps and self.reader.fps > 0 else 30.0
            self.static_frame_window_var.set(max(2, int(round(0.5 * fps))))
        # Static Search remains transient; temporal Search is restored from
        # a validated disk cache after all video/ROI/background settings load.
        self._static_search_signature = None
        self._static_exclusions = {}
        self._static_digests = None
        self._toggle_static_options()
        self.static_search_status_var.set(
            "Not searched — press Search before Processing"
            if self.static_exclusion_enabled_var.get() else "Disabled"
        )
        self._toggle_temporal_options()
        self._reset_temporal_search()
        self._toggle_area_outlier_method_controls()

        image_shape = (self.reader.height, self.reader.width) if self.reader is not None else None
        self.roi_sets, reverse = load_roi_settings(settings, image_shape)
        self.roi_reverse_var.set(reverse)
        self.roi_edit_var.set(False)
        self._roi_drag_mode = None
        self.current_mask_cache.clear()
        self.roi_active_set_idx = 0
        self._update_roi_set_selector_values()
        self._switching_roi_set = True
        try:
            self._sync_set_to_roi_vars(0)
            self._update_roi_set_selector()
        finally:
            self._switching_roi_set = False

        self._refresh_segmentation_controls()
        self._toggle_roi_params()

        try:
            legacy_additional_config = int(config.get("version", 0)) < 4
        except (TypeError, ValueError):
            legacy_additional_config = True
        outlier_sets_data = settings.get("additional_outlier_sets")
        if outlier_sets_data and isinstance(outlier_sets_data, list):
            loaded_outliers = [dict(r) for r in outlier_sets_data if isinstance(r, dict)]
            legacy_additional_config = legacy_additional_config or any(
                "contours" in item for item in loaded_outliers
            )
            self.additional_outlier_sets = [
                self._normalize_additional_outlier_set(r) for r in loaded_outliers
            ] or [self._default_additional_outlier_set()]
        else:
            self.additional_outlier_sets = [self._normalize_additional_outlier_set({
                "enabled": bool(self.additional_outlier_enabled_var.get()),
                "frame_start": int(settings.get("additional_outlier_frame_start", 0)),
                "frame_end": int(settings.get("additional_outlier_frame_end", -1)),
                "segmentation_mode": settings.get("additional_outlier_segmentation_mode", "background_diff"),
                "bg_method": settings.get("additional_outlier_bg_method", "median"),
                "threshold": settings.get("additional_outlier_threshold", 50),
                "dark_threshold": settings.get("additional_outlier_dark_threshold", 50),
                "bright_threshold": settings.get("additional_outlier_bright_threshold", 200),
                "diff_threshold": settings.get("additional_outlier_diff_threshold", 50),
                "min_area": settings.get("additional_outlier_min_area", 10),
            })]
        self.additional_outlier_active_set_idx = 0
        self._sync_additional_outlier_set_to_vars()
        self._update_additional_outlier_set_selector_values()
        self._update_additional_outlier_set_selector()
        self._refresh_additional_outlier_controls()
        self._toggle_additional_outlier_params()
        self.additional_outlier_revision += 1

        config_dir = os.path.dirname(os.path.abspath(self.config_path)) if self.config_path else ""
        result_import_path = resolve_path(settings.get("result_import_path", ""), config_dir)
        self.result_import_path_var.set(result_import_path)
        self.result_obb_import = None
        if result_import_path:
            try:
                self.result_obb_import = _load_result_obb_csv(result_import_path)
            except Exception as exc:
                self.set_status(f"Result OBB import failed: {exc}")
        self.result_import_status_var.set(self._result_import_status_text())
        self._toggle_result_import_params()
        self._refresh_result_match()

        try:
            self.analysis_frames = [int(x) for x in hidden_state.get("sampled_frame_indices", [])]
        except Exception:
            self.analysis_frames = []

        self.analysis_iqr_stats = self._parse_saved_analysis_stats(hidden_state)

        if self.analysis_iqr_stats:
            try:
                self.analysis_bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats)
                self.analysis_signature = ("loaded_config", len(self.analysis_frames), self._classifier_signature())
                self.analysis_is_stale = bool(hidden_state.get("analysis_is_stale", False))
                if self.analysis_is_stale:
                    self.analysis_ready_var.set("analysis: stale (re-run Analyze)")
                else:
                    self.analysis_ready_var.set(
                        f"analysis: loaded config (sampled frames={len(self.analysis_frames)})"
                    )
                self._schedule_area_iqr_boxplot_redraw()
            except ValueError as exc:
                self.analysis_bounds = {}
                self.analysis_signature = None
                self.analysis_is_stale = True
                self.analysis_ready_var.set("analysis: not computed")
                messagebox.showwarning("Area outlier method", str(exc))
        else:
            self.analysis_bounds = {}
            self.analysis_signature = None
            self.analysis_is_stale = True
            self.analysis_ready_var.set("analysis: not computed")

        if legacy_additional_config:
            self.analysis_blobs_by_frame.clear()
            self.analysis_outlier_frames.clear()
            self.analysis_frames = []
            self.analysis_signature = None
            self.analysis_bounds = {}
            self.analysis_iqr_stats = {}
            self.analysis_is_stale = True
            self.analysis_ready_var.set("analysis: not computed")

        self.outlier_section_revealed = bool(self.analysis_iqr_stats) and not self.analysis_is_stale
        self._guidance_stage = "outlier" if self.outlier_section_revealed else "background"

        # background.png always lives next to this JSON config itself (see
        # _default_background_path()), so no separate stored path is needed.
        loaded_background = self._load_saved_background(self._default_background_path())
        if loaded_background:
            load_background_path, bg, active_method = loaded_background
            self.background_bgr = bg
            self.active_background_method = active_method
            self.current_mask_cache.clear()
            self.background_ready_var.set(f"background: loaded ({os.path.basename(load_background_path)})")
            if self.bg_preview_button is not None:
                self.bg_preview_button.configure(state="normal")
        elif not self._segmentation_needs_background():
            self.background_ready_var.set(f"background: not required ({self.segmentation_mode_var.get()})")
            if self.bg_preview_button is not None:
                self.bg_preview_button.configure(state="disabled")
        self._restore_temporal_after_background = False
        if (
            self.reader is not None
            and self._segmentation_needs_background()
            and self.background_bgr is None
        ):
            try:
                self.compute_background()
                self._restore_temporal_after_background = self._background_building
            except Exception as exc:
                self.set_status(f"Background build failed: {exc}")

        # On a cache miss the background now finishes asynchronously.
        # Restore dependent Search data only after its completion.
        if not self._restore_temporal_after_background:
            self._restore_temporal_search_cache()
        self._set_workflow_stage("all")
        self._update_analysis_dependent_buttons()
        self._schedule_area_iqr_boxplot_redraw()
        try:
            frame_idx = int(hidden_state.get("current_frame", self.current_frame))
            if self.reader is not None and self.frame_count > 0:
                self.set_frame(frame_idx)
            else:
                self.redraw_current_frame()
        except Exception:
            self.redraw_current_frame()
        self._schedule_area_iqr_boxplot_redraw()
        self.set_status("Config loaded")

    @staticmethod
    def _parse_area_stat_value(value) -> Optional[tuple[float, float, float, float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        vals = [float(x) for x in value]
        return vals[0], vals[1], vals[2], vals[3]

    def _parse_saved_analysis_stats(self, hidden_state: dict) -> dict[str, tuple[float, float, float, float]]:
        parsed_stats: dict[str, tuple[float, float, float, float]] = {}
        try:
            raw_stats = hidden_state.get("analysis_iqr_stats", {})
            if isinstance(raw_stats, dict):
                for key, value in raw_stats.items():
                    if str(key) != "area":
                        continue
                    parsed = self._parse_area_stat_value(value)
                    if parsed is not None:
                        parsed_stats["area"] = parsed
        except Exception:
            parsed_stats = {}

        return parsed_stats

    def _resolve_saved_background_path(self, path: str) -> str:
        expanded = os.path.expanduser(str(path or ""))
        if not expanded:
            return ""
        if not os.path.isabs(expanded) and self.config_path:
            expanded = os.path.join(os.path.dirname(os.path.abspath(self.config_path)), expanded)
        return os.path.abspath(expanded)

    def _read_background_for_current_video(self, path: str) -> Optional[np.ndarray]:
        if not path or not os.path.exists(path):
            return None
        bg = cv2.imread(path, cv2.IMREAD_COLOR)
        if bg is None:
            return None
        if self.reader is not None and bg.shape[:2] != (self.reader.height, self.reader.width):
            return None
        return bg

    def _load_saved_background(self, saved_background_path: str) -> Optional[tuple[str, np.ndarray, Optional[str]]]:
        selected_method = self.bg_method_var.get()
        candidates: list[tuple[str, Optional[str]]] = []

        try:
            if self._background_cache_is_current(selected_method):
                candidates.append((self._method_background_path(selected_method), selected_method))
        except Exception:
            pass

        for candidate in (
            saved_background_path,
            self._default_background_path(),
        ):
            if candidate:
                candidates.append((candidate, None))

        try:
            candidates.append((self._method_background_path(selected_method), selected_method))
        except Exception:
            pass

        seen: set[str] = set()
        default_path = self._default_background_path()
        for candidate, active_method in candidates:
            path = self._resolve_saved_background_path(candidate)
            if not path:
                continue
            key = os.path.normcase(path)
            if key in seen:
                continue
            seen.add(key)
            bg = self._read_background_for_current_video(path)
            if bg is None:
                continue
            try:
                if os.path.normcase(os.path.abspath(default_path)) != key:
                    shutil.copyfile(path, default_path)
            except Exception:
                pass
            return path, bg, active_method
        return None

    def _default_config_path(self) -> str:
        return os.path.join(self._default_output_dir(), "segmentation_gui_config.json")

    def _write_config_to_path(self, path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self._collect_config(path), f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
        self.config_path = path

    def _save_config_to_default_path(self) -> str:
        """Write to the config this session came from, or the default location."""
        path = self.config_path or self._default_config_path()
        self._write_config_to_path(path)
        return path

    def _save_config_shortcut(self):
        # Clicking Save config commits the active numeric entry first. Do the
        # same when a keyboard shortcut is used without changing focus.
        commit = self._numeric_entry_commits.get(self.focus_get())
        if commit is not None:
            commit()
        self.save_config()

    def save_config(self):
        """Save config and confirm success with a dialog only after the write completes."""
        try:
            path = self._save_config_to_default_path()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.set_status(f"Config saved: {path}")
        messagebox.showinfo("Config saved", f"Configuration saved successfully:\n{path}", parent=self)

    @staticmethod
    def _find_default_video_for_config(config_path: str, configured_video_path: str) -> str:
        """Find the conventional source-video location implied by a config path.

        Segmentation normally writes its JSON under
        ``<video_dir>/amadeus_<stem>/segmentation/``.  A moved config can keep
        the old absolute ``video_path`` even when the video moved with the
        project, so check the config directory, its project directory, and the
        project's parent before asking the user to locate the video manually.
        """
        configured_video_path = str(configured_video_path or "").strip()
        video_name = os.path.basename(os.path.normpath(configured_video_path))
        if not video_name:
            return ""

        config_path = str(config_path or "").strip()
        if not config_path:
            return ""
        config_dir = os.path.dirname(os.path.abspath(os.path.expanduser(config_path)))
        project_dir = os.path.dirname(config_dir)
        video_dir = os.path.dirname(project_dir)
        candidates = (
            os.path.join(config_dir, video_name),
            os.path.join(project_dir, video_name),
            os.path.join(video_dir, video_name),
        )

        seen: set[str] = set()
        for candidate in candidates:
            candidate = os.path.abspath(candidate)
            key = os.path.normcase(candidate)
            if key in seen:
                continue
            seen.add(key)
            if os.path.isfile(candidate):
                return candidate
        return ""

    def load_config(self, path: str = ""):
        if not path:
            path = filedialog.askopenfilename(
                title="Load config",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)

            settings = config["settings"]
            configured_video_path = str(settings.get("video_path", "") or "")
            if configured_video_path and not os.path.exists(configured_video_path):
                default_video_path = self._find_default_video_for_config(
                    path, configured_video_path
                )
                if default_video_path:
                    settings["video_path"] = default_video_path
                else:
                    answer = messagebox.askyesno(
                        "Video path not found",
                        "The video path saved in this config was not found:\n"
                        f"{configured_video_path}\n\n"
                        "Select a new video path?\n"
                        "Yes: select a path\n"
                        "No: continue with the current path",
                        parent=self,
                    )
                    if answer:
                        configured_dir = os.path.dirname(configured_video_path)
                        initial_dir = (
                            configured_dir
                            if os.path.isdir(configured_dir)
                            else os.path.dirname(path)
                        )
                        replacement_path = filedialog.askopenfilename(
                            title="Video from config not found. Please select the corresponding video",
                            parent=self,
                            initialdir=initial_dir,
                            initialfile=os.path.basename(configured_video_path),
                            filetypes=VIDEO_EXTS,
                        )
                        if replacement_path:
                            settings["video_path"] = replacement_path

            self._load_config_dict(path, config)
        except Exception as e:
            self._applying_config = False
            messagebox.showerror("Error", str(e))

    def _load_config_dict(self, path: str, config: dict) -> None:
        """Apply an already-parsed segmentation_gui_config.json and remember its path."""
        settings = config["settings"]
        self.config_path = path
        self._applying_config = True
        try:
            # load_video runs before the rest of _apply_config; seed the
            # background method so automatic extraction uses the config value.
            if "bg_method" in settings:
                self.bg_method_var.set(settings["bg_method"])
            self._apply_config(config)
        finally:
            self._applying_config = False
        self._update_temporal_status()
        self._refresh_search_glows()
        if (
            self._additional_outlier_needs_background()
            and not self._additional_outlier_background_methods().issubset(
                self._background_cache_current_methods()
            )
        ):
            self._schedule_background_rebuild(delay_ms=50)
        self._area_reference_key = None
        self._min_area_slider_cap = None
        self._absolute_zoomed = False
        self._absolute_dragging = False
        self._result_area_control._absolute_zoomed = False
        self._result_area_control._absolute_dragging = False
        self._result_area_reference_key = None
        self._update_area_control_ranges()
        # Persist the loaded state (and any replacement video path) immediately.

    def _find_existing_config_for_video(
        self, video_path: str, session_path: str = "",
    ) -> Optional[tuple[str, dict]]:
        """Locate a previously saved segmentation_gui_config.json for this video.

        The config is not always saved next to the video: a session's config.yaml
        can point PICKLE_PATH at a differently-located segmentation output
        directory (Advanced Tracking allows a custom SESSION_PATH), and that is
        where this session then saves. Check both candidate locations, and
        only trust a candidate whose own recorded video_path actually matches the
        video being loaded.
        """
        candidates: list[str] = []

        session_path = str(session_path or "").strip()
        if session_path:
            # The GUI can finish Analyze before Easy Tracking creates config.yaml.
            candidates.append(os.path.join(session_path, "segmentation", "segmentation_gui_config.json"))
            session_cfg_path = os.path.join(session_path, "config.yaml")
            if os.path.isfile(session_cfg_path):
                try:
                    with open(session_cfg_path, "r", encoding="utf-8") as f:
                        session_cfg = yaml.safe_load(f) or {}
                    session_cfg = resolve_config_paths(session_cfg)
                    pickle_path = str(session_cfg.get("PICKLE_PATH", "") or "").strip()
                    if pickle_path:
                        seg_dir = os.path.dirname(os.path.abspath(os.path.expanduser(pickle_path)))
                        candidates.append(os.path.join(seg_dir, "segmentation_gui_config.json"))
                except Exception:
                    pass

        video_seg_dir = self._segmentation_dir_for_video(video_path)
        if video_seg_dir:
            candidates.append(os.path.join(video_seg_dir, "segmentation_gui_config.json"))

        target = os.path.normcase(os.path.abspath(os.path.expanduser(str(video_path or ""))))
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate:
                continue
            candidate = os.path.abspath(candidate)
            key = os.path.normcase(candidate)
            if key in seen or not os.path.isfile(candidate):
                continue
            seen.add(key)
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    config = json.load(f)
                configured_video = str(config.get("settings", {}).get("video_path", "") or "")
                if not configured_video:
                    continue
                configured_target = os.path.normcase(
                    os.path.abspath(os.path.expanduser(configured_video))
                )
                if configured_target != target:
                    continue
            except Exception:
                continue
            return candidate, config

        return None

    def _invalidate_sampled_analysis(self, message: str = "analysis: not computed") -> None:
        self.analysis_blobs_by_frame.clear()
        self.analysis_outlier_frames.clear()
        self.analysis_frames = []
        self.analysis_signature = None
        self.analysis_bounds = {}
        self.analysis_iqr_stats = {}
        self.analysis_is_stale = True
        self.analysis_ready_var.set(message)
        self.outlier_section_revealed = False
        if self._guidance_stage in ("outlier", "processing"):
            self._guidance_stage = "analysis"
        self._set_workflow_stage("analysis")
        self._redraw_area_iqr_boxplot()
        self._update_analysis_dependent_buttons()

    def _apply_frame_ranges(self, values: dict[str, int], *, source: str) -> None:
        old_bg = (int(self.background_frame_start_var.get()), int(self.background_frame_end_var.get()))
        old_training = (int(self.training_frame_start_var.get()), int(self.training_frame_end_var.get()))
        old_interval = max(1, int(self.training_frame_interval_var.get()))

        self._set_frame_range_vars(values)

        new_bg = (values["background_frame_start"], values["background_frame_end"])
        new_training = (values["training_frame_start"], values["training_frame_end"])
        new_interval = values["training_frame_interval"]

        background_changed = new_bg != old_bg
        training_range_changed = new_training != old_training
        interval_changed = int(new_interval) != int(old_interval)

        if background_changed:
            self.background_bgr = None
            self.active_background_method = None
            self.current_mask_cache.clear()
            self._invalidate_analysis_for_background_change()
            self.background_ready_var.set("background: not computed")
            if self.bg_preview_button is not None:
                self.bg_preview_button.configure(state="disabled")
            if self._segmentation_needs_background() or self._additional_outlier_needs_background():
                self._set_workflow_stage("background")
                self.compute_background()
            else:
                self._set_workflow_stage("analysis")
                self.redraw_current_frame()
        elif training_range_changed:
            self._invalidate_sampled_analysis()
            self.redraw_current_frame()
        elif interval_changed:
            self.redraw_current_frame()

        if source:
            self.set_status(
                f"{source}: background {self._frame_range_label(new_bg[0], new_bg[1])}, "
                f"training {self._frame_range_label(new_training[0], new_training[1])}, interval {new_interval}"
            )

    def change_frame_ranges(self):
        self._show_frame_range_dialog(auto=False)

    def _show_frame_range_dialog(self, *, auto: bool = False) -> bool:
        if self.reader is None or self.frame_count <= 0:
            messagebox.showerror("Error", "Please load a video first.")
            return False

        popup = ctk.CTkToplevel(self)
        popup.title("Frame Range Settings")
        popup.transient(self)
        popup.resizable(False, False)
        popup.configure(fg_color=_POPUP_BG)

        result = {"applied": False, "values": None}
        total = int(self.frame_count)
        last = total - 1

        outer = tk.Frame(popup, padx=18, pady=16, bg=_POPUP_BG)
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer,
            text=f"Total frames: {total:,}\nValid frame indices: 0-{last:,}\nEnd frame = -1 uses the last frame.",
            justify="left",
            anchor="w", bg=_POPUP_BG, fg=_POPUP_FG,
        ).pack(fill="x", pady=(0, 10))

        tk.Label(
            outer,
            text=(
                "These ranges are used only for background estimation and training-data preparation.\n"
                "They do not restrict the frame range of the final tracking analysis."
            ),
            justify="left",
            anchor="w", bg=_POPUP_BG, fg=_POPUP_FG,
        ).pack(fill="x", pady=(0, 12))

        entries: dict[str, tk.Spinbox] = {}

        def _section(title: str) -> tk.Frame:
            frame = tk.LabelFrame(outer, text=title, padx=10, pady=8, bg=_POPUP_BG, fg=_POPUP_FG)
            frame.pack(fill="x", pady=(0, 10))
            return frame

        def _row(parent: tk.Frame, label: str, key: str, initial: int, from_: int = 0, to_: Optional[int] = None) -> None:
            row = tk.Frame(parent, bg=_POPUP_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=16, anchor="w", bg=_POPUP_BG, fg=_POPUP_FG).pack(side="left")
            sp = dark_spinbox(row, from_=from_, to=max(int(to_ if to_ is not None else last), 1), increment=1, width=12, **_SPIN_CFG)
            sp.delete(0, tk.END)
            sp.insert(0, str(int(initial)))
            sp.pack(side="left")
            self._register_popup_integer_entry(sp)
            entries[key] = sp

        bg_frame = _section("Background estimation")
        _row(bg_frame, "Start frame", "background_frame_start", int(self.background_frame_start_var.get()))
        _row(bg_frame, "End frame (-1=last)", "background_frame_end", int(self.background_frame_end_var.get()), from_=-1)

        tr_frame = _section("Training data preparation")
        _row(tr_frame, "Start frame", "training_frame_start", int(self.training_frame_start_var.get()))
        _row(tr_frame, "End frame (-1=last)", "training_frame_end", int(self.training_frame_end_var.get()), from_=-1)
        _row(tr_frame, "Frame interval", "training_frame_interval", max(1, int(self.training_frame_interval_var.get())), from_=1, to_=1000000)

        error_var = tk.StringVar(value="")
        tk.Label(outer, textvariable=error_var, anchor="w", justify="left", fg="#FF8787", bg=_POPUP_BG).pack(fill="x")

        buttons = tk.Frame(outer, bg=_POPUP_BG)
        buttons.pack(fill="x", pady=(12, 0))

        def _read_int(key: str) -> int:
            return int(str(entries[key].get()).strip())

        def _apply():
            try:
                values = self._validate_frame_range_values(
                    _read_int("background_frame_start"),
                    _read_int("background_frame_end"),
                    _read_int("training_frame_start"),
                    _read_int("training_frame_end"),
                    _read_int("training_frame_interval"),
                )
            except Exception as exc:
                error_var.set(str(exc))
                return
            result["applied"] = True
            result["values"] = values
            popup.destroy()

        def _cancel():
            popup.destroy()

        _popup_button(buttons, text="Apply", command=_apply).pack(side="right", padx=(8, 0))
        _popup_button(buttons, text="Cancel", command=_cancel).pack(side="right")
        popup.protocol("WM_DELETE_WINDOW", _cancel)

        popup.update_idletasks()
        width = max(460, popup.winfo_reqwidth())
        height = max(420, popup.winfo_reqheight())
        root_x = self.winfo_rootx()
        root_y = self.winfo_rooty()
        root_w = max(1, self.winfo_width())
        root_h = max(1, self.winfo_height())
        popup.geometry(f"{width}x{height}+{root_x + max(0, (root_w - width) // 2)}+{root_y + max(0, (root_h - height) // 2)}")
        popup.grab_set()
        popup.focus_force()
        self.wait_window(popup)
        if result["applied"] and result["values"] is not None:
            self._flush_ui()
            self._apply_frame_ranges(result["values"], source="Frame ranges updated")
        elif auto:
            self.set_status("Frame range defaults kept.")
        return bool(result["applied"])

    def _pack_combobox(self, parent, label, var, values):
        row = ctk.CTkFrame(parent, corner_radius=0)
        row.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(row, text=label, width=162, anchor="w").pack(side="left")
        combo = ctk.CTkComboBox(row, values=values, state="readonly", width=160,
                                command=lambda choice: var.set(choice))
        combo.set(var.get())
        var.trace_add("write", lambda *_: combo.set(var.get()))
        combo.pack(side="left", fill="x", expand=True)

    def _pack_scale_entry(self, parent, label, var, lo, hi, resolution, is_float: bool = False, enable_var: Optional[tk.BooleanVar] = None, entry_prefix: str = "", slider_lo=None, slider_hi=None, end_sentinel: Optional[int] = None, show_slider: bool = True):
        outer = ctk.CTkFrame(parent, corner_radius=0)
        outer.pack(fill="x", padx=6, pady=3)

        row = ctk.CTkFrame(outer, corner_radius=0)
        row.pack(fill="x")
        if enable_var is not None:
            ctk.CTkCheckBox(row, text="", variable=enable_var).pack(side="left", padx=(0, 4))
        ctk.CTkLabel(row, text=label, width=144, anchor="w").pack(side="left")

        fmt = ("{:.6g}" if entry_prefix == "IQRx" else
               ("{:.2f}" if is_float and float(resolution) < 0.1 else
                "{:.1f}" if is_float else "{:d}"))
        default_value = float(var.get()) if is_float else int(var.get())
        state = {"updating": False}

        spinbox = dark_spinbox(row, from_=lo, to=hi, increment=resolution, width=8, **_SPIN_CFG)

        def clamp_value(v):
            lo, hi = float(spinbox.cget("from")), float(spinbox.cget("to"))
            if is_float:
                v = float(v)
                v = max(float(lo), min(float(hi), v))
                if v <= lo or v >= hi:
                    return v
                return max(lo, min(hi, round(v / float(resolution)) * float(resolution)))
            v = int(round(float(v)))
            return max(int(lo), min(int(hi), v))

        def scale_position(value):
            """A sentinel value ("-1" = last frame) belongs at the slider maximum."""
            if end_sentinel is not None and float(value) == float(end_sentinel):
                return float(scale.cget("to"))
            return float(value)

        def sync_from_var(*_):
            if state["updating"]:
                return
            state["updating"] = True
            try:
                value = float(var.get()) if is_float else int(var.get())
                spinbox.delete(0, tk.END)
                spinbox.insert(0, fmt.format(value))
                try:
                    scale.set(scale_position(value))
                except Exception:
                    pass
            finally:
                state["updating"] = False

        def commit_entry(_event=None):
            if state["updating"]:
                return
            raw = spinbox.get().strip()
            try:
                value = clamp_value(float(raw) if raw else var.get())
            except Exception:
                sync_from_var()
                return "break"
            state["updating"] = True
            try:
                var.set(value)
                spinbox.delete(0, tk.END)
                spinbox.insert(0, fmt.format(float(value) if is_float else int(value)))
                scale.set(scale_position(value))
            finally:
                state["updating"] = False
            return "break"

        def on_slider(val):
            clamped = clamp_value(val)
            if not state["updating"]:
                state["updating"] = True
                try:
                    var.set(clamped)
                    spinbox.delete(0, tk.END)
                    spinbox.insert(0, fmt.format(float(clamped) if is_float else int(clamped)))
                finally:
                    state["updating"] = False

        def on_spin():
            raw = spinbox.get().strip()
            try:
                value = clamp_value(float(raw) if raw else var.get())
            except Exception:
                sync_from_var()
                return
            if not state["updating"]:
                state["updating"] = True
                try:
                    var.set(value)
                    scale.set(scale_position(value))
                finally:
                    state["updating"] = False

        spinbox.configure(command=on_spin)
        spinbox.pack(side="right")
        if entry_prefix:
            ctk.CTkLabel(row, text=entry_prefix, anchor="e").pack(side="right", padx=(4, 2))

        scale = ctk.CTkSlider(outer,
                              from_=lo if slider_lo is None else slider_lo,
                              to=hi if slider_hi is None else slider_hi,
                              orientation="horizontal", command=on_slider)
        scale.set(float(default_value))
        if show_slider:
            scale.pack(fill="x", pady=(2, 0))

        def reset_to_default(_event=None):
            try:
                var.set(clamp_value(default_value))
                sync_from_var()
            except Exception:
                pass
            return "break"

        scale.bind("<Double-Button-1>", reset_to_default, add="+")

        def sync_enabled_state(*_):
            if enable_var is None:
                return
            s = "normal" if enable_var.get() else "disabled"
            scale.configure(state=s)
            spinbox.config(state=s)

        spinbox.bind("<Return>", commit_entry, add="+")
        spinbox.bind("<KP_Enter>", commit_entry, add="+")
        spinbox.bind("<FocusOut>", commit_entry, add="+")
        self._register_numeric_entry(spinbox, commit_entry)
        var.trace_add("write", sync_from_var)
        if enable_var is not None:
            enable_var.trace_add("write", sync_enabled_state)
        sync_from_var()
        sync_enabled_state()
        def set_value_range(lower, upper, *, clamp_current=True, slider_lower=None, slider_upper=None):
            spinbox.configure(from_=lower, to=upper)
            scale.configure(from_=lower if slider_lower is None else max(lower, slider_lower),
                            to=upper if slider_upper is None else min(upper, slider_upper))
            value = float(var.get())
            bounded = max(lower, min(upper, value))
            if clamp_current and bounded != value:
                var.set(bounded)
            sync_from_var()
        scale.set_value_range = set_value_range
        return scale

    def _build_area_absolute_control(self, parent, *, result=False):
        control = self._result_area_control if result else self
        row = ctk.CTkFrame(parent, corner_radius=0)
        row.pack(fill="x")
        row.grid_columnconfigure((0, 1), weight=1)
        control._absolute_entries = []
        control._absolute_spins = []
        control._absolute_slider_cap = 1000 if result else 100000
        control._absolute_zoomed = False
        control._absolute_dragging = False
        variables = (control.area_absolute_min_var, control.area_absolute_max_var)
        validate = (self.register(lambda value: value == "" or value.isascii() and value.isdecimal()), "%P")
        for index, label in enumerate(("Minimum area", "Maximum area")):
            side = "w" if index == 0 else "e"
            ctk.CTkLabel(row, text=label).grid(row=0, column=index, sticky=side)
            entry = tk.StringVar(value=str(int(variables[index].get())))
            spin = dark_spinbox(
                row, from_=0, to=sys.maxsize, increment=1, width=12,
                textvariable=entry, validate="key", validatecommand=validate,
                command=lambda i=index: self._commit_absolute_entry(i, result=result),
                **_SPIN_CFG,
            )
            spin.grid(row=1, column=index, sticky=side)
            for event in ("<Return>", "<KP_Enter>", "<FocusOut>"):
                spin.bind(event, lambda _e, i=index: self._commit_absolute_entry(i, result=result))
            self._register_numeric_entry(spin, lambda i=index: self._commit_absolute_entry(i, result=result))
            control._absolute_entries.append(entry)
            control._absolute_spins.append(spin)
        control._absolute_canvas = tk.Canvas(parent, height=52, bg=_SPIN_CFG["bg"], highlightthickness=0)
        control._absolute_canvas.pack(fill="x", pady=(2, 0))
        control._absolute_canvas.bind("<Configure>", lambda _e: self._draw_absolute_slider(result=result))
        control._absolute_canvas.bind(
            "<Motion>", lambda event: self._update_absolute_slider_cursor(event, result=result),
        )
        control._absolute_canvas.bind("<Button-1>", lambda _e: self._start_absolute_drag(_e, result=result))
        control._absolute_canvas.bind("<Double-Button-1>", lambda _e: self._reset_absolute_handle_to_default(_e, result=result))
        control._absolute_canvas.bind("<B1-Motion>", lambda _e: self._drag_absolute_handle(_e, result=result))
        control._absolute_canvas.bind("<ButtonRelease-1>", lambda _e: self._end_absolute_drag(_e, result=result))
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            control._absolute_canvas.bind(event, lambda _e: self._zoom_absolute_slider(_e, result=result))
        for var in variables:
            var.trace_add("write", lambda *_: self._sync_absolute_control(result=result))
        self._sync_absolute_control(result=result)

    def _commit_absolute_entry(self, index, *, result=False):
        control = self._result_area_control if result else self
        text = control._absolute_entries[index].get()
        if text and text.isascii() and text.isdecimal():
            if result:
                control._absolute_defaults_pending = False
            variables = (control.area_absolute_min_var, control.area_absolute_max_var)
            if int(text) != int(variables[index].get()):
                self._set_absolute_threshold(index, int(text), result=result)
            control._absolute_slider_cap = max(control._absolute_slider_cap, int(variables[index].get()))
            self._draw_absolute_slider(result=result)
        else:
            self._sync_absolute_control(result=result)

    def _set_absolute_threshold(self, index, value, *, result=False):
        control = self._result_area_control if result else self
        variables = (control.area_absolute_min_var, control.area_absolute_max_var)
        other = int(variables[1 - index].get())
        value = max(0, int(value))
        value = min(value, other) if index == 0 else max(value, other)
        control._absolute_defaults_pending = False
        variables[index].set(value)
        self._sync_absolute_control(result=result)

    def _sync_absolute_control(self, *, force=False, result=False):
        control = self._result_area_control if result else self
        if getattr(self, "_updating_area_ranges", False) and not force:
            return
        values = (control.area_absolute_min_var.get(), control.area_absolute_max_var.get())
        for entry, value in zip(control._absolute_entries, values):
            entry.set(str(int(value)))
        self._draw_absolute_slider(result=result)

    def _absolute_handle_positions(self, *, result=False):
        control = self._result_area_control if result else self
        width = max(1, control._absolute_canvas.winfo_width() - 24)
        return [12 + min(1.0, max(0.0, float(var.get()) / control._absolute_slider_cap)) * width
                for var in (control.area_absolute_min_var, control.area_absolute_max_var)]

    def _update_absolute_slider_cursor(self, event, *, result=False):
        control = self._result_area_control if result else self
        over_handle = any(
            (event.x - x) ** 2 + (event.y - 18) ** 2 <= 64
            for x in self._absolute_handle_positions(result=result)
        )
        control._absolute_canvas.configure(cursor="hand2" if over_handle else "")

    def _draw_absolute_slider(self, *, result=False):
        control = self._result_area_control if result else self
        canvas = control._absolute_canvas
        canvas.delete("all")
        left, right = self._absolute_handle_positions(result=result)
        canvas.create_line(12, 18, max(12, canvas.winfo_width() - 12), 18,
                           fill="#4a4d50", width=5)
        canvas.create_line(left, 18, right, 18, fill="#aab0b5", width=5)
        canvas.create_text(12, 43, text="0", anchor="w", fill=_SPIN_CFG["fg"])
        canvas.create_text(max(12, canvas.winfo_width() - 12), 43,
                           text=str(control._absolute_slider_cap), anchor="e", fill=_SPIN_CFG["fg"])
        # Off-scale handles stay at the right edge; zooming never changes thresholds.
        for position in (left, right):
            canvas.create_oval(position - 8, 10, position + 8, 26, fill="#115e37", outline="")

    def _reset_absolute_handle_to_default(self, event, *, result=False):
        control = self._result_area_control if result else self
        defaults = control._absolute_default_values
        if defaults is None:
            return "break"
        left, right = self._absolute_handle_positions(result=result)
        hit_min = (event.x - left) ** 2 + (event.y - 18) ** 2 <= 81
        hit_max = (event.x - right) ** 2 + (event.y - 18) ** 2 <= 81
        if hit_min and hit_max:
            if abs(left - right) >= 1:
                index = 0 if abs(event.x - left) < abs(event.x - right) else 1
            else:
                index = 0 if event.y < 18 else 1
        elif hit_min:
            index = 0
        elif hit_max:
            index = 1
        else:
            return None
        self._set_absolute_threshold(index, defaults[index], result=result)
        return "break"

    def _start_absolute_drag(self, event, *, result=False):
        control = self._result_area_control if result else self
        control._absolute_dragging = True
        left, right = self._absolute_handle_positions(result=result)
        if abs(left - right) < 1:
            control._absolute_active_handle = None
            control._absolute_drag_start_x = event.x
            # Do not quantize a coincident handle's value on button press.
            # The first actual movement decides which endpoint is being moved.
            return
        else:
            control._absolute_active_handle = 0 if abs(event.x - left) <= abs(event.x - right) else 1
        self._drag_absolute_handle(event, result=result)

    def _end_absolute_drag(self, _event=None, *, result=False):
        control = self._result_area_control if result else self
        control._absolute_dragging = False

    def _drag_absolute_handle(self, event, *, result=False):
        control = self._result_area_control if result else self
        width = max(1, control._absolute_canvas.winfo_width() - 24)
        fraction = max(0.0, min(1.0, (event.x - 12) / width))
        value = int(math.floor(fraction * control._absolute_slider_cap + 0.5))
        if control._absolute_active_handle is None:
            if event.x < control._absolute_drag_start_x:
                control._absolute_active_handle = 0
            elif event.x > control._absolute_drag_start_x:
                control._absolute_active_handle = 1
            else:
                return
        self._set_absolute_threshold(control._absolute_active_handle, value, result=result)

    def _zoom_absolute_slider(self, event, *, result=False):
        control = self._result_area_control if result else self
        number = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if number == 4 or delta > 0:
            cap = max(1, int(control._absolute_slider_cap / 1.2))
        elif number == 5 or delta < 0:
            cap = max(control._absolute_slider_cap + 1, math.ceil(control._absolute_slider_cap * 1.2))
        else:
            return "break"
        control._absolute_zoomed = True
        control._absolute_slider_cap = cap
        self._draw_absolute_slider(result=result)
        return "break"

    def _update_area_control_ranges(self):
        if self.reader is None or getattr(self, "_updating_area_ranges", False):
            return
        if self._roi_drag_mode is not None:
            # Every mouse position is a new segmentation signature, so recomputing
            # the reference here would re-segment frame zero on each motion event.
            # The ranges are refreshed once the drag ends.
            return
        self._updating_area_ranges = True
        try:
            key = (id(self.reader), self._segmentation_signature())
            reference_changed = key != getattr(self, "_area_reference_key", None)
            if reference_changed:
                # Frame zero leaves the reader's small frame cache as soon as the
                # user scrubs, and re-reading it seeks the shared capture away from
                # the displayed frame on every parameter change. Keep it decoded.
                if self._area_reference_frame is None:
                    self._area_reference_frame = self.reader.read_bgr(0)
                _mask, blobs = self._segment_frame(self._area_reference_frame, 0)
                self._area_reference_blobs = blobs
                values = [b.area for b in blobs if not b.manual_outlier]
                self._area_reference_max = max(values, default=0.0)
                self._area_reference_stats = self._compute_iqr_stats(blobs)["area"]
                self._area_reference_key = key
                if values:
                    q1, _median, q3, iqr = self._normalized_area_stat(self._area_reference_stats)
                    lo = max(0, int(math.floor((q1 - 1.5 * iqr) / 10 + 0.5)) * 10)
                    hi = max(lo, int(math.floor((q3 + 1.5 * iqr) / 10 + 0.5)) * 10)
                    self._absolute_default_values = (lo, hi)
                    if self._absolute_defaults_pending and not self._applying_config:
                        self.area_absolute_min_var.set(lo)
                        self.area_absolute_max_var.set(hi)
                else:
                    self._absolute_default_values = None
            _q1, median, _q3, _iqr = self._normalized_area_stat(self._area_reference_stats)
            area_cap = max(1, math.ceil(max(median * 2, self.area_absolute_max_var.get() * 1.2)))
            if self._min_area_slider_cap is None:
                # Fixed on the first reference for this video: the ROI changes which
                # blobs frame zero contains, and the slider must not rescale under
                # the user while they adjust the ROI.
                self._min_area_slider_cap = area_cap
            self.min_area_scale.set_value_range(
                1, MIN_AREA_ENTRY_MAX, clamp_current=False, slider_upper=self._min_area_slider_cap,
            )
            if reference_changed and not self._absolute_zoomed and not self._absolute_dragging:
                self._absolute_slider_cap = area_cap
            self._sync_absolute_control(force=True)
            self._update_result_area_control_ranges()
        finally:
            self._updating_area_ranges = False

    def _build_area_iqr_control(self, parent: ctk.CTkFrame):
        # Display and edit the saved IQR multipliers directly above the boxplot.
        self.area_iqr_bounds_row = ctk.CTkFrame(parent, corner_radius=0)
        self.area_iqr_bounds_row.grid_columnconfigure((0, 1), weight=1)
        self._iqr_bound_entries = []
        self._iqr_bound_spins = []
        validate = (
            self.register(lambda text: bool(re.fullmatch(r"-?\d*(?:\.\d*)?", text))),
            "%P",
        )
        for index, label in enumerate(("Min (×IQR)", "Max (×IQR)")):
            side = "w" if index == 0 else "e"
            ctk.CTkLabel(self.area_iqr_bounds_row, text=label).grid(
                row=0, column=index, sticky=side,
            )
            entry_var = tk.StringVar()
            spin = dark_spinbox(
                self.area_iqr_bounds_row,
                from_=-1.0, to=AREA_IQR_MULTIPLIER_MAX, increment=0.1,
                width=10, format="%.1f",
                textvariable=entry_var, validate="key", validatecommand=validate,
                command=lambda i=index: self._commit_iqr_area_entry(i),
                **_SPIN_CFG,
            )
            spin.grid(row=1, column=index, sticky=side)
            for event in ("<Return>", "<KP_Enter>", "<FocusOut>"):
                spin.bind(event, lambda _event, i=index: self._commit_iqr_area_entry(i))
            self._register_numeric_entry(
                spin, lambda i=index: self._commit_iqr_area_entry(i),
            )
            self._iqr_bound_entries.append(entry_var)
            self._iqr_bound_spins.append(spin)

        self._iqr_plot_axis = None
        self._iqr_drag_index = None
        self._iqr_drag_start_x = None
        self.area_iqr_boxplot = tk.Canvas(
            parent, height=38, bg="#242424",
            highlightthickness=1, highlightbackground="#565B5E",
        )
        self.area_iqr_boxplot.bind("<Configure>", lambda _e: self._redraw_area_iqr_boxplot())
        self.area_iqr_boxplot.bind("<Motion>", self._update_iqr_boxplot_cursor)
        self.area_iqr_boxplot.bind("<Button-1>", self._start_iqr_boxplot_drag)
        self.area_iqr_boxplot.bind("<Double-Button-1>", self._reset_iqr_bracket_on_double_click)
        self.area_iqr_boxplot.bind("<B1-Motion>", self._drag_iqr_boxplot_handle)
        self.area_iqr_boxplot.bind("<ButtonRelease-1>", self._end_iqr_boxplot_drag)

    def _sync_iqr_area_entries(self, *, force=False):
        if not self.analysis_iqr_stats or "area" not in self.analysis_iqr_stats:
            return
        multipliers = (self.area_iqr_min_var.get(), self.area_iqr_max_var.get())
        for spin, entry, multiplier in zip(
            self._iqr_bound_spins, self._iqr_bound_entries, multipliers
        ):
            if force or self.focus_get() is not spin:
                value = f"{multiplier:.1f}"
                if entry.get() != value:
                    entry.set(value)
            spin.configure(
                state="normal" if self.analysis_iqr_stats["area"][3] > 1e-9
                else "disabled"
            )

    def _commit_iqr_area_entry(self, index: int):
        if not self.analysis_iqr_stats or "area" not in self.analysis_iqr_stats:
            return
        entry = self._iqr_bound_entries[index]
        try:
            value = float(entry.get().strip())
            if not math.isfinite(value):
                raise ValueError("IQR multiplier must be finite.")
        except ValueError:
            self._sync_iqr_area_entries(force=True)
            return
        variable = (self.area_iqr_min_var, self.area_iqr_max_var)[index]
        # An unchanged, rounded display must not alter an older saved setting.
        if entry.get().strip() != f"{variable.get():.1f}":
            self._set_iqr_multiplier(index, value)
        self._sync_iqr_area_entries(force=True)

    def _set_iqr_multiplier(self, index: int, multiplier: float):
        multiplier = round(max(-1.0, min(AREA_IQR_MULTIPLIER_MAX, multiplier)), 1)
        variable = (self.area_iqr_min_var, self.area_iqr_max_var)[index]
        if not math.isclose(variable.get(), multiplier, rel_tol=0.0, abs_tol=1e-12):
            variable.set(multiplier)

    def _set_iqr_area_bound(self, index: int, area: float):
        if not self.analysis_iqr_stats or "area" not in self.analysis_iqr_stats:
            return
        q1, median, q3, iqr = self._normalized_area_stat(
            self.analysis_iqr_stats["area"]
        )
        if iqr <= 1e-9:
            return  # An IQR multiplier cannot move a bound when IQR is zero.
        if index == 0:
            area = max(q1 - AREA_IQR_MULTIPLIER_MAX * iqr, min(median, area))
            multiplier = (q1 - area) / iqr
        else:
            area = min(q3 + AREA_IQR_MULTIPLIER_MAX * iqr, max(median, area))
            multiplier = (area - q3) / iqr
        self._set_iqr_multiplier(index, multiplier)

    def _iqr_bracket_hit(self, event):
        axis = self._iqr_plot_axis
        if axis is None:
            return None
        _vmin, _vmax, _left, _right, x_min, x_max, top, bottom = axis
        if not top - 8 <= event.y <= bottom + 8:
            return None
        hit_min = abs(event.x - x_min) <= 10
        hit_max = abs(event.x - x_max) <= 10
        if hit_min and hit_max:
            if abs(x_max - x_min) < 1:
                return -1  # Resolve coincident handles by drag direction.
            return 0 if abs(event.x - x_min) <= abs(event.x - x_max) else 1
        if hit_min:
            return 0
        if hit_max:
            return 1
        return None

    def _update_iqr_boxplot_cursor(self, event):
        stats = self.analysis_iqr_stats.get("area")
        active = bool(stats and stats[3] > 1e-9 and self._iqr_bracket_hit(event) is not None)
        self.area_iqr_boxplot.configure(cursor="hand2" if active else "")

    def _start_iqr_boxplot_drag(self, event):
        if not self.analysis_iqr_stats or self.analysis_iqr_stats["area"][3] <= 1e-9:
            return
        self._iqr_drag_index = self._iqr_bracket_hit(event)
        self._iqr_drag_start_x = event.x
        if self._iqr_drag_index is not None:
            self.area_iqr_boxplot.focus_set()

    def _reset_iqr_bracket_on_double_click(self, event):
        """Reset only the clicked red bracket to 1.5×IQR."""
        index = self._iqr_bracket_hit(event)
        if index is None:
            return
        self._end_iqr_boxplot_drag()
        if index == -1:
            self._set_iqr_multiplier(0, 1.5)
            self._set_iqr_multiplier(1, 1.5)
        else:
            self._set_iqr_multiplier(index, 1.5)
        self._sync_iqr_area_entries(force=True)
        self._redraw_area_iqr_boxplot()
        return "break"

    def _drag_iqr_boxplot_handle(self, event):
        if self._iqr_drag_index is None or self._iqr_plot_axis is None:
            return
        if self._iqr_drag_index == -1:
            if event.x == self._iqr_drag_start_x:
                return
            self._iqr_drag_index = 0 if event.x < self._iqr_drag_start_x else 1
        vmin, vmax, left, right, _x_min, _x_max, _top, _bottom = self._iqr_plot_axis
        fraction = max(0.0, min(1.0, (event.x - left) / max(1, right - left)))
        area = vmin + fraction * (vmax - vmin)
        self._set_iqr_area_bound(self._iqr_drag_index, area)

    def _end_iqr_boxplot_drag(self, _event=None):
        self._iqr_drag_index = None
        self._iqr_drag_start_x = None

    def _toggle_area_outlier_method_controls(self):
        if self.area_iqr_frame is None or self.area_absolute_frame is None:
            return
        method = self.area_outlier_method_var.get()
        if method not in {"iqr", "absolute"}:
            method = "iqr"
            self.area_outlier_method_var.set(method)
        # Repack controls in their original order when the method changes.
        for frame in (self.analysis_run_frame, self.area_iqr_frame, self.area_absolute_frame):
            if frame is not None:
                frame.pack_forget()
        if method == "absolute":
            if self.area_iqr_boxplot is not None:
                self.area_iqr_boxplot.pack_forget()
            self.area_iqr_bounds_row.pack_forget()
            self.area_absolute_frame.pack(fill="x")
            self.analysis_ready_var.set("analysis: not required (absolute bounds)")
            self._update_area_control_ranges()
        else:
            if self.analysis_run_frame is not None:
                self.analysis_run_frame.pack(fill="x")
            self.area_iqr_frame.pack(fill="x")
            if not self.analysis_iqr_stats:
                self.analysis_ready_var.set("analysis: not computed")
            self._schedule_area_iqr_boxplot_redraw()

    def _on_area_outlier_method_changed(self):
        self._toggle_area_outlier_method_controls()
        if self._applying_config:
            return
        self._refresh_bounds_from_fixed_stats(show_error=False)

    def _redraw_area_iqr_boxplot(self):
        self._update_area_control_ranges()
        canvas = getattr(self, "area_iqr_boxplot", None)
        if canvas is None:
            return
        if (self.area_outlier_method_var.get() != "iqr"
                or not self.analysis_iqr_stats or "area" not in self.analysis_iqr_stats):
            self._iqr_plot_axis = None
            self._end_iqr_boxplot_drag()
            canvas.pack_forget()
            self.area_iqr_bounds_row.pack_forget()
            return
        if not self.area_iqr_bounds_row.winfo_manager():
            self.area_iqr_bounds_row.pack(fill="x", padx=6, pady=(2, 0))
        if not canvas.winfo_manager():
            canvas.pack(fill="x", padx=6, pady=(2, 6))
        self._sync_iqr_area_entries()

        canvas.delete("all")
        q1, median, q3, iqr = self._normalized_area_stat(self.analysis_iqr_stats["area"])
        lo, hi = self._bounds_from_fixed_stats(self.analysis_iqr_stats).get("area", (q1, q3))

        # A symmetric axis keeps the median exactly centered, including skewed data.
        multiplier = AREA_IQR_MULTIPLIER_MAX
        span = max(median - (q1 - multiplier * iqr),
                   (q3 + multiplier * iqr) - median, 1.0, abs(median) * 0.1)
        vmin, vmax = median - span, median + span

        w = max(1, canvas.winfo_width())
        h = max(1, canvas.winfo_height())
        left = 14
        right = max(left + 1, w - 14)
        y = h // 2

        def x_of(value: float) -> int:
            ratio = (float(value) - vmin) / max(1e-9, (vmax - vmin))
            return int(round(left + max(0.0, min(1.0, ratio)) * (right - left)))

        x_lo, x_q1, x_med, x_q3, x_hi = [x_of(v) for v in (lo, q1, median, q3, hi)]
        x_ref_min, x_ref_max = x_of(vmin), x_of(vmax)

        # Only the cut brackets move when the multipliers change.
        box_half = min(16, max(8, (h - 14) // 2))
        box_top = y - box_half
        box_bottom = y + box_half
        canvas.create_line(x_ref_min, y, x_ref_max, y, fill="#888888", width=1)
        canvas.create_rectangle(x_q1, box_top, x_q3, box_bottom, outline="#AAB0B5", fill="#6B7280")
        canvas.create_line(x_med, box_top, x_med, box_bottom, fill="#DCE4EE", width=2)

        cut_top = box_top
        cut_bottom = box_bottom
        self._iqr_plot_axis = (vmin, vmax, left, right, x_lo, x_hi, cut_top, cut_bottom)
        hook_len = 4
        # Lower cut: bracket opens inward to the right. Upper cut: bracket opens inward to the left.
        canvas.create_line(x_lo, cut_top, x_lo, cut_bottom, fill="#FF7777", width=2)
        canvas.create_line(x_lo, cut_top, x_lo + hook_len, cut_top, fill="#FF7777", width=2)
        canvas.create_line(x_lo, cut_bottom, x_lo + hook_len, cut_bottom, fill="#FF7777", width=2)

        canvas.create_line(x_hi, cut_top, x_hi, cut_bottom, fill="#FF7777", width=2)
        canvas.create_line(x_hi, cut_top, x_hi - hook_len, cut_top, fill="#FF7777", width=2)
        canvas.create_line(x_hi, cut_bottom, x_hi - hook_len, cut_bottom, fill="#FF7777", width=2)

    def _schedule_area_iqr_boxplot_redraw(self):
        """Redraw the Area IQR boxplot after Tk has completed layout updates.

        Load config changes section visibility and may also load a video/background.
        A direct redraw can run before the Analysis section and Canvas have a real width,
        so the boxplot appears missing. Scheduling the redraw after idle, plus a short
        delayed redraw, makes the restored boxplot deterministic.
        """
        if self.area_outlier_method_var.get() != "iqr":
            if self.area_iqr_boxplot is not None:
                self.area_iqr_boxplot.pack_forget()
            self.area_iqr_bounds_row.pack_forget()
            return
        if not self.analysis_iqr_stats or "area" not in self.analysis_iqr_stats:
            return

        def redraw_after_layout():
            try:
                self.update_idletasks()
            except Exception:
                pass
            self._redraw_area_iqr_boxplot()

        try:
            self.after_idle(redraw_after_layout)
            self.after(80, redraw_after_layout)
        except Exception:
            self._redraw_area_iqr_boxplot()

    def _bind_events(self):
        self.bind("<Configure>", self.on_window_configure)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        for key in ("Left", "Right"):
            self.bind_all(f"<KeyPress-{key}>", self._on_arrow_key_press, add="+")
            self.bind_all(f"<KeyRelease-{key}>", self._on_arrow_key_release, add="+")
        self.bind("<FocusOut>", self._stop_arrow_on_focus_out, add="+")
        self.bind_all("<KeyPress-space>", self._on_space_press, add="+")
        self.bind_all("<KeyRelease-space>", self._on_space_release, add="+")
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<Double-Button-1>", lambda event: self.on_canvas_speed_click(event, 2.0))
        self.canvas.bind("<Triple-Button-1>", lambda event: self.on_canvas_speed_click(event, 4.0))
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Leave>", self.on_canvas_leave)

    def _log_dnd(self, message: str) -> None:
        """Write drag-and-drop diagnostics without affecting the GUI workflow."""
        entry = f"[DND] {message}"
        self._dnd_diagnostics.append(entry)
        print(entry, flush=True)

    def _initialize_video_drag_and_drop(self) -> None:
        """Enable native file drops on the video canvas when tkdnd is available."""
        self._dnd_enabled = False
        self._dnd_canvas_registered = False
        self._dnd_diagnostics: list[str] = []
        try:
            tkinterdnd2_version = importlib.metadata.version("tkinterdnd2")
        except importlib.metadata.PackageNotFoundError:
            tkinterdnd2_version = "unavailable"
        except Exception as exc:
            tkinterdnd2_version = f"unavailable ({exc})"

        try:
            tcl_version = str(self.tk.call("info", "patchlevel"))
        except Exception as exc:
            tcl_version = f"unavailable ({exc})"
        try:
            tk_version = str(self.tk.call("package", "provide", "Tk"))
        except Exception as exc:
            tk_version = f"unavailable ({exc})"
        self._log_dnd(
            "startup "
            f"tkinterdnd2={tkinterdnd2_version} "
            f"platform={platform.platform()} machine={platform.machine()} "
            f"tcl={tcl_version} tk={tk_version}"
        )

        if TkinterDnD is None:
            self._log_dnd(f"initialization failed: {_TKINTERDND2_IMPORT_ERROR}")
            self._log_dnd("canvas registered=False")
            return

        try:
            self._add_tkdnd_package_path()
            tkdnd_version = TkinterDnD.require(self)
            self._log_dnd(f"tkdnd={tkdnd_version}")
            self.canvas.drop_target_register(DND_FILES)
            self._dnd_canvas_registered = True
            self._log_dnd("canvas registered=True")
            self.canvas.dnd_bind("<<DropEnter>>", self._on_video_drop_enter)
            self.canvas.dnd_bind("<<DropLeave>>", self._on_video_drop_leave)
            self.canvas.dnd_bind("<<Drop>>", self._on_video_drop)
            self._dnd_enabled = True
            self._log_dnd("initialization succeeded")
        except Exception as exc:
            self._log_dnd(f"initialization failed: {exc}")
            self._log_dnd(f"canvas registered={self._dnd_canvas_registered}")

    def _add_tkdnd_package_path(self) -> None:
        """Expose the package's native directory using Tcl-safe path separators."""
        module_path = getattr(TkinterDnD, "__file__", None)
        if not module_path:
            return
        system = platform.system()
        machine = platform.machine().lower()
        if system == "Windows":
            process_machine = os.environ.get("PROCESSOR_ARCHITECTURE", platform.machine()).upper()
            platform_name = {
                "AMD64": "win-x64",
                "ARM64": "win-arm64",
                "X86": "win-x86",
            }.get(process_machine)
        elif system == "Darwin":
            platform_name = {"arm64": "osx-arm64", "x86_64": "osx-x64"}.get(machine)
        elif system == "Linux":
            platform_name = {"aarch64": "linux-arm64", "x86_64": "linux-x64"}.get(machine)
        else:
            platform_name = None
        if platform_name is None:
            return

        tkdnd_root = Path(module_path).parent / "tkdnd"
        native_path = tkdnd_root / platform_name
        try:
            if int(str(self.tk.call("info", "tclversion")).split(".", 1)[0]) >= 9:
                tcl9_path = tkdnd_root / f"{platform_name}-tcl9"
                if tcl9_path.is_dir():
                    native_path = tcl9_path
        except Exception:
            pass
        if native_path.is_dir():
            tcl_path = native_path.as_posix()
            if system == "Windows" and len(tcl_path) >= 2 and tcl_path[1] == ":":
                tcl_path = f"//?/{tcl_path}"
            self.tk.call("lappend", "auto_path", tcl_path)
            self._log_dnd(f"tkdnd package path={tcl_path}")

    def _on_video_drop_enter(self, _event):
        return COPY

    def _on_video_drop_leave(self, _event):
        return None

    @staticmethod
    def _parse_video_drop_paths(tk_root, raw_data: str) -> tuple[str, ...]:
        """Parse the Tcl list emitted by tkdnd without losing spaces or Unicode."""
        return tuple(str(path) for path in tk_root.splitlist(raw_data))

    @staticmethod
    def _accepted_video_drop_paths(paths) -> tuple[str, ...]:
        accepted = []
        for path in paths:
            normalized = os.path.normpath(str(path))
            if os.path.isfile(normalized) and os.path.splitext(normalized)[1].lower() in VIDEO_DROP_SUFFIXES:
                accepted.append(normalized)
        return tuple(accepted)

    def _reject_video_drop(self, message: str) -> str:
        self._log_dnd(f"rejected: {message}")
        self.set_status(message, auto_clear=False)
        return REFUSE_DROP

    def _on_video_drop(self, event):
        raw_data = str(getattr(event, "data", ""))
        self._log_dnd(f"drop raw={raw_data!r}")
        try:
            parsed_paths = self._parse_video_drop_paths(self.tk, raw_data)
        except Exception as exc:
            self._log_dnd(f"drop parse failed: {exc}")
            return self._reject_video_drop("Unable to read the dropped file list.")
        self._log_dnd(f"drop parsed={parsed_paths!r}")

        accepted_paths = self._accepted_video_drop_paths(parsed_paths)
        self._log_dnd(f"drop accepted={accepted_paths!r}")
        if len(accepted_paths) > 1:
            return self._reject_video_drop("Please drop one video at a time.")
        if not accepted_paths:
            return self._reject_video_drop("Please drop one video file.")

        selected_path = accepted_paths[0]
        self.after_idle(lambda path=selected_path: self._handle_selected_video(path))
        return COPY

    def _bind_traces(self):
        self.video_path_var.trace_add("write", lambda *_: self._refresh_guidance_ui())
        self.segmentation_mode_var.trace_add("write", lambda *_: self._on_segmentation_mode_changed())
        self.bg_method_var.trace_add("write", lambda *_: self._on_background_method_changed())
        self.roi_enabled_var.trace_add("write", lambda *_: self._toggle_roi_params())
        self.additional_outlier_enabled_var.trace_add("write", lambda *_: self._toggle_additional_outlier_params())
        self.additional_outlier_segmentation_mode_var.trace_add("write", lambda *_: self._refresh_additional_outlier_controls())
        seg_vars = [
            self.threshold_var, self.dark_threshold_var, self.bright_threshold_var, self.diff_threshold_var, self.min_area_var, self.blur_ksize_var,
            self.open_iter_var, self.close_iter_var, self.fill_holes_var, self.invert_mask_var,
            self.additional_outlier_enabled_var, self.additional_outlier_frame_start_var, self.additional_outlier_frame_end_var,
            self.region_expand_px_var, self.region_expand_merge_only_var,
        ]
        for var in seg_vars:
            var.trace_add("write", lambda *_: self._mark_analysis_stale(redraw=True))
        for var in [self.roi_enabled_var, self.roi_frame_start_var, self.roi_frame_end_var]:
            var.trace_add("write", self._on_roi_var_changed)
        self.roi_reverse_var.trace_add("write", self._on_roi_reverse_changed)
        for var in [
            self.additional_outlier_enabled_var,
            self.additional_outlier_frame_start_var,
            self.additional_outlier_frame_end_var,
            self.additional_outlier_segmentation_mode_var,
            self.additional_outlier_bg_method_var,
            self.additional_outlier_threshold_var,
            self.additional_outlier_dark_threshold_var,
            self.additional_outlier_bright_threshold_var,
            self.additional_outlier_diff_threshold_var,
            self.additional_outlier_min_area_var,
        ]:
            var.trace_add("write", self._on_additional_outlier_var_changed)
        self.result_import_enabled_var.trace_add("write", lambda *_: self._toggle_result_import_params())
        for var in [
            self.result_import_enabled_var,
            self.result_import_min_coverage_var,
        ]:
            var.trace_add("write", lambda *_: self._on_result_import_changed())
        for index, var in enumerate((self.result_import_area_min_var, self.result_import_area_max_var)):
            var.trace_add("write", lambda *_, i=index: self._on_result_area_changed(i))
        self.result_import_show_obb_var.trace_add("write", lambda *_: self.redraw_current_frame())
        self.area_outlier_method_var.trace_add("write", lambda *_: self._on_area_outlier_method_changed())
        for var in [self.area_iqr_min_var, self.area_iqr_max_var, self.area_absolute_min_var, self.area_absolute_max_var]:
            var.trace_add("write", lambda *_: self._refresh_bounds_from_fixed_stats(show_error=False))
        for var in [self.show_overlay_var, self.show_centers_var, self.show_contours_var]:
            var.trace_add("write", lambda *_: self.redraw_current_frame())
        # Smoothing is post-classification: neither raw-mask caches nor the
        # area-analysis statistics depend on it. Always redraw from raw blobs.
        self.single_blob_smoothing_enabled_var.trace_add(
            "write", lambda *_: self._toggle_single_blob_smoothing_params())
        for var in [self.single_blob_smoothing_enabled_var, self.single_blob_smoothing_level_var]:
            var.trace_add("write", lambda *_: self.redraw_current_frame())

    def set_status(self, text: str, auto_clear: bool = True):
        if self.status_clear_job is not None:
            self.after_cancel(self.status_clear_job)
            self.status_clear_job = None
        self.status_var.set(text)
        if auto_clear and text:
            self.status_clear_job = self.after(5000, self.clear_status)

    def clear_status(self):
        self.status_clear_job = None
        self.status_var.set("")

    def _handle_selected_video(self, path: str) -> None:
        """Check a newly selected video, then load it or its analysis copy.

        Only this path -- browsing and dropping -- runs the compatibility check.
        Loading a video named by a saved config goes straight to load_video, so
        reopening a session never re-asks about a conversion already decided.
        """
        prepared = prepare_analysis_video(self, path, log=self.set_status)
        if prepared is None:
            return
        self._handle_prepared_video(prepared)

    def _handle_prepared_video(self, prepared) -> None:
        found = self._find_existing_config_for_video(prepared.path)
        if found:
            config_path, config = found
            try:
                self._load_config_dict(config_path, config)
                return
            except Exception as exc:
                self.set_status(f"Failed to load existing config ({config_path}): {exc}")
        self.config_path = None
        self.load_video(prepared.path)

    def browse_video(self):
        prepared = ask_open_analysis_video(self, title="Select video", log=self.set_status)
        if prepared is not None:
            self._handle_prepared_video(prepared)

    def load_video(self, path: str):
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Video file not found.")
            return
        try:
            reader = VideoFrameReader(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self._loading_video = True
        self.video_path_var.set(path)
        if self.background_rebuild_job is not None:
            try:
                self.after_cancel(self.background_rebuild_job)
            except Exception:
                pass
            self.background_rebuild_job = None
        self._arrow_key_play.stop()
        self.stop_playback(update_button=False)
        self._requested_arrow_frame = None
        self._space_key_down = False
        if self._space_release_job is not None:
            self.after_cancel(self._space_release_job)
            self._space_release_job = None
        self.playback_speed = 1.0
        self._update_play_button()
        if self._smooth_player is not None:
            self._smooth_player.close()
            self._smooth_player = None
        if self.reader is not None:
            self.reader.close()
        self.reader = reader
        self._canvas_image_id = None
        self.tk_image = None
        self.canvas.delete("all")
        self._preview_pacer.reset()
        self._area_reference_frame = None
        self._area_reference_blobs = []
        self._area_reference_key = None
        self._min_area_slider_cap = None
        self._absolute_defaults_pending = not self._applying_config
        self._absolute_zoomed = False
        self._absolute_dragging = False
        self.frame_count = reader.frame_count
        # Reinitialize to the downstream DIR_MIN_SEC=0.5 s equivalent, in
        # integer frames. Loading an explicit config overrides this value.
        fps = float(reader.fps) if reader.fps and reader.fps > 0 else 30.0
        self.static_frame_window_var.set(max(2, int(round(0.5 * fps))))
        self._static_search_signature = None
        self._static_exclusions = {}
        self._static_digests = None
        self.static_search_status_var.set(
            "Not searched — press Search before Processing"
            if self.static_exclusion_enabled_var.get() else "Disabled"
        )
        self._reset_temporal_search()
        self.current_frame = 0
        self.frame_var.set(0)
        self.frame_spin.configure(from_=0, to=max(0, self.frame_count - 1))
        self.frame_scale.configure(from_=0, to=max(0, self.frame_count - 1))
        self._set_frame_range_vars(self._default_frame_ranges(self.frame_count))
        self._update_geometry_control_ranges()
        if self.frame_ranges_button is not None:
            self.frame_ranges_button.configure(state="normal")

        self.background_bgr = None
        self.active_background_method = None
        self.current_mask_cache.clear()
        self.analysis_blobs_by_frame.clear()
        self.analysis_outlier_frames.clear()
        self.analysis_frames = []
        self.analysis_signature = None
        self.analysis_bounds = {}
        self.analysis_iqr_stats = {}
        self.analysis_is_stale = True
        self.outlier_section_revealed = False
        self._guidance_stage = "background"
        self.background_ready_var.set("background: not computed")
        self.analysis_ready_var.set("analysis: not computed")
        if self.bg_preview_button is not None:
            self.bg_preview_button.configure(state="disabled")

        # The imported OBBs describe one specific video; drop them with it.
        self.result_obb_import = None
        self.result_import_path_var.set("")
        self.result_import_status_var.set(self._result_import_status_text())
        self._result_area_updating = True
        try:
            self._result_area_control._absolute_defaults_pending = True
            self._result_area_control._absolute_default_values = None
            self._result_area_control._absolute_zoomed = False
            self._result_area_control._absolute_dragging = False
            self._result_area_reference_key = None
            self._result_area_reference_stats = (0.0, 0.0, 0.0, 0.0)
            self.result_import_area_min_var.set(0)
            self.result_import_area_max_var.set(1000)
        finally:
            self._result_area_updating = False
        self._sync_absolute_control(result=True)
        self._refresh_result_match()

        self._reset_view_state()
        self.roi_sets = [self._default_roi_set()]
        self.roi_active_set_idx = 0
        if self.roi_set_combo is not None:
            self._update_roi_set_selector_values()
            self._update_roi_set_selector()
        self._set_default_roi(enable=False, emit_status=False)
        self.additional_outlier_sets = [self._default_additional_outlier_set()]
        self.additional_outlier_active_set_idx = 0
        self._update_additional_outlier_set_selector_values()
        self._update_additional_outlier_set_selector()
        self._sync_additional_outlier_set_to_vars()
        self._refresh_additional_outlier_controls()
        self._toggle_additional_outlier_params()
        self.additional_outlier_revision += 1
        self._refresh_segmentation_controls()
        if self.frame_count >= LONG_VIDEO_FRAME_COUNT_THRESHOLD and not self._applying_config:
            self._show_frame_range_dialog(auto=True)
        cache_ready = False
        if not self._applying_config:
            try:
                cache_ready = self._activate_cached_background(self.bg_method_var.get(), redraw=False)
                if cache_ready:
                    self.active_background_method = self.bg_method_var.get()
            except Exception:
                # A damaged cache should never prevent the video itself from loading.
                self.background_bgr = None
                self.active_background_method = None
                cache_ready = False
        if not self._segmentation_needs_background():
            self.background_ready_var.set(f"background: not required ({self.segmentation_mode_var.get()})")
            self._set_workflow_stage("analysis")
        else:
            self._set_workflow_stage("analysis" if self.background_bgr is not None else "background")
        self.set_frame(0, reset_view=True)
        self._loading_video = False
        if self._segmentation_needs_background() and not cache_ready and not self._applying_config:
            # Finish the file-dialog/load callback before starting extraction.
            # Running a nested progress-loop while _loading_video was true made
            # the first build follow a different, re-entrant path from retries.
            self._schedule_background_rebuild(delay_ms=50)
        selected_ready = self._background_cache_is_current(self.bg_method_var.get())
        frame_label = str(self.frame_count)
        if reader.frame_count_adjusted:
            frame_label = f"{self.frame_count} usable (reported {reader.reported_frame_count})"
        self.set_status(
            f"Loaded: {os.path.basename(path)} / frames={frame_label} / "
            f"background={'cached' if cache_ready else ('ready' if selected_ready else 'unavailable')} / "
            f"output={self._default_output_dir()}"
        )
        self.after_idle(lambda loaded=reader: self._prime_preview_decoder(loaded))
        _start_maximized(self)

    def on_close(self):
        if self._closing:
            return
        if self._progress_busy:
            self._progress_close_requested = True
            self._request_progress_cancel()
            return
        self._closing = True
        for glow in self._guidance_glows.values():
            glow.stop()
        if self._space_release_job is not None:
            self.after_cancel(self._space_release_job)
            self._space_release_job = None
        self._space_key_down = False
        self._arrow_key_play.stop()
        self.stop_playback(update_button=False)
        if self.background_rebuild_job is not None:
            try:
                self.after_cancel(self.background_rebuild_job)
            except Exception:
                pass
            self.background_rebuild_job = None
        if self.prefetch_after_id is not None:
            self.after_cancel(self.prefetch_after_id)
        if self.status_clear_job is not None:
            self.after_cancel(self.status_clear_job)
        if self._smooth_player is not None:
            self._smooth_player.close()
            self._smooth_player = None
        if self.reader is not None:
            self.reader.close()
        self._close_progress_popup()
        self._close_bg_preview()
        if self._tooltip_window is not None:
            try:
                self._tooltip_window.destroy()
            except Exception:
                pass
        self.destroy()

    def on_window_configure(self, _event=None):
        if self._closing:
            return
        if self.frame_bgr_cache is not None:
            self._draw_canvas(preserve_view=True)

    def on_frame_spin(self):
        try:
            self.set_frame(int(self.frame_spin.get()))
        except Exception:
            pass

    def on_frame_entry_commit(self, _event=None):
        try:
            self.set_frame(int(self.frame_spin.get()))
        except Exception:
            self.frame_var.set(self.current_frame)
        return "break"

    def on_frame_scale(self, value):
        if self.reader is None:
            return
        self.set_frame(int(round(float(value))))

    @staticmethod
    def _arrow_event_is_editing(event) -> bool:
        widget = event.widget
        try:
            widget_class = widget.winfo_class()
        except (AttributeError, tk.TclError):
            return False
        return isinstance(widget, (tk.Entry, tk.Spinbox, tk.Text)) or widget_class in {
            "Entry", "TEntry", "Spinbox", "TSpinbox", "Text",
        }

    def _on_space_press(self, event):
        # Editing fields retain their normal text/space handling.
        if self._arrow_event_is_editing(event):
            return None
        if self.reader is None or self._closing:
            return None
        if self._space_release_job is not None:
            self.after_cancel(self._space_release_job)
            self._space_release_job = None
        if not self._space_key_down:
            self._space_key_down = True
            self.toggle_playback()
        return "break"

    def _on_space_release(self, event):
        if self._arrow_event_is_editing(event):
            return None
        if self._space_release_job is not None:
            self.after_cancel(self._space_release_job)
        # X11 can synthesize release/press pairs while Space is held.
        self._space_release_job = self.after(35, self._finish_space_release)
        return "break"

    def _finish_space_release(self):
        self._space_release_job = None
        self._space_key_down = False

    def _queue_arrow_step(self, frame_idx: int) -> bool:
        """One-frame navigation must not seek/decode on the Tk event loop."""
        reader = self.reader
        if reader is None or self._closing:
            return False
        index = max(0, min(self.frame_count - 1, int(frame_idx)))
        self._prefetch_cancel.set()
        if self.prefetch_after_id is not None:
            self.after_cancel(self.prefetch_after_id)
            self.prefetch_after_id = None
        try:
            if not self._ensure_smooth_player().request_frame(index):
                return False
        except Exception as exc:
            self.set_status(f"Arrow-key decoder unavailable: {exc}")
            return False
        self._requested_arrow_frame = index
        return True

    def _prime_preview_decoder(self, reader: VideoFrameReader):
        if reader is not self.reader or self._closing:
            return
        try:
            self._ensure_smooth_player()
        except Exception as exc:
            # A failed speculative decoder does not block ordinary navigation.
            self.set_status(f"PyAV preview initialization unavailable: {exc}")

    @staticmethod
    def _windows_arrow_down(direction: int) -> bool:
        """Read actual key state without waiting for queued Tk autorepeats."""
        if sys.platform != "win32":
            return True
        try:
            import ctypes
            code = 0x25 if direction < 0 else 0x27
            return bool(ctypes.windll.user32.GetAsyncKeyState(code) & 0x8000)
        except (AttributeError, OSError):
            return True

    def _held_arrow_is_physically_down(self) -> bool:
        if not self.playback_active or not self._arrow_key_play.repeating:
            return True
        direction = self._arrow_key_play.direction
        timing = self._playback_timing
        if direction not in (-1, 1) or timing is None or timing[2] * direction <= 0:
            return True
        return self._windows_arrow_down(direction)

    def _finish_physically_released_arrow(self):
        if self._arrow_key_play.direction:
            self._arrow_key_play.stop()
        elif self.playback_active:
            self.stop_playback()

    def _on_arrow_key_press(self, event):
        if self._arrow_event_is_editing(event):
            self._arrow_key_play.stop()
            return None
        if self.reader is None or self._closing:
            return None
        delta = -1 if event.keysym == "Left" else 1
        if not self._windows_arrow_down(delta):
            # A delayed native KeyPress must not restart a released key hold.
            if self._arrow_key_play.direction == delta:
                self._arrow_key_play.stop()
            return "break"
        new_press = self._arrow_key_play.direction != delta
        self._arrow_key_play.press(delta)
        if new_press and delta < 0 and self._arrow_key_play.direction == -1:
            try:
                pipeline = self._ensure_smooth_player()
                center = (self._requested_arrow_frame
                          if self._requested_arrow_frame is not None
                          else self.current_frame)
                pipeline.reverse_buffer.prepare(center, self.playback_speed, warm_blocks=2)
            except Exception:
                pass  # Single-frame navigation remains available.
        return "break"

    def _on_arrow_key_release(self, event):
        if self._arrow_event_is_editing(event):
            return None
        direction = -1 if event.keysym == "Left" else 1
        if sys.platform == "linux":
            # X11 key autorepeat may include synthetic KeyRelease events.
            self._arrow_key_play.release(direction)
        else:
            # A physical release ends the held session immediately.
            self._arrow_key_play.stop()
        return "break"

    def _stop_arrow_on_focus_out(self, event):
        if event.widget is self:
            self._arrow_key_play.stop()
            if self._space_release_job is not None:
                self.after_cancel(self._space_release_job)
                self._space_release_job = None
            self._space_key_down = False

    def step_frame(self, delta: int):
        if self.reader is None:
            return
        self.set_frame(self.current_frame + int(delta))

    def fit_canvas_to_window(self):
        if self.frame_bgr_cache is None:
            return
        self._draw_canvas(preserve_view=False)

    def set_frame(self, frame_idx: int, reset_view: bool = False, *, from_playback: bool = False):
        if self.reader is None or self.frame_count <= 0:
            return False
        if self.playback_active and not from_playback:
            self.stop_playback()
        if not self.playback_active and self._smooth_player is not None:
            self._smooth_player.invalidate()
        if not from_playback:
            self._requested_arrow_frame = None
        if self._roi_drag_mode is not None:
            self.on_canvas_release()
        if self.reader.frame_count != self.frame_count:
            self.frame_count = self.reader.frame_count
            self.frame_spin.configure(from_=0, to=max(0, self.frame_count - 1))
            self.frame_scale.configure(from_=0, to=max(0, self.frame_count - 1))
        frame_idx = max(0, min(self.frame_count - 1, int(frame_idx)))
        try:
            # Keep playback gaps sequential to avoid unnecessary long-GOP seeks.
            gap_limit = (max(12, int(self._playback_fps() * max(1.0, self.playback_speed) * 2))
                         if from_playback else 12)
            frame_bgr, actual_idx = self.reader.read_bgr_with_index(
                frame_idx, max_sequential_gap=gap_limit)
        except Exception as exc:
            self.frame_var.set(self.current_frame)
            self.frame_scale.set(self.current_frame)
            self.set_status(f"Frame read failed: {exc}")
            return False
        if self.reader.frame_count != self.frame_count:
            self.frame_count = self.reader.frame_count
            self.frame_spin.configure(from_=0, to=max(0, self.frame_count - 1))
            self.frame_scale.configure(from_=0, to=max(0, self.frame_count - 1))
        return self._present_decoded_frame(frame_bgr, actual_idx, reset_view=reset_view)

    def _present_decoded_frame(self, frame_bgr, frame_idx: int, *, reset_view=False) -> bool:
        if self.reader is None or self._closing:
            return False
        if self.reader.frame_count != self.frame_count:
            self.frame_count = self.reader.frame_count
            self.frame_spin.configure(from_=0, to=max(0, self.frame_count - 1))
            self.frame_scale.configure(from_=0, to=max(0, self.frame_count - 1))
        frame_idx = max(0, min(self.frame_count - 1, int(frame_idx)))
        self.reader._cache_put(frame_idx, frame_bgr)
        self.current_frame = frame_idx
        self._requested_arrow_frame = None
        self.frame_var.set(frame_idx)
        self.frame_scale.set(frame_idx)
        self.frame_bgr_cache = frame_bgr
        self._draw_canvas(preserve_view=not reset_view)
        self._update_analysis_dependent_buttons()
        if not self.playback_active and self._arrow_key_play.direction == 0:
            self._queue_prefetch(frame_idx)
        return True

    def _queue_prefetch(self, center_frame: int):
        reader = self.reader
        if reader is None or self.playback_active or self._arrow_key_play.direction:
            return
        if self.prefetch_after_id is not None:
            self.after_cancel(self.prefetch_after_id)
            self.prefetch_after_id = None
        self._prefetch_cancel.set()
        cancel = threading.Event()
        self._prefetch_cancel = cancel

        def start_when_idle():
            self.prefetch_after_id = None
            if (cancel.is_set() or reader is not self.reader or self.playback_active
                    or self._arrow_key_play.direction):
                return
            # Decode each direction sequentially, rather than alternately
            # seeking forwards and backwards across long-GOP videos.
            budget = max(1, reader.cache_capacity - 1)
            backward = min(PREFETCH_BACKWARD, max(1, budget // 2), center_frame)
            forward = min(PREFETCH_FORWARD, max(0, budget - backward),
                          self.frame_count - center_frame - 1)
            frames = list(range(center_frame - backward, center_frame)) + list(
                range(center_frame + 1, center_frame + forward + 1)
            )

            def worker():
                for fid in frames:
                    if cancel.is_set() or reader is not self.reader:
                        return
                    reader.prefetch(fid)

            threading.Thread(target=worker, name="segmentation-preview-prefetch", daemon=True).start()

        self.prefetch_after_id = self.after(120, start_when_idle)

    def _reset_view_state(self):
        self.base_fit_scale = 1.0
        self.zoom_scale = 1.0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.view_initialized = False
        self.current_image_shape = None

    def _fit_view(self, img_rgb: np.ndarray):
        img_h, img_w = img_rgb.shape[:2]
        can_w = max(1, self.canvas.winfo_width())
        can_h = max(1, self.canvas.winfo_height())
        self.base_fit_scale = min(can_w / img_w, can_h / img_h)
        self.zoom_scale = 1.0
        self.scale = self.base_fit_scale
        self.offset_x = (can_w - img_w * self.scale) * 0.5
        self.offset_y = (can_h - img_h * self.scale) * 0.5
        self.view_initialized = True
        self.current_image_shape = img_rgb.shape[:2]

    def _present_image(self, img_rgb: np.ndarray, preserve_view: bool):
        if (not preserve_view) or (not self.view_initialized) or self.current_image_shape != img_rgb.shape[:2]:
            self._fit_view(img_rgb)
        img_h, img_w = img_rgb.shape[:2]
        can_w = max(1, self.canvas.winfo_width())
        can_h = max(1, self.canvas.winfo_height())
        scale = self.scale

        # Crop to the visible region in image coords -- avoids resizing off-screen pixels.
        src_x0 = max(0.0, -self.offset_x / scale)
        src_y0 = max(0.0, -self.offset_y / scale)
        src_x1 = min(float(img_w), (can_w - self.offset_x) / scale)
        src_y1 = min(float(img_h), (can_h - self.offset_y) / scale)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            self.canvas.delete("all")
            self._canvas_image_id = None
            self.tk_image = None
            return
        ix0, iy0 = int(src_x0), int(src_y0)
        ix1 = min(img_w, int(np.ceil(src_x1)))
        iy1 = min(img_h, int(np.ceil(src_y1)))
        crop = img_rgb[iy0:iy1, ix0:ix1]

        dst_w = max(1, min(can_w, int(round((ix1 - ix0) * scale))))
        dst_h = max(1, min(can_h, int(round((iy1 - iy0) * scale))))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (dst_w, dst_h), interpolation=interp)

        dst_x = int(max(0.0, self.offset_x + ix0 * scale))
        dst_y = int(max(0.0, self.offset_y + iy0 * scale))
        image = Image.fromarray(resized)
        if (self.tk_image is None or self.tk_image.width() != dst_w
                or self.tk_image.height() != dst_h):
            self.tk_image = ImageTk.PhotoImage(image)
        else:
            self.tk_image.paste(image)
        if self._canvas_image_id is None or not self.canvas.type(self._canvas_image_id):
            self._canvas_image_id = self.canvas.create_image(
                dst_x, dst_y, anchor="nw", image=self.tk_image
            )
        else:
            self.canvas.coords(self._canvas_image_id, dst_x, dst_y)
            self.canvas.itemconfigure(self._canvas_image_id, image=self.tk_image)

    def _render_canvas_view_rgb(self) -> np.ndarray:
        # Build the annotated frame at the video's native resolution, then crop
        # the currently visible image area. This avoids saving the lower-quality
        # canvas-resized preview.
        img_rgb = self._build_preview_rgb()
        if (not self.view_initialized) or self.current_image_shape != img_rgb.shape[:2]:
            self._fit_view(img_rgb)
        img_h, img_w = img_rgb.shape[:2]
        can_w = max(1, self.canvas.winfo_width())
        can_h = max(1, self.canvas.winfo_height())
        scale = max(1e-9, float(self.scale))

        src_x0 = max(0.0, -self.offset_x / scale)
        src_y0 = max(0.0, -self.offset_y / scale)
        src_x1 = min(float(img_w), (can_w - self.offset_x) / scale)
        src_y1 = min(float(img_h), (can_h - self.offset_y) / scale)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            raise RuntimeError("No image area is visible in the canvas.")

        ix0, iy0 = int(src_x0), int(src_y0)
        ix1 = min(img_w, int(np.ceil(src_x1)))
        iy1 = min(img_h, int(np.ceil(src_y1)))
        return img_rgb[iy0:iy1, ix0:ix1].copy()

    def _segmentation_signature(self) -> tuple:
        self._sync_roi_vars_to_set(self.roi_active_set_idx)
        self._sync_additional_outlier_vars_to_set()
        roi_key = roi_signature(self.roi_sets, self.roi_reverse_var.get())
        additional_signature = tuple(
            (
                int(bool(item.get("enabled", False))),
                int(item.get("frame_start", 0)),
                int(item.get("frame_end", -1)),
                str(item.get("segmentation_mode", "background_diff")),
                str(item.get("bg_method", "median")),
                int(item.get("threshold", 50)),
                int(item.get("dark_threshold", 50)),
                int(item.get("bright_threshold", 200)),
                int(item.get("diff_threshold", 50)),
                int(item.get("min_area", 10)),
            )
            for item in self.additional_outlier_sets
        )
        try:
            protected_bounds = self._protected_area_bounds()
        except Exception:
            protected_bounds = None
        return (
            self.bg_method_var.get(),
            self.segmentation_mode_var.get(),
            int(self.threshold_var.get()),
            int(self.dark_threshold_var.get()),
            int(self.bright_threshold_var.get()),
            int(self.diff_threshold_var.get()),
            int(self.min_area_var.get()),
            int(self.blur_ksize_var.get()),
            int(self.open_iter_var.get()),
            int(self.close_iter_var.get()),
            int(bool(self.fill_holes_var.get())),
            int(bool(self.invert_mask_var.get())),
            int(self.region_expand_px_var.get()),
            int(bool(self.region_expand_merge_only_var.get())),
            roi_key,
            additional_signature,
            int(self.additional_outlier_revision),
            protected_bounds,
            id(self.background_bgr),
        )

    def _classifier_signature(self) -> tuple:
        return (
            self.area_outlier_method_var.get(),
            float(self.area_iqr_min_var.get()),
            float(self.area_iqr_max_var.get()),
            float(self.area_absolute_min_var.get()),
            float(self.area_absolute_max_var.get()),
            int(bool(self.result_import_enabled_var.get() and self.result_obb_import is not None)),
            str(self.result_import_path_var.get()),
            float(self.result_import_min_coverage_var.get()),
            int(self.result_import_area_min_var.get()),
            int(self.result_import_area_max_var.get()),
        )

    def _protected_area_bounds(self) -> Optional[tuple[float, float]]:
        """Return the current normal-area bounds used for blue-mask protection."""
        if self.area_outlier_method_var.get() == "absolute":
            return self._current_area_bounds({})
        if self.analysis_iqr_stats and "area" in self.analysis_iqr_stats:
            return self._current_area_bounds(self.analysis_iqr_stats)
        return None


    def _mark_analysis_stale(self, redraw: bool):
        if self._switching_roi_set or self._applying_config:
            return
        self.analysis_is_stale = True
        if self.analysis_signature is None:
            self.analysis_ready_var.set("analysis: not computed")
        else:
            self.analysis_ready_var.set("analysis: stale (segmentation changed; re-run Analyze)")
        if self.outlier_section_revealed:
            self.outlier_section_revealed = False
            self._guidance_stage = "analysis"
            self._set_workflow_stage("analysis")
        else:
            self._update_analysis_dependent_buttons()
        if redraw:
            self.redraw_current_frame()

    def _segmentation_needs_background(self) -> bool:
        return self.segmentation_mode_var.get() in {"background_diff", "dark_region_and_background_diff", "bright_region_and_background_diff"}

    def _refresh_segmentation_controls(self):
        if not hasattr(self, "bg_controls_frame"):
            return
        if self._segmentation_needs_background():
            if not self.bg_controls_frame.winfo_ismapped():
                children = self.seg_section.winfo_children()
                after_widget = children[0] if children else None
                if after_widget is not None:
                    self.bg_controls_frame.pack(fill="x", after=after_widget)
                else:
                    self.bg_controls_frame.pack(fill="x")
        else:
            self.bg_controls_frame.pack_forget()

        if (
            hasattr(self, "threshold_controls_frame")
            and hasattr(self, "dark_hybrid_threshold_controls_frame")
            and hasattr(self, "bright_hybrid_threshold_controls_frame")
        ):
            mode = self.segmentation_mode_var.get()
            self.threshold_controls_frame.pack_forget()
            self.dark_hybrid_threshold_controls_frame.pack_forget()
            self.bright_hybrid_threshold_controls_frame.pack_forget()
            if mode == "dark_region_and_background_diff":
                self.dark_hybrid_threshold_controls_frame.pack(fill="x")
            elif mode == "bright_region_and_background_diff":
                self.bright_hybrid_threshold_controls_frame.pack(fill="x")
            else:
                self.threshold_controls_frame.pack(fill="x")

    def _toggle_roi_params(self):
        if not hasattr(self, "roi_params_frame"):
            return
        if self.roi_enabled_var.get() and not self._switching_roi_set and not self._applying_config:
            self._ensure_roi_points()
        state = "normal" if self.reader is not None else "disabled"
        self.roi_edit_button.configure(state=state, text="Edit ROI: ON" if self.roi_edit_var.get() else "Edit ROI")
        if hasattr(self, "canvas"):
            self.canvas.configure(cursor="crosshair" if self.roi_edit_var.get() else "")

    def _ensure_roi_points(self) -> bool:
        roi = self.roi_sets[self.roi_active_set_idx]
        if roi["points"] or self.reader is None:
            return False
        roi["points"] = initial_roi_points((self.reader.height, self.reader.width))
        return True

    def _toggle_roi_edit(self):
        if self.reader is None:
            return
        self.on_canvas_release()
        self.roi_edit_var.set(not self.roi_edit_var.get())
        if self.roi_edit_var.get():
            self.stop_playback()
            if self._ensure_roi_points():
                self._on_roi_geometry_changed()
        self._toggle_roi_params()
        self.canvas.configure(cursor="crosshair" if self.roi_edit_var.get() else "")
        self._draw_roi_canvas()

    def _set_default_roi(self, enable: bool = False, emit_status: bool = False):
        if self.reader is None:
            return
        self.roi_sets[self.roi_active_set_idx] = {
            **default_roi_set(), "enabled": bool(enable),
            "points": initial_roi_points((self.reader.height, self.reader.width)),
        }
        self._switching_roi_set = True
        try:
            self._sync_set_to_roi_vars(self.roi_active_set_idx)
        finally:
            self._switching_roi_set = False
        self.roi_reverse_var.set(False)
        self.roi_edit_var.set(False)
        self._roi_drag_mode = None
        self._toggle_roi_params()
        if emit_status:
            self.set_status("Default rectangular polygon ROI")

    def _on_segmentation_mode_changed(self):
        if self._applying_config:
            return
        self.current_mask_cache.clear()
        self.analysis_blobs_by_frame.clear()
        self.analysis_outlier_frames.clear()
        self.analysis_frames = []
        self.analysis_signature = None
        self.analysis_bounds = {}
        self.analysis_iqr_stats = {}
        self.analysis_is_stale = True
        self.analysis_ready_var.set("analysis: not computed")
        self.outlier_section_revealed = False
        self._guidance_stage = "background" if self._segmentation_needs_background() else "threshold"
        self._update_analysis_dependent_buttons()
        self._refresh_segmentation_controls()
        if not self._segmentation_needs_background():
            self.background_ready_var.set(f"background: not required ({self.segmentation_mode_var.get()})")
            self._set_workflow_stage("analysis")
        else:
            self.background_ready_var.set("background: ready" if self.background_bgr is not None else "background: not computed")
            self._set_workflow_stage("analysis" if self.background_bgr is not None else "background")
            if self.background_bgr is None:
                self._schedule_background_rebuild()
        if self._additional_outlier_needs_background():
            self._schedule_background_rebuild()
        self.redraw_current_frame()

    def _on_background_method_changed(self):
        if self.reader is None or self._applying_config or self._loading_video:
            return
        method = self.bg_method_var.get()
        try:
            if self._activate_cached_background(method):
                self.set_status(f"Background switched to cached {method}")
                return
        except Exception as exc:
            self.set_status(f"Failed to load cached background: {exc}")

        self.background_bgr = None
        self.active_background_method = None
        self._invalidate_analysis_for_background_change()
        self.background_ready_var.set(f"background: not computed ({method})")
        if self.bg_preview_button is not None:
            self.bg_preview_button.configure(state="disabled")
        if self._segmentation_needs_background():
            self._set_workflow_stage("background")
            self._schedule_background_rebuild()
        else:
            self.background_ready_var.set(f"background: not required ({self.segmentation_mode_var.get()})")
        self.redraw_current_frame()

    def _schedule_background_rebuild(self, delay_ms: int = 500):
        if self.reader is None or self._applying_config or self._loading_video:
            return
        if self.background_rebuild_job is not None:
            try:
                self.after_cancel(self.background_rebuild_job)
            except Exception:
                pass
        self.background_rebuild_job = self.after(delay_ms, self._run_scheduled_background_rebuild)

    def _run_scheduled_background_rebuild(self):
        self.background_rebuild_job = None
        if self._background_building:
            self._schedule_background_rebuild(delay_ms=300)
            return
        if self.reader is not None and (
            self._segmentation_needs_background() or self._additional_outlier_needs_background()
        ):
            try:
                self.compute_background()
            except Exception as exc:
                self.background_ready_var.set("background: not computed")
                self.set_status(f"Background build failed: {exc}", auto_clear=False)
                messagebox.showerror("Background extraction", str(exc))

    # ROI set management

    @staticmethod
    def _default_roi_set() -> dict:
        return default_roi_set()

    @staticmethod
    def _normalize_roi_set(roi: dict) -> dict:
        return normalize_roi_set(roi)

    @staticmethod
    def _default_additional_outlier_set() -> dict:
        return {
            "enabled": False,
            "frame_start": 0,
            "frame_end": -1,
            "segmentation_mode": "background_diff",
            "bg_method": "median",
            "threshold": 50,
            "dark_threshold": 50,
            "bright_threshold": 200,
            "diff_threshold": 50,
            "min_area": 10,
        }

    @staticmethod
    def _normalize_additional_outlier_set(item: dict) -> dict:
        defaults = CrossingReviewApp._default_additional_outlier_set()
        mode = item.get("segmentation_mode", defaults["segmentation_mode"])
        if mode not in {
            "background_diff", "dark_region", "bright_region",
            "dark_region_and_background_diff", "bright_region_and_background_diff",
        }:
            mode = defaults["segmentation_mode"]
        bg_method = item.get("bg_method", defaults["bg_method"])
        if bg_method not in BACKGROUND_METHODS:
            bg_method = defaults["bg_method"]

        def _int_value(key: str) -> int:
            try:
                return int(item.get(key, defaults[key]))
            except (TypeError, ValueError):
                return int(defaults[key])

        return {
            "enabled": bool(item.get("enabled", defaults["enabled"])),
            "frame_start": max(0, _int_value("frame_start")),
            "frame_end": _int_value("frame_end"),
            "segmentation_mode": mode,
            "bg_method": bg_method,
            "threshold": max(0, min(255, _int_value("threshold"))),
            "dark_threshold": max(0, min(255, _int_value("dark_threshold"))),
            "bright_threshold": max(0, min(255, _int_value("bright_threshold"))),
            "diff_threshold": max(0, min(255, _int_value("diff_threshold"))),
            "min_area": max(1, _int_value("min_area")),
        }

    def _on_roi_var_changed(self, *_):
        if self._switching_roi_set or self._applying_config:
            return
        if self.roi_enabled_var.get():
            self._ensure_roi_points()
        self._sync_roi_vars_to_set(self.roi_active_set_idx)
        self._on_roi_geometry_changed()

    def _on_roi_reverse_changed(self, *_):
        if not self._switching_roi_set and not self._applying_config:
            self._on_roi_geometry_changed()

    def _on_roi_geometry_changed(self):
        self.current_mask_cache.clear()
        self._area_reference_key = None
        self._mark_analysis_stale(redraw=self._roi_drag_mode is None)
        if self._roi_drag_mode is not None:
            # Motion events repaint only vector outlines. Segmentation and area
            # controls are recomputed once the mouse is released.
            self._draw_roi_canvas()

    def _sync_roi_vars_to_set(self, idx: int):
        if 0 <= idx < len(self.roi_sets):
            self.roi_sets[idx] = {
                **self.roi_sets[idx],
                "enabled": bool(self.roi_enabled_var.get()),
                "frame_start": int(self.roi_frame_start_var.get()),
                "frame_end": int(self.roi_frame_end_var.get()),
            }

    def _sync_set_to_roi_vars(self, idx: int):
        if idx < 0 or idx >= len(self.roi_sets):
            return
        roi = normalize_roi_set(self.roi_sets[idx])
        self.roi_sets[idx] = roi
        self.roi_enabled_var.set(roi["enabled"])
        self.roi_frame_start_var.set(roi["frame_start"])
        self.roi_frame_end_var.set(roi["frame_end"])

    def _update_roi_set_selector_values(self):
        if self.roi_set_combo is None:
            return
        values = [f"Set {i + 1}" for i in range(len(self.roi_sets))]
        self.roi_set_combo.configure(values=values)

    def _update_roi_set_selector(self):
        if self.roi_set_combo is None:
            return
        self.roi_set_combo.set(f"Set {self.roi_active_set_idx + 1}")

    def _on_roi_set_selected(self, choice: str):
        try:
            idx = int(choice.split()[-1]) - 1
        except (ValueError, IndexError):
            return
        if 0 <= idx < len(self.roi_sets) and idx != self.roi_active_set_idx:
            self._switch_roi_set(idx)

    def _switch_roi_set(self, new_idx: int):
        if new_idx == self.roi_active_set_idx:
            return
        self.on_canvas_release()
        self._sync_roi_vars_to_set(self.roi_active_set_idx)
        self._switching_roi_set = True
        try:
            self.roi_active_set_idx = new_idx
            self._sync_set_to_roi_vars(new_idx)
            self._update_roi_set_selector()
        finally:
            self._switching_roi_set = False
        self._toggle_roi_params()
        self.redraw_current_frame()

    def _add_roi_set(self):
        if len(self.roi_sets) >= 8:
            return
        self._sync_roi_vars_to_set(self.roi_active_set_idx)
        new_roi: dict = self._default_roi_set()
        if self.reader is not None:
            new_roi["points"] = initial_roi_points((self.reader.height, self.reader.width))
        self.roi_sets.append(new_roi)
        self._update_roi_set_selector_values()
        self._switch_roi_set(len(self.roi_sets) - 1)

    def _remove_roi_set(self):
        if len(self.roi_sets) <= 1:
            return
        self.on_canvas_release()
        old_idx = self.roi_active_set_idx
        self.roi_sets.pop(old_idx)
        new_idx = min(old_idx, len(self.roi_sets) - 1)
        self._update_roi_set_selector_values()
        self._switching_roi_set = True
        try:
            self.roi_active_set_idx = new_idx
            self._sync_set_to_roi_vars(new_idx)
            self._update_roi_set_selector()
        finally:
            self._switching_roi_set = False
        self._toggle_roi_params()
        self._on_roi_geometry_changed()

    def _toggle_result_import_params(self):
        if self.result_import_params_frame is None:
            return
        if self.result_import_enabled_var.get():
            if not self.result_import_params_frame.winfo_ismapped():
                self.result_import_params_frame.pack(fill="x", padx=0, pady=(0, 4))
        else:
            self.result_import_params_frame.pack_forget()

    def _result_import_status_text(self) -> str:
        imported = self.result_obb_import
        if imported is None:
            return "result OBB: not loaded"
        return (
            f"result OBB: {len(imported.track_ids)} IDs / {imported.obb_count} OBBs / "
            f"frames {imported.frame_min}-{imported.frame_max}"
        )

    def browse_result_obb_csv(self):
        current = self.result_import_path_var.get().strip()
        initial_dir = os.path.dirname(current) if current else os.path.dirname(
            self.video_path_var.get().strip())
        path = filedialog.askopenfilename(
            title="Select tracking result CSV",
            filetypes=RESULT_CSV_EXTS,
            initialdir=initial_dir or os.getcwd(),
        )
        if path:
            self.load_result_obb_csv(path)

    def load_result_obb_csv(self, path: str) -> bool:
        try:
            imported = _load_result_obb_csv(path)
        except Exception as exc:
            messagebox.showerror("Result OBB Import", str(exc))
            return False
        self.result_obb_import = imported
        self.result_import_path_var.set(imported.path)
        self.result_import_status_var.set(self._result_import_status_text())
        if not self.result_import_enabled_var.get():
            self.result_import_enabled_var.set(True)  # traces refresh the match
        else:
            self._on_result_import_changed()
        self.set_status(f"Result OBB loaded: {os.path.basename(imported.path)}")
        return True

    def clear_result_obb_import(self):
        self.result_obb_import = None
        self.result_import_path_var.set("")
        self.result_import_status_var.set(self._result_import_status_text())
        self._on_result_import_changed()
        self.set_status("Imported result OBB cleared.")

    def _refresh_result_match(self):
        """Rebuild the plain snapshot worker threads classify blobs against."""
        if not self.result_import_enabled_var.get() or self.result_obb_import is None:
            self._result_match = None
            return
        self._result_match = _ResultMatchConfig(
            min_coverage=float(self.result_import_min_coverage_var.get()),
            obbs_by_frame=self.result_obb_import.obbs_by_frame,
            area_bounds=self._current_result_area_bounds(),
        )

    def _current_result_area_bounds(self) -> tuple[float, float]:
        lo, hi = int(self.result_import_area_min_var.get()), int(self.result_import_area_max_var.get())
        if not 0 <= lo <= hi:
            raise ValueError("Result OBB area bounds require 0 <= minimum <= maximum.")
        return float(lo), float(hi)

    def _update_result_area_control_ranges(self):
        control = self._result_area_control
        reference_key = (self._area_reference_key, None)
        values = []
        active = self._result_match.obbs_by_frame if self._result_match is not None else None
        if self.reader is not None:
            shape = (self.reader.height, self.reader.width)
            frames = []
            if self._area_reference_frame is not None:
                frames.append((0, self._area_reference_blobs))
            if self.frame_bgr_cache is not None and self.current_frame != 0:
                frames.append((self.current_frame, None))
            frames.extend(self.analysis_blobs_by_frame.items())
            for fid, blobs in frames:
                obbs = active.get(fid) if active is not None else None
                if active is not None and obbs is None:
                    continue
                if blobs is None:
                    blobs = self._segment_frame(self.frame_bgr_cache, fid)[1]
                areas = _result_blob_pixel_areas(blobs, obbs, shape)
                values = [area for area in areas if area > 0]
                if values:
                    reference_key = (self._area_reference_key, fid, id(obbs), tuple(values))
                    break
        has_reference = bool(values)
        if has_reference:
            q1, median, q3 = np.percentile(np.asarray(values, np.float64), (25, 50, 75))
            stats = (float(q1), float(median), float(q3), float(q3-q1))
        else:
            stats = (0.0, 0.0, 0.0, 0.0)
        reference_changed = reference_key != self._result_area_reference_key
        self._result_area_reference_key = reference_key
        self._result_area_reference_stats = stats
        if has_reference:
            q1, median, q3, iqr = self._normalized_area_stat(stats)
            # Round outward so a constant non-multiple-of-ten reference is
            # included by its defaults, while ordinary Absolute stays unchanged.
            lo = max(0, int(math.floor((q1 - 1.5 * iqr) / 10.0)) * 10)
            hi = max(lo, int(math.ceil((q3 + 1.5 * iqr) / 10.0)) * 10)
            control._absolute_default_values = (lo, hi)
            if control._absolute_defaults_pending and not self._applying_config and (
                int(self.result_import_area_min_var.get()), int(self.result_import_area_max_var.get())
            ) != (lo, hi):
                self._result_area_updating = True
                try:
                    self.result_import_area_min_var.set(lo)
                    self.result_import_area_max_var.set(hi)
                finally:
                    self._result_area_updating = False
                self._refresh_result_match()
            cap = max(1, math.ceil(max(median * 2, hi * 1.2)))
        else:
            control._absolute_default_values = None
            cap = 1000
        if reference_changed and not control._absolute_zoomed and not control._absolute_dragging:
            control._absolute_slider_cap = cap
        self._sync_absolute_control(force=True, result=True)

    def _on_result_area_changed(self, index: int):
        if self._applying_config or self._result_area_updating:
            return
        self._result_area_control._absolute_defaults_pending = False
        variables = (self.result_import_area_min_var, self.result_import_area_max_var)
        value, other = max(0, int(variables[index].get())), max(0, int(variables[1 - index].get()))
        bounded = min(value, other) if index == 0 else max(value, other)
        self._result_area_updating = True
        try:
            if bounded != int(variables[index].get()):
                variables[index].set(bounded)
            self._sync_absolute_control(result=True)
        finally:
            self._result_area_updating = False
        self._on_result_import_changed()

    def _reset_result_area_bound(self, index: int):
        defaults = self._result_area_control._absolute_default_values
        if defaults is not None:
            self._set_absolute_threshold(index, defaults[index], result=True)
        return "break"

    def _active_pre_result_path(self) -> str:
        """Return the loaded CSV used by the enabled Result OBB Import."""
        if not self.result_import_enabled_var.get() or self.result_obb_import is None:
            return ""
        return self.result_obb_import.path

    def _on_result_import_changed(self):
        if self._applying_config:
            return
        self._refresh_result_match()
        self._refresh_bounds_from_fixed_stats(show_error=False)

    def _current_result_obbs(self) -> "Optional[np.ndarray]":
        if not self.result_import_show_obb_var.get() or self._result_match is None:
            return None
        return self._result_match.obbs_by_frame.get(int(self.current_frame))

    def _toggle_static_options(self) -> None:
        if self.static_exclusion_enabled_var.get():
            self.static_options_frame.pack(fill="x", pady=(0, 4))
        else:
            self.static_options_frame.pack_forget()

    def _on_static_search_changed(self) -> None:
        """Enabled checkbox changes optional-section visibility."""
        if not hasattr(self, "static_options_frame"):
            return
        self._toggle_static_options()
        self._on_static_parameters_changed()
        if self.reader is not None and not getattr(self, "_loading_video", False):
            self._set_workflow_stage("static")

    def _on_static_parameters_changed(self) -> None:
        """Invalidate only the search data; never rebuild the sidebar on drag."""
        if not hasattr(self, "static_search_status_var"):
            return
        had_results = self._static_search_signature is not None
        self._static_search_signature = None
        self._static_exclusions = {}
        self._static_digests = None
        self.static_search_status_var.set(
            "Not searched — press Search before Processing"
            if self.static_exclusion_enabled_var.get() else "Disabled"
        )
        self._refresh_search_glows()
        if (had_results and self.reader is not None
                and not getattr(self, "_loading_video", False)
                and not self._applying_config):
            self.redraw_current_frame()

    def _static_search_key(self) -> tuple:
        """Cache key covering source video, segmentation, thresholds, and frame range."""
        return (
            os.path.normcase(os.path.abspath(self.video_path_var.get().strip())),
            int(self.frame_count),
            self._segmentation_signature(),
            self._classifier_signature(),
            bool(self.single_blob_smoothing_enabled_var.get()),
            int(self.single_blob_smoothing_level_var.get()),
            int(self.training_frame_start_var.get()),
            int(self.training_frame_end_var.get()),
            int(self.static_frame_window_var.get()),
            float(self.static_presence_var.get()),
            float(self.static_iou_var.get()),
            float(self.static_max_movement_var.get()),
        )

    def _static_search_is_current(self) -> bool:
        return bool(
            self.static_exclusion_enabled_var.get()
            and self._static_search_signature is not None
            and self._static_search_signature == self._static_search_key()
        )

    def _apply_static_outliers(
        self, blobs: list[BlobMetrics], fid: int,
        exclusions: Optional[dict[int, set[int]]] = None,
        base_crossing: Optional[list[bool]] = None,
    ) -> list[BlobMetrics]:
        if exclusions is None:
            if not self._static_search_is_current():
                return blobs
            exclusions = self._static_exclusions
        matched_indices = exclusions.get(int(fid), ())
        for position, blob in enumerate(blobs):
            crossing = base_crossing[position] if base_crossing is not None else blob.is_crossing
            if blob.blob_index in matched_indices and not crossing:
                blob.persistent_static_outlier = True
                blob.is_crossing = True
        return blobs

    # -- temporal merge filter ------------------------------------------------
    def _toggle_temporal_options(self) -> None:
        if self.temporal_merge_enabled_var.get():
            self.temporal_options_frame.pack(fill="x", pady=(0, 4))
        else:
            self.temporal_options_frame.pack_forget()

    def _on_temporal_enabled_changed(self) -> None:
        if not hasattr(self, "temporal_options_frame"):
            return
        self._toggle_temporal_options()
        self._on_temporal_parameters_changed()
        if self.reader is not None and not getattr(self, "_loading_video", False):
            self._set_workflow_stage(
                "temporal" if self.temporal_merge_enabled_var.get() else "analysis")

    def _on_temporal_parameters_changed(self) -> None:
        """Mark the evaluation stale; the stored graph is kept for re-evaluation."""
        if not hasattr(self, "temporal_search_status_var"):
            return
        self._refresh_search_glows()
        self._update_temporal_status()
        if (self._temporal_search is not None and self.reader is not None
                and not getattr(self, "_loading_video", False) and not self._applying_config):
            self.redraw_current_frame()

    def _reset_temporal_search(self) -> None:
        self._temporal_search = None
        if hasattr(self, "temporal_search_status_var"):
            self._update_temporal_status()
        self._refresh_search_glows()

    def _temporal_params(self):
        from main.temporal_merge_blobs import TemporalMergeParams
        tau = float(self.temporal_merge_tau_var.get()) if self.temporal_merge_tau_manual_var.get() else None
        params = TemporalMergeParams(
            tau=tau,
            include_one_sided=self.temporal_merge_mode_var.get() == "one_sided",
            share_rounding=float(self.temporal_merge_share_rounding_var.get()),
            one_sided_mass_ratio=float(self.temporal_merge_mass_ratio_var.get()),
        )
        params.validate()
        return params

    def _temporal_eval_key(self) -> tuple:
        try:
            p = self._temporal_params()
        except (ValueError, TypeError, tk.TclError):
            return ("invalid",)
        return (p.tau, p.include_one_sided, p.share_rounding, p.one_sided_mass_ratio)

    def _search_frame_range(self) -> tuple[int, int]:
        start = max(0, int(self.training_frame_start_var.get()))
        return start, self._resolved_frame_end(int(self.training_frame_end_var.get()))

    def _additional_sets_in_range(self, start: int, end: int) -> bool:
        self._sync_additional_outlier_vars_to_set()
        for item in self.additional_outlier_sets:
            if not item.get("enabled", False):
                continue
            fs, fe = int(item.get("frame_start", 0)), int(item.get("frame_end", -1))
            if fs <= end and (fe < 0 or fe >= start):
                return True
        return False

    def _temporal_geometry_dependent(self) -> bool:
        """True when blob shapes themselves depend on the area bounds."""
        start, end = self._search_frame_range()
        return bool(
            self.single_blob_smoothing_enabled_var.get()
            or (self.result_import_enabled_var.get() and self.result_obb_import is not None)
            or self._additional_sets_in_range(start, end)
        )

    def _current_area_bounds_or_none(self) -> Optional[tuple[float, float]]:
        try:
            bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats).get("area")
        except (ValueError, TypeError):
            return None
        return (float(bounds[0]), float(bounds[1])) if bounds else None

    def _temporal_cache_path(self) -> str:
        """Keep the Search graph next to its JSON config, not in the video."""
        config = os.path.abspath(self.config_path or self._default_config_path())
        return os.path.splitext(config)[0] + ".temporal_search.npz"

    def _temporal_cache_fingerprint(self) -> str:
        """Stable cross-session key; reject changed videos, settings and code.

        _temporal_geometry_key intentionally contains runtime-only identifiers
        (a background object's id and an edit revision); normalize those
        while hashing the actual background image instead.
        """
        if self.reader is None:
            raise ValueError("Load a video before saving temporal Search.")
        video = os.path.abspath(self.video_path_var.get().strip())
        st = os.stat(video)
        # Inspect both ends so equal-sized/re-timestamped videos are unlikely
        # to reuse a Search for unrelated footage.
        video_hash = hashlib.blake2b(digest_size=16)
        with open(video, "rb") as stream:
            video_hash.update(stream.read(65536))
            if st.st_size > 65536:
                stream.seek(max(0, st.st_size - 65536))
                video_hash.update(stream.read(65536))
        key = list(self._temporal_geometry_key())
        signature = list(key[4])
        signature[-3] = None  # additional_outlier_revision: per-session only
        if self.background_bgr is not None:
            bg = np.ascontiguousarray(self.background_bgr)
            signature[-1] = (
                bg.shape, hashlib.blake2b(bg.tobytes(), digest_size=16).hexdigest())
        else:
            signature[-1] = None
        key[4] = tuple(signature)
        key[0] = (os.path.basename(video), int(st.st_size),
                  int(st.st_mtime_ns), video_hash.hexdigest())
        # A changed importer or segmentation implementation must not silently
        # apply stale blob indices, even if visible controls have not changed.
        source_hash = hashlib.blake2b(digest_size=16)
        for source in (Path(__file__), PROJECT_ROOT / "main" / "temporal_merge_blobs.py",
                       PROJECT_ROOT / "main" / "frame_scan.py"):
            with open(source, "rb") as stream:
                for block in iter(lambda: stream.read(262144), b""):
                    source_hash.update(block)
        if self.result_import_enabled_var.get():
            imported = self.result_import_path_var.get().strip()
            if imported and os.path.isfile(imported):
                csv_stat = os.stat(imported)
                key.append((int(csv_stat.st_size), int(csv_stat.st_mtime_ns)))
        payload = repr((1, tuple(key), source_hash.hexdigest())).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=20).hexdigest()

    def _save_temporal_search_cache(self) -> None:
        """Persist only a current, enabled Search after saving its config."""
        state = self._temporal_search
        if state is None or not self._temporal_search_is_current():
            return
        from main.temporal_merge_cache import save_temporal_merge_cache

        save_temporal_merge_cache(
            self._temporal_cache_path(), state.graph, state.result, state.params,
            self._temporal_cache_fingerprint(),
        )

    def _restore_temporal_search_cache(self) -> None:
        """Restore verified Search data on config load; never trust stale blobs."""
        if self.reader is None or not self.config_path:
            return
        from main.temporal_merge_cache import load_temporal_merge_cache

        try:
            fingerprint = self._temporal_cache_fingerprint()
            params = self._temporal_params()
            cached = load_temporal_merge_cache(
                self._temporal_cache_path(), fingerprint, params)
            if cached is None:
                return
            graph, result = cached
            start, end = self._search_frame_range()
            if (graph.first_frame != start or graph.frame_count != end - start + 1
                    or graph.image_shape != (int(self.reader.height), int(self.reader.width))):
                return
            self._temporal_search = _TemporalSearchState(
                geometry_key=self._temporal_geometry_key(),
                kinds_key=self._temporal_kinds_key(),
                eval_key=self._temporal_eval_key(),
                reclassifiable=not self._temporal_geometry_dependent(),
                start=start, end=end, graph=graph, result=result, params=params,
            )
            if self.temporal_merge_enabled_var.get():
                self.temporal_search_status_var.set(
                    "Saved Search loaded — " + self._temporal_summary_text(result))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self._temporal_search = None
            self.temporal_search_status_var.set(
                f"Saved Search unavailable ({exc}); press Search")

    def _temporal_geometry_key(self) -> tuple:
        """Everything that changes blob geometry or order (needs a rescan)."""
        start, end = self._search_frame_range()
        signature = list(self._segmentation_signature())
        dependent = self._temporal_geometry_dependent()
        if not dependent:
            # Protected-area bounds only reach the geometry through an enabled
            # Additional Outlier Set (excluded above), so ignore them here.
            signature[-2] = None
        smoothing = bool(self.single_blob_smoothing_enabled_var.get())
        return (
            os.path.normcase(os.path.abspath(self.video_path_var.get().strip())),
            int(self.frame_count), start, end, tuple(signature),
            smoothing, int(self.single_blob_smoothing_level_var.get()) if smoothing else 0,
            self._classifier_signature() if dependent else None,
            self._current_area_bounds_or_none() if dependent else None,
        )

    def _temporal_kinds_key(self) -> tuple:
        """Temporal evidence precedes statistical outliers and has no area-bound dependency."""
        return ()

    def _temporal_search_is_current(self) -> bool:
        state = self._temporal_search
        return bool(
            self.temporal_merge_enabled_var.get() and state is not None
            and state.geometry_key == self._temporal_geometry_key()
            and state.kinds_key == self._temporal_kinds_key()
            and state.eval_key == self._temporal_eval_key()
        )

    def _temporal_graph_reusable(self) -> bool:
        state = self._temporal_search
        if state is None or state.geometry_key != self._temporal_geometry_key():
            return False
        return state.reclassifiable or state.kinds_key == self._temporal_kinds_key()

    def _refresh_search_glows(self) -> None:
        if self._applying_config or getattr(self, "_loading_video", False):
            # Keys read Tk state that is only half restored here.
            if hasattr(self, "_static_search_glow"):
                self._static_search_glow.set_active(bool(self.static_exclusion_enabled_var.get()))
            if hasattr(self, "_temporal_search_glow"):
                self._temporal_search_glow.set_active(bool(self.temporal_merge_enabled_var.get()))
            return
        if hasattr(self, "_static_search_glow"):
            self._static_search_glow.set_active(
                bool(self.static_exclusion_enabled_var.get()) and not self._static_search_is_current())
        if hasattr(self, "_temporal_search_glow"):
            self._temporal_search_glow.set_active(
                bool(self.temporal_merge_enabled_var.get()) and not self._temporal_search_is_current())

    def _update_temporal_status(self) -> None:
        if not self.temporal_merge_enabled_var.get():
            self.temporal_search_status_var.set("Disabled")
            return
        state = self._temporal_search
        if state is None:
            self.temporal_search_status_var.set("Not searched — press Search before Processing")
            return
        if self._applying_config or getattr(self, "_loading_video", False):
            return
        if not self._temporal_search_is_current():
            note = "re-evaluation only" if self._temporal_graph_reusable() else "rescan required"
            self.temporal_search_status_var.set(f"Settings changed — press Search ({note})")
            return
        self.temporal_search_status_var.set(self._temporal_summary_text(state.result))

    @staticmethod
    def _temporal_summary_text(result, detailed: bool = False) -> str:
        """One line for the sidebar (it has a fixed height); details for the status bar."""
        counts, diag = result.counts, result.diagnostics
        candidates = sum(len(items) for items in result.levels.values())
        text = f"Temporal merges {candidates} · excluded {counts['excluded_blobs']}"
        if not diag.get("reliable"):
            text = "Unreliable overlap, nothing excluded · " + text
        if detailed:
            text += (f" · containment {diag.get('tau', 0):.2f} {diag.get('tau_mode', 'auto')}, "
                     f"P10 {diag.get('self_containment_p10')}")
            if diag.get("warnings"):
                text += " · " + diag["warnings"][0]
        return text

    def _search_filter_snapshot(self) -> "_FilterPlan":
        """Current, verified Search results for preview and Export (apply only)."""
        static = None
        if self._static_search_is_current() and self._static_digests is not None:
            # Copies: a worker may read these while the user edits settings.
            static = (dict(self._static_exclusions), self._static_digests, self._static_scan_start, None)
        temporal = self._temporal_search if self._temporal_search_is_current() else None
        return _FilterPlan(mode="apply", static=static, temporal=temporal)

    def _apply_preview_filters(self, blobs: list[BlobMetrics], fid: int) -> list[BlobMetrics]:
        plan = self._search_filter_snapshot()
        self._filter_preview_mismatch = _apply_filter_plan_to_frame(blobs, fid, plan, self._apply_static_outliers)
        return blobs

    def _goto_temporal_event(self, direction: int) -> None:
        state = self._temporal_search
        if state is None or self.reader is None:
            return
        frames = state.result.event_frames()
        if not frames:
            self.set_status("No temporal merge candidates to show.")
            return
        current = int(self.current_frame)
        if direction > 0:
            target = next((f for f in frames if f > current), frames[0])
        else:
            target = next((f for f in reversed(frames) if f < current), frames[-1])
        self.set_frame(target)

    def _processing_filter_plan(self) -> _FilterPlan:
        static = None
        if self._static_search_is_current() and self._static_digests is not None:
            args = (int(self.static_frame_window_var.get()), float(self.static_presence_var.get()),
                    float(self.static_iou_var.get()), float(self.static_max_movement_var.get()))
            static = (dict(self._static_exclusions), self._static_digests, self._static_scan_start, args)
        temporal = self._temporal_search if self._temporal_search_is_current() else None
        plan = _FilterPlan(mode="processing", static=static, temporal=temporal)
        # The log reports the settings this run used, not later GUI edits.
        plan.stats.update(
            static_enabled=static is not None,
            temporal_enabled=temporal is not None,
            temporal_params=temporal.params if temporal is not None else None,
        )
        return plan

    def _report_scan_progress(self, label: str, n: int, total: int, started: float, scale: float) -> None:
        if not (n == 1 or n % 25 == 0 or n == total):
            return
        rate = n / max(1e-9, time.perf_counter() - started)
        self._set_progress_popup(
            f"{label}: {n}/{total} ({100 * n / total:.0f}%)\n"
            f"speed={rate:.1f} fps / remaining={self._format_seconds((total - n) / max(rate, 1e-9))}",
            scale * n / total,
        )

    def search_persistent_static_blobs(self) -> None:
        self.search_optional_filters("static")

    def search_temporal_merges(self) -> None:
        self.search_optional_filters("temporal")

    def search_optional_filters(self, requester: str) -> None:
        """One ordered scan feeds every enabled filter whose results are stale.

        Frames are decoded and segmented once even when both the
        persistent-static and the temporal-merge filters need data. A
        temporal parameter change only re-evaluates the stored graph.
        """
        if self.reader is None or self._progress_busy:
            return
        static_on = bool(self.static_exclusion_enabled_var.get())
        temporal_on = bool(self.temporal_merge_enabled_var.get())
        if not (static_on or temporal_on):
            return
        title = "Temporal merges" if requester == "temporal" else "Persistent static blobs"
        analysis_ready = not self.analysis_is_stale and (
            not self._needs_area_analysis() or bool(self.analysis_iqr_stats))
        if requester != "temporal" and not analysis_ready:
            messagebox.showwarning(title, "Run Analyze with the current segmentation settings first.")
            return
        # Temporal Search is deliberately available before Outlier Extraction.
        # Static filtering retains its existing dependence on Analyze.
        run_static = static_on and analysis_ready and (
            requester == "static" or not self._static_search_is_current())
        run_temporal = temporal_on and (requester == "temporal" or not self._temporal_search_is_current())
        scan_temporal = run_temporal and not self._temporal_graph_reusable()
        try:
            start, end = self._search_frame_range()
            if start > end:
                raise ValueError("Invalid training-frame range.")
            static_args = None
            if run_static:
                from main.persistent_static_blobs import PersistentStaticScanner
                window = int(self.static_frame_window_var.get())
                static_args = (window, float(self.static_presence_var.get()),
                               float(self.static_iou_var.get()), float(self.static_max_movement_var.get()))
                if not 2 <= window <= 2000:
                    raise ValueError("Frame window must be between 2 and 2000.")
                if end - start + 1 < window:
                    raise ValueError("Frame window exceeds the selected training-frame range.")
                PersistentStaticScanner(*static_args)  # validate thresholds now
            params = self._temporal_params() if run_temporal else None
            static_key = self._static_search_key() if run_static else None
            geometry_key = self._temporal_geometry_key()
            kinds_key = self._temporal_kinds_key()
            eval_key = self._temporal_eval_key()
            reclassifiable = not self._temporal_geometry_dependent()
            cfg = self._capture_seg_config()
            result_match = self._result_match
            bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats)
            area_bounds = self._current_area_bounds_or_none()
            image_shape = (int(self.reader.height), int(self.reader.width))
        except (ValueError, TypeError, tk.TclError) as exc:
            messagebox.showerror(title, str(exc))
            return

        if run_static:
            self._static_search_signature = None
            self._static_exclusions = {}
            self._static_digests = None
            self.static_search_status_var.set("Searching...")
        if run_temporal:
            self.temporal_search_status_var.set("Searching..." if scan_temporal else "Evaluating...")
        self._prefetch_cancel.set()
        self.reader.clear_cache()
        outcome: dict[str, object] = {}
        error: list[Optional[Exception]] = [None]
        previous = self._temporal_search

        def work():
            from main.frame_scan import frame_blob_digest
            from main.persistent_static_blobs import PersistentStaticScanner
            from main.temporal_merge_blobs import (
                TemporalMergeGraphBuilder, evaluate_temporal_merges,
            )
            try:
                graph = previous.graph if (run_temporal and not scan_temporal) else None
                if run_static or scan_temporal:
                    scanner = PersistentStaticScanner(*static_args) if run_static else None
                    builder = TemporalMergeGraphBuilder(image_shape) if scan_temporal else None
                    context = _FrameContext(cfg, image_shape) if scan_temporal else None
                    total = end - start + 1
                    digests = np.zeros(total, dtype=np.uint64)

                    def segment(fid, frame):
                        return _segment_blobs_with_config(frame, fid, cfg)

                    started = time.perf_counter()
                    label = "Searching optional filters"
                    # Rasterizing runs here, next to classification: in a
                    # second pool its per-blob Python work fought the
                    # segmentation workers for the GIL and became ~30x slower.
                    for n, (fid, blobs) in enumerate(self._ordered_frames(start, end, segment), 1):
                        if self._progress_cancel:
                            return False
                        blobs = self._classify_frame(blobs, fid, cfg, bounds, result_match, image_shape)
                        digest = frame_blob_digest(blobs)
                        digests[fid - start] = digest
                        if scanner is not None:
                            scanner.add_frame(fid, blobs)
                        if builder is not None:
                            builder.add_frame(context.nodes(fid, blobs, digest))
                        self._report_scan_progress(label, n, total, started, 90.0 if run_temporal else 100.0)
                    if self._progress_cancel:
                        return False
                    if scanner is not None:
                        outcome["static"] = (scanner.excluded, digests, scanner.excluded_count,
                                             scanner.qualifying_windows)
                    if builder is not None:
                        graph = builder.finish()
                if run_temporal:
                    self._set_progress_popup("Evaluating temporal merge evidence...", None)
                    result = evaluate_temporal_merges(graph, params)
                    outcome["temporal"] = (graph, result)
                return True
            except Exception as exc:
                error[0] = exc
                return False

        def _finish_search(ok):
            if not ok:
                if run_static:
                    self.static_search_status_var.set("Search cancelled or failed — press Search again")
                if run_temporal:
                    self.temporal_search_status_var.set("Search cancelled or failed — press Search again")
                self._refresh_search_glows()
                if error[0] is not None:
                    messagebox.showerror(title, str(error[0]))
                return

            if "static" in outcome:
                if static_key != self._static_search_key():
                    self.static_search_status_var.set("Settings changed — press Search again")
                else:
                    excluded, digests, count, windows = outcome["static"]
                    self._static_exclusions = excluded
                    self._static_digests = digests
                    self._static_scan_start = start
                    self._static_search_signature = static_key
                    self.static_search_status_var.set(
                        f"Found {count} blob occurrences in {windows} qualifying windows.")
            if "temporal" in outcome:
                graph, result = outcome["temporal"]
                if geometry_key != self._temporal_geometry_key():
                    self._temporal_search = None
                    self.temporal_search_status_var.set("Settings changed — press Search again")
                else:
                    self._temporal_search = _TemporalSearchState(
                        geometry_key=geometry_key, kinds_key=kinds_key, eval_key=eval_key,
                        reclassifiable=reclassifiable, start=start, end=end,
                        graph=graph, result=result, params=params,
                    )
                    self._update_temporal_status()
                    try:
                        self._save_config_to_default_path()
                        self._set_progress_popup("Saving temporal Search cache...", None)
                        self._save_temporal_search_cache()
                    except (OSError, ValueError, TypeError) as exc:
                        self.set_status(f"Temporal Search completed but could not save cache: {exc}")
            self._refresh_search_glows()
            self.redraw_current_frame()
            if "temporal" in outcome and not self.outlier_section_revealed:
                self._set_workflow_stage("analysis")
            if self._temporal_search is not None and "temporal" in outcome:
                self.set_status("Temporal merges: " + self._temporal_summary_text(
                    self._temporal_search.result, detailed=True), auto_clear=False)
            else:
                self.set_status("Optional filter search completed")

        self._with_progress_popup(title, "Searching...", work, _finish_search)

    def _toggle_single_blob_smoothing_params(self):
        if self.single_blob_smoothing_enabled_var.get():
            self.single_blob_smoothing_params_frame.pack(fill="x", pady=(0, 4))
        else:
            self.single_blob_smoothing_params_frame.pack_forget()

    def _toggle_additional_outlier_params(self):
        if not hasattr(self, "additional_outlier_params_frame"):
            return
        if self.additional_outlier_enabled_var.get():
            if not self.additional_outlier_params_frame.winfo_ismapped():
                self.additional_outlier_params_frame.pack(fill="x", padx=0, pady=(0, 4))
        else:
            self.additional_outlier_params_frame.pack_forget()

    def _active_additional_outlier_set(self) -> dict:
        if not self.additional_outlier_sets:
            self.additional_outlier_sets = [self._default_additional_outlier_set()]
        self.additional_outlier_active_set_idx = max(
            0, min(self.additional_outlier_active_set_idx, len(self.additional_outlier_sets) - 1)
        )
        idx = self.additional_outlier_active_set_idx
        self.additional_outlier_sets[idx] = self._normalize_additional_outlier_set(
            self.additional_outlier_sets[idx]
        )
        return self.additional_outlier_sets[idx]

    def _sync_additional_outlier_vars_to_set(self):
        item = self._active_additional_outlier_set()
        item["enabled"] = bool(self.additional_outlier_enabled_var.get())
        item["frame_start"] = int(self.additional_outlier_frame_start_var.get())
        item["frame_end"] = int(self.additional_outlier_frame_end_var.get())
        item["segmentation_mode"] = str(self.additional_outlier_segmentation_mode_var.get())
        item["bg_method"] = str(self.additional_outlier_bg_method_var.get())
        item["threshold"] = int(self.additional_outlier_threshold_var.get())
        item["dark_threshold"] = int(self.additional_outlier_dark_threshold_var.get())
        item["bright_threshold"] = int(self.additional_outlier_bright_threshold_var.get())
        item["diff_threshold"] = int(self.additional_outlier_diff_threshold_var.get())
        item["min_area"] = int(self.additional_outlier_min_area_var.get())

    def _sync_additional_outlier_set_to_vars(self):
        item = self._active_additional_outlier_set()
        self.additional_outlier_enabled_var.set(bool(item.get("enabled", False)))
        self.additional_outlier_frame_start_var.set(int(item.get("frame_start", 0)))
        self.additional_outlier_frame_end_var.set(int(item.get("frame_end", -1)))
        self.additional_outlier_segmentation_mode_var.set(item.get("segmentation_mode", "background_diff"))
        self.additional_outlier_bg_method_var.set(item.get("bg_method", "median"))
        self.additional_outlier_threshold_var.set(int(item.get("threshold", 50)))
        self.additional_outlier_dark_threshold_var.set(int(item.get("dark_threshold", 50)))
        self.additional_outlier_bright_threshold_var.set(int(item.get("bright_threshold", 200)))
        self.additional_outlier_diff_threshold_var.set(int(item.get("diff_threshold", 50)))
        self.additional_outlier_min_area_var.set(int(item.get("min_area", 10)))

    def _on_additional_outlier_var_changed(self, *_):
        if self._switching_additional_outlier_set or self._applying_config:
            return
        self._sync_additional_outlier_vars_to_set()
        self.additional_outlier_revision += 1
        self.current_mask_cache.clear()
        self._refresh_additional_outlier_controls()
        self._mark_analysis_stale(redraw=True)
        if self._additional_outlier_needs_background():
            self._schedule_background_rebuild(delay_ms=250)

    def _refresh_additional_outlier_controls(self):
        if not hasattr(self, "additional_outlier_bg_controls_frame"):
            return
        mode = self.additional_outlier_segmentation_mode_var.get()
        for frame in (
            self.additional_outlier_bg_controls_frame,
            self.additional_outlier_threshold_controls_frame,
            self.additional_outlier_dark_hybrid_threshold_controls_frame,
            self.additional_outlier_bright_hybrid_threshold_controls_frame,
        ):
            frame.pack_forget()
        if _mode_requires_background(mode):
            self.additional_outlier_bg_controls_frame.pack(fill="x")
        if mode == "dark_region_and_background_diff":
            self.additional_outlier_dark_hybrid_threshold_controls_frame.pack(fill="x")
        elif mode == "bright_region_and_background_diff":
            self.additional_outlier_bright_hybrid_threshold_controls_frame.pack(fill="x")
        else:
            self.additional_outlier_threshold_controls_frame.pack(fill="x")

    def _update_additional_outlier_set_selector_values(self):
        if self.additional_outlier_set_combo is not None:
            self.additional_outlier_set_combo.configure(
                values=[f"Set {i + 1}" for i in range(len(self.additional_outlier_sets))]
            )

    def _update_additional_outlier_set_selector(self):
        if self.additional_outlier_set_combo is not None:
            self.additional_outlier_set_combo.set(f"Set {self.additional_outlier_active_set_idx + 1}")

    def _on_additional_outlier_set_selected(self, choice: str):
        try:
            idx = int(choice.split()[-1]) - 1
        except (ValueError, IndexError):
            return
        if 0 <= idx < len(self.additional_outlier_sets) and idx != self.additional_outlier_active_set_idx:
            self._switch_additional_outlier_set(idx)

    def _switch_additional_outlier_set(self, new_idx: int):
        if new_idx == self.additional_outlier_active_set_idx:
            return
        self._sync_additional_outlier_vars_to_set()
        self._switching_additional_outlier_set = True
        try:
            self.additional_outlier_active_set_idx = new_idx
            self._sync_additional_outlier_set_to_vars()
            self._update_additional_outlier_set_selector()
        finally:
            self._switching_additional_outlier_set = False
        self._refresh_additional_outlier_controls()
        self.additional_outlier_revision += 1
        self.current_mask_cache.clear()
        self._toggle_additional_outlier_params()
        self._mark_analysis_stale(redraw=True)

    def _add_additional_outlier_set(self):
        if len(self.additional_outlier_sets) >= 8:
            return
        self._sync_additional_outlier_vars_to_set()
        self.additional_outlier_sets.append(self._default_additional_outlier_set())
        self._update_additional_outlier_set_selector_values()
        self._switch_additional_outlier_set(len(self.additional_outlier_sets) - 1)

    def _remove_additional_outlier_set(self):
        if len(self.additional_outlier_sets) <= 1:
            return
        old_idx = self.additional_outlier_active_set_idx
        self.additional_outlier_sets.pop(old_idx)
        new_idx = min(old_idx, len(self.additional_outlier_sets) - 1)
        self._update_additional_outlier_set_selector_values()
        self._switching_additional_outlier_set = True
        try:
            self.additional_outlier_active_set_idx = new_idx
            self._sync_additional_outlier_set_to_vars()
            self._update_additional_outlier_set_selector()
        finally:
            self._switching_additional_outlier_set = False
        self._refresh_additional_outlier_controls()
        self.additional_outlier_revision += 1
        self.current_mask_cache.clear()
        self._toggle_additional_outlier_params()
        self._mark_analysis_stale(redraw=True)

    def _additional_outlier_background_methods(self) -> set[str]:
        return {
            str(item.get("bg_method", "median"))
            for item in self.additional_outlier_sets
            if item.get("enabled", False) and _mode_requires_background(
                str(item.get("segmentation_mode", "background_diff"))
            )
        }

    def _additional_outlier_needs_background(self) -> bool:
        return bool(self._additional_outlier_background_methods())

    def _active_roi_geometry(self) -> Optional[list[list[int]]]:
        """The selected polygon is editable even when disabled or outside its range."""
        if self.frame_bgr_cache is None or not self.roi_edit_var.get():
            return None
        return self.roi_sets[self.roi_active_set_idx]["points"] or None

    def _start_roi_drag(self, ix: float, iy: float) -> bool:
        points = self._active_roi_geometry()
        if not points:
            return False
        distances = [math.hypot(ix - x, iy - y) * self.scale for x, y in points]
        nearest = min(range(len(points)), key=distances.__getitem__)
        if distances[nearest] <= ROI_HANDLE_GRAB_CANVAS_PX:
            self._roi_drag_mode = "vertex"
            self._roi_drag_vertex = nearest
        elif cv2.pointPolygonTest(np.asarray(points, dtype=np.int32), (ix, iy), False) >= 0:
            self._roi_drag_mode = "move"
            self._roi_drag_vertex = None
        else:
            return False
        self._roi_drag_anchor = (ix, iy)
        self._roi_drag_points = [list(point) for point in points]
        return True

    def _update_roi_drag(self, ix: float, iy: float) -> None:
        if self._roi_drag_mode is None or self._roi_drag_anchor is None or self.reader is None:
            return
        ax, ay = self._roi_drag_anchor
        dx, dy = round(ix - ax), round(iy - ay)
        points = [list(point) for point in self._roi_drag_points]
        width, height = self.reader.width, self.reader.height
        if self._roi_drag_mode == "move":
            dx = max(-min(p[0] for p in points), min(width - 1 - max(p[0] for p in points), dx))
            dy = max(-min(p[1] for p in points), min(height - 1 - max(p[1] for p in points), dy))
            points = [[x + dx, y + dy] for x, y in points]
        else:
            index = self._roi_drag_vertex
            x, y = points[index]
            points[index] = [max(0, min(width - 1, x + dx)), max(0, min(height - 1, y + dy))]
        roi = self.roi_sets[self.roi_active_set_idx]
        if points != roi["points"]:
            roi["points"] = points
            self._on_roi_geometry_changed()

    def _draw_roi_canvas(self):
        self.canvas.delete("roi_editor")
        if self.frame_bgr_cache is None:
            return
        color = "#%02x%02x%02x" % tuple(reversed(ROI_COLOR))
        active_any = False
        for index, roi in enumerate(self.roi_sets):
            active = roi_is_active(roi, self.current_frame)
            editing = self.roi_edit_var.get() and index == self.roi_active_set_idx
            if not roi["points"] or not (active or editing):
                continue
            active_any = active_any or active
            coordinates = [(x * self.scale + self.offset_x, y * self.scale + self.offset_y) for x, y in roi["points"]]
            flat = [value for point in coordinates for value in point]
            self.canvas.create_polygon(*flat, fill="", outline=color, width=3 if editing else 2,
                                       dash=() if active else (5, 4), tags="roi_editor")
            if editing:
                radius = ROI_HANDLE_RADIUS_CANVAS_PX
                for x, y in coordinates:
                    self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                            fill="white", outline=color, width=2, tags="roi_editor")
        if active_any and self.roi_reverse_var.get():
            self.canvas.create_text(12, 12, text="REV", anchor="nw", fill=color, tags="roi_editor")

    def _parallel_read_frames(
        self,
        frame_indices: list[int],
        on_progress=None,
    ) -> dict[int, np.ndarray]:
        """Read a (typically scattered) list of frame indices using several
        decoder handles in parallel. Each handle performs the exact same
        cap.set()+cap.read() the reader's own seek path uses, so decoded
        content matches serial reads; any handle that fails falls back to
        the reader's robust, cache-backed decode for that single frame.
        """
        results: dict[int, np.ndarray] = {}
        if not frame_indices:
            return results

        thread_caps: list[cv2.VideoCapture] = []
        caps_lock = threading.Lock()
        local = threading.local()

        def _get_local_cap() -> "Optional[cv2.VideoCapture]":
            cap = getattr(local, "cap", None)
            if cap is None:
                cap = cv2.VideoCapture(self.reader.video_path)
                local.cap = cap
                with caps_lock:
                    thread_caps.append(cap)
            return cap

        def _read_one(frame_idx: int) -> "tuple[int, np.ndarray]":
            frame_idx = self.reader._clamp_frame_idx(frame_idx)
            cap = _get_local_cap()
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    return frame_idx, frame
            return frame_idx, self.reader.read_bgr(frame_idx)

        prev_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=_EXPORT_WORKERS) as executor:
                futures = [executor.submit(_read_one, fidx) for fidx in frame_indices]
                total = len(futures)
                done = 0
                for future in as_completed(futures):
                    if self._progress_cancel:
                        break
                    fidx, frame = future.result()
                    results[fidx] = frame
                    done += 1
                    if on_progress is not None and (done == 1 or done % 5 == 0 or done == total):
                        on_progress(done, total)
        finally:
            cv2.setNumThreads(prev_threads)
            for cap in thread_caps:
                cap.release()
        return results

    def _parallel_read_background_stack(
        self,
        frame_indices: list[int],
        on_progress=None,
    ) -> "Optional[np.ndarray]":
        """Decode background samples directly into one bounded uint8 stack.

        Unlike _parallel_read_frames, this never retains both a frame dict and
        a second stacked copy.  Decoder count is also kept conservative because
        several simultaneous 4K H.264 decoders can exhaust codec resources.
        """
        if not frame_indices:
            return None

        shape = (len(frame_indices), int(self.reader.height), int(self.reader.width), 3)
        try:
            stack = np.empty(shape, dtype=np.uint8)
        except (MemoryError, ValueError) as exc:
            mib = np.prod(shape, dtype=np.int64) / (1024 * 1024)
            raise RuntimeError(
                f"Unable to reserve {mib:.0f} MiB for background samples."
            ) from exc

        total = len(frame_indices)
        completed = 0
        progress_lock = threading.Lock()

        def _read_chunk(samples):
            nonlocal completed
            cap = cv2.VideoCapture(self.reader.video_path)
            next_frame = None
            prefer_forward = None

            def seek_read(frame_idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                return frame if ok and frame is not None and frame.size > 0 else None

            def forward_read(frame_idx):
                for _ in range(next_frame, frame_idx):
                    if self._progress_cancel or not cap.grab():
                        return None
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    return None
                if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (frame_idx + 1)) > 0.5:
                    return None
                return frame

            try:
                for slot, frame_idx in samples:
                    if self._progress_cancel:
                        return
                    frame = None
                    if cap.isOpened():
                        can_forward = (
                            next_frame is not None
                            and 0 <= frame_idx - next_frame < BACKGROUND_FORWARD_MAX_FRAMES
                            and abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - next_frame) <= 0.5
                        )
                        if can_forward and prefer_forward is not False:
                            started = time.perf_counter()
                            forward = forward_read(frame_idx)
                            forward_seconds = time.perf_counter() - started
                            if self._progress_cancel:
                                return
                            if prefer_forward is None:
                                # Calibrate on the same target frame.  Keep the
                                # seek result and enable forward decoding only
                                # when its pixels agree and it is faster.
                                started = time.perf_counter()
                                frame = seek_read(frame_idx)
                                seek_seconds = time.perf_counter() - started
                                prefer_forward = (
                                    forward is not None and frame is not None
                                    and np.array_equal(forward, frame)
                                    and forward_seconds < seek_seconds
                                )
                            else:
                                frame = forward
                                if frame is None:
                                    prefer_forward = False
                            del forward
                        if frame is None:
                            frame = seek_read(frame_idx)
                        next_frame = frame_idx + 1 if frame is not None else None
                    if frame is None:
                        frame = self.reader.read_bgr(frame_idx)
                    if frame.shape != shape[1:]:
                        raise RuntimeError(
                            f"Unexpected frame size while reading background sample: {frame.shape}"
                        )
                    stack[slot] = frame
                    del frame
                    with progress_lock:
                        completed += 1
                        if on_progress is not None and (
                            completed == 1 or completed % 5 == 0 or completed == total
                        ):
                            on_progress(completed, total)
            finally:
                cap.release()

        prev_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=_BACKGROUND_READ_WORKERS) as executor:
                # Consecutive ranges keep each decoder moving forward while
                # original slots preserve the requested sample order.
                samples = sorted(
                    ((slot, self.reader._clamp_frame_idx(idx)) for slot, idx in enumerate(frame_indices)),
                    key=lambda item: item[1],
                )
                chunk_size = max(1, math.ceil(total / _BACKGROUND_READ_WORKERS))
                futures = [
                    executor.submit(_read_chunk, samples[start:start + chunk_size])
                    for start in range(0, total, chunk_size)
                ]
                for future in as_completed(futures):
                    future.result()
        finally:
            cv2.setNumThreads(prev_threads)

        if self._progress_cancel:
            return None
        if completed != len(frame_indices):
            raise RuntimeError("Failed to read all sampled frames for background extraction.")
        return stack

    def _parallel_segment_frames(
        self,
        frame_indices: list[int],
        cfg: _SegConfig,
        on_progress=None,
    ) -> dict[int, list[BlobMetrics]]:
        """Decode and segment scattered frames in one bounded pipeline.

        Only contours/metrics are retained.  Full-resolution frames and masks
        are released inside each worker instead of all being collected first.
        """
        if not frame_indices:
            return {}

        results: dict[int, list[BlobMetrics]] = {}
        total = len(frame_indices)
        completed = 0
        result_lock = threading.Lock()

        def _read_and_segment_chunk(samples: list[int]) -> None:
            nonlocal completed
            cap = cv2.VideoCapture(self.reader.video_path)
            next_frame = None
            prefer_forward = None

            def seek_read(frame_idx: int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                return frame if ok and frame is not None and frame.size > 0 else None

            def forward_read(frame_idx: int):
                for _ in range(next_frame, frame_idx):
                    if self._progress_cancel or not cap.grab():
                        return None
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    return None
                if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (frame_idx + 1)) > 0.5:
                    return None
                return frame

            try:
                for frame_idx in samples:
                    if self._progress_cancel:
                        return
                    frame = None
                    if cap.isOpened():
                        can_forward = (
                            next_frame is not None
                            and 0 <= frame_idx - next_frame < ANALYSIS_FORWARD_MAX_FRAMES
                            and abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - next_frame) <= 0.5
                        )
                        if can_forward and prefer_forward is not False:
                            started = time.perf_counter()
                            forward = forward_read(frame_idx)
                            forward_seconds = time.perf_counter() - started
                            if self._progress_cancel:
                                return
                            if prefer_forward is None:
                                # Validate the first candidate against the exact
                                # seek path before using forward decoding for the
                                # rest of this decoder chunk.
                                started = time.perf_counter()
                                frame = seek_read(frame_idx)
                                seek_seconds = time.perf_counter() - started
                                prefer_forward = (
                                    forward is not None and frame is not None
                                    and np.array_equal(forward, frame)
                                    and forward_seconds < seek_seconds
                                )
                            else:
                                frame = forward
                                if frame is None:
                                    prefer_forward = False
                            del forward
                        if frame is None:
                            frame = seek_read(frame_idx)
                        next_frame = frame_idx + 1 if frame is not None else None
                    if frame is None:
                        frame = self.reader.read_bgr(frame_idx)
                    blobs = _segment_blobs_with_config(frame, frame_idx, cfg)
                    del frame
                    with result_lock:
                        results[frame_idx] = blobs
                        completed += 1
                        if on_progress is not None and (
                            completed == 1 or completed % 5 == 0 or completed == total
                        ):
                            on_progress(completed, total)
            finally:
                cap.release()

        prev_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=_ANALYSIS_WORKERS) as executor:
                samples = [self.reader._clamp_frame_idx(fid) for fid in frame_indices]
                samples.sort()
                chunk_size = max(1, math.ceil(total / _ANALYSIS_WORKERS))
                futures = [
                    executor.submit(_read_and_segment_chunk, samples[start:start + chunk_size])
                    for start in range(0, total, chunk_size)
                ]
                for future in as_completed(futures):
                    future.result()
        finally:
            cv2.setNumThreads(prev_threads)
        return results

    def compute_background(self):
        if self._background_building:
            return
        if self._progress_busy:
            self._schedule_background_rebuild(delay_ms=300)
            return
        # Pause speculative preview decoding before opening background readers.
        self._prefetch_cancel.set()
        if self.prefetch_after_id is not None:
            self.after_cancel(self.prefetch_after_id)
            self.prefetch_after_id = None
        if self.reader is not None:
            self.reader.clear_cache()
        self._background_building = True
        try:
            started = self._compute_background_impl()
        except Exception:
            self._background_building = False
            raise
        if not started:
            self._background_building = False

    def _compute_background_impl(self):
        if self.reader is None or self.frame_count <= 0:
            return
        sample_count = self._background_sample_count()
        if sample_count <= 0:
            return
        bg_start = max(0, int(self.background_frame_start_var.get()))
        bg_end = self._resolved_frame_end(int(self.background_frame_end_var.get()))
        idx = np.linspace(bg_start, bg_end, sample_count, dtype=int)
        selected_method = self.bg_method_var.get()
        if selected_method not in BACKGROUND_METHODS:
            selected_method = "median"
            self.bg_method_var.set(selected_method)
        cache_metadata_path = self._background_cache_metadata_path()
        existing_methods = self._background_cache_current_methods()
        required_methods = {selected_method} | self._additional_outlier_background_methods()
        # The expensive part is decoding the sampled frames.  On the first
        # build, reuse that one bounded uint8 stack for median/max/min so
        # switching methods later never requires a second video decode.
        if not existing_methods:
            methods_to_compute = list(BACKGROUND_METHODS)
        else:
            methods_to_compute = [method for method in BACKGROUND_METHODS if method in required_methods - existing_methods]
        method_paths = {
            method: self._method_background_path(method)
            for method in methods_to_compute
        }
        computed_box: list[dict[str, np.ndarray]] = [{}]
        error_box: list[Optional[Exception]] = [None]

        if (
            self.background_bgr is not None
            and self.active_background_method == selected_method
            and self._background_cache_is_current(selected_method)
            and not (required_methods - existing_methods)
        ):
            self.background_ready_var.set(f"background: cached ({selected_method}, samples={sample_count})")
            self.set_status(f"Background already available: {selected_method}")
            return

        if not methods_to_compute and self._activate_cached_background(selected_method):
            self.set_status(f"Background loaded from cache: {selected_method}")
            return

        method_label = "/".join(methods_to_compute)
        self.set_status(
            f"Building {method_label} background / samples={sample_count}",
            auto_clear=False,
        )

        def work():
            try:
                total = len(idx)
                self._set_progress_popup(
                    f"Reading background frames: 0/{total} (0%)",
                    0,
                )

                def _report(n: int, n_total: int) -> None:
                    percent = 100.0 * n / max(1, n_total)
                    self._set_progress_popup(
                        f"Reading background frames: {n}/{n_total} ({percent:.0f}%)",
                        percent,
                    )

                stack = self._parallel_read_background_stack(idx.tolist(), on_progress=_report)
                if self._progress_cancel or stack is None:
                    return False
                print(f"[Segmentation] Background extraction: decoded {total} frames", flush=True)
                frames_complete = f"Frames loaded: {total}/{total} (100%)"
                self._set_progress_popup(
                    f"{frames_complete}\nPreparing background calculation...",
                    None,
                )

                # Invalidate the old manifest immediately before replacing its
                # image set, so a partial disk-write can never look like a valid cache.
                invalid_metadata_path = cache_metadata_path + ".tmp"
                with open(invalid_metadata_path, "w", encoding="utf-8") as file:
                    json.dump({"version": 0}, file)
                os.replace(invalid_metadata_path, cache_metadata_path)

                computed: dict[str, np.ndarray] = {}
                for method in methods_to_compute:
                    self._set_progress_popup(
                        f"{frames_complete}\nComputing {method} background...",
                        None,
                    )
                    background = _compute_background_method(stack, method)
                    method_path = method_paths[method]
                    self._set_progress_popup(
                        f"{frames_complete}\nSaving background_{method}.png...",
                        None,
                    )
                    if not cv2.imwrite(method_path, background):
                        raise RuntimeError(f"Failed to save background: {method_path}")
                    print(f"[Segmentation] Background extraction: saved {method_path}", flush=True)
                    computed[method] = background
                computed_box[0] = computed
                self._set_progress_popup(
                    f"{frames_complete}\nFinalizing background cache...",
                    None,
                )
            except Exception as e:
                error_box[0] = e
                return False
            return True

        def _finish_background(ok):
            try:
                if not ok:
                    if error_box[0] is not None:
                        messagebox.showerror("Error", str(error_box[0]))
                    self.set_status("", auto_clear=False)
                    return

                try:
                    computed = computed_box[0]
                    background = computed.get(selected_method)
                    if background is None and self.background_bgr is not None and self.active_background_method == selected_method:
                        background = self.background_bgr
                    if background is None:
                        background = cv2.imread(self._method_background_path(selected_method), cv2.IMREAD_COLOR)
                    if background is None:
                        raise RuntimeError(f"Failed to compute background: {selected_method}")
                    self._write_background_cache_metadata(sample_count, idx, existing_methods | set(computed.keys()))
                    active_method = selected_method
                    shutil.copyfile(self._method_background_path(active_method), self._default_background_path())
                    print("[Segmentation] Background extraction: background cache finalized", flush=True)
                    self.background_bgr = background
                    self.active_background_method = active_method
                except Exception as exc:
                    messagebox.showerror("Error", str(exc))
                    self.set_status("", auto_clear=False)
                    return

                self._invalidate_analysis_for_background_change()
                self.background_ready_var.set(f"background: ready ({active_method}, samples={sample_count})")
                if self.bg_preview_button is not None:
                    self.bg_preview_button.configure(state="normal")
                self._set_workflow_stage("analysis")
                self.redraw_current_frame()
                self.set_status(f"Background computed and cached: {active_method}")
            finally:
                self._background_building = False
                if self._restore_temporal_after_background:
                    self._restore_temporal_after_background = False
                    self._restore_temporal_search_cache()
                    self._update_temporal_status()
                    self._refresh_search_glows()
                    self._set_workflow_stage("all")

        self._with_progress_popup("Background extraction", f"Preparing {method_label} background...", work, _finish_background)
        return True

    def _segment_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> tuple[np.ndarray, list[BlobMetrics]]:
        key = (frame_idx, self._segmentation_signature())
        cached = self.current_mask_cache.get(key)
        if cached is not None:
            mask, blobs = cached
            return mask.copy(), [self._clone_blob(b) for b in blobs]

        cfg = self._capture_seg_config()
        clean_mask, blobs = _segment_with_config(frame_bgr, frame_idx, cfg)
        cloned = [self._clone_blob(b) for b in blobs]
        self.current_mask_cache[key] = (clean_mask.copy(), cloned)
        while len(self.current_mask_cache) > MASK_CACHE_MAX_ENTRIES:
            self.current_mask_cache.pop(next(iter(self.current_mask_cache)))
        return clean_mask, [self._clone_blob(b) for b in cloned]

    @staticmethod
    def _clone_blob(blob: BlobMetrics) -> BlobMetrics:
        return BlobMetrics(
            frame=blob.frame,
            blob_index=blob.blob_index,
            contour=blob.contour.copy(),
            center_x=blob.center_x,
            center_y=blob.center_y,
            area=blob.area,
            bbox_area=blob.bbox_area,
            w=blob.w,
            h=blob.h,
            area_outlier=blob.area_outlier,
            bbox_area_outlier=blob.bbox_area_outlier,
            width_outlier=blob.width_outlier,
            height_outlier=blob.height_outlier,
            low_obb_aspect_outlier=blob.low_obb_aspect_outlier,
            persistent_static_outlier=blob.persistent_static_outlier,
            temporal_merge_outlier=blob.temporal_merge_outlier,
            temporal_merge_animals=blob.temporal_merge_animals,
            temporal_merge_level=blob.temporal_merge_level,
            manual_outlier=blob.manual_outlier,
            is_crossing=blob.is_crossing,
            result_obb_count=blob.result_obb_count,
            result_obb_coverage=blob.result_obb_coverage,
            result_single=blob.result_single,
            result_missing=blob.result_missing,
            result_area_outlier=blob.result_area_outlier,
            result_obb_indices=blob.result_obb_indices,
            result_blob_area=blob.result_blob_area,
            result_generated=blob.result_generated,
            result_protected=blob.result_protected,
            result_source_index=blob.result_source_index,
            result_source_frame=blob.result_source_frame,
        )

    def _compute_iqr_stats(self, all_blobs: list[BlobMetrics]) -> dict[str, tuple[float, float, float, float]]:
        # Only Area is used for IQR outlier judgment; BBox/Width/Height IQR judgment is not used.
        values = [b.area for b in _restore_result_inputs(all_blobs) if not b.manual_outlier]
        if not values:
            return {"area": (0.0, 0.0, 0.0, 0.0)}
        q1, median, q3 = np.percentile(
            np.asarray(values, dtype=np.float64), (25, 50, 75)
        )
        q1, median, q3 = float(q1), float(median), float(q3)
        iqr = float(q3 - q1)
        return {"area": (q1, median, q3, iqr)}

    @staticmethod
    def _normalized_area_stat(stat: tuple[float, ...]) -> tuple[float, float, float, float]:
        if len(stat) >= 4:
            q1, median, q3, iqr = stat[:4]
            return float(q1), float(median), float(q3), float(iqr)
        q1, q3, iqr = stat[:3]
        return float(q1), 0.5 * (float(q1) + float(q3)), float(q3), float(iqr)

    def _current_area_bounds(self, stats: Optional[dict[str, tuple[float, ...]]] = None) -> tuple[float, float]:
        if self.area_outlier_method_var.get() == "absolute":
            lo = float(self.area_absolute_min_var.get())
            hi = float(self.area_absolute_max_var.get())
            if lo > hi:
                raise ValueError("Minimum area must be less than or equal to Maximum area.")
            return lo, hi

        if stats is None:
            stats = self.analysis_iqr_stats
        if not stats or "area" not in stats:
            raise ValueError("Sampled-frame analysis has not been run.")
        q1, median, q3, iqr = self._normalized_area_stat(stats["area"])
        lo = min(median, q1 - float(self.area_iqr_min_var.get()) * iqr)
        hi = max(median, q3 + float(self.area_iqr_max_var.get()) * iqr)
        return lo, hi

    def _needs_area_analysis(self) -> bool:
        """Absolute bounds come from the sliders; only IQR has to be fitted."""
        return self.area_outlier_method_var.get() != "absolute"

    def _bounds_from_fixed_stats(self, stats: Optional[dict[str, tuple[float, ...]]] = None) -> dict[str, tuple[float, float]]:
        if stats is None:
            stats = self.analysis_iqr_stats
        if self.area_outlier_method_var.get() != "absolute" and (not stats or "area" not in stats):
            return {}
        return {"area": self._current_area_bounds(stats)}

    def _refresh_bounds_from_fixed_stats(self, show_error: bool = False):
        if self.analysis_iqr_stats or not self._needs_area_analysis():
            try:
                self.analysis_bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats)
            except ValueError as exc:
                self.analysis_bounds = {}
                self.set_status(str(exc))
                if show_error:
                    messagebox.showerror("Error", str(exc))
                return
            if self.analysis_blobs_by_frame:
                self.analysis_blobs_by_frame = {
                    int(fid): self._classify_blobs(blobs, self.analysis_bounds, frame_idx=int(fid))
                    for fid, blobs in self.analysis_blobs_by_frame.items()
                }
                self.analysis_outlier_frames = sorted(
                    fid for fid, blobs in self.analysis_blobs_by_frame.items()
                    if any(blob.is_crossing for blob in blobs)
                )
                self._update_analysis_dependent_buttons()
            if self.analysis_signature is not None and not self.analysis_is_stale:
                label = "Absolute area bounds" if self.area_outlier_method_var.get() == "absolute" else "Area IQR stats; live IQR multipliers"
                self.analysis_ready_var.set(f"analysis: ready ({label})")
        self._redraw_area_iqr_boxplot()
        self.redraw_current_frame()

    def _classify_blobs(
        self,
        blobs: list[BlobMetrics],
        bounds: Optional[dict[str, tuple[float, float]]] = None,
        *,
        clone: bool = True,
        result_match=_UNSET,
        frame_idx: Optional[int] = None,
        image_shape: Optional[tuple] = None,
    ) -> list[BlobMetrics]:
        """Classify one frame's blobs.

        Omitting ``result_match`` uses the live import; passing None means
        disabled, so worker threads keep the snapshot they were given.
        """
        if bounds is None:
            bounds = self.analysis_bounds
        if result_match is _UNSET:
            result_match = self._result_match
        inputs = _restore_result_inputs(blobs)
        out = [self._clone_blob(b) for b in inputs] if clone else inputs
        # Rebuild imported rectangles against real segmentation, never against
        # rectangles left by a prior classification or a now-disabled import.
        had_missing = any(b.result_generated or b.result_missing or b.result_source_frame is not None for b in blobs)
        out[:] = [b for b in out if not (b.result_generated or b.result_missing)]
        area_bounds = bounds.get("area", (-float("inf"), float("inf"))) if bounds else (-float("inf"), float("inf"))

        for b in out:
            # Avoid stale temporal states from earlier classification passes.
            b.temporal_merge_outlier = False
            b.temporal_merge_animals = 0
            b.temporal_merge_level = 0
            b.area_outlier = not inside_bounds(b.area, area_bounds)
            b.bbox_area_outlier = False
            b.width_outlier = False
            b.height_outlier = False
            b.result_obb_count = 0
            b.result_obb_coverage = 0.0
            b.result_single = False
            b.result_area_outlier = False
            b.result_blob_area = None
            b.result_obb_indices = ()
            b.result_protected = False
            b.result_source_index = None
            b.result_source_frame = None
            b.is_crossing = bool(b.manual_outlier or b.area_outlier)
        if result_match is not None:
            if image_shape is None and self.reader is not None:
                image_shape = (self.reader.height, self.reader.width)
            self._rescue_single_animal_outliers(
                out, result_match, frame_idx=frame_idx, image_shape=image_shape,
            )
        # Mirror downstream Stage-1 MIN_ASPECT and min_area OBB geometry:
        # equality with 1.1 is accepted.
        from main.obb_fitting import fit_obb
        for b in out:
            b.low_obb_aspect_outlier = False
            if b.result_generated or b.result_missing or b.manual_outlier:
                continue
            contour = np.asarray(b.contour, dtype=np.float32).reshape(-1, 1, 2)
            if contour.shape[0] < 3:
                continue
            (_center, (w, h), _angle), _axis = fit_obb(
                contour.reshape(-1, 2), "min_area",
            )
            if min(w, h) > 1e-6:
                b.low_obb_aspect_outlier = max(w, h) / min(w, h) < DEFAULT_MIN_OBB_ASPECT
                if b.low_obb_aspect_outlier:
                    b.is_crossing = True
        if had_missing or any(b.result_generated for b in out):
            for index, b in enumerate(out):
                b.blob_index = index
        return out

    @staticmethod
    def _rescue_single_animal_outliers(
        blobs: list[BlobMetrics], result_match: "_ResultMatchConfig", *,
        frame_idx: Optional[int] = None, image_shape: Optional[tuple] = None,
    ) -> None:
        """Judge OBB foreground totals, then expand outliers with protection.

        Only real segment masks seed classification. Generated rectangle pixels
        never become evidence for another OBB or alter its measured area.
        Full-box expansion requires the triggering outlier mask to cover at most
        half of the visible rasterized OBB.
        """
        by_frame: dict[int, list[BlobMetrics]] = {}
        for b in blobs:
            by_frame.setdefault(int(b.frame), []).append(b)
        if frame_idx is not None:
            by_frame.setdefault(int(frame_idx), [])
        output = []
        for frame_idx, frame_blobs in by_frame.items():
            obbs = result_match.obbs_by_frame.get(frame_idx)
            if obbs is None or image_shape is None:
                output.extend(frame_blobs)
                continue
            for index, b in enumerate(frame_blobs):
                b.result_source_index = index
            regions = _match_result_obbs_to_blobs(frame_blobs, frame_idx, result_match, image_shape)
            areas = _result_blob_pixel_areas(frame_blobs, obbs, image_shape)
            owners = [[i for i, b in enumerate(frame_blobs) if ri in b.result_obb_indices] for ri in range(len(obbs))]
            for b in frame_blobs:
                if b.result_obb_count == 1:
                    b.result_blob_area = areas[b.result_obb_indices[0]]
                if b.manual_outlier:
                    continue
                if b.result_obb_count == 1 and b.result_obb_coverage >= result_match.min_coverage:
                    b.result_area_outlier = not inside_bounds(b.result_blob_area, result_match.area_bounds)
                    b.result_single = b.area_outlier and not b.result_area_outlier
                    b.is_crossing = b.result_area_outlier
            targets = {ri for ri, area in enumerate(areas)
                       if not owners[ri] or not inside_bounds(area, result_match.area_bounds)}
            expanded_targets = set()
            while True:
                for ri in targets:
                    for index in owners[ri]:
                        frame_blobs[index].is_crossing = True
                        frame_blobs[index].result_single = False
                healthy = [b for b in frame_blobs if b.result_obb_indices and not b.is_crossing]
                protected = _draw_blob_union_mask(image_shape, healthy)
                rejected = _draw_blob_union_mask(image_shape, [b for b in frame_blobs if b.is_crossing])
                # A trusted healthy mask wins conflicting duplicate/manual
                # pixels. Promotion propagates only through real source masks.
                seeds = rejected & cv2.bitwise_not(protected)
                for ri in targets - expanded_targets:
                    if owners[ri]:
                        target_outliers = _draw_blob_union_mask(
                            image_shape, [frame_blobs[index] for index in owners[ri]],
                        )
                    else:
                        target_outliers = seeds
                    if _result_region_covered_at_most_half(regions[ri], target_outliers):
                        expanded_targets.add(ri)
                additions = set()
                for ri, (x, y, mask) in enumerate(regions):
                    h, w = mask.shape
                    if mask.size and cv2.countNonZero(mask & seeds[y:y+h, x:x+w]) and \
                            _result_region_covered_at_most_half(regions[ri], seeds):
                        additions.add(ri)
                if additions <= targets:
                    break
                targets |= additions
                expanded_targets |= additions

            expanded = rejected.copy()
            for ri in expanded_targets:
                x, y, mask = regions[ri]
                h, w = mask.shape
                if mask.size:
                    expanded[y:y+h, x:x+w] |= mask
            for b in healthy:
                b.result_protected = bool(cv2.countNonZero(_draw_blob_union_mask(image_shape, [b]) & expanded))
            erased = expanded & cv2.bitwise_not(protected)
            sources = tuple(replace(b, contour=b.contour.copy(), result_source_frame=None) for b in frame_blobs)
            frame_output = []
            for b in frame_blobs:
                if not b.is_crossing and b.result_obb_indices:
                    frame_output.append(b)
                    continue
                x, y, w, h = cv2.boundingRect(b.contour)
                x0, y0, x1, y1 = max(0, x), max(0, y), min(image_shape[1], x+w), min(image_shape[0], y+h)
                if x0 >= x1 or y0 >= y1:
                    continue
                mask = np.zeros((y1-y0, x1-x0), np.uint8)
                cv2.drawContours(mask, [b.contour], -1, 255, cv2.FILLED, offset=(-x0, -y0))
                exclusion = protected[y0:y1, x0:x1] if b.is_crossing else erased[y0:y1, x0:x1]
                if not cv2.countNonZero(mask & exclusion):
                    frame_output.append(b)
                else:
                    frame_output.extend(_solid_region_blobs(b, mask & cv2.bitwise_not(exclusion), x0, y0))
            for ri in sorted(expanded_targets):
                template = _missing_result_obb_blob(obbs[ri], frame_idx, len(frame_output), image_shape)
                if template is None:
                    continue
                template.result_generated = True
                template.result_missing = not owners[ri]
                template.result_obb_indices = (ri,)
                template.result_blob_area = areas[ri]
                template.result_area_outlier = not inside_bounds(areas[ri], result_match.area_bounds)
                x, y, mask = regions[ri]
                h, w = mask.shape
                exclusion = protected[y:y+h, x:x+w]
                if not cv2.countNonZero(mask & exclusion):
                    frame_output.append(template)
                else:
                    frame_output.extend(_solid_region_blobs(template, mask & cv2.bitwise_not(exclusion), x, y))
            for b in frame_output:
                b.result_source_frame = sources
            output.extend(frame_output)
        blobs[:] = output

    def _analysis_frame_indices(self) -> list[int]:
        if self.reader is None:
            return []
        sample_count = max(10, int(self.analysis_sample_count_var.get()))
        start = max(0, int(self.training_frame_start_var.get()))
        end = self._resolved_frame_end(int(self.training_frame_end_var.get()))
        if start > end:
            return []
        range_count = end - start + 1
        sample_count = min(sample_count, range_count)
        return np.linspace(start, end, sample_count, dtype=int).tolist()

    def _run_analysis(self):
        if self.reader is None or self._progress_busy:
            return
        # Temporal merging is the preceding step when enabled. Require its
        # current graph before fitting area outlier limits from samples.
        if self.temporal_merge_enabled_var.get() and not self._temporal_search_is_current():
            messagebox.showwarning("Analysis", "Run Exclude temporal merges > Search before Analyze.")
            return
        if self._segmentation_needs_background() and self.background_bgr is None:
            messagebox.showerror("Error", "In this mode, please compute the background first.")
            return
        if self.area_outlier_method_var.get() == "absolute":
            try:
                self._current_area_bounds({})
            except ValueError as exc:
                messagebox.showerror("Error", str(exc))
                return

        frame_indices = self._analysis_frame_indices()
        if not frame_indices:
            return

        scope_text = f"{len(frame_indices)} sampled frames"
        self.set_status(f"Analyzing {scope_text}...", auto_clear=False)
        self._prefetch_cancel.set()
        self.reader.clear_cache()

        # Snapshot every Tk-backed setting before entering the worker thread.
        cfg = self._capture_seg_config()
        result_match = self._result_match
        temporal_state = self._temporal_search if self._temporal_search_is_current() else None
        area_method = self.area_outlier_method_var.get()
        area_iqr_min = float(self.area_iqr_min_var.get())
        area_iqr_max = float(self.area_iqr_max_var.get())
        absolute_bounds = (
            float(self.area_absolute_min_var.get()),
            float(self.area_absolute_max_var.get()),
        )

        result: dict[str, object] = {}

        def work():
            total_frames = len(frame_indices)
            start_time = time.perf_counter()
            self._set_progress_popup(
                f"Analyzing frames: 0/{total_frames} (0%)",
                0,
            )

            def _report_frame(n: int, n_total: int) -> None:
                percent = 100.0 * n / max(1, n_total)
                elapsed = max(1e-9, time.perf_counter() - start_time)
                fps = n / elapsed
                eta = max(0, n_total - n) / fps if fps > 0 else 0.0
                self._set_progress_popup(
                    f"Analyzing frames: {n}/{n_total} ({percent:.0f}%)\n"
                    f"speed={fps:.1f} fps / remaining={self._format_seconds(eta)}",
                    percent,
                )

            blobs_by_frame = self._parallel_segment_frames(
                frame_indices, cfg, on_progress=_report_frame
            )
            if self._progress_cancel:
                return False
            if len(blobs_by_frame) != total_frames:
                raise RuntimeError("Failed to analyze all sampled frames.")

            frames_complete = f"Frames analyzed: {total_frames}/{total_frames} (100%)"
            self._set_progress_popup(
                f"{frames_complete}\nComputing sampled-frame statistics...",
                None,
            )
            # Exclude temporal crossings (not unconfirmed candidates) from
            # the single-animal population used to calibrate Outlier Extraction. Position maps
            # are safe only when the Search digest matches the sample.
            from main.frame_scan import frame_blob_digest
            all_blobs = []
            for fid, blobs in blobs_by_frame.items():
                skip = ()
                if temporal_state is not None:
                    idx = int(fid) - int(temporal_state.start)
                    if not 0 <= idx < temporal_state.graph.frame_count:
                        raise RuntimeError("Temporal Search does not cover the analyzed frame range.")
                    if frame_blob_digest(blobs) != int(temporal_state.graph.digests[idx]):
                        raise RuntimeError(
                            "Temporal Search and Analyze use different blob geometry; "
                            "re-run Search with the current segmentation settings.")
                    skip = temporal_state.result.excluded.get(int(fid), ())
                all_blobs.extend(blob for i, blob in enumerate(blobs) if i not in skip)
            if not all_blobs:
                raise RuntimeError(
                    "No non-temporal blobs remain for Outlier Extraction statistics. "
                    "Choose a different frame range or use Absolute bounds.")
            iqr_stats_fixed = self._compute_iqr_stats(all_blobs)

            if area_method == "absolute":
                bounds = {"area": absolute_bounds}
            else:
                q1, median, q3, iqr = self._normalized_area_stat(
                    iqr_stats_fixed["area"]
                )
                lo = min(median, q1 - area_iqr_min * iqr)
                hi = max(median, q3 + area_iqr_max * iqr)
                bounds = {"area": (min(lo, hi), max(lo, hi))}

            self._set_progress_popup(
                f"{frames_complete}\nClassifying detected blobs...",
                None,
            )
            classified_by_frame: dict[int, list[BlobMetrics]] = {}
            total = 0
            outliers = 0
            for fid, blobs in blobs_by_frame.items():
                if self._progress_cancel:
                    return False
                classified = self._classify_blobs(
                    blobs, bounds, clone=False, result_match=result_match, frame_idx=fid)
                classified_by_frame[fid] = classified
                total += len(classified)
                outliers += sum(1 for b in classified if b.is_crossing)

            result["classified_by_frame"] = classified_by_frame
            result["iqr_stats"] = iqr_stats_fixed
            result["bounds"] = bounds
            result["total"] = total
            result["outliers"] = outliers
            self._set_progress_popup(
                f"{frames_complete}\nFinalizing analysis...",
                None,
            )
            return True

        def _finish_analysis(ok):
            if not ok:
                return

            classified_by_frame = result["classified_by_frame"]
            iqr_stats_fixed = result["iqr_stats"]
            bounds = result["bounds"]
            total = int(result["total"])
            outliers = int(result["outliers"])

            self.analysis_blobs_by_frame = classified_by_frame
            self.analysis_outlier_frames = sorted(
                fid for fid, blobs in classified_by_frame.items()
                if any(blob.is_crossing for blob in blobs)
            )
            self.analysis_frames = frame_indices
            self.analysis_signature = (self._segmentation_signature(), tuple(frame_indices), self._classifier_signature())
            self.analysis_iqr_stats = iqr_stats_fixed
            self.analysis_bounds = bounds
            self.analysis_is_stale = False
            self._update_analysis_dependent_buttons()
            self.analysis_ready_var.set(f"analysis: ready ({scope_text}, blobs={total}, outliers={outliers})")
            self.outlier_section_revealed = True
            self._guidance_stage = "outlier"
            self._set_workflow_stage("processing")
            self._redraw_area_iqr_boxplot()
            self.redraw_current_frame()
            try:
                config_path = self._save_config_to_default_path()
            except OSError as exc:
                self.set_status("Analysis completed, but config could not be saved.", auto_clear=False)
                messagebox.showwarning("Save config", f"Analysis completed, but config could not be saved:\n{exc}")
            else:
                self.set_status(
                    f"Analysis complete: blobs={total}, outliers={outliers} / config saved: {config_path}"
                )

        self._with_progress_popup("Analysis", "Preparing analysis...", work, _finish_analysis)

    def analyze_sample_frames(self):
        self._run_analysis()

    def _goto_sampled_outlier(self, direction: int) -> None:
        """Navigate outlier-containing analyzed sample frames, with wraparound."""
        if self.reader is None or self.analysis_is_stale or not self.analysis_blobs_by_frame:
            self.set_status("Run Analyze to navigate sampled outlier frames.")
            return
        frames = self.analysis_outlier_frames
        if not frames:
            self.set_status("No outliers in analyzed sample frames.")
            return
        current = int(self.current_frame)
        if direction > 0:
            target = next((fid for fid in frames if fid > current), frames[0])
        else:
            target = next((fid for fid in reversed(frames) if fid < current), frames[-1])
        self.set_frame(target)

    def _classified_current_frame(self) -> tuple[np.ndarray, list[BlobMetrics]]:
        if self.frame_bgr_cache is None:
            return np.zeros((480, 640), dtype=np.uint8), []
        mask, blobs = self._segment_frame(self.frame_bgr_cache, self.current_frame)
        if self.analysis_iqr_stats or not self._needs_area_analysis():
            try:
                self.analysis_bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats)
            except ValueError as exc:
                self.set_status(str(exc))
                return mask, [self._clone_blob(b) for b in blobs]
            classified = self._classify_blobs(blobs, self.analysis_bounds, frame_idx=self.current_frame)
            if self.single_blob_smoothing_enabled_var.get():
                cfg = self._capture_seg_config()
                classified = _smooth_classified_blobs(classified, mask.shape, cfg)
                classified = self._classify_blobs(
                    classified, self.analysis_bounds, clone=False, frame_idx=self.current_frame)
                mask = _draw_blob_union_mask(mask.shape, classified)
            elif any(b.result_generated for b in classified):
                mask = _draw_blob_union_mask(mask.shape, classified)
            return mask, self._apply_preview_filters(classified, self.current_frame)
        # Before analysis there are no fixed area bounds. Imported OBBs can
        # still identify entirely missed regions that must be removed.
        if self._result_match is not None:
            classified = self._classify_blobs(blobs, {}, frame_idx=self.current_frame)
            return (_draw_blob_union_mask(mask.shape, classified),
                    self._apply_preview_filters(classified, self.current_frame))
        # Temporal Search precedes Analyze, so its violet results must be
        # drawn even before any IQR statistics have been computed.
        return mask, self._apply_preview_filters(
            [self._clone_blob(b) for b in blobs], self.current_frame)

    @staticmethod
    def _format_bound_value(v: float) -> str:
        if np.isneginf(v):
            return "-inf"
        if np.isposinf(v):
            return "+inf"
        return f"{v:.1f}"

    def _format_bound_pair(self, bounds: tuple[float, float]) -> str:
        lo, hi = bounds
        return f"{self._format_bound_value(lo)} .. {self._format_bound_value(hi)}"

    @staticmethod
    def _render_labeled_frame(
        frame_bgr: np.ndarray,
        blobs: list[BlobMetrics],
        show_overlay: bool,
        show_contours: bool,
        show_centers: bool,
        roi_sets: list,
        frame_idx: int,
        result_obbs: "Optional[np.ndarray]" = None,
        roi_reverse: bool = False,
    ) -> np.ndarray:
        """Render a single frame with the same blob colors as the GUI preview."""
        base = frame_bgr.copy()
        _draw_roi_polygons(base, roi_sets, frame_idx, roi_reverse)

        base = _draw_blob_overlays(base, blobs, show_overlay, show_contours, show_centers)
        _draw_result_obbs(base, result_obbs)

        return base

    def _build_preview_rgb(self, *, draw_roi: bool = True) -> np.ndarray:
        if self.frame_bgr_cache is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame_bgr = self.frame_bgr_cache
        self._filter_preview_mismatch = False
        _, blobs = self._classified_current_frame()
        self._current_blobs = blobs
        kept = sum(1 for b in blobs if not b.is_crossing)
        outliers = sum(1 for b in blobs if b.is_crossing)
        rescued = sum(1 for b in blobs if b.result_single)
        missing = len({(b.frame, ri) for b in blobs if b.result_generated for ri in b.result_obb_indices})

        stale_text = " / stale" if self.analysis_is_stale and self.analysis_bounds else ""
        result_text = f" / result-single {rescued} / outlier OBBs {missing}" if self._result_match is not None else ""
        temporal_text = ""
        if self.temporal_merge_enabled_var.get() and self._temporal_search_is_current():
            excluded = sum(1 for b in blobs if b.temporal_merge_outlier)
            candidates = sum(1 for b in blobs if b.temporal_merge_level and not b.temporal_merge_outlier)
            temporal_text = f" / temporal merges {excluded}" + (f" (+{candidates} not excluded)" if candidates else "")
        if self._filter_preview_mismatch:
            temporal_text += " / filters n/a (segmentation differs from Search)"
        self.preview_title_var.set(
            f"frame {self.current_frame} / blobs {len(blobs)} / kept {kept} / "
            f"outliers {outliers}{result_text}{temporal_text}{stale_text}"
        )
        self._refresh_search_glows()
        if self.temporal_merge_enabled_var.get():
            self._update_temporal_status()

        base = frame_bgr.copy()

        if draw_roi:
            _draw_roi_polygons(base, self.roi_sets, self.current_frame, self.roi_reverse_var.get())

        base = _draw_blob_overlays(
            base, blobs, self.show_overlay_var.get(), self.show_contours_var.get(), self.show_centers_var.get())
        _draw_result_obbs(base, self._current_result_obbs())

        # IQR bounds are edited in the boxplot controls, not over the video.

        return cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

    def _draw_canvas(self, preserve_view: bool):
        self._update_area_control_ranges()
        if self._roi_drag_mode is not None:
            self._draw_roi_canvas()
            return
        preview_rgb = self._build_preview_rgb(draw_roi=False)
        self._present_image(preview_rgb, preserve_view=preserve_view)
        self._draw_roi_canvas()

    def redraw_current_frame(self):
        if self.frame_bgr_cache is not None:
            self._draw_canvas(preserve_view=True)

    def on_mousewheel(self, event):
        if self.frame_bgr_cache is None:
            return
        self.on_canvas_release()
        if hasattr(event, "delta") and event.delta > 0:
            factor = ZOOM_IN_FACTOR
        elif hasattr(event, "delta") and event.delta < 0:
            factor = ZOOM_OUT_FACTOR
        elif getattr(event, "num", None) == 4:
            factor = ZOOM_IN_FACTOR
        else:
            factor = ZOOM_OUT_FACTOR
        new_zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom_scale * factor))
        factor = new_zoom / self.zoom_scale
        self.zoom_scale = new_zoom
        pivot = np.array([event.x, event.y], dtype=float)
        self.offset_x = pivot[0] - (pivot[0] - self.offset_x) * factor
        self.offset_y = pivot[1] - (pivot[1] - self.offset_y) * factor
        self.scale = self.base_fit_scale * self.zoom_scale
        self.redraw_current_frame()

    def on_canvas_press(self, event):
        ix, iy = self._canvas_to_image(float(event.x), float(event.y))
        self.last_drag_canvas = None
        if self._start_roi_drag(ix, iy):
            self.stop_playback()
            return
        self.last_drag_canvas = np.array([event.x, event.y], dtype=float)

    def on_canvas_drag(self, event):
        if self._roi_drag_mode is not None:
            ix, iy = self._canvas_to_image(float(event.x), float(event.y))
            self._update_roi_drag(ix, iy)
            return
        if self.last_drag_canvas is None:
            return
        now = np.array([event.x, event.y], dtype=float)
        delta = now - self.last_drag_canvas
        self.last_drag_canvas = now
        self.offset_x += float(delta[0])
        self.offset_y += float(delta[1])
        self.redraw_current_frame()

    def on_canvas_release(self, _event=None):
        if self._roi_drag_mode is not None:
            self._roi_drag_mode = None
            self._roi_drag_anchor = None
            self._roi_drag_vertex = None
            self._roi_drag_points = []
            self._update_area_control_ranges()
            self.redraw_current_frame()
            return
        self.last_drag_canvas = None

    def _canvas_to_image(self, cx: float, cy: float) -> tuple[float, float]:
        ix = (cx - self.offset_x) / max(1e-9, self.scale)
        iy = (cy - self.offset_y) / max(1e-9, self.scale)
        return ix, iy

    def _blob_at_image_point(self, ix: float, iy: float) -> Optional[BlobMetrics]:
        pt = (float(ix), float(iy))
        for b in reversed(self._current_blobs):
            if cv2.pointPolygonTest(b.contour, pt, False) >= 0:
                return b
        return None

    def _blob_tooltip_text(self, blob: BlobMetrics) -> str:
        shape = (self.reader.height, self.reader.width)
        blob_region = _blob_mask_region(blob.contour, shape)
        blob_mask = blob_region[2]
        blob_pixels = cv2.countNonZero(blob_mask) if blob_mask.size else 0
        lines = [f"Blob mask area: {blob_pixels} px",
                 f"Contour area (Outlier Extraction): {blob.area:g} px²"]
        if self.analysis_iqr_stats and "area" in self.analysis_iqr_stats:
            q1, _median, q3, iqr = self._normalized_area_stat(self.analysis_iqr_stats["area"])
            if iqr > 1e-9:
                if blob.area < q1:
                    n = (q1 - blob.area) / iqr
                    lines.append(f"IQR pos: Q1 - {n:.2f}xIQR")
                elif blob.area > q3:
                    n = (blob.area - q3) / iqr
                    lines.append(f"IQR pos: Q3 + {n:.2f}xIQR")
                else:
                    n_from_q1 = (blob.area - q1) / iqr
                    lines.append(f"IQR pos: Q1 + {n_from_q1:.2f}xIQR")
            else:
                lines.append("IQR: 0")
        if self._result_match is not None:
            lines.append(f"Result OBBs: {blob.result_obb_count}")
            obbs = self._result_match.obbs_by_frame.get(int(blob.frame))
            if obbs is not None and blob_mask.size:
                for ri in blob.result_obb_indices:
                    inside_pixels = _mask_region_overlap(blob_region, _result_obb_mask(obbs[ri], shape))
                    label = "This blob inside OBB" if len(blob.result_obb_indices) == 1 else f"This blob inside OBB {ri+1}"
                    lines.append(f"{label}: {inside_pixels} px")
            if blob.result_blob_area is not None:
                lines.append(f"All source blob pixels inside OBB (Result threshold): {blob.result_blob_area} px")
            if blob.result_obb_count == 1 and not blob.manual_outlier:
                lines.append(f"Source blob OBB coverage: {blob.result_obb_coverage * 100:.0f}%")
            if blob.result_area_outlier:
                lines.append("Outside Result area bounds")
        if blob.persistent_static_outlier:
            lines.append("Persistent static blob")
        if blob.temporal_merge_level:
            state = "crossing" if blob.temporal_merge_outlier else "candidate only (not excluded)"
            lines.append(f"Temporal merge: {state}")
        if blob.temporal_merge_outlier and blob.temporal_merge_animals >= 2:
            lines.append(f"crossing({blob.temporal_merge_animals})")
        else:
            lines.append("outlier" if blob.is_crossing else "non-outlier")
        if blob.result_single:
            lines.append("kept as one animal (imported OBB)")
        return "\n".join(lines)

    def _show_blob_tooltip(self, blob: BlobMetrics, canvas_x: int, canvas_y: int):
        text = self._blob_tooltip_text(blob)
        if self._tooltip_window is None:
            self._tooltip_window = tk.Toplevel(self)
            self._tooltip_window.wm_overrideredirect(True)
            try:
                self._tooltip_window.wm_attributes("-topmost", True)
            except Exception:
                pass
            self._tooltip_label = tk.Label(
                self._tooltip_window,
                justify="left",
                bg="#242424",
                fg="#DCE4EE",
                relief="solid",
                borderwidth=1,
                font=("Consolas", 9),
                padx=6,
                pady=4,
            )
            self._tooltip_label.pack()
        self._tooltip_label.configure(text=text)
        rx = self.canvas.winfo_rootx() + canvas_x + 16
        ry = self.canvas.winfo_rooty() + canvas_y + 16
        self._tooltip_window.wm_geometry(f"+{rx}+{ry}")
        self._tooltip_window.deiconify()

    def _hide_tooltip(self):
        if self._tooltip_window is not None:
            try:
                self._tooltip_window.withdraw()
            except Exception:
                pass

    def on_canvas_motion(self, event):
        if self.frame_bgr_cache is None or not self._current_blobs:
            self._hide_tooltip()
            return
        if self.last_drag_canvas is not None or self._roi_drag_mode is not None:
            self._hide_tooltip()
            return
        ix, iy = self._canvas_to_image(float(event.x), float(event.y))
        blob = self._blob_at_image_point(ix, iy)
        if blob is not None:
            self._show_blob_tooltip(blob, event.x, event.y)
        else:
            self._hide_tooltip()

    def on_canvas_leave(self, _event=None):
        self._hide_tooltip()

    def toggle_playback(self):
        if self.playback_active:
            self.stop_playback()
        else:
            self.start_playback()

    def _toggle_play_event(self):
        self.toggle_playback()
        return "break"

    def cycle_playback_speed(self):
        speeds = (1.0, 2.0, 4.0, 0.5, 0.25)
        self.set_playback_speed(speeds[(speeds.index(self.playback_speed) + 1) % len(speeds)])

    def _playback_fps(self) -> float:
        return self.reader.fps if self.reader is not None and self.reader.fps > 0 else 30.0

    def set_playback_speed(self, speed: float):
        if speed not in (1.0, 2.0, 4.0, 0.5, 0.25):
            raise ValueError("Playback speed must be 0.25, 0.5, 1, 2, or 4.")
        self.playback_speed = speed
        if self.playback_active and self._smooth_player is not None:
            self._smooth_player.set_speed(speed)
            direction = -1 if self._playback_timing is not None and self._playback_timing[2] < 0 else 1
            self._playback_timing = (float(self.current_frame), time.monotonic(), direction * speed)
        self._update_play_button()

    def on_canvas_speed_click(self, event, speed: float):
        if self.reader is None:
            return "break"
        if self._roi_drag_mode is not None:
            return "break"
        points = self._active_roi_geometry()
        if points:
            ix, iy = self._canvas_to_image(float(event.x), float(event.y))
            if any(math.hypot(ix - x, iy - y) * self.scale <= ROI_HANDLE_GRAB_CANVAS_PX
                   for x, y in points):
                return "break"
            if cv2.pointPolygonTest(np.asarray(points, dtype=np.int32), (ix, iy), False) >= 0:
                return "break"
        self.last_drag_canvas = None
        self.set_playback_speed(speed)
        if not self.playback_active:
            self.start_playback()
        return "break"

    def _ensure_smooth_player(self):
        reader = self.reader
        if reader is None:
            raise RuntimeError("No video is loaded.")
        if self._smooth_player is not None and self._smooth_player.reader is reader:
            return self._smooth_player
        if self._smooth_player is not None:
            self._smooth_player.close()
        rotation = 0
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META") and hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            try:
                if bool(round(reader.cap.get(cv2.CAP_PROP_ORIENTATION_AUTO))):
                    rotation = int(round(reader.cap.get(cv2.CAP_PROP_ORIENTATION_META))) % 360
            except Exception:
                rotation = 0

        def seeded_frames(center):
            with reader._cache_lock:
                return {index: PreparedFrame(pixels=frame)
                        for index, frame in reader.cache.items() if index < center}

        self._smooth_player = SmoothPreviewPlayback(
            self, reader=reader, fps=self._playback_fps(),
            frame_count=reader.frame_count,
            read_forward=lambda index, gap: reader.read_bgr_with_index(
                index, max_sequential_gap=gap),
            cached_frames=seeded_frames, rotation=rotation,
            render=lambda image, index: self._present_decoded_frame(image, index),
            on_error=self._smooth_playback_failed,
            on_boundary=self.stop_playback,
            hold_is_active=self._held_arrow_is_physically_down,
            on_hold_released=self._finish_physically_released_arrow,
        )
        return self._smooth_player

    def _smooth_playback_failed(self, error):
        self.stop_playback()
        self.set_status(f"Preview decode failed: {error}")

    def _stop_held_playback(self, direction):
        timing = self._playback_timing
        if self.playback_active and timing is not None and timing[2] * direction > 0:
            self.stop_playback()

    def start_playback(self, direction: int = 1):
        if self.reader is None or self.frame_count <= 1 or self.playback_active:
            return
        if direction not in (-1, 1):
            raise ValueError("Playback direction must be -1 or 1.")
        if direction > 0 and self.current_frame >= self.frame_count - 1:
            if not self.set_frame(0):
                return
        elif direction < 0 and self.current_frame <= 0:
            return
        if self.prefetch_after_id is not None:
            self.after_cancel(self.prefetch_after_id)
            self.prefetch_after_id = None
        self._prefetch_cancel.set()
        try:
            pipeline = self._ensure_smooth_player()
            pipeline.start(direction, self.playback_speed)
        except Exception as exc:
            self.set_status(f"Preview playback initialization failed: {exc}")
            return
        self.playback_active = True
        self._playback_timing = (float(self.current_frame), time.monotonic(),
                                 direction * self.playback_speed)
        self._update_play_button()

    def stop_playback(self, update_button: bool = True):
        was_active = self.playback_active
        reversing = bool(self._playback_timing and self._playback_timing[2] < 0)
        self.playback_active = False
        self._playback_timing = None
        if self._smooth_player is not None:
            self._smooth_player.stop(discard_pending=reversing)
        if self.playback_job is not None:
            self.after_cancel(self.playback_job)
            self.playback_job = None
        if was_active and self.reader is not None and not self._closing:
            self._queue_prefetch(self.current_frame)
        if update_button:
            self._update_play_button()

    def _update_play_button(self):
        self.play_button.configure(text="Stop" if self.playback_active else "Play")
        self.playback_speed_button.configure(text=f"{self.playback_speed:g}×")

    def _schedule_playback(self, *, delay_ms: int = 8):
        # The shared decoder owns the render-driven playback timer.
        if self.playback_active and self._smooth_player is not None:
            self._smooth_player._schedule(delay_ms)

    def _playback_tick(self):
        # Retained for callers using the legacy callback name.
        if self._smooth_player is not None:
            self._smooth_player._tick()

    def _open_progress_popup(self, title: str):
        self._close_progress_popup()
        popup = ctk.CTkToplevel(self)
        popup.title(title)
        # Aqua can restack an independent Toplevel behind the maximized parent.
        # Keep the independent taskbar window behavior on Windows and Linux.
        if platform.system() == "Darwin":
            popup.transient(self)
        popup.resizable(False, False)
        popup.configure(fg_color=_POPUP_BG)
        try:
            popup.attributes("-topmost", False)
        except Exception:
            pass
        popup.protocol("WM_DELETE_WINDOW", self._request_progress_cancel)

        frame = tk.Frame(popup, padx=18, pady=16, bg=_POPUP_BG)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, textvariable=self.progress_label_var, anchor="w", justify="left", bg=_POPUP_BG, fg=_POPUP_FG).pack(fill="x")
        progress_bar = ctk.CTkProgressBar(
            frame, mode="determinate", width=360, height=12,
            fg_color="#4A4D50", progress_color="#1A7C48",
        )
        progress_bar.set(0.0)
        progress_bar.pack(fill="x", pady=(12, 0))

        self.progress_popup = popup
        self.progress_bar = progress_bar
        self._progress_indeterminate = False
        self.progress_label_var.set("Preparing...")
        self.progress_bar.set(0.0)
        self._place_progress_popup()
        popup.deiconify()
        popup.lift()
        popup.focus_force()
        popup.update_idletasks()
        popup.grab_set()

    def _place_progress_popup(self):
        if self.progress_popup is None:
            return
        self.progress_popup.update_idletasks()
        width = max(420, self.progress_popup.winfo_reqwidth())
        height = max(110, self.progress_popup.winfo_reqheight())
        root_x = self.winfo_rootx()
        root_y = self.winfo_rooty()
        root_w = max(1, self.winfo_width())
        root_h = max(1, self.winfo_height())
        x = root_x + max(0, (root_w - width) // 2)
        y = root_y + max(0, (root_h - height) // 2)
        self.progress_popup.geometry(f"{width}x{height}+{x}+{y}")

    def _flush_ui(self, *, maximize: bool = True):
        try:
            self.update_idletasks()
            self.update()
            if maximize:
                self._maximize_window()
        except tk.TclError:
            pass

    def _set_progress_popup(self, text: str, value: Optional[float]):
        # Thread-safe: worker pushes to queue; main thread drains it in the polling loop.
        normalized = None if value is None else float(max(0.0, min(100.0, value)))
        self._progress_queue.put_nowait((text, normalized))

    def _drain_progress_queue(self):
        # Coalesce frame updates; only Tk's main thread changes widget state.
        latest = None
        while True:
            try:
                latest = self._progress_queue.get_nowait()
            except queue.Empty:
                break
        if latest is None or self.progress_popup is None:
            return
        text, value = latest
        self.progress_label_var.set(text)
        if value is None:
            if self.progress_bar is not None and not self._progress_indeterminate:
                self.progress_bar.set(0.0)
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start()
                self._progress_indeterminate = True
        else:
            if self.progress_bar is not None and self._progress_indeterminate:
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
                self._progress_indeterminate = False
            self.progress_bar.set(value / 100.0)

    def _close_progress_popup(self):
        if self._progress_poll_job is not None:
            try:
                self.after_cancel(self._progress_poll_job)
            except tk.TclError:
                pass
            self._progress_poll_job = None
        if self.progress_bar is not None:
            try:
                self.progress_bar.stop()
            except tk.TclError:
                pass
        if self.progress_popup is not None:
            try:
                self.progress_popup.grab_release()
            except tk.TclError:
                pass
            try:
                self.progress_popup.destroy()
            except tk.TclError:
                pass
        self.progress_popup = None
        self.progress_bar = None
        self._progress_indeterminate = False

    def _request_progress_cancel(self):
        # Keep the window and lock until the worker acknowledges cancellation.
        self._progress_cancel = True
        if self.progress_popup is not None:
            self.progress_label_var.set("Cancelling... waiting for the operation to stop.")

    def _with_progress_popup(self, title: str, start_text: str, func, on_done) -> None:
        """Run work on a thread; poll with after() and finalize on the Tk thread."""
        if self._progress_busy:
            raise RuntimeError("Another Segmentation operation is in progress.")
        self._progress_busy = True
        self._progress_cancel = False
        self._progress_close_requested = False
        while not self._progress_queue.empty():
            self._progress_queue.get_nowait()
        try:
            self._open_progress_popup(title)
            self.progress_label_var.set(start_text)
            self.progress_bar.set(0.0)
            self._place_progress_popup()
        except Exception:
            self._progress_busy = False
            self._close_progress_popup()
            raise

        result_box = [False]
        done_event = threading.Event()
        started = time.perf_counter()
        print(f"[Segmentation] {title}: worker started", flush=True)

        def worker():
            try:
                result_box[0] = bool(func())
            except Exception:
                traceback.print_exc()
            finally:
                print(
                    f"[Segmentation] {title}: worker finished after "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )
                done_event.set()

        def poll():
            self._progress_poll_job = None
            if self._closing:
                return
            self._drain_progress_queue()
            if done_event.is_set():
                cancelled = self._progress_cancel
                self._progress_busy = False
                self._close_progress_popup()
                print(
                    f"[Segmentation] {title}: popup closed after "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )
                try:
                    on_done(bool(result_box[0]) and not cancelled)
                except Exception as exc:
                    traceback.print_exc()
                    messagebox.showerror(title, str(exc), parent=self)
                finally:
                    if self._progress_close_requested:
                        self._progress_close_requested = False
                        self.on_close()
                return
            self._progress_poll_job = self.after(50, poll)

        threading.Thread(target=worker, daemon=True, name="segmentation-progress-worker").start()
        self._progress_poll_job = self.after(50, poll)

    def show_background_popup(self):
        if self.background_bgr is None:
            messagebox.showerror("Error", "Please compute the background first.")
            return
        if self.bg_preview_popup is not None:
            try:
                self.bg_preview_popup.destroy()
            except Exception:
                pass
        popup = tk.Toplevel(self)
        popup.title("Background Preview")
        popup.transient(self)
        popup.geometry("1100x800")
        popup.minsize(500, 400)
        label = tk.Label(popup, bg="black")
        label.pack(fill="both", expand=True)

        rgb = cv2.cvtColor(self.background_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w, max_h = 1080, 760
        scale = min(max_w / image.width, max_h / image.height, 1.0)
        disp = image.resize((max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))), Image.LANCZOS)
        self.bg_preview_image = ImageTk.PhotoImage(disp)
        label.configure(image=self.bg_preview_image)
        self.bg_preview_popup = popup
        popup.protocol("WM_DELETE_WINDOW", lambda: self._close_bg_preview())
        if self._guidance_stage == "background":
            self._guidance_stage = "threshold"
            self._set_workflow_stage("analysis")

    def _close_bg_preview(self):
        if self.bg_preview_popup is not None:
            try:
                self.bg_preview_popup.destroy()
            except Exception:
                pass
        self.bg_preview_popup = None
        self.bg_preview_image = None

    @staticmethod
    def _segmentation_dir_for_video(video_path: str) -> str:
        """Derive the conventional <video_dir>/amadeus_<stem>/segmentation directory.

        This is only the default location; a session's config.yaml can point
        PICKLE_PATH/BACKGROUND_PATH at a different directory (SESSION_PATH is
        user-editable in Advanced Tracking), so callers that need to find an
        existing segmentation output should not assume this is the only place.
        """
        video_path = str(video_path or "").strip()
        if not video_path:
            return ""
        video_dir = os.path.dirname(video_path)
        stem = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(video_dir, f"amadeus_{stem}", "segmentation")

    def _default_output_dir(self) -> str:
        # Always <project_dir>/segmentation, where project_dir is the same
        # SESSION_PATH the downstream tracking config uses (see _project_dir):
        # the video's own directory only when neither a saved config nor an
        # explicit session was provided. This keeps every file this class reads or writes (config.json,
        # background.png, the pickle) in one single, self-consistent location.
        project_dir = self._project_dir()
        if not project_dir:
            return os.getcwd()
        out_dir = os.path.join(project_dir, "segmentation")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _default_background_path(self) -> str:
        return os.path.join(self._default_output_dir(), "background.png")

    def _method_background_path(self, method: str) -> str:
        if method not in BACKGROUND_METHODS:
            raise RuntimeError(f"Unknown background method: {method}")
        return os.path.join(self._default_output_dir(), f"background_{method}.png")

    def _background_cache_metadata_path(self) -> str:
        return os.path.join(self._default_output_dir(), "background_cache.json")

    def _background_sample_count(self) -> int:
        if self.frame_count <= 0:
            return 0
        start = max(0, int(self.background_frame_start_var.get()))
        end = self._resolved_frame_end(int(self.background_frame_end_var.get()))
        if start > end:
            return 0
        range_count = end - start + 1
        if self.reader is None:
            return min(BACKGROUND_SAMPLE_COUNT, range_count)
        frame_bytes = max(1, int(self.reader.width) * int(self.reader.height) * 3)
        memory_limited_count = max(1, BACKGROUND_STACK_MAX_BYTES // frame_bytes)
        return min(BACKGROUND_SAMPLE_COUNT, range_count, memory_limited_count)

    def _background_source_signature(self) -> dict:
        video_path = os.path.abspath(self.video_path_var.get().strip())
        signature = {
            "video_path": os.path.normcase(video_path),
            "frame_count": int(self.frame_count),
            "total_frame_count": int(self.frame_count),
            "background_frame_start": int(self.background_frame_start_var.get()),
            "background_frame_end": int(self.background_frame_end_var.get()),
        }
        try:
            stat = os.stat(video_path)
            signature["video_size"] = int(stat.st_size)
            signature["video_mtime_ns"] = int(stat.st_mtime_ns)
        except OSError:
            signature["video_size"] = None
            signature["video_mtime_ns"] = None
        return signature

    def _background_cache_current_methods(self) -> set[str]:
        if self.reader is None:
            return set()
        try:
            with open(self._background_cache_metadata_path(), "r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, ValueError, TypeError):
            return set()

        requested_samples = self._background_sample_count()
        if requested_samples <= 0:
            return set()
        bg_start = max(0, int(self.background_frame_start_var.get()))
        bg_end = self._resolved_frame_end(int(self.background_frame_end_var.get()))
        requested_indices = np.linspace(bg_start, bg_end, requested_samples, dtype=int).tolist()
        try:
            cached_indices = [int(v) for v in metadata.get("sample_indices", [])]
        except Exception:
            return set()
        source = self._background_source_signature()
        if not (
            metadata.get("version") == 2
            and int(metadata.get("sample_count", -1)) == requested_samples
            and cached_indices == [int(v) for v in requested_indices]
            and all(metadata.get(key) == value for key, value in source.items())
        ):
            return set()

        methods = set(metadata.get("methods", []))
        return {
            method
            for method in BACKGROUND_METHODS
            if method in methods and os.path.exists(self._method_background_path(method))
        }

    def _write_background_cache_metadata(self, sample_count: int, sample_indices: np.ndarray, methods: set[str]):
        valid_methods = [method for method in BACKGROUND_METHODS if method in methods]
        metadata = {
            "version": 2,
            **self._background_source_signature(),
            "sample_count": int(sample_count),
            "sample_indices": [int(value) for value in sample_indices.tolist()],
            "methods": valid_methods,
        }
        path = self._background_cache_metadata_path()
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)

    def _background_cache_is_current(self, method: str) -> bool:
        if self.reader is None or method not in BACKGROUND_METHODS:
            return False
        return method in self._background_cache_current_methods()

    def _invalidate_analysis_for_background_change(self):
        self.current_mask_cache.clear()
        self.analysis_blobs_by_frame.clear()
        self.analysis_outlier_frames.clear()
        self.analysis_frames = []
        self.analysis_signature = None
        self.analysis_bounds = {}
        self.analysis_iqr_stats = {}
        self.analysis_is_stale = True
        self.analysis_ready_var.set("analysis: not computed")
        self.outlier_section_revealed = False
        self._guidance_stage = "background" if self._segmentation_needs_background() else "threshold"
        self._set_workflow_stage("analysis")

    def _activate_cached_background(self, method: str, redraw: bool = True) -> bool:
        if not self._background_cache_is_current(method):
            return False
        method_path = self._method_background_path(method)
        background = cv2.imread(method_path, cv2.IMREAD_COLOR)
        if background is None:
            return False
        if self.reader is not None and background.shape[:2] != (self.reader.height, self.reader.width):
            return False

        # background.png is the stable downstream path; keep its bytes identical
        # to the currently selected method-specific cache.
        shutil.copyfile(method_path, self._default_background_path())
        self.background_bgr = background
        self.active_background_method = method
        self._invalidate_analysis_for_background_change()
        sample_count = self._background_sample_count()
        self.background_ready_var.set(f"background: cached ({method}, samples={sample_count})")
        if self.bg_preview_button is not None:
            self.bg_preview_button.configure(state="normal")
        if self._segmentation_needs_background():
            self._set_workflow_stage("analysis")
        if redraw:
            self.redraw_current_frame()
        return True

    def _default_pickle_path(self) -> str:
        stem = os.path.splitext(os.path.basename(self.video_path_var.get().strip() or "segmentation"))[0]
        return os.path.join(self._default_output_dir(), f"{stem}_list_of_blobs_gui.pickle")


    def _project_dir(self) -> str:
        # Prefer a location relative to the loaded/saved segmentation config
        # (config.json lives in .../amadeus_<stem>/segmentation/, so one level
        # up is the project dir) over one derived from the video path. The
        # video is often not stored under the project's own directory tree,
        # so prefer the config location or explicitly selected session.
        if self.config_path:
            config_dir = os.path.dirname(os.path.abspath(self.config_path))
            project_dir = os.path.dirname(config_dir)
            if project_dir:
                os.makedirs(project_dir, exist_ok=True)
                return project_dir

        if self.session_path:
            project_dir = os.path.abspath(os.path.expanduser(self.session_path))
            os.makedirs(project_dir, exist_ok=True)
            return project_dir

        video_path = self.video_path_var.get().strip()
        if not video_path:
            return os.getcwd()
        video_dir = os.path.dirname(video_path)
        stem = os.path.splitext(os.path.basename(video_path))[0]
        project_dir = os.path.join(video_dir, f"amadeus_{stem}")
        os.makedirs(project_dir, exist_ok=True)
        return project_dir

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(round(float(seconds))))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"

    def _capture_seg_config(self) -> _SegConfig:
        """Snapshot current segmentation params for thread-safe parallel use."""
        self._sync_roi_vars_to_set(self.roi_active_set_idx)
        self._sync_additional_outlier_vars_to_set()
        ksize = max(1, int(self.blur_ksize_var.get()))
        if ksize % 2 == 0:
            ksize += 1
        try:
            protected_area_bounds = self._protected_area_bounds()
        except (TypeError, ValueError):
            # Keep preview segmentation available while the user is correcting
            # an invalid absolute area range.
            protected_area_bounds = None
        return _SegConfig(
            mode=self.segmentation_mode_var.get(),
            ksize=ksize,
            threshold=int(self.threshold_var.get()),
            dark_threshold=int(self.dark_threshold_var.get()),
            bright_threshold=int(self.bright_threshold_var.get()),
            diff_threshold=int(self.diff_threshold_var.get()),
            min_area=float(max(1, int(self.min_area_var.get()))),
            open_iter=max(0, int(self.open_iter_var.get())),
            close_iter=max(0, int(self.close_iter_var.get())),
            fill_holes=bool(self.fill_holes_var.get()),
            invert_mask=bool(self.invert_mask_var.get()),
            expand_px=max(0, int(self.region_expand_px_var.get())),
            expand_merge_only=bool(self.region_expand_merge_only_var.get()),
            roi_sets=[normalize_roi_set(r) for r in self.roi_sets],
            roi_reverse=bool(self.roi_reverse_var.get()),
            additional_outlier_sets=[dict(r) for r in self.additional_outlier_sets],
            background_bgr=self.background_bgr,  # read-only; no copy needed
            additional_backgrounds=self._additional_outlier_backgrounds_for_config(),
            protected_area_bounds=protected_area_bounds,
            single_blob_smoothing_enabled=bool(self.single_blob_smoothing_enabled_var.get()),
            single_blob_smoothing_level=max(1, min(20, int(self.single_blob_smoothing_level_var.get()))),
            use_gpu=bool(_CUDA_AVAILABLE and _torch is not None),
        )

    def _additional_outlier_backgrounds_for_config(self) -> dict[str, np.ndarray]:
        """Snapshot cached backgrounds needed by enabled Additional Outlier Sets.

        This only reads method-specific cache files.  In particular, it never
        assigns to ``self.background_bgr`` or changes the active normal method.
        """
        backgrounds: dict[str, np.ndarray] = {}
        for method in self._additional_outlier_background_methods():
            background = None
            if self.background_bgr is not None and (
                self.active_background_method == method
                or (self.active_background_method is None and self.bg_method_var.get() == method)
            ):
                background = self.background_bgr
            if background is None:
                try:
                    if self._background_cache_is_current(method):
                        background = cv2.imread(self._method_background_path(method), cv2.IMREAD_COLOR)
                except Exception:
                    background = None
            if background is not None:
                if self.reader is None or background.shape[:2] == (self.reader.height, self.reader.width):
                    backgrounds[method] = background
        return backgrounds

    def _classify_frame(self, blobs, fid, cfg, bounds, result_match, image_shape) -> list[BlobMetrics]:
        """Final per-frame classification shared by Search, Processing and Export."""
        classified = self._classify_blobs(
            blobs, bounds, clone=False, result_match=result_match, frame_idx=fid, image_shape=image_shape)
        if cfg.single_blob_smoothing_enabled:
            classified = _smooth_classified_blobs(classified, image_shape, cfg)
            classified = self._classify_blobs(
                classified, bounds, clone=False, result_match=result_match, frame_idx=fid,
                image_shape=image_shape)
        return classified

    def _ordered_frames(self, start: int, end: int, process):
        """Yield (frame, process(frame, image)) in frame order with bounded memory.

        One sequential decoder thread feeds a worker pool batch by batch; the
        preview reader is used only where the dedicated decoder fails.
        """
        from main.frame_scan import iter_frames_in_order

        reader = self.reader
        frame_bytes = max(1, int(reader.width) * int(reader.height) * 3)
        memory_batch_limit = max(1, PROCESSING_BATCH_MAX_BYTES // frame_bytes)
        batch_size = min(max(1, _EXPORT_WORKERS * 2), memory_batch_limit)
        # Decoding the next batch ahead keeps two batches in memory. Shrink the
        # batch so both fit the same ceiling, but never below one full round for
        # the pool: with frames that large, keep the previous batch size, worker
        # occupancy and peak memory instead, and decode on demand.
        prefetch = memory_batch_limit // 2 >= _EXPORT_WORKERS
        if prefetch:
            batch_size = min(batch_size, memory_batch_limit // 2)
        decode_cap = cv2.VideoCapture(reader.video_path)
        position = [-1]

        # Use a dedicated sequential decoder.  The preview reader copies and
        # caches frames, which is useful interactively but unnecessarily
        # expensive while processing the entire range.  Only the single decoder
        # thread touches decode_cap and position.
        def decode_batch(fids: "list[int]") -> "list[tuple[int, np.ndarray]]":
            frames_data: list[tuple[int, np.ndarray]] = []
            for fid in fids:
                frame = None
                if decode_cap.isOpened():
                    if position[0] != fid:
                        decode_cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                    ok, decoded = decode_cap.read()
                    if ok and decoded is not None and decoded.size > 0:
                        frame = decoded
                        position[0] = fid + 1
                    else:
                        position[0] = -1
                if frame is None:
                    frame = reader.read_bgr(fid)
                frames_data.append((fid, frame))
            return frames_data

        # Reduce OpenCV's internal thread count so each Python thread runs a truly
        # single-threaded cv2 call -- the thread pool then provides parallelism.
        prev_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            yield from iter_frames_in_order(
                range(int(start), int(end) + 1), decode_batch, process,
                workers=_EXPORT_WORKERS, batch_size=batch_size, prefetch=prefetch,
                cancelled=lambda: self._progress_cancel,
            )
        finally:
            decode_cap.release()
            cv2.setNumThreads(prev_threads)

    def _build_export_source_all_frames(
        self,
        progress_start: float = 10.0,
        progress_end: float = 35.0,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
        cfg: Optional[_SegConfig] = None,
        bounds: Optional[dict[str, tuple[float, float]]] = None,
        progress_label: str = "Processing frames",
        result_match=_UNSET,
        filters: Optional[_FilterPlan] = None,
    ) -> dict[int, list[BlobMetrics]]:
        if filters is None:
            filters = self._search_filter_snapshot()
        if bounds is None:
            if self._needs_area_analysis() and not self.analysis_iqr_stats:
                raise RuntimeError("Please run Analyze sampled frames first.")
            try:
                self.analysis_bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats)
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
            bounds = dict(self.analysis_bounds)
        else:
            bounds = dict(bounds)
        if cfg is None:
            cfg = self._capture_seg_config()
        if result_match is _UNSET:
            result_match = self._result_match
        source: dict[int, list[BlobMetrics]] = {}
        digests: dict[int, int] = {}
        if frame_start is None:
            frame_start = int(self.training_frame_start_var.get())
        if frame_end is None:
            frame_end = int(self.training_frame_end_var.get())
        start_frame = max(0, int(frame_start))
        end_frame = self._resolved_frame_end(int(frame_end))
        if start_frame > end_frame:
            raise RuntimeError("Invalid export frame range.")
        total = max(1, end_frame - start_frame + 1)
        start_time = time.perf_counter()
        image_shape = (int(self.reader.height), int(self.reader.width))

        accel = "GPU" if cfg.use_gpu else f"CPUx{_EXPORT_WORKERS}"
        self._set_progress_popup(
            f"{progress_label}: 0/{total} (0%)\nstarting {accel}...",
            progress_start,
        )
        from main.frame_scan import frame_blob_digest

        def segment(fid: int, frame: np.ndarray):
            return _segment_blobs_with_config(frame, fid, cfg)

        # Workers only segment. Classification stays on this thread, as before:
        # its Python loops hold the GIL and get slower, not faster, in threads.
        report_every = max(1, _EXPORT_WORKERS * 2)
        for n, (fid, blobs) in enumerate(self._ordered_frames(start_frame, end_frame, segment), 1):
            if self._progress_cancel:
                return {}
            blobs = self._classify_frame(blobs, fid, cfg, bounds, result_match, image_shape)
            source[fid] = blobs
            digests[fid] = frame_blob_digest(blobs)
            if n % report_every == 0 or n == total:
                elapsed = max(1e-9, time.perf_counter() - start_time)
                fps = n / elapsed
                eta = max(0, total - n) / fps if fps > 0 else 0.0
                frame_pct = 100.0 * n / total
                pct = progress_start + (progress_end - progress_start) * n / total
                self._set_progress_popup(
                    f"{progress_label}: {n}/{total} ({frame_pct:.0f}%)\n"
                    f"speed={fps:.1f} fps / remaining={self._format_seconds(eta)}",
                    pct,
                )
        if self._progress_cancel:
            return {}
        self._apply_filters_to_source(source, digests, filters, cfg, image_shape)
        return source

    def _apply_filters_to_source(
        self, source: dict[int, list[BlobMetrics]], digests: dict[int, int],
        plan: _FilterPlan, cfg: _SegConfig, image_shape: tuple,
    ) -> None:
        """Apply the static and temporal filters to the classified frames.

        Processing reuses Search results only if every searched frame has the
        same blob digest; otherwise (for example after a GPU fallback) both
        filters are recomputed on these exact blobs. Export draws Search
        results only on frames whose digest matches.
        """
        stats = plan.stats
        frames = sorted(source)
        stats["frames"] = len(frames)
        if plan.static is None and plan.temporal is None:
            return
        if plan.mode == "apply":
            mismatched = 0
            for fid in frames:
                mismatched += _apply_filter_plan_to_frame(source[fid], fid, plan, self._apply_static_outliers)
            stats["mismatched_frames"] = mismatched
            return

        def matches(search_digests, first: int) -> int:
            different = 0
            for fid in frames:
                index = fid - first
                if not 0 <= index < len(search_digests) or int(search_digests[index]) != digests[fid]:
                    different += 1
            return different

        base = {fid: [bool(b.is_crossing) for b in source[fid]] for fid in frames}
        static_exclusions = None
        if plan.static is not None:
            exclusions, search_digests, first, args = plan.static
            different = matches(search_digests, first)
            stats["static_mismatched_frames"] = different
            if different:
                from main.persistent_static_blobs import PersistentStaticScanner
                self._set_progress_popup("Search data differs from Processing; recomputing static blobs...", None)
                scanner = PersistentStaticScanner(*args)
                for fid in frames:
                    if self._progress_cancel:
                        return
                    scanner.add_frame(fid, source[fid])
                static_exclusions = scanner.excluded
                stats["static_recomputed"] = True
            else:
                static_exclusions = exclusions
                stats["static_recomputed"] = False
            stats["static_search_count"] = sum(len(v) for v in exclusions.values())
        temporal_result = None
        if plan.temporal is not None:
            state = plan.temporal
            different = matches(state.graph.digests, state.start)
            stats["temporal_mismatched_frames"] = different
            if different:
                from main.temporal_merge_blobs import TemporalMergeGraphBuilder, evaluate_temporal_merges
                self._set_progress_popup("Search data differs from Processing; recomputing temporal merges...", None)
                context = _FrameContext(cfg, image_shape)
                builder = TemporalMergeGraphBuilder(image_shape)
                with ThreadPoolExecutor(max_workers=_EXPORT_WORKERS) as pool:
                    for start in range(0, len(frames), 64):
                        chunk = frames[start:start + 64]
                        nodes = list(pool.map(lambda f: context.nodes(f, source[f], digests[f]), chunk))
                        for item in nodes:
                            builder.add_frame(item)
                        if self._progress_cancel:
                            return
                temporal_result = evaluate_temporal_merges(builder.finish(), state.params)
                stats["temporal_recomputed"] = True
            else:
                temporal_result = state.result
                stats["temporal_recomputed"] = False
            stats["temporal_result"] = temporal_result
            stats["temporal_search_excluded"] = state.result.excluded_count
        static_count = 0
        for fid in frames:
            blobs = source[fid]
            if static_exclusions is not None:
                self._apply_static_outliers(blobs, fid, static_exclusions, base[fid])
                static_count += sum(1 for b in blobs if b.persistent_static_outlier)
            if temporal_result is not None:
                _apply_temporal_levels(blobs, temporal_result, fid, base[fid])
        stats["static_count"] = static_count
        if temporal_result is not None:
            stats["temporal_count"] = sum(1 for fid in frames for b in source[fid] if b.temporal_merge_outlier)

    def _save_background_png_to_path(self, path: str):
        if self.background_bgr is None:
            raise RuntimeError("No background available.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not cv2.imwrite(path, self.background_bgr):
            raise RuntimeError(f"Failed to save background: {path}")

    def _save_filtered_pickle_to_path(
        self,
        path: str,
        source: Optional[dict[int, list[BlobMetrics]]] = None,
        metadata: Optional[dict[str, object]] = None,
    ):
        if self.reader is None:
            raise RuntimeError("No video has been loaded.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if source is None:
            if self._segmentation_needs_background() and self.background_bgr is None:
                raise RuntimeError("In this mode, please compute the background first.")
            source = self._build_export_source_all_frames()
        blobs_in_video: list[list[SimpleNamespace]] = []
        segmentation_source_blobs: dict[int, list[SimpleNamespace]] = {}
        for fid in range(self.frame_count):
            frame_blobs = []
            for b in source.get(fid, []):
                item = _segmentation_pickle_blob(b, b.result_source_index)
                frame_blobs.append(item)
                if b.result_source_frame is not None and fid not in segmentation_source_blobs:
                    segmentation_source_blobs[fid] = [
                        _segmentation_pickle_blob(original, index)
                        for index, original in enumerate(b.result_source_frame)
                    ]
            blobs_in_video.append(frame_blobs)
        if metadata is None:
            metadata = {
                "background_frame_start": int(self.background_frame_start_var.get()),
                "background_frame_end": int(self.background_frame_end_var.get()),
                "training_frame_start": int(self.training_frame_start_var.get()),
                "training_frame_end": int(self.training_frame_end_var.get()),
                "training_frame_interval": max(1, int(self.training_frame_interval_var.get())),
                "pre_result_path": self._active_pre_result_path(),
            }
        with open(path, "wb") as f:
            exported = SimpleNamespace(
                blobs_in_video=blobs_in_video,
                source_frame_count=int(self.frame_count),
                background_frame_start=int(metadata["background_frame_start"]),
                background_frame_end=int(metadata["background_frame_end"]),
                training_frame_start=int(metadata["training_frame_start"]),
                training_frame_end=int(metadata["training_frame_end"]),
                training_frame_interval=max(1, int(metadata["training_frame_interval"])),
                pre_result_path=to_relative_path(
                    metadata.get("pre_result_path", ""), os.path.dirname(os.path.abspath(path)),
                ),
            )
            if segmentation_source_blobs:
                exported.segmentation_source_blobs = segmentation_source_blobs
            pickle.dump(exported, f)

    def _launch_easy_tracking(self, session_path: str, video_path: str):
        target = str(gui_script("gui_easy_tracking.py"))
        if not os.path.exists(target):
            raise FileNotFoundError(target)
        try:
            command = [sys.executable, "-u", target, "--video", video_path, "--session", session_path]
            if os.name == "nt":
                subprocess.Popen(["cmd", "/k", *command], cwd=str(PROJECT_ROOT))
            else:
                subprocess.Popen(command, cwd=str(PROJECT_ROOT))
        except Exception as e:
            raise RuntimeError(f"Failed to launch gui_easy_tracking.py: {e}")

    def _write_processing_log(self, elapsed_sec: float, video_path: str, frame_count: int,
                              pickle_path: str, seg_mode: str) -> None:
        import datetime
        log_path = os.path.join(self._default_output_dir(), "processing_log.txt")
        h = int(elapsed_sec // 3600)
        m = int((elapsed_sec % 3600) // 60)
        s = elapsed_sec % 60
        elapsed_str = f"{h}h {m}m {s:.1f}s" if h else (f"{m}m {s:.1f}s" if m else f"{s:.1f}s")
        bg_start = int(self.background_frame_start_var.get())
        bg_end = int(self.background_frame_end_var.get())
        tr_start = int(self.training_frame_start_var.get())
        tr_end = int(self.training_frame_end_var.get())
        interval = max(1, int(self.training_frame_interval_var.get()))
        tr_end_resolved = self._resolved_frame_end(tr_end)
        processed_frames = max(0, tr_end_resolved - tr_start + 1)
        method = self.area_outlier_method_var.get()
        method_label = "Absolute" if method == "absolute" else "IQR"
        try:
            bounds = self.analysis_bounds.get("area") or self._bounds_from_fixed_stats(self.analysis_iqr_stats).get("area")
        except Exception:
            bounds = None
        bounds_text = self._format_bound_pair(bounds) if bounds else "not available"
        lines = [
            f"=== Segmentation Processing Log ===",
            f"Date                       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Video                      : {video_path}",
            f"Total frames               : {frame_count}",
            f"Background frame range     : {self._frame_range_label(bg_start, bg_end)}",
            f"Training frame range       : {self._frame_range_label(tr_start, tr_end)}",
            f"Training frame interval    : {interval}",
            f"Processed segmentation frames : {processed_frames}",
            f"Area outlier method        : {method_label}",
            f"Area bounds                : {bounds_text}",
            f"Mode                       : {seg_mode}",
            f"Elapsed                    : {elapsed_str}  ({elapsed_sec:.2f} s)",
            f"Output                     : {pickle_path}",
        ]
        lines += self._filter_log_lines(getattr(self, "_last_filter_stats", {}) or {})
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _filter_log_lines(self, stats: dict) -> list[str]:
        lines = []
        if stats.get("static_enabled"):
            lines.append(f"Persistent static exclusion : {int(stats.get('static_count', 0))} blobs")
            if stats.get("static_recomputed"):
                lines.append(
                    f"  recomputed on Processing blobs ({int(stats.get('static_mismatched_frames', 0))} frames "
                    f"differed from Search; Search found {int(stats.get('static_search_count', 0))})")
        else:
            lines.append("Persistent static exclusion : disabled")
        params = stats.get("temporal_params")
        if not stats.get("temporal_enabled") or params is None:
            lines.append("Temporal merge exclusion   : disabled")
            return lines
        result = stats.get("temporal_result")
        mode = "include one-sided" if params.include_one_sided else "bilateral only"
        lines.append(f"Temporal merge exclusion   : {int(stats.get('temporal_count', 0))} blobs ({mode})")
        if result is None:
            return lines
        counts, diag = result.counts, result.diagnostics
        lines += [
            f"  high / medium / uncertain : {counts['high_intervals']} / {counts['medium_intervals']} / "
            f"{counts['uncertain_intervals']} intervals",
            f"  high / medium blobs       : {counts['high_blobs']} / {counts['medium_blobs']}",
            "  confirmation              : 2 frames (fixed)",
            f"  share rounding            : {params.share_rounding:.2f}",
            f"  one-sided mass ratio      : {params.one_sided_mass_ratio:.2f}",
            f"  containment threshold     : {diag.get('tau')} ({diag.get('tau_mode')}; "
            f"self-containment P10={diag.get('self_containment_p10')}, pairs={diag.get('calibration_pairs')})",
            f"  reliable for exclusion    : {'yes' if diag.get('reliable') else 'no'}",
            f"  events CSV                : {os.path.join(self._default_output_dir(), TEMPORAL_EVENTS_CSV)}"
            + (f" (NOT WRITTEN: {stats['csv_error']})" if stats.get("csv_error") else ""),
        ]
        if stats.get("temporal_recomputed"):
            lines.append(
                f"  recomputed on Processing blobs ({int(stats.get('temporal_mismatched_frames', 0))} frames "
                f"differed from Search; Search excluded {int(stats.get('temporal_search_excluded', 0))})")
        lines += [f"  warning: {text}" for text in diag.get("warnings", [])]
        return lines

    def run_processing(self):
        if self._progress_busy:
            messagebox.showwarning("Processing", "Another run is already in progress.")
            return
        if self.reader is None:
            messagebox.showerror("Error", "Please load a video first.")
            return
        if self._segmentation_needs_background() and self.background_bgr is None:
            messagebox.showerror("Error", "In this mode, please compute the background first.")
            return
        if not self.outlier_section_revealed:
            messagebox.showwarning(
                "Outlier Extraction",
                "Run Analyze before Processing.",
            )
            return
        if self.analysis_is_stale or (self._needs_area_analysis() and not self.analysis_iqr_stats):
            messagebox.showwarning(
                "Outlier Extraction",
                "Processing needs a completed, current analysis.\n\n"
                "Run Analyze in the Outlier Extraction section first.",
            )
            return

        if self.static_exclusion_enabled_var.get() and not self._static_search_is_current():
            messagebox.showwarning(
                "Persistent static blobs",
                "Search has not been completed for the current settings.\n"
                "Open 'Exclude persistent static blobs' and press Search, "
                "or disable the option before Processing.",
            )
            return
        if self.temporal_merge_enabled_var.get() and not self._temporal_search_is_current():
            messagebox.showwarning(
                "Temporal merges",
                "Search has not been completed for the current settings.\n"
                "Open 'Exclude temporal merges' and press Search, "
                "or disable the option before Processing.",
            )
            return

        self._prefetch_cancel.set()
        self.reader.clear_cache()
        try:
            # Capture GUI state once on the main thread.  The processing worker
            # can then run without cross-thread Tk variable access.
            filters = self._processing_filter_plan()
            output_dir = self._default_output_dir()
            cfg = self._capture_seg_config()
            result_match = self._result_match
            bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats)
            frame_start = max(0, int(self.training_frame_start_var.get()))
            frame_end = self._resolved_frame_end(int(self.training_frame_end_var.get()))
            if frame_start > frame_end:
                raise RuntimeError("Invalid processing frame range.")
            total_frames = frame_end - frame_start + 1
            frames_complete = f"Frames processed: {total_frames}/{total_frames} (100%)"
            config_path = self.config_path or self._default_config_path()
            config_data = self._collect_config(config_path)
            background_path = self._default_background_path()
            pickle_path = self._default_pickle_path()
            project_dir = self._project_dir()
            seg_mode = self.segmentation_mode_var.get()
            video_path = self.video_path_var.get().strip()
            video_height, video_width = int(self.reader.height), int(self.reader.width)
            background_snapshot = self.background_bgr
            launch_tracking = not self._no_launch_tracking
            pickle_metadata = {
                "background_frame_start": int(self.background_frame_start_var.get()),
                "background_frame_end": int(self.background_frame_end_var.get()),
                "training_frame_start": int(self.training_frame_start_var.get()),
                "training_frame_end": int(self.training_frame_end_var.get()),
                "training_frame_interval": max(1, int(self.training_frame_interval_var.get())),
                "pre_result_path": self._active_pre_result_path(),
            }
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        completed: list = [None]  # stores (config_path, background_path, pickle_path, project_dir) or exception

        def work():
            try:
                source = self._build_export_source_all_frames(
                    progress_start=0,
                    progress_end=100,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    cfg=cfg,
                    bounds=bounds,
                    progress_label="Processing frames",
                    result_match=result_match,
                    filters=filters,
                )
                if self._progress_cancel or not source:
                    return False

                self._set_progress_popup(f"{frames_complete}\nSaving config...", None)
                dirname = os.path.dirname(config_path)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
                temp_config_path = config_path + ".tmp"
                with open(temp_config_path, "w", encoding="utf-8") as file:
                    json.dump(config_data, file, ensure_ascii=False, indent=2)
                os.replace(temp_config_path, config_path)
                if self._progress_cancel:
                    return False

                if seg_mode == "dark_region":
                    self._set_progress_popup(f"{frames_complete}\nSaving white background.png...", None)
                    synthetic_bg = np.full((video_height, video_width, 3), 255, dtype=np.uint8)
                    os.makedirs(os.path.dirname(background_path), exist_ok=True)
                    if not cv2.imwrite(background_path, synthetic_bg):
                        raise RuntimeError(f"Failed to save background: {background_path}")
                elif seg_mode == "bright_region":
                    self._set_progress_popup(f"{frames_complete}\nSaving black background.png...", None)
                    synthetic_bg = np.zeros((video_height, video_width, 3), dtype=np.uint8)
                    os.makedirs(os.path.dirname(background_path), exist_ok=True)
                    if not cv2.imwrite(background_path, synthetic_bg):
                        raise RuntimeError(f"Failed to save background: {background_path}")
                elif background_snapshot is not None:
                    self._set_progress_popup(f"{frames_complete}\nSaving background.png...", None)
                    os.makedirs(os.path.dirname(background_path), exist_ok=True)
                    if not cv2.imwrite(background_path, background_snapshot):
                        raise RuntimeError(f"Failed to save background: {background_path}")

                self._set_progress_popup(f"{frames_complete}\nSaving blob pickle...", None)
                self._save_filtered_pickle_to_path(
                    pickle_path,
                    source=source,
                    metadata=pickle_metadata,
                )
                temporal_result = filters.stats.get("temporal_result")
                if temporal_result is not None:
                    self._set_progress_popup(f"{frames_complete}\nSaving temporal merge events...", None)
                    try:
                        temporal_result.write_csv(os.path.join(output_dir, TEMPORAL_EVENTS_CSV))
                    except OSError as exc:
                        # The pickle is complete; a locked report file must not fail the run.
                        filters.stats["csv_error"] = str(exc)
                if launch_tracking:
                    self._set_progress_popup(f"{frames_complete}\nLaunching Easy Tracking...", None)
                    self._launch_easy_tracking(project_dir, video_path)
                self._set_progress_popup(f"{frames_complete}\nCompleted", None)
                completed[0] = (config_path, background_path, pickle_path, project_dir)
                return True
            except Exception as e:
                completed[0] = e
                return False

        self.set_status("Processing...", auto_clear=False)
        _t0 = time.perf_counter()
        def _finish_processing(ok):
            _elapsed = time.perf_counter() - _t0
            if not ok or self._progress_cancel:
                if isinstance(completed[0], Exception):
                    messagebox.showerror("Error", str(completed[0]))
                    self.set_status("", auto_clear=False)
                return
            config_path, background_path, pickle_path, project_dir = completed[0]
            self._last_filter_stats = filters.stats
            # Tell the invoking Easy Tracking GUI where the actual completed
            # segmentation was saved. An existing segmentation GUI config can
            # redirect output to a different session than the launch argument.
            handoff_path = getattr(self, "_easy_handoff_file", "")
            if handoff_path:
                try:
                    if not (os.path.isfile(pickle_path) and os.path.isfile(background_path)):
                        raise FileNotFoundError("Segmentation output is incomplete")
                    handoff = {
                        "video_path": self.video_path_var.get().strip(),
                        "session_path": project_dir,
                        "pickle_path": pickle_path,
                        "background_path": background_path,
                    }
                    temp_handoff = handoff_path + ".tmp"
                    with open(temp_handoff, "w", encoding="utf-8") as file:
                        json.dump(handoff, file)
                    os.replace(temp_handoff, handoff_path)
                except (OSError, ValueError) as exc:
                    print(f"[Segmentation] Could not write Easy Tracking handoff: {exc}", flush=True)
            self.config_path = config_path
            try:
                self._write_processing_log(
                    _elapsed,
                    self.video_path_var.get().strip(),
                    self.frame_count,
                    pickle_path,
                    self.segmentation_mode_var.get(),
                )
            except Exception:
                pass
            self.set_status(f"Processing complete: {project_dir} / config saved: {config_path}")

            # Hide this window immediately so the next GUI is visible.
            # Show the completion popup after a short delay (the next GUI needs time to appear).
            self.withdraw()
            msg = f"Saved:\n{config_path}\n{background_path}\n{pickle_path}"

            def _show_done():
                messagebox.showinfo("Completed", msg)
                self.on_close()

            self.after(1200, _show_done)

        self._with_progress_popup("Processing", "Preparing...", work, _finish_processing)

    def _default_screenshot_path(self) -> str:
        stem = os.path.splitext(os.path.basename(self.video_path_var.get().strip() or "segmentation"))[0]
        return os.path.join(self._default_output_dir(), f"{stem}_frame{int(self.current_frame):06d}.png")

    def save_canvas_screenshot(self):
        if self.frame_bgr_cache is None:
            messagebox.showerror("Error", "No frame is currently visible.")
            return

        default_path = self._default_screenshot_path()
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save screenshot",
            defaultextension=".png",
            initialdir=os.path.dirname(default_path),
            initialfile=os.path.basename(default_path),
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        root, ext = os.path.splitext(path)
        if not ext:
            path = root + ".png"

        try:
            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            Image.fromarray(self._render_canvas_view_rgb()).save(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save screenshot:\n{exc}")
            return

        self.set_status(f"Screenshot saved: {path}")

    def _default_labeled_video_path(self) -> str:
        stem = os.path.splitext(os.path.basename(self.video_path_var.get().strip() or "segmentation"))[0]
        return os.path.join(self._default_output_dir(), f"{stem}_labeled.mp4")

    def _show_export_labeled_video_dialog(self) -> Optional[dict[str, object]]:
        if self.reader is None or self.frame_count <= 0:
            messagebox.showerror("Error", "Please load a video first.")
            return None

        total = int(self.frame_count)
        last = total - 1
        default_start = max(0, min(last, int(self.training_frame_start_var.get())))
        default_end = max(default_start, self._resolved_frame_end(int(self.training_frame_end_var.get())))

        popup = ctk.CTkToplevel(self)
        popup.title("Export Labeled Video")
        popup.transient(self)
        popup.resizable(False, False)
        popup.configure(fg_color=_POPUP_BG)

        result: dict[str, object] = {"applied": False}
        output_path_var = tk.StringVar(value=self._default_labeled_video_path())

        outer = tk.Frame(popup, padx=18, pady=16, bg=_POPUP_BG)
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer,
            text=f"Total frames: {total:,}\nValid frame indices: 0-{last:,}",
            justify="left",
            anchor="w", bg=_POPUP_BG, fg=_POPUP_FG,
        ).pack(fill="x", pady=(0, 12))

        path_frame = tk.LabelFrame(outer, text="Output", padx=10, pady=8, bg=_POPUP_BG, fg=_POPUP_FG)
        path_frame.pack(fill="x", pady=(0, 10))
        path_row = tk.Frame(path_frame, bg=_POPUP_BG)
        path_row.pack(fill="x")
        path_entry = tk.Entry(
            path_row, textvariable=output_path_var, width=54,
            bg="#343638", fg=_POPUP_FG, insertbackground=_POPUP_FG,
            selectbackground="#1F6AA5", selectforeground="#FFFFFF",
        )
        path_entry.pack(side="left", fill="x", expand=True)

        def _browse_output_path() -> None:
            current = output_path_var.get().strip() or self._default_labeled_video_path()
            selected = filedialog.asksaveasfilename(
                parent=popup,
                title="Save labeled video as",
                defaultextension=".mp4",
                initialdir=os.path.dirname(current) or self._default_output_dir(),
                initialfile=os.path.basename(current),
                filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            )
            if selected:
                output_path_var.set(selected)

        _popup_button(path_row, text="Browse...", command=_browse_output_path).pack(side="left", padx=(8, 0))

        range_frame = tk.LabelFrame(outer, text="Frame range", padx=10, pady=8, bg=_POPUP_BG, fg=_POPUP_FG)
        range_frame.pack(fill="x", pady=(0, 10))
        entries: dict[str, tk.Spinbox] = {}

        def _range_row(label: str, key: str, initial: int) -> None:
            row = tk.Frame(range_frame, bg=_POPUP_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=12, anchor="w", bg=_POPUP_BG, fg=_POPUP_FG).pack(side="left")
            sp = dark_spinbox(row, from_=0, to=last, increment=1, width=12, **_SPIN_CFG)
            sp.delete(0, tk.END)
            sp.insert(0, str(int(initial)))
            sp.pack(side="left")
            self._register_popup_integer_entry(sp)
            entries[key] = sp

        _range_row("Start frame", "start", default_start)
        _range_row("End frame", "end", default_end)

        error_var = tk.StringVar(value="")
        tk.Label(outer, textvariable=error_var, anchor="w", justify="left", fg="#FF8787", bg=_POPUP_BG).pack(fill="x")

        buttons = tk.Frame(outer, bg=_POPUP_BG)
        buttons.pack(fill="x", pady=(12, 0))

        def _normalized_output_path() -> str:
            value = output_path_var.get().strip().strip('"')
            if not value:
                raise ValueError("Output path is required.")
            root, ext = os.path.splitext(value)
            if not ext:
                value = root + ".mp4"
            return os.path.abspath(os.path.expanduser(value))

        def _read_frame_range() -> tuple[int, int]:
            try:
                start = int(str(entries["start"].get()).strip())
                end = int(str(entries["end"].get()).strip())
            except Exception as exc:
                raise ValueError("Frame range must be integer values.") from exc
            if not (0 <= start <= end <= last):
                raise ValueError(f"Frame range must satisfy 0 <= start <= end <= {last}.")
            return start, end

        def _apply() -> None:
            try:
                out_path = _normalized_output_path()
                start, end = _read_frame_range()
            except Exception as exc:
                error_var.set(str(exc))
                return
            result.update({
                "applied": True,
                "output_path": out_path,
                "frame_start": start,
                "frame_end": end,
            })
            popup.destroy()

        def _cancel() -> None:
            popup.destroy()

        _popup_button(buttons, text="Export", command=_apply).pack(side="right", padx=(8, 0))
        _popup_button(buttons, text="Cancel", command=_cancel).pack(side="right")
        popup.protocol("WM_DELETE_WINDOW", _cancel)

        popup.update_idletasks()
        width = max(560, popup.winfo_reqwidth())
        height = max(320, popup.winfo_reqheight())
        root_x = self.winfo_rootx()
        root_y = self.winfo_rooty()
        root_w = max(1, self.winfo_width())
        root_h = max(1, self.winfo_height())
        popup.geometry(f"{width}x{height}+{root_x + max(0, (root_w - width) // 2)}+{root_y + max(0, (root_h - height) // 2)}")
        popup.grab_set()
        popup.focus_force()
        path_entry.focus_set()
        self.wait_window(popup)

        if not result.get("applied"):
            return None
        return result

    def export_labeled_video(self):
        if self._progress_busy:
            return
        if self.reader is None:
            messagebox.showerror("Error", "No video has been loaded.")
            return
        if self._segmentation_needs_background() and self.background_bgr is None:
            messagebox.showerror("Error", "In this mode, please compute the background first.")
            return
        if not self.outlier_section_revealed:
            messagebox.showerror("Error", "Please run Analyze first.")
            return
        if self._needs_area_analysis() and not self.analysis_iqr_stats:
            messagebox.showerror("Error", "Please run Analyze sampled frames first.")
            return
        if self.static_exclusion_enabled_var.get() and not self._static_search_is_current():
            messagebox.showwarning(
                "Persistent static blobs", "Press Search before exporting labeled video.",
            )
            return
        if self.temporal_merge_enabled_var.get() and not self._temporal_search_is_current():
            messagebox.showwarning(
                "Temporal merges", "Press Search before exporting labeled video.",
            )
            return

        options = self._show_export_labeled_video_dialog()
        if options is None:
            return
        out_path = str(options["output_path"])
        start_frame = int(options["frame_start"])
        end_frame = int(options["frame_end"])
        export_frame_count = end_frame - start_frame + 1

        # Snapshot all render/ROI settings on the main thread before handing off.
        show_overlay = self.show_overlay_var.get()
        show_contours = self.show_contours_var.get()
        show_centers = self.show_centers_var.get()
        seg_cfg = self._capture_seg_config()
        roi_sets_snapshot = seg_cfg.roi_sets
        result_match = self._result_match
        show_result_obb = bool(self.result_import_show_obb_var.get())
        export_bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats)
        filters = self._search_filter_snapshot()
        fps = self.reader.fps if self.reader.fps > 0 else 25.0
        h, w = self.reader.height, self.reader.width

        completed: list = [None]

        def work():
            try:
                self._set_progress_popup("Segmenting selected frames...", 5)
                source = self._build_export_source_all_frames(
                    frame_start=start_frame,
                    frame_end=end_frame,
                    cfg=seg_cfg,
                    bounds=export_bounds,
                    progress_label="Segmenting selected frames",
                    result_match=result_match,
                    filters=filters,
                )
                if self._progress_cancel or not source:
                    return False

                out_dir = os.path.dirname(out_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

                start_time = time.perf_counter()
                try:
                    for n, fid in enumerate(range(start_frame, end_frame + 1), start=1):
                        if self._progress_cancel:
                            return False
                        frame_bgr = self.reader.read_bgr(fid)
                        blobs = source.get(fid, [])
                        result_obbs = (
                            result_match.obbs_by_frame.get(fid)
                            if result_match is not None and show_result_obb
                            else None
                        )
                        labeled = CrossingReviewApp._render_labeled_frame(
                            frame_bgr, blobs,
                            show_overlay=show_overlay,
                            show_contours=show_contours,
                            show_centers=show_centers,
                            roi_sets=roi_sets_snapshot,
                            frame_idx=fid,
                            result_obbs=result_obbs,
                            roi_reverse=seg_cfg.roi_reverse,
                        )
                        writer.write(labeled)
                        if n == 1 or n % 30 == 0 or fid == end_frame:
                            elapsed = max(1e-9, time.perf_counter() - start_time)
                            spd = n / elapsed
                            eta = max(0, export_frame_count - n) / spd if spd > 0 else 0.0
                            self._set_progress_popup(
                                f"Writing labeled video... {n}/{export_frame_count} (frame {fid})\n"
                                f"speed={spd:.1f} fps / remaining={self._format_seconds(eta)}",
                                35.0 + 65.0 * n / export_frame_count,
                            )
                finally:
                    writer.release()

                completed[0] = (out_path, start_frame, end_frame)
                return True
            except Exception as e:
                completed[0] = e
                return False

        self.set_status("Exporting labeled video...", auto_clear=False)
        def _finish_export(ok):
            if not ok or self._progress_cancel:
                if isinstance(completed[0], Exception):
                    messagebox.showerror("Error", str(completed[0]))
                self.set_status("", auto_clear=False)
                return
            saved_path, saved_start, saved_end = completed[0]
            self.set_status(f"Labeled video saved: {saved_path}")
            note = ""
            mismatched = int(filters.stats.get("mismatched_frames", 0))
            if mismatched:
                note = (f"\n\n{mismatched} frame(s) segmented differently from the Search; "
                        "optional filter colors were omitted there.")
            messagebox.showinfo("Completed", f"Labeled video saved:\n{saved_path}\nframes: {saved_start}-{saved_end}{note}")

        self._with_progress_popup("Export Labeled Video", "Preparing...", work, _finish_export)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="")
    parser.add_argument(
        "--dropped-video", default="",
        help="Open a video dropped onto Home using the normal compatibility check.",
    )
    parser.add_argument("--session", default="")
    parser.add_argument("--load-config", default="", help="Segmentation GUI JSON config to load.")
    parser.add_argument("--no-launch-tracking", action="store_true")
    parser.add_argument("--return-to-easy", action="store_true")
    parser.add_argument("--easy-handoff-file", default="")
    args = parser.parse_args()

    configure_taskbar_identity()
    app = CrossingReviewApp(no_launch_tracking=args.no_launch_tracking or args.return_to_easy)
    app._easy_handoff_file = str(args.easy_handoff_file or "").strip() if args.return_to_easy else ""
    app.session_path = str(args.session or "").strip()
    if args.load_config:
        app.after(300, lambda path=args.load_config: app.load_config(path))
    elif args.dropped_video:
        # Home drops follow the same validation, optional conversion, and
        # existing-config recovery path as selecting a video inside this GUI.
        app.after(300, lambda path=args.dropped_video: app._handle_selected_video(path))
    elif args.video and os.path.isfile(args.video):
        def _load_video_or_existing_config():
            found = app._find_existing_config_for_video(args.video, args.session)
            if found:
                path, config = found
                try:
                    app._load_config_dict(path, config)
                    return
                except Exception as exc:
                    app.set_status(f"Failed to load existing config ({path}): {exc}")
            app.load_video(args.video)
        app.after(300, _load_video_or_existing_config)
    app.mainloop()


if __name__ == "__main__":
    main()
