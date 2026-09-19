# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import importlib.metadata
import platform
import sys
import csv
import json
import math
import pickle
import queue
import re
import shutil
import subprocess
import threading
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
except ImportError:  # Preserve direct execution with: python gui/gui_segmentation.py
    from project_paths import PROJECT_ROOT, ensure_import_paths, gui_asset, gui_script
    from window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon

ensure_import_paths(PROJECT_ROOT)
from gui.color import OBB_COLOR, OUTLIER_COLOR, ROI_COLOR, WHITE
from main.path_utils import resolve_config_paths
from main.video_frame_count import detect_seekable_frame_count

APP_TITLE = "AMADEUS: Segmentation"
WINDOW_W = 1720
WINDOW_H = 980
WINDOW_MIN_W = 1200
WINDOW_MIN_H = 860
CANVAS_BG = "black"
CTK_THEME = str(gui_asset("deep_green.json"))

VIDEO_EXTS = [("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")]
VIDEO_DROP_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v"})
PKL_EXTS = [("Pickle files", "*.pickle *.pkl"), ("All files", "*.*")]
PNG_EXTS = [("PNG files", "*.png"), ("All files", "*.*")]
RESULT_CSV_EXTS = [("Result CSV files", "*.csv"), ("All files", "*.*")]

# Imported-result overlay: magenta stays readable next to the cyan/orange blobs.
RESULT_OBB_COLOR = (255, 0, 255)
# Fraction of a blob that its single imported OBB must cover for the blob to be
# accepted as one large animal instead of an area outlier.
DEFAULT_RESULT_OBB_COVERAGE = 0.7

ZOOM_IN_FACTOR = 1.12
ZOOM_OUT_FACTOR = 1.0 / ZOOM_IN_FACTOR
MIN_ZOOM = 0.2
MAX_ZOOM = 20.0

FRAME_CACHE_SIZE = 20
PREFETCH_FORWARD = 10
PREFETCH_BACKWARD = 4
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
# The area IQR sliders cover the usual working range. The spinbox keeps the
# wider data-derived limit, so a larger multiplier can still be typed or stepped.
AREA_IQR_SLIDER_MAX = 5.0
# Geometry sliders span the loaded image; the spinbox keeps a far wider limit so a
# value outside the frame can still be typed. Radii reach twice the long edge,
# which covers a circle centred anywhere in the image (diagonal < 1.5 x long edge).
ROI_ENTRY_MAX_PX = 1000000
ROI_RADIUS_SLIDER_LONG_EDGE_FACTOR = 2
REGION_EXPAND_SLIDER_MAX_PX = 50
REGION_EXPAND_ENTRY_MAX_PX = 1000
# Grabbing the ROI outline is judged in canvas pixels, so the tolerance stays the
# same on screen at any zoom level.
ROI_EDGE_GRAB_CANVAS_PX = 8
TK_CONTROL_MASK = 0x0004
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
    insertbackground="#dce4ee",
    buttonbackground="#565b5e",
    relief="flat",
    highlightthickness=2,
    highlightbackground="#565b5e",
    highlightcolor="#1f8040",
    selectbackground="#1f6aa5", selectforeground="white",
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
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else 0.0

        self.prefetch_cap = cv2.VideoCapture(video_path)
        if not self.prefetch_cap.isOpened():
            self.prefetch_cap = None

        from collections import OrderedDict
        self.cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cap_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()
        self._cap_pos: int = -1
        self._prefetch_pos: int = -1

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
        self, cap: cv2.VideoCapture, frame_idx: int, current_pos: int
    ) -> "tuple[np.ndarray, int, int]":
        if current_pos != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or not self._is_valid_frame(frame):
            recovered = self._decode_bgr_from_nearby(cap, frame_idx)
            if recovered is not None:
                return recovered
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")
        return frame, frame_idx + 1, frame_idx

    def _cache_put(self, frame_idx: int, frame_bgr: np.ndarray):
        if frame_idx in self.cache:
            self.cache.move_to_end(frame_idx)
        else:
            self.cache[frame_idx] = frame_bgr
            while len(self.cache) > FRAME_CACHE_SIZE:
                self.cache.popitem(last=False)

    def _clamp_frame_idx(self, frame_idx: int) -> int:
        frame_idx = int(frame_idx)
        if self.frame_count > 0:
            frame_idx = max(0, min(self.frame_count - 1, frame_idx))
        return frame_idx

    def read_bgr_with_index(self, frame_idx: int) -> "tuple[np.ndarray, int]":
        frame_idx = self._clamp_frame_idx(frame_idx)
        if frame_idx in self.cache:
            self.cache.move_to_end(frame_idx)
            return self.cache[frame_idx].copy(), frame_idx
        with self._cap_lock:
            if frame_idx in self.cache:
                self.cache.move_to_end(frame_idx)
                return self.cache[frame_idx].copy(), frame_idx
            frame, self._cap_pos, actual_idx = self._decode_bgr(self.cap, frame_idx, self._cap_pos)
            self._cache_put(actual_idx, frame)
            return frame.copy(), actual_idx

    def read_bgr(self, frame_idx: int) -> np.ndarray:
        frame, _actual_idx = self.read_bgr_with_index(frame_idx)
        return frame

    def prefetch(self, frame_idx: int):
        if self.prefetch_cap is None:
            return
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= self.frame_count or frame_idx in self.cache:
            return
        with self._prefetch_lock:
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
        self.cap.release()
        if self.prefetch_cap is not None:
            self.prefetch_cap.release()

    def clear_cache(self):
        """Release preview frames before a memory-intensive batch operation."""
        with self._cap_lock:
            with self._prefetch_lock:
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
    manual_outlier: bool = False
    is_crossing: bool = False
    result_obb_count: int = 0        # imported OBBs owned by this blob
    result_obb_coverage: float = 0.0  # blob fraction covered by that single OBB
    result_single: bool = False       # area outlier rescued as one large animal


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
    roi_sets: list                         # geometry dicts; per-frame mask built inside _segment_with_config
    additional_outlier_sets: list          # segmentation settings; converted to blobs per frame
    background_bgr: Optional[np.ndarray]  # read-only shared reference
    additional_backgrounds: dict[str, np.ndarray] = field(default_factory=dict)
    # Bounds used to identify the normal non-outlier blobs whose pixels must be
    # protected from Additional Outlier segmentation.  During the first IQR
    # analysis pass this is None, so every normal blob is protected.
    protected_area_bounds: Optional[tuple[float, float]] = None


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
    if _CUDA_AVAILABLE and _torch is not None and _mode_requires_background(mode):
        mask = _segment_diff_gpu(frame_bgr, local_cfg)
    else:
        mask = _segment_cpu(frame_bgr, local_cfg)

    if local_cfg.invert_mask:
        mask = cv2.bitwise_not(mask)
    roi_mask = _build_roi_mask_for_frame(local_cfg.roi_sets, frame_bgr.shape[:2], frame_idx)
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


def _build_single_roi_mask(image_shape: tuple, roi: dict) -> "Optional[np.ndarray]":
    img_h, img_w = image_shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    shape = roi.get("shape", "circle")
    if shape == "circle":
        cx = max(0, min(img_w - 1, int(roi.get("x", 0))))
        cy = max(0, min(img_h - 1, int(roi.get("y", 0))))
        radius = max(0, int(roi.get("w", 0)))
        if radius <= 0:
            return None
        cv2.circle(mask, (cx, cy), radius, 255, thickness=cv2.FILLED)
    else:
        x = max(0, min(img_w, int(roi.get("x", 0))))
        y = max(0, min(img_h, int(roi.get("y", 0))))
        w = max(0, int(roi.get("w", 0)))
        h_val = max(0, int(roi.get("h", 0)))
        x2 = max(x, min(img_w, x + w))
        y2 = max(y, min(img_h, y + h_val))
        if x2 <= x or y2 <= y:
            return None
        mask[y:y2, x:x2] = 255
    return mask


def _build_roi_mask_for_frame(roi_sets: list, image_shape: tuple, frame_idx: int) -> "Optional[np.ndarray]":
    """Intersection mask of all enabled ROI sets that cover frame_idx. -1 in frame_end = last frame."""
    combined: "Optional[np.ndarray]" = None
    for roi in roi_sets:
        if not roi.get("enabled", False):
            continue
        fs = int(roi.get("frame_start", 0))
        fe = int(roi.get("frame_end", -1))
        if frame_idx < fs:
            continue
        if fe >= 0 and frame_idx > fe:
            continue
        partial = _build_single_roi_mask(image_shape, roi)
        if partial is None:
            continue
        if bool(roi.get("reverse", False)):
            partial = cv2.bitwise_not(partial)
        combined = partial if combined is None else cv2.bitwise_and(combined, partial)
    return combined


# Merge-only corridors below are the exact shortest connection between two
# regions; one pixel of slack is what keeps that set connected once it is
# rasterized. Distances there use DIST_MASK_5 rather than DIST_MASK_PRECISE:
# the precise transform is not bit-reproducible across identical calls, and a
# segmentation run has to give the same mask every time.
_BRIDGE_TOLERANCE_PX = 1.0
_BRIDGE_DIST_MASK = 5


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
    frame_offset: int
    min_coverage: float
    obbs_by_frame: dict          # shared read-only reference


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


def _polygon_bounds(poly: np.ndarray) -> tuple[int, int, int, int]:
    """Integer half-open [x0, x1) x [y0, y1) bounding box of a polygon."""
    return (
        int(math.floor(float(poly[:, 0].min()))),
        int(math.floor(float(poly[:, 1].min()))),
        int(math.ceil(float(poly[:, 0].max()))) + 1,
        int(math.ceil(float(poly[:, 1].max()))) + 1,
    )


def _rasterize_overlap(contour: np.ndarray, poly: np.ndarray,
                       x0: int, y0: int, x1: int, y1: int) -> tuple[int, int]:
    """Pixels of the blob, and of the blob covered by the OBB, inside a window."""
    blob_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.drawContours(blob_mask, [contour], -1, 1, cv2.FILLED, offset=(-x0, -y0))
    obb_mask = np.zeros_like(blob_mask)
    cv2.fillConvexPoly(obb_mask, np.round(poly - (x0, y0)).astype(np.int32), 1)
    return int(np.count_nonzero(blob_mask)), int(np.count_nonzero(blob_mask & obb_mask))


def _match_result_obbs_to_blobs(blobs: list, frame_idx: int, cfg: "_ResultMatchConfig") -> None:
    """Attach the imported OBBs of one frame to the blobs of that same frame.

    Every OBB is owned by exactly one blob -- the blob containing its centre, or
    else the blob it overlaps most -- so two animals sharing a blob give that
    blob a count of 2.  The covered fraction, which separates one large animal
    (the OBB covers the blob) from a pair whose second individual the tracker
    missed (it does not), is measured only for the blobs that fraction can
    still rescue: single-OBB normal-segmentation area outliers that are not
    Additional Outlier regions.
    """
    for blob in blobs:
        blob.result_obb_count = 0
        blob.result_obb_coverage = 0.0
        blob.result_single = False
    obbs = cfg.obbs_by_frame.get(int(frame_idx) + int(cfg.frame_offset))
    if obbs is None or not blobs:
        return

    boxes = [cv2.boundingRect(blob.contour) for blob in blobs]
    owners: list[int] = []
    for poly in obbs:
        center = (float(poly[:, 0].mean()), float(poly[:, 1].mean()))
        owner = -1
        for index, blob in enumerate(blobs):
            bx, by, bw, bh = boxes[index]
            if not (bx <= center[0] <= bx + bw and by <= center[1] <= by + bh):
                continue
            if cv2.pointPolygonTest(blob.contour, center, False) >= 0:
                owner = index
                break
        if owner < 0:
            owner = _owner_by_largest_overlap(blobs, boxes, poly)
        owners.append(owner)
        if owner >= 0:
            blobs[owner].result_obb_count += 1

    for index, blob in enumerate(blobs):
        if blob.result_obb_count != 1 or blob.manual_outlier or not blob.area_outlier:
            continue
        bx, by, bw, bh = boxes[index]
        blob_pixels, covered_pixels = _rasterize_overlap(
            blob.contour, obbs[owners.index(index)], bx, by, bx + bw, by + bh
        )
        if blob_pixels > 0:
            blob.result_obb_coverage = covered_pixels / blob_pixels


def _owner_by_largest_overlap(blobs: list, boxes: list, poly: np.ndarray) -> int:
    """Blob index sharing the most area with an OBB whose centre hit background."""
    ox0, oy0, ox1, oy1 = _polygon_bounds(poly)
    best_index, best_overlap = -1, 0
    for index, blob in enumerate(blobs):
        bx, by, bw, bh = boxes[index]
        x0, y0 = max(ox0, bx), max(oy0, by)
        x1, y1 = min(ox1, bx + bw), min(oy1, by + bh)
        if x1 <= x0 or y1 <= y0:
            continue
        _blob_pixels, overlap = _rasterize_overlap(blob.contour, poly, x0, y0, x1, y1)
        if overlap > best_overlap:
            best_index, best_overlap = index, overlap
    return best_index


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
        self.roi_shape_var = tk.StringVar(value="circle")
        self.roi_x_var = tk.IntVar(value=0)
        self.roi_y_var = tk.IntVar(value=0)
        self.roi_w_var = tk.IntVar(value=0)
        self.roi_h_var = tk.IntVar(value=0)
        self.roi_reverse_var = tk.BooleanVar(value=False)
        self.roi_frame_start_var = tk.IntVar(value=0)
        self.roi_frame_end_var = tk.IntVar(value=-1)
        self.roi_sets: list[dict] = [{"enabled": False, "reverse": False, "shape": "circle", "x": 0, "y": 0, "w": 0, "h": 0, "frame_start": 0, "frame_end": -1}]
        self.roi_active_set_idx: int = 0
        self._switching_roi_set: bool = False
        self.roi_set_combo: Optional[ctk.CTkComboBox] = None

        self.region_expand_px_var = tk.IntVar(value=0)
        self.region_expand_merge_only_var = tk.BooleanVar(value=False)

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

        self.result_import_enabled_var = tk.BooleanVar(value=False)
        self.result_import_path_var = tk.StringVar(value="")
        self.result_import_frame_offset_var = tk.IntVar(value=0)
        self.result_import_min_coverage_var = tk.DoubleVar(value=DEFAULT_RESULT_OBB_COVERAGE)
        self.result_import_show_obb_var = tk.BooleanVar(value=True)
        self.result_import_status_var = tk.StringVar(value="result OBB: not loaded")

        self.background_frame_start_var = tk.IntVar(value=0)
        self.background_frame_end_var = tk.IntVar(value=-1)
        self.training_frame_start_var = tk.IntVar(value=0)
        self.training_frame_end_var = tk.IntVar(value=-1)
        self.training_frame_interval_var = tk.IntVar(value=5)

        self.area_outlier_method_var = tk.StringVar(value="absolute")
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
        self._min_area_slider_cap: Optional[int] = None
        self.analysis_blobs_by_frame: dict[int, list[BlobMetrics]] = {}
        self.analysis_frames: list[int] = []
        self.analysis_signature: Optional[tuple] = None
        self.analysis_bounds: dict[str, tuple[float, float]] = {}
        self.analysis_iqr_stats: dict[str, tuple[float, float, float]] = {}
        self.analysis_is_stale = True

        self.result_obb_import: Optional[_ResultObbImport] = None
        self.result_import_params_frame = None
        self._result_match: Optional[_ResultMatchConfig] = None

        self.tk_image = None
        self.bg_preview_button = None
        self.frame_ranges_button = None
        self.screenshot_button = None
        self.progress_popup = None
        self.progress_bar = None
        self._progress_indeterminate = False
        self.progress_label_var = tk.StringVar(value="")
        self.progress_value_var = tk.DoubleVar(value=0.0)
        self._progress_cancel = False
        self._progress_queue: queue.Queue = queue.Queue()
        self.bg_preview_popup = None
        self.bg_preview_image = None
        self.prefetch_after_id: Optional[str] = None
        self._prefetch_cancel = threading.Event()
        self.status_clear_job: Optional[str] = None
        self.playback_job: Optional[str] = None
        self.playback_active = False
        self.config_path: Optional[str] = None
        self.background_rebuild_job: Optional[str] = None
        self._applying_config = False
        self._loading_video = False
        self._background_building = False

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
        self._roi_scale_center: Optional[tuple[float, float]] = None

        self._build_ui()
        self._bind_events()
        self._bind_traces()
        _start_maximized(self)
        self.after(1200, _start_torch_init)

    def _maximize_window(self) -> None:
        _maximize_window_once(self)

    def _build_ui(self):
        top = ctk.CTkFrame(self, corner_radius=0)
        top.pack(fill="x", padx=8, pady=8)

        ctk.CTkLabel(top, text="Video").grid(row=0, column=0, sticky="w")
        ctk.CTkEntry(top, textvariable=self.video_path_var).grid(row=0, column=1, sticky="ew", padx=8)
        ctk.CTkButton(top, text="Reference", command=self.browse_video).grid(row=0, column=2)
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
        ctk.CTkButton(nav, text=">", width=40, command=lambda: self.step_frame(1)).pack(side="left", padx=4)
        ctk.CTkButton(nav, text=">|", width=40, command=lambda: self.set_frame(self.frame_count - 1)).pack(side="left", padx=(0, 12))
        ctk.CTkLabel(nav, text="frame").pack(side="left")
        self.frame_spin = tk.Spinbox(nav, width=10, textvariable=self.frame_var, command=self.on_frame_spin, **_SPIN_CFG)
        self.frame_spin.pack(side="left", padx=(6, 12))
        self.frame_spin.bind("<Return>", self.on_frame_entry_commit, add="+")
        self.frame_spin.bind("<KP_Enter>", self.on_frame_entry_commit, add="+")
        self.frame_spin.bind("<FocusOut>", self.on_frame_entry_commit, add="+")
        self.frame_scale = ctk.CTkSlider(nav, orientation="horizontal", command=self.on_frame_scale)
        self.frame_scale.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(nav, text="Fit window", width=90, command=self.fit_canvas_to_window).pack(side="left", padx=(8, 0))

        self.canvas = tk.Canvas(right, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._initialize_video_drag_and_drop()


    def _build_sidebar(self, parent: ctk.CTkFrame):
        def section(title: str, *, expanded: bool = False) -> ctk.CTkFrame:
            return self._make_collapsible_section(parent, title, expanded=expanded)

        self.seg_section = section("Background / Segmentation", expanded=True)
        self.roi_section = section("ROI (Optional)")
        self.region_expansion_section = section("Region Expansion (Optional)")
        self.ana_section = section("Outlier Extraction", expanded=True)
        self.result_section = section("Result OBB Import (beta)")
        self.additional_outlier_section = section("Additional Outlier (Optional)")

        seg = self.seg_section
        self._pack_combobox(seg, "Segmentation mode", self.segmentation_mode_var, ["background_diff", "dark_region", "bright_region", "dark_region_and_background_diff", "bright_region_and_background_diff"])

        self.bg_controls_frame = ctk.CTkFrame(seg, corner_radius=0)
        self.bg_controls_frame.pack(fill="x")
        self._pack_combobox(self.bg_controls_frame, "Background method", self.bg_method_var, list(BACKGROUND_METHODS))
        self.bg_preview_button = ctk.CTkButton(self.bg_controls_frame, text="Show background", command=self.show_background_popup, state="disabled")
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

        roi = self.roi_section
        _roi_set_row = ctk.CTkFrame(roi, corner_radius=0)
        _roi_set_row.pack(fill="x", padx=6, pady=(4, 2))
        ctk.CTkLabel(_roi_set_row, text="ROI Set", width=60, anchor="w").pack(side="left")
        self.roi_set_combo = ctk.CTkComboBox(_roi_set_row, values=["Set 1"], state="readonly", width=90, command=self._on_roi_set_selected)
        self.roi_set_combo.set("Set 1")
        self.roi_set_combo.pack(side="left", padx=(4, 2))
        ctk.CTkButton(_roi_set_row, text="+", width=28, command=self._add_roi_set).pack(side="left", padx=2)
        ctk.CTkButton(_roi_set_row, text="-", width=28, command=self._remove_roi_set).pack(side="left", padx=2)
        ctk.CTkCheckBox(roi, text="Use this ROI", variable=self.roi_enabled_var, command=self._toggle_roi_params).pack(anchor="w", padx=6, pady=(4, 4))
        self.roi_params_frame = ctk.CTkFrame(roi, corner_radius=0)
        ctk.CTkCheckBox(self.roi_params_frame, text="Reverse ROI", variable=self.roi_reverse_var).pack(anchor="w", padx=6, pady=(2, 4))
        self._pack_combobox(self.roi_params_frame, "ROI shape", self.roi_shape_var, ["rectangle", "circle"])
        self.roi_x_scale = self._pack_scale_entry(self.roi_params_frame, "ROI x / center x", self.roi_x_var, 0, ROI_ENTRY_MAX_PX, 1, slider_hi=10000)
        self.roi_y_scale = self._pack_scale_entry(self.roi_params_frame, "ROI y / center y", self.roi_y_var, 0, ROI_ENTRY_MAX_PX, 1, slider_hi=10000)
        self.roi_w_scale = self._pack_scale_entry(self.roi_params_frame, "ROI width / radius", self.roi_w_var, 0, ROI_ENTRY_MAX_PX, 1, slider_hi=10000)
        self.roi_h_scale = self._pack_scale_entry(self.roi_params_frame, "ROI height (rect)", self.roi_h_var, 0, ROI_ENTRY_MAX_PX, 1, slider_hi=10000)
        self.roi_frame_start_scale = self._pack_scale_entry(self.roi_params_frame, "From frame", self.roi_frame_start_var, 0, 999999, 1)
        self.roi_frame_end_scale = self._pack_scale_entry(self.roi_params_frame, "To frame (-1=end)", self.roi_frame_end_var, -1, 999999, 1, slider_lo=0, end_sentinel=-1)
        ctk.CTkButton(self.roi_params_frame, text="Reset ROI", command=self.reset_roi).pack(fill="x", padx=6, pady=(2, 6))
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
        ctk.CTkButton(self.analysis_run_frame, text="Analyze", command=self.analyze_sample_frames).pack(fill="x", padx=6, pady=(2, 6))

        self.area_iqr_frame = ctk.CTkFrame(ana, corner_radius=0)
        self.area_absolute_frame = ctk.CTkFrame(ana, corner_radius=0)
        self._build_area_iqr_control(self.area_iqr_frame)
        self._build_area_absolute_control(self.area_absolute_frame)
        self._toggle_area_outlier_method_controls()

        result = self.result_section
        ctk.CTkLabel(
            result,
            text=("Reads a finished tracking result CSV and keeps an area\n"
                  "outlier that holds exactly one OBB covering the blob.\n"
                  "Blobs that are not area outliers are never changed."),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=6, pady=(4, 2))
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
            self.result_import_params_frame, "Frame offset",
            self.result_import_frame_offset_var, -100000, 100000, 1,
        )
        self._pack_scale_entry(
            self.result_import_params_frame, "Min OBB coverage",
            self.result_import_min_coverage_var, 0.0, 1.0, 0.1, is_float=True,
        )
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
        )
        self.processing_button.pack(side="bottom", fill="x", padx=8, pady=(0, 6))


        self._refresh_segmentation_controls()
        self._refresh_segmentation_controls()
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

    def _set_workflow_stage(self, stage: str):
        """Keep all sections available; button state carries workflow readiness."""
        for section in [
            getattr(self, "seg_section", None),
            getattr(self, "roi_section", None),
            getattr(self, "region_expansion_section", None),
            getattr(self, "ana_section", None),
            getattr(self, "result_section", None),
            getattr(self, "additional_outlier_section", None),
        ]:
            if section is not None:
                self._pack_sidebar_section(section)
        self._update_analysis_dependent_buttons()

    def _update_analysis_dependent_buttons(self):
        # Processing stays clickable in every state so that pressing it explains
        # the missing step instead of silently doing nothing.
        ready = bool(self.analysis_iqr_stats) or not self._needs_area_analysis()
        state = "normal" if ready else "disabled"
        if hasattr(self, "export_button") and self.export_button is not None:
            self.export_button.configure(state=state)
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
            "analysis_sample_count": self.analysis_sample_count_var,
            "area_outlier_method": self.area_outlier_method_var,
            "area_iqr_min": self.area_iqr_min_var,
            "area_iqr_max": self.area_iqr_max_var,
            "area_absolute_min": self.area_absolute_min_var,
            "area_absolute_max": self.area_absolute_max_var,
            "result_import_enabled": self.result_import_enabled_var,
            "result_import_path": self.result_import_path_var,
            "result_import_frame_offset": self.result_import_frame_offset_var,
            "result_import_min_coverage": self.result_import_min_coverage_var,
            "result_import_show_obb": self.result_import_show_obb_var,
            "show_overlay": self.show_overlay_var,
            "show_centers": self.show_centers_var,
            "show_contours": self.show_contours_var,
        }

    def _collect_config(self) -> dict:
        self._sync_roi_vars_to_set(self.roi_active_set_idx)
        self._sync_additional_outlier_vars_to_set()
        settings = {}
        for key, var in self._config_vars().items():
            try:
                settings[key] = var.get()
            except Exception:
                pass
        settings["roi_sets"] = [dict(r) for r in self.roi_sets]
        settings["additional_outlier_sets"] = [dict(r) for r in self.additional_outlier_sets]

        analysis_iqr_stats = {
            k: [float(x) for x in v]
            for k, v in self.analysis_iqr_stats.items()
        }
        analysis_bounds = {
            k: [float(v[0]), float(v[1])]
            for k, v in self.analysis_bounds.items()
        }

        return {
            "version": 4,
            "app": APP_TITLE,
            "settings": settings,
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
        """Fit the ROI and frame sliders to the loaded video.

        The spinbox keeps the wide limit, so a coordinate outside the frame or a
        radius beyond twice the long edge can still be entered as a number.
        """
        if self.reader is None:
            return
        width = max(1, int(self.reader.width))
        height = max(1, int(self.reader.height))
        radius_max = ROI_RADIUS_SLIDER_LONG_EDGE_FACTOR * max(width, height)
        for scale, slider_upper in (
            (self.roi_x_scale, width),
            (self.roi_y_scale, height),
            (self.roi_w_scale, radius_max),
            (self.roi_h_scale, height),
        ):
            scale.set_value_range(0, ROI_ENTRY_MAX_PX, clamp_current=False, slider_upper=slider_upper)
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

        self._apply_frame_ranges_from_settings(settings)
        self._toggle_area_outlier_method_controls()

        roi_sets_data = settings["roi_sets"]
        if not isinstance(roi_sets_data, list):
            raise ValueError("settings.roi_sets must be a list.")
        loaded_roi_sets = [dict(item) for item in roi_sets_data if isinstance(item, dict)]
        self.roi_sets = (
            [self._normalize_roi_set(item) for item in loaded_roi_sets]
            if loaded_roi_sets
            else [self._default_roi_set()]
        )
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

        result_import_path = str(settings.get("result_import_path", "") or "")
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
            self.analysis_frames = []
            self.analysis_signature = None
            self.analysis_bounds = {}
            self.analysis_iqr_stats = {}
            self.analysis_is_stale = True
            self.analysis_ready_var.set("analysis: not computed")

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
        if (
            self.reader is not None
            and self._segmentation_needs_background()
            and self.background_bgr is None
            and not hidden_state.get("analysis_iqr_stats")
        ):
            try:
                self.compute_background()
            except Exception as exc:
                self.set_status(f"Background build failed: {exc}")

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
            json.dump(self._collect_config(), f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
        self.config_path = path

    def _save_config_to_default_path(self) -> str:
        """Write to the config this session came from, or the default location."""
        path = self.config_path or self._default_config_path()
        self._write_config_to_path(path)
        return path

    def save_config(self):
        """Write the settings without asking where; Processing writes the same file."""
        try:
            path = self._save_config_to_default_path()
            self.set_status(f"Config saved: {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

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

    def load_config(self):
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
        if self._additional_outlier_needs_background():
            self._schedule_background_rebuild(delay_ms=50)
        self._area_reference_key = None
        self._min_area_slider_cap = None
        self._absolute_zoomed = False
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
        self.analysis_frames = []
        self.analysis_signature = None
        self.analysis_bounds = {}
        self.analysis_iqr_stats = {}
        self.analysis_is_stale = True
        self.analysis_ready_var.set(message)
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

        popup = tk.Toplevel(self)
        popup.title("Frame Range Settings")
        popup.transient(self)
        popup.resizable(False, False)
        popup.configure(bg="#f3f4f6")

        result = {"applied": False, "values": None}
        total = int(self.frame_count)
        last = total - 1

        outer = tk.Frame(popup, padx=18, pady=16, bg="#f3f4f6")
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer,
            text=f"Total frames: {total:,}\nValid frame indices: 0-{last:,}\nEnd frame = -1 uses the last frame.",
            justify="left",
            anchor="w",
            bg="#f3f4f6",
        ).pack(fill="x", pady=(0, 10))

        tk.Label(
            outer,
            text=(
                "These ranges are used only for background estimation and training-data preparation.\n"
                "They do not restrict the frame range of the final tracking analysis."
            ),
            justify="left",
            anchor="w",
            bg="#f3f4f6",
        ).pack(fill="x", pady=(0, 12))

        entries: dict[str, tk.Spinbox] = {}

        def _section(title: str) -> tk.Frame:
            frame = tk.LabelFrame(outer, text=title, padx=10, pady=8, bg="#f3f4f6")
            frame.pack(fill="x", pady=(0, 10))
            return frame

        def _row(parent: tk.Frame, label: str, key: str, initial: int, from_: int = 0, to_: Optional[int] = None) -> None:
            row = tk.Frame(parent, bg="#f3f4f6")
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=16, anchor="w", bg="#f3f4f6").pack(side="left")
            sp = tk.Spinbox(row, from_=from_, to=max(int(to_ if to_ is not None else last), 1), increment=1, width=12, **_SPIN_CFG)
            sp.delete(0, tk.END)
            sp.insert(0, str(int(initial)))
            sp.pack(side="left")
            entries[key] = sp

        bg_frame = _section("Background estimation")
        _row(bg_frame, "Start frame", "background_frame_start", int(self.background_frame_start_var.get()))
        _row(bg_frame, "End frame (-1=last)", "background_frame_end", int(self.background_frame_end_var.get()), from_=-1)

        tr_frame = _section("Training data preparation")
        _row(tr_frame, "Start frame", "training_frame_start", int(self.training_frame_start_var.get()))
        _row(tr_frame, "End frame (-1=last)", "training_frame_end", int(self.training_frame_end_var.get()), from_=-1)
        _row(tr_frame, "Frame interval", "training_frame_interval", max(1, int(self.training_frame_interval_var.get())), from_=1, to_=1000000)

        error_var = tk.StringVar(value="")
        tk.Label(outer, textvariable=error_var, anchor="w", justify="left", fg="#b00020", bg="#f3f4f6").pack(fill="x")

        buttons = tk.Frame(outer, bg="#f3f4f6")
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

        tk.Button(buttons, text="Apply", width=10, command=_apply).pack(side="right", padx=(8, 0))
        tk.Button(buttons, text="Cancel", width=10, command=_cancel).pack(side="right")
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

    def _pack_scale_entry(self, parent, label, var, lo, hi, resolution, is_float: bool = False, enable_var: Optional[tk.BooleanVar] = None, entry_prefix: str = "", slider_lo=None, slider_hi=None, end_sentinel: Optional[int] = None):
        outer = ctk.CTkFrame(parent, corner_radius=0)
        outer.pack(fill="x", padx=6, pady=3)

        row = ctk.CTkFrame(outer, corner_radius=0)
        row.pack(fill="x")
        if enable_var is not None:
            ctk.CTkCheckBox(row, text="", variable=enable_var).pack(side="left", padx=(0, 4))
        ctk.CTkLabel(row, text=label, width=144, anchor="w").pack(side="left")

        fmt = "{:.6g}" if entry_prefix == "IQRx" else ("{:.1f}" if is_float else "{:d}")
        default_value = float(var.get()) if is_float else int(var.get())
        state = {"updating": False}

        spinbox = tk.Spinbox(row, from_=lo, to=hi, increment=resolution, width=8, **_SPIN_CFG)

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

    def _build_area_absolute_control(self, parent):
        row = ctk.CTkFrame(parent, corner_radius=0)
        row.pack(fill="x")
        row.grid_columnconfigure((0, 1), weight=1)
        self._absolute_entries = []
        self._absolute_spins = []
        self._absolute_slider_cap = 100000
        self._absolute_zoomed = False
        variables = (self.area_absolute_min_var, self.area_absolute_max_var)
        validate = (self.register(lambda value: value == "" or value.isascii() and value.isdecimal()), "%P")
        for index, label in enumerate(("Minimum area", "Maximum area")):
            side = "w" if index == 0 else "e"
            ctk.CTkLabel(row, text=label).grid(row=0, column=index, sticky=side)
            entry = tk.StringVar(value=str(int(variables[index].get())))
            spin = tk.Spinbox(
                row, from_=0, to=sys.maxsize, increment=1, width=12,
                textvariable=entry, validate="key", validatecommand=validate,
                command=lambda i=index: self._commit_absolute_entry(i),
                **_SPIN_CFG,
            )
            spin.grid(row=1, column=index, sticky=side)
            for event in ("<Return>", "<KP_Enter>", "<FocusOut>"):
                spin.bind(event, lambda _e, i=index: self._commit_absolute_entry(i))
            self._absolute_entries.append(entry)
            self._absolute_spins.append(spin)
        self._absolute_canvas = tk.Canvas(parent, height=52, bg="white", highlightthickness=0)
        self._absolute_canvas.pack(fill="x", pady=(2, 0))
        self._absolute_canvas.bind("<Configure>", lambda _e: self._draw_absolute_slider())
        self._absolute_canvas.bind("<Button-1>", self._start_absolute_drag)
        self._absolute_canvas.bind("<Double-Button-1>", self._reset_absolute_handle_to_default)
        self._absolute_canvas.bind("<B1-Motion>", self._drag_absolute_handle)
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._absolute_canvas.bind(event, self._zoom_absolute_slider)
        for var in variables:
            var.trace_add("write", lambda *_: self._sync_absolute_control())
        self._sync_absolute_control()

    def _commit_absolute_entry(self, index):
        text = self._absolute_entries[index].get()
        if text and text.isascii() and text.isdecimal():
            variables = (self.area_absolute_min_var, self.area_absolute_max_var)
            if int(text) != int(variables[index].get()):
                self._set_absolute_threshold(index, int(text))
            self._absolute_slider_cap = max(self._absolute_slider_cap, int(variables[index].get()))
            self._draw_absolute_slider()
        else:
            self._sync_absolute_control()

    def _set_absolute_threshold(self, index, value):
        variables = (self.area_absolute_min_var, self.area_absolute_max_var)
        other = int(variables[1 - index].get())
        value = max(0, int(value))
        value = min(value, other) if index == 0 else max(value, other)
        self._absolute_defaults_pending = False
        variables[index].set(value)
        self._sync_absolute_control()

    def _sync_absolute_control(self, *, force=False):
        if getattr(self, "_updating_area_ranges", False) and not force:
            return
        values = (self.area_absolute_min_var.get(), self.area_absolute_max_var.get())
        for entry, value in zip(self._absolute_entries, values):
            entry.set(str(int(value)))
        self._draw_absolute_slider()

    def _absolute_handle_positions(self):
        width = max(1, self._absolute_canvas.winfo_width() - 24)
        return [12 + min(1.0, max(0.0, float(var.get()) / self._absolute_slider_cap)) * width
                for var in (self.area_absolute_min_var, self.area_absolute_max_var)]

    def _draw_absolute_slider(self):
        canvas = self._absolute_canvas
        canvas.delete("all")
        left, right = self._absolute_handle_positions()
        canvas.create_line(12, 18, max(12, canvas.winfo_width() - 12), 18,
                           fill="#c8c8c8", width=5)
        canvas.create_line(left, 18, right, 18, fill="#245b45", width=5)
        canvas.create_text(12, 43, text="0", anchor="w", fill="#333333")
        canvas.create_text(max(12, canvas.winfo_width() - 12), 43,
                           text=str(self._absolute_slider_cap), anchor="e", fill="#333333")
        # Off-scale handles stay at the right edge; zooming never changes thresholds.
        # Opposing triangles keep both handles visible even at equal thresholds.
        canvas.create_polygon(left - 7, 5, left + 7, 5, left, 18, fill="#245b45")
        canvas.create_polygon(right - 7, 31, right + 7, 31, right, 18, fill="#245b45")

    def _reset_absolute_handle_to_default(self, event):
        defaults = self._absolute_default_values
        if defaults is None:
            return "break"
        left, right = self._absolute_handle_positions()
        hit_min = abs(event.x - left) <= 9 and 3 <= event.y <= 20
        hit_max = abs(event.x - right) <= 9 and 16 <= event.y <= 33
        if hit_min and hit_max:
            index = 0 if event.y < 18 else 1
        elif hit_min:
            index = 0
        elif hit_max:
            index = 1
        else:
            return None
        self._set_absolute_threshold(index, defaults[index])
        return "break"

    def _start_absolute_drag(self, event):
        left, right = self._absolute_handle_positions()
        if abs(left - right) < 1:
            self._absolute_active_handle = 0 if event.y < 18 else 1
        else:
            self._absolute_active_handle = 0 if abs(event.x - left) <= abs(event.x - right) else 1
        self._drag_absolute_handle(event)

    def _drag_absolute_handle(self, event):
        width = max(1, self._absolute_canvas.winfo_width() - 24)
        fraction = max(0.0, min(1.0, (event.x - 12) / width))
        value = int(math.floor(fraction * self._absolute_slider_cap + 0.5))
        self._set_absolute_threshold(self._absolute_active_handle, value)

    def _zoom_absolute_slider(self, event):
        number = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if number == 4 or delta > 0:
            cap = max(1, int(self._absolute_slider_cap / 1.2))
        elif number == 5 or delta < 0:
            cap = max(self._absolute_slider_cap + 1, math.ceil(self._absolute_slider_cap * 1.2))
        else:
            return "break"
        self._absolute_zoomed = True
        self._absolute_slider_cap = cap
        self._draw_absolute_slider()
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
            if reference_changed and not self._absolute_zoomed:
                self._absolute_slider_cap = area_cap
            self._sync_absolute_control(force=True)
            if self._area_reference_max <= 0:
                return  # No detected blob provides a meaningful size reference yet.
            cap = max(1.0, float(math.ceil(self._area_reference_max * 1.2)))
            stat = self.analysis_iqr_stats.get("area", self._area_reference_stats)
            q1, median, q3, iqr = self._normalized_area_stat(stat)
            if iqr > 1e-9:
                upper = max(10.0, float(math.ceil(max(q1, cap - q3) / iqr)))
                # Round each median endpoint outward to a 0.1 step. The slider then
                # still reaches the median, which _current_area_bounds holds.
                lower_min = math.floor((q1 - median) / iqr * 10.0) / 10.0
                upper_min = math.floor((median - q3) / iqr * 10.0) / 10.0
            else:
                upper, lower_min, upper_min = 10.0, 0.0, 0.0
            self._area_iqr_range_max = upper
            self.area_iqr_min_scale.set_value_range(lower_min, upper, slider_upper=AREA_IQR_SLIDER_MAX)
            self.area_iqr_max_scale.set_value_range(upper_min, upper, slider_upper=AREA_IQR_SLIDER_MAX)
        finally:
            self._updating_area_ranges = False

    def _build_area_iqr_control(self, parent: ctk.CTkFrame):
        self.area_iqr_min_scale = self._pack_scale_entry(parent, "Lower IQR multiplier", self.area_iqr_min_var, 0.0, 10.0, 0.1, is_float=True, entry_prefix="IQRx")
        self.area_iqr_max_scale = self._pack_scale_entry(parent, "Upper IQR multiplier", self.area_iqr_max_var, 0.0, 10.0, 0.1, is_float=True, entry_prefix="IQRx")
        for scale in (self.area_iqr_min_scale, self.area_iqr_max_scale):
            scale.set_value_range(0.0, 10.0, slider_upper=AREA_IQR_SLIDER_MAX)
        self.area_iqr_boxplot = tk.Canvas(parent, height=54, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.area_iqr_boxplot.bind("<Configure>", lambda _e: self._redraw_area_iqr_boxplot())

    def _toggle_area_outlier_method_controls(self):
        if self.area_iqr_frame is None or self.area_absolute_frame is None:
            return
        method = self.area_outlier_method_var.get()
        if method not in {"iqr", "absolute"}:
            method = "iqr"
            self.area_outlier_method_var.set(method)
        # Repacked from scratch so the analysis block keeps its place above the
        # sliders whichever way the method is switched.
        for frame in (self.analysis_run_frame, self.area_iqr_frame, self.area_absolute_frame):
            if frame is not None:
                frame.pack_forget()
        if method == "absolute":
            if self.area_iqr_boxplot is not None:
                self.area_iqr_boxplot.pack_forget()
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
        if self.area_outlier_method_var.get() != "iqr":
            if canvas.winfo_ismapped():
                canvas.pack_forget()
            return
        if not self.analysis_iqr_stats or "area" not in self.analysis_iqr_stats:
            if canvas.winfo_ismapped():
                canvas.pack_forget()
            return
        if not canvas.winfo_ismapped():
            canvas.pack(fill="x", padx=6, pady=(2, 6))

        canvas.delete("all")
        q1, median, q3, iqr = self._normalized_area_stat(self.analysis_iqr_stats["area"])
        lo, hi = self._bounds_from_fixed_stats(self.analysis_iqr_stats).get("area", (q1, q3))

        # A symmetric axis keeps the median exactly centered, including skewed data.
        multiplier = getattr(self, "_area_iqr_range_max", 10.0)
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
        canvas.create_rectangle(x_q1, box_top, x_q3, box_bottom, outline="#333333", fill="#dddddd")
        canvas.create_line(x_med, box_top, x_med, box_bottom, fill="#000000", width=2)

        cut_top = box_top
        cut_bottom = box_bottom
        hook_len = 4
        # Lower cut: bracket opens inward to the right. Upper cut: bracket opens inward to the left.
        canvas.create_line(x_lo, cut_top, x_lo, cut_bottom, fill="#cc2222", width=2)
        canvas.create_line(x_lo, cut_top, x_lo + hook_len, cut_top, fill="#cc2222", width=2)
        canvas.create_line(x_lo, cut_bottom, x_lo + hook_len, cut_bottom, fill="#cc2222", width=2)

        canvas.create_line(x_hi, cut_top, x_hi, cut_bottom, fill="#cc2222", width=2)
        canvas.create_line(x_hi, cut_top, x_hi - hook_len, cut_top, fill="#cc2222", width=2)
        canvas.create_line(x_hi, cut_bottom, x_hi - hook_len, cut_bottom, fill="#cc2222", width=2)

    def _schedule_area_iqr_boxplot_redraw(self):
        """Redraw the Area IQR boxplot after Tk has completed layout updates.

        Load config changes section visibility and may also load a video/background.
        A direct redraw can run before the Analysis section and Canvas have a real width,
        so the boxplot appears missing. Scheduling the redraw after idle, plus a short
        delayed redraw, makes the restored boxplot deterministic.
        """
        if self.area_outlier_method_var.get() != "iqr":
            if self.area_iqr_boxplot is not None and self.area_iqr_boxplot.winfo_ismapped():
                self.area_iqr_boxplot.pack_forget()
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
        self.bind_all("<Left>", lambda _e: self._step_frame_stop_play(-1), add="+")
        self.bind_all("<Right>", lambda _e: self._step_frame_stop_play(1), add="+")
        self.bind_all("<space>", lambda _e: self._toggle_play_event(), add="+")
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
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
        self.segmentation_mode_var.trace_add("write", lambda *_: self._on_segmentation_mode_changed())
        self.bg_method_var.trace_add("write", lambda *_: self._on_background_method_changed())
        self.roi_enabled_var.trace_add("write", lambda *_: self._toggle_roi_params())
        self.additional_outlier_enabled_var.trace_add("write", lambda *_: self._toggle_additional_outlier_params())
        self.additional_outlier_segmentation_mode_var.trace_add("write", lambda *_: self._refresh_additional_outlier_controls())
        seg_vars = [
            self.threshold_var, self.dark_threshold_var, self.bright_threshold_var, self.diff_threshold_var, self.min_area_var, self.blur_ksize_var,
            self.open_iter_var, self.close_iter_var, self.fill_holes_var, self.invert_mask_var,
            self.roi_enabled_var, self.roi_reverse_var, self.roi_shape_var, self.roi_x_var, self.roi_y_var, self.roi_w_var, self.roi_h_var,
            self.roi_frame_start_var, self.roi_frame_end_var,
            self.additional_outlier_enabled_var, self.additional_outlier_frame_start_var, self.additional_outlier_frame_end_var,
            self.region_expand_px_var, self.region_expand_merge_only_var,
        ]
        for var in seg_vars:
            var.trace_add("write", lambda *_: self._mark_analysis_stale(redraw=True))
        for var in [self.roi_enabled_var, self.roi_reverse_var, self.roi_shape_var, self.roi_x_var, self.roi_y_var, self.roi_w_var, self.roi_h_var,
                    self.roi_frame_start_var, self.roi_frame_end_var]:
            var.trace_add("write", self._on_roi_var_changed)
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
            self.result_import_frame_offset_var,
            self.result_import_min_coverage_var,
        ]:
            var.trace_add("write", lambda *_: self._on_result_import_changed())
        self.result_import_show_obb_var.trace_add("write", lambda *_: self.redraw_current_frame())
        self.area_outlier_method_var.trace_add("write", lambda *_: self._on_area_outlier_method_changed())
        for var in [self.area_iqr_min_var, self.area_iqr_max_var, self.area_absolute_min_var, self.area_absolute_max_var]:
            var.trace_add("write", lambda *_: self._refresh_bounds_from_fixed_stats(show_error=False))
        for var in [self.show_overlay_var, self.show_centers_var, self.show_contours_var]:
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
        """Load a selected video while preserving the existing config lookup."""
        found = self._find_existing_config_for_video(path)
        if found:
            config_path, config = found
            try:
                self._load_config_dict(config_path, config)
                return
            except Exception as exc:
                self.set_status(f"Failed to load existing config ({config_path}): {exc}")
        self.config_path = None
        self.load_video(path)

    def browse_video(self):
        path = filedialog.askopenfilename(title="Select video", filetypes=VIDEO_EXTS)
        if path:
            self._handle_selected_video(path)

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
        self.stop_playback(update_button=False)
        if self.reader is not None:
            self.reader.close()
        self.reader = reader
        self._area_reference_frame = None
        self._area_reference_key = None
        self._min_area_slider_cap = None
        self._absolute_defaults_pending = not self._applying_config
        self._absolute_zoomed = False
        self.frame_count = reader.frame_count
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
        self.analysis_frames = []
        self.analysis_signature = None
        self.analysis_bounds = {}
        self.analysis_iqr_stats = {}
        self.analysis_is_stale = True
        self.background_ready_var.set("background: not computed")
        self.analysis_ready_var.set("analysis: not computed")
        if self.bg_preview_button is not None:
            self.bg_preview_button.configure(state="disabled")

        # The imported OBBs describe one specific video; drop them with it.
        self.result_obb_import = None
        self.result_import_path_var.set("")
        self.result_import_status_var.set(self._result_import_status_text())
        self._refresh_result_match()

        self._reset_view_state()
        self.roi_sets = [self._default_roi_set()]
        self.roi_active_set_idx = 0
        if self.roi_set_combo is not None:
            self._update_roi_set_selector_values()
            self._update_roi_set_selector()
        self._set_default_roi_to_full_frame(enable=False, emit_status=False)
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
        _start_maximized(self)

    def on_close(self):
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

    def _step_frame_stop_play(self, delta: int):
        self.stop_playback()
        self.step_frame(delta)

    def step_frame(self, delta: int):
        if self.reader is None:
            return
        self.set_frame(self.current_frame + int(delta))

    def fit_canvas_to_window(self):
        if self.frame_bgr_cache is None:
            return
        self._draw_canvas(preserve_view=False)

    def set_frame(self, frame_idx: int, reset_view: bool = False):
        if self.reader is None or self.frame_count <= 0:
            return
        if self.reader.frame_count != self.frame_count:
            self.frame_count = self.reader.frame_count
            self.frame_spin.configure(from_=0, to=max(0, self.frame_count - 1))
            self.frame_scale.configure(from_=0, to=max(0, self.frame_count - 1))
        frame_idx = max(0, min(self.frame_count - 1, int(frame_idx)))
        try:
            frame_bgr, actual_idx = self.reader.read_bgr_with_index(frame_idx)
        except Exception as exc:
            self.frame_var.set(self.current_frame)
            self.frame_scale.set(self.current_frame)
            self.set_status(f"Frame read failed: {exc}")
            return
        if self.reader.frame_count != self.frame_count:
            self.frame_count = self.reader.frame_count
            self.frame_spin.configure(from_=0, to=max(0, self.frame_count - 1))
            self.frame_scale.configure(from_=0, to=max(0, self.frame_count - 1))
        frame_idx = max(0, min(self.frame_count - 1, int(actual_idx)))
        self.current_frame = frame_idx
        self.frame_var.set(frame_idx)
        self.frame_scale.set(frame_idx)
        self.frame_bgr_cache = frame_bgr
        self._draw_canvas(preserve_view=not reset_view)
        self._update_analysis_dependent_buttons()
        self._queue_prefetch(frame_idx)

    def _queue_prefetch(self, center_frame: int):
        if self.reader is None:
            return
        if self.prefetch_after_id is not None:
            self.after_cancel(self.prefetch_after_id)
            self.prefetch_after_id = None

        # Cancel any running prefetch thread, then start a new one.
        self._prefetch_cancel.set()
        cancel = threading.Event()
        self._prefetch_cancel = cancel

        frames = []
        for d in range(1, max(PREFETCH_FORWARD, PREFETCH_BACKWARD) + 1):
            if d <= PREFETCH_FORWARD:
                frames.append(center_frame + d)
            if d <= PREFETCH_BACKWARD:
                frames.append(center_frame - d)
        frames = [f for f in frames if 0 <= f < self.frame_count]

        reader = self.reader

        def worker():
            for fid in frames:
                if cancel.is_set() or reader is None:
                    return
                reader.prefetch(fid)

        threading.Thread(target=worker, daemon=True).start()

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
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(dst_x, dst_y, anchor="nw", image=self.tk_image)

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
        roi_signature = tuple(
            (
                int(bool(r.get("enabled", False))),
                int(bool(r.get("reverse", False))),
                str(r.get("shape", "circle")),
                int(r.get("x", 0)),
                int(r.get("y", 0)),
                int(r.get("w", 0)),
                int(r.get("h", 0)),
                int(r.get("frame_start", 0)),
                int(r.get("frame_end", -1)),
            )
            for r in self.roi_sets
        )
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
            roi_signature,
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
            int(self.result_import_frame_offset_var.get()),
            float(self.result_import_min_coverage_var.get()),
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
        if self.roi_enabled_var.get():
            if not self.roi_params_frame.winfo_ismapped():
                self.roi_params_frame.pack(fill="x", padx=0, pady=(0, 4))
        else:
            self.roi_params_frame.pack_forget()

    def _set_default_roi_to_full_frame(self, enable: bool = False, emit_status: bool = False):
        if self.reader is None:
            return
        cx = int(self.reader.width // 2)
        cy = int(self.reader.height // 2)
        radius = int(max(self.reader.width, self.reader.height) // 2)
        self.roi_shape_var.set("circle")
        self.roi_x_var.set(cx)
        self.roi_y_var.set(cy)
        self.roi_w_var.set(radius)
        self.roi_h_var.set(0)
        self.roi_reverse_var.set(False)
        self.roi_enabled_var.set(bool(enable))
        if emit_status:
            self.set_status(f"Default circular ROI: center=({cx},{cy}), radius={radius}")

    def _on_segmentation_mode_changed(self):
        if self._applying_config:
            return
        self.current_mask_cache.clear()
        self.analysis_blobs_by_frame.clear()
        self.analysis_frames = []
        self.analysis_signature = None
        self.analysis_bounds = {}
        self.analysis_iqr_stats = {}
        self.analysis_is_stale = True
        self.analysis_ready_var.set("analysis: not computed")
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

    def reset_roi(self):
        if self.reader is None:
            return
        cx = int(self.reader.width // 2)
        cy = int(self.reader.height // 2)
        radius = int(max(self.reader.width, self.reader.height) // 2)
        self.roi_shape_var.set("circle")
        self.roi_x_var.set(cx)
        self.roi_y_var.set(cy)
        self.roi_w_var.set(radius)
        self.roi_h_var.set(0)
        self.roi_reverse_var.set(False)
        self.roi_enabled_var.set(True)
        self.set_status(f"ROI reset: circle center=({cx},{cy}), radius={radius}")

    # ROI set management

    @staticmethod
    def _default_roi_set() -> dict:
        return {"enabled": False, "reverse": False, "shape": "circle", "x": 0, "y": 0, "w": 0, "h": 0, "frame_start": 0, "frame_end": -1}

    @staticmethod
    def _normalize_roi_set(roi: dict) -> dict:
        return {
            "enabled": bool(roi.get("enabled", False)),
            "reverse": bool(roi.get("reverse", False)),
            "shape": roi.get("shape", "circle") if roi.get("shape", "circle") in {"rectangle", "circle"} else "circle",
            "x": int(roi.get("x", 0)),
            "y": int(roi.get("y", 0)),
            "w": int(roi.get("w", 0)),
            "h": int(roi.get("h", 0)),
            "frame_start": int(roi.get("frame_start", 0)),
            "frame_end": int(roi.get("frame_end", -1)),
        }

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
        if not self._switching_roi_set:
            self._sync_roi_vars_to_set(self.roi_active_set_idx)

    def _sync_roi_vars_to_set(self, idx: int):
        if idx < 0 or idx >= len(self.roi_sets):
            return
        self.roi_sets[idx] = {
            "enabled": bool(self.roi_enabled_var.get()),
            "reverse": bool(self.roi_reverse_var.get()),
            "shape": self.roi_shape_var.get(),
            "x": int(self.roi_x_var.get()),
            "y": int(self.roi_y_var.get()),
            "w": int(self.roi_w_var.get()),
            "h": int(self.roi_h_var.get()),
            "frame_start": int(self.roi_frame_start_var.get()),
            "frame_end": int(self.roi_frame_end_var.get()),
        }

    def _sync_set_to_roi_vars(self, idx: int):
        if idx < 0 or idx >= len(self.roi_sets):
            return
        roi = self._normalize_roi_set(self.roi_sets[idx])
        self.roi_sets[idx] = roi
        self.roi_enabled_var.set(bool(roi.get("enabled", False)))
        self.roi_reverse_var.set(bool(roi.get("reverse", False)))
        self.roi_shape_var.set(roi.get("shape", "circle"))
        self.roi_x_var.set(int(roi.get("x", 0)))
        self.roi_y_var.set(int(roi.get("y", 0)))
        self.roi_w_var.set(int(roi.get("w", 0)))
        self.roi_h_var.set(int(roi.get("h", 0)))
        self.roi_frame_start_var.set(int(roi.get("frame_start", 0)))
        self.roi_frame_end_var.set(int(roi.get("frame_end", -1)))

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
        self._sync_roi_vars_to_set(self.roi_active_set_idx)
        self._switching_roi_set = True
        try:
            self.roi_active_set_idx = new_idx
            self._sync_set_to_roi_vars(new_idx)
            self._update_roi_set_selector()
        finally:
            self._switching_roi_set = False
        self._toggle_roi_params()
        self._mark_analysis_stale(redraw=True)

    def _add_roi_set(self):
        if len(self.roi_sets) >= 8:
            return
        self._sync_roi_vars_to_set(self.roi_active_set_idx)
        new_roi: dict = self._default_roi_set()
        if self.reader is not None:
            cx = self.reader.width // 2
            cy = self.reader.height // 2
            r = max(self.reader.width, self.reader.height) // 2
            new_roi.update({"x": cx, "y": cy, "w": r})
        self.roi_sets.append(new_roi)
        self._update_roi_set_selector_values()
        self._switch_roi_set(len(self.roi_sets) - 1)

    def _remove_roi_set(self):
        if len(self.roi_sets) <= 1:
            return
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
        self._mark_analysis_stale(redraw=True)

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
            frame_offset=int(self.result_import_frame_offset_var.get()),
            min_coverage=float(self.result_import_min_coverage_var.get()),
            obbs_by_frame=self.result_obb_import.obbs_by_frame,
        )

    def _on_result_import_changed(self):
        if self._applying_config:
            return
        self._refresh_result_match()
        self._refresh_bounds_from_fixed_stats(show_error=False)

    def _current_result_obbs(self) -> "Optional[np.ndarray]":
        if not self.result_import_show_obb_var.get() or self._result_match is None:
            return None
        return self._result_match.obbs_by_frame.get(
            int(self.current_frame) + int(self._result_match.frame_offset))

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

    def _active_roi_geometry(self) -> Optional[tuple[str, float, float, float, float]]:
        """The ROI as drawn on the current frame, or None when none is drawn."""
        if self.frame_bgr_cache is None or not self.roi_enabled_var.get():
            return None
        start = int(self.roi_frame_start_var.get())
        end = int(self.roi_frame_end_var.get())
        if self.current_frame < start or (end >= 0 and self.current_frame > end):
            return None
        return (
            self.roi_shape_var.get(),
            float(self.roi_x_var.get()), float(self.roi_y_var.get()),
            float(self.roi_w_var.get()), float(self.roi_h_var.get()),
        )

    def _roi_outline_hit(self, ix: float, iy: float) -> bool:
        geom = self._active_roi_geometry()
        if geom is None:
            return False
        shape, x, y, w, h = geom
        tol = max(1.0, ROI_EDGE_GRAB_CANVAS_PX / max(1e-9, self.scale))
        if shape == "circle":
            return w > 0 and abs(math.hypot(ix - x, iy - y) - w) <= tol
        if w <= 0 or h <= 0:
            return False
        on_band = (x - tol) <= ix <= (x + w + tol) and (y - tol) <= iy <= (y + h + tol)
        in_core = (x + tol) < ix < (x + w - tol) and (y + tol) < iy < (y + h - tol)
        return on_band and not in_core

    def _start_roi_drag(self, ix: float, iy: float, *, scaling: bool) -> bool:
        if not self._roi_outline_hit(ix, iy):
            return False
        self._roi_drag_mode = None
        self._set_roi_drag_mode(scaling, ix, iy)
        return True

    def _set_roi_drag_mode(self, scaling: bool, ix: float, iy: float) -> None:
        """Switch between moving and resizing, including mid-drag on a Ctrl press.

        Resizing pins the centre where the ROI stands at that moment, so taking
        over a move never shifts x/y and the centre cannot drift as the size is
        rounded to whole pixels.
        """
        mode = "scale" if scaling else "move"
        if mode == self._roi_drag_mode:
            return
        self._roi_drag_mode = mode
        self._roi_drag_anchor = (ix, iy)
        self._roi_scale_center = None
        geom = self._active_roi_geometry()
        if mode == "scale" and geom is not None:
            shape, x, y, w, h = geom
            self._roi_scale_center = (x, y) if shape == "circle" else (x + w / 2.0, y + h / 2.0)

    def _update_roi_drag(self, ix: float, iy: float) -> None:
        geom = self._active_roi_geometry()
        if geom is None or self._roi_drag_anchor is None:
            return
        shape, x, y, w, h = geom
        if self._roi_drag_mode == "move":
            ax, ay = self._roi_drag_anchor
            dx, dy = int(round(ix - ax)), int(round(iy - ay))
            if dx == 0 and dy == 0:
                return
            self.roi_x_var.set(max(0, int(x) + dx))
            self.roi_y_var.set(max(0, int(y) + dy))
            self._roi_drag_anchor = (ax + dx, ay + dy)
            return
        if self._roi_scale_center is None:
            return
        # The size follows the cursor's offset from the pinned centre: inside the
        # ROI shrinks it, outside grows it, and a still cursor holds one value.
        cx, cy = self._roi_scale_center
        if shape == "circle":
            self.roi_w_var.set(max(1, int(round(math.hypot(ix - cx, iy - cy)))))
            return
        new_w = max(1, int(round(abs(ix - cx) * 2.0)))
        new_h = max(1, int(round(abs(iy - cy) * 2.0)))
        self.roi_w_var.set(new_w)
        self.roi_h_var.set(new_h)
        self.roi_x_var.set(max(0, int(round(cx - new_w / 2.0))))
        self.roi_y_var.set(max(0, int(round(cy - new_h / 2.0))))

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
        # Frame preview prefetch uses another decoder.  Stop feeding it new
        # work before opening the background decoders, especially for 4K H.264.
        self._prefetch_cancel.set()
        if self.reader is not None:
            self.reader.clear_cache()
        self._background_building = True
        try:
            return self._compute_background_impl()
        finally:
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

        ok = self._with_progress_popup("Background extraction", f"Preparing {method_label} background...", work)
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
            manual_outlier=blob.manual_outlier,
            is_crossing=blob.is_crossing,
            result_obb_count=blob.result_obb_count,
            result_obb_coverage=blob.result_obb_coverage,
            result_single=blob.result_single,
        )

    def _compute_iqr_stats(self, all_blobs: list[BlobMetrics]) -> dict[str, tuple[float, float, float, float]]:
        # Only Area is used for IQR outlier judgment; BBox/Width/Height IQR judgment is not used.
        values = [b.area for b in all_blobs if not b.manual_outlier]
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
                    int(fid): self._classify_blobs(blobs, self.analysis_bounds)
                    for fid, blobs in self.analysis_blobs_by_frame.items()
                }
            self.analysis_is_stale = False
            if self.analysis_signature is not None:
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
        result_match: Optional["_ResultMatchConfig"] = None,
    ) -> list[BlobMetrics]:
        if bounds is None:
            bounds = self.analysis_bounds
        if result_match is None:
            result_match = self._result_match
        out = [self._clone_blob(b) for b in blobs] if clone else blobs
        area_bounds = bounds.get("area", (-float("inf"), float("inf"))) if bounds else (-float("inf"), float("inf"))

        for b in out:
            b.area_outlier = not inside_bounds(b.area, area_bounds)
            b.bbox_area_outlier = False
            b.width_outlier = False
            b.height_outlier = False
            b.result_obb_count = 0
            b.result_obb_coverage = 0.0
            b.result_single = False
            b.is_crossing = bool(b.manual_outlier or b.area_outlier)
        if result_match is not None and out:
            self._rescue_single_animal_outliers(out, result_match)
        return out

    @staticmethod
    def _rescue_single_animal_outliers(blobs: list[BlobMetrics], result_match: "_ResultMatchConfig") -> None:
        """Clear the area-outlier flag of blobs the imported result explains as one animal.

        Area alone cannot tell a genuinely large individual from two individuals
        in contact.  A blob that owns exactly one imported OBB, and whose area
        that OBB covers, holds one animal however large it is.  Everything else
        -- blobs inside the area bounds, Additional Outlier regions, blobs with two or
        more OBBs, and blobs the result does not explain -- keeps the
        classification the area rule already gave it.
        """
        by_frame: dict[int, list[BlobMetrics]] = {}
        for b in blobs:
            by_frame.setdefault(int(b.frame), []).append(b)
        for frame_idx, frame_blobs in by_frame.items():
            _match_result_obbs_to_blobs(frame_blobs, frame_idx, result_match)
            for b in frame_blobs:
                if b.manual_outlier or not b.area_outlier:
                    continue
                if b.result_obb_count == 1 and b.result_obb_coverage >= result_match.min_coverage:
                    b.result_single = True
                    b.is_crossing = False

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
        if self.reader is None:
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
            all_blobs = [
                blob
                for blobs in blobs_by_frame.values()
                for blob in blobs
            ]
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
                    blobs, bounds, clone=False, result_match=result_match)
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

        ok = self._with_progress_popup("Analysis", "Preparing analysis...", work)
        if not ok:
            return

        classified_by_frame = result["classified_by_frame"]
        iqr_stats_fixed = result["iqr_stats"]
        bounds = result["bounds"]
        total = int(result["total"])
        outliers = int(result["outliers"])

        self.analysis_blobs_by_frame = classified_by_frame
        self.analysis_frames = frame_indices
        self.analysis_signature = (self._segmentation_signature(), tuple(frame_indices), self._classifier_signature())
        self.analysis_iqr_stats = iqr_stats_fixed
        self.analysis_bounds = bounds
        self.analysis_is_stale = False
        self.analysis_ready_var.set(f"analysis: ready ({scope_text}, blobs={total}, outliers={outliers})")
        self._set_workflow_stage("processing")
        self._redraw_area_iqr_boxplot()
        self.redraw_current_frame()
        self.set_status(f"Analysis complete: blobs={total}, outliers={outliers}")

    def analyze_sample_frames(self):
        self._run_analysis()

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
            return mask, self._classify_blobs(blobs, self.analysis_bounds)
        # Before Analyze sampled frames, no fixed sampled-frame result exists.
        # Show only segmentation results without crossing classification.
        return mask, [self._clone_blob(b) for b in blobs]

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
    ) -> np.ndarray:
        """Render a single frame with ORANGE/CYAN overlay -- same visual as the GUI preview."""
        base = frame_bgr.copy()
        _ih, _iw = base.shape[:2]

        for _roi in roi_sets:
            if not _roi.get("enabled", False):
                continue
            _fs = int(_roi.get("frame_start", 0))
            _fe = int(_roi.get("frame_end", -1))
            if frame_idx < _fs or (_fe >= 0 and frame_idx > _fe):
                continue
            _shape = _roi.get("shape", "circle")
            _rev = bool(_roi.get("reverse", False))
            if _shape == "circle":
                _cx = max(0, min(_iw - 1, int(_roi.get("x", 0))))
                _cy = max(0, min(_ih - 1, int(_roi.get("y", 0))))
                _r = max(0, int(_roi.get("w", 0)))
                if _r > 0:
                    cv2.circle(base, (_cx, _cy), _r, ROI_COLOR, 2, cv2.LINE_AA)
                    if _rev:
                        cv2.putText(base, "REV", (min(_iw - 1, _cx + _r + 4), _cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_COLOR, 1, cv2.LINE_AA)
            else:
                _rx = max(0, min(_iw, int(_roi.get("x", 0))))
                _ry = max(0, min(_ih, int(_roi.get("y", 0))))
                _rw = max(0, int(_roi.get("w", 0)))
                _rh = max(0, int(_roi.get("h", 0)))
                _rx2 = min(_iw, _rx + _rw)
                _ry2 = min(_ih, _ry + _rh)
                if _rx2 > _rx and _ry2 > _ry:
                    cv2.rectangle(base, (_rx, _ry), (max(_rx, _rx2 - 1), max(_ry, _ry2 - 1)), ROI_COLOR, 2, cv2.LINE_AA)
                    if _rev:
                        cv2.putText(base, "REV", (_rx, max(14, _ry - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_COLOR, 1, cv2.LINE_AA)

        if show_overlay:
            overlay = np.zeros_like(base)
            for b in blobs:
                color = OUTLIER_COLOR if b.is_crossing else OBB_COLOR
                cv2.drawContours(overlay, [b.contour], -1, color, thickness=cv2.FILLED)
            base = cv2.addWeighted(base, 1.0, overlay, 0.35, 0)

        for b in blobs:
            color = OUTLIER_COLOR if b.is_crossing else OBB_COLOR
            if show_contours:
                cv2.drawContours(base, [b.contour], -1, color, 2, cv2.LINE_AA)
            cx_b = int(round(b.center_x))
            cy_b = int(round(b.center_y))
            if show_centers:
                cv2.circle(base, (cx_b, cy_b), 2, WHITE, -1, cv2.LINE_AA)

        _draw_result_obbs(base, result_obbs)

        return base

    def _build_preview_rgb(self) -> np.ndarray:
        if self.frame_bgr_cache is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame_bgr = self.frame_bgr_cache
        _, blobs = self._classified_current_frame()
        self._current_blobs = blobs
        kept = sum(1 for b in blobs if not b.is_crossing)
        outliers = sum(1 for b in blobs if b.is_crossing)
        rescued = sum(1 for b in blobs if b.result_single)

        stale_text = " / stale" if self.analysis_is_stale and self.analysis_bounds else ""
        result_text = f" / result-single {rescued}" if self._result_match is not None else ""
        self.preview_title_var.set(
            f"frame {self.current_frame} / blobs {len(blobs)} / kept {kept} / "
            f"outliers {outliers}{result_text}{stale_text}"
        )

        base = frame_bgr.copy()

        _ih, _iw = frame_bgr.shape[:2]
        _cur = self.current_frame
        for _roi in self.roi_sets:
            if not _roi.get("enabled", False):
                continue
            _fs = int(_roi.get("frame_start", 0))
            _fe = int(_roi.get("frame_end", -1))
            if _cur < _fs or (_fe >= 0 and _cur > _fe):
                continue
            _shape = _roi.get("shape", "circle")
            _rev = bool(_roi.get("reverse", False))
            if _shape == "circle":
                _cx = max(0, min(_iw - 1, int(_roi.get("x", 0))))
                _cy = max(0, min(_ih - 1, int(_roi.get("y", 0))))
                _r = max(0, int(_roi.get("w", 0)))
                if _r > 0:
                    cv2.circle(base, (_cx, _cy), _r, ROI_COLOR, 2, cv2.LINE_AA)
                    if _rev:
                        cv2.putText(base, "REV", (min(_iw - 1, _cx + _r + 4), _cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_COLOR, 1, cv2.LINE_AA)
            else:
                _rx = max(0, min(_iw, int(_roi.get("x", 0))))
                _ry = max(0, min(_ih, int(_roi.get("y", 0))))
                _rw = max(0, int(_roi.get("w", 0)))
                _rh = max(0, int(_roi.get("h", 0)))
                _rx2 = min(_iw, _rx + _rw)
                _ry2 = min(_ih, _ry + _rh)
                if _rx2 > _rx and _ry2 > _ry:
                    cv2.rectangle(base, (_rx, _ry), (max(_rx, _rx2 - 1), max(_ry, _ry2 - 1)), ROI_COLOR, 2, cv2.LINE_AA)
                    if _rev:
                        cv2.putText(base, "REV", (_rx, max(14, _ry - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_COLOR, 1, cv2.LINE_AA)

        if self.show_overlay_var.get():
            overlay = np.zeros_like(base)
            for b in blobs:
                color = OUTLIER_COLOR if b.is_crossing else OBB_COLOR
                cv2.drawContours(overlay, [b.contour], -1, color, thickness=cv2.FILLED)
            base = cv2.addWeighted(base, 1.0, overlay, 0.35, 0)

        for b in blobs:
            color = OUTLIER_COLOR if b.is_crossing else OBB_COLOR
            if self.show_contours_var.get():
                cv2.drawContours(base, [b.contour], -1, color, 2, cv2.LINE_AA)
            cx = int(round(b.center_x))
            cy = int(round(b.center_y))
            if self.show_centers_var.get():
                cv2.circle(base, (cx, cy), 2, WHITE, -1, cv2.LINE_AA)

        _draw_result_obbs(base, self._current_result_obbs())

        # IQR-related numeric values are intentionally not drawn on the image.
        # The box plot below the sliders is the only reference display.

        return cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

    def _draw_canvas(self, preserve_view: bool):
        self._update_area_control_ranges()
        preview_rgb = self._build_preview_rgb()
        self._present_image(preview_rgb, preserve_view=preserve_view)

    def redraw_current_frame(self):
        if self.frame_bgr_cache is not None:
            self._draw_canvas(preserve_view=True)

    def on_mousewheel(self, event):
        if self.frame_bgr_cache is None:
            return
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
        if self._start_roi_drag(ix, iy, scaling=bool(int(getattr(event, "state", 0)) & TK_CONTROL_MASK)):
            self.stop_playback()
            return
        self.last_drag_canvas = np.array([event.x, event.y], dtype=float)

    def on_canvas_drag(self, event):
        if self._roi_drag_mode is not None:
            ix, iy = self._canvas_to_image(float(event.x), float(event.y))
            self._set_roi_drag_mode(bool(int(getattr(event, "state", 0)) & TK_CONTROL_MASK), ix, iy)
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
            self._roi_scale_center = None
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
        lines = [f"Area: {int(round(blob.area))} px"]
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
            if blob.result_obb_count == 1 and blob.area_outlier and not blob.manual_outlier:
                lines.append(f"OBB coverage: {blob.result_obb_coverage * 100:.0f}%")
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
                bg="#ffffcc",
                fg="#000000",
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
        if self.last_drag_canvas is not None:
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

    def start_playback(self):
        if self.reader is None or self.frame_count <= 0:
            return
        if self.current_frame >= self.frame_count - 1:
            self.set_frame(0)
        self.playback_active = True
        self._update_play_button()
        self._schedule_playback()

    def stop_playback(self, update_button: bool = True):
        self.playback_active = False
        if self.playback_job is not None:
            self.after_cancel(self.playback_job)
            self.playback_job = None
        if update_button:
            self._update_play_button()

    def _update_play_button(self):
        self.play_button.configure(text="Stop" if self.playback_active else "Play")

    def _playback_delay_ms(self) -> int:
        if self.reader is None or self.reader.fps <= 0:
            return 33
        return max(1, int(round(1000.0 / self.reader.fps)))

    def _schedule_playback(self):
        if not self.playback_active:
            return
        if self.playback_job is not None:
            self.after_cancel(self.playback_job)
        delay = self._playback_delay_ms()
        self.playback_job = self.after(delay, self._playback_tick)

    def _playback_tick(self):
        self.playback_job = None
        if not self.playback_active or self.reader is None:
            return
        if self.current_frame >= self.frame_count - 1:
            self.stop_playback()
            return
        self.set_frame(self.current_frame + 1)
        if self.playback_active:
            self._schedule_playback()


    def _open_progress_popup(self, title: str):
        self._close_progress_popup()
        popup = tk.Toplevel(self)
        popup.title(title)
        # Do not make this popup transient/modal.
        # A normal Toplevel can be minimized and can appear on the OS taskbar.
        popup.resizable(False, False)
        popup.configure(bg="#f3f4f6")
        try:
            popup.attributes("-topmost", False)
        except Exception:
            pass
        popup.protocol("WM_DELETE_WINDOW", self._request_progress_cancel)

        frame = tk.Frame(popup, padx=18, pady=16, bg="#f3f4f6")
        frame.pack(fill="both", expand=True)
        tk.Label(frame, textvariable=self.progress_label_var, anchor="w", justify="left", bg="#f3f4f6").pack(fill="x")
        progress_bar = ttk.Progressbar(
            frame,
            variable=self.progress_value_var,
            maximum=100,
            mode="determinate",
            length=360,
        )
        progress_bar.pack(fill="x", pady=(12, 0))

        self.progress_popup = popup
        self.progress_bar = progress_bar
        self._progress_indeterminate = False
        self.progress_label_var.set("Preparing...")
        self.progress_value_var.set(0.0)
        self._place_progress_popup()
        popup.deiconify()
        popup.lift()
        popup.focus_force()
        self._flush_ui()

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

    def _flush_ui(self):
        try:
            self.update_idletasks()
            self.update()
            self._maximize_window()
        except tk.TclError:
            pass

    def _set_progress_popup(self, text: str, value: Optional[float]):
        # Thread-safe: worker pushes to queue; main thread drains it in the polling loop.
        normalized = None if value is None else float(max(0.0, min(100.0, value)))
        self._progress_queue.put_nowait((text, normalized))

    def _drain_progress_queue(self):
        try:
            while True:
                text, val = self._progress_queue.get_nowait()
                if self.progress_popup is not None:
                    self.progress_label_var.set(text)
                    if val is None:
                        if self.progress_bar is not None and not self._progress_indeterminate:
                            self.progress_value_var.set(0.0)
                            self.progress_bar.configure(mode="indeterminate")
                            self.progress_bar.start(12)
                            self._progress_indeterminate = True
                    else:
                        if self.progress_bar is not None and self._progress_indeterminate:
                            self.progress_bar.stop()
                            self.progress_bar.configure(mode="determinate")
                            self._progress_indeterminate = False
                        self.progress_value_var.set(val)
        except queue.Empty:
            pass

    def _close_progress_popup(self):
        if self.progress_bar is not None:
            try:
                self.progress_bar.stop()
            except Exception:
                pass
        if self.progress_popup is not None:
            try:
                self.progress_popup.grab_release()
            except Exception:
                pass
            try:
                self.progress_popup.destroy()
            except Exception:
                pass
        self.progress_popup = None
        self.progress_bar = None
        self._progress_indeterminate = False

    def _request_progress_cancel(self):
        self._progress_cancel = True
        self._close_progress_popup()

    def _with_progress_popup(self, title: str, start_text: str, func):
        self._progress_cancel = False
        # Flush stale messages from a previous run.
        while not self._progress_queue.empty():
            try:
                self._progress_queue.get_nowait()
            except queue.Empty:
                break

        self._open_progress_popup(title)
        if self.progress_popup is not None:
            self.progress_label_var.set(start_text)
            self.progress_value_var.set(0.0)
            self._place_progress_popup()

        result_box: list = [None]
        done_event = threading.Event()

        def worker():
            try:
                result_box[0] = func()
            except Exception:
                result_box[0] = False
            finally:
                done_event.set()

        threading.Thread(target=worker, daemon=True).start()

        # Polling loop: keeps the main thread responsive while the worker runs.
        while not done_event.is_set():
            self._drain_progress_queue()
            if self.progress_popup is not None:
                try:
                    self._place_progress_popup()
                except Exception:
                    pass
            try:
                self.update()
                self._maximize_window()
            except tk.TclError:
                pass
            done_event.wait(timeout=0.02)

        # Final drain after worker completes.
        self._drain_progress_queue()
        self._close_progress_popup()
        return result_box[0]

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
        # the video's own directory only when no config has been loaded/saved
        # yet. This keeps every file this class reads or writes (config.json,
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
        self.analysis_frames = []
        self.analysis_signature = None
        self.analysis_bounds = {}
        self.analysis_iqr_stats = {}
        self.analysis_is_stale = True
        self.analysis_ready_var.set("analysis: not computed")

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
        # so anchoring on the video path would scatter session output away
        # from where the config (and prior segmentation output) actually is.
        if self.config_path:
            config_dir = os.path.dirname(os.path.abspath(self.config_path))
            project_dir = os.path.dirname(config_dir)
            if project_dir:
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
            roi_sets=[dict(r) for r in self.roi_sets],
            additional_outlier_sets=[dict(r) for r in self.additional_outlier_sets],
            background_bgr=self.background_bgr,  # read-only; no copy needed
            additional_backgrounds=self._additional_outlier_backgrounds_for_config(),
            protected_area_bounds=protected_area_bounds,
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

    def _build_export_source_all_frames(
        self,
        progress_start: float = 10.0,
        progress_end: float = 35.0,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
        cfg: Optional[_SegConfig] = None,
        bounds: Optional[dict[str, tuple[float, float]]] = None,
        progress_label: str = "Processing frames",
        result_match: Optional["_ResultMatchConfig"] = None,
    ) -> dict[int, list[BlobMetrics]]:
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
        if result_match is None:
            result_match = self._result_match
        source: dict[int, list[BlobMetrics]] = {}
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
        frame_bytes = max(1, int(self.reader.width) * int(self.reader.height) * 3)
        memory_batch_limit = max(1, PROCESSING_BATCH_MAX_BYTES // frame_bytes)
        batch_size = min(max(1, _EXPORT_WORKERS * 2), memory_batch_limit)
        # Decoding the next batch ahead keeps two batches in memory. Shrink the
        # batch so both fit the same ceiling, but never below one full round for
        # the pool: with frames that large, keep the previous batch size, worker
        # occupancy and peak memory instead, and decode on demand.
        prefetch = memory_batch_limit // 2 >= _EXPORT_WORKERS
        if prefetch:
            batch_size = min(batch_size, memory_batch_limit // 2)

        accel = "GPU" if _CUDA_AVAILABLE else f"CPUx{_EXPORT_WORKERS}"
        self._set_progress_popup(
            f"{progress_label}: 0/{total} (0%)\nstarting {accel}...",
            progress_start,
        )

        # Reduce OpenCV's internal thread count so each Python thread runs a truly
        # single-threaded cv2 call -- the thread pool then provides parallelism.
        prev_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        decode_cap = cv2.VideoCapture(self.reader.video_path)
        decode_pos = -1

        # Use a dedicated sequential decoder.  The preview reader copies and
        # caches frames, which is useful interactively but unnecessarily
        # expensive while processing the entire range.  Only the single decoder
        # thread running this function touches decode_cap and decode_pos.
        def decode_batch(fids: "list[int]") -> "list[tuple[int, np.ndarray]]":
            nonlocal decode_pos
            frames_data: list[tuple[int, np.ndarray]] = []
            for fid in fids:
                frame = None
                if decode_cap.isOpened():
                    if decode_pos != fid:
                        decode_cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                    ok, decoded = decode_cap.read()
                    if ok and decoded is not None and decoded.size > 0:
                        frame = decoded
                        decode_pos = fid + 1
                    else:
                        decode_pos = -1
                if frame is None:
                    frame = self.reader.read_bgr(fid)
                frames_data.append((fid, frame))
            return frames_data

        batches = [list(range(b, min(b + batch_size, end_frame + 1)))
                   for b in range(start_frame, end_frame + 1, batch_size)]
        try:
            with ThreadPoolExecutor(max_workers=_EXPORT_WORKERS) as executor, \
                    ThreadPoolExecutor(max_workers=1) as decoder:
                pending = None
                for index, batch_fids in enumerate(batches):
                    if self._progress_cancel:
                        return {}
                    if pending is None:
                        pending = decoder.submit(decode_batch, batch_fids)
                    frames_data = pending.result()
                    # Decode the next batch while the pool segments this one.
                    pending = (decoder.submit(decode_batch, batches[index + 1])
                               if prefetch and index + 1 < len(batches) else None)
                    futures = {
                        executor.submit(_segment_blobs_with_config, frame, fid, cfg): fid
                        for fid, frame in frames_data
                    }
                    for future in as_completed(futures):
                        fid = futures[future]
                        blobs = future.result()
                        source[fid] = self._classify_blobs(
                            blobs, bounds, clone=False, result_match=result_match)

                    n = batch_fids[-1] + 1 - start_frame
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
        finally:
            decode_cap.release()
            cv2.setNumThreads(prev_threads)
        return source

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
        metadata: Optional[dict[str, int]] = None,
    ):
        if self.reader is None:
            raise RuntimeError("No video has been loaded.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if source is None:
            if self._segmentation_needs_background() and self.background_bgr is None:
                raise RuntimeError("In this mode, please compute the background first.")
            source = self._build_export_source_all_frames()
        blobs_in_video: list[list[SimpleNamespace]] = []
        for fid in range(self.frame_count):
            frame_blobs = []
            for b in source.get(fid, []):
                frame_blobs.append(SimpleNamespace(contour=b.contour.astype(np.int32), is_outlier=bool(b.is_crossing)))
            blobs_in_video.append(frame_blobs)
        if metadata is None:
            metadata = {
                "background_frame_start": int(self.background_frame_start_var.get()),
                "background_frame_end": int(self.background_frame_end_var.get()),
                "training_frame_start": int(self.training_frame_start_var.get()),
                "training_frame_end": int(self.training_frame_end_var.get()),
                "training_frame_interval": max(1, int(self.training_frame_interval_var.get())),
            }
        with open(path, "wb") as f:
            pickle.dump(SimpleNamespace(
                blobs_in_video=blobs_in_video,
                source_frame_count=int(self.frame_count),
                background_frame_start=int(metadata["background_frame_start"]),
                background_frame_end=int(metadata["background_frame_end"]),
                training_frame_start=int(metadata["training_frame_start"]),
                training_frame_end=int(metadata["training_frame_end"]),
                training_frame_interval=max(1, int(metadata["training_frame_interval"])),
            ), f)

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
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def run_processing(self):
        if self.progress_popup is not None:
            messagebox.showwarning("Processing", "Another run is already in progress.")
            return
        if self.reader is None:
            messagebox.showerror("Error", "Please load a video first.")
            return
        if self._segmentation_needs_background() and self.background_bgr is None:
            messagebox.showerror("Error", "In this mode, please compute the background first.")
            return
        if self._needs_area_analysis() and not self.analysis_iqr_stats:
            messagebox.showwarning(
                "Outlier Extraction",
                "Processing needs the outlier bounds from a completed analysis.\n\n"
                "Run Analyze in the Outlier Extraction section first.",
            )
            return

        self._prefetch_cancel.set()
        self.reader.clear_cache()
        try:
            # Capture GUI state once on the main thread.  The processing worker
            # can then run without cross-thread Tk variable access.
            cfg = self._capture_seg_config()
            result_match = self._result_match
            bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats)
            frame_start = max(0, int(self.training_frame_start_var.get()))
            frame_end = self._resolved_frame_end(int(self.training_frame_end_var.get()))
            if frame_start > frame_end:
                raise RuntimeError("Invalid processing frame range.")
            total_frames = frame_end - frame_start + 1
            frames_complete = f"Frames processed: {total_frames}/{total_frames} (100%)"
            config_data = self._collect_config()
            config_path = self.config_path or self._default_config_path()
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
        ok = self._with_progress_popup("Processing", "Preparing...", work)
        _elapsed = time.perf_counter() - _t0
        if not ok or self._progress_cancel:
            if isinstance(completed[0], Exception):
                messagebox.showerror("Error", str(completed[0]))
                self.set_status("", auto_clear=False)
            return
        config_path, background_path, pickle_path, project_dir = completed[0]
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
            self.destroy()

        self.after(1200, _show_done)

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

        popup = tk.Toplevel(self)
        popup.title("Export Labeled Video")
        popup.transient(self)
        popup.resizable(False, False)
        popup.configure(bg="#f3f4f6")

        result: dict[str, object] = {"applied": False}
        output_path_var = tk.StringVar(value=self._default_labeled_video_path())

        outer = tk.Frame(popup, padx=18, pady=16, bg="#f3f4f6")
        outer.pack(fill="both", expand=True)

        tk.Label(
            outer,
            text=f"Total frames: {total:,}\nValid frame indices: 0-{last:,}",
            justify="left",
            anchor="w",
            bg="#f3f4f6",
        ).pack(fill="x", pady=(0, 12))

        path_frame = tk.LabelFrame(outer, text="Output", padx=10, pady=8, bg="#f3f4f6")
        path_frame.pack(fill="x", pady=(0, 10))
        path_row = tk.Frame(path_frame, bg="#f3f4f6")
        path_row.pack(fill="x")
        path_entry = tk.Entry(path_row, textvariable=output_path_var, width=54)
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

        tk.Button(path_row, text="Browse...", width=10, command=_browse_output_path).pack(side="left", padx=(8, 0))

        range_frame = tk.LabelFrame(outer, text="Frame range", padx=10, pady=8, bg="#f3f4f6")
        range_frame.pack(fill="x", pady=(0, 10))
        entries: dict[str, tk.Spinbox] = {}

        def _range_row(label: str, key: str, initial: int) -> None:
            row = tk.Frame(range_frame, bg="#f3f4f6")
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=12, anchor="w", bg="#f3f4f6").pack(side="left")
            sp = tk.Spinbox(row, from_=0, to=last, increment=1, width=12, **_SPIN_CFG)
            sp.delete(0, tk.END)
            sp.insert(0, str(int(initial)))
            sp.pack(side="left")
            entries[key] = sp

        _range_row("Start frame", "start", default_start)
        _range_row("End frame", "end", default_end)

        error_var = tk.StringVar(value="")
        tk.Label(outer, textvariable=error_var, anchor="w", justify="left", fg="#b00020", bg="#f3f4f6").pack(fill="x")

        buttons = tk.Frame(outer, bg="#f3f4f6")
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

        tk.Button(buttons, text="Export", width=10, command=_apply).pack(side="right", padx=(8, 0))
        tk.Button(buttons, text="Cancel", width=10, command=_cancel).pack(side="right")
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
        if self.reader is None:
            messagebox.showerror("Error", "No video has been loaded.")
            return
        if self._segmentation_needs_background() and self.background_bgr is None:
            messagebox.showerror("Error", "In this mode, please compute the background first.")
            return
        if self._needs_area_analysis() and not self.analysis_iqr_stats:
            messagebox.showerror("Error", "Please run Analyze sampled frames first.")
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
        roi_sets_snapshot = [dict(r) for r in self.roi_sets]
        seg_cfg = self._capture_seg_config()
        result_match = self._result_match
        show_result_obb = bool(self.result_import_show_obb_var.get())
        export_bounds = self._bounds_from_fixed_stats(self.analysis_iqr_stats)
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
                            result_match.obbs_by_frame.get(fid + result_match.frame_offset)
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
        ok = self._with_progress_popup("Export Labeled Video", "Preparing...", work)
        if not ok or self._progress_cancel:
            if isinstance(completed[0], Exception):
                messagebox.showerror("Error", str(completed[0]))
            self.set_status("", auto_clear=False)
            return
        saved_path, saved_start, saved_end = completed[0]
        self.set_status(f"Labeled video saved: {saved_path}")
        messagebox.showinfo("Completed", f"Labeled video saved:\n{saved_path}\nframes: {saved_start}-{saved_end}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="")
    parser.add_argument("--session", default="")
    parser.add_argument("--no-launch-tracking", action="store_true")
    parser.add_argument("--return-to-easy", action="store_true")
    args = parser.parse_args()

    configure_taskbar_identity()
    app = CrossingReviewApp(no_launch_tracking=args.no_launch_tracking or args.return_to_easy)
    if args.video and os.path.isfile(args.video):
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
