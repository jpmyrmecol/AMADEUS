# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import json
import os
import pickle
import queue
import random
import re
import sys
import threading
import time
import tkinter as tk
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import customtkinter as ctk

try:
    from .project_paths import (
        PROJECT_ROOT,
        MAIN_DIR,
        ensure_import_paths,
        gui_asset,
    )
    from .canvas_file_drop import install_canvas_file_drop
    from .window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
except ImportError:  # Preserve direct execution with: python gui/gui_refinement.py
    from project_paths import (
        PROJECT_ROOT,
        MAIN_DIR,
        ensure_import_paths,
        gui_asset,
    )
    from canvas_file_drop import install_canvas_file_drop
    from window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon

CTK_THEME = str(gui_asset("deep_green.json"))

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(CTK_THEME)

try:
    from . import convert as _convert
except ImportError:  # Preserve direct execution with: python gui/gui_refinement.py
    import convert as _convert

ensure_import_paths(PROJECT_ROOT, MAIN_DIR)
from gui.color import WHITE_RGB, make_id_palette

from assign_types import (
    # id_tracking S0/S1 labels used for refinement-candidate exclusion.
    ATYPE_INIT, ATYPE_S1, ATYPE_VEL_DIST, ATYPE_GAP_DET,
    ATYPE_DIR_FIX, ATYPE_DIR_FLIP, ATYPE_VITERBI,
    # refinement.py C3: final ID resolution
    ATYPE_EMBED_ABSTAIN, ATYPE_EMBED_IDENTITY, ATYPE_EMBED_RELABEL,
    # refinement.py C1-C2: geometry completion
    ATYPE_KF_FILL, ATYPE_OVERLAP_FILL,
    # refinement.py C4: position/direction spike repair
    ATYPE_POS_FIX,
    # refinement.py C5: missing frames
    ATYPE_MISSING,
    # GUI post-processing filters
    ATYPE_PRE_FILL, ATYPE_POST_FILL, ATYPE_ANCHOR, ATYPE_SPIKE,
    ATYPE_FILTER_FILL, ATYPE_ID_SWAP,
    ASSIGN_CODE_LABELS, ASSIGN_CODE_ORDER, ASSIGN_CODE_SHORT_LABELS,
)
from checkpoint_utils import parse_checkpoint_spec, weight_to_checkpoint_number
try:
    from .config_path_recovery import prepare_config_for_gui
except ImportError:  # Preserve direct execution with: python gui/gui_refinement.py
    from config_path_recovery import prepare_config_for_gui
from tracking_artifacts import artifact_path as tracking_artifact_path
from training_paths import OUTPUT_ROOT_DIR
from experiment_utils import (
    DEFAULT_LR0,
    DEFAULT_LRF,
    experiment_dir_name,
    format_lr0_for_name,
    format_lrf_for_name,
    parse_lr0_values,
    parse_lrf_values,
    resolve_existing_experiment_dir_name,
)

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageTk

APP_TITLE = "AMADEUS: Refinement"
WINDOW_W = 1600
WINDOW_H = 980
CANVAS_BG = "black"
ZOOM_IN_FACTOR = 1.12
ZOOM_OUT_FACTOR = 1.0 / ZOOM_IN_FACTOR
MIN_ZOOM = 0.2
MAX_ZOOM = 20.0
VIDEO_EXTS = [("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")]
CSV_EXTS = [("CSV files", "*.csv"), ("All files", "*.*")]
VIDEO_DROP_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v"})
TRACKING_DROP_SUFFIXES = frozenset({".csv", ".h5", ".hdf5"})
FRAME_CACHE_SIZE = 96
PREFETCH_FORWARD = 16
PREFETCH_BACKWARD = 24
SNAP_PIXELS = 12
LABEL_HIT_RADIUS = 24
ENDPOINT_HIT_RADIUS = 14
SELECT_OUTLINE_COLOR = (255, 255, 0)
SELECT_LABEL_BG_COLOR = (40, 40, 40)
OBB_SOURCE_MISSING = 0.0
OBB_SOURCE_DETECTED = 1.0
OBB_SOURCE_TRACK_INTERPOLATED = 3.0
OBB_SOURCE_PRE_CORRECTION = 4.0
OBB_SOURCE_POST_CORRECTION = 5.0


def _maximize_window_once(window: tk.Tk) -> None:
    def maximize() -> None:
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

    maximize()


def _start_maximized(window: tk.Tk) -> None:
    _maximize_window_once(window)
    for delay_ms in (50, 250, 1000):
        window.after(delay_ms, lambda w=window: _maximize_window_once(w))


# Default review types shown in the candidate tree.
# C1-C2 expose synthetic fill frames, C3 exposes the final identity decision,
# C4 exposes position/direction spike and segment-flip repairs, and C5 flags interior gaps that
# remained unfilled. The id_tracking S0-S6 labels are supplied by assign_types.py.
DEFAULT_ENABLED_ASSIGN_CODES = {
    ATYPE_VEL_DIST,      # S5 - fast-distance active-track recovery
    ATYPE_GAP_DET,       # S6 - long-gap dormant-track recovery
    ATYPE_KF_FILL,        # C1 - KF interpolation
    ATYPE_OVERLAP_FILL,   # C2 - overlap fill
    ATYPE_EMBED_ABSTAIN,   # C3 - embedding targeted but no reliable decision
    ATYPE_EMBED_IDENTITY,  # C3 - embedding confirmed current identity
    ATYPE_EMBED_RELABEL,   # C3 - embedding relabel (ID swap applied)
    ATYPE_DIR_FLIP,       # direction / segment flip repair
    ATYPE_MISSING,        # C5 - unfilled interior gap
}

_FILTER_STAGE_INFO: dict[int, str] = {
    ATYPE_ANCHOR:       "direction_consistency_updated.csv",
    ATYPE_SPIKE:        "spike_drop_updated.csv",
    ATYPE_FILTER_FILL:  "interpolation_updated.csv",
    ATYPE_VITERBI:      "direction_viterbi_updated.csv",
    ATYPE_ID_SWAP:      "long_swap_updated.csv",
}
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
TRACK_SOURCE_FAMILIES = (
    "id_resolved",
    "final",
)
SOURCE_VARIANTS = ("refined", "base")
ASSIGN_TYPE_PICKLE_NAME = "assign_type.pkl"
ASSIGN_TYPE_HISTORY_PICKLE_NAME = "assign_type_history.pkl"
PROVENANCE_CSV_NAME = "provenance.csv"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_config_paths(cfg: dict, config_path: str, parent=None) -> dict:
    """Relocate, resolve, and recover paths for an interactive config load."""
    return prepare_config_for_gui(cfg, config_path, parent=parent)


def append_pt_if_missing(model_name: str) -> str:
    return model_name if str(model_name).endswith(".pt") else f"{model_name}.pt"


def resolve_model_name(model_name: str) -> str:
    base = append_pt_if_missing(str(model_name or ""))
    stem, ext = os.path.splitext(base)
    return base if stem.endswith("-obb") else f"{stem}-obb{ext}"


def parse_weight_spec(weight_spec) -> list[str]:
    if weight_spec is None:
        return ["last"]
    return parse_checkpoint_spec(weight_spec, key="analysis.WEIGHT")


def build_tracking_out_dir(session_path: str, model_name: str, dataset_name: str, run_name: str, video_name: str) -> str:
    return os.path.join(session_path, OUTPUT_ROOT_DIR, model_name, "tracking", dataset_name, run_name, video_name)


@dataclass(frozen=True)
class LrfExperiment:
    lr0: float
    lr0_text: str
    lrf: float
    lrf_text: str
    dataset_name: str


def lrf_experiments_from_cfg(cfg: dict) -> list[LrfExperiment]:
    training = cfg.get("training", {}) or {}
    if not isinstance(training, dict):
        raise RuntimeError("training must be a mapping in config.")
    try:
        num_total_images = int(cfg["NUM_IMAGES"])
    except KeyError as exc:
        raise RuntimeError("NUM_IMAGES is missing in config.") from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError("NUM_IMAGES is not a valid integer in config.") from exc

    lr0_values = parse_lr0_values(training.get("LR0", DEFAULT_LR0))
    lrf_values = parse_lrf_values(training.get("LRF", DEFAULT_LRF))
    return [
        LrfExperiment(
            lr0=lr0,
            lr0_text=format_lr0_for_name(lr0),
            lrf=lrf,
            lrf_text=format_lrf_for_name(lrf),
            dataset_name=experiment_dir_name(num_total_images, lrf, cfg.get("RANDOM_SEED", 0), lr0),
        )
        for lr0 in lr0_values
        for lrf in lrf_values
    ]


def video_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def list_video_files(video_path_in: str) -> list[str]:
    if os.path.isdir(video_path_in):
        files = [os.path.join(video_path_in, f) for f in sorted(os.listdir(video_path_in)) if f.lower().endswith(VIDEO_SUFFIXES)]
    else:
        files = [video_path_in]
    return [f for f in files if os.path.isfile(f)]


def artifact_filename(base_name: str, suffix: str = "") -> str:
    suffix = str(suffix or "").strip()
    if not suffix:
        return base_name
    stem, ext = os.path.splitext(base_name)
    return f"{stem}_{suffix}{ext}"


def artifact_path(out_dir: str, base_name: str, suffix: str = "") -> str:
    return tracking_artifact_path(out_dir, base_name, suffix)


def tracking_artifact_suffix_for_family(family: str) -> str:
    return {
        'id_resolved': 'id_resolved',
        'final': 'filled',
    }[family]


@dataclass
class ArrowState:
    front: np.ndarray
    rear: np.ndarray


class VideoFrameReader:
    """Thread-safe video reader with bounded, latest-request prefetching.

    Demand decoding and prefetch decoding use separate ``VideoCapture`` objects,
    but each capture is owned by a single serialized path.  The shared LRU cache
    is protected by one lock, so GUI navigation cannot race with background
    prefetch threads.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.prefetch_cap = cv2.VideoCapture(video_path)
        if not self.prefetch_cap.isOpened():
            self.prefetch_cap.release()
            self.prefetch_cap = None

        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._cap_lock = threading.Lock()
        self._prefetch_cap_lock = threading.Lock()
        self._cap_pos: int = -1
        self._prefetch_pos: int = -1
        self._closed = threading.Event()

        # A single prefetch worker handles only the newest center-frame request.
        # This prevents the unbounded thread accumulation caused by rapid
        # forward/backward navigation.
        self._prefetch_cond = threading.Condition()
        self._prefetch_request: tuple[int, int, int, int] | None = None
        self._prefetch_generation = 0
        self._prefetch_thread: threading.Thread | None = None
        if self.prefetch_cap is not None:
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_loop,
                name="video-prefetch",
                daemon=True,
            )
            self._prefetch_thread.start()

    def _decode(
        self,
        cap: cv2.VideoCapture,
        frame_idx: int,
        current_pos: int,
    ) -> tuple[np.ndarray, int]:
        if self._closed.is_set():
            raise RuntimeError("Video reader is closed.")
        if current_pos != frame_idx:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx)):
                # Some backends return False even when seeking succeeds, so the
                # subsequent read remains the authoritative check.
                pass
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_idx + 1

    def _trim_cache_locked(self) -> None:
        while len(self.cache) > FRAME_CACHE_SIZE:
            self.cache.popitem(last=False)

    def _store_cached(self, frame_idx: int, frame: np.ndarray) -> None:
        if self._closed.is_set():
            return
        with self._cache_lock:
            self.cache[frame_idx] = frame
            self.cache.move_to_end(frame_idx)
            self._trim_cache_locked()

    def get_cached(self, frame_idx: int) -> np.ndarray | None:
        """Return a copy of a cached frame without ever blocking on decoding."""
        frame_idx = int(frame_idx)
        with self._cache_lock:
            frame = self.cache.get(frame_idx)
            if frame is None:
                return None
            self.cache.move_to_end(frame_idx)
            return frame.copy()

    def read(self, frame_idx: int) -> np.ndarray:
        """Decode one demanded frame. Safe to call from one background worker."""
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"Frame index out of range: {frame_idx}")

        cached = self.get_cached(frame_idx)
        if cached is not None:
            return cached

        with self._cap_lock:
            if self._closed.is_set():
                raise RuntimeError("Video reader is closed.")
            cached = self.get_cached(frame_idx)
            if cached is not None:
                return cached
            frame, self._cap_pos = self._decode(self.cap, frame_idx, self._cap_pos)
            self._store_cached(frame_idx, frame)
            return frame.copy()

    def request_prefetch(
        self,
        center_frame: int,
        forward: int = PREFETCH_FORWARD,
        backward: int = PREFETCH_BACKWARD,
    ) -> None:
        """Replace any pending prefetch plan with a newest-frame plan."""
        if self.prefetch_cap is None or self._closed.is_set():
            return
        center_frame = int(center_frame)
        with self._prefetch_cond:
            self._prefetch_generation += 1
            self._prefetch_request = (
                self._prefetch_generation,
                center_frame,
                max(0, int(forward)),
                max(0, int(backward)),
            )
            self._prefetch_cond.notify()

    def _prefetch_order(self, center: int, forward: int, backward: int) -> list[int]:
        # Nearest frames first on both sides.  Direction reversal therefore has
        # an adjacent cached frame without launching a dedicated extra thread.
        order: list[int] = []
        for distance in range(1, max(forward, backward) + 1):
            if distance <= backward:
                fid = center - distance
                if 0 <= fid < self.frame_count:
                    order.append(fid)
            if distance <= forward:
                fid = center + distance
                if 0 <= fid < self.frame_count:
                    order.append(fid)
        return order

    def _prefetch_loop(self) -> None:
        while not self._closed.is_set():
            with self._prefetch_cond:
                self._prefetch_cond.wait_for(
                    lambda: self._closed.is_set() or self._prefetch_request is not None
                )
                if self._closed.is_set():
                    return
                request = self._prefetch_request
                self._prefetch_request = None

            if request is None:
                continue
            generation, center, forward, backward = request
            for frame_idx in self._prefetch_order(center, forward, backward):
                if self._closed.is_set():
                    return
                with self._prefetch_cond:
                    if generation != self._prefetch_generation:
                        break
                if self.get_cached(frame_idx) is not None:
                    continue
                try:
                    with self._prefetch_cap_lock:
                        if self._closed.is_set() or self.prefetch_cap is None:
                            return
                        if self.get_cached(frame_idx) is not None:
                            continue
                        frame, self._prefetch_pos = self._decode(
                            self.prefetch_cap,
                            frame_idx,
                            self._prefetch_pos,
                        )
                    self._store_cached(frame_idx, frame)
                except Exception:
                    self._prefetch_pos = -1
                    # A bad seek/read must not terminate the worker permanently.
                    continue

    def close(self) -> None:
        """Stop new work immediately and release captures without blocking Tk."""
        if self._closed.is_set():
            return
        self._closed.set()
        with self._prefetch_cond:
            self._prefetch_request = None
            self._prefetch_cond.notify_all()

        # Cleanup may need to wait for a backend read call.  Running it outside
        # the Tk thread prevents a slow codec/backend from freezing the window.
        def cleanup() -> None:
            thread = self._prefetch_thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
            with self._cap_lock:
                self.cap.release()
            if self.prefetch_cap is not None:
                with self._prefetch_cap_lock:
                    self.prefetch_cap.release()
            with self._cache_lock:
                self.cache.clear()

        threading.Thread(target=cleanup, name="video-reader-cleanup", daemon=True).start()


class UmaDirectionRefinementApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        configure_dpi_scaling(self)
        install_window_icon(self)
        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.minsize(1400, 850)

        self.video_path_var = tk.StringVar()
        self.csv_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a video and UMATracker CSV")
        self.preview_title_var = tk.StringVar(value="")
        self.frame_var = tk.IntVar(value=0)
        self.id_var = tk.IntVar(value=0)
        self.show_labels_var = tk.BooleanVar(value=False)
        self.show_arrows_var = tk.BooleanVar(value=True)
        self.show_select_circle_var = tk.BooleanVar(value=False)
        self.arrow_width_var = tk.IntVar(value=2)
        self.label_size_var = tk.IntVar(value=16)
        self.reverse_direction_var = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value="amadeus")
        self.config_path_var = tk.StringVar()
        self.dataset_var = tk.StringVar()
        self.weight_var = tk.StringVar()
        self.available_weight_options: list[dict] = []
        self.show_only_unedited_var = tk.BooleanVar(value=False)

        self.reader: Optional[VideoFrameReader] = None
        self.df: Optional[pd.DataFrame] = None
        self.frame_count = 0
        self.num_ids = 0
        self.current_frame = 0
        self.current_id = 0
        self.frame_cache_rgb: Optional[np.ndarray] = None
        self.tk_image = None
        self.edited_keys: set[tuple[int, int]] = set()
        self.last_save_path: Optional[str] = None
        self.prefetch_after_id: Optional[str] = None
        self._decode_gen: int = 0
        self._last_displayed_frame: int = -1
        self._frame_queue: queue.Queue = queue.Queue()
        self._decode_request_queue: queue.Queue = queue.Queue(maxsize=1)
        self._decode_stop = threading.Event()
        self._closing = False
        self._decode_thread = threading.Thread(
            target=self._decode_worker_loop,
            name="video-demand-decode",
            daemon=True,
        )
        self._decode_thread.start()
        self._prefetch_cancel: threading.Event = threading.Event()
        self.status_clear_job: Optional[str] = None
        self.df_np: Optional[np.ndarray] = None
        self._front_x_idx: Optional[np.ndarray] = None
        self._front_y_idx: Optional[np.ndarray] = None
        self._rear_x_idx: Optional[np.ndarray] = None
        self._rear_y_idx: Optional[np.ndarray] = None
        self._refresh_tree_job: Optional[str] = None
        self.save_count = 0
        self.cfg: dict = {}
        self.datasets: list[dict] = []
        self.current_dataset: Optional[dict] = None
        self.suspects: list[dict] = []
        self.current_suspect_index: Optional[int] = None
        self.provenance_lookup: dict[tuple[int, int], dict[str, float]] = {}
        self.candidate_labels_by_key: dict[tuple[int, int], str] = {}
        self.enabled_assign_codes: set[float] = set(DEFAULT_ENABLED_ASSIGN_CODES)
        self.tree_event_suppressed = False
        self._candidate_tree_suppress_token: int = 0
        self.pending_candidate_select_job: Optional[str] = None

        self.base_fit_scale = 1.0
        self.zoom_scale = 1.0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.view_initialized = False
        self.current_image_shape: Optional[tuple[int, int]] = None

        self.drag_mode: Optional[str] = None
        self.drag_tid: Optional[int] = None
        self.drag_last_canvas: Optional[np.ndarray] = None
        self.drag_start_front: Optional[np.ndarray] = None
        self.drag_start_rear: Optional[np.ndarray] = None
        self.drag_anchor_img: Optional[np.ndarray] = None
        self._pan_start_canvas: Optional[np.ndarray] = None
        self._pan_start_offset: Optional[np.ndarray] = None
        self.left_button_down = False
        self.hover_tid: Optional[int] = None

        self.selected_tids: list[int] = []
        self.swap_popup: Optional[ctk.CTkToplevel] = None
        self._last_right_click_tid: Optional[int] = None
        self._last_right_click_time: float = 0.0
        self._RIGHT_DBLCLICK_MS: int = 400
        self.id_palette: list[tuple[int, int, int]] = []
        self._main_pane: Optional[tk.PanedWindow] = None
        self._left_pane: Optional[ctk.CTkFrame] = None
        self._right_pane: Optional[ctk.CTkFrame] = None
        self._left_pane_visible: bool = True

        self._key_play_job: Optional[str] = None
        self._key_play_delta: int = 0
        self._key_play_wait_count: int = 0

        self.playback_active = False
        self.playback_interval_ms = 95
        self.playback_job: Optional[str] = None
        self.play_button: Optional[ctk.CTkButton] = None

        self.requested_frame = 0
        self.frame_render_job: Optional[str] = None
        self.frame_render_interval_ms = 16

        self._build_ui()
        self._bind_events()
        self._bind_traces()
        self.focus_set()
        self._pump_frame_queue()
        _start_maximized(self)

    def _build_ui(self):
        # Dark styling for ttk widgets
        _style = ttk.Style()
        _style.theme_use("clam")
        _style.configure("Treeview",
            background="#2b2b2b", foreground="#dce4ee",
            fieldbackground="#2b2b2b", bordercolor="#444444", rowheight=20)
        _style.configure("Treeview.Heading",
            background="#3a3a3a", foreground="#dce4ee", relief="flat")
        _style.map("Treeview",
            background=[("selected", "#1f6aa5")],
            foreground=[("selected", "white")])
        _style.map("Treeview.Heading",
            background=[("active", "#444444")])
        _style.configure("Vertical.TScrollbar",
            background="#3a3a3a", troughcolor="#2b2b2b",
            arrowcolor="#dce4ee", bordercolor="#2b2b2b")

        # Shared dark config for tk.Spinbox
        self._spin_cfg = dict(
            bg="#343638", fg="#dce4ee",
            insertbackground="#dce4ee",
            buttonbackground="#565b5e",
            relief="flat",
            highlightthickness=2,
            highlightbackground="#565b5e",
            highlightcolor="#1f8040",
            selectbackground="#1f6aa5", selectforeground="white",
        )

        top = ctk.CTkFrame(self, corner_radius=0)
        top.pack(fill="x", padx=10, pady=(3, 0))

        # Mode selector (always visible)
        mode_row = ctk.CTkFrame(top, corner_radius=0)
        mode_row.pack(fill="x", pady=(0, 1))
        ctk.CTkLabel(mode_row, text="Mode", width=110, anchor="w").pack(side="left")
        ctk.CTkRadioButton(mode_row, text="AMADEUS", variable=self.mode_var, value="amadeus",
                           command=self._refresh_mode_ui).pack(side="left")
        ctk.CTkRadioButton(mode_row, text="Dataset", variable=self.mode_var, value="dataset",
                           command=self._refresh_mode_ui).pack(side="left", padx=(12, 0))
        ctk.CTkLabel(mode_row, textvariable=self.status_var, anchor="e").pack(side="right", fill="x", expand=True, padx=(12, 0))

        # Config row (AMADEUS mode only)
        self.config_row = ctk.CTkFrame(top, corner_radius=0)
        ctk.CTkLabel(self.config_row, text="Config", width=110, anchor="w").pack(side="left")
        ctk.CTkEntry(self.config_row, textvariable=self.config_path_var, width=300, height=26).pack(side="left", expand=True, fill="x", padx=5)
        ctk.CTkButton(self.config_row, text="Load Config", width=100, height=26, command=self.browse_config).pack(side="left", padx=4)

        # Weight + Dataset on the same row (AMADEUS mode only)
        self.num_dataset_row = ctk.CTkFrame(top, corner_radius=0)
        ctk.CTkLabel(self.num_dataset_row, text="Weight", width=110, anchor="w").pack(side="left")
        self.weight_combo = ctk.CTkComboBox(self.num_dataset_row, variable=self.weight_var, state="readonly", width=200, height=26,
                                                   command=lambda choice: self.on_weight_selected())
        self.weight_combo.pack(side="left", padx=(5, 16))
        ctk.CTkLabel(self.num_dataset_row, text="Dataset", anchor="w").pack(side="left")
        self.dataset_combo = ctk.CTkComboBox(self.num_dataset_row, variable=self.dataset_var, state="readonly", width=400, height=26,
                                              command=lambda choice: self.on_dataset_selected())
        self.dataset_combo.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Video row (Dataset mode only)
        self.video_row = ctk.CTkFrame(top, corner_radius=0)
        ctk.CTkLabel(self.video_row, text="Video", width=110, anchor="w").pack(side="left")
        ctk.CTkEntry(self.video_row, textvariable=self.video_path_var, width=300, height=26).pack(side="left", expand=True, fill="x", padx=5)
        ctk.CTkButton(self.video_row, text="Reference", width=90, height=26, command=self.browse_video).pack(side="left", padx=4)

        # CSV row (Dataset mode only)
        self.csv_row = ctk.CTkFrame(top, corner_radius=0)
        ctk.CTkLabel(self.csv_row, text="CSV", width=110, anchor="w").pack(side="left")
        ctk.CTkEntry(self.csv_row, textvariable=self.csv_path_var, width=300, height=26).pack(side="left", expand=True, fill="x", padx=5)
        ctk.CTkButton(self.csv_row, text="Reference", width=90, height=26, command=self.browse_csv).pack(side="left", padx=4)

        self._refresh_mode_ui()

        _cbkw = dict(checkbox_width=14, checkbox_height=14)
        ctrl = ctk.CTkFrame(self, corner_radius=0, height=30)
        ctrl.pack(fill="x", padx=10, pady=0)
        ctrl.pack_propagate(False)
        ctk.CTkButton(ctrl, text="Save As", width=66, height=22, corner_radius=3, command=self.save_csv_as).pack(side="left", padx=(0, 4), pady=4)
        ctk.CTkButton(ctrl, text="Shuffle colors", width=90, height=22, corner_radius=3, command=self.shuffle_colors).pack(side="left", pady=4)
        ctk.CTkFrame(ctrl, width=2, fg_color=("gray60", "gray40"), corner_radius=0).pack(side="left", fill="y", padx=7, pady=4)
        ctk.CTkCheckBox(ctrl, text="show arrows", variable=self.show_arrows_var, **_cbkw).pack(side="left", pady=4)
        ctk.CTkCheckBox(ctrl, text="show select circle", variable=self.show_select_circle_var, **_cbkw).pack(side="left", padx=(8, 0), pady=4)
        ctk.CTkCheckBox(ctrl, text="show labels", variable=self.show_labels_var, **_cbkw).pack(side="left", padx=(8, 0), pady=4)
        ctk.CTkCheckBox(ctrl, text="swap front/rear", variable=self.reverse_direction_var, command=self.redraw_current_frame, **_cbkw).pack(side="left", padx=(8, 0), pady=4)
        ctk.CTkFrame(ctrl, width=2, fg_color=("gray60", "gray40"), corner_radius=0).pack(side="left", fill="y", padx=7, pady=4)
        ctk.CTkLabel(ctrl, text="Arrow width").pack(side="left")
        tk.Spinbox(ctrl, from_=1, to=30, width=4, textvariable=self.arrow_width_var, **self._spin_cfg).pack(side="left", padx=(3, 8))
        ctk.CTkLabel(ctrl, text="Label size").pack(side="left")
        tk.Spinbox(ctrl, from_=6, to=72, width=4, textvariable=self.label_size_var, **self._spin_cfg).pack(side="left", padx=(3, 8))
        ctk.CTkFrame(ctrl, width=2, fg_color=("gray60", "gray40"), corner_radius=0).pack(side="left", fill="y", padx=7, pady=4)
        self._assign_history_var = tk.StringVar()
        ctk.CTkLabel(ctrl, textvariable=self._assign_history_var, anchor="w", text_color=("gray20", "gray80")).pack(side="left", padx=(2, 0))
        ctk.CTkButton(ctrl, text="Save Screenshot", width=120, height=22, corner_radius=3,
                      command=self.save_canvas_screenshot).pack(side="right", padx=(4, 0), pady=4)
        ctk.CTkButton(ctrl, text="Fit Window Scale", width=120, height=22, corner_radius=3,
                      command=self.fit_window_scale).pack(side="right", padx=(4, 0), pady=4)

        main = tk.PanedWindow(self, orient="horizontal", sashrelief="raised")
        main.pack(fill="both", expand=True, padx=10, pady=(0, 2))
        self._main_pane = main

        left = ctk.CTkFrame(main, width=420, corner_radius=0)
        main.add(left, minsize=340)
        self._left_pane = left
        right = ctk.CTkFrame(main, corner_radius=0)
        main.add(right, minsize=850)
        self._right_pane = right

        candidate_frame = ctk.CTkFrame(left, corner_radius=4)
        candidate_frame.pack(fill="both", expand=True, padx=2, pady=2)
        ctk.CTkLabel(candidate_frame, text="Refinement candidates", font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=6, pady=(4, 2))

        ctrl_row = ctk.CTkFrame(candidate_frame, corner_radius=0)
        ctrl_row.pack(fill="x", padx=4, pady=(4, 0))
        ctk.CTkCheckBox(ctrl_row, text="only unedited", variable=self.show_only_unedited_var,
                        command=self.refresh_candidate_tree).pack(side="left")
        self._type_panel_open = False
        self._type_toggle_btn = ctk.CTkButton(ctrl_row, text="> types", width=80, anchor="w",
                                               command=self._toggle_type_filter_panel)
        self._type_toggle_btn.pack(side="left", padx=(8, 0))

        self._type_filter_panel = ctk.CTkFrame(candidate_frame, corner_radius=0)
        self._type_filter_cb_vars: dict[float, tk.BooleanVar] = {}
        self._type_filter_label_vars: dict[float, tk.StringVar] = {}
        self._build_type_filter_panel()

        tree_frame = ctk.CTkFrame(candidate_frame, corner_radius=0)
        tree_frame.pack(fill="both", expand=True)
        self.candidate_tree = ttk.Treeview(tree_frame, columns=("frame", "id", "type", "edit"), show="headings", height=24)
        for col, label, width in (("frame", "frame", 55), ("id", "ID", 30), ("type", "classification", 205), ("edit", "edit", 45)):
            self.candidate_tree.heading(col, text=label)
            self.candidate_tree.column(col, width=width, anchor="center")
        self.candidate_tree.pack(side="left", fill="both", expand=True)
        self.candidate_tree.bind("<<TreeviewSelect>>", self.on_candidate_select)
        self.candidate_tree.bind("<Down>",  lambda _: self._step_candidate(+1) or "break")
        self.candidate_tree.bind("<Up>",    lambda _: self._step_candidate(-1) or "break")
        self.candidate_tree.bind("<Right>", lambda _: self.step_frame(+1)      or "break")
        self.candidate_tree.bind("<Left>",  lambda _: self.step_frame(-1)      or "break")
        cand_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.candidate_tree.yview)
        cand_scroll.pack(side="right", fill="y")
        self.candidate_tree.configure(yscrollcommand=cand_scroll.set)

        export_row = ctk.CTkFrame(candidate_frame, corner_radius=0)
        export_row.pack(fill="x", padx=4, pady=(4, 2))
        ctk.CTkButton(export_row, text="Export Video", width=120, height=26,
                      command=self.open_export_video_dialog).pack(side="left")

        nav = ctk.CTkFrame(right, corner_radius=0)
        nav.pack(fill="x", pady=(0, 6))

        ctk.CTkButton(nav, text="|<", width=50, command=self.go_first_frame).pack(side="left")
        ctk.CTkButton(nav, text="<", width=50, command=lambda: self.step_frame(-1)).pack(side="left", padx=4)
        self.play_button = ctk.CTkButton(nav, text="Play", width=60, command=self.toggle_playback)
        self.play_button.pack(side="left")
        ctk.CTkButton(nav, text=">", width=50, command=lambda: self.step_frame(1)).pack(side="left", padx=4)
        ctk.CTkButton(nav, text=">|", width=50, command=self.go_last_frame).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(nav, text="frame").pack(side="left")
        self.frame_spin = tk.Spinbox(nav, width=10, textvariable=self.frame_var, command=self.on_frame_spin, **self._spin_cfg)
        self.frame_spin.pack(side="left", padx=(6, 12))
        self.frame_spin.bind("<Return>", self.on_frame_entry_commit, add="+")
        self.frame_spin.bind("<KP_Enter>", self.on_frame_entry_commit, add="+")
        self.frame_spin.bind("<FocusOut>", self.on_frame_entry_commit, add="+")

        ctk.CTkLabel(nav, text="ID").pack(side="left")
        self.id_spin = tk.Spinbox(nav, width=8, textvariable=self.id_var, command=self.on_id_spin, wrap=True, **self._spin_cfg)
        self.id_spin.pack(side="left", padx=(6, 12))
        self.id_spin.bind("<Return>", self.on_id_entry_commit, add="+")
        self.id_spin.bind("<KP_Enter>", self.on_id_entry_commit, add="+")

        self.frame_scale = ctk.CTkSlider(nav, orientation="horizontal", command=self.on_frame_scale)
        self.frame_scale.pack(side="left", fill="x", expand=True, padx=(0, 12))

        self.canvas = tk.Canvas(right, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._canvas_drop_state = install_canvas_file_drop(
            self,
            self.canvas,
            allowed_suffixes=VIDEO_DROP_SUFFIXES | TRACKING_DROP_SUFFIXES,
            on_path=self._handle_dropped_file,
            status_callback=lambda text: self.set_status(text, auto_clear=False),
            label="video or CSV",
        )


    def _bind_events(self):
        for key in ("Left", "Right", "Up", "Down"):
            self.bind_all(f"<KeyPress-{key}>", self.on_key_press, add="+")
            self.bind_all(f"<KeyRelease-{key}>", self.on_key_release, add="+")
        self.bind_all("<KeyPress-space>", self.on_space_press, add="+")
        self.bind("<Configure>", self.on_window_configure)
        self.bind("<FocusOut>", self.on_focus_out, add="+")
        self.bind_all("<Control-s>", self.on_ctrl_s)
        if sys.platform == "darwin":
            self.bind_all("<Command-s>", self.on_ctrl_s)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<ButtonPress-3>", self.on_canvas_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)

    def _bind_traces(self):
        for var in [
            self.show_labels_var,
            self.show_arrows_var,
            self.show_select_circle_var,
            self.arrow_width_var,
            self.label_size_var,
        ]:
            var.trace_add("write", lambda *_: self.redraw_current_frame())

    def _reset_session(self) -> None:
        """Discard all loaded data. Called on mode switch."""
        self.stop_playback(update_button=False)
        self._stop_key_play()
        self._close_swap_popup()
        self._invalidate_decode_requests()

        # Cancel pending after-jobs
        for attr in ("frame_render_job", "prefetch_after_id", "_refresh_tree_job",
                     "status_clear_job", "pending_candidate_select_job",
                     "playback_job"):
            job = getattr(self, attr, None)
            if job:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
                setattr(self, attr, None)

        # Close video reader
        if self.reader is not None:
            self.reader.close()
            self.reader = None

        # Data
        self.df = None
        self.df_np = None
        self._front_x_idx = None
        self._front_y_idx = None
        self._rear_x_idx = None
        self._rear_y_idx = None
        self.frame_count = 0
        self.num_ids = 0
        self.current_frame = 0
        self.requested_frame = 0
        self._last_displayed_frame = -1
        self.current_id = 0
        self.frame_cache_rgb = None
        self.tk_image = None
        self.last_save_path = None
        self.save_count = 0
        self.edited_keys.clear()
        self.selected_tids.clear()
        self.id_palette = []
        self.drag_mode = None
        self.drag_tid = None
        self.hover_tid = None
        self.left_button_down = False
        self._pan_start_canvas = None
        self._pan_start_offset = None

        # Candidates / AMADEUS state
        self.suspects = []
        self.current_suspect_index = None
        self.assign_type_history: dict = {}
        self.provenance_lookup = {}
        self.candidate_labels_by_key = {}
        self.tree_event_suppressed = False
        self._candidate_tree_suppress_token += 1
        self.current_dataset = None
        self.cfg = {}
        self.datasets = []
        self.available_weight_options = []

        # Path vars
        self.config_path_var.set("")
        self.weight_var.set("")
        self.dataset_var.set("")
        self.video_path_var.set("")
        self.csv_path_var.set("")

        # View
        self._reset_view_state()
        self.status_var.set("")
        self.preview_title_var.set("")
        self.frame_var.set(0)
        self.id_var.set(0)

        # UI elements (may not exist yet during __init__)
        try:
            self.canvas.delete("all")
        except Exception:
            pass
        try:
            self.candidate_tree.delete(*self.candidate_tree.get_children())
        except Exception:
            pass
        try:
            self.weight_combo.configure(values=[])
            self.dataset_combo.configure(values=[])
        except Exception:
            pass

    def _refresh_mode_ui(self):
        self._reset_session()
        mode = self.mode_var.get()
        for row in (self.config_row, self.num_dataset_row, self.video_row, self.csv_row):
            row.pack_forget()
        if mode == "amadeus":
            self.config_row.pack(fill="x", pady=(4, 0))
            self.num_dataset_row.pack(fill="x", pady=(4, 0))
            self._show_left_pane()
        else:
            self.video_row.pack(fill="x", pady=(4, 0))
            self.csv_row.pack(fill="x", pady=(4, 0))
            self._hide_left_pane()

    def _hide_left_pane(self):
        if self._main_pane is None or self._left_pane is None:
            return
        if not self._left_pane_visible:
            return
        try:
            self._main_pane.forget(self._left_pane)
            self._left_pane_visible = False
        except tk.TclError:
            pass

    def _show_left_pane(self):
        if self._main_pane is None or self._left_pane is None or self._right_pane is None:
            return
        if self._left_pane_visible:
            return
        try:
            # forget -> add ensures left-to-right pane order
            self._main_pane.forget(self._right_pane)
            self._main_pane.add(self._left_pane, minsize=340)
            self._main_pane.add(self._right_pane, minsize=850)
            self._left_pane_visible = True
        except tk.TclError:
            pass

    def browse_config(self):
        path = filedialog.askopenfilename(
            title="Select config.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        self.config_path_var.set(path)
        self.load_config_and_discover_datasets(path)

    def load_config_and_discover_datasets(self, config_path: str):
        try:
            cfg = normalize_config_paths(load_yaml(config_path), config_path, parent=self)
            options = self.discover_weight_options(cfg)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.cfg = cfg
        self.available_weight_options = options
        labels = [o["label"] for o in options]
        self.weight_combo.configure(values=labels)
        if not options:
            self.weight_var.set("")
            self.dataset_combo.configure(values=[])
            self.dataset_var.set("")
            self.datasets = []
            self.set_status(
                "No last, best, or epoch directory was found under tracking dir.",
                auto_clear=False,
            )
            return
        preferred = self._preferred_weight_label(cfg, options)
        self.weight_var.set(preferred)
        self.refresh_datasets_for_selected_weight(load_first=True)

    def _preferred_weight_label(self, cfg: dict, options: list[dict]) -> str:
        configured = parse_weight_spec((cfg.get("analysis", {}) or {}).get("WEIGHT", "last"))
        for requested in configured:
            for option in options:
                if option["weight"] == requested:
                    return option["label"]
        for option in options:
            if option.get("is_last"):
                return option["label"]
        return options[0]["label"]

    def discover_weight_options(self, cfg: dict) -> list[dict]:
        session_path = str(cfg.get("SESSION_PATH", "")).strip()
        if not session_path:
            raise RuntimeError("SESSION_PATH is missing in config.")
        if not os.path.isdir(session_path):
            raise RuntimeError(f"SESSION_PATH not found: {session_path}")
        training = cfg.get("training", {}) or {}
        model_name = os.path.splitext(resolve_model_name(training.get("PRETRAINED_MODEL", "yolo11n-obb")))[0]

        experiments = lrf_experiments_from_cfg(cfg)
        options_by_weight: dict[str, dict] = {}

        def add_option(weight: str, epoch: int, path: str, *, is_best: bool = False, is_last: bool = False) -> None:
            if weight in options_by_weight:
                options_by_weight[weight].setdefault("paths", []).append(path)
                return
            if is_best:
                label = "best.pt  (best)"
            elif is_last:
                label = "last.pt  (last)"
            else:
                label = f"{epoch}  ({weight})"
            options_by_weight[weight] = {
                "label": label,
                "epoch": epoch,
                "weight": weight,
                "path": path,
                "paths": [path],
                "is_best": is_best,
                "is_last": is_last,
            }

        for experiment in experiments:
            resolve_existing_experiment_dir_name(
                session_path,
                model_name,
                experiment.dataset_name,
                stages=("tracking",),
                warn_fn=lambda msg: (print(msg, flush=True), self.set_status(msg, auto_clear=False)),
            )
            tracking_root = os.path.join(
                session_path, OUTPUT_ROOT_DIR, model_name, "tracking", experiment.dataset_name
            )
            if not os.path.isdir(tracking_root):
                continue
            best_path = os.path.join(tracking_root, "best")
            if os.path.isdir(best_path):
                add_option("best", -1, best_path, is_best=True)

            last_path = os.path.join(tracking_root, "last")
            if os.path.isdir(last_path):
                add_option("last", -1, last_path, is_last=True)

            for name in sorted(os.listdir(tracking_root)):
                path = os.path.join(tracking_root, name)
                if not os.path.isdir(path):
                    continue
                match = re.fullmatch(r"epoch([1-9]\d*)", name)
                if not match:
                    continue
                epoch_idx = weight_to_checkpoint_number(name)
                add_option(name, epoch_idx, path)
        options = list(options_by_weight.values())
        options.sort(
            key=lambda d: (
                0 if d.get("is_last") else (1 if d.get("is_best") else 2),
                -int(d["epoch"]),
            )
        )
        return options

    def on_weight_selected(self):
        if not self.cfg:
            return
        self.refresh_datasets_for_selected_weight(load_first=True)

    def selected_weight(self) -> str | None:
        label = self.weight_var.get().strip()
        for opt in self.available_weight_options:
            if opt["label"] == label:
                return str(opt["weight"])
        return None

    def refresh_datasets_for_selected_weight(self, load_first: bool = False):
        try:
            datasets = self.discover_datasets_from_config(self.cfg)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.datasets = datasets
        labels = [d["label"] for d in datasets]
        self.dataset_combo.configure(values=labels)
        if not datasets:
            self.dataset_var.set("")
            self.set_status("No obb/direction dataset was found for the selected weight.", auto_clear=False)
            return
        self.dataset_var.set(labels[0])
        if load_first:
            self.load_dataset_record(datasets[0])

    def discover_datasets_from_config(self, cfg: dict) -> list[dict]:
        session_path = str(cfg.get("SESSION_PATH", "")).strip()
        if not session_path:
            raise RuntimeError("SESSION_PATH is missing in config.")
        if not os.path.isdir(session_path):
            raise RuntimeError(f"SESSION_PATH not found: {session_path}")

        training = cfg.get("training", {}) or {}
        tracking_video_path = str(cfg.get("TRACKING_VIDEO_PATH", "")).strip()
        model_name = os.path.splitext(resolve_model_name(training.get("PRETRAINED_MODEL", "yolo11n-obb")))[0]

        experiments = lrf_experiments_from_cfg(cfg)
        multi_lrf = len(experiments) > 1
        selected_weight = self.selected_weight()
        if not selected_weight:
            weights = [o["weight"] for o in self.available_weight_options] or parse_weight_spec(None)
        else:
            weights = [selected_weight]
        video_files = list_video_files(tracking_video_path) if tracking_video_path else []
        video_lookup = {video_stem(p): p for p in video_files}
        datasets: list[dict] = []
        seen: set[tuple[str, str, str]] = set()

        def candidate_video_path(video_name: str) -> str | None:
            if video_name in video_lookup and os.path.exists(video_lookup[video_name]):
                return video_lookup[video_name]
            for cand_dir in [tracking_video_path, session_path]:
                if cand_dir and os.path.isdir(cand_dir):
                    for ext in VIDEO_SUFFIXES:
                        cand = os.path.join(cand_dir, video_name + ext)
                        if os.path.exists(cand):
                            return cand
            return None

        def source_paths(out_dir: str, family: str) -> tuple[str | None, str | None, str]:
            tag = "_id_resolved" if family == "id_resolved" else ""
            refined_obb = os.path.join(out_dir, f"obbs{tag}_refined.csv")
            refined_dir = os.path.join(out_dir, f"directions{tag}_refined.csv")
            base_obb = os.path.join(out_dir, f"obbs{tag}.csv")
            base_dir = os.path.join(out_dir, f"directions{tag}.csv")
            if os.path.exists(refined_obb) and os.path.exists(refined_dir):
                return refined_obb, refined_dir, "refined"
            if os.path.exists(base_obb) and os.path.exists(base_dir):
                return base_obb, base_dir, "base"
            return None, None, ""

        def add_out_dir(out_dir: str, experiment: LrfExperiment, video_path: str | None = None):
            if not os.path.isdir(out_dir):
                return
            video_name = os.path.basename(out_dir)
            video_path_resolved = video_path or candidate_video_path(video_name)
            if not video_path_resolved:
                return
            run_name = os.path.basename(os.path.dirname(out_dir))
            lrf_label = (
                f"  [LR0 {experiment.lr0_text} / LRF {experiment.lrf_text}]"
                if multi_lrf else ""
            )
            for family in TRACK_SOURCE_FAMILIES:
                obb_path, direction_path, variant = source_paths(out_dir, family)
                if not obb_path or not direction_path:
                    continue
                converted_csv_path = str(_convert.rear_front_csv_path(obb_path))
                artifact_suffix = tracking_artifact_suffix_for_family(family)
                assign_path = artifact_path(out_dir, ASSIGN_TYPE_PICKLE_NAME, artifact_suffix)
                history_path = artifact_path(out_dir, ASSIGN_TYPE_HISTORY_PICKLE_NAME, artifact_suffix)
                provenance_path = artifact_path(out_dir, PROVENANCE_CSV_NAME, artifact_suffix)
                key = (os.path.abspath(video_path_resolved), os.path.abspath(obb_path), os.path.abspath(direction_path))
                if key in seen:
                    continue
                seen.add(key)
                datasets.append({
                    "label": f"{video_name}{lrf_label}  [{run_name}]  {family}{' refined' if variant == 'refined' else ''}",
                    "video_path": video_path_resolved,
                    "csv_path": converted_csv_path,
                    "converted_csv_path": converted_csv_path,
                    "obb_path": obb_path,
                    "direction_path": direction_path,
                    "assign_path": assign_path,
                    "history_path": history_path,
                    "provenance_path": provenance_path,
                    "out_dir": out_dir,
                    "dataset_name": experiment.dataset_name,
                    "lr0": experiment.lr0,
                    "lr0_text": experiment.lr0_text,
                    "lrf": experiment.lrf,
                    "lrf_text": experiment.lrf_text,
                    "family": family,
                    "variant": variant,
                    "run_name": run_name,
                })

        for video_path in video_files:
            vname = video_stem(video_path)
            for experiment in experiments:
                resolve_existing_experiment_dir_name(
                    session_path,
                    model_name,
                    experiment.dataset_name,
                    stages=("tracking",),
                    warn_fn=lambda msg: (print(msg, flush=True), self.set_status(msg, auto_clear=False)),
                )
                for weight in weights:
                    out_dir = build_tracking_out_dir(
                        session_path, model_name, experiment.dataset_name, weight, vname
                    )
                    add_out_dir(out_dir, experiment, video_path)

        # Also discover matching video directories under the selected weight.
        for experiment in experiments:
            resolve_existing_experiment_dir_name(
                session_path,
                model_name,
                experiment.dataset_name,
                stages=("tracking",),
                warn_fn=lambda msg: (print(msg, flush=True), self.set_status(msg, auto_clear=False)),
            )
            for weight in weights:
                weight_dir = os.path.join(
                    session_path, OUTPUT_ROOT_DIR, model_name, "tracking", experiment.dataset_name, weight
                )
                if os.path.isdir(weight_dir):
                    for name in sorted(os.listdir(weight_dir)):
                        add_out_dir(os.path.join(weight_dir, name), experiment)

        _family_order = {
            "id_resolved": 0,
            "final": 1,
        }
        datasets.sort(key=lambda d: (
            d["video_path"],
            float(d.get("lr0", 0.0)),
            float(d.get("lrf", 0.0)),
            d["run_name"],
            _family_order.get(d["family"], 9),
            d["variant"] != "refined",  # refined before base
        ))
        return datasets

    def on_dataset_selected(self):
        if self.tree_event_suppressed:
            return
        label = self.dataset_var.get().strip()
        matches = [d for d in self.datasets if d["label"] == label]
        if matches:
            self.load_dataset_record(matches[0], preserve_view=True)

    def load_dataset_record(self, dataset: dict, preserve_view: bool = False):
        self.current_dataset = dataset
        self.video_path_var.set(dataset["video_path"])
        try:
            csv_path = self.ensure_converted_dataset_csv(dataset)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.csv_path_var.set(csv_path)
        dataset["csv_path"] = csv_path
        self.load_files(preserve_view=preserve_view)
        self._load_assign_type_history(dataset.get("history_path"))
        self.load_candidates_from_assign(dataset.get("assign_path"))
        self._append_filter_candidates(self.current_dataset)
        self._append_position_spike_candidates(self.current_dataset)
        self._rebuild_candidate_label_index()
        self.refresh_candidate_tree()
        self.update_info_panel()
        # Candidate/provenance labels are loaded after the CSV. Force an immediate
        # redraw so the current frame does not wait for a navigation operation.
        self.request_frame(self.current_frame, reset_view=False, immediate=True)
        self.after_idle(self.redraw_current_frame)

    def ensure_converted_dataset_csv(self, dataset: dict) -> str:
        converted_path = str(dataset["converted_csv_path"])
        if os.path.exists(converted_path):
            center_path = _convert.center_csv_path(converted_path)
            if not center_path.exists():
                pose = _convert.load_rear_front_csv(converted_path)
                _convert.save_center_csv(_convert.rear_front_to_center(pose), center_path)
            return converted_path
        return self.convert_dataset_with_popup(dataset)

    def convert_dataset_with_popup(self, dataset: dict) -> str:
        converted_path = str(dataset["converted_csv_path"])
        obb_path = str(dataset["obb_path"])
        direction_path = str(dataset["direction_path"])

        popup = ctk.CTkToplevel(self)
        install_window_icon(popup)
        popup.title("Converting OBB + direction")
        popup.geometry("520x160")
        popup.transient(self)
        popup.grab_set()
        popup.resizable(False, False)
        status_var = tk.StringVar(value="Loading CSVs...")
        percent_var = tk.StringVar(value="0.0%")
        ctk.CTkLabel(popup, text="Converting OBB/direction to rear_front CSV").pack(anchor="w", padx=12, pady=(12, 4))
        ctk.CTkLabel(popup, textvariable=status_var).pack(anchor="w", padx=12, pady=(0, 4))
        progress = ctk.CTkProgressBar(popup, mode="determinate")
        progress.set(0.0)
        progress.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(popup, textvariable=percent_var).pack(anchor="e", padx=12)
        popup.update_idletasks()

        result: dict[str, object] = {"done": False, "error": None, "path": converted_path}

        def set_progress(done: int, total: int, text: str) -> None:
            ratio = 0.0 if total <= 0 else (done / total) * 100.0
            def update() -> None:
                progress.set(ratio / 100.0)
                status_var.set(text)
                percent_var.set(f"{ratio:.1f}%")
            popup.after(0, update)

        def worker() -> None:
            try:
                obb_df = pd.read_csv(obb_path)
                direction_df = pd.read_csv(direction_path)
                out_df = _convert.convert_obb_direction_to_rear_front(
                    obb_df=obb_df,
                    direction_df=direction_df,
                    progress_cb=set_progress,
                    round_digits=3,
                )
                Path(converted_path).parent.mkdir(parents=True, exist_ok=True)
                popup.after(0, lambda: (status_var.set("Saving CSV..."), progress.set(1.0), percent_var.set("100.0%")))
                _convert.save_rear_front_csv(out_df, converted_path, float_format="%.3f")
                _convert.save_center_csv(
                    _convert.rear_front_to_center(out_df),
                    _convert.center_csv_path(converted_path),
                )
                result["done"] = True
            except Exception as e:
                result["error"] = e
            finally:
                popup.after(0, popup.quit)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        popup.mainloop()
        try:
            popup.grab_release()
        except Exception:
            pass
        popup.destroy()
        if result["error"] is not None:
            raise RuntimeError(f"Convert failed: {result['error']}")
        if not os.path.exists(converted_path):
            raise RuntimeError("Convert failed: output CSV was not created.")
        self.set_status(f"Converted and cached: {converted_path}", auto_clear=False)
        return converted_path

    @staticmethod
    def _normalize_assign_code(value: float) -> int | None:
        if not np.isfinite(value):
            return None
        code = int(round(value))
        return code if code in ASSIGN_CODE_LABELS else None

    def _load_provenance_lookup(self, dataset: dict | None) -> None:
        self.provenance_lookup = {}
        if not dataset:
            return
        path = dataset.get("provenance_path")
        if not path or not os.path.exists(path):
            return
        try:
            df = pd.read_pickle(path) if str(path).lower().endswith(".pkl") else pd.read_csv(path, low_memory=False)
            if "position" not in df.columns:
                return
            positions = pd.to_numeric(df["position"], errors="coerce").to_numpy(dtype=float)
            tids = sorted({
                int(m.group(1))
                for col in df.columns
                for m in [re.fullmatch(r"(?:os|oc|dc|sc)(\d+)", str(col))]
                if m
            })
            for tid in tids:
                cols = {name: f"{name}{tid}" for name in ("os", "oc", "dc", "sc")}
                arrays = {
                    name: pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                    if col in df.columns else np.full(len(df), np.nan, dtype=float)
                    for name, col in cols.items()
                }
                valid_rows = np.flatnonzero(np.isfinite(positions))
                for row_idx in valid_rows.tolist():
                    meta = {name: float(arr[row_idx]) for name, arr in arrays.items() if np.isfinite(arr[row_idx])}
                    if meta:
                        self.provenance_lookup[(int(positions[row_idx]), int(tid))] = meta
        except Exception as e:
            self.set_status(f"Provenance loading failed: {e}", auto_clear=False)

    def _compose_candidate_labels(self, frame: int, tid: int, assign_code: float) -> tuple[str, str]:
        meta = self.provenance_lookup.get((int(frame), int(tid)), {})
        source = float(meta.get("os", np.nan))
        direction_fixed = bool(meta.get("dc", 0.0) >= 0.5)
        switch_fixed = bool(meta.get("sc", 0.0) >= 0.5)

        code = int(round(float(assign_code)))
        # Source-based override for provenance entries whose assign_code may not
        # reflect the actual OBB origin (e.g. correction-written frames).
        if source == OBB_SOURCE_PRE_CORRECTION and code != ATYPE_PRE_FILL:
            code = ATYPE_PRE_FILL
        elif source == OBB_SOURCE_POST_CORRECTION and code != ATYPE_POST_FILL:
            code = ATYPE_POST_FILL

        short = ASSIGN_CODE_SHORT_LABELS.get(code, f"TYPE-{code}")
        long_parts = [ASSIGN_CODE_LABELS.get(code, f"Assignment type {code}")]

        tags = short.split("|") if short else []
        has_direction_label = any(("DIR-FIX" in tag) or ("FLIP" in tag) for tag in tags)
        if direction_fixed and not has_direction_label:
            tags.append("DIR-FIX")
            long_parts.append("direction corrected")
            has_direction_label = True
        if switch_fixed and "ID-FIX" not in tags:
            tags.append("ID-FIX")
            long_parts.append("ID assignment changed by C3")
        if code == ATYPE_DIR_FIX and not has_direction_label:
            tags.append("DIR-FIX")
        return "|".join(tags), "; ".join(long_parts)

    def _rebuild_candidate_label_index(self) -> None:
        grouped: dict[tuple[int, int], list[str]] = {}
        for item in self.suspects:
            key = (int(item["frame"]), int(item["tid"]))
            label = str(item.get("label_short", "")).strip()
            if label and label not in grouped.setdefault(key, []):
                grouped[key].append(label)
        self.candidate_labels_by_key = {key: " / ".join(labels) for key, labels in grouped.items()}

    def _load_assign_type_history(self, history_path: str | None) -> None:
        self.assign_type_history = {}
        if not history_path or not os.path.exists(history_path):
            return
        with open(history_path, 'rb') as _f:
            data = pickle.load(_f)
        if not isinstance(data, dict):
            raise TypeError(f"Assignment history must be a dict: {history_path}")
        self.assign_type_history = data

    def _assign_history_entries(self, tid: int, frame: int) -> list:
        history = getattr(self, "assign_type_history", {})
        tid_history = history.get(int(tid), {})
        if not isinstance(tid_history, dict):
            return []
        entries = tid_history.get(int(frame), [])
        return list(entries) if isinstance(entries, (list, tuple)) else []

    @staticmethod
    def _history_contains_assign_code(entries: list, target_code: int) -> bool:
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            try:
                code = float(entry[1])
            except (TypeError, ValueError):
                continue
            if np.isfinite(code) and int(round(code)) == int(target_code):
                return True
        return False

    @staticmethod
    def _history_contains_source(entries: list, source: str) -> bool:
        source = str(source)
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or not entry:
                continue
            if str(entry[0]) == source:
                return True
        return False

    def load_candidates_from_assign(self, assign_path: str | None):
        self.suspects = []
        self.current_suspect_index = None
        self._load_provenance_lookup(self.current_dataset)
        # Key: (frame, tid, assign_code) -- one entry per distinct (frame, id, type) triple.
        candidate_by_key: dict[tuple[int, int, int], dict] = {}
        missing_severity: dict[tuple[int, int], float] = {}
        try:
            if assign_path and os.path.exists(assign_path):
                assign_df = pd.read_pickle(assign_path) if assign_path.lower().endswith(".pkl") else pd.read_csv(assign_path, low_memory=False)
                if "position" in assign_df.columns:
                    positions = pd.to_numeric(assign_df["position"], errors="coerce").fillna(-1).astype(int).to_numpy()
                    t_cols = sorted(
                        [c for c in assign_df.columns if str(c).startswith("t") and str(c)[1:].isdigit()],
                        key=lambda c: int(str(c)[1:]),
                    )
                    # Pre-compute MISSING run lengths per (tid, frame) for severity scoring.
                    missing_severity: dict[tuple[int, int], float] = {}
                    for _col in t_cols:
                        _tid = int(str(_col)[1:])
                        _v = pd.to_numeric(assign_df[_col], errors="coerce").to_numpy(dtype=float)
                        _miss = np.isfinite(_v) & (_v == float(ATYPE_MISSING))
                        _i = 0
                        while _i < len(_miss):
                            if _miss[_i]:
                                _j = _i + 1
                                while _j < len(_miss) and _miss[_j]:
                                    _j += 1
                                _run = float(_j - _i)
                                for _k in range(_i, _j):
                                    if _k < len(positions):
                                        missing_severity[(_tid, int(positions[_k]))] = _run
                                _i = _j
                            else:
                                _i += 1
                    for col in t_cols:
                        tid = int(str(col)[1:])
                        vals = pd.to_numeric(assign_df[col], errors="coerce").to_numpy(dtype=float)
                        for row_idx in np.flatnonzero(np.isfinite(vals)).tolist():
                            code = self._normalize_assign_code(float(vals[row_idx]))
                            if code is None or code in (ATYPE_INIT, ATYPE_S1):
                                continue
                            frame = int(positions[row_idx])
                            history = self._assign_history_entries(tid, frame)
                            if (
                                code == ATYPE_DIR_FIX
                                and self._history_contains_source(
                                    history, "correction_direction_segment_flip",
                                )
                            ):
                                code = ATYPE_DIR_FLIP
                            pkey = (frame, tid, code)
                            if pkey not in candidate_by_key:
                                short, long_label = self._compose_candidate_labels(frame, tid, code)
                                candidate_by_key[pkey] = {
                                    "frame": frame,
                                    "tid": tid,
                                    "assign_code": code,
                                    "label": long_label,
                                    "label_short": short,
                                    "_severity": missing_severity.get((tid, frame), 0.0),
                                }

            # Provenance creates separate candidate entries for OBB-source and
            # direction-fix events, independently of the assign_type_buf code.
            for (frame, tid), meta in self.provenance_lookup.items():
                source = float(meta.get("os", np.nan))
                direction_fixed = bool(meta.get("dc", 0.0) >= 0.5)
                if source == OBB_SOURCE_PRE_CORRECTION:
                    virtual_code = ATYPE_PRE_FILL
                elif source == OBB_SOURCE_POST_CORRECTION:
                    virtual_code = ATYPE_POST_FILL
                elif direction_fixed:
                    history = self._assign_history_entries(int(tid), int(frame))
                    if (
                        self._history_contains_assign_code(history, ATYPE_DIR_FLIP)
                        or self._history_contains_source(
                            history, "correction_direction_segment_flip",
                        )
                    ):
                        virtual_code = ATYPE_DIR_FLIP
                    else:
                        virtual_code = ATYPE_DIR_FIX
                else:
                    continue
                pkey = (int(frame), int(tid), int(virtual_code))
                if pkey not in candidate_by_key:
                    short, long_label = self._compose_candidate_labels(frame, tid, virtual_code)
                    candidate_by_key[pkey] = {
                        "frame": int(frame),
                        "tid": int(tid),
                        "assign_code": virtual_code,
                        "label": long_label,
                        "label_short": short,
                        "_severity": missing_severity.get((int(tid), int(frame)), 0.0),
                    }

            self.suspects = list(candidate_by_key.values())
            self.suspects.sort(key=lambda d: (d["frame"], d["tid"], d["assign_code"]))
            self._append_embedding_candidates(dataset=self.current_dataset)
            self._append_embedding_history_candidates()
            self._rebuild_candidate_label_index()
        except Exception as e:
            self.set_status(f"Candidate loading failed: {e}", auto_clear=False)

    def _append_embedding_history_candidates(self) -> None:
        """Add embedding-history frames not represented by correction-log candidates."""
        history = getattr(self, "assign_type_history", {})
        if not isinstance(history, dict) or not history:
            return

        # Identity outcomes are represented by their Stage 3 code; do not add a
        # duplicate history-recovery row for an already covered (frame, tid, code) triple.
        embedding_codes = {ATYPE_EMBED_ABSTAIN, ATYPE_EMBED_IDENTITY, ATYPE_EMBED_RELABEL}
        covered = {
            (int(item["frame"]), int(item["tid"]), int(item["assign_code"]))
            for item in self.suspects
            if int(item["assign_code"]) in embedding_codes
        }

        for tid, frame_map in history.items():
            if not isinstance(frame_map, dict):
                continue
            for frame, entries in frame_map.items():
                entry_list = list(entries) if isinstance(entries, (list, tuple)) else []
                history_codes = {
                    code for code in embedding_codes
                    if self._history_contains_assign_code(entry_list, code)
                }
                if not history_codes:
                    continue
                # Most-significant outcome wins if more than one is present
                # in history for the same (tid, frame): an actual relabel
                # outranks a confirmed identity, which outranks an abstain.
                if ATYPE_EMBED_RELABEL in history_codes:
                    history_code = ATYPE_EMBED_RELABEL
                elif ATYPE_EMBED_IDENTITY in history_codes:
                    history_code = ATYPE_EMBED_IDENTITY
                else:
                    history_code = ATYPE_EMBED_ABSTAIN
                triple = (int(frame), int(tid), int(history_code))
                if triple in covered:
                    continue
                short, long_label = self._compose_candidate_labels(frame, tid, history_code)
                self.suspects.append({
                    "frame": frame,
                    "tid": tid,
                    "assign_code": history_code,
                    "label": f"{long_label}; recovered from assignment history",
                    "label_short": short,
                    "_severity": 0.0,
                })
                covered.add(triple)

        self.suspects.sort(key=lambda d: (d["frame"], d["tid"], d["assign_code"]))

    def _append_embedding_candidates(self, dataset: dict | None) -> None:
        if not dataset or dataset.get("family") != "id_resolved":
            return

        artifact_suffix = tracking_artifact_suffix_for_family(dataset["family"])
        log_path = artifact_path(
            dataset["out_dir"], "corrections_log.json", artifact_suffix,
        )
        if not os.path.exists(log_path):
            return
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)

        covered = {
            (int(s["frame"]), int(s["tid"]), int(s["assign_code"]))
            for s in self.suspects
        }
        for entry in log:
            if entry.get("type") != "embedding_episode":
                continue
            action = entry.get("action")
            frame_start = int(entry["frame_start"])
            frame_end = int(entry["frame_end"])
            if action == "relabel":
                code = ATYPE_EMBED_RELABEL
                ids = entry.get("swapped_ids", [])
                label = f"C3 embedding episode relabelled (frames {frame_start}-{frame_end})"
            elif action == "identity":
                code = ATYPE_EMBED_IDENTITY
                ids = entry.get("ids", [])
                label = f"C3 embedding episode confirmed identity (frames {frame_start}-{frame_end})"
            else:
                code = ATYPE_EMBED_ABSTAIN
                ids = entry.get("ids", [])
                label = f"C3 embedding episode abstained (frames {frame_start}-{frame_end})"
            short = ASSIGN_CODE_SHORT_LABELS[code]

            for frame in entry["review_frames"]:
                for tid in ids:
                    key = (int(frame), int(tid), int(code))
                    if key in covered:
                        continue
                    history = self._assign_history_entries(int(tid), int(frame))
                    if not self._history_contains_assign_code(history, code):
                        continue
                    self.suspects.append({
                        "frame": int(frame),
                        "tid": int(tid),
                        "assign_code": code,
                        "label": label,
                        "label_short": short,
                        "_severity": 0.0,
                    })
                    covered.add(key)

        self.suspects.sort(key=lambda d: (d["frame"], d["tid"], d["assign_code"]))
        self._rebuild_candidate_label_index()

    def _append_filter_candidates(self, dataset: dict | None):
        if not dataset:
            return
        out_dir = dataset.get("out_dir", "")
        if not out_dir:
            return
        filter_dir = os.path.join(out_dir, "filter_info")
        if not os.path.isdir(filter_dir):
            return
        for code, csv_name in _FILTER_STAGE_INFO.items():
            csv_path = os.path.join(filter_dir, csv_name)
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                if "frame" not in df.columns or "track_id" not in df.columns:
                    continue
                frames = pd.to_numeric(df["frame"], errors="coerce")
                tids = pd.to_numeric(df["track_id"], errors="coerce")
                valid = frames.notna() & tids.notna()
                for f, t in zip(frames[valid].astype(int).tolist(), tids[valid].astype(int).tolist()):
                    self.suspects.append({
                        "frame": f,
                        "tid": t,
                        "assign_code": code,
                        "label": ASSIGN_CODE_LABELS.get(code, f"filter {code}"),
                        "label_short": ASSIGN_CODE_SHORT_LABELS.get(code, f"GUI-{code}"),
                        "_severity": 0.0,
                    })
            except Exception:
                continue
        self.suspects.sort(key=lambda d: (d["frame"], d["tid"], d["assign_code"]))
        self._rebuild_candidate_label_index()

    def _append_position_spike_candidates(self, dataset: dict | None) -> None:
        """Add one candidate per (frame, tid) for frames corrected by position-spike fix."""
        if not dataset:
            return
        family = str(dataset.get("family", ""))
        if family not in ("id_resolved", "final"):
            return
        artifact_suffix = tracking_artifact_suffix_for_family(family)
        log_path = artifact_path(
            dataset["out_dir"], "corrections_log.json", artifact_suffix,
        )
        if not os.path.exists(log_path):
            return
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            return
        covered = {
            (int(s["frame"]), int(s["tid"]), int(s["assign_code"]))
            for s in self.suspects
            if int(s["assign_code"]) == ATYPE_POS_FIX
        }
        short = ASSIGN_CODE_SHORT_LABELS.get(ATYPE_POS_FIX, "C4-POS-FIX")
        base_label = ASSIGN_CODE_LABELS.get(ATYPE_POS_FIX, "C4 - position spike corrected by OBB interpolation")
        for entry in log:
            if entry.get("type") != "position_spike":
                continue
            ids = entry.get("ids", [])
            frame = int(entry.get("frame", -1))
            if frame < 0 or not ids:
                continue
            phase = entry.get("phase", "")
            for tid in ids:
                triple = (frame, int(tid), int(ATYPE_POS_FIX))
                if triple not in covered:
                    self.suspects.append({
                        "frame": frame,
                        "tid": int(tid),
                        "assign_code": ATYPE_POS_FIX,
                        "label": f"{base_label} [{phase}]",
                        "label_short": short,
                        "_severity": 0.0,
                    })
                    covered.add(triple)
        self.suspects.sort(key=lambda d: (d["frame"], d["tid"], d["assign_code"]))
        self._rebuild_candidate_label_index()

    def _activate_dataset_input(self) -> None:
        """Switch to direct Video/CSV input without changing config files."""
        if self.mode_var.get() == "dataset":
            return
        self.mode_var.set("dataset")
        self._refresh_mode_ui()

    def _handle_dropped_file(self, path: str) -> None:
        """Route one canvas drop to the direct video or tracking-file input."""
        self._activate_dataset_input()
        suffix = Path(path).suffix.lower()
        if suffix in VIDEO_DROP_SUFFIXES:
            self.video_path_var.set(path)
            self.try_load_files()
            if self.reader is None:
                self.set_status("Video selected. Drop or select a tracking CSV to load the preview.", auto_clear=False)
            return
        if suffix in TRACKING_DROP_SUFFIXES:
            self._load_tracking_input(path)

    def browse_video(self):
        self._activate_dataset_input()
        path = filedialog.askopenfilename(title="Select video", filetypes=VIDEO_EXTS)
        if path:
            self.video_path_var.set(path)
            self.try_load_files()

    def browse_csv(self):
        self._activate_dataset_input()
        path = filedialog.askopenfilename(
            title="Select tracking file (CSV / H5) or converted CSV",
            filetypes=[
                ("Tracking files", "*.csv *.h5 *.hdf5"),
                ("CSV", "*.csv"),
                ("HDF5", "*.h5 *.hdf5"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_tracking_input(path)

    def _load_tracking_input(self, path: str) -> None:
        """Set the source path, convert it when needed, and load its contents."""
        self.csv_path_var.set(path)
        try:
            info = _convert.inspect_tracking_file(path)
        except Exception as e:
            messagebox.showerror("Conversion error", str(e))
            return
        if info.requires_keypoint_selection:
            def on_converted(csv_path: str) -> None:
                self.csv_path_var.set(csv_path)
                self.try_load_files()
            PoseConvertDialog(self, initial_path=path, on_converted=on_converted)
            return
        try:
            outputs = _convert.convert_tracking_outputs(path)
            if outputs.rear_front is None:
                raise ValueError(
                    "This file contains center coordinates only; "
                    "direction refinement requires front/rear points."
                )
            self.csv_path_var.set(str(outputs.rear_front))
            self.try_load_files()
        except Exception as e:
            messagebox.showerror("Conversion error", str(e))

    def try_load_files(self):
        video_path = self.video_path_var.get().strip()
        csv_path = self.csv_path_var.get().strip()
        if video_path and csv_path and os.path.exists(video_path) and os.path.exists(csv_path):
            self.load_files()

    def load_files(self, preserve_view: bool = False):
        video_path = self.video_path_var.get().strip()
        csv_path = self.csv_path_var.get().strip()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Video file not found.")
            return
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Error", "CSV file not found.")
            return

        try:
            df = _convert.load_wide_for_gui(csv_path, standard_front_first=True)
            self._validate_uma_csv(df)
            reader = VideoFrameReader(video_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        _ic = self._idx_col(df)
        positions = pd.to_numeric(df[_ic], errors="raise").astype(int).to_numpy()
        if reader.frame_count > 0 and (positions.min(initial=0) < 0 or positions.max(initial=0) >= reader.frame_count):
            messagebox.showerror("Error", f"CSV {_ic} is out of the video's frame range. video frames={reader.frame_count}, CSV {_ic} max={positions.max(initial=0)}")
            reader.close()
            return

        saved_frame = self.current_frame if preserve_view else 0

        self._invalidate_decode_requests()
        if self.reader is not None:
            self.reader.close()
        if self.prefetch_after_id is not None:
            self.after_cancel(self.prefetch_after_id)
            self.prefetch_after_id = None
        if self.frame_render_job is not None:
            self.after_cancel(self.frame_render_job)
            self.frame_render_job = None
        self._stop_key_play()
        self.stop_playback(update_button=False)

        self.reader = reader
        self.frame_cache_rgb = None
        self._last_displayed_frame = -1
        try:
            self.canvas.delete("all")
        except Exception:
            pass
        self.df = df.copy()
        self.frame_count = len(self.df)
        self.num_ids = (len(self.df.columns) - 1) // 4
        _tids = np.arange(self.num_ids)
        self._front_x_idx = (1 + 4 * _tids).astype(np.intp)
        self._front_y_idx = (2 + 4 * _tids).astype(np.intp)
        self._rear_x_idx  = (3 + 4 * _tids).astype(np.intp)
        self._rear_y_idx  = (4 + 4 * _tids).astype(np.intp)
        self.df_np = self.df.to_numpy(dtype=np.float64)
        self.id_palette = make_id_palette(self.num_ids, color_space="rgb")
        self.current_frame = 0
        self.requested_frame = 0
        self.current_id = 0
        self.last_save_path = None
        self.save_count = 0
        self.edited_keys.clear()
        self.selected_tids.clear()
        self._close_swap_popup()
        if not preserve_view:
            self._reset_view_state()
        self._configure_controls()
        target_frame = min(saved_frame, self.frame_count - 1) if preserve_view else 0
        self._set_frame(target_frame, reset_view=not preserve_view)
        self._set_id(0)
        self.update_info_panel()
        # Re-issue the current-frame request after Tk has applied the new data
        # arrays. This fixes stale overlays after CSV replacement at the same frame.
        self.after_idle(lambda f=target_frame, rv=not preserve_view: self.request_frame(f, reset_view=rv, immediate=True))
        self.set_status(f"Loaded: {os.path.basename(video_path)} / IDs={self.num_ids} / frames={self.frame_count}")
        _start_maximized(self)

    @staticmethod
    def _idx_col(df: pd.DataFrame) -> str:
        """Return the name of the frame/position index column, accepting either."""
        if "frame" in df.columns:
            return "frame"
        if "position" in df.columns:
            return "position"
        raise RuntimeError("CSV has no frame / position column.")

    def _validate_uma_csv(self, df: pd.DataFrame):
        idx = self._idx_col(df)
        n = len(df.columns) - 1
        if n <= 0 or n % 4 != 0:
            raise RuntimeError(f"Cannot interpret format. The number of columns other than {idx} is not a multiple of 4.")
        num_ids = n // 4
        expected = [idx]
        for i in range(num_ids * 2):
            expected.extend([f"x{i}", f"y{i}"])
        if list(df.columns) != expected:
            raise RuntimeError("Column names do not match expectations. Expected format is x0,y0,x1,y1,...")
        pos = pd.to_numeric(df[idx], errors="raise").astype(int).to_numpy()
        if not np.array_equal(pos, np.arange(len(df), dtype=int)):
            raise RuntimeError(f"The {idx} column is not a sequential range starting from 0.")

    def _configure_controls(self):
        self.frame_spin.configure(from_=0, to=max(0, self.frame_count - 1))
        self.id_spin.configure(from_=0, to=max(0, self.num_ids - 1), wrap=True)
        self.frame_scale.configure(from_=0, to=max(0, self.frame_count - 1))

    def _reset_view_state(self):
        self.base_fit_scale = 1.0
        self.zoom_scale = 1.0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.view_initialized = False
        self.current_image_shape = None

    @staticmethod
    def _drain_queue(q: queue.Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            return

    def _invalidate_decode_requests(self) -> None:
        """Invalidate queued/completed decodes without waiting for codec threads."""
        self._decode_gen += 1
        self._drain_queue(self._decode_request_queue)
        self._drain_queue(self._frame_queue)

    def _enqueue_decode_request(
        self,
        generation: int,
        reader: VideoFrameReader,
        frame_idx: int,
        reset_view: bool,
    ) -> None:
        # Queue size is one: rapid navigation always replaces the pending target
        # instead of accumulating obsolete decoding work.
        self._drain_queue(self._decode_request_queue)
        try:
            self._decode_request_queue.put_nowait(
                (generation, reader, int(frame_idx), bool(reset_view))
            )
        except queue.Full:
            # Another request won the race; discard it and keep the newest one.
            self._drain_queue(self._decode_request_queue)
            try:
                self._decode_request_queue.put_nowait(
                    (generation, reader, int(frame_idx), bool(reset_view))
                )
            except queue.Full:
                pass

    def _decode_worker_loop(self) -> None:
        while not self._decode_stop.is_set():
            try:
                item = self._decode_request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return

            # Coalesce requests that arrived before decoding started.
            try:
                while True:
                    newer = self._decode_request_queue.get_nowait()
                    if newer is None:
                        return
                    item = newer
            except queue.Empty:
                pass

            generation, reader, frame_idx, reset_view = item
            frame = None
            error: Exception | None = None
            try:
                frame = reader.read(frame_idx)
            except Exception as exc:
                error = exc
            self._frame_queue.put(
                (generation, reader, frame_idx, reset_view, frame, error)
            )

    def _finish_frame_render(
        self,
        generation: int,
        reader: VideoFrameReader,
        frame_idx: int,
        reset_view: bool,
        frame: np.ndarray,
    ) -> None:
        if (
            generation != self._decode_gen
            or reader is not self.reader
            or frame_idx != self.requested_frame
        ):
            return
        self.current_frame = frame_idx
        self.requested_frame = frame_idx
        self.frame_var.set(frame_idx)
        self.frame_scale.set(frame_idx)
        self.frame_cache_rgb = frame
        self._draw_canvas(frame, preserve_view=not reset_view)
        self.update_info_panel()
        self._last_displayed_frame = frame_idx
        self._queue_prefetch(frame_idx)

    def _handle_frame_decode_error(
        self,
        generation: int,
        reader: VideoFrameReader,
        frame_idx: int,
        error: Exception,
    ) -> None:
        if generation != self._decode_gen or reader is not self.reader:
            return
        self._stop_key_play()
        self.stop_playback()
        fallback = self._last_displayed_frame
        if fallback < 0:
            fallback = max(0, min(self.frame_count - 1, frame_idx))
        self.current_frame = fallback
        self.requested_frame = fallback
        self.frame_var.set(fallback)
        self.frame_scale.set(fallback)
        self.set_status(f"Frame decode failed at {frame_idx}: {error}", auto_clear=False)

    def _has_pending_frame_request(self) -> bool:
        return (
            self.frame_render_job is not None
            or self.requested_frame != self.current_frame
            or self.current_frame != self._last_displayed_frame
        )

    def _navigation_base_frame(self) -> int:
        return self.requested_frame if self._has_pending_frame_request() else self.current_frame

    def _cancel_pending_frame_request(self) -> None:
        if self.frame_render_job is not None:
            try:
                self.after_cancel(self.frame_render_job)
            except Exception:
                pass
            self.frame_render_job = None
        if not self._has_pending_frame_request():
            return
        self._invalidate_decode_requests()
        fallback = self.current_frame
        if self._last_displayed_frame >= 0:
            fallback = self._last_displayed_frame
        fallback = max(0, min(max(0, self.frame_count - 1), int(fallback)))
        self.current_frame = fallback
        self.requested_frame = fallback
        self.frame_var.set(fallback)
        self.frame_scale.set(fallback)

    def _pump_frame_queue(self):
        if self._closing:
            return
        try:
            while True:
                generation, reader, frame_idx, reset_view, frame, error = self._frame_queue.get_nowait()
                if generation != self._decode_gen or reader is not self.reader:
                    continue
                if error is not None:
                    self._handle_frame_decode_error(
                        generation, reader, frame_idx, error
                    )
                elif frame is not None:
                    self._finish_frame_render(
                        generation, reader, frame_idx, reset_view, frame
                    )
        except queue.Empty:
            pass
        if not self._closing:
            self.after(8, self._pump_frame_queue)

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

    def on_close(self):
        if self.edited_keys and self.df is not None:
            answer = messagebox.askyesnocancel(
                "Unsaved changes",
                "You have unsaved changes. Save before closing?",
            )
            if answer is None:
                return
            if answer:
                self.save_csv_as()

        self._closing = True
        self._stop_key_play()
        self.stop_playback(update_button=False)
        self._invalidate_decode_requests()
        self._decode_stop.set()
        try:
            self._decode_request_queue.put_nowait(None)
        except queue.Full:
            self._drain_queue(self._decode_request_queue)
            try:
                self._decode_request_queue.put_nowait(None)
            except queue.Full:
                pass

        for job in (
            self.frame_render_job,
            self.prefetch_after_id,
            self.status_clear_job,
            self._refresh_tree_job,
            self.pending_candidate_select_job,
        ):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        self._close_swap_popup()
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.destroy()

    def on_window_configure(self, _event=None):
        if self.frame_cache_rgb is not None:
            self._draw_canvas(self.frame_cache_rgb, preserve_view=True)

    def on_ctrl_s(self, _event=None):
        self.save_csv_quick()
        return "break"

    def on_frame_entry_commit(self, _event=None):
        if self.df is None:
            return None
        try:
            frame_idx = int(self.frame_spin.get())
        except Exception:
            self.frame_var.set(self.current_frame)
            return "break"
        self.request_frame(frame_idx, reset_view=False, immediate=True)
        return "break"

    def on_frame_spin(self):
        try:
            frame_idx = int(self.frame_spin.get())
        except Exception:
            return
        self.request_frame(frame_idx, reset_view=False, immediate=True)

    def on_id_spin(self):
        try:
            tid = int(self.id_spin.get())
        except Exception:
            self.id_var.set(self.current_id)
            return
        self._select_single_tid(tid)

    def on_id_entry_commit(self, _event=None):
        if self.df is None:
            return None
        try:
            tid = int(self.id_spin.get())
        except Exception:
            self.id_var.set(self.current_id)
            return "break"
        self._select_single_tid(tid)
        return "break"

    def on_frame_scale(self, value):
        if self.df is None:
            return
        self.request_frame(int(round(float(value))), reset_view=False, immediate=True)

    def request_frame(self, frame_idx: int, reset_view: bool = False, immediate: bool = False):
        if self.reader is None or self.df is None:
            return
        frame_idx = max(0, min(self.frame_count - 1, int(frame_idx)))
        self.requested_frame = frame_idx
        if immediate:
            if self.frame_render_job is not None:
                self.after_cancel(self.frame_render_job)
                self.frame_render_job = None
            self._apply_requested_frame(reset_view=reset_view)
            return
        if self.frame_render_job is None:
            self.frame_render_job = self.after(self.frame_render_interval_ms, lambda rv=reset_view: self._apply_requested_frame(reset_view=rv))

    def _apply_requested_frame(self, reset_view: bool = False):
        self.frame_render_job = None
        self._set_frame(self.requested_frame, reset_view=reset_view)

    def _set_frame(self, frame_idx: int, reset_view: bool = False):
        if self.reader is None or self.df is None:
            return
        frame_idx = max(0, min(self.frame_count - 1, int(frame_idx)))
        self.requested_frame = frame_idx

        if self.drag_mode is not None and self.drag_tid is not None and self.drag_last_canvas is not None and self.left_button_down:
            if self.drag_mode == "whole":
                held_state = self._drag_state_from_canvas(self.drag_last_canvas)
                if held_state is not None:
                    if frame_idx != self.current_frame:
                        self._set_arrow_state(self.current_frame, self.drag_tid, held_state)
                    self._set_arrow_state(frame_idx, self.drag_tid, held_state)
                    self.drag_start_front = held_state.front.copy()
                    self.drag_start_rear = held_state.rear.copy()
                    self.drag_anchor_img = self.canvas_to_img(self.drag_last_canvas)
            else:
                state = self._get_arrow_state(frame_idx, self.drag_tid)
                self.drag_start_front = state.front.copy()
                self.drag_start_rear = state.rear.copy()
                self.drag_anchor_img = self.canvas_to_img(self.drag_last_canvas)
                self._apply_drag_from_canvas(self.drag_last_canvas, frame_idx=frame_idx)

        self._decode_gen += 1
        generation = self._decode_gen
        reader = self.reader

        # Cache lookup is non-blocking.  The Tk thread never calls a codec read,
        # even if another thread evicts the frame at the same instant.
        cached = reader.get_cached(frame_idx)
        if cached is not None:
            self._drain_queue(self._decode_request_queue)
            self._finish_frame_render(
                generation, reader, frame_idx, reset_view, cached
            )
            return

        self._enqueue_decode_request(
            generation, reader, frame_idx, reset_view
        )

    def _set_id(self, tid: int):
        if self.df is None or self.num_ids <= 0:
            return
        tid = int(tid) % self.num_ids
        self.current_id = tid
        self.id_var.set(tid)
        self.update_info_panel()
        self.redraw_current_frame()

    def _select_single_tid(self, tid: int):
        if self.df is None or self.num_ids <= 0:
            return
        tid = int(tid) % self.num_ids
        self.selected_tids.clear()
        self.selected_tids.append(tid)
        self._close_swap_popup()
        self._set_id(tid)

    def step_frame(self, delta: int):
        if self.df is None:
            return
        base_frame = self._navigation_base_frame()
        self.request_frame(base_frame + int(delta), reset_view=False, immediate=False)

    def go_first_frame(self):
        self.request_frame(0, reset_view=False, immediate=True)

    def go_last_frame(self):
        if self.df is not None:
            self.request_frame(self.frame_count - 1, reset_view=False, immediate=True)

    def toggle_playback(self):
        if self.playback_active:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.df is None or self.frame_count <= 0:
            return
        if self.playback_active:
            return
        if self.current_frame >= self.frame_count - 1:
            return
        self.playback_active = True
        self._update_play_button()
        self._step_frame_for_playback(1)
        if self.current_frame < self.frame_count - 1:
            self._schedule_playback(initial=True)
        else:
            self.stop_playback()

    def stop_playback(self, update_button: bool = True):
        was_active = self.playback_active
        self.playback_active = False
        if self.playback_job is not None:
            self.after_cancel(self.playback_job)
            self.playback_job = None
        if was_active:
            self._cancel_pending_frame_request()
        if update_button:
            self._update_play_button()

    def _update_play_button(self):
        if self.play_button is not None:
            self.play_button.configure(text="Stop" if self.playback_active else "Play")

    def _schedule_playback(self, initial: bool = False, delay_ms: int | None = None):
        if not self.playback_active:
            return
        if self.playback_job is not None:
            self.after_cancel(self.playback_job)
        delay = int(delay_ms) if delay_ms is not None else (220 if initial else self.playback_interval_ms)
        self.playback_job = self.after(delay, self._playback_tick)

    def _playback_tick(self):
        self.playback_job = None
        if not self.playback_active:
            return
        if self.df is None or self.frame_count <= 0:
            self.stop_playback()
            return
        if self.current_frame >= self.frame_count - 1:
            self.stop_playback()
            return
        if self._has_pending_frame_request():
            self._schedule_playback(initial=False, delay_ms=8)
            return
        self._step_frame_for_playback(1)
        if self.playback_active and self.current_frame < self.frame_count - 1:
            self._schedule_playback(initial=False)
        else:
            self.stop_playback()

    def _step_frame_for_playback(self, delta: int):
        if self.df is None:
            return
        base_frame = self._navigation_base_frame()
        self.request_frame(base_frame + int(delta), reset_view=False, immediate=True)

    def step_id(self, delta: int):
        if self.df is None:
            return
        self._select_single_tid(self.current_id + int(delta))

    def _stop_key_play(self):
        if self._key_play_job is not None:
            try:
                self.after_cancel(self._key_play_job)
            except Exception:
                pass
            self._key_play_job = None
        self._key_play_delta = 0
        self._key_play_wait_count = 0

    def _start_key_play(self, delta: int):
        dragging = self.left_button_down and self.drag_mode is not None
        if dragging:
            # During drag: single-step per key event (no auto-repeat tick),
            # matching the old behaviour where repeat was suppressed while dragging.
            self._stop_key_play()
            self._key_play_delta = delta
            self._key_play_step()
            self._key_play_delta = 0  # reset so next OS key event fires again
            return
        if self._key_play_delta == delta:
            return  # already playing in this direction
        self._stop_key_play()
        self._key_play_delta = delta
        self._key_play_step()
        self._key_play_job = self.after(220, self._key_play_tick)

    def _key_play_tick(self):
        self._key_play_job = None
        if self._key_play_delta == 0:
            self._key_play_wait_count = 0
            return
        # Back-pressure: wait for the current frame to render before advancing.
        if self._has_pending_frame_request():
            self._key_play_wait_count += 1
            self._key_play_job = self.after(8, self._key_play_tick)
            return
        self._key_play_wait_count = 0
        self._key_play_step()
        self._key_play_job = self.after(self.playback_interval_ms, self._key_play_tick)

    def _key_play_step(self):
        if self.df is None:
            return
        if self.left_button_down and self.drag_mode is not None and self.drag_tid is not None and self.drag_last_canvas is not None:
            self._apply_drag_from_canvas(self.drag_last_canvas)
        base = self._navigation_base_frame()
        self.request_frame(base + self._key_play_delta, reset_view=False, immediate=True)

    def on_space_press(self, event):
        w = self.focus_get()
        cls = str(w.winfo_class()) if w is not None else ""
        if cls in {"Entry", "TEntry", "Text"}:
            return None
        self.toggle_playback()
        return "break"

    def on_key_press(self, event):
        keysym = str(event.keysym)
        if keysym not in ("Left", "Right", "Up", "Down"):
            return None
        if self._event_from_spinbox():
            return None
        if self.playback_active and keysym in {"Left", "Right"}:
            self.stop_playback()
            return "break"
        if keysym == "Left":
            self._start_key_play(-1)
        elif keysym == "Right":
            self._start_key_play(1)
        elif keysym == "Up":
            self.step_id(-1)
        elif keysym == "Down":
            self.step_id(1)
        return "break"

    def on_key_release(self, event):
        keysym = str(event.keysym)
        if keysym in {"Left", "Right"}:
            self._stop_key_play()
        return "break"

    def on_focus_out(self, _event=None):
        self._stop_key_play()
    def _event_from_spinbox(self) -> bool:
        w = self.focus_get()
        if w is None:
            return False
        cls = str(w.winfo_class())
        return cls in {"Entry", "TEntry", "Spinbox", "Text"}

    def redraw_current_frame(self):
        if self.frame_cache_rgb is not None:
            self._draw_canvas(self.frame_cache_rgb, preserve_view=True)

    def fit_window_scale(self):
        if self.frame_cache_rgb is None:
            return
        self._fit_view(self.frame_cache_rgb)
        self.redraw_current_frame()

    def save_canvas_screenshot(self):
        if self.frame_cache_rgb is None:
            messagebox.showwarning("Warning", "No frame is loaded.")
            return
        if not self.view_initialized:
            self._fit_view(self.frame_cache_rgb)

        display = self._render_display_frame(self.frame_cache_rgb)
        img_h, img_w = display.shape[:2]
        can_w = max(1, self.canvas.winfo_width())
        can_h = max(1, self.canvas.winfo_height())
        scale = max(1e-9, float(self.scale))

        src_x0 = max(0.0, -self.offset_x / scale)
        src_y0 = max(0.0, -self.offset_y / scale)
        src_x1 = min(float(img_w), (can_w - self.offset_x) / scale)
        src_y1 = min(float(img_h), (can_h - self.offset_y) / scale)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            messagebox.showwarning("Warning", "No image area is visible on the canvas.")
            return

        ix0, iy0 = int(src_x0), int(src_y0)
        ix1 = min(img_w, int(np.ceil(src_x1)))
        iy1 = min(img_h, int(np.ceil(src_y1)))
        crop = display[iy0:iy1, ix0:ix1]
        dst_w = max(1, int(round((ix1 - ix0) * scale)))
        dst_h = max(1, int(round((iy1 - iy0) * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        out = cv2.resize(crop, (dst_w, dst_h), interpolation=interp)

        path = filedialog.asksaveasfilename(
            title="Save screenshot",
            defaultextension=".png",
            initialfile=f"refinement_frame_{int(self.current_frame):06d}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return
        Image.fromarray(out).save(path)
        self.set_status(f"Screenshot saved: {path}")

    def _front_col(self, tid: int) -> tuple[str, str]:
        return f"x{tid * 2}", f"y{tid * 2}"

    def _rear_col(self, tid: int) -> tuple[str, str]:
        return f"x{tid * 2 + 1}", f"y{tid * 2 + 1}"

    def _get_arrow_state(self, frame_idx: int, tid: int) -> ArrowState:
        row = self.df_np[int(frame_idx)]
        front = np.array([row[self._front_x_idx[tid]], row[self._front_y_idx[tid]]], dtype=float)
        rear  = np.array([row[self._rear_x_idx[tid]],  row[self._rear_y_idx[tid]]],  dtype=float)
        if self.reverse_direction_var.get():
            return ArrowState(front=rear, rear=front)
        return ArrowState(front=front, rear=rear)

    def _set_arrow_state(self, frame_idx: int, tid: int, state: ArrowState):
        fx, fy = self._front_col(tid)
        rx, ry = self._rear_col(tid)
        if self.reverse_direction_var.get():
            front_to_store = state.rear
            rear_to_store = state.front
        else:
            front_to_store = state.front
            rear_to_store = state.rear
        fi = int(frame_idx)
        fx_v, fy_v = float(front_to_store[0]), float(front_to_store[1])
        rx_v, ry_v = float(rear_to_store[0]),  float(rear_to_store[1])
        self.df.at[fi, fx] = fx_v
        self.df.at[fi, fy] = fy_v
        self.df.at[fi, rx] = rx_v
        self.df.at[fi, ry] = ry_v
        self.df_np[fi, self._front_x_idx[tid]] = fx_v
        self.df_np[fi, self._front_y_idx[tid]] = fy_v
        self.df_np[fi, self._rear_x_idx[tid]]  = rx_v
        self.df_np[fi, self._rear_y_idx[tid]]  = ry_v
        self.current_id = int(tid)
        self.id_var.set(int(tid))
        self.edited_keys.add((fi, int(tid)))
        self.update_info_panel()
        self._schedule_refresh_candidate_tree()

    def _swap_ids_from_current_frame(self, tid_a: int, tid_b: int):
        start_frame = int(self.current_frame)
        end_frame = int(self.frame_count)
        if start_frame >= end_frame:
            return 0
        cols_a = [*self._front_col(tid_a), *self._rear_col(tid_a)]
        cols_b = [*self._front_col(tid_b), *self._rear_col(tid_b)]
        values_a = self.df.loc[start_frame:end_frame - 1, cols_a].to_numpy(copy=True)
        values_b = self.df.loc[start_frame:end_frame - 1, cols_b].to_numpy(copy=True)
        self.df.loc[start_frame:end_frame - 1, cols_a] = values_b
        self.df.loc[start_frame:end_frame - 1, cols_b] = values_a
        s, e = start_frame, end_frame
        for ci_a, ci_b in zip(
            [int(self._front_x_idx[tid_a]), int(self._front_y_idx[tid_a]),
             int(self._rear_x_idx[tid_a]),  int(self._rear_y_idx[tid_a])],
            [int(self._front_x_idx[tid_b]), int(self._front_y_idx[tid_b]),
             int(self._rear_x_idx[tid_b]),  int(self._rear_y_idx[tid_b])],
        ):
            tmp = self.df_np[s:e, ci_a].copy()
            self.df_np[s:e, ci_a] = self.df_np[s:e, ci_b]
            self.df_np[s:e, ci_b] = tmp
        self.current_id = int(tid_b)
        self.id_var.set(int(tid_b))
        for f in range(start_frame, end_frame):
            self.edited_keys.add((int(f), int(tid_a)))
            self.edited_keys.add((int(f), int(tid_b)))
        self.update_info_panel()
        self.redraw_current_frame()
        self.refresh_candidate_tree()
        return end_frame - start_frame

    def _swap_arrow_direction_range(self, tid: int, start_frame: int, end_frame: int):
        ax, ay = self._front_col(tid)
        bx, by = self._rear_col(tid)
        start_frame = max(0, int(start_frame))
        end_frame = min(self.frame_count, int(end_frame))
        if start_frame >= end_frame:
            return 0
        front_xy = self.df.loc[start_frame:end_frame - 1, [ax, ay]].to_numpy(copy=True)
        rear_xy = self.df.loc[start_frame:end_frame - 1, [bx, by]].to_numpy(copy=True)
        self.df.loc[start_frame:end_frame - 1, [ax, ay]] = rear_xy
        self.df.loc[start_frame:end_frame - 1, [bx, by]] = front_xy
        s, e = start_frame, end_frame
        fx_i, fy_i = int(self._front_x_idx[tid]), int(self._front_y_idx[tid])
        rx_i, ry_i = int(self._rear_x_idx[tid]),  int(self._rear_y_idx[tid])
        for ci_f, ci_r in [(fx_i, rx_i), (fy_i, ry_i)]:
            tmp = self.df_np[s:e, ci_f].copy()
            self.df_np[s:e, ci_f] = self.df_np[s:e, ci_r]
            self.df_np[s:e, ci_r] = tmp
        self.current_id = int(tid)
        self.id_var.set(int(tid))
        for f in range(start_frame, end_frame):
            self.edited_keys.add((int(f), int(tid)))
        self.update_info_panel()
        self.redraw_current_frame()
        self.refresh_candidate_tree()
        return end_frame - start_frame

    def _candidate_label_for(self, frame: int, tid: int) -> str:
        return self.candidate_labels_by_key.get((int(frame), int(tid)), "")

    def update_info_panel(self):
        risk = self._candidate_label_for(self.current_frame, self.current_id)
        suffix = f" | {risk}" if risk else ""
        self.preview_title_var.set(f"frame {self.current_frame} / ID {self.current_id}{suffix}")
        self._update_assign_history_label()

    def _update_assign_history_label(self):
        if not hasattr(self, '_assign_history_var'):
            return
        history = self._assign_history_entries(self.current_id, self.current_frame)
        if not history:
            self._assign_history_var.set("")
            return

        tracking_entries = [(src, code) for src, code in history if src in ('fwd', 'bwd')]
        correction_entries = [(src, code) for src, code in history if src not in ('fwd', 'bwd')]

        def _fmt_correction(src: str, code: float) -> str:
            short = ASSIGN_CODE_SHORT_LABELS.get(int(code), f"?{int(code)}")
            if src == 'correction_bwd':
                return f"bwd: {short}"
            return short

        parts = []
        if tracking_entries:
            # Group by src direction, emit "FWD: stage -> DIR-NAN -> VITERBI" per direction
            from collections import OrderedDict as _OD
            grouped: dict[str, list[str]] = {}
            order: list[str] = []
            for src, code in tracking_entries:
                if not np.isfinite(code):
                    continue
                short = ASSIGN_CODE_SHORT_LABELS.get(int(code), f"?{int(code)}")
                if src not in grouped:
                    grouped[src] = []
                    order.append(src)
                grouped[src].append(short)
            seg_parts = []
            for src in order:
                labels = grouped[src]
                seg_parts.append(f"{src.upper()}: {' -> '.join(labels)}")
            if seg_parts:
                parts.append(f"[id tracking]  {'    '.join(seg_parts)}")
        if correction_entries:
            labels = [_fmt_correction(s, c) for s, c in correction_entries if np.isfinite(c)]
            if labels:
                parts.append(f"[correction]  {' -> '.join(labels)}")

        self._assign_history_var.set("    ".join(parts))

    def _schedule_refresh_candidate_tree(self):
        if self.mode_var.get() != "amadeus":
            return
        if self._refresh_tree_job is not None:
            self.after_cancel(self._refresh_tree_job)
        self._refresh_tree_job = self.after(150, self._do_refresh_candidate_tree)

    def _do_refresh_candidate_tree(self):
        self._refresh_tree_job = None
        self.refresh_candidate_tree()

    def _cancel_pending_candidate_select(self) -> None:
        if self.pending_candidate_select_job is None:
            return
        try:
            self.after_cancel(self.pending_candidate_select_job)
        except Exception:
            pass
        self.pending_candidate_select_job = None

    def _suppress_candidate_tree_select_until_idle(self) -> None:
        """Ignore TreeviewSelect events generated by programmatic tree rebuilds."""
        self._cancel_pending_candidate_select()
        self.tree_event_suppressed = True
        self._candidate_tree_suppress_token += 1
        token = self._candidate_tree_suppress_token
        try:
            self.after_idle(lambda t=token: self._release_candidate_tree_select_suppression(t))
        except Exception:
            self.tree_event_suppressed = False

    def _release_candidate_tree_select_suppression(self, token: int) -> None:
        if token == self._candidate_tree_suppress_token:
            self.tree_event_suppressed = False

    def refresh_candidate_tree(self):
        if not hasattr(self, "candidate_tree"):
            return
        if self.mode_var.get() != "amadeus":
            return
        self._rebuild_candidate_label_index()
        self.update_info_panel()
        if getattr(self, "_type_panel_open", False):
            self._update_type_filter_counts()
        current_iid = str(self.current_suspect_index) if self.current_suspect_index is not None else None
        self.candidate_tree.delete(*self.candidate_tree.get_children())

        shown = 0
        for idx, item in enumerate(self.suspects):
            if float(item["assign_code"]) not in self.enabled_assign_codes:
                continue
            edit = "yes" if (int(item["frame"]), int(item["tid"])) in self.edited_keys else ""
            if self.show_only_unedited_var.get() and edit:
                continue
            self.candidate_tree.insert("", "end", iid=str(idx), values=(item["frame"], item["tid"], item["label_short"], edit))
            shown += 1
        if current_iid and current_iid in self.candidate_tree.get_children():
            self._suppress_candidate_tree_select_until_idle()
            self.candidate_tree.selection_set(current_iid)
            self.candidate_tree.see(current_iid)
        else:
            self._cancel_pending_candidate_select()
        if self.suspects:
            self.set_status(f"Candidates: {shown:,} shown ({len(self.suspects):,} total entries)")

    def on_candidate_select(self, _event=None):
        if self.tree_event_suppressed:
            return
        sel = self.candidate_tree.selection()
        if not sel:
            return
        try:
            suspect_index = int(sel[0])
        except Exception:
            return

        # Programmatic selection restoration during candidate-tree refresh must not
        # be interpreted as a user request to jump back to that candidate frame.
        # This is the source of the "drag an arrow on another frame -> snap back" bug.
        if suspect_index == self.current_suspect_index and self.focus_get() is not self.candidate_tree:
            return

        # Defer the actual jump until Tk finishes Treeview's own selection handling.
        # This avoids re-entrant redraw/selection events that can make the UI appear frozen
        # when a candidate row is clicked.
        self._cancel_pending_candidate_select()
        self.pending_candidate_select_job = self.after_idle(lambda idx=suspect_index: self.select_candidate(idx))

    def select_candidate(self, suspect_index: int):
        self.pending_candidate_select_job = None
        if suspect_index < 0 or suspect_index >= len(self.suspects):
            return
        self.current_suspect_index = int(suspect_index)
        item = self.suspects[suspect_index]
        frame = int(item["frame"])
        tid = int(item["tid"])
        if self.df is None or self.num_ids <= 0:
            return
        frame = max(0, min(self.frame_count - 1, frame))
        tid = max(0, min(self.num_ids - 1, tid))
        self.selected_tids.clear()
        self.selected_tids.append(tid)
        # Use the same non-blocking frame request path as the slider.
        self.request_frame(frame, reset_view=False, immediate=False)
        self._set_id(tid)
        iid = str(suspect_index)
        if iid in self.candidate_tree.get_children():
            self._suppress_candidate_tree_select_until_idle()
            if self.candidate_tree.selection() != (iid,):
                self.candidate_tree.selection_set(iid)
            self.candidate_tree.see(iid)
        self.set_status(f"Candidate: frame {frame}, ID {tid}, {item['label']}")
        self.focus_set()

    def _step_candidate(self, delta: int):
        visible = [
            idx for idx, item in enumerate(self.suspects)
            if float(item["assign_code"]) in self.enabled_assign_codes
            and not (self.show_only_unedited_var.get()
                     and (int(item["frame"]), int(item["tid"])) in self.edited_keys)
        ]
        if not visible:
            return
        if self.current_suspect_index is None:
            self.select_candidate(visible[0])
            return
        try:
            pos = visible.index(self.current_suspect_index)
        except ValueError:
            pos = 0
        new_pos = max(0, min(len(visible) - 1, pos + int(delta)))
        self.select_candidate(visible[new_pos])

    def _build_type_filter_panel(self):
        ordered = list(ASSIGN_CODE_ORDER)
        ordered_set = set(ordered)
        extras = sorted(code for code in ASSIGN_CODE_LABELS.keys() if code not in ordered_set)
        codes = ordered + extras
        n = len(codes)
        mid = (n + 1) // 2
        for i, code in enumerate(codes):
            short = ASSIGN_CODE_SHORT_LABELS.get(code, f"t{int(code)}")
            lvar = tk.StringVar(value=f"{short}  (0)")
            bvar = tk.BooleanVar(value=(code in self.enabled_assign_codes))
            self._type_filter_label_vars[code] = lvar
            self._type_filter_cb_vars[code] = bvar

            def _on_toggle(c=code, v=bvar):
                if v.get():
                    self.enabled_assign_codes.add(c)
                else:
                    self.enabled_assign_codes.discard(c)
                self.refresh_candidate_tree()

            col = 0 if i < mid else 1
            row = i if i < mid else i - mid
            cb = ctk.CTkCheckBox(self._type_filter_panel, textvariable=lvar,
                                  variable=bvar, command=_on_toggle)
            cb.grid(row=row, column=col, sticky="w", padx=(6, 12), pady=1)

        btn_row = ctk.CTkFrame(self._type_filter_panel, corner_radius=0)
        btn_row.grid(row=mid, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 4))

        def _all():
            for v in self._type_filter_cb_vars.values():
                v.set(True)
            self.enabled_assign_codes = set(self._type_filter_cb_vars.keys())
            self.refresh_candidate_tree()

        def _clear():
            for v in self._type_filter_cb_vars.values():
                v.set(False)
            self.enabled_assign_codes.clear()
            self.refresh_candidate_tree()

        ctk.CTkButton(btn_row, text="All", command=_all, width=50).pack(side="left")
        ctk.CTkButton(btn_row, text="Clear", command=_clear, width=50).pack(side="left", padx=4)

    def _toggle_type_filter_panel(self):
        if self._type_panel_open:
            self._type_filter_panel.pack_forget()
            self._type_panel_open = False
            self._type_toggle_btn.configure(text="> types")
        else:
            self._type_filter_panel.pack(fill="x", before=self.candidate_tree.master)
            self._type_panel_open = True
            self._type_toggle_btn.configure(text="v types")
            self._update_type_filter_counts()

    def _update_type_filter_counts(self):
        counts: dict[float, int] = {}
        for s in self.suspects:
            c = float(s["assign_code"])
            counts[c] = counts.get(c, 0) + 1
        for code, lvar in self._type_filter_label_vars.items():
            short = ASSIGN_CODE_SHORT_LABELS.get(code, f"t{int(code)}")
            lvar.set(f"{short}  ({counts.get(code, 0)})")

    def save_csv_quick(self):
        """Ctrl+S: save to last path without dialog; fall back to Save As if no path set."""
        if self.df is None:
            return
        if self.last_save_path:
            standard = _convert.legacy_wide_to_rear_front(self.df, "front_rear")
            _convert.save_rear_front_csv(standard, self.last_save_path)
            self.edited_keys.clear()
            self.save_count += 1
            self.set_status(f"Saved ({self.save_count}): {self.last_save_path}")
            return
        self.save_csv_as()

    def save_csv_as(self):
        """Save As button: always prompt for file path."""
        if self.df is None:
            return
        default_name = self._default_output_path(self.csv_path_var.get())
        path = filedialog.asksaveasfilename(
            title="Save refined CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=CSV_EXTS,
        )
        if not path:
            return
        standard = _convert.legacy_wide_to_rear_front(self.df, "front_rear")
        _convert.save_rear_front_csv(standard, path)
        self.edited_keys.clear()
        self.last_save_path = path
        self.save_count += 1
        self.set_status(f"Saved ({self.save_count}): {path}")

    def open_export_video_dialog(self):
        if self.reader is None or self.df is None:
            messagebox.showwarning("Warning", "Please load the video and CSV first.")
            return
        ExportVideoDialog(self)

    def export_video_async(self, *, out_path: str, start_frame: int, end_frame: int,
                           speed: float, output_fps: float, progress_cb, done_cb) -> None:
        if self.reader is None or self.df_np is None:
            done_cb(RuntimeError("Data has not been loaded."))
            return
        video_path = self.video_path_var.get().strip()
        df_np_snap = self.df_np.copy()
        front_x_idx = self._front_x_idx.copy()
        front_y_idx = self._front_y_idx.copy()
        rear_x_idx  = self._rear_x_idx.copy()
        rear_y_idx  = self._rear_y_idx.copy()
        id_palette = list(self.id_palette)
        num_ids = self.num_ids
        arrow_width = max(1, int(self.arrow_width_var.get()))
        label_size = max(6, int(self.label_size_var.get()))
        show_arrows = self.show_arrows_var.get()
        show_labels = self.show_labels_var.get()
        reverse = self.reverse_direction_var.get()

        def worker():
            err = None
            try:
                _export_video(
                    video_path, out_path, start_frame, end_frame, speed, output_fps,
                    df_np_snap, front_x_idx, front_y_idx, rear_x_idx, rear_y_idx,
                    id_palette, num_ids, arrow_width, label_size, show_arrows, show_labels, reverse,
                    progress_cb,
                )
            except Exception as e:
                err = e
            finally:
                done_cb(err)

        threading.Thread(target=worker, daemon=True).start()

    def _default_output_path(self, csv_path: str) -> str:
        stem = Path(csv_path).stem
        if stem.endswith("_rear_front"):
            stem = stem[:-len("_rear_front")]
        return f"{stem}_direction_refined_rear_front.csv"

    def img_to_canvas(self, pt: np.ndarray) -> np.ndarray:
        return np.array([pt[0] * self.scale + self.offset_x, pt[1] * self.scale + self.offset_y], dtype=float)

    def canvas_to_img(self, pt: np.ndarray) -> np.ndarray:
        return np.array([(pt[0] - self.offset_x) / self.scale, (pt[1] - self.offset_y) / self.scale], dtype=float)

    def _fit_view(self, rgb: np.ndarray):
        img_h, img_w = rgb.shape[:2]
        can_w = max(1, self.canvas.winfo_width())
        can_h = max(1, self.canvas.winfo_height())
        self.base_fit_scale = min(can_w / img_w, can_h / img_h)
        self.zoom_scale = 1.0
        self.scale = self.base_fit_scale
        self.offset_x = (can_w - img_w * self.scale) * 0.5
        self.offset_y = (can_h - img_h * self.scale) * 0.5
        self.view_initialized = True
        self.current_image_shape = rgb.shape[:2]

    def _present_image(self, rgb: np.ndarray, preserve_view: bool):
        if (not preserve_view) or (not self.view_initialized) or self.current_image_shape != rgb.shape[:2]:
            self._fit_view(rgb)
        img_h, img_w = rgb.shape[:2]
        can_w = max(1, self.canvas.winfo_width())
        can_h = max(1, self.canvas.winfo_height())
        scale = self.scale

        # Crop to the visible region in image coordinates so we never resize
        # a larger-than-canvas image (critical for zoom-in performance).
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
        crop = rgb[iy0:iy1, ix0:ix1]

        dst_w = max(1, min(can_w, int(round((ix1 - ix0) * scale))))
        dst_h = max(1, min(can_h, int(round((iy1 - iy0) * scale))))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (dst_w, dst_h), interpolation=interp)

        dst_x = int(max(0.0, self.offset_x + ix0 * scale))
        dst_y = int(max(0.0, self.offset_y + iy0 * scale))
        img = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(dst_x, dst_y, anchor="nw", image=self.tk_image)

    def _queue_prefetch(self, center_frame: int):
        reader = self.reader
        if reader is None:
            return
        reader.request_prefetch(
            center_frame,
            forward=PREFETCH_FORWARD,
            backward=PREFETCH_BACKWARD,
        )

    def _render_display_frame(self, rgb: np.ndarray) -> np.ndarray:
        display = rgb.copy()
        # Ensure scale/offset are fit before computing in_view, otherwise
        # the default scale=1.0 filters out arrows at large pixel coordinates
        # when the canvas hasn't been laid out yet.
        if not self.view_initialized:
            self._fit_view(rgb)
        if self.df_np is not None and self.num_ids > 0:
            row = self.df_np[self.current_frame]
            fronts = np.column_stack([row[self._front_x_idx], row[self._front_y_idx]])
            rears  = np.column_stack([row[self._rear_x_idx],  row[self._rear_y_idx]])
            if self.reverse_direction_var.get():
                fronts, rears = rears, fronts
            valid = np.isfinite(fronts).all(axis=1) & np.isfinite(rears).all(axis=1)
            can_w = max(1, self.canvas.winfo_width())
            can_h = max(1, self.canvas.winfo_height())
            pad = float(max(SNAP_PIXELS, LABEL_HIT_RADIUS) * 2)
            fc = fronts * self.scale + np.array([self.offset_x, self.offset_y])
            rc = rears  * self.scale + np.array([self.offset_x, self.offset_y])
            in_view = (
                (np.maximum(fc[:, 0], rc[:, 0]) >= -pad) &
                (np.minimum(fc[:, 0], rc[:, 0]) <= can_w + pad) &
                (np.maximum(fc[:, 1], rc[:, 1]) >= -pad) &
                (np.minimum(fc[:, 1], rc[:, 1]) <= can_h + pad)
            )
            for tid in np.flatnonzero(valid & in_view).tolist():
                self._draw_arrow_overlay(display, tid, ArrowState(front=fronts[tid], rear=rears[tid]))
        return display

    def _draw_canvas(self, rgb: np.ndarray, preserve_view: bool):
        display = self._render_display_frame(rgb)
        self._present_image(display, preserve_view=preserve_view)

    def _draw_arrow_overlay(self, img: np.ndarray, tid: int, state: ArrowState):
        if not (np.all(np.isfinite(state.front)) and np.all(np.isfinite(state.rear))):
            return
        front = np.round(state.front).astype(int)
        rear = np.round(state.rear).astype(int)
        arrow_width = max(1, int(self.arrow_width_var.get()))
        label_size = max(6, int(self.label_size_var.get()))
        tip_length = max(8, int(round(arrow_width * 5.0)))

        arrow_color = self._id_color(tid)
        label_color = self._id_color(tid)

        if tid in self.selected_tids:
            if self.show_select_circle_var.get():
                center = np.round((state.front.astype(float) + state.rear.astype(float)) / 2.0).astype(int)
                arrow_len = float(np.linalg.norm(state.front - state.rear))
                radius = max(12, int(arrow_len * 0.75))
                overlay = img.copy()
                cv2.circle(overlay, tuple(center), radius, SELECT_OUTLINE_COLOR, 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            outline_width = arrow_width + 4
            cv2.arrowedLine(
                img,
                tuple(rear),
                tuple(front),
                SELECT_OUTLINE_COLOR,
                outline_width,
                cv2.LINE_AA,
                tipLength=min(0.8, tip_length / max(1.0, float(np.linalg.norm(front - rear)))),
            )

        if self.show_arrows_var.get():
            cv2.arrowedLine(
                img,
                tuple(rear),
                tuple(front),
                arrow_color,
                arrow_width,
                cv2.LINE_AA,
                tipLength=min(0.8, tip_length / max(1.0, float(np.linalg.norm(front - rear)))),
            )

        if self.show_labels_var.get():
            anchor = np.round((front + rear) / 2.0).astype(int)
            label_pos = (int(anchor[0]) + 6, int(anchor[1]) - 6)
            risk = self._candidate_label_for(self.current_frame, tid)
            label_text = f"ID {tid}" + (f" [{risk}]" if risk else "")
            font_scale = label_size / 28.0
            text_thickness = max(1, label_size // 10)
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )
            if tid in self.selected_tids:
                cv2.rectangle(
                    img,
                    (label_pos[0] - 4, label_pos[1] - text_h - 4),
                    (label_pos[0] + text_w + 4, label_pos[1] + baseline + 4),
                    SELECT_LABEL_BG_COLOR,
                    -1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                img,
                label_text,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                label_color,
                text_thickness,
                cv2.LINE_AA,
            )

    def _id_color(self, tid: int) -> tuple[int, int, int]:
        if not self.id_palette:
            return WHITE_RGB
        if tid < 0 or tid >= len(self.id_palette):
            return WHITE_RGB
        return self.id_palette[tid]

    def shuffle_colors(self):
        if self.df is None or self.num_ids <= 0:
            return
        colors = list(self.id_palette)
        random.SystemRandom().shuffle(colors)
        self.id_palette = colors
        self.redraw_current_frame()
        self.set_status("ID color mapping shuffled")

    def _pick_arrow(self, canvas_pt: np.ndarray) -> Optional[int]:
        if self.df is None:
            return None
        best_tid = None
        best_score = None
        snap = float(SNAP_PIXELS)
        for tid in range(self.num_ids):
            state = self._get_arrow_state(self.current_frame, tid)
            if not (np.all(np.isfinite(state.front)) and np.all(np.isfinite(state.rear))):
                continue
            front_c = self.img_to_canvas(state.front)
            rear_c = self.img_to_canvas(state.rear)
            body_dist = self._point_to_segment_distance(canvas_pt, rear_c, front_c)
            label_anchor = (front_c + rear_c) / 2.0 + np.array([6.0, -6.0], dtype=float)
            label_dist = float(np.linalg.norm(canvas_pt - label_anchor))
            score = min(body_dist, label_dist)
            if body_dist > snap and label_dist > LABEL_HIT_RADIUS:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_tid = tid
        return best_tid

    def _pick_drag_target(self, canvas_pt: np.ndarray) -> Optional[tuple[int, str]]:
        if self.df is None:
            return None
        endpoint_best: Optional[tuple[float, int, str]] = None
        whole_best: Optional[tuple[float, int, str]] = None
        for tid in range(self.num_ids):
            state = self._get_arrow_state(self.current_frame, tid)
            front_c = self.img_to_canvas(state.front)
            rear_c = self.img_to_canvas(state.rear)
            d_front = float(np.linalg.norm(canvas_pt - front_c))
            d_rear = float(np.linalg.norm(canvas_pt - rear_c))
            body_dist = self._point_to_segment_distance(canvas_pt, rear_c, front_c)
            label_anchor = (front_c + rear_c) / 2.0 + np.array([6.0, -6.0], dtype=float)
            label_dist = float(np.linalg.norm(canvas_pt - label_anchor))

            if d_front <= ENDPOINT_HIT_RADIUS and (endpoint_best is None or d_front < endpoint_best[0]):
                endpoint_best = (d_front, tid, "front")
            if d_rear <= ENDPOINT_HIT_RADIUS and (endpoint_best is None or d_rear < endpoint_best[0]):
                endpoint_best = (d_rear, tid, "rear")

            score = min(body_dist, label_dist)
            if body_dist <= float(SNAP_PIXELS) or label_dist <= LABEL_HIT_RADIUS:
                if whole_best is None or score < whole_best[0]:
                    whole_best = (score, tid, "whole")

        if endpoint_best is not None:
            return endpoint_best[1], endpoint_best[2]
        if whole_best is not None:
            return whole_best[1], whole_best[2]
        return None

    def on_canvas_press(self, event):
        self.focus_set()
        self.left_button_down = True
        self._stop_key_play()
        canvas_pt = np.array([event.x, event.y], dtype=float)
        self.drag_last_canvas = canvas_pt.copy()
        self.hover_tid = None

        picked = self._pick_drag_target(canvas_pt) if self.df is not None else None
        if picked is None:
            if self.frame_cache_rgb is not None:
                self.drag_mode = "pan"
                self.drag_tid = None
                self._pan_start_canvas = canvas_pt.copy()
                self._pan_start_offset = np.array([self.offset_x, self.offset_y], dtype=float)
                self.canvas.configure(cursor="fleur")
            else:
                self.drag_mode = None
                self.drag_tid = None
            return

        tid, mode = picked
        state = self._get_arrow_state(self.current_frame, tid)
        self.drag_mode = mode
        self.drag_tid = tid
        self.drag_start_front = state.front.copy()
        self.drag_start_rear = state.rear.copy()
        self.drag_anchor_img = self.canvas_to_img(canvas_pt)
        self.hover_tid = tid
        self._set_id(tid)

    def on_canvas_drag(self, event):
        canvas_pt = np.array([event.x, event.y], dtype=float)
        self.drag_last_canvas = canvas_pt

        if self.drag_mode == "pan":
            if self._pan_start_canvas is not None and self._pan_start_offset is not None:
                delta = canvas_pt - self._pan_start_canvas
                self.offset_x = self._pan_start_offset[0] + delta[0]
                self.offset_y = self._pan_start_offset[1] + delta[1]
                self.redraw_current_frame()
            return

        if self.df is None or self.drag_mode is None or self.drag_tid is None:
            return
        self._apply_drag_from_canvas(canvas_pt)

    def on_canvas_release(self, _event=None):
        if self.drag_mode == "pan":
            self.canvas.configure(cursor="")
        self.left_button_down = False
        self.drag_mode = None
        self.drag_tid = None
        self.drag_last_canvas = None
        self.drag_start_front = None
        self.drag_start_rear = None
        self.drag_anchor_img = None
        self._pan_start_canvas = None
        self._pan_start_offset = None

    def on_canvas_right_click(self, event):
        self._stop_key_play()
        if self.df is None:
            return "break"
        tid = self._pick_arrow(np.array([event.x, event.y], dtype=float))
        if tid is None:
            return "break"

        now = time.monotonic()
        is_dblclick = (
            tid == self._last_right_click_tid
            and (now - self._last_right_click_time) * 1000 < self._RIGHT_DBLCLICK_MS
        )
        self._last_right_click_tid = tid
        self._last_right_click_time = now

        if is_dblclick and tid in self.selected_tids and len(self.selected_tids) == 1:
            self._close_swap_popup()
            self._show_selection_popup(event.x_root, event.y_root)
            return "break"

        self._toggle_selected_tid(tid, event.x_root, event.y_root)
        return "break"

    def _toggle_selected_tid(self, tid: int, x_root: int, y_root: int):
        was_removing = tid in self.selected_tids
        if was_removing:
            self.selected_tids.remove(tid)
            # Keep current_id on the still-selected ID, not the one just removed
            new_id = self.selected_tids[-1] if self.selected_tids else tid
            self.current_id = int(new_id)
            self.id_var.set(int(new_id))
        else:
            if len(self.selected_tids) >= 2:
                self.selected_tids.pop(0)
            self.selected_tids.append(tid)
            self.current_id = int(tid)
            self.id_var.set(int(tid))
        self.update_info_panel()
        self.redraw_current_frame()
        if was_removing:
            self._close_swap_popup()
        elif len(self.selected_tids) == 2:
            self._show_selection_popup(x_root, y_root)
        # len == 1: no popup -- wait for quick double right-click

    def _show_selection_popup(self, x_root: int, y_root: int):
        self._close_swap_popup()
        if not self.selected_tids:
            return
        popup = ctk.CTkToplevel(self)
        install_window_icon(popup)
        popup.transient(self)
        popup.resizable(False, False)
        popup.attributes("-topmost", True)
        popup.protocol("WM_DELETE_WINDOW", self._close_swap_popup)

        if len(self.selected_tids) == 1:
            tid = self.selected_tids[0]
            popup.title("reverse direction")
            ctk.CTkLabel(popup, text=f"Reverse arrow direction for ID {tid}").pack(padx=12, pady=(12, 8))
            button_frame = ctk.CTkFrame(popup, corner_radius=0)
            button_frame.pack(padx=12, pady=(0, 12))
            ctk.CTkButton(button_frame, text="this frame", width=110, command=self.swap_selected_arrow_this_frame).pack(side="left")
            ctk.CTkButton(button_frame, text="this frame onward", width=140, command=self.swap_selected_arrow_from_current).pack(side="left", padx=(8, 0))
        elif len(self.selected_tids) == 2:
            popup.title(f"ID {self.selected_tids[0]} <-> ID {self.selected_tids[1]}")
            ctk.CTkButton(popup, text="swap onward", width=110, command=self.swap_selected_ids).pack(padx=12, pady=12)
        else:
            popup.destroy()
            return

        popup.update_idletasks()
        popup.geometry(f"+{x_root + 10}+{y_root + 10}")
        self.swap_popup = popup

    def _close_swap_popup(self):
        if self.swap_popup is not None:
            try:
                self.swap_popup.destroy()
            except Exception:
                pass
            self.swap_popup = None

    def swap_selected_ids(self):
        if len(self.selected_tids) != 2 or self.df is None:
            self._close_swap_popup()
            return
        tid_a, tid_b = self.selected_tids
        swapped = self._swap_ids_from_current_frame(tid_a, tid_b)
        self.set_status(f"frame {self.current_frame} onward, {swapped} frames: swapped ID {tid_a} and ID {tid_b}")
        self.selected_tids.clear()
        self._close_swap_popup()
        self.update_info_panel()
        self.redraw_current_frame()

    def swap_selected_arrow_this_frame(self):
        if len(self.selected_tids) != 1 or self.df is None:
            self._close_swap_popup()
            return
        tid = self.selected_tids[0]
        self._swap_arrow_direction_range(tid, self.current_frame, self.current_frame + 1)
        self.set_status(f"frame {self.current_frame}: reversed arrow direction for ID {tid}")
        self.selected_tids.clear()
        self._close_swap_popup()
        self.update_info_panel()
        self.redraw_current_frame()

    def swap_selected_arrow_from_current(self):
        if len(self.selected_tids) != 1 or self.df is None:
            self._close_swap_popup()
            return
        tid = self.selected_tids[0]
        swapped = self._swap_arrow_direction_range(tid, self.current_frame, self.frame_count)
        self.set_status(f"frame {self.current_frame} onward, {swapped} frames: reversed arrow direction for ID {tid}")
        self.selected_tids.clear()
        self._close_swap_popup()
        self.update_info_panel()
        self.redraw_current_frame()

    def _apply_drag_from_canvas(self, canvas_pt: np.ndarray, frame_idx: int | None = None):
        if self.df is None or self.drag_mode is None or self.drag_tid is None:
            return
        target_frame = self.current_frame if frame_idx is None else int(frame_idx)
        state = self._drag_state_from_canvas(canvas_pt)
        if state is None:
            return
        self._set_arrow_state(target_frame, self.drag_tid, state)
        if target_frame == self.current_frame:
            self.redraw_current_frame()

    def _drag_state_from_canvas(self, canvas_pt: np.ndarray) -> Optional[ArrowState]:
        target_img = self.canvas_to_img(np.asarray(canvas_pt, dtype=float))
        if self.drag_start_front is None or self.drag_start_rear is None or self.drag_anchor_img is None:
            return None
        if self.drag_mode == "front":
            new_front = target_img
            new_rear = self.drag_start_rear.copy()
        elif self.drag_mode == "rear":
            new_front = self.drag_start_front.copy()
            new_rear = target_img
        else:
            delta = target_img - self.drag_anchor_img
            new_front = self.drag_start_front + delta
            new_rear = self.drag_start_rear + delta
        return ArrowState(front=new_front, rear=new_rear)

    def on_mousewheel(self, event):
        if self.frame_cache_rgb is None:
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

    @staticmethod
    def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))


def _export_video(
    video_path: str,
    out_path: str,
    start_frame: int,
    end_frame: int,
    speed: float,
    output_fps: float,
    df_np: np.ndarray,
    front_x_idx: np.ndarray,
    front_y_idx: np.ndarray,
    rear_x_idx: np.ndarray,
    rear_y_idx: np.ndarray,
    id_palette: list,
    num_ids: int,
    arrow_width: int,
    label_size: int,
    show_arrows: bool,
    show_labels: bool,
    reverse: bool,
    progress_cb,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first = cap.read()
        if not ok:
            raise RuntimeError("Cannot read frame.")
        h, w = first.shape[:2]

        total_source = end_frame - start_frame + 1
        n_output = max(1, int(round(total_source / speed)))
        source_indices = [
            min(end_frame, start_frame + int(round(j * speed)))
            for j in range(n_output)
        ]

        ext = os.path.splitext(out_path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if ext == ".avi" else "mp4v"))
        writer = cv2.VideoWriter(out_path, fourcc, output_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video file: {out_path}")

        total = len(source_indices)
        tip_length = max(8, int(round(arrow_width * 5.0)))
        prev_idx = -1
        try:
            for done, src_idx in enumerate(source_indices):
                if src_idx != prev_idx + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ok, bgr = cap.read()
                if not ok:
                    break
                prev_idx = src_idx
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if df_np is not None and num_ids > 0 and src_idx < len(df_np):
                    row = df_np[src_idx]
                    fronts = np.column_stack([row[front_x_idx], row[front_y_idx]])
                    rears  = np.column_stack([row[rear_x_idx],  row[rear_y_idx]])
                    if reverse:
                        fronts, rears = rears, fronts
                    valid = np.isfinite(fronts).all(axis=1) & np.isfinite(rears).all(axis=1)
                    for tid in np.flatnonzero(valid).tolist():
                        f_pt = np.round(fronts[tid]).astype(int)
                        r_pt = np.round(rears[tid]).astype(int)
                        color = id_palette[tid] if 0 <= tid < len(id_palette) else WHITE_RGB
                        alen = max(1.0, float(np.linalg.norm(f_pt - r_pt)))
                        tip = min(0.8, tip_length / alen)
                        if show_arrows:
                            cv2.arrowedLine(rgb, tuple(r_pt), tuple(f_pt), color,
                                            arrow_width, cv2.LINE_AA, tipLength=tip)
                        if show_labels:
                            anchor = np.round((fronts[tid] + rears[tid]) / 2.0).astype(int)
                            lpos = (int(anchor[0]) + 6, int(anchor[1]) - 6)
                            cv2.putText(rgb, f"ID {tid}", lpos, cv2.FONT_HERSHEY_SIMPLEX,
                                        label_size / 28.0, color, max(1, label_size // 10), cv2.LINE_AA)

                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if progress_cb and (done % 10 == 0 or done == total - 1):
                    progress_cb(done + 1, total, f"Writing frame {src_idx} ({done + 1}/{total})")
        finally:
            writer.release()

        if progress_cb:
            progress_cb(total, total, f"Done: {total} frames -> {os.path.basename(out_path)}")
    finally:
        cap.release()


class ExportVideoDialog(ctk.CTkToplevel):
    def __init__(self, parent: "UmaDirectionRefinementApp") -> None:
        super().__init__(parent)
        install_window_icon(self)
        self.app = parent
        self.title("Export Video")
        self.geometry("520x310")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        src_fps = 30.0
        if parent.reader is not None:
            cap_fps = parent.reader.cap.get(cv2.CAP_PROP_FPS)
            if cap_fps > 0:
                src_fps = cap_fps
        self._src_fps = src_fps
        frame_count = parent.frame_count

        self.start_var = tk.IntVar(value=0)
        self.end_var = tk.IntVar(value=max(0, frame_count - 1))
        self.speed_var = tk.DoubleVar(value=1.0)
        self.fps_var = tk.DoubleVar(value=round(src_fps, 2))
        self.out_var = tk.StringVar()

        self._spin_cfg = dict(
            bg="#343638", fg="#dce4ee", insertbackground="#dce4ee",
            buttonbackground="#565b5e", relief="flat", highlightthickness=2,
            highlightbackground="#565b5e", highlightcolor="#1f8040",
            selectbackground="#1f6aa5", selectforeground="white",
        )
        self._build_ui(frame_count)

    def _build_ui(self, frame_count: int) -> None:
        pad = dict(padx=8, pady=4)
        root = ctk.CTkFrame(self, corner_radius=0)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(root, text="Frame range:").grid(row=0, column=0, sticky="w", **pad)
        fr_row = ctk.CTkFrame(root, corner_radius=0)
        fr_row.grid(row=0, column=1, columnspan=2, sticky="w", **pad)
        tk.Spinbox(fr_row, from_=0, to=max(0, frame_count - 1), width=8,
                   textvariable=self.start_var, **self._spin_cfg).pack(side="left")
        ctk.CTkLabel(fr_row, text=" - ").pack(side="left")
        tk.Spinbox(fr_row, from_=0, to=max(0, frame_count - 1), width=8,
                   textvariable=self.end_var, **self._spin_cfg).pack(side="left")

        ctk.CTkLabel(root, text="Speed multiplier:").grid(row=1, column=0, sticky="w", **pad)
        sp_row = ctk.CTkFrame(root, corner_radius=0)
        sp_row.grid(row=1, column=1, columnspan=2, sticky="w", **pad)
        tk.Spinbox(sp_row, from_=0.1, to=100.0, increment=0.5, width=8,
                   textvariable=self.speed_var, format="%.1f", **self._spin_cfg).pack(side="left")
        ctk.CTkLabel(sp_row, text="x  (2.0 = 2x faster)").pack(side="left", padx=(6, 0))

        ctk.CTkLabel(root, text="Output FPS:").grid(row=2, column=0, sticky="w", **pad)
        fps_row = ctk.CTkFrame(root, corner_radius=0)
        fps_row.grid(row=2, column=1, columnspan=2, sticky="w", **pad)
        tk.Spinbox(fps_row, from_=1.0, to=240.0, increment=1.0, width=8,
                   textvariable=self.fps_var, format="%.2f", **self._spin_cfg).pack(side="left")
        ctk.CTkLabel(fps_row, text=f"fps  (source: {self._src_fps:.2f} fps)").pack(side="left", padx=(6, 0))

        ctk.CTkLabel(root, text="Output path:").grid(row=3, column=0, sticky="w", **pad)
        out_row = ctk.CTkFrame(root, corner_radius=0)
        out_row.grid(row=3, column=1, columnspan=2, sticky="we", **pad)
        ctk.CTkEntry(out_row, textvariable=self.out_var, width=280).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(out_row, text="...", width=36, command=self._browse_out).pack(side="left", padx=(4, 0))

        self.status_var = tk.StringVar(value="")
        ctk.CTkLabel(root, textvariable=self.status_var, anchor="w").grid(
            row=4, column=0, columnspan=3, sticky="we", padx=8, pady=(4, 0))
        self.progress_bar = ctk.CTkProgressBar(root, mode="determinate")
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky="we", padx=8, pady=4)

        self.export_btn = ctk.CTkButton(root, text="Export", command=self._start_export)
        self.export_btn.grid(row=6, column=0, columnspan=3, pady=(4, 0))

        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=0)

        self._ui_queue: queue.Queue = queue.Queue()
        self._poll_ui_queue()

    def _poll_ui_queue(self) -> None:
        while True:
            try:
                fn = self._ui_queue.get_nowait()
                fn()
            except queue.Empty:
                break
        try:
            self.after(50, self._poll_ui_queue)
        except tk.TclError:
            pass

    def _browse_out(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save video",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"), ("All files", "*.*")],
        )
        if path:
            self.out_var.set(path)

    def _start_export(self) -> None:
        out_path = self.out_var.get().strip()
        if not out_path:
            messagebox.showerror("Error", "Please specify an output path.", parent=self)
            return
        try:
            start = int(self.start_var.get())
            end = int(self.end_var.get())
            speed = float(self.speed_var.get())
            fps = float(self.fps_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"Parameter error: {e}", parent=self)
            return
        if start > end:
            messagebox.showerror("Error", "Start frame must be less than or equal to End frame.", parent=self)
            return
        if speed <= 0:
            messagebox.showerror("Error", "Speed multiplier must be greater than 0.", parent=self)
            return
        if fps <= 0:
            messagebox.showerror("Error", "FPS must be greater than 0.", parent=self)
            return
        self.export_btn.configure(state="disabled")
        self.app.export_video_async(
            out_path=out_path,
            start_frame=start,
            end_frame=end,
            speed=speed,
            output_fps=fps,
            progress_cb=self._update_progress,
            done_cb=self._on_done,
        )

    def _update_progress(self, done: int, total: int, text: str) -> None:
        ratio = 0.0 if total <= 0 else done / total
        def _ui(r=ratio, t=text):
            try:
                self.progress_bar.set(r)
                self.status_var.set(t)
            except tk.TclError:
                pass
        self._ui_queue.put(_ui)

    def _on_done(self, error) -> None:
        def _finish(e=error):
            try:
                self.export_btn.configure(state="normal")
                if e:
                    messagebox.showerror("Error", str(e), parent=self)
                    self.status_var.set(f"Error: {e}")
                else:
                    self.status_var.set("Done!")
                    messagebox.showinfo("Done", "Video export complete.", parent=self)
            except tk.TclError:
                pass
        self._ui_queue.put(_finish)


class PoseConvertDialog(ctk.CTkToplevel):
    """Browse a SLEAP or DLC tracking file and convert to a rear_front CSV.

    The interpolation checkbox defaults to disabled for refinement workflows.
    Outputs use the shared ``{stem}_rear_front.csv`` format.
    """

    _FILE_TYPES = [
        ("Tracking files", "*.csv *.h5 *.hdf5"),
        ("CSV", "*.csv"),
        ("HDF5", "*.h5 *.hdf5"),
        ("All files", "*.*"),
    ]

    def __init__(self, parent: tk.Misc, initial_path: str | None = None, on_converted=None) -> None:
        super().__init__(parent)
        install_window_icon(self)
        self.title("Convert tracking data -> rear_front CSV")
        self.geometry("720x480")
        self.minsize(640, 420)
        self.transient(parent)
        self.grab_set()

        self._on_converted = on_converted
        self._keypoints: list[str] = []
        self._fmt: str = "unknown"

        self.in_var      = tk.StringVar()
        self.fmt_var     = tk.StringVar(value="-")
        self.n_ind_var   = tk.StringVar(value="10")
        self.front_var   = tk.StringVar()
        self.rear_var    = tk.StringVar()
        self.interpolate_var = tk.BooleanVar(value=False)

        self._build_ui()

        if initial_path:
            self.in_var.set(initial_path)
            self._load_file_info(initial_path)

    def _build_ui(self) -> None:
        pad = dict(padx=8, pady=4)
        root = ctk.CTkFrame(self, corner_radius=0)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        # File row
        ctk.CTkLabel(root, text="Input file (CSV / H5):").grid(row=0, column=0, sticky="w", **pad)
        ctk.CTkEntry(root, textvariable=self.in_var, width=400).grid(row=1, column=0, columnspan=3, sticky="we", **pad)
        ctk.CTkButton(root, text="Browse...", width=80, command=self._browse).grid(row=1, column=3, sticky="e", padx=(0, 8))

        # Format label
        ctk.CTkLabel(root, text="Detected format:").grid(row=2, column=0, sticky="w", **pad)
        ctk.CTkLabel(root, textvariable=self.fmt_var, text_color="#006699").grid(row=2, column=1, columnspan=3, sticky="w", **pad)

        sep = ctk.CTkFrame(root, height=2, fg_color=("gray60", "gray40"), corner_radius=0)
        sep.grid(row=3, column=0, columnspan=4, sticky="we", pady=6)

        # n_individuals
        ctk.CTkLabel(root, text="Number of individuals:").grid(row=4, column=0, sticky="w", **pad)
        tk.Spinbox(root, from_=1, to=999, width=8, textvariable=self.n_ind_var,
                   bg="#343638", fg="#dce4ee", insertbackground="#dce4ee",
                   buttonbackground="#565b5e", relief="flat",
                   highlightthickness=2, highlightbackground="#565b5e", highlightcolor="#1f8040",
                   selectbackground="#1f6aa5", selectforeground="white",
                   ).grid(row=4, column=1, sticky="w", **pad)

        # Front / rear keypoint
        ctk.CTkLabel(root, text="Front keypoint:").grid(row=5, column=0, sticky="w", **pad)
        self.front_combo = ctk.CTkComboBox(root, variable=self.front_var, state="readonly", width=180)
        self.front_combo.grid(row=5, column=1, sticky="w", **pad)

        ctk.CTkLabel(root, text="Rear keypoint:").grid(row=6, column=0, sticky="w", **pad)
        self.rear_combo = ctk.CTkComboBox(root, variable=self.rear_var, state="readonly", width=180)
        self.rear_combo.grid(row=6, column=1, sticky="w", **pad)

        self.interpolate_check = ctk.CTkCheckBox(
            root,
            text="Interpolate missing values",
            variable=self.interpolate_var,
        )
        self.interpolate_check.grid(row=7, column=0, columnspan=4, sticky="w", **pad)

        sep2 = ctk.CTkFrame(root, height=2, fg_color=("gray60", "gray40"), corner_radius=0)
        sep2.grid(row=8, column=0, columnspan=4, sticky="we", pady=6)

        # Convert button
        self.convert_btn = ctk.CTkButton(root, text="Convert & Load", command=self._convert, state="disabled")
        self.convert_btn.grid(row=9, column=0, columnspan=4, pady=(4, 0))

        # Log
        self.log = tk.Text(root, height=8, wrap="none", state="disabled")
        self.log.grid(row=10, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        sb = ttk.Scrollbar(root, orient="vertical", command=self.log.yview)
        sb.grid(row=10, column=4, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)

        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.columnconfigure(3, weight=0)
        root.rowconfigure(10, weight=1)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select tracking file",
            filetypes=self._FILE_TYPES,
        )
        if not path:
            return
        self.in_var.set(path)
        self._load_file_info(path)

    def _load_file_info(self, path: str) -> None:
        try:
            fmt = _convert.detect_format(path)
            self.fmt_var.set(_convert.format_label(fmt))
            self._fmt = fmt

            if fmt == "sleap_csv":
                df, kps = _convert.read_sleap_csv(Path(path))
                n = _convert.infer_n_individuals_sleap(df)
                individuals_info = f"{df['track_id'].nunique()} raw track IDs"
            elif fmt in ("dlc_csv", "dlc_h5"):
                df_wide, kps, individuals = _convert.read_dlc_file(Path(path), fmt)
                n = _convert.infer_n_individuals_dlc(individuals)
                individuals_info = f"individuals: {', '.join(individuals)}" if individuals else "single animal"
            else:
                self._log(f"Unsupported format: {path}")
                return

            self._keypoints = kps
            self.n_ind_var.set(str(n))
            vals = [""] + kps
            self.front_combo.configure(values=vals)
            self.rear_combo.configure(values=vals)
            # Auto-fill front/rear if keypoints match common names
            for kp in kps:
                if kp.lower() in ("front", "head", "nose"):
                    self.front_var.set(kp)
                    break
            for kp in kps:
                if kp.lower() in ("rear", "tail", "tail_base", "back"):
                    self.rear_var.set(kp)
                    break

            self._log(f"Loaded: {Path(path).name}")
            self._log(f"Format: {_convert.format_label(fmt)}")
            self._log(f"Keypoints: {', '.join(kps)}")
            self._log(f"{individuals_info} | Estimated number of individuals: {n}")
            self.convert_btn.configure(state="normal")
        except Exception as e:
            self._log(f"Error: {e}")
            messagebox.showerror("Error", str(e), parent=self)

    def _convert(self) -> None:
        in_path = self.in_var.get().strip()
        front   = self.front_var.get().strip()
        rear    = self.rear_var.get().strip()
        try:
            n = int(self.n_ind_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter an integer for the number of individuals.", parent=self)
            return

        if not in_path:
            messagebox.showerror("Error", "Please select a file.", parent=self)
            return
        if not front or not rear:
            messagebox.showerror("Error", "Please select a Front / Rear keypoint.", parent=self)
            return

        self.convert_btn.configure(state="disabled")
        try:
            interpolate = bool(self.interpolate_var.get())
            outputs = _convert.convert_tracking_outputs(
                in_path,
                n_individuals=n,
                front_keypoint=front,
                rear_keypoint=rear,
                interpolate=interpolate,
                reuse_existing=True,
            )
            out_path = outputs.rear_front
            if out_path is None:
                raise ValueError("The selected format does not contain a front/rear pair.")
            saved = [str(p) for p in (outputs.rear_front, outputs.center) if p is not None]
            self._log("Saved: " + ", ".join(saved))
            messagebox.showinfo("Done", "Saved:\n" + "\n".join(saved), parent=self)
            if self._on_converted is not None:
                self._on_converted(str(out_path))
            self.destroy()
        except Exception as e:
            self._log(f"Error: {e}")
            messagebox.showerror("Error", str(e), parent=self)
            try:
                self.convert_btn.configure(state="normal")
            except tk.TclError:
                pass

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def main():
    configure_taskbar_identity()
    app = UmaDirectionRefinementApp()
    app.mainloop()


if __name__ == "__main__":
    main()
