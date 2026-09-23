# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import argparse
import codecs
import copy
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
import signal
import secrets
import sys
import subprocess
import yaml
import threading
import time

import cv2
import customtkinter as ctk

try:
    from .project_paths import (
        PROJECT_ROOT,
        MAIN_DIR,
        ensure_import_paths,
        gui_asset,
        gui_script,
        main_script,
        open_results_directory,
        with_pythonpath,
    )
    from .window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
    from .advanced_tracking_layout import partition_category_items
    from .config_path_recovery import prepare_config_for_gui
    from .video_input import ask_open_analysis_video
except ImportError:  # Preserve direct execution with: python gui/gui_advanced_tracking.py
    from project_paths import (
        PROJECT_ROOT,
        MAIN_DIR,
        ensure_import_paths,
        gui_asset,
        gui_script,
        main_script,
        open_results_directory,
        with_pythonpath,
    )
    from window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
    from advanced_tracking_layout import partition_category_items
    from config_path_recovery import prepare_config_for_gui
    from video_input import ask_open_analysis_video

ensure_import_paths(MAIN_DIR)
from checkpoint_utils import checkpoint_spec_for_config
from experiment_utils import (
    DEFAULT_LR0,
    DEFAULT_LRF,
    parse_lr0_values,
    parse_lrf_values,
)
from path_utils import relativize_config_paths, resolve_config_paths
from segmentation_metadata import (
    read_segmentation_metadata,
    segmentation_paths_for_session,
    segmentation_pickle_path_for_video,
)
from tracking_constants import FIXED_INTERACT_IOU

CTK_THEME = str(gui_asset("deep_green.json"))
INITIAL_TRACKING_SCRIPT = str(main_script("initial_tracking.py"))

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme(CTK_THEME)

_TQDM_PAT = re.compile(r"(\d+)/(\d+)\s*\[")
# Strips ANSI color/cursor escape codes (e.g. tqdm's colour="green") from subprocess output.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _simplify_progress_line(line: str) -> str:
    """Reduce one cleaned subprocess output line to a short plain-text status."""
    match = _TQDM_PAT.search(line)
    if match:
        n, total = match.group(1), match.group(2)
        desc_m = re.search(r"\s(\d+)%\|", line)
        desc = ""
        if desc_m:
            prefix = line[:desc_m.start()]
            colon = prefix.rfind(":")
            desc = prefix[:colon].strip() if colon >= 0 else prefix.strip()
        return f"{desc} {n}/{total}" if desc else f"{n}/{total}"
    if line.startswith("[initial_tracking]"):
        line = line[len("[initial_tracking]"):].strip()
    return line

# The six measured config values initial_tracking.py overwrites when AUTO_PARAMS is
# enabled (see main/initial_tracking.py::_apply_auto_params). Their fields get
# a blue marker instead of the usual red "changed from default" marker,
# since whatever is typed here is replaced automatically at run time.
_AUTO_ADJUSTED_KEYS = frozenset({
    "LOCALIZED", "NUM_CROPS", "FREE_SCALE", "DIR_MIN_SEC", "CLUSTER_FRAMES",
    "MATCH_IOU",
})
_CHANGED_MARKER_COLOR = "#cc2200"
_AUTO_MARKER_COLOR = "#3d8bd6"

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


def _strict_int(value) -> int:
    """Parse an integer without silently truncating decimal input."""
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer setting")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"not an integer: {value!r}")
        return int(value)
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ValueError(f"not an integer: {value!r}")
    return int(text)


def _safe_stdout_write(data: str) -> None:
    try:
        sys.stdout.write(data)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = data.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe)


def _stdout_supports_overwrite() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _start_maximized(window: tk.Tk) -> None:
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
    for delay_ms in (50, 250, 1000):
        window.after(delay_ms, maximize)


class _AdaptiveScrollableFrame(ctk.CTkFrame):
    """Scroll content only when it is taller than the available area."""

    def __init__(self, master, **kwargs):
        super().__init__(master, width=1, height=1, corner_radius=0, **kwargs)
        self.pack_propagate(False)
        self.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        canvas_bg = self._apply_appearance_mode(ctk.ThemeManager.theme["CTk"]["fg_color"])
        self._parent_canvas = tk.Canvas(self, bg=canvas_bg, borderwidth=0, highlightthickness=0)
        self._parent_canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self._parent_canvas.yview)
        self._parent_canvas.configure(yscrollcommand=self._scrollbar.set)

        self.content = _layout_frame(self)
        self._create_window_id = self._parent_canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.content.bind("<Configure>", self._refresh_scroll_region, add="+")
        self._parent_canvas.bind("<Configure>", self._fit_content_width, add="+")
        self._parent_canvas.bind("<Configure>", self._refresh_scrollbar, add="+")
        self._scrollbar.grid_remove()
        self._parent_canvas.bind_all("<MouseWheel>", self._mousewheel, add="+")
        self._parent_canvas.bind_all("<Button-4>", lambda event: self._mousewheel(event, -1), add="+")
        self._parent_canvas.bind_all("<Button-5>", lambda event: self._mousewheel(event, 1), add="+")
        self.after_idle(self._refresh_scrollbar)

    def _fit_content_width(self, _event=None):
        try:
            self._parent_canvas.itemconfigure(
                self._create_window_id,
                width=self._parent_canvas.winfo_width(),
            )
        except tk.TclError:
            pass

    def _refresh_scroll_region(self, _event=None):
        try:
            self._parent_canvas.configure(scrollregion=self._parent_canvas.bbox("all"))
            self._refresh_scrollbar()
        except tk.TclError:
            pass

    def _refresh_scrollbar(self, _event=None):
        try:
            self.update_idletasks()
            bbox = self._parent_canvas.bbox("all")
            viewport_height = self._parent_canvas.winfo_height()
            content_height = 0 if bbox is None else bbox[3] - bbox[1]
            needs_scroll = content_height > viewport_height + 1
            scrollbar_visible = bool(self._scrollbar.winfo_ismapped())
            if needs_scroll and not scrollbar_visible:
                self._scrollbar.grid(row=0, column=1, sticky="nsew")
                self.after_idle(self._fit_content_width)
            elif not needs_scroll and scrollbar_visible:
                self._scrollbar.grid_remove()
                self.after_idle(self._fit_content_width)
        except tk.TclError:
            # The callback can run during window teardown.
            pass

    def _mousewheel(self, event, direction=None):
        if not self._scrollbar.winfo_ismapped():
            return
        if direction is None:
            delta = getattr(event, "delta", 0)
            direction = -1 if delta > 0 else 1
        self._parent_canvas.yview_scroll(direction, "units")


def _layout_frame(parent, *, corner_radius=0, **kwargs):
    """Create a geometry-only frame that follows its children naturally."""

    return ctk.CTkFrame(parent, width=1, height=1, corner_radius=corner_radius, **kwargs)


class ConfigGUI(ctk.CTk):

    def __init__(self, session_path: str = "", training_video_path: str = "", tracking_video_path: str = ""):
        super().__init__()
        configure_dpi_scaling(self)
        install_window_icon(self)
        self._prefill_session_path = session_path or ""
        self._prefill_training_video_path = training_video_path or ""
        self._prefill_tracking_video_path = tracking_video_path or training_video_path or ""
        self.title("AMADEUS Tracking")
        self.geometry("1400x1000")
        self.minsize(1000, 600)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._combo_callbacks: dict[str, list] = {}
        self._loaded_config_path = ""
        self._loaded_cfg: dict = {}
        self._loaded_training_video_path = ""
        self._create_video_color_seed: int | str | float | None = None

        # Reserve the bottom action bar in its own grid row so scrolling content
        # receives only the remaining height and can never cover these controls.
        bottom = _layout_frame(self, corner_radius=0)
        bottom.grid(row=1, column=0, sticky="ew")
        _layout_frame(bottom).pack(side="left", expand=True)
        ctk.CTkButton(bottom, text="Load YAML", width=120, command=self.load_yaml).pack(side="left", padx=8, pady=8)
        ctk.CTkButton(bottom, text="Save YAML", width=120, command=self.save_yaml).pack(side="left", padx=8, pady=8)
        self.switch_easy_btn = ctk.CTkButton(
            bottom, text="Switch Easy Tracking Mode", width=160, command=self.switch_easy_tracking_mode,
        )
        self.switch_easy_btn.pack(side="left", padx=8, pady=8)

        self.progress_bar = ctk.CTkProgressBar(bottom, width=180)
        self.progress_bar.pack(side="left", padx=(8, 4), pady=8)
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(bottom, text="Idle", width=160, anchor="w")
        self.progress_label.pack(side="left", padx=(0, 8), pady=8)

        self.adjust_btn = ctk.CTkButton(bottom, text="Adjust Parameters", width=120, command=self.adjust_parameters)
        self.adjust_btn.pack(side="left", padx=8, pady=8)

        self.run_btn = ctk.CTkButton(bottom, text="Run", width=120, command=self.run)
        self.run_btn.pack(side="left", padx=8, pady=8)

        self.open_results_btn = ctk.CTkButton(
            bottom, text="Open results", width=120, command=self.open_results,
        )
        self.open_results_btn.pack(side="left", padx=8, pady=8)

        self.stop_btn = ctk.CTkButton(bottom, text="Stop", width=120, state="disabled", command=self.stop)
        self.stop_btn.pack(side="left", padx=8, pady=8)
        self.proc = None
        self._batch_log_path = ""
        self._stop_requested = False
        _layout_frame(bottom).pack(side="left", expand=True)

        self._content_host = _AdaptiveScrollableFrame(self, fg_color="transparent")
        self._content_host.grid(row=0, column=0, sticky="nsew", padx=10, pady=(0, 5))

        # Basic Settings
        basic_outer = _layout_frame(self._content_host.content, corner_radius=6)
        basic_outer.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(basic_outer, text="Basic Settings", font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=8, pady=(6, 2))
        basic = _layout_frame(basic_outer)
        basic.pack(fill="x", padx=4, pady=(0, 6))

        self.basic_entries = {}

        row = _layout_frame(basic)
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="Session Path", width=140, anchor="w").pack(side="left", padx=(6, 0))
        ent = ctk.CTkEntry(row, width=300)
        ent.pack(side="left", expand=True, fill="x", padx=5)
        self.basic_entries["SESSION_PATH"] = ent
        ctk.CTkButton(row, text="Reference", width=90, command=lambda: self.browse_path("SESSION_PATH")).pack(side="left", padx=4)

        # Video Path
        self.path_mode = {
            "TRACKING_VIDEO_PATH_IS_DIR": tk.BooleanVar(value=False),
        }

        # Training Video Path
        row = _layout_frame(basic)
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="Training Video Path", width=140, anchor="w").pack(side="left", padx=(6, 0))
        ent = ctk.CTkEntry(row, width=300)
        ent.pack(side="left", expand=True, fill="x", padx=5)
        self.basic_entries["TRAINING_VIDEO_PATH"] = ent
        ent.bind("<FocusOut>", self._on_training_video_path_committed)
        ent.bind("<Return>", self._on_training_video_path_committed)
        ctk.CTkButton(row, text="Reference", width=90, command=lambda: self.browse_path("TRAINING_VIDEO_PATH")).pack(side="left", padx=4)

        # Tracking Video Path
        row = _layout_frame(basic)
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="Tracking Video Path", width=140, anchor="w").pack(side="left", padx=(6, 0))
        ent = ctk.CTkEntry(row, width=300)
        ent.pack(side="left", expand=True, fill="x", padx=5)
        self.basic_entries["TRACKING_VIDEO_PATH"] = ent
        ctk.CTkButton(row, text="Reference", width=90, command=lambda: self.browse_path("TRACKING_VIDEO_PATH")).pack(side="left", padx=4)
        ctk.CTkCheckBox(row, text="Select Directory", variable=self.path_mode["TRACKING_VIDEO_PATH_IS_DIR"]).pack(side="left", padx=6)

        # Number of Objects / Training Image Size
        row = _layout_frame(basic)
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="Number of Objects", width=140, anchor="w").pack(side="left", padx=(6, 0))
        sb = tk.Spinbox(row, from_=1, to=999, increment=1, width=5, **_SPIN_CFG)
        sb.pack(side="left", padx=5)
        self.basic_entries["NUM_OBJECTS"] = sb
        self.variable_count = tk.BooleanVar(value=False)
        self.without_direction = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(row, text="Variable population [Beta]", variable=self.variable_count,
                       command=self._toggle_variable_count).pack(side="left", padx=6)

        ctk.CTkLabel(row, text="Training Image Size (px)", width=175, anchor="w").pack(side="left", padx=(20, 0))
        training_image_size = tk.Spinbox(row, from_=32, to=8192, increment=32, width=8, **_SPIN_CFG)
        training_image_size.delete(0, tk.END)
        training_image_size.insert(0, "640")
        training_image_size.pack(side="left", padx=5)
        self.basic_entries["TRAIN_IMG_SIZE"] = training_image_size
        self._setup_training_image_size_widget(training_image_size)

        ctk.CTkLabel(row, text="Num Workers", width=95, anchor="w").pack(side="left", padx=(15, 0))
        worker_entry = ctk.CTkEntry(row, width=60)
        worker_entry.insert(0, "auto")
        worker_entry.pack(side="left", padx=3)
        self.basic_entries["NUM_WORKERS"] = worker_entry
        self._direction_mode_checkbox = ctk.CTkCheckBox(
            row,
            text="Without direction estimation [Beta]",
            variable=self.without_direction,
        )
        self._direction_mode_checkbox.pack(side="left", padx=6)

        preview_row = _layout_frame(basic)
        preview_row.pack(fill="x", pady=4)
        ctk.CTkLabel(preview_row, text="Num Preview Frames", width=130, anchor="w").pack(side="left", padx=(6, 0))
        num_preview = tk.Spinbox(preview_row, from_=0, to=100000, increment=1, width=6, **_SPIN_CFG)
        num_preview.delete(0, tk.END)
        num_preview.insert(0, "10")
        num_preview.pack(side="left", padx=5)
        self.basic_entries["NUM_PREVIEW_FRAMES"] = num_preview

        ctk.CTkLabel(preview_row, text="Preview Interval", width=105, anchor="w").pack(side="left", padx=(20, 0))
        preview_interval = tk.Spinbox(preview_row, from_=0, to=1000000, increment=1, width=6, **_SPIN_CFG)
        preview_interval.delete(0, tk.END)
        preview_interval.insert(0, "100")
        preview_interval.pack(side="left", padx=5)
        self.basic_entries["PREVIEW_INTERVAL"] = preview_interval

        ctk.CTkLabel(preview_row, text="Frame Interval", width=105, anchor="w").pack(side="left", padx=(10, 0))
        frame_interval_sb = tk.Spinbox(preview_row, from_=1, to=1000, increment=1, width=5, **_SPIN_CFG)
        frame_interval_sb.delete(0, tk.END)
        frame_interval_sb.insert(0, "5")
        frame_interval_sb.pack(side="left", padx=5)
        self.basic_entries["FRAME_INTERVAL"] = frame_interval_sb

        ctk.CTkLabel(preview_row, text="Num Train+Test", width=120, anchor="w").pack(side="left", padx=(10, 0))
        num_total_sb = tk.Spinbox(preview_row, from_=1, to=10**8, increment=100, width=8, **_SPIN_CFG)
        num_total_sb.delete(0, tk.END)
        num_total_sb.insert(0, "10000")
        num_total_sb.pack(side="left", padx=5)
        self.basic_entries["NUM_IMAGES"] = num_total_sb

        def _mk_int_spin(parent, min_val: int, width=7, init=None):
            vcmd = (self.register(
                lambda s, m=min_val: (
                    s == "" or
                    (s == "-" and m < 0) or
                    (s.lstrip("-").isdigit() and int(s) >= m)
                )
            ), "%P")
            sp = tk.Spinbox(
                parent, from_=min_val, to=10**12, increment=1, width=width,
                validate="key", validatecommand=vcmd, **_SPIN_CFG
            )
            if init is not None:
                sp.delete(0, tk.END)
                sp.insert(0, str(init))
            return sp

        self.delete_tmp_files = tk.BooleanVar(value=False)

        row = _layout_frame(basic)
        row.pack(fill="x", pady=2)

        # Training Range
        tr_frame = _layout_frame(row)
        tr_frame.pack(side="left", padx=(0, 40))
        ctk.CTkLabel(tr_frame, text="Range (Training)", width=130, anchor="w").pack(side="left", padx=(6, 0))
        tr_start = _mk_int_spin(tr_frame, 0, init=0)
        tr_start.pack(side="left", padx=(5, 2))
        ctk.CTkLabel(tr_frame, text=" - ").pack(side="left")
        tr_end = _mk_int_spin(tr_frame, -1, init=-1)
        tr_end.pack(side="left", padx=(2, 5))
        ctk.CTkLabel(tr_frame, text="frame").pack(side="left")
        self.training_range = {"start": tr_start, "end": tr_end}

        # Analysis Range
        an_frame = _layout_frame(row)
        an_frame.pack(side="left", padx=(40, 0))
        ctk.CTkLabel(an_frame, text="Range (Analysis)", width=130, anchor="w").pack(side="left", padx=(6, 0))
        an_start = _mk_int_spin(an_frame, 0, init=0)
        an_start.pack(side="left", padx=(5, 2))
        ctk.CTkLabel(an_frame, text=" - ").pack(side="left")
        an_end = _mk_int_spin(an_frame, -1, init=-1)
        an_end.pack(side="left", padx=(2, 5))
        ctk.CTkLabel(an_frame, text="frame").pack(side="left")
        self.analysis_range = {"start": an_start, "end": an_end}
        ctk.CTkCheckBox(row, text="Delete tmp files", variable=self.delete_tmp_files).pack(side="left", padx=(20, 6))

        # Random Seed
        seed_frame = _layout_frame(row)
        seed_frame.pack(side="left", padx=(20, 0))
        ctk.CTkLabel(seed_frame, text="Random Seed", width=95, anchor="w").pack(side="left", padx=(6, 0))
        random_seed_sb = _mk_int_spin(seed_frame, 0, width=10, init=0)
        random_seed_sb.pack(side="left", padx=5)
        self.basic_entries["RANDOM_SEED"] = random_seed_sb

        btn_bar = _layout_frame(self._content_host.content)
        btn_bar.pack(fill="x", padx=10, pady=(5, 0))
        self.sections = {}
        self.default_value_markers = {}
        self._parameter_rows = {}
        # Last-measured AUTO_PARAMS values (from Adjust Parameters or a loaded
        # config saved with AUTO_PARAMS=True), keyed by the 6 auto field keys.
        # Lets a manually-overridden auto field's red marker restore the
        # measured value instead of the unrelated static default.
        self._auto_param_values: dict = {}

        # Skip vars
        self.skip_initial_tracking = tk.BooleanVar(value=False)
        self.skip_trajectory_direction_filtering = tk.BooleanVar(value=False)
        self.skip_refine_blobs_through_tracking = tk.BooleanVar(value=False)
        self.skip_paste_blobs_with_crossing = tk.BooleanVar(value=False)
        self.skip_paste_blobs_clustered = tk.BooleanVar(value=False)
        self.skip_cropping = tk.BooleanVar(value=False)
        # Auto-determined from video geometry in _auto_fill_from_video; no longer
        # user-facing checkboxes/fields (crop/resize/no-resize is decided for the user).
        self.include_full_resized = tk.BooleanVar(value=True)
        self.skip_creating_direction_dataset = tk.BooleanVar(value=False)
        self.skip_training = tk.BooleanVar(value=False)
        self.skip_detection = tk.BooleanVar(value=False)
        self.skip_id_tracking = tk.BooleanVar(value=False)
        self.skip_id_correction = tk.BooleanVar(value=False)
        self.skip_creating_video = tk.BooleanVar(value=False)
        self._skip_cb_by_var: dict[int, ctk.CTkCheckBox] = {}

        # Initial Tracking
        initial_tracking_fields = {
            "Initial Tracking": [
                ("AUTO_PARAMS", "Auto Parameters", True),
                ("OBB_FIT_MODE", "Mask OBB Fit", "min_area"),
                ("LOCALIZED_RATIO", "Localized Ratio", 0.9),
                ("INIT_MAX_GAP", "Max Gap", 1),
                ("SKIP_INIT_PREVIEW", "Skip Preview", False),
            ],
        }
        self.create_section(
            title="Initial Tracking",
            fields=initial_tracking_fields,
            btn_parent=btn_bar,
            skip_vars=[
                (self.skip_initial_tracking, "Skip Initial Tracking"),
            ],
        )

        # Create single animal images
        direction_class_assignment_fields = {
            "Trajectory / Direction Filtering": [
                ("TRAJ_MAX_DIST", "Max Match Dist (body L ratio)", 2.0),
                ("DIR_MIN_SEC", "Min Consecutive (sec, auto)", 0.5),
                ("TRAJ_MAX_JUMP", "Jump Max (body L ratio)", 2.0),
                ("MIN_ASPECT", "Min OBB Aspect", 1.1),
                ("DIR_MIN_DISP", "Min Disp (body L ratio)", 1.0),
                ("SKIP_DIR_PREVIEW", "Skip Preview", False),
            ],
            "Refine blobs through training": [
                ("REFINE_FRAME_RATIO", "Refine Frame Ratio", 1.0),
                ("REFINE_EPOCHS", "Refine Epochs", 5),
                ("REFINE_BATCH", "Eval Batch Size", "auto"),
                ("REFINE_ITERS", "Num Refinement", 1),
                ("REFINE_MODEL", "Pretrained Model", "yolo11n-obb"),
                ("REFINE_CONF", "Conf", 0.2),
                ("RUN_DELETE_RATIO", "Run Delete Ratio", 0.4),
                ("SKIP_REFINE_PREVIEW", "Skip Preview", False),
            ],
        }
        self.create_section(
            title="Create single animal images",
            fields=direction_class_assignment_fields,
            btn_parent=btn_bar,
            skip_vars=[
                (self.skip_trajectory_direction_filtering, "Skip Trajectory / Direction Filtering"),
                (self.skip_refine_blobs_through_tracking, "Skip Refine Blobs Through Tracking"),
            ],
        )

        # Paste blobs
        create_with_crossing_fields = {
            "Random noise on animals": [
                ("NOISE_ENABLE", "Add Random Noise", False),
                ("NOISE_SIZE_PERCENT", "Noise Size (% of body length)", 10.0),
                ("NOISE_MAX_COUNT", "Max Noise per Animal", 10),
            ],
            "Paste composition per base image": [
                ("RATIO_SINGLE", "Base single ratio", 0.1),
                ("RATIO_P2", "Base contact pair ratio (p2)", 0.5),
                ("RATIO_P3", "Base contact triple ratio (p3)", 0.4),
                ("FREE_SCALE", "Additional paste ratio (auto)", 0.25),
                ("FREE_RATIO_SINGLE", "Free single ratio", 0.0),
                ("FREE_RATIO_P2", "Free 2-blob ratio", 1.0),
                ("FREE_RATIO_P3", "Free 3-blob ratio", 1.0),
            ],
            "Paste placement / appearance": [
                ("MAX_OVERLAP", "Max Overlap", 0.5),
                ("PASTE_SCALE_MIN", "Paste Scale Min", 0.9),
                ("PASTE_SCALE_MAX", "Paste Scale Max", 1.1),
                ("WIDTH_SCALE_MIN", "Width Scale Min", 0.9),
                ("WIDTH_SCALE_MAX", "Width Scale Max", 1.1),
                ("MASK_EXPANSION_RATIO", "Mask Expansion Ratio", 1.05),

                ("BRIGHT_MIN", "Brightness Min", 0.90),
                ("BRIGHT_MAX", "Brightness Max", 1.10),
                ("CONTRAST_MIN", "Contrast Min", 0.90),
                ("CONTRAST_MAX", "Contrast Max", 1.10),
                ("PASTE_LAYER_MODE", "Paste Layer Mode", "mixed"),
                ("UNDER_PASTE_PROB", "Under Paste Prob (mixed)", 0.5),

                ("OCCLUDER_MARGIN", "Occluder Margin px", -2),
                ("ALPHA_MODE", "Alpha Mode", "distance"),
                ("FEATHER_MIN", "Edge Feather Min px", 2),
                ("FEATHER_MAX", "Edge Feather Max px", 4),
                ("EDGE_BLUR_KSIZE", "Edge Blur Ksize (gauss)", 7),
                ("EDGE_BLUR_SIGMA", "Edge Blur Sigma (gauss)", 11.0),
                ("MAX_TRIES", "Max Placement Tries", 100),
            ],
            "Clustered paste": [
                ("CLUSTERED_RATIO", "Clustered ratio", 0.05),
                ("CLUSTER_COUNT", "Cluster Count", 12),
                ("CLUSTER_FRAMES", "Cluster Frames (auto)", 1),
                ("CLUSTER_FIT_LONG", "Cluster OBB Fit Long", 0.8),
                ("CLUSTER_FIT_SHORT", "Cluster OBB Fit Short", 0.8),
                ("CLUSTER_BREAK_PROB", "Parallel Break Prob", 0.20),
            ],
        }
        self.create_section(
            title="Create with crossing",
            fields=create_with_crossing_fields,
            btn_parent=btn_bar,
            skip_vars=[
                (self.skip_paste_blobs_with_crossing, "Skip Paste Blobs"),
                (self.skip_paste_blobs_clustered, "Skip Clustered Paste Blobs"),
            ],
        )

        noise_widgets = self.sections["Create with crossing"]["widgets"]
        noise_enable_var = noise_widgets["NOISE_ENABLE"]

        def _toggle_noise_fields(*_):
            self._set_create_images_group_enabled(
                "Create with crossing",
                ["NOISE_SIZE_PERCENT", "NOISE_MAX_COUNT"],
                bool(noise_enable_var.get()),
            )

        noise_enable_var.trace_add("write", _toggle_noise_fields)
        _toggle_noise_fields()

        create_with_crossing_widgets = self.sections["Create with crossing"]["widgets"]
        alpha_mode_widget = create_with_crossing_widgets["ALPHA_MODE"]

        def _toggle_alpha_mode(*_):
            mode = str(alpha_mode_widget.get()).strip().lower()
            is_gaussian = mode == "gaussian"
            self._set_create_images_group_enabled("Create with crossing", ["EDGE_BLUR_KSIZE", "EDGE_BLUR_SIGMA"], is_gaussian)
            for _blur_key in ("EDGE_BLUR_KSIZE", "EDGE_BLUR_SIGMA"):
                self._set_parameter_row_visible("Create with crossing", _blur_key, is_gaussian)

        self._toggle_with_crossing_alpha_mode = _toggle_alpha_mode
        self._bind_combo("ALPHA_MODE", _toggle_alpha_mode)

        paste_layer_widget = create_with_crossing_widgets["PASTE_LAYER_MODE"]

        def _toggle_paste_layer_mode(*_):
            mode = str(paste_layer_widget.get()).strip().lower()
            self._set_create_images_group_enabled("Create with crossing", ["UNDER_PASTE_PROB"], mode == "mixed")
            self._set_create_images_group_enabled("Create with crossing", ["OCCLUDER_MARGIN"], mode in {"under", "mixed"})

        self._toggle_with_crossing_paste_layer_mode = _toggle_paste_layer_mode
        self._bind_combo("PASTE_LAYER_MODE", _toggle_paste_layer_mode)

        _toggle_alpha_mode()
        _toggle_paste_layer_mode()

        # Create Dataset Settings
        create_dataset_fields = {
            "Crop Images": [
                ("NUM_CROPS", "Num Crops (auto)", 2),
                ("LOCALIZED", "Localized crop (auto)", False),
            ],
            "Create Dataset Settings": [
                ("VAL_RATIO", "Validation Ratio", 0.05),
            ],
        }

        self.create_section(
            title="Create dataset",
            fields=create_dataset_fields,
            btn_parent=btn_bar,
            skip_vars=[
                (self.skip_creating_direction_dataset, "Skip Create Dataset"),
            ],
        )

        # Training
        training_fields = {
            "YOLO Training": [
                ("EPOCHS", "Num Training (epochs)", 50),
                ("SAVE_PERIOD", "Checkpoint Interval", 5),
                ("BATCH_SIZE", "YOLO Batch Size", "auto"),
                ("PRETRAINED_MODEL", "Pretrained Model", "yolo11n-obb"),
                ("DEVICE", "YOLO Device (auto/0/cpu)", "auto"),
                ("LR0", "LR0 (comma-separated)", DEFAULT_LR0),
                ("LRF", "LRF (comma-separated)", DEFAULT_LRF),
                ("ACCEPT_RESUME", "Accept Resume", True),
            ],
        }
        self.create_section(
            title="Training",
            fields=training_fields,
            btn_parent=btn_bar,
            skip_vars=[
                (self.skip_training, "Skip Training"),
            ],
        )

        # Analysis
        analysis_fields = {
            "Detection": [
                ("DEVICE", "Device (auto/0/cpu)", "auto"),
                ("BATCH_SIZE", "Batch Size", "auto"),
                ("WEIGHT", "Checkpoints", "last"),
                ("CONF", "Confidence Threshold", 0.1),
                ("NMS_IOU", "NMS IoU", 0.80),
                ("SKIP_DETECT_PREVIEW", "Skip Preview", False),
            ],
            "Tracking": [
                ("MATCH_IOU", "Tracking OBB IoU (auto)", 0.50),
                ("MATCH_ANGLE", "Strong Match Direction (deg)", 90.0),
                ("MAX_AXIS_ERR", "Max Axis Error (deg)", 45.0),
                ("MAX_AGE", "Max Age (frames)", 10),
                ("FLIP_SEC", "Flip Duration (sec)", 5.0),
                ("INTERACT_IOU", "Interact OBB IoU (fixed)", FIXED_INTERACT_IOU),
                ("IOU_WEIGHT", "IoU Weight", 1.0),
                ("DIRECTION_WEIGHT", "Direction Weight", 1.0),
                ("MISS_WEIGHT", "Miss Weight", 1.0),
                ("DISTANCE_WEIGHT", "Distance Weight", 1.0),
            ],
            "Embedding": [
                ("ENABLE", "Enable", False),
                # Internal widget key only: keep it distinct from analysis.DEVICE.
                ("EMBED_DEVICE", "Device (auto/0/cpu)", "auto"),
                ("IMG_SIZE", "Image Size (px or auto)", "auto"),
                ("PREVIEW_COUNT", "Preview Count", 100),
            ],
        }
        self.create_section(
            title="Analysis",
            fields=analysis_fields,
            btn_parent=btn_bar,
            skip_vars=[
                (self.skip_detection, "Skip Object Detection"),
                (self.skip_id_tracking, "Skip ID Tracking"),
                (self.skip_id_correction, "Skip ID Correction"),
            ],
        )
        self.sections["Analysis"]["widgets"]["INTERACT_IOU"].configure(state="disabled")

        # Create Video
        create_video_fields = {
            "Draw Elements": [
                ("DRAW_OBB", "Draw OBB", True),
                ("DRAW_MODE", "Drawing Mode", "obb"),
                ("DRAW_LABELS", "Draw Labels", False),
                ("DRAW_ARROW", "Draw Direction Triangle", True),
            ],
            "Video": [
                ("WEIGHT", "Checkpoints", "last"),
                ("FRAME_STEP", "Frame Step", 1),
                ("FPS", "FPS (blank=auto)", ""),
                ("ACCELERATION", "Video Acceleration", "cpu"),
                ("EXPORT_RAW", "Export RAW YOLO Video", False),
                ("EXPORT_IMAGES", "Export Images", False),
                ("IMAGE_FORMAT", "Export Image Format", "jpeg"),
            ],
            "Appearance": [
                ("OBB_WIDTH", "OBB Line Thickness", 2),
                ("ARROW_WIDTH", "Triangle Line Thickness", 0),
                ("ARROW_ALPHA", "Triangle Alpha", 0.6),
                ("ARROW_SCALE", "Triangle Scale", 1.2),
                ("LABEL_SCALE", "Label Font Scale", 0.5),
                ("LABEL_THICKNESS", "Label Thickness", 1),
            ],
        }
        self.create_section(
            title="Create Video",
            fields=create_video_fields,
            btn_parent=btn_bar,
            skip_vars=[
                (self.skip_creating_video, "Skip Creating Video"),
            ],
        )
        self._add_create_video_color_control()

        create_video_widgets = self.sections["Create Video"]["widgets"]
        export_images_var = create_video_widgets["EXPORT_IMAGES"]
        image_export_format_widget = create_video_widgets["IMAGE_FORMAT"]

        def _toggle_export_images(*_):
            image_export_format_widget.configure(state=("readonly" if export_images_var.get() else "disabled"))

        self._toggle_export_images = _toggle_export_images
        export_images_var.trace_add("write", _toggle_export_images)
        _toggle_export_images()

        self._apply_prefill_paths()
        _start_maximized(self)

    def open_results(self):
        try:
            open_results_directory(self.basic_entries["SESSION_PATH"].get())
        except (OSError, ValueError) as exc:
            messagebox.showwarning("Open results", str(exc), parent=self)

    def _add_create_video_color_control(self):
        """Add the Create Video action that changes the persisted color seed."""
        appearance = self.sections["Create Video"]["category_parents"]["Appearance"]
        body = appearance["body"]
        row = _layout_frame(body)
        row.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 2))

        self._shuffle_colors_button = ctk.CTkButton(
            row,
            text="Shuffle Colors",
            width=160,
            command=self._shuffle_create_video_colors,
        )
        self._shuffle_colors_button.pack(side="left")
        self._color_seed_status = ctk.CTkLabel(row, text="", anchor="w")
        self._color_seed_status.pack(side="left", padx=(10, 0))
        self._update_create_video_color_status()

    def _shuffle_create_video_colors(self):
        self._create_video_color_seed = secrets.randbits(32)
        self._update_create_video_color_status()

    def _update_create_video_color_status(self):
        seed = self._create_video_color_seed
        if seed in (None, ""):
            status = "Using default color order"
        else:
            status = f"Color seed: {seed}"
        self._color_seed_status.configure(text=status)

    def _on_combo_selected(self, key: str):
        for cb in self._combo_callbacks.get(key, []):
            cb()

    def _bind_combo(self, key: str, func):
        self._combo_callbacks.setdefault(key, []).append(func)

    def _default_segmentation_project_dir(self, video_path: str) -> str:
        if not video_path:
            return ""
        video_dir = os.path.dirname(video_path)
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(video_dir, f"amadeus_{video_stem}")

    def _sync_tracking_video_path_if_empty(self):
        tracking = self.basic_entries["TRACKING_VIDEO_PATH"].get().strip()
        if not tracking:
            training = self.basic_entries["TRAINING_VIDEO_PATH"].get().strip()
            if training:
                self.basic_entries["TRACKING_VIDEO_PATH"].delete(0, tk.END)
                self.basic_entries["TRACKING_VIDEO_PATH"].insert(0, training)

    def _on_training_video_path_committed(self, _event=None):
        self._sync_tracking_video_path_if_empty()
        current = self.basic_entries["TRAINING_VIDEO_PATH"].get().strip()
        force_metadata = bool(current and current != self._loaded_training_video_path)
        self._auto_fill_from_video()
        if force_metadata:
            self._apply_segmentation_metadata_to_training_fields(force=True)
        return "break"

    def _apply_prefill_paths(self):
        session_path = self._prefill_session_path.strip()
        training_video_path = self._prefill_training_video_path.strip()
        tracking_video_path = self._prefill_tracking_video_path.strip()
        if not session_path and training_video_path:
            session_path = self._default_segmentation_project_dir(training_video_path)
        if session_path:
            self.basic_entries["SESSION_PATH"].delete(0, tk.END)
            self.basic_entries["SESSION_PATH"].insert(0, session_path)
        if training_video_path:
            self.basic_entries["TRAINING_VIDEO_PATH"].delete(0, tk.END)
            self.basic_entries["TRAINING_VIDEO_PATH"].insert(0, training_video_path)
        if tracking_video_path:
            self.basic_entries["TRACKING_VIDEO_PATH"].delete(0, tk.END)
            self.basic_entries["TRACKING_VIDEO_PATH"].insert(0, tracking_video_path)
        self.after(100, self._auto_fill_from_video)

    @staticmethod
    def _snap_to_nearest_multiple_of_32(value: int, min_value: int = 32, max_value: int = 8192) -> int:
        value = max(min_value, min(max_value, int(value)))
        lower = (value // 32) * 32
        upper = lower if value % 32 == 0 else lower + 32
        lower = max(min_value, lower)
        upper = min(max_value, upper)
        if abs(value - lower) <= abs(upper - value):
            return lower
        return upper

    def _normalize_training_image_size(self, *_):
        widget = self.basic_entries.get("TRAIN_IMG_SIZE")
        if widget is None:
            return

        raw = widget.get().strip()
        if not raw:
            snapped = 640
        else:
            try:
                snapped = self._snap_to_nearest_multiple_of_32(int(raw))
            except ValueError:
                snapped = 640

        widget.delete(0, tk.END)
        widget.insert(0, str(snapped))
        if hasattr(self, "skip_cropping"):
            self._apply_current_dataset_geometry_flags()

    def _setup_training_image_size_widget(self, widget):
        def _allow_training_image_size_input(proposed: str) -> bool:
            return proposed == "" or proposed.isdigit()

        vcmd = (self.register(_allow_training_image_size_input), "%P")
        widget.config(validate="key", validatecommand=vcmd)
        widget.bind("<FocusOut>", self._normalize_training_image_size, add="+")
        widget.bind("<Return>", self._normalize_training_image_size, add="+")
        widget.bind("<KP_Enter>", self._normalize_training_image_size, add="+")

    @staticmethod
    def _get_video_dimensions(video_path: str) -> tuple[int, int]:
        """Return (width, height) of a video file, or (0, 0) on failure."""
        try:
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        return 0, 0

    @staticmethod
    def _image_size_from_short_side(short_side: int) -> int:
        if short_side >= 1200:
            return 1024
        return max(32, (short_side // 32) * 32)

    def _derive_dataset_geometry_flags(self, video_path: str, image_size: int) -> tuple[bool, int, bool]:
        try:
            image_size = max(32, int(image_size))
        except (TypeError, ValueError):
            image_size = 640

        w, h = self._get_video_dimensions(video_path)
        if w <= 0 or h <= 0:
            return False, 2, True

        short_side = min(w, h)
        long_side = max(w, h)
        is_square = (w == h)

        if is_square and 0 <= short_side - image_size < 32:
            return True, 1, True
        if is_square and short_side <= image_size:
            return True, 1, True

        tiles_long = max(1, long_side // image_size)
        tiles_short = max(1, short_side // image_size)
        tile_count = max(1, min(4, tiles_long * tiles_short))
        num_crops = max(1, math.ceil(tile_count / 2))
        include_full_resized = short_side >= 1200
        return False, num_crops, include_full_resized

    def _set_dataset_geometry_fields(self, skip_crop: bool, num_crops: int, include_full: bool) -> None:
        self.skip_cropping.set(bool(skip_crop))
        self.include_full_resized.set(bool(include_full))

        dataset_widgets = self.sections.get("Create dataset", {}).get("widgets", {})
        crop_w = dataset_widgets.get("NUM_CROPS")
        if crop_w is not None:
            crop_w.delete(0, tk.END)
            crop_w.insert(0, str(int(num_crops)))
            if "NUM_CROPS" in self.default_value_markers:
                self.default_value_markers["NUM_CROPS"]["default"] = int(num_crops)

        self._refresh_default_markers()

    def _apply_current_dataset_geometry_flags(self) -> None:
        video_path = self.basic_entries["TRAINING_VIDEO_PATH"].get().strip()
        if not video_path:
            return
        try:
            image_size = int(self.basic_entries["TRAIN_IMG_SIZE"].get().strip())
        except (TypeError, ValueError):
            return
        skip_crop, num_crops, include_full = self._derive_dataset_geometry_flags(video_path, image_size)
        localize_w = self.sections.get("Create dataset", {}).get("widgets", {}).get("LOCALIZED")
        if isinstance(localize_w, tk.Variable) and bool(localize_w.get()):
            num_crops = 1
        self._set_dataset_geometry_fields(skip_crop, num_crops, include_full)

    def _auto_fill_from_video(self):
        video_path = self.basic_entries["TRAINING_VIDEO_PATH"].get().strip()
        if not video_path or not os.path.isfile(video_path):
            return
        w, h = self._get_video_dimensions(video_path)
        if w <= 0 or h <= 0:
            return

        short_side = min(w, h)
        img_size = self._image_size_from_short_side(short_side)
        size_w = self.basic_entries["TRAIN_IMG_SIZE"]
        size_w.delete(0, tk.END)
        size_w.insert(0, str(img_size))

        self._set_dataset_geometry_fields(*self._derive_dataset_geometry_flags(video_path, img_size))
        self._apply_segmentation_metadata_to_training_fields(force=False)

    def _apply_segmentation_metadata_to_training_fields(self, *, force: bool = False) -> None:
        if self._loaded_config_path and not force:
            return
        video_path = self.basic_entries["TRAINING_VIDEO_PATH"].get().strip()
        if not video_path:
            return
        meta = read_segmentation_metadata(segmentation_pickle_path_for_video(video_path))
        if not meta:
            return

        def _set_widget_value(widget, value) -> None:
            widget.delete(0, tk.END)
            widget.insert(0, str(int(value)))

        _set_widget_value(self.training_range["start"], meta["training_frame_start"])
        _set_widget_value(self.training_range["end"], meta["training_frame_end"])
        _set_widget_value(self.basic_entries["FRAME_INTERVAL"], meta["training_frame_interval"])
        self._loaded_training_video_path = video_path
        self._refresh_default_markers()

    @staticmethod
    def _is_numeric_default(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _numeric_equal_to_default(current: str, default) -> bool:
        try:
            return abs(float(str(current).strip()) - float(default)) <= 1e-9
        except Exception:
            return False

    def _auto_params_enabled(self) -> bool:
        var = self.sections.get("Initial Tracking", {}).get("widgets", {}).get("AUTO_PARAMS")
        if var is None:
            return True
        try:
            return bool(var.get())
        except Exception:
            return True

    def _values_match(self, current: str, value) -> bool:
        if self._is_numeric_default(value):
            return self._numeric_equal_to_default(current, value)
        return str(current).strip() == str(value).strip()

    def _capture_auto_param_values(self, cfg: dict) -> None:
        """Record the six AUTO_PARAMS-derived values from a cfg dict, so a
        manually-edited auto field's red marker can restore the measured
        value later (not just the unrelated static default). Only meaningful
        when the cfg was produced with AUTO_PARAMS enabled -- otherwise its
        field values are not reliably the auto-derived ones.
        """
        if not bool(cfg.get("AUTO_PARAMS", True)):
            return
        analysis_cfg = cfg.get("analysis", {}) or {}
        candidates = {
            "LOCALIZED": cfg.get("LOCALIZED"),
            "NUM_CROPS": cfg.get("NUM_CROPS"),
            "FREE_SCALE": cfg.get("FREE_SCALE"),
            "DIR_MIN_SEC": cfg.get("DIR_MIN_SEC"),
            "CLUSTER_FRAMES": cfg.get("CLUSTER_FRAMES"),
            "MATCH_IOU": analysis_cfg.get("MATCH_IOU"),
        }
        for key, value in candidates.items():
            if value is not None:
                self._auto_param_values[key] = value

    def _update_default_marker(self, key: str):
        marker_info = self.default_value_markers.get(key)
        if marker_info is None:
            return
        widget = marker_info["widget"]
        marker = marker_info["marker"]
        default = marker_info["default"]
        current = widget.get()

        if key in _AUTO_ADJUSTED_KEYS and self._auto_params_enabled():
            # AUTO_PARAMS overwrites this value at run time regardless of what
            # is typed here. Blue means the field still matches the last
            # measured auto value (or no measurement has happened yet); red
            # means it was manually edited away from that measured value.
            auto_value = self._auto_param_values.get(key)
            if auto_value is None or self._values_match(current, auto_value):
                marker.configure(text="*", fg=_AUTO_MARKER_COLOR)
            else:
                marker.configure(text="*", fg=_CHANGED_MARKER_COLOR)
            return

        is_default = self._values_match(current, default)
        marker.configure(text="" if is_default else "*", fg=_CHANGED_MARKER_COLOR)

    def _refresh_default_markers(self):
        for key in list(self.default_value_markers.keys()):
            self._update_default_marker(key)

    def _register_default_marker(self, key: str, widget, marker, default):
        self.default_value_markers[key] = {
            "widget": widget,
            "marker": marker,
            "default": default,
        }
        if isinstance(widget, ctk.CTkComboBox):
            self._bind_combo(key, lambda k=key: self._update_default_marker(k))
        elif isinstance(widget, tk.BooleanVar):
            widget.trace_add("write", lambda *_, k=key: self._update_default_marker(k))
        else:
            sv = getattr(widget, '_marker_var', None)
            if sv is not None:
                sv.trace_add("write", lambda *_, k=key: self._update_default_marker(k))
            else:
                widget.bind("<KeyRelease>", lambda _e, k=key: self._update_default_marker(k), add="+")
                widget.bind("<FocusOut>", lambda _e, k=key: self._update_default_marker(k), add="+")
                widget.bind("<Return>", lambda _e, k=key: self._update_default_marker(k), add="+")
                widget.bind("<KP_Enter>", lambda _e, k=key: self._update_default_marker(k), add="+")
        marker.bind("<Button-1>", lambda _e, k=key: self._reset_to_default(k))
        marker.configure(cursor="hand2")
        self._update_default_marker(key)

    def _reset_to_default(self, key: str):
        marker_info = self.default_value_markers.get(key)
        if marker_info is None:
            return
        widget = marker_info["widget"]
        default = marker_info["default"]
        try:
            state = str(widget.cget("state"))
            if state == "disabled":
                return
        except Exception:
            pass

        target = default
        if key in _AUTO_ADJUSTED_KEYS and self._auto_params_enabled():
            auto_value = self._auto_param_values.get(key)
            current = widget.get()
            if auto_value is not None and not self._values_match(current, auto_value):
                # Currently red (manually diverged from the measured auto
                # value): clicking restores that auto value, not the static
                # default. When already blue (at the auto value, or no
                # measurement yet), fall through to the static default below.
                target = auto_value

        if isinstance(widget, ctk.CTkComboBox):
            widget.set(str(target))
            self._on_combo_selected(key)
        elif isinstance(widget, tk.BooleanVar):
            widget.set(bool(target))
        else:
            widget.delete(0, tk.END)
            widget.insert(0, str(target))
        self._update_default_marker(key)

    def _make_row_marker(self, cell, ref_widget) -> tk.Label:
        try:
            colors = cell._fg_color
            raw = colors[1] if isinstance(colors, (list, tuple)) else colors
            rgb = cell.winfo_rgb(raw)
            bg = "#{:02x}{:02x}{:02x}".format(rgb[0] >> 8, rgb[1] >> 8, rgb[2] >> 8)
        except Exception:
            bg = "#2b2b2b"
        marker = tk.Label(cell, text="", fg=_CHANGED_MARKER_COLOR, bg=bg,
                          font=("TkDefaultFont", 9, "bold"), bd=0,
                          highlightthickness=0, padx=0, pady=0,
                          width=2)
        cell.grid_columnconfigure(2, minsize=marker.winfo_reqwidth() + 2, weight=0)
        marker.grid(row=0, column=2, sticky="w", padx=(2, 0))
        return marker

    def _set_parameter_row_visible(self, section_title: str, key: str, visible: bool):
        cell = self._parameter_rows.get((section_title, key))
        if cell is None:
            return
        if visible:
            cell.grid()
        else:
            cell.grid_remove()

    def _set_create_images_group_enabled(self, section_title, keys, enabled: bool):
        state = "normal" if enabled else "disabled"
        section_widgets = self.sections[section_title]["widgets"]
        for key in keys:
            w = section_widgets[key]
            if isinstance(w, tk.Variable):
                cb = getattr(w, "_checkbutton_ref", None)
                if cb is not None:
                    cb.configure(state=state)
                continue
            if isinstance(w, ctk.CTkComboBox):
                w.configure(state=("readonly" if enabled else "disabled"))
            else:
                w.configure(state=state)

    def create_section(self, *, title: str, fields: dict, btn_parent, skip_vars=None):
        btn = ctk.CTkButton(
            btn_parent,
            text=f"Show {title} >",
            fg_color="transparent",
            hover_color=("#d0d0d0", "#3a3a3a"),
            text_color=("black", "white"),
            border_width=1,
            corner_radius=4,
            width=120,
        )
        btn.pack(side="left", padx=3, pady=4)
        frame = _layout_frame(self._content_host.content)
        widgets = {}
        category_parents = {}

        if skip_vars:
            skip_frame = _layout_frame(frame)
            skip_frame.pack(fill="x", padx=10, pady=(10, 0))
            for var, label in skip_vars:
                cb = ctk.CTkCheckBox(skip_frame, text=label, variable=var)
                cb.pack(side="left", padx=10)
                self._skip_cb_by_var[id(var)] = cb

        for category, items in fields.items():
            row = _layout_frame(frame)
            row.pack(fill="x", pady=(8, 2))
            ctk.CTkLabel(row, text=category, anchor="w", font=("TkDefaultFont", 13, "bold")).pack(side="left", padx=(4, 0))
            sep = ctk.CTkFrame(row, height=2, fg_color=("gray60", "gray40"), corner_radius=0)
            sep.pack(side="left", fill="x", expand=True, padx=(8, 0))

            inline_bool_groups = {
                ("Initial Tracking", "Initial Tracking"): {"SKIP_INIT_PREVIEW"},
                ("Create single animal images", "Trajectory / Direction Filtering"): {"SKIP_DIR_PREVIEW"},
                ("Create single animal images", "Refine blobs through training"): {"SKIP_REFINE_PREVIEW"},
                ("Analysis", "Detection"): {"SKIP_DETECT_PREVIEW"},
            }

            want_inline = inline_bool_groups.get((title, category), set())
            inline_bar = _layout_frame(row) if want_inline else None
            if inline_bar:
                inline_bar.pack(side="right", padx=(8, 0))

            column_items = partition_category_items(title, category, items)
            category_body = _layout_frame(frame)
            for column_index in range(3):
                category_body.grid_columnconfigure(
                    column_index,
                    weight=1,
                    uniform="advanced_parameter_column",
                )
            category_body.pack(fill="x", pady=2, padx=25)

            category_parents[category] = {
                "column_count": 3,
                "columns": column_items,
                "body": category_body,
                "frame": frame,
            }

            for column_index, column in enumerate(column_items):
                row_index = 0
                for key, label_text, default in column:

                    if isinstance(default, bool):
                        var = tk.BooleanVar(value=default)
                        widgets[key] = var

                        if key == "AUTO_PARAMS":
                            var.trace_add("write", lambda *_: self._refresh_default_markers())

                        if key in want_inline and inline_bar is not None:
                            ctk.CTkCheckBox(inline_bar, text=label_text, variable=var).pack(side="left", padx=12)
                            continue

                        cell = _layout_frame(category_body)
                        cell.grid(row=row_index, column=column_index, sticky="ew", pady=2)
                        self._parameter_rows[(title, key)] = cell
                        cell.grid_columnconfigure(0, minsize=180, weight=0)
                        cell.grid_columnconfigure(1, weight=0)
                        cb = ctk.CTkCheckBox(cell, text=label_text, variable=var)
                        cb.grid(row=0, column=0, columnspan=2, sticky="w")
                        try:
                            var._checkbutton_ref = cb
                        except Exception:
                            pass
                        _marker = self._make_row_marker(cell, cb)
                        if key in _AUTO_ADJUSTED_KEYS:
                            self._register_default_marker(key, var, _marker, default)
                    else:
                        cell = _layout_frame(category_body)
                        cell.grid(row=row_index, column=column_index, sticky="ew", pady=2)
                        self._parameter_rows[(title, key)] = cell
                        cell.grid_columnconfigure(0, minsize=180, weight=0)
                        cell.grid_columnconfigure(1, weight=0)
                        ctk.CTkLabel(cell, text=label_text, width=180, anchor="w").grid(
                            row=0,
                            column=0,
                            sticky="w",
                        )
                        _created_widget = None

                        if key == "CROSSING_MODE":
                            combo = ctk.CTkComboBox(cell, values=["pickle", "area", "hybrid"], state="readonly", width=130,
                                                    command=lambda _, k=key: self._on_combo_selected(k))
                            combo.set(str(default))
                            combo.grid(row=0, column=1, sticky="w")
                            widgets[key] = combo
                            _created_widget = combo
                        elif key == "OBB_FIT_MODE":
                            combo = ctk.CTkComboBox(cell, values=["min_area", "pca"], state="readonly", width=130,
                                                    command=lambda _, k=key: self._on_combo_selected(k))
                            combo.set(str(default))
                            combo.grid(row=0, column=1, sticky="w")
                            widgets[key] = combo
                            _created_widget = combo
                        elif key == "PASTE_LAYER_MODE":
                            combo = ctk.CTkComboBox(cell, values=["mixed", "over", "under"], state="readonly", width=130,
                                                    command=lambda _, k=key: self._on_combo_selected(k))
                            combo.set(str(default))
                            combo.grid(row=0, column=1, sticky="w")
                            widgets[key] = combo
                            _created_widget = combo
                        elif key == "ALPHA_MODE":
                            combo = ctk.CTkComboBox(cell, values=["distance", "gaussian"], state="readonly", width=130,
                                                    command=lambda _, k=key: self._on_combo_selected(k))
                            combo.set(str(default))
                            combo.grid(row=0, column=1, sticky="w")
                            widgets[key] = combo
                            _created_widget = combo
                        elif key == "IMAGE_FORMAT":
                            combo = ctk.CTkComboBox(cell, values=["png", "jpeg"], state="readonly", width=130,
                                                    command=lambda _, k=key: self._on_combo_selected(k))
                            combo.set(str(default))
                            combo.grid(row=0, column=1, sticky="w")
                            widgets[key] = combo
                            _created_widget = combo
                        elif key == "DRAW_MODE":
                            combo = ctk.CTkComboBox(cell, values=["obb", "ellipse"], state="readonly", width=130,
                                                    command=lambda _, k=key: self._on_combo_selected(k))
                            combo.set(str(default))
                            combo.grid(row=0, column=1, sticky="w")
                            widgets[key] = combo
                            _created_widget = combo
                        elif key == "ACCELERATION":
                            combo = ctk.CTkComboBox(cell, values=["cpu", "gpu", "auto"], state="readonly", width=130,
                                                    command=lambda _, k=key: self._on_combo_selected(k))
                            combo.set(str(default))
                            combo.grid(row=0, column=1, sticky="w")
                            widgets[key] = combo
                            _created_widget = combo
                        else:
                            _sv = tk.StringVar(value=str(default))
                            ent = ctk.CTkEntry(cell, width=120, textvariable=_sv)
                            ent.grid(row=0, column=1, sticky="w")
                            ent._marker_var = _sv
                            widgets[key] = ent
                            _created_widget = ent

                        if _created_widget is not None:
                            _marker = self._make_row_marker(cell, _created_widget)
                            self._register_default_marker(key, _created_widget, _marker, default)
                    row_index += 1

        self.sections[title] = {
            "btn": btn,
            "frame": frame,
            "visible": False,
            "widgets": widgets,
            "category_parents": category_parents,
        }
        btn.configure(command=lambda t=title: self.toggle_section(t))

    def toggle_section(self, title):
        for t, sec in self.sections.items():
            if t == title:
                if sec["visible"]:
                    sec["frame"].pack_forget()
                    sec["btn"].configure(text=f"Show {t} >")
                    sec["visible"] = False
                else:
                    sec["frame"].pack(fill="x", padx=10)
                    sec["btn"].configure(text=f"Hide {t} v")
                    sec["visible"] = True
            else:
                if sec["visible"]:
                    sec["frame"].pack_forget()
                    sec["btn"].configure(text=f"Show {t} >")
                    sec["visible"] = False

    def _browse_analysis_video(self, start_dir: str, title: str) -> str | None:
        """Pick a video, converting it first when AMADEUS cannot read it directly."""
        prepared = ask_open_analysis_video(
            self,
            title=title,
            initialdir=start_dir if os.path.isdir(start_dir) else None,
            log=lambda message: print(f"[video] {message}", flush=True),
        )
        return prepared.path if prepared is not None else None

    def browse_path(self, key):
        initial = self.basic_entries[key].get() or os.getcwd()

        if key == "SESSION_PATH":
            if os.path.isdir(initial):
                path = filedialog.askdirectory(initialdir=initial)
            else:
                path = filedialog.askdirectory(initialdir=os.path.dirname(initial))
        elif key == "TRAINING_VIDEO_PATH":
            start_dir = os.path.dirname(initial) if os.path.exists(initial) else initial
            path = self._browse_analysis_video(start_dir, "Select training video")
        elif key == "TRACKING_VIDEO_PATH":
            is_dir = self.path_mode["TRACKING_VIDEO_PATH_IS_DIR"].get()
            if is_dir:
                start_dir = initial if os.path.isdir(initial) else os.path.dirname(initial)
                path = filedialog.askdirectory(initialdir=start_dir)
            else:
                start_dir = os.path.dirname(initial) if os.path.exists(initial) else initial
                path = self._browse_analysis_video(start_dir, "Select tracking video")
        else:
            path = None

        if path:
            self.basic_entries[key].delete(0, tk.END)
            self.basic_entries[key].insert(0, path)
            if key == "TRAINING_VIDEO_PATH":
                self._sync_tracking_video_path_if_empty()
                self._auto_fill_from_video()
                self._apply_segmentation_metadata_to_training_fields(force=True)

    def load_yaml(self):
        file_path = filedialog.askopenfilename(
            title="Select YAML file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not file_path:
            return
        self._do_load_yaml(file_path, show_msg=True)

    def _do_load_yaml(self, file_path: str, show_msg: bool = True):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YAML file:\n{e}")
            return False

        self._loaded_config_path = file_path
        cfg = prepare_config_for_gui(cfg or {}, file_path, parent=self)
        # PICKLE_PATH/BACKGROUND_PATH always live under SESSION_PATH; always
        # recompute rather than trust whatever an old config.yaml stored, so
        # a stale value never survives a load.
        cfg["PICKLE_PATH"], cfg["BACKGROUND_PATH"] = segmentation_paths_for_session(
            cfg.get("SESSION_PATH", ""),
            cfg.get("TRAINING_VIDEO_PATH", ""),
        )
        embedding_cfg = cfg.setdefault("EMBEDDING", {})
        if not isinstance(embedding_cfg, dict):
            messagebox.showerror("Error", "EMBEDDING must be a mapping.")
            return False
        embedding_cfg["INTERACT_IOU"] = FIXED_INTERACT_IOU
        self._loaded_cfg = copy.deepcopy(cfg)
        self._loaded_training_video_path = str(cfg.get("TRAINING_VIDEO_PATH", "") or "")

        def _set_widget_value(w, v):
            if v is None:
                return
            if isinstance(v, list):
                v = ",".join(str(x) for x in v)
            if isinstance(w, tk.Variable):
                try:
                    w.set(v)
                except Exception:
                    w.set(str(v))
                return
            if isinstance(w, ctk.CTkComboBox):
                candidate = str(v).strip()
                try:
                    values = list(w.cget("values"))
                    if candidate not in values:
                        for val in values:
                            if val.lower() == candidate.lower():
                                candidate = val
                                break
                except Exception:
                    pass
                try:
                    current_state = str(w.cget("state"))
                    if current_state == "disabled":
                        w.configure(state="normal")
                        w.set(candidate)
                        w.configure(state="disabled")
                    else:
                        w.set(candidate)
                except Exception:
                    w.set(candidate)
                return
            restore_state = None
            try:
                current_state = str(w.cget("state"))
                if current_state in {"disabled", "readonly"}:
                    restore_state = current_state
                    try:
                        w.configure(state="normal")
                    except Exception:
                        w.config(state="normal")
            except Exception:
                pass
            try:
                w.delete(0, tk.END)
                w.insert(0, str(v))
            finally:
                if restore_state is not None:
                    try:
                        w.configure(state=restore_state)
                    except Exception:
                        try:
                            w.config(state=restore_state)
                        except Exception:
                            pass

        for key in ("SESSION_PATH", "TRACKING_VIDEO_PATH", "TRAINING_VIDEO_PATH", "NUM_OBJECTS", "TRAIN_IMG_SIZE", "NUM_WORKERS", "NUM_PREVIEW_FRAMES", "PREVIEW_INTERVAL", "FRAME_INTERVAL", "NUM_IMAGES", "RANDOM_SEED"):
            val = cfg.get(key)
            if val is not None and key in self.basic_entries:
                _set_widget_value(self.basic_entries[key], val)

        self.variable_count.set(bool(cfg.get("VARIABLE_NUM_OBJECTS", False)))
        self.without_direction.set(bool(cfg.get("WITHOUT_DIRECTION_ESTIMATION", False)))
        self._toggle_variable_count()
        self.path_mode["TRACKING_VIDEO_PATH_IS_DIR"].set(cfg.get("TRACKING_VIDEO_PATH_IS_DIR", False))

        for k, w in self.sections["Initial Tracking"]["widgets"].items():
            if k in cfg:
                _set_widget_value(w, cfg[k])

        for section_name in ("Create single animal images", "Create with crossing"):
            for k, w in self.sections[section_name]["widgets"].items():
                if k in cfg:
                    _set_widget_value(w, cfg[k])

        if hasattr(self, "_toggle_without_crossing_mode"):
            self._toggle_without_crossing_mode()
        if hasattr(self, "_toggle_with_crossing_alpha_mode"):
            self._toggle_with_crossing_alpha_mode()
        if hasattr(self, "_toggle_with_crossing_paste_layer_mode"):
            self._toggle_with_crossing_paste_layer_mode()
        if hasattr(self, "_toggle_duplicate_with_crossing_group"):
            self._toggle_duplicate_with_crossing_group()
        for k, w in self.sections["Create dataset"]["widgets"].items():
            if k in cfg:
                _set_widget_value(w, cfg[k])

        train_cfg = cfg.get("training", {})
        for k, w in self.sections["Training"]["widgets"].items():
            if k in train_cfg:
                _set_widget_value(w, train_cfg[k])
        if "FIRST_FRAME" in train_cfg:
            _set_widget_value(self.training_range["start"], train_cfg["FIRST_FRAME"])
        if "LAST_FRAME" in train_cfg:
            _set_widget_value(self.training_range["end"], train_cfg["LAST_FRAME"])

        ana_cfg = dict(cfg.get("analysis", {}) or {})
        _tc_flat = dict(ana_cfg.pop("TRACKING_COST", {}) or {})
        ana_cfg.update(_tc_flat)
        for k, w in self.sections["Analysis"]["widgets"].items():
            if k in ana_cfg:
                _set_widget_value(w, ana_cfg[k])
        if "FIRST_FRAME" in ana_cfg:
            _set_widget_value(self.analysis_range["start"], ana_cfg["FIRST_FRAME"])
        if "LAST_FRAME" in ana_cfg:
            _set_widget_value(self.analysis_range["end"], ana_cfg["LAST_FRAME"])
        if hasattr(self, "_toggle_tracking_mode"):
            self._toggle_tracking_mode()

        emb_cfg = cfg.get("EMBEDDING", {}) or {}
        _emb_widget_keys = {
            "ENABLE": "ENABLE",
            "DEVICE": "EMBED_DEVICE",
            "IMG_SIZE": "IMG_SIZE",
            "PREVIEW_COUNT": "PREVIEW_COUNT",
            "INTERACT_IOU": "INTERACT_IOU",
        }
        analysis_widgets = self.sections["Analysis"]["widgets"]
        for config_key, widget_key in _emb_widget_keys.items():
            if config_key in emb_cfg:
                _set_widget_value(analysis_widgets[widget_key], emb_cfg[config_key])

        create_video_cfg = cfg.get("create_video", {})
        self._create_video_color_seed = create_video_cfg.get("COLOR_SEED")
        self._update_create_video_color_status()
        create_video_widgets = self.sections["Create Video"]["widgets"]

        export_images_value = create_video_cfg.get("EXPORT_IMAGES")
        if export_images_value is not None:
            _set_widget_value(create_video_widgets["EXPORT_IMAGES"], export_images_value)

        if hasattr(self, "_toggle_export_images"):
            self._toggle_export_images()

        for k, w in create_video_widgets.items():
            if k == "EXPORT_IMAGES":
                continue
            if k in create_video_cfg:
                _set_widget_value(w, create_video_cfg[k])


        self.skip_initial_tracking.set(cfg.get("skip_initial_tracking", False))
        self.skip_trajectory_direction_filtering.set(cfg.get("skip_trajectory_direction_filtering", False))
        self.skip_refine_blobs_through_tracking.set(cfg.get("skip_refine_blobs_through_tracking", False))
        self.skip_paste_blobs_with_crossing.set(cfg.get("skip_paste_blobs_with_crossing", False))
        self.skip_paste_blobs_clustered.set(cfg.get("skip_paste_blobs_clustered", False))
        self.skip_cropping.set(cfg.get("skip_cropping", False))
        self.include_full_resized.set(cfg.get("USE_FULL", True))
        self.skip_creating_direction_dataset.set(cfg.get("skip_creating_direction_dataset", False))
        self.skip_training.set(cfg.get("skip_training", False))
        self.skip_detection.set(cfg.get("skip_detection", False))
        self.skip_id_tracking.set(cfg.get("skip_id_tracking", False))
        self.skip_id_correction.set(cfg.get("skip_id_correction", False))
        self.skip_creating_video.set(cfg.get("skip_creating_video", False))
        self.delete_tmp_files.set(cfg.get("delete_tmp_files", False))

        self._capture_auto_param_values(cfg)
        self._refresh_default_markers()
        if show_msg:
            messagebox.showinfo("Loaded", f"Settings loaded from:\n{file_path}")
        return True

    def _toggle_variable_count(self):
        self.basic_entries["NUM_OBJECTS"].configure(
            state="disabled" if self.variable_count.get() else "normal")

    def save_yaml(self, print_msg=True) -> bool:
        cfg = copy.deepcopy(self._loaded_cfg)
        for k, ent in self.basic_entries.items():
            val_str = ent.get().strip()
            if k == "NUM_OBJECTS":
                if self.variable_count.get():
                    cfg[k] = 2  # synthesis placeholder; never a tracking population limit
                    continue
                if not val_str.isdigit():
                    messagebox.showerror("Error", "Number of Objects must be a positive integer.")
                    return
                cfg[k] = int(val_str)
            elif k == "TRAIN_IMG_SIZE":
                if not val_str.isdigit():
                    messagebox.showerror("Error", "Training Image Size (px) must be a positive integer.")
                    return
                training_image_size = int(val_str)
                if training_image_size <= 0 or training_image_size % 32 != 0:
                    messagebox.showerror("Error", "Training Image Size (px) must be a positive multiple of 32.")
                    return
                cfg[k] = training_image_size
            elif k in ("NUM_PREVIEW_FRAMES", "PREVIEW_INTERVAL", "FRAME_INTERVAL", "NUM_IMAGES", "RANDOM_SEED"):
                defaults = {"NUM_PREVIEW_FRAMES": 10, "PREVIEW_INTERVAL": 100, "FRAME_INTERVAL": 5, "NUM_IMAGES": 10000, "RANDOM_SEED": 0}
                try:
                    cfg[k] = int(val_str) if val_str else defaults[k]
                except ValueError:
                    cfg[k] = defaults[k]
            else:
                cfg[k] = val_str

        if not cfg["SESSION_PATH"]:
            messagebox.showerror("Error", "Session path is required.")
            return
        if not cfg.get("TRACKING_VIDEO_PATH"):
            messagebox.showerror("Error", "Tracking video path is required.")
            return

        cfg["TRACKING_VIDEO_PATH_IS_DIR"] = self.path_mode["TRACKING_VIDEO_PATH_IS_DIR"].get()

        for key, widget in self.sections["Initial Tracking"]["widgets"].items():
            val = widget.get() if isinstance(widget, tk.Variable) else widget.get().strip()
            if val != "":
                cfg[key] = self._cast(key, val)

        localized_ratio = float(cfg.get("LOCALIZED_RATIO", 0.9))
        if localized_ratio <= 0.0 or localized_ratio > 1.0:
            messagebox.showerror("Error", "LOCALIZED_RATIO must be greater than 0 and at most 1.")
            return

        for section_name in ("Create single animal images", "Create with crossing"):
            cfg.update({k: self._cast(k, w.get()) for k, w in self.sections[section_name]["widgets"].items()})
        cfg["MIN_OVERLAP"] = 0.01

        if bool(cfg.get("NOISE_ENABLE", False)):
            noise_size_percent = cfg.get("NOISE_SIZE_PERCENT")
            noise_max_count = cfg.get("NOISE_MAX_COUNT")
            if noise_size_percent is None or float(noise_size_percent) <= 0.0:
                messagebox.showerror("Error", "Noise Size (%) must be a positive number.")
                return
            if noise_max_count is None or int(noise_max_count) < 1:
                messagebox.showerror("Error", "Max Noise per Animal must be at least 1.")
                return

        paste_scale_min = float(cfg.get("PASTE_SCALE_MIN", 1.0))
        paste_scale_max = float(cfg.get("PASTE_SCALE_MAX", 1.0))
        if paste_scale_min <= 0 or paste_scale_max <= 0:
            messagebox.showerror("Error", "Paste Scale Min/Max must be positive numbers.")
            return
        if paste_scale_min > paste_scale_max:
            messagebox.showerror("Error", "Paste Scale Min must be less than or equal to Paste Scale Max.")
            return
        for lo_key, hi_key, label in (
            ("WIDTH_SCALE_MIN", "WIDTH_SCALE_MAX", "Width Scale"),
        ):
            lo = float(cfg.get(lo_key, 0.95))
            hi = float(cfg.get(hi_key, 1.05))
            if lo <= 0.0 or hi <= 0.0:
                messagebox.showerror("Error", f"{label} Min/Max must be positive numbers.")
                return
            if lo > hi:
                messagebox.showerror("Error", f"{label} Min must be less than or equal to {label} Max.")
                return
        cfg["PASTE_MIN_ASPECT"] = 1.1

        for key in ("RATIO_SINGLE", "RATIO_P2", "RATIO_P3"):
            if float(cfg.get(key, 0.0)) < 0.0:
                messagebox.showerror("Error", f"{key} must be zero or a positive ratio.")
                return
        if (
            float(cfg.get("RATIO_SINGLE", 0.0))
            + float(cfg.get("RATIO_P2", 0.0))
            + float(cfg.get("RATIO_P3", 0.0))
        ) <= 0.0:
            messagebox.showerror("Error", "RATIO_SINGLE + RATIO_P2 + RATIO_P3 must be greater than zero.")
            return

        free_paste_scale = float(cfg.get("FREE_SCALE", 1.0))
        if free_paste_scale < 0.0:
            messagebox.showerror("Error", "FREE_SCALE must be zero or a positive number.")
            return
        for key in ("FREE_RATIO_SINGLE", "FREE_RATIO_P2", "FREE_RATIO_P3"):
            if float(cfg.get(key, 0.0)) < 0.0:
                messagebox.showerror("Error", f"{key} must be zero or a positive ratio.")
                return
        if free_paste_scale > 0.0 and (
            float(cfg.get("FREE_RATIO_SINGLE", 0.0))
            + float(cfg.get("FREE_RATIO_P2", 0.0))
            + float(cfg.get("FREE_RATIO_P3", 0.0))
        ) <= 0.0:
            messagebox.showerror(
                "Error",
                "FREE_RATIO_SINGLE + FREE_RATIO_P2 + FREE_RATIO_P3 must be > 0 when FREE_SCALE > 0.",
            )
            return

        cluster_count = int(cfg.get("CLUSTER_COUNT", 12))
        if cluster_count < 2:
            messagebox.showerror("Error", "CLUSTER_COUNT must be at least 2.")
            return
        cluster_fit_scale_long = float(cfg.get("CLUSTER_FIT_LONG", 0.8))
        cluster_fit_scale_short = float(cfg.get("CLUSTER_FIT_SHORT", 0.8))
        cluster_parallel_break_prob = float(cfg.get("CLUSTER_BREAK_PROB", 0.20))
        if cluster_fit_scale_long <= 0.0 or cluster_fit_scale_long > 1.0:
            messagebox.showerror("Error", "CLUSTER_FIT_LONG must be > 0.0 and <= 1.0.")
            return
        if cluster_fit_scale_short <= 0.0 or cluster_fit_scale_short > 1.0:
            messagebox.showerror("Error", "CLUSTER_FIT_SHORT must be > 0.0 and <= 1.0.")
            return
        if cluster_parallel_break_prob < 0.0 or cluster_parallel_break_prob > 1.0:
            messagebox.showerror("Error", "CLUSTER_BREAK_PROB must be >= 0.0 and <= 1.0.")
            return

        cfg.update({k: self._cast(k, w.get()) for k, w in self.sections["Create dataset"]["widgets"].items()})
        skip_crop, _, include_full = self._derive_dataset_geometry_flags(
            str(cfg.get("TRAINING_VIDEO_PATH", "")),
            int(cfg["TRAIN_IMG_SIZE"]),
        )
        if bool(cfg.get("LOCALIZED", False)):
            cfg["NUM_CROPS"] = 1
        cfg["skip_cropping"] = skip_crop
        cfg["USE_FULL"] = include_full
        cfg["USE_CROP"] = not skip_crop

        training_cfg = dict(cfg.get("training", {}) or {})
        training_cfg.update({k: self._cast(k, w.get()) for k, w in self.sections["Training"]["widgets"].items()})
        cfg["training"] = training_cfg
        try:
            lr0_values = parse_lr0_values(cfg["training"].get("LR0"))
            lrf_values = parse_lrf_values(cfg["training"].get("LRF"))
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return False
        cfg["training"]["LR0"] = lr0_values[0] if len(lr0_values) == 1 else lr0_values
        cfg["training"]["LRF"] = lrf_values[0] if len(lrf_values) == 1 else lrf_values
        try:
            training_epochs = _strict_int(cfg["training"].get("EPOCHS", 0))
        except (TypeError, ValueError):
            training_epochs = 0
        if training_epochs < 1:
            messagebox.showerror("Error", "Num Training must be an integer greater than or equal to 1.")
            return False
        cfg["training"]["EPOCHS"] = training_epochs
        save_period = cfg["training"].get("SAVE_PERIOD", 5)
        if isinstance(save_period, str) and save_period.strip().lower() == "best":
            cfg["training"]["SAVE_PERIOD"] = "best"
        else:
            try:
                save_period = 5 if save_period in ("", None) else _strict_int(save_period)
            except (TypeError, ValueError):
                messagebox.showerror("Error", "Save Period must be a positive integer, -1, or 'best'.")
                return False
            if save_period != -1 and save_period < 1:
                messagebox.showerror("Error", "Save Period must be a positive integer, -1, or 'best'.")
                return False
        cfg["training"]["SAVE_PERIOD"] = save_period
        analysis_cfg = dict(cfg.get("analysis", {}) or {})
        _emb_widget_keys = {
            "ENABLE": "ENABLE",
            "DEVICE": "EMBED_DEVICE",
            "IMG_SIZE": "IMG_SIZE",
            "PREVIEW_COUNT": "PREVIEW_COUNT",
            "INTERACT_IOU": "INTERACT_IOU",
        }
        _emb_internal_keys = set(_emb_widget_keys.values())
        analysis_widgets = self.sections["Analysis"]["widgets"]
        analysis_cfg.update({
            k: self._cast(k, w.get())
            for k, w in analysis_widgets.items()
            if k not in _emb_internal_keys
        })
        try:
            analysis_cfg["WEIGHT"] = checkpoint_spec_for_config(
                analysis_cfg.get("WEIGHT"), key="analysis.WEIGHT"
            )
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return False
        cfg["analysis"] = analysis_cfg
        try:
            flip_duration_sec = float(cfg["analysis"]["FLIP_SEC"])
        except (KeyError, TypeError, ValueError):
            flip_duration_sec = float("nan")
        if flip_duration_sec <= 0.0 or not math.isfinite(flip_duration_sec):
            messagebox.showerror("Error", "FLIP_SEC must be a positive number of seconds.")
            return
        _tc_keys = ("IOU_WEIGHT", "DIRECTION_WEIGHT", "MISS_WEIGHT", "DISTANCE_WEIGHT")
        tracking_cost_dict = dict(cfg["analysis"].get("TRACKING_COST", {}) or {})
        for _k in _tc_keys:
            if _k in cfg["analysis"]:
                _v = cfg["analysis"].pop(_k)
                if _v is not None:
                    tracking_cost_dict[_k] = _v
        if tracking_cost_dict:
            cfg["analysis"]["TRACKING_COST"] = tracking_cost_dict
        embedding_dict = dict(cfg.get("EMBEDDING", {}) or {})
        for config_key, widget_key in _emb_widget_keys.items():
            embedding_dict[config_key] = self._cast(
                config_key,
                analysis_widgets[widget_key].get(),
            )
        embedding_dict["INTERACT_IOU"] = FIXED_INTERACT_IOU
        cfg["EMBEDDING"] = embedding_dict
        create_video_cfg = dict(cfg.get("create_video", {}) or {})
        create_video_cfg.update({k: self._cast(k, w.get()) for k, w in self.sections["Create Video"]["widgets"].items()})
        if self._create_video_color_seed in (None, ""):
            create_video_cfg.pop("COLOR_SEED", None)
        else:
            try:
                create_video_cfg["COLOR_SEED"] = int(self._create_video_color_seed)
            except (TypeError, ValueError):
                messagebox.showerror("Error", "Color seed must be an integer.")
                return False
        try:
            create_video_cfg["WEIGHT"] = checkpoint_spec_for_config(
                create_video_cfg.get("WEIGHT"), key="create_video.WEIGHT"
            )
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return False
        cfg["create_video"] = create_video_cfg
        def _int_or_default(s, default):
            try:
                return int(s)
            except Exception:
                return default

        cfg["training"]["FIRST_FRAME"] = _int_or_default(self.training_range["start"].get(), 0)
        cfg["training"]["LAST_FRAME"] = _int_or_default(self.training_range["end"].get(), -1)
        cfg["analysis"]["FIRST_FRAME"] = _int_or_default(self.analysis_range["start"].get(), 0)
        cfg["analysis"]["LAST_FRAME"] = _int_or_default(self.analysis_range["end"].get(), -1)
        cfg["skip_initial_tracking"] = self.skip_initial_tracking.get()
        cfg["skip_trajectory_direction_filtering"] = self.skip_trajectory_direction_filtering.get()
        cfg["skip_refine_blobs_through_tracking"] = self.skip_refine_blobs_through_tracking.get()
        cfg["skip_paste_blobs_with_crossing"] = self.skip_paste_blobs_with_crossing.get()
        cfg["skip_paste_blobs_clustered"] = self.skip_paste_blobs_clustered.get()
        cfg["skip_cropping"] = skip_crop
        cfg["USE_FULL"] = include_full
        cfg["USE_CROP"] = not skip_crop
        cfg["skip_creating_direction_dataset"] = self.skip_creating_direction_dataset.get()
        cfg["skip_training"] = self.skip_training.get()
        cfg["skip_detection"] = self.skip_detection.get()
        cfg["skip_id_tracking"] = self.skip_id_tracking.get()
        cfg["skip_id_correction"] = self.skip_id_correction.get()
        cfg["skip_creating_video"] = self.skip_creating_video.get()
        cfg["delete_tmp_files"] = self.delete_tmp_files.get()
        cfg.setdefault("SINGLE_PASTE", False)

        cfg["VARIABLE_NUM_OBJECTS"] = self.variable_count.get()
        cfg["WITHOUT_DIRECTION_ESTIMATION"] = self.without_direction.get()
        if not cfg["VARIABLE_NUM_OBJECTS"] and int(cfg.get("NUM_OBJECTS", 1)) == 1 and not bool(cfg["SINGLE_PASTE"]):
            cfg["skip_paste_blobs_with_crossing"] = True
            cfg["skip_paste_blobs_clustered"] = True
            cfg["EMBEDDING"]["ENABLE"] = False

        session = cfg.get("SESSION_PATH", "")
        training_video_path = str(cfg.get("TRAINING_VIDEO_PATH", ""))
        # Always <session>/segmentation/, matching where segmentation itself
        # writes (see gui_segmentation.py's _default_output_dir).
        cfg["PICKLE_PATH"], cfg["BACKGROUND_PATH"] = segmentation_paths_for_session(session, training_video_path)
        cfg["INIT_CSV_PATH"] = os.path.join(session, "initial_tracking", "track_assignments.csv")

        fn = os.path.join(session, "config.yaml")
        try:
            with open(fn, "w", encoding="utf-8") as f:
                yaml.dump(relativize_config_paths(cfg), f, sort_keys=False, allow_unicode=True)
            self._loaded_config_path = fn
            self._loaded_cfg = copy.deepcopy(cfg)
            if print_msg:
                messagebox.showinfo("Saved", f"YAML saved to:\n{fn}")
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False

    def _existing_config_path(self) -> str:
        if self._loaded_config_path and os.path.isfile(self._loaded_config_path):
            return self._loaded_config_path
        session = self.basic_entries["SESSION_PATH"].get().strip()
        if session:
            candidate = os.path.join(session, "config.yaml")
            if os.path.isfile(candidate):
                return candidate
        return ""

    def switch_easy_tracking_mode(self):
        if self.proc and self.proc.poll() is None:
            return

        easy_script = str(gui_script("gui_easy_tracking.py"))
        cmd = [sys.executable, "-u", easy_script]
        cmd += ["--without-direction-estimation", str(int(self.without_direction.get()))]
        cfg_path = self._existing_config_path()
        if cfg_path:
            if not self._do_load_yaml(cfg_path, show_msg=False):
                return
            cmd += ["--load-config", cfg_path]
        else:
            # Easy Tracking can start without a config. Carry over only the
            # paths that have an unambiguous Easy Tracking equivalent.
            session = self.basic_entries["SESSION_PATH"].get().strip()
            training_video = self.basic_entries["TRAINING_VIDEO_PATH"].get().strip()
            if session:
                cmd += ["--session", session]
            if training_video:
                cmd += ["--video", training_video]

        subprocess.Popen(cmd)
        self.after(100, self.destroy)

    def _set_run_progress(self, pct: float, text: str) -> None:
        pct = max(0.0, min(1.0, float(pct)))
        self.progress_bar.set(pct)
        self.progress_label.configure(text=text[:80])

    def _echo_tqdm_line(self, line: str, previous_len: int) -> int:
        import shutil

        if not _stdout_supports_overwrite():
            return int(previous_len)

        text = line.rstrip()
        cols = shutil.get_terminal_size((120, 24)).columns
        if len(text) > cols:
            text = text[:cols]
        pad = " " * max(0, min(int(previous_len), cols) - len(text))
        _safe_stdout_write("\r" + text + pad)
        sys.stdout.flush()
        return len(text)

    def _handle_batch_output_record(self, line: str, state: dict) -> None:
        line = line.rstrip("\r\n")

        if line == "":
            if not state.get("last_was_tqdm", False):
                _safe_stdout_write("\n")
                sys.stdout.flush()
            return

        match = _TQDM_PAT.search(line)
        if match:
            state["last_tqdm_len"] = self._echo_tqdm_line(line, int(state.get("last_tqdm_len", 0)))
            state["last_was_tqdm"] = True

            n, total = int(match.group(1)), int(match.group(2))
            pct = n / max(total, 1)
            desc_m = re.search(r"\s(\d+)%\|", line)
            if desc_m:
                prefix = line[:desc_m.start()]
                colon = prefix.rfind(":")
                desc = prefix[:colon].strip() if colon >= 0 else prefix.strip()
            else:
                desc = ""
            text = f"{desc}  {n}/{total}" if desc else f"{n}/{total}"
            self.after(0, lambda p=pct, t=text: self._set_run_progress(p, t))
            return

        if state.get("last_was_tqdm", False):
            if _stdout_supports_overwrite():
                _safe_stdout_write("\n")
                sys.stdout.flush()
            state["last_was_tqdm"] = False
            state["last_tqdm_len"] = 0

        _safe_stdout_write(line + "\n")
        sys.stdout.flush()

        if line.startswith("Saving CLI log to:"):
            self._batch_log_path = line.split(":", 1)[1].strip()
        elif "] START: " in line:
            script = line.split("] START: ", 1)[-1].strip()
            self.after(0, lambda s=script: self._set_run_progress(0.0, s + "..."))
        elif "] END:   " in line:
            script = line.split("] END:   ", 1)[-1].split("  (")[0].strip()
            self.after(0, lambda s=script: self._set_run_progress(1.0, s + " done"))

    def _monitor_proc_output(self):
        state = {"last_was_tqdm": False, "last_tqdm_len": 0}
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buf: list[str] = []
        last_delim_was_cr = False
        stdout = self.proc.stdout if self.proc else None

        def flush_record() -> None:
            nonlocal buf
            line = "".join(buf)
            buf = []
            self._handle_batch_output_record(line, state)

        if stdout is not None:
            while True:
                chunk = stdout.read1(4096)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                for ch in text:
                    if ch == "\r" or ch == "\n":
                        if ch == "\n" and last_delim_was_cr and not buf:
                            last_delim_was_cr = False
                            continue
                        flush_record()
                        last_delim_was_cr = (ch == "\r")
                    else:
                        buf.append(ch)
                        last_delim_was_cr = False

            tail = decoder.decode(b"", final=True)
            if tail:
                for ch in tail:
                    if ch == "\r" or ch == "\n":
                        if ch == "\n" and last_delim_was_cr and not buf:
                            last_delim_was_cr = False
                            continue
                        flush_record()
                        last_delim_was_cr = (ch == "\r")
                    else:
                        buf.append(ch)
                        last_delim_was_cr = False

        if buf:
            flush_record()
        if state.get("last_was_tqdm", False) and _stdout_supports_overwrite():
            _safe_stdout_write("\n")
            sys.stdout.flush()

        self.proc.wait()
        rc = self.proc.returncode
        self.after(0, lambda rc=rc: self._on_proc_finished(rc))

    def adjust_parameters(self):
        if not self.save_yaml(print_msg=False):
            return
        self._batch_log_path = ""
        self._stop_requested = False
        session = self.basic_entries["SESSION_PATH"].get()
        yaml_path = os.path.join(session, "config.yaml")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        with_pythonpath(env, PROJECT_ROOT, MAIN_DIR)
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
        proc = subprocess.Popen(
            [sys.executable, "-u", INITIAL_TRACKING_SCRIPT, yaml_path, "--adjust-only"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **kw,
        )

        popup = tk.Toplevel(self)
        popup.title("Adjust Parameters")
        popup.transient(self)
        popup.resizable(False, False)
        popup.protocol("WM_DELETE_WINDOW", lambda: None)
        tk.Label(
            popup,
            text="Measuring segmentation results and updating auto parameters...",
            anchor="w", justify="left", wraplength=420,
        ).pack(fill="x", padx=16, pady=(16, 8))
        from tkinter import ttk
        bar = ttk.Progressbar(popup, mode="indeterminate", length=420)
        bar.pack(padx=16, pady=4)
        bar.start(12)
        status_var = tk.StringVar(value="Starting...")
        tk.Label(popup, textvariable=status_var, anchor="w", justify="left",
                 fg="#888888", wraplength=420).pack(fill="x", padx=16, pady=(4, 16))
        popup.update_idletasks()
        px = self.winfo_rootx() + max(0, (self.winfo_width() - popup.winfo_reqwidth()) // 2)
        py = self.winfo_rooty() + max(0, (self.winfo_height() - popup.winfo_reqheight()) // 2)
        popup.geometry(f"+{px}+{py}")
        popup.grab_set()

        self.adjust_btn.configure(state="disabled")
        self.run_btn.configure(state="disabled")

        def _pump() -> None:
            assert proc.stdout is not None
            buf = bytearray()
            last_update = 0.0

            def _flush(raw: bytes) -> None:
                nonlocal last_update
                text = _ANSI_ESCAPE_RE.sub("", raw.decode("utf-8", errors="replace")).strip()
                if not text:
                    return
                text = _simplify_progress_line(text)
                if not text:
                    return
                now = time.monotonic()
                if now - last_update < 0.15:
                    return
                last_update = now
                self.after(0, lambda t=text: status_var.set(t))

            while True:
                chunk = proc.stdout.read(1)
                if not chunk:
                    break
                if chunk in (b"\r", b"\n"):
                    if buf:
                        _flush(bytes(buf))
                        buf.clear()
                else:
                    buf.extend(chunk)
            if buf:
                _flush(bytes(buf))
            proc.wait()
            self.after(0, lambda: _finish(proc.returncode))

        def _finish(returncode: int) -> None:
            bar.stop()
            popup.grab_release()
            popup.destroy()
            self.adjust_btn.configure(state="normal")
            self.run_btn.configure(state="normal")
            if returncode != 0:
                messagebox.showerror(
                    "Error",
                    f"Adjust Parameters failed (exit code {returncode}). See console output for details.",
                )
                return
            self._reload_adjusted_config(yaml_path)

        threading.Thread(target=_pump, daemon=True).start()

    def _reload_adjusted_config(self, yaml_path: str):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg = resolve_config_paths(cfg)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload adjusted config:\n{e}")
            return

        if not bool(cfg.get("AUTO_PARAMS", True)):
            messagebox.showinfo(
                "Adjust Parameters",
                "AUTO_PARAMS is Off, so the measured values were logged to "
                "tracking_stats.csv but not applied to the config.",
            )
            return

        analysis_cfg = cfg.get("analysis", {}) or {}
        updates = {
            "LOCALIZED": cfg.get("LOCALIZED"),
            "NUM_CROPS": cfg.get("NUM_CROPS"),
            "FREE_SCALE": cfg.get("FREE_SCALE"),
            "DIR_MIN_SEC": cfg.get("DIR_MIN_SEC"),
            "CLUSTER_FRAMES": cfg.get("CLUSTER_FRAMES"),
            "MATCH_IOU": analysis_cfg.get("MATCH_IOU"),
        }

        def _set_value(widget, value):
            if value is None:
                return
            if isinstance(widget, tk.BooleanVar):
                widget.set(bool(value))
            elif isinstance(widget, ctk.CTkComboBox):
                widget.set(str(value))
            else:
                widget.delete(0, tk.END)
                widget.insert(0, str(value))

        for key, value in updates.items():
            for section in self.sections.values():
                widget = section.get("widgets", {}).get(key)
                if widget is not None:
                    _set_value(widget, value)

        self._capture_auto_param_values(cfg)
        self._refresh_default_markers()
        messagebox.showinfo(
            "Adjust Parameters",
            "Auto parameters updated from the current segmentation results:\n"
            f"  MATCH_IOU = {updates['MATCH_IOU']}\n"
            f"  INTERACT_IOU = {FIXED_INTERACT_IOU} (fixed)\n"
            f"  DIR_MIN_SEC = {updates['DIR_MIN_SEC']}\n"
            f"  LOCALIZED = {updates['LOCALIZED']}\n"
            f"  NUM_CROPS = {updates['NUM_CROPS']}\n"
            f"  FREE_SCALE = {updates['FREE_SCALE']}\n"
            f"  CLUSTER_FRAMES = {updates['CLUSTER_FRAMES']}\n\n"
            "Run will reuse this config, so there is no need to adjust again.",
        )

    def run(self):
        batch_path = str(main_script("batch.py"))

        if not self.save_yaml(print_msg=False):
            return
        session = self.basic_entries["SESSION_PATH"].get()
        yaml_path = os.path.join(session, "config.yaml")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env.setdefault("TQDM_ASCII", " 123456789#")
        env.setdefault("COLUMNS", "120")
        with_pythonpath(env, PROJECT_ROOT, MAIN_DIR)
        # Preserve raw tqdm redraws for the live GUI progress widget; batch.py
        # compacts the archive and all non-GUI output paths independently.
        env["AMADEUS_GUI_PROGRESS_PROTOCOL"] = "1"

        if sys.platform == "win32":
            self.proc = subprocess.Popen(
                [sys.executable, "-u", batch_path, yaml_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
        else:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", batch_path, yaml_path],
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )

        self._set_run_progress(0.0, "Starting...")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.switch_easy_btn.configure(state="disabled")
        self._direction_mode_checkbox.configure(state="disabled")
        threading.Thread(target=self._monitor_proc_output, daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self._stop_requested = True
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                print("=== Batch job was terminated by user ===")
            except Exception as e:
                messagebox.showerror("Error", f"Stop failed:\n{e}")

    def _on_proc_finished(self, returncode: int):
        stopped_by_user = self._stop_requested
        self._stop_requested = False
        if returncode == 0:
            self._set_run_progress(1.0, "Complete")
        elif stopped_by_user:
            self._set_run_progress(0.0, f"Stopped ({returncode})")
        else:
            log_path = self._batch_log_path or "unavailable"
            self._set_run_progress(0.0, f"Processing failed (exit {returncode})")
            messagebox.showerror(
                "Processing failed",
                f"Processing failed with exit code {returncode}.\n\n"
                f"Full log:\n{log_path}",
            )
        self.stop_btn.configure(state="disabled")
        self.run_btn.configure(state="normal")
        self.switch_easy_btn.configure(state="normal")
        self._direction_mode_checkbox.configure(state="normal")

    @staticmethod
    def _cast(key: str, value):
        int_keys = {
            "EDGE_BLUR_KSIZE",
            "MAX_TRIES",
            "FEATHER_MIN",
            "FEATHER_MAX",
            "OCCLUDER_MARGIN",
            "NUM_PREVIEW_FRAMES",
            "PREVIEW_INTERVAL",
            "NUM_IMAGES",
            "NUM_OBJECTS",
            "FIRST_FRAME",
            "LAST_FRAME",
            "NUM_CROPS",
            "FRAME_INTERVAL",
            "MAX_AGE",
            "FRAME_STEP",
            "ARROW_WIDTH",
            "LABEL_THICKNESS",
            "INIT_MAX_GAP",
            "OBB_WIDTH",
            "TRAIN_IMG_SIZE",
            "REFINE_EPOCHS",
            "REFINE_ITERS",
            "EPOCHS",
            "CLUSTER_COUNT",
            "CLUSTER_FRAMES",
            "PREVIEW_COUNT",
            "NOISE_MAX_COUNT",
        }
        float_keys = {
            "MASK_EXPANSION_RATIO",
            "MIN_OVERLAP",
            "MAX_OVERLAP",
            "CONF",
            "MATCH_IOU",
            "INTERACT_IOU",
            "NMS_IOU",
            "VAL_RATIO",
            "REFINE_FRAME_RATIO",
            "DIR_MIN_SEC",
            "DIR_MIN_DISP",
            "TRAJ_MAX_JUMP",
            "MIN_ASPECT",
            "UNDER_PASTE_PROB",
            "EDGE_BLUR_SIGMA",
            "FPS",
            "LABEL_SCALE",
            "ARROW_ALPHA",
            "ARROW_SCALE",
            "LOCALIZED_RATIO",
            "REFINE_CONF",
            "RUN_DELETE_RATIO",
            "RATIO_SINGLE",
            "RATIO_P2",
            "RATIO_P3",
            "PASTE_SCALE_MIN",
            "PASTE_SCALE_MAX",
            "WIDTH_SCALE_MIN",
            "WIDTH_SCALE_MAX",
            "PASTE_MIN_ASPECT",
            "MATCH_ANGLE",
            "MAX_AXIS_ERR",
            "CLUSTER_FIT_LONG",
            "CLUSTER_FIT_SHORT",
            "CLUSTER_BREAK_PROB",
            "CLUSTERED_RATIO",
            "FREE_SCALE",
            "FREE_RATIO_SINGLE",
            "FREE_RATIO_P2",
            "FREE_RATIO_P3",
            "TRAJ_MAX_DIST",
            "IOU_WEIGHT",
            "DIRECTION_WEIGHT",
            "MISS_WEIGHT",
            "DISTANCE_WEIGHT",
            "FLIP_SEC",
            "BRIGHT_MIN",
            "BRIGHT_MAX",
            "CONTRAST_MIN",
            "CONTRAST_MAX",
            "NOISE_SIZE_PERCENT",
        }
        if isinstance(value, bool):
            return value
        try:
            if key == "IMG_SIZE":
                if isinstance(value, str) and value.strip().lower() == "auto":
                    return "auto"
                try:
                    v = int(float(value))
                    return max(32, ((v + 31) // 32) * 32)
                except (ValueError, TypeError):
                    return "auto"
            if key == "WEIGHT":
                if value == "" or value is None:
                    return None
                if isinstance(value, str):
                    s = value.strip()
                    if "," in s:
                        return ",".join(part.strip() for part in s.split(",") if part.strip())
                    if s.lower() in {"best", "last"}:
                        return s.lower()
                    return s
                return value
            if key == "SAVE_PERIOD":
                if value == "" or value is None:
                    return 5
                if isinstance(value, str) and value.strip().lower() == "best":
                    return "best"
                return _strict_int(value)
            if key in int_keys:
                return _strict_int(value)
            if key in float_keys:
                if value == "" or value is None:
                    return None
                return float(value)
        except (ValueError, TypeError):
            pass
        return value if value != "" else None


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="")
    parser.add_argument("--training-video", dest="training_video", default="")
    parser.add_argument("--tracking-video", dest="tracking_video", default="")
    parser.add_argument("--load-config", dest="load_config", default="")
    parser.add_argument("--without-direction-estimation", type=int, choices=(0, 1), default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_taskbar_identity()
    app = ConfigGUI(
        session_path=args.session,
        training_video_path=args.training_video,
        tracking_video_path=args.tracking_video,
    )
    if args.load_config and os.path.isfile(args.load_config):
        app.after(400, lambda p=args.load_config: app._do_load_yaml(p, show_msg=False))
    if args.without_direction_estimation is not None:
        app.after(500, lambda: app.without_direction.set(bool(args.without_direction_estimation)))
    app.mainloop()


if __name__ == "__main__":
    main()
