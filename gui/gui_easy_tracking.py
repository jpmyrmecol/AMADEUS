# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import codecs
import csv
import glob
import math
import os
import re
import sys
import copy
import argparse
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import yaml

try:
    from .project_paths import (
        GUI_DIR,
        PROJECT_ROOT,
        MAIN_DIR as MAIN_PATH,
        gui_asset,
        gui_script,
        main_script,
        open_results_directory,
        with_pythonpath,
    )
    from .window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
    from .config_path_recovery import prepare_config_for_gui
    from .video_input import ask_open_analysis_video
except ImportError:  # Preserve direct execution with: python gui/gui_easy_tracking.py
    from project_paths import (
        GUI_DIR,
        PROJECT_ROOT,
        MAIN_DIR as MAIN_PATH,
        gui_asset,
        gui_script,
        main_script,
        open_results_directory,
        with_pythonpath,
    )
    from window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
    from config_path_recovery import prepare_config_for_gui
    from video_input import ask_open_analysis_video

# Matches tqdm lines: anything containing "N/TOTAL [" (works for both TTY and non-TTY output)
_TQDM_PAT = re.compile(r'(\d+)/(\d+)\s*\[')
# Strips ANSI color/cursor escape codes (e.g. tqdm's colour="green") from subprocess output.
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


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
# Matches YOLO training tqdm desc: "3/50  3.05G  0.8756  0.7188  1.492 ..."
_YOLO_LOSS_PAT = re.compile(r'\d+/\d+\s+[\d.]+G?\s+([\d.]+)')

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

CURRENT_DIR = str(GUI_DIR)
MAIN_DIR = str(MAIN_PATH)
BATCH_SCRIPT = str(main_script("batch.py"))
CTK_THEME = str(gui_asset("deep_green.json"))

if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)
from experiment_utils import DEFAULT_LR0, DEFAULT_LRF
from path_utils import relativize_config_paths
from segmentation_metadata import (
    read_segmentation_metadata,
    segmentation_paths_for_session,
)
from training_paths import OUTPUT_ROOT_DIR, TRAINING_STAGE_DIRS
from tracking_constants import FIXED_INTERACT_IOU

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

# Process steps: (display_label, skip_key, script_stem)
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


STEPS = [
    ("Initial\nTracking",    "skip_initial_tracking",              "initial_tracking"),
    ("Direction\nEstimation","skip_trajectory_direction_filtering", "direction_class_assignment"),
    ("Refine\nBlobs",        "skip_refine_blobs_through_tracking",  "direction_class_filtering"),
    ("Clustered\nPaste",     "skip_paste_blobs_clustered",          "interaction_image_synthesis_clustered"),
    ("Mixed\nPaste",         "skip_paste_blobs_with_crossing",      "interaction_image_synthesis"),
    ("Crop\nImages",         "skip_cropping",                       "crop_images"),
    ("Create\nDataset",      "skip_creating_direction_dataset",     "create_dataset"),
    ("Training",             "skip_training",                       "obb_detector_training"),
    ("Detection",            "skip_detection",                      "obb_detection"),
    ("ID\nTracking",         "skip_id_tracking",                    "multi_staged_association"),
    ("ID\nCorrection",       "skip_id_correction",                  "refinement"),
    ("Create\nVideo",        "skip_creating_video",                 "create_video"),
]
SCRIPT_TO_IDX: dict[str, int] = {s[2]: i for i, s in enumerate(STEPS)}

_C_INACTIVE  = "#3a3a3a"
_C_RUNNING   = "#1f8040"
_C_DONE      = "#155a2e"
_C_SELECTED  = _C_INACTIVE
_C_DISABLED  = "#252525"
_C_BG_ROW    = "#1e1e1e"
_WIDTH_SCALE_MIN = 0.9
_WIDTH_SCALE_MAX = 1.1


class EasyTrackingGUI(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()
        configure_dpi_scaling(self)
        install_window_icon(self)
        self.title("AMADEUS Easy Tracking")
        self.geometry("1300x840")
        self.minsize(900, 600)

        self._seg_proc: subprocess.Popen | None = None
        self._batch_proc: subprocess.Popen | None = None
        self._batch_log_path: str = ""
        self._stop_requested = False
        self._base_cfg: dict = {}
        self._loaded_cfg: dict = {}
        self._loaded_config_path: str = ""
        self._loaded_easy_answers: dict = {}
        self._skip_flags: dict[str, bool] = {}
        self._manual_skip_keys: set[str] = set()
        self._num_confirmed = False
        self._variable_count = tk.BooleanVar(value=False)
        self._without_direction = tk.BooleanVar(value=False)
        self._without_direction.trace_add(
            "write", lambda *_: self._apply_backward_question_state())
        self._backward_radios: list[ctk.CTkRadioButton] = []
        self._confirmed_num_objects = 1
        self._overlap_radios: list[ctk.CTkRadioButton] = []
        self._single_animal_paste: tk.BooleanVar | None = None
        self._vis_indices: list[int] = []
        self._done_steps: set[int] = set()
        self._active_step: int = -1
        self._active_block_vis: int = -1
        self._selected_block: int = -1
        self._block_frames: list[tk.Canvas] = []
        self._block_text_ids: list[int] = []
        self._block_states: list[str] = []
        self._shimmer_after_id: str | None = None
        self._shimmer_phase: float = 0.0

        # Canvas panel state
        self._canvas_poll_id: str | None = None
        self._display_photo = None          # keeps PIL PhotoImage from GC
        self._display_step: int = -2        # last step rendered (-2 = none)
        self._last_display_img: str = ""    # last successfully shown image path
        self._canvas_content: dict = {}     # cache for resize re-render
        self._step_start_times: dict[int, float] = {}
        self._batch_losses: list[float] = []   # live per-batch box_loss from YOLO training
        self._refine_losses: list[float] = []  # live per-batch box_loss from direction_class_filtering

        self._build_ui()
        self._set_skip_flags(self._default_skip_flags(), render=True)
        self._sync_refine_skip_from_backward_question()
        self._remember_easy_answers()
        _start_maximized(self)

    # UI construction
    def _build_ui(self) -> None:
        # Resizable split layout: left controls | right preview canvas.
        pane = tk.PanedWindow(
            self,
            orient=tk.HORIZONTAL,
            bg="#2b2b2b",
            bd=0,
            sashwidth=7,
            sashrelief="flat",
            showhandle=True,
            handlesize=14,
            opaqueresize=True,
        )
        pane.pack(fill="both", expand=True)

        left_wrap = ctk.CTkFrame(pane, corner_radius=0, width=800)
        left_wrap.pack_propagate(False)
        left = ctk.CTkScrollableFrame(left_wrap, corner_radius=0)
        left.pack(fill="both", expand=True)
        right = tk.Frame(pane, bg="#181818", width=480)

        pane.add(left_wrap, minsize=540)
        pane.add(right, minsize=360)

        self._build_video_section(left)
        self._build_questions_section(left)
        self._build_action_buttons(left)
        self._build_progress_section(left)
        self._build_canvas_panel(right)

    # Step 1: Video
    def _build_video_section(self, parent: ctk.CTkScrollableFrame) -> None:
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", padx=12, pady=(10, 5))
        ctk.CTkLabel(frame, text="Step 1: Select Video",
                     font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=8, pady=(6, 2))

        row = ctk.CTkFrame(frame, corner_radius=0)
        row.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(row, text="Video File", width=90, anchor="w").pack(side="left")
        self._video_entry = ctk.CTkEntry(row)
        self._video_entry.pack(side="left", expand=True, fill="x", padx=5)
        self._video_entry.bind("<FocusOut>", lambda _e: self._update_session_label())
        self._video_entry.bind("<Return>", lambda _e: self._update_session_label())
        ctk.CTkButton(row, text="Browse", width=80, command=self._browse_video).pack(side="left", padx=3)

        row2 = ctk.CTkFrame(frame, corner_radius=0)
        row2.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(row2, text="Session", width=90, anchor="w").pack(side="left")
        self._session_entry = ctk.CTkEntry(row2, placeholder_text="(auto)")
        self._session_entry.pack(side="left", expand=True, fill="x", padx=5)
        self._session_entry.bind(
            "<FocusOut>", lambda _e: self._sync_segmentation_status_from_outputs())
        self._session_entry.bind(
            "<Return>", lambda _e: self._sync_segmentation_status_from_outputs())
        ctk.CTkButton(row2, text="Browse", width=80, command=self._browse_session).pack(side="left", padx=3)

        row3 = ctk.CTkFrame(frame, corner_radius=0)
        row3.pack(fill="x", padx=8, pady=(2, 10))
        self._seg_btn = ctk.CTkButton(row3, text="Launch Segmentation",
                                      width=230, command=self._launch_segmentation)
        self._seg_btn.pack(side="left", padx=(0, 12))
        self._seg_status = ctk.CTkLabel(row3, text="Not started", text_color="gray", anchor="w")
        self._seg_status.pack(side="left")

    def _browse_video(self) -> None:
        prepared = ask_open_analysis_video(
            self, title="Select video", log=lambda message: print(f"[video] {message}", flush=True)
        )
        if prepared is None:
            return
        self._video_entry.delete(0, tk.END)
        self._video_entry.insert(0, prepared.path)
        self._update_session_label()

    def _update_session_label(self) -> None:
        # Only auto-fill if the user hasn't typed a custom path
        if not self._session_entry.get().strip():
            v = self._video_entry.get().strip()
            if v:
                stem = os.path.splitext(os.path.basename(v))[0]
                session = os.path.join(os.path.dirname(v), f"amadeus_{stem}")
                self._session_entry.delete(0, tk.END)
                self._session_entry.insert(0, session)
        self._sync_segmentation_status_from_outputs()

    def _browse_session(self) -> None:
        current = self._session_entry.get().strip()
        initial = current if os.path.isdir(current) else os.path.dirname(current) if current else os.getcwd()
        path = filedialog.askdirectory(initialdir=initial)
        if path:
            self._session_entry.delete(0, tk.END)
            self._session_entry.insert(0, path)
            self._sync_segmentation_status_from_outputs()

    def _get_session(self) -> str:
        s = self._session_entry.get().strip()
        if s:
            return s
        # Fallback: derive from video path
        v = self._video_entry.get().strip()
        if not v:
            return ""
        stem = os.path.splitext(os.path.basename(v))[0]
        return os.path.join(os.path.dirname(v), f"amadeus_{stem}")

    def _apply_startup_paths(self, video_path: str = "", session_path: str = "") -> None:
        if video_path:
            self._video_entry.delete(0, tk.END)
            self._video_entry.insert(0, video_path)
        if session_path:
            self._session_entry.delete(0, tk.END)
            self._session_entry.insert(0, session_path)
        elif video_path:
            self._update_session_label()
        self._sync_segmentation_status_from_outputs()

    def _launch_segmentation(self) -> None:
        video = self._video_entry.get().strip()
        if not video:
            messagebox.showwarning("No video", "Please select a video file first.")
            return
        seg_script = str(gui_script("gui_segmentation.py"))
        cmd = [sys.executable, "-u", seg_script, "--return-to-easy"]
        if video:
            cmd += ["--video", video]
        session = self._get_session()
        if session:
            cmd += ["--session", session]
        self._seg_proc = subprocess.Popen(cmd)
        self._seg_status.configure(text="Running...", text_color="#f0a000")
        self._seg_btn.configure(state="disabled")
        threading.Thread(target=self._watch_seg, daemon=True).start()

    def _watch_seg(self) -> None:
        self._seg_proc.wait()
        self.after(0, self._on_seg_done)

    def _on_seg_done(self) -> None:
        self._sync_segmentation_status_from_outputs()
        self._seg_btn.configure(state="normal")
        self.lift()
        self.focus_force()

    def _segmentation_outputs_exist(self, cfg: dict | None = None) -> bool:
        cfg = cfg or {}
        video = str(cfg.get("TRAINING_VIDEO_PATH", "") or "").strip()
        if not video:
            video = self._video_entry.get().strip()
        session = str(cfg.get("SESSION_PATH", "") or "").strip()
        if not session:
            session = self._get_session()
        pickle_path, background_path = segmentation_paths_for_session(session, video)
        return bool(
            pickle_path
            and background_path
            and os.path.isfile(pickle_path)
            and os.path.isfile(background_path)
        )

    def _sync_segmentation_status_from_outputs(self, cfg: dict | None = None) -> None:
        if not hasattr(self, "_seg_status"):
            return
        if self._seg_proc is not None and self._seg_proc.poll() is None:
            return
        if self._segmentation_outputs_exist(cfg):
            self._seg_status.configure(text="Done", text_color="#1f8040")
        else:
            self._seg_status.configure(text="Not started", text_color="gray")

    # Step 2: Questions
    def _build_questions_section(self, parent: ctk.CTkScrollableFrame) -> None:
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", padx=12, pady=5)
        header = ctk.CTkFrame(frame, corner_radius=0)
        header.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(
            header,
            text="Step 2: Settings",
            font=("TkDefaultFont", 13, "bold"),
            anchor="w",
        ).pack(side="left")
        self._delete_tmp_files = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            header,
            text="Delete tmp files",
            variable=self._delete_tmp_files,
            width=150,
        ).pack(side="right", padx=(0, 28))
        self._direction_mode_checkbox = ctk.CTkCheckBox(
            header,
            text="Without direction estimation [Beta]",
            variable=self._without_direction,
        )
        self._direction_mode_checkbox.pack(side="right", padx=(0, 12))

        inner = ctk.CTkFrame(frame, corner_radius=0)
        inner.pack(fill="x", padx=8, pady=(2, 10))
        question_label_width = 300
        question_option_gap = 14
        first_choice_width = 104
        radio_gap = 6

        def pack_question_label(parent, text):
            ctk.CTkLabel(parent, text=text, width=question_label_width, anchor="w").pack(
                side="left",
                padx=(0, question_option_gap),
            )

        def pack_radio(parent, text, var, value, *, width, command=None, gap=None):
            padx_right = radio_gap if gap is None else int(gap)
            radio = ctk.CTkRadioButton(
                parent,
                text=text,
                variable=var,
                value=value,
                command=command,
                width=width,
            )
            radio.pack(side="left", padx=(0, padx_right))
            return radio

        # Animals count
        row = ctk.CTkFrame(inner, corner_radius=0)
        row.pack(fill="x", pady=4)
        pack_question_label(row, "How many animals are in the video?")
        self._num_objects_sb = tk.Spinbox(
            row,
            from_=1,
            to=999,
            increment=1,
            width=6,
            command=self._invalidate_num_confirm,
            **_SPIN_CFG,
        )
        self._num_objects_sb.pack(side="left", padx=5)
        self._num_objects_sb.delete(0, tk.END)
        self._num_objects_sb.insert(0, "1")
        self._num_objects_sb.bind("<Return>", lambda _e: self._confirm_num_objects(), add="+")
        self._num_objects_sb.bind("<KP_Enter>", lambda _e: self._confirm_num_objects(), add="+")
        self._num_objects_sb.bind("<KeyRelease>", self._on_num_objects_key_release, add="+")
        self._num_enter_button = ctk.CTkButton(row, text="Enter", width=70, command=self._confirm_num_objects)
        self._num_enter_button.pack(side="left", padx=(6, 8))
        ctk.CTkCheckBox(row, text="Variable population [Beta]", variable=self._variable_count,
                       command=self._on_variable_count_changed).pack(side="left", padx=6)
        self._num_confirm_hint = ctk.CTkLabel(row, text="Press Enter to confirm", text_color="gray", anchor="w")
        self._num_confirm_hint.pack(side="left")

        row = ctk.CTkFrame(inner, corner_radius=0)
        row.pack(fill="x", pady=4)
        pack_question_label(row, "")
        self._single_animal_paste = tk.BooleanVar(value=False)
        self._single_animal_paste_checkbox = ctk.CTkCheckBox(
            row,
            text="Use paste augmentation when animals = 1",
            variable=self._single_animal_paste,
            command=self._on_single_animal_paste_changed,
        )
        self._single_animal_paste_checkbox.pack(side="left", padx=5)

        # Overlap
        row = ctk.CTkFrame(inner, corner_radius=0)
        row.pack(fill="x", pady=4)
        pack_question_label(row, "How severe is the overlap?")
        self._overlap = tk.StringVar(value="heavy")
        for text, val, width, gap in [
            ("Very heavy", "super_heavy", first_choice_width, radio_gap),
            ("Heavy", "heavy", 66, 14),
            ("Light", "light", 66, radio_gap),
        ]:
            command = self._show_super_heavy_overlap_warning if val == "super_heavy" else None
            self._overlap_radios.append(
                pack_radio(row, text, self._overlap, val, width=width, command=command, gap=gap)
            )

        # Backward movement
        row = ctk.CTkFrame(inner, corner_radius=0)
        row.pack(fill="x", pady=4)
        pack_question_label(row, "Do the animals ever move backward?")
        self._backward_movement = tk.StringVar(value="no")
        self._backward_movement.trace_add("write", lambda *_: self._sync_refine_skip_from_backward_question())
        self._backward_radios = []
        for text, val in [("Yes", "yes"), ("No", "no")]:
            width = first_choice_width if val == "yes" else 66
            self._backward_radios.append(
                pack_radio(row, text, self._backward_movement, val, width=width))
        self._apply_backward_question_state()

        self._apply_individual_count_state()

    def _show_super_heavy_overlap_warning(self) -> None:
        messagebox.showwarning(
            "Very Heavy Overlap",
            "Very heavy overlap uses contrastive learning and can take a long time.\n\n"
            "Press OK to continue.",
            parent=self,
        )

    # Action buttons
    def _build_action_buttons(self, parent: ctk.CTkScrollableFrame) -> None:
        frame = ctk.CTkFrame(parent, corner_radius=0)
        frame.pack(fill="x", padx=12, pady=8)

        row1 = ctk.CTkFrame(frame, corner_radius=0)
        row1.pack(fill="x")

        self._run_btn = ctk.CTkButton(
            row1, text="Processing", width=180, height=42,
            font=("TkDefaultFont", 13, "bold"), command=self._run_easy_tracking,
        )
        self._run_btn.pack(side="left", padx=(0, 10))

        self._open_results_btn = ctk.CTkButton(
            row1, text="Open results", width=130, height=42,
            command=self._open_results,
        )
        self._open_results_btn.pack(side="left", padx=(0, 10))

        self._save_btn = ctk.CTkButton(
            row1, text="Save Config", width=130, height=42,
            command=self._save_config,
        )
        self._save_btn.pack(side="left", padx=(0, 10))

        self._load_btn = ctk.CTkButton(
            row1, text="Load Config", width=130, height=42,
            command=self._load_config,
        )
        self._load_btn.pack(side="left")

        self._stop_btn = ctk.CTkButton(
            row1, text="Stop", width=100, height=42,
            state="disabled",
            fg_color="#8b1a1a", hover_color="#a02020",
            command=self._stop_batch,
        )
        self._stop_btn.pack(side="right")

    def _open_results(self) -> None:
        try:
            open_results_directory(self._get_session())
        except (OSError, ValueError) as exc:
            messagebox.showwarning("Open results", str(exc), parent=self)

    # Progress section
    def _build_progress_section(self, parent: ctk.CTkScrollableFrame) -> None:
        frame = ctk.CTkFrame(parent, corner_radius=6)
        frame.pack(fill="x", padx=12, pady=(5, 14))
        ctk.CTkLabel(frame, text="Progress",
                     font=("TkDefaultFont", 13, "bold"), anchor="w").pack(fill="x", padx=8, pady=(6, 2))

        # Overall progress bar
        pb_wrap = ctk.CTkFrame(frame, corner_radius=0)
        pb_wrap.pack(fill="x", padx=8, pady=4)
        self._progress_canvas = tk.Canvas(pb_wrap, height=30, bg="#1e1e1e", highlightthickness=0)
        self._progress_canvas.pack(fill="x")
        self._progress_canvas.bind("<Configure>", lambda _e: self._redraw_progress())
        self._progress_pct: float = 0.0
        self._progress_text: str = "Idle"

        # Process blocks
        self._blocks_wrap = ctk.CTkFrame(frame, corner_radius=0, fg_color="#1e1e1e")
        self._blocks_wrap.pack(fill="x", padx=8, pady=4)
        self._blocks_row: tk.Frame | None = None
        self._render_blocks([])

        # Bottom bar: mode switch
        bot = ctk.CTkFrame(frame, corner_radius=0)
        bot.pack(fill="x", padx=8, pady=(2, 10))
        self._switch_btn = ctk.CTkButton(
            bot, text="Switch Advanced Mode", width=180, height=34,
            command=self._switch_advanced_mode,
        )
        self._switch_btn.pack(side="left")

    # Canvas panel - visual progress
    def _build_canvas_panel(self, parent: tk.Frame) -> None:
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        hdr = tk.Frame(parent, bg="#222222")
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="Visual Progress", bg="#222222", fg="#cccccc",
                 font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=10, pady=5)

        self._display_canvas = tk.Canvas(parent, bg="#1a1a1a", highlightthickness=0)
        self._display_canvas.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._display_canvas.bind("<Configure>", lambda _e: self._on_canvas_resize())

        self._canvas_status = tk.Label(
            parent, text="Waiting for processing to start...",
            bg="#181818", fg="#666666", font=("TkDefaultFont", 8),
            wraplength=600, justify="center",
        )
        self._canvas_status.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 6))

    def _on_canvas_resize(self) -> None:
        cc = self._canvas_content
        if cc.get("type") == "image":
            self._show_image_on_canvas(cc["path"], cc.get("caption", ""))
        elif cc.get("type") == "chart":
            self._render_line_chart(
                self._display_canvas,
                cc["values"], cc["title"], cc.get("x_label", ""), cc.get("color", "#1f8040"),
            )
        elif cc.get("type") == "multi_chart":
            self._render_multi_line_chart(
                self._display_canvas,
                cc["series"], cc["title"], cc.get("x_label", ""),
            )
        elif cc.get("type") == "dual_chart":
            self._render_dual_chart(cc["data"])
        elif cc.get("type") == "training_monitor":
            self._render_training_monitor(cc["data"])
        else:
            self._draw_placeholder(cc.get("msg", "Waiting for processing to start..."))

    def _draw_placeholder(self, msg: str = "Waiting for processing to start...") -> None:
        self._canvas_content = {"type": "placeholder", "msg": msg}
        c = self._display_canvas
        c.delete("all")
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 100)
        c.create_rectangle(0, 0, w, h, fill="#1a1a1a", outline="")
        c.create_text(w // 2, h // 2, text=msg, fill="#555555",
                      font=("TkDefaultFont", 10), justify="center",
                      width=w - 40, anchor="center")

    def _show_image_on_canvas(self, path: str, caption: str = "") -> None:
        c = self._display_canvas
        cw = max(c.winfo_width(), 100)
        ch = max(c.winfo_height(), 100)
        try:
            from PIL import Image, ImageTk
            img = Image.open(path)
            iw, ih = img.size
            scale = min(cw / iw, ch / ih)
            new_w = int(round(iw * scale))
            new_h = int(round(ih * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            x = (cw - new_w) // 2
            y = (ch - new_h) // 2
            photo = ImageTk.PhotoImage(img)
            c.delete("all")
            c.create_rectangle(0, 0, cw, ch, fill="#1a1a1a", outline="")
            c.create_image(x, y, anchor="nw", image=photo)
            self._display_photo = photo        # prevent GC
            self._last_display_img = path
            self._canvas_content = {"type": "image", "path": path, "caption": caption}
        except Exception:
            pass  # Keep previous canvas content if image load fails

    # Line chart (drawn directly on a tk.Canvas, no matplotlib)

    def _draw_loss_chart(self, values: list, title: str,
                         x_label: str = "", color: str = "#1f8040") -> None:
        self._canvas_content = {
            "type": "chart", "values": values, "title": title,
            "x_label": x_label, "color": color,
        }
        self._display_canvas.delete("all")
        self._render_line_chart(self._display_canvas, values, title, x_label, color)

    def _render_line_chart(
        self, canvas: tk.Canvas, values: list, title: str,
        x_label: str, color: str,
        y_offset: int = 0, panel_height: int = 0,
    ) -> None:
        cw = max(canvas.winfo_width(), 100)
        ph = panel_height or max(canvas.winfo_height(), 100)

        ml, mr, mt, mb = 54, 8, 22, 30
        pw = cw - ml - mr
        ph_inner = ph - mt - mb

        if not values or pw < 20 or ph_inner < 20:
            return

        min_v = min(values)
        max_v = max(values)
        rng = (max_v - min_v) or 1e-8
        n = len(values)

        def px(i: int) -> int:
            return ml + int(i / max(n - 1, 1) * pw)

        def py(v: float) -> int:
            return y_offset + mt + int((1 - (v - min_v) / rng) * ph_inner)

        # background
        canvas.create_rectangle(0, y_offset, cw, y_offset + ph,
                                 fill="#1a1a1a", outline="")
        # title
        canvas.create_text(cw // 2, y_offset + 11, text=title,
                            fill="#cccccc", font=("TkDefaultFont", 9, "bold"), anchor="center")
        # horizontal grid + y-axis labels
        for frac in (0.0, 0.5, 1.0):
            v = min_v + frac * rng
            y = py(v)
            canvas.create_line(ml, y, ml + pw, y, fill="#272727", dash=(3, 5))
            lbl = f"{v:.4g}" if abs(v) < 1e4 else f"{v:.2e}"
            canvas.create_text(ml - 3, y, text=lbl,
                                fill="#666", font=("TkDefaultFont", 7), anchor="e")
        # axes
        canvas.create_line(ml, y_offset + mt, ml, y_offset + mt + ph_inner,
                            fill="#555", width=1)
        canvas.create_line(ml, y_offset + mt + ph_inner,
                            ml + pw, y_offset + mt + ph_inner,
                            fill="#555", width=1)
        # x-axis labels
        canvas.create_text(ml, y_offset + mt + ph_inner + 8, text="0",
                            fill="#666", font=("TkDefaultFont", 7), anchor="n")
        canvas.create_text(ml + pw, y_offset + mt + ph_inner + 8, text=str(n - 1),
                            fill="#666", font=("TkDefaultFont", 7), anchor="n")
        if x_label:
            canvas.create_text(ml + pw // 2, y_offset + mt + ph_inner + 20,
                                text=x_label, fill="#888",
                                font=("TkDefaultFont", 7), anchor="n")
        # data line
        if n >= 2:
            pts = []
            for i, v in enumerate(values):
                pts.append(px(i))
                pts.append(py(v))
            canvas.create_line(pts, fill=color, width=3, smooth=True)
        elif n == 1:
            x0, y0 = px(0), py(values[0])
            canvas.create_oval(x0 - 3, y0 - 3, x0 + 3, y0 + 3,
                                fill=color, outline="")

    def _render_multi_line_chart(
        self, canvas: tk.Canvas, series: list, title: str,
        x_label: str = "",
        y_offset: int = 0, panel_height: int = 0,
    ) -> None:
        """Draw multiple series on the same axes. series: [(values, color, label), ...]"""
        cw = max(canvas.winfo_width(), 100)
        ph = panel_height or max(canvas.winfo_height(), 100)

        ml, mr, mt, mb = 54, 8, 22, 30
        pw = cw - ml - mr
        ph_inner = ph - mt - mb

        valid_series = [(vals, col, lbl) for vals, col, lbl in series if vals]
        if not valid_series or pw < 20 or ph_inner < 20:
            return

        all_vals = [v for vals, _, _ in valid_series for v in vals]
        min_v = min(all_vals)
        max_v = max(all_vals)
        rng = (max_v - min_v) or 1e-8
        n_max = max(len(vals) for vals, _, _ in valid_series)

        def px(i: int, n: int) -> int:
            return ml + int(i / max(n - 1, 1) * pw)

        def py(v: float) -> int:
            return y_offset + mt + int((1 - (v - min_v) / rng) * ph_inner)

        canvas.create_rectangle(0, y_offset, cw, y_offset + ph, fill="#1a1a1a", outline="")
        canvas.create_text(cw // 2, y_offset + 11, text=title,
                           fill="#cccccc", font=("TkDefaultFont", 9, "bold"), anchor="center")

        for frac in (0.0, 0.5, 1.0):
            v = min_v + frac * rng
            y = py(v)
            canvas.create_line(ml, y, ml + pw, y, fill="#272727", dash=(3, 5))
            lbl = f"{v:.4g}" if abs(v) < 1e4 else f"{v:.2e}"
            canvas.create_text(ml - 3, y, text=lbl, fill="#666", font=("TkDefaultFont", 7), anchor="e")

        canvas.create_line(ml, y_offset + mt, ml, y_offset + mt + ph_inner, fill="#555", width=1)
        canvas.create_line(ml, y_offset + mt + ph_inner,
                           ml + pw, y_offset + mt + ph_inner, fill="#555", width=1)
        canvas.create_text(ml, y_offset + mt + ph_inner + 8, text="0",
                           fill="#666", font=("TkDefaultFont", 7), anchor="n")
        canvas.create_text(ml + pw, y_offset + mt + ph_inner + 8, text=str(n_max - 1),
                           fill="#666", font=("TkDefaultFont", 7), anchor="n")
        if x_label:
            canvas.create_text(ml + pw // 2, y_offset + mt + ph_inner + 20,
                                text=x_label, fill="#888", font=("TkDefaultFont", 7), anchor="n")

        for vals, color, label in valid_series:
            n = len(vals)
            if n >= 2:
                pts = []
                for i, v in enumerate(vals):
                    pts.append(px(i, n))
                    pts.append(py(v))
                canvas.create_line(pts, fill=color, width=3, smooth=True)
            elif n == 1:
                x0, y0 = px(0, 1), py(vals[0])
                canvas.create_oval(x0 - 3, y0 - 3, x0 + 3, y0 + 3, fill=color, outline="")

        # Legend (top-right inside plot area)
        ly = y_offset + mt + 8
        for vals, color, label in valid_series:
            if not vals:
                continue
            lx = ml + pw - 4
            canvas.create_line(lx - 18, ly + 4, lx - 4, ly + 4, fill=color, width=3)
            canvas.create_text(lx - 22, ly + 4, text=label, fill=color,
                               font=("TkDefaultFont", 7), anchor="e")
            ly += 14

    def _render_dual_chart(self, data: dict) -> None:
        """Two stacked charts: embedding loss (top) + silhouette score (bottom)."""
        self._canvas_content = {"type": "dual_chart", "data": data}
        c = self._display_canvas
        c.delete("all")
        cw = max(c.winfo_width(), 100)
        ch = max(c.winfo_height(), 100)
        c.create_rectangle(0, 0, cw, ch, fill="#1a1a1a", outline="")

        half = ch // 2
        losses = data.get("losses", [])
        ss_vals = data.get("ss", [])

        self._render_line_chart(c, losses, "Embedding Loss",
                                "batch", "#f0a000", y_offset=0, panel_height=half)
        if ss_vals:
            self._render_line_chart(c, ss_vals, "Silhouette Score",
                                    "batch", "#1f8040", y_offset=half, panel_height=half)
            # Target line at 0.91
            ml, mt, mb = 54, 22, 30
            ph_inner = half - mt - mb
            min_ss = min(ss_vals + [0.0])
            max_ss = max(ss_vals + [0.95])
            rng = (max_ss - min_ss) or 1e-8
            y_tgt = half + mt + int((1 - (0.91 - min_ss) / rng) * ph_inner)
            if half + mt <= y_tgt <= half + mt + ph_inner:
                c.create_line(ml, y_tgt, cw - 8, y_tgt,
                              fill="#cc4444", dash=(4, 4), width=1)
                c.create_text(cw - 10, y_tgt - 6, text="0.91",
                              fill="#cc4444", font=("TkDefaultFont", 7), anchor="e")

    def _draw_training_monitor(self, data: dict) -> None:
        self._canvas_content = {"type": "training_monitor", "data": data}
        self._render_training_monitor(data)

    def _render_empty_chart_panel(
        self,
        canvas: tk.Canvas,
        title: str,
        y_offset: int,
        panel_height: int,
        msg: str = "",
    ) -> None:
        cw = max(canvas.winfo_width(), 100)
        ph = max(panel_height, 50)
        canvas.create_rectangle(0, y_offset, cw, y_offset + ph, fill="#1a1a1a", outline="")
        canvas.create_text(cw // 2, y_offset + 11, text=title,
                           fill="#cccccc", font=("TkDefaultFont", 9, "bold"), anchor="center")
        if msg:
            canvas.create_text(cw // 2, y_offset + ph // 2, text=msg,
                               fill="#555555", font=("TkDefaultFont", 9),
                               justify="center", width=max(80, cw - 40), anchor="center")

    def _render_training_monitor(self, data: dict) -> None:
        """Two stacked charts: YOLO loss (top) + weighted mAP fitness (bottom)."""
        self._canvas_content = {"type": "training_monitor", "data": data}
        c = self._display_canvas
        c.delete("all")
        cw = max(c.winfo_width(), 100)
        ch = max(c.winfo_height(), 100)
        c.create_rectangle(0, 0, cw, ch, fill="#1a1a1a", outline="")

        half = ch // 2
        loss_series = data.get("loss_series", [])
        fitness = data.get("fitness", [])

        if loss_series:
            self._render_multi_line_chart(
                c,
                loss_series,
                data.get("loss_title", "Direction Model: Training Loss"),
                data.get("loss_x_label", ""),
                y_offset=0,
                panel_height=half,
            )
        else:
            self._render_empty_chart_panel(c, "Direction Model: Training Loss", 0, half)

        c.create_line(0, half, cw, half, fill="#2d2d2d", width=1)
        if fitness:
            self._render_line_chart(
                c,
                fitness,
                "Weighted mAP Fitness",
                "epoch",
                "#4dbbd5",
                y_offset=half,
                panel_height=ch - half,
            )
        else:
            self._render_empty_chart_panel(
                c,
                "Weighted mAP Fitness",
                half,
                ch - half,
                "Waiting for first epoch mAP",
            )

    # Data readers

    def _read_yolo_results_csv(self, path: str) -> dict:
        train_losses, val_losses, map50_vals, map5095_vals, fitness_vals = [], [], [], [], []

        def metric_key(text: object) -> str:
            key = str(text).strip().lower()
            for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
                key = key.replace(dash, "-")
            return re.sub(r"\s+", "", key)

        def compact_key(text: object) -> str:
            return re.sub(r"[^a-z0-9]+", "", metric_key(text))

        def as_float(value: object) -> float | None:
            try:
                out = float(str(value).strip())
            except (TypeError, ValueError):
                return None
            return out if math.isfinite(out) else None

        def has_map5095(col: object) -> bool:
            key = metric_key(col)
            compact = compact_key(col)
            return "map50-95" in key or "map5095" in compact

        def has_map50(col: object) -> bool:
            return "map50" in compact_key(col) and not has_map5095(col)

        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = list(reader.fieldnames or [])
                train_col = next(
                    (c for c in fields if "train" in metric_key(c) and "box_loss" in metric_key(c)),
                    None,
                )
                val_col = next(
                    (c for c in fields if "val" in metric_key(c) and "box_loss" in metric_key(c)),
                    None,
                )
                map50_col = next((c for c in fields if has_map50(c)), None)
                map5095_col = next((c for c in fields if has_map5095(c)), None)

                for row in reader:
                    train_loss = as_float(row.get(train_col)) if train_col else None
                    val_loss = as_float(row.get(val_col)) if val_col else None
                    map50 = as_float(row.get(map50_col)) if map50_col else None
                    map5095 = as_float(row.get(map5095_col)) if map5095_col else None

                    if train_loss is not None:
                        train_losses.append(train_loss)
                    if val_loss is not None:
                        val_losses.append(val_loss)
                    if map50 is not None:
                        map50_vals.append(map50)
                    if map5095 is not None:
                        map5095_vals.append(map5095)
                    if map50 is not None and map5095 is not None:
                        fitness_vals.append(0.1 * map50 + 0.9 * map5095)
        except Exception:
            pass
        return {
            "train": train_losses,
            "val": val_losses,
            "mAP50": map50_vals,
            "mAP5095": map5095_vals,
            "fitness": fitness_vals,
        }

    def _read_embedding_csv(self, path: str) -> dict:
        losses, ss_vals = [], []
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("event", "")).strip() == "metric":
                        try:
                            losses.append(float(row["loss"]))
                        except (KeyError, ValueError):
                            pass
                        try:
                            ss_vals.append(float(row["ss"]))
                        except (KeyError, ValueError):
                            pass
        except Exception:
            pass
        return {"losses": losses, "ss": ss_vals}

    # Step-specific image finder

    def _step_image_dir_groups(self, step_i: int, session: str) -> list[tuple[str, list[str]]]:
        groups: dict[int, list[tuple[str, list[str]]]] = {
            1: [
                ("preview", [os.path.join(session, "single_animal_images", "preview")]),
                ("image", [os.path.join(session, "single_animal_images", "images")]),
            ],
            2: [
                ("preview", [
                    os.path.join(session, "single_animal_images", "preview"),
                    os.path.join(session, "single_animal_images", "refine", "dataset", "preview"),
                    os.path.join(session, "single_animal_images", "refine_add*", "dataset", "preview"),
                ]),
                ("image", [os.path.join(session, "single_animal_images", "images")]),
            ],
            3: [
                ("preview", [os.path.join(session, "paste_blobs_clustered", "preview")]),
                ("image", [os.path.join(session, "paste_blobs_clustered", "images")]),
            ],
            4: [
                ("preview", [os.path.join(session, "paste_blobs", "preview")]),
                ("image", [os.path.join(session, "paste_blobs", "images")]),
            ],
            5: [
                ("preview", [
                    os.path.join(session, "paste_blobs", "cropping", "preview"),
                    os.path.join(session, "paste_blobs_clustered", "cropping", "preview"),
                    os.path.join(session, "single_animal_images", "cropping", "preview"),
                ]),
                ("image", [
                    os.path.join(session, "paste_blobs", "cropping", "images"),
                    os.path.join(session, "paste_blobs_clustered", "cropping", "images"),
                    os.path.join(session, "single_animal_images", "cropping", "images"),
                ]),
            ],
            6: [
                ("image", [os.path.join(session, "yolo_dataset*", "train", "images")]),
            ],
            8: [
                ("preview", [os.path.join(session, OUTPUT_ROOT_DIR, "*", "tracking", "dataset*", "*", "*", *(["variable"] if self._variable_count.get() else []), "preview")]),
            ],
        }
        return groups.get(step_i, [])

    def _step_image_dirs(self, step_i: int, session: str) -> list[str]:
        return [d for _, dirs in self._step_image_dir_groups(step_i, session) for d in dirs]

    def _collect_image_entries(
        self,
        dirs: list[str],
        cutoff: float | None,
    ) -> list[tuple[float, str]]:
        entries: list[tuple[float, str]] = []
        for pattern in dirs:
            for d in sorted(glob.glob(pattern)):
                if os.path.isdir(d):
                    candidates = (
                        glob.glob(os.path.join(d, "*.png")) +
                        glob.glob(os.path.join(d, "*.jpg")) +
                        glob.glob(os.path.join(d, "*.jpeg"))
                    )
                    for path in candidates:
                        try:
                            mtime = os.path.getmtime(path)
                        except OSError:
                            continue
                        if cutoff is not None and mtime < cutoff:
                            continue
                        entries.append((mtime, path))
        return entries

    def _find_step_image(self, step_i: int, session: str) -> tuple[str, int, int]:
        start_time = None if step_i == 6 else self._step_start_times.get(step_i)
        cutoff = (start_time - 0.5) if start_time is not None else None

        # Collect from all directory groups so preview images and subsequently
        # generated main-output images are both visible as the step progresses.
        all_entries: list[tuple[float, str]] = []
        preview_entries: list[tuple[float, str]] = []
        for kind, dirs in self._step_image_dir_groups(step_i, session):
            found = self._collect_image_entries(dirs, cutoff)
            all_entries.extend(found)
            if kind == "preview":
                preview_entries.extend(found)

        if not all_entries:
            return "", 0, 0

        all_entries.sort(key=lambda item: (item[0], item[1].lower()))
        count = len(all_entries)

        # Prefer the most recent OBB+triangle preview image over a plain output
        # image. Plain images are written far more often than previews (every
        # frame vs. every PREVIEW_INTERVAL-th), so picking the single most
        # recent file overall would almost always be a plain image and the
        # preview would effectively never be shown.
        if preview_entries:
            preview_entries.sort(key=lambda item: (item[0], item[1].lower()))
            chosen_path = preview_entries[-1][1]
        else:
            chosen_path = all_entries[-1][1]

        display_index = count - 1
        return chosen_path, count, display_index + 1

    # Canvas content updaters

    def _show_training_loss(self, session: str) -> None:
        # YOLO writes mAP metrics to results.csv after each validation epoch.
        paths = []
        for stage in TRAINING_STAGE_DIRS:
            paths.extend(glob.glob(
                os.path.join(session, OUTPUT_ROOT_DIR, "*", stage, "dataset*", "results.csv")
            ))
        paths = list(dict.fromkeys(paths))
        try:
            paths = sorted(paths, key=os.path.getmtime, reverse=True)
        except Exception:
            paths = sorted(paths)
        data = self._read_yolo_results_csv(paths[0]) if paths else {}
        fitness = data.get("fitness", [])

        # Prefer live per-batch losses extracted from tqdm (available while training runs)
        if self._batch_losses:
            n = len(self._batch_losses)
            self._draw_training_monitor(
                {
                    "loss_series": [(self._batch_losses, "#1f8040", "train")],
                    "loss_title": "Direction Model: Training Loss (batch)",
                    "loss_x_label": "batch",
                    "fitness": fitness,
                }
            )
            status = f"Batch {n}  |  train/box_loss = {self._batch_losses[-1]:.4f}"
            if fitness:
                status += f"  |  fitness(0.1*mAP50+0.9*mAP50-95) = {fitness[-1]:.4f}"
            self._canvas_status.configure(text=status)
            return

        # Fallback: epoch-level CSV written by YOLO after each epoch
        if not paths:
            self._draw_placeholder("Training in progress...\nLoss chart will appear here.")
            self._canvas_status.configure(text="Waiting for training results...")
            return
        train_losses = data.get("train", [])
        val_losses = data.get("val", [])
        if train_losses:
            series = [
                (train_losses, "#1f8040", "train"),
                (val_losses,   "#f0a000", "val"),
            ]
            self._draw_training_monitor(
                {
                    "loss_series": series,
                    "loss_title": "Direction Model: Training Loss",
                    "loss_x_label": "epoch",
                    "fitness": fitness,
                }
            )
            n = len(train_losses)
            status = f"Epoch {n}  |  train/box_loss = {train_losses[-1]:.4f}"
            if val_losses:
                status += f"  |  val/box_loss = {val_losses[-1]:.4f}"
            if fitness:
                status += f"  |  fitness(0.1*mAP50+0.9*mAP50-95) = {fitness[-1]:.4f}"
            self._canvas_status.configure(text=status)
        else:
            self._draw_placeholder("Training in progress...")
            self._canvas_status.configure(text="Waiting for first epoch...")

    def _show_refine_training_loss(self) -> None:
        n = len(self._refine_losses)
        self._draw_loss_chart(
            self._refine_losses, "Refine Blobs: Training Loss (batch)", "batch", "#4dbbd5"
        )
        self._canvas_status.configure(
            text=f"Batch {n}  |  train/box_loss = {self._refine_losses[-1]:.4f}"
        )

    def _show_embedding_loss(self, session: str) -> None:
        paths = glob.glob(
            os.path.join(
                session, OUTPUT_ROOT_DIR, "*", "tracking", "dataset*", "*", "*",
                *(["variable"] if self._variable_count.get() else []),
                "embedding_training_metrics.csv",
            )
        )
        if not paths:
            self._draw_placeholder(
                "ID Correction running...\n"
                "Embedding chart appears here if embedding is enabled."
            )
            self._canvas_status.configure(text="No embedding metrics yet")
            return
        data = self._read_embedding_csv(paths[0])
        losses = data.get("losses", [])
        ss_vals = data.get("ss", [])
        if losses or ss_vals:
            self._render_dual_chart(data)
            n = len(losses)
            last_ss = ss_vals[-1] if ss_vals else float("nan")
            self._canvas_status.configure(
                text=f"Batch {n}  |  loss = {losses[-1]:.4f}  |  SS = {last_ss:.3f}"
                if losses else "Embedding training starting..."
            )
        else:
            self._draw_placeholder("Embedding training starting...")
            self._canvas_status.configure(text="Waiting for embedding metrics...")

    def _update_canvas_for_step(self, step_i: int, session: str) -> None:
        if session and self._without_direction.get() and os.path.basename(session) != "without_direction_estimation":
            session = os.path.join(session, "without_direction_estimation")
        if step_i < 0 or not session:
            self._draw_placeholder()
            self._canvas_status.configure(text="Waiting...")
            return

        label = STEPS[step_i][0].replace("\n", " ") if step_i < len(STEPS) else ""

        if step_i == 7:
            self._show_training_loss(session)
        elif step_i == 10 and self._overlap.get() == "super_heavy":
            self._show_embedding_loss(session)
        elif step_i == 2 and self._refine_losses:
            self._show_refine_training_loss()
        else:
            img, count, display_no = self._find_step_image(step_i, session)
            if img:
                if img != self._last_display_img:
                    self._show_image_on_canvas(img, label)
                self._canvas_status.configure(
                    text=f"{label}  |  {os.path.basename(img)}  |  image {display_no}/{count}"
                )
            else:
                if self._step_image_dirs(step_i, session):
                    # Keep the previous image visible rather than showing a placeholder
                    last = self._last_display_img
                    if last and os.path.isfile(last):
                        self._show_image_on_canvas(last, os.path.basename(last))
                        self._canvas_status.configure(text=f"Running: {label}")
                    else:
                        self._draw_placeholder(f"{label}\nWaiting for first output image...")
                        self._canvas_status.configure(text=f"Running: {label}")
                    return
                # Fall back to last known image so canvas stays informative
                last = self._last_display_img
                if last and os.path.isfile(last):
                    self._show_image_on_canvas(last, os.path.basename(last))
                    self._canvas_status.configure(text=f"{label} running...")
                else:
                    self._draw_placeholder(f"{label}...")
                    self._canvas_status.configure(text=f"Running: {label}")

    # Polling

    def _canvas_poll(self) -> None:
        """Check for new visual content while the batch runs."""
        self._canvas_poll_id = None
        if self._batch_proc is None or self._batch_proc.poll() is not None:
            return

        step_i = self._active_step
        session = self._base_cfg.get("SESSION_PATH", "")
        live_steps = {7, 10}
        if step_i == 2 and self._refine_losses:
            live_steps.add(2)
        image_steps = {1, 2, 3, 4, 5, 6, 8}

        # Keep live charts fresh, and keep checking image steps so the first
        # generated image appears without waiting for the stage to finish.
        if step_i != self._display_step or step_i in live_steps or step_i in image_steps:
            self._display_step = step_i
            self._update_canvas_for_step(step_i, session)

        interval_ms = 1000 if step_i in live_steps or step_i in image_steps else 3000
        self._canvas_poll_id = self.after(interval_ms, self._canvas_poll)

    # Progress helpers

    def _redraw_progress(self) -> None:
        c = self._progress_canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1:
            return
        c.create_rectangle(0, 0, w, h, fill="#2b2b2b", outline="")
        filled = max(0, int(w * self._progress_pct))
        if filled:
            c.create_rectangle(0, 0, filled, h, fill="#1f8040", outline="")
        c.create_text(w // 2, h // 2, text=self._progress_text, fill="white",
                      font=("TkDefaultFont", 10, "bold"), anchor="center")

    def _set_progress(self, pct: float, text: str) -> None:
        self._progress_pct = max(0.0, min(1.0, pct))
        self._progress_text = text
        self._redraw_progress()

    def _default_skip_flags(self) -> dict[str, bool]:
        return {
            "skip_initial_tracking": False,
            "skip_trajectory_direction_filtering": False,
            "skip_refine_blobs_through_tracking": False,
            "skip_paste_blobs_with_crossing": False,
            "skip_paste_blobs_clustered": False,
            "skip_cropping": False,
            "skip_creating_direction_dataset": False,
            "skip_training": False,
            "skip_detection": False,
            "skip_id_tracking": False,
            "skip_id_correction": False,
            "skip_creating_video": False,
        }

    def _set_skip_flags(self, flags: dict | None, *, render: bool = False) -> None:
        merged = self._default_skip_flags()
        for key, value in (flags or {}).items():
            if key in merged:
                merged[key] = bool(value)
        if merged.get("skip_id_tracking", False):
            merged["skip_id_correction"] = True
        self._skip_flags = merged
        if hasattr(self, "_num_objects_sb"):
            self._apply_individual_count_state()
        if render and hasattr(self, "_blocks_wrap"):
            self._vis_indices = list(range(len(STEPS)))
            self._render_blocks(self._vis_indices)

    def _skip_flags_from_config(self, cfg: dict | None, *, fallback: dict | None = None) -> dict[str, bool]:
        flags = self._default_skip_flags()
        for source in (fallback or {}, cfg or {}):
            for key in flags:
                if key in source:
                    flags[key] = bool(source[key])
        return flags

    @staticmethod
    def _deep_setdefault(target: dict, defaults: dict) -> None:
        for key, value in defaults.items():
            if key not in target:
                target[key] = copy.deepcopy(value)
            elif isinstance(target[key], dict) and isinstance(value, dict):
                EasyTrackingGUI._deep_setdefault(target[key], value)

    def _ensure_config_section(self, cfg: dict, key: str) -> dict:
        section = cfg.get(key)
        if not isinstance(section, dict):
            section = {}
            cfg[key] = section
        return section

    def _current_easy_answers(self) -> dict:
        num_str = self._num_objects_sb.get().strip() if hasattr(self, "_num_objects_sb") else ""
        try:
            num_objects: int | str = int(num_str)
        except (TypeError, ValueError):
            num_objects = num_str
        return {
            "video": self._video_entry.get().strip() if hasattr(self, "_video_entry") else "",
            "session": self._get_session() if hasattr(self, "_session_entry") else "",
            "num_objects": num_objects,
            "variable_count": self._variable_count.get(),
            "without_direction": self._without_direction.get(),
            "overlap": self._overlap.get() if hasattr(self, "_overlap") else "",
            "backward_movement": self._backward_movement.get() if hasattr(self, "_backward_movement") else "",
            "single_animal_paste": self._use_single_animal_paste(),
            "delete_tmp_files": bool(self._delete_tmp_files.get()) if hasattr(self, "_delete_tmp_files") else True,
        }

    def _remember_easy_answers(self) -> None:
        self._loaded_easy_answers = self._current_easy_answers()

    def _on_variable_count_changed(self):
        variable = self._variable_count.get()
        self._num_objects_sb.configure(state="disabled" if variable else "normal")
        self._num_enter_button.configure(state="disabled" if variable else "normal")
        self._num_confirmed = bool(variable)
        if not variable:
            self._confirmed_num_objects = self._parse_num_objects_input() or 1
        self._apply_individual_count_state(reset_multi_paste=True)

    def _parse_num_objects_input(self) -> int | None:
        if not hasattr(self, "_num_objects_sb"):
            return None
        value = self._num_objects_sb.get().strip()
        if not value.isdigit():
            return None
        n = int(value)
        return n if n >= 1 else None

    def _confirm_num_objects(self):
        if self._variable_count.get():
            self._num_confirmed = True
            return "break"
        n = self._parse_num_objects_input()
        if n is None:
            messagebox.showwarning("Invalid number", "Number of animals must be a positive integer.", parent=self)
            return "break"
        self._confirmed_num_objects = int(n)
        self._num_confirmed = True
        self._apply_individual_count_state(reset_multi_paste=True)
        return "break"

    def _invalidate_num_confirm(self):
        self._num_confirmed = False
        self._apply_individual_count_state()

    def _on_num_objects_key_release(self, event):
        if getattr(event, "keysym", "") in {"Return", "KP_Enter"}:
            return
        self._invalidate_num_confirm()

    def _is_single_animal(self) -> bool:
        return bool(not self._variable_count.get() and self._num_confirmed and self._confirmed_num_objects == 1)

    def _use_single_animal_paste(self) -> bool:
        return bool(
            self._is_single_animal()
            and self._single_animal_paste is not None
            and self._single_animal_paste.get()
        )

    def _single_animal_skips_paste(self) -> bool:
        return self._is_single_animal() and not self._use_single_animal_paste()

    def _on_single_animal_paste_changed(self) -> None:
        self._manual_skip_keys.discard("skip_paste_blobs_with_crossing")
        self._manual_skip_keys.discard("skip_paste_blobs_clustered")
        self._apply_individual_count_state(reset_multi_paste=True)

    def _apply_individual_count_state(self, *, reset_multi_paste: bool = False) -> None:
        is_single = self._is_single_animal()
        if not is_single and self._single_animal_paste is not None:
            self._single_animal_paste.set(False)
        if hasattr(self, "_single_animal_paste_checkbox"):
            self._single_animal_paste_checkbox.configure(
                state="normal" if is_single else "disabled"
            )

        enable_multi_questions = bool(
            self._num_confirmed
            and (self._variable_count.get() or self._confirmed_num_objects >= 2 or self._use_single_animal_paste())
        )
        radio_state = "normal" if enable_multi_questions else "disabled"
        for radio in getattr(self, "_overlap_radios", []):
            radio.configure(state=radio_state)

        if hasattr(self, "_num_confirm_hint"):
            if self._variable_count.get():
                self._num_confirm_hint.configure(text="Count inferred from detections", text_color="#1f8040")
            elif not self._num_confirmed:
                self._num_confirm_hint.configure(text="Press Enter to confirm", text_color="gray")
            elif self._single_animal_skips_paste():
                self._num_confirm_hint.configure(text="Single animal: paste steps skipped", text_color="#1f8040")
            elif self._is_single_animal():
                self._num_confirm_hint.configure(text="Single animal: paste steps enabled", text_color="#1f8040")
            else:
                self._num_confirm_hint.configure(text=f"{self._confirmed_num_objects} animals confirmed", text_color="#1f8040")

        paste_keys = ("skip_paste_blobs_with_crossing", "skip_paste_blobs_clustered")
        if self._single_animal_skips_paste():
            for key in paste_keys:
                self._skip_flags[key] = True
        elif enable_multi_questions and reset_multi_paste:
            for key in paste_keys:
                if key not in self._manual_skip_keys:
                    self._skip_flags[key] = False

        if hasattr(self, "_blocks_wrap"):
            self._refresh_block_colors()

    def _warn_individual_count_inconsistency(self, message: str) -> None:
        print(f"[WARN] Easy Tracking individual count state: {message}", flush=True)

    def _apply_backward_question_state(self) -> None:
        """Grey out the backward question while direction estimation is off.

        Direction-free runs never discard an individual for an uncertain
        heading, and refine never deletes on a direction difference, so the
        answer cannot change anything. Advanced Tracking still exposes the
        skip directly for anyone who wants the pass in that mode.
        """
        state = "disabled" if self._without_direction.get() else "normal"
        for radio in getattr(self, "_backward_radios", []):
            radio.configure(state=state)
        self._sync_refine_skip_from_backward_question()

    def _sync_refine_skip_from_backward_question(self) -> None:
        if not hasattr(self, "_backward_movement"):
            return
        answered_yes = self._backward_movement.get() == "yes"
        if self._without_direction.get():
            answered_yes = False
        self._skip_flags["skip_refine_blobs_through_tracking"] = not answered_yes
        if hasattr(self, "_blocks_wrap"):
            self._refresh_block_colors()

    def _step_is_skipped(self, step_i: int) -> bool:
        if step_i < 0 or step_i >= len(STEPS):
            return False
        _, skip_key, _ = STEPS[step_i]
        if skip_key == "skip_cropping" and self._skip_flags.get("skip_creating_direction_dataset", False):
            return True
        if skip_key == "skip_id_correction" and self._skip_flags.get("skip_id_tracking", False):
            return True
        return bool(self._skip_flags.get(skip_key, False))

    def _refresh_block_colors(self) -> None:
        for vis_i, step_i in enumerate(self._vis_indices):
            if step_i in self._done_steps:
                state = "done"
            elif step_i == self._active_step:
                state = "running"
            else:
                state = "disabled" if self._step_is_skipped(step_i) else "inactive"
            if vis_i == self._selected_block and state == "inactive":
                state = "selected"
            self._set_block_color(vis_i, state)

    def _render_blocks(self, vis_indices: list[int]) -> None:
        self._cancel_shimmer()
        for w in self._blocks_wrap.winfo_children():
            w.destroy()
        self._block_frames = []
        self._block_text_ids = []
        self._block_states = []

        if not vis_indices:
            tk.Label(self._blocks_wrap, text="(No process blocks)",
                     bg="#1e1e1e", fg="#555555", font=("TkDefaultFont", 9)).pack(pady=8)
            return

        grid = tk.Frame(self._blocks_wrap, bg=_C_BG_ROW)
        grid.pack(fill="x", padx=4, pady=6)
        for col in range(6):
            grid.grid_columnconfigure(col, weight=1)

        for vis_i, step_i in enumerate(vis_indices):
            lbl_text = STEPS[step_i][0]
            blk = tk.Canvas(
                grid, width=112, height=58, bg=_C_INACTIVE,
                highlightthickness=1, highlightbackground="#555555",
                bd=0, relief="flat",
            )
            blk.grid(row=vis_i // 6, column=vis_i % 6, padx=3, pady=3)
            text_id = blk.create_text(
                56, 29, text=lbl_text, fill="#aaaaaa",
                font=("TkDefaultFont", 8), justify="center", anchor="center",
            )
            blk.bind("<Button-1>", lambda _e, vi=vis_i: self._toggle_skip_block(vi))
            self._block_frames.append(blk)
            self._block_text_ids.append(text_id)
            self._block_states.append("inactive")
            state = "disabled" if self._step_is_skipped(step_i) else "inactive"
            self._set_block_color(vis_i, state)

    def _set_block_color(self, vis_i: int, state: str) -> None:
        if vis_i < 0 or vis_i >= len(self._block_frames):
            return
        step_i = self._vis_indices[vis_i] if vis_i < len(self._vis_indices) else -1
        if state in {"inactive", "selected"} and step_i >= 0 and self._step_is_skipped(step_i):
            state = "disabled"
        palette = {
            "inactive": (_C_INACTIVE, "#aaaaaa"),
            "running":  (_C_RUNNING,  "#ffffff"),
            "done":     (_C_DONE,     "#aaffaa"),
            "selected": (_C_SELECTED, "#aaaaaa"),
            "disabled": (_C_DISABLED, "#555555"),
        }
        bg, fg = palette.get(state, (_C_INACTIVE, "#aaaaaa"))
        canvas = self._block_frames[vis_i]
        canvas.delete("shimmer")
        canvas.configure(bg=bg, highlightbackground="#555555")
        canvas.itemconfigure(self._block_text_ids[vis_i], fill=fg)
        canvas.tag_raise(self._block_text_ids[vis_i])
        self._block_states[vis_i] = state

    def _toggle_skip_block(self, vis_i: int) -> None:
        if vis_i >= len(self._block_frames):
            return
        if self._batch_proc is not None and self._batch_proc.poll() is None:
            return
        if vis_i >= len(self._vis_indices):
            return
        step_i = self._vis_indices[vis_i]
        skip_key = STEPS[step_i][1]
        if self._single_animal_skips_paste() and skip_key in {"skip_paste_blobs_with_crossing", "skip_paste_blobs_clustered"}:
            return
        if skip_key == "skip_cropping":
            return
        new_value = not bool(self._skip_flags.get(skip_key, False))
        self._skip_flags[skip_key] = new_value
        self._manual_skip_keys.add(skip_key)
        if skip_key == "skip_id_tracking" and new_value:
            self._skip_flags["skip_id_correction"] = True
            self._manual_skip_keys.add("skip_id_correction")
        if skip_key == "skip_id_correction" and not new_value:
            self._skip_flags["skip_id_tracking"] = False
            self._manual_skip_keys.add("skip_id_tracking")

        self._selected_block = vis_i
        self._refresh_block_colors()

    @staticmethod
    def _mix_color(c1: str, c2: str, ratio: float) -> str:
        ratio = max(0.0, min(1.0, ratio))
        a = tuple(int(c1.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        b = tuple(int(c2.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        mixed = tuple(int(x + (y - x) * ratio) for x, y in zip(a, b))
        return "#{:02x}{:02x}{:02x}".format(*mixed)

    def _start_shimmer(self, vis_i: int) -> None:
        self._active_block_vis = vis_i
        self._shimmer_phase = 0.0
        if self._shimmer_after_id is None:
            self._animate_shimmer()

    def _cancel_shimmer(self) -> None:
        if self._shimmer_after_id is not None:
            try:
                self.after_cancel(self._shimmer_after_id)
            except Exception:
                pass
            self._shimmer_after_id = None
        for canvas in self._block_frames:
            try:
                canvas.delete("shimmer")
            except Exception:
                pass

    def _animate_shimmer(self) -> None:
        self._shimmer_after_id = None
        vis_i = self._active_block_vis
        if (
            vis_i < 0 or
            vis_i >= len(self._block_frames) or
            self._block_states[vis_i] != "running"
        ):
            return

        canvas = self._block_frames[vis_i]
        canvas.delete("shimmer")
        w = max(1, canvas.winfo_width())
        h = max(1, canvas.winfo_height())
        stripe = 4
        for x in range(0, w, stripe):
            wave = (math.sin(((x / max(w, 1)) - self._shimmer_phase) * math.tau) + 1.0) / 2.0
            color = self._mix_color(_C_RUNNING, "#6ee08c", 0.18 + wave * 0.42)
            canvas.create_rectangle(x, 0, min(x + stripe, w), h, fill=color, outline="", tags="shimmer")
        canvas.tag_raise(self._block_text_ids[vis_i])
        self._shimmer_phase = (self._shimmer_phase + 0.045) % 1.0
        self._shimmer_after_id = self.after(60, self._animate_shimmer)

    def _overlap_values(self) -> tuple[float, float]:
        return {
            "super_heavy": (0.01, 0.5),
            "heavy": (0.01, 0.5),
            "light": (0.01, 0.2),
        }.get(
            self._overlap.get(), (0.01, 0.5)
        )

    def _video_dims(self, path: str) -> tuple[int, int]:
        if not _HAS_CV2 or not os.path.isfile(path):
            return 0, 0
        try:
            cap = cv2.VideoCapture(path)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (w, h) if w > 0 and h > 0 else (0, 0)
        except Exception:
            return 0, 0

    @staticmethod
    def _img_size_from_short(short: int) -> int:
        return 1024 if short >= 1200 else max(32, (short // 32) * 32)

    def _derive_dataset_geometry_flags(self, video_path: str, image_size: int) -> tuple[bool, int, bool]:
        try:
            image_size = max(32, int(image_size))
        except (TypeError, ValueError):
            image_size = 640

        w, h = self._video_dims(video_path)
        if w <= 0 or h <= 0:
            return False, 2, True

        short = min(w, h)
        long_ = max(w, h)
        is_square = (w == h)

        if is_square and 0 <= short - image_size < 32:
            return True, 1, True
        if is_square and short <= image_size:
            return True, 1, True

        tiles_long = max(1, long_ // image_size)
        tiles_short = max(1, short // image_size)
        tile_count = max(1, min(4, tiles_long * tiles_short))
        num_crops = max(1, math.ceil(tile_count / 2))
        include_full = short >= 1200
        return False, num_crops, include_full

    def _check_single_animal_config_consistency(self, cfg: dict) -> None:
        if cfg.get("VARIABLE_NUM_OBJECTS", False):
            return
        try:
            num_objects = int(cfg.get("NUM_OBJECTS", 1))
        except (TypeError, ValueError):
            self._warn_individual_count_inconsistency("NUM_OBJECTS is not an integer in generated config.")
            return

        single_paste_enabled = bool(cfg.get("SINGLE_PASTE", False))
        expected_radio_state = "disabled" if num_objects == 1 and not single_paste_enabled else "normal"
        if self._num_confirmed and (
            self._confirmed_num_objects >= 2
            or (self._confirmed_num_objects == 1 and self._use_single_animal_paste())
        ):
            expected_radio_state = "normal"
        elif not self._num_confirmed:
            expected_radio_state = "disabled"
        for radio in getattr(self, "_overlap_radios", []):
            if str(radio.cget("state")) != expected_radio_state:
                self._warn_individual_count_inconsistency(
                    f"overlap radio state is {radio.cget('state')}, expected {expected_radio_state}."
                )
                break

        if num_objects != 1 or single_paste_enabled:
            return

        checks = [
            ("skip_paste_blobs_with_crossing", cfg.get("skip_paste_blobs_with_crossing") is True),
            ("skip_paste_blobs_clustered", cfg.get("skip_paste_blobs_clustered") is True),
            ("EMBEDDING.ENABLE", ((cfg.get("EMBEDDING", {}) or {}).get("ENABLE") is False)),
        ]
        for label, ok in checks:
            if not ok:
                self._warn_individual_count_inconsistency(f"{label} is inconsistent for NUM_OBJECTS=1.")
        for step_name, step_key in (
            ("Clustered Paste", "skip_paste_blobs_clustered"),
            ("Mixed Paste", "skip_paste_blobs_with_crossing"),
        ):
            step_i = next((i for i, item in enumerate(STEPS) if item[1] == step_key), -1)
            if step_i >= 0 and not self._step_is_skipped(step_i):
                self._warn_individual_count_inconsistency(f"{step_name} block is not skipped for NUM_OBJECTS=1.")

    def _build_config(self) -> dict | None:
        video = self._video_entry.get().strip()
        if not video:
            messagebox.showerror("Error", "Please select a video file.")
            return None
        session = self._get_session()
        if not session:
            messagebox.showerror("Error", "Session path could not be determined.")
            return None

        if not self._num_confirmed:
            self._confirm_num_objects()
        if not self._num_confirmed:
            return None
        num_objects = 2 if self._variable_count.get() else int(self._confirmed_num_objects)

        cfg: dict = copy.deepcopy(self._loaded_cfg) if isinstance(self._loaded_cfg, dict) else {}
        has_loaded_config = bool(self._loaded_config_path) or bool(self._loaded_cfg)
        current_answers = self._current_easy_answers()
        loaded_answers = self._loaded_easy_answers or {}
        force_easy_values = not has_loaded_config

        def answer_changed(*keys: str) -> bool:
            return force_easy_values or any(current_answers.get(key) != loaded_answers.get(key) for key in keys)

        flags = self._skip_flags_from_config({}, fallback=self._skip_flags)
        min_overlap, max_overlap = self._overlap_values()
        embedding_enabled = self._overlap.get() == "super_heavy"
        match_iou = 0.5
        interact_iou = FIXED_INTERACT_IOU
        dir_min_sec = 0.5
        dir_min_disp = 1.0
        refine_frame_ratio = 1.0
        run_delete_ratio = 0.4
        single_animal_paste = self._use_single_animal_paste()
        if self._single_animal_skips_paste():
            flags["skip_paste_blobs_with_crossing"] = True
            flags["skip_paste_blobs_clustered"] = True
            embedding_enabled = False

        # Auto-derive image size & crop settings from video geometry
        vw, vh = self._video_dims(video)
        if vw > 0 and vh > 0:
            short = min(vw, vh)
            img_size = self._img_size_from_short(short)
        else:
            img_size = 640

        try:
            geometry_img_size = img_size if answer_changed("video") else int(cfg.get("TRAIN_IMG_SIZE", img_size))
        except (TypeError, ValueError):
            geometry_img_size = img_size
        skip_crop, num_crops, include_full = self._derive_dataset_geometry_flags(video, geometry_img_size)
        localized = False

        # Derived paths: always <session>/segmentation/, matching where
        # segmentation itself writes (see gui_segmentation.py's _default_output_dir).
        pickle_path, background_path = segmentation_paths_for_session(session, video)
        seg_meta = read_segmentation_metadata(pickle_path)
        meta_training_start = int(seg_meta["training_frame_start"]) if seg_meta else 0
        meta_training_end = int(seg_meta["training_frame_end"]) if seg_meta else -1
        meta_frame_interval = int(seg_meta["training_frame_interval"]) if seg_meta else 5

        # Cluster frames count
        try:
            total_images = int(cfg.get("NUM_IMAGES", 10000))
        except (TypeError, ValueError):
            total_images = 10000
        effective_skip_crop = skip_crop if answer_changed("video") else bool(flags.get("skip_cropping", skip_crop))
        effective_include_full = include_full
        if not answer_changed("video") and "USE_FULL" in cfg:
            effective_include_full = bool(cfg.get("USE_FULL"))
        images_per_frame = (0 if effective_skip_crop else num_crops) + (1 if effective_include_full else 0)
        cluster_frames = max(1, math.ceil(total_images * 0.05 / max(1, images_per_frame)))

        default_cfg = {
            "SESSION_PATH":               session,
            "TRAINING_VIDEO_PATH":        video,
            "TRACKING_VIDEO_PATH":        video,
            "NUM_OBJECTS":                num_objects,
            "TRAIN_IMG_SIZE":             geometry_img_size,
            "TRACKING_VIDEO_PATH_IS_DIR": False,
            "AUTO_PARAMS":                True,
            "LOCALIZED_RATIO":            0.9,
            "MIN_OVERLAP":                min_overlap,
            "MAX_OVERLAP":                max_overlap,
            "DIR_MIN_SEC":                dir_min_sec,
            "WIDTH_SCALE_MIN":            _WIDTH_SCALE_MIN,
            "WIDTH_SCALE_MAX":            _WIDTH_SCALE_MAX,
            "NUM_CROPS":                  num_crops,
            "LOCALIZED":                  localized,
            "USE_FULL":                   effective_include_full,
            "USE_CROP":                   not skip_crop,
            "CLUSTER_FRAMES":             cluster_frames,
            "INIT_CSV_PATH":   os.path.join(session, "initial_tracking", "track_assignments.csv"),
            "SINGLE_PASTE":    single_animal_paste,
        }

        # Loaded configs keep their existing values; these defaults only fill keys
        # that are absent, or provide the complete config for a new Easy run.
        default_cfg.update({
            "NUM_WORKERS":             "auto",
            "NUM_PREVIEW_FRAMES":      10,
            "PREVIEW_INTERVAL":        100,
            "FRAME_INTERVAL":          meta_frame_interval,
            "NUM_IMAGES":              10000,
            "RANDOM_SEED":             0,
            "delete_tmp_files":        bool(self._delete_tmp_files.get()) if hasattr(self, "_delete_tmp_files") else True,

            # Initial tracking
            "INIT_MAX_GAP":            1,
            "SKIP_INIT_PREVIEW":       False,

            # Direction filtering
            "TRAJ_MAX_DIST":           2.0,
            "TRAJ_MAX_JUMP":           2.0,
            "MIN_ASPECT":              1.1,
            "DIR_MIN_DISP":            dir_min_disp,
            "SKIP_DIR_PREVIEW":        False,

            # Refine blobs
            "REFINE_FRAME_RATIO":      refine_frame_ratio,
            "REFINE_EPOCHS":           5,
            "REFINE_BATCH":            "auto",
            "REFINE_ITERS":            1,
            "REFINE_MODEL":            "yolo11n-obb",
            "REFINE_CONF":             0.2,
            "RUN_DELETE_RATIO":        run_delete_ratio,
            "SKIP_REFINE_PREVIEW":     False,

            # Paste composition
            "RATIO_SINGLE":            0.1,
            "RATIO_P2":                0.5,
            "RATIO_P3":                0.4,
            "FREE_SCALE":              0.25,
            "FREE_RATIO_SINGLE":       0.0,
            "FREE_RATIO_P2":           1.0,
            "FREE_RATIO_P3":           1.0,

            # Paste placement / appearance
            "MAX_TRIES":               100,
            "PASTE_SCALE_MIN":         0.9,
            "PASTE_SCALE_MAX":         1.1,
            "PASTE_MIN_ASPECT":        1.1,
            "PASTE_LAYER_MODE":        "mixed",
            "UNDER_PASTE_PROB":        0.5,
            "OCCLUDER_MARGIN":         -2,
            "ALPHA_MODE":              "distance",
            "FEATHER_MIN":             2,
            "FEATHER_MAX":             4,
            "EDGE_BLUR_KSIZE":         7,
            "EDGE_BLUR_SIGMA":         11.0,
            "BRIGHT_MIN":              0.90,
            "BRIGHT_MAX":              1.10,
            "CONTRAST_MIN":            0.90,
            "CONTRAST_MAX":            1.10,

            # Clustered paste
            "CLUSTERED_RATIO":         0.05,
            "CLUSTER_COUNT":           12,
            "CLUSTER_FIT_LONG":        0.8,
            "CLUSTER_FIT_SHORT":       0.8,
            "CLUSTER_BREAK_PROB":      0.20,

            # Dataset
            "VAL_RATIO":               0.05,
        })

        default_cfg["training"] = {
            "EPOCHS":            50,
            "LR0":               DEFAULT_LR0,
            "LRF":               DEFAULT_LRF,
            "SAVE_PERIOD":       5,
            "BATCH_SIZE":        "auto",
            "PRETRAINED_MODEL":  "yolo11n-obb",
            "DEVICE":            "auto",
            "FIRST_FRAME":       meta_training_start,
            "LAST_FRAME":        meta_training_end,
        }
        default_cfg["analysis"] = {
            "DEVICE":                    "auto",
            "BATCH_SIZE":                "auto",
            "WEIGHT":                    "last",
            "CONF":                      0.1,
            "NMS_IOU":                   0.8,
            "SKIP_DETECT_PREVIEW":       False,
            "MATCH_IOU":                 match_iou,
            "MATCH_ANGLE":               90.0,
            "MAX_AXIS_ERR":              45.0,
            "MAX_AGE":                   10,
            "FLIP_SEC":                  5.0,
            "TRACKING_COST": {
                "IOU_WEIGHT":            1.0,
                "DIRECTION_WEIGHT":      1.0,
                "MISS_WEIGHT":           1.0,
                "DISTANCE_WEIGHT":       1.0,
            },
            "FIRST_FRAME":       0,
            "LAST_FRAME":        -1,
        }
        default_cfg["EMBEDDING"] = {
            "ENABLE":                    embedding_enabled,
            "DEVICE":                    "auto",
            "IMG_SIZE":                  "auto",
            "PREVIEW_COUNT":             100,
            "INTERACT_IOU":              interact_iou,
        }
        default_cfg["create_video"] = {
            "WEIGHT":               "last",
            "FRAME_STEP":           1,
            "FPS":                  None,
            "ACCELERATION":         "cpu",
            "EXPORT_RAW":           False,
            "EXPORT_IMAGES":        False,
            "IMAGE_FORMAT":         "jpeg",
            "DRAW_OBB":             True,
            "DRAW_MODE":            "obb",
            "DRAW_LABELS":          False,
            "DRAW_ARROW":           True,
            "OBB_WIDTH":            1,
            "ARROW_WIDTH":          0,
            "ARROW_ALPHA":          0.3,
            "ARROW_SCALE":          1.2,
            "LABEL_SCALE":          0.5,
            "LABEL_THICKNESS":      1,
        }

        self._deep_setdefault(cfg, default_cfg)
        cfg["AUTO_PARAMS"] = True
        cfg["EMBEDDING"]["INTERACT_IOU"] = interact_iou
        cfg["DIR_MIN_DISP"] = 1.0
        cfg["REFINE_FRAME_RATIO"] = 1.0
        cfg["RUN_DELETE_RATIO"] = 0.4
        # Always refresh, regardless of whether the video/session answers
        # changed: these are pure derived lookups, never user-customized, and
        # a stale value (e.g. relativized against a since-changed SESSION_PATH)
        # must not survive a save/build.
        cfg["PICKLE_PATH"] = pickle_path
        cfg["BACKGROUND_PATH"] = background_path

        if answer_changed("video"):
            cfg.update({
                "TRAINING_VIDEO_PATH":        video,
                "TRACKING_VIDEO_PATH":        video,
                "TRACKING_VIDEO_PATH_IS_DIR": False,
                "TRAIN_IMG_SIZE":             img_size,
                "NUM_CROPS":                  num_crops,
                "LOCALIZED":                  False,
                "FREE_SCALE":                 0.25,
                "USE_FULL":                   include_full,
                "USE_CROP":                   not skip_crop,
                "CLUSTER_FRAMES":             cluster_frames,
            })
            cfg["FRAME_INTERVAL"] = meta_frame_interval
            training_cfg = self._ensure_config_section(cfg, "training")
            training_cfg["FIRST_FRAME"] = meta_training_start
            training_cfg["LAST_FRAME"] = meta_training_end
            flags["skip_cropping"] = skip_crop
        if answer_changed("session"):
            cfg["SESSION_PATH"] = session
            cfg["INIT_CSV_PATH"] = os.path.join(
                session, "initial_tracking", "track_assignments.csv"
            )
        if answer_changed("num_objects", "variable_count"):
            cfg["NUM_OBJECTS"] = num_objects
        if answer_changed("overlap"):
            cfg["MIN_OVERLAP"] = min_overlap
            cfg["MAX_OVERLAP"] = max_overlap
            self._ensure_config_section(cfg, "EMBEDDING")["ENABLE"] = embedding_enabled
        if answer_changed("single_animal_paste"):
            cfg["SINGLE_PASTE"] = single_animal_paste
        if answer_changed("delete_tmp_files"):
            cfg["delete_tmp_files"] = bool(self._delete_tmp_files.get())

        if num_objects == 1 and not single_animal_paste:
            flags["skip_paste_blobs_with_crossing"] = True
            flags["skip_paste_blobs_clustered"] = True
            self._ensure_config_section(cfg, "EMBEDDING")["ENABLE"] = False

        cfg["VARIABLE_NUM_OBJECTS"] = self._variable_count.get()
        cfg["WITHOUT_DIRECTION_ESTIMATION"] = self._without_direction.get()
        cfg.update(flags)
        self._check_single_animal_config_consistency(cfg)
        return cfg

    def _write_config(self, cfg: dict, path: str | None = None) -> str | None:
        session = cfg["SESSION_PATH"]
        try:
            if path is None:
                path = self._loaded_config_path or os.path.join(session, "config.yaml")
            target_dir = os.path.dirname(path) or session
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            temp_path = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
            with open(temp_path, "w", encoding="utf-8") as f:
                yaml.dump(relativize_config_paths(cfg), f, sort_keys=False, allow_unicode=True)
            os.replace(temp_path, path)
            self._loaded_cfg = copy.deepcopy(cfg)
            self._loaded_config_path = path
            self._remember_easy_answers()
            return path
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{e}")
            return None

    def _save_config(self) -> None:
        cfg = self._build_config()
        if cfg is None:
            return
        path = self._write_config(cfg)
        if path is not None:
            saved_flags = {key: cfg.get(key, self._skip_flags.get(key, False)) for key in self._default_skip_flags()}
            batch_running = self._batch_proc is not None and self._batch_proc.poll() is None
            self._set_skip_flags(saved_flags, render=not batch_running)
            messagebox.showinfo("Saved", f"Config saved to:\n{path}")

    # Batch execution
    def _run_easy_tracking(self) -> None:
        cfg = self._build_config()
        if cfg is None:
            return
        self._start_batch(cfg)

    def _start_batch(self, cfg: dict, cfg_path: str | None = None) -> None:
        if cfg_path is None:
            cfg_path = self._write_config(cfg)
        if cfg_path is None:
            return

        loaded_flags = {key: cfg.get(key, self._skip_flags.get(key, False)) for key in self._default_skip_flags()}
        self._set_skip_flags(loaded_flags, render=False)
        self._base_cfg = copy.deepcopy(cfg)
        self._vis_indices = list(range(len(STEPS)))
        self._done_steps = set()
        self._active_step = -1
        self._active_block_vis = -1
        self._selected_block = -1

        self._render_blocks(self._vis_indices)
        self._set_progress(0.0, "Starting...")
        self._batch_log_path = ""
        self._stop_requested = False

        self._run_btn.configure(state="disabled")
        self._switch_btn.configure(state="disabled")
        self._direction_mode_checkbox.configure(state="disabled")
        self._save_btn.configure(state="normal")
        self._load_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        kw = {"creationflags": flags} if sys.platform == "win32" else {"start_new_session": True}
        env = with_pythonpath(os.environ.copy(), PROJECT_ROOT, MAIN_PATH)
        # Keep raw tqdm redraws available to this GUI for its live loss chart,
        # while batch.py still writes a compact persistent log.
        env["AMADEUS_GUI_PROGRESS_PROTOCOL"] = "1"
        self._batch_proc = subprocess.Popen(
            [sys.executable, "-u", BATCH_SCRIPT, cfg_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env,
            **kw,
        )
        threading.Thread(target=self._monitor_batch, daemon=True).start()

    def _echo_tqdm_line(self, line: str, previous_len: int) -> int:
        import shutil
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

            n, tot = int(match.group(1)), int(match.group(2))
            pct = n / max(tot, 1)
            desc_m = re.search(r"\s(\d+)%\|", line)
            if desc_m:
                prefix = line[:desc_m.start()]
                colon = prefix.rfind(":")
                desc = prefix[:colon].strip() if colon >= 0 else prefix.strip()
            else:
                desc = ""
            text = f"{desc}  {n}/{tot}" if desc else f"{n}/{tot}"
            self.after(0, lambda p=pct, t=text: self._on_tqdm_progress(p, t))

            # Extract per-batch box_loss from YOLO tqdm lines (training and refine_blobs)
            if self._active_step in (7, 2):
                loss_m = _YOLO_LOSS_PAT.search(line)
                if loss_m:
                    try:
                        box_loss = float(loss_m.group(1))
                        if self._active_step == 7:
                            self._batch_losses.append(box_loss)
                        else:
                            self._refine_losses.append(box_loss)
                    except ValueError:
                        pass
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
            step_i = SCRIPT_TO_IDX.get(script, -1)
            if step_i >= 0 and step_i in self._vis_indices:
                vi = self._vis_indices.index(step_i)
                self.after(0, lambda vi=vi, s=script: self._on_step_start(vi, s))
        elif "] END:   " in line:
            script = line.split("] END:   ", 1)[-1].split("  (")[0].strip()
            step_i = SCRIPT_TO_IDX.get(script, -1)
            if step_i >= 0 and step_i in self._vis_indices:
                vi = self._vis_indices.index(step_i)
                state["done_count"] = int(state.get("done_count", 0)) + 1
                self.after(0, lambda vi=vi, si=step_i, s=script:
                           self._on_step_done(vi, si, s))

    def _monitor_batch(self) -> None:
        state = {"done_count": 0, "last_was_tqdm": False, "last_tqdm_len": 0}
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buf: list[str] = []
        last_delim_was_cr = False
        stdout = self._batch_proc.stdout

        def flush_record() -> None:
            nonlocal buf
            line = "".join(buf)
            buf = []
            self._handle_batch_output_record(line, state)

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
        self._batch_proc.wait()
        rc = self._batch_proc.returncode
        self.after(0, lambda: self._on_batch_finished(rc))

    def _on_tqdm_progress(self, pct: float, text: str) -> None:
        self._set_progress(pct, text)

    def _on_step_start(self, vis_i: int, script: str) -> None:
        self._active_step = self._vis_indices[vis_i] if vis_i < len(self._vis_indices) else -1
        if self._active_step >= 0:
            self._step_start_times[self._active_step] = time.time()
        if script == "obb_detector_training":
            self._batch_losses = []
        elif script == "direction_class_filtering":
            self._refine_losses = []
        self._display_step = -2
        self._set_block_color(vis_i, "running")
        self._start_shimmer(vis_i)
        self._set_progress(0.0, script + "...")
        # Kick off canvas polling if not already running
        if self._canvas_poll_id is None:
            self._canvas_poll()

    def _on_step_done(self, vis_i: int, step_i: int, script: str) -> None:
        self._done_steps.add(step_i)
        if self._active_block_vis == vis_i:
            self._active_block_vis = -1
            self._cancel_shimmer()
        self._set_block_color(vis_i, "done")
        self._set_progress(1.0, script + " done")
        # Refresh canvas to show the just-completed step's output
        session = self._base_cfg.get("SESSION_PATH", "")
        self._display_step = -2         # force re-render on next poll
        self._update_canvas_for_step(step_i, session)

    def _on_batch_finished(self, returncode: int) -> None:
        stopped_by_user = self._stop_requested
        self._stop_requested = False
        # Cancel canvas polling
        if self._canvas_poll_id is not None:
            try:
                self.after_cancel(self._canvas_poll_id)
            except Exception:
                pass
            self._canvas_poll_id = None

        self._batch_proc = None
        self._active_step = -1
        self._active_block_vis = -1
        self._cancel_shimmer()

        # Reset block state so the user can edit skip flags immediately
        self._done_steps = set()
        self._selected_block = -1
        flags = {key: self._base_cfg.get(key, False) for key in self._default_skip_flags()}
        self._set_skip_flags(flags, render=True)

        self._run_btn.configure(state="normal")
        self._switch_btn.configure(state="normal")
        self._direction_mode_checkbox.configure(state="normal")
        self._save_btn.configure(state="normal")
        self._load_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        if returncode == 0:
            self._set_progress(1.0, "Complete!")
            self._canvas_status.configure(text="Processing complete!")
        elif stopped_by_user:
            self._set_progress(0.0, "Stopped")
            self._canvas_status.configure(text=f"Stopped (exit {returncode})")
        else:
            self._set_progress(0.0, "Processing failed")
            log_path = self._batch_log_path or "unavailable"
            self._canvas_status.configure(text=f"Processing failed (exit {returncode})")
            messagebox.showerror(
                "Processing failed",
                f"Processing failed with exit code {returncode}.\n\n"
                f"Full log:\n{log_path}",
            )

    def _stop_batch(self) -> None:
        if self._batch_proc and self._batch_proc.poll() is None:
            self._stop_requested = True
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(self._batch_proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                else:
                    import signal
                    os.killpg(os.getpgid(self._batch_proc.pid), signal.SIGTERM)
            except Exception:
                pass
        self._stop_btn.configure(state="disabled")

    # Load / apply config

    def _load_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Load config",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_config_path(path)

    def _load_config_path(self, path: str, *, show_msg: bool = False) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg = prepare_config_for_gui(cfg, path, parent=self)
            # PICKLE_PATH/BACKGROUND_PATH always live under SESSION_PATH; always
            # recompute rather than trust whatever an old config.yaml stored, so
            # a stale value never survives a load.
            cfg["PICKLE_PATH"], cfg["BACKGROUND_PATH"] = segmentation_paths_for_session(
                cfg.get("SESSION_PATH", ""),
                cfg.get("TRAINING_VIDEO_PATH", ""),
            )
            self._loaded_cfg = copy.deepcopy(cfg)
            self._loaded_config_path = path
            self._apply_config_to_gui(cfg)
            if show_msg:
                messagebox.showinfo("Loaded", f"Config loaded from:\n{path}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")
            return False

    def _apply_config_to_gui(self, cfg: dict) -> None:
        cfg = cfg or {}
        self._manual_skip_keys.clear()

        # Paths
        video = cfg.get("TRAINING_VIDEO_PATH", "")
        if video:
            self._video_entry.delete(0, tk.END)
            self._video_entry.insert(0, video)
        session = cfg.get("SESSION_PATH", "")
        if session:
            self._session_entry.delete(0, tk.END)
            self._session_entry.insert(0, session)

        # Number of animals
        self._variable_count.set(bool(cfg.get("VARIABLE_NUM_OBJECTS", False)))
        self._without_direction.set(bool(cfg.get("WITHOUT_DIRECTION_ESTIMATION", False)))
        self._num_objects_sb.configure(state="normal")
        num_objects = cfg.get("NUM_OBJECTS", 1)
        self._num_objects_sb.delete(0, tk.END)
        num_objects = int(num_objects)
        self._num_objects_sb.insert(0, str(num_objects))
        self._confirmed_num_objects = num_objects
        self._num_confirmed = True
        if self._single_animal_paste is not None:
            self._single_animal_paste.set(bool(cfg.get("SINGLE_PASTE", False)))
        self._apply_individual_count_state()

        self._num_objects_sb.configure(state="disabled" if self._variable_count.get() else "normal")
        self._num_enter_button.configure(state="disabled" if self._variable_count.get() else "normal")

        # Very heavy is heavy overlap plus embedding-based ID validation.
        try:
            max_overlap = float(cfg.get("MAX_OVERLAP", 0.5))
        except (TypeError, ValueError):
            max_overlap = 0.5
        embedding_cfg = cfg.get("EMBEDDING", {}) or {}
        raw_embedding_enabled = embedding_cfg.get("ENABLE", False)
        if isinstance(raw_embedding_enabled, bool):
            embedding_enabled = raw_embedding_enabled
        elif isinstance(raw_embedding_enabled, (int, float)):
            embedding_enabled = bool(raw_embedding_enabled)
        elif isinstance(raw_embedding_enabled, str):
            embedding_enabled = raw_embedding_enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            embedding_enabled = bool(raw_embedding_enabled)
        if embedding_enabled:
            self._overlap.set("super_heavy")
        elif max_overlap <= 0.2:
            self._overlap.set("light")
        else:
            self._overlap.set("heavy")
        refine_skipped = cfg.get("skip_refine_blobs_through_tracking", False)
        self._backward_movement.set("no" if bool(refine_skipped) else "yes")
        self._delete_tmp_files.set(bool(cfg.get("delete_tmp_files", True)))

        loaded_flags = self._skip_flags_from_config(cfg)
        video = self._video_entry.get().strip()
        try:
            loaded_img_size = int(cfg.get("TRAIN_IMG_SIZE", 640))
        except (TypeError, ValueError):
            loaded_img_size = 640
        vw, vh = self._video_dims(video)
        if vw > 0 and vh > 0:
            loaded_img_size = self._img_size_from_short(min(vw, vh))
        self._manual_skip_keys.clear()
        self._set_skip_flags(loaded_flags, render=True)
        self._sync_segmentation_status_from_outputs(cfg)
        self._remember_easy_answers()

    # Advanced GUI

    def _existing_config_path(self) -> str:
        if self._loaded_config_path and os.path.isfile(self._loaded_config_path):
            return self._loaded_config_path
        session = self._get_session()
        if session:
            candidate = os.path.join(session, "config.yaml")
            if os.path.isfile(candidate):
                return candidate
        return ""

    def _switch_advanced_mode(self) -> None:
        if self._batch_proc is not None and self._batch_proc.poll() is None:
            return

        script = str(gui_script("gui_advanced_tracking.py"))
        cmd = [sys.executable, "-u", script]
        cmd += ["--without-direction-estimation", str(int(self._without_direction.get()))]
        cfg_path = self._existing_config_path()
        if cfg_path:
            if not self._load_config_path(cfg_path):
                return
            cmd += ["--load-config", cfg_path]
        else:
            # Advanced Tracking can start without a config. Carry over only the
            # paths that are already available in Easy Tracking; all other
            # settings remain Advanced Tracking defaults.
            video = self._video_entry.get().strip()
            session = self._get_session()
            if session:
                cmd += ["--session", session]
            if video:
                cmd += ["--training-video", video, "--tracking-video", video]

        subprocess.Popen(cmd)
        self.after(100, self.destroy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config", dest="load_config", default="")
    parser.add_argument("--without-direction-estimation", type=int, choices=(0, 1), default=None)
    parser.add_argument("--video", default="")
    parser.add_argument("--session", default="")
    args = parser.parse_args()
    configure_taskbar_identity()
    app = EasyTrackingGUI()
    if args.video or args.session:
        app.after(200, lambda: app._apply_startup_paths(args.video, args.session))
    if args.load_config and os.path.isfile(args.load_config):
        app.after(400, lambda p=args.load_config: app._load_config_path(p))
    if args.without_direction_estimation is not None:
        app.after(500, lambda: app._without_direction.set(bool(args.without_direction_estimation)))
    app.mainloop()


if __name__ == "__main__":
    main()
