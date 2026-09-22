# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""The one place a GUI turns a chosen file into a video AMADEUS can analyse.

Every screen that lets the user pick a video -- Cropping & Trimming,
Segmentation, Easy Tracking, Advanced Tracking -- calls
:func:`prepare_analysis_video`. It inspects the file with
:mod:`main.video_compat`, returns it untouched when AMADEUS can already read it,
and otherwise offers a conversion in a dialog. Nothing is ever converted without
the user saying so, and the source file is only ever read.
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable

import customtkinter as ctk

try:
    from .project_paths import PROJECT_ROOT, ensure_import_paths
except ImportError:  # Direct execution of a GUI script.
    from project_paths import PROJECT_ROOT, ensure_import_paths

ensure_import_paths(PROJECT_ROOT)
from main.video_compat import (  # noqa: E402
    STATUS_OK,
    STATUS_RECOMMENDED,
    VIDEO_DROP_SUFFIXES,
    VIDEO_FILETYPES,
    CONVERTED_DIR_NAME,
    ConversionCancelled,
    ConversionResult,
    FfmpegUnavailableError,
    VideoAssessment,
    assess_video,
    build_plan,
    config_conversion_record,
    conversion_directory_for,
    ffmpeg_executable,
    ffmpeg_is_available,
    find_existing_conversions,
    metadata_path_for,
    read_conversion_metadata,
    run_conversion,
    validate_frame_range,
)

__all__ = [
    "VIDEO_DROP_SUFFIXES",
    "VIDEO_FILETYPES",
    "PreparedVideo",
    "ask_open_analysis_video",
    "prepare_analysis_video",
    "video_conversion_record",
]

MUTED_TEXT = "#9aa4ad"
WARNING_TEXT = "#f0a000"
ERROR_TEXT = "#ff5f56"


class PreparedVideo:
    """The video a GUI should load, and how it came about."""

    def __init__(
        self,
        path: str,
        *,
        source_path: str,
        converted: bool,
        assessment: VideoAssessment | None = None,
        metadata: dict | None = None,
        metadata_path: str = "",
    ):
        self.path = path
        self.source_path = source_path
        self.converted = converted
        self.assessment = assessment
        self.metadata = metadata
        self.metadata_path = metadata_path

    def config_record(self) -> dict:
        """The conversion summary a GUI stores in its own config file."""
        if not self.converted:
            return {}
        return config_conversion_record(self.metadata, self.metadata_path)

    def __str__(self) -> str:  # keeps "print(prepared)" useful in logs
        return self.path


# --------------------------------------------------------------------------
# Running slow work without freezing the GUI
# --------------------------------------------------------------------------

class _BusyDialog(ctk.CTkToplevel):
    """Modal 'working on it' window with an indeterminate bar."""

    def __init__(self, parent, title: str, message: str):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        frame = ctk.CTkFrame(self, corner_radius=0)
        frame.pack(fill="both", expand=True, padx=16, pady=16)
        ctk.CTkLabel(frame, text=message, anchor="w", justify="left").pack(
            fill="x", padx=8, pady=(4, 10)
        )
        self.bar = ctk.CTkProgressBar(frame, width=360, mode="indeterminate")
        self.bar.pack(fill="x", padx=8, pady=(0, 6))
        self.bar.start()
        _centre_on(self, parent)
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def finish(self) -> None:
        try:
            self.bar.stop()
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()


def _centre_on(window: tk.Misc, parent) -> None:
    window.update_idletasks()
    try:
        parent_x, parent_y = parent.winfo_rootx(), parent.winfo_rooty()
        parent_w, parent_h = parent.winfo_width(), parent.winfo_height()
    except (AttributeError, tk.TclError):
        return
    width, height = window.winfo_reqwidth(), window.winfo_reqheight()
    x = parent_x + max(0, (parent_w - width) // 2)
    y = parent_y + max(0, (parent_h - height) // 3)
    window.geometry(f"+{int(x)}+{int(y)}")


def _run_with_busy_dialog(parent, title: str, message: str, work: Callable[[], object]):
    """Run ``work`` on a worker thread while the GUI keeps redrawing."""
    outcome: dict = {}

    def runner() -> None:
        try:
            outcome["value"] = work()
        except BaseException as exc:  # re-raised on the GUI thread below
            outcome["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    dialog = _BusyDialog(parent, title, message)
    thread.start()

    def poll() -> None:
        if thread.is_alive():
            dialog.after(80, poll)
        else:
            dialog.finish()

    dialog.after(80, poll)
    dialog.wait_window()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


# --------------------------------------------------------------------------
# The conversion dialog
# --------------------------------------------------------------------------

class VideoConversionDialog(ctk.CTkToplevel):
    """Ask whether to convert one video, and run the conversion if so."""

    def __init__(self, parent, assessment: VideoAssessment, *, allow_source: bool):
        super().__init__(parent)
        self.assessment = assessment
        self.allow_source = allow_source
        self.result: PreparedVideo | None = None

        self._total_frames = max(0, assessment.usable_frame_count)
        self._last_frame_index = max(0, self._total_frames - 1)
        self._existing = find_existing_conversions(assessment.path)
        self._cancel_event = threading.Event()
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False

        self.title("Video conversion")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        _centre_on(self, parent)
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            pass
        self.start_entry.focus_set()

    # -- layout ------------------------------------------------------------

    def _build(self) -> None:
        outer = ctk.CTkFrame(self, corner_radius=0)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        headline = (
            "This video must be converted before AMADEUS can analyse it."
            if self.assessment.status != STATUS_RECOMMENDED
            else "Converting this video is recommended before analysis."
        )
        ctk.CTkLabel(
            outer, text=headline, anchor="w", justify="left",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(fill="x", padx=6, pady=(2, 10))

        body = ctk.CTkFrame(outer, corner_radius=4)
        body.pack(fill="both", expand=True, padx=6)
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(body, text="Source video", anchor="nw", justify="left",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="nw", padx=(10, 8), pady=(10, 2))
        info = tk.Text(
            body, width=48, height=len(self.assessment.summary_lines()),
            relief="flat", highlightthickness=0, wrap="none",
            bg="#2b2b2b", fg="#dce4ee", font=("TkFixedFont", 10),
        )
        info.insert("1.0", "\n".join(self.assessment.summary_lines()))
        info.configure(state="disabled")
        info.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 2))

        reasons = list(self.assessment.reasons) or [
            "AMADEUS could not confirm that this video is safe to read directly."
        ]
        ctk.CTkLabel(body, text="Why", anchor="nw", justify="left",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=1, column=0, sticky="nw", padx=(10, 8), pady=(10, 2))
        ctk.CTkLabel(
            body, text="\n".join(f"• {reason}" for reason in reasons),
            anchor="w", justify="left", wraplength=470,
        ).grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(10, 2))

        if self.assessment.notes:
            ctk.CTkLabel(
                body, text="\n".join(f"• {note}" for note in self.assessment.notes),
                anchor="w", justify="left", wraplength=470, text_color=MUTED_TEXT,
            ).grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(0, 2))

        ctk.CTkLabel(body, text="Result", anchor="nw", justify="left",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=3, column=0, sticky="nw", padx=(10, 8), pady=(10, 2))
        ctk.CTkLabel(
            body, text=self._plan_description(), anchor="w", justify="left",
            wraplength=470, text_color=MUTED_TEXT,
        ).grid(row=3, column=1, sticky="ew", padx=(0, 10), pady=(10, 2))

        # -- frame range --
        ctk.CTkLabel(body, text="Frame range", anchor="nw", justify="left",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=4, column=0, sticky="nw", padx=(10, 8), pady=(10, 2))
        range_row = ctk.CTkFrame(body, fg_color="transparent")
        range_row.grid(row=4, column=1, sticky="ew", padx=(0, 10), pady=(10, 2))

        self.start_var = tk.StringVar(value="0")
        self.end_var = tk.StringVar(value=str(self._last_frame_index))
        ctk.CTkLabel(range_row, text="start").pack(side="left")
        self.start_entry = ctk.CTkEntry(range_row, width=80, textvariable=self.start_var)
        self.start_entry.pack(side="left", padx=(4, 12))
        ctk.CTkLabel(range_row, text="end").pack(side="left")
        self.end_entry = ctk.CTkEntry(range_row, width=80, textvariable=self.end_var)
        self.end_entry.pack(side="left", padx=(4, 12))
        self.reset_button = ctk.CTkButton(
            range_row, text="Whole video", width=110, command=self._reset_range)
        self.reset_button.pack(side="left")

        self.range_hint = ctk.CTkLabel(
            body, text="", anchor="w", justify="left", wraplength=470, text_color=MUTED_TEXT)
        self.range_hint.grid(row=5, column=1, sticky="ew", padx=(0, 10), pady=(0, 6))
        self.start_var.trace_add("write", lambda *_: self._validate_range())
        self.end_var.trace_add("write", lambda *_: self._validate_range())

        # -- an analysis copy that already exists --
        self.existing_var = tk.StringVar()
        if self._existing:
            ctk.CTkLabel(body, text="Existing copy", anchor="nw", justify="left",
                         font=ctk.CTkFont(weight="bold")).grid(
                row=6, column=0, sticky="nw", padx=(10, 8), pady=(6, 2))
            existing_row = ctk.CTkFrame(body, fg_color="transparent")
            existing_row.grid(row=6, column=1, sticky="ew", padx=(0, 10), pady=(6, 2))
            labels = [self._existing_label(item) for item in self._existing]
            self.existing_var.set(labels[0])
            ctk.CTkOptionMenu(
                existing_row, values=labels, variable=self.existing_var, width=300,
            ).pack(side="left")
            ctk.CTkButton(
                existing_row, text="Use it", width=90, command=self._use_existing,
            ).pack(side="left", padx=(8, 0))

        # -- progress --
        self.progress = ctk.CTkProgressBar(outer, width=560)
        self.progress.set(0.0)
        self.progress.pack(fill="x", padx=6, pady=(12, 2))
        self.status_var = tk.StringVar(value="")
        ctk.CTkLabel(
            outer, textvariable=self.status_var, anchor="w", justify="left",
            text_color=MUTED_TEXT, wraplength=560,
        ).pack(fill="x", padx=6, pady=(0, 8))

        # -- buttons --
        buttons = ctk.CTkFrame(outer, fg_color="transparent")
        buttons.pack(fill="x", padx=6, pady=(2, 2))
        self.convert_button = ctk.CTkButton(
            buttons, text="Convert", width=140, command=self._start_conversion)
        self.convert_button.pack(side="right")
        self.source_button = ctk.CTkButton(
            buttons, text="Use original anyway", width=170, fg_color="transparent",
            border_width=1, command=self._use_source)
        if self.allow_source:
            self.source_button.pack(side="right", padx=(0, 8))
        self.cancel_button = ctk.CTkButton(
            buttons, text="Cancel", width=110, fg_color="transparent", border_width=1,
            command=self._on_close)
        self.cancel_button.pack(side="right", padx=(0, 8))

        self._validate_range()

    def _plan_description(self) -> str:
        directory = conversion_directory_for(self.assessment.path)
        parts = [
            "An analysis copy is written to "
            f"{os.path.join(os.path.basename(os.path.dirname(directory)), CONVERTED_DIR_NAME)}"
            ", beside the video. The original file is never changed.",
            "The copy keeps the original resolution and is re-encoded as H.264 MP4, "
            "8-bit SDR, constant frame rate, visually lossless (CRF 12), with short "
            "keyframe intervals so frame seeking stays exact.",
        ]
        if self.assessment.is_hdr:
            parts.append("HDR is tone mapped to SDR; the conversion is recorded in the metadata file.")
        if self.assessment.is_variable_frame_rate:
            parts.append(
                "The variable frame rate is resampled to a constant rate, so converted "
                "frame numbers follow time rather than the source frame order."
            )
        return "\n".join(parts)

    @staticmethod
    def _existing_label(metadata: dict) -> str:
        converted = metadata.get("converted_video", {})
        frame_range = metadata.get("frame_range", {})
        name = os.path.basename(str(converted.get("path", "")))
        first = frame_range.get("source_first_frame", 0)
        last = frame_range.get("source_last_frame", 0)
        return f"{name}  (source frames {first}-{last})"

    # -- range handling ----------------------------------------------------

    def _reset_range(self) -> None:
        self.start_var.set("0")
        self.end_var.set(str(self._last_frame_index))

    def _parsed_range(self) -> tuple[int, int] | None:
        try:
            return int(self.start_var.get().strip()), int(self.end_var.get().strip())
        except ValueError:
            return None

    def _validate_range(self) -> bool:
        if self._running:
            return False
        parsed = self._parsed_range()
        if parsed is None:
            problem = "Enter whole numbers for the start and end frame."
        else:
            problem = validate_frame_range(parsed[0], parsed[1], self._total_frames)
        if problem:
            self.range_hint.configure(text=problem, text_color=ERROR_TEXT)
            self.convert_button.configure(state="disabled")
            return False
        first, last = parsed
        whole = " (whole video)" if first == 0 and last == self._last_frame_index else ""
        self.range_hint.configure(
            text=f"{last - first + 1} frames of {self._total_frames}"
                 f" — usable range is 0 to {self._last_frame_index}{whole}",
            text_color=MUTED_TEXT,
        )
        self.convert_button.configure(state="normal")
        return True

    # -- outcomes ----------------------------------------------------------

    def _use_source(self) -> None:
        self.result = PreparedVideo(
            self.assessment.path,
            source_path=self.assessment.path,
            converted=False,
            assessment=self.assessment,
        )
        self._close()

    def _use_existing(self) -> None:
        label = self.existing_var.get()
        for metadata in self._existing:
            if self._existing_label(metadata) == label:
                converted = str(metadata.get("converted_video", {}).get("path", ""))
                if not os.path.isfile(converted):
                    messagebox.showerror(
                        "Video conversion",
                        f"The analysis copy is no longer there:\n{converted}",
                        parent=self,
                    )
                    return
                self.result = PreparedVideo(
                    converted,
                    source_path=self.assessment.path,
                    converted=True,
                    assessment=self.assessment,
                    metadata=metadata,
                    metadata_path=metadata_path_for(converted),
                )
                self._close()
                return

    def _on_close(self) -> None:
        if self._running:
            self._cancel_event.set()
            self.status_var.set("Stopping the conversion...")
            return
        self.result = None
        self._close()

    def _close(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()

    # -- conversion --------------------------------------------------------

    def _set_inputs_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (self.start_entry, self.end_entry, self.reset_button,
                       self.convert_button, self.source_button):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _start_conversion(self) -> None:
        if self._running or not self._validate_range():
            return
        parsed = self._parsed_range()
        assert parsed is not None
        try:
            plan = build_plan(self.assessment, parsed[0], parsed[1])
        except (RuntimeError, ValueError) as exc:
            messagebox.showerror("Video conversion", str(exc), parent=self)
            return
        if os.path.isfile(plan.output_path) and not messagebox.askyesno(
            "Video conversion",
            f"An analysis copy already exists and will be replaced:\n{plan.output_path}\n\n"
            "Continue?",
            parent=self,
        ):
            return

        self._running = True
        self._cancel_event.clear()
        self._set_inputs_enabled(False)
        self.cancel_button.configure(text="Stop")
        self.progress.set(0.0)
        self.status_var.set(
            f"Converting frames {plan.first_frame}-{plan.last_frame} "
            f"({plan.expected_output_frames} frames)..."
        )

        def work() -> None:
            try:
                result = run_conversion(
                    plan,
                    progress=lambda fraction, frames: self._queue.put(("progress", fraction, frames)),
                    cancel=self._cancel_event,
                )
                self._queue.put(("done", result))
            except ConversionCancelled:
                self._queue.put(("cancelled",))
            except BaseException as exc:
                self._queue.put(("error", exc))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self.after(60, self._drain_queue)

    def _drain_queue(self) -> None:
        try:
            while True:
                message = self._queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    _, fraction, frames = message
                    self.progress.set(float(fraction))
                    self.status_var.set(f"Converting... {frames} frames written")
                elif kind == "done":
                    self._conversion_finished(message[1])
                    return
                elif kind == "cancelled":
                    self._running = False
                    self._set_inputs_enabled(True)
                    self.cancel_button.configure(text="Cancel")
                    self.progress.set(0.0)
                    self.status_var.set("The conversion was stopped. Nothing was written.")
                    self._validate_range()
                    return
                elif kind == "error":
                    self._running = False
                    self._set_inputs_enabled(True)
                    self.cancel_button.configure(text="Cancel")
                    self.progress.set(0.0)
                    self.status_var.set("The conversion failed.")
                    self._validate_range()
                    messagebox.showerror("Video conversion", str(message[1]), parent=self)
                    return
        except queue.Empty:
            pass
        self.after(60, self._drain_queue)

    def _conversion_finished(self, result: ConversionResult) -> None:
        self._running = False
        self.progress.set(1.0)
        metadata = read_conversion_metadata(result.output_path)
        self.result = PreparedVideo(
            result.output_path,
            source_path=self.assessment.path,
            converted=True,
            assessment=self.assessment,
            metadata=metadata,
            metadata_path=result.metadata_path,
        )
        if result.warnings:
            messagebox.showwarning(
                "Video conversion",
                "The analysis video was created, with these remarks:\n\n"
                + "\n".join(f"• {warning}" for warning in result.warnings),
                parent=self,
            )
        self._close()


# --------------------------------------------------------------------------
# Entry points used by the GUIs
# --------------------------------------------------------------------------

def prepare_analysis_video(
    parent,
    path: str,
    *,
    log: Callable[[str], None] | None = None,
) -> PreparedVideo | None:
    """Return the video a GUI should load for ``path``, or None if abandoned.

    A file AMADEUS already reads reliably comes back untouched and without a
    dialog, so the existing MP4/MOV/AVI workflow is unchanged.
    """
    path = str(Path(str(path)).expanduser())
    if not os.path.isfile(path):
        messagebox.showerror("Error", f"Video file not found:\n{path}", parent=parent)
        return None

    def note(message: str) -> None:
        if log is not None:
            log(message)

    try:
        assessment = _run_with_busy_dialog(
            parent,
            "Checking the video",
            f"Checking whether AMADEUS can read\n{os.path.basename(path)}\ndirectly...",
            lambda: assess_video(path),
        )
    except Exception as exc:
        messagebox.showerror(
            "Error", f"The video could not be inspected:\n\n{exc}", parent=parent
        )
        return None

    if assessment is None:
        return None

    if assessment.status == STATUS_OK:
        for message in assessment.notes:
            note(message)
        return PreparedVideo(path, source_path=path, converted=False, assessment=assessment)

    for message in assessment.reasons:
        note(f"Video compatibility: {message}")

    if not ffmpeg_is_available():
        return _handle_missing_ffmpeg(parent, assessment, note)

    dialog = VideoConversionDialog(
        parent, assessment, allow_source=assessment.status == STATUS_RECOMMENDED
    )
    parent.wait_window(dialog)
    result = dialog.result
    if result is None:
        note("Video selection cancelled; no conversion was made.")
        return None
    if result.converted:
        note(f"Using the converted analysis video: {result.path}")
        note(f"Conversion record: {result.metadata_path}")
    else:
        note("Using the original video without conversion, as requested.")
    return result


def _handle_missing_ffmpeg(
    parent, assessment: VideoAssessment, note: Callable[[str], None]
) -> PreparedVideo | None:
    """Explain why nothing can be converted, and offer what is still possible."""
    try:
        ffmpeg_executable()
        detail = ""
    except FfmpegUnavailableError as exc:
        detail = str(exc)

    reasons = "\n".join(f"• {reason}" for reason in assessment.reasons)
    note("FFmpeg is unavailable, so no conversion could be offered.")

    if assessment.status == STATUS_RECOMMENDED:
        answer = messagebox.askyesno(
            "FFmpeg is unavailable",
            f"This video would be better converted first:\n\n{reasons}\n\n"
            f"{detail}\n\nUse the original video anyway?",
            parent=parent,
        )
        if answer:
            return PreparedVideo(
                assessment.path, source_path=assessment.path,
                converted=False, assessment=assessment,
            )
        return None

    messagebox.showerror(
        "FFmpeg is unavailable",
        f"This video cannot be used as it is:\n\n{reasons}\n\n{detail}",
        parent=parent,
    )
    return None


def ask_open_analysis_video(
    parent,
    *,
    title: str = "Select video",
    initialdir: str | None = None,
    log: Callable[[str], None] | None = None,
) -> PreparedVideo | None:
    """Show the video file dialog and prepare whatever the user picks."""
    path = filedialog.askopenfilename(
        parent=parent,
        title=title,
        initialdir=initialdir or None,
        filetypes=VIDEO_FILETYPES,
    )
    if not path:
        return None
    return prepare_analysis_video(parent, path, log=log)


def video_conversion_record(video_path: str) -> dict:
    """Return the stored conversion summary for a video, or an empty dict.

    Lets a GUI re-attach the conversion record to its config after reloading a
    session whose video was converted in an earlier run.
    """
    metadata = read_conversion_metadata(video_path)
    if not metadata:
        return {}
    return config_conversion_record(metadata, metadata_path_for(video_path))
