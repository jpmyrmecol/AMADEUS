# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox

import customtkinter as ctk
import yaml

try:
    from .project_paths import (
        GUI_DIR,
        PROJECT_ROOT,
        MAIN_DIR as MAIN_PATH,
        ensure_import_paths,
        main_script,
        with_pythonpath,
    )
    from .window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
except ImportError:  # Preserve direct execution with: python gui/gui_multi_videoset_batch.py
    from project_paths import (
        GUI_DIR,
        PROJECT_ROOT,
        MAIN_DIR as MAIN_PATH,
        ensure_import_paths,
        main_script,
        with_pythonpath,
    )
    from window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon

ensure_import_paths(PROJECT_ROOT, MAIN_PATH)
from path_utils import resolve_config_paths

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

CURRENT_DIR = str(GUI_DIR)
MAIN_DIR = str(MAIN_PATH)
BATCH_SCRIPT = str(main_script("batch.py"))
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PERF_BOTTLENECK_RE = re.compile(r"\[PERF\].*?bottleneck=([^\r\n]+)")


@dataclass
class ConfigItem:
    path: str
    session_path: str
    tracking_video_path: str


class MultiVideosetBatchGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        configure_dpi_scaling(self)
        install_window_icon(self)
        self.title("AMADEUS: Multi Videoset Batch")
        self.geometry("1100x720")
        self.minsize(900, 560)

        self.items: list[ConfigItem] = []
        self.proc: subprocess.Popen | None = None
        self._batch_log_path = ""
        self.worker: threading.Thread | None = None
        self.stop_requested = False
        self._batch_log_path = ""
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._perf_parse_buffer = ""
        self._pending_cr = False
        self._log_lines: list[str] = []
        self._active_log_line = ""
        self._log_dirty = False
        self._log_render_scheduled = False
        self._max_log_lines = 5000

        self._build_ui()
        self.after(100, self._poll_events)

    def _build_ui(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        root = ctk.CTkFrame(self, corner_radius=0)
        root.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=2)
        root.grid_columnconfigure(1, weight=3)

        header = ctk.CTkFrame(root, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(
            header,
            text="Run Saved Tracking Configs",
            font=("TkDefaultFont", 16, "bold"),
            anchor="w",
        ).pack(side="left")
        self.status_label = ctk.CTkLabel(header, text="Idle", anchor="e")
        self.status_label.pack(side="right")
        self.perf_label = ctk.CTkLabel(header, text="Bottleneck: n/a", anchor="e")
        self.perf_label.pack(side="right", padx=(0, 18))

        left = ctk.CTkFrame(root, corner_radius=6)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Config Queue", font=("TkDefaultFont", 13, "bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", padx=8, pady=(8, 4)
        )
        list_wrap = ctk.CTkFrame(left, corner_radius=0)
        list_wrap.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        list_wrap.grid_rowconfigure(0, weight=1)
        list_wrap.grid_columnconfigure(0, weight=1)
        self.listbox = tk.Listbox(
            list_wrap,
            selectmode=tk.EXTENDED,
            bg="#202020",
            fg="#f0f0f0",
            selectbackground="#2f7d32",
            highlightthickness=0,
            activestyle="none",
            relief="flat",
        )
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ctk.CTkScrollbar(list_wrap, command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scroll.set)

        btns = ctk.CTkFrame(left, corner_radius=0)
        btns.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))
        ctk.CTkButton(btns, text="Add Configs", command=self._add_configs).pack(side="left", padx=(0, 5), pady=4)
        ctk.CTkButton(btns, text="Add Folder", command=self._add_folder).pack(side="left", padx=5, pady=4)
        ctk.CTkButton(btns, text="Remove", command=self._remove_selected).pack(side="left", padx=5, pady=4)
        ctk.CTkButton(btns, text="Up", width=55, command=lambda: self._move_selected(-1)).pack(side="left", padx=5, pady=4)
        ctk.CTkButton(btns, text="Down", width=65, command=lambda: self._move_selected(1)).pack(side="left", padx=5, pady=4)
        ctk.CTkButton(btns, text="Clear", width=65, command=self._clear_items).pack(side="left", padx=(5, 0), pady=4)

        right = ctk.CTkFrame(root, corner_radius=6)
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Batch Log", font=("TkDefaultFont", 13, "bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", padx=8, pady=(8, 4)
        )
        self.log_box = ctk.CTkTextbox(right, wrap="word")
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        bottom = ctk.CTkFrame(root, corner_radius=0)
        bottom.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bottom.grid_columnconfigure(1, weight=1)
        self.run_btn = ctk.CTkButton(bottom, text="Run Queue", command=self._run_queue)
        self.run_btn.grid(row=0, column=0, padx=(0, 8), pady=4)
        self.progress = ctk.CTkProgressBar(bottom, mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        self.progress.set(0.0)
        self.stop_btn = ctk.CTkButton(bottom, text="Stop", state="disabled", command=self._stop)
        self.stop_btn.grid(row=0, column=2, padx=(8, 0), pady=4)

    def _load_config_item(self, path: str) -> ConfigItem:
        path = os.path.abspath(path)
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg = resolve_config_paths(cfg)
        return ConfigItem(
            path=path,
            session_path=str(cfg.get("SESSION_PATH", "")),
            tracking_video_path=str(cfg.get("TRACKING_VIDEO_PATH", "")),
        )

    def _add_paths(self, paths: list[str]) -> None:
        known = {item.path for item in self.items}
        added = 0
        errors: list[str] = []
        for path in paths:
            if not path:
                continue
            path = os.path.abspath(path)
            if path in known:
                continue
            try:
                item = self._load_config_item(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
            self.items.append(item)
            known.add(path)
            added += 1
        self._refresh_list()
        if errors:
            messagebox.showerror("Config Load Error", "\n".join(errors[:8]))
        if added:
            self._append_log(f"[queue] added {added} config(s)\n")

    def _add_configs(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select config YAML files",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            parent=self,
        )
        self._add_paths(list(paths))

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder containing config.yaml files", parent=self)
        if not folder:
            return
        paths: list[str] = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower() in {"config.yaml", "config.yml"}:
                    paths.append(os.path.join(root, name))
        paths.sort()
        self._add_paths(paths)

    def _selected_indices(self) -> list[int]:
        return sorted(int(i) for i in self.listbox.curselection())

    def _remove_selected(self) -> None:
        for idx in reversed(self._selected_indices()):
            del self.items[idx]
        self._refresh_list()

    def _move_selected(self, delta: int) -> None:
        indices = self._selected_indices()
        if not indices:
            return
        if delta < 0:
            iterable = indices
        else:
            iterable = reversed(indices)
        selected_after: list[int] = []
        for idx in iterable:
            new_idx = idx + delta
            if new_idx < 0 or new_idx >= len(self.items):
                selected_after.append(idx)
                continue
            self.items[idx], self.items[new_idx] = self.items[new_idx], self.items[idx]
            selected_after.append(new_idx)
        self._refresh_list()
        for idx in selected_after:
            self.listbox.selection_set(idx)

    def _clear_items(self) -> None:
        self.items.clear()
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.listbox.delete(0, tk.END)
        for i, item in enumerate(self.items, start=1):
            session = item.session_path or "(no session)"
            video = item.tracking_video_path or "(no tracking video)"
            self.listbox.insert(tk.END, f"{i:02d}. {session} | {video}")

    def _maybe_update_perf_status(self, text: str) -> None:
        self._perf_parse_buffer = (self._perf_parse_buffer + text)[-4000:]
        matches = list(PERF_BOTTLENECK_RE.finditer(self._perf_parse_buffer))
        if matches:
            bottleneck = matches[-1].group(1).strip()
            self.perf_label.configure(text=f"Bottleneck: {bottleneck}")

    def _append_log(self, text: str) -> None:
        """Append subprocess output with terminal-like CR handling.

        tqdm/Ultralytics redraw progress bars with carriage returns, ANSI
        cursor-control sequences, and space padding.  A Tk Text widget is not a
        terminal, so raw insertion can create many invisible blank lines.  This
        method keeps completed log lines and one live line separately, then
        rewrites only that live line when a carriage return arrives.
        """
        if not text:
            return

        text = ANSI_ESCAPE_RE.sub("", str(text))
        self._maybe_update_perf_status(text)

        for ch in text:
            if self._pending_cr:
                if ch == "\r":
                    # Multiple CR characters are equivalent to a single cursor
                    # return.  Do not clear repeatedly, otherwise the line above
                    # can be affected in Tk's text model.
                    continue
                if ch == "\n":
                    self._pending_cr = False
                    self._finalize_log_line()
                    continue

                # Lone CR: redraw the current terminal line.
                self._active_log_line = ""
                self._pending_cr = False

            if ch == "\r":
                self._pending_cr = True
                continue

            if ch == "\n":
                self._finalize_log_line()
                continue

            if ch == "\b":
                self._active_log_line = self._active_log_line[:-1]
                self._mark_log_dirty()
                continue

            if ch == "\t" or ch >= " ":
                self._active_log_line += ch
                self._mark_log_dirty()

    def _finalize_log_line(self) -> None:
        line = self._active_log_line.rstrip()
        self._active_log_line = ""

        # A long run of blank lines is nearly always an artifact of progress-bar
        # redrawing after ANSI cursor movement has been stripped.  Preserve a
        # single separator, but do not let blanks accumulate.
        if line == "" and self._log_lines and self._log_lines[-1] == "":
            self._mark_log_dirty()
            return

        self._log_lines.append(line)
        if len(self._log_lines) > self._max_log_lines:
            marker = "[log trimmed: older lines removed from GUI view]"
            keep = max(0, self._max_log_lines - 1)
            self._log_lines = [marker] + self._log_lines[-keep:]
        self._mark_log_dirty()

    def _mark_log_dirty(self) -> None:
        self._log_dirty = True
        if not self._log_render_scheduled:
            self._log_render_scheduled = True
            self.after_idle(self._render_log)

    def _log_pinned_to_bottom(self) -> bool:
        try:
            _first, last = self.log_box.yview()
        except Exception:
            return True
        return last >= 0.999

    def _render_log(self) -> None:
        self._log_render_scheduled = False
        if not self._log_dirty:
            return

        if not self._log_pinned_to_bottom():
            # The user scrolled up to read or copy earlier output. Rebuilding
            # the widget now would reset that scroll position and clear any
            # selection, so leave it untouched -- keep polling at a low rate
            # until they either scroll back to the bottom or new output
            # arrives while they're there, at which point normal live
            # updates resume automatically.
            self._log_render_scheduled = True
            self.after(200, self._render_log)
            return

        self._log_dirty = False

        lines = list(self._log_lines)
        if self._active_log_line:
            lines.append(self._active_log_line.rstrip())
        content = "\n".join(lines)

        self.log_box.delete("1.0", "end")
        if content:
            self.log_box.insert("1.0", content)
        self.log_box.see("end")

    def _set_running(self, running: bool) -> None:
        self.run_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")

    def _run_queue(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        if not self.items:
            messagebox.showerror("Error", "Add at least one config YAML.")
            return
        if not os.path.isfile(BATCH_SCRIPT):
            messagebox.showerror("Error", f"batch.py was not found:\n{BATCH_SCRIPT}")
            return
        self.stop_requested = False
        self.progress.set(0.0)
        self.status_label.configure(text="Running")
        self.perf_label.configure(text="Bottleneck: n/a")
        self._set_running(True)
        self.worker = threading.Thread(target=self._worker_run, args=(list(self.items),), daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_requested = True
        self.status_label.configure(text="Stopping")
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()

    def _worker_run(self, items: list[ConfigItem]) -> None:
        total = len(items)
        try:
            for idx, item in enumerate(items, start=1):
                if self.stop_requested:
                    self.events.put(("stopped", None))
                    return
                self.events.put(("status", f"{idx}/{total}: {item.session_path or os.path.basename(item.path)}"))
                self.events.put(("log", f"\n=== START {idx}/{total}: {item.path} ===\n"))
                rc = self._run_one(item.path)
                if self.stop_requested:
                    self.events.put(("log", f"=== STOPPED {idx}/{total}: {item.path} ===\n"))
                    self.events.put(("stopped", None))
                    return
                if rc != 0:
                    log_path = self._batch_log_path or "unavailable"
                    self.events.put(("log", f"=== FAILED {idx}/{total}: exit code {rc} ===\n"))
                    self.events.put(
                        ("failed", f"Processing failed: exit code {rc}; full log: {log_path}")
                    )
                    return
                self.events.put(("progress", idx / total))
                self.events.put(("log", f"=== DONE {idx}/{total}: {item.path} ===\n"))
            self.events.put(("done", None))
        finally:
            self.proc = None

    def _run_one(self, config_path: str) -> int:
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONFAULTHANDLER"] = "1"
        with_pythonpath(env, PROJECT_ROOT, MAIN_PATH)
        # The GUI renders carriage-return updates in place.  This preserves
        # its live progress view while batch.py keeps log.txt compact.
        env["AMADEUS_GUI_PROGRESS_PROTOCOL"] = "1"
        popen_kwargs = {
            "cwd": MAIN_DIR,
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            # Binary mode is intentional. Text mode performs universal-newline
            # conversion, which turns tqdm carriage returns (\r) into new lines
            # and floods the GUI log.
            "bufsize": 0,
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        self.proc = subprocess.Popen([sys.executable, "-u", BATCH_SCRIPT, config_path], **popen_kwargs)
        assert self.proc.stdout is not None
        decoder = self._make_utf8_decoder()
        log_parse_buffer = ""
        while True:
            chunk = self.proc.stdout.read(256)
            if not chunk:
                break
            text = decoder.decode(chunk, final=False)
            log_parse_buffer += text
            lines = log_parse_buffer.splitlines(keepends=True)
            log_parse_buffer = "" if not lines or lines[-1].endswith(("\n", "\r")) else lines.pop()
            for line in lines:
                if line.startswith("Saving CLI log to:"):
                    self._batch_log_path = line.split(":", 1)[1].strip()
            self.events.put(("log", text))
        tail = decoder.decode(b"", final=True)
        if tail:
            log_parse_buffer += tail
        if log_parse_buffer:
            for line in log_parse_buffer.splitlines():
                if line.startswith("Saving CLI log to:"):
                    self._batch_log_path = line.split(":", 1)[1].strip()
        if tail:
            self.events.put(("log", tail))
        return int(self.proc.wait())

    @staticmethod
    def _make_utf8_decoder():
        import codecs

        return codecs.getincrementaldecoder("utf-8")("replace")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "progress":
                    self.progress.set(float(payload))
                elif kind == "status":
                    self.status_label.configure(text=str(payload))
                elif kind == "done":
                    self.status_label.configure(text="Done")
                    self.progress.set(1.0)
                    self._set_running(False)
                elif kind == "failed":
                    self.status_label.configure(text=str(payload))
                    self._set_running(False)
                elif kind == "stopped":
                    self.status_label.configure(text="Stopped")
                    self._set_running(False)
        except queue.Empty:
            pass
        self.after(100, self._poll_events)


def main() -> None:
    configure_taskbar_identity()
    app = MultiVideosetBatchGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
