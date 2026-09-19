# SPDX-License-Identifier: AGPL-3.0-only

"""Prepare an AMADEUS project for Google Colab."""

from __future__ import annotations

import argparse
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.window_icon import configure_dpi_scaling, install_window_icon
from main.colab_utils import (
    build_colab_browser_target,
    ColabPreparationState,
    ConfigSummary,
    BundlePlan,
    RuntimePackagePlan,
    execute_bundle,
    execute_runtime_package,
    load_colab_config,
    plan_input_bundle,
    plan_runtime_package,
    validate_colab_config,
)


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")


def _set_readonly(entry: ctk.CTkEntry, value: str) -> None:
    entry.configure(state="normal")
    entry.delete(0, tk.END)
    entry.insert(0, value)
    entry.configure(state="readonly")


class ColabSetupApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        configure_dpi_scaling(self)
        install_window_icon(self)
        self.title("AMADEUS Google Colab")
        self.geometry("900x760")
        self.minsize(820, 680)

        self._config_path = ""
        self._summary: ConfigSummary | None = None
        self._plan: BundlePlan | None = None
        self._runtime_plan: RuntimePackagePlan | None = None
        self._colab_config_path = ""
        self._prepared_notebook_path = ""
        self._runtime_manifest_path = ""
        self._preparation_state = ColabPreparationState()
        self._message_queue: list[tuple[str, object]] = []
        self._queue_lock = threading.Lock()

        self._build_widgets()

    def _build_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(self, text="AMADEUS on Google Colab", font=ctk.CTkFont(size=26, weight="bold")).grid(
            row=0, column=0, padx=24, pady=(22, 4), sticky="w"
        )
        ctk.CTkLabel(
            self,
            text="Prepare the Drive input and runtime bundle, then open its notebook on a GPU runtime.",
            text_color="#bdbdbd",
        ).grid(row=1, column=0, padx=24, pady=(0, 14), sticky="w")

        content = ctk.CTkScrollableFrame(self, label_text="Setup")
        content.grid(row=3, column=0, padx=18, pady=4, sticky="nsew")
        content.grid_columnconfigure(1, weight=1)

        row = 0
        ctk.CTkLabel(content, text="AMADEUS Config", font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=row, column=0, columnspan=3, padx=10, pady=(8, 8), sticky="w"
        )
        row += 1
        self._config_entry = ctk.CTkEntry(content, height=34)
        self._config_entry.grid(row=row, column=0, columnspan=2, padx=(10, 6), pady=4, sticky="ew")
        self._config_entry.bind("<KeyRelease>", self._on_inputs_changed)
        ctk.CTkButton(content, text="Browse", width=110, command=self._browse_config).grid(
            row=row, column=2, padx=(4, 10), pady=4
        )
        row += 1
        self._project_entry = self._readonly_row(content, row, "Project folder:")
        row += 1
        self._training_entry = self._readonly_row(content, row, "Training video:")
        row += 1
        self._tracking_entry = self._readonly_row(content, row, "Tracking video:")
        row += 1
        self._segmentation_entry = self._readonly_row(content, row, "Segmentation:")
        row += 1
        self._easy_entry = self._readonly_row(content, row, "Easy Tracking config:")
        row += 1

        ctk.CTkLabel(content, text="Google Drive Destination", font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=row, column=0, columnspan=3, padx=10, pady=(22, 8), sticky="w"
        )
        row += 1
        self._destination_entry = ctk.CTkEntry(content, height=34)
        self._destination_entry.grid(row=row, column=0, columnspan=2, padx=(10, 6), pady=4, sticky="ew")
        self._destination_entry.bind("<KeyRelease>", self._on_inputs_changed)
        ctk.CTkButton(content, text="Browse", width=110, command=self._browse_destination).grid(
            row=row, column=2, padx=(4, 10), pady=4
        )
        row += 1
        self._prepare_button = ctk.CTkButton(content, text="Prepare for Google Colab", state="disabled", command=self._prepare)
        self._prepare_button.grid(row=row, column=0, columnspan=3, padx=10, pady=(14, 8), sticky="ew")
        row += 1
        self._progress = ctk.CTkProgressBar(content)
        self._progress.set(0)
        self._progress.grid(row=row, column=0, columnspan=3, padx=10, pady=4, sticky="ew")
        row += 1
        self._status_label = ctk.CTkLabel(content, text="Select the config and a folder below My Drive.", anchor="w", justify="left")
        self._status_label.grid(row=row, column=0, columnspan=3, padx=10, pady=(4, 12), sticky="ew")
        row += 1

        ctk.CTkLabel(content, text="Completion", font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=row, column=0, columnspan=3, padx=10, pady=(12, 8), sticky="w"
        )
        row += 1
        self._drive_project_entry = self._readonly_row(content, row, "Google Drive project:")
        row += 1
        self._colab_path_entry = self._readonly_row(content, row, "Colab config path:")
        row += 1
        self._notebook_path_entry = self._readonly_row(content, row, "Prepared notebook:")
        row += 1
        self._manifest_path_entry = self._readonly_row(content, row, "Runtime manifest:")
        row += 1
        button_row = ctk.CTkFrame(content, fg_color="transparent")
        button_row.grid(row=row, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="ew")
        self._open_button = ctk.CTkButton(button_row, text="Open Google Colab", state="disabled", command=self._open_colab)
        self._open_button.grid(row=0, column=0, columnspan=3, sticky="ew")
        row += 1
        ctk.CTkLabel(
            content,
            text=(
                "1. Select the config and a folder below My Drive.\n"
                "2. Select Prepare for Google Colab and wait for success.\n"
                "3. Select Open Google Colab when you are ready.\n"
                "4. In Colab, select Run all manually and authorize Drive if prompted."
            ),
            text_color="#bdbdbd",
            justify="left",
            anchor="w",
        ).grid(row=row, column=0, columnspan=3, padx=10, pady=(6, 16), sticky="ew")
        self._update_action_states()

    @staticmethod
    def _readonly_row(parent: ctk.CTkFrame, row: int, label: str) -> ctk.CTkEntry:
        ctk.CTkLabel(parent, text=label, width=170, anchor="w").grid(row=row, column=0, padx=(10, 6), pady=4, sticky="w")
        entry = ctk.CTkEntry(parent, state="readonly", height=32)
        entry.grid(row=row, column=1, columnspan=2, padx=(4, 10), pady=4, sticky="ew")
        return entry

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select AMADEUS config.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self._load_config(path)

    def _load_config(self, path: str) -> None:
        try:
            _cfg, summary = load_colab_config(path)
        except Exception as exc:
            messagebox.showerror("Config Load Error", str(exc), parent=self)
            return
        self._config_path = summary.config_path
        self._summary = summary
        _set_readonly(self._project_entry, summary.session_path or "Missing")
        _set_readonly(self._training_entry, summary.training_video_path or "Missing")
        _set_readonly(self._tracking_entry, summary.tracking_video_path or "Missing")
        _set_readonly(self._segmentation_entry, "Ready" if summary.segmentation_ready else "Missing")
        _set_readonly(self._easy_entry, "Valid" if summary.easy_tracking_valid else "Invalid")
        self._config_entry.delete(0, tk.END)
        self._config_entry.insert(0, summary.config_path)
        self._sync_inputs()
        if summary.errors:
            self._set_status("Config loaded with issues:\n" + "\n".join(summary.errors))
        else:
            self._set_status("Config loaded. Choose a destination below My Drive.")
        self._update_action_states()

    def _browse_destination(self) -> None:
        path = filedialog.askdirectory(title="Select a Google Drive folder below My Drive")
        if path:
            self._destination_entry.delete(0, tk.END)
            self._destination_entry.insert(0, path)
            self._sync_inputs()

    def _set_status(self, text: str) -> None:
        self._status_label.configure(text=text)

    def _clear_preparation_outputs(self) -> None:
        self._plan = None
        self._runtime_plan = None
        self._colab_config_path = ""
        self._prepared_notebook_path = ""
        self._runtime_manifest_path = ""
        for entry in (
            self._drive_project_entry,
            self._colab_path_entry,
            self._notebook_path_entry,
            self._manifest_path_entry,
        ):
            _set_readonly(entry, "")
        self._progress.set(0)

    def _update_action_states(self) -> None:
        self._prepare_button.configure(
            state="normal" if self._preparation_state.prepare_enabled else "disabled"
        )
        self._open_button.configure(
            state="normal" if self._preparation_state.open_enabled else "disabled"
        )

    def _sync_inputs(self) -> None:
        config = self._config_entry.get().strip()
        destination = self._destination_entry.get().strip()
        self._config_path = config
        if self._preparation_state.set_inputs(config, destination):
            self._clear_preparation_outputs()
            if self._preparation_state.running:
                self._set_status("Inputs changed. The current preparation will be discarded when it finishes.")
            else:
                self._set_status("Inputs changed. Select Prepare for Google Colab to prepare the new destination.")
        self._update_action_states()

    def _on_inputs_changed(self, *_event: object) -> None:
        self._sync_inputs()

    def _prepare(self) -> None:
        self._sync_inputs()
        config = self._config_path
        destination = self._destination_entry.get().strip()
        if not config or not destination:
            messagebox.showerror("Prepare", "Select both the config file and a Google Drive destination.", parent=self)
            return
        token = self._preparation_state.begin()
        if token is None:
            self._update_action_states()
            return
        self._update_action_states()
        self._progress.set(0)
        self._set_status("Planning the clean input bundle...")
        threading.Thread(target=self._prepare_worker, args=(config, destination, token), daemon=True).start()
        self.after(100, self._poll_worker)

    def _prepare_worker(self, config: str, destination: str, token: int) -> None:
        try:
            validate_colab_config(config, check_video=True)
            plan = plan_input_bundle(config, destination)
            runtime_plan = plan_runtime_package(
                str(PROJECT_ROOT),
                plan.destination_session,
                colab_config_path=plan.colab_config_path,
            )
            with self._queue_lock:
                self._message_queue.append(("plan", (token, plan, runtime_plan)))

            input_total = plan.total_bytes
            combined_total = input_total + runtime_plan.total_bytes

            def progress(copied: int, total: int, reason: str) -> None:
                with self._queue_lock:
                    self._message_queue.append(("progress", (token, copied, combined_total, reason)))

            def runtime_progress(copied: int, total: int, reason: str) -> None:
                with self._queue_lock:
                    self._message_queue.append(
                        ("progress", (token, input_total + copied, combined_total, f"runtime package: {reason}"))
                    )

            execute_bundle(plan, progress_callback=progress)
            execute_runtime_package(runtime_plan, progress_callback=runtime_progress)
            with self._queue_lock:
                self._message_queue.append(("done", (token, plan, runtime_plan)))
        except Exception as exc:
            with self._queue_lock:
                self._message_queue.append(("error", (token, str(exc))))

    def _poll_worker(self) -> None:
        messages: list[tuple[str, object]] = []
        with self._queue_lock:
            messages, self._message_queue = self._message_queue, []
        waiting = True
        for kind, payload in messages:
            if kind == "plan":
                token, plan, runtime_plan = payload  # type: ignore[misc]
                if not self._preparation_state.is_current(token):
                    continue
                self._plan, self._runtime_plan = plan, runtime_plan
                total = self._plan.total_bytes + self._runtime_plan.total_bytes
                self._set_status(f"Copying input and runtime bundle ({total:,} bytes)...")
            elif kind == "progress":
                token, copied, total, reason = payload  # type: ignore[misc]
                if not self._preparation_state.is_current(token):
                    continue
                self._progress.set(copied / total if total else 1)
                self._set_status(f"Copying: {reason}\n{copied:,} / {total:,} bytes")
            elif kind == "done":
                waiting = False
                token, plan, runtime_plan = payload  # type: ignore[misc]
                current = self._preparation_state.finish(token, successful=True)
                if not current:
                    self._update_action_states()
                    continue
                self._plan = plan
                self._runtime_plan = runtime_plan
                self._colab_config_path = self._plan.colab_config_path
                self._prepared_notebook_path = self._runtime_plan.notebook_path
                self._runtime_manifest_path = self._runtime_plan.manifest_path
                _set_readonly(self._drive_project_entry, self._plan.destination_session)
                _set_readonly(self._colab_path_entry, self._colab_config_path)
                _set_readonly(self._notebook_path_entry, self._prepared_notebook_path)
                _set_readonly(self._manifest_path_entry, self._runtime_manifest_path)
                self._progress.set(1)
                self._set_status(
                    "Prepared successfully. Select Open Google Colab when you are ready. "
                    "The source config was not modified."
                )
                self._open_button.configure(state="normal")
                self._update_action_states()
            elif kind == "error":
                waiting = False
                token, error = payload  # type: ignore[misc]
                current = self._preparation_state.finish(token, successful=False)
                if not current:
                    self._update_action_states()
                    continue
                self._progress.set(0)
                self._set_status("Prepare failed.")
                self._update_action_states()
                messagebox.showerror("Prepare Error", str(error), parent=self)
        if waiting:
            self.after(150, self._poll_worker)

    def _open_colab(self) -> None:
        if (
            not self._preparation_state.open_enabled
            or not self._colab_config_path
            or not self._prepared_notebook_path
        ):
            return
        try:
            target = build_colab_browser_target(self._prepared_notebook_path)
        except ValueError as exc:
            messagebox.showerror("Open Google Colab", str(exc), parent=self)
            return
        opened = webbrowser.open(target, new=2)
        instructions = (
            f"Open this exact prepared notebook in Google Drive with Colab:\n{self._prepared_notebook_path}\n\n"
            "The project-specific CONFIG_PATH is already filled in. "
            "Authorize Google Drive when requested and select Run all."
        )
        if not opened:
            messagebox.showinfo(
                "Open Google Colab",
                f"Open the browser target manually, then use the prepared Drive notebook.\n\n{instructions}",
                parent=self,
            )
            return
        self._set_status(
            "Opened the exact prepared notebook search target. In Colab, select Run all and authorize Drive if prompted."
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an AMADEUS project for Google Colab.")
    parser.add_argument("--config", default="", help="Config path to load when the setup window opens.")
    args = parser.parse_args()
    app = ColabSetupApp()
    if args.config:
        app.after(200, lambda path=args.config: app._load_config(path))
    app.mainloop()


if __name__ == "__main__":
    main()
