# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Home sidebar and installer for independent add-on applications."""
from __future__ import annotations

from pathlib import Path
import queue
import time
import threading
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import webbrowser

if __package__:
    from .addon_manager import PLATFORMS, Store, download_update, inspect_package, platform_key, resolve_uv
    from .canvas_file_drop import install_canvas_file_drop
else:
    from addon_manager import PLATFORMS, Store, download_update, inspect_package, platform_key, resolve_uv
    from canvas_file_drop import install_canvas_file_drop

BG = "#000000"
CARD = "#111111"
FG = "#ffffff"
MUTED = "#aaaaaa"
GUIDE_URL = "https://github.com/jpmyrmecol/AMADEUS/blob/main/tools/ADDONS.md"
TEMPLATE_URL = "https://github.com/jpmyrmecol/AMADEUS-Addon-Template"


def _dark_button(master, *, text, command, width=120, height=30):
    """Draw add-on buttons consistently on macOS Aqua and other Tk platforms."""
    return ctk.CTkButton(
        master,
        text=text,
        command=command,
        width=width,
        height=height,
        fg_color="#222222",
        hover_color="#333333",
        text_color=FG,
        corner_radius=5,
    )


class AddonController:
    def __init__(self, root, project_root: Path, *, store: Store | None = None):
        self.root = root
        self.project_root = project_root
        self.storage_error = None
        try:
            self.store = store if store is not None else Store()
        except (OSError, ValueError) as exc:
            self.store = None
            self.storage_error = str(exc)
        self.busy = False
        self.dialog = None
        self.progress_dialog = None
        self.progress_status = tk.StringVar(root, "")
        self.progress_job = None
        self.drag = None
        self.drop_ready = False
        self.panel_frames = {}
        self.panel_ids = []
        self.status = tk.StringVar(root, "Ready.")
        self.results = queue.Queue()
        self.signature = None
        self.errors = []
        self.children = []
        self.sidebar = tk.Frame(root, bg=BG, width=290)
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(1, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)
        tk.Label(self.sidebar, text="Add-ons", bg=BG, fg=FG, font=("Arial", 16, "bold")).grid(row=0, column=0, pady=16)
        self.canvas = tk.Canvas(self.sidebar, bg=BG, width=274, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(self.sidebar, orient="vertical", command=self.canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.cards = tk.Frame(self.canvas, bg=BG)
        self.window_id = self.canvas.create_window(0, 0, window=self.cards, anchor="nw")
        self.cards.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window_id, width=e.width))
        self._bind_scroll(self.canvas)
        tk.Label(self.sidebar, textvariable=self.status, bg=BG, fg=MUTED, wraplength=260).grid(row=2, column=0, columnspan=2, padx=8, pady=8)
        self.root.protocol("WM_DELETE_WINDOW", self.close_home)
        self.refresh()
        self.root.after(2000, self._poll_installed)

    def _bind_scroll(self, widget):
        def wheel(event):
            if self.canvas.bbox("all") and self.cards.winfo_height() > self.canvas.winfo_height():
                delta = -1 if getattr(event, "num", 0) == 4 else 1
                if getattr(event, "delta", 0):
                    delta = -1 if event.delta > 0 else 1
                self.canvas.yview_scroll(delta, "units")
            return "break"
        widget.bind("<MouseWheel>", wheel)
        widget.bind("<Button-4>", wheel)
        widget.bind("<Button-5>", wheel)

    def check_idle(self) -> bool:
        if self.busy:
            messagebox.showinfo("Add-ons", "An add-on operation is in progress. Finish it before closing or updating AMADEUS.", parent=self.root)
            return False
        return True

    def close_home(self):
        if self.check_idle():
            self.root.destroy()

    def _poll_installed(self):
        if not self.busy:
            self.refresh()
        self.root.after(2000, self._poll_installed)

    def refresh(self):
        try:
            if self.store is None:
                raise ValueError(self.storage_error)
            items, errors = self.store.list()
        except (OSError, ValueError) as exc:
            items, errors = [], [str(exc)]
        self.errors = errors
        signature = [(item.manifest, str(item.release)) for item in items], errors
        if signature == self.signature:
            return
        self.signature = signature
        self.panel_ids = [item.manifest["id"] for item in items]
        self.panel_frames = {}
        self.drag = None
        for widget in self.cards.winfo_children():
            widget.destroy()
        if not items:
            self.sidebar.grid_remove()
            return
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        for item in items:
            self._card(item)
        if self.drop_ready:
            self._register_panel_drops(items)

    def enable_file_drop(self):
        """Enable native file drops only after the Home window has been revealed."""
        self.drop_ready = True
        if self.store is not None:
            items, _errors = self.store.list()
            self._register_panel_drops(items)

    def _register_panel_drops(self, items):
        for item in items:
            manifest = item.manifest
            addon_id = manifest["id"]
            extensions = manifest.get("drop_extensions", [])
            frame = self.panel_frames.get(addon_id)
            if not extensions or frame is None or platform_key() not in manifest["platforms"]:
                continue
            if not frame.winfo_exists():
                continue
            install_canvas_file_drop(
                self.root,
                frame,
                allowed_suffixes=extensions,
                on_path=lambda path, key=addon_id: self.launch(key, dropped_file=Path(path)),
                status_callback=lambda message: messagebox.showwarning("Add-on file drop", message, parent=self.root),
                label="supported add-on file",
                include_descendants=True,
            )

    def _card(self, item):
        manifest = item.manifest
        addon_id = manifest["id"]
        supported = platform_key() in manifest["platforms"]
        frame = tk.Frame(self.cards, bg=CARD, highlightbackground="#333333", highlightthickness=1, cursor="hand2" if supported else "arrow")
        frame.pack(fill="x", padx=8, pady=5)
        self.panel_frames[addon_id] = frame
        menu_button = _dark_button(frame, text="...", command=lambda: None, width=36, height=26)
        menu_button.pack(anchor="ne", padx=6, pady=(3, 0))
        menu = tk.Menu(menu_button, tearoff=False)
        menu.add_command(label="Update", command=lambda: self.update_addon(item))
        menu.add_command(label="Uninstall", command=lambda: self.uninstall(addon_id, manifest["name"]))

        def show_menu():
            try:
                menu.tk_popup(menu_button.winfo_rootx(),
                              menu_button.winfo_rooty() + menu_button.winfo_height())
            finally:
                menu.grab_release()

        menu_button.configure(command=show_menu)
        labels = [
            (manifest["name"], ("Arial", 15, "bold"), FG),
            (f"v{manifest['version']}", ("Arial", 11), MUTED),
            ("\n".join(PLATFORMS[p] for p in manifest["platforms"]), ("Arial", 10), MUTED),
            ("Click to open" if supported else "Not supported on this OS / CPU", ("Arial", 10), MUTED),
        ]
        if supported and manifest.get("drop_extensions"):
            labels.append(("Drop files: " + ", ".join(manifest["drop_extensions"]), ("Arial", 10), MUTED))
        for text, font, color in labels:
            label = tk.Label(frame, text=text, font=font, fg=color, bg=CARD, wraplength=236, justify="center")
            label.pack(fill="x", padx=8, pady=(2, 5))
            self._bind_panel_drag(label, addon_id, supported)
            self._bind_scroll(label)
        self._bind_panel_drag(frame, addon_id, supported)
        if supported:
            frame.bind("<Return>", lambda _e: self.launch(addon_id))
            frame.bind("<space>", lambda _e: self.launch(addon_id))
            frame.configure(takefocus=True)
        self._bind_scroll(frame)

    def _bind_panel_drag(self, widget, addon_id, supported):
        widget.bind("<ButtonPress-1>", lambda event: self._drag_start(event, addon_id))
        widget.bind("<B1-Motion>", self._drag_move)
        widget.bind("<ButtonRelease-1>", lambda event: self._drag_finish(event, addon_id, supported))

    def _drag_start(self, event, addon_id):
        if self.busy:
            return
        self.drag = {"id": addon_id, "start_y": event.y_root, "moved": False,
                     "candidate": list(self.panel_ids)}

    def _drag_move(self, event):
        gesture = self.drag
        if gesture is None:
            return
        if not gesture["moved"] and abs(event.y_root - gesture["start_y"]) < 7:
            return
        gesture["moved"] = True
        viewport_top = self.canvas.winfo_rooty()
        viewport_bottom = viewport_top + self.canvas.winfo_height()
        if event.y_root < viewport_top + 24:
            self.canvas.yview_scroll(-1, "units")
        elif event.y_root > viewport_bottom - 24:
            self.canvas.yview_scroll(1, "units")

        moving = gesture["id"]
        others = [key for key in self.panel_ids if key != moving]
        position = sum(event.y_root > self.panel_frames[key].winfo_rooty()
                       + self.panel_frames[key].winfo_height() / 2 for key in others)
        gesture["candidate"] = others[:position] + [moving] + others[position:]
        for key, frame in self.panel_frames.items():
            frame.configure(highlightbackground="#188b55" if key == moving else "#333333")

    def _drag_finish(self, _event, addon_id, supported):
        gesture = self.drag
        self.drag = None
        if gesture is None or gesture["id"] != addon_id:
            return
        for frame in self.panel_frames.values():
            frame.configure(highlightbackground="#333333")
        if not gesture["moved"]:
            if supported:
                self.launch(addon_id)
            return
        candidate = gesture["candidate"]
        if candidate != self.panel_ids:
            try:
                self.store.set_order(candidate)
                self.signature = None
                self.refresh()
            except (OSError, ValueError, RuntimeError) as exc:
                messagebox.showerror("Add-on order", str(exc), parent=self.root)

    def open_installer(self):
        if self.dialog is not None and self.dialog.winfo_exists():
            self.dialog.lift()
            return
        dialog = self.dialog = tk.Toplevel(self.root)
        dialog.title("AMADEUS Add-ons")
        dialog.configure(bg=BG)
        dialog.geometry("620x440")
        dialog.minsize(480, 330)
        dialog.transient(self.root)
        tk.Label(dialog, text="Install an add-on", bg=BG, fg=FG, font=("Arial", 18, "bold")).pack(pady=(15, 10))
        drop = tk.Canvas(dialog, bg=CARD, highlightbackground="#444444", highlightthickness=1, height=220)
        drop.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        label = drop.create_text(0, 0, text="Drop an add-on ZIP here", fill=FG, font=("Arial", 14))
        drop.bind("<Configure>", lambda e: drop.coords(label, e.width / 2, e.height / 2))
        _dark_button(dialog, text="Browse...", command=self.browse, width=100).pack(pady=(0, 8))
        tk.Label(dialog, textvariable=self.status, bg=BG, fg=MUTED, wraplength=560).pack(fill="x", padx=20, pady=4)
        tk.Label(dialog, text="OS support is declared by the developer. Add-ons are not sandboxed.", bg=BG, fg=MUTED, wraplength=560).pack(padx=20, pady=6)
        links = tk.Frame(dialog, bg=BG)
        links.pack(pady=(4, 14))
        for text, url in (("Developer guide", GUIDE_URL), ("Official template", TEMPLATE_URL)):
            _dark_button(links, text=text, command=lambda value=url: webbrowser.open(value), width=134).pack(side="left", padx=5)
        if self.errors:
            self.status.set("Some installation records could not be read: " + " | ".join(self.errors)[:600])
        def register_drop():
            if not dialog.winfo_exists():
                return
            state = install_canvas_file_drop(self.root, drop, allowed_suffixes=(".zip",),
                                             on_path=self.select_package, status_callback=self.status.set, label="add-on ZIP")
            if not state.enabled:
                self.status.set("Drag and drop is unavailable in this Tk session. Use Browse.")
        dialog.after(100, register_drop)

    def _show_install_progress(self, name):
        if self.dialog is not None and self.dialog.winfo_exists():
            self.dialog.destroy()
        self.dialog = None
        window = self.progress_dialog = tk.Toplevel(self.root)
        window.title("Installing add-on")
        window.configure(bg=BG)
        window.geometry("490x165")
        window.resizable(False, False)
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", lambda: None)
        tk.Label(window, text=f"Installing {name}", bg=BG, fg=FG,
                 font=("Arial", 15, "bold")).pack(pady=(16, 12))
        track = tk.Canvas(window, height=20, bg="#292929", highlightthickness=0, bd=0)
        track.pack(fill="x", padx=24)
        segment = track.create_rectangle(0, 0, 0, 20, fill="#21a366", outline="")
        self.progress_status.set("Preparing installation...")
        tk.Label(window, textvariable=self.progress_status, bg=BG, fg=MUTED,
                 wraplength=450).pack(pady=(12, 8))
        started = time.monotonic()

        def animate():
            if not window.winfo_exists():
                return
            width = max(1, track.winfo_width())
            chunk = max(40, int(width * 0.28))
            phase = ((time.monotonic() - started) / 1.25) % 2
            position = 1 - abs(1 - phase)
            left = int(max(0, width - chunk) * position)
            track.coords(segment, left, 0, left + chunk, 20)
            self.progress_job = window.after(30, animate)

        animate()

    def _close_install_progress(self):
        window = self.progress_dialog
        self.progress_dialog = None
        if window is not None and window.winfo_exists():
            if self.progress_job is not None:
                window.after_cancel(self.progress_job)
            window.destroy()
        self.progress_job = None

    def browse(self, expected_id=None):
        if not self.check_idle():
            return
        path = filedialog.askopenfilename(parent=self.dialog if self.dialog is not None and self.dialog.winfo_exists() else self.root,
                                          title="Select add-on ZIP", filetypes=[("Add-on ZIP", "*.zip")])
        if path:
            self.select_package(path, expected_id)

    def select_package(self, path, expected_id=None):
        if self.store is None:
            messagebox.showerror("Add-on storage", self.storage_error, parent=self.root)
            return
        if not self.check_idle():
            return
        self._task(lambda: inspect_package(Path(path)), lambda package: self._offer(package, expected_id))

    def _offer(self, package, expected_id=None, downloaded=False):
        def cleanup():
            if downloaded:
                package.path.unlink(missing_ok=True)
        manifest = package.manifest
        if expected_id is not None and manifest["id"] != expected_id:
            cleanup()
            raise ValueError("The selected package belongs to a different add-on.")
        if platform_key() not in manifest["platforms"]:
            cleanup()
            raise ValueError(f"The developer has not declared support for {platform_key()}.")
        try:
            current = self.store.get(manifest["id"])
        except (OSError, ValueError, KeyError, TypeError):
            cleanup()
            raise
        text = (f"{manifest['name']}  v{manifest['version']}\n"
                f"Developer (self-declared): {manifest['author']}\n"
                f"ID: {manifest['id']}\nLicense: {manifest['license']}\n"
                f"OS: {', '.join(PLATFORMS[p] for p in manifest['platforms'])}\n"
                f"Update source: {manifest.get('update_url') or 'Local ZIP only'}\n"
                f"SHA-256: {package.sha256}\n\n")
        if current:
            text += f"Installed version: {current.manifest['version']}\nPrevious update source: {current.manifest.get('update_url') or 'Local ZIP only'}\n\n"
        text += ("Only install software from a developer you trust. This application and its dependencies can access your files and network. "
                 "Developer identity is not verified. This is not a sandbox.\n\n"
                 "Install in a separate Python environment? The first setup may download Python and packages.")
        if not messagebox.askyesno("Install add-on", text, parent=self.root, default=messagebox.NO):
            cleanup()
            self.status.set("Installation cancelled.")
            return
        self._show_install_progress(manifest["name"])
        self._task(lambda: self.store.install(package, resolve_uv(self.project_root), self._progress, expected_id),
                   lambda _item: self._installed(), cleanup=cleanup, installation=True)

    def _installed(self):
        self.refresh()
        self.status.set("Installation complete. Click the new panel on the right of Home to open it.")

    def update_addon(self, item):
        if not self.check_idle():
            return
        if not item.manifest.get("update_url"):
            self.browse(item.manifest["id"])
            return
        def received(package):
            if package is None:
                self.status.set("This add-on is up to date.")
                messagebox.showinfo("Add-ons", "This add-on is up to date.", parent=self.root)
            else:
                self._offer(package, item.manifest["id"], downloaded=True)
        self._task(lambda: download_update(item, self.store.path(".downloads"), self._progress), received)

    def uninstall(self, addon_id, name):
        if not self.check_idle():
            return
        if messagebox.askyesno(
            "Uninstall add-on",
            f"Uninstall {name}?\n\n"
            "The entire add-on folder will be deleted, including its Python environment, "
            "saved settings, generated data and logs. Files stored elsewhere will not be removed.",
            parent=self.root,
            default=messagebox.NO,
        ):
            self._task(lambda: self.store.uninstall(addon_id), lambda _result: self._uninstalled())

    def _uninstalled(self):
        self.refresh()
        self.status.set("Add-on folder removed, including settings, data and logs.")

    def launch(self, addon_id, *, dropped_file=None):
        if not self.check_idle():
            return
        try:
            process, log = self.store.launch(addon_id, **({"dropped_file": dropped_file} if dropped_file is not None else {}))
        except Exception as exc:
            messagebox.showerror("Unable to launch add-on", str(exc), parent=self.root)
            return
        self.children.append(process)
        def poll():
            status = process.poll()
            if status is None:
                self.root.after(1000, poll)
                return
            self.children.remove(process)
            if status:
                messagebox.showerror("Add-on exited", f"The add-on exited with code {status}.\n\nLog: {log}", parent=self.root)
        self.root.after(500, poll)

    def _progress(self, message):
        self.results.put(("progress", message))

    def _task(self, action, success, cleanup=None, installation=False):
        if not self.check_idle():
            return
        self.busy = True
        self.status.set("Working...")
        def worker():
            try:
                self.results.put(("success", action()))
            except Exception as exc:
                self.results.put(("error", str(exc)))
        threading.Thread(target=worker, name="AMADEUS add-on manager", daemon=False).start()
        def poll():
            while True:
                try:
                    kind, value = self.results.get_nowait()
                except queue.Empty:
                    self.root.after(50, poll)
                    return
                if kind == "progress":
                    self.status.set(value)
                    if installation:
                        self.progress_status.set(value)
                    continue
                self.busy = False
                try:
                    if kind == "error":
                        raise RuntimeError(value)
                    success(value)
                except Exception as exc:
                    self.status.set(str(exc))
                    messagebox.showerror("Add-ons", str(exc), parent=self.root)
                finally:
                    if installation:
                        self._close_install_progress()
                    if cleanup is not None:
                        cleanup()
                return
        self.root.after(50, poll)
