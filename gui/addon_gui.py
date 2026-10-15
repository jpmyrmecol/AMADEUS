# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Home sidebar and installer for independent add-on applications."""
from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
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
        for widget in self.cards.winfo_children():
            widget.destroy()
        if not items:
            self.sidebar.grid_remove()
            return
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        for item in items:
            self._card(item)

    def _card(self, item):
        manifest = item.manifest
        addon_id = manifest["id"]
        supported = platform_key() in manifest["platforms"]
        frame = tk.Frame(self.cards, bg=CARD, highlightbackground="#333333", highlightthickness=1, cursor="hand2" if supported else "arrow")
        frame.pack(fill="x", padx=8, pady=5)
        menu_button = tk.Menubutton(frame, text="...", bg=CARD, fg=FG, activebackground="#333333", activeforeground=FG, relief="flat")
        menu_button.pack(anchor="ne", padx=6, pady=(3, 0))
        menu = tk.Menu(menu_button, tearoff=False)
        menu.add_command(label="Update", command=lambda: self.update_addon(item))
        menu.add_command(label="Uninstall", command=lambda: self.uninstall(addon_id, manifest["name"]))
        menu_button.configure(menu=menu)
        labels = [
            (manifest["name"], ("Arial", 15, "bold"), FG),
            (f"v{manifest['version']}", ("Arial", 11), MUTED),
            ("\n".join(PLATFORMS[p] for p in manifest["platforms"]), ("Arial", 10), MUTED),
            ("Click to open" if supported else "Not supported on this OS / CPU", ("Arial", 10), MUTED),
        ]
        for text, font, color in labels:
            label = tk.Label(frame, text=text, font=font, fg=color, bg=CARD, wraplength=236, justify="center")
            label.pack(fill="x", padx=8, pady=(2, 5))
            if supported:
                label.bind("<Button-1>", lambda _e, key=addon_id: self.launch(key))
            self._bind_scroll(label)
        if supported:
            frame.bind("<Button-1>", lambda _e: self.launch(addon_id))
            frame.bind("<Return>", lambda _e: self.launch(addon_id))
            frame.bind("<space>", lambda _e: self.launch(addon_id))
            frame.configure(takefocus=True)
        self._bind_scroll(frame)

    def open_installer(self):
        if self.dialog is not None and self.dialog.winfo_exists():
            self.dialog.lift()
            return
        dialog = self.dialog = tk.Toplevel(self.root)
        dialog.title("AMADEUS Add-ons")
        dialog.configure(bg=BG)
        dialog.geometry("620x390")
        dialog.minsize(480, 330)
        dialog.transient(self.root)
        tk.Label(dialog, text="Install an add-on", bg=BG, fg=FG, font=("Arial", 18, "bold")).pack(pady=(18, 12))
        drop = tk.Canvas(dialog, bg=CARD, highlightbackground="#444444", highlightthickness=1, height=105)
        drop.pack(fill="x", padx=20)
        label = drop.create_text(0, 0, text="Drop an add-on ZIP here", fill=FG, font=("Arial", 14))
        drop.bind("<Configure>", lambda e: drop.coords(label, e.width / 2, e.height / 2))
        tk.Button(dialog, text="Browse...", command=self.browse, bg="#222222", fg=FG, activebackground="#333333", activeforeground=FG).pack(pady=10)
        tk.Label(dialog, textvariable=self.status, bg=BG, fg=MUTED, wraplength=560).pack(fill="x", padx=20, pady=4)
        tk.Label(dialog, text="OS support is declared by the developer. Add-ons are not sandboxed.", bg=BG, fg=MUTED, wraplength=560).pack(padx=20, pady=6)
        links = tk.Frame(dialog, bg=BG)
        links.pack(pady=6)
        for text, url in (("Developer guide", GUIDE_URL), ("Official template", TEMPLATE_URL)):
            tk.Button(links, text=text, command=lambda value=url: webbrowser.open(value), bg="#222222", fg=FG).pack(side="left", padx=5)
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
        self._task(lambda: self.store.install(package, resolve_uv(self.project_root), self._progress, expected_id),
                   lambda _item: self._installed(), cleanup=cleanup)

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

    def launch(self, addon_id):
        if not self.check_idle():
            return
        try:
            process, log = self.store.launch(addon_id)
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

    def _task(self, action, success, cleanup=None):
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
                    if cleanup is not None:
                        cleanup()
                return
        self.root.after(50, poll)
