# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import os
import subprocess
import sys
import time
import tkinter as tk
import uuid
import webbrowser
from tkinter import messagebox
from pathlib import Path
from typing import Callable
from PIL import Image, ImageTk

import customtkinter as ctk

try:
    from .project_paths import PROJECT_ROOT, gui_asset, gui_script
    from .splash_ipc import TOKEN_ENV_VAR, external_started_at, signal_stop
    from .splash_reveal import MAX_DISPLAY_MS, reveal_from_left, reveal_total_ms
    from .window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon
except ImportError:  # Preserve direct execution with: python gui/gui_home.py
    from project_paths import PROJECT_ROOT, gui_asset, gui_script
    from splash_ipc import TOKEN_ENV_VAR, external_started_at, signal_stop
    from splash_reveal import MAX_DISPLAY_MS, reveal_from_left, reveal_total_ms
    from window_icon import configure_dpi_scaling, configure_taskbar_identity, install_window_icon

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

APP_TITLE = "AMADEUS"
ONLINE_MANUAL_BASE_URL = "https://amadeus.jpmyrmecol.com"
DEFAULT_BG = "#000000"
PANEL_BORDER = "#171717"
TEXT_COLOR = "#ffffff"
SHADOW_COLOR = "#050505"
ACCENT_BLUE = "#0066d2"
ACCENT_RED_ORANGE = "#f22800"
BLACK_RGB = (0, 0, 0)
GRADIENT_STEPS = 96
HOVER_INTERVAL_MS = 16
HOVER_EASING = 0.24
FLOW_SPEED = 0.012
FLOW_RAMP = 0.055
WINDOW_UNIT = 60
WINDOW_W = WINDOW_UNIT * 24  # 6 bottom panels x 4 units, keeps each panel's original width
EASY_PANEL_H = WINDOW_UNIT * 12
BOTTOM_PANEL_H = WINDOW_UNIT * 3
WINDOW_H = EASY_PANEL_H + BOTTOM_PANEL_H
SPLASH_BG = "black"


def python_executable() -> str:
    if os.name == "nt":
        candidate = Path(sys.prefix) / "Scripts" / "python.exe"
    else:
        candidate = Path(sys.prefix) / "bin" / "python"
    return str(candidate if candidate.is_file() else Path(sys.executable))


def ensure_external_splash() -> bool:
    """Use a separate splash process so reveal stays smooth during GUI setup."""
    if os.environ.get(TOKEN_ENV_VAR):
        return True

    token = uuid.uuid4().hex
    environment = os.environ.copy()
    environment[TOKEN_ENV_VAR] = token
    try:
        kwargs = {
            "cwd": str(PROJECT_ROOT),
            "env": environment,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(
            [
                python_executable(),
                "-u",
                str(gui_script("splash_standalone.py")),
                token,
                str(MAX_DISPLAY_MS / 1000),
            ],
            **kwargs,
        )
    except OSError:
        return False

    os.environ[TOKEN_ENV_VAR] = token
    return True


def show_splash(root: ctk.CTk, reveal: bool = False) -> tk.Toplevel:
    """Show a fullscreen black window with the AMADEUS logo centered on it."""
    splash = tk.Toplevel(root)
    splash.configure(bg=SPLASH_BG)
    splash.attributes("-fullscreen", True)
    splash.attributes("-topmost", True)

    screen_w = splash.winfo_screenwidth()
    screen_h = splash.winfo_screenheight()
    canvas = tk.Canvas(splash, width=screen_w, height=screen_h, bg=SPLASH_BG, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    splash.AMADEUS_SPLASH_CANVAS = canvas

    logo_path = gui_asset("logo_amadeus_splash.png")
    if logo_path.is_file():
        # Pre-sized to match splash_standalone.py's rendering exactly, so the
        # logo doesn't jump when handing off from that process.
        photo = ImageTk.PhotoImage(Image.open(logo_path), master=splash)
        canvas.AMADEUS_SPLASH_PHOTO = photo  # keep a reference alive
        cx, cy = screen_w / 2, screen_h / 2
        canvas.create_image(cx, cy, image=photo)
        if reveal:
            w, h = photo.width(), photo.height()
            reveal_from_left(canvas, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, SPLASH_BG)
    else:
        canvas.create_text(screen_w / 2, screen_h / 2, text=APP_TITLE, font=("Arial", 32, "bold"), fill="white")

    splash.lift()
    return splash


def launch_gui_script(name: str) -> None:
    kwargs = {"cwd": str(PROJECT_ROOT)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen([sys.executable, "-u", str(gui_script(name))], **kwargs)


def launch_advanced_tracking() -> None:
    launch_gui_script("gui_advanced_tracking.py")


def launch_refinement() -> None:
    launch_gui_script("gui_refinement.py")


def launch_segmentation() -> None:
    launch_gui_script("gui_segmentation.py")


def launch_multi_config_batch() -> None:
    launch_gui_script("gui_multi_config_batch.py")


def launch_preprocess() -> None:
    launch_gui_script("gui_preprocess.py")


def launch_easy_tracking() -> None:
    launch_gui_script("gui_easy_tracking.py")


def launch_colab() -> None:
    launch_gui_script("gui_colab.py")


def _mix_rgb(start: tuple[int, int, int], end: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(int(round(s + (e - s) * amount)) for s, e in zip(start, end))


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _gradient_color(position: float, hover_amount: float, flow_amount: float) -> str:
    flow_position = (position % 2.0)
    accent_position = (1.0 - abs(flow_position - 1.0)) * flow_amount
    accent = _mix_rgb(_hex_to_rgb(ACCENT_BLUE), _hex_to_rgb(ACCENT_RED_ORANGE), accent_position)
    return _rgb_to_hex(_mix_rgb(BLACK_RGB, accent, hover_amount))


def make_panel(
    root: ctk.CTk,
    parent: tk.Misc,
    text: str,
    command: Callable[[], None],
    *,
    font_size: int,
    hover_strength: float = 1.0,
) -> tk.Canvas:
    canvas = tk.Canvas(
        parent,
        bg=DEFAULT_BG,
        highlightthickness=0,
        bd=0,
        cursor="hand2",
        takefocus=1,
    )
    font = ("Arial", font_size, "bold")
    shadow = canvas.create_text(0, 0, text=text, fill=SHADOW_COLOR, font=font, anchor="center", justify="center")
    label = canvas.create_text(0, 0, text=text, fill=TEXT_COLOR, font=font, anchor="center", justify="center")
    state = {"hover": 0.0, "target": 0.0, "phase": 0.0, "flow": 0.0, "job_id": None}

    def draw() -> None:
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        hover = float(state["hover"]) * hover_strength
        flow = float(state["flow"])
        canvas.delete("background")
        if hover <= 0.001:
            canvas.create_rectangle(
                0,
                0,
                width,
                height,
                fill=DEFAULT_BG,
                outline=PANEL_BORDER,
                width=1,
                tags="background",
            )
        else:
            for index in range(GRADIENT_STEPS):
                x0 = round(width * index / GRADIENT_STEPS)
                x1 = round(width * (index + 1) / GRADIENT_STEPS)
                position = (index / max(1, GRADIENT_STEPS - 1)) + float(state["phase"])
                color = _gradient_color(position, hover, flow)
                canvas.create_rectangle(
                    x0,
                    0,
                    x1,
                    height,
                    fill=color,
                    outline=color,
                    tags="background",
                )
            canvas.create_rectangle(
                0,
                0,
                width,
                height,
                outline=PANEL_BORDER,
                width=1,
                tags="background",
            )
        canvas.tag_lower("background")
        canvas.coords(shadow, width / 2 + 2, height / 2 + 2)
        canvas.coords(label, width / 2, height / 2)

    def animate() -> None:
        current = float(state["hover"])
        target = float(state["target"])
        state["phase"] = (float(state["phase"]) - FLOW_SPEED) % 2.0
        if abs(target - current) < 0.015 and target <= 0.0:
            state["hover"] = target
            state["flow"] = 0.0
            state["job_id"] = None
            draw()
            return
        if abs(target - current) < 0.015:
            state["hover"] = target
        else:
            state["hover"] = current + (target - current) * HOVER_EASING
        draw()
        if target > 0.0:
            state["flow"] = min(1.0, float(state["flow"]) + FLOW_RAMP)
        else:
            state["flow"] = max(0.0, float(state["flow"]) - FLOW_RAMP)
        state["job_id"] = root.after(HOVER_INTERVAL_MS, animate)

    def animate_to(target: float) -> None:
        if target > 0.0 and float(state["target"]) <= 0.0:
            state["flow"] = 0.0
            state["phase"] = 1.0
        state["target"] = target
        if state["job_id"] is None:
            animate()

    def on_click(_event=None) -> None:
        command()

    canvas.bind("<Configure>", lambda _event: draw())
    canvas.bind("<Enter>", lambda _event: animate_to(1.0))
    canvas.bind("<Leave>", lambda _event: animate_to(0.0))
    canvas.bind("<Button-1>", on_click)
    canvas.bind("<Return>", on_click)
    canvas.bind("<space>", on_click)
    draw()
    return canvas


def manual_url(language: str) -> str:
    """Return the canonical online manual URL for the requested language."""
    return f"{ONLINE_MANUAL_BASE_URL}/Manual_{language}.html"


def open_manual(language: str) -> None:
    """Open the language-specific online manual in the default browser."""
    url = manual_url(language)
    if not webbrowser.open(url):
        messagebox.showinfo("Open manual", f"Open this page in your browser:\n{url}")


def build_home(root: ctk.CTk) -> None:
    root.grid_rowconfigure(0, weight=9, minsize=EASY_PANEL_H)
    root.grid_rowconfigure(1, weight=2, minsize=BOTTOM_PANEL_H)
    for col in range(6):
        root.grid_columnconfigure(col, weight=1, minsize=WINDOW_W // 6)

    easy_panel = make_panel(
        root,
        root,
        "Easy Tracking",
        launch_easy_tracking,
        font_size=44,
        hover_strength=0.9,
    )
    easy_panel.grid(row=0, column=0, columnspan=6, sticky="nsew")

    help_state = {"hover": False, "shape_id": None, "text_id": None}

    def help_bounds() -> tuple[int, int, int, int]:
        x1, y0 = easy_panel.winfo_width() - 24, 24
        x0, y1 = x1 - 96, 54
        return x0, y0, x1, y1

    def draw_help(_event=None) -> None:
        x0, y0, x1, y1 = help_bounds()
        radius = 8
        points = [
            x0 + radius, y0, x1 - radius, y0, x1, y0, x1, y0 + radius,
            x1, y1 - radius, x1, y1, x1 - radius, y1, x0 + radius, y1,
            x0, y1, x0, y1 - radius, x0, y0 + radius, x0, y0,
        ]
        fill = ACCENT_RED_ORANGE if help_state["hover"] else "#181818"

        if help_state["shape_id"] is None:
            help_state["shape_id"] = easy_panel.create_polygon(
                points,
                smooth=True,
                splinesteps=24,
                fill=fill,
                outline="#333333",
                tags="help",
            )
            help_state["text_id"] = easy_panel.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text="Help",
                fill=TEXT_COLOR,
                font=("Arial", 13),
                tags="help",
            )
            return

        easy_panel.coords(help_state["shape_id"], *points)
        easy_panel.itemconfigure(help_state["shape_id"], fill=fill)
        easy_panel.coords(help_state["text_id"], (x0 + x1) / 2, (y0 + y1) / 2)

    def set_help_hover(value: bool) -> None:
        if help_state["hover"] == value:
            return
        help_state["hover"] = value
        if help_state["shape_id"] is not None:
            easy_panel.itemconfigure(
                help_state["shape_id"],
                fill=ACCENT_RED_ORANGE if value else "#181818",
            )

    def help_motion(event) -> None:
        x0, y0, x1, y1 = help_bounds()
        set_help_hover(x0 <= event.x <= x1 and y0 <= event.y <= y1)

    def help_leave(_event=None) -> None:
        set_help_hover(False)

    def help_click(event):
        x0, y0, x1, y1 = help_bounds()
        if x0 <= event.x <= x1 and y0 <= event.y <= y1:
            open_manual("EN")
            return "break"
        launch_easy_tracking()

    easy_panel.bind("<Motion>", help_motion, add="+")
    easy_panel.bind("<Leave>", help_leave, add="+")
    easy_panel.bind("<Button-1>", help_click)
    easy_panel.bind("<Configure>", draw_help, add="+")
    easy_panel.after_idle(draw_help)
    root.bind("<F1>", lambda e: open_manual("EN"))

    lower_panels = (
        ("Cropping\n&\nTrimming", launch_preprocess),
        ("Segmentation", launch_segmentation),
        ("Advanced Tracking", launch_advanced_tracking),
        ("Multi Config Batch", launch_multi_config_batch),
        ("Refinement", launch_refinement),
        ("Colab（beta）", launch_colab),
    )
    for col, (text, command) in enumerate(lower_panels):
        panel = make_panel(root, root, text, command, font_size=18)
        panel.grid(row=1, column=col, sticky="nsew")


def main() -> None:
    configure_taskbar_identity()
    splash_shown_at = time.monotonic()
    has_external_splash = ensure_external_splash()
    root = ctk.CTk()
    configure_dpi_scaling(root)
    root.withdraw()
    root.title(APP_TITLE)
    install_window_icon(root)

    splash: tk.Toplevel | None = None
    if not has_external_splash:
        splash_shown_at = time.monotonic()
        splash = show_splash(root, reveal=True)
        root.update()  # force the splash to render before the slower setup below

    install_window_icon(root)
    root.geometry(f"{WINDOW_W}x{WINDOW_H}")
    root.resizable(False, False)
    install_window_icon(root)

    build_home(root)

    external_start = external_started_at() if has_external_splash else None
    if external_start is None:
        elapsed_ms = int((time.monotonic() - splash_shown_at) * 1000)
    else:
        elapsed_ms = max(0, int((time.time() - external_start) * 1000))
    remaining_ms = max(0, min(reveal_total_ms(), MAX_DISPLAY_MS) - elapsed_ms)
    reveal_job: str | None = None
    main_revealed = False

    def reveal_main_window(_event=None) -> None:
        nonlocal main_revealed, reveal_job
        if main_revealed:
            return
        main_revealed = True
        if reveal_job is not None:
            try:
                root.after_cancel(reveal_job)
            except tk.TclError:
                pass
            reveal_job = None
        if splash is not None:
            try:
                splash.attributes("-fullscreen", False)
                splash.attributes("-topmost", False)
                splash.destroy()
            except tk.TclError:
                pass
        signal_stop()
        root.deiconify()
        install_window_icon(root)
        # Windows can leave the owner window (root) stuck with the topmost
        # style inherited from its topmost splash child; force-clear it.
        root.attributes("-topmost", True)
        root.attributes("-topmost", False)

    if splash is not None:
        splash.bind("<ButtonPress>", reveal_main_window, add="+")
        splash.AMADEUS_SPLASH_CANVAS.bind("<ButtonPress>", reveal_main_window, add="+")
    reveal_job = root.after(remaining_ms, reveal_main_window)
    root.mainloop()


if __name__ == "__main__":
    main()
