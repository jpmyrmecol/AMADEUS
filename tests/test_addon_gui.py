# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Tk integration tests. External updaters and native D&D registration are mocked.

The Home layout functions are loaded without its unrelated startup dependencies.
This does not replace a full CustomTkinter/launcher test on each supported OS.
"""
import ast
import math
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
from typing import Callable
import unittest
from unittest.mock import Mock, patch
import venv

from gui import addon_gui as ui
from gui import addon_manager as manager
from gui import canvas_file_drop as drop_module
from test_addon_manager import manifest, package


@unittest.skipIf(sys.platform == "linux" and not os.environ.get("DISPLAY"), "Run GUI tests with xvfb-run -a")
class AddonGuiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.store = manager.Store(self.path / "persistent")
        self.root = tk.Tk()
        self.root.geometry("1440x900")
        self.patches = [patch.object(ui.messagebox, "askyesno", return_value=True),
                        patch.object(ui.messagebox, "showerror"), patch.object(ui.messagebox, "showinfo"),
                        patch.object(ui, "resolve_uv", return_value=Path(sys.executable)),
                        patch.object(manager.Store, "_prepare_environment", self.environment)]
        for item in self.patches:
            item.start()
        self.environment_globals = self.load_home_layout()
        self.targets = self.environment_globals["build_home"](self.root)
        self.controller = self.root.AMADEUS_ADDONS
        self.root.update()

    @staticmethod
    def environment(store, release, data, uv, progress):
        venv.EnvBuilder(with_pip=False).create(release / ".venv")

    def load_home_layout(self):
        source = Path(__file__).resolve().parents[1] / "gui" / "gui_home.py"
        parsed = ast.parse(source.read_text(encoding="utf-8"))
        selected_functions = {"build_home", "make_panel", "_mix_rgb", "_rgb_to_hex", "_hex_to_rgb", "_gradient_color"}
        nodes = [node for node in parsed.body if (isinstance(node, ast.FunctionDef) and node.name in selected_functions)
                 or (isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets))]
        environment = {"ctk": SimpleNamespace(CTk=tk.Tk), "tk": tk, "Callable": Callable, "math": math,
                       "sys": sys, "queue": queue, "threading": threading, "time": time,
                       "PROJECT_ROOT": self.path / "core", "messagebox": ui.messagebox,
                       "tk_font_spec": lambda root, family, size, *styles: (family, size, *styles),
                       "home_window_title": lambda: "AMADEUS test", "probe_update_status": lambda: ("1.2.0", "1.2.0", False),
                       "AddonController": lambda root, path: ui.AddonController(root, path, store=self.store)}
        for name in ("launch_easy_tracking", "launch_preprocess", "launch_segmentation", "launch_advanced_tracking",
                     "launch_multi_config_batch", "launch_refinement", "launch_colab", "open_manual", "start_update_check", "start_uninstall"):
            environment[name] = Mock()
        exec(compile(ast.Module(nodes, type_ignores=[]), str(source), "exec"), environment)
        return environment

    def tearDown(self):
        for job in self.root.tk.call("after", "info"):
            self.root.tk.call("after", "cancel", job)
        self.root.destroy()
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def wait_for_idle(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.root.update()
            if not self.controller.busy:
                return
            time.sleep(0.01)
        self.fail("Add-on GUI task did not finish")

    def click_corner(self, name):
        easy = self.targets[0][0]
        text = easy.find_withtag(name)[-1]
        x, y = easy.coords(text)
        easy.event_generate("<Button-1>", x=int(x), y=int(y))
        self.root.update()

    def install(self, version="1.0.0"):
        self.controller.select_package(str(package(self.path, manifest(version=version))))
        self.wait_for_idle()
        self.assertIsNotNone(self.store.get("org.example.viewer"))

    def test_empty_sidebar_and_existing_targets(self):
        self.assertFalse(self.controller.sidebar.winfo_ismapped())
        self.assertEqual(len(self.targets), 6)
        self.assertEqual(self.targets[0][1], "gui_easy_tracking.py")

    def test_addons_button_between_help_and_update(self):
        easy = self.targets[0][0]
        y = {name: easy.coords(easy.find_withtag(name)[-1])[1] for name in ("help", "addons", "update")}
        self.assertLess(y["help"], y["addons"])
        self.assertLess(y["addons"], y["update"])
        self.click_corner("addons")
        self.assertTrue(self.controller.dialog.winfo_exists())
        self.environment_globals["launch_easy_tracking"].assert_not_called()

    def test_browse_install_and_right_sidebar(self):
        path = package(self.path)
        with patch.object(ui.filedialog, "askopenfilename", return_value=str(path)):
            self.controller.browse()
        self.wait_for_idle()
        self.assertEqual(len(self.controller.cards.winfo_children()), 1)
        self.assertTrue(self.controller.sidebar.winfo_ismapped())
        self.assertGreater(self.controller.sidebar.winfo_x(), self.targets[0][0].winfo_width() - 10)

    def test_local_update_menu_does_not_launch(self):
        self.install()
        frame = self.controller.cards.winfo_children()[0]
        button = next(widget for widget in frame.winfo_children() if isinstance(widget, tk.Menubutton))
        menu = self.root.nametowidget(button["menu"])
        update = package(self.path, manifest(version="2.0.0"))
        with patch.object(ui.filedialog, "askopenfilename", return_value=str(update)), patch.object(self.controller, "launch") as launch:
            menu.invoke(0)
            self.wait_for_idle()
            launch.assert_not_called()
        self.assertEqual(self.store.get("org.example.viewer").manifest["version"], "2.0.0")
        self.assertEqual(len(self.controller.cards.winfo_children()), 1)

    def test_uninstall_menu_removes_panel_and_keeps_settings(self):
        self.install()
        saved = self.store.path("org.example.viewer/data/settings.json")
        saved.write_text("keep")
        self.controller.uninstall("org.example.viewer", "Test viewer")
        self.wait_for_idle()
        self.assertFalse(self.controller.sidebar.winfo_ismapped())
        self.assertEqual(saved.read_text(), "keep")

    def test_unsupported_platform_stays_visible_without_launch_binding(self):
        other = next(key for key in manager.PLATFORMS if key != manager.platform_key())
        item = manager.Installed(manifest(platforms=[other]), self.path, 0)
        with patch.object(self.store, "list", return_value=([item], [])):
            self.controller.refresh()
            self.root.update()
        frame = self.controller.cards.winfo_children()[0]
        self.assertEqual(frame.bind("<Button-1>"), "")
        texts = [widget.cget("text") for widget in frame.winfo_children() if isinstance(widget, tk.Label)]
        self.assertIn("Not supported on this OS / CPU", texts)

    def test_many_panels_scroll_in_one_column(self):
        items = [manager.Installed(manifest(id=f"org.example.viewer{i}", name=f"Viewer {i}"), self.path / str(i), i) for i in range(8)]
        with patch.object(self.store, "list", return_value=(items, [])):
            self.controller.refresh()
            self.root.update()
        panels = self.controller.cards.winfo_children()
        self.assertEqual(len(panels), 8)
        self.assertEqual(len({panel.winfo_x() for panel in panels}), 1)
        self.assertGreater(self.controller.cards.winfo_height(), self.controller.canvas.winfo_height())
        self.assertEqual(self.root.winfo_width(), 1440)

    def test_busy_addon_operation_blocks_core_update(self):
        self.controller.busy = True
        self.click_corner("update")
        self.environment_globals["start_update_check"].assert_not_called()
        self.controller.busy = False
        self.click_corner("update")
        self.environment_globals["start_update_check"].assert_called_once()

    def test_declining_package_does_not_install(self):
        with patch.object(ui.messagebox, "askyesno", return_value=False):
            self.controller.select_package(str(package(self.path)))
            self.wait_for_idle()
        self.assertEqual(self.store.list()[0], [])

    def test_drop_callback_uses_same_install_path(self):
        # Exercise Tcl path parsing and the real shared callback with native registration mocked.
        canvas = tk.Canvas(self.root)
        handlers = {}
        canvas.drop_target_register = Mock()
        canvas.dnd_bind = lambda name, callback: handlers.setdefault(name, callback)
        fake_dnd = SimpleNamespace(require=lambda root: "test-tkdnd")
        archive = package(self.path / "." )
        spaced = self.path / "package with spaces.zip"
        archive.rename(spaced)
        with patch.object(drop_module, "TkinterDnD", fake_dnd), patch.object(drop_module, "_add_tkdnd_package_path"), patch.object(drop_module, "DND_FILES", "DND_Files"):
            result = drop_module.install_canvas_file_drop(self.root, canvas, allowed_suffixes=(".zip",), on_path=self.controller.select_package)
            self.assertTrue(result.enabled)
            handlers["<<Drop>>"](SimpleNamespace(data="{" + str(spaced) + "}"))
            self.root.update()
            self.wait_for_idle()
        self.assertEqual(len(self.controller.cards.winfo_children()), 1)
        canvas.destroy()


if __name__ == "__main__":
    unittest.main()
