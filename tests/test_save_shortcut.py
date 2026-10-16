"""Headless regression tests for Save Config keyboard shortcuts."""
from __future__ import annotations

import ast
from pathlib import Path
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gui.save_shortcut import bind_save_shortcut


ROOT = Path(__file__).resolve().parents[1]


class FakeWindow:
    def __init__(self):
        self.bindings = {}

    def bind_all(self, sequence, callback, add=None):
        assert add == "+"
        self.bindings[sequence] = callback


class FakeWidget:
    def __init__(self, root):
        self.root = root

    def winfo_toplevel(self):
        return self.root


class BadWidget:
    def winfo_toplevel(self):
        raise tk.TclError("window destroyed")


class SaveShortcutTests(unittest.TestCase):
    def test_ctrl_s_only_saves_own_window(self):
        window, foreign = FakeWindow(), FakeWindow()
        saved = []
        with patch("gui.save_shortcut.sys.platform", "win32"):
            bind_save_shortcut(window, lambda: saved.append("saved"))
        self.assertEqual(set(window.bindings), {"<Control-s>"})
        handler = window.bindings["<Control-s>"]
        self.assertEqual(handler(SimpleNamespace(widget=FakeWidget(window))), "break")
        self.assertEqual(saved, ["saved"])
        self.assertIsNone(handler(SimpleNamespace(widget=FakeWidget(foreign))))
        self.assertIsNone(handler(SimpleNamespace(widget=BadWidget())))
        self.assertEqual(saved, ["saved"])

    def test_command_s_and_ctrl_s_on_macos(self):
        window = FakeWindow()
        saved = []
        with patch("gui.save_shortcut.sys.platform", "darwin"):
            bind_save_shortcut(window, lambda: saved.append("saved"))
        self.assertEqual(set(window.bindings), {"<Control-s>", "<Command-s>"})
        event = SimpleNamespace(widget=FakeWidget(window))
        self.assertEqual(window.bindings["<Command-s>"](event), "break")
        self.assertEqual(window.bindings["<Control-s>"](event), "break")
        self.assertEqual(len(saved), 2)

    def test_nested_save_is_blocked_and_callback_can_run_again(self):
        window = FakeWindow()
        event = SimpleNamespace(widget=FakeWidget(window))
        calls = []

        def save():
            calls.append("save")
            self.assertEqual(window.bindings["<Control-s>"](event), "break")

        with patch("gui.save_shortcut.sys.platform", "linux"):
            bind_save_shortcut(window, save)
        handler = window.bindings["<Control-s>"]
        handler(event)
        handler(event)
        self.assertEqual(calls, ["save", "save"])

    def test_callback_failure_releases_guard(self):
        window = FakeWindow()
        calls = []

        def save():
            calls.append("save")
            if len(calls) == 1:
                raise RuntimeError("disk error")

        bind_save_shortcut(window, save)
        event = SimpleNamespace(widget=FakeWidget(window))
        with self.assertRaisesRegex(RuntimeError, "disk error"):
            window.bindings["<Control-s>"](event)
        self.assertEqual(window.bindings["<Control-s>"](event), "break")
        self.assertEqual(calls, ["save", "save"])

    def test_each_config_gui_registers_its_actual_save_action(self):
        cases = {
            "gui_easy_tracking.py": ("EasyTrackingGUI", "_save_config"),
            "gui_advanced_tracking.py": ("ConfigGUI", "save_yaml"),
            "gui_segmentation.py": ("CrossingReviewApp", "_save_config_shortcut"),
        }
        for filename, (cls_name, save_name) in cases.items():
            with self.subTest(gui=filename):
                tree = ast.parse((ROOT / "gui" / filename).read_text(encoding="utf-8"))
                cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls_name)
                constructor = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "__init__")
                calls = [
                    node for node in ast.walk(constructor)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "bind_save_shortcut"
                ]
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(calls[0].args), 2)
                self.assertIsInstance(calls[0].args[1], ast.Attribute)
                self.assertEqual(calls[0].args[1].attr, save_name)


if __name__ == "__main__":
    unittest.main()
