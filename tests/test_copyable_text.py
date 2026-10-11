# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Regression tests for copying visible GUI text without changing GUI interactions."""

from __future__ import annotations

from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

import customtkinter as ctk

from gui.copyable_text import copyable_text_at, install_right_click_copy


class CopyableTextTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = ctk.CTk()
        except tk.TclError as exc:
            self.skipTest(f"No Tk display: {exc}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def test_native_and_customtkinter_labels(self):
        native = tk.Label(self.root, text="Number of Objects")
        custom = ctk.CTkLabel(self.root, text="Variable population")
        self.assertEqual(copyable_text_at(native), "Number of Objects")
        self.assertEqual(copyable_text_at(custom), "Variable population")
        self.assertEqual(copyable_text_at(custom._label), "Variable population")

    def test_textvariable_labels_and_buttons(self):
        state = tk.StringVar(master=self.root, value="Frame 123")
        native = tk.Label(self.root, textvariable=state)
        self.assertEqual(copyable_text_at(native), "Frame 123")
        state.set("Frame 124")
        self.assertEqual(copyable_text_at(native), "Frame 124")
        button = ctk.CTkButton(self.root, text="Advanced Tracking")
        self.assertEqual(copyable_text_at(button), "Advanced Tracking")
        self.assertEqual(copyable_text_at(ttk.Label(self.root, text="Status")), "Status")

    def test_editable_fields_and_unlabeled_canvas_are_untouched(self):
        entry = ctk.CTkEntry(self.root)
        native = tk.Entry(self.root)
        canvas = tk.Canvas(self.root, width=150, height=30)
        canvas.create_rectangle(0, 0, 30, 25)
        self.assertIsNone(copyable_text_at(entry._entry))
        self.assertIsNone(copyable_text_at(native))
        self.assertIsNone(copyable_text_at(canvas, 10, 10))

    def test_canvas_text_only_under_pointer(self):
        canvas = tk.Canvas(self.root, width=200, height=60)
        canvas.create_text(45, 20, text="AMADEUS", anchor="center")
        self.assertEqual(copyable_text_at(canvas, 45, 20), "AMADEUS")
        self.assertIsNone(copyable_text_at(canvas, 150, 50))

    def test_one_installer_and_menu_copy(self):
        install_right_click_copy(self.root)
        initial_binding = self.root.bind_all("<Button-3>")
        install_right_click_copy(self.root)
        self.assertEqual(self.root.bind_all("<Button-3>"), initial_binding)
        label = tk.Label(self.root, text="Interior Max Age")
        event = SimpleNamespace(widget=label, x=2, y=2, x_root=10, y_root=10)
        with patch.object(self.root._amadeus_copy_text_menu, "tk_popup"):
            self.assertEqual(self.root._amadeus_copy_text_handler(event), "break")
        self.root._amadeus_copy_text_menu.invoke(0)
        self.assertEqual(self.root.clipboard_get(), "Interior Max Age")


if __name__ == "__main__":
    unittest.main()
