# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Regression tests for the Easy Tracking label resize callback (headless)."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "gui" / "gui_easy_tracking.py"


def _load_wrap_callback():
    """Load only the callback to avoid Tk windows and GUI dependencies."""
    module = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    callbacks = [
        item
        for owner in module.body
        if isinstance(owner, ast.ClassDef)
        for item in owner.body
        if isinstance(item, ast.FunctionDef) and item.name == "_wrap_canvas_panel_label"
    ]
    if len(callbacks) != 1:
        raise AssertionError("Expected one Easy Tracking label wrap callback")
    namespace = {"tk": SimpleNamespace(Event=object)}
    callback_module = ast.Module(body=callbacks, type_ignores=[])
    exec(compile(callback_module, str(SOURCE), "exec"), namespace)
    return namespace["_wrap_canvas_panel_label"]


class _TclValue:
    """Behaves like Tcl_Obj, which supports str() but not int()."""

    def __init__(self, text):
        self.text = str(text)

    def __str__(self):
        return self.text


class _Label:
    def __init__(self, **settings):
        self.settings = {key: _TclValue(value) for key, value in settings.items()}
        self.configure_calls = []

    def cget(self, name):
        return self.settings[name]

    def winfo_pixels(self, distance):
        if not isinstance(distance, str):
            raise AssertionError("Tk distance should be passed as a string")
        return {"2p": 3, "1m": 4}[distance] if distance in ("2p", "1m") else int(distance)

    def configure(self, **settings):
        self.configure_calls.append(settings)
        self.settings.update({key: _TclValue(value) for key, value in settings.items()})


class EasyTrackingLabelWrapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.callback = staticmethod(_load_wrap_callback())

    def test_tcl_values_are_converted_and_no_redundant_reconfigure(self):
        label = _Label(padx=10, borderwidth=2, highlightthickness=1, wraplength=600)
        event = SimpleNamespace(widget=label, width=250)
        self.callback(None, event)
        self.assertEqual(label.configure_calls, [{"wraplength": 224}])
        self.callback(None, event)
        self.assertEqual(label.configure_calls, [{"wraplength": 224}])

    def test_tk_distance_units_are_converted_to_pixels(self):
        label = _Label(padx="2p", borderwidth="1m", highlightthickness=1, wraplength=600)
        self.callback(None, SimpleNamespace(widget=label, width=150))
        self.assertEqual(label.configure_calls, [{"wraplength": 134}])

    def test_wraplength_is_never_below_one(self):
        label = _Label(padx=10, borderwidth=2, highlightthickness=1, wraplength=600)
        self.callback(None, SimpleNamespace(widget=label, width=15))
        self.assertEqual(label.configure_calls, [{"wraplength": 1}])


if __name__ == "__main__":
    unittest.main()
