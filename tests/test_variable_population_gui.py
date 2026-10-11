# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Hidden-window config and control-state checks for variable population modes."""

from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gui import gui_advanced_tracking as advanced
from gui import gui_easy_tracking as easy


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class VariablePopulationGUI(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.session = Path(self.directory.name)
        self.video = self.session / "clip.mp4"
        self.video.write_bytes(b"")
        self.errors = self.contexts.enter_context(patch.object(advanced.messagebox, "showerror"))
        self.contexts.enter_context(patch.object(advanced.messagebox, "showinfo"))
        self.contexts.enter_context(patch.object(advanced, "_start_maximized"))
        self.contexts.enter_context(patch.object(advanced, "configure_dpi_scaling"))
        self.contexts.enter_context(patch.object(advanced, "install_window_icon"))
        self.contexts.enter_context(patch.object(advanced, "prepare_config_for_gui", side_effect=lambda cfg, *_a, **_k: cfg))

        original_init = advanced.ctk.CTk.__init__

        def hidden_init(window, *args, **kwargs):
            original_init(window, *args, **kwargs)
            window.withdraw()

        self.contexts.enter_context(patch.object(advanced.ctk.CTk, "__init__", hidden_init))
        self.app = advanced.ConfigGUI()
        self.addCleanup(self.destroy_app)
        self.app._derive_dataset_geometry_flags = lambda *_: (True, 1, True)
        self.widgets = self.app.sections["Analysis"]["widgets"]

    def destroy_app(self):
        # Cancel Tk timers before creating another hidden root in this process.
        for timer in self.app.tk.call("after", "info"):
            self.app.tk.call("after", "cancel", timer)
        self.app.destroy()

    def load(self, analysis=None, *, variable=True):
        cfg = {
            "SESSION_PATH": str(self.session),
            "TRAINING_VIDEO_PATH": str(self.video),
            "TRACKING_VIDEO_PATH": str(self.video),
            "NUM_OBJECTS": 2,
            "VARIABLE_NUM_OBJECTS": variable,
            "analysis": analysis or {},
        }
        path = self.session / "input.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        self.assertTrue(self.app._do_load_yaml(str(path), show_msg=False))

    def set_value(self, key, value):
        widget = self.widgets[key]
        widget.configure(state="normal")
        widget.delete(0, "end")
        widget.insert(0, str(value))

    def test_mode_and_thresholds_roundtrip_without_losing_other_analysis_settings(self):
        for mode in ("edge", "anywhere"):
            with self.subTest(mode=mode):
                settings = {
                    "VARIABLE_POPULATION_MODE": mode,
                    "VARIABLE_EDGE_CONFIRM_FRAMES": 3,
                    "VARIABLE_INTERIOR_CONFIRM_FRAMES": 12,
                    "VARIABLE_INTERIOR_MAX_AGE": 45,
                    "custom_analysis_setting": "preserved",
                }
                self.load(settings)
                self.assertTrue(self.app.save_yaml(print_msg=False))
                saved = yaml.safe_load((self.session / "config.yaml").read_text(encoding="utf-8"))
                for key, value in settings.items():
                    self.assertEqual(saved["analysis"][key], value)
                self.assertNotIn("VARIABLE_POPULATION_MODE", saved["analysis"]["TRACKING_COST"])

    def test_loading_config_without_mode_resets_prior_anywhere_to_edge_defaults(self):
        self.load({"VARIABLE_POPULATION_MODE": "anywhere"})
        self.load({"MAX_AGE": 50})
        self.assertEqual(self.widgets["VARIABLE_POPULATION_MODE"].get(), "edge")
        self.assertEqual(self.widgets["VARIABLE_EDGE_CONFIRM_FRAMES"].get(), "2")
        self.assertEqual(self.widgets["VARIABLE_INTERIOR_CONFIRM_FRAMES"].get(), "10")
        self.assertEqual(self.widgets["VARIABLE_INTERIOR_MAX_AGE"].get(), "50")

    def test_controls_follow_variable_mode_and_id_tracking_skip_independently(self):
        self.load()
        mode = self.widgets["VARIABLE_POPULATION_MODE"]
        threshold = self.widgets["VARIABLE_INTERIOR_CONFIRM_FRAMES"]
        self.assertEqual(mode.cget("state"), "readonly")
        self.assertEqual(threshold.cget("state"), "normal")
        mode.set("anywhere")
        self.app._on_combo_selected("VARIABLE_POPULATION_MODE")
        self.assertEqual(mode.cget("state"), "readonly")
        self.assertEqual(threshold.cget("state"), "disabled")
        mode.set("edge")
        self.app._on_combo_selected("VARIABLE_POPULATION_MODE")
        self.app.skip_id_tracking.set(True)
        self.assertEqual(mode.cget("state"), "disabled")
        self.assertEqual(threshold.cget("state"), "disabled")
        self.assertFalse(self.app.skip_refinement.get())
        self.app.skip_id_tracking.set(False)
        self.app.variable_count.set(False)
        self.assertEqual(mode.cget("state"), "disabled")
        self.app.variable_count.set(True)
        self.assertEqual(mode.cget("state"), "readonly")
        self.assertEqual(threshold.cget("state"), "normal")

    def test_invalid_edge_settings_are_rejected_before_writing(self):
        for key, value in (
            ("VARIABLE_EDGE_CONFIRM_FRAMES", 1),
            ("VARIABLE_EDGE_CONFIRM_FRAMES", "2.5"),
            ("VARIABLE_INTERIOR_CONFIRM_FRAMES", 2),
            ("VARIABLE_INTERIOR_MAX_AGE", 9),
        ):
            with self.subTest(key=key, value=value):
                self.load()
                self.set_value(key, value)
                self.assertFalse(self.app.save_yaml(print_msg=False))
                self.errors.assert_called()
                self.assertFalse((self.session / "config.yaml").exists())

    def test_easy_tracking_defaults_to_edge_and_preserves_loaded_anywhere(self):
        # Exercise Easy Tracking's actual config builder without creating its
        # preview UI or starting its background monitor threads.
        builder = SimpleNamespace(
            _video_entry=Value(str(self.video)),
            _tracking_video_entry=Value(str(self.video)),
            _tracking_video_is_dir=Value(False),
            _num_confirmed=True,
            _variable_count=Value(True),
            _without_direction=Value(False),
            _overlap=Value("moderate"),
            _delete_tmp_files=Value(True),
            _loaded_config_path="",
            _loaded_cfg={},
            _loaded_easy_answers={},
            _skip_flags={},
            _get_session=lambda: str(self.session),
            _current_easy_answers=lambda: {},
            _overlap_values=lambda: (0.01, 0.5),
            _use_single_animal_paste=lambda: False,
            _single_animal_skips_paste=lambda: False,
            _video_dims=lambda *_: (640, 480),
            _img_size_from_short=lambda *_: 480,
            _derive_dataset_geometry_flags=lambda *_: (True, 1, True),
            _check_single_animal_config_consistency=lambda *_: None,
            _deep_setdefault=easy.EasyTrackingGUI._deep_setdefault,
        )
        builder._default_skip_flags = MethodType(easy.EasyTrackingGUI._default_skip_flags, builder)
        builder._skip_flags_from_config = MethodType(easy.EasyTrackingGUI._skip_flags_from_config, builder)
        builder._ensure_config_section = MethodType(easy.EasyTrackingGUI._ensure_config_section, builder)
        with patch.object(easy, "read_segmentation_metadata", return_value={}):
            fresh = easy.EasyTrackingGUI._build_config(builder)
            self.assertEqual(fresh["analysis"]["VARIABLE_POPULATION_MODE"], "edge")
            self.assertEqual(fresh["analysis"]["VARIABLE_INTERIOR_CONFIRM_FRAMES"], 10)
            builder._loaded_cfg = {"analysis": {"VARIABLE_POPULATION_MODE": "anywhere", "MAX_AGE": 50}}
            before = copy.deepcopy(builder._loaded_cfg)
            loaded = easy.EasyTrackingGUI._build_config(builder)
        self.assertEqual(loaded["analysis"]["VARIABLE_POPULATION_MODE"], "anywhere")
        self.assertEqual(loaded["analysis"]["VARIABLE_INTERIOR_MAX_AGE"], 50)
        self.assertEqual(builder._loaded_cfg, before)


if __name__ == "__main__":
    unittest.main()
