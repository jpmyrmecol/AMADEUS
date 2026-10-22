# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Verify actual Advanced Tracking control states without loading GUI dependencies."""

from __future__ import annotations

import ast
import copy
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace


GUI_SOURCE = Path(__file__).resolve().parents[1] / "gui" / "gui_advanced_tracking.py"


def load_skip_state_method():
    """Execute the unmodified GUI method in isolation to keep CI headless."""
    module = ast.parse(GUI_SOURCE.read_text(encoding="utf-8"), filename=str(GUI_SOURCE))
    method = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_refresh_skip_dependent_states"
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    tree = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {}
    exec(compile(tree, str(GUI_SOURCE), "exec"), namespace)
    return namespace[method.name]


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Control:
    def __init__(self, value, *, state="normal"):
        self.value = value
        self.state = state

    def configure(self, **kwargs):
        self.state = kwargs["state"]

    def get(self):
        return self.value


def make_gui_stub(*, detection_skipped, id_skipped, refinement_skipped, video_skipped):
    controls = {
        "start": Control("25", state="disabled"),
        "end": Control("175", state="disabled"),
    }
    gui = SimpleNamespace(
        _sections_ready=True,
        sections={
            "Initial Tracking": {"widgets": {"AUTO_PARAMS": Value(False)}},
            "Analysis": {
                "widgets": {
                    "ENABLE": Value(False),
                    "VARIABLE_POPULATION_MODE": Value("edge"),
                }
            },
            "Create Video": {"widgets": {"EXPORT_IMAGES": Value(False)}},
        },
        variable_count=Value(False),
        _parameter_rows={},
        training_range={"start": Control("5"), "end": Control("200")},
        analysis_range=controls,
        _shuffle_colors_button=Control(None),
        _set_create_images_group_enabled=lambda *_: None,
        _set_category_visible=lambda *_: None,
        skip_initial_tracking=Value(False),
        skip_trajectory_direction_filtering=Value(False),
        skip_refine_blobs_through_tracking=Value(False),
        skip_paste_blobs_with_crossing=Value(False),
        skip_paste_blobs_clustered=Value(False),
        skip_creating_direction_dataset=Value(False),
        skip_cropping=Value(False),
        skip_training=Value(False),
        skip_detection=Value(detection_skipped),
        skip_id_tracking=Value(id_skipped),
        skip_refinement=Value(refinement_skipped),
        skip_creating_video=Value(video_skipped),
    )
    return gui


class AnalysisRangeControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refresh = staticmethod(load_skip_state_method())

    def test_range_stays_enabled_if_any_analysis_or_video_stage_runs(self):
        for skips in itertools.product((False, True), repeat=4):
            with self.subTest(skips=skips):
                gui = make_gui_stub(
                    detection_skipped=skips[0],
                    id_skipped=skips[1],
                    refinement_skipped=skips[2],
                    video_skipped=skips[3],
                )
                self.refresh(gui)
                expected = "disabled" if all(skips) else "normal"
                self.assertEqual({w.state for w in gui.analysis_range.values()}, {expected})
                # Disabling controls must never clear values that will be saved.
                self.assertEqual(gui.analysis_range["start"].get(), "25")
                self.assertEqual(gui.analysis_range["end"].get(), "175")

    def test_video_toggle_updates_range_without_touching_other_skips(self):
        gui = make_gui_stub(
            detection_skipped=True,
            id_skipped=True,
            refinement_skipped=True,
            video_skipped=True,
        )
        self.refresh(gui)
        self.assertEqual(gui.analysis_range["start"].state, "disabled")
        gui.skip_creating_video.set(False)
        self.refresh(gui)
        self.assertEqual(gui.analysis_range["start"].state, "normal")
        gui.skip_creating_video.set(True)
        self.refresh(gui)
        self.assertEqual(gui.analysis_range["end"].state, "disabled")
        self.assertTrue(gui.skip_detection.get())
        self.assertTrue(gui.skip_id_tracking.get())
        self.assertTrue(gui.skip_refinement.get())
        self.assertEqual(gui.training_range["start"].state, "normal")

    def test_no_changes_before_sections_are_ready(self):
        gui = make_gui_stub(
            detection_skipped=True, id_skipped=True,
            refinement_skipped=True, video_skipped=False,
        )
        gui._sections_ready = False
        self.refresh(gui)
        self.assertEqual({w.state for w in gui.analysis_range.values()}, {"disabled"})


if __name__ == "__main__":
    unittest.main()
