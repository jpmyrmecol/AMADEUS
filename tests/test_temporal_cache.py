# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Temporary persistence and IQR interaction tests, removed after verification."""
import ast
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from main.temporal_merge_cache import load_temporal_merge_cache, save_temporal_merge_cache
from main.temporal_merge_blobs import TemporalMergeGraphBuilder, TemporalMergeParams, extract_frame_nodes, evaluate_temporal_merges


def build_graph():
    builder = TemporalMergeGraphBuilder((80,80))
    for fid in range(150):
        boxes = ((20,25,30,35),(32,25,42,35)) if fid < 65 or fid >= 85 else ((20,25,42,35),)
        objs = []
        for x0,y0,x1,y1 in boxes:
            contour = np.array([[[x0,y0]],[[x1,y0]],[[x1,y1]],[[x0,y1]]], np.int32)
            objs.append(SimpleNamespace(contour=contour, area=float((x1-x0)*(y1-y0)),
                         center_x=(x0+x1)/2, center_y=(y0+y1)/2, is_crossing=False))
        builder.add_frame(extract_frame_nodes(fid,objs,(80,80),digest=fid+50))
    return builder.finish()


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name)/"config.temporal_search.npz")
        self.graph = build_graph()
        self.params = TemporalMergeParams(tau=0.3,include_one_sided=True)
        self.result = evaluate_temporal_merges(self.graph,self.params)
        self.assertTrue(self.result.excluded_count)

    def test_round_trip_all_data(self):
        save_temporal_merge_cache(self.path,self.graph,self.result,self.params,"valid-key")
        loaded = load_temporal_merge_cache(self.path,"valid-key",self.params)
        self.assertIsNotNone(loaded)
        graph,res = loaded
        for name in ("offsets","area","contour_area","cx","cy","length",
                     "width","theta","flags","src","dst","overlap","digests","breaks"):
            np.testing.assert_array_equal(getattr(graph,name),getattr(self.graph,name))
        self.assertEqual(res.levels,self.result.levels)
        self.assertEqual(res.animals,self.result.animals)
        self.assertEqual(res.excluded,self.result.excluded)
        self.assertEqual(res.intervals,self.result.intervals)
        self.assertEqual(res.diagnostics,self.result.diagnostics)
        self.assertEqual(res.counts,self.result.counts)

    def test_changed_settings_or_video_reject_cache(self):
        save_temporal_merge_cache(self.path,self.graph,self.result,self.params,"valid-key")
        self.assertIsNone(load_temporal_merge_cache(self.path,"modified-video",self.params))
        other=replace(self.params,share_rounding=0.40)
        self.assertIsNone(load_temporal_merge_cache(self.path,"valid-key",other))

    def test_corrupt_cache_falls_back(self):
        Path(self.path).write_bytes(b"not a compressed NPZ archive")
        self.assertIsNone(load_temporal_merge_cache(self.path,"valid-key",self.params))

    def test_no_unsafe_pickle_loading(self):
        source=(Path(__file__).resolve().parents[1]/"main"/"temporal_merge_cache.py").read_text()
        self.assertIn("allow_pickle=False",source)


class GuiTests(unittest.TestCase):
    @staticmethod
    def function(name):
        code=Path(__file__).resolve().parents[1]/"gui"/"gui_segmentation.py"
        mod=ast.parse(code.read_text(encoding="utf-8"))
        klass=next(v for v in mod.body if isinstance(v,ast.ClassDef)
                   and any(isinstance(child,ast.FunctionDef) and child.name==name for child in v.body))
        fn=next(v for v in klass.body if isinstance(v,ast.FunctionDef) and v.name==name)
        env={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(code),"exec"),env)
        return env[name]

    def test_bracket_double_click_resets_one_multiplier(self):
        reset=self.function("_reset_iqr_bracket_on_double_click")
        calls=[]
        fake=SimpleNamespace(
            _iqr_bracket_hit=lambda event: 1,
            _end_iqr_boxplot_drag=lambda: calls.append("end"),
            _set_iqr_multiplier=lambda i,v: calls.append((i,v)),
            _sync_iqr_area_entries=lambda **kw: calls.append("sync"),
            _redraw_area_iqr_boxplot=lambda: calls.append("redraw"))
        self.assertEqual(reset(fake,SimpleNamespace(x=20,y=10)),"break")
        self.assertEqual(calls,["end",(1,1.5),"sync","redraw"])

    def test_temporal_optional_visibility_or_enabled(self):
        source=(Path(__file__).resolve().parents[1]/"gui"/"gui_segmentation.py").read_text()
        self.assertIn('(getattr(self, "temporal_blob_section", None), True)',source)
        self.assertIn('section is getattr(self, "temporal_blob_section", None) and self.temporal_merge_enabled_var.get()',source)
        self.assertIn('height=38, bg="white"',source)
        self.assertIn('f"crossing({blob.temporal_merge_animals})"',source)


if __name__ == "__main__":
    unittest.main()
