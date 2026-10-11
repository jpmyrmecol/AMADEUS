# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Regression cases for both variable-population association pipelines."""
import importlib
import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(__file__).resolve().parents[1] / 'main'
sys.path.insert(0, str(MAIN))
from variable_population import PopulationPolicy


def box(x, y=100):
    return np.array([[x-10, y-5], [x+10, y-5],
                     [x+10, y+5], [x-10, y+5]], dtype=np.float32)


def frames(observations, last=None):
    last = max(observations, default=0) if last is None else last
    return {f: (np.array([box(x, y) for x, y in observations.get(f, [])],
                         dtype=np.float32).reshape(-1, 4, 2),
                np.ones(len(observations.get(f, []))),
                np.zeros(len(observations.get(f, [])))) for f in range(last+1)}


def visible(buffers):
    return {tid: {f for f, value in rows.items() if np.isfinite(value).all()}
            for tid, rows in buffers['obb_buf'].items()}


class VariablePopulationTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trackers = [importlib.import_module(name) for name in (
            'multi_staged_association_variable',
            'without_direction_estimation.multi_staged_association_variable')]

    def run_both(self, data, check, mode='edge'):
        for module in self.trackers:
            with self.subTest(pipeline=module.__name__):
                policy = PopulationPolicy(
                    mode=mode, frame_size=(400, 200),
                    initial_stats={'individual_distance_px': 20,
                                   'disp_ratio_p90': 0.1},
                    max_age=3, interior_max_age=12,
                    edge_confirm_frames=2, interior_confirm_frames=5)
                result = module.track_from_detection_frames(
                    data, num_objects=99, nms_iou=0.8, max_age=3,
                    strong_iou=0.5, strong_dir_deg=90,
                    population_policy=policy)
                check(result, module)

    def test_single_and_short_interior_false_positives_are_absent(self):
        data = frames({f: [(200, 100)] for f in range(4)}, last=7)
        self.run_both(data, lambda result, _: self.assertEqual(visible(result), {}))

    def test_persistent_initial_interior_animal_is_backdated(self):
        data = frames({f: [(200, 100)] for f in range(7)})
        self.run_both(data, lambda result, _: self.assertEqual(
            visible(result), {0: set(range(7))}))

    def test_edge_birth_needs_only_two_observations(self):
        data = frames({3: [(12, 100)], 4: [(13, 100)]}, last=8)
        self.run_both(data, lambda result, _: self.assertEqual(visible(result), {0: {3, 4}}))

    def test_one_frame_edge_false_positive_is_also_absent(self):
        data = frames({3: [(12, 100)]}, last=4)
        self.run_both(data, lambda result, _: self.assertEqual(visible(result), {}))

    def test_intermittent_interior_false_positives_cannot_accumulate_hits(self):
        data = frames({f: [(200, 100)] for f in (0, 1, 3, 4, 6, 7, 9, 10)})
        self.run_both(data, lambda result, _: self.assertEqual(visible(result), {}))

    def test_interior_birth_requirement_does_not_drop_when_candidate_reaches_edge(self):
        data = frames({f: [(x, 100)] for f, x in enumerate((60, 44, 28, 12))})
        self.run_both(data, lambda result, _: self.assertEqual(visible(result), {}))

    def test_missing_interior_animal_recovers_same_id_after_edge_timeout(self):
        observed = set(range(5)) | {11, 12}
        data = frames({f: [(200, 100)] for f in observed})
        def check(result, _):
            self.assertEqual(visible(result), {0: observed})
            self.assertTrue(all(f not in visible(result)[0] for f in range(5, 11)))
        self.run_both(data, check)

    def test_edge_exit_retires_and_reentry_gets_new_monotonic_id(self):
        data = frames({f: [(12, 100)] for f in (0, 1, 7, 8)})
        self.run_both(data, lambda result, _: self.assertEqual(
            visible(result), {0: {0, 1}, 1: {7, 8}}))

    def test_persistent_interior_disappearance_eventually_retires(self):
        data = frames({f: [(200, 100)] for f in list(range(5)) + list(range(20, 25))})
        self.run_both(data, lambda result, _: self.assertEqual(
            visible(result), {0: set(range(5)), 1: set(range(20, 25))}))

    def test_sparse_frame_keys_reset_candidate_confirmation(self):
        data = frames({f: [(200, 100)] for f in (0, 1, 3, 4, 5)})
        del data[2]
        self.run_both(data, lambda result, _: self.assertEqual(visible(result), {}))

    def test_legacy_anywhere_preserves_immediate_birth_and_uniform_retirement(self):
        data = frames({0: [(200, 100)], 7: [(200, 100)]})
        self.run_both(data, lambda result, _: self.assertEqual(
            visible(result), {0: {0}, 1: {7}}), mode='anywhere')

    def test_empty_population_retains_full_frame_range(self):
        data = frames({}, last=10)
        def check(result, _):
            self.assertEqual(visible(result), {})
            self.assertEqual(result['variable_frame_range'], (0, 10))
        self.run_both(data, check)

    def test_default_policy_with_video_dimensions_rejects_short_interior_detection(self):
        data = frames({f: [(200, 100)] for f in range(4)}, last=7)
        for module in self.trackers:
            with self.subTest(pipeline=module.__name__):
                result = module.track_from_detection_frames(
                    data, num_objects=99, nms_iou=.8, max_age=3,
                    strong_iou=.5, strong_dir_deg=90, frame_size=(400, 200))
                self.assertEqual(visible(result), {})
                self.assertEqual(result['variable_population_stats']['mode'], 'edge')

    def cached_job(self, module, session_path, mode='edge'):
        """Use the real detection DataFrame schema and artifact layout."""
        observations = {f: [(320, 180)] for f in range(6)}
        observations[7] = [(320, 180)]
        observations[2].append((180, 280))  # One-frame interior false positive.
        for frame in (3, 4, 5):
            observations[frame].append((12, 50))
        rows = []
        for frame, centers in observations.items():
            for det_index, (x, y) in enumerate(centers):
                points = box(x, y)
                rows.append(dict(frame=frame, det_index=det_index, score=.9, direction=0.,
                    **{f'{axis}{corner}': float(points[corner, index])
                       for index, axis in enumerate('xy') for corner in range(4)}))
        analysis = dict(MAX_AGE=3, VARIABLE_POPULATION_MODE=mode,
                        VARIABLE_EDGE_CONFIRM_FRAMES=3,
                        VARIABLE_INTERIOR_CONFIRM_FRAMES=6,
                        VARIABLE_INTERIOR_MAX_AGE=12)
        if mode == 'anywhere':
            # These settings are unused in the legacy mode, even when stale.
            analysis.update(VARIABLE_EDGE_CONFIRM_FRAMES='obsolete',
                            VARIABLE_INTERIOR_CONFIRM_FRAMES=0,
                            VARIABLE_INTERIOR_MAX_AGE=-1)
        spec = dict(session_path=str(session_path), model_name='test_model',
                    dataset_name='test_dataset', run_name='last', num_objects=0,
                    info=dict(name='demo', path='unused.mp4', first_frame=0,
                              last_frame=8, width=640, height=360),
                    conf_th=.3, nms_iou=.8, strong_iou=.5, strong_dir_deg=90,
                    max_age=3, tracking_cost_cfg={}, sharp_turn_accept_run=3,
                    population_config=dict(SESSION_PATH=str(session_path), analysis=analysis))
        out_dir = module.build_tracking_out_dir(
            spec['session_path'], spec['model_name'], spec['dataset_name'],
            spec['run_name'], spec['info']['name'])
        cache = Path(module._artifact_path(out_dir, 'all_blobs.pkl'))
        cache.parent.mkdir(parents=True)
        pd.DataFrame(rows).to_pickle(cache)
        initial = session_path / 'initial_tracking'
        initial.mkdir()
        with (initial / 'tracking_stats.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['metric', 'value'])
            writer.writerows([['individual_distance_px', 20], ['disp_ratio_p90', .1],
                              ['frame_width', 100], ['frame_height', 100]])
        return spec, out_dir

    def test_cached_job_policy_and_correction_preserve_confirmed_observations(self):
        for module in self.trackers:
            with self.subTest(pipeline=module.__name__), tempfile.TemporaryDirectory() as directory:
                spec, out_dir = self.cached_job(module, Path(directory))
                correction_name = module.__name__.replace('multi_staged_association_variable',
                                                           'refinement_variable')
                correction = importlib.import_module(correction_name)
                # Only suppress progress display; association, serialization,
                # config loading and correction all run their real code.
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    module._run_single_tracking_job(spec)
                observed = correction.load_tracking_buffers(out_dir)
                expected = {0: set(range(6)) | {7}, 1: {3, 4, 5}}
                self.assertEqual(visible(observed), expected)
                self.assertEqual(observed['variable_frame_range'], (0, 8))
                for frame in range(6):
                    self.assertEqual(observed['pos_buf'][0][frame], (320., 180.))
                self.assertTrue(all(observed['pos_buf'][tid][frame][0] != 180
                                    for tid, seen in expected.items() for frame in seen))
                with (Path(out_dir) / 'population_policy.json').open(encoding='utf-8') as file:
                    persisted_policy = json.load(file)
                expected_settings = dict(mode='edge', frame_size=[640, 360],
                    edge_confirm_frames=3, interior_confirm_frames=6,
                    edge_max_age=3, interior_max_age=12, edge_margin_px=16.,
                    body_length_source='initial_tracking', motion_source='initial_tracking',
                    confirmed_births={'edge': 1, 'interior': 1})
                for key, value in expected_settings.items():
                    self.assertEqual(persisted_policy[key], value)
                self.assertGreaterEqual(persisted_policy['rejected_candidates'], 1)
                self.assertEqual(observed['variable_population_stats']['frame_size'], (640, 360))
                correction_spec = dict(
                    out_dir=out_dir, session_path=spec['session_path'],
                    model_name=spec['model_name'], dataset_name=spec['dataset_name'],
                    run_name=spec['run_name'], video_name=spec['info']['name'],
                    num_objects=0, strong_iou=.5, strong_dir_deg=90, max_age=3,
                    flip_frame_threshold=150, flip_duration_sec=5., source_fps=30.,
                    num_workers=1, embedding_enabled=False, embedding={})
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    correction._run_single_correction_job(correction_spec)
                corrected = correction.load_tracking_buffers(out_dir, 'filled')
                self.assertEqual(visible(corrected), expected)
                self.assertEqual(corrected['variable_frame_range'], (0, 8))
                self.assertEqual(corrected['variable_population_stats'],
                                 observed['variable_population_stats'])
                self.assertFalse(np.isfinite(corrected['obb_buf'][0].get(6, (np.nan,))).all())

    def test_cached_anywhere_job_restores_interior_transient_to_persisted_output(self):
        for module in self.trackers:
            with self.subTest(pipeline=module.__name__), tempfile.TemporaryDirectory() as directory:
                spec, out_dir = self.cached_job(module, Path(directory), mode='anywhere')
                correction = importlib.import_module(module.__name__.replace(
                    'multi_staged_association_variable', 'refinement_variable'))
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    module._run_single_tracking_job(spec)
                observed = correction.load_tracking_buffers(out_dir)
                self.assertEqual(visible(observed),
                                 {0: set(range(6)) | {7}, 1: {2}, 2: {3, 4, 5}})
                self.assertEqual(observed['pos_buf'][1][2], (180., 280.))
                with (Path(out_dir) / 'population_policy.json').open(encoding='utf-8') as file:
                    persisted_policy = json.load(file)
                self.assertEqual(persisted_policy['mode'], 'anywhere')
                self.assertEqual(persisted_policy['edge_confirm_frames'], 1)
                self.assertEqual(persisted_policy['interior_confirm_frames'], 1)
                self.assertEqual(persisted_policy['edge_max_age'], 3)
                self.assertEqual(persisted_policy['interior_max_age'], 3)


if __name__ == '__main__':
    unittest.main()
