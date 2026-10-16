# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Test production direction correction with real OBB geometry and no model imports."""
import ast
import copy
import math
import pickle
import unittest

import cv2
import numpy as np

from tests.source_loader import ROOT, MODES, load_definitions

NAME = '_eval_direction_segment_flips_id'


def correction_scope(mode='main'):
    scope = load_definitions(ROOT / 'main/obb_detection.py', {
        'wrap_angle_deg', 'angle_to_unit_vec', 'unit_vec_to_angle_deg',
        'normalize_direction_vec', 'positive_long_axis_direction',
        'polygon_signed_area', 'top_left_start_index', 'rotate_start',
        'ensure_clockwise', 'obb_major_axis', 'obb_polygon_area',
    }, {'math': math, 'cv2': cv2})
    scope = load_definitions(ROOT / 'main/multi_staged_association.py', {
        'tuple_to_obb', 'valid_obb', 'angular_distance_deg',
        'reference_to_unit_vec', 'closest_long_axis_direction',
    }, scope)
    scope.update(load_definitions(ROOT / 'main/assign_types.py', {'ATYPE_DIR_FLIP'}))
    return load_definitions(ROOT / mode / 'refinement.py', {
        'get_obb', 'get_dir', '_valid_frames', '_direction_on_long_axis', NAME,
    }, scope)


def make_buffers(headings, invalid_obbs=None):
    result = {'obb_buf': {0: {}}, 'class_buf': {0: {}},
              'obb_source_buf': {0: {}}, 'direction_corrected_buf': {0: {}}}
    for frame, direction in headings.items():
        angle = float(direction) if direction is not None and np.isfinite(direction) else 0.0
        radians = math.radians(angle)
        major = np.array([math.sin(radians), -math.cos(radians)])
        minor = np.array([-major[1], major[0]])
        points = np.array([np.array([100., 100.]) + a * 10. * major + b * 2. * minor
                           for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        result['obb_buf'][0][frame] = tuple(points[:, 0].tolist() + points[:, 1].tolist())
        if direction is not None:
            result['class_buf'][0][frame] = (float(direction),)
        result['obb_source_buf'][0][frame] = (1.0,)
        result['direction_corrected_buf'][0][frame] = (0.0,)
    result['obb_buf'][0].update(invalid_obbs or {})
    return result


class DirectionGapAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = correction_scope()
        cls.before = load_definitions(
            ROOT / 'tests/fixtures/direction_segment_flips_before.py.txt', {NAME}, cls.scope,
        )

    def evaluate(self, buffers, threshold=5, baseline=False):
        scope = self.before if baseline else self.scope
        return scope[NAME](buffers, 0, 'test', 45.0, threshold)

    def assert_missing_boundary_logged(self, logs):
        reasons = [neighbor['reason'] for log in logs
                   for neighbor in log.get('details', {}).get('rejected_neighbors', [])]
        self.assertIn('missing_boundary_frame_or_heading', reasons)

    def test_sparse_observations_are_not_a_long_anchor(self):
        buffers = make_buffers({0: 0., 150: 0., 151: 180.})
        self.assertEqual(dict(self.evaluate(buffers, 150, baseline=True)[0]), {151: 0.})
        changes, logs = self.evaluate(buffers, 150)
        self.assertEqual(changes, [])
        self.assertEqual([(log['run_span'], log['valid_count']) for log in logs],
                         [(1, 1), (1, 1), (1, 1)])

    def test_short_sequences_do_not_add_up_across_a_gap(self):
        headings = {**dict.fromkeys(range(3), 0.), **dict.fromkeys(range(5, 8), 0.), 8: 180.}
        buffers = make_buffers(headings)
        self.assertNotEqual(self.evaluate(buffers, baseline=True)[0], [])
        self.assertEqual(self.evaluate(buffers)[0], [])

    def test_missing_obb_blocks_left_and_right_anchors(self):
        for headings in (
            {**dict.fromkeys(range(5), 0.), **dict.fromkeys(range(6, 8), 180.)},
            {**dict.fromkeys(range(2), 0.), **dict.fromkeys(range(3, 8), 180.)},
        ):
            with self.subTest(headings=headings):
                buffers = make_buffers(headings)
                self.assertNotEqual(self.evaluate(buffers, baseline=True)[0], [])
                changes, logs = self.evaluate(buffers)
                self.assertEqual(changes, [])
                self.assert_missing_boundary_logged(logs)

    def test_missing_heading_blocks_left_and_right_anchors(self):
        for missing, headings in (
            (5, {**dict.fromkeys(range(5), 0.), **dict.fromkeys(range(6, 8), 180.)}),
            (2, {**dict.fromkeys(range(2), 0.), **dict.fromkeys(range(3, 8), 180.)}),
        ):
            for value in (None, float('nan'), float('inf')):
                with self.subTest(missing=missing, value=value):
                    buffers = make_buffers({**headings, missing: value})
                    changes, logs = self.evaluate(buffers)
                    self.assertEqual(changes, [])
                    self.assert_missing_boundary_logged(logs)

    def test_invalid_obb_with_finite_heading_is_a_gap(self):
        headings = {**dict.fromkeys(range(5), 0.), 5: 0., 6: 180., 7: 180.}
        for invalid in (None, (float('nan'),) * 8, (0.,) * 8):
            with self.subTest(invalid=invalid):
                changes, logs = self.evaluate(make_buffers(headings, {5: invalid}))
                self.assertEqual(changes, [])
                self.assert_missing_boundary_logged(logs)

    def test_local_left_anchor_is_used_without_remote_right_anchor(self):
        headings = {**dict.fromkeys(range(5), 0.), 5: 180., 6: 180.,
                    **dict.fromkeys(range(8, 13), 180.)}
        changes, logs = self.evaluate(make_buffers(headings))
        self.assertEqual(dict(changes), {5: 0., 6: 0.})
        flips = [log for log in logs if log['action'] == 'segment_flip']
        self.assertEqual(len(flips), 1)
        self.assertEqual([a['side'] for a in flips[0]['anchors']], ['left'])
        self.assertEqual([a['side'] for a in flips[0]['anchor_merge_status']['merged_anchors']], ['left'])
        self.assertEqual(flips[0]['anchor_merge_status']['run_count_after'], 2)

    def test_sequence_after_gap_can_use_its_own_right_anchor(self):
        headings = {**dict.fromkeys(range(5), 0.), 6: 0., **dict.fromkeys(range(7, 12), 180.)}
        changes, logs = self.evaluate(make_buffers(headings))
        self.assertEqual(dict(changes), {6: 180.})
        flips = [log for log in logs if log['action'] == 'segment_flip']
        self.assertEqual([a['side'] for a in flips[0]['anchors']], ['right'])

    def test_iterative_merges_never_propagate_across_gap(self):
        headings = {**dict.fromkeys(range(5), 0.), 5: 180., 6: 0., 7: 180., 9: 180., 10: 0.}
        changes, logs = self.evaluate(make_buffers(headings))
        self.assertEqual(dict(changes), {5: 0., 7: 0.})
        flips = [log for log in logs if log['action'] == 'segment_flip']
        self.assertEqual(len(flips), 2)
        self.assertTrue(all(log['frame_end'] < 8 for log in flips))
        self.assertTrue(all(log['anchor_merge_status']['merged'] for log in flips))

    def test_previously_filled_frames_remain_eligible(self):
        buffers = make_buffers({**dict.fromkeys(range(5), 0.), 5: 180., 6: 180.})
        for frame in (3, 4):
            buffers['obb_source_buf'][0][frame] = (3.0,)
            buffers['direction_corrected_buf'][0][frame] = (1.0,)
        self.assertEqual(dict(self.evaluate(buffers)[0]), {5: 0., 6: 0.})
        self.assertEqual(self.evaluate(buffers), self.evaluate(buffers, baseline=True))

    def test_contiguous_short_flip_is_preserved(self):
        buffers = make_buffers({**dict.fromkeys(range(5), 0.), 5: 180., 6: 180.,
                                **dict.fromkeys(range(7, 12), 0.)})
        self.assertEqual(dict(self.evaluate(buffers)[0]), {5: 0., 6: 0.})
        self.assertEqual(self.evaluate(buffers), self.evaluate(buffers, baseline=True))

    def test_full_threshold_run_is_preserved(self):
        buffers = make_buffers({**dict.fromkeys(range(5), 0.), **dict.fromkeys(range(5, 10), 180.)})
        self.assertEqual(self.evaluate(buffers)[0], [])

    def test_angle_thresholds_are_unchanged(self):
        # The 60-degree boundary stays inside the six-frame anchor.
        for target, expected in ((180., {6: 0.}), (179., {})):
            with self.subTest(target=target):
                buffers = make_buffers({**dict.fromkeys(range(5), 0.), 5: 60., 6: target})
                self.assertEqual(dict(self.evaluate(buffers, 6)[0]), expected)
                self.assertEqual(self.evaluate(buffers, 6), self.evaluate(buffers, 6, baseline=True))

    def test_no_gap_results_and_logs_match_original_exactly(self):
        rng = np.random.default_rng(712)
        for case in range(40):
            directions = []
            for _ in range(8):
                directions.extend([float(rng.choice([0, 60, 90, 120, 180, 240, 270, 300]))]
                                  * int(rng.integers(1, 8)))
            buffers = make_buffers(dict(enumerate(directions, start=100)))
            with self.subTest(case=case):
                self.assertEqual(self.evaluate(buffers), self.evaluate(buffers, baseline=True))

    def test_disconnected_blocks_have_independent_results(self):
        rng = np.random.default_rng(902)
        for case in range(12):
            all_headings = {}
            separate_changes = {}
            start = 0
            for _ in range(3):
                directions = []
                for _ in range(4):
                    directions.extend([float(rng.choice([0, 60, 120, 180, 240, 300]))]
                                      * int(rng.integers(1, 8)))
                headings = dict(enumerate(directions, start=start))
                separate_changes.update(self.evaluate(make_buffers(headings))[0])
                all_headings.update(headings)
                start += len(directions) + int(rng.integers(1, 100))
            with self.subTest(case=case):
                self.assertEqual(dict(self.evaluate(make_buffers(all_headings))[0]), separate_changes)

    def test_empty_single_and_all_missing_inputs(self):
        for headings in ({}, {5: 0.}, {0: None, 1: None}, {0: float('nan'), 3: float('inf')}):
            with self.subTest(headings=headings):
                self.assertEqual(self.evaluate(make_buffers(headings))[0], [])

    def test_evaluation_does_not_mutate_buffers(self):
        buffers = make_buffers({**dict.fromkeys(range(5), 0.), 5: 180., 7: 0.})
        before = pickle.dumps(buffers)
        self.evaluate(buffers)
        self.assertEqual(pickle.dumps(buffers), before)

    def test_directionless_mode_remains_a_no_op(self):
        scope = correction_scope('main/without_direction_estimation')
        buffers = make_buffers({**dict.fromkeys(range(5), 0.), 5: 180., 7: 0.})
        self.assertEqual(scope[NAME](buffers, 0, 'test', 45., 5), ([], []))

    def test_duplicate_function_implementations_match(self):
        nodes = []
        for mode in MODES:
            tree = ast.parse((ROOT / mode / 'refinement.py').read_text(encoding='utf-8'))
            nodes.append(ast.dump(next(n for n in tree.body if getattr(n, 'name', None) == NAME)))
        self.assertEqual(*nodes)


if __name__ == '__main__':
    unittest.main()
