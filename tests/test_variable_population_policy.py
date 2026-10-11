# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only policy checks, independent of the model/tracking runtime."""
import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main'))
from variable_population import PopulationPolicy, load_population_policy


def box(x, y, width=20, height=8):
    return np.array([[x - width / 2, y - height / 2],
                     [x + width / 2, y - height / 2],
                     [x + width / 2, y + height / 2],
                     [x - width / 2, y + height / 2]], dtype=float)


class PopulationPolicyTests(unittest.TestCase):
    def policy(self, **kwargs):
        settings = dict(frame_size=(640, 480),
                        initial_stats={'individual_distance_px': 20, 'disp_ratio_p90': .15})
        settings.update(kwargs)
        return PopulationPolicy(**settings)

    def test_corner_distance_and_measured_speed_define_edge_band(self):
        policy = self.policy()
        self.assertEqual(policy.edge_margin_px, 16)
        # A center-only gate would reject x=25: its near OBB corner is x=15.
        self.assertTrue(policy.is_edge(box(25, 240)))
        self.assertFalse(policy.is_edge(box(27, 240)))
        self.assertTrue(policy.is_edge(box(615, 240)))
        self.assertTrue(policy.is_edge(box(320, 19)))
        self.assertTrue(policy.is_edge(box(320, 461)))
        self.assertTrue(policy.is_edge(box(-10, 240)))

    def test_edge_and_interior_use_distinct_confirmation_and_grace(self):
        policy = self.policy()
        self.assertEqual(policy.threshold_for_birth(box(25, 240)), 2)
        self.assertEqual(policy.threshold_for_birth(box(320, 240)), 10)
        self.assertEqual(policy.max_age_for_track(box(25, 240)), 10)
        self.assertEqual(policy.max_age_for_track(box(320, 240)), 30)

    def test_corrupt_or_missing_box_is_treated_conservatively(self):
        policy = self.policy()
        for malformed in (None, (1, 2), np.full((4, 2), np.nan)):
            self.assertFalse(policy.is_edge(malformed))
            self.assertEqual(policy.threshold_for_birth(malformed), 10)
            self.assertEqual(policy.max_age_for_track(malformed), 30)

    def test_margin_is_capped_for_large_body_or_speed_measurements(self):
        policy = self.policy(initial_stats={'individual_distance_px': 400, 'disp_ratio_p90': 100})
        self.assertEqual(policy.edge_margin_px, .15 * 480)
        self.assertFalse(policy.is_edge(box(320, 240)))

    def test_anywhere_preserves_immediate_birth_and_uniform_expiry(self):
        policy = PopulationPolicy(mode='anywhere', max_age=17,
                                  edge_confirm_frames='stale', interior_confirm_frames=0,
                                  interior_max_age=-1)
        for observation in (box(25, 240), box(320, 240), None):
            self.assertEqual(policy.threshold_for_birth(observation), 1)
            self.assertEqual(policy.max_age_for_track(observation), 17)
        self.assertEqual(policy.summary()['interior_max_age'], 17)

    def test_giant_interior_false_box_cannot_reach_the_edge_by_size_alone(self):
        policy = self.policy()
        rotation = np.array([[np.cos(.6), -np.sin(.6)], [np.sin(.6), np.cos(.6)]])
        for observation in (box(320, 240, 2000, 1000),
                            box(0, 0, 2000, 100) @ rotation.T + [320, 240]):
            self.assertFalse(policy.is_edge(observation))
            self.assertEqual(policy.threshold_for_birth(observation), 10)

    def test_missing_initial_stats_use_robust_detection_size_and_motion(self):
        frames = {}
        for frame in range(24):
            observations = [box(150 + 3 * frame, 120), box(350 + 3 * frame, 300)]
            if frame == 4:
                observations.append(box(320, 240, width=10000, height=8000))
            frames[frame] = (np.array(observations), np.ones(len(observations)), np.zeros(len(observations)))
        policy = self.policy(initial_stats={}, detection_frames=frames)
        self.assertEqual(policy.body_length_px, 20)
        self.assertEqual(policy.speed_px, 3)
        self.assertEqual(policy.edge_margin_px, 16)
        self.assertEqual(policy.body_length_source, 'detections')
        self.assertEqual(policy.motion_source, 'detections')

    def test_calibration_is_bounded_and_does_not_bridge_missing_frame_keys(self):
        frames = {2 * frame: (np.array([box(200, 200)]), [1], [0]) for frame in range(600)}
        policy = self.policy(initial_stats={}, detection_frames=frames)
        self.assertEqual(policy.body_length_px, 20)
        self.assertLessEqual(policy.calibration_frame_count, 256)
        self.assertEqual(policy.calibration_motion_count, 0)
        self.assertEqual(policy.speed_px, 0)

    def test_default_interior_grace_does_not_shorten_large_max_age(self):
        policy = self.policy(max_age=50)
        self.assertEqual(policy.interior_max_age, 50)

    def test_invalid_configuration_is_rejected(self):
        invalid = [dict(mode='unknown'), dict(frame_size=None),
                   dict(frame_size=(640, 0)), dict(frame_size=(640, np.inf)),
                   dict(edge_confirm_frames=1), dict(edge_confirm_frames=2.5),
                   dict(interior_confirm_frames=2, edge_confirm_frames=3),
                   dict(interior_confirm_frames=2, edge_confirm_frames=2),
                   dict(max_age=0), dict(interior_max_age=9)]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.policy(**settings)

    def test_loader_uses_initial_measurements_but_actual_video_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / 'initial_tracking'
            initial.mkdir()
            with (initial / 'tracking_stats.csv').open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['metric', 'value'])
                writer.writerows([['individual_distance_px', 20], ['disp_ratio_p90', .15],
                                  ['frame_width', 320], ['frame_height', 240]])
            config = dict(SESSION_PATH=directory, analysis={
                'VARIABLE_POPULATION_MODE': 'edge', 'VARIABLE_EDGE_CONFIRM_FRAMES': 3,
                'VARIABLE_INTERIOR_CONFIRM_FRAMES': 12, 'VARIABLE_INTERIOR_MAX_AGE': 40})
            policy = load_population_policy(config, frame_size=(640, 480))
            self.assertEqual(policy.frame_size, (640, 480))
            self.assertEqual(policy.edge_margin_px, 19)
            self.assertEqual(policy.threshold_for_birth(box(320, 240)), 12)
            self.assertEqual(policy.max_age_for_track(box(320, 240)), 40)

    def test_loader_probes_video_dimensions_when_caller_does_not_supply_them(self):
        probe = SimpleNamespace(read_video_frame_info=lambda path: SimpleNamespace(width=1280, height=720))
        with patch.dict(sys.modules, {'video_frame_count': probe}):
            policy = load_population_policy({'TRACKING_VIDEO_PATH': 'demo.mp4'})
        self.assertEqual(policy.frame_size, (1280, 720))


if __name__ == '__main__':
    unittest.main()
