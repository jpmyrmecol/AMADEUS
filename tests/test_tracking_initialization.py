# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import contextlib
import io
import unittest
import numpy as np
from tests.source_loader import MODES, initializer


def frames(scores):
    return {
        frame: (np.zeros((len(values), 4, 2)), np.asarray(values, dtype=float),
                np.zeros(len(values)))
        for frame, values in scores.items()
    }


class InitialAssignmentTests(unittest.TestCase):
    def check(self, scores, count, expected_frame, expected_order):
        for mode in MODES:
            with self.subTest(mode=mode):
                data = frames(scores)
                saved = {frame: values[1].copy() for frame, values in data.items()}
                with contextlib.redirect_stdout(io.StringIO()):
                    frame, order = initializer(mode)(data, count)
                self.assertEqual(frame, expected_frame)
                np.testing.assert_array_equal(order, expected_order)
                for frame, before in saved.items():
                    np.testing.assert_array_equal(data[frame][1], before)

    def test_first_primary_frame_is_preserved(self):
        self.check({8: [.99, .98], 3: [.60, .70, .50]}, 2, 3, [1, 0])

    def test_later_primary_frame_beats_earlier_lower_threshold(self):
        self.check({0: [.99, .499], 10: [.50, .60]}, 2, 10, [1, 0])

    def test_threshold_boundary_is_inclusive(self):
        self.check({0: [.9, .499999999], 1: [.5, .5]}, 2, 1, [0, 1])

    def test_maximizes_nth_score_not_top_score_or_mean(self):
        self.check({0: [.99, .11], 1: [.44, .43], 2: [.85, .35]}, 2, 1, [0, 1])

    def test_lower_threshold_uses_exact_score(self):
        self.check({0: [.9, .410001], 1: [.8, .410002]}, 2, 1, [0, 1])

    def test_equal_lower_threshold_prefers_earliest_frame(self):
        self.check({20: [.8, .4], 5: [.4, .8, .4]}, 2, 5, [1, 0])

    def test_tied_detection_scores_preserve_input_order(self):
        self.check({0: [.2, .45, .45, .45]}, 3, 0, [1, 2, 3])

    def test_nonfinite_scores_are_not_detections(self):
        self.check({0: [np.inf, np.nan, .49, .48, -np.inf]}, 2, 0, [2, 3])

    def test_frames_with_too_few_detections_are_ignored(self):
        self.check({0: [.99], 1: [], 2: [.3, .4]}, 2, 2, [1, 0])

    def test_no_frame_with_n_detections_reports_available_count(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, 'maximum available: 1'):
                    initializer(mode)(frames({0: [.9], 1: [.4, np.nan]}), 2)

    def test_empty_input_reports_no_detections(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, 'maximum available: 0'):
                    initializer(mode)({}, 1)

    def test_count_must_be_positive(self):
        for mode in MODES:
            for count in (0, -1):
                with self.subTest(mode=mode, count=count):
                    with self.assertRaises(ValueError):
                        initializer(mode)({}, count)

    def test_single_animal(self):
        self.check({0: [.1], 2: [.49], 1: [.25]}, 1, 2, [0])

    def test_lower_threshold_is_logged(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    initializer(mode)(frames({7: [.49, .48]}), 2)
                self.assertIn('frame 7', output.getvalue())
                self.assertIn('confidence >= 0.48', output.getvalue())

    def test_random_primary_cases_match_original_rule(self):
        rng = np.random.default_rng(187)
        for mode in MODES:
            select = initializer(mode)
            for case in range(50):
                count = case % 5 + 1
                data = frames({frame: rng.random(count + 3) for frame in range(6)})
                data[6] = frames({6: [.6] * count})[6]
                for frame in sorted(data):
                    scores = data[frame][1]
                    eligible = np.flatnonzero(scores >= .5)
                    if len(eligible) >= count:
                        expected = eligible[np.argsort(-scores[eligible], kind='stable')][:count]
                        break
                selected_frame, order = select(data, count)
                self.assertEqual(selected_frame, frame)
                np.testing.assert_array_equal(order, expected)

    def test_random_lower_threshold_cases_match_order_statistics(self):
        rng = np.random.default_rng(20261017)
        for mode in MODES:
            select = initializer(mode)
            for case in range(50):
                count = case % 5 + 1
                data = frames({frame: rng.uniform(.1, .49, count + 3) for frame in range(6)})
                nth = {frame: np.sort(value[1])[-count] for frame, value in data.items()}
                expected_frame = max(sorted(nth), key=nth.get)
                with contextlib.redirect_stdout(io.StringIO()):
                    frame, order = select(data, count)
                self.assertEqual(frame, expected_frame)
                self.assertEqual(len(order), count)
                self.assertEqual(float(data[frame][1][order[-1]]), nth[frame])


if __name__ == '__main__':
    unittest.main()
