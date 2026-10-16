# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import unittest
import numpy as np
from tests.source_loader import MODES, ArrayLosses, samplers


def sample_records(pairs):
    return [(p.row_a, p.row_b, p.label, p.frag_i, p.frag_j) for p in pairs]


def fixture_fragments(Fragment):
    definitions = [
        (0, [0, 1, 2, 3]), (1, [0, 1, 2, 3]), (2, [2, 3, 4, 5]),
        (0, [10, 11, 12]), (1, [10, 12, 14]), (2, [11, 13, 15]),
        (3, [1, 4, 7]), (3, [3, 6, 9]), (4, [30]), (5, []),
    ]
    return [Fragment(i, obj_id, values, crop_indices=list(range(i * 10, i * 10 + len(values))))
            for i, (obj_id, values) in enumerate(definitions)]


class PairSamplerTests(unittest.TestCase):
    def test_randomized_sampling_and_scores_are_exactly_equal(self):
        for mode in MODES:
            Fragment, _, Current, Dense = samplers(mode)
            for seed in range(20):
                current = Current(fixture_fragments(Fragment), seed=seed)
                dense = Dense(fixture_fragments(Fragment), seed=seed)
                self.assertEqual(current._negative_candidates, dense._negative_candidates)
                loss_rng = np.random.default_rng(seed + 1000)
                for step in range(30):
                    pairs = current.sample_positive(128) + current.sample_negative(128)
                    before = dense.sample_positive(128) + dense.sample_negative(128)
                    self.assertEqual(sample_records(pairs), sample_records(before))
                    losses = ArrayLosses(loss_rng.choice([0.0, -1.0, .2, 1.5], size=len(pairs)))
                    current.update_scores(pairs, losses)
                    dense.update_scores(before, losses)
                    np.testing.assert_array_equal(current.pos_scores, dense.pos_scores)
                    dense_scores = dense.neg_scores[dense._neg_fi_idx, dense._neg_fj_idx]
                    np.testing.assert_array_equal(current.neg_scores, dense_scores)
                    np.testing.assert_array_equal(
                        current._blend_weights(current.neg_scores, current._neg_size_w),
                        dense._blend_weights(dense_scores, dense._neg_size_w),
                    )
                    self.assertEqual(current.np_rng.bit_generator.state, dense.np_rng.bit_generator.state)
                    self.assertEqual(current.py_rng.getstate(), dense.py_rng.getstate())

    def test_duplicate_and_reversed_pair_updates(self):
        for mode in MODES:
            Fragment, Pair, Current, Dense = samplers(mode)
            fragments = fixture_fragments(Fragment)
            current, dense = Current(fragments), Dense(fragments)
            pairs = [Pair(0, 10, 0, 0, 1), Pair(10, 0, 0, 1, 0),
                     Pair(0, 10, 0, 0, 1), Pair(0, 1, 1, 0, 0), Pair(0, 1, 1, 0, 0)]
            for _ in range(10):
                losses = ArrayLosses([1, 1, 1, 1, 0])
                current.update_scores(pairs, losses)
                dense.update_scores(pairs, losses)
            np.testing.assert_array_equal(current.pos_scores, dense.pos_scores)
            np.testing.assert_array_equal(
                current.neg_scores, dense.neg_scores[dense._neg_fi_idx, dense._neg_fj_idx],
            )

    def test_candidate_order_matches_dense_with_sparse_frame_lists(self):
        rng = np.random.default_rng(149)
        for mode in MODES:
            Fragment, _, Current, Dense = samplers(mode)
            for case in range(50):
                fragments = []
                for i in range(40):
                    values = sorted(rng.choice(80, size=int(rng.integers(0, 10)), replace=False).tolist())
                    crops = list(range(i * 10, i * 10 + len(values))) if i % 7 else []
                    fragments.append(Fragment(i, int(rng.integers(0, 8)), values, crop_indices=crops))
                current, dense = Current(fragments, case), Dense(fragments, case)
                self.assertEqual(current._negative_candidates, dense._negative_candidates)

    def test_equal_interval_endpoints_are_included(self):
        for mode in MODES:
            Fragment, _, Current, _ = samplers(mode)
            fragments = [Fragment(0, 0, [1, 2], crop_indices=[0, 1]),
                         Fragment(1, 1, [2, 3], crop_indices=[2, 3])]
            self.assertEqual(Current(fragments)._negative_candidates, [(0, 1)])

    def test_overlapping_intervals_without_shared_frames_are_excluded(self):
        for mode in MODES:
            Fragment, _, Current, _ = samplers(mode)
            fragments = [Fragment(0, 0, [1, 3], crop_indices=[0, 1]),
                         Fragment(1, 1, [2, 4], crop_indices=[2, 3])]
            self.assertEqual(Current(fragments)._negative_candidates, [])

    def test_empty_sampler(self):
        for mode in MODES:
            _, _, Current, _ = samplers(mode)
            current = Current([])
            self.assertEqual(current.sample_positive(10), [])
            self.assertEqual(current.sample_negative(10), [])
            self.assertEqual(current.neg_scores.nbytes, 0)
            current.update_scores([], ArrayLosses([]))

    def test_same_id_fragments_are_not_negative_pairs(self):
        for mode in MODES:
            Fragment, _, Current, _ = samplers(mode)
            fragments = [Fragment(i, 0, [0, 1, 2], crop_indices=[i * 3, i * 3 + 1]) for i in range(3)]
            self.assertEqual(Current(fragments)._negative_candidates, [])

    def test_score_storage_scales_with_eligible_pairs(self):
        for mode in MODES:
            Fragment, _, Current, _ = samplers(mode)
            fragments = [Fragment(i, i % 2, [10 * (i // 2) + j for j in range(4)],
                                  crop_indices=[i * 4 + j for j in range(4)]) for i in range(10000)]
            current = Current(fragments)
            self.assertEqual(len(current._negative_candidates), 5000)
            self.assertEqual(current.neg_scores.shape, (5000,))
            self.assertEqual(current.neg_scores.dtype, np.dtype(np.float64))
            self.assertEqual(current.neg_scores.nbytes, 40000)
            self.assertEqual(current._neg_pair_keys.nbytes, 40000)

    def test_dense_overlap_keeps_all_eligible_pairs(self):
        for mode in MODES:
            Fragment, _, Current, Dense = samplers(mode)
            fragments = [Fragment(i, i, [0, 1], crop_indices=[2 * i, 2 * i + 1]) for i in range(40)]
            current, dense = Current(fragments), Dense(fragments)
            self.assertEqual(len(current._negative_candidates), 780)
            self.assertEqual(current._negative_candidates, dense._negative_candidates)
            self.assertEqual(sample_records(current.sample_negative(400)), sample_records(dense.sample_negative(400)))

    def test_ineligible_negative_update_fails_explicitly(self):
        for mode in MODES:
            Fragment, Pair, Current, _ = samplers(mode)
            current = Current(fixture_fragments(Fragment))
            with self.assertRaises(ValueError):
                current.update_scores([Pair(0, 80, 0, 0, 8)], ArrayLosses([1]))


if __name__ == '__main__':
    unittest.main()
