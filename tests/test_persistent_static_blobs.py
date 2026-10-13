# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for frame-window persistent static blob exclusion."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from main.persistent_static_blobs import PersistentStaticScanner, contour_iou


def blob(x=40, y=30, w=22, h=9, blob_index=0, outlier=False):
    cnt = np.array([
        [x, y], [x + w, y], [x + w, y + h], [x, y + h],
    ], dtype=np.int32).reshape(-1, 1, 2)
    return SimpleNamespace(
        contour=cnt, center_x=x + w / 2, center_y=y + h / 2,
        blob_index=blob_index, is_crossing=outlier,
    )


def test_full_static_window_excludes_all_original_frame_occurrences():
    scanner = PersistentStaticScanner(15, presence=0.98, min_iou=0.8, max_displacement=0.2)
    for fid in range(20):
        scanner.add_frame(fid, [blob(x=40 + fid % 2)])
    assert set(scanner.excluded) == set(range(20))
    assert scanner.excluded_count == 20


def test_stationary_candidate_does_not_exclude_motion_or_other_blobs():
    scanner = PersistentStaticScanner(15)
    for fid in range(30):
        scanner.add_frame(fid, [
            blob(x=40, blob_index=0),
            blob(x=120 + fid * 3, blob_index=1),
        ])
    assert all(indices == {0} for indices in scanner.excluded.values())
    assert len(scanner.excluded) == 30


def test_requires_complete_window_at_configured_presence():
    scanner = PersistentStaticScanner(15, presence=0.98)
    for fid in range(15):
        scanner.add_frame(fid, [] if fid == 7 else [blob()])
    assert scanner.excluded_count == 0


def test_reappearance_elsewhere_does_not_mask_area_globally():
    scanner = PersistentStaticScanner(15)
    for fid in range(15):
        scanner.add_frame(fid, [blob()])
    for fid in range(15, 30):
        scanner.add_frame(fid, [blob(x=40 + (fid - 14) * 3)])
    assert scanner.excluded[0] == {0}
    assert 29 not in scanner.excluded


def test_prior_segmentation_outlier_is_not_part_of_static_search():
    scanner = PersistentStaticScanner(15)
    for fid in range(15):
        scanner.add_frame(fid, [blob(outlier=True)])
    assert scanner.excluded_count == 0


def test_contour_iou_is_symmetric_and_exact():
    a, b = blob().contour, blob(x=41).contour
    assert contour_iou(a, a) == 1.0
    assert 0 < contour_iou(a, b) < 1
    assert contour_iou(a, b) == pytest.approx(contour_iou(b, a))


def test_nonconsecutive_frames_raise_error():
    scanner = PersistentStaticScanner(15)
    scanner.add_frame(0, [blob()])
    with pytest.raises(ValueError, match="consecutive"):
        scanner.add_frame(2, [blob()])


def test_presence_98_percent_accepts_one_missing_frame_in_fifty():
    scanner = PersistentStaticScanner(50, presence=0.98)
    for fid in range(50):
        scanner.add_frame(fid, [] if fid == 25 else [blob()])
    assert scanner.excluded_count == 49
    assert 25 not in scanner.excluded


def test_presence_98_percent_rejects_two_missing_frames_in_fifty():
    scanner = PersistentStaticScanner(50, presence=0.98)
    for fid in range(50):
        scanner.add_frame(fid, [] if fid in (25, 30) else [blob()])
    assert scanner.excluded_count == 0


def test_median_shape_iou_filters_inconsistent_rotation():
    scanner = PersistentStaticScanner(6, min_iou=0.8)
    for fid in range(6):
        # Both blobs have the same center and major-axis length, but
        # alternately occupy horizontal and vertical rectangles.
        candidate = (
            blob(x=30, y=38, w=20, h=8)
            if fid % 2 == 0
            else blob(x=36, y=32, w=8, h=20)
        )
        scanner.add_frame(fid, [candidate])
    assert scanner.excluded_count == 0
