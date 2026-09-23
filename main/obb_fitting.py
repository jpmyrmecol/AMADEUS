# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Fit oriented rectangles to segmentation contours with selectable orientation."""

import math

import cv2
import numpy as np


DEFAULT_OBB_FIT_MODE = "min_area"
OBB_FIT_MODES = (DEFAULT_OBB_FIT_MODE, "pca")


def normalize_obb_fit_mode(value: object = DEFAULT_OBB_FIT_MODE) -> str:
    """Return a supported OBB fit mode, defaulting only for an omitted value."""
    mode = DEFAULT_OBB_FIT_MODE if value is None or not str(value).strip() else str(value).strip().lower()
    if mode not in OBB_FIT_MODES:
        raise ValueError(f"OBB_FIT_MODE must be one of {OBB_FIT_MODES}, got {value!r}")
    return mode


def _rect_long_axis(rect) -> np.ndarray:
    (_, _), (width, height), angle = rect
    axis_angle = float(angle) if float(width) >= float(height) else float(angle) + 90.0
    radians = math.radians(axis_angle)
    return np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float32)


def fit_obb(points: np.ndarray, mode: object = DEFAULT_OBB_FIT_MODE):
    """Return (OpenCV RotatedRect, fitted long/PCA axis) for a point cloud.

    The min_area mode delegates directly to OpenCV to retain existing geometry.
    The pca mode constrains the rectangle orientation to the point cloud's first
    principal component, then fits its bounds along that axis and its
    perpendicular. Nearly isotropic contours have no stable PCA direction and
    use the established min-area result.
    """
    fit_mode = normalize_obb_fit_mode(mode)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError("At least three points are required to fit an OBB.")
    if not np.isfinite(pts).all():
        raise ValueError("OBB points must contain only finite coordinates.")

    if fit_mode == "min_area":
        rect = cv2.minAreaRect(pts)
        return rect, _rect_long_axis(rect)

    points64 = pts.astype(np.float64)
    centroid = points64.mean(axis=0)
    centered = points64 - centroid
    covariance = centered.T @ centered / float(len(points64))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_variance = float(eigenvalues[-1])
    minor_variance = float(eigenvalues[-2])
    if (
        not math.isfinite(major_variance)
        or major_variance <= 1e-12
        or major_variance - minor_variance <= max(major_variance, 1e-12) * 1e-6
    ):
        rect = cv2.minAreaRect(pts)
        return rect, _rect_long_axis(rect)

    axis = eigenvectors[:, -1]
    if axis[0] < 0.0 or (abs(float(axis[0])) <= 1e-12 and axis[1] < 0.0):
        axis = -axis
    perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    along = centered @ axis
    across = centered @ perpendicular
    along_min, along_max = float(along.min()), float(along.max())
    across_min, across_max = float(across.min()), float(across.max())

    width = along_max - along_min
    height = across_max - across_min
    rect_center = (
        centroid
        + axis * ((along_min + along_max) / 2.0)
        + perpendicular * ((across_min + across_max) / 2.0)
    )
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    rect = (
        (float(rect_center[0]), float(rect_center[1])),
        (float(width), float(height)),
        float(angle),
    )
    return rect, axis.astype(np.float32)


def fit_obb_points(points: np.ndarray, mode: object = DEFAULT_OBB_FIT_MODE) -> np.ndarray:
    """Return the four fitted rectangle corners as float32 pixel coordinates."""
    rect, _ = fit_obb(points, mode)
    return cv2.boxPoints(rect).astype(np.float32)
