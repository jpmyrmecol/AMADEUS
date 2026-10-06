# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Polygon ROI settings and masks shared by preview and processing.

Image libraries are imported only when a mask is requested. Importing this
module during GUI startup does not load OpenCV, NumPy, or the segmentation core.
"""

from __future__ import annotations

import math

ROI_SCHEMA = "polygon-union-v1"


class RoiMigrationError(ValueError):
    """A legacy ROI configuration cannot retain its meaning after migration."""


def default_roi_set() -> dict:
    return {"enabled": False, "points": [], "frame_start": 0, "frame_end": -1}


def _fractional_roi_points(image_shape: tuple, low: float, high: float) -> list[list[int]]:
    height, width = image_shape[:2]
    x0, x1 = round((width - 1) * low), round((width - 1) * high)
    y0, y1 = round((height - 1) * low), round((height - 1) * high)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def initial_roi_points(image_shape: tuple) -> list[list[int]]:
    """A centered rectangle spanning half the image width and height."""
    return _fractional_roi_points(image_shape, 0.25, 0.75)


def _refresh_saved_inactive_default_roi(
    roi_sets: list[dict], reverse: bool, image_shape: tuple | None
) -> None:
    """Replace only obsolete, untouched default ROI geometry.

    Older GUI versions eagerly saved their generated default rectangle even
    while ROI use was disabled. Reopening such a session would therefore
    restore the obsolete geometry instead of using the current initial ROI.
    Preserve every enabled, ranged, reversed, multi-set, or custom polygon.
    """
    if image_shape is None or reverse or len(roi_sets) != 1:
        return
    roi = roi_sets[0]
    if (
        roi["enabled"]
        or roi["frame_start"] != 0
        or roi["frame_end"] != -1
        or not roi["points"]
    ):
        return

    height, width = image_shape[:2]
    full_frame = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    previous_default = _fractional_roi_points(image_shape, 0.10, 0.90)
    if roi["points"] in (full_frame, previous_default):
        roi["points"] = initial_roi_points(image_shape)


def normalize_roi_set(roi: dict, image_shape: tuple | None = None) -> dict:
    points = roi.get("points", [])
    if not isinstance(points, (list, tuple)) or len(points) not in (0, 4):
        raise ValueError("Each ROI must have four polygon vertices, or no vertices yet.")
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each ROI vertex must contain x and y.")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("ROI vertex coordinates must be finite.")
        x, y = round(x), round(y)
        if image_shape is not None:
            height, width = image_shape[:2]
            x, y = max(0, min(width - 1, x)), max(0, min(height - 1, y))
        normalized.append([x, y])
    return {
        "enabled": bool(roi.get("enabled", False)),
        "points": normalized,
        "frame_start": int(roi.get("frame_start", 0)),
        "frame_end": int(roi.get("frame_end", -1)),
    }


def load_roi_settings(settings: dict, image_shape: tuple | None = None) -> tuple[list, bool, list[str]]:
    """Read polygon settings or explicitly migrate the former per-set schema.

    Rectangles retain their inclusive raster boundaries. Circles become their
    bounding quadrilaterals. Returned notices describe that approximation and
    the change from a legacy intersection to a union.
    """
    raw_sets = settings.get("roi_sets", [])
    if not isinstance(raw_sets, list) or any(not isinstance(roi, dict) for roi in raw_sets):
        raise ValueError("settings.roi_sets must be a list of ROI objects.")
    schema = settings.get("roi_schema")
    if schema is not None and schema != ROI_SCHEMA:
        raise ValueError(f"Unsupported ROI schema: {schema!r}")
    if schema == ROI_SCHEMA:
        normalized = [normalize_roi_set(roi, image_shape) for roi in raw_sets] or [default_roi_set()]
        reverse = bool(settings.get("roi_reverse", False))
        _refresh_saved_inactive_default_roi(normalized, reverse, image_shape)
        return normalized, reverse, []

    reverse_values = {bool(roi.get("reverse", False)) for roi in raw_sets}
    if len(reverse_values) > 1:
        raise RoiMigrationError(
            "Legacy ROI sets contain different Reverse ROI values. Their intersection "
            "cannot be represented by a polygon union and one global Reverse ROI. "
            "This configuration has not been converted. Open it with the previous "
            "AMADEUS version and make all Reverse ROI values equal before loading it here."
        )
    reverse = next(iter(reverse_values), bool(settings.get("roi_reverse", False)))
    converted, notices = [], []
    has_circle = False
    for roi in raw_sets:
        if "points" in roi:
            converted.append(normalize_roi_set(roi, image_shape))
            continue
        shape = roi.get("shape", "circle")
        x, y, w, h = (int(roi.get(key, 0)) for key in ("x", "y", "w", "h"))
        points = []
        if shape == "circle" and w > 0:
            if image_shape is not None:
                height, width = image_shape[:2]
                x, y = max(0, min(width - 1, x)), max(0, min(height - 1, y))
            points = [[x - w, y - w], [x + w, y - w], [x + w, y + w], [x - w, y + w]]
            has_circle = True
        elif shape == "rectangle" and w > 0 and h > 0:
            if image_shape is not None:
                height, width = image_shape[:2]
                x, y = max(0, min(width, x)), max(0, min(height, y))
                w, h = min(w, width - x), min(h, height - y)
            if w > 0 and h > 0:
                points = [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]
        elif shape not in {"circle", "rectangle"}:
            raise ValueError(f"Unsupported legacy ROI shape: {shape!r}")
        converted.append(normalize_roi_set({**roi, "points": points}, image_shape))
    if has_circle:
        notices.append("Circular ROIs will become bounding rectangles with four vertices. Their masks will change.")
    if not reverse and sum(bool(roi.get("enabled", False)) for roi in raw_sets) > 1:
        notices.append("Multiple enabled ROIs will be combined by union. The previous configuration used intersection.")
    return converted or [default_roi_set()], reverse, notices


def roi_is_active(roi: dict, frame_idx: int | None) -> bool:
    if not roi.get("enabled", False):
        return False
    if frame_idx is None:
        return True
    start, end = int(roi.get("frame_start", 0)), int(roi.get("frame_end", -1))
    return frame_idx >= start and (end < 0 or frame_idx <= end)


def roi_signature(roi_sets: list, reverse: bool) -> tuple:
    return (
        ROI_SCHEMA,
        bool(reverse),
        tuple((bool(roi["enabled"]), tuple(tuple(point) for point in roi["points"]),
               int(roi["frame_start"]), int(roi["frame_end"])) for roi in roi_sets),
    )


def build_single_roi_mask(image_shape: tuple, roi: dict):
    import cv2
    import numpy as np

    points = roi.get("points", [])
    if not points:
        return None
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    vertices = np.asarray(points, dtype=np.int32)
    cv2.fillPoly(mask, [vertices], 255)
    return mask


def build_roi_mask_for_frame(roi_sets: list, image_shape: tuple, frame_idx: int | None, reverse: bool = False):
    import cv2

    combined = None
    for roi in roi_sets:
        if not roi_is_active(roi, frame_idx):
            continue
        partial = build_single_roi_mask(image_shape, roi)
        if partial is not None:
            combined = partial if combined is None else cv2.bitwise_or(combined, partial)
    if combined is not None and reverse:
        combined = cv2.bitwise_not(combined)
    return combined
