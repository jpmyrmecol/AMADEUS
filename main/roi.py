# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Polygon ROI settings and masks shared by preview and processing.

Image libraries are imported only when a mask is requested. Importing this
module during GUI startup does not load OpenCV, NumPy, or the segmentation core.
"""

from __future__ import annotations

import math

def default_roi_set() -> dict:
    return {"enabled": False, "points": [], "frame_start": 0, "frame_end": -1}


def initial_roi_points(image_shape: tuple) -> list[list[int]]:
    """A centered rectangle spanning half the image width and height."""
    height, width = image_shape[:2]
    x0, x1 = round((width - 1) * 0.25), round((width - 1) * 0.75)
    y0, y1 = round((height - 1) * 0.25), round((height - 1) * 0.75)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def insert_longest_edge_midpoint(points: list[list[int]]) -> list[list[int]] | None:
    """Insert a vertex on the longest polygon edge, including the closing edge.

    Equal-length edges are resolved by their order in the polygon. Return None
    if pixel rounding cannot produce a distinct new vertex.
    """
    if len(points) < 3:
        raise ValueError("A polygon requires at least three vertices.")

    edge_index = max(
        range(len(points)),
        key=lambda i: (
            (points[(i + 1) % len(points)][0] - points[i][0]) ** 2
            + (points[(i + 1) % len(points)][1] - points[i][1]) ** 2
        ),
    )
    start, end = points[edge_index], points[(edge_index + 1) % len(points)]
    midpoint = [round((start[0] + end[0]) / 2), round((start[1] + end[1]) / 2)]
    if midpoint in points:
        return None
    result = [list(vertex) for vertex in points]
    result.insert(edge_index + 1, midpoint)
    return result


def insert_roi_vertex_at(points: list[list[int]], point: tuple[int, int]) -> list[list[int]] | None:
    """Insert at the click location after the closest polygon edge.

    Edge distance is measured to the finite segment rather than to the
    infinite line so corners and concave polygons behave consistently.
    """
    if len(points) < 3:
        raise ValueError("A polygon requires at least three vertices.")
    vertex = [int(point[0]), int(point[1])]
    if vertex in points:
        return None

    def distance_sq(i: int) -> float:
        ax, ay = points[i]
        bx, by = points[(i + 1) % len(points)]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq:
            t = ((vertex[0] - ax) * dx + (vertex[1] - ay) * dy) / length_sq
            t = max(0.0, min(1.0, t))
        else:
            t = 0.0
        return (vertex[0] - ax - t * dx) ** 2 + (vertex[1] - ay - t * dy) ** 2

    edge_index = min(range(len(points)), key=distance_sq)
    result = [list(existing) for existing in points]
    result.insert(edge_index + 1, vertex)
    return result


def normalize_roi_set(roi: dict, image_shape: tuple | None = None) -> dict:
    points = roi.get("points", [])
    if not isinstance(points, (list, tuple)) or (points and len(points) < 3):
        raise ValueError("Each ROI must have at least three polygon vertices, or no vertices yet.")
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


def load_roi_settings(settings: dict, image_shape: tuple | None = None) -> tuple[list, bool]:
    """Read the current polygon ROI settings."""
    raw_sets = settings.get("roi_sets", [])
    if not isinstance(raw_sets, list) or any(not isinstance(roi, dict) for roi in raw_sets):
        raise ValueError("settings.roi_sets must be a list of ROI objects.")
    roi_sets = [normalize_roi_set(roi, image_shape) for roi in raw_sets]
    return roi_sets or [default_roi_set()], bool(settings.get("roi_reverse", False))


def roi_is_active(roi: dict, frame_idx: int | None) -> bool:
    if not roi.get("enabled", False):
        return False
    if frame_idx is None:
        return True
    start, end = int(roi.get("frame_start", 0)), int(roi.get("frame_end", -1))
    return frame_idx >= start and (end < 0 or frame_idx <= end)


def roi_signature(roi_sets: list, reverse: bool) -> tuple:
    return (
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


def contour_mask_region(contour, image_shape: tuple):
    """Filled contour pixels in their bounding box, clipped to the image."""
    import cv2
    import numpy as np

    x, y, w, h = cv2.boundingRect(contour)
    H, W = image_shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
    if x0 >= x1 or y0 >= y1:
        return 0, 0, np.zeros((0, 0), dtype=np.uint8)
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED, offset=(-x0, -y0))
    return x0, y0, mask


def convex_polygon_mask_region(poly, image_shape: tuple):
    """Rasterize a globally rounded polygon before cropping it to any blob.

    Drawing directly into a smaller blob window changes OpenCV's edge clipping
    and can change the intersection by a few pixels.
    """
    import cv2
    import numpy as np

    H, W = image_shape[:2]
    x0 = max(0, math.floor(float(poly[:, 0].min())))
    y0 = max(0, math.floor(float(poly[:, 1].min())))
    x1 = min(W, math.ceil(float(poly[:, 0].max())) + 1)
    y1 = min(H, math.ceil(float(poly[:, 1].max())) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0, 0, np.zeros((0, 0), dtype=np.uint8)
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    vertices = np.round(poly).astype(np.int32) - np.array((x0, y0), dtype=np.int32)
    cv2.fillConvexPoly(mask, vertices, 255)
    return x0, y0, mask


def mask_region_overlap(first, second) -> int:
    """Count shared pixels by intersecting two already rasterized regions."""
    import cv2

    ax, ay, a = first
    bx, by, b = second
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax+a.shape[1], bx+b.shape[1]), min(ay+a.shape[0], by+b.shape[0])
    if x0 >= x1 or y0 >= y1:
        return 0
    return cv2.countNonZero(a[y0-ay:y1-ay, x0-ax:x1-ax] & b[y0-by:y1-by, x0-bx:x1-bx])
