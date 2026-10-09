# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Place raster blobs at first four-neighbour contact without occlusion."""

import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def try_place_adjacent(
    *,
    rng: random.Random,
    donor_mask: np.ndarray,
    occupied: np.ndarray,
    contact_mask: np.ndarray,
    max_tries: int,
) -> Optional[Tuple[int, int]]:
    """Approach the nearest contour from random non-overlapping positions.

    Distance fields are computed once in the local search region. Coarse moves
    stay strictly short of any possible contact; the final moves change one
    coordinate by one pixel, so first contact cannot be skipped. Other occupied
    blobs can block an approach, in which case a fresh starting point is tried.
    """
    H, W = occupied.shape
    dh, dw = donor_mask.shape
    if dw > W or dh > H or max_tries <= 0:
        return None
    donor = (donor_mask > 0).astype(np.uint8)
    target_points = cv2.findNonZero((contact_mask > 0).astype(np.uint8))
    if not cv2.countNonZero(donor) or target_points is None:
        return None
    tx, ty, tw, th = cv2.boundingRect(target_points)
    # One donor-sized margin gives starting points on every side of the group.
    x_low, x_high = max(0, tx - dw), min(W - dw, tx + tw)
    y_low, y_high = max(0, ty - dh), min(H - dh, ty + th)
    if x_low > x_high or y_low > y_high:
        return None

    region = occupied[y_low:y_high + dh, x_low:x_high + dw] > 0
    target = contact_mask[y_low:y_high + dh, x_low:x_high + dw] > 0
    distances, nearest = distance_transform_edt(~target, return_indices=True)
    clearance = distances if contact_mask is occupied else distance_transform_edt(~region)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    adjacent = cv2.dilate(region.astype(np.uint8), cross)
    target_adjacent = adjacent if contact_mask is occupied else cv2.dilate(target.astype(np.uint8), cross)
    boundary = donor & (1 - cv2.erode(donor, cross, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    by, bx = np.nonzero(boundary)

    for _ in range(int(max_tries)):
        x = rng.randint(x_low, x_high) - x_low
        y = rng.randint(y_low, y_high) - y_low
        if np.any(region[y:y + dh, x:x + dw] & donor):
            continue
        # A cardinal walk across the search region has a finite length.
        for _step in range(2 * (region.shape[0] + region.shape[1])):
            if np.any(adjacent[y:y + dh, x:x + dw] & donor):
                if np.any(target_adjacent[y:y + dh, x:x + dw] & donor):
                    # Validate the entire mask, including disconnected parts.
                    if not np.any(region[y:y + dh, x:x + dw] & donor):
                        return x + x_low, y + y_low
                break
            ys, xs = by + y, bx + x
            closest = int(np.argmin(distances[ys, xs]))
            py, px = int(ys[closest]), int(xs[closest])
            dy, dx = int(nearest[0, py, px]) - py, int(nearest[1, py, px]) - px
            gap = float(np.min(clearance[ys, xs]))
            # Rounding cannot exceed sqrt(2) times the largest axis step.
            coarse = max(0, int(math.floor((gap - 1.0) / math.sqrt(2.0))) - 1)
            if coarse:
                extent = max(abs(dx), abs(dy))
                step = min(coarse, extent)
                nx = x + int(round(dx * step / extent))
                ny = y + int(round(dy * step / extent))
            elif abs(dx) > abs(dy) or (abs(dx) == abs(dy) and rng.random() < 0.5):
                nx, ny = x + (1 if dx > 0 else -1), y
            else:
                nx, ny = x, y + (1 if dy > 0 else -1)
            if not (0 <= nx <= x_high - x_low and 0 <= ny <= y_high - y_low):
                break
            if np.any(region[ny:ny + dh, nx:nx + dw] & donor):
                break
            x, y = nx, ny
    return None
