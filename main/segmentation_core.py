# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Pure, dependency-light foreground-mask segmentation helpers.

This intentionally duplicates (rather than imports) the small pixel-math
functions gui/gui_segmentation.py keeps privately (_segment_cpu,
_build_single_roi_mask, _build_roi_mask_for_frame, _expand_mask): that
module imports
cv2/numpy lazily (see its _ensure_video_modules()) to keep GUI startup
fast, so importing from it here at module load time would reintroduce an
eager cv2 import into the GUI's import chain. These functions are small,
stable, and pixel-level only, so keeping a second copy here for the
tracking pipeline is lower-risk than coupling the two modules' import
timing together.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegConfig:
    """Immutable snapshot of the segmentation parameters needed to
    regenerate a foreground mask from a raw BGR frame."""
    mode: str
    ksize: int
    threshold: int
    dark_threshold: int
    bright_threshold: int
    diff_threshold: int
    open_iter: int
    close_iter: int
    fill_holes: bool
    invert_mask: bool
    expand_px: int
    expand_merge_only: bool
    background_bgr: "np.ndarray | None"


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background; all image borders remain connected to outside.

    Flooding from the unpadded (0, 0) fails when that pixel is foreground,
    or when foreground splits the exterior into multiple border components.
    """
    flood = np.pad(mask, 1, mode="constant")
    cv2.floodFill(flood, None, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood[1:-1, 1:-1]))


def segment_cpu(frame_bgr: np.ndarray, cfg: SegConfig) -> np.ndarray:
    """CPU threshold/diff pipeline. Mirrors gui_segmentation._segment_cpu."""
    mode, ksize = cfg.mode, cfg.ksize
    if mode in {"dark_region", "bright_region"}:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if ksize > 1:
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        ttype = cv2.THRESH_BINARY_INV if mode == "dark_region" else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, cfg.threshold, 255, ttype)
    elif mode == "background_diff":
        diff_gray = cv2.cvtColor(cv2.absdiff(frame_bgr, cfg.background_bgr), cv2.COLOR_BGR2GRAY)
        if ksize > 1:
            diff_gray = cv2.GaussianBlur(diff_gray, (ksize, ksize), 0)
        _, mask = cv2.threshold(diff_gray, cfg.threshold, 255, cv2.THRESH_BINARY)
    else:  # hybrid modes
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.cvtColor(cv2.absdiff(frame_bgr, cfg.background_bgr), cv2.COLOR_BGR2GRAY)
        if ksize > 1:
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
            diff_gray = cv2.GaussianBlur(diff_gray, (ksize, ksize), 0)
        ttype = cv2.THRESH_BINARY_INV if "dark" in mode else cv2.THRESH_BINARY
        tval = cfg.dark_threshold if "dark" in mode else cfg.bright_threshold
        _, imask = cv2.threshold(gray, tval, 255, ttype)
        _, dmask = cv2.threshold(diff_gray, cfg.diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(imask, dmask)
    return mask


# Discrete-grid allowance for the merge-only corridors below: a corridor is the
# exact shortest connection between two regions, and one pixel of slack is what
# keeps that set connected once it is rasterized.
BRIDGE_TOLERANCE_PX = 1.0


# Merge-only corridors below are the exact shortest connection between two
# regions; one pixel of slack is what keeps that set connected once it is
# rasterized. Distances there use DIST_MASK_5 rather than DIST_MASK_PRECISE:
# the precise transform is not bit-reproducible across identical calls, and a
# segmentation run has to give the same mask every time.
BRIDGE_TOLERANCE_PX = 1.0
BRIDGE_DIST_MASK = 5


def disk_kernel(radius: int) -> np.ndarray:
    """Structuring element of every pixel within `radius` of the centre."""
    offsets = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    return (np.hypot(dy, dx) <= radius).astype(np.uint8)


def smooth_single_animal_contour(contour: np.ndarray, level: int) -> np.ndarray:
    """Remove thin protrusions with a thickness-relative disk opening.

    Level 1..20 selects 4..80% of the maximum inscribed radius, rounded to
    pixels. Opening restores the eroded body rather than replacing it with
    a fitted ellipse. Work on the original contour on every call, not the
    previous preview, so changing the slider is reversible.

    Reduce the radius if opening would split the animal, create a hole, or
    remove over a third of its pixels. Never add foreground pixels.
    """
    level = max(1, min(20, int(level)))
    x, y, w, h = cv2.boundingRect(contour)
    original = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.drawContours(original, [contour], -1, 255, cv2.FILLED, offset=(1 - x, 1 - y))
    thickness = float(cv2.distanceTransform(original, cv2.DIST_L2, 5).max())
    radius = min(int(thickness) - 1, max(1, int(round(thickness * level * 0.04))))
    original_pixels = cv2.countNonZero(original)
    for r in range(radius, 0, -1):
        opened = cv2.morphologyEx(
            original, cv2.MORPH_OPEN, disk_kernel(r),
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        opened = cv2.bitwise_and(opened, original)
        if cv2.countNonZero(opened) < original_pixels * (2.0 / 3.0):
            continue
        contours, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1 or cv2.contourArea(contours[0]) <= 0:
            continue
        return contours[0] + np.array([[[x - 1, y - 1]]], dtype=np.int32)
    return contour


def solid_contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Represent a mask as filled contours without accidentally filling holes.

    Segmentation pickles store filled contours, not contour hierarchies.
    If subtracting smoothed-away pixels leaves an outlier with a hole, split
    that residual at a row through the hole. The pieces keep exactly the
    remaining foreground; none of the removed pixels can reappear on export.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    holes = [i for i, entry in enumerate(hierarchy[0]) if entry[3] >= 0]
    if not holes:
        return list(contours)
    _, y, _, h = cv2.boundingRect(contours[holes[0]])
    split = max(1, min(mask.shape[0] - 1, y + h // 2))
    top = solid_contours_from_mask(mask[:split].copy())
    bottom = solid_contours_from_mask(mask[split:].copy())
    offset = np.array([[[0, split]]], dtype=np.int32)
    return top + [cnt + offset for cnt in bottom]


def expand_mask(mask: np.ndarray, expand_px: int, merge_only: bool) -> np.ndarray:
    """Mirrors gui_segmentation._expand_mask.

    Grows every blob outline outward by expand_px pixels (Euclidean).

    With merge_only, only the expansion that actually joins separate regions
    is kept: inside each group of regions that the expansion merges, the
    shortest corridor is filled in for every edge of a minimum spanning tree
    over the gaps between them.  The group therefore ends up as one region
    exactly as it would under the full expansion, while isolated blobs keep
    their original outline -- the intended use is closing notches and
    unintended gaps without inflating every blob.
    """
    expand_px = int(expand_px)
    if expand_px <= 0 or cv2.countNonZero(mask) == 0:
        return mask
    expanded = cv2.dilate(mask, disk_kernel(expand_px))
    if not merge_only:
        return expanded
    return keep_merging_expansion(mask, expanded)


def keep_merging_expansion(mask: np.ndarray, expanded: np.ndarray) -> np.ndarray:
    n_src, src_labels = cv2.connectedComponents(mask, connectivity=8)
    if n_src <= 2:  # background plus at most one region: nothing can merge
        return mask
    _n_dst, dst_labels, dst_stats, _centroids = cv2.connectedComponentsWithStats(expanded, connectivity=8)
    foreground = mask > 0
    pairs = np.unique(np.stack([dst_labels[foreground], src_labels[foreground]], axis=1), axis=0)
    groups: "dict[int, list[int]]" = {}
    for dst_label, src_label in pairs:
        groups.setdefault(int(dst_label), []).append(int(src_label))

    out = mask.copy()
    for dst_label, group in groups.items():
        if len(group) < 2:
            continue
        x0 = int(dst_stats[dst_label, cv2.CC_STAT_LEFT])
        y0 = int(dst_stats[dst_label, cv2.CC_STAT_TOP])
        x1 = x0 + int(dst_stats[dst_label, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(dst_stats[dst_label, cv2.CC_STAT_HEIGHT])
        inside = dst_labels[y0:y1, x0:x1] == dst_label
        src_roi = src_labels[y0:y1, x0:x1]
        dists = [
            cv2.distanceTransform(
                np.where(src_roi == src_label, 0, 255).astype(np.uint8),
                cv2.DIST_L2, BRIDGE_DIST_MASK,
            )
            for src_label in group
        ]
        out[y0:y1, x0:x1][merge_corridors(inside, src_roi, group, dists)] = 255
    return out


def merge_corridors(inside, src_roi, group, dists) -> np.ndarray:
    """Corridors joining every region of one merged group.

    Edges come from a minimum spanning tree over the gaps between the regions,
    so the group is connected with the shortest corridors and nothing more.
    Corridor pixels that reach no region are dropped: they would otherwise
    show up as spurious extra blobs.
    """
    size = len(group)
    gaps = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i + 1, size):
            gaps[i, j] = gaps[j, i] = float(np.where(inside, dists[i] + dists[j], np.inf).min())

    bridge = np.zeros(inside.shape, dtype=bool)
    joined = [0]
    remaining = set(range(1, size))
    while remaining:
        i, j, gap = min(
            ((a, b, gaps[a, b]) for a in joined for b in sorted(remaining)),
            key=lambda edge: edge[2],
        )
        bridge |= inside & (dists[i] + dists[j] <= gap + BRIDGE_TOLERANCE_PX)
        joined.append(j)
        remaining.discard(j)

    sources = np.isin(src_roi, group)
    _n_labels, labels = cv2.connectedComponents((sources | bridge).astype(np.uint8), connectivity=8)
    return bridge & np.isin(labels, np.unique(labels[sources]))


def build_single_roi_mask(image_shape: tuple, roi: dict) -> "np.ndarray | None":
    """Mirrors gui_segmentation._build_single_roi_mask."""
    img_h, img_w = image_shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    shape = roi.get("shape", "circle")
    if shape == "circle":
        cx = max(0, min(img_w - 1, int(roi.get("x", 0))))
        cy = max(0, min(img_h - 1, int(roi.get("y", 0))))
        radius = max(0, int(roi.get("w", 0)))
        if radius <= 0:
            return None
        cv2.circle(mask, (cx, cy), radius, 255, thickness=cv2.FILLED)
    else:
        x = max(0, min(img_w, int(roi.get("x", 0))))
        y = max(0, min(img_h, int(roi.get("y", 0))))
        w = max(0, int(roi.get("w", 0)))
        h_val = max(0, int(roi.get("h", 0)))
        x2 = max(x, min(img_w, x + w))
        y2 = max(y, min(img_h, y + h_val))
        if x2 <= x or y2 <= y:
            return None
        mask[y:y2, x:x2] = 255
    return mask


def build_static_roi_mask(roi_sets: list, image_shape: tuple) -> "np.ndarray | None":
    """Intersection mask of all enabled ROI sets, ignoring frame_start/frame_end.

    Unlike gui_segmentation.build_roi_mask_for_frame, this is used to reapply
    an arena-boundary ROI to a *different* video than the one segmentation
    was configured on -- the original frame_start/frame_end gating refers to
    that video's own frame numbering and has no meaning here, so every
    enabled ROI is treated as always active.
    """
    combined: "np.ndarray | None" = None
    for roi in roi_sets:
        if not roi.get("enabled", False):
            continue
        partial = build_single_roi_mask(image_shape, roi)
        if partial is None:
            continue
        if bool(roi.get("reverse", False)):
            partial = cv2.bitwise_not(partial)
        combined = partial if combined is None else cv2.bitwise_and(combined, partial)
    return combined


def compute_foreground_mask(
    frame_bgr: np.ndarray, cfg: SegConfig, roi_mask: "np.ndarray | None" = None
) -> np.ndarray:
    """segment_cpu + invert + roi-AND + open/close morphology + fill_holes + expansion.

    Mirrors the pre-contour-extraction portion of
    gui_segmentation._segment_with_config's CPU path, without the
    outlier-mask union or contour/blob extraction -- callers here only need
    the binary foreground mask itself.
    """
    mask = segment_cpu(frame_bgr, cfg)

    if cfg.invert_mask:
        mask = cv2.bitwise_not(mask)
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    if cfg.open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg.open_iter)
    if cfg.close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.close_iter)

    if cfg.fill_holes:
        mask = fill_binary_holes(mask)
        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, roi_mask)

    if cfg.expand_px > 0:
        mask = expand_mask(mask, cfg.expand_px, cfg.expand_merge_only)
        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, roi_mask)

    return mask
