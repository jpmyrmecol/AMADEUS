# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Track isolated foreground animals and estimate video-dependent parameters."""

import csv
import math
import os
import pickle
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import yaml
from batch_utils import tqdm
from path_utils import relativize_config_paths, resolve_config_paths
from scipy.optimize import linear_sum_assignment
from segmentation_metadata import segmentation_metadata_from_object
from tracking_constants import FIXED_INTERACT_IOU, MATCH_IOU_CANDIDATES
from video_frame_count import (
    clamp_frame_range_to_usable_count,
    read_video_frame_info,
    warn_if_frame_count_adjusted,
)

AABB = Tuple[float, float, float, float]
OBBRect = Tuple[Tuple[float, float], Tuple[float, float], float]

DEFAULT_LOCALIZED_RATIO = 0.9
DEFAULT_DIR_MIN_DISP = 1.0

# Internal gates for the inter-frame motion-measurement Hungarian assignment
# (auto-parameter estimation only). Not user-configurable.
_MOTION_MIN_AXIS_RATIO = 0.5
_MOTION_MAX_AXIS_RATIO = 2.0

# Upper bound on the number of (t-2, t-1, t) measurement windows used for
# MATCH_IOU/DIR_MIN_SEC auto-measurement. Not user-configurable -- see
# _select_measurement_windows.
_MAX_MOTION_MEASUREMENT_WINDOWS = 100

# Share of the measured pseudo-ground-truth matches that MATCH_IOU must still
# admit -- the standard 5% tolerance. See _select_match_iou.
_MIN_MATCH_RETENTION = 0.95

# Runtime defaults when a config omits the corresponding value.
FALLBACK_MATCH_IOU = 0.5
FALLBACK_DIR_MIN_SEC = 0.5

@dataclass
class BlobInfo:
    frame: int
    blob_index: int
    center: Tuple[float, float]
    area: float
    is_crossing: bool
    obb_major_axis_len: float
    obb_rect: OBBRect
    aabb: AABB


@dataclass
class TrackState:
    last_obb_rect: OBBRect
    last_frame: int


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return resolve_config_paths(cfg)


def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_blob_sequence(obj):
    return obj.blobs_in_video


def get_num_blob_frames(blob_seq) -> int:
    return len(blob_seq)


def get_frame_blobs(blob_seq, fid: int):
    return blob_seq[fid]


def resolve_num_workers(cfg: dict, default: int | str = "auto", cap: int | None = None) -> int:
    from batch_utils import auto_num_workers as _auto
    value = cfg.get("NUM_WORKERS", default)
    if value is None or str(value).strip().lower() in {"", "auto", "none"}:
        workers = _auto("process")
    else:
        workers = int(value)
    workers = max(1, workers)
    if cap is not None:
        workers = min(workers, max(1, int(cap)))
    return workers


def contour_center(cnt: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) > 1e-8:
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    pts = cnt.reshape(-1, 2).astype(np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def contour_area(cnt: np.ndarray) -> float:
    return float(abs(cv2.contourArea(cnt)))


def obb_rect_from_contour(cnt: np.ndarray) -> OBBRect:
    pts = np.asarray(cnt, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        cx, cy = contour_center(cnt)
        return ((float(cx), float(cy)), (0.0, 0.0), 0.0)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    return ((float(cx), float(cy)), (float(w), float(h)), float(angle))


def obb_major_axis_len_from_rect(rect: OBBRect) -> float:
    (_, _), (w, h), _ = rect
    return float(max(float(w), float(h)))


def obb_area_from_rect(rect: OBBRect) -> float:
    (_, _), (w, h), _ = rect
    return max(0.0, float(w)) * max(0.0, float(h))


def aabb_from_contour(cnt: np.ndarray) -> AABB:
    x, y, w, h = cv2.boundingRect(np.asarray(cnt, dtype=np.float32))
    return float(x), float(y), float(x + w), float(y + h)


def obb_iou(a: OBBRect, b: OBBRect) -> float:
    area_a = obb_area_from_rect(a)
    area_b = obb_area_from_rect(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    status, inter_pts = cv2.rotatedRectangleIntersection(a, b)
    if status == cv2.INTERSECT_NONE or inter_pts is None:
        return 0.0
    if status == cv2.INTERSECT_FULL:
        inter_area = min(area_a, area_b)
    else:
        pts = np.asarray(inter_pts, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            return 0.0
        inter_area = float(abs(cv2.contourArea(pts)))

    denom = area_a + area_b - inter_area
    return 0.0 if denom <= 0.0 else float(inter_area / denom)


def iqr_bounds(values: List[float], min_scale: float, max_scale: float) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return -float("inf"), float("inf")
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    return q1 - min_scale * iqr, q3 + max_scale * iqr


def inside_bounds(v: float, bounds: Tuple[float, float]) -> bool:
    lo, hi = bounds
    return bool(lo <= v <= hi)


def get_init_max_dist_ratio(cfg: dict) -> float:
    return float(cfg.get("INIT_MAX_DIST", 1.5))


def compute_individual_distance_stats(blob_records: Dict[int, List[BlobInfo]], cfg: dict) -> Dict[str, float]:
    # This body-length estimate is not used by the OBB IoU initial matcher,
    # but it normalizes the measured inter-frame displacement below.
    all_non_crossing = [b for items in blob_records.values() for b in items if not b.is_crossing]
    major_axis_lengths = [float(b.obb_major_axis_len) for b in all_non_crossing]
    axis_bounds = iqr_bounds(
        major_axis_lengths,
        float(cfg.get("SIZE_IQR_MULTIPLIER_MIN", 1.5)),
        float(cfg.get("SIZE_IQR_MULTIPLIER_MAX", 1.5)),
    )
    source = [
        float(b.obb_major_axis_len)
        for b in all_non_crossing
        if inside_bounds(float(b.obb_major_axis_len), axis_bounds)
    ]
    if not source:
        source = major_axis_lengths
    individual_distance_px = float(np.mean(np.asarray(source, dtype=np.float64))) if source else 0.0
    if individual_distance_px <= 0.0:
        raise RuntimeError("Failed to compute body length in initial_tracking.")
    init_max_dist_px = individual_distance_px * get_init_max_dist_ratio(cfg)
    return {
        "individual_distance_px": individual_distance_px,
        "individual_distance_source_count": len(source),
        "individual_distance_source_total_non_crossing": len(all_non_crossing),
        "obb_major_axis_lo": float(axis_bounds[0]),
        "obb_major_axis_hi": float(axis_bounds[1]),
        "init_max_dist_ratio": get_init_max_dist_ratio(cfg),
        "init_max_dist_px": init_max_dist_px,
    }


def _as_contour_array(value) -> np.ndarray:
    cnt = np.asarray(value, dtype=np.float32)
    if cnt.size == 0:
        return np.empty((0, 1, 2), dtype=np.float32)
    if cnt.ndim == 1:
        raise ValueError(f"Contour has invalid ndim=1 and shape={cnt.shape}")
    if cnt.ndim == 2:
        if cnt.shape[1] != 2:
            raise ValueError(f"Contour 2D array must have shape (N,2), got {cnt.shape}")
        return cnt.reshape(-1, 1, 2)
    if cnt.ndim == 3:
        if cnt.shape[1:] == (1, 2):
            return cnt
        if cnt.shape[0] == 1 and cnt.shape[2] == 2:
            return cnt.reshape(-1, 1, 2)
        raise ValueError(f"Contour 3D array must have shape (N,1,2), got {cnt.shape}")
    raise ValueError(f"Unsupported contour shape: {cnt.shape}")


def blob_contour(blob) -> np.ndarray:
    cnt = _as_contour_array(blob.contour)
    if cnt.size == 0:
        raise ValueError("Blob contour is empty.")
    return cnt


def _blob_info_from_blob(fid: int, bi: int, bl) -> BlobInfo | None:
    cnt = blob_contour(bl)
    if cnt.size == 0:
        return None
    obb_rect = obb_rect_from_contour(cnt)
    aabb = aabb_from_contour(cnt)
    return BlobInfo(
        frame=int(fid),
        blob_index=int(bi),
        center=contour_center(cnt),
        area=contour_area(cnt),
        is_crossing=bool(bl.is_outlier),
        obb_major_axis_len=obb_major_axis_len_from_rect(obb_rect),
        obb_rect=obb_rect,
        aabb=aabb,
    )


def _process_frame_blobs(args) -> Tuple[int, List[BlobInfo], List[str]]:
    fid, frame_blobs = args
    items: List[BlobInfo] = []
    errors: List[str] = []
    for bi, bl in enumerate(frame_blobs):
        try:
            info = _blob_info_from_blob(int(fid), int(bi), bl)
        except Exception as e:
            errors.append(f"[initial_tracking] skipped frame={fid} blob_index={bi}: {e}")
            continue
        if info is not None:
            items.append(info)
    return int(fid), items, errors


def build_blob_index(blob_seq, frame_indices: Sequence[int], num_workers: int = 1) -> Dict[int, List[BlobInfo]]:
    out: Dict[int, List[BlobInfo]] = {}
    tasks = [(int(fid), list(get_frame_blobs(blob_seq, int(fid)))) for fid in frame_indices]
    errors: List[str] = []

    if int(num_workers) <= 1:
        for task in tqdm(tasks, desc="Preparing blobs"):
            fid, items, err = _process_frame_blobs(task)
            out[fid] = items
            errors.extend(err)
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
            futures = [ex.submit(_process_frame_blobs, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Preparing blobs", unit="frame"):
                fid, items, err = fut.result()
                out[fid] = items
                errors.extend(err)

    for msg in errors:
        print(msg)
    if errors:
        print(f"[initial_tracking] skipped invalid blobs: {len(errors)}")
    return out


def validate_threshold(value, key: str, *, min_value: float = 0.0, max_value: float = 1.0) -> float:
    threshold = float(value)
    if threshold < min_value or threshold > max_value:
        raise ValueError(f"{key} must be between {min_value} and {max_value}, got {value!r}")
    return threshold


def candidate_score(
    blob: BlobInfo,
    track: TrackState,
    match_iou: float,
) -> Optional[float]:
    """Return the standard OBB IoU (intersection / union) between a blob and
    a track's last OBB, or None if it is below match_iou."""
    iou = obb_iou(blob.obb_rect, track.last_obb_rect)
    if iou < match_iou:
        return None
    return iou


def run_initial_tracking(
    blob_records: Dict[int, List[BlobInfo]],
    frame_indices: Sequence[int],
    match_iou: float,
    max_gap: int,
) -> List[Dict[str, object]]:
    next_track_id = 0
    active_tracks: Dict[int, TrackState] = {}
    rows: List[Dict[str, object]] = []

    match_iou = validate_threshold(
        match_iou,
        "analysis.MATCH_IOU",
        min_value=0.0,
        max_value=1.0,
    )

    for fid in tqdm(frame_indices, desc="Initial tracking by OBB IoU"):
        active_tracks = {
            tid: tr for tid, tr in active_tracks.items()
            if int(fid) - int(tr.last_frame) <= int(max_gap)
        }

        blobs = [b for b in blob_records.get(int(fid), []) if not b.is_crossing]
        if not blobs:
            continue

        candidate_pairs: List[Tuple[float, int, int]] = []
        track_ids = list(active_tracks.keys())
        for bi, blob in enumerate(blobs):
            for tid in track_ids:
                iou = candidate_score(blob, active_tracks[tid], match_iou)
                if iou is None:
                    continue
                candidate_pairs.append((float(iou), bi, tid))

        candidate_pairs.sort(key=lambda x: x[0], reverse=True)
        used_blob_idx = set()
        used_track_ids = set()

        for iou, bi, tid in candidate_pairs:
            if bi in used_blob_idx or tid in used_track_ids:
                continue
            blob = blobs[bi]
            rows.append({
                "frame": int(blob.frame),
                "blob_index": int(blob.blob_index),
                "track_id": int(tid),
                "center_x": float(blob.center[0]),
                "center_y": float(blob.center[1]),
                "area": float(blob.area),
                "match_obb_iou": f"{float(iou):.6f}",
            })
            active_tracks[tid] = TrackState(
                last_obb_rect=blob.obb_rect,
                last_frame=int(fid),
            )
            used_blob_idx.add(bi)
            used_track_ids.add(tid)

        for bi, blob in enumerate(blobs):
            if bi in used_blob_idx:
                continue
            tid = next_track_id
            next_track_id += 1
            rows.append({
                "frame": int(blob.frame),
                "blob_index": int(blob.blob_index),
                "track_id": int(tid),
                "center_x": float(blob.center[0]),
                "center_y": float(blob.center[1]),
                "area": float(blob.area),
                "match_obb_iou": "",
            })
            active_tracks[tid] = TrackState(
                last_obb_rect=blob.obb_rect,
                last_frame=int(fid),
            )

    return rows


def _derive_num_crops(frame_width: int, frame_height: int, image_size: int) -> int:
    """Mirror the GUI's non-localized tile-count calculation."""
    image_size = max(32, int(image_size))
    short_side = min(int(frame_width), int(frame_height))
    long_side = max(int(frame_width), int(frame_height))
    tiles_long = max(1, long_side // image_size)
    tiles_short = max(1, short_side // image_size)
    tile_count = max(1, min(4, tiles_long * tiles_short))
    return max(1, int(math.ceil(tile_count / 2)))


def _derive_cluster_frames(cfg: dict, num_crops: int) -> int:
    num_images = max(0, int(cfg.get("NUM_IMAGES", 10000)))
    clustered_ratio = max(0.0, float(cfg.get("CLUSTERED_RATIO", 0.05)))
    images_per_frame = (
        (int(num_crops) if bool(cfg.get("USE_CROP", True)) else 0)
        + (1 if bool(cfg.get("USE_FULL", True)) else 0)
    )
    return max(1, int(math.ceil(num_images * clustered_ratio / max(1, images_per_frame))))


def compute_localized_stats(
    blob_records: Dict[int, List[BlobInfo]],
    frame_indices: Sequence[int],
    image_size: int,
    localized_ratio_threshold: float,
) -> Dict[str, object]:
    """Judge LOCALIZED per frame: does the group's union bbox fit inside one
    image_size x image_size training crop?

    Outlier blobs are excluded (blob.is_crossing covers both merged/crossing
    animals and undersized noise, since segmentation's area-outlier gate
    flags anything outside the IQR bounds in either direction). A frame with
    no remaining blobs is excluded from the ratio.
    """
    image_size = int(image_size)

    valid_frame_count = 0
    fitting_frame_count = 0
    for fid in frame_indices:
        blobs = [b for b in blob_records.get(int(fid), []) if not b.is_crossing]
        if not blobs:
            continue
        valid_frame_count += 1
        x1 = min(blob.aabb[0] for blob in blobs)
        y1 = min(blob.aabb[1] for blob in blobs)
        x2 = max(blob.aabb[2] for blob in blobs)
        y2 = max(blob.aabb[3] for blob in blobs)
        group_width = x2 - x1
        group_height = y2 - y1
        fits_crop = group_width <= image_size and group_height <= image_size
        if fits_crop:
            fitting_frame_count += 1

    localized_fit_ratio = (
        float(fitting_frame_count) / float(valid_frame_count)
        if valid_frame_count > 0
        else 0.0
    )
    localized = bool(valid_frame_count > 0 and localized_fit_ratio >= localized_ratio_threshold)

    return {
        "localized_valid_frames": int(valid_frame_count),
        "localized_fitting_frames": int(fitting_frame_count),
        "localized_fit_ratio": float(localized_fit_ratio),
        "LOCALIZED": localized,
    }


def _hungarian_motion_matches(
    prev_blobs: Sequence[BlobInfo],
    curr_blobs: Sequence[BlobInfo],
    max_distance_px: float,
) -> List[Tuple[int, int, float]]:
    """Return the existing distance-Hungarian correspondences for one frame pair."""
    if not prev_blobs or not curr_blobs:
        return []

    cost = np.full((len(prev_blobs), len(curr_blobs)), np.inf, dtype=np.float64)
    for i, pb in enumerate(prev_blobs):
        for j, cb in enumerate(curr_blobs):
            distance_px = math.hypot(
                float(cb.center[0]) - float(pb.center[0]),
                float(cb.center[1]) - float(pb.center[1]),
            )
            if distance_px > max_distance_px:
                continue
            major_prev = float(pb.obb_major_axis_len)
            major_curr = float(cb.obb_major_axis_len)
            if major_prev <= 0.0 or major_curr <= 0.0:
                continue
            axis_ratio = major_curr / major_prev
            if axis_ratio < _MOTION_MIN_AXIS_RATIO or axis_ratio > _MOTION_MAX_AXIS_RATIO:
                continue
            cost[i, j] = distance_px

    valid = np.isfinite(cost)
    if not valid.any():
        return []

    work = np.where(valid, cost, 1e12)
    row_idx, col_idx = linear_sum_assignment(work)
    return [
        (int(r), int(c), float(cost[r, c]))
        for r, c in zip(row_idx, col_idx)
        if valid[r, c]
    ]


def _select_measurement_windows(
    frame_indices: Sequence[int], max_windows: int = _MAX_MOTION_MEASUREMENT_WINDOWS,
) -> List[int]:
    """Return up to max_windows starting frame ids (the "t-2" of each
    window) for evenly-spaced, deterministic (t-2, t-1, t) measurement
    windows spanning the full frame range, instead of every consecutive
    frame pair in the video. Each window contributes the two consecutive
    frame pairs (t-2, t-1) and (t-1, t).

    A window's three frames must all be present and numerically consecutive
    (t-1 = t-2+1, t = t-2+2).
    Returns every valid window when there are max_windows or fewer;
    otherwise selects max_windows evenly spaced ones by position in the
    valid-window list (no randomness), so a long video measures a bounded,
    representative sample instead of scanning every frame.
    """
    frame_set = {int(f) for f in frame_indices}
    valid_starts = sorted(
        fid for fid in frame_set
        if (fid + 1) in frame_set and (fid + 2) in frame_set
    )
    if len(valid_starts) <= max_windows:
        return valid_starts
    positions = np.linspace(0, len(valid_starts) - 1, num=max_windows)
    return sorted({valid_starts[int(round(p))] for p in positions})


def _measure_hungarian_samples(
    blob_records: Dict[int, List[BlobInfo]],
    frame_indices: Sequence[int],
    individual_distance_px: float,
    max_distance_ratio: float,
) -> Tuple[List[float], List[float]]:
    """Measure inter-frame displacement ratios and the OBB IoU of each
    pseudo-ground-truth match via a Hungarian assignment over non-crossing
    blobs in consecutive frames.

    Only frames touched by _select_measurement_windows's bounded, evenly
    spaced (t-2, t-1, t) windows are scanned -- not every frame in
    frame_indices -- so this stays O(measurement windows) instead of
    O(video length). Deduplicating those frames into sorted_frames means
    two windows that happen to be adjacent share their common pair's
    Hungarian match, with no pair recomputed and no sample double-counted.

    This correspondence is dedicated to automatic-parameter measurement: it is
    never written to track_assignments.csv and never used as an initial-track ID.
    """
    displacement_ratios: List[float] = []
    match_ious: List[float] = []

    if individual_distance_px <= 0.0:
        return displacement_ratios, match_ious

    max_distance_px = float(individual_distance_px) * float(max_distance_ratio)
    touched_frames: Set[int] = set()
    for window_start in _select_measurement_windows(frame_indices):
        touched_frames.update((window_start, window_start + 1, window_start + 2))
    sorted_frames = sorted(touched_frames)

    for prev_fid, curr_fid in zip(sorted_frames[:-1], sorted_frames[1:]):
        if curr_fid != prev_fid + 1:
            continue
        prev_blobs = [b for b in blob_records.get(prev_fid, []) if not b.is_crossing]
        curr_blobs = [b for b in blob_records.get(curr_fid, []) if not b.is_crossing]

        for r, c, distance_px in _hungarian_motion_matches(prev_blobs, curr_blobs, max_distance_px):
            displacement_ratios.append(distance_px / float(individual_distance_px))
            match_ious.append(float(obb_iou(prev_blobs[r].obb_rect, curr_blobs[c].obb_rect)))

    return displacement_ratios, match_ious


def _match_iou_retention(match_ious: Sequence[float]) -> Dict[float, float]:
    """{threshold: share of the pseudo-ground-truth matches it still admits}.

    An empty sample set retains every candidate vacuously (1.0), so
    _select_match_iou stays deterministic without a separate fallback branch.
    """
    total = len(match_ious)
    return {
        threshold: (
            float(sum(1 for iou in match_ious if float(iou) >= threshold) / total)
            if total > 0
            else 1.0
        )
        for threshold in MATCH_IOU_CANDIDATES
    }


def _select_match_iou(match_ious: Sequence[float]) -> float:
    """Select the strictest (highest) candidate gate that still admits at least
    _MIN_MATCH_RETENTION of the measured pseudo-ground-truth matches.

    Competing blobs practically never reach even the loosest candidate, so the
    binding constraint is how many genuine consecutive-frame matches a gate
    throws away: raising the gate only ever costs recall. Slow, compact
    subjects keep a high IoU frame to frame and can afford the strictest gate;
    fast or elongated subjects (whose OBBs barely overlap between frames)
    cannot, and fall back to a looser one.

    MATCH_IOU_CANDIDATES is ascending, so the last candidate that clears the
    floor is the strictest one. When none clears it, the loosest candidate is
    the closest the gate can get to preserving those matches.
    """
    retention = _match_iou_retention(match_ious)
    best_threshold = MATCH_IOU_CANDIDATES[0]
    for threshold in MATCH_IOU_CANDIDATES:
        if retention[threshold] >= _MIN_MATCH_RETENTION:
            best_threshold = threshold
    return float(best_threshold)


def _derive_dir_min_sec(disp_ratio_p90: float, fps: float, dir_min_disp: float) -> float:
    """Seconds needed to cover DIR_MIN_DISP body lengths at the observed speed."""
    if not np.isfinite(disp_ratio_p90) or disp_ratio_p90 < 0.0:
        raise ValueError(f"disp_ratio_p90 must be a finite non-negative value, got {disp_ratio_p90!r}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be a finite positive value, got {fps!r}")
    if not np.isfinite(dir_min_disp) or dir_min_disp <= 0.0:
        raise ValueError(f"DIR_MIN_DISP must be a finite positive value, got {dir_min_disp!r}")

    return float(
        np.clip(
            round((dir_min_disp / max(disp_ratio_p90, 1e-3)) / fps, 2),
            0.20,
            0.50,
        )
    )


def compute_auto_params(
    blob_records: Dict[int, List[BlobInfo]],
    cfg: dict,
    frame_shape: Sequence[int],
    fps: float,
) -> dict:
    """Measure the segmented frames and return automatic config values plus statistics."""
    if len(frame_shape) < 2:
        raise ValueError(f"frame_shape must contain height and width, got {frame_shape!r}")
    frame_height = int(frame_shape[0])
    frame_width = int(frame_shape[1])
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Invalid frame shape: {frame_shape!r}")

    non_crossing = [
        blob
        for frame_blobs in blob_records.values()
        for blob in frame_blobs
        if not blob.is_crossing
    ]
    if not non_crossing:
        raise RuntimeError("No non-crossing blobs were found for automatic parameter measurement.")

    frame_indices = sorted(int(f) for f in blob_records.keys())

    distance_stats = compute_individual_distance_stats(blob_records, cfg)
    individual_distance_px = float(distance_stats["individual_distance_px"])

    localized_ratio_threshold = validate_threshold(
        cfg.get("LOCALIZED_RATIO", DEFAULT_LOCALIZED_RATIO),
        "LOCALIZED_RATIO",
    )
    localized_stats = compute_localized_stats(
        blob_records,
        frame_indices,
        int(cfg.get("TRAIN_IMG_SIZE", 640)),
        localized_ratio_threshold,
    )
    localized = bool(localized_stats["LOCALIZED"])

    displacement_ratios, match_ious = _measure_hungarian_samples(
        blob_records,
        frame_indices,
        individual_distance_px,
        get_init_max_dist_ratio(cfg),
    )
    motion_sample_count = len(displacement_ratios)
    match_iou = _select_match_iou(match_ious)
    match_iou_retention = _match_iou_retention(match_ious)

    if motion_sample_count == 0:
        disp_ratio_p90 = 0.0
        dir_min_sec = FALLBACK_DIR_MIN_SEC
        motion_fallback = "insufficient_frame_pairs"
    else:
        disp_ratio_p90 = float(np.percentile(np.asarray(displacement_ratios, dtype=np.float64), 90))
        dir_min_sec = _derive_dir_min_sec(
            disp_ratio_p90,
            float(fps),
            float(cfg.get("DIR_MIN_DISP", DEFAULT_DIR_MIN_DISP)),
        )
        motion_fallback = ""
    interact_iou = FIXED_INTERACT_IOU

    num_crops = (
        1
        if localized
        else _derive_num_crops(frame_width, frame_height, int(cfg.get("TRAIN_IMG_SIZE", 640)))
    )
    result = dict(distance_stats)
    result.update(localized_stats)
    result.update({
        "NUM_CROPS": int(num_crops),
        "FREE_SCALE": 1.0 if localized else 0.25,
        "MATCH_IOU": match_iou,
        "INTERACT_IOU": interact_iou,
        "DIR_MIN_SEC": dir_min_sec,
        "CLUSTER_FRAMES": _derive_cluster_frames(cfg, num_crops),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "localized": localized,
        "motion_sample_count": int(motion_sample_count),
        "disp_ratio_p90": float(disp_ratio_p90),
        "match_iou_min_retention": _MIN_MATCH_RETENTION,
        "auto_match_iou": match_iou,
        "auto_interact_iou": interact_iou,
        "auto_dir_min_sec": dir_min_sec,
        "motion_fallback": motion_fallback,
        "source_fps": float(fps),
    })
    for threshold, retention in match_iou_retention.items():
        result[f"match_iou_retention_{str(threshold).replace('.', '_')}"] = float(retention)
    return result


def _apply_auto_params(cfg: dict, auto_params: dict) -> None:
    """Apply the six measured values plus the fixed interaction threshold."""
    for key in ("LOCALIZED", "NUM_CROPS", "FREE_SCALE", "DIR_MIN_SEC", "CLUSTER_FRAMES"):
        cfg[key] = auto_params[key]

    analysis_cfg = cfg.setdefault("analysis", {})
    embedding_cfg = cfg.setdefault("EMBEDDING", {})
    if not isinstance(analysis_cfg, dict):
        raise ValueError("analysis must be a mapping.")
    if not isinstance(embedding_cfg, dict):
        raise ValueError("EMBEDDING must be a mapping.")
    analysis_cfg["MATCH_IOU"] = auto_params["MATCH_IOU"]
    embedding_cfg["INTERACT_IOU"] = FIXED_INTERACT_IOU


def _apply_fixed_interact_iou(cfg: dict) -> bool:
    embedding_cfg = cfg.setdefault("EMBEDDING", {})
    if not isinstance(embedding_cfg, dict):
        raise ValueError("EMBEDDING must be a mapping.")
    changed = embedding_cfg.get("INTERACT_IOU") != FIXED_INTERACT_IOU
    embedding_cfg["INTERACT_IOU"] = FIXED_INTERACT_IOU
    return changed


def _dump_config_atomic(path: str, cfg: dict) -> None:
    target_path = os.path.abspath(path)
    target_dir = os.path.dirname(target_path) or os.getcwd()
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target_path)}.",
        suffix=".tmp",
        dir=target_dir,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
        os.replace(temp_path, target_path)
    except Exception:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise


def _persist_auto_params_if_enabled(
    cfg_path: str,
    cfg: dict,
    auto_params: dict,
) -> bool:
    """Persist auto values when enabled and always normalize INTERACT_IOU."""
    auto_enabled = bool(cfg.get("AUTO_PARAMS", True))
    fixed_value_changed = _apply_fixed_interact_iou(cfg)
    if auto_enabled:
        _apply_auto_params(cfg, auto_params)
    if not auto_enabled and not fixed_value_changed:
        return False
    # cfg is kept absolute in memory for the rest of this process; only the
    # on-disk copy stores paths relative to SESSION_PATH.
    _dump_config_atomic(cfg_path, relativize_config_paths(cfg))
    return auto_enabled

def save_assignments(rows: List[Dict[str, object]], out_dir: str, tracking_stats: Dict[str, object]) -> Tuple[str, str]:
    csv_path = os.path.join(out_dir, "track_assignments.csv")
    fieldnames = [
        "frame", "blob_index", "track_id", "center_x", "center_y", "area",
        "match_obb_iou",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats_path = os.path.join(out_dir, "tracking_stats.csv")
    unique_track_ids = sorted({int(r["track_id"]) for r in rows})
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["assigned_rows", len(rows)])
        writer.writerow(["unique_tracks", len(unique_track_ids)])
        writer.writerow(["first_track_id", unique_track_ids[0] if unique_track_ids else ""])
        writer.writerow(["last_track_id", unique_track_ids[-1] if unique_track_ids else ""])
        for key in [
            "individual_distance_px",
            "individual_distance_source_count",
            "individual_distance_source_total_non_crossing",
            "obb_major_axis_lo",
            "obb_major_axis_hi",
            "init_max_dist_ratio",
            "init_max_dist_px",
            "initial_tracking_match_iou",
            "init_max_gap",
            "initial_tracking_num_workers",
            "frame_width",
            "frame_height",
            "localized",
            "localized_valid_frames",
            "localized_fitting_frames",
            "localized_fit_ratio",
            "motion_sample_count",
            "disp_ratio_p90",
            "match_iou_min_retention",
            "match_iou_retention_0_3",
            "match_iou_retention_0_4",
            "match_iou_retention_0_5",
            "auto_match_iou",
            "auto_interact_iou",
            "auto_dir_min_sec",
            "motion_fallback",
            "source_fps",
        ]:
            if key in tracking_stats:
                value = tracking_stats[key]
                if isinstance(value, float):
                    value = f"{value:.6f}"
                writer.writerow([key, value])
    return csv_path, stats_path


def _prepare_context(cfg_path: str, cfg: dict) -> dict:
    """Load the pickle, resolve the frame range, build the blob index, and
    measure + (if AUTO_PARAMS) apply the automatic config values.

    Shared by the full measure-then-track run and the parameter-only
    adjustment mode: compute_auto_params() does not depend on initial-track
    IDs, so this always runs before any OBB IoU association.
    """
    session_path = str(cfg["SESSION_PATH"])
    pickle_path = str(cfg["PICKLE_PATH"])
    out_dir = os.path.join(session_path, "initial_tracking")

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"PICKLE_PATH not found: {pickle_path}")

    ensure_dirs([out_dir])

    print(f"[initial_tracking] config: {cfg_path}")
    print(f"[initial_tracking] pickle: {pickle_path}")
    print(f"[initial_tracking] output: {out_dir}")

    pickle_obj = load_pickle(pickle_path)
    seg_meta = segmentation_metadata_from_object(pickle_obj)
    blob_seq = resolve_blob_sequence(pickle_obj)

    training_cfg = cfg.get("training", {}) or {}
    first_frame = int(training_cfg.get("FIRST_FRAME", 0))
    last_frame = int(training_cfg.get("LAST_FRAME", -1))
    pickle_frame_count = get_num_blob_frames(blob_seq)
    video_path = str(cfg["TRAINING_VIDEO_PATH"])
    if pickle_frame_count <= 0:
        raise RuntimeError("No blob frames were found in the pickle.")
    frame_info = read_video_frame_info(video_path)
    warn_if_frame_count_adjusted(frame_info, label="initial_tracking training video")
    max_frames = min(pickle_frame_count, frame_info.usable_frame_count)
    if max_frames <= 0:
        raise RuntimeError("No usable frames were found for initial tracking.")
    first_frame, last_frame = clamp_frame_range_to_usable_count(
        first_frame, last_frame, max_frames,
    )
    frame_indices = list(range(first_frame, last_frame + 1))
    suffix = (
        f", pickle_frames={pickle_frame_count}, "
        f"usable_video_frames={frame_info.usable_frame_count}"
    )
    print(f"[initial_tracking] frame range: {first_frame}..{last_frame} ({len(frame_indices)} frames{suffix})")
    if seg_meta:
        print(
            "[initial_tracking] segmentation metadata: "
            f"training {seg_meta['training_frame_start']}..{seg_meta['training_frame_end']}, "
            f"interval={seg_meta['training_frame_interval']}"
        )

    num_workers = resolve_num_workers(cfg)
    blob_records = build_blob_index(blob_seq, frame_indices, num_workers=num_workers)

    auto_params = compute_auto_params(
        blob_records,
        cfg,
        (frame_info.height, frame_info.width),
        frame_info.fps,
    )
    auto_enabled = bool(cfg.get("AUTO_PARAMS", True))
    print(f"[initial_tracking] AUTO_PARAMS={auto_enabled}")
    print("[initial_tracking] LOCALIZED judgement:")
    print(
        f"[initial_tracking]   fitting frames / valid frames = "
        f"{auto_params['localized_fitting_frames']}/{auto_params['localized_valid_frames']}"
    )
    print(f"[initial_tracking]   localized_fit_ratio={auto_params['localized_fit_ratio']:.3f}")
    print(f"[initial_tracking]   LOCALIZED={auto_params['LOCALIZED']}")
    print("[initial_tracking] motion measurement:")
    print(f"[initial_tracking]   motion_sample_count={auto_params['motion_sample_count']}")
    print(f"[initial_tracking]   disp_ratio_p90={auto_params['disp_ratio_p90']:.3f}")
    print(f"[initial_tracking]   match_iou_min_retention={_MIN_MATCH_RETENTION:.2f}")
    for threshold in MATCH_IOU_CANDIDATES:
        suffix = str(threshold).replace('.', '_')
        print(
            f"[initial_tracking]   t={threshold:.1f}: "
            f"retention={auto_params[f'match_iou_retention_{suffix}']:.6f}"
        )
    if auto_params.get("motion_fallback"):
        print(f"[initial_tracking]   motion_fallback={auto_params['motion_fallback']}")
    print(f"[initial_tracking]   auto MATCH_IOU={auto_params['MATCH_IOU']:.1f}")
    print(f"[initial_tracking]   fixed INTERACT_IOU={auto_params['INTERACT_IOU']:.1f}")
    print(f"[initial_tracking]   auto DIR_MIN_SEC={auto_params['DIR_MIN_SEC']:.2f}")

    if _persist_auto_params_if_enabled(cfg_path, cfg, auto_params):
        print(f"[initial_tracking]   applied to config (AUTO_PARAMS=True)")
    else:
        print(
            "[initial_tracking]   AUTO_PARAMS=False; measured auto values were not applied; "
            f"INTERACT_IOU remains fixed at {FIXED_INTERACT_IOU:.1f}"
        )

    analysis_cfg = cfg.get("analysis", {}) or {}
    match_iou = float(analysis_cfg.get("MATCH_IOU", FALLBACK_MATCH_IOU))
    print(f"[initial_tracking] initial tracking MATCH_IOU={match_iou:.1f}")

    return {
        "out_dir": out_dir,
        "frame_indices": frame_indices,
        "blob_records": blob_records,
        "num_workers": num_workers,
        "match_iou": match_iou,
        "auto_params": auto_params,
    }


def _run_main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python initial_tracking.py config.yaml")

    print("Initial Tracking Process...")

    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    ctx = _prepare_context(cfg_path, cfg)

    max_gap = int(cfg.get("INIT_MAX_GAP", 1))
    print(f"[initial_tracking] max gap: {max_gap}")
    print(f"[initial_tracking] prep workers: {ctx['num_workers']}")

    rows = run_initial_tracking(
        blob_records=ctx["blob_records"],
        frame_indices=ctx["frame_indices"],
        match_iou=ctx["match_iou"],
        max_gap=max_gap,
    )

    tracking_stats = dict(ctx["auto_params"])
    tracking_stats.update({
        "initial_tracking_match_iou": float(ctx["match_iou"]),
        "init_max_gap": int(max_gap),
        "initial_tracking_num_workers": int(ctx["num_workers"]),
    })
    csv_path, stats_path = save_assignments(rows, ctx["out_dir"], tracking_stats)

    print(f"Saved initial tracking to: {csv_path}")
    print(f"Saved tracking stats to: {stats_path}")


def _run_adjust_only() -> None:
    """Measure and (if AUTO_PARAMS) apply the auto values without
    running the OBB IoU association or writing track_assignments.csv.

    Lets the GUI show the freshly-measured config (LOCALIZED, MATCH_IOU, ...)
    before the user commits to running the full pipeline. The full run later
    measures again (compute_auto_params is deterministic given the same
    segmentation output), so this is purely a fast preview step, not a
    required prerequisite.
    """
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python initial_tracking.py config.yaml --adjust-only")

    print("Initial Tracking: Adjust Parameters Only...")

    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    _prepare_context(cfg_path, cfg)
    print("[initial_tracking] adjust-only: done (initial tracking association was not run)")


def main() -> None:
    try:
        if "--adjust-only" in sys.argv[2:]:
            _run_adjust_only()
        else:
            _run_main()
    except Exception:
        cfg_path = sys.argv[1] if len(sys.argv) >= 2 else None
        out_dir = None
        if cfg_path and os.path.exists(cfg_path):
            try:
                cfg = load_config(cfg_path)
                session_path = str(cfg.get("SESSION_PATH", ""))
                if session_path:
                    out_dir = os.path.join(session_path, "initial_tracking")
                    os.makedirs(out_dir, exist_ok=True)
            except Exception:
                out_dir = None

        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if out_dir:
            err_path = os.path.join(out_dir, "initial_tracking_error.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(tb)
            print(f"[initial_tracking] wrote traceback to: {err_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
