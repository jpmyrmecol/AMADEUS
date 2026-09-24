# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Match detections across frames and export per-animal tracking tables."""

import os
import sys
import copy
import cv2
import re
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from batch_utils import auto_num_workers as _auto_num_workers
import numpy as np
import pandas as pd
from typing import Optional
from filterpy.kalman import KalmanFilter
from tracking_artifacts import (
    BUFFER_DIR_NAME,
    LOG_DIR_NAME,
    PROVENANCE_CSV_NAME,
    artifact_filename as _artifact_filename,
    artifact_path as _artifact_path,
)
from scipy.optimize import linear_sum_assignment

from obb_detection import (
    angle_to_unit_vec,
    build_tracking_out_dir,
    build_tracking_run_name,
    ensure_clockwise,
    iou_obb,
    load_blob_pickle,
    load_config,
    normalize_direction_vec,
    obb_center,
    obb_polygon_area,
    parse_weight_spec,
    positive_long_axis_direction,
    resolve_model_name,
    tqdm_it,
    unit_vec_to_angle_deg,
    wrap_angle_deg,
)
from checkpoint_utils import normalize_checkpoint_weight
from experiment_utils import (
    experiment_dir_name_from_cfg,
    resolve_existing_experiment_dir_name,
)
from weight_utils import deduplicate_best_epoch_weights
from video_frame_count import (
    clamp_frame_range_to_usable_count,
    read_video_frame_info,
    warn_if_frame_count_adjusted,
)
from assign_types import (
    ATYPE_BKFILL_DET,
    ATYPE_DIR_FLIP,
    ATYPE_DIR_NAN,
    ATYPE_GAP_DET,
    ATYPE_INIT,
    ATYPE_KF_FILL,
    ATYPE_MISSING,
    ATYPE_OVERLAP_FILL,
    ATYPE_POST_FILL,
    ATYPE_PRE_FILL,
    ATYPE_S1,
    ATYPE_S2,
    ATYPE_S3,
    ATYPE_S4,
    ATYPE_TURN_ACCEPT,
    ATYPE_VEL_DIST,
    ATYPE_VITERBI,
)

OBB_SOURCE_MISSING = 0.0
OBB_SOURCE_DETECTED = 1.0
OBB_SOURCE_TRACK_INTERPOLATED = 3.0
OBB_SOURCE_PRE_CORRECTION = 4.0
OBB_SOURCE_POST_CORRECTION = 5.0


def solve_assignment_by_cost(cost: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    cost = np.asarray(cost, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if cost.ndim != 2 or valid.shape != cost.shape:
        raise ValueError('cost/valid shape mismatch')
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []
    big = 1e12
    work = np.where(valid, cost, big)
    row_ind, col_ind = linear_sum_assignment(work)
    pairs = []
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if r < n_rows and c < n_cols and valid[r, c] and work[r, c] < big:
            pairs.append((int(r), int(c)))
    return pairs

# Tracking stages whose direction output is unreliable (both groups write direction=NaN;
# the post-pass Viterbi DP resolves the correct sign from OBB long-axis candidates).
# Group 1 - IoU overlap guaranteed (same object, high confidence).
#   Direction KF is updated with nearest-axis to last_valid_direction to keep
#   rotation tracking fresh; no risk of polluting KF with a 180-degree flip.
UNRELIABLE_DIR_OVERLAP: frozenset[int] = frozenset({
    ATYPE_S2,
    ATYPE_S4,
})
# Group 2 - Distance-only match (IoU=0; misassignment risk).
#   Direction KF is frozen to avoid contaminating the filter with a wrong axis.
UNRELIABLE_DIR_DISTANCE: frozenset[int] = frozenset({
    ATYPE_VEL_DIST,
    ATYPE_GAP_DET,
})
UNRELIABLE_DIR_ATYPES: frozenset[int] = UNRELIABLE_DIR_OVERLAP | UNRELIABLE_DIR_DISTANCE

# Main detection stages that can serve as a trusted anchor before a one-frame
# S5/fast-distance excursion.  If a S5 OBB follows one of these anchors and is
# immediately followed by a missing frame, the S5 OBB is treated as an
# unconfirmed excursion and retroactively excluded.
FAST_DIST_TRUSTED_ANCHOR_ATYPES: frozenset[int] = frozenset({
    ATYPE_INIT,
    ATYPE_S1, ATYPE_S2, ATYPE_S3, ATYPE_S4,
})

PROVENANCE_BUFFER_NAMES = (
    'obb_source_buf',
    'obb_corrected_buf',
    'direction_corrected_buf',
    'switch_corrected_buf',
)
ASSIGN_TYPE_PICKLE_NAME = 'assign_type.pkl'
ASSIGN_TYPE_HISTORY_PICKLE_NAME = 'assign_type_history.pkl'
BUFFERS_PICKLE_NAME = 'buffers.pkl'
CORRECTIONS_LOG_NAME = 'corrections_log.json'
CORRECTIONS_SUMMARY_NAME = 'corrections_summary.csv'
FWD_ARTIFACT_SUFFIX = ''
BACKCORR_ARTIFACT_SUFFIX = 'corrected'
INITIAL_CONFIDENCE_THRESHOLD = 0.5



def resolve_num_workers(
    cfg: dict,
    section_name: Optional[str] = None,
    default: Optional[int] = None,
    cap: Optional[int] = None,
) -> Optional[int]:
    """Resolve worker count from config (section -> top-level -> default).

    Keys checked: NUM_WORKERS, WORKERS, N_WORKERS, workers, num_workers.
    Returns None when value is 'auto'/'none'/empty and default is None.
    """
    keys = ("NUM_WORKERS", "WORKERS", "N_WORKERS", "workers", "num_workers")
    value = None
    if section_name:
        section = cfg.get(section_name, {}) or {}
        if isinstance(section, dict):
            for key in keys:
                if key in section:
                    value = section[key]
                    break
    if value is None:
        for key in keys:
            if key in cfg:
                value = cfg[key]
                break
    if value is None or str(value).strip().lower() in {"", "auto", "none"}:
        if default is None:
            return None
        workers = int(default)
    else:
        workers = int(value)
    workers = max(1, workers)
    if cap is not None:
        workers = min(workers, max(1, int(cap)))
    return workers

def _worker_budget(total_workers: int, concurrent_jobs: int) -> tuple[int, int]:
    """Return (outer_processes, workers_per_job) without oversubscribing."""
    jobs = max(1, int(concurrent_jobs))
    total = max(1, int(total_workers))
    outer = min(total, jobs)
    inner = max(1, total // outer)
    return outer, inner

def create_constant_velocity_kf_2d(
    x0: float,
    y0: float,
    *,
    dt: float = 1.0,
    measurement_noise: float = 0.01,
    process_noise: tuple[float, float, float, float] = (0.01, 0.01, 0.05, 0.05),
) -> KalmanFilter:
    """Create a 2D constant-velocity Kalman filter (state: x, y, vx, vy)."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    kf.x = np.array([float(x0), float(y0), 0.0, 0.0], dtype=float)
    kf.P = np.eye(4, dtype=float) * 10.0
    kf.R = np.eye(2, dtype=float) * float(measurement_noise)
    kf.Q = np.diag(process_noise).astype(float)
    return kf

def create_position_kf(cx: float, cy: float, dt: float = 1.0) -> KalmanFilter:
    """Constant-velocity Kalman filter for OBB center tracking (state: cx, cy, vx, vy)."""
    return create_constant_velocity_kf_2d(cx, cy, dt=dt)

def create_bbox_shape_kf(w: float, h: float, dt: float = 1.0) -> KalmanFilter:
    """Constant-velocity Kalman filter for OBB width/height tracking (state: w, h, vw, vh)."""
    return create_constant_velocity_kf_2d(max(float(w), 1.0), max(float(h), 1.0), dt=dt)

def create_direction_kf(direction_vec: np.ndarray, dt: float = 1.0) -> KalmanFilter:
    """Constant-velocity Kalman filter for OBB direction unit-vector tracking (state: dx, dy, vdx, vdy)."""
    v = normalize_direction_vec(direction_vec)
    return create_constant_velocity_kf_2d(
        float(v[0]),
        float(v[1]),
        dt=dt,
        measurement_noise=0.005,
        process_noise=(0.005, 0.005, 0.02, 0.02),
    )

def reference_to_unit_vec(reference: float | int | np.ndarray | None) -> np.ndarray | None:
    if reference is None:
        return None
    if isinstance(reference, np.ndarray):
        return normalize_direction_vec(reference)
    if isinstance(reference, (tuple, list)) and len(reference) == 2:
        return normalize_direction_vec(np.asarray(reference, dtype=float))
    if isinstance(reference, (int, float, np.integer, np.floating)) and np.isfinite(reference):
        return angle_to_unit_vec(float(reference))
    return None

def angular_distance_deg(a: float | int | None, b: float | int | None) -> float:
    if a is None or b is None:
        return 0.0
    da = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return float(abs(da))

def axis_angular_distance_deg_from_obbs(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Heading-independent long-axis angle difference in [0, 90] degrees."""
    va = positive_long_axis_direction(pts_a)
    vb = positive_long_axis_direction(pts_b)
    cosv = abs(float(np.dot(va, vb)))
    cosv = max(-1.0, min(1.0, cosv))
    return float(math.degrees(math.acos(cosv)))

def closest_long_axis_direction(pts: np.ndarray, reference: float | int | np.ndarray | None) -> np.ndarray:
    major_axis = positive_long_axis_direction(pts)
    ref_vec = reference_to_unit_vec(reference)
    if ref_vec is None:
        return major_axis
    return major_axis if float(np.dot(major_axis, ref_vec)) >= 0.0 else -major_axis

def ensure_clockwise_batch(obbs: np.ndarray) -> np.ndarray:
    """Vectorized ensure_clockwise for (N, 4, 2) array. Avoids per-detection Python loop."""
    obbs = np.asarray(obbs, dtype=np.float32).reshape(-1, 4, 2)
    n = len(obbs)
    if n == 0:
        return obbs
    centers = obbs.mean(axis=1, keepdims=True)
    diff = obbs - centers
    angles = np.arctan2(diff[:, :, 1], diff[:, :, 0])
    order = np.argsort(angles, axis=1)
    sorted_pts = obbs[np.arange(n)[:, None], order]
    x, y = sorted_pts[:, :, 0], sorted_pts[:, :, 1]
    signed_areas = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    flip_mask = signed_areas > 0
    if flip_mask.any():
        sorted_pts[flip_mask] = sorted_pts[flip_mask, ::-1, :]
    yx_key = sorted_pts[:, :, 1] * 1e6 + sorted_pts[:, :, 0]
    start_idx = np.argmin(yx_key, axis=1)
    idx4 = (np.arange(4)[None, :] + start_idx[:, None]) % 4
    return sorted_pts[np.arange(n)[:, None], idx4].astype(np.float32)

def align_cyclic_to_reference(pts: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ref = np.asarray(reference, dtype=np.float32).reshape(4, 2)
    best = None
    best_cost = float('inf')
    for shift in range(4):
        cand = np.roll(pts, -shift, axis=0)
        cost = float(np.sum((cand - ref) ** 2))
        if cost < best_cost:
            best_cost = cost
            best = cand
    return best.astype(np.float32)

def canonicalize_obb_points(pts: np.ndarray, reference: Optional[np.ndarray]=None) -> np.ndarray:
    pts = ensure_clockwise(pts)
    if reference is not None:
        ref = ensure_clockwise(reference)
        pts = align_cyclic_to_reference(pts, ref)
    return pts.astype(np.float32)

def wrap_periodic_near(angle_deg: float, reference_deg: float, period_deg: float) -> float:
    angle_deg = float(angle_deg)
    reference_deg = float(reference_deg)
    period_deg = float(period_deg)
    return angle_deg + period_deg * round((reference_deg - angle_deg) / period_deg)

def normalize_long_side_rect(rect: np.ndarray, reference_angle: Optional[float]=None) -> np.ndarray:
    cx, cy, w, h, angle = map(float, rect)
    w = max(w, 0.001)
    h = max(h, 0.001)
    if h > w:
        w, h = (h, w)
        angle += 90.0
    if reference_angle is not None and np.isfinite(reference_angle):
        angle = wrap_periodic_near(angle, float(reference_angle), 180.0)
    else:
        angle = (angle + 90.0) % 180.0 - 90.0
    return np.array([cx, cy, w, h, angle], dtype=float)

def obb_points_to_rect(pts: np.ndarray, reference_angle: Optional[float]=None) -> np.ndarray:
    pts = canonicalize_obb_points(pts)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    rect_arr = np.array([float(cx), float(cy), float(w), float(h), float(angle)], dtype=float)
    return normalize_long_side_rect(rect_arr, reference_angle=reference_angle)

def polygon_area_batch(obbs: np.ndarray) -> np.ndarray:
    """Shoelace formula for (N, 4, 2) arrays - replaces per-OBB cv2.contourArea loops."""
    x, y = obbs[:, :, 0], obbs[:, :, 1]
    return 0.5 * np.abs((x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1))

def obb_aabbs_batch(obbs: np.ndarray) -> np.ndarray:
    """Return (N, 4) array of [x1, y1, x2, y2] axis-aligned bounding boxes for (N, 4, 2) OBBs."""
    return np.stack([
        obbs[:, :, 0].min(axis=1),
        obbs[:, :, 1].min(axis=1),
        obbs[:, :, 0].max(axis=1),
        obbs[:, :, 1].max(axis=1),
    ], axis=1)

def obb_aabb(pts: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    )

def aabbs_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

def valid_obb(pts) -> bool:
    if pts is None:
        return False
    arr = np.asarray(pts, dtype=float)
    return arr.shape == (4, 2) and np.isfinite(arr).all() and (obb_polygon_area(arr) > 0.0)

def obb_to_tuple(pts: np.ndarray) -> tuple:
    pts = np.asarray(pts, dtype=float).reshape(4, 2)
    return tuple(pts[:, 0].tolist() + pts[:, 1].tolist())

def tuple_to_obb(vals) -> np.ndarray | None:
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=float)
    if arr.size != 8 or np.isnan(arr).any():
        return None
    return ensure_clockwise(np.stack([arr[:4], arr[4:]], axis=1))

def direction_vec_from_kf(direction_kf: KalmanFilter | None, fallback: np.ndarray | None=None) -> np.ndarray:
    if direction_kf is None:
        return normalize_direction_vec(np.array([1.0, 0.0], dtype=float), fallback=fallback)
    return normalize_direction_vec(direction_kf.x[:2], fallback=fallback)

def predict_obb_from_filters(prev_obb: np.ndarray, position_kf: KalmanFilter | None, bbox_shape_kf: KalmanFilter | None, direction_kf: KalmanFilter | None) -> np.ndarray:
    if not valid_obb(prev_obb):
        return None
    center_prev = obb_center(prev_obb)
    rect_prev = obb_points_to_rect(prev_obb)
    dir_prev = closest_long_axis_direction(prev_obb, None)
    if position_kf is None:
        center = center_prev.astype(float)
    else:
        center = np.asarray(position_kf.x[:2], dtype=float)
    if bbox_shape_kf is None:
        w_pred = max(float(rect_prev[2]), 1.0)
        h_pred = max(float(rect_prev[3]), 1.0)
    else:
        w_pred = max(float(bbox_shape_kf.x[0]), 1.0)
        h_pred = max(float(bbox_shape_kf.x[1]), 1.0)
    kf_vec = direction_vec_from_kf(direction_kf, fallback=dir_prev)
    if is_long_axis_compatible(prev_obb, kf_vec, max_axis_error_deg=67.5):
        dir_vec = closest_long_axis_direction(prev_obb, kf_vec)
    else:
        dir_vec = dir_prev
    minor_vec = np.array([-dir_vec[1], dir_vec[0]], dtype=float)
    half_w = 0.5 * float(w_pred)
    half_h = 0.5 * float(h_pred)
    pts = np.vstack([center - dir_vec * half_w - minor_vec * half_h, center + dir_vec * half_w - minor_vec * half_h, center + dir_vec * half_w + minor_vec * half_h, center - dir_vec * half_w + minor_vec * half_h]).astype(np.float32)
    return ensure_clockwise(pts)

def build_detection_frames(df: pd.DataFrame, first_frame: int, last_frame: int, score_threshold: float) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Convert a detection DataFrame to a per-frame dict of (obbs, scores, directions) arrays."""
    frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    def empty_frame():
        return (
            np.zeros((0, 4, 2), dtype=np.float32),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
        )

    if df is None or df.empty:
        for f in range(int(first_frame), int(last_frame) + 1):
            frames[f] = empty_frame()
        return frames

    work = df.copy()
    work = work[(work['frame'] >= int(first_frame)) & (work['frame'] <= int(last_frame))]
    work['score'] = pd.to_numeric(work['score'], errors='coerce')
    work = work[np.isfinite(work['score']) & (work['score'] >= float(score_threshold))].copy()
    if work.empty:
        for f in range(int(first_frame), int(last_frame) + 1):
            frames[f] = empty_frame()
        return frames

    work = work.sort_values(['frame', 'score', 'det_index'], ascending=[True, False, True]).reset_index(drop=True)
    for f, g in work.groupby('frame', sort=True):
        vals = g[['x0', 'x1', 'x2', 'x3', 'y0', 'y1', 'y2', 'y3']].to_numpy(dtype=np.float32)
        pts = ensure_clockwise_batch(np.stack([vals[:, :4], vals[:, 4:]], axis=2))
        frames[int(f)] = (pts, g['score'].to_numpy(dtype=float), g['direction'].to_numpy(dtype=float))

    for f in range(int(first_frame), int(last_frame) + 1):
        if f not in frames:
            frames[f] = empty_frame()
    return frames

def prepare_obb_geometry_array(obbs: np.ndarray) -> dict[str, np.ndarray]:
    obbs = np.asarray(obbs, dtype=np.float32).reshape(-1, 4, 2)
    n = int(len(obbs))
    if n == 0:
        return {
            'obbs': np.zeros((0, 4, 2), dtype=np.float32),
            'centers': np.zeros((0, 2), dtype=float),
            'areas': np.zeros((0,), dtype=float),
            'scales': np.zeros((0,), dtype=float),
            'aabbs': np.zeros((0, 4), dtype=float),
        }
    centers = obbs.mean(axis=1, dtype=float)
    areas = polygon_area_batch(obbs.astype(float))
    scales = np.sqrt(np.maximum(areas, 1.0))
    aabbs = obb_aabbs_batch(obbs.astype(float))
    return {'obbs': obbs, 'centers': centers, 'areas': areas, 'scales': scales, 'aabbs': aabbs}

def prepare_prev_obb_geometry(prev_obbs: list[np.ndarray | None]) -> list[dict | None]:
    valid_idx = [i for i, o in enumerate(prev_obbs) if o is not None]
    result: list[dict | None] = [None] * len(prev_obbs)
    if not valid_idx:
        return result
    batch = np.asarray([prev_obbs[i] for i in valid_idx], dtype=np.float32).reshape(-1, 4, 2)
    centers = batch.mean(axis=1, dtype=float)
    areas = polygon_area_batch(batch.astype(float))
    aabbs = obb_aabbs_batch(batch.astype(float))
    for k, i in enumerate(valid_idx):
        a = float(areas[k])
        result[i] = {'obb': batch[k], 'center': centers[k], 'area': a, 'scale': math.sqrt(max(a, 1.0)), 'aabb': aabbs[k]}
    return result

def iou_obb_prepared(pts_a: np.ndarray | None, area_a: float, pts_b: np.ndarray | None, area_b: float) -> float:
    if pts_a is None or pts_b is None or area_a <= 0.0 or (area_b <= 0.0):
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(pts_a, pts_b)
    inter_area = float(inter_area)
    union = float(area_a) + float(area_b) - inter_area
    return inter_area / union if union > 0.0 else 0.0

def assign_ids_iou_mode(
    predicted_obbs: list[np.ndarray | None],
    last_valid_obbs: list[np.ndarray | None],
    miss_counts: list[int],
    yolo_obbs: np.ndarray,
    yolo_directions: np.ndarray,
    prev_directions: list[float | None],
    strong_iou: float,
    strong_dir_deg: float,
    max_age: int,
    *,
    require_iou_gate: bool = True,
    require_dir_gate: bool = True,
    add_dir_cost: bool = True,
    direction_cost_mode: str = 'heading',
    iou_weight: float = 1.0,
    direction_weight: float = 1.0,
    miss_weight: float = 1.0,
    pred_directions: list[float | None] | None = None,
) -> tuple[list[int | None], dict[int, tuple[float, float]]]:
    """Hungarian assignment using best-of{predicted, last_valid} OBB references.

    Costs use quadratic normalization (u^2) with gate boundary at u=1 (no upper clip).
    Hard gates still exclude candidates in stages that declare them.
    age_cost is linear and clipped to [0,1].
    direction_cost_mode='heading' uses signed heading disagreement; 'axis'
    uses heading-independent OBB long-axis disagreement.
    """
    n_tracks = len(predicted_obbs)
    n_det = int(len(yolo_obbs))
    det_of_id: list[int | None] = [None] * n_tracks
    assign_info: dict[int, tuple[float, float]] = {}
    if n_tracks == 0 or n_det == 0:
        return det_of_id, assign_info

    det_geo = prepare_obb_geometry_array(yolo_obbs)
    det_aabbs = det_geo['aabbs']
    pred_geo = prepare_prev_obb_geometry(predicted_obbs)
    lv_geo = prepare_prev_obb_geometry(last_valid_obbs)

    big = 1e12
    # Separate cost matrices per reference so the predicted (KF) reference can be
    # matched first within the stage; the last_valid reference fills the remainder.
    cost_pred = np.full((n_tracks, n_det), big, dtype=float)
    cost_prev = np.full((n_tracks, n_det), big, dtype=float)
    valid_pred = np.zeros((n_tracks, n_det), dtype=bool)
    valid_prev = np.zeros((n_tracks, n_det), dtype=bool)
    iou_pred = np.zeros((n_tracks, n_det), dtype=float)
    iou_prev = np.zeros((n_tracks, n_det), dtype=float)

    s_iou = float(strong_iou)
    s_dir = max(float(strong_dir_deg), 1.0)
    m_age = max(float(max_age), 1.0)
    iou_denom = max(1.0 - s_iou, 1e-6)

    for tidx in range(n_tracks):
        mc = int(miss_counts[tidx]) if tidx < len(miss_counts) else 0
        age_cost = min(mc / m_age, 1.0)

        prev_dir: float | None = None
        if tidx < len(prev_directions) and prev_directions[tidx] is not None:
            try:
                v = float(prev_directions[tidx])
                if np.isfinite(v):
                    prev_dir = wrap_angle_deg(v)
            except (TypeError, ValueError):
                pass

        # pred reference uses KF-extrapolated direction; lv reference uses last_valid_direction
        _pred_dir: float | None = None
        if pred_directions is not None and tidx < len(pred_directions) and pred_directions[tidx] is not None:
            try:
                v = float(pred_directions[tidx])
                if np.isfinite(v):
                    _pred_dir = wrap_angle_deg(v)
            except (TypeError, ValueError):
                pass
        # fallback: if no KF direction available, use last_valid_direction for pred reference too
        ref_dirs_by_ref = (_pred_dir if _pred_dir is not None else prev_dir, prev_dir)

        # AABB union over both references for candidate pre-filter
        candidate_mask = np.zeros(n_det, dtype=bool)
        for rg in (pred_geo[tidx], lv_geo[tidx]):
            if rg is None:
                continue
            ra = rg['aabb']
            candidate_mask |= (
                (ra[2] > det_aabbs[:, 0]) & (det_aabbs[:, 2] > ra[0]) &
                (ra[3] > det_aabbs[:, 1]) & (det_aabbs[:, 3] > ra[1])
            )

        for didx in np.flatnonzero(candidate_mask).tolist():
            det_area = float(det_geo['areas'][didx])
            if det_area <= 0.0:
                continue

            det_dir: float | None = None
            if didx < len(yolo_directions):
                try:
                    dv = float(yolo_directions[didx])
                    if np.isfinite(dv):
                        det_dir = wrap_angle_deg(dv)
                except (TypeError, ValueError):
                    pass

            # Score each reference independently and route into its own matrix
            # (ref_pos 0 = predicted/KF, 1 = last_valid).  No best-of min: the two
            # references are resolved by separate Hungarian passes below, so a
            # predicted-reference match always gets first claim on a detection.
            for ref_pos, (rg, ref_dir) in enumerate(
                zip((pred_geo[tidx], lv_geo[tidx]), ref_dirs_by_ref)
            ):
                if rg is None:
                    continue
                iouv = iou_obb_prepared(rg['obb'], rg['area'], det_geo['obbs'][didx], det_area)

                # Hard gate checks
                if require_iou_gate:
                    if iouv < s_iou:
                        continue
                else:
                    if iouv <= 0.0:
                        continue
                if require_dir_gate and ref_dir is not None and det_dir is not None:
                    if angular_distance_deg(ref_dir, det_dir) > s_dir:
                        continue

                u_iou = (1.0 - iouv) / iou_denom
                total = iou_weight * u_iou * u_iou + miss_weight * age_cost
                if add_dir_cost:
                    if direction_cost_mode == 'axis':
                        u_dir = axis_angular_distance_deg_from_obbs(
                            rg['obb'], det_geo['obbs'][didx],
                        ) / s_dir
                        total += direction_weight * u_dir * u_dir
                    elif ref_dir is not None and det_dir is not None:
                        u_dir = angular_distance_deg(ref_dir, det_dir) / s_dir
                        total += direction_weight * u_dir * u_dir

                if ref_pos == 0:
                    cost_pred[tidx, didx] = total
                    valid_pred[tidx, didx] = True
                    iou_pred[tidx, didx] = iouv
                else:
                    cost_prev[tidx, didx] = total
                    valid_prev[tidx, didx] = True
                    iou_prev[tidx, didx] = iouv

    # Pass 1: predicted-reference matches take priority within the stage.
    for tidx, didx in solve_assignment_by_cost(cost_pred, valid_pred):
        det_of_id[tidx] = int(didx)
        assign_info[tidx] = (0.0, float(iou_pred[tidx, didx]))

    # Pass 2: last_valid-reference matches fill the still-unassigned tracks,
    # restricted to detections not already claimed by a predicted-reference match.
    if any(d is None for d in det_of_id):
        valid_prev2 = valid_prev.copy()
        for tidx in range(n_tracks):
            if det_of_id[tidx] is not None:
                valid_prev2[tidx, :] = False
        for didx in (d for d in det_of_id if d is not None):
            valid_prev2[:, didx] = False
        for tidx, didx in solve_assignment_by_cost(cost_prev, valid_prev2):
            if det_of_id[tidx] is None:
                det_of_id[tidx] = int(didx)
                assign_info[tidx] = (0.0, float(iou_prev[tidx, didx]))
    return det_of_id, assign_info


def assign_ids_distance_mode(
    ref_obbs: list[np.ndarray | None],
    miss_counts: list[int],
    yolo_obbs: np.ndarray,
    max_age: int,
    distance_weight: float = 1.0,
    miss_weight: float = 1.0,
) -> tuple[list[int | None], dict[int, tuple[float, float]]]:
    """Hungarian NN assignment for S6 (long-gap recovery).

    Cost = (center_distance / sqrt(obb_area)) + age_cost.
    No distance gate: any unmatched detection is a candidate.
    """
    n_tracks = len(ref_obbs)
    n_det = int(len(yolo_obbs))
    det_of_id: list[int | None] = [None] * n_tracks
    assign_info: dict[int, tuple[float, float]] = {}
    if n_tracks == 0 or n_det == 0:
        return det_of_id, assign_info

    det_centers = np.asarray(yolo_obbs, dtype=float).reshape(-1, 4, 2).mean(axis=1)
    m_age = max(float(max_age), 1.0)
    big = 1e12
    cost = np.full((n_tracks, n_det), big, dtype=float)
    valid = np.zeros((n_tracks, n_det), dtype=bool)
    dist_cache = np.zeros((n_tracks, n_det), dtype=float)

    for tidx in range(n_tracks):
        ref_obb = ref_obbs[tidx]
        if not valid_obb(ref_obb):
            continue
        mc = int(miss_counts[tidx]) if tidx < len(miss_counts) else 0
        age_cost = min(mc / m_age, 1.0)
        ref_center = obb_center(ref_obb)
        ref_scale = math.sqrt(max(obb_polygon_area(ref_obb), 1.0))
        dists = np.linalg.norm(det_centers - ref_center, axis=1)
        for didx in range(n_det):
            dist = float(dists[didx])
            dist_cost = dist / ref_scale
            cost[tidx, didx] = distance_weight * dist_cost + miss_weight * age_cost
            valid[tidx, didx] = True
            dist_cache[tidx, didx] = dist

    for tidx, didx in solve_assignment_by_cost(cost, valid):
        det_of_id[tidx] = int(didx)
        assign_info[tidx] = (1, float(dist_cache[tidx, didx]))
    return det_of_id, assign_info

def assign_ids_gated_distance_mode(
    pred_obbs: list[np.ndarray | None],
    last_valid_obbs: list[np.ndarray | None],
    kf_speeds: list[float],
    miss_counts: list[int],
    yolo_obbs: np.ndarray,
    max_age: int,
    *,
    base_gate: float = 2.0,
    speed_k: float = 1.5,
    distance_weight: float = 1.0,
    miss_weight: float = 1.0,
) -> tuple[list[int | None], dict[int, tuple[float, float]]]:
    """Velocity-aware gated NN for ACTIVE tracks whose IoU stages failed at IoU=0.

    Uses the Kalman-predicted OBB center.  The acceptance radius grows with the
    KF speed estimate and miss_count, so a fast-moving object is recovered on the
    next frame instead of waiting MAX_AGE frames for the dormant S6 path.
    Cost = center_distance / sqrt(area) + age_cost.  A hard radius gate prevents
    grabbing unrelated far detections.
    """
    n_tracks = len(pred_obbs)
    n_det = int(len(yolo_obbs))
    det_of_id: list[int | None] = [None] * n_tracks
    assign_info: dict[int, tuple[float, float]] = {}
    if n_tracks == 0 or n_det == 0:
        return det_of_id, assign_info

    det_centers = np.asarray(yolo_obbs, dtype=float).reshape(-1, 4, 2).mean(axis=1)
    m_age = max(float(max_age), 1.0)
    big = 1e12
    cost = np.full((n_tracks, n_det), big, dtype=float)
    valid = np.zeros((n_tracks, n_det), dtype=bool)
    dist_cache = np.zeros((n_tracks, n_det), dtype=float)

    for tidx in range(n_tracks):
        ref = pred_obbs[tidx] if valid_obb(pred_obbs[tidx]) else last_valid_obbs[tidx]
        if not valid_obb(ref):
            continue
        mc = int(miss_counts[tidx]) if tidx < len(miss_counts) else 0
        age_cost = min(mc / m_age, 1.0)
        ref_center = obb_center(ref)
        ref_scale = math.sqrt(max(obb_polygon_area(ref), 1.0))
        speed = float(kf_speeds[tidx]) if tidx < len(kf_speeds) else 0.0
        # Radius expands with speed and each consecutive miss.
        radius = base_gate * ref_scale * (1.0 + mc) + speed_k * speed
        dists = np.linalg.norm(det_centers - ref_center, axis=1)
        for didx in range(n_det):
            dist = float(dists[didx])
            if dist > radius:
                continue
            cost[tidx, didx] = distance_weight * (dist / ref_scale) + miss_weight * age_cost
            valid[tidx, didx] = True
            dist_cache[tidx, didx] = dist

    for tidx, didx in solve_assignment_by_cost(cost, valid):
        det_of_id[tidx] = int(didx)
        assign_info[tidx] = (ATYPE_VEL_DIST, float(dist_cache[tidx, didx]))
    return det_of_id, assign_info

def to_df(buf: dict[int, dict[int, tuple]], pats: list[str], interp: bool, max_ids: int | None=None) -> pd.DataFrame:
    ids = sorted(buf)
    if max_ids is not None:
        ids = ids[:max_ids]
    header = ['position'] + [f'{c}{tid}' for tid in ids for c in pats]
    min_f: int | None = None
    max_f: int | None = None
    for tid in ids:
        per_id = buf.get(tid, {})
        if not per_id:
            continue
        for frame in per_id.keys():
            frame = int(frame)
            min_f = frame if min_f is None else min(min_f, frame)
            max_f = frame if max_f is None else max(max_f, frame)
    if min_f is None or max_f is None:
        return pd.DataFrame(columns=header)

    row_count = int(max_f) - int(min_f) + 1
    data: dict[str, np.ndarray] = {
        'position': np.arange(int(min_f), int(max_f) + 1, dtype=int),
    }
    frame_numbers = range(int(min_f), int(max_f) + 1)
    width = len(pats)
    default_values = (np.nan,) * width
    for tid in ids:
        per_id = buf.get(tid, {})
        if len(per_id) >= row_count * 0.5:
            def iter_values():
                for frame in frame_numbers:
                    values = per_id.get(frame, default_values)
                    if width == 1 and not isinstance(values, (tuple, list, np.ndarray)):
                        yield values
                    else:
                        yield from values

            block = np.fromiter(
                iter_values(),
                dtype=float,
                count=row_count * width,
            ).reshape(row_count, width)
        else:
            block = np.full((row_count, width), np.nan, dtype=float)
            for frame, values in per_id.items():
                row_idx = int(frame) - int(min_f)
                if row_idx < 0 or row_idx >= row_count:
                    continue
                if values is None:
                    continue
                if width == 1 and not isinstance(values, (tuple, list, np.ndarray)):
                    block[row_idx, 0] = float(values)
                else:
                    block[row_idx, :] = values
        for col_idx, pat in enumerate(pats):
            data[f'{pat}{tid}'] = block[:, col_idx]

    df = pd.DataFrame(data, columns=header)
    if interp:
        df = df.interpolate('linear', axis=0, limit_direction='both', limit_area='inside')
    return df

def obb_buf_to_df(obb_buf: dict[int, dict[int, tuple]], interp: bool, max_ids: int | None=None) -> pd.DataFrame:
    pats = [f'x{i}' for i in range(4)] + [f'y{i}' for i in range(4)]
    return to_df(obb_buf, pats, interp, max_ids)


def _safe_result_component(value: object, fallback: str = 'output') -> str:
    text = re.sub(r'[<>:"/\\|?*\s]+', '_', str(value or '').strip())
    text = re.sub(r'_+', '_', text).strip(' ._')
    return text or fallback


def final_result_csv_path(
    session_path: str,
    model_name: str,
    dataset_name: str,
    run_name: str,
    video_name: str,
    family: str,
) -> str:
    family = str(family or '').strip()
    if family not in {'filled', 'id_resolved'}:
        raise ValueError(f'Unsupported final result family: {family!r}')
    result_dir = os.path.join(session_path, 'results')
    suffix = '_id_resolved' if family == 'id_resolved' else ''
    filename = '_'.join([
        _safe_result_component(model_name, 'model'),
        _safe_result_component(dataset_name, 'dataset'),
        _safe_result_component(run_name, 'run'),
        _safe_result_component(video_name, 'video'),
    ]) + suffix + '.csv'
    return os.path.join(result_dir, filename)



def final_result_df_from_buffers(buffers: dict, num_objects: int) -> pd.DataFrame:
    ids = list(range(int(num_objects)))
    columns = ['frame'] + [
        f'{name}{tid}'
        for tid in ids
        for name in ('cx', 'cy', 'w', 'h', 'heading')
    ]

    frame_set: set[int] = set()
    for buf_name in ('obb_buf', 'class_buf', 'pos_buf'):
        buf = buffers.get(buf_name, {}) or {}
        for tid in ids:
            frame_set.update(int(frame) for frame in buf.get(tid, {}).keys())
    if not frame_set:
        return pd.DataFrame(columns=columns)

    min_f = min(frame_set)
    max_f = max(frame_set)
    frames = np.arange(min_f, max_f + 1, dtype=int)
    data: dict[str, np.ndarray] = {'frame': frames}
    row_by_frame = {int(frame): idx for idx, frame in enumerate(frames.tolist())}

    obb_buf = buffers.get('obb_buf', {}) or {}
    pos_buf = buffers.get('pos_buf', {}) or {}
    class_buf = buffers.get('class_buf', {}) or {}

    for tid in ids:
        block = np.full((len(frames), 5), np.nan, dtype=float)

        for frame, raw_obb in obb_buf.get(tid, {}).items():
            row_idx = row_by_frame.get(int(frame))
            if row_idx is None:
                continue
            obb = tuple_to_obb(raw_obb)
            if obb is None or not valid_obb(obb):
                continue
            rect = obb_points_to_rect(obb)
            block[row_idx, 0:4] = rect[0:4]

        for frame, raw_pos in pos_buf.get(tid, {}).items():
            row_idx = row_by_frame.get(int(frame))
            if row_idx is None or raw_pos is None:
                continue
            try:
                block[row_idx, 0] = float(raw_pos[0])
                block[row_idx, 1] = float(raw_pos[1])
            except (TypeError, ValueError, IndexError):
                pass

        for frame, raw_dir in class_buf.get(tid, {}).items():
            row_idx = row_by_frame.get(int(frame))
            if row_idx is None:
                continue
            try:
                heading = float(raw_dir[0] if isinstance(raw_dir, (tuple, list, np.ndarray)) else raw_dir)
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(heading):
                block[row_idx, 4] = heading % 360.0

        for col_idx, name in enumerate(('cx', 'cy', 'w', 'h', 'heading')):
            data[f'{name}{tid}'] = block[:, col_idx]

    return pd.DataFrame(data, columns=columns)


def save_final_result_csv(
    session_path: str,
    model_name: str,
    dataset_name: str,
    run_name: str,
    video_name: str,
    buffers: dict,
    num_objects: int,
    family: str,
) -> str:
    path = final_result_csv_path(
        session_path,
        model_name,
        dataset_name,
        run_name,
        video_name,
        family,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = final_result_df_from_buffers(buffers, num_objects)
    tmp_path = path + '.tmp'
    df.to_csv(tmp_path, index=False, float_format='%.6f')
    os.replace(tmp_path, path)
    return path




def resolve_weight_from_tracking_dir(session_path: str, model_name: str, dataset_name: str, weight: str) -> str:
    """Return the requested tracking weight without substituting another run."""
    del session_path, model_name, dataset_name
    return normalize_checkpoint_weight(weight)


def resolve_tracking_jobs_for_requested_weights(
    session_path: str,
    model_name: str,
    dataset_name: str,
    requested_weights: list[str],
) -> list[dict]:
    """Build tracking jobs without best/last/epoch alias substitution."""
    normalized_weights = [
        resolve_weight_from_tracking_dir(session_path, model_name, dataset_name, weight)
        for weight in requested_weights
    ]
    selected_weights = deduplicate_best_epoch_weights(
        session_path,
        model_name,
        dataset_name,
        normalized_weights,
    )
    jobs: list[dict] = []
    seen_runs: set[str] = set()
    for source_weight in selected_weights:
        run_name = build_tracking_run_name(source_weight)
        if run_name in seen_runs:
            print(
                f'[WEIGHT DEDUP] skip weight={source_weight}: output run {run_name} already requested',
                flush=True,
            )
            continue
        seen_runs.add(run_name)
        jobs.append({
            'run_name': run_name,
            'source_weight': source_weight,
            'requested_weight': source_weight,
        })
    return jobs

def pickle_has_data(path: str) -> bool:
    if not os.path.exists(path):
        return False
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return not obj.empty and len(obj.columns) > 0
    if isinstance(obj, dict):
        if not obj:
            return False
        for value in obj.values():
            if isinstance(value, pd.DataFrame):
                if not value.empty and len(value.columns) > 0:
                    return True
            elif isinstance(value, dict):
                if any(bool(inner_value) for inner_value in value.values()):
                    return True
            elif isinstance(value, (list, tuple, set)):
                if len(value) > 0:
                    return True
            elif value is not None:
                return True
        return False
    if isinstance(obj, (list, tuple, set)):
        return len(obj) > 0
    return obj is not None

def _buffer_scalar(buf: dict, tid: int, frame: int, default: float = np.nan) -> float:
    value = buf.get(int(tid), {}).get(int(frame))
    if value is None:
        return float(default)
    try:
        raw = value[0] if isinstance(value, (tuple, list, np.ndarray)) else value
        result = float(raw)
    except (TypeError, ValueError, IndexError):
        return float(default)
    return result


def require_provenance_buffers(buffers: dict, num_objects: int) -> dict:
    """Validate that current tracking output contains complete provenance."""
    obb_buf = buffers.get('obb_buf', {})
    assign_type_buf = buffers.get('assign_type_buf', {})

    frames_by_tid: dict[int, set[int]] = {}
    for tid in range(int(num_objects)):
        frame_ids = {int(f) for f in obb_buf.get(tid, {}).keys()}
        frame_ids.update(int(f) for f in assign_type_buf.get(tid, {}).keys())
        if frame_ids:
            frames_by_tid[tid] = frame_ids

    for name in PROVENANCE_BUFFER_NAMES:
        pbuf = buffers.get(name)
        if not isinstance(pbuf, dict):
            raise RuntimeError(f'Missing required provenance buffer: {name}')
        for tid, frame_ids in frames_by_tid.items():
            per_id = pbuf.get(tid)
            if not isinstance(per_id, dict):
                raise RuntimeError(f'Missing required provenance buffer: {name}[{tid}]')
            missing = frame_ids.difference(per_id)
            if missing:
                raise RuntimeError(
                    f'Incomplete provenance buffer: {name}[{tid}] missing frame {min(missing)}'
                )
    return buffers


def _resolve_nan_direction_runs(
    class_buf: dict,
    obb_buf: dict,
    direction_corrected_buf: dict,
    assign_type_history_buf: dict,
    frame_ids: list,
    source_label: str = 'merged',
) -> None:
    """Post-tracking Viterbi pass: resolve direction-NaN frames from unreliable stages.

    This version resolves every direction=NaN frame with a valid OBB, including
    blocks without neighbouring anchors and blocks
    that contain OBB gaps.  Frames without a valid OBB cannot define an OBB
    long-axis direction and are therefore left as NaN, but they no longer split
    the Viterbi block or stop anchor search.

    If no finite heading anchor exists on either side of a block, the absolute
    180-degree sign is unidentifiable from OBB geometry alone.  In that case the
    first valid OBB in the block is assigned the positive long-axis sign, and the
    remaining signs are chosen by transition-cost minimization.

    """
    _src = str(source_label)
    n = len(frame_ids)

    def _direction_at(tid_class: dict, fid: int) -> float:
        raw = tid_class.get(fid, (np.nan,))
        try:
            val = raw[0] if isinstance(raw, (tuple, list, np.ndarray)) else raw
            return float(val)
        except (TypeError, ValueError, IndexError):
            return float('nan')

    def _nearest_anchor(tid_class: dict, start_idx: int, step: int) -> float | None:
        j = int(start_idx)
        while 0 <= j < n:
            val = _direction_at(tid_class, frame_ids[j])
            if np.isfinite(val):
                return wrap_angle_deg(val)
            j += int(step)
        return None

    for tid in list(obb_buf.keys()):
        tid_class = class_buf.get(tid, {})
        tid_obb = obb_buf.get(tid, {})

        i = 0
        while i < n:
            fid = frame_ids[i]
            dir_val = _direction_at(tid_class, fid)
            obb_pts = tuple_to_obb(tid_obb.get(fid))

            # Start only at frames whose heading is pending and whose OBB can
            # actually supply a long-axis candidate.
            if not (np.isnan(dir_val) and valid_obb(obb_pts)):
                i += 1
                continue

            # Extend over a contiguous direction-NaN block.  Invalid-OBB frames
            # inside the block are allowed; they are skipped during DP output
            # but no longer split the run.
            run_start = i
            run_end = i
            while run_end + 1 < n:
                nfid = frame_ids[run_end + 1]
                ndir = _direction_at(tid_class, nfid)
                if np.isnan(ndir):
                    run_end += 1
                else:
                    break

            left_anchor = _nearest_anchor(tid_class, run_start - 1, -1)
            right_anchor = _nearest_anchor(tid_class, run_end + 1, 1)

            # Collect only valid-OBB frames.  OBB-gap frames cannot receive a
            # heading because no long-axis candidate exists at those frames.
            run_fids: list[int] = []
            angles_plus: list[float] = []
            for fid_k in frame_ids[run_start:run_end + 1]:
                obb_k = tuple_to_obb(tid_obb.get(fid_k))
                if not valid_obb(obb_k):
                    continue
                v = positive_long_axis_direction(obb_k)
                run_fids.append(int(fid_k))
                angles_plus.append(unit_vec_to_angle_deg(v))

            if not run_fids:
                i = run_end + 1
                continue

            # 2-state Viterbi DP.
            # State 0 = positive long-axis direction (+v).
            # State 1 = negative long-axis direction (-v = +180 deg).
            n_run = len(run_fids)
            a0_p = angles_plus[0]
            a0_m = (a0_p + 180.0) % 360.0

            if left_anchor is not None:
                dp = [
                    angular_distance_deg(left_anchor, a0_p),
                    angular_distance_deg(left_anchor, a0_m),
                ]
            elif right_anchor is not None:
                # The right anchor will resolve the global 180-degree sign.
                dp = [0.0, 0.0]
            else:
                # With no anchors, OBB geometry defines only an unoriented axis.
                # Choose a deterministic convention: first valid frame = +v.
                dp = [0.0, float('inf')]

            # back[k][s] = predecessor state at valid-OBB frame k-1 for arriving
            # at state s at valid-OBB frame k.
            back: list[list[int]] = [[0, 0] for _ in range(n_run)]

            for k in range(1, n_run):
                ak_p = angles_plus[k]
                ak_m = (ak_p + 180.0) % 360.0
                prev_ap = angles_plus[k - 1]
                prev_am = (prev_ap + 180.0) % 360.0

                c00 = dp[0] + angular_distance_deg(prev_ap, ak_p)
                c10 = dp[1] + angular_distance_deg(prev_am, ak_p)
                if c00 <= c10:
                    new_p, back[k][0] = c00, 0
                else:
                    new_p, back[k][0] = c10, 1

                c01 = dp[0] + angular_distance_deg(prev_ap, ak_m)
                c11 = dp[1] + angular_distance_deg(prev_am, ak_m)
                if c01 <= c11:
                    new_m, back[k][1] = c01, 0
                else:
                    new_m, back[k][1] = c11, 1

                dp = [new_p, new_m]

            if right_anchor is not None:
                last_ap = angles_plus[-1]
                last_am = (last_ap + 180.0) % 360.0
                dp[0] += angular_distance_deg(last_ap, right_anchor)
                dp[1] += angular_distance_deg(last_am, right_anchor)

            final_state = 0 if dp[0] <= dp[1] else 1

            states = [0] * n_run
            states[n_run - 1] = final_state
            for k in range(n_run - 1, 0, -1):
                states[k - 1] = back[k][states[k]]

            tid_class_w = class_buf.setdefault(tid, {})
            tid_dc_w = direction_corrected_buf.setdefault(tid, {})
            tid_hist_w = assign_type_history_buf.setdefault(tid, {})
            for k, fid_k in enumerate(run_fids):
                ap = angles_plus[k]
                final_angle = ap if states[k] == 0 else (ap + 180.0) % 360.0
                tid_class_w[fid_k] = (final_angle,)
                tid_dc_w[fid_k] = (1.0,)
                tid_hist_w.setdefault(fid_k, []).append((_src, float(ATYPE_VITERBI)))

            i = run_end + 1

def long_axis_alignment_abs(pts: np.ndarray, reference: float | int | np.ndarray | None) -> float:
    ref_vec = reference_to_unit_vec(reference)
    if ref_vec is None:
        return 0.0
    major = positive_long_axis_direction(pts)
    return abs(float(np.dot(major, ref_vec)))

def is_long_axis_compatible(
    pts: np.ndarray,
    reference: float | int | np.ndarray | None,
    *,
    max_axis_error_deg: float = 45,
) -> bool:
    cos_thr = math.cos(math.radians(float(max_axis_error_deg)))
    return long_axis_alignment_abs(pts, reference) >= cos_thr

def _select_initial_assignment(
    detection_frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_objects: int,
) -> tuple[int, np.ndarray]:
    """Return the first S0 with enough score>=0.5 YOLO detections."""
    if int(num_objects) <= 0:
        raise ValueError('num_objects must be positive.')
    for abs_fid in sorted(int(f) for f in detection_frames):
        yolo_obbs, yolo_scores, _ = detection_frames[abs_fid]
        yolo_scores = np.asarray(yolo_scores, dtype=float)
        eligible = np.flatnonzero(yolo_scores >= INITIAL_CONFIDENCE_THRESHOLD)
        if len(eligible) < num_objects:
            continue
        ranked_keep = eligible[np.argsort(-yolo_scores[eligible], kind='stable')]
        selected_order = np.asarray(ranked_keep[:num_objects], dtype=int)
        assert len(selected_order) == num_objects
        assert np.array_equal(selected_order, ranked_keep[:num_objects])
        return int(abs_fid), selected_order
    raise RuntimeError(
        f'Could not find an initial frame with at least {num_objects} detections '
        f'after confidence >= {INITIAL_CONFIDENCE_THRESHOLD:g} filtering.'
    )


def _track_detection_pass(
    detection_frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_objects: int,
    nms_iou: float,
    max_age: int,
    strong_iou: float,
    strong_dir_deg: float,
    *,
    reverse: bool,
    initial_frame: int,
    initial_assignment: tuple[tuple[int, np.ndarray, float, float], ...],
    tracking_cost: dict | None = None,
    sharp_turn_accept_run: int = 3,
):
    """Run one directional pass from a caller-supplied common S0 assignment."""
    all_frame_ids = sorted(int(f) for f in detection_frames.keys())
    if not all_frame_ids:
        raise RuntimeError('No detection frames found.')
    if int(initial_frame) not in detection_frames:
        raise ValueError(f'Initial frame {initial_frame} is absent from detection_frames.')
    initial_pos = all_frame_ids.index(int(initial_frame))
    if reverse:
        processing_order = [int(initial_frame)] + list(reversed(all_frame_ids[:initial_pos]))
    else:
        processing_order = all_frame_ids[initial_pos:]
    frame_ids = processing_order
    max_age = max(1, int(max_age))
    _tc = dict(tracking_cost or {})
    _iou_weight = float(_tc.get('IOU_WEIGHT', 1.0))
    _direction_weight = float(_tc.get('DIRECTION_WEIGHT', 1.0))
    _miss_weight = float(_tc.get('MISS_WEIGHT', 1.0))
    _distance_weight = float(_tc.get('DISTANCE_WEIGHT', 1.0))
    sharp_turn_accept_run = max(2, int(_tc.get('SHARP_TURN_ACCEPT_RUN', sharp_turn_accept_run)))

    # Both passes share S0; only temporal order and Kalman dt differ.
    kf_dt: float = -1.0 if reverse else 1.0
    assert kf_dt == (-1.0 if reverse else 1.0)
    _src: str = 'bwd' if reverse else 'fwd'
    direction_label = 'backward' if reverse else 'forward'
    frame_order_index = {int(f): i for i, f in enumerate(processing_order)}
    overlap_guard_iou = float(nms_iou)
    overlap_guard_enabled = (
        np.isfinite(overlap_guard_iou)
        and overlap_guard_iou > 0.0
        and overlap_guard_iou < 1.0
    )

    pos_buf: dict[int, dict[int, tuple]] = {}
    obb_buf: dict[int, dict[int, tuple]] = {}
    class_buf: dict[int, dict[int, tuple]] = {}
    score_buf: dict[int, dict[int, tuple]] = {}
    assign_type_buf: dict[int, dict[int, tuple]] = {}
    assign_type_history_buf: dict[int, dict[int, list]] = {}
    assign_val_buf: dict[int, dict[int, tuple]] = {}
    assign_total_buf: dict[int, dict[int, tuple]] = {}
    assign_iou_cost_buf: dict[int, dict[int, tuple]] = {}
    assign_direction_cost_buf: dict[int, dict[int, tuple]] = {}
    assign_distance_cost_buf: dict[int, dict[int, tuple]] = {}
    assign_score_cost_buf: dict[int, dict[int, tuple]] = {}
    obb_source_buf: dict[int, dict[int, tuple]] = {}
    obb_corrected_buf: dict[int, dict[int, tuple]] = {}
    direction_corrected_buf: dict[int, dict[int, tuple]] = {}
    switch_corrected_buf: dict[int, dict[int, tuple]] = {}

    position_kfs = [None] * num_objects
    bbox_shape_kfs = [None] * num_objects
    direction_kfs = [None] * num_objects
    last_valid_obb = [None] * num_objects
    last_valid_direction = [None] * num_objects
    last_valid_frame: list[int | None] = [None] * num_objects
    miss_count = [0] * num_objects
    # Per-frame references used by long-gap recovery.  Unlike obb_buf, this
    # history remains populated during missing runs with the Kalman-predicted
    # OBB, so the exact frame MAX_AGE steps behind is always addressable.
    reference_obb_history: list[dict[int, np.ndarray]] = [dict() for _ in range(num_objects)]
    # Per-track state for detecting spurious VEL-DIST excursions.
    vel_dist_prev: list[dict | None] = [None] * num_objects
    # Per-track state for sharp-turn NaN-chain detection.
    sharp_turn_count: list[int] = [0] * num_objects
    sharp_turn_buf: list[list] = [[] for _ in range(num_objects)]
    sharp_turn_heading: list[float | None] = [None] * num_objects
    def write_tracking_row(
        tid: int,
        abs_fid: int,
        *,
        pos: tuple[float, float],
        obb: tuple,
        direction: float,
        score: float,
        assign_type: float,
        assign_val: float,
        iou_val: float,
        distance_val: float,
        obb_source: float,
        obb_corrected: float,
        direction_corrected: float,
        switch_corrected: float,
    ) -> None:
        pos_buf.setdefault(tid, {})[abs_fid] = pos
        obb_buf.setdefault(tid, {})[abs_fid] = obb
        class_buf.setdefault(tid, {})[abs_fid] = (direction,)
        score_buf.setdefault(tid, {})[abs_fid] = (score,)
        assign_type_buf.setdefault(tid, {})[abs_fid] = (assign_type,)
        if np.isfinite(assign_type):
            assign_type_history_buf.setdefault(tid, {}).setdefault(abs_fid, []).append((_src, float(assign_type)))
        assign_val_buf.setdefault(tid, {})[abs_fid] = (assign_val,)
        assign_total_buf.setdefault(tid, {})[abs_fid] = (assign_val,)
        assign_iou_cost_buf.setdefault(tid, {})[abs_fid] = (iou_val,)
        assign_direction_cost_buf.setdefault(tid, {})[abs_fid] = (np.nan,)
        assign_distance_cost_buf.setdefault(tid, {})[abs_fid] = (distance_val,)
        assign_score_cost_buf.setdefault(tid, {})[abs_fid] = (np.nan,)
        obb_source_buf.setdefault(tid, {})[abs_fid] = (obb_source,)
        obb_corrected_buf.setdefault(tid, {})[abs_fid] = (obb_corrected,)
        direction_corrected_buf.setdefault(tid, {})[abs_fid] = (direction_corrected,)
        switch_corrected_buf.setdefault(tid, {})[abs_fid] = (switch_corrected,)

    def write_nan(
        tid: int,
        abs_fid: int,
        *,
        history_type: float | None = None,
    ) -> None:
        write_tracking_row(
            tid,
            abs_fid,
            pos=(np.nan, np.nan),
            obb=(np.nan,) * 8,
            direction=np.nan,
            score=np.nan,
            assign_type=np.nan,
            assign_val=np.nan,
            iou_val=np.nan,
            distance_val=np.nan,
            obb_source=OBB_SOURCE_MISSING,
            obb_corrected=0.0,
            direction_corrected=0.0,
            switch_corrected=0.0,
        )
        if history_type is not None:
            try:
                hist_val = float(history_type)
            except (TypeError, ValueError):
                hist_val = np.nan
            if np.isfinite(hist_val):
                assign_type_history_buf.setdefault(tid, {}).setdefault(abs_fid, []).append(
                    (_src, hist_val)
                )

    def _reset_sharp_turn(tid: int) -> None:
        sharp_turn_count[tid] = 0
        sharp_turn_buf[tid] = []
        sharp_turn_heading[tid] = None

    def _rollback_unconfirmed_fast_dist_before_missing(tid: int, current_proc_idx: int) -> bool:
        """Erase a one-frame S5/fast-distance excursion followed by a miss.

        A S5 match is useful for true fast motion, but it is unreliable when it
        is a single OBB excursion from a trusted IoU/initial anchor and the next
        processed frame is missing.  In that pattern, treat the S5 OBB itself as
        missing and restore the Kalman/last-valid state to the pre-S5 anchor so
        the deleted OBB does not contaminate subsequent prediction.
        """
        prev_vd = vel_dist_prev[int(tid)]
        if prev_vd is None:
            return False
        if int(prev_vd.get('proc_idx', -10**9)) != int(current_proc_idx) - 1:
            return False
        if not bool(prev_vd.get('trusted_anchor', False)):
            return False

        vd_frame = int(prev_vd['vd_frame'])
        write_nan(int(tid), vd_frame, history_type=ATYPE_MISSING)
        reference_obb_history[int(tid)].pop(vd_frame, None)

        prev_pts = prev_vd.get('prev_valid_pts')
        last_valid_obb[int(tid)] = prev_pts.copy() if valid_obb(prev_pts) else None
        prev_dir = prev_vd.get('prev_valid_direction')
        last_valid_direction[int(tid)] = (
            float(prev_dir) if prev_dir is not None and np.isfinite(float(prev_dir)) else None
        )
        prev_frame = prev_vd.get('prev_valid_frame')
        last_valid_frame[int(tid)] = int(prev_frame) if prev_frame is not None else None
        position_kfs[int(tid)] = copy.deepcopy(prev_vd.get('prev_position_kf'))
        bbox_shape_kfs[int(tid)] = copy.deepcopy(prev_vd.get('prev_bbox_shape_kf'))
        direction_kfs[int(tid)] = copy.deepcopy(prev_vd.get('prev_direction_kf'))

        # The stored filter states are the pre-measurement predictions for the
        # erased S5 frame.  Advance them once so the current missing frame is
        # represented by a prediction from the trusted anchor path, not from the
        # deleted S5 measurement.
        for _kf in (position_kfs[int(tid)], bbox_shape_kfs[int(tid)], direction_kfs[int(tid)]):
            if _kf is not None:
                _kf.predict()
        if valid_obb(last_valid_obb[int(tid)]):
            predicted_obbs[int(tid)] = predict_obb_from_filters(
                last_valid_obb[int(tid)],
                position_kfs[int(tid)],
                bbox_shape_kfs[int(tid)],
                direction_kfs[int(tid)],
            )
        else:
            predicted_obbs[int(tid)] = None

        # Count the erased S5 frame itself as the first missing frame.  The
        # caller then increments once more for the current missing frame.
        miss_count[int(tid)] = int(prev_vd.get('prev_miss_count', 0)) + 1
        vel_dist_prev[int(tid)] = None
        _reset_sharp_turn(int(tid))
        return True

    def write_measurement(
        tid: int,
        abs_fid: int,
        pts: np.ndarray,
        direction: float | None,
        score: float,
        assign_type: float,
        assign_val: float,
        iou_val: float | None = None,
        distance_val: float | None = None,
        obb_source: float = OBB_SOURCE_DETECTED,
        obb_corrected: bool = False,
        direction_corrected: bool = False,
    ) -> None:
        c = obb_center(pts)
        write_tracking_row(
            tid,
            abs_fid,
            pos=(float(c[0]), float(c[1])),
            obb=obb_to_tuple(pts),
            direction=float(direction) if direction is not None else np.nan,
            score=float(score),
            assign_type=float(assign_type),
            assign_val=float(assign_val),
            iou_val=np.nan if iou_val is None else float(iou_val),
            distance_val=np.nan if distance_val is None else float(distance_val),
            obb_source=float(obb_source),
            obb_corrected=1.0 if obb_corrected else 0.0,
            direction_corrected=1.0 if direction_corrected else 0.0,
            switch_corrected=0.0,
        )

    def remember_reference(tid: int, abs_fid: int, pts: np.ndarray | None) -> None:
        if valid_obb(pts):
            reference_obb_history[int(tid)][int(abs_fid)] = ensure_clockwise(pts).copy()

    def backfill_gap(tid: int, recover_fid: int, recovered_pts: np.ndarray) -> None:
        """Retroactively fill [last_valid_frame, recover_fid) for a recovered track.

        When a track is recovered after one or more missed frames (fast motion /
        IoU=0), the correct OBB was often already present in those frames but
        ignored by the IoU stages.  This re-scans each gap frame for an unmatched
        detection consistent with the linearly interpolated path (with a
        collision guard against other IDs' committed OBBs) and adopts it; missing
        frames fall back to the interpolated OBB.  Overwrites the NaN rows.
        """
        start_fid = last_valid_frame[tid]
        if start_fid is None:
            return
        start_pts = last_valid_obb[tid]
        if not valid_obb(start_pts) or not valid_obb(recovered_pts):
            return
        i0 = frame_order_index.get(int(start_fid))
        i1 = frame_order_index.get(int(recover_fid))
        if i0 is None or i1 is None or abs(i1 - i0) <= 1:
            return
        step = 1 if i1 > i0 else -1
        span = abs(i1 - i0)
        p0 = ensure_clockwise(start_pts).astype(float)
        p1 = align_cyclic_to_reference(ensure_clockwise(recovered_pts).astype(float), p0)
        # Per-frame displacement along the interpolation chord. Used as a
        # speed-proportional gate term: deviation from the straight constant-
        # velocity chord scales with how far the object travels per frame.
        step_len = float(np.linalg.norm(obb_center(p1) - obb_center(p0))) / max(span, 1)
        for k in range(1, span):
            idx = i0 + step * k
            fid = processing_order[idx]
            alpha = k / float(span)
            interp_pts = ((1.0 - alpha) * p0 + alpha * p1).astype(np.float32)
            interp_center = obb_center(interp_pts)
            interp_scale = math.sqrt(max(obb_polygon_area(interp_pts), 1.0))

            # Other IDs' committed OBBs at this frame (collision guard).
            others = [
                tuple_to_obb(obb_buf.get(o, {}).get(int(fid)))
                for o in range(num_objects) if o != tid
            ]
            others = [o for o in others if valid_obb(o)]
            others_aabbs = [obb_aabb(o) for o in others]

            g_obbs, g_scores, g_classes = detection_frames[int(fid)]

            best_didx: int | None = None
            best_d = max(2.0 * interp_scale, 1.5 * step_len)  # size floor + speed term
            for d in range(len(g_obbs)):
                cand = g_obbs[d]
                if overlap_guard_enabled:
                    cand_aabb = obb_aabb(cand)
                    overlaps_other = False
                    for other, other_aabb in zip(others, others_aabbs):
                        if not aabbs_overlap(cand_aabb, other_aabb):
                            continue
                        if iou_obb(cand, other) > overlap_guard_iou:
                            overlaps_other = True
                            break
                    if overlaps_other:
                        continue
                dist = float(np.linalg.norm(obb_center(cand) - interp_center))
                if dist < best_d:
                    best_d, best_didx = dist, d

            if best_didx is not None:
                pts = g_obbs[best_didx].copy()
                dir_b = unit_vec_to_angle_deg(
                    closest_long_axis_direction(pts, float(g_classes[best_didx])))
                write_measurement(
                    tid, int(fid), pts, dir_b, float(g_scores[best_didx]),
                    ATYPE_BKFILL_DET, best_d, distance_val=best_d,
                    obb_source=OBB_SOURCE_DETECTED, obb_corrected=True,
                )
                remember_reference(tid, int(fid), pts)
            # else: detection not found - leave the NaN row written by the main loop

    def update_filters_with_measurement(tid: int, pts: np.ndarray, dir_angle: float | None, reset: bool, abs_fid: int | None = None, freeze_direction: bool = False) -> float | None:
        pts = ensure_clockwise(pts)
        center = obb_center(pts)
        rect = obb_points_to_rect(pts)
        direction_vec = closest_long_axis_direction(pts, dir_angle)
        if reset or position_kfs[tid] is None:
            position_kfs[tid] = create_position_kf(center[0], center[1], dt=kf_dt)
        else:
            position_kfs[tid].update(center.astype(float))
        if reset or bbox_shape_kfs[tid] is None:
            bbox_shape_kfs[tid] = create_bbox_shape_kf(rect[2], rect[3], dt=kf_dt)
        else:
            bbox_shape_kfs[tid].update(np.array([rect[2], rect[3]], dtype=float))
        if not freeze_direction:
            if reset or direction_kfs[tid] is None:
                direction_kfs[tid] = create_direction_kf(direction_vec, dt=kf_dt)
            else:
                direction_kfs[tid].update(direction_vec.astype(float))
            kf_vec = direction_vec_from_kf(direction_kfs[tid], fallback=direction_vec)
            # If KF vector is near the current OBB short axis, do not use it
            # to choose the long-axis sign.
            if is_long_axis_compatible(pts, kf_vec, max_axis_error_deg=45):
                out_vec = closest_long_axis_direction(pts, kf_vec)
            else:
                # Fall back to the current measurement-derived long-axis direction.
                out_vec = direction_vec
            dir_out = unit_vec_to_angle_deg(out_vec)
            last_valid_direction[tid] = dir_out
        else:
            dir_out = None
        last_valid_obb[tid] = pts.copy()
        if abs_fid is not None:
            last_valid_frame[tid] = int(abs_fid)
        miss_count[tid] = 0
        return dir_out

    # Initialize both directional passes from the exact same S0 assignment.
    start_idx = 0
    abs_fid = int(initial_frame)
    assert processing_order[start_idx] == abs_fid
    if len(initial_assignment) != num_objects:
        raise ValueError('Initial assignment must contain exactly num_objects detections.')
    for expected_tid, (tid, initial_obb, initial_score, init_dir) in enumerate(initial_assignment):
        if int(tid) != expected_tid:
            raise ValueError('Initial assignment track IDs must be contiguous and ordered.')
        pts = initial_obb.copy()
        dir_out = update_filters_with_measurement(tid, pts, init_dir, reset=True, abs_fid=abs_fid)
        write_measurement(tid, abs_fid, pts, dir_out, initial_score, ATYPE_INIT, initial_score)
        remember_reference(tid, abs_fid, pts)

    print(f'{"BWD_S0" if reverse else "S0"}={abs_fid}')

    # Central-frame direction flip correction state
    _dfc_hist: list[list] = [[] for _ in range(num_objects)]
    _dfc_count: int = 0

    # Main tracking loop
    for processing_idx, abs_fid in enumerate(
        tqdm_it(processing_order[start_idx + 1:], desc=f'ID tracking ({direction_label})', unit='frame'),
        start=start_idx + 1,
    ):
        yolo_obbs, yolo_scores, yolo_classes = detection_frames[abs_fid]
        predicted_obbs = []
        for tid in range(num_objects):
            if position_kfs[tid] is not None:
                position_kfs[tid].predict()
            if bbox_shape_kfs[tid] is not None:
                bbox_shape_kfs[tid].predict()
            if direction_kfs[tid] is not None:
                direction_kfs[tid].predict()
            if last_valid_obb[tid] is not None:
                predicted_obbs.append(predict_obb_from_filters(last_valid_obb[tid], position_kfs[tid], bbox_shape_kfs[tid], direction_kfs[tid]))
            else:
                predicted_obbs.append(None)
        # KF-extrapolated direction for pred reference (after predict step)
        kf_pred_dirs = []
        for tid in range(num_objects):
            if direction_kfs[tid] is None:
                kf_pred_dirs.append(None)
                continue
            kf_vec = direction_vec_from_kf(direction_kfs[tid])
            ref_obb = predicted_obbs[tid] if valid_obb(predicted_obbs[tid]) else last_valid_obb[tid]
            if valid_obb(ref_obb) and is_long_axis_compatible(ref_obb, kf_vec, max_axis_error_deg=67.5):
                kf_pred_dirs.append(unit_vec_to_angle_deg(
                    closest_long_axis_direction(ref_obb, kf_vec)
                ))
            else:
                kf_pred_dirs.append(last_valid_direction[tid])

        det_of_id = [None] * num_objects
        used_dets: set[int] = set()
        assign_info: dict[int, tuple[float, float]] = {}
        # Tracks with miss_count < MAX_AGE participate in IoU stages (S1-S4).
        active_ids = [tid for tid in range(num_objects) if predicted_obbs[tid] is not None and miss_count[tid] < max_age]
        all_dets = list(range(len(yolo_obbs)))

        # Confirm each detection direction once via OBB long-axis + heading reference.
        if len(yolo_obbs) > 0:
            yolo_directions = np.array([
                unit_vec_to_angle_deg(closest_long_axis_direction(yolo_obbs[i], float(yolo_classes[i])))
                for i in range(len(yolo_obbs))
            ], dtype=float)
        else:
            yolo_directions = np.empty((0,), dtype=float)

        def apply_iou_stage(
            stage: float,
            ids: list[int],
            dets: list[int],
            require_iou_gate: bool,
            require_dir_gate: bool,
            add_dir_cost: bool,
            direction_cost_mode: str,
            ref_yolo_obbs: np.ndarray | None = None,
            ref_yolo_dirs: np.ndarray | None = None,
        ) -> None:
            if not ids or not dets:
                return
            _yobbs = ref_yolo_obbs if ref_yolo_obbs is not None else yolo_obbs
            _ydirs = ref_yolo_dirs if ref_yolo_dirs is not None else yolo_directions
            det_obbs = np.asarray([_yobbs[d] for d in dets], dtype=np.float32)
            det_dirs = np.asarray([_ydirs[d] for d in dets], dtype=float)
            pred = [predicted_obbs[tid] for tid in ids]
            lv = [last_valid_obb[tid] for tid in ids]
            mc = [miss_count[tid] for tid in ids]
            pd = [last_valid_direction[tid] for tid in ids]
            pdpred = [kf_pred_dirs[tid] for tid in ids]
            local_det_of_id, local_asg = assign_ids_iou_mode(
                pred, lv, mc, det_obbs, det_dirs, pd,
                strong_iou, strong_dir_deg, max_age,
                require_iou_gate=require_iou_gate,
                require_dir_gate=require_dir_gate,
                add_dir_cost=add_dir_cost,
                direction_cost_mode=direction_cost_mode,
                iou_weight=_iou_weight,
                direction_weight=_direction_weight,
                miss_weight=_miss_weight,
                pred_directions=pdpred,
            )
            for local_tid, local_didx in enumerate(local_det_of_id):
                if local_didx is None:
                    continue
                tid = ids[local_tid]
                didx = dets[local_didx]
                if det_of_id[tid] is None and didx not in used_dets:
                    det_of_id[tid] = int(didx)
                    used_dets.add(int(didx))
                    assign_info[tid] = (stage, float(local_asg[local_tid][1]))

        # S1-S4: main IoU detections, progressively loosening gates
        # (best-of{predicted, last_valid}).
        for stage_num, req_iou, req_dir, add_dir, dir_mode in (
            (ATYPE_S1, True,  True,  True, 'heading'),  # S1: strong IoU + heading gate/cost
            (ATYPE_S2, True,  False, True, 'axis'),     # S2: strong IoU + axis cost
            (ATYPE_S3, False, True,  True, 'heading'),  # S3: IoU>0 + heading gate/cost
            (ATYPE_S4, False, False, True, 'axis'),     # S4: IoU>0 + axis cost
        ):
            remaining_ids = [tid for tid in active_ids if det_of_id[tid] is None]
            remaining_dets = [d for d in all_dets if d not in used_dets]
            apply_iou_stage(stage_num, remaining_ids, remaining_dets, req_iou, req_dir, add_dir, dir_mode)

        # S5: active-track velocity-gated distance recovery.
        # IoU stages fail when a fast object's displacement exceeds its OBB size
        # (IoU=0) even though the detection is present.  This matches such tracks
        # on the next frame instead of waiting MAX_AGE for the dormant S6 path.
        remaining_ids = [tid for tid in active_ids if det_of_id[tid] is None]
        remaining_dets = [d for d in all_dets if d not in used_dets]
        if remaining_ids and remaining_dets:
            kf_speeds = [
                float(np.linalg.norm(position_kfs[tid].x[2:4]))
                if position_kfs[tid] is not None else 0.0
                for tid in remaining_ids
            ]
            det_obbs_g = np.asarray([yolo_obbs[d] for d in remaining_dets], dtype=np.float32)
            local_det_of_id, local_asg = assign_ids_gated_distance_mode(
                [predicted_obbs[tid] for tid in remaining_ids],
                [last_valid_obb[tid] for tid in remaining_ids],
                kf_speeds,
                [miss_count[tid] for tid in remaining_ids],
                det_obbs_g, max_age,
                distance_weight=_distance_weight,
                miss_weight=_miss_weight,
            )
            for local_tid, local_didx in enumerate(local_det_of_id):
                if local_didx is None:
                    continue
                tid = remaining_ids[local_tid]
                didx = remaining_dets[local_didx]
                if det_of_id[tid] is None and didx not in used_dets:
                    det_of_id[tid] = int(didx)
                    used_dets.add(int(didx))
                    assign_info[tid] = (ATYPE_VEL_DIST, float(local_asg[local_tid][1]))

        # S6: dormant-track NN recovery (miss_count >= MAX_AGE).
        # Uses lagged reference OBB (MAX_AGE steps back) with scale-normalized distance + age cost.
        nn_main_dets = [d for d in all_dets if d not in used_dets]
        dormant_ids = [
            tid for tid in range(num_objects)
            if last_valid_obb[tid] is not None
            and det_of_id[tid] is None
            and miss_count[tid] >= max_age
        ]
        if dormant_ids:
            lag_index = int(processing_idx) - int(max_age)
            lag_frame = processing_order[lag_index] if lag_index >= 0 else None
            lagged_refs = []
            for tid in dormant_ids:
                ref = None if lag_frame is None else reference_obb_history[tid].get(int(lag_frame))
                lagged_refs.append(ref if valid_obb(ref) else last_valid_obb[tid])

            if nn_main_dets:
                nn_pool_obbs = np.asarray([yolo_obbs[d] for d in nn_main_dets], dtype=np.float32)
                nn_mc = [miss_count[tid] for tid in dormant_ids]
                stage5_det_of_id, stage5_asg = assign_ids_distance_mode(
                    lagged_refs, nn_mc, nn_pool_obbs, max_age,
                    distance_weight=_distance_weight, miss_weight=_miss_weight)
                for local_tid, local_pool_didx in enumerate(stage5_det_of_id):
                    if local_pool_didx is None:
                        continue
                    tid = dormant_ids[local_tid]
                    val = float(stage5_asg[local_tid][1])
                    real_didx = nn_main_dets[local_pool_didx]
                    if det_of_id[tid] is None and real_didx not in used_dets:
                        det_of_id[tid] = int(real_didx)
                        used_dets.add(int(real_didx))
                        assign_info[tid] = (ATYPE_GAP_DET, val)

        # Track-level duplicate suppression over committed OBBs after ID assignment.
        # This is intentionally separate from detection-stage NMS: it resolves cases
        # where two IDs end up committed to overlapping OBBs.
        if overlap_guard_enabled:
            def _committed_obb_info(tid: int) -> tuple[np.ndarray, float, tuple[float, float, float, float]] | None:
                if det_of_id[tid] is None:
                    return None
                raw = yolo_obbs[det_of_id[tid]]
                pts = ensure_clockwise(raw)
                area = obb_polygon_area(pts)
                aabb = (
                    float(pts[:, 0].min()),
                    float(pts[:, 1].min()),
                    float(pts[:, 0].max()),
                    float(pts[:, 1].max()),
                )
                return pts, area, aabb

            def _quality_key(tid: int) -> tuple[float, float]:
                # Lower is better: (stage_rank, -score).
                stage = float(assign_info.get(tid, (99.0, 0.0))[0])
                score = float(yolo_scores[det_of_id[tid]])
                return (stage, -score)

            def _prepared_iou_gt(
                a: tuple[np.ndarray, float, tuple[float, float, float, float]],
                b: tuple[np.ndarray, float, tuple[float, float, float, float]],
                threshold: float,
            ) -> bool:
                if a[1] <= 0.0 or b[1] <= 0.0:
                    return False
                aa = a[2]
                ba = b[2]
                if aa[2] <= ba[0] or ba[2] <= aa[0] or aa[3] <= ba[1] or ba[3] <= aa[1]:
                    return False
                inter_area, _ = cv2.intersectConvexConvex(a[0], b[0])
                inter_area = float(inter_area)
                union = float(a[1]) + float(b[1]) - inter_area
                return bool(union > 0.0 and (inter_area / union) > threshold)

            committed_info = {
                t: info
                for t in range(num_objects)
                for info in (_committed_obb_info(t),)
                if info is not None
            }
            committed = list(committed_info)
            active_committed = set(committed)
            quality_keys = {t: _quality_key(t) for t in committed}
            i = 0
            while i < len(committed):
                ta = committed[i]
                if ta not in active_committed:
                    i += 1
                    continue
                oa = committed_info[ta]
                demoted_a = False
                for j in range(i + 1, len(committed)):
                    tb = committed[j]
                    if tb not in active_committed:
                        continue
                    ob = committed_info[tb]
                    if _prepared_iou_gt(oa, ob, overlap_guard_iou):
                        loser = tb if quality_keys[tb] >= quality_keys[ta] else ta
                        loser_didx = det_of_id[loser]
                        det_of_id[loser] = None
                        if loser_didx is not None:
                            used_dets.discard(int(loser_didx))
                        active_committed.discard(loser)
                        if loser == ta:
                            demoted_a = True
                            break
                if not demoted_a:
                    i += 1

        _dfc_cur: dict[int, tuple | None] = {}

        for tid in range(num_objects):
            didx = det_of_id[tid]
            if didx is None:
                _rollback_unconfirmed_fast_dist_before_missing(tid, processing_idx)
                if last_valid_obb[tid] is not None:
                    miss_count[tid] += 1
                write_nan(tid, abs_fid)
                remember_reference(tid, abs_fid, predicted_obbs[tid])
                _dfc_cur[tid] = None
                _reset_sharp_turn(tid)
                continue
            pts = yolo_obbs[didx].copy()
            assign_type, assign_val = assign_info.get(tid, (ATYPE_S1, np.nan))
            is_vel_dist = bool(assign_type == ATYPE_VEL_DIST)
            # is_risky: stages where the match may be a spurious excursion.
            # S5/fast-distance and S6/long-gap dormant recovery both qualify.
            is_risky = is_vel_dist or bool(assign_type == ATYPE_GAP_DET)
            # Retroactive gap backfill: skip for S5/fast-distance - the gap frames are
            # hard to fill reliably when IoU continuity is already broken.
            if miss_count[tid] > 0 and last_valid_frame[tid] is not None and not is_vel_dist:
                backfill_gap(tid, abs_fid, pts)
            # Excursion correction: two consecutive risky matches (S5/fast-distance or
            # S6/long-gap, in any combination) within max_age frames where the 2nd
            # is closer to the pre-gap position than the 1st -> the 1st was a
            # spurious detection; erase it.
            # State is saved BEFORE update_filters so last_valid_obb still
            # reflects the position that preceded the current risky match.
            if is_risky:
                prev_vd = vel_dist_prev[tid]
                erase_prev = False
                if prev_vd is not None:
                    gap = processing_idx - prev_vd['proc_idx'] - 1
                    f0_pts = prev_vd['prev_valid_pts']
                    if valid_obb(f0_pts) and 0 <= gap <= max_age:
                        f0_center = obb_center(f0_pts)
                        dist_cur  = float(np.linalg.norm(obb_center(pts) - f0_center))
                        dist_prev = float(np.linalg.norm(obb_center(prev_vd['vd_pts']) - f0_center))
                        if dist_cur < dist_prev:
                            write_nan(tid, int(prev_vd['vd_frame']), history_type=ATYPE_MISSING)
                            reference_obb_history[tid].pop(int(prev_vd['vd_frame']), None)
                            erase_prev = True
                if erase_prev:
                    f0_for_next = prev_vd['prev_valid_pts']
                elif valid_obb(last_valid_obb[tid]):
                    f0_for_next = last_valid_obb[tid].copy()
                else:
                    f0_for_next = None
                anchor_frame = last_valid_frame[tid]
                anchor_atype = (
                    _buffer_scalar(assign_type_buf, tid, int(anchor_frame))
                    if anchor_frame is not None else np.nan
                )
                trusted_anchor = (
                    np.isfinite(anchor_atype)
                    and int(round(float(anchor_atype))) in FAST_DIST_TRUSTED_ANCHOR_ATYPES
                )
                vel_dist_prev[tid] = {
                    'proc_idx': processing_idx,
                    'vd_frame': abs_fid,
                    'vd_pts': pts.copy(),
                    'prev_valid_pts': f0_for_next,
                    'prev_valid_direction': last_valid_direction[tid],
                    'prev_valid_frame': anchor_frame,
                    'prev_miss_count': int(miss_count[tid]),
                    'prev_position_kf': copy.deepcopy(position_kfs[tid]),
                    'prev_bbox_shape_kf': copy.deepcopy(bbox_shape_kfs[tid]),
                    'prev_direction_kf': copy.deepcopy(direction_kfs[tid]),
                    'prev_assign_type': float(anchor_atype) if np.isfinite(anchor_atype) else np.nan,
                    'trusted_anchor': bool(trusted_anchor),
                }
            else:
                vel_dist_prev[tid] = None
            det_dir = float(yolo_directions[didx]) if len(yolo_directions) > didx else None
            det_score = float(yolo_scores[didx])
            reset = bool(assign_type == ATYPE_GAP_DET and miss_count[tid] >= max_age)
            iou_val = (
                float(assign_val)
                if assign_type in (ATYPE_S1, ATYPE_S2, ATYPE_S3, ATYPE_S4)
                else None
            )
            distance_val = None if iou_val is not None else float(assign_val)
            if int(assign_type) in UNRELIABLE_DIR_ATYPES:
                # Sharp-turn detection: if an OVERLAP-group stage produces a
                # reversed direction for SHARP_TURN_ACCEPT_RUN consecutive frames,
                # commit those frames with the new heading instead of NaN.
                _do_sharp = (
                    int(assign_type) in UNRELIABLE_DIR_OVERLAP
                    and det_dir is not None
                    and np.isfinite(det_dir)
                    and last_valid_direction[tid] is not None
                    and np.isfinite(float(last_valid_direction[tid]))
                    and angular_distance_deg(last_valid_direction[tid], det_dir) > strong_dir_deg
                )
                if _do_sharp:
                    _sh = sharp_turn_heading[tid]
                    if _sh is not None and angular_distance_deg(_sh, det_dir) > strong_dir_deg:
                        # Consistency broken: restart run at current frame
                        sharp_turn_count[tid] = 1
                        sharp_turn_buf[tid] = [(abs_fid, pts.copy(), det_dir)]
                        sharp_turn_heading[tid] = det_dir
                    elif _sh is None:
                        sharp_turn_count[tid] = 1
                        sharp_turn_buf[tid] = [(abs_fid, pts.copy(), det_dir)]
                        sharp_turn_heading[tid] = det_dir
                    else:
                        sharp_turn_count[tid] += 1
                        sharp_turn_buf[tid].append((abs_fid, pts.copy(), det_dir))
                        sharp_turn_heading[tid] = det_dir
                    if sharp_turn_count[tid] >= sharp_turn_accept_run:
                        # Backfill all prior frames in the run (already written as NaN)
                        for fid_k, pts_k, det_dir_k in sharp_turn_buf[tid][:-1]:
                            vec_k = closest_long_axis_direction(pts_k, det_dir_k)
                            angle_k = unit_vec_to_angle_deg(vec_k)
                            class_buf.setdefault(tid, {})[fid_k] = (angle_k,)
                            direction_corrected_buf.setdefault(tid, {})[fid_k] = (1.0,)
                            assign_type_history_buf.setdefault(tid, {}).setdefault(fid_k, []).append(
                                (_src, float(ATYPE_TURN_ACCEPT)))
                        # Accept the current (final) frame with the new heading
                        vec_cur = closest_long_axis_direction(pts, det_dir)
                        angle_cur = unit_vec_to_angle_deg(vec_cur)
                        direction_kfs[tid] = create_direction_kf(vec_cur, dt=kf_dt)
                        dir_out = update_filters_with_measurement(
                            tid, pts, angle_cur, reset=reset, abs_fid=abs_fid)
                        write_measurement(
                            tid, abs_fid, pts, dir_out, det_score, float(assign_type),
                            float(assign_val), iou_val=iou_val, distance_val=distance_val,
                            direction_corrected=True)
                        assign_type_history_buf.setdefault(tid, {}).setdefault(abs_fid, []).append(
                            (_src, float(ATYPE_TURN_ACCEPT)))
                        remember_reference(tid, abs_fid, pts)
                        _dfc_cur[tid] = None
                        _reset_sharp_turn(tid)
                        continue
                else:
                    _reset_sharp_turn(tid)
                # OVERLAP group: KF updated with nearest-axis for tracking freshness.
                # DISTANCE group: KF frozen to prevent contamination by a wrong-object axis.
                _freeze = int(assign_type) in UNRELIABLE_DIR_DISTANCE
                update_filters_with_measurement(tid, pts, last_valid_direction[tid], reset=reset, abs_fid=abs_fid, freeze_direction=_freeze)
                write_measurement(tid, abs_fid, pts, None, det_score, float(assign_type), float(assign_val), iou_val=iou_val, distance_val=distance_val)
                assign_type_history_buf.setdefault(tid, {}).setdefault(abs_fid, []).append((_src, float(ATYPE_DIR_NAN)))
                remember_reference(tid, abs_fid, pts)
                _dfc_cur[tid] = None
                continue
            _reset_sharp_turn(tid)
            _dfc_kfx = direction_kfs[tid].x.copy() if direction_kfs[tid] is not None and not reset else None
            _dfc_kfP = direction_kfs[tid].P.copy() if direction_kfs[tid] is not None and not reset else None
            dir_out = update_filters_with_measurement(tid, pts, det_dir, reset=reset, abs_fid=abs_fid)
            write_measurement(tid, abs_fid, pts, dir_out, det_score, float(assign_type), float(assign_val), iou_val=iou_val, distance_val=distance_val)
            remember_reference(tid, abs_fid, pts)
            _raw_h = det_dir if det_dir is not None and np.isfinite(det_dir) else None
            _dfc_cur[tid] = (_raw_h, pts, _dfc_kfx, _dfc_kfP, reset) if _raw_h is not None else None

        # Central-frame direction flip correction (3-frame window)
        for tid in range(num_objects):
            cur = _dfc_cur.get(tid)
            hist = _dfc_hist[tid]
            if cur is None:
                hist.clear()
                continue
            raw_t1, pts_t1, kfx_t1, kfP_t1, reset_t1 = cur
            new_kfx_t1, new_kfP_t1 = kfx_t1, kfP_t1
            raw_t1_for_hist = raw_t1
            if len(hist) >= 2:
                _fid_tm1, raw_tm1, _pts_tm1, _kfx_tm1, _kfP_tm1, reset_tm1 = hist[-2]
                fid_t, raw_t, pts_t, kf_x_t, kf_P_t, reset_t = hist[-1]
                if (not reset_tm1 and not reset_t
                        and kf_x_t is not None and kf_P_t is not None):
                    dist_tm1_t = angular_distance_deg(raw_tm1, raw_t)
                    dist_t_t1 = angular_distance_deg(raw_t, raw_t1)
                    if dist_tm1_t > strong_dir_deg and dist_t_t1 > strong_dir_deg:
                        df = (raw_t + 180.0) % 360.0
                        C0 = dist_tm1_t + dist_t_t1
                        C1 = angular_distance_deg(raw_tm1, df) + angular_distance_deg(df, raw_t1)
                        if C1 < C0:
                            _dfc_count += 1
                            direction_kfs[tid].x = kf_x_t.copy()
                            direction_kfs[tid].P = kf_P_t.copy()
                            flipped_vec = closest_long_axis_direction(pts_t, df)
                            direction_kfs[tid].update(flipped_vec.astype(float))
                            kf_vec_t = direction_vec_from_kf(direction_kfs[tid], fallback=flipped_vec)
                            if is_long_axis_compatible(pts_t, kf_vec_t, max_axis_error_deg=67.5):
                                out_vec_t = closest_long_axis_direction(pts_t, kf_vec_t)
                            else:
                                out_vec_t = flipped_vec
                            kf_dir_t = unit_vec_to_angle_deg(out_vec_t)
                            direction_kfs[tid].predict()
                            new_kfx_t1 = direction_kfs[tid].x.copy()
                            new_kfP_t1 = direction_kfs[tid].P.copy()
                            if not reset_t1:
                                vec_t1 = closest_long_axis_direction(pts_t1, raw_t1)
                                direction_kfs[tid].update(vec_t1.astype(float))
                                kf_vec_t1 = direction_vec_from_kf(direction_kfs[tid], fallback=vec_t1)
                                if is_long_axis_compatible(pts_t1, kf_vec_t1, max_axis_error_deg=67.5):
                                    out_vec_t1 = closest_long_axis_direction(pts_t1, kf_vec_t1)
                                else:
                                    out_vec_t1 = vec_t1
                                kf_dir_t1 = unit_vec_to_angle_deg(out_vec_t1)
                                last_valid_direction[tid] = kf_dir_t1
                                class_buf[tid][abs_fid] = (kf_dir_t1,)
                                direction_corrected_buf[tid][abs_fid] = (1.0,)
                                assign_type_history_buf.setdefault(tid, {}).setdefault(abs_fid, []).append((_src, float(ATYPE_DIR_FLIP)))
                                raw_t1_for_hist = kf_dir_t1
                            else:
                                direction_kfs[tid] = create_direction_kf(
                                    closest_long_axis_direction(pts_t1, raw_t1), dt=kf_dt)
                                new_kfx_t1, new_kfP_t1 = None, None
                            class_buf[tid][fid_t] = (kf_dir_t,)
                            direction_corrected_buf[tid][fid_t] = (1.0,)
                            assign_type_history_buf.setdefault(tid, {}).setdefault(fid_t, []).append((_src, float(ATYPE_DIR_FLIP)))
                            hist[-1] = (fid_t, kf_dir_t, pts_t, kf_x_t, kf_P_t, reset_t)
            hist.append((abs_fid, raw_t1_for_hist, pts_t1, new_kfx_t1, new_kfP_t1, reset_t1))
            if len(hist) > 2:
                del hist[0]

    print(f'direction flip corrections: {_dfc_count}')
    print(f'{"Backward" if reverse else "Forward"} tracking finished.')

    return {
        'pos_buf': pos_buf,
        'obb_buf': obb_buf,
        'class_buf': class_buf,
        'score_buf': score_buf,
        'assign_type_buf': assign_type_buf,
        'assign_type_history_buf': assign_type_history_buf,
        'assign_val_buf': assign_val_buf,
        'assign_total_buf': assign_total_buf,
        'assign_iou_cost_buf': assign_iou_cost_buf,
        'assign_direction_cost_buf': assign_direction_cost_buf,
        'assign_distance_cost_buf': assign_distance_cost_buf,
        'assign_score_cost_buf': assign_score_cost_buf,
        'obb_source_buf': obb_source_buf,
        'obb_corrected_buf': obb_corrected_buf,
        'direction_corrected_buf': direction_corrected_buf,
        'switch_corrected_buf': switch_corrected_buf,
    }


def _merge_directional_passes(
    forward_buffers: dict,
    backward_buffers: dict,
    initial_frame: int,
) -> dict:
    """Merge backward frames before S0 with forward frames from S0 onward."""
    buffer_names = (
        'pos_buf',
        'obb_buf',
        'class_buf',
        'score_buf',
        'assign_type_buf',
        'assign_type_history_buf',
        'assign_val_buf',
        'assign_total_buf',
        'assign_iou_cost_buf',
        'assign_direction_cost_buf',
        'assign_distance_cost_buf',
        'assign_score_cost_buf',
        'obb_source_buf',
        'obb_corrected_buf',
        'direction_corrected_buf',
        'switch_corrected_buf',
    )
    merged: dict[str, dict] = {}
    for name in buffer_names:
        out: dict[int, dict] = {}
        for source, use_frame in (
            (backward_buffers.get(name, {}), lambda fid: int(fid) < int(initial_frame)),
            (forward_buffers.get(name, {}), lambda fid: int(fid) >= int(initial_frame)),
        ):
            for tid, per_frame in source.items():
                dest = out.setdefault(int(tid), {})
                for fid, value in per_frame.items():
                    if use_frame(fid):
                        dest[int(fid)] = list(value) if name == 'assign_type_history_buf' else value
        merged[name] = out
    return merged


def track_from_detection_frames(
    detection_frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_objects: int,
    nms_iou: float,
    max_age: int,
    strong_iou: float,
    strong_dir_deg: float,
    tracking_cost: dict | None = None,
    sharp_turn_accept_run: int = 3,
):
    """Track from one chronological S0 in both directions and merge the result."""
    frame_ids = sorted(int(f) for f in detection_frames)
    if not frame_ids:
        raise RuntimeError('No detection frames found.')

    initial_frame, selected_order = _select_initial_assignment(
        detection_frames, num_objects)
    initial_obbs, initial_scores, initial_classes = detection_frames[initial_frame]
    initial_assignment = tuple(
        (
            int(tid),
            initial_obbs[int(didx)].copy(),
            float(initial_scores[int(didx)]),
            unit_vec_to_angle_deg(closest_long_axis_direction(
                initial_obbs[int(didx)], float(initial_classes[int(didx)])
            )),
        )
        for tid, didx in enumerate(selected_order.tolist())
    )
    common_kw = dict(
        detection_frames=detection_frames,
        num_objects=num_objects,
        nms_iou=nms_iou,
        max_age=max_age,
        strong_iou=strong_iou,
        strong_dir_deg=strong_dir_deg,
        initial_frame=initial_frame,
        initial_assignment=initial_assignment,
        tracking_cost=tracking_cost,
        sharp_turn_accept_run=sharp_turn_accept_run,
    )

    forward_buffers = _track_detection_pass(reverse=False, **common_kw)
    backward_buffers = _track_detection_pass(reverse=True, **common_kw)

    # Both passes must start from byte-for-byte equivalent S0 observations.
    for tid in range(num_objects):
        assert forward_buffers['obb_buf'][tid][initial_frame] == backward_buffers['obb_buf'][tid][initial_frame]
        assert forward_buffers['score_buf'][tid][initial_frame] == backward_buffers['score_buf'][tid][initial_frame]

    buffers = _merge_directional_passes(
        forward_buffers=forward_buffers,
        backward_buffers=backward_buffers,
        initial_frame=initial_frame,
    )

    # Resolve heading gaps once over the complete chronological trajectory so a
    # NaN run can use anchors on opposite sides of S0.
    _resolve_nan_direction_runs(
        class_buf=buffers['class_buf'],
        obb_buf=buffers['obb_buf'],
        direction_corrected_buf=buffers['direction_corrected_buf'],
        assign_type_history_buf=buffers['assign_type_history_buf'],
        frame_ids=frame_ids,
        source_label='merged',
    )

    expected_frames = set(frame_ids)
    for tid in range(num_objects):
        assert set(buffers['pos_buf'].get(tid, {})) == expected_frames
        assert initial_frame in buffers['pos_buf'][tid]
        assert initial_frame in buffers['assign_type_buf'][tid]
    return buffers


def save_tracking_outputs(
    out_dir: str,
    buffers: dict,
    num_objects: int,
    save_csv: bool = True,
    artifact_suffix: str = FWD_ARTIFACT_SUFFIX,
    return_dataframes: bool = False,
    save_assign_type_pickle: bool = True,
    save_provenance_csv: bool = True,
    save_resume_pickle: bool = True,
) -> dict[str, pd.DataFrame] | None:
    """Write internal tracking state and final OBB/direction CSVs.

    Forward tracking keeps only internal artifacts needed by refinement. User-facing
    OBB/direction CSVs are written only after refinement: unsuffixed for the
    standard final result and ``_id_resolved`` when contrastive identity
    resolution ran. Position data remains in the tracking buffers and any
    requested in-memory dataframe/final result exports.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, BUFFER_DIR_NAME), exist_ok=True)
    if save_provenance_csv:
        require_provenance_buffers(buffers, num_objects)

    final_csv_suffix = None
    if artifact_suffix == 'filled':
        final_csv_suffix = ''
    elif artifact_suffix == 'id_resolved':
        final_csv_suffix = '_id_resolved'
    write_final_csv = bool(save_csv and final_csv_suffix is not None)

    dfs: dict[str, pd.DataFrame] = {}
    if return_dataframes:
        pos_df = to_df(buffers['pos_buf'], ['x', 'y'], False, num_objects)
        dfs['pos'] = pos_df
        del pos_df

    if write_final_csv or return_dataframes:
        obb_df = obb_buf_to_df(buffers['obb_buf'], False, num_objects)
        if write_final_csv:
            obb_df.to_csv(
                os.path.join(out_dir, f'obbs{final_csv_suffix}.csv'),
                index=False,
                float_format='%.3f',
            )
        if return_dataframes:
            dfs['obb'] = obb_df
        del obb_df

        class_df = to_df(buffers['class_buf'], ['c'], False, num_objects)
        if write_final_csv:
            class_df.to_csv(
                os.path.join(out_dir, f'directions{final_csv_suffix}.csv'),
                index=False,
                float_format='%.6f',
            )
        if return_dataframes:
            dfs['class'] = class_df
        del class_df

    if save_assign_type_pickle:
        assign_type_df = to_df(buffers['assign_type_buf'], ['t'], False, num_objects)
        assign_type_df.to_pickle(_artifact_path(out_dir, ASSIGN_TYPE_PICKLE_NAME, artifact_suffix))
        if return_dataframes:
            dfs['assign_type'] = assign_type_df
        del assign_type_df
        hist = buffers.get('assign_type_history_buf', {})
        with open(_artifact_path(out_dir, ASSIGN_TYPE_HISTORY_PICKLE_NAME, artifact_suffix), 'wb') as _fh:
            pickle.dump(hist, _fh, protocol=pickle.HIGHEST_PROTOCOL)

    if save_provenance_csv:
        provenance_parts = [
            to_df(buffers['obb_source_buf'], ['os'], False, num_objects),
            to_df(buffers['obb_corrected_buf'], ['oc'], False, num_objects),
            to_df(buffers['direction_corrected_buf'], ['dc'], False, num_objects),
            to_df(buffers['switch_corrected_buf'], ['sc'], False, num_objects),
        ]
        base_position = provenance_parts[0]['position'] if 'position' in provenance_parts[0] else None
        if (
            base_position is not None
            and all('position' in part and part['position'].equals(base_position) for part in provenance_parts[1:])
        ):
            provenance_df = pd.concat(
                [provenance_parts[0]] + [part.drop(columns='position') for part in provenance_parts[1:]],
                axis=1,
            )
        else:
            provenance_df = provenance_parts[0].copy()
            for extra in provenance_parts[1:]:
                provenance_df = provenance_df.merge(extra, on='position', how='outer', sort=True)
        provenance_df.to_csv(
            _artifact_path(out_dir, PROVENANCE_CSV_NAME, artifact_suffix),
            index=False,
            float_format='%.0f',
        )
        del provenance_parts, provenance_df

    if save_resume_pickle:
        with open(_artifact_path(out_dir, BUFFERS_PICKLE_NAME, artifact_suffix), 'wb') as _f:
            pickle.dump(buffers, _f, protocol=pickle.HIGHEST_PROTOCOL)

    if return_dataframes:
        return dfs
    return None



def required_tracking_outputs(
    out_dir: str,
    artifact_suffix: str = FWD_ARTIFACT_SUFFIX,
    include_csv: bool = True,
    include_assign_type: bool = True,
    include_provenance: bool = True,
    include_resume_pickle: bool = True,
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if include_resume_pickle:
        result.append(('pickle', _artifact_path(out_dir, BUFFERS_PICKLE_NAME, artifact_suffix)))
    if include_assign_type:
        result.append(('pickle', _artifact_path(out_dir, ASSIGN_TYPE_PICKLE_NAME, artifact_suffix)))
    if include_provenance:
        result.append(('csv', _artifact_path(out_dir, PROVENANCE_CSV_NAME, artifact_suffix)))
    if include_csv and artifact_suffix in {'filled', 'id_resolved'}:
        suffix = '_id_resolved' if artifact_suffix == 'id_resolved' else ''
        result += [
            ('csv', os.path.join(out_dir, f'obbs{suffix}.csv')),
            ('csv', os.path.join(out_dir, f'directions{suffix}.csv')),
        ]
    return result



def output_has_data(kind: str, path: str) -> bool:
    if kind == 'pickle':
        return pickle_has_data(path)
    if kind == 'file':
        return os.path.exists(path) and os.path.getsize(path) > 0
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path)
    return not df.empty and len(df.columns) > 0




def tracking_analysis_outputs_exist(out_dir: str) -> bool:
    """Return True when forward tracking state already exists."""
    return all(
        output_has_data(kind, path)
        for kind, path in required_tracking_outputs(out_dir)
    )



def _run_single_tracking_job(spec: dict) -> None:
    """Process one (video_info, tracking_job) pair. Designed to run in a subprocess."""
    info          = spec['info']
    run_name      = spec['run_name']
    session_path  = spec['session_path']
    model_name    = spec['model_name']
    dataset_name  = spec['dataset_name']
    num_objects   = spec['num_objects']
    conf_th       = spec['conf_th']
    nms_iou       = spec['nms_iou']
    strong_iou    = spec['strong_iou']
    strong_dir_deg = spec['strong_dir_deg']
    max_age       = spec['max_age']
    tracking_cost_cfg = spec['tracking_cost_cfg']
    sharp_turn_accept_run = spec['sharp_turn_accept_run']
    export_final_result = bool(spec.get('export_final_result', False))

    out_dir = build_tracking_out_dir(
        session_path, model_name, dataset_name, run_name, info['name']
    )
    direction_pickle = _artifact_path(out_dir, 'all_blobs.pkl')
    if not pickle_has_data(direction_pickle):
        raise FileNotFoundError(f'Direction blobs pickle not found or empty: {direction_pickle}')

    fwd_req = required_tracking_outputs(out_dir)
    fwd_needed = not all(output_has_data(kind, p) for kind, p in fwd_req)

    if not fwd_needed or tracking_analysis_outputs_exist(out_dir):
        print(f"Tracking outputs already exist. Skipping. [{run_name}:{info['name']}]")
        if export_final_result:
            for suffix, family in (('id_resolved', 'id_resolved'), ('filled', 'filled')):
                buffer_path = _artifact_path(out_dir, BUFFERS_PICKLE_NAME, suffix)
                if not os.path.exists(buffer_path):
                    continue
                with open(buffer_path, 'rb') as f:
                    existing_buffers = pickle.load(f)
                save_final_result_csv(
                    session_path, model_name, dataset_name, run_name, info['name'],
                    existing_buffers, num_objects, family,
                )
                break
        return

    direction_rows_df = load_blob_pickle(direction_pickle)
    detection_frames = build_detection_frames(
        direction_rows_df, info['first_frame'], info['last_frame'], conf_th)

    common_kw = dict(
        num_objects=num_objects,
        nms_iou=nms_iou,
        max_age=max_age,
        strong_iou=strong_iou,
        strong_dir_deg=strong_dir_deg,
        tracking_cost=tracking_cost_cfg,
        sharp_turn_accept_run=sharp_turn_accept_run,
    )

    print(f"Running bidirectional tracking. [{run_name}:{info['name']}]")
    fwd_buffers = track_from_detection_frames(
        detection_frames=detection_frames,
        **common_kw,
    )
    save_tracking_outputs(out_dir, fwd_buffers, num_objects,
                          artifact_suffix=FWD_ARTIFACT_SUFFIX, save_csv=False)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python multi_staged_association.py config.yaml')
    cfg = load_config(sys.argv[1])
    if cfg.get('skip_id_tracking', False):
        print('skip_id_tracking is True; exiting.')
        return

    analysis = cfg.get('analysis', {}) or {}

    # -- Workers --
    workers_cfg = resolve_num_workers(cfg, 'id_tracking', default=None)
    num_workers = (
        _auto_num_workers('process')
        if workers_cfg is None
        else max(1, int(workers_cfg))
    )
    worker_mode = 'auto' if workers_cfg is None else 'configured'
    print(f'id_tracking: workers={num_workers} ({worker_mode})')

    # -- Paths & basic settings --
    session_path = str(cfg['SESSION_PATH'])
    video_path_in = str(cfg['TRACKING_VIDEO_PATH'])
    num_objects = int(cfg['NUM_OBJECTS'])

    # -- Tracking parameters --
    conf_th          = float(analysis.get('CONF', 0.1))
    orig_first_frame = int(analysis.get('FIRST_FRAME', 0))
    orig_last_frame  = int(analysis.get('LAST_FRAME', -1))

    nms_iou        = float(analysis.get('NMS_IOU', 0.80))
    strong_iou     = float(analysis.get('MATCH_IOU', 0.50))
    strong_dir_deg = float(analysis.get('MATCH_ANGLE',   90.0))
    max_age        = int(analysis.get('MAX_AGE', 10))
    tracking_cost_cfg = dict(analysis.get('TRACKING_COST', {}) or {})
    sharp_turn_accept_run = int(
        tracking_cost_cfg.get('SHARP_TURN_ACCEPT_RUN',
            analysis.get('SHARP_TURN_ACCEPT_RUN', 3))
    )

    # -- Model and checkpoint parameters --
    model_name = os.path.splitext(cfg.get('training', {}).get('PRETRAINED_MODEL', ''))[0]
    model_name = resolve_model_name(model_name).split('.')[0]
    dataset_name = resolve_existing_experiment_dir_name(
        session_path,
        model_name,
        experiment_dir_name_from_cfg(cfg),
        stages=('tracking', 'training'),
        warn_fn=print,
    )
    requested_direction_weights = list(dict.fromkeys(parse_weight_spec(analysis.get('WEIGHT', 'last'))))
    tracking_jobs = resolve_tracking_jobs_for_requested_weights(
        session_path,
        model_name,
        dataset_name,
        requested_direction_weights,
    )

    if os.path.isdir(video_path_in):
        video_files = [os.path.join(video_path_in, f) for f in os.listdir(video_path_in) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    else:
        video_files = [video_path_in]
    if not video_files:
        raise RuntimeError(f'Video not found: {video_path_in}')

    video_infos = []
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_info = read_video_frame_info(video_path)
        warn_if_frame_count_adjusted(frame_info, label=f'id_tracking video={video_name}')
        first_frame, last_frame = clamp_frame_range_to_usable_count(
            orig_first_frame, orig_last_frame, frame_info.usable_frame_count,
        )
        print(
            f'[INFO] id_tracking video={video_name}, usable_frames={frame_info.usable_frame_count}, '
            f'frame_range={first_frame}-{last_frame}'
        )
        video_infos.append({'path': video_path, 'name': video_name, 'first_frame': first_frame, 'last_frame': last_frame})

    specs = []
    for info in video_infos:
        for job in tracking_jobs:
            specs.append({
                'info': info,
                'run_name': str(job['run_name']),
                'session_path': session_path,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'num_objects': num_objects,
                'conf_th': conf_th,
                'nms_iou': nms_iou,
                'strong_iou': strong_iou,
                'strong_dir_deg': strong_dir_deg,
                'max_age': max_age,
                'tracking_cost_cfg': tracking_cost_cfg,
                'sharp_turn_accept_run': sharp_turn_accept_run,
                'export_final_result': bool(cfg.get('skip_refinement', False)),
            })

    if not specs:
        return

    n_epoch_jobs = len(specs)
    outer_epoch, _ = _worker_budget(num_workers, n_epoch_jobs)

    if outer_epoch <= 1:
        for spec in specs:
            _run_single_tracking_job(spec)
    else:
        print(
            f'id_tracking: {n_epoch_jobs} epoch job(s), '
            f'outer={outer_epoch} parallel'
        )
        with ProcessPoolExecutor(max_workers=outer_epoch) as pool:
            futures = [pool.submit(_run_single_tracking_job, spec) for spec in specs]
            for f in futures:
                f.result()


if __name__ == '__main__':
    main()
