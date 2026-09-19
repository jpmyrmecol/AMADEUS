# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Extract single-animal masks and label headings from movement trajectories."""

import csv
import itertools
import math
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import psutil
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from batch_utils import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gui.color import OBB_COLOR, OUTLIER_COLOR
from path_utils import resolve_config_paths
from video_frame_count import (
    clamp_frame_range_to_usable_count,
    read_video_frame_info,
    warn_if_frame_count_adjusted,
)

_torch = None
_F = None
_CUDA_AVAILABLE = False
_CUDA_BACKEND_CHECKED = False


DIRECTION_CLASS_NAMES = [
    "upper",
    "upper_right",
    "right",
    "lower_right",
    "lower",
    "lower_left",
    "left",
    "upper_left",
]

LOW_OBB_ASPECT_REASON = "low_obb_aspect"
SINGLE_ANIMAL_IMAGES_DIR_NAME = "single_animal_images"

# Donor material for interaction_image_synthesis.py is padded out to this ratio (see
# _extract_masked_blob_crop) so paste-time mask expansion -- randomized per
# paste between 1.0 and this value, see paste_blobs.random_mask_expansion_ratio
# -- always has real source pixels behind it. The single_animal_images frame
# itself and its background-infill (removal) mask always use the exact,
# unexpanded contour and never this ratio. Overridable via
# cfg["MASK_EXPANSION_RATIO"] (Advanced Tracking > Paste > Segmentation Mask).
DEFAULT_MASK_EXPANSION_RATIO = 1.05


def get_mask_expansion_ratio(cfg: dict) -> float:
    return float(cfg.get("MASK_EXPANSION_RATIO", DEFAULT_MASK_EXPANSION_RATIO))


def expand_contour_from_centroid(cnt: np.ndarray, ratio: float) -> np.ndarray:
    """Scale contour points outward from the centroid by ratio."""
    m = cv2.moments(cnt)
    if abs(m["m00"]) > 1e-8:
        cx, cy = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    else:
        pts0 = cnt.reshape(-1, 2).astype(np.float32)
        cx, cy = float(np.mean(pts0[:, 0])), float(np.mean(pts0[:, 1]))
    center = np.asarray([cx, cy], dtype=np.float64)
    pts = cnt.reshape(-1, 2).astype(np.float64)
    scaled = center + (pts - center) * float(ratio)
    return np.round(scaled).astype(np.int32).reshape(cnt.shape)


def _load_cuda_backend() -> bool:
    """Load PyTorch only when the optional GPU writer is explicitly requested."""
    global _torch, _F, _CUDA_AVAILABLE, _CUDA_BACKEND_CHECKED
    if _CUDA_BACKEND_CHECKED:
        return bool(_CUDA_AVAILABLE)
    _CUDA_BACKEND_CHECKED = True
    try:
        import torch as torch_module
        import torch.nn.functional as functional

        _torch = torch_module
        _F = functional
        _CUDA_AVAILABLE = bool(torch_module.cuda.is_available())
    except ImportError:
        _torch = None
        _F = None
        _CUDA_AVAILABLE = False
    return bool(_CUDA_AVAILABLE)


@dataclass
class BlobRecord:
    contour: np.ndarray
    rect: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area: float
    bbox_area: float
    w: float
    h: float
    pickle_outlier: bool
    frame: int
    blob_index: int
    traj_id: Optional[int] = None
    class_id: Optional[int] = None
    direction_vec: Optional[Tuple[float, float]] = None
    axis_length: Optional[float] = None

    area_outlier: bool = False
    bbox_area_outlier: bool = False
    width_outlier: bool = False
    height_outlier: bool = False
    obb_major_axis_len: Optional[float] = None
    obb_major_axis_outlier: bool = False
    obb_aspect_ratio: Optional[float] = None
    low_obb_aspect_outlier: bool = False
    jump_distance_px: Optional[float] = None
    jump_outlier: bool = False

    erase_stage1: bool = False
    erase_pre_direction: bool = False
    erase_final: bool = False

    erase_reasons_stage1: List[str] = field(default_factory=list)
    erase_reasons_pre_direction: List[str] = field(default_factory=list)
    erase_reasons_final: List[str] = field(default_factory=list)
    direction_failure_reasons: List[str] = field(default_factory=list)



def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return resolve_config_paths(cfg)


def resolve_num_workers(cfg: dict, section_name: str | None = None, default: Optional[int] = None, cap: Optional[int] = None) -> Optional[int]:
    """Resolve worker count from GUI/config settings.

    Priority:
      1. section-specific NUM_WORKERS / WORKERS / N_WORKERS / workers / num_workers
      2. top-level NUM_WORKERS / WORKERS / N_WORKERS / workers / num_workers
      3. default
    Returns None when no value/default is provided, allowing memory-aware auto mode.
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


def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def reset_without_crossing_output_dir(out_dir: str) -> None:
    import shutil
    keep_names = {"refine"}
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        return
    for name in os.listdir(out_dir):
        if name in keep_names:
            continue
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    # Delete refine/original/ so apply_refine_deletions uses the newly generated data.
    # refine/delete_blobs.csv (user's manual deletion marks) is preserved.
    original_dir = os.path.join(out_dir, "refine", "original")
    if os.path.isdir(original_dir):
        shutil.rmtree(original_dir)

    # refine/original was just invalidated, so any past apply-completion marker
    # is stale: it would make direction_class_filtering.py skip re-running
    # apply_class_label_filtering.py against the newly generated data.
    refine_dir = os.path.join(out_dir, "refine")
    if os.path.isdir(refine_dir):
        for name in os.listdir(refine_dir):
            if name.startswith(".apply_refine_deletions.") and name.endswith(".done"):
                os.remove(os.path.join(refine_dir, name))


def load_refine_delete_map(session_path: str) -> Dict[Tuple[int, int], List[str]]:
    path = os.path.join(session_path, SINGLE_ANIMAL_IMAGES_DIR_NAME, "refine", "delete_blobs.csv")
    out: Dict[Tuple[int, int], List[str]] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            blob_index = int(row["blob_index"])
            reasons = [x for x in str(row.get("reasons", "")).split("|") if x]
            out[(frame, blob_index)] = reasons or ["refine_delete"]
    return out


def apply_refine_deletions(blob_records: Dict[int, List[BlobRecord]], refine_delete_map: Dict[Tuple[int, int], List[str]]) -> Dict[str, int]:
    count = 0
    for fid, records in blob_records.items():
        for b in records:
            reasons = refine_delete_map.get((int(fid), int(b.blob_index)))
            if not reasons:
                continue
            b.erase_final = True
            merged = list(b.erase_reasons_final)
            for reason in reasons:
                append_unique_reason(merged, reason)
            b.erase_reasons_final = merged
            count += 1
    return {"refine_deleted": count}

def resolve_blob_sequence(obj):
    return obj.blobs_in_video


def get_frame_blobs(blob_seq, fid: int):
    return blob_seq[fid]


def get_num_blob_frames(blob_seq) -> int:
    return len(blob_seq)


def contour_area(cnt: np.ndarray) -> float:
    return float(abs(cv2.contourArea(cnt)))


def contour_center(cnt: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) > 1e-8:
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    pts = cnt.reshape(-1, 2).astype(np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def unit_vec(x: float, y: float) -> Optional[np.ndarray]:
    n = math.hypot(float(x), float(y))
    if n < 1e-8:
        return None
    return np.asarray([float(x) / n, float(y) / n], dtype=np.float32)


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


def angle_to_class(dx: float, dy: float) -> int:
    angle = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
    return int(((angle + 22.5) % 360.0) // 45.0)


def _obb_geometry_from_contour(cnt: np.ndarray) -> Tuple[Optional[np.ndarray], float, Optional[float], Optional[np.ndarray]]:
    pts = np.asarray(cnt, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None, 0.0, None, None
    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), angle = rect
    w = float(w)
    h = float(h)
    major_axis_len = float(max(w, h))
    long_side = float(max(w, h))
    short_side = float(min(w, h))

    axis = None
    if major_axis_len > 1e-8 and math.isfinite(float(angle)):
        axis_angle = float(angle) if w >= h else float(angle) + 90.0
        theta = math.radians(axis_angle)
        axis = unit_vec(math.cos(theta), math.sin(theta))

    aspect_ratio = None
    if long_side > 1e-6:
        aspect_ratio = float("inf") if short_side <= 1e-6 else long_side / short_side

    return axis, major_axis_len, aspect_ratio, cv2.boxPoints(rect).astype(np.float32)


def obb_axis_from_contour(cnt: np.ndarray) -> Optional[np.ndarray]:
    axis, _, _, _ = _obb_geometry_from_contour(cnt)
    return axis


def contour_to_obb_points(cnt: np.ndarray) -> np.ndarray:
    _, _, _, box_points = _obb_geometry_from_contour(cnt)
    if box_points is None:
        raise ValueError("Contour must contain at least 3 points for OBB conversion.")
    return box_points


def get_min_obb_aspect_ratio(cfg: dict) -> float:
    return float(cfg["MIN_ASPECT"])


def get_source_video_fps(cfg: dict) -> float:
    video_path = str(cfg["TRAINING_VIDEO_PATH"])
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"Could not read a positive FPS from TRAINING_VIDEO_PATH: {video_path}")
    return fps


def resolve_direction_min_valid_frames(cfg: dict) -> Tuple[int, float, float]:
    fps = get_source_video_fps(cfg)
    ratio = float(cfg["DIR_MIN_SEC"])
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(f"DIR_MIN_SEC must be > 0, got {ratio!r}")
    frames = max(1, int(round(fps * ratio)))
    return frames, ratio, fps


def get_individual_distance_jump_ratio(cfg: dict) -> float:
    return float(cfg.get("TRAJ_MAX_JUMP", 1.5))


def get_individual_distance_disp_ratio(cfg: dict) -> float:
    return float(cfg.get("DIR_MIN_DISP", 2.0))


def get_init_max_dist_ratio(cfg: dict) -> float:
    return float(cfg.get("INIT_MAX_DIST", 1.5))


def get_traj_max_dist_ratio(cfg: dict) -> float:
    return float(cfg.get("TRAJ_MAX_DIST", 1.5))


def yolo_line_from_obb_points(cls_id: int, pts: np.ndarray, W: int, H: int) -> str:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0] / float(W), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(H), 0.0, 1.0)
    flat = " ".join(f"{float(v):.6f}" for v in pts.reshape(-1))
    return f"{int(cls_id)} {flat}\n"


def draw_obb(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    poly = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    poly = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, color, thickness, cv2.LINE_AA)


def head_angle_deg_from_axis(axis: Tuple[float, float]) -> float:
    dx, dy = float(axis[0]), float(axis[1])
    return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0


def polygon_line_from_contour(
    cnt: np.ndarray,
    class_id: Optional[int] = None,
    class_name: Optional[str] = None,
    occlusion_state: Optional[int] = None,
    is_pasted: Optional[int] = None,
    overlap_pixels: Optional[int] = None,
    overlap_ratio: Optional[float] = None,
) -> str:
    pts = cnt.reshape(-1, 2)
    parts: List[str] = []
    if class_id is not None:
        parts.append(f"class_id={int(class_id)}")
    if class_name is not None:
        parts.append(f"class_name={class_name}")
    if occlusion_state is not None:
        parts.append(f"occlusion_state={int(occlusion_state)}")
    if is_pasted is not None:
        parts.append(f"is_pasted={int(is_pasted)}")
    if overlap_pixels is not None:
        parts.append(f"overlap_pixels={int(overlap_pixels)}")
    if overlap_ratio is not None:
        parts.append(f"overlap_ratio={float(overlap_ratio):.6f}")
    return " | ".join([
        " ".join(parts),
        "points=" + " ".join(f"{int(x)},{int(y)}" for x, y in pts),
    ]) + "\n"


def read_background(bg_path: str, frame_shape: Tuple[int, int]) -> np.ndarray:
    bg = cv2.imread(bg_path)
    if bg is None:
        raise FileNotFoundError(bg_path)
    h, w = frame_shape
    if (bg.shape[0], bg.shape[1]) != (h, w):
        # Do not silently resize: contour coordinates are in the video's own
        # pixel space, so a background of a different size means something is
        # actually wrong (stale/mismatched background.png), not a harmless
        # size difference to paper over.
        raise RuntimeError(
            f"Background size {(bg.shape[0], bg.shape[1])} does not match "
            f"video frame size {(h, w)}: {bg_path}"
        )
    return bg


def load_initial_tracking_csv(csv_path: str) -> Dict[Tuple[int, int], int]:
    assignments: Dict[Tuple[int, int], int] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            blob_index = int(row["blob_index"])
            track_id = int(row["track_id"])
            assignments[(frame, blob_index)] = track_id
    return assignments

def load_tracking_stats_csv(csv_path: str) -> Dict[str, str]:
    stats: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = str(row.get("metric", "")).strip()
            if not metric:
                continue
            stats[metric] = str(row.get("value", "")).strip()
    return stats


def get_required_tracking_stat(stats: Dict[str, str], key: str, cast):
    if key not in stats or stats[key] == "":
        raise KeyError(f"Missing '{key}' in initial tracking stats csv.")
    return cast(stats[key])


def build_blob_records(blob_seq, frame_indices: Sequence[int]) -> Dict[int, List[BlobRecord]]:
    out: Dict[int, List[BlobRecord]] = {}
    for fid in tqdm(frame_indices, desc="Preparing blobs"):
        records: List[BlobRecord] = []
        for bi, bl in enumerate(get_frame_blobs(blob_seq, fid)):
            cnt = np.asarray(bl.contour, dtype=np.int32)
            if cnt.size == 0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = contour_center(cnt)
            _, obb_major_axis_len, obb_aspect_ratio, _ = _obb_geometry_from_contour(cnt)
            records.append(
                BlobRecord(
                    contour=cnt,
                    rect=(int(x), int(y), int(w), int(h)),
                    center=(cx, cy),
                    area=contour_area(cnt),
                    bbox_area=float(w * h),
                    w=float(w),
                    h=float(h),
                    obb_major_axis_len=obb_major_axis_len,
                    obb_aspect_ratio=obb_aspect_ratio,
                    pickle_outlier=bool(bl.is_outlier),
                    frame=int(fid),
                    blob_index=int(bi),
                )
            )
        out[int(fid)] = records
    return out


def append_unique_reason(items: List[str], reason: str) -> None:
    if reason not in items:
        items.append(reason)


def classify_blobs_stage1(
    blob_records: Dict[int, List[BlobRecord]],
    cfg: dict,
    individual_distance_px: float,
    individual_distance_source_count: int,
    individual_distance_source_total_non_crossing: int,
    init_max_dist_px: float,
) -> Dict[str, Tuple[float, float]]:
    all_non_crossing = [b for lst in blob_records.values() for b in lst if not b.pickle_outlier]

    area_values = [b.area for b in all_non_crossing]
    bbox_values = [b.bbox_area for b in all_non_crossing]
    w_values = [b.w for b in all_non_crossing]
    h_values = [b.h for b in all_non_crossing]
    obb_values = [float(b.obb_major_axis_len or 0.0) for b in all_non_crossing]
    min_obb_aspect_ratio = get_min_obb_aspect_ratio(cfg)

    area_bounds = (float(np.min(area_values)) if area_values else -float("inf"), float(np.max(area_values)) if area_values else float("inf"))
    bbox_bounds = (float(np.min(bbox_values)) if bbox_values else -float("inf"), float(np.max(bbox_values)) if bbox_values else float("inf"))
    w_bounds = (float(np.min(w_values)) if w_values else -float("inf"), float(np.max(w_values)) if w_values else float("inf"))
    h_bounds = (float(np.min(h_values)) if h_values else -float("inf"), float(np.max(h_values)) if h_values else float("inf"))
    obb_major_axis_bounds = (float(np.min(obb_values)) if obb_values else -float("inf"), float(np.max(obb_values)) if obb_values else float("inf"))

    for lst in blob_records.values():
        for b in lst:
            b.area_outlier = False
            b.bbox_area_outlier = False
            b.width_outlier = False
            b.height_outlier = False
            b.obb_major_axis_outlier = False
            b.low_obb_aspect_outlier = False

            reasons: List[str] = []
            if b.pickle_outlier:
                reasons.append("pickle_outlier")
            if b.obb_aspect_ratio is not None and np.isfinite(b.obb_aspect_ratio):
                b.low_obb_aspect_outlier = bool(float(b.obb_aspect_ratio) < min_obb_aspect_ratio)
                if b.low_obb_aspect_outlier:
                    reasons.append(LOW_OBB_ASPECT_REASON)

            b.erase_reasons_stage1 = reasons
            b.erase_stage1 = len(reasons) > 0
            b.erase_pre_direction = b.erase_stage1
            b.erase_final = b.erase_stage1
            b.erase_reasons_pre_direction = list(reasons)
            b.erase_reasons_final = list(reasons)

    jump_threshold_px = individual_distance_px * get_individual_distance_jump_ratio(cfg)
    direction_min_total_disp_px = individual_distance_px * get_individual_distance_disp_ratio(cfg)
    min_valid_frames, min_valid_ratio, source_fps = resolve_direction_min_valid_frames(cfg)

    return {
        "area_bounds": area_bounds,
        "bbox_bounds": bbox_bounds,
        "w_bounds": w_bounds,
        "h_bounds": h_bounds,
        "obb_major_axis_bounds": obb_major_axis_bounds,
        "min_obb_aspect_ratio": min_obb_aspect_ratio,
        "jump_bounds": (0.0, jump_threshold_px),
        "individual_distance_px": individual_distance_px,
        "individual_distance_source_count": int(individual_distance_source_count),
        "individual_distance_source_total_non_crossing": int(individual_distance_source_total_non_crossing),
        "jump_threshold_px": jump_threshold_px,
        "init_max_dist_px": float(init_max_dist_px),
        "traj_max_dist_px": individual_distance_px * get_traj_max_dist_ratio(cfg),
        "direction_min_valid_frames": (
            int(min_valid_frames),
            int(min_valid_frames),
        ),
        "direction_min_valid_frames_fps_ratio": (
            float(min_valid_ratio),
            float(min_valid_ratio),
        ),
        "source_fps": (
            float(source_fps),
            float(source_fps),
        ),
        "direction_min_total_disp_px": (
            direction_min_total_disp_px,
            direction_min_total_disp_px,
        ),
        "jump_individual_distance_ratio": (
            get_individual_distance_jump_ratio(cfg),
            get_individual_distance_jump_ratio(cfg),
        ),
        "disp_individual_distance_ratio": (
            get_individual_distance_disp_ratio(cfg),
            get_individual_distance_disp_ratio(cfg),
        ),
        "init_max_dist_ratio": (
            get_init_max_dist_ratio(cfg),
            get_init_max_dist_ratio(cfg),
        ),
        "traj_max_dist_ratio": (
            get_traj_max_dist_ratio(cfg),
            get_traj_max_dist_ratio(cfg),
        ),
    }


def assign_track_ids_from_initial_tracking(
    blob_records: Dict[int, List[BlobRecord]],
    assignments: Dict[Tuple[int, int], int],
) -> None:
    for fid, blobs in tqdm(blob_records.items(), desc="Loading initial tracking"):
        for b in blobs:
            track_id = assignments.get((int(fid), int(b.blob_index)))
            if track_id is not None:
                b.traj_id = int(track_id)


def classify_jump_outliers_stage1(
    blob_records: Dict[int, List[BlobRecord]],
    cfg: dict,
    individual_distance_px: float,
) -> Dict[str, object]:
    jump_threshold_px = individual_distance_px * get_individual_distance_jump_ratio(cfg)

    by_tid: Dict[int, Dict[int, BlobRecord]] = {}
    for blobs in blob_records.values():
        for b in blobs:
            if b.erase_stage1:
                continue
            if b.traj_id is None:
                continue
            by_tid.setdefault(int(b.traj_id), {})[int(b.frame)] = b

    jump_pairs: List[Tuple[BlobRecord, BlobRecord, float]] = []
    jump_values: List[float] = []
    for fid_map in by_tid.values():
        frames = sorted(fid_map.keys())
        for prev_fid, curr_fid in zip(frames[:-1], frames[1:]):
            if curr_fid != prev_fid + 1:
                continue
            prev_b = fid_map[prev_fid]
            curr_b = fid_map[curr_fid]
            jump = float(math.hypot(
                float(curr_b.center[0]) - float(prev_b.center[0]),
                float(curr_b.center[1]) - float(prev_b.center[1]),
            ))
            curr_b.jump_distance_px = jump
            jump_pairs.append((prev_b, curr_b, jump))
            jump_values.append(jump)

    jump_bounds = (0.0, jump_threshold_px)
    jump_outlier_count = 0
    for _, curr_b, jump in jump_pairs:
        is_outlier = bool(jump > jump_threshold_px)
        curr_b.jump_outlier = is_outlier
        if not is_outlier:
            continue
        append_unique_reason(curr_b.erase_reasons_stage1, "jump_outlier")
        curr_b.erase_stage1 = True
        curr_b.erase_pre_direction = True
        curr_b.erase_final = True
        curr_b.erase_reasons_pre_direction = list(curr_b.erase_reasons_stage1)
        curr_b.erase_reasons_final = list(curr_b.erase_reasons_stage1)
        jump_outlier_count += 1

    return {
        "jump_bounds": jump_bounds,
        "jump_threshold_px": jump_threshold_px,
        "jump_pair_count": len(jump_pairs),
        "jump_outlier_count": jump_outlier_count,
        "jump_mean_px": float(np.mean(np.asarray(jump_values, dtype=np.float64))) if jump_values else 0.0,
    }


def split_runs(frames: List[int], max_gap: int = 1) -> List[List[int]]:
    if not frames:
        return []
    runs: List[List[int]] = []
    cur = [frames[0]]
    for f in frames[1:]:
        if f - cur[-1] <= max_gap:
            cur.append(f)
        else:
            runs.append(cur)
            cur = [f]
    runs.append(cur)
    return runs


def build_pre_direction_run_rows(blob_records: Dict[int, List[BlobRecord]]) -> List[Dict[str, object]]:
    by_tid: Dict[int, Dict[int, BlobRecord]] = {}
    for blobs in blob_records.values():
        for b in blobs:
            if b.erase_pre_direction:
                continue
            if b.traj_id is None:
                continue
            by_tid.setdefault(int(b.traj_id), {})[int(b.frame)] = b

    rows: List[Dict[str, object]] = []
    run_id = 0
    for traj_id in sorted(by_tid.keys()):
        fid_map = by_tid[traj_id]
        frames = sorted(fid_map.keys())
        for run_frames in split_runs(frames, max_gap=1):
            if not run_frames:
                continue
            rows.append({
                "run_id": run_id,
                "traj_id": traj_id,
                "start_frame": int(run_frames[0]),
                "end_frame": int(run_frames[-1]),
                "run_frame_count": len(run_frames),
            })
            run_id += 1
    return rows


def save_pre_direction_run_csv(blob_records: Dict[int, List[BlobRecord]], out_dir: str) -> str:
    path = os.path.join(out_dir, "pre_direction_runs.csv")
    rows = build_pre_direction_run_rows(blob_records)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "traj_id",
                "start_frame",
                "end_frame",
                "run_frame_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def classify_blobs_pre_direction(blob_records: Dict[int, List[BlobRecord]], cfg: dict) -> Dict[str, int]:
    min_valid_frames, _, _ = resolve_direction_min_valid_frames(cfg)

    by_tid: Dict[int, Dict[int, BlobRecord]] = {}
    unmatched_count = 0

    for lst in blob_records.values():
        for b in lst:
            b.erase_pre_direction = True
            b.erase_reasons_pre_direction = list(b.erase_reasons_stage1)
            if b.erase_stage1:
                continue
            if b.traj_id is None:
                append_unique_reason(b.erase_reasons_pre_direction, "traj_unmatched")
                unmatched_count += 1
                continue
            by_tid.setdefault(int(b.traj_id), {})[int(b.frame)] = b

    short_run_count = 0
    kept_for_direction = 0

    for fid_map in by_tid.values():
        frames = sorted(fid_map.keys())
        if not frames:
            continue

        run_start = 0
        while run_start < len(frames):
            run_end = run_start + 1
            while run_end < len(frames) and frames[run_end] == frames[run_end - 1] + 1:
                run_end += 1

            run_frames = frames[run_start:run_end]
            run_len = len(run_frames)
            is_short = run_len <= min_valid_frames

            for fid in run_frames:
                b = fid_map[fid]
                if is_short:
                    append_unique_reason(b.erase_reasons_pre_direction, "short_run")
                    b.erase_pre_direction = True
                    short_run_count += 1
                else:
                    b.erase_pre_direction = False
                    b.erase_reasons_pre_direction = list(b.erase_reasons_stage1)
                    kept_for_direction += 1

            run_start = run_end

    return {
        "pre_direction_unmatched": unmatched_count,
        "pre_direction_short_run": short_run_count,
        "pre_direction_kept": kept_for_direction,
    }


def estimate_directions(blob_records: Dict[int, List[BlobRecord]], cfg: dict, individual_distance_px: float) -> Dict[str, int]:
    min_total_disp = individual_distance_px * get_individual_distance_disp_ratio(cfg)
    # Guards the mv/sp normalization below against near-zero-motion frames;
    # not derived from a configurable ratio (see removed DIR_MIN_SPEED).
    eps_speed = 1e-6

    by_tid: Dict[int, Dict[int, BlobRecord]] = {}
    for lst in blob_records.values():
        for b in lst:
            if b.erase_pre_direction or b.traj_id is None:
                continue
            by_tid.setdefault(int(b.traj_id), {})[int(b.frame)] = b

    stats = {
        "direction_input_blobs": sum(1 for lst in blob_records.values() for b in lst if not b.erase_pre_direction),
        "valid_segments": 0,
        "direction_assigned_blobs": 0,
    }

    for tid, fid_map in tqdm(by_tid.items(), desc="Estimating directions"):
        frames = sorted(fid_map.keys())
        for run_frames in split_runs(frames, max_gap=1):
            valid_blobs: List[BlobRecord] = []
            axes: List[np.ndarray] = []
            centers: List[np.ndarray] = []

            for fid in run_frames:
                b = fid_map[fid]
                axis = obb_axis_from_contour(b.contour)
                if axis is None:
                    append_unique_reason(b.direction_failure_reasons, "axis_invalid")
                    continue
                center = np.asarray(b.center, dtype=np.float32)
                if not np.all(np.isfinite(center)):
                    append_unique_reason(b.direction_failure_reasons, "center_invalid")
                    continue
                valid_blobs.append(b)
                axes.append(axis.astype(np.float32))
                centers.append(center)

            if len(valid_blobs) < 2:
                for b in valid_blobs:
                    append_unique_reason(b.direction_failure_reasons, "insufficient_valid_frames")
                continue

            axes_arr = np.asarray(axes, dtype=np.float32)
            centers_arr = np.asarray(centers, dtype=np.float32)

            for i in range(1, len(axes_arr)):
                if float(np.dot(axes_arr[i - 1], axes_arr[i])) < 0.0:
                    axes_arr[i] *= -1.0

            move = np.zeros_like(centers_arr)
            move[1:-1] = (centers_arr[2:] - centers_arr[:-2]) / 2.0
            move[0] = centers_arr[1] - centers_arr[0]
            move[-1] = centers_arr[-1] - centers_arr[-2]

            total_disp = float(np.linalg.norm(centers_arr[-1] - centers_arr[0]))

            if total_disp < min_total_disp:
                for b in valid_blobs:
                    append_unique_reason(b.direction_failure_reasons, "disp_too_small")
                continue

            score_pos = 0.0
            score_neg = 0.0
            used_motion_count = 0
            for axis, mv in zip(axes_arr, move):
                sp = float(np.linalg.norm(mv))
                if sp < eps_speed:
                    continue
                used_motion_count += 1
                d = float(np.dot(axis, mv / sp))
                score_pos += sp * d
                score_neg += sp * (-d)

            if used_motion_count == 0:
                for b in valid_blobs:
                    append_unique_reason(b.direction_failure_reasons, "motion_ambiguous")
                continue

            if score_neg > score_pos:
                axes_arr *= -1.0

            stats["valid_segments"] += 1
            assigned_in_segment = 0
            for b, axis in zip(valid_blobs, axes_arr):
                u = unit_vec(float(axis[0]), float(axis[1]))
                if u is None:
                    append_unique_reason(b.direction_failure_reasons, "direction_norm_invalid")
                    continue
                dx, dy = float(u[0]), float(u[1])
                axis_length = contour_axis_length(b.center, (dx, dy), b.contour)
                if axis_length is None:
                    append_unique_reason(b.direction_failure_reasons, "axis_length_invalid")
                    continue
                b.class_id = int(angle_to_class(dx, dy))
                b.direction_vec = (dx, dy)
                b.axis_length = float(axis_length)
                b.direction_failure_reasons.clear()
                assigned_in_segment += 1
            stats["direction_assigned_blobs"] += assigned_in_segment

    for lst in blob_records.values():
        for b in lst:
            if b.erase_pre_direction:
                b.erase_final = True
                b.erase_reasons_final = list(b.erase_reasons_pre_direction)
                continue
            if b.class_id is None:
                b.erase_final = True
                reasons = list(b.erase_reasons_pre_direction)
                for r in b.direction_failure_reasons:
                    append_unique_reason(reasons, r)
                if not reasons:
                    reasons.append("direction_unassigned")
                b.erase_reasons_final = reasons
            else:
                b.erase_final = False
                b.erase_reasons_final = []

    return stats


def save_blob_classification_csv(blob_records: Dict[int, List[BlobRecord]], bounds: Dict[str, Tuple[float, float]], out_dir: str) -> str:
    path = os.path.join(out_dir, "blob_classification.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "blob_index",
            "rect_x",
            "rect_y",
            "rect_w",
            "rect_h",
            "center_x",
            "center_y",
            "area",
            "bbox_area",
            "pickle_outlier",
            "area_outlier",
            "bbox_area_outlier",
            "width_outlier",
            "height_outlier",
            "obb_major_axis_len",
            "obb_major_axis_outlier",
            "obb_aspect_ratio",
            "min_obb_aspect_ratio",
            "low_obb_aspect_outlier",
            "individual_distance_px",
            "jump_threshold_px",
            "init_max_dist_px",
            "traj_max_dist_px",
            "direction_min_total_disp_threshold_px",
            "jump_distance_px",
            "jump_outlier",
            "erase_stage1",
            "erase_stage1_reasons",
            "traj_id",
            "erase_pre_direction",
            "erase_pre_direction_reasons",
            "class_id",
            "direction_name",
            "direction_dx",
            "direction_dy",
            "axis_length",
            "head_angle_deg",
            "erase_final",
            "erase_final_reasons",
            "direction_failure_reasons",
            "area_lo",
            "area_hi",
            "bbox_lo",
            "bbox_hi",
            "w_lo",
            "w_hi",
            "h_lo",
            "h_hi",
            "obb_major_axis_lo",
            "obb_major_axis_hi",
            "jump_lo",
            "jump_hi",
        ])
        for fid in sorted(blob_records.keys()):
            for b in blob_records[fid]:
                dx = "" if b.direction_vec is None else f"{b.direction_vec[0]:.6f}"
                dy = "" if b.direction_vec is None else f"{b.direction_vec[1]:.6f}"
                axis_length = "" if b.axis_length is None else f"{float(b.axis_length):.6f}"
                head_angle_deg = "" if b.direction_vec is None else f"{head_angle_deg_from_axis(b.direction_vec):.6f}"
                writer.writerow([
                    b.frame,
                    b.blob_index,
                    b.rect[0],
                    b.rect[1],
                    b.rect[2],
                    b.rect[3],
                    f"{b.center[0]:.3f}",
                    f"{b.center[1]:.3f}",
                    f"{b.area:.3f}",
                    f"{b.bbox_area:.3f}",
                    int(b.pickle_outlier),
                    int(b.area_outlier),
                    int(b.bbox_area_outlier),
                    int(b.width_outlier),
                    int(b.height_outlier),
                    "" if b.obb_major_axis_len is None else f"{b.obb_major_axis_len:.3f}",
                    int(b.obb_major_axis_outlier),
                    "" if b.obb_aspect_ratio is None else f"{b.obb_aspect_ratio:.6f}",
                    f"{float(bounds['min_obb_aspect_ratio']):.6f}",
                    int(b.low_obb_aspect_outlier),
                    f"{float(bounds.get('individual_distance_px', 0.0)):.3f}",
                    f"{float(bounds.get('jump_threshold_px', 0.0)):.3f}",
                    f"{float(bounds.get('init_max_dist_px', 0.0)):.3f}",
                    f"{float(bounds.get('traj_max_dist_px', 0.0)):.3f}",
                    f"{float(bounds.get('direction_min_total_disp_px', (0.0, 0.0))[0]):.3f}",
                    "" if b.jump_distance_px is None else f"{b.jump_distance_px:.3f}",
                    int(b.jump_outlier),
                    int(b.erase_stage1),
                    "|".join(b.erase_reasons_stage1),
                    -1 if b.traj_id is None else b.traj_id,
                    int(b.erase_pre_direction),
                    "|".join(b.erase_reasons_pre_direction),
                    -1 if b.class_id is None else b.class_id,
                    "" if b.class_id is None else DIRECTION_CLASS_NAMES[b.class_id],
                    dx,
                    dy,
                    axis_length,
                    head_angle_deg,
                    int(b.erase_final),
                    "|".join(b.erase_reasons_final),
                    "|".join(b.direction_failure_reasons),
                    f"{bounds['area_bounds'][0]:.3f}",
                    f"{bounds['area_bounds'][1]:.3f}",
                    f"{bounds['bbox_bounds'][0]:.3f}",
                    f"{bounds['bbox_bounds'][1]:.3f}",
                    f"{bounds['w_bounds'][0]:.3f}",
                    f"{bounds['w_bounds'][1]:.3f}",
                    f"{bounds['h_bounds'][0]:.3f}",
                    f"{bounds['h_bounds'][1]:.3f}",
                    f"{bounds['obb_major_axis_bounds'][0]:.3f}",
                    f"{bounds['obb_major_axis_bounds'][1]:.3f}",
                    f"{bounds['jump_bounds'][0]:.3f}" if np.isfinite(bounds['jump_bounds'][0]) else str(bounds['jump_bounds'][0]),
                    f"{bounds['jump_bounds'][1]:.3f}" if np.isfinite(bounds['jump_bounds'][1]) else str(bounds['jump_bounds'][1]),
                ])
    return path


def draw_text(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int], scale: float = 0.4) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_multiline_text(img: np.ndarray, lines: List[str], x: int, y: int, color: Tuple[int, int, int], scale: float = 0.4) -> None:
    line_step = max(12, int(round(14 * scale / 0.4)))
    base_y = max(12, y)
    for i, line in enumerate(lines):
        draw_text(img, line, x, base_y + i * line_step, color, scale)


def contour_axis_length(
    center: Tuple[float, float],
    axis: Tuple[float, float],
    contour: np.ndarray,
) -> Optional[float]:
    u = unit_vec(float(axis[0]), float(axis[1]))
    if u is None:
        return None
    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    c = np.asarray([float(center[0]), float(center[1])], dtype=np.float32)
    proj = (pts - c) @ np.asarray(u, dtype=np.float32)
    if proj.size == 0:
        return None
    length = float(np.max(proj) - np.min(proj))
    if not np.isfinite(length) or length <= 0.0:
        return None
    return length


def ensure_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    area2 = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area2 += x1 * y2 - x2 * y1
    if area2 > 0:
        pts = pts[::-1]
    return pts.astype(np.float32)


def obb_center(pts: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(pts, dtype=np.float32).reshape(4, 2), axis=0)


def get_short_edge_candidates(pts: np.ndarray, rel_tol: float = 0.05) -> List[Tuple[int, int, float]]:
    pts = ensure_clockwise(pts)
    edges: List[Tuple[int, int, float]] = []
    lengths: List[float] = []
    for i in range(4):
        j = (i + 1) % 4
        edge_len = float(np.linalg.norm(pts[j] - pts[i]))
        edges.append((i, j, edge_len))
        lengths.append(edge_len)
    if not lengths:
        return []
    min_len = min(lengths)
    tol = max(1e-6, min_len * float(rel_tol))
    return [e for e in edges if abs(e[2] - min_len) <= tol]


def outward_normal_for_edge(pts: np.ndarray, i: int, j: int) -> np.ndarray:
    pts = ensure_clockwise(pts)
    p0 = pts[i]
    p1 = pts[j]
    edge = p1 - p0
    edge_len = float(np.linalg.norm(edge))
    if edge_len <= 0.0:
        return np.asarray([0.0, 0.0], dtype=np.float32)
    normal = np.asarray([edge[1], -edge[0]], dtype=np.float32) / edge_len
    midpoint = 0.5 * (p0 + p1)
    center = obb_center(pts)
    if float(np.dot(midpoint - center, normal)) < 0.0:
        normal = -normal
    return normal.astype(np.float32)


def draw_heading_triangle_for_obb(
    img: np.ndarray,
    pts: np.ndarray,
    direction_vec: Tuple[float, float],
    color: Tuple[int, int, int],
    alpha: float = 0.6,
    outline_thickness: int = 1,
    scale: float = 1.2,
) -> None:
    pts = ensure_clockwise(pts)
    u = unit_vec(float(direction_vec[0]), float(direction_vec[1]))
    if u is None:
        return
    short_edges = get_short_edge_candidates(pts)
    if not short_edges:
        return

    best = None
    for i, j, edge_len in short_edges:
        normal = outward_normal_for_edge(pts, i, j)
        score = float(np.dot(normal, np.asarray(u, dtype=np.float32)))
        cand = (score, i, j, edge_len, normal)
        if best is None or score > best[0]:
            best = cand
    if best is None:
        return

    _, i, j, base_len, normal = best
    if base_len <= 0.0:
        return
    p0 = pts[i].astype(np.float32)
    p1 = pts[j].astype(np.float32)
    base_mid = 0.5 * (p0 + p1)
    height = float(scale) * (math.sqrt(3.0) / 2.0) * float(base_len)
    apex = base_mid + normal.astype(np.float32) * height

    tri = np.vstack([p0, p1, apex]).astype(np.float32)
    tri_i32 = np.round(tri).astype(np.int32).reshape(-1, 1, 2)

    a = float(np.clip(alpha, 0.0, 1.0))
    if a > 0.0:
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, tri_i32, color, lineType=cv2.LINE_AA)
        img[...] = cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0)
    if int(outline_thickness) > 0:
        cv2.polylines(img, [tri_i32], True, color, int(outline_thickness), cv2.LINE_AA)


_WITHOUT_CROSSING_WORKER = {}

# GPU caches: keyed by (ksize, sigma) for the Gaussian kernel and id(bg_f32) for the background tensor.
_GPU_GAUSS_CACHE: dict = {}
_GPU_BG_CACHE: dict = {}


def _get_gauss_kernel_gpu(ksize: int, sigma: float):
    key = (ksize, sigma)
    if key not in _GPU_GAUSS_CACHE:
        ax = _torch.arange(ksize, dtype=_torch.float32) - ksize // 2
        g = _torch.exp(-ax ** 2 / (2.0 * sigma ** 2))
        k2d = g.outer(g)
        k2d /= k2d.sum()
        _GPU_GAUSS_CACHE[key] = k2d.view(1, 1, ksize, ksize).cuda()
    return _GPU_GAUSS_CACHE[key]


def _without_crossing_blend_gpu(
    frame_bgr: np.ndarray,
    remove_mask: np.ndarray,
    bg_f32: np.ndarray,
    ksize: int,
    sigma: float,
) -> np.ndarray:
    """GPU-accelerated GaussianBlur + alpha blend. Falls back to CPU on any error."""
    try:
        keep = (remove_mask == 0).astype(np.float32)
        keep_t = _torch.from_numpy(keep).cuda().unsqueeze(0).unsqueeze(0)
        if ksize > 1:
            keep_t = _F.conv2d(keep_t, _get_gauss_kernel_gpu(ksize, sigma), padding=ksize // 2)
        alpha_t = keep_t.squeeze().unsqueeze(-1)
        fr_t = _torch.from_numpy(frame_bgr).float().cuda()
        bg_key = id(bg_f32)
        if bg_key not in _GPU_BG_CACHE:
            _GPU_BG_CACHE[bg_key] = _torch.from_numpy(bg_f32).cuda()
        bg_t = _GPU_BG_CACHE[bg_key]
        return (fr_t * alpha_t + bg_t * (1.0 - alpha_t)).clamp(0, 255).byte().cpu().numpy()
    except Exception:
        keep = (remove_mask == 0).astype(np.float32)
        alpha = cv2.GaussianBlur(keep, (ksize, ksize), sigma)[..., None]
        return (frame_bgr.astype(np.float32) * alpha + bg_f32 * (1.0 - alpha)).astype(np.uint8)


def _probe_video_frame_shape(video_path: str) -> Tuple[int, int]:
    """Return (H, W) by actually decoding the video's first frame.

    Uses the real decoded frame rather than CAP_PROP_FRAME_WIDTH/HEIGHT
    container metadata, which can disagree with what actually comes out of
    decode for some files.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        ret, frame0 = cap.read()
        if not ret or frame0 is None or frame0.size == 0:
            raise RuntimeError(f"Failed to read first frame: {video_path}")
        return int(frame0.shape[0]), int(frame0.shape[1])
    finally:
        cap.release()


def _iter_needed_video_frames(video_path: str, needed_frame_ids: Sequence[int]):
    """Decode exactly the requested frame ids, in ascending order, and nothing else.

    Never seeks: CAP_PROP_POS_FRAMES is not guaranteed frame-exact for
    inter-frame codecs (H.264/H.265), and a single mis-seek would silently
    pair a contour from the segmentation pickle with the wrong video frame,
    shifting the mask by however far the animal moved in that time. Instead
    this walks the decoder forward from frame 0 with grab() (cheap skip, no
    color-space conversion) for frames we don't need and read() (full decode)
    for frames we do, so the frame handed back for a given id is always
    exactly that frame.
    """
    ordered_ids = sorted(int(f) for f in needed_frame_ids)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        current = 0
        for fid in ordered_ids:
            while current < fid:
                if not cap.grab():
                    raise RuntimeError(
                        f"Video ended while advancing to frame {fid} "
                        f"(stopped at frame {current}): {video_path}"
                    )
                current += 1
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                raise RuntimeError(f"Failed to read frame {fid} from video: {video_path}")
            current += 1
            yield fid, frame
    finally:
        cap.release()


def _init_without_crossing_writer_worker(
    cfg: dict,
    bounds: dict,
    preview_ids: Sequence[int],
    frame_shape: Tuple[int, int],
    use_gpu: bool = False,
) -> None:
    """Initialize the output context (no video decoding here).

    Called once in the main thread (serial / ThreadPoolExecutor paths) or once per
    worker process (ProcessPoolExecutor path). Frames are decoded exactly once,
    serially, by the caller (_iter_needed_video_frames) and handed to
    _process_without_crossing_frame directly, so workers never touch a
    VideoCapture and can never seek.
    """
    cv2.setNumThreads(1)
    session_path = str(cfg["SESSION_PATH"])
    bg_path = str(cfg["BACKGROUND_PATH"])
    out_dir = os.path.join(session_path, SINGLE_ANIMAL_IMAGES_DIR_NAME)

    H, W = int(frame_shape[0]), int(frame_shape[1])
    bg = read_background(bg_path, (H, W))

    edge_blur_ksize = int(cfg.get("EDGE_BLUR_KSIZE", 7))
    edge_blur_sigma = float(cfg.get("EDGE_BLUR_SIGMA", 11))
    if edge_blur_ksize < 1:
        edge_blur_ksize = 1
    if edge_blur_ksize % 2 == 0:
        edge_blur_ksize += 1
    png_compression = min(
        9,
        max(0, int(cfg.get("WITHOUT_CROSSING_PNG_COMPRESSION", 1))),
    )

    _WITHOUT_CROSSING_WORKER.clear()
    _WITHOUT_CROSSING_WORKER.update({
        "cfg": cfg,
        "bounds": bounds,
        "preview_ids": {int(x) for x in preview_ids},
        "H": int(H),
        "W": int(W),
        "bg_f32": bg.astype(np.float32),
        "edge_blur_ksize": int(edge_blur_ksize),
        "edge_blur_sigma": float(edge_blur_sigma),
        "mask_expansion_ratio": get_mask_expansion_ratio(cfg),
        "use_gpu": bool(use_gpu),
        "png_write_params": [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
        "img_dir": os.path.join(out_dir, "images"),
        "lbl_dir": os.path.join(out_dir, "labels"),
        "mask_dir": os.path.join(out_dir, "masks"),
        "prev_dir": os.path.join(out_dir, "preview"),
        "pool_img_dir": os.path.join(out_dir, "object_pool", "images"),
        "pool_mask_dir": os.path.join(out_dir, "object_pool", "masks"),
    })


# Paste time re-derives its expansion contour from the rasterized exact mask
# (see paste_blobs.expand_mask_raster), not from this file's original smooth
# vector contour, so its cv2.fillPoly/findContours round-trip lands on a
# slightly different centroid and boundary than expanding the vector contour
# directly. A few pixels of extra padding absorb that rounding gap instead of
# relying on both independent computations to agree to the pixel.
_CROP_PADDING_MARGIN_PX = 3


def _extract_masked_blob_crop(
    frame: np.ndarray,
    contour: np.ndarray,
    mask_expansion_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Create an object-pool donor crop, padded out to mask_expansion_ratio.

    The padding is computed in source-frame coordinates (expand the contour
    from its centroid, then clip to the frame boundary) before cropping, so
    the crop always has real pixels behind the maximum paste-time mask
    expansion instead of being clipped by a tight crop. The crop itself is
    the raw frame ROI (not masked to the object silhouette): interaction_image_synthesis.py's
    compositing is entirely mask-driven, so leaving real background pixels in
    the padding ring is what lets an expanded paste mask reveal them. The
    exported mask is always the exact, unexpanded contour, rasterized in the
    same padded coordinate frame.

    Returns (crop_bgr, crop_mask_u8, (x, y, w, h)) where (x, y, w, h) is the
    padded crop's rect in frame coordinates.
    """
    frame_h, frame_w = frame.shape[:2]
    padded_contour = (
        expand_contour_from_centroid(contour, mask_expansion_ratio)
        if mask_expansion_ratio > 1.0 + 1e-9
        else contour
    )
    x, y, w, h = cv2.boundingRect(np.asarray(padded_contour, dtype=np.int32))
    margin = _CROP_PADDING_MARGIN_PX if mask_expansion_ratio > 1.0 + 1e-9 else 0
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame_w, x + w + margin)
    y2 = min(frame_h, y + h + margin)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype), np.empty((0, 0), dtype=np.uint8), (x1, y1, 0, 0)

    crop = frame[y1:y2, x1:x2].copy()

    crop_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    local_contour = np.asarray(contour, dtype=np.int32).reshape(-1, 2).copy()
    local_contour[:, 0] -= x1
    local_contour[:, 1] -= y1
    cv2.fillPoly(crop_mask, [local_contour], 255)
    return crop, crop_mask, (x1, y1, x2 - x1, y2 - y1)


def _empty_frame_reason_counts(frame_blob_records: Sequence[BlobRecord]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    if not frame_blob_records:
        counts["no_source_blobs"] += 1
        return dict(counts)
    for b in frame_blob_records:
        if b.class_id is not None and not b.erase_final:
            continue
        reasons = list(b.erase_reasons_final)
        if b.class_id is None and not b.erase_final:
            append_unique_reason(reasons, "class_unassigned")
        if not reasons:
            reasons = ["unknown_removed"]
        for reason in reasons:
            counts[str(reason)] += 1
    return dict(counts)


def _process_without_crossing_frame(
    fid: int, frame: np.ndarray, frame_blob_records: List[BlobRecord],
) -> Tuple[int, int, int, int, List[dict], Dict[str, int]]:
    """Write one single_animal_images frame and its per-frame object-pool files.

    `frame` must already be the exact, correctly-decoded video frame for
    `fid` (see _iter_needed_video_frames) -- this function does no video I/O.
    """
    fid = int(fid)
    ctx = _WITHOUT_CROSSING_WORKER
    H = int(ctx["H"])
    W = int(ctx["W"])
    bounds = ctx["bounds"]

    valid_blobs = [b for b in frame_blob_records if b.class_id is not None and not b.erase_final]
    if not valid_blobs:
        return fid, 0, 0, 0, [], _empty_frame_reason_counts(frame_blob_records)

    if frame.shape[:2] != (H, W):
        # Do not silently resize: contour coordinates are in the video's own
        # pixel space, so a differently-sized frame means something is
        # actually wrong, not a harmless size difference to paper over.
        raise RuntimeError(
            f"Frame shape mismatch at frame={fid}: got {frame.shape[:2]}, expected {(H, W)}."
        )

    remove_blobs = [b for b in frame_blob_records if b.erase_final]

    if not remove_blobs:
        canvas = frame
    else:
        remove_mask = np.zeros((H, W), dtype=np.uint8)
        for b in remove_blobs:
            cv2.fillPoly(remove_mask, [b.contour.astype(np.int32)], 255)

        ksize = int(ctx["edge_blur_ksize"])
        sigma = float(ctx["edge_blur_sigma"])
        if bool(ctx.get("use_gpu", False)):
            canvas = _without_crossing_blend_gpu(frame, remove_mask, ctx["bg_f32"], ksize, sigma)
        else:
            keep = (remove_mask == 0).astype(np.float32)
            alpha = cv2.GaussianBlur(keep, (ksize, ksize), sigma)[..., None]
            canvas = (frame.astype(np.float32) * alpha + ctx["bg_f32"] * (1.0 - alpha)).astype(np.uint8)

    label_lines: List[str] = []
    mask_lines: List[str] = []
    manifest_rows: List[dict] = []
    accepted_count = 0

    for out_idx, b in enumerate(valid_blobs):
        class_name = DIRECTION_CLASS_NAMES[b.class_id]
        obb_pts = contour_to_obb_points(b.contour)
        label_lines.append(yolo_line_from_obb_points(b.class_id, obb_pts, W, H))
        mask_lines.append(
            polygon_line_from_contour(
                b.contour,
                class_id=b.class_id,
                class_name=class_name,
                occlusion_state=0,
                is_pasted=0,
                overlap_pixels=0,
                overlap_ratio=0.0,
            )
        )

        crop, crop_mask, (x, y, w, h) = _extract_masked_blob_crop(
            frame, b.contour, ctx["mask_expansion_ratio"],
        )
        pool_id = f"f{fid:06d}_b{out_idx:03d}"
        crop_name = pool_id + ".png"
        mask_name = pool_id + "_mask.png"
        cv2.imwrite(os.path.join(ctx["pool_img_dir"], crop_name), crop, ctx["png_write_params"])
        cv2.imwrite(os.path.join(ctx["pool_mask_dir"], mask_name), crop_mask, ctx["png_write_params"])
        if b.direction_vec is None:
            raise ValueError(f"Accepted blob has no direction_vec: frame={fid}, blob_index={b.blob_index}")
        direction_norm = math.hypot(float(b.direction_vec[0]), float(b.direction_vec[1]))
        if not np.isfinite(direction_norm) or not math.isclose(direction_norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(
                f"Accepted blob direction_vec is not unit length: "
                f"frame={fid}, blob_index={b.blob_index}, norm={direction_norm}"
            )
        if b.axis_length is None or not np.isfinite(float(b.axis_length)) or float(b.axis_length) <= 0.0:
            raise ValueError(f"Accepted blob has invalid axis_length: frame={fid}, blob_index={b.blob_index}")
        manifest_rows.append({
            "pool_id": pool_id,
            "frame": fid,
            "blob_index": b.blob_index,
            "traj_id": -1 if b.traj_id is None else b.traj_id,
            "direction_valid": 1,
            "class_id": b.class_id,
            "class_name": class_name,
            "head_angle_deg": f"{head_angle_deg_from_axis(b.direction_vec):.6f}",
            "direction_dx": f"{b.direction_vec[0]:.6f}",
            "direction_dy": f"{b.direction_vec[1]:.6f}",
            "center_x": f"{float(b.center[0]):.6f}",
            "center_y": f"{float(b.center[1]):.6f}",
            "axis_length": f"{float(b.axis_length):.6f}",
            "obb_aspect_ratio": "" if b.obb_aspect_ratio is None else f"{float(b.obb_aspect_ratio):.6f}",
            "min_obb_aspect_ratio": f"{float(bounds['min_obb_aspect_ratio']):.6f}",
            "rect_x": x,
            "rect_y": y,
            "rect_w": w,
            "rect_h": h,
            "occlusion_state": 0,
            "crop_image": os.path.join("object_pool", "images", crop_name),
            "crop_mask": os.path.join("object_pool", "masks", mask_name),
            "is_pasted": 0,
            "overlap_pixels": 0,
            "overlap_ratio": 0.0,
        })
        accepted_count += 1

    img_name = f"frame_{fid:06d}.png"
    cv2.imwrite(os.path.join(ctx["img_dir"], img_name), canvas, ctx["png_write_params"])

    with open(os.path.join(ctx["lbl_dir"], img_name.replace(".png", ".txt")), "w", encoding="utf-8") as f:
        f.writelines(label_lines)
    with open(os.path.join(ctx["mask_dir"], img_name.replace(".png", "_maskinfo.txt")), "w", encoding="utf-8") as f:
        f.writelines(mask_lines)

    saved_preview_count = 0
    if fid in ctx["preview_ids"]:
        preview = canvas.copy()
        for b in valid_blobs:
            obb_pts = contour_to_obb_points(b.contour)
            draw_obb(preview, obb_pts, OBB_COLOR, 2)
            if b.direction_vec is not None:
                draw_heading_triangle_for_obb(
                    preview,
                    obb_pts,
                    b.direction_vec,
                    OBB_COLOR,
                    alpha=0.6,
                    outline_thickness=1,
                    scale=1.2,
                )
        for b in remove_blobs:
            draw_obb(preview, contour_to_obb_points(b.contour), OUTLIER_COLOR, 2)
        cv2.imwrite(os.path.join(ctx["prev_dir"], img_name), preview, ctx["png_write_params"])
        saved_preview_count = 1

    return fid, accepted_count, 1, saved_preview_count, manifest_rows, {}


def _resolve_without_crossing_writer_workers(cfg: dict) -> int:
    from batch_utils import auto_num_workers as _auto
    workers_cfg = resolve_num_workers(cfg, "create_single_animals_images", default=None)
    if workers_cfg is not None:
        return max(1, int(workers_cfg))
    return _auto("process")


def _resolve_without_crossing_writer_gpu(cfg: dict) -> bool:
    """Use CPU processes by default; enable CUDA only through an explicit opt-in."""
    requested = str(cfg.get("WITHOUT_CROSSING_WRITER_DEVICE", "cpu")).strip().lower()
    if requested not in {"gpu", "cuda"}:
        return False
    if not _load_cuda_backend():
        print("[WARN] WITHOUT_CROSSING_WRITER_DEVICE requests CUDA, but CUDA is unavailable; using CPU.")
        return False
    return True


def save_without_crossing_dataset(
    blob_records: Dict[int, List[BlobRecord]],
    cfg: dict,
    output_frame_indices: Sequence[int],
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, int]:
    session_path = str(cfg["SESSION_PATH"])
    out_dir = os.path.join(session_path, SINGLE_ANIMAL_IMAGES_DIR_NAME)
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    mask_dir = os.path.join(out_dir, "masks")
    prev_dir = os.path.join(out_dir, "preview")
    pool_img_dir = os.path.join(out_dir, "object_pool", "images")
    pool_mask_dir = os.path.join(out_dir, "object_pool", "masks")
    ensure_dirs([img_dir, lbl_dir, mask_dir, prev_dir, pool_img_dir, pool_mask_dir])

    frame_tasks: List[Tuple[int, List[BlobRecord]]] = []
    for fid in output_frame_indices:
        fid_int = int(fid)
        if fid_int not in blob_records:
            continue
        frame_tasks.append((fid_int, list(blob_records[fid_int])))

    # Select preview frames by actual frame id, at least PREVIEW_INTERVAL video
    # frames apart, starting at the first available frame id. `frame_tasks` is
    # already thinned by the upstream frame interval, so slicing by list
    # position would compound the two intervals instead of spacing previews by
    # PREVIEW_INTERVAL video frames.
    preview_interval = max(0, int(cfg.get("PREVIEW_INTERVAL", 0)))
    preview_ids: List[int] = []
    if preview_interval > 0:
        last_preview_fid = None
        for fid, _ in frame_tasks:
            if last_preview_fid is None or fid - last_preview_fid >= preview_interval:
                preview_ids.append(fid)
                last_preview_fid = fid

    workers = _resolve_without_crossing_writer_workers(cfg)
    use_gpu = _resolve_without_crossing_writer_gpu(cfg)
    if len(frame_tasks) <= 1:
        workers = 1
    else:
        workers = min(int(workers), len(frame_tasks))

    results: List[Tuple[int, int, int, int, List[dict], Dict[str, int]]] = []
    accel = "GPU" if use_gpu else "CPU"

    # Frames with no accepted blob never need decoding (they are cheaply
    # grab()-skipped by _iter_needed_video_frames); split them out up front so
    # only frames that actually produce an output image go through a full
    # video decode.
    decode_tasks: Dict[int, List[BlobRecord]] = {}
    for fid, frame_blob_records in frame_tasks:
        valid_blobs = [b for b in frame_blob_records if b.class_id is not None and not b.erase_final]
        if not valid_blobs:
            results.append((fid, 0, 0, 0, [], _empty_frame_reason_counts(frame_blob_records)))
        else:
            decode_tasks[fid] = frame_blob_records
    needed_ids = sorted(decode_tasks.keys())

    if needed_ids:
        video_path = str(cfg["TRAINING_VIDEO_PATH"])
        frame_shape = _probe_video_frame_shape(video_path)
        desc = "Writing single_animal_images segments" + (f" x{workers}" if workers > 1 else "") + f" ({accel})"

        if workers <= 1:
            # Serial path: single VideoCapture, decoded strictly in order.
            _init_without_crossing_writer_worker(cfg, bounds, preview_ids, frame_shape, use_gpu)
            try:
                for fid, frame in tqdm(
                    _iter_needed_video_frames(video_path, needed_ids),
                    total=len(needed_ids),
                    desc=desc,
                ):
                    results.append(_process_without_crossing_frame(fid, frame, decode_tasks[fid]))
            finally:
                _WITHOUT_CROSSING_WORKER.clear()
                _GPU_BG_CACHE.clear()
        else:
            # Decoding is strictly serial and seek-free (see
            # _iter_needed_video_frames), which guarantees each frame handed
            # to a worker is really the frame its pickle contour was computed
            # from. Frames are pulled off that single decode stream and
            # dispatched in bounded batches, so parallelism is kept for
            # background substitution / PNG writing without ever holding more
            # than a couple batches' worth of full-resolution frames in memory.
            batch_size = max(1, workers * 2)
            frame_iter = _iter_needed_video_frames(video_path, needed_ids)
            if use_gpu:
                # GPU path: threads share process memory, so the frame handed
                # to a worker is the exact array this loop decoded.
                _init_without_crossing_writer_worker(cfg, bounds, preview_ids, frame_shape, True)
                executor_cm = ThreadPoolExecutor(max_workers=workers)
            else:
                # CPU path: each worker process only ever receives frames
                # already decoded by the main process; it never opens the
                # video itself.
                executor_cm = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_without_crossing_writer_worker,
                    initargs=(cfg, bounds, preview_ids, frame_shape, False),
                )
            try:
                with executor_cm as ex, tqdm(total=len(needed_ids), desc=desc) as pbar:
                    while True:
                        batch = list(itertools.islice(frame_iter, batch_size))
                        if not batch:
                            break
                        futures = {
                            ex.submit(_process_without_crossing_frame, fid, frame, decode_tasks[fid]): fid
                            for fid, frame in batch
                        }
                        for fut in as_completed(futures):
                            results.append(fut.result())
                            pbar.update(1)
            finally:
                if use_gpu:
                    _WITHOUT_CROSSING_WORKER.clear()
                    _GPU_BG_CACHE.clear()

    results.sort(key=lambda x: int(x[0]))

    manifest_rows: List[dict] = []
    accepted_count = 0
    saved_image_count = 0
    saved_preview_count = 0
    skipped_empty_frames = 0
    empty_frame_reason_counts: Counter[str] = Counter()
    for _, accepted, saved_image, saved_preview, rows, reason_counts in results:
        accepted_count += int(accepted)
        saved_image_count += int(saved_image)
        saved_preview_count += int(saved_preview)
        manifest_rows.extend(rows)
        if int(saved_image) == 0:
            skipped_empty_frames += 1
            empty_frame_reason_counts.update({str(k): int(v) for k, v in reason_counts.items()})

    manifest_path = os.path.join(out_dir, "object_pool", "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pool_id", "frame", "blob_index", "traj_id", "direction_valid", "class_id", "class_name",
                "head_angle_deg", "direction_dx", "direction_dy",
                "center_x", "center_y", "axis_length",
                "obb_aspect_ratio", "min_obb_aspect_ratio",
                "rect_x", "rect_y", "rect_w", "rect_h",
                "crop_image", "crop_mask",
                "occlusion_state", "is_pasted", "overlap_pixels", "overlap_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    names_path = os.path.join(out_dir, "class_names.txt")
    with open(names_path, "w", encoding="utf-8") as f:
        for name in DIRECTION_CLASS_NAMES:
            f.write(name + "\n")

    print(f"single_animal_images writer workers: {workers}")

    return {
        "accepted_objects": accepted_count,
        "object_pool_rows": len(manifest_rows),
        "saved_images": saved_image_count,
        "saved_previews": saved_preview_count,
        "skipped_empty_frames": skipped_empty_frames,
        "empty_frame_removed_blobs": sum(empty_frame_reason_counts.values()),
        "empty_frame_reason_counts": dict(sorted(empty_frame_reason_counts.items())),
    }


def save_individual_distance_csv(out_dir: str, bounds: Dict[str, Tuple[float, float]], cfg: dict) -> str:
    path = os.path.join(out_dir, "individual_distance.csv")
    rows = [
        ("individual_distance_px", float(bounds.get("individual_distance_px", 0.0))),
        ("individual_distance_source_count", int(bounds.get("individual_distance_source_count", 0))),
        ("individual_distance_source_total_non_crossing", int(bounds.get("individual_distance_source_total_non_crossing", 0))),
        ("obb_major_axis_lo", float(bounds.get("obb_major_axis_bounds", (0.0, 0.0))[0])),
        ("obb_major_axis_hi", float(bounds.get("obb_major_axis_bounds", (0.0, 0.0))[1])),
        ("min_obb_aspect_ratio", float(bounds["min_obb_aspect_ratio"])),
        ("jump_threshold_px", float(bounds.get("jump_threshold_px", 0.0))),
        ("jump_ratio_to_individual_distance", get_individual_distance_jump_ratio(cfg)),
        ("source_fps", float(bounds.get("source_fps", (0.0, 0.0))[0])),
        ("direction_min_valid_frames", int(bounds.get("direction_min_valid_frames", (0, 0))[0])),
        ("direction_min_valid_frames_fps_ratio", float(bounds.get("direction_min_valid_frames_fps_ratio", (0.0, 0.0))[0])),
        ("init_max_dist_px", float(bounds.get("init_max_dist_px", 0.0))),
        ("init_max_dist_ratio", get_init_max_dist_ratio(cfg)),
        ("traj_max_dist_px", float(bounds.get("traj_max_dist_px", 0.0))),
        ("traj_max_dist_ratio", get_traj_max_dist_ratio(cfg)),
        ("direction_min_total_disp_px", float(bounds.get("direction_min_total_disp_px", (0.0, 0.0))[0])),
        ("direction_min_total_disp_ratio", get_individual_distance_disp_ratio(cfg)),
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in rows:
            writer.writerow([metric, numeric_or_text(value)])
    return path


def collect_kept_bbox_rows(blob_records: Dict[int, List[BlobRecord]]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for fid in sorted(blob_records.keys()):
        for b in blob_records[fid]:
            if b.erase_final:
                continue
            rows.append({
                "frame": int(b.frame),
                "bbox_index": int(b.blob_index),
                "w": float(b.w),
                "h": float(b.h),
                "sum": float(b.w + b.h),
                "area": float(b.w * b.h),
            })
    return rows


def find_source_bbox_row(rows: List[Dict[str, float]], metric: str, kind: str) -> Optional[Dict[str, float]]:
    if not rows or metric not in {"w", "h", "sum", "area"}:
        return None
    reverse = str(kind).lower().strip() == "max"
    return sorted(
        rows,
        key=lambda r: (float(r[metric]), int(r["frame"]), int(r["bbox_index"])),
        reverse=reverse,
    )[0]


def numeric_or_text(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return str(int(v))
        return f"{float(v):.6f}"
    return str(v)


def save_stats(blob_records: Dict[int, List[BlobRecord]], pre_direction_stats: Dict[str, int], direction_stats: Dict[str, int], save_stats_dict: Dict[str, int], cfg: dict, out_dir: str, bounds: Dict[str, Tuple[float, float]]) -> str:
    stats_path = os.path.join(out_dir, "bbox_stats_single_animal_images.csv")
    total_blob_count = sum(len(lst) for lst in blob_records.values())
    stage1_erased = sum(1 for lst in blob_records.values() for b in lst if b.erase_stage1)
    pre_direction_erased = sum(1 for lst in blob_records.values() for b in lst if b.erase_pre_direction)
    final_erased = sum(1 for lst in blob_records.values() for b in lst if b.erase_final)
    final_kept = total_blob_count - final_erased
    low_obb_aspect_erased = sum(1 for lst in blob_records.values() for b in lst if b.low_obb_aspect_outlier)
    kept_rows = collect_kept_bbox_rows(blob_records)

    summary_rows: List[List[object]] = []

    def add_metric(metric: str, kind: str, value_px, frame="", bbox_index="", w="", h="") -> None:
        summary_rows.append([metric, kind, numeric_or_text(value_px), frame, bbox_index, w, h])

    def add_metric_with_source(metric: str, kind: str, value_px) -> None:
        src = find_source_bbox_row(kept_rows, metric, kind)
        if src is None:
            add_metric(metric, kind, value_px)
            return
        add_metric(
            metric,
            kind,
            value_px,
            int(src["frame"]),
            int(src["bbox_index"]),
            numeric_or_text(src["w"]),
            numeric_or_text(src["h"]),
        )

    meta_items = [
        ("n_total_boxes", "meta", total_blob_count),
        ("frame_interval", "meta", int(cfg.get("FRAME_INTERVAL", 1))),
        ("first_frame", "meta", int((cfg.get("training", {}) or {}).get("FIRST_FRAME", 0))),
        ("last_frame_effective", "meta", max((int(fid) for fid in blob_records.keys()), default=-1)),
        ("crossing_mode", "meta", "segmentation_pickle"),
        ("source_fps", "meta", float(bounds.get("source_fps", (0.0, 0.0))[0])),
        ("direction_min_valid_frames", "meta", int(bounds.get("direction_min_valid_frames", (0, 0))[0])),
        ("direction_min_valid_frames_fps_ratio", "meta", float(bounds.get("direction_min_valid_frames_fps_ratio", (0.0, 0.0))[0])),
        ("individual_distance_px", "meta", float(bounds.get("individual_distance_px", 0.0))),
        ("individual_distance_source_count", "meta", int(bounds.get("individual_distance_source_count", 0))),
        ("direction_min_total_disp_px", "meta", float(cfg.get("_derived_direction_min_total_disp_px", 0.0))),
        ("min_obb_aspect_ratio", "meta", float(bounds["min_obb_aspect_ratio"])),
        ("jump_threshold_px", "meta", float(cfg.get("_derived_jump_threshold_px", 0.0))),
        ("init_max_dist_px", "meta", float(cfg.get("_derived_init_max_dist_px", 0.0))),
        ("traj_max_dist_px", "meta", float(cfg.get("_derived_traj_max_dist_px", 0.0))),
        ("jump_ratio_to_individual_distance", "meta", get_individual_distance_jump_ratio(cfg)),
        ("init_max_dist_ratio", "meta", get_init_max_dist_ratio(cfg)),
        ("traj_max_dist_ratio", "meta", get_traj_max_dist_ratio(cfg)),
        ("direction_min_total_disp_ratio", "meta", get_individual_distance_disp_ratio(cfg)),
        ("stage1_erased_blobs", "meta", stage1_erased),
        ("low_obb_aspect_erased_blobs", "meta", low_obb_aspect_erased),
        ("pre_direction_erased_blobs", "meta", pre_direction_erased),
        ("final_erased_blobs", "meta", final_erased),
        ("final_kept_blobs", "meta", final_kept),
        ("pre_direction_unmatched", "meta", pre_direction_stats["pre_direction_unmatched"]),
        ("pre_direction_short_run", "meta", pre_direction_stats["pre_direction_short_run"]),
        ("pre_direction_kept", "meta", pre_direction_stats["pre_direction_kept"]),
        ("direction_input_blobs", "meta", direction_stats["direction_input_blobs"]),
        ("valid_segments", "meta", direction_stats["valid_segments"]),
        ("direction_assigned_blobs", "meta", direction_stats["direction_assigned_blobs"]),
        ("accepted_objects", "meta", save_stats_dict["accepted_objects"]),
        ("object_pool_rows", "meta", save_stats_dict["object_pool_rows"]),
        ("saved_images", "meta", save_stats_dict["saved_images"]),
        ("saved_previews", "meta", save_stats_dict["saved_previews"]),
        ("skipped_empty_frames", "meta", int(save_stats_dict.get("skipped_empty_frames", 0))),
        ("empty_frame_removed_blobs", "meta", int(save_stats_dict.get("empty_frame_removed_blobs", 0))),
        ("direction_classes", "meta", len(DIRECTION_CLASS_NAMES)),
        ("output_frame_interval", "meta", int(cfg.get("FRAME_INTERVAL", 1))),
        ("direction_frame_interval", "meta", 1),
    ]
    for metric, kind, value in meta_items:
        add_metric(metric, kind, value)

    for reason, count in sorted((save_stats_dict.get("empty_frame_reason_counts") or {}).items()):
        add_metric(f"empty_frame_reason:{reason}", "removed_blob_count", int(count))

    if kept_rows:
        for metric in ("w", "h", "sum", "area"):
            arr = np.asarray([float(r[metric]) for r in kept_rows], dtype=np.float64)
            add_metric_with_source(metric, "min", float(np.min(arr)))
            add_metric_with_source(metric, "max", float(np.max(arr)))
            add_metric(metric, "q1", float(np.percentile(arr, 25)))
            add_metric(metric, "q3", float(np.percentile(arr, 75)))
            add_metric(metric, "median", float(np.median(arr)))
            add_metric(metric, "mean", float(np.mean(arr)))
    else:
        for metric in ("w", "h", "sum", "area"):
            add_metric(metric, "min", "")
            add_metric(metric, "max", "")
            add_metric(metric, "q1", "")
            add_metric(metric, "q3", "")
            add_metric(metric, "median", "")
            add_metric(metric, "mean", "")

    jump_rows = [
        {"frame": int(b.frame), "bbox_index": int(b.blob_index), "jump": float(b.jump_distance_px)}
        for fid in sorted(blob_records.keys())
        for b in blob_records[fid]
        if b.jump_distance_px is not None
    ]
    if jump_rows:
        jump_arr = np.asarray([float(r["jump"]) for r in jump_rows], dtype=np.float64)
        jump_min_row = min(jump_rows, key=lambda r: (float(r["jump"]), int(r["frame"]), int(r["bbox_index"])))
        jump_max_row = max(jump_rows, key=lambda r: (float(r["jump"]), -int(r["frame"]), -int(r["bbox_index"])))
        add_metric("jump", "min", float(np.min(jump_arr)), jump_min_row["frame"], jump_min_row["bbox_index"], "", "")
        add_metric("jump", "max", float(np.max(jump_arr)), jump_max_row["frame"], jump_max_row["bbox_index"], "", "")
        add_metric("jump", "q1", float(np.percentile(jump_arr, 25)))
        add_metric("jump", "q3", float(np.percentile(jump_arr, 75)))
        add_metric("jump", "median", float(np.median(jump_arr)))
        add_metric("jump", "mean", float(np.mean(jump_arr)))
    else:
        for kind in ("min", "max", "q1", "q3", "median", "mean"):
            add_metric("jump", kind, "")

    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "kind", "value_px", "frame", "bbox_index", "w", "h"])
        writer.writerows(summary_rows)
        for r in kept_rows:
            writer.writerow([
                "bbox",
                "kept",
                numeric_or_text(r["area"]),
                int(r["frame"]),
                int(r["bbox_index"]),
                numeric_or_text(r["w"]),
                numeric_or_text(r["h"]),
            ])

    return stats_path

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python direction_class_assignment.py config.yaml")

    cfg = load_config(sys.argv[1])
    session_path = str(cfg["SESSION_PATH"])
    pickle_path = str(cfg["PICKLE_PATH"])
    initial_tracking_csv_path = str(cfg["INIT_CSV_PATH"])
    initial_tracking_stats_path = os.path.join(
        session_path, "initial_tracking", "tracking_stats.csv"
    )

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(pickle_path)
    if not os.path.exists(initial_tracking_csv_path):
        raise FileNotFoundError(initial_tracking_csv_path)
    if not os.path.exists(initial_tracking_stats_path):
        raise FileNotFoundError(initial_tracking_stats_path)

    with open(pickle_path, "rb") as f:
        blob_seq = resolve_blob_sequence(pickle.load(f))
    initial_tracking_assignments = load_initial_tracking_csv(initial_tracking_csv_path)
    initial_tracking_stats = load_tracking_stats_csv(initial_tracking_stats_path)

    training_cfg = cfg.get("training", {}) or {}
    first_frame = int(training_cfg.get("FIRST_FRAME", 0))
    last_frame = int(training_cfg.get("LAST_FRAME", -1))
    output_frame_interval = max(1, int(cfg.get("FRAME_INTERVAL", 1)))
    direction_frame_interval = 1

    pickle_frame_count = get_num_blob_frames(blob_seq)
    frame_info = read_video_frame_info(str(cfg["TRAINING_VIDEO_PATH"]))
    warn_if_frame_count_adjusted(frame_info, label="create_single_animals_images training video")
    max_frames = min(pickle_frame_count, frame_info.usable_frame_count)
    first_frame, last_frame = clamp_frame_range_to_usable_count(
        first_frame, last_frame, max_frames,
    )
    print(
        "[create_single_animals_images] frame range: "
        f"{first_frame}..{last_frame} ({last_frame - first_frame + 1} frames), "
        f"pickle_frames={pickle_frame_count}, usable_video_frames={frame_info.usable_frame_count}"
    )

    direction_frame_indices = list(range(first_frame, last_frame + 1, direction_frame_interval))
    output_frame_indices = list(range(first_frame, last_frame + 1, output_frame_interval))

    out_dir = os.path.join(session_path, SINGLE_ANIMAL_IMAGES_DIR_NAME)
    reset_without_crossing_output_dir(out_dir)
    ensure_dirs([out_dir])

    individual_distance_px = float(get_required_tracking_stat(initial_tracking_stats, "individual_distance_px", float))
    individual_distance_source_count = int(get_required_tracking_stat(initial_tracking_stats, "individual_distance_source_count", int))
    individual_distance_source_total_non_crossing = int(get_required_tracking_stat(initial_tracking_stats, "individual_distance_source_total_non_crossing", int))
    init_max_dist_px = float(get_required_tracking_stat(initial_tracking_stats, "init_max_dist_px", float))

    blob_records = build_blob_records(blob_seq, direction_frame_indices)
    min_valid_frames, min_valid_ratio, source_fps = resolve_direction_min_valid_frames(cfg)
    cfg["_derived_source_fps"] = float(source_fps)
    cfg["_derived_direction_min_valid_frames"] = int(min_valid_frames)
    cfg["_derived_direction_min_valid_frames_fps_ratio"] = float(min_valid_ratio)
    bounds = classify_blobs_stage1(
        blob_records,
        cfg,
        individual_distance_px=individual_distance_px,
        individual_distance_source_count=individual_distance_source_count,
        individual_distance_source_total_non_crossing=individual_distance_source_total_non_crossing,
        init_max_dist_px=init_max_dist_px,
    )

    assign_track_ids_from_initial_tracking(
        blob_records=blob_records,
        assignments=initial_tracking_assignments,
    )
    jump_stats = classify_jump_outliers_stage1(blob_records, cfg, float(bounds.get("individual_distance_px", 0.0)))
    bounds["jump_bounds"] = tuple(jump_stats["jump_bounds"])
    bounds["jump_threshold_px"] = float(jump_stats["jump_threshold_px"])

    pre_direction_stats = classify_blobs_pre_direction(blob_records, cfg)
    pre_direction_run_csv_path = save_pre_direction_run_csv(blob_records, out_dir)
    direction_stats = estimate_directions(blob_records, cfg, float(bounds.get("individual_distance_px", 0.0)))
    refine_delete_stats = apply_refine_deletions(blob_records, load_refine_delete_map(session_path))

    cfg["_derived_jump_threshold_px"] = float(bounds.get("jump_threshold_px", 0.0))
    cfg["_derived_init_max_dist_px"] = float(bounds.get("init_max_dist_px", 0.0))
    cfg["_derived_traj_max_dist_px"] = float(bounds.get("traj_max_dist_px", 0.0))
    cfg["_derived_direction_min_total_disp_px"] = float(bounds.get("direction_min_total_disp_px", (0.0, 0.0))[0])
    cfg["INIT_MAX_DIST_PX"] = float(bounds.get("init_max_dist_px", 0.0))
    cfg["TRAJ_MAX_DIST_PX"] = float(bounds.get("traj_max_dist_px", 0.0))

    classification_csv_path = save_blob_classification_csv(blob_records, bounds, out_dir)
    individual_distance_csv_path = save_individual_distance_csv(out_dir, bounds, cfg)
    save_stats_dict = save_without_crossing_dataset(blob_records, cfg, output_frame_indices=output_frame_indices, bounds=bounds)
    save_stats_dict.update(refine_delete_stats)
    stats_path = save_stats(blob_records, pre_direction_stats, direction_stats, save_stats_dict, cfg, out_dir, bounds)

    print(f"Saved single_animal_images segments to: {out_dir}")
    print(f"Skipped empty single_animal_images frames: {int(save_stats_dict.get('skipped_empty_frames', 0))}")
    for reason, count in sorted((save_stats_dict.get("empty_frame_reason_counts") or {}).items()):
        print(f"  empty-frame removal reason {reason}: {count}")
    print(f"Saved blob classification to: {classification_csv_path}")
    print(f"Saved individual distance info to: {individual_distance_csv_path}")
    print(f"Saved pre-direction runs to: {pre_direction_run_csv_path}")
    print(f"Saved manifest to: {os.path.join(out_dir, 'object_pool', 'manifest.csv')}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
