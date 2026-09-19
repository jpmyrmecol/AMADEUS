# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Contrastive learning for ID-switch correction (idtracker.ai v6, Torrents et al. 2026)."""

from __future__ import annotations

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)


import os
import csv
import gc
import hashlib
import itertools
import json
import math
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import cv2
import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as tv_models
from batch_utils import resolve_device, tqdm, empty_accelerator_cache, get_accelerator_memory
from tracking_artifacts import artifact_path as tracking_artifact_path

from without_direction_estimation.multi_staged_association import (
    tuple_to_obb,
    valid_obb,
    ensure_clockwise,
    angular_distance_deg,
    canonicalize_obb_points,
)
from without_direction_estimation.obb_detection import iou_obb
from random_utils import derive_seed, make_numpy_rng, make_python_rng, normalize_seed
from segmentation_metadata import segmentation_paths_for_session
from segmentation_core import SegConfig, build_static_roi_mask, compute_foreground_mask

# Constants (paper-specified, not exposed to GUI)
_EMBED_DIM = 8
_D_POS = 1.0
_D_NEG = 10.0
# Hard veto for identity-best decisions only: if even the identity
# permutation's worst individual pairing is this far apart, the centroids
# aren't trustworthy enough to confirm "no swap happened".
_IDENTITY_ABS_MAX_DIST = 5.0
# A swap/relabel is accepted purely on relative separation from the
# alternatives -- not on max_matched_dist (still logged, but not a swap
# veto). Empirically, an absolute-distance veto rejects real swaps whose
# supporting evidence (a low best_cost/identity_cost ratio) is otherwise
# strong, while cost-ratio alone already rejects the weak/ambiguous
# swap-best cases that the absolute veto was also catching. best_cost must
# be at most this fraction of min(identity_cost, second_cost); min() folds
# together "beat identity" and "beat the runner-up permutation" into one
# comparison: at roster == 2 there are only two permutations, so
# second_cost (the non-winning one) always equals identity_cost when the
# winner is a swap, and the test reduces to a relative-improvement-over-
# identity check. At roster == 3, if some other permutation undercuts
# identity (second_cost < identity_cost), the test tightens to require the
# winner to be decisively better than that runner-up instead.
_ASSIGN_COST_RATIO_MAX = 0.5
_ASSIGN_MAX_ROSTER = 3
_LR = 1.0e-3
_BATCH_PAIRS = 400          # positive pairs per batch = negative pairs per batch
_GPU_MEMORY_LOG_BATCHES = 3
_CUDA_CROP_CACHE_MAX_FREE_FRACTION = 0.25
_CUDA_CROP_CACHE_SAFETY_BYTES = 1024 ** 3
# CUDA-only manual clamp. With VRAM autotune enabled, an integer is an upper
# bound for the measured batch size; None lets autotune use the measured budget.
# With autotune disabled, an integer keeps the legacy fixed CUDA pair count.
# Changing the batch pair count changes BatchNorm batch statistics and the loss
# composition per optimizer step, so training results will not be bit-identical.
_CUDA_BATCH_PAIRS_OVERRIDE: int | None = None
# --- VRAM-adaptive batch sizing ---
_VRAM_AUTOTUNE_ENABLE = True
_VRAM_BUDGET_FRACTION = 0.90
_VRAM_PROBE_PAIRS = (64, 160)
_VRAM_PROBE_WARMUP_BATCHES = 5
_VRAM_BATCH_PAIRS_MIN = 32
_VRAM_BATCH_PAIRS_MAX = 4096
_VRAM_BATCH_PAIRS_ROUND = 16
_VRAM_RESTART_GUARD_FRACTION = 0.95
_VRAM_MAX_RESTARTS = 1
_VRAM_CALIB_CACHE_NAME = 'embedding_vram_calib.json'
_CHECK_EVERY_BASE = 100     # minimum batch interval between SS checks
_CHECK_EVERY_SCALE = 5      # batches per object between SS checks
_SS_STOP_CONSECUTIVE = 30   # max consecutive no-improve checks before stop
_SS_EARLY_CONSECUTIVE = 2   # consecutive no-improve checks if SS >= 0.91 already
_SS_TARGET = 0.91
_SCORE_ALPHA = 0.5          # blend weight between size-based and loss-based sampling
_SCORE_DECAY = 0.98         # per-batch score decay
_SCORE_BUMP = 1.0           # score bump for non-zero-loss pairs
_MIN_FRAGMENT_FRAMES = 4
_INFER_BATCH = 512
_CROP_CACHE_NAME = 'embedding.h5'
_MODEL_CACHE_NAME = 'embedding.pt'
_PREVIEW_DIR_NAME = 'embedding_preview'
_TRAINING_METRICS_NAME = 'embedding_training_metrics.csv'
_CROP_METADATA_VERSION = 2
_CROP_NEIGHBOR_EXPAND = 1.2
_CROP_RECORDS_NAME = 'crop_records'
_WHITE_BACKGROUND_KEY = '__white_background__'
# Tier 3 opt-in: True uses GPU PyTorch silhouette (same statistic as sklearn,
# but float32 rounding may change boundary decisions in rare edge cases).
_USE_TORCH_SILHOUETTE = False
# DataLoader workers for CPU crop loading in _embed_crop_rows; 0 = synchronous.
# Set > 0 on Linux/macOS; Windows spawn-mode startup cost may negate gains for
# small inference passes.
_CPU_LOADER_WORKERS = 0
# Set > 0 to print per-phase timings for the first N training batches (0 = off).
_PROFILE_BATCHES = 0


def tqdm_it(*args, **kwargs):
    kw = dict(file=sys.stdout, dynamic_ncols=True, mininterval=0.2)
    kw.update(kwargs)
    return tqdm(*args, **kw)


def _worker_count(num_workers: int | None, total_items: int | None = None) -> int:
    n = max(1, int(num_workers or 1))
    if total_items is not None and total_items > 0:
        n = min(n, int(total_items))
    return n


def _configure_torch_parallelism(num_workers: int | None, device: torch.device | None = None) -> None:
    if device is not None and device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        return
    if device is None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return
    torch.set_num_threads(_worker_count(num_workers))


def _embedding_device_from_config(device='auto', purpose: str = 'embedding') -> torch.device:
    if isinstance(device, torch.device):
        return device

    resolved = resolve_device(device, purpose=purpose)
    s = str(resolved).strip().lower()
    if s == 'cpu':
        return torch.device('cpu')
    if s == 'cuda':
        return torch.device('cuda:0')
    if s.startswith('cuda:'):
        s = s[5:]

    first = s.split(',')[0].strip()
    if first.isdigit():
        if ',' in s:
            print(
                f'  [WARN] {purpose} supports one torch device; using cuda:{first}.',
                flush=True,
            )
        return torch.device(f'cuda:{int(first)}')

    try:
        return torch.device(s)
    except (RuntimeError, TypeError, ValueError):
        print(
            f"  [WARN] {purpose} device='{resolved}' is not a valid torch device; using CPU.",
            flush=True,
        )
        return torch.device('cpu')


# Data structures
@dataclass
class Fragment:
    frag_id: int
    obj_id: int
    frames: list[int]           # sorted frame indices
    headings: list[float] = field(default_factory=list)
    crop_indices: list[int] = field(default_factory=list)  # into HDF5 rows


# Fragment construction
def _valid_heading(dir_buf: dict, obj_id: int, frame: int) -> float | None:
    raw = dir_buf.get(obj_id, {}).get(frame)
    if raw is None or len(raw) == 0:
        return None
    heading = float(raw[0])
    if not math.isfinite(heading):
        return None
    return heading % 360.0


def _append_fragment(fragments: list[Fragment], obj_id: int, frames: list[int], headings: list[float]) -> None:
    if len(frames) >= _MIN_FRAGMENT_FRAMES:
        fragments.append(Fragment(frag_id=-1, obj_id=obj_id, frames=list(frames), headings=list(headings)))


def _isolation_by_frame(
    obb_buf: dict,
    dir_buf: dict,
    num_objects: int,
) -> dict[tuple[int, int], bool]:
    """Precompute per-(obj_id, frame) isolation with AABB pruning.  [A]

    Returns {(obj_id, frame): True/False} only for entries where both
    valid_heading and valid_obb hold (matching the guards in
    _build_fragments_for_object).  For the blocker side only valid_obb is
    required, mirroring the original inner loop.  AABB non-overlap is a cheap
    sufficient condition for IoU=0 -- only AABB-overlapping pairs call iou_obb.
    """
    all_frames: set[int] = set()
    for obj_id in range(num_objects):
        all_frames.update(obb_buf.get(obj_id, {}).keys())

    result: dict[tuple[int, int], bool] = {}

    for fr in all_frames:
        # All objects with valid OBB at this frame (used as potential blockers).
        frame_obbs: dict[int, np.ndarray] = {}
        frame_aabbs: dict[int, tuple[float, float, float, float]] = {}
        for obj_id in range(num_objects):
            obb_t = obb_buf.get(obj_id, {}).get(fr)
            if obb_t is None:
                continue
            obb = tuple_to_obb(obb_t)
            if not valid_obb(obb):
                continue
            pts = obb
            frame_obbs[obj_id] = obb
            frame_aabbs[obj_id] = (
                float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()),
            )

        # Candidate objects also need a valid heading (same guard as the original).
        for obj_id in range(num_objects):
            if _valid_heading(dir_buf, obj_id, fr) is None:
                continue
            if obj_id not in frame_obbs:
                continue
            obb_a = frame_obbs[obj_id]
            ax1, ay1, ax2, ay2 = frame_aabbs[obj_id]
            is_isolated = True
            for other_id, obb_b in frame_obbs.items():
                if other_id == obj_id:
                    continue
                bx1, by1, bx2, by2 = frame_aabbs[other_id]
                # AABB non-overlap -> IoU must be 0; skip the exact check.
                if ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1:
                    continue
                if iou_obb(obb_a, obb_b) > 0:
                    is_isolated = False
                    break
            result[(obj_id, fr)] = is_isolated

    return result


def _build_fragments_for_object(
    obb_buf: dict,
    dir_buf: dict,
    obj_id: int,
    num_objects: int,
    isolation_map: dict[tuple[int, int], bool] | None = None,
) -> list[Fragment]:
    obj_frames = sorted(obb_buf.get(obj_id, {}).keys())
    if not obj_frames:
        return []

    # Find frames where this object's OBB is isolated (IoU=0 with all others).
    isolated: list[tuple[int, float]] = []
    for fr in obj_frames:
        heading = _valid_heading(dir_buf, obj_id, fr)
        if heading is None:
            continue
        obb_t = obb_buf[obj_id].get(fr)
        if obb_t is None:
            continue
        obb = tuple_to_obb(obb_t)
        if not valid_obb(obb):
            continue
        if isolation_map is not None:
            # [A] Use precomputed per-frame isolation map (AABB-pruned).
            is_isolated = isolation_map.get((obj_id, fr), False)
        else:
            is_isolated = True
            for other_id in range(num_objects):
                if other_id == obj_id:
                    continue
                obb_t2 = obb_buf.get(other_id, {}).get(fr)
                if obb_t2 is None:
                    continue
                obb2 = tuple_to_obb(obb_t2)
                if not valid_obb(obb2):
                    continue
                if iou_obb(obb, obb2) > 0:
                    is_isolated = False
                    break
        if is_isolated:
            isolated.append((fr, heading))

    if not isolated:
        return []

    fragments: list[Fragment] = []
    run_frames = [isolated[0][0]]
    run_headings = [isolated[0][1]]
    for (prev, prev_heading), (cur, cur_heading) in zip(isolated, isolated[1:]):
        continuous = cur == prev + 1
        heading_jump = angular_distance_deg(prev_heading, cur_heading) >= 90.0
        if continuous and not heading_jump:
            run_frames.append(cur)
            run_headings.append(cur_heading)
        else:
            _append_fragment(fragments, obj_id, run_frames, run_headings)
            run_frames = [cur]
            run_headings = [cur_heading]
    _append_fragment(fragments, obj_id, run_frames, run_headings)
    return fragments


def build_fragments(corrected, num_objects, num_workers=1):
    """Select contiguous isolated observations, without an angular gate.

    Four-frame minimum is the existing embedding sampling requirement; it
    neither excludes stationary detections nor filters the tracking output.
    Fragment.headings stores crop rotations only for the shared cache format.
    """
    from without_direction_estimation.multi_staged_association import tuple_to_obb, valid_obb, iou_obb, aabbs_overlap, obb_aabb
    obbs = corrected['obb_buf']
    by_frame = {}
    for tid in range(num_objects):
        for frame, raw in obbs.get(tid, {}).items():
            pts = tuple_to_obb(raw)
            if valid_obb(pts):
                by_frame.setdefault(frame, {})[tid] = pts
    isolated = {tid: [] for tid in range(num_objects)}
    for frame, frame_obbs in sorted(by_frame.items()):
        boxes = {tid: obb_aabb(pts) for tid, pts in frame_obbs.items()}
        for tid, pts in frame_obbs.items():
            if not any(other != tid and aabbs_overlap(boxes[tid], boxes[other])
                       and iou_obb(pts, other_pts) > 0
                       for other, other_pts in frame_obbs.items()):
                isolated[tid].append((frame, horizontal_angle(pts)))
    fragments = []
    for tid, entries in isolated.items():
        run = []
        for entry in entries + [None]:
            if run and (entry is None or entry[0] != run[-1][0] + 1):
                if len(run) >= 4:
                    fragments.append(Fragment(frag_id=len(fragments), obj_id=tid,
                        frames=[v[0] for v in run], headings=[v[1] for v in run]))
                run = []
            if entry is not None:
                run.append(entry)
    return fragments


# Crop extraction
def _load_background_gray(background_path: str) -> np.ndarray | None:
    """Load a preferred background image, or return None for white fallback."""
    path = str(background_path or '').strip()
    if not path or not os.path.isfile(path):
        return None

    try:
        bg = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if bg is None:
            return None
        if bg.ndim == 2:
            gray = bg
        elif bg.ndim == 3 and bg.shape[2] == 1:
            gray = bg[:, :, 0]
        elif bg.ndim == 3 and bg.shape[2] == 3:
            gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        elif bg.ndim == 3 and bg.shape[2] == 4:
            gray = cv2.cvtColor(bg, cv2.COLOR_BGRA2GRAY)
        else:
            return None
    except (cv2.error, OSError):
        return None

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _background_cache_key(background_path: str) -> str:
    """Normalize valid backgrounds to absolute paths and fallback to a stable key."""
    path = str(background_path or '').strip()
    return os.path.abspath(path) if _load_background_gray(path) is not None else _WHITE_BACKGROUND_KEY


def _fill_convex_poly_in_roi(
    mask_roi: np.ndarray,
    pts_i: np.ndarray,
    rx1: int,
    ry1: int,
    rx2: int,
    ry2: int,
    frame_w: int,
    frame_h: int,
) -> None:
    """Fill a frame-space polygon into an ROI mask, matching full-frame rasterization."""
    if mask_roi.size == 0:
        return

    poly_x1 = int(pts_i[:, 0].min())
    poly_y1 = int(pts_i[:, 1].min())
    poly_x2 = int(pts_i[:, 0].max()) + 1
    poly_y2 = int(pts_i[:, 1].max()) + 1
    if poly_x2 <= rx1 or poly_y2 <= ry1 or poly_x1 >= rx2 or poly_y1 >= ry2:
        return

    origin_i = np.array([rx1, ry1], dtype=np.int32)
    if poly_x1 >= rx1 and poly_y1 >= ry1 and poly_x2 <= rx2 and poly_y2 <= ry2:
        cv2.fillConvexPoly(mask_roi, pts_i - origin_i, 255)
        return

    # OpenCV clips slanted polygon edges differently when drawing directly into
    # a clipped ROI. Draw only the needed surrounding region, then slice the ROI.
    px1 = max(0, min(rx1, poly_x1))
    py1 = max(0, min(ry1, poly_y1))
    px2 = min(frame_w, max(rx2, poly_x2))
    py2 = min(frame_h, max(ry2, poly_y2))
    if px2 <= px1 or py2 <= py1:
        return

    pad = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
    pad_origin = np.array([px1, py1], dtype=np.int32)
    cv2.fillConvexPoly(pad, pts_i - pad_origin, 255)
    pad_roi = pad[ry1 - py1:ry2 - py1, rx1 - px1:rx2 - px1]
    mask_roi[pad_roi > 0] = 255


def _expanded_crop_aabb(pts: np.ndarray) -> tuple[int, int, int, int]:
    center = pts.mean(axis=0)
    expanded_pts = center + _CROP_NEIGHBOR_EXPAND * (pts - center)
    pts_i = np.round(expanded_pts).astype(np.int32)
    return (
        int(pts_i[:, 0].min()),
        int(pts_i[:, 1].min()),
        int(pts_i[:, 0].max()) + 1,
        int(pts_i[:, 1].max()) + 1,
    )


def _crop_roi_aabb(pts: np.ndarray, frame_shape: tuple[int, int], img_size: int) -> tuple[int, int, int, int]:
    center = pts.mean(axis=0).astype(np.float32)
    h, w = frame_shape[:2]
    cx = float(center[0])
    cy = float(center[1])
    roi_half = int(math.ceil(int(img_size) * math.sqrt(0.5))) + 3
    return (
        min(w, max(0, int(math.floor(cx - roi_half)))),
        min(h, max(0, int(math.floor(cy - roi_half)))),
        min(w, max(0, int(math.ceil(cx + roi_half)))),
        min(h, max(0, int(math.ceil(cy + roi_half)))),
    )


def _build_frame_obb_info(frame_obbs: dict) -> dict[int, tuple[tuple, np.ndarray, tuple[int, int, int, int]]]:
    frame_obb_info = {}
    for oid, obb_t in frame_obbs.items():
        obb = tuple_to_obb(obb_t)
        if not valid_obb(obb):
            continue
        pts = canonicalize_obb_points(ensure_clockwise(obb))
        frame_obb_info[oid] = (obb_t, pts, _expanded_crop_aabb(pts))
    return frame_obb_info


def _pruned_other_obbs_for_crop(
    target_id: int,
    frame_obbs: dict,
    frame_obb_info: dict[int, tuple[tuple, np.ndarray, tuple[int, int, int, int]]],
    frame_shape: tuple[int, int],
    img_size: int,
) -> list:
    target_info = frame_obb_info.get(target_id)
    if target_info is None:
        return [obb for oid, obb in frame_obbs.items() if oid != target_id]

    _target_obb_t, target_pts, _target_aabb = target_info
    rx1, ry1, rx2, ry2 = _crop_roi_aabb(target_pts, frame_shape, img_size)
    other_obbs = []
    for oid, (obb_t, _pts, (ox1, oy1, ox2, oy2)) in frame_obb_info.items():
        if oid == target_id:
            continue
        if ox2 <= rx1 or oy2 <= ry1 or ox1 >= rx2 or oy1 >= ry2:
            continue
        other_obbs.append(obb_t)
    return other_obbs


@dataclass(frozen=True)
class _TrackingSegContext:
    """A resolved segmentation config + precomputed static ROI mask, ready
    to regenerate foreground masks for a tracking video's crops."""
    cfg: SegConfig
    roi_mask: "np.ndarray | None"


class EmbeddingSegmentationRequired(RuntimeError):
    """Raised when a session has no segmentation_gui_config.json: embedding
    crops require regenerating a foreground mask from this session's
    segmentation settings (see _load_tracking_seg_config), and no longer
    fall back to OBB-based painting when segmentation was never run at all.
    Callers should treat this the same as embedding being unusable -- abort
    embedding-based ID resolution and continue with geometry-only
    post-processing -- rather than letting it crash the whole job."""


_SEG_CONFIG_FILENAME = 'segmentation_gui_config.json'
_SEG_MODES_NEEDING_BACKGROUND = {
    'background_diff', 'dark_region_and_background_diff', 'bright_region_and_background_diff',
}


def _load_tracking_seg_config(
    session_path: str, video_path: str, frame_shape: tuple[int, int]
) -> "_TrackingSegContext | None":
    """Load this session's already-decided segmentation settings and
    background image, to regenerate a fresh foreground mask per crop on a
    tracking video assumed to share the same physical recording setup.

    Returns None when no segmentation was run for this session (the
    embedding crop pipeline then falls back to its prior
    other-OBB-painting behavior unchanged) -- this is not a new manual
    setting, just detecting whether segmentation_gui_config.json exists.
    """
    session_path = str(session_path or '').strip()
    if not session_path:
        return None
    _pickle_path, background_path = segmentation_paths_for_session(session_path, video_path)
    seg_dir = os.path.dirname(background_path) if background_path else ''
    config_path = os.path.join(seg_dir, _SEG_CONFIG_FILENAME) if seg_dir else ''
    if not config_path or not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        settings = raw['settings']
        mode = str(settings['segmentation_mode'])
        cfg = SegConfig(
            mode=mode,
            ksize=int(settings.get('blur_ksize', 1)),
            threshold=int(settings.get('threshold', 0)),
            dark_threshold=int(settings.get('dark_threshold', 0)),
            bright_threshold=int(settings.get('bright_threshold', 0)),
            diff_threshold=int(settings.get('diff_threshold', 0)),
            open_iter=int(settings.get('open_iter', 0)),
            close_iter=int(settings.get('close_iter', 0)),
            fill_holes=bool(settings.get('fill_holes', False)),
            invert_mask=bool(settings.get('invert_mask', False)),
            expand_px=max(0, int(settings.get('region_expand_px', 0))),
            expand_merge_only=bool(settings.get('region_expand_merge_only', False)),
            background_bgr=None,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'Malformed segmentation config for tracking: {config_path}: {exc}') from exc

    if mode in _SEG_MODES_NEEDING_BACKGROUND:
        bg = cv2.imread(background_path, cv2.IMREAD_COLOR)
        if bg is None:
            raise RuntimeError(
                f'Segmentation mode {mode!r} requires a background image, '
                f'but it could not be read: {background_path}'
            )
        if bg.shape[:2] != tuple(frame_shape[:2]):
            raise RuntimeError(
                'Segmentation background image size must match the tracking video frame size: '
                f'background={bg.shape[:2]}, frame={tuple(frame_shape[:2])}'
            )
        cfg.background_bgr = bg

    roi_mask = build_static_roi_mask(settings.get('roi_sets', []) or [], frame_shape)
    return _TrackingSegContext(cfg=cfg, roi_mask=roi_mask)


def _probe_video_frame_shape(video_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open video to determine frame size: {video_path}')
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()
    if height <= 0 or width <= 0:
        raise RuntimeError(f'Invalid video frame size probed for: {video_path}')
    return height, width


def _resolve_seg_config(session_path: str, video_path: str) -> "_TrackingSegContext | None":
    """Probe the tracking video's frame size and load this session's
    segmentation settings, if any (see _load_tracking_seg_config)."""
    if not str(session_path or '').strip():
        return None
    frame_shape = _probe_video_frame_shape(video_path)
    return _load_tracking_seg_config(session_path, video_path, frame_shape)


def _compute_frame_labels(
    frame_bgr: np.ndarray, seg_config: "_TrackingSegContext | None"
) -> "tuple[np.ndarray, int] | None":
    """Foreground segmentation + connected-components labeling for one whole
    frame, to be computed once per frame and shared by every individual's
    crop in that frame (see callers). Runs on the full frame rather than a
    per-crop ROI so the pixel operations and boundary conditions (Gaussian
    blur, morphology, floodFill) match GUI segmentation's own full-frame
    computation exactly. Returns None when no segmentation config is
    available for this session.
    """
    if seg_config is None:
        return None
    fg_mask = compute_foreground_mask(frame_bgr, seg_config.cfg, seg_config.roi_mask)
    num_labels, labels = cv2.connectedComponents(fg_mask, connectivity=8)
    return labels, num_labels


def _extract_crop_from_gray(
    frame_gray: np.ndarray,
    background_gray: np.ndarray | None,
    obb_t: tuple,
    heading_deg: float,
    img_size: int,
    other_obbs=None,
    frame_labels: "tuple[np.ndarray, int] | None" = None,
) -> np.ndarray:
    heading_deg = horizontal_angle(obb_t)
    """Extract a heading-aligned crop from already-grayscale frame data.

    This uses a direct source-to-output affine transform, so it warps only the
    img_size x img_size crop instead of rotating the full video frame for every
    OBB.  The result is equivalent to rotating the full frame around the animal
    center and then taking the centered square patch. The output background is
    always flat black (0) -- everything except the target individual, whether
    excluded via frame_labels or via other_obbs below -- matching idtracker.ai's
    "target individual only, black background" identification-image convention.
    background_gray is only used to validate BACKGROUND_PATH's configured size;
    its pixels are never pasted into the crop.

    frame_labels is (labels, num_labels) from cv2.connectedComponents on that
    *whole frame's* freshly regenerated segmentation foreground mask (see
    _compute_frame_labels) -- computed once per frame by the caller and shared
    across every individual's crop in that frame, not recomputed here. This
    keeps only the connected component overlapping the target OBB, blanking
    everything else -- arena background *and* any other individual in one step,
    since a non-overlapping neighbor is necessarily a separate connected
    component. A touching/fused neighbor is kept as part of the same
    component (never split at the OBB boundary), matching "not fused =
    independent individual". Falls back to the legacy other_obbs painting when
    frame_labels is None, or when the labeling doesn't cover the target OBB at
    all.
    """
    img_size = int(img_size)
    obb = tuple_to_obb(obb_t)
    if not valid_obb(obb):
        raise ValueError(f'Invalid OBB for embedding crop: {obb_t}')

    pts = canonicalize_obb_points(ensure_clockwise(obb))
    center = pts.mean(axis=0).astype(np.float32)

    h, w = frame_gray.shape[:2]
    if background_gray is not None and background_gray.shape[:2] != (h, w):
        raise RuntimeError(
            'BACKGROUND_PATH image size must match the video frame size: '
            f'background={background_gray.shape[:2]}, frame={(h, w)}'
        )

    cx = float(center[0])
    cy = float(center[1])
    roi_half = int(math.ceil(img_size * math.sqrt(0.5))) + 3
    rx1 = min(w, max(0, int(math.floor(cx - roi_half))))
    ry1 = min(h, max(0, int(math.floor(cy - roi_half))))
    rx2 = min(w, max(0, int(math.ceil(cx + roi_half))))
    ry2 = min(h, max(0, int(math.ceil(cy + roi_half))))
    roi = frame_gray[ry1:ry2, rx1:rx2]

    # Step 1: keep only the target individual within the ROI, blanking
    # everything else -- arena background and any other individual -- to
    # black (0). Prefers the precomputed full-frame segmentation labeling
    # (frame_labels); falls back to painting other OBBs' 1.2x expanded
    # regions black when no labeling is available or it doesn't cover the
    # target OBB, so the two paths always produce the same background.
    roi_composite = roi.copy()
    origin = np.array([rx1, ry1], dtype=np.float64)
    if roi_composite.size == 0:
        return np.zeros((img_size, img_size), dtype=np.uint8)

    roi_h, roi_w = roi.shape[:2]
    masked_ok = False

    if frame_labels is not None:
        labels_full, num_labels = frame_labels
        if num_labels > 1:
            labels_roi = labels_full[ry1:ry2, rx1:rx2]
            target_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            _fill_convex_poly_in_roi(
                target_mask, np.round(pts).astype(np.int32), rx1, ry1, rx2, ry2, w, h,
            )
            overlap = np.bincount(labels_roi[target_mask > 0].ravel(), minlength=num_labels)
            overlap[0] = 0  # component 0 is background; never select it as "the individual"
            target_label = int(np.argmax(overlap))
            if overlap[target_label] > 0:
                roi_composite[labels_roi != target_label] = 0
                masked_ok = True

    if not masked_ok and other_obbs:
        c_t = pts.mean(axis=0)
        pts_target_exp = c_t + _CROP_NEIGHBOR_EXPAND * (pts - c_t)
        target_protect = np.zeros((roi_h, roi_w), dtype=np.uint8)
        _fill_convex_poly_in_roi(
            target_protect, np.round(pts_target_exp).astype(np.int32),
            rx1, ry1, rx2, ry2, w, h,
        )

        forbidden = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for ob in other_obbs:
            p = tuple_to_obb(ob)
            if not valid_obb(p):
                continue
            c = p.mean(axis=0)
            p_exp = c + _CROP_NEIGHBOR_EXPAND * (p - c)
            _fill_convex_poly_in_roi(
                forbidden, np.round(p_exp).astype(np.int32),
                rx1, ry1, rx2, ry2, w, h,
            )
        forbidden[target_protect > 0] = 0
        roi_composite[forbidden > 0] = 0

    # Step 2: rotate composite so heading points up, then crop to img_size square.
    # Project heading_deg follows 0=up, 90=right.  An OpenCV positive rotation
    # subtracts that angle from vectors in this convention, so rotating by the
    # heading itself makes the animal point upward in the crop.
    M = cv2.getRotationMatrix2D((cx, cy), float(heading_deg), 1.0)
    x1 = int(round(cx - img_size / 2.0))
    y1 = int(round(cy - img_size / 2.0))
    M_local = M.copy()
    M_local[:, 2] += M[:, :2] @ origin
    M_local[0, 2] -= float(x1)
    M_local[1, 2] -= float(y1)

    return cv2.warpAffine(
        roi_composite, M_local, (img_size, img_size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _prepare_frame_to_crops(fragments: list[Fragment]) -> dict[int, list[tuple[Fragment, int]]]:
    for frag in fragments:
        frag.crop_indices.clear()

    frame_to_crops: dict[int, list[tuple[Fragment, int]]] = {}
    for frag in fragments:
        for fi, fr in enumerate(frag.frames):
            frame_to_crops.setdefault(fr, []).append((frag, fi))

    row_idx = 0
    for fr in sorted(frame_to_crops.keys()):
        for frag, _fi in frame_to_crops[fr]:
            frag.crop_indices.append(row_idx)
            row_idx += 1
    return frame_to_crops


def _fragment_crop_records(fragments: list[Fragment]) -> list[tuple[int, int, int, int]]:
    """Return records as (row, obj_id, frame, frag_id) for all cached crops."""
    records: list[tuple[int, int, int, int]] = []
    for frag in fragments:
        for fi, frame in enumerate(frag.frames):
            row = int(frag.crop_indices[fi])
            obj_id = int(frag.obj_id)
            frag_id = int(frag.frag_id)
            frame_i = int(frame)
            records.append((row, obj_id, frame_i, frag_id))
    records.sort(key=lambda x: x[0])
    return records


def _fragment_crop_record_array(fragments: list[Fragment]) -> np.ndarray:
    """Return rows as (row, obj_id, frame, frag_id, heading)."""
    records: list[tuple[float, float, float, float, float]] = []
    for frag in fragments:
        for fi, frame in enumerate(frag.frames):
            row = int(frag.crop_indices[fi])
            records.append((
                float(row),
                float(frag.obj_id),
                float(frame),
                float(frag.frag_id),
                float(frag.headings[fi]),
            ))
    records.sort(key=lambda x: x[0])
    return np.asarray(records, dtype=np.float64).reshape(-1, 5)


def _fragments_from_crop_record_array(records: np.ndarray) -> list[Fragment]:
    records = np.asarray(records, dtype=np.float64).reshape(-1, 5)
    fragments_by_id: dict[int, Fragment] = {}
    for row, obj_id, frame, frag_id, heading in records[np.argsort(records[:, 0])]:
        fid = int(round(float(frag_id)))
        frag = fragments_by_id.get(fid)
        if frag is None:
            frag = Fragment(
                frag_id=fid,
                obj_id=int(round(float(obj_id))),
                frames=[],
            )
            fragments_by_id[fid] = frag
        frag.frames.append(int(round(float(frame))))
        frag.headings.append(float(heading) % 360.0)
        frag.crop_indices.append(int(round(float(row))))
    return [fragments_by_id[fid] for fid in sorted(fragments_by_id)]


def _seg_config_signature(seg_config: "_TrackingSegContext | None") -> str:
    """Hash of everything a regenerated foreground mask depends on, so the
    crop cache is invalidated when segmentation settings/background change."""
    if seg_config is None:
        return ''
    cfg = seg_config.cfg
    h = hashlib.sha256()
    h.update(repr((
        cfg.mode, cfg.ksize, cfg.threshold, cfg.dark_threshold, cfg.bright_threshold,
        cfg.diff_threshold, cfg.open_iter, cfg.close_iter, cfg.fill_holes, cfg.invert_mask,
        cfg.expand_px, cfg.expand_merge_only,
    )).encode('utf-8'))
    if cfg.background_bgr is not None:
        h.update(cfg.background_bgr.tobytes())
    if seg_config.roi_mask is not None:
        h.update(seg_config.roi_mask.tobytes())
    return h.hexdigest()


def _write_crop_cache_metadata(
    hf: h5py.File, fragments: list[Fragment], background_path: str, seg_config_signature: str = '',
) -> None:
    hf.attrs['background_path'] = _background_cache_key(background_path)
    hf.attrs['seg_config_signature'] = seg_config_signature
    hf.attrs['crop_metadata_version'] = int(_CROP_METADATA_VERSION)
    hf.create_dataset(_CROP_RECORDS_NAME, data=_fragment_crop_record_array(fragments))


def _read_crop_cache_metadata(
    h5_path: str,
    background_path: str,
    seg_config_signature: str | None = None,
) -> tuple[list[Fragment], int] | None:
    if not os.path.exists(h5_path):
        return None
    try:
        with h5py.File(h5_path, 'r') as hf:
            if 'crops' not in hf or _CROP_RECORDS_NAME not in hf:
                return None
            crops_shape = tuple(hf['crops'].shape)
            if len(crops_shape) != 3 or int(crops_shape[1]) != int(crops_shape[2]):
                return None
            cached_bg = str(hf.attrs.get('background_path', ''))
            current_bg = _background_cache_key(background_path)
            if cached_bg != current_bg:
                return None
            if seg_config_signature is not None:
                if str(hf.attrs.get('seg_config_signature', '')) != seg_config_signature:
                    return None
            if int(hf.attrs.get('crop_metadata_version', 0)) != int(_CROP_METADATA_VERSION):
                return None
            records = np.asarray(hf[_CROP_RECORDS_NAME][:], dtype=np.float64).reshape(-1, 5)
            if int(len(records)) != int(crops_shape[0]):
                return None
            return _fragments_from_crop_record_array(records), int(crops_shape[1])
    except (OSError, KeyError, ValueError):
        return None


def load_cached_crop_metadata(out_dir: str, background_path: str) -> tuple[list[Fragment], int] | None:
    """Return (fragments, img_size) from embedding.h5 when metadata is available."""
    h5_path = os.path.join(out_dir, _CROP_CACHE_NAME)
    return _read_crop_cache_metadata(h5_path, background_path)


@dataclass(frozen=True)
class _TrainingMetricsState:
    exists: bool
    done: bool
    best_ss: float | None


def _training_metrics_state(out_dir: str) -> _TrainingMetricsState:
    metrics_path = os.path.join(out_dir, _TRAINING_METRICS_NAME)
    exists = os.path.exists(metrics_path)
    rows = _read_training_metric_rows(metrics_path)
    best_values: list[float] = []
    done = False
    for row in rows:
        if row.get('event') == 'done':
            done = True
        best_ss = _float_or_none(row.get('best_ss'))
        if best_ss is not None and math.isfinite(best_ss):
            best_values.append(float(best_ss))
    return _TrainingMetricsState(exists, done, max(best_values) if best_values else None)


def _embedding_model_cache_complete(out_dir: str, metrics_state: _TrainingMetricsState | None = None) -> bool:
    model_path = os.path.join(out_dir, _MODEL_CACHE_NAME)
    if not os.path.exists(model_path):
        return False
    state = metrics_state if metrics_state is not None else _training_metrics_state(out_dir)
    # Old caches may not have a metrics CSV.  A present metrics CSV is trusted
    # only after the explicit done row has been written.
    return (not state.exists) or state.done


def load_cached_embedding_artifacts(
    out_dir: str,
    background_path: str,
    num_objects: int,
    seed: int = 0,
    device_config='auto',
    session_path: str = '',
    video_path: str = '',
    *,
    silhouette_fn=None,
) -> tuple[list[Fragment], dict[int, np.ndarray], float] | None:
    """Load existing embedding.pt + embedding.h5 without rebuilding or training."""
    if silhouette_fn is None:
        silhouette_fn = _compute_silhouette
    h5_path = os.path.join(out_dir, _CROP_CACHE_NAME)
    model_path = os.path.join(out_dir, _MODEL_CACHE_NAME)
    seg_config = _resolve_seg_config(session_path, video_path)
    if seg_config is None:
        raise EmbeddingSegmentationRequired(
            f'No segmentation configuration found for this session ({session_path!r}); '
            'embedding crops require segmentation to have been run first.'
        )
    cached_metadata = _read_crop_cache_metadata(
        h5_path, background_path, seg_config_signature=_seg_config_signature(seg_config),
    )
    metrics_state = _training_metrics_state(out_dir)
    if cached_metadata is None or not _embedding_model_cache_complete(out_dir, metrics_state):
        return None

    fragments, _img_size = cached_metadata
    if not fragments:
        return fragments, {}, 0.0

    device = _embedding_device_from_config(device_config, purpose='embedding cache')
    model = _EmbedNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    with h5py.File(h5_path, 'r') as hf:
        crops_np = hf['crops'][:]

    crops_t = _cache_crops_on_device(crops_np, device)
    best_ss = metrics_state.best_ss
    if best_ss is None:
        ss_sample_total = min(len(crops_np), 1000 * int(num_objects))
        best_ss = silhouette_fn(model, crops_np, fragments, ss_sample_total, device, crops_t=crops_t, seed=seed)

    centroids = fragment_embeddings(model, fragments, h5_path, crops_t=crops_t)
    return fragments, centroids, float(best_ss)


def _reset_preview_dir(out_dir: str) -> str:
    preview_dir = os.path.join(out_dir, _PREVIEW_DIR_NAME)
    os.makedirs(preview_dir, exist_ok=True)

    # Keep the directory as a preview of the current crops, not an archive.
    for name in os.listdir(preview_dir):
        if name.lower().endswith('.png'):
            os.remove(os.path.join(preview_dir, name))
    return preview_dir


def _preview_filename(preview_idx: int, obj_id: int, frame: int, frag_id: int, row: int) -> str:
    return (
        f'preview{preview_idx:02d}_id{obj_id:03d}_'
        f'frame{frame:06d}_frag{frag_id:05d}_row{row:08d}.png'
    )


def write_initial_crop_previews(
    corrected: dict,
    video_path: str,
    out_dir: str,
    img_size: int,
    background_path: str,
    fragments: list[Fragment],
    preview_count: int = 100,
    seg_config: "_TrackingSegContext | None" = None,
) -> list[str]:
    """Write preview crops directly from the video before full HDF5 extraction."""
    count = int(preview_count)
    if count <= 0:
        return []

    records: list[tuple[int, int, int, int, float]] = []
    for frag in fragments:
        for fi, frame in enumerate(frag.frames):
            row = int(frag.crop_indices[fi])
            obj_id = int(frag.obj_id)
            frag_id = int(frag.frag_id)
            frame_i = int(frame)
            heading = float(frag.headings[fi])
            records.append((row, obj_id, frame_i, frag_id, heading))
    records.sort(key=lambda x: x[0])
    if not records:
        return []

    preview_dir = _reset_preview_dir(out_dir)
    background_gray = _load_background_gray(background_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video for embedding crop previews: {video_path}')

    obb_buf = corrected['obb_buf']
    written: list[str] = []
    chosen = records[:min(count, len(records))]
    # records are sorted by row, not frame, so consecutive entries can jump
    # between frames -- this remembers only the single most-recently-computed
    # frame's labeling, recomputing on every actual frame change, rather than
    # holding the whole video's masks in memory.
    last_labels_frame: int | None = None
    last_frame_labels: "tuple[np.ndarray, int] | None" = None
    try:
        for preview_idx, (row, obj_id, frame, frag_id, heading) in enumerate(chosen, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                raise RuntimeError(f'Failed to read frame {frame} for embedding crop preview: {video_path}')
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if frame != last_labels_frame:
                last_frame_labels = _compute_frame_labels(frame_bgr, seg_config)
                last_labels_frame = frame
            frame_obbs = {
                oid: obb_buf[oid][frame]
                for oid in obb_buf
                if frame in obb_buf.get(oid, {})
            }
            frame_obb_info = _build_frame_obb_info(frame_obbs)
            obb_t = frame_obbs[obj_id]
            other_obbs = _pruned_other_obbs_for_crop(
                obj_id, frame_obbs, frame_obb_info, frame_gray.shape[:2], img_size
            )
            crop = _extract_crop_from_gray(
                frame_gray, background_gray, obb_t, heading, img_size, other_obbs,
                frame_labels=last_frame_labels,
            )

            filename = _preview_filename(preview_idx, obj_id, frame, frag_id, row)
            path = os.path.join(preview_dir, filename)
            if not cv2.imwrite(path, crop):
                raise RuntimeError(f'Failed to write embedding crop preview: {path}')
            written.append(path)
    finally:
        cap.release()

    return written


def write_crop_previews(
    h5_path: str,
    fragments: list[Fragment],
    out_dir: str,
    preview_count: int = 100,
) -> list[str]:
    """Write only the first cached OBB crops as PNG previews.

    Output directory: <out_dir>/embedding_preview/
    Filename format: preview<NN>_id<ID>_frame<FRAME>_frag<FRAG>_row<ROW>.png
    """
    count = int(preview_count)
    if count <= 0:
        return []

    records = _fragment_crop_records(fragments)
    if not records or not os.path.exists(h5_path):
        return []

    preview_dir = _reset_preview_dir(out_dir)
    written: list[str] = []
    with h5py.File(h5_path, 'r') as hf:
        crops = hf['crops']

        chosen = records[:min(count, len(records))]
        for preview_idx, (row, obj_id, frame, frag_id) in enumerate(chosen, start=1):
            crop = np.asarray(crops[int(row)], dtype=np.uint8)
            filename = _preview_filename(preview_idx, obj_id, frame, frag_id, row)
            path = os.path.join(preview_dir, filename)
            if not cv2.imwrite(path, crop):
                raise RuntimeError(f'Failed to write embedding crop preview: {path}')
            written.append(path)

    return written


def extract_and_cache_crops(
    corrected: dict,
    video_path: str,
    out_dir: str,
    img_size: int,
    background_path: str,
    frame_to_crops: dict[int, list[tuple[Fragment, int]]],
    fragments: list[Fragment],
    num_workers: int = 1,
    seg_config: "_TrackingSegContext | None" = None,
) -> str:
    """Extract OBB crops for all fragments; write to HDF5. Returns HDF5 path.

    A valid BACKGROUND_PATH is preferred.  Empty, missing, or unreadable paths
    use deterministic white-background crops and cache metadata.

    Speed notes:
    - The video is still read in frame order, but crop generation is pipelined
      through a bounded CPU worker pool when num_workers > 1.
    - Each video frame is converted to grayscale once.
    - Each OBB crop uses direct img_size x img_size affine warps instead of
      rotating the full frame/background image for every OBB.
    """
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, _CROP_CACHE_NAME)

    obb_buf = corrected['obb_buf']
    background_gray = _load_background_gray(background_path)

    # Build frame -> list[(frag, frame_idx)] for video scan.
    if not frame_to_crops:
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('crops', shape=(0, img_size, img_size), dtype=np.uint8)
            _write_crop_cache_metadata(hf, fragments, background_path, _seg_config_signature(seg_config))
        return h5_path

    total_frames_needed = sorted(frame_to_crops.keys())
    total_images = sum(len(v) for v in frame_to_crops.values())
    crops_arr = np.zeros((total_images, img_size, img_size), dtype=np.uint8)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video for embedding crops: {video_path}')

    num_w = _worker_count(num_workers, total_images)
    executor = ThreadPoolExecutor(max_workers=num_w) if num_w > 1 else None
    max_pending = max(1, num_w * 8)
    pending = set()

    prev_cv_threads = None
    if num_w > 1:
        prev_cv_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)

    def crop_spec(
        item: tuple[Fragment, int],
        frame_obbs: dict,
        frame_obb_info: dict[int, tuple[tuple, np.ndarray, tuple[int, int, int, int]]],
        frame_shape: tuple[int, int],
    ) -> tuple[int, tuple, float, list]:
        frag, fi = item
        row = int(frag.crop_indices[fi])
        obb_t = frame_obbs[frag.obj_id]
        other_obbs = _pruned_other_obbs_for_crop(
            frag.obj_id, frame_obbs, frame_obb_info, frame_shape, img_size
        )
        return row, obb_t, float(frag.headings[fi]), other_obbs

    def crop_job(
        row: int, frame_gray: np.ndarray, obb_t: tuple, h_deg: float, other_obbs: list,
        frame_labels: "tuple[np.ndarray, int] | None",
    ) -> tuple[int, np.ndarray]:
        return row, _extract_crop_from_gray(
            frame_gray, background_gray, obb_t, h_deg, img_size, other_obbs,
            frame_labels=frame_labels,
        )

    def consume_done(done: set) -> None:
        for fut in done:
            row, crop = fut.result()
            crops_arr[int(row)] = crop

    def flush_pending(block: bool) -> None:
        nonlocal pending
        if not pending:
            return
        if block:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
        else:
            done = {fut for fut in pending if fut.done()}
            pending -= done
        if done:
            consume_done(done)

    # cur_frame_idx is the next frame index expected from cap.read().
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cur_frame_idx = 0
    sequential_grab_skip_max = 64

    def read_target_frame(target_fr: int) -> tuple[bool, np.ndarray | None]:
        nonlocal cur_frame_idx
        target_fr = int(target_fr)
        if target_fr < cur_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_fr)
            cur_frame_idx = target_fr
        elif target_fr > cur_frame_idx:
            gap = target_fr - cur_frame_idx
            if gap <= sequential_grab_skip_max:
                ok = True
                for _ in range(gap):
                    ok = bool(cap.grab())
                    cur_frame_idx += 1
                    if not ok:
                        break
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_fr)
                    cur_frame_idx = target_fr
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_fr)
                cur_frame_idx = target_fr

        ret, frame_bgr = cap.read()
        if ret:
            cur_frame_idx = target_fr + 1
            return True, frame_bgr
        cur_frame_idx = target_fr
        return False, None

    try:
        with tqdm_it(total=len(total_frames_needed), desc='Embedding extract crops', unit='frame') as pbar:
            for target_fr in total_frames_needed:
                ret, frame_bgr = read_target_frame(target_fr)
                if not ret or frame_bgr is None:
                    raise RuntimeError(f'Failed to read frame {target_fr} for embedding crops: {video_path}')

                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                # Segmentation + connected-components labeling is computed
                # once per frame here and shared by every individual's crop
                # below, instead of being recomputed per individual.
                frame_labels = _compute_frame_labels(frame_bgr, seg_config)
                frame_obbs = {
                    oid: obb_buf[oid][target_fr]
                    for oid in obb_buf
                    if target_fr in obb_buf[oid]
                }
                frame_obb_info = _build_frame_obb_info(frame_obbs)
                for item in frame_to_crops[target_fr]:
                    row, obb_t, h_deg, other_obbs = crop_spec(
                        item, frame_obbs, frame_obb_info, frame_gray.shape[:2]
                    )
                    if executor is None:
                        row, crop = crop_job(row, frame_gray, obb_t, h_deg, other_obbs, frame_labels)
                        crops_arr[row] = crop
                    else:
                        pending.add(executor.submit(crop_job, row, frame_gray, obb_t, h_deg, other_obbs, frame_labels))
                        if len(pending) >= max_pending:
                            flush_pending(block=True)

                if executor is not None:
                    flush_pending(block=False)
                pbar.update(1)

            while pending:
                flush_pending(block=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if prev_cv_threads is not None:
            cv2.setNumThreads(prev_cv_threads)
        cap.release()

    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('crops', data=crops_arr)
        _write_crop_cache_metadata(hf, fragments, background_path, _seg_config_signature(seg_config))
    return h5_path


# Model
class _EmbedNet(nn.Module):
    """ResNet18 (v1) contrastive embedding, per idtracker.ai v6.

    Two modifications to torchvision's ResNet18: conv1 is changed to
    single-channel for grayscale input, and the output is a bias-free
    fully-connected layer with _EMBED_DIM units using identity activation.
    """

    def __init__(self):
        super().__init__()
        net = tv_models.resnet18(weights=None, num_classes=_EMBED_DIM)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # torchvision's resnet18() Kaiming-inits every conv layer during
        # super().__init__(), but that runs before conv1 is replaced above, so
        # the replacement would otherwise silently fall back to nn.Conv2d's
        # plain default init instead of matching every other conv layer's
        # scheme. idtracker.ai v6 re-applies this same init after the same
        # replacement (idtrackerai.base.network.models.ResNet18.__init__).
        nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
        net.fc = nn.Linear(net.fc.in_features, _EMBED_DIM, bias=False)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return axial_forward(self.net, x)


def _chopra_pair_losses(z1: torch.Tensor, z2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Chopra 2005 contrastive loss per pair. labels=1 for positive, 0 for negative."""
    d = (z1 - z2).norm(dim=1)
    pos = labels * (torch.clamp(d - _D_POS, min=0.0) ** 2)
    neg = (1.0 - labels) * (torch.clamp(_D_NEG - d, min=0.0) ** 2)
    return pos + neg


@dataclass(frozen=True)
class _SampledPair:
    row_a: int
    row_b: int
    label: int
    frag_i: int
    frag_j: int


# Loss-aware hard sampling
class _PairSampler:
    """Size + loss-score blended sampler with loss-triggered score updates."""

    def __init__(self, fragments: list[Fragment], seed: int = 0):
        self.fragments = fragments
        self.np_rng = make_numpy_rng(seed, "pair_sampler", "numpy")
        self.py_rng = make_python_rng(seed, "pair_sampler", "python")
        n = len(fragments)
        # Score arrays indexed by fragment index. Initialized to 10 (not 1),
        # matching idtracker.ai v6 (ContrastiveLearning.loss_scores): scores
        # are always normalized by their own sum before use (_blend_weights),
        # so the initial value doesn't change the starting distribution
        # either way, but it does change how sharply one early bump swings
        # it -- from 1, a single bump nearly doubles a pair's weight before
        # the embedding is even remotely converged; from 10, the same bump
        # is a ~10% nudge, ramping in the hard-pair bias more gradually.
        self.pos_scores = np.full(n, 10.0, dtype=np.float64)
        self.neg_scores = np.full((n, n), 10.0, dtype=np.float64)
        self._fi_sizes = np.array([len(f.frames) for f in fragments], dtype=np.float64)
        self._negative_candidates = self._build_negative_candidates()

        # [D] Cache eligible index lists and size weights; avoids per-batch rebuild.
        # crop_indices are populated by _prepare_frame_to_crops before __init__ is called.
        self._pos_eligible: list[int] = [
            i for i, frag in enumerate(fragments) if len(frag.crop_indices) >= 2
        ]
        self._pos_size_w: np.ndarray = np.array(
            [len(fragments[fi].crop_indices) for fi in self._pos_eligible],
            dtype=np.float64,
        )
        if self._negative_candidates:
            _neg_idx = np.array(self._negative_candidates, dtype=np.intp)
            self._neg_fi_idx: np.ndarray = _neg_idx[:, 0]
            self._neg_fj_idx: np.ndarray = _neg_idx[:, 1]
            self._neg_size_w: np.ndarray = (
                self._fi_sizes[self._neg_fi_idx] + self._fi_sizes[self._neg_fj_idx]
            )
        else:
            self._neg_fi_idx = np.empty(0, dtype=np.intp)
            self._neg_fj_idx = np.empty(0, dtype=np.intp)
            self._neg_size_w = np.empty(0, dtype=np.float64)

    @staticmethod
    def _temporally_overlap(a: Fragment, b: Fragment) -> bool:
        i = 0
        j = 0
        while i < len(a.frames) and j < len(b.frames):
            fa = a.frames[i]
            fb = b.frames[j]
            if fa == fb:
                return True
            if fa < fb:
                i += 1
            else:
                j += 1
        return False

    @staticmethod
    def _blend_weights(score_w: np.ndarray, size_w: np.ndarray) -> np.ndarray:
        if (
            score_w.size == 0
            or score_w.sum() <= 0.0
            or size_w.sum() <= 0.0
            or not np.isfinite(score_w).all()
            or not np.isfinite(size_w).all()
        ):
            return np.full(len(score_w), 1.0 / max(1, len(score_w)), dtype=np.float64)
        weights = _SCORE_ALPHA * (score_w / score_w.sum()) + (1.0 - _SCORE_ALPHA) * (size_w / size_w.sum())
        if weights.sum() <= 0.0 or not np.isfinite(weights).all():
            return np.full(len(score_w), 1.0 / max(1, len(score_w)), dtype=np.float64)
        return weights / weights.sum()

    def _build_negative_candidates(self) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        for i, frag_i in enumerate(self.fragments):
            if not frag_i.crop_indices:
                continue
            for j in range(i + 1, len(self.fragments)):
                frag_j = self.fragments[j]
                if (
                    frag_i.obj_id != frag_j.obj_id
                    and frag_j.crop_indices
                    and self._temporally_overlap(frag_i, frag_j)
                ):
                    candidates.append((i, j))
        return candidates

    def _decay(self) -> None:
        self.pos_scores *= _SCORE_DECAY
        self.neg_scores *= _SCORE_DECAY

    def update_scores(self, pairs: list[_SampledPair], losses: torch.Tensor) -> None:
        # [B] Vectorised score updates via np.add.at (handles duplicate indices
        # correctly, identical to sequential += loop, bit-exact for float64 + 1.0).
        loss_values = losses.detach().cpu().numpy()
        pos_fi: list[int] = []
        neg_fi: list[int] = []
        neg_fj: list[int] = []
        for pair, pair_loss in zip(pairs, loss_values):
            if float(pair_loss) <= 0.0:
                continue
            if pair.label == 1:
                pos_fi.append(pair.frag_i)
            else:
                neg_fi.append(pair.frag_i)
                neg_fj.append(pair.frag_j)
        if pos_fi:
            np.add.at(self.pos_scores, pos_fi, _SCORE_BUMP)
        if neg_fi:
            np.add.at(self.neg_scores, (neg_fi, neg_fj), _SCORE_BUMP)
            np.add.at(self.neg_scores, (neg_fj, neg_fi), _SCORE_BUMP)
        self._decay()

    def sample_positive(self, batch_size: int) -> list[_SampledPair]:
        """Sample positive pairs as two crop rows from the same fragment."""
        # [D] Use cached eligible list and size_w; batch-sample with np.random.choice.
        eligible = self._pos_eligible
        if not eligible:
            return []
        frags = self.fragments
        # score_w order matches original list comprehension over eligible.
        score_w = np.array([self.pos_scores[fi] for fi in eligible], dtype=np.float64)
        frag_w = self._blend_weights(score_w, self._pos_size_w)
        chosen = self.np_rng.choice(len(eligible), size=batch_size, p=frag_w)
        pairs: list[_SampledPair] = []
        for ci in chosen:
            fi = int(eligible[int(ci)])
            row_a, row_b = self.py_rng.sample(frags[fi].crop_indices, 2)
            pairs.append(_SampledPair(int(row_a), int(row_b), 1, fi, fi))
        return pairs

    def sample_negative(self, batch_size: int) -> list[_SampledPair]:
        """Sample negative pairs from different-ID fragments with overlapping frames."""
        # [D] Use cached eligible list, size_w, and index arrays; batch-sample.
        eligible = self._negative_candidates
        if not eligible:
            return []
        frags = self.fragments
        # score_w order matches original list comprehension over eligible.
        score_w = self.neg_scores[self._neg_fi_idx, self._neg_fj_idx]
        pair_w = self._blend_weights(score_w, self._neg_size_w)
        chosen = self.np_rng.choice(len(eligible), size=batch_size, p=pair_w)
        pairs: list[_SampledPair] = []
        for ci in chosen:
            fi, fj = eligible[int(ci)]
            ra = self.py_rng.choice(frags[fi].crop_indices)
            rb = self.py_rng.choice(frags[fj].crop_indices)
            pairs.append(_SampledPair(int(ra), int(rb), 0, fi, fj))
        return pairs


# Training

class _CropDataset(torch.utils.data.Dataset):
    """Maps HDF5 row indices to float32 (1,H,W) tensors for DataLoader workers.  [C]

    Conversion uint8 -> float32 / 255 is done per-item in __getitem__ so it can
    be parallelised across workers; the default collate stacks results into
    (B, 1, H, W), bit-identical to the numpy batch conversion in the CPU path.
    """

    def __init__(self, crops_np: np.ndarray, rows: list[int]) -> None:
        self._crops = crops_np
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        crop = self._crops[self._rows[idx]]          # (H, W) uint8
        return torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)


def _log_gpu_memory(device: torch.device, batch_idx: int) -> None:
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f'  [gpu memory batch {batch_idx}] '
            f'allocated={allocated / (1024 ** 3):.2f} GiB, '
            f'reserved={reserved / (1024 ** 3):.2f} GiB, '
            f'free={free_bytes / (1024 ** 3):.2f} GiB',
            flush=True,
        )
    elif device.type == 'mps':
        # torch.mps has no memory_allocated()/mem_get_info() equivalent; report
        # the same driver_allocated/recommended_max pair used by calibration.
        used, budget = get_accelerator_memory(device)
        if used is not None and budget:
            print(
                f'  [mps memory batch {batch_idx}] '
                f'driver_allocated={used / (1024 ** 3):.2f} GiB, '
                f'recommended_max={budget / (1024 ** 3):.2f} GiB',
                flush=True,
            )


def _cache_crops_on_device(crops_np: np.ndarray, device: torch.device) -> torch.Tensor | None:
    """Keep uint8 crops on CUDA when there is enough free VRAM.

    The HDF5 cache stores uint8 crops as (N, H, W).  Training repeatedly indexes
    these rows, converts them to float32, and normalizes them per batch.  CUDA
    runs may keep the uint8 tensor on-device to avoid repeated host-to-device
    transfer, but skip the cache when it could risk shared GPU memory fallback.
    CPU runs use the per-batch numpy path.
    """
    if device.type != 'cuda' or crops_np.size == 0:
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    needed_bytes = int(crops_np.nbytes)
    if needed_bytes + _CUDA_CROP_CACHE_SAFETY_BYTES <= free_bytes * _CUDA_CROP_CACHE_MAX_FREE_FRACTION:
        print(
            f'  CUDA crop cache: enabled '
            f'({needed_bytes / (1024 ** 3):.2f} GiB, free {free_bytes / (1024 ** 3):.2f} GiB)',
            flush=True,
        )
        return torch.from_numpy(crops_np[:, None, :, :]).to(device=device)
    print(
        f'  CUDA crop cache: disabled '
        f'({needed_bytes / (1024 ** 3):.2f} GiB would risk shared GPU memory fallback)',
        flush=True,
    )
    return None


def _fallback_batch_pairs(device: torch.device) -> int:
    if device.type == 'cuda' and _CUDA_BATCH_PAIRS_OVERRIDE is not None:
        return int(_CUDA_BATCH_PAIRS_OVERRIDE)
    return int(_BATCH_PAIRS)


def _round_down_batch_pairs(batch_pairs: int | float) -> int:
    round_to = max(1, int(_VRAM_BATCH_PAIRS_ROUND))
    return int(math.floor(float(batch_pairs) / round_to) * round_to)


def _clamp_autotuned_batch_pairs(batch_pairs: int | float) -> int:
    value = _round_down_batch_pairs(batch_pairs)
    value = max(int(_VRAM_BATCH_PAIRS_MIN), min(int(_VRAM_BATCH_PAIRS_MAX), value))
    if _CUDA_BATCH_PAIRS_OVERRIDE is not None:
        value = min(value, int(_CUDA_BATCH_PAIRS_OVERRIDE))
        value = _round_down_batch_pairs(value)
        value = max(int(_VRAM_BATCH_PAIRS_MIN), value)
    return max(1, int(value))


def _scaled_check_every(num_objects: int, batch_pairs: int) -> int:
    check_every = max(_CHECK_EVERY_BASE, _CHECK_EVERY_SCALE * int(num_objects))
    return max(1, int(round(check_every * (_BATCH_PAIRS / max(1, int(batch_pairs))))))


def _shrink_batch_pairs_for_restart(batch_pairs: int) -> int:
    shrunk = _round_down_batch_pairs(float(batch_pairs) * 0.90)
    return max(int(_VRAM_BATCH_PAIRS_MIN), int(shrunk))


def _cuda_rng_states() -> list[torch.Tensor] | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_rng_state_all()


def _restore_cuda_rng_states(states: list[torch.Tensor] | None) -> None:
    if states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states)


def _autotune_log(batch_pairs: int, F: float, slope: float, budget: float, cached: bool = False) -> None:
    suffix = ', cached' if cached else ''
    print(
        f'  VRAM autotune: batch_pairs={batch_pairs} '
        f'(F={F / (1024 ** 3):.2f} GiB, '
        f'slope={slope / (1024 ** 2):.2f} MiB/pair, '
        f'budget={budget / (1024 ** 3):.2f} GiB{suffix})',
        flush=True,
    )


def _autotuned_batch_pairs_from_fit(F: float, slope: float, ceiling: float) -> tuple[int, float]:
    budget = float(ceiling) * float(_VRAM_BUDGET_FRACTION)
    raw = math.floor((budget - float(F)) / float(slope))
    return _clamp_autotuned_batch_pairs(raw), budget


def _autotune_batch_pairs(
    fragments: list[Fragment],
    sampler: "_PairSampler",
    get_tensor,
    device: torch.device,
    img_size: int,
    out_dir: str,
    crop_cache_enabled: bool,
) -> int:
    if device.type != 'cuda' or not _VRAM_AUTOTUNE_ENABLE:
        return _fallback_batch_pairs(device)

    fallback = _fallback_batch_pairs(device)
    if fallback <= 0:
        raise ValueError(f'batch_pairs must be positive, got {fallback}')

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_index)
    props = torch.cuda.get_device_properties(device_index)
    total_vram = int(props.total_memory)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    current_ceiling = float(torch.cuda.memory_reserved(device) + free_bytes)
    cache_path = tracking_artifact_path(out_dir, _VRAM_CALIB_CACHE_NAME)

    def _cache_matches(data: dict) -> bool:
        return (
            data.get('gpu_name') == gpu_name
            and int(data.get('total_vram', -1)) == total_vram
            and int(data.get('img_size', -1)) == int(img_size)
            and bool(data.get('crop_cache_enabled')) == bool(crop_cache_enabled)
            and int(data.get('embed_dim', -1)) == int(_EMBED_DIM)
        )

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            saved_ceiling = float(cached.get('ceiling', 0.0))
            F = float(cached.get('F', 0.0))
            slope = float(cached.get('slope', 0.0))
            ceiling_delta = abs(current_ceiling - saved_ceiling) / max(saved_ceiling, 1.0)
            if _cache_matches(cached) and slope > 0.0 and ceiling_delta <= 0.10:
                bp_opt, budget = _autotuned_batch_pairs_from_fit(F, slope, current_ceiling)
                _autotune_log(bp_opt, F, slope, budget, cached=True)
                return bp_opt
        except Exception as exc:
            print(f'  VRAM autotune: ignored calibration cache ({exc})', flush=True)

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = _cuda_rng_states()
    np_rng_state = np.random.get_state()
    py_rng_state = random.getstate()

    model = None
    optimizer = None
    peaks: dict[int, int] = {}
    try:
        model = _EmbedNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
        model.train()

        for bp in sorted(int(x) for x in _VRAM_PROBE_PAIRS):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(int(_VRAM_PROBE_WARMUP_BATCHES)):
                pos_pairs = sampler.sample_positive(bp)
                neg_pairs = sampler.sample_negative(bp)
                all_pairs = pos_pairs + neg_pairs
                if not all_pairs:
                    return fallback
                rows_a = [p.row_a for p in all_pairs]
                rows_b = [p.row_b for p in all_pairs]
                za = model(get_tensor(rows_a))
                zb = model(get_tensor(rows_b))
                lab = torch.tensor([p.label for p in all_pairs], dtype=torch.float32, device=device)
                loss = _chopra_pair_losses(za, zb, lab).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize(device)
            peaks[int(bp)] = int(torch.cuda.max_memory_reserved(device))

        lo, hi = sorted(peaks.keys())[:2]
        slope = (float(peaks[hi]) - float(peaks[lo])) / float(hi - lo)
        F = float(peaks[lo]) - slope * float(lo)
        if slope <= 0.0 or not math.isfinite(slope) or not math.isfinite(F):
            print('  VRAM autotune: unusable calibration slope; using default batch_pairs.', flush=True)
            return fallback

        free_bytes, _ = torch.cuda.mem_get_info(device)
        ceiling = float(torch.cuda.memory_reserved(device) + free_bytes)
        bp_opt, budget = _autotuned_batch_pairs_from_fit(F, slope, ceiling)

        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'gpu_name': gpu_name,
                        'total_vram': total_vram,
                        'img_size': int(img_size),
                        'crop_cache_enabled': bool(crop_cache_enabled),
                        'embed_dim': int(_EMBED_DIM),
                        'F': float(F),
                        'slope': float(slope),
                        'ceiling': float(ceiling),
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
        except Exception as exc:
            print(f'  VRAM autotune: failed to write calibration cache ({exc})', flush=True)

        _autotune_log(bp_opt, F, slope, budget)
        return bp_opt
    except RuntimeError as exc:
        if 'out of memory' in str(exc).lower():
            torch.cuda.empty_cache()
            min_pairs = _clamp_autotuned_batch_pairs(_VRAM_BATCH_PAIRS_MIN)
            print(
                f'  VRAM autotune: probe ran out of memory; using batch_pairs={min_pairs}.',
                flush=True,
            )
            return min_pairs
        raise
    finally:
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.random.set_rng_state(cpu_rng_state)
        _restore_cuda_rng_states(cuda_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(py_rng_state)


def _read_training_metric_rows(metrics_path: str) -> list[dict]:
    if not os.path.exists(metrics_path):
        return []
    rows: list[dict] = []
    try:
        with open(metrics_path, 'r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append(row)
    except Exception:
        return rows
    return rows


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return '0m 00s'
    total = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total, 60)
    return f'{minutes}m {secs:02d}s'


def train_embedding(
    fragments: list[Fragment],
    crops_path: str,
    out_dir: str,
    num_objects: int,
    num_workers: int = 1,
    seed: int = 0,
    max_batches: int | None = None,
    device_config='auto',
    *,
    silhouette_fn=None,
) -> tuple[_EmbedNet, float]:
    """Train embedding; return (model, best_silhouette_score)."""
    if silhouette_fn is None:
        silhouette_fn = _compute_silhouette
    device = _embedding_device_from_config(device_config, purpose='embedding training')
    _configure_torch_parallelism(num_workers, device)
    seed = normalize_seed(seed)
    python_seed = derive_seed(seed, "python")
    numpy_seed = derive_seed(seed, "numpy") % (2 ** 32)
    torch_seed = derive_seed(seed, "torch") % (2 ** 63 - 1)
    torch.manual_seed(torch_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(torch_seed)
    np.random.seed(numpy_seed)
    random.seed(python_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with h5py.File(crops_path, 'r') as hf:
        crops_np = hf['crops'][:]  # (N, H, W) uint8

    total_crops = len(crops_np)
    if total_crops < 2:
        return _EmbedNet().to(device), 0.0
    ss_sample_total = min(total_crops, 1000 * int(num_objects))

    crops_t = _cache_crops_on_device(crops_np, device)

    def get_tensor(rows: list[int]) -> torch.Tensor:
        if crops_t is not None:
            idx = torch.as_tensor(rows, dtype=torch.long, device=device)
            return crops_t.index_select(0, idx).to(torch.float32).div_(255.0)
        imgs = crops_np[rows]
        t = torch.from_numpy(imgs[:, None, :, :]).to(device=device, dtype=torch.float32)
        return t.div_(255.0)

    img_size = int(crops_np.shape[1]) if crops_np.ndim >= 2 else 0
    tune_sampler_seed = derive_seed(seed, "vram_autotune_sampler")
    train_sampler_seed = derive_seed(seed, "training_sampler")
    tune_sampler = _PairSampler(fragments, seed=tune_sampler_seed)
    if device.type == 'cuda' and _VRAM_AUTOTUNE_ENABLE:
        batch_pairs = _autotune_batch_pairs(
            fragments,
            tune_sampler,
            get_tensor,
            device,
            img_size,
            out_dir,
            crop_cache_enabled=(crops_t is not None),
        )
    else:
        batch_pairs = _fallback_batch_pairs(device)
    if batch_pairs <= 0:
        raise ValueError(f'batch_pairs must be positive, got {batch_pairs}')
    check_every = _scaled_check_every(num_objects, batch_pairs)
    _log_gpu_memory(device, 0)

    model = _EmbedNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
    sampler = _PairSampler(fragments, seed=train_sampler_seed)
    metrics_path = os.path.join(out_dir, _TRAINING_METRICS_NAME)
    metrics_fields = [
        'event',
        'batch',
        'loss',
        'ss',
        'best_ss',
        'no_improve',
        'stop_limit',
        'elapsed_sec',
    ]
    metrics_f = open(metrics_path, 'w', newline='', encoding='utf-8')
    metrics_writer = csv.DictWriter(metrics_f, fieldnames=metrics_fields)
    metrics_writer.writeheader()
    metrics_f.flush()
    print(f'  Writing embedding training metrics: {metrics_path}', flush=True)
    start_time = time.time()

    best_ss = -1.0
    consecutive_no_improve = 0
    batch_idx = 0
    best_state = None
    metric_flush_interval = 50

    def log_metric(loss_value: float, ss: float | None = None, stop_limit: int | None = None) -> None:
        metrics_writer.writerow({
            'event': 'metric',
            'batch': int(batch_idx),
            'loss': f'{float(loss_value):.8f}',
            'ss': '' if ss is None else f'{float(ss):.8f}',
            'best_ss': '' if best_ss < 0.0 else f'{float(best_ss):.8f}',
            'no_improve': int(consecutive_no_improve),
            'stop_limit': '' if stop_limit is None else int(stop_limit),
            'elapsed_sec': f'{time.time() - start_time:.3f}',
        })
        if ss is not None or batch_idx % metric_flush_interval == 0:
            metrics_f.flush()

    def log_done() -> None:
        elapsed = time.time() - start_time
        metrics_writer.writerow({
            'event': 'done',
            'batch': int(batch_idx),
            'loss': '',
            'ss': '',
            'best_ss': '' if best_ss < 0.0 else f'{float(best_ss):.8f}',
            'no_improve': int(consecutive_no_improve),
            'stop_limit': '',
            'elapsed_sec': f'{elapsed:.3f}',
        })
        metrics_f.flush()
        print(f'  Embedding training finished in {_format_elapsed(elapsed)}.', flush=True)

    profile_n = int(_PROFILE_BATCHES)
    _t: dict[str, float] = {}

    def _tick(key: str) -> None:
        if profile_n > 0:
            _t[key] = time.perf_counter()

    def _tock(key: str) -> float:
        return time.perf_counter() - _t[key] if profile_n > 0 else 0.0

    restarts = 0
    restart_requested = False
    attempt_failed = True
    try:
        while True:
            restart_requested = False
            attempt_failed = True
            model.train()
            with tqdm_it(total=check_every, desc='Embedding training', unit='batch') as pbar:
                while True:
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    _tick('sample')
                    pos_pairs = sampler.sample_positive(batch_pairs)
                    neg_pairs = sampler.sample_negative(batch_pairs)
                    t_sample = _tock('sample')

                    all_pairs = pos_pairs + neg_pairs
                    if not all_pairs:
                        break

                    rows_a = [p.row_a for p in all_pairs]
                    rows_b = [p.row_b for p in all_pairs]

                    _tick('forward')
                    za = model(get_tensor(rows_a))
                    zb = model(get_tensor(rows_b))
                    t_forward = _tock('forward')

                    lab = torch.tensor([p.label for p in all_pairs], dtype=torch.float32, device=device)
                    pair_losses = _chopra_pair_losses(za, zb, lab)
                    loss = pair_losses.mean()
                    loss_value = float(loss.detach().cpu().item())

                    _tick('backward')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    t_backward = _tock('backward')

                    sampler.update_scores(all_pairs, pair_losses)
                    batch_idx += 1
                    pbar.update(1)
                    if batch_idx <= _GPU_MEMORY_LOG_BATCHES:
                        _log_gpu_memory(device, batch_idx)

                    if (
                        device.type in ('cuda', 'mps')
                        and batch_idx <= _GPU_MEMORY_LOG_BATCHES
                        and restarts < int(_VRAM_MAX_RESTARTS)
                    ):
                        if device.type == 'cuda':
                            reserved = torch.cuda.memory_reserved(device)
                            _, total_bytes = torch.cuda.mem_get_info(device)
                        else:
                            # Apple Silicon unified memory: no CUDA-style total_vram,
                            # so this guards against the recommended MPS budget instead.
                            reserved, total_bytes = get_accelerator_memory(device)
                        if reserved is not None and total_bytes and reserved >= total_bytes * float(_VRAM_RESTART_GUARD_FRACTION):
                            new_batch_pairs = _shrink_batch_pairs_for_restart(batch_pairs)
                            if new_batch_pairs < batch_pairs:
                                print(
                                    f'  VRAM restart guard: reserved={reserved / (1024 ** 3):.2f} GiB '
                                    f'>= {float(_VRAM_RESTART_GUARD_FRACTION):.2f} '
                                    f'* total={total_bytes / (1024 ** 3):.2f} GiB; '
                                    f'restarting with batch_pairs={new_batch_pairs} '
                                    f'(was {batch_pairs}).',
                                    flush=True,
                                )
                                batch_pairs = new_batch_pairs
                                check_every = _scaled_check_every(num_objects, batch_pairs)
                                restarts += 1
                                restart_requested = True
                                break

                    if profile_n > 0 and batch_idx <= profile_n:
                        print(
                            f'  [profile batch {batch_idx}]'
                            f'  sample={t_sample:.3f}s'
                            f'  forward={t_forward:.3f}s'
                            f'  backward={t_backward:.3f}s'
                            f'  total={t_sample + t_forward + t_backward:.3f}s',
                            flush=True,
                        )

                    if batch_idx % check_every == 0:
                        _tick('ss')
                        ss = silhouette_fn(model, crops_np, fragments, ss_sample_total, device, crops_t=crops_t, seed=seed)
                        t_ss = _tock('ss')
                        _log_gpu_memory(device, batch_idx)
                        if profile_n > 0:
                            print(f'  [profile SS check]  silhouette={t_ss:.3f}s  ss={ss:.4f}', flush=True)
                        improved = ss > best_ss
                        if improved:
                            best_ss = ss
                            consecutive_no_improve = 0
                            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        else:
                            consecutive_no_improve += 1

                        stop_long = consecutive_no_improve >= _SS_STOP_CONSECUTIVE
                        stop_early = (consecutive_no_improve >= _SS_EARLY_CONSECUTIVE and best_ss >= _SS_TARGET)
                        stop_limit = _SS_EARLY_CONSECUTIVE if best_ss >= _SS_TARGET else _SS_STOP_CONSECUTIVE
                        log_metric(loss_value, ss=ss, stop_limit=stop_limit)
                        pbar.set_postfix(SS=f'{best_ss:.3f}', no_improve=f'{consecutive_no_improve}/{stop_limit}')
                        if stop_long or stop_early:
                            break
                        pbar.reset(total=check_every)
                    else:
                        log_metric(loss_value)
            attempt_failed = False
            if not restart_requested:
                break

            metrics_f.close()
            del optimizer, sampler, model, best_state
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            elif device.type == 'mps':
                empty_accelerator_cache(device)
            # Re-seed before rebuilding so a VRAM-triggered restart with the
            # same batch_pairs reproduces the same model init and sample order.
            torch.manual_seed(torch_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(torch_seed)
            model = _EmbedNet().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
            sampler = _PairSampler(fragments, seed=train_sampler_seed)
            best_ss = -1.0
            consecutive_no_improve = 0
            batch_idx = 0
            best_state = None
            metrics_f = open(metrics_path, 'w', newline='', encoding='utf-8')
            metrics_writer = csv.DictWriter(metrics_f, fieldnames=metrics_fields)
            metrics_writer.writeheader()
            metrics_f.flush()
            print(f'  Writing embedding training metrics: {metrics_path}', flush=True)
            start_time = time.time()
    finally:
        if not attempt_failed and not restart_requested:
            log_done()
        if not metrics_f.closed:
            metrics_f.close()

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model_path = os.path.join(out_dir, _MODEL_CACHE_NAME)
    torch.save(model.state_dict(), model_path)

    return model, float(best_ss)


def _embed_crop_rows(
    model: _EmbedNet,
    crops_np: np.ndarray,
    rows: list[int],
    device: torch.device,
    batch_size: int = _INFER_BATCH,
    progress_desc: str | None = None,
    progress_unit: str = 'crop',
    crops_t: torch.Tensor | None = None,
) -> np.ndarray:
    if not rows:
        return np.zeros((0, _EMBED_DIM), dtype=np.float32)

    was_training = model.training
    model.eval()
    out: list[np.ndarray] = []
    batch_size = max(1, int(batch_size))

    ranges = range(0, len(rows), batch_size)
    pbar = None
    if progress_desc is not None:
        pbar = tqdm_it(total=len(rows), desc=progress_desc, unit=progress_unit)

    try:
        with torch.inference_mode():
            if crops_t is not None:
                # CUDA path: uint8 on device; cast per batch.
                for start in ranges:
                    batch_rows = rows[start:start + batch_size]
                    idx = torch.as_tensor(batch_rows, dtype=torch.long, device=device)
                    t = crops_t.index_select(0, idx).to(torch.float32).div_(255.0)
                    out.append(model(t).cpu().numpy())
                    if pbar is not None:
                        pbar.update(len(batch_rows))
            else:
                # [C] CPU path: DataLoader with optional workers + pin_memory.
                _pin = device.type == 'cuda'
                _nw = _CPU_LOADER_WORKERS
                loader = torch.utils.data.DataLoader(
                    _CropDataset(crops_np, rows),
                    batch_size=batch_size,
                    num_workers=_nw,
                    pin_memory=_pin,
                    prefetch_factor=2 if _nw > 0 else None,
                    persistent_workers=False,
                    drop_last=False,
                )
                for t in loader:
                    t = t.to(device, non_blocking=_pin)
                    out.append(model(t).cpu().numpy())
                    if pbar is not None:
                        pbar.update(len(t))
    finally:
        if pbar is not None:
            pbar.close()
        if was_training:
            model.train()

    return np.concatenate(out, axis=0) if out else np.zeros((0, _EMBED_DIM), dtype=np.float32)


def _torch_silhouette_score(
    embeds: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> float:
    """PyTorch GPU mean silhouette width.  [E]

    Implements the same formula as sklearn.metrics.silhouette_score:
      a(i) = mean intra-cluster distance,
      b(i) = min mean distance to any other cluster,
      s(i) = (b(i) - a(i)) / max(a(i), b(i)).
    Computed in float32 on ``device``; the result is the same statistic as
    sklearn but floating-point rounding at cluster boundaries may differ in
    rare cases (see _USE_TORCH_SILHOUETTE).
    """
    n = len(labels)
    unique_labels, label_ids = np.unique(labels, return_inverse=True)
    k = int(len(unique_labels))
    if k < 2 or k >= n:
        return 0.0

    X = torch.from_numpy(embeds.astype(np.float32)).to(device)          # (n, d)
    L = torch.from_numpy(label_ids.astype(np.int64)).to(device)         # (n,)

    # Pairwise Euclidean distances (n, n).
    dist = (X.unsqueeze(1) - X.unsqueeze(0)).norm(dim=2)

    a = torch.zeros(n, dtype=torch.float32, device=device)
    b = torch.full((n,), float('inf'), dtype=torch.float32, device=device)

    for lbl_id in range(k):
        mask = L == lbl_id
        idx = mask.nonzero(as_tuple=True)[0]
        cnt = int(idx.shape[0])
        if cnt > 1:
            intra = dist[idx][:, idx]                                    # (cnt, cnt)
            a[idx] = (intra.sum(dim=1) - intra.diagonal()) / (cnt - 1)
        not_idx = (~mask).nonzero(as_tuple=True)[0]
        if len(not_idx) > 0:
            inter_mean = dist[not_idx][:, idx].mean(dim=1)
            b[not_idx] = torch.minimum(b[not_idx], inter_mean)

    denom = torch.maximum(a, b)
    s = torch.where(denom > 0, (b - a) / denom, torch.zeros_like(a))
    return float(s.mean().cpu().item())


def _compute_silhouette(
    model: _EmbedNet,
    crops_np: np.ndarray,
    fragments: list[Fragment],
    ss_sample_total: int,
    device: torch.device,
    crops_t: torch.Tensor | None = None,
    seed: int = 0,
) -> float:
    """Evaluate silhouette score using MiniBatchKMeans on sampled crops.

    The sampled-row RNG is re-seeded from (seed, "silhouette_rows") on every
    call rather than reused from a persistent generator, so repeated calls
    within the same training run (constant total_crops/sample_count) draw the
    same crop subset -- silhouette movement then reflects model updates, not
    subset churn.
    """
    num_objects = len({frag.obj_id for frag in fragments if frag.crop_indices})
    if num_objects < 2:
        return 0.0

    total_crops = int(len(crops_np))
    sample_count = min(int(ss_sample_total), total_crops)
    if sample_count < max(2, num_objects):
        return 0.0

    silhouette_rng = make_python_rng(seed, "silhouette_rows")
    sampled_rows = silhouette_rng.sample(range(total_crops), k=sample_count)
    embeds = _embed_crop_rows(model, crops_np, sampled_rows, device, crops_t=crops_t)

    kmeans_seed = derive_seed(seed, "silhouette_kmeans") % (2**32 - 1)
    kmeans = MiniBatchKMeans(n_clusters=num_objects, random_state=int(kmeans_seed), n_init='auto')
    cluster_labels = kmeans.fit_predict(embeds)
    cluster_count = len(set(cluster_labels))
    if cluster_count < 2 or cluster_count >= len(sampled_rows):
        return 0.0

    if _USE_TORCH_SILHOUETTE:
        # [E] Tier-3 opt-in: GPU silhouette via _torch_silhouette_score.
        return _torch_silhouette_score(embeds, cluster_labels, device)
    return float(silhouette_score(embeds, cluster_labels, sample_size=None))


# Fragment centroids
def fragment_embeddings(
    model: _EmbedNet,
    fragments: list[Fragment],
    crops_path: str,
    infer_batch_size: int = _INFER_BATCH,
    crops_t: torch.Tensor | None = None,
) -> dict[int, np.ndarray]:
    """Return frag_id -> 8-dim centroid embedding."""
    device = next(model.parameters()).device
    with h5py.File(crops_path, 'r') as hf:
        crops_np = hf['crops'][:]

    centroids: dict[int, np.ndarray] = {
        frag.frag_id: np.zeros(_EMBED_DIM, dtype=np.float32)
        for frag in fragments
    }
    all_rows: list[int] = []
    row_frag_ids: list[int] = []
    for frag in fragments:
        rows = list(frag.crop_indices)
        all_rows.extend(rows)
        row_frag_ids.extend([frag.frag_id] * len(rows))

    if not all_rows:
        return centroids

    if crops_t is None:
        crops_t = _cache_crops_on_device(crops_np, device)

    embeds = _embed_crop_rows(
        model,
        crops_np,
        all_rows,
        device,
        batch_size=infer_batch_size,
        progress_desc='Embedding fragment centroids',
        crops_t=crops_t,
    )

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for frag_id, emb in zip(row_frag_ids, embeds):
        if frag_id not in sums:
            sums[frag_id] = np.zeros(_EMBED_DIM, dtype=np.float32)
            counts[frag_id] = 0
        sums[frag_id] += emb
        counts[frag_id] += 1

    for frag_id, total in sums.items():
        centroids[frag_id] = total / max(1, counts[frag_id])
    return centroids


# Public gates
@dataclass(frozen=True)
class EpisodeDecision:
    mapping: dict[int, int]
    is_swap: bool
    abstained: bool
    reason: str
    best_cost: float
    identity_cost: float
    second_cost: float
    max_matched_dist: float


def embedding_available(best_ss: float) -> bool:
    try:
        score = float(best_ss)
    except (TypeError, ValueError):
        return False
    # The 0.91 target controls early stopping only.  If training exhausts the
    # no-improvement budget below the target, still use the saved best model for
    # the downstream assignment gates.
    return math.isfinite(score) and score > -1.0


def assign_episode(
    centroids: dict[int, np.ndarray],
    pre_frags: dict[int, int],
    post_frags: dict[int, int],
) -> EpisodeDecision:
    """Resolve a contact episode with Hungarian assignment and abstention gates."""
    roster = sorted(int(r) for r in pre_frags.keys())
    identity = {r: r for r in roster}
    n = len(roster)

    if n < 2 or set(int(r) for r in post_frags.keys()) != set(roster) or n > _ASSIGN_MAX_ROSTER:
        return EpisodeDecision(identity, False, True, 'roster_invalid', 0.0, 0.0, float('inf'), 0.0)

    frag_ids = [int(pre_frags[r]) for r in roster] + [int(post_frags[r]) for r in roster]
    if any(fid not in centroids for fid in frag_ids):
        return EpisodeDecision(identity, False, True, 'centroid_missing', 0.0, 0.0, float('inf'), 0.0)

    pre_vecs = [np.asarray(centroids[int(pre_frags[r])], dtype=np.float64) for r in roster]
    post_vecs = [np.asarray(centroids[int(post_frags[r])], dtype=np.float64) for r in roster]

    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            cost[i, j] = float(np.linalg.norm(pre_vecs[i] - post_vecs[j]))

    row_ind, col_ind = linear_sum_assignment(cost)
    ordered_cols = col_ind[np.argsort(row_ind)]
    best_perm = tuple(int(c) for c in ordered_cols)
    best_cost = float(cost[row_ind, col_ind].sum())
    identity_cost = float(np.trace(cost))
    max_matched = float(max(cost[i, best_perm[i]] for i in range(n)))

    second_cost = float('inf')
    for perm in itertools.permutations(range(n)):
        if perm == best_perm:
            continue
        c = float(sum(cost[i, int(perm[i])] for i in range(n)))
        if c < second_cost:
            second_cost = c

    mapping = {int(roster[i]): int(roster[best_perm[i]]) for i in range(n)}
    is_swap = any(int(k) != int(v) for k, v in mapping.items())

    if is_swap:
        if best_cost > _ASSIGN_COST_RATIO_MAX * min(identity_cost, second_cost):
            return EpisodeDecision(
                identity, False, True, 'gate_cost_ratio',
                best_cost, identity_cost, second_cost, max_matched,
            )
    elif max_matched >= _IDENTITY_ABS_MAX_DIST:
        return EpisodeDecision(
            mapping, is_swap, True, 'gate_abs_dist',
            best_cost, identity_cost, second_cost, max_matched,
        )

    return EpisodeDecision(
        mapping,
        is_swap,
        False,
        'swap' if is_swap else 'identity',
        best_cost,
        identity_cost,
        second_cost,
        max_matched,
    )


@dataclass(frozen=True)
class RescueDecision:
    attempted: bool
    action: str                        # 'relabel' or 'abstain'
    failure_stage: str | None          # 'prototype' / 'pre_check' / 'post_check', None on relabel
    mapping: dict[int, int] | None     # decided mapping; only set when action == 'relabel'

    pre_cost_matrix: list[list[float]] | None
    pre_best_cost: float | None
    pre_identity_cost: float | None
    pre_max_matched_dist: float | None
    pre_reason: str | None

    post_cost_matrix: list[list[float]] | None
    post_best_cost: float | None
    post_identity_cost: float | None
    post_second_cost: float | None
    post_max_matched_dist: float | None
    post_reason: str | None
    post_assignment: dict[int, int] | None  # raw Hungarian best assignment, even if abstained

    prototype_fragment_counts: dict[int, int] | None
    prototype_fragment_ids: dict[int, list[int]] | None
    pre_fragments: dict[int, dict] | None   # tid -> {'frag_id': int, 'frame_range': [int, int]}
    post_fragments: dict[int, dict] | None


def _fragment_global_prototype(
    fragment_index: dict,
    centroids: dict[int, np.ndarray],
    tid: int,
    episode_start: int,
) -> tuple[np.ndarray | None, list[int]]:
    """Component-wise median centroid across tid's fragments strictly before episode_start.

    Deliberately past-only, not "outside the episode window": a fragment that
    starts after this episode may already carry a wrong identity if this
    episode is itself an as-yet-uncorrected ID switch (the failure mode this
    rescue exists to catch). Including such fragments would let the prototype
    -- the rescue's supposed ground truth -- self-contaminate with the very
    error under test. fragment_index is keyed by obj_id -> (end_frames,
    by_end_fragments, start_frames, by_start_fragments), matching
    correction._build_fragment_index. Returns (None, []) when no qualifying
    fragment (with a trained centroid) exists, else (median, frag_ids used).
    """
    entry = fragment_index.get(int(tid))
    if entry is None:
        return None, []
    _, by_end, _, _ = entry
    used_ids: list[int] = []
    vecs: list[np.ndarray] = []
    for frag in by_end:
        if not frag.frames or max(frag.frames) >= episode_start:
            continue
        fid = int(frag.frag_id)
        if fid not in centroids:
            continue
        used_ids.append(fid)
        vecs.append(np.asarray(centroids[fid], dtype=np.float64))
    if not vecs:
        return None, []
    return np.median(np.stack(vecs, axis=0), axis=0), used_ids


def _fragment_frame_range(fragment_index: dict, tid: int, frag_id: int) -> list[int] | None:
    """[min_frame, max_frame] of a specific known fragment of tid, for diagnostics."""
    entry = fragment_index.get(int(tid))
    if entry is None:
        return None
    _, by_end, _, _ = entry
    for frag in by_end:
        if int(frag.frag_id) == int(frag_id) and frag.frames:
            return [int(min(frag.frames)), int(max(frag.frames))]
    return None


def _rescue_cost_matrix_and_raw_mapping(
    centroids: dict[int, np.ndarray],
    roster: list[int],
    ref_frags: dict[int, int],
    cand_frags: dict[int, int],
) -> tuple[list[list[float]], dict[int, int]]:
    """Cost matrix and raw (pre-gate) Hungarian mapping, for rescue diagnostics only.

    Duplicates assign_episode's cost-matrix/Hungarian computation rather than
    reusing it, because assign_episode intentionally resets its returned
    mapping to identity when a swap is gate-rejected -- exactly the
    information the diagnostics below need to see (e.g. what the post side
    would have proposed even when pre_check aborts the rescue first).
    assign_episode itself is left untouched, and this never feeds into any
    relabel decision -- only assign_episode's own gated result does.
    """
    ref_vecs = [np.asarray(centroids[int(ref_frags[r])], dtype=np.float64) for r in roster]
    cand_vecs = [np.asarray(centroids[int(cand_frags[r])], dtype=np.float64) for r in roster]
    n = len(roster)
    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            cost[i, j] = float(np.linalg.norm(ref_vecs[i] - cand_vecs[j]))
    row_ind, col_ind = linear_sum_assignment(cost)
    ordered_cols = col_ind[np.argsort(row_ind)]
    best_perm = tuple(int(c) for c in ordered_cols)
    mapping = {int(roster[i]): int(roster[best_perm[i]]) for i in range(n)}
    return cost.tolist(), mapping


def attempt_embedding_rescue(
    fragment_index: dict,
    centroids: dict[int, np.ndarray],
    roster: list[int],
    pre_frags: dict[int, int],
    post_frags: dict[int, int],
    start: int,
) -> RescueDecision:
    """Second-stage rescue for an episode assign_episode abstained on gate_abs_dist
    or gate_cost_ratio.

    A single local pre/post fragment centroid can be noisy enough that the local
    Hungarian assignment itself is unreliable (not just the confidence gates on
    top of it).  This builds a per-ID "global prototype" -- the component-wise
    median centroid across that ID's fragments strictly before this episode
    (see _fragment_global_prototype) -- and re-runs assign_episode's own
    Hungarian-and-gate logic unmodified, anchored to that prototype instead of
    a single fragment:

    - pre_check matches the (trusted, pre-episode) pre_frags against the
      prototypes.  It should come back as a confident identity match; if it
      doesn't, the prototypes aren't reliable reference points for this roster
      (or an earlier, out-of-scope labeling problem exists), so the rescue
      abstains rather than trust the post-side test below.
    - post_check matches post_frags -- the fragments whose identity is actually
      in question -- against the same prototypes.  A confident (non-abstained)
      swap here is the rescue's relabel signal.

    post-side diagnostics (cost matrix, gated numbers, raw Hungarian
    assignment) are always computed once prototypes exist, even when
    pre_check fails, purely so the returned RescueDecision can show what the
    post side would have concluded -- this has no effect on the actual
    action, which still abstains whenever pre_check fails, unchanged.

    No new thresholds are introduced: both checks reuse assign_episode's
    existing _IDENTITY_ABS_MAX_DIST / _ASSIGN_COST_RATIO_MAX gates as-is.
    """
    prototypes: dict[int, np.ndarray] = {}
    prototype_fragment_ids: dict[int, list[int]] = {}
    for tid in roster:
        proto, used_ids = _fragment_global_prototype(fragment_index, centroids, tid, start)
        if proto is None:
            return RescueDecision(
                attempted=True, action='abstain', failure_stage='prototype', mapping=None,
                pre_cost_matrix=None, pre_best_cost=None, pre_identity_cost=None,
                pre_max_matched_dist=None, pre_reason=None,
                post_cost_matrix=None, post_best_cost=None, post_identity_cost=None,
                post_second_cost=None, post_max_matched_dist=None, post_reason=None,
                post_assignment=None,
                prototype_fragment_counts=None, prototype_fragment_ids=None,
                pre_fragments=None, post_fragments=None,
            )
        prototypes[tid] = proto
        prototype_fragment_ids[tid] = used_ids
    prototype_fragment_counts = {tid: len(ids) for tid, ids in prototype_fragment_ids.items()}

    # Synthetic frag ids for the prototypes, disjoint from real (always >= 0)
    # frag ids, so assign_episode's centroid lookups resolve unambiguously.
    proto_frag_ids = {int(tid): -(int(tid) + 1) for tid in roster}
    rescue_centroids = dict(centroids)
    for tid, proto_id in proto_frag_ids.items():
        rescue_centroids[proto_id] = prototypes[tid]

    pre_fragments = {
        tid: {'frag_id': int(pre_frags[tid]), 'frame_range': _fragment_frame_range(fragment_index, tid, pre_frags[tid])}
        for tid in roster
    }
    post_fragments = {
        tid: {'frag_id': int(post_frags[tid]), 'frame_range': _fragment_frame_range(fragment_index, tid, post_frags[tid])}
        for tid in roster
    }

    pre_cost_matrix, _pre_raw_mapping = _rescue_cost_matrix_and_raw_mapping(
        rescue_centroids, roster, proto_frag_ids, pre_frags,
    )
    pre_check = assign_episode(rescue_centroids, proto_frag_ids, pre_frags)

    post_cost_matrix, post_raw_mapping = _rescue_cost_matrix_and_raw_mapping(
        rescue_centroids, roster, proto_frag_ids, post_frags,
    )
    post_check = assign_episode(rescue_centroids, proto_frag_ids, post_frags)

    diagnostics = dict(
        pre_cost_matrix=pre_cost_matrix,
        pre_best_cost=pre_check.best_cost,
        pre_identity_cost=pre_check.identity_cost,
        pre_max_matched_dist=pre_check.max_matched_dist,
        pre_reason=pre_check.reason,
        post_cost_matrix=post_cost_matrix,
        post_best_cost=post_check.best_cost,
        post_identity_cost=post_check.identity_cost,
        post_second_cost=post_check.second_cost,
        post_max_matched_dist=post_check.max_matched_dist,
        post_reason=post_check.reason,
        post_assignment=post_raw_mapping,
        prototype_fragment_counts=prototype_fragment_counts,
        prototype_fragment_ids=prototype_fragment_ids,
        pre_fragments=pre_fragments,
        post_fragments=post_fragments,
    )

    if pre_check.abstained or pre_check.is_swap:
        return RescueDecision(
            attempted=True, action='abstain', failure_stage='pre_check', mapping=None, **diagnostics,
        )

    if post_check.abstained or not post_check.is_swap:
        return RescueDecision(
            attempted=True, action='abstain', failure_stage='post_check', mapping=None, **diagnostics,
        )

    return RescueDecision(
        attempted=True, action='relabel', failure_stage=None, mapping=post_check.mapping, **diagnostics,
    )


@dataclass(frozen=True)
class OneSidedRescueDecision:
    attempted: bool
    action: str                        # 'relabel' or 'abstain'
    anchor_id: int | None              # the track whose transition confirmed the swap
    mapping: dict[int, int] | None     # {roster[0]: roster[1], roster[1]: roster[0]} when relabel
    pre_match: dict[int, int | None] | None   # tid -> prototype tid its pre fragment mutually matched
    post_match: dict[int, int | None] | None  # tid -> prototype tid its post fragment mutually matched
    pre_adjacent: dict[int, bool] | None   # tid -> pre fragment ends exactly at episode_start - 1
    post_adjacent: dict[int, bool] | None  # tid -> post fragment starts exactly at episode_end + 1


def _prototype_cost_row(centroids: dict[int, np.ndarray], frag_id: int, prototypes: list[np.ndarray]) -> np.ndarray:
    frag_vec = np.asarray(centroids[int(frag_id)], dtype=np.float64)
    return np.asarray([float(np.linalg.norm(frag_vec - proto)) for proto in prototypes], dtype=np.float64)


def _mutual_nearest_prototype(cost: np.ndarray, i: int) -> int | None:
    """Index j (0 or 1) such that j is row i's strict nearest column, i is
    column j's strict nearest row, and cost[i, j] < _IDENTITY_ABS_MAX_DIST.

    A strict local (row+column) nearest-neighbor agreement, not a joint
    Hungarian optimum: this is deliberately a per-pair check so a single
    track's confident, mutually-agreed match can stand on its own even when
    the other roster member's fragment is too unstable to form any
    confident pairing at all.
    """
    other_i = 1 - i
    j = int(np.argmin(cost[i]))
    other_j = 1 - j
    if cost[i, j] >= cost[i, other_j]:
        return None  # not a strict row nearest (tie)
    if cost[i, j] >= _IDENTITY_ABS_MAX_DIST:
        return None
    if cost[i, j] >= cost[other_i, j]:
        return None  # the other track's fragment is at least as close to prototype j
    return j


def attempt_one_sided_rescue(
    fragment_index: dict,
    centroids: dict[int, np.ndarray],
    roster: list[int],
    pre_frags: dict[int, int],
    post_frags: dict[int, int],
    start: int,
    end: int,
) -> OneSidedRescueDecision:
    """Third-stage fallback, roster == 2 only, for episodes attempt_embedding_rescue
    (the full-roster rescue) still could not resolve.

    A full-roster rescue can abstain because ONE track's local fragment is too
    noisy to pass the joint gates, even when the OTHER track's own
    prototype-relative appearance tells a completely unambiguous story on its
    own: it was near its own historical prototype before the episode, and near
    the *other* roster member's historical prototype after -- a one-track,
    time-direction identity transition (prototype A -> pre A -> interaction ->
    post B -> prototype B), not "trust whichever local fragment happens to be
    close". This checks that per track, independently, using only its own
    pre/post fragment against both prototypes:

    - pre_match: its pre fragment's nearest prototype is a strict, mutual
      (row+column) match, under _IDENTITY_ABS_MAX_DIST -- the existing gate
      threshold, no new number introduced.
    - post_match: same check for its post fragment.

    A track only qualifies as the anchor for a confirmed swap when, in
    addition, its pre fragment ends at exactly episode_start - 1 and its
    post fragment starts at exactly episode_end + 1 -- i.e. it directly
    brackets this specific episode with no gap. Without this, pre_frags/
    post_frags can resolve to a fragment far away in time (see
    _resolve_pre_post_fragments's nearest-before/after pivot search), whose
    prototype-relative similarity says nothing about identity across *this*
    episode specifically. This is a pass/fail adjacency check on existing
    frame numbers, not a new distance or time-window threshold.

    A track "confirms" a swap when pre_match is itself, post_match is the
    other roster member, and both adjacency checks hold. With roster == 2,
    only one non-identity mapping is possible, so a single confirming track
    is enough (the bijective constraint forces the other track's relabel).
    If the other track's own evidence instead confirms it did *not*
    transition (pre_match and post_match both itself), that is a genuine
    contradiction and this abstains rather than override it.
    """
    if len(roster) != 2:
        return OneSidedRescueDecision(False, 'abstain', None, None, None, None, None, None)

    prototypes: dict[int, np.ndarray] = {}
    for tid in roster:
        proto, _used_ids = _fragment_global_prototype(fragment_index, centroids, tid, start)
        if proto is None:
            return OneSidedRescueDecision(False, 'abstain', None, None, None, None, None, None)
        prototypes[tid] = proto
    proto_vecs = [prototypes[tid] for tid in roster]

    cost_pre = np.stack([
        _prototype_cost_row(centroids, pre_frags[roster[0]], proto_vecs),
        _prototype_cost_row(centroids, pre_frags[roster[1]], proto_vecs),
    ])
    cost_post = np.stack([
        _prototype_cost_row(centroids, post_frags[roster[0]], proto_vecs),
        _prototype_cost_row(centroids, post_frags[roster[1]], proto_vecs),
    ])

    pre_match: dict[int, int | None] = {}
    post_match: dict[int, int | None] = {}
    pre_adjacent: dict[int, bool] = {}
    post_adjacent: dict[int, bool] = {}
    for idx, tid in enumerate(roster):
        j_pre = _mutual_nearest_prototype(cost_pre, idx)
        pre_match[tid] = int(roster[j_pre]) if j_pre is not None else None
        j_post = _mutual_nearest_prototype(cost_post, idx)
        post_match[tid] = int(roster[j_post]) if j_post is not None else None

        pre_range = _fragment_frame_range(fragment_index, tid, pre_frags[tid])
        pre_adjacent[tid] = pre_range is not None and int(pre_range[1]) == int(start) - 1
        post_range = _fragment_frame_range(fragment_index, tid, post_frags[tid])
        post_adjacent[tid] = post_range is not None and int(post_range[0]) == int(end) + 1

    def _claim(tid: int, other: int) -> str:
        if pre_match.get(tid) != tid:
            return 'inconclusive'
        if post_match.get(tid) == other:
            if not (pre_adjacent.get(tid) and post_adjacent.get(tid)):
                return 'inconclusive'
            return 'swap'
        if post_match.get(tid) == tid:
            return 'no_swap'
        return 'inconclusive'

    a, b = int(roster[0]), int(roster[1])
    claims = {a: _claim(a, b), b: _claim(b, a)}
    swap_claims = [tid for tid, claim in claims.items() if claim == 'swap']
    no_swap_claims = [tid for tid, claim in claims.items() if claim == 'no_swap']

    if not swap_claims or no_swap_claims:
        # No track confirms a transition, or one track's own evidence
        # contradicts a transition the other track confirms.
        return OneSidedRescueDecision(
            True, 'abstain', None, None, pre_match, post_match, pre_adjacent, post_adjacent,
        )

    anchor_id = swap_claims[0]
    return OneSidedRescueDecision(
        True, 'relabel', anchor_id, {a: b, b: a}, pre_match, post_match, pre_adjacent, post_adjacent,
    )


# Caching orchestration
def load_or_train_embedding(
    fragments: list[Fragment],
    corrected: dict,
    video_path: str,
    out_dir: str,
    num_objects: int,
    img_size: int,
    background_path: str,
    preview_count: int = 100,
    num_workers: int = 1,
    seed: int = 0,
    device_config='auto',
    session_path: str = '',
    *,
    silhouette_fn=None,
) -> tuple[_EmbedNet | None, dict[int, np.ndarray], float]:
    """
    Orchestrate crop extraction + training with HDF5/model caching.
    Returns (model, centroids, best_ss). model is None if no fragments.
    """
    if silhouette_fn is None:
        silhouette_fn = _compute_silhouette
    device = _embedding_device_from_config(device_config, purpose='embedding')
    _configure_torch_parallelism(num_workers, device)
    seed = normalize_seed(seed)
    h5_path = os.path.join(out_dir, _CROP_CACHE_NAME)
    model_path = os.path.join(out_dir, _MODEL_CACHE_NAME)
    seg_config = _resolve_seg_config(session_path, video_path)
    if seg_config is None:
        raise EmbeddingSegmentationRequired(
            f'No segmentation configuration found for this session ({session_path!r}); '
            'embedding crops require segmentation to have been run first.'
        )
    seg_signature = _seg_config_signature(seg_config)
    cached_metadata = _read_crop_cache_metadata(h5_path, background_path, seg_config_signature=seg_signature)
    if not fragments and cached_metadata is None:
        return None, {}, 0.0
    frame_to_crops: dict[int, list[tuple[Fragment, int]]] | None = None

    def _h5_cache_valid(h5_path: str, expected_shape: tuple[int, int, int]) -> bool:
        with h5py.File(h5_path, 'r') as hf:
            if tuple(hf['crops'].shape) != expected_shape:
                return False
            cached_bg = str(hf.attrs.get('background_path', ''))
            current_bg = _background_cache_key(background_path)
            if cached_bg != current_bg:
                return False
            if str(hf.attrs.get('seg_config_signature', '')) != seg_signature:
                return False
            return int(hf.attrs.get('crop_metadata_version', 0)) == int(_CROP_METADATA_VERSION)

    if cached_metadata is not None and int(cached_metadata[1]) == int(img_size):
        fragments = cached_metadata[0]
        crop_cache_valid = True
    else:
        frame_to_crops = _prepare_frame_to_crops(fragments)
        expected_crops = sum(len(v) for v in frame_to_crops.values())
        expected_shape = (expected_crops, img_size, img_size)
        crop_cache_valid = os.path.exists(h5_path) and _h5_cache_valid(h5_path, expected_shape)

    preview_paths: list[str] = []
    if not crop_cache_valid:
        if frame_to_crops is None:
            frame_to_crops = _prepare_frame_to_crops(fragments)
        preview_paths = write_initial_crop_previews(
            corrected,
            video_path,
            out_dir,
            img_size,
            background_path,
            fragments,
            preview_count=preview_count,
            seg_config=seg_config,
        )
        if preview_paths:
            print(
                f'  Wrote {len(preview_paths)} initial embedding crop previews: '
                f'{os.path.dirname(preview_paths[0])}',
                flush=True,
            )
        h5_path = extract_and_cache_crops(
            corrected,
            video_path,
            out_dir,
            img_size,
            background_path,
            frame_to_crops,
            fragments,
            num_workers=num_workers,
            seg_config=seg_config,
        )
        # The model is tied to the exact crop generation mode/background/size.
        # If crops were regenerated, do not reuse a stale checkpoint.
        if os.path.exists(model_path):
            os.remove(model_path)

    if not preview_paths:
        preview_paths = write_crop_previews(
            h5_path,
            fragments,
            out_dir,
            preview_count=preview_count,
        )
        if preview_paths:
            print(
                f'  Wrote {len(preview_paths)} embedding crop previews: {os.path.dirname(preview_paths[0])}',
                flush=True,
            )

    best_ss = 0.0
    centroid_crops_t = None
    metrics_state = _training_metrics_state(out_dir)

    if _embedding_model_cache_complete(out_dir, metrics_state):
        model = _EmbedNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        with h5py.File(h5_path, 'r') as hf:
            crops_np = hf['crops'][:]
        crops_t = _cache_crops_on_device(crops_np, device)
        best_ss = metrics_state.best_ss
        if best_ss is None:
            ss_sample_total = min(len(crops_np), 1000 * int(num_objects))
            best_ss = silhouette_fn(model, crops_np, fragments, ss_sample_total, device, crops_t=crops_t, seed=seed)
        centroid_crops_t = crops_t
    else:
        if os.path.exists(model_path) and metrics_state.exists and not metrics_state.done:
            print(
                '  Existing embedding model cache has unfinished metrics; retraining embedding.',
                flush=True,
            )
        model, best_ss = train_embedding(
            fragments,
            h5_path,
            out_dir,
            num_objects,
            num_workers=num_workers,
            seed=seed,
            device_config=device,
            silhouette_fn=silhouette_fn,
        )

    centroids = fragment_embeddings(model, fragments, h5_path, crops_t=centroid_crops_t)
    return model, centroids, best_ss


def horizontal_angle(obb_t):
    """OpenCV rotation in degrees, modulo 180; zero means a horizontal axis."""
    raw = np.asarray(obb_t, dtype=float)
    pts = raw if raw.shape == (4, 2) else np.stack((raw[:4], raw[4:]), axis=1)
    # Serialized OBBs contain all four x values followed by all four y values.
    if not np.isfinite(pts).all():
        raise ValueError('Embedding requires a finite OBB.')
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    if min(lengths) <= 0:
        raise ValueError('Embedding requires a non-degenerate OBB.')
    edge = edges[int(np.argmax(lengths))]
    return float(np.degrees(np.arctan2(edge[1], edge[0])) % 180.0)

def axial_forward(net, x):
    """Average both polarities in one batch in training and inference."""
    import torch
    output = net(torch.cat((x, torch.flip(x, dims=(-2, -1))), dim=0))
    direct, reverse = output.chunk(2, dim=0)
    return (direct + reverse) * 0.5
