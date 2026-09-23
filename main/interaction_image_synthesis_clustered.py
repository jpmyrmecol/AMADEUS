# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Synthesize clustered interaction scenes from isolated animal images."""

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from animal_noise import apply_animal_noise, load_noise_background
from batch_utils import tqdm

from interaction_image_synthesis import (
    ImageMaskCache,
    RNGPool,
    append_annotation,
    base_object_to_group_obj,
    build_feather_alpha,
    build_even_frame_jobs,
    build_additional_frame_jobs,
    preview_additional_frame_jobs,
    preview_frame_set_ids_by_interval,
    build_frame_items,
    choose_under,
    donor_image_and_mask,
    donor_result_to_group_obj,
    get_mask_expansion_ratio,
    draw_preview_from_label_lines,
    ensure_clockwise,
    ensure_dirs,
    full_mask_from_local,
    load_config,
    paste_mask_into_full,
    read_manifest,
    reset_output_dir,
    resolve_num_workers,
    rotate_crop_with_mask,
    rotate_crop_with_mask_and_obb_scaling,
    sample_paste_width_scales,
    paste_width_scale_range_from_config,
    valid_obb_aspect_ratio,
    head_angle_deg_from_axis,
    obb_center,
    mask_to_polygons,
    intersection_slices,
    count_overlap_rect_mask,
    build_external_aabb_array,
    external_aabb_intersecting_indices,
    transform_heading_info,
    occlusion_state_from_mode,
    output_stem_for_frame_set,
    apply_brightness_contrast,
)
from random_utils import make_python_rng, normalize_seed

DIR_NAMES = [
    "upper",
    "upper_right",
    "right",
    "lower_right",
    "lower",
    "lower_left",
    "left",
    "upper_left",
]


def donor_head_angle_from_direction(donor: dict) -> Optional[float]:
    direction_vec = donor.get("direction_vec")
    if direction_vec is None:
        return None
    dx = float(direction_vec[0])
    dy = float(direction_vec[1])
    if math.hypot(dx, dy) < 1e-8:
        return None
    return head_angle_deg_from_axis((dx, dy))


def make_group_patch(out_img: np.ndarray, group_objects: List[dict], H: int, W: int) -> Optional[dict]:
    full_union = np.zeros((H, W), np.uint8)
    for obj in group_objects:
        paste_mask_into_full(full_union, obj["mask"], obj["x"], obj["y"])
    ys, xs = np.where(full_union > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    patch = out_img[y1:y2, x1:x2].copy()
    union_local = full_union[y1:y2, x1:x2].copy()
    local_objects = []
    for obj in group_objects:
        lx = int(obj["x"] - x1)
        ly = int(obj["y"] - y1)
        local_objects.append({**obj, "x": lx, "y": ly})
    return {"patch": patch, "union_mask": union_local, "objects": local_objects}


def normalize_vec2(vec: Tuple[float, float]) -> np.ndarray:
    x, y = float(vec[0]), float(vec[1])
    n = math.hypot(x, y)
    if n < 1e-8:
        raise ValueError("Cannot normalize a zero-length vector.")
    return np.asarray([x / n, y / n], dtype=np.float32)


def target_long_axis_from_item(target: dict) -> np.ndarray:
    direction_vec = target.get("direction_vec")
    if direction_vec is None:
        raise ValueError("Target has no direction_vec.")
    return normalize_vec2((float(direction_vec[0]), float(direction_vec[1])))


def target_obb_frame(target: dict) -> Tuple[np.ndarray, np.ndarray, float]:
    long_axis = target_long_axis_from_item(target)
    short_axis = np.asarray([-float(long_axis[1]), float(long_axis[0])], dtype=np.float32)
    target_head_angle = head_angle_deg_from_axis((float(long_axis[0]), float(long_axis[1])))
    return long_axis, short_axis, target_head_angle


def full_mask_overlap(full_mask: np.ndarray, mask: np.ndarray, x: int, y: int) -> int:
    H, W = full_mask.shape[:2]
    h, w = mask.shape[:2]
    sl = intersection_slices(int(x), int(y), int(w), int(h), 0, 0, int(W), int(H))
    if sl is None:
        return 0
    mask_slice, full_slice = sl
    return int(cv2.countNonZero(cv2.bitwise_and(mask[mask_slice], full_mask[full_slice])))


def cluster_obb_overlap_tolerances(donor_obb_area: int) -> Tuple[int, int]:
    area = max(1, int(donor_obb_area))
    contact_tol = max(int(CLUSTER_OBB_CONTACT_MIN_PIXELS), int(math.ceil(float(area) * CLUSTER_OBB_CONTACT_MAX_RATIO)))
    non_contact_tol = max(int(CLUSTER_OBB_NON_CONTACT_MIN_PIXELS), int(math.ceil(float(area) * CLUSTER_OBB_NON_CONTACT_MAX_RATIO)))
    return int(contact_tol), int(non_contact_tol)


def reject_cluster_fit_obb_overlap(*,
                                   group_obb_mask_full_u8: np.ndarray,
                                   donor_obb_mask_local: np.ndarray,
                                   donor_obb_area: int,
                                   xo: int,
                                   yo: int,
                                   allowed_contact_overlap_pixels: int = 0) -> Tuple[bool, int, float, int, float]:
    """Return whether a fit-scaled donor OBB collides with the current cluster.

    The donor is allowed only a thin rasterized contact strip against the edge(s)
    it is explicitly being fitted to.  Any overlap with other target/pasted OBBs
    is treated as a hard collision.
    """
    area = max(1, int(donor_obb_area))
    total_overlap = full_mask_overlap(group_obb_mask_full_u8, donor_obb_mask_local, int(xo), int(yo))
    non_contact_overlap = max(0, int(total_overlap) - int(allowed_contact_overlap_pixels))
    contact_tol, non_contact_tol = cluster_obb_overlap_tolerances(area)
    if int(allowed_contact_overlap_pixels) <= 0:
        reject = int(total_overlap) > int(contact_tol)
    else:
        reject = int(non_contact_overlap) > int(non_contact_tol)
    return (
        bool(reject),
        int(total_overlap),
        float(total_overlap) / float(area),
        int(non_contact_overlap),
        float(non_contact_overlap) / float(area),
    )


def polygon_mask_from_points(points: np.ndarray, H: int, W: int) -> np.ndarray:
    mask = np.zeros((int(H), int(W)), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts_i], 255, lineType=cv2.LINE_8)
    return mask


def obb_points_from_mask_local(mask_u8_255: np.ndarray) -> Optional[np.ndarray]:
    cnts = mask_to_polygons(mask_u8_255)
    if not cnts:
        return None
    pts_all = np.concatenate([cnt.astype(np.float32) for cnt in cnts], axis=0).reshape(-1, 2)
    rect = cv2.minAreaRect(pts_all)
    return ensure_clockwise(cv2.boxPoints(rect).astype(np.float32))


def scale_obb_points_about_center(obb_pts: np.ndarray, scale: float) -> np.ndarray:
    pts = ensure_clockwise(np.asarray(obb_pts, dtype=np.float32).reshape(4, 2))
    s = float(np.clip(float(scale), 0.05, 1.0))
    c = obb_center(pts).astype(np.float32)
    return (c.reshape(1, 2) + (pts - c.reshape(1, 2)) * s).astype(np.float32)




def scale_obb_points_long_short(obb_pts: np.ndarray, long_scale: float, short_scale: float) -> np.ndarray:
    pts = ensure_clockwise(np.asarray(obb_pts, dtype=np.float32).reshape(4, 2))
    ls = float(np.clip(float(long_scale), 0.05, 1.0))
    ss = float(np.clip(float(short_scale), 0.05, 1.0))
    c = obb_center(pts).astype(np.float32)

    edge_lengths = []
    edge_vecs = []
    for i in range(4):
        v = pts[(i + 1) % 4] - pts[i]
        edge_vecs.append(v.astype(np.float32))
        edge_lengths.append(float(np.linalg.norm(v)))
    if not edge_lengths or max(edge_lengths) < 1e-8:
        return scale_obb_points_about_center(pts, min(ls, ss))

    long_vec = edge_vecs[int(np.argmax(edge_lengths))]
    long_axis = normalize_vec2((float(long_vec[0]), float(long_vec[1])))
    short_axis = np.asarray([-float(long_axis[1]), float(long_axis[0])], dtype=np.float32)

    out = []
    for pt in pts:
        rel = pt.astype(np.float32) - c
        long_coord = float(np.dot(rel, long_axis)) * ls
        short_coord = float(np.dot(rel, short_axis)) * ss
        out.append(c + long_axis * long_coord + short_axis * short_coord)
    return ensure_clockwise(np.asarray(out, dtype=np.float32).reshape(4, 2))

def orientation_angle(anchor_angle_deg: float, orientation: str) -> float:
    o = str(orientation).strip().lower()
    if o == "parallel":
        return float(anchor_angle_deg) % 360.0
    if o == "anti_parallel":
        return (float(anchor_angle_deg) + 180.0) % 360.0
    if o == "perpendicular":
        return (float(anchor_angle_deg) + 90.0) % 360.0
    if o == "anti_perpendicular":
        return (float(anchor_angle_deg) + 270.0) % 360.0
    raise ValueError(f"Unknown cluster orientation: {orientation}")


# OBB collision gates are evaluated on the fit-scaled OBB masks, not on the
# instance masks.  A small raster-contact tolerance is necessary because two
# polygons sharing an edge can produce a few intersecting pixels after integer
# filling.  Anything beyond this is treated as true OBB overlap and rejected.
CLUSTER_OBB_CONTACT_MAX_RATIO = 0.035
CLUSTER_OBB_NON_CONTACT_MAX_RATIO = 0.006
CLUSTER_OBB_CONTACT_MIN_PIXELS = 8
CLUSTER_OBB_NON_CONTACT_MIN_PIXELS = 2


_WORKER_CFG: Optional[dict] = None
_WORKER_IMG_DIR: Optional[str] = None
_WORKER_DIRS: Optional[Dict[str, str]] = None
_WORKER_ALL_OBJECTS: Optional[List[dict]] = None
_WORKER_OBJECTS_BY_FRAME: Optional[Dict[int, List[dict]]] = None
_WORKER_OBJECTS_BY_POOL: Optional[Dict[str, List[dict]]] = None
_WORKER_CACHE: Optional[ImageMaskCache] = None
_WORKER_RNG_POOL: Optional[RNGPool] = None
_WORKER_PREVIEW_IDS: Optional[set] = None


_CLUSTER_NOISE_BACKGROUND = None


def _init_worker(config_path: str, preview_jobs: List[Tuple[int, int]]) -> None:
    global _WORKER_CFG, _WORKER_IMG_DIR, _WORKER_DIRS, _WORKER_ALL_OBJECTS
    global _WORKER_OBJECTS_BY_FRAME, _WORKER_OBJECTS_BY_POOL, _WORKER_CACHE
    global _WORKER_RNG_POOL, _WORKER_PREVIEW_IDS

    global _CLUSTER_NOISE_BACKGROUND

    cv2.setNumThreads(1)
    _WORKER_CFG = load_config(config_path)
    _CLUSTER_NOISE_BACKGROUND = load_noise_background(_WORKER_CFG)
    session_path = str(_WORKER_CFG["SESSION_PATH"])
    without_crossing_dir = os.path.join(session_path, "single_animal_images")
    _WORKER_IMG_DIR = os.path.join(without_crossing_dir, "images")
    manifest_path = os.path.join(without_crossing_dir, "object_pool", "manifest.csv")
    _WORKER_ALL_OBJECTS, _WORKER_OBJECTS_BY_FRAME, _WORKER_OBJECTS_BY_POOL = read_manifest(manifest_path)
    out_dir = os.path.join(session_path, "paste_blobs_clustered")
    _WORKER_DIRS = {k: os.path.join(out_dir, k) for k in ("images", "labels", "masks", "preview")}
    _WORKER_CACHE = ImageMaskCache(without_crossing_dir)
    _WORKER_RNG_POOL = RNGPool(normalize_seed(_WORKER_CFG.get("RANDOM_SEED", 0)))
    _WORKER_PREVIEW_IDS = {(int(fid), int(rep)) for fid, rep in preview_jobs}


def _process_frame(job: Tuple[int, int]) -> Tuple[int, int, int]:
    cfg = _WORKER_CFG
    img_dir = _WORKER_IMG_DIR
    dirs = _WORKER_DIRS
    all_objects = _WORKER_ALL_OBJECTS
    objects_by_frame = _WORKER_OBJECTS_BY_FRAME
    objects_by_pool = _WORKER_OBJECTS_BY_POOL
    cache = _WORKER_CACHE
    rng_pool = _WORKER_RNG_POOL
    preview_ids = _WORKER_PREVIEW_IDS or set()
    fid = int(job[0])
    set_count = int(job[1])
    rep_offset = int(job[2]) if len(job) > 2 else 0
    if cfg is None or img_dir is None or dirs is None or all_objects is None or objects_by_frame is None or objects_by_pool is None or cache is None or rng_pool is None:
        raise RuntimeError("Worker is not initialized.")

    frame_objects = objects_by_frame.get(int(fid), [])
    if not frame_objects:
        return 0, 0, 0

    base_img = cv2.imread(os.path.join(img_dir, f"frame_{int(fid):06d}.png"), cv2.IMREAD_COLOR)
    if base_img is None:
        return 0, 0, 0

    H, W = base_img.shape[:2]
    ref_items, base_label_lines, base_mask_lines = build_frame_items(frame_objects, cache, W, H)
    if not ref_items:
        return 0, 0, 0

    cluster_count = int(cfg.get("CLUSTER_COUNT", 12))
    if cluster_count < 2:
        raise ValueError("CLUSTER_COUNT must be at least 2.")

    # Cluster-specific parameters are intentionally separated from normal interaction_image_synthesis.py.
    min_cover = float(cfg.get("MIN_OVERLAP", 0.01))
    max_cover = float(cfg.get("MAX_OVERLAP", 0.5))
    max_tries = int(cfg.get("MAX_TRIES", 100))
    paste_layer_mode = str(cfg.get("PASTE_LAYER_MODE", "mixed"))
    under_paste_prob = float(cfg.get("UNDER_PASTE_PROB", 0.5))
    alpha_mode = str(cfg.get("ALPHA_MODE", "distance"))
    edge_feather_px = int(cfg.get("EDGE_FEATHER_PX", 4))
    _efmin = cfg.get("FEATHER_MIN")
    _efmax = cfg.get("FEATHER_MAX")
    edge_feather_min_px: Optional[int] = int(_efmin) if _efmin is not None else None
    edge_feather_max_px: Optional[int] = int(_efmax) if _efmax is not None else None
    paste_brightness_min = float(cfg.get("BRIGHT_MIN", 1.0))
    paste_brightness_max = float(cfg.get("BRIGHT_MAX", 1.0))
    paste_contrast_min = float(cfg.get("CONTRAST_MIN", 1.0))
    paste_contrast_max = float(cfg.get("CONTRAST_MAX", 1.0))
    edge_blur_ksize = int(cfg.get("EDGE_BLUR_KSIZE", 7))
    edge_blur_sigma = float(cfg.get("EDGE_BLUR_SIGMA", 11.0))
    occluder_margin_px = int(cfg.get("OCCLUDER_MARGIN", -2))
    paste_scale_min = float(cfg.get("PASTE_SCALE_MIN", 1.0))
    paste_scale_max = float(cfg.get("PASTE_SCALE_MAX", 1.0))
    paste_width_scale_min, paste_width_scale_max = paste_width_scale_range_from_config(cfg)
    paste_min_obb_aspect_ratio = float(cfg.get("PASTE_MIN_ASPECT", 1.1))
    mask_expansion_ratio = get_mask_expansion_ratio(cfg)
    cluster_obb_fit_scale_long = float(cfg.get("CLUSTER_FIT_LONG", 0.8))
    cluster_obb_fit_scale_short = float(cfg.get("CLUSTER_FIT_SHORT", 0.8))
    parallel_break_prob = float(cfg.get("CLUSTER_BREAK_PROB", 0.20))
    cluster_donor_scope = "same_frame"


    def _generate_one_set(repeat_index: int) -> Tuple[int, int, int]:
        rng = make_python_rng(
            rng_pool.seed, "paste_blobs_clustered", int(fid), cluster_count, int(repeat_index),
        )

        out_img = base_img.copy()
        occupied = np.zeros((H, W), np.uint8)
        for item in ref_items:
            x, y, _, _ = item["rect"]
            paste_mask_into_full(occupied, item["mask"], int(x), int(y))
        label_lines = list(base_label_lines)
        mask_lines = list(base_mask_lines)

        # Per-object masks for pairwise external overlap checking.
        _base_external_objects: List[Tuple[int, int, np.ndarray]] = [
            (int(item["rect"][0]), int(item["rect"][1]), item["mask"]) for item in ref_items
        ]
        # Accumulates individual donor masks as clusters are accepted.
        _pasted_cluster_masks: List[Tuple[int, int, np.ndarray]] = []

        # Debug counters for this set.
        _cnt_two_edge = 0
        _cnt_one_edge = 0
        _cnt_ext_rejected_donor = 0
        _cnt_ext_precheck_skipped = 0

        def make_cluster_group_candidate(target: dict, cluster_size: int) -> Optional[dict]:
            """Build one target-centered cluster by deterministic fit-scale OBB geometry.

            Intra-cluster fit OBB contacts use a hard collision gate (no MAX_OVERLAP relaxation).
            Extra-cluster overlap (base blobs / pasted donors outside this cluster) is checked
            per-object with pairwise real-mask overlap; MAX_OVERLAP applies there.
            """
            nonlocal _cnt_two_edge, _cnt_one_edge, _cnt_ext_rejected_donor, _cnt_ext_precheck_skipped
            k_neighbors = int(cluster_size) - 1
            if k_neighbors <= 0:
                return None

            donor_candidates = list(ref_items)
            if not donor_candidates:
                return None
            if len(donor_candidates) >= k_neighbors:
                donor_pool = rng.sample(donor_candidates, k_neighbors)
            else:
                donor_pool = [rng.choice(donor_candidates) for _ in range(k_neighbors)]

            cx, cy, cw, ch = target["rect"]
            center_area = int(cv2.countNonZero(target["mask"]))
            if center_area <= 0:
                return None

            center_mask_full = full_mask_from_local(target["mask"], int(cx), int(cy), H, W)
            group_mask_full = np.zeros((H, W), np.uint8)
            target_fit_obb_pts = scale_obb_points_long_short(
                target["obb_pts"],
                float(cluster_obb_fit_scale_long),
                float(cluster_obb_fit_scale_short),
            )
            group_obb_mask_full = polygon_mask_from_points(target_fit_obb_pts, H, W)
            target_center = obb_center(target_fit_obb_pts).astype(np.float32)
            target_u, target_v, target_head_angle = target_obb_frame({**target, "obb_pts": target_fit_obb_pts})
            target_u = normalize_vec2((float(target_u[0]), float(target_u[1])))
            target_v = normalize_vec2((float(target_v[0]), float(target_v[1])))
            if target.get("head_angle_deg") is not None:
                target_head_angle = float(target.get("head_angle_deg"))

            def _project_target_xy(pt: np.ndarray) -> Tuple[float, float]:
                q = np.asarray(pt, dtype=np.float32).reshape(2) - target_center
                return float(np.dot(q, target_u)), float(np.dot(q, target_v))

            tx = []
            ty = []
            for pnt in np.asarray(target_fit_obb_pts, dtype=np.float32).reshape(4, 2):
                px, py = _project_target_xy(pnt)
                tx.append(px)
                ty.append(py)

            def _rect_record(name: str, cx0: float, cy0: float, hx: float, hy: float, angle_label: str, index: int) -> dict:
                return {
                    "name": name,
                    "cx": float(cx0), "cy": float(cy0),
                    "hx": float(hx), "hy": float(hy),
                    "xmin": float(cx0 - hx), "xmax": float(cx0 + hx),
                    "ymin": float(cy0 - hy), "ymax": float(cy0 + hy),
                    "angle_label": str(angle_label),
                    "index": int(index),
                }

            placed_rects: List[dict] = [
                _rect_record(
                    "target", 0.0, 0.0,
                    0.5 * (max(tx) - min(tx)),
                    0.5 * (max(ty) - min(ty)),
                    "target", 0,
                )
            ]

            orientation_cycle = ["perpendicular", "anti_perpendicular", "parallel", "anti_parallel"]
            orientation_slots = [orientation_cycle[ii % len(orientation_cycle)] for ii in range(k_neighbors)]
            rng.shuffle(orientation_slots)

            def _orientation_for_slot(slot: int) -> str:
                # Keep the four target-relative directions balanced, but randomize their order
                # for each target cluster so the cluster grows outward from the target without
                # a fixed deterministic direction pattern.
                return orientation_slots[int(slot) % len(orientation_slots)]

            def _desired_angle(label: str) -> float:
                return orientation_angle(float(target_head_angle), label)

            def _fit_obb_long_short_lengths(obb_pts: np.ndarray) -> Tuple[float, float]:
                pts = ensure_clockwise(np.asarray(obb_pts, dtype=np.float32).reshape(4, 2))
                lens = [float(np.linalg.norm(pts[(ii + 1) % 4] - pts[ii])) for ii in range(4)]
                if not lens:
                    return 0.0, 0.0
                return float(max(lens)), float(min(lens))

            def _short_edge_centers_local(obb_pts: np.ndarray) -> List[np.ndarray]:
                pts = ensure_clockwise(np.asarray(obb_pts, dtype=np.float32).reshape(4, 2))
                lens = [float(np.linalg.norm(pts[(ii + 1) % 4] - pts[ii])) for ii in range(4)]
                if not lens:
                    return []
                min_len = min(lens)
                tol = max(1e-6, 0.05 * max(lens))
                centers = []
                for ii, ln in enumerate(lens):
                    if abs(float(ln) - float(min_len)) <= tol:
                        centers.append((0.5 * (pts[ii] + pts[(ii + 1) % 4])).astype(np.float32))
                return centers

            def _prepare_donor_for_angle(donor: dict, desired_angle: float, orientation_label: str) -> Optional[dict]:
                donor_img, donor_mask = donor_image_and_mask(
                    donor, base_img, cache, cluster_donor_scope,
                    rng=rng, mask_expansion_ratio=mask_expansion_ratio,
                )
                if donor_img is None or donor_mask is None or donor_img.shape[:2] != donor_mask.shape[:2]:
                    return None
                donor_head_angle = donor_head_angle_from_direction(donor)
                paste_scale = rng.uniform(min(paste_scale_min, paste_scale_max), max(paste_scale_min, paste_scale_max))
                width_scale, height_scale = sample_paste_width_scales(
                    rng,
                    paste_width_scale_min,
                    paste_width_scale_max,
                )
                rot_deg = (float(donor_head_angle) - float(desired_angle)) % 360.0 if donor_head_angle is not None else float(desired_angle)
                transformed = rotate_crop_with_mask_and_obb_scaling(
                    donor_img,
                    donor_mask,
                    rot_deg,
                    paste_scale,
                    width_scale,
                    height_scale,
                    paste_min_obb_aspect_ratio,
                )
                if transformed is None:
                    return None
                rot_img, rot_mask, M = transformed
                dh, dw = rot_mask.shape[:2]
                if dh <= 0 or dw <= 0 or dh > H or dw > W or cv2.countNonZero(rot_mask) <= 0:
                    return None
                donor_obb_local = obb_points_from_mask_local(rot_mask)
                if donor_obb_local is None:
                    return None
                donor_obb_local = ensure_clockwise(donor_obb_local)
                donor_fit_obb_local = scale_obb_points_long_short(
                    donor_obb_local,
                    float(cluster_obb_fit_scale_long),
                    float(cluster_obb_fit_scale_short),
                )
                donor_fit_center_local = obb_center(donor_fit_obb_local).astype(np.float32)
                rel = donor_fit_obb_local - donor_fit_center_local.reshape(1, 2)
                hx = float(np.max(np.abs(rel @ target_u.reshape(2, 1))))
                hy = float(np.max(np.abs(rel @ target_v.reshape(2, 1))))
                if hx <= 0.5 or hy <= 0.5:
                    return None
                donor_fit_obb_mask_local = polygon_mask_from_points(donor_fit_obb_local, dh, dw)
                donor_fit_area = int(cv2.countNonZero(donor_fit_obb_mask_local))
                if donor_fit_area <= 0:
                    return None
                fit_long_len, fit_short_len = _fit_obb_long_short_lengths(donor_fit_obb_local)
                short_edge_centers = _short_edge_centers_local(donor_fit_obb_local)
                return {
                    "donor": donor,
                    "orientation_label": str(orientation_label),
                    "desired_angle": float(desired_angle),
                    "rot_deg": float(rot_deg),
                    "paste_scale": float(paste_scale),
                    "paste_width_scale": float(width_scale),
                    "rot_img": rot_img,
                    "rot_mask": rot_mask,
                    "M": M,
                    "dh": int(dh), "dw": int(dw),
                    "real_obb_local": donor_obb_local,
                    "fit_obb_local": donor_fit_obb_local,
                    "fit_center_local": donor_fit_center_local,
                    "fit_mask_local": donor_fit_obb_mask_local,
                    "fit_area": int(donor_fit_area),
                    "hx": float(hx), "hy": float(hy),
                    "fit_long_len": float(fit_long_len),
                    "fit_short_len": float(fit_short_len),
                    "short_edge_centers_local": short_edge_centers,
                }

            def _prepare_donor_for_orientation(donor: dict, orientation_label: str) -> Optional[dict]:
                return _prepare_donor_for_angle(donor, _desired_angle(orientation_label), str(orientation_label))

            prepared: List[dict] = []
            for slot, donor in enumerate(donor_pool):
                prep = _prepare_donor_for_orientation(donor, _orientation_for_slot(slot))
                if prep is None:
                    return None
                prepared.append(prep)

            def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
                return max(0.0, min(float(a1), float(b1)) - max(float(a0), float(b0)))

            def _rect_interior_overlap(a: dict, b: dict, eps: float = 0.75) -> bool:
                return (_interval_overlap(a["xmin"] + eps, a["xmax"] - eps, b["xmin"] + eps, b["xmax"] - eps) > 0.0 and
                        _interval_overlap(a["ymin"] + eps, a["ymax"] - eps, b["ymin"] + eps, b["ymax"] - eps) > 0.0)

            def _contact_stats(cand: dict, rects: List[dict], eps: float = 1.25) -> Tuple[int, float, int, float]:
                contacts = 0
                length = 0.0
                target_contacts = 0
                long_long = 0.0
                for rct in rects:
                    local_len = 0.0
                    # vertical contact edges: long side only for target-parallel rectangles.
                    if abs(cand["xmin"] - rct["xmax"]) <= eps or abs(cand["xmax"] - rct["xmin"]) <= eps:
                        local_len = max(local_len, _interval_overlap(cand["ymin"], cand["ymax"], rct["ymin"], rct["ymax"]))
                    # horizontal contact edges: long side for target/perpendicular rectangles.
                    if abs(cand["ymin"] - rct["ymax"]) <= eps or abs(cand["ymax"] - rct["ymin"]) <= eps:
                        local_len = max(local_len, _interval_overlap(cand["xmin"], cand["xmax"], rct["xmin"], rct["xmax"]))
                    if local_len > 1.0:
                        contacts += 1
                        length += float(local_len)
                        if int(rct.get("index", -1)) == 0:
                            target_contacts += 1
                        cand_long_is_x = cand["hx"] >= cand["hy"]
                        r_long_is_x = rct["hx"] >= rct["hy"]
                        if (local_len > 1.0) and cand_long_is_x == r_long_is_x:
                            long_long = max(long_long, float(local_len))
                return int(contacts), float(length), int(target_contacts), float(long_long)

            def _target_xy_to_global(x: float, y: float) -> np.ndarray:
                return (target_center + target_u * float(x) + target_v * float(y)).astype(np.float32)

            def _global_to_target_xy(pt: np.ndarray) -> Tuple[float, float]:
                q = np.asarray(pt, dtype=np.float32).reshape(2) - target_center
                return float(np.dot(q, target_u)), float(np.dot(q, target_v))

            def _translated_fit_obb_rect(prep: dict, xo: int, yo: int, name: str, index: int) -> dict:
                pts_global = np.asarray(prep["fit_obb_local"], dtype=np.float32).reshape(4, 2) + np.asarray([float(xo), float(yo)], dtype=np.float32).reshape(1, 2)
                xs = []
                ys = []
                for pp in pts_global:
                    px, py = _global_to_target_xy(pp)
                    xs.append(px)
                    ys.append(py)
                cx0 = 0.5 * (min(xs) + max(xs))
                cy0 = 0.5 * (min(ys) + max(ys))
                return _rect_record(
                    name, cx0, cy0,
                    0.5 * (max(xs) - min(xs)),
                    0.5 * (max(ys) - min(ys)),
                    str(prep.get("orientation_label", "diagonal")),
                    int(index),
                )

            def _edge_covering_rects_for_gap(x0: float, x1: float, y0: float, y1: float, eps: float = 1.25) -> Optional[dict]:
                width = float(x1) - float(x0)
                height = float(y1) - float(y0)
                if width <= 2.0 or height <= 2.0:
                    return None
                probe = _rect_record("gap", 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * width, 0.5 * height, "gap", -1)
                if any(_rect_interior_overlap(probe, old, eps=0.25) for old in placed_rects):
                    return None
                side_owners = {"left": set(), "right": set(), "bottom": set(), "top": set()}
                for rr in placed_rects:
                    idx = int(rr.get("index", -1))
                    if abs(float(rr["xmax"]) - float(x0)) <= eps and _interval_overlap(y0, y1, rr["ymin"], rr["ymax"]) >= 0.60 * height:
                        side_owners["left"].add(idx)
                    if abs(float(rr["xmin"]) - float(x1)) <= eps and _interval_overlap(y0, y1, rr["ymin"], rr["ymax"]) >= 0.60 * height:
                        side_owners["right"].add(idx)
                    if abs(float(rr["ymax"]) - float(y0)) <= eps and _interval_overlap(x0, x1, rr["xmin"], rr["xmax"]) >= 0.60 * width:
                        side_owners["bottom"].add(idx)
                    if abs(float(rr["ymin"]) - float(y1)) <= eps and _interval_overlap(x0, x1, rr["xmin"], rr["xmax"]) >= 0.60 * width:
                        side_owners["top"].add(idx)
                if not all(side_owners[k] for k in side_owners):
                    return None
                owners = set().union(*side_owners.values())
                # The intended case is a rectangular void formed by four placed OBBs.
                # Allow three unique owners as a practical minimum because one elongated
                # OBB can sometimes cover two sides after anisotropic fit scaling.
                if len(owners) < 3:
                    return None
                return {"rect": probe, "owners": owners, "side_owners": side_owners, "width": width, "height": height}

            def _rectangular_gap_candidates(max_gaps: int = 16) -> List[dict]:
                xs = sorted({round(float(v), 4) for rr in placed_rects for v in (rr["xmin"], rr["xmax"])})
                ys = sorted({round(float(v), 4) for rr in placed_rects for v in (rr["ymin"], rr["ymax"])})
                gaps: List[dict] = []
                for ii in range(len(xs) - 1):
                    x0, x1 = float(xs[ii]), float(xs[ii + 1])
                    if x1 - x0 <= 2.0:
                        continue
                    for jj in range(len(ys) - 1):
                        y0, y1 = float(ys[jj]), float(ys[jj + 1])
                        if y1 - y0 <= 2.0:
                            continue
                        bounded = _edge_covering_rects_for_gap(x0, x1, y0, y1)
                        if bounded is None:
                            continue
                        diag = math.hypot(x1 - x0, y1 - y0)
                        bounded.update({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "diag_len": float(diag)})
                        gaps.append(bounded)
                target_bias_center = np.asarray([0.0, 0.0], dtype=np.float32)
                gaps.sort(key=lambda g: (
                    -float(g["diag_len"]),
                    math.hypot(float(g["rect"]["cx"] - target_bias_center[0]), float(g["rect"]["cy"] - target_bias_center[1])),
                    rng.random(),
                ))
                return gaps[:int(max_gaps)]

            def _find_diagonal_gap_placement(donor: dict, slot: int) -> Optional[Tuple[dict, dict]]:
                nonlocal _cnt_ext_rejected_donor
                if len(placed_rects) < 4:
                    return None
                _ext_aabbs_diag = build_external_aabb_array(external_objs_no_target)
                candidates: List[Tuple[Tuple[float, ...], dict, dict]] = []
                angle_cache: Dict[float, Optional[dict]] = {}

                def _prep_for_diag(dx: float, dy: float) -> Optional[dict]:
                    diag_vec_global = target_u * float(dx) + target_v * float(dy)
                    if float(np.linalg.norm(diag_vec_global)) <= 1e-6:
                        return None
                    desired_angle = head_angle_deg_from_axis((float(diag_vec_global[0]), float(diag_vec_global[1])))
                    key = round(float(desired_angle) % 360.0, 3)
                    if key not in angle_cache:
                        angle_cache[key] = _prepare_donor_for_angle(donor, desired_angle, "diagonal_gap")
                    return angle_cache[key]

                for gap in _rectangular_gap_candidates():
                    x0, x1, y0, y1 = float(gap["x0"]), float(gap["x1"]), float(gap["y0"]), float(gap["y1"])
                    diag_pairs = [
                        ((x0, y0), (x1, y1)),
                        ((x0, y1), (x1, y0)),
                        ((x1, y1), (x0, y0)),
                        ((x1, y0), (x0, y1)),
                    ]
                    rng.shuffle(diag_pairs)
                    for start_xy, end_xy in diag_pairs:
                        dx = float(end_xy[0] - start_xy[0])
                        dy = float(end_xy[1] - start_xy[1])
                        diag_len = math.hypot(dx, dy)
                        prep = _prep_for_diag(dx, dy)
                        if prep is None:
                            continue
                        if float(diag_len) <= float(prep.get("fit_long_len", 0.0)):
                            continue
                        short_centers = list(prep.get("short_edge_centers_local", []))
                        if not short_centers:
                            continue
                        diag_unit_global = normalize_vec2((float((target_u * dx + target_v * dy)[0]), float((target_u * dx + target_v * dy)[1])))
                        fit_center_local = np.asarray(prep["fit_center_local"], dtype=np.float32).reshape(2)
                        # Use the short edge at the rear of the diagonal direction.
                        short_centers.sort(key=lambda cc: float(np.dot(fit_center_local - np.asarray(cc, dtype=np.float32).reshape(2), diag_unit_global)), reverse=True)
                        start_global = _target_xy_to_global(float(start_xy[0]), float(start_xy[1]))
                        for short_center_local in short_centers[:2]:
                            top_left = start_global - np.asarray(short_center_local, dtype=np.float32).reshape(2)
                            xo = int(round(float(top_left[0])))
                            yo = int(round(float(top_left[1])))
                            if xo < 0 or yo < 0 or xo + int(prep["dw"]) > W or yo + int(prep["dh"]) > H:
                                continue
                            pts_global = np.asarray(prep["fit_obb_local"], dtype=np.float32).reshape(4, 2) + np.asarray([float(xo), float(yo)], dtype=np.float32).reshape(1, 2)
                            pts_xy = [_global_to_target_xy(pp) for pp in pts_global]
                            xs2 = [pp[0] for pp in pts_xy]
                            ys2 = [pp[1] for pp in pts_xy]
                            margin = 1.5
                            if min(xs2) < x0 - margin or max(xs2) > x1 + margin or min(ys2) < y0 - margin or max(ys2) > y1 + margin:
                                continue
                            cand_rect = _translated_fit_obb_rect(prep, xo, yo, "donor", slot + 1)
                            if any(_rect_interior_overlap(cand_rect, old, eps=0.25) for old in placed_rects):
                                continue
                            reject, total_obb_overlap, total_obb_ratio, non_contact_overlap, non_contact_ratio = reject_cluster_fit_obb_overlap(
                                group_obb_mask_full_u8=group_obb_mask_full,
                                donor_obb_mask_local=prep["fit_mask_local"],
                                donor_obb_area=int(prep["fit_area"]),
                                xo=xo,
                                yo=yo,
                                allowed_contact_overlap_pixels=0,
                            )
                            if reject:
                                continue
                            # Inline external pairwise check for diagonal gap candidate.
                            _rot_m_d = prep["rot_mask"]
                            _d_area_d = max(1, cv2.countNonZero(_rot_m_d))
                            _outside_d = full_mask_overlap(external_occupied, _rot_m_d, xo, yo)
                            if _outside_d > 0:
                                if float(_outside_d) / float(_d_area_d) > float(max_cover):
                                    _cnt_ext_rejected_donor += 1
                                    continue
                                _diag_ext_reject = False
                                for _idx_d in external_aabb_intersecting_indices(
                                    xo, yo, xo + int(prep["dw"]), yo + int(prep["dh"]), _ext_aabbs_diag
                                ):
                                    _ex, _ey, _emask = external_objs_no_target[_idx_d]
                                    _ep = count_overlap_rect_mask(xo, yo, _rot_m_d, _ex, _ey, _emask)
                                    if float(_ep) / float(_d_area_d) > float(max_cover):
                                        _cnt_ext_rejected_donor += 1
                                        _diag_ext_reject = True
                                        break
                                if _diag_ext_reject:
                                    continue
                            target_dist = math.hypot(float(cand_rect["cx"]), float(cand_rect["cy"]))
                            quality = (
                                5000.0
                                + 20.0 * float(diag_len - float(prep.get("fit_long_len", 0.0)))
                                - 2.0 * float(target_dist)
                                + rng.random()
                            )
                            geom = {
                                "rect": cand_rect, "xo": xo, "yo": yo,
                                "total_obb_overlap": int(total_obb_overlap),
                                "total_obb_ratio": float(total_obb_ratio),
                                "non_contact_overlap": int(non_contact_overlap),
                                "non_contact_ratio": float(non_contact_ratio),
                                "contacts": 2,
                                "contact_len": float(prep.get("fit_short_len", 0.0)),
                                "target_contacts": 0,
                                "target_dist": float(target_dist),
                                "bucket": -1,
                                "long_long_len": 0.0,
                                "parallel_break_forced": 0,
                                "diagonal_gap_fill": 1,
                                "gap_diag_len": float(diag_len),
                            }
                            candidates.append(((quality, -target_dist, rng.random()), prep, geom))
                if not candidates:
                    return None
                candidates.sort(key=lambda x: x[0], reverse=True)
                top = candidates[:max(1, min(5, len(candidates)))]
                weights = [max(1e-6, 1.0 / float(ii + 1)) for ii in range(len(top))]
                _, prep, geom = rng.choices(top, weights=weights, k=1)[0]
                return prep, geom

            def _generate_second_edge_contact_candidates(owner: dict, hx: float, hy: float) -> List[Tuple[float, float]]:
                """Slide a 1-edge contact to also touch a second placed rect (2-edge contact)."""
                eps = 1.25
                results: List[Tuple[float, float]] = []
                sides = [
                    ("right",  float(owner["xmax"]) + float(hx), None),
                    ("left",   float(owner["xmin"]) - float(hx), None),
                    ("top",    None, float(owner["ymax"]) + float(hy)),
                    ("bottom", None, float(owner["ymin"]) - float(hy)),
                ]
                for side, fixed_cx, fixed_cy in sides:
                    for r in placed_rects:
                        if r is owner:
                            continue
                        if side in ("right", "left"):
                            cx = float(fixed_cx)
                            for cy in [float(r["ymax"]) + float(hy), float(r["ymin"]) - float(hy)]:
                                # Anchor (owner) vertical edge contact must be maintained.
                                y_ov = _interval_overlap(cy - hy, cy + hy, owner["ymin"], owner["ymax"])
                                if y_ov <= 1.0:
                                    continue
                                # Must have horizontal edge contact with r.
                                if not (abs((cy - hy) - float(r["ymax"])) <= eps or
                                        abs((cy + hy) - float(r["ymin"])) <= eps):
                                    continue
                                x_ov = _interval_overlap(cx - hx, cx + hx, r["xmin"], r["xmax"])
                                if x_ov <= 1.0:
                                    continue
                                results.append((cx, cy))
                        else:
                            cy = float(fixed_cy)
                            for cx in [float(r["xmax"]) + float(hx), float(r["xmin"]) - float(hx)]:
                                # Anchor horizontal edge contact must be maintained.
                                x_ov = _interval_overlap(cx - hx, cx + hx, owner["xmin"], owner["xmax"])
                                if x_ov <= 1.0:
                                    continue
                                # Must have vertical edge contact with r.
                                if not (abs((cx - hx) - float(r["xmax"])) <= eps or
                                        abs((cx + hx) - float(r["xmin"])) <= eps):
                                    continue
                                y_ov = _interval_overlap(cy - hy, cy + hy, r["ymin"], r["ymax"])
                                if y_ov <= 1.0:
                                    continue
                                results.append((cx, cy))
                return results

            def _rect_long_axis(rect: dict) -> str:
                return "x" if float(rect["hx"]) >= float(rect["hy"]) else "y"

            def _rect_long_len(rect: dict) -> float:
                return 2.0 * max(float(rect["hx"]), float(rect["hy"]))

            def _detect_long_long_half_contact(new_rect: dict, rects: List[dict], eps: float = 1.25) -> Optional[dict]:
                """Detect a long-edge row and force the next donor to stand perpendicular.

                If the newly placed OBB shares at least half of the shorter long edge
                with another OBB's long edge, the next placement is made from the
                outer long edge of this newly placed OBB instead of from an L socket.
                """
                new_axis = _rect_long_axis(new_rect)
                best = None
                for old in rects:
                    if _rect_long_axis(old) != new_axis:
                        continue
                    overlap_len = 0.0
                    contact_side = None
                    if new_axis == "x":
                        if abs(float(new_rect["ymin"]) - float(old["ymax"])) <= eps:
                            overlap_len = _interval_overlap(new_rect["xmin"], new_rect["xmax"], old["xmin"], old["xmax"])
                            contact_side = "bottom"
                        elif abs(float(new_rect["ymax"]) - float(old["ymin"])) <= eps:
                            overlap_len = _interval_overlap(new_rect["xmin"], new_rect["xmax"], old["xmin"], old["xmax"])
                            contact_side = "top"
                    else:
                        if abs(float(new_rect["xmin"]) - float(old["xmax"])) <= eps:
                            overlap_len = _interval_overlap(new_rect["ymin"], new_rect["ymax"], old["ymin"], old["ymax"])
                            contact_side = "left"
                        elif abs(float(new_rect["xmax"]) - float(old["xmin"])) <= eps:
                            overlap_len = _interval_overlap(new_rect["ymin"], new_rect["ymax"], old["ymin"], old["ymax"])
                            contact_side = "right"
                    if contact_side is None:
                        continue
                    threshold = 0.5 * min(_rect_long_len(new_rect), _rect_long_len(old))
                    if float(overlap_len) < float(threshold):
                        continue
                    outside_side = {
                        "bottom": "top",
                        "top": "bottom",
                        "left": "right",
                        "right": "left",
                    }[contact_side]
                    score = float(overlap_len) - float(threshold)
                    payload = {
                        "owner": dict(new_rect),
                        "outside_side": outside_side,
                        "owner_long_axis": new_axis,
                        "overlap_len": float(overlap_len),
                    }
                    if best is None or score > best[0]:
                        best = (score, payload)
                return None if best is None else best[1]

            def _perpendicular_orientation_for_owner_long_axis(owner_long_axis: str) -> str:
                if str(owner_long_axis) == "x":
                    return rng.choice(["perpendicular", "anti_perpendicular"])
                return rng.choice(["parallel", "anti_parallel"])

            def _forced_parallel_break_centers(owner: dict, hx: float, hy: float, outside_side: str) -> List[Tuple[float, float]]:
                centers: List[Tuple[float, float]] = []
                side = str(outside_side)
                owner_axis = _rect_long_axis(owner)
                if owner_axis == "x":
                    if float(hx) <= float(owner["hx"]):
                        max_shift = max(0.0, float(owner["hx"]) - float(hx))
                        offsets = [0.0, 0.5 * max_shift, -0.5 * max_shift, max_shift, -max_shift]
                    else:
                        offsets = [0.0]
                    offsets = list(dict.fromkeys([round(float(v), 6) for v in offsets]))
                    rng.shuffle(offsets)
                    y = float(owner["ymax"]) + float(hy) if side == "top" else float(owner["ymin"]) - float(hy)
                    for ox in offsets:
                        centers.append((float(owner["cx"]) + float(ox), y))
                else:
                    if float(hy) <= float(owner["hy"]):
                        max_shift = max(0.0, float(owner["hy"]) - float(hy))
                        offsets = [0.0, 0.5 * max_shift, -0.5 * max_shift, max_shift, -max_shift]
                    else:
                        offsets = [0.0]
                    offsets = list(dict.fromkeys([round(float(v), 6) for v in offsets]))
                    rng.shuffle(offsets)
                    x = float(owner["xmax"]) + float(hx) if side == "right" else float(owner["xmin"]) - float(hx)
                    for oy in offsets:
                        centers.append((x, float(owner["cy"]) + float(oy)))
                return centers

            def _candidate_centers_for_owner(owner: dict, hx: float, hy: float, slot: int) -> List[Tuple[float, float]]:
                centers: List[Tuple[float, float]] = []
                x_offsets = [0.0, owner["hx"] - hx, -owner["hx"] + hx, owner["hx"] + hx, -owner["hx"] - hx]
                y_offsets = [0.0, owner["hy"] - hy, -owner["hy"] + hy, owner["hy"] + hy, -owner["hy"] - hy]
                rng.shuffle(x_offsets)
                rng.shuffle(y_offsets)

                # The first donor is always grown directly from the target.  Prefer one of
                # the target long-edge sides, but choose the side randomly so clusters do
                # not all start in the same direction.
                if slot == 0 and int(owner.get("index", -1)) == 0:
                    long_edge_sides = ["top", "bottom"] if owner["hx"] >= owner["hy"] else ["left", "right"]
                    other_sides = [s for s in ["right", "left", "top", "bottom"] if s not in long_edge_sides]
                    rng.shuffle(long_edge_sides)
                    rng.shuffle(other_sides)
                    sides = long_edge_sides + other_sides
                else:
                    sides = ["right", "left", "top", "bottom"]
                    rng.shuffle(sides)

                for side in sides:
                    if side == "right":
                        x = owner["xmax"] + hx
                        for oy in y_offsets:
                            centers.append((x, owner["cy"] + oy))
                    elif side == "left":
                        x = owner["xmin"] - hx
                        for oy in y_offsets:
                            centers.append((x, owner["cy"] + oy))
                    elif side == "top":
                        y = owner["ymax"] + hy
                        for ox in x_offsets:
                            centers.append((owner["cx"] + ox, y))
                    elif side == "bottom":
                        y = owner["ymin"] - hy
                        for ox in x_offsets:
                            centers.append((owner["cx"] + ox, y))
                return centers

            def _candidate_bucket(*, contacts: int, target_contacts: int, slot: int) -> int:
                # Lower is better.  L-shaped sockets are still the primary primitive.
                # Among L sockets, sockets touching the original target are preferred.
                if slot == 0 and target_contacts > 0:
                    return 0
                if contacts >= 2 and target_contacts > 0:
                    return 0
                if contacts >= 2:
                    return 1
                if target_contacts > 0:
                    return 2
                return 3

            def _choose_randomized_candidate(candidates: List[Tuple[Tuple[float, ...], dict]]) -> dict:
                # Preserve geometric quality, but avoid a deterministic best-only choice.
                # First take the best priority bucket, then randomly choose within the
                # upper quality band.  This makes growth spread from the target while still
                # preferring target-touching L sockets when such sockets exist.
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_bucket = int(candidates[0][1]["bucket"])
                same_bucket = [x for x in candidates if int(x[1]["bucket"]) == best_bucket]
                same_bucket.sort(key=lambda x: x[0], reverse=True)
                keep_n = max(1, min(len(same_bucket), 8))
                quality_pool = same_bucket[:keep_n]
                weights = []
                for rank, (_, payload) in enumerate(quality_pool):
                    contact_len = float(payload.get("contact_len", 0.0))
                    contacts = float(payload.get("contacts", 1.0))
                    dist = float(payload.get("target_dist", 0.0))
                    # Target-near bias: after direct target-touching L sockets are exhausted,
                    # prefer ordinary L sockets that remain close to the original target.
                    # The divisor is intentionally mild so contact quality still matters.
                    weights.append(max(1e-6, (1.0 + contacts) * (1.0 + 0.02 * contact_len) / ((1.0 + 0.002 * dist) * float(rank + 1))))
                return rng.choices([payload for _, payload in quality_pool], weights=weights, k=1)[0]

            def _find_geometric_position(prep: dict, slot: int, parallel_break: Optional[dict] = None) -> Optional[dict]:
                hx = float(prep["hx"])
                hy = float(prep["hy"])
                candidates: List[Tuple[Tuple[float, ...], dict]] = []
                _ext_aabbs_geo = build_external_aabb_array(external_objs_no_target)

                def _try_add_candidate(gx: float, gy: float, forced_bucket: Optional[int] = None) -> None:
                    nonlocal _cnt_ext_rejected_donor, _cnt_ext_precheck_skipped
                    cand = _rect_record("donor", gx, gy, hx, hy, prep["orientation_label"], slot + 1)
                    if any(_rect_interior_overlap(cand, old) for old in placed_rects):
                        return
                    global_center = target_center + target_u * float(gx) + target_v * float(gy)
                    top_left = global_center - prep["fit_center_local"]
                    xo = int(round(float(top_left[0])))
                    yo = int(round(float(top_left[1])))
                    if xo < 0 or yo < 0 or xo + int(prep["dw"]) > W or yo + int(prep["dh"]) > H:
                        return
                    reject, total_obb_overlap, total_obb_ratio, non_contact_overlap, non_contact_ratio = reject_cluster_fit_obb_overlap(
                        group_obb_mask_full_u8=group_obb_mask_full,
                        donor_obb_mask_local=prep["fit_mask_local"],
                        donor_obb_area=int(prep["fit_area"]),
                        xo=xo,
                        yo=yo,
                        allowed_contact_overlap_pixels=0,
                    )
                    if reject:
                        return
                    # Inline external pairwise check: union precheck then per-object.
                    _rot_m = prep["rot_mask"]
                    _d_area_m = max(1, cv2.countNonZero(_rot_m))
                    _outside_m = full_mask_overlap(external_occupied, _rot_m, xo, yo)
                    if _outside_m == 0:
                        _cnt_ext_precheck_skipped += 1
                    elif float(_outside_m) / float(_d_area_m) > float(max_cover):
                        _cnt_ext_rejected_donor += 1
                        return
                    else:
                        for _idx_g in external_aabb_intersecting_indices(
                            xo, yo, xo + int(prep["dw"]), yo + int(prep["dh"]), _ext_aabbs_geo
                        ):
                            _ex, _ey, _emask = external_objs_no_target[_idx_g]
                            _ep = count_overlap_rect_mask(xo, yo, _rot_m, _ex, _ey, _emask)
                            if float(_ep) / float(_d_area_m) > float(max_cover):
                                _cnt_ext_rejected_donor += 1
                                return
                    contacts, contact_len, target_contacts, long_long_len = _contact_stats(cand, placed_rects)
                    if contacts <= 0:
                        return
                    target_dist = math.hypot(float(gx), float(gy))
                    bucket = int(forced_bucket) if forced_bucket is not None else _candidate_bucket(contacts=contacts, target_contacts=target_contacts, slot=slot)
                    # Long-long contact is now treated as a row signal, not as a quality bonus.
                    # Penalizing it here prevents repeated parallel tiling; the explicit
                    # parallel-break directive below handles the next perpendicular placement.
                    quality = (
                        -100000.0 * float(bucket)
                        + 250.0 * min(float(contacts), 3.0)
                        + 30.0 * float(contact_len)
                        - 80.0 * float(long_long_len)
                        - 3.0 * float(target_dist)
                        + rng.random()
                    )
                    candidates.append(((quality, contact_len, target_dist, rng.random()), {
                        "rect": cand, "xo": xo, "yo": yo,
                        "total_obb_overlap": int(total_obb_overlap),
                        "total_obb_ratio": float(total_obb_ratio),
                        "non_contact_overlap": int(non_contact_overlap),
                        "non_contact_ratio": float(non_contact_ratio),
                        "contacts": int(contacts),
                        "contact_len": float(contact_len),
                        "target_contacts": int(target_contacts),
                        "target_dist": float(target_dist),
                        "bucket": int(bucket),
                        "long_long_len": float(long_long_len),
                        "parallel_break_forced": int(forced_bucket is not None),
                    }))

                if parallel_break is not None:
                    owner = parallel_break["owner"]
                    side = str(parallel_break["outside_side"])
                    for gx, gy in _forced_parallel_break_centers(owner, hx, hy, side):
                        _try_add_candidate(gx, gy, forced_bucket=0)
                    if not candidates:
                        return None
                    return _choose_randomized_candidate(candidates)

                if slot == 0:
                    owners = [placed_rects[0]]
                else:
                    target_owner = placed_rects[0]
                    other_owners = placed_rects[1:]
                    rng.shuffle(other_owners)
                    # Candidate generation is target-anchored first, then grows through
                    # already accepted donors in random order.  The final selection still
                    # prioritizes target-touching L sockets globally.
                    owners = [target_owner] + other_owners

                for owner in owners:
                    for gx, gy in _candidate_centers_for_owner(owner, hx, hy, slot):
                        _try_add_candidate(gx, gy)
                    # Two-edge contact: slide 1-edge positions to also touch a second rect.
                    if len(placed_rects) >= 2:
                        for gx, gy in _generate_second_edge_contact_candidates(owner, hx, hy):
                            _try_add_candidate(gx, gy)
                if not candidates:
                    return None
                return _choose_randomized_candidate(candidates)

            def _paste_prepared(prep: dict, geom: dict) -> Optional[dict]:
                xo = int(geom["xo"])
                yo = int(geom["yo"])
                dw = int(prep["dw"])
                dh = int(prep["dh"])
                rot_img = prep["rot_img"]
                rot_mask = prep["rot_mask"]
                M = prep["M"]
                roi = out_img[yo:yo + dh, xo:xo + dw]
                image_backup = roi.copy()
                occupied_backup = occupied[yo:yo + dh, xo:xo + dw].copy()
                group_backup = group_mask_full[yo:yo + dh, xo:xo + dw].copy()
                group_obb_backup = group_obb_mask_full[yo:yo + dh, xo:xo + dw].copy()

                under = choose_under(rng, paste_layer_mode, under_paste_prob)
                if under:
                    occluder_roi = cv2.bitwise_or(center_mask_full[yo:yo + dh, xo:xo + dw], group_mask_full[yo:yo + dh, xo:xo + dw])
                    if occluder_margin_px != 0:
                        k = 2 * abs(int(occluder_margin_px)) + 1
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                        occluder_roi = cv2.dilate(occluder_roi, kernel, iterations=1) if occluder_margin_px > 0 else cv2.erode(occluder_roi, kernel, iterations=1)
                    visible_mask = cv2.bitwise_and(rot_mask, cv2.bitwise_not(occluder_roi))
                    mode_str = "under"
                else:
                    visible_mask = rot_mask
                    mode_str = "over"
                if cv2.countNonZero(visible_mask) <= 0:
                    return None
                if not valid_obb_aspect_ratio(visible_mask, paste_min_obb_aspect_ratio):
                    return None

                brightness = rng.uniform(paste_brightness_min, paste_brightness_max)
                contrast = rng.uniform(paste_contrast_min, paste_contrast_max)
                rot_img = apply_brightness_contrast(rot_img, brightness, contrast)
                _efp = edge_feather_px
                if edge_feather_min_px is not None and edge_feather_max_px is not None:
                    lo_f = min(int(edge_feather_min_px), int(edge_feather_max_px))
                    hi_f = max(int(edge_feather_min_px), int(edge_feather_max_px))
                    _efp = rng.randint(lo_f, hi_f)
                alpha = build_feather_alpha(
                    visible_mask,
                    alpha_mode=alpha_mode,
                    edge_blur_ksize=edge_blur_ksize,
                    edge_blur_sigma=edge_blur_sigma,
                    edge_feather_px=_efp,
                )[..., None]
                tmp = roi.copy()
                visible = visible_mask > 0
                tmp[visible] = rot_img[visible]
                out_img[yo:yo + dh, xo:xo + dw] = (
                    tmp.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
                ).astype(np.uint8)
                paste_mask_into_full(occupied, rot_mask, xo, yo)
                paste_mask_into_full(group_mask_full, rot_mask, xo, yo)
                paste_mask_into_full(group_obb_mask_full, prep["fit_mask_local"], xo, yo)

                nz = cv2.findNonZero(visible_mask)
                if nz is None:
                    return None
                x1, y1, bw, bh = cv2.boundingRect(nz)

                donor = prep["donor"]
                heading = transform_heading_info(
                    M,
                    direction_vec=donor["direction_vec"],
                    center=(float(donor["center_x"]) - float(donor["x"]), float(donor["center_y"]) - float(donor["y"])),
                    axis_length=float(donor["axis_length"]),
                    offset=(float(xo), float(yo)),
                )
                if heading is None:
                    out_img[yo:yo + dh, xo:xo + dw] = image_backup
                    occupied[yo:yo + dh, xo:xo + dw] = occupied_backup
                    group_mask_full[yo:yo + dh, xo:xo + dw] = group_backup
                    group_obb_mask_full[yo:yo + dh, xo:xo + dw] = group_obb_backup
                    return None

                donor_obb_global = np.asarray(prep["fit_obb_local"], dtype=np.float32).copy()
                donor_obb_global[:, 0] += float(xo)
                donor_obb_global[:, 1] += float(yo)
                donor_area = int(cv2.countNonZero(rot_mask))
                owner_overlap_pixels = int(full_mask_overlap(center_mask_full, rot_mask, xo, yo))
                return {
                    "bbox": (xo + int(x1), yo + int(y1), int(bw), int(bh)),
                    "mode": mode_str,
                    "xo": xo, "yo": yo,
                    "visible_mask": visible_mask.copy(),
                    "rot_img": rot_img.copy(),
                    "rot_mask": rot_mask.copy(),
                    "rot_deg": float(prep["rot_deg"]),
                    "paste_scale": float(prep["paste_scale"]),
                    "paste_width_scale": float(prep["paste_width_scale"]),
                    "direction_dx": heading["direction_dx"],
                    "direction_dy": heading["direction_dy"],
                    "direction_vec": heading["direction_vec"],
                    "center_x": heading["center_x"],
                    "center_y": heading["center_y"],
                    "axis_length": heading["axis_length"],
                    "class_id": heading["class_id"],
                    "class_name": heading["class_name"],
                    "head_angle_deg": heading["head_angle_deg"],
                    "occlusion_state": occlusion_state_from_mode(mode_str),
                    "is_pasted": 1,
                    "overlap_pixels": owner_overlap_pixels,
                    "overlap_ratio": float(owner_overlap_pixels) / float(max(1, donor_area)),
                    "non_target_overlap_pixels": 0,
                    "non_target_overlap_ratio": 0.0,
                    "global_forbidden_overlap_pixels": 0,
                    "global_forbidden_overlap_ratio": 0.0,
                    "obb_pts": donor_obb_global.astype(np.float32),
                    "obb_mask_full_u8": polygon_mask_from_points(donor_obb_global, H, W),
                    "rect_frame": geom["rect"],
                    "restore_roi": (int(xo), int(yo), int(dw), int(dh), image_backup, occupied_backup, group_backup, group_obb_backup),
                    "sampled_edge_feather_px": int(_efp),
                }

            group_objs = [base_object_to_group_obj(target)]
            accepted_results: List[dict] = []
            cluster_start_img = out_img.copy()
            cluster_start_occupied = occupied.copy()

            # Compute external context once for this cluster (used by inline candidate check
            # in _try_add_candidate and _find_diagonal_gap_placement).
            external_occupied = cv2.bitwise_and(cluster_start_occupied, cv2.bitwise_not(center_mask_full))
            _target_id = id(target)
            external_objs_no_target: List[Tuple[int, int, np.ndarray]] = [
                beo for item, beo in zip(ref_items, _base_external_objects) if id(item) != _target_id
            ] + list(_pasted_cluster_masks)

            pending_parallel_break: Optional[dict] = None
            for slot, prep0 in enumerate(prepared):
                prep = prep0
                geom = None
                active_parallel_break = pending_parallel_break
                pending_parallel_break = None
                if active_parallel_break is not None:
                    forced_orientation = _perpendicular_orientation_for_owner_long_axis(active_parallel_break["owner_long_axis"])
                    forced_prep = _prepare_donor_for_orientation(prep0["donor"], forced_orientation)
                    if forced_prep is None:
                        continue
                    prep = forced_prep
                else:
                    diagonal_gap = _find_diagonal_gap_placement(prep0["donor"], slot)
                    if diagonal_gap is not None:
                        prep, geom = diagonal_gap

                if geom is None:
                    geom = _find_geometric_position(prep, slot, active_parallel_break)
                if geom is None:
                    continue
                res = _paste_prepared(prep, geom)
                if res is None:
                    continue
                # Track two-edge vs one-edge contact for debug logging.
                if int(geom.get("contacts", 0)) >= 2:
                    _cnt_two_edge += 1
                else:
                    _cnt_one_edge += 1
                next_break = _detect_long_long_half_contact(res["rect_frame"], placed_rects)
                accepted_results.append(res)
                placed_rects.append(res["rect_frame"])
                group_objs.append(donor_result_to_group_obj(res))
                # Long-long row breaking is intentionally sparse.  Detect the row every
                # time, but force a perpendicular OBB only with the configured probability
                # so clusters do not overproduce perpendicular-standing donors.
                if next_break is not None and rng.random() < float(parallel_break_prob):
                    pending_parallel_break = next_break

            if not accepted_results:
                out_img[...] = cluster_start_img
                occupied[...] = cluster_start_occupied
                return None

            # Safety-net filter: most rejections handled by inline candidate check above.
            _ext_aabbs_filter = build_external_aabb_array(external_objs_no_target)
            kept_results: List[dict] = []
            for res in accepted_results:
                rot_mask = res.get("rot_mask")
                if rot_mask is None:
                    continue
                donor_area = int(cv2.countNonZero(rot_mask))
                if donor_area <= 0:
                    continue
                outside_pixels = full_mask_overlap(external_occupied, rot_mask, int(res["xo"]), int(res["yo"]))
                if outside_pixels > 0:
                    outside_ratio = float(outside_pixels) / float(donor_area)
                    # Union precheck: fast reject.
                    if outside_ratio > float(max_cover):
                        _cnt_ext_rejected_donor += 1
                        continue
                    # Pairwise check: AABB broad-phase then narrow-phase.
                    _ext_reject = False
                    _xo_r, _yo_r = int(res["xo"]), int(res["yo"])
                    for _idx_f in external_aabb_intersecting_indices(
                        _xo_r, _yo_r, _xo_r + rot_mask.shape[1], _yo_r + rot_mask.shape[0], _ext_aabbs_filter
                    ):
                        _ex, _ey, _emask = external_objs_no_target[_idx_f]
                        _ep = count_overlap_rect_mask(_xo_r, _yo_r, rot_mask, _ex, _ey, _emask)
                        if float(_ep) / float(donor_area) > float(max_cover):
                            _ext_reject = True
                            _cnt_ext_rejected_donor += 1
                            break
                    if _ext_reject:
                        continue
                    outside_ratio = float(outside_pixels) / float(donor_area)
                else:
                    outside_ratio = 0.0
                res["global_forbidden_overlap_pixels"] = int(outside_pixels)
                res["global_forbidden_overlap_ratio"] = float(outside_ratio)
                kept_results.append(res)

            out_img[...] = cluster_start_img
            occupied[...] = cluster_start_occupied

            if not kept_results:
                return None

            group_objs = [base_object_to_group_obj(target)]
            trial_label_lines: List[str] = []
            trial_mask_lines: List[str] = []
            final_group_mask = np.zeros((H, W), np.uint8)
            for res in kept_results:
                xo = int(res["xo"])
                yo = int(res["yo"])
                rot_img = res["rot_img"]
                rot_mask = res["rot_mask"]
                visible_mask = res["visible_mask"]
                dh, dw = visible_mask.shape[:2]
                roi = out_img[yo:yo + dh, xo:xo + dw]
                alpha = build_feather_alpha(
                    visible_mask,
                    alpha_mode=alpha_mode,
                    edge_blur_ksize=edge_blur_ksize,
                    edge_blur_sigma=edge_blur_sigma,
                    edge_feather_px=res.get("sampled_edge_feather_px", edge_feather_px),
                )[..., None]
                tmp = roi.copy()
                visible = visible_mask > 0
                tmp[visible] = rot_img[visible]
                out_img[yo:yo + dh, xo:xo + dw] = (
                    tmp.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
                ).astype(np.uint8)
                paste_mask_into_full(occupied, rot_mask, xo, yo)
                paste_mask_into_full(final_group_mask, rot_mask, xo, yo)
                # Track full donor mask for pairwise external overlap in subsequent clusters.
                _pasted_cluster_masks.append((int(xo), int(yo), rot_mask.copy()))
                obj = donor_result_to_group_obj(res)
                group_objs.append(obj)
                append_annotation(obj, trial_label_lines, trial_mask_lines, W, H)

            return {
                "group_objs": group_objs,
                "label_lines": trial_label_lines,
                "mask_lines": trial_mask_lines,
                "success_count": len(group_objs) - 1,
                "required_count": k_neighbors,
            }

        def build_cluster_groups(cluster_size: int, target_pool: List[dict], groups: List[dict], max_targets: Optional[int] = None) -> int:
            success = 0
            processed = 0
            target_count = len(target_pool) if max_targets is None else max(0, int(max_targets))
            while processed < target_count and target_pool:
                target_index = rng.randrange(len(target_pool))
                target = target_pool.pop(target_index)
                candidate = make_cluster_group_candidate(target, cluster_size)
                processed += 1
                if candidate is None:
                    continue

                label_lines.extend(candidate["label_lines"])
                mask_lines.extend(candidate["mask_lines"])
                group_objs = candidate["group_objs"]
                group = make_group_patch(out_img, group_objs, H, W)
                if group is not None:
                    groups.append(group)
                success += 1
            return success

        # Cluster-only mode: every object in the selected frame is a paste target.
        # The same cluster_count is used for all targets in this frame.
        cluster_groups: List[dict] = []
        cluster_targets = list(ref_items)
        rng.shuffle(cluster_targets)
        success_cluster = build_cluster_groups(cluster_count, cluster_targets, cluster_groups)

        if len(label_lines) == len(base_label_lines):
            return 0, 0, 0

        img_name = output_stem_for_frame_set(int(fid), int(repeat_index)) + ".png"
        out_img = apply_animal_noise(
            out_img, mask_lines, _CLUSTER_NOISE_BACKGROUND, cfg,
            "paste_blobs_clustered", fid, repeat_index,
        )
        cv2.imwrite(os.path.join(dirs["images"], img_name), out_img)
        with open(os.path.join(dirs["labels"], img_name.replace(".png", ".txt")), "w", encoding="utf-8") as f:
            f.writelines(label_lines)
        with open(os.path.join(dirs["masks"], img_name.replace(".png", "_maskinfo.txt")), "w", encoding="utf-8") as f:
            f.writelines(mask_lines)

        preview_written = 0
        if (int(fid), int(repeat_index)) in preview_ids:
            preview = draw_preview_from_label_lines(out_img, label_lines, W, H)
            cv2.imwrite(os.path.join(dirs["preview"], img_name), preview)
            preview_written = 1

        return 1, len(label_lines), preview_written



    frames_written = 0
    total_boxes = 0
    preview_written = 0
    for i in range(max(0, int(set_count))):
        fw, tb, pv = _generate_one_set(rep_offset + i)
        frames_written += fw
        total_boxes += tb
        preview_written += pv
    return frames_written, total_boxes, preview_written

def _validate_config_values(cfg: dict) -> None:
    cluster_count = int(cfg.get("CLUSTER_COUNT", 12))
    if cluster_count < 2:
        raise ValueError("CLUSTER_COUNT must be at least 2.")
    num_frames = int(cfg.get("CLUSTER_FRAMES", -1))
    if num_frames < -1 or num_frames == 0:
        raise ValueError("CLUSTER_FRAMES must be -1 for all frames or a positive integer.")
    fit_scale_long = float(cfg.get("CLUSTER_FIT_LONG", 0.8))
    fit_scale_short = float(cfg.get("CLUSTER_FIT_SHORT", 0.8))
    if fit_scale_long <= 0.0 or fit_scale_long > 1.0:
        raise ValueError("CLUSTER_FIT_LONG must be > 0.0 and <= 1.0.")
    if fit_scale_short <= 0.0 or fit_scale_short > 1.0:
        raise ValueError("CLUSTER_FIT_SHORT must be > 0.0 and <= 1.0.")
    parallel_break_prob = float(cfg.get("CLUSTER_BREAK_PROB", 0.20))
    if parallel_break_prob < 0.0 or parallel_break_prob > 1.0:
        raise ValueError("CLUSTER_BREAK_PROB must be >= 0.0 and <= 1.0.")
    for lo_key, hi_key in (("PASTE_SCALE_MIN", "PASTE_SCALE_MAX"),):
        lo = float(cfg.get(lo_key, 1.0))
        hi = float(cfg.get(hi_key, 1.0))
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError(f"{lo_key}/{hi_key} must be positive numbers.")
        if lo > hi:
            raise ValueError(f"{lo_key} must be less than or equal to {hi_key}.")
    width_lo, width_hi = paste_width_scale_range_from_config(cfg)
    if width_lo <= 0.0 or width_hi <= 0.0:
        raise ValueError("WIDTH_SCALE_MIN/WIDTH_SCALE_MAX must be positive numbers.")
    if width_lo > width_hi:
        raise ValueError("WIDTH_SCALE_MIN must be less than or equal to WIDTH_SCALE_MAX.")
    if float(cfg.get("PASTE_MIN_ASPECT", 1.1)) <= 1.0:
        raise ValueError("PASTE_MIN_ASPECT must be greater than 1.0.")

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python interaction_image_synthesis_clustered.py config.yaml")

    config_path = sys.argv[1]
    cfg = load_config(config_path)
    print(f"[SEED] paste_blobs_clustered master={normalize_seed(cfg.get('RANDOM_SEED', 0))}")
    _validate_config_values(cfg)

    session_path = str(cfg["SESSION_PATH"])
    without_crossing_dir = os.path.join(session_path, "single_animal_images")
    manifest_path = os.path.join(without_crossing_dir, "object_pool", "manifest.csv")
    if not os.path.exists(manifest_path):
        refine_original_manifest = os.path.join(
            session_path, "single_animal_images", "refine", "original", "object_pool", "manifest.csv"
        )
        if os.path.exists(refine_original_manifest):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                f"Found refine/original manifest instead: {refine_original_manifest}. "
                "This means apply_class_label_filtering.py was probably skipped or not completed. "
                "Re-run direction_class_filtering.py or apply_class_label_filtering.py."
            )
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    all_objects, objects_by_frame, _ = read_manifest(manifest_path)
    if not all_objects:
        raise RuntimeError(
            f"No usable donor objects in {manifest_path}. "
            "The manifest is empty or all rows failed geometry/direction validation. "
            "Re-run single-animal image extraction and refinement before clustered paste."
        )

    append_mode = bool(cfg.get("CLUSTER_APPEND", False))
    out_dir = os.path.join(session_path, "paste_blobs_clustered")
    if not append_mode:
        reset_output_dir(out_dir)
    dirs = {k: os.path.join(out_dir, k) for k in ("images", "labels", "masks", "preview")}
    ensure_dirs(dirs.values())

    frame_ids = sorted(objects_by_frame.keys())
    explicit_cluster_frames = cfg.get("CLUSTER_FRAMES")
    if explicit_cluster_frames is not None:
        cluster_num_frames = int(explicit_cluster_frames)
        if cluster_num_frames == 0 or cluster_num_frames < -1:
            raise ValueError(f"CLUSTER_FRAMES must be -1 or a positive integer, got {cluster_num_frames}")
    else:
        # CLI-safe automatic target. GUI usually writes CLUSTER_FRAMES,
        # but direct config execution may not.  This keeps clustered generation aligned
        # with NUM_IMAGES and CLUSTERED_RATIO.
        num_total_images = int(cfg.get("NUM_IMAGES", 0))
        clustered_ratio = float(cfg.get("CLUSTERED_RATIO", 0.05))
        clustered_target = num_total_images - math.ceil(num_total_images * (1.0 - clustered_ratio)) if num_total_images > 0 else 0
        skip_crop = bool(cfg.get("skip_cropping", False))
        num_crops = 0 if skip_crop else int(cfg.get("NUM_CROPS", 2))
        include_full = bool(cfg.get("USE_FULL", True))
        images_per_frame = num_crops + (1 if include_full else 0)
        cluster_num_frames = max(1, math.ceil(clustered_target / images_per_frame)) if clustered_target > 0 and images_per_frame > 0 else -1

    if append_mode:
        additional_jobs = build_additional_frame_jobs(
            frame_ids,
            dirs["images"],
            cluster_num_frames,
            normalize_seed(cfg.get("RANDOM_SEED", 0)),
        )
        paste_jobs = [(fid, 1, repeat_index) for fid, repeat_index in additional_jobs]
        preview_ids = preview_additional_frame_jobs(
            additional_jobs, max(0, int(cfg.get("PREVIEW_INTERVAL", 0))),
        )
        attempted_jobs = set(additional_jobs)
    else:
        paste_jobs = build_even_frame_jobs(
            frame_ids, cluster_num_frames, normalize_seed(cfg.get("RANDOM_SEED", 0)),
        )
        preview_interval = max(0, int(cfg.get("PREVIEW_INTERVAL", 0)))
        preview_ids = preview_frame_set_ids_by_interval(paste_jobs, preview_interval)
        attempted_jobs = set()

    from batch_utils import auto_num_workers as _auto_num_workers
    workers_cfg = resolve_num_workers(cfg, "paste_blobs_clustered", default=None)
    workers = _auto_num_workers("process") if workers_cfg is None else max(1, int(workers_cfg))

    frames_written = 0
    total_boxes = 0
    preview_written = 0

    if workers <= 1:
        _init_worker(config_path, preview_ids)
        for job in tqdm(paste_jobs, desc="Pasting clustered blobs"):
            fw, tb, pv = _process_frame(job)
            frames_written += fw
            total_boxes += tb
            preview_written += pv
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(config_path, preview_ids)) as ex:
            futures = {ex.submit(_process_frame, job): job for job in paste_jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Pasting clustered blobs x{workers}"):
                fw, tb, pv = fut.result()
                frames_written += fw
                total_boxes += tb
                preview_written += pv


    # Add fresh candidate-frame sets until the requested supplement is satisfied.
    if cluster_num_frames > 0 and frames_written < cluster_num_frames and frame_ids:
        needed = cluster_num_frames - frames_written
        if workers > 1:
            _init_worker(config_path, [])
        max_retries = needed * max(10, len(frame_ids))
        retry_idx = 0
        with tqdm(total=needed, desc="Adding clustered blobs") as retry_progress:
            while frames_written < cluster_num_frames and retry_idx < max_retries:
                if append_mode:
                    retry_target = min(
                        cluster_num_frames - frames_written,
                        max_retries - retry_idx,
                    )
                    retry_jobs = build_additional_frame_jobs(
                        frame_ids,
                        dirs["images"],
                        retry_target,
                        normalize_seed(cfg.get("RANDOM_SEED", 0)),
                        attempted_jobs=attempted_jobs,
                    )
                    if not retry_jobs:
                        break
                    for fid, repeat_index in retry_jobs:
                        fw, tb, pv = _process_frame((fid, 1, repeat_index))
                        frames_written += fw
                        total_boxes += tb
                        preview_written += pv
                        attempted_jobs.add((fid, repeat_index))
                        retry_idx += 1
                        retry_progress.update(fw)
                        if frames_written >= cluster_num_frames or retry_idx >= max_retries:
                            break
                else:
                    fid = frame_ids[retry_idx % len(frame_ids)]
                    rep_offset = cluster_num_frames + retry_idx
                    fw, tb, pv = _process_frame((fid, 1, rep_offset))
                    frames_written += fw
                    total_boxes += tb
                    preview_written += pv
                    retry_idx += 1
                    retry_progress.update(fw)
        if frames_written < cluster_num_frames:
            print(
                f"paste_blobs_clustered: WARNING - only {frames_written}/{cluster_num_frames} frame sets written "
                f"after {retry_idx} retries; dataset may be smaller than NUM_IMAGES."
            )

if __name__ == "__main__":
    main()
