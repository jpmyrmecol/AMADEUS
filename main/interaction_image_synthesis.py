# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Synthesize mixed interaction scenes from isolated animal images."""

import csv
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from animal_noise import apply_animal_noise, load_noise_background
from batch_utils import auto_num_workers as _auto_num_workers, tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from direction_class_assignment import (
    DEFAULT_MASK_EXPANSION_RATIO,
    expand_contour_from_centroid,
    get_mask_expansion_ratio,
)
from gui.color import OBB_COLOR
from path_utils import resolve_config_paths
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


def occlusion_state_from_mode(mode: str) -> int:
    mode_norm = str(mode).strip().lower()
    return {
        "none": 0,
        "over": 1,
        "under": 2,
    }[mode_norm]


class ImageMaskCache:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._img_cache: Dict[str, Optional[np.ndarray]] = {}
        self._mask_cache: Dict[str, Optional[np.ndarray]] = {}

    def image(self, rel_path: str) -> Optional[np.ndarray]:
        img = self._img_cache.get(rel_path)
        if img is None and rel_path not in self._img_cache:
            img = cv2.imread(os.path.join(self.root_dir, rel_path), cv2.IMREAD_COLOR)
            self._img_cache[rel_path] = img
        return img

    def mask(self, rel_path: str) -> Optional[np.ndarray]:
        mask = self._mask_cache.get(rel_path)
        if mask is None and rel_path not in self._mask_cache:
            mask = cv2.imread(os.path.join(self.root_dir, rel_path), cv2.IMREAD_GRAYSCALE)
            self._mask_cache[rel_path] = mask
        return mask


class RNGPool:
    def __init__(self, seed: int = 0):
        self.seed = normalize_seed(seed)

    def for_frame_dataset(self, frame_id: int, num_pastes: int, repeat_index: int = 0) -> random.Random:
        return make_python_rng(
            self.seed,
            "paste_blobs",
            int(frame_id),
            int(num_pastes),
            int(repeat_index),
        )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return resolve_config_paths(cfg)


def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def resolve_num_workers(cfg: dict, section_name: Optional[str] = None, default: Optional[int] = None, cap: Optional[int] = None) -> Optional[int]:
    """Resolve worker count from GUI/config settings.

    Priority:
      1. section-specific NUM_WORKERS / WORKERS / N_WORKERS / workers / num_workers
      2. top-level NUM_WORKERS / WORKERS / N_WORKERS / workers / num_workers
      3. default
    Returns None when no value/default is provided, allowing paste_blobs to use its memory-aware auto mode.
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


# Mask expansion applies only here, at paste time, never to the background-infill
# mask used to erase blobs in direction_class_assignment.py (which also pads
# the donor RGB crop out to MASK_EXPANSION_RATIO so the expansion below always
# has real pixels behind it -- see its _extract_masked_blob_crop). Each paste
# draws its own ratio uniformly from [1.0, MASK_EXPANSION_RATIO], so pasted
# edges vary slightly instead of using one fixed expansion every time.
def expand_mask_raster(mask_u8_255: np.ndarray, ratio: float) -> np.ndarray:
    """Expand a rasterized 0/255 mask outward from each contour's centroid by ratio."""
    if ratio <= 1.0 + 1e-9:
        return mask_u8_255
    cnts = mask_to_polygons(mask_u8_255)
    if not cnts:
        return mask_u8_255
    out = np.zeros_like(mask_u8_255)
    for cnt in cnts:
        expanded = expand_contour_from_centroid(cnt, ratio)
        cv2.fillPoly(out, [expanded.reshape(-1, 1, 2).astype(np.int32)], 255)
    return out


def random_mask_expansion_ratio(rng: random.Random, mask_expansion_ratio: float) -> float:
    hi = max(1.0, float(mask_expansion_ratio))
    return rng.uniform(1.0, hi)


def donor_image_and_mask(
    donor: dict,
    base_img: np.ndarray,
    cache: "ImageMaskCache",
    scope: str,
    *,
    rng: random.Random,
    mask_expansion_ratio: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if str(scope).strip().lower() == "same_frame":
        x, y, w, h = donor["rect"]
        if int(w) <= 0 or int(h) <= 0:
            return None, None
        crop = base_img[int(y):int(y) + int(h), int(x):int(x) + int(w)]
        mask = donor.get("mask")
        if crop.size == 0 or mask is None:
            return None, None
    else:
        crop = cache.image(donor["crop_image"])
        mask = cache.mask(donor["crop_mask"])
    if mask is not None:
        mask = expand_mask_raster(mask, random_mask_expansion_ratio(rng, mask_expansion_ratio))
    return crop, mask


def mask_to_obb_points(mask_u8_255: np.ndarray, x0: int = 0, y0: int = 0) -> np.ndarray:
    cnts = mask_to_polygons(mask_u8_255)
    if not cnts:
        raise ValueError("Mask does not contain a valid polygon for OBB conversion.")
    pts = np.concatenate([cnt.astype(np.float32) for cnt in cnts], axis=0).reshape(-1, 2)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] += float(x0)
    box[:, 1] += float(y0)
    return box


def yolo_line_from_obb_points(class_id: int, pts: np.ndarray, W: int, H: int) -> str:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0] / float(W), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(H), 0.0, 1.0)
    flat = " ".join(f"{float(v):.6f}" for v in pts.reshape(-1))
    return f"{int(class_id)} {flat}\n"


def draw_obb(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    poly = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    poly = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, color, thickness, cv2.LINE_AA)


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


def class_id_to_unit_vec(class_id: int) -> np.ndarray:
    angle_deg = float(int(class_id) * 45)
    th = math.radians(angle_deg % 360.0)
    return np.array([math.sin(th), -math.cos(th)], dtype=np.float32)


def get_short_edge_candidates(pts: np.ndarray, rel_tol: float = 0.05) -> List[Tuple[int, int, float]]:
    pts = ensure_clockwise(pts)
    edges: List[Tuple[int, int, float]] = []
    lengths: List[float] = []
    for i in range(4):
        j = (i + 1) % 4
        edge_len = float(np.linalg.norm(pts[j] - pts[i]))
        edges.append((i, j, edge_len))
        lengths.append(edge_len)
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


def draw_direction_triangle_for_obb(
    img: np.ndarray,
    pts: np.ndarray,
    class_id: int,
    color: Tuple[int, int, int],
    alpha: float = 0.6,
    outline_thickness: int = 1,
    scale: float = 1.2,
) -> None:
    pts = ensure_clockwise(pts)
    dir_vec = class_id_to_unit_vec(int(class_id))
    short_edges = get_short_edge_candidates(pts)
    if not short_edges:
        raise ValueError("Cannot draw direction triangle: OBB has no valid short edge.")

    best = None
    for i, j, edge_len in short_edges:
        normal = outward_normal_for_edge(pts, i, j)
        score = float(np.dot(normal, dir_vec))
        cand = (score, i, j, edge_len, normal)
        if best is None or score > best[0]:
            best = cand
    if best is None:
        raise ValueError("Cannot draw direction triangle: direction edge selection failed.")

    _, i, j, base_len, normal = best
    if base_len <= 0.0:
        raise ValueError("Cannot draw direction triangle: base edge length is zero.")

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


def parse_yolo_obb_label_lines(label_lines: List[str], W: int, H: int) -> List[Tuple[int, np.ndarray]]:
    out: List[Tuple[int, np.ndarray]] = []
    for line_index, line in enumerate(label_lines):
        parts = line.strip().split()
        if len(parts) != 9:
            raise ValueError(f"Invalid OBB label at line {line_index + 1}: expected 9 fields, got {len(parts)}")
        class_id = int(float(parts[0]))
        pts = np.asarray([float(x) for x in parts[1:]], dtype=np.float32).reshape(4, 2)
        pts[:, 0] *= float(W)
        pts[:, 1] *= float(H)
        out.append((class_id, pts))
    return out


def draw_preview_from_label_lines(img: np.ndarray, label_lines: List[str], W: int, H: int) -> np.ndarray:
    preview = img.copy()
    for class_id, pts in parse_yolo_obb_label_lines(label_lines, W, H):
        draw_obb(preview, pts, OBB_COLOR, 2)
        draw_direction_triangle_for_obb(preview, pts, class_id, OBB_COLOR, alpha=0.6, outline_thickness=1, scale=1.2)
    return preview


def _odd_ksize(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


def choose_under(rng: random.Random, paste_layer_mode: str, under_prob: float) -> bool:
    mode = str(paste_layer_mode).strip().lower()
    if mode == "under":
        return True
    if mode == "over":
        return False
    return rng.random() < float(under_prob)


def build_feather_alpha(mask_u8_255: np.ndarray, *, alpha_mode: str, edge_blur_ksize: int, edge_blur_sigma: float, edge_feather_px: int) -> np.ndarray:
    mode = str(alpha_mode).strip().lower()
    if mode == "distance":
        feather = int(edge_feather_px)
        if feather <= 0:
            return mask_u8_255.astype(np.float32) / 255.0
        bin01 = (mask_u8_255 > 0).astype(np.uint8)
        dist = cv2.distanceTransform(bin01, cv2.DIST_L2, 3).astype(np.float32)
        alpha = np.clip(dist / float(feather), 0.0, 1.0)
        alpha *= bin01.astype(np.float32)
        return alpha.astype(np.float32)
    m = mask_u8_255.astype(np.float32) / 255.0
    a = cv2.GaussianBlur(m, (_odd_ksize(edge_blur_ksize), _odd_ksize(edge_blur_ksize)), float(edge_blur_sigma))
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def apply_brightness_contrast(img_bgr: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    if abs(float(brightness) - 1.0) < 1e-7 and abs(float(contrast) - 1.0) < 1e-7:
        return img_bgr
    out = img_bgr.astype(np.float32) * float(brightness) * float(contrast)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def unit_vec(x: float, y: float) -> Optional[np.ndarray]:
    n = math.hypot(float(x), float(y))
    if n < 1e-8:
        return None
    return np.asarray([float(x) / n, float(y) / n], dtype=np.float32)


def rotate_crop_with_mask(part_bgr: np.ndarray, mask_u8_255: np.ndarray, angle_deg: float, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask_u8_255.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    scale = max(1e-6, float(scale))
    M = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), scale)
    cos = abs(float(M[0, 0]))
    sin = abs(float(M[0, 1]))
    new_w = int(math.ceil(h * sin + w * cos))
    new_h = int(math.ceil(h * cos + w * sin))
    M[0, 2] += (new_w / 2.0 - cx)
    M[1, 2] += (new_h / 2.0 - cy)
    rot_img = cv2.warpAffine(part_bgr, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rot_mask = cv2.warpAffine(mask_u8_255, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rot_img, rot_mask, M.astype(np.float32)


def _mask_obb_frame(mask_u8_255: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    cnts = mask_to_polygons(mask_u8_255)
    if not cnts:
        return None
    pts_all = np.concatenate([cnt.astype(np.float32) for cnt in cnts], axis=0).reshape(-1, 2)
    rect = cv2.minAreaRect(pts_all)
    box = ensure_clockwise(cv2.boxPoints(rect).astype(np.float32))
    center = obb_center(box).astype(np.float32)

    edge_vecs = []
    edge_lengths = []
    for i in range(4):
        vec = (box[(i + 1) % 4] - box[i]).astype(np.float32)
        length = float(np.linalg.norm(vec))
        edge_vecs.append(vec)
        edge_lengths.append(length)
    long_idx = int(np.argmax(edge_lengths))
    long_len = float(edge_lengths[long_idx])
    short_len = float(min(edge_lengths))
    if long_len <= 1e-6 or short_len <= 1e-6:
        return None
    long_axis = (edge_vecs[long_idx] / long_len).astype(np.float32)
    short_axis = np.asarray([-float(long_axis[1]), float(long_axis[0])], dtype=np.float32)
    return center, long_axis, short_axis, long_len, short_len


def obb_aspect_ratio_from_mask(mask_u8_255: np.ndarray) -> Optional[float]:
    frame = _mask_obb_frame(mask_u8_255)
    if frame is None:
        return None
    _, _, _, long_len, short_len = frame
    if short_len <= 1e-6:
        return None
    return float(long_len) / float(short_len)


def valid_obb_aspect_ratio(mask_u8_255: np.ndarray, min_aspect_ratio: float) -> bool:
    aspect = obb_aspect_ratio_from_mask(mask_u8_255)
    return aspect is not None and float(aspect) >= float(min_aspect_ratio)


def sample_paste_width_scales(
    rng: random.Random,
    width_scale_min: float,
    width_scale_max: float,
) -> Tuple[float, float]:
    """Return variable OBB long-axis scale and fixed short-axis scale."""
    lo = max(1e-6, float(width_scale_min))
    hi = max(1e-6, float(width_scale_max))
    if lo > hi:
        lo, hi = hi, lo
    return float(rng.uniform(lo, hi)), 1.0


def paste_width_scale_range_from_config(
    cfg: dict,
    default_min: float = 0.9,
    default_max: float = 1.1,
) -> Tuple[float, float]:
    return (
        float(cfg.get("WIDTH_SCALE_MIN", default_min)),
        float(cfg.get("WIDTH_SCALE_MAX", default_max)),
    )


def rotate_crop_with_mask_and_obb_scaling(
    part_bgr: np.ndarray,
    mask_u8_255: np.ndarray,
    angle_deg: float,
    scale: float,
    width_scale: float,
    height_scale: float,
    min_aspect_ratio: float = 1.1,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    frame = _mask_obb_frame(mask_u8_255)
    if frame is None:
        return None
    center, long_axis, short_axis, long_len, short_len = frame
    scale = max(1e-6, float(scale))
    width_scale = max(1e-6, float(width_scale))
    height_scale = max(1e-6, float(height_scale))

    scaled_long_len = float(long_len) * float(width_scale)
    scaled_short_len = float(short_len) * float(height_scale)
    if scaled_long_len <= scaled_short_len:
        return None
    if scaled_short_len <= 1e-6 or scaled_long_len / scaled_short_len < float(min_aspect_ratio):
        return None

    theta = math.radians(float(angle_deg))
    rot = np.asarray(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]],
        dtype=np.float32,
    )
    u = long_axis.reshape(2, 1).astype(np.float32)
    v = short_axis.reshape(2, 1).astype(np.float32)
    stretch = float(width_scale) * (u @ u.T) + float(height_scale) * (v @ v.T)
    A = (float(scale) * (rot @ stretch)).astype(np.float32)

    h, w = mask_u8_255.shape[:2]
    corners = np.asarray(
        [[0.0, 0.0], [float(w), 0.0], [0.0, float(h)], [float(w), float(h)]],
        dtype=np.float32,
    )
    transformed = (corners - center.reshape(1, 2)) @ A.T + center.reshape(1, 2)
    min_xy = np.floor(np.min(transformed, axis=0)).astype(np.float32)
    max_xy = np.ceil(np.max(transformed, axis=0)).astype(np.float32)
    new_w = max(1, int(max_xy[0] - min_xy[0]))
    new_h = max(1, int(max_xy[1] - min_xy[1]))
    shift = -min_xy
    b = center + shift - A @ center
    M = np.asarray([[A[0, 0], A[0, 1], b[0]], [A[1, 0], A[1, 1], b[1]]], dtype=np.float32)

    out_img = cv2.warpAffine(
        part_bgr, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    out_mask = cv2.warpAffine(
        mask_u8_255, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    if cv2.countNonZero(out_mask) <= 0 or not valid_obb_aspect_ratio(out_mask, float(min_aspect_ratio)):
        return None
    return out_img, out_mask, M


def angle_to_class_id(angle_deg: float) -> int:
    a = float(angle_deg) % 360.0
    return int(((a + 22.5) % 360.0) // 45.0)


def head_angle_deg_from_axis(axis: Tuple[float, float]) -> float:
    dx, dy = float(axis[0]), float(axis[1])
    return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0


def mask_to_polygons(mask_u8_255: np.ndarray) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(mask_u8_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[np.ndarray] = []
    for cnt in cnts:
        cnt = np.asarray(cnt, np.int32)
        if cnt.ndim == 3 and cnt.shape[1] == 1:
            cnt = cnt[:, 0, :]
        if cnt.shape[0] >= 3:
            out.append(cnt)
    return out


def polygon_line_from_shifted_mask(
    mask_u8_255: np.ndarray,
    x0: int,
    y0: int,
    class_id: int,
    class_name: str,
    direction_vec: Optional[Tuple[float, float]] = None,
    center: Optional[Tuple[float, float]] = None,
    axis_length: Optional[float] = None,
    occlusion_state: Optional[int] = None,
    is_pasted: Optional[int] = None,
    overlap_pixels: Optional[int] = None,
    overlap_ratio: Optional[float] = None,
) -> str:
    polys = mask_to_polygons(mask_u8_255)
    parts: List[str] = [f"class_id={int(class_id)}", f"class_name={class_name}"]
    if direction_vec is not None:
        parts.append(f"direction_dx={float(direction_vec[0]):.6f}")
        parts.append(f"direction_dy={float(direction_vec[1]):.6f}")
    if center is not None:
        parts.append(f"center_x={float(center[0]):.6f}")
        parts.append(f"center_y={float(center[1]):.6f}")
    if axis_length is not None:
        parts.append(f"axis_length={float(axis_length):.6f}")
    if occlusion_state is not None:
        parts.append(f"occlusion_state={int(occlusion_state)}")
    if is_pasted is not None:
        parts.append(f"is_pasted={int(is_pasted)}")
    if overlap_pixels is not None:
        parts.append(f"overlap_pixels={int(overlap_pixels)}")
    if overlap_ratio is not None:
        parts.append(f"overlap_ratio={float(overlap_ratio):.6f}")
    poly_chunks: List[str] = []
    for poly in polys:
        pts = " ".join(f"{int(x + x0)},{int(y + y0)}" for x, y in poly)
        poly_chunks.append(f"points={pts}")
    return " | ".join([" ".join(parts)] + poly_chunks) + "\n"


def _parse_int_or_none(value: str) -> Optional[int]:
    s = str(value).strip()
    if s == "":
        return None
    return int(s)


def _parse_float_or_none(value: str) -> Optional[float]:
    s = str(value).strip()
    if s == "":
        return None
    return float(s)


def has_valid_direction_info(item: dict) -> bool:
    class_id = item.get("class_id")
    class_name = str(item.get("class_name", "")).strip()
    head_angle_deg = item.get("head_angle_deg")
    direction_dx = item.get("direction_dx")
    direction_dy = item.get("direction_dy")
    center_x = item.get("center_x")
    center_y = item.get("center_y")
    axis_length = item.get("axis_length")

    if class_id is None or not (0 <= int(class_id) < len(DIR_NAMES)):
        return False
    if class_name != DIR_NAMES[int(class_id)]:
        return False
    if direction_dx is None or direction_dy is None:
        return False
    if not np.isfinite(float(direction_dx)) or not np.isfinite(float(direction_dy)):
        return False
    direction_norm = math.hypot(float(direction_dx), float(direction_dy))
    if not np.isfinite(direction_norm) or not math.isclose(direction_norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        return False
    if int(class_id) != angle_to_class_id(head_angle_deg_from_axis((float(direction_dx), float(direction_dy)))):
        return False
    if head_angle_deg is not None and not np.isfinite(float(head_angle_deg)):
        return False
    if center_x is None or center_y is None:
        return False
    if not np.isfinite(float(center_x)) or not np.isfinite(float(center_y)):
        return False
    if axis_length is None or not np.isfinite(float(axis_length)) or float(axis_length) <= 0.0:
        return False
    return True


def read_manifest(path: str) -> Tuple[List[dict], Dict[int, List[dict]], Dict[str, List[dict]]]:
    rows: List[dict] = []
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    by_pool: Dict[str, List[dict]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            class_id = _parse_int_or_none(row.get("class_id", ""))
            head_angle_deg = _parse_float_or_none(row.get("head_angle_deg", ""))
            direction_dx = _parse_float_or_none(row.get("direction_dx", ""))
            direction_dy = _parse_float_or_none(row.get("direction_dy", ""))
            center_x = _parse_float_or_none(row.get("center_x", ""))
            center_y = _parse_float_or_none(row.get("center_y", ""))
            axis_length = _parse_float_or_none(row.get("axis_length", ""))

            occlusion_state = _parse_int_or_none(row.get("occlusion_state", ""))
            is_pasted = _parse_int_or_none(row.get("is_pasted", ""))
            overlap_pixels = _parse_int_or_none(row.get("overlap_pixels", ""))
            overlap_ratio = _parse_float_or_none(row.get("overlap_ratio", ""))

            item = {
                "pool_id": row["pool_id"],
                "frame": int(row["frame"]),
                "blob_index": int(row["blob_index"]),
                "traj_id": int(row["traj_id"]),
                "x": int(row["rect_x"]),
                "y": int(row["rect_y"]),
                "w": int(row["rect_w"]),
                "h": int(row["rect_h"]),
                "class_id": class_id,
                "class_name": str(row.get("class_name", "")).strip(),
                "head_angle_deg": head_angle_deg,
                "direction_dx": direction_dx,
                "direction_dy": direction_dy,
                "direction_vec": None if None in (direction_dx, direction_dy) else (direction_dx, direction_dy),
                "center_x": center_x,
                "center_y": center_y,
                "axis_length": axis_length,
                "crop_image": row["crop_image"],
                "crop_mask": row["crop_mask"],
                "occlusion_state": 0 if occlusion_state is None else int(occlusion_state),
                "is_pasted": 0 if is_pasted is None else int(is_pasted),
                "overlap_pixels": 0 if overlap_pixels is None else int(overlap_pixels),
                "overlap_ratio": 0.0 if overlap_ratio is None else float(overlap_ratio),
            }
            if not has_valid_direction_info(item):
                continue
            rows.append(item)
            by_frame[item["frame"]].append(item)
            by_pool[item["pool_id"]].append(item)
    return rows, by_frame, by_pool


def intersection_slices(ax: int, ay: int, aw: int, ah: int, bx: int, by: int, bw: int, bh: int):
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix1 >= ix2 or iy1 >= iy2:
        return None
    a_slice = (slice(iy1 - ay, iy2 - ay), slice(ix1 - ax, ix2 - ax))
    b_slice = (slice(iy1 - by, iy2 - by), slice(ix1 - bx, ix2 - bx))
    return a_slice, b_slice


def count_overlap_rect_mask(ax: int, ay: int, a_mask: np.ndarray, bx: int, by: int, b_mask: np.ndarray) -> int:
    ah, aw = a_mask.shape[:2]
    bh, bw = b_mask.shape[:2]
    sl = intersection_slices(ax, ay, aw, ah, bx, by, bw, bh)
    if sl is None:
        return 0
    a_slice, b_slice = sl
    return int(cv2.countNonZero(cv2.bitwise_and(a_mask[a_slice], b_mask[b_slice])))


def build_external_aabb_array(
    external_object_masks: Optional[List[Tuple[int, int, np.ndarray]]],
) -> np.ndarray:
    """Return Nx4 int32 array of [x1, y1, x2, y2] AABBs for external_object_masks.

    Each entry (x, y, mask) contributes [x, y, x+w, y+h] where w = mask.shape[1],
    h = mask.shape[0].  Returns shape (0, 4) for None or empty input.
    Build this once outside any per-try or per-candidate loop.
    """
    if not external_object_masks:
        return np.empty((0, 4), dtype=np.int32)
    rows = []
    for x, y, mask in external_object_masks:
        h, w = mask.shape[:2]
        rows.append([int(x), int(y), int(x) + int(w), int(y) + int(h)])
    return np.array(rows, dtype=np.int32)


def external_aabb_intersecting_indices(
    donor_x1: int, donor_y1: int, donor_x2: int, donor_y2: int,
    ext_aabbs: np.ndarray,
) -> np.ndarray:
    """Return ascending indices into ext_aabbs whose AABB intersects the donor AABB.

    Uses half-open interval test matching intersection_slices semantics:
    overlap iff donor_x1 < ext.x2 and ext.x1 < donor_x2 (and same for y).
    Non-intersecting objects have zero pixel overlap so skipping them is exact.
    """
    if ext_aabbs.shape[0] == 0:
        return np.empty(0, dtype=np.intp)
    ax1, ay1, ax2, ay2 = int(donor_x1), int(donor_y1), int(donor_x2), int(donor_y2)
    intersects = (
        (ax1 < ext_aabbs[:, 2]) &
        (ext_aabbs[:, 0] < ax2) &
        (ay1 < ext_aabbs[:, 3]) &
        (ext_aabbs[:, 1] < ay2)
    )
    return np.nonzero(intersects)[0]


def transform_point(M: np.ndarray, pt: Tuple[float, float]) -> Tuple[float, float]:
    x, y = float(pt[0]), float(pt[1])
    return (
        float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
        float(M[1, 0] * x + M[1, 1] * y + M[1, 2]),
    )


def transform_direction_vec(M: np.ndarray, direction_vec: Tuple[float, float]) -> Optional[np.ndarray]:
    u = unit_vec(float(direction_vec[0]), float(direction_vec[1]))
    if u is None:
        return None
    linear = np.asarray(M, dtype=np.float32)[:, :2]
    v = linear @ np.asarray(u, dtype=np.float32)
    return unit_vec(float(v[0]), float(v[1]))


def direction_scale_factor(M: np.ndarray, direction_vec: Tuple[float, float]) -> Optional[float]:
    u = unit_vec(float(direction_vec[0]), float(direction_vec[1]))
    if u is None:
        return None
    linear = np.asarray(M, dtype=np.float32)[:, :2]
    v = linear @ np.asarray(u, dtype=np.float32)
    scale = float(np.linalg.norm(v))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def transform_heading_info(
    M: np.ndarray,
    *,
    direction_vec: Tuple[float, float],
    center: Tuple[float, float],
    axis_length: float,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> Optional[dict]:
    transformed_direction = transform_direction_vec(M, direction_vec)
    scale = direction_scale_factor(M, direction_vec)
    if transformed_direction is None or scale is None:
        return None
    center_t = transform_point(M, center)
    cx = float(center_t[0]) + float(offset[0])
    cy = float(center_t[1]) + float(offset[1])
    length = float(axis_length) * float(scale)
    if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(length) or length <= 0.0:
        return None
    dx, dy = float(transformed_direction[0]), float(transformed_direction[1])
    class_id = angle_to_class_id(head_angle_deg_from_axis((dx, dy)))
    return {
        "direction_dx": dx,
        "direction_dy": dy,
        "direction_vec": (dx, dy),
        "center_x": cx,
        "center_y": cy,
        "axis_length": length,
        "class_id": class_id,
        "class_name": DIR_NAMES[class_id],
        "head_angle_deg": head_angle_deg_from_axis((dx, dy)),
    }


def try_place_donor(*, rng: random.Random, W: int, H: int, ref_rect: Tuple[int, int, int, int], ref_mask_crop_u8: np.ndarray,
                    donor_mask_crop_u8: np.ndarray, donor_area: int, placed_mask_full_u8: np.ndarray,
                    min_cover: float, max_cover: float, max_tries: int, fallback_to_ref: bool = True, free_anywhere: bool = False,
                    allowed_overlap_full_u8: Optional[np.ndarray] = None,
                    external_object_masks: Optional[List[Tuple[int, int, np.ndarray]]] = None) -> Optional[Tuple[int, int]]:
    xr, yr, wr, hr = ref_rect
    dh, dw = donor_mask_crop_u8.shape[:2]
    if free_anywhere:
        x_low = 0
        x_high = max(0, W - dw)
        y_low = 0
        y_high = max(0, H - dh)
    else:
        x_low = max(0, xr - dw)
        x_high = min(W - dw, xr + wr - 1)
        y_low = max(0, yr - dh)
        y_high = min(H - dh, yr + hr - 1)
    if x_low > x_high or y_low > y_high:
        return (max(0, min(W - dw, xr)), max(0, min(H - dh, yr))) if fallback_to_ref else None
    _ext_aabbs = build_external_aabb_array(external_object_masks)
    for _ in range(int(max_tries)):
        xo = rng.randint(x_low, x_high)
        yo = rng.randint(y_low, y_high)
        # Target overlap must be in [MIN_OVERLAP, MAX_OVERLAP].
        ov = count_overlap_rect_mask(xo, yo, donor_mask_crop_u8, xr, yr, ref_mask_crop_u8)
        target_overlap_ratio = float(ov) / float(donor_area)
        if not (min_cover <= target_overlap_ratio <= max_cover):
            continue
        # External overlap check: fast union precheck, then pairwise if available.
        placed_roi = placed_mask_full_u8[yo:yo + dh, xo:xo + dw]
        placed_overlap = cv2.bitwise_and(donor_mask_crop_u8, placed_roi)
        placed_px = cv2.countNonZero(placed_overlap)
        if placed_px > 0:
            if allowed_overlap_full_u8 is None:
                allowed_px = 0
            else:
                allowed_roi = allowed_overlap_full_u8[yo:yo + dh, xo:xo + dw]
                allowed_px = cv2.countNonZero(cv2.bitwise_and(placed_overlap, allowed_roi))
            external_union_px = max(0, placed_px - allowed_px)
            if external_union_px > 0:
                if external_object_masks is not None:
                    # Pairwise: AABB broad-phase then narrow-phase per intersecting object.
                    _reject = False
                    for _idx in external_aabb_intersecting_indices(xo, yo, xo + dw, yo + dh, _ext_aabbs):
                        _ex, _ey, _emask = external_object_masks[_idx]
                        _ep = count_overlap_rect_mask(xo, yo, donor_mask_crop_u8, _ex, _ey, _emask)
                        if float(_ep) / float(donor_area) > max_cover:
                            _reject = True
                            break
                    if _reject:
                        continue
                else:
                    # Fallback: union-level check.
                    if float(external_union_px) / float(donor_area) > max_cover:
                        continue
        return xo, yo
    return (max(0, min(W - dw, xr)), max(0, min(H - dh, yr))) if fallback_to_ref else None


def paste_one_donor(*, rng: random.Random, out_img_bgr: np.ndarray, placed_mask_full_u8: np.ndarray,
                    donor_part_bgr: np.ndarray, donor_mask_crop_u8: np.ndarray,
                    ref_rect: Tuple[int, int, int, int], ref_mask_crop_u8: np.ndarray,
                    donor_direction_vec: Tuple[float, float],
                    donor_center_local: Tuple[float, float],
                    donor_axis_length: float,
                    min_cover: float, max_cover: float, max_tries: int,
                    paste_layer_mode: str, under_paste_prob: float, alpha_mode: str, edge_feather_px: int,
                    edge_blur_ksize: int, edge_blur_sigma: float, occluder_margin_px: int,
                    paste_scale_min: float = 1.0, paste_scale_max: float = 1.0,
                    paste_width_scale_min: float = 0.95, paste_width_scale_max: float = 1.05,
                    paste_min_obb_aspect_ratio: float = 1.1,
                    fallback_to_ref: bool = True, free_anywhere: bool = False,
                    allowed_overlap_full_u8: Optional[np.ndarray] = None,
                    external_object_masks: Optional[List[Tuple[int, int, np.ndarray]]] = None,
                    paste_brightness_min: float = 1.0, paste_brightness_max: float = 1.0,
                    paste_contrast_min: float = 1.0, paste_contrast_max: float = 1.0,
                    edge_feather_min_px: Optional[int] = None, edge_feather_max_px: Optional[int] = None) -> Optional[dict]:
    lo = float(paste_scale_min)
    hi = float(paste_scale_max)
    if lo > hi:
        lo, hi = hi, lo
    lo = max(1e-6, lo)
    hi = max(1e-6, hi)
    rot_deg = rng.uniform(0.0, 360.0)
    paste_scale = rng.uniform(lo, hi)
    width_scale, height_scale = sample_paste_width_scales(
        rng,
        paste_width_scale_min,
        paste_width_scale_max,
    )
    transformed = rotate_crop_with_mask_and_obb_scaling(
        donor_part_bgr,
        donor_mask_crop_u8,
        rot_deg,
        paste_scale,
        width_scale,
        height_scale,
        paste_min_obb_aspect_ratio,
    )
    if transformed is None:
        return None
    donor_part_bgr, donor_mask_crop_u8, M = transformed
    brightness = rng.uniform(float(paste_brightness_min), float(paste_brightness_max))
    contrast = rng.uniform(float(paste_contrast_min), float(paste_contrast_max))
    donor_part_bgr = apply_brightness_contrast(donor_part_bgr, brightness, contrast)
    if edge_feather_min_px is not None and edge_feather_max_px is not None:
        lo_f = min(int(edge_feather_min_px), int(edge_feather_max_px))
        hi_f = max(int(edge_feather_min_px), int(edge_feather_max_px))
        edge_feather_px = rng.randint(lo_f, hi_f)
    H, W = out_img_bgr.shape[:2]
    donor_area = int(cv2.countNonZero(donor_mask_crop_u8))
    if donor_area <= 0:
        return None

    placed_xy = try_place_donor(
        rng=rng,
        W=W,
        H=H,
        ref_rect=ref_rect,
        ref_mask_crop_u8=ref_mask_crop_u8,
        donor_mask_crop_u8=donor_mask_crop_u8,
        donor_area=donor_area,
        placed_mask_full_u8=placed_mask_full_u8,
        min_cover=min_cover,
        max_cover=max_cover,
        max_tries=max_tries,
        fallback_to_ref=fallback_to_ref,
        free_anywhere=free_anywhere,
        allowed_overlap_full_u8=allowed_overlap_full_u8,
        external_object_masks=external_object_masks,
    )
    if placed_xy is None:
        return None
    xo, yo = placed_xy

    dh, dw = donor_mask_crop_u8.shape[:2]
    under = choose_under(rng, paste_layer_mode, under_paste_prob)
    roi = out_img_bgr[yo:yo + dh, xo:xo + dw]
    cover_pixels = count_overlap_rect_mask(
        xo, yo, donor_mask_crop_u8,
        ref_rect[0], ref_rect[1], ref_mask_crop_u8,
    )
    cover_ratio = float(cover_pixels) / float(max(1, donor_area))

    if under:
        xr, yr, rw, rh = ref_rect
        ref_roi = np.zeros((dh, dw), np.uint8)
        sl = intersection_slices(xo, yo, dw, dh, xr, yr, rw, rh)
        if sl is not None:
            donor_slice, ref_slice = sl
            ref_roi[donor_slice] = ref_mask_crop_u8[ref_slice]
        if occluder_margin_px != 0:
            k = 2 * abs(int(occluder_margin_px)) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            ref_roi = cv2.dilate(ref_roi, kernel, iterations=1) if occluder_margin_px > 0 else cv2.erode(ref_roi, kernel, iterations=1)
        visible_mask = cv2.bitwise_and(donor_mask_crop_u8, cv2.bitwise_not(ref_roi))
        mode_str = "under"
    else:
        visible_mask = donor_mask_crop_u8
        mode_str = "over"

    if cv2.countNonZero(visible_mask) == 0:
        return None
    if not valid_obb_aspect_ratio(visible_mask, paste_min_obb_aspect_ratio):
        return None

    alpha = build_feather_alpha(
        visible_mask,
        alpha_mode=alpha_mode,
        edge_blur_ksize=edge_blur_ksize,
        edge_blur_sigma=edge_blur_sigma,
        edge_feather_px=edge_feather_px,
    )[..., None]
    tmp = roi.copy()
    visible = visible_mask > 0
    tmp[visible] = donor_part_bgr[visible]
    out_img_bgr[yo:yo + dh, xo:xo + dw] = (tmp.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    placed_roi = placed_mask_full_u8[yo:yo + dh, xo:xo + dw]
    cv2.bitwise_or(placed_roi, donor_mask_crop_u8, dst=placed_roi)

    nz = cv2.findNonZero(visible_mask)
    if nz is None:
        return None
    x1, y1, bw_bb, bh_bb = cv2.boundingRect(nz)
    bbox = (xo + x1, yo + y1, bw_bb, bh_bb)

    heading = transform_heading_info(
        M,
        direction_vec=donor_direction_vec,
        center=donor_center_local,
        axis_length=donor_axis_length,
        offset=(float(xo), float(yo)),
    )
    if heading is None:
        return None

    return {
        "bbox": bbox,
        "mode": mode_str,
        "xo": xo,
        "yo": yo,
        "visible_mask": visible_mask.copy(),
        "rot_deg": rot_deg,
        "paste_scale": float(paste_scale),
        "paste_width_scale": float(width_scale),
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
        "overlap_pixels": int(cover_pixels),
        "overlap_ratio": float(cover_ratio),
    }


def build_base_mask_lines(ref_items: List[dict]) -> List[str]:
    base_mask_lines: List[str] = []
    for item in ref_items:
        x, y, w, h = item["rect"]
        base_mask_lines.append(
            polygon_line_from_shifted_mask(
                item["mask"],
                x,
                y,
                item["class_id"],
                item["class_name"],
                direction_vec=item.get("direction_vec"),
                center=(item.get("center_x"), item.get("center_y")),
                axis_length=item.get("axis_length"),
                occlusion_state=int(item.get("occlusion_state", 0)),
                is_pasted=int(item.get("is_pasted", 0)),
                overlap_pixels=int(item.get("overlap_pixels", 0)),
                overlap_ratio=float(item.get("overlap_ratio", 0.0)),
            )
        )
    return base_mask_lines


def localized_crop_origin(
    localize_bbox: Tuple[float, float, float, float],
    crop_size: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int]:
    """Return the deterministic crop origin centered on a localization bbox."""
    crop_size = int(crop_size)
    max_x = int(image_w) - crop_size
    max_y = int(image_h) - crop_size
    if max_x < 0 or max_y < 0:
        raise ValueError("image size must be at least crop_size")

    bx1, by1, bx2, by2 = (float(v) for v in localize_bbox)
    x = int(round((bx1 + bx2) / 2.0 - float(crop_size) / 2.0))
    y = int(round((by1 + by2) / 2.0 - float(crop_size) / 2.0))
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))
    return x, y


def union_bbox_from_ref_items(ref_items: List[dict]) -> Optional[Tuple[float, float, float, float]]:
    """Merge base ref-item OBB AABBs into one full-frame AABB."""
    boxes: List[Tuple[float, float, float, float]] = []
    for item in ref_items:
        pts = item.get("obb_pts")
        if pts is not None:
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            if arr.size > 0:
                boxes.append((
                    float(arr[:, 0].min()),
                    float(arr[:, 1].min()),
                    float(arr[:, 0].max()),
                    float(arr[:, 1].max()),
                ))
                continue
        x, y, w, h = item["rect"]
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def build_frame_items(frame_objects: List[dict], cache: ImageMaskCache, W: int, H: int) -> Tuple[List[dict], List[str], List[str]]:
    ref_items: List[dict] = []
    base_label_lines: List[str] = []

    for obj in frame_objects:
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        mask = cache.mask(obj["crop_mask"])
        if mask is None or mask.shape[:2] != (h, w):
            continue
        obb_pts = mask_to_obb_points(mask, x, y)
        if obb_pts is None:
            continue
        ref_item = {**obj, "mask": mask, "rect": (x, y, w, h)}
        ref_item["obb_pts"] = obb_pts
        ref_item["occlusion_state"] = int(obj.get("occlusion_state", 0) or 0)
        ref_item["is_pasted"] = 0
        ref_item["base_mask_area"] = int(cv2.countNonZero(mask))
        ref_item["overlap_union_mask"] = np.zeros((h, w), dtype=np.uint8)
        ref_item["overlap_pixels"] = int(obj.get("overlap_pixels", 0) or 0)
        ref_item["overlap_ratio"] = float(obj.get("overlap_ratio", 0.0) or 0.0)
        ref_items.append(ref_item)
        base_label_lines.append(yolo_line_from_obb_points(obj["class_id"], obb_pts, W, H))

    base_mask_lines = build_base_mask_lines(ref_items)
    return ref_items, base_label_lines, base_mask_lines


def sample_donors_from_candidates(rng: random.Random, candidates: List[dict], num_pastes: int) -> List[dict]:
    if not candidates:
        return []
    if len(candidates) >= num_pastes:
        return rng.sample(candidates, num_pastes)
    return [rng.choice(candidates) for _ in range(num_pastes)]


# Mixed paste pipeline
def reset_output_dir(out_dir: str) -> None:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    ensure_dirs([out_dir])


def build_even_frame_jobs(frame_ids: List[int], requested_num_sets: int, seed: int) -> List[Tuple[int, int]]:
    """Build (frame_id, set_count) jobs for frame-level paste generation.

    The returned list is intentionally one job per base frame so that each worker
    can reuse the base image, base annotations, frame items, and image/mask cache
    across all repeated paste sets for that frame.

    - requested_num_sets == -1: one set for every available frame.
    - 1 <= requested_num_sets <= available frames: sample that many frames, one set each.
    - requested_num_sets > available frames: use all frames and distribute repeated
      paste sets as evenly as possible.
    """
    ids = [int(x) for x in sorted(frame_ids)]
    if not ids:
        return []

    target = int(requested_num_sets)
    if target == -1:
        return [(fid, 1) for fid in ids]
    if target <= 0:
        raise ValueError("requested_num_sets must be -1 or a positive integer.")

    if target <= len(ids):
        rng = make_python_rng(normalize_seed(seed), "paste_blobs", "build_even_frame_jobs")
        selected = sorted(rng.sample(ids, target))
        return [(fid, 1) for fid in selected]

    n = len(ids)
    jobs: List[Tuple[int, int]] = []
    for i, fid in enumerate(ids):
        set_count = (target * (i + 1)) // n - (target * i) // n
        if set_count > 0:
            jobs.append((fid, int(set_count)))
    allocated = sum(int(count) for _, count in jobs)
    if allocated != target:
        raise RuntimeError(f"Internal paste job allocation error: {allocated} != {target}")
    return jobs


def preview_frame_set_ids_by_interval(frame_jobs: List[Tuple[int, int]], interval: int) -> List[Tuple[int, int]]:
    """One (frame_id, 0) pair per frame selected by actual frame id, at least
    `interval` video frames apart, starting at the first available frame id.

    Selecting by frame id (not by position in `frame_jobs`) matters because that
    list is already thinned by the upstream frame interval; indexing by position
    would compound the two intervals instead of spacing previews by `interval`
    video frames.
    """
    step = max(0, int(interval))
    if step <= 0:
        return []
    out: List[Tuple[int, int]] = []
    last = None
    for fid, set_count in sorted(frame_jobs, key=lambda job: job[0]):
        if int(set_count) <= 0:
            continue
        fid = int(fid)
        if last is None or fid - last >= step:
            out.append((fid, 0))
            last = fid
    return out


def output_stem_for_frame_set(frame_id: int, repeat_index: int) -> str:
    stem = f"frame_{int(frame_id):06d}"
    if int(repeat_index) > 0:
        stem += f"_set_{int(repeat_index):03d}"
    return stem


def paste_mask_into_full(full: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    """Paste mask into full with boundary clipping.

    Duplicated/rotated group-local masks can be partially outside a temporary
    canvas after rounding. OpenCV bitwise operations require identical ROI and
    mask sizes, so both sides are clipped before composing.
    """
    if full is None or mask is None:
        return

    H, W = full.shape[:2]
    h, w = mask.shape[:2]
    if H <= 0 or W <= 0 or h <= 0 or w <= 0:
        return

    x = int(x)
    y = int(y)
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(W, x + w)
    dst_y2 = min(H, y + h)
    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return

    src_x1 = dst_x1 - x
    src_y1 = dst_y1 - y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    roi = full[dst_y1:dst_y2, dst_x1:dst_x2]
    src = mask[src_y1:src_y2, src_x1:src_x2]
    if roi.shape[:2] != src.shape[:2]:
        return
    cv2.bitwise_or(roi, src, dst=roi)


def full_mask_from_local(mask: np.ndarray, x: int, y: int, H: int, W: int) -> np.ndarray:
    out = np.zeros((H, W), np.uint8)
    paste_mask_into_full(out, mask, x, y)
    return out


def build_free_group_patch(
    rng: random.Random,
    donors: List[dict],
    cache: "ImageMaskCache",
    min_cover: float,
    max_cover: float,
    max_tries: int,
    base_img: np.ndarray,
    paste_width_scale_min: float = 0.95,
    paste_width_scale_max: float = 1.05,
    paste_min_obb_aspect_ratio: float = 1.1,
    mask_expansion_ratio: float = 1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[dict]]]:
    """Build a canvas patch with blobs arranged in contact as a chain.

    Returns (patch_bgr, union_mask_u8, local_objects) cropped to the bounding box,
    where each local_object has: mask, x, y, direction_vec, center_x, center_y, axis_length in patch coords.
    Returns None if any donor is unavailable or placement fails.
    """
    if not donors:
        return None

    loaded: List[Tuple[np.ndarray, np.ndarray, dict]] = []
    for d in donors:
        img, mask = donor_image_and_mask(
            d, base_img, cache, "same_frame", rng=rng, mask_expansion_ratio=mask_expansion_ratio,
        )
        if img is None or mask is None:
            return None
        if img.shape[:2] != mask.shape[:2]:
            return None
        width_scale, height_scale = sample_paste_width_scales(
            rng,
            paste_width_scale_min,
            paste_width_scale_max,
        )
        transformed = rotate_crop_with_mask_and_obb_scaling(
            img.copy(),
            mask.copy(),
            0.0,
            1.0,
            width_scale,
            height_scale,
            paste_min_obb_aspect_ratio,
        )
        if transformed is None:
            return None
        img_t, mask_t, M_t = transformed
        center_local = (float(d["center_x"]) - float(d["x"]), float(d["center_y"]) - float(d["y"]))
        heading = transform_heading_info(
            M_t,
            direction_vec=d["direction_vec"],
            center=center_local,
            axis_length=float(d["axis_length"]),
        )
        if heading is None:
            return None
        loaded.append((img_t, mask_t, heading))

    max_side = max(max(img.shape[0], img.shape[1]) for img, _, _ in loaded)
    n = len(loaded)
    canvas_dim = max(int(max_side * (n + 2) * 2), 64)

    canvas_bgr = np.zeros((canvas_dim, canvas_dim, 3), np.uint8)
    placed = np.zeros((canvas_dim, canvas_dim), np.uint8)
    local_objects: List[dict] = []

    img0, mask0, heading0 = loaded[0]
    h0, w0 = mask0.shape[:2]
    cx0 = canvas_dim // 2 - w0 // 2
    cy0 = canvas_dim // 2 - h0 // 2
    roi0 = canvas_bgr[cy0:cy0 + h0, cx0:cx0 + w0]
    canvas_bgr[cy0:cy0 + h0, cx0:cx0 + w0] = np.where(mask0[..., None] > 0, img0, roi0)
    placed[cy0:cy0 + h0, cx0:cx0 + w0] |= mask0
    local_objects.append({
        "mask": mask0.copy(),
        "x": cx0,
        "y": cy0,
        "direction_vec": heading0["direction_vec"],
        "direction_dx": heading0["direction_dx"],
        "direction_dy": heading0["direction_dy"],
        "center_x": heading0["center_x"] + float(cx0),
        "center_y": heading0["center_y"] + float(cy0),
        "axis_length": heading0["axis_length"],
    })

    for img_d, mask_d, heading_d in loaded[1:]:
        hd, wd = mask_d.shape[:2]
        donor_area = int(cv2.countNonZero(mask_d))
        if donor_area <= 0:
            return None

        ref_obj = local_objects[-1]
        ref_rect = (ref_obj["x"], ref_obj["y"], ref_obj["mask"].shape[1], ref_obj["mask"].shape[0])

        xy = try_place_donor(
            rng=rng,
            W=canvas_dim,
            H=canvas_dim,
            ref_rect=ref_rect,
            ref_mask_crop_u8=ref_obj["mask"],
            donor_mask_crop_u8=mask_d,
            donor_area=donor_area,
            placed_mask_full_u8=placed,
            min_cover=min_cover,
            max_cover=max_cover,
            max_tries=max_tries,
            fallback_to_ref=False,
            free_anywhere=False,
            allowed_overlap_full_u8=placed,
        )
        if xy is None:
            return None
        xd, yd = xy

        roi_d = canvas_bgr[yd:yd + hd, xd:xd + wd]
        canvas_bgr[yd:yd + hd, xd:xd + wd] = np.where(mask_d[..., None] > 0, img_d, roi_d)
        placed[yd:yd + hd, xd:xd + wd] |= mask_d
        local_objects.append({
            "mask": mask_d.copy(),
            "x": xd,
            "y": yd,
            "direction_vec": heading_d["direction_vec"],
            "direction_dx": heading_d["direction_dx"],
            "direction_dy": heading_d["direction_dy"],
            "center_x": heading_d["center_x"] + float(xd),
            "center_y": heading_d["center_y"] + float(yd),
            "axis_length": heading_d["axis_length"],
        })

    ys, xs = np.where(placed > 0)
    if xs.size == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    patch = canvas_bgr[y1:y2, x1:x2].copy()
    union = placed[y1:y2, x1:x2].copy()
    for obj in local_objects:
        obj["x"] -= x1
        obj["y"] -= y1
        obj["center_x"] = float(obj["center_x"]) - float(x1)
        obj["center_y"] = float(obj["center_y"]) - float(y1)
    return patch, union, local_objects


def paste_free_group_into_frame(
    *,
    rng: random.Random,
    out_img: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    union_mask: np.ndarray,
    local_objects: List[dict],
    alpha_mode: str,
    edge_feather_px: int,
    edge_blur_ksize: int,
    edge_blur_sigma: float,
    paste_scale_min: float,
    paste_scale_max: float,
    paste_min_obb_aspect_ratio: float,
    max_tries: int,
    label_lines: List[str],
    mask_lines: List[str],
    max_cover: float = 0.0,
    external_object_masks: Optional[List[Tuple[int, int, np.ndarray]]] = None,
    paste_brightness_min: float = 1.0, paste_brightness_max: float = 1.0,
    paste_contrast_min: float = 1.0, paste_contrast_max: float = 1.0,
    edge_feather_min_px: Optional[int] = None, edge_feather_max_px: Optional[int] = None,
    placement_region: Optional[Tuple[int, int, int, int]] = None,
) -> List[Tuple[int, int, np.ndarray]]:
    """Rotate a free group patch and paste it into a free area of the frame.

    Returns per-donor (global_x, global_y, mask_crop) for each successfully annotated
    donor so the caller can extend the external-object list for subsequent placements.
    Returns an empty list on failure.

    Position search: fast group-union precheck, then per-donor pairwise overlap check
    against external_object_masks so no single donor exceeds max_cover against any
    external object.
    """
    H, W = out_img.shape[:2]
    lo = max(1e-6, min(float(paste_scale_min), float(paste_scale_max)))
    hi = max(1e-6, max(float(paste_scale_min), float(paste_scale_max)))

    angle = rng.uniform(0.0, 360.0)
    scale = rng.uniform(lo, hi)
    rot_patch, rot_union, M = rotate_crop_with_mask(patch, union_mask, angle, scale)
    dh, dw = rot_union.shape[:2]
    if dw > W or dh > H:
        return []
    if placement_region is None:
        x_min, x_max = 0, W - dw
        y_min, y_max = 0, H - dh
    else:
        crop_x1, crop_y1, crop_x2, crop_y2 = (int(v) for v in placement_region)
        x_min = max(0, crop_x1)
        x_max = min(W - dw, crop_x2 - dw)
        y_min = max(0, crop_y1)
        y_max = min(H - dh, crop_y2 - dh)
        if x_min > x_max or y_min > y_max:
            return []
    union_area = max(1, cv2.countNonZero(rot_union))

    # Pre-compute per-donor rotated masks in the group output space.  Each entry is
    # (bx, by, crop, donor_area) in rot_union coordinates, or None if the donor
    # vanished after rotation.  Reused for both pairwise checking and annotation.
    donor_rot_crops: List[Optional[Tuple[int, int, np.ndarray, int]]] = []
    for obj in local_objects:
        local_full = np.zeros(union_mask.shape[:2], np.uint8)
        paste_mask_into_full(local_full, obj["mask"], int(obj["x"]), int(obj["y"]))
        rot_obj_mask = cv2.warpAffine(
            local_full, M, (dw, dh),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        nz = cv2.findNonZero(rot_obj_mask)
        if nz is None:
            donor_rot_crops.append(None)
            continue
        bx, by, bw, bh = cv2.boundingRect(nz)
        crop = rot_obj_mask[by:by + bh, bx:bx + bw].copy()
        if not valid_obb_aspect_ratio(crop, paste_min_obb_aspect_ratio):
            return []
        d_area = max(1, cv2.countNonZero(crop))
        donor_rot_crops.append((bx, by, crop, d_area))

    # Find a free position: fast group-union precheck, then per-donor pairwise check.
    placed_at: Optional[Tuple[int, int]] = None
    for _ in range(int(max_tries)):
        xc = rng.randint(x_min, x_max)
        yc = rng.randint(y_min, y_max)
        roi = occupied[yc:yc + dh, xc:xc + dw]
        union_overlap_px = cv2.countNonZero(cv2.bitwise_and(rot_union, roi))
        if float(union_overlap_px) / float(union_area) > float(max_cover):
            continue
        if union_overlap_px > 0 and external_object_masks:
            _reject = False
            for _di, _donor_info in enumerate(donor_rot_crops):
                if _donor_info is None:
                    continue
                _bx, _by, _crop, _d_area = _donor_info
                _gx, _gy = xc + _bx, yc + _by
                for _ex, _ey, _emask in external_object_masks:
                    _ep = count_overlap_rect_mask(_gx, _gy, _crop, _ex, _ey, _emask)
                    if float(_ep) / float(_d_area) > float(max_cover):
                        _reject = True
                        break
                if _reject:
                    break
            if _reject:
                continue
        placed_at = (xc, yc)
        break

    if placed_at is None:
        return []

    xo, yo = placed_at
    brightness = rng.uniform(float(paste_brightness_min), float(paste_brightness_max))
    contrast = rng.uniform(float(paste_contrast_min), float(paste_contrast_max))
    rot_patch = apply_brightness_contrast(rot_patch, brightness, contrast)
    _efp = edge_feather_px
    if edge_feather_min_px is not None and edge_feather_max_px is not None:
        lo_f = min(int(edge_feather_min_px), int(edge_feather_max_px))
        hi_f = max(int(edge_feather_min_px), int(edge_feather_max_px))
        _efp = rng.randint(lo_f, hi_f)
    roi = out_img[yo:yo + dh, xo:xo + dw]
    alpha = build_feather_alpha(
        rot_union,
        alpha_mode=alpha_mode,
        edge_blur_ksize=edge_blur_ksize,
        edge_blur_sigma=edge_blur_sigma,
        edge_feather_px=_efp,
    )[..., None]
    tmp = roi.copy()
    visible = rot_union > 0
    tmp[visible] = rot_patch[visible]
    out_img[yo:yo + dh, xo:xo + dw] = (
        tmp.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    paste_mask_into_full(occupied, rot_union, xo, yo)

    placed_donors: List[Tuple[int, int, np.ndarray]] = []
    for i, obj in enumerate(local_objects):
        donor_info = donor_rot_crops[i] if i < len(donor_rot_crops) else None
        if donor_info is None:
            continue
        bx, by, obj_crop, _ = donor_info

        heading = transform_heading_info(
            M,
            direction_vec=obj["direction_vec"],
            center=(float(obj["center_x"]), float(obj["center_y"])),
            axis_length=float(obj["axis_length"]),
            offset=(float(xo), float(yo)),
        )
        if heading is None:
            continue

        group_obj = {
            "mask": obj_crop,
            "x": xo + bx,
            "y": yo + by,
            "class_id": heading["class_id"],
            "class_name": heading["class_name"],
            "direction_dx": heading["direction_dx"],
            "direction_dy": heading["direction_dy"],
            "direction_vec": heading["direction_vec"],
            "center_x": heading["center_x"],
            "center_y": heading["center_y"],
            "axis_length": heading["axis_length"],
            "occlusion_state": 0,
            "is_pasted": 1,
            "overlap_pixels": 0,
            "overlap_ratio": 0.0,
        }
        append_annotation(group_obj, label_lines, mask_lines, W, H)
        placed_donors.append((xo + bx, yo + by, obj_crop))

    return placed_donors


def append_annotation(obj: dict, label_lines: List[str], mask_lines: List[str], W: int, H: int) -> None:
    class_id = obj.get("class_id")
    class_name = obj.get("class_name")
    if class_id is None or class_name is None:
        raise ValueError("Cannot append annotation: class_id/class_name is missing.")
    obb_pts = mask_to_obb_points(obj["mask"], int(obj["x"]), int(obj["y"]))
    label_lines.append(yolo_line_from_obb_points(int(class_id), obb_pts, W, H))
    mask_lines.append(
        polygon_line_from_shifted_mask(
            obj["mask"], int(obj["x"]), int(obj["y"]), int(class_id), str(class_name),
            direction_vec=obj.get("direction_vec"),
            center=(obj.get("center_x"), obj.get("center_y")),
            axis_length=obj.get("axis_length"),
            occlusion_state=int(obj.get("occlusion_state", 0)),
            is_pasted=int(obj.get("is_pasted", 1)),
            overlap_pixels=int(obj.get("overlap_pixels", 0)),
            overlap_ratio=float(obj.get("overlap_ratio", 0.0)),
        )
    )


def base_object_to_group_obj(item: dict) -> dict:
    x, y, _, _ = item["rect"]
    return {
        "mask": item["mask"],
        "x": int(x),
        "y": int(y),
        "global_x0": int(x),
        "global_y0": int(y),
        "class_id": item.get("class_id"),
        "class_name": item.get("class_name"),
        "direction_dx": item.get("direction_dx"),
        "direction_dy": item.get("direction_dy"),
        "direction_vec": item.get("direction_vec"),
        "center_x": item.get("center_x"),
        "center_y": item.get("center_y"),
        "axis_length": item.get("axis_length"),
        "occlusion_state": int(item.get("occlusion_state", 0)),
        "is_pasted": int(item.get("is_pasted", 0)),
        "overlap_pixels": int(item.get("overlap_pixels", 0)),
        "overlap_ratio": float(item.get("overlap_ratio", 0.0)),
    }


def donor_result_to_group_obj(res: dict) -> dict:
    return {
        "mask": res["visible_mask"],
        "x": int(res["xo"]),
        "y": int(res["yo"]),
        "global_x0": int(res["xo"]),
        "global_y0": int(res["yo"]),
        "class_id": res.get("class_id"),
        "class_name": res.get("class_name"),
        "direction_dx": res.get("direction_dx"),
        "direction_dy": res.get("direction_dy"),
        "direction_vec": res.get("direction_vec"),
        "center_x": res.get("center_x"),
        "center_y": res.get("center_y"),
        "axis_length": res.get("axis_length"),
        "occlusion_state": int(res.get("occlusion_state", 0)),
        "is_pasted": 1,
        "overlap_pixels": int(res.get("overlap_pixels", 0)),
        "overlap_ratio": float(res.get("overlap_ratio", 0.0)),
    }


def allocate_base_composition(n_base: int, ratio_single: float, ratio_p2: float, ratio_p3: float) -> Tuple[int, int, int]:
    """Allocate base blobs into single / p2 target / p3 target counts by normalized ratios.

    The returned counts always sum to n_base.  The single count means base blobs
    that remain as non-contact objects; no paste operation is needed for them.
    """
    n_base = int(n_base)
    ratios = [float(ratio_single), float(ratio_p2), float(ratio_p3)]
    if any(r < 0.0 for r in ratios):
        raise ValueError("RATIO_SINGLE, RATIO_P2, and RATIO_P3 must be zero or positive.")
    total = sum(ratios)
    if total <= 0.0:
        raise ValueError("RATIO_SINGLE + RATIO_P2 + RATIO_P3 must be greater than zero.")
    if n_base <= 0:
        return 0, 0, 0

    quotas = [n_base * r / total for r in ratios]
    counts = [int(math.floor(q)) for q in quotas]
    remainder = n_base - sum(counts)
    order = sorted(range(3), key=lambda i: (quotas[i] - counts[i], ratios[i]), reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    return counts[0], counts[1], counts[2]


def compute_num_free_groups(free_scale: float, num_objects: int) -> int:
    """Round FREE_SCALE * NUM_OBJECTS with standard (not banker's) rounding.

    Python's builtin round() rounds 0.5 to the nearest even integer, so e.g.
    FREE_SCALE=0.25 with NUM_OBJECTS=2 would floor to 0 free groups instead of
    the intended 1. Any positive FREE_SCALE therefore always yields at least
    one free group.
    """
    free_scale = float(free_scale)
    num_objects = int(num_objects)
    if free_scale <= 0.0 or num_objects <= 0:
        return 0
    raw_free_groups = free_scale * num_objects
    return max(1, math.floor(raw_free_groups + 0.5))

# Parallel worker state
_DONOR_SCOPE_LOGGED: bool = False
_LOCALIZED_FREE_PASTE_REGION_LOGGED: bool = False
_PASTE_WORKER_CFG: Optional[dict] = None
_PASTE_WORKER_IMG_DIR: Optional[str] = None
_PASTE_WORKER_DIRS: Optional[Dict[str, str]] = None
_PASTE_WORKER_ALL_OBJECTS: Optional[List[dict]] = None
_PASTE_WORKER_OBJECTS_BY_FRAME: Optional[Dict[int, List[dict]]] = None
_PASTE_WORKER_OBJECTS_BY_POOL: Optional[Dict[str, List[dict]]] = None
_PASTE_WORKER_CACHE: Optional[ImageMaskCache] = None
_PASTE_WORKER_RNG_POOL: Optional[RNGPool] = None
_PASTE_WORKER_PREVIEW_IDS: Optional[set] = None


_PASTE_NOISE_BACKGROUND = None


def _init_paste_worker(config_path: str, preview_jobs: List[Tuple[int, int]]) -> None:
    global _PASTE_WORKER_CFG, _PASTE_WORKER_IMG_DIR, _PASTE_WORKER_DIRS
    global _PASTE_WORKER_ALL_OBJECTS, _PASTE_WORKER_OBJECTS_BY_FRAME
    global _PASTE_WORKER_OBJECTS_BY_POOL, _PASTE_WORKER_CACHE
    global _PASTE_WORKER_RNG_POOL, _PASTE_WORKER_PREVIEW_IDS

    global _PASTE_NOISE_BACKGROUND

    cv2.setNumThreads(1)
    _PASTE_WORKER_CFG = load_config(config_path)
    _PASTE_NOISE_BACKGROUND = load_noise_background(_PASTE_WORKER_CFG)
    session_path = str(_PASTE_WORKER_CFG["SESSION_PATH"])
    without_crossing_dir = os.path.join(session_path, "single_animal_images")
    _PASTE_WORKER_IMG_DIR = os.path.join(without_crossing_dir, "images")
    manifest_path = os.path.join(without_crossing_dir, "object_pool", "manifest.csv")
    _PASTE_WORKER_ALL_OBJECTS, _PASTE_WORKER_OBJECTS_BY_FRAME, _PASTE_WORKER_OBJECTS_BY_POOL = read_manifest(manifest_path)
    if _PASTE_WORKER_CFG.get("VARIABLE_NUM_OBJECTS", False):
        _PASTE_WORKER_CFG["NUM_OBJECTS"] = max(2, max(
            (len(v) for v in _PASTE_WORKER_OBJECTS_BY_FRAME.values()), default=0))
    out_dir = os.path.join(session_path, "paste_blobs")
    _PASTE_WORKER_DIRS = {k: os.path.join(out_dir, k) for k in ("images", "labels", "masks", "preview")}
    _PASTE_WORKER_CACHE = ImageMaskCache(without_crossing_dir)
    _PASTE_WORKER_RNG_POOL = RNGPool(normalize_seed(_PASTE_WORKER_CFG.get("RANDOM_SEED", 0)))
    _PASTE_WORKER_PREVIEW_IDS = {(int(fid), int(rep)) for fid, rep in preview_jobs}


def _process_paste_frame(job: Tuple[int, int]) -> Tuple[int, int, bool]:
    """Process one (frame_id, repeat_index) job. Returns (frames_written, total_boxes, preview_written)."""
    cfg = _PASTE_WORKER_CFG
    img_dir = _PASTE_WORKER_IMG_DIR
    dirs = _PASTE_WORKER_DIRS
    all_objects = _PASTE_WORKER_ALL_OBJECTS
    objects_by_frame = _PASTE_WORKER_OBJECTS_BY_FRAME
    objects_by_pool = _PASTE_WORKER_OBJECTS_BY_POOL
    cache = _PASTE_WORKER_CACHE
    rng_pool = _PASTE_WORKER_RNG_POOL
    preview_ids = _PASTE_WORKER_PREVIEW_IDS or set()

    if cfg is None or img_dir is None or dirs is None or all_objects is None \
            or objects_by_frame is None or objects_by_pool is None \
            or cache is None or rng_pool is None:
        raise RuntimeError("Worker is not initialized.")

    fid, repeat_index = int(job[0]), int(job[1])
    frame_objects = objects_by_frame.get(int(fid), [])
    if not frame_objects:
        return 0, 0, False

    base_img = cv2.imread(os.path.join(img_dir, f"frame_{fid:06d}.png"), cv2.IMREAD_COLOR)
    if base_img is None:
        return 0, 0, False
    H, W = base_img.shape[:2]
    ref_items, base_label_lines, base_mask_lines = build_frame_items(frame_objects, cache, W, H)
    if not ref_items:
        return 0, 0, False

    localized_crop_region: Optional[Tuple[int, int, int, int]] = None
    if bool(cfg.get("LOCALIZED", False)) and not bool(cfg.get("skip_cropping", False)):
        crop_size = int(cfg["TRAIN_IMG_SIZE"])
        if W >= crop_size and H >= crop_size:
            localize_bbox = union_bbox_from_ref_items(ref_items)
            if localize_bbox is not None:
                crop_x1, crop_y1 = localized_crop_origin(localize_bbox, crop_size, W, H)
                localized_crop_region = (
                    crop_x1,
                    crop_y1,
                    crop_x1 + crop_size,
                    crop_y1 + crop_size,
                )
                global _LOCALIZED_FREE_PASTE_REGION_LOGGED
                if not _LOCALIZED_FREE_PASTE_REGION_LOGGED:
                    print(f"[paste_blobs] localized free-paste region enabled: crop_size={crop_size}")
                    _LOCALIZED_FREE_PASTE_REGION_LOGGED = True

    ratio_single = float(cfg.get("RATIO_SINGLE", 0.1))
    ratio_p2 = float(cfg.get("RATIO_P2", 0.5))
    ratio_p3 = float(cfg.get("RATIO_P3", 0.4))
    mask_expansion_ratio = get_mask_expansion_ratio(cfg)
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

    n_base = len(ref_items)
    num_base_single, num_p2, num_p3 = allocate_base_composition(n_base, ratio_single, ratio_p2, ratio_p3)

    out_img = base_img.copy()
    occupied = np.zeros((H, W), np.uint8)
    for item in ref_items:
        x, y, _, _ = item["rect"]
        paste_mask_into_full(occupied, item["mask"], int(x), int(y))

    label_lines = list(base_label_lines)
    mask_lines = list(base_mask_lines)
    rng = rng_pool.for_frame_dataset(fid, num_base_single + num_p2 * 10 + num_p3 * 100, repeat_index=repeat_index)

    frame_donor_candidates = list(ref_items)

    global _DONOR_SCOPE_LOGGED
    if not _DONOR_SCOPE_LOGGED:
        print(f"[paste_blobs] donor scope: same_frame, candidates per frame: {len(frame_donor_candidates)}")
        _DONOR_SCOPE_LOGGED = True

    if not frame_donor_candidates:
        return 0, 0, False

    # Build per-object external mask list for pairwise overlap checking.
    # base_external_objects: one entry per base blob (x, y, mask_crop).
    base_external_objects: List[Tuple[int, int, np.ndarray]] = [
        (int(item["rect"][0]), int(item["rect"][1]), item["mask"]) for item in ref_items
    ]
    # Grows as donors are successfully pasted; used by subsequent placements.
    pasted_donor_masks: List[Tuple[int, int, np.ndarray]] = []

    # Debug counters (per-frame).
    _cnt_contact_placed = 0
    _cnt_contact_failed = 0
    _cnt_free_placed = 0
    _cnt_free_failed = 0

    contact_target_pool = list(ref_items)
    rng.shuffle(contact_target_pool)

    def make_contact_group(target: dict, group_size: int) -> Optional[List[dict]]:
        nonlocal _cnt_contact_placed, _cnt_contact_failed
        k = group_size - 1
        donors = sample_donors_from_candidates(rng, frame_donor_candidates, k)
        if not donors:
            return None
        group_objs = [base_object_to_group_obj(target)]
        target_mask_full = full_mask_from_local(target["mask"], target["rect"][0], target["rect"][1], H, W)
        target_id = id(target)
        # External objects exclude the target itself but include base blobs + pasted donors.
        ext_objs = [
            beo for item, beo in zip(ref_items, base_external_objects) if id(item) != target_id
        ] + list(pasted_donor_masks)
        for donor in donors:
            donor_img, donor_mask = donor_image_and_mask(
                donor, base_img, cache, "same_frame", rng=rng, mask_expansion_ratio=mask_expansion_ratio,
            )
            if donor_img is None or donor_mask is None or donor_img.shape[:2] != donor_mask.shape[:2]:
                _cnt_contact_failed += 1
                continue
            res = paste_one_donor(
                rng=rng,
                out_img_bgr=out_img,
                placed_mask_full_u8=occupied,
                donor_part_bgr=donor_img,
                donor_mask_crop_u8=donor_mask,
                ref_rect=target["rect"],
                ref_mask_crop_u8=target["mask"],
                donor_direction_vec=donor["direction_vec"],
                donor_center_local=(float(donor["center_x"]) - float(donor["x"]), float(donor["center_y"]) - float(donor["y"])),
                donor_axis_length=float(donor["axis_length"]),
                min_cover=min_cover,
                max_cover=max_cover,
                max_tries=max_tries,
                paste_layer_mode=paste_layer_mode,
                under_paste_prob=under_paste_prob,
                alpha_mode=alpha_mode,
                edge_feather_px=edge_feather_px,
                edge_blur_ksize=edge_blur_ksize,
                edge_blur_sigma=edge_blur_sigma,
                occluder_margin_px=occluder_margin_px,
                paste_scale_min=paste_scale_min,
                paste_scale_max=paste_scale_max,
                paste_width_scale_min=paste_width_scale_min,
                paste_width_scale_max=paste_width_scale_max,
                paste_min_obb_aspect_ratio=paste_min_obb_aspect_ratio,
                fallback_to_ref=False,
                free_anywhere=False,
                allowed_overlap_full_u8=target_mask_full,
                external_object_masks=ext_objs,
                paste_brightness_min=paste_brightness_min,
                paste_brightness_max=paste_brightness_max,
                paste_contrast_min=paste_contrast_min,
                paste_contrast_max=paste_contrast_max,
                edge_feather_min_px=edge_feather_min_px,
                edge_feather_max_px=edge_feather_max_px,
            )
            if res is None:
                _cnt_contact_failed += 1
                continue
            if res.get("class_id") is None or res.get("class_name") is None:
                raise ValueError("Pasted contact blob has no direction class. OBB label cannot be generated.")
            obj = donor_result_to_group_obj(res)
            group_objs.append(obj)
            paste_mask_into_full(occupied, obj["mask"], obj["x"], obj["y"])
            pasted_donor_masks.append((int(obj["x"]), int(obj["y"]), obj["mask"]))
            # Update ext_objs so subsequent donors in this group see the newly placed one.
            ext_objs = ext_objs + [(int(obj["x"]), int(obj["y"]), obj["mask"])]
            append_annotation(obj, label_lines, mask_lines, W, H)
            _cnt_contact_placed += 1
        if len(group_objs) != group_size:
            return None
        return group_objs

    for _ in range(max(0, num_p2)):
        if not contact_target_pool:
            break
        target = contact_target_pool.pop()
        make_contact_group(target, 2)

    for _ in range(max(0, num_p3)):
        if not contact_target_pool:
            break
        target = contact_target_pool.pop()
        make_contact_group(target, 3)

    _free_scale = float(cfg.get("FREE_SCALE", 1.0))
    _num_objects_cfg = max(1, int(cfg.get("NUM_OBJECTS", 1)))
    num_free_groups = compute_num_free_groups(_free_scale, _num_objects_cfg)
    if num_free_groups > 0 and frame_donor_candidates:
        free_ratio_single = float(cfg.get("FREE_RATIO_SINGLE", 1.0))
        free_ratio_p2 = float(cfg.get("FREE_RATIO_P2", 1.0))
        free_ratio_p3 = float(cfg.get("FREE_RATIO_P3", 0.5))
        n_free_single, n_free_p2, n_free_p3 = allocate_base_composition(
            num_free_groups, free_ratio_single, free_ratio_p2, free_ratio_p3
        )
        # Current accumulated external objects for free-group placement boundary check.
        _free_ext = base_external_objects + pasted_donor_masks
        _free_kwargs = dict(
            rng=rng,
            out_img=out_img,
            occupied=occupied,
            alpha_mode=alpha_mode,
            edge_feather_px=edge_feather_px,
            edge_blur_ksize=edge_blur_ksize,
            edge_blur_sigma=edge_blur_sigma,
            paste_scale_min=paste_scale_min,
            paste_scale_max=paste_scale_max,
            paste_min_obb_aspect_ratio=paste_min_obb_aspect_ratio,
            max_tries=max_tries,
            label_lines=label_lines,
            mask_lines=mask_lines,
            max_cover=max_cover,
            external_object_masks=_free_ext,
            paste_brightness_min=paste_brightness_min,
            paste_brightness_max=paste_brightness_max,
            paste_contrast_min=paste_contrast_min,
            paste_contrast_max=paste_contrast_max,
            edge_feather_min_px=edge_feather_min_px,
            edge_feather_max_px=edge_feather_max_px,
            placement_region=localized_crop_region,
        )
        for _ in range(n_free_single):
            _result = build_free_group_patch(rng, [rng.choice(frame_donor_candidates)], cache, min_cover, max_cover, max_tries, base_img, paste_width_scale_min, paste_width_scale_max, paste_min_obb_aspect_ratio, mask_expansion_ratio)
            if _result is not None:
                _placed = paste_free_group_into_frame(**_free_kwargs, patch=_result[0], union_mask=_result[1], local_objects=_result[2])
                if _placed:
                    _cnt_free_placed += 1
                    _free_ext.extend(_placed)
                else:
                    _cnt_free_failed += 1
            else:
                _cnt_free_failed += 1
        for _ in range(n_free_p2):
            _d2 = sample_donors_from_candidates(rng, frame_donor_candidates, 2)
            _result = build_free_group_patch(rng, _d2, cache, min_cover, max_cover, max_tries, base_img, paste_width_scale_min, paste_width_scale_max, paste_min_obb_aspect_ratio, mask_expansion_ratio)
            if _result is not None:
                _placed = paste_free_group_into_frame(**_free_kwargs, patch=_result[0], union_mask=_result[1], local_objects=_result[2])
                if _placed:
                    _cnt_free_placed += 1
                    _free_ext.extend(_placed)
                else:
                    _cnt_free_failed += 1
            else:
                _cnt_free_failed += 1
        for _ in range(n_free_p3):
            _d3 = sample_donors_from_candidates(rng, frame_donor_candidates, 3)
            _result = build_free_group_patch(rng, _d3, cache, min_cover, max_cover, max_tries, base_img, paste_width_scale_min, paste_width_scale_max, paste_min_obb_aspect_ratio, mask_expansion_ratio)
            if _result is not None:
                _placed = paste_free_group_into_frame(**_free_kwargs, patch=_result[0], union_mask=_result[1], local_objects=_result[2])
                if _placed:
                    _cnt_free_placed += 1
                    _free_ext.extend(_placed)
                else:
                    _cnt_free_failed += 1
            else:
                _cnt_free_failed += 1

    if len(label_lines) == len(base_label_lines):
        return 0, 0, False

    img_name = output_stem_for_frame_set(fid, repeat_index) + ".png"
    out_img = apply_animal_noise(
        out_img, mask_lines, _PASTE_NOISE_BACKGROUND, cfg,
        "paste_blobs", fid, repeat_index,
    )
    cv2.imwrite(os.path.join(dirs["images"], img_name), out_img)
    with open(os.path.join(dirs["labels"], img_name.replace(".png", ".txt")), "w", encoding="utf-8") as f:
        f.writelines(label_lines)
    with open(os.path.join(dirs["masks"], img_name.replace(".png", "_maskinfo.txt")), "w", encoding="utf-8") as f:
        f.writelines(mask_lines)

    wrote_preview = False
    if (fid, repeat_index) in preview_ids:
        preview = draw_preview_from_label_lines(out_img, label_lines, W, H)
        cv2.imwrite(os.path.join(dirs["preview"], img_name), preview)
        wrote_preview = True

    return 1, len(label_lines), wrote_preview


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python interaction_image_synthesis.py config.yaml")

    config_path = sys.argv[1]
    cfg = load_config(config_path)
    print(f"[SEED] paste_blobs master={normalize_seed(cfg.get('RANDOM_SEED', 0))}")
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
        raise RuntimeError("single_animal_images/object_pool/manifest.csv is empty.")

    if cfg.get("VARIABLE_NUM_OBJECTS", False):
        # Synthesis density only: observed isolated-donor peak is a lower bound,
        # not a conserved animal count and never a tracking limit.
        cfg = dict(cfg)
        cfg["NUM_OBJECTS"] = max(2, max((len(v) for v in objects_by_frame.values()), default=0))
    num_objects = int(cfg.get("NUM_OBJECTS", 1))
    ratio_single = float(cfg.get("RATIO_SINGLE", 0.1))
    ratio_p2 = float(cfg.get("RATIO_P2", 0.5))
    ratio_p3 = float(cfg.get("RATIO_P3", 0.4))

    if num_objects <= 0:
        raise ValueError("NUM_OBJECTS must be a positive integer.")
    if ratio_single < 0.0 or ratio_p2 < 0.0 or ratio_p3 < 0.0:
        raise ValueError("RATIO_SINGLE, RATIO_P2, and RATIO_P3 must be zero or positive.")
    if ratio_single + ratio_p2 + ratio_p3 <= 0.0:
        raise ValueError("RATIO_SINGLE + RATIO_P2 + RATIO_P3 must be greater than zero.")
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

    out_dir = os.path.join(session_path, "paste_blobs")
    reset_output_dir(out_dir)
    dirs = {k: os.path.join(out_dir, k) for k in ("images", "labels", "masks", "preview")}
    ensure_dirs(dirs.values())

    # Compute target frame count from NUM_IMAGES, ratio, crops, and full-frame flag.
    # PASTE_BLOBS_NUM_FRAMES is an explicit override used by create_dataset.py
    # when it needs additional non-clustered candidates.
    explicit_frames = cfg.get("PASTE_BLOBS_NUM_FRAMES")
    num_total_images = int(cfg.get("NUM_IMAGES", 0))
    reserve_step = 0
    if num_total_images > 0 and not bool(cfg.get("skip_creating_direction_dataset", False)):
        reserve_ratio = float(cfg.get("DATASET_SPLIT_RESERVE_RATIO", 0.02))
        reserve_min = int(cfg.get("DATASET_SPLIT_RESERVE_MIN", 100))
        reserve_step = max(reserve_min, math.ceil(num_total_images * reserve_ratio))
    candidate_total = num_total_images + reserve_step
    _nc_ratio = 1.0 - float(cfg.get("CLUSTERED_RATIO", 0.05))
    _nc_target = math.ceil(candidate_total * _nc_ratio) if candidate_total > 0 else 0
    _skip_crop = bool(cfg.get("skip_cropping", False))
    _include_crop = bool(cfg.get("USE_CROP", True))
    _num_crops = 0 if (_skip_crop or not _include_crop) else int(cfg.get("NUM_CROPS", 2))
    _include_full = bool(cfg.get("USE_FULL", True))
    _images_per_frame = _num_crops + (1 if _include_full else 0)
    if explicit_frames is not None:
        requested_num_sets = int(explicit_frames)
        if requested_num_sets == 0 or requested_num_sets < -1:
            raise ValueError(f"PASTE_BLOBS_NUM_FRAMES must be -1 or a positive integer, got {requested_num_sets}")
    elif _nc_target > 0 and _images_per_frame > 0:
        requested_num_sets = max(1, math.ceil(_nc_target / _images_per_frame))
    else:
        requested_num_sets = -1  # all frames once

    preview_interval = int(cfg.get("PREVIEW_INTERVAL", 0))
    frame_ids = sorted(objects_by_frame.keys())
    paste_jobs = build_even_frame_jobs(frame_ids, requested_num_sets, normalize_seed(cfg.get("RANDOM_SEED", 0)))
    preview_jobs = preview_frame_set_ids_by_interval(paste_jobs, preview_interval)

    workers_cfg = resolve_num_workers(cfg, "paste_blobs", default=None)
    workers = _auto_num_workers("process") if workers_cfg is None else max(1, int(workers_cfg))

    print(f"paste_blobs: target={requested_num_sets} frames (nc_target={_nc_target}, images_per_frame={_images_per_frame})")

    frames_written = 0
    total_boxes = 0
    preview_written = 0

    all_jobs = [(fid, rep) for fid, set_count in paste_jobs for rep in range(set_count)]

    if workers <= 1:
        _init_paste_worker(config_path, preview_jobs)
        for job in tqdm(all_jobs, desc="Pasting mixed blobs"):
            fw, tb, pv = _process_paste_frame(job)
            frames_written += fw
            total_boxes += tb
            if pv:
                preview_written += 1
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_paste_worker,
            initargs=(config_path, preview_jobs),
        ) as ex:
            futures = {ex.submit(_process_paste_frame, job): job for job in all_jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Pasting mixed blobs x{workers}"):
                fw, tb, pv = fut.result()
                frames_written += fw
                total_boxes += tb
                if pv:
                    preview_written += 1

    # Retry any shortfall caused by frames where paste produced no new blobs.
    # Each retry job uses a repeat_index >= requested_num_sets, guaranteeing a
    # fresh RNG seed that differs from all jobs in the main pass.
    if requested_num_sets > 0 and frames_written < requested_num_sets:
        needed = requested_num_sets - frames_written
        print(f"paste_blobs: {needed} frame(s) produced no new blobs; retrying with fresh seeds...")
        if workers > 1:
            # Re-initialise globals in the main process for single-threaded retry.
            _init_paste_worker(config_path, preview_jobs)
        max_retries = needed * max(10, len(frame_ids))
        retry_idx = 0
        while frames_written < requested_num_sets and retry_idx < max_retries:
            fid = frame_ids[retry_idx % len(frame_ids)]
            rep = requested_num_sets + retry_idx
            fw, tb, _ = _process_paste_frame((fid, rep))
            frames_written += fw
            total_boxes += tb
            retry_idx += 1
        if frames_written < requested_num_sets:
            print(
                f"paste_blobs: WARNING - only {frames_written}/{requested_num_sets} frames written "
                f"after {retry_idx} retries; dataset will be smaller than NUM_IMAGES."
            )

    print("Done.")
    print(f"paste_blobs: frames={frames_written} total_boxes={total_boxes} previews={preview_written} workers={workers}")


if __name__ == "__main__":
    main()
