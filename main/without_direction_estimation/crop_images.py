# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Crop synthesized scenes and transform their OBB labels."""

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)


import os
import sys
import glob
import math
import yaml
import cv2
import random
import pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from batch_utils import tqdm
from typing import Iterable, List, Tuple, Dict, Optional
from batch_utils import resolve_num_workers as _resolve_num_workers_bt
from gui.color import OBB_COLOR
from path_utils import resolve_config_paths
from random_utils import derive_seed, normalize_seed

# Utilities

def ensure_dirs(paths: Iterable[str]) -> None:
    """Create directories."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_yaml(path: str) -> dict:
    """Load YAML file into a dict."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict):
        return resolve_config_paths(cfg)
    return cfg


def normalize_contour(cnt: np.ndarray) -> np.ndarray:
    """
    Normalize contour array to OpenCV format (N,1,2), dtype int32.
    Accepts (N,2) or (N,1,2).
    """
    cnt = np.asarray(cnt)
    if cnt.ndim == 2 and cnt.shape[1] == 2:
        cnt = cnt[:, None, :]  # (N,2) -> (N,1,2)
    if cnt.ndim != 3 or cnt.shape[1] != 1 or cnt.shape[2] != 2:
        raise ValueError(f"Invalid contour shape: {cnt.shape}")
    return cnt.astype(np.int32, copy=False)

def parse_frame_id_from_stem(stem: str) -> Optional[int]:
    """stem like 'frame_000123' -> 123"""
    if not stem.startswith("frame_"):
        return None
    s = stem[len("frame_"):]
    if not s.isdigit():
        return None
    try:
        return int(s)
    except Exception:
        return None


def select_frame_ids_by_interval(frame_ids: Iterable[int], interval: int) -> set:
    """Frame ids at least `interval` apart (by actual video frame id, not list
    position), walking ascending order starting from the first available id."""
    step = max(0, int(interval))
    if step <= 0:
        return set()
    selected = []
    last = None
    for fid in sorted(set(int(f) for f in frame_ids)):
        if last is None or fid - last >= step:
            selected.append(fid)
            last = fid
    return set(selected)


def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = a_area + b_area - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def bbox_from_contour(cnt: np.ndarray) -> Tuple[float, float, float, float]:
    cnt = normalize_contour(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    return float(x), float(y), float(x + w), float(y + h)


# Data IO


def polygon_to_local_mask(points: List[List[int]], image_h: int, image_w: int) -> Optional[dict]:
    """Create a bbox-local uint8 mask directly from polygon points.

    This avoids allocating one full HxW mask per object. The returned bbox is
    in full-image coordinates as a half-open box (x1, y1, x2, y2), while the
    mask is local to that bbox.
    """
    if len(points) < 3:
        return None
    pts = np.asarray(points, np.int32).reshape(-1, 2)
    x, y, bw, bh = cv2.boundingRect(pts)
    x1 = max(0, min(int(image_w), int(x)))
    y1 = max(0, min(int(image_h), int(y)))
    x2 = max(0, min(int(image_w), int(x + bw)))
    y2 = max(0, min(int(image_h), int(y + bh)))
    if x2 <= x1 or y2 <= y1:
        return None

    local = np.zeros((y2 - y1, x2 - x1), np.uint8)
    pts_local = pts.copy()
    pts_local[:, 0] -= x1
    pts_local[:, 1] -= y1
    cv2.fillPoly(local, [pts_local], 255)
    area = int(cv2.countNonZero(local))
    if area <= 0:
        return None
    return {"mask": local, "bbox": (x1, y1, x2, y2), "area": area}


def merge_local_mask_infos(infos: List[dict]) -> Optional[dict]:
    """Merge multiple bbox-local polygon masks into one bbox-local mask."""
    infos = [info for info in infos if isinstance(info, dict) and int(info.get("area", 0)) > 0]
    if not infos:
        return None
    x1 = min(int(info["bbox"][0]) for info in infos)
    y1 = min(int(info["bbox"][1]) for info in infos)
    x2 = max(int(info["bbox"][2]) for info in infos)
    y2 = max(int(info["bbox"][3]) for info in infos)
    if x2 <= x1 or y2 <= y1:
        return None
    merged = np.zeros((y2 - y1, x2 - x1), np.uint8)
    for info in infos:
        bx1, by1, bx2, by2 = [int(v) for v in info["bbox"]]
        dst = merged[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1]
        cv2.bitwise_or(dst, info["mask"], dst=dst)
    area = int(cv2.countNonZero(merged))
    if area <= 0:
        return None
    return {"mask": merged, "bbox": (x1, y1, x2, y2), "area": area}


def parse_object_maskinfo(mask_info_path: str, h: int, w: int) -> List[dict]:
    objects: List[dict] = []
    with open(mask_info_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            class_id = None
            class_name = None
            occlusion_state = None
            is_pasted = None
            polygon_infos: List[dict] = []
            if not line:
                objects.append({"mask": np.zeros((0, 0), np.uint8), "mask_bbox": (0, 0, 0, 0), "class_id": class_id, "class_name": class_name, "occlusion_state": occlusion_state, "is_pasted": is_pasted})
                continue
            chunks = [chunk.strip() for chunk in line.split("|") if chunk.strip()]
            for chunk in chunks:
                if "class_id=" in chunk:
                    token = chunk.split("class_id=", 1)[1].split()[0]
                    class_id = int(token)
                if "class_name=" in chunk:
                    token = chunk.split("class_name=", 1)[1].split()[0]
                    class_name = token
                if "occlusion_state=" in chunk:
                    token = chunk.split("occlusion_state=", 1)[1].split()[0]
                    occlusion_state = int(float(token))
                if "is_pasted=" in chunk:
                    token = chunk.split("is_pasted=", 1)[1].split()[0]
                    is_pasted = int(float(token))
                if "points=" not in chunk:
                    continue
                pts: List[List[int]] = []
                for token in chunk.split("points=", 1)[1].strip().split():
                    if "," not in token:
                        continue
                    xs, ys = token.split(",", 1)
                    pts.append([int(float(xs)), int(float(ys))])
                if len(pts) >= 3:
                    info = polygon_to_local_mask(pts, h, w)
                    if info is not None:
                        polygon_infos.append(info)
            merged = merge_local_mask_infos(polygon_infos)
            obj_mask = np.zeros((0, 0), np.uint8) if merged is None else merged["mask"]
            mask_bbox = (0, 0, 0, 0) if merged is None else tuple(merged["bbox"])
            objects.append({"mask": obj_mask, "mask_bbox": mask_bbox, "class_id": class_id, "class_name": class_name, "occlusion_state": occlusion_state, "is_pasted": is_pasted})
    return objects


def load_bboxes_yolo(lbl_path: str, w: int, h: int) -> List[dict]:
    """Load YOLO txt in detect or OBB format and return annotation dicts with pixel-space AABB/OBB."""
    boxes: List[dict] = []
    with open(lbl_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            nums = list(map(float, parts[1:]))
            if len(nums) == 4:
                xc, yc, nw, nh = nums
                cx, cy = xc * w, yc * h
                ww, hh = nw * w, nh * h
                x1, y1 = cx - ww / 2.0, cy - hh / 2.0
                x2, y2 = x1 + ww, y1 + hh
                obb_pts = np.asarray(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                )
            elif len(nums) == 8:
                pts = np.asarray(nums, dtype=np.float32).reshape(4, 2)
                pts[:, 0] *= float(w)
                pts[:, 1] *= float(h)
                x1 = float(np.min(pts[:, 0]))
                y1 = float(np.min(pts[:, 1]))
                x2 = float(np.max(pts[:, 0]))
                y2 = float(np.max(pts[:, 1]))
                obb_pts = pts
            else:
                raise ValueError(f"Unsupported YOLO label format in {lbl_path}: {line.strip()}")
            boxes.append({"cls": cls, "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "obb_pts": obb_pts})
    return boxes


def mask_to_obb_points(mask_u8_255: np.ndarray, x0: int = 0, y0: int = 0) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask_u8_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts_list = []
    for cnt in cnts:
        cnt = np.asarray(cnt, np.float32)
        if cnt.ndim == 3 and cnt.shape[1] == 1:
            cnt = cnt[:, 0, :]
        if cnt.shape[0] >= 3:
            pts_list.append(cnt)
    if not pts_list:
        return None
    pts = np.concatenate(pts_list, axis=0).reshape(-1, 2)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] += float(x0)
    box[:, 1] += float(y0)
    return box


def yolo_line_from_obb_points(cls_id: int, pts: np.ndarray, W: int, H: int) -> str:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0] / float(W), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(H), 0.0, 1.0)
    flat = " ".join(f"{float(v):.6f}" for v in pts.reshape(-1))
    return f"{int(cls_id)} {flat}"


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


def obb_coverage_ratio(obb_pts: np.ndarray, crop_x1: float, crop_y1: float, crop_x2: float, crop_y2: float) -> float:
    obb_poly = ensure_clockwise(obb_pts)
    obb_area = float(abs(cv2.contourArea(obb_poly)))
    if obb_area <= 1e-6:
        return 0.0
    crop_poly = ensure_clockwise(np.asarray(
        [
            [float(crop_x1), float(crop_y1)],
            [float(crop_x2), float(crop_y1)],
            [float(crop_x2), float(crop_y2)],
            [float(crop_x1), float(crop_y2)],
        ],
        dtype=np.float32,
    ))
    inter_area, _ = cv2.intersectConvexConvex(obb_poly, crop_poly)
    coverage = float(inter_area) / obb_area
    return max(0.0, min(1.0, coverage))


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


def draw_direction_triangle_for_obb(*args, **kwargs):
    return None


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


# Core ops

def feather_composite(src: np.ndarray, dst: np.ndarray, mask255: np.ndarray,
                      ksize: int, sigma: float) -> np.ndarray:
    """Feather blend src over dst using mask255, with Gaussian boundary smoothing."""
    k = int(ksize)
    if k <= 1:
        k = 1
    if k % 2 == 0:
        k += 1
    alpha = cv2.GaussianBlur(mask255.astype(np.float32) / 255.0, (k, k), float(sigma))[..., None]
    return (src.astype(np.float32) * alpha + dst.astype(np.float32) * (1 - alpha)).astype(np.uint8)


# cache for background resizing to avoid repeated cv2.resize calls
_bg_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

def read_background_path_cached(bg_full: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize background once per (w,h) and cache the result."""
    key = (w, h, id(bg_full))
    hit = _bg_cache.get(key)
    if hit is not None:
        return hit
    if (bg_full.shape[1], bg_full.shape[0]) != (w, h):
        bg = cv2.resize(bg_full, (w, h))
    else:
        bg = bg_full
    _bg_cache[key] = bg
    return bg

def build_blob_matcher(
    blobs_for_frame: Optional[list],
    bboxes: List[dict],
    *,
    iou_threshold: float = 0.30,
) -> Dict[int, np.ndarray]:
    """Return mapping: bbox_index -> contour(np.int32) for best matched blob."""
    if not blobs_for_frame:
        return {}

    blob_boxes: List[Tuple[Tuple[float, float, float, float], np.ndarray]] = []
    for bl in blobs_for_frame:
        cnt = np.asarray(getattr(bl, "contour", None), np.int32)
        if cnt is None or cnt.size == 0:
            continue
        blob_boxes.append((bbox_from_contour(cnt), cnt))

    if not blob_boxes:
        return {}

    out: Dict[int, np.ndarray] = {}
    used_blob = set()

    for bi, ann in enumerate(bboxes):
        x1 = float(ann["x1"])
        y1 = float(ann["y1"])
        x2 = float(ann["x2"])
        y2 = float(ann["y2"])
        best_iou = 0.0
        best_j = None
        for j, (bb, cnt) in enumerate(blob_boxes):
            if j in used_blob:
                continue
            iou = box_iou((x1, y1, x2, y2), bb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None and best_iou >= float(iou_threshold):
            out[int(bi)] = blob_boxes[best_j][1]
            used_blob.add(best_j)

    return out


def is_base_object_info(info: dict) -> bool:
    """Return True only when maskinfo explicitly marks an object as a base blob."""
    if not isinstance(info, dict) or info.get("is_pasted") is None:
        return False
    try:
        return int(float(info.get("is_pasted"))) == 0
    except (TypeError, ValueError):
        return False


def base_bbox_indices(object_info_map: Dict[int, dict]) -> List[int]:
    """Return bbox indices whose maskinfo object is marked is_pasted == 0."""
    return [int(i) for i, info in object_info_map.items() if is_base_object_info(info)]


def union_bbox_from_indices(bboxes: List[dict], indices: List[int]) -> Optional[Tuple[float, float, float, float]]:
    """Merge selected bbox indices into one full-image AABB."""
    valid = [int(i) for i in indices if 0 <= int(i) < len(bboxes)]
    if not valid:
        return None
    x1 = min(float(bboxes[i]["x1"]) for i in valid)
    y1 = min(float(bboxes[i]["y1"]) for i in valid)
    x2 = max(float(bboxes[i]["x2"]) for i in valid)
    y2 = max(float(bboxes[i]["y2"]) for i in valid)
    return x1, y1, x2, y2


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


def fill_contour_on_crop_mask(mask: np.ndarray, cnt_full: np.ndarray, crop_x: int, crop_y: int) -> None:
    """Fill contour (in full-image coords) onto crop-local mask."""
    cnt = normalize_contour(cnt_full).copy()
    cnt[:, 0, 0] -= int(crop_x)
    cnt[:, 0, 1] -= int(crop_y)
    cv2.fillPoly(mask, [cnt], 255)


def fill_localmask_on_crop_mask(mask: np.ndarray, mask_info: dict, crop_x: int, crop_y: int) -> None:
    """Copy a bbox-local object mask into the intersecting crop ROI."""
    if not isinstance(mask_info, dict):
        return
    local_mask = mask_info.get("mask")
    bbox = mask_info.get("bbox")
    if local_mask is None or not isinstance(local_mask, np.ndarray) or local_mask.size == 0 or bbox is None:
        return

    bx1, by1, bx2, by2 = (int(v) for v in bbox)
    crop_h, crop_w = mask.shape[:2]
    ix1 = max(bx1, int(crop_x))
    iy1 = max(by1, int(crop_y))
    ix2 = min(bx2, int(crop_x) + crop_w)
    iy2 = min(by2, int(crop_y) + crop_h)
    if ix2 <= ix1 or iy2 <= iy1:
        return

    src = local_mask[iy1 - by1:iy2 - by1, ix1 - bx1:ix2 - bx1]
    dst = mask[iy1 - int(crop_y):iy2 - int(crop_y), ix1 - int(crop_x):ix2 - int(crop_x)]
    cv2.bitwise_or(dst, src, dst=dst)


def try_random_crops(
    *,
    img: np.ndarray,
    bg_full: np.ndarray,
    bboxes: List[dict],
    mask_full: Optional[np.ndarray],
    crop_size: int,
    num_crops: int,
    num_min_objs: int,
    remove_th: float,
    edge_blur_ksize: int,
    edge_blur_sigma: float,
    blob_contours_by_bbox_index: Optional[Dict[int, np.ndarray]] = None,
    object_masks_by_bbox_index: Optional[Dict[int, dict]] = None,
    object_infos_by_bbox_index: Optional[Dict[int, dict]] = None,
    rng_seed: int = None,
    localize_to_bbox: Optional[Tuple[float, float, float, float]] = None,
):
    """Generator yielding (x,y,crop,label_lines,preview_objects)."""
    h_img, w_img = img.shape[:2]
    bg = read_background_path_cached(bg_full, w_img, h_img)

    saved_xy = set()
    tries = 0
    rng = random.Random(rng_seed)
    max_x = int(w_img - crop_size)
    max_y = int(h_img - crop_size)
    if max_x < 0 or max_y < 0:
        return
    target_num_crops = max(0, int(num_crops))
    localized_xy: Optional[Tuple[int, int]] = None
    if localize_to_bbox is not None:
        localized_xy = localized_crop_origin(localize_to_bbox, crop_size, w_img, h_img)
        target_num_crops = 1 if target_num_crops > 0 else 0
    base_ids: set[int] = set()
    if isinstance(object_infos_by_bbox_index, dict):
        base_ids = {
            int(i) for i, info in object_infos_by_bbox_index.items()
            if is_base_object_info(info)
        }
    require_base = len(base_ids) > 0

    while len(saved_xy) < target_num_crops and tries < 1000:
        tries += 1
        if localized_xy is not None:
            if tries > 1:
                break
            x, y = localized_xy
        else:
            x = rng.randint(0, max_x)
            y = rng.randint(0, max_y)
        if (x, y) in saved_xy:
            continue
        kept, removed = [], []

        # Decide kept/removed based on OBB coverage ratio; keep bbox index for blob lookup.
        for bbox_idx, ann in enumerate(bboxes):
            cls = int(ann["cls"])
            x1 = float(ann["x1"])
            y1 = float(ann["y1"])
            x2 = float(ann["x2"])
            y2 = float(ann["y2"])
            ix1, iy1 = max(x1, x), max(y1, y)
            ix2, iy2 = min(x2, x + crop_size), min(y2, y + crop_size)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            coverage = obb_coverage_ratio(
                np.asarray(ann["obb_pts"], dtype=np.float32).reshape(4, 2),
                float(x),
                float(y),
                float(x + crop_size),
                float(y + crop_size),
            )
            if coverage <= 0.0:
                continue
            target = removed if coverage < remove_th else kept
            target.append((int(cls), int(bbox_idx), float(ix1), float(iy1), float(ix2), float(iy2),
                           float(x1), float(y1), float(x2), float(y2)))

        if len(kept) < num_min_objs:
            continue
        if require_base and not any(int(entry[1]) in base_ids for entry in kept):
            continue

        crop = img[y:y + crop_size, x:x + crop_size].copy()
        bg_patch = bg[y:y + crop_size, x:x + crop_size]

        # Build removed mask for boundary-cut objects, then subtract kept object
        # blob/object masks so a removed object never erases a kept neighbor.
        mask_removed = np.zeros((crop_size, crop_size), np.uint8)

        use_blob = isinstance(blob_contours_by_bbox_index, dict) and len(blob_contours_by_bbox_index) > 0
        use_object_masks = isinstance(object_masks_by_bbox_index, dict) and len(object_masks_by_bbox_index) > 0

        for _, bbox_idx, ix1, iy1, ix2, iy2, _, _, _, _ in removed:
            obj_mask = object_masks_by_bbox_index.get(int(bbox_idx)) if use_object_masks else None
            cnt = blob_contours_by_bbox_index.get(int(bbox_idx)) if use_blob else None
            if isinstance(obj_mask, dict) and isinstance(obj_mask.get("mask"), np.ndarray) and obj_mask["mask"].size > 0:
                fill_localmask_on_crop_mask(mask_removed, obj_mask, crop_x=x, crop_y=y)
            elif cnt is not None and cnt.size > 0:
                fill_contour_on_crop_mask(mask_removed, cnt, crop_x=x, crop_y=y)
            else:
                rx1, ry1 = int(ix1 - x), int(iy1 - y)
                rx2, ry2 = int(ix2 - x), int(iy2 - y)
                cv2.rectangle(mask_removed, (rx1, ry1), (rx2, ry2), 255, -1)

        mask_keep = np.zeros_like(mask_removed)
        for _, bbox_idx, ix1, iy1, ix2, iy2, _, _, _, _ in kept:
            obj_mask = object_masks_by_bbox_index.get(int(bbox_idx)) if use_object_masks else None
            cnt = blob_contours_by_bbox_index.get(int(bbox_idx)) if use_blob else None
            if isinstance(obj_mask, dict) and isinstance(obj_mask.get("mask"), np.ndarray) and obj_mask["mask"].size > 0:
                fill_localmask_on_crop_mask(mask_keep, obj_mask, crop_x=x, crop_y=y)
            elif cnt is not None and cnt.size > 0:
                fill_contour_on_crop_mask(mask_keep, cnt, crop_x=x, crop_y=y)
            else:
                rx1, ry1 = int(ix1 - x), int(iy1 - y)
                rx2, ry2 = int(ix2 - x), int(iy2 - y)
                cv2.rectangle(mask_keep, (rx1, ry1), (rx2, ry2), 255, -1)

        mask_removed = cv2.bitwise_and(mask_removed, cv2.bitwise_not(mask_keep))
        if cv2.countNonZero(mask_removed):
            crop = feather_composite(bg_patch, crop, mask_removed, edge_blur_ksize, edge_blur_sigma)

        # Labels (YOLO; preserve class IDs)
        preview_objects = []
        for cls_id, bbox_idx, ix1, iy1, ix2, iy2, _, _, _, _ in kept:
            rx, ry = ix1 - x, iy1 - y
            rw, rh = ix2 - ix1, iy2 - iy1
            local_obb_pts = None
            obj_mask = object_masks_by_bbox_index.get(int(bbox_idx)) if use_object_masks else None
            if isinstance(obj_mask, dict) and isinstance(obj_mask.get("mask"), np.ndarray) and obj_mask["mask"].size > 0:
                local_mask = np.zeros((crop_size, crop_size), np.uint8)
                fill_localmask_on_crop_mask(local_mask, obj_mask, crop_x=x, crop_y=y)
                local_obb_pts = mask_to_obb_points(local_mask)
            if local_obb_pts is None:
                local_obb_pts = np.asarray(bboxes[int(bbox_idx)]["obb_pts"], dtype=np.float32).reshape(4, 2).copy()
                local_obb_pts[:, 0] -= float(x)
                local_obb_pts[:, 1] -= float(y)
            preview_objects.append({
                "class_id": int(cls_id),
                "bbox": (int(round(rx)), int(round(ry)), int(round(rw)), int(round(rh))),
                "obb_pts": np.asarray(local_obb_pts, dtype=np.float32),
                "bbox_idx": int(bbox_idx),
            })

        if len(preview_objects) < num_min_objs:
            continue
        if require_base and not any(
            obj.get("bbox_idx") is not None and int(obj["bbox_idx"]) in base_ids
            for obj in preview_objects
        ):
            continue

        lines = [
            yolo_line_from_obb_points(obj["class_id"], obj["obb_pts"], crop_size, crop_size)
            for obj in preview_objects
        ]

        saved_xy.add((x, y))
        yield x, y, crop, lines, preview_objects


# Main worker

_REMOVE_THRESHOLD = 0.95

# Parallel worker for one frame
_CROP_WORKER_BG: Optional[np.ndarray] = None


def _init_crop_worker(bg_path: str) -> None:
    global _CROP_WORKER_BG
    _CROP_WORKER_BG = cv2.imread(bg_path)


def _crop_one_frame_worker(task: dict) -> int:
    """Process one frame: generate crops and write image/label/preview files. Returns number of crops written."""
    bg_full = _CROP_WORKER_BG
    if bg_full is None:
        return 0

    img = cv2.imread(str(task["img_path"]))
    if img is None:
        return 0
    h_img, w_img = img.shape[:2]
    crop_size = int(task["CROP_SIZE"])
    if crop_size <= 0:
        raise ValueError(f"CROP_SIZE must be positive: {crop_size}")
    if h_img < crop_size or w_img < crop_size:
        return 0

    lbl_path = str(task["lbl_path"])
    bboxes = load_bboxes_yolo(lbl_path, w_img, h_img)

    object_mask_map: Dict[int, dict] = {}
    object_info_map: Dict[int, dict] = {}
    ompath = str(task["object_maskinfo_path"])
    if os.path.exists(ompath):
        object_infos = parse_object_maskinfo(ompath, h_img, w_img)
        object_info_map = {i: obj for i, obj in enumerate(object_infos[:len(bboxes)])}
        object_mask_map = {
            i: {"mask": obj["mask"], "bbox": obj.get("mask_bbox", (0, 0, 0, 0))}
            for i, obj in object_info_map.items()
        }

    blob_frame_data = task["blob_frame_data"]
    blob_map: Dict[int, np.ndarray] = {}
    if not object_mask_map and blob_frame_data is not None:
        blob_map = build_blob_matcher(blob_frame_data, bboxes, iou_threshold=0.30)

    localize_on_base = bool(task.get("LOCALIZED", False))
    localize_to_bbox = None
    if localize_on_base and object_info_map:
        localize_to_bbox = union_bbox_from_indices(bboxes, base_bbox_indices(object_info_map))

    stem = str(task["stem"])
    IMG_EXT = str(task["IMG_EXT"])
    CROP_SIZE = crop_size
    OUT_IMG_DIR = str(task["OUT_IMG_DIR"])
    OUT_LBL_DIR = str(task["OUT_LBL_DIR"])
    PREV_DIR = str(task["PREV_DIR"])
    preview_budget = int(task["preview_budget"])

    written = 0
    preview_written = 0
    for x, y, crop, lines, _ in try_random_crops(
        img=img,
        bg_full=bg_full,
        bboxes=bboxes,
        mask_full=None,
        crop_size=CROP_SIZE,
        num_crops=int(task["NUM_CROPS"]),
        num_min_objs=1,
        remove_th=_REMOVE_THRESHOLD,
        edge_blur_ksize=int(task["EDGE_BLUR_KSIZE"]),
        edge_blur_sigma=float(task["EDGE_BLUR_SIGMA"]),
        blob_contours_by_bbox_index=blob_map,
        object_masks_by_bbox_index=object_mask_map,
        object_infos_by_bbox_index=object_info_map,
        localize_to_bbox=localize_to_bbox,
        rng_seed=int(task["rng_seed"]),
    ):
        out_img_name = f"{stem}_{y}_{x}{IMG_EXT}"
        cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img_name), crop)
        with open(os.path.join(OUT_LBL_DIR, out_img_name.replace(IMG_EXT, ".txt")), "w") as f:
            f.write("\n".join(lines))
        if preview_written < preview_budget:
            prev = draw_preview_from_label_lines(crop, lines, CROP_SIZE, CROP_SIZE)
            cv2.imwrite(os.path.join(PREV_DIR, out_img_name), prev)
            preview_written += 1
        written += 1
    return written


def main(
    *,
    BASE_DIR_LIST: List[str],
    CROP_SIZE: int,
    IMG_EXT: str,
    PREVIEW_INTERVAL: int,
    NUM_CROPS: int = 1,
    BACKGROUND_PATH: str = "",
    EDGE_BLUR_KSIZE: int = 7,
    EDGE_BLUR_SIGMA: float = 11.0,
    MASK_DIR: str = "",
    BLOBS_IN_VIDEO: Optional[list] = None,
    LOCALIZED: bool = False,
    num_workers: int = 1,
    random_seed: int = 0,
    append: bool = False,
):
    """Run random cropping across base directories with blob-aware background replacement for removed (partial) objects."""

    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError(f"BACKGROUND_PATH not found: {BACKGROUND_PATH}")

    random_seed = normalize_seed(random_seed)

    for BASE_DIR in BASE_DIR_LIST:
        base_key = os.path.basename(os.path.normpath(BASE_DIR))
        IMG_DIR = os.path.join(BASE_DIR, "images")
        LBL_DIR = os.path.join(BASE_DIR, "labels")
        OUT_DIR = os.path.join(BASE_DIR, "cropping")
        OUT_IMG_DIR = os.path.join(OUT_DIR, "images")
        OUT_LBL_DIR = os.path.join(OUT_DIR, "labels")
        PREV_DIR = os.path.join(OUT_DIR, "preview")
        # Full runs rebuild crops; shortfall recovery keeps existing crops and
        # processes only newly appended source frame sets.
        if os.path.isdir(OUT_DIR) and not append:
            shutil.rmtree(OUT_DIR)
        ensure_dirs((OUT_DIR, OUT_IMG_DIR, OUT_LBL_DIR, PREV_DIR))
        cropped_source_stems = set()
        if append:
            for crop_path in glob.glob(os.path.join(OUT_IMG_DIR, f"*{IMG_EXT}")):
                crop_stem = os.path.splitext(os.path.basename(crop_path))[0]
                parts = crop_stem.rsplit("_", 2)
                if len(parts) != 3:
                    continue
                try:
                    int(parts[-2])
                    int(parts[-1])
                except ValueError:
                    continue
                cropped_source_stems.add(parts[0])

        img_paths = sorted(glob.glob(os.path.join(IMG_DIR, f"frame_*{IMG_EXT}")))

        effective_crops_per_img = 1 if LOCALIZED and int(NUM_CROPS) > 0 else NUM_CROPS
        preview_interval = max(0, int(PREVIEW_INTERVAL))

        valid_entries: List[Tuple[str, str, str, Optional[int]]] = []
        for img_path in img_paths:
            fname = os.path.basename(img_path)
            stem = fname[:-len(IMG_EXT)]
            lbl_path = os.path.join(LBL_DIR, stem + ".txt")
            if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
                continue
            if append and stem in cropped_source_stems:
                continue
            valid_entries.append((img_path, stem, lbl_path, parse_frame_id_from_stem(stem)))

        # Preview every PREVIEW_INTERVAL-th *video* frame by actual frame id, not by
        # position in this list (which is already thinned by the upstream frame
        # interval, so indexing by position would compound the two intervals).
        preview_fids = select_frame_ids_by_interval(
            (fid for _, _, _, fid in valid_entries if fid is not None), preview_interval
        )

        tasks: List[dict] = []
        for img_path, stem, lbl_path, fid in valid_entries:
            blob_frame_data = None
            if BLOBS_IN_VIDEO is not None and fid is not None and 0 <= fid < len(BLOBS_IN_VIDEO):
                blob_frame_data = BLOBS_IN_VIDEO[fid]
            crop_seed = derive_seed(random_seed, "crop_images", base_key, stem)
            tasks.append({
                "img_path": img_path,
                "stem": stem,
                "lbl_path": lbl_path,
                "rng_seed": crop_seed,
                "object_maskinfo_path": os.path.join(BASE_DIR, "masks", stem + "_maskinfo.txt"),
                "blob_frame_data": blob_frame_data,
                "OUT_IMG_DIR": OUT_IMG_DIR,
                "OUT_LBL_DIR": OUT_LBL_DIR,
                "PREV_DIR": PREV_DIR,
                "IMG_EXT": IMG_EXT,
                "CROP_SIZE": CROP_SIZE,
                "NUM_CROPS": NUM_CROPS,
                "EDGE_BLUR_KSIZE": EDGE_BLUR_KSIZE,
                "EDGE_BLUR_SIGMA": EDGE_BLUR_SIGMA,
                "LOCALIZED": LOCALIZED,
                "preview_budget": effective_crops_per_img if (fid is not None and fid in preview_fids) else 0,
            })

        actual_workers = max(1, min(num_workers, len(tasks))) if tasks else 1
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            initializer=_init_crop_worker,
            initargs=(BACKGROUND_PATH,),
        ) as ex:
            total = sum(tqdm(
                ex.map(_crop_one_frame_worker, tasks),
                total=len(tasks),
                desc=f"Random cropping {os.path.basename(BASE_DIR)}",
            ))
        print(f"Finished random cropping for {os.path.basename(BASE_DIR)}: {total} crops written")


# CLI
def cli() -> None:
    """Read the CLI configuration and call the parameterized crop implementation."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python crop_images.py config.yaml")

    cfg_path = sys.argv[1]
    cfg = load_yaml(cfg_path)

    SESSION_PATH = cfg["SESSION_PATH"]
    CROP_SIZE = cfg["TRAIN_IMG_SIZE"]
    PREVIEW_INTERVAL = cfg["PREVIEW_INTERVAL"]
    BACKGROUND_PATH = cfg["BACKGROUND_PATH"]
    EDGE_BLUR_KSIZE = cfg.get("EDGE_BLUR_KSIZE", 7)
    EDGE_BLUR_SIGMA = float(cfg.get("EDGE_BLUR_SIGMA", 11))
    NUM_CROPS = cfg.get("NUM_CROPS", 1)
    LOCALIZED = bool(cfg.get("LOCALIZED", False))
    # Blob source (for virtual/single_animal_images frames)
    PICKLE_PATH = cfg.get("PICKLE_PATH", "")
    BLOBS_IN_VIDEO = None
    if isinstance(PICKLE_PATH, str) and PICKLE_PATH and os.path.exists(PICKLE_PATH):
        try:
            with open(PICKLE_PATH, "rb") as f:
                BLOBS_IN_VIDEO = pickle.load(f).blobs_in_video
        except Exception as e:
            print(f"Warning: failed to load PICKLE_PATH for blob masking: {PICKLE_PATH} ({e})")
            BLOBS_IN_VIDEO = None
    else:
        if PICKLE_PATH:
            print(f"Warning: PICKLE_PATH not found, blob masking disabled: {PICKLE_PATH}")

    training_datasets = cfg.get("training_datasets", {})
    use_single_animal_images = training_datasets.get("single_animal_images", True)

    BASE_DIR_LIST: List[str] = []

    if not cfg.get("skip_cropping"):
        paste_dirs = (
            os.path.join(SESSION_PATH, "paste_blobs"),
            os.path.join(SESSION_PATH, "paste_blobs_clustered"),
            os.path.join(SESSION_PATH, "clustered"),
        )
        single_animal_images_source = (
            int(cfg.get("NUM_OBJECTS", 1)) == 1
            and bool(cfg.get("skip_paste_blobs_with_crossing", False))
            and bool(cfg.get("skip_paste_blobs_clustered", False))
            and not any(os.path.isdir(p) for p in paste_dirs)
        )
        if single_animal_images_source:
            print("Cropping single_animal_images...")
            BASE_DIR_LIST.append(os.path.join(SESSION_PATH, "single_animal_images"))
        elif use_single_animal_images:
            print("Cropping pasted/single_animal_images animal images...")
            paste_blobs_dir = os.path.join(SESSION_PATH, "paste_blobs")
            if os.path.isdir(paste_blobs_dir):
                BASE_DIR_LIST.append(paste_blobs_dir)
            else:
                BASE_DIR_LIST.append(os.path.join(SESSION_PATH, "single_animal_images"))

        if not single_animal_images_source:
            paste_blobs_clustered_dir = os.path.join(SESSION_PATH, "paste_blobs_clustered")
            if os.path.isdir(paste_blobs_clustered_dir):
                print("Cropping paste_blobs_clustered images...")
                BASE_DIR_LIST.append(paste_blobs_clustered_dir)

    IMG_EXT = ".png"
    MASK_DIR = os.path.join(SESSION_PATH, "single_animal_images", "masks")
    num_workers = _resolve_num_workers_bt(cfg.get("NUM_WORKERS", "auto"), task="process")
    RANDOM_SEED = normalize_seed(cfg.get("RANDOM_SEED", 0))
    print(f"[SEED] crop_images master={RANDOM_SEED}")

    if not BASE_DIR_LIST:
        sys.exit("No valid base directories found.")

    main(
        BASE_DIR_LIST=BASE_DIR_LIST,
        CROP_SIZE=CROP_SIZE,
        IMG_EXT=IMG_EXT,
        PREVIEW_INTERVAL=PREVIEW_INTERVAL,
        NUM_CROPS=NUM_CROPS,
        BACKGROUND_PATH=BACKGROUND_PATH,
        EDGE_BLUR_KSIZE=EDGE_BLUR_KSIZE,
        EDGE_BLUR_SIGMA=EDGE_BLUR_SIGMA,
        MASK_DIR=MASK_DIR,
        BLOBS_IN_VIDEO=BLOBS_IN_VIDEO,
        LOCALIZED=LOCALIZED,
        num_workers=num_workers,
        random_seed=RANDOM_SEED,
        append=bool(cfg.get("CROP_APPEND", False)),
    )


if __name__ == "__main__":
    from without_direction_estimation import prepare_config
    if len(sys.argv) > 1:
        sys.argv[1] = prepare_config(sys.argv[1])
    cli()

