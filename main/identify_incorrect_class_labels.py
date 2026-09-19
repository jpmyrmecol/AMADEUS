# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Use a preliminary detector to identify unreliable single-animal labels."""

import argparse
import csv
import gc
import math
import os
import pickle
import shutil
import sys
from collections import defaultdict
from glob import glob
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml
from batch_utils import tqdm
from ultralytics import YOLO

from batch_utils import (
    resolve_batch_size,
    resolve_device,
    resolve_num_workers as _resolve_num_workers,
    empty_accelerator_cache,
)
from create_dataset import (
    ALLOWED_ROTATION_ANGLES,
    cfg_bool,
    normalize_rotation_angles,
    write_yolo_dataset,
)
from crop_images import main as crop_images_main
from experiment_utils import DEFAULT_LR0, DEFAULT_LRF
from path_utils import resolve_config_paths
from obb_detector_training import DIRECTION_CLASS_NAMES, main as training_main, resolve_epoch_count

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gui.color import OBB_COLOR, OUTLIER_COLOR
from random_utils import derive_seed, normalize_seed

LOW_OBB_ASPECT_REASON = "low_obb_aspect"

RAW_YOLO_SUBDIR = "yolo_raw"
REFINE_DATASET_PREVIEW_COUNT = 10


def draw_text_plain(img, text, x, y, color, scale=0.6, thickness=1):
    cv2.putText(
        img,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def ensure_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    pts = pts[order]
    area2 = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area2 += x1 * y2 - x2 * y1
    if area2 > 0:
        pts = pts[::-1]
    return pts.astype(np.float32)


def draw_obb(img, pts: np.ndarray, color, thickness=2):
    poly = np.round(ensure_clockwise(pts)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, color, int(thickness), cv2.LINE_AA)


def obb_center(pts: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(pts, dtype=float).reshape(4, 2), axis=0)


def obb_to_xyxy(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=float).reshape(4, 2)
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], dtype=float)


def class_to_unit_vec(class_id: int) -> np.ndarray:
    th = math.radians(float(int(class_id) % 8) * 45.0)
    return np.array([math.sin(th), -math.cos(th)], dtype=float)


def get_short_edge_candidates(pts: np.ndarray, rel_tol: float = 0.05) -> List[Tuple[int, int, float]]:
    pts = ensure_clockwise(pts)
    edges = []
    lengths = []
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
        return np.asarray([0.0, 0.0], dtype=float)

    normal = np.asarray([edge[1], -edge[0]], dtype=float) / edge_len
    midpoint = 0.5 * (p0 + p1)
    center = obb_center(pts)
    if float(np.dot(midpoint - center, normal)) < 0.0:
        normal = -normal
    return normal


def draw_direction_triangle_for_obb(
    img,
    pts: np.ndarray,
    cls_idx: int,
    color,
    alpha: float = 0.6,
    scale: float = 1.0,
    outline_thickness: int = 1,
):
    img_c = np.ascontiguousarray(img)

    pts = ensure_clockwise(pts)
    dir_vec = class_to_unit_vec(int(cls_idx))
    short_edges = get_short_edge_candidates(pts)
    if not short_edges:
        return

    best = None
    for i, j, edge_len in short_edges:
        normal = outward_normal_for_edge(pts, i, j)
        score = float(np.dot(normal, dir_vec))
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
        overlay = img_c.copy()
        cv2.fillConvexPoly(overlay, tri_i32, color, lineType=cv2.LINE_AA)
        img_c[...] = cv2.addWeighted(overlay, a, img_c, 1.0 - a, 0.0)
    if int(outline_thickness) > 0:
        cv2.polylines(img_c, [tri_i32], True, color, int(outline_thickness), cv2.LINE_AA)

    img[...] = img_c


def render_yolo_prediction_overlay(
    image: np.ndarray,
    preds: List[dict],
    gt_objects: List[dict] | None = None,
    class_names: List[str] | None = None,
    line_thickness: int = 2,
    label_font_scale: float = 0.6,
    label_thickness: int = 1,
    arrow_alpha: float = 0.6,
    arrow_scale: float = 1.0,
) -> np.ndarray:
    # Generate the refine evaluation preview from the raw image and label/prediction
    # geometry every time.  This mirrors the create_single_animals_images / paste_blobs
    # preview policy and avoids reusing stale annotated preview images.
    out = np.ascontiguousarray(image.copy())

    gt_color = OBB_COLOR
    for gt in gt_objects or []:
        pts = gt["pts"]
        cls_idx = int(gt["class_id"])
        draw_obb(out, pts, gt_color, thickness=2)
        draw_direction_triangle_for_obb(out, pts, cls_idx, gt_color, alpha=0.6, scale=arrow_scale)

    pred_color = OUTLIER_COLOR
    for pred in preds:
        pts = pred["pts"]
        cls_idx = int(pred["class_id"])
        draw_obb(out, pts, pred_color, thickness=2)
        draw_direction_triangle_for_obb(out, pts, cls_idx, pred_color, alpha=0.6, scale=arrow_scale)
    return out


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def get_refine_dir_name(cfg: dict) -> str:
    name = str(cfg.get("REFINE_DIR_NAME", "refine")).strip()
    if not name:
        name = "refine"
    if os.path.basename(name) != name or name in {".", ".."}:
        raise ValueError(f"Invalid REFINE_DIR_NAME: {name!r}")
    return name


def is_refine_workspace_name(name: str) -> bool:
    return name == "refine" or name.startswith("refine_add") or name == "_iterative_refine_tmp"


def bboxes_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def move_path_without_deleting_original(src_path: str, dst_path: str) -> None:
    if not os.path.exists(src_path):
        return
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return
    if not os.path.exists(dst_path):
        shutil.move(src_path, dst_path)
        return
    if os.path.isdir(src_path) and os.path.isdir(dst_path):
        for name in os.listdir(src_path):
            move_path_without_deleting_original(
                os.path.join(src_path, name),
                os.path.join(dst_path, name),
            )
        if os.path.isdir(src_path) and not os.listdir(src_path):
            os.rmdir(src_path)
        return
    if os.path.isfile(src_path) and os.path.isfile(dst_path):
        src_size = os.path.getsize(src_path)
        dst_size = os.path.getsize(dst_path)
        if src_size == dst_size:
            os.remove(src_path)
            return
        raise FileExistsError(f"Destination already exists with different content: {dst_path}")
    raise FileExistsError(f"Destination already exists: {dst_path}")


def prepare_refine_workspace(without_crossing_dir: str, refine_root: str | None = None) -> Tuple[str, str]:
    if refine_root is None:
        refine_root = os.path.join(without_crossing_dir, "refine")
    original_dir = os.path.join(refine_root, "original")
    ensure_dirs([refine_root, original_dir])

    # If already evacuated to the "original" side, just reuse it as-is
    original_images = os.path.join(original_dir, "images")
    original_labels = os.path.join(original_dir, "labels")
    original_manifest = os.path.join(original_dir, "object_pool", "manifest.csv")
    if os.path.isdir(original_images) and os.path.isdir(original_labels) and os.path.exists(original_manifest):
        return refine_root, original_dir

    for name in sorted(os.listdir(without_crossing_dir)):
        if is_refine_workspace_name(name):
            continue
        move_path_without_deleting_original(
            os.path.join(without_crossing_dir, name),
            os.path.join(original_dir, name),
        )
    return refine_root, original_dir


def polygon_signed_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def obb_iou(a_pts: np.ndarray, b_pts: np.ndarray) -> float:
    a = np.asarray(a_pts, dtype=np.float32).reshape(4, 1, 2)
    b = np.asarray(b_pts, dtype=np.float32).reshape(4, 1, 2)
    area_a = float(cv2.contourArea(a))
    area_b = float(cv2.contourArea(b))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_area = float(cv2.intersectConvexConvex(a, b)[0])
    if inter_area <= 0.0:
        return 0.0
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0.0 else float(inter_area / union)


def get_min_obb_aspect_ratio(cfg: dict) -> float:
    return float(cfg["MIN_ASPECT"])


def obb_aspect_ratio_from_points(pts: np.ndarray) -> float | None:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    edge_lengths = [
        float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
        for i in range(4)
    ]
    long_side = max(edge_lengths) if edge_lengths else 0.0
    short_side = min(edge_lengths) if edge_lengths else 0.0
    if long_side <= 1e-6:
        return None
    if short_side <= 1e-6:
        return float("inf")
    return long_side / short_side


def circular_class_distance(a: int, b: int, num_classes: int = 8) -> int:
    diff = abs(int(a) - int(b)) % int(num_classes)
    return min(diff, int(num_classes) - diff)


def class_diff_to_angle_diff_deg(class_diff: int, num_classes: int = 8) -> float:
    return 360.0 * float(int(class_diff)) / float(int(num_classes))


def select_evenly_spaced_image_paths(
    image_paths_all: List[str],
    training_ratio: float,
    random_seed: int | None = None,
) -> List[str]:
    total = len(image_paths_all)
    if total == 0:
        return []

    ratio = float(training_ratio)
    if not np.isfinite(ratio):
        raise ValueError(f"REFINE_FRAME_RATIO must be finite: {training_ratio}")
    ratio = min(1.0, max(0.0, ratio))

    num_selected = int(round(total * ratio))
    num_selected = max(1, min(total, num_selected))
    if num_selected >= total:
        return list(image_paths_all)

    if random_seed is not None:
        rng = np.random.default_rng(int(random_seed))
        selected_indices = sorted(int(i) for i in rng.choice(total, size=num_selected, replace=False))
        return [image_paths_all[i] for i in selected_indices]

    indices = np.linspace(0, total - 1, num=num_selected, dtype=float)
    selected_indices = []
    used = set()
    for idx in indices:
        cand = int(round(float(idx)))
        if cand < 0:
            cand = 0
        elif cand >= total:
            cand = total - 1

        if cand in used:
            left = cand - 1
            right = cand + 1
            replacement = None
            while left >= 0 or right < total:
                if left >= 0 and left not in used:
                    replacement = left
                    break
                if right < total and right not in used:
                    replacement = right
                    break
                left -= 1
                right += 1
            if replacement is None:
                continue
            cand = replacement

        used.add(cand)
        selected_indices.append(cand)

    selected_indices.sort()
    if len(selected_indices) != num_selected:
        missing = [i for i in range(total) if i not in used]
        selected_indices.extend(missing[: num_selected - len(selected_indices)])
        selected_indices.sort()

    return [image_paths_all[i] for i in selected_indices]


def render_refine_dataset_preview(image: np.ndarray, label_lines: List[str]) -> np.ndarray:
    """Render OBB labels and their direction triangles on a YOLO dataset image."""
    preview = np.ascontiguousarray(image.copy())
    height, width = preview.shape[:2]
    obb_color = OBB_COLOR
    direction_color = obb_color

    for line_number, line in enumerate(label_lines, start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 9:
            raise ValueError(
                f"Invalid refine OBB label at line {line_number}: expected 9 fields, got {len(parts)}"
            )
        class_id = int(float(parts[0]))
        pts = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(4, 2)
        pts[:, 0] *= float(width)
        pts[:, 1] *= float(height)
        draw_obb(preview, pts, obb_color, thickness=2)
        draw_direction_triangle_for_obb(
            preview,
            pts,
            class_id,
            direction_color,
            alpha=0.6,
            scale=1.2,
            outline_thickness=1,
        )
    return preview


def write_refine_dataset_previews(
    dataset_root: str,
    max_previews: int = REFINE_DATASET_PREVIEW_COUNT,
) -> int:
    """Write previews sampled from the final train/test YOLO dataset."""
    preview_dir = os.path.join(dataset_root, "preview")
    if os.path.isdir(preview_dir):
        shutil.rmtree(preview_dir)
    os.makedirs(preview_dir, exist_ok=True)

    image_paths = []
    for split in ("train", "test"):
        split_image_dir = os.path.join(dataset_root, split, "images")
        if not os.path.isdir(split_image_dir):
            continue
        for name in sorted(os.listdir(split_image_dir)):
            if os.path.splitext(name)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                image_paths.append(os.path.join(split_image_dir, name))

    limit = max(0, int(max_previews))
    if limit == 0 or not image_paths:
        return 0
    selected_paths = select_evenly_spaced_image_paths(
        image_paths,
        training_ratio=min(1.0, float(limit) / float(len(image_paths))),
    )

    written = 0
    for image_path in selected_paths:
        split = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(dataset_root, split, "labels", f"{stem}.txt")
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Refine dataset label not found: {label_path}")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read refine dataset image: {image_path}")
        with open(label_path, "r", encoding="utf-8") as f:
            preview = render_refine_dataset_preview(image, f.readlines())

        output_path = os.path.join(preview_dir, f"{split}_{stem}.png")
        if not cv2.imwrite(output_path, preview):
            raise RuntimeError(f"Failed to write refine dataset preview: {output_path}")
        written += 1
    return written


def resolve_refine_dataset_workers(requested_workers=None) -> int:
    requested = "auto" if requested_workers is None else requested_workers
    return _resolve_num_workers(requested, task="process")


def build_training_dataset(
    original_dir: str,
    refine_root: str,
    image_size: int,
    background_path: str | None,
    *,
    img_ext: str = ".png",
    num_preview_frames: int = 100,
    preview_interval: int = 100,
    num_crops_per_img: int = 1,
    random_seed: int = 0,
    edge_blur_ksize: int = 7,
    edge_blur_sigma: float = 11.0,
    blobs_in_video=None,
    num_refine_training_ratio: float = 1.0,
    num_dataset_workers: int | None = None,
    val_ratio: float = 0.05,
    include_full_resized: bool = True,
    include_crop_highres: bool = True,
    skip_cropping: bool = False,
    localize_crop_on_base_blob: bool = False,
    rotation_angles: Sequence[int] | None = None,
) -> str:
    """Build the refine dataset with the regular crop/create-dataset pipeline."""
    image_size = int(image_size)
    if image_size <= 0:
        raise ValueError(f"TRAIN_IMG_SIZE must be positive: {image_size}")

    random_seed = normalize_seed(random_seed)
    img_ext = str(img_ext or ".png")
    image_paths_all = sorted(glob(os.path.join(original_dir, "images", f"frame_*{img_ext}")))
    if not image_paths_all:
        raise RuntimeError("No single_animal_images segment images found for refinement.")

    refine_selection_seed = derive_seed(random_seed, "refine_blobs", "source_frame_selection")
    image_paths = select_evenly_spaced_image_paths(
        image_paths_all=image_paths_all,
        training_ratio=float(num_refine_training_ratio),
        random_seed=refine_selection_seed,
    )
    source_selected_dir = os.path.join(refine_root, "source_selected")
    source_images_dir = os.path.join(source_selected_dir, "images")
    source_labels_dir = os.path.join(source_selected_dir, "labels")
    source_masks_dir = os.path.join(source_selected_dir, "masks")
    if os.path.isdir(source_selected_dir):
        shutil.rmtree(source_selected_dir)
    ensure_dirs([source_images_dir, source_labels_dir, source_masks_dir])

    original_labels_dir = os.path.join(original_dir, "labels")
    original_masks_dir = os.path.join(original_dir, "masks")
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(original_labels_dir, f"{stem}.txt")
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Refine source label not found: {label_path}")
        shutil.copy2(image_path, os.path.join(source_images_dir, os.path.basename(image_path)))
        shutil.copy2(label_path, os.path.join(source_labels_dir, f"{stem}.txt"))

        mask_path = os.path.join(original_masks_dir, f"{stem}_maskinfo.txt")
        if os.path.isfile(mask_path):
            shutil.copy2(mask_path, os.path.join(source_masks_dir, os.path.basename(mask_path)))

    num_dataset_workers = resolve_refine_dataset_workers(num_dataset_workers)
    num_crops_per_img = int(num_crops_per_img)
    if num_crops_per_img < 0:
        raise ValueError(f"NUM_CROPS must be non-negative: {num_crops_per_img}")
    include_full_resized = bool(include_full_resized)
    include_crop_highres = bool(include_crop_highres) and not bool(skip_cropping)
    rotation_angles = normalize_rotation_angles(
        list(ALLOWED_ROTATION_ANGLES if rotation_angles is None else rotation_angles)
    )

    print(
        f"[INFO] Selected {len(image_paths)}/{len(image_paths_all)} refine source frames "
        f"with REFINE_FRAME_RATIO={float(num_refine_training_ratio):.6f}"
    )
    print(f"[INFO] source_selected_dir={source_selected_dir}")
    print(f"[INFO] skip_cropping={bool(skip_cropping)}")
    print(f"[INFO] NUM_CROPS={num_crops_per_img}")
    print(f"[INFO] USE_FULL={include_full_resized}")
    print(f"[INFO] USE_CROP={include_crop_highres}")
    print(f"[INFO] LOCALIZED={bool(localize_crop_on_base_blob)}")
    print(f"[INFO] CREATE_DATASET_ROTATION_ANGLES={list(rotation_angles)}")

    cropping_dir = os.path.join(source_selected_dir, "cropping")
    if os.path.isdir(cropping_dir):
        shutil.rmtree(cropping_dir)
    if not skip_cropping:
        crop_images_main(
            BASE_DIR_LIST=[source_selected_dir],
            CROP_SIZE=image_size,
            IMG_EXT=img_ext,
            PREVIEW_INTERVAL=int(preview_interval),
            NUM_CROPS=num_crops_per_img,
            BACKGROUND_PATH=str(background_path or ""),
            EDGE_BLUR_KSIZE=int(edge_blur_ksize),
            EDGE_BLUR_SIGMA=float(edge_blur_sigma),
            MASK_DIR=source_masks_dir,
            BLOBS_IN_VIDEO=blobs_in_video,
            LOCALIZED=bool(localize_crop_on_base_blob),
            num_workers=num_dataset_workers,
            random_seed=derive_seed(random_seed, "refine_blobs", "crop_images"),
        )

    if not include_full_resized and not include_crop_highres:
        raise RuntimeError(
            "No refine dataset sample type is enabled. Enable USE_FULL or "
            "enable cropping with USE_CROP."
        )
    if not include_full_resized and include_crop_highres:
        crop_images_dir = os.path.join(cropping_dir, "images")
        crop_labels_dir = os.path.join(cropping_dir, "labels")
        has_crop_pair = any(
            os.path.isfile(os.path.join(crop_labels_dir, f"{os.path.splitext(name)[0]}.txt"))
            for name in os.listdir(crop_images_dir)
        ) if os.path.isdir(crop_images_dir) else False
        if not has_crop_pair:
            raise RuntimeError(
                "No refine crops were generated, and USE_FULL is disabled. "
                "Check TRAIN_IMG_SIZE, labels, and crop settings."
            )

    dataset_root = os.path.join(refine_root, "dataset")
    print(f"[INFO] output dataset path={dataset_root}")
    write_yolo_dataset(
        dataset_dirs=[source_selected_dir],
        output_root=dataset_root,
        img_ext=img_ext,
        val_ratio=float(val_ratio),
        num_total_images=0,
        nonclustered_ratio=1.0,
        output_size=image_size,
        seed=derive_seed(random_seed, "refine_blobs", "dataset_split"),
        include_full_resized=include_full_resized,
        include_crop_highres=include_crop_highres,
        num_workers=num_dataset_workers,
        background_path=background_path,
        allow_small_dataset=True,
        min_one_val=True,
        rotation_angles=rotation_angles,
        num_preview_frames=int(num_preview_frames),
    )
    write_refine_dataset_previews(dataset_root, max_previews=REFINE_DATASET_PREVIEW_COUNT)
    return dataset_root

def find_existing_last_model(refine_root: str) -> str | None:
    last_path = os.path.join(refine_root, "training", "detection", "weights", "last.pt")
    return last_path if os.path.isfile(last_path) else None


def keep_only_last_checkpoint(last_path: str) -> None:
    weights_dir = os.path.dirname(os.path.abspath(last_path))
    last_abs = os.path.abspath(last_path)
    if not os.path.isfile(last_abs):
        raise FileNotFoundError(f"last.pt was not found: {last_abs}")

    for name in os.listdir(weights_dir):
        path = os.path.join(weights_dir, name)
        if os.path.isfile(path) and name.endswith(".pt") and os.path.abspath(path) != last_abs:
            os.remove(path)


def train_refine_model(
    cfg: dict,
    refine_root: str,
    dataset_root: str | None = None,
    random_seed: int = 0,
) -> str:
    existing_last = find_existing_last_model(refine_root)
    if existing_last is not None:
        print(f"Found existing last.pt, skipping training: {existing_last}")
        keep_only_last_checkpoint(existing_last)
        return existing_last

    if dataset_root is None:
        raise RuntimeError("dataset_root is required when training is not skipped.")

    training_cfg = cfg.get("training", {}) or {}
    refine_epochs = resolve_epoch_count(cfg.get("REFINE_EPOCHS", 5), key="REFINE_EPOCHS")
    train_dir = os.path.join(refine_root, "training")
    ensure_dirs([train_dir])

    # Both detector types use training.main(), so optimizer, Mosaic and LR
    # policy stay aligned. Refine values are passed explicitly here.
    last_path = training_main(
        PRETRAINED_MODEL=str(cfg.get("REFINE_MODEL", "yolo11n-obb")),
        EPOCHS=refine_epochs,
        LR0=DEFAULT_LR0,
        LRF=DEFAULT_LRF,
        TRAIN_IMG_SIZE=int(cfg["TRAIN_IMG_SIZE"]),
        BATCH_SIZE=training_cfg.get("BATCH_SIZE", "auto"),
        DEVICE=str(training_cfg.get("DEVICE", "auto")),
        DATASET_DIR=dataset_root,
        NUM_CLASSES=len(DIRECTION_CLASS_NAMES),
        CLASS_NAMES=list(DIRECTION_CLASS_NAMES),
        PROJECT_DIR=train_dir,
        EXPERIMENT_NAME="detection",
        NUM_WORKERS=cfg.get("NUM_WORKERS", "auto"),
        SAVE_PERIOD=-1,
        RANDOM_SEED=derive_seed(normalize_seed(random_seed), "refine_blobs", "training"),
    )
    keep_only_last_checkpoint(last_path)
    return last_path


def load_manifest_blob_order(original_dir: str) -> Dict[int, List[int]]:
    manifest_path = os.path.join(original_dir, "object_pool", "manifest.csv")
    by_frame: Dict[int, List[int]] = defaultdict(list)
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_frame[int(row["frame"])].append(int(row["blob_index"]))
    return by_frame


def parse_gt_objects(
    original_dir: str,
    frame_id: int,
    blob_order_map: Dict[int, List[int]],
    img_w: int,
    img_h: int,
) -> List[dict]:
    stem = f"frame_{frame_id:06d}"
    lbl_path = os.path.join(original_dir, "labels", f"{stem}.txt")
    if not os.path.exists(lbl_path):
        return []
    blob_indices = blob_order_map.get(frame_id, [])
    out = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls_id = int(float(parts[0]))
            pts = np.asarray(list(map(float, parts[1:])), dtype=np.float32).reshape(4, 2)
            pts[:, 0] *= float(img_w)
            pts[:, 1] *= float(img_h)
            x1 = float(np.min(pts[:, 0]))
            y1 = float(np.min(pts[:, 1]))
            x2 = float(np.max(pts[:, 0]))
            y2 = float(np.max(pts[:, 1]))
            obb_aspect_ratio = obb_aspect_ratio_from_points(pts)
            out.append({
                "blob_index": blob_indices[i] if i < len(blob_indices) else i,
                "class_id": cls_id,
                "pts": pts,
                "bbox": (x1, y1, x2, y2),
                "obb_aspect_ratio": obb_aspect_ratio,
            })
    return out


def read_predictions(result) -> List[dict]:
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return []
    pts = obb.xyxyxyxy.cpu().numpy()
    confs = obb.conf.cpu().numpy()
    classes = obb.cls.cpu().numpy().astype(int)
    out = []
    for p, c, conf in zip(pts, classes, confs):
        p = np.asarray(p, dtype=np.float32).reshape(4, 2)
        x1 = float(np.min(p[:, 0]))
        y1 = float(np.min(p[:, 1]))
        x2 = float(np.max(p[:, 0]))
        y2 = float(np.max(p[:, 1]))
        out.append({
            "class_id": int(c),
            "conf": float(conf),
            "pts": p,
            "bbox": (x1, y1, x2, y2),
        })
    return out


def evaluate_one_prediction_result(
    *,
    img_path: str,
    image: np.ndarray,
    result,
    original_dir: str,
    blob_order_map: Dict[int, List[int]],
    conf_thresh: float,
    assignment_iou_thresh: float,
    direction_diff_thresh: int,
    min_obb_aspect_ratio: float,
    save_raw_overlay: bool,
    yolo_raw_dir: str,
    class_names: List[str],
    yolo_line_thickness: int,
    yolo_label_font_scale: float,
    yolo_label_thickness: int,
    yolo_arrow_alpha: float,
    yolo_arrow_scale: float,
) -> Tuple[int, int, int, int, int, List[dict]]:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    frame_id = int(stem.split("_")[1])

    if image is None:
        raise RuntimeError(f"Missing evaluation image array: {img_path}")
    h, w = image.shape[:2]

    gt_objects = parse_gt_objects(original_dir, frame_id, blob_order_map, w, h)
    preds_all = read_predictions(result)
    preds = [pred for pred in preds_all if float(pred["conf"]) >= conf_thresh]

    if save_raw_overlay:
        overlay = render_yolo_prediction_overlay(
            image=image,
            preds=preds,
            gt_objects=gt_objects,
            class_names=class_names,
            line_thickness=yolo_line_thickness,
            label_font_scale=yolo_label_font_scale,
            label_thickness=yolo_label_thickness,
            arrow_alpha=yolo_arrow_alpha,
            arrow_scale=yolo_arrow_scale,
        )
        out_overlay_path = os.path.join(yolo_raw_dir, f"{frame_id:06d}.png")
        if not cv2.imwrite(out_overlay_path, overlay):
            raise RuntimeError(f"Failed to write YOLO overlay image: {out_overlay_path}")

    delete_rows: List[dict] = []
    num_gt_pred_pairs_iou = 0

    for gt in gt_objects:
        gt_obb_aspect_ratio = gt.get("obb_aspect_ratio")
        if gt_obb_aspect_ratio is not None and np.isfinite(gt_obb_aspect_ratio) and float(gt_obb_aspect_ratio) < float(min_obb_aspect_ratio):
            delete_rows.append({
                "frame": frame_id,
                "blob_index": int(gt["blob_index"]),
                "gt_class_id": int(gt["class_id"]),
                "pred_class_id": "",
                "pred_conf": "",
                "best_obb_iou": "",
                "angle_diff_deg": "",
                "class_diff": "",
                "obb_aspect_ratio": f"{float(gt_obb_aspect_ratio):.6f}",
                "min_obb_aspect_ratio": f"{float(min_obb_aspect_ratio):.6f}",
                "reasons": LOW_OBB_ASPECT_REASON,
            })
            continue

        best_pred = None
        best_iou = 0.0
        for pred in preds:
            if not bboxes_overlap(gt["bbox"], pred["bbox"]):
                continue
            iou = obb_iou(gt["pts"], pred["pts"])
            if best_pred is None or iou > best_iou:
                best_pred = pred
                best_iou = float(iou)

        if best_pred is None:
            delete_rows.append({
                "frame": frame_id,
                "blob_index": int(gt["blob_index"]),
                "gt_class_id": int(gt["class_id"]),
                "pred_class_id": "",
                "pred_conf": "",
                "best_obb_iou": "0.000000",
                "angle_diff_deg": "",
                "class_diff": "",
                "obb_aspect_ratio": "",
                "min_obb_aspect_ratio": "",
                "reasons": "refine_no_prediction",
            })
            continue

        if best_iou < assignment_iou_thresh:
            delete_rows.append({
                "frame": frame_id,
                "blob_index": int(gt["blob_index"]),
                "gt_class_id": int(gt["class_id"]),
                "pred_class_id": int(best_pred["class_id"]),
                "pred_conf": f"{float(best_pred['conf']):.6f}",
                "best_obb_iou": f"{float(best_iou):.6f}",
                "angle_diff_deg": "",
                "class_diff": "",
                "obb_aspect_ratio": "",
                "min_obb_aspect_ratio": "",
                "reasons": "refine_no_overlap",
            })
            continue

        num_gt_pred_pairs_iou += 1
        class_diff = circular_class_distance(
            int(gt["class_id"]),
            int(best_pred["class_id"]),
            len(DIRECTION_CLASS_NAMES),
        )
        if class_diff >= direction_diff_thresh:
            angle_diff_deg = class_diff_to_angle_diff_deg(class_diff, len(DIRECTION_CLASS_NAMES))
            delete_rows.append({
                "frame": frame_id,
                "blob_index": int(gt["blob_index"]),
                "gt_class_id": int(gt["class_id"]),
                "pred_class_id": int(best_pred["class_id"]),
                "pred_conf": f"{float(best_pred['conf']):.6f}",
                "best_obb_iou": f"{float(best_iou):.6f}",
                "angle_diff_deg": f"{float(angle_diff_deg):.6f}",
                "class_diff": int(class_diff),
                "obb_aspect_ratio": "",
                "min_obb_aspect_ratio": "",
                "reasons": "refine_direction_mismatch_ge_90deg",
            })

    return (
        1,
        len(gt_objects),
        len(preds_all),
        len(preds),
        num_gt_pred_pairs_iou,
        delete_rows,
    )


def evaluate_predictions(cfg: dict, model_path: str, original_dir: str, refine_root: str) -> str:
    conf_thresh = float(cfg.get("REFINE_CONF", 0.2))
    assignment_iou_thresh = float(cfg.get("REFINE_ASSIGN_IOU_THRESH", 0.80))
    direction_diff_thresh = int(cfg.get("REFINE_DIRECTION_DIFF_THRESH", 2))
    image_size = int(cfg.get("REFINE_EVAL_IMAGE_SIZE", cfg["TRAIN_IMG_SIZE"]))
    device = resolve_device((cfg.get("training", {}) or {}).get("DEVICE", "auto"), purpose="refine eval")
    yolo_line_thickness = int(cfg.get("REFINE_YOLO_LINE_THICKNESS", 2))
    yolo_label_font_scale = float(cfg.get("REFINE_YOLO_LABEL_SCALE", 0.6))
    yolo_label_thickness = int(cfg.get("REFINE_YOLO_LABEL_THICKNESS", 1))
    yolo_arrow_alpha = float(cfg.get("REFINE_YOLO_ARROW_ALPHA", 0.6))
    yolo_arrow_scale = float(cfg.get("REFINE_YOLO_ARROW_SCALE", 1.0))
    min_obb_aspect_ratio = get_min_obb_aspect_ratio(cfg)

    out_csv = os.path.join(refine_root, "delete_blobs.csv")
    if os.path.exists(out_csv):
        print(f"[SKIP] Found existing completed delete CSV: {out_csv}")
        return out_csv

    tmp_csv = out_csv + ".tmp"
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    model = YOLO(model_path)
    class_names = list(getattr(model, "names", {}) .values()) if isinstance(getattr(model, "names", None), dict) else list(getattr(model, "names", []) or DIRECTION_CLASS_NAMES)
    print(f"[INFO] refine model path: {model_path}")
    print(f"[INFO] refine model task: {getattr(model, 'task', 'unknown')}")
    print(f"[INFO] refine model class names: {class_names}")
    blob_order_map = load_manifest_blob_order(original_dir)
    save_raw_overlay = bool(cfg.get("REFINE_SAVE_RAW_OVERLAY", False))
    yolo_raw_dir = os.path.join(refine_root, RAW_YOLO_SUBDIR)
    if save_raw_overlay:
        ensure_dirs([yolo_raw_dir])

    image_paths = sorted(glob(os.path.join(original_dir, "images", "frame_*.png")))
    if not image_paths:
        raise RuntimeError(f"No evaluation images found in: {os.path.join(original_dir, 'images')}")

    print(f"[INFO] evaluate_predictions uses original images: {os.path.join(original_dir, 'images')}")
    print(f"[INFO] REFINE_SAVE_RAW_OVERLAY={save_raw_overlay}")
    if save_raw_overlay:
        print(f"[INFO] YOLO raw overlays will be written to: {yolo_raw_dir}")

    num_images = 0
    num_gt_objects = 0
    num_preds_returned_by_yolo = 0
    num_preds_after_guard_conf = 0
    num_gt_pred_pairs_iou = 0
    eval_batch_size = resolve_batch_size(
        cfg.get("REFINE_BATCH", "auto"),
        image_size, device, mode="infer",
    )
    print(f"[INFO] REFINE_CONF={conf_thresh}")
    print(f"[INFO] MIN_ASPECT={min_obb_aspect_ratio}")
    print(f"[INFO] REFINE_BATCH={eval_batch_size}")
    print(f"[INFO] REFINE_EVAL_IMAGE_SIZE={image_size}")
    print("[INFO] REFINE_EVAL_CHUNK_SIZE is ignored by explicit Python-side batching.")

    _csv_fieldnames = [
        "frame", "blob_index", "gt_class_id", "pred_class_id", "pred_conf",
        "best_obb_iou", "angle_diff_deg", "class_diff",
        "obb_aspect_ratio", "min_obb_aspect_ratio", "reasons",
    ]
    _total_delete_rows = [0]
    _flush_counter = [0]
    _csv_f = open(tmp_csv, "w", encoding="utf-8", newline="")
    _csv_writer = csv.DictWriter(_csv_f, fieldnames=_csv_fieldnames)
    _csv_writer.writeheader()

    batch_imgs: List[np.ndarray] = []
    batch_paths: List[str] = []

    def flush_batch() -> None:
        nonlocal num_images, num_gt_objects, num_preds_returned_by_yolo, num_preds_after_guard_conf, num_gt_pred_pairs_iou, eval_batch_size
        if not batch_imgs:
            return

        _chunk_size = len(batch_imgs)
        results = []
        _pred_start = 0
        while _pred_start < len(batch_imgs):
            _chunk = list(batch_imgs[_pred_start: _pred_start + _chunk_size])
            try:
                results.extend(model.predict(
                    _chunk, imgsz=image_size, device=device,
                    verbose=False, task="obb", conf=conf_thresh, batch=len(_chunk),
                ))
                _pred_start += len(_chunk)
            except Exception as _oom_exc:
                if "out of memory" not in str(_oom_exc).lower():
                    raise
                empty_accelerator_cache(device)
                _chunk_size = max(1, _chunk_size // 2)
                eval_batch_size = min(eval_batch_size, _chunk_size)
                print(f"[WARN] OOM in refine eval -- retrying with chunk_size={_chunk_size}")
                if _chunk_size < 1:
                    raise RuntimeError("OOM at chunk_size=1") from _oom_exc
        if len(results) != len(batch_imgs):
            raise RuntimeError(f"YOLO returned {len(results)} results for {len(batch_imgs)} batched images.")

        for img_path, image, result in zip(batch_paths, batch_imgs, results):
            n_img, n_gt, n_pred_all, n_pred_guard, n_pairs, rows = evaluate_one_prediction_result(
                img_path=img_path,
                image=image,
                result=result,
                original_dir=original_dir,
                blob_order_map=blob_order_map,
                conf_thresh=conf_thresh,
                assignment_iou_thresh=assignment_iou_thresh,
                direction_diff_thresh=direction_diff_thresh,
                min_obb_aspect_ratio=min_obb_aspect_ratio,
                save_raw_overlay=save_raw_overlay,
                yolo_raw_dir=yolo_raw_dir,
                class_names=class_names,
                yolo_line_thickness=yolo_line_thickness,
                yolo_label_font_scale=yolo_label_font_scale,
                yolo_label_thickness=yolo_label_thickness,
                yolo_arrow_alpha=yolo_arrow_alpha,
                yolo_arrow_scale=yolo_arrow_scale,
            )
            num_images += n_img
            num_gt_objects += n_gt
            num_preds_returned_by_yolo += n_pred_all
            num_preds_after_guard_conf += n_pred_guard
            num_gt_pred_pairs_iou += n_pairs
            if rows:
                _csv_writer.writerows(rows)
                _total_delete_rows[0] += len(rows)

        del results
        _flush_counter[0] += 1
        if _flush_counter[0] % 10 == 0:
            gc.collect()

        batch_imgs.clear()
        batch_paths.clear()

    with tqdm(total=len(image_paths), desc="Evaluating predictions") as pbar:
        for img_path in image_paths:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read evaluation image: {img_path}")
            batch_imgs.append(image)
            batch_paths.append(img_path)
            if len(batch_imgs) >= eval_batch_size:
                flush_batch()
                pbar.update(eval_batch_size)
        tail_count = len(batch_imgs)
        flush_batch()
        if tail_count:
            pbar.update(tail_count)

    _csv_f.close()

    if num_images != len(image_paths):
        raise RuntimeError(f"YOLO evaluated {num_images} results for {len(image_paths)} images.")

    os.replace(tmp_csv, out_csv)

    print(
        f"[SUMMARY] evaluated_images={num_images}, gt_objects={num_gt_objects}, "
        f"preds_returned_by_yolo_conf={num_preds_returned_by_yolo}, "
        f"preds_after_guard_conf={num_preds_after_guard_conf}, "
        f"pairs_above_iou={num_gt_pred_pairs_iou}, delete_rows={_total_delete_rows[0]}"
    )
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, train, and evaluate the refine blob OBB model.")
    parser.add_argument("config", help="Path to config.yaml")
    parser.add_argument(
        "--dataset-workers",
        type=int,
        default=None,
        help=(
            "Number of parallel workers for refine dataset building. "
            "If omitted, NUM_WORKERS in config is used."
        ),
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = resolve_config_paths(cfg)

    without_crossing_dir = os.path.join(str(cfg["SESSION_PATH"]), "single_animal_images")
    if not os.path.isdir(without_crossing_dir):
        raise FileNotFoundError(without_crossing_dir)

    image_size = int(cfg["TRAIN_IMG_SIZE"])
    background_path = str(cfg["BACKGROUND_PATH"])
    img_ext = str(cfg.get("IMG_EXT", ".png") or ".png")
    num_preview_frames = int(cfg.get("NUM_PREVIEW_FRAMES", 100))
    preview_interval = int(cfg.get("PREVIEW_INTERVAL", 100))
    num_crops_per_img = int(cfg.get("NUM_CROPS", 1))
    random_seed = normalize_seed(cfg.get("RANDOM_SEED", 0))
    print(f"[SEED] refine_blobs master={random_seed}")
    edge_blur_ksize = int(cfg.get("EDGE_BLUR_KSIZE", 7))
    edge_blur_sigma = float(cfg.get("EDGE_BLUR_SIGMA", 11.0))
    pickle_path = str(cfg.get("PICKLE_PATH", ""))
    num_refine_training_ratio = float(cfg.get("REFINE_FRAME_RATIO", 1.0))
    refine_val_ratio = float(cfg.get("REFINE_VAL_RATIO", 0.05))
    skip_cropping = cfg_bool(cfg, "skip_cropping", False)
    include_full_resized = cfg_bool(cfg, "USE_FULL", True)
    include_crop_highres = cfg_bool(cfg, "USE_CROP", True)
    localize_crop_on_base_blob = cfg_bool(cfg, "LOCALIZED", False)
    rotation_angles = normalize_rotation_angles(
        cfg.get("CREATE_DATASET_ROTATION_ANGLES", list(ALLOWED_ROTATION_ANGLES))
    )

    cfg_workers = cfg.get("NUM_WORKERS", None)
    requested_dataset_workers = args.dataset_workers if args.dataset_workers is not None else cfg_workers
    num_dataset_workers = resolve_refine_dataset_workers(requested_dataset_workers)

    blobs_in_video = None
    if pickle_path:
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"PICKLE_PATH not found: {pickle_path}")
        with open(pickle_path, "rb") as f:
            blobs_obj = pickle.load(f)
        blobs_in_video = getattr(blobs_obj, "blobs_in_video", None)
        if blobs_in_video is None:
            raise RuntimeError(f"blobs_in_video was not found in pickle: {pickle_path}")

    refine_dir_name = get_refine_dir_name(cfg)
    refine_root = os.path.join(without_crossing_dir, refine_dir_name)
    refine_root, original_dir = prepare_refine_workspace(without_crossing_dir, refine_root=refine_root)

    existing_last = find_existing_last_model(refine_root)
    dataset_root = None
    if existing_last is None:
        dataset_root = build_training_dataset(
            original_dir=original_dir,
            refine_root=refine_root,
            image_size=image_size,
            background_path=background_path,
            img_ext=img_ext,
            num_preview_frames=num_preview_frames,
            preview_interval=preview_interval,
            num_crops_per_img=num_crops_per_img,
            random_seed=random_seed,
            edge_blur_ksize=edge_blur_ksize,
            edge_blur_sigma=edge_blur_sigma,
            blobs_in_video=blobs_in_video,
            num_refine_training_ratio=num_refine_training_ratio,
            num_dataset_workers=num_dataset_workers,
            val_ratio=refine_val_ratio,
            include_full_resized=include_full_resized,
            include_crop_highres=include_crop_highres,
            skip_cropping=skip_cropping,
            localize_crop_on_base_blob=localize_crop_on_base_blob,
            rotation_angles=rotation_angles,
        )
    else:
        print(f"Found existing last.pt before dataset build: {existing_last}")

    model_path = train_refine_model(cfg, refine_root, dataset_root, random_seed=random_seed)
    delete_csv = os.path.join(refine_root, "delete_blobs.csv")
    if os.path.exists(delete_csv):
        print(f"Found existing delete_blobs.csv, skipping evaluation: {delete_csv}")
    else:
        tmp_csv = delete_csv + ".tmp"
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
        print(f"[INFO] Refinement evaluation target images: {os.path.join(original_dir, 'images')}")
        delete_csv = evaluate_predictions(cfg, model_path, original_dir, refine_root)
    print(f"Saved refine workspace to: {refine_root}")
    print(f"Saved YOLO raw overlays to: {os.path.join(refine_root, RAW_YOLO_SUBDIR)}")
    print(f"Saved delete list to: {delete_csv}")

if __name__ == "__main__":
    main()
