# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Detect oriented animal boxes and directional classes in video frames."""

import os
import sys
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import cv2
import pandas as pd
from tracking_artifacts import artifact_path as tracking_artifact_path
import re
import math
import yaml
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from batch_utils import resolve_batch_size, resolve_device, run_with_oom_retry, tqdm
from checkpoint_utils import (
    normalize_checkpoint_weight,
    parse_checkpoint_spec,
    weight_to_checkpoint_number,
)
from experiment_utils import (
    experiment_dir_name_from_cfg,
    resolve_existing_experiment_dir_name,
)
from training_paths import OUTPUT_ROOT_DIR, training_weight_path
from weight_utils import deduplicate_best_epoch_weights
from video_frame_count import (
    clamp_frame_range_to_usable_count,
    read_video_frame_info,
    warn_if_frame_count_adjusted,
)
from gui.color import OBB_COLOR
from path_utils import resolve_config_paths

DIRECTION_CLASS_NAMES = ['upper', 'upper_right', 'right', 'lower_right', 'lower', 'lower_left', 'left', 'upper_left']

def tqdm_it(*args, **kwargs):
    kw = dict(file=sys.stdout, dynamic_ncols=True, mininterval=0.2)
    kw.update(kwargs)
    return tqdm(*args, **kw)

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return resolve_config_paths(cfg)

def resolve_num_workers(cfg: dict, section_name: str | None = None, default: int | None = None, cap: int | None = None) -> int:
    from batch_utils import auto_num_workers as _auto
    keys = ('NUM_WORKERS', 'WORKERS', 'N_WORKERS', 'workers', 'num_workers')
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
    if value is None or str(value).strip().lower() in {'', 'auto', 'none'}:
        workers = int(default) if default is not None else _auto("process")
    else:
        workers = int(value)
    workers = max(1, workers)
    if cap is not None:
        workers = min(workers, max(1, int(cap)))
    return workers

def parse_weight_spec(weight_spec):
    """Parse 1-based analysis/create-video selectors into YOLO weight stems."""
    return parse_checkpoint_spec(weight_spec, key='WEIGHT')

def append_pt_if_missing(model_name: str) -> str:
    return model_name if model_name.endswith('.pt') else model_name + '.pt'

def resolve_model_name(model_name: str) -> str:
    model_name = append_pt_if_missing(model_name)
    stem, ext = os.path.splitext(model_name)
    return model_name if stem.endswith('-obb') else f'{stem}-obb{ext}'

def wrap_angle_deg(angle_deg: float) -> float:
    return float(angle_deg) % 360.0

def class_to_angle_deg(class_id: float | int) -> float:
    value = float(class_id)
    rounded = int(round(value))
    if abs(value - rounded) <= 1e-06 and 0 <= rounded < 8:
        return float(rounded) * 45.0
    return wrap_angle_deg(value)

def yolo_class_to_angle_deg(class_id: float | int) -> float:
    return float(int(round(float(class_id))) % 8) * 45.0

def angle_to_unit_vec(angle_deg: float | int) -> np.ndarray:
    th = math.radians(wrap_angle_deg(float(angle_deg)))
    return np.array([math.sin(th), -math.cos(th)], dtype=float)

def class_to_unit_vec(class_id: float | int) -> np.ndarray:
    return angle_to_unit_vec(class_to_angle_deg(class_id))

def yolo_class_to_unit_vec(class_id: float | int) -> np.ndarray:
    return angle_to_unit_vec(yolo_class_to_angle_deg(class_id))

def reference_to_unit_vec(reference: float | int | np.ndarray | None) -> np.ndarray | None:
    if reference is None:
        return None
    if isinstance(reference, np.ndarray):
        return normalize_direction_vec(reference)
    if isinstance(reference, (tuple, list)) and len(reference) == 2:
        return normalize_direction_vec(np.asarray(reference, dtype=float))
    if isinstance(reference, (int, float, np.integer, np.floating)) and np.isfinite(reference):
        return class_to_unit_vec(float(reference))
    return None

def unit_vec_to_angle_deg(vec: np.ndarray) -> float:
    v = np.asarray(vec, dtype=float).reshape(2)
    n = float(np.linalg.norm(v))
    if n <= 1e-06:
        return 0.0
    v = v / n
    deg = math.degrees(math.atan2(v[0], -v[1])) % 360.0
    return float(deg)

def normalize_direction_vec(vec: np.ndarray, fallback: np.ndarray | None=None) -> np.ndarray:
    v = np.asarray(vec, dtype=float).reshape(2)
    n = float(np.linalg.norm(v))
    if n > 1e-06:
        return v / n
    if fallback is not None:
        fb = np.asarray(fallback, dtype=float).reshape(2)
        n_fb = float(np.linalg.norm(fb))
        if n_fb > 1e-06:
            return fb / n_fb
    return np.array([1.0, 0.0], dtype=float)

def positive_long_axis_direction(pts: np.ndarray) -> np.ndarray:
    major_axis, major_len, _ = obb_major_axis(pts)
    if major_len <= 0.0:
        return np.array([1.0, 0.0], dtype=float)
    return normalize_direction_vec(major_axis)

def closest_long_axis_direction(pts: np.ndarray, reference: float | int | np.ndarray | None) -> np.ndarray:
    major_axis = positive_long_axis_direction(pts)
    ref_vec = reference_to_unit_vec(reference)
    if ref_vec is None:
        return major_axis
    return major_axis if float(np.dot(major_axis, ref_vec)) >= 0.0 else -major_axis

def polygon_signed_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=float).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def top_left_start_index(pts: np.ndarray) -> int:
    pts = np.asarray(pts, dtype=float).reshape(4, 2)
    return int(np.lexsort((pts[:, 0], pts[:, 1]))[0])

def rotate_start(pts: np.ndarray, start_idx: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    return np.roll(pts, -int(start_idx), axis=0).astype(np.float32)

def ensure_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    pts = pts[order]
    if polygon_signed_area(pts) > 0.0:
        pts = pts[::-1]
    pts = rotate_start(pts, top_left_start_index(pts))
    return pts.astype(np.float32)

def obb_center(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(4, 2)
    return np.mean(pts, axis=0)

def obb_major_axis(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    pts = ensure_clockwise(pts)
    e0 = pts[1] - pts[0]
    e1 = pts[2] - pts[1]
    l0 = float(np.linalg.norm(e0))
    l1 = float(np.linalg.norm(e1))
    if l0 >= l1:
        major = e0 / max(l0, 1e-06)
        return (major.astype(float), l0, l1)
    major = e1 / max(l1, 1e-06)
    return (major.astype(float), l1, l0)

def obb_to_xyxy(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(4, 2)
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=float)

def obb_polygon_area(pts: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2))))

def iou_obb(pts_a: np.ndarray | None, pts_b: np.ndarray | None) -> float:
    if pts_a is None or pts_b is None:
        return 0.0
    pa = ensure_clockwise(pts_a)
    pb = ensure_clockwise(pts_b)
    area_a = obb_polygon_area(pa)
    area_b = obb_polygon_area(pb)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    inter_area = float(inter_area)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0

def draw_obb(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], thickness: int=2) -> None:
    poly = np.round(ensure_clockwise(pts)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, color, thickness, cv2.LINE_AA)

def draw_text_plain(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float=0.55, thickness: int=1) -> None:
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), color, int(thickness), cv2.LINE_AA)

def get_short_edge_candidates(pts: np.ndarray, rel_tol: float=0.05) -> list[tuple[int, int, float]]:
    pts = ensure_clockwise(pts)
    edges = []
    lengths = []
    for i in range(4):
        j = (i + 1) % 4
        edge_len = float(np.linalg.norm(pts[j] - pts[i]))
        edges.append((i, j, edge_len))
        lengths.append(edge_len)
    min_len = min(lengths)
    tol = max(1e-06, min_len * float(rel_tol))
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

def draw_direction_triangle_for_obb(img: np.ndarray, pts: np.ndarray, class_id: float, color: tuple[int, int, int], alpha: float=0.6, outline_thickness: int=1, scale: float=1.0) -> None:
    pts = ensure_clockwise(pts)
    dir_vec = angle_to_unit_vec(float(class_id))
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
    p0 = pts[i].astype(float)
    p1 = pts[j].astype(float)
    base_mid = 0.5 * (p0 + p1)
    height = float(scale) * (math.sqrt(3.0) / 2.0) * float(base_len)
    apex = base_mid + normal * height
    tri = np.vstack([p0, p1, apex]).astype(np.float32)
    tri_i32 = np.round(tri).astype(np.int32)
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, tri_i32, color, lineType=cv2.LINE_AA)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)
    if int(outline_thickness) > 0:
        cv2.polylines(img, [tri_i32], True, color, int(outline_thickness), cv2.LINE_AA)

def extract_obb_detections(pred) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if getattr(pred, 'obb', None) is None or len(pred.obb) == 0:
        return (np.zeros((0, 4, 2), dtype=np.float32), np.zeros((0,), dtype=float), np.zeros((0,), dtype=int))
    pts = pred.obb.xyxyxyxy.cpu().numpy().astype(np.float32).reshape(-1, 4, 2)
    conf = pred.obb.conf.cpu().numpy().astype(float) if pred.obb.conf is not None else np.zeros((len(pts),), dtype=float)
    cls = pred.obb.cls.cpu().numpy().astype(int) if pred.obb.cls is not None else np.zeros((len(pts),), dtype=int)
    pts = np.asarray([ensure_clockwise(p) for p in pts], dtype=np.float32)
    return (pts, conf, cls)

def class_name_from_direction(value: float | int | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return 'unknown'
    idx = int(round(float(value))) % len(DIRECTION_CLASS_NAMES)
    return DIRECTION_CLASS_NAMES[idx]

def filter_candidate_df_for_tracking(df: pd.DataFrame, conf_threshold: float) -> pd.DataFrame:
    """Keep YOLO-NMS output intact, applying only confidence filtering and indexing."""
    if df is None or df.empty:
        return make_empty_candidate_df()
    work = df.copy().reset_index(drop=True)
    work['score'] = pd.to_numeric(work['score'], errors='coerce')
    work = work[np.isfinite(work['score']) & (work['score'] >= float(conf_threshold))].copy()
    if work.empty:
        return make_empty_candidate_df()
    if 'source_det_index' not in work.columns:
        work['source_det_index'] = work['det_index'].astype(int)
    work = work.sort_values(['frame', 'score', 'source_weight', 'det_index'], ascending=[True, False, True, True]).reset_index(drop=True)
    work['det_index'] = work.groupby('frame').cumcount().astype(int)
    return work

def candidate_df_to_blob_df(df: pd.DataFrame, kind: str = 'direction') -> pd.DataFrame:
    cols = [
        'frame', 'det_index', 'source_det_index', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'score', 'direction', 'direction_name'
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    work = df.copy().reset_index(drop=True)
    if 'source_det_index' not in work.columns:
        work['source_det_index'] = work.get('det_index', pd.Series(np.arange(len(work)), index=work.index)).astype(int)

    x_cols = ['x0', 'x1', 'x2', 'x3']
    y_cols = ['y0', 'y1', 'y2', 'y3']
    x = work[x_cols].to_numpy(dtype=float)
    y = work[y_cols].to_numpy(dtype=float)
    direction = pd.to_numeric(work['direction'], errors='coerce')

    out = pd.DataFrame({
        'frame': pd.to_numeric(work['frame'], errors='coerce').fillna(-1).astype(int),
        'det_index': pd.to_numeric(work.get('det_index', -1), errors='coerce').fillna(-1).astype(int),
        'source_det_index': pd.to_numeric(work['source_det_index'], errors='coerce').fillna(-1).astype(int),
        'bbox_x1': np.nanmin(x, axis=1),
        'bbox_y1': np.nanmin(y, axis=1),
        'bbox_x2': np.nanmax(x, axis=1),
        'bbox_y2': np.nanmax(y, axis=1),
        'x0': x[:, 0], 'y0': y[:, 0],
        'x1': x[:, 1], 'y1': y[:, 1],
        'x2': x[:, 2], 'y2': y[:, 2],
        'x3': x[:, 3], 'y3': y[:, 3],
        'score': pd.to_numeric(work.get('score', np.nan), errors='coerce'),
        'direction': direction,
    })
    out['direction_name'] = direction.apply(class_name_from_direction)
    return out[cols]

def save_blob_pickle(path: str, rows_or_df) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(rows_or_df, pd.DataFrame):
        df = rows_or_df
    else:
        df = pd.DataFrame(rows_or_df) if rows_or_df else pd.DataFrame()
    df.to_pickle(path)

def load_blob_pickle(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_pickle(path)

def _preview_frame_ids(first_frame: int, last_frame: int, interval: int) -> list[int]:
    """Every `interval`-th frame starting from the first frame of the range."""
    first = int(first_frame)
    last = int(last_frame)
    if last < first:
        return []
    step = int(interval)
    if step <= 0:
        return []
    return list(range(first, last + 1, step))

def _rows_to_preview_records(g: pd.DataFrame | None) -> list[dict]:
    if g is None or g.empty:
        return []
    cols = ['x0', 'x1', 'x2', 'x3', 'y0', 'y1', 'y2', 'y3', 'direction', 'score']
    records: list[dict] = []
    for _, row in g.iterrows():
        rec = {}
        for col in cols:
            rec[col] = float(row.get(col, np.nan))
        records.append(rec)
    return records

def _draw_blob_preview_image(img: np.ndarray, records: list[dict], out_path: str) -> int:
    """Draw and save one preview without owning the video reader."""
    for row in records:
        pts = np.stack([
            np.asarray([row.get('x0', np.nan), row.get('x1', np.nan), row.get('x2', np.nan), row.get('x3', np.nan)], dtype=float),
            np.asarray([row.get('y0', np.nan), row.get('y1', np.nan), row.get('y2', np.nan), row.get('y3', np.nan)], dtype=float),
        ], axis=1).astype(np.float32)
        if not np.isfinite(pts).all():
            continue
        draw_obb(img, pts, OBB_COLOR, 2)
        direction = float(row.get('direction', np.nan))
        if np.isfinite(direction):
            draw_direction_triangle_for_obb(img, pts, direction, OBB_COLOR, alpha=0.6, outline_thickness=1, scale=1.2)
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    return 1 if cv2.imwrite(out_path, img, jpeg_params) else 0


def _render_blob_preview_batch(
    tasks: list[tuple[str, int, list[dict], str]],
    show_progress: bool = False,
    progress_desc: str = 'PREVIEW',
) -> int:
    if not tasks:
        return 0
    video_path = tasks[0][0]
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    next_frame = None
    written = 0
    try:
        task_iter = tqdm_it(tasks, desc=progress_desc, unit='frame') if show_progress else tasks
        for _, fid, records, out_path in task_iter:
            fid = int(fid)
            if next_frame != fid:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, img = cap.read()
            next_frame = fid + 1 if ok else None
            if not ok or img is None:
                continue
            written += _draw_blob_preview_image(img, records, out_path)
    finally:
        cap.release()
    return written


def render_blob_preview(video_path: str, df: pd.DataFrame, out_dir: str, first_frame: int, last_frame: int, preview_interval: int, kind: str, workers: int = 1) -> None:
    frame_ids = _preview_frame_ids(first_frame, last_frame, preview_interval)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not frame_ids:
        print(f'PREVIEW {kind}: no frames in analysis range.')
        return

    work = df.copy() if df is not None else pd.DataFrame()
    if not work.empty and 'frame' in work.columns:
        work['frame'] = pd.to_numeric(work['frame'], errors='coerce').astype('Int64')
        work = work[work['frame'].isin(frame_ids)].copy()
    grouped = {int(f): _rows_to_preview_records(g) for f, g in work.groupby('frame', sort=True)} if not work.empty else {}
    tasks = [
        (video_path, int(fid), grouped.get(int(fid), []), os.path.join(out_dir, f'{int(fid):06d}.jpg'))
        for fid in frame_ids
    ]
    workers = max(1, min(int(workers), len(tasks)))
    print(f'PREVIEW {kind}: frames={frame_ids[0]}-{frame_ids[-1]} count={len(frame_ids)} workers={workers}')

    written = 0
    if workers <= 1:
        written = _render_blob_preview_batch(tasks, show_progress=True, progress_desc=f'PREVIEW {kind}')
    else:
        chunk_size = max(1, math.ceil(len(tasks) / workers))
        task_batches = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_render_blob_preview_batch, batch): len(batch) for batch in task_batches}
            with tqdm_it(total=len(tasks), desc=f'PREVIEW {kind} x{workers}', unit='frame') as pbar:
                for fut in as_completed(futures):
                    written += int(fut.result())
                    pbar.update(futures[fut])
    print(f'PREVIEW {kind}: written={written}/{len(tasks)}')

def build_tracking_run_name(weight: str) -> str:
    """Return the single-checkpoint tracking directory name."""
    return normalize_checkpoint_weight(weight)

def build_tracking_out_dir(session_path: str, model_name: str, dataset_name: str, run_name: str, video_name: str) -> str:
    return os.path.join(session_path, OUTPUT_ROOT_DIR, model_name, 'tracking', dataset_name, run_name, video_name)

def resolve_tracking_jobs(weights: list[str]) -> list[dict]:
    weights = list(dict.fromkeys((str(w) for w in weights)))
    if not weights:
        raise ValueError('weights is empty')
    return [
        {'run_name': build_tracking_run_name(weight), 'source_weight': weight}
        for weight in weights
    ]

def build_weight_path(session_path: str, model_name: str, dataset_name: str, weight: str) -> str:
    return training_weight_path(session_path, model_name, dataset_name, weight)

def pickle_has_data(path: str) -> bool:
    if not os.path.exists(path):
        return False
    df = pd.read_pickle(path)
    return not df.empty and len(df.columns) > 0

def make_empty_candidate_df() -> pd.DataFrame:
    cols = ['frame', 'det_index', 'score', 'direction', 'source_weight', 'x0', 'x1', 'x2', 'x3', 'y0', 'y1', 'y2', 'y3']
    return pd.DataFrame(columns=cols)

def direction_reference_is_compatible_with_long_axis(
    pts: np.ndarray,
    reference: float | int | np.ndarray | None,
    *,
    reference_is_yolo_class: bool,
    max_axis_error_deg: float = 67.5,
) -> bool:
    if reference is None:
        return False

    if isinstance(reference, np.ndarray):
        ref_vec = normalize_direction_vec(reference)
    elif isinstance(reference, (tuple, list)) and len(reference) == 2:
        ref_vec = normalize_direction_vec(np.asarray(reference, dtype=float))
    elif isinstance(reference, (int, float, np.integer, np.floating)) and np.isfinite(reference):
        ref_vec = (
            yolo_class_to_unit_vec(reference)
            if reference_is_yolo_class
            else angle_to_unit_vec(reference)
        )
    else:
        return False

    major_vec = positive_long_axis_direction(pts)
    cos_thr = math.cos(math.radians(float(max_axis_error_deg)))

    # Use abs() because both head and tail directions along the OBB long axis
    # are geometrically valid. Perpendicular short-axis classes should fail.
    return abs(float(np.dot(major_vec, ref_vec))) >= cos_thr

def normalize_candidate_df_direction(df: pd.DataFrame, reference_is_yolo_class: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return make_empty_candidate_df()
    work = df.copy()
    xy_cols = ['x0', 'x1', 'x2', 'x3', 'y0', 'y1', 'y2', 'y3']
    ref_values = pd.to_numeric(work['direction'], errors='coerce').to_numpy(dtype=float)
    vals = work[xy_cols].to_numpy(dtype=float, copy=False)
    new_dir = np.full(len(work), np.nan, dtype=float)
    for i, row in enumerate(vals):
        if not np.isfinite(row).all():
            continue

        pts = ensure_clockwise(np.stack([row[:4], row[4:]], axis=1))

        if not np.isfinite(ref_values[i]):
            continue

        # Reject labels that point to the OBB short axis.
        if not direction_reference_is_compatible_with_long_axis(
            pts,
            ref_values[i],
            reference_is_yolo_class=reference_is_yolo_class,
            max_axis_error_deg = 45 + 22.5,
        ):
            continue

        reference = (
            yolo_class_to_unit_vec(ref_values[i])
            if reference_is_yolo_class
            else angle_to_unit_vec(ref_values[i])
        )

        signed_axis = closest_long_axis_direction(pts, reference)
        new_dir[i] = unit_vec_to_angle_deg(signed_axis)
    work['direction'] = new_dir
    work = work[np.isfinite(work['direction'])].copy()
    if work.empty:
        return make_empty_candidate_df()
    work['direction'] = pd.to_numeric(work['direction'], errors='coerce')
    work = work.sort_values(['frame', 'score', 'source_weight', 'det_index'], ascending=[True, False, True, True]).reset_index(drop=True)
    work['det_index'] = work.groupby('frame').cumcount().astype(int)
    return work

def video_inference_image_size(width: int, height: int) -> int:
    """Return the native video long side rounded up to YOLO's stride."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid video dimensions: {width}x{height}')
    return ((max(width, height) + 31) // 32) * 32


def collect_candidate_df(
    model,
    video_path: str,
    first_frame: int,
    last_frame: int,
    device,
    min_conf: float,
    source_weight: str,
    image_size: int,
    batch_size: int = 16,
    nms_iou: float = 1.0,
    max_det: int = 300,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    n_frames = max(0, int(last_frame) - int(first_frame) + 1)
    imgsz = max(32, int(image_size))
    batch_size = max(1, int(batch_size))
    batch_imgs: list[np.ndarray] = []
    batch_fids: list[int] = []
    batch_idx = 0
    temp_dir = tempfile.mkdtemp(prefix='od_batches_')

    def flush_batch():
        nonlocal batch_idx
        if not batch_imgs:
            return
        preds = model.predict(batch_imgs, device=device, imgsz=imgsz, conf=min_conf, iou=nms_iou, verbose=False, agnostic_nms=True, max_det=max_det)
        batch_rows = []
        for abs_fid, pred in zip(batch_fids, preds):
            obbs_all, scores_all, classes_all = extract_obb_detections(pred)
            if len(obbs_all) == 0:
                continue
            for det_index, (pts, score, cls_id) in enumerate(zip(obbs_all, scores_all, classes_all)):
                pts = np.asarray(pts, dtype=float).reshape(4, 2)
                batch_rows.append({
                    'frame': abs_fid,
                    'det_index': int(det_index),
                    'source_det_index': int(det_index),
                    'source_weight': str(source_weight),
                    'score': float(score),
                    'direction': float(cls_id),
                    'x0': float(pts[0, 0]), 'x1': float(pts[1, 0]), 'x2': float(pts[2, 0]), 'x3': float(pts[3, 0]),
                    'y0': float(pts[0, 1]), 'y1': float(pts[1, 1]), 'y2': float(pts[2, 1]), 'y3': float(pts[3, 1]),
                })
        if batch_rows:
            pd.DataFrame(batch_rows).to_pickle(os.path.join(temp_dir, f'{batch_idx:07d}.pkl'))
            batch_idx += 1
        batch_imgs.clear()
        batch_fids.clear()

    try:
        for rel_idx in tqdm_it(range(n_frames), desc=f'Detect OBBs {source_weight}:{os.path.basename(video_path)}', unit='frame'):
            ok, img = cap.read()
            if not ok:
                break
            batch_imgs.append(img)
            batch_fids.append(int(first_frame + rel_idx))
            if len(batch_imgs) >= batch_size:
                flush_batch()
        flush_batch()
        cap.release()

        if batch_idx == 0:
            return make_empty_candidate_df()

        part_files = sorted(os.listdir(temp_dir))
        df = pd.concat([pd.read_pickle(os.path.join(temp_dir, f)) for f in part_files], ignore_index=True)
        if not df.empty:
            df = df.sort_values(['frame', 'det_index']).reset_index(drop=True)
            df = normalize_candidate_df_direction(df, reference_is_yolo_class=True)
        return df
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main() -> None:
    from ultralytics import YOLO
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python obb_detection.py config.yaml')
    cfg = load_config(sys.argv[1])
    if cfg.get('skip_detection', False):
        print('skip_detection is True; exiting.')
        return

    analysis = cfg.get('analysis', {}) or {}
    session_path = str(cfg['SESSION_PATH'])
    video_path_in = str(cfg['TRACKING_VIDEO_PATH'])
    conf_th = float(analysis.get('CONF', 0.1))
    nms_iou = float(analysis.get('NMS_IOU', 0.80))
    device = resolve_device(analysis.get('DEVICE', 'auto'), purpose='detection')
    orig_first_frame = int(analysis.get('FIRST_FRAME', 0))
    orig_last_frame = int(analysis.get('LAST_FRAME', -1))
    preview_interval = int(cfg.get('PREVIEW_INTERVAL', 100))
    skip_pose_preview_frames = bool(analysis.get('SKIP_DETECT_PREVIEW', False))
    batch_size_config = analysis.get('BATCH_SIZE', 'auto')
    num_objects = int(cfg.get('NUM_OBJECTS', 0))
    max_det = int(analysis.get("MAX_DET", 10000)) if cfg.get("VARIABLE_NUM_OBJECTS", False) else (max(300, num_objects * 2) if num_objects > 0 else 300)
    if max_det < 1:
        raise ValueError("analysis.MAX_DET must be positive.")

    model_name = os.path.splitext(cfg.get('training', {}).get('PRETRAINED_MODEL', ''))[0]
    model_name = resolve_model_name(model_name).split('.')[0]
    dataset_name = resolve_existing_experiment_dir_name(
        session_path,
        model_name,
        experiment_dir_name_from_cfg(cfg),
        stages=('tracking', 'training'),
        warn_fn=print,
    )
    direction_weights = list(dict.fromkeys(parse_weight_spec(analysis.get('WEIGHT', 'last'))))
    direction_weights = deduplicate_best_epoch_weights(
        session_path, model_name, dataset_name, direction_weights,
    )
    direction_jobs = resolve_tracking_jobs(direction_weights)

    all_jobs = direction_jobs

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
        warn_if_frame_count_adjusted(frame_info, label=f'detection video={video_name}')
        total_all = frame_info.usable_frame_count
        frame_width = frame_info.width
        frame_height = frame_info.height
        inference_image_size = video_inference_image_size(frame_width, frame_height)
        video_batch_size = resolve_batch_size(
            batch_size_config, inference_image_size, device, mode='infer',
        )
        print(
            f'[INFO] detection video={video_name}, frame_size={frame_width}x{frame_height}, '
            f'usable_frames={total_all}, imgsz={inference_image_size}, batch_size={video_batch_size}'
        )
        first_frame, last_frame = clamp_frame_range_to_usable_count(
            orig_first_frame, orig_last_frame, total_all,
        )
        video_infos.append({
            'path': video_path,
            'name': video_name,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'image_size': inference_image_size,
            'batch_size': video_batch_size,
        })

    direction_models = {}
    for weight in list(dict.fromkeys(job['source_weight'] for job in all_jobs)):
        direction_models[weight] = YOLO(build_weight_path(session_path, model_name, dataset_name, weight))

    for info in video_infos:
        for direction_job in all_jobs:
            run_name = str(direction_job['run_name'])
            out_dir = build_tracking_out_dir(session_path, model_name, dataset_name, run_name, info['name'])
            if cfg.get('VARIABLE_NUM_OBJECTS', False):
                out_dir = os.path.join(out_dir, 'variable')
            os.makedirs(out_dir, exist_ok=True)
            blobs_direction_pickle = tracking_artifact_path(out_dir, 'all_blobs.pkl')
            preview_direction_dir = os.path.join(out_dir, 'preview')

            if pickle_has_data(blobs_direction_pickle):
                print(f'[SKIP] Import existing direction pickle without re-filtering: {blobs_direction_pickle}')
                direction_rows_df = load_blob_pickle(blobs_direction_pickle)
            else:
                print(f'Direction estimation... [{run_name}:{info["name"]}]')
                weight = str(direction_job['source_weight'])
                model = direction_models[weight]
                direction_df = run_with_oom_retry(
                    lambda bs, _m=model, _w=weight: collect_candidate_df(
                        _m, info['path'], info['first_frame'], info['last_frame'],
                        device, conf_th, _w, image_size=info['image_size'], batch_size=bs,
                        nms_iou=nms_iou, max_det=max_det,
                    ),
                    info['batch_size'],
                )
                filtered_direction_df = filter_candidate_df_for_tracking(direction_df, conf_th)
                direction_rows_df = candidate_df_to_blob_df(filtered_direction_df, 'direction')
                save_blob_pickle(blobs_direction_pickle, direction_rows_df)

            if not skip_pose_preview_frames:
                render_blob_preview(info['path'], direction_rows_df, preview_direction_dir, info['first_frame'], info['last_frame'], preview_interval, 'all_blobs')


if __name__ == '__main__':
    main()
