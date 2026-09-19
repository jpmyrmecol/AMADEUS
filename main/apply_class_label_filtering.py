# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Apply reviewed label deletions to the single-animal image dataset."""

import csv
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from batch_utils import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gui.color import Color, DELETION_COLOR, OBB_COLOR
from path_utils import resolve_config_paths

try:
    import torch as _torch
    import torch.nn.functional as _F
    _CUDA_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _torch = None
    _F = None
    _CUDA_AVAILABLE = False


MANIFEST_BASE_FIELDNAMES = [
    "pool_id", "frame", "blob_index", "traj_id", "direction_valid", "class_id", "class_name",
    "head_angle_deg", "direction_dx", "direction_dy",
    "center_x", "center_y", "axis_length",
    "obb_aspect_ratio", "min_obb_aspect_ratio",
    "rect_x", "rect_y", "rect_w", "rect_h",
    "occlusion_state", "is_pasted", "overlap_pixels", "overlap_ratio",
    "crop_image", "crop_mask",
]

UNSUPPORTED_MANIFEST_FIELDS = {
    "axis_neg", "axis_pos", "axis_neg_x", "axis_neg_y", "axis_pos_x", "axis_pos_y",
}


def resolve_manifest_fieldnames(rows: Sequence[dict]) -> List[str]:
    fieldnames = list(MANIFEST_BASE_FIELDNAMES)
    seen = set(fieldnames)
    for row in rows:
        for key in row.keys():
            if str(key) in UNSUPPORTED_MANIFEST_FIELDS:
                raise ValueError(f"Unsupported manifest field: {key}")
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    return fieldnames


LOW_OBB_ASPECT_REASON = "low_obb_aspect"
REFINE_COLOR = DELETION_COLOR
SINGLE_ANIMAL_IMAGES_DIR_NAME = "single_animal_images"


@dataclass
class DeleteInfo:
    reasons: List[str]
    gt_class_id: str = ""
    pred_class_id: str = ""
    pred_conf: str = ""
    best_obb_iou: str = ""
    angle_diff_deg: str = ""
    obb_aspect_ratio: str = ""
    min_obb_aspect_ratio: str = ""


@dataclass
class PreviewMark:
    draw_box: bool = False
    draw_label: bool = False
    deleted: bool = False
    info: Optional[DeleteInfo] = None
    decision: str = ""
    run_id: int = -1
    traj_id: int = -1
    run_frame_count: int = 0
    flagged_frame_count: int = 0
    run_blob_count: int = 0
    flagged_blob_count: int = 0
    ratio_text: str = ""


class Progress:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def bar(self, iterable, desc: str, total=None, unit: str = "item"):
        return tqdm(iterable, desc=desc, total=total, unit=unit, disable=not self.enabled)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return resolve_config_paths(cfg)


def resolve_num_workers(cfg: dict, section_name: str | None = None, default: int = 8, cap: int | None = None) -> int:
    """Resolve worker count from GUI/config settings.

    Priority:
      1. section-specific NUM_WORKERS / WORKERS / N_WORKERS / workers / num_workers
      2. top-level NUM_WORKERS / WORKERS / N_WORKERS / workers / num_workers
      3. auto-computed from CPU/RAM
    """
    from batch_utils import auto_num_workers as _auto
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
        workers = _auto("process")
    else:
        workers = int(value)
    workers = max(1, workers)
    if cap is not None:
        workers = min(workers, max(1, int(cap)))
    return workers


def parse_run_delete_ratio(value, default=Fraction(2, 5)) -> Fraction:
    if isinstance(default, Fraction):
        default_threshold = default
    else:
        default_threshold = Fraction(str(float(default))).limit_denominator()

    if isinstance(value, Fraction):
        threshold = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            threshold = default_threshold
        elif "/" in s:
            threshold = Fraction(s)
        else:
            threshold = Fraction(str(float(s))).limit_denominator()
    elif value is None:
        threshold = default_threshold
    else:
        threshold = Fraction(str(float(value))).limit_denominator()

    if threshold <= 0 or threshold > 1:
        raise ValueError(f"RUN_DELETE_RATIO must be in (0, 1], got {value!r}")
    return threshold


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


def parse_optional_float(value) -> Optional[float]:
    s = str(value).strip()
    if not s:
        return None
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def append_unique_reason(items: List[str], reason: str) -> None:
    if reason not in items:
        items.append(reason)


def merge_delete_info(primary: DeleteInfo, extra: DeleteInfo) -> DeleteInfo:
    reasons = list(primary.reasons)
    for reason in extra.reasons:
        append_unique_reason(reasons, reason)
    return DeleteInfo(
        reasons=reasons,
        gt_class_id=primary.gt_class_id or extra.gt_class_id,
        pred_class_id=primary.pred_class_id or extra.pred_class_id,
        pred_conf=primary.pred_conf or extra.pred_conf,
        best_obb_iou=primary.best_obb_iou or extra.best_obb_iou,
        angle_diff_deg=primary.angle_diff_deg or extra.angle_diff_deg,
        obb_aspect_ratio=primary.obb_aspect_ratio or extra.obb_aspect_ratio,
        min_obb_aspect_ratio=primary.min_obb_aspect_ratio or extra.min_obb_aspect_ratio,
    )


def merge_delete_maps(
    base: Dict[Tuple[int, int], DeleteInfo],
    extra: Dict[Tuple[int, int], DeleteInfo],
) -> Dict[Tuple[int, int], DeleteInfo]:
    merged = dict(base)
    for key, info in extra.items():
        merged[key] = merge_delete_info(merged[key], info) if key in merged else info
    return merged


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_refine_dir_name(cfg: dict) -> str:
    name = str(cfg.get("REFINE_DIR_NAME", "refine")).strip()
    if not name:
        name = "refine"
    if os.path.basename(name) != name or name in {".", ".."}:
        raise ValueError(f"Invalid REFINE_DIR_NAME: {name!r}")
    return name


def is_refine_workspace_name(name: str) -> bool:
    return name == "refine" or name.startswith("refine_add") or name == "_iterative_refine_tmp"


def reset_without_crossing_output_dir(without_crossing_dir: str, progress: Progress) -> None:
    if not os.path.isdir(without_crossing_dir):
        os.makedirs(without_crossing_dir, exist_ok=True)
        return
    names = [name for name in os.listdir(without_crossing_dir) if not is_refine_workspace_name(name)]
    for name in progress.bar(names, desc="Reset output dir"):
        path = os.path.join(without_crossing_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def backup_without_crossing_to_refine_original(
    without_crossing_dir: str,
    refine_root: str,
    original_dir: str,
    progress: Progress,
) -> bool:
    ensure_dir(refine_root)

    existing_names = []
    if os.path.isdir(without_crossing_dir):
        existing_names = sorted(name for name in os.listdir(without_crossing_dir) if not is_refine_workspace_name(name))

    if os.path.isdir(original_dir):
        return False

    if not existing_names:
        return False

    ensure_dir(original_dir)
    for name in progress.bar(existing_names, desc="Backup single_animal_images to refine/original"):
        shutil.move(os.path.join(without_crossing_dir, name), os.path.join(original_dir, name))
    return True

def parse_frame_id_from_name(name: str) -> int:
    m = re.match(r"frame_(\d+)\.png$", name)
    if not m:
        raise ValueError(f"Invalid frame image name: {name}")
    return int(m.group(1))


def select_preview_frame_ids(frame_ids: Sequence[int], num_preview_frames: int, frame_interval: int) -> Set[int]:
    ordered = sorted(dict.fromkeys(int(frame_id) for frame_id in frame_ids))
    if num_preview_frames <= 0 or not ordered:
        return set()

    interval = max(1, int(frame_interval))
    selected: List[int] = []
    last_selected: Optional[int] = None

    for frame_id in ordered:
        if last_selected is None or frame_id - last_selected >= interval:
            selected.append(frame_id)
            last_selected = frame_id
            if len(selected) >= num_preview_frames:
                break

    return set(selected)


def read_manifest_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: str, rows: Sequence[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    fieldnames = resolve_manifest_fieldnames(rows)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def read_delete_map(csv_path: str, progress: Progress) -> Dict[Tuple[int, int], DeleteInfo]:
    delete_map: Dict[Tuple[int, int], DeleteInfo] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in progress.bar(rows, desc="Load delete map"):
        key = (int(row["frame"]), int(row["blob_index"]))
        reasons = [x for x in str(row.get("reasons", "")).split("|") if x]
        delete_map[key] = DeleteInfo(
            reasons=reasons or ["refine_delete"],
            gt_class_id=str(row.get("gt_class_id", "")),
            pred_class_id=str(row.get("pred_class_id", "")),
            pred_conf=str(row.get("pred_conf", "")),
            best_obb_iou=str(row.get("best_obb_iou", "")),
            angle_diff_deg=str(row.get("angle_diff_deg", "")),
            obb_aspect_ratio=str(row.get("obb_aspect_ratio", "")),
            min_obb_aspect_ratio=str(row.get("min_obb_aspect_ratio", "")),
        )
    return delete_map


def group_manifest_rows(manifest_rows: Sequence[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in manifest_rows:
        grouped[int(row["frame"])].append(dict(row))
    return dict(grouped)


def obb_aspect_ratio_from_points(pts: np.ndarray) -> Optional[float]:
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


def read_frame_size(original_dir: str, frame_id: int) -> Tuple[int, int]:
    image_path = os.path.join(original_dir, "images", f"frame_{int(frame_id):06d}.png")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    return int(w), int(h)


def build_low_obb_aspect_delete_map(
    original_dir: str,
    manifest_rows: Sequence[dict],
    min_obb_aspect_ratio: float,
    progress: Progress,
) -> Dict[Tuple[int, int], DeleteInfo]:
    delete_map: Dict[Tuple[int, int], DeleteInfo] = {}
    by_frame = group_manifest_rows(manifest_rows)
    for frame_id in progress.bar(sorted(by_frame.keys()), desc="Find low OBB aspect blobs", unit="frame"):
        rows = by_frame[int(frame_id)]
        label_path = os.path.join(original_dir, "labels", f"frame_{int(frame_id):06d}.txt")
        label_lines = read_lines_if_exists(label_path)
        if label_lines and len(label_lines) != len(rows):
            raise RuntimeError(
                f"Label count mismatch for frame {frame_id}: labels={len(label_lines)} manifest={len(rows)}"
            )

        frame_w = frame_h = None
        for row_index, row in enumerate(rows):
            aspect = parse_optional_float(row.get("obb_aspect_ratio", ""))
            if aspect is None and label_lines:
                if frame_w is None or frame_h is None:
                    frame_w, frame_h = read_frame_size(original_dir, int(frame_id))
                line = label_lines[row_index] if row_index < len(label_lines) else ""
                if str(line).strip():
                    _, pts = parse_obb_label_line_with_class(line, frame_w, frame_h)
                    aspect = obb_aspect_ratio_from_points(pts)

            if aspect is None or not np.isfinite(aspect):
                continue
            if float(aspect) >= float(min_obb_aspect_ratio):
                continue

            key = (int(row["frame"]), int(row["blob_index"]))
            delete_map[key] = DeleteInfo(
                reasons=[LOW_OBB_ASPECT_REASON],
                gt_class_id=str(row.get("class_id", "")),
                obb_aspect_ratio=f"{float(aspect):.6f}",
                min_obb_aspect_ratio=f"{float(min_obb_aspect_ratio):.6f}",
            )
    return delete_map


def read_pre_direction_run_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def build_run_maps_from_csv(
    run_rows: Sequence[dict],
    manifest_rows: Sequence[dict],
) -> Tuple[List[dict], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[int, dict]]:
    manifest_key_to_run: Dict[Tuple[int, int], int] = {}
    blob_key_to_run: Dict[Tuple[int, int], int] = {}
    manifest_row_keys_by_run: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    runs: List[dict] = []
    run_meta_by_id: Dict[int, dict] = {}

    manifest_by_traj: Dict[int, List[dict]] = defaultdict(list)
    for row in manifest_rows:
        traj_id = int(row["traj_id"])
        manifest_by_traj[traj_id].append(dict(row))

    for run_row in run_rows:
        run_id = int(run_row["run_id"])
        traj_id = int(run_row["traj_id"])
        start_frame = int(run_row["start_frame"])
        end_frame = int(run_row["end_frame"])
        run_frame_count = int(run_row["run_frame_count"])
        frames = list(range(start_frame, end_frame + 1)) if end_frame >= start_frame else []

        manifest_row_keys: List[Tuple[int, int]] = []
        for row in manifest_by_traj.get(traj_id, []):
            frame = int(row["frame"])
            if start_frame <= frame <= end_frame:
                key = (frame, int(row["blob_index"]))
                if key in manifest_key_to_run:
                    prev_run_id = manifest_key_to_run[key]
                    raise RuntimeError(
                        f"Manifest row {key} matched multiple runs: {prev_run_id} and {run_id}"
                    )
                manifest_key_to_run[key] = run_id
                blob_key_to_run[key] = run_id
                manifest_row_keys.append(key)

        run = {
            "run_id": run_id,
            "traj_id": traj_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": set(frames),
            "run_frame_count": run_frame_count,
            "blob_row_keys": list(manifest_row_keys),
            "manifest_row_keys": list(manifest_row_keys),
        }
        runs.append(run)
        run_meta_by_id[run_id] = run
        manifest_row_keys_by_run[run_id] = manifest_row_keys

    return runs, manifest_key_to_run, blob_key_to_run, run_meta_by_id


def filter_valid_run_rows(run_rows: Sequence[dict], min_valid_frames: int) -> List[dict]:
    filtered: List[dict] = []
    for row in run_rows:
        run_frame_count = int(row["run_frame_count"])
        if run_frame_count <= min_valid_frames:
            continue
        filtered.append(dict(row))
    return filtered


def write_run_delete_summary_csv(
    path: str,
    run_rows: Sequence[dict],
    runs: Sequence[dict],
    delete_map: Dict[Tuple[int, int], DeleteInfo],
    progress: Progress,
) -> List[dict]:
    run_meta_by_id = {int(run["run_id"]): dict(run) for run in runs}
    rows: List[dict] = []
    for run_row in progress.bar(run_rows, desc="Summarize refine deletes by run", total=len(run_rows), unit="run"):
        row = dict(run_row)
        run_id = int(row["run_id"])
        run = run_meta_by_id.get(run_id)
        if run is None:
            raise RuntimeError(f"Missing run metadata for run_id={run_id}")
        frames = sorted(int(x) for x in run["frames"])
        delete_blob_keys = [key for key in run["blob_row_keys"] if key in delete_map]
        delete_frames = sorted({int(frame) for frame, _ in delete_blob_keys})
        run_blob_count = len(run["blob_row_keys"])
        delete_blob_count = len(delete_blob_keys)
        run_frame_count = int(run["run_frame_count"])
        delete_frame_count = len(delete_frames)
        row.update({
            "run_blob_count": run_blob_count,
            "manifest_blob_count": len(run["manifest_row_keys"]),
            "delete_blob_count": delete_blob_count,
            "non_delete_blob_count": max(0, run_blob_count - delete_blob_count),
            "delete_frame_count": delete_frame_count,
            "non_delete_frame_count": max(0, run_frame_count - delete_frame_count),
            "delete_blob_ratio": f"{delete_blob_count}/{run_blob_count}" if run_blob_count > 0 else "0/0",
            "delete_blob_ratio_float": f"{(delete_blob_count / run_blob_count):.6f}" if run_blob_count > 0 else "0.000000",
            "delete_frame_ratio": f"{delete_frame_count}/{run_frame_count}" if run_frame_count > 0 else "0/0",
            "delete_frame_ratio_float": f"{(delete_frame_count / run_frame_count):.6f}" if run_frame_count > 0 else "0.000000",
            "delete_frame_list": "|".join(str(x) for x in delete_frames),
            "run_frame_list": "|".join(str(x) for x in frames),
        })
        rows.append(row)

    fieldnames = list(run_rows[0].keys()) if run_rows else [
        "run_id", "traj_id", "start_frame", "end_frame", "run_frame_count",
    ]
    for field in [
        "run_blob_count", "manifest_blob_count", "delete_blob_count", "non_delete_blob_count",
        "delete_frame_count", "non_delete_frame_count", "delete_blob_ratio", "delete_blob_ratio_float",
        "delete_frame_ratio", "delete_frame_ratio_float", "delete_frame_list", "run_frame_list",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv_rows(path, fieldnames, rows)
    return rows


def build_policy_maps(
    manifest_rows: Sequence[dict],
    run_rows: Sequence[dict],
    delete_map: Dict[Tuple[int, int], DeleteInfo],
    run_summary_rows: Sequence[dict],
    run_delete_ratio: Fraction,
    progress: Progress,
) -> Tuple[Dict[Tuple[int, int], DeleteInfo], Dict[Tuple[int, int], PreviewMark], Dict[str, int]]:
    runs, manifest_key_to_run, blob_key_to_run, run_meta_by_id = build_run_maps_from_csv(run_rows, manifest_rows)
    if not runs:
        stats = {
            "run_count": 0,
            "run_delete_keep": 0,
            "run_delete_expand": 0,
            "run_delete_reject": 0,
            "run_deleted_added": 0,
            "run_deleted_rejected": len(delete_map),
            "run_deleted_outside_valid_runs": len(delete_map),
            "run_deleted_total": 0,
        }
        preview_marks = {
            key: PreviewMark(
                draw_box=True,
                draw_label=True,
                deleted=False,
                info=info,
                decision="outside_valid_run_keep",
            )
            for key, info in delete_map.items()
        }
        return {}, preview_marks, stats

    threshold = run_delete_ratio
    kept_runs = 0
    expanded_runs = 0
    rejected_runs = 0
    added_count = 0
    rejected_count = 0
    outside_valid_run_count = 0
    adjusted_delete_map: Dict[Tuple[int, int], DeleteInfo] = {}
    preview_marks: Dict[Tuple[int, int], PreviewMark] = {}

    summary_by_run_id = {int(row["run_id"]): dict(row) for row in run_summary_rows}

    for key, info in delete_map.items():
        if key not in blob_key_to_run:
            outside_valid_run_count += 1
            rejected_count += 1
            preview_marks[key] = PreviewMark(
                draw_box=True,
                draw_label=True,
                deleted=False,
                info=info,
                decision="outside_valid_run_keep",
            )

    for run in progress.bar(runs, desc="Apply run delete policy", total=len(runs), unit="run"):
        run_id = int(run["run_id"])
        summary = summary_by_run_id.get(run_id)
        if summary is None:
            raise RuntimeError(f"Missing run summary for run_id={run_id}")

        flagged_blob_count = int(summary["delete_blob_count"])
        run_blob_count = int(summary["run_blob_count"])
        flagged_frame_count = int(summary["delete_frame_count"])
        run_frame_count = int(summary["run_frame_count"])
        flagged_keys = [key for key in run["blob_row_keys"] if key in delete_map]
        traj_id = int(run["traj_id"])

        if run_blob_count > 0:
            ratio = Fraction(flagged_blob_count, run_blob_count)
            ratio_text = f"{flagged_blob_count}/{run_blob_count}"
        else:
            ratio = Fraction(flagged_frame_count, run_frame_count) if run_frame_count > 0 else Fraction(0, 1)
            ratio_text = f"{flagged_frame_count}/{run_frame_count}"

        if run_blob_count <= 0 or flagged_blob_count <= 0:
            kept_runs += 1
            continue

        if ratio >= threshold:
            expanded_runs += 1
            template = delete_map[flagged_keys[0]]
            raw_flagged_key_set = set(flagged_keys)
            for key in run["manifest_row_keys"]:
                if key in delete_map:
                    base = delete_map[key]
                    reasons = list(base.reasons)
                    draw_label = True
                    decision = "run_delete_ge_threshold_raw"
                else:
                    base = template
                    reasons = list(template.reasons)
                    if "refine_run_delete_ge_threshold" not in reasons:
                        reasons.append("refine_run_delete_ge_threshold")
                    added_count += 1
                    draw_label = False
                    decision = "run_delete_ge_threshold_collateral"
                info = DeleteInfo(
                    reasons=reasons,
                    gt_class_id=base.gt_class_id,
                    pred_class_id=base.pred_class_id,
                    pred_conf=base.pred_conf,
                    best_obb_iou=base.best_obb_iou,
                    angle_diff_deg=base.angle_diff_deg,
                )
                adjusted_delete_map[key] = info
                preview_marks[key] = PreviewMark(
                    draw_box=True,
                    draw_label=draw_label,
                    deleted=True,
                    info=(info if key in raw_flagged_key_set else None),
                    decision=decision,
                    run_id=run_id,
                    traj_id=traj_id,
                    run_frame_count=run_frame_count,
                    flagged_frame_count=flagged_frame_count,
                    run_blob_count=run_blob_count,
                    flagged_blob_count=flagged_blob_count,
                    ratio_text=ratio_text,
                )

            for key in flagged_keys:
                if key in preview_marks:
                    continue
                info = delete_map[key]
                preview_marks[key] = PreviewMark(
                    draw_box=True,
                    draw_label=True,
                    deleted=False,
                    info=info,
                    decision="run_delete_ge_threshold_raw_nonmanifest",
                    run_id=run_id,
                    traj_id=traj_id,
                    run_frame_count=run_frame_count,
                    flagged_frame_count=flagged_frame_count,
                    run_blob_count=run_blob_count,
                    flagged_blob_count=flagged_blob_count,
                    ratio_text=ratio_text,
                )
        else:
            rejected_runs += 1
            rejected_count += len(flagged_keys)
            for key in flagged_keys:
                preview_marks[key] = PreviewMark(
                    draw_box=True,
                    draw_label=True,
                    deleted=False,
                    info=delete_map[key],
                    decision="run_keep_lt_threshold",
                    run_id=run_id,
                    traj_id=traj_id,
                    run_frame_count=run_frame_count,
                    flagged_frame_count=flagged_frame_count,
                    run_blob_count=run_blob_count,
                    flagged_blob_count=flagged_blob_count,
                    ratio_text=ratio_text,
                )

    stats = {
        "run_count": len(runs),
        "run_delete_keep": kept_runs,
        "run_delete_expand": expanded_runs,
        "run_delete_reject": rejected_runs,
        "run_deleted_added": added_count,
        "run_deleted_rejected": rejected_count,
        "run_deleted_outside_valid_runs": outside_valid_run_count,
        "run_deleted_total": len(adjusted_delete_map),
    }
    return adjusted_delete_map, preview_marks, stats

def filter_manifest_rows(rows: Sequence[dict], delete_map: Dict[Tuple[int, int], DeleteInfo], progress: Progress) -> List[dict]:
    kept: List[dict] = []
    for row in progress.bar(rows, desc="Filter manifest"):
        key = (int(row["frame"]), int(row["blob_index"]))
        if key in delete_map:
            continue
        kept.append(dict(row))
    return kept


def read_lines_if_exists(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_text_lines(path: str, lines: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def move_file_fast(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)


def read_background(bg_path: str, frame_shape: Tuple[int, int]) -> np.ndarray:
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bg is None:
        raise FileNotFoundError(bg_path)
    h, w = frame_shape
    if bg.shape[:2] != (h, w):
        bg = cv2.resize(bg, (w, h))
    return bg


def load_mask_image(mask_path: str, expected_h: int, expected_w: int) -> Optional[np.ndarray]:
    if not os.path.exists(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != (expected_h, expected_w):
        return None
    return mask


def build_remove_mask(
    frame_shape: Tuple[int, int],
    deleted_rows: Sequence[dict],
    original_dir: str,
) -> np.ndarray:
    h, w = frame_shape
    remove_mask = np.zeros((h, w), dtype=np.uint8)
    for row in deleted_rows:
        rect_x = int(row["rect_x"])
        rect_y = int(row["rect_y"])
        rect_w = int(row["rect_w"])
        rect_h = int(row["rect_h"])
        mask_path = os.path.join(original_dir, str(row["crop_mask"]))
        crop_mask = load_mask_image(mask_path, rect_h, rect_w)
        if crop_mask is None:
            continue

        x1 = max(0, rect_x)
        y1 = max(0, rect_y)
        x2 = min(w, rect_x + rect_w)
        y2 = min(h, rect_y + rect_h)
        if x1 >= x2 or y1 >= y2:
            continue

        mx1 = x1 - rect_x
        my1 = y1 - rect_y
        mx2 = mx1 + (x2 - x1)
        my2 = my1 + (y2 - y1)
        roi = crop_mask[my1:my2, mx1:mx2]
        if roi.size == 0:
            continue
        remove_mask[y1:y2, x1:x2] = np.maximum(remove_mask[y1:y2, x1:x2], roi)
    return remove_mask


# GPU caches: (ksize, sigma) -> kernel tensor; (background_path, h, w) -> bg tensor.
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


def apply_background_erase(
    src_img: np.ndarray,
    bg: np.ndarray,
    remove_mask: np.ndarray,
    edge_blur_ksize: int,
    edge_blur_sigma: float,
    background_path: str = "",
) -> np.ndarray:
    if _CUDA_AVAILABLE and _torch is not None:
        try:
            h, w = remove_mask.shape[:2]
            keep = (remove_mask == 0).astype(np.float32)
            keep_t = _torch.from_numpy(keep).cuda().unsqueeze(0).unsqueeze(0)
            if edge_blur_ksize > 1:
                keep_t = _F.conv2d(
                    keep_t,
                    _get_gauss_kernel_gpu(edge_blur_ksize, edge_blur_sigma),
                    padding=edge_blur_ksize // 2,
                )
            alpha_t = keep_t.squeeze().unsqueeze(-1)
            fr_t = _torch.from_numpy(src_img).float().cuda()
            bg_key = (background_path, h, w) if background_path else id(bg)
            if bg_key not in _GPU_BG_CACHE:
                _GPU_BG_CACHE[bg_key] = _torch.from_numpy(bg.astype(np.float32)).cuda()
            bg_t = _GPU_BG_CACHE[bg_key]
            return (fr_t * alpha_t + bg_t * (1.0 - alpha_t)).clamp(0, 255).byte().cpu().numpy()
        except Exception:
            pass
    keep = (remove_mask == 0).astype(np.float32)
    alpha = cv2.GaussianBlur(keep, (edge_blur_ksize, edge_blur_ksize), edge_blur_sigma)[..., None]
    blended = src_img.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_obb(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    poly = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    poly = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, color, thickness, cv2.LINE_AA)


def ensure_clockwise_obb(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    area2 = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area2 += x1 * y2 - x2 * y1
    if area2 > 0:
        pts = pts[::-1]
    return pts.astype(np.float32)


def class_id_to_unit_vec(class_id: int) -> np.ndarray:
    angle_deg = float(int(class_id) % 8) * 45.0
    th = np.deg2rad(angle_deg)
    return np.asarray([np.sin(th), -np.cos(th)], dtype=np.float32)


def draw_direction_triangle_for_obb(
    img: np.ndarray,
    pts: np.ndarray,
    class_id: int,
    color: Tuple[int, int, int],
    alpha: float = 0.6,
    outline_thickness: int = 1,
    scale: float = 1.2,
) -> None:
    pts = ensure_clockwise_obb(pts)
    direction = class_id_to_unit_vec(int(class_id))

    candidates: List[Tuple[float, int, int, float, np.ndarray]] = []
    edge_lengths: List[float] = []
    edges: List[Tuple[int, int, float]] = []
    for i in range(4):
        j = (i + 1) % 4
        edge_len = float(np.linalg.norm(pts[j] - pts[i]))
        edges.append((i, j, edge_len))
        edge_lengths.append(edge_len)
    if not edge_lengths:
        return

    min_len = min(edge_lengths)
    tol = max(1e-6, min_len * 0.05)
    center = np.mean(pts, axis=0)
    for i, j, edge_len in edges:
        if abs(edge_len - min_len) > tol or edge_len <= 0.0:
            continue
        p0 = pts[i]
        p1 = pts[j]
        edge = p1 - p0
        normal = np.asarray([edge[1], -edge[0]], dtype=np.float32) / float(edge_len)
        midpoint = 0.5 * (p0 + p1)
        if float(np.dot(midpoint - center, normal)) < 0.0:
            normal = -normal
        candidates.append((float(np.dot(normal, direction)), i, j, edge_len, normal))

    if not candidates:
        return

    _, i, j, base_len, normal = max(candidates, key=lambda x: x[0])
    p0 = pts[i].astype(np.float32)
    p1 = pts[j].astype(np.float32)
    base_mid = 0.5 * (p0 + p1)
    height = float(scale) * (np.sqrt(3.0) / 2.0) * float(base_len)
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


def parse_obb_label_line_with_class(line: str, width: int, height: int) -> Tuple[int, np.ndarray]:
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"Invalid OBB label line: {line!r}")
    class_id = int(float(parts[0]))
    pts = np.asarray(list(map(float, parts[1:])), dtype=np.float32).reshape(4, 2)
    pts[:, 0] *= float(width)
    pts[:, 1] *= float(height)
    return class_id, pts


def draw_preview_from_label_lines(
    img: np.ndarray,
    label_lines: Sequence[str],
    color: Color = OBB_COLOR,
) -> np.ndarray:
    preview = np.ascontiguousarray(img.copy())
    h, w = preview.shape[:2]
    for line in label_lines:
        if not str(line).strip():
            continue
        class_id, pts = parse_obb_label_line_with_class(line, w, h)
        draw_obb(preview, pts, color, 2)
        draw_direction_triangle_for_obb(preview, pts, class_id, color, alpha=0.6, outline_thickness=1, scale=1.2)
    return preview


def draw_refine_marks_on_preview(
    preview: np.ndarray,
    rows: Sequence[dict],
    original_label_lines: Sequence[str],
    preview_marks: Dict[Tuple[int, int], PreviewMark],
    conf_thresh: float,
) -> None:
    h, w = preview.shape[:2]
    parsed_labels: List[Optional[Tuple[int, np.ndarray]]] = []
    for line in original_label_lines:
        if str(line).strip():
            parsed_labels.append(parse_obb_label_line_with_class(line, w, h))
        else:
            parsed_labels.append(None)

    for row_index, row in enumerate(rows):
        key = (int(row["frame"]), int(row["blob_index"]))
        mark = preview_marks.get(key)
        if mark is None or not mark.draw_box:
            continue

        label = parsed_labels[row_index] if row_index < len(parsed_labels) else None
        if label is not None:
            class_id, pts = label
            draw_obb(preview, pts, REFINE_COLOR, 2)
            draw_direction_triangle_for_obb(
                preview,
                pts,
                class_id,
                REFINE_COLOR,
                alpha=0.6,
                outline_thickness=1,
                scale=1.2,
            )
        else:
            x = int(row["rect_x"])
            y = int(row["rect_y"])
            rw = int(row["rect_w"])
            rh = int(row["rect_h"])
            cv2.rectangle(preview, (x, y), (x + rw, y + rh), REFINE_COLOR, 2, cv2.LINE_AA)


def draw_text(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int], scale: float = 0.4) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_multiline_text(img: np.ndarray, lines: Sequence[str], x: int, y: int, color: Tuple[int, int, int], scale: float = 0.4) -> None:
    line_step = max(12, int(round(14 * scale / 0.4)))
    base_y = max(12, y)
    for i, line in enumerate(lines):
        draw_text(img, str(line), x, base_y + i * line_step, color, scale)


def _empty_rewrite_reason_counts(
    rows: Sequence[dict],
    delete_flags: Sequence[bool],
    delete_map: Dict[Tuple[int, int], DeleteInfo],
    label_lines: Sequence[str],
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    if not rows:
        counts["no_manifest_rows"] += 1
    for row, deleted in zip(rows, delete_flags):
        if not deleted:
            continue
        key = (int(row["frame"]), int(row["blob_index"]))
        info = delete_map.get(key)
        reasons = list(info.reasons) if info is not None and info.reasons else ["refine_delete"]
        for reason in reasons:
            counts[str(reason)] += 1
    if not counts and not label_lines:
        counts["no_label_blobs"] += 1
    if not counts:
        counts["unknown_empty_frame"] += 1
    return dict(counts)


def _rewrite_one_frame_asset(args) -> Tuple[int, int, Dict[str, int]]:
    (
        frame_name,
        original_dir,
        without_crossing_dir,
        rows,
        delete_map,
        preview_marks,
        changed_frames,
        selected_preview_frames,
        background_path,
        background,
        edge_blur_ksize,
        edge_blur_sigma,
        conf_thresh,
    ) = args

    img_src = os.path.join(original_dir, "images")
    lbl_src = os.path.join(original_dir, "labels")
    mask_src = os.path.join(original_dir, "masks")

    img_dst = os.path.join(without_crossing_dir, "images")
    lbl_dst = os.path.join(without_crossing_dir, "labels")
    mask_dst = os.path.join(without_crossing_dir, "masks")
    prev_dst = os.path.join(without_crossing_dir, "preview")

    frame_id = parse_frame_id_from_name(frame_name)
    src_img_path = os.path.join(img_src, frame_name)
    src_lbl_path = os.path.join(lbl_src, frame_name.replace(".png", ".txt"))
    src_maskinfo_path = os.path.join(mask_src, frame_name.replace(".png", "_maskinfo.txt"))

    dst_img_path = os.path.join(img_dst, frame_name)
    dst_lbl_path = os.path.join(lbl_dst, frame_name.replace(".png", ".txt"))
    dst_maskinfo_path = os.path.join(mask_dst, frame_name.replace(".png", "_maskinfo.txt"))

    rows = list(rows)
    delete_flags = [(int(r["frame"]), int(r["blob_index"])) in delete_map for r in rows]
    deleted_rows = [dict(r) for r, deleted in zip(rows, delete_flags) if deleted]

    label_lines = read_lines_if_exists(src_lbl_path)
    mask_lines = read_lines_if_exists(src_maskinfo_path)

    if len(label_lines) not in {0, len(rows)}:
        raise RuntimeError(f"Label count mismatch for frame {frame_id}: labels={len(label_lines)} manifest={len(rows)}")
    if len(mask_lines) not in {0, len(rows)}:
        raise RuntimeError(f"Mask info count mismatch for frame {frame_id}: masks={len(mask_lines)} manifest={len(rows)}")

    frame_changed = frame_id in changed_frames

    if frame_changed:
        kept_label_lines = [line for line, deleted in zip(label_lines, delete_flags) if not deleted]
        kept_mask_lines = [line for line, deleted in zip(mask_lines, delete_flags) if not deleted]
    else:
        kept_label_lines = list(label_lines)
        kept_mask_lines = list(mask_lines)

    if not kept_label_lines:
        return 0, 0, _empty_rewrite_reason_counts(rows, delete_flags, delete_map, label_lines)

    if frame_changed and deleted_rows:
        frame_img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
        if frame_img is None:
            raise FileNotFoundError(src_img_path)
        h, w = frame_img.shape[:2]

        bg = background
        if bg is None or bg.shape[:2] != (h, w):
            bg = read_background(background_path, (h, w))
        remove_mask = build_remove_mask((h, w), deleted_rows, original_dir)
        updated_img = apply_background_erase(frame_img, bg, remove_mask, edge_blur_ksize, edge_blur_sigma, background_path)

        cv2.imwrite(dst_img_path, updated_img)
        write_text_lines(dst_lbl_path, kept_label_lines)
        write_text_lines(dst_maskinfo_path, kept_mask_lines)
    else:
        move_file_fast(src_img_path, dst_img_path)
        if os.path.exists(src_lbl_path):
            move_file_fast(src_lbl_path, dst_lbl_path)
        else:
            write_text_lines(dst_lbl_path, label_lines)
        if os.path.exists(src_maskinfo_path):
            move_file_fast(src_maskinfo_path, dst_maskinfo_path)
        else:
            write_text_lines(dst_maskinfo_path, mask_lines)

    preview_written = 0
    if frame_id in selected_preview_frames:
        preview_base = cv2.imread(dst_img_path, cv2.IMREAD_COLOR)
        if preview_base is None:
            raise FileNotFoundError(dst_img_path)

        # Generate previews from the current image and current label lines every time.
        # Do not reuse original/preview/*.png, because those files may already contain
        # annotations from previous refinement passes.
        preview = draw_preview_from_label_lines(preview_base, kept_label_lines, color=OBB_COLOR)
        draw_refine_marks_on_preview(preview, rows, label_lines, preview_marks, conf_thresh)

        dst_prev_path = os.path.join(prev_dst, frame_name)
        cv2.imwrite(dst_prev_path, preview)
        preview_written = 1

    return 1, preview_written, {}


def rewrite_frame_assets(
    original_dir: str,
    without_crossing_dir: str,
    manifest_by_frame: Dict[int, List[dict]],
    delete_map: Dict[Tuple[int, int], DeleteInfo],
    preview_marks: Dict[Tuple[int, int], PreviewMark],
    cfg: dict,
    progress: Progress,
) -> Tuple[int, int, Dict[str, int]]:
    img_src = os.path.join(original_dir, "images")
    img_dst = os.path.join(without_crossing_dir, "images")
    lbl_dst = os.path.join(without_crossing_dir, "labels")
    mask_dst = os.path.join(without_crossing_dir, "masks")
    prev_dst = os.path.join(without_crossing_dir, "preview")
    ensure_dir(img_dst)
    ensure_dir(lbl_dst)
    ensure_dir(mask_dst)
    ensure_dir(prev_dst)

    frame_names = sorted([name for name in os.listdir(img_src) if name.endswith(".png")])

    edge_blur_ksize = int(cfg.get("EDGE_BLUR_KSIZE", 7))
    edge_blur_sigma = float(cfg.get("EDGE_BLUR_SIGMA", 11))
    if edge_blur_ksize < 1:
        edge_blur_ksize = 1
    if edge_blur_ksize % 2 == 0:
        edge_blur_ksize += 1

    conf_thresh = float(cfg.get("REFINE_CONF", 0.2))
    num_preview_frames = int(cfg.get("NUM_PREVIEW_FRAMES", 100))
    frame_interval = int(cfg.get("FRAME_INTERVAL", 5))
    selected_preview_frames = select_preview_frame_ids(
        [parse_frame_id_from_name(name) for name in frame_names],
        num_preview_frames=num_preview_frames,
        frame_interval=frame_interval,
    )

    changed_frames = {int(frame) for frame, _ in preview_marks.keys()}
    background_path = str(cfg["BACKGROUND_PATH"])
    background = None
    if frame_names:
        first_image = cv2.imread(os.path.join(img_src, frame_names[0]), cv2.IMREAD_COLOR)
        if first_image is None:
            raise FileNotFoundError(os.path.join(img_src, frame_names[0]))
        background = read_background(background_path, first_image.shape[:2])
    num_workers = resolve_num_workers(cfg, "apply_refine_deletions", default=resolve_num_workers(cfg, None, default=8))
    num_workers = max(1, int(num_workers))

    tasks = []
    for frame_name in frame_names:
        frame_id = parse_frame_id_from_name(frame_name)
        frame_rows = list(manifest_by_frame.get(frame_id, []))
        row_keys = {(int(r["frame"]), int(r["blob_index"])) for r in frame_rows}
        frame_delete_map = {key: value for key, value in delete_map.items() if key in row_keys}
        frame_preview_marks = {key: value for key, value in preview_marks.items() if key in row_keys}
        tasks.append((
            frame_name,
            original_dir,
            without_crossing_dir,
            frame_rows,
            frame_delete_map,
            frame_preview_marks,
            changed_frames,
            selected_preview_frames,
            background_path,
            background,
            edge_blur_ksize,
            edge_blur_sigma,
            conf_thresh,
        ))

    accel = "GPU" if _CUDA_AVAILABLE else f"CPUx{num_workers}"
    cv2.setNumThreads(1)
    saved_images = 0
    saved_previews = 0
    empty_frame_reason_counts: Counter[str] = Counter()
    skipped_empty_frames = 0
    if num_workers <= 1 or len(tasks) <= 1:
        for task in progress.bar(tasks, desc=f"Rewrite frame assets ({accel})"):
            n_img, n_prev, reason_counts = _rewrite_one_frame_asset(task)
            saved_images += int(n_img)
            saved_previews += int(n_prev)
            if int(n_img) == 0:
                skipped_empty_frames += 1
                empty_frame_reason_counts.update({str(k): int(v) for k, v in reason_counts.items()})
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_rewrite_one_frame_asset, task) for task in tasks]
            for fut in progress.bar(as_completed(futures), desc=f"Rewrite frame assets x{num_workers} ({accel})", total=len(futures), unit="frame"):
                n_img, n_prev, reason_counts = fut.result()
                saved_images += int(n_img)
                saved_previews += int(n_prev)
                if int(n_img) == 0:
                    skipped_empty_frames += 1
                    empty_frame_reason_counts.update({str(k): int(v) for k, v in reason_counts.items()})
    _GPU_BG_CACHE.clear()

    return saved_images, saved_previews, {
        "skipped_empty_frames": skipped_empty_frames,
        "empty_frame_removed_blobs": sum(empty_frame_reason_counts.values()),
        "empty_frame_reason_counts": dict(sorted(empty_frame_reason_counts.items())),
    }


def move_selected_files(src_dir: str, dst_dir: str, filenames: Sequence[str], desc: str, progress: Progress) -> None:
    ensure_dir(dst_dir)
    for name in progress.bar(sorted(set(filenames)), desc=desc):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)


def write_filtered_object_pool(original_dir: str, without_crossing_dir: str, manifest_rows: Sequence[dict], progress: Progress) -> None:
    obj_src = os.path.join(original_dir, "object_pool")
    img_src = os.path.join(obj_src, "images")
    mask_src = os.path.join(obj_src, "masks")

    obj_dst = os.path.join(without_crossing_dir, "object_pool")
    img_dst = os.path.join(obj_dst, "images")
    mask_dst = os.path.join(obj_dst, "masks")

    crop_images = [os.path.basename(row["crop_image"]) for row in manifest_rows]
    crop_masks = [os.path.basename(row["crop_mask"]) for row in manifest_rows]

    move_selected_files(img_src, img_dst, crop_images, "Move pool images", progress)
    move_selected_files(mask_src, mask_dst, crop_masks, "Move pool masks", progress)
    write_manifest(os.path.join(obj_dst, "manifest.csv"), manifest_rows)


def filter_blob_classification(
    original_dir: str,
    without_crossing_dir: str,
    delete_map: Dict[Tuple[int, int], DeleteInfo],
    preview_marks: Dict[Tuple[int, int], PreviewMark],
    progress: Progress,
) -> None:
    src = os.path.join(original_dir, "blob_classification.csv")
    if not os.path.exists(src):
        return
    dst = os.path.join(without_crossing_dir, "blob_classification.csv")
    with open(src, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    extra_fields = [
        "refine_deleted",
        "refine_delete_reasons",
        "refine_gt_class_id",
        "refine_pred_class_id",
        "refine_pred_conf",
        "refine_best_obb_iou",
        "refine_angle_diff_deg",
        "refine_obb_aspect_ratio",
        "refine_min_obb_aspect_ratio",
        "refine_preview_box",
        "refine_preview_label",
        "refine_decision",
        "refine_run_id",
        "refine_run_traj_id",
        "refine_run_frame_count",
        "refine_run_flagged_frame_count",
        "refine_run_flagged_ratio",
        "refine_run_delete_blob_count",
        "refine_run_blob_count",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    out_rows = []
    for row in progress.bar(rows, desc="Update blob classification"):
        row = dict(row)
        key = (int(row["frame"]), int(row["blob_index"]))
        mark = preview_marks.get(key)
        info = delete_map.get(key)

        if info is not None:
            row["erase_final"] = "1"
            existing = [x for x in str(row.get("erase_final_reasons", "")).split("|") if x]
            for reason in info.reasons:
                if reason not in existing:
                    existing.append(reason)
            row["erase_final_reasons"] = "|".join(existing)
            row["refine_deleted"] = "1"
            row["refine_delete_reasons"] = "|".join(info.reasons)
            row["refine_gt_class_id"] = info.gt_class_id
            row["refine_pred_class_id"] = info.pred_class_id
            row["refine_pred_conf"] = info.pred_conf
            row["refine_best_obb_iou"] = info.best_obb_iou
            row["refine_angle_diff_deg"] = info.angle_diff_deg
            row["refine_obb_aspect_ratio"] = info.obb_aspect_ratio
            row["refine_min_obb_aspect_ratio"] = info.min_obb_aspect_ratio
        else:
            row["refine_deleted"] = "0"
            row["refine_delete_reasons"] = ""
            row["refine_gt_class_id"] = ""
            row["refine_pred_class_id"] = ""
            row["refine_pred_conf"] = ""
            row["refine_best_obb_iou"] = ""
            row["refine_angle_diff_deg"] = ""
            row["refine_obb_aspect_ratio"] = ""
            row["refine_min_obb_aspect_ratio"] = ""

        if mark is not None:
            row["refine_preview_box"] = "1" if mark.draw_box else "0"
            row["refine_preview_label"] = "1" if mark.draw_label else "0"
            row["refine_decision"] = mark.decision
            row["refine_run_id"] = "" if mark.run_id < 0 else str(mark.run_id)
            row["refine_run_traj_id"] = "" if mark.traj_id < 0 else str(mark.traj_id)
            row["refine_run_frame_count"] = "" if mark.run_frame_count <= 0 else str(mark.run_frame_count)
            row["refine_run_flagged_frame_count"] = "" if mark.flagged_frame_count <= 0 else str(mark.flagged_frame_count)
            row["refine_run_flagged_ratio"] = mark.ratio_text
            row["refine_run_delete_blob_count"] = "" if mark.flagged_blob_count <= 0 else str(mark.flagged_blob_count)
            row["refine_run_blob_count"] = "" if mark.run_blob_count <= 0 else str(mark.run_blob_count)
        else:
            row["refine_preview_box"] = "0"
            row["refine_preview_label"] = "0"
            row["refine_decision"] = ""
            row["refine_run_id"] = ""
            row["refine_run_traj_id"] = ""
            row["refine_run_frame_count"] = ""
            row["refine_run_flagged_frame_count"] = ""
            row["refine_run_flagged_ratio"] = ""
            row["refine_run_delete_blob_count"] = ""
            row["refine_run_blob_count"] = ""
        out_rows.append(row)

    with open(dst, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def numeric_or_text(value) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}"
    return str(value)


def build_bbox_stats(
    manifest_rows: Sequence[dict],
    cfg: dict,
    delete_count: int,
    saved_images: int,
    saved_previews: int,
    empty_frame_stats: dict,
    progress: Progress,
) -> List[List[str]]:
    kept_rows = []
    for row in progress.bar(manifest_rows, desc="Build bbox stats"):
        w = float(row["rect_w"])
        h = float(row["rect_h"])
        kept_rows.append({
            "frame": int(row["frame"]),
            "bbox_index": int(row["blob_index"]),
            "w": w,
            "h": h,
            "sum": w + h,
            "area": w * h,
        })
    min_valid_frames, min_valid_ratio, source_fps = resolve_direction_min_valid_frames(cfg)

    summary: List[List[str]] = [["metric", "kind", "value_px", "frame", "bbox_index", "w", "h"]]

    def add(metric: str, kind: str, value, frame="", bbox_index="", w="", h="") -> None:
        summary.append([
            metric, kind,
            numeric_or_text(value) if value != "" else "",
            numeric_or_text(frame) if frame != "" else "",
            numeric_or_text(bbox_index) if bbox_index != "" else "",
            numeric_or_text(w) if w != "" else "",
            numeric_or_text(h) if h != "" else "",
        ])

    def source_row(metric: str, kind: str):
        reverse = kind == "max"
        return sorted(kept_rows, key=lambda r: (float(r[metric]), int(r["frame"]), int(r["bbox_index"])), reverse=reverse)[0]

    meta_items = [
        ("n_total_boxes", "meta", len(manifest_rows) + delete_count),
        ("frame_interval", "meta", int(cfg.get("FRAME_INTERVAL", 1))),
        ("first_frame", "meta", int((cfg.get("training", {}) or {}).get("FIRST_FRAME", 0))),
        ("last_frame_effective", "meta", max((int(r["frame"]) for r in manifest_rows), default=-1)),
        ("crossing_mode", "meta", str(cfg.get("CROSSING_MODE", "hybrid"))),
        ("source_fps", "meta", float(source_fps)),
        ("direction_min_valid_frames", "meta", int(min_valid_frames)),
        ("direction_min_valid_frames_fps_ratio", "meta", float(min_valid_ratio)),
        ("final_erased_blobs", "meta", delete_count),
        ("final_kept_blobs", "meta", len(manifest_rows)),
        ("accepted_objects", "meta", len(manifest_rows)),
        ("object_pool_rows", "meta", len(manifest_rows)),
        ("saved_images", "meta", saved_images),
        ("saved_previews", "meta", saved_previews),
        ("skipped_empty_frames", "meta", int(empty_frame_stats.get("skipped_empty_frames", 0))),
        ("empty_frame_removed_blobs", "meta", int(empty_frame_stats.get("empty_frame_removed_blobs", 0))),
        ("direction_classes", "meta", 8),
        ("output_frame_interval", "meta", int(cfg.get("FRAME_INTERVAL", 1))),
        ("direction_frame_interval", "meta", 1),
    ]
    for metric, kind, value in meta_items:
        add(metric, kind, value)

    for reason, count in sorted((empty_frame_stats.get("empty_frame_reason_counts") or {}).items()):
        add(f"empty_frame_reason:{reason}", "removed_blob_count", int(count))

    for metric in ("w", "h", "sum", "area"):
        if kept_rows:
            values = [float(r[metric]) for r in kept_rows]
            min_src = source_row(metric, "min")
            max_src = source_row(metric, "max")
            add(metric, "min", min(values), min_src["frame"], min_src["bbox_index"], min_src["w"], min_src["h"])
            add(metric, "max", max(values), max_src["frame"], max_src["bbox_index"], max_src["w"], max_src["h"])
            values_sorted = sorted(values)
            add(metric, "q1", float(np.percentile(values_sorted, 25)))
            add(metric, "q3", float(np.percentile(values_sorted, 75)))
            add(metric, "median", float(np.median(values_sorted)))
            add(metric, "mean", float(np.mean(values_sorted)))
        else:
            for kind in ("min", "max", "q1", "q3", "median", "mean"):
                add(metric, kind, "")

    for row in progress.bar(kept_rows, desc="Write bbox rows"):
        add("bbox", "kept", row["area"], row["frame"], row["bbox_index"], row["w"], row["h"])
    return summary


def write_bbox_stats(
    without_crossing_dir: str,
    manifest_rows: Sequence[dict],
    cfg: dict,
    delete_count: int,
    saved_images: int,
    saved_previews: int,
    empty_frame_stats: dict,
    progress: Progress,
) -> None:
    path = os.path.join(without_crossing_dir, "bbox_stats_single_animal_images.csv")
    rows = build_bbox_stats(manifest_rows, cfg, delete_count, saved_images, saved_previews, empty_frame_stats, progress)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def copy_static_files(original_dir: str, without_crossing_dir: str, progress: Progress) -> None:
    for name in progress.bar(("class_names.txt", "pre_direction_runs.csv"), desc="Move static files"):
        src = os.path.join(original_dir, name)
        dst = os.path.join(without_crossing_dir, name)
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)


def delete_map_to_rows(delete_map: Dict[Tuple[int, int], DeleteInfo]) -> List[dict]:
    rows: List[dict] = []
    for frame, blob_index in sorted(delete_map.keys()):
        info = delete_map[(frame, blob_index)]
        rows.append({
            "frame": int(frame),
            "blob_index": int(blob_index),
            "gt_class_id": info.gt_class_id,
            "pred_class_id": info.pred_class_id,
            "pred_conf": info.pred_conf,
            "best_obb_iou": info.best_obb_iou,
            "angle_diff_deg": info.angle_diff_deg,
            "class_diff": "",
            "obb_aspect_ratio": info.obb_aspect_ratio,
            "min_obb_aspect_ratio": info.min_obb_aspect_ratio,
            "reasons": "|".join(info.reasons),
        })
    return rows


def write_delete_map_csv(path: str, delete_map: Dict[Tuple[int, int], DeleteInfo]) -> None:
    fieldnames = [
        "frame", "blob_index", "gt_class_id", "pred_class_id", "pred_conf",
        "best_obb_iou", "angle_diff_deg", "class_diff",
        "obb_aspect_ratio", "min_obb_aspect_ratio", "reasons",
    ]
    write_csv_rows(path, fieldnames, delete_map_to_rows(delete_map))


def build_all_candidate_preview_marks(delete_map: Dict[Tuple[int, int], DeleteInfo]) -> Dict[Tuple[int, int], PreviewMark]:
    return {
        key: PreviewMark(
            draw_box=True,
            draw_label=True,
            deleted=True,
            info=info,
            decision="all_candidates_delete",
        )
        for key, info in delete_map.items()
    }


def merge_forced_delete_preview_marks(
    base: Dict[Tuple[int, int], PreviewMark],
    forced_delete_map: Dict[Tuple[int, int], DeleteInfo],
    decision: str,
) -> Dict[Tuple[int, int], PreviewMark]:
    merged = dict(base)
    for key, info in forced_delete_map.items():
        prev = merged.get(key)
        if prev is None:
            merged[key] = PreviewMark(
                draw_box=True,
                draw_label=True,
                deleted=True,
                info=info,
                decision=decision,
            )
            continue

        prev_info = merge_delete_info(prev.info, info) if prev.info is not None else info
        merged_decision = prev.decision
        if decision not in str(merged_decision).split("|"):
            merged_decision = f"{merged_decision}|{decision}" if merged_decision else decision
        merged[key] = PreviewMark(
            draw_box=True,
            draw_label=True,
            deleted=True,
            info=prev_info,
            decision=merged_decision,
            run_id=prev.run_id,
            traj_id=prev.traj_id,
            run_frame_count=prev.run_frame_count,
            flagged_frame_count=prev.flagged_frame_count,
            run_blob_count=prev.run_blob_count,
            flagged_blob_count=prev.flagged_blob_count,
            ratio_text=prev.ratio_text,
        )
    return merged


def resolve_refine_apply_mode(cfg: dict) -> str:
    mode = str(cfg.get("REFINE_APPLY_MODE", "run_policy")).strip().lower()
    if mode not in {"run_policy", "all_candidates"}:
        raise ValueError(
            "REFINE_APPLY_MODE must be 'run_policy' or 'all_candidates', "
            f"got {cfg.get('REFINE_APPLY_MODE')!r}"
        )
    return mode


def cleanup_empty_dirs(root: str) -> int:
    removed = 0
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == root or filenames:
            continue
        try:
            os.rmdir(dirpath)
            removed += 1
        except OSError:
            pass
    return removed


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python apply_class_label_filtering.py config.yaml")

    cfg = load_config(sys.argv[1])
    session_path = str(cfg["SESSION_PATH"])
    without_crossing_dir = os.path.join(session_path, SINGLE_ANIMAL_IMAGES_DIR_NAME)
    refine_dir_name = get_refine_dir_name(cfg)
    refine_root = os.path.join(without_crossing_dir, refine_dir_name)
    original_dir = os.path.join(refine_root, "original")
    delete_csv = os.path.join(refine_root, "delete_blobs.csv")
    progress = Progress(enabled=True)

    if not os.path.exists(delete_csv):
        raise FileNotFoundError(delete_csv)

    backup_created = backup_without_crossing_to_refine_original(without_crossing_dir, refine_root, original_dir, progress)
    if backup_created:
        print(f"Backed up existing single_animal_images to: {original_dir}")

    manifest_src = os.path.join(original_dir, "object_pool", "manifest.csv")

    if not os.path.isdir(original_dir):
        raise FileNotFoundError(original_dir)
    if not os.path.exists(manifest_src):
        raise FileNotFoundError(manifest_src)
    if "BACKGROUND_PATH" not in cfg:
        raise KeyError("BACKGROUND_PATH is required")

    refine_delete_map_raw = read_delete_map(delete_csv, progress)
    original_manifest_rows = read_manifest_rows(manifest_src)
    min_obb_aspect_ratio = get_min_obb_aspect_ratio(cfg)
    low_obb_aspect_delete_map = build_low_obb_aspect_delete_map(
        original_dir,
        original_manifest_rows,
        min_obb_aspect_ratio,
        progress,
    )
    delete_map_raw = merge_delete_maps(refine_delete_map_raw, low_obb_aspect_delete_map)
    run_csv_src = os.path.join(original_dir, "pre_direction_runs.csv")
    if not os.path.exists(run_csv_src):
        raise FileNotFoundError(run_csv_src)
    run_rows_all = read_pre_direction_run_rows(run_csv_src)
    min_valid_frames, min_valid_ratio, source_fps = resolve_direction_min_valid_frames(cfg)
    run_delete_ratio = parse_run_delete_ratio(cfg.get("RUN_DELETE_RATIO", 0.4))
    run_rows = filter_valid_run_rows(run_rows_all, min_valid_frames)
    run_summary_csv = os.path.join(without_crossing_dir, "runs_deletion_counts.csv")
    runs, _, _, _ = build_run_maps_from_csv(run_rows, original_manifest_rows)
    run_summary_rows = write_run_delete_summary_csv(run_summary_csv, run_rows, runs, refine_delete_map_raw, progress)

    run_policy_delete_map, run_policy_preview_marks, run_policy_stats = build_policy_maps(
        original_manifest_rows,
        run_rows,
        refine_delete_map_raw,
        run_summary_rows,
        run_delete_ratio,
        progress,
    )
    run_policy_delete_map = merge_delete_maps(run_policy_delete_map, low_obb_aspect_delete_map)
    run_policy_preview_marks = merge_forced_delete_preview_marks(
        run_policy_preview_marks,
        low_obb_aspect_delete_map,
        LOW_OBB_ASPECT_REASON,
    )

    delete_all_csv = os.path.join(refine_root, "delete_blobs_all_candidates.csv")
    delete_run_policy_csv = os.path.join(refine_root, "delete_blobs_run_policy.csv")
    write_delete_map_csv(delete_all_csv, delete_map_raw)
    write_delete_map_csv(delete_run_policy_csv, run_policy_delete_map)

    apply_mode = resolve_refine_apply_mode(cfg)
    if apply_mode == "all_candidates":
        delete_map = delete_map_raw
        preview_marks = build_all_candidate_preview_marks(delete_map_raw)
        applied_csv = delete_all_csv
        print("[INFO] REFINE_APPLY_MODE=all_candidates: applying every blue candidate. This mode is intended for intermediate iterative refinement inputs.")
    else:
        delete_map = run_policy_delete_map
        preview_marks = run_policy_preview_marks
        applied_csv = delete_run_policy_csv
        print("[INFO] REFINE_APPLY_MODE=run_policy: applying run-policy deletions for downstream output.")

    applied_delete_csv = os.path.join(refine_root, "delete_blobs_applied.csv")
    write_delete_map_csv(applied_delete_csv, delete_map)

    original_manifest_by_frame = group_manifest_rows(original_manifest_rows)
    filtered_manifest_rows = filter_manifest_rows(original_manifest_rows, delete_map, progress)

    reset_without_crossing_output_dir(without_crossing_dir, progress)
    ensure_dir(without_crossing_dir)
    copy_static_files(original_dir, without_crossing_dir, progress)
    saved_images, saved_previews, empty_frame_stats = rewrite_frame_assets(
        original_dir=original_dir,
        without_crossing_dir=without_crossing_dir,
        manifest_by_frame=original_manifest_by_frame,
        delete_map=delete_map,
        preview_marks=preview_marks,
        cfg=cfg,
        progress=progress,
    )
    write_filtered_object_pool(original_dir, without_crossing_dir, filtered_manifest_rows, progress)
    filter_blob_classification(original_dir, without_crossing_dir, delete_map, preview_marks, progress)
    write_bbox_stats(without_crossing_dir, filtered_manifest_rows, cfg, len(delete_map), saved_images, saved_previews, empty_frame_stats, progress)

    preview_box_count = sum(1 for mark in preview_marks.values() if mark.draw_box)
    preview_label_count = sum(1 for mark in preview_marks.values() if mark.draw_label)
    kept_target_count = sum(1 for mark in preview_marks.values() if (not mark.deleted) and mark.draw_label)
    collateral_delete_count = sum(1 for mark in preview_marks.values() if mark.deleted and mark.draw_box and not mark.draw_label)

    print(f"Restored single_animal_images segments from: {original_dir}")
    print(f"Applied refine deletions from: {applied_csv}")
    print(f"Saved all-candidate delete CSV to: {delete_all_csv}")
    print(f"Saved run-policy delete CSV to: {delete_run_policy_csv}")
    print(f"Saved applied delete CSV to: {applied_delete_csv}")
    print(f"Read pre-direction runs from: {run_csv_src}")
    print(f"Saved run delete summary to: {run_summary_csv}")
    print(
        f"Ignored short runs with run_frame_count <= {min_valid_frames} "
        f"(source_fps={float(source_fps):.6f}, ratio={float(min_valid_ratio):.6f})"
    )
    print(f"Run policy: use pre_direction_runs.csv run units (same traj_id, start_frame, end_frame). If delete_blobs.csv covers >= {float(run_delete_ratio):.6f} of manifest blobs inside the run, delete the whole run. Otherwise delete nothing in that run.")
    print(f"Low OBB aspect deletion: aspect < {float(min_obb_aspect_ratio):.6f}, blobs={len(low_obb_aspect_delete_map)}")
    print(f"Valid runs checked: {run_policy_stats['run_count']}")
    print(f"Expanded runs: {run_policy_stats['run_delete_expand']}")
    print(f"Rejected runs: {run_policy_stats['run_delete_reject']}")
    print(f"Added deletions by run expansion: {run_policy_stats['run_deleted_added']}")
    print(f"Rejected original deletions by run policy: {run_policy_stats['run_deleted_rejected']}")
    print(f"Deleted objects: {len(delete_map)}")
    print(f"Kept objects: {len(filtered_manifest_rows)}")
    print(f"Preview blue boxes: {preview_box_count}")
    print(f"Preview blue labels: {preview_label_count}")
    print(f"Preview kept refine targets: {kept_target_count}")
    print(f"Preview collateral deleted blobs: {collateral_delete_count}")
    print(f"Saved images: {saved_images}")
    print(f"Saved previews: {saved_previews}")
    print(f"Skipped empty single_animal_images frames: {int(empty_frame_stats.get('skipped_empty_frames', 0))}")
    for reason, count in sorted((empty_frame_stats.get("empty_frame_reason_counts") or {}).items()):
        print(f"  empty-frame removal reason {reason}: {count}")
    print(f"Saved filtered single_animal_images segments to: {without_crossing_dir}")

    removed_dirs = cleanup_empty_dirs(original_dir)
    if removed_dirs:
        print(f"Removed {removed_dirs} empty directories from: {original_dir}")


if __name__ == "__main__":
    main()
