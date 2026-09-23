# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Assemble detector training and validation images with OBB labels."""

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)


import csv
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from batch_utils import tqdm
from batch_utils import resolve_num_workers as _resolve_num_workers_bt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gui.color import OBB_COLOR
from path_utils import resolve_config_paths
from random_utils import derive_seed, make_python_rng, normalize_seed

DIRECTION_CLASS_NAMES = ['animal']

# (image_path, label_path, source_prefix, sample_type)
Pair = Tuple[str, str, str, str]
_RESIZED_BACKGROUND_CACHE: Dict[Tuple[str, int, str], np.ndarray] = {}

# ---------------------------------------------------------------------------
# Right-angle (0/90/180/270) rotation augmentation for YOLO OBB samples.
#
# See normalize_rotation_angles(), select_rotation_deg_for_pair(), and
# rotate_yolo_obb_sample() below. The same deterministic rotation policy is
# applied to both train and validation/test samples. write_yolo_dataset()
# defaults rotation_angles to (0,), so callers that do not enable rotation are
# unaffected.
# ---------------------------------------------------------------------------

ALLOWED_ROTATION_ANGLES: Tuple[int, ...] = (0, 90, 180, 270)
NONZERO_ROTATION_PROB: float = 0.75


def normalize_rotation_angles(values) -> Tuple[int, ...]:
    """Validate CREATE_DATASET_ROTATION_ANGLES into a tuple of unique ints in {0,90,180,270}."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(
            f"CREATE_DATASET_ROTATION_ANGLES must be a list of integers, got {values!r}"
        )
    if len(values) == 0:
        raise ValueError("CREATE_DATASET_ROTATION_ANGLES must not be empty")

    allowed = set(ALLOWED_ROTATION_ANGLES)
    seen = set()
    result: List[int] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"CREATE_DATASET_ROTATION_ANGLES values must be int in {sorted(allowed)}, got {v!r}"
            )
        if v not in allowed:
            raise ValueError(
                f"CREATE_DATASET_ROTATION_ANGLES values must be one of {sorted(allowed)}, got {v!r}"
            )
        if v in seen:
            raise ValueError(f"CREATE_DATASET_ROTATION_ANGLES contains duplicate angle: {v!r}")
        seen.add(v)
        result.append(int(v))
    return tuple(result)

def make_background_canvas(
    output_size: int,
    dtype: np.dtype,
    background_path: str | None = None,
) -> np.ndarray:
    """Return an output-sized BGR background canvas for full-frame samples."""
    output_size = int(output_size)
    if background_path is None:
        return np.full((output_size, output_size, 3), 255, dtype=dtype)

    cache_key = (os.path.abspath(background_path), output_size, np.dtype(dtype).str)
    resized_bg = _RESIZED_BACKGROUND_CACHE.get(cache_key)
    if resized_bg is None:
        bg = cv2.imread(background_path, cv2.IMREAD_COLOR)
        if bg is None:
            raise RuntimeError(f"Failed to read background image: {background_path}")
        resized_bg = cv2.resize(
            bg,
            (output_size, output_size),
            interpolation=cv2.INTER_LINEAR,
        ).astype(dtype)
        _RESIZED_BACKGROUND_CACHE[cache_key] = resized_bg
    return resized_bg.copy()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return resolve_config_paths(cfg)


def cfg_bool(cfg: dict, key: str, default: bool = False) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def ensure_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def write_yolo_data_yaml(dataset_root: str, class_names: List[str]) -> str:
    data = {
        "train": os.path.join(dataset_root, "train"),
        "val": os.path.join(dataset_root, "test"),
        "nc": len(class_names),
        "names": list(class_names),
    }
    out_path = os.path.join(dataset_root, "data.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return out_path


def validate_val_ratio(val_ratio: float) -> float:
    val_ratio = float(val_ratio)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"VAL_RATIO must be > 0.0 and < 1.0, got {val_ratio}")
    return val_ratio


def split_train_val(
    items: List[Pair],
    val_ratio: float,
    seed: int,
    *,
    allow_small_dataset: bool = False,
    min_one_val: bool = False,
) -> Tuple[List[Pair], List[Pair]]:
    items = list(items)
    make_python_rng(seed, "create_dataset", "split_train_val").shuffle(items)
    val_ratio = validate_val_ratio(val_ratio)

    if allow_small_dataset and min_one_val:
        if len(items) < 2:
            raise RuntimeError(
                f"Dataset is too small for train/test split: total={len(items)}, "
                f"val_ratio={val_ratio}; at least 2 samples are required"
            )
        n_val = max(1, min(len(items) - 1, round(len(items) * val_ratio)))
    else:
        n_val = int(round(len(items) * val_ratio))

    if len(items) < 2 or n_val <= 0 or n_val >= len(items):
        raise RuntimeError(
            f"Dataset is too small for train/test split: total={len(items)}, val_ratio={val_ratio}"
        )
    return items[n_val:], items[:n_val]


# ---------------------------------------------------------------------------
# Frame-group aware train/validation split
#
# All samples derived from the same original video frame (full-frame resize,
# every clustered/non-clustered paste set built from it, and every crop taken
# from any of those pastes) must land entirely in train or entirely in
# validation. See select_validation_frame_groups() / build_frame_based_split().
# ---------------------------------------------------------------------------

FRAME_ID_PATTERN = re.compile(r"^frame_(\d+)")


class SplitInfeasibleError(RuntimeError):
    """Raised when no frame-group combination satisfies the train/val pool window."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.details = details or {}


def extract_frame_id(stem: str) -> int:
    match = FRAME_ID_PATTERN.match(stem)
    if not match:
        raise ValueError(
            f"Cannot extract source frame id from filename stem: {stem!r}. "
            "Expected a stem starting with 'frame_<digits>'."
        )
    return int(match.group(1))


def pair_stem(pair: Pair) -> str:
    return os.path.splitext(os.path.basename(pair[0]))[0]


def output_stem_for_pair(pair: Pair) -> str:
    _image_path, _label_path, prefix, sample_type = pair
    out_suffix = "crop_highres" if sample_type == "crop_highres" else "full_resized"
    return f"{prefix}_{pair_stem(pair)}_{out_suffix}"


def pair_frame_id(pair: Pair) -> int:
    return extract_frame_id(pair_stem(pair))


def is_clustered_prefix(prefix: str) -> bool:
    return prefix in ("clustered", "paste_blobs_clustered")


def group_pairs_by_frame(pairs: List[Pair]) -> Dict[int, List[Pair]]:
    """Group samples by their source video frame id, ignoring source_prefix.

    Non-clustered and clustered samples derived from the same original frame
    must stay in the same group, otherwise the train/val split can leak the
    same background/individuals/pose across the split.
    """
    groups: Dict[int, List[Pair]] = defaultdict(list)
    for pair in pairs:
        groups[pair_frame_id(pair)].append(pair)
    return dict(groups)


def _derive_seed(seed: int, *parts: object) -> int:
    return derive_seed(seed, "create_dataset", *parts)


def _deterministic_unit_interval(seed: int, *parts: object) -> float:
    return _derive_seed(seed, *parts) / float(0xFFFFFFFFFFFFFFFF)


def select_rotation_deg_for_pair(
    rotation_angles: Sequence[int],
    seed: int,
    source_prefix: str,
    sample_type: str,
    source_stem: str,
    *,
    is_train: bool,
) -> int:
    """Deterministically select 0 degrees or a configured non-zero rotation.

    Every sample remains unrotated with probability 1 -
    NONZERO_ROTATION_PROB. Otherwise, one configured non-zero angle is selected
    uniformly. This policy is identical for train and validation/test samples.
    The result is independent of worker count, execution order, and absolute
    paths. ``is_train`` is intentionally not used: it is retained only as an
    internal split descriptor for the existing call chain.
    """
    nonzero_angles = [int(angle) for angle in rotation_angles if int(angle) != 0]
    if not nonzero_angles:
        return 0

    local_seed = _derive_seed(
        seed, "create_dataset_rotation", source_prefix, sample_type, source_stem,
    )
    rng = random.Random(local_seed)

    if rng.random() >= NONZERO_ROTATION_PROB:
        return 0
    return rng.choice(nonzero_angles)


def _nearest_achievable_sum(full_reach: int, n_val: int, total: int) -> int:
    """Return the achievable sum (bit set in full_reach) closest to n_val.

    Bits 0 and `total` are always achievable (empty / full subset), so this
    never fails to find a candidate. Ties are broken toward the larger sum.
    """
    if (full_reach >> n_val) & 1:
        return n_val
    below_mask = full_reach & ((1 << n_val) - 1)
    above_mask = full_reach & ~((1 << (n_val + 1)) - 1) & ((1 << (total + 1)) - 1)
    candidates: List[int] = []
    if below_mask:
        candidates.append(below_mask.bit_length() - 1)
    if above_mask:
        lowest_bit = above_mask & (-above_mask)
        candidates.append(lowest_bit.bit_length() - 1)
    candidates.sort(key=lambda s: (abs(s - n_val), -s))
    return candidates[0]


def select_validation_frame_groups(
    frame_groups: Dict[int, List[Pair]],
    n_val: int,
    n_train: int,
    clustered_ratio_target: float,
    seed: int,
    *,
    strict: bool = True,
) -> Tuple[List[int], List[int]]:
    """Choose which source frame ids become the validation pool.

    Uses a bitset subset-sum DP over per-frame candidate counts so the
    validation pool size lands in [n_val, total_candidates - n_train], with
    ties among reachable combinations broken toward the requested clustered
    ratio and, failing that, a deterministic seed-derived order. This is a
    polynomial (samples x frame-groups) DP, not a combinatorial search over
    all frame-group subsets.

    When strict=True, raises SplitInfeasibleError if no combination of frame
    groups reaches a validation pool size in that window. When strict=False
    (best-effort fallback after regeneration retries are exhausted), the
    window constraint is dropped and the achievable sum closest to n_val is
    used instead; this never raises SplitInfeasibleError.
    """
    frame_ids_sorted = sorted(frame_groups.keys())
    sizes: List[int] = []
    clustered_counts: List[int] = []
    for fid in frame_ids_sorted:
        group = frame_groups[fid]
        sizes.append(len(group))
        clustered_counts.append(sum(1 for p in group if is_clustered_prefix(p[2])))

    n = len(frame_ids_sorted)
    total = sum(sizes)
    upper_bound = total - n_train
    details = {
        "n_val": n_val,
        "n_train": n_train,
        "total_candidates": total,
        "num_frame_groups": n,
        "min_frame_group_size": min(sizes) if sizes else 0,
        "max_frame_group_size": max(sizes) if sizes else 0,
    }
    if strict and (n_val < 0 or n_train < 0 or n_val > upper_bound):
        raise SplitInfeasibleError(
            f"No feasible validation pool window: n_val={n_val}, n_train={n_train}, "
            f"total_candidates={total}, frame_groups={n}",
            details=details,
        )

    # reach[i] is a bitmask where bit s is set iff some subset of the first i
    # frame groups (by candidate count) sums to s.
    reach: List[int] = [1]
    for sz in sizes:
        prev = reach[-1]
        reach.append(prev | (prev << sz))

    full_reach = reach[n]

    if strict:
        window_mask = ((1 << (upper_bound + 1)) - 1) & ~((1 << n_val) - 1)
        feasible_mask = full_reach & window_mask
        if feasible_mask == 0:
            raise SplitInfeasibleError(
                f"No frame-group combination reaches a validation pool within "
                f"[{n_val}, {upper_bound}] (total_candidates={total}, frame_groups={n})",
                details=details,
            )
        if (full_reach >> n_val) & 1:
            target_sum = n_val
        else:
            lowest_bit = feasible_mask & (-feasible_mask)
            target_sum = lowest_bit.bit_length() - 1
    else:
        target_sum = _nearest_achievable_sum(full_reach, max(0, min(n_val, total)), total)

    # Backward reconstruction. At each step, if both taking and skipping the
    # current frame group can still reach target_sum, prefer whichever keeps
    # the running clustered ratio closer to clustered_ratio_target; break
    # remaining ties deterministically from RANDOM_SEED.
    selected = [False] * n
    remaining = target_sum
    running_total = 0
    running_clustered = 0
    for idx in range(n - 1, -1, -1):
        sz = sizes[idx]
        cl = clustered_counts[idx]
        prev_reach = reach[idx]
        can_skip = bool((prev_reach >> remaining) & 1)
        can_take = remaining >= sz and bool((prev_reach >> (remaining - sz)) & 1)
        if can_take and not can_skip:
            take = True
        elif can_skip and not can_take:
            take = False
        elif not can_take and not can_skip:
            raise AssertionError(
                f"Subset-sum reconstruction invariant violated at idx={idx}, remaining={remaining}"
            )
        else:
            denom_take = running_total + sz
            ratio_take = (running_clustered + cl) / denom_take if denom_take > 0 else clustered_ratio_target
            ratio_skip = (running_clustered / running_total) if running_total > 0 else clustered_ratio_target
            score_take = abs(ratio_take - clustered_ratio_target)
            score_skip = abs(ratio_skip - clustered_ratio_target)
            if score_take < score_skip:
                take = True
            elif score_skip < score_take:
                take = False
            else:
                take = _deterministic_unit_interval(seed, "split-tiebreak", frame_ids_sorted[idx]) < 0.5
        if take:
            selected[idx] = True
            remaining -= sz
            running_total += sz
            running_clustered += cl

    if remaining != 0:
        raise AssertionError(f"Subset-sum reconstruction failed to reach target_sum={target_sum}")

    val_frame_ids = [frame_ids_sorted[i] for i in range(n) if selected[i]]
    train_frame_ids = [frame_ids_sorted[i] for i in range(n) if not selected[i]]
    return val_frame_ids, train_frame_ids


def _bucket_key(pair: Pair) -> Tuple[bool, str]:
    return (is_clustered_prefix(pair[2]), pair[3])


def largest_remainder_allocation(
    bucket_sizes: Dict[Tuple[bool, str], int],
    total_target: int,
    tie_key,
) -> Dict[Tuple[bool, str], int]:
    """Proportionally allocate total_target across buckets (largest-remainder method)."""
    grand_total = sum(bucket_sizes.values())
    if total_target < 0 or total_target > grand_total:
        raise ValueError(
            f"largest_remainder_allocation: total_target={total_target} out of range [0, {grand_total}]"
        )
    if grand_total == 0:
        return {k: 0 for k in bucket_sizes}

    raw = {k: bucket_sizes[k] * total_target / grand_total for k in bucket_sizes}
    alloc = {k: min(bucket_sizes[k], int(math.floor(raw[k]))) for k in bucket_sizes}
    remainder = total_target - sum(alloc.values())
    order = sorted(bucket_sizes.keys(), key=lambda k: (-(raw[k] - math.floor(raw[k])), tie_key(k)))
    while remainder > 0:
        progressed = False
        for k in order:
            if remainder <= 0:
                break
            if alloc[k] < bucket_sizes[k]:
                alloc[k] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    if remainder > 0:
        raise RuntimeError(
            f"largest_remainder_allocation: could not allocate remainder={remainder} "
            f"(total_target={total_target}, grand_total={grand_total})"
        )
    return alloc


def trim_pool_to_target(
    pairs: List[Pair],
    target_count: int,
    seed: int,
    tag: str,
) -> Tuple[List[Pair], List[Pair]]:
    """Select target_count samples from pairs, preserving clustered/non-clustered and
    full_resized/crop_highres ratios via largest-remainder allocation across the
    (is_clustered, sample_type) buckets. Returns (selected, unused)."""
    if target_count > len(pairs):
        raise ValueError(f"target_count={target_count} exceeds pool size={len(pairs)}")
    if target_count == len(pairs):
        return list(pairs), []
    if target_count == 0:
        return [], list(pairs)

    buckets: Dict[Tuple[bool, str], List[Pair]] = defaultdict(list)
    for p in pairs:
        buckets[_bucket_key(p)].append(p)
    bucket_sizes = {k: len(v) for k, v in buckets.items()}
    alloc = largest_remainder_allocation(bucket_sizes, target_count, tie_key=lambda k: str(k))

    selected: List[Pair] = []
    unused: List[Pair] = []
    for k, plist in buckets.items():
        take_n = alloc[k]
        shuffled = list(plist)
        random.Random(_derive_seed(seed, tag, "trim", k)).shuffle(shuffled)
        selected.extend(shuffled[:take_n])
        unused.extend(shuffled[take_n:])
    return selected, unused


def build_frame_based_split(
    pairs: List[Pair],
    num_total_images: int,
    val_ratio: float,
    nonclustered_ratio: float,
    seed: int,
    *,
    best_effort: bool = False,
) -> Tuple[List[Pair], List[Pair], List[Pair], List[Pair], dict]:
    """Split candidate pairs into train/val sets with zero frame leakage.

    Returns (train_pairs, val_pairs, train_unused, val_unused, stats).

    When best_effort=False (default), the result is forced to exactly
    n_train/n_val samples; raises SplitInfeasibleError if no frame-group
    combination fits the required validation pool window, so the caller can
    regenerate more candidates and retry.

    When best_effort=True, the window constraint is dropped: validation gets
    whichever achievable frame-group combination lands closest to n_val, and
    train takes up to n_train from what remains (or all of it, if that falls
    short). This never raises SplitInfeasibleError; stats["n_train_actual"]
    and stats["n_val_actual"] may then differ from the requested counts, and
    the caller is expected to warn instead of failing.
    """
    n_val = round(num_total_images * val_ratio)
    n_train = num_total_images - n_val
    clustered_ratio_target = 1.0 - nonclustered_ratio

    frame_groups = group_pairs_by_frame(pairs)
    total_candidates = len(pairs)
    num_frames = len(frame_groups)

    val_frame_ids, train_frame_ids = select_validation_frame_groups(
        frame_groups, n_val, n_train, clustered_ratio_target, seed, strict=not best_effort,
    )
    val_pool = [p for fid in val_frame_ids for p in frame_groups[fid]]
    train_pool = [p for fid in train_frame_ids for p in frame_groups[fid]]

    if best_effort:
        # val_pool is already the closest achievable combination to n_val;
        # use it as-is. train gets up to n_train from whatever remains.
        val_target = len(val_pool)
        train_target = min(len(train_pool), n_train)
    else:
        val_target = n_val
        train_target = n_train

    val_selected, val_unused = trim_pool_to_target(val_pool, val_target, seed, "val")
    train_selected, train_unused = trim_pool_to_target(train_pool, train_target, seed, "train")

    def _counts(sel: List[Pair]) -> dict:
        clustered = sum(1 for p in sel if is_clustered_prefix(p[2]))
        full = sum(1 for p in sel if p[3] == "full_resized")
        crop = sum(1 for p in sel if p[3] == "crop_highres")
        return {
            "clustered": clustered,
            "nonclustered": len(sel) - clustered,
            "full_resized": full,
            "crop_highres": crop,
        }

    group_sizes = list(sizes for sizes in (len(v) for v in frame_groups.values()))
    stats = {
        "num_total_images": num_total_images,
        "best_effort": best_effort,
        "n_train_target": n_train,
        "n_val_target": n_val,
        "n_train_actual": len(train_selected),
        "n_val_actual": len(val_selected),
        "total_candidates": total_candidates,
        "num_frames": num_frames,
        "val_pool_size": len(val_pool),
        "train_pool_size": len(train_pool),
        "val_frame_ids": val_frame_ids,
        "train_frame_ids": train_frame_ids,
        "train_counts": _counts(train_selected),
        "val_counts": _counts(val_selected),
        "min_frame_group_size": min(group_sizes) if group_sizes else 0,
        "max_frame_group_size": max(group_sizes) if group_sizes else 0,
    }
    return train_selected, val_selected, train_unused, val_unused, stats


def write_split_manifest(
    output_root: str,
    train_pairs: List[Pair],
    val_pairs: List[Pair],
    unused_pairs: List[Pair],
    rotation_by_output_stem: Optional[Dict[str, int]] = None,
) -> str:
    manifest_path = os.path.join(output_root, "split_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split", "output_stem", "source_frame_id", "source_prefix",
                "sample_type", "source_image_path", "source_label_path", "rotation_deg",
            ]
        )
        for split_name, plist in (("train", train_pairs), ("val", val_pairs), ("unused", unused_pairs)):
            for pair in plist:
                image_path, label_path, prefix, sample_type = pair
                stem = pair_stem(pair)
                fid = extract_frame_id(stem)
                output_stem = output_stem_for_pair(pair)
                if split_name == "unused" or rotation_by_output_stem is None:
                    rotation_value = ""
                else:
                    rotation_value = rotation_by_output_stem.get(output_stem, 0)
                writer.writerow(
                    [split_name, output_stem, fid, prefix, sample_type, image_path, label_path, rotation_value]
                )
    return manifest_path


def load_obb_labels(label_path: str, image_w: int, image_h: int) -> List[dict]:
    objects: List[dict] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 9:
                raise ValueError(
                    f"Invalid YOLO OBB label in {label_path}:{line_idx}. "
                    f"Expected 9 fields, got {len(parts)}: {line.strip()}"
                )
            class_id = int(float(parts[0]))
            pts = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(4, 2)
            pts[:, 0] *= float(image_w)
            pts[:, 1] *= float(image_h)
            objects.append({"class_id": class_id, "pts": pts})
    return objects


def yolo_line_from_obb_points(class_id: int, pts: np.ndarray, image_w: int, image_h: int) -> str:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0] / float(image_w), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(image_h), 0.0, 1.0)
    flat = " ".join(f"{float(v):.6f}" for v in pts.reshape(-1))
    return f"{int(class_id)} {flat}\n"


def class_id_to_unit_vec(class_id: int) -> Tuple[float, float]:
    """Heading class id -> unit direction vector (dx, dy) in image coordinates.

    Matches the convention used throughout the pipeline (interaction_image_synthesis.py,
    direction_class_assignment.py, apply_class_label_filtering.py): class 0 is
    "up" (0, -1), classes increase clockwise in 45 degree steps.
    """
    angle_deg = float(int(class_id) % 8) * 45.0
    th = math.radians(angle_deg)
    return math.sin(th), -math.cos(th)


def angle_to_class_id(dx: float, dy: float) -> int:
    """Direction vector (dx, dy) -> nearest heading class id (0-7)."""
    angle_deg = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
    return int(((angle_deg + 22.5) % 360.0) // 45.0)


def rotate_direction_vec_right_angle(dx: float, dy: float, rotation_deg: int) -> Tuple[float, float]:
    """Rotate a direction vector by a clockwise right-angle rotation."""
    rotation_deg = int(rotation_deg)
    if rotation_deg == 0:
        return dx, dy
    if rotation_deg == 90:
        return -dy, dx
    if rotation_deg == 180:
        return -dx, -dy
    if rotation_deg == 270:
        return dy, -dx
    raise ValueError(f"Unsupported rotation_deg={rotation_deg}, expected one of {ALLOWED_ROTATION_ANGLES}")


def rotate_heading_class(*args, **kwargs):
    return 0


def rotate_obb_points_right_angle(pts: np.ndarray, rotation_deg: int, width: int, height: int) -> np.ndarray:
    """Rotate 4 OBB vertices (pixel, boundary coordinates) by a clockwise right angle.

    pts are rotated directly (no cv2.minAreaRect() reconstruction), so vertex
    order and the original rectangle's long/short axes are preserved exactly.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    rotation_deg = int(rotation_deg)
    w = float(width)
    h = float(height)
    if rotation_deg == 0:
        new_x, new_y = x, y
    elif rotation_deg == 90:
        new_x, new_y = h - y, x
    elif rotation_deg == 180:
        new_x, new_y = w - x, h - y
    elif rotation_deg == 270:
        new_x, new_y = y, w - x
    else:
        raise ValueError(f"Unsupported rotation_deg={rotation_deg}, expected one of {ALLOWED_ROTATION_ANGLES}")
    return np.stack([new_x, new_y], axis=1)


def rotate_yolo_obb_label_lines(
    label_lines: Sequence[str],
    rotation_deg: int,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
    *,
    source_description: str,
) -> List[str]:
    """Rotate YOLO OBB label lines to match an image rotated by rotate_yolo_obb_sample().

    Each 4-vertex OBB is rotated directly (rotate_obb_points_right_angle) and
    each heading class is re-derived from its rotated direction vector
    (rotate_heading_class), never from the OBB's unsigned long axis. Results
    are validated before being turned back into YOLO label lines; out-of-range
    coordinates are only clipped when within a 1e-6 floating point tolerance,
    otherwise a ValueError identifying the image/label/angle/coordinates is
    raised instead of silently clipping.
    """
    rotation_deg = int(rotation_deg)
    tol = 1e-6
    new_lines: List[str] = []
    for line_idx, line in enumerate(label_lines, start=1):
        if not str(line).strip():
            continue
        parts = line.strip().split()
        if len(parts) != 9:
            raise ValueError(
                f"Invalid YOLO OBB label line for {source_description} at line {line_idx}: "
                f"expected 9 fields, got {len(parts)}: {line.strip()!r}"
            )
        class_id = int(float(parts[0]))
        if not 0 <= class_id <= 7:
            raise ValueError(
                f"Invalid heading class id for {source_description} at line {line_idx}: {class_id}"
            )

        norm_pts = np.asarray([float(v) for v in parts[1:]], dtype=np.float64).reshape(4, 2)
        pts = norm_pts.copy()
        pts[:, 0] *= float(src_width)
        pts[:, 1] *= float(src_height)

        rotated_pts = rotate_obb_points_right_angle(pts, rotation_deg, src_width, src_height)
        if rotated_pts.shape != (4, 2):
            raise ValueError(
                f"Rotated OBB has invalid shape for {source_description}, "
                f"rotation_deg={rotation_deg}: shape={rotated_pts.shape}"
            )
        if not np.all(np.isfinite(rotated_pts)):
            raise ValueError(
                f"Rotated OBB has non-finite coordinates for {source_description}, "
                f"rotation_deg={rotation_deg}, pts={rotated_pts.tolist()}"
            )

        area2 = 0.0
        for i in range(4):
            xa, ya = rotated_pts[i]
            xb, yb = rotated_pts[(i + 1) % 4]
            area2 += xa * yb - xb * ya
        if abs(area2) <= 1e-9:
            raise ValueError(
                f"Rotated OBB has non-positive area for {source_description}, "
                f"rotation_deg={rotation_deg}, pts={rotated_pts.tolist()}"
            )

        new_class_id = rotate_heading_class(class_id, rotation_deg)
        if not 0 <= new_class_id <= 7:
            raise ValueError(
                f"Rotated heading class id out of range for {source_description}, "
                f"rotation_deg={rotation_deg}: {new_class_id}"
            )

        normalized = rotated_pts.copy()
        normalized[:, 0] /= float(dst_width)
        normalized[:, 1] /= float(dst_height)
        if np.any(normalized < -tol) or np.any(normalized > 1.0 + tol):
            raise ValueError(
                f"Rotated OBB normalized coordinates out of [0,1] range for {source_description}, "
                f"image_w={dst_width}, image_h={dst_height}, rotation_deg={rotation_deg}, "
                f"normalized={normalized.tolist()}"
            )
        normalized = np.clip(normalized, 0.0, 1.0)

        flat = " ".join(f"{float(v):.6f}" for v in normalized.reshape(-1))
        new_lines.append(f"{new_class_id} {flat}\n")
    return new_lines


def rotate_yolo_obb_sample(
    image: np.ndarray,
    label_lines: Sequence[str],
    rotation_deg: int,
    *,
    source_description: str,
) -> Tuple[np.ndarray, List[str]]:
    """Rotate an image and its YOLO OBB label lines together by a right angle.

    Uses cv2.rotate() (ROTATE_90_CLOCKWISE / ROTATE_180 / ROTATE_90_COUNTERCLOCKWISE)
    rather than warpAffine(), so there is no interpolation or border fill. The
    label rotation always uses the same rotation_deg and the actual rotated
    image dimensions, so image and label can never disagree.
    """
    rotation_deg = int(rotation_deg)
    src_h, src_w = image.shape[:2]
    if rotation_deg == 0:
        rotated_image = image
    elif rotation_deg == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_deg == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_deg == 270:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(
            f"Unsupported rotation_deg={rotation_deg} for {source_description}, "
            f"expected one of {ALLOWED_ROTATION_ANGLES}"
        )

    dst_h, dst_w = rotated_image.shape[:2]
    rotated_label_lines = rotate_yolo_obb_label_lines(
        label_lines, rotation_deg, src_w, src_h, dst_w, dst_h,
        source_description=source_description,
    )
    return rotated_image, rotated_label_lines


def make_resized_full_frame_sample(
    image_path: str,
    label_path: str,
    output_size: int,
    background_path: str | None = None,
) -> Tuple[np.ndarray, List[str], bool] | None:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    src_h, src_w = image.shape[:2]
    objects = load_obb_labels(label_path, src_w, src_h)
    if not objects:
        return None

    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError(f"output_size must be a positive integer, got {output_size}")

    background_canvas = make_background_canvas(output_size, image.dtype, background_path)

    if src_w == output_size and src_h == output_size:
        # Already the requested square size. Store without resizing or padding.
        resized = image.copy()
        scale = 1.0
        pad_x = 0.0
        pad_y = 0.0
        image_was_resized = False
    else:
        # Letterbox resize: preserve aspect ratio and fill the remaining area with background.
        scale = min(float(output_size) / float(src_w), float(output_size) / float(src_h))
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        image_was_resized = (new_w != src_w or new_h != src_h)
        if image_was_resized:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized_content = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        else:
            resized_content = image.copy()

        resized = background_canvas.copy()
        pad_x_i = (output_size - new_w) // 2
        pad_y_i = (output_size - new_h) // 2
        resized[pad_y_i:pad_y_i + new_h, pad_x_i:pad_x_i + new_w] = resized_content
        pad_x = float(pad_x_i)
        pad_y = float(pad_y_i)

    label_lines: List[str] = []
    for obj in objects:
        pts = np.asarray(obj["pts"], dtype=np.float32).copy()
        pts[:, 0] = pts[:, 0] * scale + pad_x
        pts[:, 1] = pts[:, 1] * scale + pad_y
        label_lines.append(yolo_line_from_obb_points(int(obj["class_id"]), pts, output_size, output_size))

    return resized, label_lines, image_was_resized


def write_image_and_label(
    image: np.ndarray,
    label_lines: List[str],
    dst_img_dir: str,
    dst_lbl_dir: str,
    out_stem: str,
) -> None:
    image_out = os.path.join(dst_img_dir, f"{out_stem}.png")
    label_out = os.path.join(dst_lbl_dir, f"{out_stem}.txt")
    if not cv2.imwrite(image_out, image):
        raise RuntimeError(f"Failed to write image: {image_out}")
    with open(label_out, "w", encoding="utf-8") as f:
        f.writelines(label_lines)


def collect_full_frame_pairs(src_dir: str, img_ext: str, include: bool) -> List[Pair]:
    if not include:
        return []

    img_dir = os.path.join(src_dir, "images")
    lbl_dir = os.path.join(src_dir, "labels")
    prefix = os.path.basename(os.path.normpath(src_dir))
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        return []

    pairs: List[Pair] = []
    pattern = os.path.join(img_dir, f"*{img_ext}")
    for image_path in tqdm(sorted(glob(pattern)), desc=f"Collecting resized full frames from {prefix}", unit="img"):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(lbl_dir, f"{stem}.txt")
        if os.path.exists(label_path):
            pairs.append((image_path, label_path, prefix, "full_resized"))
    return pairs


def collect_crop_pairs(
    src_dir: str,
    img_ext: str,
    include_crop_highres: bool,
) -> List[Pair]:
    if not include_crop_highres:
        return []

    crop_dir = os.path.join(src_dir, "cropping")
    img_dir = os.path.join(crop_dir, "images")
    lbl_dir = os.path.join(crop_dir, "labels")
    prefix = os.path.basename(os.path.normpath(src_dir))
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        return []

    pairs: List[Pair] = []
    pattern = os.path.join(img_dir, f"*{img_ext}")
    for image_path in tqdm(sorted(glob(pattern)), desc=f"Collecting high-res crops from {prefix}", unit="img"):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(lbl_dir, f"{stem}.txt")
        if not os.path.exists(label_path):
            continue
        pairs.append((image_path, label_path, prefix, "crop_highres"))
    return pairs


def collect_pairs(
    src_dir: str,
    img_ext: str,
    include_full_resized: bool,
    include_crop_highres: bool,
) -> List[Pair]:
    pairs: List[Pair] = []
    pairs.extend(collect_full_frame_pairs(src_dir, img_ext, include_full_resized))
    pairs.extend(collect_crop_pairs(src_dir, img_ext, include_crop_highres))
    return pairs


def write_sample(
    pair: Pair,
    dst_img_dir: str,
    dst_lbl_dir: str,
    output_size: int,
    background_path: str | None = None,
    *,
    rotation_angles: Sequence[int] = (0,),
    seed: int = 0,
    is_train: bool = False,
) -> bool:
    image_path, label_path, prefix, sample_type = pair
    stem = os.path.splitext(os.path.basename(image_path))[0]

    if sample_type == "crop_highres":
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        with open(label_path, "r", encoding="utf-8") as f:
            label_lines = f.readlines()
        out_suffix = "crop_highres"
    elif sample_type == "full_resized":
        sample = make_resized_full_frame_sample(image_path, label_path, output_size, background_path)
        if sample is None:
            return False
        image, label_lines, _image_was_resized = sample
        out_suffix = "full_resized"
    else:
        raise ValueError(f"Unknown sample_type: {sample_type}")

    # Rotation is applied last, after full-frame letterbox resize / crop loading,
    # to whichever of the two image types is being written -- never before, and
    # never separately for image vs. label (see rotate_yolo_obb_sample()).
    rotation_deg = select_rotation_deg_for_pair(
        rotation_angles, seed, prefix, sample_type, stem, is_train=is_train,
    )
    if rotation_deg != 0:
        image, label_lines = rotate_yolo_obb_sample(
            image, label_lines, rotation_deg,
            source_description=f"image={image_path} label={label_path}",
        )

    write_image_and_label(image, label_lines, dst_img_dir, dst_lbl_dir, f"{prefix}_{stem}_{out_suffix}")
    return True


def _write_sample_worker(args: tuple) -> bool:
    pair, dst_img_dir, dst_lbl_dir, output_size, background_path, rotation_angles, seed, is_train = args
    return write_sample(
        pair,
        dst_img_dir,
        dst_lbl_dir,
        output_size,
        background_path,
        rotation_angles=rotation_angles,
        seed=seed,
        is_train=is_train,
    )


def _write_batch(
    pairs: List[Pair],
    dst_img_dir: str,
    dst_lbl_dir: str,
    output_size: int,
    background_path: Optional[str],
    num_workers: int,
    desc: str,
    *,
    rotation_angles: Sequence[int] = (0,),
    seed: int = 0,
    is_train: bool = False,
) -> List[bool]:
    workers = max(1, int(num_workers))
    if workers > 1 and len(pairs) > 1:
        tasks = [
            (p, dst_img_dir, dst_lbl_dir, output_size, background_path, tuple(rotation_angles), seed, is_train)
            for p in pairs
        ]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(
                ex.map(_write_sample_worker, tasks, chunksize=max(1, min(16, math.ceil(len(tasks) / max(1, workers * 8))))),
                total=len(tasks), desc=desc, unit="sample",
            ))
        return [bool(r) for r in results]
    return [
        write_sample(
            p, dst_img_dir, dst_lbl_dir, output_size, background_path,
            rotation_angles=rotation_angles, seed=seed, is_train=is_train,
        )
        for p in tqdm(pairs, desc=desc, unit="sample")
    ]


def write_pairs_with_replenishment(
    pairs: List[Pair],
    unused_pool: List[Pair],
    dst_img_dir: str,
    dst_lbl_dir: str,
    output_size: int,
    background_path: Optional[str],
    num_workers: int,
    desc: str,
    split_name: str,
    *,
    rotation_angles: Sequence[int] = (0,),
    seed: int = 0,
    is_train: bool = False,
) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    """Write pairs, replenishing any write_sample() failures from unused_pool.

    Replenishment only draws from unused_pool, which must already be scoped to
    the same split (train or val) so a replacement can never cross into the
    other split and introduce frame leakage. Returns
    (written_pairs, remaining_unused_pool, permanently_failed_pairs).

    Each replacement pair gets its own deterministic rotation angle from
    select_rotation_deg_for_pair() (keyed on its own prefix/sample_type/stem),
    same as any other pair -- retries and pool order never change what angle a
    given sample receives.
    """
    remaining_pool = list(unused_pool)
    to_write = list(pairs)
    written: List[Pair] = []
    permanently_failed: List[Pair] = []
    while to_write:
        results = _write_batch(
            to_write, dst_img_dir, dst_lbl_dir, output_size, background_path,
            num_workers, desc,
            rotation_angles=rotation_angles, seed=seed, is_train=is_train,
        )
        failed = [p for p, ok in zip(to_write, results) if not ok]
        written.extend(p for p, ok in zip(to_write, results) if ok)
        if not failed:
            break
        permanently_failed.extend(failed)
        if len(remaining_pool) < len(failed):
            raise RuntimeError(
                f"{split_name}: {len(failed)} sample(s) failed to write and only "
                f"{len(remaining_pool)} unused candidate(s) remain in the {split_name} pool for replenishment."
            )
        to_write = remaining_pool[:len(failed)]
        remaining_pool = remaining_pool[len(failed):]
        desc = f"{desc} (replenish)"
    return written, remaining_pool, permanently_failed


def build_rotation_entries(
    pairs: List[Pair],
    rotation_angles: Sequence[int],
    seed: int,
    is_train: bool,
) -> List[Tuple[str, int]]:
    """Recompute (output_stem, rotation_deg) for each written pair.

    Uses the same select_rotation_deg_for_pair() decision as write_sample(),
    so manifest logging and preview grouping never disagree with what was
    actually written to disk.
    """
    entries: List[Tuple[str, int]] = []
    for pair in pairs:
        _image_path, _label_path, prefix, sample_type = pair
        stem = pair_stem(pair)
        rotation_deg = select_rotation_deg_for_pair(
            rotation_angles, seed, prefix, sample_type, stem, is_train=is_train,
        )
        entries.append((output_stem_for_pair(pair), rotation_deg))
    return entries


def log_rotation_angle_counts(train_entries: List[Tuple[str, int]], val_entries: List[Tuple[str, int]]) -> None:
    train_counts = {0: 0, 90: 0, 180: 0, 270: 0}
    val_counts = {0: 0, 90: 0, 180: 0, 270: 0}
    for _stem, rotation_deg in train_entries:
        train_counts[rotation_deg] = train_counts.get(rotation_deg, 0) + 1
    for _stem, rotation_deg in val_entries:
        val_counts[rotation_deg] = val_counts.get(rotation_deg, 0) + 1
    print(
        f"[ROTATION] train angle counts: 0={train_counts[0]}, 90={train_counts[90]}, "
        f"180={train_counts[180]}, 270={train_counts[270]}"
    )
    print(
        f"[ROTATION] validation angle counts: 0={val_counts[0]}, 90={val_counts[90]}, "
        f"180={val_counts[180]}, 270={val_counts[270]}"
    )


def allocate_preview_counts(group_sizes: Dict[Tuple[str, int], int], total: int) -> Dict[Tuple[str, int], int]:
    """Distribute `total` preview slots across groups, deterministically.

    Every non-empty group gets at least one slot (if total allows), then any
    remainder is handed out round-robin in group_sizes iteration order so a
    single angle never monopolizes the preview set.
    """
    keys = list(group_sizes.keys())
    alloc: Dict[Tuple[str, int], int] = {k: 0 for k in keys}
    remaining = max(0, int(total))
    for k in keys:
        if remaining <= 0:
            break
        if group_sizes[k] > 0:
            alloc[k] = 1
            remaining -= 1
    while remaining > 0:
        progressed = False
        for k in keys:
            if remaining <= 0:
                break
            if alloc[k] < group_sizes[k]:
                alloc[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def pick_evenly_spaced(items: List[str], count: int) -> List[str]:
    ordered = sorted(items)
    n = len(ordered)
    if count <= 0 or n == 0:
        return []
    if count >= n:
        return ordered
    if count == 1:
        return [ordered[0]]
    indices = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
    return [ordered[i] for i in indices]


def write_rotation_dataset_previews(
    output_root: str,
    train_entries: List[Tuple[str, int]],
    val_entries: List[Tuple[str, int]],
    num_preview_frames: int,
) -> int:
    """Regenerate <output_root>/preview/ from the final saved train/test files.

    Reads back the images and labels actually written to disk (never the
    in-memory pre-save arrays), so previews always match the real dataset.
    Reuses the existing OBB + heading-triangle drawing routine from
    apply_refine_deletions.draw_preview_from_label_lines() instead of
    duplicating that rendering logic.
    """
    preview_dir = os.path.join(output_root, "preview")
    reset_dir(preview_dir)

    group_order: List[Tuple[str, int]] = [
        ("train", 0), ("train", 90), ("train", 180), ("train", 270),
        ("test", 0), ("test", 90), ("test", 180), ("test", 270),
    ]
    stems_by_group: Dict[Tuple[str, int], List[str]] = {g: [] for g in group_order}
    for stem, rotation_deg in train_entries:
        key = ("train", int(rotation_deg))
        if key in stems_by_group:
            stems_by_group[key].append(stem)
    for stem, rotation_deg in val_entries:
        key = ("test", int(rotation_deg))
        if key in stems_by_group:
            stems_by_group[key].append(stem)

    limit = max(0, int(num_preview_frames))
    if limit <= 0:
        print(f"[PREVIEW] written=0 dir={preview_dir}")
        return 0

    present_groups = [g for g in group_order if stems_by_group[g]]
    total_available = sum(len(stems_by_group[g]) for g in present_groups)
    group_sizes = {g: len(stems_by_group[g]) for g in present_groups}
    alloc = allocate_preview_counts(group_sizes, min(limit, total_available))

    from without_direction_estimation.apply_class_label_filtering import draw_preview_from_label_lines

    written = 0
    for group in present_groups:
        split, rotation_deg = group
        count = alloc.get(group, 0)
        if count <= 0:
            continue
        img_dir = os.path.join(output_root, split, "images")
        lbl_dir = os.path.join(output_root, split, "labels")
        for stem in pick_evenly_spaced(stems_by_group[group], count):
            image_path = os.path.join(img_dir, f"{stem}.png")
            label_path = os.path.join(lbl_dir, f"{stem}.txt")
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read dataset image for preview: {image_path}")
            with open(label_path, "r", encoding="utf-8") as f:
                label_lines = f.readlines()

            preview = draw_preview_from_label_lines(image, label_lines, color=OBB_COLOR)

            out_path = os.path.join(preview_dir, f"{split}_{stem}_rot{int(rotation_deg):03d}.png")
            if not cv2.imwrite(out_path, preview):
                raise RuntimeError(f"Failed to write preview image: {out_path}")
            written += 1

    print(f"[PREVIEW] written={written} dir={preview_dir}")
    return written


def _parse_clustered_ratio(value) -> float:
    """Parse CLUSTERED_RATIO (float 0-1) and return the non-clustered fraction."""
    c = float(value)
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"CLUSTERED_RATIO must be between 0.0 and 1.0, got: {value!r}")
    return 1.0 - c


def _script_path(script_name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)


def _write_stage_config(cfg: dict, overrides: dict, suffix: str) -> str:
    session_path = str(cfg["SESSION_PATH"])
    out_path = os.path.join(session_path, f"_autogen_{suffix}_config.yaml")
    merged = dict(cfg)
    merged.update(overrides)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)
    return out_path


def _run_python_stage(script_name: str, cfg: dict, overrides: dict, suffix: str) -> None:
    script = _script_path(script_name)
    if not os.path.exists(script):
        raise FileNotFoundError(f"Required script not found: {script}")
    stage_cfg = _write_stage_config(cfg, overrides, suffix)
    print(f"Running {script_name} with overrides: {overrides}")
    subprocess.run([sys.executable, script, stage_cfg], check=True)


def _count_labeled_images(img_dir: str, lbl_dir: str, img_ext: str) -> int:
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        return 0
    n = 0
    for image_path in glob(os.path.join(img_dir, f"*{img_ext}")):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        if os.path.exists(os.path.join(lbl_dir, f"{stem}.txt")):
            n += 1
    return n


def _count_candidate_pairs_fast(
    src_dir: str,
    img_ext: str,
    include_full_resized: bool,
    include_crop_highres: bool,
) -> int:
    count = 0
    if include_full_resized:
        count += _count_labeled_images(
            os.path.join(src_dir, "images"),
            os.path.join(src_dir, "labels"),
            img_ext,
        )
    if include_crop_highres:
        count += _count_labeled_images(
            os.path.join(src_dir, "cropping", "images"),
            os.path.join(src_dir, "cropping", "labels"),
            img_ext,
        )
    return count


def _count_pairs_by_group_fast(
    session_path: str,
    img_ext: str,
    include_full_resized: bool,
    include_crop_highres: bool,
    include_single_animal_images: bool = False,
) -> tuple[int, int]:
    nonclustered = 0
    clustered = 0
    src_dirs = [
        os.path.join(session_path, "paste_blobs"),
        os.path.join(session_path, "paste_blobs_clustered"),
        os.path.join(session_path, "clustered"),
    ]
    if include_single_animal_images:
        src_dirs.insert(0, os.path.join(session_path, "single_animal_images"))
    for src_dir in src_dirs:
        if not os.path.isdir(src_dir):
            continue
        n = _count_candidate_pairs_fast(
            src_dir,
            img_ext,
            include_full_resized,
            include_crop_highres,
        )
        if os.path.basename(os.path.normpath(src_dir)) in ("clustered", "paste_blobs_clustered"):
            clustered += n
        else:
            nonclustered += n
    return nonclustered, clustered


def _use_single_animal_images_source(cfg: dict, session_path: str) -> bool:
    try:
        num_objects = int(cfg.get("NUM_OBJECTS", 1))
    except (TypeError, ValueError):
        return False
    if num_objects != 1:
        return False
    if not (
        cfg_bool(cfg, "skip_paste_blobs_with_crossing", False)
        and cfg_bool(cfg, "skip_paste_blobs_clustered", False)
    ):
        return False
    paste_dirs = (
        os.path.join(session_path, "paste_blobs"),
        os.path.join(session_path, "paste_blobs_clustered"),
        os.path.join(session_path, "clustered"),
    )
    return not any(os.path.isdir(p) for p in paste_dirs)


def _images_per_frame_from_cfg(cfg: dict, include_full_resized: bool, include_crop_highres: bool) -> int:
    if cfg_bool(cfg, "skip_cropping", False):
        crop_outputs = 0
    else:
        crop_outputs = int(cfg.get("NUM_CROPS", 2)) * int(bool(include_crop_highres))
    return crop_outputs + int(bool(include_full_resized))


def _generated_frame_count(src_dir: str, img_ext: str) -> int:
    return _count_labeled_images(os.path.join(src_dir, "images"), os.path.join(src_dir, "labels"), img_ext)


def _estimate_additional_frame_target(
    *,
    required_samples: int,
    observed_samples: int,
    observed_frames: int,
    optimistic_images_per_frame: int,
) -> int:
    """Estimate only the missing frame sets to append, with a small safety margin."""
    deficit = max(1, int(required_samples) - int(observed_samples))
    observed_frames = max(0, int(observed_frames))
    optimistic = max(1, int(optimistic_images_per_frame))
    if observed_samples > 0 and observed_frames > 0:
        observed_per_frame = max(0.25, float(observed_samples) / float(observed_frames))
        estimated = math.ceil(float(deficit) / observed_per_frame)
    else:
        estimated = math.ceil(float(deficit) / float(optimistic))
    return max(1, int(math.ceil(estimated * 1.15)))


def _rerun_crop_if_needed(
    cfg: dict,
    needs_crops: bool,
    *,
    append: bool = False,
) -> None:
    if needs_crops and not cfg_bool(cfg, "skip_cropping", False):
        overrides = {"CROP_APPEND": True} if append else {}
        suffix = "crop_append" if append else "crop"
        _run_python_stage("crop_images.py", cfg, overrides, suffix)


def ensure_generation_targets(
    cfg: dict,
    *,
    num_total_images: int,
    nonclustered_ratio: float,
    img_ext: str,
    include_full_resized: bool,
    include_crop_highres: bool,
) -> None:
    """Regenerate paste sources until final candidate counts can satisfy NUM_IMAGES.

    Clustered candidates are generated up to their requested ratio first.  The
    clustered loop is bounded by DATASET_GENERATION_MAX_LOOPS, default 20.  Any
    remaining clustered deficit is filled by increasing the non-clustered
    paste_blobs target.
    """
    if num_total_images <= 0:
        return

    session_path = str(cfg["SESSION_PATH"])
    use_single_animal_images = _use_single_animal_images_source(cfg, session_path)
    max_loops = int(cfg.get("DATASET_GENERATION_MAX_LOOPS", 20))
    max_loops = max(1, max_loops)
    needs_crops = bool(include_crop_highres)
    optimistic_images_per_frame = _images_per_frame_from_cfg(
        cfg, include_full_resized, include_crop_highres
    )
    if optimistic_images_per_frame <= 0:
        raise ValueError(
            "No dataset outputs are enabled: USE_FULL and USE_CROP "
            "are both false or ineffective."
        )

    nc_target = math.ceil(num_total_images * nonclustered_ratio)
    c_target = num_total_images - nc_target

    if c_target > 0 and not use_single_animal_images:
        for loop_index in range(max_loops + 1):
            nc_count, c_count = _count_pairs_by_group_fast(
                session_path,
                img_ext,
                include_full_resized,
                include_crop_highres,
                include_single_animal_images=use_single_animal_images,
            )
            if c_count >= c_target:
                print(f"Clustered target satisfied: {c_count}/{c_target} samples.")
                break
            if loop_index >= max_loops:
                print(
                    f"Clustered target not satisfied after {max_loops} loops: "
                    f"{c_count}/{c_target}. The deficit will be filled from paste_blobs."
                )
                break
            clustered_dir = os.path.join(session_path, "paste_blobs_clustered")
            observed_frames = _generated_frame_count(clustered_dir, img_ext)
            next_frames = _estimate_additional_frame_target(
                required_samples=c_target,
                observed_samples=c_count,
                observed_frames=observed_frames,
                optimistic_images_per_frame=optimistic_images_per_frame,
            )
            print(
                f"Clustered shortfall loop {loop_index + 1}/{max_loops}: "
                f"{c_count}/{c_target} samples, requesting {next_frames} additional clustered frames."
            )
            _run_python_stage(
                "interaction_image_synthesis_clustered.py",
                cfg,
                {
                    "CLUSTER_FRAMES": int(next_frames),
                    "CLUSTER_APPEND": True,
                    "RANDOM_SEED": derive_seed(
                        normalize_seed(cfg.get("RANDOM_SEED", 0)),
                        "create_dataset",
                        "clustered_supplement",
                        loop_index + 1,
                    ),
                },
                "paste_blobs_clustered",
            )
            _rerun_crop_if_needed(cfg, needs_crops, append=True)

    _, final_c_count = _count_pairs_by_group_fast(
        session_path,
        img_ext,
        include_full_resized,
        include_crop_highres,
        include_single_animal_images=use_single_animal_images,
    )
    clustered_used = min(c_target, final_c_count)
    required_nc = num_total_images - clustered_used

    if not use_single_animal_images:
        for loop_index in range(max_loops + 1):
            nc_count, c_count = _count_pairs_by_group_fast(
                session_path,
                img_ext,
                include_full_resized,
                include_crop_highres,
                include_single_animal_images=use_single_animal_images,
            )
            clustered_used = min(c_target, c_count)
            required_nc = num_total_images - clustered_used
            if nc_count >= required_nc:
                print(f"Non-clustered target satisfied: {nc_count}/{required_nc} samples.")
                return
            if loop_index >= max_loops:
                raise RuntimeError(
                    f"paste_blobs did not produce enough candidates after {max_loops} loops: "
                    f"non-clustered={nc_count}/{required_nc}, clustered={c_count}/{c_target}."
                )
            paste_dir = os.path.join(session_path, "paste_blobs")
            observed_frames = _generated_frame_count(paste_dir, img_ext)
            next_frames = _estimate_additional_frame_target(
                required_samples=required_nc,
                observed_samples=nc_count,
                observed_frames=observed_frames,
                optimistic_images_per_frame=optimistic_images_per_frame,
            )
            print(
                f"paste_blobs shortfall loop {loop_index + 1}/{max_loops}: "
                f"{nc_count}/{required_nc} samples, requesting {next_frames} additional non-clustered frames."
            )
            _run_python_stage(
                "interaction_image_synthesis.py",
                cfg,
                {
                    "PASTE_BLOBS_NUM_FRAMES": int(next_frames),
                    "PASTE_BLOBS_APPEND": True,
                    "RANDOM_SEED": derive_seed(
                        normalize_seed(cfg.get("RANDOM_SEED", 0)),
                        "create_dataset",
                        "mixed_supplement",
                        loop_index + 1,
                    ),
                },
                "paste_blobs",
            )
            _rerun_crop_if_needed(cfg, needs_crops, append=True)

    nc_count, c_count = _count_pairs_by_group_fast(
        session_path,
        img_ext,
        include_full_resized,
        include_crop_highres,
        include_single_animal_images=use_single_animal_images,
    )
    clustered_used = min(c_target, c_count)
    required_nc = num_total_images - clustered_used
    if nc_count < required_nc:
        raise RuntimeError(
            f"Insufficient dataset candidates and paste_blobs generation is skipped: "
            f"non-clustered={nc_count}/{required_nc}, clustered={c_count}/{c_target}."
        )


def write_yolo_dataset(
    dataset_dirs: List[str],
    output_root: str,
    img_ext: str,
    val_ratio: float,
    num_total_images: int,
    nonclustered_ratio: float,
    output_size: int,
    seed: int,
    include_full_resized: bool,
    include_crop_highres: bool,
    num_workers: int = 1,
    background_path: str | None = None,
    allow_small_dataset: bool = False,
    min_one_val: bool = False,
    candidate_target: Optional[int] = None,
    best_effort: bool = False,
    rotation_angles: Sequence[int] = (0,),
    num_preview_frames: int = 0,
) -> None:
    # Defaults to no rotation so callers that do not opt in are unaffected.
    # create_dataset.py::main() and identify_incorrect_class_labels.py pass the configured
    # rotation_angles for direction-class datasets.
    rotation_angles = normalize_rotation_angles(list(rotation_angles))
    reset_dir(output_root)
    train_img_dir = os.path.join(output_root, "train", "images")
    train_lbl_dir = os.path.join(output_root, "train", "labels")
    test_img_dir = os.path.join(output_root, "test", "images")
    test_lbl_dir = os.path.join(output_root, "test", "labels")
    ensure_dirs([train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir])
    write_yolo_data_yaml(output_root, DIRECTION_CLASS_NAMES)

    nonclustered_pairs: List[Pair] = []
    clustered_pairs: List[Pair] = []
    for src_dir in dataset_dirs:
        src_name = os.path.basename(os.path.normpath(src_dir))
        src_pairs = collect_pairs(
            src_dir=src_dir,
            img_ext=img_ext,
            include_full_resized=include_full_resized,
            include_crop_highres=include_crop_highres,
        )
        if src_name in ("clustered", "paste_blobs_clustered"):
            clustered_pairs.extend(src_pairs)
        else:
            nonclustered_pairs.extend(src_pairs)

    make_python_rng(seed, "create_dataset", "shuffle_pool", "nonclustered").shuffle(nonclustered_pairs)
    make_python_rng(seed, "create_dataset", "shuffle_pool", "clustered").shuffle(clustered_pairs)

    if num_total_images > 0:
        # Frame-group aware split: every sample derived from the same source
        # video frame (both clustered and non-clustered, every paste "set",
        # every crop) is forced onto the same side of the split so that
        # backgrounds/individuals/poses/frames are never shared between train
        # and validation. Candidates are NOT truncated to NUM_IMAGES
        # before splitting; frame-group selection happens over the full
        # candidate pool first, and exact counts are enforced afterwards.
        all_pairs = nonclustered_pairs + clustered_pairs
        total_collected = len(all_pairs)
        print(
            f"Collected {total_collected} YOLO OBB candidate samples "
            f"(non-clustered={len(nonclustered_pairs)}, clustered={len(clustered_pairs)}, "
            f"full_resized={include_full_resized}, crop_highres={include_crop_highres})"
        )

        train_pairs, val_pairs, train_unused, val_unused, stats = build_frame_based_split(
            all_pairs,
            num_total_images=num_total_images,
            val_ratio=val_ratio,
            nonclustered_ratio=nonclustered_ratio,
            seed=seed,
            best_effort=best_effort,
        )
        n_train_target = stats["n_train_target"]
        n_val_target = stats["n_val_target"]
        n_train = len(train_pairs)
        n_val = len(val_pairs)

        pre_train_fids = {pair_frame_id(p) for p in train_pairs}
        pre_val_fids = {pair_frame_id(p) for p in val_pairs}
        assert pre_train_fids.isdisjoint(pre_val_fids), "Frame leakage detected between train/validation selections."
        if not best_effort:
            assert n_train == n_train_target, f"train pool size mismatch: {n_train} != {n_train_target}"
            assert n_val == n_val_target, f"val pool size mismatch: {n_val} != {n_val_target}"

        print(f"[FRAME SPLIT] target: total={num_total_images} train={n_train_target} val={n_val_target}")
        print(
            f"[FRAME SPLIT] candidate_target={candidate_target if candidate_target is not None else 'n/a'} "
            f"candidates={stats['total_candidates']} frames={stats['num_frames']} "
            f"reserve={stats['total_candidates'] - num_total_images}"
        )
        print(
            f"[FRAME SPLIT] val_pool={stats['val_pool_size']} samples/{len(stats['val_frame_ids'])} frames "
            f"-> selected={n_val}"
        )
        print(
            f"[FRAME SPLIT] train_pool={stats['train_pool_size']} samples/{len(stats['train_frame_ids'])} frames "
            f"-> selected={n_train}"
        )
        if best_effort and (n_train != n_train_target or n_val != n_val_target):
            print(
                "[FRAME SPLIT] WARNING: best-effort split in use (exact frame-group split was not found "
                f"after regeneration retries). target: train={n_train_target} val={n_val_target} "
                f"total={num_total_images}; actual: train={n_train} val={n_val} total={n_train + n_val} "
                f"(diff={n_train + n_val - num_total_images:+d})"
            )
        vc, tc = stats["val_counts"], stats["train_counts"]
        print(
            f"[FRAME SPLIT] train clustered={tc['clustered']} non-clustered={tc['nonclustered']} "
            f"full_resized={tc['full_resized']} crop_highres={tc['crop_highres']}"
        )
        print(
            f"[FRAME SPLIT] val clustered={vc['clustered']} non-clustered={vc['nonclustered']} "
            f"full_resized={vc['full_resized']} crop_highres={vc['crop_highres']}"
        )

        train_written, train_unused_left, train_failed = write_pairs_with_replenishment(
            train_pairs, train_unused, train_img_dir, train_lbl_dir, output_size, background_path,
            num_workers, "Writing train", "train",
            rotation_angles=rotation_angles, seed=seed, is_train=True,
        )
        val_written, val_unused_left, val_failed = write_pairs_with_replenishment(
            val_pairs, val_unused, test_img_dir, test_lbl_dir, output_size, background_path,
            num_workers, "Writing val", "val",
            rotation_angles=rotation_angles, seed=seed, is_train=False,
        )

        written_train = len(train_written)
        written_val = len(val_written)
        unused_pairs = train_unused_left + val_unused_left + train_failed + val_failed

        post_train_fids = {pair_frame_id(p) for p in train_written}
        post_val_fids = {pair_frame_id(p) for p in val_written}
        leakage = len(post_train_fids & post_val_fids)
        print(f"[FRAME SPLIT] unused={len(unused_pairs)}")
        print(f"[FRAME SPLIT] leakage={leakage}")

        # Frame leakage is never acceptable, even in best-effort mode.
        if leakage != 0:
            raise RuntimeError(f"Frame leakage detected between written train/validation sets: {leakage} frame(s).")

        if not best_effort:
            if written_train != n_train:
                raise RuntimeError(f"written_train={written_train} != n_train={n_train}")
            if written_val != n_val:
                raise RuntimeError(f"written_val={written_val} != n_val={n_val}")
            if written_train + written_val != num_total_images:
                raise RuntimeError(
                    f"written_train+written_val={written_train + written_val} != NUM_IMAGES={num_total_images}"
                )
        else:
            # Best-effort: writing must still match what was actually selected;
            # only the deviation from the original NUM_IMAGES target is
            # tolerated (and was already warned about above).
            if written_train != n_train:
                raise RuntimeError(f"written_train={written_train} != selected train count={n_train}")
            if written_val != n_val:
                raise RuntimeError(f"written_val={written_val} != selected val count={n_val}")
            if written_train + written_val != num_total_images:
                print(
                    f"[FRAME SPLIT] WARNING: final dataset total={written_train + written_val} "
                    f"!= requested NUM_IMAGES={num_total_images}"
                )

        train_entries = build_rotation_entries(train_written, rotation_angles, seed, True)
        val_entries = build_rotation_entries(val_written, rotation_angles, seed, False)
        log_rotation_angle_counts(train_entries, val_entries)
        rotation_by_output_stem = {**dict(train_entries), **dict(val_entries)}

        manifest_path = write_split_manifest(
            output_root, train_written, val_written, unused_pairs, rotation_by_output_stem,
        )
        write_rotation_dataset_previews(output_root, train_entries, val_entries, num_preview_frames)

        print("Done.")
        print(f"  output_root={output_root}")
        print(f"  train={written_train}")
        print(f"  test={written_val}")
        print(f"  data_yaml={os.path.join(output_root, 'data.yaml')}")
        print(f"  split_manifest={manifest_path}")
        return

    # NUM_IMAGES <= 0: legacy unlimited-candidate, sample-level split.
    # Used as-is by callers (e.g. identify_incorrect_class_labels.py) that intentionally build a
    # small ad-hoc dataset from a single source directory.
    pairs = nonclustered_pairs + clustered_pairs
    total_nc = len(nonclustered_pairs)
    total_c = len(clustered_pairs)
    total_collected = total_nc + total_c
    print(
        f"Collected {total_collected} YOLO OBB samples "
        f"(non-clustered={total_nc}, clustered={total_c}, "
        f"full_resized={include_full_resized}, crop_highres={include_crop_highres})"
    )

    train_pairs, test_pairs = split_train_val(
        pairs,
        val_ratio,
        seed,
        allow_small_dataset=allow_small_dataset,
        min_one_val=min_one_val,
    )

    train_results = _write_batch(
        train_pairs, train_img_dir, train_lbl_dir, output_size, background_path,
        num_workers, "Writing train",
        rotation_angles=rotation_angles, seed=seed, is_train=True,
    )
    test_results = _write_batch(
        test_pairs, test_img_dir, test_lbl_dir, output_size, background_path,
        num_workers, "Writing test",
        rotation_angles=rotation_angles, seed=seed, is_train=False,
    )
    written_train = sum(int(r) for r in train_results)
    written_test = sum(int(r) for r in test_results)

    train_written = [p for p, ok in zip(train_pairs, train_results) if ok]
    test_written = [p for p, ok in zip(test_pairs, test_results) if ok]
    train_entries = build_rotation_entries(train_written, rotation_angles, seed, True)
    val_entries = build_rotation_entries(test_written, rotation_angles, seed, False)
    log_rotation_angle_counts(train_entries, val_entries)
    write_rotation_dataset_previews(output_root, train_entries, val_entries, num_preview_frames)

    print("Done.")
    print(f"  output_root={output_root}")
    print(f"  train={written_train}")
    print(f"  test={written_test}")
    print(f"  data_yaml={os.path.join(output_root, 'data.yaml')}")


def default_yolo_dataset_dir_name(num_total_images: int) -> str:
    """Default YOLO dataset directory name, tagged with its requested sample count.

    e.g. NUM_IMAGES=10000 -> "yolo_dataset_10000". Kept as a single
    source of truth (imported by obb_detector_training.py) so the writer and the reader
    can never disagree on where the dataset lives.
    """
    return f"yolo_dataset_{int(num_total_images)}"


def resolve_yolo_dataset_dir(cfg: dict, session_path: str) -> str:
    """Resolve the YOLO dataset directory: explicit YOLO_DATASET_DIR wins, else the default name."""
    num_total_images = int(cfg.get("NUM_IMAGES", 0))
    default_dir = os.path.join(session_path, default_yolo_dataset_dir_name(num_total_images))
    return str(cfg.get("YOLO_DATASET_DIR", default_dir))


def resolve_dataset_dirs(cfg: dict, session_path: str) -> List[str]:
    explicit = cfg.get("CREATE_DATASET_SOURCE_DIRS", None)
    if explicit is not None:
        if isinstance(explicit, str):
            dataset_dirs = [explicit]
        else:
            dataset_dirs = [str(x) for x in explicit]
    elif _use_single_animal_images_source(cfg, session_path):
        dataset_dirs = [os.path.join(session_path, "single_animal_images")]
    else:
        dataset_dirs = [
            os.path.join(session_path, "paste_blobs"),
            os.path.join(session_path, "paste_blobs_clustered"),
            os.path.join(session_path, "clustered"),
        ]

    valid_dirs = [p for p in dataset_dirs if os.path.isdir(p)]
    missing_dirs = [p for p in dataset_dirs if not os.path.isdir(p)]
    for p in missing_dirs:
        print(f"Warning: source directory not found, skipped: {p}")

    if not valid_dirs:
        raise RuntimeError(
            "No valid source directories found. Expected paste_blobs or clustered output under "
            f"{session_path}"
        )
    return valid_dirs


def _delete_tmp_dirs(session_path: str) -> None:
    for name in ("initial_tracking", "paste_blobs", "paste_blobs_clustered", "single_animal_images"):
        path = os.path.join(session_path, name)
        if os.path.isdir(path):
            print(f"Deleting tmp directory: {path}")
            shutil.rmtree(path)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python create_dataset_paste_blobs_yolo.py config.yaml")

    cfg = load_yaml(sys.argv[1])
    create_dataset = not cfg_bool(cfg, "skip_creating_direction_dataset", False)
    if not create_dataset:
        print("YOLO direction dataset creation is skipped.")
        return

    session_path = str(cfg["SESSION_PATH"])
    output_size = int(cfg["TRAIN_IMG_SIZE"])
    val_ratio = validate_val_ratio(float(cfg["VAL_RATIO"]))
    num_total_images = int(cfg.get("NUM_IMAGES", 0))
    nonclustered_ratio = _parse_clustered_ratio(cfg.get("CLUSTERED_RATIO", 0.05))
    seed = normalize_seed(cfg.get("RANDOM_SEED", 0))
    print(f"[SEED] create_dataset master={seed}")
    rotation_angles = normalize_rotation_angles(
        cfg.get("CREATE_DATASET_ROTATION_ANGLES", list(ALLOWED_ROTATION_ANGLES))
    )
    num_preview_frames = int(cfg.get("NUM_PREVIEW_FRAMES", 0))

    output_root = resolve_yolo_dataset_dir(cfg, session_path)
    img_ext = str(cfg.get("IMG_EXT", ".png"))

    include_full_resized = cfg_bool(cfg, "USE_FULL", True)
    include_crop_highres = cfg_bool(cfg, "USE_CROP", True)
    use_single_animal_images = _use_single_animal_images_source(cfg, session_path)

    background_path = cfg.get("BACKGROUND_PATH", None)
    if background_path is not None:
        background_path = str(background_path)

    num_workers = _resolve_num_workers_bt(cfg.get("NUM_WORKERS", "auto"), task="process")

    def _generate_and_resolve(target_num_images: int) -> List[str]:
        if use_single_animal_images:
            print("Single-animal paste sources are absent; using single_animal_images for YOLO dataset.")
        else:
            ensure_generation_targets(
                cfg,
                num_total_images=target_num_images,
                nonclustered_ratio=nonclustered_ratio,
                img_ext=img_ext,
                include_full_resized=include_full_resized,
                include_crop_highres=include_crop_highres,
            )
        dirs = resolve_dataset_dirs(cfg, session_path)
        print("YOLO dataset sources:")
        for p in dirs:
            print(f"  - {p}")
        return dirs

    if use_single_animal_images and num_total_images > 0:
        dataset_dirs = _generate_and_resolve(num_total_images)
        available_candidates = sum(
            _count_candidate_pairs_fast(
                p,
                img_ext,
                include_full_resized,
                include_crop_highres,
            )
            for p in dataset_dirs
        )
        if available_candidates < 2:
            raise RuntimeError(
                "Not enough single_animal_images samples for train/test split: "
                f"{available_candidates} candidate(s)."
            )
        effective_total = min(num_total_images, available_candidates)
        if effective_total < num_total_images:
            print(
                "Requested NUM_IMAGES exceeds single-animal candidates; "
                f"using {effective_total}/{num_total_images} available samples."
            )
        try:
            write_yolo_dataset(
                dataset_dirs=dataset_dirs,
                output_root=output_root,
                img_ext=img_ext,
                val_ratio=val_ratio,
                num_total_images=effective_total,
                nonclustered_ratio=nonclustered_ratio,
                output_size=output_size,
                seed=seed,
                include_full_resized=include_full_resized,
                include_crop_highres=include_crop_highres,
                num_workers=num_workers,
                background_path=background_path,
                candidate_target=available_candidates,
                rotation_angles=rotation_angles,
                num_preview_frames=num_preview_frames,
            )
        except SplitInfeasibleError as e:
            print(
                "[FRAME SPLIT] WARNING: exact frame-group split was not feasible for "
                f"single-animal samples ({e}). Continuing with best-effort split."
            )
            write_yolo_dataset(
                dataset_dirs=dataset_dirs,
                output_root=output_root,
                img_ext=img_ext,
                val_ratio=val_ratio,
                num_total_images=effective_total,
                nonclustered_ratio=nonclustered_ratio,
                output_size=output_size,
                seed=seed,
                include_full_resized=include_full_resized,
                include_crop_highres=include_crop_highres,
                num_workers=num_workers,
                background_path=background_path,
                candidate_target=available_candidates,
                best_effort=True,
                rotation_angles=rotation_angles,
                num_preview_frames=num_preview_frames,
            )
    elif num_total_images > 0:
        reserve_ratio = float(cfg.get("DATASET_SPLIT_RESERVE_RATIO", 0.02))
        reserve_min = int(cfg.get("DATASET_SPLIT_RESERVE_MIN", 100))
        max_retries = int(cfg.get("DATASET_SPLIT_MAX_RETRIES", 5))
        reserve_step = max(reserve_min, math.ceil(num_total_images * reserve_ratio))
        candidate_total = num_total_images + reserve_step
        n_val = round(num_total_images * val_ratio)
        n_train = num_total_images - n_val

        attempt = 0
        while True:
            dataset_dirs = _generate_and_resolve(candidate_total)
            try:
                write_yolo_dataset(
                    dataset_dirs=dataset_dirs,
                    output_root=output_root,
                    img_ext=img_ext,
                    val_ratio=val_ratio,
                    num_total_images=num_total_images,
                    nonclustered_ratio=nonclustered_ratio,
                    output_size=output_size,
                    seed=seed,
                    include_full_resized=include_full_resized,
                    include_crop_highres=include_crop_highres,
                    num_workers=num_workers,
                    background_path=background_path,
                    candidate_target=candidate_total,
                    rotation_angles=rotation_angles,
                    num_preview_frames=num_preview_frames,
                )
                break
            except SplitInfeasibleError as e:
                attempt += 1
                if attempt > max_retries:
                    d = e.details
                    print(
                        "[FRAME SPLIT] WARNING: could not find an exact frame-group train/validation split "
                        f"after {max_retries} regeneration retries. "
                        f"NUM_IMAGES={num_total_images}, n_train={n_train}, n_val={n_val}, "
                        f"total_candidate_count={d.get('total_candidates', 'n/a')}, "
                        f"number_of_frame_groups={d.get('num_frame_groups', 'n/a')}, "
                        f"min_frame_group_size={d.get('min_frame_group_size', 'n/a')}, "
                        f"max_frame_group_size={d.get('max_frame_group_size', 'n/a')}, "
                        f"current_reserve={candidate_total - num_total_images}. "
                        f"Last failure: {e}. "
                        "Continuing with the closest achievable best-effort split instead of stopping."
                    )
                    write_yolo_dataset(
                        dataset_dirs=dataset_dirs,
                        output_root=output_root,
                        img_ext=img_ext,
                        val_ratio=val_ratio,
                        num_total_images=num_total_images,
                        nonclustered_ratio=nonclustered_ratio,
                        output_size=output_size,
                        seed=seed,
                        include_full_resized=include_full_resized,
                        include_crop_highres=include_crop_highres,
                        num_workers=num_workers,
                        background_path=background_path,
                        candidate_target=candidate_total,
                        best_effort=True,
                        rotation_angles=rotation_angles,
                        num_preview_frames=num_preview_frames,
                    )
                    break
                candidate_total += reserve_step
                print(
                    f"[FRAME SPLIT] retry {attempt}/{max_retries}: infeasible split ({e}); "
                    f"increasing candidate_total to {candidate_total}"
                )
    else:
        dataset_dirs = _generate_and_resolve(num_total_images)
        write_yolo_dataset(
            dataset_dirs=dataset_dirs,
            output_root=output_root,
            img_ext=img_ext,
            val_ratio=val_ratio,
            num_total_images=num_total_images,
            nonclustered_ratio=nonclustered_ratio,
            output_size=output_size,
            seed=seed,
            include_full_resized=include_full_resized,
            include_crop_highres=include_crop_highres,
            num_workers=num_workers,
            background_path=background_path,
            rotation_angles=rotation_angles,
            num_preview_frames=num_preview_frames,
        )

    if cfg_bool(cfg, "delete_tmp_files", False):
        _delete_tmp_dirs(session_path)


if __name__ == "__main__":
    from without_direction_estimation import prepare_config
    if len(sys.argv) > 1:
        sys.argv[1] = prepare_config(sys.argv[1])
    main()

