# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""ID-switch correction for OBB tracking with optional contrastive embedding."""
from __future__ import annotations

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)


import json
import os
import pickle
import shutil
import sys
from bisect import bisect_left, bisect_right
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
import cv2
import numpy as np
import psutil
from batch_utils import tqdm
import math

from tracking_artifacts import (
    artifact_filename as _artifact_filename,
    artifact_path as _artifact_path,
)

from without_direction_estimation.obb_detection import (
    # Config / job resolution
    load_config,
    build_tracking_out_dir,
    resolve_model_name,
    parse_weight_spec,
    # Geometry
    iou_obb,
    obb_center,
    ensure_clockwise,
    unit_vec_to_angle_deg,
)
from experiment_utils import (
    experiment_dir_name_from_cfg,
    resolve_existing_experiment_dir_name,
)
from assign_types import (
    ATYPE_DIR_FLIP,
    ATYPE_EMBED_ABSTAIN,
    ATYPE_EMBED_IDENTITY,
    ATYPE_EMBED_RELABEL,
    ATYPE_FILTER_FILL,
    ATYPE_GAP_DET,
    ATYPE_KF_FILL,
    ATYPE_MISSING,
    ATYPE_OVERLAP_FILL,
    ATYPE_POS_FIX,
    ATYPE_POST_FILL,
    ATYPE_PRE_FILL,
    ATYPE_VEL_DIST,
    REFINEMENT_EXCLUDED_ASSIGN_CODES,
)
from without_direction_estimation.multi_staged_association import (
    # Config / job resolution
    resolve_tracking_jobs_for_requested_weights,
    resolve_num_workers,
    _worker_budget,
    save_tracking_outputs,
    save_final_result_csv,
    output_has_data,
    # Constants
    FWD_ARTIFACT_SUFFIX,
    BUFFERS_PICKLE_NAME,
    CORRECTIONS_LOG_NAME,
    OBB_SOURCE_TRACK_INTERPOLATED,
    # Geometry
    obb_points_to_rect,
    obb_aabb,
    aabbs_overlap,
    obb_to_tuple,
    tuple_to_obb,
    valid_obb,
    angular_distance_deg,
    canonicalize_obb_points,
    closest_long_axis_direction,
    direction_vec_from_kf,
    # Kalman filters
    create_position_kf,
    create_bbox_shape_kf,
    create_direction_kf,
    predict_obb_from_filters,
)
from random_utils import derive_seed, normalize_seed
from tracking_constants import FIXED_INTERACT_IOU
FILLED_ARTIFACT_SUFFIX = 'filled'
ID_RESOLVED_ARTIFACT_SUFFIX = 'id_resolved'

# A spawned correction process imports this module and its tracking dependencies
# before receiving a task.  Capture that real import footprint once, then add the
# per-ID payload estimate when deciding how many processes fit in available RAM.
# This replaces the generic fixed-GiB-per-worker assumption for correction work.
_CORRECTION_PROCESS_BASE_BYTES = max(
    64 * 1024 ** 2,
    int(psutil.Process().memory_info().rss),
)
_CORRECTION_RAM_BUDGET_FRACTION = 0.75
_CORRECTION_PAYLOAD_HEADROOM = 1.25
_TRACKING_PICKLE_MEMORY_FACTOR = 10.0

# How many IDs _id_payload_sizes measures exactly before extrapolating the
# rest.  Comfortably above any realistic inner worker count, so the payloads
# that actually decide the worker cap are never estimates.
_PAYLOAD_SAMPLE_IDS = 16

def tqdm_it(*args, **kwargs):
    kw = dict(file=sys.stdout, dynamic_ncols=True, mininterval=0.2)
    kw.update(kwargs)
    return tqdm(*args, **kw)



def load_tracking_buffers(
    out_dir: str,
    artifact_suffix: str = FWD_ARTIFACT_SUFFIX,
) -> dict:
    path = _artifact_path(out_dir, BUFFERS_PICKLE_NAME, artifact_suffix)
    with open(path, 'rb') as f:
        return pickle.load(f)


def circular_interpolate_deg(a: float, b: float, alpha: float) -> float:
    delta = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    return float(float(a) + float(alpha) * delta) % 360.0


def _cfg_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def _source_video_fps(video_path: str) -> float:
    video_path = str(video_path or '').strip()
    if not video_path:
        raise RuntimeError('Could not read FPS: video path is empty.')
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f'Could not read a positive FPS from video: {video_path}')
    return fps


def _resolve_flip_frame_threshold(
    analysis: dict,
    video_path: str,
) -> tuple[int, float, float]:
    """Return frame threshold derived from analysis.FLIP_SEC and video FPS."""
    duration_sec = float(analysis['FLIP_SEC'])
    if not np.isfinite(duration_sec) or duration_sec <= 0.0:
        raise ValueError(
            f'analysis.FLIP_SEC must be > 0, got {analysis["FLIP_SEC"]!r}.'
        )
    fps = _source_video_fps(video_path)
    frames = max(1, int(round(float(fps) * duration_sec)))
    return frames, duration_sec, fps


# Assign-types that mark risky or synthetic anchors - excluded from KF-fill gate G2.
_RISKY_OR_SYNTH_ATYPES: frozenset[int] = frozenset({
    ATYPE_VEL_DIST,      # S5 velocity-gated distance match
    ATYPE_GAP_DET,       # S6 dormant-track recovery
    ATYPE_PRE_FILL,      # pre-correction interpolation
    ATYPE_POST_FILL,     # post-correction fill
    ATYPE_FILTER_FILL,   # GUI post-filter interpolation
    ATYPE_KF_FILL,       # correction KF-gated gap fill
    ATYPE_OVERLAP_FILL,  # correction overlap gap fill
})

# Local sub-episode decomposition constants
# Trigger atypes that seed local pair extraction for oversized episodes.
# ATYPE_MISSING is intentionally excluded: it is assigned after embedding
# and missing frames have no OBB, so _is_non_s1_frame returns False for them.
_LOCAL_TRIGGER_ATYPES: frozenset[int] = frozenset({
    ATYPE_KF_FILL, ATYPE_OVERLAP_FILL,
    ATYPE_PRE_FILL, ATYPE_POST_FILL, ATYPE_FILTER_FILL,
})


def _frame_has_fill_trigger(corrected: dict, tid: int, frame: int) -> bool:
    """Return True when current provenance or its history records a fill type."""
    atype = _get_atype(corrected, int(tid), int(frame))
    if (
        np.isfinite(atype)
        and int(round(float(atype))) in _LOCAL_TRIGGER_ATYPES
    ):
        return True

    history = (
        corrected.get('assign_type_history_buf', {})
        .get(int(tid), {})
        .get(int(frame), [])
    )
    for event in history:
        raw_code = event[1] if isinstance(event, (tuple, list)) and len(event) >= 2 else event
        try:
            code = float(raw_code)
        except (TypeError, ValueError):
            continue
        if np.isfinite(code) and int(round(code)) in _LOCAL_TRIGGER_ATYPES:
            return True
    return False

def _get_atype(buffers: dict, tid: int, frame: int) -> float:
    raw = buffers.get('assign_type_buf', {}).get(tid, {}).get(frame)
    if raw is None:
        return np.nan
    v = raw[0] if isinstance(raw, (tuple, list)) else raw
    return float(v) if v is not None else np.nan


def _record_assign_type(
    buffers: dict,
    tid: int,
    frame: int,
    assign_type: int | float,
    history_source: str,
) -> None:
    tid = int(tid)
    frame = int(frame)
    code = float(assign_type)
    buffers.setdefault('assign_type_buf', {}).setdefault(tid, {})[frame] = (code,)
    buffers.setdefault('assign_type_history_buf', {}).setdefault(tid, {}).setdefault(frame, []).append(
        (str(history_source), code)
    )


# Corrected-output contract

def required_filled_outputs(out_dir: str) -> list[tuple[str, str]]:
    return [
        ('pickle', _artifact_path(out_dir, BUFFERS_PICKLE_NAME, FILLED_ARTIFACT_SUFFIX)),
        ('csv', os.path.join(out_dir, 'obbs.csv')),
        ('csv', os.path.join(out_dir, 'directions.csv')),
        ('file', _artifact_path(out_dir, CORRECTIONS_LOG_NAME, FILLED_ARTIFACT_SUFFIX)),
    ]



def required_id_resolved_outputs(out_dir: str) -> list[tuple[str, str]]:
    return [
        ('pickle', _artifact_path(out_dir, BUFFERS_PICKLE_NAME, ID_RESOLVED_ARTIFACT_SUFFIX)),
        ('csv', os.path.join(out_dir, 'obbs_id_resolved.csv')),
        ('csv', os.path.join(out_dir, 'directions_id_resolved.csv')),
        ('file', _artifact_path(out_dir, CORRECTIONS_LOG_NAME, ID_RESOLVED_ARTIFACT_SUFFIX)),
    ]



def save_corrected_outputs(
    out_dir: str,
    buffers: dict,
    num_objects: int,
    corrections_log: list[dict] | None = None,
    artifact_suffix: str = ID_RESOLVED_ARTIFACT_SUFFIX,
    *,
    save_outputs=None,
) -> None:
    if save_outputs is None:
        save_outputs = save_tracking_outputs
    save_outputs(
        out_dir, buffers, num_objects,
        artifact_suffix=artifact_suffix,
        save_csv=True,
        save_assign_type_pickle=True,
        save_provenance_csv=True,
    )
    if corrections_log is not None:
        log_path = _artifact_path(out_dir, CORRECTIONS_LOG_NAME, artifact_suffix)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(corrections_log, f, indent=2, default=str)


def save_existing_corrected_final_result(spec: dict, family: str, *, save_final=None) -> str:
    if save_final is None:
        save_final = save_final_result_csv
    buffers = load_tracking_buffers(spec['out_dir'], family)
    return save_final(
        spec['session_path'],
        spec['model_name'],
        spec['dataset_name'],
        spec['run_name'],
        spec['video_name'],
        buffers,
        int(spec['num_objects']),
        family,
    )


# Buffer accessors

def get_obb(buffers: dict, tid: int, frame: int) -> np.ndarray | None:
    raw = buffers.get('obb_buf', {}).get(tid, {}).get(frame)
    if raw is None:
        return None
    obb = tuple_to_obb(raw)
    return obb if valid_obb(obb) else None


def _cached_get_obb(
    buffers: dict,
    cache: dict[tuple[int, int], np.ndarray | None],
    tid: int,
    frame: int,
) -> np.ndarray | None:
    key = (int(tid), int(frame))
    if key not in cache:
        cache[key] = get_obb(buffers, *key)
    return cache[key]


def _valid_frames_cached(
    buffers: dict,
    tid: int,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    valid_frames_cache: dict[int, list[int]],
) -> list[int]:
    tid = int(tid)
    if tid not in valid_frames_cache:
        valid_frames_cache[tid] = sorted(
            int(frame)
            for frame in buffers.get('obb_buf', {}).get(tid, {}).keys()
            if _cached_get_obb(buffers, obb_cache, tid, int(frame)) is not None
        )
    return valid_frames_cache[tid]


def get_dir(*args, **kwargs):
    return None


def _direction_on_long_axis(*args, **kwargs):
    return None


def _valid_frames(buffers: dict, tid: int) -> list[int]:
    """Sorted list of frames in buffers where the OBB is valid."""
    return sorted(
        t for t in buffers.get('obb_buf', {}).get(tid, {}).keys()
        if get_obb(buffers, tid, t) is not None
    )


_SWAP_BUFS = (
    'obb_buf', 'class_buf', 'pos_buf', 'score_buf',
    'assign_type_buf', 'assign_val_buf', 'assign_total_buf',
    'assign_iou_cost_buf', 'assign_direction_cost_buf',
    'assign_distance_cost_buf', 'assign_score_cost_buf',
    'assign_type_history_buf',
    'obb_source_buf', 'obb_corrected_buf',
    'direction_corrected_buf', 'switch_corrected_buf',
)


def _write_synth_frame(
    dst: dict,
    tid: int,
    frame: int,
    pts: np.ndarray,
    direction: float | None,
    assign_type: float,
    history_source: str = 'correction_synth',
    max_axis_error_deg: float = 45.0,
    project_direction: bool = True,
) -> None:
    """Write a synthesised (filled or interpolated) OBB frame to dst."""
    pts = ensure_clockwise(pts)
    c = obb_center(pts)
    if project_direction:
        dir_val = _direction_on_long_axis(
            pts, direction, max_axis_error_deg=max_axis_error_deg,
        )
    else:
        try:
            dir_val = float(direction) if direction is not None else np.nan
        except (TypeError, ValueError):
            dir_val = np.nan
        if not np.isfinite(dir_val):
            dir_val = np.nan
    if dir_val is None:
        dir_val = np.nan
    dst.setdefault('pos_buf', {}).setdefault(tid, {})[frame] = (float(c[0]), float(c[1]))
    dst.setdefault('obb_buf', {}).setdefault(tid, {})[frame] = obb_to_tuple(pts)
    dst.setdefault('class_buf', {}).setdefault(tid, {})[frame] = (dir_val,)
    dst.setdefault('score_buf', {}).setdefault(tid, {})[frame] = (np.nan,)
    dst.setdefault('assign_type_buf', {}).setdefault(tid, {})[frame] = (float(assign_type),)
    dst.setdefault('assign_type_history_buf', {}).setdefault(tid, {}).setdefault(frame, []).append((history_source, float(assign_type)))
    for name in (
        'assign_val_buf', 'assign_total_buf', 'assign_iou_cost_buf',
        'assign_direction_cost_buf', 'assign_distance_cost_buf', 'assign_score_cost_buf',
    ):
        dst.setdefault(name, {}).setdefault(tid, {})[frame] = (np.nan,)
    src_val = float(OBB_SOURCE_TRACK_INTERPOLATED)
    dst.setdefault('obb_source_buf', {}).setdefault(tid, {})[frame] = (src_val,)
    dst.setdefault('obb_corrected_buf', {}).setdefault(tid, {})[frame] = (1.0,)
    dst.setdefault('direction_corrected_buf', {}).setdefault(tid, {})[frame] = (1.0,)
    dst.setdefault('switch_corrected_buf', {}).setdefault(tid, {})[frame] = (0.0,)


def _swap_segment(corrected: dict, id_x: int, id_y: int, t_from: int, t_to: int) -> None:
    """Swap all tracking/provenance/history entries between two IDs."""
    id_x = int(id_x)
    id_y = int(id_y)
    t_from = int(t_from)
    t_to = int(t_to)
    if id_x == id_y or t_to < t_from:
        return

    sentinel = object()
    for name in _SWAP_BUFS:
        buf = corrected.setdefault(name, {})
        per_x = buf.setdefault(id_x, {})
        per_y = buf.setdefault(id_y, {})
        for frame in range(t_from, t_to + 1):
            val_x = per_x.get(frame, sentinel)
            val_y = per_y.get(frame, sentinel)
            if val_y is sentinel:
                per_x.pop(frame, None)
            else:
                per_x[frame] = val_y
            if val_x is sentinel:
                per_y.pop(frame, None)
            else:
                per_y[frame] = val_x


# Embedding helpers

def _last_frag_before(fragment_index: dict, obj_id: int, frame: int):
    """Last indexed fragment ending strictly before ``frame``."""
    entry = fragment_index.get(int(obj_id))
    if entry is None:
        return None
    end_frames, by_end, _, _ = entry
    idx = bisect_left(end_frames, int(frame)) - 1
    return by_end[idx] if idx >= 0 else None


def _first_frag_after(fragment_index: dict, obj_id: int, frame: int):
    """First indexed fragment starting strictly after ``frame``."""
    entry = fragment_index.get(int(obj_id))
    if entry is None:
        return None
    _, _, start_frames, by_start = entry
    idx = bisect_right(start_frames, int(frame))
    return by_start[idx] if idx < len(by_start) else None


def _resolve_pre_post_fragments(
    fragment_index: dict, centroids: dict, roster: list, pivot: int,
) -> tuple[dict[int, int], dict[int, int], str | None]:
    """Resolve each roster id's pre/post fragment (and confirm both have a
    trained centroid) around pivot.

    Returns (pre_frags, post_frags, reason). reason is None on success, else
    the specific condition that first failed -- distinguishing a missing
    fragment from a missing centroid, and pre from post, instead of one
    generic 'fragments_incomplete' bucket -- so the log/event record can tell
    which cause is actually driving abstentions:
    'fragments_incomplete_missing_pre_fragment',
    'fragments_incomplete_missing_post_fragment',
    'fragments_incomplete_missing_pre_centroid', or
    'fragments_incomplete_missing_post_centroid'.
    """
    pre_frags: dict[int, int] = {}
    post_frags: dict[int, int] = {}
    for tid in roster:
        pre = _last_frag_before(fragment_index, tid, pivot)
        post = _first_frag_after(fragment_index, tid, pivot)
        if pre is None:
            return {}, {}, 'fragments_incomplete_missing_pre_fragment'
        if post is None:
            return {}, {}, 'fragments_incomplete_missing_post_fragment'
        if pre.frag_id not in centroids:
            return {}, {}, 'fragments_incomplete_missing_pre_centroid'
        if post.frag_id not in centroids:
            return {}, {}, 'fragments_incomplete_missing_post_centroid'
        pre_frags[tid] = int(pre.frag_id)
        post_frags[tid] = int(post.frag_id)
    return pre_frags, post_frags, None


def _build_fragment_index(fragments: list) -> dict[int, tuple[list[int], list, list[int], list]]:
    grouped: dict[int, list[tuple[int, int, int, object]]] = {}
    for order, fragment in enumerate(fragments):
        if not fragment.frames:
            continue
        grouped.setdefault(int(fragment.obj_id), []).append((
            int(min(fragment.frames)),
            int(max(fragment.frames)),
            int(order),
            fragment,
        ))

    index: dict[int, tuple[list[int], list, list[int], list]] = {}
    for obj_id, items in grouped.items():
        by_end_items = sorted(items, key=lambda item: (item[1], -item[2]))
        by_start_items = sorted(items, key=lambda item: (item[0], item[2]))
        index[obj_id] = (
            [item[1] for item in by_end_items],
            [item[3] for item in by_end_items],
            [item[0] for item in by_start_items],
            [item[3] for item in by_start_items],
        )
    return index


# Local sub-episode decomposition helpers

def _collect_trigger_frames(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    tid: int,
    lo: int,
    hi: int,
) -> list[int]:
    """Frames of tid in [lo, hi] that are fill-type atype triggers or missing-gap reentries."""
    triggers: set[int] = set()
    seen_gap = False
    for frame in range(lo, hi + 1):
        present = _cached_get_obb(corrected, obb_cache, tid, frame) is not None
        if present:
            if _frame_has_fill_trigger(corrected, tid, frame):
                triggers.add(frame)
            if seen_gap:
                triggers.add(frame)  # first present frame after a missing run
            seen_gap = False
        else:
            seen_gap = True
    return sorted(triggers)


def _split_episode_by_local_non_s1_events(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    episode: dict,
    max_roster: int = 3,
) -> list[dict]:
    """Decompose a large contact episode around fill/reentry trigger IDs.

    Candidate IDs are ranked solely by their maximum OBB IoU with the trigger ID
    over the complete episode interval.  This intentionally avoids the previous
    multi-term heuristic involving heading, distance, KF confidence, and anchors.
    """
    roster = [int(tid) for tid in episode['roster']]
    start = int(episode['start'])
    end = int(episode['end'])

    _ATYPE_LABEL: dict[int, str] = {
        ATYPE_KF_FILL: 'KF_FILL',
        ATYPE_OVERLAP_FILL: 'OVERLAP_FILL',
        ATYPE_PRE_FILL: 'PRE_FILL',
        ATYPE_POST_FILL: 'POST_FILL',
        ATYPE_FILTER_FILL: 'FILTER_FILL',
    }

    trigger_info: list[tuple[int, list[int], str]] = []
    for tid in roster:
        frames = _collect_trigger_frames(corrected, obb_cache, tid, start, end)
        if not frames:
            continue
        atype_ints = [
            int(round(float(_get_atype(corrected, tid, f))))
            for f in frames
            if np.isfinite(_get_atype(corrected, tid, f))
        ]
        if atype_ints:
            dominant = max(set(atype_ints), key=atype_ints.count)
            label = _ATYPE_LABEL.get(dominant, str(dominant))
        else:
            label = 'MISSING_REENTRY'
        trigger_info.append((tid, frames, label))

    if not trigger_info:
        return []

    result: list[dict] = []
    for trigger_id, trigger_frames, trigger_atype in trigger_info:
        candidate_overlap_scores: dict[int, float] = {}

        for cand_id in roster:
            if cand_id == trigger_id:
                continue
            max_overlap = 0.0
            for f in range(start, end + 1):
                obb_t = _cached_get_obb(corrected, obb_cache, trigger_id, f)
                obb_c = _cached_get_obb(corrected, obb_cache, cand_id, f)
                if obb_t is None or obb_c is None:
                    continue
                if not aabbs_overlap(obb_aabb(obb_t), obb_aabb(obb_c)):
                    continue
                v = float(iou_obb(obb_t, obb_c))
                if np.isfinite(v) and v > max_overlap:
                    max_overlap = v
            if max_overlap > 0.0:
                candidate_overlap_scores[int(cand_id)] = float(max_overlap)

        selection_rule = 'max_pairwise_iou_with_trigger'
        if not candidate_overlap_scores:
            result.append({
                'trigger_id': trigger_id,
                'trigger_frames': trigger_frames,
                'trigger_atype': trigger_atype,
                'candidate_ids': [],
                'sub_roster': [],
                'sub_start': start,
                'sub_end': end,
                'pair_score': 0.0,
                'candidate_overlap_scores': {},
                'selection_rule': selection_rule,
                '_no_candidate': True,
            })
            continue

        sorted_cands = sorted(
            candidate_overlap_scores.items(),
            key=lambda item: (-float(item[1]), int(item[0])),
        )
        selected = [cid for cid, _ in sorted_cands[:max_roster - 1]]
        top_overlap = float(sorted_cands[0][1])

        result.append({
            'trigger_id': trigger_id,
            'trigger_frames': trigger_frames,
            'trigger_atype': trigger_atype,
            'candidate_ids': selected,
            'sub_roster': sorted([trigger_id] + selected),
            'sub_start': start,
            'sub_end': end,
            'pair_score': top_overlap,
            'candidate_overlap_scores': dict(sorted_cands),
            'selection_rule': selection_rule,
            '_no_candidate': False,
        })

    return result


# Correction Stage 3: final identity resolution

def _track_end_for_ids(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    valid_frames_cache: dict[int, list[int]],
    ids: frozenset[int],
) -> int | None:
    track_end: int | None = None
    for tid in ids:
        frames = _valid_frames_cached(
            corrected, int(tid), obb_cache, valid_frames_cache,
        )
        if frames and (track_end is None or frames[-1] > track_end):
            track_end = int(frames[-1])
    return track_end


def _mark_embedding_episode_frames(
    corrected: dict,
    ids: set[int],
    frames: set[int],
) -> None:
    for tid in sorted(ids):
        for frame in sorted(frames):
            if get_obb(corrected, int(tid), int(frame)) is None:
                continue
            corrected.setdefault('assign_type_buf', {}).setdefault(int(tid), {})[int(frame)] = (float(ATYPE_EMBED_RELABEL),)
            corrected.setdefault('switch_corrected_buf', {}).setdefault(int(tid), {})[int(frame)] = (1.0,)
            corrected.setdefault('assign_type_history_buf', {}).setdefault(int(tid), {}).setdefault(int(frame), []).append(
                ('correction_embedding_episode', float(ATYPE_EMBED_RELABEL))
            )


def _apply_assignment_swaps(
    corrected: dict,
    assignment: dict[int, int],
    t_from: int,
    t_to: int,
) -> None:
    visited: set[int] = set()
    for start in sorted(assignment.keys()):
        start = int(start)
        if start in visited:
            continue
        cycle: list[int] = []
        cur = start
        while cur not in cycle:
            cycle.append(cur)
            visited.add(cur)
            cur = int(assignment[cur])
        if cur != start:
            continue
        for other in reversed(cycle[1:]):
            _swap_segment(corrected, start, int(other), int(t_from), int(t_to))


def _relabel_fragments_after_assignment(
    fragments: list | None,
    assignment: dict[int, int],
    t_from: int,
    t_to: int,
) -> None:
    """Keep embedding fragments in the same ID label space as corrected buffers.

    A swap only changes track labels; the crop rows and fragment centroids remain
    valid appearance descriptors.  Fully covered fragments can therefore be
    relabelled in place.  Fragments partially crossing the swap seam are removed
    from later embedding decisions because their centroid would mix two label
    regimes.
    """
    if not fragments:
        return

    affected_ids = {int(k) for k in assignment.keys()}
    t_from = int(t_from)
    t_to = int(t_to)
    kept: list = []

    for frag in fragments:
        obj_id = int(getattr(frag, 'obj_id', -1))
        if obj_id not in affected_ids:
            kept.append(frag)
            continue

        frames = [int(fr) for fr in getattr(frag, 'frames', [])]
        if not frames:
            continue

        overlap = any(t_from <= fr <= t_to for fr in frames)
        if not overlap:
            kept.append(frag)
            continue

        fully_covered = all(t_from <= fr <= t_to for fr in frames)
        if fully_covered:
            frag.obj_id = int(assignment.get(obj_id, obj_id))
            kept.append(frag)
        # Partially covered fragments are intentionally dropped.

    fragments[:] = kept


def _connected_components_from_edges(
    nodes: list[int],
    edges: list[tuple[int, int]],
) -> list[set[int]]:
    """Return edge-connected components, excluding isolated nodes."""
    allowed = {int(node) for node in nodes}
    adjacency: dict[int, set[int]] = {}
    for raw_a, raw_b in edges:
        a = int(raw_a)
        b = int(raw_b)
        if a == b or a not in allowed or b not in allowed:
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    components: list[set[int]] = []
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        component: set[int] = set()
        stack = [seed]
        while stack:
            node = int(stack.pop())
            if node not in unseen:
                continue
            unseen.remove(node)
            component.add(node)
            stack.extend(adjacency.get(node, set()) & unseen)
        components.append(component)

    components.sort(key=lambda component: tuple(sorted(component)))
    return components


def _build_contact_episodes(
    corrected: dict,
    num_objects: int,
    episode_max_len: int,
) -> list[dict]:
    """Build component-aware IoU>0 episodes and bridge interior gaps."""
    valid_frames = {int(tid): [] for tid in range(int(num_objects))}
    observed_frames = sorted({
        int(frame)
        for tid in range(int(num_objects))
        for frame in corrected.get('obb_buf', {}).get(int(tid), {}).keys()
    })
    if not observed_frames:
        return []

    contact_components: dict[int, list[set[int]]] = {}
    contact_members: dict[int, set[int]] = {}
    for frame in tqdm_it(observed_frames, desc='Embedding contact scan', unit='frame'):
        present_obbs: dict[int, np.ndarray] = {}
        for tid in range(int(num_objects)):
            obb = get_obb(corrected, int(tid), int(frame))
            if obb is not None:
                present_obbs[int(tid)] = obb
                valid_frames[int(tid)].append(int(frame))
        present = sorted(present_obbs)
        present_aabbs = {tid: obb_aabb(present_obbs[tid]) for tid in present}
        edges: list[tuple[int, int]] = []
        for ii in range(len(present)):
            for jj in range(ii + 1, len(present)):
                tid_a = int(present[ii])
                tid_b = int(present[jj])
                if not aabbs_overlap(present_aabbs[tid_a], present_aabbs[tid_b]):
                    continue
                iouv = float(iou_obb(present_obbs[tid_a], present_obbs[tid_b]))
                if np.isfinite(iouv) and iouv > 0.0:
                    edges.append((tid_a, tid_b))
        components = _connected_components_from_edges(present, edges)
        if components:
            contact_components[int(frame)] = components
            contact_members[int(frame)] = set().union(*components)

    # A bridge is represented as a separate singleton component.  It can keep
    # an overlapping active episode alive, but can never join two episodes.
    bridged_components: dict[int, list[set[int]]] = {}
    for tid, frames in tqdm_it(valid_frames.items(), total=len(valid_frames), desc='Embedding bridge scan', unit='id'):
        for before, after in zip(frames[:-1], frames[1:]):
            gap_len = int(after) - int(before) - 1
            if gap_len < 1:
                continue
            if (
                int(tid) not in contact_members.get(int(before), set())
                or int(tid) not in contact_members.get(int(after), set())
            ):
                continue
            for frame in range(int(before) + 1, int(after)):
                bridged_components.setdefault(int(frame), []).append({int(tid)})

    episodes: list[dict] = []
    active_episodes: list[dict] = []

    def finish(active: dict) -> None:
        episodes.append({
            'start': int(active['start']),
            'end': int(active['last_active']),
            'roster': sorted(int(tid) for tid in active['roster']),
        })

    event_frames = sorted(set(contact_components) | set(bridged_components))
    for frame in tqdm_it(event_frames, desc='Embedding episode assembly', unit='frame'):
        frame = int(frame)

        consecutive: list[dict] = []
        for active in active_episodes:
            if frame == int(active['last_active']) + 1:
                consecutive.append(active)
            else:
                finish(active)
        active_episodes = consecutive

        current = [set(component) for component in contact_components.get(frame, [])]

        # If one previous episode fans out into multiple independent components,
        # close it before the split.  Reusing it for both branches would recreate
        # a combined roster for components that are disconnected in this frame.
        active_by_tid: dict[int, set[int]] = {}
        for active_idx, active in enumerate(active_episodes):
            for tid in active['roster']:
                active_by_tid.setdefault(int(tid), set()).add(active_idx)

        active_to_current: list[list[int]] = [[] for _ in active_episodes]
        raw_current_to_active: list[list[int]] = []
        for component_idx, component in enumerate(current):
            matches: set[int] = set()
            for tid in component:
                matches.update(active_by_tid.get(int(tid), set()))
            sorted_matches = sorted(matches)
            raw_current_to_active.append(sorted_matches)
            for active_idx in sorted_matches:
                active_to_current[active_idx].append(component_idx)
        split_active = {
            active_idx
            for active_idx, matches in enumerate(active_to_current)
            if len(matches) > 1
        }
        for active_idx in sorted(split_active):
            finish(active_episodes[active_idx])

        current_to_active: list[list[int]] = [
            [
                active_idx
                for active_idx in matches
                if active_idx not in split_active
            ]
            for matches in raw_current_to_active
        ]

        matched_active: set[int] = set()
        next_active: list[dict] = []
        for component, matches in zip(current, current_to_active):
            if matches:
                matched_active.update(matches)
                merged_roster = set(component)
                for active_idx in matches:
                    merged_roster |= set(active_episodes[active_idx]['roster'])
                active = {
                    'start': min(int(active_episodes[idx]['start']) for idx in matches),
                    'last_active': frame,
                    'last_component': set(component),
                    'roster': merged_roster,
                }
            else:
                active = {
                    'start': frame,
                    'last_active': frame,
                    'last_component': set(component),
                    'roster': set(component),
                }
            next_active.append(active)

        # Bridges only extend an unmatched active episode with ID overlap.  Each
        # bridge belongs to at most one episode and never creates a new episode.
        bridge_additions: dict[int, set[int]] = {}
        available_active = [
            idx
            for idx in range(len(active_episodes))
            if idx not in split_active and idx not in matched_active
        ]
        available_active_set = set(available_active)
        for bridge in bridged_components.get(frame, []):
            candidate_set: set[int] = set()
            for tid in bridge:
                candidate_set.update(active_by_tid.get(int(tid), set()))
            candidates = sorted(candidate_set & available_active_set)
            if not candidates:
                continue
            active_idx = max(
                candidates,
                key=lambda idx: (
                    bool(bridge & active_episodes[idx]['last_component']),
                    int(active_episodes[idx]['start']),
                    -idx,
                ),
            )
            bridge_additions.setdefault(active_idx, set()).update(bridge)

        for active_idx in available_active:
            active = active_episodes[active_idx]
            additions = bridge_additions.get(active_idx)
            if additions:
                active['roster'] |= additions
                active['last_active'] = frame
                next_active.append(active)
            else:
                finish(active)

        active_episodes = []
        for active in next_active:
            duration = frame - int(active['start']) + 1
            if int(episode_max_len) > 0 and duration >= int(episode_max_len):
                finish(active)
            else:
                active_episodes.append(active)

    for active in active_episodes:
        finish(active)

    return [episode for episode in episodes if len(episode['roster']) >= 2]


def _is_non_s1_frame(corrected: dict, tid: int, frame: int) -> bool:
    """Return True for present frames whose assign_type is not trusted S1/init."""
    if get_obb(corrected, int(tid), int(frame)) is None:
        return False
    atype = _get_atype(corrected, int(tid), int(frame))
    return bool(
        np.isfinite(atype)
        and int(round(float(atype))) not in REFINEMENT_EXCLUDED_ASSIGN_CODES
    )


def _is_trusted_anchor(corrected: dict, tid: int, frame: int) -> bool:
    """Return True for frames whose matching-confirmed heading must be preserved."""
    if get_obb(corrected, int(tid), int(frame)) is None:
        return False
    atype = _get_atype(corrected, int(tid), int(frame))
    return bool(
        np.isfinite(atype)
        and int(round(float(atype))) in REFINEMENT_EXCLUDED_ASSIGN_CODES
    )


def _has_interior_missing_in_episode(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    valid_frames_cache: dict[int, list[int]],
    tid: int,
    lo: int,
    hi: int,
) -> bool:
    """Return True when tid has at least one interior missing frame within [lo, hi].

    Interior means the frame lies between the track's first and last valid OBB
    frame, so it is genuinely absent rather than before/after the track's life.
    Missing frames represent unresolved identity and should be treated as non-S1
    for the purpose of deciding whether an embedding episode is worth evaluating.
    """
    valid = _valid_frames_cached(
        corrected, int(tid), obb_cache, valid_frames_cache,
    )
    if len(valid) < 2:
        return False
    track_lo, track_hi = valid[0], valid[-1]
    valid_set = set(valid)
    return any(
        frame not in valid_set
        for frame in range(max(int(lo), track_lo), min(int(hi), track_hi) + 1)
    )


def _episode_max_pairwise_iou(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    roster: list[int],
    start: int,
    end: int,
) -> float:
    """Maximum pairwise OBB IoU among episode participants over its interval."""
    ids = sorted({int(tid) for tid in roster})
    max_iou = 0.0
    for frame in range(int(start), int(end) + 1):
        present = {
            tid: _cached_get_obb(corrected, obb_cache, tid, frame)
            for tid in ids
        }
        present_ids = [tid for tid in ids if present[tid] is not None]
        present_aabbs = {tid: obb_aabb(present[tid]) for tid in present_ids}
        for i, tid_a in enumerate(present_ids):
            for tid_b in present_ids[i + 1:]:
                if not aabbs_overlap(present_aabbs[tid_a], present_aabbs[tid_b]):
                    continue
                iouv = float(iou_obb(present[tid_a], present[tid_b]))
                if np.isfinite(iouv) and iouv > max_iou:
                    max_iou = iouv
    return float(max_iou)


def _episode_has_fill_or_missing_trigger(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    valid_frames_cache: dict[int, list[int]],
    roster: list[int],
    start: int,
    end: int,
) -> bool:
    """Return True for an interior gap, fill frame, or in-episode reentry."""
    lo, hi = int(start), int(end)
    for tid_raw in roster:
        tid = int(tid_raw)
        if _has_interior_missing_in_episode(
            corrected, obb_cache, valid_frames_cache, tid, lo, hi,
        ):
            return True

        seen_missing = False
        for frame in range(lo, hi + 1):
            present = _cached_get_obb(corrected, obb_cache, tid, frame) is not None
            if not present:
                seen_missing = True
                continue
            if _frame_has_fill_trigger(corrected, tid, frame):
                return True
            if seen_missing:
                return True
            seen_missing = False
    return False


def _select_embedding_target_episodes(
    corrected: dict,
    episodes: list[dict],
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    valid_frames_cache: dict[int, list[int]],
    min_interact_iou: float,
) -> tuple[list[dict], int]:
    """Keep only contact episodes with high ID-switch risk.

    A target episode is an IoU>0 contact episode that either reaches
    INTERACT_IOU at least once, or contains an interior missing frame /
    post-processing fill history.  Low-overlap episodes without those triggers
    are not evaluated by the embedding resolver.
    """
    threshold = float(min_interact_iou)
    selected: list[dict] = []
    skipped = 0

    for episode in episodes:
        roster = [int(tid) for tid in episode.get('roster', [])]
        start = int(episode.get('start', 0))
        end = int(episode.get('end', start))
        max_iou = _episode_max_pairwise_iou(
            corrected, obb_cache, roster, start, end,
        )
        has_fill_or_missing = _episode_has_fill_or_missing_trigger(
            corrected, obb_cache, valid_frames_cache, roster, start, end,
        )

        risk_reasons: list[str] = []
        if max_iou >= threshold:
            risk_reasons.append('min_interact_iou')
        if has_fill_or_missing:
            risk_reasons.append('fill_or_missing')

        if not risk_reasons:
            skipped += 1
            continue

        enriched = dict(episode)
        enriched['max_pairwise_iou'] = float(max_iou)
        enriched['min_interact_iou'] = float(threshold)
        enriched['has_fill_or_missing_trigger'] = bool(has_fill_or_missing)
        enriched['risk_reasons'] = risk_reasons
        selected.append(enriched)

    return selected, int(skipped)


def _episode_max_iou_frame(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    tid: int,
    roster: list[int],
    lo: int,
    hi: int,
    *,
    frames: list[int] | None = None,
) -> int | None:
    """Frame where tid has maximum IoU with any other roster member.

    If ``frames`` is provided, only those frames are searched instead of range [lo, hi].
    """
    best_frame: int | None = None
    best_iou = -1.0
    frame_iter: range | list[int] = frames if frames is not None else range(int(lo), int(hi) + 1)
    for frame in frame_iter:
        obb_a = _cached_get_obb(corrected, obb_cache, int(tid), int(frame))
        if obb_a is None:
            continue
        aabb_a = obb_aabb(obb_a)
        frame_max = 0.0
        for other in roster:
            if int(other) == int(tid):
                continue
            obb_b = _cached_get_obb(corrected, obb_cache, int(other), int(frame))
            if obb_b is None:
                continue
            if not aabbs_overlap(aabb_a, obb_aabb(obb_b)):
                continue
            iouv = float(iou_obb(obb_a, obb_b))
            if np.isfinite(iouv) and iouv > frame_max:
                frame_max = iouv
        if frame_max > best_iou:
            best_iou = frame_max
            best_frame = int(frame)
    return best_frame


def _localize_swap_seam_for_id(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    tid: int,
    roster: list[int],
    lo: int,
    hi: int,
) -> int:
    """(1) Prefer fill-type (gap-interpolated) frames; (2) pick the max-IoU frame among them."""
    # (1) Fill-type frames (kf_fill, overlap_fill, etc.) are preferred switch candidates
    fill_frames = [
        frame for frame in range(int(lo), int(hi) + 1)
        if _cached_get_obb(corrected, obb_cache, int(tid), frame) is not None
        and np.isfinite(_get_atype(corrected, int(tid), frame))
        and int(round(float(_get_atype(corrected, int(tid), frame)))) in _LOCAL_TRIGGER_ATYPES
    ]

    # (2) Within the candidate set, pick the frame with the highest IoU
    if fill_frames:
        apex = _episode_max_iou_frame(
            corrected, obb_cache, int(tid), roster, int(lo), int(hi), frames=fill_frames,
        )
        if apex is not None:
            return int(apex)

    apex = _episode_max_iou_frame(corrected, obb_cache, int(tid), roster, int(lo), int(hi))
    if apex is not None:
        return int(apex)

    return int((int(lo) + int(hi)) // 2)


def _localize_swap_seam(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    swapped_ids: list[int],
    roster: list[int],
    lo: int,
    hi: int,
) -> int:
    """Return the shared t_from as the earliest per-ID seam candidate."""
    candidates = [
        _localize_swap_seam_for_id(
            corrected, obb_cache, int(tid), roster, int(lo), int(hi),
        )
        for tid in swapped_ids
    ]
    return int(min(candidates)) if candidates else int(lo)


def _seam_source_label(
    corrected: dict,
    obb_cache: dict[tuple[int, int], np.ndarray | None],
    swapped_ids: list[int],
    roster: list[int],
    lo: int,
    hi: int,
    seam: int,
) -> str:
    """Explain which localization rule produced the selected shared seam."""
    for tid in swapped_ids:
        fill_frames = [
            frame for frame in range(int(lo), int(hi) + 1)
            if _cached_get_obb(corrected, obb_cache, int(tid), frame) is not None
            and np.isfinite(_get_atype(corrected, int(tid), frame))
            and int(round(float(_get_atype(corrected, int(tid), frame)))) in _LOCAL_TRIGGER_ATYPES
        ]
        if fill_frames:
            apex = _episode_max_iou_frame(
                corrected, obb_cache, int(tid), roster, int(lo), int(hi), frames=fill_frames,
            )
            if apex == int(seam):
                return 'fill_iou_apex'
    for tid in swapped_ids:
        apex = _episode_max_iou_frame(corrected, obb_cache, int(tid), roster, int(lo), int(hi))
        if apex == int(seam):
            return 'iou_apex'
    return 'midpoint'


# gate_abs_dist/gate_cost_ratio abstentions are the only ones eligible for the
# second-stage rescue in _apply_embedding_rescue -- both stem from an
# untrustworthy local pre/post fragment centroid, which a robust cross-episode
# prototype can address. Other abstain reasons (missing fragments/centroids,
# invalid roster) reflect data that simply isn't available, which no rescue
# can supply.
_RESCUE_ELIGIBLE_REASONS = frozenset({'gate_abs_dist', 'gate_cost_ratio'})

# Default rescue_* log schema for episodes that never reach _apply_embedding_rescue
# (rescue ineligible, or not needed). Keeps every logged episode/subepisode entry
# self-describing with the same key set regardless of whether rescue ran.
_RESCUE_LOG_DEFAULTS = {
    'rescue_attempted': False,
    'rescue_action': None,
    'rescue_failure_stage': None,
    'rescue_assignment': None,
    'rescue_pre_cost_matrix': None,
    'rescue_pre_best_cost': None,
    'rescue_pre_identity_cost': None,
    'rescue_pre_max_matched_dist': None,
    'rescue_pre_reason': None,
    'rescue_post_cost_matrix': None,
    'rescue_post_best_cost': None,
    'rescue_post_identity_cost': None,
    'rescue_post_second_cost': None,
    'rescue_post_max_matched_dist': None,
    'rescue_post_reason': None,
    'rescue_post_assignment': None,
    'rescue_prototype_fragment_counts': None,
    'rescue_prototype_fragment_ids': None,
    'rescue_pre_fragments': None,
    'rescue_post_fragments': None,
    'rescue_one_sided_attempted': False,
    'rescue_one_sided_action': None,
    'rescue_one_sided_anchor_id': None,
    'rescue_one_sided_pre_match': None,
    'rescue_one_sided_post_match': None,
    'rescue_one_sided_pre_adjacent': None,
    'rescue_one_sided_post_adjacent': None,
    'rescue_one_sided_assignment': None,
}


def _str_keyed(mapping: dict | None) -> dict | None:
    """Stringify a dict's keys for JSON-log friendliness (e.g. tid -> ... maps)."""
    if mapping is None:
        return None
    return {str(k): v for k, v in sorted(mapping.items())}


def _apply_embedding_rescue(
    entry: dict,
    fragment_index: dict,
    centroids: dict,
    roster: list[int],
    pre_frags: dict[int, int],
    post_frags: dict[int, int],
    start: int,
    decision,
):
    """Second-stage rescue for an episode abstained by gate_abs_dist/gate_cost_ratio.

    Re-anchors the same Hungarian-and-gate logic assign_episode already uses
    to a robust per-ID global prototype instead of the single local pre/post
    fragment centroid that triggered the abstention (see
    correction_embedding.attempt_embedding_rescue). Records the rescue_*
    diagnostic fields on ``entry`` in place -- including post-side numbers
    even when the rescue stops at pre_check, purely for post-hoc debugging.
    Returns the original (still-abstained) decision unless the rescue
    confidently detects a swap, in which case it returns a replacement
    EpisodeDecision for the caller's normal swap-handling path
    (track_end/seam/apply/relabel) to apply unchanged.
    """
    from without_direction_estimation.identity_correction import EpisodeDecision, attempt_embedding_rescue

    rescue = attempt_embedding_rescue(
        fragment_index, centroids, roster, pre_frags, post_frags, start,
    )
    entry['rescue_attempted'] = rescue.attempted
    entry['rescue_action'] = rescue.action
    entry['rescue_failure_stage'] = rescue.failure_stage
    entry['rescue_assignment'] = _str_keyed(rescue.mapping)

    entry['rescue_pre_cost_matrix'] = rescue.pre_cost_matrix
    entry['rescue_pre_best_cost'] = rescue.pre_best_cost
    entry['rescue_pre_identity_cost'] = rescue.pre_identity_cost
    entry['rescue_pre_max_matched_dist'] = rescue.pre_max_matched_dist
    entry['rescue_pre_reason'] = rescue.pre_reason

    entry['rescue_post_cost_matrix'] = rescue.post_cost_matrix
    entry['rescue_post_best_cost'] = rescue.post_best_cost
    entry['rescue_post_identity_cost'] = rescue.post_identity_cost
    entry['rescue_post_second_cost'] = rescue.post_second_cost
    entry['rescue_post_max_matched_dist'] = rescue.post_max_matched_dist
    entry['rescue_post_reason'] = rescue.post_reason
    entry['rescue_post_assignment'] = _str_keyed(rescue.post_assignment)

    entry['rescue_prototype_fragment_counts'] = _str_keyed(rescue.prototype_fragment_counts)
    entry['rescue_prototype_fragment_ids'] = _str_keyed(rescue.prototype_fragment_ids)
    entry['rescue_pre_fragments'] = _str_keyed(rescue.pre_fragments)
    entry['rescue_post_fragments'] = _str_keyed(rescue.post_fragments)

    if rescue.action != 'relabel':
        return decision
    return EpisodeDecision(
        rescue.mapping, True, False, 'rescue_relabel',
        rescue.post_best_cost, rescue.post_identity_cost, decision.second_cost, decision.max_matched_dist,
    )


def _apply_one_sided_rescue(
    entry: dict,
    fragment_index: dict,
    centroids: dict,
    roster: list[int],
    pre_frags: dict[int, int],
    post_frags: dict[int, int],
    start: int,
    end: int,
    decision,
):
    """Third-stage fallback for roster == 2 episodes the full-roster rescue
    (_apply_embedding_rescue) still could not resolve. Only called when that
    rescue has already run and is still abstained -- see
    correction_embedding.attempt_one_sided_rescue for the one-sided
    prototype-transition detection this applies, including the
    episode-adjacency requirement on the anchor track's pre/post fragments.
    Records the rescue_one_sided_* fields on ``entry`` in place. Returns the
    original (still-abstained) decision unless exactly one track confirms an
    uncontradicted, episode-adjacent transition, in which case it returns a
    replacement EpisodeDecision for the caller's normal swap-handling path
    (track_end/seam/apply/relabel) to apply unchanged.
    """
    from without_direction_estimation.identity_correction import EpisodeDecision, attempt_one_sided_rescue

    result = attempt_one_sided_rescue(
        fragment_index, centroids, roster, pre_frags, post_frags, start, end,
    )
    entry['rescue_one_sided_attempted'] = result.attempted
    entry['rescue_one_sided_action'] = result.action
    entry['rescue_one_sided_anchor_id'] = result.anchor_id
    entry['rescue_one_sided_pre_match'] = _str_keyed(result.pre_match)
    entry['rescue_one_sided_post_match'] = _str_keyed(result.post_match)
    entry['rescue_one_sided_pre_adjacent'] = _str_keyed(result.pre_adjacent)
    entry['rescue_one_sided_post_adjacent'] = _str_keyed(result.post_adjacent)
    entry['rescue_one_sided_assignment'] = _str_keyed(result.mapping)

    if result.action != 'relabel':
        return decision
    return EpisodeDecision(
        result.mapping, True, False, 'rescue_one_sided_relabel',
        0.0, 0.0, decision.second_cost, decision.max_matched_dist,
    )


def _resolve_contact_episodes_embedding(
    corrected: dict,
    num_objects: int,
    fragments: list,
    centroids: dict,
    episode_max_len: int,
    corrections_log: list[dict],
    min_interact_iou: float,
    progress_label: str = '',
) -> list[dict]:
    """Resolve high-risk N-body contact episodes with contrastive embedding assignments.

    Candidate episodes are limited to IoU>0 contact episodes that either reach
    INTERACT_IOU at least once or contain an interior missing/fill trigger.
    Episodes with roster <= EMBED_MAX_ROSTER use the standard path unchanged
    (INVARIANT: existing swap decisions are not affected by the oversized path).
    Episodes with roster > EMBED_MAX_ROSTER are locally decomposed into
    2..EMBED_MAX_ROSTER sub-rosters seeded by KF/overlap fill trigger events.
    """
    # Import the heavy embedding stack only on the enabled execution path.
    from without_direction_estimation.identity_correction import (
        assign_episode,
        _ASSIGN_MAX_ROSTER as embed_max_roster,
    )

    log: list[dict] = []
    progress_label = str(progress_label or '')

    _ = corrections_log  # retained in the public workflow signature

    fragment_index = _build_fragment_index(fragments)
    obb_cache: dict[tuple[int, int], np.ndarray | None] = {}
    valid_frames_cache: dict[int, list[int]] = {}

    print(f'  Building embedding contact episodes...{progress_label}', flush=True)
    all_episodes = _build_contact_episodes(corrected, num_objects, episode_max_len)
    all_episodes.sort(key=lambda e: (int(e['start']), list(e['roster'])))
    episodes, skipped_low_risk = _select_embedding_target_episodes(
        corrected, all_episodes, obb_cache, valid_frames_cache, min_interact_iou,
    )
    print(
        f'  Built embedding contact episodes: total={len(all_episodes)}, '
        f'target={len(episodes)}, skipped_low_risk={skipped_low_risk}, '
        f'min_interact_iou={float(min_interact_iou):.3f}{progress_label}',
        flush=True,
    )
    resolved_regions: set[tuple] = set()

    _MAX_EPISODE_REBUILDS = 20
    rebuild_count = 0
    _ep_idx = 0
    while _ep_idx < len(episodes):
        episode = episodes[_ep_idx]
        _ep_idx += 1
        if _ep_idx == 1 or _ep_idx % 100 == 0 or _ep_idx == len(episodes):
            print(
                f'  Resolving embedding episodes: {_ep_idx}/{len(episodes)} '
                f'rebuild={rebuild_count}{progress_label}',
                flush=True,
            )
        roster = [int(tid) for tid in episode['roster']]
        start = int(episode['start'])
        end = int(episode['end'])

        region_key = (frozenset(roster), start, end)
        if region_key in resolved_regions:
            log.append({
                'type': 'embedding_episode',
                'ids': sorted(roster),
                'frame_start': start,
                'frame_end': end,
                'action': 'abstain',
                'reason': 'region_already_resolved',
            })
            continue

        episode_max_iou = float(episode.get('max_pairwise_iou', 0.0))
        episode_risk_reasons = list(episode.get('risk_reasons', []))
        episode_has_fill_or_missing = bool(episode.get('has_fill_or_missing_trigger', False))

        # Oversized roster: local sub-episode decomposition
        # INVARIANT: only entered when len(roster) > EMBED_MAX_ROSTER.
        # Episodes with len(roster) <= EMBED_MAX_ROSTER pass through the
        # standard path below without any modification.
        if len(roster) > embed_max_roster:
            parent_ids = sorted(roster)

            subs = _split_episode_by_local_non_s1_events(
                corrected, obb_cache, episode, embed_max_roster,
            )

            if not subs:
                log.append({
                    'type': 'embedding_episode',
                    'ids': parent_ids,
                    'frame_start': start,
                    'frame_end': end,
                    'action': 'abstain',
                    'reason': 'no_trigger_found',
                })
                continue

            viable = [s for s in subs if s['sub_roster'] and len(s['sub_roster']) >= 2]
            if not viable:
                log.append({
                    'type': 'embedding_episode',
                    'ids': parent_ids,
                    'frame_start': start,
                    'frame_end': end,
                    'action': 'abstain',
                    'reason': 'no_candidate_above_threshold',
                })
                continue

            # Deterministic processing order: (sub_start, sub_roster) ascending
            viable.sort(key=lambda s: (int(s['sub_start']), s['sub_roster']))
            sub_resolved: set[tuple] = set()

            for sub in viable:
                sub_roster = sub['sub_roster']
                sub_start = int(sub['sub_start'])
                sub_end = int(sub['sub_end'])
                trigger_id = int(sub['trigger_id'])

                sub_key = (frozenset(sub_roster), sub_start, sub_end)
                sub_entry: dict = {
                    'type': 'embedding_subepisode',
                    'parent_ids': parent_ids,
                    'parent_start': start,
                    'parent_end': end,
                    'trigger_id': trigger_id,
                    'trigger_frames': sub['trigger_frames'],
                    'trigger_atype': sub['trigger_atype'],
                    'candidate_ids': sub['candidate_ids'],
                    'sub_roster': sub_roster,
                    'sub_start': sub_start,
                    'sub_end': sub_end,
                    'pair_score': float(sub['pair_score']),
                    'candidate_overlap_scores': sub['candidate_overlap_scores'],
                    'selection_rule': sub['selection_rule'],
                    'episode_max_iou': float(episode_max_iou),
                    'min_interact_iou': float(min_interact_iou),
                    'risk_reasons': episode_risk_reasons,
                    'has_fill_or_missing_trigger': episode_has_fill_or_missing,
                    'action': 'abstain',
                    'reason': None,
                    'seam': None,
                    'assignment': None,
                    **_RESCUE_LOG_DEFAULTS,
                }

                if sub_key in sub_resolved or sub_key in resolved_regions:
                    sub_entry['reason'] = 'region_already_resolved'
                    log.append(sub_entry)
                    continue

                if len(sub_roster) > embed_max_roster:
                    sub_entry['reason'] = 'subroster_still_oversized'
                    log.append(sub_entry)
                    continue

                # Build pre_frags / post_frags for the sub-roster.
                # Pivot = sub_start (parent episode start) mirrors the standard path.
                # Both key sets must exactly match sub_roster for assign_episode.
                pre_frags, post_frags, frag_reason = _resolve_pre_post_fragments(
                    fragment_index, centroids, sub_roster, sub_start,
                )
                if frag_reason is not None:
                    sub_entry['reason'] = frag_reason
                    log.append(sub_entry)
                    continue

                decision = assign_episode(centroids, pre_frags, post_frags)
                sub_entry['best_cost'] = decision.best_cost
                sub_entry['identity_cost'] = decision.identity_cost
                sub_entry['second_cost'] = decision.second_cost
                sub_entry['max_matched_dist'] = decision.max_matched_dist

                if decision.abstained and decision.reason in _RESCUE_ELIGIBLE_REASONS:
                    decision = _apply_embedding_rescue(
                        sub_entry, fragment_index, centroids, sub_roster, pre_frags, post_frags,
                        sub_start, decision,
                    )
                    if decision.abstained and len(sub_roster) == 2:
                        decision = _apply_one_sided_rescue(
                            sub_entry, fragment_index, centroids, sub_roster, pre_frags, post_frags,
                            sub_start, sub_end, decision,
                        )

                if decision.abstained:
                    sub_entry['reason'] = decision.reason
                    log.append(sub_entry)
                    continue

                if not decision.is_swap:
                    # Embedding assignment ran to completion and positively
                    # confirmed the current identity mapping -- distinct from
                    # the abstain cases above, which never reached a reliable
                    # decision at all.
                    sub_entry['action'] = 'identity'
                    sub_entry['reason'] = decision.reason
                    log.append(sub_entry)
                    continue

                track_end = _track_end_for_ids(
                    corrected, obb_cache, valid_frames_cache, frozenset(sub_roster),
                )
                if track_end is None:
                    sub_entry['reason'] = 'missing_track_end'
                    log.append(sub_entry)
                    continue

                swapped_ids = [int(k) for k, v in decision.mapping.items() if int(k) != int(v)]
                if not swapped_ids:
                    sub_entry['reason'] = 'identity'
                    log.append(sub_entry)
                    continue

                # Seam localization uses sub_roster (not parent roster)
                seam = _localize_swap_seam(
                    corrected, obb_cache, swapped_ids, sub_roster, sub_start, sub_end,
                )
                if int(track_end) < int(seam):
                    sub_entry['reason'] = 'empty_apply_segment'
                    log.append(sub_entry)
                    continue

                seam_source = _seam_source_label(
                    corrected, obb_cache, swapped_ids, sub_roster, sub_start, sub_end, seam,
                )
                _apply_assignment_swaps(corrected, decision.mapping, int(seam), int(track_end))
                obb_cache.clear()
                valid_frames_cache.clear()
                # Rebuild immediately so the next subepisode sees relabelled fragments.
                _relabel_fragments_after_assignment(
                    fragments, decision.mapping, int(seam), int(track_end),
                )
                fragment_index = _build_fragment_index(fragments)

                review_frames = {int(seam)}
                _mark_embedding_episode_frames(corrected, set(swapped_ids), review_frames)
                sub_resolved.add(sub_key)
                resolved_regions.add(sub_key)

                sub_entry['action'] = 'relabel'
                sub_entry['reason'] = 'local_kf_trigger_pair'
                sub_entry['seam'] = int(seam)
                sub_entry['seam_source'] = seam_source
                sub_entry['track_end'] = int(track_end)
                sub_entry['source_episode_start'] = start
                sub_entry['source_episode_end'] = end
                sub_entry['assignment'] = {
                    str(k): int(v) for k, v in sorted(decision.mapping.items())
                }
                sub_entry['swapped_ids'] = sorted(int(tid) for tid in swapped_ids)
                sub_entry['review_frames'] = sorted(int(f) for f in review_frames)
                log.append(sub_entry)
                if rebuild_count < _MAX_EPISODE_REBUILDS:
                    rebuild_count += 1
                    print(
                        f'  Rebuilding embedding contact episodes: '
                        f'{rebuild_count}/{_MAX_EPISODE_REBUILDS}{progress_label}',
                        flush=True,
                    )
                    all_episodes = _build_contact_episodes(corrected, num_objects, episode_max_len)
                    all_episodes.sort(key=lambda e: (int(e['start']), list(e['roster'])))
                    episodes, skipped_low_risk = _select_embedding_target_episodes(
                        corrected, all_episodes, obb_cache, valid_frames_cache, min_interact_iou,
                    )
                    print(
                        f'  Rebuilt embedding contact episodes: total={len(all_episodes)}, '
                        f'target={len(episodes)}, skipped_low_risk={skipped_low_risk}{progress_label}',
                        flush=True,
                    )
                    log.append({
                        'type': 'episode_rebuild',
                        'rebuild_count': rebuild_count,
                        'trigger_seam': int(seam),
                        'trigger_track_end': int(track_end),
                        'prev_episode_start': start,
                        'prev_episode_end': end,
                        'new_episode_count': len(episodes),
                        'new_total_episode_count': len(all_episodes),
                        'skipped_low_risk': int(skipped_low_risk),
                        'min_interact_iou': float(min_interact_iou),
                    })
                    _ep_idx = 0
                    break  # exit sub-loop; outer while restarts from episode 0

            continue  # done with this oversized episode

        # Standard path for roster <= EMBED_MAX_ROSTER (unchanged)
        entry: dict = {
            'type': 'embedding_episode',
            'ids': sorted(roster),
            'frame_start': start,
            'frame_end': end,
            'episode_max_iou': float(episode_max_iou),
            'min_interact_iou': float(min_interact_iou),
            'risk_reasons': episode_risk_reasons,
            'has_fill_or_missing_trigger': episode_has_fill_or_missing,
            'action': 'abstain',
            'reason': None,
            **_RESCUE_LOG_DEFAULTS,
        }

        pre_frags, post_frags, frag_reason = _resolve_pre_post_fragments(
            fragment_index, centroids, roster, start,
        )
        if frag_reason is not None:
            entry['reason'] = frag_reason
            log.append(entry)
            continue

        decision = assign_episode(centroids, pre_frags, post_frags)
        entry['best_cost'] = decision.best_cost
        entry['identity_cost'] = decision.identity_cost
        entry['second_cost'] = decision.second_cost
        entry['max_matched_dist'] = decision.max_matched_dist

        if decision.abstained and decision.reason in _RESCUE_ELIGIBLE_REASONS:
            decision = _apply_embedding_rescue(
                entry, fragment_index, centroids, roster, pre_frags, post_frags,
                start, decision,
            )
            if decision.abstained and len(roster) == 2:
                decision = _apply_one_sided_rescue(
                    entry, fragment_index, centroids, roster, pre_frags, post_frags,
                    start, end, decision,
                )

        if decision.abstained:
            entry['reason'] = decision.reason
            log.append(entry)
            continue

        if not decision.is_swap:
            # Embedding assignment ran to completion and positively confirmed
            # the current identity mapping -- distinct from the abstain cases
            # above, which never reached a reliable decision at all.
            entry['action'] = 'identity'
            entry['reason'] = decision.reason
            log.append(entry)
            continue

        track_end = _track_end_for_ids(
            corrected, obb_cache, valid_frames_cache, frozenset(roster),
        )
        if track_end is None:
            entry['reason'] = 'missing_track_end'
            log.append(entry)
            continue

        swapped_ids = [int(k) for k, v in decision.mapping.items() if int(k) != int(v)]
        if not swapped_ids:
            entry['reason'] = 'identity'
            log.append(entry)
            continue

        seam = _localize_swap_seam(
            corrected, obb_cache, swapped_ids, roster, start, end,
        )
        if int(track_end) < int(seam):
            entry['reason'] = 'empty_apply_segment'
            log.append(entry)
            continue

        seam_source = _seam_source_label(
            corrected, obb_cache, swapped_ids, roster, start, end, seam,
        )
        _apply_assignment_swaps(corrected, decision.mapping, int(seam), int(track_end))
        obb_cache.clear()
        valid_frames_cache.clear()
        _relabel_fragments_after_assignment(fragments, decision.mapping, int(seam), int(track_end))
        fragment_index = _build_fragment_index(fragments)

        review_frames = {int(seam)}
        _mark_embedding_episode_frames(corrected, set(swapped_ids), review_frames)
        resolved_regions.add(region_key)

        entry['action'] = 'relabel'
        entry['reason'] = 'swap'
        entry['seam'] = int(seam)
        entry['seam_source'] = seam_source
        entry['track_end'] = int(track_end)
        entry['source_episode_start'] = start
        entry['source_episode_end'] = end
        entry['assignment'] = {str(k): int(v) for k, v in sorted(decision.mapping.items())}
        entry['swapped_ids'] = sorted(int(tid) for tid in swapped_ids)
        entry['review_frames'] = sorted(int(frame) for frame in review_frames)
        log.append(entry)
        if rebuild_count < _MAX_EPISODE_REBUILDS:
            rebuild_count += 1
            print(
                f'  Rebuilding embedding contact episodes: '
                f'{rebuild_count}/{_MAX_EPISODE_REBUILDS}{progress_label}',
                flush=True,
            )
            all_episodes = _build_contact_episodes(corrected, num_objects, episode_max_len)
            all_episodes.sort(key=lambda e: (int(e['start']), list(e['roster'])))
            episodes, skipped_low_risk = _select_embedding_target_episodes(
                corrected, all_episodes, obb_cache, valid_frames_cache, min_interact_iou,
            )
            print(
                f'  Rebuilt embedding contact episodes: total={len(all_episodes)}, '
                f'target={len(episodes)}, skipped_low_risk={skipped_low_risk}{progress_label}',
                flush=True,
            )
            log.append({
                'type': 'episode_rebuild',
                'rebuild_count': rebuild_count,
                'trigger_seam': int(seam),
                'trigger_track_end': int(track_end),
                'prev_episode_start': start,
                'prev_episode_end': end,
                'new_episode_count': len(episodes),
                'new_total_episode_count': len(all_episodes),
                'skipped_low_risk': int(skipped_low_risk),
                'min_interact_iou': float(min_interact_iou),
            })
            _ep_idx = 0

    # Abstained/identity-confirmed episodes do not overwrite the frame's
    # assignment type, but they are still correction decisions. Persist one
    # review-frame history entry so gui_refinement shows the same outcome in
    # Candidate and History. action=='identity' (embedding ran to completion
    # and positively confirmed no swap) and action=='abstain' (no reliable
    # decision reached at all) are recorded with different codes so they stay
    # distinguishable downstream -- only action=='relabel' (handled earlier,
    # via _mark_embedding_episode_frames) actually changes assign_type_buf.
    for entry in log:
        action = entry.get('action')
        if action == 'identity':
            marker_code = ATYPE_EMBED_IDENTITY
            history_source = 'correction_embedding_identity'
        elif action == 'abstain':
            marker_code = ATYPE_EMBED_ABSTAIN
            history_source = 'correction_embedding_abstain'
        else:
            continue
        if entry.get('type') == 'embedding_subepisode':
            episode_ids = entry['sub_roster']
            episode_start = int(entry['sub_start'])
        elif entry.get('type') == 'embedding_episode':
            episode_ids = entry['ids']
            episode_start = int(entry['frame_start'])
        else:
            continue
        review_frames = entry.get('review_frames')
        if not isinstance(review_frames, list):
            review_frames = [episode_start]
            entry['review_frames'] = review_frames
        for tid in episode_ids:
            for frame in review_frames:
                tid_i = int(tid)
                frame_i = int(frame)
                if get_obb(corrected, tid_i, frame_i) is None:
                    continue
                history = (
                    corrected.setdefault('assign_type_history_buf', {})
                    .setdefault(tid_i, {})
                    .setdefault(frame_i, [])
                )
                marker = (history_source, float(marker_code))
                if marker not in history:
                    history.append(marker)

    return log


# Parallel ID evaluation helpers


def _nested_size_bytes(value) -> int:
    """Estimate the resident size of the simple nested tracking containers."""
    size = int(sys.getsizeof(value))
    if isinstance(value, dict):
        return size + sum(
            _nested_size_bytes(key) + _nested_size_bytes(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return size + sum(_nested_size_bytes(item) for item in value)
    if isinstance(value, np.ndarray) and not value.flags.owndata:
        return size + int(value.nbytes)
    return size


def _id_buffer_view(
    corrected: dict,
    tid: int,
    buffer_names: tuple[str, ...],
) -> dict:
    """Return a shallow view containing only one ID's required buffers."""
    tid = int(tid)
    return {
        name: {tid: corrected.get(name, {}).get(tid, {})}
        for name in buffer_names
    }


def _id_payload_sizes(
    corrected: dict,
    n_ids: int,
    buffer_names: tuple[str, ...],
    *,
    sample_ids: int = _PAYLOAD_SAMPLE_IDS,
) -> list[int]:
    """Per-ID spawn payload sizes for _memory_safe_process_workers.

    Sizing every ID with _nested_size_bytes walks every frame of every selected
    buffer and recurses into each stored tuple.  On a long multi-hundred-ID
    recording that is tens of seconds of pure overhead, paid before any real
    work begins and repeated at every stage and pass -- enough to erase the
    parallel speed-up it exists to make safe.

    Only the largest payloads actually matter: _memory_safe_process_workers
    sums the top `workers` entries and ignores the rest.  So the `sample_ids`
    IDs with the most stored frames are measured exactly, and the remainder are
    extrapolated from the worst bytes-per-entry ratio in that sample.

    Extrapolated values are additionally clamped to the smallest measured
    sample size.  A dict's own overhead per entry is not constant -- it steps
    with the table's growth -- so a purely linear fit off the largest tracks can
    fall a fraction of a percent short on smaller ones.  The clamp makes that
    irrelevant by construction: no estimate can ever outrank a measured value,
    so as long as `sample_ids` is at least the worker count (see the caller in
    _run_id_eval), every payload the cap actually sums is an exact measurement
    and the estimates only order the tail.
    """
    n_ids = max(0, int(n_ids))
    if n_ids <= 0:
        return []

    counts = [
        sum(len(corrected.get(name, {}).get(tid, ())) for name in buffer_names)
        for tid in range(n_ids)
    ]
    sample = sorted(
        range(n_ids),
        key=lambda tid: counts[tid],
        reverse=True,
    )[:max(1, int(sample_ids))]
    measured = {
        tid: _nested_size_bytes(_id_buffer_view(corrected, tid, buffer_names))
        for tid in sample
    }
    if len(measured) >= n_ids:
        return [measured[tid] for tid in range(n_ids)]

    # An ID absent from every buffer yields the view's fixed container cost,
    # which is the intercept the per-entry rate is measured against.
    empty_bytes = _nested_size_bytes(_id_buffer_view(corrected, -1, buffer_names))
    rate = 0.0
    for tid, size in measured.items():
        if counts[tid] > 0:
            rate = max(rate, (size - empty_bytes) / counts[tid])
    sample_floor = min(measured.values())

    return [
        measured[tid] if tid in measured
        else min(int(empty_bytes + rate * counts[tid]), sample_floor)
        for tid in range(n_ids)
    ]


def _memory_safe_process_workers(
    requested: int,
    payload_sizes: list[int],
    *,
    concurrent_jobs: int = 1,
) -> int:
    """Cap spawned workers using actual task data and currently available RAM."""
    requested = min(max(1, int(requested)), max(1, len(payload_sizes)))
    ram_share = int(psutil.virtual_memory().available) // max(1, int(concurrent_jobs))
    budget = int(ram_share * _CORRECTION_RAM_BUDGET_FRACTION)
    largest = sorted((max(0, int(size)) for size in payload_sizes), reverse=True)

    for workers in range(requested, 1, -1):
        live_payload = sum(largest[:workers])
        required = (
            workers * _CORRECTION_PROCESS_BASE_BYTES
            + int(live_payload * _CORRECTION_PAYLOAD_HEADROOM)
        )
        if required <= budget:
            return workers
    return 1


def _run_id_eval_process_task(
    eval_fn,
    buffers: dict,
    tid: int,
    worker_args: tuple,
):
    """Evaluate one ID from a spawn-safe, ID-local buffer view."""
    return eval_fn(buffers, int(tid), *worker_args)


def _run_id_eval(
    corrected: dict,
    num_objects: int,
    num_workers: int,
    *,
    desc: str,
    serial_fn,
    worker_args: tuple = (),
    buffer_names: tuple[str, ...] = (),
    shared_read: bool = False,
    concurrent_jobs: int = 1,
) -> list:
    """Evaluate independent ID jobs in parallel while preserving ID order.

    ID-local work uses processes, but each task receives only the selected
    buffers for its own ID.  Cross-ID work uses threads so the read-only
    tracking snapshot remains shared instead of being copied by Windows spawn.
    """
    n_ids = max(0, int(num_objects))
    if n_ids <= 0:
        return []

    requested = min(max(1, int(num_workers)), n_ids)
    num_w = requested

    if not shared_read and not buffer_names:
        raise ValueError('buffer_names are required for ID-local evaluation')

    if not shared_read and num_w > 1:
        # The sample must cover at least as many IDs as workers, so that every
        # payload _memory_safe_process_workers sums is an exact measurement.
        payload_sizes = _id_payload_sizes(
            corrected,
            n_ids,
            buffer_names,
            sample_ids=max(_PAYLOAD_SAMPLE_IDS, requested),
        )
        num_w = _memory_safe_process_workers(
            requested,
            payload_sizes,
            concurrent_jobs=concurrent_jobs,
        )
        if num_w < requested:
            largest_mib = max(payload_sizes, default=0) / (1024.0 ** 2)
            print(
                f'{desc}: workers reduced {requested}->{num_w} for available RAM '
                f'(largest ID payload={largest_mib:.1f} MiB, '
                f'concurrent jobs={max(1, int(concurrent_jobs))}).',
                flush=True,
            )

    if num_w <= 1:
        return [
            serial_fn(
                corrected if shared_read else _id_buffer_view(corrected, tid, buffer_names),
                tid,
                *worker_args,
            )
            for tid in tqdm_it(range(n_ids), desc=desc, unit='id')
        ]

    results = [None] * n_ids
    if shared_read:
        executor = ThreadPoolExecutor(max_workers=num_w)
        submit = lambda ex, tid: ex.submit(serial_fn, corrected, tid, *worker_args)
    else:
        executor = ProcessPoolExecutor(max_workers=num_w)
        submit = lambda ex, tid: ex.submit(
            _run_id_eval_process_task,
            serial_fn,
            _id_buffer_view(corrected, tid, buffer_names),
            tid,
            worker_args,
        )

    with executor as ex, tqdm_it(total=n_ids, desc=desc, unit='id') as progress:
        futures = {}
        next_tid = 0
        while next_tid < min(num_w, n_ids):
            futures[submit(ex, next_tid)] = next_tid
            next_tid += 1

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                tid = futures.pop(fut)
                results[tid] = fut.result()
                progress.update(1)
                if next_tid < n_ids:
                    futures[submit(ex, next_tid)] = next_tid
                    next_tid += 1
    return results


def _eval_kalman_gap_fill_id(
    corrected: dict,
    tid: int,
    strong_iou: float,
    strong_dir_deg: float,
    max_age: int,
    max_axis_error_deg: float,
) -> tuple[list[tuple], list[dict]]:
    """Return (fills, log) for one ID without writing to corrected."""
    fills: list[tuple] = []   # (frame, pts, direction)
    local_log: list[dict] = []

    valid_frames = [
        t for t in sorted(corrected.get('obb_buf', {}).get(tid, {}).keys())
        if get_obb(corrected, tid, t) is not None
    ]
    if len(valid_frames) < 2:
        return fills, local_log

    f0, f1 = valid_frames[0], valid_frames[-1]
    valid_set = set(valid_frames)

    if not any(t not in valid_set for t in range(f0 + 1, f1)):
        return fills, local_log  # no internal gap

    obb_f0 = get_obb(corrected, tid, f0)
    center_f0 = obb_center(obb_f0).astype(float)
    rect_f0 = obb_points_to_rect(obb_f0)

    pos_kf = create_position_kf(float(center_f0[0]), float(center_f0[1]))
    shape_kf = create_bbox_shape_kf(float(rect_f0[2]), float(rect_f0[3]))

    last_valid_t = f0
    last_valid_obb_v = obb_f0
    in_gap = False

    for t in range(f0 + 1, f1 + 1):
        pos_kf.predict()
        shape_kf.predict()
        # No angular state prediction.

        is_valid = t in valid_set

        if is_valid:
            obb_t = get_obb(corrected, tid, t)

            if in_gap:
                t_a = last_valid_t
                t_b = t
                gap_len = t_b - t_a - 1  # >= 1

                obb_a = last_valid_obb_v
                obb_b = obb_t

                # KF-predicted OBB at t_b (after gap_len predict steps)
                pred_obb_tb = predict_obb_from_filters(last_valid_obb_v, pos_kf, shape_kf)

                # G1: gap length strictly less than max_age
                g1 = gap_len < max_age

                # G2: neither anchor is risky or synthetic
                atype_a = _get_atype(corrected, tid, t_a)
                atype_b = _get_atype(corrected, tid, t_b)
                g2 = (
                    not (np.isfinite(atype_a) and int(atype_a) in _RISKY_OR_SYNTH_ATYPES)
                    and not (np.isfinite(atype_b) and int(atype_b) in _RISKY_OR_SYNTH_ATYPES)
                )

                if g1 and g2:
                    # G3: try prediction branch first, then prev-OBB branch
                    obb_b_aabb = obb_aabb(obb_b)
                    iou_pred = 0.0
                    if valid_obb(pred_obb_tb) and aabbs_overlap(obb_b_aabb, obb_aabb(pred_obb_tb)):
                        iou_pred = iou_obb(obb_b, pred_obb_tb)
                    iou_prev = 0.0
                    if aabbs_overlap(obb_b_aabb, obb_aabb(obb_a)):
                        iou_prev = iou_obb(obb_b, obb_a)

                    if iou_pred >= strong_iou:
                        branch = 'prediction'
                        g3_iou = iou_pred
                    elif iou_prev >= strong_iou:
                        branch = 'prev'
                        g3_iou = iou_prev
                    else:
                        branch = None
                        g3_iou = max(iou_pred, iou_prev)

                    if branch is not None:
                        a_pts = ensure_clockwise(obb_a)
                        b_pts = canonicalize_obb_points(obb_b, reference=a_pts)
                        for t_fill in range(t_a + 1, t_b):
                            alpha = (t_fill - t_a) / float(t_b - t_a)
                            pts_fill = ensure_clockwise(
                                ((1.0 - alpha) * a_pts.astype(float)
                                 + alpha * b_pts.astype(float)).astype(np.float32)
                            )
                            fills.append((t_fill, pts_fill, None))
                        local_log.append({
                            'type': 'kf_fill', 'action': 'fill', 'ids': [tid],
                            'frame_start': t_a + 1, 'frame_end': t_b - 1,
                            'frames': list(range(t_a + 1, t_b)),
                            'gap_len': gap_len, 'branch': branch, 'iou': g3_iou,
                        })
                    else:
                        local_log.append({
                            'type': 'kf_fill',
                            'action': 'reject_g3',
                            'ids': [tid],
                            'frame_start': t_a + 1,
                            'frame_end': t_b - 1,
                            'gap_len': gap_len,
                            'iou_pred': iou_pred,
                            'iou_prev': iou_prev,
                        })
                else:
                    local_log.append({
                        'type': 'kf_fill',
                        'action': 'reject_g1' if not g1 else 'reject_g2',
                        'ids': [tid],
                        'frame_start': t_a + 1,
                        'frame_end': t_b - 1,
                        'gap_len': gap_len,
                    })

                in_gap = False

            # Update KFs with this valid frame's measurement
            center_t = obb_center(obb_t).astype(float)
            rect_t = obb_points_to_rect(obb_t)
            pos_kf.update(center_t)
            shape_kf.update(np.array([float(rect_t[2]), float(rect_t[3])], dtype=float))
            # No angular state update.

            last_valid_t = t
            last_valid_obb_v = obb_t
        else:
            in_gap = True

    return fills, local_log


def _eval_overlap_gap_fill_id(
    corrected: dict,
    tid_a: int,
    num_objects: int,
    max_gap: int,
    min_interact_iou: float,
    max_axis_error_deg: float,
) -> tuple[list[tuple[int, int, np.ndarray, float | None, bool]], list[dict]]:
    local_fills: list[tuple[int, int, np.ndarray, float | None, bool]] = []
    local_log: list[dict] = []
    min_interact_iou = float(min_interact_iou)

    valid_a = _valid_frames(corrected, tid_a)
    if len(valid_a) < 2:
        return local_fills, local_log

    valid_set_a = set(valid_a)

    # Collect interior missing segments (strictly between first and last valid frame).
    segments: list[tuple[int, int]] = []
    seg_start: int | None = None
    for t in range(valid_a[0] + 1, valid_a[-1]):
        if t not in valid_set_a:
            if seg_start is None:
                seg_start = t
        elif seg_start is not None:
            segments.append((seg_start, t - 1))
            seg_start = None
    if seg_start is not None:
        segments.append((seg_start, valid_a[-1] - 1))

    if not segments:
        return local_fills, local_log

    for s_start, s_end in segments:
        if s_end - s_start + 1 >= max_gap:
            continue

        t_before = max(f for f in valid_set_a if f < s_start)
        t_after = min(f for f in valid_set_a if f > s_end)

        obb_a_before = get_obb(corrected, tid_a, t_before)
        obb_a_after = get_obb(corrected, tid_a, t_after)

        if not valid_obb(obb_a_before) or not valid_obb(obb_a_after):
            continue
        aabb_a_before = obb_aabb(obb_a_before)
        aabb_a_after = obb_aabb(obb_a_after)

        best_tid_b: int | None = None
        best_score: float = 0.0
        best_iou_before: float = 0.0
        best_iou_after: float = 0.0

        for tid_b in range(num_objects):
            if tid_b == tid_a:
                continue

            obb_b_before = get_obb(corrected, tid_b, t_before)
            obb_b_after = get_obb(corrected, tid_b, t_after)

            if obb_b_before is None or obb_b_after is None:
                continue
            if not aabbs_overlap(aabb_a_before, obb_aabb(obb_b_before)):
                continue
            if not aabbs_overlap(aabb_a_after, obb_aabb(obb_b_after)):
                continue

            iou_before = float(iou_obb(obb_a_before, obb_b_before))
            iou_after = float(iou_obb(obb_a_after, obb_b_after))

            if not (np.isfinite(iou_before) and np.isfinite(iou_after)):
                continue
            if iou_before <= 0.0 or iou_after <= 0.0:
                continue

            interact_iou = max(iou_before, iou_after)
            if interact_iou < min_interact_iou:
                continue

            score = iou_before + iou_after
            if score > best_score:
                best_score = score
                best_tid_b = tid_b
                best_iou_before = iou_before
                best_iou_after = iou_after

        if best_tid_b is None:
            continue

        # Fill unfilled gap frames of ID_A with ID_B's OBB.  Prefer ID_A's
        # interpolated heading only when it agrees with the copied OBB's
        # long axis.  If that gate fails, adopt ID_B's heading for that
        # frame without applying an additional long-axis gate to the B
        # heading; this mirrors the method description.
        dir_a_before = get_dir(corrected, tid_a, t_before)
        dir_a_after = get_dir(corrected, tid_a, t_after)
        gap_span = float(t_after - t_before)
        filled: list[int] = []
        direction_from_b: list[int] = []
        for t in range(s_start, s_end + 1):
            if get_obb(corrected, tid_a, t) is not None:
                continue  # already filled by an earlier stage
            obb_b_t = get_obb(corrected, best_tid_b, t)
            if obb_b_t is None:
                continue
            if dir_a_before is not None and dir_a_after is not None:
                alpha = (t - t_before) / gap_span
                dir_a_t: float | None = circular_interpolate_deg(dir_a_before, dir_a_after, alpha)
            elif dir_a_before is not None:
                dir_a_t = dir_a_before
            elif dir_a_after is not None:
                dir_a_t = dir_a_after
            else:
                dir_a_t = None

            dir_fill = _direction_on_long_axis(
                obb_b_t,
                dir_a_t,
                max_axis_error_deg=max_axis_error_deg,
            )
            project_direction = True
            if dir_fill is None:
                dir_fill = get_dir(corrected, best_tid_b, t)
                project_direction = False
                direction_from_b.append(t)

            local_fills.append((tid_a, t, obb_b_t, dir_fill, project_direction))
            filled.append(t)

        if filled:
            local_log.append({
                'type': 'overlap_fill',
                'action': 'fill',
                'ids': [tid_a, best_tid_b],
                'id_a': tid_a,
                'id_b': best_tid_b,
                'frame_start': s_start,
                'frame_end': s_end,
                'filled_frames': filled,
                'direction_from_id_b_frames': direction_from_b,
                'iou_before': best_iou_before,
                'iou_after': best_iou_after,
                'interact_iou': max(best_iou_before, best_iou_after),
                'min_interact_iou': min_interact_iou,
                'score': best_score,
            })

    return local_fills, local_log


def _eval_position_spikes_id(
    corrected: dict,
    tid: int,
    max_axis_error_deg: float,
) -> list[tuple]:
    local_pending: list[tuple] = []
    valid = _valid_frames(corrected, tid)
    for k in range(1, len(valid) - 1):
        t_prev, t, t_next = valid[k - 1], valid[k], valid[k + 1]
        if t_prev != t - 1 or t_next != t + 1:
            continue

        obb_P = get_obb(corrected, tid, t_prev)
        obb_C = get_obb(corrected, tid, t)
        obb_N_raw = get_obb(corrected, tid, t_next)
        if obb_P is None or obb_C is None or obb_N_raw is None:
            continue
        P = obb_center(obb_P)
        C = obb_center(obb_C)
        N = obb_center(obb_N_raw)

        # Single-frame out-and-back reversal only.  This excludes ordinary
        # monotone motion, bends, and multi-frame excursions.
        if float(np.dot(C - P, N - C)) >= 0.0:
            continue

        alpha = (t - t_prev) / (t_next - t_prev)
        obb_N_can = canonicalize_obb_points(
            obb_N_raw,
            reference=ensure_clockwise(obb_P),
        )
        new_obb = ensure_clockwise((1.0 - alpha) * obb_P + alpha * obb_N_can)
        interp_center = obb_center(new_obb)

        cost_current = max(
            float(np.linalg.norm(C - P)),
            float(np.linalg.norm(N - C)),
        )
        cost_interp = max(
            float(np.linalg.norm(interp_center - P)),
            float(np.linalg.norm(N - interp_center)),
        )
        if cost_interp >= cost_current:
            continue

        new_dir = np.nan

        local_pending.append((
            tid, t_prev, t, t_next, new_obb, interp_center, new_dir, C,
            cost_current, cost_interp,
        ))
    return local_pending


def _eval_direction_segment_flips_id(
    corrected: dict,
    tid: int,
    phase: str,
    max_axis_error_deg: float,
    flip_frame_threshold: int,
) -> tuple[list[tuple[int, float]], list[dict]]:
    """Resolve direction flips by iterative anchor-run merging.

    Valid heading frames are split into direction runs at >60 degree jumps.
    Runs spanning at least ``flip_frame_threshold`` video frames are treated as
    reliable anchors and are preserved.  Short runs are eligible only when an
    immediate neighboring anchor has a raw boundary jump of at least 120
    degrees; 60-120 degree boundaries are considered ambiguous and are excluded
    from the flip evidence.  When a flip is accepted, it is applied to a local
    direction copy, the runs are rebuilt, and the procedure is repeated until no
    further eligible flip remains.  This makes a flipped run merge into the
    adjacent anchor before the next decision is made.
    """
    logs: list[dict] = []

    valid = _valid_frames(corrected, tid)
    n = len(valid)
    if n < 2:
        return [], logs

    original_dirs: list[float | None] = [get_dir(corrected, tid, valid[k]) for k in range(n)]
    current_dirs: list[float | None] = [
        None if value is None else float(value)
        for value in original_dirs
    ]

    split_deg = 60.0
    anchor_boundary_deg = 120.0
    max_iterations = max(1, 2 * n)

    def _build_runs() -> list[tuple[int, int]]:
        """Maximal finite-heading runs separated by >60 degree jumps or missing headings."""
        runs_local: list[tuple[int, int]] = []
        k = 0
        while k < n:
            if current_dirs[k] is None:
                k += 1
                continue
            j = k
            while (
                j + 1 < n
                and current_dirs[j + 1] is not None
                and angular_distance_deg(current_dirs[j], current_dirs[j + 1]) <= split_deg  # type: ignore[arg-type]
            ):
                j += 1
            runs_local.append((k, j))
            k = j + 1
        return runs_local

    def _run_span(runs_local: list[tuple[int, int]], r: int) -> int:
        i0, i1 = runs_local[r]
        return int(valid[i1]) - int(valid[i0]) + 1

    def _run_valid_count(runs_local: list[tuple[int, int]], r: int) -> int:
        i0, i1 = runs_local[r]
        return int(i1) - int(i0) + 1

    def _is_reliable_anchor_run(runs_local: list[tuple[int, int]], r: int) -> bool:
        return _run_span(runs_local, r) >= int(flip_frame_threshold)

    def _boundary_delta(runs_local: list[tuple[int, int]], left_r: int, right_r: int) -> float | None:
        _, li1 = runs_local[left_r]
        ri0, _ = runs_local[right_r]
        if current_dirs[li1] is None or current_dirs[ri0] is None:
            return None
        return float(angular_distance_deg(current_dirs[li1], current_dirs[ri0]))  # type: ignore[arg-type]

    def _boundary_anchor_ok(
        runs_local: list[tuple[int, int]],
        anchor_r: int,
        target_r: int,
    ) -> tuple[bool, float | None, str]:
        if not _is_reliable_anchor_run(runs_local, anchor_r):
            return False, None, 'anchor_run_too_short'
        if anchor_r == target_r - 1:
            delta = _boundary_delta(runs_local, anchor_r, target_r)
        elif anchor_r == target_r + 1:
            delta = _boundary_delta(runs_local, target_r, anchor_r)
        else:
            return False, None, 'not_adjacent'
        if delta is None:
            return False, None, 'missing_boundary_heading'
        if delta < anchor_boundary_deg:
            if delta >= split_deg:
                return False, delta, 'ambiguous_boundary_90_135'
            return False, delta, 'boundary_below_90'
        return True, delta, 'ok'

    def _boundary_cost(
        runs_local: list[tuple[int, int]],
        anchor_r: int,
        target_r: int,
        target_flip: int,
    ) -> float:
        """Squared angular cost at the immediate anchor-target boundary only."""
        ai0, ai1 = runs_local[anchor_r]
        ti0, ti1 = runs_local[target_r]

        if anchor_r == target_r - 1:
            if current_dirs[ai1] is None or current_dirs[ti0] is None:
                return float('inf')
            a_dir = float(current_dirs[ai1])  # type: ignore[arg-type]
            t_dir = float(current_dirs[ti0])  # type: ignore[arg-type]
            if target_flip:
                t_dir = (t_dir + 180.0) % 360.0
            return float(angular_distance_deg(a_dir, t_dir) ** 2)

        if anchor_r == target_r + 1:
            if current_dirs[ti1] is None or current_dirs[ai0] is None:
                return float('inf')
            t_dir = float(current_dirs[ti1])  # type: ignore[arg-type]
            a_dir = float(current_dirs[ai0])  # type: ignore[arg-type]
            if target_flip:
                t_dir = (t_dir + 180.0) % 360.0
            return float(angular_distance_deg(t_dir, a_dir) ** 2)

        return float('inf')

    def _collect_target_evidence(
        runs_local: list[tuple[int, int]],
        r: int,
    ) -> tuple[list[dict], list[dict]]:
        anchor_infos: list[dict] = []
        rejected_neighbors: list[dict] = []
        for side, anchor_r in (('left', r - 1), ('right', r + 1)):
            if anchor_r < 0 or anchor_r >= len(runs_local):
                continue
            ok, delta, reason = _boundary_anchor_ok(runs_local, anchor_r, r)
            info = {
                'side': side,
                'run_index': int(anchor_r),
                'frame_start': int(valid[runs_local[anchor_r][0]]),
                'frame_end': int(valid[runs_local[anchor_r][1]]),
                'run_span': _run_span(runs_local, anchor_r),
                'valid_count': _run_valid_count(runs_local, anchor_r),
                'boundary_delta': None if delta is None else float(delta),
                'reason': reason,
            }
            if not ok:
                rejected_neighbors.append(info)
                continue
            keep_cost = _boundary_cost(runs_local, anchor_r, r, 0)
            flip_cost = _boundary_cost(runs_local, anchor_r, r, 1)
            if not (np.isfinite(keep_cost) and np.isfinite(flip_cost)):
                info['reason'] = 'invalid_boundary_cost'
                rejected_neighbors.append(info)
                continue
            info.update({
                'reason': 'ok',
                'cost_keep': float(keep_cost),
                'cost_flip': float(flip_cost),
                'cost_pairs': 1,
            })
            anchor_infos.append(info)
        return anchor_infos, rejected_neighbors

    def _flip_run_in_local_dirs(runs_local: list[tuple[int, int]], r: int) -> list[tuple[int, float]]:
        i0, i1 = runs_local[r]
        run_changes: list[tuple[int, float]] = []
        for k_s in range(i0, i1 + 1):
            if current_dirs[k_s] is None:
                continue
            t = int(valid[k_s])
            new_raw = (float(current_dirs[k_s]) + 180.0) % 360.0
            obb_t = get_obb(corrected, tid, t)
            if obb_t is not None:
                projected = _direction_on_long_axis(
                    obb_t,
                    new_raw,
                    max_axis_error_deg=max_axis_error_deg,
                )
                new_dir = projected if projected is not None else new_raw
            else:
                new_dir = new_raw
            current_dirs[k_s] = float(new_dir)
            run_changes.append((t, float(new_dir)))
        return run_changes

    def _snapshot_run_dirs(runs_local: list[tuple[int, int]], r: int) -> list[tuple[int, float | None]]:
        i0, i1 = runs_local[r]
        return [(k_s, current_dirs[k_s]) for k_s in range(i0, i1 + 1)]

    def _restore_run_dirs(snapshot: list[tuple[int, float | None]]) -> None:
        for k_s, value in snapshot:
            current_dirs[int(k_s)] = value

    def _after_run_containing_interval(
        runs_local: list[tuple[int, int]],
        i0: int,
        i1: int,
    ) -> int | None:
        for rr, (r0, r1) in enumerate(runs_local):
            if int(r0) <= int(i0) and int(i1) <= int(r1):
                return int(rr)
        return None

    def _anchor_merge_status(
        runs_before: list[tuple[int, int]],
        runs_after: list[tuple[int, int]],
        target_r: int,
        anchor_infos: list[dict],
    ) -> dict:
        """Return whether a tentative flip created a new target-anchor run merge."""
        ti0, ti1 = runs_before[int(target_r)]
        target_after = _after_run_containing_interval(runs_after, ti0, ti1)
        merged_anchors: list[dict] = []

        if target_after is not None:
            for info in anchor_infos:
                anchor_r = int(info['run_index'])
                ai0, ai1 = runs_before[anchor_r]
                anchor_after = _after_run_containing_interval(runs_after, ai0, ai1)
                if anchor_after is None or anchor_after != target_after:
                    continue
                merged_anchors.append({
                    'side': info.get('side'),
                    'run_index_before': int(anchor_r),
                    'frame_start': int(valid[ai0]),
                    'frame_end': int(valid[ai1]),
                    'boundary_delta_before': info.get('boundary_delta'),
                    'after_run_index': int(anchor_after),
                })

        return {
            'merged': bool(merged_anchors),
            'merged_anchors': merged_anchors,
            'run_count_before': int(len(runs_before)),
            'run_count_after': int(len(runs_after)),
            'target_run_index_before': int(target_r),
            'target_after_run_index': target_after,
            'target_frame_start': int(valid[ti0]),
            'target_frame_end': int(valid[ti1]),
        }

    total_flipped_runs = 0
    total_rejected_no_merge = 0
    terminated_by_limit = False

    for iteration in range(1, max_iterations + 1):
        runs = _build_runs()
        if not runs:
            break

        candidates: list[dict] = []
        for r in range(len(runs)):
            if _is_reliable_anchor_run(runs, r):
                continue

            i0, i1 = runs[r]
            anchor_infos, rejected_neighbors = _collect_target_evidence(runs, r)
            if not anchor_infos:
                continue

            cost_keep = float(sum(info['cost_keep'] for info in anchor_infos))
            cost_flip = float(sum(info['cost_flip'] for info in anchor_infos))
            if cost_flip < cost_keep:
                candidates.append({
                    'run_index': int(r),
                    'frame_start': int(valid[i0]),
                    'frame_end': int(valid[i1]),
                    'length': int(valid[i1]) - int(valid[i0]) + 1,
                    'valid_count': _run_valid_count(runs, r),
                    'run_span': _run_span(runs, r),
                    'cost_keep': cost_keep,
                    'cost_flip': cost_flip,
                    'improvement': float(cost_keep - cost_flip),
                    'anchors': anchor_infos,
                    'rejected_neighbors': rejected_neighbors,
                })

        if not candidates:
            break

        # Try candidates in deterministic best-first order.  A flip is accepted
        # only if the tentative 180-degree inversion creates a new run merge
        # between the target and at least one supporting anchor.  Tentative
        # flips that do not create a merge are reverted and logged once in this
        # iteration; if no candidate creates a new merge, the loop stops without
        # rebuilding the identical state again.
        candidates.sort(
            key=lambda item: (
                -float(item['improvement']),
                int(item['frame_start']),
                int(item['frame_end']),
            ),
        )

        accepted: tuple[dict, list[tuple[int, float]], dict] | None = None
        rejected_no_merge_this_iteration = 0
        runs_before = runs

        for selected in candidates:
            selected_run = int(selected['run_index'])
            snapshot = _snapshot_run_dirs(runs_before, selected_run)
            run_changes = _flip_run_in_local_dirs(runs_before, selected_run)
            runs_after = _build_runs()
            merge_status = _anchor_merge_status(
                runs_before,
                runs_after,
                selected_run,
                selected['anchors'],
            )

            if bool(merge_status['merged']):
                accepted = (selected, run_changes, merge_status)
                break

            _restore_run_dirs(snapshot)
            rejected_no_merge_this_iteration += 1
            total_rejected_no_merge += 1
            logs.append({
                'type': 'direction_segment_flip',
                'phase': phase,
                'iteration': int(iteration),
                'ids': [tid],
                'frame_start': int(selected['frame_start']),
                'frame_end': int(selected['frame_end']),
                'length': int(selected['length']),
                'valid_count': int(selected['valid_count']),
                'run_span': int(selected['run_span']),
                'action': 'abstain',
                'reason': 'reject_no_new_anchor_merge',
                'selection_policy': 'iterative_adjacent_reliable_anchor_boundary_cost_require_merge',
                'cost_keep': float(selected['cost_keep']),
                'cost_flip': float(selected['cost_flip']),
                'improvement': float(selected['improvement']),
                'anchor_boundary_threshold_deg': float(anchor_boundary_deg),
                'split_threshold_deg': float(split_deg),
                'anchor_frame_threshold': int(flip_frame_threshold),
                'anchors': selected['anchors'],
                'rejected_neighbors': selected['rejected_neighbors'],
                'changed_frames_if_accepted': [int(frame) for frame, _ in run_changes],
                'anchor_merge_status': merge_status,
                'num_candidate_flips_this_iteration': int(len(candidates)),
                'anchor_merge_rebuild': False,
            })

        if accepted is None:
            logs.append({
                'type': 'direction_segment_flip',
                'phase': phase,
                'iteration': int(iteration),
                'ids': [tid],
                'action': 'abstain',
                'reason': 'stop_no_new_anchor_merge',
                'selection_policy': 'iterative_adjacent_reliable_anchor_boundary_cost_require_merge',
                'num_candidate_flips_this_iteration': int(len(candidates)),
                'num_rejected_no_merge_this_iteration': int(rejected_no_merge_this_iteration),
                'total_rejected_no_merge': int(total_rejected_no_merge),
                'anchor_merge_rebuild': False,
            })
            break

        selected, run_changes, merge_status = accepted
        total_flipped_runs += 1

        logs.append({
            'type': 'direction_segment_flip',
            'phase': phase,
            'iteration': int(iteration),
            'ids': [tid],
            'frame_start': int(selected['frame_start']),
            'frame_end': int(selected['frame_end']),
            'length': int(selected['length']),
            'valid_count': int(selected['valid_count']),
            'run_span': int(selected['run_span']),
            'action': 'segment_flip',
            'assign_type': int(ATYPE_DIR_FLIP),
            'assign_type_source': 'correction_direction_segment_flip',
            'selection_policy': 'iterative_adjacent_reliable_anchor_boundary_cost_require_merge',
            'cost_keep': float(selected['cost_keep']),
            'cost_flip': float(selected['cost_flip']),
            'improvement': float(selected['improvement']),
            'anchor_boundary_threshold_deg': float(anchor_boundary_deg),
            'split_threshold_deg': float(split_deg),
            'anchor_frame_threshold': int(flip_frame_threshold),
            'anchors': selected['anchors'],
            'rejected_neighbors': selected['rejected_neighbors'],
            'changed_frames': [int(frame) for frame, _ in run_changes],
            'num_candidate_flips_this_iteration': int(len(candidates)),
            'num_rejected_no_merge_this_iteration': int(rejected_no_merge_this_iteration),
            'anchor_merge_status': merge_status,
            'anchor_merge_rebuild': True,
        })
    else:
        terminated_by_limit = True

    # Final abstain log for short runs that still have no reliable adjacent
    # anchor after all accepted flips have been merged.  Cost-favors-keep cases
    # are intentionally not logged to avoid noisy expected decisions.
    final_runs = _build_runs()
    for r, (i0, i1) in enumerate(final_runs):
        if _is_reliable_anchor_run(final_runs, r):
            continue
        anchor_infos, rejected_neighbors = _collect_target_evidence(final_runs, r)
        if anchor_infos:
            continue
        logs.append({
            'type': 'direction_segment_flip',
            'phase': phase,
            'iteration': int(total_flipped_runs + 1),
            'ids': [tid],
            'frame_start': int(valid[i0]),
            'frame_end': int(valid[i1]),
            'length': int(valid[i1]) - int(valid[i0]) + 1,
            'valid_count': int(_run_valid_count(final_runs, r)),
            'run_span': int(_run_span(final_runs, r)),
            'action': 'abstain',
            'reason': 'no_reliable_adjacent_anchor',
            'selection_policy': 'iterative_adjacent_reliable_anchor_boundary_cost',
            'details': {
                'role': 'target',
                'flip': 0,
                'reason': 'no_reliable_adjacent_anchor',
                'run_span': int(_run_span(final_runs, r)),
                'run_valid_count': int(_run_valid_count(final_runs, r)),
                'rejected_neighbors': rejected_neighbors,
                'anchor_boundary_threshold_deg': float(anchor_boundary_deg),
                'split_threshold_deg': float(split_deg),
                'anchor_frame_threshold': int(flip_frame_threshold),
            },
        })

    if terminated_by_limit:
        logs.append({
            'type': 'direction_segment_flip',
            'phase': phase,
            'ids': [tid],
            'action': 'abstain',
            'reason': 'iteration_limit_reached',
            'max_iterations': int(max_iterations),
            'selection_policy': 'iterative_adjacent_reliable_anchor_boundary_cost',
        })

    changes: list[tuple[int, float]] = []
    for k, (old_dir, new_dir) in enumerate(zip(original_dirs, current_dirs)):
        if old_dir is None or new_dir is None:
            continue
        if angular_distance_deg(float(old_dir), float(new_dir)) > 1e-6:
            changes.append((int(valid[k]), float(new_dir)))

    return changes, logs

def _eval_missing_frames_id(corrected: dict, tid: int) -> list[int]:
    valid = _valid_frames(corrected, tid)
    if len(valid) < 2:
        return []
    present = set(valid)
    return sorted(set(range(valid[0], valid[-1] + 1)) - present)

# Correction Stage 1: KF-gated gap fill

def _kalman_gap_fill(
    corrected: dict,
    num_objects: int,
    strong_iou: float,
    strong_dir_deg: float,
    max_age: int,
    num_workers: int = 1,
    *,
    max_axis_error_deg: float = 45.0,
    concurrent_jobs: int = 1,
) -> list[dict]:
    """Correction Stage 1: fill confirmed interior gaps with KF-gated chords.

    Rebuilds independent KFs per ID (forward, dt=+1) from the first valid frame
    to the last.  Interior gaps pass through four gates (G1-G4) before being
    filled with a linear chord between the anchor OBBs.  Rejected gaps are left
    empty for missing-frame marking.  Processing is parallelised per ID;
    output order is ID-ascending regardless of worker count.
    """

    # Run per-ID workers, collecting results in ID-ascending order for determinism
    results_list = _run_id_eval(
        corrected,
        num_objects,
        num_workers,
        desc='Correction 1/5 KF gap fill',
        serial_fn=_eval_kalman_gap_fill_id,
        worker_args=(strong_iou, strong_dir_deg, max_age, max_axis_error_deg),
        buffer_names=('obb_buf', 'class_buf', 'assign_type_buf'),
        concurrent_jobs=concurrent_jobs,
    )

    # Merge results into corrected sequentially (ID-ascending)
    log: list[dict] = []
    for tid, (fills, local_log) in enumerate(results_list):
        for t_fill, pts_fill, dir_fill in fills:
            _write_synth_frame(corrected, tid, t_fill, pts_fill, dir_fill,
                               assign_type=float(ATYPE_KF_FILL),
                               max_axis_error_deg=max_axis_error_deg)
        log.extend(local_log)

    num_corrections = sum(1 for entry in log if entry.get('action') == 'fill')
    print(
        f'Correction 1/5 KF gap fill: num corrections={num_corrections}',
        flush=True,
    )
    return log


# Correction Stage 2: overlap gap fill

def _overlap_gap_fill(
    corrected: dict,
    num_objects: int,
    max_gap: int,
    min_interact_iou: float,
    num_workers: int = 1,
    *,
    max_axis_error_deg: float = 45.0,
    concurrent_jobs: int = 1,
) -> list[dict]:
    """Correction Stage 2: fill gaps caused by overlap with another ID.

    Evaluation is parallelised per ID_A with threads because every ID_A may read
    every other track.  Threads share the read-only snapshot without Windows
    spawn copies.  The main thread then materialises all accepted fills in
    deterministic ID/frame order, so completion order cannot affect the output.
    Candidate overlap IDs must reach ``min_interact_iou`` at one of the two gap
    anchors, matching the embedding interaction threshold semantics.
    """
    min_interact_iou = float(min_interact_iou)

    results = _run_id_eval(
        corrected,
        num_objects,
        num_workers,
        desc='Correction 2/5 overlap gap fill',
        serial_fn=_eval_overlap_gap_fill_id,
        worker_args=(num_objects, max_gap, min_interact_iou, max_axis_error_deg),
        shared_read=True,
        concurrent_jobs=concurrent_jobs,
    )

    log: list[dict] = []
    for tid_a, (fills, local_log) in enumerate(results):
        for fill_tid, t, obb_b_t, dir_fill, project_direction in sorted(
            fills,
            key=lambda item: (int(item[0]), int(item[1])),
        ):
            _write_synth_frame(
                corrected,
                int(fill_tid),
                int(t),
                obb_b_t,
                dir_fill,
                assign_type=float(ATYPE_OVERLAP_FILL),
                history_source='correction_overlap_fill',
                max_axis_error_deg=max_axis_error_deg,
                project_direction=bool(project_direction),
            )
        log.extend(local_log)

    print(
        f'Correction 2/5 overlap gap fill: num corrections={len(log)}',
        flush=True,
    )
    return sorted(
        log,
        key=lambda entry: (int(entry.get('id_a', entry['ids'][0])), int(entry.get('frame_start', 0))),
    )


# Correction Stage 4: fix position and direction spikes

def _select_adjacent_candidate_winners(
    candidates: list[tuple],
    *,
    target_frame,
    improvement,
) -> tuple[list[tuple], list[tuple[tuple, tuple]]]:
    """Keep the maximum-improvement item in each run of adjacent candidates."""
    by_tid: dict[int, list[tuple]] = {}
    for item in candidates:
        by_tid.setdefault(int(item[0]), []).append(item)

    selected: list[tuple] = []
    rejected: list[tuple[tuple, tuple]] = []
    for tid in sorted(by_tid):
        ordered = sorted(by_tid[tid], key=lambda item: int(target_frame(item)))
        groups: list[list[tuple]] = []
        for item in ordered:
            frame = int(target_frame(item))
            if not groups or frame > int(target_frame(groups[-1][-1])) + 1:
                groups.append([item])
            else:
                groups[-1].append(item)
        for group in groups:
            winner = min(
                group,
                key=lambda item: (-float(improvement(item)), int(target_frame(item))),
            )
            selected.append(winner)
            rejected.extend((item, winner) for item in group if item is not winner)
    selected.sort(key=lambda item: (int(item[0]), int(target_frame(item))))
    rejected.sort(key=lambda pair: (int(pair[0][0]), int(target_frame(pair[0]))))
    return selected, rejected


def _fix_position_spikes(
    corrected: dict,
    num_objects: int,
    *,
    num_workers: int,
    phase: str,
    pass_index: int,
    max_axis_error_deg: float = 45.0,
    concurrent_jobs: int = 1,
) -> list[dict]:
    """Replace only single-frame position spikes with interpolated OBBs.

    This intentionally uses the conservative old-style detector:
    - candidate is exactly one frame bracketed by valid prev/next frames;
    - geometric reversal gate: dot(C-P, N-C) < 0;
    - strict improvement gate: interpolated local cost < current local cost;
    - evaluation is read-only on the pre-pass snapshot;
    - adjacent candidates compete by local-cost improvement.

    Multi-frame excursions are deliberately not treated as position spikes.
    """
    # Evaluation is per-ID and read-only, so it runs through _run_id_eval
    # with the module-level _eval_position_spikes_id: a nested closure
    # cannot be pickled for a spawned worker.  Result tuple fields are
    # tid, t_prev, t, t_next, new_obb, interp_center, new_dir, C,
    # cost_current, cost_interp.
    pending_by_tid = _run_id_eval(
        corrected,
        num_objects,
        num_workers,
        desc=f'Fix position spikes [{phase} pass {pass_index}]',
        serial_fn=_eval_position_spikes_id,
        worker_args=(max_axis_error_deg,),
        buffer_names=('obb_buf', 'class_buf', 'assign_type_buf'),
        concurrent_jobs=concurrent_jobs,
    )

    all_pending = sorted([
        item
        for tid_pending in pending_by_tid
        for item in tid_pending
    ], key=lambda item: (int(item[0]), int(item[2])))

    pending, adjacent_losers = _select_adjacent_candidate_winners(
        all_pending,
        target_frame=lambda item: int(item[2]),
        improvement=lambda item: float(item[-2]) - float(item[-1]),
    )

    # Application phase is write-only after all candidates are frozen/selected.
    correction_log: list[dict] = []
    abstain_log: list[dict] = []
    for item, winner in adjacent_losers:
        (
            tid, t_prev, t, t_next, _new_obb, interp_center, _new_dir, C,
            cost_current, cost_interp,
        ) = item
        improvement = float(cost_current) - float(cost_interp)
        abstain_log.append({
            'type': 'position_spike',
            'phase': phase,
            'pass_index': pass_index,
            'frame': int(t),
            'ids': [int(tid)],
            'action': 'abstain',
            'reason': 'lower_adjacent_improvement',
            'anchor_frames': [int(t_prev), int(t_next)],
            'competing_frame': int(winner[2]),
            'old_center': [float(C[0]), float(C[1])],
            'new_center': [float(interp_center[0]), float(interp_center[1])],
            'cost_current': float(cost_current),
            'cost_interp': float(cost_interp),
            'improvement': improvement,
            'improvement_rate': improvement / float(cost_current) if float(cost_current) > 0.0 else float('inf'),
            'reversal_detected': True,
        })

    pos = corrected.setdefault('pos_buf', {})
    for (
        tid, t_prev, t, t_next, new_obb, interp_center, new_dir, C,
        cost_current, cost_interp,
    ) in pending:
        tid = int(tid)
        t = int(t)
        pos.setdefault(tid, {})[t] = (float(interp_center[0]), float(interp_center[1]))
        corrected.setdefault('obb_buf', {}).setdefault(tid, {})[t] = obb_to_tuple(new_obb)
        corrected.setdefault('class_buf', {}).setdefault(tid, {})[t] = (float(new_dir),)
        corrected.setdefault('obb_corrected_buf', {}).setdefault(tid, {})[t] = (1.0,)
        corrected.setdefault('direction_corrected_buf', {}).setdefault(tid, {})[t] = (1.0,)
        _record_assign_type(corrected, tid, t, ATYPE_POS_FIX, 'correction_position_spike')

        improvement = float(cost_current) - float(cost_interp)
        correction_log.append({
            'type': 'position_spike',
            'phase': phase,
            'pass_index': pass_index,
            'frame': t,
            'ids': [tid],
            'action': 'replace_with_interpolated_obb',
            'anchor_prev': int(t_prev),
            'anchor_next': int(t_next),
            'anchor_frames': [int(t_prev), int(t_next)],
            'run_len': 1,
            'old_center': [float(C[0]), float(C[1])],
            'new_center': [float(interp_center[0]), float(interp_center[1])],
            'cost_current': float(cost_current),
            'cost_interp': float(cost_interp),
            'improvement': improvement,
            'improvement_rate': improvement / float(cost_current) if float(cost_current) > 0.0 else float('inf'),
            'reversal_detected': True,
            'nonoverlap_selected': True,
            'assign_type': int(ATYPE_POS_FIX),
            'assign_type_source': 'correction_position_spike',
        })

    print(
        f'Fix position spikes [{phase} pass {pass_index}]: '
        f'num corrections={len(correction_log)}, abstain={len(abstain_log)}',
        flush=True,
    )
    return sorted(
        correction_log + abstain_log,
        key=lambda entry: (int(entry['ids'][0]), int(entry['frame'])),
    )


def _fix_direction_segment_flips(*args, **kwargs):
    return []

def _fix_local_spikes(
    corrected: dict,
    num_objects: int,
    *,
    num_workers: int,
    phase: str,
    flip_frame_threshold: int,
    max_axis_error_deg: float = 45.0,
    concurrent_jobs: int = 1,
) -> list[dict]:
    """Run one conservative pass for position spikes and anchor-based direction flips.

    Pass A: position spikes once, retaining the best adjacent candidate.
    Pass B: single- and multi-frame direction flips by adjacent reliable anchor runs.
    """

    logs: list[dict] = _fix_position_spikes(
        corrected, num_objects,
        num_workers=num_workers,
        phase=phase,
        pass_index=1,
        max_axis_error_deg=max_axis_error_deg,
        concurrent_jobs=concurrent_jobs,
    )

    # B. Fix bounded direction runs, including single-frame 180-degree runs.
    logs += _fix_direction_segment_flips(
        corrected, num_objects,
        num_workers=num_workers,
        phase=phase,
        max_axis_error_deg=max_axis_error_deg,
        flip_frame_threshold=flip_frame_threshold,
        concurrent_jobs=concurrent_jobs,
    )

    return logs


# Correction Stage 5: mark interior missing frames

def _mark_missing_frames(
    corrected: dict,
    num_objects: int,
    num_workers: int = 1,
    *,
    concurrent_jobs: int = 1,
) -> list[dict]:
    """Assign ATYPE_MISSING to interior gaps in each track's assign_type_buf.

    This inexpensive final scan remains serial to avoid another Windows process
    startup after the geometry pools. Writes are applied in deterministic order.
    ``num_workers`` and ``concurrent_jobs`` remain accepted for call compatibility.
    """

    missing_by_tid = _run_id_eval(
        corrected,
        num_objects,
        1,
        desc='Correction 5/5 mark missing frames',
        serial_fn=_eval_missing_frames_id,
        buffer_names=('obb_buf',),
    )

    log: list[dict] = []
    for tid, frames in enumerate(missing_by_tid):
        for frame in frames:
            corrected.setdefault('assign_type_buf', {}).setdefault(tid, {})[frame] = (float(ATYPE_MISSING),)
            corrected.setdefault('assign_type_history_buf', {}).setdefault(tid, {}).setdefault(frame, []).append(
                ('correction_missing', float(ATYPE_MISSING))
            )
            log.append({'type': 'missing', 'ids': [tid], 'frame': frame, 'action': 'mark_missing'})

    print(
        f'Correction 5/5 mark missing frames: num corrections={len(log)}',
        flush=True,
    )
    return log


# Main

def _auto_img_size(corrected: dict, fragments: list) -> int:
    """Max long side among isolated fragment OBBs, rounded up to a multiple of 32."""
    max_side = 0.0
    obb_buf = corrected.get('obb_buf', {})
    for frag in fragments:
        for frame in frag.frames:
            raw = obb_buf.get(frag.obj_id, {}).get(frame)
            if raw is None:
                continue
            obb = tuple_to_obb(raw)
            if obb is None or not valid_obb(obb):
                continue
            rect = obb_points_to_rect(obb)
            side = max(float(rect[2]), float(rect[3]))
            if side > max_side:
                max_side = side
    size = max(32, int(math.ceil(max_side)))
    return ((size + 31) // 32) * 32


def _resolve_existing_correction_out_dir(spec: dict, *, path_builder=None) -> tuple[str, str]:
    if path_builder is None:
        path_builder = build_tracking_out_dir
    dataset_name = spec['dataset_name']
    canonical_dataset_name = resolve_existing_experiment_dir_name(
        spec['session_path'],
        spec['model_name'],
        dataset_name,
        stages=('tracking',),
        warn_fn=print,
    )
    canonical_out_dir = path_builder(
        spec['session_path'],
        spec['model_name'],
        canonical_dataset_name,
        spec['run_name'],
        spec['video_name'],
    )
    return canonical_dataset_name, canonical_out_dir


def _run_single_correction_job(spec: dict) -> None:
    """Run correction for one (video_name, run_name) pair.

    Ordering is intentionally different when embedding is enabled:
      - embedding enabled and usable: fwd -> embedding ID resolution -> gap fill -> spike/flip -> missing -> id_resolved
      - embedding skipped/unusable: fwd -> gap fill -> spike/flip -> missing -> filled
      - embedding disabled: fwd -> gap fill -> spike/flip -> missing -> filled
    """
    out_dir          = spec['out_dir']
    session_path     = spec['session_path']
    model_name       = spec['model_name']
    dataset_name     = spec['dataset_name']
    run_name         = spec['run_name']
    video_name       = spec['video_name']
    num_objects      = spec['num_objects']
    strong_iou       = spec['strong_iou']
    strong_dir_deg   = spec['strong_dir_deg']
    max_age          = spec['max_age']
    flip_frame_threshold = int(spec['flip_frame_threshold'])
    flip_duration_sec = float(spec['flip_duration_sec'])
    flip_source_fps = float(spec['source_fps'])
    max_axis_error_deg = spec.get('max_axis_error_deg', 45.0)
    num_workers      = spec['num_workers']
    video_path       = spec.get('video_path', '')
    embedding_seed   = normalize_seed(spec.get('embedding_seed', 0))
    embedding_cfg    = spec.get('embedding', {}) or {}
    background_path  = str(spec.get('background_path') or embedding_cfg.get('BACKGROUND_PATH', '')).strip()
    emb_enable       = _cfg_bool(
        spec.get('embedding_enabled', embedding_cfg.get('ENABLE', False))
    )
    _img_size_raw    = embedding_cfg.get('IMG_SIZE', 'auto')
    embedding_device = embedding_cfg.get('DEVICE', 'auto')
    preview_count    = int(embedding_cfg.get('PREVIEW_COUNT', 100))
    episode_max_len  = int(embedding_cfg.get('EPISODE_MAX_LEN', 0))
    min_interact_iou = FIXED_INTERACT_IOU
    concurrent_jobs = max(1, int(spec.get('concurrent_correction_jobs', 1)))

    dataset_name, out_dir = _resolve_existing_correction_out_dir(spec)
    spec = {**spec, 'dataset_name': dataset_name, 'out_dir': out_dir}

    id_resolved_req = required_id_resolved_outputs(out_dir)
    filled_req = required_filled_outputs(out_dir)
    if emb_enable:
        if all(output_has_data(kind, p) for kind, p in id_resolved_req):
            print(f'ID-resolved outputs already exist. Skipping. [{run_name}:{video_name}]')
            save_existing_corrected_final_result(spec, ID_RESOLVED_ARTIFACT_SUFFIX)
            return
        if all(output_has_data(kind, p) for kind, p in filled_req):
            print(f'Filled outputs already exist. Skipping. [{run_name}:{video_name}]')
            save_existing_corrected_final_result(spec, FILLED_ARTIFACT_SUFFIX)
            return
    else:
        if all(output_has_data(kind, p) for kind, p in filled_req):
            print(f'Filled outputs already exist. Skipping. [{run_name}:{video_name}]')
            save_existing_corrected_final_result(spec, FILLED_ARTIFACT_SUFFIX)
            return

    fwd_pkl = _artifact_path(out_dir, BUFFERS_PICKLE_NAME, FWD_ARTIFACT_SUFFIX)
    if not os.path.exists(fwd_pkl):
        print(f'fwd buffers not found; skipping. [{run_name}:{video_name}]')
        return

    print(f'Correcting [{run_name}:{video_name}]...')
    print(
        f'  Flip anchor threshold: {flip_frame_threshold} frames '
        f'({flip_duration_sec:.3g}s at {flip_source_fps:.3f} fps). '
        f'[{run_name}:{video_name}]',
        flush=True,
    )
    corrected = load_tracking_buffers(out_dir, FWD_ARTIFACT_SUFFIX)
    corrections_log: list[dict] = []
    embedding_log: list[dict] = []
    embedding_resolver_ran = False

    if emb_enable:
        from without_direction_estimation.identity_correction import (
            build_fragments,
            load_cached_crop_metadata,
            load_cached_embedding_artifacts,
            load_or_train_embedding,
            embedding_available,
            EmbeddingSegmentationRequired,
        )

        fragments: list = []
        centroids: dict = {}
        best_ss = 0.0
        segmentation_missing = False

        print(f'  [SEED] embedding video={video_name} run={run_name} seed={embedding_seed}')
        try:
            cached_artifacts = load_cached_embedding_artifacts(
                out_dir, background_path, num_objects, seed=embedding_seed,
                device_config=embedding_device,
                session_path=session_path, video_path=video_path,
            )
            if cached_artifacts is not None:
                fragments, centroids, best_ss = cached_artifacts
                print(f'  Reusing embedding artifacts. [{run_name}:{video_name}]')
            else:
                cached_embedding = load_cached_crop_metadata(out_dir, background_path)
                if cached_embedding is not None:
                    fragments, emb_img_size = cached_embedding
                    print(
                        f'  Reusing embedding crop cache metadata; img_size={emb_img_size}. '
                        f'[{run_name}:{video_name}]'
                    )
                else:
                    fragments = build_fragments(corrected, num_objects, num_workers=num_workers)
                    emb_img_size = (
                        _auto_img_size(corrected, fragments)
                        if str(_img_size_raw).strip().lower() == 'auto'
                        else max(32, ((int(_img_size_raw) + 31) // 32) * 32)
                    )
                _, centroids, best_ss = load_or_train_embedding(
                    fragments, corrected, video_path, out_dir, num_objects, emb_img_size,
                    background_path,
                    preview_count=preview_count,
                    num_workers=num_workers,
                    seed=embedding_seed,
                    device_config=embedding_device,
                    session_path=session_path,
                )
        except EmbeddingSegmentationRequired as exc:
            segmentation_missing = True
            centroids = {}
            print(
                f'  {exc} Aborting embedding-based ID resolution; '
                f'continuing with geometry-only post-processing. [{run_name}:{video_name}]',
                flush=True,
            )

        if not segmentation_missing:
            if embedding_available(best_ss) and best_ss >= 0.91:
                print(
                    f'  Embedding SS={best_ss:.3f} >= 0.91; '
                    f'embedding corrections active. [{run_name}:{video_name}]'
                )
            elif embedding_available(best_ss):
                print(
                    f'  Embedding SS={best_ss:.3f} < 0.91; '
                    f'using best checkpoint for embedding corrections. [{run_name}:{video_name}]',
                    flush=True,
                )
            else:
                centroids = {}
                print(
                    f'  Embedding SS={best_ss:.3f} is not usable; '
                    f'continuing geometry correction without embedding relabels. '
                    f'[{run_name}:{video_name}]',
                    flush=True,
                )

        if centroids:
            embedding_resolver_ran = True
            embedding_log = _resolve_contact_episodes_embedding(
                corrected, num_objects, fragments, centroids, episode_max_len,
                corrections_log, min_interact_iou,
                progress_label=f' [{run_name}:{video_name}]',
            )
            embedding_corrections = sum(
                1 for entry in embedding_log if entry.get('action') == 'relabel'
            )
            print(
                f'  Finished embedding contact episode resolution; '
                f'num corrections={embedding_corrections}; '
                f'decisions={len(embedding_log)}. [{run_name}:{video_name}]',
                flush=True,
            )
            corrections_log += embedding_log

    # Geometry correction is intentionally applied only after embedding when
    # embedding is enabled.  When embedding is disabled, this is the ordinary
    # single correction pass from fwd.
    geometry_phase = 'post_embedding' if emb_enable else 'no_embedding'

    kf_log = _kalman_gap_fill(
        corrected, num_objects, strong_iou, strong_dir_deg, max_age, num_workers,
        max_axis_error_deg=max_axis_error_deg,
        concurrent_jobs=concurrent_jobs,
    )
    corrections_log += kf_log

    overlap_log = _overlap_gap_fill(
        corrected, num_objects, max_age, min_interact_iou, num_workers,
        max_axis_error_deg=max_axis_error_deg,
        concurrent_jobs=concurrent_jobs,
    )
    corrections_log += overlap_log

    corrections_log += _fix_local_spikes(
        corrected, num_objects,
        num_workers=num_workers,
        phase=geometry_phase,
        max_axis_error_deg=max_axis_error_deg,
        flip_frame_threshold=flip_frame_threshold,
        concurrent_jobs=concurrent_jobs,
    )

    missing_log = _mark_missing_frames(
        corrected,
        num_objects,
        num_workers,
        concurrent_jobs=concurrent_jobs,
    )
    corrections_log += missing_log

    kf_filled_gaps   = sum(1 for e in kf_log if e.get('action') == 'fill')
    kf_filled_frames = sum(e.get('gap_len', 0) for e in kf_log if e.get('action') == 'fill')
    kf_rejected_gaps = sum(1 for e in kf_log if e.get('action', '').startswith('reject'))
    ov_filled_gaps   = sum(1 for e in overlap_log if e.get('action') == 'fill')
    ov_filled_frames = sum(len(e.get('filled_frames', [])) for e in overlap_log if e.get('action') == 'fill')
    emb_relabels     = sum(1 for e in embedding_log if e.get('action') == 'relabel')
    emb_identity     = sum(1 for e in embedding_log if e.get('action') == 'identity')
    emb_abstains     = sum(1 for e in embedding_log if e.get('action') == 'abstain')
    missing_frames_n = len(missing_log)

    fill_summary = (
        f'KF fill: {kf_filled_gaps} gaps, {kf_filled_frames} frames, '
        f'{kf_rejected_gaps} rejected. Overlap fill: {ov_filled_gaps} gaps, '
        f'{ov_filled_frames} frames.'
    )
    embedding_summary = (
        f'Embedding episodes: {emb_relabels} relabel, {emb_identity} identity, {emb_abstains} abstain.'
        if emb_enable
        else 'Embedding: disabled.'
    )
    print(
        f'{embedding_summary} {fill_summary} '
        f'ATYPE_MISSING: {missing_frames_n}. [{run_name}:{video_name}]'
    )

    output_suffix = (
        ID_RESOLVED_ARTIFACT_SUFFIX
        if embedding_resolver_ran
        else FILLED_ARTIFACT_SUFFIX
    )
    print(
        f'Saving corrected outputs as {output_suffix}... '
        f'[{run_name}:{video_name}]',
        flush=True,
    )
    save_corrected_outputs(
        out_dir,
        corrected,
        num_objects,
        corrections_log=corrections_log,
        artifact_suffix=output_suffix,
    )
    save_final_result_csv(
        session_path,
        model_name,
        dataset_name,
        run_name,
        video_name,
        corrected,
        num_objects,
        output_suffix,
    )
    # 'identity' (embedding positively confirmed no swap was needed) is a
    # completed decision, not an applied change -- excluded here the same
    # way 'abstain' is.
    applied = sum(1 for e in corrections_log if e.get('action') not in ('abstain', 'identity'))
    print(f'Correction done. {applied} applied ({len(corrections_log)} decisions). [{run_name}:{video_name}]')

def _run_correction_specs(
    specs: list[dict],
    outer_epoch: int,
    workers_per_epoch: int,
    phase_label: str,
    *,
    run_job=None,
) -> None:
    if run_job is None:
        run_job = _run_single_correction_job
    n_epoch_jobs = len(specs)
    job_specs = [
        {**spec, 'concurrent_correction_jobs': max(1, int(outer_epoch))}
        for spec in specs
    ]
    if outer_epoch <= 1:
        for spec in job_specs:
            run_job(spec)
    else:
        print(
            f'{phase_label}: {n_epoch_jobs} epoch job(s), '
            f'outer={outer_epoch} parallel, {workers_per_epoch} worker(s)/job'
        )
        with ProcessPoolExecutor(max_workers=outer_epoch) as pool:
            futures = [pool.submit(run_job, spec) for spec in job_specs]
            for f in futures:
                f.result()


def _auto_correction_workers() -> int:
    """Return the CPU ceiling; RAM is checked against real correction data.

    AMADEUS_CORRECTION_WORKERS overrides the ceiling.  The default counts
    logical CPUs, but per-ID correction work is CPU-bound Python over NumPy and
    OpenCV, where SMT siblings share the execution units that work depends on
    and Windows process spawn adds a fixed cost per worker -- so the fastest
    setting is often nearer the physical core count.  Which one wins is a
    property of the machine and the data, so this is left measurable rather
    than guessed.
    """
    logical_cpu = os.cpu_count() or 1
    override = os.environ.get('AMADEUS_CORRECTION_WORKERS', '').strip()
    if override:
        try:
            requested = int(override)
        except ValueError:
            print(
                f'correction: ignoring non-integer '
                f'AMADEUS_CORRECTION_WORKERS={override!r}.',
                flush=True,
            )
        else:
            return max(1, min(requested, logical_cpu))
    return max(1, min(12, logical_cpu - 1))


def _estimated_correction_job_bytes(spec: dict) -> int:
    """Resident RAM one correction job needs for its own tracking buffers.

    This exists only to decide how many jobs may run *concurrently*, so it
    covers exactly what a job holds for its whole lifetime: one full set of
    tracking buffers plus the process that loads them.

    Inner ID workers are deliberately not modelled here.  Their payloads are
    single-ID views -- with a few hundred IDs, the bounded in-flight set of a
    stage's pool is a low single-digit percentage of one full buffer set -- and
    their real sizes are measured against real free RAM at every stage by
    _memory_safe_process_workers, which is both far more accurate and correctly
    timed.  Guessing that cost here instead, from a pickle size and before any
    buffer is loaded, previously added a second whole buffer set to the
    estimate and so forced inner=1 on machines with ample free RAM, collapsing
    every per-ID stage to a single core.
    """
    fwd_path = _artifact_path(
        spec['out_dir'],
        BUFFERS_PICKLE_NAME,
        FWD_ARTIFACT_SUFFIX,
    )
    try:
        pickle_bytes = max(0, int(os.path.getsize(fwd_path)))
    except OSError:
        pickle_bytes = 0

    buffers_bytes = int(pickle_bytes * _TRACKING_PICKLE_MEMORY_FACTOR)
    return buffers_bytes + _CORRECTION_PROCESS_BASE_BYTES


def _correction_worker_budget(
    total_workers: int,
    specs: list[dict],
) -> tuple[int, int]:
    """Split outer/inner workers without admitting more jobs than RAM fits.

    Only the outer count is decided against RAM here, because only it changes
    how many full tracking buffer sets are resident at once.  The returned
    inner count is a CPU ceiling, not a promise: each stage passes it to
    _memory_safe_process_workers, which lowers it using measured per-ID
    payloads and the RAM actually free once the buffers are loaded.
    """
    jobs = max(1, len(specs))
    total = max(1, int(total_workers))
    requested_outer, requested_inner = _worker_budget(total, jobs)
    budget = int(
        psutil.virtual_memory().available
        * _CORRECTION_RAM_BUDGET_FRACTION
    )
    estimates = sorted(
        (_estimated_correction_job_bytes(spec) for spec in specs),
        reverse=True,
    )

    for outer in range(requested_outer, 0, -1):
        if sum(estimates[:outer]) > budget:
            continue
        inner = max(1, total // outer)
        if (outer, inner) != (requested_outer, requested_inner):
            largest_gib = (estimates[0] if estimates else 0) / (1024.0 ** 3)
            print(
                f'correction: worker layout reduced '
                f'outer={requested_outer}, inner={requested_inner} -> '
                f'outer={outer}, inner={inner} for available RAM '
                f'(largest estimated job={largest_gib:.2f} GiB).',
                flush=True,
            )
        return outer, inner

    # Even one job's buffers exceed the budget.  Running that job is still the
    # only way forward and it must load those buffers regardless, so throttling
    # its ID workers here would cost the parallelism without saving the memory
    # that caused the shortfall.  The per-stage check does that job properly,
    # against measured payloads and the RAM free at that moment.
    largest_gib = (estimates[0] if estimates else 0) / (1024.0 ** 3)
    print(
        f'correction: available RAM does not cover the estimate for one job '
        f'({largest_gib:.2f} GiB); running one job at a time and letting each '
        f'stage size its own ID workers.',
        flush=True,
    )
    return 1, max(1, total)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python refinement.py config.yaml')
    cfg = load_config(sys.argv[1])
    if cfg.get('skip_refinement', False):
        print('skip_refinement is True; exiting.')
        return

    analysis       = cfg.get('analysis', {}) or {}
    strong_iou     = float(analysis.get('MATCH_IOU', 0.50))
    strong_dir_deg = float(analysis.get('MATCH_ANGLE',   90.0))
    max_age        = int(analysis.get('MAX_AGE', 10))
    max_axis_error_deg = float(analysis.get('MAX_AXIS_ERR', 45.0))
    workers_cfg = resolve_num_workers(cfg, 'correction', default=None)
    if workers_cfg is None:
        workers_cfg = resolve_num_workers(cfg, 'id_correction', default=None)
    num_workers = (
        _auto_correction_workers()
        if workers_cfg is None
        else max(1, int(workers_cfg))
    )
    worker_mode = 'auto' if workers_cfg is None else 'configured'
    print(f'correction: workers={num_workers} ({worker_mode})')
    print('correction: RAM worker caps are adjusted per correction data.')
    embedding_cfg  = dict(cfg.get('EMBEDDING', {}) or {})
    embedding_enabled = _cfg_bool(embedding_cfg.get('ENABLE', False))

    session_path  = str(cfg['SESSION_PATH'])
    num_objects   = int(cfg['NUM_OBJECTS'])
    video_path_in = str(cfg['TRACKING_VIDEO_PATH'])
    background_path = str(cfg.get('BACKGROUND_PATH', '')).strip()
    master_seed = normalize_seed(cfg.get('RANDOM_SEED', 0))
    print(f'[SEED] correction master={master_seed}')

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
        video_names = [
            os.path.splitext(f)[0]
            for f in os.listdir(video_path_in)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        ]
        video_paths = {
            os.path.splitext(f)[0]: os.path.join(video_path_in, f)
            for f in os.listdir(video_path_in)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        }
    else:
        video_names = [os.path.splitext(os.path.basename(video_path_in))[0]]
        video_paths = {video_names[0]: video_path_in}
    if not video_names:
        raise RuntimeError(f'No video files found at: {video_path_in}')

    specs = []
    for video_name in video_names:
        video_path = video_paths.get(video_name, '')
        flip_frame_threshold, flip_duration_sec, source_fps = _resolve_flip_frame_threshold(
            analysis, video_path,
        )
        for job in tracking_jobs:
            run_name = str(job['run_name'])
            out_dir = build_tracking_out_dir(session_path, model_name, dataset_name, run_name, video_name)
            specs.append({
                'out_dir': out_dir,
                'session_path': session_path,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'run_name': run_name,
                'video_name': video_name,
                'video_path': video_path,
                'background_path': background_path,
                'num_objects': num_objects,
                'strong_iou': strong_iou,
                'strong_dir_deg': strong_dir_deg,
                'max_age': max_age,
                'flip_frame_threshold': flip_frame_threshold,
                'flip_duration_sec': flip_duration_sec,
                'source_fps': source_fps,
                'max_axis_error_deg': max_axis_error_deg,
                'num_workers': num_workers,
                'embedding': embedding_cfg,
                'embedding_enabled': embedding_enabled,
                'random_seed': derive_seed(master_seed, 'correction', video_name, run_name),
                'embedding_seed': derive_seed(master_seed, 'correction_embedding', video_name, run_name),
            })

    if not specs:
        return

    if embedding_enabled:
        # Embedding can use CUDA or CPU; keep it single-job to avoid concurrent
        # contrastive training jobs exhausting VRAM/RAM. No pre-embedding filled
        # phase is run: geometry correction happens after embedding.
        embedding_specs = [
            {**spec, 'phase': 'embedding', 'num_workers': num_workers}
            for spec in specs
        ]
        _run_correction_specs(
            embedding_specs,
            1,
            num_workers,
            'correction embedding-first phase',
        )
    else:
        outer_epoch, workers_per_epoch = _correction_worker_budget(
            num_workers,
            specs,
        )
        for spec in specs:
            spec['num_workers'] = workers_per_epoch
        _run_correction_specs(specs, outer_epoch, workers_per_epoch, 'correction')


if __name__ == '__main__':
    from without_direction_estimation import prepare_config
    if len(sys.argv) > 1:
        sys.argv[1] = prepare_config(sys.argv[1])
    main()
