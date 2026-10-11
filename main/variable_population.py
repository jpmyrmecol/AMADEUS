# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Population policy, correction guards and colors (no deep-learning imports)."""
import csv
import heapq
import math
from pathlib import Path
import numpy as np


def _positive_integer(value, name, minimum=1):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be an integer >= {minimum}.') from None
    if (isinstance(value, (bool, np.bool_)) or not math.isfinite(numeric)
            or numeric != int(numeric) or numeric < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(numeric)


def _finite_number(value, minimum=0.0):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric >= minimum else None


def _sample_obbs(frame_data, limit=128):
    """Bound calibration work and exclude malformed boxes before measuring."""
    try:
        obbs = np.asarray(frame_data[0], dtype=float).reshape(-1, 4, 2)
    except (IndexError, TypeError, ValueError):
        return np.empty((0, 4, 2)), np.empty(0)
    if len(obbs) > limit:
        scores = np.asarray(frame_data[1], dtype=float).reshape(-1)
        indices = (np.argsort(-scores, kind='stable')[:limit]
                   if len(scores) == len(obbs) else np.arange(limit))
        obbs = obbs[indices]
    sides = np.linalg.norm(np.roll(obbs, -1, axis=1) - obbs, axis=2)
    valid = np.isfinite(obbs).all(axis=(1, 2)) & (sides > 0).all(axis=1)
    obbs, sides = obbs[valid], sides[valid]
    major = np.maximum((sides[:, 0] + sides[:, 2]) * .5,
                       (sides[:, 1] + sides[:, 3]) * .5)
    return obbs, major


def _detection_calibration(detection_frames, body_length=None):
    """Estimate size and motion from <=256 neighboring frame pairs.

    Mutual nearest centers reduce accidental matches. Motion only uses boxes
    within a factor of two of the robust body scale and moves <=half a body
    per frame, so isolated large false OBBs cannot inflate the edge margin.
    """
    frames = sorted(detection_frames or {})
    if not frames:
        return body_length, None, 0, 0
    selected = np.unique(np.linspace(0, len(frames) - 1,
                                    min(256, len(frames)), dtype=int))
    samples = {}
    lengths = []
    for index in selected:
        frame = frames[int(index)]
        samples[frame] = _sample_obbs(detection_frames[frame])
        lengths.extend(samples[frame][1].tolist())
    if body_length is None and lengths:
        body_length = float(np.median(lengths))
    if body_length is None or body_length <= 0:
        return None, None, len(selected), 0
    distances = []
    for index in selected:
        index = int(index)
        if index + 1 >= len(frames) or frames[index + 1] - frames[index] != 1:
            continue
        first, second = frames[index], frames[index + 1]
        if second not in samples:
            samples[second] = _sample_obbs(detection_frames[second])
        centers = []
        for frame in (first, second):
            obbs, major = samples[frame]
            usable = (major >= body_length * .5) & (major <= body_length * 2)
            centers.append(obbs[usable].mean(axis=1))
        if not len(centers[0]) or not len(centers[1]):
            continue
        distance = np.linalg.norm(centers[0][:, None] - centers[1][None], axis=2)
        nearest_next = np.argmin(distance, axis=1)
        nearest_prev = np.argmin(distance, axis=0)
        for previous, following in enumerate(nearest_next):
            if nearest_prev[following] == previous:
                move = float(distance[previous, following])
                if move <= body_length * .5:
                    distances.append(move)
    speed = float(np.percentile(distances, 90)) if len(distances) >= 10 else None
    return body_length, speed, len(selected), len(distances)


class PopulationPolicy:
    """Use the last *observed* OBB to distinguish edges from the interior.

    The caller holds tentative observations until their consecutive birth
    threshold is reached, freezes that threshold at the first observation,
    and never feeds an unobserved prediction to ``max_age_for_track``.
    Anywhere mode retains the previous immediate birth/uniform expiry rules.
    """

    def __init__(self, mode='edge', frame_size=None, initial_stats=None,
                 detection_frames=None, edge_confirm_frames=2,
                 interior_confirm_frames=10, max_age=10, interior_max_age=None):
        self.mode = str(mode).strip().lower()
        if self.mode not in ('edge', 'anywhere'):
            raise ValueError('VARIABLE_POPULATION_MODE must be edge or anywhere.')
        self.max_age = _positive_integer(max_age, 'MAX_AGE')
        if self.mode == 'anywhere':
            # Edge-only settings may be stale or invalid in an old config;
            # they have no effect on the original population behavior.
            self.edge_confirm_frames = self.interior_confirm_frames = 1
            self.interior_max_age = self.max_age
        else:
            self.edge_confirm_frames = _positive_integer(
                edge_confirm_frames, 'VARIABLE_EDGE_CONFIRM_FRAMES', minimum=2)
            self.interior_confirm_frames = _positive_integer(
                interior_confirm_frames, 'VARIABLE_INTERIOR_CONFIRM_FRAMES', minimum=2)
            if self.interior_confirm_frames <= self.edge_confirm_frames:
                raise ValueError('VARIABLE_INTERIOR_CONFIRM_FRAMES must be > VARIABLE_EDGE_CONFIRM_FRAMES.')
            self.interior_max_age = _positive_integer(
                max(30, self.max_age) if interior_max_age is None else interior_max_age,
                'VARIABLE_INTERIOR_MAX_AGE')
            if self.interior_max_age < self.max_age:
                raise ValueError('VARIABLE_INTERIOR_MAX_AGE must be >= MAX_AGE.')
        self.frame_size = None
        if frame_size is not None:
            try:
                width, height = frame_size
            except (TypeError, ValueError):
                raise ValueError('frame_size must contain video width and height.') from None
            self.frame_size = (_positive_integer(width, 'video frame width'),
                               _positive_integer(height, 'video frame height'))
        if self.mode == 'edge' and self.frame_size is None:
            raise ValueError('Edge population mode requires actual video width and height.')
        stats = initial_stats or {}
        body = _finite_number(stats.get('individual_distance_px'), minimum=1e-6)
        self.body_length_source = 'initial_tracking' if body is not None else 'detections'
        ratio = _finite_number(stats.get('disp_ratio_p90'))
        need_detection_calibration = body is None or ratio is None
        sampled_body, sampled_speed, sample_count, motion_count = (
            _detection_calibration(detection_frames, body)
            if self.mode == 'edge' and need_detection_calibration
            else (body, None, 0, 0))
        if body is None:
            body = sampled_body
        if body is None:
            body = .025 * min(self.frame_size) if self.frame_size else 1.0
            self.body_length_source = 'frame_scale'
        self.body_length_px = float(body)
        if ratio is not None:
            self.speed_px = ratio * self.body_length_px
            self.motion_source = 'initial_tracking'
        elif sampled_speed is not None:
            self.speed_px = sampled_speed
            self.motion_source = 'detections'
        else:
            self.speed_px = 0.0
            self.motion_source = 'unavailable'
        self.calibration_frame_count = int(sample_count)
        self.calibration_motion_count = int(motion_count)
        self.edge_margin_px = (min(.5 * self.body_length_px
                                  + self.speed_px * self.edge_confirm_frames,
                                  .15 * min(self.frame_size))
                               if self.frame_size else 0.0)

    def is_edge(self, obb):
        if self.frame_size is None:
            return False
        try:
            points = np.asarray(obb, dtype=float).reshape(4, 2)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(points).all():
            return False
        width, height = self.frame_size
        center = points.mean(axis=0)
        # Keep normal/partially clipped OBB geometry, but do not let a giant
        # false box whose center is interior reach a boundary by its size alone.
        half_extent = np.minimum(np.ptp(points, axis=0) * .5, self.body_length_px)
        boundary_distance = min(float(center[0] - half_extent[0]),
                                float(center[1] - half_extent[1]),
                                width - float(center[0] + half_extent[0]),
                                height - float(center[1] + half_extent[1]))
        return boundary_distance <= self.edge_margin_px

    def threshold_for_birth(self, obb):
        if self.mode == 'anywhere':
            return 1
        return self.edge_confirm_frames if self.is_edge(obb) else self.interior_confirm_frames

    def max_age_for_track(self, last_obb):
        if self.mode == 'anywhere' or self.is_edge(last_obb):
            return self.max_age
        return self.interior_max_age

    def summary(self):
        return dict(mode=self.mode, frame_size=self.frame_size,
                    edge_confirm_frames=self.edge_confirm_frames,
                    interior_confirm_frames=self.interior_confirm_frames,
                    edge_max_age=self.max_age,
                    interior_max_age=self.max_age if self.mode == 'anywhere' else self.interior_max_age,
                    edge_margin_px=float(self.edge_margin_px),
                    body_length_px=self.body_length_px, speed_px=float(self.speed_px),
                    body_length_source=self.body_length_source, motion_source=self.motion_source,
                    calibration_frame_count=self.calibration_frame_count,
                    calibration_motion_count=self.calibration_motion_count)


def load_population_policy(config, session_root=None, frame_size=None,
                           detection_frames=None, video_path=None):
    """Load policy settings and optional initial measurements for one video.

    Caller-supplied dimensions are authoritative. When needed, probe the real
    video rather than reuse dimensions from another video's initial tracking.
    """
    analysis = config.get('analysis', config) or {}
    mode = analysis.get('VARIABLE_POPULATION_MODE', 'edge')
    session_root = session_root or config.get('SESSION_PATH')
    stats = {}
    if session_root:
        stats_path = Path(session_root) / 'initial_tracking' / 'tracking_stats.csv'
        if stats_path.is_file():
            with stats_path.open('r', encoding='utf-8-sig', newline='') as file:
                for row in csv.DictReader(file):
                    if row.get('metric'):
                        stats[str(row['metric']).strip()] = str(row.get('value', '')).strip()
    if str(mode).strip().lower() == 'edge' and frame_size is None:
        video_path = video_path or config.get('TRACKING_VIDEO_PATH')
        if video_path:
            video = Path(video_path)
            if not video.is_absolute() and session_root:
                video = Path(session_root) / video
            from video_frame_count import read_video_frame_info
            info = read_video_frame_info(str(video))
            frame_size = (info.width, info.height)
    return PopulationPolicy(
        mode=mode, frame_size=frame_size, initial_stats=stats,
        detection_frames=detection_frames,
        edge_confirm_frames=analysis.get('VARIABLE_EDGE_CONFIRM_FRAMES', 2),
        interior_confirm_frames=analysis.get('VARIABLE_INTERIOR_CONFIRM_FRAMES', 10),
        max_age=analysis.get('MAX_AGE', 10),
        interior_max_age=analysis.get('VARIABLE_INTERIOR_MAX_AGE'))


def observed_frames(buffers, tid):
    return {int(f) for f, obb in buffers['obb_buf'].get(tid, {}).items()
            if len(obb) == 8 and np.isfinite(obb).all()}


def closed_episode(buffers, episode):
    start, end = int(episode['start']), int(episode['end'])
    roster = set(episode['roster'])
    # Require every participant to remain observed across the whole episode
    # and its two anchors. Birth/death/reappearance ambiguity means abstention.
    required = set(range(start-1, end+2))
    for tid in roster:
        if not required <= observed_frames(buffers, tid):
            return False
    # Do not infer a conserved local group across a population transition.
    for tid in buffers['obb_buf']:
        frames = observed_frames(buffers, tid)
        if frames and (start-1 <= min(frames) <= end+1 or start-1 <= max(frames) <= end+1):
            return False
    return True


def safe_swap(buffers, mapping, start, end):
    if set(mapping) != set(mapping.values()):
        return False
    masks = {tid: {f for f in observed_frames(buffers, tid) if start <= f <= end}
             for tid in mapping}
    return all(masks[tid] == masks[other] for tid, other in mapping.items())


def allocate_color_slots(present_by_frame):
    """Use exactly peak simultaneous occupancy slots; retain colors when possible.

    A color is released while an ID is absent. A returning live ID prefers its
    previous color if free; otherwise its numeric label remains authoritative.
    """
    peak = max((len(set(ids)) for ids in present_by_frame.values()), default=0)
    free = list(range(peak))
    active, previous, result = {}, {}, {}
    for frame in sorted(present_by_frame):
        ids = set(present_by_frame[frame])
        for tid in list(active):
            if tid not in ids:
                heapq.heappush(free, active.pop(tid))
        for tid in sorted(ids):
            if tid in active:
                continue
            preferred = previous.get(tid)
            if preferred in free:
                free.remove(preferred)
                heapq.heapify(free)
                slot = preferred
            else:
                slot = heapq.heappop(free)
            active[tid] = previous[tid] = slot
        result[frame] = dict(active)
    return peak, result
