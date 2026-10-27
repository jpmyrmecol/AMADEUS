# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Compare original and refined trajectories without guessing nearby identities.

The GUI's endpoint conversion rounds coordinates to three decimal places and
AMADEUS result CSVs are written to six decimal places.  The 0.002-pixel tolerance
allows those numeric round trips.  It is not a distance gate for tracking.
Both inputs must describe the same video frames and the same set of ID slots.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class IdentitySwitch:
    """A changed correspondence, with inclusive bounds on its first frame."""

    source_tid: int
    previous_tid: int
    loaded_tid: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class IdentityInterval:
    """An inclusive interval whose destination ID could not be established."""

    source_tid: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class IdentityMapping:
    """Destination IDs indexed by [frame, source ID], with -1 for unknown IDs."""

    source_to_loaded: np.ndarray
    switches: tuple[IdentitySwitch, ...]
    unresolved_intervals: tuple[IdentityInterval, ...]


def _wide_points(data: pd.DataFrame, name: str) -> np.ndarray:
    if data.empty or data.columns[0] not in {"frame", "position"}:
        raise ValueError(f"{name} must contain sequential frames and front/rear coordinates.")
    count = (len(data.columns) - 1) // 4
    expected = [data.columns[0]]
    for point_id in range(2 * count):
        expected.extend((f"x{point_id}", f"y{point_id}"))
    if count == 0 or list(data.columns) != expected:
        raise ValueError(f"{name} must have x0,y0,x1,y1,... columns for complete ID slots.")
    frames = data.iloc[:, 0].to_numpy(dtype=float)
    if not np.array_equal(frames, np.arange(len(data))):
        raise ValueError(f"{name} frame numbers must be consecutive from 0.")
    points = data.iloc[:, 1:].to_numpy(dtype=float).reshape(len(data), count, 2, 2)
    if np.isinf(points).any():
        raise ValueError(f"{name} contains infinite coordinates.")
    return points


def _assign_unique(candidates: dict[int, set[int]], mapping: np.ndarray) -> None:
    """Accept only edges unique at both ends of the remaining bipartite graph."""
    used = set(mapping[mapping >= 0].tolist())
    remaining = {
        source: targets - used
        for source, targets in candidates.items()
        if mapping[source] < 0
    }
    while remaining:
        owners: dict[int, list[int]] = {}
        for source, targets in remaining.items():
            for target in targets:
                owners.setdefault(target, []).append(source)
        accepted = {
            source: next(iter(targets))
            for source, targets in remaining.items()
            if len(targets) == 1 and len(owners[next(iter(targets))]) == 1
        }
        if not accepted:
            break
        used_now = set(accepted.values())
        for source, target in accepted.items():
            mapping[source] = target
        remaining = {
            source: targets - used_now
            for source, targets in remaining.items()
            if source not in accepted
        }


def _frame_mapping(
    sources: Sequence[np.ndarray], loaded: np.ndarray, tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(loaded)
    mapping = np.full(count, -1, dtype=np.int32)
    conflicting = np.zeros(count, dtype=bool)
    loaded_points = loaded.reshape(-1, 2)
    loaded_indices = np.flatnonzero(np.isfinite(loaded_points).all(axis=1))
    complete: dict[int, set[int]] = {}
    partial: dict[int, set[int]] = {}
    if len(loaded_indices):
        tree = cKDTree(loaded_points[loaded_indices])
        for source in sources:
            source_points = source.reshape(-1, 2)
            source_indices = np.flatnonzero(np.isfinite(source_points).all(axis=1))
            neighbors = tree.query_ball_point(source_points[source_indices], tolerance, p=np.inf)
            # Pair both endpoints within one representation before merging its
            # evidence. Endpoints from different variants cannot form an arrow.
            matches: dict[int, dict[int, int]] = {}
            for point_index, found in zip(source_indices.tolist(), neighbors):
                source_id, source_end = divmod(point_index, 2)
                for index in found:
                    target_id, target_end = divmod(int(loaded_indices[index]), 2)
                    pairings = matches.setdefault(source_id, {})
                    pairings[target_id] = pairings.get(target_id, 0) | (1 << (2 * source_end + target_end))
            for source_id, pairings in matches.items():
                complete.setdefault(source_id, set()).update(
                    target_id for target_id, mask in pairings.items()
                    if mask & 9 == 9 or mask & 6 == 6
                )
                partial.setdefault(source_id, set()).update(pairings)
        _assign_unique(complete, mapping)
        # A manually moved endpoint can still leave the other endpoint intact.
        _assign_unique(partial, mapping)
        if len(sources) > 1:
            owners: dict[int, list[int]] = {}
            for source_id, targets in complete.items():
                if len(targets) > 1:
                    conflicting[source_id] = True
                for target in targets:
                    owners.setdefault(target, []).append(source_id)
            for source_ids in owners.values():
                if len(source_ids) > 1:
                    conflicting[source_ids] = True
    unresolved = np.flatnonzero(mapping < 0)
    if len(unresolved) == 1:
        # Under the required same-ID-set contract, the remaining slot has only
        # one possible destination even if both endpoints were moved or absent.
        available = np.setdiff1d(np.arange(count), mapping[mapping >= 0])
        mapping[unresolved[0]] = available[0]
    return mapping, conflicting


def _bridge_matching_neighbors(mapping: np.ndarray, conflicting: np.ndarray) -> None:
    """Bridge only unknown runs bounded by the same established destination."""
    proposed = mapping.copy()
    for source_id in range(mapping.shape[1]):
        known = np.flatnonzero(mapping[:, source_id] >= 0)
        if len(known) < 2:
            continue
        same = mapping[known[:-1], source_id] == mapping[known[1:], source_id]
        for gap in np.flatnonzero(same & (np.diff(known) > 1)):
            left, right = int(known[gap]), int(known[gap + 1])
            allowed = ~conflicting[left + 1:right, source_id]
            proposed[left + 1:right, source_id][allowed] = mapping[left, source_id]
    # Simultaneous bridges must not claim a destination already established
    # for another source, or create a collision with another proposed bridge.
    for frame in np.flatnonzero(np.any(proposed != mapping, axis=1)):
        row = proposed[frame]
        counts = np.bincount(row[row >= 0], minlength=mapping.shape[1])
        bridge = (mapping[frame] < 0) & (row >= 0)
        for source_id in np.flatnonzero(bridge):
            if counts[row[source_id]] == 1:
                mapping[frame, source_id] = row[source_id]


def _mapping_intervals(mapping: np.ndarray) -> tuple[tuple[IdentitySwitch, ...], tuple[IdentityInterval, ...]]:
    switches = []
    unresolved = []
    for source_id in range(mapping.shape[1]):
        column = mapping[:, source_id]
        known = np.flatnonzero(column >= 0)
        previous_tid = source_id
        previous_frame = -1
        for frame in known.tolist():
            target = int(column[frame])
            if target != previous_tid:
                switches.append(IdentitySwitch(source_id, previous_tid, target, previous_frame + 1, frame))
            previous_tid = target
            previous_frame = frame
        unknown = column < 0
        edges = np.diff(np.r_[False, unknown, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        unresolved.extend(
            IdentityInterval(source_id, int(start), int(stop - 1))
            for start, stop in zip(starts, stops)
        )
    switches.sort(key=lambda item: (item.start_frame, item.end_frame, item.source_tid))
    unresolved.sort(key=lambda item: (item.start_frame, item.end_frame, item.source_tid))
    return tuple(switches), tuple(unresolved)


def infer_refinement_identity(
    source_wide: pd.DataFrame,
    loaded_wide: pd.DataFrame,
    *,
    coordinate_tolerance: float = 0.002,
    source_variants: Sequence[pd.DataFrame] = (),
) -> IdentityMapping:
    """Infer source-ID to loaded-ID correspondence at every video frame.

    Exact endpoint matches, allowing numeric serialization error and front/rear
    reversal, provide the evidence.  Complete arrows take precedence over a
    single unchanged endpoint.  No nearest-neighbor assignment is performed.
    Additional source representations describe the same original ID slots.
    Their match candidates are combined before assigning IDs. Conflicting
    complete-arrow evidence stays unknown and is not bridged across time.
    Unknown gaps are bridged only when both adjacent established mappings agree
    and the one-to-one constraint is preserved.  A gap between different mappings
    stays unknown and widens the reported switch boundary interval.  Bridging
    assumes no temporary, unobserved switch occurred entirely inside that gap.
    """
    if not np.isfinite(coordinate_tolerance) or coordinate_tolerance <= 0:
        raise ValueError("Coordinate tolerance must be finite and positive.")
    source = _wide_points(source_wide, "Original CSV")
    loaded = _wide_points(loaded_wide, "Loaded CSV")
    if source.shape != loaded.shape:
        raise ValueError("Original and loaded CSVs must have the same frame count and ID count.")
    sources = [source]
    for index, variant in enumerate(source_variants):
        points = _wide_points(variant, f"Original CSV variant {index + 1}")
        if points.shape != source.shape:
            raise ValueError("All original CSV variants must have the same frames and ID count.")
        sources.append(points)
    mapping = np.empty(source.shape[:2], dtype=np.int32)
    conflicting = np.zeros(source.shape[:2], dtype=bool)
    for frame in range(len(source)):
        mapping[frame], conflicting[frame] = _frame_mapping(
            [points[frame] for points in sources], loaded[frame], float(coordinate_tolerance),
        )
    _bridge_matching_neighbors(mapping, conflicting)
    switches, unresolved = _mapping_intervals(mapping)
    return IdentityMapping(mapping, switches, unresolved)
