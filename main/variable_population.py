# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Population-independent guards and color allocation (no deep-learning imports)."""
import heapq
import numpy as np


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
