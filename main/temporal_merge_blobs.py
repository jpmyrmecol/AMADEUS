# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Detect blobs that contain several animals from consecutive-frame evidence.

Area thresholds cannot see two animals that overlap so much that their blob
has a normal area. Over time the merge still shows: two separate animals
flow into one blob, and the blob later splits into two separate animals.

Graph (built while scanning, no contours kept)
    Nodes are the final classified blobs of each frame. Every blob is
    rasterized into a label map; pixel co-occurrence between consecutive
    label maps gives the overlap of every pair of blobs. All positive
    overlaps are kept, so no early one-to-one association can hide a
    many-to-one merge. Containment is measured on both sides of an edge:
    ``f = overlap / parent area`` (how much of the parent flowed into the
    child) and ``g = overlap / child area``. Containment, not IoU, is used
    because a merged blob has a low IoU with each parent although each
    parent lies almost entirely inside it.

Evaluation (re-runnable on the stored graph)
    Dominant links (a parent's best child with f >= tau, or a child's best
    parent with g >= tau) form transition components between frames: 1:1,
    merge, split, complex, birth, death. Maximal 1:1 sequences are *chains*.
    tau is calibrated from unambiguous 1:1 pairs (0.8 x the 10th percentile
    of their self-containment, clipped to [0.3, 0.6]) unless set manually.
    Evidence is a lower bound on the number of animals in a blob, held as a
    set of *atoms* (chains of confirmed, mutually independent animals), and
    is propagated forward from merges and backward from splits:

    * a merge sums only arms that are confirmed (k stable, normal observations
      close to the event and no ambiguous flow) and pairwise independent:
      both arms must coexist as separate, normal, substantial blobs for k frames.
      No minimum OBB gap is required. Recent common split ancestry prevents
      fragments of the same animal from being counted twice.
    * arms descending from the same recent split (within a short fixed
      fragmentation window) are one group and contribute the count before that split,
      never a sum, so a fragmenting animal that rejoins keeps its count.
    * a merge followed by a split, and complex exchanges, are bridged by
      a unique one-to-one match of substantial blobs, compared by
      their median areas next to the contact (a body touching another one
      already looks larger in the contact frames). The matched blob keeps the
      arm's identity (its history counts towards confirmation and
      independence) and its count.
    * otherwise a known count is shared by mass: a blob that takes a share s
      of the mass leaving a split (or, backward, entering a merge) holds
      floor(n * s + share_rounding) of the n known animals, as disjoint atoms. Two
      animals parting stay one each; three parting as a pair and a single
      leave the pair with a count of 2.
    * along a chain the count is kept (occlusion hides area, not animals)
      unless a visible exit explains a loss: a substantial birth or death
      nearby, part of the mass flowing to another blob, or an area below
      0.75 of the largest member. Image edges, manual and Result-generated
      regions, Result single-animal rescues and ROI / Additional-Outlier
      frame-range switches reset counts.

Decisions (only blobs that are normal after classification are flagged)
    High    forward and backward evidence >= 2 at the blob; excluded only when
            the overlap calibration is reliable (>= 100 unambiguous pairs,
            self-containment P10 >= 0.5, continuation rate >= 0.85).
    Medium  one-sided evidence >= 2 that has not passed an ROI edge and
            preserves the configured minimum fraction of the chain's
            reference mass. No arbitrary time horizon is imposed. Excluded
            only in "include one-sided" mode with reliable calibration.
    Other evidence (after an ROI edge or a mass loss, conflicts and failed
    events) is reported as uncertain and never excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import itertools
import math
import os
import threading

import cv2
import numpy as np


# Node flag bits (uint8).
FLAG_NORMAL = 1          # not an outlier after classification (is_crossing False)
FLAG_UNKNOWN = 2         # manual outlier or Result generated/missing region
FLAG_BORDER = 4          # touches the image edge
FLAG_RESULT_SINGLE = 8   # area outlier kept as one animal by an imported Result OBB
FLAG_LOW_ASPECT = 16     # classification-independent low OBB aspect outlier
FLAG_MANUAL = 32         # manual (Additional Outlier) region
FLAG_ROI_EDGE = 64       # touches the outside of the frame's active ROI


@dataclass
class FrameNodes:
    """Compact per-frame blob data; pixels are kept only until the next frame."""
    frame: int
    area: np.ndarray           # int32 raster pixels owned by each blob
    contour_area: np.ndarray   # float32 area used by Outlier Extraction
    cx: np.ndarray             # float32 contour centroid
    cy: np.ndarray             # float32
    length: np.ndarray         # float32 OBB major axis
    width: np.ndarray          # float32 OBB minor axis
    theta: np.ndarray          # float32 OBB major-axis angle (radians)
    flags: np.ndarray          # uint8
    flat: np.ndarray           # int32 flat pixel indices, grouped by blob
    digest: int
    context: int


_local = threading.local()


def _label_buffer(shape: tuple[int, int]) -> np.ndarray:
    buffer = getattr(_local, "labels", None)
    if buffer is None or buffer.shape != shape:
        buffer = np.zeros(shape, dtype=np.int32)
        _local.labels = buffer
    return buffer


def roi_border_pixels(roi_mask: "np.ndarray | None") -> "np.ndarray | None":
    """Flat bool array: pixels within one pixel of the outside of an ROI mask."""
    if roi_mask is None:
        return None
    outside = (roi_mask == 0).astype(np.uint8)
    if not outside.any():
        return None
    near = cv2.dilate(outside, np.ones((3, 3), np.uint8))
    return near.reshape(-1).astype(bool)


def _blob_flags(blob) -> int:
    manual = bool(getattr(blob, "manual_outlier", False))
    generated = bool(getattr(blob, "result_generated", False) or getattr(blob, "result_missing", False))
    flags = 0
    # Temporal tracking precedes Outlier Extraction: statistical area/shape
    # exclusions must not erase an otherwise valid animal-history node.
    # Only manual and synthetic regions cannot seed a confirmed animal.
    if not (manual or generated):
        flags |= FLAG_NORMAL
    if manual or generated:
        flags |= FLAG_UNKNOWN
    if manual:
        flags |= FLAG_MANUAL
    if bool(getattr(blob, "result_single", False)):
        flags |= FLAG_RESULT_SINGLE
    if bool(getattr(blob, "low_obb_aspect_outlier", False)):
        flags |= FLAG_LOW_ASPECT
    return flags


def extract_frame_nodes(
    frame: int,
    blobs,
    image_shape: tuple,
    *,
    border_pixels: "np.ndarray | None" = None,
    digest: int = 0,
    context: int = 0,
) -> FrameNodes:
    """Rasterize one frame's final blobs into compact node data.

    Nodes are addressed by their position in ``blobs`` (not ``blob_index``,
    which is not unique after every classification path). Runs in worker
    threads. Real segmentation blobs own their pixels;
    manual and Result-generated regions only fill pixels no real blob owns,
    so every pixel carries at most one node. ``border_pixels`` marks pixels
    next to the outside of the frame's active ROI.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    count = len(blobs)
    labels = _label_buffer((height, width))
    flags = np.zeros(count, dtype=np.uint8)
    contour_area = np.zeros(count, dtype=np.float32)
    cx = np.zeros(count, dtype=np.float32)
    cy = np.zeros(count, dtype=np.float32)
    length = np.zeros(count, dtype=np.float32)
    minor = np.zeros(count, dtype=np.float32)
    theta = np.zeros(count, dtype=np.float32)
    boxes = []
    contours = []
    for index, blob in enumerate(blobs):
        contour = np.asarray(blob.contour, dtype=np.int32).reshape(-1, 1, 2)
        contours.append(contour)
        flags[index] = _blob_flags(blob)
        contour_area[index] = float(getattr(blob, "area", 0.0))
        cx[index] = float(blob.center_x)
        cy[index] = float(blob.center_y)
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
        if contour.shape[0] >= 3:
            (_c, (rw, rh), angle) = cv2.minAreaRect(contour.astype(np.float32))
            length[index], minor[index] = max(rw, rh), min(rw, rh)
            theta[index] = math.radians(angle if rw >= rh else angle + 90.0)
        else:
            length[index], minor[index] = max(w, h), min(w, h)
            theta[index] = 0.0 if w >= h else math.pi / 2
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            flags[index] |= FLAG_BORDER
    # Lower-priority regions first, so real blobs overwrite them.
    order = sorted(range(count), key=lambda i: not (flags[i] & FLAG_UNKNOWN))
    for index in order:
        cv2.drawContours(labels, [contours[index]], -1, index + 1, cv2.FILLED)
    area = np.zeros(count, dtype=np.int32)
    pieces = []
    clipped = []
    for index, (x, y, w, h) in enumerate(boxes):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x0 >= x1 or y0 >= y1:
            pieces.append(np.zeros(0, dtype=np.int32))
            continue
        clipped.append((x0, y0, x1, y1))
        ys, xs = np.nonzero(labels[y0:y1, x0:x1] == index + 1)
        pieces.append((ys.astype(np.int32) + y0) * width + (xs.astype(np.int32) + x0))
        area[index] = pieces[-1].size
    for x0, y0, x1, y1 in clipped:
        labels[y0:y1, x0:x1] = 0
    flat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)
    if border_pixels is not None and flat.size:
        hits = border_pixels[flat]
        start = 0
        for index, size in enumerate(area):
            if size and hits[start:start + size].any():
                flags[index] |= FLAG_ROI_EDGE
            start += int(size)
    return FrameNodes(
        frame=int(frame), area=area, contour_area=contour_area, cx=cx, cy=cy,
        length=length, width=minor, theta=theta, flags=flags,
        flat=flat.astype(np.int32, copy=False), digest=int(digest), context=int(context),
    )


_NODE_COLUMNS = ("area", "contour_area", "cx", "cy", "length", "width", "theta", "flags")


class TemporalMergeGraphBuilder:
    """Streaming overlap graph over consecutive frames.

    Only the previous frame's label map is kept. Overlaps are counted at the
    current foreground pixels, so the cost is proportional to foreground area,
    never to the full frame. Every positive overlap is stored: no early 1:1
    association removes a many-to-one candidate. Labels are int32, so the
    number of blobs per frame cannot overflow them.
    """

    def __init__(self, image_shape: tuple):
        self.height, self.width = int(image_shape[0]), int(image_shape[1])
        self._prev_map = np.zeros(self.height * self.width, dtype=np.int32)
        self._prev_flat: "np.ndarray | None" = None
        self._prev_frame: "int | None" = None
        self._prev_context: "int | None" = None
        self._prev_offset = 0
        self._prev_count = 0
        self._first_frame: "int | None" = None
        self._node_count = 0
        self._columns: dict[str, list] = {name: [] for name in _NODE_COLUMNS}
        self._counts: list[int] = []
        self._digests: list[int] = []
        self._breaks: list[bool] = []
        self._edges: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add_frame(self, nodes: FrameNodes) -> None:
        fid = int(nodes.frame)
        if self._prev_frame is not None and fid != self._prev_frame + 1:
            raise ValueError("Temporal merge search requires consecutive video frames.")
        if self._first_frame is None:
            self._first_frame = fid
        count = int(nodes.area.size)
        offset = self._node_count
        linked = self._prev_frame is not None and nodes.context == self._prev_context
        if self._prev_frame is not None:
            self._breaks.append(not linked)
        labels = np.repeat(np.arange(1, count + 1, dtype=np.int32), nodes.area)
        if linked and nodes.flat.size and self._prev_count:
            previous = self._prev_map[nodes.flat]
            hit = previous > 0
            if hit.any():
                base = np.int64(count + 1)
                keys = previous[hit].astype(np.int64) * base + labels[hit]
                bins = int(self._prev_count + 1) * int(base)
                if bins <= (1 << 22):
                    counts = np.bincount(keys, minlength=bins)
                    keys = np.flatnonzero(counts)
                    counts = counts[keys]
                else:
                    keys, counts = np.unique(keys, return_counts=True)
                src = (keys // base - 1).astype(np.int32) + np.int32(self._prev_offset)
                dst = (keys % base - 1).astype(np.int32) + np.int32(offset)
                self._edges.append((src, dst, counts.astype(np.int32)))
        if self._prev_flat is not None:
            self._prev_map[self._prev_flat] = 0
        self._prev_map[nodes.flat] = labels
        self._prev_flat = nodes.flat
        self._prev_frame = fid
        self._prev_context = nodes.context
        self._prev_offset = offset
        self._prev_count = count
        self._node_count += count
        for name in _NODE_COLUMNS:
            self._columns[name].append(getattr(nodes, name))
        self._counts.append(count)
        self._digests.append(int(nodes.digest))

    def finish(self) -> "TemporalMergeGraph":
        def joined(name, dtype):
            parts = self._columns[name]
            return np.concatenate(parts).astype(dtype, copy=False) if parts else np.zeros(0, dtype)
        if self._edges:
            src = np.concatenate([e[0] for e in self._edges])
            dst = np.concatenate([e[1] for e in self._edges])
            overlap = np.concatenate([e[2] for e in self._edges])
        else:
            src = dst = overlap = np.zeros(0, dtype=np.int32)
        offsets = np.zeros(len(self._counts) + 1, dtype=np.int64)
        np.cumsum(self._counts, out=offsets[1:])
        graph = TemporalMergeGraph(
            first_frame=int(self._first_frame or 0), offsets=offsets,
            area=joined("area", np.int32), contour_area=joined("contour_area", np.float32),
            cx=joined("cx", np.float32), cy=joined("cy", np.float32),
            length=joined("length", np.float32), width=joined("width", np.float32),
            theta=joined("theta", np.float32), flags=joined("flags", np.uint8),
            src=src.astype(np.int32, copy=False), dst=dst.astype(np.int32, copy=False),
            overlap=overlap.astype(np.int32, copy=False),
            digests=np.asarray(self._digests, dtype=np.uint64),
            breaks=np.asarray(self._breaks, dtype=bool),
            image_shape=(self.height, self.width),
        )
        self._prev_map = np.zeros(0, dtype=np.int32)
        self._columns = {name: [] for name in _NODE_COLUMNS}
        self._edges = []
        return graph


@dataclass
class TemporalMergeGraph:
    """Compact node/edge arrays of one scanned frame range (no contours)."""
    first_frame: int
    offsets: np.ndarray
    area: np.ndarray
    contour_area: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    length: np.ndarray
    width: np.ndarray
    theta: np.ndarray
    flags: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    overlap: np.ndarray
    digests: np.ndarray
    breaks: np.ndarray
    image_shape: tuple

    @property
    def frame_count(self) -> int:
        return int(self.offsets.size - 1)

    @property
    def node_count(self) -> int:
        return int(self.area.size)

    def nbytes(self) -> int:
        return int(sum(getattr(self, name).nbytes for name in (
            "offsets", "src", "dst", "overlap", "digests", "breaks") + _NODE_COLUMNS))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

LEVEL_MEDIUM = 1
LEVEL_HIGH = 2
LEVEL_UNCERTAIN = 3   # visually the same violet candidate, not an automatic exclusion
LEVEL_NAMES = {LEVEL_MEDIUM: "medium", LEVEL_HIGH: "high", LEVEL_UNCERTAIN: "uncertain"}

# Internal constants. Areas are relative to S_lo, the lower single-animal
# scale (10th percentile of clean single-blob track medians, so small morphs
# of polymorphic species still count); radii use L_ref, the median length.
TINY_AREA_RATIO = 0.2            # ignored entirely (legs, antennae, noise)
SUBSTANTIAL_AREA_RATIO = 0.5     # smaller blobs never decide an event alone
STABLE_AREA_RATIO = 1.5          # consecutive confirmation frames within [1/1.5, 1.5]
CONTAINMENT_AREA_RATIO = 0.75    # a merged blob must be able to hold its largest member
AMBIGUOUS_FRACTION = 0.5         # a substantial second link suggests mass went elsewhere
SECONDARY_LINK_FLOOR = 0.25      # avoid making low containment thresholds *more* restrictive
EXIT_RADIUS = 3.0                # births/deaths within 3 L_ref may be an exit or a fast jump
EXIT_FRAMES = 2                  # ... in this many following/preceding frames
CONFIRM_FRAMES = 2               # fixed minimum consecutive normal observations
SPLIT_HISTORY_FRAMES = 10        # short fragmentation safeguard, not a merge lifetime limit
MATCH_AREA_RATIO = 1.33          # pass-through match: area ratio within [1/1.33, 1.33]
MATCH_UNIQUENESS = 1.5           # second-best assignment must cost >= 1.5x the best
SHARE_ROUNDING = 0.25            # default mass-sharing rounding bias
ONE_SIDED_MASS_RATIO = 0.6       # default retained mass for one-sided evidence
TAU_SCALE = 0.8
TAU_MIN = 0.3
TAU_MAX = 0.6
MIN_CALIBRATION_PAIRS = 100
RELIABLE_P10 = 0.5               # automatic exclusion needs P10 self-containment >= 0.5
RELIABLE_CONTINUATION = 0.85


@dataclass(frozen=True)
class TemporalMergeParams:
    tau: "float | None" = None          # None: automatic calibration
    include_one_sided: bool = False
    share_rounding: float = SHARE_ROUNDING
    one_sided_mass_ratio: float = ONE_SIDED_MASS_RATIO

    def validate(self) -> None:
        if self.tau is not None and not 0.05 <= float(self.tau) <= 0.95:
            raise ValueError("Containment threshold must be between 0.05 and 0.95.")
        if not 0.0 <= float(self.share_rounding) <= 1.0:
            raise ValueError("Mass share rounding must be between 0 and 1.")
        if not 0.0 < float(self.one_sided_mass_ratio) <= 1.0:
            raise ValueError("One-sided retained mass ratio must be in (0, 1].")


@dataclass
class TemporalMergeResult:
    levels: dict            # frame -> {blob_index: LEVEL_*}
    excluded: dict          # frame -> set of excluded blob indices
    intervals: list         # report rows (one dict per chain with evidence)
    diagnostics: dict
    counts: dict
    first_frame: int = 0
    digests: "np.ndarray | None" = None

    @property
    def excluded_count(self) -> int:
        return sum(len(v) for v in self.excluded.values())

    def level(self, frame: int, blob_index: int) -> int:
        return int(self.levels.get(int(frame), {}).get(int(blob_index), 0))

    def event_frames(self, minimum_level: int = LEVEL_MEDIUM) -> list[int]:
        # All candidate types (including uncertain) share the violet overlay
        # and must be navigable. Keep the legacy argument for compatibility.
        return sorted({int(row["first_flagged_frame"]) for row in self.intervals
                       if row["first_flagged_frame"] >= 0})

    def write_csv(self, path: str) -> None:
        columns = [
            "interval_id", "confidence", "excluded", "excluded_blobs", "high_blobs",
            "medium_blobs", "frame_start", "frame_end", "n_forward", "n_backward",
            "merge_frame", "merge_type", "merge_parents", "split_frame", "split_type",
            "split_children", "forward_evidence", "backward_evidence", "reasons", "blobs",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        temp = path + ".tmp"
        try:
            with open(temp, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["# temporal merge diagnostics"] + [
                    f"{key}={value}" for key, value in self.diagnostics.items()
                    if not isinstance(value, (list, dict))] + [
                    f"warning={text}" for text in self.diagnostics.get("warnings", [])])
                writer.writerow(columns)
                for row in self.intervals:
                    writer.writerow([row.get(column, "") for column in columns])
            os.replace(temp, path)
        except OSError:
            if os.path.exists(temp):
                os.remove(temp)
            raise


def _inside(values: np.ndarray, bounds) -> np.ndarray:
    """Same comparison as the GUI's float64 inside_bounds (float32 would round)."""
    values = np.asarray(values, dtype=np.float64)
    lo, hi = float(bounds[0]), float(bounds[1])
    return (values >= lo) & (values <= hi)


def _blob_refs(frames: np.ndarray, local: np.ndarray, limit: int = 0) -> str:
    items = [f"{int(f)}:{int(i)}" for f, i in zip(frames, local)]
    if limit and len(items) > limit:
        items = items[:limit] + [f"...(+{len(items) - limit})"]
    return ";".join(items)


@dataclass(frozen=True)
class _Evidence:
    atoms: frozenset         # chains of independent confirmed animals
    direct: bool             # False: passed an ROI edge, where an animal may leave unseen
    frame: int               # relative frame of the event that established it, -1 none
    event: int               # event id, -1 none
    max_area: float          # largest member area (a blob must be able to hold it)

    @property
    def n(self) -> int:
        return len(self.atoms)


def _median(values) -> float:
    """Median of a short array (np.median costs ~30 us per call on tiny input)."""
    v = sorted(np.asarray(values).tolist())
    n = len(v)
    if not n:
        return 0.0
    return float(v[n // 2]) if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _fresh(chain: int) -> _Evidence:
    return _Evidence(frozenset((chain,)), True, -1, -1, 0.0)


class _Evaluation:
    """One evaluation of a stored graph; see the module docstring for rules."""

    def __init__(self, graph: TemporalMergeGraph, params: TemporalMergeParams, area_bounds=None):
        params.validate()
        self.g = graph
        self.p = params
        self.k = CONFIRM_FRAMES
        # Only recent common split ancestry groups fragments of one animal;
        # the lifetime of a confirmed merged blob is not time-limited.
        self.split_history = SPLIT_HISTORY_FRAMES
        n = graph.node_count
        counts = np.diff(graph.offsets).astype(np.int64)
        self.frame = np.repeat(np.arange(graph.frame_count, dtype=np.int32), counts)
        self.local = (np.arange(n, dtype=np.int64) - graph.offsets[:-1][self.frame]).astype(np.int32)
        # Temporal merge evidence must not depend on the later statistical
        # Outlier Extraction bounds. Keep the optional argument only for
        # backward API compatibility.
        flags = graph.flags.copy()
        self.area = graph.area.astype(np.float32)
        self.normal = (flags & FLAG_NORMAL) != 0
        self.edge = (flags & FLAG_BORDER) != 0
        self.roi_edge = (flags & FLAG_ROI_EDGE) != 0
        self.unknown = (flags & FLAG_UNKNOWN) != 0
        self.result_single = (flags & FLAG_RESULT_SINGLE) != 0
        self.events: list[dict] = []
        self.stops: dict[tuple[str, int], str] = {}
        # Chain -> chain it was uniquely matched to through a contact
        # (forward: the earlier arm; backward: the later child).
        self.identity: dict[bool, dict[int, int]] = {True: {}, False: {}}
        self._confirmed_cache: dict = {}
        self.diag: dict = {"warnings": []}
        self._match_cache: dict = {}
        self._share_cache: dict = {}
        self._mass_cache: dict = {}

    # -- reference scale and calibration -------------------------------------
    def _unique_pairs(self):
        g = self.g
        n = g.node_count
        if not g.src.size:
            self.cal_step = np.zeros(0)
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
        outdeg = np.bincount(g.src, minlength=n)
        indeg = np.bincount(g.dst, minlength=n)
        src, dst = g.src.astype(np.int64), g.dst.astype(np.int64)
        ratio = self.area[dst] / np.maximum(self.area[src], 1.0)
        keep = (outdeg[src] == 1) & (indeg[dst] == 1) & self.clean[src] & self.clean[dst] \
            & (ratio >= 1 / STABLE_AREA_RATIO) & (ratio <= STABLE_AREA_RATIO)
        o = g.overlap[keep].astype(np.float64)
        contain = o / np.maximum(self.area[src[keep]], self.area[dst[keep]])
        s, d = src[keep], dst[keep]
        self.cal_step = np.hypot(g.cx[d] - g.cx[s], g.cy[d] - g.cy[s]).astype(np.float64)
        return s, d, contain

    def _reference(self) -> bool:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        g = self.g
        self.clean = self.normal & ~self.edge & ~self.roi_edge & ~self.unknown & (self.area > 0)
        self.diag.update(nodes=int(g.node_count), edges=int(g.src.size), frames=int(g.frame_count))
        if not self.clean.any():
            self.diag["warnings"].append("No normal blobs were found.")
            return False
        src, dst, contain = self._unique_pairs()
        self.cal_contain = contain
        n = g.node_count
        nodes = np.zeros(n, dtype=bool)
        nodes[src] = nodes[dst] = True
        medians_a, medians_l = [], []
        if src.size:
            graph = coo_matrix((np.ones(src.size, np.int8), (src, dst)), shape=(n, n))
            _count, comp = connected_components(graph, directed=False)
            members = np.flatnonzero(nodes)
            order = members[np.argsort(comp[members], kind="stable")]
            labels = comp[order]
            bounds = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b - a >= self.k:
                    medians_a.append(_median(self.area[order[a:b]]))
                    medians_l.append(_median(g.length[order[a:b]]))
        if len(medians_a) >= 3:
            self.A = float(np.median(medians_a))
            self.S = float(np.percentile(medians_a, 10))
            self.L = float(np.median(medians_l))
            self.diag["reference"] = "tracks"
        else:
            self.A = float(np.median(self.area[self.clean]))
            self.S = float(np.percentile(self.area[self.clean], 10))
            self.L = float(np.median(g.length[self.clean]))
            self.diag["reference"] = "blobs"
        self.S = max(min(self.S, self.A), 1.0)
        self.diag.update(A_ref=round(self.A, 1), S_lo=round(self.S, 1), L_ref=round(self.L, 2),
                         reference_tracks=len(medians_a))
        self.tiny = self.area < TINY_AREA_RATIO * self.S
        self.substantial = self.area >= SUBSTANTIAL_AREA_RATIO * self.S
        return self.L > 0

    def _calibrate(self) -> None:
        contain = self.cal_contain
        samples = int(contain.size)
        p10 = float(np.percentile(contain, 10)) if samples else 0.0
        tau_auto = float(np.clip(TAU_SCALE * p10, TAU_MIN, TAU_MAX)) if samples else TAU_MIN
        self.tau = float(self.p.tau) if self.p.tau is not None else tau_auto
        self.p10, self.samples = p10, samples
        step = float(np.percentile(self.cal_step, 99)) if self.cal_step.size else 0.0
        # Unexplained births/deaths this close to a blob may be its exit/entry.
        self.exit_radius = max(EXIT_RADIUS * self.L, 2.0 * step)
        self.diag["exit_radius"] = round(self.exit_radius, 1)
        self.diag.update(
            calibration_pairs=samples, self_containment_p10=round(p10, 3),
            tau_auto=round(tau_auto, 3), tau=round(self.tau, 3),
            tau_mode="manual" if self.p.tau is not None else "auto")

    # -- graph topology -----------------------------------------------------
    def _topology(self, active: np.ndarray) -> None:
        """Dominant links, transition components and 1:1 chains on ``active`` nodes."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        g = self.g
        n = g.node_count
        keep = active[g.src] & active[g.dst] if g.src.size else np.zeros(0, bool)
        src = g.src[keep].astype(np.int64)
        dst = g.dst[keep].astype(np.int64)
        o = g.overlap[keep].astype(np.float32)
        f = o / self.area[src] if o.size else o
        gb = o / self.area[dst] if o.size else o
        self.second_f = np.zeros(n, dtype=np.float32)
        self.second_g = np.zeros(n, dtype=np.float32)
        fwd = np.zeros(src.size, dtype=bool)
        bwd = np.zeros(src.size, dtype=bool)
        if src.size:
            for key, val, second, mask in ((src, f, self.second_f, fwd), (dst, gb, self.second_g, bwd)):
                order = np.lexsort((-val, key))
                first = np.r_[True, key[order][1:] != key[order][:-1]]
                nxt = np.r_[False, first[:-1]] & ~first
                best = order[first]
                second[key[order[nxt]]] = val[order[nxt]]
                mask[best] = val[best] >= self.tau
        dom = fwd | bwd
        self.dsrc, self.ddst = src[dom], dst[dom]
        self.has_fwd = np.zeros(n, dtype=bool)
        self.has_fwd[src[fwd]] = True
        # Lookup of overlap for dominant edges (sorted keys).
        dkey = self.dsrc * n + self.ddst
        order = np.argsort(dkey)
        self.dkey = dkey[order]
        self.dover = o[dom][order]
        # Substantial neighbours with any overlap (for explained shrinks).
        sub = self.substantial
        both = sub[src] & sub[dst]
        self.sub_children = np.bincount(src[both], minlength=n).astype(np.int32)
        self.sub_parents = np.bincount(dst[both], minlength=n).astype(np.int32)

        m = self.dsrc.size
        graph = coo_matrix((np.ones(m, dtype=np.int8), (self.dsrc, n + self.ddst)), shape=(2 * n, 2 * n))
        _count, labels = connected_components(graph, directed=False)
        par = np.unique(self.dsrc)
        chi = np.unique(self.ddst)
        self.pc = np.full(n, -1, dtype=np.int32)
        self.cc = np.full(n, -1, dtype=np.int32)
        self.pc[par] = labels[par]
        self.cc[chi] = labels[n + chi]
        ncomp = int(labels.max()) + 1 if labels.size else 0
        self.nP = np.bincount(labels[par], minlength=ncomp)
        self.nC = np.bincount(labels[n + chi], minlength=ncomp)
        self.comp_par = par[np.argsort(labels[par], kind="stable")]
        self.comp_par_ptr = np.r_[0, np.cumsum(self.nP)]
        self.comp_chi = chi[np.argsort(labels[n + chi], kind="stable")]
        self.comp_chi_ptr = np.r_[0, np.cumsum(self.nC)]

        one = np.flatnonzero((self.nP == 1) & (self.nC == 1))
        p1 = self.comp_par[self.comp_par_ptr[one]]
        c1 = self.comp_chi[self.comp_chi_ptr[one]]
        hard = self.edge | self.unknown
        link = ~hard[p1] & ~hard[c1]
        graph = coo_matrix((np.ones(int(link.sum()), np.int8), (p1[link], c1[link])), shape=(n, n))
        nchain, chain = connected_components(graph, directed=False)
        chain = chain.astype(np.int32)
        chain[~active] = -1
        self.chain = chain
        valid = np.flatnonzero(active)
        order = valid[np.lexsort((self.frame[valid], chain[valid]))]
        lengths = np.bincount(chain[valid], minlength=nchain)
        self.nchain = int(nchain)
        self.ch_ptr = np.r_[0, np.cumsum(lengths)]
        self.ch_nodes = order
        self.ch_len = lengths
        has = lengths > 0
        self.ch_start = np.full(nchain, -1, dtype=np.int64)
        self.ch_end = np.full(nchain, -1, dtype=np.int64)
        self.ch_start[has] = order[self.ch_ptr[:-1][has]]
        self.ch_end[has] = order[self.ch_ptr[1:][has] - 1]

    def parents(self, comp: int) -> np.ndarray:
        return self.comp_par[self.comp_par_ptr[comp]:self.comp_par_ptr[comp + 1]]

    def children(self, comp: int) -> np.ndarray:
        return self.comp_chi[self.comp_chi_ptr[comp]:self.comp_chi_ptr[comp + 1]]

    def nodes_of(self, c: int) -> np.ndarray:
        return self.ch_nodes[self.ch_ptr[c]:self.ch_ptr[c + 1]]

    def overlap(self, a: int, b: int) -> float:
        key = a * self.g.node_count + b
        i = int(np.searchsorted(self.dkey, key))
        return float(self.dover[i]) if i < self.dkey.size and self.dkey[i] == key else 0.0

    def _comp_kind(self, c: int) -> str:
        p, q = int(self.nP[c]), int(self.nC[c])
        if p == 1 and q == 1:
            return "reset"          # a 1:1 link that crosses an edge/unknown node
        if p >= 2 and q == 1:
            return "merge"
        if p == 1 and q >= 2:
            return "split"
        return "complex"

    def _kinds(self) -> None:
        last = self.g.frame_count - 1
        breaks = self.g.breaks
        self.start_kind = [""] * self.nchain
        self.end_kind = [""] * self.nchain
        births: dict[int, list[int]] = {}
        deaths: dict[int, list[int]] = {}
        for c in range(self.nchain):
            m = int(self.ch_start[c])
            if m < 0:
                continue
            fm = int(self.frame[m])
            if fm == 0:
                kind = "scan_start"
            elif breaks.size and breaks[fm - 1]:
                kind = "context_break"
            elif self.cc[m] < 0:
                kind = "birth"
            else:
                kind = self._comp_kind(int(self.cc[m]))
            self.start_kind[c] = kind
            if kind == "birth" and self.substantial[m]:
                births.setdefault(fm, []).append(m)
            e = int(self.ch_end[c])
            fe = int(self.frame[e])
            if fe == last:
                kind = "scan_end"
            elif breaks.size and breaks[fe]:
                kind = "context_break"
            elif self.pc[e] < 0:
                kind = "death"
            else:
                kind = self._comp_kind(int(self.pc[e]))
            self.end_kind[c] = kind
            if kind == "death" and self.substantial[e]:
                deaths.setdefault(fe, []).append(e)
        self.births = {f: np.asarray(v, np.int64) for f, v in births.items()}
        self.deaths = {f: np.asarray(v, np.int64) for f, v in deaths.items()}

    # -- tests on arms -------------------------------------------------------
    def _window(self, c: int, forward: bool, size: int) -> np.ndarray:
        """Up to ``size`` nodes of chain c next to its event (end if forward)."""
        lo, hi = int(self.ch_ptr[c]), int(self.ch_ptr[c + 1])
        return self.ch_nodes[max(lo, hi - size):hi] if forward else self.ch_nodes[lo:min(hi, lo + size)]

    def _confirmed(self, c: int, forward: bool, depth: int = 0) -> str:
        """'' when the chain is a confirmed animal right at its event.

        A chain shorter than k is confirmed when its own nodes are clean and
        the chain it was uniquely matched to through a contact is confirmed.
        """
        key = (c, forward)
        if depth == 0 and key in self._confirmed_cache:
            return self._confirmed_cache[key]
        result = self._confirm(c, forward, depth)
        if depth == 0:
            self._confirmed_cache[key] = result
        return result

    def _confirm(self, c: int, forward: bool, depth: int) -> str:
        if int(self.ch_len[c]) < self.k:
            previous = self.identity[forward].get(c)
            if previous is None or depth >= 3:
                return "short"
            nodes = self.nodes_of(c)
            if not self.normal[nodes].all() or (self.edge[nodes] | self.roi_edge[nodes] | self.unknown[nodes]).any():
                return "short"
            if _median(self.area[nodes]) < SUBSTANTIAL_AREA_RATIO * self.S:
                return "small"
            return "" if not self._confirmed(previous, forward, depth + 1) else "short"
        # Bodies often change apparent area just before contact. Require a
        # stable, clean k-frame observation near the event, but do not demand
        # that the last k frames themselves have stable contour areas.
        nodes = self._window(c, forward, min(int(self.ch_len[c]), max(10, 4 * self.k)))
        if not self.clean[nodes[-1] if forward else nodes[0]]:
            return "edge_or_outlier"
        if not self.substantial[nodes[-1] if forward else nodes[0]]:
            return "small"
        for i in range(nodes.size - self.k + 1):
            sample = nodes[i:i + self.k]
            if not self.clean[sample].all() or not self.substantial[sample].all():
                continue
            areas = self.area[sample]
            ratio = areas[1:] / np.maximum(areas[:-1], 1.0)
            if ((ratio >= 1 / STABLE_AREA_RATIO) & (ratio <= STABLE_AREA_RATIO)).all():
                return ""
        return "unstable_or_not_normal"

    def _near(self, table: dict, frames, x: float, y: float, radius: float) -> bool:
        for f in frames:
            nodes = table.get(int(f))
            if nodes is not None and (np.hypot(self.g.cx[nodes] - x, self.g.cy[nodes] - y) <= radius).any():
                return True
        return False

    def _ambiguous(self, arm: int, event_node: int, chain: int, forward: bool) -> str:
        secondary_limit = max(AMBIGUOUS_FRACTION * self.tau, SECONDARY_LINK_FLOOR)
        if forward and self.second_f[arm] >= secondary_limit:
            return "split_mass"
        if not forward and self.second_g[arm] >= secondary_limit:
            return "mixed_source"
        median = float(_median(self.area[self._window(chain, forward, self.k)]))
        o = self.overlap(arm, event_node) if forward else self.overlap(event_node, arm)
        if median > 0 and o / median < self.tau:
            return "weak_containment"
        radius = max(self.exit_radius, 1.5 * float(self.g.length[arm]))
        t = int(self.frame[event_node])
        frames = range(t, t + EXIT_FRAMES + 1) if forward else range(t - EXIT_FRAMES, t + 1)
        if self._near(self.births if forward else self.deaths, frames,
                      float(self.g.cx[arm]), float(self.g.cy[arm]), radius):
            return "possible_fast_motion"
        return ""

    def _common_nodes(self, c1: int, c2: int, forward: bool, t: int):
        """Find up to k clean, substantial shared frames without a time cutoff.

        Scan from the event outwards, stopping as soon as independent
        observations are sufficient. This avoids allocating an entire
        long-video history at each contact event.
        """
        def valid(a, b):
            return self.clean[a] & self.clean[b] & self.substantial[a] & self.substantial[b]

        ident = self.identity[forward]
        matched_a, matched_b = [], []
        if c1 not in ident and c2 not in ident:
            s1, s2 = int(self.frame[self.ch_start[c1]]), int(self.frame[self.ch_start[c2]])
            e1, e2 = int(self.frame[self.ch_end[c1]]), int(self.frame[self.ch_end[c2]])
            lo = max(s1, s2, t + 1 if not forward else s1)
            hi = min(e1, e2, t - 1 if forward else e1)
            # Check at most 1024 frames at a time, nearest the event first.
            cursor = hi if forward else lo
            while lo <= cursor <= hi and len(matched_a) < self.k:
                if forward:
                    first = max(lo, cursor - 1023)
                    frames = np.arange(first, cursor + 1)
                    cursor = first - 1
                else:
                    last = min(hi, cursor + 1023)
                    frames = np.arange(cursor, last + 1)
                    cursor = last + 1
                a = self.ch_nodes[self.ch_ptr[c1] + frames - s1]
                b = self.ch_nodes[self.ch_ptr[c2] + frames - s2]
                good = np.flatnonzero(valid(a, b))
                if forward:
                    good = good[::-1]
                good = good[:self.k - len(matched_a)]
                matched_a.extend(a[good].tolist())
                matched_b.extend(b[good].tolist())
        else:
            def walk(c):
                last_frame = t
                for _ in range(4):  # follow unique identity across up to 3 contacts
                    nodes = self.nodes_of(c)
                    for node in (nodes[::-1] if forward else nodes):
                        f = int(self.frame[node])
                        if (f < last_frame) if forward else (f > last_frame):
                            last_frame = f
                            yield f, int(node)
                    c = ident.get(c)
                    if c is None:
                        break

            left, right = walk(c1), walk(c2)
            a, b = next(left, None), next(right, None)
            while a is not None and b is not None and len(matched_a) < self.k:
                if a[0] == b[0]:
                    if bool(valid(a[1], b[1])):
                        matched_a.append(a[1])
                        matched_b.append(b[1])
                    a, b = next(left, None), next(right, None)
                elif (a[0] > b[0]) if forward else (a[0] < b[0]):
                    a = next(left, None)
                else:
                    b = next(right, None)
        return (np.asarray(matched_a, dtype=np.int64),
                np.asarray(matched_b, dtype=np.int64))

    def _independence(self, c1: int, c2: int, forward: bool, t: int):
        """Were both arms separate, credible blobs in at least k shared frames?

        Only separate normal, substantial segmented components qualify.
        A recent common split ancestor is handled separately in _combine().
        """
        a, b = self._common_nodes(c1, c2, forward, t)
        shared = int(a.size)
        return shared >= self.k, shared

    @staticmethod
    def _best_clique(units: list, independent) -> list[int]:
        """Pairwise-independent subset with the largest atom union.

        Ties prefer the lexicographically smallest sorted atom list, so the
        result never depends on the order arms were visited. Events with
        more than 8 usable arms are left uncertain.
        """
        count = len(units)
        if count < 2 or count > 8:
            return []
        best, best_key = [], None
        for size in range(2, count + 1):
            for subset in itertools.combinations(range(count), size):
                if not all(independent[i][j] for i, j in itertools.combinations(subset, 2)):
                    continue
                atoms = frozenset().union(*(units[i]["ev"].atoms for i in subset))
                key = (-len(atoms), sorted(atoms))
                if best_key is None or key < best_key:
                    best, best_key = list(subset), key
        return best

    def _match(self, sources: list[int], targets: list[int], forward: bool):
        """Unique one-to-one match target -> source, or None.

        Sizes are chain medians next to the contact (sources end there when
        forward, targets start there), not the contact frames themselves.
        """
        n = len(sources)
        if n < 2 or n != len(targets) or n > 5:
            return None
        g = self.g
        source_mass = [max(self._mass(int(self.chain[s]), forward), 1.0) for s in sources]
        target_mass = [self._mass(int(self.chain[t]), not forward) for t in targets]
        cost = np.full((n, n), np.inf)
        for i, s in enumerate(sources):
            for j, t in enumerate(targets):
                ratio = target_mass[j] / source_mass[i]
                if 1 / MATCH_AREA_RATIO <= ratio <= MATCH_AREA_RATIO:
                    d = math.hypot(float(g.cx[t] - g.cx[s]), float(g.cy[t] - g.cy[s])) / self.L
                    cost[i, j] = d + 2.0 * abs(math.log(ratio))
        totals = sorted(((sum(cost[i, perm[i]] for i in range(n)), perm)
                         for perm in itertools.permutations(range(n))), key=lambda item: item[0])
        best, perm = totals[0]
        if not math.isfinite(best):
            return None
        second = totals[1][0]
        if math.isfinite(second) and second < max(MATCH_UNIQUENESS * best, best + 0.5):
            return None
        return {int(targets[perm[i]]): int(sources[i]) for i in range(n)}

    def _ref(self, node: int) -> str:
        return f"{int(self.frame[node]) + self.g.first_frame}:{int(self.local[node])}"

    # -- evidence passes ------------------------------------------------------
    def _pass(self, forward: bool):
        """Per-node evidence index and the list of evidence objects."""
        n, C, W = self.g.node_count, self.nchain, self.split_history
        store: list[_Evidence] = []
        node_ev = np.full(n, -1, dtype=np.int32)
        boundary: list = [None] * C          # evidence at the chain end handed to events
        lineage: list = [None] * C           # recent splits (forward) / merges (backward)
        heads = self.ch_start if forward else self.ch_end
        key = np.where(heads >= 0, self.frame[np.maximum(heads, 0)], 0)
        key = key if forward else -key
        table = self.births if forward else self.deaths
        for c in np.argsort(key, kind="stable"):
            c = int(c)
            if self.ch_len[c] == 0:
                continue
            nodes = self.nodes_of(c)
            if not forward:
                nodes = nodes[::-1]
            head = int(nodes[0])
            kind = self.start_kind[c] if forward else self.end_kind[c]
            t = int(self.frame[head])
            anc: dict = {}
            if self.edge[head] or self.unknown[head] or self.result_single[head]:
                ev = _fresh(c)
            elif kind == ("merge" if forward else "split"):
                ev, anc = self._combine(c, head, forward, boundary, lineage)
            elif kind in (("split", "complex") if forward else ("merge", "complex")):
                ev, anc = self._pass_through(c, head, kind, forward, boundary, lineage)
            else:
                ev = _fresh(c)
            lineage[c] = {k: v for k, v in anc.items() if abs(t - v[0]) <= W}
            store.append(ev)
            node_ev[head] = len(store) - 1
            if ev.n < 2:
                node_ev[nodes] = len(store) - 1      # nothing to propagate or stop
                boundary[c] = ev
                continue
            for i in range(1, len(nodes)):
                v, u = int(nodes[i]), int(nodes[i - 1])
                if ev.n >= 2:
                    reason = self._exit(u, v, ev, forward, table)
                    if reason:
                        self.stops[("forward" if forward else "backward", c)] = \
                            f"{reason}@{int(self.frame[v]) + self.g.first_frame}"
                        ev = _fresh(c)
                        store.append(ev)
                    elif self.roi_edge[v] and ev.direct:
                        ev = _Evidence(ev.atoms, False, ev.frame, ev.event, ev.max_area)
                        store.append(ev)
                node_ev[v] = len(store) - 1
            boundary[c] = ev
        return store, node_ev

    def _exit(self, u: int, v: int, ev: _Evidence, forward: bool, table: dict) -> str:
        """Why the count may have dropped between consecutive chain nodes u -> v.

        Counts move only with mass: lost area alone (occlusion) is no exit,
        but part of the mass flowing to another blob, a nearby unexplained
        birth/death, or a blob too small for its largest member is.
        """
        if self.result_single[v]:
            return "result_single"
        if self.area[v] < CONTAINMENT_AREA_RATIO * ev.max_area:
            return "area_floor"
        secondary_limit = max(AMBIGUOUS_FRACTION * self.tau, SECONDARY_LINK_FLOOR)
        if forward and self.second_f[u] >= secondary_limit:
            return "split_mass"
        if not forward and self.second_g[u] >= secondary_limit:
            return "mixed_source"
        if self.area[v] < self.area[u] / STABLE_AREA_RATIO:
            other = self.sub_children[u] if forward else self.sub_parents[u]
            if other >= 2:
                return "shrink_exit"
        # A nearby new blob is not evidence that animals have left an
        # unchanged merged blob. Apply proximity exits only when the blob
        # actually loses substantial area.
        if self.area[v] < self.area[u] * CONTAINMENT_AREA_RATIO:
            t = int(self.frame[v])
            frames = range(t, t + EXIT_FRAMES + 1) if forward else range(t - EXIT_FRAMES, t + 1)
            radius = max(self.exit_radius, 1.5 * float(self.g.length[u]))
            for node in (u, v):
                if self._near(table, frames, float(self.g.cx[node]), float(self.g.cy[node]), radius):
                    return "possible_departure" if forward else "possible_arrival"
        return ""

    def _combine(self, c: int, head: int, forward: bool, boundary: list, lineage: list):
        t = int(self.frame[head])
        comp = int(self.cc[head] if forward else self.pc[head])
        arms = [int(a) for a in (self.parents(comp) if forward else self.children(comp))]
        rows = []
        for a in arms:
            ca = int(self.chain[a])
            recent = {k: v for k, v in (lineage[ca] or {}).items() if abs(t - v[0]) <= self.split_history}
            rows.append({"node": a, "chain": ca, "why": self._confirmed(ca, forward),
                         "ambiguous": self._ambiguous(a, head, ca, forward),
                         "substantial": bool(self.substantial[a]), "ev": boundary[ca],
                         "lineage": recent})
        # Arms descending from one recent split (or, backward, merging into
        # one later merge) form a group: one body may have fragmented.
        parent = list(range(len(rows)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j in itertools.combinations(range(len(rows)), 2):
            if set(rows[i]["lineage"]) & set(rows[j]["lineage"]):
                parent[find(i)] = find(j)
        groups: dict[int, list[dict]] = {}
        for i, row in enumerate(rows):
            groups.setdefault(find(i), []).append(row)
        units = []
        for members in groups.values():
            usable = [r for r in members if not r["ambiguous"]]
            confirmed = [r for r in usable if not r["why"]]
            rep = max(confirmed, key=lambda r: self.area[r["node"]]) if confirmed else None
            best_member = max(usable, key=lambda r: r["ev"].n) if usable else None
            ev = best_member["ev"] if best_member is not None else None
            closed = False
            if len(members) >= 2:
                shared = set.intersection(*(set(r["lineage"]) for r in members))
                if shared:
                    key = max(shared, key=lambda k: members[0]["lineage"][k][0] * (1 if forward else -1))
                    _frame, origin_chain, branch_chains = members[0]["lineage"][key]
                    if set(branch_chains) == {r["chain"] for r in members} and boundary[origin_chain] is not None:
                        ev = boundary[origin_chain]       # conservation through a closed bundle
                        closed = True
            units.append({"members": members, "rep": rep, "ev": ev, "closed": closed})
        good = [u for u in units if u["rep"] is not None and u["ev"] is not None]
        independent = [[False] * len(good) for _ in good]
        pairs = []
        for i, j in itertools.combinations(range(len(good)), 2):
            ok, shared = self._independence(good[i]["rep"]["chain"], good[j]["rep"]["chain"], forward, t)
            independent[i][j] = independent[j][i] = ok
            pairs.append((self._ref(good[i]["rep"]["node"]), self._ref(good[j]["rep"]["node"]), ok, shared))
        clique = [good[i] for i in self._best_clique(good, independent)]
        reasons = []
        union = frozenset()
        max_area = 0.0
        if len(clique) >= 2:
            max_area = max(_median(self.area[self._window(u["rep"]["chain"], forward, self.k)])
                           for u in clique)
            if self.area[head] < CONTAINMENT_AREA_RATIO * max_area:
                reasons.append("area_conflict")
            else:
                union = frozenset().union(*(u["ev"].atoms for u in clique))
        donors = [u for u in units if u["ev"] is not None]
        donor = max(donors, key=lambda u: (u["ev"].n, -min(u["ev"].atoms))) if donors else None
        inherited = donor["ev"] if donor is not None and "area_conflict" not in reasons else _fresh(c)
        if len(good) > 8:
            reasons.append("too_many_arms")
        if len(union) >= 2 and len(union) >= inherited.n:
            ev = _Evidence(union, True, t, len(self.events), max_area)
            source = "independent"
        elif inherited.n >= 2:
            ev = inherited
            source = "bundle" if donor["closed"] else "inherited"
        else:
            ev, source = _fresh(c), "none"
        merged_lineage: dict = {}
        for row in rows:
            merged_lineage.update(row["lineage"])
        substantial = sum(r["substantial"] for r in rows)
        if substantial >= 2 or ev.n >= 2:
            if ev.n < 2:
                if any(r["ambiguous"] for r in rows):
                    reasons.append("ambiguous_arm")
                if len(good) < 2:
                    reasons.append("arms_not_confirmed")
                elif len(clique) < 2:
                    reasons.append("arms_not_independent")
            self.events.append({
                "id": len(self.events), "direction": "forward" if forward else "backward",
                "type": "merge" if forward else "split", "frame": t + self.g.first_frame,
                "node": head, "chain": c, "arms": rows, "pairs": pairs, "n": ev.n,
                "source": source, "reasons": reasons})
        return ev, merged_lineage

    def _pass_through(self, c: int, head: int, kind: str, forward: bool, boundary: list, lineage: list):
        """Evidence for a split/complex child (forward) or merge/complex parent (backward)."""
        t = int(self.frame[head])
        comp = int(self.cc[head] if forward else self.pc[head])
        sources = [int(x) for x in (self.parents(comp) if forward else self.children(comp))]
        targets = [int(x) for x in (self.children(comp) if forward else self.parents(comp))]
        givers, receivers = list(sources), list(targets)
        anc: dict = {}
        for s in sources:
            anc.update(lineage[int(self.chain[s])] or {})
        if kind == ("split" if forward else "merge"):
            origin = int(self.chain[sources[0]])
            branches = tuple(sorted(int(self.chain[x]) for x in targets))
            anc[("s" if forward else "m", comp)] = (t, origin, branches)
            # Match through the intervening merged blob when the geometric correspondence is unique.
            node = int(self.ch_start[origin] if forward else self.ch_end[origin])
            far = int(self.frame[sources[0]])
            mirror = self.start_kind[origin] if forward else self.end_kind[origin]
            if mirror == ("merge" if forward else "split"):
                mcomp = int(self.cc[node] if forward else self.pc[node])
                sources = [int(x) for x in (self.parents(mcomp) if forward else self.children(mcomp))]
            else:
                sources = []
        sources = [s for s in sources if self.substantial[s]]
        targets = [x for x in targets if self.substantial[x]]
        if head in targets and sources:
            cache_key = (forward, tuple(sorted(sources)), tuple(sorted(targets)))
            if cache_key not in self._match_cache:
                self._match_cache[cache_key] = self._match(sources, targets, forward)
            match = self._match_cache[cache_key]
            if match and head in match:
                source = int(self.chain[match[head]])
                self.identity[forward][c] = source
                donor = boundary[source]
                if donor is not None:
                    return donor, anc          # same animals (one atom for a single)
        shares = self._share(comp, forward, givers, receivers, boundary, t)
        if head in shares:
            return shares[head], anc
        return _fresh(c), anc

    def _mass(self, c: int, at_end: bool) -> float:
        """Median area of chain c over the nodes next to its end (or start)."""
        key = (c, at_end)
        mass = self._mass_cache.get(key)
        if mass is None:
            size = min(int(self.ch_len[c]), max(10, 4 * self.k))
            mass = float(_median(self.area[self._window(c, at_end, size)])) if size else 0.0
            self._mass_cache[key] = mass
        return mass

    def _share(self, comp: int, forward: bool, givers: list, receivers: list,
               boundary: list, t: int) -> dict:
        """Receiver node -> its share of the givers' known count, where >= 2.

        Animals move with mass: a receiver taking a fraction s of the
        receivers' mass holds floor(n * s + 0.25) of the n known animals.
        Disjoint atoms keep receivers that later re-merge from counting more
        animals than were known before they parted.
        """
        key = (forward, comp)
        if key in self._share_cache:
            return self._share_cache[key]
        out: dict = {}
        self._share_cache[key] = out
        evs = [boundary[int(self.chain[g])] for g in givers]
        if len(receivers) < 2 or any(e is None for e in evs):
            return out
        pool = sorted(frozenset().union(*(e.atoms for e in evs)))
        n = len(pool)
        if n < 2:
            return out
        direct = all(e.direct for e in evs)
        # Receivers end at a merge (backward) or start at a split (forward).
        masses = [self._mass(int(self.chain[r]), not forward) for r in receivers]
        total = float(sum(masses))
        if total <= 0:
            return out
        for i in sorted(range(len(receivers)), key=lambda i: (-masses[i], receivers[i])):
            r = receivers[i]
            share = masses[i] / total
            count = min(int(math.floor(n * share + self.p.share_rounding)), len(pool))
            if count < 2 or not self.substantial[r]:
                continue
            event = len(self.events)
            # The area floor along the chain: one animal's share of the mass.
            out[r] = _Evidence(frozenset(pool[:count]), direct, t, event, masses[i] / count)
            pool = pool[count:]
            self.events.append({
                "id": event, "direction": "forward" if forward else "backward",
                "type": "split_share" if forward else "merge_share", "frame": t + self.g.first_frame,
                "node": r, "chain": int(self.chain[r]), "arms": [], "pairs": [], "n": count,
                "source": "mass_share", "reasons": [],
                "detail": f"share={share:.2f} of n={n} from {';'.join(self._ref(g) for g in givers)}"})
        return out

    # -- decisions ------------------------------------------------------------
    def _describe(self, event_id: int) -> str:
        if event_id < 0 or event_id >= len(self.events):
            return ""
        event = self.events[int(event_id)]
        parts = [f"{event['type']}@{event['frame']} n={event['n']} source={event['source']}"]
        if event.get("detail"):
            parts.append(event["detail"])
        for row in event["arms"]:
            status = row["ambiguous"] or row["why"] or "ok"
            parts.append(f"arm {self._ref(row['node'])}:{status}")
        for a, b, ok, shared in event["pairs"]:
            parts.append(f"{a}~{b}:{'independent' if ok else 'dependent'} shared={shared}frames")
        if event["reasons"]:
            parts.append("reasons=" + "+".join(event["reasons"]))
        return " | ".join(parts)

    def run(self) -> TemporalMergeResult:
        g = self.g
        empty_counts = {"high_intervals": 0, "medium_intervals": 0, "uncertain_intervals": 0,
                        "high_blobs": 0, "medium_blobs": 0, "excluded_blobs": 0}
        if g.node_count == 0 or not self._reference():
            self.diag["reliable"] = False
            return TemporalMergeResult({}, {}, [], self.diag, empty_counts, g.first_frame, g.digests)
        self._calibrate()
        active = ~self.tiny & (self.area > 0)
        self._topology(active)
        # Short-lived blobs between tiny and substantial are satellites
        # (shadows, legs, a cut-off abdomen tip): drop them and rebuild, so
        # they neither type an event nor reset a count.
        satellite = active & ~self.substantial & (self.chain >= 0)
        satellite &= self.ch_len[np.maximum(self.chain, 0)] < self.k
        if satellite.any():
            active &= ~satellite
            self._topology(active)
        self.diag["satellites"] = int(satellite.sum())
        self.diag["chains"] = self.nchain
        self._reliability(active)
        self._kinds()
        fstore, fnode = self._pass(True)
        bstore, bnode = self._pass(False)

        n = g.node_count
        nf = np.zeros(n, dtype=np.int32)
        nb = np.zeros(n, dtype=np.int32)
        f_direct = np.zeros(n, dtype=bool)
        b_direct = np.zeros(n, dtype=bool)
        f_frame = np.full(n, -1, dtype=np.int32)
        b_frame = np.full(n, -1, dtype=np.int32)
        f_event = np.full(n, -1, dtype=np.int32)
        b_event = np.full(n, -1, dtype=np.int32)
        for node_ev, store, cnt, direct, frame, event in (
                (fnode, fstore, nf, f_direct, f_frame, f_event),
                (bnode, bstore, nb, b_direct, b_frame, b_event)):
            has = np.flatnonzero(node_ev >= 0)
            if not store:
                continue
            table = np.array([(e.n, e.direct, e.frame, e.event) for e in store], dtype=np.int64)
            idx = node_ev[has]
            cnt[has] = table[idx, 0]
            direct[has] = table[idx, 1].astype(bool)
            frame[has] = table[idx, 2]
            event[has] = table[idx, 3]
        flaggable = self.normal & ~self.edge & ~self.unknown
        high = (nf >= 2) & (nb >= 2)
        med_f = (nf >= 2) & f_direct & (f_frame >= 0)
        med_b = (nb >= 2) & b_direct & (b_frame >= 0)
        # Beyond H a blob must also keep its mass: an animal that left unseen
        # takes its area along, while occlusion alone shrinks a group less.
        med_f &= self._keeps_mass(med_f, at_end=False)
        med_b &= self._keeps_mass(med_b, at_end=True)
        medium = (med_f | med_b) & ~high
        level = np.zeros(n, dtype=np.int8)
        target = flaggable & ~self.result_single
        level[high & target] = LEVEL_HIGH
        level[medium & target] = LEVEL_MEDIUM
        evidence = (nf >= 2) | (nb >= 2)
        uncertain = evidence & flaggable & (level == 0) & ~self.result_single
        # Expose every temporal merge candidate, including uncertain ones,
        # with the same violet appearance. Only verified exclusions affect
        # the actual training mask.
        level[uncertain] = LEVEL_UNCERTAIN
        # Interval confidence also counts blobs that are already outliers
        # (for example side-by-side merges caught by the area bounds).
        consistent = ~self.result_single & ~self.unknown & ~self.edge
        evidence_level = np.zeros(n, dtype=np.int8)
        evidence_level[medium & consistent] = LEVEL_MEDIUM
        evidence_level[high & consistent] = LEVEL_HIGH

        auto = self.reliable
        exclude_medium = bool(self.p.include_one_sided) and auto
        if not auto:
            self.diag["warnings"].append(
                "Automatic exclusion is disabled for this video: consecutive-frame overlap is "
                "not reliable enough. Candidates are shown but not excluded.")
        levels: dict[int, dict[int, int]] = {}
        excluded: dict[int, set[int]] = {}
        for node in np.flatnonzero(level):
            fid = int(self.frame[node]) + g.first_frame
            code = int(level[node])
            levels.setdefault(fid, {})[int(self.local[node])] = code
            if auto and (code == LEVEL_HIGH or (exclude_medium and code == LEVEL_MEDIUM)):
                excluded.setdefault(fid, set()).add(int(self.local[node]))
        rows = self._report(level, evidence_level, uncertain, nf, nb, f_event, b_event, auto, exclude_medium)
        counts = {
            "high_intervals": sum(1 for r in rows if r["level_code"] == LEVEL_HIGH),
            "medium_intervals": sum(1 for r in rows if r["level_code"] == LEVEL_MEDIUM),
            "uncertain_intervals": sum(1 for r in rows if r["level_code"] == 0),
            "high_blobs": int((level == LEVEL_HIGH).sum()),
            "medium_blobs": int((level == LEVEL_MEDIUM).sum()),
            "excluded_blobs": sum(len(v) for v in excluded.values()),
        }
        self.diag["events"] = len(self.events)
        self.diag["mass_shares"] = sum(1 for e in self.events if e["source"] == "mass_share")
        return TemporalMergeResult(levels, excluded, rows, self.diag, counts, g.first_frame, g.digests)

    def _keeps_mass(self, candidates: np.ndarray, at_end: bool) -> np.ndarray:
        """Candidates retaining the configured fraction of their chain's reference mass."""
        out = np.zeros(self.g.node_count, dtype=bool)
        nodes = np.flatnonzero(candidates & (self.chain >= 0))
        if nodes.size:
            chains, index = np.unique(self.chain[nodes], return_inverse=True)
            mass = np.array([self._mass(int(c), at_end) for c in chains], dtype=np.float32)
            out[nodes] = self.area[nodes] >= self.p.one_sided_mass_ratio * mass[index]
        return out

    def _reliability(self, active: np.ndarray) -> None:
        last = self.frame == self.g.frame_count - 1
        nxt_break = np.zeros(self.g.node_count, dtype=bool)
        if self.g.breaks.size:
            inner = ~last
            nxt_break[inner] = self.g.breaks[self.frame[inner]]
        eligible = self.clean & active & ~last & ~nxt_break
        rate = float(self.has_fwd[eligible].mean()) if eligible.any() else 0.0
        self.diag["continuation_rate"] = round(rate, 3)
        warnings = self.diag["warnings"]
        reliable = True
        if self.samples < MIN_CALIBRATION_PAIRS:
            reliable = False
            warnings.append(f"Only {self.samples} unambiguous 1:1 frame pairs (< {MIN_CALIBRATION_PAIRS}).")
        if self.p10 < RELIABLE_P10:
            reliable = False
            warnings.append(
                f"Consecutive-frame overlap is small (self-containment P10={self.p10:.2f} < {RELIABLE_P10}): "
                "animals move too far between frames for reliable temporal association.")
        if rate < RELIABLE_CONTINUATION:
            reliable = False
            warnings.append(f"Only {rate:.0%} of single blobs have an overlapping successor "
                            f"(< {RELIABLE_CONTINUATION:.0%}).")
        self.reliable = reliable
        self.diag["reliable"] = bool(reliable)

    def _report(self, level, evidence_level, uncertain, nf, nb, f_event, b_event, auto,
                exclude_medium) -> list[dict]:
        first = self.g.first_frame
        by_chain: dict[int, list[dict]] = {}
        for event in self.events:
            by_chain.setdefault(int(event["chain"]), []).append(event)
        interesting = set(by_chain)
        flagged_nodes = np.flatnonzero((level > 0) | uncertain)
        interesting |= {int(c) for c in np.unique(self.chain[flagged_nodes]) if c >= 0}
        interesting |= {c for (_d, c) in self.stops}
        rows = []
        for c in sorted(interesting, key=lambda c: int(self.frame[self.ch_start[c]])):
            nodes = self.nodes_of(c)
            lv = level[nodes]
            high = nodes[lv == LEVEL_HIGH]
            medium = nodes[lv == LEVEL_MEDIUM]
            code = int(evidence_level[nodes].max()) if nodes.size else 0
            if not auto:
                excluded_nodes = nodes[:0]
            else:
                excluded_nodes = nodes[(lv == LEVEL_HIGH) | (lv == LEVEL_MEDIUM)] if exclude_medium else high
            events = by_chain.get(c, [])
            merge = next((e for e in events if e["type"] == "merge"), None)
            split = next((e for e in events if e["type"] == "split"), None)
            fwd_ids = [int(x) for x in f_event[nodes] if x >= 0]
            bwd_ids = [int(x) for x in b_event[nodes] if x >= 0]
            reasons = []
            for event in events:
                if event["n"] < 2:
                    reasons += [f"{event['type']}@{event['frame']}:{r}" for r in event["reasons"]]
            for direction in ("forward", "backward"):
                if (direction, c) in self.stops:
                    reasons.append(f"{direction}_stop:{self.stops[(direction, c)]}")
            single = nodes[self.result_single[nodes] & ((nf[nodes] >= 2) | (nb[nodes] >= 2))]
            if single.size:
                reasons.append("result_single_conflict")
            if uncertain[nodes].any():
                reasons.append("uncertain_blobs=" + str(int(uncertain[nodes].sum())))
            if code and not auto:
                reasons.append("not_excluded:unreliable_overlap")
            if code and not (lv > 0).any():
                reasons.append("already_outlier")
            flagged = nodes[lv > 0]
            listed = flagged if flagged.size else nodes
            rows.append({
                "interval_id": len(rows) + 1,
                "level_code": code,
                "confidence": LEVEL_NAMES.get(code, "uncertain"),
                "excluded": "yes" if excluded_nodes.size else "no",
                "excluded_blobs": int(excluded_nodes.size),
                "high_blobs": int(high.size),
                "medium_blobs": int(medium.size),
                "frame_start": int(self.frame[nodes[0]]) + first,
                "frame_end": int(self.frame[nodes[-1]]) + first,
                "first_flagged_frame": int(self.frame[flagged[0]]) + first if flagged.size else -1,
                "n_forward": int(nf[nodes].max()),
                "n_backward": int(nb[nodes].max()),
                "merge_frame": merge["frame"] if merge else "",
                "merge_type": self.start_kind[c],
                "merge_parents": ";".join(self._ref(r["node"]) for r in merge["arms"]) if merge else "",
                "split_frame": split["frame"] if split else "",
                "split_type": self.end_kind[c],
                "split_children": ";".join(self._ref(r["node"]) for r in split["arms"]) if split else "",
                "forward_evidence": self._describe(fwd_ids[0] if fwd_ids else (merge["id"] if merge else -1)),
                "backward_evidence": self._describe(bwd_ids[-1] if bwd_ids else (split["id"] if split else -1)),
                "reasons": ";".join(dict.fromkeys(reasons)),
                "blobs": _blob_refs(self.frame[listed] + first, self.local[listed],
                                    limit=0 if flagged.size else 50),
            })
        return rows


def evaluate_temporal_merges(graph: TemporalMergeGraph, params: TemporalMergeParams,
                             area_bounds=None) -> TemporalMergeResult:
    """Classify merged-blob evidence on a stored graph (no frames are decoded).

    ``area_bounds`` re-derives the normal/outlier state of every node from its
    stored contour area. Only pass it for graphs whose blob geometry does not
    depend on the area bounds (no smoothing, Result import or Additional
    Outlier Set).
    """
    return _Evaluation(graph, params, area_bounds).run()
