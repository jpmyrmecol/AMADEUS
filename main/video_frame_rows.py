# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Bounded-memory CSV readers and color allocation for Create Video."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


class CsvFrameRows:
    """Read ordered CSV rows incrementally, retaining at most one chunk."""

    def __init__(
        self,
        path: str,
        on_row: Callable[[int, dict], None] | None = None,
        chunk_size: int = 64,
        selection: range | None = None,
    ) -> None:
        self._chunks = iter(pd.read_csv(path, chunksize=chunk_size))
        self._rows = iter(())
        self._columns: list[str] = []
        self._position_index = -1
        self._pending: tuple[int, tuple] | None = None
        self._last_position: int | None = None
        self._on_row = on_row
        self.count = 0
        self.selected_count = 0
        self._selection = selection
        self._advance()

    def _advance(self) -> None:
        while self._pending is None:
            try:
                values = next(self._rows)
            except StopIteration:
                try:
                    chunk = next(self._chunks)
                except StopIteration:
                    return
                self._columns = list(chunk.columns)
                if "position" not in self._columns:
                    raise ValueError("Tracking CSV has no position column.")
                self._position_index = self._columns.index("position")
                self._rows = iter(chunk.itertuples(index=False, name=None))
                continue
            position = pd.to_numeric(values[self._position_index], errors="coerce")
            if pd.isna(position):
                continue
            frame = int(position)
            if self._last_position is not None and frame <= self._last_position:
                raise ValueError("Tracking CSV frame positions must be strictly increasing.")
            self._last_position = frame
            self._pending = (frame, values)

    def _consume(self, *, deliver: bool, observe: bool) -> dict | None:
        assert self._pending is not None
        frame, values = self._pending
        self._pending = None
        self.count += 1
        if self._selection is not None and frame in self._selection:
            self.selected_count += 1
        row = dict(zip(self._columns, values)) if deliver or (observe and self._on_row) else None
        if observe and self._on_row is not None:
            self._on_row(frame, row)
        self._advance()
        return row if deliver else None

    def take(self, frame: int) -> dict | None:
        while self._pending is not None and self._pending[0] < frame:
            self._consume(deliver=False, observe=True)
        if self._pending is not None and self._pending[0] == frame:
            return self._consume(deliver=True, observe=True)
        return None

    def finish(self) -> None:
        """Count remaining CSV frames without allocating row dictionaries."""
        while self._pending is not None:
            self._consume(deliver=False, observe=False)


def peak_visible_obb_count(path: str, track_cols: dict) -> int:
    """Find peak concurrent occupancy using small numeric CSV chunks."""
    names = [cols.x_cols + cols.y_cols for cols in track_cols.values()]
    header = set(pd.read_csv(path, nrows=0).columns)
    complete = [columns for columns in names if all(name in header for name in columns)]
    if not complete:
        return 0
    ordered = [name for group in complete for name in group]
    peak = 0
    for chunk in pd.read_csv(path, usecols=["position", *ordered], chunksize=64):
        valid_position = pd.to_numeric(chunk["position"], errors="coerce").notna().to_numpy()
        if not valid_position.any():
            continue
        coords = chunk[ordered].to_numpy(dtype=float, copy=False)
        visible = np.isfinite(coords).reshape(len(chunk), len(complete), 8).all(axis=2)
        peak = max(peak, int(visible[valid_position].sum(axis=1).max()))
    return peak


class StreamingColorSlots:
    """Assign the same slots as variable_population.allocate_color_slots."""

    def __init__(self, peak: int) -> None:
        self._free = list(range(peak))
        heapq.heapify(self._free)
        self._active: dict[int, int] = {}
        self._previous: dict[int, int] = {}

    def observe(self, present: Iterable[int]) -> None:
        present_set = set(present)
        for tid in list(self._active):
            if tid not in present_set:
                heapq.heappush(self._free, self._active.pop(tid))
        for tid in sorted(present_set):
            if tid in self._active:
                continue
            preferred = self._previous.get(tid)
            if preferred in self._free:
                self._free.remove(preferred)
                heapq.heapify(self._free)
                slot = preferred
            else:
                slot = heapq.heappop(self._free)
            self._active[tid] = self._previous[tid] = slot

    def current(self) -> dict[int, int]:
        return dict(self._active)


def indexed_assign_types(path: str) -> pd.DataFrame:
    """Index label metadata without expanding the whole table to row dictionaries."""
    rows = pd.read_pickle(path)
    if "position" not in rows.columns:
        return pd.DataFrame()
    rows["position"] = pd.to_numeric(rows["position"], errors="coerce")
    rows = rows.dropna(subset=["position"])
    rows["position"] = rows["position"].astype(int)
    return rows.set_index("position", verify_integrity=True)
