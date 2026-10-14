# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
"""Versioned, non-pickle on-disk cache for Segmentation temporal merge Search.

The graph and decisions are NumPy numeric arrays plus UTF-8 JSON inside an
atomic .npz archive. Loading always uses allow_pickle=False. Video and GUI
setting fingerprints must match before any saved result is applied.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from zipfile import BadZipFile

import numpy as np

from main.temporal_merge_blobs import TemporalMergeGraph, TemporalMergeParams, TemporalMergeResult


CACHE_SCHEMA = 1
GRAPH_COLUMNS = (
    "offsets", "area", "contour_area", "cx", "cy", "length", "width",
    "theta", "flags", "src", "dst", "overlap", "digests", "breaks",
)
GRAPH_DTYPES = {
    "offsets": np.dtype("int64"), "area": np.dtype("int32"),
    "contour_area": np.dtype("float32"), "cx": np.dtype("float32"),
    "cy": np.dtype("float32"), "length": np.dtype("float32"),
    "width": np.dtype("float32"), "theta": np.dtype("float32"),
    "flags": np.dtype("uint8"), "src": np.dtype("int32"),
    "dst": np.dtype("int32"), "overlap": np.dtype("int32"),
    "digests": np.dtype("uint64"), "breaks": np.dtype("bool"),
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode cache value of type {type(value).__name__}")


def save_temporal_merge_cache(
    path: str, graph: TemporalMergeGraph, result: TemporalMergeResult,
    params: TemporalMergeParams, fingerprint: str,
) -> None:
    """Save an already verified Search atomically, without executable objects."""
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("A Search fingerprint is required.")
    meta = {
        "schema": CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "params": asdict(params),
        "first_frame": int(graph.first_frame),
        "image_shape": list(graph.image_shape),
        "levels": result.levels,
        "animals": result.animals,
        "excluded": {str(k): sorted(map(int, v)) for k, v in result.excluded.items()},
        "intervals": result.intervals,
        "diagnostics": result.diagnostics,
        "counts": result.counts,
    }
    encoded = json.dumps(
        meta, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    arrays = {name: np.asarray(getattr(graph, name)) for name in GRAPH_COLUMNS}
    arrays["metadata"] = np.frombuffer(encoded, dtype=np.uint8)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.isfile(temporary):
            os.remove(temporary)


def load_temporal_merge_cache(
    path: str, fingerprint: str, params: TemporalMergeParams,
) -> tuple[TemporalMergeGraph, TemporalMergeResult] | None:
    """Return cached graph/result if valid and unchanged; otherwise None.

    A corrupt or stale cache is ignored. Numeric arrays are checked against
    the expected schema to avoid loading arbitrary Python objects.
    """
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(GRAPH_COLUMNS) | {"metadata"}:
                return None
            metadata = archive["metadata"]
            if metadata.dtype != np.dtype("uint8") or metadata.ndim != 1:
                return None
            meta = json.loads(metadata.tobytes().decode("utf-8"))
            if (meta.get("schema") != CACHE_SCHEMA
                    or meta.get("fingerprint") != fingerprint
                    or meta.get("params") != asdict(params)):
                return None
            arrays = {}
            for name in GRAPH_COLUMNS:
                value = archive[name]
                if value.ndim != 1 or value.dtype != GRAPH_DTYPES[name]:
                    return None
                arrays[name] = value
            offsets = arrays["offsets"]
            total = int(offsets[-1]) if offsets.size else -1
            frames = int(offsets.size - 1)
            if (frames < 1 or offsets[0] != 0 or total < 0
                    or np.any(offsets[1:] < offsets[:-1])
                    or arrays["digests"].size != frames
                    or arrays["breaks"].size != frames - 1
                    or any(arrays[k].size != total for k in (
                        "area", "contour_area", "cx", "cy", "length",
                        "width", "theta", "flags"))
                    or any(arrays[k].size != arrays["src"].size for k in ("dst", "overlap"))
                    or (arrays["src"].size and (
                        arrays["src"].min() < 0 or arrays["dst"].min() < 0
                        or arrays["src"].max() >= total or arrays["dst"].max() >= total))):
                return None
            shape = tuple(meta["image_shape"])
            if len(shape) != 2 or min(shape) <= 0:
                return None
            first = int(meta["first_frame"])
            graph = TemporalMergeGraph(
                first_frame=first, image_shape=shape, **arrays,
            )
            levels = {int(f): {int(i): int(n) for i, n in row.items()}
                      for f, row in meta["levels"].items()}
            excluded = {int(f): set(map(int, indices))
                        for f, indices in meta["excluded"].items()}
            animals = {int(f): {int(i): int(n) for i, n in row.items()}
                       for f, row in meta["animals"].items()}
            for mapping in (levels, excluded, animals):
                for frame, selected in mapping.items():
                    offset = int(frame) - first
                    if not 0 <= offset < frames:
                        return None
                    count = int(offsets[offset + 1] - offsets[offset])
                    if any(not 0 <= int(index) < count for index in selected):
                        return None
            if any(n < 2 for row in animals.values() for n in row.values()):
                return None
            result = TemporalMergeResult(
                levels=levels, excluded=excluded,
                intervals=meta["intervals"], diagnostics=meta["diagnostics"],
                counts=meta["counts"], first_frame=first,
                digests=arrays["digests"], animals=animals,
            )
            return graph, result
    except (OSError, ValueError, TypeError, KeyError, IndexError,
            OverflowError, UnicodeError, BadZipFile, EOFError, RuntimeError):
        return None
