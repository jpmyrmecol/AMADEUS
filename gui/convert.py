# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd


REAR_FRONT_SUFFIX = "_rear_front.csv"
CENTER_SUFFIX = "_center.csv"
ProgressCallback = Callable[[int, int, str], None]
LegacyPoseOrder = Literal["front_rear", "rear_front"]
DataKind = Literal["rear_front", "center", "keypoint_selection"]


class KeypointSelectionRequired(ValueError):
    """Raised when a pose format has no unambiguous front/rear names."""


@dataclass(frozen=True)
class TrackingFormatInfo:
    format_id: str
    label: str
    n_individuals: int
    keypoints: tuple[str, ...] = ()
    data_kind: DataKind = "center"
    front_keypoint: str | None = None
    rear_keypoint: str | None = None

    @property
    def requires_keypoint_selection(self) -> bool:
        return self.data_kind == "keypoint_selection"


@dataclass(frozen=True)
class ConversionOutputs:
    source: Path
    center: Path | None = None
    rear_front: Path | None = None

    def preferred(self, kind: Literal["rear_front", "center"]) -> Path | None:
        return self.rear_front if kind == "rear_front" else self.center


@dataclass(frozen=True)
class ConversionOptions:
    n_individuals: int | None = None
    front_keypoint: str | None = None
    rear_keypoint: str | None = None
    interpolate: bool = False
    reuse_existing: bool = False
    nan_threshold: float = 0.5
    legacy_order: LegacyPoseOrder | None = None


@dataclass(frozen=True)
class FormatAdapter:
    """One independently replaceable input-format branch."""

    format_ids: tuple[str, ...]
    inspect: Callable[[Path, int | None], TrackingFormatInfo]
    convert: Callable[[Path, TrackingFormatInfo, ConversionOptions], ConversionOutputs]


_FORMAT_ADAPTERS: dict[str, FormatAdapter] = {}


def register_format_adapter(adapter: FormatAdapter) -> None:
    """Register an adapter; new formats only need inspect/convert callables."""
    for format_id in adapter.format_ids:
        _FORMAT_ADAPTERS[format_id] = adapter


# Output paths and format detection

def converted_csv_path(source_path: str | Path, kind: Literal["rear_front", "center"]) -> Path:
    """Return the canonical converted path without duplicating its suffix."""
    source = Path(source_path)
    suffix = REAR_FRONT_SUFFIX if kind == "rear_front" else CENTER_SUFFIX
    lower_name = source.name.lower()
    if lower_name.endswith(suffix):
        return source
    stem = source.stem
    for known_suffix in ("_rear_front", "_center"):
        if stem.lower().endswith(known_suffix):
            stem = stem[:-len(known_suffix)]
            break
    return source.with_name(stem + suffix)


def rear_front_csv_path(source_path: str | Path) -> Path:
    return converted_csv_path(source_path, "rear_front")


def center_csv_path(source_path: str | Path) -> Path:
    return converted_csv_path(source_path, "center")


def detect_format(path: str | Path) -> str:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".h5", ".hdf5"):
        return "dlc_h5"
    if ext != ".csv":
        return "unknown"

    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return "unknown"
    cols = [str(c).strip() for c in header.columns]
    col_set = set(cols)
    if "track" in col_set and "frame_idx" in col_set:
        return "sleap_csv"
    if _result_obb_ids(cols):
        return "amadeus_result_csv"
    if _rear_front_ids(cols):
        return "rear_front_csv"
    if _center_ids(cols):
        return "center_csv"
    if _legacy_xy_ids(cols):
        if "time" in col_set or not ({"frame", "position"} & col_set):
            return "idtrackerai_csv"
        return "legacy_wide_csv"

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first_cell = handle.readline().split(",", 1)[0].strip().lower()
    except OSError:
        return "unknown"
    return "dlc_csv" if first_cell == "scorer" else "unknown"


def format_label(fmt: str) -> str:
    return {
        "sleap_csv": "SLEAP analysis CSV",
        "dlc_csv": "DeepLabCut analysis CSV",
        "dlc_h5": "DeepLabCut analysis H5",
        "rear_front_csv": "Standard rear/front CSV",
        "center_csv": "Standard center CSV",
        "amadeus_result_csv": "AMADEUS result CSV",
        "legacy_wide_csv": "Legacy wide CSV",
        "idtrackerai_csv": "idtracker.ai CSV",
        "unknown": "Unknown",
    }.get(fmt, fmt)


# SLEAP / DeepLabCut readers

def read_sleap_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Return a normalized SLEAP table and its keypoint names."""
    raw = pd.read_csv(path)
    missing = {"track", "frame_idx"} - set(raw.columns)
    if missing:
        raise RuntimeError(f"SLEAP CSV is missing columns: {sorted(missing)}")
    keypoints = sorted(
        {c[:-2] for c in raw.columns if str(c).endswith(".x")}
        & {c[:-2] for c in raw.columns if str(c).endswith(".y")}
    )
    if not keypoints:
        raise RuntimeError("No keypoints were found in the SLEAP CSV.")

    keep = ["track", "frame_idx"] + [f"{kp}.{axis}" for kp in keypoints for axis in ("x", "y")]
    df = raw[keep].copy()
    df["track_id"] = df["track"].map(_parse_track_id).astype(np.int32)
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="raise").astype(np.int32)
    for kp in keypoints:
        for axis in ("x", "y"):
            df[f"{kp}.{axis}"] = pd.to_numeric(df[f"{kp}.{axis}"], errors="coerce")
    return df.sort_values(["frame_idx", "track_id"]).reset_index(drop=True), keypoints


def infer_n_individuals_sleap(df: pd.DataFrame) -> int:
    per_frame = df.groupby("frame_idx")["track_id"].nunique()
    return int(per_frame.mode().iloc[0]) if len(per_frame) else 1


def read_dlc_file(path: Path, fmt: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return a DLC dataframe, keypoint names, and individual names."""
    df = _load_dlc_dataframe(path, fmt)
    while df.columns.nlevels > 1:
        first = [str(v).strip() for v in df.columns.get_level_values(0).unique()]
        if all(v.lower() in ("x", "y", "likelihood") for v in first):
            break
        if len(first) == 1:
            df = df.copy()
            df.columns = df.columns.droplevel(0)
            continue
        break

    if df.columns.nlevels == 3:
        individuals = sorted(
            (str(v).strip() for v in df.columns.get_level_values(0).unique()),
            key=_natural_sort_key,
        )
        keypoints = sorted(str(v).strip() for v in df.columns.get_level_values(1).unique())
    elif df.columns.nlevels == 2:
        individuals = []
        keypoints = sorted(str(v).strip() for v in df.columns.get_level_values(0).unique())
    else:
        raise RuntimeError(f"Unsupported DLC column structure (levels={df.columns.nlevels}).")
    return df, keypoints, individuals


def infer_n_individuals_dlc(individuals: list[str]) -> int:
    return len(individuals) if individuals else 1


def _load_dlc_dataframe(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "dlc_h5":
        return _load_dlc_h5(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [handle.readline() for _ in range(6)]
    is_multi = len(lines) > 1 and lines[1].split(",")[0].strip().lower() == "individuals"
    n_header = 4 if is_multi else 3
    first_data = lines[n_header].split(",")[0].strip() if len(lines) > n_header else ""
    n_index = 3 if first_data == "labeled-data" else 1
    df = pd.read_csv(path, header=list(range(n_header)), index_col=list(range(n_index)))
    if n_index == 3:
        df.index = pd.RangeIndex(len(df))
    return df


def _load_dlc_h5(path: Path) -> pd.DataFrame:
    try:
        import tables  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Reading DLC H5 files requires the 'tables' package.") from exc
    with pd.HDFStore(str(path), "r") as store:
        keys = store.keys()
        if not keys:
            raise RuntimeError(f"No dataframe was found in {path}.")
        key = next((k for k in keys if "df_with_missing" in k), keys[0])
        return store[key]


# Pose conversion

def sleap_to_rear_front(
    df: pd.DataFrame,
    keypoints: list[str],
    n_individuals: int,
    front_keypoint: str,
    rear_keypoint: str,
) -> pd.DataFrame:
    _require_keypoints(front_keypoint, rear_keypoint, keypoints)
    front_idx = keypoints.index(front_keypoint)
    rear_idx = keypoints.index(rear_keypoint)
    xy_cols = [f"{kp}.{axis}" for kp in keypoints for axis in ("x", "y")]
    frame_arr = df["frame_idx"].to_numpy(np.int32)
    track_arr = df["track_id"].to_numpy(np.int32)
    values = df[xy_cols].to_numpy(np.float64).reshape(len(df), len(keypoints), 2)
    max_frame = int(frame_arr.max()) if len(frame_arr) else -1
    data = np.full((max_frame + 1, n_individuals * 4), np.nan)
    id_to_slot = {
        track_id: slot
        for slot, track_id in enumerate(sorted(int(v) for v in np.unique(track_arr))[:n_individuals])
    }
    for row_idx in range(len(df)):
        slot = id_to_slot.get(int(track_arr[row_idx]))
        frame = int(frame_arr[row_idx])
        if slot is None or frame < 0:
            continue
        col = slot * 4
        data[frame, col:col + 2] = values[row_idx, front_idx]
        data[frame, col + 2:col + 4] = values[row_idx, rear_idx]
    return _build_rear_front_output(data)


def dlc_to_rear_front(
    df_wide: pd.DataFrame,
    keypoints: list[str],
    individuals: list[str],
    n_individuals: int,
    front_keypoint: str,
    rear_keypoint: str,
) -> pd.DataFrame:
    _require_keypoints(front_keypoint, rear_keypoint, keypoints)
    try:
        frames = pd.to_numeric(df_wide.index, errors="raise").astype(int).tolist()
    except Exception:
        frames = list(range(len(df_wide)))
    if any(frame < 0 for frame in frames):
        raise ValueError("Frame numbers must be non-negative.")
    max_frame = max(frames, default=-1)
    data = np.full((max_frame + 1, n_individuals * 4), np.nan)
    is_multi = bool(individuals)
    selected = sorted(individuals, key=_natural_sort_key)[:n_individuals] if is_multi else [""]
    for slot, individual in enumerate(selected):
        try:
            fx = _dlc_column(df_wide, individual if is_multi else None, front_keypoint, "x")
            fy = _dlc_column(df_wide, individual if is_multi else None, front_keypoint, "y")
            rx = _dlc_column(df_wide, individual if is_multi else None, rear_keypoint, "x")
            ry = _dlc_column(df_wide, individual if is_multi else None, rear_keypoint, "y")
        except KeyError as exc:
            raise RuntimeError(f"A selected DLC keypoint column is missing: {exc}") from exc
        for source_row, frame in enumerate(frames):
            col = slot * 4
            data[frame, col:col + 4] = (fx[source_row], fy[source_row], rx[source_row], ry[source_row])
    return _build_rear_front_output(data)


def rear_front_to_center(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate one center per animal as the front/rear midpoint."""
    pose = normalize_rear_front(df)
    columns: dict[str, object] = {"frame": pose["frame"].to_numpy(copy=True)}
    for animal_id in _rear_front_ids(list(pose.columns)):
        columns[f"X{animal_id}"] = (
            pd.to_numeric(pose[f"front_X{animal_id}"], errors="coerce").to_numpy(dtype=float, copy=False)
            + pd.to_numeric(pose[f"rear_X{animal_id}"], errors="coerce").to_numpy(dtype=float, copy=False)
        ) / 2.0
        columns[f"Y{animal_id}"] = (
            pd.to_numeric(pose[f"front_Y{animal_id}"], errors="coerce").to_numpy(dtype=float, copy=False)
            + pd.to_numeric(pose[f"rear_Y{animal_id}"], errors="coerce").to_numpy(dtype=float, copy=False)
        ) / 2.0
    return normalize_center(pd.DataFrame(columns))


def sleap_to_center(
    df: pd.DataFrame,
    keypoints: list[str],
    n_individuals: int,
    center_keypoint: str,
) -> pd.DataFrame:
    if center_keypoint not in keypoints:
        raise ValueError(f'Center keypoint "{center_keypoint}" was not found: {keypoints}')
    xy_cols = [f"{center_keypoint}.x", f"{center_keypoint}.y"]
    frames = df["frame_idx"].to_numpy(np.int32)
    tracks = df["track_id"].to_numpy(np.int32)
    values = df[xy_cols].to_numpy(np.float64)
    data = np.full((int(frames.max()) + 1 if len(frames) else 0, n_individuals * 2), np.nan)
    id_to_slot = {
        track_id: slot
        for slot, track_id in enumerate(sorted(int(v) for v in np.unique(tracks))[:n_individuals])
    }
    for row_idx, frame in enumerate(frames):
        slot = id_to_slot.get(int(tracks[row_idx]))
        if slot is not None and frame >= 0:
            data[int(frame), slot * 2:slot * 2 + 2] = values[row_idx]
    return _build_center_output(data)


def dlc_to_center(
    df_wide: pd.DataFrame,
    keypoints: list[str],
    individuals: list[str],
    n_individuals: int,
    center_keypoint: str,
) -> pd.DataFrame:
    if center_keypoint not in keypoints:
        raise ValueError(f'Center keypoint "{center_keypoint}" was not found: {keypoints}')
    try:
        frames = pd.to_numeric(df_wide.index, errors="raise").astype(int).tolist()
    except Exception:
        frames = list(range(len(df_wide)))
    if any(frame < 0 for frame in frames):
        raise ValueError("Frame numbers must be non-negative.")
    data = np.full((max(frames, default=-1) + 1, n_individuals * 2), np.nan)
    is_multi = bool(individuals)
    selected = sorted(individuals, key=_natural_sort_key)[:n_individuals] if is_multi else [""]
    for slot, individual in enumerate(selected):
        x = _dlc_column(df_wide, individual if is_multi else None, center_keypoint, "x")
        y = _dlc_column(df_wide, individual if is_multi else None, center_keypoint, "y")
        for source_row, frame in enumerate(frames):
            data[frame, slot * 2:slot * 2 + 2] = (x[source_row], y[source_row])
    return _build_center_output(data)


# Backward-compatible callable names for any local integrations.
sleap_to_front_rear = sleap_to_rear_front
dlc_to_front_rear = dlc_to_rear_front


# Extensible per-format dispatch

def inspect_tracking_file(
    path: str | Path,
    n_individuals: int | None = None,
) -> TrackingFormatInfo:
    """Inspect one input through its registered format adapter."""
    source = Path(path)
    format_id = detect_format(source)
    adapter = _FORMAT_ADAPTERS.get(format_id)
    if adapter is None:
        raise ValueError(f"Unsupported tracking format: {source} ({format_id})")
    return adapter.inspect(source, n_individuals)


def convert_tracking_outputs(
    path: str | Path,
    *,
    n_individuals: int | None = None,
    front_keypoint: str | None = None,
    rear_keypoint: str | None = None,
    interpolate: bool = False,
    reuse_existing: bool = False,
    nan_threshold: float = 0.5,
    legacy_order: LegacyPoseOrder | None = None,
) -> ConversionOutputs:
    """Convert one supported file to every applicable standard output.

    Two-point inputs produce both ``_rear_front.csv`` and ``_center.csv``.
    One-point inputs produce only ``_center.csv``.
    """
    source = Path(path)
    info = inspect_tracking_file(source, n_individuals)
    adapter = _FORMAT_ADAPTERS[info.format_id]
    options = ConversionOptions(
        n_individuals=n_individuals,
        front_keypoint=front_keypoint,
        rear_keypoint=rear_keypoint,
        interpolate=interpolate,
        reuse_existing=reuse_existing,
        nan_threshold=nan_threshold,
        legacy_order=legacy_order,
    )
    return adapter.convert(source, info, options)


def auto_front_rear_keypoints(keypoints: list[str] | tuple[str, ...]) -> tuple[str, str] | None:
    """Return exact (case-insensitive) ``front``/``rear`` names when present."""
    by_lower = {str(name).strip().lower(): str(name) for name in keypoints}
    if "front" in by_lower and "rear" in by_lower:
        return by_lower["front"], by_lower["rear"]
    return None


def _inspect_sleap(path: Path, requested_n: int | None) -> TrackingFormatInfo:
    df, keypoints = read_sleap_csv(path)
    n = requested_n or infer_n_individuals_sleap(df)
    return _pose_format_info("sleap_csv", n, keypoints)


def _inspect_dlc(path: Path, requested_n: int | None) -> TrackingFormatInfo:
    df, keypoints, individuals = read_dlc_file(path, detect_format(path))
    del df
    n = requested_n or infer_n_individuals_dlc(individuals)
    return _pose_format_info(detect_format(path), n, keypoints)


def _pose_format_info(format_id: str, n: int, keypoints: list[str]) -> TrackingFormatInfo:
    if not keypoints:
        raise ValueError("No bodyparts/keypoints were found.")
    named_pair = auto_front_rear_keypoints(keypoints)
    if named_pair is not None:
        front, rear = named_pair
        kind: DataKind = "rear_front"
    elif len(keypoints) == 1:
        front = rear = None
        kind = "center"
    else:
        front = rear = None
        kind = "keypoint_selection"
    return TrackingFormatInfo(
        format_id=format_id,
        label=format_label(format_id),
        n_individuals=int(n),
        keypoints=tuple(keypoints),
        data_kind=kind,
        front_keypoint=front,
        rear_keypoint=rear,
    )


def _convert_pose_source(
    path: Path,
    info: TrackingFormatInfo,
    options: ConversionOptions,
) -> ConversionOutputs:
    n = options.n_individuals or info.n_individuals
    center_path = center_csv_path(path)
    rear_front_path = rear_front_csv_path(path)
    if info.data_kind == "center":
        if options.reuse_existing and center_path.is_file():
            return ConversionOutputs(path, center=center_path)
        if info.format_id == "sleap_csv":
            df, keypoints = read_sleap_csv(path)
            center = sleap_to_center(df, keypoints, n, keypoints[0])
        else:
            df, keypoints, individuals = read_dlc_file(path, info.format_id)
            center = dlc_to_center(df, keypoints, individuals, n, keypoints[0])
        _interpolate_coordinates(center, options.interpolate)
        return ConversionOutputs(path, center=save_center_csv(center, center_path))

    front = options.front_keypoint or info.front_keypoint
    rear = options.rear_keypoint or info.rear_keypoint
    if not front or not rear:
        raise KeypointSelectionRequired(
            f"Select front and rear bodyparts from: {', '.join(info.keypoints)}"
        )
    if options.reuse_existing and rear_front_path.is_file() and center_path.is_file():
        return ConversionOutputs(path, center=center_path, rear_front=rear_front_path)
    if options.reuse_existing and rear_front_path.is_file():
        pose = load_rear_front_csv(rear_front_path)
    elif info.format_id == "sleap_csv":
        df, keypoints = read_sleap_csv(path)
        pose = sleap_to_rear_front(df, keypoints, n, front, rear)
    else:
        df, keypoints, individuals = read_dlc_file(path, info.format_id)
        pose = dlc_to_rear_front(df, keypoints, individuals, n, front, rear)
    _interpolate_coordinates(pose, options.interpolate)
    if not (options.reuse_existing and rear_front_path.is_file()):
        save_rear_front_csv(pose, rear_front_path)
    if not (options.reuse_existing and center_path.is_file()):
        save_center_csv(rear_front_to_center(pose), center_path)
    return ConversionOutputs(path, center=center_path, rear_front=rear_front_path)


def _inspect_umatracker(path: Path, requested_n: int | None) -> TrackingFormatInfo:
    header = pd.read_csv(path, nrows=0)
    point_ids = _legacy_xy_ids(list(header.columns))
    if not point_ids or point_ids != list(range(len(point_ids))):
        raise ValueError("UmaTracker wide CSV requires consecutive x0,y0,x1,y1,... columns.")
    point_count = len(point_ids)
    if requested_n is not None:
        if point_count == requested_n:
            kind: DataKind = "center"
            n = requested_n
        elif point_count == 2 * requested_n:
            kind = "rear_front"
            n = requested_n
        else:
            raise ValueError(
                f"{point_count} coordinate points do not match {requested_n} individuals "
                f"(expected {requested_n} or {2 * requested_n})."
            )
    else:
        kind = _infer_umatracker_kind(path, point_ids)
        n = point_count // 2 if kind == "rear_front" else point_count
    return TrackingFormatInfo(
        format_id="legacy_wide_csv",
        label="UmaTracker wide CSV",
        n_individuals=n,
        data_kind=kind,
    )


def _convert_umatracker(
    path: Path,
    info: TrackingFormatInfo,
    options: ConversionOptions,
) -> ConversionOutputs:
    df = pd.read_csv(path)
    center_path = center_csv_path(path)
    if info.data_kind == "center":
        if options.reuse_existing and center_path.is_file():
            return ConversionOutputs(path, center=center_path)
        center = legacy_wide_to_center(df)
        return ConversionOutputs(path, center=save_center_csv(center, center_path))

    rear_front_path = rear_front_csv_path(path)
    if options.reuse_existing and rear_front_path.is_file() and center_path.is_file():
        return ConversionOutputs(path, center=center_path, rear_front=rear_front_path)
    pose = legacy_wide_to_rear_front(
        df,
        options.legacy_order or _infer_legacy_pose_order(path),
    )
    _interpolate_coordinates(pose, options.interpolate)
    if not (options.reuse_existing and rear_front_path.is_file()):
        save_rear_front_csv(pose, rear_front_path)
    if not (options.reuse_existing and center_path.is_file()):
        save_center_csv(rear_front_to_center(pose), center_path)
    return ConversionOutputs(path, center=center_path, rear_front=rear_front_path)


def _inspect_idtrackerai(path: Path, requested_n: int | None) -> TrackingFormatInfo:
    header = pd.read_csv(path, nrows=0)
    point_count = len(_legacy_xy_ids(list(header.columns)))
    return TrackingFormatInfo(
        format_id="idtrackerai_csv",
        label=format_label("idtrackerai_csv"),
        n_individuals=requested_n or point_count,
        data_kind="center",
    )


def _convert_idtrackerai_adapter(
    path: Path,
    info: TrackingFormatInfo,
    options: ConversionOptions,
) -> ConversionOutputs:
    del info
    center = convert_idtrackerai_csv(
        path,
        nan_threshold=options.nan_threshold,
        reuse_existing=options.reuse_existing,
    )
    return ConversionOutputs(path, center=center)


def _inspect_standard_rear_front(path: Path, requested_n: int | None) -> TrackingFormatInfo:
    header = pd.read_csv(path, nrows=0)
    detected_n = len(_rear_front_ids(list(header.columns)))
    return TrackingFormatInfo(
        "rear_front_csv", format_label("rear_front_csv"), requested_n or detected_n,
        data_kind="rear_front",
    )


def _convert_standard_rear_front(
    path: Path,
    info: TrackingFormatInfo,
    options: ConversionOptions,
) -> ConversionOutputs:
    del info
    center_path = center_csv_path(path)
    if not (options.reuse_existing and center_path.is_file()):
        save_center_csv(rear_front_to_center(load_rear_front_csv(path)), center_path)
    return ConversionOutputs(path, center=center_path, rear_front=path)


def _inspect_standard_center(path: Path, requested_n: int | None) -> TrackingFormatInfo:
    header = pd.read_csv(path, nrows=0)
    detected_n = len(_center_ids(list(header.columns)))
    return TrackingFormatInfo(
        "center_csv", format_label("center_csv"), requested_n or detected_n,
        data_kind="center",
    )


def _convert_standard_center(
    path: Path,
    info: TrackingFormatInfo,
    options: ConversionOptions,
) -> ConversionOutputs:
    del info, options
    return ConversionOutputs(path, center=path)


def _infer_umatracker_kind(path: Path, point_ids: list[int]) -> DataKind:
    """Distinguish paired endpoints from centers using adjacent-neighbor geometry."""
    if len(point_ids) < 2 or len(point_ids) % 2:
        return "center"
    sample = pd.read_csv(path, nrows=300)
    points = np.stack(
        [sample[[f"x{i}", f"y{i}"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
         for i in point_ids],
        axis=1,
    )
    adjacent_is_nearest: list[float] = []
    stride = max(1, len(points) // 60)
    for row in points[::stride]:
        if not np.isfinite(row).all() or len(row) < 4:
            continue
        distances = np.linalg.norm(row[:, None, :] - row[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = np.argmin(distances, axis=1)
        expected_mate = np.arange(len(row)) ^ 1
        adjacent_is_nearest.append(float(np.mean(nearest == expected_mate)))
    if not adjacent_is_nearest:
        return "center"
    return "rear_front" if float(np.mean(adjacent_is_nearest)) >= 0.5 else "center"


def _interpolate_coordinates(df: pd.DataFrame, enabled: bool) -> None:
    if not enabled:
        return
    columns = [column for column in df.columns if column != "frame"]
    df[columns] = df[columns].interpolate(method="linear", limit_direction="both")


register_format_adapter(FormatAdapter(("sleap_csv",), _inspect_sleap, _convert_pose_source))
register_format_adapter(FormatAdapter(("dlc_csv", "dlc_h5"), _inspect_dlc, _convert_pose_source))
register_format_adapter(FormatAdapter(("legacy_wide_csv",), _inspect_umatracker, _convert_umatracker))
register_format_adapter(FormatAdapter(("idtrackerai_csv",), _inspect_idtrackerai, _convert_idtrackerai_adapter))
register_format_adapter(FormatAdapter(("rear_front_csv",), _inspect_standard_rear_front, _convert_standard_rear_front))
register_format_adapter(FormatAdapter(("center_csv",), _inspect_standard_center, _convert_standard_center))


# AMADEUS OBB + direction conversion

def infer_object_ids(columns: list[str]) -> list[int]:
    ids: set[int] = set()
    for name in map(str, columns):
        obb_match = re.fullmatch(r"x[0-3](\d+)", name)
        direction_match = re.fullmatch(r"c(\d+)", name)
        if obb_match:
            ids.add(int(obb_match.group(1)))
        if direction_match:
            ids.add(int(direction_match.group(1)))
    return sorted(ids)


def angle_deg_to_unit_vec_array(angle_deg: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(np.mod(angle_deg.astype(np.float64), 360.0))
    return np.stack([np.sin(theta), -np.cos(theta)], axis=1).astype(np.float32)


def convert_obb_direction_to_rear_front(
    obb_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    progress_cb: ProgressCallback | None = None,
    round_digits: int | None = 3,
) -> pd.DataFrame:
    """Convert OBB corners and direction angles to explicit front/rear points."""
    obb = _with_frame_column(obb_df, "OBB")
    direction = _with_frame_column(direction_df, "Direction")
    merged = pd.merge(obb, direction, on="frame", how="outer", sort=True, suffixes=("", "_dir"))
    merged["frame"] = pd.to_numeric(merged["frame"], errors="raise").astype(np.int32)
    if (merged["frame"] < 0).any():
        raise ValueError("Frame numbers must be non-negative.")
    merged = merged.sort_values("frame", kind="stable").drop_duplicates("frame", keep="last")
    merged = merged.set_index("frame").reindex(range(int(merged["frame"].max()) + 1)).reset_index()

    object_ids = infer_object_ids(list(merged.columns))
    if not object_ids:
        raise ValueError("No object IDs were found in the OBB/direction CSVs.")
    data = np.full((len(merged), len(object_ids) * 4), np.nan, dtype=np.float32)
    for slot, track_id in enumerate(object_ids):
        _convert_obb_id(merged, track_id, data, slot, round_digits)
        if progress_cb is not None:
            progress_cb(slot + 1, len(object_ids), f"Converting ID {track_id} ({slot + 1}/{len(object_ids)})")
    out = _build_rear_front_output(data)
    if progress_cb is not None:
        progress_cb(len(object_ids), len(object_ids), "Done")
    return out


def _convert_obb_id(
    merged: pd.DataFrame,
    track_id: int,
    data: np.ndarray,
    slot: int,
    round_digits: int | None,
) -> None:
    columns = [
        f"x0{track_id}", f"x1{track_id}", f"x2{track_id}", f"x3{track_id}",
        f"y0{track_id}", f"y1{track_id}", f"y2{track_id}", f"y3{track_id}",
    ]
    direction_col = f"c{track_id}"
    if not all(c in merged.columns for c in columns) or direction_col not in merged.columns:
        return
    values = merged[columns].to_numpy(dtype=np.float32, copy=False)
    directions = pd.to_numeric(merged[direction_col], errors="coerce").to_numpy(np.float32)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(directions)
    if not np.any(valid):
        return
    points = np.stack([values[valid, :4], values[valid, 4:]], axis=2)
    center = points.mean(axis=1, keepdims=True)
    next_points = np.roll(points, -1, axis=1)
    midpoints = 0.5 * (points + next_points)
    edges = next_points - points
    edge_length = np.linalg.norm(edges, axis=2)
    nonzero = edge_length > 0
    normals = np.full_like(edges, np.nan)
    normals[nonzero] = np.stack(
        [edges[..., 1][nonzero], -edges[..., 0][nonzero]], axis=1,
    ) / edge_length[nonzero, None]
    normals[np.sum((midpoints - center) * normals, axis=2) < 0] *= -1
    scores = np.sum(normals * angle_deg_to_unit_vec_array(directions[valid])[:, None, :], axis=2)
    scores[~nonzero] = np.nan
    row_ids = np.flatnonzero(valid)
    front = midpoints[np.arange(len(row_ids)), np.nanargmax(scores, axis=1)]
    rear = midpoints[np.arange(len(row_ids)), np.nanargmin(scores, axis=1)]
    if round_digits is not None:
        front = np.round(front, round_digits)
        rear = np.round(rear, round_digits)
    col = slot * 4
    data[row_ids, col:col + 2] = front
    data[row_ids, col + 2:col + 4] = rear


def amadeus_result_to_rear_front(
    result_df: pd.DataFrame,
    progress_cb: ProgressCallback | None = None,
    round_digits: int | None = 3,
) -> pd.DataFrame:
    """Convert final ``results/*.csv`` OBB rows into front/rear points.

    AMADEUS result files store one row per frame and ``cxN``, ``cyN``, ``wN``,
    ``hN``, and ``headingN`` for each track.  ``headingN`` is the unit-vector
    direction used by the tracking result, so the front/rear endpoints are
    derived from the long OBB axis without modifying the source CSV.
    """
    normalized = _normalize_frame_column(result_df)
    object_ids = _result_obb_ids(list(normalized.columns))
    if not object_ids:
        raise ValueError(
            "The AMADEUS result CSV must contain cxN, cyN, wN, hN, and headingN columns."
        )
    if object_ids != list(range(len(object_ids))):
        raise ValueError(f"AMADEUS result IDs must be consecutive from 0; found {object_ids}.")

    data = np.full((len(normalized), len(object_ids) * 4), np.nan, dtype=np.float32)
    for slot, track_id in enumerate(object_ids):
        columns = {
            name: pd.to_numeric(normalized[f"{name}{track_id}"], errors="coerce").to_numpy(
                dtype=np.float64,
                copy=False,
            )
            for name in ("cx", "cy", "w", "h", "heading")
        }
        valid = (
            np.isfinite(columns["cx"])
            & np.isfinite(columns["cy"])
            & np.isfinite(columns["w"])
            & np.isfinite(columns["h"])
            & np.isfinite(columns["heading"])
            & (columns["w"] > 0)
            & (columns["h"] > 0)
        )
        if np.any(valid):
            direction = angle_deg_to_unit_vec_array(columns["heading"][valid])
            half_length = (columns["w"][valid] * 0.5)[:, None]
            centers = np.column_stack((columns["cx"][valid], columns["cy"][valid]))
            front = centers + direction * half_length
            rear = centers - direction * half_length
            if round_digits is not None:
                front = np.round(front, round_digits)
                rear = np.round(rear, round_digits)
            row_ids = np.flatnonzero(valid)
            col = slot * 4
            data[row_ids, col:col + 2] = front
            data[row_ids, col + 2:col + 4] = rear
        if progress_cb is not None:
            progress_cb(slot + 1, len(object_ids), f"Converting ID {track_id} ({slot + 1}/{len(object_ids)})")
    if progress_cb is not None:
        progress_cb(len(object_ids), len(object_ids), "Done")
    return _build_rear_front_output(data)


def _inspect_amadeus_result(path: Path, requested_n: int | None) -> TrackingFormatInfo:
    header = pd.read_csv(path, nrows=0)
    object_ids = _result_obb_ids(list(header.columns))
    if not object_ids:
        raise ValueError(
            "AMADEUS result CSV must contain cxN, cyN, wN, hN, and headingN columns."
        )
    if object_ids != list(range(len(object_ids))):
        raise ValueError(f"AMADEUS result IDs must be consecutive from 0; found {object_ids}.")
    if requested_n is not None and int(requested_n) != len(object_ids):
        raise ValueError(
            f"AMADEUS result CSV contains {len(object_ids)} objects, not {requested_n}."
        )
    return TrackingFormatInfo(
        "amadeus_result_csv",
        format_label("amadeus_result_csv"),
        len(object_ids),
        data_kind="rear_front",
    )


def _convert_amadeus_result(
    path: Path,
    info: TrackingFormatInfo,
    options: ConversionOptions,
) -> ConversionOutputs:
    del info
    rear_front_path = rear_front_csv_path(path)
    center_path = center_csv_path(path)
    if options.reuse_existing and rear_front_path.is_file() and center_path.is_file():
        return ConversionOutputs(path, center=center_path, rear_front=rear_front_path)
    pose = amadeus_result_to_rear_front(pd.read_csv(path))
    if not (options.reuse_existing and rear_front_path.is_file()):
        save_rear_front_csv(pose, rear_front_path)
    if not (options.reuse_existing and center_path.is_file()):
        save_center_csv(rear_front_to_center(pose), center_path)
    return ConversionOutputs(path, center=center_path, rear_front=rear_front_path)


register_format_adapter(FormatAdapter(("amadeus_result_csv",), _inspect_amadeus_result, _convert_amadeus_result))


# idtracker.ai center conversion


def convert_idtrackerai_csv(
    input_path: str | Path,
    log_fn: Callable[[str], None] | None = None,
    nan_threshold: float = 0.5,
    out_path: str | Path | None = None,
    reuse_existing: bool = True,
) -> Path:
    source = Path(input_path)
    output = Path(out_path) if out_path is not None else center_csv_path(source)
    if reuse_existing and output.is_file():
        if log_fn:
            log_fn(f"using existing: {output}")
        return output
    df = pd.read_csv(source)
    if "time" in df.columns:
        if log_fn:
            log_fn("dropping 'time' column")
        df = df.drop(columns=["time"])
    pairs = [(f"x{i}", f"y{i}") for i in _legacy_xy_ids(list(df.columns))]
    if not pairs:
        raise ValueError("No x/y coordinate columns found in idtracker.ai CSV.")
    kept: list[tuple[str, str]] = []
    for x_col, y_col in pairs:
        missing_ratio = max(df[x_col].isna().mean(), df[y_col].isna().mean())
        if missing_ratio > nan_threshold:
            if log_fn:
                log_fn(f"dropping {x_col}/{y_col} (NaN {missing_ratio:.1%})")
        else:
            kept.append((x_col, y_col))
    if not kept:
        raise ValueError("All animals exceed the NaN threshold; nothing to convert.")

    columns: dict[str, object] = {"frame": np.arange(len(df), dtype=int)}
    for animal_id, (x_col, y_col) in enumerate(kept):
        columns[f"X{animal_id}"] = pd.to_numeric(df[x_col], errors="coerce").to_numpy(copy=True)
        columns[f"Y{animal_id}"] = pd.to_numeric(df[y_col], errors="coerce").to_numpy(copy=True)
    result = save_center_csv(pd.DataFrame(columns), output)
    if log_fn:
        log_fn(f"saved: {result}")
    return result


# Standard/legacy adapters used at GUI boundaries

def load_rear_front_csv(
    path: str | Path,
    legacy_order: LegacyPoseOrder | None = None,
) -> pd.DataFrame:
    source = Path(path)
    df = pd.read_csv(source)
    if _rear_front_ids(list(df.columns)):
        return normalize_rear_front(df)
    order = legacy_order or _infer_legacy_pose_order(source)
    return legacy_wide_to_rear_front(df, order)


def load_wide_for_gui(
    path: str | Path,
    standard_front_first: bool,
) -> pd.DataFrame:
    """Load standard formats as legacy wide data for existing GUI internals.

    A genuinely legacy file is left in its original point order.  This keeps
    old projects readable while making the interpretation of new named-column
    files unambiguous.
    """
    source = Path(path)
    df = pd.read_csv(source)
    if _rear_front_ids(list(df.columns)):
        return rear_front_to_legacy_wide(normalize_rear_front(df), standard_front_first)
    if _center_ids(list(df.columns)):
        return center_to_legacy_wide(normalize_center(df))
    return _normalize_legacy_wide(df)


def rear_front_to_legacy_wide(
    df: pd.DataFrame,
    front_first: bool = True,
    index_col: str = "position",
) -> pd.DataFrame:
    standard = normalize_rear_front(df)
    columns: dict[str, object] = {index_col: standard["frame"].to_numpy(copy=True)}
    for animal_id in _rear_front_ids(list(standard.columns)):
        pairs = (("front", 2 * animal_id), ("rear", 2 * animal_id + 1))
        if not front_first:
            pairs = (("rear", 2 * animal_id), ("front", 2 * animal_id + 1))
        for point, point_id in pairs:
            columns[f"x{point_id}"] = standard[f"{point}_X{animal_id}"].to_numpy(copy=True)
            columns[f"y{point_id}"] = standard[f"{point}_Y{animal_id}"].to_numpy(copy=True)
    return pd.DataFrame(columns)


def center_to_legacy_wide(df: pd.DataFrame, index_col: str = "position") -> pd.DataFrame:
    standard = normalize_center(df)
    columns: dict[str, object] = {index_col: standard["frame"].to_numpy(copy=True)}
    for animal_id in _center_ids(list(standard.columns)):
        columns[f"x{animal_id}"] = standard[f"X{animal_id}"].to_numpy(copy=True)
        columns[f"y{animal_id}"] = standard[f"Y{animal_id}"].to_numpy(copy=True)
    return pd.DataFrame(columns)


def legacy_wide_to_rear_front(df: pd.DataFrame, order: LegacyPoseOrder) -> pd.DataFrame:
    legacy = _normalize_legacy_wide(df)
    point_ids = _legacy_xy_ids(list(legacy.columns))
    if len(point_ids) % 2:
        raise ValueError("A rear/front CSV must contain two coordinate pairs per animal.")
    columns: dict[str, object] = {"frame": legacy["position"].to_numpy(copy=True)}
    for animal_id in range(len(point_ids) // 2):
        first, second = 2 * animal_id, 2 * animal_id + 1
        front_id, rear_id = (first, second) if order == "front_rear" else (second, first)
        columns[f"front_X{animal_id}"] = legacy[f"x{front_id}"].to_numpy(copy=True)
        columns[f"front_Y{animal_id}"] = legacy[f"y{front_id}"].to_numpy(copy=True)
        columns[f"rear_X{animal_id}"] = legacy[f"x{rear_id}"].to_numpy(copy=True)
        columns[f"rear_Y{animal_id}"] = legacy[f"y{rear_id}"].to_numpy(copy=True)
    return normalize_rear_front(pd.DataFrame(columns))


def legacy_wide_to_center(df: pd.DataFrame) -> pd.DataFrame:
    legacy = _normalize_legacy_wide(df)
    columns: dict[str, object] = {"frame": legacy["position"].to_numpy(copy=True)}
    for animal_id in _legacy_xy_ids(list(legacy.columns)):
        columns[f"X{animal_id}"] = legacy[f"x{animal_id}"].to_numpy(copy=True)
        columns[f"Y{animal_id}"] = legacy[f"y{animal_id}"].to_numpy(copy=True)
    return normalize_center(pd.DataFrame(columns))


def normalize_rear_front(df: pd.DataFrame) -> pd.DataFrame:
    ids = _rear_front_ids(list(df.columns))
    if not ids:
        raise ValueError("No front_XN/front_YN/rear_XN/rear_YN columns were found.")
    expected_ids = list(range(len(ids)))
    if ids != expected_ids:
        raise ValueError(f"Rear/front IDs must be consecutive from 0; found {ids}.")
    normalized = _normalize_frame_column(df)
    cols = ["frame"]
    for animal_id in ids:
        cols += [
            f"front_X{animal_id}", f"front_Y{animal_id}",
            f"rear_X{animal_id}", f"rear_Y{animal_id}",
        ]
    return normalized[cols]


def normalize_center(df: pd.DataFrame) -> pd.DataFrame:
    ids = _center_ids(list(df.columns))
    if not ids:
        raise ValueError("No XN/YN center columns were found.")
    if ids != list(range(len(ids))):
        raise ValueError(f"Center IDs must be consecutive from 0; found {ids}.")
    normalized = _normalize_frame_column(df)
    cols = ["frame"]
    for animal_id in ids:
        cols += [f"X{animal_id}", f"Y{animal_id}"]
    return normalized[cols]


def save_rear_front_csv(
    df: pd.DataFrame,
    path: str | Path,
    float_format: str | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalize_rear_front(df).to_csv(output, index=False, float_format=float_format)
    return output


def save_center_csv(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalize_center(df).to_csv(output, index=False)
    return output


# Internal helpers

def _rear_front_ids(columns: list[object]) -> list[int]:
    col_set = set(map(str, columns))
    ids = sorted(
        int(match.group(1))
        for col in col_set
        if (match := re.fullmatch(r"front_X(\d+)", col))
    )
    return [
        animal_id for animal_id in ids
        if {f"front_Y{animal_id}", f"rear_X{animal_id}", f"rear_Y{animal_id}"} <= col_set
    ]


def _result_obb_ids(columns: list[object]) -> list[int]:
    col_set = set(map(str, columns))
    ids = sorted(
        int(match.group(1))
        for col in col_set
        if (match := re.fullmatch(r"cx(\d+)", col))
    )
    return [
        track_id for track_id in ids
        if {f"cy{track_id}", f"w{track_id}", f"h{track_id}", f"heading{track_id}"} <= col_set
    ]


def _center_ids(columns: list[object]) -> list[int]:
    col_set = set(map(str, columns))
    return sorted(
        int(match.group(1))
        for col in col_set
        if (match := re.fullmatch(r"X(\d+)", col)) and f"Y{match.group(1)}" in col_set
    )


def _legacy_xy_ids(columns: list[object]) -> list[int]:
    col_set = set(map(str, columns))
    return sorted(
        int(match.group(1))
        for col in col_set
        if (match := re.fullmatch(r"x(\d+)", col)) and f"y{match.group(1)}" in col_set
    )


def _normalize_frame_column(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "frame" not in result.columns:
        if "position" in result.columns:
            result = result.rename(columns={"position": "frame"})
        else:
            raise ValueError('Expected a "frame" column.')
    result["frame"] = pd.to_numeric(result["frame"], errors="raise").astype(int)
    if (result["frame"] < 0).any():
        raise ValueError("Frame numbers must be non-negative.")
    result = result.sort_values("frame", kind="stable").drop_duplicates("frame", keep="last")
    max_frame = int(result["frame"].max()) if len(result) else -1
    result = result.set_index("frame").reindex(range(max_frame + 1)).reset_index()
    result["frame"] = result["frame"].astype(int)
    return result


def _normalize_legacy_wide(df: pd.DataFrame) -> pd.DataFrame:
    point_ids = _legacy_xy_ids(list(df.columns))
    if not point_ids or point_ids != list(range(len(point_ids))):
        raise ValueError("Expected consecutive x0,y0,x1,y1,... coordinate columns.")
    normalized = _normalize_frame_column(df).rename(columns={"frame": "position"})
    cols = ["position"]
    for point_id in point_ids:
        cols += [f"x{point_id}", f"y{point_id}"]
    return normalized[cols]


def _build_rear_front_output(data: np.ndarray) -> pd.DataFrame:
    if data.ndim != 2 or data.shape[1] % 4:
        raise ValueError("Pose data must contain four values per animal.")
    columns = ["frame"]
    for animal_id in range(data.shape[1] // 4):
        columns += [
            f"front_X{animal_id}", f"front_Y{animal_id}",
            f"rear_X{animal_id}", f"rear_Y{animal_id}",
        ]
    return pd.DataFrame(
        np.column_stack([np.arange(len(data), dtype=int), data]),
        columns=columns,
    ).astype({"frame": int})


def _build_center_output(data: np.ndarray) -> pd.DataFrame:
    if data.ndim != 2 or data.shape[1] % 2:
        raise ValueError("Center data must contain two values per animal.")
    columns = ["frame"]
    for animal_id in range(data.shape[1] // 2):
        columns += [f"X{animal_id}", f"Y{animal_id}"]
    return pd.DataFrame(
        np.column_stack([np.arange(len(data), dtype=int), data]),
        columns=columns,
    ).astype({"frame": int})


def _with_frame_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    result = df.copy()
    if "frame" not in result.columns:
        if "position" in result.columns:
            result = result.rename(columns={"position": "frame"})
        else:
            raise ValueError(f'{label} CSV must contain a "frame" or "position" column.')
    return result


def _parse_track_id(value: object) -> int:
    match = re.search(r"(\d+)$", str(value))
    if match is None:
        raise ValueError(f"Track ID must end with a number: {value}")
    return int(match.group(1))


def _require_keypoints(front: str, rear: str, keypoints: list[str]) -> None:
    if front not in keypoints:
        raise ValueError(f'Front keypoint "{front}" was not found: {keypoints}')
    if rear not in keypoints:
        raise ValueError(f'Rear keypoint "{rear}" was not found: {keypoints}')
    if front == rear:
        raise ValueError("Front and rear keypoints must be different.")


def _dlc_column(
    df: pd.DataFrame,
    individual: str | None,
    bodypart: str,
    coordinate: str,
) -> np.ndarray:
    key = (individual, bodypart, coordinate) if individual is not None else (bodypart, coordinate)
    return pd.to_numeric(df[key], errors="coerce").to_numpy(np.float64)


def _infer_legacy_pose_order(path: Path) -> LegacyPoseOrder:
    stem = path.stem.lower()
    if stem.startswith("front_rear_from_obb_direction") or "_rear_front" in stem:
        return "rear_front"
    return "front_rear"


def _natural_sort_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )
