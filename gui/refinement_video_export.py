# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Export Refinement arrows and optional AMADEUS OBB geometry to video."""

from __future__ import annotations

import os
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from gui import convert as _convert
from gui.color import WHITE_RGB


def prepare_refined_obb_rows(
    result_df: pd.DataFrame,
    pose_df: pd.DataFrame | None,
    num_ids: int,
    position_edited_keys: set[tuple[int, int]],
    direction_reversed_keys: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot the OBBs that would be saved, including unsaved Refinement edits.

    ID swaps already update result_df in the GUI. Direction flips and manually
    moved arrows are applied through the same conversion as Save As/Ctrl+S.
    """
    if position_edited_keys or direction_reversed_keys:
        if pose_df is None:
            raise ValueError("Edited Refinement OBBs require a pose snapshot.")
        rear_front = _convert.legacy_wide_to_rear_front(pose_df, "front_rear")
        result_df = _convert.refine_amadeus_result_from_rear_front(
            result_df, rear_front, position_edited_keys, direction_reversed_keys
        )

    columns = [
        f"{name}{tid}"
        for tid in range(num_ids)
        for name in ("cx", "cy", "w", "h", "heading")
    ]
    if "frame" not in result_df.columns or not set(columns).issubset(result_df.columns):
        raise ValueError("AMADEUS OBB columns do not match the loaded Refinement IDs.")

    frames = result_df["frame"].to_numpy(dtype=np.int64, copy=True)
    values = result_df[columns].to_numpy(dtype=np.float64, copy=True).reshape(-1, num_ids, 5)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    values = values[order]
    if len(frames) > 1 and np.any(np.diff(frames) == 0):
        raise ValueError("AMADEUS result contains duplicate frame numbers.")
    return frames, values


def obb_corners(record: np.ndarray) -> np.ndarray | None:
    """OBB corners using AMADEUS heading (0=up, positive=counterclockwise)."""
    cx, cy, width, height, heading = np.asarray(record, dtype=float)
    if not np.isfinite([cx, cy, width, height, heading]).all() or width <= 0 or height <= 0:
        return None

    theta = np.deg2rad(heading)
    major = np.array([np.sin(theta), -np.cos(theta)]) * (width / 2.0)
    minor = np.array([-np.cos(theta), -np.sin(theta)]) * (height / 2.0)
    center = np.array([cx, cy], dtype=float)
    corners = np.stack([
        center - major - minor,
        center - major + minor,
        center + major + minor,
        center + major - minor,
    ])
    return np.rint(corners).astype(np.int32).reshape(4, 1, 2)


def draw_refined_obbs(
    image: np.ndarray,
    frame_obbs: np.ndarray,
    id_palette: list[tuple[int, int, int]],
    thickness: int = 2,
) -> None:
    """Draw only valid boxes. The image and palette both use RGB ordering."""
    for tid, record in enumerate(frame_obbs):
        corners = obb_corners(record)
        if corners is None:
            continue
        color = id_palette[tid] if tid < len(id_palette) else WHITE_RGB
        cv2.polylines(image, [corners], True, color, thickness, cv2.LINE_AA)


def export_refinement_video(
    video_path: str,
    out_path: str,
    start_frame: int,
    end_frame: int,
    speed: float,
    output_fps: float,
    df_np: np.ndarray,
    front_x_idx: np.ndarray,
    front_y_idx: np.ndarray,
    rear_x_idx: np.ndarray,
    rear_y_idx: np.ndarray,
    id_palette: list,
    num_ids: int,
    arrow_width: int,
    label_size: int,
    show_arrows: bool,
    show_labels: bool,
    reverse: bool,
    progress_cb: Callable[[int, int, str], None] | None,
    *,
    show_obbs: bool = False,
    obb_frames: np.ndarray | None = None,
    obb_values: np.ndarray | None = None,
) -> None:
    """Write the existing arrow overlay and, optionally, corrected OBB outlines."""
    if show_obbs:
        if obb_frames is None or obb_values is None:
            raise ValueError("OBB export requires AMADEUS OBB data.")
        if obb_values.shape != (len(obb_frames), num_ids, 5):
            raise ValueError("Refinement OBB data does not match the selected individuals.")

    pose_row_for_frame = {
        int(frame): i
        for i, frame in enumerate(df_np[:, 0])
        if np.isfinite(frame)
    }
    obb_row_for_frame = (
        {int(frame): i for i, frame in enumerate(obb_frames)}
        if show_obbs else {}
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {start_frame}.")
        h, w = first.shape[:2]

        total_source = end_frame - start_frame + 1
        n_output = max(1, int(round(total_source / speed)))
        source_indices = [
            min(end_frame, start_frame + int(round(j * speed)))
            for j in range(n_output)
        ]

        ext = os.path.splitext(out_path)[1].lower()
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if ext == ".avi" else "mp4v"))
        writer = cv2.VideoWriter(out_path, fourcc, output_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video file: {out_path}")

        total = len(source_indices)
        tip_length = max(8, int(round(arrow_width * 5.0)))
        prev_idx = -1
        try:
            for done, src_idx in enumerate(source_indices):
                if src_idx != prev_idx + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot read frame {src_idx}.")
                prev_idx = src_idx
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if show_obbs:
                    obb_idx = obb_row_for_frame.get(src_idx)
                    if obb_idx is not None:
                        draw_refined_obbs(rgb, obb_values[obb_idx], id_palette)

                pose_idx = pose_row_for_frame.get(src_idx)
                if pose_idx is not None and num_ids > 0:
                    row = df_np[pose_idx]
                    fronts = np.column_stack([row[front_x_idx], row[front_y_idx]])
                    rears = np.column_stack([row[rear_x_idx], row[rear_y_idx]])
                    if reverse:
                        fronts, rears = rears, fronts
                    valid = np.isfinite(fronts).all(axis=1) & np.isfinite(rears).all(axis=1)
                    for tid in np.flatnonzero(valid).tolist():
                        f_pt = np.round(fronts[tid]).astype(int)
                        r_pt = np.round(rears[tid]).astype(int)
                        color = id_palette[tid] if tid < len(id_palette) else WHITE_RGB
                        alen = max(1.0, float(np.linalg.norm(f_pt - r_pt)))
                        tip = min(0.8, tip_length / alen)
                        if show_arrows:
                            cv2.arrowedLine(
                                rgb, tuple(r_pt), tuple(f_pt), color,
                                arrow_width, cv2.LINE_AA, tipLength=tip,
                            )
                        if show_labels:
                            anchor = np.round((fronts[tid] + rears[tid]) / 2.0).astype(int)
                            lpos = (int(anchor[0]) + 6, int(anchor[1]) - 6)
                            cv2.putText(
                                rgb, f"ID {tid}", lpos, cv2.FONT_HERSHEY_SIMPLEX,
                                label_size / 28.0, color, max(1, label_size // 10), cv2.LINE_AA,
                            )

                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if progress_cb and (done % 10 == 0 or done == total - 1):
                    progress_cb(done + 1, total, f"Writing frame {src_idx} ({done + 1}/{total})")
        finally:
            writer.release()

        if progress_cb:
            progress_cb(total, total, f"Done: {total} frames -> {os.path.basename(out_path)}")
    finally:
        cap.release()
