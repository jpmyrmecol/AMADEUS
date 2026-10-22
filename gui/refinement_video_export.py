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

# create_video.py defaults. Triangle/ellipse direction markers are drawn with
# the same opacity and size as the normal tracking result video.
RENDER_MODES = ("center", "arrow", "obb", "ellipse")
DEFAULT_CENTER_RADIUS = 3
DEFAULT_OBB_WIDTH = 1
DEFAULT_ELLIPSE_WIDTH = 1
DIRECTION_MARKER_ALPHA = 0.3
OBB_TRIANGLE_SCALE = 1.2


def obb_corners(record: np.ndarray) -> np.ndarray | None:
    """Return unrounded OBB corners (0 degrees points upward)."""
    cx, cy, width, height, heading = np.asarray(record, dtype=float)
    if not np.isfinite([cx, cy, width, height, heading]).all() or width <= 0 or height <= 0:
        return None
    theta = np.deg2rad(heading)
    major = np.array([np.sin(theta), -np.cos(theta)]) * (width * 0.5)
    minor = np.array([-np.cos(theta), -np.sin(theta)]) * (height * 0.5)
    center = np.array([cx, cy], dtype=float)
    return np.stack((
        center - major - minor,
        center - major + minor,
        center + major + minor,
        center + major - minor,
    )).astype(np.float32)


def _draw_obb_triangle(
    image: np.ndarray, pts: np.ndarray, heading: float, color: tuple, alpha: float,
) -> None:
    """Draw the OBB head triangle using the selected opacity."""
    direction = np.array([
        np.sin(np.deg2rad(heading)), -np.cos(np.deg2rad(heading))
    ], dtype=float)
    center = np.mean(pts, axis=0)
    lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    shortest = float(np.min(lengths))
    if shortest <= 0:
        return
    candidates = []
    for i, edge_len in enumerate(lengths):
        if abs(float(edge_len) - shortest) > max(1e-6, shortest * 0.05):
            continue
        p0, p1 = pts[i].astype(float), pts[(i + 1) % 4].astype(float)
        edge = p1 - p0
        normal = np.array([edge[1], -edge[0]], dtype=float) / max(float(edge_len), 1e-6)
        midpoint = (p0 + p1) / 2
        if float(np.dot(midpoint - center, normal)) < 0:
            normal = -normal
        candidates.append((float(np.dot(normal, direction)), p0, p1, normal, float(edge_len)))
    if not candidates:
        return
    _, p0, p1, normal, base_len = max(candidates, key=lambda item: item[0])
    apex = (p0 + p1) * 0.5 + normal * (OBB_TRIANGLE_SCALE * np.sqrt(3.0) * 0.5 * base_len)
    triangle = np.rint(np.stack((p0, p1, apex))).astype(np.int32)
    overlay = image.copy()
    cv2.fillConvexPoly(overlay, triangle, color, lineType=cv2.LINE_AA)
    cv2.addWeighted(
        overlay, alpha, image, 1.0 - alpha,
        0.0, dst=image,
    )


def _draw_ellipse_with_arrowhead(
    image: np.ndarray, record: np.ndarray, color: tuple, line_width: int, alpha: float,
) -> None:
    """Match create_video's fitted ellipse and tangent-arc direction marker."""
    cx, cy, width, height, heading = map(float, record)
    theta = np.deg2rad(heading)
    major = np.array([np.sin(theta), -np.cos(theta)], dtype=float)
    minor = np.array([-major[1], major[0]], dtype=float)
    center = np.array([cx, cy], dtype=float)
    axis_angle = float(np.degrees(np.arctan2(major[1], major[0])))
    cv2.ellipse(
        image, tuple(np.rint(center).astype(np.int32)),
        (max(1, int(round(width / 2))), max(1, int(round(height / 2)))),
        axis_angle, 0.0, 360.0, color, line_width, cv2.LINE_AA,
    )

    semi_major, semi_minor = width / 2.0, height / 2.0
    tip_distance = semi_major + width / 3.0
    if semi_major <= 0 or semi_minor <= 0:
        return
    tangent_x = semi_major * semi_major / tip_distance
    tangent_angle = np.arccos(np.clip(tangent_x / semi_major, -1.0, 1.0))
    arc = np.linspace(tangent_angle, -tangent_angle, 25)
    arc_points = np.array([
        center + major * (semi_major * np.cos(a)) + minor * (semi_minor * np.sin(a))
        for a in arc
    ])
    tip = center + major * tip_distance
    polygon = np.rint(np.vstack((tip, arc_points))).astype(np.int32).reshape(-1, 1, 2)
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(
        overlay, alpha, image, 1.0 - alpha,
        0.0, dst=image,
    )


def draw_refined_shapes(
    image: np.ndarray,
    frame_obbs: np.ndarray,
    id_palette: list[tuple[int, int, int]],
    mode: str,
    line_width: int,
    *,
    reverse: bool = False,
    triangle_alpha: float = DIRECTION_MARKER_ALPHA,
) -> None:
    """Draw the mutually exclusive OBB or ellipse style with head direction."""
    if mode not in ("obb", "ellipse"):
        raise ValueError(f"Unsupported box drawing style: {mode}")
    if not np.isfinite(triangle_alpha) or not 0.0 <= triangle_alpha <= 1.0:
        raise ValueError("Triangle alpha must be between 0 and 1.")
    for tid, raw in enumerate(frame_obbs):
        record = np.array(raw, dtype=float, copy=True)
        if reverse:
            record[4] = (record[4] + 180.0) % 360.0
        pts = obb_corners(record)
        if pts is None:
            continue
        color = id_palette[tid] if tid < len(id_palette) else WHITE_RGB
        if mode == "obb":
            polygon = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [polygon], True, color, line_width, cv2.LINE_AA)
            _draw_obb_triangle(image, pts, float(record[4]), color, triangle_alpha)
        else:
            _draw_ellipse_with_arrowhead(image, record, color, line_width, triangle_alpha)



def draw_refinement_annotations(
    image: np.ndarray,
    pose_row: np.ndarray | None,
    frame_obbs: np.ndarray | None,
    front_x_idx: np.ndarray,
    front_y_idx: np.ndarray,
    rear_x_idx: np.ndarray,
    rear_y_idx: np.ndarray,
    id_palette: list,
    num_ids: int,
    line_width: int,
    label_size: int,
    render_mode: str,
    show_labels: bool,
    reverse: bool,
    triangle_alpha: float = DIRECTION_MARKER_ALPHA,
) -> None:
    """Draw one source frame identically for video export and preview."""
    if render_mode in ("obb", "ellipse"):
        if frame_obbs is None:
            return
        draw_refined_shapes(
            image, frame_obbs, id_palette, render_mode, line_width,
            reverse=reverse, triangle_alpha=triangle_alpha,
        )
        if show_labels:
            for tid, record in enumerate(frame_obbs):
                if obb_corners(record) is None:
                    continue
                color = id_palette[tid] if tid < len(id_palette) else WHITE_RGB
                anchor = np.rint(record[:2]).astype(int)
                cv2.putText(
                    image, f"ID {tid}",
                    (int(anchor[0]) + 6, int(anchor[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, label_size / 28.0,
                    color, max(1, label_size // 10), cv2.LINE_AA,
                )
        return

    if pose_row is None or num_ids == 0:
        return
    fronts = np.column_stack([pose_row[front_x_idx], pose_row[front_y_idx]])
    rears = np.column_stack([pose_row[rear_x_idx], pose_row[rear_y_idx]])
    if reverse:
        fronts, rears = rears, fronts
    valid = np.isfinite(fronts).all(axis=1) & np.isfinite(rears).all(axis=1)
    tip_length = max(8, int(round(line_width * 5.0)))
    for tid in np.flatnonzero(valid).tolist():
        f_pt = np.rint(fronts[tid]).astype(int)
        r_pt = np.rint(rears[tid]).astype(int)
        color = id_palette[tid] if tid < len(id_palette) else WHITE_RGB
        if render_mode == "center":
            center = np.rint((fronts[tid] + rears[tid]) * 0.5).astype(int)
            cv2.circle(image, tuple(center), line_width, color, -1, cv2.LINE_AA)
        else:
            alen = max(1.0, float(np.linalg.norm(f_pt - r_pt)))
            cv2.arrowedLine(
                image, tuple(r_pt), tuple(f_pt), color, line_width,
                cv2.LINE_AA, tipLength=min(0.8, tip_length / alen),
            )
            if show_labels:
                center = np.rint((fronts[tid] + rears[tid]) * 0.5).astype(int)
                cv2.putText(
                    image, f"ID {tid}",
                    (int(center[0]) + 6, int(center[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, label_size / 28.0,
                    color, max(1, label_size // 10), cv2.LINE_AA,
                )


def crop_refinement_preview(
    image: np.ndarray,
    pose_row: np.ndarray | None,
    frame_obbs: np.ndarray | None,
    front_x_idx: np.ndarray,
    front_y_idx: np.ndarray,
    rear_x_idx: np.ndarray,
    rear_y_idx: np.ndarray,
    render_mode: str,
) -> tuple[np.ndarray, int | None]:
    """Focus on the visible animal nearest the image center, with surrounding context."""
    height, width = image.shape[:2]
    if render_mode in ("obb", "ellipse"):
        if frame_obbs is None:
            return image.copy(), None
        obbs = np.asarray(frame_obbs, dtype=float)
        centers = obbs[:, :2]
        spans = np.maximum(obbs[:, 2], obbs[:, 3])
        valid = np.isfinite(obbs).all(axis=1) & (obbs[:, 2] > 0) & (obbs[:, 3] > 0)
    else:
        if pose_row is None:
            return image.copy(), None
        fronts = np.column_stack([pose_row[front_x_idx], pose_row[front_y_idx]])
        rears = np.column_stack([pose_row[rear_x_idx], pose_row[rear_y_idx]])
        centers = (fronts + rears) * 0.5
        spans = np.linalg.norm(fronts - rears, axis=1)
        valid = np.isfinite(fronts).all(axis=1) & np.isfinite(rears).all(axis=1)

    valid &= (
        (centers[:, 0] >= 0) & (centers[:, 0] < width)
        & (centers[:, 1] >= 0) & (centers[:, 1] < height)
    )
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        return image.copy(), None

    relative = (centers[ids] - np.array([width / 2, height / 2])) / [width, height]
    tid = int(ids[np.argmin(np.sum(relative * relative, axis=1))])
    center_x, center_y = centers[tid]
    animal_size = max(1.0, float(spans[tid]))
    crop_w = min(width, max(160, int(round(animal_size * 6))))
    crop_h = min(height, max(120, int(round(animal_size * 4.5))))
    x0 = int(np.clip(round(center_x - crop_w / 2), 0, width - crop_w))
    y0 = int(np.clip(round(center_y - crop_h / 2), 0, height - crop_h))
    return image[y0:y0 + crop_h, x0:x0 + crop_w].copy(), tid


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
    line_width: int,
    label_size: int,
    render_mode: str,
    show_labels: bool,
    reverse: bool,
    progress_cb: Callable[[int, int, str], None] | None,
    *,
    obb_frames: np.ndarray | None = None,
    obb_values: np.ndarray | None = None,
    triangle_alpha: float = DIRECTION_MARKER_ALPHA,
) -> None:
    """Export one of center, arrow, OBB, or ellipse for each animal."""
    if render_mode not in RENDER_MODES:
        raise ValueError(f"Unknown Refinement drawing mode: {render_mode}")
    if line_width < 1:
        raise ValueError("Drawing width or point radius must be positive.")
    if not np.isfinite(triangle_alpha) or not 0.0 <= triangle_alpha <= 1.0:
        raise ValueError("Triangle alpha must be between 0 and 1.")
    if render_mode in ("obb", "ellipse"):
        if obb_frames is None or obb_values is None:
            raise ValueError("OBB and ellipse export require AMADEUS result geometry.")
        if obb_values.shape != (len(obb_frames), num_ids, 5):
            raise ValueError("Refinement OBB data does not match the selected individuals.")

    pose_row_for_frame = {
        int(frame): i for i, frame in enumerate(df_np[:, 0]) if np.isfinite(frame)
    }
    obb_row_for_frame = (
        {int(frame): i for i, frame in enumerate(obb_frames)}
        if render_mode in ("obb", "ellipse") else {}
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
        prev_idx = None
        try:
            for done, src_idx in enumerate(source_indices):
                if prev_idx is None or src_idx != prev_idx + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot read frame {src_idx}.")
                prev_idx = src_idx
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                pose_idx = pose_row_for_frame.get(src_idx)
                obb_idx = obb_row_for_frame.get(src_idx)
                draw_refinement_annotations(
                    rgb,
                    df_np[pose_idx] if pose_idx is not None else None,
                    obb_values[obb_idx] if obb_idx is not None else None,
                    front_x_idx, front_y_idx, rear_x_idx, rear_y_idx,
                    id_palette, num_ids, line_width, label_size,
                    render_mode, show_labels, reverse, triangle_alpha,
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
