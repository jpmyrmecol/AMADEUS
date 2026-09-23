# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Render configured tracking results over the input video."""

import sys
from pathlib import Path
_MAIN_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_MAIN_DIR), str(_MAIN_DIR.parent)):
    if _path not in sys.path:
        sys.path.append(_path)


import os
import re
import sys
import subprocess
from dataclasses import dataclass
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, FIRST_COMPLETED, wait
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from without_direction_estimation.obb_detection import (
    build_tracking_out_dir,
    draw_direction_triangle_for_obb,
    load_config,
    obb_to_xyxy,
    parse_weight_spec,
    tqdm_it,
)
from checkpoint_utils import weight_to_checkpoint_number
from without_direction_estimation.multi_staged_association import (
    _artifact_path,
    resolve_weight_from_tracking_dir,
)
from experiment_utils import (
    experiment_dir_name_from_cfg,
    resolve_existing_experiment_dir_name,
)
from weight_utils import deduplicate_best_epoch_weights
from video_frame_count import read_video_frame_info, warn_if_frame_count_adjusted
from gui.color import make_id_palette, resolve_color_seed
from main.video_compat import ffmpeg_executable
from main.video_encoders import detect_hardware_video_encoder, hardware_encoder_args


DIRECTION_CLASS_NAMES = ['animal']


def resolve_num_workers(cfg: dict, section_name: str | None = None, default: int | None = 1, cap: int | None = None) -> int:
    """Resolve worker count from GUI/config settings.

    Priority:
      1. section-specific NUM_WORKERS / WORKERS / N_WORKERS / workers / num_workers
      2. top-level NUM_WORKERS / WORKERS / N_WORKERS / workers / num_workers
      3. default
    """
    keys = ("NUM_WORKERS", "WORKERS", "N_WORKERS", "workers", "num_workers")
    value = None
    if section_name:
        section = cfg.get(section_name, {}) or {}
        if isinstance(section, dict):
            for key in keys:
                if key in section:
                    value = section[key]
                    break
    if value is None:
        for key in keys:
            if key in cfg:
                value = cfg[key]
                break
    if value is None or str(value).strip().lower() in {"", "auto", "none"}:
        workers = int(default if default is not None else 1)
    else:
        workers = int(value)
    workers = max(1, workers)
    if cap is not None:
        workers = min(workers, max(1, int(cap)))
    return workers


def resolve_model_name(name: str, use_obb: bool) -> str:
    base = os.path.splitext(str(name or "").replace("\\", "/").split("/")[-1])[0]
    if use_obb and base and not base.endswith("-obb"):
        base += "-obb"
    return base


def weight_label(weight: str) -> str:
    s = str(weight).strip()
    if s in {"best", "last"}:
        return s
    m = re.fullmatch(r"epoch(\d+)", s)
    if not m:
        raise ValueError(f"Invalid weight name: {weight!r}")
    return str(weight_to_checkpoint_number(s))


def resolve_create_video_run_names(session_path: str, model_name: str, dataset_name: str, create_video_cfg: dict) -> tuple[list[str], list[str]]:
    weights = list(
        dict.fromkeys(
            parse_weight_spec(create_video_cfg.get("WEIGHT"))
        )
    )
    weights = list(dict.fromkeys(resolve_weight_from_tracking_dir(session_path, model_name, dataset_name, w) for w in weights))
    weights = deduplicate_best_epoch_weights(
        session_path, model_name, dataset_name, weights,
    )
    return [weight for weight in weights], weights


def draw_text_plain(img, text, x, y, color, scale=0.6, thickness=1):
    cv2.putText(
        img,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def ensure_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    pts = pts[order]
    area2 = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        area2 += x1 * y2 - x2 * y1
    if area2 > 0:
        pts = pts[::-1]
    return pts.astype(np.float32)


def draw_obb(img, pts: np.ndarray, color, thickness=2):
    poly = np.round(ensure_clockwise(pts)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, color, int(thickness), cv2.LINE_AA)


def _tracking_csv_pair(out_dir: str, suffix: str) -> tuple[str, str]:
    tag = f'_{suffix}' if suffix else ''
    return (
        os.path.join(out_dir, f'obbs{tag}.csv'),
        os.path.join(out_dir, f'directions{tag}.csv'),
    )



def _csv_pair_exists(paths: tuple[str, str]) -> bool:
    return all(os.path.exists(path) for path in paths)


def find_tracking_csvs(out_dir: str) -> tuple[str, str]:
    """Prefer ID-resolved OBB/direction outputs, then standard outputs."""
    for suffix in ('id_resolved', ''):
        paths = _tracking_csv_pair(out_dir, suffix)
        if _csv_pair_exists(paths):
            return paths
    raise FileNotFoundError(f'Final tracking OBB/direction CSVs not found in {out_dir}')



def resolve_assign_type_pickle(out_dir: str, obb_csv: str) -> str:
    name = os.path.basename(obb_csv)
    if name == 'obbs_id_resolved.csv':
        artifact_suffix = 'id_resolved'
    elif name == 'obbs.csv':
        artifact_suffix = 'filled'
    else:
        raise ValueError(f'Unsupported final OBB CSV name: {name}')
    return _artifact_path(out_dir, 'assign_type.pkl', artifact_suffix)


def infer_ids_from_obb_df(df: pd.DataFrame) -> list[int]:
    ids = set()
    for col in df.columns:
        m = re.fullmatch(r"[xy][0-3](\d+)", str(col))
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def build_row_lookup_dict(df: pd.DataFrame | None) -> dict[int, dict]:
    if df is None or "position" not in df.columns:
        return {}
    work = df.copy()
    work["position"] = pd.to_numeric(work["position"], errors="coerce")
    work = work.dropna(subset=["position"])
    if work.empty:
        return {}
    work["position"] = work["position"].astype(int)
    return work.set_index("position").to_dict(orient="index")


@dataclass(frozen=True)
class TrackColumns:
    x_cols: list[str]
    y_cols: list[str]
    class_col: str



def build_track_columns(ids: list[int]) -> dict[int, TrackColumns]:
    out = {}
    for tid in ids:
        out[tid] = TrackColumns(
            x_cols=[f"x{i}{tid}" for i in range(4)],
            y_cols=[f"y{i}{tid}" for i in range(4)],
            class_col=f"c{tid}",
        )
    return out


def get_obb_from_dict(row: dict | None, cols: TrackColumns) -> np.ndarray | None:
    if row is None:
        return None
    vals = [row.get(name) for name in cols.x_cols + cols.y_cols]
    arr = np.asarray(vals, dtype=float)
    if arr.shape != (8,) or not np.isfinite(arr).all():
        return None
    return np.column_stack([arr[:4].astype(np.float32, copy=False), arr[4:].astype(np.float32, copy=False)])


def build_track_label(assign_type: float | None) -> str:
    base_map = {
        0.0: "ini",
        1.0: "s1",
        2.0: "s2",
        3.0: "s3",
        4.0: "s4",
        5.0: "old5",
        6.0: "s5",
        7.0: "bf",
        11.0: "s6",
        13.0: "dn",
        14.0: "df",
        15.0: "dp",
        16.0: "ta",
        17.0: "flp",
        20.0: "c1",
        21.0: "c2",
        22.0: "c3s",
        23.0: "c3e",
        24.0: "c5",
        25.0: "c4p",
        30.0: "pre",
        31.0: "post",
        32.0: "anc",
        33.0: "spk",
        34.0: "int",
        35.0: "swap",
    }
    if assign_type is None or not np.isfinite(assign_type):
        return "na"
    code = float(assign_type)
    return base_map.get(code, f"t{int(round(code))}")

def list_video_files(video_path_in: str) -> list[str]:
    if os.path.isdir(video_path_in):
        video_files = [
            os.path.join(video_path_in, f)
            for f in sorted(os.listdir(video_path_in))
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".m4v"))
        ]
    else:
        video_files = [video_path_in]
    if not video_files:
        raise RuntimeError(f"Video not found: {video_path_in}")
    return video_files


def purge_images(image_dir: str, extensions: Iterable[str]) -> int:
    removed = 0
    exts = {str(ext).lower() for ext in extensions}
    for name in os.listdir(image_dir):
        if os.path.splitext(name)[1].lower() in exts:
            os.remove(os.path.join(image_dir, name))
            removed += 1
    return removed


def summarize_frame_coverage(selected_frames: list[int], obb_rows: dict[int, dict]) -> tuple[int, int, int]:
    selected_set = set(selected_frames)
    csv_frame_set = set(obb_rows.keys())
    matched = len(selected_set & csv_frame_set)
    missing = len(selected_set - csv_frame_set)
    extra = len(csv_frame_set - selected_set)
    return matched, missing, extra


def choose_frame_range(analysis: dict, total_frames: int) -> tuple[int, int, str, str]:
    first_frame = int(analysis.get("FIRST_FRAME", 0))
    last_frame = int(analysis.get("LAST_FRAME", -1))
    start_src = "analysis.FIRST_FRAME"
    end_src = "analysis.LAST_FRAME"

    if first_frame < 0:
        first_frame = 0
    if last_frame < 0 or last_frame >= total_frames:
        last_frame = total_frames - 1
    if last_frame < first_frame:
        raise RuntimeError(f"Invalid frame range: first_frame={first_frame}, last_frame={last_frame}")

    return first_frame, last_frame, start_src, end_src


_GPU_VIDEO_ENCODER: tuple[str, str | None] | None | bool = False


def resolve_video_acceleration(value) -> str:
    mode = str(value if value is not None else "cpu").strip().lower()
    if mode not in {"cpu", "gpu", "auto"}:
        raise RuntimeError(f"Unsupported ACCELERATION: {value}. Use 'cpu', 'gpu', or 'auto'.")
    return mode


def _ffmpeg_path() -> str | None:
    """Return the one FFmpeg executable selected for this AMADEUS install."""
    try:
        return ffmpeg_executable()
    except Exception:
        return None


def _gpu_video_encoder() -> tuple[str, str | None] | None:
    global _GPU_VIDEO_ENCODER
    if _GPU_VIDEO_ENCODER is not False:
        return _GPU_VIDEO_ENCODER
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        _GPU_VIDEO_ENCODER = None
    else:
        _GPU_VIDEO_ENCODER = detect_hardware_video_encoder(ffmpeg)
    return _GPU_VIDEO_ENCODER


def _should_use_gpu_video_encoder(video_acceleration: str, video_codec: str = "auto") -> bool:
    mode = resolve_video_acceleration(video_acceleration)
    codec = str(video_codec or "auto").strip().lower()
    if codec not in {"auto", "h264", "mp4v"}:
        raise RuntimeError(f"Unsupported VIDEO_CODEC: {video_codec}. Use 'auto', 'h264', or 'mp4v'.")
    if mode == "cpu":
        return False
    if codec == "mp4v":
        if mode == "gpu":
            print("[WARN] ACCELERATION=gpu requires H.264; VIDEO_CODEC=mp4v requested, using CPU MPEG-4 encoding.")
        return False
    available = _gpu_video_encoder()
    if available is None:
        if mode == "gpu":
            print("[WARN] ACCELERATION=gpu requested, but no supported hardware H.264 encoder is available; using CPU video encoding.")
        return False
    return True


def _selected_video_encoder(video_acceleration: str, video_codec: str) -> tuple[str, str | None]:
    codec = str(video_codec or "auto").strip().lower()
    if codec not in {"auto", "h264", "mp4v"}:
        raise RuntimeError(f"Unsupported VIDEO_CODEC: {video_codec}. Use 'auto', 'h264', or 'mp4v'.")
    if _should_use_gpu_video_encoder(video_acceleration, video_codec):
        selected = _gpu_video_encoder()
        assert selected is not None
        return selected
    return ("mpeg4" if codec == "mp4v" else "libx264", None)


def make_video_from_image_sequence(
    image_dir: str,
    first_frame: int,
    fps: float,
    out_path: str,
    extension: str,
    video_acceleration: str = "cpu",
):
    ext = str(extension).strip().lower().lstrip(".")
    if ext not in {"png", "jpg", "jpeg"}:
        raise RuntimeError(f"Unsupported image sequence extension: {extension}")

    encoder, device = _selected_video_encoder(video_acceleration, "h264")
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("The pinned AMADEUS FFmpeg is unavailable.")
    filters = ["pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    if encoder == "h264_vaapi":
        filters.append("format=nv12,hwupload")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if device:
        cmd.extend(["-vaapi_device", device])
    cmd.extend([
        "-start_number", str(int(first_frame)),
        "-framerate", str(float(fps)),
        "-i", os.path.join(image_dir, f"%06d.{ext}"),
        "-vf", ",".join(filters),
        *(hardware_encoder_args(encoder, profile="create_video") if encoder.startswith("h264_") else _cpu_encoder_args(encoder)),
        out_path,
    ])
    print(f"[INFO] AMADEUS FFmpeg image-sequence encoder for {os.path.basename(out_path)}: {encoder}")
    subprocess.run(cmd, check=True)


def _cpu_encoder_args(encoder: str) -> list[str]:
    if encoder == "mpeg4":
        return ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]


class FfmpegRawVideoWriter:
    """Write rendered BGR frames through AMADEUS's fixed FFmpeg build."""

    def __init__(
        self,
        path: str,
        fps: float,
        frame_shape: tuple[int, int, int],
        encoder: str,
        device: str | None = None,
    ) -> None:
        h, w = frame_shape[:2]
        ffmpeg = _ffmpeg_path()
        if not ffmpeg:
            raise RuntimeError("The pinned AMADEUS FFmpeg is unavailable.")
        self.path = path
        self.width = int(w)
        self.height = int(h)
        filters = ["pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        if encoder == "h264_vaapi":
            if not device:
                raise RuntimeError("VAAPI requires a detected render device.")
            filters.append("format=nv12,hwupload")
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if device:
            command.extend(["-vaapi_device", device])
        command.extend([
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(float(fps)),
            "-i", "pipe:0",
            "-vf", ",".join(filters),
            *(hardware_encoder_args(encoder, profile="create_video") if encoder.startswith("h264_") else _cpu_encoder_args(encoder)),
            path,
        ])
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("FFmpeg writer stdin is closed.")
        arr = np.ascontiguousarray(frame)
        if arr.shape[0] != self.height or arr.shape[1] != self.width or arr.shape[2] != 3:
            raise RuntimeError(f"Unexpected frame shape for FFmpeg writer: {arr.shape}")
        try:
            self.proc.stdin.write(arr.tobytes())
        except BrokenPipeError as exc:
            err = b""
            if self.proc.stderr is not None:
                err = self.proc.stderr.read()
            self.proc.wait()
            msg = err.decode("utf-8", errors="replace").strip()
            detail = f": {msg}" if msg else ""
            raise RuntimeError(f"FFmpeg GPU video writer stopped unexpectedly{detail}") from exc

    def release(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        err = b""
        if self.proc.stderr is not None:
            err = self.proc.stderr.read()
        rc = self.proc.wait()
        if rc != 0:
            msg = err.decode("utf-8", errors="replace").strip()
            detail = f": {msg}" if msg else ""
            raise RuntimeError(f"FFmpeg GPU video writer failed with exit code {rc}{detail}")


@dataclass
class _FrameRenderCtx:
    ids: list
    track_cols: dict
    obb_rows: dict
    class_rows: dict
    assign_type_rows: dict
    colors: list
    color_slots: dict | None
    draw_obb_flag: bool
    obb_thickness: int
    draw_arrow: bool
    triangle_alpha: float
    triangle_outline_thickness: int
    triangle_scale: float
    draw_labels: bool
    label_font_scale: float
    label_thickness: int


def _render_video_frame(
    frame_idx: int, raw_frame: np.ndarray, ctx: _FrameRenderCtx
) -> tuple:
    """Render OBBs/arrows/labels onto a copy of raw_frame. Thread-safe (no shared state).
    Returns (rendered_frame, frame_box_count, boxes_drawn).
    """
    frame = raw_frame.copy()
    row = ctx.obb_rows.get(frame_idx)
    cls_row = ctx.class_rows.get(frame_idx)
    assign_type_row = ctx.assign_type_rows.get(frame_idx)

    frame_box_count = 0
    boxes_drawn = 0
    for tid in ctx.ids:
        cols = ctx.track_cols[tid]
        pts = get_obb_from_dict(row, cols)
        if pts is None:
            continue
        slot = ctx.color_slots[frame_idx][tid] if ctx.color_slots is not None else tid % len(ctx.colors)
        color = ctx.colors[slot]
        if ctx.draw_obb_flag:
            draw_obb(frame, pts, color, ctx.obb_thickness)
        frame_box_count += 1
        boxes_drawn += 1

        cls_idx = None
        if cls_row is not None:
            cval = cls_row.get(cols.class_col)
            if cval is not None and not pd.isna(cval):
                cls_idx = float(cval)

        if ctx.draw_arrow and cls_idx is not None:
            draw_direction_triangle_for_obb(
                frame, pts, cls_idx, color,
                ctx.triangle_alpha, ctx.triangle_outline_thickness, ctx.triangle_scale,
            )

        if ctx.draw_labels:
            assign_type = None if assign_type_row is None else assign_type_row.get(f"t{tid}")
            label = build_track_label(assign_type)
            x0, y0, _, _ = obb_to_xyxy(pts)
            draw_text_plain(
                frame, label,
                int(round(x0)) + 2, max(16, int(round(y0)) - 6),
                color, ctx.label_font_scale, ctx.label_thickness,
            )

    return frame, frame_box_count, boxes_drawn


def render_video_for_weight(
    src_video: str,
    out_dir: str,
    png_image_dir: str,
    jpg_image_dir: str,
    num_objects: int,
    draw_obb_flag: bool,
    draw_arrow: bool,
    draw_labels: bool,
    output_fps: float,
    first_frame: int,
    last_frame: int,
    frame_step: int,
    colors: list[tuple[int, int, int]],
    obb_thickness: int,
    triangle_outline_thickness: int,
    label_font_scale: float,
    label_thickness: int,
    triangle_alpha: float,
    triangle_scale: float,
    export_raw_yolo_video: bool,
    save_png_frames: bool,
    save_jpg_frames: bool,
    jpg_quality: int,
    video_codec: str,
    video_acceleration: str,
    video_filename: str = "tracking.mp4",
    num_io_workers: int = 4,
    num_render_workers: int = 1,
    variable_population: bool = False,
    color_seed: int | None = None,
) -> None:
    obb_csv, class_csv = find_tracking_csvs(out_dir)
    obb_df = pd.read_csv(obb_csv)
    class_df = pd.read_csv(class_csv)
    assign_pkl = resolve_assign_type_pickle(out_dir, obb_csv)
    assign_type_df = pd.read_pickle(assign_pkl) if os.path.exists(assign_pkl) else None

    obb_rows = build_row_lookup_dict(obb_df)
    class_rows = build_row_lookup_dict(class_df)
    assign_type_rows = build_row_lookup_dict(assign_type_df)

    ids = infer_ids_from_obb_df(obb_df)
    if not ids and not variable_population:
        ids = list(range(num_objects))
    track_cols = build_track_columns(ids)
    color_slots = None
    if variable_population:
        from variable_population import allocate_color_slots
        present = {frame: [tid for tid in ids if get_obb_from_dict(row, track_cols[tid]) is not None]
                   for frame, row in obb_rows.items()}
        peak, color_slots = allocate_color_slots(present)
        colors = make_id_palette(peak, color_space="bgr", seed=color_seed) if peak else []
        print(f"[INFO] Variable population: {len(ids)} lifetime IDs; peak visible={peak}; colors={len(colors)}")

    selected_frames = list(range(first_frame, last_frame + 1, frame_step))
    matched_frames, missing_frames, extra_csv_frames = summarize_frame_coverage(selected_frames, obb_rows)
    print(
        f"[INFO] {os.path.basename(src_video)}: selected_frames={len(selected_frames)}, "
        f"csv_frames_matched={matched_frames}, csv_frames_missing={missing_frames}, csv_frames_extra={extra_csv_frames}"
    )

    if save_png_frames:
        removed_pngs = purge_images(png_image_dir, [".png"])
        if removed_pngs > 0:
            print(f"[INFO] Removed {removed_pngs} existing PNGs from {png_image_dir}")

    if save_jpg_frames:
        removed_jpgs = purge_images(jpg_image_dir, [".jpg", ".jpeg"])
        if removed_jpgs > 0:
            print(f"[INFO] Removed {removed_jpgs} existing JPEGs from {jpg_image_dir}")

    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError(src_video)

    video_path_out = os.path.join(out_dir, video_filename)
    export_images_enabled = bool(save_png_frames or save_jpg_frames)
    writer = None
    n_written = 0
    n_frames_with_boxes = 0
    n_boxes_drawn = 0
    current_frame_idx = first_frame
    jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    video_acceleration = resolve_video_acceleration(video_acceleration)

    def _write_png(path: str, img: np.ndarray) -> None:
        if not cv2.imwrite(path, img):
            raise RuntimeError(f"Failed to write image: {path}")

    def _write_jpg(path: str, img: np.ndarray) -> None:
        if not cv2.imwrite(path, img, jpg_params):
            raise RuntimeError(f"Failed to write image: {path}")

    render_workers = max(1, int(num_render_workers))
    io_workers = max(1, int(num_io_workers)) if export_images_enabled else 1
    RENDER_BATCH = max(16, render_workers * 4)

    ctx = _FrameRenderCtx(
        ids=ids,
        track_cols=track_cols,
        obb_rows=obb_rows,
        class_rows=class_rows,
        assign_type_rows=assign_type_rows,
        colors=colors,
        color_slots=color_slots,
        draw_obb_flag=draw_obb_flag,
        obb_thickness=obb_thickness,
        draw_arrow=draw_arrow,
        triangle_alpha=triangle_alpha,
        triangle_outline_thickness=triangle_outline_thickness,
        triangle_scale=triangle_scale,
        draw_labels=draw_labels,
        label_font_scale=label_font_scale,
        label_thickness=label_thickness,
    )

    batch: list = []
    pending_futures: list = []

    frame_iter = tqdm_it(
        total=len(selected_frames),
        desc=f"Writing {os.path.basename(src_video)} frames {first_frame}-{last_frame}",
        unit="frame",
    )
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        with ThreadPoolExecutor(max_workers=render_workers) as render_pool, \
             ThreadPoolExecutor(max_workers=io_workers) as io_pool:

            def _flush():
                nonlocal writer, n_written, n_frames_with_boxes, n_boxes_drawn, pending_futures
                if not batch:
                    return
                futures = [render_pool.submit(_render_video_frame, fid, frm, ctx) for fid, frm in batch]
                for (fid, _), fut in zip(batch, futures):
                    rendered, box_count, boxes = fut.result()
                    if writer is None and not export_images_enabled:
                        encoder, device = _selected_video_encoder(video_acceleration, video_codec)
                        writer = FfmpegRawVideoWriter(
                            video_path_out, output_fps, rendered.shape, encoder, device
                        )
                        print(f"[INFO] AMADEUS FFmpeg encoder for {os.path.basename(video_path_out)}: {encoder}")
                    if writer is not None:
                        writer.write(rendered)
                    if save_png_frames:
                        pending_futures.append(io_pool.submit(_write_png, os.path.join(png_image_dir, f"{fid:06d}.png"), rendered))
                    if save_jpg_frames:
                        pending_futures.append(io_pool.submit(_write_jpg, os.path.join(jpg_image_dir, f"{fid:06d}.jpg"), rendered))
                    if len(pending_futures) > io_workers * 8:
                        done, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
                        for f in done:
                            f.result()
                        pending_futures = [f for f in pending_futures if not f.done()]
                    n_written += 1
                    if box_count > 0:
                        n_frames_with_boxes += 1
                        n_boxes_drawn += boxes
                batch.clear()

            for target_frame_idx in selected_frames:
                if current_frame_idx != target_frame_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                    current_frame_idx = target_frame_idx
                ok, frame = cap.read()
                if not ok:
                    print(f"[WARN] Failed to read frame {target_frame_idx} from {src_video}")
                    break

                batch.append((target_frame_idx, np.ascontiguousarray(frame)))
                if len(batch) >= RENDER_BATCH:
                    _flush()

                current_frame_idx = target_frame_idx + 1
                frame_iter.update(1)

            _flush()
            for f in pending_futures:
                f.result()
    finally:
        frame_iter.close()
        cap.release()
        if writer is not None:
            writer.release()

    if export_images_enabled:
        if save_png_frames:
            make_video_from_image_sequence(
                png_image_dir, first_frame, output_fps, video_path_out, "png",
                video_acceleration=video_acceleration,
            )
            print(f"Video written -> {video_path_out}")
        elif save_jpg_frames:
            make_video_from_image_sequence(
                jpg_image_dir, first_frame, output_fps, video_path_out, "jpg",
                video_acceleration=video_acceleration,
            )
            print(f"Video written -> {video_path_out}")
    elif writer is not None:
        print(f"Video written -> {video_path_out}")

    print(
        f"[SUMMARY] {os.path.basename(src_video)}: written_frames={n_written}, "
        f"frames_with_obb={n_frames_with_boxes}, total_obbs_drawn={n_boxes_drawn}"
    )
    expected_variable_boxes = sum(len(color_slots.get(f, {})) for f in selected_frames) if color_slots is not None else 0
    if n_written > 0 and n_boxes_drawn == 0 and (not variable_population or expected_variable_boxes > 0):
        raise RuntimeError(
            "No OBB was drawn in any written image. "
            "Likely frame/CSV mismatch: the renderer uses absolute frame numbers and only draws when "
            "obbs_*.csv has an exact matching frame row with all x/y columns present."
        )

    if export_raw_yolo_video:
        raw_dir = os.path.join(out_dir, "yolo_raw")
        if os.path.isdir(raw_dir):
            raw_video_out = os.path.join(out_dir, "yolo_raw_obb.mp4")
            make_video_from_image_sequence(
                raw_dir, first_frame, output_fps, raw_video_out, "png",
                video_acceleration=video_acceleration,
            )
            print(f"Video written -> {raw_video_out}")


def _render_single_video_job(spec: dict) -> None:
    """Render one (src_video, run_name) combination. Designed to run in a subprocess."""
    src_video     = spec['src_video']
    out_dir       = spec['out_dir']
    png_image_dir = spec['png_image_dir']
    jpg_image_dir = spec['jpg_image_dir']

    if spec['save_png_frames']:
        os.makedirs(png_image_dir, exist_ok=True)
    if spec['save_jpg_frames']:
        os.makedirs(jpg_image_dir, exist_ok=True)

    render_video_for_weight(
        src_video=src_video,
        out_dir=out_dir,
        png_image_dir=png_image_dir,
        jpg_image_dir=jpg_image_dir,
        num_objects=spec['num_objects'],
        variable_population=spec.get('variable_population', False),
        color_seed=spec.get('color_seed'),
        draw_obb_flag=spec['draw_obb_flag'],
        draw_arrow=spec['draw_arrow'],
        draw_labels=spec['draw_labels'],
        output_fps=spec['output_fps'],
        first_frame=spec['first_frame'],
        last_frame=spec['last_frame'],
        frame_step=spec['frame_step'],
        colors=spec['colors'],
        obb_thickness=spec['obb_thickness'],
        triangle_outline_thickness=spec['triangle_outline_thickness'],
        label_font_scale=spec['label_font_scale'],
        label_thickness=spec['label_thickness'],
        triangle_alpha=spec['triangle_alpha'],
        triangle_scale=spec['triangle_scale'],
        export_raw_yolo_video=spec['export_raw_yolo_video'],
        save_png_frames=spec['save_png_frames'],
        save_jpg_frames=spec['save_jpg_frames'],
        jpg_quality=spec['jpg_quality'],
        video_codec=spec['video_codec'],
        video_acceleration=spec['video_acceleration'],
        num_io_workers=spec['num_io_workers'],
        num_render_workers=spec['num_render_workers'],
    )


def resolve_render_out_dir(
    session_path: str,
    model_name: str,
    dataset_name: str,
    run_name: str,
    video_name: str,
) -> str:
    canonical_dataset_name = resolve_existing_experiment_dir_name(
        session_path,
        model_name,
        dataset_name,
        stages=("tracking",),
        warn_fn=print,
    )
    return build_tracking_out_dir(
        session_path,
        model_name,
        canonical_dataset_name,
        run_name,
        video_name,
    )


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python create_video.py config.yaml")

    cfg = load_config(sys.argv[1])
    if cfg.get("skip_creating_video", False):
        print("skip_creating_video is True; exiting.")
        return

    session_path = str(cfg["SESSION_PATH"])
    video_path_in = str(cfg["TRACKING_VIDEO_PATH"])
    num_objects = int(cfg["NUM_OBJECTS"])
    analysis = cfg.get("analysis", {}) or {}
    track_video = cfg.get("create_video", {}) or {}

    use_obb = bool(cfg.get("USE_OBB", True))
    if not use_obb:
        raise RuntimeError("This script is OBB-only. Set USE_OBB: true.")

    model_name = resolve_model_name(cfg.get("training", {}).get("PRETRAINED_MODEL", ""), True)
    dataset_name = resolve_existing_experiment_dir_name(
        session_path,
        model_name,
        experiment_dir_name_from_cfg(cfg),
        stages=("tracking", "training"),
        warn_fn=print,
    )
    run_names, weights = resolve_create_video_run_names(
        session_path, model_name, dataset_name, track_video
    )

    draw_obb_flag = bool(track_video.get("DRAW_OBB", True))
    draw_arrow = bool(track_video.get("DRAW_ARROW", True))
    draw_labels = bool(track_video.get("DRAW_LABELS", False))
    export_raw_yolo_video = bool(track_video.get("EXPORT_RAW", False))

    print("[INFO] Label format: ini/s1/s2/s3/s4/s5/s6/c1/c2/c3/c4/c5")

    export_images = bool(track_video.get("EXPORT_IMAGES", False))
    image_export_format = str(track_video.get("IMAGE_FORMAT", "jpeg") or "jpeg").strip().lower()
    if image_export_format not in {"png", "jpeg", "jpg"}:
        raise RuntimeError(f"Unsupported IMAGE_FORMAT: {image_export_format}. Use 'png' or 'jpeg'.")
    if image_export_format == "jpg":
        image_export_format = "jpeg"

    save_png_frames = export_images and image_export_format == "png"
    save_jpg_frames = export_images and image_export_format == "jpeg"
    jpg_quality = int(track_video.get("JPEG_QUALITY", 95))
    video_codec = str(track_video.get("VIDEO_CODEC", "auto")).strip().lower()
    video_acceleration = resolve_video_acceleration(track_video.get("ACCELERATION", "cpu"))
    obb_thickness = int(track_video.get("OBB_WIDTH", 1))
    triangle_outline_thickness = int(track_video.get("ARROW_WIDTH", 0))
    label_font_scale = float(track_video.get("LABEL_SCALE", 0.5))
    label_thickness = int(track_video.get("LABEL_THICKNESS", 1))
    triangle_alpha = float(track_video.get("ARROW_ALPHA", 0.3))
    triangle_scale = float(track_video.get("ARROW_SCALE", 1.2))

    if video_codec not in {"auto", "h264", "mp4v"}:
        raise RuntimeError(f"Unsupported VIDEO_CODEC: {video_codec}. Use 'auto', 'h264', or 'mp4v'.")
    if not (0 <= jpg_quality <= 100):
        raise RuntimeError(f"JPEG_QUALITY must be between 0 and 100, got {jpg_quality}")
    print(f"[INFO] create_video ACCELERATION={video_acceleration}")

    try:
        color_seed = resolve_color_seed(track_video.get("COLOR_SEED"))
    except ValueError as exc:
        raise RuntimeError(f"Invalid create_video.COLOR_SEED: {exc}") from exc
    if color_seed is not None:
        print(f"[INFO] create_video COLOR_SEED={color_seed}")

    colors = [] if cfg.get("VARIABLE_NUM_OBJECTS", False) else make_id_palette(
        num_objects,
        color_space="bgr",
        seed=color_seed,
    )
    video_files = list_video_files(video_path_in)
    from batch_utils import resolve_num_workers as _bt_resolve_workers
    _workers_raw = (cfg.get("create_video", {}) or {}).get("NUM_WORKERS") or cfg.get("NUM_WORKERS", "auto")
    num_io_workers = _bt_resolve_workers(_workers_raw, task="process")
    num_render_workers = num_io_workers

    specs = []
    for src_video in video_files:
        frame_info = read_video_frame_info(src_video)
        warn_if_frame_count_adjusted(frame_info, label=f'create_video video={os.path.basename(src_video)}')
        total_frames = frame_info.usable_frame_count
        default_fps = frame_info.fps or 30.0

        first_frame, last_frame, start_src, end_src = choose_frame_range(analysis, total_frames)
        print(
            f"[INFO] Frame range for {os.path.basename(src_video)}: "
            f"first_frame={first_frame} from {start_src}, last_frame={last_frame} from {end_src}, "
            f"usable_frames={total_frames}"
        )

        frame_step = max(1, int(track_video.get("FRAME_STEP", 1)))
        output_fps_value = track_video.get("FPS", "")
        if output_fps_value is None or (isinstance(output_fps_value, str) and not output_fps_value.strip()):
            output_fps = default_fps / frame_step
        else:
            output_fps = float(output_fps_value)
        if output_fps <= 0:
            raise RuntimeError(f"FPS must be positive, got {output_fps}")

        video_name = os.path.splitext(os.path.basename(src_video))[0]
        print(
            f"[INFO] create_video.WEIGHT={','.join(weight_label(w) for w in weights)} "
            f"-> using individual checkpoint directories"
        )

        for run_name in run_names:
            out_dir = resolve_render_out_dir(
                session_path,
                model_name,
                dataset_name,
                run_name,
                video_name,
            )
            if cfg.get('VARIABLE_NUM_OBJECTS', False):
                out_dir = os.path.join(out_dir, 'variable')
            specs.append({
                'variable_population': bool(cfg.get('VARIABLE_NUM_OBJECTS', False)),
                'src_video': src_video,
                'out_dir': out_dir,
                'png_image_dir': os.path.join(out_dir, "images"),
                'jpg_image_dir': os.path.join(out_dir, "images_jpg"),
                'num_objects': num_objects,
                'draw_obb_flag': draw_obb_flag,
                'draw_arrow': draw_arrow,
                'draw_labels': draw_labels,
                'output_fps': output_fps,
                'first_frame': first_frame,
                'last_frame': last_frame,
                'frame_step': frame_step,
                'colors': colors,
                'color_seed': color_seed,
                'obb_thickness': obb_thickness,
                'triangle_outline_thickness': triangle_outline_thickness,
                'label_font_scale': label_font_scale,
                'label_thickness': label_thickness,
                'triangle_alpha': triangle_alpha,
                'triangle_scale': triangle_scale,
                'export_raw_yolo_video': export_raw_yolo_video,
                'save_png_frames': save_png_frames,
                'save_jpg_frames': save_jpg_frames,
                'jpg_quality': jpg_quality,
                'video_codec': video_codec,
                'video_acceleration': video_acceleration,
                'num_io_workers': num_io_workers,
                'num_render_workers': num_render_workers,
            })

    n_jobs = len(specs)
    outer = min(num_io_workers, n_jobs)
    workers_per_job = max(1, num_io_workers // outer) if outer > 0 else 1
    for spec in specs:
        spec['num_io_workers'] = workers_per_job
        spec['num_render_workers'] = workers_per_job

    if outer <= 1:
        for spec in specs:
            _render_single_video_job(spec)
    else:
        print(f'create_video: {n_jobs} job(s), outer={outer} parallel, {workers_per_job} worker(s)/job')
        with ProcessPoolExecutor(max_workers=outer) as pool:
            futures = [pool.submit(_render_single_video_job, spec) for spec in specs]
            for f in futures:
                f.result()

if __name__ == "__main__":
    from without_direction_estimation import prepare_config
    if len(sys.argv) > 1:
        sys.argv[1] = prepare_config(sys.argv[1])
    main()
