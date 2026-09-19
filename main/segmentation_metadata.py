# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import os
import pickle
from typing import Any


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def segmentation_pickle_path_for_video(video_path: str) -> str:
    if not video_path:
        return ""
    video_dir = os.path.dirname(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(
        video_dir,
        f"amadeus_{stem}",
        "segmentation",
        f"{stem}_list_of_blobs_gui.pickle",
    )


def segmentation_paths_for_session(session_path: str, video_path: str) -> tuple[str, str]:
    """Return (pickle_path, background_path) for a session's segmentation output.

    Always <session_path>/segmentation/ -- no fallback. SESSION_PATH is the
    directory config.yaml itself lives in, and segmentation always writes its
    output right next to its own config (see gui_segmentation.py's
    _default_output_dir(), which is anchored the same way), so this is the
    one and only place a session's pickle/background can be.
    """
    session_path = str(session_path or "").strip()
    if not session_path:
        return "", ""
    seg_dir = os.path.join(session_path, "segmentation")
    video_path = str(video_path or "").strip()
    stem = os.path.splitext(os.path.basename(video_path))[0] if video_path else "segmentation"
    return (
        os.path.join(seg_dir, f"{stem}_list_of_blobs_gui.pickle"),
        os.path.join(seg_dir, "background.png"),
    )


def read_segmentation_metadata(pickle_path: str) -> dict:
    if not pickle_path or not os.path.exists(pickle_path):
        return {}
    obj = load_pickle(pickle_path)
    return segmentation_metadata_from_object(obj)


def segmentation_metadata_from_object(obj: Any) -> dict:
    try:
        total = int(obj.source_frame_count)
        bg_start = int(obj.background_frame_start)
        bg_end = int(obj.background_frame_end)
        tr_start = int(obj.training_frame_start)
        tr_end = int(obj.training_frame_end)
        interval = int(obj.training_frame_interval)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Segmentation pickle is missing required 0.3.0 frame-range metadata."
        ) from exc
    if total <= 0:
        raise ValueError("source_frame_count must be positive in segmentation metadata.")
    if interval < 1:
        raise ValueError("training_frame_interval must be at least 1.")

    last = total - 1
    def _validate_range(label: str, start: int, end: int) -> tuple[int, int]:
        end = last if end == -1 else end
        if start < 0 or end < 0 or start > end or end > last:
            raise ValueError(
                f"Invalid {label} frame range {start}..{end}; expected 0..{last}."
            )
        return start, end

    bg_start, bg_end = _validate_range("background", bg_start, bg_end)
    tr_start, tr_end = _validate_range("training", tr_start, tr_end)

    return {
        "source_frame_count": total,
        "background_frame_start": bg_start,
        "background_frame_end": bg_end,
        "training_frame_start": tr_start,
        # Keep the public/config convention that -1 means "through the last frame".
        # Explicit earlier end frames remain unchanged.
        "training_frame_end": -1 if tr_end == last else tr_end,
        "training_frame_interval": interval,
        "has_frame_range_metadata": True,
    }
