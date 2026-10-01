# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Resolve the concrete video files used by analysis stages."""

from __future__ import annotations

import os


LEGACY_DIRECTORY_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".m4v")


def configured_tracking_video_files(
    cfg: dict,
    video_path_in: str | None = None,
) -> list[str]:
    """Return the concrete analysis videos configured for this run."""
    configured = cfg.get("TRACKING_VIDEO_FILES")
    if isinstance(configured, (list, tuple)) and configured:
        return [str(path) for path in configured if str(path).strip()]

    path = str(
        video_path_in
        if video_path_in is not None
        else cfg.get("TRACKING_VIDEO_PATH", "")
    ).strip()
    if os.path.isdir(path):
        return [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, name))
            and os.path.splitext(name)[1].lower() in LEGACY_DIRECTORY_VIDEO_SUFFIXES
        ]
    return [path] if path else []
