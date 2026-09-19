# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Canonical paths for internal tracking artifacts."""

import os

BUFFER_DIR_NAME = "buffer"
LOG_DIR_NAME = "log"
PROVENANCE_CSV_NAME = "provenance.csv"


def artifact_filename(base_name: str, suffix: str = "") -> str:
    suffix = str(suffix or "").strip()
    if not suffix:
        return base_name
    stem, ext = os.path.splitext(base_name)
    return f"{stem}_{suffix}{ext}"


def artifact_path(out_dir: str, base_name: str, suffix: str = "") -> str:
    """Return the single canonical path for a tracking-internal artifact.

    Pickle state and provenance live in ``buffer/``. JSON runtime logs/cache
    live in ``log/``. Other files remain at the tracking output root.
    """
    filename = artifact_filename(base_name, suffix)
    ext = os.path.splitext(base_name)[1].lower()
    if ext == ".pkl" or base_name == PROVENANCE_CSV_NAME:
        return os.path.join(out_dir, BUFFER_DIR_NAME, filename)
    if ext == ".json":
        return os.path.join(out_dir, LOG_DIR_NAME, filename)
    return os.path.join(out_dir, filename)
