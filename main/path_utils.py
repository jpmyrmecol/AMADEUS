# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared helpers for storing config.yaml paths relative to SESSION_PATH.

Every path key in config.yaml except SESSION_PATH itself is stored relative to
SESSION_PATH, so a session directory (and its config.yaml) can be moved or
copied without invalidating the paths inside it. SESSION_PATH stays absolute
and is the resolution anchor for everything else. A path with no relative
form against SESSION_PATH (e.g. a different drive on Windows) is kept
absolute instead.
"""

import os

CONFIG_PATH_KEYS = (
    "TRAINING_VIDEO_PATH",
    "TRACKING_VIDEO_PATH",
    "PICKLE_PATH",
    "BACKGROUND_PATH",
    "INIT_CSV_PATH",
    "YOLO_DATASET_DIR",
)


def to_relative_path(path, base_dir: str) -> str:
    """Convert an absolute path to a path relative to base_dir, for storage in config.yaml."""
    path = str(path or "").strip()
    if not path:
        return path
    base_dir = str(base_dir or "").strip()
    if not base_dir:
        return path
    try:
        return os.path.relpath(os.path.abspath(path), os.path.abspath(base_dir))
    except ValueError:
        # e.g. path and base_dir are on different drives on Windows.
        return os.path.abspath(path)


def resolve_path(value, base_dir: str) -> str:
    """Resolve a config.yaml path value (relative or absolute) to an absolute path."""
    value = str(value or "").strip()
    if not value or os.path.isabs(value):
        return value
    base_dir = str(base_dir or "").strip()
    if not base_dir:
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _map_source_dirs(value, fn, base_dir: str):
    if isinstance(value, list):
        return [fn(v, base_dir) for v in value]
    if isinstance(value, str) and value:
        return fn(value, base_dir)
    return value


def resolve_config_paths(cfg: dict) -> dict:
    """Resolve every known path key in cfg (relative or absolute) to absolute, in place.

    SESSION_PATH itself must already be absolute; it is the resolution anchor.
    Call this immediately after loading config.yaml, before reading any path key.
    """
    session_path = str(cfg.get("SESSION_PATH", "") or "")
    for key in CONFIG_PATH_KEYS:
        if key in cfg and cfg[key]:
            cfg[key] = resolve_path(cfg[key], session_path)
    if "CREATE_DATASET_SOURCE_DIRS" in cfg:
        cfg["CREATE_DATASET_SOURCE_DIRS"] = _map_source_dirs(
            cfg["CREATE_DATASET_SOURCE_DIRS"], resolve_path, session_path
        )
    return cfg


def relativize_config_paths(cfg: dict) -> dict:
    """Return a shallow copy of cfg with known path keys made relative to SESSION_PATH.

    Call this immediately before writing cfg out to config.yaml.
    """
    session_path = str(cfg.get("SESSION_PATH", "") or "")
    out = dict(cfg)
    for key in CONFIG_PATH_KEYS:
        if key in out and out[key]:
            out[key] = to_relative_path(out[key], session_path)
    if "CREATE_DATASET_SOURCE_DIRS" in out:
        out["CREATE_DATASET_SOURCE_DIRS"] = _map_source_dirs(
            out["CREATE_DATASET_SOURCE_DIRS"], to_relative_path, session_path
        )
    return out
