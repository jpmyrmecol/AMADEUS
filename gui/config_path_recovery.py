# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Recover project paths when a saved AMADEUS config is moved."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from typing import Any

try:
    from .video_input import VIDEO_DROP_SUFFIXES, VIDEO_FILETYPES
except ImportError:  # Preserve direct execution of the GUIs that import this.
    from video_input import VIDEO_DROP_SUFFIXES, VIDEO_FILETYPES


_CONFIG_PATH_KEYS = (
    "TRAINING_VIDEO_PATH",
    "TRACKING_VIDEO_PATH",
    "PICKLE_PATH",
    "BACKGROUND_PATH",
    "INIT_CSV_PATH",
    "YOLO_DATASET_DIR",
)
_VIDEO_KEYS = ("TRAINING_VIDEO_PATH", "TRACKING_VIDEO_PATH")
_VIDEO_SUFFIXES = VIDEO_DROP_SUFFIXES
_SKIP_SEARCH_DIRS = frozenset({".git", ".venv", "venv", "__pycache__"})


@dataclass(frozen=True)
class ConfigRelocation:
    config_path: str
    old_config_dir: str
    new_config_dir: str
    old_session_path: str
    new_session_path: str
    raw_values: dict[str, Any]
    relocated: bool


def _normalized_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(text)))


def _same_path(left: str, right: str) -> bool:
    return bool(left and right) and os.path.normcase(left) == os.path.normcase(right)


def _is_within(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        return os.path.commonpath((os.path.normcase(path), os.path.normcase(root))) == os.path.normcase(root)
    except ValueError:
        return False


def _resolve_config_paths(cfg: dict) -> dict:
    """Use the project's path resolver without imposing import order on GUI scripts."""
    try:
        from main.path_utils import resolve_config_paths
    except ImportError:
        from path_utils import resolve_config_paths
    return resolve_config_paths(cfg)


def _resolve_session_path(raw_session: object, config_dir: str) -> str:
    text = str(raw_session or "").strip()
    if not text:
        return ""
    if os.path.isabs(os.path.expanduser(text)):
        return _normalized_path(text)
    return _normalized_path(os.path.join(config_dir, text))


def _relocate_absolute_path(path: str, old_config_dir: str, new_config_dir: str) -> str:
    """Map a path inside the old project tree to the new config location."""
    if not path or not old_config_dir or not new_config_dir:
        return path
    old_project_dir = os.path.dirname(old_config_dir)
    if not _is_within(path, old_project_dir):
        return path
    try:
        relative = os.path.relpath(path, old_config_dir)
    except ValueError:
        return path
    return os.path.normpath(os.path.join(new_config_dir, relative))


def _relocate_value(value: object, old_config_dir: str, new_config_dir: str) -> object:
    if isinstance(value, list):
        return [_relocate_value(item, old_config_dir, new_config_dir) for item in value]
    if not isinstance(value, str) or not value.strip() or not os.path.isabs(os.path.expanduser(value)):
        return value
    path = _normalized_path(value)
    return _relocate_absolute_path(path, old_config_dir, new_config_dir)


def relocate_config_paths(cfg: dict, config_path: str) -> tuple[dict, ConfigRelocation]:
    """Relocate a config's session and project-relative absolute paths in memory.

    Configs written by the GUI place ``config.yaml`` under ``SESSION_PATH``.
    When the selected config is no longer under that stored directory, the
    selected config directory becomes the new anchor while preserving each
    path's position within the old tree.
    Relative config values remain relative and are resolved after relocation.
    """
    result = copy.deepcopy(cfg or {})
    selected_config = _normalized_path(config_path)
    new_config_dir = os.path.dirname(selected_config)
    raw_values = {
        key: copy.deepcopy(result.get(key))
        for key in (*_CONFIG_PATH_KEYS, "SESSION_PATH", "CREATE_DATASET_SOURCE_DIRS")
        if key in result
    }
    old_session_path = _resolve_session_path(result.get("SESSION_PATH", ""), new_config_dir)
    old_config_dir = old_session_path
    new_session_path = old_session_path
    relocated = False

    if old_session_path and not _same_path(old_session_path, new_config_dir):
        new_session_path = new_config_dir
        relocated = True
        result["SESSION_PATH"] = new_session_path
        for key in _CONFIG_PATH_KEYS:
            if key in result:
                result[key] = _relocate_value(result[key], old_config_dir, new_config_dir)
        if "CREATE_DATASET_SOURCE_DIRS" in result:
            result["CREATE_DATASET_SOURCE_DIRS"] = _relocate_value(
                result["CREATE_DATASET_SOURCE_DIRS"], old_config_dir, new_config_dir
            )
    elif old_session_path and not os.path.isabs(str(result.get("SESSION_PATH", "") or "")):
        # resolve_config_paths requires an absolute SESSION_PATH anchor.
        result["SESSION_PATH"] = old_session_path

    relocation = ConfigRelocation(
        config_path=selected_config,
        old_config_dir=old_config_dir,
        new_config_dir=new_config_dir,
        old_session_path=old_session_path,
        new_session_path=new_session_path,
        raw_values=raw_values,
        relocated=relocated,
    )
    return result, relocation


def _path_exists(path: object, *, directory: bool) -> bool:
    text = str(path or "").strip()
    if not text:
        return False
    return os.path.isdir(text) if directory else os.path.isfile(text)


def _candidate_paths(
    cfg: dict,
    key: str,
    relocation: ConfigRelocation,
) -> list[str]:
    raw = relocation.raw_values.get(key, "")
    current = str(cfg.get(key, "") or "").strip()
    candidates: list[str] = []

    def add(path: object) -> None:
        text = str(path or "").strip()
        if not text:
            return
        normalized = _normalized_path(text)
        if normalized not in candidates:
            candidates.append(normalized)

    add(current)
    if isinstance(raw, str) and raw.strip():
        if os.path.isabs(os.path.expanduser(raw)):
            add(_relocate_absolute_path(
                _normalized_path(raw), relocation.old_config_dir, relocation.new_config_dir,
            ))
        elif relocation.new_session_path:
            add(os.path.join(relocation.new_session_path, raw))
        add(os.path.join(relocation.new_config_dir, os.path.basename(raw)))
    if current:
        add(os.path.join(relocation.new_config_dir, os.path.basename(current)))
    if relocation.new_session_path:
        add(os.path.join(relocation.new_session_path, os.path.basename(current or raw)))
    return candidates


def _search_by_name(root: str, basename: str, *, directory: bool) -> list[str]:
    if not root or not basename or not os.path.isdir(root):
        return []
    matches: list[str] = []
    for current_root, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_SEARCH_DIRS]
        names = dirnames if directory else filenames
        if basename not in names:
            continue
        candidate = os.path.join(current_root, basename)
        if directory or os.path.isfile(candidate):
            matches.append(os.path.normpath(candidate))
    return matches


def _find_video_path(cfg: dict, key: str, relocation: ConfigRelocation) -> str | None:
    directory = key == "TRACKING_VIDEO_PATH" and bool(cfg.get("TRACKING_VIDEO_PATH_IS_DIR", False))
    candidates = _candidate_paths(cfg, key, relocation)
    for candidate in candidates:
        if _path_exists(candidate, directory=directory):
            return candidate

    raw = relocation.raw_values.get(key, "")
    basename = os.path.basename(os.path.normpath(str(raw or "")))
    if not basename:
        basename = os.path.basename(os.path.normpath(str(cfg.get(key, "") or "")))
    if not directory and os.path.splitext(basename)[1].lower() not in _VIDEO_SUFFIXES:
        return None

    roots = [relocation.new_config_dir]
    if relocation.new_session_path and not _same_path(relocation.new_session_path, relocation.new_config_dir):
        roots.append(relocation.new_session_path)
    project_dir = os.path.dirname(relocation.new_config_dir)
    if project_dir and not any(_same_path(project_dir, root) for root in roots):
        roots.append(project_dir)
    matches: list[str] = []
    for root in roots:
        for match in _search_by_name(root, basename, directory=directory):
            if match not in matches:
                matches.append(match)
    return matches[0] if len(matches) == 1 else None


def _ask_for_video_path(parent, key: str, current_path: str, *, directory: bool) -> str | None:
    kind = "tracking video folder" if directory else "video file"
    answer = messagebox.askyesno(
        "Video path not found",
        f"The configured {kind} for {key} was not found:\n{current_path or '(empty)'}\n\n"
        "Select a new path?\nYes: select a path\nNo: continue with the current path",
        parent=parent,
    )
    if not answer:
        return None

    initial_dir = os.path.dirname(current_path) if current_path else ""
    if not os.path.isdir(initial_dir):
        initial_dir = ""
    if directory:
        return filedialog.askdirectory(
            parent=parent,
            title="Select tracking video folder",
            initialdir=initial_dir or None,
        ) or None
    return filedialog.askopenfilename(
        parent=parent,
        title="Select video file",
        initialdir=initial_dir or None,
        initialfile=os.path.basename(current_path) if current_path else "",
        filetypes=VIDEO_FILETYPES,
    ) or None


def recover_missing_video_paths(cfg: dict, relocation: ConfigRelocation, parent=None) -> dict:
    """Find moved videos and ask whether unresolved paths should be replaced."""
    replaced_for_source: dict[tuple[str, bool], str] = {}
    skipped_sources: set[tuple[str, bool]] = set()
    for key in _VIDEO_KEYS:
        if key not in cfg and key not in relocation.raw_values:
            continue
        current = str(cfg.get(key, "") or "").strip()
        directory = key == "TRACKING_VIDEO_PATH" and bool(cfg.get("TRACKING_VIDEO_PATH_IS_DIR", False))
        if _path_exists(current, directory=directory):
            continue

        source_key = (current, directory)
        if source_key in replaced_for_source:
            cfg[key] = replaced_for_source[source_key]
            continue
        if source_key in skipped_sources:
            continue

        recovered = _find_video_path(cfg, key, relocation)
        if recovered:
            cfg[key] = recovered
            replaced_for_source[source_key] = recovered
            continue

        replacement = _ask_for_video_path(parent, key, current, directory=directory)
        if replacement:
            cfg[key] = os.path.normpath(os.path.abspath(os.path.expanduser(replacement)))
            replaced_for_source[source_key] = cfg[key]
        else:
            skipped_sources.add(source_key)
    return cfg


def prepare_config_for_gui(cfg: dict, config_path: str, parent=None) -> dict:
    """Relocate, resolve, and recover video paths for an interactive GUI load."""
    relocated_cfg, relocation = relocate_config_paths(cfg, config_path)
    resolved_cfg = _resolve_config_paths(relocated_cfg)
    return recover_missing_video_paths(resolved_cfg, relocation, parent=parent)
