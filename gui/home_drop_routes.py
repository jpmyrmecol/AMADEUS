# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Classify Home panel drops without importing GUI or opening input files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


_YAML = frozenset({".yaml", ".yml"})
_SEGMENTATION_CONFIG = "segmentation_gui_config.json"
_CROP_CONFIG_SUFFIXES = ("_cropping_trimming_config.yaml", "_cropping_trimming_config.yml")
_TRACKING_PANELS = frozenset({
    "gui_easy_tracking.py",
    "gui_advanced_tracking.py",
    "gui_refinement.py",
    "gui_multi_config_batch.py",
})
_VIDEO_PANELS = frozenset({
    "gui_easy_tracking.py",
    "gui_preprocess.py",
    "gui_segmentation.py",
    "gui_advanced_tracking.py",
    "gui_refinement.py",
    "gui_colab.py",
})


def home_drop_suffixes(script: str, video_suffixes: Iterable[str]) -> tuple[str, ...]:
    """Return the file extensions accepted by a Home panel's drop target."""
    extensions: set[str] = set()
    if script in _VIDEO_PANELS:
        extensions.update(str(s).lower() for s in video_suffixes)
    if script in _TRACKING_PANELS or script in {"gui_preprocess.py", "gui_colab.py"}:
        extensions.update(_YAML)
    if script == "gui_segmentation.py":
        extensions.add(".json")
    return tuple(sorted(extensions))


def home_drop_args(script: str, paths: Iterable[str], video_suffixes: Iterable[str]) -> tuple[str, ...]:
    """Translate dropped files into the target GUI's supported CLI arguments.

    YAML configs and the Segmentation JSON are never treated as videos.
    The Multi Config Batch panel can accept several YAML files in one drop.
    """
    selected = tuple(str(Path(p).expanduser().resolve()) for p in paths)
    if not selected:
        raise ValueError("Drop an AMADEUS video or config file.")
    if len(selected) > 1 and script != "gui_multi_config_batch.py":
        raise ValueError("Drop only one file on this panel at a time.")

    video = frozenset(str(s).lower() for s in video_suffixes)
    arguments: list[str] = []
    for path in selected:
        name = Path(path).name.lower()
        extension = Path(path).suffix.lower()
        if extension in video and script in _VIDEO_PANELS:
            arguments.extend(("--dropped-video", path))
        elif extension in _YAML:
            crop_config = name.endswith(_CROP_CONFIG_SUFFIXES)
            if script == "gui_preprocess.py" and crop_config:
                arguments.extend(("--load-config", path))
            elif script == "gui_colab.py" and not crop_config:
                arguments.extend(("--config", path))
            elif script in _TRACKING_PANELS and not crop_config:
                arguments.extend(("--load-config", path))
            else:
                raise ValueError(
                    "This panel does not accept that YAML. Use a tracking config.yaml "
                    "on Easy/Advanced/Refinement/Multi Config Batch/Colab, or "
                    "a *_cropping_trimming_config.yaml on Cropping & Trimming."
                )
        elif script == "gui_segmentation.py" and extension == ".json" and name == _SEGMENTATION_CONFIG:
            arguments.extend(("--load-config", path))
        else:
            raise ValueError(
                "Unsupported file for this panel. Segmentation accepts "
                "segmentation_gui_config.json; other config panels accept YAML."
            )
    return tuple(arguments)
