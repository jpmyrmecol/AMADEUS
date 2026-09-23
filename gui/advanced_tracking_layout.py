# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Declarative three-column layouts for the Advanced Tracking GUI."""

from __future__ import annotations

from collections.abc import Iterable


# Each entry is a logical three-column matrix. Empty columns are intentional
# for small categories. The renderer uses the same three grid anchors for all
# categories and only creates cells for fields that exist.
CATEGORY_COLUMN_LAYOUT: dict[tuple[str, str], tuple[tuple[str, ...], ...]] = {
    ("Initial Tracking", "Initial Tracking"): (
        ("AUTO_PARAMS", "LOCALIZED_RATIO"),
        ("INIT_MAX_GAP", "OBB_FIT_MODE"),
        ("SKIP_INIT_PREVIEW",),
    ),
    ("Create single animal images", "Trajectory / Direction Filtering"): (
        ("TRAJ_MAX_DIST", "MIN_ASPECT"),
        ("DIR_MIN_SEC", "DIR_MIN_DISP"),
        ("TRAJ_MAX_JUMP", "SKIP_DIR_PREVIEW"),
    ),
    ("Create single animal images", "Refine blobs through training"): (
        ("REFINE_FRAME_RATIO", "REFINE_EPOCHS", "REFINE_BATCH"),
        ("REFINE_ITERS", "REFINE_MODEL", "REFINE_CONF"),
        ("RUN_DELETE_RATIO", "SKIP_REFINE_PREVIEW"),
    ),
    ("Create with crossing", "Random noise on animals"): (
        ("NOISE_ENABLE",),
        ("NOISE_SIZE_PERCENT",),
        ("NOISE_MAX_COUNT",),
    ),
    ("Create with crossing", "Paste composition per base image"): (
        ("RATIO_SINGLE", "RATIO_P2", "RATIO_P3"),
        ("FREE_SCALE", "FREE_RATIO_SINGLE"),
        ("FREE_RATIO_P2", "FREE_RATIO_P3"),
    ),
    ("Create with crossing", "Paste placement / appearance"): (
        (
            "MAX_OVERLAP",
            "PASTE_SCALE_MIN",
            "PASTE_SCALE_MAX",
            "WIDTH_SCALE_MIN",
            "WIDTH_SCALE_MAX",
            "MASK_EXPANSION_RATIO",
        ),
        (
            "BRIGHT_MIN",
            "BRIGHT_MAX",
            "CONTRAST_MIN",
            "CONTRAST_MAX",
            "PASTE_LAYER_MODE",
            "UNDER_PASTE_PROB",
        ),
        (
            "OCCLUDER_MARGIN",
            "ALPHA_MODE",
            "FEATHER_MIN",
            "FEATHER_MAX",
            "EDGE_BLUR_KSIZE",
            "EDGE_BLUR_SIGMA",
            "MAX_TRIES",
        ),
    ),
    ("Create with crossing", "Clustered paste"): (
        ("CLUSTERED_RATIO", "CLUSTER_COUNT"),
        ("CLUSTER_FRAMES", "CLUSTER_FIT_LONG"),
        ("CLUSTER_FIT_SHORT", "CLUSTER_BREAK_PROB"),
    ),
    ("Create dataset", "Crop Images"): (
        ("NUM_CROPS",),
        ("LOCALIZED",),
        (),
    ),
    ("Create dataset", "Create Dataset Settings"): (
        ("VAL_RATIO",),
        (),
        (),
    ),
    ("Training", "YOLO Training"): (
        ("EPOCHS", "SAVE_PERIOD", "BATCH_SIZE"),
        ("PRETRAINED_MODEL", "DEVICE", "LR0"),
        ("LRF", "ACCEPT_RESUME"),
    ),
    ("Analysis", "Detection"): (
        ("DEVICE", "BATCH_SIZE"),
        ("WEIGHT", "CONF"),
        ("NMS_IOU", "SKIP_DETECT_PREVIEW"),
    ),
    ("Analysis", "Tracking"): (
        ("MATCH_IOU", "MATCH_ANGLE", "MAX_AXIS_ERR", "MAX_AGE"),
        ("FLIP_SEC", "INTERACT_IOU", "IOU_WEIGHT"),
        ("DIRECTION_WEIGHT", "MISS_WEIGHT", "DISTANCE_WEIGHT"),
    ),
    ("Analysis", "Embedding"): (
        ("ENABLE", "IMG_SIZE"),
        ("EMBED_DEVICE",),
        ("PREVIEW_COUNT",),
    ),
    ("Create Video", "Draw Elements"): (
        ("DRAW_OBB", "DRAW_MODE"),
        ("DRAW_LABELS",),
        ("DRAW_ARROW",),
    ),
    ("Create Video", "Video"): (
        ("WEIGHT", "FRAME_STEP", "FPS"),
        ("ACCELERATION", "EXPORT_RAW"),
        ("EXPORT_IMAGES", "IMAGE_FORMAT"),
    ),
    ("Create Video", "Appearance"): (
        ("OBB_WIDTH", "ARROW_WIDTH"),
        ("ARROW_ALPHA", "ARROW_SCALE"),
        ("LABEL_SCALE", "LABEL_THICKNESS"),
    ),
}


def partition_category_items(
    title: str,
    category: str,
    items: Iterable[tuple[str, str, object]],
) -> tuple[tuple[tuple[str, str, object], ...], ...]:
    """Return category fields in its logical three columns and validate it."""

    item_list = list(items)
    layout = CATEGORY_COLUMN_LAYOUT.get((title, category))
    if layout is None:
        return (tuple(item_list), (), ())

    actual_keys = [item[0] for item in item_list]
    expected_keys = [key for column in layout for key in column]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError(f"duplicate field key in {title}/{category}")
    if actual_keys and set(actual_keys) != set(expected_keys):
        missing = sorted(set(actual_keys) - set(expected_keys))
        unknown = sorted(set(expected_keys) - set(actual_keys))
        raise ValueError(
            f"layout mismatch for {title}/{category}: "
            f"unassigned={missing}, unknown={unknown}"
        )
    if len(actual_keys) != len(expected_keys):
        raise ValueError(f"layout field count mismatch for {title}/{category}")

    by_key = {item[0]: item for item in item_list}
    return tuple(
        tuple(by_key[key] for key in column)
        for column in layout
    )
