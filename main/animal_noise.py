# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Apply background-coloured noise only after scene composition is complete."""

from typing import Tuple

import cv2
import numpy as np
from random_utils import make_numpy_rng, normalize_seed

DEFAULT_NOISE_SIZE_PERCENT = 10.0
DEFAULT_NOISE_MAX_COUNT = 10
NOISE_POLYGON_VERTICES = 32


def get_noise_settings(cfg: dict) -> Tuple[bool, float, int]:
    enabled = bool(cfg.get("NOISE_ENABLE", False))
    size_percent = float(cfg.get("NOISE_SIZE_PERCENT", DEFAULT_NOISE_SIZE_PERCENT))
    max_count = int(cfg.get("NOISE_MAX_COUNT", DEFAULT_NOISE_MAX_COUNT))
    if enabled:
        if not (size_percent > 0.0):
            raise ValueError("NOISE_SIZE_PERCENT must be > 0 when NOISE_ENABLE is on.")
        if max_count < 1:
            raise ValueError("NOISE_MAX_COUNT must be >= 1 when NOISE_ENABLE is on.")
    return enabled, size_percent, max_count


def random_noise_polygon(rng: np.random.Generator, center: Tuple[float, float], radius: float) -> np.ndarray:
    """A random polygon inscribed in the disk of the given radius.

    Every vertex sits at a fixed angle and a random radius, so the silhouette
    ranges from jagged to nearly round (the per-patch lower bound decides
    which) while never exceeding the requested size.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, NOISE_POLYGON_VERTICES, endpoint=False)
    lower = float(rng.uniform(0.0, 0.9))
    radii = radius * rng.uniform(lower, 1.0, size=NOISE_POLYGON_VERTICES)
    xs = center[0] + radii * np.cos(angles)
    ys = center[1] + radii * np.sin(angles)
    return np.round(np.stack([xs, ys], axis=1)).astype(np.int32)


def load_noise_background(cfg: dict):
    enabled, _, _ = get_noise_settings(cfg)
    if not enabled:
        return None
    background = cv2.imread(str(cfg["BACKGROUND_PATH"]), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"Cannot read noise background: {cfg['BACKGROUND_PATH']}")
    return background


def apply_animal_noise(image, mask_lines, background, cfg, stage, frame_id, repeat_index):
    """Draw fresh patches per output, leaving source material and annotations intact."""
    enabled, size_percent, max_count = get_noise_settings(cfg)
    if not enabled:
        return image
    if background is None or background.shape != image.shape:
        raise ValueError("Noise background and composed image must have identical shapes.")
    rng = make_numpy_rng(normalize_seed(cfg.get("RANDOM_SEED", 0)),
                         stage, "animal_noise", frame_id, repeat_index)
    polygons = []
    for line in mask_lines:
        parts = line.strip().split(" | ")
        metadata = dict(item.split("=", 1) for item in parts[0].split())
        length = float(metadata["axis_length"])
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Animal noise requires a positive finite axis_length.")
        count = int(rng.integers(0, max_count + 1))
        radius = 0.5 * length * size_percent / 100.0
        if count == 0 or radius < 1.0:
            continue
        contours = [np.asarray([tuple(map(int, xy.split(",")))
                    for xy in part.removeprefix("points=").split()], dtype=np.int32)
                    for part in parts[1:]]
        if not contours:
            continue
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        filled = np.zeros((h, w), dtype=np.uint8)
        for contour in contours:
            cv2.fillPoly(filled, [contour - [x, y]], 255)
        ys, xs = np.nonzero(filled)
        if not len(ys):
            continue
        picks = rng.integers(0, len(ys), size=count)
        for i in picks:
            polygons.append(random_noise_polygon(rng, (float(xs[i] + x), float(ys[i] + y)), radius))
    if not polygons:
        return image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255)
    result = image.copy()
    hit = mask > 0
    result[hit] = background[hit]
    return result
