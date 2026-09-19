# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import colorsys
import random
from functools import lru_cache
from typing import Literal

Color = tuple[int, int, int]
ColorSpace = Literal["bgr", "rgb"]
_Lab = tuple[float, float, float]

HUE_STEP = 3
SATURATION_LEVELS = (0.75, 0.85, 0.95)
VALUE_LEVELS = (0.76, 0.84, 0.92)

# OpenCV BGR
CYAN: Color = (210, 165, 0)
ORANGE: Color = (0, 140, 242)
GREEN: Color = (60, 220, 60)
WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)

# Meaning aliases
OBB_COLOR: Color = CYAN
OUTLIER_COLOR: Color = ORANGE
DELETION_COLOR: Color = ORANGE
ROI_COLOR: Color = GREEN


def bgr_to_rgb(color: Color) -> Color:
    return color[2], color[1], color[0]


def rgb_to_bgr(color: Color) -> Color:
    return color[2], color[1], color[0]


CYAN_RGB: Color = bgr_to_rgb(CYAN)
ORANGE_RGB: Color = bgr_to_rgb(ORANGE)
GREEN_RGB: Color = bgr_to_rgb(GREEN)
WHITE_RGB: Color = bgr_to_rgb(WHITE)
BLACK_RGB: Color = bgr_to_rgb(BLACK)
OBB_COLOR_RGB: Color = bgr_to_rgb(OBB_COLOR)
OUTLIER_COLOR_RGB: Color = bgr_to_rgb(OUTLIER_COLOR)
DELETION_COLOR_RGB: Color = bgr_to_rgb(DELETION_COLOR)
ROI_COLOR_RGB: Color = bgr_to_rgb(ROI_COLOR)


def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _hsv_to_rgb_color(hue_degrees: int, saturation: float, value: float) -> Color:
    r, g, b = colorsys.hsv_to_rgb(
        (int(hue_degrees) % 360) / 360.0,
        float(saturation),
        float(value),
    )
    return (
        _clamp_channel(r * 255.0),
        _clamp_channel(g * 255.0),
        _clamp_channel(b * 255.0),
    )


def _srgb_channel_to_linear(channel: int) -> float:
    c = float(channel) / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _rgb_to_oklab(color: Color) -> _Lab:
    r = _srgb_channel_to_linear(color[0])
    g = _srgb_channel_to_linear(color[1])
    b = _srgb_channel_to_linear(color[2])

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = l ** (1.0 / 3.0)
    m_ = m ** (1.0 / 3.0)
    s_ = s ** (1.0 / 3.0)

    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _oklab_distance_squared(a: _Lab, b: _Lab) -> float:
    dl = a[0] - b[0]
    da = a[1] - b[1]
    db = a[2] - b[2]
    return dl * dl + da * da + db * db


@lru_cache(maxsize=1)
def _hsv_candidate_colors_rgb() -> tuple[tuple[Color, _Lab], ...]:
    candidates: list[tuple[Color, _Lab]] = []
    seen: set[Color] = {CYAN_RGB, ORANGE_RGB}
    for hue in range(0, 360, HUE_STEP):
        for saturation in SATURATION_LEVELS:
            for value in VALUE_LEVELS:
                rgb = _hsv_to_rgb_color(hue, saturation, value)
                if rgb in seen:
                    continue
                seen.add(rgb)
                candidates.append((rgb, _rgb_to_oklab(rgb)))
    return tuple(candidates)


@lru_cache(maxsize=1)
def _id_palette_sequence_bgr() -> tuple[Color, ...]:
    selected_bgr: list[Color] = [CYAN, ORANGE]
    selected_labs: list[_Lab] = [_rgb_to_oklab(CYAN_RGB), _rgb_to_oklab(ORANGE_RGB)]
    candidates = _hsv_candidate_colors_rgb()
    remaining = list(range(len(candidates)))
    min_distances = [
        min(_oklab_distance_squared(lab, selected_lab) for selected_lab in selected_labs)
        for _, lab in candidates
    ]

    while remaining:
        best_pos = 0
        best_distance = -1.0
        for pos, candidate_index in enumerate(remaining):
            distance = min_distances[candidate_index]
            if distance > best_distance:
                best_pos = pos
                best_distance = distance

        candidate_index = remaining.pop(best_pos)
        rgb, lab = candidates[candidate_index]
        selected_bgr.append(rgb_to_bgr(rgb))
        selected_labs.append(lab)

        for candidate_index in remaining:
            distance = _oklab_distance_squared(candidates[candidate_index][1], lab)
            if distance < min_distances[candidate_index]:
                min_distances[candidate_index] = distance

    return tuple(selected_bgr)


def resolve_color_seed(value: object) -> int | None:
    """Validate a configured color-shuffle seed, preserving the default when blank."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("COLOR_SEED must be an integer, not a boolean")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"COLOR_SEED must be an integer, got {value!r}")
        return int(value)

    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 10)
    except ValueError as exc:
        raise ValueError(f"COLOR_SEED must be an integer, got {value!r}") from exc


def make_id_palette(
    count: int,
    *,
    color_space: ColorSpace = "bgr",
    seed: int | str | float | None = None,
) -> list[Color]:
    if count <= 0:
        return []

    palette = _id_palette_sequence_bgr()
    if count > len(palette):
        raise ValueError(f"count={count} exceeds available unique ID colors ({len(palette)})")

    colors = list(palette[:count])
    normalized_seed = resolve_color_seed(seed)
    if normalized_seed is not None:
        random.Random(normalized_seed).shuffle(colors)

    if color_space == "bgr":
        return colors
    if color_space == "rgb":
        return [bgr_to_rgb(color) for color in colors]
    raise ValueError(f"Unsupported color_space: {color_space!r}")
