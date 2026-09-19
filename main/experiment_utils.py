# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any

from random_utils import normalize_seed


DEFAULT_LR0 = 0.01
DEFAULT_LRF = 1.0


def _parse_lr_values(
    value: Any,
    *,
    key: str,
    label: str,
    max_value: float | None = None,
) -> list[float]:
    """Parse one or more positive LR values, validate them, and remove duplicates."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"training.{key} must not be empty.")
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"training.{key} must contain at least one value.")
        items = list(value)
    else:
        items = [value]

    values: list[float] = []
    seen: set[float] = set()
    for item in items:
        if isinstance(item, bool) or item is None:
            raise ValueError(f"Invalid training.{key} value: {item!r}")
        if isinstance(item, str) and not item.strip():
            raise ValueError(f"training.{key} contains an empty value.")
        try:
            parsed = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid training.{key} value: {item!r}") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"training.{key} must be finite, got {item!r}.")
        if parsed <= 0.0 or (max_value is not None and parsed > max_value):
            if max_value is None:
                raise ValueError(f"training.{key} must satisfy {label} > 0, got {item!r}.")
            raise ValueError(
                f"training.{key} must satisfy 0 < {label} <= {max_value:g}, got {item!r}."
            )
        if parsed not in seen:
            seen.add(parsed)
            values.append(parsed)

    if not values:
        raise ValueError(f"training.{key} must contain at least one value.")
    return values


def parse_lr0_values(value: Any) -> list[float]:
    """Parse one or more YOLO initial learning-rate values."""
    return _parse_lr_values(value, key="LR0", label="LR0")


def parse_lrf_values(value: Any) -> list[float]:
    """Parse one or more YOLO final LR factor values."""
    return _parse_lr_values(value, key="LRF", label="LRF", max_value=1.0)


def require_single_lr0(value: Any) -> float:
    """Return one validated LR0, rejecting a batch list explicitly."""
    values = parse_lr0_values(value)
    if len(values) != 1:
        raise ValueError(
            "training.LR0 contains multiple values. Run the configuration through batch.py."
        )
    return values[0]


def require_single_lrf(value: Any) -> float:
    """Return one validated LRF, rejecting a batch list explicitly."""
    values = parse_lrf_values(value)
    if len(values) != 1:
        raise ValueError(
            "training.LRF contains multiple values. Run the configuration through batch.py."
        )
    return values[0]


def _format_lr_for_name(value: float, *, label: str) -> str:
    """Format a finite number as locale-independent, non-scientific decimal text."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {label} value for directory name: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    try:
        text = format(Decimal(str(number)), "f")
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid {label} value for directory name: {value!r}") from exc
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def format_lr0_for_name(value: float) -> str:
    return _format_lr_for_name(value, label="LR0")


def format_lrf_for_name(value: float) -> str:
    return _format_lr_for_name(value, label="LRF")


def experiment_dir_name(
    num_total_images: int,
    lrf: float,
    random_seed: int,
    lr0: float,
) -> str:
    """Build the canonical directory name for one training/tracking experiment."""
    validated_lr0 = require_single_lr0(lr0)
    validated_lrf = require_single_lrf(lrf)
    seed = normalize_seed(random_seed)
    return (
        f"dataset{int(num_total_images)}"
        f"_lr0{format_lr0_for_name(validated_lr0)}"
        f"_lrf{format_lrf_for_name(validated_lrf)}"
        f"_seed{seed}"
    )


def resolve_existing_experiment_dir_name(
    session_path: str,
    model_name: str,
    dataset_name: str,
    *,
    stages: tuple[str, ...] = ("tracking", "training"),
    warn_fn=None,
) -> str:
    """Return the explicitly requested canonical experiment name.

    Stage-boundary callers pass the surrounding paths for a consistent API, but
    no alternate directory names are inspected or migrated.
    """
    _ = session_path, model_name, stages, warn_fn
    return str(dataset_name or "").strip()


def experiment_dir_name_from_cfg(cfg: dict) -> str:
    """Build the canonical experiment name from a single-LR configuration."""
    training = cfg.get("training", {}) or {}
    return experiment_dir_name(
        cfg["NUM_IMAGES"],
        require_single_lrf(training.get("LRF", DEFAULT_LRF)),
        cfg.get("RANDOM_SEED", 0),
        require_single_lr0(training.get("LR0", DEFAULT_LR0)),
    )
