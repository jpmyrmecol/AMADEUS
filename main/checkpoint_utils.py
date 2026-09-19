# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""One-based AMADEUS checkpoint naming and validation.

AMADEUS configuration values, weight filenames, and tracking run directories
are all 1-based: checkpoint ``1`` is ``epoch1.pt``.  Only the epoch value stored
inside an Ultralytics checkpoint is 0-based; obb_detector_obb_detector_training.py converts that metadata
at the framework boundary.
"""

from __future__ import annotations

import math
import numbers
import re
from typing import Any


SPECIAL_CHECKPOINTS = frozenset({"best", "last"})


def checkpoint_number_to_weight(number: Any) -> str:
    """Convert a positive, 1-based checkpoint number to an AMADEUS weight stem."""
    if isinstance(number, bool):
        raise ValueError(f"Checkpoint number must be an integer >= 1, got {number!r}")
    if isinstance(number, numbers.Real):
        numeric = float(number)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"Checkpoint number must be an integer >= 1, got {number!r}")
        completed_epoch = int(numeric)
    else:
        text = str(number).strip()
        if not text.isdigit():
            raise ValueError(f"Checkpoint number must be an integer >= 1, got {number!r}")
        completed_epoch = int(text)
    if completed_epoch < 1:
        raise ValueError(f"Checkpoint number must be >= 1, got {completed_epoch}")
    return f"epoch{completed_epoch}"


def weight_to_checkpoint_number(weight: Any) -> int:
    """Return the 1-based number encoded by an AMADEUS ``epochN`` stem."""
    match = re.fullmatch(r"epoch([1-9]\d*)", str(weight).strip().lower())
    if not match:
        raise ValueError(f"Invalid epoch weight stem: {weight!r}")
    return int(match.group(1))


def normalize_checkpoint_weight(weight: Any) -> str:
    """Validate a 1-based AMADEUS checkpoint stem without changing it."""
    normalized = str(weight).strip().lower()
    if normalized in SPECIAL_CHECKPOINTS:
        return normalized
    match = re.fullmatch(r"epoch([1-9]\d*)", normalized)
    if not match:
        raise ValueError(f"Invalid normalized checkpoint value: {weight!r}")
    return f"epoch{int(match.group(1))}"


def parse_checkpoint_spec(value: Any, *, key: str = "WEIGHT") -> list[str]:
    """Parse one or more 1-based checkpoint selectors into internal stems."""
    if value is None:
        raise ValueError(f"{key} is required.")
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            raise ValueError(f"{key} is empty.")
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError as exc:
            raise ValueError(f"Invalid {key} type: {type(value).__name__}") from exc

    parsed: list[str] = []
    seen: set[str] = set()
    for item in items:
        special = str(item).strip().lower()
        weight = special if special in SPECIAL_CHECKPOINTS else checkpoint_number_to_weight(item)
        if weight not in seen:
            seen.add(weight)
            parsed.append(weight)

    if not parsed:
        raise ValueError(f"{key} must contain at least one checkpoint selector.")
    return parsed


def checkpoint_spec_for_config(value: Any, *, key: str = "WEIGHT") -> int | str | list[int | str]:
    """Validate GUI input and return canonical 1-based YAML selector values."""
    selectors: list[int | str] = []
    for weight in parse_checkpoint_spec(value, key=key):
        selectors.append(weight if weight in SPECIAL_CHECKPOINTS else weight_to_checkpoint_number(weight))
    return selectors[0] if len(selectors) == 1 else selectors
