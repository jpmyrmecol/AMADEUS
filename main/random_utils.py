# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np


def normalize_seed(value: Any, default: int = 0) -> int:
    """Normalize a user-supplied seed value to a non-negative int.

    ``None`` or an empty string fall back to ``default``. ``0`` is a valid
    seed and must never be treated as "unspecified". Negative integers are
    accepted but folded into a stable non-negative value. Anything else that
    cannot be parsed as an integer raises ``ValueError`` rather than being
    silently coerced.
    """
    if value is None:
        return int(default)
    if isinstance(value, str):
        if value.strip() == "":
            return int(default)
        value = int(value.strip())
    if isinstance(value, bool):
        # bool is a subclass of int; treat explicitly to avoid True/False
        # silently becoming 1/0 seeds through the isinstance(int) branch.
        return int(value)
    if isinstance(value, int):
        # Fold into the 32-bit range unconditionally: downstream consumers
        # (numpy.random.seed, torch/ultralytics seeding) require 0 <= seed <
        # 2**32, but derive_seed() (and user-supplied config values) can
        # produce/pass values far outside that range.
        if value < 0:
            value = -value
        return value & 0xFFFFFFFF
    if isinstance(value, float) and value.is_integer():
        return normalize_seed(int(value), default=default)
    raise ValueError(f"Invalid RANDOM_SEED value: {value!r}")


def derive_seed(master_seed: int, *parts: object) -> int:
    """Deterministically derive a sub-seed from a master seed and context.

    Uses blake2b rather than Python's built-in ``hash()`` so the result does
    not depend on the process's ``PYTHONHASHSEED`` and is stable across runs,
    processes, and machines.
    """
    payload = "\x1f".join(
        [str(int(master_seed)), *(str(part) for part in parts)]
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def make_python_rng(master_seed: int, *parts: object) -> random.Random:
    """Build a standalone ``random.Random`` seeded deterministically."""
    return random.Random(derive_seed(master_seed, *parts))


def make_numpy_rng(master_seed: int, *parts: object) -> np.random.Generator:
    """Build a standalone ``numpy`` random Generator seeded deterministically."""
    seed = derive_seed(master_seed, *parts) % (2**32)
    return np.random.default_rng(seed)
