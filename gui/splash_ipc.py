# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

TOKEN_ENV_VAR = "AMADEUS_SPLASH_TOKEN"


def stop_file_path(token: str) -> Path:
    return Path(tempfile.gettempdir()) / "AMADEUS" / f"splash_stop_{token}"


def started_file_path(token: str) -> Path:
    return Path(tempfile.gettempdir()) / "AMADEUS" / f"splash_started_{token}"


def record_started(token: str) -> Path:
    """Record when an external splash process began its display setup."""
    marker = started_file_path(token)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{time.time():.9f}", encoding="ascii")
    return marker


def external_started_at() -> float | None:
    """Return the current external splash start time, if it has reported one."""
    token = os.environ.get(TOKEN_ENV_VAR)
    if not token:
        return None
    try:
        return float(started_file_path(token).read_text(encoding="ascii"))
    except (OSError, ValueError):
        return None


def signal_stop() -> None:
    """Tell the standalone splash launched for this run (if any) to close."""
    token = os.environ.get(TOKEN_ENV_VAR)
    if not token:
        return
    marker = stop_file_path(token)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass
