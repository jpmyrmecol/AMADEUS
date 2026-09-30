# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Installed/latest AMADEUS version helpers used by logs and status displays."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LATEST_RELEASE_URL = "https://api.github.com/repos/jpmyrmecol/AMADEUS/releases/latest"
VERSION_PATTERN = re.compile(
    r"v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z", re.IGNORECASE
)


@dataclass(frozen=True)
class VersionStatus:
    current: str
    latest: str | None
    state: str
    detail: str = ""

    @property
    def update_available(self) -> bool:
        return self.state == "update_available"


def version_tuple(value: str) -> tuple[int, int, int]:
    match = VERSION_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Unsupported AMADEUS version format: {value!r}")
    return tuple(int(part) for part in match.groups())


def version_label(value: str) -> str:
    return f"v{value.strip().removeprefix('v').removeprefix('V')}"


def installed_version() -> str:
    version = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    version_tuple(version)
    return version


def fetch_latest_version(*, timeout: float = 2.0) -> str:
    """Return the version of GitHub's latest published AMADEUS Release."""
    request = urllib.request.Request(
        LATEST_RELEASE_URL,
        headers={
            "User-Agent": "AMADEUS-Version-Check",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read(64 * 1024 + 1)
    if len(payload) > 64 * 1024:
        raise ValueError("GitHub returned an unexpectedly large release response.")
    release = json.loads(payload.decode("utf-8"))
    tag = str(release.get("tag_name", "")).strip()
    version_tuple(tag)
    return tag.removeprefix("v").removeprefix("V")


def check_version_status(*, timeout: float = 2.0) -> VersionStatus:
    current = installed_version()
    try:
        latest = fetch_latest_version(timeout=timeout)
    except Exception as exc:
        return VersionStatus(
            current=current,
            latest=None,
            state="offline_or_unavailable",
            detail=f"{type(exc).__name__}: {exc}",
        )

    current_tuple = version_tuple(current)
    latest_tuple = version_tuple(latest)
    if latest_tuple > current_tuple:
        state = "update_available"
    elif latest_tuple == current_tuple:
        state = "up_to_date"
    else:
        state = "installed_newer_than_remote"
    return VersionStatus(current=current, latest=latest, state=state)
