# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Installed/latest AMADEUS version helpers used by logs and status displays."""

from __future__ import annotations

import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LATEST_RELEASE_URL = "https://github.com/jpmyrmecol/AMADEUS/releases/latest"
LATEST_RELEASE_TAG_PATH = "/jpmyrmecol/AMADEUS/releases/tag/"
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


def _https_context() -> ssl.SSLContext:
    """Use certifi plus the operating system trust store for GitHub HTTPS."""
    context = ssl.create_default_context(cafile=certifi.where())
    context.load_default_certs()
    return context


def fetch_latest_version(*, timeout: float = 2.0) -> str:
    """Return the version of GitHub's latest published AMADEUS Release."""
    request = urllib.request.Request(
        LATEST_RELEASE_URL,
        headers={"User-Agent": "AMADEUS-Version-Check", "Accept": "text/html"},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=timeout, context=_https_context()) as response:
        final_url = response.geturl()

    parsed = urllib.parse.urlparse(final_url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError(f"GitHub redirected the latest release to an unexpected URL: {final_url}")
    if not parsed.path.startswith(LATEST_RELEASE_TAG_PATH):
        raise ValueError(f"GitHub did not redirect to a release tag: {final_url}")

    tag = urllib.parse.unquote(parsed.path[len(LATEST_RELEASE_TAG_PATH):]).strip("/")
    if not tag or "/" in tag:
        raise ValueError(f"GitHub returned an invalid release tag URL: {final_url}")
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
