# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Store downloaded pretrained weights under <Session>/model/.

Resolve bare model names and redirect Ultralytics asset downloads to that
folder. Explicit paths are preserved; matching legacy downloads are reused.
Trained output weights are located by training_paths.py."""

from __future__ import annotations

import functools
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR_NAME = "model"
# Directories where older AMADEUS versions (or a bare Ultralytics download)
# could have left weights. They are reused instead of downloading again.
LEGACY_MODEL_DIRS = (
    PROJECT_ROOT / MODEL_DIR_NAME,
    PROJECT_ROOT,
    PROJECT_ROOT / "main",
)

_ORIGINAL_ATTR = "_amadeus_original_attempt_download_asset"


def session_model_dir(session_path) -> Path:
    """Return the directory holding the pretrained weights of one session."""
    value = str(session_path or "").strip()
    if not value:
        raise ValueError("session_path must not be empty")
    return Path(value).expanduser() / MODEL_DIR_NAME


def resolve_pretrained_model_path(model_name: str, session_path) -> str:
    """Return the path a pretrained model is loaded from (and downloaded to).

    Bare model names are stored under the session's ``model`` directory.
    Explicit absolute paths and paths containing a directory are preserved so
    users can still load their own weights from elsewhere.

    A matching file already downloaded into the AMADEUS installation by an
    older version is copied into the session instead of downloading it again.
    """
    value = str(model_name or "").strip()
    if not value:
        raise ValueError("model_name must not be empty")

    requested = Path(value).expanduser()
    if requested.is_absolute() or requested.parent != Path("."):
        return str(requested)

    model_dir = session_model_dir(session_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / requested.name
    if not destination.exists():
        legacy_path = _find_legacy_model(requested.name, destination)
        if legacy_path is not None:
            shutil.copy2(str(legacy_path), str(destination))
            print(f"[INFO] Reused an already downloaded pretrained model: {legacy_path}")

    return str(destination)


def redirect_asset_downloads(session_path) -> Path:
    """Send Ultralytics' own asset downloads into the session model directory.

    Ultralytics downloads a few assets without asking AMADEUS for a path -- the
    most visible one is the ``yolo11n.pt`` its AMP check fetches at the start of
    GPU training -- and resolves them relative to the current working
    directory. Patching the single resolution point keeps those files inside
    the session as well.

    The patch is process-wide and idempotent (a second call just repoints it at
    the new directory); every script that trains runs in its own subprocess.
    Returns the directory the downloads now land in.
    """
    model_dir = session_model_dir(session_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics.utils import downloads as ultralytics_downloads

    current = ultralytics_downloads.attempt_download_asset
    original = getattr(current, _ORIGINAL_ATTR, current)

    @functools.wraps(original)
    def attempt_download_asset(file, *args, **kwargs):
        return original(_redirect_asset(file, model_dir), *args, **kwargs)

    setattr(attempt_download_asset, _ORIGINAL_ATTR, original)
    # Modules that imported the function by name hold their own reference, so
    # rebind every one of them, not just ultralytics.utils.downloads.
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "attempt_download_asset", None) is current:
            module.attempt_download_asset = attempt_download_asset

    return model_dir


def _redirect_asset(file, model_dir: Path):
    """Map a bare asset name onto model_dir, leaving URLs and paths alone."""
    value = str(file).strip().replace("'", "")
    if value.startswith(("http:/", "https:/")):
        return file
    requested = Path(value)
    if requested.is_absolute() or requested.parent != Path("."):
        return file
    return model_dir / requested.name


def _find_legacy_model(name: str, destination: Path) -> Path | None:
    for directory in LEGACY_MODEL_DIRS:
        path = directory / name
        if path.is_file() and path != destination:
            return path
    return None
