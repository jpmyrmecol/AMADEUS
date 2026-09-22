# SPDX-License-Identifier: AGPL-3.0-only

"""Google Colab input bundling and final-artifact selection helpers.

This module deliberately contains no GUI code.  The functions that describe a
bundle or an export allowlist can therefore be tested without starting
Tkinter, Google Drive, or the AMADEUS batch process.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
from urllib.parse import urlencode
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping

import yaml

try:
    from .path_utils import CONFIG_PATH_KEYS, resolve_config_paths
    from .segmentation_metadata import read_segmentation_metadata, segmentation_paths_for_session
except ImportError:  # Direct execution/import with main on sys.path.
    from path_utils import CONFIG_PATH_KEYS, resolve_config_paths
    from segmentation_metadata import read_segmentation_metadata, segmentation_paths_for_session


GOOGLE_COLAB_HOME_URL = "https://colab.research.google.com/"
GOOGLE_DRIVE_SEARCH_URL = "https://drive.google.com/drive/u/0/search"
# The source-controlled notebook is a deterministic template.  A prepared
# project receives a personalized launcher at RUNTIME_NOTEBOOK_RELATIVE_PATH;
# the launcher is deliberately not an immutable manifest member.
RUNTIME_NOTEBOOK_RELATIVE_PATH = "colab/AMADEUS_Colab.ipynb"
RUNTIME_NOTEBOOK_TEMPLATE_RELATIVE_PATH = "colab/AMADEUS_Colab.template.ipynb"
# Kept as a compatibility name for callers that imported the old constant.
# A local Drive path cannot safely be turned into a Colab /drive/<file-id>
# URL, so callers should use build_colab_browser_target() instead.
GOOGLE_COLAB_NOTEBOOK_URL = GOOGLE_COLAB_HOME_URL
COLAB_DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_MARKERS = ("マイドライブ", "my drive", "mydrive")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
MIN_FREE_BYTES = 5 * 1024**3
WARN_FREE_BYTES = 10 * 1024**3

RUNTIME_PACKAGE_DIRECTORY = "amadeus_runtime"
RUNTIME_PACKAGE_MANIFEST = "amadeus_runtime_manifest.json"
_RUNTIME_PACKAGE_DIRECTORIES = (
    "assets",
    "gui",
    "main",
    "main/without_direction_estimation",
    "tools",
)
_RUNTIME_PACKAGE_ASSETS = (
    "assets/deep_green.json",
    "assets/logo_amadeus_mini.png",
    "assets/logo_amadeus_splash.png",
)
_RUNTIME_REQUIRED_FILES = (
    "pyproject.toml",
    "uv.lock",
    # The pinned uv version travels with the lockfile it resolves.
    "UV_VERSION",
    "VERSION",
    "README.md",
    RUNTIME_NOTEBOOK_TEMPLATE_RELATIVE_PATH,
)
_RUNTIME_LEGAL_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
)
_RUNTIME_BETA_LEGAL_FILE = "docs/legal/BETA_TEST_LICENSE.txt"
_YOLO_IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")
_UNSAFE_INPUT_DIR_NAMES = {".git", ".venv", "__pycache__"}


def _split_path(value: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", str(value or "").strip()) if part and part != "."]


def _drive_marker_index(parts: Iterable[str]) -> int:
    markers = {marker.casefold() for marker in DRIVE_MARKERS}
    for index, part in enumerate(parts):
        if part.casefold() in markers:
            return index
    return -1


def drive_path_to_colab_path(path: str) -> str:
    """Convert a Google Drive Desktop path to a mounted Colab path.

    The drive letter is intentionally ignored.  A path is accepted only when
    it contains an explicit ``My Drive``/``マイドライブ`` marker.
    """

    raw = str(path or "").strip()
    if not raw:
        raise ValueError("Google Drive path is empty.")
    parts = _split_path(raw)
    marker_index = _drive_marker_index(parts)
    if marker_index < 0:
        if raw.replace("\\", "/").startswith(COLAB_DRIVE_ROOT):
            return raw.replace("\\", "/")
        raise ValueError(
            "The selected path is not inside Google Drive. "
            "Choose a folder below My Drive (マイドライブ)."
        )
    suffix = parts[marker_index + 1 :]
    suffix_text = "/".join(suffix)
    return f"{COLAB_DRIVE_ROOT}/{suffix_text}" if suffix_text else COLAB_DRIVE_ROOT


local_drive_path_to_colab_path = drive_path_to_colab_path


def build_colab_browser_target(notebook_path: str) -> str:
    """Build a semantic Google Drive target for the prepared Colab notebook.

    Google Colab's direct ``/drive/<file-id>`` URL requires a Drive file ID;
    a Google Drive for desktop path does not contain one.  Never invent an ID
    from a local path.  Instead, open Drive's authenticated search view with
    the complete relative path to the prepared notebook as its query.  This
    is the narrowest credential-free target available without a Drive API
    lookup or a new browser-automation dependency.
    """

    raw = str(notebook_path or "").strip()
    if not raw:
        raise ValueError("Prepared notebook path is empty.")
    parts = _split_path(raw)
    marker_index = _drive_marker_index(parts)
    if marker_index < 0:
        raise ValueError(
            "The prepared notebook path is not inside Google Drive. "
            "Choose a folder below My Drive (マイドライブ)."
        )
    relative = "/".join(parts[marker_index + 1 :])
    if not relative or not relative.casefold().endswith(".ipynb"):
        raise ValueError("The prepared notebook path must name an .ipynb file.")
    if PurePosixPath(relative).name.casefold() != PurePosixPath(RUNTIME_NOTEBOOK_RELATIVE_PATH).name.casefold():
        raise ValueError("The prepared notebook path must name AMADEUS_Colab.ipynb.")
    return f"{GOOGLE_DRIVE_SEARCH_URL}?{urlencode({'q': relative})}"


colab_browser_target = build_colab_browser_target


def _normcase_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _same_path(left: str | Path, right: str | Path) -> bool:
    return _normcase_path(left) == _normcase_path(right)


def _is_within(path: str | Path, parent: str | Path) -> bool:
    try:
        return os.path.commonpath([_normcase_path(path), _normcase_path(parent)]) == _normcase_path(parent)
    except ValueError:
        return False


def _relative_posix(path: str, base: str) -> str:
    result = posixpath.relpath(path.replace("\\", "/"), base.replace("\\", "/"))
    return "." if result == "." else result


def _relative_colab(path: str, base: str) -> str:
    """Return a relative path when both values are mounted Colab POSIX paths."""

    result = posixpath.relpath(str(path).replace("\\", "/"), str(base).replace("\\", "/"))
    return "." if result == "." else result


def _file_fingerprint(path: str | Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return os.path.getsize(path), digest.hexdigest()


def _copy_file_no_overwrite(source: str, destination: str, *, overwrite: bool = False) -> None:
    source_abs = os.path.abspath(source)
    destination_abs = os.path.abspath(destination)
    if _same_path(source_abs, destination_abs):
        return
    if os.path.exists(destination_abs):
        if os.path.isfile(destination_abs) and os.path.isfile(source_abs):
            if _file_fingerprint(source_abs) == _file_fingerprint(destination_abs):
                return
        if not overwrite:
            raise FileExistsError(
                f"Destination already contains a different file: {destination_abs}. "
                "Choose another Drive folder."
            )
    os.makedirs(os.path.dirname(destination_abs), exist_ok=True)
    temp_path = destination_abs + f".part.{os.getpid()}"
    try:
        with open(source_abs, "rb") as source_stream, open(temp_path, "wb") as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream, length=1024 * 1024)
        shutil.copystat(source_abs, temp_path, follow_symlinks=True)
        os.replace(temp_path, destination_abs)
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class ConfigSummary:
    config_path: str
    session_path: str
    training_video_path: str
    tracking_video_path: str
    segmentation_pickle_path: str
    background_path: str
    segmentation_ready: bool
    easy_tracking_valid: bool
    errors: tuple[str, ...] = ()


@dataclass
class ColabPreparationState:
    """Track whether the current inputs have a usable prepared bundle."""

    config_path: str = ""
    destination: str = ""
    generation: int = 0
    running: bool = False
    active_generation: int | None = None
    prepared_generation: int | None = None

    def set_inputs(self, config_path: str, destination: str) -> bool:
        """Update inputs and invalidate any preparation for older inputs."""

        normalized = (str(config_path or "").strip(), str(destination or "").strip())
        if normalized == (self.config_path, self.destination):
            return False
        self.config_path, self.destination = normalized
        self.generation += 1
        self.prepared_generation = None
        return True

    @property
    def has_inputs(self) -> bool:
        return bool(self.config_path and self.destination)

    @property
    def is_prepared(self) -> bool:
        return self.prepared_generation == self.generation

    @property
    def prepare_enabled(self) -> bool:
        return self.has_inputs and not self.running and not self.is_prepared

    @property
    def open_enabled(self) -> bool:
        return self.is_prepared and not self.running

    def begin(self) -> int | None:
        """Start preparation for the current inputs and return its token."""

        if not self.prepare_enabled:
            return None
        self.running = True
        self.active_generation = self.generation
        return self.active_generation

    def is_current(self, token: int) -> bool:
        """Return whether a worker token still matches the current inputs."""

        return self.active_generation == token and self.generation == token

    def finish(self, token: int, *, successful: bool) -> bool:
        """Finish a worker and report whether it prepared current inputs."""

        if self.active_generation != token:
            return False
        self.running = False
        self.active_generation = None
        current = token == self.generation
        self.prepared_generation = token if successful and current else None
        return successful and current


@dataclass(frozen=True)
class BundleCopy:
    source: str
    destination: str
    size_bytes: int
    reason: str


@dataclass(frozen=True)
class BundlePlan:
    source_config: str
    destination_parent: str
    source_session: str
    destination_session: str
    destination_config: str
    colab_config_path: str
    copies: tuple[BundleCopy, ...]
    runtime_config: Mapping[str, object]

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.copies)


@dataclass(frozen=True)
class RuntimePackageCopy:
    """One allowlisted source file in the revision-bound runtime package."""

    source: str
    destination: str
    relative_path: str
    size_bytes: int
    sha256: str
    reason: str


@dataclass(frozen=True)
class RuntimePackagePlan:
    """A deterministic package of the local AMADEUS runtime for Drive."""

    source_root: str
    destination_session: str
    destination_root: str
    manifest_path: str
    notebook_path: str
    colab_notebook_path: str
    source_ref: str
    source_revision: str
    version: str
    lock_sha256: str
    template_sha256: str
    package_manifest_sha256: str
    manifest: Mapping[str, object]
    manifest_content: str
    launcher_content: str
    copies: tuple[RuntimePackageCopy, ...]

    @property
    def total_bytes(self) -> int:
        return (
            sum(item.size_bytes for item in self.copies)
            + len(self.launcher_content.encode("utf-8"))
            + len(self.manifest_content.encode("utf-8"))
        )


def _load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping.")
    return cfg


def load_colab_config(config_path: str) -> tuple[dict, ConfigSummary]:
    """Load a config and return a read-only summary suitable for the GUI."""

    config_abs = os.path.abspath(os.fspath(config_path))
    cfg = _load_yaml_config(config_abs)
    errors: list[str] = []
    session_value = str(cfg.get("SESSION_PATH", "") or "").strip()
    if not session_value:
        errors.append("SESSION_PATH is missing.")
        session_path = ""
    else:
        session_path = os.path.abspath(session_value)
        if not os.path.isdir(session_path):
            errors.append(f"SESSION_PATH does not exist: {session_path}")

    try:
        resolved = resolve_config_paths(dict(cfg))
    except Exception as exc:
        errors.append(f"Could not resolve config paths: {exc}")
        resolved = dict(cfg)

    training_video = str(resolved.get("TRAINING_VIDEO_PATH", "") or "").strip()
    tracking_video = str(resolved.get("TRACKING_VIDEO_PATH", "") or "").strip()
    if not training_video:
        errors.append("TRAINING_VIDEO_PATH is missing.")
    if not tracking_video:
        errors.append("TRACKING_VIDEO_PATH is missing.")

    pickle_path, background_path = segmentation_paths_for_session(session_path, training_video)
    pickle_path = str(resolved.get("PICKLE_PATH", pickle_path) or pickle_path)
    background_path = str(resolved.get("BACKGROUND_PATH", background_path) or background_path)
    resolved.setdefault("PICKLE_PATH", pickle_path)
    resolved.setdefault("BACKGROUND_PATH", background_path)
    segmentation_ready = bool(
        pickle_path and os.path.isfile(pickle_path) and background_path and os.path.isfile(background_path)
    )
    if not segmentation_ready:
        errors.append("Segmentation output is incomplete. Run Segmentation processing first.")
    else:
        try:
            read_segmentation_metadata(pickle_path)
        except Exception as exc:
            errors.append(f"Segmentation metadata is invalid: {exc}")
            segmentation_ready = False

    variable_population = bool(resolved.get("VARIABLE_NUM_OBJECTS", False))
    try:
        number_of_animals = int(resolved.get("NUM_OBJECTS", 0))
    except (TypeError, ValueError):
        number_of_animals = 0
    easy_tracking_valid = variable_population or number_of_animals > 0
    if not easy_tracking_valid:
        errors.append("NUM_OBJECTS must be a positive integer, unless VARIABLE_NUM_OBJECTS is enabled.")
    if not isinstance(resolved.get("training", {}) or {}, dict):
        errors.append("training must be a mapping.")
        easy_tracking_valid = False

    summary = ConfigSummary(
        config_path=config_abs,
        session_path=session_path,
        training_video_path=training_video,
        tracking_video_path=tracking_video,
        segmentation_pickle_path=pickle_path,
        background_path=background_path,
        segmentation_ready=segmentation_ready,
        easy_tracking_valid=easy_tracking_valid,
        errors=tuple(dict.fromkeys(errors)),
    )
    return resolved, summary


def validate_video(path: str) -> None:
    """Validate one video or the top-level files in a video directory."""

    candidates = [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if name.lower().endswith(VIDEO_EXTENSIONS)
    ] if os.path.isdir(path) else [path]
    candidates = [candidate for candidate in candidates if os.path.isfile(candidate)]
    if not candidates:
        raise ValueError(f"No readable video file was found at: {path}")
    import cv2

    for candidate in candidates:
        capture = cv2.VideoCapture(candidate)
        try:
            if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) <= 0:
                raise ValueError(f"Video could not be opened or has no frames: {candidate}")
        finally:
            capture.release()


def validate_colab_config(config_path: str, *, check_video: bool = True) -> tuple[dict, ConfigSummary]:
    cfg, summary = load_colab_config(config_path)
    errors = list(summary.errors)
    if not os.path.isfile(summary.config_path):
        errors.append(f"Config file does not exist: {summary.config_path}")
    if check_video:
        for label, path in (
            ("TRAINING_VIDEO_PATH", summary.training_video_path),
            ("TRACKING_VIDEO_PATH", summary.tracking_video_path),
        ):
            try:
                validate_video(path)
            except Exception as exc:
                errors.append(f"{label} is invalid: {exc}")
    if errors:
        raise ValueError("\n".join(dict.fromkeys(errors)))
    return cfg, summary


def _video_files(path: str) -> list[str]:
    if os.path.isdir(path):
        return [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if name.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(path, name))
        ]
    return [path] if os.path.isfile(path) else []


def _safe_destination(parent: str, relative: str) -> str:
    destination = os.path.abspath(os.path.join(parent, relative))
    if not _is_within(destination, parent):
        raise ValueError(f"Unsafe bundle destination: {relative}")
    return destination


def _add_copy(copies: list[BundleCopy], seen: dict[str, str], source: str, destination: str, reason: str) -> None:
    source = os.path.abspath(source)
    destination = os.path.abspath(destination)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Required input file not found: {source}")
    existing = seen.get(_normcase_path(source))
    if existing is not None:
        if not _same_path(existing, destination):
            raise ValueError(f"The same input was assigned two destinations: {source}")
        return
    for item in copies:
        if _same_path(item.destination, destination) and not _same_path(item.source, source):
            raise ValueError(
                f"Different inputs were assigned the same destination: {item.source} and {source}"
            )
    seen[_normcase_path(source)] = destination
    copies.append(BundleCopy(source, destination, os.path.getsize(source), reason))


def _destination_session_path(source_session: str, destination_parent: str) -> str:
    candidate = _safe_destination(destination_parent, os.path.basename(os.path.normpath(source_session)))
    if Path(candidate).is_symlink():
        raise FileExistsError(f"Destination project is a symlink: {candidate}")
    if os.path.exists(candidate) and not os.path.isdir(candidate):
        raise FileExistsError(f"Destination project is not a directory: {candidate}")
    if os.path.isdir(candidate) and any(Path(candidate).iterdir()) and not _same_path(candidate, source_session):
        raise FileExistsError(
            f"Destination project is not empty: {candidate}. Choose a new Drive folder or project name."
        )
    return candidate


def _posix_runtime_config(
    cfg: dict,
    *,
    source_session: str,
    destination_session: str,
    colab_destination_session: str,
    path_map: Mapping[str, str],
) -> dict:
    runtime = dict(cfg)
    runtime["SESSION_PATH"] = colab_destination_session
    for key in CONFIG_PATH_KEYS:
        value = runtime.get(key)
        if isinstance(value, str) and value.strip():
            source_value = os.path.abspath(value)
            mapped = path_map.get(_normcase_path(source_value), source_value)
            mapped_colab = path_map.get(_normcase_path(mapped), mapped)
            # ``path_map`` stores local destination paths.  The caller adds a
            # parallel _COLAB_PATH_MAP entry for the serialized config.
            colab_map = path_map.get(f"__colab__:{_normcase_path(source_value)}")
            runtime[key] = (
                _relative_colab(colab_map, colab_destination_session)
                if colab_map
                else _relative_posix(mapped_colab, destination_session)
            )
    source_dirs = runtime.get("CREATE_DATASET_SOURCE_DIRS")
    if source_dirs:
        values = [source_dirs] if isinstance(source_dirs, str) else list(source_dirs)
        rewritten: list[str] = []
        for value in values:
            source_value = os.path.abspath(str(value))
            colab_map = path_map.get(f"__colab__:{_normcase_path(source_value)}")
            rewritten.append(
                _relative_colab(colab_map, colab_destination_session)
                if colab_map
                else _relative_posix(source_value, destination_session)
            )
        runtime["CREATE_DATASET_SOURCE_DIRS"] = rewritten if not isinstance(source_dirs, str) else rewritten[0]
    return runtime


def _is_credential_like_name(name: str) -> bool:
    """Return whether a file or directory name looks like secret material."""

    lower = str(name).casefold()
    return (
        lower in {".env", ".npmrc", ".pypirc", "credentials", "secrets", "tokens", "passwords"}
        or lower.startswith(("credentials", "oauth", "service-account", "id_rsa"))
        or any(token in lower for token in ("secret", "token", "password"))
        or lower.endswith((".pem", ".key", ".p12", ".pfx"))
    )


def _scan_referenced_tree(source: Path) -> None:
    """Reject links and credential-like names before selecting input files.

    The scan deliberately does not copy anything.  The callers below use a
    narrow, known-consumer allowlist after this safety pass, so ignored files
    such as Finder metadata and old intermediate output never enter the plan.
    """

    if source.is_symlink():
        raise ValueError(f"Symlinked input directories are not allowed: {source}")
    for current, dir_names, file_names in os.walk(source, topdown=True, followlinks=False):
        current_path = Path(current)
        dir_names.sort()
        file_names.sort()
        for name in (*dir_names, *file_names):
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"Symlinked input files are not allowed: {child}")
            if _is_credential_like_name(name):
                raise ValueError(f"Credential-like input path is not allowed: {child}")


def _validated_image_extension(value: object) -> str:
    extension = str(value or ".png").strip().casefold()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
        raise ValueError(f"Unsupported dataset image extension: {value!r}")
    return extension


def _is_visible_regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and not path.name.startswith(".")


def _paired_dataset_files(source: Path, image_extension: str) -> list[Path]:
    """Return only image/label pairs consumed by ``create_dataset.py``."""

    pairs: list[Path] = []
    for image_dir, label_dir in (
        (source / "images", source / "labels"),
        (source / "cropping" / "images", source / "cropping" / "labels"),
    ):
        if not image_dir.is_dir() or image_dir.is_symlink() or not label_dir.is_dir() or label_dir.is_symlink():
            continue
        for image in sorted(image_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not _is_visible_regular_file(image) or image.suffix.casefold() != image_extension:
                continue
            label = label_dir / f"{image.stem}.txt"
            if _is_visible_regular_file(label):
                pairs.extend((image.relative_to(source), label.relative_to(source)))
    return pairs


def _filtered_dataset_files(source: Path, *, image_extension: str, yolo: bool = False) -> list[Path]:
    """Select the narrow external-input shapes used by the AMADEUS stages."""

    _scan_referenced_tree(source)
    if yolo:
        selected: list[Path] = []
        data_yaml = source / "data.yaml"
        if _is_visible_regular_file(data_yaml):
            selected.append(data_yaml.relative_to(source))
        for split in ("train", "test"):
            image_dir = source / split / "images"
            label_dir = source / split / "labels"
            if not image_dir.is_dir() or image_dir.is_symlink() or not label_dir.is_dir() or label_dir.is_symlink():
                continue
            for image in sorted(image_dir.iterdir(), key=lambda item: item.name.casefold()):
                if not _is_visible_regular_file(image) or image.suffix.casefold() not in _YOLO_IMAGE_EXTENSIONS:
                    continue
                label = label_dir / f"{image.stem}.txt"
                if _is_visible_regular_file(label):
                    selected.extend((image.relative_to(source), label.relative_to(source)))
        return selected
    return _paired_dataset_files(source, image_extension)


def plan_input_bundle(config_path: str, destination_parent: str) -> BundlePlan:
    """Plan a minimal, collision-safe Drive input bundle.

    Only the saved config, the current segmentation inputs, referenced videos,
    and explicitly referenced external input paths are included.  Existing
    unrelated files in the source project are never traversed.
    """

    cfg, summary = validate_colab_config(config_path, check_video=False)
    destination_parent = os.path.abspath(os.fspath(destination_parent))
    drive_path_to_colab_path(destination_parent)
    if not os.path.isdir(destination_parent):
        raise ValueError(f"Google Drive destination does not exist: {destination_parent}")
    source_session = os.path.abspath(summary.session_path)
    if not source_session:
        raise ValueError("SESSION_PATH is required before preparing a Colab bundle.")
    destination_session = _destination_session_path(source_session, destination_parent)
    same_destination = _same_path(destination_session, source_session)
    config_name = "config_colab.yaml" if same_destination else "config.yaml"
    destination_config = _safe_destination(destination_session, config_name)
    if os.path.exists(destination_config) and not _same_path(destination_config, summary.config_path):
        raise FileExistsError(
            f"Destination config already exists: {destination_config}. Choose another Drive folder."
        )
    colab_destination_session = drive_path_to_colab_path(destination_session)
    colab_destination_config = f"{colab_destination_session}/{config_name}"

    copies: list[BundleCopy] = []
    seen: dict[str, str] = {}
    path_map: dict[str, str] = {}
    colab_map: dict[str, str] = {}

    def add_file(source: str, destination: str, reason: str) -> None:
        source_abs = os.path.abspath(source)
        destination_abs = os.path.abspath(destination)
        if Path(source_abs).is_symlink():
            raise ValueError(f"Symlinked input files are not allowed: {source_abs}")
        _add_copy(copies, seen, source_abs, destination_abs, reason)
        path_map[_normcase_path(source_abs)] = destination_abs
        colab_map[_normcase_path(source_abs)] = drive_path_to_colab_path(destination_abs)

    add_file(summary.config_path, destination_config, "config.yaml")

    seg_dir = os.path.join(source_session, "segmentation")
    required_segmentation = (
        summary.segmentation_pickle_path,
        summary.background_path,
        os.path.join(seg_dir, "segmentation_gui_config.json"),
    )
    for source in required_segmentation:
        if os.path.isfile(source):
            add_file(source, _safe_destination(destination_session, os.path.join("segmentation", os.path.basename(source))), "segmentation")
    if not os.path.isfile(summary.segmentation_pickle_path) or not os.path.isfile(summary.background_path):
        raise ValueError("Segmentation output is incomplete. Run Segmentation processing first.")

    video_sources: list[str] = []
    for value in (summary.training_video_path, summary.tracking_video_path):
        if os.path.isdir(value):
            video_sources.extend(_video_files(value))
        elif value:
            video_sources.append(value)
    unique_videos: list[str] = []
    for source in video_sources:
        if _normcase_path(source) not in {_normcase_path(item) for item in unique_videos}:
            unique_videos.append(source)
    for source in unique_videos:
        source_abs = os.path.abspath(source)
        if _is_within(source_abs, source_session):
            relative = os.path.relpath(source_abs, source_session)
            destination = _safe_destination(destination_session, relative)
        else:
            destination = _safe_destination(destination_parent, os.path.basename(source_abs))
        add_file(source_abs, destination, "training/tracking video")
    for value in (summary.training_video_path, summary.tracking_video_path):
        if os.path.isdir(value):
            source_dir = os.path.abspath(value)
            if _is_within(source_dir, source_session):
                relative = os.path.relpath(source_dir, source_session)
                destination_dir = _safe_destination(destination_session, relative)
            else:
                destination_dir = _safe_destination(destination_parent, os.path.basename(os.path.normpath(source_dir)))
            path_map[_normcase_path(source_dir)] = destination_dir
            colab_map[_normcase_path(source_dir)] = drive_path_to_colab_path(destination_dir)

    # A skipped stage may depend on an existing CSV or a user-provided dataset.
    # Such paths are copied explicitly.  The source project itself is never
    # recursively copied.
    referenced_values: list[tuple[str, object]] = [
        (key, cfg.get(key)) for key in CONFIG_PATH_KEYS if key not in {"TRAINING_VIDEO_PATH", "TRACKING_VIDEO_PATH"}
    ]
    raw_dirs = cfg.get("CREATE_DATASET_SOURCE_DIRS")
    if raw_dirs:
        referenced_values.append(("CREATE_DATASET_SOURCE_DIRS", raw_dirs))
    for key, raw_value in referenced_values:
        values = [raw_value] if isinstance(raw_value, str) else list(raw_value or [])
        for raw in values:
            source = os.path.abspath(str(raw))
            if not os.path.exists(source):
                if key == "INIT_CSV_PATH" and not bool(cfg.get("skip_initial_tracking", False)):
                    continue
                raise FileNotFoundError(f"Config references a missing input: {key}={source}")
            if _normcase_path(source) in seen:
                continue
            if key in {"CREATE_DATASET_SOURCE_DIRS", "YOLO_DATASET_DIR"}:
                source_path = Path(source)
                if not source_path.is_dir():
                    raise ValueError(f"Config path must be a directory: {key}={source}")
                image_extension = _validated_image_extension(cfg.get("IMG_EXT", ".png"))
                selected = _filtered_dataset_files(
                    source_path,
                    image_extension=image_extension,
                    yolo=key == "YOLO_DATASET_DIR",
                )
                if _is_within(source, source_session):
                    directory_destination = _safe_destination(
                        destination_session,
                        os.path.relpath(source, source_session),
                    )
                else:
                    directory_destination = _safe_destination(
                        destination_session,
                        os.path.join("inputs", os.path.basename(os.path.normpath(source))),
                    )
                for relative in selected:
                    add_file(
                        str(source_path / relative),
                        _safe_destination(directory_destination, relative.as_posix()),
                        f"filtered config directory {key}",
                    )
                path_map[_normcase_path(source)] = directory_destination
                colab_map[_normcase_path(source)] = drive_path_to_colab_path(directory_destination)
                continue
            if key in {"PICKLE_PATH", "BACKGROUND_PATH", "INIT_CSV_PATH"} and os.path.isdir(source):
                raise ValueError(f"Config path must be a file: {key}={source}")
            if os.path.isfile(source):
                if _is_within(source, source_session):
                    destination = _safe_destination(destination_session, os.path.relpath(source, source_session))
                else:
                    destination = _safe_destination(destination_session, os.path.join("inputs", os.path.basename(source)))
                add_file(source, destination, f"config path {key}")
            elif os.path.isdir(source):
                raise ValueError(f"Unsupported config directory: {key}={source}")

    combined_map = dict(path_map)
    combined_map.update({f"__colab__:{key}": value for key, value in colab_map.items()})
    for key in CONFIG_PATH_KEYS:
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            source_value = os.path.abspath(value)
            if _normcase_path(source_value) in colab_map:
                continue
            # Missing INIT_CSV_PATH is valid when initial tracking will create it.
            if not os.path.exists(source_value) and key == "INIT_CSV_PATH" and not bool(cfg.get("skip_initial_tracking", False)):
                continue
            if os.path.exists(source_value):
                raise ValueError(f"Could not map config path {key}: {source_value}")
    runtime = _posix_runtime_config(
        cfg,
        source_session=source_session,
        destination_session=destination_session,
        colab_destination_session=colab_destination_session,
        path_map=combined_map,
    )
    # The serialized bundle config uses Colab paths for SESSION_PATH and path
    # keys, while the source config remains byte-for-byte untouched.
    runtime["SESSION_PATH"] = colab_destination_session
    for key in CONFIG_PATH_KEYS:
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            source_value = os.path.abspath(value)
            if source_value and _normcase_path(source_value) in colab_map:
                runtime[key] = _relative_colab(colab_map[_normcase_path(source_value)], colab_destination_session)
            elif key == "INIT_CSV_PATH" and not os.path.exists(source_value) and not bool(cfg.get("skip_initial_tracking", False)):
                runtime[key] = posixpath.join("initial_tracking", os.path.basename(source_value))
    if isinstance(cfg.get("CREATE_DATASET_SOURCE_DIRS"), str):
        raw = str(cfg["CREATE_DATASET_SOURCE_DIRS"])
        mapped = colab_map.get(_normcase_path(os.path.abspath(raw)))
        if mapped:
            runtime["CREATE_DATASET_SOURCE_DIRS"] = _relative_colab(mapped, colab_destination_session)
    elif cfg.get("CREATE_DATASET_SOURCE_DIRS"):
        runtime["CREATE_DATASET_SOURCE_DIRS"] = [
            _relative_colab(
                colab_map.get(_normcase_path(os.path.abspath(str(value))), str(value)),
                colab_destination_session,
            )
            for value in cfg["CREATE_DATASET_SOURCE_DIRS"]
        ]

    return BundlePlan(
        source_config=summary.config_path,
        destination_parent=destination_parent,
        source_session=source_session,
        destination_session=destination_session,
        destination_config=destination_config,
        colab_config_path=colab_destination_config,
        copies=tuple(copies),
        runtime_config=runtime,
    )


def execute_bundle(
    plan: BundlePlan,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    """Copy a planned bundle and write the Colab-compatible config atomically."""

    if os.path.exists(plan.destination_config) and not _same_path(plan.destination_config, plan.source_config):
        raise FileExistsError(
            f"Destination config already exists: {plan.destination_config}. Choose another Drive folder."
        )
    total = plan.total_bytes
    copied = 0
    for item in plan.copies:
        _copy_file_no_overwrite(item.source, item.destination)
        copied += item.size_bytes
        if progress_callback:
            progress_callback(copied, total, item.reason)
    os.makedirs(os.path.dirname(plan.destination_config), exist_ok=True)
    temp_config = plan.destination_config + f".part.{os.getpid()}"
    try:
        with open(temp_config, "w", encoding="utf-8") as stream:
            yaml.safe_dump(dict(plan.runtime_config), stream, sort_keys=False, allow_unicode=True)
        os.replace(temp_config, plan.destination_config)
    finally:
        try:
            os.remove(temp_config)
        except FileNotFoundError:
            pass


def _git_value(project_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _local_provenance(
    project_root: Path,
    *,
    source_ref: str | None,
    source_revision: str | None,
) -> tuple[str, str]:
    revision = str(source_revision or "").strip() or _git_value(project_root, "rev-parse", "HEAD")
    ref = str(source_ref or "").strip() or _git_value(project_root, "symbolic-ref", "--short", "-q", "HEAD")
    if not ref:
        ref = _git_value(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    # A downloaded/source-only installation has no .git metadata.  Keep the
    # package usable in that case while making the limitation explicit in the
    # manifest; the content hashes still identify the exact files packaged.
    return ref or "local-package", revision or "unavailable-local-revision"


def _runtime_source_specs(project_root: Path, *, source_ref: str = "") -> list[tuple[str, Path, str]]:
    """Build the fixed source allowlist for the headless Colab runtime."""

    specs: dict[str, tuple[Path, str]] = {}

    def add(relative: str, reason: str, *, source_relative: str | None = None) -> None:
        normalized = PurePosixPath(relative).as_posix()
        if normalized in specs:
            return
        source_name = source_relative or normalized
        source = project_root.joinpath(*source_name.split("/"))
        if source.is_symlink():
            raise ValueError(f"Symlinked runtime package input is not allowed: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"Required runtime package file not found: {source}")
        specs[normalized] = (source, reason)

    for relative_dir in _RUNTIME_PACKAGE_DIRECTORIES:
        directory = project_root.joinpath(*relative_dir.split("/"))
        if directory.is_symlink():
            raise ValueError(f"Symlinked runtime package directory is not allowed: {directory}")
        if not directory.is_dir():
            raise FileNotFoundError(f"Required runtime package directory not found: {directory}")
        for current, dir_names, file_names in os.walk(directory, topdown=True, followlinks=False):
            current_path = Path(current)
            dir_names.sort()
            file_names.sort()
            kept_dirs: list[str] = []
            for name in dir_names:
                child = current_path / name
                if child.is_symlink():
                    raise ValueError(f"Symlinked runtime package directory is not allowed: {child}")
                if name.casefold() in _UNSAFE_INPUT_DIR_NAMES:
                    continue
                if _is_credential_like_name(name):
                    raise ValueError(f"Credential-like runtime package path is not allowed: {child}")
                kept_dirs.append(name)
            dir_names[:] = kept_dirs
            for name in file_names:
                child = current_path / name
                if child.is_symlink():
                    raise ValueError(f"Symlinked runtime package file is not allowed: {child}")
                if _is_credential_like_name(name):
                    raise ValueError(f"Credential-like runtime package path is not allowed: {child}")
                if child.suffix.casefold() != ".py":
                    continue
                relative = child.relative_to(project_root).as_posix()
                add(relative, "runtime Python source")

    for relative in _RUNTIME_PACKAGE_ASSETS:
        add(relative, "runtime asset")
    for relative in _RUNTIME_REQUIRED_FILES:
        add(
            relative,
            "immutable runtime metadata/bootstrap",
            source_relative=(
                RUNTIME_NOTEBOOK_RELATIVE_PATH
                if relative == RUNTIME_NOTEBOOK_TEMPLATE_RELATIVE_PATH
                else None
            ),
        )
    for relative in _RUNTIME_LEGAL_FILES:
        add(relative, "runtime legal notice")

    pyproject = project_root / "pyproject.toml"
    pyproject_text = pyproject.read_text(encoding="utf-8").casefold()
    branch_ref = source_ref.casefold() or _git_value(project_root, "symbolic-ref", "--short", "-q", "HEAD").casefold()
    if "closed-beta" in pyproject_text or branch_ref.endswith("beta-v5"):
        add(_RUNTIME_BETA_LEGAL_FILE, "closed-beta legal notice")
    return [(relative, source, reason) for relative, (source, reason) in sorted(specs.items())]


def _manifest_file_entries(copies: Iterable[RuntimePackageCopy]) -> list[dict[str, object]]:
    return [
        {
            "path": item.relative_path,
            "size": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in copies
    ]


def _canonical_manifest_hash(entries: Iterable[Mapping[str, object]]) -> str:
    payload = json.dumps(
        list(entries),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def personalize_colab_notebook(template: str | Path, config_path: str) -> str:
    """Return a deterministic project-specific Colab launcher.

    The repository notebook is read as trusted JSON and only its CONFIG_PATH
    parameter is changed.  JSON string encoding, rather than source-text
    interpolation, keeps spaces, Japanese characters, quotes, and backslashes
    safe in both the notebook document and the generated Python source.
    """

    if isinstance(template, Path) or (isinstance(template, str) and os.path.isfile(template)):
        template_text = Path(template).read_text(encoding="utf-8")
    else:
        template_text = str(template)
    value = str(config_path or "")
    if "\x00" in value or any(ord(character) < 32 for character in value):
        raise ValueError("Colab config path contains a control character.")
    normalized = value.replace("\\", "/")
    if value and not (
        normalized == COLAB_DRIVE_ROOT
        or normalized.startswith(COLAB_DRIVE_ROOT + "/")
    ):
        raise ValueError("Colab config path must be under /content/drive/MyDrive.")
    try:
        notebook = json.loads(template_text)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Colab notebook template is not valid JSON: {exc}") from exc
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise ValueError("Colab notebook template must contain a cells list.")

    parameter_line = f"CONFIG_PATH = {json.dumps(value, ensure_ascii=False)}  # @param {{type:\"string\"}}\n"
    replaced = False
    for cell in notebook["cells"]:
        if not isinstance(cell, dict) or not isinstance(cell.get("source"), list):
            continue
        source = cell["source"]
        for index, line in enumerate(source):
            if isinstance(line, str) and line.startswith("CONFIG_PATH = "):
                source[index] = parameter_line
                replaced = True
                break
        if replaced:
            break
    if not replaced:
        raise ValueError("Colab notebook template has no CONFIG_PATH parameter.")
    metadata = notebook.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Colab notebook template metadata must be an object.")
    # This is the notebook-level accelerator declaration used by Colab's
    # standard notebook metadata.  It requests a generic GPU; allocation is
    # still subject to the account/quota/runtime availability.
    metadata["accelerator"] = "GPU"
    return json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"


def _runtime_destination_compatibility(plan: RuntimePackagePlan) -> None:
    """Fail before copying if a Drive package would overwrite user data."""

    root = Path(plan.destination_root)
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise FileExistsError(f"Runtime package destination is not a directory: {root}")
    expected_files = {
        _normcase_path(item.destination): (item.source, item.size_bytes, item.sha256)
        for item in plan.copies
    }
    expected_files[_normcase_path(plan.manifest_path)] = (
        "__manifest__",
        len(plan.manifest_content.encode("utf-8")),
        hashlib.sha256(plan.manifest_content.encode("utf-8")).hexdigest(),
    )
    # The prepared notebook is project-specific and may be rewritten by
    # Colab/Drive.  It remains allowlisted, but its existing bytes are not a
    # collision signal and are never compared with the immutable manifest.
    expected_files[_normcase_path(plan.notebook_path)] = (
        "__mutable_launcher__",
        0,
        "",
    )
    allowed_dirs = {_normcase_path(root)}
    for destination in expected_files:
        destination_path = Path(destination)
        allowed_dirs.update(_normcase_path(parent) for parent in destination_path.parents if _is_within(parent, root))
    for child in root.rglob("*"):
        normalized = _normcase_path(child)
        if child.is_symlink():
            raise FileExistsError(f"Runtime package destination contains a symlink: {child}")
        if child.is_dir():
            if normalized not in allowed_dirs:
                raise FileExistsError(f"Runtime package destination contains unrelated data: {child}")
            continue
        expected = expected_files.get(normalized)
        if expected is None:
            raise FileExistsError(f"Runtime package destination contains unrelated data: {child}")
        if expected[0] == "__manifest__":
            actual_digest = hashlib.sha256(child.read_bytes()).hexdigest()
            actual_size = child.stat().st_size
        elif expected[0] == "__mutable_launcher__":
            continue
        else:
            actual_size, actual_digest = _file_fingerprint(child)
        if (actual_size, actual_digest) != (expected[1], expected[2]):
            raise FileExistsError(f"Runtime package destination already contains a different file: {child}")


def plan_runtime_package(
    project_root: str,
    destination_session: str,
    *,
    colab_config_path: str | None = None,
    source_ref: str | None = None,
    source_revision: str | None = None,
) -> RuntimePackagePlan:
    """Plan a revision-bound, allowlisted AMADEUS package for Google Drive.

    The package preserves repository-relative paths for the executable Python
    source, the three declared asset files, locked metadata, an immutable
    notebook template, the runner, and the applicable legal notices.  The
    project-specific launcher is generated separately and is not included in
    the immutable file-entry hash.  It never traverses a user session or
    copies repository management files.
    """

    root = Path(os.path.abspath(os.fspath(project_root)))
    if not root.is_dir():
        raise ValueError(f"AMADEUS project root does not exist: {root}")
    destination_session = os.path.abspath(os.fspath(destination_session))
    drive_path_to_colab_path(destination_session)
    destination_root = _safe_destination(destination_session, RUNTIME_PACKAGE_DIRECTORY)
    destination_manifest = _safe_destination(destination_root, RUNTIME_PACKAGE_MANIFEST)
    source_ref, source_revision = _local_provenance(
        root,
        source_ref=source_ref,
        source_revision=source_revision,
    )
    specs = _runtime_source_specs(root, source_ref=source_ref)
    copies: list[RuntimePackageCopy] = []
    for relative, source, reason in specs:
        size_bytes, digest = _file_fingerprint(source)
        destination = _safe_destination(destination_root, relative)
        copies.append(RuntimePackageCopy(str(source), destination, relative, size_bytes, digest, reason))

    entries = _manifest_file_entries(copies)
    package_manifest_sha256 = _canonical_manifest_hash(entries)
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"VERSION is empty: {root / 'VERSION'}")
    lock_sha256 = next(item.sha256 for item in copies if item.relative_path == "uv.lock")
    template_sha256 = next(
        item.sha256 for item in copies if item.relative_path == RUNTIME_NOTEBOOK_TEMPLATE_RELATIVE_PATH
    )
    launcher_content = personalize_colab_notebook(
        root / RUNTIME_NOTEBOOK_RELATIVE_PATH,
        str(colab_config_path or ""),
    )
    legal_files = [item["path"] for item in entries if item["path"] in _RUNTIME_LEGAL_FILES or item["path"] == _RUNTIME_BETA_LEGAL_FILE]
    manifest: dict[str, object] = {
        "schema_version": 2,
        "package_name": "amadeus-colab-runtime",
        "package_root": RUNTIME_PACKAGE_DIRECTORY,
        "source_ref": source_ref,
        "source_revision": source_revision,
        "version": version,
        "lock_sha256": lock_sha256,
        "template_sha256": template_sha256,
        "package_manifest_sha256": package_manifest_sha256,
        "legal_files": legal_files,
        "mutable_files": [
            {
                "path": RUNTIME_NOTEBOOK_RELATIVE_PATH,
                "kind": "project-specific-launcher",
                "template_path": RUNTIME_NOTEBOOK_TEMPLATE_RELATIVE_PATH,
            }
        ],
        "files": entries,
        "provenance": {
            "source_ref": source_ref,
            "source_revision": source_revision,
            "version": version,
            "lock_sha256": lock_sha256,
            "package_manifest_sha256": package_manifest_sha256,
            "template_sha256": template_sha256,
        },
        "excluded": [
            ".git",
            ".venv",
            "__pycache__",
            "session data",
            "transient outputs",
            "credentials and unrelated user data",
        ],
    }
    manifest_content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    plan = RuntimePackagePlan(
        source_root=str(root),
        destination_session=destination_session,
        destination_root=destination_root,
        manifest_path=destination_manifest,
        notebook_path=_safe_destination(destination_root, RUNTIME_NOTEBOOK_RELATIVE_PATH),
        colab_notebook_path=drive_path_to_colab_path(
            _safe_destination(destination_root, RUNTIME_NOTEBOOK_RELATIVE_PATH)
        ),
        source_ref=source_ref,
        source_revision=source_revision,
        version=version,
        lock_sha256=lock_sha256,
        template_sha256=template_sha256,
        package_manifest_sha256=package_manifest_sha256,
        manifest=manifest,
        manifest_content=manifest_content,
        launcher_content=launcher_content,
        copies=tuple(copies),
    )
    _runtime_destination_compatibility(plan)
    return plan


def execute_runtime_package(
    plan: RuntimePackagePlan,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    """Copy an allowlisted runtime package and atomically write its manifest."""

    _runtime_destination_compatibility(plan)
    os.makedirs(plan.destination_root, exist_ok=True)
    total = plan.total_bytes
    copied = 0
    for item in plan.copies:
        current_size, current_digest = _file_fingerprint(item.source)
        if (current_size, current_digest) != (item.size_bytes, item.sha256):
            raise RuntimeError(
                f"Runtime package source changed after planning: {item.source}. "
                "Prepare the package again."
            )
        _copy_file_no_overwrite(item.source, item.destination)
        copied += item.size_bytes
        if progress_callback:
            progress_callback(copied, total, item.reason)

    launcher_bytes = plan.launcher_content.encode("utf-8")
    launcher_path = Path(plan.notebook_path)
    if launcher_path.is_symlink():
        raise FileExistsError(f"Prepared notebook destination is a symlink: {launcher_path}")
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_temp = launcher_path.with_name(f"{launcher_path.name}.part.{os.getpid()}")
    try:
        launcher_temp.write_bytes(launcher_bytes)
        os.replace(launcher_temp, launcher_path)
    finally:
        try:
            launcher_temp.unlink()
        except FileNotFoundError:
            pass
    copied += len(launcher_bytes)
    if progress_callback:
        progress_callback(copied, total, "mutable Colab launcher")

    manifest_bytes = plan.manifest_content.encode("utf-8")
    if os.path.exists(plan.manifest_path):
        if Path(plan.manifest_path).read_bytes() != manifest_bytes:
            raise FileExistsError(
                f"Runtime package manifest already contains different data: {plan.manifest_path}"
            )
    else:
        temp_path = plan.manifest_path + f".part.{os.getpid()}"
        try:
            with open(temp_path, "wb") as stream:
                stream.write(manifest_bytes)
            os.replace(temp_path, plan.manifest_path)
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
    if progress_callback:
        progress_callback(total, total, "runtime package manifest")


# Explicit aliases make the package API easy to discover from callers that
# describe the operation as a Colab runtime package rather than a bundle.
plan_colab_runtime_package = plan_runtime_package
execute_colab_runtime_package = execute_runtime_package


_TEMP_DIR_NAMES = {
    "single_animal_images",
    "paste_blobs",
    "paste_blobs_clustered",
    "clustered",
    "yolo_dataset",
    "cropping",
    "preview",
    "images",
    "images_jpg",
    "yolo_raw",
    "cache",
    "buffer",
    "log",
    "__pycache__",
}
_FINAL_TRACKING_PREFIXES = (
    "obbs",
    "directions",
)


def _is_final_artifact(relative: Path, *, export_images: bool = False, export_raw: bool = False) -> bool:
    parts = relative.parts
    logical_parts = parts[1:] if parts and parts[0].casefold() == "without_direction_estimation" else parts
    logical = Path(*logical_parts) if logical_parts else Path(relative.name)
    name = logical.name
    lower_parts = {part.casefold() for part in logical.parts}
    if any(part in _TEMP_DIR_NAMES for part in lower_parts):
        if export_images and ("images" in lower_parts or "images_jpg" in lower_parts):
            pass
        else:
            return False
    if name in {"log.txt", "time.csv"} and len(logical.parts) == 1:
        return True
    if logical.parts and logical.parts[0] == "logs":
        return True
    if logical.parts and logical.parts[0] == "results" and name.lower().endswith(".csv"):
        return True
    if logical.parts and logical.parts[0] == "model" and name.lower().endswith(".pt"):
        return True
    if logical.parts and logical.parts[0] == "main":
        if name in {"best.pt", "last.pt", "results.csv", "args.yaml"}:
            return True
        if len(logical.parts) >= 2 and "training" in lower_parts and name.lower().endswith(".csv") and name == "results.csv":
            return True
        if "tracking" in lower_parts:
            if name.lower().endswith((".csv", ".json")) and name.startswith(_FINAL_TRACKING_PREFIXES):
                return True
            if name.lower().endswith(VIDEO_EXTENSIONS) and (export_raw or not name.lower().startswith("yolo_raw")):
                return True
            if export_images and ("images" in lower_parts or "images_jpg" in lower_parts) and name.lower().endswith((".png", ".jpg", ".jpeg")):
                return True
    if name.endswith("_error.txt") or "traceback" in name.casefold():
        return True
    return False


def select_final_artifacts(
    local_session: str,
    *,
    config: Mapping[str, object] | None = None,
) -> list[Path]:
    """Return the allowlisted files that may be synchronized to Drive."""

    create_video = (config or {}).get("create_video", {}) or {}
    export_images = bool(create_video.get("EXPORT_IMAGES", False)) if isinstance(create_video, Mapping) else False
    export_raw = bool(create_video.get("EXPORT_RAW", False)) if isinstance(create_video, Mapping) else False
    root = Path(local_session)
    if not root.is_dir():
        return []
    selected: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and _is_final_artifact(path.relative_to(root), export_images=export_images, export_raw=export_raw):
            selected.append(path)
    return selected


def collect_final_artifacts(
    local_session: str,
    drive_session: str,
    *,
    config: Mapping[str, object] | None = None,
) -> list[str]:
    """Copy only allowlisted final artifacts, preserving canonical relative paths."""

    local_root = Path(local_session)
    drive_root = Path(drive_session)
    drive_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in select_final_artifacts(local_session, config=config):
        relative = source.relative_to(local_root)
        destination = drive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_file_no_overwrite(str(source), str(destination))
        copied.append(str(destination))
    return copied


def free_disk_bytes(path: str) -> tuple[int, int, int]:
    usage = shutil.disk_usage(path)
    return int(usage.total), int(usage.used), int(usage.free)
