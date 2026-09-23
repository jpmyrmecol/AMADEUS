# SPDX-License-Identifier: AGPL-3.0-only

"""Run AMADEUS in a Colab VM and export only final artifacts to Drive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main.colab_utils import (
    MIN_FREE_BYTES,
    RUNTIME_NOTEBOOK_RELATIVE_PATH,
    RUNTIME_NOTEBOOK_TEMPLATE_RELATIVE_PATH,
    RUNTIME_PACKAGE_DIRECTORY,
    RUNTIME_PACKAGE_MANIFEST,
    WARN_FREE_BYTES,
    collect_final_artifacts,
    free_disk_bytes,
    validate_colab_config,
)
from main.path_utils import CONFIG_PATH_KEYS


_RUNTIME_REQUIRED_FILES = ("VERSION", "tools/UV_VERSION", "uv.lock", RUNTIME_NOTEBOOK_TEMPLATE_RELATIVE_PATH)
_RUNTIME_PROVENANCE_FIELDS = (
    "source_ref",
    "source_revision",
    "version",
    "lock_sha256",
    "template_sha256",
    "package_manifest_sha256",
)


def _runtime_package_error(message: str) -> RuntimeError:
    return RuntimeError(f"Prepared AMADEUS runtime package {message}.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _runtime_package_error(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_manifest_hash(entries: list[Mapping[str, object]]) -> str:
    payload = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_runtime_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise _runtime_package_error(f"contains an invalid file path: {value!r}")
    if value.startswith(("/", "\\")) or "\\" in value or ":" in value:
        raise _runtime_package_error(f"contains an unsafe file path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise _runtime_package_error(f"contains an unsafe file path: {value!r}")
    return value


def _package_path(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*relative.split("/"))
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise _runtime_package_error(f"contains a symlinked path: {current}")
    return candidate


def _validated_manifest_entries(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise _runtime_package_error("manifest has no file entries")
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise _runtime_package_error(f"manifest file entry {index} is not an object")
        relative = _safe_runtime_relative_path(raw_entry.get("path"))
        key = relative.casefold()
        if key in seen:
            raise _runtime_package_error(f"manifest contains a duplicate file path: {relative}")
        seen.add(key)
        size = raw_entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise _runtime_package_error(f"manifest has an invalid size for {relative}")
        sha256 = raw_entry.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise _runtime_package_error(f"manifest has an invalid SHA-256 for {relative}")
        entries.append({"path": relative, "size": size, "sha256": sha256})
    return entries


def _validate_runtime_tree(root: Path, allowed_files: set[str]) -> None:
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(directory_names):
            child = current_path / name
            if child.is_symlink():
                raise _runtime_package_error(f"contains a symlink: {child}")
            relative = child.relative_to(root).as_posix()
            if not any(path == relative or path.startswith(relative + "/") for path in allowed_files):
                raise _runtime_package_error(f"contains unrelated data: {child}")
        for name in file_names:
            child = current_path / name
            if child.is_symlink():
                raise _runtime_package_error(f"contains a symlink: {child}")
            relative = child.relative_to(root).as_posix()
            if relative not in allowed_files:
                raise _runtime_package_error(f"contains unrelated data: {child}")


def validate_runtime_package(package_root: str | Path, *, strict_tree: bool = False) -> dict[str, object]:
    """Validate the revision-bound runtime package before a Colab run."""

    root = Path(os.path.abspath(os.fspath(package_root)))
    if root.is_symlink() or not root.is_dir():
        raise _runtime_package_error(f"is missing or is not a directory: {root}")
    manifest_path = root / RUNTIME_PACKAGE_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise _runtime_package_error(f"manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _runtime_package_error(f"manifest cannot be read: {manifest_path} ({exc})") from exc
    if not isinstance(manifest, dict):
        raise _runtime_package_error("manifest root is not an object")
    if manifest.get("schema_version") != 2:
        raise _runtime_package_error("manifest schema_version is unsupported")
    if manifest.get("package_name") != "amadeus-colab-runtime":
        raise _runtime_package_error("manifest package_name is unexpected")
    if manifest.get("package_root") != RUNTIME_PACKAGE_DIRECTORY:
        raise _runtime_package_error("manifest package_root is unexpected")
    for field in ("source_ref", "source_revision", "version"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise _runtime_package_error(f"manifest {field} is missing")
    entries = _validated_manifest_entries(manifest)
    mutable_files = manifest.get("mutable_files")
    expected_mutable_files = [
        {
            "path": RUNTIME_NOTEBOOK_RELATIVE_PATH,
            "kind": "project-specific-launcher",
            "template_path": RUNTIME_NOTEBOOK_TEMPLATE_RELATIVE_PATH,
        }
    ]
    if mutable_files != expected_mutable_files:
        raise _runtime_package_error("manifest mutable_files is invalid")
    launcher_path = _package_path(root, RUNTIME_NOTEBOOK_RELATIVE_PATH)
    if not launcher_path.is_file():
        raise _runtime_package_error(f"mutable launcher is missing: {launcher_path}")
    manifest_digest = manifest.get("package_manifest_sha256")
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
        raise _runtime_package_error("manifest package_manifest_sha256 is missing")
    if _canonical_manifest_hash(entries) != manifest_digest:
        raise _runtime_package_error("manifest file-entry hash does not match package_manifest_sha256")

    entry_by_path = {str(entry["path"]): entry for entry in entries}
    for relative in _RUNTIME_REQUIRED_FILES:
        if relative not in entry_by_path:
            raise _runtime_package_error(f"manifest is missing required file {relative}")
    if any(
        str(entry["path"]).casefold() == RUNTIME_NOTEBOOK_RELATIVE_PATH.casefold()
        for entry in entries
    ):
        raise _runtime_package_error("mutable launcher must not be an immutable file entry")
    allowed_files = set(entry_by_path) | {RUNTIME_PACKAGE_MANIFEST, RUNTIME_NOTEBOOK_RELATIVE_PATH}
    if strict_tree:
        _validate_runtime_tree(root, allowed_files)
    for entry in entries:
        relative = str(entry["path"])
        path = _package_path(root, relative)
        if not path.is_file():
            raise _runtime_package_error(f"is missing listed file: {path}")
        expected_size = int(entry["size"])
        expected_digest = str(entry["sha256"])
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise _runtime_package_error(f"cannot stat {path}: {exc}") from exc
        if actual_size != expected_size:
            raise _runtime_package_error(
                f"has an unexpected size for {relative} (expected {expected_size}, got {actual_size})"
            )
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise _runtime_package_error(f"has a SHA-256 mismatch for {relative}")

    version_path = _package_path(root, "VERSION")
    try:
        version = version_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise _runtime_package_error(f"cannot read VERSION: {version_path} ({exc})") from exc
    if version != manifest["version"]:
        raise _runtime_package_error("VERSION does not match the manifest version")
    for field, relative in (("lock_sha256", "uv.lock"), ("template_sha256", RUNTIME_NOTEBOOK_TEMPLATE_RELATIVE_PATH)):
        expected_digest = manifest.get(field)
        if not isinstance(expected_digest, str) or expected_digest != entry_by_path[relative]["sha256"]:
            raise _runtime_package_error(f"manifest {field} does not match {relative}")

    legal_files = manifest.get("legal_files")
    if not isinstance(legal_files, list):
        raise _runtime_package_error("manifest legal_files is missing")
    for raw_path in legal_files:
        relative = _safe_runtime_relative_path(raw_path)
        if relative not in entry_by_path:
            raise _runtime_package_error(f"manifest legal file is not packaged: {relative}")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise _runtime_package_error("manifest provenance is missing")
    for field in _RUNTIME_PROVENANCE_FIELDS:
        if provenance.get(field) != manifest.get(field):
            raise _runtime_package_error(f"manifest provenance does not match {field}")
    return dict(manifest)


def stage_runtime_package(
    source_root: str | Path,
    destination_root: str | Path = "/content/AMADEUS",
) -> dict[str, object]:
    """Validate and copy only the prepared runtime package into local storage."""

    source = Path(os.path.abspath(os.fspath(source_root)))
    destination = Path(os.path.abspath(os.fspath(destination_root)))
    if source.is_symlink() or not source.is_dir():
        raise _runtime_package_error(f"source is missing or is not a directory: {source}")
    if _key(str(source)) == _key(str(destination)) or _inside(str(source), str(destination)) or _inside(
        str(destination), str(source)
    ):
        raise _runtime_package_error("source and local destination must be separate directories")
    manifest = validate_runtime_package(source, strict_tree=True)
    launcher_path = _package_path(source, RUNTIME_NOTEBOOK_RELATIVE_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise _runtime_package_error(f"local destination is not a directory: {destination}")

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.part-", dir=str(destination.parent))
    )
    backup: Path | None = None
    try:
        for entry in manifest["files"]:
            relative = str(entry["path"])
            source_path = _package_path(source, relative)
            assert temporary is not None
            destination_path = temporary.joinpath(*relative.split("/"))
            _copy_file(str(source_path), str(destination_path))
        launcher_destination = temporary.joinpath(*RUNTIME_NOTEBOOK_RELATIVE_PATH.split("/"))
        _copy_file(str(launcher_path), str(launcher_destination))
        assert temporary is not None
        _copy_file(str(source / RUNTIME_PACKAGE_MANIFEST), str(temporary / RUNTIME_PACKAGE_MANIFEST))
        validate_runtime_package(temporary, strict_tree=True)

        if destination.exists():
            backup = destination.parent / f".{destination.name}.previous-{os.getpid()}"
            if backup.exists() or backup.is_symlink():
                raise _runtime_package_error(f"cannot create a safe backup beside {destination}")
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except BaseException:
            if backup is not None and not destination.exists() and backup.exists():
                os.replace(backup, destination)
            raise
        temporary = None
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
    return validate_runtime_package(destination, strict_tree=True)


stage_colab_runtime_package = stage_runtime_package


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _inside(path: str, parent: str) -> bool:
    try:
        return os.path.commonpath([_key(path), _key(parent)]) == _key(parent)
    except ValueError:
        return False


def _copy_file(source: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree_files(source: str, destination: str) -> None:
    source_path = Path(source)
    for child in sorted(source_path.rglob("*")):
        if child.is_file():
            relative = child.relative_to(source_path)
            _copy_file(str(child), str(Path(destination) / relative))


def _runtime_input_bundle(config: Mapping[str, object], summary, workspace_root: Path) -> tuple[Path, Path, dict[str, str]]:
    """Copy Drive inputs to local storage and return local session/config paths."""

    drive_session = os.path.abspath(summary.session_path)
    local_session = workspace_root / "project"
    local_session.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}

    def map_file(source: str, destination: str) -> None:
        source = os.path.abspath(source)
        destination = os.path.abspath(destination)
        if _key(source) in mapping:
            return
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Required Drive input is missing: {source}")
        _copy_file(source, destination)
        mapping[_key(source)] = destination

    def map_path(source: str, *, name: str, allow_missing: bool = False) -> str:
        source = os.path.abspath(source)
        if not os.path.exists(source):
            if allow_missing:
                if _inside(source, drive_session):
                    return str(local_session / os.path.relpath(source, drive_session))
                return str(local_session / name)
            raise FileNotFoundError(f"Required Drive input is missing: {source}")
        if _inside(source, drive_session):
            destination = local_session / os.path.relpath(source, drive_session)
        else:
            destination = workspace_root / "inputs" / name
        if os.path.isfile(source):
            map_file(source, str(destination))
        elif os.path.isdir(source):
            destination.mkdir(parents=True, exist_ok=True)
            _copy_tree_files(source, str(destination))
            mapping[_key(source)] = str(destination)
        return str(destination)

    # The exact segmentation files used by the current source are copied.  No
    # segmentation cache or preview directory is copied.
    segmentation_sources = (
        summary.segmentation_pickle_path,
        summary.background_path,
        os.path.join(drive_session, "segmentation", "segmentation_gui_config.json"),
    )
    for source in segmentation_sources:
        if os.path.isfile(source):
            map_file(source, str(local_session / "segmentation" / os.path.basename(source)))

    video_values = [summary.training_video_path, summary.tracking_video_path]
    video_mapping: dict[str, str] = {}
    for raw_value in video_values:
        source = os.path.abspath(raw_value)
        if os.path.isdir(source):
            destination = workspace_root / os.path.basename(os.path.normpath(source))
            destination.mkdir(parents=True, exist_ok=True)
            for child in sorted(source.iterdir()):
                if child.is_file() and child.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
                    map_file(str(child), str(destination / child.name))
            video_mapping[_key(source)] = str(destination)
        else:
            destination = local_session / os.path.relpath(source, drive_session) if _inside(source, drive_session) else workspace_root / os.path.basename(source)
            map_file(source, str(destination))
            video_mapping[_key(source)] = str(destination)

    # Explicitly referenced files and directories are copied when they are
    # needed by a skipped stage or a custom dataset.  Unsupported missing paths
    # stop the run before any batch stage starts.
    path_mapping = dict(video_mapping)
    path_mapping[_key(summary.segmentation_pickle_path)] = str(local_session / "segmentation" / os.path.basename(summary.segmentation_pickle_path))
    path_mapping[_key(summary.background_path)] = str(local_session / "segmentation" / os.path.basename(summary.background_path))
    for key in CONFIG_PATH_KEYS:
        if key in {"TRAINING_VIDEO_PATH", "TRACKING_VIDEO_PATH"}:
            continue
        raw_value = config.get(key)
        values = [raw_value] if isinstance(raw_value, str) else []
        for raw in values:
            if not str(raw or "").strip():
                continue
            path = os.path.abspath(str(raw))
            target_name = os.path.basename(path) or key.lower()
            allow_missing = key == "INIT_CSV_PATH" and not bool(config.get("skip_initial_tracking", False))
            path_mapping[_key(path)] = map_path(path, name=target_name, allow_missing=allow_missing)

    raw_dirs = config.get("CREATE_DATASET_SOURCE_DIRS")
    if raw_dirs:
        values = [raw_dirs] if isinstance(raw_dirs, str) else list(raw_dirs)
        rewritten_dirs = []
        for raw in values:
            path = os.path.abspath(str(raw))
            target = map_path(path, name=os.path.basename(os.path.normpath(path)))
            path_mapping[_key(path)] = target
            rewritten_dirs.append(target)
        config = dict(config)
        config["CREATE_DATASET_SOURCE_DIRS"] = rewritten_dirs if not isinstance(raw_dirs, str) else rewritten_dirs[0]

    runtime = dict(config)
    runtime["SESSION_PATH"] = str(local_session)
    for key in CONFIG_PATH_KEYS:
        value = runtime.get(key)
        if isinstance(value, str) and value.strip():
            source = os.path.abspath(value)
            if source in {"", os.path.abspath(".")}:  # pragma: no cover - defensive only
                continue
            if _key(source) in path_mapping:
                runtime[key] = path_mapping[_key(source)]
            elif key == "INIT_CSV_PATH" and not bool(runtime.get("skip_initial_tracking", False)):
                runtime[key] = str(local_session / "initial_tracking" / os.path.basename(source))
            else:
                raise ValueError(f"Could not map runtime config path {key}: {source}")
    if runtime.get("CREATE_DATASET_SOURCE_DIRS"):
        values = runtime["CREATE_DATASET_SOURCE_DIRS"]
        values = [values] if isinstance(values, str) else list(values)
        runtime["CREATE_DATASET_SOURCE_DIRS"] = [
            path_mapping.get(_key(os.path.abspath(str(value))), str(value)) for value in values
        ]

    local_config = local_session / "config_colab.yaml"
    with open(local_config, "w", encoding="utf-8") as stream:
        yaml.safe_dump(runtime, stream, sort_keys=False, allow_unicode=True)
    return local_session, Path(drive_session), runtime


class DiskGuard:
    """Stop a batch when Colab local storage falls below the hard limit."""

    def __init__(
        self,
        path: str,
        process: subprocess.Popen,
        *,
        interval_seconds: int = 60,
        minimum_free_bytes: int = MIN_FREE_BYTES,
        warning_free_bytes: int = WARN_FREE_BYTES,
        notify: Callable[[str], None] = print,
    ) -> None:
        self.path = path
        self.process = process
        self.interval_seconds = max(1, int(interval_seconds))
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.warning_free_bytes = int(warning_free_bytes)
        self.notify = notify
        self.stop_event = threading.Event()
        self.failed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="amadeus-colab-disk-guard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            _total, _used, free = free_disk_bytes(self.path)
            if free < self.minimum_free_bytes:
                self.failed.set()
                self.notify(f"[DISK] free space is below 5 GiB ({free / 1024**3:.2f} GiB); stopping batch gracefully.")
                _terminate_process(self.process)
                return
            if free < self.warning_free_bytes:
                self.notify(f"[DISK] warning: free space is {free / 1024**3:.2f} GiB.")


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.send_signal(signal.CTRL_BREAK_EVENT)
    except (OSError, ValueError):
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except (OSError, ValueError):
            pass


def _write_environment_log(
    path: Path,
    *,
    provenance: Mapping[str, object],
) -> None:
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
    except Exception as exc:
        torch_version = cuda_version = "unavailable"
        cuda_available = False
        gpu_name = f"unavailable ({exc})"
    total, used, free = free_disk_bytes("/content")
    source_revision = str(provenance.get("source_revision") or "unavailable")
    lines = [
        f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"amadeus_version={provenance.get('version', 'unavailable')}",
        f"source_ref={provenance.get('source_ref', 'unavailable')}",
        f"source_revision={source_revision}",
        f"lock_sha256={provenance.get('lock_sha256', 'unavailable')}",
        f"template_sha256={provenance.get('template_sha256', 'unavailable')}",
        f"package_manifest_sha256={provenance.get('package_manifest_sha256', 'unavailable')}",
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"torch={torch_version}",
        f"torch_cuda={cuda_version}",
        f"torch_cuda_available={cuda_available}",
        f"gpu={gpu_name}",
        f"disk_total_bytes={total}",
        f"disk_used_bytes={used}",
        f"disk_free_bytes={free}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gpu_preflight() -> None:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("GPU runtime is required, but PyTorch could not be imported.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU runtime is required. Select Runtime -> Change runtime type -> GPU, then run all again."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}; CUDA build: {torch.version.cuda}")


def run_colab_workflow(config_path: str) -> int:
    provenance = validate_runtime_package(PROJECT_ROOT)
    cfg, summary = validate_colab_config(config_path, check_video=True)
    total, used, free = free_disk_bytes("/content")
    print(f"Colab disk: total={total / 1024**3:.2f} GiB, used={used / 1024**3:.2f} GiB, free={free / 1024**3:.2f} GiB")
    if free < MIN_FREE_BYTES:
        raise RuntimeError("There is less than 5 GiB free on the Colab local disk. The batch was not started.")
    if free < WARN_FREE_BYTES:
        print("[DISK] warning: less than 10 GiB is free before the batch starts.")
    _gpu_preflight()

    workspace_root = Path("/content") / f"amadeus_workspace_{int(time.time())}"
    workspace_root.mkdir(parents=True, exist_ok=False)
    local_session: Path | None = None
    drive_session: Path | None = None
    runtime_cfg: dict = {}
    log_path: Path | None = None
    logger = None
    status = "failed"
    try:
        local_session = workspace_root / "project"
        drive_session = Path(summary.session_path)
        local_session.mkdir(parents=True, exist_ok=True)
        local_session, drive_session, runtime_cfg = _runtime_input_bundle(cfg, summary, workspace_root)
        log_path = local_session / "logs" / "colab_runner.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = open(log_path, "a", encoding="utf-8", buffering=1)
        _write_environment_log(local_session / "logs" / "colab_environment.txt", provenance=provenance)
        print(f"Local workspace: {workspace_root}")
        print(
            "AMADEUS package: "
            f"version={provenance['version']} "
            f"source_ref={provenance['source_ref']} "
            f"source_revision={provenance['source_revision']}"
        )
        print(f"Drive project: {drive_session}")

        def emit(line: str) -> None:
            print(line, end="")
            if logger is not None:
                logger.write(line)
                logger.flush()

        command = [sys.executable, "-u", str(PROJECT_ROOT / "main" / "batch.py"), str(local_session / "config_colab.yaml")]
        emit(">>> " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
        guard = DiskGuard(str(workspace_root), process, notify=emit)
        guard.start()
        assert process.stdout is not None
        for line in process.stdout:
            emit(line)
        return_code = process.wait()
        guard.stop()
        if guard.failed.is_set():
            raise RuntimeError("Batch stopped because Colab local disk space fell below 5 GiB.")
        if return_code != 0:
            raise RuntimeError(f"AMADEUS batch failed with exit code {return_code}.")
        status = "success"
        emit("AMADEUS batch completed. Collecting final artifacts only.\n")
        return 0
    except BaseException:
        if local_session is not None:
            failure_path = local_session / "logs" / "colab_failure_traceback.txt"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if logger is not None:
            logger.write(f"status={status}\n")
            logger.flush()
            logger.close()
        if local_session is not None and drive_session is not None:
            try:
                exported = collect_final_artifacts(str(local_session), str(drive_session), config=runtime_cfg)
                print(f"Exported {len(exported)} final artifacts to Drive.")
                print("Drive export excludes regenerable training images, paste images, datasets, labels, previews, and caches.")
            except Exception as sync_exc:
                print(f"[ERROR] final artifact sync failed: {sync_exc}")
                if status == "success":
                    raise
        if status == "success":
            shutil.rmtree(workspace_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Colab-compatible config path under /content/drive/MyDrive")
    args = parser.parse_args()
    try:
        run_colab_workflow(args.config)
    except Exception as exc:
        print(f"AMADEUS Colab workflow failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
