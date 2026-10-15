# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Install and launch independent applications. Never import add-on code."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import ssl
import stat
import subprocess
import sys
import tempfile
import time
from typing import Callable, Iterator
import urllib.parse
import urllib.request
import uuid
import zipfile

import certifi
import psutil

MAX_ZIP_BYTES = 512 * 1024 * 1024
MAX_EXPANDED_BYTES = 1024 * 1024 * 1024
MAX_FILES = 10000
MAX_JSON_BYTES = 64 * 1024
PLATFORMS = {
    "windows-x86_64": "Windows (x64)",
    "windows-arm64": "Windows (ARM64)",
    "macos-arm64": "macOS (Apple Silicon)",
    "macos-x86_64": "macOS (Intel)",
    "linux-x86_64": "Linux (x64)",
    "linux-arm64": "Linux (ARM64)",
}
ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[.-][a-z0-9]+)+\Z")
VERSION_PATTERN = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")
PIN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?==[A-Za-z0-9][A-Za-z0-9.!+_-]*\Z")
Progress = Callable[[str], None]


def version_tuple(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not VERSION_PATTERN.fullmatch(value):
        raise ValueError("Version must be MAJOR.MINOR.PATCH (for example 1.0.0).")
    return tuple(map(int, value.split(".")))


def platform_key() -> str:
    system = {"win32": "windows", "darwin": "macos", "linux": "linux"}.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{arch}"


def storage_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ["LOCALAPPDATA"]) / "AMADEUS"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "AMADEUS"
    else:
        value = os.environ.get("XDG_DATA_HOME")
        base = (Path(value) if value and Path(value).is_absolute() else Path.home() / ".local" / "share") / "amadeus"
    return base / "addons"


def _identifier(value: str) -> str:
    if not isinstance(value, str) or len(value) > 100 or not ID_PATTERN.fullmatch(value):
        raise ValueError("Add-on id must be a lowercase qualified name, for example org.example.viewer.")
    return value


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Unsafe package path: {value!r}")
    parts = value.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for part in parts:
        if (part in {"", ".", ".."} or part[-1:] in {".", " "}
                or any(ord(c) < 32 or c in '<>:"|?*' for c in part)
                or part.split(".")[0].upper() in reserved):
            raise ValueError(f"Unsafe package path: {value!r}")
    return PurePosixPath(value)


def _https(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Update URL must be an HTTPS URL.")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Update URLs require HTTPS without credentials or fragments.")
    return value


def _json(data: bytes) -> dict:
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON metadata exceeds 64 KiB.")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("JSON metadata must be an object.")
    return value


def validate_manifest(data: dict) -> dict:
    allowed = {"manifest_version", "id", "name", "version", "author", "description", "license",
               "platforms", "python", "entry_point", "dependencies", "update_url"}
    if set(data) - allowed:
        raise ValueError(f"Unknown manifest fields: {', '.join(sorted(set(data) - allowed))}")
    if type(data.get("manifest_version")) is not int or data["manifest_version"] != 1:
        raise ValueError("This AMADEUS supports add-on manifest_version 1.")
    _identifier(data.get("id"))
    version_tuple(data.get("version"))
    for key, maximum in (("name", 80), ("author", 160), ("license", 160)):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(ord(c) < 32 for c in value):
            raise ValueError(f"Invalid {key} in addon.json.")
    description = data.get("description", "")
    if not isinstance(description, str) or len(description) > 1000:
        raise ValueError("Description must be at most 1000 characters.")
    platforms = data.get("platforms")
    if (not isinstance(platforms, list) or not platforms or any(not isinstance(p, str) or p not in PLATFORMS for p in platforms)
            or len(set(platforms)) != len(platforms)):
        raise ValueError("platforms must list supported OS/architecture identifiers without duplicates.")
    if data.get("python") not in {"3.12", "3.13", "3.14"}:
        raise ValueError("python must be 3.12, 3.13 or 3.14.")
    entry = _relative(data.get("entry_point"))
    if entry.suffix != ".py":
        raise ValueError("entry_point must be a relative .py file.")
    dependencies = data.get("dependencies", [])
    if (not isinstance(dependencies, list) or len(dependencies) > 100
            or any(not isinstance(d, str) or not PIN_PATTERN.fullmatch(d) for d in dependencies)):
        raise ValueError("dependencies must contain exact package pins, for example PyYAML==6.0.2.")
    if data.get("update_url") is not None:
        _https(data["update_url"])
    return dict(data)


@dataclass(frozen=True)
class Package:
    path: Path
    manifest: dict
    sha256: str
    prefix: str


@dataclass(frozen=True)
class Installed:
    manifest: dict
    release: Path
    installed_at: float


def _archive_index(archive: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], str]:
    entries = archive.infolist()
    if not entries or len(entries) > MAX_FILES:
        raise ValueError("An add-on ZIP must contain 1 to 10000 entries.")
    total = 0
    seen = set()
    files = []
    for info in entries:
        path = _relative(info.filename.rstrip("/") if info.is_dir() else info.filename)
        normalized = str(path).casefold()
        if normalized in seen:
            raise ValueError(f"Duplicate or case-colliding ZIP path: {path}")
        seen.add(normalized)
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or info.flag_bits & 1:
            raise ValueError("Links, special files and encrypted ZIP entries are not supported.")
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            raise ValueError("Do not package Git metadata, virtual environments or Python caches.")
        total += info.file_size
        if total > MAX_EXPANDED_BYTES:
            raise ValueError("Add-on ZIP expands beyond the 1 GiB limit.")
        if not info.is_dir():
            files.append(info)
    names = {entry.filename for entry in files}
    candidates = [name for name in names if name == "addon.json" or (name.endswith("/addon.json") and name.count("/") == 1)]
    if len(candidates) != 1:
        raise ValueError("ZIP must contain addon.json at its root or inside one enclosing folder.")
    prefix = candidates[0][:-len("addon.json")]
    if any(not name.startswith(prefix) for name in names):
        raise ValueError("All files must be inside the add-on package folder.")
    manifest_info = archive.getinfo(prefix + "addon.json")
    if manifest_info.file_size > MAX_JSON_BYTES:
        raise ValueError("addon.json exceeds 64 KiB.")
    return files, prefix


def inspect_package(path: Path) -> Package:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError("Select an add-on ZIP no larger than 512 MiB.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    with zipfile.ZipFile(path) as archive:
        _files, prefix = _archive_index(archive)
        manifest = validate_manifest(_json(archive.read(prefix + "addon.json")))
        names = {item.filename for item in _files}
        if prefix + manifest["entry_point"] not in names or prefix + "LICENSE" not in names:
            raise ValueError("Package must include its entry_point and a LICENSE file.")
    return Package(path, manifest, digest.hexdigest(), prefix)


def _safe_path(root: Path, relative: str) -> Path:
    path = root
    for part in _relative(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Refusing a symbolic link in add-on storage: {path}")
    path.resolve().relative_to(root.resolve())
    return path


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another AMADEUS window is managing this add-on.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def clean_environment() -> dict[str, str]:
    blocked = {"VIRTUAL_ENV", "TCL_LIBRARY", "TK_LIBRARY", "__PYVENV_LAUNCHER__", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}
    return {key: value for key, value in os.environ.items()
            if key not in blocked and not key.startswith(("PYTHON", "UV_", "AMADEUS_"))}


def resolve_uv(project_root: Path) -> Path:
    # These are the locations supported by AMADEUS's existing launchers.
    candidates = ([project_root / ".uv" / "uv.exe"] if os.name == "nt" else
                  [project_root / ".uv" / "uv", project_root / ".uv" / "bin" / "uv"])
    located = shutil.which("uv")
    if located:
        candidates.append(Path(located))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("uv was not found. Start AMADEUS using its normal launcher to complete setup.")


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else storage_root()
        if not self.root.is_absolute():
            raise ValueError("Add-on storage must be an absolute path.")
        if self.root.is_symlink():
            raise ValueError("Add-on storage must not be a symbolic link.")
        self.root = self.root.resolve()

    def path(self, relative: str) -> Path:
        return _safe_path(self.root, relative)

    def get(self, addon_id: str) -> Installed | None:
        addon_id = _identifier(addon_id)
        current = self.path(f"{addon_id}/current.json")
        if not current.exists():
            return None
        state = _json(current.read_bytes())
        if state.get("schema") != 1 or not re.fullmatch(r"[0-9a-f]{32}", str(state.get("release", ""))):
            raise ValueError(f"Invalid installation record for {addon_id}.")
        release = self.path(f"{addon_id}/releases/{state['release']}")
        manifest = validate_manifest(_json((release / "payload" / "addon.json").read_bytes()))
        if manifest["id"] != addon_id:
            raise ValueError(f"Installation id mismatch for {addon_id}.")
        installed_at = state.get("installed_at")
        if type(installed_at) not in (int, float) or not math.isfinite(installed_at):
            raise ValueError(f"Invalid installation time for {addon_id}.")
        return Installed(manifest, release, float(installed_at))

    def list(self) -> tuple[list[Installed], list[str]]:
        installed, errors = [], []
        if not self.root.exists():
            return installed, errors
        for directory in sorted(self.root.iterdir()):
            if directory.name.startswith("."):
                continue
            try:
                item = self.get(directory.name)
                if item is not None:
                    installed.append(item)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"{directory.name}: {exc}")
        return sorted(installed, key=lambda item: (item.installed_at, item.manifest["id"])), errors

    def _guard(self, addon_id: str):
        return _lock(self.path(f".locks/{_identifier(addon_id)}.lock"))

    def running(self, addon_id: str) -> bool:
        release_root = str(self.path(f"{_identifier(addon_id)}/releases")) + os.sep
        for process in psutil.process_iter(["cmdline"]):
            try:
                if any(os.path.normcase(arg).startswith(os.path.normcase(release_root))
                       for arg in (process.info["cmdline"] or [])):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _require_stopped(self, addon_id: str) -> None:
        if self.running(addon_id):
            raise RuntimeError("Close this add-on before updating, reinstalling or uninstalling it.")

    def _prepare_environment(self, release: Path, manifest: dict, uv: Path, progress: Progress) -> None:
        env = clean_environment()
        env["UV_PYTHON_INSTALL_DIR"] = str(self.path(".python"))
        env["UV_CACHE_DIR"] = str(self.path(".cache"))
        env["UV_PYTHON_INSTALL_BIN"] = "false"
        commands = [[str(uv), "--no-config", "venv", "--managed-python", "--python", manifest["python"], str(release / ".venv")]]
        if manifest.get("dependencies"):
            commands.append([str(uv), "--no-config", "pip", "install", "--python", str(self.python(release)),
                             "--only-binary", ":all:", "--", *manifest["dependencies"]])
        log_path = self.path(f"{manifest['id']}/install.log")
        with log_path.open("w", encoding="utf-8") as log:
            for command in commands:
                progress("Preparing Python environment and dependencies...")
                log.write("$ " + subprocess.list2cmdline(command) + "\n")
                log.flush()
                kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                completed = subprocess.run(command, cwd=release, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, timeout=1200, **kwargs)
                if completed.returncode:
                    raise RuntimeError(f"Environment setup failed (exit {completed.returncode}). See {log_path}")
        if not self.python(release).is_file():
            raise RuntimeError("The add-on Python environment was not created.")

    @staticmethod
    def python(release: Path) -> Path:
        return release / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def install(self, package: Package, uv: Path, progress: Progress = lambda _message: None,
                expected_id: str | None = None) -> Installed:
        manifest = package.manifest
        addon_id = _identifier(manifest["id"])
        if expected_id is not None and addon_id != expected_id:
            raise ValueError("The selected package belongs to a different add-on.")
        if platform_key() not in manifest["platforms"]:
            raise ValueError(f"This package does not declare support for {platform_key()}.")
        with self._guard(addon_id):
            self._require_stopped(addon_id)
            old = self.get(addon_id)
            if old and version_tuple(manifest["version"]) < version_tuple(old.manifest["version"]):
                raise ValueError("Downgrades are not supported. Select the installed version or a newer version.")
            release = self.path(f"{addon_id}/releases/{uuid.uuid4().hex}")
            release.mkdir(parents=True)
            try:
                # Copy before trusting the approval digest. Nothing from the ZIP runs here.
                private_zip = release / "package.zip"
                with package.path.open("rb") as source, private_zip.open("xb") as target:
                    total = 0
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        total += len(block)
                        if total > MAX_ZIP_BYTES:
                            raise ValueError("Package grew beyond the size limit.")
                        target.write(block)
                checked = inspect_package(private_zip)
                if checked.sha256 != package.sha256 or checked.manifest != manifest:
                    raise ValueError("The package changed after it was selected. Select it again.")
                payload = release / "payload"
                payload.mkdir()
                progress("Extracting validated package...")
                with zipfile.ZipFile(private_zip) as archive:
                    entries, prefix = _archive_index(archive)
                    for info in entries:
                        target = _safe_path(payload, info.filename[len(prefix):])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as source, target.open("xb") as destination:
                            shutil.copyfileobj(source, destination)
                private_zip.unlink()
                self._prepare_environment(release, manifest, uv, progress)
                self.path(f"{addon_id}/data").mkdir(parents=True, exist_ok=True)
                installed_at = old.installed_at if old else time.time()
                _atomic_json(self.path(f"{addon_id}/current.json"),
                             {"schema": 1, "release": release.name, "installed_at": installed_at})
            except BaseException:
                shutil.rmtree(release)
                raise
            # Environments are built at their final paths. Moving a venv breaks scripts.
            for directory in self.path(f"{addon_id}/releases").iterdir():
                if directory != release:
                    try:
                        self._remove_release(directory)
                    except (OSError, ValueError) as exc:
                        progress(f"Installed. Could not remove unused environment {directory.name}: {exc}")
            progress("Installation complete.")
            return self.get(addon_id)

    def _remove_release(self, path: Path) -> None:
        if path.is_symlink() or not re.fullmatch(r"[0-9a-f]{32}", path.name):
            raise ValueError(f"Unsafe release directory: {path}")
        path.resolve().relative_to(self.root)
        shutil.rmtree(path)

    def uninstall(self, addon_id: str) -> None:
        with self._guard(addon_id):
            self._require_stopped(addon_id)
            for release in self.path(f"{addon_id}/releases").glob("*"):
                self._remove_release(release)
            self.path(f"{addon_id}/current.json").unlink(missing_ok=True)
            # data/ and logs deliberately survive uninstall. Never delete research inputs.

    def launch(self, addon_id: str, *, config: Path | None = None, results: Path | None = None) -> tuple[subprocess.Popen, Path]:
        with self._guard(addon_id):
            item = self.get(addon_id)
            if item is None:
                raise ValueError("This add-on is not installed.")
            if platform_key() not in item.manifest["platforms"]:
                raise ValueError(f"This add-on does not declare support for {platform_key()}.")
            self._require_stopped(addon_id)
            script = _safe_path(item.release / "payload", item.manifest["entry_point"])
            command = [str(self.python(item.release)), "-E", "-s", str(script)]
            for flag, path in (("--config", config), ("--results", results)):
                if path is not None:
                    path = Path(path).resolve(strict=True)
                    if not path.is_file():
                        raise ValueError(f"Input must be a file: {path}")
                    command.extend([flag, str(path)])
            env = clean_environment()
            data = self.path(f"{addon_id}/data")
            data.mkdir(parents=True, exist_ok=True)
            env["AMADEUS_ADDON_DATA_DIR"] = str(data)
            env["PATH"] = str(self.python(item.release).parent) + os.pathsep + env.get("PATH", "")
            log_path = self.path(f"{addon_id}/run.log")
            kwargs = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True})
            with log_path.open("ab") as log:
                log.write(f"\nStarting {addon_id} {item.manifest['version']}\n".encode())
                log.flush()
                process = subprocess.Popen(command, cwd=data, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, **kwargs)
            return process, log_path


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url: str, destination: Path, limit: int) -> None:
    _https(url)
    context = ssl.create_default_context(cafile=certifi.where())
    context.load_default_certs()
    opener = urllib.request.build_opener(_HTTPSRedirect(), urllib.request.HTTPSHandler(context=context))
    request = urllib.request.Request(url, headers={"User-Agent": "AMADEUS-Addons/1"})
    with opener.open(request, timeout=30) as response, destination.open("xb") as output:
        _https(response.geturl())
        total = 0
        for block in iter(lambda: response.read(1024 * 1024), b""):
            total += len(block)
            if total > limit:
                raise ValueError("Downloaded file exceeds the size limit.")
            output.write(block)


def download_update(item: Installed, destination: Path, progress: Progress = lambda _message: None) -> Package | None:
    """Download only. The caller must obtain consent again before installing."""
    update_url = item.manifest.get("update_url")
    if not update_url:
        raise ValueError("This add-on uses local ZIP updates.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="amadeus-update-") as temporary:
        metadata = Path(temporary) / "update.json"
        progress("Checking the add-on's update source...")
        _download(update_url, metadata, MAX_JSON_BYTES)
        update = _json(metadata.read_bytes())
        if update.get("manifest_version") != 1 or update.get("id") != item.manifest["id"]:
            raise ValueError("Update metadata does not match this add-on.")
        if version_tuple(update.get("version")) <= version_tuple(item.manifest["version"]):
            return None
        digest = update.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Update metadata must include the ZIP's SHA-256 digest.")
        archive = destination / f"{uuid.uuid4().hex}.zip"
        try:
            progress("Downloading add-on update...")
            _download(update.get("url"), archive, MAX_ZIP_BYTES)
            package = inspect_package(archive)
            if (package.sha256 != digest or package.manifest["id"] != item.manifest["id"]
                    or package.manifest["version"] != update["version"]):
                raise ValueError("Downloaded update failed the identity, version or SHA-256 check.")
            return package
        except BaseException:
            archive.unlink(missing_ok=True)
            raise
