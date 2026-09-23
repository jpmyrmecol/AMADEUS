# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Provision the checksum-pinned FFmpeg build used for hardware video encoding."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class FFmpegAsset:
    build: str
    url: str
    archive_name: str
    archive_sha256: str
    archive_size: int
    binary_name: str


# Pinned monthly BtbN builds (FFmpeg N-126342, 2026-08-31). The upstream
# retention policy keeps each month's final build for two years.
_ASSETS = {
    ("Windows", "x86_64"): FFmpegAsset(
        build="btbn-2026-08-31-N-126342-win64",
        url=(
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
            "autobuild-2026-08-31-13-27/"
            "ffmpeg-N-126342-gf88b741dbf-win64-gpl.zip"
        ),
        archive_name="ffmpeg-N-126342-gf88b741dbf-win64-gpl.zip",
        archive_sha256="b4da332540eaebc6939181b59e267f163dd57407ef6596f7f3452845921d1d91",
        archive_size=170_732_198,
        binary_name="ffmpeg.exe",
    ),
    ("Linux", "x86_64"): FFmpegAsset(
        build="btbn-2026-08-31-N-126342-linux64",
        url=(
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
            "autobuild-2026-08-31-13-27/"
            "ffmpeg-N-126342-gf88b741dbf-linux64-gpl.tar.xz"
        ),
        archive_name="ffmpeg-N-126342-gf88b741dbf-linux64-gpl.tar.xz",
        archive_sha256="d1cf19f669510448f18a4cffcdbd8fa9592ee7c15c92feb5b96ad7e9ccc30114",
        archive_size=128_065_756,
        binary_name="ffmpeg",
    ),
    ("Linux", "aarch64"): FFmpegAsset(
        build="btbn-2026-08-31-N-126342-linuxarm64",
        url=(
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
            "autobuild-2026-08-31-13-27/"
            "ffmpeg-N-126342-gf88b741dbf-linuxarm64-gpl.tar.xz"
        ),
        archive_name="ffmpeg-N-126342-gf88b741dbf-linuxarm64-gpl.tar.xz",
        archive_sha256="b65dfcc901e081ea76f8fd25c8fe2f3318f458f1d52d7d6c5423aef95b693da6",
        archive_size=109_726_420,
        binary_name="ffmpeg",
    ),
}

_HARDWARE_ENCODER_PATTERN = re.compile(
    r"\bh264_(?:nvenc|qsv|amf|vaapi|videotoolbox)\b",
    re.IGNORECASE,
)
DEFAULT_INSTALL_ROOT = Path(__file__).resolve().parents[1] / ".ffmpeg-hardware"


def _normalized_machine() -> str:
    machine = platform.machine().lower()
    if machine in {"amd64", "x64"}:
        return "x86_64"
    if machine in {"arm64", "armv8l"}:
        return "aarch64"
    return machine


def _asset_for_current_platform() -> FFmpegAsset | None:
    system = platform.system()
    asset = _ASSETS.get((system, _normalized_machine()))
    if asset is not None:
        return asset
    return None


def _windows_gpu_present() -> bool:
    """Look for NVIDIA, AMD, or Intel display adapters without requiring admin."""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell:
        try:
            completed = subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.PNPDeviceID }",
                ],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=8,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if completed.returncode == 0 and re.search(
                r"\bVEN_(?:10DE|1002|8086)\b",
                completed.stdout,
                re.IGNORECASE,
            ):
                return True
        except (OSError, subprocess.SubprocessError):
            pass

    return shutil.which("nvidia-smi") is not None


def _linux_gpu_present() -> bool:
    if shutil.which("nvidia-smi") or Path("/proc/driver/nvidia/version").is_file():
        return True
    if any(Path("/dev").glob("nvidia*")):
        return True
    if any(Path("/dev/dri").glob("renderD*")):
        return True

    # PCI vendor IDs plus the display-controller class avoid treating ordinary
    # Intel chipset, network, or storage devices as a GPU.
    pci_devices = Path("/sys/bus/pci/devices")
    try:
        devices = tuple(pci_devices.iterdir())
    except OSError:
        devices = ()

    for device in devices:
        try:
            vendor = (device / "vendor").read_text(encoding="ascii").strip().lower()
            device_class = (device / "class").read_text(encoding="ascii").strip().lower()
        except OSError:
            continue
        if vendor in {"0x10de", "0x1002", "0x8086"} and device_class.startswith("0x03"):
            return True
    return False


def gpu_hardware_present() -> bool:
    """Return whether the host exposes a GPU for which the pinned build is meant."""
    if platform.system() == "Windows":
        return _windows_gpu_present()
    if platform.system() == "Linux":
        return _linux_gpu_present()
    return False


def _binary_path(install_root: Path, asset: FFmpegAsset) -> Path:
    return install_root / asset.build / asset.binary_name


def _run_encoder_inventory(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg could not list its encoders (exit code {completed.returncode})."
        )
    output = f"{completed.stdout}\n{completed.stderr}"
    if not _HARDWARE_ENCODER_PATTERN.search(output):
        raise RuntimeError("The downloaded FFmpeg build has no supported H.264 hardware encoder.")
    return output


def _download_verified(asset: FFmpegAsset, destination: Path) -> None:
    request = urllib.request.Request(
        asset.url,
        headers={"User-Agent": "AMADEUS hardware FFmpeg installer"},
    )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) != asset.archive_size:
                raise RuntimeError(
                    f"Unexpected FFmpeg archive size: {content_length} bytes."
                )
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > asset.archive_size:
                    raise RuntimeError("FFmpeg archive exceeded its pinned size.")
                digest.update(chunk)
                out.write(chunk)
    except (OSError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the hardware FFmpeg build: {exc}") from exc

    if byte_count != asset.archive_size:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Incomplete FFmpeg download: expected {asset.archive_size} bytes, got {byte_count}."
        )
    if digest.hexdigest() != asset.archive_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError("The FFmpeg archive SHA-256 did not match the pinned build.")


def _copy_stream(stream, destination: Path) -> None:
    with destination.open("wb") as out:
        shutil.copyfileobj(stream, out)


def _extract_ffmpeg_and_license(archive_path: Path, asset: FFmpegAsset, destination: Path) -> Path:
    binary_path = destination / asset.binary_name
    license_data: bytes | None = None

    def accept_license(name: str) -> bool:
        return PurePosixPath(name).name.lower() in {
            "license.txt",
            "copying.txt",
            "copying",
        }

    if asset.archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            binary_member = None
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                mode = member.external_attr >> 16
                if member.is_dir() or stat.S_ISLNK(mode):
                    continue
                if path.name == asset.binary_name and path.parent.name == "bin":
                    binary_member = member
                elif accept_license(member.filename) and license_data is None:
                    license_data = archive.read(member)
            if binary_member is None:
                raise RuntimeError("The pinned FFmpeg archive did not contain bin/ffmpeg.exe.")
            with archive.open(binary_member) as source:
                _copy_stream(source, binary_path)
    else:
        with tarfile.open(archive_path, mode="r:*") as archive:
            binary_member = None
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if not member.isfile():
                    continue
                if path.name == asset.binary_name and path.parent.name == "bin":
                    binary_member = member
                elif accept_license(member.name) and license_data is None:
                    source = archive.extractfile(member)
                    if source is not None:
                        with source:
                            license_data = source.read()
            if binary_member is None:
                raise RuntimeError("The pinned FFmpeg archive did not contain bin/ffmpeg.")
            source = archive.extractfile(binary_member)
            if source is None:
                raise RuntimeError("Could not read FFmpeg from the pinned archive.")
            with source:
                _copy_stream(source, binary_path)

    if not license_data:
        raise RuntimeError("The pinned FFmpeg archive did not contain its license notice.")
    (destination / "LICENSE.txt").write_bytes(license_data)
    if os.name != "nt":
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return binary_path


def ensure_hardware_ffmpeg(install_root: Path | None = None) -> str | None:
    """Download and cache the pinned GPU-enabled FFmpeg for this platform."""
    asset = _asset_for_current_platform()
    if asset is None:
        return None

    root = Path(install_root) if install_root is not None else DEFAULT_INSTALL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    install_dir = root / asset.build
    executable = _binary_path(root, asset)
    manifest_path = install_dir / "build.json"

    if executable.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("archive_sha256") == asset.archive_sha256
                and manifest.get("binary_name") == asset.binary_name
            ):
                _run_encoder_inventory(executable)
                return str(executable)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            pass

    with tempfile.TemporaryDirectory(prefix=".ffmpeg-hardware-", dir=root) as temp_name:
        temp_dir = Path(temp_name)
        archive_path = temp_dir / asset.archive_name
        extracted_dir = temp_dir / "install"
        extracted_dir.mkdir()
        _download_verified(asset, archive_path)
        staged_binary = _extract_ffmpeg_and_license(archive_path, asset, extracted_dir)
        encoders = _run_encoder_inventory(staged_binary)
        found_encoders = sorted(set(_HARDWARE_ENCODER_PATTERN.findall(encoders)))
        (extracted_dir / "build.json").write_text(
            json.dumps(
                {
                    "build": asset.build,
                    "url": asset.url,
                    "archive_sha256": asset.archive_sha256,
                    "binary_name": asset.binary_name,
                    "hardware_encoders": found_encoders,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if install_dir.exists():
            shutil.rmtree(install_dir)
        extracted_dir.replace(install_dir)

    return str(executable)
