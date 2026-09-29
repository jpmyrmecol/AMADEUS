#!/usr/bin/env python3
"""Prepare a private Xft-enabled Tk 9 runtime on Ubuntu when Tk lacks Xft."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = APP_DIR / ".uv" / "tk-font-runtime"


def _multiarch() -> str:
    import sysconfig

    return str(sysconfig.get_config_var("MULTIARCH") or "")


def _tk_runtime_info() -> tuple[str, str]:
    root = tk.Tk()
    root.withdraw()
    try:
        version = str(root.tk.call("package", "present", "Tk"))
        backend = str(root.tk.call("tk::pkgconfig", "get", "fontsystem"))
        return version, backend
    finally:
        root.destroy()


def _apt_options(work_dir: Path, source_list: Path) -> list[str]:
    return [
        "-o",
        f"Dir::Etc::sourcelist={source_list}",
        "-o",
        "Dir::Etc::sourceparts=-",
        "-o",
        f"Dir::State::lists={work_dir / 'lists'}",
        "-o",
        f"Dir::Cache::archives={work_dir / 'archives'}",
        "-o",
        "APT::Get::List-Cleanup=0",
        "-o",
        "APT::Sandbox::User=",
        "-o",
        "Acquire::Languages=none",
    ]


def _verify_runtime(runtime_root: Path, library_dir: Path) -> None:
    import _tkinter

    tcl_runtime_dir = Path(_tkinter.__file__).resolve().parents[2]
    env = os.environ.copy()
    env["TK_LIBRARY"] = str(runtime_root / "usr/lib/tk9.0")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(library_dir), str(tcl_runtime_dir), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    preload = str(library_dir / "libtcl9tk9.0.so")
    if env.get("LD_PRELOAD"):
        preload += os.pathsep + env["LD_PRELOAD"]
    env["LD_PRELOAD"] = preload

    probe = (
        "import tkinter as tk; root=tk.Tk(); root.withdraw(); "
        "print(root.tk.call('tk::pkgconfig','get','fontsystem')); root.destroy()"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != "xft":
        detail = (result.stderr or result.stdout or "Tk did not report Xft support").strip()
        raise RuntimeError(f"The private Tk runtime failed its Xft check: {detail[:700]}")


def _install_runtime() -> None:
    release: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                release[key] = value.strip().strip('"')
    except OSError as exc:
        raise RuntimeError(f"Cannot read /etc/os-release: {exc}") from exc

    if release.get("ID") != "ubuntu" or platform.machine().lower() != "x86_64":
        raise RuntimeError("Automatic Xft Tk setup is supported on Ubuntu x86_64 only.")
    if not release.get("VERSION_CODENAME"):
        raise RuntimeError("Ubuntu did not report VERSION_CODENAME.")

    apt_get = shutil.which("apt-get")
    dpkg_deb = shutil.which("dpkg-deb")
    if not apt_get or not dpkg_deb:
        raise RuntimeError("apt-get and dpkg-deb are required for the private Tk runtime.")

    uv_dir = APP_DIR / ".uv"
    uv_dir.mkdir(parents=True, exist_ok=True)
    if RUNTIME_DIR.exists():
        raise RuntimeError(f"An incomplete Tk runtime already exists at {RUNTIME_DIR}.")

    with tempfile.TemporaryDirectory(prefix="amadeus-tk-xft-", dir=uv_dir) as temp_name:
        work_dir = Path(temp_name)
        work_dir.chmod(0o755)
        lists_dir = work_dir / "lists"
        (lists_dir / "partial").mkdir(parents=True)
        (lists_dir / "auxfiles").mkdir(parents=True)
        archives_dir = work_dir / "archives"
        archives_dir.mkdir()
        download_dir = work_dir / "downloads"
        download_dir.mkdir()
        source_list = work_dir / "sources.list"
        source_list.write_text(
            f"deb https://archive.ubuntu.com/ubuntu {release['VERSION_CODENAME']} main universe\n",
            encoding="utf-8",
        )

        apt_options = _apt_options(work_dir, source_list)
        subprocess.run(
            [apt_get, *apt_options, "-qq", "update"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        subprocess.run(
            [apt_get, *apt_options, "-qq", "download", "libtk9.0", "libxft2", "libxss1"],
            cwd=download_dir,
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )

        staged_root = work_dir / "runtime"
        staged_root.mkdir()
        debs = tuple(download_dir.glob("*.deb"))
        if len(debs) != 3:
            raise RuntimeError(f"Expected 3 Ubuntu packages, downloaded {len(debs)}.")
        for deb in debs:
            subprocess.run(
                [dpkg_deb, "-x", str(deb), str(staged_root)],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )

        multiarch = _multiarch()
        if not multiarch:
            raise RuntimeError("Python did not report its Linux multiarch library directory.")
        staged_library_dir = staged_root / "usr/lib" / multiarch
        required = (
            staged_library_dir / "libtcl9tk9.0.so",
            staged_library_dir / "libXft.so.2",
            staged_library_dir / "libXss.so.1",
            staged_root / "usr/lib/tk9.0",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"The Ubuntu packages did not provide required Tk files: {missing}")

        _verify_runtime(staged_root, staged_library_dir)
        RUNTIME_DIR.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_root, RUNTIME_DIR)


def main() -> int:
    if not sys.platform.startswith("linux"):
        return 0

    try:
        tk_version, backend = _tk_runtime_info()
    except tk.TclError as exc:
        print(f"[AMADEUS] Could not inspect the Tk font backend: {exc}", file=sys.stderr)
        return 1

    if backend == "xft":
        print("[AMADEUS] Tk Xft font backend is already active.")
        return 0

    if not tk_version.startswith("9."):
        print(
            f"[AMADEUS] Tk {tk_version} uses '{backend}' fonts; automatic Xft setup requires Tk 9.",
            file=sys.stderr,
        )
        return 1

    runtime_lib_dir = RUNTIME_DIR / "usr/lib" / _multiarch()
    runtime_ready = all(
        path.exists()
        for path in (
            runtime_lib_dir / "libtcl9tk9.0.so",
            runtime_lib_dir / "libXft.so.2",
            runtime_lib_dir / "libXss.so.1",
            RUNTIME_DIR / "usr/lib/tk9.0",
        )
    )
    if runtime_ready:
        print("[AMADEUS] Private Xft-enabled Tk runtime is ready.")
        return 0

    print(f"[AMADEUS] Tk reports the '{backend}' font backend; preparing Xft support.")
    try:
        _install_runtime()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        detail = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or detail).strip()
        print(f"[AMADEUS] Could not prepare the private Xft Tk runtime: {detail[:900]}", file=sys.stderr)
        return 1

    print("[AMADEUS] Prepared a private Xft-enabled Tk runtime in .uv/tk-font-runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

