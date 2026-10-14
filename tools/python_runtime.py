"""Select a compatible system Python and Xft-enabled Tk runtime for AMADEUS."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def check_python_runtime(python: str, required_minor: str | None = None) -> None:
    """Check interpreter and Tk in the candidate's own process."""
    code = """
import platform
import sys
import tkinter as tk
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise RuntimeError('Python 3.10-3.12 required')
if sys.platform == 'darwin' and sys.version_info[:2] != (3, 12):
    raise RuntimeError('macOS requires Python 3.12')
if sys.platform == 'darwin' and platform.machine() not in ('arm64', 'aarch64'):
    raise RuntimeError('Native Apple Silicon Python required')
root = tk.Tk()
try:
    version = str(root.tk.call('info', 'patchlevel'))
    if sys.platform == 'darwin' and not version.startswith('8.6.'):
        raise RuntimeError('macOS requires Tk 8.6')
    if sys.platform.startswith('linux'):
        fontsystem = str(root.tk.call('tk::pkgconfig', 'get', 'fontsystem'))
        if not version.startswith('8.6.') or fontsystem != 'xft':
            raise RuntimeError(f'Linux requires Tk 8.6/Xft; found Tk {version}, {fontsystem}')
finally:
    root.destroy()
"""
    if required_minor is not None:
        code += f"""
import sys
if f'{{sys.version_info.major}}.{{sys.version_info.minor}}' != {required_minor!r}:
    raise RuntimeError('This accelerator profile requires Python {required_minor}')
"""
    subprocess.run([python, "-c", code], capture_output=True, text=True, check=True)


def python_base_prefix(python: str) -> str:
    return subprocess.check_output(
        [python, "-c", "import sys; print(sys.base_prefix)"], text=True
    ).strip()


def python_base_executable(python: str) -> str:
    return subprocess.check_output(
        [python, "-c", "import sys; print(sys._base_executable)"], text=True
    ).strip()


def _candidates() -> list[str]:
    if sys.platform == "darwin":
        candidates = [
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12",
            "/opt/homebrew/opt/python@3.12/bin/python3.12",
            shutil.which("python3.12"),
        ]
    elif sys.platform.startswith("linux"):
        candidates = []
        for minor in (12, 11, 10):
            candidates.extend((shutil.which(f"python3.{minor}"), f"/usr/bin/python3.{minor}"))
    else:
        # uv's isolated bootstrap executable is temporary; always select its base Python.
        candidates = [sys._base_executable]
        candidates.extend(shutil.which(name) for name in ("python3.12", "python3.11", "python3.10", "python"))
    return list(dict.fromkeys(str(path) for path in candidates if path and Path(path).is_file()))


def _install_ubuntu_tk() -> bool:
    """Prompt for sudo only if the distribution's supported Python lacks tkinter."""
    if not sys.platform.startswith("linux"):
        return False
    if platform.freedesktop_os_release().get("ID") != "ubuntu":
        return False
    default = "/usr/bin/python3"
    if not Path(default).is_file():
        return False
    version_check = subprocess.run(
        [default, "-c", "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)"],
        capture_output=True,
    )
    if version_check.returncode:
        return False
    check = subprocess.run(
        [default, "-c", "import tkinter"], capture_output=True, text=True
    )
    if check.returncode == 0 or "No module named 'tkinter'" not in check.stderr:
        return False
    print("[AMADEUS] tkinter is missing. Installing Ubuntu python3-tk and libxft2...",
          flush=True)
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    subprocess.run([*prefix, "apt-get", "install", "-y", "python3-tk", "libxft2"], check=True)
    return True


def select_python_runtime(
    requested: str | None,
    existing: Path,
    required_minor: str | None,
) -> str:
    """Preserve existing compatible Python, then try system Python before installing Tk."""
    if requested:
        try:
            check_python_runtime(requested, required_minor)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Selected Python is incompatible: {requested}\n{exc.stderr}"
            ) from exc
        return requested

    if existing.is_file():
        try:
            check_python_runtime(str(existing), required_minor)
            return str(existing)
        except subprocess.CalledProcessError:
            print("[AMADEUS] Existing Python/Tk is incompatible; selecting a supported Python.",
                  flush=True)

    errors = []
    for attempt in range(2):
        for candidate in _candidates():
            try:
                check_python_runtime(candidate, required_minor)
                print(f"[AMADEUS] Selected Python: {candidate}", flush=True)
                return candidate
            except subprocess.CalledProcessError as exc:
                errors.append(f"{candidate}: {(exc.stderr or '').strip()[-220:]}")
        if attempt or not _install_ubuntu_tk():
            break

    if sys.platform == "darwin":
        guidance = "Install native Python 3.12 with Tk 8.6."
    elif sys.platform.startswith("linux"):
        guidance = (
            "Install a Python 3.10-3.12 linked to Xft-enabled Tk 8.6. "
            "Ubuntu provides python3-tk and libxft2 for its default supported Python. "
            "uv's bundled Tk 9 may lack Xft."
        )
    else:
        guidance = "Install Python 3.10-3.12 with tkinter."
    raise RuntimeError(
        "No compatible Python/Tk runtime found. " + guidance + "\n" + "\n".join(errors[-5:])
    )
