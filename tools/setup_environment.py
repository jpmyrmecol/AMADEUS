# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "gui"))
from splash_ipc import signal_stop  # noqa: E402

TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
NUMPY_VERSION = "1.26.4"
SCIPY_VERSION = "1.11.4"
PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
ENVIRONMENT_ROOT = Path(os.environ.get("AMADEUS_VENV", PROJECT_ROOT / ".venv")).expanduser().absolute()
READY_MARKER = ENVIRONMENT_ROOT / ".amadeus-ready"
# uv resolves uv.lock, so the executable version is pinned separately in
# [tool.uv].required-version. The launchers read the same setting and hand over
# a matching uv; this check keeps a hand-run setup honest too.
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"


def nvidia_smi_candidates() -> list[str]:
    """Return nvidia-smi locations, including common Windows driver paths."""
    candidates: list[Path | str] = []
    discovered = shutil.which("nvidia-smi")
    if discovered:
        candidates.append(discovered)

    if os.name == "nt":
        windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        program_files = Path(
            os.environ.get("ProgramW6432", os.environ.get("ProgramFiles", r"C:\Program Files"))
        )
        candidates.extend(
            [
                windows_dir / "System32" / "nvidia-smi.exe",
                program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            ]
        )
        driver_store = windows_dir / "System32" / "DriverStore" / "FileRepository"
        if driver_store.is_dir():
            candidates.extend(driver_store.glob("nv*/*nvidia-smi.exe"))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = str(candidate)
        key = os.path.normcase(os.path.abspath(value))
        if key not in seen and Path(value).is_file():
            seen.add(key)
            unique.append(value)
    return unique


def nvidia_gpu_is_available() -> bool:
    """Return whether a working NVIDIA GPU is reported by the driver."""
    for executable in nvidia_smi_candidates():
        try:
            result = subprocess.run(
                [executable],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
            )
        except OSError:
            continue
        if result.returncode == 0:
            return True
    return False


def _is_apple_silicon() -> bool:
    import platform

    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def select_torch_profile() -> tuple[str, str]:
    """Select CUDA when NVIDIA is available; macOS uses wheels with MPS support."""
    if _is_apple_silicon():
        return "macos", "Apple Silicon detected; using macOS wheels (MPS/Metal and CPU)."
    if not nvidia_gpu_is_available():
        return "cpu", "No usable NVIDIA GPU was detected; using CPU wheels."
    return "cu128", "An NVIDIA GPU was detected; using CUDA 12.8 wheels."


def venv_python() -> Path:
    return ENVIRONMENT_ROOT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def sync_environment(uv_executable: str, profile: str, python: str) -> None:
    """Sync locked dependencies while preserving the installed PyTorch pair.

    ``--no-install-package`` prevents uv from selecting torch/torchvision from
    the lockfile, while ``--inexact`` is essential here: uv sync is exact by
    default and would otherwise remove the already-installed CUDA wheels as
    extraneous packages. The pair is verified and repaired separately below.
    """
    command = [
        uv_executable,
        "sync",
        "--locked",
        "--inexact",
        "--python",
        python,
        "--no-dev",
        "--extra",
        profile,
        "--no-install-package",
        "torch",
        "--no-install-package",
        "torchvision",
    ]
    environment = os.environ.copy()
    environment["UV_PROJECT_ENVIRONMENT"] = str(ENVIRONMENT_ROOT)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def check_python(python: str, *, macos: bool) -> None:
    """Reject unsupported Python/Tk before uv can replace an existing environment."""
    code = """
import sys
if not (3, 10) <= sys.version_info[:2] < (3, 13):
    raise RuntimeError('Python 3.10–3.12 required')
"""
    if macos:
        code += """
import platform
import tkinter as tk
if platform.machine() not in ('arm64', 'aarch64'):
    raise RuntimeError('Native Apple Silicon Python required')
if tk.TkVersion != 8.6 or tk.TclVersion != 8.6:
    raise RuntimeError('Tcl/Tk 8.6 required with CustomTkinter 5.2.2')
root = tk.Tk()
root.withdraw()
root.update()
root.destroy()
"""
    subprocess.run([python, "-c", code], check=True, capture_output=True, text=True)


def select_python(requested: str | None) -> str:
    macos = sys.platform == "darwin"
    if requested:
        try:
            check_python(requested, macos=macos)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Selected Python is incompatible: {requested}\n{exc.stderr}") from exc
        # uv must not silently replace a different existing environment.
        if venv_python().exists():
            code = "import sys; print(sys.base_prefix)"
            existing = subprocess.check_output([str(venv_python()), "-c", code], text=True).strip()
            selected = subprocess.check_output([requested, "-c", code], text=True).strip()
            if existing != selected:
                raise RuntimeError(
                    "The selected Python differs from the existing environment. "
                    "Set AMADEUS_VENV to a new directory (for example .venv-tk86); "
                    "the existing environment has been preserved."
                )
        return requested
    if venv_python().exists():
        try:
            check_python(str(venv_python()), macos=macos)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Existing environment is incompatible: {venv_python()}\n{exc.stderr}\n"
                "Select a Python 3.10–3.12 with Tcl/Tk 8.6 using AMADEUS_PYTHON "
                "and set AMADEUS_VENV=.venv-tk86 to preserve the old environment."
            ) from exc
        return str(venv_python())
    if macos:
        # Python.org framework installations are candidates, not an assumed Tk guarantee.
        for minor in (12, 11, 10):
            candidate = f"/Library/Frameworks/Python.framework/Versions/3.{minor}/bin/python3.{minor}"
            if Path(candidate).is_file():
                try:
                    check_python(candidate, macos=True)
                    return candidate
                except subprocess.CalledProcessError:
                    continue
        raise RuntimeError(
            "macOS requires native Python 3.10–3.12 with Tcl/Tk 8.6. "
            "Run AMADEUS.command (or AMADEUS-Setup.command) to check Python and offer installation. "
            "Tk 9 is not supported with CustomTkinter 5.2.2. See README macOS setup."
        )
    return "3.10"


def verify_gui_environment() -> None:
    """Exercise the dropdown that failed in the macOS report, not just Tk()."""
    code = """
import sys
import tkinter as tk
import customtkinter as ctk
import cv2
print('[AMADEUS] Python:', sys.version, sys.executable)
print('[AMADEUS] Tcl/Tk:', tk.TclVersion, tk.TkVersion)
if sys.platform == 'darwin' and (tk.TclVersion != 8.6 or tk.TkVersion != 8.6):
    raise RuntimeError('macOS requires Tcl/Tk 8.6 with CustomTkinter 5.2.2')
app = ctk.CTk()
app.withdraw()
try:
    combo = ctk.CTkComboBox(app, values=['Color', 'Grayscale'])
    combo.pack()
    combo.configure(values=['Grayscale', 'Color'])
    option = ctk.CTkOptionMenu(app, values=['A', 'B'])
    option.pack()
    app.update()
    print('[AMADEUS] CustomTkinter dropdown verification: OK')
finally:
    app.destroy()
"""
    subprocess.run([str(venv_python()), "-c", code], cwd=PROJECT_ROOT, check=True)


def prepare_ffmpeg() -> None:
    """Install and verify the one pinned FFmpeg used by every video feature."""
    print("[AMADEUS] Preparing the pinned FFmpeg build (one-time setup)...", flush=True)
    code = """
from tools.ffmpeg_runtime import ensure_ffmpeg, ffmpeg_build_identity
try:
    executable = ensure_ffmpeg()
except Exception as exc:
    raise SystemExit(f"[ERROR] Could not prepare AMADEUS FFmpeg: {exc}") from exc
print(f"[AMADEUS] FFmpeg ready ({ffmpeg_build_identity()}): {executable}", flush=True)
"""
    subprocess.run([str(venv_python()), "-c", code], cwd=PROJECT_ROOT, check=True)


def verify_numeric_stack() -> None:
    """Verify the NumPy/SciPy versions and SciPy binary compatibility."""
    python = venv_python()
    code = f'''
import numpy
import scipy
from scipy.ndimage import gaussian_filter1d

print("[AMADEUS] NumPy:", numpy.__version__)
print("[AMADEUS] SciPy:", scipy.__version__)

if numpy.__version__ != {NUMPY_VERSION!r}:
    raise RuntimeError(f"Expected NumPy {NUMPY_VERSION}, found {{numpy.__version__}}")
if scipy.__version__ != {SCIPY_VERSION!r}:
    raise RuntimeError(f"Expected SciPy {SCIPY_VERSION}, found {{scipy.__version__}}")

x = numpy.asarray([0.0, 1.0, 0.0], dtype=numpy.float64)
y = gaussian_filter1d(x, 1.0)
if y.shape != x.shape or not numpy.isfinite(y).all():
    raise RuntimeError("SciPy ndimage verification returned an unexpected result")

print("[AMADEUS] NumPy/SciPy binary compatibility: OK")
'''
    result = subprocess.run([str(python), "-c", code], cwd=PROJECT_ROOT, check=False)
    if result.returncode:
        raise RuntimeError("NumPy/SciPy environment verification failed")


def verify_pytorch_profile(profile: str) -> None:
    """Verify torch, torchvision, CUDA computation, and CUDA NMS when required."""
    python = venv_python()
    code = f'''
import torch
import torchvision

profile = {profile!r}
print("[AMADEUS] PyTorch:", torch.__version__)
print("[AMADEUS] torchvision:", torchvision.__version__)
print("[AMADEUS] PyTorch CUDA runtime:", torch.version.cuda or "CPU")
print("[AMADEUS] GPU available:", torch.cuda.is_available())

if torch.__version__.split("+")[0] != {TORCH_VERSION!r}:
    raise RuntimeError(f"Expected torch {TORCH_VERSION}, found {{torch.__version__}}")
if torchvision.__version__.split("+")[0] != {TORCHVISION_VERSION!r}:
    raise RuntimeError(
        f"Expected torchvision {TORCHVISION_VERSION}, found {{torchvision.__version__}}"
    )

if profile == "cu128":
    if "+cu128" not in torch.__version__:
        raise RuntimeError(f"Expected torch +cu128, found {{torch.__version__}}")
    import platform
    arm_linux = platform.system() == "Linux" and platform.machine() in ("aarch64", "arm64")
    if not arm_linux and "+cu128" not in torchvision.__version__:
        raise RuntimeError(f"Expected torchvision +cu128, found {{torchvision.__version__}}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected CUDA runtime 12.8, found {{torch.version.cuda}}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA verification failed: PyTorch cannot access an NVIDIA GPU")

    print("[AMADEUS] GPU:", torch.cuda.get_device_name(0))
    print("[AMADEUS] CUDA architectures:", torch.cuda.get_arch_list())

    x = torch.ones(1, device="cuda")
    y = x + 1
    torch.cuda.synchronize()
    if y.item() != 2:
        raise RuntimeError("CUDA computation returned an unexpected result")
    print("[AMADEUS] CUDA computation verification: OK")

    from torchvision.ops import nms

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]],
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8], device="cuda")
    keep = nms(boxes, scores, 0.5)
    torch.cuda.synchronize()
    if keep.numel() < 1:
        raise RuntimeError("torchvision CUDA NMS returned an unexpected result")
    print("[AMADEUS] torchvision CUDA NMS verification: OK")
else:
    if torch.version.cuda is not None:
        raise RuntimeError(
            f"Expected a CPU PyTorch build, found CUDA runtime {{torch.version.cuda}}"
        )
    print("[AMADEUS]", "macOS PyTorch verification: OK" if profile == "macos" else "CPU PyTorch verification: OK")

    import platform

    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        from main.batch_utils import _mps_available
        if _mps_available():
            print("[AMADEUS] MPS computation verification: OK")
        else:
            print("[AMADEUS] MPS cannot execute a test operation; automatic device selection will use CPU.")
'''
    result = subprocess.run([str(python), "-c", code], cwd=PROJECT_ROOT, check=False)
    if result.returncode:
        raise RuntimeError("PyTorch/torchvision environment verification failed")


def install_exact_pytorch_profile(uv_executable: str, profile: str) -> None:
    """Install only torch and torchvision, without modifying their dependencies."""
    python = venv_python()

    if profile == "cu128":
        torch_spec = f"torch=={TORCH_VERSION}+cu128"
        torchvision_spec = f"torchvision=={TORCHVISION_VERSION}+cu128"
        index = PYTORCH_CU128_INDEX
    elif sys.platform == "darwin":
        # PyTorch macOS wheels do not use the +cpu local-version suffix.
        torch_spec = f"torch=={TORCH_VERSION}"
        torchvision_spec = f"torchvision=={TORCHVISION_VERSION}"
        index = None
    else:
        torch_spec = f"torch=={TORCH_VERSION}+cpu"
        torchvision_spec = f"torchvision=={TORCHVISION_VERSION}+cpu"
        index = PYTORCH_CPU_INDEX

    if sys.platform.startswith("linux"):
        import platform

        if platform.machine() in ("aarch64", "arm64"):
            torchvision_spec = f"torchvision=={TORCHVISION_VERSION}"

    command = [
        uv_executable,
        "pip",
        "install",
        "--python",
        str(python),
        "--reinstall",
        "--no-deps",
    ]
    if index:
        command.extend(["--index", index])
    else:
        command.extend(["--default-index", "https://pypi.org/simple"])
    command.extend([torch_spec, torchvision_spec])

    print(
        f"[AMADEUS] Installing exact PyTorch pair without changing other dependencies: "
        f"{torch_spec}, {torchvision_spec}",
        flush=True,
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_pytorch_profile(uv_executable: str, profile: str) -> None:
    """Keep a valid existing pair; repair only torch/torchvision when necessary."""
    try:
        verify_pytorch_profile(profile)
        return
    except RuntimeError:
        print(
            "[AMADEUS] PyTorch/torchvision profile is missing or incorrect; repairing only those two packages...",
            flush=True,
        )

    install_exact_pytorch_profile(uv_executable, profile)
    verify_pytorch_profile(profile)


def _broadcast_windows_environment_change() -> None:
    import ctypes

    result = ctypes.c_size_t()
    ctypes.windll.user32.SendMessageTimeoutW(
        0xFFFF,  # HWND_BROADCAST
        0x001A,  # WM_SETTINGCHANGE
        0,
        "Environment",
        0x0002,  # SMTO_ABORTIFHUNG
        5000,
        ctypes.byref(result),
    )


def _add_to_windows_user_path(directory: Path) -> None:
    import winreg

    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER,
        "Environment",
        0,
        winreg.KEY_READ | winreg.KEY_SET_VALUE,
    ) as key:
        try:
            current, value_type = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current, value_type = "", winreg.REG_EXPAND_SZ

        target = os.path.normcase(os.path.abspath(str(directory)))
        entries = [entry.strip() for entry in str(current).split(";") if entry.strip()]
        normalized = {
            os.path.normcase(os.path.abspath(os.path.expandvars(entry))) for entry in entries
        }
        if target not in normalized:
            entries.append(str(directory))
            winreg.SetValueEx(key, "Path", 0, value_type, ";".join(entries) + ";")
            _broadcast_windows_environment_change()


def install_amadeus_command() -> Path:
    """Install small per-user commands that start this checkout's environment.

    Two names are installed side by side -- "amadeus" and the short alias
    "amade" -- so either one launches AMADEUS from any terminal. The alias is
    a thin forwarder to the primary command, not a copy of its logic.
    """
    if os.name == "nt":
        local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        command_dir = local_app_data / "AMADEUS" / "bin"
        command_dir.mkdir(parents=True, exist_ok=True)
        project_root = str(PROJECT_ROOT).replace("%", "%%")
        command_path = command_dir / "amadeus.cmd"
        command_path.write_text(
            "@echo off\n"
            f'set "AMADEUS_ROOT={project_root}"\n'
            'if not exist "%AMADEUS_ROOT%\\.venv\\Scripts\\activate.bat" (\n'
            "    echo [ERROR] AMADEUS is not installed at %AMADEUS_ROOT%.\n"
            "    echo Run AMADEUS.bat in the AMADEUS folder to repair the installation.\n"
            "    exit /b 1\n"
            ")\n"
            'cd /d "%AMADEUS_ROOT%"\n'
            'call "%AMADEUS_ROOT%\\.venv\\Scripts\\activate.bat"\n'
            "if errorlevel 1 exit /b 1\n"
            'if defined CUDA_VISIBLE_DEVICES "%AMADEUS_ROOT%\\.venv\\Scripts\\python.exe" -c "import torch; print(\'[AMADEUS] Selected GPU:\', torch.cuda.get_device_name(0)); print(\'[AMADEUS] GPU utilization:\', torch.cuda.utilization(0), \'%%\')"\n'
            'set "AMADEUS_SPLASH_TOKEN=%RANDOM%%RANDOM%%RANDOM%"\n'
            'start "AMADEUS Splash" /min "%AMADEUS_ROOT%\\.venv\\Scripts\\python.exe" "%AMADEUS_ROOT%\\gui\\splash_standalone.py" "%AMADEUS_SPLASH_TOKEN%" "4"\n'
            '"%AMADEUS_ROOT%\\.venv\\Scripts\\amadeus.exe" %*\n'
            "if errorlevel 1 exit /b 1\n"
            "echo [AMADEUS] The GUI has closed. The AMADEUS virtual environment is active.\n"
            'echo [AMADEUS] Type "amadeus" or "amade" for the home GUI.\n',
            encoding="utf-8",
        )
        alias_target = str(command_path).replace("%", "%%")
        alias_path = command_dir / "amade.cmd"
        alias_path.write_text(
            f'@echo off\ncall "{alias_target}" %*\nexit /b %errorlevel%\n', encoding="utf-8"
        )
        _add_to_windows_user_path(command_dir)
        return command_path

    command_dir = Path.home() / ".local" / "bin"
    command_dir.mkdir(parents=True, exist_ok=True)
    command_path = command_dir / "amadeus"
    command_path.write_text(
        "#!/usr/bin/env sh\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        f"exec {shlex.quote(str(ENVIRONMENT_ROOT / 'bin' / 'amadeus'))} \"$@\"\n",
        encoding="utf-8",
    )
    command_path.chmod(0o755)

    alias_path = command_dir / "amade"
    alias_path.write_text(
        f"#!/usr/bin/env sh\nexec {shlex.quote(str(command_path))} \"$@\"\n",
        encoding="utf-8",
    )
    alias_path.chmod(0o755)
    return command_path


def required_uv_version() -> str:
    """Return the exact uv version pinned in pyproject.toml."""
    try:
        lines = PYPROJECT_FILE.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Could not read {PYPROJECT_FILE}: {exc}") from exc

    in_tool_uv = False
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_tool_uv = line == "[tool.uv]"
            continue
        if not in_tool_uv:
            continue
        key, separator, value = line.partition("=")
        if key.strip() != "required-version" or not separator:
            continue
        specifier = value.strip()
        if not (specifier.startswith('"==') and specifier.endswith('"')):
            break
        version = specifier[3:-1].strip()
        if version:
            return version
        break

    raise RuntimeError(
        "[tool.uv].required-version in pyproject.toml must be an exact == version pin."
    )


def uv_version(uv_executable: str) -> str:
    """Return the version an uv executable reports, or "" if it cannot be asked."""
    try:
        completed = subprocess.run(
            [uv_executable, "--version"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    # "uv 1.2.3 (abc1234 2026-01-01)" -> "1.2.3"
    fields = (completed.stdout or "").strip().splitlines()
    fields = fields[0].split() if fields else []
    return fields[1] if len(fields) >= 2 else ""


def check_uv_version(uv_executable: str) -> str:
    """Refuse to sync with a uv other than the pinned one."""
    required = required_uv_version()
    found = uv_version(uv_executable)
    if found == required:
        return required
    reported = found or "no version"
    raise RuntimeError(
        f"AMADEUS is pinned to uv {required}, but {uv_executable} reports {reported}.\n"
        "Run the AMADEUS launcher (AMADEUS.bat, AMADEUS.command or AMADEUS.sh) instead "
        "of calling this script directly: it installs the pinned uv into the AMADEUS "
        "folder without touching any uv you already have."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", required=True, help="Path to the uv executable.")
    parser.add_argument("--python", default=os.environ.get("AMADEUS_PYTHON"),
                        help="Python executable used to create the application environment.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile, explanation = select_torch_profile()
    print(f"[AMADEUS] {explanation}", flush=True)
    print("[AMADEUS] Preparing the locked Python environment...", flush=True)

    # AMADEUS.bat uses this marker for its fast path. Remove it before making
    # any changes so an interrupted or failed setup is fully checked next time.
    READY_MARKER.unlink(missing_ok=True)

    try:
        if os.name == "nt" and ENVIRONMENT_ROOT != PROJECT_ROOT / ".venv":
            raise RuntimeError("AMADEUS_VENV is supported by the macOS/Linux launcher only.")
        uv_version_in_use = check_uv_version(args.uv)
        print(f"[AMADEUS] Using uv {uv_version_in_use}: {args.uv}", flush=True)
        python = select_python(args.python)
        sync_environment(args.uv, profile, python)
        verify_numeric_stack()
        ensure_pytorch_profile(args.uv, profile)
        # Re-check after any PyTorch repair to ensure no unrelated package changed.
        verify_numeric_stack()
        verify_gui_environment()
        prepare_ffmpeg()
        command_path = install_amadeus_command()
        READY_MARKER.touch()
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        signal_stop()
        return getattr(exc, "returncode", 1) or 1

    print(f"[AMADEUS] Command installed: {command_path}", flush=True)
    print('[AMADEUS] Open a new terminal and type: amadeus (or the short alias: amade)', flush=True)
    print("[AMADEUS] Environment is ready. Starting the GUI...", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
