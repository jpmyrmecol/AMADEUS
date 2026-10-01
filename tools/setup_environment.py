# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gui"))
from tools.runtime_profiles import PROFILES, check_support
from main.compute_telemetry import query_nvidia_driver_info
from splash_ipc import signal_stop  # noqa: E402

NUMPY_VERSION = "1.26.4"
SCIPY_VERSION = "1.11.4"
ENVIRONMENT_ROOT = Path(os.environ.get("AMADEUS_VENV", PROJECT_ROOT / ".venv")).expanduser().absolute()
READY_MARKER = ENVIRONMENT_ROOT / ".amadeus-ready"
# uv resolves uv.lock, so the executable version is pinned separately in
# [tool.uv].required-version. The launchers read the same setting and hand over
# a matching uv; this check keeps a hand-run setup honest too.
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"


def nvidia_gpu_is_available() -> bool:
    """Return whether a working NVIDIA GPU is reported by the driver."""
    return query_nvidia_driver_info() is not None


def _nvidia_driver_diagnostic() -> str:
    info = query_nvidia_driver_info()
    if info is None:
        return (
            "nvidia-smi could not report an NVIDIA GPU. On WSL2, AMADEUS also checks "
            "/usr/lib/wsl/lib/nvidia-smi even when that directory is not on PATH."
        )
    name, driver, executable = info
    return f"NVIDIA GPU: {name}; driver: {driver}; nvidia-smi: {executable}"


def _nvidia_compute_capability() -> tuple[int, int] | None:
    """Return GPU0 compute capability using the NVIDIA driver tool."""
    info = query_nvidia_driver_info()
    if info is None:
        return None
    _, _, executable = info
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first = (result.stdout or "").splitlines()
    if not first:
        return None
    try:
        major, minor = first[0].strip().split(".", 1)
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


def _nvidia_driver_major() -> int | None:
    info = query_nvidia_driver_info()
    if info is None:
        return None
    _, driver, _ = info
    try:
        return int(str(driver).split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _is_apple_silicon() -> bool:
    import platform

    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def amd_runtime_candidate() -> bool:
    """Driver hint only, NOT a hardware support verdict; verify real ops later."""
    import platform
    if not (sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"}):
        return False
    if not Path("/dev/kfd").exists():
        return False
    for vendor in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            if vendor.read_text().strip().lower() == "0x1002":
                return True
        except OSError:
            pass
    return False


def windows_amd_gpu_candidate() -> bool:
    """Detect an AMD display adapter before installing the Windows ROCm profile."""
    if sys.platform != "win32":
        return False
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty PNPDeviceID"],
            capture_output=True, text=True, errors="replace", timeout=8, check=False,
        )
        return result.returncode == 0 and "VEN_1002" in result.stdout.upper()
    except (OSError, subprocess.TimeoutExpired):
        return False


def select_torch_profiles() -> tuple[list[str], str]:
    """Return ordered accelerator profiles, newest compatible candidate first."""
    requested = os.environ.get("AMADEUS_TORCH_PROFILE", "auto").strip().lower()
    if requested not in {"", "auto"}:
        check_support(requested)
        return [requested], (
            f"Requested PyTorch profile: {requested}; automatic fallback is disabled "
            "for an explicit profile."
        )
    if sys.platform == "darwin":
        return ["macos"], "Using macOS wheels (MPS/Metal when available, otherwise CPU)."
    if nvidia_gpu_is_available():
        capability = _nvidia_compute_capability()
        driver_major = _nvidia_driver_major()
        if capability is not None and capability < (7, 5):
            return ["cu126"], (
                f"NVIDIA GPU0 compute capability is {capability[0]}.{capability[1]}; "
                "using the legacy CUDA 12.6 profile retained for pre-Turing GPUs. "
                + _nvidia_driver_diagnostic()
            )
        if (
            capability is not None
            and capability < (10, 0)
            and driver_major is not None
            and driver_major < 580
        ):
            return ["cu126"], (
                f"NVIDIA GPU0 compute capability is {capability[0]}.{capability[1]} "
                f"but driver {driver_major}.x is below the CUDA 13.x compatibility floor; "
                "using CUDA 12.6. "
                + _nvidia_driver_diagnostic()
            )
        capability_text = (
            f"GPU0 compute capability {capability[0]}.{capability[1]}"
            if capability is not None
            else "GPU0 compute capability could not be queried"
        )
        if capability is not None and capability >= (10, 0):
            return ["cu132"], (
                f"An NVIDIA Blackwell-class GPU was detected ({capability_text}); "
                "using CUDA 13.2 because the CUDA 12.6 PyTorch build has no compatible "
                "sm_100/sm_120 kernels. "
                + _nvidia_driver_diagnostic()
            )
        return ["cu132", "cu126"], (
            f"An NVIDIA GPU was detected ({capability_text}); trying CUDA 13.2 first. "
            "CUDA 12.6 is retained as an automatic compatibility fallback. "
            + _nvidia_driver_diagnostic()
        )
    if amd_runtime_candidate():
        return ["rocm100", "rocm72"], (
            "AMD GPU runtime detected on Linux; trying AMD ROCm 10.0 / PyTorch 2.13 first. "
            "If real GPU computation or torchvision NMS is incompatible, AMADEUS will "
            "retry the PyTorch ROCm 7.2 profile."
        )
    if windows_amd_gpu_candidate():
        return ["rocmwin100"], (
            "AMD GPU detected on Windows; using AMD ROCm 10.0 / PyTorch 2.13 wheels. "
            "A supported Windows 11 GPU/driver is required and GPU computation plus NMS must pass."
        )
    return ["cpu"], "No accelerator runtime candidate detected; using CPU wheels."


def select_torch_profile() -> tuple[str, str]:
    """Compatibility helper returning the preferred profile only."""
    profiles, explanation = select_torch_profiles()
    return profiles[0], explanation


def venv_python() -> Path:
    return ENVIRONMENT_ROOT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def sync_environment(uv_executable: str, profile: str, python: str) -> None:
    """Sync locked dependencies while preserving the separately managed PyTorch pair.

    ``--no-install-package`` prevents uv from selecting torch/torchvision from
    the lockfile, while ``--inexact`` is essential here: uv sync is exact by
    default and would otherwise remove the already-installed CUDA wheels as
    extraneous packages. The pair is verified and repaired separately below.
    """
    selected = PROFILES[profile]
    sync_extra = selected.sync_extra or profile
    command = [
        uv_executable,
        "sync",
        "--locked",
        "--inexact",
        "--python",
        python,
        "--no-dev",
        "--extra",
        sync_extra,
        "--no-install-package",
        "torch",
        "--no-install-package",
        "torchvision",
    ]
    environment = os.environ.copy()
    environment["UV_PROJECT_ENVIRONMENT"] = str(ENVIRONMENT_ROOT)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def check_python(python: str, *, macos: bool, required_minor: str | None = None) -> None:
    """Reject unsupported Python/Tk before uv can replace an existing environment."""
    code = """
import sys
if not (3, 10) <= sys.version_info[:2] < (3, 13):
    raise RuntimeError('Python 3.10–3.12 required')
"""
    if required_minor:
        code += f"""
if f'{{sys.version_info.major}}.{{sys.version_info.minor}}' != {required_minor!r}:
    raise RuntimeError('Python {required_minor} required for this accelerator profile')
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


def select_python(requested: str | None, profile: str) -> str:
    macos = sys.platform == "darwin"
    required_minor = PROFILES[profile].python_minor
    if requested:
        try:
            check_python(requested, macos=macos, required_minor=required_minor)
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
            check_python(str(venv_python()), macos=macos, required_minor=required_minor)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Existing environment is incompatible: {venv_python()}\n{exc.stderr}\n"
                "Select a Python 3.10–3.12 with Tcl/Tk 8.6 using AMADEUS_PYTHON "
                "and set AMADEUS_VENV=.venv-tk86 to preserve the old environment."
            ) from exc
        return str(venv_python())
    if required_minor:
        return required_minor
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


def _ready_marker_profile() -> str | None:
    try:
        payload = json.loads(READY_MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    profile = payload.get("profile") if isinstance(payload, dict) else None
    return profile if isinstance(profile, str) else None


def prepare_profile_environment(profile: str, previous_profile: str | None = None) -> None:
    """Rebuild the managed venv when a profile requires another Python/runtime stack."""
    if ENVIRONMENT_ROOT != PROJECT_ROOT / ".venv":
        return

    if previous_profile is None:
        previous_profile = _ready_marker_profile()
    rebuild = False
    if previous_profile is not None and previous_profile != profile:
        previous = PROFILES.get(previous_profile)
        current = PROFILES[profile]
        rebuild = bool(
            (previous and previous.install_dependencies)
            or current.install_dependencies
        )

    required_minor = PROFILES[profile].python_minor
    if not rebuild and required_minor and venv_python().exists():
        try:
            running_minor = subprocess.check_output(
                [
                    str(venv_python()),
                    "-c",
                    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
                ],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            running_minor = ""
        rebuild = running_minor != required_minor

    if rebuild and ENVIRONMENT_ROOT.exists():
        print(
            f"[AMADEUS] Rebuilding the managed environment for PyTorch profile {profile}...",
            flush=True,
        )
        shutil.rmtree(ENVIRONMENT_ROOT)


def verify_gui_environment() -> None:
    """Exercise CustomTkinter dropdowns under the supported Tk runtime."""
    code = """
import sys
import tkinter as tk
import customtkinter as ctk
import cv2
from gui.tk_compat import apply_tk_compatibility
apply_tk_compatibility()
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
    """Verify the selected wheel build and accelerator computation/NMS when required."""
    check_support(profile)
    code = f"""
import torch
import torchvision
from tools.runtime_profiles import verify_execution
from main.compute_backend import detect_backend
verify_execution({profile!r}, torch, torchvision)
print("[AMADEUS] PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print("[AMADEUS] Runtime backend:", detect_backend().value)
print("[AMADEUS] PyTorch profile verification: OK")
"""
    result = subprocess.run(
        [str(venv_python()), "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 3000:
            detail = detail[-3000:]
        lines = ["PyTorch/torchvision environment verification failed."]
        if profile in {"cu132", "cu126"}:
            lines.append(_nvidia_driver_diagnostic())
            capability = _nvidia_compute_capability()
            if capability is not None:
                lines.append(
                    f"NVIDIA GPU0 compute capability: {capability[0]}.{capability[1]}."
                )
            if profile == "cu132":
                lines.append(
                    "CUDA 13.2 is the primary profile for Turing (CC 7.5) and newer GPUs. "
                    "It requires a CUDA 13-compatible NVIDIA driver."
                )
            else:
                lines.append(
                    "CUDA 12.6 is the legacy profile retained for Maxwell, Pascal and Volta. "
                    "It does not provide Blackwell sm_100/sm_120 kernels."
                )
        elif profile in {"rocm100", "rocmwin100"}:
            lines.append(
                "The AMD ROCm 10.0 profile requires supported AMD hardware and a compatible "
                "driver/runtime. AMADEUS verifies real GPU computation and torchvision NMS "
                "before accepting it."
            )
        elif profile == "rocm72":
            lines.append(
                "The ROCm 7.2 compatibility profile requires a supported Linux AMD GPU and "
                "driver. Real GPU computation and torchvision NMS must both pass."
            )
        if detail:
            lines.append("Verifier output:\n" + detail)
        raise RuntimeError("\n".join(lines))


def install_exact_pytorch_profile(uv_executable: str, profile: str) -> None:
    """Install the exact torch/torchvision pair required by the selected profile."""
    python = venv_python()

    check_support(profile)
    selected = PROFILES[profile]
    suffix = f"+{selected.suffix}" if selected.suffix and sys.platform != "darwin" else ""
    package_extra = f"[{selected.package_extra}]" if selected.package_extra else ""
    torch_spec = f"torch{package_extra}=={selected.torch_version}{suffix}"
    torchvision_spec = f"torchvision{package_extra}=={selected.torchvision_version}{suffix}"
    index = selected.index if sys.platform != "darwin" else None

    if sys.platform.startswith("linux"):
        import platform

        if platform.machine() in ("aarch64", "arm64"):
            torchvision_spec = f"torchvision=={selected.torchvision_version}"

    command = [
        uv_executable,
        "pip",
        "install",
        "--python",
        str(python),
    ]
    if selected.install_dependencies:
        # Keep already-satisfied locked AMADEUS dependencies intact while the
        # AMD device-all extras add their required ROCm runtime packages.
        pass
    else:
        command.extend(["--reinstall", "--no-deps"])
    if index:
        command.extend(["--index", index])
    else:
        command.extend(["--default-index", "https://pypi.org/simple"])
    command.extend([torch_spec, torchvision_spec])

    dependency_note = (
        " with AMD device runtime dependencies"
        if selected.install_dependencies
        else " without changing other dependencies"
    )
    print(
        f"[AMADEUS] Installing exact PyTorch pair{dependency_note}: "
        f"{torch_spec}, {torchvision_spec}",
        flush=True,
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def ensure_pytorch_profile(uv_executable: str, profile: str) -> None:
    """Keep a valid existing pair; repair the selected accelerator profile when necessary."""
    try:
        verify_pytorch_profile(profile)
        return
    except RuntimeError:
        print(
            "[AMADEUS] PyTorch/torchvision profile is missing or incorrect; repairing the accelerator profile...",
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
            'set "AMADEUS_LAUNCH_CWD=%CD%"\n'
            f'set "AMADEUS_ROOT={project_root}"\n'
            'if not exist "%AMADEUS_ROOT%\\.venv\\Scripts\\activate.bat" (\n'
            "    echo [ERROR] AMADEUS is not installed at %AMADEUS_ROOT%.\n"
            "    echo Run AMADEUS.bat in the AMADEUS folder to repair the installation.\n"
            "    exit /b 1\n"
            ")\n"
            'cd /d "%AMADEUS_ROOT%"\n'
            'call "%AMADEUS_ROOT%\\.venv\\Scripts\\activate.bat"\n'
            "if errorlevel 1 exit /b 1\n"
            'if defined CUDA_VISIBLE_DEVICES "%AMADEUS_ROOT%\\.venv\\Scripts\\python.exe" -c "from main.compute_telemetry import print_device_summary; print_device_summary()"\n'
            'set "AMADEUS_SPLASH_TOKEN=%RANDOM%%RANDOM%%RANDOM%"\n'
            'start "AMADEUS Splash" /min "%AMADEUS_ROOT%\\.venv\\Scripts\\python.exe" "%AMADEUS_ROOT%\\gui\\splash_standalone.py" "%AMADEUS_SPLASH_TOKEN%" "4"\n'
            '"%AMADEUS_ROOT%\\.venv\\Scripts\\amadeus.exe" %*\n'
            'set "AMADEUS_GUI_EXIT=%ERRORLEVEL%"\n'
            'if "%AMADEUS_GUI_EXIT%"=="42" (\n'
            '    cd /d "%AMADEUS_LAUNCH_CWD%"\n'
            "    echo [AMADEUS] Update accepted. The updater will restart AMADEUS automatically.\n"
            "    exit /b 0\n"
            ")\n"
            'if "%AMADEUS_GUI_EXIT%"=="43" (\n'
            '    cd /d "%AMADEUS_LAUNCH_CWD%"\n'
            "    echo [AMADEUS] Uninstall accepted. A separate Command Prompt will remove AMADEUS.\n"
            "    exit /b 0\n"
            ")\n"
            'if not "%AMADEUS_GUI_EXIT%"=="0" exit /b %AMADEUS_GUI_EXIT%\n'
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
        f"{shlex.quote(str(ENVIRONMENT_ROOT / 'bin' / 'amadeus'))} \"$@\"\n"
        "status=$?\n"
        "if [ \"$status\" -eq 42 ] || [ \"$status\" -eq 43 ]; then\n"
        "    exit 0\n"
        "fi\n"
        "exit \"$status\"\n",
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


def ready_identity(profile: str) -> dict:
    inputs = ("pyproject.toml", "uv.lock", "VERSION", "tools/setup_environment.py",
              "tools/runtime_profiles.py", "main/compute_backend.py", "gui/tk_compat.py")
    digest = hashlib.sha256()
    for name in inputs:
        digest.update((PROJECT_ROOT / name).read_bytes())
    return {"schema": 1, "profile": profile, "inputs": digest.hexdigest()}


def environment_ready() -> bool:
    """Legacy empty markers are invalid. Verify installed wheels on every fast path."""
    try:
        profiles, _ = select_torch_profiles()
        marker = json.loads(READY_MARKER.read_text())
        profile = marker.get("profile") if isinstance(marker, dict) else None
        if profile not in profiles:
            return False
        if marker != ready_identity(profile):
            return False
        verify_pytorch_profile(profile)
        return True
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-ready", action="store_true")
    parser.add_argument("--uv", help="Path to the uv executable.")
    parser.add_argument("--python", default=os.environ.get("AMADEUS_PYTHON"),
                        help="Python executable used to create the application environment.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_ready:
        return 0 if environment_ready() else 1
    if not args.uv:
        raise RuntimeError("--uv is required for setup")
    profiles, explanation = select_torch_profiles()
    print(f"[AMADEUS] {explanation}", flush=True)
    print("[AMADEUS] Preparing the locked Python environment...", flush=True)

    try:
        if os.name == "nt" and ENVIRONMENT_ROOT != PROJECT_ROOT / ".venv":
            raise RuntimeError("AMADEUS_VENV is supported by the macOS/Linux launcher only.")
        previous_profile = _ready_marker_profile()
        # AMADEUS.bat uses this marker for its fast path. Remove it before making
        # any changes so an interrupted or failed setup is fully checked next time.
        READY_MARKER.unlink(missing_ok=True)
        uv_version_in_use = check_uv_version(args.uv)
        print(f"[AMADEUS] Using uv {uv_version_in_use}: {args.uv}", flush=True)

        profile = ""
        last_error: Exception | None = None
        for index, candidate in enumerate(profiles):
            try:
                if index:
                    print(
                        f"[AMADEUS] Trying compatibility profile {candidate}...",
                        flush=True,
                    )
                prepare_profile_environment(candidate, previous_profile)
                python = select_python(args.python, candidate)
                sync_environment(args.uv, candidate, python)
                verify_numeric_stack()
                ensure_pytorch_profile(args.uv, candidate)
                # Re-check after any PyTorch repair to ensure no unrelated package changed.
                verify_numeric_stack()
                profile = candidate
                break
            except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                last_error = exc
                if index + 1 >= len(profiles):
                    raise
                print(
                    f"[AMADEUS] PyTorch profile {candidate} was not usable: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"[AMADEUS] Falling back to {profiles[index + 1]}.",
                    flush=True,
                )
                previous_profile = candidate

        if not profile:
            raise RuntimeError(f"No usable PyTorch profile was found: {last_error}")

        verify_gui_environment()
        prepare_ffmpeg()
        command_path = install_amadeus_command()
        READY_MARKER.write_text(json.dumps(ready_identity(profile)), encoding="utf-8")
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
