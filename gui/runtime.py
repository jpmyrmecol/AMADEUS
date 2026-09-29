"""Launch AMADEUS after preparing Linux Tk's Xft font runtime when required."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


_BOOTSTRAPPED_ENV = "AMADEUS_TK_XFT_BOOTSTRAPPED"


def _runtime_paths(app_dir: Path) -> tuple[Path, Path, Path] | None:
    runtime_dir = app_dir / ".uv" / "tk-font-runtime"
    script_dir = runtime_dir / "usr/lib/tk9.0"
    if not script_dir.is_dir():
        return None

    for library in sorted((runtime_dir / "usr/lib").glob("*/libtcl9tk9.0.so")):
        if library.is_file():
            return library, library.parent, script_dir
    return None


def _is_configured(library: Path, library_dir: Path, script_dir: Path, tcl_dir: Path) -> bool:
    preload = re.split(r"[:\s]+", os.environ.get("LD_PRELOAD", ""))
    search_path = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    return (
        str(library) in preload
        and str(library_dir) in search_path
        and str(tcl_dir) in search_path
        and os.environ.get("TK_LIBRARY") == str(script_dir)
    )


def _prepend_path(value: str, *paths: Path) -> str:
    items = [str(path) for path in paths]
    items.extend(item for item in value.split(os.pathsep) if item)
    return os.pathsep.join(dict.fromkeys(items))


def _relaunch_command() -> list[str]:
    entry = Path(sys.argv[0])
    if entry.is_file() and entry.resolve() != Path(__file__).resolve():
        return [sys.executable, str(entry), *sys.argv[1:]]
    return [sys.executable, "-m", "gui.runtime", *sys.argv[1:]]


def _activate_linux_tk_runtime() -> None:
    if os.environ.get(_BOOTSTRAPPED_ENV) == "1":
        return

    app_dir = Path(__file__).resolve().parents[1]
    paths = _runtime_paths(app_dir)
    helper = app_dir / "tools" / "prepare_linux_tk_xft.py"

    if paths is None and helper.is_file():
        try:
            subprocess.run([sys.executable, str(helper)], cwd=app_dir, check=False)
        except OSError as exc:
            print(f"[AMADEUS] Could not start the Linux Tk font check: {exc}", file=sys.stderr)
        paths = _runtime_paths(app_dir)

    if paths is None:
        return

    library, library_dir, script_dir = paths
    tcl_dir = Path(sys.base_prefix) / "lib"
    if _is_configured(library, library_dir, script_dir, tcl_dir):
        return

    env = os.environ.copy()
    env["TK_LIBRARY"] = str(script_dir)
    env["LD_LIBRARY_PATH"] = _prepend_path(env.get("LD_LIBRARY_PATH", ""), library_dir, tcl_dir)
    preload = [str(library)]
    preload.extend(
        item
        for item in re.split(r"[:\s]+", env.get("LD_PRELOAD", ""))
        if item and item != str(library)
    )
    env["LD_PRELOAD"] = os.pathsep.join(preload)
    env[_BOOTSTRAPPED_ENV] = "1"
    os.execve(sys.executable, _relaunch_command(), env)


def main() -> int:
    if sys.platform.startswith("linux"):
        _activate_linux_tk_runtime()

    from .gui_home import main as gui_home_main

    return gui_home_main()


if __name__ == "__main__":
    raise SystemExit(main())

