"""Launch AMADEUS using the environment verified during setup."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    # Probe the GPU in a short-lived process rather than retaining a CUDA/MPS
    # context in the Home GUI process.
    subprocess.run(
        [sys.executable, "-m", "tools.launch_summary"],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
    )
    from .gui_home import main as gui_home_main
    print("[AMADEUS] Home GUI module loaded.", flush=True)
    return gui_home_main()


if __name__ == "__main__":
    raise SystemExit(main())
