"""Launch AMADEUS using the environment verified during setup."""

from __future__ import annotations


def main() -> int:
    from tools.launch_summary import main as print_runtime_summary

    print_runtime_summary()
    from .gui_home import main as gui_home_main
    return gui_home_main()


if __name__ == "__main__":
    raise SystemExit(main())
