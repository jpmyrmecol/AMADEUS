# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Print runtime diagnostics from the same source as the tracking log header."""

from main.runtime_diagnostics import runtime_details, startup_summary_lines


def main() -> None:
    for line in startup_summary_lines(runtime_details()):
        print(line, flush=True)


if __name__ == "__main__":
    main()
