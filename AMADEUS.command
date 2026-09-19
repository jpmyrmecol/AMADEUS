#!/usr/bin/env bash
# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
# Open the shared launcher from Finder or Terminal.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +e
bash "$SCRIPT_DIR/AMADEUS.sh" "$@"
status=$?
set -e
if (( status != 0 )); then
  echo "AMADEUS failed with exit code ${status}. See the run's log.txt for the full diagnostic log."
  if [[ -t 0 && -t 1 ]]; then
    read -r -p "Press Enter to close..."
  fi
fi
exit "$status"
