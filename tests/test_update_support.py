"""Tests for update locking, process detection, and managed-file state."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from gui.update_support import (
    LOCK_FILENAME,
    acquire_update_lock,
    adopt_update_lock,
    find_other_amadeus_processes,
    release_update_lock,
)


class UpdateSupportTests(unittest.TestCase):
    def test_update_lock_is_exclusive_and_transferable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = acquire_update_lock(root)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                acquire_update_lock(root)

            adopt_update_lock(root, token)
            state = json.loads((root / LOCK_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(state["pid"], os.getpid())
            self.assertEqual(state["token"], token)

            release_update_lock(root, token)
            self.assertFalse((root / LOCK_FILENAME).exists())

    def test_stale_update_lock_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / LOCK_FILENAME).write_text(
                json.dumps({"pid": 2**30, "token": "stale"}),
                encoding="utf-8",
            )
            token = acquire_update_lock(root)
            try:
                state = json.loads((root / LOCK_FILENAME).read_text(encoding="utf-8"))
                self.assertEqual(state["token"], token)
            finally:
                release_update_lock(root, token)

    def test_python_process_running_from_installation_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "amadeus-test-child.py"
            script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
            process = subprocess.Popen([sys.executable, str(script)])
            try:
                matches: list[tuple[int, str]] = []
                for _ in range(40):
                    matches = find_other_amadeus_processes(root)
                    if any(pid == process.pid for pid, _ in matches):
                        break
                    time.sleep(0.05)
                self.assertTrue(any(pid == process.pid for pid, _ in matches), matches)
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
