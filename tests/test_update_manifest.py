"""Keep the bootstrap updater manifest synchronized with tracked repository files."""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "tools" / "update_manifest.txt"


class UpdateManifestTests(unittest.TestCase):
    def test_bootstrap_manifest_matches_git_tracked_files(self) -> None:
        if shutil.which("git") is None or not (PROJECT_ROOT / ".git").exists():
            self.skipTest("Git checkout is required for the tracked-file manifest check.")

        tracked = {
            line.strip()
            for line in subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "ls-files"],
                text=True,
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        }
        manifest = {
            line.strip()
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        self.assertEqual(manifest, tracked)


if __name__ == "__main__":
    unittest.main()
