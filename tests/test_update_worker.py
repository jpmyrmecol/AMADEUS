"""Tests for archive replacement, obsolete-file removal, and rollback."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from gui.update_support import MANAGED_STATE_FILENAME
from tools.update_worker import _apply_archive, _restore_archive


class UpdateWorkerArchiveTests(unittest.TestCase):
    def test_managed_obsolete_file_is_removed_and_rollback_restores_everything(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            staging = base / "staging"
            temporary = base / "temporary"
            (root / "tools").mkdir(parents=True)
            (staging / "tools").mkdir(parents=True)
            temporary.mkdir()

            (root / "keep.txt").write_text("old", encoding="utf-8")
            (root / "obsolete.txt").write_text("obsolete", encoding="utf-8")
            (root / "user-note.txt").write_text("preserve", encoding="utf-8")
            (root / "tools" / "update_manifest.txt").write_text(
                "keep.txt\nobsolete.txt\ntools/update_manifest.txt\n",
                encoding="utf-8",
            )

            (staging / "keep.txt").write_text("new", encoding="utf-8")
            (staging / "added.txt").write_text("added", encoding="utf-8")
            (staging / "tools" / "update_manifest.txt").write_text(
                "bootstrap", encoding="utf-8"
            )

            records: list[tuple[Path, Path | None]] = []
            _apply_archive(root, staging, temporary, records, "1.2.3")

            self.assertEqual((root / "keep.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse((root / "obsolete.txt").exists())
            self.assertEqual(
                (root / "user-note.txt").read_text(encoding="utf-8"), "preserve"
            )
            self.assertEqual((root / "added.txt").read_text(encoding="utf-8"), "added")

            state = json.loads(
                (root / MANAGED_STATE_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(state["version"], "1.2.3")
            self.assertEqual(
                set(state["files"]),
                {"added.txt", "keep.txt", "tools/update_manifest.txt"},
            )

            _restore_archive(root, records, io.StringIO())

            self.assertEqual((root / "keep.txt").read_text(encoding="utf-8"), "old")
            self.assertEqual(
                (root / "obsolete.txt").read_text(encoding="utf-8"), "obsolete"
            )
            self.assertEqual(
                (root / "user-note.txt").read_text(encoding="utf-8"), "preserve"
            )
            self.assertFalse((root / "added.txt").exists())
            self.assertFalse((root / MANAGED_STATE_FILENAME).exists())

    def test_unknown_user_file_is_preserved_without_previous_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            staging = base / "staging"
            temporary = base / "temporary"
            root.mkdir()
            staging.mkdir()
            temporary.mkdir()

            (root / "user-note.txt").write_text("preserve", encoding="utf-8")
            (staging / "managed.txt").write_text("managed", encoding="utf-8")

            records: list[tuple[Path, Path | None]] = []
            _apply_archive(root, staging, temporary, records, "1.2.3")

            self.assertEqual(
                (root / "user-note.txt").read_text(encoding="utf-8"), "preserve"
            )
            self.assertTrue((root / "managed.txt").is_file())


if __name__ == "__main__":
    unittest.main()
