# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Regression tests for the Linux Add-ons storage path and migration."""

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from gui import addon_manager


class LinuxAddonStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name) / "home"
        self.home.mkdir()
        self.data_home = Path(temporary.name) / "xdg"
        self.env_patch = patch.dict(os.environ, {"XDG_DATA_HOME": str(self.data_home)})
        self.home_patch = patch.object(addon_manager.Path, "home", return_value=self.home)
        self.platform_patch = patch.object(addon_manager.sys, "platform", "linux")
        for active_patch in (self.env_patch, self.home_patch, self.platform_patch):
            active_patch.start()
            self.addCleanup(active_patch.stop)

    @property
    def legacy(self):
        return self.data_home / "amadeus" / "addons"

    @property
    def unified(self):
        return self.data_home / "AMADEUS" / "addons"

    def test_absolute_xdg_data_home(self):
        self.assertEqual(addon_manager.storage_root(), self.unified)
        self.assertEqual(addon_manager.legacy_linux_storage_root(), self.legacy)

    def test_relative_xdg_data_home_uses_default(self):
        with patch.dict(os.environ, {"XDG_DATA_HOME": "relative/share"}):
            self.assertEqual(
                addon_manager.storage_root(),
                self.home / ".local" / "share" / "AMADEUS" / "addons",
            )

    def test_empty_xdg_data_home_uses_default(self):
        with patch.dict(os.environ, {"XDG_DATA_HOME": ""}):
            self.assertEqual(
                addon_manager.storage_root(),
                self.home / ".local" / "share" / "AMADEUS" / "addons",
            )

    def test_preserves_addon_data_order_and_legacy_absolute_paths(self):
        addon_data = self.legacy / "example.addon" / "data"
        addon_data.mkdir(parents=True)
        (addon_data / "settings.json").write_text('{"setting": 42}', encoding="utf-8")
        (self.legacy / ".order.json").write_text(
            '{"schema": 1, "ids": ["example.addon"]}', encoding="utf-8"
        )
        store = addon_manager.Store()

        self.assertEqual(store.root, self.unified)
        self.assertTrue(self.legacy.is_symlink())
        self.assertEqual(self.legacy.resolve(), self.unified)
        self.assertEqual(
            (self.unified / "example.addon" / "data" / "settings.json").read_text(),
            '{"setting": 42}',
        )
        self.assertEqual(
            (self.legacy / ".order.json").read_text(),
            (self.unified / ".order.json").read_text(),
        )
        self.assertEqual(addon_manager.Store().root, self.unified)

    def test_reuses_empty_unified_directory(self):
        self.legacy.mkdir(parents=True)
        self.unified.mkdir(parents=True)
        addon_manager.Store()
        self.assertTrue(self.legacy.is_symlink())
        self.assertTrue(self.unified.is_dir())

    def test_does_not_overwrite_two_populated_storage_directories(self):
        self.legacy.mkdir(parents=True)
        (self.legacy / "legacy-marker").write_text("legacy")
        self.unified.mkdir(parents=True)
        (self.unified / "new-marker").write_text("new")
        with self.assertRaisesRegex(ValueError, "Both legacy and new"):
            addon_manager.Store()
        self.assertEqual((self.legacy / "legacy-marker").read_text(), "legacy")
        self.assertEqual((self.unified / "new-marker").read_text(), "new")
        self.assertFalse(self.legacy.is_symlink())

    def test_refuses_unexpected_legacy_symlink(self):
        self.legacy.parent.mkdir(parents=True)
        other = self.data_home / "unrelated"
        other.mkdir()
        self.legacy.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Unexpected legacy"):
            addon_manager.Store()

    def test_refuses_migration_while_addon_process_is_running(self):
        script = self.legacy / "example.addon" / "releases" / "version" / "payload" / "main.py"
        script.parent.mkdir(parents=True)
        script.write_text("pass\n")
        running = SimpleNamespace(info={"cmdline": ["python", str(script)]})
        with patch.object(addon_manager.psutil, "process_iter", return_value=[running]):
            with self.assertRaisesRegex(RuntimeError, "Close running add-ons"):
                addon_manager.Store()
        self.assertTrue(self.legacy.is_dir())
        self.assertFalse(self.legacy.is_symlink())
        self.assertFalse(self.unified.exists())


if __name__ == "__main__":
    unittest.main()
