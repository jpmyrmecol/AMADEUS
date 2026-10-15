# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Run with python -m unittest discover -s tests -p 'test_addon*.py'."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import venv
import zipfile

from gui import addon_manager as m


def manifest(**changes):
    data = {"manifest_version": 1, "id": "org.example.viewer", "name": "Test viewer", "version": "1.0.0",
            "author": "Test author", "license": "0BSD", "platforms": [m.platform_key()], "python": "3.12",
            "entry_point": "app.py", "dependencies": [], "update_url": None}
    data.update(changes)
    return data


def package(directory, data=None, extra=None, prefix="", app="print('test add-on')\n"):
    path = Path(directory) / f"package-{time.time_ns()}.zip"
    files = {"addon.json": json.dumps(data or manifest()), "app.py": app, "LICENSE": "Example license\n"}
    files.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(prefix + name, content)
    return path


class ManifestTests(unittest.TestCase):
    def test_valid_manifest(self):
        self.assertEqual(m.validate_manifest(manifest())["id"], "org.example.viewer")

    def test_invalid_fields(self):
        cases = [{"manifest_version": True}, {"manifest_version": 2}, {"id": "../../escape"}, {"id": "Upper.id"},
                 {"id": "simple"}, {"name": ""}, {"name": "a\nb"}, {"author": 123}, {"license": ""},
                 {"platforms": []}, {"platforms": ["windows"]}, {"platforms": [m.platform_key()] * 2},
                 {"platforms": [None]}, {"python": "system"}, {"entry_point": "../app.py"},
                 {"entry_point": "app.exe"}, {"entry_point": "/app.py"}, {"dependencies": ["--target=/tmp"]},
                 {"dependencies": ["pkg @ https://example.com/x.whl"]}, {"dependencies": ["pkg==*"]},
                 {"update_url": "http://example.com/update.json"}, {"update_url": "https://user:secret@example.com/x"},
                 {"version": "01.0.0"}, {"version": "1.0"}, {"unknown": 1}]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                m.validate_manifest(manifest(**changes))

    def test_exact_pins(self):
        m.validate_manifest(manifest(dependencies=["PyYAML==6.0.2", "package[extra]==1.2.3+cu126"]))

    def test_version_order_is_numeric(self):
        self.assertGreater(m.version_tuple("1.10.0"), m.version_tuple("1.9.0"))

    def test_platform_mapping(self):
        for system, machine, expected in [("win32", "AMD64", "windows-x86_64"), ("win32", "ARM64", "windows-arm64"),
                                           ("darwin", "arm64", "macos-arm64"), ("darwin", "x86_64", "macos-x86_64"),
                                           ("linux", "aarch64", "linux-arm64")]:
            with self.subTest(system=system, machine=machine), patch.object(m.sys, "platform", system), patch.object(m.platform, "machine", return_value=machine):
                self.assertEqual(m.platform_key(), expected)

    def test_storage_locations(self):
        with patch.object(m.sys, "platform", "win32"), patch.dict(os.environ, {"LOCALAPPDATA": "/example/local"}):
            self.assertEqual(m.storage_root(), Path("/example/local/AMADEUS/addons"))
        with patch.object(m.sys, "platform", "darwin"), patch.object(Path, "home", return_value=Path("/example/home")):
            self.assertEqual(m.storage_root(), Path("/example/home/Library/Application Support/AMADEUS/addons"))
        with patch.object(m.sys, "platform", "linux"), patch.dict(os.environ, {"XDG_DATA_HOME": "/example/data"}):
            self.assertEqual(m.storage_root(), Path("/example/data/amadeus/addons"))

    def test_json_duplicate_and_size(self):
        with self.assertRaises(ValueError):
            m._json(b'{"id": "one", "id": "two"}')
        with self.assertRaises(ValueError):
            m._json(b" " * (m.MAX_JSON_BYTES + 1))

    def test_environment_isolation(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/core", "PYTHONHOME": "/core", "VIRTUAL_ENV": "/core/.venv",
                                     "TCL_LIBRARY": "/core/tcl", "TK_LIBRARY": "/core/tk", "UV_INDEX_URL": "https://private",
                                     "AMADEUS_SPLASH_TOKEN": "x", "DISPLAY": ":88"}):
            result = m.clean_environment()
            for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "TCL_LIBRARY", "TK_LIBRARY", "UV_INDEX_URL", "AMADEUS_SPLASH_TOKEN"):
                self.assertNotIn(key, result)
            self.assertEqual(result["DISPLAY"], ":88")


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_root_and_single_wrapper(self):
        for prefix in ("", "repository-main/"):
            checked = m.inspect_package(package(self.root, prefix=prefix))
            self.assertEqual(checked.prefix, prefix)
            self.assertEqual(len(checked.sha256), 64)

    def test_unsafe_paths(self):
        for name in ("../escape", "/absolute", "a/../../escape", "a\\escape", "CON.txt", "folder/NUL", "a:stream",
                     "a/./b", "a//b", "a/trailing.", "a/trailing ", ".venv/bin/python", ".git/config", "__pycache__/a.pyc"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.inspect_package(package(self.root, extra={name: "bad"}))

    def test_case_collision(self):
        with self.assertRaises(ValueError):
            m.inspect_package(package(self.root, extra={"APP.PY": "bad"}))

    def test_symlink(self):
        path = package(self.root)
        with zipfile.ZipFile(path, "a") as archive:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../../escape")
        with self.assertRaises(ValueError):
            m.inspect_package(path)

    def test_special_file(self):
        path = package(self.root)
        with zipfile.ZipFile(path, "a") as archive:
            info = zipfile.ZipInfo("fifo")
            info.create_system = 3
            info.external_attr = (stat.S_IFIFO | 0o600) << 16
            archive.writestr(info, "")
        with self.assertRaises(ValueError):
            m.inspect_package(path)

    def test_expansion_limit(self):
        path = package(self.root)
        with patch.object(m, "MAX_EXPANDED_BYTES", 4), self.assertRaises(ValueError):
            m.inspect_package(path)

    def test_file_count_limit(self):
        path = package(self.root)
        with patch.object(m, "MAX_FILES", 2), self.assertRaises(ValueError):
            m.inspect_package(path)

    def test_compressed_limit(self):
        path = package(self.root)
        with patch.object(m, "MAX_ZIP_BYTES", 4), self.assertRaises(ValueError):
            m.inspect_package(path)

    def test_missing_entry_and_license(self):
        for names in (("addon.json", "app.py"), ("addon.json", "LICENSE")):
            path = self.root / "missing.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for name in names:
                    archive.writestr(name, json.dumps(manifest()) if name == "addon.json" else "")
            with self.assertRaises(ValueError):
                m.inspect_package(path)

    def test_multiple_roots(self):
        with self.assertRaises(ValueError):
            m.inspect_package(package(self.root, extra={"nested/addon.json": json.dumps(manifest())}))

    def test_inspection_never_executes_code(self):
        marker = self.root / "executed"
        checked = m.inspect_package(package(self.root, app=f"open({str(marker)!r}, 'w').write('bad')\n"))
        self.assertFalse(marker.exists())
        self.assertEqual(checked.manifest["name"], "Test viewer")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = m.Store(self.root / "persistent")
        self.uv = Path(sys.executable)  # Environment construction is replaced explicitly in these tests.
        self.prepare = patch.object(m.Store, "_prepare_environment", self.fake_environment)
        self.prepare.start()

    def tearDown(self):
        self.prepare.stop()
        self.temporary.cleanup()

    @staticmethod
    def fake_environment(store, release, data, uv, progress):
        venv.EnvBuilder(with_pip=False).create(release / ".venv")

    def install(self, data=None, **kwargs):
        checked = m.inspect_package(package(self.root, data, **kwargs))
        return self.store.install(checked, self.uv)

    def test_install_update_persistence_and_data(self):
        first = self.install()
        data = self.store.path("org.example.viewer/data/settings.json")
        data.write_text('{"setting": 42}')
        second = self.install(manifest(version="1.1.0"))
        self.assertNotEqual(first.release, second.release)
        self.assertFalse(first.release.exists())
        self.assertEqual(first.installed_at, second.installed_at)
        self.assertEqual(data.read_text(), '{"setting": 42}')
        restored = m.Store(self.store.root).get("org.example.viewer")
        self.assertEqual(restored.manifest["version"], "1.1.0")
        self.assertEqual(restored.release, second.release)
        self.assertEqual(len(self.store.list()[0]), 1)

    def test_reinstall_same_version(self):
        first = self.install()
        second = self.install()
        self.assertNotEqual(first.release, second.release)
        self.assertEqual(second.installed_at, first.installed_at)

    def test_failed_setup_keeps_previous(self):
        first = self.install()
        checked = m.inspect_package(package(self.root, manifest(version="2.0.0")))
        with patch.object(m.Store, "_prepare_environment", side_effect=RuntimeError("setup failed")), self.assertRaises(RuntimeError):
            self.store.install(checked, self.uv)
        self.assertEqual(self.store.get("org.example.viewer").release, first.release)
        self.assertEqual(len(list(first.release.parent.iterdir())), 1)

    def test_failed_first_install_has_no_panel(self):
        checked = m.inspect_package(package(self.root))
        with patch.object(m.Store, "_prepare_environment", side_effect=RuntimeError("setup failed")), self.assertRaises(RuntimeError):
            self.store.install(checked, self.uv)
        self.assertEqual(self.store.list()[0], [])

    def test_downgrade_rejected(self):
        self.install(manifest(version="2.0.0"))
        with self.assertRaises(ValueError):
            self.install(manifest(version="1.0.0"))

    def test_wrong_id_update(self):
        checked = m.inspect_package(package(self.root))
        with self.assertRaises(ValueError):
            self.store.install(checked, self.uv, expected_id="org.other.viewer")

    def test_changed_package_rejected(self):
        checked = m.inspect_package(package(self.root))
        replacement = package(self.root, manifest(version="9.0.0"))
        shutil.copyfile(replacement, checked.path)
        with self.assertRaises(ValueError):
            self.store.install(checked, self.uv)
        self.assertIsNone(self.store.get("org.example.viewer"))

    def test_wrong_os_rejected(self):
        other = next(key for key in m.PLATFORMS if key != m.platform_key())
        with self.assertRaises(ValueError):
            self.install(manifest(platforms=[other]))

    def test_corrupt_record_does_not_hide_other_addons(self):
        self.install()
        broken = self.store.path("org.broken.viewer/current.json")
        broken.parent.mkdir()
        broken.write_text("bad json")
        items, errors = self.store.list()
        self.assertEqual(len(items), 1)
        self.assertEqual(len(errors), 1)

    def test_record_cannot_escape_storage(self):
        record = self.store.path("org.example.viewer/current.json")
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"schema": 1, "release": "../../escape", "installed_at": 0}))
        with self.assertRaises(ValueError):
            self.store.get("org.example.viewer")

    @unittest.skipIf(os.name == "nt", "Creating symlinks may require Windows privileges")
    def test_symlink_storage_child_rejected(self):
        self.store.root.mkdir()
        (self.store.root / "org.example.viewer").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.store.get("org.example.viewer")

    def test_uninstall_preserves_data_and_external_files(self):
        item = self.install()
        data = self.store.path("org.example.viewer/data/settings.json")
        data.write_text("keep")
        external = self.root / "research.csv"
        external.write_text("keep too")
        self.store.uninstall("org.example.viewer")
        self.assertFalse(item.release.exists())
        self.assertIsNone(self.store.get("org.example.viewer"))
        self.assertEqual(data.read_text(), "keep")
        self.assertEqual(external.read_text(), "keep too")

    def test_concurrent_operation_rejected(self):
        checked = m.inspect_package(package(self.root))
        with self.store._guard("org.example.viewer"), self.assertRaises(RuntimeError):
            self.store.install(checked, self.uv)

    def test_launch_has_separate_process_environment_and_file_arguments(self):
        script = '''import json, os, pathlib, sys
pathlib.Path(os.environ['AMADEUS_ADDON_DATA_DIR'], 'report.json').write_text(json.dumps({'prefix':sys.prefix, 'args':sys.argv[1:], 'cwd':os.getcwd(), 'pythonpath':os.environ.get('PYTHONPATH')}))
'''
        item = self.install(app=script)
        config, results = self.root / "config with spaces.yaml", self.root / "results with spaces.csv"
        config.write_text("a: 1")
        results.write_text("frame,id\n1,2\n")
        with patch.dict(os.environ, {"PYTHONPATH": "/core/not-for-addons"}):
            child, log = self.store.launch("org.example.viewer", config=config, results=results)
            self.assertEqual(child.wait(timeout=10), 0)
        record = json.loads(self.store.path("org.example.viewer/data/report.json").read_text())
        self.assertEqual(Path(record["prefix"]), item.release / ".venv")
        self.assertNotEqual(record["prefix"], sys.prefix)
        self.assertEqual(record["args"], ["--config", str(config), "--results", str(results)])
        self.assertIsNone(record["pythonpath"])
        self.assertEqual(Path(record["cwd"]), self.store.path("org.example.viewer/data"))
        self.assertTrue(log.exists())

    def test_running_addon_cannot_be_removed_or_updated(self):
        self.install(app="import time\ntime.sleep(30)\n")
        child, _log = self.store.launch("org.example.viewer")
        try:
            with self.assertRaises(RuntimeError):
                self.store.uninstall("org.example.viewer")
            with self.assertRaises(RuntimeError):
                self.install(manifest(version="1.1.0"))
            with self.assertRaises(RuntimeError):
                self.store.launch("org.example.viewer")
        finally:
            child.terminate()
            child.wait(timeout=10)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.old = m.Installed(manifest(update_url="https://example.org/update.json"), self.root, 0)
        self.archive = package(self.root, manifest(version="2.0.0"))
        self.metadata = {"manifest_version": 1, "id": "org.example.viewer", "version": "2.0.0",
                         "url": "https://example.org/addon.zip", "sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest()}

    def tearDown(self):
        self.temporary.cleanup()

    def fake_download(self, url, destination, limit):
        if url.endswith(".json"):
            destination.write_text(json.dumps(self.metadata))
        else:
            shutil.copyfile(self.archive, destination)

    def test_verified_download_does_not_install(self):
        with patch.object(m, "_download", self.fake_download):
            result = m.download_update(self.old, self.root / "downloads")
        self.assertEqual(result.manifest["version"], "2.0.0")
        self.assertFalse((self.root / "current.json").exists())

    def test_invalid_update_rejected_and_download_removed(self):
        for changes in ({"sha256": "0" * 64}, {"id": "org.wrong.viewer"}, {"version": "3.0.0"}, {"sha256": None}):
            old = self.metadata.copy()
            self.metadata.update(changes)
            with self.subTest(changes=changes), patch.object(m, "_download", self.fake_download), self.assertRaises(ValueError):
                m.download_update(self.old, self.root / "downloads")
            self.assertEqual(list((self.root / "downloads").glob("*.zip")), [])
            self.metadata = old

    def test_up_to_date(self):
        self.metadata["version"] = "1.0.0"
        with patch.object(m, "_download", self.fake_download):
            self.assertIsNone(m.download_update(self.old, self.root / "downloads"))

    def test_http_redirect_rejected(self):
        handler = m._HTTPSRedirect()
        with self.assertRaises(ValueError):
            handler.redirect_request(None, None, 302, "", {}, "http://example.com/addon.zip")


class RuntimeCommandTests(unittest.TestCase):
    def test_uv_targets_external_runtime_and_separate_venv(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = m.Store(Path(temporary) / "persistent")
            release = store.path("org.example.viewer/releases/" + "a" * 32)
            release.mkdir(parents=True)
            calls = []
            def run(command, **kwargs):
                calls.append((command, kwargs))
                executable = store.python(release)
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.touch()
                return subprocess.CompletedProcess(command, 0)
            with patch.object(m.subprocess, "run", side_effect=run):
                store._prepare_environment(release, manifest(dependencies=["PyYAML==6.0.2"]), Path("/uv"), lambda _message: None)
            self.assertIn("--managed-python", calls[0][0])
            self.assertEqual(calls[0][0][-1], str(release / ".venv"))
            self.assertEqual(calls[0][1]["env"]["UV_PYTHON_INSTALL_DIR"], str(store.root / ".python"))
            self.assertIn(str(store.python(release)), calls[1][0])
            self.assertNotIn("PYTHONPATH", calls[1][1]["env"])
            self.assertFalse(calls[1][1].get("shell", False))


if __name__ == "__main__":
    unittest.main()
