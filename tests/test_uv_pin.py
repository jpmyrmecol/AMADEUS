# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""uv is pinned like any other dependency, because it resolves uv.lock.

UV_VERSION is the single source of truth. These tests keep every launcher, the
Colab notebook and tools/setup_environment.py reading it rather than embedding a
version of their own, and exercise the POSIX launcher's resolution logic --
including the case the pin exists for: a different uv already on the machine.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UV_VERSION_FILE = ROOT / "UV_VERSION"
PINNED = UV_VERSION_FILE.read_text(encoding="utf-8").strip()
BASH = shutil.which("bash")


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class PinnedVersionFileTests(unittest.TestCase):
    def test_the_pin_is_a_plain_version_on_one_line(self):
        raw = UV_VERSION_FILE.read_text(encoding="utf-8")
        self.assertRegex(PINNED, r"^\d+\.\d+\.\d+$")
        self.assertEqual(raw.splitlines()[0].strip(), PINNED)
        self.assertEqual(len([line for line in raw.splitlines() if line.strip()]), 1)

    def test_it_ships_with_the_colab_runtime_package(self):
        # The notebook and the runner both refuse a package without it, so the
        # builder has to include it.
        from main import colab_utils

        self.assertIn("UV_VERSION", colab_utils._RUNTIME_REQUIRED_FILES)
        self.assertIn("'UV_VERSION'", read("colab/AMADEUS_Colab.ipynb"))
        self.assertIn('"UV_VERSION"', read("tools/colab_runner.py"))


class NoSecondSourceOfTruthTests(unittest.TestCase):
    """Nothing may name a uv version of its own; drift would be silent."""

    FILES = (
        "AMADEUS.sh",
        "AMADEUS.bat",
        "AMADEUS.command",
        "tools/uv_bootstrap.ps1",
        "tools/setup_environment.py",
        "colab/AMADEUS_Colab.ipynb",
    )
    def test_the_pinned_version_appears_in_exactly_one_file(self):
        """A copy of the pin anywhere else is a second source of truth."""
        for relative in self.FILES:
            with self.subTest(file=relative):
                self.assertNotIn(
                    PINNED,
                    read(relative),
                    f"{relative} repeats the pinned version instead of reading UV_VERSION",
                )

    def test_every_installer_url_carries_the_pinned_version(self):
        for relative, expected in (
            ("AMADEUS.sh", "https://astral.sh/uv/$UV_VERSION/install.sh"),
            ("tools/uv_bootstrap.ps1", 'https://astral.sh/uv/$required/install.ps1'),
        ):
            with self.subTest(file=relative):
                text = read(relative)
                self.assertIn(expected, text)
                # The unversioned URL always installs the newest release.
                self.assertNotIn("astral.sh/uv/install.", text)

    def test_the_installers_are_told_not_to_touch_the_user_path(self):
        for relative in ("AMADEUS.sh", "tools/uv_bootstrap.ps1"):
            with self.subTest(file=relative):
                text = read(relative)
                self.assertIn("INSTALLER_NO_MODIFY_PATH", text)
                self.assertIn("UV_INSTALL_DIR", text)

    def test_the_windows_launcher_delegates_to_the_bootstrap_script(self):
        text = read("AMADEUS.bat")
        self.assertIn("uv_bootstrap.ps1", text)
        # The old unconditional "use whatever uv is on PATH" path must be gone.
        self.assertNotIn("where uv.exe", text)

    def test_the_macos_launcher_reuses_the_shared_one(self):
        self.assertIn("AMADEUS.sh", read("AMADEUS.command"))

    def test_colab_pins_the_same_version_it_then_verifies(self):
        cell = "".join(
            "".join(cell["source"])
            for cell in json.loads(read("colab/AMADEUS_Colab.ipynb"))["cells"]
            if "pip" in "".join(cell["source"])
        )
        self.assertIn("f'uv=={UV_VERSION}'", cell)
        self.assertIn("if reported != UV_VERSION:", cell)


class SetupEnvironmentCheckTests(unittest.TestCase):
    """tools/setup_environment.py refuses to sync with the wrong uv."""

    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault("splash_ipc", type(sys)("splash_ipc")).signal_stop = lambda: None
        sys.path.insert(0, str(ROOT / "tools"))
        import setup_environment

        cls.setup_environment = setup_environment
        cls.tmp = tempfile.mkdtemp(prefix="amadeus-uv-pin-")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fake_uv(self, name: str, reported: str) -> str:
        path = os.path.join(self.tmp, name)
        Path(path).write_text(f'#!/bin/sh\necho "uv {reported} (abc1234 2026-01-01)"\n', encoding="utf-8")
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return path

    @unittest.skipIf(os.name == "nt", "the fake uv is a shell script")
    def test_the_pinned_version_is_accepted(self):
        uv = self._fake_uv("uv-good", PINNED)
        self.assertEqual(self.setup_environment.uv_version(uv), PINNED)
        self.assertEqual(self.setup_environment.check_uv_version(uv), PINNED)

    @unittest.skipIf(os.name == "nt", "the fake uv is a shell script")
    def test_another_version_is_refused_with_advice(self):
        uv = self._fake_uv("uv-old", "0.1.0")
        with self.assertRaises(RuntimeError) as caught:
            self.setup_environment.check_uv_version(uv)
        message = str(caught.exception)
        self.assertIn(PINNED, message)
        self.assertIn("0.1.0", message)
        self.assertIn("launcher", message)

    def test_a_missing_uv_is_refused_rather_than_ignored(self):
        with self.assertRaises(RuntimeError):
            self.setup_environment.check_uv_version(os.path.join(self.tmp, "does-not-exist"))

    def test_the_check_reads_the_same_file_the_launchers_do(self):
        self.assertEqual(self.setup_environment.required_uv_version(), PINNED)
        self.assertEqual(self.setup_environment.UV_VERSION_FILE, UV_VERSION_FILE)


@unittest.skipUnless(BASH, "bash is required")
class PosixLauncherResolutionTests(unittest.TestCase):
    """Run the launcher's uv-resolution block against stubbed uv binaries."""

    @classmethod
    def setUpClass(cls):
        launcher = read("AMADEUS.sh")
        start = launcher.index('# --- pinned uv (tests split AMADEUS.sh at this marker) ---')
        end_marker = 'echo "[AMADEUS] Using uv $UV_VERSION: $UV_EXE"'
        end = launcher.index(end_marker) + len(end_marker)
        cls.block = 'set -euo pipefail\nSCRIPT_DIR="$1"\n' + launcher[start:end] + "\n"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="amadeus-uv-launcher-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = os.path.join(self.tmp, "amadeus")
        os.makedirs(os.path.join(self.root, ".uv"))
        shutil.copy(UV_VERSION_FILE, os.path.join(self.root, "UV_VERSION"))
        self.stub_bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.stub_bin)
        self.script = os.path.join(self.tmp, "block.sh")
        Path(self.script).write_text(self.block, encoding="utf-8")

    def _stub(self, directory: str, name: str, body: str) -> str:
        path = os.path.join(directory, name)
        Path(path).write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return path

    def _uv_stub(self, directory: str, name: str, reported: str) -> str:
        return self._stub(directory, name, f'echo "uv {reported} (abc1234 2026-01-01)"')

    def _run(self, *, extra_path: str = "") -> subprocess.CompletedProcess:
        env = dict(os.environ)
        if extra_path:
            env["PATH"] = extra_path + os.pathsep + env.get("PATH", "")
        env["HOME"] = self.tmp  # keep ~/.local/bin and ~/.cargo/bin out of the way
        return subprocess.run(
            [BASH, self.script, self.root],
            capture_output=True, text=True, check=False, env=env,
        )

    def test_the_amadeus_copy_is_used(self):
        expected = self._uv_stub(os.path.join(self.root, ".uv"), "uv", PINNED)
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(expected, result.stdout)

    def test_a_matching_uv_already_installed_is_reused_without_downloading(self):
        expected = self._uv_stub(self.stub_bin, "uv", PINNED)
        # curl and wget fail, so any download attempt would be visible.
        self._stub(self.stub_bin, "curl", "exit 1")
        self._stub(self.stub_bin, "wget", "exit 1")
        result = self._run(extra_path=self.stub_bin)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(expected, result.stdout)
        self.assertNotIn("Installing uv", result.stdout)

    def test_a_different_uv_on_the_machine_is_not_used(self):
        """The whole point of the pin: uv 0.1.0 on PATH must not resolve uv.lock."""
        other = self._uv_stub(self.stub_bin, "uv", "0.1.0")
        self._stub(self.stub_bin, "curl", "exit 1")
        self._stub(self.stub_bin, "wget", "exit 1")
        result = self._run(extra_path=self.stub_bin)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(other, result.stdout)
        combined = result.stdout + result.stderr
        self.assertIn(f"Installing uv {PINNED}", combined)
        self.assertIn("left untouched", combined)

    def test_a_missing_version_file_is_reported_clearly(self):
        os.remove(os.path.join(self.root, "UV_VERSION"))
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UV_VERSION is missing", result.stdout + result.stderr)

    def test_without_curl_or_wget_the_failure_names_the_pinned_version(self):
        offline = os.path.join(self.tmp, "offline")
        os.makedirs(offline)
        self._stub(offline, "curl", "exit 127")
        # A stub that exists but is not on PATH as "curl"/"wget" is what we want;
        # here neither is reachable because the block checks command -v first.
        result = subprocess.run(
            [BASH, "-c",
             f'PATH="{os.path.dirname(BASH)}:/usr/bin:/bin"; '
             f'export PATH; exec "{BASH}" "{self.script}" "{self.root}"'],
            capture_output=True, text=True, check=False,
            env={**os.environ, "HOME": self.tmp},
        )
        # Either it reports the missing downloader or it fails to reach the network;
        # what matters is that it never silently accepts a different uv.
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(PINNED, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
