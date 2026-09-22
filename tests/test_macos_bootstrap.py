"""Exercise consent and failure paths without installing anything on the host."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = (r"C:\Git\bin\bash.exe" if os.name == "nt" else shutil.which("bash"))
HELPER = (ROOT / "tools/macos_python.sh").read_text(encoding="utf-8")


@unittest.skipUnless(BASH and Path(BASH).exists(), "Bash is required")
class BootstrapTests(unittest.TestCase):
    def test_probe_rejects_unsupported_versions_architecture_and_tk(self):
        probe = HELPER.split("<<'PY' >/dev/null 2>&1\n", 1)[1].split("\nPY\n", 1)[0]
        for version, arch, tk_version, runtime, valid in (
            ((3, 10), "arm64", 8.6, "8.6.16", True),
            ((3, 12), "arm64", 8.6, "8.6.16", True),
            ((3, 9), "arm64", 8.6, "8.6.16", False),
            ((3, 13), "arm64", 8.6, "8.6.16", False),
            ((3, 12), "x86_64", 8.6, "8.6.16", False),
            ((3, 12), "arm64", 9.0, "9.0.1", False),
            ((3, 12), "arm64", 8.6, "9.0.1", False),
        ):
            with self.subTest(version=version, arch=arch, tk=tk_version, runtime=runtime):
                setup = f'''
import sys, platform, types
platform.machine = lambda: {arch!r}
sys.platform = 'darwin'
sys.version_info = {version!r}
root = types.SimpleNamespace(withdraw=lambda: None, update=lambda: None,
    destroy=lambda: None, tk=types.SimpleNamespace(call=lambda *args: {runtime!r}))
sys.modules['tkinter'] = types.SimpleNamespace(TclVersion={tk_version},
    TkVersion={tk_version}, Tk=lambda: root)
'''
                result = subprocess.run([sys.executable, "-I", "-c", setup + probe],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, valid, result.stderr)

    def run_bootstrap(self, setup="", *, ui_ok=True, open_ok=True):
        # OS confirmation and package handoff are stubbed; no native UI or downloads.
        script = HELPER.replace("/usr/bin/osascript", "mock_osascript") + "\n" + r'''
set -eu
export AMADEUS_VENV="$PWD/env"
unset AMADEUS_PYTHON
printf '#!/bin/sh\n' > compatible
cp compatible incompatible
chmod +x compatible incompatible
amadeus_python_candidates() { printf '%s\n' "$PWD/incompatible"; }
amadeus_check_python() {
    if [ -e opened ]; then echo CHECK_AFTER_OPEN >&2; fi
    case "$1" in
        */compatible|*/env/bin/python) return 0 ;;
        *) return 1 ;;
    esac
}
mock_osascript() { echo UI_CALLED >&2; UI_RESULT; }
amadeus_open_python_installer() { echo OPEN_CALLED; touch opened; OPEN_RESULT; }
''' .replace("UI_RESULT", "return 0" if ui_ok else "return 1").replace(
            "OPEN_RESULT", "return 0" if open_ok else "return 1")
        script += setup + "\namadeus_ensure_macos_python || exit $?\n"
        script += 'printf "SELECTED=%s\\n" "$AMADEUS_PYTHON"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.sh"
            path.write_text(script, encoding="utf-8", newline="\n")
            env = os.environ.copy()
            for key in ("AMADEUS_PYTHON", "AMADEUS_VENV", "BASH_ENV"):
                env.pop(key, None)
            result = subprocess.run([BASH, "case.sh"], input=b"",
                                    capture_output=True, cwd=directory, env=env)
            result.stdout = result.stdout.decode()
            result.stderr = result.stderr.decode()
            return result

    def test_compatible_candidate_reused(self):
        result = self.run_bootstrap('amadeus_python_candidates() { printf "%s\\n" "$PWD/incompatible" "$PWD/compatible"; }')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SELECTED=", result.stdout)
        self.assertNotIn("UI_CALLED", result.stderr)
        self.assertNotIn("OPEN_CALLED", result.stdout)

    def test_explicit_python_reused(self):
        result = self.run_bootstrap('export AMADEUS_PYTHON="$PWD/compatible"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("OPEN_CALLED", result.stdout)

    def test_invalid_explicit_python_preserved(self):
        result = self.run_bootstrap('export AMADEUS_PYTHON="$PWD/incompatible"')
        self.assertEqual(result.returncode, 1)
        self.assertIn("AMADEUS_PYTHON is incompatible", result.stderr)
        self.assertNotIn("OPEN_CALLED", result.stdout)

    def test_existing_environment_reused(self):
        result = self.run_bootstrap('mkdir -p env/bin; cp compatible env/bin/python')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/env/bin/python", result.stdout)
        self.assertNotIn("OPEN_CALLED", result.stdout)

    def test_incomplete_environment_preserved(self):
        result = self.run_bootstrap('mkdir env')
        self.assertEqual(result.returncode, 1)
        self.assertIn("preserved", result.stderr)
        self.assertNotIn("OPEN_CALLED", result.stdout)

    def test_ui_cancel_or_failure_never_downloads(self):
        result = self.run_bootstrap(ui_ok=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("UI_CALLED", result.stderr)
        self.assertNotIn("OPEN_CALLED", result.stdout)
        self.assertNotIn("SELECTED=", result.stdout)

    def test_approval_hands_off_without_rechecking_or_continuing(self):
        result = self.run_bootstrap()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout.count("OPEN_CALLED"), 1)
        self.assertNotIn("CHECK_AFTER_OPEN", result.stderr)
        self.assertNotIn("SELECTED=", result.stdout)

    def test_handoff_failure_stops(self):
        result = self.run_bootstrap(open_ok=False)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("SELECTED=", result.stdout)

    def test_package_verification_handoff_and_lifetime(self):
        # Substitute only OS commands; exercise the actual installer function.
        for failure in ("curl", "checksum", "signature", "gatekeeper", "open", "none"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                script = HELPER
                replacements = {
                    "/usr/bin/curl": "mock_curl",
                    "/usr/bin/shasum": "mock_shasum",
                    "/usr/sbin/pkgutil": "mock_pkgutil",
                    "/usr/sbin/spctl": "mock_spctl",
                    "/usr/bin/open": "mock_open",
                }
                for original, replacement in replacements.items():
                    script = script.replace(original, replacement)
                script += r'''
mktemp() { mkdir package; printf '%s/package\n' "$PWD"; }
mock_curl() { echo curl; touch "$pkg"; [ "$FAILURE" != curl ]; }
mock_shasum() {
    if [ "$FAILURE" = checksum ]; then echo WRONG;
    else echo '8373e58da4ea146b3eb1c1f9834f19a319440b6b679b06050b1f9ee3237aa8e4  python.pkg'; fi
}
mock_pkgutil() { echo signature; [ "$FAILURE" != signature ]; }
mock_spctl() { echo gatekeeper; [ "$FAILURE" != gatekeeper ]; }
mock_open() {
    echo open
    [ "$#" -eq 3 ] && [ "$1" = -a ] &&
        [ "$2" = /System/Library/CoreServices/Installer.app ] && [ -f "$3" ] || return 1
    [ "$FAILURE" != open ]
}
amadeus_open_python_installer
'''
                path = Path(directory) / "case.sh"
                path.write_text(script, encoding="utf-8", newline="\n")
                env = dict(os.environ, FAILURE=failure)
                env.pop("BASH_ENV", None)
                result = subprocess.run([BASH, "case.sh"], cwd=directory,
                                        capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode == 0, failure == "none", result.stderr)
                if failure in ("curl", "checksum", "signature", "gatekeeper"):
                    self.assertNotIn("\nopen\n", result.stdout)
                self.assertEqual((Path(directory) / "package/python.pkg").exists(), failure == "none")
                if failure == "none":
                    self.assertLess(result.stdout.index("signature"), result.stdout.index("gatekeeper"))
                    self.assertLess(result.stdout.index("gatekeeper"), result.stdout.index("\nopen\n"))
                    self.assertIn("then run AMADEUS Setup again", result.stdout)

    def test_launchers_stop_normally_after_handoff(self):
        # AMADEUS.sh marks where the macOS Python handoff ends and uv setup
        # begins; splitting there keeps this test off the uv logic.
        launchers = [(ROOT / "AMADEUS.sh", '\n# --- pinned uv (tests split AMADEUS.sh at this marker) ---')]
        site = ROOT.parent / "AMADEUS-site/AMADEUS-Setup.command"
        if site.exists():
            launchers.append((site, '\nmkdir -p "$(dirname "$app")"'))
        for path, boundary in launchers:
            for status in (0, 1, 2):
                with self.subTest(launcher=path.name, status=status), tempfile.TemporaryDirectory() as directory:
                    stub = f"amadeus_ensure_macos_python() {{ return {status}; }}\n"
                    script = path.read_text(encoding="utf-8").split(boundary, 1)[0]
                    if path == site:
                        script = script.replace(HELPER, stub)
                    else:
                        helper = Path(directory) / "tools/macos_python.sh"
                        helper.parent.mkdir()
                        helper.write_text(stub, encoding="utf-8")
                    script = 'uname() { case "$1" in -s) echo Darwin;; -m) echo arm64;; esac; }\n' + script
                    script += '\necho CONTINUED\n'
                    (Path(directory) / "launch.sh").write_text(script, encoding="utf-8", newline="\n")
                    env = os.environ.copy()
                    env.pop("BASH_ENV", None)
                    result = subprocess.run([BASH, "launch.sh"], cwd=directory,
                                            input="", capture_output=True, text=True, env=env)
                    self.assertEqual(result.returncode, 1 if status == 1 else 0, result.stderr)
                    self.assertEqual("CONTINUED" in result.stdout, status == 0)
                    if status == 2:
                        self.assertNotIn("ERROR", result.stderr)

    def test_site_copy_matches_and_checks_before_download(self):
        site = ROOT.parent / "AMADEUS-site/AMADEUS-Setup.command"
        if not site.exists():
            self.skipTest("Place AMADEUS-site alongside AMADEUS for cross-repository check")
        text = site.read_text(encoding="utf-8")
        embedded = text.split("# BEGIN shared macOS Python bootstrap\n", 1)[1].split(
            "# END shared macOS Python bootstrap", 1)[0]
        self.assertEqual(embedded, HELPER)
        self.assertLess(text.index('if amadeus_ensure_macos_python; then'), text.index('mkdir -p'))


if __name__ == "__main__":
    unittest.main()
