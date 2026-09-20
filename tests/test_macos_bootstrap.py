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

    def run_bootstrap(self, setup="", answer="", *, installed_ok=True, install_ok=True):
        # Every candidate and install operation is stubbed; no network or sudo.
        script = HELPER + "\n" + r'''
set -eu
export AMADEUS_VENV="$PWD/env"
unset AMADEUS_PYTHON
printf '#!/bin/sh\n' > compatible
cp compatible incompatible
chmod +x compatible incompatible
amadeus_python_candidates() { printf '%s\n' "$PWD/incompatible"; }
amadeus_check_python() {
    case "$1" in
        */compatible|*/env/bin/python) return 0 ;;
        /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12) INSTALLED_RESULT ;;
        *) return 1 ;;
    esac
}
amadeus_install_python() { echo INSTALL_CALLED; INSTALL_RESULT; }
''' .replace("INSTALLED_RESULT", "return 0" if installed_ok else "return 1").replace(
            "INSTALL_RESULT", "return 0" if install_ok else "return 1")
        script += setup + "\namadeus_ensure_macos_python || exit 7\n"
        script += 'printf "SELECTED=%s\\n" "$AMADEUS_PYTHON"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.sh"
            path.write_text(script, encoding="utf-8", newline="\n")
            env = os.environ.copy()
            for key in ("AMADEUS_PYTHON", "AMADEUS_VENV", "BASH_ENV"):
                env.pop(key, None)
            result = subprocess.run([BASH, "case.sh"], input=answer.encode(),
                                    capture_output=True, cwd=directory, env=env)
            result.stdout = result.stdout.decode()
            result.stderr = result.stderr.decode()
            return result

    def test_compatible_candidate_reused(self):
        result = self.run_bootstrap('amadeus_python_candidates() { printf "%s\\n" "$PWD/incompatible" "$PWD/compatible"; }')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SELECTED=", result.stdout)
        self.assertNotIn("Allow Python", result.stdout)
        self.assertNotIn("INSTALL_CALLED", result.stdout)

    def test_explicit_python_reused(self):
        result = self.run_bootstrap('export AMADEUS_PYTHON="$PWD/compatible"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("INSTALL_CALLED", result.stdout)

    def test_invalid_explicit_python_preserved(self):
        result = self.run_bootstrap('export AMADEUS_PYTHON="$PWD/incompatible"', "y\n")
        self.assertEqual(result.returncode, 7)
        self.assertIn("AMADEUS_PYTHON is incompatible", result.stderr)
        self.assertNotIn("INSTALL_CALLED", result.stdout)

    def test_existing_environment_reused(self):
        result = self.run_bootstrap('mkdir -p env/bin; cp compatible env/bin/python')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/env/bin/python", result.stdout)
        self.assertNotIn("INSTALL_CALLED", result.stdout)

    def test_incomplete_environment_preserved(self):
        result = self.run_bootstrap('mkdir env', "y\n")
        self.assertEqual(result.returncode, 7)
        self.assertIn("preserved", result.stderr)
        self.assertNotIn("INSTALL_CALLED", result.stdout)

    def test_decline_empty_eof_and_unknown_never_install(self):
        for answer in ("n\n", "\n", "", "maybe\n"):
            with self.subTest(answer=answer):
                result = self.run_bootstrap(answer=answer)
                self.assertEqual(result.returncode, 7)
                self.assertNotIn("INSTALL_CALLED", result.stdout)

    def test_consent_installs_and_selects_verified_python(self):
        result = self.run_bootstrap(answer="y\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("INSTALL_CALLED"), 1)
        self.assertIn("SELECTED=/Library/Frameworks/", result.stdout)

    def test_installer_failure_stops(self):
        result = self.run_bootstrap(answer="yes\n", install_ok=False)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("SELECTED=", result.stdout)

    def test_post_install_validation_failure_stops(self):
        result = self.run_bootstrap(answer="y\n", installed_ok=False)
        self.assertEqual(result.returncode, 7)
        self.assertIn("GUI check", result.stderr)
        self.assertNotIn("SELECTED=", result.stdout)

    def test_installer_verifies_before_privilege_escalation(self):
        # Substitute only OS commands; exercise the actual installer function.
        for failure in ("curl", "checksum", "signature", "gatekeeper", "sudo", "none"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                script = HELPER
                replacements = {
                    "/usr/bin/curl": "mock_curl",
                    "/usr/bin/shasum": "mock_shasum",
                    "/usr/sbin/pkgutil": "mock_pkgutil",
                    "/usr/sbin/spctl": "mock_spctl",
                    "/usr/bin/sudo": "mock_sudo",
                }
                for original, replacement in replacements.items():
                    script = script.replace(original, replacement)
                script += r'''
mock_curl() { echo curl; [ "$FAILURE" != curl ]; }
mock_shasum() {
    if [ "$FAILURE" = checksum ]; then echo WRONG;
    else echo '8373e58da4ea146b3eb1c1f9834f19a319440b6b679b06050b1f9ee3237aa8e4  python.pkg'; fi
}
mock_pkgutil() { echo signature; [ "$FAILURE" != signature ]; }
mock_spctl() { echo gatekeeper; [ "$FAILURE" != gatekeeper ]; }
mock_sudo() { echo sudo; [ "$FAILURE" != sudo ]; }
amadeus_install_python
'''
                path = Path(directory) / "case.sh"
                path.write_text(script, encoding="utf-8", newline="\n")
                env = dict(os.environ, FAILURE=failure)
                env.pop("BASH_ENV", None)
                result = subprocess.run([BASH, "case.sh"], cwd=directory,
                                        capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode == 0, failure == "none", result.stderr)
                if failure in ("curl", "checksum", "signature", "gatekeeper"):
                    self.assertNotIn("\nsudo\n", result.stdout)
                if failure == "none":
                    self.assertLess(result.stdout.index("signature"), result.stdout.index("gatekeeper"))
                    self.assertLess(result.stdout.index("gatekeeper"), result.stdout.index("\nsudo\n"))

    def test_site_copy_matches_and_checks_before_download(self):
        site = ROOT.parent / "AMADEUS-site/AMADEUS-Setup.command"
        if not site.exists():
            self.skipTest("Place AMADEUS-site alongside AMADEUS for cross-repository check")
        text = site.read_text(encoding="utf-8")
        embedded = text.split("# BEGIN shared macOS Python bootstrap\n", 1)[1].split(
            "# END shared macOS Python bootstrap", 1)[0]
        self.assertEqual(embedded, HELPER)
        self.assertLess(text.index('amadeus_ensure_macos_python ||'), text.index('mkdir -p'))


if __name__ == "__main__":
    unittest.main()
