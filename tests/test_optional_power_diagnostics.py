"""Test optional power telemetry without importing PyTorch or opening a GUI."""
import ast
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil


def load_system_details():
    path = Path(__file__).resolve().parents[1] / 'main' / 'runtime_diagnostics.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == 'system_details')
    namespace = dict(os=os, platform=platform, subprocess=subprocess, sys=sys)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['system_details']


class OptionalPowerDiagnosticsTest(unittest.TestCase):
    def test_missing_or_inaccessible_power_telemetry_is_unknown(self):
        for error in (FileNotFoundError('no power_supply sysfs'),
                      PermissionError('power_supply inaccessible'),
                      psutil.AccessDenied(), NotImplementedError()):
            with self.subTest(error=type(error).__name__):
                with patch.object(psutil, 'sensors_battery', side_effect=error):
                    details = load_system_details()()
                self.assertEqual(details['power_source'], 'unknown')
                self.assertGreater(float(details['ram_total_gib']), 0)
                self.assertIn('cpu_cores', details)

    def test_available_power_states_are_preserved(self):
        for battery, expected in ((None, 'unknown'),
                                  (SimpleNamespace(power_plugged=True), 'AC'),
                                  (SimpleNamespace(power_plugged=False), 'battery')):
            with self.subTest(expected=expected):
                with patch.object(psutil, 'sensors_battery', return_value=battery):
                    self.assertEqual(load_system_details()()['power_source'], expected)

    def test_unrelated_errors_are_not_suppressed(self):
        with patch.object(psutil, 'sensors_battery', side_effect=ValueError('unexpected')):
            with self.assertRaises(ValueError):
                load_system_details()()

    def test_actual_host_telemetry(self):
        details = load_system_details()()
        self.assertIn(details['power_source'], ('unknown', 'AC', 'battery'))


if __name__ == '__main__':
    unittest.main()
