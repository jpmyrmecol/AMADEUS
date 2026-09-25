# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import runtime_profiles as profiles, setup_environment as setup
from test_compute_backend import fake_torch


def pair(profile):
    p = profiles.PROFILES[profile]
    torch = fake_torch(hip='6.3.42131' if profile == 'rocm63' else None,
                       cuda='12.8' if profile == 'cu128' else None)
    torch.__version__ = '2.7.1' + ('+' + p.suffix if p.suffix else '')
    vision = SimpleNamespace(__version__='0.22.1' + ('+' + p.suffix if p.suffix else ''),
                             ops=SimpleNamespace(nms=Mock(return_value=SimpleNamespace(numel=lambda: 1))))
    return torch, vision


class ProfileTests(unittest.TestCase):
    def test_selection(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(setup.sys, 'platform', 'linux'), patch.object(
            setup, 'nvidia_gpu_is_available', return_value=True
        ) as nvidia, patch.object(setup, 'amd_runtime_candidate', return_value=True) as amd:
            self.assertEqual(setup.select_torch_profile()[0], 'cu128')
            nvidia.return_value = False
            self.assertEqual(setup.select_torch_profile()[0], 'rocm63')
            amd.return_value = False
            self.assertEqual(setup.select_torch_profile()[0], 'cpu')
            with patch.dict(os.environ, AMADEUS_TORCH_PROFILE='cpu'):
                nvidia.return_value = True
                self.assertEqual(setup.select_torch_profile()[0], 'cpu')
            with patch.object(setup.sys, 'platform', 'darwin'):
                self.assertEqual(setup.select_torch_profile()[0], 'macos')

    def test_rocm_support_policy_is_separate(self):
        with patch.object(profiles.sys, 'platform', 'win32'):
            with self.assertRaises(RuntimeError):
                profiles.check_support('rocm63')
        with patch.object(profiles.sys, 'platform', 'linux'), patch.object(profiles.platform, 'machine', return_value='x86_64'):
            profiles.check_support('rocm63')
        with self.assertRaises(RuntimeError):
            profiles.check_support('unknown')

    def test_build_verification_matrix_rejects_wrong_vendor_and_profile(self):
        with patch.object(profiles.sys, 'platform', 'linux'), patch.object(profiles.platform, 'machine', return_value='x86_64'):
            for wanted in ('cpu', 'cu128', 'rocm63'):
                for installed in ('cpu', 'cu128', 'rocm63'):
                    with self.subTest(wanted=wanted, installed=installed):
                        torch, vision = pair(installed)
                        if wanted == installed:
                            profiles.verify_build(wanted, torch, vision)
                        else:
                            with self.assertRaises(RuntimeError):
                                profiles.verify_build(wanted, torch, vision)
            torch, vision = pair('rocm63')
            vision.__version__ = '0.22.1+cpu'
            with self.assertRaises(RuntimeError):
                profiles.verify_build('rocm63', torch, vision)
            torch, vision = pair('rocm63')
            torch.version.hip = '6.4.0'
            with self.assertRaises(RuntimeError):
                profiles.verify_build('rocm63', torch, vision)

    def test_macos_cpu_wheels(self):
        with patch.object(profiles.sys, 'platform', 'darwin'):
            profiles.verify_build('macos', *pair('macos'))
            profiles.verify_build('cpu', *pair('macos'))
            with self.assertRaises(RuntimeError):
                profiles.verify_build('macos', *pair('rocm63'))

    def test_execution_checks_kernel_and_nms_for_both_gpu_profiles(self):
        with patch.object(profiles.sys, 'platform', 'linux'), patch.object(profiles.platform, 'machine', return_value='x86_64'):
            for profile in ('cu128', 'rocm63'):
                torch, vision = pair(profile)
                profiles.verify_execution(profile, torch, vision)
                torch.ones.assert_called_once_with(1, device='cuda:0')
                vision.ops.nms.assert_called_once()
                torch.cuda.is_available.return_value = False
                with self.assertRaises(RuntimeError):
                    profiles.verify_execution(profile, torch, vision)
                torch.cuda.is_available.return_value = True
                torch.ones.side_effect = RuntimeError('no kernel image available')
                with self.assertRaises(RuntimeError):
                    profiles.verify_execution(profile, torch, vision)
                torch.ones.side_effect = None
                vision.ops.nms.side_effect = RuntimeError('NMS not implemented')
                with self.assertRaises(RuntimeError):
                    profiles.verify_execution(profile, torch, vision)

    def test_ready_marker_requires_matching_profile_and_actual_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'ready'
            with patch.object(setup, 'READY_MARKER', marker), patch.object(
                setup, 'select_torch_profile', return_value=('cu128', '')
            ), patch.object(setup, 'verify_pytorch_profile') as verify:
                self.assertFalse(setup.environment_ready())
                marker.write_text('')
                self.assertFalse(setup.environment_ready())
                marker.write_text(json.dumps(setup.ready_identity('cpu')))
                self.assertFalse(setup.environment_ready())
                verify.assert_not_called()
                marker.write_text(json.dumps(setup.ready_identity('cu128')))
                self.assertTrue(setup.environment_ready())
                verify.assert_called_once_with('cu128')
                verify.side_effect = RuntimeError('wrong wheel')
                self.assertFalse(setup.environment_ready())
                stale = setup.ready_identity('cu128')
                stale['inputs'] = 'stale'
                marker.write_text(json.dumps(stale))
                self.assertFalse(setup.environment_ready())

    def test_install_and_sync_preserve_existing_policy(self):
        with patch.object(profiles.sys, 'platform', 'linux'), patch.object(profiles.platform, 'machine', return_value='x86_64'), patch.object(
            setup.subprocess, 'run'
        ) as run:
            for profile in ('cpu', 'cu128', 'rocm63'):
                setup.install_exact_pytorch_profile('uv', profile)
                command = run.call_args.args[0]
                self.assertIn('--no-deps', command)
                self.assertIn('torch==2.7.1+' + profiles.PROFILES[profile].suffix, command)
                self.assertIn(profiles.PROFILES[profile].index, command)
                setup.sync_environment('uv', profile, 'python')
                command = run.call_args.args[0]
                self.assertIn('--locked', command)
                self.assertIn('--inexact', command)
                self.assertIn(profile, command)
                self.assertEqual(command.count('--no-install-package'), 2)

    def test_repair_reverifies_and_failure_does_not_pass(self):
        with patch.object(setup, 'verify_pytorch_profile', side_effect=[RuntimeError('wrong wheel'), None]) as verify, patch.object(
            setup, 'install_exact_pytorch_profile'
        ) as install:
            setup.ensure_pytorch_profile('uv', 'cu128')
            install.assert_called_once_with('uv', 'cu128')
            self.assertEqual(verify.call_count, 2)
        with patch.object(setup, 'verify_pytorch_profile', side_effect=RuntimeError('unsupported GPU')), patch.object(
            setup, 'install_exact_pytorch_profile'
        ):
            with self.assertRaises(RuntimeError):
                setup.ensure_pytorch_profile('uv', 'rocm63')


if __name__ == '__main__':
    unittest.main()
