# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""Hardware-independent contract and regression tests: python -m unittest discover -s tests."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock, patch

from main import compute_backend as cb, compute_telemetry as telemetry, batch_utils as batch

GIB = 1024 ** 3


def fake_torch(hip=None, cuda=None, available=True, mps=False):
    value = Mock()
    value.__add__ = Mock(return_value=value)
    value.item.return_value = 2
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip, cuda=cuda),
        cuda=SimpleNamespace(
            is_available=Mock(return_value=available), device_count=Mock(return_value=2),
            empty_cache=Mock(), synchronize=Mock(), memory_reserved=Mock(return_value=2 * GIB),
            memory_allocated=Mock(return_value=GIB),
            get_device_properties=Mock(return_value=SimpleNamespace(total_memory=12 * GIB)),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=mps))),
        mps=SimpleNamespace(empty_cache=Mock(), synchronize=Mock(),
            driver_allocated_memory=Mock(return_value=3 * GIB),
            current_allocated_memory=Mock(return_value=GIB),
            recommended_max_memory=Mock(return_value=8 * GIB)),
        ones=Mock(return_value=value), tensor=Mock(),
    )


class BackendTests(unittest.TestCase):
    def test_detection_matrix_and_wddm(self):
        for hip, cuda, available, mps, expected in [
            (None, '12.8', True, False, cb.Backend.NVIDIA_CUDA),
            ('6.3.0', None, True, False, cb.Backend.AMD_ROCM),
            ('6.3.0', '12.8', True, False, cb.Backend.AMD_ROCM),
            (None, None, False, True, cb.Backend.APPLE_MPS),
            (None, None, False, False, cb.Backend.CPU),
            ('6.3.0', None, False, False, cb.Backend.CPU),
            (None, None, True, False, cb.Backend.CPU),
        ]:
            with self.subTest(expected=expected, hip=hip, available=available), patch.object(
                cb, '_torch', return_value=fake_torch(hip, cuda, available, mps)
            ):
                self.assertEqual(cb.detect_backend(), expected)
                for system in ('win32', 'linux', 'darwin'):
                    with patch.object(cb.sys, 'platform', system):
                        self.assertEqual(cb.runtime_for().wddm,
                                         expected == cb.Backend.NVIDIA_CUDA and system == 'win32')

    def test_auto_and_explicit_devices(self):
        for backend, torch, auto in [
            (cb.Backend.NVIDIA_CUDA, fake_torch(cuda='12.8'), '0'),
            (cb.Backend.AMD_ROCM, fake_torch(hip='6.3'), '0'),
            (cb.Backend.APPLE_MPS, fake_torch(available=False, mps=True), 'mps'),
            (cb.Backend.CPU, fake_torch(available=False), 'cpu'),
        ]:
            with self.subTest(backend=backend), patch.object(cb, '_torch', return_value=torch), redirect_stdout(io.StringIO()) as log:
                self.assertEqual(batch.resolve_device('auto'), auto)
                self.assertIn(backend.value, log.getvalue())
                self.assertEqual(batch.resolve_device('cpu'), 'cpu')
                self.assertEqual(batch.resolve_device('mps'), 'mps' if auto == 'mps' else 'cpu')
                self.assertEqual(batch.resolve_device('mps:0'), 'mps' if auto == 'mps' else 'cpu')
                for spec in ('0', 'cuda', 'cuda:1', '0,1'):
                    self.assertEqual(batch.resolve_device(spec), spec if auto == '0' else 'cpu')
                self.assertEqual(batch.resolve_device('99'), 'cpu')
                if backend == cb.Backend.AMD_ROCM:
                    self.assertEqual(cb.runtime_for('cuda:1').torch_device, 'cuda:1')
                    self.assertEqual(batch._accelerator_type('0'), 'cuda')

    def test_unusable_hip_is_not_selected(self):
        torch = fake_torch(hip='6.3')
        torch.ones.side_effect = RuntimeError('invalid device function')
        with patch.object(cb, '_torch', return_value=torch), redirect_stdout(io.StringIO()):
            self.assertEqual(cb.resolve_device('auto'), 'cpu')
            self.assertEqual(cb.resolve_device('cuda:1'), 'cpu')

    def test_mps_probe_failure_falls_back(self):
        torch = fake_torch(available=False, mps=True)
        torch.mps.synchronize.side_effect = RuntimeError('MPS unavailable')
        with patch.object(cb, '_torch', return_value=torch), redirect_stdout(io.StringIO()):
            self.assertEqual(cb.resolve_device('auto'), 'cpu')
            self.assertEqual(cb.resolve_device('mps'), 'cpu')

    def test_memory_cache_sync_and_worker_capabilities(self):
        for backend, spec in [(cb.Backend.NVIDIA_CUDA, '1'), (cb.Backend.AMD_ROCM, 'cuda:1'),
                              (cb.Backend.APPLE_MPS, 'mps'), (cb.Backend.CPU, 'cpu')]:
            torch = fake_torch(hip='6.3' if backend == cb.Backend.AMD_ROCM else None,
                               cuda='12.8' if backend == cb.Backend.NVIDIA_CUDA else None)
            with self.subTest(backend=backend), patch.object(cb, '_torch', return_value=torch):
                runtime = cb.runtime_for(spec)
                batch.empty_accelerator_cache(spec)
                batch.synchronize_accelerator(spec)
                if runtime.capabilities.torch_device == 'cuda':
                    torch.cuda.empty_cache.assert_called_once()
                    torch.cuda.synchronize.assert_called_once_with(1)
                    torch.mps.empty_cache.assert_not_called()
                    self.assertEqual(runtime.memory(), (2 * GIB, 12 * GIB))
                    self.assertEqual(batch.effective_dataloader_workers(spec, 4), 4)
                elif spec == 'mps':
                    torch.mps.empty_cache.assert_called_once()
                    torch.mps.synchronize.assert_called_once()
                    torch.cuda.empty_cache.assert_not_called()
                    self.assertEqual(runtime.memory(), (3 * GIB, 8 * GIB))
                    self.assertEqual(batch.effective_dataloader_workers(spec, 4), 0)
                else:
                    torch.cuda.empty_cache.assert_not_called()
                    torch.mps.empty_cache.assert_not_called()
                    self.assertEqual(runtime.memory(), (None, None))
                    self.assertEqual(batch.effective_dataloader_workers(spec, 4), 0)

    def test_telemetry_missing_tools_and_vendor_isolation(self):
        for hip, cuda in [('6.3', None), (None, '12.8')]:
            torch = fake_torch(hip=hip, cuda=cuda)
            with patch.object(cb, '_torch', return_value=torch), patch.object(
                telemetry.subprocess, 'check_output', side_effect=FileNotFoundError
            ) as process:
                self.assertEqual(telemetry.sample_accelerator('0'), (None, 2., 12.))
                if hip:
                    process.assert_not_called()
                    self.assertEqual(telemetry.query_nvidia_smi('0'), (None, None, None))
                    process.assert_not_called()
                else:
                    self.assertEqual(process.call_args.args[0][0], 'nvidia-smi')

    def test_unavailable_measurements_and_optional_mps_apis(self):
        torch = fake_torch(hip='6.3')
        torch.cuda.memory_reserved.side_effect = RuntimeError('unavailable')
        with patch.object(cb, '_torch', return_value=torch):
            self.assertEqual(telemetry.sample_accelerator('0'), (None, None, None))
        torch.mps.empty_cache.side_effect = AttributeError('old torch')
        torch.mps.synchronize.side_effect = AttributeError('old torch')
        torch.mps.recommended_max_memory.side_effect = AttributeError('old torch')
        with patch.object(cb, '_torch', return_value=torch):
            batch.empty_accelerator_cache('mps')
            batch.synchronize_accelerator('mps')
            self.assertEqual(batch.get_accelerator_memory('mps'), (None, None))
        with patch.object(cb, '_torch', side_effect=ImportError):
            self.assertEqual(cb.detect_backend(), cb.Backend.CPU)

    def test_wddm_only_windows_nvidia(self):
        for hip, cuda in [('6.3', None), (None, '12.8')]:
            for system in ('linux', 'win32'):
                with patch.object(cb, '_torch', return_value=fake_torch(hip, cuda)), patch.object(
                    cb.sys, 'platform', system
                ), patch.object(telemetry.subprocess, 'check_output', return_value=str(GIB)) as process:
                    result = telemetry.query_wddm_non_local_usage(123, '0')
                    if system == 'win32' and not hip:
                        self.assertEqual(result, 1.)
                        self.assertIn('pid_123_', process.call_args.args[0][-1])
                    else:
                        self.assertIsNone(result)
                        process.assert_not_called()

    def test_nvidia_telemetry_units_unchanged(self):
        with patch.object(cb, '_torch', return_value=fake_torch(cuda='12.8')), patch.object(
            telemetry.subprocess, 'check_output', return_value='70, 2048, 12288'
        ):
            self.assertEqual(telemetry.sample_accelerator('0'), (70., 2., 12.))

    def test_both_training_samplers_route_without_nvidia_or_wddm_on_hip(self):
        # Execute the actual lightweight sampler class without importing YOLO/GUI.
        for path in ('main/obb_detector_training.py', 'main/without_direction_estimation/obb_detector_training.py'):
            tree = ast.parse(Path(path).read_text(encoding='utf-8'))
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_DeviceSampler')
            query = Mock(side_effect=AssertionError('WDDM must not run'))
            namespace = dict(threading=threading, time=time, runtime_for=cb.runtime_for,
                sample_accelerator=telemetry.sample_accelerator, _query_wddm_non_local_usage=query,
                TRAIN_SAMPLER_INTERVAL_SECONDS=5, TRAIN_SAMPLER_WDDM_INTERVAL_SECONDS=5)
            exec(compile(ast.Module(body=[cls], type_ignores=[]), path, 'exec'), namespace)
            with patch.object(cb, '_torch', return_value=fake_torch(hip='6.3')), patch.object(
                telemetry.subprocess, 'check_output', side_effect=AssertionError('No external tool')
            ) as process:
                sampler = namespace['_DeviceSampler']('0')
                sampler.start()
                sampler.stop()
                self.assertEqual(sampler._sample_memory(), (None, 2., 12.))
                query.assert_not_called()
                process.assert_not_called()

    def test_existing_batch_values_and_independent_dispatch(self):
        psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=64 * GIB),
                                  cpu_count=lambda logical=False: 8)
        for backend, spec, expected in [(cb.Backend.NVIDIA_CUDA, '0', 36),
            (cb.Backend.AMD_ROCM, '0', 36), (cb.Backend.APPLE_MPS, 'mps', 16),
            (cb.Backend.CPU, 'cpu', 16)]:
            torch = fake_torch(hip='6.3' if backend == cb.Backend.AMD_ROCM else None,
                               cuda='12.8' if backend == cb.Backend.NVIDIA_CUDA else None)
            with patch.object(cb, '_torch', return_value=torch), patch.dict('sys.modules', psutil=psutil), redirect_stdout(io.StringIO()):
                self.assertEqual(batch.auto_batch_size(640, spec), expected)
                self.assertEqual(batch.resolve_batch_size(7, 640, spec), 7)
                with patch.dict(batch.BATCH_POLICIES, {backend: Mock(return_value=13)}):
                    self.assertEqual(batch.auto_batch_size(640, spec), 13)
        self.assertEqual(len(set(batch.BATCH_POLICIES.values())), 4)

    def test_oom_recovery_and_non_oom_errors(self):
        torch = fake_torch(hip='6.3')
        attempted = []
        def work(size):
            attempted.append(size)
            if size > 2:
                raise RuntimeError('HIP out of memory')
            return size
        with patch.object(cb, '_torch', return_value=torch), redirect_stdout(io.StringIO()):
            self.assertEqual(batch.run_with_oom_retry(work, 8), 2)
            self.assertEqual(attempted, [8, 4, 2])
            self.assertEqual(batch._is_oom_error(RuntimeError('CUDA out of memory')), (True, 'AMD ROCm'))
            self.assertFalse(batch._is_oom_error(RuntimeError('invalid device function'))[0])
        self.assertEqual(batch.next_batch_candidate_soft_down(12), 8)


if __name__ == '__main__':
    unittest.main()
