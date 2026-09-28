# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

"""MPS policy and immutable c0e2483 NVIDIA/CPU behavior contracts."""
import ast
from contextlib import redirect_stdout
import io
import itertools
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Callable
import unittest
from unittest.mock import Mock, patch

from main import batch_utils as batch, compute_backend as cb
from test_compute_backend import fake_torch, GIB

BASE = 'c0e2483b8e0d486bdfb860da240fd6e89974c107'
PATHS = ['main/obb_detector_training.py', 'main/without_direction_estimation/obb_detector_training.py']

def golden(path):
    return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], text=True)

def definition(source, name):
    return next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)

def load_class(source, name, **namespace):
    node = definition(source, name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<contract>', 'exec'), namespace)
    return namespace[name]

class MPSPolicyTests(unittest.TestCase):
    def test_288_cuda_and_288_cpu_golden_batch_combinations(self):
        old = dict(__name__='main.batch_golden', __package__='main')
        exec(compile(golden('main/batch_utils.py'), '<golden>', 'exec'), old)
        for spec in ['0', 'cpu']:
            count = 0
            for memory, size, density, mode, cap in itertools.product(
                    [4, 8, 12, 24], [288, 576, 640, 1280], [0, 35, 200],
                    ['train', 'predict'], ['', '8', '64']):
                torch = fake_torch(cuda='12.8')
                torch.cuda.get_device_properties.return_value.total_memory = memory * GIB
                psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=memory * GIB),
                                         cpu_count=lambda logical=False: 8)
                density_result = (density, density, 100, 100)
                with patch.object(cb, '_torch', return_value=torch), patch.dict('sys.modules', psutil=psutil), \
                     patch.dict('os.environ', {'AMADEUS_MAX_AUTO_BATCH': cap}, clear=True), \
                     patch.object(batch, '_estimate_yolo_label_density', return_value=density_result), \
                     patch.dict(old, _estimate_yolo_label_density=lambda *a, **kw: density_result), \
                     redirect_stdout(io.StringIO()):
                    args=dict(image_size=size, device=spec, mode=mode, labels_dir='unused', task='obb')
                    self.assertEqual(batch.auto_batch_size(**args), old['auto_batch_size'](**args), args)
                count += 1
            self.assertEqual(count, 288)

    def test_mps_default_does_not_profile_or_use_memory_formula(self):
        with patch.object(batch, '_estimate_batch_size', side_effect=AssertionError('no memory formula')), redirect_stdout(io.StringIO()):
            for size in [288, 576, 1280]:
                self.assertEqual(batch.auto_batch_size(size, 'mps', task='obb'), 16)
                self.assertEqual(batch.resolve_batch_size(48, size, 'mps'), 48)
        with patch.object(batch, '_estimate_batch_size', return_value=7) as estimate:
            self.assertEqual(batch.auto_batch_size(576, 'mps', mode='predict'), 7)
            estimate.assert_called_once()

    def test_ratio_and_ram_do_not_change_mps_batch_cuda_unchanged(self):
        for path in PATHS:
            for spec in ['mps', '0', 'cpu']:
                for fraction, ram in [(1.2, .99), (.1, .2)]:
                    results=[]
                    for source in [golden(path), Path(path).read_text()]:
                        sampler=SimpleNamespace(snapshot=lambda: (None, 12., 10., fraction, None))
                        cls=load_class(source, '_InitialLoadCheck', time=SimpleNamespace(monotonic=lambda: 1.),
                            _DeviceSampler=object, effective_dataloader_workers=lambda d,w: 0,
                            _accelerator_type=lambda d: 'mps' if d=='mps' else 'cuda' if d=='0' else 'cpu',
                            _query_ram_fraction=lambda:ram, TRAIN_INITIAL_RAM_TARGET=.9,
                            TRAIN_INITIAL_VRAM_HEADROOM_TRIGGER=.4, TRAIN_VRAM_PRESSURE_RATIO=.95)
                        obj=cls(device=spec,sampler=sampler,batch_size=16,workers=0,allow_headroom_raise=True)
                        obj.record_batch(0);obj.finish_epoch(0);results.append(obj.verdict())
                    if spec=='mps': self.assertIsNone(results[1])
                    else: self.assertEqual(*results)

    def test_clear_backport_only_mps(self):
        # Extract the actual nested trainer: exercise its super call, no hardware needed.
        for path in PATHS:
            factory=definition(Path(path).read_text(), '_get_epochs_obb_trainer')
            node=next(n for n in factory.body if isinstance(n, ast.ClassDef))
            class Parent:
                def _clear_memory(self, threshold=None): return threshold
            ns=dict(OBBTrainer=Parent)
            exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),ns)
            for kind in ['cuda','cpu','mps']:
                trainer=ns['EpochsOBBTrainer']();trainer.device=SimpleNamespace(type=kind)
                for threshold in [None,.5,.9]:
                    self.assertEqual(trainer._clear_memory(threshold), None if kind=='mps' else threshold)

    def test_mps_oom_remains_actionable(self):
        torch = fake_torch(available=False, mps=True)
        attempts = []
        def train(size):
            attempts.append(size)
            if size > 8:
                raise RuntimeError('MPS backend out of memory')
            return size
        with patch.object(cb, '_torch', return_value=torch), redirect_stdout(io.StringIO()):
            self.assertEqual(batch.run_with_oom_retry(train, 16), 8)
            self.assertEqual(attempts, [16, 8])
            torch.mps.empty_cache.assert_called_once()
            torch.cuda.empty_cache.assert_not_called()
            self.assertFalse(batch._is_oom_error(RuntimeError('unsupported operation'))[0])

    def test_monitor_non_mps_stays_noop(self):
        for path in PATHS:
            for source in [golden(path), Path(path).read_text()]:
                cls = load_class(source, '_PerformanceMonitor', _DeviceSampler=object, Callable=Callable,
                    _accelerator_type=lambda d: d)
                obj = object.__new__(cls)
                for device in ['cuda', 'cpu']:
                    obj.device = device
                    obj._pressure_action = None
                    obj._check_memory_pressure('training')
                    self.assertIsNone(obj._pressure_action)

    def test_modified_initial_methods_have_only_mps_deltas(self):
        for path in PATHS:
            old = definition(golden(path), '_InitialLoadCheck')
            new = definition(Path(path).read_text(), '_InitialLoadCheck')
            for node in ast.walk(new):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.BoolOp):
                    if any(isinstance(t, ast.Attribute) and t.attr == 'allow_headroom_raise' for t in node.targets):
                        node.value = node.value.values[0]
                if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
                    values = node.test.values
                    if isinstance(values[0], ast.Compare) and isinstance(values[0].left, ast.Call):
                        call = values[0].left
                        if isinstance(call.func, ast.Name) and call.func.id == '_accelerator_type':
                            node.test.values = values[1:]
            evaluate = next(n for n in old.body if getattr(n, 'name', None) == '_evaluate_device_state')
            outer = next(n for n in evaluate.body if isinstance(n, ast.If))
            outer.body = [n for n in outer.body if not isinstance(n, ast.If)]
            self.assertEqual(ast.dump(old), ast.dump(new), path)

    def test_all_other_training_definitions_and_constants_unchanged(self):
        # Structural comparison covers optimizer/AMP/kwargs, resume/save, recovery,
        # watchdog, throughput, WDDM, workers, thresholds and both entry points.
        for path in PATHS:
            old=ast.parse(golden(path));new=ast.parse(Path(path).read_text())
            for tree in [old,new]:
                for node in tree.body:
                    if isinstance(node,ast.FunctionDef) and node.name=='_get_epochs_obb_trainer':
                        cls=next(n for n in node.body if isinstance(n,ast.ClassDef))
                        cls.body=[n for n in cls.body if getattr(n,'name',None)!='_clear_memory']
                    if isinstance(node,ast.ClassDef) and node.name in ['_InitialLoadCheck','_PerformanceMonitor']:
                        excluded={'_check_memory_pressure'} if node.name=='_PerformanceMonitor' else {'__init__','record_batch','_evaluate_device_state'}
                        node.body=[n for n in node.body if getattr(n,'name',None) not in excluded]
            self.assertEqual(ast.dump(old),ast.dump(new),path)
        for path in ['tools/runtime_profiles.py','pyproject.toml','uv.lock']:
            self.assertEqual(Path(path).read_text(),golden(path),path)

if __name__=='__main__':unittest.main()
