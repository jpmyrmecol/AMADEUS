"""Dependency-free tests of actual training closures with mocked GPU work.

The backend cases verify recovery bookkeeping only, not hardware performance.
"""
import ast
import contextlib
import io
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "main/obb_detector_training.py",
    "main/without_direction_estimation/obb_detector_training.py",
)
MISSING = object()


class Resource:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def start(self):
        pass

    def stop(self):
        pass


class Model:
    def __init__(self, selected=MISSING, failure="setup", exception=None):
        self.selected = selected
        self.failure = failure
        self.exception = exception or MemoryError("setup allocation failed")
        self.callbacks = {}

    def add_callback(self, event, callback):
        self.callbacks[event] = callback

    def train(self, **kwargs):
        if self.selected is not MISSING:
            self.trainer = SimpleNamespace(batch_size=self.selected)
        if self.failure == "setup":
            raise self.exception
        self.callbacks["on_train_start"](self.trainer)
        if self.failure == "training":
            raise self.exception


def runner(variant, requested, model, backend="cuda"):
    """Compile the complete source closure, retaining its nonlocal binding."""
    tree = ast.parse((ROOT / variant).read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    train = next(node for node in ast.walk(main)
                 if isinstance(node, ast.FunctionDef) and node.name == "_do_train")
    wrapper = ast.parse("""
def make_runner(requested, model):
    batch_size = requested
    def current_batch():
        return batch_size
    return _do_train, current_batch
""").body[0]
    wrapper.body.insert(1, train)
    errors = [node for node in tree.body if isinstance(node, ast.ClassDef)
              and node.name in {"_TrainingAborted", "_BatchTooLargeError"}]
    module = ast.fix_missing_locations(ast.Module(body=errors + [wrapper], type_ignores=[]))
    namespace = {
        "os": SimpleNamespace(environ={}),
        "logging": logging,
        "base_kwargs": {},
        "device": backend,
        "TRAIN_IMG_SIZE": 640,
        "dl_workers": 8,
        "explicit_batch": isinstance(requested, int) and requested > 0,
        "ultralytics_auto_batch": requested == -1 or isinstance(requested, float),
        "vram_headroom_locked": False,
        "Backend": SimpleNamespace(AMD_ROCM="rocm"),
        "runtime_for": lambda device: SimpleNamespace(backend=device),
        "_WATCHDOG_STATE_ENV": "unused",
        "_parse_cuda_device_index": lambda device: 0,
        "_DeviceSampler": Resource,
        "_PerformanceMonitor": Resource,
        "_InitialLoadCheck": Resource,
        "_GpuFallbackWatcher": logging.Handler,
        "resume_checkpoint": None,
        "trainer_cls": object,
        "_enable_cudnn_autotuner": lambda device: None,
        "_is_oom_error": lambda exc: (isinstance(exc, MemoryError), backend),
        "_is_dataloader_failure": lambda exc: False,
    }
    exec(compile(module, variant, "exec"), namespace)
    train, current_batch = namespace["make_runner"](requested, model)
    return train, current_batch, namespace["_BatchTooLargeError"]


class SetupAutoBatchTests(unittest.TestCase):
    def check_oom(self, variant, requested, model, expected, backend="cuda"):
        train, current_batch, recovery_error = runner(variant, requested, model, backend)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(recovery_error) as caught:
                train()
        self.assertEqual(caught.exception.action, "lower_batch")
        self.assertEqual(current_batch(), expected)
        self.assertIn(f"batch={expected}", str(caught.exception))

    def test_setup_oom_after_selection_retains_resolved_batch(self):
        for variant in VARIANTS:
            for backend, selected in (("cpu", 16), ("mps", 16), ("cuda", 24), ("rocm", 24)):
                for requested in (-1, .6):
                    with self.subTest(variant=variant, backend=backend, requested=requested):
                        self.check_oom(variant, requested, Model(selected), selected, backend)

    def test_unselected_or_invalid_batch_is_not_used(self):
        for variant in VARIANTS:
            for requested in (-1, .6):
                for selected in (MISSING, None, -1, .6, 0, -4, True, False, 16.0, "16"):
                    with self.subTest(variant=variant, requested=requested, selected=selected):
                        self.check_oom(variant, requested, Model(selected), requested)
                model = Model()
                model.trainer = None
                self.check_oom(variant, requested, model, requested)
                model = Model()
                model.trainer = SimpleNamespace()
                self.check_oom(variant, requested, model, requested)

    def test_existing_positive_batch_is_preserved(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.check_oom(variant, 8, Model(24), 8)

    def test_selected_batch_one_is_available_to_existing_recovery_floor(self):
        for variant in VARIANTS:
            for requested in (-1, .6):
                with self.subTest(variant=variant, requested=requested):
                    self.check_oom(variant, requested, Model(1), 1)

    def test_success_and_post_callback_oom_keep_existing_behavior(self):
        for variant in VARIANTS:
            for requested in (-1, .6, 24):
                with self.subTest(variant=variant, requested=requested):
                    train, current_batch, _ = runner(variant, requested, Model(24, failure=None))
                    with contextlib.redirect_stdout(io.StringIO()):
                        train()
                    self.assertEqual(current_batch(), 24)
                    self.check_oom(variant, requested, Model(24, failure="training"), 24)

    def test_unrelated_exception_does_not_change_pending_batch(self):
        for variant in VARIANTS:
            for requested in (-1, .6):
                with self.subTest(variant=variant, requested=requested):
                    original = ValueError("bad dataset")
                    train, current_batch, _ = runner(variant, requested, Model(24, exception=original))
                    with self.assertRaises(ValueError) as caught:
                        train()
                    self.assertIs(caught.exception, original)
                    self.assertEqual(current_batch(), requested)


if __name__ == "__main__":
    unittest.main()
