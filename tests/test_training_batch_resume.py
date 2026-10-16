"""Dependency-free regression tests for the actual trainer batch-selection code."""
import ast
import contextlib
import io
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    'main/obb_detector_training.py',
    'main/without_direction_estimation/obb_detector_training.py',
)


def function_node(path, name):
    return next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


class ResumeBatchTests(unittest.TestCase):
    def select(self, variant, value, override=None):
        import math
        namespace = {'math': math}
        parser = function_node(ROOT / 'main/batch_utils.py', 'parse_training_batch_size')
        exec(compile(ast.Module(body=[parser], type_ignores=[]), 'batch_parser', 'exec'), namespace)
        requested = namespace['parse_training_batch_size'](value)
        main = function_node(ROOT / variant, 'main')
        resume = next(node for node in ast.walk(main)
                      if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                      and node.test.id == 'resume_checkpoint')
        selection = next(node for node in resume.body
                         if isinstance(node, ast.If)
                         and ast.unparse(node.test) == 'override_batch is not None')
        namespace.update(BATCH_SIZE=value, requested_batch=requested,
                         ultralytics_auto_batch=requested == -1 or isinstance(requested, float),
                         explicit_batch=isinstance(requested, int) and requested > 0,
                         override_batch=override, ckpt_batch=16)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=[selection], type_ignores=[]), variant, 'exec'), namespace)
        return namespace['batch_size']

    def test_normalized_integer_values_preserve_checkpoint_batch(self):
        for variant in VARIANTS:
            for value in (16, '16', 16.0, '16.0', '1e1', 10, '10'):
                with self.subTest(variant=variant, value=value):
                    self.assertEqual(self.select(variant, value), 16)

    def test_autobatch_and_recovery_override_are_preserved(self):
        for variant in VARIANTS:
            for value, expected in ((-1, -1), ('-1', -1), (.6, .6), ('0.5', .5), ('auto', 16)):
                with self.subTest(variant=variant, value=value):
                    self.assertEqual(self.select(variant, value), expected)
                    self.assertEqual(self.select(variant, value, override=8), 8)


if __name__ == '__main__':
    unittest.main()
