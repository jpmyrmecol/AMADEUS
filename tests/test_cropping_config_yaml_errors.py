"""Exercise crop-config parsing without importing GUI or video dependencies.

Run with: python -m unittest discover -s tests -p test_cropping_config_yaml_errors.py
"""
import ast
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import yaml


def _load_config_method(namespace):
    source = Path(__file__).resolve().parents[1] / "gui" / "gui_preprocess.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    app = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "PreprocessApp")
    method = next(node for node in app.body
                  if isinstance(node, ast.FunctionDef) and node.name == "load_cropping_config")
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["load_cropping_config"]


class CroppingConfigYamlErrorsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "cropping.yaml"
        self.video = self.root / "video.mp4"
        self.video.touch()  # Parsing only; no real video is opened.
        self.dialog = Mock()
        self.prepare = Mock(return_value=None)
        self.load = _load_config_method({
            "Path": Path,
            "math": math,
            "messagebox": self.dialog,
            "prepare_analysis_video": self.prepare,
            "OUTPUT_SPEED_MIN": 0.1,
            "OUTPUT_SPEED_MAX": 100.0,
        })
        self.original_reader = object()
        self.app = SimpleNamespace(
            export_running=False,
            reader=self.original_reader,
            _loading_cropping_config=False,
            config_save_job="pending-save",
            load_video=Mock(),
            _flush_crop_trimming_config_save=Mock(),
            _write_crop_trimming_config=Mock(),
            _regions_from_cropping_config=Mock(return_value=[]),
            _log=Mock(),
        )

    def assert_no_mutation(self, original_bytes):
        self.assertEqual(self.config.read_bytes(), original_bytes)
        self.assertIs(self.app.reader, self.original_reader)
        self.assertFalse(self.app._loading_cropping_config)
        self.assertEqual(self.app.config_save_job, "pending-save")
        self.app.load_video.assert_not_called()
        self.app._flush_crop_trimming_config_save.assert_not_called()
        self.app._write_crop_trimming_config.assert_not_called()

    def test_malformed_yaml_is_reported_without_loading_or_saving(self):
        for content in ("tool: [\n", "tool: !unsupported value\n"):
            with self.subTest(content=content):
                self.dialog.reset_mock()
                self.config.write_text(content, encoding="utf-8")
                original = self.config.read_bytes()
                self.assertFalse(self.load(self.app, str(self.config)))
                self.dialog.showerror.assert_called_once()
                args, kwargs = self.dialog.showerror.call_args
                self.assertEqual(args[0], "Cropping config")
                self.assertTrue(args[1])
                self.assertIs(kwargs["parent"], self.app)
                self.prepare.assert_not_called()
                self.assert_no_mutation(original)

    def test_ordinary_validation_errors_remain_reported(self):
        for content in ("[]\n", "tool: Another application\n"):
            with self.subTest(content=content):
                self.dialog.reset_mock()
                self.config.write_text(content, encoding="utf-8")
                original = self.config.read_bytes()
                self.assertFalse(self.load(self.app, str(self.config)))
                self.dialog.showerror.assert_called_once()
                self.prepare.assert_not_called()
                self.assert_no_mutation(original)

    def test_valid_yaml_reaches_video_preparation_and_can_cancel(self):
        self.config.write_text(yaml.safe_dump({
            "tool": "AMADEUS Cropping & Trimming",
            "video_path": "video.mp4",
            "source_width": 640,
            "source_height": 480,
            "regions": [{"kind": "full"}],
        }), encoding="utf-8")
        original = self.config.read_bytes()
        self.assertFalse(self.load(self.app, str(self.config)))
        self.prepare.assert_called_once_with(self.app, str(self.video), log=self.app._log)
        self.dialog.showerror.assert_not_called()
        self.assert_no_mutation(original)


if __name__ == "__main__":
    unittest.main()
