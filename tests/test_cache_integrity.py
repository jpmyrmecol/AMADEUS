"""Cache decisions and file lifecycle without GUI/GPU dependencies."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'main'))
import cache_integrity as ci


class CacheFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        # Isolate source-fingerprint tests from the installed repository.
        names = ('cache_integrity.py', 'obb_detection.py', 'training_paths.py',
                 'batch_utils.py', 'compute_backend.py', 'checkpoint_utils.py',
                 'weight_utils.py', 'video_frame_count.py', 'path_utils.py',
                 'multi_staged_association.py', 'assign_types.py', 'tracking_artifacts.py',
                 'multi_staged_association_variable.py', 'variable_population.py',
                 'identity_correction.py', 'identity_correction_variable.py',
                 'segmentation_metadata.py', 'segmentation_core.py', 'roi.py', 'random_utils.py',
                 'refinement.py', 'refinement_variable.py', 'tracking_constants.py')
        for directory in (self.source, self.source / 'without_direction_estimation'):
            directory.mkdir(parents=True)
            for name in names:
                (directory / name).write_text('# stable implementation\n')
        p = patch.object(ci, '_MAIN_DIR', self.source)
        p.start()
        self.addCleanup(p.stop)
        self.session = self.root / 'session'
        self.session.mkdir()
        self.video = self.root / 'video.mp4'
        self.video.write_bytes(b'video source 0000')
        self.weights = self.session / 'main' / 'model' / 'training' / 'dataset' / 'weights' / 'last.pt'
        self.weights.parent.mkdir(parents=True)
        self.weights.write_bytes(b'weights 0000')
        self.cfg = {'SESSION_PATH': str(self.session), 'TRACKING_VIDEO_PATH': str(self.video),
                    'NUM_OBJECTS': 2, 'RANDOM_SEED': 0, 'training': {'PRETRAINED_MODEL': 'model'},
                    'analysis': {'FLIP_SEC': 5.0}, 'EMBEDDING': {'ENABLE': False}}
        self.out = self.session / 'main' / 'model' / 'tracking' / 'dataset' / 'last' / 'video'
        self.cache = ci.TrackingCache(self.cfg, str(self.video), 'model', 'dataset', 'last', str(self.out))
        training_paths = SimpleNamespace(training_weight_path=lambda *args: str(self.weights))
        segmentation = SimpleNamespace(segmentation_paths_for_session=lambda *args:
            ('', str(self.session / 'segmentation' / 'background.png')))
        p = patch.dict(sys.modules, {'training_paths': training_paths,
            'without_direction_estimation.training_paths': training_paths,
            'segmentation_metadata': segmentation})
        p.start()
        self.addCleanup(p.stop)

    def make(self, stage, family=''):
        self.cache.begin(stage)
        for path in self.cache.outputs(stage, family):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'generated ' + path.name.encode())
        self.cache.complete(stage, family)

    def pipeline(self, embedding=False):
        self.make('detection')
        self.make('tracking')
        if embedding:
            self.make('embedding')
        self.make('refinement', 'id_resolved' if embedding else 'filled')


class CacheIntegrityTests(CacheFixture):
    def test_legacy_outputs_are_not_a_cache(self):
        path = self.cache.outputs('detection')[0]
        path.parent.mkdir(parents=True)
        path.write_bytes(b'old but nonempty')
        self.assertFalse(self.cache.current('detection'))

    def test_unchanged_pipeline_reuses_every_stage(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        for stage in ci._STAGE_ORDER:
            self.assertTrue(self.cache.current(stage), stage)

    def test_confidence_change_invalidates_descendants(self):
        self.pipeline()
        self.cfg['analysis']['CONF'] = 0.2
        for stage in ('detection', 'tracking', 'refinement'):
            self.assertFalse(self.cache.current(stage), stage)

    def test_nms_and_frame_range_are_detection_inputs(self):
        self.pipeline()
        for key, value in [('NMS_IOU', 0.6), ('FIRST_FRAME', 3), ('LAST_FRAME', 90), ('DEVICE', 'cpu'), ('BATCH_SIZE', 1)]:
            with self.subTest(key=key):
                self.cfg['analysis'][key] = value
                self.assertFalse(self.cache.current('detection'))
                del self.cfg['analysis'][key]
                self.assertTrue(self.cache.current('detection'))

    def test_matching_change_preserves_detection_only(self):
        self.pipeline()
        self.cfg['analysis']['MATCH_IOU'] = 0.3
        self.assertTrue(self.cache.current('detection'))
        self.assertFalse(self.cache.current('tracking'))
        self.assertFalse(self.cache.current('refinement'))

    def test_flip_change_preserves_detection_and_tracking(self):
        self.pipeline()
        self.cfg['analysis']['FLIP_SEC'] = 6
        self.assertTrue(self.cache.current('tracking'))
        self.assertFalse(self.cache.current('refinement'))

    def test_enabling_embedding_does_not_reuse_geometry_only_result(self):
        self.pipeline()
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.assertTrue(self.cache.current('tracking'))
        self.assertFalse(self.cache.current('refinement'))

    def test_disabling_embedding_does_not_reuse_resolved_result(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        self.cfg['EMBEDDING']['ENABLE'] = False
        self.assertFalse(self.cache.current('refinement'))
        self.assertTrue(self.cache.current('tracking'))

    def test_embedding_seed_and_crop_size_are_part_of_cache_key(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        self.cfg['RANDOM_SEED'] = 1
        self.assertFalse(self.cache.current('embedding'))
        self.assertFalse(self.cache.current('refinement'))
        self.assertTrue(self.cache.current('tracking'))
        self.cfg['RANDOM_SEED'] = 0
        self.cfg['EMBEDDING']['IMG_SIZE'] = 64
        self.assertFalse(self.cache.current('embedding'))

    def test_episode_settings_do_not_retrain_embedding(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        self.cfg['EMBEDDING']['EPISODE_MAX_LEN'] = 80
        self.assertTrue(self.cache.current('embedding'))
        self.assertFalse(self.cache.current('refinement'))

    def test_preview_worker_and_skip_changes_are_not_scientific_inputs(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        self.cfg.update(NUM_PREVIEW_FRAMES=2, NUM_WORKERS=1, skip_detection=True, skip_training=True)
        self.cfg['analysis']['SKIP_DETECT_PREVIEW'] = True
        self.cfg['EMBEDDING']['PREVIEW_COUNT'] = 3
        self.cfg['create_video'] = {'FPS': 20, 'DRAW_LABELS': True}
        for stage in ci._STAGE_ORDER:
            self.assertTrue(self.cache.current(stage), stage)

    def test_weight_replaced_in_same_path_invalidates_cache(self):
        self.pipeline()
        self.weights.write_bytes(b'weights 1111')
        self.assertFalse(self.cache.current('detection'))

    def test_content_change_with_same_size_and_mtime_is_found(self):
        self.pipeline()
        st = self.video.stat()
        self.video.write_bytes(b'video source 1111')
        os.utime(self.video, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertFalse(self.cache.current('detection'))

    def test_touch_without_content_change_reuses_cache(self):
        self.pipeline()
        st = self.video.stat()
        os.utime(self.video, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000000))
        self.assertTrue(self.cache.current('refinement'))

    def test_partial_or_modified_output_is_not_reused(self):
        self.pipeline()
        self.cache.outputs('tracking')[0].write_bytes(b'partial')
        self.assertFalse(self.cache.current('tracking'))
        self.assertFalse(self.cache.current('refinement'))

    def test_deleted_output_invalidates_complete_record(self):
        self.pipeline()
        self.cache.outputs('refinement', 'filled')[0].unlink()
        self.assertFalse(self.cache.current('refinement'))

    def test_corrupt_metadata_is_a_miss(self):
        self.make('detection')
        for content in ('{bad json', '[]', '{}', '{"family": "unexpected"}'):
            self.cache.record_path('detection').write_text(content)
            self.assertFalse(self.cache.current('detection'))

    def test_incomplete_run_has_no_completion_record(self):
        self.cache.begin('detection')
        with self.assertRaisesRegex(RuntimeError, 'outputs are incomplete'):
            self.cache.complete('detection')
        self.assertFalse(self.cache.record_path('detection').exists())

    def test_changed_input_during_run_is_not_committed(self):
        self.cache.begin('detection')
        output = self.cache.outputs('detection')[0]
        output.parent.mkdir(parents=True)
        output.write_bytes(b'result')
        self.weights.write_bytes(b'changed in progress')
        with self.assertRaisesRegex(RuntimeError, 'inputs changed during processing'):
            self.cache.complete('detection')
        self.assertFalse(self.cache.record_path('detection').exists())

    def test_crash_after_begin_cannot_reuse_old_completed_result(self):
        self.pipeline()
        self.cache.begin('tracking')
        self.assertFalse(self.cache.current('tracking'))
        self.assertFalse(self.cache.current('refinement'))
        self.assertTrue(self.cache.current('detection'))

    def test_atomic_completion_leaves_no_temporary_file(self):
        self.make('detection')
        self.assertFalse(list(self.cache.record_path('detection').parent.glob('*.tmp')))

    def test_changed_segmentation_invalidates_embedding_and_refinement(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        seg = self.session / 'segmentation'
        seg.mkdir()
        (seg / 'segmentation_gui_config.json').write_text('{"new": true}')
        self.assertFalse(self.cache.current('embedding'))
        self.assertFalse(self.cache.current('refinement'))
        self.assertTrue(self.cache.current('tracking'))

    def test_background_content_change_is_checked(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        seg = self.session / 'segmentation'
        seg.mkdir()
        bg = seg / 'background.png'
        bg.write_bytes(b'background a')
        self.pipeline(True)
        bg.write_bytes(b'background b')
        self.assertFalse(self.cache.current('embedding'))

    def test_refinement_code_change_keeps_embedding(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        (self.source / 'refinement.py').write_text('# changed\n')
        self.assertTrue(self.cache.current('embedding'))
        self.assertFalse(self.cache.current('refinement'))

    def test_detection_code_change_invalidates_all_descendants(self):
        self.pipeline()
        (self.source / 'obb_detection.py').write_text('# changed\n')
        self.assertFalse(self.cache.current('refinement'))

    def test_dependency_version_change_invalidates_cache(self):
        self.pipeline()
        with patch.object(ci, '_versions', return_value={'numpy': 'another-version'}):
            self.assertFalse(self.cache.current('detection'))

    def test_archive_preserves_final_csv_video_and_unrelated_files(self):
        self.pipeline()
        results = self.session / 'results'
        results.mkdir()
        for suffix in ('', '_id_resolved'):
            for ext in ('.csv', '.mp4'):
                (results / ('model_dataset_last_video' + suffix + ext)).write_bytes(b'keep old result')
        untouched = results / 'manual_results.csv'
        untouched.write_bytes(b'user file')
        self.cache.begin('detection')
        self.assertEqual(len(list((results / 'cache_archive').glob('*/*'))), 4)
        self.assertEqual(untouched.read_bytes(), b'user file')
        self.assertEqual(self.video.read_bytes(), b'video source 0000')
        self.assertEqual(self.weights.read_bytes(), b'weights 0000')

    def test_skipped_upstream_error_identifies_first_required_stage(self):
        self.pipeline()
        self.cfg['analysis']['CONF'] = 0.3
        with self.assertRaisesRegex(RuntimeError, 'Rerun detection; clear skip_detection'):
            self.cache.require('refinement')

    def test_variable_policy_and_stats_are_dependencies(self):
        self.cfg['VARIABLE_NUM_OBJECTS'] = True
        self.cache = ci.TrackingCache(self.cfg, str(self.video), 'model', 'dataset', 'last', str(self.out / 'variable'))
        self.pipeline()
        self.cfg['analysis']['VARIABLE_INTERIOR_CONFIRM_FRAMES'] = 5
        self.assertFalse(self.cache.current('tracking'))
        self.assertTrue(self.cache.current('detection'))
        del self.cfg['analysis']['VARIABLE_INTERIOR_CONFIRM_FRAMES']
        self.assertTrue(self.cache.current('tracking'))
        path = self.session / 'initial_tracking' / 'tracking_stats.csv'
        path.parent.mkdir()
        path.write_text('metric,value\nlength,40\n')
        self.assertFalse(self.cache.current('tracking'))

    def test_all_four_modes_complete_and_invalidate_independently(self):
        for directionless in (False, True):
            for variable in (False, True):
                with self.subTest(directionless=directionless, variable=variable):
                    self.cfg.update(WITHOUT_DIRECTION_ESTIMATION=directionless, VARIABLE_NUM_OBJECTS=variable)
                    self.out = self.session / f'mode_{directionless}_{variable}'
                    self.cache = ci.TrackingCache(self.cfg, str(self.video), 'model', 'dataset', 'last', str(self.out))
                    self.pipeline()
                    self.assertTrue(self.cache.current('refinement'))

    def test_empty_detection_output_is_still_a_completed_cache(self):
        self.cache.begin('detection')
        path = self.cache.outputs('detection')[0]
        path.parent.mkdir(parents=True)
        path.write_bytes(__import__('pickle').dumps([]))
        self.cache.complete('detection')
        self.assertTrue(self.cache.current('detection'))

    def test_invalid_stage_and_family_fail(self):
        for call in (lambda: self.cache.record_path('unknown'),
                     lambda: self.cache.outputs('refinement', 'bad')):
            with self.assertRaises(ValueError):
                call()


if __name__ == '__main__':
    unittest.main()
