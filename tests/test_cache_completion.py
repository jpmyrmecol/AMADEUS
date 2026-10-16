"""Completion-record dependencies and explicitly skipped upstream stages."""
from types import SimpleNamespace
import sys
from unittest.mock import patch
from test_cache_integrity import CacheFixture, ci


class CacheCompletionTests(CacheFixture):
    def test_zero_byte_output_is_not_complete(self):
        self.cache.begin('detection')
        path = self.cache.outputs('detection')[0]
        path.parent.mkdir(parents=True)
        path.write_bytes(b'')
        with self.assertRaisesRegex(RuntimeError, 'outputs are incomplete'):
            self.cache.complete('detection')
        self.assertFalse(self.cache.current('detection'))

    def test_embedding_replacement_invalidates_refinement(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        self.cache.outputs('embedding')[1].write_bytes(b'replaced model')
        self.assertFalse(self.cache.current('refinement'))
        self.assertTrue(self.cache.current('tracking'))

    def test_embedding_deletion_invalidates_refinement(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        self.cache.outputs('embedding')[0].unlink()
        self.assertFalse(self.cache.current('refinement'))

    def test_geometry_output_after_unavailable_embedding_can_be_reused(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.make('detection')
        self.make('tracking')
        self.make('refinement', 'filled')
        self.assertTrue(self.cache.current('refinement'))
        self.cache.outputs('embedding')[1].write_bytes(b'newly available model')
        self.assertFalse(self.cache.current('refinement'))

    def detection_namespace(self):
        return SimpleNamespace(
            resolve_model_name=lambda name: 'model',
            resolve_existing_experiment_dir_name=lambda *a, **k: 'dataset',
            experiment_dir_name_from_cfg=lambda cfg: 'dataset',
            parse_weight_spec=lambda value: ['last'],
            deduplicate_best_epoch_weights=lambda *a: ['last'],
            build_tracking_run_name=lambda weight: weight,
            configured_tracking_video_files=lambda *a: [str(self.video)],
            build_tracking_out_dir=lambda *a: str(self.out))

    def test_preflight_preserves_valid_skips(self):
        self.pipeline()
        self.cfg.update(skip_detection=True, skip_id_tracking=True, skip_refinement=False)
        with patch.dict(sys.modules, {'obb_detection': self.detection_namespace()}):
            ci.validate_skipped_analysis(self.cfg)
        self.assertTrue(self.cfg['skip_detection'])
        self.assertTrue(self.cfg['skip_id_tracking'])

    def test_preflight_stops_before_consuming_stale_skipped_input(self):
        self.pipeline()
        self.cfg.update(skip_detection=True, skip_id_tracking=True, skip_refinement=False)
        self.cfg['analysis']['CONF'] = 0.2
        with patch.dict(sys.modules, {'obb_detection': self.detection_namespace()}):
            with self.assertRaisesRegex(RuntimeError, 'Rerun detection; clear skip_detection'):
                ci.validate_skipped_analysis(self.cfg)

    def test_video_only_checks_its_actual_weight_in_renderer(self):
        self.cfg.update(skip_detection=True, skip_id_tracking=True, skip_refinement=True,
                        skip_creating_video=False)
        with patch.object(ci.importlib, 'import_module', side_effect=AssertionError('Wrong analysis-weight validation')):
            ci.validate_skipped_analysis(self.cfg)

    def test_never_uses_a_stale_embedding_completion_record(self):
        self.cfg['EMBEDDING']['ENABLE'] = True
        self.pipeline(True)
        self.cache.record_path('embedding').write_text('{}')
        self.assertFalse(self.cache.current('refinement'))
