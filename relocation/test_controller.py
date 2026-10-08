"""Integration contract checks with file-based worker responses, without GPU weights."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile
import numpy as np
from PIL import Image
from .config import REPO, Settings
from .controller import Controller, working_image
from .geometry import image_hash


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='relocation-test-', dir=REPO)
        self.addCleanup(self.temp.cleanup)
        self.controller = Controller(Settings(output_dir=self.temp.name))
        self.image = np.full((96, 128, 3), 80, dtype=np.uint8)
        self.mask = np.zeros((96, 128), dtype=np.uint8)
        self.mask[20:40, 20:40] = 255
        self.canvas = {'image': self.image.copy(), 'mask': np.zeros((96, 128, 4), dtype=np.uint8)}
        self.state = self.controller.import_mask(self.canvas, self.mask,)

    def test_source_swap_is_rejected(self):
        changed = self.image.copy()
        changed[0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'Ảnh đã thay đổi'):
            self.controller.validate(changed, self.state)

    def test_empty_prompt_fails_before_loading_gpu_libraries(self):
        with self.assertRaisesRegex(ValueError, 'Vẽ scribble'):
            self.controller.seem.segment(self.image, None, None, '')
        self.assertIsNone(self.controller.seem.model)

    def test_changed_scribble_is_rejected(self):
        self.state['scribble_hash'] = image_hash(np.zeros((96, 128), dtype=bool))
        self.canvas['mask'][20, 20, 3] = 255
        with self.assertRaisesRegex(ValueError, 'Scribble đã thay đổi'):
            self.controller.validate(self.canvas, self.state)

    def test_large_upload_is_capped_and_mask_aligns(self):
        large = np.zeros((600, 2400, 3), dtype=np.uint8)
        raw, image = working_image(large)
        self.assertEqual(raw.shape, (600, 2400, 3))
        self.assertEqual(image.shape, (512, 2048, 3))
        mask = np.zeros((600, 2400), dtype=np.uint8)
        mask[200:400, 1000:1400] = 255
        state = self.controller.import_mask(large, mask)
        self.assertEqual(state['mask'].shape, image.shape[:2])

    def test_depth_cache_uses_raw_metric_values_and_recomputes_target_stats(self):
        calls = []

        def worker(settings, mode, request, job):
            calls.append(mode)
            raw = np.full((96, 128), 4, dtype=np.float32)
            raw[20:40, 20:40] = 2
            np.save(request['output'], raw)
            Path(request['metadata']).write_text(json.dumps({'metric': True}))

        with patch('relocation.controller.run_worker', side_effect=worker):
            state, info, _ = self.controller.estimate(self.canvas, self.state, (80, 50), 'Metric indoor')
            self.assertEqual(info['scale'], .5)
            state, info, _ = self.controller.estimate(self.canvas, state, (30, 30), 'Metric indoor')
            self.assertEqual(info['scale'], 1)
            self.assertEqual(calls, ['depth'])

    def test_relocation_exports_complete_archive_and_separate_attempts(self):
        calls = []

        def worker(settings, mode, request, job):
            calls.append(request)
            for name in ['result.png', 'removal_mask.png', 'harmonization_mask.png', 'background.png', 'target_cutout.png', 'pasted.png']:
                Image.fromarray(self.image).save(Path(job) / name)
            (Path(job) / 'brushnet_result.json').write_text(json.dumps({'scale': request['scale']}))

        with patch('relocation.controller.run_worker', side_effect=worker):
            parameters = (self.canvas, self.state, (80, 50), 1, 'red apple', 'wooden tabletop', '',
                'artifacts', 30, 7.5, 1, 1234, 12, 12, False)
            _, _, archive, first = self.controller.relocate(*parameters)
            _, _, _, second = self.controller.relocate(*parameters)
        self.assertNotEqual(first, second)
        self.assertEqual(calls[0]['target'], [80, 50])
        self.assertEqual(calls[0]['seed'], 1234)
        with zipfile.ZipFile(archive) as zipped:
            self.assertTrue({'source.png', 'source_mask.png', 'result.png', 'metadata.json'} <= set(zipped.namelist()))
            metadata = json.loads(zipped.read('metadata.json'))
            self.assertEqual(metadata['request']['scale'], 1)
            self.assertEqual(metadata['object_text'], 'red apple')


if __name__ == '__main__':
    unittest.main()
