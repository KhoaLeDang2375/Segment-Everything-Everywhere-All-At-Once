"""CPU regression checks for relocation invariants; no model downloads needed."""
import unittest
import numpy as np
from .geometry import (ink, transform_foreground, composite, harmonization_mask,
                       blend_repair, suggest_scale, centroid)


class RelocationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((96, 128, 3), 80, dtype=np.uint8)
        self.mask = np.zeros((96, 128), dtype=bool)
        self.mask[24:40, 20:36] = True
        self.image[self.mask] = [230, 50, 20]

    def test_translation_preserves_rgb_and_maps_center(self):
        source_center = centroid(self.mask)
        target = (source_center[0]+48, source_center[1]+32)
        rgb, alpha, info = transform_foreground(self.image, self.mask, target, 1)
        self.assertEqual(info['target_center'], list(target))
        np.testing.assert_allclose(centroid(alpha > .5), target)
        np.testing.assert_array_equal(rgb[alpha > .99], np.tile([230, 50, 20], (int((alpha > .99).sum()), 1)))
        self.assertAlmostEqual(float(alpha.sum()), float(self.mask.sum()))

    def test_scale_changes_area_and_keeps_color_at_alpha_edges(self):
        rgb, alpha, _ = transform_foreground(self.image, self.mask, (79.5, 47.5), 1.5)
        self.assertAlmostEqual(float(alpha.sum())/self.mask.sum(), 2.25, delta=.06)
        y, x = np.indices(alpha.shape)
        np.testing.assert_allclose([(x*alpha).sum()/alpha.sum(), (y*alpha).sum()/alpha.sum()], [79.5, 47.5], atol=.01)
        # Premultiplied warping must not introduce black color fringes.
        np.testing.assert_allclose(rgb[alpha > .01], np.tile([230, 50, 20], (int((alpha > .01).sum()), 1)), atol=1)

    def test_clipping_requires_explicit_choice(self):
        with self.assertRaises(ValueError):
            transform_foreground(self.image, self.mask, (1, 1), 2)
        _, alpha, info = transform_foreground(self.image, self.mask, (1, 1), 2, True)
        self.assertTrue(info['clipped'])
        self.assertGreater(alpha.sum(), 0)

    def test_repair_preserves_outside_pixels_and_object_core(self):
        rgb, alpha, _ = transform_foreground(self.image, self.mask, (79.5, 47.5), 1)
        pasted = composite(self.image, rgb, alpha)
        ring, core = harmonization_mask(alpha, 6)
        fake_diffusion = np.full_like(pasted, 255)
        final = blend_repair(pasted, fake_diffusion, ring)
        np.testing.assert_array_equal(final[~ring], pasted[~ring])
        np.testing.assert_array_equal(final[core], pasted[core])
        self.assertFalse((ring & core).any())

    def test_metric_ratio_and_relative_map_fallback(self):
        depth = np.full(self.mask.shape, 4, dtype=np.float32)
        depth[self.mask] = 2
        info = suggest_scale(depth, self.mask, (80, 50), True)
        self.assertEqual(info['scale'], .5)
        self.assertEqual(suggest_scale(depth, self.mask, (80, 50), False)['scale'], 1)

    def test_invalid_depth_and_heterogeneous_target_fall_back(self):
        bad = np.full(self.mask.shape, np.nan, dtype=np.float32)
        self.assertEqual(suggest_scale(bad, self.mask, (80, 50))['scale'], 1)
        bad.fill(2)
        bad[47:54, 77:84] = np.tile([1, 2, 3, 4, 5, 6, 7], (7, 1))
        self.assertEqual(suggest_scale(bad, self.mask, (80, 50))['scale'], 1)
        # Explicit invalid target coordinate is rejected, rather than clipped.
        with self.assertRaises(ValueError):
            suggest_scale(bad, self.mask, (128, 50))

    def test_black_alpha_scribble_is_not_empty(self):
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        rgba[2:4, 3:6, 3] = 255
        self.assertEqual(int(ink(rgba).sum()), 6)

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            transform_foreground(self.image, np.zeros_like(self.mask), (30, 30), 1)


if __name__ == '__main__':
    unittest.main()
