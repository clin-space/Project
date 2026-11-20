from __future__ import annotations

import unittest

import numpy as np

from superpixel import (
    fourier_mask,
    lee_method,
    phase_and_amplitude_to_dmdpixels_lee,
    phase_and_amplitude_to_dmdpixels_lookup_table,
    rescale_target_to_superpixel_resolution,
    spatial_filter,
    superpixel_method,
)


class SuperPixelTests(unittest.TestCase):
    def test_fourier_masks_preserve_matlab_one_based_center(self):
        # The even-sized MATLAB grid is centred at index (ny/2, nx/2) in Python.
        circle = fourier_mask(8, 8, 4, "circle")
        self.assertEqual(circle[4, 4], 1)
        self.assertEqual(circle[2, 4], 0)  # strict '<' aperture edge
        self.assertEqual(fourier_mask(8, 8, 4, "square1")[4, 4], 1)

    def test_spatial_filter_of_constant_is_constant(self):
        field = np.ones((16, 16), dtype=complex)
        np.testing.assert_allclose(spatial_filter(field, 8), field, atol=1e-12)

    def test_lee_encoder_has_binary_amplitude_and_expected_shape(self):
        field = np.exp(1j * np.arange(64).reshape(8, 8) / 7)
        dmd = phase_and_amplitude_to_dmdpixels_lee(field, 1 / 12, 1 / 12)
        self.assertEqual(dmd.shape, field.shape)
        self.assertTrue(np.all(np.isclose(np.abs(dmd), 0) | np.isclose(np.abs(dmd), 1)))

    def test_lookup_encoder_has_binary_amplitude_and_4x_shape(self):
        field = np.array([[0, 0.5j], [-0.4 + 0.2j, -0.8j]], dtype=complex)
        dmd = phase_and_amplitude_to_dmdpixels_lookup_table(field)
        self.assertEqual(dmd.shape, (8, 8))
        self.assertTrue(np.all(np.isclose(np.abs(dmd), 0) | np.isclose(np.abs(dmd), 1)))

    def test_rescale_and_end_to_end_methods(self):
        y, x = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        field = (1 + 0.2 * np.cos(x)) * np.exp(1j * (x + y) / 9)
        self.assertEqual(rescale_target_to_superpixel_resolution(field).shape, (4, 4))
        superpixel = superpixel_method(field, 8, "circle")
        lee = lee_method(field, 8, "circle", 1 / 12, 1 / 12)
        self.assertEqual(superpixel[0].shape, field.shape)
        self.assertEqual(lee[0].shape, field.shape)
        for result in (superpixel, lee):
            self.assertTrue(np.isfinite(result[2]))
            self.assertTrue(np.isfinite(result[3]))
            self.assertGreaterEqual(result[2], 0)
            self.assertLessEqual(result[2], 1 + 1e-12)


if __name__ == "__main__":
    unittest.main()
