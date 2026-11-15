from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from structured_light.experimental import (
    ExperimentalConfig,
    classify_vector_channels,
    oam_pairs,
    oam_values,
    stokes_oam_pairs,
    wrapping_numbers,
    run_experiment,
)


class ExperimentalTests(unittest.TestCase):
    def test_all_oam_pairs_match_matlab_loop_order(self) -> None:
        pairs = oam_pairs(3, "all")

        np.testing.assert_array_equal(
            pairs,
            np.array([[-1, 1], [-1, 0], [-1, -1], [0, 1], [0, 0], [0, -1], [1, 1], [1, 0], [1, -1]]),
        )

    def test_wrapping_number_and_channel_classification(self) -> None:
        measured, wrapping = classify_vector_channels(
            np.array([[0.1, 0.2], [0.8, 0.1], [0.1, 0.7]]), oam_values(3)
        )

        np.testing.assert_array_equal(measured, np.array([0, 1]))
        self.assertEqual(wrapping, -1)
        np.testing.assert_array_equal(wrapping_numbers(np.array([[3, 0], [4, 4]])), [-3, 0])

    def test_stokes_defaults_and_simulation_mapping(self) -> None:
        self.assertEqual(stokes_oam_pairs(9).shape, (10, 2))
        config = ExperimentalConfig(
            grid_size=48, phase_screen_size=24, num_screens=2, max_epochs=1,
            backend="numpy", device="cpu",
        )
        simulation = config.simulation_config()

        self.assertEqual(simulation.learning_rate, 3e3)
        self.assertEqual(simulation.perturbation, 3.0)
        self.assertEqual(simulation.cost_weights, (10.0, 0.0, 3.0, 0.0, 2.0))
        self.assertEqual(simulation.epochs, 1)

    def test_characterisation_reuses_saved_simulation_coefficients(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            coefficient_path = Path(temporary_directory) / "phase_coefficients.npz"
            np.savez(coefficient_path, coefficients=np.zeros((1, 11**2)))
            result = run_experiment(ExperimentalConfig(
                mode="char", dimen=3, grid_size=48, phase_screen_size=24,
                num_screens=1, coefficient_path=coefficient_path,
                backend="numpy", device="cpu",
            ), verbose=False)

        self.assertIsNotNone(result.simulation)
        self.assertEqual(result.simulation.outputs.shape, (3, 48, 48))
