from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from structured_light import SimulationConfig, run_simulation
from structured_light.simulation import (
    _phase_screens,
    _propagate,
    _transfer_function,
    save_result,
)


class SimulationTests(unittest.TestCase):
    def test_free_space_propagation_preserves_energy(self) -> None:
        config = SimulationConfig(grid_size=48, epochs=0)
        rng = np.random.default_rng(4)
        field = rng.normal(size=(48, 48)) + 1j * rng.normal(size=(48, 48))
        transfer = _transfer_function(config, config.screen_distance)

        propagated = _propagate(field, transfer)

        self.assertAlmostEqual(
            float(np.sum(np.abs(field) ** 2)),
            float(np.sum(np.abs(propagated) ** 2)),
            places=9,
        )

    def test_small_training_run_and_saved_outputs(self) -> None:
        config = SimulationConfig(grid_size=48, epochs=1, num_screens=2, seed=2)
        result = run_simulation(config, verbose=False)

        self.assertEqual(result.inputs.shape, (3, 48, 48))
        self.assertEqual(result.targets.shape, (3, 48, 48))
        self.assertEqual(result.outputs.shape, (3, 48, 48))
        self.assertEqual(result.coefficients.shape, (2, 11**2))
        self.assertTrue(np.iscomplexobj(result.coefficients))
        self.assertEqual(len(result.history), 1)
        self.assertTrue(np.isfinite(result.final_metrics["cost"]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory)
            save_result(result, output_path)
            self.assertTrue((output_path / "metrics.csv").is_file())
            self.assertTrue((output_path / "phase_coefficients.npz").is_file())
            self.assertTrue((output_path / "simulation.png").is_file())
            summary = json.loads((output_path / "summary.json").read_text())
            self.assertIn("visibility", summary["final_metrics"])

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SimulationConfig(grid_size=16)

    def test_matlab_style_phase_screen_is_centered_and_unit_amplitude(self) -> None:
        config = SimulationConfig(
            grid_size=48,
            phase_screen_size=24,
            num_screens=2,
            fourier_basis_size=11,
            epochs=0,
        )
        coefficients = np.zeros((2, 11**2), dtype=np.complex128)
        coefficients[0, 5] = 1 + 0.5j

        screens = _phase_screens(
            coefficients,
            config,
            np.empty((0, 9), dtype=np.complex128),
        )

        self.assertEqual(screens.shape, (2, 48, 48))
        np.testing.assert_allclose(np.abs(screens), 1.0, atol=1e-12)
        np.testing.assert_array_equal(screens[1], np.ones((48, 48)))
        np.testing.assert_array_equal(screens[0, :12], np.ones((12, 48)))
        self.assertGreater(float(np.max(np.abs(screens[0] - 1))), 0)

    def test_subharmonics_run_with_physical_pixel_grid(self) -> None:
        config = SimulationConfig(
            grid_size=48,
            phase_screen_size=24,
            num_screens=1,
            num_subharmonics=1,
            epochs=0,
        )
        result = run_simulation(config, verbose=False)

        self.assertTrue(np.isfinite(result.final_metrics["cost"]))


if __name__ == "__main__":
    unittest.main()
