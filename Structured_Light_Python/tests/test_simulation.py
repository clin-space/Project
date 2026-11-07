from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from structured_light import SimulationConfig, run_simulation
from structured_light.simulation import (
    _draw_output_channels,
    _forward,
    _phase_screens,
    _propagate,
    _transfer_function,
    save_result,
)


class SimulationTests(unittest.TestCase):
    def test_free_space_propagation_preserves_energy(self) -> None:
        config = SimulationConfig(grid_size=48, epochs=0, backend="numpy")
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
        config = SimulationConfig(
            grid_size=48, epochs=1, num_screens=2, seed=2, backend="numpy"
        )
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

    def test_output_channel_overlay_matches_measurement_regions(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        config = SimulationConfig(grid_size=48, backend="numpy")
        centers = np.array([[12, 16], [24, 24], [36, 32]])
        figure, axis = plt.subplots()
        try:
            _draw_output_channels(axis, centers, config, plt.get_cmap("tab10"))

            self.assertEqual(len(axis.patches), len(centers))
            self.assertEqual(len(axis.texts), len(centers))
            for patch, (center_y, center_x) in zip(axis.patches, centers):
                self.assertEqual(patch.center, (center_x, center_y))
                self.assertEqual(
                    patch.radius,
                    config.output_channel_size / (2 * config.pixel_size),
                )
        finally:
            plt.close(figure)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional.")
    def test_default_backend_matches_numpy_without_training(self) -> None:
        config = SimulationConfig(
            grid_size=48,
            phase_screen_size=24,
            num_screens=2,
            epochs=0,
            device="cpu",
        )

        torch_result = run_simulation(config, verbose=False)
        numpy_result = run_simulation(
            SimulationConfig(
                grid_size=48,
                phase_screen_size=24,
                num_screens=2,
                epochs=0,
                backend="numpy",
            ),
            verbose=False,
        )

        np.testing.assert_allclose(torch_result.outputs, numpy_result.outputs, atol=1e-12)
        self.assertAlmostEqual(
            torch_result.final_metrics["cost"],
            numpy_result.final_metrics["cost"],
            places=12,
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional.")
    def test_torch_backend_runs_training_with_subharmonics(self) -> None:
        result = run_simulation(
            SimulationConfig(
                grid_size=32,
                phase_screen_size=16,
                num_screens=1,
                num_subharmonics=1,
                epochs=1,
                backend="torch",
                device="cpu",
            ),
            verbose=False,
        )

        self.assertEqual(len(result.history), 1)
        self.assertTrue(np.isfinite(result.final_metrics["cost"]))

    def test_invalid_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SimulationConfig(backend="jax")

    def test_numpy_backend_does_not_require_torch_device(self) -> None:
        config = SimulationConfig(backend="numpy")

        self.assertEqual(config.device, "cuda:0")

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
            backend="numpy",
        )
        result = run_simulation(config, verbose=False)

        self.assertTrue(np.isfinite(result.final_metrics["cost"]))

    def test_matlab_default_training_parameters(self) -> None:
        config = SimulationConfig()

        self.assertEqual(config.grid_size, 600)
        self.assertEqual(config.phase_screen_size, 300)
        self.assertEqual(config.num_screens, 5)
        self.assertEqual(config.epochs, 5000)
        self.assertEqual(config.input_waist, 0.4e-3)
        self.assertEqual(config.output_channel_size, 0.5e-3)
        self.assertEqual(config.learning_rate, 5e3)
        self.assertEqual(config.perturbation, 0.1)
        self.assertEqual(config.learning_rate_exponent, 0.18)
        self.assertEqual(config.perturbation_exponent, 0.101)
        self.assertEqual(config.failsafe_check, 10)
        self.assertEqual(config.failsafe_tolerance, 2e-4)
        self.assertEqual(config.backend, "torch")
        self.assertEqual(config.device, "cuda:0")

    def test_failsafe_restores_best_coefficients(self) -> None:
        config = SimulationConfig(
            grid_size=48,
            num_screens=1,
            epochs=1,
            learning_rate=1e-30,
            failsafe_check=1,
            seed=3,
            backend="numpy",
        )

        result = run_simulation(config, verbose=False)

        np.testing.assert_array_equal(result.coefficients, np.zeros_like(result.coefficients))

    def test_forward_starts_from_input_without_prelens(self) -> None:
        config = SimulationConfig(
            grid_size=48,
            phase_screen_size=24,
            num_screens=1,
            epochs=0,
            backend="numpy",
        )
        rng = np.random.default_rng(7)
        inputs = rng.normal(size=(1, 48, 48)) + 1j * rng.normal(size=(1, 48, 48))
        x = np.zeros((48, 48))
        y = np.zeros((48, 48))
        coefficients = np.zeros((1, config.fourier_basis_size**2), dtype=np.complex128)

        actual = _forward(
            inputs,
            coefficients,
            np.empty((0, 9), dtype=np.complex128),
            config,
            x,
            y,
        )
        expected = _propagate(
            _propagate(inputs, _transfer_function(config, config.focal_length)),
            _transfer_function(config, config.focal_length),
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
