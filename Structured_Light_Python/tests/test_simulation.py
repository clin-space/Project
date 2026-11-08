from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from structured_light import SimulationConfig, run_simulation
from structured_light.simulation import (
    _adjust_cost_weights,
    _draw_output_channels,
    _forward,
    _honeycomb_centers,
    _measure,
    _phase_screens,
    _propagate,
    _target_mode_width,
    _target_modes,
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
                    _target_mode_width(config) / 2,
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

    def test_matlab_style_phase_screen_is_zero_outside_aperture(self) -> None:
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
        np.testing.assert_allclose(np.abs(screens[:, 12:36, 12:36]), 1.0, atol=1e-12)
        np.testing.assert_array_equal(screens[1, 12:36, 12:36], np.ones((24, 24)))
        np.testing.assert_array_equal(screens[:, :12], np.zeros((2, 12, 48)))
        self.assertGreater(float(np.max(np.abs(screens[0, 12:36, 12:36] - 1))), 0)

    def test_honeycomb_centers_match_matlab_axis_order_and_floor(self) -> None:
        centers = _honeycomb_centers(3, 200, 63)

        np.testing.assert_array_equal(
            centers, np.array([[68, 45], [100, 100], [131, 45]])
        )

    def test_matlab_target_mode_has_finite_support_and_sampled_waist(self) -> None:
        config = SimulationConfig(grid_size=96, num_modes=1, backend="numpy")
        target = _target_modes(config, np.array([[48, 48]]))[0]
        width = config.output_channel_size / config.pixel_size
        axis = np.linspace(-width / 2, width / 2 - 1, 62)
        x, y = np.meshgrid(axis, axis)
        expected = np.exp(-(x**2 + y**2) / (width / 4) ** 2)
        expected /= np.linalg.norm(expected)

        self.assertEqual(_target_mode_width(config), 62)
        self.assertEqual(np.count_nonzero(target), 62**2)
        np.testing.assert_allclose(target[17:79, 17:79], expected, atol=1e-15)

    def test_adjusted_cost_weights_cannot_be_negative(self) -> None:
        base_weights = np.array([1.0, 0.0, 3.0, 2.0, 3.0])
        metrics = {
            "channel_accuracy": 0.9,
            "visibility": 0.95,
            "loss": 0.003,
            "efficiency": 0.776,
        }

        adjusted = _adjust_cost_weights(base_weights, metrics, reset_count=1)

        self.assertEqual(adjusted[0], 0.0)
        self.assertTrue(np.all(adjusted >= 0))
        self.assertGreater(adjusted.sum(), 0)

    def test_efficiency_uses_all_inputs_per_channel(self) -> None:
        config = SimulationConfig(
            grid_size=48,
            num_modes=2,
            output_channel_size=4 * 8e-6,
            backend="numpy",
        )
        centers = np.array([[12, 12], [36, 36]])
        outputs = np.zeros((2, 48, 48), dtype=np.complex128)
        outputs[0, 12, 12] = np.sqrt(2)
        outputs[0, 36, 36] = np.sqrt(3)
        outputs[1, 12, 12] = np.sqrt(5)
        outputs[1, 36, 36] = np.sqrt(7)
        targets = np.zeros_like(outputs)

        metrics = _measure(outputs, targets, centers, config)["metrics"]

        self.assertAlmostEqual(metrics["efficiency"], ((2 + 5) / 5 + (3 + 7) / 12) / 2)
        self.assertAlmostEqual(metrics["channel_accuracy"], (2 / 7 + 7 / 10) / 2)

    def test_visibility_normalizes_each_input_mode(self) -> None:
        config = SimulationConfig(grid_size=32, num_modes=3, backend="numpy")
        centers = np.array([[5, 5], [16, 16], [26, 26]])
        intensities = np.array([[2.0, 10.0, 1.0], [1.0, 10.0, 1.0], [1.0, 10.0, 2.0]])
        outputs = np.zeros((3, 32, 32), dtype=np.complex128)
        targets = np.zeros_like(outputs)
        for mode_index in range(3):
            for channel_index, (y, x) in enumerate(centers):
                outputs[mode_index, y, x] = np.sqrt(intensities[mode_index, channel_index])
            targets[mode_index, *centers[mode_index]] = 1

        measured = _measure(outputs, targets, centers, config)
        expected = intensities.T**2
        expected /= expected.sum(axis=0, keepdims=True)

        np.testing.assert_allclose(measured["visibility_matrix"], expected)
        self.assertAlmostEqual(measured["metrics"]["visibility"], np.trace(expected) / 3)

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
            _propagate(
                inputs
                * _phase_screens(
                    coefficients,
                    config,
                    np.empty((0, 9), dtype=np.complex128),
                )[0],
                _transfer_function(config, config.focal_length),
            ),
            _transfer_function(config, config.focal_length),
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
