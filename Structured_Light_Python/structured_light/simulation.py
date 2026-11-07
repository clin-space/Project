"""Numerical model and training loop for a small structured-light D2NN."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SimulationConfig:
    """Physical and numerical settings for a simulation run."""

    grid_size: int = 96
    num_modes: int = 3
    num_screens: int = 3
    epochs: int = 10
    wavelength: float = 633e-9
    pixel_size: float = 8e-6
    focal_length: float = 500e-3
    screen_distance: float = 100e-3
    input_waist: float = 0.2e-3
    # Target output-channel diameter in meters.
    output_channel_size: float = 0.16e-3
    phase_screen_size: int | None = None
    fourier_basis_size: int = 11
    num_subharmonics: int = 0
    learning_rate: float = 0.2
    perturbation: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.grid_size < 32:
            raise ValueError("grid_size must be at least 32.")
        if self.num_modes < 1:
            raise ValueError("num_modes must be positive.")
        if self.num_modes > 9:
            raise ValueError("num_modes must not exceed 9 for this lightweight model.")
        if self.num_screens < 1:
            raise ValueError("num_screens must be positive.")
        if self.epochs < 0:
            raise ValueError("epochs cannot be negative.")
        if self.phase_screen_size is not None and not (
            1 <= self.phase_screen_size <= self.grid_size
        ):
            raise ValueError("phase_screen_size must be between 1 and grid_size.")
        if self.fourier_basis_size < 1 or self.fourier_basis_size % 2 == 0:
            raise ValueError("fourier_basis_size must be a positive odd integer.")
        effective_screen_size = self.phase_screen_size or self.grid_size // 2
        if self.fourier_basis_size > effective_screen_size:
            raise ValueError("fourier_basis_size cannot exceed the phase screen size.")
        if self.num_subharmonics < 0:
            raise ValueError("num_subharmonics cannot be negative.")
        if min(
            self.wavelength,
            self.pixel_size,
            self.focal_length,
            self.screen_distance,
            self.input_waist,
            self.output_channel_size,
            self.learning_rate,
            self.perturbation,
        ) <= 0:
            raise ValueError("Physical scales and optimizer step sizes must be positive.")


@dataclass
class SimulationResult:
    """Outputs and measurements produced by :func:`run_simulation`."""

    config: SimulationConfig
    inputs: ComplexArray
    targets: ComplexArray
    outputs: ComplexArray
    centers_yx: NDArray[np.int64]
    coefficients: ComplexArray
    history: list[dict[str, float]]

    @property
    def final_metrics(self) -> dict[str, float]:
        if self.history:
            return {key: value for key, value in self.history[-1].items() if key != "epoch"}
        return _measure(self.outputs, self.targets, self.centers_yx, self.config)["metrics"]


def _grid(config: SimulationConfig) -> tuple[FloatArray, FloatArray]:
    coordinates = (
        np.arange(config.grid_size, dtype=np.float64) - config.grid_size // 2
    ) * config.pixel_size
    x, y = np.meshgrid(coordinates, coordinates)
    return x, y

#让每个输入模式的总能量相同
def _normalize(field: ComplexArray) -> ComplexArray:
    norm = float(np.sqrt(np.sum(np.abs(field) ** 2)))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot normalize a zero or non-finite optical field.")
    return field / norm


def _laguerre_polynomial(order: int, alpha: int, value: FloatArray) -> FloatArray:
    if order == 0:
        return np.ones_like(value)
    previous = np.ones_like(value)
    current = 1 + alpha - value
    for index in range(2, order + 1):
        following = (
            (alpha + 2 * index - 1 - value) * current
            - (alpha + index - 1) * previous
        ) / index
        previous, current = current, following
    return current


def _lg_modes(config: SimulationConfig, x: FloatArray, y: FloatArray) -> ComplexArray:
    # 转化成极坐标
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    # 模式数是偶数时不包含零荷模式
    if config.num_modes % 2:
        half_width = config.num_modes // 2
        charges = range(-half_width, half_width + 1)
    else:
        half_width = config.num_modes // 2
        charges = (*range(-half_width, 0), *range(1, half_width + 1))
    fields = []
    for charge in charges:
        amplitude = (
            (np.sqrt(2) * radius / config.input_waist) ** abs(charge)
            * np.exp(-(radius / config.input_waist) ** 2)
        )
        field = amplitude * np.exp(-1j * charge * theta)
        fields.append(_normalize(field.astype(np.complex128)))
    return np.stack(fields)

#计算每个输出通道的中心位置，按蜂窝形排列
def _honeycomb_centers(num_modes: int, grid_size: int, spacing: int) -> NDArray[np.int64]:
    """Return a center-out, three-row honeycomb arrangement as (y, x) pixels."""
    left_count = num_modes // 2
    left: list[tuple[float, float]] = []
    ring = 1
    vertical_spacing = np.sqrt(3) * spacing / 2
    while len(left) < left_count:
        midpoint_x = -ring * spacing
        outer_x = midpoint_x + spacing / 2
        column = [(outer_x, -vertical_spacing), (midpoint_x, 0), (outer_x, vertical_spacing)]
        left = column[: left_count - len(left)] + left
        ring += 1

    right = [(-x, -y) for x, y in reversed(left)]
    positions = left + ([(0.0, 0.0)] if num_modes % 2 else []) + right
    center = grid_size // 2
    # MATLAB's original helper rotates its generated (x, y) coordinate pairs.
    centers = np.array(
        [(round(center + y), round(center + x)) for x, y in positions],
        dtype=np.int64,
    )
    margin = max(4, spacing // 2)
    centers[:, 0] = np.clip(centers[:, 0], margin, grid_size - 1 - margin)
    centers[:, 1] = np.clip(centers[:, 1], margin, grid_size - 1 - margin)
    return centers


def _target_modes(config: SimulationConfig, centers_yx: NDArray[np.int64]) -> ComplexArray:
    grid_size = config.grid_size
    y, x = np.indices((grid_size, grid_size), dtype=np.float64)
    waist_pixels = config.output_channel_size / (4 * config.pixel_size)
    targets = []
    for center_y, center_x in centers_yx:
        field = np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / waist_pixels**2
        ).astype(np.complex128)
        targets.append(_normalize(field))
    return np.stack(targets)


def _fourier_coefficient_count(config: SimulationConfig) -> int:
    """Return the number of entries in the centered MATLAB-style coefficient grid."""
    return config.fourier_basis_size**2


def _phase_screens(
    coefficients: ComplexArray,
    config: SimulationConfig,
    subharmonic_coefficients: ComplexArray,
) -> ComplexArray:
    """Build unit-amplitude phase screens from centered Fourier coefficients."""
    grid_size = config.grid_size
    screen_size = config.phase_screen_size or grid_size // 2
    basis_size = config.fourier_basis_size
    if subharmonic_coefficients.shape != (config.num_subharmonics, 9):
        raise ValueError(
            "subharmonic_coefficients must have shape "
            f"({config.num_subharmonics}, 9)."
        )
    center = screen_size // 2
    half_basis = basis_size // 2
    insertion = slice(center - half_basis, center + half_basis + 1)

    #准备物理坐标
    #用 config.pixel_size 把像素索引换算成米
    pixel = (
        np.arange(screen_size, dtype=np.float64) - screen_size // 2
    ) * config.pixel_size
    x, y = np.meshgrid(pixel, pixel)
    screens = np.ones(
        (config.num_screens, grid_size, grid_size), dtype=np.complex128
    )
    grid_center = grid_size // 2
    grid_insertion = slice(
        grid_center - screen_size // 2,
        grid_center - screen_size // 2 + screen_size,
    )

    for screen_index, coefficients_for_screen in enumerate(coefficients):
        #中心位置是零频（DC）分量，代码把它设为 0
        spectrum = np.zeros((screen_size, screen_size), dtype=np.complex128)
        coefficient_grid = coefficients_for_screen.reshape(
            (basis_size, basis_size), order="F"
        ).copy()
        coefficient_grid[half_basis, half_basis] = 0
        #将 11×11 系数放到较大的屏幕频谱数组中心
        spectrum[insertion, insertion] = coefficient_grid
        #ifft2 把频谱转换成空间分布；前后的 ifftshift 用于调整频谱中心与 FFT 的排列方式
        phase = (
            np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))
            * basis_size**2
        )

        #训练时是否添加相位载频
        if config.num_subharmonics:
            length = basis_size * config.pixel_size
            base_frequency_spacing = 1 / length
            frequencies = np.array([-1.0, 0.0, 1.0]) * base_frequency_spacing
            frequency_x, frequency_y = np.meshgrid(frequencies, frequencies)
            for subharmonic_index, coefficient_vector in enumerate(
                subharmonic_coefficients
            ):
                level = subharmonic_index + 1
                frequency_scale = 3**level
                low_coefficients = coefficient_vector.reshape(
                    (3, 3), order="F"
                ).copy()
                low_coefficients[1, 1] = 0
                low_coefficients = low_coefficients / frequency_scale
                phase += np.sum(
                    low_coefficients[:, :, None, None]
                    * np.exp(
                        2j
                        * np.pi
                        * (
                            frequency_x[:, :, None, None] * x / frequency_scale
                            + frequency_y[:, :, None, None] * y / frequency_scale
                        )
                    ),
                    axis=(0, 1),
                )

        phase = np.real(phase)
        phase -= np.mean(phase)
        local_transmission = np.exp(1j * phase)
        screens[screen_index, grid_insertion, grid_insertion] = local_transmission
    return screens


def _transfer_function(config: SimulationConfig, distance: float) -> ComplexArray:
    frequency = np.fft.fftfreq(config.grid_size, d=config.pixel_size)
    fx, fy = np.meshgrid(frequency, frequency)
    wave_number = 2 * np.pi / config.wavelength
    argument = 1 - config.wavelength**2 * (fx**2 + fy**2)
    return np.exp(1j * wave_number * distance * np.sqrt(argument.astype(np.complex128)))


def _propagate(field: ComplexArray, transfer: ComplexArray) -> ComplexArray:
    shifted = np.fft.fftshift(field, axes=(-2, -1))
    spectrum = np.fft.fft2(shifted, axes=(-2, -1))
    result = np.fft.ifft2(spectrum * transfer, axes=(-2, -1))
    return np.fft.ifftshift(result, axes=(-2, -1))


def _lens_propagate(field: ComplexArray, config: SimulationConfig, x: FloatArray, y: FloatArray) -> ComplexArray:
    transfer = _transfer_function(config, config.focal_length)
    lens = np.exp(
        -1j
        * (2 * np.pi / config.wavelength)
        * (x**2 + y**2)
        / (2 * config.focal_length)
    )
    return _propagate(_propagate(field, transfer) * lens, transfer)


def _forward(
    inputs: ComplexArray,
    coefficients: ComplexArray,
    subharmonic_coefficients: ComplexArray,
    config: SimulationConfig,
    x: FloatArray,
    y: FloatArray,
) -> ComplexArray:
    field = _lens_propagate(inputs, config, x, y)
    screens = _phase_screens(coefficients, config, subharmonic_coefficients)
    between_screens = _transfer_function(config, config.screen_distance)
    for index, screen in enumerate(screens):
        field = field * screen
        if index < len(screens) - 1:
            field = _propagate(field, between_screens)
    return _lens_propagate(field, config, x, y)


def _measure(
    outputs: ComplexArray,
    targets: ComplexArray,
    centers_yx: NDArray[np.int64],
    config: SimulationConfig,
) -> dict[str, Any]:
    output_intensity = np.abs(outputs) ** 2
    target_intensity = np.abs(targets) ** 2
    output_vectors = output_intensity.reshape(config.num_modes, -1)
    target_vectors = target_intensity.reshape(config.num_modes, -1)

    overlap = output_vectors @ target_vectors.T
    visibility_matrix = overlap**2
    column_sums = visibility_matrix.sum(axis=0, keepdims=True)
    visibility_matrix = np.divide(
        visibility_matrix,
        column_sums,
        out=np.zeros_like(visibility_matrix),
        where=column_sums > 0,
    )
    visibility = float(np.mean(np.diag(visibility_matrix)))

    yy, xx = np.indices((config.grid_size, config.grid_size))
    radius = config.output_channel_size / (2 * config.pixel_size)
    masks = np.stack(
        [
            (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
            for center_y, center_x in centers_yx
        ]
    )
    channel_energy = np.einsum("iyx,jyx->ij", output_intensity, masks, optimize=True)
    channel_totals = channel_energy.sum(axis=1, keepdims=True)
    channel_matrix = np.divide(
        channel_energy,
        channel_totals,
        out=np.zeros_like(channel_energy),
        where=channel_totals > 0,
    )
    channel_accuracy = float(np.mean(np.diag(channel_matrix)))
    efficiency = float(np.mean(np.sum(output_intensity * masks, axis=(1, 2))))

    mse = float(np.mean(np.sum((output_intensity - target_intensity) ** 2, axis=(1, 2))))
    output_energy = float(np.sum(output_intensity))
    # Every input LG mode is normalized to unit energy before propagation.
    loss = max(0.0, 1 - output_energy / config.num_modes)
    weights = np.array([1.0, 0.0, 3.0, 2.0, 3.0])
    metrics = {
        "mse": mse,
        "visibility": visibility,
        "channel_accuracy": channel_accuracy,
        "efficiency": efficiency,
        "loss": loss,
        "cost": float(
            np.dot(weights, [mse, 1 - channel_accuracy, 1 - visibility, loss, 1 - efficiency])
            / weights.sum()
        ),
    }
    return {"metrics": metrics, "visibility_matrix": visibility_matrix, "channel_matrix": channel_matrix}


def run_simulation(
    config: SimulationConfig | None = None,
    *,
    verbose: bool = True,
) -> SimulationResult:
    """Run SPSA training and return the final complex output fields."""
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)
    x, y = _grid(config)
    inputs = _lg_modes(config, x, y)
    # 几个目标光斑的中心要隔多少个像素
    spacing = min(
        int(np.ceil(config.output_channel_size / config.pixel_size)),
        max(8, config.grid_size // 3),
    )
    centers_yx = _honeycomb_centers(config.num_modes, config.grid_size, spacing)
    #在这些中心位置生成目标高斯光斑
    targets = _target_modes(config, centers_yx)
    #基函数权重
    coefficient_count = _fourier_coefficient_count(config)
    coefficients = np.zeros(
        (config.num_screens, coefficient_count), dtype=np.complex128
    )
    if config.num_subharmonics:
        subharmonic_coefficients = (
            rng.standard_normal((config.num_subharmonics, 9))
            + 1j * rng.standard_normal((config.num_subharmonics, 9))
        )
    else:
        subharmonic_coefficients = np.empty((0, 9), dtype=np.complex128)
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        direction = rng.choice(np.array([-1.0, 1.0]), size=coefficients.shape)
        direction = direction + 1j * rng.choice(
            np.array([-1.0, 1.0]), size=coefficients.shape
        )
        plus_coefficients = coefficients + config.perturbation * direction
        minus_coefficients = coefficients - config.perturbation * direction
        plus_outputs = _forward(
            inputs, plus_coefficients, subharmonic_coefficients, config, x, y
        )
        minus_outputs = _forward(
            inputs, minus_coefficients, subharmonic_coefficients, config, x, y
        )
        plus_cost = _measure(plus_outputs, targets, centers_yx, config)["metrics"]["cost"]
        minus_cost = _measure(minus_outputs, targets, centers_yx, config)["metrics"]["cost"]

        gradient = ((plus_cost - minus_cost) / (2 * config.perturbation)) * direction
        step_size = config.learning_rate / ((epoch + 1) ** 0.602)
        coefficients -= step_size * gradient

        outputs = _forward(
            inputs, coefficients, subharmonic_coefficients, config, x, y
        )
        metrics = _measure(outputs, targets, centers_yx, config)["metrics"]
        history.append({"epoch": float(epoch + 1), **metrics})
        if verbose:
            print(
                f"Epoch {epoch + 1:03d}/{config.epochs:03d} "
                f"cost={metrics['cost']:.5f} "
                f"visibility={metrics['visibility']:.4f} "
                f"channel_accuracy={metrics['channel_accuracy']:.4f} "
                f"efficiency={metrics['efficiency']:.4f}"
            )

    final_outputs = _forward(
        inputs, coefficients, subharmonic_coefficients, config, x, y
    )
    if config.epochs == 0:
        history.append({"epoch": 0.0, **_measure(final_outputs, targets, centers_yx, config)["metrics"]})
    return SimulationResult(
        config=config,
        inputs=inputs,
        targets=targets,
        outputs=final_outputs,
        centers_yx=centers_yx,
        coefficients=coefficients,
        history=history,
    )


def save_result(result: SimulationResult, output_dir: Path | str) -> None:
    """Write reproducible numeric outputs and a compact diagnostic plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        if result.history:
            writer = csv.DictWriter(file, fieldnames=list(result.history[0]))
            writer.writeheader()
            writer.writerows(result.history)

    summary = {
        "config": asdict(result.config),
        "final_metrics": result.final_metrics,
        "channel_centers_yx": result.centers_yx.tolist(),
    }
    with (output_path / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    np.savez_compressed(
        output_path / "phase_coefficients.npz",
        coefficients=result.coefficients,
        centers_yx=result.centers_yx,
    )
    _save_plot(result, output_path / "simulation.png")


def _save_plot(result: SimulationResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = result.config.num_modes
    figure, axes = plt.subplots(count + 1, 3, figsize=(10, max(4, (count + 1) * 2.4)), squeeze=False)
    for index in range(count):
        items = (
            (np.abs(result.inputs[index]) ** 2, "Input"),
            (np.abs(result.targets[index]) ** 2, "Target"),
            (np.abs(result.outputs[index]) ** 2, "Output"),
        )
        for axis, (intensity, title) in zip(axes[index], items):
            axis.imshow(intensity, cmap="inferno")
            axis.set_title(f"Mode {index + 1}: {title}")
            axis.set_axis_off()

    training_axis = axes[-1, 0]
    if result.history:
        training_axis.plot(
            [row["epoch"] for row in result.history],
            [row["cost"] for row in result.history],
        )
    training_axis.set_title("SPSA training cost")
    training_axis.set_xlabel("Epoch")
    training_axis.set_ylabel("Cost")
    axes[-1, 1].set_axis_off()
    axes[-1, 2].set_axis_off()

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
