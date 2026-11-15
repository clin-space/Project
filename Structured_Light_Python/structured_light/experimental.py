"""Experiment-facing helpers translated from ``D2NN_Experimental.m``.

The MATLAB entry point combines two quite different jobs: it configures an
experiment and talks to two SLMs plus a Thorlabs camera.  The reusable
numerical D2NN part already lives in :mod:`structured_light.simulation`, so
this module deliberately imports it instead of maintaining a second SPSA
implementation.

Hardware-specific calls (``sl.Fullscreen`` and the Thorlabs SDK) are not
portable Python APIs.  They are represented by :class:`ExperimentalHardware`;
an installation-specific adapter can implement that protocol without changing
the mode-set, coefficient, or measurement logic below.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .simulation import (
    SimulationConfig,
    SimulationResult,
    evaluate_coefficients,
    run_simulation,
    save_result,
)

ExperimentMode = Literal["align", "char", "train", "measure", "stokes"]
InputMode = Literal["LG", "vortex", "HG", "hadamard"]
ScreenType = Literal["fourier", "zernike"]
RangeSelection = Literal["single", "select", "all"]

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@runtime_checkable
class ExperimentalHardware(Protocol):
    """Minimal adapter required to execute a real optical experiment.

    ``input_hologram`` and ``d2nn_hologram`` are deliberately left as NumPy
    arrays: their pixel encoding and display orientation depend on the SLM
    model and calibration.  ``D2NN_Experimental.m`` obtains those details
    from MATLAB packages ``sl``, ``lsm`` and ``cam``; their Python drivers are
    not present in this repository.
    """

    def show_input(self, input_hologram: NDArray[np.generic]) -> None:
        """Display an input hologram on SLM 1."""

    def show_d2nn(self, d2nn_hologram: NDArray[np.generic]) -> None:
        """Display a D2NN hologram on SLM 2."""

    def capture(self) -> FloatArray:
        """Capture and return one camera frame as a two-dimensional array."""

    def close(self) -> None:
        """Release camera/SLM resources."""


@dataclass(frozen=True)
class ExperimentalConfig:
    """Settings at the top of MATLAB's ``D2NN_Experimental.m``.

    ``grid_size`` is for the offline numerical surrogate.  Physical SLM sizes,
    rotations, gratings and aberration maps belong in a hardware adapter,
    because they are calibration-specific rather than properties of a D2NN.
    """

    mode: ExperimentMode = "train"
    input_mode: InputMode = "LG"
    dimen: int = 9
    input_type: Literal[1, 2] = 2
    switch_input: Literal[1, 2] = 1
    screen_type: ScreenType = "fourier"
    num_screens: int = 5
    output_type: int = 5
    max_epochs: int = 10_000
    eps: float = 5e-4
    cost_weights: tuple[float, float, float] = (10.0, 3.0, 2.0)
    gamma: tuple[float, float] = (0.2, 0.101)
    failsafe_check: int = 5
    grid_size: int = 600
    phase_screen_size: int | None = None
    output_channel_size: float = 1e-3
    coefficient_path: Path | str | None = None
    output_dir: Path | str | None = None
    seed: int = 0
    backend: Literal["numpy", "torch"] = "torch"
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.dimen < 1:
            raise ValueError("dimen must be positive.")
        if self.num_screens < 1 or self.max_epochs < 0:
            raise ValueError("num_screens must be positive and max_epochs non-negative.")
        if self.output_type not in (1, 2, 3, 4, 5):
            raise ValueError("output_type must be an integer from 1 to 5.")
        if self.output_type != 5:
            raise NotImplementedError(
                "The reusable Python simulator currently implements the MATLAB "
                "honeycomb output layout (output_type=5)."
            )
        if self.screen_type != "fourier":
            raise NotImplementedError(
                "The reusable Python simulator currently implements the Fourier "
                "phase-screen basis; MATLAB Zernike screen generation needs a "
                "separate Python implementation."
            )
        if self.input_mode != "LG":
            raise NotImplementedError(
                "The reusable Python simulator currently implements LG inputs."
            )
        if self.dimen > 9:
            raise ValueError("dimen cannot exceed simulation.py's supported nine LG modes.")

    def simulation_config(self, *, epochs: int | None = None) -> SimulationConfig:
        """Translate shared experimental values to ``SimulationConfig``.

        This mirrors the experimental Fourier SPSA defaults ``a=3000`` and
        ``b=3`` for LG inputs.  It is intentionally a mapping, not a copied
        training loop.
        """
        return SimulationConfig(
            grid_size=self.grid_size,
            phase_screen_size=self.phase_screen_size,
            num_modes=self.dimen,
            num_screens=self.num_screens,
            epochs=self.max_epochs if epochs is None else epochs,
            output_channel_size=self.output_channel_size,
            learning_rate=3e3,
            perturbation=3.0,
            learning_rate_exponent=self.gamma[0],
            perturbation_exponent=self.gamma[1],
            failsafe_check=self.failsafe_check,
            failsafe_tolerance=self.eps,
            # MATLAB experimental objective: [MSE, 1-visibility, 1-efficiency].
            # The shared simulator has two extra terms, held at zero here.
            cost_weights=(
                self.cost_weights[0], 0.0, self.cost_weights[1], 0.0,
                self.cost_weights[2],
            ),
            seed=self.seed,
            backend=self.backend,
            device=self.device,
        )


@dataclass
class ExperimentResult:
    """Result of the portable train/characterise portion of an experiment."""

    config: ExperimentalConfig
    simulation: SimulationResult | None = None
    measured_oam: IntArray | None = None
    wrapping_number: int | None = None


def oam_values(dimen: int) -> IntArray:
    """Return the MATLAB OAM range ``-floor(dimen/2):floor(dimen/2)``.

    For an even dimension MATLAB's expression has ``dimen + 1`` values.  The
    experimental script is normally used with odd ``dimen=9``; rejecting an
    even value prevents a silent dimension mismatch with the output channels.
    """
    if dimen < 1 or dimen % 2 == 0:
        raise ValueError("dimen must be a positive odd number for experimental OAM sets.")
    half = dimen // 2
    return np.arange(-half, half + 1, dtype=np.int64)


def oam_pairs(
    dimen: int,
    selection: RangeSelection = "all",
    *,
    single: tuple[int, int] = (3, -3),
    selected: tuple[tuple[int, int], ...] = (
        (0, 1), (0, 2), (3, 0), (1, 2), (3, 1), (1, 3),
    ),
) -> IntArray:
    """Create the ``measure`` OAM pairs from the corresponding MATLAB switch."""
    if selection == "single":
        return np.asarray([single], dtype=np.int64)
    if selection == "select":
        return np.asarray(selected, dtype=np.int64)
    charges = oam_values(dimen)
    # MATLAB loops l1 ascending and l2 descending.
    return np.asarray([(left, right) for left in charges for right in charges[::-1]], dtype=np.int64)


def stokes_oam_pairs(
    dimen: int,
    selection: RangeSelection = "select",
) -> IntArray:
    """Return the OAM selections hard-coded in MATLAB's ``stokes`` case."""
    selected = (
        (0, 1), (0, 2), (3, 0), (1, 2), (4, 2), (-1, -3),
        (-4, -1), (-4, 2), (-3, -2), (3, -3),
    )
    return oam_pairs(dimen, selection, single=(4, 4), selected=selected)


def wrapping_numbers(pairs: IntArray) -> IntArray:
    """Calculate MATLAB's target Skyrmion wrapping number for each OAM pair."""
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (number_of_states, 2).")
    return np.sign(np.abs(pairs[:, 0]) - np.abs(pairs[:, 1])).astype(np.int64) * (
        pairs[:, 1] - pairs[:, 0]
    )


def classify_vector_channels(
    channel_visibilities: FloatArray,
    input_charges: IntArray,
) -> tuple[IntArray, int]:
    """Decode the two polarisation channels as ``measure_skyrm`` does.

    ``channel_visibilities`` has shape ``(dimen, 2)``.  Each column is
    normalised before its maximum is used, matching MATLAB's per-channel
    visibility normalisation.  The returned OAM pair and wrapping number are
    independent of a particular camera SDK.
    """
    visibility = np.asarray(channel_visibilities, dtype=np.float64)
    charges = np.asarray(input_charges, dtype=np.int64)
    if visibility.shape != (len(charges), 2):
        raise ValueError("channel_visibilities must have shape (len(input_charges), 2).")
    if np.any(visibility < 0):
        raise ValueError("channel_visibilities cannot contain negative values.")
    totals = visibility.sum(axis=0)
    if np.any(totals == 0):
        raise ValueError("each polarisation channel needs positive total visibility.")
    normalized = visibility / totals
    measured = charges[np.argmax(normalized, axis=0)]
    return measured, int(wrapping_numbers(measured[np.newaxis, :])[0])


def load_coefficients(path: Path | str) -> NDArray[np.complex128]:
    """Load coefficients exported by :func:`structured_light.save_result`.

    Native MATLAB ``stats.mat`` files require ``scipy.io.loadmat`` and may use
    a different array layout, so they are intentionally not guessed here.
    Export the coefficient tensor to NPZ (key ``coefficients``) first.
    """
    coefficient_path = Path(path)
    if coefficient_path.suffix.lower() != ".npz":
        raise ValueError("Only .npz coefficient files are portable in this package.")
    with np.load(coefficient_path, allow_pickle=False) as saved:
        if "coefficients" not in saved:
            raise ValueError(f"{coefficient_path} does not contain a 'coefficients' array.")
        return np.asarray(saved["coefficients"], dtype=np.complex128)


def run_experiment(
    config: ExperimentalConfig | None = None,
    *,
    verbose: bool = True,
) -> ExperimentResult:
    """Run portable train/characterise operations from ``D2NN_Experimental``.

    ``train`` delegates to :func:`run_simulation`.  ``char`` evaluates loaded
    coefficients without retraining.  ``align``, ``measure`` and ``stokes``
    require an installation-specific hardware adapter and are therefore kept
    as explicit, non-silent boundaries rather than simulated camera data.
    """
    config = config or ExperimentalConfig()
    if config.mode == "train":
        result = ExperimentResult(config=config, simulation=run_simulation(
            config.simulation_config(), verbose=verbose
        ))
        _save_experiment_result(result)
        return result
    if config.mode == "char":
        if config.coefficient_path is None:
            raise ValueError("char mode requires coefficient_path pointing to phase_coefficients.npz.")
        result = ExperimentResult(
            config=config,
            simulation=evaluate_coefficients(
                config.simulation_config(epochs=0), load_coefficients(config.coefficient_path)
            ),
        )
        _save_experiment_result(result)
        return result
    raise NotImplementedError(
        f"{config.mode!r} mode needs calibrated SLM/camera drivers. "
        "Implement ExperimentalHardware for this optical bench, then connect "
        "its hologram-generation and capture calls here."
    )


def _save_experiment_result(result: ExperimentResult) -> None:
    if result.config.output_dir is None or result.simulation is None:
        return
    output_dir = Path(result.config.output_dir)
    save_result(result.simulation, output_dir)
    metadata_path = output_dir / "experimental_config.json"
    metadata = asdict(result.config)
    for key, value in metadata.items():
        if isinstance(value, Path):
            metadata[key] = str(value)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
