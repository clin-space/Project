"""Numerical routines ported from the MATLAB SuperPixel implementation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

Normalization = Literal["maxAmplitude", "highRes"]


def _matlab_coordinates(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return MATLAB-style (one-based) coordinate grids and aperture centers."""
    center_x = int(np.ceil((nx + 1) / 2))
    center_y = int(np.ceil((ny + 1) / 2))
    y, x = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1), indexing="ij")
    return y, x, center_y, center_x


def _centered_fft2(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def _centered_ifft2(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def _normalise(field: np.ndarray) -> np.ndarray:
    return field / np.sqrt(np.sum(np.abs(field) ** 2))


def _matlab_round_nonnegative(values: np.ndarray) -> np.ndarray:
    """MATLAB `round` for the non-negative lookup-table coordinates."""
    return np.floor(values + 0.5).astype(np.int64)


def fourier_mask(ny: int, nx: int, resolution: float, mask: str = "circle") -> np.ndarray:
    """Create the MATLAB `FourierMask` aperture."""
    y, x, center_y, center_x = _matlab_coordinates(ny, nx)
    dy = y - center_y
    dx = ny / nx * (x - center_x)
    radius = ny / resolution / 2

    if mask == "gaussian":
        return np.exp(-(dy**2 + dx**2) / radius**2)
    if mask == "square1":
        return (np.maximum(np.abs(dy), np.abs(dx)) < radius).astype(float)
    if mask == "square2":
        return ((np.abs(dy) + np.abs(dx)) < radius).astype(float)
    return ((dy**2 + dx**2) < radius**2).astype(float)


def inner_product(field_1: np.ndarray, field_2: np.ndarray) -> complex:
    """MATLAB `innerProduct`, including its conjugation convention."""
    norm_1 = np.sqrt(np.sum(np.abs(field_1) ** 2))
    norm_2 = np.sqrt(np.sum(np.abs(field_2) ** 2))
    return np.sum(field_1 * np.conj(field_2)) / (norm_1 * norm_2)


def spatial_filter(field: np.ndarray, resolution: float) -> np.ndarray:
    """Apply the circular system-resolution aperture to a complex field."""
    return _centered_ifft2(_centered_fft2(field) * fourier_mask(*field.shape, resolution, "circle"))


def rescale_target_to_superpixel_resolution(field: np.ndarray, superpixel_size: int = 4) -> np.ndarray:
    """Port of `rescaleTargetToSuperpixelResolution.m` (4x4 by default)."""
    ny, nx = field.shape
    if ny % superpixel_size or nx % superpixel_size:
        raise ValueError("field dimensions must be divisible by superpixel_size")

    y, x, center_y, center_x = _matlab_coordinates(ny, nx)
    aperture = (
        (y - center_y) ** 2 + (ny / nx * (x - center_x)) ** 2
        < (ny / superpixel_size / 2) ** 2
    )
    field_ft = _centered_fft2(field) * aperture
    out_ny, out_nx = ny // superpixel_size, nx // superpixel_size

    # MATLAB endpoints are inclusive and one-based.
    start_y = center_y - int(np.ceil((out_ny - 1) / 2))
    end_y = center_y + int(np.floor((out_ny - 1) / 2))
    start_x = center_x - int(np.ceil((out_nx - 1) / 2))
    end_x = center_x + int(np.floor((out_nx - 1) / 2))
    reduced_ft = field_ft[start_y - 1 : end_y, start_x - 1 : end_x]

    row, col = np.meshgrid(np.arange(1, out_ny + 1), np.arange(1, out_nx + 1), indexing="ij")
    phase_factor = np.exp(2j * np.pi * (row / out_ny + col / out_nx) * 3 / 8)
    return _centered_ifft2(reduced_ft * phase_factor)


def phase_and_amplitude_to_dmdpixels_lee(field: np.ndarray, kx: float, ky: float) -> np.ndarray:
    """Encode a field using the Lee method."""
    field = np.asarray(field, dtype=np.complex128)
    field = field / np.max(np.abs(field))
    ny, nx = field.shape
    row, col = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1), indexing="ij")
    carrier = row * ky + col * kx
    on = np.abs(np.mod(np.angle(field) / (2 * np.pi) + carrier, 1) - 0.5) <= (
        np.arcsin(np.abs(field)) / np.pi / 2
    )
    reference_beam = np.exp(-2j * np.pi * (carrier + 0.5))
    return on * reference_beam


def _default_data_root() -> Path:
    # SuperPixel_python/superpixel/core.py -> sibling SuperPixel_matlab assets.
    return Path(__file__).resolve().parents[2] / "SuperPixel_matlab"


@lru_cache(maxsize=None)
def _load_lookup_data(data_root: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    root = Path(data_root) / "superpixel_lookupTable"
    def complex_parser(value: bytes | str) -> complex:
        text = value.decode("ascii") if isinstance(value, bytes) else value
        return complex(text.replace("i", "j"))
    target_fields = np.loadtxt(
        root / "targetFields", delimiter=",", dtype=np.complex128, converters=complex_parser
    )
    lookup_table = np.loadtxt(root / "lookupTable", delimiter=",", dtype=np.int64)
    max_amplitude, step_size = np.loadtxt(root / "gridParameters", delimiter=",")
    return target_fields, lookup_table, float(max_amplitude), float(step_size)


def phase_and_amplitude_to_dmdpixels_lookup_table(
    field: np.ndarray,
    normalization: Normalization = "maxAmplitude",
    data_root: str | Path | None = None,
) -> np.ndarray:
    """Encode low-resolution pixels as 4x4 DMD superpixels using the supplied table."""
    if normalization not in ("maxAmplitude", "highRes"):
        raise ValueError("normalization must be 'maxAmplitude' or 'highRes'")
    root = Path(data_root) if data_root is not None else _default_data_root()
    target_fields, lookup_table, max_amplitude, step_size = _load_lookup_data(str(root.resolve()))
    field = np.asarray(field, dtype=np.complex128)
    ny, nx = field.shape
    m = 4

    # `exp(angle(E)*1i).*min(abs(E),1)` in the source implementation.
    encoded_field = np.exp(1j * np.angle(field)) * np.minimum(np.abs(field), 1)
    encoded_field *= np.exp(11j * np.pi / 16)
    encoded_field *= max_amplitude if normalization == "maxAmplitude" else 0.906131
    encoded_field /= step_size

    origin = (lookup_table.shape[0] + 1) / 2
    table_rows = _matlab_round_nonnegative(np.imag(encoded_field) + origin) - 1
    table_cols = _matlab_round_nonnegative(np.real(encoded_field) + origin) - 1
    if np.any(table_rows < 0) or np.any(table_rows >= lookup_table.shape[0]) or np.any(table_cols < 0) or np.any(table_cols >= lookup_table.shape[1]):
        raise ValueError("field maps outside the supplied lookup-table grid")

    # MATLAB table values are one-based rows.  Rotating the flattened vector
    # before the Fortran-order reshape preserves MATLAB's `reshape(pixels,4,4)`.
    pixels = target_fields[lookup_table[table_rows, table_cols] - 1, 1:].real
    shifts = (m * np.arange(ny)) % (m * m)
    indices = (np.arange(m * m)[None, :] + shifts[:, None]) % (m * m)
    pixels = np.take_along_axis(pixels, indices[:, None, :], axis=2)
    blocks = pixels.reshape(ny, nx, m, m).swapaxes(-1, -2)
    dmd_pixels = blocks.transpose(0, 2, 1, 3).reshape(ny * m, nx * m)

    row, col = np.meshgrid(np.arange(1, ny * m + 1), np.arange(1, nx * m + 1), indexing="ij")
    return dmd_pixels * np.exp(1j * np.pi * (row + 4 * col) / 8)


def superpixel_method(
    field_target: np.ndarray,
    resolution: float,
    mask: str = "circle",
    data_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run the 4x4 superpixel method and return pattern, field, fidelity, efficiency."""
    ny, nx = field_target.shape
    system_mask = fourier_mask(ny, nx, resolution, mask)
    reduced = rescale_target_to_superpixel_resolution(field_target)
    reduced /= np.max(np.abs(reduced))
    dmd_pixels = phase_and_amplitude_to_dmdpixels_lookup_table(reduced, "maxAmplitude", data_root)
    dmd_pattern = np.abs(dmd_pixels)
    obtained = _centered_ifft2(_centered_fft2(dmd_pixels) * system_mask)
    efficiency = float(np.sum(np.abs(obtained / np.sqrt(nx * ny)) ** 2))
    obtained = _normalise(obtained)
    target = _normalise(field_target)
    fidelity = float(np.abs(inner_product(target, obtained)) ** 2)
    return dmd_pattern, obtained, fidelity, efficiency


def lee_method(
    field_target: np.ndarray, resolution: float, mask: str, kx: float, ky: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run the Lee method and return pattern, field, fidelity, efficiency."""
    ny, nx = field_target.shape
    system_mask = fourier_mask(ny, nx, resolution, mask)
    dmd_pixels = phase_and_amplitude_to_dmdpixels_lee(field_target, kx, ky)
    dmd_pattern = np.abs(dmd_pixels)
    obtained = _centered_ifft2(_centered_fft2(dmd_pixels) * system_mask)
    efficiency = float(np.sum(np.abs(obtained / np.sqrt(nx * ny)) ** 2))
    obtained = _normalise(obtained)
    target = _normalise(field_target)
    fidelity = float(np.abs(inner_product(target, obtained)) ** 2)
    return dmd_pattern, obtained, fidelity, efficiency


def make_field_from_image(data_root: str | Path | None = None) -> tuple[np.ndarray, float]:
    """Build the supplied intensity/phase target field (requires Pillow)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("make_field_from_image requires Pillow") from exc
    root = Path(data_root) if data_root is not None else _default_data_root()
    intensity_rgb = np.asarray(Image.open(root / "images" / "pebbles_768.jpg"), dtype=float)
    intensity = np.mean(intensity_rgb[:, :, :3], axis=2) / 255
    intensity = np.minimum(1, intensity * 1.3)
    max_intensity = float(np.max(intensity) / np.sum(intensity))
    phase_rgb = np.asarray(Image.open(root / "images" / "barsik_768.jpg"), dtype=float)
    phase = phase_rgb[:, :, 0]
    field = np.sqrt(intensity) * np.exp(1j * (1.9 * np.pi * phase / np.max(phase) - 0.95 * np.pi))
    return field, max_intensity


def make_speckle_pattern(
    resolution: float, rng: np.random.Generator | None = None, shape: tuple[int, int] = (768, 1024)
) -> tuple[np.ndarray, float]:
    """Generate a band-limited complex speckle field (the MATLAB RNG stream differs)."""
    rng = np.random.default_rng() if rng is None else rng
    ny, nx = shape
    noise = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))
    return_field = _centered_ifft2(noise * fourier_mask(ny, nx, resolution, "circle"))
    intensity = np.abs(return_field) ** 2
    return return_field, float(np.max(intensity) / np.sum(intensity))
