"""Optional Python replacement for `createDMDsuperpixelLookupTable.m`.

Normally the checked-in lookup table should be reused.  Rebuilding it is
computationally expensive: an 855 by 855 grid is matched against 2^16 binary
patterns.  This module is provided for calibration experiments and custom
superpixel tables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _matlab_complex_text(value: complex) -> str:
    """Write the `dlmwrite`-style complex spelling consumed by the main port."""
    if value.imag == 0:
        return f"{value.real:.12g}"
    return f"{value.real:.12g}{value.imag:+.12g}i"


def create_dmd_superpixel_lookup_table(
    output_directory: str | Path,
    superpixel_size: int = 4,
    step_size: float = 0.003,
    chunk_size: int = 4_096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate `targetFields`, `lookupTable`, and `gridParameters` files.

    The default parameters match the MATLAB file.  Duplicate fields retain the
    binary pattern with the most enabled mirrors, as in the source code.
    """
    if superpixel_size != 4:
        raise ValueError("the supplied encoder is fixed to 4x4 superpixels")
    if step_size <= 0 or chunk_size <= 0:
        raise ValueError("step_size and chunk_size must be positive")

    n = superpixel_size
    m = n * n
    basepoints = np.exp(1j * np.arange(m) / m * 2 * np.pi)
    numbers = np.arange(1, 2**m + 1, dtype=np.uint32)
    pixels = ((numbers[:, None] >> np.arange(m, dtype=np.uint32)) & 1).astype(float)
    fields = np.sum(pixels * basepoints, axis=1)
    fields = (np.round(fields.real * 1e10) + 1j * np.round(fields.imag * 1e10)) / 1e10

    # `sortrows(targetFields, 1)` only establishes an order before duplicate
    # removal; lexicographic real/imag ordering is stable and deterministic.
    order = np.lexsort((fields.imag, fields.real))
    fields, pixels = fields[order], pixels[order]
    counts = pixels.sum(axis=1)
    distinct = np.r_[True, fields[1:] != fields[:-1]]
    group_starts = np.flatnonzero(distinct)
    group_ends = np.r_[group_starts[1:], len(fields)]
    selected = np.array([start + np.argmax(counts[start:end]) for start, end in zip(group_starts, group_ends)])
    target_fields = np.column_stack((fields[selected] / n, pixels[selected]))

    max_amplitude = float(np.max(np.abs(target_fields[:, 0])))
    linear_grid = np.r_[-np.arange(step_size, max_amplitude + step_size / 2, step_size)[::-1], np.arange(0, max_amplitude + step_size / 2, step_size)]
    grid = linear_grid[None, :] + 1j * linear_grid[:, None]
    flattened_grid = grid.ravel()
    nearest = np.empty(flattened_grid.size, dtype=np.int64)
    candidates = target_fields[:, 0]
    for begin in range(0, flattened_grid.size, chunk_size):
        end = min(begin + chunk_size, flattened_grid.size)
        nearest[begin:end] = np.abs(flattened_grid[begin:end, None] - candidates).argmin(axis=1) + 1
    lookup_table = nearest.reshape(grid.shape)
    grid_parameters = np.array([max_amplitude, step_size])

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "targetFields").open("w", encoding="ascii") as file:
        for row in target_fields:
            file.write(",".join(_matlab_complex_text(value) for value in row) + "\n")
    np.savetxt(output / "lookupTable", lookup_table, fmt="%d", delimiter=",")
    np.savetxt(output / "gridParameters", grid_parameters[None, :], fmt="%.12g", delimiter=",")
    return target_fields, lookup_table, grid_parameters
