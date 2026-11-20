"""Optional Matplotlib visualisation helpers corresponding to the MATLAB plots."""

from __future__ import annotations

import numpy as np


def plot_dmdpixels(dmd_pixels: np.ndarray, name: str):
    import matplotlib.pyplot as plt

    ny, nx = dmd_pixels.shape
    figure, axis = plt.subplots()
    image = axis.imshow(np.abs(dmd_pixels), cmap="gray_r")
    axis.set(xticks=[0, nx / 2 - 1, nx - 1], yticks=[0, ny / 2 - 1, ny - 1], title=f"DMD pattern {name}")
    figure.colorbar(image, ax=axis, ticks=[0, 1])
    axis.set_aspect("equal")
    return figure


def plot_e(field: np.ndarray, max_intensity: float, phase_colormap: str, name: str):
    import matplotlib.pyplot as plt

    ny, nx = field.shape
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    intensity = axes[0].imshow(np.abs(field) ** 2, cmap="gray", vmin=0, vmax=max_intensity)
    axes[0].set(title=f"{name} intensity", xticks=[0, nx / 2 - 1, nx - 1], yticks=[0, ny / 2 - 1, ny - 1])
    figure.colorbar(intensity, ax=axes[0])
    phase = axes[1].imshow(np.angle(field), cmap=phase_colormap, vmin=-np.pi, vmax=np.pi)
    axes[1].set(title=f"{name} phase", xticks=[0, nx / 2 - 1, nx - 1], yticks=[0, ny / 2 - 1, ny - 1])
    figure.colorbar(phase, ax=axes[1])
    for axis in axes:
        axis.set_aspect("equal")
    return figure
