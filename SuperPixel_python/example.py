"""Python equivalent of example.m; save plots to the local result directory."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Render PNG files on servers without a display.

from superpixel import inner_product, lee_method, make_field_from_image, spatial_filter, superpixel_method
from superpixel.plotting import plot_dmdpixels, plot_e


def main() -> None:
    resolution = 8
    mask = "circle"
    target, max_intensity = make_field_from_image()
    dmd_superpixel, field_superpixel, fidelity_superpixel, efficiency_superpixel = superpixel_method(
        target, resolution, mask
    )
    dmd_lee, field_lee, fidelity_lee, efficiency_lee = lee_method(target, resolution, mask, 1 / 12, 1 / 12)
    target_lowres = spatial_filter(target, resolution)
    print(f"superpixel: fidelity={fidelity_superpixel:.12g}, efficiency={efficiency_superpixel:.12g}")
    print(f"Lee:        fidelity={fidelity_lee:.12g}, efficiency={efficiency_lee:.12g}")
    print(f"superpixel low-resolution fidelity={abs(inner_product(field_superpixel, target_lowres)) ** 2:.12g}")
    print(f"Lee low-resolution fidelity={abs(inner_product(field_lee, target_lowres)) ** 2:.12g}")
    result_dir = Path(__file__).resolve().parent / "result"
    result_dir.mkdir(exist_ok=True)
    figures = {
        "target_field.png": plot_e(target / ((abs(target) ** 2).sum() ** 0.5), max_intensity, "gray", "target"),
        "superpixel_dmd.png": plot_dmdpixels(dmd_superpixel, "superpixel"),
        "superpixel_field.png": plot_e(field_superpixel, max_intensity, "gray", "superpixel"),
        "lee_dmd.png": plot_dmdpixels(dmd_lee, "Lee"),
        "lee_field.png": plot_e(field_lee, max_intensity, "gray", "Lee"),
    }
    import matplotlib.pyplot as plt

    for filename, figure in figures.items():
        figure.savefig(result_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
    print(f"Saved {len(figures)} images to {result_dir}")


if __name__ == "__main__":
    main()
