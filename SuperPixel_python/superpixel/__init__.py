"""NumPy implementation of the SuperPixel MATLAB project."""

from .core import (
    fourier_mask,
    inner_product,
    lee_method,
    make_field_from_image,
    make_speckle_pattern,
    phase_and_amplitude_to_dmdpixels_lee,
    phase_and_amplitude_to_dmdpixels_lookup_table,
    rescale_target_to_superpixel_resolution,
    spatial_filter,
    superpixel_method,
)

__all__ = [
    "fourier_mask",
    "inner_product",
    "lee_method",
    "make_field_from_image",
    "make_speckle_pattern",
    "phase_and_amplitude_to_dmdpixels_lee",
    "phase_and_amplitude_to_dmdpixels_lookup_table",
    "rescale_target_to_superpixel_resolution",
    "spatial_filter",
    "superpixel_method",
]
