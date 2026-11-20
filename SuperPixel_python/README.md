# SuperPixel Python port

This is a NumPy port of the MATLAB implementation in the sibling
`SuperPixel_matlab/` directory. It reads the existing
`SuperPixel_matlab/superpixel_lookupTable/` files and images directly;
the lookup table is not regenerated. If the MATLAB directory is moved,
pass its path as `data_root` to the functions that load assets.

## Requirements

Python 3.10+ with `numpy`.  `Pillow` is additionally needed for
`make_field_from_image`, and `matplotlib` is only needed for the plotting
helpers.

## Run the example

```bash
cd /home/chenlin/Project/Project/SuperPixel_python
python example.py
```

The example runs without a graphical desktop and saves five PNG images to
`result/` beside `example.py`: the target field, both DMD patterns, and both
obtained fields. Fidelity and efficiency are also printed to the terminal.

The public functions retain the MATLAB routine names in snake_case:
`superpixel_method`, `lee_method`, `fourier_mask`, `spatial_filter`, and
`phase_and_amplitude_to_dmdpixels_lookup_table`.

The optional `superpixel.create_lookup_table` module is only for future
calibration experiments. The example and default functions never call it.

## Compatibility notes

The port preserves MATLAB's one-based coordinate formulas, FFT shift/order,
column-major 4x4 block reshape, and half-away-from-zero rounding for lookup
indices.  Therefore it is designed to give numerically equivalent results for
the supplied lookup tables and deterministic input arrays.  Random speckle
patterns are statistically equivalent but cannot be bit-identical to MATLAB
because MATLAB and NumPy use different random-number generators.
